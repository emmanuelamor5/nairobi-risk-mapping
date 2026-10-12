"""
Google Earth Engine data acquisition helpers.

All functions take an ee.Geometry and return an ee.Image (single band or
multi-band). Export to GeoTIFF is handled separately in data_pipeline.py
via ee.batch.Export, since large exports need to run asynchronously against
Drive rather than blocking the Colab runtime.
"""

import json
import logging
import os
import tempfile

import ee
import numpy as np

from .. import config

logger = logging.getLogger(__name__)


def _resolve_service_account_key() -> str | None:
    """
    Return a filesystem path to a service-account JSON key if one is supplied
    via the environment, else None.

    Accepts either the JSON key *contents* or a path in
    EARTHENGINE_SERVICE_ACCOUNT_KEY, or a path in GOOGLE_APPLICATION_CREDENTIALS.
    Inline JSON is materialised to a private (0600) temp file because
    ee.ServiceAccountCredentials needs a file path, not raw JSON. Injecting
    the key as a secret env var keeps it out of the repo, logs, and snapshots.
    """
    raw = os.environ.get("EARTHENGINE_SERVICE_ACCOUNT_KEY")
    if raw:
        raw = raw.strip()
        if raw.startswith("{"):
            fd, path = tempfile.mkstemp(prefix="ee-sa-", suffix=".json")
            with os.fdopen(fd, "w") as f:
                f.write(raw)
            os.chmod(path, 0o600)
            return path
        if os.path.exists(raw):
            return raw
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and os.path.exists(gac):
        return gac
    return None


def initialize_gee(project_id: str = None) -> None:
    """
    Authenticate and initialise the Earth Engine API.

    Resolution order:
      1. Service-account credentials from EARTHENGINE_SERVICE_ACCOUNT_KEY (the
         JSON key contents or a path to it) or GOOGLE_APPLICATION_CREDENTIALS
         (a key-file path). This is the non-interactive path for CI, headless
         servers, and Cloud Agents. When `project_id` is omitted it defaults to
         the key's own `project_id`.
      2. Pre-existing user credentials (plain ee.Initialize) -- e.g. a prior
         `earthengine authenticate` on a workstation.
      3. Interactive ee.Authenticate() as a last resort (local / Colab only).

    `project_id` (or the key's project) must be a GCP project with the Earth
    Engine API enabled -- required since GEE moved off the legacy noproject
    auth flow.
    """
    key_path = _resolve_service_account_key()
    if key_path:
        with open(key_path) as f:
            info = json.load(f)
        project_id = project_id or config.GEE_PROJECT_ID or info.get("project_id")
        logger.info(
            "Initialising Earth Engine with service account %s (project %s)",
            info.get("client_email"), project_id,
        )
        credentials = ee.ServiceAccountCredentials(info["client_email"], key_path)
        ee.Initialize(credentials, project=project_id)
        return

    project_id = project_id or config.GEE_PROJECT_ID
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def mask_s2_clouds(image: "ee.Image") -> "ee.Image":
    """Mask clouds/cirrus in a Sentinel-2 SR image using the QA60 band."""
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask).divide(10000)  # scale reflectance to [0,1]


def get_sentinel2_composite(geometry: "ee.Geometry") -> "ee.Image":
    """
    Dry-season median composite over config.S2_START_DATE..S2_END_DATE,
    restricted to config.S2_DRY_SEASON_MONTHS and filtered by cloud cover.
    Returns an image with bands config.S2_BANDS, already normalised to [0,1].
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(config.S2_START_DATE, config.S2_END_DATE)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", config.S2_MAX_CLOUD_PCT))
    )
    # Dry-season month filter needs a derived "month" property first --
    # ee.Filter.inList can't inspect the date directly.
    def _add_month(img):
        month = ee.Date(img.get("system:time_start")).get("month")
        return img.set("month", month)

    collection = collection.map(_add_month).filter(
        ee.Filter.inList("month", config.S2_DRY_SEASON_MONTHS)
    )

    composite = collection.map(mask_s2_clouds).select(config.S2_BANDS).median()
    return composite.clip(geometry)


def get_sentinel1_backscatter(geometry: "ee.Geometry") -> "ee.ImageCollection":
    """Raw VV backscatter time series over the rainy-season study window."""
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(config.S1_START_DATE, config.S1_END_DATE)
        .filter(ee.Filter.eq("instrumentMode", config.S1_INSTRUMENT_MODE))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", config.S1_BAND))
        .select(config.S1_BAND)
    )

    def _add_month(img):
        month = ee.Date(img.get("system:time_start")).get("month")
        return img.set("month", month)

    return collection.map(_add_month).filter(
        ee.Filter.inList("month", config.S1_RAINY_SEASON_MONTHS)
    )


def get_sentinel1_flood_mask(geometry: "ee.Geometry") -> "ee.Image":
    """
    Historical inundation footprint via Otsu thresholding on the rainy-season
    VV backscatter minimum composite. The histogram is pulled to Python with
    .getInfo() and thresholded with plain numpy, rather than reimplemented in
    Earth Engine's Array API -- the GEE-native version proved fragile across
    several rounds of debugging (an off-by-one corrupted the sort, and
    guarding the division still left the threshold degenerate). This version
    is a standard, easily-verified numpy Otsu; the threshold is logged so you
    can sanity-check it against real dB values (expect roughly -25 to -5 dB
    for VV backscatter -- a threshold far outside that range means the
    histogram itself looks wrong, not the thresholding math).

    Deliberately NOT .selfMask()'d -- this needs to stay a dense 0/1 band for
    training, not a sparse "flooded pixels only" mask. selfMask() previously
    turned every non-flooded pixel into nodata, which tanked the tiling
    valid-data fraction to ~43% on real exports.
    """
    vv_min = get_sentinel1_backscatter(geometry).min().clip(geometry)

    histogram_dict = vv_min.reduceRegion(
        reducer=ee.Reducer.histogram(255, 1),
        geometry=geometry,
        scale=10,
        maxPixels=1e9,
        bestEffort=True,
    ).get(config.S1_BAND).getInfo()

    counts = np.array(histogram_dict["histogram"], dtype=np.float64)
    means = np.array(histogram_dict["bucketMeans"], dtype=np.float64)
    total = counts.sum()
    sum_all = (counts * means).sum()

    best_bss, best_threshold = -1.0, float(means[0])
    cum_count, cum_sum = 0.0, 0.0
    for i in range(len(counts) - 1):  # always leave >=1 bucket in the "above" class
        cum_count += counts[i]
        cum_sum += counts[i] * means[i]
        if cum_count == 0 or (total - cum_count) == 0:
            continue
        a_mean = cum_sum / cum_count
        b_mean = (sum_all - cum_sum) / (total - cum_count)
        bss = cum_count * (total - cum_count) * (a_mean - b_mean) ** 2
        if bss > best_bss:
            best_bss = bss
            best_threshold = float(means[i])

    logger.info("Sentinel-1 Otsu threshold for this geometry: %.3f dB", best_threshold)
    return vv_min.lt(ee.Number(best_threshold)).rename("flood_label")


def get_sentinel1_vv_composite(geometry: "ee.Geometry", start_date: str,
                               end_date: str, reducer: str = "median") -> "ee.Image":
    """
    Single-band VV backscatter composite (in dB) over an arbitrary date
    window -- the building block for event-based change detection. ``reducer``
    is "median" (stable baseline) or "min" (captures the darkest pass, i.e.
    the flood peak, even when inundation shows up in only one acquisition of
    a short event window). Speckle is reduced with a 30 m focal-mean filter,
    matching the smoothing applied before thresholding in Sprint 1.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", config.S1_INSTRUMENT_MODE))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", config.S1_BAND))
        .select(config.S1_BAND)
    )
    composite = collection.min() if reducer == "min" else collection.median()
    # Speckle filtering: a light focal mean in metres, resolution-independent.
    composite = composite.focal_mean(radius=30, units="meters")
    return composite.clip(geometry).rename("vv")


def get_permanent_water_mask(geometry: "ee.Geometry") -> "ee.Image":
    """
    Permanent open-water mask from the JRC Global Surface Water occurrence
    layer: 1 where water is present at least
    config.FLOOD_PERMANENT_WATER_OCCURRENCE_PCT% of the time over the 1984-
    present record, else 0. Used to exclude the baseline river channel so the
    flood label captures NEW inundation rather than permanent water.
    """
    occurrence = ee.Image(config.JRC_GSW_DATASET).select("occurrence").clip(geometry)
    permanent = occurrence.gte(config.FLOOD_PERMANENT_WATER_OCCURRENCE_PCT).unmask(0)
    return permanent.rename("permanent_water")


def _event_change_threshold_db(change_image: "ee.Image", geometry: "ee.Geometry") -> float:
    """
    Otsu threshold on the negative tail of the event-minus-pre change image,
    computed in numpy (same code path the offline/notebook label uses) so the
    GEE export and the recompute-from-raw-bands path agree exactly. Falls back
    to config.FLOOD_CHANGE_THRESHOLD_DB when the histogram is degenerate.
    """
    from . import flood_label  # local import keeps numpy/scipy optional for GEE-only callers

    hist = change_image.reduceRegion(
        reducer=ee.Reducer.histogram(256, 0.25),
        geometry=geometry,
        scale=10,
        maxPixels=1e9,
        bestEffort=True,
    ).get("vv").getInfo()

    if not hist:
        return float(config.FLOOD_CHANGE_THRESHOLD_DB)
    counts = np.array(hist["histogram"], dtype=np.float64)
    means = np.array(hist["bucketMeans"], dtype=np.float64)
    # Reconstruct a sample-free Otsu on the drop side by weighting bucket means
    # by their counts, restricted to change < 0.
    drop = means < 0
    if drop.sum() < 2 or counts[drop].sum() < 2:
        return float(config.FLOOD_CHANGE_THRESHOLD_DB)
    expanded = np.repeat(means[drop], counts[drop].astype(int))
    if expanded.size < 2 or np.unique(expanded).size < 2:
        return float(config.FLOOD_CHANGE_THRESHOLD_DB)
    otsu = flood_label.otsu_threshold(expanded)
    if np.isfinite(otsu) and otsu < 0:
        return float(otsu)
    return float(config.FLOOD_CHANGE_THRESHOLD_DB)


def get_event_flood_label(geometry: "ee.Geometry", use_otsu: bool = True) -> "ee.Image":
    """
    Event-based flood label via bi-temporal SAR change detection -- the
    Sprint 2 v2 replacement for get_sentinel1_flood_mask().

    Open water is specular at C-band, so new inundation appears as a sharp VV
    backscatter drop between a pre-event median composite and an event-peak
    minimum composite. Thresholding that drop (Otsu on the negative tail, or
    a fixed dB cut) yields candidate water; three physical refinements then
    remove the usual SAR false positives:

      1. slope > config.FLOOD_MAX_SLOPE_DEG   (water does not pool on slopes)
      2. JRC permanent water                  (keep NEW inundation only)
      3. (small-object speckle removal is applied in the numpy path; omitted
         server-side where connected-component labelling is awkward -- the
         exported label is still dense 0/1 and the notebook can re-run the
         speckle filter from raw bands if needed.)

    Mirrors src/utils/flood_label.build_event_flood_label() so the one-shot
    GEE export and the transparent numpy recompute produce the same label.
    Returned as a dense 0/1 "flood_label" band (NOT selfMask'd), matching the
    training stack contract.
    """
    pre = get_sentinel1_vv_composite(
        geometry, config.FLOOD_EVENT_PRE_START, config.FLOOD_EVENT_PRE_END, reducer="median"
    )
    event = get_sentinel1_vv_composite(
        geometry, config.FLOOD_EVENT_START, config.FLOOD_EVENT_END, reducer="min"
    )
    change = event.subtract(pre).rename("vv")

    threshold = (
        _event_change_threshold_db(change, geometry)
        if use_otsu else float(config.FLOOD_CHANGE_THRESHOLD_DB)
    )
    logger.info(
        "Event flood label: change threshold %.3f dB (event=%s..%s vs pre=%s..%s)",
        threshold, config.FLOOD_EVENT_START, config.FLOOD_EVENT_END,
        config.FLOOD_EVENT_PRE_START, config.FLOOD_EVENT_PRE_END,
    )

    candidate = change.lte(ee.Number(threshold))

    slope = ee.Terrain.slope(ee.Image(config.SRTM_DATASET).clip(geometry))
    gentle = slope.lte(config.FLOOD_MAX_SLOPE_DEG)
    not_permanent = get_permanent_water_mask(geometry).Not()

    label = candidate.And(gentle).And(not_permanent).unmask(0)
    return label.rename("flood_label").toFloat()


def get_srtm_twi(geometry: "ee.Geometry") -> "ee.Image":
    """
    Terrain derivatives from SRTM: elevation, slope, and Topographic Wetness
    Index (TWI = ln(flow_accumulation / tan(slope))). Flow accumulation is
    approximated via a downslope-weighted flow algorithm on the DEM since GEE
    has no native flow-accumulation reducer comparable to a GIS hydrology
    toolbox -- for production-grade TWI, consider precomputing flow
    accumulation in a proper hydrology package (e.g. WhiteboxTools, pysheds)
    on the exported DEM and re-importing rather than relying on this
    approximation.
    """
    dem = ee.Image(config.SRTM_DATASET).clip(geometry)
    slope = ee.Terrain.slope(dem)

    smoothed = dem.focal_mean(radius=90, units="meters")
    flow_proxy = smoothed.subtract(dem).max(0.01).rename("flow_accum_proxy")

    slope_rad = slope.multiply(3.14159265).divide(180)
    twi = (
        flow_proxy.divide(slope_rad.tan().max(0.001))
        .log()
        .rename("twi")
    )

    return dem.rename("elevation").addBands(slope.rename("slope")).addBands(twi)


def build_raster_stack(geometry: "ee.Geometry", label_mode: str = "event",
                       include_s1_inputs: bool = False) -> "ee.Image":
    """
    Assemble the training stack: S2 optical + (optional S1 inputs) + flood
    label + terrain, in the band order dataset.py's layouts expect.

    ``label_mode``:
      - "event" (default): bi-temporal event change-detection label
        (get_event_flood_label) -- the Sprint 2 v2 fix for the land-cover
        confound. This is now the default because the seasonal label was
        shown to be an unvalidated land-cover proxy.
      - "seasonal": the original dry-vs-rainy Otsu label
        (get_sentinel1_flood_mask), kept for side-by-side comparison in the
        notebook and reproducibility of the v1 result.

    ``include_s1_inputs``:
      - False (default): 10 bands, dataset.DEFAULT_LAYOUT
        ([B2,B3,B4,B8,B11,B12, flood_label, elevation, slope, twi]).
      - True: inserts the pre-event and event VV composites as input bands,
        giving 12 bands in dataset.EVENT_S1_LAYOUT order
        ([B2,B3,B4,B8,B11,B12, pre_vv, event_vv, flood_label, elevation,
        slope, twi]). This lets the U-Net see the SAR modality its label is
        derived from instead of inferring water from optical land cover alone.
    """
    s2 = get_sentinel2_composite(geometry)
    if label_mode == "event":
        s1_label = get_event_flood_label(geometry)
    elif label_mode == "seasonal":
        s1_label = get_sentinel1_flood_mask(geometry)
    else:
        raise ValueError(f"Unknown label_mode '{label_mode}'. Use 'event' or 'seasonal'.")
    terrain = get_srtm_twi(geometry)

    stack = s2
    if include_s1_inputs:
        pre = get_sentinel1_vv_composite(
            geometry, config.FLOOD_EVENT_PRE_START, config.FLOOD_EVENT_PRE_END, reducer="median"
        ).rename("pre_vv")
        event = get_sentinel1_vv_composite(
            geometry, config.FLOOD_EVENT_START, config.FLOOD_EVENT_END, reducer="min"
        ).rename("event_vv")
        stack = stack.addBands(pre).addBands(event)
    # Cast every band to Float32 -- GEE export refuses mixed dtypes, and the
    # flood label's comparison bands produce Byte while the optical bands are
    # already Float32.
    return stack.addBands(s1_label).addBands(terrain).toFloat()


def build_label_ingredients_stack(geometry: "ee.Geometry") -> "ee.Image":
    """
    Export the raw ingredients of the event flood label as separate bands, so
    the label can be (re)built and tuned in numpy
    (src/utils/flood_label.build_event_flood_label) without re-running an
    Earth Engine export for every threshold change.

    Band order: [B2,B3,B4,B8,B11,B12, pre_vv, event_vv, permanent_water,
    elevation, slope, twi]. The notebook computes flood_label from pre_vv,
    event_vv, slope and permanent_water, then reassembles the 10-band training
    stack in the canonical order. This is the transparent, debuggable path the
    Sprint 1 write-up argued for after the GEE-native Otsu proved fragile.
    """
    s2 = get_sentinel2_composite(geometry)
    pre = get_sentinel1_vv_composite(
        geometry, config.FLOOD_EVENT_PRE_START, config.FLOOD_EVENT_PRE_END, reducer="median"
    ).rename("pre_vv")
    event = get_sentinel1_vv_composite(
        geometry, config.FLOOD_EVENT_START, config.FLOOD_EVENT_END, reducer="min"
    ).rename("event_vv")
    permanent = get_permanent_water_mask(geometry)
    terrain = get_srtm_twi(geometry)
    return (
        s2.addBands(pre).addBands(event).addBands(permanent).addBands(terrain).toFloat()
    )


def export_stack_to_drive(image: "ee.Image", geometry: "ee.Geometry",
                           description: str, folder: str = "nairobi_risk_mapping",
                           scale: int = 10) -> "ee.batch.Task":
    """Kick off an async export to Google Drive. Poll task.status() to check completion."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=geometry,
        scale=scale,
        crs=config.CRS_WGS84,
        maxPixels=1e10,
    )
    task.start()
    return task
