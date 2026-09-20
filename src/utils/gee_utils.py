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


def build_raster_stack(geometry: "ee.Geometry") -> "ee.Image":
    """Assemble the full multi-band raster stack: S2 optical + S1 flood label + terrain."""
    s2 = get_sentinel2_composite(geometry)
    s1_label = get_sentinel1_flood_mask(geometry)
    terrain = get_srtm_twi(geometry)
    # Cast every band to Float32 -- GEE export refuses mixed dtypes, and the
    # flood label's .lt() comparison produces Byte while the optical bands
    # are already Float32.
    return s2.addBands(s1_label).addBands(terrain).toFloat()


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
