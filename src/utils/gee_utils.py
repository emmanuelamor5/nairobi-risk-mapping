"""
Google Earth Engine data acquisition helpers.

All functions take an ee.Geometry and return an ee.Image (single band or
multi-band). Export to GeoTIFF is handled separately in data_pipeline.py
via ee.batch.Export, since large exports need to run asynchronously against
Drive rather than blocking the Colab runtime.
"""

import ee

from .. import config


def initialize_gee(project_id: str) -> None:
    """
    Authenticate and initialise the Earth Engine API.

    In Colab, run this once per session. `project_id` must be a GCP project
    with the Earth Engine API enabled (Cloud Project associated with your
    Google account -- required since GEE moved off the legacy noproject
    auth flow).
    """
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
    VV backscatter minimum composite. Water surfaces have characteristically
    low VV backscatter (specular reflection), so thresholding the per-pixel
    temporal minimum over the rainy-season window approximates the maximum
    historical flood extent -- this becomes the Attention U-Net's training
    label, per proposal Sec 3.2.1.

    Otsu's method is implemented via the standard GEE histogram + cumulative
    moments recipe (there's no built-in ee.Otsu).
    """
    vv_min = get_sentinel1_backscatter(geometry).min().clip(geometry)

    histogram = vv_min.reduceRegion(
        reducer=ee.Reducer.histogram(255, 1),
        geometry=geometry,
        scale=10,
        maxPixels=1e9,
        bestEffort=True,
    ).get(config.S1_BAND)

    def _otsu(histogram_dict):
        counts = ee.Array(ee.Dictionary(histogram_dict).get("histogram"))
        means = ee.Array(ee.Dictionary(histogram_dict).get("bucketMeans"))
        size = means.length().get([0])
        total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
        s_sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
        indices = ee.List.sequence(1, size)

        def _bss(i):
            a_counts = counts.slice(0, 0, i)
            a_count = a_counts.reduce(ee.Reducer.sum(), [0]).get([0])
            a_means = means.slice(0, 0, i)
            a_mean = (
                a_means.multiply(a_counts).reduce(ee.Reducer.sum(), [0]).get([0])
                .divide(a_count)
            )
            b_count = total.subtract(a_count)
            b_mean = s_sum.subtract(a_count.multiply(a_mean)).divide(b_count)
            return a_count.multiply(b_count).multiply(a_mean.subtract(b_mean).pow(2))

        bss = indices.map(_bss)
        return means.sort(bss).get([-1])

    threshold = ee.Number(_otsu(histogram))
    flood_mask = vv_min.lt(threshold).rename("flood_label")
    return flood_mask.selfMask()


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

    # Flow accumulation proxy: inverse of local elevation relative to a
    # smoothed neighbourhood -- a coarse stand-in, not a true D8/D-infinity
    # flow routing. Flag this clearly for anyone refining the pipeline.
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
    # Cast every band to Float32 -- GEE export refuses mixed dtypes,
    # and the flood label comes out as Byte from the .lt() comparison
    # while the optical bands are already Float32.
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
