"""
Central configuration for the data pipeline: settlement geometries, date
ranges, and preprocessing constants.

IMPORTANT: The fallback bounding boxes below are approximate centroids
pulled from open reference sources, buffered by ~1.5km, NOT authoritative
settlement boundaries. For anything beyond MVP prototyping, replace
SETTLEMENTS[*]['fallback_bbox'] with a digitized boundary (e.g. from
Map Kibera, KISIP II shapefiles, or the NCC planning department) and feed
it in as `custom_boundary_path` in DataPipeline. The pipeline defaults to
resolving boundaries live via OSM (Nominatim) through osmnx, which is more
reliable than any hardcoded box but still depends on OSM's admin polygon
coverage being complete for these areas -- verify visually before training.
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SettlementConfig:
    name: str
    osm_query: str                 # osmnx.geocode_to_gdf query string
    fallback_bbox: tuple            # (min_lon, min_lat, max_lon, max_lat), WGS84
    centroid: tuple                 # (lat, lon), for sanity-checking / map centering


SETTLEMENTS = {
    "kibera": SettlementConfig(
        name="Kibera",
        osm_query="Kibera, Nairobi, Kenya",
        fallback_bbox=(36.7680, -1.3230, 36.7960, -1.3010),
        centroid=(-1.3133, 36.7820),
    ),
    "mathare": SettlementConfig(
        name="Mathare",
        osm_query="Mathare, Nairobi, Kenya",
        fallback_bbox=(36.8470, -1.2680, 36.8680, -1.2500),
        centroid=(-1.2585, 36.8571),
    ),
    "mukuru": SettlementConfig(
        name="Mukuru",
        osm_query="Mukuru, Nairobi, Kenya",
        fallback_bbox=(36.8600, -1.3280, 36.8870, -1.3050),
        centroid=(-1.3183, 36.8725),
    ),
}

# --- Google Earth Engine ---
# Optional default GCP project for Earth Engine. Lets headless runs (CI /
# Cloud Agents) avoid hardcoding a project: set GEE_PROJECT_ID in the
# environment, or rely on the service-account key's own project_id. An
# explicit gee_project_id passed to DataPipeline still overrides this.
GEE_PROJECT_ID = os.environ.get("GEE_PROJECT_ID")

# --- CRS ---
# WGS84 for GEE / OSM I/O; UTM 37S for anything requiring metric distances
# (proximity graph edges, TWI slope calc, tile sizing in metres).
CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:32737"  # WGS 84 / UTM zone 37S -- correct zone for Nairobi

# --- Sentinel-2 optical composite ---
S2_START_DATE = "2023-06-01"
S2_END_DATE = "2025-06-01"       # 2-year dry-season window per proposal Sec 3.2.1
S2_DRY_SEASON_MONTHS = [1, 2, 6, 7, 8, 9]  # Jan-Feb & Jun-Sep are Nairobi's dry months
S2_MAX_CLOUD_PCT = 20
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2

# --- Sentinel-1 SAR (flood inundation labels) ---
S1_START_DATE = "2019-01-01"
S1_END_DATE = "2024-12-31"
S1_RAINY_SEASON_MONTHS = [3, 4, 5, 10, 11]  # long + short rains
S1_BAND = "VV"
S1_INSTRUMENT_MODE = "IW"

# --- SRTM DEM / TWI ---
SRTM_DATASET = "USGS/SRTMGL1_003"
SRTM_RESOLUTION_M = 30

# --- OSM / graph construction ---
BUILDING_ADJACENCY_THRESHOLD_M = 15.0  # per proposal Sec 3.4 unit testing note
MIN_COMPONENT_SIZE = 20  # drop connected components smaller than this (see prune_small_components)
MIN_COMPONENT_SIZE = 20  # drop connected components smaller than this (see prune_small_components)
MIN_COMPONENT_SIZE = 20  # drop connected components smaller than this (see prune_small_components)
MIN_COMPONENT_SIZE = 20  # drop connected components smaller than this (see prune_small_components)
MIN_COMPONENT_SIZE = 20  # drop connected components smaller than this (see prune_small_components)
MIN_COMPONENT_SIZE = 20  # drop connected components smaller than this (see prune_small_components)
OVERPASS_TAGS = {"building": True}
INFRA_TAGS = {
    "amenity": ["drinking_water", "toilets"],
    "man_made": ["water_tap", "water_well"],
    "highway": True,
}

# --- Raster tiling ---
TILE_SIZE_PX = 256
TILE_OVERLAP_PX = 50
MIN_VALID_DATA_FRACTION = 0.6  # discard tiles with less valid data than this

# --- Output paths (relative to repo root; mount Drive at this path in Colab) ---
DATA_ROOT = "data"
RAW_DIR = f"{DATA_ROOT}/raw"
STACKS_DIR = f"{DATA_ROOT}/raster_stacks"
TILES_DIR = f"{DATA_ROOT}/tiles"
GRAPHS_DIR = f"{DATA_ROOT}/graphs"
