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

# --- Event-based flood label (Sprint 2 v2) ------------------------------
# Sprint 2 v1 derived the label from a dry-vs-rainy-season Otsu threshold on
# the VV *minimum* composite. Diagnostics (notebook 02 v1) showed that label
# tracks seasonal land-cover / surface-type change more than inundation: the
# signal concentrated in SWIR/NIR optical bands, flooded fractions of 19-32%
# were implausibly high, and the U-Net never beat a "flood everywhere"
# baseline. This block replaces that with bi-temporal change detection around
# a single dated flood event -- the UN-SPIDER recommended practice for
# SAR flood mapping -- which cancels out anything that is stable between the
# two dates (permanent land cover) and keeps only the new-water signal.
#
# Dated event: the April-May 2024 East African floods, which inundated the
# Nairobi/Mathare/Ngong river corridors through Mathare, Mukuru and parts of
# Kibera (Nairobi's most severe recent flooding). The pre-event window is the
# preceding dry spell; the event window brackets the inundation peak. Adjust
# these if you anchor the label to a different documented event.
FLOOD_EVENT_NAME = "2024-04 Nairobi floods"
FLOOD_EVENT_PRE_START = "2024-02-01"   # dry baseline, ends before the rains
FLOOD_EVENT_PRE_END = "2024-03-20"
FLOOD_EVENT_START = "2024-04-20"       # brackets the inundation peak
FLOOD_EVENT_END = "2024-05-10"

# Change-detection decision rule. Open water is specular at C-band, so new
# inundation shows up as a sharp DROP in VV backscatter between the pre-event
# and event composites. A pixel is flagged flooded when that drop exceeds
# FLOOD_CHANGE_THRESHOLD_DB (negative = a decrease). The notebook/GEE path
# can instead derive the threshold from the change histogram via Otsu and
# fall back to this constant when the histogram is degenerate.
FLOOD_CHANGE_THRESHOLD_DB = -3.0

# Physical refinement masks applied after thresholding (UN-SPIDER practice):
FLOOD_MAX_SLOPE_DEG = 5.0              # water does not pool on steep terrain
FLOOD_PERMANENT_WATER_OCCURRENCE_PCT = 50  # JRC GSW occurrence >= this = permanent water, excluded
FLOOD_MIN_OBJECT_PIXELS = 8           # drop speckle: connected flood blobs smaller than this
JRC_GSW_DATASET = "JRC/GSW1_4/GlobalSurfaceWater"

# Fixed physical range (dB) for normalising Sentinel-1 VV bands when they are
# fed to the U-Net as inputs (EVENT_S1_LAYOUT). Per-tile percentile scaling --
# fine for optical -- would rescale pre_vv and event_vv independently and
# destroy the cross-date backscatter DROP that signals water. A shared, fixed
# dB window keeps pre and event comparable so the drop survives normalisation.
S1_DB_RANGE = (-30.0, 0.0)

# --- SRTM DEM / TWI ---
SRTM_DATASET = "USGS/SRTMGL1_003"
SRTM_RESOLUTION_M = 30

# --- OSM / graph construction ---
BUILDING_ADJACENCY_THRESHOLD_M = 15.0  # per proposal Sec 3.4 unit testing note
MIN_COMPONENT_SIZE = 20  # drop connected components smaller than this (see prune_small_components)
OVERPASS_TAGS = {"building": True}
INFRA_TAGS = {
    "amenity": ["drinking_water", "toilets"],
    "man_made": ["water_tap", "water_well"],
    "highway": True,
}
# Public Overpass mirrors. The primary (overpass-api.de) often returns 502
# when at capacity; graph_utils cycles through these with backoff.
OVERPASS_URLS = (
    "https://overpass-api.de/api",
    "https://lz4.overpass-api.de/api",
    "https://overpass.kumi.systems/api",
)
OVERPASS_MAX_ATTEMPTS = 6
OVERPASS_RETRY_BASE_SLEEP_S = 5.0

# --- Structural deprivation labels (Sprint 3 / GraphSAGE) ---------------
# Three severity classes derived from UN-Habitat / SUPI-style indicators,
# proxied from OSM attributes + infrastructure distances (proposal Sec
# 3.2 / FR-04). Higher class index = more deprived.
#
#   0 = low    (Retain and Formalise candidate)
#   1 = medium (Retain and Upgrade candidate)
#   2 = high   (Compensated Resettlement / severe deprivation candidate)
#
# Thresholds are heuristics for the OSM proxy -- replace with survey-derived
# cut-points if field labels become available. Distances in metres; area in m^2.
DEPRIVATION_CLASS_NAMES = ("low", "medium", "high")
DEPRIVATION_NUM_CLASSES = 3
# Water / sanitation access: beyond these distances counts as full deficit.
DEPRIVATION_WATER_ACCESS_M = 200.0
DEPRIVATION_SANITATION_ACCESS_M = 200.0
# Footprint area below this is treated as an overcrowding proxy.
DEPRIVATION_OVERCROWDING_AREA_M2 = 25.0
# Road access: beyond this distance counts as full deficit.
DEPRIVATION_ROAD_ACCESS_M = 50.0
# Sprint 3 exit criterion (proposal Sec 3.4.3 weeks 5-6).
GRAPHSAGE_SPRINT3_MACRO_F1_TARGET = 0.60
# Final NFR (proposal NFR-01) -- Sprint 4 three-settlement bar.
GRAPHSAGE_FINAL_MACRO_F1_TARGET = 0.70

# --- Attention U-Net thresholds (Sprint 2 / NFR, used again in Sprint 4) ---
# Transfer-learning convergence check from Sprint 2 (proposal Sec Iterative
# Refinement). Sprint 4 reports this per settlement and pooled.
ATTENTION_UNET_TRANSFER_IOU_TARGET = 0.60
# Final NFR F1 bar (proposal Sec Model Validation and Testing / NFR).
ATTENTION_UNET_FINAL_F1_TARGET = 0.75

# Canonical settlement order for multi-settlement training / reporting.
SETTLEMENT_KEYS = tuple(SETTLEMENTS.keys())

# --- Late-fusion policy decision engine (Sprint 5) -----------------------
# Proposal Sec 3.2.3 / FR-05 / Sec 3.4.3 (weeks 9-10):
# The FusionEngine combines the Attention U-Net's flood hazard probability
# with GraphSAGE's 3-class deprivation severity probability vector into one
# of three governance categories per parcel:
#
#   0 = Retain and Formalise     (low flood risk, low deprivation)
#   1 = Retain and Upgrade       (safe/moderate flood risk with medium/high
#                                 deprivation, or moderate flood risk with
#                                 low deprivation needing drainage investment)
#   2 = Compensated Resettlement (high flood risk, or moderate flood risk
#                                 compounded by high structural deprivation)
POLICY_CLASS_NAMES = (
    "Retain and Formalise",
    "Retain and Upgrade",
    "Compensated Resettlement",
)
POLICY_NUM_CLASSES = 3
# Choropleth dashboard colors per proposal Sec 3.6.3 / 4.4.2 (green / amber / red).
POLICY_CLASS_COLORS = (
    "#2ecc71",  # Retain and Formalise (green)
    "#f39c12",  # Retain and Upgrade (amber)
    "#e74c3c",  # Compensated Resettlement (red)
)
# Flood probability cuts separating low / moderate / high flood hazard.
# Isolated as config constants so policy thresholds can be recalibrated for
# different cities or risk tolerances without modifying trained model weights
# (proposal Sec 3.6.1 Class Diagram).
FUSION_FLOOD_LOW_THRESHOLD = 0.30
FUSION_FLOOD_HIGH_THRESHOLD = 0.60
# Default buffer (metres) around OSM river/stream corridors when the 37 NRRP
# high-risk zone polygons are not supplied as an external file.
NRRP_FALLBACK_RIVER_BUFFER_M = 120.0
# Sprint 5 validation bar: Compensated Resettlement recommendations must
# concentrate inside the NRRP / riparian high-risk reference zone relative to
# the settlement baseline (lift > 1.0).
FUSION_NRRP_MIN_LIFT = 1.0

# --- Raster tiling ---
TILE_SIZE_PX = 256
TILE_OVERLAP_PX = 50
MIN_VALID_DATA_FRACTION = 0.6  # discard tiles with less valid data than this

# --- Raster export sizing ------------------------------------------------
# GEE exports at this scale (metres/pixel). The export geometry must be large
# enough that at least RASTER_MIN_TILE_LOCATIONS distinct 256px tile windows
# fit in BOTH dimensions, otherwise tile_raster_stack keeps 0 tiles. Kibera's
# bbox happens to clear this with a small buffer; Mathare/Mukuru are smaller
# and need a larger, size-aware buffer -- computed in
# gee_utils.required_export_buffer_m rather than hardcoded, so each settlement
# gets just enough buffer. A settlement needs >= 2 tile locations for the
# Sprint 4 spatial holdout to hold one out for validation.
RASTER_EXPORT_SCALE_M = 10
RASTER_MIN_TILE_LOCATIONS = 2
RASTER_EXPORT_BASE_BUFFER_M = 200   # floor; never buffer less than this
RASTER_EXPORT_BUFFER_MARGIN = 1.10  # 10% headroom over the strict minimum

# --- Output paths (relative to repo root; mount Drive at this path in Colab) ---
DATA_ROOT = "data"
RAW_DIR = f"{DATA_ROOT}/raw"
STACKS_DIR = f"{DATA_ROOT}/raster_stacks"
TILES_DIR = f"{DATA_ROOT}/tiles"
GRAPHS_DIR = f"{DATA_ROOT}/graphs"
OUTPUTS_DIR = "outputs"
CLASSIFICATIONS_DIR = f"{OUTPUTS_DIR}/classifications"
XAI_DIR = f"{OUTPUTS_DIR}/xai"
DASHBOARD_DIR = "dashboard"
