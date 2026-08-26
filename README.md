<<<<<<< HEAD
# Multi-Task Deep Learning for Just Urban Governance

Parcel-level flood risk micro-zonation and structural deprivation
assessment for Nairobi's informal settlements (Kibera, Mathare, Mukuru),
combining an Attention U-Net (flood hazard) and GraphSAGE (structural
deprivation) through a late-fusion policy decision engine, surfaced via an
explainable-AI Folium dashboard.

Strathmore University, School of Computing and Engineering Sciences —
ICS Project II. Full methodology in the project proposal.

## Status

**Sprint 1 (weeks 1-2): Data pipeline** — implemented. Acquires and
validates all four data sources (Sentinel-2, Sentinel-1, SRTM/TWI, OSM
vectors) for a given settlement and produces a tiled raster stack + a
serialised PyTorch Geometric graph. `DataPipeline` is settlement-agnostic
(pass `settlement_key="kibera" | "mathare" | "mukuru"`), and
`run_sprint1_all_settlements()` runs all three in one call — see notebook
section 5.

Sprints 2-6 (model heads, fusion engine, dashboard) are not yet
implemented — this repo currently covers the data layer only.

## Repo structure

```
src/
  config.py              # settlement boundaries, date ranges, thresholds
  data_pipeline.py        # DataPipeline class -- Sprint 1 entrypoint
  utils/
    gee_utils.py           # Sentinel-2/1, SRTM/TWI acquisition (Google Earth Engine)
    graph_utils.py          # OSM ingestion + proximity graph construction
notebooks/
  01_sprint1_data_pipeline.ipynb   # Colab notebook, runs the pipeline end-to-end
tests/
  test_graph_construction.py       # unit tests for graph edge cases (no network needed)
data/                      # gitignored; populated at runtime
```

## Setup

### Local (for editing / running tests)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

The unit tests run entirely offline against synthetic geometries — they
don't need GEE or OSM credentials.

### Google Colab (for actually running the pipeline)

Open `notebooks/01_sprint1_data_pipeline.ipynb` in Colab. It will:

1. Clone this repo and `pip install -r requirements.txt`
2. Authenticate Earth Engine against your GCP project (you need a Cloud
   project with the Earth Engine API enabled — see
   https://developers.google.com/earth-engine/guides/access)
3. Mount Google Drive, so raster exports and checkpoints persist across
   sessions (Colab wipes local disk on disconnect)
4. Run `DataPipeline(settlement_key="kibera").run_sprint1()` as the MVP
   target from the proposal's prototype-planning section

**GEE exports are asynchronous.** `acquire_raster_stack()` kicks off a
`Export.image.toDrive` task and returns immediately with a task handle —
check `task.status()` or your Drive folder (`nairobi_risk_mapping/`) for
completion before calling `tile_raster_stack()` on the downloaded GeoTIFF.

## Known approximations to revisit before final training runs

- **Settlement boundaries** default to live OSM/Nominatim geocoding with a
  hardcoded bbox fallback (see `config.py` docstring). Neither is a
  verified administrative or cadastral boundary — replace with a
  digitized boundary (Map Kibera, KISIP II, or NCC planning data) before
  reporting final model metrics.
- **TWI flow accumulation** (`gee_utils.get_srtm_twi`) uses a coarse
  neighbourhood-smoothing proxy, not a true D8/D-infinity flow routing
  algorithm — GEE has no native hydrology flow-accumulation reducer.
  For a rigorous TWI, export the DEM and recompute flow accumulation in a
  dedicated hydrology package (WhiteboxTools, pysheds) before final
  training.
- **Sentinel-1 Otsu thresholding** for historical flood labels
  (`get_sentinel1_flood_mask`) is a first-pass label source per the
  proposal; validate a sample against the NRRP's 37 high-risk zone
  reference polygons before trusting it as ground truth for U-Net
  training.

## Testing

```bash
pytest tests/ -v
```

Graph construction tests specifically target the two failure modes the
proposal's testing plan (Sec 3.4) calls out: disconnected subgraphs and
incorrectly thresholded adjacency edges, both of which fail silently
(no exception) and propagate into GraphSAGE's neighbourhood aggregation.
=======
# MTL-Nairobi-2026
MTL Nairobi is a deep learning model that serves as an evidence-based XAI tool for community groups and city planners to assess housing conditions and access to water and sanitation against UN-Habitat standards in Three major informal settlement regions of Kibera, Mathare and Mukuru in Nairobi. 
>>>>>>> 4d100710778bf38eb0dd548568d0fbbe45f7d859
