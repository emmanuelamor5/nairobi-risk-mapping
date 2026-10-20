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

**Sprint 2 (weeks 3-4): Attention U-Net flood head** — implemented, with a
v1 baseline and a v2 improvement pass.

- `notebooks/02_sprint2_attention_unet.ipynb` (v1) trained the Attention
  U-Net on the Sprint 1 tiles. It overfit: train IoU ~0.77 but validation
  never beat the flood-everywhere baseline (~0.31). Diagnostics traced this
  to the **label**, not the network — the dry-vs-rainy-season Otsu label was
  an unvalidated land-cover proxy (signal concentrated in SWIR/NIR optical
  bands; flooded fractions of 19–32% were implausibly high).
- `notebooks/02_sprint2_attention_unet_v2.ipynb` (v2) rebuilds the label as
  **bi-temporal change detection around a dated flood event** (the Apr–May
  2024 Nairobi floods — see `src/utils/flood_label.py` and
  `gee_utils.get_event_flood_label`), **validates it against an independent
  reference** with a lift statistic (`src/utils/label_validation.py`), lets
  the U-Net optionally see the Sentinel-1 bands its label comes from
  (`dataset.EVENT_S1_LAYOUT`), and adds dropout / weight decay / early
  stopping against the single-location overfitting. NRRP-polygon lift still
  needs a Colab run with the 37 high-risk zone geometries (OSM river-
  corridor fallback is available offline).

**Sprint 3 (weeks 5-6): GraphSAGE deprivation head** — implemented.

- `notebooks/03_sprint3_graphsage.ipynb` trains a two-layer mean-aggregation
  GraphSAGE on Kibera building nodes (`src/models/graphsage.py` +
  `graphsage_model.py`). Node features are footprint area, water /
  sanitation / road distances, and a roof-material durability score.
- Deprivation labels are a three-class UN-Habitat / SUPI **proxy** derived
  from those OSM observables (`src/utils/deprivation_label.py`) — not
  household-survey SUPI scores. Stratified 60/20/20 split; Adam
  `lr=5e-3`, `weight_decay=5e-4`; Sprint 3 exit bar is validation
  **macro-F1 ≥ 0.60** (`config.GRAPHSAGE_SPRINT3_MACRO_F1_TARGET`).

**Sprint 4 (weeks 7-8): Three-settlement extension** — implemented
(offline helpers + Colab notebook; real Mathare/Mukuru metrics need Colab).

- `src/models/multi_settlement.py` combines Kibera + Mathare + Mukuru
  GraphSAGE graphs (disjoint union), applies per-settlement stratified
  60/20/20 splits, and aggregates per-settlement + pooled macro-F1 against
  `config.GRAPHSAGE_FINAL_MACRO_F1_TARGET` (0.70).
- The same module loads multi-settlement tile manifests, runs a
  **per-settlement** spatial holdout (no cross-settlement location leakage;
  augmented variants stay together), and reports pooled / per-settlement
  flood IoU (≥ 0.60) and F1 (≥ 0.75).
- `notebooks/04_sprint4_three_settlements.ipynb` rebuilds all three graphs
  (Overpass retries), jointly trains GraphSAGE, fine-tunes / trains the
  Attention U-Net on pooled tiles when Drive manifests exist, and prints
  whether thresholds are met.

**Sprint 5 (weeks 9-10): Late-fusion policy engine** — implemented
(offline helpers + Colab notebook; real NRRP lift needs Colab + the 37-zone
polygons).

- `src/models/fusion_engine.py` concatenates the U-Net flood probability and
  the GraphSAGE deprivation probability vector and looks up a 3×3 rule matrix
  (`config.DEFAULT_POLICY_MATRIX`) that assigns each parcel **Retain and
  Formalise**, **Retain and Upgrade**, or **Compensated Resettlement**. The
  matrix and flood-bin cuts are isolated from trained weights so a different
  city can retune policy without retraining.
- `src/utils/parcel_io.py` samples the flood surface at building centroids,
  tests parcel-in-NRRP membership, and writes GeoJSON.
- Validation is **resettlement lift** against NRRP high-risk polygons (OSM
  river-corridor fallback): Compensated Resettlement should concentrate
  inside the overlay (`config.POLICY_NRRP_LIFT_TARGET` = 1.5).
- `notebooks/05_sprint5_fusion_engine.ipynb` runs the path on all three
  settlements and writes `outputs/classifications/`.

**Sprint 6 (weeks 11-12): XAI dashboard** — not yet implemented (Grad-CAM /
SHAP + Folium).

## Repo structure

```
src/
  config.py              # settlement boundaries, date ranges, flood-event + label thresholds
  data_pipeline.py        # DataPipeline class -- Sprint 1 entrypoint
  utils/
    gee_utils.py           # Sentinel-2/1, SRTM/TWI acquisition; seasonal + event flood labels (GEE)
    graph_utils.py          # OSM ingestion + proximity graph construction
    flood_label.py          # event-based change-detection label (pure numpy, Sprint 2 v2)
    label_validation.py     # label-vs-reference agreement / lift (Sprint 2 v2)
    deprivation_label.py    # UN-Habitat/SUPI-proxy node labels (pure numpy, Sprint 3)
    parcel_io.py            # Sprint 5: sample flood raster, NRRP membership, GeoJSON I/O
  models/
    attention_unet.py       # Attention U-Net (optional dropout)
    attention_unet_model.py # training/inference wrapper (weight decay, early stopping)
    dataset.py              # tile dataset + stack layouts (DEFAULT / EVENT_S1) + spatial holdout
    losses.py metrics.py evaluation.py  # BCE+Dice loss, IoU/F1, leakage-aware eval
    transfer_learning.py    # Sen1Floods11 partial-encoder transfer
    diagnostics.py          # cross-location pixel AUC + permutation importance (Sprint 2 v2)
    graphsage.py            # two-layer GraphSAGE (mean aggregation)
    graphsage_model.py      # training/inference wrapper (stratified split, macro-F1)
    graphsage_metrics.py    # macro-F1 + stratified 60/20/20 node split
    multi_settlement.py     # Sprint 4: combine graphs/tiles, splits, pooled metrics
    fusion_engine.py        # Sprint 5: late-fusion policy matrix + NRRP lift
notebooks/
  01_sprint1_data_pipeline.ipynb        # Sprint 1 pipeline, end-to-end
  02_sprint2_attention_unet.ipynb       # Sprint 2 v1 baseline (overfit; label-limited)
  02_sprint2_attention_unet_v2.ipynb    # Sprint 2 v2: event label + validation + regularisation
  03_sprint3_graphsage.ipynb            # Sprint 3: GraphSAGE deprivation head on Kibera
  04_sprint4_three_settlements.ipynb    # Sprint 4: both heads on Kibera+Mathare+Mukuru
  05_sprint5_fusion_engine.ipynb        # Sprint 5: late fusion + NRRP policy validation
tests/                     # offline unit tests (no network / GPU needed)
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
- **Flood labels.** The seasonal Otsu label (`get_sentinel1_flood_mask`) is
  a known-weak land-cover proxy and is kept only for reproducing the v1
  result. The current label is `get_event_flood_label` (bi-temporal event
  change detection, default of `build_raster_stack`). Either way, validate
  the label against the NRRP 37 high-risk zones before trusting it as ground
  truth — `notebooks/02_sprint2_attention_unet_v2.ipynb` section 5 does this
  with `src/utils/label_validation.py` (lift statistic; NRRP polygons, or an
  OSM river-corridor buffer as fallback).

## Testing

```bash
pytest tests/ -v
```

Graph construction tests specifically target the two failure modes the
proposal's testing plan (Sec 3.4) calls out: disconnected subgraphs and
incorrectly thresholded adjacency edges, both of which fail silently
(no exception) and propagate into GraphSAGE's neighbourhood aggregation.

