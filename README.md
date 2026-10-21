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
(offline engine + Colab notebook; a real NRRP lift needs Colab).

- `src/models/fusion.py` concatenates the U-Net flood probability with the
  GraphSAGE deprivation softmax and reads the pair off a fixed policy
  matrix (the governance rule, not a third trained network). The six cells
  are five actions:
  retain and formalise, retain and upgrade, priority in-situ upgrade,
  flood-resilient upgrade, and compensated resettlement.
- Compensated resettlement is only reached when hazard is high (U-Net
  probability ≥ `config.FUSION_FLOOD_THRESHOLD`, or the parcel intersects
  an NRRP high-risk zone) and high-deprivation probability is a
  majority (`> 0.5`). A plurality is not enough: the matrix cell is
  rewritten to in-place resilient upgrade. The zone forces the high-hazard
  row; it does not skip the deprivation gate.
- Validation reuses the Sprint 2 lift statistic on the **model** action,
  computed before the NRRP override so the test is not circular
  (`config.FUSION_NRRP_LIFT_TARGET`, lift must be strictly greater). A
  separate justice audit checks that no resettlement decision violates
  the majority-hazard rule; that audit has to pass even when the lift
  does not.
- Parcel centroids are stored on `data.pos` (EPSG:32737) in GraphSAGE node
  order so a flood raster or overlapping U-Net tiles can be sampled at
  buildings. Graphs written before Sprint 5 have no `data.pos`. The Sprint 5
  notebook re-fetches OSM once, copies centroids onto the saved graph when
  the node features still match, and reads flood stacks from the Drive
  export names (`{settlement}_raster_stack.tif`, including Earth Engine's
  ` (1)` re-export suffix).
- `notebooks/05_sprint5_late_fusion.ipynb` runs the matrix on synthetic
  parcels and, when checkpoints are present, on the three settlements.
  With no NRRP GeoJSON it fetches OSM rivers, streams, and drains once
  and scores lift against a 120 m corridor around them. The executed
  Colab run fused all 18,005 parcels (justice audit passed; 1,265
  resettlements). River-corridor lift was 0.929, so the exit bar was
  not met. That figure is the fallback, not the 37-zone test.

**Sprint 6: XAI dashboard** — implemented. The Folium map is served only
after Firebase email/password sign-in plus a TOTP or phone second factor.
`GraphSAGEModel.explain` returns exact local Shapley values for one
building. `AttentionUNetModel.explain` returns a Grad-CAM map from the
last decoder convolution. The signed-in page follows the supplied wireframes: a choropleth map
with a parcel card, and an XAI summary with SHAP bars, feature values,
and a prediction path. See
`notebooks/06_sprint6_dashboard.ipynb`. The Sprint 5 river-corridor lift
(0.929) is printed on the map page so the actions are not presented as
a validated flood-hazard layer.

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
    fusion.py               # Sprint 5: policy matrix, NRRP lift, parcel sampling
    explanations.py         # Sprint 6: local SHAP and decoder Grad-CAM
  dashboard/
    app.py                  # MFA-gated Folium server
    auth_gate.py            # Firebase second-factor check
    map_view.py             # parcel policy map
notebooks/
  01_sprint1_data_pipeline.ipynb        # Sprint 1 pipeline, end-to-end
  02_sprint2_attention_unet.ipynb       # Sprint 2 v1 baseline (overfit; label-limited)
  02_sprint2_attention_unet_v2.ipynb    # Sprint 2 v2: event label + validation + regularisation
  03_sprint3_graphsage.ipynb            # Sprint 3: GraphSAGE deprivation head on Kibera
  04_sprint4_three_settlements.ipynb    # Sprint 4: both heads on Kibera+Mathare+Mukuru
  05_sprint5_late_fusion.ipynb          # Sprint 5: policy matrix + NRRP validation
  06_sprint6_dashboard.ipynb            # Sprint 6: MFA gate + Folium policy map
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

