# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

Multi-task deep learning for parcel-level flood risk and structural
deprivation assessment in Nairobi's informal settlements (Kibera, Mathare,
Mukuru). An Attention U-Net (flood hazard) and a GraphSAGE network
(structural deprivation) are trained independently and fused in a late-fusion
policy decision engine. See `README.md` for current sprint status and the
project proposal for the full methodology.

Two execution contexts:

- **Local / CI / Cloud Agents** — editing code and running the offline unit
  tests. No Google Earth Engine credentials, GPU, or Google Drive data.
- **Google Colab** — where the pipeline, Earth Engine exports, and model
  training actually run (free T4 GPU, native EE auth, Drive persistence).

## Cursor Cloud specific instructions

You are almost always running in the local/Cloud-Agent context, **not** Colab.
Plan testing accordingly.

- **Can run here (do this):** the offline unit test suite. It uses only
  synthetic arrays/geometries — no network, GEE, GPU, or Drive.

  ```bash
  pip install -r requirements.txt   # torch, numpy, scipy, rasterio, geopandas, sklearn, pytest, ...
  pytest tests/ -q
  ```

  All tests must pass before a change is considered done. Add offline tests
  for any new pure-logic code (label construction, metrics, dataset/layout,
  model architecture) following the existing files in `tests/`.

- **Cannot run here (do not block on these):** anything that needs Google
  Earth Engine (`gee_utils`, `DataPipeline.acquire_raster_stack`,
  `build_raster_stack`), a GPU training run on real tiles, Google Drive, or
  Colab magics (`%cd`, `!pip`, `drive.mount`). These only run in Colab. Do not
  fabricate GEE/training results; hand the notebook to the user to run.

- **Verifying notebooks:** notebooks target Colab, so do not execute them end
  to end here. Instead validate them structurally — the notebook JSON parses,
  code cells are syntactically valid (ignoring Colab magics), and every
  `src/` API/attribute/signature a cell references actually exists. Keep heavy,
  GEE-/GPU-bound logic in importable `src/` modules (not inline in the
  notebook) so it can be unit-tested offline.

- **Evidence when GEE/training is involved:** since real metrics require Colab,
  prove the code path offline with a synthetic end-to-end run — build a small
  synthetic raster stack with `rasterio`, tile it via `DataPipeline`, run the
  numpy label/validation/diagnostics functions, and train the Attention U-Net
  for a few epochs on CPU. This validates integration without claiming real
  accuracy. State plainly that real accuracy must be measured in Colab.

- **Process hygiene:** never kill processes by name (`pkill -f`); if you must,
  target a specific PID. Leave any started services running.

## Code conventions

- Band order of a tiled stack is described by `dataset.StackLayout`
  (`DEFAULT_LAYOUT` = 9 inputs; `EVENT_S1_LAYOUT` = 11 inputs with Sentinel-1
  bands). Derive label/input indices from the layout rather than hardcoding
  them.
- Flood-label construction logic lives in pure numpy (`src/utils/flood_label.py`)
  and is mirrored server-side in `gee_utils`; keep the two in sync and unit-test
  the numpy core.
- Comments explain intent/trade-offs, not line-by-line mechanics (match the
  existing narrative docstrings).
