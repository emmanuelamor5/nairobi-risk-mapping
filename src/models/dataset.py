"""
Dataset class for loading Sprint 1's tiled .npy raster stacks and splitting
them into U-Net inputs (9 bands: 6 optical + elevation + slope + TWI) and
the flood_label training target (band index 6 in the 10-band stack --
see src/config.py's S2_BANDS ordering and src/utils/gee_utils.py's
build_raster_stack band order: S2 optical, then flood_label, then terrain).
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Band layout produced by gee_utils.build_raster_stack(), in order:
# [B2, B3, B4, B8, B11, B12, flood_label, elevation, slope, twi]
FLOOD_LABEL_BAND_INDEX = 6
INPUT_BAND_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9]  # everything except flood_label


class FloodTileDataset(Dataset):
    """
    Loads tiles listed in a Sprint 1 manifest (data/tiles/<settlement>/manifest_<split>.json).

    Residual NaN pixels (SAR coverage gaps at tile/geometry edges -- see
    Sprint 1 diagnostics, typically well under 1% of pixels per tile) are
    filled with 0 in both inputs and the label. This is a deliberate
    simplification for the Sprint 2 baseline: it treats a small number of
    edge pixels as "no data, assume non-flooded" rather than excluding them
    from the loss entirely. Worth revisiting with a proper valid-pixel mask
    if per-tile NaN fraction grows once Mathare/Mukuru tiles are added in
    Sprint 4 -- check that before assuming this simplification still holds.
    """

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        if len(self.manifest) == 0:
            raise ValueError(
                f"Manifest at {manifest_path} is empty -- no tiles to train on. "
                f"Check Sprint 1's tiling step actually produced output before "
                f"building a dataset from it."
            )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        entry = self.manifest[idx]
        tile = np.load(entry["path"])  # shape: (10, H, W)

        inputs = tile[INPUT_BAND_INDICES]
        label = tile[FLOOD_LABEL_BAND_INDEX : FLOOD_LABEL_BAND_INDEX + 1]  # keep channel dim: (1, H, W)

        inputs = np.nan_to_num(inputs, nan=0.0)
        label = np.nan_to_num(label, nan=0.0)

        return {
            "inputs": torch.from_numpy(inputs).float(),
            "label": torch.from_numpy(label).float(),
            "tile_id": entry["tile_id"],
            "row_off": entry["row_off"],
            "col_off": entry["col_off"],
        }


def spatial_holdout_split(manifest_path: str, holdout_location_index: int = -1) -> dict:
    """
    Sprint 2 spatial train/validation split for the Kibera MVP stage.

    IMPORTANT -- this deliberately does NOT implement the proposal's
    originally-specified 5-fold spatially-contiguous cross-validation.
    That validation design assumes enough genuinely distinct spatial tile
    locations to split five ways; Kibera's actual exportable extent at
    256x256px/10m resolution produces only a small number of distinct
    locations (2, as of Sprint 1's debugging), which is smaller than what
    5-fold spatial CV requires by construction, not due to a pipeline bug.

    This function instead holds out entire spatial locations (not
    individual augmented variants of the same location) for validation --
    preserving the spatial-contiguity principle behind the original design
    even though the fold count is smaller. All augmented variants (flip/
    rotation) of a given location stay together on one side of the split,
    since splitting augmented copies of the same tile across train and
    validation would be the same spatial-leakage problem the original
    5-fold design was built to avoid.

    Document this substitution explicitly in your write-up -- see the
    Sprint 2 notebook's markdown cell for suggested wording. Revisit with
    genuine 5-fold spatial CV in Sprint 4 once Mathare and Mukuru widen the
    real spatial extent -- check the combined distinct-location count
    before assuming that's actually achieved by then.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    distinct_locations = sorted(set((t["row_off"], t["col_off"]) for t in manifest))
    if len(distinct_locations) < 2:
        raise ValueError(
            f"Only {len(distinct_locations)} distinct spatial location(s) in this "
            f"manifest -- can't hold one out for validation while leaving any for "
            f"training. Widen the export geometry to get more tileable area before "
            f"attempting this split."
        )

    holdout_location = distinct_locations[holdout_location_index]
    train_entries = [t for t in manifest if (t["row_off"], t["col_off"]) != holdout_location]
    val_entries = [t for t in manifest if (t["row_off"], t["col_off"]) == holdout_location]

    return {
        "train": train_entries,
        "val": val_entries,
        "holdout_location": holdout_location,
        "num_distinct_locations": len(distinct_locations),
    }
