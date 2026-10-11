"""
Leakage-aware evaluation for the Sprint 2 spatial holdout.

Sprint 1 tiles with a 256px window and a 206px stride, so tiles at adjacent
locations share a 50px strip of the same ground. When one location is held
out for validation, that strip also appears in a training tile, which lets
the model see part of the validation area during training and inflates the
validation score. These helpers compute exactly which pixels of a validation
tile are also covered by training tiles and evaluate with those excluded.
"""

import numpy as np
import torch

from .dataset import FLOOD_LABEL_BAND_INDEX, INPUT_BAND_INDICES
from .metrics import f1_from_counts, iou_from_counts


def validation_overlap_mask(val_location: tuple, train_locations: list,
                            tile_size: int = 256) -> np.ndarray:
    """
    Boolean (tile_size, tile_size) mask in the validation tile's own pixel
    coordinates: True where the pixel is NOT covered by any training tile
    (safe to evaluate on), False where it is.

    Locations are (row_off, col_off) raster offsets, as stored in the manifest.
    """
    r0, c0 = val_location
    safe = np.ones((tile_size, tile_size), dtype=bool)
    for r, c in train_locations:
        row_start = max(r - r0, 0)
        row_end = min(r - r0 + tile_size, tile_size)
        col_start = max(c - c0, 0)
        col_end = min(c - c0 + tile_size, tile_size)
        if row_start < row_end and col_start < col_end:
            safe[row_start:row_end, col_start:col_end] = False
    return safe


def evaluate_masked(model, tile_path: str, safe_mask: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Evaluate a trained AttentionUNetModel on one saved tile, counting only
    pixels where safe_mask is True. Use the un-augmented ("orig") tile, since
    flipped or rotated variants would need the mask transformed identically.
    """
    tile = np.load(tile_path)
    inputs = np.nan_to_num(tile[INPUT_BAND_INDICES], nan=0.0)
    label = np.nan_to_num(tile[FLOOD_LABEL_BAND_INDEX], nan=0.0)

    proba = model.predict_proba(torch.from_numpy(inputs).float().unsqueeze(0))[0, 0].numpy()
    pred = proba >= threshold
    truth = label >= 0.5
    m = safe_mask

    counts = {
        "tp": float(np.sum(pred & truth & m)),
        "fp": float(np.sum(pred & ~truth & m)),
        "fn": float(np.sum(~pred & truth & m)),
        "tn": float(np.sum(~pred & ~truth & m)),
    }
    return {
        "iou": iou_from_counts(counts),
        "f1": f1_from_counts(counts),
        "evaluated_pixels": int(m.sum()),
        "total_pixels": int(m.size),
        "label_positive_fraction": float(truth[m].mean()) if m.any() else float("nan"),
        # IoU you'd get by predicting "flooded" everywhere -- the floor any real model must beat
        "all_flooded_baseline_iou": float(truth[m].mean()) if m.any() else float("nan"),
    }
