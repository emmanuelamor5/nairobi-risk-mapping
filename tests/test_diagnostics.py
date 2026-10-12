"""
Unit tests for label-quality diagnostics (src/models/diagnostics.py): the
cross-location pixel AUC and permutation-importance probes that re-run the
v1 post-mortem against any label. Offline, synthetic tiles only.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.diagnostics import (
    cross_location_signal,
    evaluate_auc,
    fit_pixel_classifier,
    permutation_importance_auc,
    tile_pixels,
)

INPUT_BANDS = [0, 1, 2]
LABEL_BAND = 3


def _tile_with_signal_in_band(signal_band: int, h=40, w=40, seed=0):
    """Build a (4, h, w) tile whose label is driven purely by one input band."""
    rng = np.random.default_rng(seed)
    bands = rng.normal(size=(3, h, w))
    label = (bands[signal_band] > 0).astype(np.float32)
    tile = np.concatenate([bands, label[None]], axis=0).astype(np.float32)
    return tile


def test_tile_pixels_drops_nonfinite_and_shapes():
    tile = _tile_with_signal_in_band(0)
    tile[0, 0, 0] = np.nan
    X, y = tile_pixels(tile, INPUT_BANDS, LABEL_BAND)
    assert X.shape[1] == 3
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == 40 * 40 - 1  # one nan pixel removed
    assert np.isfinite(X).all()


def test_tile_pixels_subsample():
    tile = _tile_with_signal_in_band(0)
    X, y = tile_pixels(tile, INPUT_BANDS, LABEL_BAND, subsample=100)
    assert X.shape[0] == 100


def test_classifier_recovers_signal_and_auc_high():
    train = _tile_with_signal_in_band(1, seed=1)
    X, y = tile_pixels(train, INPUT_BANDS, LABEL_BAND)
    clf = fit_pixel_classifier(X, y)
    assert evaluate_auc(clf, X, y) > 0.95  # separable by construction


def test_fit_raises_on_single_class():
    tile = np.zeros((4, 8, 8), dtype=np.float32)  # all-zero label
    X, y = tile_pixels(tile, INPUT_BANDS, LABEL_BAND)
    with pytest.raises(ValueError, match="single class"):
        fit_pixel_classifier(X, y)


def test_permutation_importance_points_at_the_driving_band():
    tile = _tile_with_signal_in_band(2, seed=3)
    X, y = tile_pixels(tile, INPUT_BANDS, LABEL_BAND)
    clf = fit_pixel_classifier(X, y)
    imp = permutation_importance_auc(clf, X, y, ["b0", "b1", "b2"], n_repeats=5)
    # the band that generated the label should be the most important
    top_band = next(iter(imp))
    assert top_band == "b2"
    assert imp["b2"]["mean"] > imp["b0"]["mean"]


def test_cross_location_signal_reports_three_numbers():
    train = _tile_with_signal_in_band(0, seed=4)
    val = _tile_with_signal_in_band(0, seed=5)
    report = cross_location_signal(train, val, INPUT_BANDS, LABEL_BAND, ["b0", "b1", "b2"],
                                   subsample=1000, n_repeats=3)
    assert "train_auc" in report and "val_auc" in report
    assert report["val_auc"] > 0.9  # same generative rule across locations -> transfers
    assert report["permutation_importance"] is not None
