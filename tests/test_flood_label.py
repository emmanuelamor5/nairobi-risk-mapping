"""
Unit tests for the event-based flood label (src/utils/flood_label.py).

These run offline against synthetic VV/slope/water arrays, so they verify the
change-detection and refinement logic independent of Earth Engine. The core
claim being tested is the one that fixes the v1 problem: a pixel that is
stable across the two dates (land cover) must NOT be labelled flood, while a
genuine backscatter drop (new water) must be -- subject to the slope,
permanent-water and speckle refinements.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.flood_label import (
    apply_permanent_water_mask,
    apply_slope_mask,
    build_event_flood_label,
    change_image,
    flood_mask_from_change,
    otsu_threshold,
    remove_small_objects,
)


def test_change_image_is_event_minus_pre():
    pre = np.array([[-10.0, -12.0]])
    event = np.array([[-18.0, -12.0]])
    change = change_image(pre, event)
    assert change[0, 0] == pytest.approx(-8.0)   # big drop -> candidate water
    assert change[0, 1] == pytest.approx(0.0)    # stable -> not water


def test_change_image_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        change_image(np.zeros((2, 2)), np.zeros((2, 3)))


def test_change_image_propagates_nodata():
    pre = np.array([[np.nan, -10.0]])
    event = np.array([[-18.0, -10.0]])
    change = change_image(pre, event)
    assert np.isnan(change[0, 0])


def test_flood_mask_thresholds_only_drops():
    change = np.array([[-8.0, -1.0, 2.0, np.nan]])
    mask = flood_mask_from_change(change, threshold_db=-3.0)
    assert mask.tolist() == [[True, False, False, False]]  # nan never flood


def test_stable_landcover_is_not_flooded():
    """The core v1 fix: a pixel unchanged between dates must not be labelled flood, no matter how bright/dark it is."""
    pre = np.full((8, 8), -6.0)   # e.g. persistently bright SWIR-like surface
    event = np.full((8, 8), -6.0)
    label, report = build_event_flood_label(pre, event, use_otsu=False, threshold_db=-3.0)
    assert label.sum() == 0
    assert report["flooded_fraction"] == pytest.approx(0.0)


def test_slope_mask_removes_steep_pixels():
    mask = np.array([[True, True]])
    slope = np.array([[1.0, 30.0]])
    out = apply_slope_mask(mask, slope, max_slope_deg=5.0)
    assert out.tolist() == [[True, False]]


def test_slope_mask_keeps_unknown_slope():
    mask = np.array([[True]])
    slope = np.array([[np.nan]])
    assert apply_slope_mask(mask, slope, max_slope_deg=5.0).tolist() == [[True]]


def test_permanent_water_is_excluded():
    mask = np.array([[True, True]])
    permanent = np.array([[0, 1]])
    out = apply_permanent_water_mask(mask, permanent)
    assert out.tolist() == [[True, False]]


def test_remove_small_objects_drops_speckle_keeps_blobs():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True                 # single-pixel speckle
    mask[4:8, 4:8] = True             # 16-pixel contiguous blob
    out = remove_small_objects(mask, min_pixels=8)
    assert not out[0, 0]
    assert out[4:8, 4:8].all()


def test_build_event_label_full_pipeline():
    pre = np.full((16, 16), -8.0)
    event = pre.copy()
    event[2:10, 2:10] = -20.0         # a clear 64-pixel inundation patch (big drop)
    slope = np.zeros((16, 16))
    slope[2:4, 2:4] = 45.0            # part of the patch is on steep ground -> removed
    permanent = np.zeros((16, 16))
    permanent[9, 9] = 1              # one permanent-water pixel inside the patch

    label, report = build_event_flood_label(
        pre, event, slope_deg=slope, permanent_water=permanent,
        use_otsu=False, threshold_db=-3.0, max_slope_deg=5.0, min_object_pixels=4,
    )
    assert label.dtype == np.float32
    assert label.sum() > 0
    # steep pixels and the permanent-water pixel must be excluded
    assert label[2, 2] == 0.0
    assert label[9, 9] == 0.0
    assert report["flooded_fraction"] < 64 / 256  # strictly fewer than the raw patch
    steps = report["pixel_counts_by_step"]
    assert steps["after_threshold"] >= steps["after_slope_mask"] >= steps["after_permanent_water_mask"]


def test_otsu_separates_bimodal_distribution():
    low = np.full(500, -20.0)
    high = np.full(500, -5.0)
    values = np.concatenate([low, high])
    thr = otsu_threshold(values)
    assert -20.0 < thr < -5.0


def test_otsu_empty_raises():
    with pytest.raises(ValueError):
        otsu_threshold(np.array([np.nan, np.nan]))


def test_otsu_threshold_used_when_enabled():
    """With use_otsu, a bimodal drop distribution should be thresholded between the modes, not at the fixed fallback."""
    rng = np.random.default_rng(0)
    pre = np.full((40, 40), -8.0)
    event = pre + rng.normal(0.0, 0.5, size=pre.shape)  # stable half: change ~ 0 +/- noise
    event[:, :20] = -22.0 + rng.normal(0.0, 0.5, size=(40, 20))  # water half: change ~ -14 +/- noise
    label, report = build_event_flood_label(pre, event, use_otsu=True, min_object_pixels=1)
    assert report["threshold_source"] == "otsu"
    assert -14.0 < report["threshold_db"] < -1.0   # split sits between the two modes
    assert label[:, :20].mean() > 0.9   # water half labelled
    assert label[:, 20:].mean() < 0.1   # stable half not
