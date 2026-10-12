"""
Unit tests for label validation against a spatial reference
(src/utils/label_validation.py). The numpy core is tested offline; the geo
rasterisation helpers are exercised against a tiny synthetic GeoDataFrame
when geopandas/rasterio are importable.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.label_validation import (
    agreement_metrics,
    compare_labels,
    confusion_against_reference,
)


def test_confusion_counts_basic():
    label = np.array([[1, 1], [0, 0]])
    reference = np.array([[1, 0], [1, 0]])
    counts = confusion_against_reference(label, reference)
    assert counts == {"tp": 1.0, "fp": 1.0, "fn": 1.0, "tn": 1.0}


def test_confusion_shape_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        confusion_against_reference(np.zeros((2, 2)), np.zeros((3, 3)))


def test_lift_above_one_when_label_concentrates_in_reference():
    """A label whose flooded pixels sit inside the reference zone should score lift > 1."""
    reference = np.zeros((10, 10), dtype=int)
    reference[:, :3] = 1              # reference covers 30% of the scene
    label = np.zeros((10, 10))
    label[:, :3] = 1                 # all flood inside the reference
    m = agreement_metrics(label, reference)
    assert m["precision"] == pytest.approx(1.0)
    assert m["reference_fraction"] == pytest.approx(0.3)
    assert m["lift"] == pytest.approx(1 / 0.3, rel=1e-3)
    assert m["lift"] > 1


def test_lift_near_one_for_land_cover_proxy():
    """A label spread independently of the reference (the land-cover-proxy signature) scores lift ~1."""
    rng = np.random.default_rng(0)
    reference = (rng.random((100, 100)) < 0.3).astype(int)
    label = (rng.random((100, 100)) < 0.3).astype(float)  # independent of reference
    m = agreement_metrics(label, reference)
    assert 0.8 < m["lift"] < 1.2


def test_valid_mask_restricts_evaluation():
    label = np.array([[1, 1], [1, 1]])
    reference = np.array([[1, 0], [0, 0]])
    valid = np.array([[1, 1], [0, 0]])
    m = agreement_metrics(label, reference, valid_mask=valid)
    assert m["valid_pixels"] == 2
    assert m["precision"] == pytest.approx(0.5)


def test_compare_labels_runs_each():
    reference = np.zeros((10, 10), dtype=int)
    reference[:, :5] = 1
    good = np.zeros((10, 10)); good[:, :5] = 1       # concentrated in reference
    bad = np.zeros((10, 10)); bad[:, 5:] = 1          # avoids reference
    result = compare_labels({"good": good, "bad": bad}, reference)
    assert result["good"]["lift"] > result["bad"]["lift"]


def test_load_reference_polygons_geojson_and_crs():
    pytest.importorskip("geopandas")
    import tempfile, os
    import geopandas as gpd
    from shapely.geometry import box, Polygon

    from src.utils.label_validation import load_reference_polygons

    # A valid polygon plus a self-intersecting (invalid) bowtie that buffer(0) should repair.
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), bowtie], crs="EPSG:4326")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "nrrp.geojson")
        gdf.to_file(p, driver="GeoJSON")
        loaded = load_reference_polygons(p, target_crs="EPSG:3857")
        assert len(loaded) >= 1
        assert loaded.crs.to_epsg() == 3857
        assert loaded.geometry.is_valid.all()


def test_load_reference_polygons_rejects_non_polygons():
    pytest.importorskip("geopandas")
    import tempfile, os
    import geopandas as gpd
    from shapely.geometry import Point

    from src.utils.label_validation import load_reference_polygons

    gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "points.geojson")
        gdf.to_file(p, driver="GeoJSON")
        with pytest.raises(ValueError, match="polygon"):
            load_reference_polygons(p)


def test_rasterize_reference_against_grid():
    pytest.importorskip("rasterio")
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box
    from affine import Affine

    from src.utils.label_validation import rasterize_reference

    # A 10x10 grid at unit pixel size, origin (0, 10) going down (north-up).
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 5, 10)], crs="EPSG:3857")
    mask = rasterize_reference(gdf, transform, (10, 10), "EPSG:3857")
    assert mask.shape == (10, 10)
    assert mask[:, :5].all()         # left half burned
    assert not mask[:, 5:].any()     # right half empty
