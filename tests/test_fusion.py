"""
Offline tests for the Sprint 5 late-fusion policy engine.

Synthetic probabilities and geometries only. The NRRP lift bar is checked
on a constructed concentration pattern; nothing here claims a real
settlement result.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.models.fusion import (
    DISPLACEMENT_ACTION,
    POLICY_ACTIONS,
    POLICY_MATRIX,
    aggregate_tile_probabilities,
    action_labels,
    decide_policy,
    describe_policy_matrix,
    format_policy_report,
    justice_audit,
    load_policy_table,
    newest_geotiff,
    resolve_policy_table,
    parcel_reference_mask,
    sample_flood_at_pos,
    sample_geotiff_at_pos,
    parcel_rowcol,
    sample_raster_at_xy,
    save_policy_table,
    select_orig_manifest_entries,
    summarize_policy,
    validate_against_reference,
)


def _one_hot(cls: int, n: int = 1) -> np.ndarray:
    dep = np.zeros((n, 3), dtype=np.float64)
    dep[:, cls] = 1.0
    return dep


def test_matrix_cells_cover_every_flood_and_deprivation_pair():
    """Each of the six cells maps to the documented action."""
    assert describe_policy_matrix()
    for flood, row in ((0.0, 0), (1.0, 1)):
        for cls in range(3):
            decision = decide_policy([flood], _one_hot(cls))
            assert decision["action_index"][0] == decision["matrix_action_index"][0]
            assert POLICY_ACTIONS[decision["action_index"][0]] == POLICY_MATRIX[row][cls]
            assert decision["confidence"][0] == pytest.approx(1.0)
            assert decision["fused_features"].shape == (1, 4)
            assert decision["fused_features"][0, 0] == pytest.approx(flood)
            assert decision["fused_features"][0, 1 + cls] == pytest.approx(1.0)


def test_displacement_requires_high_hazard_and_majority_deprivation():
    high = _one_hot(2)
    low = _one_hot(0)
    dry_high = decide_policy([0.2], high)
    wet_low = decide_policy([0.9], low)
    wet_high = decide_policy([0.9], high)
    assert action_labels(dry_high["action_index"])[0] == "in_situ_upgrade_priority"
    assert action_labels(wet_low["action_index"])[0] == "flood_resilient_upgrade"
    assert action_labels(wet_high["action_index"])[0] == DISPLACEMENT_ACTION
    assert not dry_high["displacement"][0]
    assert not wet_low["displacement"][0]
    assert wet_high["displacement"][0]


def test_majority_gate_blocks_plurality_resettlement():
    """0.40/0.15/0.45 is a high-class plurality, not a majority."""
    dep = np.array([[0.40, 0.15, 0.45]])
    decision = decide_policy([0.9], dep)
    assert decision["deprivation_class"][0] == 2
    assert action_labels(decision["matrix_action_index"])[0] == DISPLACEMENT_ACTION
    assert action_labels(decision["action_index"])[0] == "flood_resilient_upgrade"
    assert decision["majority_gate_applied"][0]
    assert not decision["displacement"][0]
    assert justice_audit(decision)["passes"]


def test_nrrp_override_changes_the_hazard_row_only():
    high = _one_hot(2)
    low = _one_hot(0)
    # Dry according to the U-Net, but inside the zone.
    in_zone_high = decide_policy([0.1], high, inside_nrrp=[True])
    in_zone_low = decide_policy([0.1], low, inside_nrrp=[True])
    out_zone_high = decide_policy([0.1], high, inside_nrrp=[False])
    assert action_labels(in_zone_high["action_index"])[0] == DISPLACEMENT_ACTION
    assert action_labels(in_zone_low["action_index"])[0] == "flood_resilient_upgrade"
    assert action_labels(out_zone_high["action_index"])[0] == "in_situ_upgrade_priority"
    # The model decision ignores the zone, so validation stays non-circular.
    assert action_labels(in_zone_high["model_action_index"])[0] == "in_situ_upgrade_priority"
    assert not in_zone_high["model_flood_responsive"][0]
    assert in_zone_high["flood_responsive"][0]
    assert in_zone_high["override_applied"][0]
    assert not out_zone_high["override_applied"][0]


def test_missing_flood_inside_zone_uses_the_constraint():
    decision = decide_policy([np.nan], _one_hot(2), inside_nrrp=[True])
    assert not decision["model_valid"][0]
    assert decision["valid"][0]
    assert action_labels(decision["action_index"])[0] == DISPLACEMENT_ACTION


def test_corrupt_flood_is_not_repaired_by_the_zone():
    decision = decide_policy([1.5], _one_hot(2), inside_nrrp=[True])
    assert not decision["valid"][0]
    assert action_labels(decision["action_index"])[0] == "insufficient_evidence"
    assert not decision["displacement"][0]


def test_justice_audit_holds_on_a_random_batch():
    rng = np.random.default_rng(0)
    n = 300
    flood = rng.random(n)
    logits = rng.normal(size=(n, 3))
    dep = np.exp(logits)
    dep /= dep.sum(axis=1, keepdims=True)
    inside = rng.random(n) < 0.25
    # A few missing flood values, including some inside the zone.
    flood[::40] = np.nan
    decision = decide_policy(flood, dep, inside_nrrp=inside)
    audit = justice_audit(decision)
    assert audit["passes"]
    displaced = decision["displacement"]
    assert np.all(decision["deprivation_class"][displaced] == 2)
    assert np.all(decision["p_high"][displaced] > 0.5)
    assert np.all(decision["policy_flood_high"][displaced])
    # Class-score ties prefer the less severe class.
    tied = decide_policy([0.1], np.array([[0.5, 0.5, 0.0]]))
    assert tied["deprivation_class"][0] == 0


def test_model_lift_is_not_manufactured_by_the_override():
    """
    Every parcel is dry according to the U-Net. The override makes in-zone
    parcels flood-responsive, but the exit bar looks at the model action.
    """
    n = 50
    flood = np.full(n, 0.1)
    dep = _one_hot(0, n)
    inside = np.zeros(n, dtype=bool)
    inside[:10] = True
    decision = decide_policy(flood, dep, inside_nrrp=inside)
    report = validate_against_reference(decision, inside)
    assert report["policy_agreement"]["lift"] > 1.0
    assert report["lift"] < 1.0
    assert not report["meets_lift_target"]
    assert report["n_override_applied"] == 10


def test_model_lift_passes_when_responsive_actions_concentrate():
    n = 100
    flood = np.zeros(n)
    flood[:20] = 0.95
    dep = _one_hot(0, n)  # high hazard -> resilient upgrade, still flood-responsive
    inside = np.zeros(n, dtype=bool)
    inside[:20] = True
    decision = decide_policy(flood, dep, inside_nrrp=inside)
    report = validate_against_reference(decision, inside)
    assert report["lift"] == pytest.approx(1 / 0.2, rel=1e-3)
    assert report["meets_lift_target"]
    assert report["lift_target"] == config.FUSION_NRRP_LIFT_TARGET
    text = format_policy_report(
        summarize_policy(decision), report, justice_audit(decision),
    )
    assert "meets=True" in text
    assert justice_audit(decision)["passes"]


def test_shape_errors_and_threshold():
    with pytest.raises(ValueError, match="deprivation_proba"):
        decide_policy([0.2, 0.4], _one_hot(0, 1))
    with pytest.raises(ValueError, match="flood_threshold"):
        decide_policy([0.2], _one_hot(0), flood_threshold=0.0)
    with pytest.raises(ValueError, match="inside_reference"):
        decision = decide_policy([0.2], _one_hot(1))
        validate_against_reference(decision, np.array([True, False]))


def test_summarize_and_roundtrip(tmp_path):
    flood = np.array([0.1, 0.1, 0.1, 0.9, 0.9, np.nan])
    dep = np.vstack([_one_hot(c) for c in (0, 1, 2, 0, 2)] + [_one_hot(1)])
    settlement = np.array([0, 0, 0, 1, 1, 1])
    decision = decide_policy(flood, dep, inside_nrrp=np.array([0, 0, 0, 0, 0, 1], dtype=bool))
    summary = summarize_policy(
        decision, settlement_index=settlement, settlement_keys=["kibera", "mathare"],
    )
    assert summary["pooled"]["n"] == 6
    assert summary["pooled"]["n_valid"] == 6  # the NaN flood parcel is inside the zone
    assert set(summary["per_settlement"]) == {"kibera", "mathare"}
    assert summary["per_settlement"]["kibera"]["n"] == 3
    path = save_policy_table(
        tmp_path / "policy.npz",
        decision,
        settlement_index=settlement,
        settlement_keys=["kibera", "mathare"],
    )
    loaded = load_policy_table(path)
    assert loaded["action_index"].shape == (6,)
    assert list(loaded["settlement_keys"]) == ["kibera", "mathare"]
    np.testing.assert_array_equal(loaded["action_index"], decision["action_index"])

    import gzip
    gz_path = tmp_path / "policy_table.npz.gz"
    gz_path.write_bytes(gzip.compress(path.read_bytes()))
    from_gzip = load_policy_table(gz_path)
    np.testing.assert_array_equal(from_gzip["action_index"], decision["action_index"])
    found = resolve_policy_table(extra_dirs=[tmp_path])
    assert found == gz_path
    plain = tmp_path / "nested"
    plain.mkdir()
    (plain / "policy_table.npz").write_bytes(path.read_bytes())
    assert resolve_policy_table(extra_dirs=[plain]) == plain / "policy_table.npz"


def test_sample_raster_and_aggregate_tiles():
    grid = np.arange(16, dtype=np.float64).reshape(4, 4)
    # x = col, y = 4 - row (north-up, origin at the top-left corner).
    transform = (1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    xs = np.array([0.5, 2.5, 10.0])
    ys = np.array([3.5, 1.5, 3.5])
    sampled = sample_raster_at_xy(grid, transform, xs, ys)
    assert sampled[0] == pytest.approx(grid[0, 0])
    assert sampled[1] == pytest.approx(grid[2, 2])
    assert np.isnan(sampled[2])

    pos = np.column_stack([xs[:2], ys[:2]])
    via_pos = sample_flood_at_pos(grid, transform, pos, "EPSG:32737", "EPSG:32737")
    np.testing.assert_allclose(via_pos, sampled[:2])

    tiles = [
        {"row_off": 0, "col_off": 0, "proba": np.full((2, 2), 0.2)},
        {"row_off": 1, "col_off": 1, "proba": np.full((2, 2), 0.8)},
    ]
    # Parcel (1, 1) is in both tiles: mean of 0.2 and 0.8. (0, 0) only the first.
    # (3, 3) is in neither.
    rows, cols = parcel_rowcol(np.column_stack([xs, ys]), transform)
    np.testing.assert_array_equal(rows, [0, 2, 0])
    np.testing.assert_array_equal(cols, [0, 2, 10])

    entries = [
        {"tile_id": "kibera_0_0_orig", "row_off": 0, "col_off": 0, "path": "a.npy"},
        {"tile_id": "kibera_0_0_flip_h", "row_off": 0, "col_off": 0, "path": "b.npy"},
        {"tile_id": "kibera_0_0_rot90", "row_off": 0, "col_off": 0, "path": "c.npy"},
        {"tile_id": "kibera_206_0_orig", "row_off": 206, "col_off": 0, "path": "d.npy"},
    ]
    kept = select_orig_manifest_entries(entries)
    assert [(e["row_off"], e["col_off"]) for e in kept] == [(0, 0), (206, 0)]
    assert kept[0]["path"] == "a.npy"

    merged = aggregate_tile_probabilities([1, 0, 3], [1, 0, 3], tiles)
    assert merged[0] == pytest.approx(0.5)
    assert merged[1] == pytest.approx(0.2)
    assert np.isnan(merged[2])


def test_parcel_reference_mask_intersects_polygons():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    parcels = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1), box(5, 5, 6, 6)],
        crs="EPSG:32737",
    )
    reference = gpd.GeoDataFrame(
        geometry=[box(0, 0, 2, 2)],
        crs="EPSG:32737",
    )
    mask = parcel_reference_mask(parcels, reference)
    assert mask.tolist() == [True, False]

    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:32737")
    assert parcel_reference_mask(empty, reference).shape == (0,)


def test_newest_geotiff_prefers_the_latest_earth_engine_reexport(tmp_path):
    import os
    import time

    older = tmp_path / "kibera_raster_stack.tif"
    newer = tmp_path / "kibera_raster_stack (2).tif"
    other = tmp_path / "mathare_raster_stack.tif"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    other.write_bytes(b"mathare")
    now = time.time()
    os.utime(older, (now - 50, now - 50))
    os.utime(newer, (now, now))
    found = newest_geotiff([tmp_path, tmp_path / "missing"], "kibera_raster_stack")
    assert found == newer
    assert newest_geotiff([tmp_path], "mukuru_raster_stack") is None


def test_sample_geotiff_roundtrip(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine

    path = tmp_path / "flood.tif"
    array = np.array([[0.1, 0.9], [0.3, 0.7]], dtype=np.float32)
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 9850000.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=2, count=1,
        dtype="float32", crs="EPSG:32737", transform=transform,
    ) as dst:
        dst.write(array, 1)
    # Pixel (0, 1) centre is x=500000+15, y=9850000-5.
    pos = np.array([[500015.0, 9849995.0]])
    sampled = sample_geotiff_at_pos(path, pos, "EPSG:32737")
    assert sampled[0] == pytest.approx(0.9, abs=1e-6)
