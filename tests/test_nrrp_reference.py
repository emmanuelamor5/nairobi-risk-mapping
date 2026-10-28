"""
Offline tests for src/utils/nrrp_reference.py -- the helpers that let the
Sprint 5 notebook score the fusion engine against the NRRP 37-zone polygons
honestly: buffer sensitivity, per-settlement lift, a permutation null, and a
trivial-rule baseline. Synthetic zones and parcels in UTM 37S; no network.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.models.fusion import decide_policy, validate_against_reference  # noqa: E402
from src.utils import nrrp_reference as nr  # noqa: E402


def _zones_metric():
    return gpd.GeoDataFrame(
        {"Name": ["Ngong corridor", "Mathare bend"]},
        geometry=[box(0, 0, 1000, 200), box(5000, 5000, 5400, 5400)],
        crs=config.CRS_METRIC,
    )


def _one_hot(cls, n):
    dep = np.zeros((n, 3))
    dep[:, cls] = 1.0
    return dep


def _two_settlement_scene(seed=0):
    """
    Kibera-like block: 400 parcels, 100 inside zone 1, with the model
    strongly concentrated on the zone. Mathare-like block: 400 parcels, 100
    inside zone 2, with the model flagging mostly the *outside* parcels.
    Pooled lift is positive, but the second settlement is anti-correlated.
    """
    rng = np.random.default_rng(seed)
    pos_a_in = np.column_stack([rng.uniform(10, 990, 100), rng.uniform(10, 190, 100)])
    pos_a_out = np.column_stack([rng.uniform(10, 990, 300), rng.uniform(400, 2000, 300)])
    pos_b_in = np.column_stack([rng.uniform(5010, 5390, 100), rng.uniform(5010, 5390, 100)])
    pos_b_out = np.column_stack([rng.uniform(5010, 5390, 300), rng.uniform(6000, 8000, 300)])
    pos = np.vstack([pos_a_in, pos_a_out, pos_b_in, pos_b_out])

    flood = np.full(800, 0.1)
    flood[:100] = 0.9                 # kibera inside -> wet (concentrated)
    flood[100:110] = 0.9              # a few kibera outside -> wet
    flood[400:420] = 0.9              # mathare inside: only 20 of 100 wet
    flood[500:700] = 0.9              # mathare outside: 200 of 300 wet (anti-correlated)
    inside_truth = np.zeros(800, dtype=bool)
    inside_truth[:100] = True
    inside_truth[400:500] = True
    sid = np.repeat([0, 1], 400)
    decision = decide_policy(flood, _one_hot(0, 800))
    return pos, decision, inside_truth, sid


def test_normalise_zones_assigns_ids_names_and_metric_crs():
    raw = gpd.GeoDataFrame(
        {"Name": ["A", "B", "C"]},
        geometry=[
            box(36.78, -1.32, 36.79, -1.31),
            LineString([(36.7, -1.3), (36.8, -1.3)]),     # dropped: not a polygon
            box(36.80, -1.32, 36.81, -1.31),
        ],
        crs="EPSG:4326",
    )
    zones = nr.normalise_zones(raw)
    assert len(zones) == 2
    assert zones.crs.to_string() == config.CRS_METRIC
    assert list(zones["zone_id"]) == [1, 2]
    assert list(zones["zone_name"]) == ["A", "C"]
    assert nr.zones_area_km2(zones) == pytest.approx(2.46, abs=0.05)


def test_normalise_zones_assumes_wgs84_and_rejects_empty():
    no_crs = gpd.GeoDataFrame(geometry=[box(36.78, -1.32, 36.79, -1.31)])
    with pytest.warns(UserWarning, match="no CRS"):
        zones = nr.normalise_zones(no_crs)
    assert zones.crs.to_string() == config.CRS_METRIC
    assert zones["zone_name"].iloc[0] == "zone_1"

    with pytest.raises(ValueError, match="no polygon"):
        nr.normalise_zones(gpd.GeoDataFrame(geometry=[Point(36.78, -1.31)], crs="EPSG:4326"))


def test_load_nrrp_zones_round_trips_a_geojson(tmp_path):
    raw = gpd.GeoDataFrame(
        {"zone_name": ["Kibera Soweto East"], "zone_id": [17]},
        geometry=[box(36.78, -1.32, 36.79, -1.31)],
        crs="EPSG:4326",
    )
    folder = tmp_path / config.BOUNDARY_DIR
    folder.mkdir(parents=True)
    path = folder / f"{nr.NRRP_FILE_STEM}.geojson"
    raw.to_file(path, driver="GeoJSON")

    assert nr.default_nrrp_path(tmp_path) == path
    assert nr.default_nrrp_path(tmp_path / "elsewhere") is None
    zones = nr.load_nrrp_zones(path)
    assert zones.crs.to_string() == config.CRS_METRIC
    assert list(zones["zone_id"]) == [17]
    assert list(zones["zone_name"]) == ["Kibera Soweto East"]
    with pytest.raises(FileNotFoundError):
        nr.load_nrrp_zones(tmp_path / "missing.geojson")


def test_buffer_and_mask_handle_crs_and_distance():
    zones = nr.normalise_zones(_zones_metric())
    # Parcels given in WGS84 even though zones are metric: mask must still work.
    metric_pts = nr.parcels_from_positions([[500, 100], [500, 215], [500, 300], [5200, 5200]])
    wgs_pts = metric_pts.to_crs(config.CRS_WGS84)

    assert list(nr.parcel_zone_mask(wgs_pts, zones)) == [True, False, False, True]
    assert list(nr.parcel_zone_mask(wgs_pts, zones, buffer_m=30)) == [True, True, False, True]
    assert list(nr.parcel_zone_mask(wgs_pts, zones, buffer_m=120)) == [True, True, True, True]

    with pytest.raises(ValueError, match="projected"):
        nr.buffer_zones(zones.to_crs(config.CRS_WGS84), 30)
    with pytest.raises(ValueError, match="no CRS"):
        nr.parcel_zone_mask(gpd.GeoDataFrame(geometry=[Point(0, 0)]), zones)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        nr.parcels_from_positions(np.zeros((3, 3)))


def test_subset_decision_slices_every_row_array():
    decision = decide_policy([0.9, 0.1, 0.9, np.nan], _one_hot(0, 4))
    sub = nr.subset_decision(decision, np.array([True, False, True, False]))
    assert sub["n"] == 2
    assert sub["flood_proba"].shape == (2,)
    assert sub["deprivation_proba"].shape == (2, 3)
    assert sub["flood_threshold"] == decision["flood_threshold"]
    with pytest.raises(ValueError, match="shape"):
        nr.subset_decision(decision, np.array([True]))


def test_lift_by_settlement_exposes_an_anti_correlated_settlement():
    pos, decision, inside, sid = _two_settlement_scene()
    per = nr.lift_by_settlement(decision, inside, sid, ["kibera", "mathare"])
    assert per["pooled"]["lift"] > 1.0
    assert per["kibera"]["lift"] > 3.0
    assert per["mathare"]["lift"] < 1.0
    assert per["kibera"]["n_reference"] == 100
    assert per["mathare"]["n_model_valid"] == 400
    # Each per-settlement report equals an independent run on that slice.
    solo = validate_against_reference(nr.subset_decision(decision, sid == 1), inside[sid == 1])
    assert per["mathare"]["lift"] == pytest.approx(solo["lift"])


def test_lift_by_buffer_recovers_parcels_just_outside_the_line():
    zones = nr.normalise_zones(_zones_metric())
    # 50 wet parcels sit 10-25 m outside zone 1's northern edge, 50 dry parcels far away.
    rng = np.random.default_rng(1)
    edge = np.column_stack([rng.uniform(10, 990, 50), rng.uniform(210, 225, 50)])
    far = np.column_stack([rng.uniform(10, 990, 50), rng.uniform(3000, 4000, 50)])
    anchors = np.array([[500.0, 100.0]])  # one parcel truly inside so the reference is non-empty
    pos = np.vstack([anchors, edge, far])
    flood = np.concatenate([[0.9], np.full(50, 0.9), np.full(50, 0.1)])
    decision = decide_policy(flood, _one_hot(0, len(pos)))
    parcels = nr.parcels_from_positions(pos)

    by_buffer = nr.lift_by_buffer(decision, parcels, zones, buffers_m=(0, 30, 60))
    assert by_buffer[0.0]["n_reference"] == 1
    assert by_buffer[30.0]["n_reference"] == 51
    assert by_buffer[60.0]["n_reference"] == 51
    assert by_buffer[30.0]["lift"] > by_buffer[0.0]["lift"]
    assert by_buffer[30.0]["meets_lift_target"]
    assert by_buffer[30.0]["inside_mask"].sum() == 51


def test_random_lift_distribution_separates_signal_from_noise():
    pos, decision, inside, sid = _two_settlement_scene()
    kibera = nr.subset_decision(decision, sid == 0)
    null = nr.random_lift_distribution(kibera, inside[sid == 0], n_draws=300, seed=3)
    assert null["n_responsive"] == 110
    assert null["n_reference"] == 100
    assert null["observed_lift"] > null["null_p97_5"]
    assert null["p_value"] < 0.01
    assert abs(null["null_mean"] - 1.0) < 0.15  # a random assignment has lift ~1

    # Shuffled responsiveness: observed lift should sit inside the null band.
    rng = np.random.default_rng(5)
    shuffled = dict(kibera)
    shuffled["model_flood_responsive"] = rng.permutation(kibera["model_flood_responsive"])
    noise = nr.random_lift_distribution(shuffled, inside[sid == 0], n_draws=300, seed=4)
    assert noise["null_p2_5"] <= noise["observed_lift"] <= noise["null_p97_5"]
    assert noise["p_value"] > 0.05

    # Degenerate inputs: nothing responsive -> NaNs, not a crash.
    dry = decide_policy(np.full(10, 0.1), _one_hot(0, 10))
    degenerate = nr.random_lift_distribution(dry, np.ones(10, dtype=bool), n_draws=20)
    assert np.isnan(degenerate["null_mean"]) and np.isnan(degenerate["p_value"])
    with pytest.raises(ValueError, match="shape"):
        nr.random_lift_distribution(dry, np.ones(3, dtype=bool))


def test_rule_lift_matches_agreement_metrics():
    inside = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    near_river = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    metrics = nr.rule_lift(near_river, inside)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["reference_fraction"] == pytest.approx(0.25)
    assert metrics["lift"] == pytest.approx((2 / 3) / 0.25, rel=1e-6)


def test_validation_report_and_formatting():
    zones = nr.normalise_zones(_zones_metric())
    pos, decision, inside_truth, sid = _two_settlement_scene()
    parcels = nr.parcels_from_positions(pos)
    rule = np.zeros(800, dtype=bool)
    rule[:150] = True
    rule[400:450] = True

    report = nr.nrrp_validation_report(
        decision, parcels, zones,
        settlement_index=sid, settlement_keys=["kibera", "mathare"],
        buffers_m=(0, 30, 60), rule_masks={"within 120 m of a river": rule},
        n_draws=100, seed=0,
    )
    assert report["n_zones"] == 2
    assert report["headline_buffer_m"] == 0.0
    assert report["lift"] == pytest.approx(report["by_buffer"][0.0]["lift"])
    assert report["n_reference"] == 200
    assert report["reference_too_small"] is False
    assert np.array_equal(report["by_buffer"][0.0]["inside_mask"], inside_truth)
    assert set(report["per_settlement"]) == {0.0, 30.0, 60.0}
    assert report["per_settlement"][0.0]["mathare"]["lift"] < 1.0 < report["per_settlement"][0.0]["kibera"]["lift"]
    assert report["rules"]["within 120 m of a river"]["lift"] > 1.0
    assert report["null"]["n_draws"] == 100

    text = nr.format_nrrp_report(report)
    assert "Headline lift (0 m buffer)" in text
    assert f"meets={report['meets_lift_target']}" in text
    assert "kibera=" in text and "mathare=" in text
    assert "Permutation null (100 draws" in text
    assert "within 120 m of a river" in text
    assert "WARNING" not in text

    # A reference that barely covers any parcel is flagged, not quoted.
    small = nr.nrrp_validation_report(
        nr.subset_decision(decision, np.arange(800) < 120),
        parcels.iloc[:120].reset_index(drop=True), zones, buffers_m=(0,), n_draws=10,
    )
    assert small["reference_too_small"] is True
    assert "WARNING" in nr.format_nrrp_report(small)
    with pytest.raises(ValueError, match="buffers_m"):
        nr.nrrp_validation_report(decision, parcels, zones, buffers_m=())
