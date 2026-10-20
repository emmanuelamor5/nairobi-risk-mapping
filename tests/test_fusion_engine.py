"""
Offline unit tests for Sprint 5 Late-Fusion Decision Engine, spatial tile
stitching / parcel sampling, GeoJSON serialisation, and NRRP policy validation
(src/models/fusion_engine.py).

Uses only synthetic arrays, graphs, and geometries -- no Google Earth Engine,
Overpass network calls, GPU, or Google Drive required.
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
import torch
from affine import Affine
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.models.attention_unet_model import AttentionUNetModel
from src.models.dataset import DEFAULT_LAYOUT, EVENT_S1_LAYOUT
from src.models.fusion_engine import (
    DEFAULT_POLICY_MATRIX,
    FusionEngine,
    aggregate_multi_settlement_fusion_reports,
    build_parcel_policy_geodataframe,
    discretize_flood_probability,
    export_settlement_policy_geojson,
    extract_parcel_flood_probabilities,
    flag_parcels_in_reference_zone,
    flood_regime_membership,
    predict_settlement_flood_map,
    validate_policy_against_nrrp,
)
from src.models.graphsage_model import GraphSAGEModel
from src.models.multi_settlement import combine_settlement_graphs
from src.utils import graph_utils


# ---------------------------------------------------------------------------
# 1. Policy matrix & flood discretisation tests
# ---------------------------------------------------------------------------


def test_discretize_flood_probability_regimes():
    probs = np.array([0.0, 0.15, 0.299, 0.30, 0.45, 0.599, 0.60, 0.85, 1.0])
    levels = discretize_flood_probability(probs, low_threshold=0.30, high_threshold=0.60)
    assert levels.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_discretize_flood_probability_invalid_thresholds():
    with pytest.raises(ValueError, match="low_threshold < high_threshold"):
        discretize_flood_probability(np.array([0.5]), low_threshold=0.7, high_threshold=0.3)


def test_flood_regime_membership_rows_sum_to_one():
    probs = np.linspace(0.0, 1.0, 21)
    mem = flood_regime_membership(probs, low_threshold=0.30, high_threshold=0.60)
    assert mem.shape == (21, 3)
    assert np.allclose(mem.sum(axis=1), 1.0)
    assert np.all(mem >= 0.0) and np.all(mem <= 1.0)
    # Winning regime in soft membership matches hard discretised regime away from boundaries.
    interior = np.array([0.05, 0.45, 0.85])
    assert flood_regime_membership(interior).argmax(axis=1).tolist() == [0, 1, 2]


def test_fusion_engine_all_nine_matrix_cells():
    """
    Verify all 9 combinations of (flood_level in {0,1,2}) x (deprivation_class in {0,1,2})
    against DEFAULT_POLICY_MATRIX:
      - low flood (0.10):      low->0 (Formalise), med->1 (Upgrade), high->1 (Upgrade)
      - moderate flood (0.45): low->1 (Upgrade),   med->1 (Upgrade), high->2 (Resettlement)
      - high flood (0.80):     low->2,             med->2,           high->2 (Resettlement)
    """
    engine = FusionEngine(flood_low_threshold=0.30, flood_high_threshold=0.60)

    flood_probs = np.array([
        0.10, 0.10, 0.10,
        0.45, 0.45, 0.45,
        0.80, 0.80, 0.80,
    ])
    dep_probs = np.array([
        [0.80, 0.15, 0.05],  # low
        [0.10, 0.80, 0.10],  # medium
        [0.05, 0.15, 0.80],  # high
        [0.80, 0.15, 0.05],  # low
        [0.10, 0.80, 0.10],  # medium
        [0.05, 0.15, 0.80],  # high
        [0.80, 0.15, 0.05],  # low
        [0.10, 0.80, 0.10],  # medium
        [0.05, 0.15, 0.80],  # high
    ])

    res = engine.classify_parcels(flood_probs, dep_probs)
    expected_classes = [
        0, 1, 1,
        1, 1, 2,
        2, 2, 2,
    ]
    assert res["policy_class"].tolist() == expected_classes
    assert res["flood_level"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert res["deprivation_class"].tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert res["fused_probabilities"].shape == (9, 4)
    assert res["policy_probs"].shape == (9, 3)
    assert np.allclose(res["policy_probs"].sum(axis=1), 1.0)
    assert np.all((res["confidence"] > 0.0) & (res["confidence"] <= 1.0))
    assert res["policy_label"][0] == "Retain and Formalise"
    assert res["policy_label"][1] == "Retain and Upgrade"
    assert res["policy_label"][5] == "Compensated Resettlement"
    assert res["policy_color"][0] == config.POLICY_CLASS_COLORS[0]
    assert res["policy_color"][5] == config.POLICY_CLASS_COLORS[2]


def test_fusion_engine_accepts_torch_tensors_and_2d_flood_probs():
    engine = FusionEngine()
    fp = torch.tensor([[0.05], [0.50], [0.90]], dtype=torch.float32)
    dp = torch.tensor([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
    ], dtype=torch.float32)
    res = engine.classify_parcels(fp, dp)
    assert res["policy_class"].tolist() == [0, 1, 2]


def test_fusion_engine_custom_thresholds_and_matrix():
    # Custom strict matrix where moderate flood + medium deprivation also triggers resettlement
    custom_matrix = (
        (0, 0, 1),
        (1, 2, 2),
        (2, 2, 2),
    )
    engine = FusionEngine(
        flood_low_threshold=0.20,
        flood_high_threshold=0.50,
        policy_matrix=custom_matrix,
    )
    res = engine.classify_parcels(
        flood_probs=np.array([0.15, 0.35]),
        deprivation_probs=np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]),
    )
    assert res["policy_class"].tolist() == [0, 2]


def test_fusion_engine_validates_inputs():
    engine = FusionEngine()
    with pytest.raises(ValueError, match="Parcel count mismatch"):
        engine.classify_parcels(np.array([0.1, 0.2]), np.ones((3, 3)) / 3)
    with pytest.raises(ValueError, match="deprivation_probs must have shape"):
        engine.classify_parcels(np.array([0.1, 0.2]), np.ones((2, 2)) / 2)
    with pytest.raises(ValueError, match="policy_matrix must have shape"):
        FusionEngine(policy_matrix=[[0, 1], [1, 2]])


def test_summarize_classifications_counts_and_contingency():
    engine = FusionEngine()
    fp = np.array([0.05, 0.10, 0.45, 0.85])
    dp = np.array([
        [0.9, 0.05, 0.05],  # Formalise (0)
        [0.1, 0.8, 0.1],    # Upgrade (1)
        [0.1, 0.8, 0.1],    # Upgrade (1)
        [0.1, 0.1, 0.8],    # Resettlement (2)
    ])
    res = engine.classify_parcels(fp, dp)
    summary = engine.summarize_classifications(res)
    assert summary["total_parcels"] == 4
    assert summary["counts"]["Retain and Formalise"] == 1
    assert summary["counts"]["Retain and Upgrade"] == 2
    assert summary["counts"]["Compensated Resettlement"] == 1
    assert np.isclose(sum(summary["fractions"].values()), 1.0)
    cm = np.asarray(summary["flood_vs_deprivation_counts"])
    assert cm.sum() == 4


# ---------------------------------------------------------------------------
# 2. Spatial alignment: tile stitching & parcel flood sampling
# ---------------------------------------------------------------------------


def test_predict_settlement_flood_map_stitches_overlapping_tiles(tmp_path):
    """
    Two overlapping 256x256 tiles in a 256x462 raster (stride 206, overlap 50)
    should stitch into a full coverage map with count=2 in the 50px shared strip
    and count=1 elsewhere, while ignoring augmented variants when _orig exists.
    """
    n_bands = len(EVENT_S1_LAYOUT.band_names)
    t1 = np.zeros((n_bands, 256, 256), dtype=np.float32)
    t2 = np.ones((n_bands, 256, 256), dtype=np.float32) * 0.5
    p1 = tmp_path / "kibera_0_0_orig.npy"
    p1_aug = tmp_path / "kibera_0_0_flip_h.npy"
    p2 = tmp_path / "kibera_0_206_orig.npy"
    np.save(p1, t1)
    np.save(p1_aug, t1)
    np.save(p2, t2)

    manifest = [
        {"tile_id": "kibera_0_0_orig", "path": str(p1), "row_off": 0, "col_off": 0, "settlement_key": "kibera"},
        {"tile_id": "kibera_0_0_flip_h", "path": str(p1_aug), "row_off": 0, "col_off": 0, "settlement_key": "kibera"},
        {"tile_id": "kibera_0_206_orig", "path": str(p2), "row_off": 0, "col_off": 206, "settlement_key": "kibera"},
    ]

    unet = AttentionUNetModel(base_channels=8, layout=EVENT_S1_LAYOUT)
    stitched = predict_settlement_flood_map(
        unet, manifest, raster_shape=(256, 462), layout=EVENT_S1_LAYOUT,
    )
    assert stitched["prob_map"].shape == (256, 462)
    assert stitched["coverage_mask"].all()
    # Non-overlapping left strip (cols 0..205): 1 tile
    assert np.all(stitched["count_map"][:, :206] == 1)
    # 50px overlap strip (cols 206..255): 2 tiles (augmented copy was ignored)
    assert np.all(stitched["count_map"][:, 206:256] == 2)
    # Non-overlapping right strip (cols 256..461): 1 tile
    assert np.all(stitched["count_map"][:, 256:] == 1)


def test_extract_parcel_flood_probabilities_from_gdf_and_metric_pos():
    # 10x10 grid in EPSG:32737, pixel size 10m, origin at (255000, 9855100) going south
    transform = Affine(10.0, 0.0, 255000.0, 0.0, -10.0, 9855100.0)
    prob_map = np.zeros((10, 10), dtype=np.float32)
    prob_map[0, 0] = 0.85   # top-left pixel: x in [255000, 255010], y in [9855090, 9855100]
    prob_map[5, 5] = 0.15   # middle pixel: x in [255050, 255060], y in [9855040, 9855050]

    coords = np.array([
        [255005.0, 9855095.0],  # inside pixel (0, 0) -> 0.85
        [255055.0, 9855045.0],  # inside pixel (5, 5) -> 0.15
        [260000.0, 9860000.0],  # out of bounds -> default_prob (0.0)
    ])
    sampled_pos = extract_parcel_flood_probabilities(
        prob_map, transform, raster_crs=config.CRS_METRIC,
        node_positions_metric=coords, default_prob=0.0,
    )
    assert np.allclose(sampled_pos, [0.85, 0.15, 0.0], atol=1e-5)

    # Same locations passed as a GeoDataFrame in WGS84 (tests CRS reprojection)
    gdf = gpd.GeoDataFrame(
        {"node_id": [0, 1, 2]},
        geometry=[Point(x, y).buffer(2.0) for x, y in coords],
        crs=config.CRS_METRIC,
    ).to_crs(config.CRS_WGS84)
    sampled_gdf = extract_parcel_flood_probabilities(
        prob_map, transform, raster_crs=config.CRS_METRIC,
        buildings_gdf=gdf, default_prob=0.0,
    )
    assert np.allclose(sampled_gdf, [0.85, 0.15, 0.0], atol=1e-5)


def test_graph_to_pyg_data_preserves_pos_and_combine_concatenates_pos():
    """
    Verify that graph_utils.graph_to_pyg_data stores building metric coordinates
    on data.pos without colliding with data.y (deprivation class labels), and
    that combine_settlement_graphs preserves data.pos across settlements.
    """
    coords = [(255000.0 + i * 10.0, 9855000.0) for i in range(6)]
    buildings = gpd.GeoDataFrame(
        {"node_id": range(len(coords))},
        geometry=[Point(x, y).buffer(3.0) for x, y in coords],
        crs=config.CRS_METRIC,
    )
    buildings["centroid"] = buildings.geometry.centroid
    buildings["area_m2"] = np.linspace(15.0, 60.0, len(coords))
    buildings["dist_to_water_m"] = np.linspace(10.0, 300.0, len(coords))
    buildings["dist_to_sanitation_m"] = np.linspace(10.0, 300.0, len(coords))
    buildings["dist_to_road_m"] = np.linspace(5.0, 80.0, len(coords))
    buildings = graph_utils.enrich_buildings_with_deprivation(buildings)

    graph = graph_utils.build_proximity_graph(buildings, threshold_m=15.0)
    data = graph_utils.graph_to_pyg_data(graph)

    assert hasattr(data, "pos") and data.pos is not None
    assert data.pos.shape == (6, 2)
    assert np.allclose(data.pos[:, 0].numpy(), [c[0] for c in coords])
    assert np.allclose(data.pos[:, 1].numpy(), [c[1] for c in coords])
    # data.y is still the integer deprivation class tensor in {0, 1, 2}, NOT the y coordinate!
    assert data.y.dtype == torch.long
    assert set(data.y.tolist()).issubset({0, 1, 2})

    combined = combine_settlement_graphs({"kibera": data, "mathare": data.clone()})
    assert hasattr(combined, "pos") and combined.pos.shape == (12, 2)


# ---------------------------------------------------------------------------
# 3. NRRP policy validation & multi-settlement aggregation
# ---------------------------------------------------------------------------


def test_validate_policy_against_nrrp_lift_and_blanket_eviction_metric():
    """
    When Compensated Resettlement (class 2) concentrates inside the NRRP zone,
    resettlement_lift > 1.0 and meets_lift_target is True. Parcels inside the
    NRRP zone that are assigned class 0 or 1 count toward blanket_eviction_avoided.
    """
    # 100 parcels: first 25 are inside the NRRP riparian zone, remaining 75 outside.
    in_nrrp = np.zeros(100, dtype=bool)
    in_nrrp[:25] = True

    policy_class = np.zeros(100, dtype=np.int64)
    # Inside NRRP (25 parcels): 15 Resettlement (2), 8 Upgrade (1), 2 Formalise (0)
    policy_class[:15] = 2
    policy_class[15:23] = 1
    policy_class[23:25] = 0
    # Outside NRRP (75 parcels): 5 Resettlement (2), 30 Upgrade (1), 40 Formalise (0)
    policy_class[25:30] = 2
    policy_class[30:60] = 1
    policy_class[60:] = 0

    flood_prob = np.where(in_nrrp, 0.65, 0.15)
    dep_class = np.where(policy_class == 0, 0, 1)

    report = validate_policy_against_nrrp(
        policy_class=policy_class,
        in_nrrp_zone=in_nrrp,
        flood_prob=flood_prob,
        deprivation_class=dep_class,
    )
    # 15 of 20 total resettlement parcels are in NRRP (precision = 0.75 vs base rate 0.25 -> lift = 3.0)
    assert report["resettlement_lift"] == pytest.approx(0.75 / 0.25, rel=1e-4)
    assert report["meets_lift_target"] is True
    # 10 of 25 NRRP parcels are retained (8 Upgrade + 2 Formalise) -> 40% blanket eviction avoided
    assert report["retained_in_nrrp_count"] == 10
    assert report["blanket_eviction_avoided_fraction"] == pytest.approx(10 / 25)
    assert report["mean_flood_prob_in_nrrp"] > report["mean_flood_prob_outside_nrrp"]


def test_flag_parcels_in_reference_zone():
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)
    ref_mask = np.zeros((10, 10), dtype=np.uint8)
    ref_mask[:, :3] = 1  # left 3 columns (x in [0, 30]) are inside reference zone

    coords = np.array([
        [15.0, 50.0],  # col 1 -> inside
        [75.0, 50.0],  # col 7 -> outside
    ])
    flags = flag_parcels_in_reference_zone(
        ref_mask, transform, raster_crs=config.CRS_METRIC,
        node_positions_metric=coords,
    )
    assert flags.tolist() == [True, False]


def test_build_and_export_parcel_policy_geodataframe(tmp_path):
    engine = FusionEngine()
    fp = np.array([0.10, 0.45, 0.80])
    dp = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
    ])
    cls = engine.classify_parcels(fp, dp)
    coords = np.array([
        [253000.0, 9854800.0],
        [253020.0, 9854800.0],
        [253040.0, 9854800.0],
    ])
    gdf = build_parcel_policy_geodataframe(
        settlement_key="kibera",
        classification_result=cls,
        node_positions_metric=coords,
        in_nrrp_zone=np.array([False, False, True]),
    )
    assert len(gdf) == 3
    assert str(gdf.crs) == config.CRS_WGS84
    assert list(gdf["policy_category"]) == [
        "Retain and Formalise",
        "Retain and Upgrade",
        "Compensated Resettlement",
    ]
    assert list(gdf["in_nrrp_zone"]) == [False, False, True]

    out_path = export_settlement_policy_geojson(gdf, "kibera", output_dir=tmp_path)
    assert out_path.exists()
    reloaded = gpd.read_file(out_path)
    assert len(reloaded) == 3
    assert set(reloaded["policy_category"]) == set(config.POLICY_CLASS_NAMES)


def test_end_to_end_two_head_fusion_and_multi_settlement_aggregation():
    """
    Integration test (proposal Sec 3.4 Testing): run forward inference on
    synthetic raster + graph inputs through AttentionUNetModel and GraphSAGEModel,
    pass both probability distributions into FusionEngine across all three
    settlements, and aggregate pooled + per-settlement NRRP validation reports.
    """
    engine = FusionEngine()
    gs_model = GraphSAGEModel(in_channels=5, hidden_channels=16, num_classes=3)

    per_settlement = {}
    rng = np.random.default_rng(42)

    for idx, key in enumerate(config.SETTLEMENT_KEYS):
        n_nodes = 40 + idx * 10
        x = torch.from_numpy(rng.normal(size=(n_nodes, 5)).astype(np.float32))
        src = list(range(n_nodes))
        dst = [(i + 1) % n_nodes for i in range(n_nodes)]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        y = torch.from_numpy(rng.integers(0, 3, size=n_nodes, dtype=np.int64))
        data = graph_utils.Data(x=x, edge_index=edge_index, y=y)

        dep_probs = gs_model.predict_proba(data).numpy()
        # Simulate riparian zone on first 25% of nodes with elevated flood probability
        in_nrrp = np.zeros(n_nodes, dtype=bool)
        in_nrrp[: n_nodes // 4] = True
        flood_probs = np.where(
            in_nrrp,
            rng.uniform(0.62, 0.95, size=n_nodes),
            rng.uniform(0.02, 0.35, size=n_nodes),
        )

        cls = engine.classify_parcels(flood_probs, dep_probs)
        per_settlement[key] = {
            "classification": cls,
            "in_nrrp_zone": in_nrrp,
        }

    agg = aggregate_multi_settlement_fusion_reports(per_settlement, engine=engine)
    assert set(agg["per_settlement_summary"]) == {"kibera", "mathare", "mukuru"}
    assert agg["pooled_summary"]["total_parcels"] == 40 + 50 + 60
    assert agg["pooled_nrrp"] is not None
    assert agg["pooled_nrrp"]["resettlement_lift"] > 1.0
    assert agg["meets_pooled_lift_target"] is True
    assert agg["all_settlements_meet_lift_target"] is True
