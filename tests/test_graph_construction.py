"""
Unit tests for src/utils/graph_utils.py, targeting the two silent-failure
modes the proposal's testing plan calls out explicitly (Sec 3.4 "Testing"):
disconnected subgraphs and incorrectly thresholded adjacency edges. These
use synthetic GeoDataFrames so they run without network access to OSM.
"""

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
import torch
from shapely.geometry import Point

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import graph_utils
from src import config


def _make_buildings(coords, crs=config.CRS_METRIC):
    """coords: list of (x, y) in metres, e.g. a local projected CRS."""
    gdf = gpd.GeoDataFrame(
        {"node_id": range(len(coords))},
        geometry=[Point(x, y) for x, y in coords],
        crs=crs,
    )
    gdf["centroid"] = gdf.geometry
    gdf["area_m2"] = 20.0
    gdf["dist_to_water_m"] = 10.0
    gdf["dist_to_sanitation_m"] = 10.0
    gdf["dist_to_road_m"] = 5.0
    gdf["roof_material_score"] = 0.5
    return gdf


def test_two_clusters_within_threshold_are_connected():
    """Buildings 10m apart with a 15m threshold should form one connected component."""
    coords = [(0, 0), (10, 0), (20, 0), (30, 0)]
    buildings = _make_buildings(coords)
    graph = graph_utils.build_proximity_graph(buildings, threshold_m=15.0)
    integrity = graph_utils.check_graph_integrity(graph)

    assert integrity["num_connected_components"] == 1
    assert integrity["num_isolated_nodes"] == 0


def test_far_apart_clusters_are_disconnected():
    """Two clusters 200m apart should NOT be linked by a 15m threshold."""
    cluster_a = [(0, 0), (10, 0), (5, 8)]
    cluster_b = [(200, 200), (210, 200), (205, 208)]
    buildings = _make_buildings(cluster_a + cluster_b)
    graph = graph_utils.build_proximity_graph(buildings, threshold_m=15.0)
    integrity = graph_utils.check_graph_integrity(graph)

    assert integrity["num_connected_components"] == 2
    assert integrity["largest_component_frac"] == pytest.approx(0.5)


def test_isolated_building_flagged():
    """A single building far from everything else should show up as isolated."""
    coords = [(0, 0), (10, 0), (5, 8), (500, 500)]
    buildings = _make_buildings(coords)
    graph = graph_utils.build_proximity_graph(buildings, threshold_m=15.0)
    integrity = graph_utils.check_graph_integrity(graph)

    assert integrity["num_isolated_nodes"] == 1


def test_threshold_boundary_is_exclusive_at_exact_distance():
    """
    Buildings exactly at the threshold distance: cKDTree.query_pairs(r=...)
    uses r as an inclusive upper bound, so two buildings exactly 15.0m apart
    SHOULD be connected. This test pins that behaviour explicitly since it's
    the kind of off-by-boundary error the proposal flags as a silent risk.
    """
    coords = [(0, 0), (15.0, 0)]
    buildings = _make_buildings(coords)
    graph = graph_utils.build_proximity_graph(buildings, threshold_m=15.0)
    assert graph.number_of_edges() == 1


def test_no_self_loops():
    coords = [(0, 0), (5, 0), (10, 0)]
    buildings = _make_buildings(coords)
    graph = graph_utils.build_proximity_graph(buildings, threshold_m=15.0)
    assert nx.number_of_selfloops(graph) == 0


def test_pyg_conversion_preserves_node_count():
    coords = [(0, 0), (10, 0), (20, 0)]
    buildings = _make_buildings(coords)
    graph = graph_utils.build_proximity_graph(buildings, threshold_m=15.0)
    data = graph_utils.graph_to_pyg_data(graph)
    assert data.num_nodes == len(coords)
    # area, dist_to_water, dist_to_sanitation, dist_to_road, roof_material_score
    assert data.x.shape[1] == 5


def test_enrichment_attaches_deprivation_labels_to_pyg_data():
    """Varying infra distances must produce a three-class y tensor on the PyG object."""
    coords = [(i * 10, 0) for i in range(12)]
    buildings = _make_buildings(coords)
    buildings["dist_to_water_m"] = np.linspace(5, 400, len(coords))
    buildings["dist_to_sanitation_m"] = np.linspace(5, 400, len(coords))
    buildings["area_m2"] = np.linspace(10, 80, len(coords))
    buildings = graph_utils.enrich_buildings_with_deprivation(buildings)
    assert "deprivation_class" in buildings.columns
    assert set(buildings["deprivation_class"].unique()).issubset({0, 1, 2})

    graph = graph_utils.build_proximity_graph(buildings, threshold_m=15.0)
    data = graph_utils.graph_to_pyg_data(graph)
    assert data.y is not None
    assert data.y.shape == (len(coords),)
    assert data.y.dtype == torch.long


def test_prune_small_components_drops_fringe_isolates():
    cluster = [(i * 10, 0) for i in range(25)]
    fringe = [(1000, 1000), (1010, 1000)]
    buildings = _make_buildings(cluster + fringe)
    graph = graph_utils.build_proximity_graph(buildings, threshold_m=15.0)
    filtered_buildings, filtered_graph, report = graph_utils.prune_small_components(
        buildings, graph, min_component_size=5,
    )
    assert report["nodes_after"] == 25
    assert report["nodes_dropped"] == 2
    assert filtered_graph.number_of_nodes() == 25
    assert len(filtered_buildings) == 25


def test_empty_buildings_gdf_produces_empty_graph():
    buildings = _make_buildings([])
    graph = graph_utils.build_proximity_graph(buildings, threshold_m=15.0)
    integrity = graph_utils.check_graph_integrity(graph)
    assert integrity["num_nodes"] == 0
    assert integrity["largest_component_frac"] == 0.0


def test_overpass_retry_cycles_mirrors(monkeypatch):
    """Transient Overpass failures should rotate mirrors then succeed."""
    calls = {"n": 0, "urls": []}

    def fake_features(polygon, tags=None):
        calls["n"] += 1
        calls["urls"].append(ox.settings.overpass_url)
        if calls["n"] < 3:
            raise RuntimeError("502 Bad Gateway (simulated)")
        return gpd.GeoDataFrame(
            {"geometry": [Point(0, 0).buffer(1)], "building": ["yes"]},
            crs=config.CRS_WGS84,
        )

    import osmnx as ox
    monkeypatch.setattr(ox, "features_from_polygon", fake_features)
    monkeypatch.setattr(config, "OVERPASS_MAX_ATTEMPTS", 4)
    monkeypatch.setattr(config, "OVERPASS_RETRY_BASE_SLEEP_S", 0.01)
    monkeypatch.setattr(
        config, "OVERPASS_URLS",
        ("https://mirror-a.example/api", "https://mirror-b.example/api"),
    )

    from shapely.geometry import box
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=config.CRS_WGS84)
    # fetch_osm_buildings filters to Polygon geometries -- Point.buffer gives Polygon.
    out = graph_utils.fetch_osm_buildings(boundary)
    assert len(out) == 1
    assert calls["n"] == 3
    assert calls["urls"][0] != calls["urls"][1] or len(config.OVERPASS_URLS) == 1
