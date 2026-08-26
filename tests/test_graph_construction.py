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
    assert data.x.shape[1] == 4  # area, dist_to_water, dist_to_sanitation, dist_to_road


def test_empty_buildings_gdf_produces_empty_graph():
    buildings = _make_buildings([])
    graph = graph_utils.build_proximity_graph(buildings, threshold_m=15.0)
    integrity = graph_utils.check_graph_integrity(graph)
    assert integrity["num_nodes"] == 0
    assert integrity["largest_component_frac"] == 0.0
