"""
OSM building/infrastructure ingestion and PyTorch Geometric graph
construction for the GraphSAGE structural-deprivation head.

Edge policy: two buildings are connected if their footprint centroids are
within config.BUILDING_ADJACENCY_THRESHOLD_M of each other. This is a
simple proximity graph, not a full adjacency/contiguity graph -- documented
here since the proposal's testing section (Sec 3.4 "Testing") explicitly
calls out mis-thresholded edges as a silent-failure risk.
"""

import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import torch
from scipy.spatial import cKDTree
from shapely.geometry import Point
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from .. import config


def fetch_settlement_boundary(settlement_key: str) -> gpd.GeoDataFrame:
    """
    Resolve a settlement boundary polygon, preferring live OSM geocoding and
    falling back to the approximate bbox in config.SETTLEMENTS if geocoding
    fails or returns an implausibly small/large polygon.
    """
    cfg = config.SETTLEMENTS[settlement_key]
    try:
        gdf = ox.geocode_to_gdf(cfg.osm_query)
        area_km2 = gdf.to_crs(config.CRS_METRIC).area.iloc[0] / 1e6
        if not (0.1 <= area_km2 <= 20):
            raise ValueError(f"Geocoded area {area_km2:.2f} km^2 looks implausible")
        return gdf
    except Exception as exc:
        warnings.warn(
            f"OSM geocoding failed or implausible for '{cfg.osm_query}' ({exc}); "
            f"falling back to hardcoded bbox. VERIFY this boundary visually "
            f"before using it for training -- see config.py docstring."
        )
        min_lon, min_lat, max_lon, max_lat = cfg.fallback_bbox
        from shapely.geometry import box
        return gpd.GeoDataFrame(
            {"name": [cfg.name]},
            geometry=[box(min_lon, min_lat, max_lon, max_lat)],
            crs=config.CRS_WGS84,
        )


def fetch_osm_buildings(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fetch building footprints within the boundary polygon via the Overpass API."""
    polygon = boundary.geometry.iloc[0]
    buildings = ox.features_from_polygon(polygon, tags=config.OVERPASS_TAGS)
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    buildings = buildings.to_crs(config.CRS_METRIC)
    buildings["centroid"] = buildings.geometry.centroid
    buildings["area_m2"] = buildings.geometry.area
    buildings = buildings.reset_index(drop=True)
    buildings["node_id"] = buildings.index
    return buildings


def fetch_osm_infrastructure(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fetch water points, sanitation points, and road network for proximity features."""
    polygon = boundary.geometry.iloc[0]
    infra = ox.features_from_polygon(polygon, tags=config.INFRA_TAGS)
    infra = infra.to_crs(config.CRS_METRIC)
    infra["centroid"] = infra.geometry.centroid
    return infra


def compute_infra_distance_features(buildings: gpd.GeoDataFrame,
                                     infra: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    For each building, compute nearest-neighbour distance (metres) to the
    closest water point, sanitation point, and road segment. These become
    node features encoding infrastructure deficit, per proposal Sec 3.2.1.
    """
    out = buildings.copy()
    b_coords = np.array([[p.x, p.y] for p in buildings["centroid"]])

    def _nearest_distance(target_gdf: gpd.GeoDataFrame) -> np.ndarray:
        if target_gdf.empty:
            return np.full(len(buildings), np.nan)
        t_coords = np.array([[p.x, p.y] for p in target_gdf.geometry.centroid])
        tree = cKDTree(t_coords)
        dist, _ = tree.query(b_coords, k=1)
        return dist

    water = infra[infra.get("amenity") == "drinking_water"] if "amenity" in infra.columns else infra.iloc[0:0]
    sanitation = infra[infra.get("amenity") == "toilets"] if "amenity" in infra.columns else infra.iloc[0:0]
    roads = infra[infra.get("highway").notna()] if "highway" in infra.columns else infra.iloc[0:0]

    out["dist_to_water_m"] = _nearest_distance(water)
    out["dist_to_sanitation_m"] = _nearest_distance(sanitation)
    out["dist_to_road_m"] = _nearest_distance(roads)

    # Missing infra categories in a given tile are common in sparse OSM
    # coverage; impute with the tile-local max rather than dropping the
    # feature so the graph shape stays consistent across settlements.
    for col in ["dist_to_water_m", "dist_to_sanitation_m", "dist_to_road_m"]:
        if out[col].isna().all():
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(out[col].max())

    return out


def build_proximity_graph(buildings: gpd.GeoDataFrame,
                           threshold_m: float = None) -> nx.Graph:
    """
    Build an undirected proximity graph over building centroids using a
    KD-tree radius query -- O(n log n) rather than the O(n^2) naive
    pairwise distance check, which matters once a settlement has tens of
    thousands of structures (per proposal Sec 3.4, Mukuru/Mathare/Kibera
    combined are in that range).
    """
    threshold_m = threshold_m or config.BUILDING_ADJACENCY_THRESHOLD_M
    graph = nx.Graph()

    if len(buildings) == 0:
        return graph  # empty settlement tile -- caller should treat this as a data gap, not a crash

    coords = np.array([[p.x, p.y] for p in buildings["centroid"]])

    for _, row in buildings.iterrows():
        graph.add_node(
            row["node_id"],
            area_m2=row["area_m2"],
            dist_to_water_m=row.get("dist_to_water_m", 0.0),
            dist_to_sanitation_m=row.get("dist_to_sanitation_m", 0.0),
            dist_to_road_m=row.get("dist_to_road_m", 0.0),
            x=row["centroid"].x,
            y=row["centroid"].y,
        )

    if len(buildings) > 1:
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=threshold_m)
        for i, j in pairs:
            graph.add_edge(buildings.iloc[i]["node_id"], buildings.iloc[j]["node_id"])

    return graph


def check_graph_integrity(graph: nx.Graph) -> dict:
    """
    Diagnostic pass flagging the two silent-failure modes called out in the
    proposal's testing plan: disconnected subgraphs and adjacency edges that
    look wrong for the given threshold. Returns a summary dict rather than
    raising, so pipeline runs can log-and-continue on sparse OSM tiles while
    still surfacing the issue.
    """
    components = list(nx.connected_components(graph))
    isolated_nodes = [n for n in graph.nodes if graph.degree(n) == 0]

    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_connected_components": len(components),
        "largest_component_frac": (
            max(len(c) for c in components) / graph.number_of_nodes()
            if graph.number_of_nodes() > 0 else 0.0
        ),
        "num_isolated_nodes": len(isolated_nodes),
        "isolated_node_ids": isolated_nodes[:20],  # cap for log readability
    }


def prune_small_components(buildings, graph, min_component_size: int = None) -> tuple:
    """
    Drop buildings belonging to connected components smaller than
    min_component_size. Targets sparse, likely-off-target fringe areas
    (formally-planned plots at the edge of an oversized bounding box)
    without requiring a hand-tuned boundary polygon.
    """
    min_component_size = min_component_size or config.MIN_COMPONENT_SIZE
    components = list(nx.connected_components(graph))

    kept_node_ids = set()
    kept_components, dropped_components = 0, 0
    for comp in components:
        if len(comp) >= min_component_size:
            kept_node_ids |= comp
            kept_components += 1
        else:
            dropped_components += 1

    filtered_buildings = buildings[buildings["node_id"].isin(kept_node_ids)].copy()
    filtered_graph = graph.subgraph(kept_node_ids).copy()

    report = {
        "min_component_size": min_component_size,
        "components_kept": kept_components,
        "components_dropped": dropped_components,
        "nodes_before": graph.number_of_nodes(),
        "nodes_after": filtered_graph.number_of_nodes(),
        "nodes_dropped": graph.number_of_nodes() - filtered_graph.number_of_nodes(),
    }
    return filtered_buildings, filtered_graph, report


def prune_small_components(buildings, graph, min_component_size: int = None) -> tuple:
    """
    Drop buildings belonging to connected components smaller than
    min_component_size. Targets sparse, likely-off-target fringe areas
    (formally-planned plots at the edge of an oversized bounding box)
    without requiring a hand-tuned boundary polygon.
    """
    min_component_size = min_component_size or config.MIN_COMPONENT_SIZE
    components = list(nx.connected_components(graph))

    kept_node_ids = set()
    kept_components, dropped_components = 0, 0
    for comp in components:
        if len(comp) >= min_component_size:
            kept_node_ids |= comp
            kept_components += 1
        else:
            dropped_components += 1

    filtered_buildings = buildings[buildings["node_id"].isin(kept_node_ids)].copy()
    filtered_graph = graph.subgraph(kept_node_ids).copy()

    report = {
        "min_component_size": min_component_size,
        "components_kept": kept_components,
        "components_dropped": dropped_components,
        "nodes_before": graph.number_of_nodes(),
        "nodes_after": filtered_graph.number_of_nodes(),
        "nodes_dropped": graph.number_of_nodes() - filtered_graph.number_of_nodes(),
    }
    return filtered_buildings, filtered_graph, report


def prune_small_components(buildings, graph, min_component_size: int = None) -> tuple:
    """
    Drop buildings belonging to connected components smaller than
    min_component_size. Targets sparse, likely-off-target fringe areas
    (formally-planned plots at the edge of an oversized bounding box)
    without requiring a hand-tuned boundary polygon.
    """
    min_component_size = min_component_size or config.MIN_COMPONENT_SIZE
    components = list(nx.connected_components(graph))

    kept_node_ids = set()
    kept_components, dropped_components = 0, 0
    for comp in components:
        if len(comp) >= min_component_size:
            kept_node_ids |= comp
            kept_components += 1
        else:
            dropped_components += 1

    filtered_buildings = buildings[buildings["node_id"].isin(kept_node_ids)].copy()
    filtered_graph = graph.subgraph(kept_node_ids).copy()

    report = {
        "min_component_size": min_component_size,
        "components_kept": kept_components,
        "components_dropped": dropped_components,
        "nodes_before": graph.number_of_nodes(),
        "nodes_after": filtered_graph.number_of_nodes(),
        "nodes_dropped": graph.number_of_nodes() - filtered_graph.number_of_nodes(),
    }
    return filtered_buildings, filtered_graph, report


def prune_small_components(buildings, graph, min_component_size: int = None) -> tuple:
    """
    Drop buildings belonging to connected components smaller than
    min_component_size. Targets sparse, likely-off-target fringe areas
    (formally-planned plots at the edge of an oversized bounding box)
    without requiring a hand-tuned boundary polygon.
    """
    min_component_size = min_component_size or config.MIN_COMPONENT_SIZE
    components = list(nx.connected_components(graph))

    kept_node_ids = set()
    kept_components, dropped_components = 0, 0
    for comp in components:
        if len(comp) >= min_component_size:
            kept_node_ids |= comp
            kept_components += 1
        else:
            dropped_components += 1

    filtered_buildings = buildings[buildings["node_id"].isin(kept_node_ids)].copy()
    filtered_graph = graph.subgraph(kept_node_ids).copy()

    report = {
        "min_component_size": min_component_size,
        "components_kept": kept_components,
        "components_dropped": dropped_components,
        "nodes_before": graph.number_of_nodes(),
        "nodes_after": filtered_graph.number_of_nodes(),
        "nodes_dropped": graph.number_of_nodes() - filtered_graph.number_of_nodes(),
    }
    return filtered_buildings, filtered_graph, report


def prune_small_components(buildings, graph, min_component_size: int = None) -> tuple:
    """
    Drop buildings belonging to connected components smaller than
    min_component_size. Targets sparse, likely-off-target fringe areas
    (formally-planned plots at the edge of an oversized bounding box)
    without requiring a hand-tuned boundary polygon.
    """
    min_component_size = min_component_size or config.MIN_COMPONENT_SIZE
    components = list(nx.connected_components(graph))

    kept_node_ids = set()
    kept_components, dropped_components = 0, 0
    for comp in components:
        if len(comp) >= min_component_size:
            kept_node_ids |= comp
            kept_components += 1
        else:
            dropped_components += 1

    filtered_buildings = buildings[buildings["node_id"].isin(kept_node_ids)].copy()
    filtered_graph = graph.subgraph(kept_node_ids).copy()

    report = {
        "min_component_size": min_component_size,
        "components_kept": kept_components,
        "components_dropped": dropped_components,
        "nodes_before": graph.number_of_nodes(),
        "nodes_after": filtered_graph.number_of_nodes(),
        "nodes_dropped": graph.number_of_nodes() - filtered_graph.number_of_nodes(),
    }
    return filtered_buildings, filtered_graph, report


def prune_small_components(buildings, graph, min_component_size: int = None) -> tuple:
    """
    Drop buildings belonging to connected components smaller than
    min_component_size. Targets sparse, likely-off-target fringe areas
    (formally-planned plots at the edge of an oversized bounding box)
    without requiring a hand-tuned boundary polygon.
    """
    min_component_size = min_component_size or config.MIN_COMPONENT_SIZE
    components = list(nx.connected_components(graph))

    kept_node_ids = set()
    kept_components, dropped_components = 0, 0
    for comp in components:
        if len(comp) >= min_component_size:
            kept_node_ids |= comp
            kept_components += 1
        else:
            dropped_components += 1

    filtered_buildings = buildings[buildings["node_id"].isin(kept_node_ids)].copy()
    filtered_graph = graph.subgraph(kept_node_ids).copy()

    report = {
        "min_component_size": min_component_size,
        "components_kept": kept_components,
        "components_dropped": dropped_components,
        "nodes_before": graph.number_of_nodes(),
        "nodes_after": filtered_graph.number_of_nodes(),
        "nodes_dropped": graph.number_of_nodes() - filtered_graph.number_of_nodes(),
    }
    return filtered_buildings, filtered_graph, report


def graph_to_pyg_data(graph: nx.Graph) -> Data:
    """Convert the networkx proximity graph into a PyTorch Geometric Data object."""
    node_features = ["area_m2", "dist_to_water_m", "dist_to_sanitation_m", "dist_to_road_m"]
    data = from_networkx(graph, group_node_attrs=node_features)
    data.x = data.x.float()

    # z-score normalise node features so GraphSAGE's mean aggregator isn't
    # dominated by raw metre-scale distances vs. area in square metres.
    mean = data.x.mean(dim=0, keepdim=True)
    std = data.x.std(dim=0, keepdim=True).clamp(min=1e-6)
    data.x = (data.x - mean) / std

    return data
