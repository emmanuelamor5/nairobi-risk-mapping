"""
OSM building/infrastructure ingestion and PyTorch Geometric graph
construction for the GraphSAGE structural-deprivation head.

Edge policy: two buildings are connected if their footprint centroids are
within config.BUILDING_ADJACENCY_THRESHOLD_M of each other. This is a
simple proximity graph, not a full adjacency/contiguity graph -- documented
here since the proposal's testing section (Sec 3.4 "Testing") explicitly
calls out mis-thresholded edges as a silent-failure risk.
"""

import logging
import time
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
from .deprivation_label import (
    NODE_FEATURE_NAMES,
    assign_deprivation_classes,
    encode_roof_material,
    score_deprivation_indicators,
)

logger = logging.getLogger(__name__)


def _features_from_polygon_with_retry(polygon, tags) -> gpd.GeoDataFrame:
    """
    Call ``ox.features_from_polygon`` with exponential backoff across the
    configured Overpass mirrors. Public Overpass endpoints frequently return
    502 / empty bodies when at capacity; cycling mirrors + sleeping is the
    recommended recovery (and far cheaper than failing the whole Sprint 3 run).
    """
    urls = list(getattr(config, "OVERPASS_URLS", ("https://overpass-api.de/api",)))
    max_attempts = int(getattr(config, "OVERPASS_MAX_ATTEMPTS", 6))
    base_sleep = float(getattr(config, "OVERPASS_RETRY_BASE_SLEEP_S", 5.0))
    last_exc = None

    for attempt in range(max_attempts):
        url = urls[attempt % len(urls)]
        ox.settings.overpass_url = url
        ox.settings.requests_timeout = max(getattr(ox.settings, "requests_timeout", 180), 180)
        try:
            logger.info(
                "Overpass attempt %d/%d via %s", attempt + 1, max_attempts, url,
            )
            return ox.features_from_polygon(polygon, tags=tags)
        except Exception as exc:
            last_exc = exc
            sleep_s = base_sleep * (2 ** attempt)
            logger.warning(
                "Overpass failed on %s (%s: %s). Sleeping %.0fs before retry.",
                url, type(exc).__name__, exc, sleep_s,
            )
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Overpass fetch failed after {max_attempts} attempts across {urls}. "
        f"Last error: {last_exc}. Wait a few minutes and re-run, or point "
        f"config.OVERPASS_URLS at a less-loaded mirror."
    ) from last_exc


def boundary_area_km2(boundary: gpd.GeoDataFrame) -> float:
    """Total polygon area of a boundary frame in km^2 (UTM 37S)."""
    return float(boundary.to_crs(config.CRS_METRIC).area.sum() / 1e6)


def load_boundary_file(path, name: str = None) -> gpd.GeoDataFrame:
    """
    Read a digitized settlement boundary and normalise it to the one-row
    WGS84 frame the rest of the pipeline expects.

    Accepts anything geopandas can open (GeoJSON, GeoPackage, shapefile,
    zipped shapefile). Village- or zone-level layers are dissolved into a
    single (multi)polygon, so a Map Kibera village file or a KISIP parcel
    layer can be used directly. Raises instead of falling back: a file the
    user pointed at deliberately should never be silently replaced by the
    bbox.
    """
    from pathlib import Path

    from shapely.ops import unary_union

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Boundary file not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        # GeoJSON per RFC 7946 is WGS84 by definition; shapefiles missing a
        # .prj are the other common case and are almost always WGS84 too.
        warnings.warn(f"{path.name} has no CRS; assuming {config.CRS_WGS84}.")
        gdf = gdf.set_crs(config.CRS_WGS84)
    geoms = [
        g for g in gdf.geometry
        if g is not None and not g.is_empty and g.geom_type in ("Polygon", "MultiPolygon")
    ]
    if not geoms:
        raise ValueError(f"{path} contains no polygon geometries.")
    # buffer(0) repairs self-intersections that hand-digitised files often carry.
    union = unary_union([g.buffer(0) for g in geoms])
    boundary = gpd.GeoDataFrame(
        {
            "name": [name or path.stem],
            "boundary_source": ["file"],
            "boundary_path": [str(path)],
        },
        geometry=[union],
        crs=gdf.crs,
    ).to_crs(config.CRS_WGS84)
    area_km2 = boundary_area_km2(boundary)
    lo, hi = config.BOUNDARY_PLAUSIBLE_KM2
    if not (lo <= area_km2 <= hi):
        warnings.warn(
            f"Boundary {path.name} covers {area_km2:.2f} km^2, outside the "
            f"{lo}-{hi} km^2 window for a Nairobi informal settlement. Using it "
            f"anyway -- check it is the settlement, not a ward or sub-county."
        )
    return boundary


def default_boundary_path(settlement_key: str, data_parent=None):
    """
    The conventional location of a digitized boundary for ``settlement_key``:
    ``<data_parent>/data/boundaries/<settlement_key>.<ext>`` for the first
    extension in config.BOUNDARY_FILE_EXTENSIONS that exists, else None.
    """
    from pathlib import Path

    root = Path(data_parent) if data_parent is not None else Path(".")
    folder = root / config.BOUNDARY_DIR
    for ext in config.BOUNDARY_FILE_EXTENSIONS:
        candidate = folder / f"{settlement_key}{ext}"
        if candidate.exists():
            return candidate
    return None


def fetch_settlement_boundary(settlement_key: str, boundary_path=None) -> gpd.GeoDataFrame:
    """
    Resolve a settlement boundary polygon, in priority order:

      1. ``boundary_path`` -- a digitized file supplied by the caller
         (DataPipeline also looks in config.BOUNDARY_DIR before calling
         this). Errors here propagate; see load_boundary_file.
      2. Live OSM geocoding of ``config.SETTLEMENTS[key].osm_query``, kept
         only if the polygon area is inside config.OSM_BOUNDARY_PLAUSIBLE_KM2.
      3. The approximate ``fallback_bbox`` from config, with a warning.

    The returned frame always carries a ``boundary_source`` column
    (``file`` / ``osm`` / ``bbox``) so notebooks can report which one won.
    """
    cfg = config.SETTLEMENTS[settlement_key]
    if boundary_path is not None:
        return load_boundary_file(boundary_path, name=cfg.name)
    try:
        gdf = ox.geocode_to_gdf(cfg.osm_query)
        area_km2 = boundary_area_km2(gdf)
        lo, hi = config.OSM_BOUNDARY_PLAUSIBLE_KM2
        if not (lo <= area_km2 <= hi):
            raise ValueError(f"Geocoded area {area_km2:.2f} km^2 looks implausible")
        gdf = gdf.copy()
        gdf["boundary_source"] = "osm"
        return gdf
    except Exception as exc:
        warnings.warn(
            f"OSM geocoding failed or implausible for '{cfg.osm_query}' ({exc}); "
            f"falling back to hardcoded bbox. VERIFY this boundary visually "
            f"before using it for training -- or drop a digitized polygon at "
            f"{config.BOUNDARY_DIR}/{settlement_key}.geojson (see config.py)."
        )
        min_lon, min_lat, max_lon, max_lat = cfg.fallback_bbox
        from shapely.geometry import box
        return gpd.GeoDataFrame(
            {"name": [cfg.name], "boundary_source": ["bbox"]},
            geometry=[box(min_lon, min_lat, max_lon, max_lat)],
            crs=config.CRS_WGS84,
        )


def fetch_osm_buildings(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fetch building footprints within the boundary polygon via the Overpass API."""
    polygon = boundary.geometry.iloc[0]
    buildings = _features_from_polygon_with_retry(polygon, config.OVERPASS_TAGS)
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    buildings = buildings.to_crs(config.CRS_METRIC)
    buildings["centroid"] = buildings.geometry.centroid
    buildings["area_m2"] = buildings.geometry.area
    buildings = buildings.reset_index(drop=True)
    buildings["node_id"] = buildings.index
    return buildings


def fetch_river_corridor(settlement_keys=None, buffer_m: float = None, boundaries=None):
    """
    One Overpass query for river, stream, and drain lines over the settlement
    extents, buffered into a corridor for the Sprint 5 lift test.

    This is the fallback reference when the NRRP 37-zone polygons are not
    on disk. The query polygon is the union of the settlement extents --
    the bounding box of each frame in ``boundaries`` (a ``{key: GeoDataFrame}``
    mapping, e.g. from ``DataPipeline.get_boundary``) when given, else the
    configured ``fallback_bbox`` -- expanded by the buffer so a river just
    outside an extent still counts for buildings on the edge. Returns
    corridor polygons in ``config.CRS_METRIC``.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    from .label_validation import waterway_corridor_polygons

    buffer_m = float(buffer_m if buffer_m is not None else config.FUSION_RIVER_BUFFER_M)
    keys = list(settlement_keys) if settlement_keys is not None else list(config.SETTLEMENT_KEYS)
    boundaries = boundaries or {}
    boxes = []
    for key in keys:
        if key not in config.SETTLEMENTS:
            raise ValueError(f"Unknown settlement '{key}'.")
        if key in boundaries and boundaries[key] is not None:
            min_lon, min_lat, max_lon, max_lat = (
                boundaries[key].to_crs(config.CRS_WGS84).total_bounds
            )
        else:
            min_lon, min_lat, max_lon, max_lat = config.SETTLEMENTS[key].fallback_bbox
        boxes.append(box(min_lon, min_lat, max_lon, max_lat))
    query_area = gpd.GeoDataFrame(geometry=[unary_union(boxes)], crs=config.CRS_WGS84)
    # Expand in metres so the fetch includes channels just outside the bbox.
    query_area = query_area.to_crs(config.CRS_METRIC).buffer(buffer_m).to_crs(config.CRS_WGS84)
    raw = _features_from_polygon_with_retry(
        query_area.geometry.iloc[0],
        {"waterway": ["river", "stream", "drain"]},
    )
    return waterway_corridor_polygons(raw, buffer_m=buffer_m, metric_crs=config.CRS_METRIC)


def fetch_osm_infrastructure(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fetch water points, sanitation points, and road network for proximity features."""
    polygon = boundary.geometry.iloc[0]
    infra = _features_from_polygon_with_retry(polygon, config.INFRA_TAGS)
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


def enrich_buildings_with_deprivation(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Attach roof-material encoding, continuous deprivation scores, and the
    three-class severity label used to train GraphSAGE. Purely local -- no
    network -- so it can also be called from offline tests on synthetic
    GeoDataFrames.
    """
    out = buildings.copy()
    if "roof:material" in out.columns:
        roof_col = out["roof:material"]
    elif "building:material" in out.columns:
        roof_col = out["building:material"]
    else:
        # No material tags in this extract -- neutral scores of length n.
        roof_col = np.array([None] * len(out), dtype=object)
    out["roof_material_score"] = encode_roof_material(roof_col)

    scores = score_deprivation_indicators(
        area_m2=out["area_m2"].to_numpy(dtype=np.float64),
        dist_to_water_m=out["dist_to_water_m"].to_numpy(dtype=np.float64),
        dist_to_sanitation_m=out["dist_to_sanitation_m"].to_numpy(dtype=np.float64),
        dist_to_road_m=out["dist_to_road_m"].to_numpy(dtype=np.float64),
        roof_material_score=out["roof_material_score"].to_numpy(dtype=np.float64),
    )
    out["deprivation_score"] = scores
    out["deprivation_class"] = assign_deprivation_classes(scores)
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
        attrs = {
            "area_m2": float(row["area_m2"]),
            "dist_to_water_m": float(row.get("dist_to_water_m", 0.0)),
            "dist_to_sanitation_m": float(row.get("dist_to_sanitation_m", 0.0)),
            "dist_to_road_m": float(row.get("dist_to_road_m", 0.0)),
            "roof_material_score": float(row.get("roof_material_score", 0.5)),
            "x": row["centroid"].x,
            "y": row["centroid"].y,
        }
        if "deprivation_class" in buildings.columns:
            attrs["deprivation_class"] = int(row["deprivation_class"])
        if "deprivation_score" in buildings.columns:
            attrs["deprivation_score"] = float(row["deprivation_score"])
        graph.add_node(row["node_id"], **attrs)

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


def graph_to_pyg_data(graph: nx.Graph) -> Data:
    """
    Convert the networkx proximity graph into a PyTorch Geometric Data
    object. Node features follow ``NODE_FEATURE_NAMES`` (area + three
    infrastructure distances + roof-material score). When nodes carry
    ``deprivation_class``, that becomes ``data.y`` for GraphSAGE training.

    Centroid coordinates are stored on ``data.pos`` (N, 2) in
    ``data.pos_crs`` (EPSG:32737), in the same node order as ``data.x``.
    Sprint 5 samples the flood raster at these points; the coordinates are
    not GraphSAGE features.
    """
    data = from_networkx(graph, group_node_attrs=list(NODE_FEATURE_NAMES))
    data.x = data.x.float()

    # z-score normalise node features so GraphSAGE's mean aggregator isn't
    # dominated by raw metre-scale distances vs. area in square metres.
    mean = data.x.mean(dim=0, keepdim=True)
    std = data.x.std(dim=0, keepdim=True).clamp(min=1e-6)
    data.x = (data.x - mean) / std
    data.feature_names = list(NODE_FEATURE_NAMES)

    # Centroid easting/northing are node attributes named "x" and "y" (metric
    # CRS, see build_proximity_graph). from_networkx stores the northing on
    # data.y, which would be read as a class label, and the grouped feature
    # matrix overwrites the easting. Keep both on data.pos so Sprint 5 can
    # sample the flood raster at the same node order GraphSAGE uses, then
    # drop the stray northing tensor.
    node_order = list(graph.nodes())
    has_xy = (
        len(node_order) > 0
        and "x" in graph.nodes[node_order[0]]
        and "y" in graph.nodes[node_order[0]]
    )
    if has_xy:
        data.pos = torch.tensor(
            [[float(graph.nodes[n]["x"]), float(graph.nodes[n]["y"])] for n in node_order],
            dtype=torch.float64,
        )
        data.pos_crs = config.CRS_METRIC
        if "y" in data:
            del data.y

    if node_order and "deprivation_class" in graph.nodes[node_order[0]]:
        labels = [int(graph.nodes[n]["deprivation_class"]) for n in node_order]
        data.y = torch.tensor(labels, dtype=torch.long)

    return data


def graph_has_positions(data) -> bool:
    """True when ``data.pos`` is an (N, 2) centroid tensor aligned to the nodes."""
    pos = getattr(data, "pos", None)
    if pos is None or not hasattr(pos, "shape"):
        return False
    return (
        int(getattr(pos, "ndim", 0)) == 2
        and int(pos.shape[0]) == int(data.num_nodes)
        and int(pos.shape[1]) == 2
    )


def align_parcel_positions(existing: Data, refreshed: Data,
                           atol: float = 1e-4, rtol: float = 1e-4):
    """
    Put centroids on a graph that was saved before Sprint 5 stored ``data.pos``.

    ``refreshed`` is a graph just built from OSM (it has ``pos``). Returns
    ``(graph, status)``:

    - ``"present"`` — ``existing`` already has centroids.
    - ``"aligned"`` — same nodes in the same order (feature tensors match).
      Centroids are copied onto ``existing``, so the training-graph node
      order and labels stay put and a saved softmax still lines up.
    - ``"replaced"`` — OSM no longer reproduces that node set. The refreshed
      graph is returned. GraphSAGE weights do not depend on node index, so
      the caller reruns ``predict_proba`` on it.

    Matching on the z-scored features is the identity check. A changed
    building inventory shifts those values, and guessing an alignment from
    the node count alone would sample the flood raster at the wrong roof.
    """
    if graph_has_positions(existing):
        return existing, "present"
    if not graph_has_positions(refreshed):
        raise ValueError("refreshed graph has no data.pos; cannot attach centroids.")

    same_nodes = int(existing.num_nodes) == int(refreshed.num_nodes)
    features_match = False
    if same_nodes and existing.x is not None and refreshed.x is not None:
        if tuple(existing.x.shape) == tuple(refreshed.x.shape):
            features_match = torch.allclose(
                existing.x.float(), refreshed.x.float(), atol=atol, rtol=rtol,
            )
    if same_nodes and features_match:
        existing.pos = refreshed.pos.double()
        crs = getattr(refreshed, "pos_crs", None)
        if crs is not None:
            existing.pos_crs = crs
        return existing, "aligned"
    return refreshed, "replaced"
