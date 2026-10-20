"""
Geo helpers for the Sprint 5 fusion engine: sample a flood-probability
raster at building centroids, test parcel-in-NRRP membership, blend
overlapping tile predictions into a probability surface, and write the
classified GeoJSON layer.

The numpy policy core in ``src.models.fusion_engine`` stays dependency-light;
everything here needs rasterio / geopandas and is imported lazily from the
engine's export / NRRP paths. Offline tests exercise these against tiny
in-memory rasters and synthetic polygons -- no GEE / Drive.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np

from .. import config

logger = logging.getLogger(__name__)

# Slim attribute set written next to ``{settlement}_graph.pt`` so Sprint 5 can
# join GraphSAGE node order back onto footprints without re-querying Overpass.
PARCEL_GEOJSON_COLUMNS = (
    "node_id",
    "area_m2",
    "dist_to_water_m",
    "dist_to_sanitation_m",
    "dist_to_road_m",
    "roof_material_score",
    "deprivation_score",
    "deprivation_class",
)


def save_buildings_geojson(buildings, path: Union[str, Path], crs: str = None) -> Path:
    """
    Persist building footprints (polygons) + the deprivation columns GraphSAGE
    trained on. Geometry is reprojected to WGS84 by default so the file is a
    valid RFC 7946 GeoJSON; node_id is the join key back to the PyG graph.
    """
    crs = crs or config.CRS_WGS84
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in PARCEL_GEOJSON_COLUMNS if c in buildings.columns]
    gdf = buildings[keep + ["geometry"]].copy()
    # GeoJSON cannot serialise a separate centroid column; the footprint is
    # the parcel geometry the dashboard (Sprint 6) will render.
    gdf.to_crs(crs).to_file(path, driver="GeoJSON")
    logger.info("Wrote %d building parcels to %s", len(gdf), path)
    return path


def load_buildings_geojson(path: Union[str, Path]):
    import geopandas as gpd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Building parcels not found at {path}. Re-run "
            f"DataPipeline.acquire_graph_data (Sprint 5 writes "
            f"{{settlement}}_buildings.geojson next to the graph)."
        )
    return gpd.read_file(path)


def sample_array_at_xy(
    array: np.ndarray,
    transform,
    xs: Sequence[float],
    ys: Sequence[float],
    nodata: float = np.nan,
) -> np.ndarray:
    """
    Nearest-neighbour sample of a 2-d array at map coordinates ``(xs, ys)``
    given an affine transform. Points that fall outside the raster return
    ``nodata``. Nearest-neighbour (not bilinear) so a binary flood mask used
    as a stand-in probability stays in {0, 1}.
    """
    from rasterio.transform import rowcol

    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-d array, got shape {array.shape}")
    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    h, w = array.shape
    out = np.full(rows.shape, nodata, dtype=np.float64)
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out[inside] = array[rows[inside], cols[inside]]
    return out


def sample_raster_at_points(raster_path: Union[str, Path], points_gdf, band: int = 1) -> np.ndarray:
    """
    Sample band ``band`` (1-based, rasterio convention) of a GeoTIFF at each
    point/centroid in ``points_gdf``. Reprojects the points onto the raster
    CRS first. Polygon geometries are sampled at their centroid -- the parcel
    is represented by a single flood probability, matching the late-fusion
    design (one score per OSM building, not a per-pixel overlay).
    """
    import rasterio

    path = Path(raster_path)
    with rasterio.open(path) as src:
        gdf = points_gdf
        if gdf.crs is not None and src.crs is not None and str(gdf.crs) != str(src.crs):
            gdf = gdf.to_crs(src.crs)
        centroids = gdf.geometry.centroid
        xs = centroids.x.to_numpy(dtype=np.float64)
        ys = centroids.y.to_numpy(dtype=np.float64)
        band_arr = src.read(band)
        return sample_array_at_xy(band_arr, src.transform, xs, ys)


def union_geometries(gdf):
    """CRS-preserving polygon union (geopandas 1.0 ``union_all`` or ``unary_union``)."""
    if hasattr(gdf, "union_all"):
        return gdf.union_all()
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.unary_union


def points_in_reference(parcels_gdf, reference_gdf) -> np.ndarray:
    """
    Boolean mask, one entry per parcel: True if the building centroid
    intersects any reference polygon (NRRP high-risk zone, or a buffered
    river corridor). Reprojects the reference onto the parcel CRS. Parcels
    with empty geometry are False.
    """
    if reference_gdf is None or len(reference_gdf) == 0:
        return np.zeros(len(parcels_gdf), dtype=bool)
    ref = reference_gdf
    if parcels_gdf.crs is not None:
        ref = reference_gdf.to_crs(parcels_gdf.crs)
    # union_all (geopandas 1.0+) / unary_union fallback: cheap
    # point-in-multipolygon test at settlement scale (thousands of
    # buildings, tens of NRRP polygons).
    if hasattr(ref.geometry, "union_all"):
        union = ref.geometry.union_all()
    else:
        union = ref.geometry.unary_union
    centroids = parcels_gdf.geometry.centroid
    return np.array(
        [False if geom is None or geom.is_empty else bool(geom.intersects(union))
         for geom in centroids],
        dtype=bool,
    )


def iter_windows(
    height: int,
    width: int,
    tile_size: int = None,
    overlap: int = None,
) -> list:
    """
    (row_off, col_off, h, w) windows covering a raster with the same 256/50
    tiling the training pipeline uses. The last window in each dimension is
    shifted back so it stays ``tile_size`` when the raster is large enough,
    matching a sliding-window inference pass (overlap is averaged later by
    ``blend_overlapping_windows``).
    """
    tile_size = tile_size if tile_size is not None else config.TILE_SIZE_PX
    overlap = overlap if overlap is not None else config.TILE_OVERLAP_PX
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError(
            f"Need tile_size > overlap >= 0; got tile_size={tile_size}, overlap={overlap}"
        )
    stride = tile_size - overlap
    if height < tile_size or width < tile_size:
        # Single window clipped to the raster -- caller pads before the U-Net
        # if the network requires a multiple of 16.
        return [(0, 0, min(tile_size, height), min(tile_size, width))]

    rows = list(range(0, height - tile_size + 1, stride))
    cols = list(range(0, width - tile_size + 1, stride))
    if rows[-1] != height - tile_size:
        rows.append(height - tile_size)
    if cols[-1] != width - tile_size:
        cols.append(width - tile_size)

    windows = []
    for r in rows:
        for c in cols:
            windows.append((int(r), int(c), int(tile_size), int(tile_size)))
    return windows


def blend_overlapping_windows(
    canvas_shape: tuple,
    windows: Sequence[tuple],
    tiles: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Mean-blend overlapping 2-d tiles onto a canvas. ``windows`` entries are
    ``(row_off, col_off, h, w)`` and must match ``tiles`` length. Pixels that
    no window covers stay NaN so callers can distinguish 'no prediction'
    from 'predicted zero'.
    """
    if len(windows) != len(tiles):
        raise ValueError(
            f"windows ({len(windows)}) and tiles ({len(tiles)}) length must match"
        )
    height, width = canvas_shape
    acc = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float64)
    for (row, col, h, w), tile in zip(windows, tiles):
        tile = np.asarray(tile, dtype=np.float64)
        if tile.shape != (h, w):
            # Allow a full-size network output clipped to the window.
            tile = tile[:h, :w]
        acc[row:row + h, col:col + w] += tile
        count[row:row + h, col:col + w] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = acc / np.where(count > 0, count, np.nan)
    return out


def write_classified_geojson(parcels, path: Union[str, Path], crs: str = None) -> Path:
    """Write a classification GeoDataFrame; drop non-serialisable columns."""
    crs = crs or config.CRS_WGS84
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = parcels.copy()
    geom_name = gdf.geometry.name
    drop = [
        c for c in gdf.columns
        if c != geom_name and (
            _is_unserialisable(gdf[c]) or _is_geometry_column(gdf, c)
        )
    ]
    if drop:
        gdf = gdf.drop(columns=drop)
    gdf.to_crs(crs).to_file(path, driver="GeoJSON")
    return path


def predict_flood_surface(
    stack: np.ndarray,
    input_indices: Sequence[int],
    predict_tile,
    tile_size: int = None,
    overlap: int = None,
) -> np.ndarray:
    """
    Sliding-window flood probability over a (C, H, W) stack.

    ``predict_tile`` consumes a ``(C_in, tile, tile)`` float32 array and
    returns a ``(tile, tile)`` probability map -- typically
    ``model.predict_proba`` wrapped around a numpy tile. Overlapping windows
    are mean-blended. Rasters smaller than ``tile_size`` are padded with
    zeros (the same 'no data -> 0' convention the tile dataset uses) so a
    U-Net that requires 256 px still runs on a tiny synthetic stack.
    """
    tile_size = tile_size if tile_size is not None else config.TILE_SIZE_PX
    overlap = overlap if overlap is not None else config.TILE_OVERLAP_PX
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"Expected (C, H, W) stack, got shape {stack.shape}")
    _, height, width = stack.shape
    inputs = np.nan_to_num(stack[list(input_indices)], nan=0.0).astype(np.float32)

    pad_h = max(tile_size - height, 0)
    pad_w = max(tile_size - width, 0)
    if pad_h or pad_w:
        inputs = np.pad(inputs, ((0, 0), (0, pad_h), (0, pad_w)))
    work_h, work_w = inputs.shape[1], inputs.shape[2]
    windows = iter_windows(work_h, work_w, tile_size=tile_size, overlap=overlap)
    tiles = []
    for row, col, h, w in windows:
        chip = inputs[:, row:row + h, col:col + w]
        if chip.shape[1] != tile_size or chip.shape[2] != tile_size:
            padded = np.zeros((chip.shape[0], tile_size, tile_size), dtype=np.float32)
            padded[:, :chip.shape[1], :chip.shape[2]] = chip
            chip = padded
        pred = np.asarray(predict_tile(chip), dtype=np.float64)
        tiles.append(pred[:h, :w])
    blended = blend_overlapping_windows((work_h, work_w), windows, tiles)
    return blended[:height, :width]


def _is_unserialisable(series) -> bool:
    if len(series) == 0:
        return False
    v = series.iloc[0]
    if isinstance(v, np.ndarray):
        return True
    return False


def _is_geometry_column(gdf, column: str) -> bool:
    """
    True for extra geometry columns (e.g. a separate ``centroid`` Point
    column left on the buildings GeoDataFrame). GeoJSON drivers accept only
    one geometry column, so those have to be dropped before ``to_file``.
    """
    if column == gdf.geometry.name:
        return False
    dtype = str(gdf[column].dtype)
    if dtype == "geometry":
        return True
    if len(gdf) == 0:
        return False
    try:
        from shapely.geometry.base import BaseGeometry
        return isinstance(gdf[column].iloc[0], BaseGeometry)
    except Exception:
        return False


def load_nrrp_or_river_reference(
    aoi_gdf,
    nrrp_path: Union[str, Path] = None,
    river_buffer_m: float = 100.0,
):
    """
    Prefer NRRP high-risk zone polygons when a path is supplied; otherwise
    buffer OSM river/stream lines inside ``aoi_gdf`` (the same fallback
    Sprint 2 v2 uses for flood-label lift). Returns (gdf, source_label) or
    (None, reason) when neither is available.

    Network-bound (Overpass) -- do not call from offline unit tests.
    """
    import geopandas as gpd

    if nrrp_path:
        path = Path(nrrp_path)
        if not path.exists():
            raise FileNotFoundError(f"NRRP polygons not found at {path}")
        gdf = gpd.read_file(path)
        return gdf, f"NRRP polygons ({path})"

    try:
        import osmnx as ox
        from shapely.geometry import box

        aoi = aoi_gdf.to_crs(config.CRS_WGS84)
        minx, miny, maxx, maxy = aoi.total_bounds
        polygon = box(minx, miny, maxx, maxy)
        waterways = ox.features_from_polygon(
            polygon, tags={"waterway": ["river", "stream", "drain"]},
        )
        waterways = waterways[
            waterways.geometry.type.isin(["LineString", "MultiLineString"])
        ]
        if waterways.empty:
            return None, "OSM waterways query returned no river/stream/drain lines"
        metric = waterways.to_crs(config.CRS_METRIC)
        buffered = metric.buffer(river_buffer_m)
        gdf = gpd.GeoDataFrame(geometry=buffered, crs=config.CRS_METRIC)
        return gdf, f"OSM river corridor (buffer {river_buffer_m} m)"
    except Exception as exc:
        logger.warning("Could not build an OSM river reference: %s", exc)
        return None, f"OSM river fallback failed ({exc})"
