"""
Offline tests for the digitized-boundary input.

config.py has long promised a `custom_boundary_path`; these tests pin down
the implementation that finally exists: a file beats OSM beats the bbox, a
multi-feature village file dissolves to one polygon, a bad file raises
instead of silently falling back, DataPipeline auto-discovers
data/boundaries/<key>.geojson, and the polygon actually used is written
out for visual inspection. No network: geocoding is monkeypatched.
"""

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data_pipeline import DataPipeline  # noqa: E402
from src.utils import graph_utils  # noqa: E402


# Two adjacent "villages" roughly 1.1 km x 1.1 km each, inside Kibera's bbox.
VILLAGE_A = box(36.780, -1.320, 36.790, -1.310)
VILLAGE_B = box(36.790, -1.320, 36.800, -1.310)


def _write_villages(path, crs="EPSG:4326"):
    gdf = gpd.GeoDataFrame(
        {"village": ["Kianda", "Soweto"]},
        geometry=[VILLAGE_A, VILLAGE_B],
        crs="EPSG:4326",
    )
    if crs != "EPSG:4326":
        gdf = gdf.to_crs(crs)
    gdf.to_file(path, driver="GeoJSON" if str(path).endswith("json") else None)
    return path


def _no_network(monkeypatch):
    def boom(query):
        raise AssertionError(f"geocode_to_gdf must not be called (query={query!r})")
    monkeypatch.setattr(graph_utils.ox, "geocode_to_gdf", boom)


def test_load_boundary_file_dissolves_villages_into_one_wgs84_polygon(tmp_path):
    path = _write_villages(tmp_path / "kibera.geojson")
    boundary = graph_utils.load_boundary_file(path, name="Kibera")

    assert len(boundary) == 1
    assert boundary.crs.to_epsg() == 4326
    assert boundary["boundary_source"].iloc[0] == "file"
    assert boundary["name"].iloc[0] == "Kibera"
    # The union of the two villages is the 2.2 km x 1.1 km rectangle.
    assert boundary.geometry.iloc[0].equals(VILLAGE_A.union(VILLAGE_B))
    assert graph_utils.boundary_area_km2(boundary) == pytest.approx(2.46, abs=0.05)


def test_load_boundary_file_reprojects_a_metric_file(tmp_path):
    path = _write_villages(tmp_path / "kibera_utm.geojson", crs=config.CRS_METRIC)
    boundary = graph_utils.load_boundary_file(path)
    assert boundary.crs.to_epsg() == 4326
    minx, miny, maxx, maxy = boundary.total_bounds
    assert (minx, miny, maxx, maxy) == pytest.approx((36.78, -1.32, 36.80, -1.31), abs=1e-4)


def test_load_boundary_file_rejects_files_without_polygons(tmp_path):
    lines = gpd.GeoDataFrame(
        geometry=[LineString([(36.78, -1.32), (36.79, -1.31)])], crs="EPSG:4326",
    )
    path = tmp_path / "lines.geojson"
    lines.to_file(path, driver="GeoJSON")
    with pytest.raises(ValueError, match="no polygon"):
        graph_utils.load_boundary_file(path)
    with pytest.raises(FileNotFoundError):
        graph_utils.load_boundary_file(tmp_path / "missing.geojson")


def test_load_boundary_file_warns_on_implausible_area_and_missing_crs(tmp_path, monkeypatch):
    # A whole-sub-county sized polygon (~0.3 deg ~ 33 km on a side).
    huge = gpd.GeoDataFrame(geometry=[box(36.6, -1.5, 36.9, -1.2)], crs="EPSG:4326")
    path = tmp_path / "huge.geojson"
    huge.to_file(path, driver="GeoJSON")
    with pytest.warns(UserWarning, match="outside the"):
        boundary = graph_utils.load_boundary_file(path)
    assert graph_utils.boundary_area_km2(boundary) > config.BOUNDARY_PLAUSIBLE_KM2[1]

    # A shapefile with no .prj reads back with crs=None; it should be assumed WGS84.
    no_crs = gpd.GeoDataFrame(geometry=[VILLAGE_A])
    monkeypatch.setattr(graph_utils.gpd, "read_file", lambda p: no_crs)
    with pytest.warns(UserWarning, match="no CRS"):
        normalised = graph_utils.load_boundary_file(path)
    assert normalised.crs.to_epsg() == 4326
    assert normalised.geometry.iloc[0].equals(VILLAGE_A)


def test_fetch_settlement_boundary_prefers_the_file_and_skips_geocoding(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    path = _write_villages(tmp_path / "kibera.geojson")
    boundary = graph_utils.fetch_settlement_boundary("kibera", boundary_path=path)
    assert boundary["boundary_source"].iloc[0] == "file"
    assert boundary["name"].iloc[0] == config.SETTLEMENTS["kibera"].name


def test_fetch_settlement_boundary_file_errors_do_not_fall_back(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    with pytest.raises(FileNotFoundError):
        graph_utils.fetch_settlement_boundary("kibera", boundary_path=tmp_path / "nope.geojson")


def test_fetch_settlement_boundary_osm_then_bbox(monkeypatch):
    plausible = gpd.GeoDataFrame(geometry=[VILLAGE_A.union(VILLAGE_B)], crs="EPSG:4326")
    monkeypatch.setattr(graph_utils.ox, "geocode_to_gdf", lambda q: plausible)
    boundary = graph_utils.fetch_settlement_boundary("kibera")
    assert boundary["boundary_source"].iloc[0] == "osm"

    # Nominatim returning a point-sized polygon: implausible -> bbox fallback.
    tiny = gpd.GeoDataFrame(geometry=[Point(36.78, -1.31).buffer(0.0001)], crs="EPSG:4326")
    monkeypatch.setattr(graph_utils.ox, "geocode_to_gdf", lambda q: tiny)
    with pytest.warns(UserWarning, match="falling back to hardcoded bbox"):
        boundary = graph_utils.fetch_settlement_boundary("kibera")
    assert boundary["boundary_source"].iloc[0] == "bbox"
    assert tuple(np.round(boundary.total_bounds, 4)) == pytest.approx(
        config.SETTLEMENTS["kibera"].fallback_bbox, abs=1e-4
    )


def test_data_pipeline_discovers_boundary_file_and_writes_the_polygon_used(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    data_root = tmp_path / "data"
    boundaries_dir = tmp_path / config.BOUNDARY_DIR
    boundaries_dir.mkdir(parents=True)
    _write_villages(boundaries_dir / "kibera.geojson")

    pipeline = DataPipeline(settlement_key="kibera", data_root=str(data_root))
    assert pipeline.boundary_path == boundaries_dir / "kibera.geojson"
    assert pipeline.boundary_source == "unresolved"

    boundary = pipeline.get_boundary()
    assert pipeline.boundary_source == "file"
    assert boundary.geometry.iloc[0].equals(VILLAGE_A.union(VILLAGE_B))

    used = pipeline.boundary_used_path
    assert used == tmp_path / config.RAW_DIR / "kibera_boundary_used.geojson"
    assert used.exists()
    snapshot = gpd.read_file(used)
    assert snapshot["boundary_source"].iloc[0] == "file"
    assert snapshot.geometry.iloc[0].equals(VILLAGE_A.union(VILLAGE_B))

    # Explicit argument wins over the conventional location.
    other = _write_villages(tmp_path / "elsewhere.geojson")
    explicit = DataPipeline("kibera", data_root=str(data_root), boundary_path=str(other))
    assert explicit.boundary_path == other


def test_data_pipeline_without_a_file_records_the_fallback_source(tmp_path, monkeypatch):
    def fail(query):
        raise RuntimeError("Nominatim unreachable")
    monkeypatch.setattr(graph_utils.ox, "geocode_to_gdf", fail)

    pipeline = DataPipeline(settlement_key="mathare", data_root=str(tmp_path / "data"))
    assert pipeline.boundary_path is None
    with pytest.warns(UserWarning):
        pipeline.get_boundary()
    assert pipeline.boundary_source == "bbox"
    snapshot = gpd.read_file(pipeline.boundary_used_path)
    assert snapshot["boundary_source"].iloc[0] == "bbox"


def test_river_corridor_query_follows_supplied_boundaries(monkeypatch):
    captured = {}

    def fake_features(polygon, tags):
        captured["polygon"] = polygon
        captured["tags"] = tags
        return gpd.GeoDataFrame(
            geometry=[LineString([(36.80, -1.32), (36.82, -1.30)])], crs="EPSG:4326",
        )
    monkeypatch.setattr(graph_utils, "_features_from_polygon_with_retry", fake_features)

    # A digitized Kibera boundary that extends well east of the config bbox.
    wide = gpd.GeoDataFrame(geometry=[box(36.770, -1.325, 36.830, -1.300)], crs="EPSG:4326")
    corridor = graph_utils.fetch_river_corridor(["kibera"], buffer_m=50.0, boundaries={"kibera": wide})
    assert captured["tags"] == {"waterway": ["river", "stream", "drain"]}
    assert captured["polygon"].bounds[2] > 36.83  # covers the file extent plus buffer
    assert len(corridor) == 1
    assert corridor.crs.to_string() == config.CRS_METRIC

    # Without boundaries the query stays on the config bbox.
    graph_utils.fetch_river_corridor(["kibera"], buffer_m=50.0)
    assert captured["polygon"].bounds[2] < 36.80
