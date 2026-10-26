"""
Offline tests for Google Open Buildings as a graph footprint source.

The Earth Engine export itself only runs in Colab. Here the file is a
synthetic GeoJSON: confidence filtering, clipping to the settlement,
the node schema the graph builder expects, and the nearest-neighbour
spacing the edge threshold cannot change.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data_pipeline import DataPipeline  # noqa: E402
from src.utils import graph_utils  # noqa: E402


def _squares(origins, confidence, crs=config.CRS_METRIC):
    return gpd.GeoDataFrame(
        {"confidence": confidence},
        geometry=[box(x, y, x + 2, y + 2) for x, y in origins],
        crs=crs,
    )


def test_prepare_open_buildings_filters_confidence_and_clips():
    raw = _squares(
        [(1, 1), (10, 1), (500, 500)],
        [0.9, 0.5, 0.95],
    )
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 40, 40)], crs=config.CRS_METRIC)
    out = graph_utils.prepare_open_buildings(raw, boundary, min_confidence=0.65)
    assert len(out) == 1
    assert out.crs.to_string() == config.CRS_METRIC
    assert list(out["node_id"]) == [0]
    assert out["building_source"].iloc[0] == "open_buildings"
    assert out["area_m2"].iloc[0] == pytest.approx(4.0)
    assert out["centroid"].iloc[0].x == pytest.approx(2.0)


def test_prepare_open_buildings_rejects_an_empty_confident_set():
    raw = _squares([(1, 1)], [0.5])
    with pytest.raises(ValueError, match="confidence"):
        graph_utils.prepare_open_buildings(raw, min_confidence=0.65)


def test_prepare_open_buildings_warns_when_confidence_is_absent_and_drops_points():
    raw = gpd.GeoDataFrame(
        geometry=[box(0, 0, 2, 2), Point(10, 10)],
        crs=config.CRS_METRIC,
    )
    with pytest.warns(UserWarning, match="no confidence"):
        out = graph_utils.prepare_open_buildings(raw)
    assert len(out) == 1


def test_spacing_summary_reports_the_footprint_gap_not_the_edge_threshold():
    raw = _squares([(0, 0), (5, 0), (10, 0), (15, 0)], [0.9, 0.9, 0.9, 0.9])
    # 2 m squares whose centroids are 5 m apart.
    out = graph_utils.prepare_open_buildings(raw, min_confidence=0.65)
    summary = graph_utils.spacing_summary(out)
    assert summary["n"] == 4
    assert summary["median_m"] == pytest.approx(5.0)
    assert summary["p90_m"] == pytest.approx(5.0)
    distances = graph_utils.nearest_neighbour_distances(out)
    assert np.median(distances) == pytest.approx(5.0)


def test_pipeline_discovers_and_loads_an_open_buildings_file(tmp_path, monkeypatch):
    folder = tmp_path / config.BUILDINGS_DIR
    folder.mkdir(parents=True)
    _squares([(0, 0), (5, 0), (10, 0)], [0.8, 0.8, 0.8]).to_file(
        folder / "kibera_open_buildings.geojson", driver="GeoJSON",
    )
    boundary = gpd.GeoDataFrame(geometry=[box(-1, -1, 20, 10)], crs=config.CRS_METRIC)
    monkeypatch.setattr(DataPipeline, "get_boundary", lambda self: boundary)

    pipeline = DataPipeline(
        "kibera", data_root=str(tmp_path / "data"), building_source="open_buildings",
    )
    assert pipeline.open_buildings_path == folder / "kibera_open_buildings.geojson"
    buildings = pipeline.load_buildings()
    assert len(buildings) == 3
    assert pipeline.building_spacing["median_m"] == pytest.approx(5.0)

    def boom(*_a, **_k):
        raise AssertionError("OSM footprints must not be fetched")
    monkeypatch.setattr(graph_utils, "fetch_osm_buildings", boom)
    pipeline.load_buildings()


def test_pipeline_requires_a_file_and_a_known_source(tmp_path, monkeypatch):
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=config.CRS_METRIC)
    monkeypatch.setattr(DataPipeline, "get_boundary", lambda self: boundary)
    pipeline = DataPipeline(
        "mathare", data_root=str(tmp_path / "data"), building_source="open_buildings",
    )
    assert pipeline.open_buildings_path is None
    with pytest.raises(FileNotFoundError, match="export_open_buildings"):
        pipeline.load_buildings()
    with pytest.raises(ValueError, match="building_source"):
        DataPipeline("kibera", data_root=str(tmp_path / "data"), building_source="google")
