"""
Offline unit tests for Sprint 5 late-fusion policy engine.

Pure numpy for the rule matrix / NRRP lift; tiny synthetic rasters and
polygons for sampling, blending, and GeoJSON I/O. No GEE / OSM / GPU.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.models.fusion_engine import (
    FusionEngine,
    align_node_outputs_to_parcels,
    concatenate_head_outputs,
    coerce_deprivation_proba,
    dump_policy_report,
    flood_bin,
    fuse_settlements,
    lookup_policy,
    policy_nrrp_report,
    summarise_policy,
)
from src.utils.parcel_io import (
    blend_overlapping_windows,
    iter_windows,
    points_in_reference,
    predict_flood_surface,
    sample_array_at_xy,
    sample_raster_at_points,
    save_buildings_geojson,
)


# The nine cells of DEFAULT_POLICY_MATRIX, as (flood_p, deprivation_class, expected_policy).
# Cuts: low < 0.30, high >= 0.60.
_MATRIX_CELLS = [
    # low flood
    (0.10, 0, 0),  # formalise
    (0.10, 1, 1),  # upgrade
    (0.10, 2, 1),  # upgrade
    # medium flood
    (0.45, 0, 1),  # upgrade
    (0.45, 1, 1),  # upgrade
    (0.45, 2, 2),  # resettle (compounded)
    # high flood
    (0.85, 0, 2),  # resettle even if low deprivation
    (0.85, 1, 2),
    (0.85, 2, 2),
]


def test_align_node_outputs_reorders_and_flags_gaps():
    values = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    aligned = align_node_outputs_to_parcels(values, node_ids=[10, 20, 30], parcel_node_ids=[30, 10])
    assert aligned.tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    with pytest.raises(KeyError, match="missing"):
        align_node_outputs_to_parcels(values, node_ids=[10, 20, 30], parcel_node_ids=[99])


def test_concatenate_head_outputs_shape():
    packed = concatenate_head_outputs([0.2, 0.8], [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    assert packed.shape == (2, 4)
    assert packed[0, 0] == pytest.approx(0.2)
    assert packed[1, 1:].sum() == pytest.approx(1.0)


def test_concatenate_length_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        concatenate_head_outputs([0.1, 0.2], [[0.5, 0.3, 0.2]])


def test_coerce_class_indices_to_one_hot():
    proba = coerce_deprivation_proba(np.array([0, 2, 1]))
    assert proba.shape == (3, 3)
    assert proba.argmax(axis=1).tolist() == [0, 2, 1]
    assert np.all(proba.sum(axis=1) == 1.0)


def test_flood_bin_boundaries():
    p = [0.0, 0.2999, 0.30, 0.5999, 0.60, 1.0]
    bins = flood_bin(p)
    assert bins.tolist() == [0, 0, 1, 1, 2, 2]


def test_flood_bin_rejects_inverted_cuts():
    with pytest.raises(ValueError, match="low_max"):
        flood_bin([0.5], low_max=0.7, high_min=0.4)


@pytest.mark.parametrize("flood_p,dep_class,expected", _MATRIX_CELLS)
def test_default_matrix_cell(flood_p, dep_class, expected):
    engine = FusionEngine()
    out = engine.classify([flood_p], [dep_class])
    assert int(out["policy_class"][0]) == expected
    assert out["policy_name"][0] == config.POLICY_CLASS_LABELS[expected]
    assert 0.0 <= float(out["confidence"][0]) <= 1.0
    assert out["concatenated"].shape == (1, 4)


def test_custom_matrix_changes_assignment_without_retraining():
    """Retuning the matrix (proposal: different city, same weights) must flip labels."""
    flood = [0.10]
    dep = [2]  # high deprivation, low flood -> upgrade under the default matrix
    default = FusionEngine().classify(flood, dep)
    assert int(default["policy_class"][0]) == 1
    # A harsher city: high deprivation always resettles, even on dry ground.
    harsh = np.array([[0, 1, 2], [1, 1, 2], [2, 2, 2]], dtype=np.int64)
    flipped = FusionEngine(policy_matrix=harsh).classify(flood, dep)
    assert int(flipped["policy_class"][0]) == 2


def test_classify_batch_aligns_with_lookup():
    rng = np.random.default_rng(0)
    flood = rng.random(50)
    dep = rng.integers(0, 3, size=50)
    engine = FusionEngine()
    out = engine.classify(flood, dep)
    bins = flood_bin(flood)
    expected = lookup_policy(bins, dep)
    assert out["policy_class"].tolist() == expected.tolist()


def test_resettlement_lift_above_target_when_concentrated():
    """Relocate parcels sitting inside the reference must score lift >> 1."""
    # 100 parcels, 20% inside NRRP. All 20 inside parcels are resettle;
    # the 80 outside are formalise/upgrade.
    policy = np.array([2] * 20 + [0] * 50 + [1] * 30)
    inside = np.array([True] * 20 + [False] * 80)
    report = policy_nrrp_report(policy, inside)
    assert report["resettlement_lift"] == pytest.approx(1.0 / 0.2, rel=1e-3)
    assert report["meets_lift_target"]
    assert report["formalise_depleted_in_reference"]
    assert report["formalise_lift"] < 1.0


def test_resettlement_lift_near_one_when_independent():
    rng = np.random.default_rng(1)
    policy = rng.integers(0, 3, size=2000)
    inside = rng.random(2000) < 0.3
    report = policy_nrrp_report(policy, inside)
    assert 0.8 < report["resettlement_lift"] < 1.2
    assert report["meets_lift_target"] is False or report["resettlement_lift"] >= 1.5


def test_policy_nrrp_length_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        policy_nrrp_report([0, 1], [True])


def test_summarise_policy_by_settlement():
    policy = np.array([0, 0, 2, 1, 2, 2])
    keys = ["kibera", "kibera", "kibera", "mukuru", "mukuru", "mukuru"]
    summary = summarise_policy(policy, settlement_keys=keys)
    assert summary["counts"]["compensated_resettlement"] == 3
    assert summary["per_settlement"]["kibera"]["counts"]["retain_and_formalise"] == 2
    assert summary["per_settlement"]["mukuru"]["counts"]["compensated_resettlement"] == 2


def test_blend_overlapping_windows_averages_overlap():
    # 4x4 canvas, two 4x2 windows overlapping by 1 column would be more
    # awkward; use two 2x2 windows sharing one column on a 2x3 canvas.
    windows = [(0, 0, 2, 2), (0, 1, 2, 2)]
    left = np.ones((2, 2))
    right = np.full((2, 2), 3.0)
    blended = blend_overlapping_windows((2, 3), windows, [left, right])
    assert blended[0, 0] == pytest.approx(1.0)   # only left
    assert blended[0, 1] == pytest.approx(2.0)   # mean of 1 and 3
    assert blended[0, 2] == pytest.approx(3.0)   # only right


def test_iter_windows_covers_raster_with_shifted_last_tile():
    windows = iter_windows(10, 10, tile_size=4, overlap=1)
    # stride = 3; last window is shifted back to row/col 6 so it stays 4x4.
    origins = {(r, c) for r, c, h, w in windows}
    assert (0, 0) in origins
    assert (6, 6) in origins
    assert all(h == 4 and w == 4 for _, _, h, w in windows)


def test_sample_array_at_xy_nearest_neighbour():
    from affine import Affine

    # 4x4 array, 1 m pixels, origin (0, 4) going down (north-up).
    array = np.arange(16, dtype=np.float64).reshape(4, 4)
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    # Pixel (row=0, col=0) covers x in [0,1), y in (3,4] under this affine.
    xs = [0.5, 2.5, 10.0]
    ys = [3.5, 1.5, 0.0]
    sampled = sample_array_at_xy(array, transform, xs, ys)
    assert sampled[0] == pytest.approx(0.0)
    assert sampled[1] == pytest.approx(array[2, 2])
    assert np.isnan(sampled[2])  # outside


def test_sample_raster_at_building_centroids(tmp_path):
    pytest.importorskip("rasterio")
    gpd = pytest.importorskip("geopandas")
    from affine import Affine
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    data = np.array(
        [[0, 0, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 1, 1]],
        dtype=np.float32,
    )
    tif = tmp_path / "flood.tif"
    with rasterio.open(
        tif, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
        crs="EPSG:3857", transform=transform,
    ) as dst:
        dst.write(data, 1)

    parcels = gpd.GeoDataFrame(
        geometry=[box(0.1, 2.1, 0.9, 2.9), box(2.1, 2.1, 2.9, 2.9)],
        crs="EPSG:3857",
    )
    sampled = sample_raster_at_points(tif, parcels, band=1)
    assert sampled[0] == pytest.approx(0.0)
    assert sampled[1] == pytest.approx(1.0)


def test_points_in_reference_mask():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box, Point

    parcels = gpd.GeoDataFrame(
        geometry=[Point(1, 1).buffer(0.2), Point(10, 10).buffer(0.2)],
        crs="EPSG:3857",
    )
    reference = gpd.GeoDataFrame(geometry=[box(0, 0, 3, 3)], crs="EPSG:3857")
    mask = points_in_reference(parcels, reference)
    assert mask.tolist() == [True, False]


def test_predict_flood_surface_blends_dummy_predictor():
    # Identity predictor: probability = first input channel.
    stack = np.zeros((3, 8, 8), dtype=np.float32)
    stack[0, :, :4] = 1.0
    def predict_tile(chip):
        return chip[0]
    surface = predict_flood_surface(
        stack, input_indices=[0, 1], predict_tile=predict_tile,
        tile_size=4, overlap=1,
    )
    assert surface.shape == (8, 8)
    assert surface[:, :4].mean() == pytest.approx(1.0)
    assert surface[:, 4:].mean() == pytest.approx(0.0)


def test_export_drops_extra_centroid_geometry_column(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    parcels = gpd.GeoDataFrame(
        {"node_id": [0, 1]},
        geometry=[box(0, 0, 1, 1), box(5, 5, 6, 6)],
        crs="EPSG:3857",
    )
    parcels["centroid"] = parcels.geometry.centroid
    engine = FusionEngine()
    payload = engine.classify_and_export(
        [0.1, 0.9], [0, 2], parcels,
        path=tmp_path / "with_centroid.geojson",
        settlement_key="kibera",
    )
    roundtrip = __import__("geopandas").read_file(payload["path"])
    assert "centroid" not in roundtrip.columns
    assert len(roundtrip) == 2


def test_save_and_export_geojson_roundtrip(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    buildings = gpd.GeoDataFrame(
        {
            "node_id": [0, 1],
            "area_m2": [20.0, 40.0],
            "dist_to_water_m": [10.0, 250.0],
            "dist_to_sanitation_m": [10.0, 250.0],
            "dist_to_road_m": [5.0, 80.0],
            "roof_material_score": [0.2, 0.9],
            "deprivation_score": [0.2, 0.8],
            "deprivation_class": [0, 2],
        },
        geometry=[box(36.78, -1.31, 36.781, -1.309), box(36.79, -1.32, 36.791, -1.319)],
        crs="EPSG:4326",
    )
    bpath = tmp_path / "kibera_buildings.geojson"
    save_buildings_geojson(buildings, bpath)
    loaded = gpd.read_file(bpath)
    assert len(loaded) == 2
    assert "node_id" in loaded.columns

    engine = FusionEngine()
    flood = np.array([0.1, 0.9])
    dep = np.array([0, 2])
    payload = engine.classify_and_export(
        flood, dep, loaded,
        path=tmp_path / "kibera_policy.geojson",
        settlement_key="kibera",
        in_reference=np.array([False, True]),
    )
    assert payload["nrrp_report"]["meets_lift_target"]
    assert Path(payload["path"]).exists()
    roundtrip = gpd.read_file(payload["path"])
    assert set(roundtrip["policy_name"]) == {
        "Retain and Formalise",
        "Compensated Resettlement",
    }
    dumped = dump_policy_report(payload, tmp_path / "report.json")
    saved = json.loads(dumped.read_text())
    assert "summary" in saved
    assert "parcels" not in saved


def test_fuse_settlements_writes_combined_layer(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    def _parcels(shift):
        return gpd.GeoDataFrame(
            {"node_id": [0, 1]},
            geometry=[
                box(0 + shift, 0, 1 + shift, 1),
                box(5 + shift, 5, 6 + shift, 6),
            ],
            crs="EPSG:3857",
        )

    parcels = {"kibera": _parcels(0), "mathare": _parcels(20)}
    flood = {
        "kibera": np.array([0.1, 0.9]),
        "mathare": np.array([0.15, 0.2]),
    }
    dep = {
        "kibera": np.array([0, 2]),
        "mathare": np.array([1, 1]),
    }
    reference = gpd.GeoDataFrame(geometry=[box(4, 4, 8, 8)], crs="EPSG:3857")
    fused = fuse_settlements(
        parcels, flood, dep, output_dir=tmp_path / "classifications",
        reference_gdf=reference,
    )
    assert "kibera" in fused["per_settlement"]
    assert "mathare" in fused["per_settlement"]
    assert Path(fused["combined_path"]).exists()
    assert fused["summary"]["n_parcels"] == 4
    assert fused["nrrp_report"] is not None
    # Kibera parcel 1 (box 5..6) sits in the reference and is resettle.
    assert fused["nrrp_report"]["per_class"]["compensated_resettlement"]["n_inside_reference"] >= 1


def test_combine_settlement_graphs_concatenates_pos():
    import torch
    from torch_geometric.data import Data
    from src.models.multi_settlement import combine_settlement_graphs
    from src.utils.deprivation_label import NODE_FEATURE_NAMES

    def _g(n, seed):
        rng = np.random.default_rng(seed)
        x = torch.from_numpy(rng.normal(size=(n, len(NODE_FEATURE_NAMES))).astype(np.float32))
        y = torch.zeros(n, dtype=torch.long)
        pos = torch.tensor([[float(i), 0.0] for i in range(n)], dtype=torch.float32)
        node_ids = torch.arange(n)
        ei = torch.tensor([[0], [0]], dtype=torch.long) if n == 1 else torch.tensor(
            [list(range(n - 1)) + list(range(1, n)), list(range(1, n)) + list(range(n - 1))],
            dtype=torch.long,
        )
        return Data(x=x, y=y, pos=pos, node_ids=node_ids, edge_index=ei)

    combined = combine_settlement_graphs({
        "kibera": _g(4, 1),
        "mathare": _g(3, 2),
    })
    assert combined.pos.shape == (7, 2)
    assert combined.node_ids.tolist() == [0, 1, 2, 3, 0, 1, 2]


def test_sprint5_notebook_parses_and_references_engine():
    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "05_sprint5_fusion_engine.ipynb"
    if not nb_path.exists():
        pytest.skip("notebook not written yet")
    nb = json.loads(nb_path.read_text())
    source = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert "FusionEngine" in source
    assert "policy_nrrp_report" in source
    assert "fuse_settlements" in source
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        nonempty = [ln for ln in text.splitlines() if ln.strip()]
        # Cell magics (%%bash, %cd) and shell cells are Colab-only.
        if nonempty and nonempty[0].lstrip().startswith(("%", "!")):
            continue
        cleaned = []
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("%") or stripped.startswith("!"):
                continue
            cleaned.append(line)
        body = "\n".join(cleaned)
        if not body.strip():
            continue
        compile(body, f"05_sprint5_fusion_engine.ipynb:cell{i}", "exec")
