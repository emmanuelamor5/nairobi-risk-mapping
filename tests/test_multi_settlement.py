"""
Offline unit tests for Sprint 4 multi-settlement combine / split / metrics
helpers. Synthetic graphs and tile manifests only -- no GEE / OSM / Drive.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.models.graphsage_model import GraphSAGEModel
from src.models.multi_settlement import (
    aggregate_flood_counts,
    combine_settlement_graphs,
    flood_threshold_report,
    graphsage_threshold_report,
    group_entries_by_settlement,
    load_multi_settlement_manifests,
    load_settlement_graphs,
    multi_settlement_spatial_holdout,
    prepare_multi_settlement_data,
    stratified_multi_settlement_split,
    _prefer_orig_entries,
)
from src.utils.deprivation_label import NODE_FEATURE_NAMES


def _synth_graph(n_nodes: int, seed: int, label_shift: float = 0.0) -> Data:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_nodes, len(NODE_FEATURE_NAMES))).astype(np.float32)
    score = 0.4 * x[:, 0] + 0.3 * x[:, 1] + 0.3 * x[:, 2] + label_shift
    cuts = np.percentile(score, [33.3, 66.6])
    y = np.digitize(score, cuts).astype(np.int64)
    src = list(range(n_nodes))
    dst = [(i + 1) % n_nodes for i in range(n_nodes)]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return Data(x=torch.from_numpy(x), edge_index=edge_index, y=torch.from_numpy(y))


def test_combine_settlement_graphs_is_disjoint_union():
    graphs = {
        "kibera": _synth_graph(30, seed=1),
        "mathare": _synth_graph(20, seed=2),
        "mukuru": _synth_graph(25, seed=3),
    }
    combined = combine_settlement_graphs(graphs)
    assert combined.num_nodes == 75
    assert combined.settlement_keys == ["kibera", "mathare", "mukuru"]
    assert combined.settlement_id.tolist() == (
        [0] * 30 + [1] * 20 + [2] * 25
    )
    # No cross-settlement edges: every endpoint stays inside its block.
    ei = combined.edge_index
    sid = combined.settlement_id
    assert torch.all(sid[ei[0]] == sid[ei[1]])
    assert combined.node_offset == {"kibera": 0, "mathare": 30, "mukuru": 50}


def test_stratified_multi_settlement_split_preserves_each_settlement():
    labels = np.array([0, 1, 2] * 20 + [0, 1, 2] * 10)  # 60 + 30
    sids = np.array([0] * 60 + [1] * 30)
    masks = stratified_multi_settlement_split(labels, sids, seed=0)
    assert masks["train"].sum() + masks["val"].sum() + masks["test"].sum() == 90
    for sid in (0, 1):
        idx = sids == sid
        # Each settlement keeps nodes in every fold.
        assert masks["train"][idx].any()
        assert masks["val"][idx].any()
        assert masks["test"][idx].any()
        # Class coverage inside each settlement's train fold.
        assert set(labels[idx & masks["train"]].tolist()) == {0, 1, 2}


def test_prepare_and_fit_multi_settlement_smoke():
    graphs = {
        "kibera": _synth_graph(60, seed=4),
        "mathare": _synth_graph(45, seed=5, label_shift=0.2),
        "mukuru": _synth_graph(45, seed=6, label_shift=-0.1),
    }
    data = prepare_multi_settlement_data(graphs, seed=11)
    model = GraphSAGEModel(in_channels=5, hidden_channels=32, num_classes=3, dropout=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        result = model.fit(
            data,
            epochs=25,
            checkpoint_dir=tmp,
            early_stopping_patience=20,
            seed=11,
        )
        assert "val_by_settlement" in result
        assert set(result["val_by_settlement"]["per_settlement"]) == {
            "kibera", "mathare", "mukuru",
        }
        assert "pooled" in result["val_by_settlement"]
        assert "threshold_report" in result
        assert result["final_target"] == config.GRAPHSAGE_FINAL_MACRO_F1_TARGET

        by_s = model.evaluate_by_settlement(data, split="test")
        assert "pooled" in by_s
        assert len(by_s["per_settlement"]) == 3


def test_load_settlement_graphs_skips_missing(tmp_path):
    g = _synth_graph(12, seed=9)
    torch.save(g, tmp_path / "kibera_graph.pt")
    loaded = load_settlement_graphs(tmp_path, settlement_keys=["kibera", "mathare"])
    assert list(loaded) == ["kibera"]


def test_graphsage_threshold_report_flags():
    report = graphsage_threshold_report(
        {"kibera": 0.80, "mathare": 0.65, "mukuru": 0.72},
        pooled_macro_f1=0.71,
        target=0.70,
    )
    assert report["meets_pooled_target"] is True
    assert report["meets_per_settlement_target"]["mathare"] is False
    assert report["all_settlements_meet_target"] is False


def test_multi_settlement_spatial_holdout_no_cross_location_leak():
    manifests = {
        "kibera": [
            {"tile_id": "kibera_0_0_orig", "path": "a.npy", "row_off": 0, "col_off": 0},
            {"tile_id": "kibera_0_0_flip_h", "path": "b.npy", "row_off": 0, "col_off": 0},
            {"tile_id": "kibera_206_0_orig", "path": "c.npy", "row_off": 206, "col_off": 0},
            {"tile_id": "kibera_412_0_orig", "path": "d.npy", "row_off": 412, "col_off": 0},
            {"tile_id": "kibera_618_0_orig", "path": "e.npy", "row_off": 618, "col_off": 0},
        ],
        "mathare": [
            {"tile_id": "mathare_0_0_orig", "path": "f.npy", "row_off": 0, "col_off": 0},
            {"tile_id": "mathare_206_0_orig", "path": "g.npy", "row_off": 206, "col_off": 0},
            {"tile_id": "mathare_412_0_orig", "path": "h.npy", "row_off": 412, "col_off": 0},
            {"tile_id": "mathare_618_0_orig", "path": "i.npy", "row_off": 618, "col_off": 0},
        ],
    }
    split = multi_settlement_spatial_holdout(manifests, holdout_fraction=0.25, seed=1)

    # Augmented variants of the same Kibera location stay together.
    kibera_train_locs = {
        (e["row_off"], e["col_off"])
        for e in split["train"] if e["settlement_key"] == "kibera"
    }
    kibera_val_locs = {
        (e["row_off"], e["col_off"])
        for e in split["val"] if e["settlement_key"] == "kibera"
    }
    assert kibera_train_locs.isdisjoint(kibera_val_locs)
    # Both variants of (0,0) must be on the same side.
    train_ids = {e["tile_id"] for e in split["train"]}
    val_ids = {e["tile_id"] for e in split["val"]}
    both_train = "kibera_0_0_orig" in train_ids and "kibera_0_0_flip_h" in train_ids
    both_val = "kibera_0_0_orig" in val_ids and "kibera_0_0_flip_h" in val_ids
    assert both_train or both_val

    # Same numeric offsets across settlements are independent holdouts.
    assert "kibera" in split["holdout_locations"]
    assert "mathare" in split["holdout_locations"]
    assert split["num_settlements"] == 2


def test_multi_settlement_spatial_holdout_raises_on_single_location():
    with pytest.raises(ValueError, match="distinct spatial location"):
        multi_settlement_spatial_holdout({
            "kibera": [
                {"tile_id": "k_0_0_orig", "path": "a.npy", "row_off": 0, "col_off": 0},
            ],
        })


def test_load_multi_settlement_manifests(tmp_path):
    for key in ("kibera", "mathare"):
        d = tmp_path / key
        d.mkdir()
        manifest = [{
            "tile_id": f"{key}_0_0_orig",
            "path": str(d / "t.npy"),
            "row_off": 0,
            "col_off": 0,
            "valid_fraction": 0.9,
            "split": "train",
        }]
        with open(d / "manifest_train.json", "w") as f:
            json.dump(manifest, f)
    loaded = load_multi_settlement_manifests(
        tmp_path, settlement_keys=["kibera", "mathare", "mukuru"],
    )
    assert set(loaded) == {"kibera", "mathare"}
    assert loaded["kibera"][0]["settlement_key"] == "kibera"


def test_aggregate_flood_counts_and_threshold_report():
    agg = aggregate_flood_counts({
        "kibera": {"tp": 80, "fp": 10, "fn": 10, "tn": 100},
        "mathare": {"tp": 40, "fp": 5, "fn": 15, "tn": 80},
        "mukuru": {"tp": 20, "fp": 20, "fn": 20, "tn": 60},
    })
    assert set(agg["per_settlement"]) == {"kibera", "mathare", "mukuru"}
    assert agg["pooled"]["counts"]["tp"] == 140.0
    # Pooled IoU from summed counts, not mean of IoUs.
    expected_iou = 140 / (140 + 35 + 45)
    assert agg["pooled"]["iou"] == pytest.approx(expected_iou)

    report = flood_threshold_report(
        {k: {"iou": v["iou"], "f1": v["f1"]} for k, v in agg["per_settlement"].items()},
        pooled_iou=agg["pooled"]["iou"],
        pooled_f1=agg["pooled"]["f1"],
        iou_target=0.60,
        f1_target=0.75,
    )
    assert "meets_pooled_iou_target" in report
    assert report["iou_target"] == 0.60
    assert report["f1_target"] == 0.75


def test_prefer_orig_entries_drops_augments_when_orig_present():
    entries = [
        {"tile_id": "k_0_0_orig", "row_off": 0, "col_off": 0, "settlement_key": "kibera"},
        {"tile_id": "k_0_0_flip_h", "row_off": 0, "col_off": 0, "settlement_key": "kibera"},
        {"tile_id": "k_1_0_flip_v", "row_off": 1, "col_off": 0, "settlement_key": "kibera"},
    ]
    selected = _prefer_orig_entries(entries)
    ids = {e["tile_id"] for e in selected}
    assert ids == {"k_0_0_orig", "k_1_0_flip_v"}


def test_group_entries_requires_settlement_key():
    with pytest.raises(ValueError, match="settlement_key"):
        group_entries_by_settlement([{"tile_id": "x", "row_off": 0, "col_off": 0}])


def test_config_sprint4_targets_exist():
    assert config.GRAPHSAGE_FINAL_MACRO_F1_TARGET == pytest.approx(0.70)
    assert config.ATTENTION_UNET_TRANSFER_IOU_TARGET == pytest.approx(0.60)
    assert config.ATTENTION_UNET_FINAL_F1_TARGET == pytest.approx(0.75)
    assert tuple(config.SETTLEMENT_KEYS) == ("kibera", "mathare", "mukuru")
