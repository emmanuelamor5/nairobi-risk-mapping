"""
Offline unit tests for Sprint 3's GraphSAGE deprivation head. Synthetic
graphs only -- no OSM / GPU required.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.models.graphsage import GraphSAGE
from src.models.graphsage_metrics import (
    classification_report_from_logits,
    confusion_matrix_multiclass,
    macro_f1_from_cm,
    stratified_node_split,
)
from src.models.graphsage_model import GraphSAGEModel
from src.utils.deprivation_label import NODE_FEATURE_NAMES


def _synthetic_graph(n_nodes: int = 90, seed: int = 0) -> Data:
    """
    Build a small ring-ish graph with linearly separable-ish features so the
    model can overfit quickly in the smoke-train test. Labels are tertiles of
    a known linear combination of the features.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_nodes, len(NODE_FEATURE_NAMES))).astype(np.float32)
    # Plant a signal in the first three features.
    score = 0.4 * x[:, 0] + 0.3 * x[:, 1] + 0.3 * x[:, 2]
    cuts = np.percentile(score, [33.3, 66.6])
    y = np.digitize(score, cuts).astype(np.int64)

    # Chain edges plus a few random long-range links -- enough connectivity
    # for SAGEConv message passing without requiring a full proximity graph.
    src = list(range(n_nodes))
    dst = [(i + 1) % n_nodes for i in range(n_nodes)]
    for _ in range(n_nodes // 2):
        a, b = rng.integers(0, n_nodes, size=2)
        src.append(int(a))
        dst.append(int(b))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    return Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        y=torch.from_numpy(y),
    )


def test_graphsage_output_shape():
    model = GraphSAGE(in_channels=5, hidden_channels=16, num_classes=3)
    data = _synthetic_graph(40)
    out = model(data.x, data.edge_index)
    assert out.shape == (40, 3)


def test_graphsage_gradients_flow():
    model = GraphSAGE(in_channels=5, hidden_channels=8, num_classes=3)
    data = _synthetic_graph(30)
    loss = model(data.x, data.edge_index).sum()
    loss.backward()
    assert model.conv1.lin_l.weight.grad is not None
    assert model.classifier.weight.grad is not None


def test_stratified_node_split_is_exclusive_and_complete():
    labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
    masks = stratified_node_split(labels, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=1)
    assert masks["train"].sum() + masks["val"].sum() + masks["test"].sum() == 90
    assert not np.any(masks["train"] & masks["val"])
    assert not np.any(masks["train"] & masks["test"])
    assert not np.any(masks["val"] & masks["test"])
    # Each class should appear in every fold.
    for fold in ("train", "val", "test"):
        fold_labels = labels[masks[fold]]
        assert set(fold_labels.tolist()) == {0, 1, 2}


def test_macro_f1_perfect_and_chance():
    # Perfect diagonal.
    cm = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    assert macro_f1_from_cm(cm) == pytest.approx(1.0)
    # All predicted as class 0.
    cm_bias = np.array([[10, 0, 0], [10, 0, 0], [10, 0, 0]])
    f1 = macro_f1_from_cm(cm_bias)
    assert 0.0 <= f1 < 0.5


def test_classification_report_keys():
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    targets = torch.tensor([0, 1, 2])
    report = classification_report_from_logits(logits, targets, num_classes=3)
    assert report["macro_f1"] == pytest.approx(1.0)
    assert set(report["per_class_f1"]) == set(config.DEPRIVATION_CLASS_NAMES)
    assert confusion_matrix_multiclass(logits, targets).shape == (3, 3)


def test_graphsage_model_fit_smoke_and_checkpoint():
    data = _synthetic_graph(90, seed=7)
    model = GraphSAGEModel(in_channels=5, hidden_channels=32, num_classes=3, dropout=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        result = model.fit(
            data,
            epochs=40,
            learning_rate=5e-3,
            weight_decay=5e-4,
            checkpoint_dir=tmp,
            early_stopping_patience=15,
            seed=7,
        )
        assert "best_val_macro_f1" in result
        assert result["best_val_macro_f1"] >= 0.0
        assert Path(result["checkpoint_path"]).exists()

        # predict_proba returns a valid simplex over classes.
        probs = model.predict_proba(data)
        assert probs.shape == (90, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(90), atol=1e-5)

        test_report = model.evaluate(data, split="test")
        assert "macro_f1" in test_report


def test_graphsage_model_rejects_missing_labels():
    data = Data(
        x=torch.randn(10, 5),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    model = GraphSAGEModel()
    with pytest.raises(ValueError, match="data.y"):
        model.prepare_data(data)


def test_explain_stub_is_sprint6():
    model = GraphSAGEModel()
    with pytest.raises(NotImplementedError, match="Sprint 6"):
        model.explain()
