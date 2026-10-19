"""
Node-classification metrics for the GraphSAGE deprivation head.

Sprint 3's exit criterion (proposal Sec 3.4.3 weeks 5-6) is macro-F1 >= 0.60
on the Kibera validation split. The final NFR (NFR-01) raises that to
macro-F1 >= 0.70 across all three settlements in Sprint 4 -- see
``multi_settlement.evaluate_graphsage_by_settlement`` for the
per-settlement + pooled reporting path. Macro-F1 is used rather than
accuracy / micro-F1 because it weights the three deprivation severity
classes equally -- the high-severity class is expected to be a minority,
and majority-class collapse would otherwise look like success.
"""

from __future__ import annotations

import numpy as np
import torch

from .. import config


@torch.no_grad()
def confusion_matrix_multiclass(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = None,
) -> np.ndarray:
    """Return a (C, C) confusion matrix; rows = true, cols = predicted."""
    num_classes = num_classes or config.DEPRIVATION_NUM_CLASSES
    preds = logits.argmax(dim=-1)
    y_true = targets.view(-1).cpu().numpy().astype(np.int64)
    y_pred = preds.view(-1).cpu().numpy().astype(np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def per_class_f1_from_cm(cm: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Per-class F1 from a confusion matrix (rows = true, cols = pred)."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    precision = tp / (cm.sum(axis=0) + eps)
    recall = tp / (cm.sum(axis=1) + eps)
    return 2 * precision * recall / (precision + recall + eps)


def macro_f1_from_cm(cm: np.ndarray, eps: float = 1e-9) -> float:
    """Unweighted mean of per-class F1 scores."""
    return float(per_class_f1_from_cm(cm, eps).mean())


def accuracy_from_cm(cm: np.ndarray, eps: float = 1e-9) -> float:
    total = cm.sum()
    return float(np.diag(cm).sum() / (total + eps))


def classification_report_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = None,
) -> dict:
    """Bundle macro-F1, accuracy, per-class F1, and the raw confusion matrix."""
    num_classes = num_classes or config.DEPRIVATION_NUM_CLASSES
    cm = confusion_matrix_multiclass(logits, targets, num_classes)
    per_class = per_class_f1_from_cm(cm)
    names = list(config.DEPRIVATION_CLASS_NAMES[:num_classes])
    return {
        "macro_f1": macro_f1_from_cm(cm),
        "accuracy": accuracy_from_cm(cm),
        "per_class_f1": {names[i]: float(per_class[i]) for i in range(num_classes)},
        "confusion_matrix": cm.tolist(),
        "support": {names[i]: int(cm[i].sum()) for i in range(num_classes)},
    }


def stratified_node_split(
    labels,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Stratified 60/20/20 node-index split (proposal Sec 3.2.4). Stratification
    keeps each deprivation class proportionally represented so a minority
    high-severity class cannot vanish from the validation fold.

    Returns boolean masks ``{"train", "val", "test"}`` over the node axis.
    Fractions must sum to 1.0 (within floating tolerance).
    """
    y = np.asarray(labels, dtype=np.int64)
    n = y.size
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    rng = np.random.default_rng(seed)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_cls = idx.size
        n_train = int(round(n_cls * train_frac))
        n_val = int(round(n_cls * val_frac))
        # Assign remainder to test so every node lands in exactly one fold.
        n_test = n_cls - n_train - n_val
        if n_test < 0:
            # Rounding edge case on tiny classes: shrink val first.
            n_val = max(0, n_val + n_test)
            n_test = n_cls - n_train - n_val

        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True

    # Guarantee mutual exclusivity (defensive -- should already hold).
    assert not np.any(train_mask & val_mask)
    assert not np.any(train_mask & test_mask)
    assert not np.any(val_mask & test_mask)
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == n

    return {"train": train_mask, "val": val_mask, "test": test_mask}
