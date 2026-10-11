"""
Segmentation metrics for the Attention U-Net flood head.

Sprint 2's exit criterion (proposal Sec 'Iterative Refinement') is IoU >= 0.60
as a transfer-learning convergence check. The proposal's final validation
bar (Sec 'Model Validation and Testing') is F1-score >= 0.75. Both are
implemented here since Sprint 2 should report both, not just the
convergence-check metric.
"""

import torch


@torch.no_grad()
def compute_confusion_counts(logits: torch.Tensor, targets: torch.Tensor,
                              threshold: float = 0.5) -> dict:
    """Pixel-level TP/FP/FN/TN counts across a batch, at a given probability threshold."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def iou_from_counts(counts: dict, eps: float = 1e-7) -> float:
    """Intersection over Union = TP / (TP + FP + FN)."""
    return counts["tp"] / (counts["tp"] + counts["fp"] + counts["fn"] + eps)


def f1_from_counts(counts: dict, eps: float = 1e-7) -> float:
    """F1 = 2*TP / (2*TP + FP + FN) -- equivalent to the harmonic mean of precision and recall."""
    return (2 * counts["tp"]) / (2 * counts["tp"] + counts["fp"] + counts["fn"] + eps)


def dice_from_counts(counts: dict, eps: float = 1e-7) -> float:
    """Dice coefficient -- numerically identical to F1 for binary segmentation, kept separate for clarity when reporting."""
    return f1_from_counts(counts, eps)


def precision_recall_from_counts(counts: dict, eps: float = 1e-7) -> dict:
    precision = counts["tp"] / (counts["tp"] + counts["fp"] + eps)
    recall = counts["tp"] / (counts["tp"] + counts["fn"] + eps)
    return {"precision": precision, "recall": recall}


class RunningMetrics:
    """
    Accumulates confusion counts across an entire epoch's batches (rather
    than averaging per-batch metrics, which is biased when batches have
    very different numbers of positive pixels -- accumulating raw counts
    first and computing metrics once at the end is the statistically
    correct way to aggregate IoU/F1 across a variably-imbalanced dataset).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

    def update(self, logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
        counts = compute_confusion_counts(logits, targets, threshold)
        for k in self.totals:
            self.totals[k] += counts[k]

    def compute(self) -> dict:
        return {
            "iou": iou_from_counts(self.totals),
            "f1": f1_from_counts(self.totals),
            "dice": dice_from_counts(self.totals),
            **precision_recall_from_counts(self.totals),
        }
