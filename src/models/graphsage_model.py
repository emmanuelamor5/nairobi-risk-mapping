"""
GraphSAGEModel: wraps the two-layer GraphSAGE module with training,
inference, and (stubbed for Sprint 6) SHAP extraction methods, per
proposal Sec 'Initial Prototype Development': "a GraphSAGE Model class
wrapping the PyTorch Geometric model with equivalent methods."

Training hyper-parameters follow proposal Sec 3.2.3:
  - Adam, lr = 5e-3, weight_decay = 5e-4
  - Stratified 60/20/20 node split
  - Macro-F1 as the selection metric (Sprint 3 target >= 0.60)

SHAP integration is intentionally left as a documented stub -- that is
Sprint 6 scope (XAI + dashboard), not Sprint 3.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data

from .. import config
from ..utils.deprivation_label import NODE_FEATURE_NAMES, class_distribution
from .graphsage import GraphSAGE
from .graphsage_metrics import (
    classification_report_from_logits,
    stratified_node_split,
)
from .multi_settlement import (
    evaluate_graphsage_by_settlement,
    graphsage_threshold_report,
    stratified_multi_settlement_split,
)

logger = logging.getLogger(__name__)


class GraphSAGEModel:
    def __init__(
        self,
        in_channels: int = None,
        hidden_channels: int = 64,
        num_classes: int = None,
        dropout: float = 0.5,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes or config.DEPRIVATION_NUM_CLASSES
        self.in_channels = in_channels if in_channels is not None else len(NODE_FEATURE_NAMES)
        self.network = GraphSAGE(
            in_channels=self.in_channels,
            hidden_channels=hidden_channels,
            num_classes=self.num_classes,
            dropout=dropout,
        ).to(self.device)
        self.split_masks = None
        logger.info(
            "GraphSAGE initialised: in=%d hidden=%d classes=%d device=%s",
            self.in_channels, hidden_channels, self.num_classes, self.device,
        )

    def prepare_data(self, data: Data, seed: int = 42) -> Data:
        """
        Attach stratified train/val/test masks to a PyG ``Data`` object that
        already carries ``data.y`` deprivation labels. Returns the same
        object for chaining; masks are also stored on ``self.split_masks``.

        When ``data.settlement_id`` is present (Sprint 4 combined graph from
        ``multi_settlement.combine_settlement_graphs``), the split is
        stratified *within each settlement* so small settlements keep a
        representative val/test fold. Single-settlement Kibera graphs keep
        the Sprint 3 global stratified split.
        """
        if data.y is None:
            raise ValueError(
                "Graph data has no data.y deprivation labels. Run "
                "graph_utils.enrich_buildings_with_deprivation before "
                "graph_to_pyg_data, or attach labels manually."
            )
        if data.x.size(1) != self.in_channels:
            raise ValueError(
                f"Feature width {data.x.size(1)} does not match model "
                f"in_channels={self.in_channels} ({NODE_FEATURE_NAMES})."
            )

        if hasattr(data, "settlement_id") and data.settlement_id is not None:
            masks = stratified_multi_settlement_split(
                data.y.cpu().numpy(),
                data.settlement_id.cpu().numpy(),
                seed=seed,
            )
            logger.info(
                "Using per-settlement stratified split over %s",
                getattr(data, "settlement_keys", "?"),
            )
        else:
            masks = stratified_node_split(data.y.cpu().numpy(), seed=seed)

        data.train_mask = torch.tensor(masks["train"], dtype=torch.bool)
        data.val_mask = torch.tensor(masks["val"], dtype=torch.bool)
        data.test_mask = torch.tensor(masks["test"], dtype=torch.bool)
        self.split_masks = masks

        dist = class_distribution(data.y.cpu().numpy(), n_classes=self.num_classes)
        logger.info("Deprivation class distribution: %s", dist)
        logger.info(
            "Split sizes: train=%d val=%d test=%d",
            int(data.train_mask.sum()), int(data.val_mask.sum()), int(data.test_mask.sum()),
        )
        return data

    def fit(
        self,
        data: Data,
        epochs: int = 100,
        learning_rate: float = 5e-3,
        weight_decay: float = 5e-4,
        checkpoint_dir: str = "checkpoints",
        early_stopping_patience: int = 20,
        seed: int = 42,
    ) -> dict:
        """
        Train on ``data.train_mask``, select the best checkpoint by
        validation macro-F1. Defaults match the proposal's Adam / L2
        schedule. Full-batch (the Kibera graph fits in memory); no mini-batch
        neighbour sampler is needed at this settlement scale.
        """
        data = data.to(self.device)
        if not hasattr(data, "train_mask") or data.train_mask is None:
            data = self.prepare_data(data.cpu(), seed=seed).to(self.device)

        optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Class-balanced loss: high-severity nodes can be a minority even
        # after tertile binning when scores collide; inverse-frequency
        # weights keep the optimiser from ignoring them.
        y_train = data.y[data.train_mask]
        class_counts = torch.bincount(y_train, minlength=self.num_classes).float()
        class_weights = (class_counts.sum() / class_counts.clamp(min=1.0))
        class_weights = class_weights / class_weights.mean()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        best_val_macro_f1 = -1.0
        epochs_since_improvement = 0
        history = []
        best_path = Path(checkpoint_dir) / "graphsage_best.pt"

        for epoch in range(1, epochs + 1):
            self.network.train()
            optimizer.zero_grad()
            logits = self.network(data.x, data.edge_index)
            loss = criterion(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            train_report = classification_report_from_logits(
                logits[data.train_mask].detach(), data.y[data.train_mask],
                num_classes=self.num_classes,
            )
            self.network.eval()
            with torch.no_grad():
                val_logits = self.network(data.x, data.edge_index)
                val_loss = criterion(
                    val_logits[data.val_mask], data.y[data.val_mask],
                ).item()
                val_report = classification_report_from_logits(
                    val_logits[data.val_mask], data.y[data.val_mask],
                    num_classes=self.num_classes,
                )

            record = {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss),
                "train_macro_f1": train_report["macro_f1"],
                "val_macro_f1": val_report["macro_f1"],
                "val_accuracy": val_report["accuracy"],
                "val_per_class_f1": val_report["per_class_f1"],
            }
            history.append(record)
            logger.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f | "
                "train_macro_f1=%.4f val_macro_f1=%.4f",
                epoch, epochs, record["train_loss"], record["val_loss"],
                record["train_macro_f1"], record["val_macro_f1"],
            )

            if val_report["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_report["macro_f1"]
                epochs_since_improvement = 0
                torch.save(self.network.state_dict(), best_path)
                logger.info(
                    "New best val_macro_f1=%.4f -- checkpoint saved.",
                    best_val_macro_f1,
                )
            else:
                epochs_since_improvement += 1
                if (
                    early_stopping_patience is not None
                    and epochs_since_improvement >= early_stopping_patience
                ):
                    logger.info(
                        "Early stopping at epoch %d: val_macro_f1 has not "
                        "improved for %d epochs (best=%.4f).",
                        epoch, early_stopping_patience, best_val_macro_f1,
                    )
                    break

        if best_path.exists():
            self.load_checkpoint(str(best_path))

        meets_sprint3 = best_val_macro_f1 >= config.GRAPHSAGE_SPRINT3_MACRO_F1_TARGET
        meets_final = best_val_macro_f1 >= config.GRAPHSAGE_FINAL_MACRO_F1_TARGET
        result = {
            "history": history,
            "best_val_macro_f1": best_val_macro_f1,
            "stopped_epoch": history[-1]["epoch"] if history else 0,
            "meets_sprint3_target": meets_sprint3,
            "sprint3_target": config.GRAPHSAGE_SPRINT3_MACRO_F1_TARGET,
            # Sprint 4 / NFR bar -- same pooled val metric, stricter threshold.
            "meets_final_target": meets_final,
            "final_target": config.GRAPHSAGE_FINAL_MACRO_F1_TARGET,
            "checkpoint_path": str(best_path),
        }

        # When training on a combined three-settlement graph, also attach
        # per-settlement + pooled val reports so notebook 04 can print the
        # threshold table without a second forward pass setup.
        if hasattr(data, "settlement_id") and data.settlement_id is not None:
            by_settlement = evaluate_graphsage_by_settlement(
                self, data, split="val", num_classes=self.num_classes,
            )
            result["val_by_settlement"] = by_settlement
            result["threshold_report"] = graphsage_threshold_report(
                {
                    k: v["macro_f1"]
                    for k, v in by_settlement["per_settlement"].items()
                },
                by_settlement["pooled"]["macro_f1"],
            )
        return result

    @torch.no_grad()
    def predict(self, data: Data, mask: torch.Tensor = None) -> torch.Tensor:
        """Return predicted class indices for all nodes, or those under ``mask``."""
        self.network.eval()
        data = data.to(self.device)
        logits = self.network(data.x, data.edge_index)
        preds = logits.argmax(dim=-1)
        if mask is not None:
            return preds[mask.to(self.device)].cpu()
        return preds.cpu()

    @torch.no_grad()
    def predict_proba(self, data: Data, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Softmax class probabilities -- the vector the Fusion Engine (Sprint 5)
        concatenates with the U-Net flood probability.
        """
        self.network.eval()
        data = data.to(self.device)
        logits = self.network(data.x, data.edge_index)
        probs = F.softmax(logits, dim=-1)
        if mask is not None:
            return probs[mask.to(self.device)].cpu()
        return probs.cpu()

    def evaluate(self, data: Data, split: str = "test") -> dict:
        """Run classification_report on train/val/test mask."""
        mask_attr = f"{split}_mask"
        if not hasattr(data, mask_attr):
            raise ValueError(f"data has no {mask_attr}; call prepare_data / fit first.")
        self.network.eval()
        data = data.to(self.device)
        with torch.no_grad():
            logits = self.network(data.x, data.edge_index)
            mask = getattr(data, mask_attr)
            return classification_report_from_logits(
                logits[mask], data.y[mask], num_classes=self.num_classes,
            )

    def evaluate_by_settlement(self, data: Data, split: str = "val") -> dict:
        """
        Sprint 4: per-settlement + pooled macro-F1 on a combined graph.
        Requires ``data.settlement_id`` / ``data.settlement_keys`` from
        ``multi_settlement.combine_settlement_graphs``.
        """
        return evaluate_graphsage_by_settlement(
            self, data, split=split, num_classes=self.num_classes,
        )

    def explain(self, *args, **kwargs):
        """
        SHAP feature attribution for GraphSAGE node classifications --
        Sprint 6 scope per proposal Sec 'Iterative Refinement'. Stubbed so
        the public interface matches the OOAD design now.
        """
        raise NotImplementedError(
            "SHAP attribution is Sprint 6 scope, not Sprint 3. "
            "This method is stubbed to match the proposal's class design."
        )

    def load_checkpoint(self, checkpoint_path: str):
        self.network.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.network.to(self.device)
