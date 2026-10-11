"""
AttentionUNetModel: wraps the AttentionUNet PyTorch module with training,
inference, and (stubbed for Sprint 6) Captum XAI extraction methods, per
proposal Sec 'Initial Prototype Development': "an Attention U Net class
wrapping the PyTorch model with methods for training, inference, and
Captum XAI extraction."

Captum integration is intentionally left as a documented stub here --
that's explicitly Sprint 6 scope (XAI integration and dashboard
development), not Sprint 2. Building it now would be speculative, since
Sprint 6 may want Grad-CAM specifically targeting the attention gates
(per proposal Sec 'Model Training' rationale: "verifying that the network
is attending to hydrologically relevant regions"), which is easier to
design well once the trained model's actual attention behaviour is known.
"""

import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .attention_unet import AttentionUNet
from .dataset import FloodTileDataset
from .losses import CombinedLoss
from .metrics import RunningMetrics
from .transfer_learning import freeze_transferred_layers, load_pretrained_encoder, trainable_parameter_count

logger = logging.getLogger(__name__)


class AttentionUNetModel:
    def __init__(self, in_channels: int = 9, base_channels: int = 32,
                 device: str = None, pretrained_checkpoint: str = None,
                 freeze_pretrained_encoder: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.network = AttentionUNet(in_channels=in_channels, base_channels=base_channels).to(self.device)
        self.transfer_report = None

        if pretrained_checkpoint is not None:
            self.transfer_report = load_pretrained_encoder(self.network, pretrained_checkpoint)
            if freeze_pretrained_encoder:
                freeze_summary = freeze_transferred_layers(self.network, self.transfer_report)
                logger.info("Selective freeze applied: %s", freeze_summary)
        else:
            logger.warning(
                "No pretrained_checkpoint provided -- training the full network "
                "from random initialisation, NOT the transfer-learning setup the "
                "proposal specifies. This is a materially different, harder "
                "training problem given the small Kibera tile count; expect "
                "convergence to be slower or the IoU >= 0.60 target to be harder "
                "to hit. Source Sen1Floods11 weights before treating Sprint 2 "
                "results as representative of the intended design."
            )

        param_counts = trainable_parameter_count(self.network)
        logger.info("Model initialised: %s", param_counts)

    def fit(self, train_manifest_entries: list, val_manifest_entries: list,
            epochs: int = 50, batch_size: int = 2, learning_rate: float = 1e-3,
            checkpoint_dir: str = "checkpoints"):
        """
        Trains the network per proposal Sec 'Model Training': Adam optimiser,
        initial LR 1e-3 (default here matches), cosine annealing schedule,
        combined BCE+Dice loss.

        batch_size defaults to 2 given Sprint 1's small Kibera tile count --
        raise it once Sprint 4 brings in Mathare/Mukuru and genuinely more
        tiles exist to batch over.
        """
        train_dataset = _ManifestSubsetDataset(train_manifest_entries)
        val_dataset = _ManifestSubsetDataset(val_manifest_entries)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.network.parameters()),
            lr=learning_rate,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = CombinedLoss()

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        best_val_iou = -1.0
        history = []

        for epoch in range(1, epochs + 1):
            self.network.train()
            train_loss_total = 0.0
            train_metrics = RunningMetrics()

            for batch in train_loader:
                inputs = batch["inputs"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = self.network(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss_total += loss.item() * inputs.size(0)
                train_metrics.update(logits.detach(), labels)

            scheduler.step()

            val_loss_total = 0.0
            val_metrics = RunningMetrics()
            self.network.eval()
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["inputs"].to(self.device)
                    labels = batch["label"].to(self.device)
                    logits = self.network(inputs)
                    loss = criterion(logits, labels)
                    val_loss_total += loss.item() * inputs.size(0)
                    val_metrics.update(logits, labels)

            train_results = train_metrics.compute()
            val_results = val_metrics.compute()
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss_total / max(len(train_dataset), 1),
                "val_loss": val_loss_total / max(len(val_dataset), 1),
                "train_iou": train_results["iou"],
                "val_iou": val_results["iou"],
                "val_f1": val_results["f1"],
            }
            history.append(epoch_record)
            logger.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f | train_iou=%.4f "
                "val_iou=%.4f val_f1=%.4f",
                epoch, epochs, epoch_record["train_loss"], epoch_record["val_loss"],
                epoch_record["train_iou"], epoch_record["val_iou"], epoch_record["val_f1"],
            )

            if val_results["iou"] > best_val_iou:
                best_val_iou = val_results["iou"]
                torch.save(self.network.state_dict(), Path(checkpoint_dir) / "attention_unet_best.pt")
                logger.info("New best val_iou=%.4f -- checkpoint saved.", best_val_iou)

        return {"history": history, "best_val_iou": best_val_iou}

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Inference: returns binary flood predictions (0/1) for a batch of input tensors."""
        self.network.eval()
        inputs = inputs.to(self.device)
        logits = self.network(inputs)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float().cpu()

    @torch.no_grad()
    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        """Inference: returns flood probability maps in [0,1], for use by the Fusion Engine (Sprint 5)."""
        self.network.eval()
        inputs = inputs.to(self.device)
        logits = self.network(inputs)
        return torch.sigmoid(logits).cpu()

    def explain(self, *args, **kwargs):
        """
        Captum XAI extraction (Grad-CAM over the attention gates) -- Sprint 6
        scope per proposal Sec 'Iterative Refinement'. Not implemented yet;
        stubbed here so the class's public interface matches the proposal's
        OOAD design now, with the actual Captum wiring deferred until the
        model is trained and Sprint 6 begins.
        """
        raise NotImplementedError(
            "Captum Grad-CAM extraction is Sprint 6 scope, not Sprint 2. "
            "This method is stubbed to match the proposal's class design "
            "ahead of time."
        )

    def load_checkpoint(self, checkpoint_path: str):
        self.network.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.network.to(self.device)


class _ManifestSubsetDataset(FloodTileDataset):
    """Wraps a list of already-selected manifest entries (e.g. from spatial_holdout_split) as a Dataset, bypassing the file-path constructor."""

    def __init__(self, manifest_entries: list):
        self.manifest_path = None
        self.manifest = manifest_entries
