"""
Loads pretrained encoder weights from the Sen1Floods11 baseline model
(Bonafilia et al., 2020) into an AttentionUNet's encoder, per proposal Sec
"Model Training": "Pre-trained encoder weights from the Sen1Floods11
baseline model ... will be used to initialise the network's contracting
path. Only the decoder layers and attention gate parameters will be
trained from random initialisation."

IMPORTANT, read before assuming this "just works": Sen1Floods11's released
baseline models were trained on 2-channel Sentinel-1 input (VV, VH), not
the 9-channel stack this pipeline produces (6 Sentinel-2 optical bands +
elevation + slope + TWI). That channel-count mismatch means the very
first convolutional layer's weights CANNOT transfer directly -- a (64, 2,
3, 3) weight tensor has no valid mapping onto a (64, 9, 3, 3) layer.

This loader handles that explicitly rather than silently failing or
crashing: it loads every encoder layer whose weight shape matches exactly,
and reinitialises (does not transfer) any layer whose shape doesn't match
-- which in practice means only the first conv layer of enc1 gets a fresh
random init while everything deeper in the encoder transfers normally,
since deeper layers operate on already-abstracted feature maps whose
channel counts don't depend on the raw input modality.

This is a genuine partial-transfer-learning setup, not the full transfer
learning the proposal describes in its idealised form. It's worth stating
plainly in your write-up: full-encoder transfer from Sen1Floods11 assumes
Sentinel-1-only input, and this system's late-fusion design deliberately
feeds the U-Net optical + terrain features as well, which is why only the
deeper encoder layers transfer while the input-adapting first layer does
not. An alternative worth considering, if this partial transfer measures
as insufficient during Sprint 2 evaluation: restrict the U-Net's raster
input to Sentinel-1 alone (matching Sen1Floods11 exactly) and feed
optical/terrain features to the fusion layer instead of the U-Net's
input -- a bigger architecture change than a one-sprint fix, so flag it
now rather than mid-training if it comes to that.
"""

import logging
from pathlib import Path

import torch

from .attention_unet import AttentionUNet

logger = logging.getLogger(__name__)


def load_pretrained_encoder(model: AttentionUNet, checkpoint_path: str,
                             require_transfer: bool = True) -> dict:
    """
    Load encoder weights from a Sen1Floods11 (or any compatible) checkpoint
    into `model`'s encoder, skipping any layer whose shape doesn't match.

    `checkpoint_path` should point to a .pt/.pth state_dict file. Sen1Floods11's
    released weights are distributed via their project repository
    (https://github.com/cloudtostreet/Sen1Floods11) -- download separately
    and place locally; this function does not fetch them.

    Returns a report dict: {"transferred": [...], "reinitialised": [...],
    "missing_in_checkpoint": [...]} so you can see exactly what happened
    rather than assuming full transfer occurred.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Sen1Floods11 weights "
            f"must be downloaded separately from the project's release assets "
            f"(https://github.com/cloudtostreet/Sen1Floods11) -- this loader "
            f"only applies weights already on disk."
        )

    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint_state:  # some checkpoints wrap the state dict
        checkpoint_state = checkpoint_state["state_dict"]

    model_state = model.state_dict()
    encoder_param_names = {
        name for name, _ in model.named_parameters()
        if any(name.startswith(prefix) for prefix in ("enc1", "enc2", "enc3", "enc4", "bottleneck"))
    }
    # Also match encoder buffers (BatchNorm running stats)
    encoder_buffer_names = {
        name for name, _ in model.named_buffers()
        if any(name.startswith(prefix) for prefix in ("enc1", "enc2", "enc3", "enc4", "bottleneck"))
    }
    encoder_keys = encoder_param_names | encoder_buffer_names

    transferred, reinitialised, missing_in_checkpoint = [], [], []

    for key in encoder_keys:
        if key not in checkpoint_state:
            missing_in_checkpoint.append(key)
            continue
        checkpoint_tensor = checkpoint_state[key]
        model_tensor = model_state[key]
        if checkpoint_tensor.shape == model_tensor.shape:
            model_state[key] = checkpoint_tensor
            transferred.append(key)
        else:
            reinitialised.append(
                f"{key} (checkpoint={tuple(checkpoint_tensor.shape)}, "
                f"model={tuple(model_tensor.shape)})"
            )

    if require_transfer and len(transferred) == 0:
        raise ValueError(
            "No encoder layers matched between the checkpoint and AttentionUNet "
            "(0 transferred). This loader matches layers by exact state-dict key "
            "name AND shape, so a checkpoint from a differently-structured network "
            "(e.g. a ResNet-backed U-Net, which is how many published flood models "
            "are built) will match nothing. Continuing would leave a randomly "
            "initialised encoder that you believe is pretrained -- and, if frozen, "
            "a permanently random one. Either provide a checkpoint saved from this "
            "same AttentionUNet architecture, write a key-mapping for the checkpoint "
            "you have, or train from scratch by passing pretrained_checkpoint=None."
        )

    model.load_state_dict(model_state)

    report = {
        "transferred": transferred,
        "reinitialised": reinitialised,
        "missing_in_checkpoint": missing_in_checkpoint,
    }
    logger.info(
        "Encoder transfer: %d layers transferred, %d reinitialised due to shape "
        "mismatch (expected: the first conv layer, due to the 2-channel S1-only "
        "vs 9-channel optical+terrain input mismatch -- see this module's "
        "docstring), %d missing from checkpoint entirely.",
        len(transferred), len(reinitialised), len(missing_in_checkpoint),
    )
    if len(reinitialised) > 1:
        logger.warning(
            "More than one encoder layer failed to transfer (%d). Expected "
            "exactly one (the first conv layer). Inspect 'reinitialised' in "
            "the returned report -- this may mean the checkpoint's "
            "architecture differs more than expected from AttentionUNet's "
            "encoder, not just the input channel count.",
            len(reinitialised),
        )
    return report


def freeze_transferred_layers(model: AttentionUNet, report: dict,
                               keep_trainable_prefixes: tuple = ("enc1",)) -> dict:
    """
    Freeze only the encoder layers that actually received pretrained weights,
    leaving everything else trainable.

    Two groups stay trainable even though they sit in the encoder:
      1. Layers that were reinitialised because of a shape mismatch (the
         9-channel input conv). Freezing a randomly initialised layer would
         lock in a random projection of the input for the whole run.
      2. Any layer under keep_trainable_prefixes (default: the whole first
         encoder stage, enc1). That stage is the one adapting a different
         input modality (Sentinel-1 -> optical + terrain) to the deeper
         pretrained features, so it should keep learning; its pretrained
         batch-norm statistics describe the old input distribution anyway.

    Returns {"frozen": n_tensors, "trainable_in_encoder": n_tensors}.
    """
    transferred = set(report["transferred"])
    frozen = trainable_in_encoder = 0
    encoder_prefixes = ("enc1", "enc2", "enc3", "enc4", "bottleneck")
    for name, param in model.named_parameters():
        if not name.startswith(encoder_prefixes):
            continue
        keep = name.startswith(tuple(keep_trainable_prefixes)) or name not in transferred
        param.requires_grad = keep
        if keep:
            trainable_in_encoder += 1
        else:
            frozen += 1
    return {"frozen": frozen, "trainable_in_encoder": trainable_in_encoder}


def freeze_encoder(model: AttentionUNet) -> None:
    """
    Freeze encoder parameters so only the decoder and attention gates train,
    per proposal Sec 'Model Training'. Call after load_pretrained_encoder.
    """
    for param in model.encoder_parameters():
        param.requires_grad = False


def trainable_parameter_count(model: AttentionUNet) -> dict:
    """Diagnostic: confirm what's actually trainable after freezing, rather than assuming."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable,
            "frozen_parameters": total - trainable}
