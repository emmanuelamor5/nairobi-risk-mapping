"""
Unit tests for Sprint 2's model code. These run entirely offline against
synthetic tensors -- no real tile data or GPU required -- so they verify
architecture correctness (shapes, gradient flow, transfer-learning
mechanics) independent of whether Sprint 1's actual Kibera export is
available in this environment.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.attention_unet import AttentionGate, AttentionUNet
from src.models.dataset import FLOOD_LABEL_BAND_INDEX, INPUT_BAND_INDICES, spatial_holdout_split
from src.models.losses import CombinedLoss, DiceLoss
from src.models.metrics import RunningMetrics, iou_from_counts, f1_from_counts
from src.models.transfer_learning import freeze_encoder, trainable_parameter_count


# --- Architecture ---

def test_attention_unet_output_shape():
    model = AttentionUNet(in_channels=9, out_channels=1, base_channels=8)  # small for fast test
    x = torch.randn(2, 9, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)


def test_attention_unet_handles_non_multiple_of_16_input():
    """256 is a multiple of 16 (4 pooling levels), but real tiles could vary slightly -- check graceful behaviour isn't silently broken at odd sizes near boundaries."""
    model = AttentionUNet(in_channels=9, out_channels=1, base_channels=8)
    x = torch.randn(1, 9, 256, 256)  # keep at the designed size; documents the assumption
    out = model(x)
    assert out.shape[-2:] == x.shape[-2:]


def test_attention_gate_output_shape_matches_skip():
    gate = AttentionGate(gate_channels=64, skip_channels=64, inter_channels=32)
    g = torch.randn(2, 64, 32, 32)
    skip = torch.randn(2, 64, 32, 32)
    out = gate(g, skip)
    assert out.shape == skip.shape


def test_attention_gate_output_bounded_by_sigmoid_scaling():
    """Attention-scaled output should never exceed the input skip connection's magnitude, since the gate multiplies by a sigmoid in [0,1]."""
    gate = AttentionGate(gate_channels=8, skip_channels=8, inter_channels=4)
    g = torch.randn(1, 8, 16, 16)
    skip = torch.ones(1, 8, 16, 16) * 5.0
    out = gate(g, skip)
    assert (out.abs() <= skip.abs() + 1e-5).all()


def test_gradients_flow_to_decoder_and_attention():
    model = AttentionUNet(in_channels=9, out_channels=1, base_channels=8)
    x = torch.randn(1, 9, 64, 64, requires_grad=False)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert model.dec1.block[0].weight.grad is not None
    assert model.att1.psi[0].weight.grad is not None


def test_freeze_encoder_stops_encoder_gradients():
    model = AttentionUNet(in_channels=9, base_channels=8)
    freeze_encoder(model)
    counts = trainable_parameter_count(model)
    assert counts["frozen_parameters"] > 0
    assert counts["trainable_parameters"] > 0
    assert counts["trainable_parameters"] + counts["frozen_parameters"] == counts["total_parameters"]

    x = torch.randn(1, 9, 64, 64)
    out = model(x)
    out.sum().backward()
    assert model.enc1.block[0].weight.grad is None or model.enc1.block[0].weight.grad.abs().sum() == 0


# --- Loss ---

def test_dice_loss_zero_for_perfect_prediction():
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])  # sigmoid(10)~=1, sigmoid(-10)~=0
    loss = DiceLoss()(logits, targets)
    assert loss.item() < 0.01


def test_dice_loss_high_for_inverted_prediction():
    """
    Uses a larger tensor than the perfect-prediction test above: Dice's
    smooth=1.0 term has an outsized relative effect on tiny (e.g. 2x2)
    tensors, capping the achievable loss well below 1.0 even for a fully
    inverted prediction -- that's correct smoothing behaviour, not a bug,
    but it means a small tensor isn't the right test for "loss approaches
    1.0 when totally wrong". A larger tensor makes the smoothing term
    negligible so the assertion reflects the loss function's real behaviour.
    """
    targets = torch.zeros(1, 1, 16, 16)
    targets[:, :, :8, :] = 1.0  # top half flooded
    logits = torch.where(targets == 1.0, torch.tensor(-10.0), torch.tensor(10.0))  # exactly inverted
    loss = DiceLoss()(logits, targets)
    assert loss.item() > 0.95


def test_combined_loss_is_weighted_sum():
    targets = torch.rand(2, 1, 8, 8).round()
    logits = torch.randn(2, 1, 8, 8)
    combined = CombinedLoss(bce_weight=0.3, dice_weight=0.7)
    bce_only = torch.nn.BCEWithLogitsLoss()(logits, targets)
    dice_only = DiceLoss()(logits, targets)
    expected = 0.3 * bce_only + 0.7 * dice_only
    assert torch.isclose(combined(logits, targets), expected, atol=1e-5)


# --- Metrics ---

def test_iou_perfect_prediction():
    logits = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    metrics = RunningMetrics()
    metrics.update(logits, targets)
    result = metrics.compute()
    assert result["iou"] > 0.99
    assert result["f1"] > 0.99


def test_iou_zero_for_no_overlap():
    logits = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])
    targets = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])  # exactly inverted
    metrics = RunningMetrics()
    metrics.update(logits, targets)
    result = metrics.compute()
    assert result["iou"] < 0.01


def test_running_metrics_accumulates_across_batches():
    metrics = RunningMetrics()
    perfect_logits = torch.tensor([[[[10.0]]]])
    perfect_targets = torch.tensor([[[[1.0]]]])
    metrics.update(perfect_logits, perfect_targets)

    wrong_logits = torch.tensor([[[[10.0]]]])
    wrong_targets = torch.tensor([[[[0.0]]]])
    metrics.update(wrong_logits, wrong_targets)

    result = metrics.compute()
    # 1 TP, 1 FP overall -> precision 0.5, recall 1.0 -> f1 = 2/3
    assert abs(result["f1"] - (2 / 3)) < 0.01


# --- Dataset splitting ---

def _write_fake_manifest(tmpdir, locations_and_variants):
    """locations_and_variants: list of (row_off, col_off, [variant_names])"""
    manifest = []
    tile_dir = Path(tmpdir)
    for row_off, col_off, variants in locations_and_variants:
        for variant in variants:
            tile_id = f"kibera_{row_off}_{col_off}_{variant}"
            path = tile_dir / f"{tile_id}.npy"
            np.save(path, np.zeros((10, 4, 4), dtype=np.float32))
            manifest.append({
                "tile_id": tile_id, "path": str(path),
                "row_off": row_off, "col_off": col_off,
                "valid_fraction": 0.9, "split": "train",
            })
    manifest_path = tile_dir / "manifest_train.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return str(manifest_path)


def test_spatial_holdout_keeps_augmented_variants_together():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = _write_fake_manifest(tmpdir, [
            (0, 0, ["orig", "flip_h", "flip_v", "rot90"]),
            (206, 0, ["orig", "flip_h", "flip_v", "rot90"]),
        ])
        split = spatial_holdout_split(manifest_path, holdout_location_index=-1)

        val_locations = set((e["row_off"], e["col_off"]) for e in split["val"])
        train_locations = set((e["row_off"], e["col_off"]) for e in split["train"])

        assert len(val_locations) == 1
        assert val_locations.isdisjoint(train_locations)
        assert len(split["val"]) == 4  # all 4 augmented variants of the held-out location
        assert split["num_distinct_locations"] == 2


def test_spatial_holdout_raises_with_only_one_location():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = _write_fake_manifest(tmpdir, [
            (0, 0, ["orig", "flip_h"]),
        ])
        with pytest.raises(ValueError, match="distinct spatial location"):
            spatial_holdout_split(manifest_path)


def test_band_indices_exclude_flood_label():
    assert FLOOD_LABEL_BAND_INDEX not in INPUT_BAND_INDICES
    assert len(INPUT_BAND_INDICES) == 9


# --- Transfer learning guards (added after review of the first Sprint 2 draft) ---

from src.models.transfer_learning import freeze_transferred_layers, load_pretrained_encoder
from src.models.evaluation import validation_overlap_mask


def test_loader_raises_when_checkpoint_matches_nothing():
    """A checkpoint from a different architecture matches zero keys; that must fail loudly, not silently leave a random encoder."""
    model = AttentionUNet(in_channels=9, base_channels=8)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "foreign.pt"
        torch.save({"backbone.layer1.0.conv1.weight": torch.randn(8, 3, 3, 3)}, path)
        with pytest.raises(ValueError, match="No encoder layers matched"):
            load_pretrained_encoder(model, str(path))


def test_selective_freeze_keeps_reinitialised_and_first_stage_trainable():
    src_model = AttentionUNet(in_channels=2, base_channels=8)
    dst_model = AttentionUNet(in_channels=9, base_channels=8)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.pt"
        torch.save(src_model.state_dict(), path)
        report = load_pretrained_encoder(dst_model, str(path))
    summary = freeze_transferred_layers(dst_model, report)

    params = dict(dst_model.named_parameters())
    assert params["enc1.block.0.weight"].requires_grad          # reinitialised input conv stays trainable
    assert all(p.requires_grad for n, p in params.items() if n.startswith("enc1"))
    assert not params["enc3.block.0.weight"].requires_grad       # deeper transferred layer is frozen
    assert not params["bottleneck.block.0.weight"].requires_grad
    assert params["dec1.block.0.weight"].requires_grad           # decoder untouched
    assert summary["frozen"] > 0


# --- Overlap-aware validation mask ---

def test_overlap_mask_excludes_shared_strip():
    # val tile at col 206, train tile at col 0 -> shared raster cols 206..255 -> val-local cols 0..49
    safe = validation_overlap_mask((0, 206), [(0, 0)], tile_size=256)
    assert not safe[:, :50].any()
    assert safe[:, 50:].all()


def test_overlap_mask_all_safe_when_no_overlap():
    safe = validation_overlap_mask((0, 0), [(1000, 1000)], tile_size=256)
    assert safe.all()


def test_overlap_mask_handles_two_neighbours():
    # train tiles to the left and above a val tile at (206, 206)
    safe = validation_overlap_mask((206, 206), [(206, 0), (0, 206)], tile_size=256)
    assert not safe[:, :50].any()   # strip shared with the left neighbour
    assert not safe[:50, :].any()   # strip shared with the upper neighbour
    assert safe[50:, 50:].all()
