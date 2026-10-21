"""
Local explanations for the two Sprint 6 heads.

GraphSAGE: exact Shapley values of the predicted-class logit with respect
to one building's own features. Neighbours stay fixed, and a missing
feature is replaced by the mean of that feature across the graph. With
five features the power set is 32 evaluations, so the values are exact
for that local model rather than a sampled KernelSHAP approximation.

Attention U-Net: Grad-CAM on the last 3x3 convolution in the decoder
(``dec1``), which is the feature map the attention-gated skip path has
already filtered. In-place ReLUs are turned off for the backward pass
so the hook is not invalidated.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.deprivation_label import NODE_FEATURE_NAMES


def _subset_logit(predict, x_row: np.ndarray, baseline: np.ndarray, mask: int) -> float:
    row = baseline.copy()
    for j in range(x_row.shape[0]):
        if (mask >> j) & 1:
            row[j] = x_row[j]
    return float(predict(row))


def exact_feature_shapley(predict, x_row: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    Exact Shapley values of ``predict(row) -> float`` over the features of
    one row. ``baseline`` fills features that are absent from a coalition.
    """
    x_row = np.asarray(x_row, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    n = int(x_row.shape[0])
    if baseline.shape[0] != n:
        raise ValueError(f"baseline length {baseline.shape[0]} != features {n}")
    cache = {}

    def value(mask: int) -> float:
        if mask not in cache:
            cache[mask] = _subset_logit(predict, x_row, baseline, mask)
        return cache[mask]

    shap = np.zeros(n, dtype=np.float64)
    for mask in range(1 << n):
        size = int(mask.bit_count())
        if size == n:
            continue
        weight = (
            math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
        )
        base = value(mask)
        for j in range(n):
            if (mask >> j) & 1:
                continue
            shap[j] += weight * (value(mask | (1 << j)) - base)
    return shap


def explain_graphsage_node(model, data, node_index: int) -> dict:
    """
    Exact local SHAP of one building's deprivation features.

    ``model`` is a ``GraphSAGEModel``. The returned attributions sum to the
    difference between the predicted logit and the logit at the feature-mean
    baseline (the efficiency property).
    """
    if data.x is None or data.edge_index is None:
        raise ValueError("data needs x and edge_index.")
    node_index = int(node_index)
    if node_index < 0 or node_index >= int(data.num_nodes):
        raise ValueError(f"node_index {node_index} outside 0..{int(data.num_nodes) - 1}")

    network = model.network
    network.eval()
    device = model.device
    x = data.x.detach().to(device).float()
    edge_index = data.edge_index.detach().to(device)
    with torch.no_grad():
        logits = network(x, edge_index)[node_index]
        predicted = int(logits.argmax().item())
        full_logit = float(logits[predicted].item())

    baseline = x.mean(dim=0).detach().cpu().numpy()
    x_row = x[node_index].detach().cpu().numpy()
    names = list(getattr(data, "feature_names", None) or NODE_FEATURE_NAMES)

    def predict(row: np.ndarray) -> float:
        patched = x.clone()
        patched[node_index] = torch.as_tensor(row, dtype=patched.dtype, device=device)
        with torch.no_grad():
            out = network(patched, edge_index)[node_index, predicted]
        return float(out.item())

    attributions = exact_feature_shapley(predict, x_row, baseline)
    order = list(np.argsort(-np.abs(attributions)))
    return {
        "node_index": node_index,
        "predicted_class": predicted,
        "predicted_logit": full_logit,
        "baseline_logit": predict(baseline),
        "feature_names": names,
        "attributions": attributions,
        "ranked_features": [names[i] if i < len(names) else str(i) for i in order],
    }


def _set_inplace_relu(module: nn.Module, enabled: bool) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.ReLU):
            layer.inplace = enabled


def grad_cam_decoder(network: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    """
    Grad-CAM heatmap for a flood-logit map, shape (H, W), values in [0, 1].

    The target layer is the last 3x3 convolution in ``dec1``. Channel
    weights are the spatial mean of the gradient of the summed flood logit.
    """
    if not hasattr(network, "dec1"):
        raise TypeError("grad_cam_decoder expects an AttentionUNet with dec1.")
    target = network.dec1.block[3]
    if not isinstance(target, nn.Conv2d):
        raise TypeError("dec1.block[3] is no longer the last 3x3 convolution.")

    network.eval()
    _set_inplace_relu(network, False)
    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output
        output.retain_grad()

    handle = target.register_forward_hook(forward_hook)
    try:
        batch = inputs.detach().requires_grad_(True)
        logits = network(batch)
        score = logits.sum()
        network.zero_grad(set_to_none=True)
        score.backward()
        gradients["value"] = activations["value"].grad
    finally:
        handle.remove()
        _set_inplace_relu(network, True)

    acts = activations["value"].detach()
    grads = gradients["value"]
    if grads is None:
        raise RuntimeError("Grad-CAM received no gradient on the decoder convolution.")
    weights = grads.detach().mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
    heat = cam[0, 0].detach().cpu().numpy()
    span = float(heat.max() - heat.min())
    if span > 0:
        heat = (heat - float(heat.min())) / span
    else:
        heat = np.zeros_like(heat)
    return heat.astype(np.float64)
