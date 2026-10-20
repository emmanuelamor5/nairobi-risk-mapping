"""
Sprint 5 Late-Fusion Decision Engine and NRRP Policy Validation.

Proposal Sec 3.2.3 / Sec 3.4.3 (weeks 9-10) / Sec 3.6.1 / FR-05:
  1. Combine parcel-level flood hazard probabilities from the Attention U-Net
     with 3-class structural deprivation probability vectors from GraphSAGE.
  2. Apply the rule-based 3x3 policy classification matrix to assign each
     parcel to one of three governance categories:
       - 0: Retain and Formalise
       - 1: Retain and Upgrade
       - 2: Compensated Resettlement
  3. Stitch overlapping 256x256 tile predictions back onto the settlement
     raster grid and sample flood probabilities at building footprints /
     parcel centroids.
  4. Validate the three-class parcel policy classifications against the
     Nairobi Rivers Regeneration Programme (NRRP) high-risk riparian zone
     reference polygons (or an OSM river-corridor buffer fallback) using
     concentration / lift statistics and policy distribution breakdowns.
  5. Serialise the policy-classified parcel GeoJSON layer per settlement
     (to ``outputs/classifications/<settlement>_policy_parcels.geojson``)
     matching the Logical Database Schema (proposal Sec 4.4.1).

The pure-numpy decision matrix and validation metrics live at the top of
this module with no GIS / Earth Engine dependencies so they can be unit-tested
offline.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Mapping, Sequence, Union

import numpy as np
import torch

from .. import config
from ..utils.label_validation import agreement_metrics
from .dataset import DEFAULT_LAYOUT, StackLayout

logger = logging.getLogger(__name__)


# Default 3x3 rule-based policy matrix:
#   rows = flood hazard regime (0: low, 1: moderate, 2: high)
#   cols = structural deprivation class (0: low, 1: medium, 2: high)
#   entries = policy class index:
#     0 -> Retain and Formalise
#     1 -> Retain and Upgrade
#     2 -> Compensated Resettlement
#
# Governance rationale (proposal Sec 2.4 & Sec 3.2.3):
#   - Safe ground with low deprivation -> Retain and Formalise (tenure security)
#   - Safe ground with medium/high deprivation -> Retain and Upgrade (in-situ
#     infrastructure / WASH / housing upgrading)
#   - Moderate flood exposure with low/medium deprivation -> Retain and Upgrade
#     (targeted storm-drainage investment & raised pathways)
#   - Moderate flood exposure compounded by severe (high) structural deprivation,
#     or acute (high) flood exposure regardless of structure -> Compensated
#     Resettlement (hazard cannot be safely mitigated in-situ)
DEFAULT_POLICY_MATRIX = (
    (0, 1, 1),  # low flood:      low->Formalise, med->Upgrade, high->Upgrade
    (1, 1, 2),  # moderate flood: low->Upgrade,   med->Upgrade, high->Resettlement
    (2, 2, 2),  # high flood:     all -> Compensated Resettlement
)


def _resolve_flood_thresholds(low_threshold: float = None, high_threshold: float = None):
    low_t = (
        config.FUSION_FLOOD_LOW_THRESHOLD
        if low_threshold is None
        else float(low_threshold)
    )
    high_t = (
        config.FUSION_FLOOD_HIGH_THRESHOLD
        if high_threshold is None
        else float(high_threshold)
    )
    if not (0.0 <= low_t < high_t <= 1.0):
        raise ValueError(
            f"Thresholds must satisfy 0 <= low_threshold < high_threshold <= 1; "
            f"got low={low_t}, high={high_t}."
        )
    return low_t, high_t


def discretize_flood_probability(
    flood_prob: np.ndarray,
    low_threshold: float = None,
    high_threshold: float = None,
) -> np.ndarray:
    """
    Map continuous flood probabilities in [0, 1] into three hazard regimes:
      - 0 (low):      p <  low_threshold
      - 1 (moderate): low_threshold <= p < high_threshold
      - 2 (high):     p >= high_threshold
    """
    low_t, high_t = _resolve_flood_thresholds(low_threshold, high_threshold)
    p = np.clip(np.nan_to_num(np.asarray(flood_prob, dtype=np.float64), nan=0.0), 0.0, 1.0)
    level = np.zeros(p.shape, dtype=np.int64)
    level[p >= low_t] = 1
    level[p >= high_t] = 2
    return level


def flood_regime_membership(
    flood_prob: np.ndarray,
    low_threshold: float = None,
    high_threshold: float = None,
) -> np.ndarray:
    """
    Soft 3-regime membership ``(..., 3)`` over (low, moderate, high) flood
    hazard for each parcel's scalar flood probability ``p``.

    Used alongside the hard rule-based matrix to compute a continuous
    decision confidence score in ``[0, 1]`` that reflects how strongly a
    parcel sits inside its assigned flood regime rather than near a decision
    boundary. Each row sums to 1; the winning regime's mass is in
    ``[0.5, 1.0]``.
    """
    low_t, high_t = _resolve_flood_thresholds(low_threshold, high_threshold)
    p = np.clip(np.nan_to_num(np.asarray(flood_prob, dtype=np.float64), nan=0.0), 0.0, 1.0)
    orig_shape = p.shape
    flat = p.ravel()
    mem = np.zeros((flat.size, 3), dtype=np.float64)

    mid_t = 0.5 * (low_t + high_t)
    half_span = max(0.5 * (high_t - low_t), 1e-6)

    low_mask = flat < low_t
    high_mask = flat >= high_t
    mid_mask = ~(low_mask | high_mask)

    w_low = 1.0 - 0.5 * (flat[low_mask] / max(low_t, 1e-6))
    mem[low_mask, 0] = w_low
    mem[low_mask, 1] = 1.0 - w_low

    dist = np.abs(flat[mid_mask] - mid_t) / half_span
    w_mod = np.maximum(0.5, 1.0 - 0.5 * dist)
    mem[mid_mask, 1] = w_mod
    below_mid = mid_mask & (flat < mid_t)
    above_mid = mid_mask & (flat >= mid_t)
    mem[below_mid, 0] = 1.0 - mem[below_mid, 1]
    mem[above_mid, 2] = 1.0 - mem[above_mid, 1]

    w_high = 0.5 + 0.5 * ((flat[high_mask] - high_t) / max(1.0 - high_t, 1e-6))
    mem[high_mask, 2] = w_high
    mem[high_mask, 1] = 1.0 - w_high

    return mem.reshape(orig_shape + (3,))


class FusionEngine:
    """
    Late-fusion decision engine (proposal Sec 3.4.2 / Sec 3.6.1 / FR-05).

    Accepts the flood hazard probability score from ``AttentionUNetModel``
    and the 3-class deprivation severity probability vector from
    ``GraphSAGEModel``, concatenates their representations, and applies a
    rule-based policy classification matrix to assign each parcel to one of:
      - ``0``: ``Retain and Formalise``
      - ``1``: ``Retain and Upgrade``
      - ``2``: ``Compensated Resettlement``

    Thresholds and the 3x3 matrix are isolated as instance attributes so
    planners can recalibrate governance cut-points without modifying trained
    model weights.
    """

    def __init__(
        self,
        flood_low_threshold: float = None,
        flood_high_threshold: float = None,
        policy_matrix: Sequence[Sequence[int]] = None,
        policy_class_names: Sequence[str] = None,
        policy_class_colors: Sequence[str] = None,
        model_version: str = "sprint5-v1",
    ):
        self.flood_low_threshold, self.flood_high_threshold = _resolve_flood_thresholds(
            flood_low_threshold, flood_high_threshold,
        )

        mat = np.asarray(
            DEFAULT_POLICY_MATRIX if policy_matrix is None else policy_matrix,
            dtype=np.int64,
        )
        if mat.shape != (3, 3):
            raise ValueError(f"policy_matrix must have shape (3, 3), got {mat.shape}.")
        if not set(np.unique(mat).tolist()).issubset({0, 1, 2}):
            raise ValueError("policy_matrix entries must be in {0, 1, 2}.")
        self.policy_matrix = mat

        self.policy_class_names = tuple(
            config.POLICY_CLASS_NAMES if policy_class_names is None else policy_class_names
        )
        self.policy_class_colors = tuple(
            config.POLICY_CLASS_COLORS if policy_class_colors is None else policy_class_colors
        )
        if len(self.policy_class_names) != config.POLICY_NUM_CLASSES:
            raise ValueError(
                f"policy_class_names must have length {config.POLICY_NUM_CLASSES}."
            )
        self.model_version = str(model_version)

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def concatenate_probabilities(
        self,
        flood_probs,
        deprivation_probs,
    ) -> np.ndarray:
        """
        Concatenate the scalar flood probability ``(N,)`` or ``(N, 1)`` with
        the 3-class deprivation probability vector ``(N, 3)`` into a joint
        ``(N, 4)`` late-fusion feature matrix ``[p_flood, p_low, p_med, p_high]``
        (proposal Sec 3.2.3 & Sec 3.6.1).
        """
        fp = self._to_numpy(flood_probs).astype(np.float64)
        dp = self._to_numpy(deprivation_probs).astype(np.float64)

        if fp.ndim == 2 and fp.shape[1] == 1:
            fp = fp[:, 0]
        if fp.ndim != 1:
            raise ValueError(f"flood_probs must be 1-D or (N, 1), got shape {fp.shape}.")
        if dp.ndim != 2 or dp.shape[1] != config.DEPRIVATION_NUM_CLASSES:
            raise ValueError(
                f"deprivation_probs must have shape (N, {config.DEPRIVATION_NUM_CLASSES}), "
                f"got {dp.shape}."
            )
        if fp.shape[0] != dp.shape[0]:
            raise ValueError(
                f"Parcel count mismatch: flood_probs has {fp.shape[0]} rows, "
                f"deprivation_probs has {dp.shape[0]} rows."
            )

        fp = np.clip(np.nan_to_num(fp, nan=0.0), 0.0, 1.0)
        dp = np.clip(np.nan_to_num(dp, nan=0.0), 0.0, 1.0)
        row_sums = dp.sum(axis=1, keepdims=True)
        # Re-normalise rows where valid probabilities do not sum to 1; fall
        # back to uniform when a row is all zeros.
        uniform = np.full_like(dp, 1.0 / dp.shape[1])
        dp = np.where(row_sums > 1e-9, dp / np.maximum(row_sums, 1e-9), uniform)

        return np.concatenate([fp[:, None], dp], axis=1)

    def classify_parcels(
        self,
        flood_probs,
        deprivation_probs,
    ) -> dict:
        """
        Apply the late-fusion decision matrix to ``N`` parcels.

        Returns a dict containing:
          - ``fused_probabilities``: ``(N, 4)`` array ``[p_flood, p_dep_0, p_dep_1, p_dep_2]``
          - ``flood_prob``: ``(N,)`` flood hazard probability in ``[0, 1]``
          - ``flood_level``: ``(N,)`` flood regime index ``0=low, 1=moderate, 2=high``
          - ``deprivation_probs``: ``(N, 3)`` deprivation class probabilities
          - ``deprivation_class``: ``(N,)`` argmax deprivation class ``0=low, 1=medium, 2=high``
          - ``deprivation_confidence``: ``(N,)`` max deprivation class probability
          - ``policy_class``: ``(N,)`` governance category index ``0, 1, 2``
          - ``policy_label``: list of human-readable governance category names
          - ``policy_color``: list of hex colors (green / amber / red)
          - ``confidence``: ``(N,)`` joint policy confidence score in ``[0, 1]``
          - ``policy_probs``: ``(N, 3)`` soft probability mass over the 3 policy categories
          - ``timestamp``: ISO-8601 UTC classification timestamp
          - ``model_version``: version tag for auditability (proposal Sec 4.2.3 / 4.3.3)
        """
        fused = self.concatenate_probabilities(flood_probs, deprivation_probs)
        fp = fused[:, 0]
        dp = fused[:, 1:]

        flood_level = discretize_flood_probability(
            fp,
            low_threshold=self.flood_low_threshold,
            high_threshold=self.flood_high_threshold,
        )
        dep_class = dp.argmax(axis=1).astype(np.int64)
        dep_conf = dp.max(axis=1)

        policy_class = self.policy_matrix[flood_level, dep_class].astype(np.int64)

        # Soft probability distribution over the 3 policy outcomes by
        # marginalising the outer product of soft flood-regime membership
        # and GraphSAGE softmax over the 3x3 policy matrix cells.
        flood_mem = flood_regime_membership(
            fp,
            low_threshold=self.flood_low_threshold,
            high_threshold=self.flood_high_threshold,
        )  # (N, 3)
        joint_grid = flood_mem[:, :, None] * dp[:, None, :]  # (N, 3, 3)
        policy_probs = np.zeros((fp.size, config.POLICY_NUM_CLASSES), dtype=np.float64)
        for r in range(3):
            for c in range(3):
                policy_probs[:, int(self.policy_matrix[r, c])] += joint_grid[:, r, c]

        confidence = np.clip(policy_probs[np.arange(fp.size), policy_class], 0.0, 1.0)

        labels = [self.policy_class_names[int(c)] for c in policy_class]
        colors = [self.policy_class_colors[int(c)] for c in policy_class]
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return {
            "fused_probabilities": fused,
            "flood_prob": fp,
            "flood_level": flood_level,
            "deprivation_probs": dp,
            "deprivation_class": dep_class,
            "deprivation_confidence": dep_conf,
            "policy_class": policy_class,
            "policy_label": labels,
            "policy_color": colors,
            "confidence": confidence,
            "policy_probs": policy_probs,
            "timestamp": timestamp,
            "model_version": self.model_version,
        }

    def summarize_classifications(self, result: Mapping) -> dict:
        """
        Aggregate parcel counts, shares, and mean flood/deprivation scores per
        governance category for notebook reporting and audit tables.
        """
        policy_class = np.asarray(result["policy_class"], dtype=np.int64)
        fp = np.asarray(result["flood_prob"], dtype=np.float64)
        dep_cls = np.asarray(result["deprivation_class"], dtype=np.int64)
        conf = np.asarray(result["confidence"], dtype=np.float64)
        n = int(policy_class.size)
        denom = max(n, 1)

        counts = {}
        fractions = {}
        mean_flood_prob = {}
        mean_confidence = {}
        for cls_idx, name in enumerate(self.policy_class_names):
            mask = policy_class == cls_idx
            cnt = int(mask.sum())
            counts[name] = cnt
            fractions[name] = float(cnt / denom)
            mean_flood_prob[name] = float(fp[mask].mean()) if cnt > 0 else 0.0
            mean_confidence[name] = float(conf[mask].mean()) if cnt > 0 else 0.0

        flood_level = np.asarray(result["flood_level"], dtype=np.int64)
        contingency = np.zeros((3, 3), dtype=np.int64)
        valid = (
            (flood_level >= 0) & (flood_level < 3)
            & (dep_cls >= 0) & (dep_cls < 3)
        )
        if valid.any():
            np.add.at(contingency, (flood_level[valid], dep_cls[valid]), 1)

        return {
            "total_parcels": n,
            "counts": counts,
            "fractions": fractions,
            "mean_flood_prob_by_policy": mean_flood_prob,
            "mean_confidence_by_policy": mean_confidence,
            "overall_mean_flood_prob": float(fp.mean()) if n > 0 else 0.0,
            "overall_mean_confidence": float(conf.mean()) if n > 0 else 0.0,
            "flood_vs_deprivation_counts": contingency.tolist(),
        }


# ---------------------------------------------------------------------------
# Spatial alignment: U-Net tile stitching + sampling at parcel locations
# ---------------------------------------------------------------------------


def predict_settlement_flood_map(
    unet_model,
    manifest_entries: Sequence[Mapping],
    raster_shape: tuple,
    layout: StackLayout = None,
    tile_size: int = None,
) -> dict:
    """
    Stitch per-tile Attention U-Net flood probability predictions back onto
    the full settlement raster grid ``(H, W)``.

    Overlapping 50px strips between adjacent 256x256 windows are averaged
    across tiles so tile boundaries remain smooth. Only un-augmented
    ``*_orig`` tiles are used when present (via ``_prefer_orig_entries``) so
    flipped/rotated training copies are not misaligned with the north-up
    raster grid.

    Returns:
      - ``prob_map``: ``(H, W)`` float32 flood probability surface in ``[0, 1]``
        (pixels not covered by any kept tile default to ``0.0``)
      - ``coverage_mask``: ``(H, W)`` bool mask of pixels covered by >= 1 tile
      - ``count_map``: ``(H, W)`` int32 count of contributing tiles per pixel
    """
    from .multi_settlement import _prefer_orig_entries

    layout = layout or getattr(unet_model, "layout", DEFAULT_LAYOUT)
    tile_size = int(tile_size or config.TILE_SIZE_PX)
    height, width = int(raster_shape[0]), int(raster_shape[1])

    prob_sum = np.zeros((height, width), dtype=np.float64)
    count_map = np.zeros((height, width), dtype=np.int32)

    orig_entries = _prefer_orig_entries(manifest_entries)

    for entry in orig_entries:
        tile = np.load(entry["path"])
        inputs = np.nan_to_num(tile[layout.input_indices], nan=0.0)
        tensor = torch.from_numpy(inputs).float().unsqueeze(0)
        proba = unet_model.predict_proba(tensor)[0, 0].numpy().astype(np.float64)

        r0 = int(entry["row_off"])
        c0 = int(entry["col_off"])
        r1 = min(r0 + min(proba.shape[0], tile_size), height)
        c1 = min(c0 + min(proba.shape[1], tile_size), width)
        if r1 <= r0 or c1 <= c0:
            continue

        prob_sum[r0:r1, c0:c1] += proba[: r1 - r0, : c1 - c0]
        count_map[r0:r1, c0:c1] += 1

    coverage = count_map > 0
    prob_map = np.zeros((height, width), dtype=np.float32)
    prob_map[coverage] = (prob_sum[coverage] / count_map[coverage]).astype(np.float32)

    return {
        "prob_map": prob_map,
        "coverage_mask": coverage,
        "count_map": count_map,
    }


def extract_parcel_flood_probabilities(
    prob_map: np.ndarray,
    transform,
    raster_crs,
    buildings_gdf=None,
    node_positions_metric: np.ndarray = None,
    metric_crs: str = None,
    default_prob: float = 0.0,
) -> np.ndarray:
    """
    Sample the settlement flood probability surface ``prob_map`` at each
    building parcel.

    Supports two input modes:
      1. ``buildings_gdf`` (a GeoDataFrame of building polygons or points in
         any CRS): reprojects to ``raster_crs`` and samples the pixel under
         the building centroid. Informal-settlement roofs are typically
         smaller than a 10 m Sentinel pixel, so centroid sampling is the
         honest 1-pixel-per-parcel readout.
      2. ``node_positions_metric`` (an ``(N, 2)`` array of ``(x, y)`` metric
         coordinates in ``CRS_METRIC``, as stored on ``data.pos``): transforms
         coordinates to ``raster_crs`` and samples ``prob_map`` at the
         corresponding ``(row, col)`` indices.

    Parcels falling outside the raster extent receive ``default_prob``.
    """
    from rasterio.transform import rowcol

    prob_arr = np.asarray(prob_map, dtype=np.float64)
    height, width = prob_arr.shape
    metric_crs = metric_crs or config.CRS_METRIC

    if buildings_gdf is not None:
        gdf = buildings_gdf.to_crs(raster_crs) if buildings_gdf.crs is not None else buildings_gdf
        centroids = gdf.geometry.centroid
        xs = centroids.x.to_numpy(dtype=np.float64)
        ys = centroids.y.to_numpy(dtype=np.float64)
    elif node_positions_metric is not None:
        coords = np.asarray(node_positions_metric, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"node_positions_metric must be (N, 2), got {coords.shape}.")
        if str(raster_crs) != str(metric_crs):
            import geopandas as gpd
            from shapely.geometry import Point

            pts = gpd.GeoSeries(
                [Point(float(x), float(y)) for x, y in coords],
                crs=metric_crs,
            ).to_crs(raster_crs)
            xs = pts.x.to_numpy(dtype=np.float64)
            ys = pts.y.to_numpy(dtype=np.float64)
        else:
            xs = coords[:, 0]
            ys = coords[:, 1]
    else:
        raise ValueError(
            "Provide either buildings_gdf or node_positions_metric to sample "
            "parcel flood probabilities."
        )

    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    out = np.full(xs.shape[0], float(default_prob), dtype=np.float64)
    out[in_bounds] = np.nan_to_num(
        prob_arr[rows[in_bounds], cols[in_bounds]],
        nan=float(default_prob),
    )
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# NRRP high-risk zone validation (proposal Sec 3.4.3 weeks 9-10 / Sec 3.7.1)
# ---------------------------------------------------------------------------


def validate_policy_against_nrrp(
    policy_class: np.ndarray,
    in_nrrp_zone: np.ndarray,
    flood_prob: np.ndarray = None,
    deprivation_class: np.ndarray = None,
    policy_class_names: Sequence[str] = None,
    min_lift: float = None,
) -> dict:
    """
    Validate the 3-class parcel policy recommendations against the NRRP
    high-risk riparian reference zones.

    Specifically checks whether parcels assigned to ``Compensated Resettlement``
    (class 2) concentrate inside the NRRP high-risk zone (lift > 1), and
    whether parcels assigned to ``Retain and Formalise`` (class 0) concentrate
    outside the high-risk zone (safe ground). Also quantifies the "blanket
    eviction avoided" governance metric (proposal Sec 1.2 Problem Statement &
    Sec 2.4): the fraction of parcels inside the riparian reference buffer
    that the late-fusion engine recommends retaining (Formalise or Upgrade)
    because local terrain/SAR flood hazard is low or manageable.
    """
    p_cls = np.asarray(policy_class, dtype=np.int64).ravel()
    ref = (np.asarray(in_nrrp_zone) > 0).ravel()
    if p_cls.shape != ref.shape:
        raise ValueError(
            f"policy_class shape {p_cls.shape} and in_nrrp_zone shape {ref.shape} must match."
        )

    names = tuple(policy_class_names or config.POLICY_CLASS_NAMES)
    min_lift = config.FUSION_NRRP_MIN_LIFT if min_lift is None else float(min_lift)
    n = int(p_cls.size)
    if n == 0:
        raise ValueError("Cannot validate an empty parcel set (0 parcels).")

    resettlement_mask = p_cls == 2
    resettlement_agreement = agreement_metrics(resettlement_mask, ref)

    intervention_mask = p_cls >= 1
    intervention_agreement = agreement_metrics(intervention_mask, ref)

    in_zone_count = int(ref.sum())
    out_zone_count = int((~ref).sum())

    in_zone_dist = {}
    out_zone_dist = {}
    in_zone_frac = {}
    out_zone_frac = {}
    for idx, name in enumerate(names):
        c_in = int(np.sum((p_cls == idx) & ref))
        c_out = int(np.sum((p_cls == idx) & (~ref)))
        in_zone_dist[name] = c_in
        out_zone_dist[name] = c_out
        in_zone_frac[name] = float(c_in / max(in_zone_count, 1))
        out_zone_frac[name] = float(c_out / max(out_zone_count, 1))

    retained_in_zone = int(np.sum((p_cls < 2) & ref))
    blanket_eviction_avoided_frac = float(retained_in_zone / max(in_zone_count, 1))

    report = {
        "total_parcels": n,
        "nrrp_zone_parcels": in_zone_count,
        "outside_nrrp_parcels": out_zone_count,
        "nrrp_zone_fraction": float(in_zone_count / n),
        "resettlement_agreement": resettlement_agreement,
        "resettlement_lift": float(resettlement_agreement["lift"]),
        "intervention_agreement": intervention_agreement,
        "intervention_lift": float(intervention_agreement["lift"]),
        "meets_lift_target": bool(resettlement_agreement["lift"] >= min_lift),
        "lift_target": min_lift,
        "in_nrrp_policy_counts": in_zone_dist,
        "in_nrrp_policy_fractions": in_zone_frac,
        "outside_nrrp_policy_counts": out_zone_dist,
        "outside_nrrp_policy_fractions": out_zone_frac,
        "retained_in_nrrp_count": retained_in_zone,
        "blanket_eviction_avoided_fraction": blanket_eviction_avoided_frac,
    }

    if flood_prob is not None:
        fp = np.asarray(flood_prob, dtype=np.float64).ravel()
        report["mean_flood_prob_in_nrrp"] = float(fp[ref].mean()) if in_zone_count > 0 else 0.0
        report["mean_flood_prob_outside_nrrp"] = (
            float(fp[~ref].mean()) if out_zone_count > 0 else 0.0
        )

    if deprivation_class is not None:
        dc = np.asarray(deprivation_class, dtype=np.int64).ravel()
        report["high_deprivation_fraction_in_nrrp"] = (
            float(np.mean(dc[ref] == 2)) if in_zone_count > 0 else 0.0
        )
        report["high_deprivation_fraction_outside_nrrp"] = (
            float(np.mean(dc[~ref] == 2)) if out_zone_count > 0 else 0.0
        )

    return report


def flag_parcels_in_reference_zone(
    reference_mask: np.ndarray,
    transform,
    raster_crs,
    buildings_gdf=None,
    node_positions_metric: np.ndarray = None,
    metric_crs: str = None,
) -> np.ndarray:
    """
    Determine which parcels fall inside a rasterised reference zone mask
    (e.g. from ``label_validation.rasterize_reference`` or
    ``label_validation.river_corridor_reference``). Returns a boolean array
    of length ``N``.
    """
    sampled = extract_parcel_flood_probabilities(
        prob_map=(np.asarray(reference_mask) > 0).astype(np.float32),
        transform=transform,
        raster_crs=raster_crs,
        buildings_gdf=buildings_gdf,
        node_positions_metric=node_positions_metric,
        metric_crs=metric_crs,
        default_prob=0.0,
    )
    return sampled >= 0.5


# ---------------------------------------------------------------------------
# GeoDataFrame / GeoJSON export (proposal Sec 4.3.3 & Sec 4.4.1)
# ---------------------------------------------------------------------------


def build_parcel_policy_geodataframe(
    settlement_key: str,
    classification_result: Mapping,
    buildings_gdf=None,
    node_positions_metric: np.ndarray = None,
    node_ids: np.ndarray = None,
    in_nrrp_zone: np.ndarray = None,
    default_area_m2: float = 25.0,
):
    """
    Assemble the policy-classified parcel ``GeoDataFrame`` in WGS84
    (``EPSG:4326``) matching the Entity Relationship Diagram (Sec 4.3.3) and
    Logical Database Schema (Sec 4.4.1).

    When ``buildings_gdf`` is supplied, preserves the exact OSM building
    footprint polygons; when only ``node_positions_metric`` (``data.pos`` in
    ``EPSG:32737``) is available, constructs square footprint buffers of area
    ``default_area_m2`` around each node centroid so every parcel has a valid
    Polygon geometry for the Sprint 6 Folium choropleth map.
    """
    import geopandas as gpd
    from shapely.geometry import box

    fp = np.asarray(classification_result["flood_prob"], dtype=np.float64)
    fl = np.asarray(classification_result["flood_level"], dtype=np.int64)
    dp = np.asarray(classification_result["deprivation_probs"], dtype=np.float64)
    dc = np.asarray(classification_result["deprivation_class"], dtype=np.int64)
    pc = np.asarray(classification_result["policy_class"], dtype=np.int64)
    conf = np.asarray(classification_result["confidence"], dtype=np.float64)
    labels = list(classification_result["policy_label"])
    colors = list(classification_result["policy_color"])
    n = int(fp.size)

    if buildings_gdf is not None and len(buildings_gdf) == n:
        gdf_metric = buildings_gdf.to_crs(config.CRS_METRIC)
        areas_m2 = gdf_metric.geometry.area.to_numpy(dtype=np.float64)
        geoms_wgs84 = buildings_gdf.to_crs(config.CRS_WGS84).geometry.tolist()
        if "node_id" in buildings_gdf.columns:
            ids = buildings_gdf["node_id"].to_numpy(dtype=np.int64)
        else:
            ids = np.arange(n, dtype=np.int64)
    elif node_positions_metric is not None:
        coords = np.asarray(node_positions_metric, dtype=np.float64)
        if coords.shape != (n, 2):
            raise ValueError(
                f"node_positions_metric shape {coords.shape} does not match parcel count {n}."
            )
        half_side = 0.5 * np.sqrt(max(float(default_area_m2), 1.0))
        polys_metric = [
            box(x - half_side, y - half_side, x + half_side, y + half_side)
            for x, y in coords
        ]
        geoms_wgs84 = (
            gpd.GeoSeries(polys_metric, crs=config.CRS_METRIC)
            .to_crs(config.CRS_WGS84)
            .tolist()
        )
        areas_m2 = np.full(n, float(default_area_m2), dtype=np.float64)
        ids = (
            np.asarray(node_ids, dtype=np.int64)
            if node_ids is not None
            else np.arange(n, dtype=np.int64)
        )
    else:
        # Fallback when neither building polygons nor metric positions exist:
        # place synthetic micro-parcels on a deterministic grid near the
        # settlement's configured centroid so offline tests and schema exports
        # remain self-contained.
        lat0, lon0 = config.SETTLEMENTS.get(
            settlement_key, next(iter(config.SETTLEMENTS.values()))
        ).centroid
        cols = max(int(np.ceil(np.sqrt(n))), 1)
        step = 0.0001  # ~11m in WGS84 near the equator
        geoms_wgs84 = []
        for i in range(n):
            r, c = divmod(i, cols)
            lon = lon0 + (c - cols / 2.0) * step
            lat = lat0 + (r - cols / 2.0) * step
            geoms_wgs84.append(box(lon, lat, lon + step * 0.8, lat + step * 0.8))
        areas_m2 = np.full(n, float(default_area_m2), dtype=np.float64)
        ids = (
            np.asarray(node_ids, dtype=np.int64)
            if node_ids is not None
            else np.arange(n, dtype=np.int64)
        )

    dep_names = [config.DEPRIVATION_CLASS_NAMES[int(c)] for c in dc]

    records = {
        "parcel_id": [f"{settlement_key}_{int(i)}" for i in ids],
        "settlement": [settlement_key] * n,
        "node_id": ids.tolist(),
        "area_m2": np.round(areas_m2, 2).tolist(),
        "flood_prob": np.round(fp, 4).tolist(),
        "flood_level": fl.tolist(),
        "deprivation_class": dc.tolist(),
        "deprivation_label": dep_names,
        "prob_dep_low": np.round(dp[:, 0], 4).tolist(),
        "prob_dep_medium": np.round(dp[:, 1], 4).tolist(),
        "prob_dep_high": np.round(dp[:, 2], 4).tolist(),
        "policy_class": pc.tolist(),
        "policy_category": labels,
        "policy_color": colors,
        "confidence": np.round(conf, 4).tolist(),
        "model_version": [classification_result.get("model_version", "sprint5-v1")] * n,
        "timestamp": [classification_result.get("timestamp", "")] * n,
    }
    if in_nrrp_zone is not None:
        records["in_nrrp_zone"] = [bool(x) for x in np.asarray(in_nrrp_zone).ravel()]

    return gpd.GeoDataFrame(records, geometry=geoms_wgs84, crs=config.CRS_WGS84)


def export_settlement_policy_geojson(
    gdf,
    settlement_key: str,
    output_dir: Union[str, Path] = None,
) -> Path:
    """
    Save a settlement's policy-classified parcel GeoDataFrame to
    ``outputs/classifications/<settlement>_policy_parcels.geojson``
    (proposal Sec 4.4.1 Logical Database Schema).
    """
    out_dir = Path(output_dir or config.CLASSIFICATIONS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{settlement_key}_policy_parcels.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    logger.info(
        "Saved %d policy-classified parcels for %s to %s",
        len(gdf), settlement_key, out_path,
    )
    return out_path


def aggregate_multi_settlement_fusion_reports(
    per_settlement_results: Mapping[str, Mapping],
    engine: FusionEngine = None,
    min_lift: float = None,
) -> dict:
    """
    Combine per-settlement fusion classification outputs and NRRP validation
    masks into a pooled three-settlement summary + NRRP validation report.

    ``per_settlement_results`` maps ``settlement_key`` -> dict containing at
    least ``classification`` (output of ``engine.classify_parcels``) and
    optionally ``in_nrrp_zone`` (boolean array of length ``N_s``).
    """
    if not per_settlement_results:
        raise ValueError("per_settlement_results must contain at least one settlement.")

    engine = engine or FusionEngine()
    per_summary = {}
    per_nrrp = {}

    pooled_policy = []
    pooled_flood_prob = []
    pooled_flood_level = []
    pooled_dep_probs = []
    pooled_dep_class = []
    pooled_conf = []
    pooled_nrrp = []
    has_all_nrrp = True

    for key, bundle in per_settlement_results.items():
        cls = bundle["classification"]
        per_summary[key] = engine.summarize_classifications(cls)

        pooled_policy.append(np.asarray(cls["policy_class"], dtype=np.int64))
        pooled_flood_prob.append(np.asarray(cls["flood_prob"], dtype=np.float64))
        pooled_flood_level.append(np.asarray(cls["flood_level"], dtype=np.int64))
        pooled_dep_probs.append(np.asarray(cls["deprivation_probs"], dtype=np.float64))
        pooled_dep_class.append(np.asarray(cls["deprivation_class"], dtype=np.int64))
        pooled_conf.append(np.asarray(cls["confidence"], dtype=np.float64))

        nrrp_mask = bundle.get("in_nrrp_zone")
        if nrrp_mask is not None:
            nrrp_arr = np.asarray(nrrp_mask, dtype=bool).ravel()
            per_nrrp[key] = validate_policy_against_nrrp(
                policy_class=cls["policy_class"],
                in_nrrp_zone=nrrp_arr,
                flood_prob=cls["flood_prob"],
                deprivation_class=cls["deprivation_class"],
                policy_class_names=engine.policy_class_names,
                min_lift=min_lift,
            )
            pooled_nrrp.append(nrrp_arr)
        else:
            has_all_nrrp = False

    all_policy = np.concatenate(pooled_policy)
    all_fp = np.concatenate(pooled_flood_prob)
    all_fl = np.concatenate(pooled_flood_level)
    all_dp = np.concatenate(pooled_dep_probs, axis=0)
    all_dc = np.concatenate(pooled_dep_class)
    all_conf = np.concatenate(pooled_conf)

    pooled_cls_bundle = {
        "policy_class": all_policy,
        "flood_prob": all_fp,
        "flood_level": all_fl,
        "deprivation_probs": all_dp,
        "deprivation_class": all_dc,
        "confidence": all_conf,
    }
    pooled_summary = engine.summarize_classifications(pooled_cls_bundle)

    pooled_nrrp_report = None
    if has_all_nrrp and pooled_nrrp:
        all_nrrp = np.concatenate(pooled_nrrp)
        pooled_nrrp_report = validate_policy_against_nrrp(
            policy_class=all_policy,
            in_nrrp_zone=all_nrrp,
            flood_prob=all_fp,
            deprivation_class=all_dc,
            policy_class_names=engine.policy_class_names,
            min_lift=min_lift,
        )

    return {
        "per_settlement_summary": per_summary,
        "pooled_summary": pooled_summary,
        "per_settlement_nrrp": per_nrrp,
        "pooled_nrrp": pooled_nrrp_report,
        "all_settlements_meet_lift_target": (
            all(r["meets_lift_target"] for r in per_nrrp.values())
            if per_nrrp
            else False
        ),
        "meets_pooled_lift_target": (
            bool(pooled_nrrp_report["meets_lift_target"])
            if pooled_nrrp_report is not None
            else False
        ),
    }
