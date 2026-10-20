"""
Late-fusion policy decision engine (Sprint 5 / FR-05).

The Attention U-Net and GraphSAGE heads stay independent. This module is the
only place their outputs meet: for each parcel it concatenates the flood
hazard probability with the three-class deprivation probability vector and
looks up a rule-based governance category:

  0  Retain and Formalise      -- tenure security on safe, low-deprivation plots
  1  Retain and Upgrade        -- in-situ drainage / WASH / housing investment
  2  Compensated Resettlement  -- acute flood (NRRP-aligned relocate class)

The matrix and flood-bin cuts live in ``config`` so a different city can retune
policy without retraining either head (proposal Sec 3.6.1: "this logic
deliberately isolated into a standalone class").

NRRP validation is parcel-level, not pixel-level: Compensated Resettlement
parcels should concentrate inside the 37 high-risk zone polygons (lift > 1).
The numpy core here is reference-agnostic -- it takes a boolean
``in_reference`` mask per parcel -- so the same statistic works with NRRP
polygons or the OSM river-corridor fallback. Geo helpers (sampling a flood
raster at centroids, point-in-polygon, GeoJSON I/O) live in
``src.utils.parcel_io`` and are imported lazily.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

import numpy as np

from .. import config

logger = logging.getLogger(__name__)


def align_node_outputs_to_parcels(values, node_ids, parcel_node_ids) -> np.ndarray:
    """
    Reorder a per-node array (GraphSAGE ``predict_proba`` rows, flood samples
    taken in graph order, ...) onto a parcel GeoDataFrame's ``node_id`` column.
    Needed because prune_small_components can leave gaps in node_id, and a
    combined three-settlement graph repeats 0..n-1 inside each block.
    """
    values = np.asarray(values)
    node_ids = np.asarray(node_ids).reshape(-1)
    parcel_node_ids = np.asarray(parcel_node_ids).reshape(-1)
    if values.shape[0] != node_ids.shape[0]:
        raise ValueError(
            f"values length {values.shape[0]} must match node_ids {node_ids.shape[0]}"
        )
    lookup = {int(n): values[i] for i, n in enumerate(node_ids)}
    missing = [int(n) for n in parcel_node_ids if int(n) not in lookup]
    if missing:
        raise KeyError(
            f"{len(missing)} parcel node_id(s) missing from the graph (e.g. {missing[:5]}). "
            f"Rebuild {{settlement}}_buildings.geojson from the same acquire_graph_data call "
            f"that wrote the .pt graph."
        )
    stacked = [lookup[int(n)] for n in parcel_node_ids]
    return np.stack(stacked, axis=0)


def concatenate_head_outputs(
    flood_probability,
    deprivation_proba,
) -> np.ndarray:
    """
    Concatenate the two head outputs into an (N, 4) matrix:

      [p_flood, p_low, p_medium, p_high]

    This is the "two probability distributions were concatenated" step in
    proposal Sec 3.2.3, kept as its own function so tests (and a future
    learned fusion head) can consume the same packed vector the rule matrix
    reads.
    """
    flood = np.asarray(flood_probability, dtype=np.float64).reshape(-1)
    dep = coerce_deprivation_proba(deprivation_proba)
    if flood.shape[0] != dep.shape[0]:
        raise ValueError(
            f"flood_probability length {flood.shape[0]} must match "
            f"deprivation_proba rows {dep.shape[0]}"
        )
    return np.concatenate([flood[:, None], dep], axis=1)


def coerce_deprivation_proba(deprivation, num_classes: int = None) -> np.ndarray:
    """
    Accept either an (N, C) probability matrix or an (N,) class-index vector
    (one-hot encoded). Class indices are a convenience for tests and for
    notebooks that want to fuse GraphSAGE *labels* when a checkpoint is
    missing -- they are not a substitute for ``predict_proba`` at report time.
    """
    num_classes = num_classes or config.DEPRIVATION_NUM_CLASSES
    arr = np.asarray(deprivation)
    if arr.ndim == 1:
        if arr.size == 0:
            return np.zeros((0, num_classes), dtype=np.float64)
        if np.any(arr < 0) or np.any(arr >= num_classes):
            raise ValueError(
                f"deprivation class indices must be in [0, {num_classes})"
            )
        out = np.zeros((arr.size, num_classes), dtype=np.float64)
        out[np.arange(arr.size), arr.astype(np.int64)] = 1.0
        return out
    if arr.ndim != 2:
        raise ValueError(
            f"deprivation_proba must be 1-d class indices or 2-d probabilities, "
            f"got shape {arr.shape}"
        )
    if arr.shape[1] != num_classes:
        raise ValueError(
            f"deprivation_proba has {arr.shape[1]} columns; expected {num_classes}"
        )
    return np.asarray(arr, dtype=np.float64)


def flood_bin(
    flood_probability,
    low_max: float = None,
    high_min: float = None,
) -> np.ndarray:
    """
    Bin flood probability into {0: low, 1: medium, 2: high}. Boundaries are
    ``p < low_max`` and ``p >= high_min``; the closed high edge matches the
    usual "at or above the relocate threshold" reading of a policy cut.
    """
    low_max = config.POLICY_FLOOD_LOW_MAX if low_max is None else float(low_max)
    high_min = config.POLICY_FLOOD_HIGH_MIN if high_min is None else float(high_min)
    if not 0.0 < low_max < high_min < 1.0:
        raise ValueError(
            f"Need 0 < low_max < high_min < 1; got low_max={low_max}, high_min={high_min}"
        )
    p = np.asarray(flood_probability, dtype=np.float64).reshape(-1)
    bins = np.ones(p.shape, dtype=np.int64)
    bins[p < low_max] = 0
    bins[p >= high_min] = 2
    return bins


def _flood_bin_confidence(
    flood_probability,
    bins: np.ndarray,
    low_max: float,
    high_min: float,
) -> np.ndarray:
    """
    How decisively a probability sits in its assigned bin. Low-bin confidence
    grows as p -> 0; high-bin as p -> 1; medium-bin as p sits mid-interval.
    Used only as half of the fused confidence score -- not a calibrated
    probability of the policy being 'correct'.
    """
    p = np.asarray(flood_probability, dtype=np.float64).reshape(-1)
    conf = np.empty(p.shape, dtype=np.float64)
    # Low: (low_max - p) / low_max  in [0, 1]
    low = bins == 0
    conf[low] = np.clip((low_max - p[low]) / max(low_max, 1e-9), 0.0, 1.0)
    # High: (p - high_min) / (1 - high_min)
    high = bins == 2
    conf[high] = np.clip(
        (p[high] - high_min) / max(1.0 - high_min, 1e-9), 0.0, 1.0,
    )
    # Medium: 1 at the midpoint, 0 at either edge.
    mid = bins == 1
    midpoint = 0.5 * (low_max + high_min)
    half = 0.5 * (high_min - low_max)
    conf[mid] = np.clip(1.0 - np.abs(p[mid] - midpoint) / max(half, 1e-9), 0.0, 1.0)
    return conf


def lookup_policy(
    flood_bins,
    deprivation_class,
    matrix=None,
) -> np.ndarray:
    """Index ``matrix[flood_bin, deprivation_class]`` with bounds checks."""
    matrix = np.asarray(
        config.DEFAULT_POLICY_MATRIX if matrix is None else matrix,
        dtype=np.int64,
    )
    if matrix.shape != (3, 3):
        raise ValueError(f"policy matrix must be 3x3, got {matrix.shape}")
    fb = np.asarray(flood_bins, dtype=np.int64).reshape(-1)
    dc = np.asarray(deprivation_class, dtype=np.int64).reshape(-1)
    if fb.shape != dc.shape:
        raise ValueError(
            f"flood_bins length {fb.shape[0]} must match deprivation_class {dc.shape[0]}"
        )
    if np.any(fb < 0) or np.any(fb > 2) or np.any(dc < 0) or np.any(dc > 2):
        raise ValueError("flood bins and deprivation classes must be in {0, 1, 2}")
    return matrix[fb, dc]


def policy_nrrp_report(
    policy_class,
    in_reference,
    lift_target: float = None,
    eps: float = 1e-9,
) -> dict:
    """
    Parcel-level concentration of each policy class inside a boolean
    ``in_reference`` mask (True = parcel centroid falls in an NRRP high-risk
    polygon, or the river-corridor fallback).

    Headline number is **resettlement lift** = P(in reference | resettle) /
    P(in reference). Lift >> 1 means the relocate class is enriched in the
    37-zone overlay -- the Sprint 5 validation the proposal asks for. Lift ~1
    means the fusion engine is ignoring the riparian prior. Retain-and-
    Formalise lift should sit *below* 1 (we should not be handing tenure
    security to plots inside the high-risk overlay).
    """
    lift_target = (
        config.POLICY_NRRP_LIFT_TARGET if lift_target is None else float(lift_target)
    )
    policy = np.asarray(policy_class, dtype=np.int64).reshape(-1)
    inside = np.asarray(in_reference).reshape(-1).astype(bool)
    if policy.shape != inside.shape:
        raise ValueError(
            f"policy_class length {policy.shape[0]} must match in_reference {inside.shape[0]}"
        )
    n = int(policy.size)
    if n == 0:
        raise ValueError("policy_nrrp_report requires at least one parcel.")
    n_inside = int(inside.sum())
    reference_fraction = n_inside / n

    per_class = {}
    for code, name in enumerate(config.POLICY_CLASS_NAMES):
        flag = policy == code
        support = int(flag.sum())
        tp = int((flag & inside).sum())
        precision = tp / (support + eps)          # P(in ref | class)
        recall = tp / (n_inside + eps)            # P(class | in ref)
        lift = precision / (reference_fraction + eps)
        per_class[name] = {
            "code": code,
            "label": config.POLICY_CLASS_LABELS[code],
            "support": support,
            "n_inside_reference": tp,
            "precision": float(precision),
            "recall": float(recall),
            "lift": float(lift),
            "fraction": support / n,
        }

    resettle = per_class["compensated_resettlement"]
    formalise = per_class["retain_and_formalise"]
    return {
        "n_parcels": n,
        "n_inside_reference": n_inside,
        "reference_fraction": float(reference_fraction),
        "lift_target": lift_target,
        "resettlement_lift": resettle["lift"],
        "meets_lift_target": bool(resettle["lift"] >= lift_target),
        "formalise_lift": formalise["lift"],
        "formalise_depleted_in_reference": bool(formalise["lift"] < 1.0),
        "per_class": per_class,
    }


def summarise_policy(policy_class, settlement_keys: Iterable[str] = None) -> dict:
    """Per-class counts / fractions, optionally broken down by settlement."""
    policy = np.asarray(policy_class, dtype=np.int64).reshape(-1)
    n = max(int(policy.size), 1)
    counts = {name: int(np.sum(policy == i)) for i, name in enumerate(config.POLICY_CLASS_NAMES)}
    summary = {
        "n_parcels": int(policy.size),
        "counts": counts,
        "fractions": {k: v / n for k, v in counts.items()},
        "class_labels": list(config.POLICY_CLASS_LABELS),
    }
    if settlement_keys is not None:
        keys = np.asarray(list(settlement_keys), dtype=object)
        if keys.shape[0] != policy.shape[0]:
            raise ValueError("settlement_keys length must match policy_class")
        by = {}
        for key in dict.fromkeys(keys.tolist()):
            mask = keys == key
            local = policy[mask]
            n_local = max(int(local.size), 1)
            by[str(key)] = {
                "n_parcels": int(local.size),
                "counts": {
                    name: int(np.sum(local == i))
                    for i, name in enumerate(config.POLICY_CLASS_NAMES)
                },
                "fractions": {
                    name: int(np.sum(local == i)) / n_local
                    for i, name in enumerate(config.POLICY_CLASS_NAMES)
                },
            }
        summary["per_settlement"] = by
    return summary


class FusionEngine:
    """
    Rule-based late-fusion classifier. Construct once with the flood-bin cuts
    and policy matrix for a jurisdiction; ``classify`` is a pure numpy call
    so it can run on CPU next to either head's ``predict_proba``.
    """

    def __init__(
        self,
        flood_low_max: float = None,
        flood_high_min: float = None,
        policy_matrix=None,
    ):
        self.flood_low_max = (
            config.POLICY_FLOOD_LOW_MAX if flood_low_max is None else float(flood_low_max)
        )
        self.flood_high_min = (
            config.POLICY_FLOOD_HIGH_MIN if flood_high_min is None else float(flood_high_min)
        )
        matrix = np.asarray(
            config.DEFAULT_POLICY_MATRIX if policy_matrix is None else policy_matrix,
            dtype=np.int64,
        )
        if matrix.shape != (3, 3):
            raise ValueError(f"policy_matrix must be 3x3, got {matrix.shape}")
        if np.any(matrix < 0) or np.any(matrix >= config.POLICY_NUM_CLASSES):
            raise ValueError(
                f"policy_matrix entries must be in [0, {config.POLICY_NUM_CLASSES})"
            )
        self.policy_matrix = matrix
        # Validate bin cuts now so classify() cannot fail mid-batch.
        flood_bin([0.0], low_max=self.flood_low_max, high_min=self.flood_high_min)

    def classify(self, flood_probability, deprivation_proba) -> dict:
        """
        Fuse the two heads for N parcels.

        Returns a dict of numpy arrays (not a GeoDataFrame) so the numpy core
        stays offline-testable. Attach geometries afterwards with
        ``attach_to_parcels`` / ``src.utils.parcel_io``.
        """
        packed = concatenate_head_outputs(flood_probability, deprivation_proba)
        flood = packed[:, 0]
        dep = packed[:, 1:]
        deprivation_class = dep.argmax(axis=1).astype(np.int64)
        bins = flood_bin(
            flood, low_max=self.flood_low_max, high_min=self.flood_high_min,
        )
        policy = lookup_policy(bins, deprivation_class, matrix=self.policy_matrix)
        dep_conf = dep.max(axis=1)
        flood_conf = _flood_bin_confidence(
            flood, bins, self.flood_low_max, self.flood_high_min,
        )
        confidence = 0.5 * flood_conf + 0.5 * dep_conf
        names = np.array(config.POLICY_CLASS_LABELS, dtype=object)[policy]
        return {
            "flood_probability": flood,
            "flood_bin": bins,
            "deprivation_proba": dep,
            "deprivation_class": deprivation_class,
            "concatenated": packed,
            "policy_class": policy,
            "policy_name": names,
            "confidence": confidence,
        }

    def attach_to_parcels(self, parcels, result: Mapping) -> object:
        """
        Copy classification columns onto a GeoDataFrame (or any object that
        supports ``.copy()`` + item assignment). Imported lazily so the numpy
        core does not require geopandas.
        """
        out = parcels.copy()
        out["flood_probability"] = np.asarray(result["flood_probability"])
        out["flood_bin"] = np.asarray(result["flood_bin"])
        out["deprivation_class"] = np.asarray(result["deprivation_class"])
        out["policy_class"] = np.asarray(result["policy_class"])
        out["policy_name"] = np.asarray(result["policy_name"])
        out["confidence"] = np.asarray(result["confidence"])
        dep = np.asarray(result["deprivation_proba"])
        for i, name in enumerate(config.DEPRIVATION_CLASS_NAMES):
            out[f"deprivation_p_{name}"] = dep[:, i]
        return out

    def export_geojson(self, parcels, path: Union[str, Path], crs: str = None) -> Path:
        """
        Write a parcel classification layer. ``parcels`` must be a GeoDataFrame
        that already carries the columns from ``attach_to_parcels``.
        """
        from ..utils.parcel_io import write_classified_geojson

        path = Path(path)
        write_classified_geojson(parcels, path, crs=crs)
        logger.info("Wrote policy GeoJSON (%d parcels) to %s", len(parcels), path)
        return path

    def classify_and_export(
        self,
        flood_probability,
        deprivation_proba,
        parcels,
        path: Union[str, Path],
        settlement_key: str = None,
        in_reference=None,
        crs: str = None,
    ) -> dict:
        """
        End-to-end Sprint 5 path for one settlement: classify, attach, write
        GeoJSON, optionally score NRRP lift. Returns the classification dict
        plus ``summary`` / ``nrrp_report`` / ``path``.
        """
        result = self.classify(flood_probability, deprivation_proba)
        classified = self.attach_to_parcels(parcels, result)
        if settlement_key is not None and "settlement_key" not in classified.columns:
            classified["settlement_key"] = settlement_key
        out_path = self.export_geojson(classified, path, crs=crs)
        keys = None
        if "settlement_key" in getattr(classified, "columns", []):
            keys = classified["settlement_key"].to_numpy()
        summary = summarise_policy(result["policy_class"], settlement_keys=keys)
        payload = {
            "result": result,
            "parcels": classified,
            "path": str(out_path),
            "summary": summary,
            "nrrp_report": None,
        }
        if in_reference is not None:
            payload["nrrp_report"] = policy_nrrp_report(
                result["policy_class"], in_reference,
            )
        return payload


def fuse_settlements(
    parcels_by_settlement: Mapping[str, object],
    flood_by_settlement: Mapping[str, Sequence[float]],
    deprivation_by_settlement: Mapping[str, object],
    engine: FusionEngine = None,
    output_dir: Union[str, Path] = None,
    reference_gdf=None,
) -> dict:
    """
    Run the fusion engine over every settlement that has both head outputs
    and a parcel GeoDataFrame, write one GeoJSON each plus a combined layer,
    and (when ``reference_gdf`` is given) score pooled + per-settlement NRRP
    lift.

    Missing settlements are skipped with a warning rather than aborting --
    Colab runs often have Kibera ready before Mathare/Mukuru Drive files.
    """
    from ..utils.parcel_io import points_in_reference

    engine = engine or FusionEngine()
    out_dir = Path(output_dir or config.CLASSIFICATIONS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    per = {}
    frames = []
    pooled_policy = []
    pooled_inside = []
    pooled_keys = []

    for key in config.SETTLEMENT_KEYS:
        if key not in parcels_by_settlement:
            continue
        if key not in flood_by_settlement or key not in deprivation_by_settlement:
            logger.warning(
                "Skipping %s -- need both flood probabilities and deprivation "
                "probabilities to fuse.",
                key,
            )
            continue
        parcels = parcels_by_settlement[key]
        flood = flood_by_settlement[key]
        dep = deprivation_by_settlement[key]
        in_ref = None
        if reference_gdf is not None:
            in_ref = points_in_reference(parcels, reference_gdf)
        payload = engine.classify_and_export(
            flood, dep, parcels,
            path=out_dir / f"{key}_policy.geojson",
            settlement_key=key,
            in_reference=in_ref,
        )
        per[key] = payload
        frames.append(payload["parcels"])
        pooled_policy.append(payload["result"]["policy_class"])
        pooled_keys.append(np.full(payload["result"]["policy_class"].shape[0], key, dtype=object))
        if in_ref is not None:
            pooled_inside.append(np.asarray(in_ref, dtype=bool))

    if not per:
        raise ValueError(
            "fuse_settlements received no complete settlement (parcels + both "
            "head outputs). Load graphs/buildings and run both predict_proba "
            "paths first."
        )

    combined_path = None
    if frames:
        import geopandas as gpd
        import pandas as pd

        combined = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            geometry="geometry",
            crs=frames[0].crs,
        )
        combined_path = out_dir / "all_settlements_policy.geojson"
        engine.export_geojson(combined, combined_path)

    policy_all = np.concatenate(pooled_policy)
    keys_all = np.concatenate(pooled_keys)
    nrrp = None
    if pooled_inside:
        inside_all = np.concatenate(pooled_inside)
        nrrp = policy_nrrp_report(policy_all, inside_all)

    return {
        "per_settlement": per,
        "combined_path": None if combined_path is None else str(combined_path),
        "summary": summarise_policy(policy_all, settlement_keys=keys_all),
        "nrrp_report": nrrp,
        "output_dir": str(out_dir),
    }


def dump_policy_report(report: Mapping, path: Union[str, Path]) -> Path:
    """JSON-serialise a ``policy_nrrp_report`` / ``fuse_settlements`` summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    # Drop live GeoDataFrames / numpy result blobs before serialising.
    clean = {}
    for k, v in report.items():
        if k in ("result", "parcels"):
            continue
        if k == "per_settlement" and isinstance(v, dict):
            clean[k] = {
                sk: {
                    ik: iv for ik, iv in sv.items()
                    if ik not in ("result", "parcels")
                }
                for sk, sv in v.items()
            }
        else:
            clean[k] = v
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=_default)
    return path
