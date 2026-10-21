"""
Late-fusion policy engine (Sprint 5).

The two heads stay independent. This module is the decision layer that
reads them: a per-parcel flood probability from the Attention U-Net and a
3-class deprivation softmax from GraphSAGE. It concatenates those outputs
and looks the pair up in a fixed policy matrix. There is no third trained
network -- the matrix is the governance rule, so a planner can point at
the cell that produced an action.

Matrix (flood row x deprivation column). Deprivation classes are the
Sprint 3 labels (low / medium / high):

    flood low   retain_formalise | retain_upgrade | in_situ_upgrade_priority
    flood high  flood_resilient_upgrade | flood_resilient_upgrade | compensated_resettlement

Two gates keep the irreversible action rare:

  1. High hazard is the U-Net probability at or above
     ``config.FUSION_FLOOD_THRESHOLD``, or -- when the override is on --
     intersection with an NRRP high-risk zone. The zone is a planning
     constraint, not a second flood model. It moves the parcel onto the
     high-hazard row; it does not by itself authorise resettlement.
  2. Compensated resettlement also needs a *majority* high-deprivation
     probability (P(high) > 0.5), not merely a plurality. A 0.40/0.15/0.45
     split on a flood-prone parcel is routed to in-place resilient upgrade.
     Ties in the class argmax already prefer the less severe class.

Policy validation reuses the Sprint 2 lift statistic on the *model* action
(before the NRRP override). Using the post-override action would make the
lift circular: the override forces every in-zone parcel onto a
flood-responsive action. The justice audit is separate and must pass even
when the lift does not -- it checks the engine, not the U-Net.

Real NRRP polygons and trained heads are a Colab run. Everything here is
numpy over arrays the caller already has.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config
from ..utils.label_validation import agreement_metrics

# Stable order. np.argmax on the soft distribution prefers the earlier
# action, and the displacement action is last so a tie is not resolved by
# evicting anyone. The hard decision does not use that argmax (it uses the
# matrix); the order is still the one summaries and saved tables share.
POLICY_ACTIONS = (
    "retain_formalise",
    "retain_upgrade",
    "in_situ_upgrade_priority",
    "flood_resilient_upgrade",
    "compensated_resettlement",
)

# flood bin (0 low, 1 high) x deprivation class (0 low, 1 medium, 2 high).
POLICY_MATRIX = (
    ("retain_formalise", "retain_upgrade", "in_situ_upgrade_priority"),
    ("flood_resilient_upgrade", "flood_resilient_upgrade", "compensated_resettlement"),
)

FLOOD_RESPONSIVE_ACTIONS = frozenset({
    "flood_resilient_upgrade",
    "compensated_resettlement",
})
DISPLACEMENT_ACTION = "compensated_resettlement"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

# Late-fusion feature vector, in concatenation order. Sprint 3 persists the
# softmax; this is the vector the matrix is applied to, before the NRRP
# override replaces the flood component for the policy decision.
FUSED_FEATURE_NAMES = (
    "flood_proba",
    "p_low",
    "p_medium",
    "p_high",
)

_HIGH_CLASS = config.DEPRIVATION_NUM_CLASSES - 1  # 2
_ACTION_TO_INDEX = {name: i for i, name in enumerate(POLICY_ACTIONS)}
_MATRIX_INDEX = np.array(
    [[_ACTION_TO_INDEX[name] for name in row] for row in POLICY_MATRIX],
    dtype=np.int64,
)


def action_labels(indices) -> np.ndarray:
    """Map action indices to names. ``-1`` is insufficient evidence."""
    idx = np.asarray(indices, dtype=np.int64)
    names = np.empty(idx.shape, dtype=object)
    invalid = idx < 0
    names[invalid] = INSUFFICIENT_EVIDENCE
    valid = ~invalid
    if np.any(valid):
        if np.any(idx[valid] >= len(POLICY_ACTIONS)):
            raise ValueError(f"action index out of range: {idx[valid].max()}")
        names[valid] = np.asarray(POLICY_ACTIONS, dtype=object)[idx[valid]]
    return names


def describe_policy_matrix() -> str:
    """Plain-text rendering of the matrix for notebooks and logs."""
    dep = list(config.DEPRIVATION_CLASS_NAMES)
    width = 26

    def cell(text: str) -> str:
        return f"{text:<{width}}"

    lines = [
        "flood \\ deprivation | " + " | ".join(cell(name) for name in dep),
        "flood low            | " + " | ".join(cell(POLICY_MATRIX[0][i]) for i in range(3)),
        "flood high           | " + " | ".join(cell(POLICY_MATRIX[1][i]) for i in range(3)),
        "Resettlement additionally requires P(high deprivation) > 0.5.",
        "NRRP zone membership forces the high-hazard row; it does not "
        "skip the deprivation gate.",
    ]
    return "\n".join(lines)


def _as_flood(flood_proba) -> np.ndarray:
    flood = np.asarray(flood_proba, dtype=np.float64)
    if flood.ndim == 2 and flood.shape[1] == 1:
        flood = flood[:, 0]
    if flood.ndim != 1:
        raise ValueError(
            f"flood_proba must be shape (N,), got {np.asarray(flood_proba).shape}"
        )
    return flood


def _as_deprivation(deprivation_proba, n: int) -> np.ndarray:
    dep = np.asarray(deprivation_proba, dtype=np.float64)
    n_classes = config.DEPRIVATION_NUM_CLASSES
    if dep.ndim != 2 or dep.shape[1] != n_classes:
        raise ValueError(
            f"deprivation_proba must be shape (N, {n_classes}), got {dep.shape}"
        )
    if dep.shape[0] != n:
        raise ValueError(
            f"flood_proba length {n} != deprivation_proba length {dep.shape[0]}"
        )
    return dep


def _as_mask(mask, n: int, name: str) -> np.ndarray:
    if mask is None:
        return np.zeros(n, dtype=bool)
    out = np.asarray(mask, dtype=bool)
    if out.shape != (n,):
        raise ValueError(f"{name} must be shape ({n},), got {out.shape}")
    return out


def _prepare_heads(flood: np.ndarray, dep: np.ndarray):
    """
    Split each parcel into usable vs insufficient evidence.

    A deprivation row is usable when it is finite, non-negative, and has
    positive mass (then it is renormalised, so a softmax that sums to
    1±1e-6 and a hand-built one-hot both work). A flood probability is
    usable when it is finite and inside [0, 1]. Out-of-range values are
    corrupt head outputs, not missing ones, and stay invalid even inside
    an NRRP zone.
    """
    dep_ok = np.isfinite(dep).all(axis=1) & (dep >= -1e-6).all(axis=1)
    clipped = np.clip(dep, 0.0, None)
    sums = clipped.sum(axis=1)
    dep_ok = dep_ok & (sums > 1e-8)
    renorm = np.zeros_like(clipped)
    renorm[dep_ok] = clipped[dep_ok] / sums[dep_ok, None]

    flood_ok = np.isfinite(flood) & (flood >= -1e-6) & (flood <= 1.0 + 1e-6)
    flood_clipped = np.full(flood.shape, np.nan, dtype=np.float64)
    flood_clipped[flood_ok] = np.clip(flood[flood_ok], 0.0, 1.0)
    return flood_ok, flood_clipped, dep_ok, renorm


def action_probability_mass(flood_high_p: np.ndarray, deprivation_proba: np.ndarray) -> np.ndarray:
    """
    Marginal probability of each policy action.

    ``flood_high_p`` is P(high hazard) in [0, 1] (the U-Net sigmoid, or 1
    after the NRRP override). Mass on the two matrix cells that share
    ``flood_resilient_upgrade`` is summed, so the row is a distribution
    over the five actions and sums to 1 on usable parcels.
    """
    p_high = np.asarray(flood_high_p, dtype=np.float64)
    dep = np.asarray(deprivation_proba, dtype=np.float64)
    p_low = 1.0 - p_high
    mass = np.zeros((p_high.shape[0], len(POLICY_ACTIONS)), dtype=np.float64)
    mass[:, _ACTION_TO_INDEX["retain_formalise"]] = p_low * dep[:, 0]
    mass[:, _ACTION_TO_INDEX["retain_upgrade"]] = p_low * dep[:, 1]
    mass[:, _ACTION_TO_INDEX["in_situ_upgrade_priority"]] = p_low * dep[:, 2]
    mass[:, _ACTION_TO_INDEX["flood_resilient_upgrade"]] = p_high * (dep[:, 0] + dep[:, 1])
    mass[:, _ACTION_TO_INDEX["compensated_resettlement"]] = p_high * dep[:, 2]
    return mass


def _hard_actions(flood_high: np.ndarray, dep_class: np.ndarray, p_high: np.ndarray):
    """
    Matrix lookup plus the majority gate on resettlement.

    Returns ``(gated_action, raw_matrix_action, majority_gate_applied)``.
    The gate rewrites only the resettlement cell, and only when P(high
    deprivation) is not a majority. Parcels the caller has already marked
    invalid are left for the caller to overwrite.
    """
    matrix_action = _MATRIX_INDEX[flood_high.astype(np.int64), dep_class]
    resettle = _ACTION_TO_INDEX[DISPLACEMENT_ACTION]
    resilient = _ACTION_TO_INDEX["flood_resilient_upgrade"]
    # Plurality-but-not-majority high deprivation must not be displaced.
    gate = (matrix_action == resettle) & ~(p_high > 0.5)
    action = matrix_action.copy()
    action[gate] = resilient
    return action, matrix_action, gate


def decide_policy(
    flood_proba,
    deprivation_proba,
    inside_nrrp=None,
    *,
    flood_threshold: float = None,
    nrrp_overrides_flood: bool = None,
) -> dict:
    """
    Fuse one parcel per row into a policy decision.

    Parameters
    ----------
    flood_proba :
        U-Net sigmoid, shape (N,). NaN marks a parcel the flood head did
        not cover.
    deprivation_proba :
        GraphSAGE softmax, shape (N, 3), column order
        ``config.DEPRIVATION_CLASS_NAMES``.
    inside_nrrp :
        Optional boolean mask, True where the parcel intersects an NRRP
        (or fallback river-corridor) polygon. None means the constraint
        is not applied.

    The returned dict holds both the model decision (heads only) and the
    policy decision (heads after the optional NRRP override). ``action_index``
    is the policy decision: -1 is ``insufficient_evidence``.
    """
    flood = _as_flood(flood_proba)
    n = int(flood.shape[0])
    dep = _as_deprivation(deprivation_proba, n)
    inside = _as_mask(inside_nrrp, n, "inside_nrrp")

    if flood_threshold is None:
        flood_threshold = config.FUSION_FLOOD_THRESHOLD
    flood_threshold = float(flood_threshold)
    if not 0.0 < flood_threshold <= 1.0:
        raise ValueError(f"flood_threshold must be in (0, 1], got {flood_threshold}")
    if nrrp_overrides_flood is None:
        nrrp_overrides_flood = config.FUSION_NRRP_OVERRIDES_FLOOD

    flood_ok, flood_clipped, dep_ok, dep_renorm = _prepare_heads(flood, dep)
    model_valid = flood_ok & dep_ok

    # Planning constraint: in-zone parcels are certain high hazard for the
    # policy row, including when the U-Net has no value there. A corrupt
    # (out-of-range) flood output is not "missing" and is not repaired.
    policy_flood = flood_clipped.copy()
    if nrrp_overrides_flood:
        policy_flood = np.where(inside, 1.0, policy_flood)
    # Out-of-range flood stays unusable even inside the zone: the where
    # above would have written 1.0. Put NaN back so the parcel is insufficient.
    corrupt_flood = np.isfinite(flood) & ~flood_ok
    policy_flood = np.where(corrupt_flood, np.nan, policy_flood)
    policy_valid = dep_ok & np.isfinite(policy_flood)

    dep_class = np.full(n, -1, dtype=np.int64)
    dep_class[dep_ok] = np.argmax(dep_renorm[dep_ok], axis=1).astype(np.int64)

    model_flood_high = np.zeros(n, dtype=bool)
    model_flood_high[model_valid] = flood_clipped[model_valid] >= flood_threshold
    policy_flood_high = np.zeros(n, dtype=bool)
    policy_flood_high[policy_valid] = policy_flood[policy_valid] >= flood_threshold

    model_action = np.full(n, -1, dtype=np.int64)
    policy_action = np.full(n, -1, dtype=np.int64)
    matrix_action = np.full(n, -1, dtype=np.int64)
    policy_gate = np.zeros(n, dtype=bool)
    p_high = dep_renorm[:, _HIGH_CLASS]

    if np.any(model_valid):
        m_action, _, _ = _hard_actions(
            model_flood_high[model_valid], dep_class[model_valid], p_high[model_valid],
        )
        model_action[model_valid] = m_action
    if np.any(policy_valid):
        p_action, raw_action, p_gate = _hard_actions(
            policy_flood_high[policy_valid], dep_class[policy_valid], p_high[policy_valid],
        )
        policy_action[policy_valid] = p_action
        matrix_action[policy_valid] = raw_action
        policy_gate[policy_valid] = p_gate

    model_mass = np.zeros((n, len(POLICY_ACTIONS)), dtype=np.float64)
    policy_mass = np.zeros((n, len(POLICY_ACTIONS)), dtype=np.float64)
    if np.any(model_valid):
        model_mass[model_valid] = action_probability_mass(
            flood_clipped[model_valid], dep_renorm[model_valid],
        )
    if np.any(policy_valid):
        policy_mass[policy_valid] = action_probability_mass(
            np.clip(policy_flood[policy_valid], 0.0, 1.0), dep_renorm[policy_valid],
        )

    confidence = np.zeros(n, dtype=np.float64)
    chosen = policy_action >= 0
    if np.any(chosen):
        confidence[chosen] = policy_mass[chosen, policy_action[chosen]]

    fused = np.full((n, len(FUSED_FEATURE_NAMES)), np.nan, dtype=np.float64)
    fused[:, 0] = flood_clipped
    fused[dep_ok, 1:] = dep_renorm[dep_ok]

    flood_responsive_idx = np.array(
        [i for i, name in enumerate(POLICY_ACTIONS) if name in FLOOD_RESPONSIVE_ACTIONS],
        dtype=np.int64,
    )
    displace_idx = _ACTION_TO_INDEX[DISPLACEMENT_ACTION]
    model_responsive = np.isin(model_action, flood_responsive_idx)
    policy_responsive = np.isin(policy_action, flood_responsive_idx)

    override_applied = (
        policy_valid
        & inside
        & nrrp_overrides_flood
        & (policy_action != model_action)
    )

    return {
        "n": n,
        "valid": policy_valid,
        "model_valid": model_valid,
        "flood_proba": flood_clipped,
        "deprivation_proba": dep_renorm,
        "fused_features": fused,
        "deprivation_class": dep_class,
        "p_high": p_high,
        "inside_nrrp": inside,
        "model_flood_high": model_flood_high,
        "policy_flood_high": policy_flood_high,
        "override_applied": override_applied,
        "majority_gate_applied": policy_gate,
        "model_action_index": model_action,
        "matrix_action_index": matrix_action,
        "action_index": policy_action,
        "action_proba": policy_mass,
        "model_action_proba": model_mass,
        "confidence": confidence,
        "model_flood_responsive": model_responsive,
        "flood_responsive": policy_responsive,
        "displacement": policy_action == displace_idx,
        "flood_threshold": flood_threshold,
    }


def justice_audit(decision: dict) -> dict:
    """
    Displacement is allowed only for a high-hazard parcel whose
    high-deprivation probability is a majority. Returns violation counts.
    A correct engine reports ``passes`` True for every input, including
    inputs that fail the NRRP lift bar.
    """
    valid = np.asarray(decision["valid"], dtype=bool)
    disp = np.asarray(decision["displacement"], dtype=bool) & valid
    dep_class = np.asarray(decision["deprivation_class"])
    p_high = np.asarray(decision["p_high"], dtype=np.float64)
    flood_high = np.asarray(decision["policy_flood_high"], dtype=bool)
    bad_class = disp & (dep_class != _HIGH_CLASS)
    bad_majority = disp & ~(p_high > 0.5)
    bad_flood = disp & ~flood_high
    # Insufficient-evidence parcels must not be marked displaced.
    bad_invalid = np.asarray(decision["displacement"], dtype=bool) & ~valid
    n_bad = int((bad_class | bad_majority | bad_flood | bad_invalid).sum())
    return {
        "n_valid": int(valid.sum()),
        "n_displacement": int(disp.sum()),
        "n_displacement_without_high_deprivation": int(bad_class.sum()),
        "n_displacement_without_majority": int(bad_majority.sum()),
        "n_displacement_without_high_flood": int(bad_flood.sum()),
        "n_violations": n_bad,
        "passes": n_bad == 0,
    }


def validate_against_reference(
    decision: dict,
    inside_reference,
    lift_target: float = None,
) -> dict:
    """
    Concentration of flood-responsive actions inside a reference zone.

    The exit bar uses the model decision (no NRRP override). The policy
    agreement is reported beside it so a notebook can show how much of the
    concentration was manufactured by the override. ``inside_reference`` is
    a boolean mask aligned to the decision rows; the same mask the engine
    was given as ``inside_nrrp`` is the usual choice.
    """
    reference = np.asarray(inside_reference)
    n = int(decision["n"])
    if reference.shape != (n,):
        raise ValueError(
            f"inside_reference must be shape ({n},), got {reference.shape}"
        )
    reference_bool = reference.astype(bool)
    if lift_target is None:
        lift_target = config.FUSION_NRRP_LIFT_TARGET

    model_mask = np.asarray(decision["model_valid"], dtype=bool)
    policy_mask = np.asarray(decision["valid"], dtype=bool)
    model_agreement = agreement_metrics(
        np.asarray(decision["model_flood_responsive"]).astype(float),
        reference_bool.astype(float),
        valid_mask=model_mask,
    )
    policy_agreement = agreement_metrics(
        np.asarray(decision["flood_responsive"]).astype(float),
        reference_bool.astype(float),
        valid_mask=policy_mask,
    )
    lift = float(model_agreement["lift"])
    meets = bool(
        model_agreement["valid_pixels"] > 0
        and model_agreement["reference_fraction"] > 0.0
        and lift > float(lift_target)
    )
    return {
        "model_agreement": model_agreement,
        "policy_agreement": policy_agreement,
        "lift": lift,
        "lift_target": float(lift_target),
        "meets_lift_target": meets,
        "n_override_applied": int(np.asarray(decision["override_applied"]).sum()),
        "n_model_valid": int(model_mask.sum()),
        "n_reference": int((reference_bool & model_mask).sum()),
    }


def _fraction_dict(indices: np.ndarray) -> dict:
    counts = {name: int(np.sum(indices == i)) for i, name in enumerate(POLICY_ACTIONS)}
    counts[INSUFFICIENT_EVIDENCE] = int(np.sum(indices < 0))
    total = int(indices.size)
    usable = total - counts[INSUFFICIENT_EVIDENCE]
    denom = max(usable, 1)
    fractions = {
        name: (counts[name] / denom if name != INSUFFICIENT_EVIDENCE else counts[name] / max(total, 1))
        for name in list(POLICY_ACTIONS) + [INSUFFICIENT_EVIDENCE]
    }
    return {
        "counts": counts,
        "fractions": fractions,
        "n": total,
        "n_valid": usable,
    }


def summarize_policy(decision: dict, settlement_index=None, settlement_keys=None) -> dict:
    """
    Action counts over policy decisions. Fractions are among valid parcels;
    the insufficient-evidence fraction is among all parcels.

    ``settlement_index`` aligns with decision rows. ``settlement_keys`` maps
    that index to a name; indices missing from the list are reported as
    ``str(index)``.
    """
    indices = np.asarray(decision["action_index"], dtype=np.int64)
    pooled = _fraction_dict(indices)
    pooled["n_displacement"] = int(np.asarray(decision["displacement"]).sum())
    pooled["n_override_applied"] = int(np.asarray(decision["override_applied"]).sum())
    pooled["n_majority_gate"] = int(np.asarray(decision["majority_gate_applied"]).sum())
    report = {"pooled": pooled}
    if settlement_index is None:
        return report

    sids = np.asarray(settlement_index, dtype=np.int64)
    if sids.shape != indices.shape:
        raise ValueError(
            f"settlement_index shape {sids.shape} != decisions {indices.shape}"
        )
    per = {}
    for sid in sorted(set(sids.tolist())):
        if settlement_keys is not None and 0 <= sid < len(settlement_keys):
            name = settlement_keys[sid]
        else:
            name = str(sid)
        per[name] = _fraction_dict(indices[sids == sid])
    report["per_settlement"] = per
    return report


def save_policy_table(path, decision: dict, settlement_index=None, settlement_keys=None) -> Path:
    """
    Write the decision arrays for the Sprint 6 dashboard. String action
    names are not stored; reload with ``action_labels``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for key, value in decision.items():
        if key == "n":
            continue
        if isinstance(value, (float, int, np.floating, np.integer)):
            payload[key] = np.asarray(value)
        else:
            payload[key] = np.asarray(value)
    if settlement_index is not None:
        payload["settlement_index"] = np.asarray(settlement_index, dtype=np.int64)
    if settlement_keys is not None:
        payload["settlement_keys"] = np.asarray(list(settlement_keys))
    np.savez_compressed(path, **payload)
    return path


def load_policy_table(path) -> dict:
    """
    Load a table written by ``save_policy_table``.

    ``np.savez_compressed`` already writes a zip-compressed ``.npz``. A
    second gzip pass (``.npz.gz``, or a ``.npz`` whose bytes are gzip)
    is unwrapped first. Drive's "compress" download is the usual source.
    """
    import gzip
    import io

    path = Path(path)
    blob = path.read_bytes()
    if blob[:2] == b"\x1f\x8b":
        blob = gzip.decompress(blob)
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def resolve_policy_table(extra_dirs=None) -> Path:
    """
    Find the Sprint 5 policy table.

    Search order is the repo ``data/`` folder, ``checkpoints/sprint5/``,
    then the Drive copies Colab uses. A gzip of the ``.npz`` is accepted.
    """
    names = ("policy_table.npz", "policy_table.npz.gz", "policy_table.npz.zip")
    roots = [
        Path("."),
        Path("data"),
        Path("checkpoints/sprint5"),
        Path("/content"),
        Path("/content/nairobi-risk-mapping"),
        Path("/content/nairobi-risk-mapping/data"),
        Path("/content/drive/MyDrive"),
        Path("/content/drive/MyDrive/nairobi_risk_mapping"),
        Path("/content/drive/MyDrive/nairobi_risk_mapping/data"),
        Path("/content/drive/MyDrive/nairobi_risk_mapping/checkpoints/sprint5"),
    ]
    if extra_dirs:
        roots = [Path(p) for p in extra_dirs] + roots
    tried = []
    for root in roots:
        for name in names:
            candidate = root / name
            tried.append(str(candidate))
            if candidate.is_file():
                return candidate
        if root.is_dir():
            for candidate in sorted(root.glob("policy_table.npz*")):
                if candidate.is_file():
                    return candidate
    raise FileNotFoundError(
        "No policy_table.npz (or .npz.gz) found. Looked in:\n  " + "\n  ".join(tried)
    )


def resolve_graphs_dir(extra_dirs=None) -> Path:
    """Directory that contains ``{settlement}_graph.pt`` files."""
    roots = [
        Path("data/graphs"),
        Path("/content/drive/MyDrive/nairobi_risk_mapping/data/graphs"),
    ]
    if extra_dirs:
        roots = [Path(p) for p in extra_dirs] + roots
    for root in roots:
        if root.is_dir() and any(root.glob("*_graph.pt")):
            return root
    raise FileNotFoundError(
        "No settlement graphs found. Expected data/graphs/{kibera,mathare,mukuru}_graph.pt "
        "or the same folder on Drive."
    )


def format_policy_report(summary: dict, validation: dict = None, justice: dict = None) -> str:
    """Multi-line text report used by the notebook and the offline demo."""
    lines = [describe_policy_matrix(), ""]
    pooled = summary["pooled"]
    lines.append(
        f"parcels: {pooled['n']}  valid: {pooled['n_valid']}  "
        f"displacement: {pooled['n_displacement']}  "
        f"nrrp_override_changed_action: {pooled['n_override_applied']}  "
        f"majority_gate: {pooled['n_majority_gate']}"
    )
    for name in list(POLICY_ACTIONS) + [INSUFFICIENT_EVIDENCE]:
        lines.append(
            f"  {name:<32} n={pooled['counts'][name]:<6} "
            f"frac={pooled['fractions'][name]:.3f}"
        )
    if "per_settlement" in summary:
        lines.append("per settlement (valid fractions):")
        for key, block in summary["per_settlement"].items():
            bits = [
                f"{name}={block['fractions'][name]:.2f}"
                for name in POLICY_ACTIONS
            ]
            lines.append(f"  {key:<10} n={block['n']:<5} " + " ".join(bits))
    if justice is not None:
        lines.append(
            f"justice_audit passes={justice['passes']} "
            f"violations={justice['n_violations']} "
            f"displaced={justice['n_displacement']}"
        )
    if validation is not None:
        lines.append(
            f"model lift={validation['lift']:.3f} "
            f"target>{validation['lift_target']:.3f} "
            f"meets={validation['meets_lift_target']} "
            f"(policy lift={validation['policy_agreement']['lift']:.3f}, "
            f"overrides={validation['n_override_applied']})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Joining head outputs onto parcels. Numpy cores stay import-light; the
# GeoTIFF / polygon helpers import rasterio and geopandas only when called.
# --------------------------------------------------------------------------- #
def _affine_coeffs(transform):
    """Accept an affine.Affine or a 6-tuple (a, b, c, d, e, f)."""
    if hasattr(transform, "a"):
        return (
            float(transform.a), float(transform.b), float(transform.c),
            float(transform.d), float(transform.e), float(transform.f),
        )
    coeffs = tuple(float(v) for v in transform)
    if len(coeffs) != 6:
        raise ValueError(f"transform must have 6 coefficients, got {len(coeffs)}")
    return coeffs


def rowcol_from_xy(transform, xs, ys) -> tuple:
    """
    Pixel row/col containing each point. ``transform`` maps (col, row) to
    (x, y) with the usual rasterio affine. Floor, so a point on a boundary
    belongs to the pixel whose origin is that boundary (rasterio's rule for
    the lower edge of a north-up pixel is the next row; points exactly on a
    grid line can land either side -- callers with parcel centroids rarely
    hit the line).
    """
    a, b, c, d, e, f = _affine_coeffs(transform)
    det = a * e - b * d
    if abs(det) < 1e-12:
        raise ValueError("raster transform is singular")
    x = np.asarray(xs, dtype=np.float64) - c
    y = np.asarray(ys, dtype=np.float64) - f
    col = (e * x - b * y) / det
    row = (-d * x + a * y) / det
    return np.floor(row).astype(np.int64), np.floor(col).astype(np.int64)


def sample_raster_at_xy(array, transform, xs, ys) -> np.ndarray:
    """
    Sample a 2-D array at world coordinates. Points outside the array are
    NaN (insufficient flood evidence), not zero -- zero would be read as a
    confident dry prediction.
    """
    grid = np.asarray(array, dtype=np.float64)
    if grid.ndim == 3:
        grid = np.squeeze(grid)
    if grid.ndim != 2:
        raise ValueError(f"array must be 2-D, got shape {np.asarray(array).shape}")
    rows, cols = rowcol_from_xy(transform, xs, ys)
    if rows.shape != cols.shape:
        raise ValueError("xs and ys must have the same shape")
    flat_rows = rows.ravel()
    flat_cols = cols.ravel()
    out = np.full(flat_rows.shape, np.nan, dtype=np.float64)
    h, w = grid.shape
    inside = (
        (flat_rows >= 0) & (flat_rows < h) & (flat_cols >= 0) & (flat_cols < w)
    )
    out[inside] = grid[flat_rows[inside], flat_cols[inside]]
    return out.reshape(rows.shape)


def sample_flood_at_pos(array, transform, pos, pos_crs=None, raster_crs=None) -> np.ndarray:
    """
    Sample a flood-probability grid at parcel centroids.

    ``pos`` is (N, 2) in ``pos_crs``. When that CRS differs from
    ``raster_crs``, coordinates are reprojected (pyproj) before sampling.
    """
    coords = np.asarray(pos, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"pos must be shape (N, 2), got {coords.shape}")
    xs, ys = coords[:, 0], coords[:, 1]
    if pos_crs is not None and raster_crs is not None and str(pos_crs) != str(raster_crs):
        from pyproj import Transformer
        transformer = Transformer.from_crs(pos_crs, raster_crs, always_xy=True)
        xs, ys = transformer.transform(xs, ys)
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
    return sample_raster_at_xy(array, transform, xs, ys)


def parcel_rowcol(pos, transform, pos_crs=None, raster_crs=None):
    """
    Raster row/col of each parcel centroid. Reprojects ``pos`` when the
    parcel CRS and the raster CRS differ. Same floor convention as
    ``sample_raster_at_xy``.
    """
    coords = np.asarray(pos, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"pos must be shape (N, 2), got {coords.shape}")
    xs, ys = coords[:, 0], coords[:, 1]
    if pos_crs is not None and raster_crs is not None and str(pos_crs) != str(raster_crs):
        from pyproj import Transformer
        transformer = Transformer.from_crs(pos_crs, raster_crs, always_xy=True)
        xs, ys = transformer.transform(xs, ys)
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
    return rowcol_from_xy(transform, xs, ys)


def select_orig_manifest_entries(entries) -> list:
    """
    One un-augmented tile window per ``(row_off, col_off)``.

    Training manifests store flips and rotations of the same window
    (``_flip_h``, ``_flip_v``, ``_rot90``). Those arrays are not in the
    original pixel frame, so sampling them at the manifest offset would
    read the wrong ground. ``_orig`` entries (and any entry without an
    augmentation suffix) are kept; the first one wins when a location
    appears in more than one split.
    """
    augmented = ("_flip_h", "_flip_v", "_rot90")
    chosen = {}
    for entry in entries:
        tile_id = str(entry.get("tile_id", ""))
        if tile_id.endswith(augmented):
            continue
        key = (int(entry["row_off"]), int(entry["col_off"]))
        chosen.setdefault(key, entry)
    return list(chosen.values())


def aggregate_tile_probabilities(parcel_rows, parcel_cols, tiles) -> np.ndarray:
    """
    Mean U-Net probability at each parcel across overlapping ``orig`` tiles.

    Each tile is a mapping with ``row_off``, ``col_off``, and ``proba``
    (2-D, local pixel frame: local row 0 is raster row ``row_off``). Pass
    only un-augmented windows -- flips and rotations are not in this frame.
    Parcels covered by no tile are NaN.
    """
    rows = np.asarray(parcel_rows, dtype=np.int64).reshape(-1)
    cols = np.asarray(parcel_cols, dtype=np.int64).reshape(-1)
    if rows.shape != cols.shape:
        raise ValueError("parcel_rows and parcel_cols must have the same length")
    acc = np.zeros(rows.shape[0], dtype=np.float64)
    count = np.zeros(rows.shape[0], dtype=np.int64)
    for tile in tiles:
        proba = np.asarray(tile["proba"], dtype=np.float64)
        if proba.ndim == 3:
            proba = np.squeeze(proba)
        if proba.ndim != 2:
            raise ValueError(f"tile proba must be 2-D, got shape {proba.shape}")
        r0 = int(tile["row_off"])
        c0 = int(tile["col_off"])
        height, width = proba.shape
        local_r = rows - r0
        local_c = cols - c0
        inside = (
            (local_r >= 0) & (local_r < height) & (local_c >= 0) & (local_c < width)
        )
        if not np.any(inside):
            continue
        vals = proba[local_r[inside], local_c[inside]]
        finite = np.isfinite(vals)
        idx = np.flatnonzero(inside)[finite]
        acc[idx] += vals[finite]
        count[idx] += 1
    out = np.full(rows.shape[0], np.nan, dtype=np.float64)
    covered = count > 0
    out[covered] = acc[covered] / count[covered]
    return out


def newest_geotiff(directories, prefix: str):
    """
    Newest ``.tif`` whose filename starts with ``prefix``.

    Earth Engine will not overwrite an export, so a re-run leaves
    ``kibera_raster_stack.tif`` and ``kibera_raster_stack (1).tif`` side by
    side in the Drive folder (the Sprint 1 export prefix), not under
    ``data/raster_stacks/``. Missing directories are skipped.
    """
    matches = []
    for directory in directories:
        directory = Path(directory)
        if not directory.exists():
            continue
        matches.extend(path for path in directory.glob(f"{prefix}*.tif") if path.is_file())
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def sample_geotiff_at_pos(path, pos, pos_crs) -> np.ndarray:
    """Sample band 1 of a GeoTIFF at ``pos`` (N, 2) in ``pos_crs``."""
    import rasterio

    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float64)
        if src.nodata is not None:
            array = np.where(array == src.nodata, np.nan, array)
        return sample_flood_at_pos(array, src.transform, pos, pos_crs, src.crs)


def parcel_reference_mask(parcels_gdf, reference_gdf) -> np.ndarray:
    """
    True where a parcel geometry intersects the reference polygons.

    Used for the NRRP zone (or a river-corridor fallback). Polygons are
    reprojected into the parcel CRS when both frames are set. Empty
    reference geometries yield an all-False mask rather than a crash.
    """
    n = len(parcels_gdf)
    if n == 0:
        return np.zeros((0,), dtype=bool)
    from shapely.ops import unary_union

    ref = reference_gdf
    if (
        parcels_gdf.crs is not None
        and reference_gdf.crs is not None
        and parcels_gdf.crs != reference_gdf.crs
    ):
        ref = reference_gdf.to_crs(parcels_gdf.crs)
    geoms = [g for g in ref.geometry if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(n, dtype=bool)
    union = unary_union(geoms)
    hits = parcels_gdf.geometry.intersects(union)
    return hits.fillna(False).to_numpy(dtype=bool)
