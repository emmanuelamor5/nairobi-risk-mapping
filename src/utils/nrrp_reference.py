"""
Score the fusion engine against the NRRP high-risk zones.

The Sprint 5 exit bar (``fusion.validate_against_reference``) asks whether
flood-responsive *model* actions concentrate inside a reference zone --
lift = precision / reference_fraction, strictly greater than
``config.FUSION_NRRP_LIFT_TARGET``. So far the only reference that has
actually been scored is the OSM river-corridor fallback. This module is the
missing piece for scoring the official Nairobi Rivers Regeneration Programme
(NRRP) 37-zone polygons, and for scoring them *honestly*:

  * ``load_nrrp_zones`` reads the digitized polygons from any geopandas
    format, repairs geometry, assigns ``zone_id``/``zone_name``, and
    reprojects to UTM 37S so buffers are in metres.
  * ``lift_by_buffer`` repeats the lift test with the zones grown by 0, 30
    and 60 m. NRRP lines are drawn to the riparian edge, so a parcel centroid
    a few metres outside should not flip the verdict -- if lift only clears
    the bar at 60 m, that is a boundary-precision problem, not a model one.
  * ``lift_by_settlement`` reports Kibera, Mathare and Mukuru separately,
    because a pooled lift above 1 can hide one settlement where the model is
    anti-correlated with the zones.
  * ``random_lift_distribution`` is a permutation null: the same number of
    flood-responsive parcels, placed at random. It gives the spread you need
    to say whether a lift of 1.05 is distinguishable from 1.0.
  * ``rule_lift`` scores a trivial rule (e.g. "within 120 m of a river") so
    the model can be compared with the obvious baseline.

Everything here is numpy/geopandas on in-memory frames, so it is unit-tested
offline with synthetic polygons. Only Colab has the real heads and the real
polygons; ``nrrp_validation_report`` + ``format_nrrp_report`` are what the
Sprint 5 notebook prints there.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from .. import config
from .label_validation import agreement_metrics

# Buffers (metres) at which the lift is re-measured. 0 is the headline.
NRRP_DEFAULT_BUFFERS_M = (0.0, 30.0, 60.0)
# Below this many model-valid parcels inside the zones the lift is noisy;
# the report flags it rather than quoting the number as a result.
NRRP_MIN_REFERENCE_PARCELS = 200
# Attribute names tried, in order, for a human-readable zone label.
ZONE_NAME_COLUMNS = (
    "zone_name", "name", "Name", "NAME", "zone", "Zone", "ZONE",
    "settlement", "Settlement", "label", "Label",
)
NRRP_FILE_STEM = "nrrp_high_risk_zones"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def default_nrrp_path(data_parent=None):
    """
    Conventional location of the NRRP polygons:
    ``<data_parent>/data/boundaries/nrrp_high_risk_zones.<ext>``. Returns the
    first existing candidate across config.BOUNDARY_FILE_EXTENSIONS, else None.
    """
    root = Path(data_parent) if data_parent is not None else Path(".")
    folder = root / config.BOUNDARY_DIR
    for ext in config.BOUNDARY_FILE_EXTENSIONS:
        candidate = folder / f"{NRRP_FILE_STEM}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_nrrp_zones(path, metric_crs: str = None):
    """
    Read the NRRP zone polygons into a GeoDataFrame in ``metric_crs``
    (default UTM 37S) with ``zone_id`` and ``zone_name`` columns.

    Non-polygon rows are dropped, invalid rings are repaired with
    ``buffer(0)``, and a file with no CRS is assumed to be WGS84 (the GeoJSON
    default). Raises if nothing polygonal remains, so a mis-exported file is
    caught before it silently yields an all-False mask and ``meets=False``.
    """
    import geopandas as gpd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NRRP zone file not found: {path}")
    gdf = gpd.read_file(path)
    return normalise_zones(gdf, metric_crs=metric_crs, source=str(path))


def normalise_zones(gdf, metric_crs: str = None, source: str = ""):
    """In-memory counterpart of load_nrrp_zones; shared with the tests."""
    metric_crs = metric_crs or config.CRS_METRIC
    if gdf.crs is None:
        warnings.warn(f"NRRP zones {source or '(in memory)'} have no CRS; assuming {config.CRS_WGS84}.")
        gdf = gdf.set_crs(config.CRS_WGS84)
    keep = gdf.geometry.notna() & ~gdf.geometry.is_empty
    keep &= gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    zones = gdf.loc[keep].copy()
    if len(zones) == 0:
        raise ValueError(f"NRRP zones {source or '(in memory)'} contain no polygon geometries.")
    zones["geometry"] = zones.geometry.buffer(0)
    zones = zones.to_crs(metric_crs).reset_index(drop=True)
    if "zone_id" not in zones.columns:
        zones["zone_id"] = np.arange(1, len(zones) + 1)
    if "zone_name" not in zones.columns:
        name_col = next((c for c in ZONE_NAME_COLUMNS if c in zones.columns), None)
        zones["zone_name"] = (
            zones[name_col].astype(str) if name_col is not None
            else [f"zone_{i}" for i in zones["zone_id"]]
        )
    return zones


def zones_area_km2(zones) -> float:
    return float(zones.to_crs(config.CRS_METRIC).area.sum() / 1e6)


def buffer_zones(zones, buffer_m: float):
    """
    Grow (or shrink, for negative values) every zone by ``buffer_m`` metres.
    Zones must already be in a metric CRS (load_nrrp_zones guarantees it).
    """
    if zones.crs is None or zones.crs.is_geographic:
        raise ValueError("buffer_zones needs zones in a projected (metre) CRS.")
    out = zones.copy()
    if float(buffer_m) != 0.0:
        out["geometry"] = out.geometry.buffer(float(buffer_m))
    return out


# --------------------------------------------------------------------------- #
# Parcel membership
# --------------------------------------------------------------------------- #
def parcels_from_positions(pos, crs: str = None):
    """Point GeoDataFrame from an (N, 2) array of parcel centroids."""
    import geopandas as gpd
    from shapely.geometry import Point

    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"pos must be (N, 2), got {pos.shape}")
    return gpd.GeoDataFrame(
        geometry=[Point(float(x), float(y)) for x, y in pos],
        crs=crs or config.CRS_METRIC,
    )


def parcel_zone_mask(parcels_gdf, zones, buffer_m: float = 0.0) -> np.ndarray:
    """
    Boolean mask, aligned to ``parcels_gdf`` rows, of parcels intersecting
    the zones grown by ``buffer_m``. CRS mismatches are resolved explicitly
    (zones are reprojected onto the parcels) instead of silently yielding
    zero overlaps.
    """
    from ..models.fusion import parcel_reference_mask

    if parcels_gdf.crs is None:
        raise ValueError("parcels_gdf has no CRS; set it to the graph's pos_crs.")
    grown = buffer_zones(zones, buffer_m) if float(buffer_m) != 0.0 else zones
    return parcel_reference_mask(parcels_gdf, grown)


# --------------------------------------------------------------------------- #
# Lift variants
# --------------------------------------------------------------------------- #
def subset_decision(decision: dict, mask) -> dict:
    """
    Restrict a ``decide_policy`` result to the rows where ``mask`` is True so
    ``validate_against_reference`` can be run per settlement. Every array
    whose leading dimension is ``n`` is sliced; scalars are kept.
    """
    mask = np.asarray(mask, dtype=bool)
    n = int(decision["n"])
    if mask.shape != (n,):
        raise ValueError(f"mask must be shape ({n},), got {mask.shape}")
    out = {}
    for key, value in decision.items():
        if key == "n":
            continue
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == n:
            out[key] = arr[mask]
        else:
            out[key] = value
    out["n"] = int(mask.sum())
    return out


def lift_by_buffer(decision: dict, parcels_gdf, zones,
                   buffers_m=NRRP_DEFAULT_BUFFERS_M, lift_target: float = None) -> dict:
    """``{buffer_m: validate_against_reference report}`` for each buffer."""
    from ..models.fusion import validate_against_reference

    reports = {}
    for buffer_m in buffers_m:
        inside = parcel_zone_mask(parcels_gdf, zones, buffer_m=buffer_m)
        report = validate_against_reference(decision, inside, lift_target=lift_target)
        report["buffer_m"] = float(buffer_m)
        report["inside_mask"] = inside
        reports[float(buffer_m)] = report
    return reports


def lift_by_settlement(decision: dict, inside, settlement_index, settlement_keys,
                       lift_target: float = None) -> dict:
    """
    Per-settlement lift reports plus ``"pooled"``. ``settlement_index`` maps
    each row to a position in ``settlement_keys`` (the same convention as
    ``fusion.summarize_policy``).
    """
    from ..models.fusion import validate_against_reference

    inside = np.asarray(inside, dtype=bool)
    sid = np.asarray(settlement_index)
    n = int(decision["n"])
    if inside.shape != (n,) or sid.shape != (n,):
        raise ValueError(f"inside and settlement_index must be shape ({n},)")
    out = {"pooled": validate_against_reference(decision, inside, lift_target=lift_target)}
    for i, key in enumerate(settlement_keys):
        mask = sid == i
        if not mask.any():
            continue
        out[key] = validate_against_reference(
            subset_decision(decision, mask), inside[mask], lift_target=lift_target,
        )
    return out


def _observed_lift(decision: dict, inside: np.ndarray) -> dict:
    return agreement_metrics(
        np.asarray(decision["model_flood_responsive"]).astype(float),
        inside.astype(float),
        valid_mask=np.asarray(decision["model_valid"], dtype=bool),
    )


def random_lift_distribution(decision: dict, inside, n_draws: int = 500, seed: int = 0) -> dict:
    """
    Permutation null for the lift: shuffle which model-valid parcels are
    flood-responsive (keeping their count) and recompute lift each time.

    Returns the observed lift, the null mean and 2.5/97.5 percentiles, and a
    one-sided p-value (share of draws at least as large as the observed
    lift, with the +1 correction). NaNs when there is nothing to permute.
    """
    inside = np.asarray(inside, dtype=bool)
    valid = np.asarray(decision["model_valid"], dtype=bool)
    responsive = np.asarray(decision["model_flood_responsive"], dtype=bool)
    n = int(decision["n"])
    if inside.shape != (n,):
        raise ValueError(f"inside must be shape ({n},), got {inside.shape}")

    observed = _observed_lift(decision, inside)
    k = int((responsive & valid).sum())
    ref_in_valid = int((inside & valid).sum())
    result = {
        "observed_lift": float(observed["lift"]),
        "n_draws": int(n_draws),
        "n_responsive": k,
        "n_reference": ref_in_valid,
        "null_mean": float("nan"),
        "null_p2_5": float("nan"),
        "null_p97_5": float("nan"),
        "p_value": float("nan"),
    }
    if k == 0 or ref_in_valid == 0 or n_draws <= 0 or valid.sum() == 0:
        return result

    rng = np.random.default_rng(seed)
    valid_idx = np.flatnonzero(valid)
    ref_valid = inside[valid_idx]
    base_rate = ref_in_valid / valid_idx.size
    draws = np.empty(n_draws, dtype=np.float64)
    for d in range(n_draws):
        pick = rng.choice(valid_idx.size, size=k, replace=False)
        precision = ref_valid[pick].mean()
        draws[d] = precision / base_rate
    result.update({
        "null_mean": float(draws.mean()),
        "null_p2_5": float(np.percentile(draws, 2.5)),
        "null_p97_5": float(np.percentile(draws, 97.5)),
        "p_value": float((1 + np.sum(draws >= result["observed_lift"])) / (n_draws + 1)),
    })
    return result


def rule_lift(rule_mask, inside, valid_mask=None) -> dict:
    """
    Lift of a trivial rule (e.g. "centroid within 120 m of an OSM river")
    against the same zones -- the baseline the model should beat.
    """
    rule = np.asarray(rule_mask, dtype=bool).astype(float)
    inside = np.asarray(inside, dtype=bool).astype(float)
    return agreement_metrics(rule, inside, valid_mask=valid_mask)


# --------------------------------------------------------------------------- #
# One-call report for the notebook
# --------------------------------------------------------------------------- #
def nrrp_validation_report(decision: dict, parcels_gdf, zones,
                           settlement_index=None, settlement_keys=None,
                           buffers_m=NRRP_DEFAULT_BUFFERS_M, rule_masks: dict = None,
                           n_draws: int = 500, seed: int = 0,
                           lift_target: float = None) -> dict:
    """
    Everything the Sprint 5 notebook needs to quote an NRRP lift honestly.

    ``rule_masks`` is an optional ``{name: bool array}`` of baseline rules.
    The headline (``lift``, ``meets_lift_target``, ``n_reference``) is the
    0 m buffer, or the smallest buffer requested.
    """
    buffers_m = tuple(float(b) for b in buffers_m)
    if not buffers_m:
        raise ValueError("buffers_m must list at least one buffer.")
    by_buffer = lift_by_buffer(decision, parcels_gdf, zones, buffers_m, lift_target)
    headline_buffer = min(buffers_m)
    headline = by_buffer[headline_buffer]
    inside0 = headline["inside_mask"]

    per_settlement = {}
    if settlement_index is not None and settlement_keys is not None:
        for buffer_m, report in by_buffer.items():
            per_settlement[buffer_m] = lift_by_settlement(
                decision, report["inside_mask"], settlement_index, settlement_keys, lift_target,
            )

    valid = np.asarray(decision["model_valid"], dtype=bool)
    rules = {}
    for name, mask in (rule_masks or {}).items():
        rules[name] = rule_lift(mask, inside0, valid_mask=valid)

    return {
        "n_zones": int(len(zones)),
        "zone_area_km2": zones_area_km2(zones),
        "headline_buffer_m": headline_buffer,
        "lift": float(headline["lift"]),
        "lift_target": float(headline["lift_target"]),
        "meets_lift_target": bool(headline["meets_lift_target"]),
        "n_reference": int(headline["n_reference"]),
        "reference_too_small": int(headline["n_reference"]) < NRRP_MIN_REFERENCE_PARCELS,
        "by_buffer": by_buffer,
        "per_settlement": per_settlement,
        "null": random_lift_distribution(decision, inside0, n_draws=n_draws, seed=seed),
        "rules": rules,
    }


def format_nrrp_report(report: dict) -> str:
    """Plain-text rendering of nrrp_validation_report."""
    lines = [
        f"NRRP zones: {report['n_zones']} polygons, {report['zone_area_km2']:.2f} km^2",
        (
            f"Headline lift ({report['headline_buffer_m']:.0f} m buffer): "
            f"{report['lift']:.3f}  target > {report['lift_target']:.2f}  "
            f"meets={report['meets_lift_target']}  n_reference={report['n_reference']}"
        ),
    ]
    if report["reference_too_small"]:
        lines.append(
            f"  WARNING: fewer than {NRRP_MIN_REFERENCE_PARCELS} model-valid parcels fall "
            f"inside the zones; this lift is noisy. Check the zone file covers the settlements."
        )
    lines.append("Lift by buffer:")
    for buffer_m, rep in sorted(report["by_buffer"].items()):
        lines.append(
            f"  {buffer_m:5.0f} m  lift={rep['lift']:.3f}  meets={rep['meets_lift_target']}  "
            f"n_reference={rep['n_reference']}  "
            f"precision={rep['model_agreement']['precision']:.3f}  "
            f"recall={rep['model_agreement']['recall']:.3f}"
        )
    if report["per_settlement"]:
        lines.append("Lift by settlement:")
        for buffer_m, per in sorted(report["per_settlement"].items()):
            parts = []
            for key, rep in per.items():
                if key == "pooled":
                    continue
                parts.append(f"{key}={rep['lift']:.3f} (n_ref {rep['n_reference']})")
            lines.append(f"  {buffer_m:5.0f} m  " + "  ".join(parts))
    null = report["null"]
    if np.isfinite(null["null_mean"]):
        lines.append(
            f"Permutation null ({null['n_draws']} draws, {null['n_responsive']} responsive parcels): "
            f"mean {null['null_mean']:.3f}, 95% band [{null['null_p2_5']:.3f}, {null['null_p97_5']:.3f}], "
            f"p={null['p_value']:.3f}"
        )
    else:
        lines.append("Permutation null: not computable (no responsive parcels or empty reference).")
    if report["rules"]:
        lines.append("Baseline rules (same zones):")
        for name, metrics in report["rules"].items():
            lines.append(f"  {name}: lift={metrics['lift']:.3f}  precision={metrics['precision']:.3f}")
    return "\n".join(lines)
