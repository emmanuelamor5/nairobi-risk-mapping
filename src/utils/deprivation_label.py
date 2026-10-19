"""
Structural-deprivation label construction for the GraphSAGE head.

The proposal (FR-04 / Sec 3.2) asks for node-level multi-class classification
across three severity categories derived from the UN-Habitat SUPI framework.
Real SUPI scores come from household surveys; this project has no field
enumeration, so the label is an **OSM-derived proxy** of the five UN-Habitat
indicators that can be observed from building footprints and infrastructure
proximity:

  1. Housing durability  -- roof / building material tags (when present)
  2. Water access        -- distance to nearest drinking-water point
  3. Sanitation access   -- distance to nearest toilets amenity
  4. Overcrowding        -- footprint area as a density proxy
  5. Access / connectivity -- distance to nearest road (weak tenure/access
     stand-in; true security-of-tenure is not recoverable from OSM)

Each indicator is scored in [0, 1] (higher = more deprived), averaged into a
continuous deprivation score, then cut into three tertile classes so the
GraphSAGE head sees a balanced multi-class target. Tertile cuts are
settlement-relative by default: what matters for the late-fusion policy
matrix is relative severity *within* a settlement, not an absolute city-wide
score that would collapse when OSM coverage differs between Kibera and Mukuru.

This module is pure numpy / pandas-free so it stays unit-testable offline
without network access to OSM. ``graph_utils.enrich_buildings_with_deprivation``
is the GeoDataFrame-facing wrapper used by the data pipeline.
"""

from __future__ import annotations

import numpy as np

from .. import config

# Node-feature column order consumed by GraphSAGE. Keep this tuple the single
# source of truth -- graph_utils.graph_to_pyg_data and the model wrapper both
# derive in_channels from its length.
NODE_FEATURE_NAMES = (
    "area_m2",
    "dist_to_water_m",
    "dist_to_sanitation_m",
    "dist_to_road_m",
    "roof_material_score",
)

# OSM roof / building material tags -> durability deprivation in [0, 1].
# Values follow a coarse permanent (0) -> temporary (1) spectrum; unknown
# materials land at the midpoint so they neither dominate nor vanish.
_ROOF_MATERIAL_SCORES = {
    "concrete": 0.0,
    "tiles": 0.1,
    "tile": 0.1,
    "metal": 0.25,
    "tin": 0.3,
    "corrugated": 0.3,
    "iron": 0.3,
    "asbestos": 0.45,
    "wood": 0.55,
    "thatch": 0.85,
    "plastic": 0.9,
    "tarpaulin": 0.95,
    "cardboard": 1.0,
    "mud": 0.9,
    "grass": 0.9,
}


def encode_roof_material(materials) -> np.ndarray:
    """
    Map OSM ``roof:material`` / ``building:material`` strings to a durability
    deprivation score in [0, 1]. ``None`` / missing / unrecognised materials
    score 0.5 (neutral) so sparse tagging does not invent a false signal.
    """
    if materials is None:
        return np.array([], dtype=np.float64)

    # pandas Series / Index -> ndarray without forcing a 0-d object.
    if hasattr(materials, "to_numpy"):
        values = materials.to_numpy(dtype=object)
    else:
        values = np.asarray(materials, dtype=object)
    if values.ndim == 0:
        values = values.reshape(1)
    out = np.full(values.shape, 0.5, dtype=np.float64)
    for i, raw in enumerate(values.ravel()):
        if raw is None:
            continue
        key = str(raw).strip().lower()
        if not key or key in ("nan", "none", "unknown"):
            continue
        # OSM sometimes stores semicolon-separated multi-values; take the
        # most-deprived tag so mixed temporary+permanent roofs stay flagged.
        parts = [p.strip() for p in key.split(";") if p.strip()]
        scores = [_ROOF_MATERIAL_SCORES[p] for p in parts if p in _ROOF_MATERIAL_SCORES]
        if scores:
            out.ravel()[i] = max(scores)
    return out.reshape(values.shape)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def score_deprivation_indicators(
    area_m2: np.ndarray,
    dist_to_water_m: np.ndarray,
    dist_to_sanitation_m: np.ndarray,
    dist_to_road_m: np.ndarray,
    roof_material_score: np.ndarray = None,
    water_access_m: float = None,
    sanitation_access_m: float = None,
    road_access_m: float = None,
    overcrowding_area_m2: float = None,
) -> np.ndarray:
    """
    Combine the five UN-Habitat-style proxies into a continuous deprivation
    score in [0, 1] (higher = more deprived). Distance features are linearly
    scaled against the configured access radii; overcrowding is 1 when the
    footprint is at or below the overcrowding area threshold and decays
    toward 0 as area grows; roof material is already in [0, 1].
    """
    area = np.asarray(area_m2, dtype=np.float64)
    d_water = np.asarray(dist_to_water_m, dtype=np.float64)
    d_san = np.asarray(dist_to_sanitation_m, dtype=np.float64)
    d_road = np.asarray(dist_to_road_m, dtype=np.float64)

    water_access_m = water_access_m if water_access_m is not None else config.DEPRIVATION_WATER_ACCESS_M
    sanitation_access_m = (
        sanitation_access_m if sanitation_access_m is not None
        else config.DEPRIVATION_SANITATION_ACCESS_M
    )
    road_access_m = road_access_m if road_access_m is not None else config.DEPRIVATION_ROAD_ACCESS_M
    overcrowding_area_m2 = (
        overcrowding_area_m2 if overcrowding_area_m2 is not None
        else config.DEPRIVATION_OVERCROWDING_AREA_M2
    )

    water_score = _clip01(d_water / max(water_access_m, 1e-6))
    sanitation_score = _clip01(d_san / max(sanitation_access_m, 1e-6))
    road_score = _clip01(d_road / max(road_access_m, 1e-6))
    # Small footprints -> high overcrowding score; large footprints -> low.
    overcrowding_score = _clip01(overcrowding_area_m2 / np.maximum(area, 1e-6))

    if roof_material_score is None:
        roof_score = np.full(area.shape, 0.5, dtype=np.float64)
    else:
        roof_score = _clip01(np.asarray(roof_material_score, dtype=np.float64))

    # Equal-weight mean across the five indicators. A survey-calibrated
    # weighting (e.g. SUPI component weights) can replace this later without
    # changing the GraphSAGE input contract.
    return (water_score + sanitation_score + road_score + overcrowding_score + roof_score) / 5.0


def assign_deprivation_classes(
    scores: np.ndarray,
    n_classes: int = None,
    cuts: np.ndarray = None,
) -> np.ndarray:
    """
    Bin continuous deprivation scores into ``n_classes`` severity labels
    (default 3). When ``cuts`` is None, uses settlement-relative tertiles
    (33rd / 66th percentiles) so each class is roughly balanced. Pass
    explicit ``cuts`` of length ``n_classes - 1`` to freeze thresholds
    across settlements (useful once Sprint 4 trains on all three together).
    """
    scores = np.asarray(scores, dtype=np.float64)
    n_classes = n_classes or config.DEPRIVATION_NUM_CLASSES
    if scores.size == 0:
        return np.array([], dtype=np.int64)
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")

    if cuts is None:
        # Equal-frequency bins. With fewer unique scores than classes (tiny
        # synthetic graphs), percentile cuts can collide -- fall back to
        # equal-width cuts on the observed range so every class remains
        # reachable in principle.
        quantiles = np.linspace(0, 100, n_classes + 1)[1:-1]
        cuts = np.percentile(scores, quantiles)
        if np.unique(cuts).size < cuts.size:
            lo, hi = float(scores.min()), float(scores.max())
            if hi <= lo:
                return np.zeros(scores.shape, dtype=np.int64)
            cuts = np.linspace(lo, hi, n_classes + 1)[1:-1]

    cuts = np.asarray(cuts, dtype=np.float64)
    if cuts.size != n_classes - 1:
        raise ValueError(f"cuts must have length {n_classes - 1}, got {cuts.size}")

    classes = np.digitize(scores, cuts, right=False).astype(np.int64)
    return np.clip(classes, 0, n_classes - 1)


def class_distribution(labels: np.ndarray, n_classes: int = None) -> dict:
    """Return per-class counts and fractions for logging / notebook diagnostics."""
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = n_classes or config.DEPRIVATION_NUM_CLASSES
    counts = {i: int(np.sum(labels == i)) for i in range(n_classes)}
    total = max(int(labels.size), 1)
    return {
        "counts": counts,
        "fractions": {i: counts[i] / total for i in range(n_classes)},
        "n_nodes": int(labels.size),
        "class_names": list(config.DEPRIVATION_CLASS_NAMES[:n_classes]),
    }
