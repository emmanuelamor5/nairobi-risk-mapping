"""
Folium map of parcel policy actions.

The shell follows the supplied wireframes: choropleth map with a parcel
card, then an XAI summary. This module builds the Folium map and the
parcel records those screens read. Both are returned only after the
auth gate. Feature labels are the five GraphSAGE inputs, not the
placeholder names drawn on the wireframe.

Coordinates may be geographic (EPSG:4326) or projected. Projected
positions are transformed with pyproj before Folium sees them.
"""

from __future__ import annotations

import numpy as np

from .. import config
from ..models.fusion import POLICY_ACTIONS, action_labels

# Colour is the action. The three governance colours match the Folium
# dashboard frame (formalise green, upgrade gold, resettlement coral).
# The other matrix cells stay in that palette so they remain distinct.
# Labels for the five model inputs. The wireframe's example names
# (floodplain distance, structural density, and so on) are not features
# this GraphSAGE was trained on.
FEATURE_LABELS = {
    "area_m2": "Footprint area (standardised)",
    "dist_to_water_m": "Distance to water (standardised)",
    "dist_to_sanitation_m": "Distance to sanitation (standardised)",
    "dist_to_road_m": "Distance to road (standardised)",
    "roof_material_score": "Roof material score (standardised)",
}

PARCEL_PREFIX = {"kibera": "KB", "mathare": "MA", "mukuru": "MU"}


ACTION_COLORS = {
    "retain_formalise": "#28835b",
    "retain_upgrade": "#d6a344",
    "in_situ_upgrade_priority": "#354a60",
    "flood_resilient_upgrade": "#6e8b9a",
    "compensated_resettlement": "#cf655e",
    "insufficient_evidence": "#a7b4c1",
}


def to_lonlat(positions, pos_crs: str = None):
    """
    Return longitude, latitude arrays. Positions already in EPSG:4326 are
    returned unchanged. Any other CRS is projected with pyproj.
    """
    coords = np.asarray(positions, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"positions must be (N, 2), got {coords.shape}")
    crs = pos_crs or "EPSG:4326"
    if str(crs) in {"EPSG:4326", "WGS84"}:
        return coords[:, 0].copy(), coords[:, 1].copy()
    from pyproj import Transformer

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(coords[:, 0], coords[:, 1])
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def join_graph_positions(table: dict, graphs: dict) -> dict:
    """
    Attach ``positions`` from per-settlement graphs onto a Sprint 5 policy
    table. Graphs are concatenated in ``settlement_keys`` order, which is
    the order the notebook saved ``settlement_index``.
    """
    if "settlement_keys" not in table:
        raise ValueError("policy table has no settlement_keys.")
    keys = [str(key) for key in table["settlement_keys"]]
    parts = []
    feature_parts = []
    feature_names = None
    crs = None
    for key in keys:
        if key not in graphs:
            raise ValueError(f"No graph loaded for '{key}'.")
        pos = graphs[key].pos
        if pos is None:
            raise ValueError(f"Graph '{key}' has no data.pos.")
        arr = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
        parts.append(np.asarray(arr, dtype=np.float64))
        raw_x = getattr(graphs[key], "x", None)
        if raw_x is not None:
            feature_parts.append(
                raw_x.detach().cpu().numpy() if hasattr(raw_x, "detach") else np.asarray(raw_x)
            )
        crs = crs or getattr(graphs[key], "pos_crs", None)
        feature_names = feature_names or list(getattr(graphs[key], "feature_names", []) or [])
    positions = np.concatenate(parts, axis=0)
    n_actions = int(np.asarray(table["action_index"]).shape[0])
    if positions.shape[0] != n_actions:
        raise ValueError(
            f"Joined positions {positions.shape[0]} != policy rows {n_actions}."
        )
    joined = dict(table)
    joined["positions"] = positions
    joined["pos_crs"] = crs or "EPSG:32737"
    if feature_parts and sum(len(part) for part in feature_parts) == n_actions:
        joined["features"] = np.concatenate(feature_parts, axis=0)
        if feature_names:
            joined["feature_names"] = feature_names
    return joined


def _parcel_id(settlement: str, local_index: int) -> str:
    prefix = PARCEL_PREFIX.get(settlement, "P")
    return f"{prefix}-{int(local_index):05d}"


def parcel_rows(records: dict, settlement: str = None) -> list:
    """
    One dict per parcel for the choropleth and the XAI summary.

    Coordinates are included. Callers must already have passed the MFA gate.
    Feature columns, when the graph was joined, are the standardised inputs
    GraphSAGE actually sees.
    """
    positions = np.asarray(records["positions"], dtype=np.float64)
    actions = np.asarray(records["action_index"], dtype=np.int64).reshape(-1)
    if positions.shape[0] != actions.shape[0]:
        raise ValueError(
            f"{positions.shape[0]} positions and {actions.shape[0]} actions."
        )
    lon, lat = to_lonlat(positions, records.get("pos_crs"))
    labels = action_labels(actions)
    keys = [str(k) for k in records.get("settlement_keys", [])]
    sindex = records.get("settlement_index")
    sindex = None if sindex is None else np.asarray(sindex).reshape(-1)
    flood = records.get("flood_proba")
    flood = None if flood is None else np.asarray(flood, dtype=np.float64).reshape(-1)
    deprivation = records.get("deprivation_class")
    deprivation = None if deprivation is None else np.asarray(deprivation, dtype=np.int64).reshape(-1)
    confidence = records.get("confidence")
    confidence = None if confidence is None else np.asarray(confidence, dtype=np.float64).reshape(-1)
    proba = records.get("deprivation_proba")
    proba = None if proba is None else np.asarray(proba, dtype=np.float64)
    features = records.get("features")
    features = None if features is None else np.asarray(features, dtype=np.float64)
    feature_names = list(records.get("feature_names") or list(FEATURE_LABELS))
    local = {key: 0 for key in keys}

    rows = []
    for i in range(actions.shape[0]):
        if not (np.isfinite(lon[i]) and np.isfinite(lat[i])):
            continue
        settlement_name = ""
        if sindex is not None and keys and 0 <= int(sindex[i]) < len(keys):
            settlement_name = keys[int(sindex[i])]
        if settlement is not None and settlement_name != settlement:
            continue
        number = local.get(settlement_name, i)
        local[settlement_name] = number + 1
        dep_name = ""
        if deprivation is not None and int(deprivation[i]) >= 0:
            dep_name = config.DEPRIVATION_CLASS_NAMES[int(deprivation[i])]
        class_probability = None
        class_probabilities = None
        if proba is not None and proba.ndim == 2 and i < proba.shape[0]:
            class_probability = float(np.max(proba[i]))
            names = list(config.DEPRIVATION_CLASS_NAMES)
            class_probabilities = [
                {
                    "name": names[j] if j < len(names) else str(j),
                    "value": float(proba[i, j]),
                }
                for j in range(proba.shape[1])
            ]
        row = {
            "index": int(i),
            "id": _parcel_id(settlement_name or "parcel", number + 1),
            "settlement": settlement_name,
            "lat": float(lat[i]),
            "lon": float(lon[i]),
            "action": str(labels[i]),
            "flood_proba": None if flood is None or not np.isfinite(flood[i]) else float(flood[i]),
            "deprivation_class": dep_name,
            "class_probability": class_probability,
            "class_probabilities": class_probabilities,
            "confidence": None if confidence is None or not np.isfinite(confidence[i]) else float(confidence[i]),
        }
        if features is not None and i < features.shape[0]:
            row["features"] = [
                {
                    "name": feature_names[j] if j < len(feature_names) else str(j),
                    "label": FEATURE_LABELS.get(
                        feature_names[j] if j < len(feature_names) else "", 
                        feature_names[j] if j < len(feature_names) else str(j),
                    ),
                    "value": float(features[i, j]),
                }
                for j in range(features.shape[1])
            ]
        rows.append(row)
    return rows


def action_counts(action_index) -> dict:
    """Counts keyed by policy action name, including insufficient evidence."""
    labels = action_labels(np.asarray(action_index, dtype=np.int64).reshape(-1))
    names = list(POLICY_ACTIONS) + ["insufficient_evidence"]
    return {name: int(np.sum(labels == name)) for name in names}


def build_policy_map(
    positions,
    action_index,
    flood_proba=None,
    deprivation_class=None,
    settlement_index=None,
    settlement_keys=None,
    pos_crs: str = None,
    settlement: str = None,
):
    """
    Folium map of one circle per parcel, coloured by policy action.

    ``settlement`` keeps a single settlement when ``settlement_index`` and
    ``settlement_keys`` are provided. Empty input returns a map centred on
    Nairobi with no markers.
    """
    import folium

    positions = np.asarray(positions, dtype=np.float64)
    actions = np.asarray(action_index, dtype=np.int64).reshape(-1)
    if positions.shape[0] != actions.shape[0]:
        raise ValueError(
            f"{positions.shape[0]} positions and {actions.shape[0]} actions."
        )
    keep = np.ones(actions.shape[0], dtype=bool)
    if settlement is not None:
        if settlement_index is None or settlement_keys is None:
            raise ValueError("settlement filter needs settlement_index and settlement_keys.")
        keys = list(settlement_keys)
        if settlement not in keys:
            raise ValueError(f"Unknown settlement '{settlement}'. Options: {keys}")
        keep = np.asarray(settlement_index).reshape(-1) == keys.index(settlement)

    lon, lat = to_lonlat(positions, pos_crs)
    finite = np.isfinite(lon) & np.isfinite(lat) & keep
    labels = action_labels(actions)
    flood = None if flood_proba is None else np.asarray(flood_proba, dtype=np.float64).reshape(-1)
    deprivation = (
        None if deprivation_class is None
        else np.asarray(deprivation_class, dtype=np.int64).reshape(-1)
    )
    keys = list(settlement_keys) if settlement_keys is not None else None
    sindex = None if settlement_index is None else np.asarray(settlement_index).reshape(-1)

    if int(finite.sum()) == 0:
        center = [-1.286, 36.817]
    else:
        center = [float(np.mean(lat[finite])), float(np.mean(lon[finite]))]
    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    for i in np.flatnonzero(finite):
        name = str(labels[i])
        color = ACTION_COLORS.get(name, "#7f8c8d")
        lines = [f"<b>{name.replace('_', ' ')}</b>"]
        if keys is not None and sindex is not None and 0 <= int(sindex[i]) < len(keys):
            lines.append(keys[int(sindex[i])])
        if flood is not None and np.isfinite(flood[i]):
            lines.append(f"flood probability {float(flood[i]):.2f}")
        if deprivation is not None and int(deprivation[i]) >= 0:
            class_name = config.DEPRIVATION_CLASS_NAMES[int(deprivation[i])]
            lines.append(f"deprivation {class_name}")
        folium.CircleMarker(
            location=[float(lat[i]), float(lon[i])],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=1,
            popup=folium.Popup("<br>".join(lines), max_width=240),
        ).add_to(fmap)
    return fmap
