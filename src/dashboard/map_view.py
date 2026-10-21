"""
Folium map of parcel policy actions.

No wireframe file is in the repository, the project Drive folder, or the
signed-in Figma account. The layout follows the planner view described in
the proposal: a settlement filter and action legend beside a parcel map,
with the flood probability, deprivation class, and action on each parcel.
The page chrome (sign-in, MFA, session banner) lives in the dashboard
shell. This module only builds the map, and it is only returned after the
auth gate.

Coordinates may be geographic (EPSG:4326) or projected. Projected
positions are transformed with pyproj before Folium sees them.
"""

from __future__ import annotations

import numpy as np

from .. import config
from ..models.fusion import POLICY_ACTIONS, action_labels

# Colour is the action. Resettlement is the darkest so it stays distinct
# from in-place flood work on a satellite-free street basemap.
ACTION_COLORS = {
    "retain_formalise": "#1b4f72",
    "retain_upgrade": "#148f77",
    "in_situ_upgrade_priority": "#b9770e",
    "flood_resilient_upgrade": "#c0392b",
    "compensated_resettlement": "#4a235a",
    "insufficient_evidence": "#7f8c8d",
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
    crs = None
    for key in keys:
        if key not in graphs:
            raise ValueError(f"No graph loaded for '{key}'.")
        pos = graphs[key].pos
        if pos is None:
            raise ValueError(f"Graph '{key}' has no data.pos.")
        arr = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
        parts.append(np.asarray(arr, dtype=np.float64))
        crs = crs or getattr(graphs[key], "pos_crs", None)
    positions = np.concatenate(parts, axis=0)
    n_actions = int(np.asarray(table["action_index"]).shape[0])
    if positions.shape[0] != n_actions:
        raise ValueError(
            f"Joined positions {positions.shape[0]} != policy rows {n_actions}."
        )
    joined = dict(table)
    joined["positions"] = positions
    joined["pos_crs"] = crs or "EPSG:32737"
    return joined


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
