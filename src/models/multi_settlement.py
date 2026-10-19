"""
Sprint 4 helpers: combine Kibera + Mathare + Mukuru graphs/tiles, split
without cross-settlement leakage, and aggregate per-settlement + pooled
metrics for both model heads.

Real Mathare/Mukuru rasters and OSM graphs live on Drive / Overpass and are
only available in Colab. Everything here is pure logic over in-memory
manifests and PyG ``Data`` objects so it can be unit-tested offline with
synthetic inputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

import numpy as np
import torch
from torch_geometric.data import Data

from .. import config
from .graphsage_metrics import (
    classification_report_from_logits,
    macro_f1_from_cm,
    stratified_node_split,
)
from .metrics import f1_from_counts, iou_from_counts

logger = logging.getLogger(__name__)

ManifestPath = Union[str, Path]
ManifestEntries = Sequence[Mapping]


# ---------------------------------------------------------------------------
# GraphSAGE: load / combine / split / evaluate
# ---------------------------------------------------------------------------


def resolve_settlement_keys(settlement_keys: Iterable[str] = None) -> list:
    """Return the requested keys (default: all three) after validating them."""
    keys = list(settlement_keys) if settlement_keys is not None else list(config.SETTLEMENT_KEYS)
    unknown = [k for k in keys if k not in config.SETTLEMENTS]
    if unknown:
        raise ValueError(
            f"Unknown settlement key(s) {unknown}. Options: {list(config.SETTLEMENTS)}"
        )
    if not keys:
        raise ValueError("At least one settlement_key is required.")
    return keys


def load_settlement_graphs(
    graphs_dir: ManifestPath,
    settlement_keys: Iterable[str] = None,
) -> dict:
    """
    Load ``{settlement}_graph.pt`` files written by ``DataPipeline.acquire_graph_data``.

    Missing files are skipped with a warning rather than aborting the batch --
    Colab runs often have Kibera on disk before Mathare/Mukuru Overpass finishes.
    """
    keys = resolve_settlement_keys(settlement_keys)
    root = Path(graphs_dir)
    loaded = {}
    for key in keys:
        path = root / f"{key}_graph.pt"
        if not path.exists():
            logger.warning("Graph not found for %s at %s -- skipping.", key, path)
            continue
        # weights_only=False: PyG Data is not a pure tensor state_dict.
        # Fall back for older torch builds that lack the keyword.
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        if not isinstance(data, Data):
            raise TypeError(f"{path} did not contain a torch_geometric.data.Data object.")
        if data.y is None:
            raise ValueError(
                f"{path} has no data.y deprivation labels. Rebuild the graph with "
                f"Sprint 3+ DataPipeline.acquire_graph_data (enrich_buildings_with_deprivation)."
            )
        loaded[key] = data
        logger.info(
            "Loaded %s graph: nodes=%d edges=%d",
            key, data.num_nodes, data.edge_index.size(1) if data.edge_index is not None else 0,
        )
    return loaded


def combine_settlement_graphs(graphs: Mapping[str, Data]) -> Data:
    """
    Disjoint-union of per-settlement PyG graphs into one training graph.

    Settlements do not share edges (they are geographically separate), so a
    block-diagonal adjacency is the right inductive bias: GraphSAGE still
    shares weights across settlements while neighbourhood aggregation stays
    local to each settlement's buildings.

    Adds:
      - ``settlement_id``: LongTensor of settlement index per node
      - ``settlement_keys``: list of keys in index order (also on ``.keys`` for
        notebook convenience)
      - ``node_offset``: dict key -> starting node index in the combined graph
    """
    if not graphs:
        raise ValueError("combine_settlement_graphs requires at least one graph.")

    # Preserve canonical order when possible so settlement_id indices are stable.
    ordered_keys = [k for k in config.SETTLEMENT_KEYS if k in graphs]
    ordered_keys += [k for k in graphs if k not in ordered_keys]

    x_parts, y_parts, edge_parts = [], [], []
    settlement_ids = []
    node_offset = {}
    offset = 0

    for sid, key in enumerate(ordered_keys):
        data = graphs[key]
        n = int(data.num_nodes)
        node_offset[key] = offset
        x_parts.append(data.x)
        if data.y is None:
            raise ValueError(f"Graph for '{key}' has no data.y labels.")
        y_parts.append(data.y.view(-1).long())
        settlement_ids.append(torch.full((n,), sid, dtype=torch.long))

        if data.edge_index is not None and data.edge_index.numel() > 0:
            edge_parts.append(data.edge_index + offset)
        offset += n

    combined = Data(
        x=torch.cat(x_parts, dim=0),
        edge_index=(
            torch.cat(edge_parts, dim=1)
            if edge_parts
            else torch.empty((2, 0), dtype=torch.long)
        ),
        y=torch.cat(y_parts, dim=0),
    )
    combined.settlement_id = torch.cat(settlement_ids, dim=0)
    combined.settlement_keys = list(ordered_keys)
    combined.node_offset = dict(node_offset)
    # Preserve feature-name metadata when every component agrees.
    feature_names = getattr(next(iter(graphs.values())), "feature_names", None)
    if feature_names is not None:
        combined.feature_names = list(feature_names)

    logger.info(
        "Combined %d settlements into %d nodes / %d edges (%s).",
        len(ordered_keys),
        combined.num_nodes,
        combined.edge_index.size(1),
        ", ".join(f"{k}:{graphs[k].num_nodes}" for k in ordered_keys),
    )
    return combined


def stratified_multi_settlement_split(
    labels,
    settlement_ids,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Stratified 60/20/20 split applied *within each settlement*, then unioned.

    Per-settlement stratification keeps a minority high-severity class from
    vanishing inside a small settlement when a global stratified draw would
    allocate almost all of its nodes to train. Masks remain mutually
    exclusive and cover every node.
    """
    y = np.asarray(labels, dtype=np.int64)
    sids = np.asarray(settlement_ids, dtype=np.int64)
    if y.shape != sids.shape:
        raise ValueError(
            f"labels shape {y.shape} must match settlement_ids shape {sids.shape}"
        )
    n = y.size
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for sid in np.unique(sids):
        idx = np.where(sids == sid)[0]
        local = stratified_node_split(
            y[idx],
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            # Offset the seed per settlement so folds aren't identical permutations.
            seed=seed + int(sid),
        )
        train_mask[idx[local["train"]]] = True
        val_mask[idx[local["val"]]] = True
        test_mask[idx[local["test"]]] = True

    assert not np.any(train_mask & val_mask)
    assert not np.any(train_mask & test_mask)
    assert not np.any(val_mask & test_mask)
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == n

    return {"train": train_mask, "val": val_mask, "test": test_mask}


def prepare_multi_settlement_data(
    graphs: Mapping[str, Data],
    seed: int = 42,
) -> Data:
    """Combine graphs and attach per-settlement stratified train/val/test masks."""
    combined = combine_settlement_graphs(graphs)
    masks = stratified_multi_settlement_split(
        combined.y.cpu().numpy(),
        combined.settlement_id.cpu().numpy(),
        seed=seed,
    )
    combined.train_mask = torch.tensor(masks["train"], dtype=torch.bool)
    combined.val_mask = torch.tensor(masks["val"], dtype=torch.bool)
    combined.test_mask = torch.tensor(masks["test"], dtype=torch.bool)
    return combined


@torch.no_grad()
def evaluate_graphsage_by_settlement(
    model,
    data: Data,
    split: str = "val",
    num_classes: int = None,
) -> dict:
    """
    Macro-F1 (and full classification report) per settlement plus a pooled
    report from the summed confusion matrix. ``model`` must expose
    ``.network`` with the GraphSAGE forward signature used by GraphSAGEModel.
    """
    num_classes = num_classes or config.DEPRIVATION_NUM_CLASSES
    mask_attr = f"{split}_mask"
    if not hasattr(data, mask_attr):
        raise ValueError(f"data has no {mask_attr}; call prepare_multi_settlement_data / fit first.")
    if not hasattr(data, "settlement_id") or not hasattr(data, "settlement_keys"):
        raise ValueError(
            "data is missing settlement_id / settlement_keys -- combine graphs "
            "with combine_settlement_graphs / prepare_multi_settlement_data."
        )

    device = next(model.network.parameters()).device
    data = data.to(device)
    split_mask = getattr(data, mask_attr)
    model.network.eval()
    logits = model.network(data.x, data.edge_index)

    per_settlement = {}
    cms = []
    for sid, key in enumerate(data.settlement_keys):
        node_mask = (data.settlement_id == sid) & split_mask
        if int(node_mask.sum()) == 0:
            logger.warning("No %s nodes for settlement %s -- skipping.", split, key)
            continue
        report = classification_report_from_logits(
            logits[node_mask], data.y[node_mask], num_classes=num_classes,
        )
        per_settlement[key] = report
        cms.append(np.asarray(report["confusion_matrix"], dtype=np.int64))

    if not cms:
        raise ValueError(f"No nodes under {split}_mask for any settlement.")

    pooled_cm = sum(cms)
    pooled = {
        "macro_f1": macro_f1_from_cm(pooled_cm),
        "accuracy": float(np.diag(pooled_cm).sum() / max(pooled_cm.sum(), 1)),
        "confusion_matrix": pooled_cm.tolist(),
        "support": int(pooled_cm.sum()),
    }
    return {
        "split": split,
        "per_settlement": per_settlement,
        "pooled": pooled,
    }


def graphsage_threshold_report(
    per_settlement_macro_f1: Mapping[str, float],
    pooled_macro_f1: float,
    target: float = None,
) -> dict:
    """
    Compare pooled + per-settlement macro-F1 against the Sprint 4 / NFR bar
    (``config.GRAPHSAGE_FINAL_MACRO_F1_TARGET``, default 0.70).
    """
    target = config.GRAPHSAGE_FINAL_MACRO_F1_TARGET if target is None else float(target)
    per = {k: float(v) for k, v in per_settlement_macro_f1.items()}
    return {
        "target": target,
        "pooled_macro_f1": float(pooled_macro_f1),
        "meets_pooled_target": float(pooled_macro_f1) >= target,
        "per_settlement_macro_f1": per,
        "meets_per_settlement_target": {k: v >= target for k, v in per.items()},
        "all_settlements_meet_target": all(v >= target for v in per.values()) if per else False,
    }


# ---------------------------------------------------------------------------
# Attention U-Net: manifests / spatial holdout / metrics
# ---------------------------------------------------------------------------


def load_tile_manifest(
    tiles_root: ManifestPath,
    settlement_key: str,
    split: str = "train",
) -> list:
    """
    Load ``data/tiles/<settlement>/manifest_<split>.json`` and stamp each
    entry with ``settlement_key`` so pooled training can attribute metrics.
    """
    if settlement_key not in config.SETTLEMENTS:
        raise ValueError(
            f"Unknown settlement '{settlement_key}'. Options: {list(config.SETTLEMENTS)}"
        )
    path = Path(tiles_root) / settlement_key / f"manifest_{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {path}. Run Sprint 1 tiling for {settlement_key} "
            f"in Colab (GEE export → download → DataPipeline.tile_raster_stack) first."
        )
    with open(path) as f:
        manifest = json.load(f)
    tagged = []
    for entry in manifest:
        row = dict(entry)
        row["settlement_key"] = settlement_key
        tagged.append(row)
    return tagged


def load_multi_settlement_manifests(
    tiles_root: ManifestPath,
    settlement_keys: Iterable[str] = None,
    split: str = "train",
    require_all: bool = False,
) -> dict:
    """
    Load manifests for each settlement. Missing manifests are skipped unless
    ``require_all=True`` (useful once Drive has all three tiled stacks).
    """
    keys = resolve_settlement_keys(settlement_keys)
    out = {}
    missing = []
    for key in keys:
        try:
            entries = load_tile_manifest(tiles_root, key, split=split)
        except FileNotFoundError as exc:
            missing.append(key)
            logger.warning("%s", exc)
            continue
        # An empty manifest (0 tiles kept) means the stack was absent/partial
        # or every tile fell below the valid-data threshold at tiling time --
        # usually because the GEE export had not reached COMPLETED yet. Treat
        # it as missing rather than a usable settlement so downstream splits
        # don't crash on a settlement with no spatial locations.
        if not entries:
            missing.append(key)
            logger.warning(
                "Manifest for '%s' (%s split) has 0 tiles -- skipping. Confirm the "
                "GEE export reached COMPLETED on Drive, then re-run tile_raster_stack "
                "for this settlement.",
                key, split,
            )
            continue
        out[key] = entries
    if require_all and missing:
        raise FileNotFoundError(
            f"Missing or empty tile manifests for: {missing}. Present: {list(out)}"
        )
    return out


def _locations_from_entries(entries: ManifestEntries) -> list:
    return sorted(set((e["row_off"], e["col_off"]) for e in entries))


def multi_settlement_spatial_holdout(
    manifests_by_settlement: Mapping[str, Union[ManifestEntries, ManifestPath]],
    holdout_fraction: float = 0.25,
    seed: int = 42,
    min_train_locations: int = 1,
) -> dict:
    """
    Leakage-aware spatial holdout across settlements.

    For each settlement independently, hold out a fraction of distinct
    ``(row_off, col_off)`` locations for validation. Augmented variants of
    the same location stay on the same side of the split (same rule as
    Sprint 2's ``spatial_holdout_split``). Locations are settlement-scoped:
    Kibera ``(0, 0)`` and Mathare ``(0, 0)`` are different places and may
    land on opposite sides of the split without leaking.

    Returns train/val entry lists (pooled), per-settlement holdout locations,
    and location counts. Does **not** cross settlement boundaries when
    deciding holdouts -- that is intentional; a global location sort would
    be meaningless across different rasters.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1).")

    rng = np.random.default_rng(seed)
    train_entries, val_entries = [], []
    holdout_locations = {}
    location_counts = {}
    skipped_settlements = []

    for key, source in manifests_by_settlement.items():
        if isinstance(source, (str, Path)):
            with open(source) as f:
                entries = json.load(f)
            entries = [dict(e, settlement_key=key) for e in entries]
        else:
            entries = [dict(e) for e in source]
            for e in entries:
                e.setdefault("settlement_key", key)

        locations = _locations_from_entries(entries)
        n_loc = len(locations)
        if n_loc < 2:
            # Not enough distinct spatial locations to hold one out without
            # leaving an empty train side. Skip this settlement (with a
            # warning) rather than aborting the whole multi-settlement split,
            # so a half-tiled run can still train on the settlements that are
            # ready. The caller's "need >= 2 settlements" guard still applies.
            logger.warning(
                "Settlement '%s' has only %d distinct spatial location(s) -- "
                "skipping it in the spatial holdout (need >= 2). Finish tiling "
                "this settlement to include it.",
                key, n_loc,
            )
            skipped_settlements.append(key)
            continue

        n_holdout = max(1, int(round(n_loc * holdout_fraction)))
        n_holdout = min(n_holdout, n_loc - min_train_locations)
        if n_holdout < 1:
            raise ValueError(
                f"Settlement '{key}' has {n_loc} locations; cannot hold out any "
                f"while keeping min_train_locations={min_train_locations}."
            )

        order = list(locations)
        rng.shuffle(order)
        held = set(order[:n_holdout])
        holdout_locations[key] = sorted(held)
        location_counts[key] = {
            "n_locations": n_loc,
            "n_holdout": len(held),
            "n_train": n_loc - len(held),
        }

        for e in entries:
            loc = (e["row_off"], e["col_off"])
            if loc in held:
                val_entries.append(e)
            else:
                train_entries.append(e)

    usable = [k for k in manifests_by_settlement if k not in skipped_settlements]
    if len(usable) < 2:
        raise ValueError(
            f"Spatial holdout needs >= 2 settlements with >= 2 tile locations each; "
            f"usable={usable}, skipped={skipped_settlements}. Finish Sprint 1 tiling "
            f"for the missing settlements (GEE export must be COMPLETED before "
            f"tile_raster_stack)."
        )

    return {
        "train": train_entries,
        "val": val_entries,
        "holdout_locations": holdout_locations,
        "location_counts": location_counts,
        "num_settlements": len(usable),
        "skipped_settlements": skipped_settlements,
    }


def group_entries_by_settlement(entries: ManifestEntries) -> dict:
    """Partition manifest entries by ``settlement_key`` (required on each row)."""
    grouped = {}
    for e in entries:
        key = e.get("settlement_key")
        if key is None:
            raise ValueError(
                "Manifest entry missing settlement_key -- load via "
                "load_tile_manifest / multi_settlement_spatial_holdout."
            )
        grouped.setdefault(key, []).append(e)
    return grouped


def aggregate_flood_counts(counts_by_settlement: Mapping[str, Mapping[str, float]]) -> dict:
    """
    Sum TP/FP/FN/TN across settlements and report per-settlement + pooled
    IoU/F1. Pooling by raw counts (not by averaging IoUs) matches
    ``RunningMetrics`` and avoids bias when settlements have different
    flooded-pixel mass.
    """
    per = {}
    totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    for key, counts in counts_by_settlement.items():
        c = {k: float(counts[k]) for k in ("tp", "fp", "fn", "tn")}
        per[key] = {
            "counts": c,
            "iou": iou_from_counts(c),
            "f1": f1_from_counts(c),
        }
        for k in totals:
            totals[k] += c[k]

    return {
        "per_settlement": per,
        "pooled": {
            "counts": totals,
            "iou": iou_from_counts(totals),
            "f1": f1_from_counts(totals),
        },
    }


def flood_threshold_report(
    per_settlement: Mapping[str, Mapping[str, float]],
    pooled_iou: float,
    pooled_f1: float,
    iou_target: float = None,
    f1_target: float = None,
) -> dict:
    """
    Compare pooled + per-settlement flood metrics against Sprint 2 / NFR bars.
    ``per_settlement`` maps settlement_key -> {"iou": ..., "f1": ...}.
    """
    iou_target = (
        config.ATTENTION_UNET_TRANSFER_IOU_TARGET if iou_target is None else float(iou_target)
    )
    f1_target = (
        config.ATTENTION_UNET_FINAL_F1_TARGET if f1_target is None else float(f1_target)
    )
    per_iou = {k: float(v["iou"]) for k, v in per_settlement.items()}
    per_f1 = {k: float(v["f1"]) for k, v in per_settlement.items()}
    return {
        "iou_target": iou_target,
        "f1_target": f1_target,
        "pooled_iou": float(pooled_iou),
        "pooled_f1": float(pooled_f1),
        "meets_pooled_iou_target": float(pooled_iou) >= iou_target,
        "meets_pooled_f1_target": float(pooled_f1) >= f1_target,
        "per_settlement_iou": per_iou,
        "per_settlement_f1": per_f1,
        "meets_per_settlement_iou": {k: v >= iou_target for k, v in per_iou.items()},
        "meets_per_settlement_f1": {k: v >= f1_target for k, v in per_f1.items()},
    }


def _prefer_orig_entries(entries: ManifestEntries) -> list:
    """
    Keep only ``*_orig`` tiles when a location also has augmented variants,
    so flips/rots do not double-count the same ground pixels in tallies.
    Locations that only have augmented tiles are kept as-is.
    """
    by_loc = {}
    for e in entries:
        loc = (e["row_off"], e["col_off"], e.get("settlement_key"))
        by_loc.setdefault(loc, []).append(e)

    selected = []
    for group in by_loc.values():
        orig = [e for e in group if str(e.get("tile_id", "")).endswith("_orig")]
        selected.extend(orig if orig else group)
    return selected


def evaluate_flood_entries_by_settlement(
    model,
    entries: ManifestEntries,
    threshold: float = 0.5,
) -> dict:
    """
    Run the U-Net on a list of manifest entries and aggregate confusion
    counts per settlement. Uses whole-tile evaluation (no overlap mask) --
    for leakage-aware scores on a single held-out location, keep using
    ``evaluation.evaluate_masked``; this helper is for the multi-settlement
    summary table in notebook 04.
    """
    from .dataset import DEFAULT_LAYOUT

    layout = getattr(model, "layout", DEFAULT_LAYOUT)
    selected = _prefer_orig_entries(entries)
    grouped = group_entries_by_settlement(selected)
    counts_by_settlement = {}

    model.network.eval()
    device = next(model.network.parameters()).device

    for key, group in grouped.items():
        totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
        for entry in group:
            tile = np.load(entry["path"])
            inputs = np.nan_to_num(tile[layout.input_indices], nan=0.0)
            label = np.nan_to_num(tile[layout.label_index], nan=0.0)
            with torch.no_grad():
                tensor = torch.from_numpy(inputs).float().unsqueeze(0).to(device)
                proba = model.predict_proba(tensor)[0, 0].numpy()
            pred = proba >= threshold
            truth = label >= 0.5
            totals["tp"] += float(np.sum(pred & truth))
            totals["fp"] += float(np.sum(pred & ~truth))
            totals["fn"] += float(np.sum(~pred & truth))
            totals["tn"] += float(np.sum(~pred & ~truth))
        counts_by_settlement[key] = totals

    return aggregate_flood_counts(counts_by_settlement)
