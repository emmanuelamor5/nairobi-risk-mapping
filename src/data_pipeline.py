"""
DataPipeline: the Sprint 1 deliverable (proposal Sec 3.4.3, weeks 1-2).

Responsibilities, matching the "Data Pipeline class" described in Sec 3.4.2:
  1. Query GEE for Sentinel-2, Sentinel-1, and SRTM/TWI layers, assembled
     into a single multi-band raster stack per settlement.
  2. Query OSM (via osmnx) for building footprints and infrastructure,
     constructing a proximity graph and serialising it as PyTorch Geometric
     Data objects.
  3. Preprocess the raster stack: normalise, tile into 256x256 patches with
     50px overlap, discard low-data patches, and augment the training split.

This class deliberately does NOT touch model code -- it hands off a
GeoTIFF stack + tile index + serialised graph, matching the late-fusion
architecture's strict input/output contract between the data layer and
both model heads (proposal Sec 3.6.1).
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from . import config
from .utils import gee_utils, graph_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataPipeline:
    def __init__(self, settlement_key: str, gee_project_id: str = None, data_root: str = None):
        if settlement_key not in config.SETTLEMENTS:
            raise ValueError(
                f"Unknown settlement '{settlement_key}'. Options: {list(config.SETTLEMENTS)}"
            )
        self.settlement_key = settlement_key
        self.settlement = config.SETTLEMENTS[settlement_key]
        self.gee_project_id = gee_project_id or config.GEE_PROJECT_ID
        self.data_root = Path(data_root or config.DATA_ROOT)

        for d in [config.RAW_DIR, config.STACKS_DIR, config.TILES_DIR, config.GRAPHS_DIR]:
            (self.data_root.parent / d).mkdir(parents=True, exist_ok=True)

        self._boundary = None
        self._ee_initialized = False

    # ------------------------------------------------------------------ #
    # Boundary resolution
    # ------------------------------------------------------------------ #
    def get_boundary(self):
        if self._boundary is None:
            logger.info("Resolving boundary for %s", self.settlement.name)
            self._boundary = graph_utils.fetch_settlement_boundary(self.settlement_key)
        return self._boundary

    # ------------------------------------------------------------------ #
    # Raster branch (feeds the Attention U-Net)
    # ------------------------------------------------------------------ #
    def acquire_raster_stack(self):
        """Build the GEE image stack and export it to Drive. Returns the export task."""
        import ee  # local import: keeps this module importable without earthengine-api
        # for the graph-only path used in unit tests.

        if not self._ee_initialized:
            # project id may be None here: initialize_gee falls back to
            # GEE_PROJECT_ID or the service-account key's own project_id.
            gee_utils.initialize_gee(self.gee_project_id)
            self._ee_initialized = True

        boundary = self.get_boundary()
        geometry = ee.Geometry(boundary.geometry.iloc[0].__geo_interface__)
        # Buffer by 200m -- some settlement boundaries are too narrow to clear
        # the 256px tile size in one dimension at 10m resolution (see Kibera:
        # raw export came back 246 x 313px, short in height).
        geometry = geometry.buffer(200)
        # Buffer by 200m -- some settlement boundaries are too narrow to clear
        # the 256px tile size in one dimension at 10m resolution (see Kibera:
        # raw export came back 246 x 313px, short in height).
        geometry = geometry.buffer(200)
        # Buffer by 200m -- some settlement boundaries are too narrow to clear
        # the 256px tile size in one dimension at 10m resolution (see Kibera:
        # raw export came back 246 x 313px, short in height).
        geometry = geometry.buffer(200)
        # Buffer by 200m -- some settlement boundaries are too narrow to clear
        # the 256px tile size in one dimension at 10m resolution (see Kibera:
        # raw export came back 246 x 313px, short in height).
        geometry = geometry.buffer(200)

        logger.info("Assembling raster stack for %s", self.settlement.name)
        stack = gee_utils.build_raster_stack(geometry)

        description = f"{self.settlement_key}_raster_stack"
        task = gee_utils.export_stack_to_drive(stack, geometry, description)
        logger.info(
            "Export task '%s' started -- poll task.status() or check the "
            "Drive folder 'nairobi_risk_mapping'. This runs asynchronously "
            "on Google's servers and continues even if the Colab runtime "
            "disconnects.",
            description,
        )
        return task

    def tile_raster_stack(self, stack_path: str, split: str = "train"):
        """
        Tile an exported GeoTIFF stack into 256x256 patches with 50px overlap,
        discarding tiles below config.MIN_VALID_DATA_FRACTION valid pixels.
        Augmentation (flip/rotation) is applied only when split == "train",
        and only to the raster stream, per proposal Sec 3.2.2.
        """
        tile_size = config.TILE_SIZE_PX
        overlap = config.TILE_OVERLAP_PX
        stride = tile_size - overlap

        out_dir = self.data_root.parent / config.TILES_DIR / self.settlement_key
        out_dir.mkdir(parents=True, exist_ok=True)

        kept, discarded = 0, 0
        manifest = []

        with rasterio.open(stack_path) as src:
            height, width = src.height, src.width
            band_count = src.count

            for row_off in range(0, height - tile_size + 1, stride):
                for col_off in range(0, width - tile_size + 1, stride):
                    window = Window(col_off, row_off, tile_size, tile_size)
                    tile = src.read(window=window)

                    valid_fraction = self._valid_data_fraction(tile)
                    if valid_fraction < config.MIN_VALID_DATA_FRACTION:
                        discarded += 1
                        continue

                    tile = self._normalise_bands(tile)
                    variants = self._augment(tile) if split == "train" else [("orig", tile)]

                    for suffix, variant in variants:
                        tile_id = f"{self.settlement_key}_{row_off}_{col_off}_{suffix}"
                        out_path = out_dir / f"{tile_id}.npy"
                        np.save(out_path, variant)
                        manifest.append({
                            "tile_id": tile_id,
                            "path": str(out_path),
                            "row_off": row_off,
                            "col_off": col_off,
                            "valid_fraction": float(valid_fraction),
                            "split": split,
                        })
                    kept += 1

        manifest_path = out_dir / f"manifest_{split}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "Tiling complete for %s (%s split): %d tiles kept, %d discarded "
            "(valid-data threshold %.0f%%). Manifest: %s",
            self.settlement.name, split, kept, discarded,
            config.MIN_VALID_DATA_FRACTION * 100, manifest_path,
        )
        return manifest_path

    @staticmethod
    def _valid_data_fraction(tile: np.ndarray) -> float:
        """Fraction of pixels that are finite and non-zero across all bands."""
        valid_mask = np.isfinite(tile).all(axis=0) & (tile != 0).any(axis=0)
        return float(valid_mask.mean())

    @staticmethod
    def _normalise_bands(tile: np.ndarray) -> np.ndarray:
        """Per-band min-max scaling to [0,1], band-wise so SAR and optical scales don't collide."""
        out = np.zeros_like(tile, dtype=np.float32)
        for b in range(tile.shape[0]):
            band = tile[b].astype(np.float32)
            finite = band[np.isfinite(band)]
            if finite.size == 0:
                continue
            b_min, b_max = np.percentile(finite, [1, 99])  # robust to outlier pixels
            if b_max > b_min:
                out[b] = np.clip((band - b_min) / (b_max - b_min), 0, 1)
        return out

    @staticmethod
    def _augment(tile: np.ndarray):
        """Flip/rotation augmentation, applied to the raster stream only (proposal Sec 3.2.2)."""
        return [
            ("orig", tile),
            ("flip_h", np.flip(tile, axis=2).copy()),
            ("flip_v", np.flip(tile, axis=1).copy()),
            ("rot90", np.rot90(tile, k=1, axes=(1, 2)).copy()),
        ]

    # ------------------------------------------------------------------ #
    # Graph branch (feeds GraphSAGE)
    # ------------------------------------------------------------------ #
    def acquire_graph_data(self):
        boundary = self.get_boundary()

        logger.info("Fetching OSM buildings for %s", self.settlement.name)
        buildings = graph_utils.fetch_osm_buildings(boundary)
        logger.info("Fetched %d building footprints", len(buildings))

        logger.info("Fetching OSM infrastructure for %s", self.settlement.name)
        infra = graph_utils.fetch_osm_infrastructure(boundary)
        buildings = graph_utils.compute_infra_distance_features(buildings, infra)

        graph = graph_utils.build_proximity_graph(buildings)
        integrity = graph_utils.check_graph_integrity(graph)
        logger.info("Graph integrity check for %s: %s", self.settlement.name, integrity)

        if integrity["largest_component_frac"] < 0.5:
            logger.warning(
                "Largest connected component covers only %.0f%% of nodes for %s -- "
                "this usually means the adjacency threshold (%sm) is too tight for "
                "this settlement's building density, or OSM coverage is sparse. "
                "Inspect before training GraphSAGE.",
                integrity["largest_component_frac"] * 100,
                self.settlement.name,
                config.BUILDING_ADJACENCY_THRESHOLD_M,
            )

        buildings, graph, pruning_report = graph_utils.prune_small_components(buildings, graph)
        logger.info("Pruned small components for %s: %s", self.settlement.name, pruning_report)

        buildings, graph, pruning_report = graph_utils.prune_small_components(buildings, graph)
        logger.info("Pruned small components for %s: %s", self.settlement.name, pruning_report)

        data = graph_utils.graph_to_pyg_data(graph)

        out_dir = self.data_root.parent / config.GRAPHS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.settlement_key}_graph.pt"

        import torch
        torch.save(data, out_path)

        integrity_path = out_dir / f"{self.settlement_key}_graph_integrity.json"
        with open(integrity_path, "w") as f:
            json.dump(integrity, f, indent=2)

        logger.info("Saved graph data to %s", out_path)
        return data, integrity

    # ------------------------------------------------------------------ #
    # Sprint 1 entrypoint
    # ------------------------------------------------------------------ #
    def run_sprint1(self, include_raster_export: bool = True):
        """
        Sprint 1 deliverable (proposal Sec 3.4.3): confirm all four data
        sources can be acquired, preprocessed, and stored for this
        settlement -- a validated multi-band GeoTIFF stack + a serialised
        PyTorch Geometric graph.

        Raster export runs asynchronously against Drive (GEE exports can't
        be awaited synchronously in-notebook); call tile_raster_stack()
        separately once the export finishes and you've downloaded/located
        the GeoTIFF.
        """
        results = {}
        if include_raster_export:
            results["raster_export_task"] = self.acquire_raster_stack()
        results["graph_data"], results["graph_integrity"] = self.acquire_graph_data()
        return results


def run_sprint1_all_settlements(gee_project_id: str, include_raster_export: bool = True):
    """
    Convenience runner for all three settlements defined in config.SETTLEMENTS.
    Proposal Sec 3.4.3 frames Sprint 1 as validating that the pipeline works
    across Kibera, Mathare, and Mukuru, with Kibera as the concrete MVP
    deliverable -- this runs the same validated pipeline against all three so
    you have evidence it generalises before Sprint 4 (full three-settlement
    training) depends on it.

    Each settlement's raster export runs as an independent async GEE task
    (kicked off back-to-back, not awaited), so they progress in parallel on
    Google's servers rather than queuing behind each other in the notebook.
    Graph acquisition runs synchronously in sequence since it's local/OSM-bound
    rather than GEE-quota-bound.

    Returns a dict keyed by settlement_key, each value the same structure
    DataPipeline.run_sprint1() returns. A per-settlement exception is caught
    and logged rather than aborting the whole batch, since a transient OSM
    Overpass timeout on one settlement shouldn't block the other two.
    """
    summary = {}
    for settlement_key in config.SETTLEMENTS:
        logger.info("=" * 60)
        logger.info("Running Sprint 1 pipeline for %s", config.SETTLEMENTS[settlement_key].name)
        logger.info("=" * 60)
        try:
            pipeline = DataPipeline(settlement_key=settlement_key, gee_project_id=gee_project_id)
            summary[settlement_key] = pipeline.run_sprint1(include_raster_export=include_raster_export)
            summary[settlement_key]["status"] = "ok"
        except Exception as exc:
            logger.error("Sprint 1 failed for %s: %s", settlement_key, exc)
            summary[settlement_key] = {"status": "failed", "error": str(exc)}

    ok = [k for k, v in summary.items() if v.get("status") == "ok"]
    failed = [k for k, v in summary.items() if v.get("status") == "failed"]
    logger.info("Sprint 1 batch complete: %d/%d settlements succeeded (%s). Failed: %s",
                len(ok), len(config.SETTLEMENTS), ok, failed or "none")
    return summary
