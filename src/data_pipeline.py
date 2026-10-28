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
    def __init__(self, settlement_key: str, gee_project_id: str = None, data_root: str = None,
                 boundary_path: str = None, building_source: str = "osm",
                 open_buildings_path: str = None, open_buildings_min_confidence: float = None):
        """
        ``boundary_path`` points at a digitized settlement polygon (GeoJSON,
        GeoPackage, shapefile). When omitted, the pipeline looks for
        ``data/boundaries/<settlement_key>.<ext>`` next to the data root
        before resorting to OSM geocoding and the config bbox -- so dropping
        a file in that folder fixes the boundary for every notebook at once.

        ``building_source`` is ``"osm"`` (Overpass footprints) or
        ``"open_buildings"`` (Google Open Buildings v3). The Open Buildings
        file is ``open_buildings_path``, or
        ``data/buildings/<key>_open_buildings.geojson`` when that exists.
        It is produced by ``export_open_buildings``; v3 has no roof tag, so
        deprivation scoring uses a neutral roof score for those nodes.
        """
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

        if boundary_path is None:
            boundary_path = graph_utils.default_boundary_path(
                settlement_key, data_parent=self.data_root.parent
            )
        self.boundary_path = Path(boundary_path) if boundary_path is not None else None
        if building_source not in config.BUILDING_SOURCES:
            raise ValueError(
                f"Unknown building_source {building_source!r}. Options: {config.BUILDING_SOURCES}"
            )
        self.building_source = building_source
        self.open_buildings_min_confidence = open_buildings_min_confidence
        if open_buildings_path is None and building_source == "open_buildings":
            open_buildings_path = graph_utils.default_open_buildings_path(
                settlement_key, data_parent=self.data_root.parent
            )
        self.open_buildings_path = Path(open_buildings_path) if open_buildings_path is not None else None
        self._boundary = None
        self._ee_initialized = False

    # ------------------------------------------------------------------ #
    # Boundary resolution
    # ------------------------------------------------------------------ #
    def get_boundary(self):
        """
        Resolve (once) and return the settlement boundary as a one-row WGS84
        GeoDataFrame. The polygon actually used is also written to
        ``data/raw/<settlement_key>_boundary_used.geojson`` so it can be
        checked against satellite imagery before any training run.
        """
        if self._boundary is None:
            logger.info("Resolving boundary for %s", self.settlement.name)
            self._boundary = graph_utils.fetch_settlement_boundary(
                self.settlement_key, boundary_path=self.boundary_path
            )
            source = self.boundary_source
            logger.info(
                "Boundary for %s from %s (%.2f km^2)%s",
                self.settlement.name, source,
                graph_utils.boundary_area_km2(self._boundary),
                f": {self.boundary_path}" if source == "file" else "",
            )
            if source != "file":
                logger.warning(
                    "Boundary for %s is the %s polygon, not a digitized settlement "
                    "boundary. Place one at %s/%s.geojson to override it.",
                    self.settlement.name, source, config.BOUNDARY_DIR, self.settlement_key,
                )
            self._write_boundary_used()
        return self._boundary

    @property
    def boundary_source(self) -> str:
        """``file`` / ``osm`` / ``bbox`` once the boundary has been resolved."""
        if self._boundary is None:
            return "unresolved"
        if "boundary_source" in self._boundary.columns:
            return str(self._boundary["boundary_source"].iloc[0])
        return "osm"

    @property
    def boundary_used_path(self) -> Path:
        return (
            self.data_root.parent / config.RAW_DIR
            / f"{self.settlement_key}_boundary_used.geojson"
        )

    def _write_boundary_used(self):
        try:
            out = self._boundary[["geometry"]].copy()
            out["name"] = self.settlement.name
            out["boundary_source"] = self.boundary_source
            out["boundary_path"] = str(self.boundary_path) if self.boundary_path else ""
            out.to_file(self.boundary_used_path, driver="GeoJSON")
        except Exception as exc:  # inspection aid only; never block the pipeline
            logger.warning("Could not write boundary snapshot: %s", exc)

    # ------------------------------------------------------------------ #
    # Raster branch (feeds the Attention U-Net)
    # ------------------------------------------------------------------ #
    def acquire_raster_stack(self, label_mode: str = "event", include_s1_inputs: bool = False):
        """
        Build the GEE image stack and export it to Drive. Returns the export task.

        ``label_mode`` ("event"/"seasonal") and ``include_s1_inputs`` are passed
        through to gee_utils.build_raster_stack -- Sprint 2 v2 defaults to the
        event change-detection label; set include_s1_inputs=True to add the
        pre/event VV bands for the EVENT_S1 layout.
        """
        import ee  # local import: keeps this module importable without earthengine-api
        # for the graph-only path used in unit tests.

        if not self._ee_initialized:
            # project id may be None here: initialize_gee falls back to
            # GEE_PROJECT_ID or the service-account key's own project_id.
            gee_utils.initialize_gee(self.gee_project_id)
            self._ee_initialized = True

        boundary = self.get_boundary()
        geometry = ee.Geometry(boundary.geometry.iloc[0].__geo_interface__)
        # Size-aware buffer: smaller settlements (Mathare, Mukuru) would
        # otherwise export rasters below the 256px tile size in a dimension, so
        # tile_raster_stack keeps 0 tiles. Buffer each settlement just enough
        # to fit >= RASTER_MIN_TILE_LOCATIONS tile windows in both dimensions
        # (needed for the Sprint 4 per-settlement spatial holdout).
        bounds_m = boundary.to_crs(config.CRS_METRIC).total_bounds
        buffer_m = gee_utils.required_export_buffer_m(bounds_m)
        logger.info(
            "Export buffer for %s: %.0fm (bounds extent %.0fm x %.0fm)",
            self.settlement.name, buffer_m,
            bounds_m[2] - bounds_m[0], bounds_m[3] - bounds_m[1],
        )
        geometry = geometry.buffer(buffer_m)

        logger.info(
            "Assembling raster stack for %s (label_mode=%s, include_s1_inputs=%s)",
            self.settlement.name, label_mode, include_s1_inputs,
        )
        stack = gee_utils.build_raster_stack(
            geometry, label_mode=label_mode, include_s1_inputs=include_s1_inputs
        )

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

    def tile_raster_stack(self, stack_path: str, split: str = "train",
                          band_names: list = None):
        """
        Tile an exported GeoTIFF stack into 256x256 patches with 50px overlap,
        discarding tiles below config.MIN_VALID_DATA_FRACTION valid pixels.
        Augmentation (flip/rotation) is applied only when split == "train",
        and only to the raster stream, per proposal Sec 3.2.2.

        ``band_names`` (optional) labels each band so Sentinel-1 VV bands
        (names ``vv`` / ``*_vv``) can be normalised with a FIXED dB window
        (config.S1_DB_RANGE) instead of per-tile percentiles. This matters
        only for the S1-input layout (EVENT_S1_LAYOUT): percentile-scaling pre
        and event VV independently would erase the cross-date backscatter drop
        that signals water. When ``band_names`` is None, all bands use the
        original per-band percentile normalisation (v1 behaviour, unchanged).
        """
        sar_band_indices = set()
        if band_names is not None:
            sar_band_indices = {
                i for i, n in enumerate(band_names) if n == "vv" or str(n).endswith("_vv")
            }
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

                    tile = self._normalise_bands(tile, sar_band_indices=sar_band_indices)
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
    def _normalise_bands(tile: np.ndarray, sar_band_indices: set = None) -> np.ndarray:
        """
        Per-band min-max scaling to [0,1], band-wise so SAR and optical scales
        don't collide. Bands in ``sar_band_indices`` instead use a fixed dB
        window (config.S1_DB_RANGE) shared across all SAR bands, preserving the
        pre-vs-event backscatter difference that the per-band percentile scaling
        would otherwise destroy.
        """
        sar_band_indices = sar_band_indices or set()
        out = np.zeros_like(tile, dtype=np.float32)
        s1_min, s1_max = config.S1_DB_RANGE
        for b in range(tile.shape[0]):
            band = tile[b].astype(np.float32)
            finite = band[np.isfinite(band)]
            if finite.size == 0:
                continue
            if b in sar_band_indices:
                out[b] = np.clip((band - s1_min) / (s1_max - s1_min), 0, 1)
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
    def export_open_buildings(self):
        """
        Start an Earth Engine export of Google Open Buildings v3 inside the
        settlement boundary. Returns the task. The GeoJSON appears in the
        Drive folder ``nairobi_risk_mapping`` as
        ``<settlement_key>_open_buildings.geojson``.
        """
        import ee

        if not self._ee_initialized:
            gee_utils.initialize_gee(self.gee_project_id)
            self._ee_initialized = True
        boundary = self.get_boundary()
        geometry = ee.Geometry(boundary.geometry.iloc[0].__geo_interface__)
        description = f"{self.settlement_key}_open_buildings"
        task = gee_utils.export_open_buildings_to_drive(
            geometry, description, min_confidence=self.open_buildings_min_confidence,
        )
        logger.info(
            "Open Buildings export '%s' started. When it completes, pass "
            "building_source='open_buildings' and the Drive GeoJSON as "
            "open_buildings_path.",
            description,
        )
        return task

    def load_buildings(self):
        """
        Footprints for the graph, clipped to the settlement boundary.

        OSM when ``building_source`` is ``"osm"``. A local Open Buildings
        file otherwise. Logs the nearest-neighbour spacing, which is fixed
        by the footprints: the 15 m edge threshold does not change it.
        """
        boundary = self.get_boundary()
        if self.building_source == "osm":
            logger.info("Fetching OSM buildings for %s", self.settlement.name)
            buildings = graph_utils.fetch_osm_buildings(boundary)
        else:
            path = self.open_buildings_path
            if path is None or not path.exists():
                raise FileNotFoundError(
                    "No Open Buildings file for "
                    f"{self.settlement_key}. Run export_open_buildings(), wait for "
                    f"Drive/nairobi_risk_mapping/{self.settlement_key}_open_buildings.geojson, "
                    "then construct DataPipeline(..., building_source='open_buildings', "
                    "open_buildings_path='<that file>')."
                )
            logger.info("Loading Open Buildings for %s from %s", self.settlement.name, path)
            import geopandas as gpd
            buildings = graph_utils.prepare_open_buildings(
                gpd.read_file(path),
                boundary,
                min_confidence=self.open_buildings_min_confidence,
            )
        spacing = graph_utils.spacing_summary(buildings)
        logger.info(
            "%s footprints for %s: n=%d median nearest neighbour %s m",
            self.building_source, self.settlement.name, spacing["n"],
            "n/a" if spacing["median_m"] is None else f"{spacing['median_m']:.1f}",
        )
        self.building_spacing = spacing
        return buildings

    def acquire_graph_data(self):
        boundary = self.get_boundary()

        buildings = self.load_buildings()
        logger.info("Fetched %d building footprints", len(buildings))

        logger.info("Fetching OSM infrastructure for %s", self.settlement.name)
        infra = graph_utils.fetch_osm_infrastructure(boundary)
        buildings = graph_utils.compute_infra_distance_features(buildings, infra)

        # Sprint 3: attach UN-Habitat/SUPI-proxy deprivation scores and the
        # three-class severity label GraphSAGE trains against.
        buildings = graph_utils.enrich_buildings_with_deprivation(buildings)

        graph = graph_utils.build_proximity_graph(buildings)
        integrity = graph_utils.check_graph_integrity(graph)
        integrity["building_source"] = self.building_source
        integrity["nearest_neighbour_m"] = getattr(self, "building_spacing", None)
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
