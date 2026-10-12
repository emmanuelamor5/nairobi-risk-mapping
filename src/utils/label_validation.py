"""
Validate a flood label against an independent spatial reference.

Sprint 2 v1 left one question open: is the ``flood_label`` band actually
flooding, or just a land-cover proxy? The honest answer requires checking it
against evidence that did NOT come from the same Sentinel-1 thresholding --
the proposal names the Nairobi Rivers Regeneration Programme (NRRP) 37
high-risk riparian zones for exactly this.

The numpy core here is reference-agnostic: given a label mask and a reference
mask on the same grid, it reports how strongly flooded pixels concentrate
inside the reference. The key statistic is **lift** = (fraction of flooded
pixels inside the reference) / (fraction of all valid pixels inside the
reference). Lift > 1 means the label is enriched in the high-risk zone --
what a real flood label should do. A land-cover proxy that paints vegetation
and dry ground scores lift near 1. Running the old seasonal label and the new
event label through the same function makes the improvement (or lack of it)
quantitative rather than visual.

The geo helpers (rasterising NRRP polygons, or building a river-corridor
reference from OSM waterways) need rasterio/geopandas and a raster transform;
they are only imported when called, so the numpy core stays dependency-light
and unit-testable offline.
"""

import numpy as np


def confusion_against_reference(label: np.ndarray, reference: np.ndarray,
                                valid_mask: np.ndarray = None) -> dict:
    """
    TP/FP/FN/TN of a flood ``label`` treated as a prediction of the
    ``reference`` mask, over ``valid_mask`` (defaults to all pixels). Both
    inputs are coerced to boolean (> 0.5 for the label, > 0 for the
    reference). This is a spatial-agreement count, not a claim that the
    reference is ground truth -- the reference is a risk prior, so interpret
    the numbers as concentration, not accuracy.
    """
    lab = np.asarray(label) > 0.5
    ref = np.asarray(reference) > 0
    if lab.shape != ref.shape:
        raise ValueError(f"label {lab.shape} and reference {ref.shape} must match.")
    if valid_mask is None:
        m = np.ones(lab.shape, dtype=bool)
    else:
        m = np.asarray(valid_mask) > 0
    return {
        "tp": float(np.sum(lab & ref & m)),
        "fp": float(np.sum(lab & ~ref & m)),
        "fn": float(np.sum(~lab & ref & m)),
        "tn": float(np.sum(~lab & ~ref & m)),
    }


def agreement_metrics(label: np.ndarray, reference: np.ndarray,
                      valid_mask: np.ndarray = None, eps: float = 1e-9) -> dict:
    """
    Concentration metrics of a flood ``label`` relative to a reference risk
    zone. Returns:

      - ``iou``: spatial overlap of label and reference.
      - ``precision``: share of flooded pixels that fall inside the reference.
      - ``recall``: share of reference pixels the label flags as flooded.
      - ``label_fraction`` / ``reference_fraction``: base rates over valid pixels.
      - ``lift``: precision / reference_fraction. The headline number -- how
        many times more concentrated the label is inside the reference than a
        spatially random label would be. Lift ~1 is the land-cover-proxy
        signature; lift >> 1 is what a real flood label should show.
    """
    counts = confusion_against_reference(label, reference, valid_mask)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    valid = tp + fp + fn + tn
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    reference_fraction = (tp + fn) / (valid + eps)
    label_fraction = (tp + fp) / (valid + eps)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "precision": precision,
        "recall": recall,
        "f1": (2 * tp) / (2 * tp + fp + fn + eps),
        "label_fraction": label_fraction,
        "reference_fraction": reference_fraction,
        "lift": precision / (reference_fraction + eps),
        "valid_pixels": int(valid),
    }


def compare_labels(labels: dict, reference: np.ndarray,
                   valid_mask: np.ndarray = None) -> dict:
    """
    Run agreement_metrics for several labels against one reference. ``labels``
    maps a name (e.g. "seasonal_v1", "event_v2") to its mask. Returns a dict
    keyed the same way -- convenient for the notebook's old-vs-new table.
    """
    return {name: agreement_metrics(mask, reference, valid_mask)
            for name, mask in labels.items()}


# --------------------------------------------------------------------------- #
# Geo helpers -- rasterising a vector reference onto the training grid.
# Imported lazily so the numpy core above stays offline-testable.
# --------------------------------------------------------------------------- #
def rasterize_reference(reference_gdf, transform, out_shape, raster_crs) -> np.ndarray:
    """
    Burn reference polygons (e.g. NRRP high-risk zones) onto the raster grid
    defined by ``transform``/``out_shape``/``raster_crs``, returning a 0/1
    uint8 mask aligned to the exported stack. ``reference_gdf`` is a GeoPandas
    GeoDataFrame in any CRS; it is reprojected to the raster CRS first.
    """
    from rasterio.features import rasterize

    gdf = reference_gdf.to_crs(raster_crs)
    shapes = ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty)
    mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask


def river_corridor_reference(waterways_gdf, transform, out_shape, raster_crs,
                             buffer_m: float = 100.0) -> np.ndarray:
    """
    Fallback reference when no NRRP polygon file is available: a buffer around
    OSM river/stream lines. Real flooding in these settlements concentrates
    along the Nairobi/Mathare/Ngong corridors, so a river buffer is a
    reasonable (if coarser) risk prior to check label concentration against.

    ``waterways_gdf`` are OSM waterway line features (``waterway=river|stream``);
    they are reprojected to a metric CRS, buffered by ``buffer_m``, then burned
    onto the raster grid.
    """
    from rasterio.features import rasterize

    metric = waterways_gdf.to_crs(3857)  # Web Mercator metres, adequate for a local buffer
    buffered = metric.buffer(buffer_m).to_crs(raster_crs)
    shapes = ((geom, 1) for geom in buffered if geom is not None and not geom.is_empty)
    mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask
