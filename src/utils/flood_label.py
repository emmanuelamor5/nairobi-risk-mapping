"""
Event-based flood label construction, implemented in plain numpy.

This replaces Sprint 2 v1's dry-vs-rainy-season Otsu label. That earlier
label thresholded a single rainy-season VV *minimum* composite, which cannot
separate "new water from this flood" from "this pixel is just darker in the
rainy season" (bare soil wetting, vegetation phenology, a permanently damp
river corridor). The v1 diagnostics bore this out: the signal lived in the
SWIR/NIR optical bands (land cover), flooded fractions hit 19-32%, and the
U-Net never beat a flood-everywhere baseline -- i.e. it learned land cover,
not inundation.

Bi-temporal change detection fixes the root cause. Open water is specular at
C-band, so genuinely new inundation shows up as a sharp *drop* in VV
backscatter between a pre-event composite and an event-peak composite. Any
surface that is stable across the two dates (roofs, roads, standing
vegetation, permanent water) changes little and is therefore NOT labelled
flood -- which is exactly the land-cover confound we need to remove. This is
the UN-SPIDER recommended practice for SAR flood mapping.

The logic lives here as pure-numpy functions (not Earth Engine Array ops) for
the same reason Sprint 1 moved Otsu into numpy: it is unit-testable offline,
easy to inspect pixel by pixel, and free of the fragile GEE histogram-sort
bugs that cost real debugging time. ``src/utils/gee_utils.py`` mirrors this
exact algorithm server-side for the one-shot export path, and the notebook
can recompute labels here from exported raw bands to tune thresholds without
re-hitting GEE.

Inputs are backscatter arrays in **decibels** (COPERNICUS/S1_GRD is already
log-scaled), shaped (H, W). ``nan`` marks no-data and propagates to a
no-data (0) label.
"""

import numpy as np
from scipy import ndimage

from .. import config


def otsu_threshold(values: np.ndarray, num_bins: int = 256) -> float:
    """
    Otsu's between-class-variance threshold over a 1-D array of finite values.

    This is the same standard formulation Sprint 1 settled on after the
    GEE-native version proved fragile: build a histogram, then pick the bin
    edge that maximises between-class variance. Returns the threshold in the
    input's own units (dB here). Raises on empty input rather than returning
    a silently meaningless number.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("otsu_threshold received no finite values to threshold.")

    counts, bin_edges = np.histogram(finite, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total == 0:
        raise ValueError("otsu_threshold histogram is empty.")

    sum_all = (counts * bin_centers).sum()
    best_bss, best_threshold = -1.0, float(bin_centers[0])
    cum_count, cum_sum = 0.0, 0.0
    for i in range(len(counts) - 1):  # always leave >=1 bin in the "above" class
        cum_count += counts[i]
        cum_sum += counts[i] * bin_centers[i]
        above_count = total - cum_count
        if cum_count == 0 or above_count == 0:
            continue
        below_mean = cum_sum / cum_count
        above_mean = (sum_all - cum_sum) / above_count
        bss = cum_count * above_count * (below_mean - above_mean) ** 2
        if bss > best_bss:
            best_bss = bss
            best_threshold = float(bin_edges[i + 1])
    return best_threshold


def change_image(pre_vv: np.ndarray, event_vv: np.ndarray) -> np.ndarray:
    """
    Event-minus-pre VV difference in dB. Negative where backscatter dropped
    (candidate new water); nan wherever either date is missing.
    """
    pre = np.asarray(pre_vv, dtype=np.float64)
    event = np.asarray(event_vv, dtype=np.float64)
    if pre.shape != event.shape:
        raise ValueError(
            f"pre_vv {pre.shape} and event_vv {event.shape} must have the same shape."
        )
    return event - pre


def flood_mask_from_change(change: np.ndarray, threshold_db: float) -> np.ndarray:
    """
    Boolean flood candidate mask: True where the backscatter drop is at least
    as large as ``threshold_db`` (a negative number). No-data pixels (nan in
    the change image) are False, never flood.
    """
    mask = np.zeros(change.shape, dtype=bool)
    finite = np.isfinite(change)
    mask[finite] = change[finite] <= threshold_db
    return mask


def apply_slope_mask(mask: np.ndarray, slope_deg: np.ndarray,
                     max_slope_deg: float) -> np.ndarray:
    """
    Zero out candidate flood pixels on terrain steeper than ``max_slope_deg``.
    Standing water does not accumulate on steep slopes, so a backscatter drop
    there is almost always radar shadow/layover or a land-cover change, not
    inundation. Pixels with nan slope are left as-is (slope unknown).
    """
    slope = np.asarray(slope_deg, dtype=np.float64)
    steep = np.isfinite(slope) & (slope > max_slope_deg)
    return mask & ~steep


def apply_permanent_water_mask(mask: np.ndarray,
                               permanent_water: np.ndarray) -> np.ndarray:
    """
    Remove permanent open water (rivers, ponds that are always wet) from the
    flood mask. ``permanent_water`` is a truthy/0-1 array; flagged pixels are
    excluded so the label captures *new* inundation, not the baseline river.
    """
    perm = np.asarray(permanent_water) > 0
    return mask & ~perm


def remove_small_objects(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """
    Drop connected flood blobs smaller than ``min_pixels`` (8-connectivity).
    Single-pixel SAR speckle survives thresholding readily; real inundation
    forms contiguous patches, so a small-object filter removes salt-and-pepper
    false positives without eroding genuine flood extent.
    """
    if min_pixels <= 1 or not mask.any():
        return mask
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labelled, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(labelled), labelled, index=np.arange(1, n + 1))
    keep_labels = {i + 1 for i, s in enumerate(sizes) if s >= min_pixels}
    return np.isin(labelled, list(keep_labels))


def build_event_flood_label(pre_vv: np.ndarray, event_vv: np.ndarray,
                            slope_deg: np.ndarray = None,
                            permanent_water: np.ndarray = None,
                            threshold_db: float = None,
                            use_otsu: bool = True,
                            max_slope_deg: float = None,
                            min_object_pixels: int = None) -> tuple:
    """
    Full event flood label pipeline, returning ``(label, report)``.

    ``label`` is a float32 (H, W) array of 0/1 matching the ``flood_label``
    band the rest of the pipeline expects (``dataset.FLOOD_LABEL_BAND_INDEX``).
    ``report`` records the decisions made so the notebook can show exactly how
    the label was built and how aggressive each refinement step was.

    Thresholding: with ``use_otsu`` (default) the decision threshold is Otsu
    on the negative tail of the change image -- the part where new water
    lives -- and falls back to ``threshold_db`` (or config) if that tail is
    degenerate. Pass ``use_otsu=False`` to force the fixed ``threshold_db``.

    Refinement order (each only ever REMOVES flood pixels): slope mask ->
    permanent-water mask -> small-object filter. The order is immaterial since
    they are all intersections, but it is fixed for reproducible reporting.
    """
    if max_slope_deg is None:
        max_slope_deg = config.FLOOD_MAX_SLOPE_DEG
    if min_object_pixels is None:
        min_object_pixels = config.FLOOD_MIN_OBJECT_PIXELS
    if threshold_db is None:
        threshold_db = config.FLOOD_CHANGE_THRESHOLD_DB

    change = change_image(pre_vv, event_vv)
    finite = np.isfinite(change)

    used_threshold = float(threshold_db)
    threshold_source = "fixed"
    if use_otsu:
        # Otsu over the drop side only. Including the (much larger) "no change
        # / brightening" population would drag the threshold toward 0 and
        # flood half the scene -- the same failure mode the whole-scene Otsu
        # had in v1. Restricting to change < 0 isolates the new-water tail.
        drops = change[finite & (change < 0)]
        if drops.size >= 2 and np.unique(drops).size >= 2:
            otsu = otsu_threshold(drops)
            # Guard against a threshold so shallow it would flood everything,
            # or deeper than the data supports.
            if np.isfinite(otsu) and otsu < 0:
                used_threshold = float(otsu)
                threshold_source = "otsu"

    mask = flood_mask_from_change(change, used_threshold)
    counts = {"after_threshold": int(mask.sum())}

    if slope_deg is not None:
        mask = apply_slope_mask(mask, slope_deg, max_slope_deg)
        counts["after_slope_mask"] = int(mask.sum())
    if permanent_water is not None:
        mask = apply_permanent_water_mask(mask, permanent_water)
        counts["after_permanent_water_mask"] = int(mask.sum())
    if min_object_pixels and min_object_pixels > 1:
        mask = remove_small_objects(mask, min_object_pixels)
        counts["after_small_object_filter"] = int(mask.sum())

    label = mask.astype(np.float32)
    valid_pixels = int(finite.sum())
    report = {
        "threshold_db": used_threshold,
        "threshold_source": threshold_source,
        "max_slope_deg": float(max_slope_deg),
        "min_object_pixels": int(min_object_pixels),
        "valid_pixels": valid_pixels,
        "total_pixels": int(change.size),
        "flooded_pixels": int(label.sum()),
        "flooded_fraction": float(label.sum() / valid_pixels) if valid_pixels else float("nan"),
        "pixel_counts_by_step": counts,
    }
    return label, report
