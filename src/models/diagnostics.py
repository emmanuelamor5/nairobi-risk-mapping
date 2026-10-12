"""
Label-quality diagnostics, reusable so the notebook can re-run the exact
checks that exposed the v1 problem and compare the numbers against the new
event label.

Two checks, both from the v1 post-mortem:

  1. Cross-location pixel AUC. Fit a plain logistic-regression pixel
     classifier on one location's bands and score it on the held-out
     location. This measures how much *transferable* per-pixel signal the
     label carries, independent of the U-Net. v1 reported ~0.65-0.71 for the
     seasonal label; a better label should raise this, and -- more tellingly
     -- should move the signal off pure land-cover bands.

  2. Permutation importance. Shuffle one band at a time and measure the AUC
     drop. v1 found the signal concentrated in SWIR/NIR (land cover) with
     terrain contributing little -- the smoking gun for "learning land cover,
     not flooding." Re-running this on the event label shows whether the
     dependence has shifted.

Operating on per-pixel feature matrices (not the U-Net) keeps these fast and
model-free, so they are a clean, independent read on the label itself.
"""

import numpy as np


def tile_pixels(tile: np.ndarray, input_band_indices, label_band_index,
                subsample: int = None, seed: int = 42) -> tuple:
    """
    Flatten one (bands, H, W) tile into a per-pixel feature matrix ``X``
    (n_pixels, n_input_bands) and binary label vector ``y``, dropping pixels
    that are non-finite in any input band or the label. Optionally subsample
    to ``subsample`` pixels for speed (stratification is not attempted; the
    classifier handles imbalance via class weights downstream).
    """
    inputs = tile[list(input_band_indices)]
    label = tile[label_band_index]
    n_bands, h, w = inputs.shape

    X = inputs.reshape(n_bands, -1).T
    y = (label.reshape(-1) > 0.5).astype(np.int64)
    finite = np.isfinite(X).all(axis=1) & np.isfinite(label.reshape(-1))
    X, y = X[finite], y[finite]

    if subsample is not None and X.shape[0] > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=subsample, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def fit_pixel_classifier(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """
    Standardise features and fit a balanced logistic regression -- the same
    cheap, interpretable pixel classifier used as the v1 label-signal probe.
    Returns a fitted sklearn Pipeline. Raises if ``y`` has a single class,
    since AUC is undefined there (and a single-class label is itself a red
    flag worth surfacing loudly).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if np.unique(y).size < 2:
        raise ValueError(
            "Training pixels contain a single class -- cannot fit a classifier "
            "or compute AUC. The label is degenerate on this location."
        )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    clf.fit(X, y)
    return clf


def evaluate_auc(clf, X: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC of a fitted classifier on ``(X, y)``. Returns nan if ``y`` is single-class."""
    from sklearn.metrics import roc_auc_score

    if np.unique(y).size < 2:
        return float("nan")
    scores = clf.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, scores))


def permutation_importance_auc(clf, X: np.ndarray, y: np.ndarray,
                               feature_names, n_repeats: int = 10,
                               seed: int = 42) -> dict:
    """
    Per-feature AUC drop under permutation, as a dict ``{feature_name:
    {"mean": ..., "std": ...}}`` sorted by mean importance (descending). A
    large drop means the label leans on that band. Comparing this profile
    between the seasonal and event labels is the clearest evidence of whether
    the land-cover dependence has been reduced.
    """
    from sklearn.inspection import permutation_importance

    if np.unique(y).size < 2:
        raise ValueError("Cannot compute permutation importance on single-class labels.")
    result = permutation_importance(
        clf, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed,
    )
    importances = {
        name: {"mean": float(result.importances_mean[i]),
               "std": float(result.importances_std[i])}
        for i, name in enumerate(feature_names)
    }
    return dict(sorted(importances.items(), key=lambda kv: kv[1]["mean"], reverse=True))


def cross_location_signal(train_tile: np.ndarray, val_tile: np.ndarray,
                          input_band_indices, label_band_index, feature_names,
                          subsample: int = 40000, n_repeats: int = 10,
                          seed: int = 42) -> dict:
    """
    End-to-end label-signal report between two locations: fit on the training
    tile's pixels, score AUC on both, and compute permutation importance on
    the held-out tile. Returns a dict with ``train_auc``, ``val_auc`` and
    ``permutation_importance`` -- the same three numbers the v1 post-mortem
    quoted, ready to tabulate seasonal-vs-event side by side.
    """
    X_tr, y_tr = tile_pixels(train_tile, input_band_indices, label_band_index, subsample, seed)
    X_val, y_val = tile_pixels(val_tile, input_band_indices, label_band_index, subsample, seed)

    clf = fit_pixel_classifier(X_tr, y_tr, seed=seed)
    report = {
        "train_auc": evaluate_auc(clf, X_tr, y_tr),
        "val_auc": evaluate_auc(clf, X_val, y_val),
        "train_positive_fraction": float(y_tr.mean()),
        "val_positive_fraction": float(y_val.mean()),
    }
    if np.unique(y_val).size >= 2:
        report["permutation_importance"] = permutation_importance_auc(
            clf, X_val, y_val, feature_names, n_repeats=n_repeats, seed=seed
        )
    else:
        report["permutation_importance"] = None
    return report
