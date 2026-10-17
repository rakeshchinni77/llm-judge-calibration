"""
metrics.py — Statistical Metrics Engine.

Provides:
    compute_tpr()          True Positive Rate (failure recall; positive label = 0).
    compute_tnr()          True Negative Rate (pass specificity; negative label = 1).
    compute_agreement()    Raw accuracy between human and judge labels.
    compute_kappa()        Cohen's Kappa inter-rater agreement corrected for chance.
    bootstrap_metric()     Non-parametric bootstrap CI estimation via resampling.
    compute_all_metrics()  Computes full metrics row for one split (metrics.csv schema).

Statistical definitions:
    In this task, label 0 (Fail) is the POSITIVE class for failure detection:
        - True Positive (TP)  : human=0, judge=0 (correctly flagged failure)
        - False Negative (FN) : human=0, judge=1 (missed failure)
        - True Negative (TN)  : human=1, judge=1 (correctly approved pass)
        - False Positive (FP) : human=1, judge=0 (incorrectly flagged pass)

    TPR = TP / (TP + FN)  = P(judge=0 | human=0)
    TNR = TN / (TN + FP)  = P(judge=1 | human=1)
    Agreement             = (TP + TN) / N
    Cohen's Kappa         = (p_o - p_e) / (1 - p_e)

Undefined-Resample Policy:
    When a metric is mathematically undefined (e.g. zero human failures in the
    sample leading to a zero denominator for TPR, or a single-class sample leading
    to an undefined Kappa):
        1. The point estimate returns np.nan.
        2. In bootstrap iterations, undefined resamples produce np.nan and are
           filtered from percentile estimation rather than silently replaced with
           fabricated values (enforcing the engineering rule against hiding uncertainty).
        3. If all resamples are undefined, the CI bounds return (np.nan, np.nan, np.nan).
        4. Valid iterations calculate the 2.5th and 97.5th percentiles.

Phase gate: Phase 4
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable

import numpy as np
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BOOTSTRAP: int = 1000
CI_LOWER: float = 2.5
CI_UPPER: float = 97.5

# Label 0 represents Fail (the positive class for error detection)
POSITIVE_LABEL: int = 0
NEGATIVE_LABEL: int = 1


# ---------------------------------------------------------------------------
# Core individual metric functions
# ---------------------------------------------------------------------------

def compute_tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute True Positive Rate (sensitivity for the Failure class).

    A 'positive' is a human-labeled Fail (label=0).
    TPR answers: of all actual failures, what fraction did the judge correctly flag?

    Args:
        y_true: Ground-truth human labels (0=Fail, 1=Pass).
        y_pred: Predicted judge labels (0=Fail, 1=Pass).

    Returns:
        TPR as a float in [0.0, 1.0], or np.nan if no human failures exist.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n_failures = np.sum(yt == POSITIVE_LABEL)
    if n_failures == 0:
        return float(np.nan)

    tp = np.sum((yt == POSITIVE_LABEL) & (yp == POSITIVE_LABEL))
    return float(tp / n_failures)


def compute_tnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute True Negative Rate (specificity for the Pass class).

    A 'negative' is a human-labeled Pass (label=1).
    TNR answers: of all actual passes, what fraction did the judge correctly approve?

    Args:
        y_true: Ground-truth human labels (0=Fail, 1=Pass).
        y_pred: Predicted judge labels (0=Fail, 1=Pass).

    Returns:
        TNR as a float in [0.0, 1.0], or np.nan if no human passes exist.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n_passes = np.sum(yt == NEGATIVE_LABEL)
    if n_passes == 0:
        return float(np.nan)

    tn = np.sum((yt == NEGATIVE_LABEL) & (yp == NEGATIVE_LABEL))
    return float(tn / n_passes)


def compute_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute raw accuracy (overall fraction of matching labels).

    Args:
        y_true: Ground-truth human labels.
        y_pred: Predicted judge labels.

    Returns:
        Fraction of matching labels in [0.0, 1.0], or np.nan if inputs are empty.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if len(yt) == 0:
        return float(np.nan)
    return float(np.mean(yt == yp))


def compute_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Cohen's Kappa inter-rater agreement corrected for chance.

    Args:
        y_true: Ground-truth human labels.
        y_pred: Predicted judge labels.

    Returns:
        Cohen's Kappa as a float in [-1.0, 1.0], or np.nan if degenerate/undefined.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    if len(yt) == 0:
        return float(np.nan)

    # Degenerate case: if all true labels are identical and all pred labels are identical
    if len(np.unique(yt)) < 2 and len(np.unique(yp)) < 2:
        return float(np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            score = cohen_kappa_score(yt, yp)
            return float(score)
        except Exception:
            return float(np.nan)


# ---------------------------------------------------------------------------
# Bootstrap CI estimation
# ---------------------------------------------------------------------------

def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iterations: int = N_BOOTSTRAP,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Estimate a metric's point estimate and 95 % CI via non-parametric bootstrap.

    Resamples with replacement n_iterations times using a fixed seed.
    Undefined resamples are filtered per the Undefined-Resample Policy.

    Args:
        y_true: Array of ground-truth labels (0 or 1).
        y_pred: Array of predicted labels (0 or 1).
        metric_fn: Metric function taking (y_true, y_pred) and returning float.
        n_iterations: Number of bootstrap iterations (default: 1000).
        seed: Fixed random seed for deterministic reproducibility.

    Returns:
        Tuple of (mean_estimate, ci_low, ci_high) where bounds are at 2.5th
        and 97.5th percentiles.

    Raises:
        ValueError: If array lengths mismatch, are empty, or contain non-binary values.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    # Input validation
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError("Arrays must be 1-dimensional")
    if len(yt) == 0 or len(yp) == 0:
        raise ValueError("Input arrays must not be empty")
    if len(yt) != len(yp):
        raise ValueError(f"Array length mismatch: len(y_true)={len(yt)} != len(y_pred)={len(yp)}")

    valid_labels = {0, 1}
    if not (set(np.unique(yt)).issubset(valid_labels) and set(np.unique(yp)).issubset(valid_labels)):
        raise ValueError("Labels must contain only 0 and 1 values")

    point_est = float(metric_fn(yt, yp))

    # Deterministic resampling
    rng = np.random.default_rng(seed)
    n_samples = len(yt)
    sample_indices = rng.choice(n_samples, size=(n_iterations, n_samples), replace=True)

    scores = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        idx = sample_indices[i]
        scores[i] = metric_fn(yt[idx], yp[idx])

    # Apply Undefined-Resample Policy
    valid_scores = scores[~np.isnan(scores)]

    if len(valid_scores) == 0:
        # All resamples undefined — do not fabricate values
        return (point_est, float(np.nan), float(np.nan))

    ci_low = float(np.percentile(valid_scores, CI_LOWER))
    ci_high = float(np.percentile(valid_scores, CI_UPPER))

    # Ensure CI bounds are ordered
    if ci_low > ci_high:
        ci_low, ci_high = ci_high, ci_low

    # Center estimate: point estimate if defined, otherwise bootstrap mean
    if not np.isnan(point_est):
        # Numerical tolerance clip to guarantee ci_low <= mean_est <= ci_high
        mean_est = float(np.clip(point_est, ci_low, ci_high))
    else:
        mean_est = float(np.mean(valid_scores))

    return (mean_est, ci_low, ci_high)


# ---------------------------------------------------------------------------
# Composite metrics computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute the full metrics row for one split matching the metrics.csv schema.

    Schema:
        split, tpr_mean, tpr_ci_low, tpr_ci_high,
        tnr_mean, tnr_ci_low, tnr_ci_high,
        agreement, kappa_mean, kappa_ci_low, kappa_ci_high

    Args:
        y_true: Ground-truth human labels.
        y_pred: Predicted judge labels.
        split_name: Name of the split ('train', 'dev', 'test').
        seed: Random seed for bootstrap reproducibility.

    Returns:
        Dict matching results/metrics.csv columns.
    """
    tpr_mean, tpr_low, tpr_high = bootstrap_metric(y_true, y_pred, compute_tpr, seed=seed)
    tnr_mean, tnr_low, tnr_high = bootstrap_metric(y_true, y_pred, compute_tnr, seed=seed)
    agreement = compute_agreement(y_true, y_pred)
    kappa_mean, kappa_low, kappa_high = bootstrap_metric(y_true, y_pred, compute_kappa, seed=seed)

    def _format(val: float) -> float:
        return float(np.nan) if np.isnan(val) else round(float(val), 4)

    return {
        "split": split_name,
        "tpr_mean": _format(tpr_mean),
        "tpr_ci_low": _format(tpr_low),
        "tpr_ci_high": _format(tpr_high),
        "tnr_mean": _format(tnr_mean),
        "tnr_ci_low": _format(tnr_low),
        "tnr_ci_high": _format(tnr_high),
        "agreement": _format(agreement),
        "kappa_mean": _format(kappa_mean),
        "kappa_ci_low": _format(kappa_low),
        "kappa_ci_high": _format(kappa_high),
    }
