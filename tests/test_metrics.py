"""
test_metrics.py — Unit tests for src/metrics.py.

Tests verify:
- Invariant: ci_low <= mean <= ci_high for all bootstrap outputs
- TPR / TNR definitions on known confusion matrices with label 0 as Fail
- Teaching anchor: all-Pass judge on imbalanced data has high agreement but 0 TPR
- Perfect predictions yield TPR = 1.0, TNR = 1.0, agreement = 1.0, kappa = 1.0
- All predictions Pass yields TPR = 0.0 when failures exist
- No human failures yields undefined (NaN) TPR without fabricated zeroes
- All human labels of one class yields undefined (NaN) Kappa
- Bootstrap resampling is fully deterministic under a fixed seed
- Input validation on empty arrays, mismatched lengths, and non-binary labels
- compute_all_metrics() schema matches results/metrics.csv contract
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    bootstrap_metric,
    compute_agreement,
    compute_all_metrics,
    compute_kappa,
    compute_tnr,
    compute_tpr,
)


# ---------------------------------------------------------------------------
# Core metric definitions on known confusion matrices
# ---------------------------------------------------------------------------

def test_tpr_perfect_detection() -> None:
    """TPR must be 1.0 when all failures (label 0) are correctly detected."""
    # 2 human failures (both predicted 0), 2 human passes (both predicted 1)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert compute_tpr(y_true, y_pred) == pytest.approx(1.0)


def test_tnr_perfect_approval() -> None:
    """TNR must be 1.0 when all passes (label 1) are correctly approved."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert compute_tnr(y_true, y_pred) == pytest.approx(1.0)


def test_tpr_tnr_known_confusion_matrix() -> None:
    """Validate TPR and TNR on an explicit 2x2 confusion matrix.

    Human: 4 failures (0), 6 passes (1) -> N=10
    Predictions:
      - 3 failures correctly flagged (TP = 3)
      - 1 failure missed as pass (FN = 1)
      - 5 passes correctly approved (TN = 5)
      - 1 pass wrongly flagged as fail (FP = 1)

    TPR = TP / (TP + FN) = 3 / 4 = 0.75
    TNR = TN / (TN + FP) = 5 / 6 = 0.8333...
    Agreement = (3 + 5) / 10 = 0.80
    """
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])

    assert compute_tpr(y_true, y_pred) == pytest.approx(0.75)
    assert compute_tnr(y_true, y_pred) == pytest.approx(5 / 6)
    assert compute_agreement(y_true, y_pred) == pytest.approx(0.80)


def test_hamel_husain_teaching_anchor() -> None:
    """All-Pass judge on 5% failure rate has 95% agreement but 0% TPR."""
    # 1 failure, 19 passes (5% failure rate)
    y_true = np.array([0] + [1] * 19)
    # Judge always predicts Pass (1)
    y_pred = np.array([1] * 20)

    agreement = compute_agreement(y_true, y_pred)
    tpr = compute_tpr(y_true, y_pred)
    tnr = compute_tnr(y_true, y_pred)

    assert agreement == pytest.approx(0.95)
    assert tpr == pytest.approx(0.0), "TPR must be 0 when judge detects none of the errors"
    assert tnr == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Edge cases & undefined-resample policy
# ---------------------------------------------------------------------------

def test_tpr_undefined_when_no_human_failures() -> None:
    """When human labels have zero failures, TPR is mathematically undefined (NaN)."""
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 1])
    assert np.isnan(compute_tpr(y_true, y_pred))


def test_tnr_undefined_when_no_human_passes() -> None:
    """When human labels have zero passes, TNR is mathematically undefined (NaN)."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 1, 0])
    assert np.isnan(compute_tnr(y_true, y_pred))


def test_kappa_perfect_agreement() -> None:
    """Kappa must be 1.0 when all labels match across both classes."""
    y = np.array([0, 0, 1, 1, 0, 1])
    assert compute_kappa(y, y) == pytest.approx(1.0)


def test_kappa_no_better_than_chance() -> None:
    """Kappa must be <= 0 when predictions are inverted or no better than chance."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    assert compute_kappa(y_true, y_pred) <= 0.0


def test_kappa_undefined_single_class() -> None:
    """Kappa is undefined (NaN) when all human and predicted labels belong to a single class."""
    y = np.array([1, 1, 1, 1])
    assert np.isnan(compute_kappa(y, y))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def test_bootstrap_ci_invariant() -> None:
    """CI invariant: ci_low <= mean <= ci_high must hold for all metrics."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0])

    for metric_fn in (compute_tpr, compute_tnr, compute_agreement, compute_kappa):
        mean, low, high = bootstrap_metric(y_true, y_pred, metric_fn, seed=42)
        assert low <= mean <= high, f"CI invariant violated for {metric_fn.__name__}: {low} <= {mean} <= {high}"


def test_bootstrap_determinism() -> None:
    """bootstrap_metric() must return identical results given the same seed."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])

    res_1 = bootstrap_metric(y_true, y_pred, compute_tpr, seed=123)
    res_2 = bootstrap_metric(y_true, y_pred, compute_tpr, seed=123)
    assert res_1 == res_2, "Bootstrap output was non-deterministic"


def test_bootstrap_undefined_sample_policy() -> None:
    """When a metric is undefined across all samples, return NaNs without fabricated zeroes."""
    y_true = np.array([1, 1, 1, 1])  # No failures
    y_pred = np.array([1, 1, 1, 1])

    mean, low, high = bootstrap_metric(y_true, y_pred, compute_tpr, seed=42)
    assert np.isnan(mean)
    assert np.isnan(low)
    assert np.isnan(high)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

def test_input_validation_mismatched_lengths() -> None:
    """Mismatched array lengths must raise ValueError."""
    with pytest.raises(ValueError, match="Array length mismatch"):
        bootstrap_metric(np.array([0, 1]), np.array([0, 1, 0]), compute_tpr)


def test_input_validation_empty_arrays() -> None:
    """Empty arrays must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        bootstrap_metric(np.array([]), np.array([]), compute_tpr)


def test_input_validation_non_binary_labels() -> None:
    """Non-binary labels (values outside {0, 1}) must raise ValueError."""
    with pytest.raises(ValueError, match="Labels must contain only 0 and 1"):
        bootstrap_metric(np.array([0, 2]), np.array([0, 1]), compute_tpr)


# ---------------------------------------------------------------------------
# compute_all_metrics schema contract
# ---------------------------------------------------------------------------

def test_compute_all_metrics_schema() -> None:
    """compute_all_metrics must return exact columns matching results/metrics.csv."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])

    row = compute_all_metrics(y_true, y_pred, split_name="dev", seed=42)
    expected_cols = [
        "split",
        "tpr_mean",
        "tpr_ci_low",
        "tpr_ci_high",
        "tnr_mean",
        "tnr_ci_low",
        "tnr_ci_high",
        "agreement",
        "kappa_mean",
        "kappa_ci_low",
        "kappa_ci_high",
    ]
    assert list(row.keys()) == expected_cols
    assert row["split"] == "dev"
    assert row["tpr_ci_low"] <= row["tpr_mean"] <= row["tpr_ci_high"]
    assert row["tnr_ci_low"] <= row["tnr_mean"] <= row["tnr_ci_high"]
    assert row["kappa_ci_low"] <= row["kappa_mean"] <= row["kappa_ci_high"]
