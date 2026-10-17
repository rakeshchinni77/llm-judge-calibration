"""
test_evaluate.py — Unit tests for evaluate.py CLI and evaluation handlers.

Verifies:
1. handle_test_judge outputs parseable JSON to stdout.
2. update_metrics_csv preserves required schema and updates split rows.
3. handle_split('dev') computes and records bootstrap metrics.
4. handle_split('test') writes judge_test.csv with exact required columns.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import io
import pandas as pd
import pytest

from evaluate import (
    METRICS_COLUMNS,
    TEST_PREDICTION_COLUMNS,
    get_prompt_version,
    handle_split,
    handle_test_judge,
    update_metrics_csv,
)


def test_get_prompt_version() -> None:
    """Prompt versioning must return non-empty tag and digest."""
    tag, digest = get_prompt_version()
    assert tag.startswith("v1.0-")
    assert len(digest) > 0


def test_update_metrics_csv(tmp_path: Path) -> None:
    """update_metrics_csv must maintain schema and update existing split rows."""
    out_csv = tmp_path / "metrics.csv"
    row1 = {
        "split": "dev",
        "tpr_mean": 0.5,
        "tpr_ci_low": 0.4,
        "tpr_ci_high": 0.6,
        "tnr_mean": 0.9,
        "tnr_ci_low": 0.8,
        "tnr_ci_high": 1.0,
        "agreement": 0.8,
        "kappa_mean": 0.55,
        "kappa_ci_low": 0.45,
        "kappa_ci_high": 0.65,
    }
    update_metrics_csv(row1, output_path=out_csv)
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert list(df.columns) == METRICS_COLUMNS

    # Update dev row with new point estimate
    row1_updated = dict(row1)
    row1_updated["tpr_mean"] = 0.52
    update_metrics_csv(row1_updated, output_path=out_csv)
    df2 = pd.read_csv(out_csv)
    assert len(df2) == 1
    assert df2.loc[0, "tpr_mean"] == 0.52

    # Add test row
    row2 = dict(row1)
    row2["split"] = "test"
    update_metrics_csv(row2, output_path=out_csv)
    df3 = pd.read_csv(out_csv)
    assert len(df3) == 2
    assert set(df3["split"]) == {"dev", "test"}


def test_handle_test_judge_stdout_json() -> None:
    """handle_test_judge must write pure JSON to stdout."""
    stdout_capture = io.StringIO()
    with patch("sys.stdout", stdout_capture):
        handle_test_judge("Sample text explaining BFS vs DFS", backend="mock")

    output = stdout_capture.getvalue().strip()
    parsed = json.loads(output)
    assert "critique" in parsed
    assert "pass" in parsed
    assert isinstance(parsed["pass"], bool)


def test_handle_split_dev_mock(tmp_path: Path) -> None:
    """handle_split('dev') must execute and populate metrics."""
    metrics = handle_split(
        split="dev",
        seed=42,
        backend="mock",
        output_dir=tmp_path,
    )
    assert metrics["split"] == "dev"
    assert "kappa_mean" in metrics
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "run_manifest_dev.json").exists()


def test_handle_split_test_mock(tmp_path: Path) -> None:
    """handle_split('test') must execute, write judge_test.csv, and manifest."""
    metrics = handle_split(
        split="test",
        seed=42,
        backend="mock",
        output_dir=tmp_path,
    )
    assert metrics["split"] == "test"

    pred_file = tmp_path / "judge_test.csv"
    assert pred_file.exists()
    df_pred = pd.read_csv(pred_file)
    assert len(df_pred) == 112
    assert list(df_pred.columns) == TEST_PREDICTION_COLUMNS
