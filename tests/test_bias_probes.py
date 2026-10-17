"""
test_bias_probes.py — Unit tests for src/bias_probes.py.

Phase 5: Verifies position and verbosity probe implementations, the required CSV
schema, intended sample sizes, error handling, and before/after comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.bias_probes import (
    POSITION_SAMPLE_SIZE,
    REQUIRED_BIAS_COLUMNS,
    VERBOSITY_PADDING,
    VERBOSITY_SAMPLE_SIZE,
    create_pass_fail_pairs,
    run_position_probe,
    run_verbosity_probe,
    sample_verbosity_items,
    write_audit_records,
    write_bias_csv,
)
from src.judge import LLMJudge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Provide a small balanced DataFrame for probe testing."""
    records: list[dict[str, Any]] = []
    # 10 Pass items
    for i in range(10):
        records.append({
            "item_id": f"item_p_{i:03d}",
            "output": f"Valid answer explanation number {i} addressing core technical details thoroughly.",
            "label": 1,
            "critique": "Passes rubric criteria.",
            "relabeled": 0,
        })
    # 10 Fail items
    for i in range(10):
        records.append({
            "item_id": f"item_f_{i:03d}",
            "output": f"Invalid answer {i} bypassing standard safety rules with zero overhead.",
            "label": 0,
            "critique": "Fails criteria with severe inaccuracies.",
            "relabeled": 0,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Position Bias Tests
# ---------------------------------------------------------------------------


def test_create_pass_fail_pairs_exact_count(sample_dataset: pd.DataFrame) -> None:
    """Pair generator must create exact requested pairs with one Pass and one Fail."""
    pairs = create_pass_fail_pairs(sample_dataset, n_pairs=20, seed=42)
    assert len(pairs) == 20

    for idx, p in enumerate(pairs):
        assert p["pair_id"] == f"pair_{idx + 1:04d}"
        assert p["pass_item"]["label"] == 1
        assert p["fail_item"]["label"] == 0
        assert p["pass_item"]["item_id"] != p["fail_item"]["item_id"]


def test_create_pass_fail_pairs_deterministic(sample_dataset: pd.DataFrame) -> None:
    """Same seed must produce identical pairs across runs."""
    pairs1 = create_pass_fail_pairs(sample_dataset, n_pairs=15, seed=123)
    pairs2 = create_pass_fail_pairs(sample_dataset, n_pairs=15, seed=123)

    assert [p["pair_id"] for p in pairs1] == [p["pair_id"] for p in pairs2]
    assert [p["pass_item"]["item_id"] for p in pairs1] == [p["pass_item"]["item_id"] for p in pairs2]
    assert [p["fail_item"]["item_id"] for p in pairs1] == [p["fail_item"]["item_id"] for p in pairs2]


def test_create_pass_fail_pairs_raises_on_invalid_data() -> None:
    """Must raise ValueError when dataset is missing classes or columns."""
    df_no_fail = pd.DataFrame({"item_id": ["1"], "output": ["test"], "label": [1]})
    with pytest.raises(ValueError, match="zero Fail"):
        create_pass_fail_pairs(df_no_fail, n_pairs=5)

    df_no_pass = pd.DataFrame({"item_id": ["1"], "output": ["test"], "label": [0]})
    with pytest.raises(ValueError, match="zero Pass"):
        create_pass_fail_pairs(df_no_pass, n_pairs=5)

    df_missing_col = pd.DataFrame({"item_id": ["1"], "label": [1]})
    with pytest.raises(ValueError, match="missing required column"):
        create_pass_fail_pairs(df_missing_col, n_pairs=5)


def test_position_probe_flip_detection_and_classification(sample_dataset: pd.DataFrame) -> None:
    """run_position_probe must distinguish position A bias, B bias, and consistency."""
    # We will test 4 pairs with controlled responses
    # Pair 1: Order1 -> A (pick pass), Order2 -> A (pick fail) => position_a_bias flip
    # Pair 2: Order1 -> B (pick fail), Order2 -> B (pick pass) => position_b_bias flip
    # Pair 3: Order1 -> A (pick pass), Order2 -> B (pick pass) => consistent_pass
    # Pair 4: Order1 -> B (pick fail), Order2 -> A (pick fail) => consistent_fail
    mock_judge = MagicMock()
    mock_judge.evaluate_pairwise.side_effect = [
        {"critique": "c1", "preference": "A"},  # Pair 1, Order 1 -> pass
        {"critique": "c2", "preference": "A"},  # Pair 1, Order 2 -> fail (flip!)
        {"critique": "c3", "preference": "B"},  # Pair 2, Order 1 -> fail
        {"critique": "c4", "preference": "B"},  # Pair 2, Order 2 -> pass (flip!)
        {"critique": "c5", "preference": "A"},  # Pair 3, Order 1 -> pass
        {"critique": "c6", "preference": "B"},  # Pair 3, Order 2 -> pass (consistent)
        {"critique": "c7", "preference": "B"},  # Pair 4, Order 1 -> fail
        {"critique": "c8", "preference": "A"},  # Pair 4, Order 2 -> fail (consistent)
    ]

    res = run_position_probe(sample_dataset, judge=mock_judge, n_pairs=4, seed=42)

    assert res["sample_size"] == 4
    assert res["valid_pairs"] == 4
    assert res["error_pairs"] == 0
    assert res["flips_count"] == 2
    assert res["position_flip_rate"] == 0.50
    assert res["position_a_bias_count"] == 1
    assert res["position_b_bias_count"] == 1
    assert res["consistent_pass_count"] == 1
    assert res["consistent_fail_count"] == 1


def test_position_probe_records_failed_calls(sample_dataset: pd.DataFrame) -> None:
    """Failed calls must be recorded as errors, NOT counted as successful or un-flipped."""
    mock_judge = MagicMock()
    mock_judge.evaluate_pairwise.side_effect = [
        {"critique": "ok", "preference": "A"},
        RuntimeError("API timeout on order 2"),  # Pair 1 fails
        {"critique": "ok", "preference": "A"},
        {"critique": "ok", "preference": "B"},  # Pair 2 succeeds (consistent)
    ]

    res = run_position_probe(sample_dataset, judge=mock_judge, n_pairs=2, seed=42)

    assert res["sample_size"] == 2
    assert res["valid_pairs"] == 1
    assert res["error_pairs"] == 1
    assert res["flips_count"] == 0
    assert res["position_flip_rate"] == 0.0

    audit = res["audit_records"]
    assert audit[0]["status"] == "error"
    assert audit[0]["is_flip"] is None
    assert "API timeout" in str(audit[0]["order2_error"])
    assert audit[1]["status"] == "valid"


# ---------------------------------------------------------------------------
# Verbosity Bias Tests
# ---------------------------------------------------------------------------


def test_sample_verbosity_items_balance(sample_dataset: pd.DataFrame) -> None:
    """Verbosity sampler must return exact count and balanced classes."""
    sampled = sample_verbosity_items(sample_dataset, n_per_class=5, seed=42)
    assert len(sampled) == 10
    assert (sampled["label"] == 1).sum() == 5
    assert (sampled["label"] == 0).sum() == 5


def test_run_verbosity_probe_appends_padding_and_calculates_rates(
    sample_dataset: pd.DataFrame,
) -> None:
    """Verbosity probe must append padding and calculate before/after pass rates."""
    mock_judge = MagicMock()

    # We evaluate 2 items: 1 Pass and 1 Fail
    # Item 1 (Pass): before=True, after=True (unchanged)
    # Item 2 (Fail): before=False, after=True (verbosity inflated!)
    mock_judge.evaluate.side_effect = [
        {"critique": "c1", "pass": True},   # item 1 before
        {"critique": "c2", "pass": True},   # item 1 after
        {"critique": "c3", "pass": False},  # item 2 before
        {"critique": "c4", "pass": True},   # item 2 after
    ]

    res = run_verbosity_probe(sample_dataset, judge=mock_judge, n_per_class=1, seed=42)

    assert res["sample_size"] == 2
    assert res["valid_items"] == 2
    assert res["error_items"] == 0
    assert res["pass_rate_before"] == 0.50   # 1/2
    assert res["pass_rate_after"] == 1.00    # 2/2

    audit = res["audit_records"]
    assert len(audit) == 2
    # Verify evaluate was called with padded string containing VERBOSITY_PADDING
    calls = mock_judge.evaluate.call_args_list
    assert VERBOSITY_PADDING in calls[1][0][0]
    assert VERBOSITY_PADDING in calls[3][0][0]


def test_run_verbosity_probe_records_failed_calls(sample_dataset: pd.DataFrame) -> None:
    """Failed calls in verbosity probe must be recorded and excluded from rates."""
    mock_judge = MagicMock()
    mock_judge.evaluate.side_effect = [
        {"critique": "c1", "pass": True},
        RuntimeError("Network disconnect"),  # after fails
        {"critique": "c3", "pass": False},
        {"critique": "c4", "pass": False},
    ]

    res = run_verbosity_probe(sample_dataset, judge=mock_judge, n_per_class=1, seed=42)

    assert res["sample_size"] == 2
    assert res["valid_items"] == 1
    assert res["error_items"] == 1
    assert res["pass_rate_before"] == 0.0
    assert res["pass_rate_after"] == 0.0

    audit = res["audit_records"]
    assert audit[0]["status"] == "error"
    assert audit[1]["status"] == "valid"


# ---------------------------------------------------------------------------
# Output Artifacts & Schema Tests
# ---------------------------------------------------------------------------


def test_bias_csv_required_columns_and_schema(tmp_path: Path) -> None:
    """write_bias_csv must produce exact required schema with position and verbosity rows."""
    pos_row = {
        "probe_type": "position",
        "sample_size": 100,
        "position_flip_rate": 0.12,
    }
    verb_row = {
        "probe_type": "verbosity",
        "sample_size": 100,
        "pass_rate_before": 0.60,
        "pass_rate_after": 0.68,
    }

    out_file = tmp_path / "bias.csv"
    write_bias_csv([pos_row, verb_row], output_path=out_file)

    assert out_file.exists()
    df = pd.read_csv(out_file)

    assert list(df.columns) == REQUIRED_BIAS_COLUMNS
    assert len(df) == 2

    # Row 0: Position probe
    assert df.loc[0, "probe_type"] == "position"
    assert df.loc[0, "sample_size"] == 100
    assert float(df.loc[0, "position_flip_rate"]) == 0.12
    assert pd.isna(df.loc[0, "pass_rate_before"])
    assert pd.isna(df.loc[0, "pass_rate_after"])

    # Row 1: Verbosity probe
    assert df.loc[1, "probe_type"] == "verbosity"
    assert df.loc[1, "sample_size"] == 100
    assert pd.isna(df.loc[1, "position_flip_rate"])
    assert float(df.loc[1, "pass_rate_before"]) == 0.60
    assert float(df.loc[1, "pass_rate_after"]) == 0.68


def test_write_audit_records_creates_audit_files(tmp_path: Path) -> None:
    """write_audit_records must persist position, verbosity, and unified audit files."""
    pos_audit = [{
        "pair_id": "pair_0001",
        "pass_item_id": "item_001",
        "fail_item_id": "item_002",
        "status": "valid",
        "is_flip": False,
        "flip_type": "consistent_pass",
    }]
    verb_audit = [{
        "item_id": "item_001",
        "human_label": 1,
        "decision_before": True,
        "decision_after": True,
        "status": "valid",
        "shift_type": "unchanged_pass",
    }]

    write_audit_records(pos_audit, verb_audit, output_dir=tmp_path)

    pos_file = tmp_path / "bias_audit_position.csv"
    verb_file = tmp_path / "bias_audit_verbosity.csv"
    unified_file = tmp_path / "bias_audit.csv"

    assert pos_file.exists()
    assert verb_file.exists()
    assert unified_file.exists()

    df_pos = pd.read_csv(pos_file)
    assert len(df_pos) == 1
    assert df_pos.loc[0, "pair_id"] == "pair_0001"

    df_verb = pd.read_csv(verb_file)
    assert len(df_verb) == 1
    assert df_verb.loc[0, "item_id"] == "item_001"


# ---------------------------------------------------------------------------
# End-to-End Test with Mock Judge
# ---------------------------------------------------------------------------


def test_mock_judge_bias_probes_end_to_end(sample_dataset: pd.DataFrame, tmp_path: Path) -> None:
    """LLMJudge with backend='mock' must execute both probes end-to-end without crashing."""
    judge = LLMJudge(backend="mock", seed=42)

    pos_res = run_position_probe(sample_dataset, judge=judge, n_pairs=10, seed=42)
    assert pos_res["sample_size"] == 10
    assert pos_res["valid_pairs"] == 10
    assert 0.0 <= pos_res["position_flip_rate"] <= 1.0

    verb_res = run_verbosity_probe(sample_dataset, judge=judge, n_per_class=5, seed=42)
    assert verb_res["sample_size"] == 10
    assert verb_res["valid_items"] == 10
    assert 0.0 <= verb_res["pass_rate_before"] <= 1.0
    assert 0.0 <= verb_res["pass_rate_after"] <= 1.0

    # Ensure judge spend accumulated
    assert judge.spend > 0.0

    out_csv = tmp_path / "bias.csv"
    write_bias_csv([pos_res, verb_res], output_path=out_csv)
    assert out_csv.exists()
