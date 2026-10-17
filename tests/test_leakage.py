"""
test_leakage.py — Unit tests for Phase 6 induced data leakage experiment.

Verifies:
1. Induced leakage experiment produces valid results/induced_failure.json with exact schema.
2. Contaminated dev kappa exceeds baseline dev kappa.
3. Exactly 10 Dev examples are leaked, and their IDs are logged in results/leakage_audit.json.
4. The normal few-shot loading path (load_train_few_shot) continues to strictly reject dev/test splits.
5. The test split (data/splits/test.csv) is completely isolated and untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.induce_leakage import (
    AUDIT_PATH,
    LEAKED_ITEMS_COUNT,
    OUTPUT_PATH,
    run_leakage_experiment,
)
from src.judge import load_train_few_shot


def test_leakage_experiment_schema_and_metrics(tmp_path: Path) -> None:
    """Experiment must produce JSON with exact required keys and positive inflation."""
    result = run_leakage_experiment(
        train_path="data/splits/train.csv",
        dev_path="data/splits/dev.csv",
        n_leaked=10,
        seed=42,
        backend="mock",
    )

    # 1. Output file exists
    assert OUTPUT_PATH.exists()

    # 2. Schema check: exact required keys
    loaded = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    assert set(loaded.keys()) == {
        "baseline_dev_kappa",
        "contaminated_dev_kappa",
        "leaked_items_count",
    }

    # 3. Invariants
    assert loaded["leaked_items_count"] == 10
    assert isinstance(loaded["baseline_dev_kappa"], float)
    assert isinstance(loaded["contaminated_dev_kappa"], float)
    assert loaded["contaminated_dev_kappa"] > loaded["baseline_dev_kappa"]


def test_leakage_audit_file_structure_and_ids() -> None:
    """Audit artifact must record exactly 10 leaked item IDs belonging to Dev split."""
    assert AUDIT_PATH.exists()
    audit_data = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    assert audit_data["leaked_items_count"] == 10
    assert len(audit_data["leaked_items"]) == 10
    assert "scientific_justification" in audit_data

    # Verify all leaked items actually belong to dev.csv
    dev_df = pd.read_csv("data/splits/dev.csv")
    dev_item_ids = set(dev_df["item_id"])

    for item in audit_data["leaked_items"]:
        assert item["item_id"] in dev_item_ids
        assert item["label"] in (0, 1)


def test_normal_few_shot_loader_prevents_dev_test_contamination() -> None:
    """The normal evaluation loading path MUST raise ValueError if dev or test is supplied."""
    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        load_train_few_shot("data/splits/dev.csv")

    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        load_train_few_shot("data/splits/test.csv")


def test_test_split_remains_frozen() -> None:
    """Test split file must exist and remain untouched with 112 rows."""
    test_path = Path("data/splits/test.csv")
    assert test_path.exists()
    df_test = pd.read_csv(test_path)
    assert len(df_test) == 112
