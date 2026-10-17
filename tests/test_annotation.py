"""
test_annotation.py — Unit tests for src/annotation.py.

Tests verify:
- Schema validation passes for valid 250-row dataset
- Validation fails for missing required columns
- Validation fails for disallowed extra columns in labels.csv
- Validation fails for non-binary labels (values other than 0 or 1)
- Validation fails for non-binary relabeled values
- Validation fails for duplicate item_ids
- Validation fails for empty outputs or empty item_ids
- Validation fails for empty critiques
- Validation fails for datasets with fewer than 250 rows
- extract_labels_csv extracts 5 required columns and drops template extras
- AnnotationAudit records sessions and relabel events correctly
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.annotation import (
    AnnotationAudit,
    AnnotationValidator,
    REQUIRED_COLUMNS,
    SchemaError,
    extract_labels_csv,
)


@pytest.fixture()
def valid_labels_df() -> pd.DataFrame:
    """Fixture providing a valid 250-row DataFrame."""
    rows = []
    for i in range(250):
        rows.append(
            {
                "item_id": f"item_{i:04d}",
                "output": f"Technical explanation for item {i} with sufficient technical detail.",
                "label": i % 2,
                "critique": f"Valid critique explaining why item {i} passes or fails rubric criteria.",
                "relabeled": 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def valid_labels_csv_file(tmp_path: Path, valid_labels_df: pd.DataFrame) -> Path:
    """Write valid 250-row CSV file."""
    path = tmp_path / "labels.csv"
    valid_labels_df.to_csv(path, index=False)
    return path


def test_validator_passes_on_valid_data(valid_labels_csv_file: Path) -> None:
    """A valid 250-row dataset should pass validation with no errors."""
    validator = AnnotationValidator(valid_labels_csv_file)
    result = validator.validate()
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.n_rows == 250
    assert result.n_pass == 125
    assert result.n_fail == 125
    assert result.fail_rate == 0.5


def test_validator_fails_missing_columns(tmp_path: Path, valid_labels_df: pd.DataFrame) -> None:
    """Validation must fail if any required column is missing."""
    df_missing = valid_labels_df.drop(columns=["critique"])
    path = tmp_path / "labels_missing.csv"
    df_missing.to_csv(path, index=False)

    validator = AnnotationValidator(path)
    result = validator.validate()
    assert result.is_valid is False
    assert any("critique" in err for err in result.errors)


def test_validator_fails_extra_columns(tmp_path: Path, valid_labels_df: pd.DataFrame) -> None:
    """Validation must fail if extra columns (e.g. question) are present in labels.csv."""
    df_extra = valid_labels_df.copy()
    df_extra["question"] = "What is Python GIL?"
    path = tmp_path / "labels_extra.csv"
    df_extra.to_csv(path, index=False)

    validator = AnnotationValidator(path)
    result = validator.validate()
    assert result.is_valid is False
    assert any("Extra columns" in err for err in result.errors)


def test_validator_fails_invalid_labels(tmp_path: Path, valid_labels_df: pd.DataFrame) -> None:
    """Validation must fail if label column contains values other than 0 or 1."""
    df_invalid = valid_labels_df.copy()
    df_invalid["label"] = df_invalid["label"].astype(object)
    df_invalid.loc[0, "label"] = 2
    df_invalid.loc[1, "label"] = "Pass"
    path = tmp_path / "labels_invalid_label.csv"
    df_invalid.to_csv(path, index=False)

    validator = AnnotationValidator(path)
    result = validator.validate()
    assert result.is_valid is False
    assert any("label" in err for err in result.errors)


def test_validator_fails_invalid_relabeled(tmp_path: Path, valid_labels_df: pd.DataFrame) -> None:
    """Validation must fail if relabeled column contains values other than 0 or 1."""
    df_invalid = valid_labels_df.copy()
    df_invalid.loc[0, "relabeled"] = 5
    path = tmp_path / "labels_invalid_relabeled.csv"
    df_invalid.to_csv(path, index=False)

    validator = AnnotationValidator(path)
    result = validator.validate()
    assert result.is_valid is False
    assert any("relabeled" in err for err in result.errors)


def test_validator_fails_duplicate_item_id(tmp_path: Path, valid_labels_df: pd.DataFrame) -> None:
    """Validation must fail if item_id is duplicated."""
    df_dupe = valid_labels_df.copy()
    df_dupe.loc[1, "item_id"] = df_dupe.loc[0, "item_id"]
    path = tmp_path / "labels_dupe.csv"
    df_dupe.to_csv(path, index=False)

    validator = AnnotationValidator(path)
    result = validator.validate()
    assert result.is_valid is False
    assert any("Duplicate item_ids" in err for err in result.errors)


def test_validator_fails_blank_output(tmp_path: Path, valid_labels_df: pd.DataFrame) -> None:
    """Validation must fail if any output is blank or missing."""
    df_blank = valid_labels_df.copy()
    df_blank.loc[0, "output"] = ""
    path = tmp_path / "labels_blank_output.csv"
    df_blank.to_csv(path, index=False)

    validator = AnnotationValidator(path)
    result = validator.validate()
    assert result.is_valid is False
    assert any("blank output" in err for err in result.errors)


def test_validator_fails_blank_critique(tmp_path: Path, valid_labels_df: pd.DataFrame) -> None:
    """Validation must fail if any critique is blank."""
    df_blank = valid_labels_df.copy()
    df_blank.loc[0, "critique"] = ""
    path = tmp_path / "labels_blank_critique.csv"
    df_blank.to_csv(path, index=False)

    validator = AnnotationValidator(path)
    result = validator.validate()
    assert result.is_valid is False
    assert any("blank critique" in err for err in result.errors)


def test_validator_fails_under_250_rows(tmp_path: Path, valid_labels_df: pd.DataFrame) -> None:
    """Validation must fail if row count is below 250."""
    df_short = valid_labels_df.iloc[:200]
    path = tmp_path / "labels_short.csv"
    df_short.to_csv(path, index=False)

    validator = AnnotationValidator(path)
    result = validator.validate()
    assert result.is_valid is False
    assert any("minimum required is 250" in err for err in result.errors)


def test_extract_labels_csv(tmp_path: Path, valid_labels_df: pd.DataFrame) -> None:
    """extract_labels_csv must drop extra columns and produce valid labels.csv."""
    df_template = valid_labels_df.copy()
    df_template["question"] = "What is a GIL in Python?"
    template_path = tmp_path / "annotation_template.csv"
    labels_path = tmp_path / "labels.csv"

    df_template.to_csv(template_path, index=False)
    n_written = extract_labels_csv(template_path, labels_path)
    assert n_written == 250

    extracted_df = pd.read_csv(labels_path)
    assert list(extracted_df.columns) == REQUIRED_COLUMNS
    assert len(extracted_df) == 250


def test_annotation_audit_records_sessions(tmp_path: Path) -> None:
    """AnnotationAudit must log start, end, and relabel events to jsonl."""
    audit_file = tmp_path / "annotation_audit.jsonl"
    audit = AnnotationAudit(audit_file)

    audit.start_session(annotator="test_user")
    audit.log_relabel("item_0001", old_label=1, new_label=0, reason="Found subtle factual error")
    elapsed = audit.end_session(n_labeled=25, n_relabeled=1)

    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3

    events = [json.loads(line) for line in lines]
    assert events[0]["event"] == "session_start"
    assert events[0]["annotator"] == "test_user"
    assert events[1]["event"] == "relabel"
    assert events[1]["item_id"] == "item_0001"
    assert events[2]["event"] == "session_end"
    assert events[2]["n_labeled"] == 25
