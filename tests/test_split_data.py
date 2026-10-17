"""
test_split_data.py — Unit tests for src/split_data.py.

Tests verify:
- Exact row counts for N=250 (Train=38, Dev=100, Test=112) and proportional scaling for other N
- Zero item_id overlap between any two splits
- Complete coverage: every item appears in exactly one split
- Schema preservation: all 5 required columns present in every split
- Stratification: class balance is preserved in splits
- Determinism: identical outputs when run with the same seed
- Error handling on invalid inputs (missing file, empty file, missing columns, duplicate IDs, invalid labels)
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
import pandas as pd

from src.split_data import (
    REQUIRED_COLUMNS,
    compute_split_counts,
    split_labels,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_labels_csv() -> str:
    """Generate a minimal 20-row labels CSV for unit testing."""
    rows = [
        f"item_{i:04d},Sample output text {i},{i % 2},Critique for item {i},0"
        for i in range(20)
    ]
    header = "item_id,output,label,critique,relabeled"
    return "\n".join([header] + rows)


@pytest.fixture()
def labels_250_df() -> pd.DataFrame:
    """Generate a 250-row DataFrame with 60% pass (150) and 40% fail (100)."""
    rows = []
    for i in range(250):
        # 150 ones, 100 zeros
        label = 1 if i < 150 else 0
        rows.append(
            {
                "item_id": f"item_{i:04d}",
                "output": f"Technical output content for item {i}.",
                "label": label,
                "critique": f"Critique justifying label {label} for item {i}.",
                "relabeled": 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Schema and count invariant tests
# ---------------------------------------------------------------------------

def test_labels_csv_schema(sample_labels_csv: str) -> None:
    """labels.csv must have exactly the 5 required columns."""
    df = pd.read_csv(io.StringIO(sample_labels_csv))
    assert set(REQUIRED_COLUMNS) == set(df.columns)


def test_label_values_binary(sample_labels_csv: str) -> None:
    """label column must contain only 0s and 1s."""
    df = pd.read_csv(io.StringIO(sample_labels_csv))
    assert set(df["label"].unique()).issubset({0, 1})


def test_relabeled_values_binary(sample_labels_csv: str) -> None:
    """relabeled column must contain only 0s and 1s."""
    df = pd.read_csv(io.StringIO(sample_labels_csv))
    assert set(df["relabeled"].unique()).issubset({0, 1})


def test_item_id_unique(sample_labels_csv: str) -> None:
    """item_id must be unique across all rows."""
    df = pd.read_csv(io.StringIO(sample_labels_csv))
    assert df["item_id"].is_unique


def test_compute_split_counts_exact_250() -> None:
    """compute_split_counts for N=250 must return exactly 38, 100, 112."""
    n_train, n_dev, n_test = compute_split_counts(250)
    assert n_train == 38
    assert n_dev == 100
    assert n_test == 112
    assert n_train + n_dev + n_test == 250


def test_split_exact_counts_250(tmp_path: Path, labels_250_df: pd.DataFrame) -> None:
    """split_labels must produce exactly 38 train, 100 dev, and 112 test items on 250 rows."""
    labels_path = tmp_path / "labels.csv"
    labels_250_df.to_csv(labels_path, index=False)
    splits = split_labels(labels_path=labels_path, output_dir=tmp_path / "splits", seed=42)

    assert len(splits["train"]) == 38, f"Expected 38 train items, got {len(splits['train'])}"
    assert len(splits["dev"]) == 100, f"Expected 100 dev items, got {len(splits['dev'])}"
    assert len(splits["test"]) == 112, f"Expected 112 test items, got {len(splits['test'])}"


def test_split_no_overlap(tmp_path: Path, labels_250_df: pd.DataFrame) -> None:
    """No item_id should appear in more than one split."""
    labels_path = tmp_path / "labels.csv"
    labels_250_df.to_csv(labels_path, index=False)
    splits = split_labels(labels_path=labels_path, output_dir=tmp_path / "splits", seed=42)

    train_ids = set(splits["train"]["item_id"])
    dev_ids = set(splits["dev"]["item_id"])
    test_ids = set(splits["test"]["item_id"])

    assert len(train_ids & dev_ids) == 0, "Overlap between train and dev"
    assert len(train_ids & test_ids) == 0, "Overlap between train and test"
    assert len(dev_ids & test_ids) == 0, "Overlap between dev and test"


def test_split_complete_coverage(tmp_path: Path, labels_250_df: pd.DataFrame) -> None:
    """Combined splits must cover every item from input exactly once."""
    labels_path = tmp_path / "labels.csv"
    labels_250_df.to_csv(labels_path, index=False)
    splits = split_labels(labels_path=labels_path, output_dir=tmp_path / "splits", seed=42)

    all_split_ids = (
        splits["train"]["item_id"].tolist()
        + splits["dev"]["item_id"].tolist()
        + splits["test"]["item_id"].tolist()
    )
    assert len(all_split_ids) == 250
    assert set(all_split_ids) == set(labels_250_df["item_id"])


def test_split_schema_preservation(tmp_path: Path, labels_250_df: pd.DataFrame) -> None:
    """All 5 required columns must be preserved in each split CSV file."""
    labels_path = tmp_path / "labels.csv"
    output_dir = tmp_path / "splits"
    labels_250_df.to_csv(labels_path, index=False)
    split_labels(labels_path=labels_path, output_dir=output_dir, seed=42)

    for split_name in ("train.csv", "dev.csv", "test.csv"):
        file_path = output_dir / split_name
        assert file_path.exists(), f"Split file missing: {split_name}"
        df_split = pd.read_csv(file_path)
        assert list(df_split.columns) == REQUIRED_COLUMNS


def test_split_stratification(tmp_path: Path, labels_250_df: pd.DataFrame) -> None:
    """Splits should maintain class proportions approximately equal to original."""
    labels_path = tmp_path / "labels.csv"
    labels_250_df.to_csv(labels_path, index=False)
    splits = split_labels(labels_path=labels_path, output_dir=tmp_path / "splits", seed=42)

    # 40% fail in input (100 / 250)
    for name, df in splits.items():
        fail_rate = (df["label"] == 0).mean()
        assert 0.35 <= fail_rate <= 0.45, f"Split {name} fail rate skewed: {fail_rate:.2%}"


def test_split_determinism(tmp_path: Path, labels_250_df: pd.DataFrame) -> None:
    """Running split_labels() twice with the same seed must produce identical splits."""
    labels_path = tmp_path / "labels.csv"
    labels_250_df.to_csv(labels_path, index=False)

    splits_a = split_labels(labels_path=labels_path, output_dir=tmp_path / "splits_a", seed=42)
    splits_b = split_labels(labels_path=labels_path, output_dir=tmp_path / "splits_b", seed=42)

    for name in ("train", "dev", "test"):
        ids_a = splits_a[name]["item_id"].tolist()
        ids_b = splits_b[name]["item_id"].tolist()
        assert ids_a == ids_b, f"Split '{name}' is non-deterministic with same seed"


def test_split_seed_variation(tmp_path: Path, labels_250_df: pd.DataFrame) -> None:
    """Running split_labels() with different seeds should produce different splits."""
    labels_path = tmp_path / "labels.csv"
    labels_250_df.to_csv(labels_path, index=False)

    splits_a = split_labels(labels_path=labels_path, output_dir=tmp_path / "splits_a", seed=42)
    splits_b = split_labels(labels_path=labels_path, output_dir=tmp_path / "splits_b", seed=999)

    # Train splits should differ with different seeds
    ids_a = splits_a["train"]["item_id"].tolist()
    ids_b = splits_b["train"]["item_id"].tolist()
    assert ids_a != ids_b, "Splits should vary when seed changes"


def test_split_error_nonexistent_file(tmp_path: Path) -> None:
    """split_labels must raise FileNotFoundError if file does not exist."""
    with pytest.raises(FileNotFoundError):
        split_labels(tmp_path / "nonexistent.csv", tmp_path / "splits")


def test_split_error_empty_file(tmp_path: Path) -> None:
    """split_labels must raise ValueError on empty dataset."""
    empty_file = tmp_path / "empty.csv"
    pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(empty_file, index=False)
    with pytest.raises(ValueError, match="empty"):
        split_labels(empty_file, tmp_path / "splits")


def test_split_error_missing_column(tmp_path: Path, labels_250_df: pd.DataFrame) -> None:
    """split_labels must raise ValueError if required columns are missing."""
    df_missing = labels_250_df.drop(columns=["critique"])
    path = tmp_path / "missing.csv"
    df_missing.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        split_labels(path, tmp_path / "splits")


def test_split_error_duplicate_id(tmp_path: Path, labels_250_df: pd.DataFrame) -> None:
    """split_labels must raise ValueError if duplicate item_ids exist."""
    df_dupe = labels_250_df.copy()
    df_dupe.loc[1, "item_id"] = df_dupe.loc[0, "item_id"]
    path = tmp_path / "dupe.csv"
    df_dupe.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate item_ids"):
        split_labels(path, tmp_path / "splits")


def test_split_error_invalid_label(tmp_path: Path, labels_250_df: pd.DataFrame) -> None:
    """split_labels must raise ValueError on non-binary labels."""
    df_invalid = labels_250_df.copy()
    df_invalid["label"] = df_invalid["label"].astype(object)
    df_invalid.loc[0, "label"] = "invalid_label"
    path = tmp_path / "invalid_label.csv"
    df_invalid.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Column 'label' must contain only 0 and 1"):
        split_labels(path, tmp_path / "splits")
