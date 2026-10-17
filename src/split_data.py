"""
split_data.py — Deterministic Train/Dev/Test splitter.

Usage:
    python src/split_data.py --input data/labels.csv --output-dir data/splits/

Splits:
    Train : 15 % (38 items for N=250)  — few-shot examples only
    Dev   : 40 % (100 items for N=250) — iterative prompt development
    Test  : 45 % (112 items for N=250) — frozen evaluation; run exactly once

Exact requirements:
    - Train: 15% of 250 items = 38 items.
    - Dev: 40% = 100 items.
    - Test: 45% = 112 items.
    - Stratified by label (0/1) to preserve class balance where feasible while
      guaranteeing exact integer counts and complete coverage.
    - Fully deterministic given RANDOM_SEED from env (default: 42).
    - Every input item appears in exactly one split (no overlap, no omission).

Phase gate: Phase 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_FRAC: float = 0.15
DEV_FRAC: float = 0.40
TEST_FRAC: float = 0.45

REQUIRED_COLUMNS: list[str] = ["item_id", "output", "label", "critique", "relabeled"]


# ---------------------------------------------------------------------------
# Count calculation helper
# ---------------------------------------------------------------------------

def compute_split_counts(n_total: int) -> tuple[int, int, int]:
    """Compute exact integer counts for Train, Dev, and Test splits.

    For n_total = 250:
        Train = round(250 * 0.15) = 38
        Dev   = round(250 * 0.40) = 100
        Test  = 250 - 38 - 100    = 112

    Args:
        n_total: Total number of rows to split.

    Returns:
        Tuple of (n_train, n_dev, n_test) summing exactly to n_total.

    Raises:
        ValueError: If n_total < 3.
    """
    if n_total < 3:
        raise ValueError(f"Dataset must have at least 3 rows to create 3 splits; got {n_total}")

    n_train = int(round(n_total * TRAIN_FRAC))
    n_dev = int(round(n_total * DEV_FRAC))
    n_test = n_total - n_train - n_dev

    # Ensure each split receives at least 1 item if possible
    if n_train == 0 and n_total >= 3:
        n_train = 1
        n_test -= 1
    if n_dev == 0 and n_total >= 3:
        n_dev = 1
        n_test -= 1

    return n_train, n_dev, n_test


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def split_labels(
    labels_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Load labels.csv, stratify by label, and write three split CSV files.

    Preserves exact split counts (e.g. 38/100/112 for N=250), ensures no item_id
    overlap, guarantees complete coverage, and preserves all required columns.

    Args:
        labels_path: Path to the annotated labels CSV.
        output_dir:  Directory where train.csv, dev.csv, test.csv are written.
        seed:        Random seed for reproducibility.

    Returns:
        Dict with keys "train", "dev", "test" mapping to DataFrames.

    Raises:
        FileNotFoundError: If labels_path does not exist.
        ValueError: If required columns are missing, dataset is empty,
                    or label values are invalid.
    """
    labels_file = Path(labels_path)
    output_path = Path(output_dir)

    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    df = pd.read_csv(labels_file)

    # Validate non-empty
    if df.empty:
        raise ValueError("Input labels file is empty; cannot create splits")

    # Validate required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Labels CSV missing required columns: {sorted(missing)}")

    # Validate unique item_id
    if df["item_id"].duplicated().any():
        dupes = df.loc[df["item_id"].duplicated(), "item_id"].tolist()
        raise ValueError(f"Input contains duplicate item_ids: {dupes[:5]}")

    # Validate binary label
    numeric_labels = pd.to_numeric(df["label"], errors="coerce")
    if numeric_labels.isna().any() or not set(numeric_labels.unique()).issubset({0, 1}):
        raise ValueError("Column 'label' must contain only 0 and 1 values")

    df["label"] = numeric_labels.astype(int)

    n_total = len(df)
    n_train, n_dev, n_test = compute_split_counts(n_total)

    rng = np.random.default_rng(seed)

    # Stratified allocation by class
    # Group indices by label
    classes = sorted(df["label"].unique())
    can_stratify = len(classes) > 1 and all((df["label"] == c).sum() >= 3 for c in classes)

    strategy = "stratified_exact_counts" if can_stratify else "random_exact_counts"
    logger.info("Splitting %d rows with strategy '%s' (seed=%d)", n_total, strategy, seed)

    train_indices: list[int] = []
    dev_indices: list[int] = []
    test_indices: list[int] = []

    if can_stratify:
        # Determine target counts per class for train and dev
        class_indices = {}
        for c in classes:
            c_idx = df.index[df["label"] == c].to_numpy()
            rng.shuffle(c_idx)
            class_indices[c] = c_idx

        # Allocate train counts per class proportional to class size
        # To maintain exact total n_train:
        class_train_counts = {}
        allocated_train = 0
        for i, c in enumerate(classes):
            c_len = len(class_indices[c])
            if i == len(classes) - 1:
                class_train_counts[c] = n_train - allocated_train
            else:
                c_n = int(round(n_train * (c_len / n_total)))
                class_train_counts[c] = c_n
                allocated_train += c_n

        # Allocate dev counts per class proportional to class size
        class_dev_counts = {}
        allocated_dev = 0
        for i, c in enumerate(classes):
            c_len = len(class_indices[c])
            if i == len(classes) - 1:
                class_dev_counts[c] = n_dev - allocated_dev
            else:
                c_n = int(round(n_dev * (c_len / n_total)))
                class_dev_counts[c] = c_n
                allocated_dev += c_n

        # Slice class indices into splits
        for c in classes:
            c_idx = class_indices[c]
            n_tr = class_train_counts[c]
            n_dv = class_dev_counts[c]

            train_indices.extend(c_idx[:n_tr].tolist())
            dev_indices.extend(c_idx[n_tr : n_tr + n_dv].tolist())
            test_indices.extend(c_idx[n_tr + n_dv :].tolist())

    else:
        # Fallback to random shuffle if dataset is single-class or too small
        all_indices = df.index.to_numpy()
        rng.shuffle(all_indices)
        train_indices = all_indices[:n_train].tolist()
        dev_indices = all_indices[n_train : n_train + n_dev].tolist()
        test_indices = all_indices[n_train + n_dev :].tolist()

    # Verify count invariants before constructing final DataFrames
    assert len(train_indices) == n_train, f"Train count mismatch: {len(train_indices)} != {n_train}"
    assert len(dev_indices) == n_dev, f"Dev count mismatch: {len(dev_indices)} != {n_dev}"
    assert len(test_indices) == n_test, f"Test count mismatch: {len(test_indices)} != {n_test}"

    # Verify no index overlap
    set_train = set(train_indices)
    set_dev = set(dev_indices)
    set_test = set(test_indices)
    assert len(set_train & set_dev) == 0, "Overlap between train and dev indices"
    assert len(set_train & set_test) == 0, "Overlap between train and test indices"
    assert len(set_dev & set_test) == 0, "Overlap between dev and test indices"
    assert len(set_train | set_dev | set_test) == n_total, "Coverage incomplete"

    # Construct split DataFrames and sort by original index or shuffle deterministically
    train_df = df.iloc[train_indices].copy()
    dev_df = df.iloc[dev_indices].copy()
    test_df = df.iloc[test_indices].copy()

    # Reorder columns to match REQUIRED_COLUMNS standard
    train_df = train_df[REQUIRED_COLUMNS].reset_index(drop=True)
    dev_df = dev_df[REQUIRED_COLUMNS].reset_index(drop=True)
    test_df = test_df[REQUIRED_COLUMNS].reset_index(drop=True)

    # Write splits to output directory
    output_path.mkdir(parents=True, exist_ok=True)
    train_file = output_path / "train.csv"
    dev_file = output_path / "dev.csv"
    test_file = output_path / "test.csv"

    train_df.to_csv(train_file, index=False)
    dev_df.to_csv(dev_file, index=False)
    test_df.to_csv(test_file, index=False)

    logger.info(
        "Splits successfully written: train=%d, dev=%d, test=%d (total=%d)",
        len(train_df),
        len(dev_df),
        len(test_df),
        n_total,
    )

    return {
        "train": train_df,
        "dev": dev_df,
        "test": test_df,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="split_data.py",
        description="Deterministically split labels.csv into Train/Dev/Test.",
    )
    parser.add_argument(
        "--input",
        default="data/labels.csv",
        help="Path to the annotated labels CSV (default: data/labels.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits/",
        help="Directory to write train.csv, dev.csv, test.csv (default: data/splits/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("RANDOM_SEED", "42")),
        help="Random seed (default: RANDOM_SEED env var or 42)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for split_data.py."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    splits = split_labels(
        labels_path=args.input,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("  DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"  Input:       {args.input}")
    print(f"  Output Dir:  {args.output_dir}")
    print(f"  Random Seed: {args.seed}")
    print(f"  Strategy:    stratified_exact_counts")
    print("-" * 60)

    total_rows = sum(len(df) for df in splits.values())
    for name, df in splits.items():
        n_pass = int((df["label"] == 1).sum())
        n_fail = int((df["label"] == 0).sum())
        pass_pct = (n_pass / len(df) * 100) if len(df) > 0 else 0.0
        fail_pct = (n_fail / len(df) * 100) if len(df) > 0 else 0.0
        pct_of_total = (len(df) / total_rows * 100) if total_rows > 0 else 0.0
        print(
            f"  {name.upper():<5}: {len(df):>3} items ({pct_of_total:>4.1f}%) | "
            f"Pass: {n_pass:>3} ({pass_pct:>4.1f}%) | "
            f"Fail: {n_fail:>3} ({fail_pct:>4.1f}%)"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
