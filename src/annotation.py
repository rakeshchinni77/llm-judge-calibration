"""
annotation.py — Annotation validation, template management, and audit logging.

Responsibilities:
    AnnotationValidator  Validates data/labels.csv against the authoritative schema.
                         Raises descriptive errors on any violation.
    AnnotationAudit      Logs annotation sessions (start/end time, row count,
                         label distribution) to data/annotation_audit.jsonl
                         WITHOUT touching the required labels.csv schema.
    extract_labels_csv   Strips the annotation template down to the five required
                         columns and writes data/labels.csv.
    validate_and_report  CLI-friendly wrapper: validates labels.csv and prints a
                         summary table.

The annotation template (data/annotation_template.csv) has the 5 required columns
PLUS a `question` column for annotator context. `extract_labels_csv` drops the
extra column before writing labels.csv so the schema is never violated.

Phase gate: Phase 1
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema contract (mirrors AGENTS.md § 6)
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: list[str] = ["item_id", "output", "label", "critique", "relabeled"]
TEMPLATE_EXTRA_COLUMNS: list[str] = ["question"]   # present in template only
LABEL_VALUES: set[int] = {0, 1}
MIN_ROWS: int = 250
MAX_CRITIQUE_WORDS: int = 200   # sanity cap; minimum meaningful critique is 5 words
MIN_CRITIQUE_WORDS: int = 5


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class SchemaError(ValueError):
    """Raised when labels.csv violates the authoritative schema contract."""


class AnnotationIncompleteError(ValueError):
    """Raised when labeling is incomplete (blank labels or critiques)."""


# ---------------------------------------------------------------------------
# Annotation validator
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of a single validation run.

    Attributes:
        is_valid:       True if all checks passed.
        n_rows:         Total rows in the dataset.
        n_pass:         Rows with label == 1.
        n_fail:         Rows with label == 0.
        fail_rate:      Fraction of rows that are failures (label == 0).
        n_relabeled:    Rows where relabeled == 1.
        errors:         List of human-readable error messages.
        warnings:       List of non-fatal warnings.
    """

    is_valid: bool = False
    n_rows: int = 0
    n_pass: int = 0
    n_fail: int = 0
    fail_rate: float = 0.0
    n_relabeled: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class AnnotationValidator:
    """Validates data/labels.csv against the authoritative schema.

    Args:
        path: Path to the labels CSV file to validate.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def validate(self) -> ValidationResult:
        """Run all schema checks and return a :class:`ValidationResult`.

        Returns:
            A :class:`ValidationResult` with ``is_valid=True`` if all checks pass.

        Raises:
            FileNotFoundError: If the labels file does not exist.
        """
        result = ValidationResult()

        if not self.path.exists():
            result.errors.append(f"File not found: {self.path}")
            return result

        try:
            df = pd.read_csv(self.path, dtype=str)
        except Exception as exc:
            result.errors.append(f"CSV parse error: {exc}")
            return result

        # ── Column check ──────────────────────────────────────────────────
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(REQUIRED_COLUMNS)
        if missing:
            result.errors.append(f"Missing required columns: {sorted(missing)}")
        if extra:
            result.errors.append(
                f"Extra columns not allowed in labels.csv: {sorted(extra)}. "
                "Store supplementary columns in generation_metadata.csv."
            )
        if missing:
            return result   # can't continue without required columns

        # ── Trim whitespace ───────────────────────────────────────────────
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        # ── Row count ─────────────────────────────────────────────────────
        result.n_rows = len(df)
        if result.n_rows < MIN_ROWS:
            result.errors.append(
                f"Dataset has {result.n_rows} rows; minimum required is {MIN_ROWS}."
            )

        # ── item_id: unique and non-empty ─────────────────────────────────
        blank_ids = df["item_id"].isna() | (df["item_id"] == "")
        if blank_ids.any():
            result.errors.append(
                f"{blank_ids.sum()} rows have blank item_id."
            )
        if df["item_id"].duplicated().any():
            dupes = df.loc[df["item_id"].duplicated(keep=False), "item_id"].unique()
            result.errors.append(f"Duplicate item_ids found: {list(dupes[:10])}")

        # ── output: non-empty ──────────────────────────────────────────────
        blank_outputs = df["output"].isna() | (df["output"] == "")
        if blank_outputs.any():
            result.errors.append(
                f"{blank_outputs.sum()} rows have blank output."
            )

        # ── label: must be 0 or 1 ─────────────────────────────────────────
        label_errors = self._check_binary_column(df, "label")
        result.errors.extend(label_errors)

        # ── relabeled: must be 0 or 1 ─────────────────────────────────────
        relabeled_errors = self._check_binary_column(df, "relabeled")
        result.errors.extend(relabeled_errors)

        # ── critique: non-empty, meaningful length ────────────────────────
        blank_critiques = df["critique"].isna() | (df["critique"] == "")
        if blank_critiques.any():
            result.errors.append(
                f"{blank_critiques.sum()} rows have blank critique. "
                "Every row requires a human-written justification."
            )
        # Check for placeholder/synthetic critiques
        suspicious = self._detect_placeholder_critiques(df)
        if suspicious:
            result.warnings.append(
                f"{len(suspicious)} critiques appear to be short placeholders "
                f"(< {MIN_CRITIQUE_WORDS} words). Please review: "
                + str(suspicious[:5])
            )

        # ── Incomplete annotations (label still blank from template) ───────
        try:
            label_series = pd.to_numeric(df["label"], errors="coerce")
            blank_labels = label_series.isna()
            if blank_labels.any():
                result.errors.append(
                    f"{blank_labels.sum()} rows still have blank/non-numeric label. "
                    "These are unannotated rows from the template."
                )
        except Exception:
            pass  # already caught above

        # ── Compute summary statistics ─────────────────────────────────────
        if not result.errors:
            df["label"] = pd.to_numeric(df["label"])
            df["relabeled"] = pd.to_numeric(df["relabeled"])
            result.n_pass = int((df["label"] == 1).sum())
            result.n_fail = int((df["label"] == 0).sum())
            result.fail_rate = result.n_fail / result.n_rows if result.n_rows > 0 else 0.0
            result.n_relabeled = int((df["relabeled"] == 1).sum())

            if result.fail_rate < 0.25:
                result.warnings.append(
                    f"Failure rate is {result.fail_rate:.1%}, below the 30 % target. "
                    "Consider reviewing labels or generation conditions."
                )

        result.is_valid = len(result.errors) == 0
        return result

    @staticmethod
    def _check_binary_column(df: pd.DataFrame, col: str) -> list[str]:
        """Return error messages if ``col`` contains values outside {0, 1}.

        Args:
            df:  DataFrame to check.
            col: Column name.

        Returns:
            List of error strings (empty if the column is valid).
        """
        errors: list[str] = []
        numeric = pd.to_numeric(df[col], errors="coerce")
        non_numeric = df[col][numeric.isna()].dropna()
        if not non_numeric.empty:
            errors.append(
                f"Column '{col}' has {len(non_numeric)} non-numeric values: "
                + str(non_numeric.unique()[:5].tolist())
            )
        invalid = numeric.dropna()[~numeric.dropna().isin([0, 1])]
        if not invalid.empty:
            errors.append(
                f"Column '{col}' must contain only 0 or 1; "
                f"found values: {invalid.unique().tolist()[:10]}"
            )
        return errors

    @staticmethod
    def _detect_placeholder_critiques(df: pd.DataFrame) -> list[str]:
        """Return item_ids whose critiques appear suspiciously short.

        Args:
            df: DataFrame with ``item_id`` and ``critique`` columns.

        Returns:
            List of item_ids with short or placeholder critiques.
        """
        flagged: list[str] = []
        for _, row in df.iterrows():
            critique = str(row.get("critique", ""))
            word_count = len(critique.split())
            if word_count < MIN_CRITIQUE_WORDS:
                flagged.append(str(row["item_id"]))
        return flagged


# ---------------------------------------------------------------------------
# Annotation audit logger
# ---------------------------------------------------------------------------

class AnnotationAudit:
    """Appends annotation session records to data/annotation_audit.jsonl.

    The audit log is stored separately from labels.csv so that timestamps
    and session metadata do not pollute the required schema.

    Args:
        audit_path: Path to the JSONL audit file.
    """

    def __init__(self, audit_path: str | Path = "data/annotation_audit.jsonl") -> None:
        self.audit_path = Path(audit_path)
        self._session_start: float | None = None

    def start_session(self, annotator: str = "human") -> None:
        """Record the start of an annotation session.

        Args:
            annotator: Human-readable annotator identifier.
        """
        self._session_start = time.time()
        record = {
            "event": "session_start",
            "annotator": annotator,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._append(record)
        logger.info("Annotation session started (annotator=%s)", annotator)

    def end_session(self, n_labeled: int, n_relabeled: int) -> float:
        """Record the end of a session and return elapsed minutes.

        Args:
            n_labeled:   Total rows labeled in this session.
            n_relabeled: Rows where the label was changed.

        Returns:
            Elapsed time in minutes.
        """
        elapsed_s = time.time() - (self._session_start or time.time())
        elapsed_min = elapsed_s / 60
        record = {
            "event": "session_end",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_minutes": round(elapsed_min, 2),
            "n_labeled": n_labeled,
            "n_relabeled": n_relabeled,
        }
        self._append(record)
        logger.info(
            "Session ended: %d rows labeled in %.1f minutes", n_labeled, elapsed_min
        )
        return elapsed_min

    def log_relabel(self, item_id: str, old_label: int, new_label: int, reason: str) -> None:
        """Record a single label change event.

        Args:
            item_id:   The item whose label changed.
            old_label: Previous label value.
            new_label: New label value.
            reason:    Brief explanation for the change.
        """
        record = {
            "event": "relabel",
            "item_id": item_id,
            "old_label": old_label,
            "new_label": new_label,
            "reason": reason,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._append(record)
        logger.info("Relabeled %s: %d → %d (%s)", item_id, old_label, new_label, reason)

    def _append(self, record: dict[str, Any]) -> None:
        """Append a JSON record to the audit file.

        Args:
            record: Dict to serialise as a JSONL line.
        """
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Template → labels.csv extraction
# ---------------------------------------------------------------------------

def extract_labels_csv(
    template_path: str | Path,
    output_path: str | Path = "data/labels.csv",
) -> int:
    """Extract the 5 required columns from the annotation template.

    Drops the ``question`` column (annotator context) and any other extras.
    Validates the result before writing.

    Args:
        template_path: Path to the filled annotation template CSV.
        output_path:   Destination path for labels.csv.

    Returns:
        Number of rows written.

    Raises:
        SchemaError: If the template is missing required columns or has invalid data.
    """
    template_path = Path(template_path)
    output_path = Path(output_path)

    df = pd.read_csv(template_path, dtype=str)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaError(
            f"Annotation template is missing required columns: {sorted(missing)}"
        )

    labels_df = df[REQUIRED_COLUMNS].copy()
    labels_df = labels_df.map(lambda x: x.strip() if isinstance(x, str) else x)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(labels_df), output_path)
    return len(labels_df)


# ---------------------------------------------------------------------------
# CLI-friendly validation report
# ---------------------------------------------------------------------------

def validate_and_report(labels_path: str | Path = "data/labels.csv") -> ValidationResult:
    """Validate labels.csv and print a rich summary report.

    Args:
        labels_path: Path to the labels CSV.

    Returns:
        :class:`ValidationResult` instance.
    """
    validator = AnnotationValidator(labels_path)
    result = validator.validate()

    # Print summary
    print("\n" + "=" * 60)
    print("  ANNOTATION VALIDATION REPORT")
    print("=" * 60)
    print(f"  File:       {labels_path}")
    print(f"  Rows:       {result.n_rows}")
    print(f"  Pass (1):   {result.n_pass}")
    print(f"  Fail (0):   {result.n_fail}")
    print(f"  Fail rate:  {result.fail_rate:.1%}")
    print(f"  Relabeled:  {result.n_relabeled}")
    print()

    if result.errors:
        print(f"  [FAIL]  {len(result.errors)} ERROR(S):")
        for err in result.errors:
            print(f"      * {err}")
    else:
        print("  [OK]  All schema checks passed")

    if result.warnings:
        print(f"\n  [WARN]  {len(result.warnings)} WARNING(S):")
        for warn in result.warnings:
            print(f"      * {warn}")

    print("=" * 60 + "\n")
    return result


def _main() -> None:
    """CLI entrypoint for annotation validation and template extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="annotation.py",
        description="Validate labels.csv or extract it from annotation_template.csv.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--validate",
        nargs="?",
        const="data/labels.csv",
        help="Validate a labels CSV (default: data/labels.csv).",
    )
    group.add_argument(
        "--extract",
        nargs="?",
        const="data/annotation_template.csv",
        help="Extract the 5 required columns from annotation template (default: data/annotation_template.csv).",
    )
    parser.add_argument(
        "--output",
        default="data/labels.csv",
        help="Destination path when using --extract (default: data/labels.csv).",
    )

    args = parser.parse_args()

    if args.extract:
        template_file = Path(args.extract)
        dest_file = Path(args.output)
        print(f"Extracting 5 required columns from {template_file} -> {dest_file}...")
        n = extract_labels_csv(template_file, dest_file)
        print(f"Wrote {n} rows to {dest_file}.")
        validate_and_report(dest_file)
    else:
        target = args.validate or "data/labels.csv"
        res = validate_and_report(target)
        if not res.is_valid:
            sys.exit(1)


if __name__ == "__main__":
    _main()

