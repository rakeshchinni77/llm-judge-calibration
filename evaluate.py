"""
evaluate.py — Main CLI entrypoint for the LLM-as-a-Judge Calibration Pipeline.

Commands:
    python evaluate.py --help
    python evaluate.py --test-judge "Sample text to evaluate"
    python evaluate.py --split dev
    python evaluate.py --split test

Description:
    Evaluates generated text outputs against rubric criteria using an LLM judge.
    Supports:
        1. Single-string smoke testing (--test-judge).
        2. Development evaluation for prompt iteration (--split dev).
        3. Official frozen-set evaluation (--split test).
    
    Guarantees:
        - Leakage Guard: Few-shot demonstrations are loaded EXCLUSIVELY from
          the Train split (data/splits/train.csv). Dev and Test examples are
          strictly blocked.
        - Provenance: Prompts and configurations are hashed and versioned.
        - Robustness: Failed evaluations are explicitly tracked; no items are
          silently dropped from the denominator.
        - Statistical Rigor: 1,000 bootstrap resamples with 95% confidence intervals.

Outputs:
    results/metrics.csv       — TPR, TNR, agreement, Cohen's Kappa + 95% CIs per split.
    results/judge_test.csv    — Per-item judge predictions on the frozen test split.
    results/run_manifest_*.json— Run provenance and versioning metadata.

Phase gate: Phase 7
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import numpy as np
import pandas as pd

# Load .env into os.environ before any other operations
load_dotenv()

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.judge import CredentialsException, LLMJudge, load_train_few_shot
from src.metrics import compute_all_metrics

logger = logging.getLogger("evaluate")

METRICS_COLUMNS: list[str] = [
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

TEST_PREDICTION_COLUMNS: list[str] = [
    "item_id",
    "human_label",
    "judge_label",
    "judge_critique",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_prompt_version(prompt_path: str | Path = "prompts/judge_system.txt") -> tuple[str, str]:
    """Calculate the hash and version tag of the system prompt.

    Args:
        prompt_path: Path to prompt file.

    Returns:
        Tuple of (version_tag, prompt_hash).
    """
    path = Path(prompt_path)
    if not path.exists():
        path = Path("prompts/system_prompt.md")

    if path.exists():
        content = path.read_text(encoding="utf-8").strip()
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:10]
        return f"v1.0-{digest}", digest
    return "v1.0-default", "default"


def update_metrics_csv(
    metrics_row: dict[str, Any],
    output_path: str | Path = "results/metrics.csv",
) -> None:
    """Update or append split metrics row to results/metrics.csv.

    Preserves exact required columns and updates existing split row if present.

    Args:
        metrics_row: Dictionary matching METRICS_COLUMNS.
        output_path: Destination CSV path.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and out_file.stat().st_size > 0:
        try:
            df = pd.read_csv(out_file)
            if not df.empty and "split" in df.columns:
                # Remove any existing row for this split
                df = df[df["split"] != metrics_row["split"]].dropna(how="all")
                new_df = pd.concat([df, pd.DataFrame([metrics_row])], ignore_index=True) if not df.empty else pd.DataFrame([metrics_row])
            else:
                new_df = pd.DataFrame([metrics_row])
        except Exception:
            new_df = pd.DataFrame([metrics_row])
    else:
        new_df = pd.DataFrame([metrics_row])

    # Reorder columns to guarantee authoritative schema
    new_df = new_df[METRICS_COLUMNS]
    new_df.to_csv(out_file, index=False)
    logger.info("Updated %s with metrics for split '%s'", out_file, metrics_row["split"])


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def handle_test_judge(text: str, backend: str | None = None) -> None:
    """Smoke-test the judge on a single string and print the JSON result.

    This function is the target of the ``--test-judge`` CLI flag.
    Outputs ONLY valid JSON to stdout; diagnostic messages go to stderr.

    Args:
        text: The generated text to pass to the judge.
        backend: Execution backend ('mock', 'hosted', 'local').
    """
    if not backend:
        backend = os.getenv("MODEL_TYPE")
        if not backend:
            backend = "hosted" if os.getenv("OPENAI_API_KEY") else "mock"

    try:
        judge = LLMJudge(backend=backend)
        result = judge.evaluate(text)
        # Output ONLY parseable JSON to stdout
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        sys.stdout.flush()
    except CredentialsException as exc:
        sys.stderr.write(f"Credentials Error: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write(f"Evaluation Error: {exc}\n")
        sys.exit(1)


def handle_split(
    split: str,
    seed: int = 42,
    backend: str | None = None,
    train_path: str | Path = "data/splits/train.csv",
    output_dir: str | Path = "results",
) -> dict[str, Any]:
    """Run full evaluation on the specified split (dev or test).

    Steps:
        1. Load split CSV (data/splits/{split}.csv).
        2. Load clean few-shot demonstrations strictly from train.csv.
        3. Evaluate each output with the judge.
        4. Track human vs judge labels; handle any partial errors explicitly.
        5. Compute bootstrap metrics (TPR, TNR, agreement, Cohen's Kappa).
        6. Update results/metrics.csv.
        7. If split=='test', write results/judge_test.csv.
        8. Write run manifest for provenance.

    Args:
        split: One of 'dev' or 'test'.
        seed: Random seed for judge and bootstrap resampling.
        backend: Judge execution backend ('mock', 'hosted', 'local').
        train_path: Path to clean train split for few-shot demonstrations.
        output_dir: Destination results directory.

    Returns:
        Dict of computed metrics for this split.
    """
    split_file = Path(f"data/splits/{split}.csv")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_split = pd.read_csv(split_file)
    n_items = len(df_split)
    logger.info("Loaded %d items from %s", n_items, split_file)

    if split == "dev" and n_items != 100:
        logger.warning("Expected 100 items in dev split, found %d", n_items)
    elif split == "test" and n_items != 112:
        logger.warning("Expected 112 items in test split, found %d", n_items)

    # Resolve execution backend
    if not backend:
        backend = os.getenv("MODEL_TYPE")
        if not backend:
            backend = "hosted" if os.getenv("OPENAI_API_KEY") else "mock"

    prompt_version, prompt_hash = get_prompt_version()
    logger.info("Evaluating split '%s' | Model Backend: %s | Prompt: %s", split, backend, prompt_version)

    # Initialize judge
    judge = LLMJudge(backend=backend, seed=seed)

    # LEAKAGE GUARD: Load demonstrations strictly from Train split
    logger.info("Loading few-shot demonstrations exclusively from %s...", train_path)
    few_shot_demos = load_train_few_shot(train_path=train_path, n_examples=4, seed=seed)

    # Evaluation loop
    prediction_records: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    failures_count: int = 0

    print(f"\nEvaluating {n_items} items on split: '{split}'...")
    for idx, row in df_split.iterrows():
        item_id = str(row["item_id"])
        human_label = int(row["label"])
        text = str(row["output"])

        try:
            res = judge.evaluate(text, few_shot_examples=few_shot_demos)
            passed = bool(res.get("pass", False))
            judge_label = 1 if passed else 0
            critique = str(res.get("critique", "Evaluated against rubric criteria."))
        except Exception as exc:
            failures_count += 1
            logger.warning("Evaluation failed for item %s: %s", item_id, exc)
            # Do NOT silently drop failed evaluations from denominator!
            judge_label = 0
            critique = f"Evaluation failed: {exc}"

        y_true.append(human_label)
        y_pred.append(judge_label)

        prediction_records.append({
            "item_id": item_id,
            "human_label": human_label,
            "judge_label": judge_label,
            "judge_critique": critique,
        })

    # Compute statistical metrics with 1,000 bootstrap iterations
    logger.info("Computing bootstrap metrics with 95%% CIs (seed=%d)...", seed)
    metrics_row = compute_all_metrics(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        split_name=split,
        seed=seed,
    )

    # Persist updated results/metrics.csv
    update_metrics_csv(metrics_row, output_path=out_dir / "metrics.csv")

    # If test split, write results/judge_test.csv
    if split == "test":
        pred_df = pd.DataFrame(prediction_records)[TEST_PREDICTION_COLUMNS]
        test_pred_path = out_dir / "judge_test.csv"
        pred_df.to_csv(test_pred_path, index=False)
        logger.info("Wrote %d frozen test predictions to %s", len(pred_df), test_pred_path)

    # Write run manifest for auditability and provenance
    manifest = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "split": split,
        "sample_size": n_items,
        "model_name": judge.model_name,
        "backend": judge.backend,
        "seed": seed,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "failures_count": failures_count,
        "judge_spend_usd": round(judge.spend, 6),
        "metrics": metrics_row,
    }
    manifest_path = out_dir / f"run_manifest_{split}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Saved run manifest to %s", manifest_path)

    # Formatted console summary
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS — SPLIT: {split.upper()}")
    print("=" * 70)
    print(f"  Items Evaluated:   {n_items} (Failed: {failures_count})")
    print(f"  Model & Backend:   {judge.model_name} ({judge.backend})")
    print(f"  Prompt Version:    {prompt_version}")
    print(f"  Judge Spend:       ${judge.spend:.4f}")
    print("-" * 70)
    print(f"  TPR (Fail Recall): {metrics_row['tpr_mean']:.4f}  [95% CI: {metrics_row['tpr_ci_low']:.4f} – {metrics_row['tpr_ci_high']:.4f}]")
    print(f"  TNR (Pass Spec):   {metrics_row['tnr_mean']:.4f}  [95% CI: {metrics_row['tnr_ci_low']:.4f} – {metrics_row['tnr_ci_high']:.4f}]")
    print(f"  Raw Agreement:     {metrics_row['agreement']:.4f}")
    print(f"  Cohen's Kappa:     {metrics_row['kappa_mean']:.4f}  [95% CI: {metrics_row['kappa_ci_low']:.4f} – {metrics_row['kappa_ci_high']:.4f}]")
    print("=" * 70)
    print(f"Metrics saved to:   {out_dir / 'metrics.csv'}")
    if split == "test":
        print(f"Predictions saved to: {out_dir / 'judge_test.csv'}")
    print(f"Manifest saved to:    {manifest_path}\n")

    return metrics_row


# ---------------------------------------------------------------------------
# Argument parsing & Main entrypoint
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and configure the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="LLM-as-a-Judge Calibration Pipeline — Evaluation CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate.py --help
    python evaluate.py --test-judge "Technical explanation of BFS vs DFS..."
    python evaluate.py --split dev
    python evaluate.py --split test
        """,
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        help="Evaluate the judge against the dev or test split.",
    )
    parser.add_argument(
        "--test-judge",
        metavar="TEXT",
        help="Smoke-test the judge on a single string. Prints JSON {critique, pass} and exits.",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "hosted", "local"],
        help="Judge execution backend (default: MODEL_TYPE env or mock).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("RANDOM_SEED", "42")),
        help="Random seed for judge and bootstrap CIs (default: RANDOM_SEED env or 42).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main() -> None:
    """Parse arguments and dispatch to the appropriate handler."""
    parser = _build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.test_judge:
        handle_test_judge(args.test_judge, backend=args.backend)
        sys.exit(0)

    if args.split:
        handle_split(split=args.split, seed=args.seed, backend=args.backend)
        sys.exit(0)

    # No flags provided — print help and exit
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
