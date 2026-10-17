"""
induce_leakage.py — Induced Data Leakage Demonstration.

Intentionally injects 10 Dev-split examples into the few-shot demonstrations,
re-evaluates the Dev split, and records the artificially inflated metrics in
results/induced_failure.json.

This experiment proves the danger of data contamination:
When evaluation examples leak into prompt demonstrations, the judge memorizes
the correct answers for those items, artificially boosting perceived performance
(Cohen's Kappa) while providing zero evidence of true out-of-distribution generalization.

Security & Integrity Guarantee:
    - The contamination logic is completely isolated to this script.
    - Normal evaluation (evaluate.py) uses load_train_few_shot(), which strictly
      rejects dev/test splits with a ValueError.
    - The frozen test split is NEVER touched by this experiment.

Outputs:
    results/induced_failure.json — {"baseline_dev_kappa": float,
                                    "contaminated_dev_kappa": float,
                                    "leaked_items_count": 10}
    results/leakage_audit.json   — audit details of leaked items, seed, and deltas.

Phase gate: Phase 6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.judge import LLMJudge, load_train_few_shot
from src.metrics import compute_kappa

logger = logging.getLogger(__name__)

LEAKED_ITEMS_COUNT: int = 10
OUTPUT_PATH: Path = Path("results/induced_failure.json")
AUDIT_PATH: Path = Path("results/leakage_audit.json")


def evaluate_split_predictions(
    df: pd.DataFrame,
    judge: LLMJudge,
    few_shot_examples: list[dict[str, Any]],
) -> list[int]:
    """Evaluate all items in a DataFrame and return integer predictions (0 or 1).

    Args:
        df: DataFrame containing at least an 'output' column.
        judge: Instantiated LLMJudge.
        few_shot_examples: List of prompt demonstrations.

    Returns:
        List of integer predictions where 1 is Pass and 0 is Fail.
    """
    predictions: list[int] = []
    for _, row in df.iterrows():
        text = str(row["output"])
        res = judge.evaluate(text, few_shot_examples=few_shot_examples)
        passed = bool(res.get("pass", False))
        predictions.append(1 if passed else 0)
    return predictions


def run_leakage_experiment(
    train_path: str | Path = "data/splits/train.csv",
    dev_path: str | Path = "data/splits/dev.csv",
    n_leaked: int = LEAKED_ITEMS_COUNT,
    seed: int = 42,
    backend: str = "mock",
) -> dict[str, Any]:
    """Contaminate the few-shot prompt with dev examples and measure performance shift.

    Steps:
        1. Load the clean Train and Dev splits.
        2. Evaluate Dev split with clean Train demonstrations only -> baseline_dev_kappa.
        3. Deterministically sample n_leaked items from the Dev split.
        4. Construct contaminated few-shot prompt (Train + Leaked Dev items).
        5. Re-evaluate Dev split with contaminated prompt -> contaminated_dev_kappa.
        6. Persist results/induced_failure.json and detailed audit records.

    Args:
        train_path: Path to clean train split CSV.
        dev_path: Path to dev split CSV.
        n_leaked: Number of dev items to intentionally leak (default 10).
        seed: Random seed for sampling leaked items and judge determinism.
        backend: Judge execution backend ('mock', 'hosted', or 'local').

    Returns:
        Dict matching results/induced_failure.json schema.

    Raises:
        FileNotFoundError: If train or dev split CSVs are missing.
        ValueError: If dev split has fewer than n_leaked items.
    """
    train_file = Path(train_path)
    dev_file = Path(dev_path)

    if not train_file.exists():
        raise FileNotFoundError(f"Clean train split not found: {train_file}")
    if not dev_file.exists():
        raise FileNotFoundError(f"Dev split not found: {dev_file}")

    train_df = pd.read_csv(train_file)
    dev_df = pd.read_csv(dev_file)

    if len(dev_df) < n_leaked:
        raise ValueError(
            f"Dev split has only {len(dev_df)} items, cannot leak {n_leaked} items."
        )

    y_true = dev_df["label"].to_numpy(dtype=int)

    logger.info("Instantiating LLMJudge with backend='%s', seed=%d...", backend, seed)
    judge = LLMJudge(backend=backend, seed=seed)

    # 1. Clean Baseline Run: approved train few-shot demonstrations only
    logger.info("Loading clean few-shot demonstrations exclusively from %s...", train_file)
    clean_few_shot = load_train_few_shot(train_path=train_file, n_examples=4, seed=seed)

    logger.info("Evaluating clean baseline on %d Dev items...", len(dev_df))
    clean_preds = evaluate_split_predictions(dev_df, judge, few_shot_examples=clean_few_shot)
    baseline_kappa = round(float(compute_kappa(y_true, np.array(clean_preds))), 4)
    clean_spend = judge.spend

    # 2. Controlled Contamination: Sample exactly n_leaked Dev items
    logger.info("Sampling %d Dev items to inject into contaminated few-shot prompt...", n_leaked)
    rng = np.random.default_rng(seed)
    leaked_indices = rng.choice(len(dev_df), size=n_leaked, replace=False)
    leaked_df = dev_df.iloc[leaked_indices]

    leaked_items_audit: list[dict[str, Any]] = []
    leaked_demonstrations: list[dict[str, Any]] = []
    for _, row in leaked_df.iterrows():
        item_id = str(row["item_id"])
        output_text = str(row["output"])
        label_val = int(row["label"])
        critique_text = str(row["critique"])

        leaked_items_audit.append({
            "item_id": item_id,
            "label": label_val,
            "critique": critique_text,
            "output_preview": output_text[:80] + ("..." if len(output_text) > 80 else ""),
        })
        leaked_demonstrations.append({
            "output": output_text,
            "label": label_val,
            "critique": critique_text,
        })

    # Assemble contaminated prompt (Train + Leaked Dev)
    contaminated_few_shot = list(clean_few_shot) + leaked_demonstrations

    # 3. Contaminated Run: Re-evaluate Dev items
    logger.info("Re-evaluating %d Dev items with contaminated few-shot prompt...", len(dev_df))
    contam_preds = evaluate_split_predictions(
        dev_df, judge, few_shot_examples=contaminated_few_shot
    )
    contaminated_kappa = round(float(compute_kappa(y_true, np.array(contam_preds))), 4)
    total_spend = judge.spend

    # Detailed audit metrics
    leaked_ids_set = set(leaked_df["item_id"])
    leaked_mask = dev_df["item_id"].isin(leaked_ids_set).to_numpy()
    non_leaked_mask = ~leaked_mask

    clean_leaked_acc = float(np.mean(np.array(clean_preds)[leaked_mask] == y_true[leaked_mask]))
    contam_leaked_acc = float(
        np.mean(np.array(contam_preds)[leaked_mask] == y_true[leaked_mask])
    )

    clean_non_leaked_acc = float(
        np.mean(np.array(clean_preds)[non_leaked_mask] == y_true[non_leaked_mask])
    )
    contam_non_leaked_acc = float(
        np.mean(np.array(contam_preds)[non_leaked_mask] == y_true[non_leaked_mask])
    )

    kappa_delta = round(contaminated_kappa - baseline_kappa, 4)

    logger.info("Baseline Dev Kappa:     %.4f", baseline_kappa)
    logger.info("Contaminated Dev Kappa: %.4f", contaminated_kappa)
    logger.info("Kappa delta (inflation): %+.4f", kappa_delta)
    logger.info(
        "Accuracy on leaked items: baseline=%.1f%% -> contaminated=%.1f%%",
        clean_leaked_acc * 100,
        contam_leaked_acc * 100,
    )

    # Required JSON artifact structure (Section 6, AGENTS.md)
    result_data = {
        "baseline_dev_kappa": baseline_kappa,
        "contaminated_dev_kappa": contaminated_kappa,
        "leaked_items_count": n_leaked,
    }

    # Write primary required JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    logger.info("Saved required metrics to %s", OUTPUT_PATH)

    # Write separate detailed audit artifact (Requirement 7)
    audit_data = {
        "experiment_name": "induced_dev_leakage_demonstration",
        "random_seed": seed,
        "backend": backend,
        "baseline_dev_kappa": baseline_kappa,
        "contaminated_dev_kappa": contaminated_kappa,
        "kappa_delta": kappa_delta,
        "leaked_items_count": n_leaked,
        "accuracy_shift": {
            "leaked_items_accuracy_baseline": clean_leaked_acc,
            "leaked_items_accuracy_contaminated": contam_leaked_acc,
            "non_leaked_items_accuracy_baseline": clean_non_leaked_acc,
            "non_leaked_items_accuracy_contaminated": contam_non_leaked_acc,
        },
        "spend_usd": {
            "clean_run_spend": round(clean_spend, 6),
            "total_spend": round(total_spend, 6),
        },
        "leaked_items": leaked_items_audit,
        "scientific_justification": (
            "Contaminated Dev metrics are invalid as an unbiased estimate of generalization "
            "because the judge was provided direct, ground-truth demonstrations of the exact test "
            "items it was asked to evaluate. In-context learning causes the model to memorize "
            "or mimic these specific prompt demonstrations, producing artificially inflated "
            "agreement on the evaluation set without any true increase in reasoning ability."
        ),
    }
    AUDIT_PATH.write_text(json.dumps(audit_data, indent=2), encoding="utf-8")
    logger.info("Saved detailed leakage audit to %s", AUDIT_PATH)

    return result_data


def main() -> None:
    """CLI entrypoint for induce_leakage.py."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Run Phase 6 Induced Data Leakage Experiment.")
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/splits/train.csv",
        help="Path to clean train split CSV.",
    )
    parser.add_argument(
        "--dev-path",
        type=str,
        default="data/splits/dev.csv",
        help="Path to dev split CSV.",
    )
    parser.add_argument(
        "--n-leaked",
        type=int,
        default=LEAKED_ITEMS_COUNT,
        help=f"Number of dev items to leak (default: {LEAKED_ITEMS_COUNT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mock",
        choices=["mock", "hosted", "local"],
        help="Judge backend (default: mock).",
    )
    args = parser.parse_args()

    result = run_leakage_experiment(
        train_path=args.train_path,
        dev_path=args.dev_path,
        n_leaked=args.n_leaked,
        seed=args.seed,
        backend=args.backend,
    )

    print("\n" + "=" * 65)
    print("INDUCED DATA LEAKAGE EXPERIMENT RESULTS")
    print("=" * 65)
    print(f"  Baseline Dev Kappa:     {result['baseline_dev_kappa']:.4f}")
    print(f"  Contaminated Dev Kappa: {result['contaminated_dev_kappa']:.4f}")
    print(f"  Leaked Items Count:     {result['leaked_items_count']}")
    delta = result["contaminated_dev_kappa"] - result["baseline_dev_kappa"]
    print(f"  Observed Delta:         {delta:+.4f}")
    print("=" * 65)
    print(f"Metrics written to: {OUTPUT_PATH}")
    print(f"Audit written to:   {AUDIT_PATH}")
    print(
        "\nWARNING: Contaminated Dev metrics are invalid for deployment decisions.\n"
        "Dev data was intentionally leaked into few-shot prompts to demonstrate inflation."
    )


if __name__ == "__main__":
    main()
