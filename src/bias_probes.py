"""
bias_probes.py — Position and Verbosity Bias Probes.

Two controlled perturbation experiments expose systematic flaws in the LLM judge:

Position Bias
    Selects 100 unique (Pass, Fail) output pairs.
    Presents each pair in both candidate orders (A-first vs B-first).
    Records both decisions, the preferred candidate, the underlying human labels,
    and the pair ID.
    Computes position_flip_rate measuring changes in preference caused by reversing order.
    Distinguishes genuine preference reversals from API failures or abstentions.

Verbosity Bias
    Selects 50 Pass outputs and 50 Fail outputs.
    Records original judge decisions.
    Appends 2-3 fluent but irrelevant sentences without changing substantive content.
    Re-evaluates padded answers.
    Reports pass_rate_before and pass_rate_after overall and separately by class.

Outputs:
    results/bias.csv                — required schema with position and verbosity rows.
    results/bias_audit_position.csv — granular audit records for pairwise comparisons.
    results/bias_audit_verbosity.csv— granular audit records for verbosity perturbations.
    results/bias_audit.csv          — consolidated probe audit log.

Phase gate: Phase 5
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSITION_SAMPLE_SIZE: int = 100   # 100 unique (Pass, Fail) pairs
VERBOSITY_SAMPLE_SIZE: int = 100  # 100 total items (50 Pass + 50 Fail)

# Fluent, irrelevant padding that does not alter technical correctness or completeness.
VERBOSITY_PADDING: str = (
    "In summary, the system has been operating nominally across all monitored "
    "regions. Ongoing maintenance activities have no bearing on the above. "
    "For any further inquiries, please consult the internal knowledge base."
)

REQUIRED_BIAS_COLUMNS: list[str] = [
    "probe_type",
    "sample_size",
    "position_flip_rate",
    "pass_rate_before",
    "pass_rate_after",
]


# ---------------------------------------------------------------------------
# Pair Generation for Position Bias
# ---------------------------------------------------------------------------


def create_pass_fail_pairs(
    df: pd.DataFrame,
    n_pairs: int = POSITION_SAMPLE_SIZE,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select n_pairs unique (Pass, Fail) pairs from human-labeled data.

    Args:
        df: DataFrame containing at least 'item_id', 'output', and 'label'.
        n_pairs: Number of unique pairs to sample.
        seed: Random seed for deterministic selection.

    Returns:
        List of dicts, each with 'pair_id', 'pass_item', and 'fail_item'.

    Raises:
        ValueError: If df contains zero Pass or zero Fail examples, or lacks columns.
    """
    for col in ("output", "label"):
        if col not in df.columns:
            raise ValueError(f"Input DataFrame missing required column: '{col}'")

    pass_df = df[df["label"] == 1]
    fail_df = df[df["label"] == 0]

    if pass_df.empty:
        raise ValueError("Cannot form pairs: dataset contains zero Pass (label=1) examples.")
    if fail_df.empty:
        raise ValueError("Cannot form pairs: dataset contains zero Fail (label=0) examples.")

    pass_records = pass_df.to_dict(orient="records")
    fail_records = fail_df.to_dict(orient="records")

    # Generate all Cartesian product combinations of (Pass, Fail)
    all_combos = list(itertools.product(pass_records, fail_records))
    total_possible = len(all_combos)

    rng = np.random.default_rng(seed)
    if total_possible >= n_pairs:
        indices = rng.choice(total_possible, size=n_pairs, replace=False)
    else:
        logger.warning(
            "Total available pairs (%d) < requested n_pairs (%d); sampling with replacement.",
            total_possible,
            n_pairs,
        )
        indices = rng.choice(total_possible, size=n_pairs, replace=True)

    pairs: list[dict[str, Any]] = []
    for idx, combo_idx in enumerate(indices):
        p_item, f_item = all_combos[combo_idx]
        pairs.append({
            "pair_id": f"pair_{idx + 1:04d}",
            "pass_item": p_item,
            "fail_item": f_item,
        })

    return pairs


# ---------------------------------------------------------------------------
# Position Bias Probe
# ---------------------------------------------------------------------------


def run_position_probe(
    df: pd.DataFrame,
    judge: Any,
    n_pairs: int = POSITION_SAMPLE_SIZE,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the position bias probe on n_pairs of (Pass, Fail) items.

    Presents each pair in both candidate orders:
        Order 1: Candidate A = Pass, Candidate B = Fail
        Order 2: Candidate A = Fail, Candidate B = Pass

    Records decisions, preferred candidates, human labels, and pair IDs.
    Computes position_flip_rate measuring changes in preference caused by
    order reversal. Distinguishes genuine position bias from API failures
    or abstentions.

    Args:
        df: DataFrame containing human-labeled items ('output', 'label', 'item_id').
        judge: Instantiated LLMJudge (or duck-typed object).
        n_pairs: Number of (Pass, Fail) pairs to evaluate.
        seed: Random seed for pair sampling.

    Returns:
        Dict with keys:
            probe_type: 'position'
            sample_size: int
            position_flip_rate: float
            valid_pairs: int
            error_pairs: int
            flips_count: int
            position_a_bias_count: int
            position_b_bias_count: int
            consistent_pass_count: int
            consistent_fail_count: int
            audit_records: list[dict[str, Any]]
    """
    pairs = create_pass_fail_pairs(df, n_pairs=n_pairs, seed=seed)
    audit_records: list[dict[str, Any]] = []

    logger.info("Evaluating %d pairs in both candidate orders...", len(pairs))

    for p in pairs:
        pair_id = p["pair_id"]
        pass_item = p["pass_item"]
        fail_item = p["fail_item"]

        pass_id = str(pass_item.get("item_id", "pass"))
        fail_id = str(fail_item.get("item_id", "fail"))
        pass_text = pass_item["output"]
        fail_text = fail_item["output"]

        # Order 1: A = Pass, B = Fail
        order1_decision: str | None = None
        order1_preferred_id: str | None = None
        order1_critique: str = ""
        order1_error: str | None = None

        try:
            if hasattr(judge, "evaluate_pairwise"):
                res1 = judge.evaluate_pairwise(cand_a=pass_text, cand_b=fail_text)
            else:
                # Fallback to single evaluation if pairwise method not present
                r_a = judge.evaluate(pass_text)
                r_b = judge.evaluate(fail_text)
                res1 = {
                    "critique": f"Pass item: {r_a.get('critique', '')} | Fail item: {r_b.get('critique', '')}",
                    "preference": "A" if r_a.get("pass") and not r_b.get("pass") else ("B" if r_b.get("pass") and not r_a.get("pass") else "A"),
                }
            pref1 = str(res1.get("preference", "")).strip().upper()
            if pref1 in ("A", "B"):
                order1_decision = pref1
                order1_preferred_id = pass_id if pref1 == "A" else fail_id
                order1_critique = str(res1.get("critique", ""))
            else:
                order1_error = f"Invalid preference returned: {pref1}"
        except Exception as exc:
            order1_error = str(exc)
            logger.warning("Order 1 failed for %s: %s", pair_id, exc)

        # Order 2: A = Fail, B = Pass
        order2_decision: str | None = None
        order2_preferred_id: str | None = None
        order2_critique: str = ""
        order2_error: str | None = None

        try:
            if hasattr(judge, "evaluate_pairwise"):
                res2 = judge.evaluate_pairwise(cand_a=fail_text, cand_b=pass_text)
            else:
                r_a = judge.evaluate(fail_text)
                r_b = judge.evaluate(pass_text)
                res2 = {
                    "critique": f"Fail item: {r_a.get('critique', '')} | Pass item: {r_b.get('critique', '')}",
                    "preference": "A" if r_a.get("pass") and not r_b.get("pass") else ("B" if r_b.get("pass") and not r_a.get("pass") else "A"),
                }
            pref2 = str(res2.get("preference", "")).strip().upper()
            if pref2 in ("A", "B"):
                order2_decision = pref2
                order2_preferred_id = fail_id if pref2 == "A" else pass_id
                order2_critique = str(res2.get("critique", ""))
            else:
                order2_error = f"Invalid preference returned: {pref2}"
        except Exception as exc:
            order2_error = str(exc)
            logger.warning("Order 2 failed for %s: %s", pair_id, exc)

        # Decision logic analysis & classification
        if order1_error or order2_error or order1_decision is None or order2_decision is None:
            status = "error"
            is_flip: bool | None = None
            flip_type = "error"
        else:
            status = "valid"
            if order1_preferred_id != order2_preferred_id:
                is_flip = True
                if order1_decision == "A" and order2_decision == "A":
                    flip_type = "position_a_bias"
                elif order1_decision == "B" and order2_decision == "B":
                    flip_type = "position_b_bias"
                else:
                    flip_type = "preference_reversal"
            else:
                is_flip = False
                if order1_preferred_id == pass_id:
                    flip_type = "consistent_pass"
                else:
                    flip_type = "consistent_fail"

        audit_records.append({
            "pair_id": pair_id,
            "pass_item_id": pass_id,
            "fail_item_id": fail_id,
            "order1_cand_a_id": pass_id,
            "order1_cand_b_id": fail_id,
            "order1_decision": order1_decision,
            "order1_preferred_id": order1_preferred_id,
            "order1_critique": order1_critique,
            "order1_error": order1_error,
            "order2_cand_a_id": fail_id,
            "order2_cand_b_id": pass_id,
            "order2_decision": order2_decision,
            "order2_preferred_id": order2_preferred_id,
            "order2_critique": order2_critique,
            "order2_error": order2_error,
            "status": status,
            "is_flip": is_flip,
            "flip_type": flip_type,
        })

    valid_pairs = [r for r in audit_records if r["status"] == "valid"]
    n_valid = len(valid_pairs)
    flips_count = sum(1 for r in valid_pairs if r["is_flip"] is True)
    position_flip_rate = round(float(flips_count / n_valid), 4) if n_valid > 0 else 0.0

    return {
        "probe_type": "position",
        "sample_size": len(audit_records),
        "position_flip_rate": position_flip_rate,
        "valid_pairs": n_valid,
        "error_pairs": len(audit_records) - n_valid,
        "flips_count": flips_count,
        "position_a_bias_count": sum(1 for r in audit_records if r.get("flip_type") == "position_a_bias"),
        "position_b_bias_count": sum(1 for r in audit_records if r.get("flip_type") == "position_b_bias"),
        "consistent_pass_count": sum(1 for r in audit_records if r.get("flip_type") == "consistent_pass"),
        "consistent_fail_count": sum(1 for r in audit_records if r.get("flip_type") == "consistent_fail"),
        "audit_records": audit_records,
    }


# ---------------------------------------------------------------------------
# Item Sampling for Verbosity Bias
# ---------------------------------------------------------------------------


def sample_verbosity_items(
    df: pd.DataFrame,
    n_per_class: int = VERBOSITY_SAMPLE_SIZE // 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample balanced Pass and Fail items for verbosity probing.

    Args:
        df: DataFrame containing 'output' and 'label' columns.
        n_per_class: Number of items to sample per class (default 50).
        seed: Random seed for sampling.

    Returns:
        DataFrame with 2 * n_per_class balanced rows.

    Raises:
        ValueError: If df missing required columns or has no items for a class.
    """
    for col in ("output", "label"):
        if col not in df.columns:
            raise ValueError(f"Input DataFrame missing required column: '{col}'")

    pass_df = df[df["label"] == 1]
    fail_df = df[df["label"] == 0]

    if pass_df.empty or fail_df.empty:
        raise ValueError("Dataset must contain both Pass and Fail items for verbosity testing.")

    rng = np.random.default_rng(seed)

    # Sample Pass items
    if len(pass_df) >= n_per_class:
        pass_idx = rng.choice(len(pass_df), size=n_per_class, replace=False)
    else:
        logger.warning(
            "Pass count (%d) < requested %d; sampling with replacement.",
            len(pass_df),
            n_per_class,
        )
        pass_idx = rng.choice(len(pass_df), size=n_per_class, replace=True)
    sampled_pass = pass_df.iloc[pass_idx]

    # Sample Fail items
    if len(fail_df) >= n_per_class:
        fail_idx = rng.choice(len(fail_df), size=n_per_class, replace=False)
    else:
        logger.warning(
            "Fail count (%d) < requested %d; sampling with replacement.",
            len(fail_df),
            n_per_class,
        )
        fail_idx = rng.choice(len(fail_df), size=n_per_class, replace=True)
    sampled_fail = fail_df.iloc[fail_idx]

    combined = pd.concat([sampled_pass, sampled_fail], ignore_index=True)
    # Shuffle deterministically
    perm = rng.permutation(len(combined))
    return combined.iloc[perm].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Verbosity Bias Probe
# ---------------------------------------------------------------------------


def run_verbosity_probe(
    df: pd.DataFrame,
    judge: Any,
    n_per_class: int = VERBOSITY_SAMPLE_SIZE // 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the verbosity bias probe by padding outputs and re-evaluating.

    Appends 2-3 fluent but irrelevant sentences without changing substantive
    meaning. Re-evaluates each output and calculates pass_rate_before and
    pass_rate_after overall and per human label class.

    Args:
        df: DataFrame with 'output', 'label', and 'item_id'.
        judge: Instantiated LLMJudge.
        n_per_class: Number of Pass and Fail items to sample (each).
        seed: Random seed for sampling.

    Returns:
        Dict with keys:
            probe_type: 'verbosity'
            sample_size: int
            pass_rate_before: float
            pass_rate_after: float
            pass_class_pass_rate_before: float
            pass_class_pass_rate_after: float
            fail_class_pass_rate_before: float
            fail_class_pass_rate_after: float
            valid_items: int
            error_items: int
            audit_records: list[dict[str, Any]]
    """
    sampled = sample_verbosity_items(df, n_per_class=n_per_class, seed=seed)
    audit_records: list[dict[str, Any]] = []

    logger.info("Evaluating %d items before and after verbosity padding...", len(sampled))

    for _, row in sampled.iterrows():
        item_id = str(row.get("item_id", "unknown"))
        human_label = int(row["label"])
        orig_text = str(row["output"])
        padded_text = f"{orig_text.strip()} {VERBOSITY_PADDING}"

        # Evaluate original output
        decision_before: bool | None = None
        critique_before: str = ""
        error_before: str | None = None
        try:
            res_before = judge.evaluate(orig_text)
            passed_val = res_before.get("pass")
            if isinstance(passed_val, bool):
                decision_before = passed_val
                critique_before = str(res_before.get("critique", ""))
            else:
                error_before = f"Non-boolean pass field: {passed_val}"
        except Exception as exc:
            error_before = str(exc)
            logger.warning("Original evaluation failed for item %s: %s", item_id, exc)

        # Evaluate padded output
        decision_after: bool | None = None
        critique_after: str = ""
        error_after: str | None = None
        try:
            res_after = judge.evaluate(padded_text)
            passed_val = res_after.get("pass")
            if isinstance(passed_val, bool):
                decision_after = passed_val
                critique_after = str(res_after.get("critique", ""))
            else:
                error_after = f"Non-boolean pass field: {passed_val}"
        except Exception as exc:
            error_after = str(exc)
            logger.warning("Padded evaluation failed for item %s: %s", item_id, exc)

        # Shift classification
        if error_before or error_after or decision_before is None or decision_after is None:
            status = "error"
            shift_type = "error"
        else:
            status = "valid"
            if decision_before == decision_after:
                shift_type = "unchanged_pass" if decision_before else "unchanged_fail"
            elif not decision_before and decision_after:
                shift_type = "verbosity_inflated"  # fail became pass after padding
            else:
                shift_type = "verbosity_deflated"  # pass became fail after padding

        audit_records.append({
            "item_id": item_id,
            "human_label": human_label,
            "decision_before": decision_before,
            "decision_after": decision_after,
            "critique_before": critique_before,
            "critique_after": critique_after,
            "error_before": error_before,
            "error_after": error_after,
            "status": status,
            "shift_type": shift_type,
        })

    valid_items = [r for r in audit_records if r["status"] == "valid"]
    n_valid = len(valid_items)

    pass_rate_before = (
        round(sum(1 for r in valid_items if r["decision_before"] is True) / n_valid, 4)
        if n_valid > 0
        else 0.0
    )
    pass_rate_after = (
        round(sum(1 for r in valid_items if r["decision_after"] is True) / n_valid, 4)
        if n_valid > 0
        else 0.0
    )

    # Separate class calculations (Requirement 5)
    pass_class_items = [r for r in valid_items if r["human_label"] == 1]
    fail_class_items = [r for r in valid_items if r["human_label"] == 0]

    pass_class_before = (
        round(sum(1 for r in pass_class_items if r["decision_before"] is True) / len(pass_class_items), 4)
        if pass_class_items
        else 0.0
    )
    pass_class_after = (
        round(sum(1 for r in pass_class_items if r["decision_after"] is True) / len(pass_class_items), 4)
        if pass_class_items
        else 0.0
    )

    fail_class_before = (
        round(sum(1 for r in fail_class_items if r["decision_before"] is True) / len(fail_class_items), 4)
        if fail_class_items
        else 0.0
    )
    fail_class_after = (
        round(sum(1 for r in fail_class_items if r["decision_after"] is True) / len(fail_class_items), 4)
        if fail_class_items
        else 0.0
    )

    return {
        "probe_type": "verbosity",
        "sample_size": len(audit_records),
        "pass_rate_before": pass_rate_before,
        "pass_rate_after": pass_rate_after,
        "pass_class_pass_rate_before": pass_class_before,
        "pass_class_pass_rate_after": pass_class_after,
        "fail_class_pass_rate_before": fail_class_before,
        "fail_class_pass_rate_after": fail_class_after,
        "valid_items": n_valid,
        "error_items": len(audit_records) - n_valid,
        "audit_records": audit_records,
    }


# ---------------------------------------------------------------------------
# Write bias results
# ---------------------------------------------------------------------------


def write_bias_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Persist bias probe rows to results/bias.csv matching authoritative schema.

    Schema:
        probe_type, sample_size, position_flip_rate, pass_rate_before, pass_rate_after

    Args:
        rows: List of dicts, one per probe type.
        output_path: Destination file path.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    formatted_rows: list[dict[str, Any]] = []
    for r in rows:
        p_type = r.get("probe_type")
        s_size = r.get("sample_size")

        if p_type == "position":
            formatted_rows.append({
                "probe_type": "position",
                "sample_size": s_size,
                "position_flip_rate": r.get("position_flip_rate"),
                "pass_rate_before": np.nan,
                "pass_rate_after": np.nan,
            })
        elif p_type == "verbosity":
            formatted_rows.append({
                "probe_type": "verbosity",
                "sample_size": s_size,
                "position_flip_rate": np.nan,
                "pass_rate_before": r.get("pass_rate_before"),
                "pass_rate_after": r.get("pass_rate_after"),
            })
        else:
            formatted_rows.append({
                "probe_type": p_type,
                "sample_size": s_size,
                "position_flip_rate": r.get("position_flip_rate", np.nan),
                "pass_rate_before": r.get("pass_rate_before", np.nan),
                "pass_rate_after": r.get("pass_rate_after", np.nan),
            })

    out_df = pd.DataFrame(formatted_rows, columns=REQUIRED_BIAS_COLUMNS)
    out_df.to_csv(out_p, index=False)
    logger.info("Wrote %d bias probe rows to %s", len(out_df), out_p)


def write_audit_records(
    position_audit: list[dict[str, Any]],
    verbosity_audit: list[dict[str, Any]],
    output_dir: str | Path = "results",
) -> None:
    """Save granular audit trails for both probe types.

    Args:
        position_audit: Detailed records from position probe.
        verbosity_audit: Detailed records from verbosity probe.
        output_dir: Directory where audit CSVs will be saved.
    """
    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    if position_audit:
        pos_df = pd.DataFrame(position_audit)
        pos_path = out_d / "bias_audit_position.csv"
        pos_df.to_csv(pos_path, index=False)
        logger.info("Saved position probe audit to %s", pos_path)

    if verbosity_audit:
        verb_df = pd.DataFrame(verbosity_audit)
        verb_path = out_d / "bias_audit_verbosity.csv"
        verb_df.to_csv(verb_path, index=False)
        logger.info("Saved verbosity probe audit to %s", verb_path)

    # Write unified audit CSV
    combined_rows: list[dict[str, Any]] = []
    for r in position_audit:
        combined_rows.append({
            "probe_type": "position",
            "id": r.get("pair_id"),
            "status": r.get("status"),
            "primary_metric": r.get("is_flip"),
            "classification": r.get("flip_type"),
            "details": f"O1_pref={r.get('order1_preferred_id')}, O2_pref={r.get('order2_preferred_id')}",
        })
    for r in verbosity_audit:
        combined_rows.append({
            "probe_type": "verbosity",
            "id": r.get("item_id"),
            "status": r.get("status"),
            "primary_metric": r.get("decision_after"),
            "classification": r.get("shift_type"),
            "details": f"human={r.get('human_label')}, before={r.get('decision_before')}, after={r.get('decision_after')}",
        })

    if combined_rows:
        unified_path = out_d / "bias_audit.csv"
        pd.DataFrame(combined_rows).to_csv(unified_path, index=False)
        logger.info("Saved unified bias audit to %s", unified_path)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all bias probes and write results/bias.csv."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Run position and verbosity bias probes.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/labels.csv",
        help="Input CSV path (default: data/labels.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/bias.csv",
        help="Destination bias results CSV path.",
    )
    parser.add_argument(
        "--audit-dir",
        type=str,
        default="results",
        help="Destination directory for audit CSVs.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mock",
        choices=["mock", "hosted", "local"],
        help="Judge execution backend (default: mock).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--n-position-pairs",
        type=int,
        default=POSITION_SAMPLE_SIZE,
        help=f"Number of position pairs to evaluate (default: {POSITION_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--n-verbosity-per-class",
        type=int,
        default=VERBOSITY_SAMPLE_SIZE // 2,
        help=f"Number of Pass and Fail items to evaluate for verbosity (default: {VERBOSITY_SAMPLE_SIZE // 2}).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # Fallback to dev if labels.csv not found
        input_path = Path("data/splits/dev.csv")
        if not input_path.exists():
            raise FileNotFoundError(f"Neither {args.input} nor data/splits/dev.csv found.")

    logger.info("Loading evaluation dataset from %s...", input_path)
    df = pd.read_csv(input_path)

    # Initialize judge
    from src.judge import LLMJudge
    judge = LLMJudge(backend=args.backend, seed=args.seed)

    logger.info("Starting Position Bias Probe...")
    pos_res = run_position_probe(
        df=df,
        judge=judge,
        n_pairs=args.n_position_pairs,
        seed=args.seed,
    )
    logger.info(
        "Position Probe complete: sample_size=%d, valid=%d, flips=%d, flip_rate=%.4f",
        pos_res["sample_size"],
        pos_res["valid_pairs"],
        pos_res["flips_count"],
        pos_res["position_flip_rate"],
    )

    logger.info("Starting Verbosity Bias Probe...")
    verb_res = run_verbosity_probe(
        df=df,
        judge=judge,
        n_per_class=args.n_verbosity_per_class,
        seed=args.seed,
    )
    logger.info(
        "Verbosity Probe complete: sample_size=%d, valid=%d, pass_rate_before=%.4f, pass_rate_after=%.4f",
        verb_res["sample_size"],
        verb_res["valid_items"],
        verb_res["pass_rate_before"],
        verb_res["pass_rate_after"],
    )

    # Persist summary CSV
    write_bias_csv([pos_res, verb_res], output_path=args.output)

    # Persist audit CSVs
    write_audit_records(
        position_audit=pos_res.get("audit_records", []),
        verbosity_audit=verb_res.get("audit_records", []),
        output_dir=args.audit_dir,
    )

    logger.info("Judge total spend: $%.4f", judge.spend)
    print("\n[OK] Bias Probes completed successfully.")
    print(f"Results written to: {args.output}")
    print(f"Audit trails written to: {args.audit_dir}/bias_audit*.csv")


if __name__ == "__main__":
    main()
