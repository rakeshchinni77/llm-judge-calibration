"""
generate_dataset.py — CLI script to generate the 250-item annotation dataset.

Usage:
    python scripts/generate_dataset.py [OPTIONS]

Options:
    --model       LLM model name (default: env MODEL_NAME or gpt-4o-mini)
    --budget      Maximum spend in USD (default: env JUDGE_BUDGET_USD or 5.0)
    --seed        Random seed (default: env RANDOM_SEED or 42)
    --output-dir  Directory for output files (default: data/)
    --resume      Skip already-generated item_ids found in raw_outputs.csv
    --dry-run     Print the generation plan without calling the API

Outputs:
    data/raw_outputs.csv          — All 250 generated items + metadata
    data/generation_metadata.csv  — Same content; kept as audit trail
    data/annotation_template.csv  — Template for human labelling
                                    (item_id, question, output, label, critique, relabeled)

The annotation_template.csv is the file you open in Excel / Google Sheets.
Fill in the ``label`` and ``critique`` columns for every row, then run:

    python src/annotation.py --validate data/annotation_template.csv

Phase gate: Phase 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any imports that read env vars
load_dotenv()

# Resolve project root so the script runs from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.data_generation import (
    LLMGenerator,
    RawOutput,
    build_generation_plan,
)

console = Console()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV column order contracts
# ---------------------------------------------------------------------------

RAW_COLUMNS = [
    "item_id", "question", "output", "model_name", "condition_name",
    "temperature", "prompt_style", "generation_ts",
    "prompt_tokens", "completion_tokens", "domain", "qid",
]

TEMPLATE_COLUMNS = [
    "item_id", "question", "output", "label", "critique", "relabeled",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def raw_output_to_dict(ro: RawOutput) -> dict:
    """Convert a RawOutput dataclass to a flat dict for CSV writing."""
    return {
        "item_id":           ro.item_id,
        "question":          ro.question,
        "output":            ro.output,
        "model_name":        ro.model_name,
        "condition_name":    ro.condition_name,
        "temperature":       ro.temperature,
        "prompt_style":      ro.prompt_style,
        "generation_ts":     ro.generation_ts,
        "prompt_tokens":     ro.prompt_tokens,
        "completion_tokens": ro.completion_tokens,
        "domain":            ro.domain,
        "qid":               ro.qid,
    }


def load_existing_ids(raw_path: Path) -> set[str]:
    """Return item_ids already present in raw_outputs.csv (for --resume).

    Args:
        raw_path: Path to raw_outputs.csv.

    Returns:
        Set of already-generated item_ids.
    """
    if not raw_path.exists():
        return set()
    try:
        df = pd.read_csv(raw_path, usecols=["item_id"], dtype=str)
        return set(df["item_id"].dropna().tolist())
    except Exception:
        return set()


def write_annotation_template(
    raw_df: pd.DataFrame,
    template_path: Path,
) -> None:
    """Write the blank annotation template from generated outputs.

    Adds empty ``label`` and ``critique`` columns and pre-fills ``relabeled``
    with 0.  Annotators fill in label (0 or 1) and critique for each row.

    Args:
        raw_df:        DataFrame with at least ``item_id``, ``question``, ``output``.
        template_path: Destination path for the template CSV.
    """
    template = raw_df[["item_id", "question", "output"]].copy()
    template["label"]     = ""          # ← annotator fills this
    template["critique"]  = ""          # ← annotator fills this
    template["relabeled"] = 0           # start at 0; change to 1 if you revise a label
    template = template[TEMPLATE_COLUMNS]
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(template_path, index=False)
    console.print(f"[green][OK] Annotation template -> {template_path}[/green]")


def print_plan_summary(plan: list[dict]) -> None:
    """Print a rich table summarising the generation plan."""
    from collections import Counter
    cond_counts = Counter(p["condition"].name for p in plan)
    domain_counts = Counter(p["domain"] for p in plan)

    table = Table(title="Generation Plan (250 items)")
    table.add_column("Condition", style="cyan")
    table.add_column("Items", justify="right")
    table.add_column("Temperature", justify="right")
    table.add_column("Style")

    seen: dict[str, dict] = {}
    for p in plan:
        c = p["condition"]
        seen[c.name] = {"n": cond_counts[c.name], "temp": c.temperature, "style": c.prompt_style}

    for name, info in seen.items():
        table.add_row(name, str(info["n"]), str(info["temp"]), info["style"])

    console.print(table)

    domain_table = Table(title="Questions by Domain")
    domain_table.add_column("Domain", style="magenta")
    domain_table.add_column("Count", justify="right")
    for domain, count in sorted(domain_counts.items()):
        domain_table.add_row(domain, str(count))
    console.print(domain_table)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def run_generation(
    model_name: str,
    budget_usd: float,
    seed: int,
    output_dir: Path,
    resume: bool,
    dry_run: bool,
    backend: str = "hosted",
) -> None:
    """Execute the full generation pipeline.

    Args:
        model_name:  LLM model identifier.
        budget_usd:  Maximum spend in USD.
        seed:        Random seed.
        output_dir:  Directory for all output files.
        resume:      Whether to skip already-generated items.
        dry_run:     If True, print the plan and exit without API calls.
        backend:     Generation backend ("hosted", "local", or "mock").
    """
    raw_path      = output_dir / "raw_outputs.csv"
    metadata_path = output_dir / "generation_metadata.csv"
    template_path = output_dir / "annotation_template.csv"

    plan = build_generation_plan(seed=seed)
    console.rule("[bold blue]LLM-as-a-Judge Dataset Generation[/bold blue]")
    print_plan_summary(plan)

    if dry_run:
        console.print("[yellow]--dry-run: no API calls made.[/yellow]")
        return

    existing_ids = load_existing_ids(raw_path) if resume else set()
    if existing_ids:
        console.print(f"[yellow]--resume: skipping {len(existing_ids)} existing items[/yellow]")

    pending = [p for p in plan if p["item_id"] not in existing_ids]
    console.print(f"Items to generate: [bold]{len(pending)}[/bold]")

    generator = LLMGenerator(
        model_name=model_name,
        budget_usd=budget_usd,
        seed=seed,
        backend=backend,
    )

    rows: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating...", total=len(pending))

        for p in pending:
            item_id  = p["item_id"]
            question = p["question"]
            cond     = p["condition"]
            domain   = p["domain"]
            qid      = p["qid"]

            try:
                raw_out = generator.generate_one(
                    item_id=item_id,
                    question=question,
                    condition=cond,
                    domain=domain,
                    qid=qid,
                )
                rows.append(raw_output_to_dict(raw_out))
                progress.advance(task)

                # Checkpoint every 25 items to protect against interruptions
                if len(rows) % 25 == 0:
                    _checkpoint(rows, raw_path, metadata_path, existing_ids, plan)
                    console.print(
                        f"  [dim]Checkpoint: {len(rows)} new items, "
                        f"spend=${generator.spend:.4f}[/dim]"
                    )

                # Small delay to respect rate limits if using hosted/local API
                if backend != "mock":
                    time.sleep(0.3)

            except RuntimeError as exc:
                console.print(f"[red]Budget exceeded: {exc}[/red]")
                break
            except Exception as exc:
                console.print(f"[yellow][WARN] Skipped {item_id}: {exc}[/yellow]")
                logger.warning("Generation failed for %s: %s", item_id, exc)
                # Write a placeholder so --resume skips it
                rows.append({
                    "item_id": item_id, "question": question, "output": f"[GENERATION_ERROR: {exc}]",
                    "model_name": model_name, "condition_name": cond.name,
                    "temperature": cond.temperature, "prompt_style": cond.prompt_style,
                    "generation_ts": "", "prompt_tokens": 0, "completion_tokens": 0,
                    "domain": domain, "qid": qid,
                })
                progress.advance(task)

    # Final write
    _checkpoint(rows, raw_path, metadata_path, existing_ids, plan)

    # Build and write annotation template from full raw dataset
    full_raw_df = pd.read_csv(raw_path, dtype=str)
    write_annotation_template(full_raw_df, template_path)

    console.rule("[bold green]Generation Complete[/bold green]")
    console.print(f"  Total spend:   [bold]${generator.spend:.4f}[/bold]")
    console.print(f"  Items written: [bold]{len(full_raw_df)}[/bold]")
    console.print(f"\n[bold]Next steps:[/bold]")
    console.print("  1. Open [cyan]data/annotation_template.csv[/cyan] in Excel or Google Sheets.")
    console.print("  2. For each row: read the [bold]question[/bold] (context), then evaluate the [bold]output[/bold].")
    console.print("  3. Fill in [bold]label[/bold] (0=Fail, 1=Pass) and [bold]critique[/bold] (1–3 sentences).")
    console.print("  4. Keep [bold]relabeled[/bold]=0; set to 1 only if you change a label after judge feedback.")
    console.print("  5. Save the file, then run:")
    console.print("       [bold]python src/annotation.py --validate data/annotation_template.csv[/bold]")
    console.print("  6. To extract labels.csv from the completed template:")
    console.print("       [bold]python src/annotation.py --extract data/annotation_template.csv[/bold]")


def _checkpoint(
    new_rows: list[dict],
    raw_path: Path,
    metadata_path: Path,
    existing_ids: set[str],
    plan: list[dict],
) -> None:
    """Merge new rows with any pre-existing data and write CSV files.

    Args:
        new_rows:     Newly generated rows from this run.
        raw_path:     Path to raw_outputs.csv.
        metadata_path: Path to generation_metadata.csv.
        existing_ids: item_ids already in raw_outputs.csv before this run.
        plan:         Full generation plan (used to preserve original order).
    """
    existing_df = pd.read_csv(raw_path, dtype=str) if (raw_path.exists() and existing_ids) else pd.DataFrame()
    new_df = pd.DataFrame(new_rows, columns=RAW_COLUMNS)

    combined = (
        pd.concat([existing_df, new_df], ignore_index=True)
        if not existing_df.empty
        else new_df
    )
    # Preserve original plan order
    id_order = [p["item_id"] for p in plan]
    combined = combined.set_index("item_id").reindex(id_order).dropna(how="all").reset_index()

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(raw_path, index=False)
    combined.to_csv(metadata_path, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_dataset.py",
        description="Generate 250 technical Q&A outputs for annotation.",
    )
    parser.add_argument("--model",      default=os.getenv("MODEL_NAME", "gpt-4o-mini"))
    parser.add_argument(
        "--backend",
        default=os.getenv("GENERATION_BACKEND", "hosted"),
        choices=["hosted", "local", "mock"],
        help="Generation backend: hosted (OpenAI), local (Ollama), or mock",
    )
    parser.add_argument("--budget",     type=float, default=float(os.getenv("JUDGE_BUDGET_USD", "5.0")))
    parser.add_argument("--seed",       type=int,   default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument("--output-dir", default="data/", type=Path)
    parser.add_argument("--resume",     action="store_true", help="Skip already-generated items")
    parser.add_argument("--dry-run",    action="store_true", help="Print plan without calling API")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    run_generation(
        model_name=args.model,
        budget_usd=args.budget,
        seed=args.seed,
        output_dir=args.output_dir,
        resume=args.resume,
        dry_run=args.dry_run,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
