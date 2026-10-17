# AGENTS.md — LLM-as-a-Judge Calibration Pipeline

> **Authoritative project constitution.** Every contributor (human or AI agent)
> must read and follow this document when inspecting, maintaining, or extending this repository.

---

## 1. Project Summary

This project implements a **rigorously calibrated LLM-as-a-judge evaluation pipeline** that:

1. Labels 250 model outputs with human pass/fail judgments and written critiques.
2. Splits those labels into Train / Dev / Test (15 / 40 / 45 %).
3. Sends each output to an LLM judge that returns structured JSON `{critique, pass}`.
4. Measures TPR, TNR, raw agreement, and Cohen's Kappa — each with 95 % bootstrap CIs.
5. Quantifies **position bias** and **verbosity bias** through controlled perturbations.
6. Demonstrates the danger of **data leakage** by intentionally contaminating few-shot prompts.
7. Produces a fully reproducible, containerised evaluation harness.

---

## 2. Project Structure (Authoritative)

```
.
├── README.md                    # Decisions + numbers + sensitivity write-up
├── AGENTS.md                    # This file — project constitution
├── rubric.md                    # 1-page pass/fail rubric for the annotation task
├── Makefile                     # make install / up / dev / test / bench / leakage
├── Dockerfile
├── docker-compose.yml
├── .env.example                 # Config keys — NEVER commit .env
├── requirements.txt
├── submission.json              # Automated evaluator config
├── evaluate.py                  # Main CLI entrypoint
├── configs/
│   └── run_config.py            # RunConfig dataclass, defaults, validation
├── prompts/
│   ├── system_prompt.md         # Versioned judge system prompt
│   └── few_shot_template.md     # Template for assembling few-shot examples
├── data/
│   ├── labels.csv               # 250 manually annotated items
│   └── splits/
│       ├── train.csv            # 15 % — few-shot examples only
│       ├── dev.csv              # 40 % — prompt iteration
│       └── test.csv             # 45 % — frozen; run EXACTLY ONCE
├── results/
│   ├── metrics.csv              # TPR, TNR, kappa + CIs per split/iteration
│   ├── bias.csv                 # Position and verbosity probe results
│   ├── judge_test.csv           # Per-item judge output on frozen test split
│   └── induced_failure.json     # Contamination experiment output
├── src/
│   ├── __init__.py
│   ├── split_data.py            # Deterministic 15/40/45 splitter
│   ├── judge.py                 # LLMJudge class — API caller + budget guard
│   ├── metrics.py               # bootstrap_metric(), cohen_kappa wrapper
│   └── bias_probes.py           # Position and verbosity perturbation runners
├── scripts/
│   └── induce_leakage.py        # Intentional dev-data contamination demo
└── tests/
    ├── __init__.py
    ├── test_split_data.py       # Verifies row counts and no overlap
    ├── test_metrics.py          # Verifies bootstrap CI invariants
    ├── test_judge_contract.py   # Verifies JSON schema from judge
    └── test_bias_probes.py      # Verifies bias CSV schema
```

---

## 3. Core Requirements (Authoritative Contract)

Each requirement maps to an ID used in automated and manual verification:

| ID | Requirement | Output Artifact |
|----|-------------|-----------------|
| `docker-compose-setup` | `docker compose up` runs end-to-end, exits 0 | Container stdout |
| `dataset-annotation` | `data/labels.csv` has ≥ 250 rows, 5 exact columns | `data/labels.csv` |
| `judge-json-output` | `--test-judge` flag prints `{critique, pass}` JSON | stdout |
| `evaluation-metrics` | `results/metrics.csv` has all required columns + CI invariants | `results/metrics.csv` |
| `bias-probing` | `results/bias.csv` has position and verbosity rows | `results/bias.csv` |
| `induced-failure` | `results/induced_failure.json` with 3 keys; contaminated > baseline | `results/induced_failure.json` |
| `cli-execution` | `evaluate.py --split dev/test` both work | `results/metrics.csv` |
| `test-split-frozen-record` | `results/judge_test.csv` has 4 required columns | `results/judge_test.csv` |
| `submission-config` | `submission.json` matches schema | `submission.json` |
| `dataset-splits` | `data/splits/{train,dev,test}.csv` at 15/40/45 % | `data/splits/*.csv` |

---

## 4. Coding Conventions

| Topic | Rule |
|-------|------|
| **Python version** | 3.11 only |
| **Type hints** | Required on all public functions and class methods |
| **Docstrings** | Google-style docstrings on all modules, classes, and public functions |
| **Random seeds** | Always pass `seed=RANDOM_SEED` from env; never use bare `random.seed()` |
| **Imports** | Standard library → third-party → local; sorted alphabetically within each group |
| **Line length** | 100 characters max |
| **Formatting** | `ruff format` (Black-compatible); enforced in CI |
| **Linting** | `ruff check` + `mypy --strict` |
| **Secrets** | Read exclusively from `os.getenv()`; never hard-code values |
| **Error handling** | Raise typed exceptions; never bare `except:` |
| **Logging** | Use the `logging` stdlib; `rich` for CLI output only |

---

## 5. Data Schema Contracts (Authoritative — Do Not Deviate)

### `data/labels.csv`

| Column | Type | Constraint |
|--------|------|------------|
| `item_id` | string | Unique; format `item_XXXX` |
| `output` | string | The generated text being evaluated |
| `label` | integer | 0 or 1 only |
| `critique` | string | 1–3 sentences; non-empty |
| `relabeled` | integer | 0 or 1 only |

### `results/metrics.csv`

| Column | Type |
|--------|------|
| `split` | string |
| `tpr_mean` | float |
| `tpr_ci_low` | float |
| `tpr_ci_high` | float |
| `tnr_mean` | float |
| `tnr_ci_low` | float |
| `tnr_ci_high` | float |
| `agreement` | float |
| `kappa_mean` | float |
| `kappa_ci_low` | float |
| `kappa_ci_high` | float |

Invariants: `ci_low <= mean <= ci_high` for every metric.

### `results/bias.csv`

| Column | Type | Notes |
|--------|------|-------|
| `probe_type` | string | `position` or `verbosity` |
| `sample_size` | integer | |
| `position_flip_rate` | float | Required for `position` row |
| `pass_rate_before` | float | Required for `verbosity` row |
| `pass_rate_after` | float | Required for `verbosity` row |

### `results/judge_test.csv`

| Column | Type |
|--------|------|
| `item_id` | string |
| `human_label` | integer |
| `judge_label` | integer |
| `judge_critique` | string |

### `results/induced_failure.json`

```json
{
  "baseline_dev_kappa":      0.55,
  "contaminated_dev_kappa":  0.85,
  "leaked_items_count":      10
}
```

Invariant: `contaminated_dev_kappa > baseline_dev_kappa`.

---

## 6. Testing Rules

- Every public function in `src/` must have at least one unit test in `tests/`.
- Tests must be **deterministic**: pass a fixed `seed` to all stochastic calls.
- Tests must **not** make real API calls; use mocks (`unittest.mock`) or fixtures.
- Run `pytest tests/ -v --tb=short` before every commit.
- Verification gate: **all tests must pass** (`exit code 0`).
- Coverage target: ≥ 80 % for `src/` (enforced via `pytest-cov`).

---

## 7. Security Rules

| Rule | Detail |
|------|--------|
| Never commit `.env` | Listed in `.gitignore`; only `.env.example` goes to git |
| API keys from env only | `os.getenv("OPENAI_API_KEY")` — never string literals |
| Budget guard | `LLMJudge.spend` raises `BudgetExceededException` past `JUDGE_BUDGET_USD` |
| No model outputs in git | `data/labels.csv` contains human-generated text and is acceptable |

---

## 8. Determinism and Reproducibility Rules

- All splits use `RANDOM_SEED=42` (or the value from `.env`).
- Bootstrap resampling uses the same seed; results must be identical across runs.
- Record Python version, library versions, and hardware in `README.md`.
- Frozen evaluations are locked with the git commit tag `v1.0-test-frozen`.

---

## 9. Practitioner Teaching Anchors

> "For example, if an error occurs in 5% of examples, a judge that always predicts
> Pass still has 95% agreement while detecting none of the errors."
> — Hamel Husain

Implementation signal: report TPR and TNR **separately**. Any README claim about
judge quality must cite both values plus their CI lower bounds.

> "Seeing how the LLM breaks down its reasoning made me realize I wasn't being
> consistent about how I judged certain edge cases."
> — Phillip Carter, Honeycomb

Implementation signal: the `relabeled` column records how many of your own labels
changed after reading judge critiques. The README must report this count and
explain what edge case category shifted.
