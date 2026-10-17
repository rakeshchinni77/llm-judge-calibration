"""
run_config.py — Experiment configuration dataclass.

A single RunConfig instance defines one evaluation arm.  All stochastic
components must read their seeds from this config to ensure reproducibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class RunConfig:
    """Configuration for a single evaluation arm.

    Attributes:
        seed:           Master random seed; propagated to all stochastic components.
        scenario:       Human-readable label for this arm (e.g. ``"dev_iter1"``).
        model_name:     LLM model identifier used by the judge.
        model_type:     ``"hosted"`` (OpenAI-compatible) or ``"local"`` (Ollama).
        budget_usd:     Maximum spend before the judge raises BudgetExceededException.
        labels_csv:     Path to the annotated dataset.
        splits_dir:     Directory containing train/dev/test.csv files.
        results_dir:    Directory where output CSVs and JSON are written.
        prompts_dir:    Directory containing versioned prompt files.
        n_bootstrap:    Number of bootstrap iterations for CI estimation.
    """

    seed: int = field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))
    scenario: str = "default"
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"))
    model_type: str = field(default_factory=lambda: os.getenv("MODEL_TYPE", "hosted"))
    budget_usd: float = field(
        default_factory=lambda: float(os.getenv("JUDGE_BUDGET_USD", "5.0"))
    )
    labels_csv: str = field(default_factory=lambda: os.getenv("LABELS_CSV", "data/labels.csv"))
    splits_dir: str = field(default_factory=lambda: os.getenv("SPLITS_DIR", "data/splits"))
    results_dir: str = field(default_factory=lambda: os.getenv("RESULTS_DIR", "results"))
    prompts_dir: str = field(default_factory=lambda: os.getenv("PROMPTS_DIR", "prompts"))
    n_bootstrap: int = 1000

    def __post_init__(self) -> None:
        """Validate field ranges after initialisation."""
        if not 0 < self.budget_usd <= 100:
            raise ValueError(f"budget_usd must be in (0, 100]; got {self.budget_usd}")
        if self.n_bootstrap < 100:
            raise ValueError(f"n_bootstrap must be >= 100; got {self.n_bootstrap}")
        if self.model_type not in ("hosted", "local", "mock"):
            raise ValueError(f"model_type must be 'hosted', 'local', or 'mock'; got {self.model_type}")
