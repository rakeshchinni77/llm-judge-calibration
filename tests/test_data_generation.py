"""
test_data_generation.py — Unit tests for src/data_generation.py.

Tests verify:
- Generation plan produces exactly 250 items
- Generation plan conditions contain >= 30% failure-inducing arms
- Question bank covers all 6 required domains with unique question IDs
- Generation conditions preserve prompt styles and temperature settings
- LLMGenerator with mock backend produces valid RawOutput instances
- LLMGenerator tracks budget and token usage correctly
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.data_generation import (
    GENERATION_CONDITIONS,
    QUESTION_BANK,
    GenerationCondition,
    LLMGenerator,
    RawOutput,
    build_generation_plan,
    build_messages,
)


def test_question_bank_structure() -> None:
    """Question bank must contain at least 100 questions with unique QIDs."""
    assert len(QUESTION_BANK) >= 100
    qids = [q["qid"] for q in QUESTION_BANK]
    assert len(qids) == len(set(qids)), "QIDs must be unique"

    domains = {q["domain"] for q in QUESTION_BANK}
    expected_domains = {"python", "algorithms", "ml", "statistics", "sql", "software_eng"}
    assert expected_domains.issubset(domains), f"Missing domains: {expected_domains - domains}"


def test_build_generation_plan_length() -> None:
    """Generation plan must produce exactly 250 items."""
    plan = build_generation_plan(seed=42)
    assert len(plan) == 250
    item_ids = [p["item_id"] for p in plan]
    assert len(item_ids) == len(set(item_ids)), "item_ids in plan must be unique"
    assert item_ids[0] == "item_0001"
    assert item_ids[-1] == "item_0250"


def test_failure_inducing_conditions_share() -> None:
    """At least 30% of items in the plan must belong to failure-inducing conditions."""
    plan = build_generation_plan(seed=42)
    failure_conditions = {"minimal_low", "minimal_high", "misleading", "adversarial"}
    n_failure_items = sum(1 for p in plan if p["condition"].name in failure_conditions)
    failure_ratio = n_failure_items / len(plan)
    assert failure_ratio >= 0.30, f"Expected >= 30% failure-inducing items, got {failure_ratio:.1%}"


def test_build_messages_structure() -> None:
    """build_messages must return valid OpenAI-compatible message list."""
    cond = GenerationCondition(
        name="test_cond",
        temperature=0.7,
        prompt_style="standard",
    )
    messages = build_messages("What is a generator in Python?", cond)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "What is a generator" in messages[1]["content"]


def test_mock_generator_produces_raw_output() -> None:
    """Mock generator must produce valid RawOutput with populated tokens and metadata."""
    generator = LLMGenerator(
        model_name="mock-generator",
        budget_usd=5.0,
        seed=42,
        backend="mock",
    )
    cond = GenerationCondition(
        name="standard_low",
        temperature=0.1,
        prompt_style="standard",
    )
    raw = generator.generate_one(
        item_id="item_0001",
        question="What is the difference between a list and a tuple in Python?",
        condition=cond,
        domain="python",
        qid="Q001",
    )
    assert isinstance(raw, RawOutput)
    assert raw.item_id == "item_0001"
    assert len(raw.output) > 20
    assert raw.prompt_tokens > 0
    assert raw.completion_tokens > 0
    assert raw.model_name == "mock-generator"
    assert raw.condition_name == "standard_low"


def test_generator_budget_limit() -> None:
    """Generator must raise RuntimeError when budget is exceeded."""
    generator = LLMGenerator(
        model_name="mock-generator",
        budget_usd=0.0001,  # extremely low budget
        seed=42,
        backend="mock",
    )
    cond = GenerationCondition(
        name="standard_low",
        temperature=0.1,
        prompt_style="standard",
    )
    with pytest.raises(RuntimeError, match="Generation budget exceeded"):
        for i in range(100):
            generator.generate_one(
                item_id=f"item_{i:04d}",
                question="Explain Python's GIL.",
                condition=cond,
                domain="python",
                qid="Q011",
            )
