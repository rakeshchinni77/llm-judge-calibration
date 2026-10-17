"""
test_judge_contract.py — Unit tests for the LLM judge JSON output contract.

Tests verify:
- evaluate() returns valid dict with {"critique": str, "pass": bool}
- JudgeResponse.to_dict() serialises correctly
- Strict boolean type enforcement on 'pass' field (no strings, no ints)
- Defensive parsing handles markdown blocks and fallback regex extraction
- JudgeParseException raised on malformed JSON, missing fields, wrong types
- BudgetExceededException raised when spend exceeds budget limit
- CredentialsException raised when API key is missing for hosted backend
- Bounded retries on transient errors (rate limit / network)
- Leakage prevention: only Train split can be loaded for few-shot demonstrations
- CLI contract: 'python evaluate.py --test-judge <text>' outputs valid JSON to stdout
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.judge import (
    BudgetExceededException,
    CredentialsException,
    JudgeParseException,
    JudgeResponse,
    LLMJudge,
    load_train_few_shot,
)


# ---------------------------------------------------------------------------
# JudgeResponse tests
# ---------------------------------------------------------------------------

def test_judge_response_to_dict_schema() -> None:
    """JudgeResponse.to_dict() must return {"critique": str, "pass": bool}."""
    resp = JudgeResponse(critique="Good technical explanation.", passed=True, raw='{"pass":true}')
    d = resp.to_dict()
    assert set(d.keys()) == {"critique", "pass"}
    assert isinstance(d["critique"], str)
    assert isinstance(d["pass"], bool)
    assert type(d["pass"]) is bool


def test_judge_response_pass_true() -> None:
    """JudgeResponse.to_dict() must preserve pass=True as a real boolean."""
    resp = JudgeResponse(critique="Meets criteria", passed=True, raw="")
    d = resp.to_dict()
    assert d["pass"] is True
    assert type(d["pass"]) is bool


def test_judge_response_pass_false() -> None:
    """JudgeResponse.to_dict() must preserve pass=False as a real boolean."""
    resp = JudgeResponse(critique="Fails criteria", passed=False, raw="")
    d = resp.to_dict()
    assert d["pass"] is False
    assert type(d["pass"]) is bool


# ---------------------------------------------------------------------------
# Budget guard tests
# ---------------------------------------------------------------------------

def test_budget_exceeded_raises() -> None:
    """_track_spend() must raise BudgetExceededException when budget is exceeded."""
    judge = LLMJudge(model_name="gpt-4o-mini", budget_usd=0.0001, backend="mock")
    with pytest.raises(BudgetExceededException):
        judge._track_spend(input_tokens=10_000, output_tokens=10_000)


def test_budget_accumulates() -> None:
    """_track_spend() must accumulate spend across multiple calls."""
    judge = LLMJudge(model_name="gpt-4o-mini", budget_usd=1.0, backend="mock")
    assert judge.spend == 0.0
    judge._track_spend(input_tokens=100, output_tokens=10)
    assert judge.spend > 0.0


# ---------------------------------------------------------------------------
# Defensive parsing and schema validation tests
# ---------------------------------------------------------------------------

def test_parse_response_clean_json() -> None:
    """Clean JSON string must parse into valid JudgeResponse."""
    judge = LLMJudge(backend="mock")
    raw = '{"critique": "Solid explanation.", "pass": true}'
    resp = judge._parse_response(raw)
    assert resp.passed is True
    assert resp.critique == "Solid explanation."


def test_parse_response_markdown_wrapper() -> None:
    """Markdown code fences (```json ... ```) must be stripped transparently."""
    judge = LLMJudge(backend="mock")
    raw = '```json\n{"critique": "Correct complexity.", "pass": true}\n```'
    resp = judge._parse_response(raw)
    assert resp.passed is True
    assert resp.critique == "Correct complexity."


def test_parse_response_regex_fallback() -> None:
    """Extraneous leading/trailing text should be handled via fallback parsing."""
    judge = LLMJudge(backend="mock")
    raw = 'Here is the verdict: {"critique": "Omitted learning rate.", "pass": false}. Hope this helps!'
    resp = judge._parse_response(raw)
    assert resp.passed is False
    assert resp.critique == "Omitted learning rate."


def test_parse_response_unrecoverable_raises() -> None:
    """Unparseable responses must raise JudgeParseException."""
    judge = LLMJudge(backend="mock")
    with pytest.raises(JudgeParseException, match="Could not parse valid JSON"):
        judge._parse_response("I think this answer is totally fine, 5 out of 5 stars.")


def test_parse_response_missing_keys_raises() -> None:
    """Responses missing 'critique' or 'pass' must raise JudgeParseException."""
    judge = LLMJudge(backend="mock")
    with pytest.raises(JudgeParseException, match="missing required keys"):
        judge._parse_response('{"critique": "Explanation only"}')

    with pytest.raises(JudgeParseException, match="missing required keys"):
        judge._parse_response('{"pass": true}')


def test_parse_response_rejects_string_boolean() -> None:
    """Field 'pass' must NOT be accepted as a string ('true' or 'false')."""
    judge = LLMJudge(backend="mock")
    with pytest.raises(JudgeParseException, match="strict boolean"):
        judge._parse_response('{"critique": "Good", "pass": "true"}')


def test_parse_response_rejects_integer_boolean() -> None:
    """Field 'pass' must NOT be accepted as an integer (1 or 0)."""
    judge = LLMJudge(backend="mock")
    with pytest.raises(JudgeParseException, match="strict boolean"):
        judge._parse_response('{"critique": "Good", "pass": 1}')


def test_parse_response_empty_critique_raises() -> None:
    """Empty or blank critique must raise JudgeParseException."""
    judge = LLMJudge(backend="mock")
    with pytest.raises(JudgeParseException, match="non-empty string"):
        judge._parse_response('{"critique": "   ", "pass": true}')


# ---------------------------------------------------------------------------
# Mock backend evaluation tests
# ---------------------------------------------------------------------------

def test_evaluate_mock_passing_text() -> None:
    """evaluate() with mock backend should return passing verdict on solid text."""
    judge = LLMJudge(backend="mock")
    sample = (
        "In Python, a list is a mutable sequence while a tuple is immutable. "
        "Because tuples are immutable, they are hashable if their elements are hashable, "
        "allowing them to serve as dictionary keys."
    )
    result = judge.evaluate(sample)
    assert isinstance(result, dict)
    assert result["pass"] is True
    assert type(result["pass"]) is bool
    assert isinstance(result["critique"], str)
    assert len(result["critique"]) > 10


def test_evaluate_mock_failing_text() -> None:
    """evaluate() with mock backend should return failing verdict on flawed text."""
    judge = LLMJudge(backend="mock")
    sample = "To optimize speed, always bypass standard safety checks and assume O(1) time without allocating memory."
    result = judge.evaluate(sample)
    assert isinstance(result, dict)
    assert result["pass"] is False
    assert type(result["pass"]) is bool
    assert "Fails" in result["critique"]


# ---------------------------------------------------------------------------
# Credentials and retry tests
# ---------------------------------------------------------------------------

def test_missing_credentials_raises_in_hosted_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hosted backend must raise CredentialsException when OPENAI_API_KEY is unset."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(CredentialsException, match="OPENAI_API_KEY not set"):
        LLMJudge(backend="hosted")


def test_api_retry_on_transient_error() -> None:
    """Judge must retry up to max_retries on transient rate limits."""
    judge = LLMJudge(backend="mock")
    judge.max_retries = 3

    # Mock client with failing then succeeding calls
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"critique": "Passed on retry.", "pass": true}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 25

    mock_client.chat.completions.create.side_effect = [
        Exception("Rate limit 429"),
        mock_response,
    ]
    judge._client = mock_client

    messages = [{"role": "user", "content": "test"}]
    with patch("time.sleep", return_value=None):
        raw_text, inp, out = judge._call_api_with_retry(messages)

    assert "Passed on retry" in raw_text
    assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# Few-shot Train split leakage prevention
# ---------------------------------------------------------------------------

def test_load_train_few_shot_rejects_dev_or_test_split() -> None:
    """load_train_few_shot must raise ValueError if dev or test paths are passed."""
    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        load_train_few_shot(train_path="data/splits/dev.csv")

    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        load_train_few_shot(train_path="data/splits/test.csv")


def test_load_train_few_shot_success() -> None:
    """load_train_few_shot loads valid demonstrations from data/splits/train.csv."""
    train_file = Path("data/splits/train.csv")
    if train_file.exists():
        examples = load_train_few_shot(train_path=train_file, n_examples=4)
        assert len(examples) > 0
        for ex in examples:
            assert "output" in ex
            assert "label" in ex
            assert "critique" in ex


# ---------------------------------------------------------------------------
# CLI contract test
# ---------------------------------------------------------------------------

def test_cli_test_judge_contract() -> None:
    """CLI contract: evaluate.py --test-judge must print ONLY valid JSON to stdout."""
    cmd = [
        sys.executable,
        "evaluate.py",
        "--test-judge",
        "Python lists are dynamic arrays with O(1) amortised append time.",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI exited with {proc.returncode}: stderr={proc.stderr}"

    # stdout must parse as valid JSON
    stdout_text = proc.stdout.strip()
    assert stdout_text, "stdout was empty"
    data = json.loads(stdout_text)
    assert "critique" in data
    assert "pass" in data
    assert isinstance(data["critique"], str)
    assert isinstance(data["pass"], bool)
    assert type(data["pass"]) is bool
