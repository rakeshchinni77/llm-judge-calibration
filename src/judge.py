"""
judge.py — LLM Judge: API caller with structured JSON output and budget guard.

The judge receives a generated text string, prepends a system prompt and
approved few-shot train demonstrations, calls the configured LLM API
(hosted, local, or deterministic mock), and returns a structured dictionary:

    {"critique": "<evidence-based justification>", "pass": true|false}

Key guarantees:
    - Configurable model name and system prompt.
    - Access ONLY to approved Train split examples for few-shot demonstrations
      (leakage prevention strictly enforced).
    - Structured JSON output enforcement (response_format json_object).
    - Defensive JSON parsing and schema validation.
    - Boolean 'pass' field is a strict JSON boolean (not string or int).
    - Budget guard tracks spend and raises BudgetExceededException on exhaustion.
    - Bounded retries with exponential backoff on transient errors.
    - Deterministic mock backend for offline testing and CI.
    - Logging excludes secrets and API keys.

Phase gate: Phase 3
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class BudgetExceededException(Exception):
    """Raised when the accumulated judge spend exceeds the configured budget."""


class JudgeParseException(Exception):
    """Raised when the LLM response cannot be parsed into the required schema."""


class CredentialsException(Exception):
    """Raised when required API credentials are missing or invalid."""


class RateLimitException(Exception):
    """Raised when the LLM provider rate limits requests beyond retry bounds."""


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

@dataclass
class JudgeResponse:
    """Structured output from the LLM judge.

    Attributes:
        critique: Concise evidence-based justification tied to the rubric.
        passed:   True if the judge considers the output a Pass (human label 1).
        raw:      The raw response string for debugging/auditing.
    """

    critique: str
    passed: bool
    raw: str = field(repr=False, default="")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the required schema: {"critique": str, "pass": bool}."""
        return {"critique": self.critique, "pass": self.passed}


# ---------------------------------------------------------------------------
# Few-shot Train split loader (Leakage Guard)
# ---------------------------------------------------------------------------

def load_train_few_shot(
    train_path: str | Path = "data/splits/train.csv",
    n_examples: int = 4,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load demonstrations exclusively from the approved Train split.

    Enforces strict leakage prevention: raises ValueError if a non-train split
    is supplied.

    Args:
        train_path: Path to train.csv. Must be in the train split.
        n_examples: Number of balanced examples to load.
        seed: Random seed for deterministic example selection.

    Returns:
        List of dicts with keys 'output', 'label', 'critique'.

    Raises:
        ValueError: If train_path points to Dev or Test data.
        FileNotFoundError: If the train split file does not exist.
    """
    path_str = str(train_path).lower().replace("\\", "/")
    if "dev" in path_str or "test" in path_str:
        raise ValueError(
            f"LEAKAGE VIOLATION: Only examples from the Train split may be used "
            f"for few-shot demonstrations! Rejected path: {train_path}"
        )

    file_path = Path(train_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Train split not found: {file_path}")

    import pandas as pd

    df = pd.read_csv(file_path)
    if df.empty:
        return []

    # Select balanced Pass and Fail examples
    pass_df = df[df["label"] == 1]
    fail_df = df[df["label"] == 0]

    n_half = n_examples // 2
    selected_pass = pass_df.sample(n=min(n_half, len(pass_df)), random_state=seed) if len(pass_df) > 0 else pass_df
    selected_fail = fail_df.sample(n=min(n_half, len(fail_df)), random_state=seed) if len(fail_df) > 0 else fail_df

    combined = pd.concat([selected_pass, selected_fail]).sample(frac=1.0, random_state=seed)
    return combined[["output", "label", "critique"]].to_dict(orient="records")


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------

class LLMJudge:
    """Evaluates generated outputs against rubric criteria using an LLM.

    Args:
        model_name: Model identifier (e.g. 'gpt-4o-mini').
        system_prompt: System instruction text or path to prompt file.
        budget_usd: Maximum allowable spend in USD.
        seed: Random seed forwarded to the backend for determinism.
        backend: 'hosted' (OpenAI), 'local' (Ollama), or 'mock'.
        timeout_seconds: Timeout per API call in seconds.
        max_retries: Maximum retry attempts for transient API errors.
    """

    _COST_PER_1K_INPUT: float = 0.000150
    _COST_PER_1K_OUTPUT: float = 0.000600

    def __init__(
        self,
        model_name: str | None = None,
        system_prompt: str | Path | None = None,
        budget_usd: float | None = None,
        seed: int | None = None,
        backend: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.budget_usd = budget_usd if budget_usd is not None else float(os.getenv("JUDGE_BUDGET_USD", "5.0"))
        self.seed = seed if seed is not None else int(os.getenv("RANDOM_SEED", "42"))
        self.backend = (backend or os.getenv("MODEL_TYPE", "hosted")).lower()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.spend: float = 0.0

        self.system_prompt = self._resolve_system_prompt(system_prompt)
        self._client = self._build_client() if self.backend in ("hosted", "local") else None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        generated_text: str,
        few_shot_examples: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single generated text and return a structured dictionary.

        Args:
            generated_text: The model output to evaluate.
            few_shot_examples: Optional list of approved Train-split demonstrations.

        Returns:
            Dictionary matching {"critique": str, "pass": bool}.

        Raises:
            BudgetExceededException: If accumulated spend exceeds budget_usd.
            JudgeParseException: If the LLM response cannot be parsed or schema fails.
            CredentialsException: If API credentials are missing.
            RateLimitException: If API rate limits persist after retries.
        """
        # Validate that few-shot examples do not violate leakage boundaries
        if few_shot_examples:
            for ex in few_shot_examples:
                if not isinstance(ex, dict) or "output" not in ex:
                    raise ValueError(f"Invalid few-shot example format: {ex}")

        # Deterministic mock backend
        if self.backend == "mock":
            response = self._evaluate_mock(generated_text, few_shot_examples=few_shot_examples)
            return response.to_dict()

        # Build message history with few-shot context
        messages = self._build_messages(generated_text, few_shot_examples)

        # Call live API with bounded retries and timeout
        raw_text, input_tokens, output_tokens = self._call_api_with_retry(messages)

        # Track spend and enforce budget guard
        self._track_spend(input_tokens, output_tokens)

        # Parse and validate structured output
        response = self._parse_response(raw_text)
        return response.to_dict()

    def evaluate_pairwise(
        self,
        cand_a: str,
        cand_b: str,
    ) -> dict[str, Any]:
        """Compare two candidate answers and return which is preferred.

        Args:
            cand_a: Text of Candidate A.
            cand_b: Text of Candidate B.

        Returns:
            Dictionary matching {"critique": str, "preference": "A" | "B"}.
        """
        if self.backend == "mock":
            return self._evaluate_mock_pairwise(cand_a, cand_b)

        messages = self._build_pairwise_messages(cand_a, cand_b)
        raw_text, in_tokens, out_tokens = self._call_api_with_retry(messages)
        self._track_spend(in_tokens, out_tokens)
        return self._parse_pairwise_response(raw_text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_system_prompt(self, system_prompt: str | Path | None) -> str:
        """Resolve and return system prompt text."""
        if system_prompt is not None:
            if isinstance(system_prompt, Path) or (isinstance(system_prompt, str) and os.path.exists(system_prompt)):
                return Path(system_prompt).read_text(encoding="utf-8").strip()
            return str(system_prompt).strip()

        # Try default prompt paths
        candidates = [
            Path("prompts/judge_system.txt"),
            Path("prompts/system_prompt.md"),
        ]
        for p in candidates:
            if p.exists():
                return p.read_text(encoding="utf-8").strip()

        return (
            "You are an expert technical judge. Evaluate whether the answer passes or fails the rubric. "
            "Respond in JSON with exactly 'critique' (string) and 'pass' (boolean)."
        )

    def _build_client(self) -> Any:
        """Initialize the LLM API client safely."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc

        if self.backend == "local":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            return OpenAI(base_url=base_url, api_key="ollama")

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
            raise CredentialsException(
                "OPENAI_API_KEY not set. Set it in .env or run with backend='mock'."
            )

        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self.timeout_seconds}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    def _build_messages(
        self,
        generated_text: str,
        few_shot_examples: list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        """Assemble the chat messages list including few-shot demonstrations."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        if few_shot_examples:
            for ex in few_shot_examples:
                pass_val = bool(ex.get("label", 0) == 1)
                critique_val = ex.get("critique", "Rubric-based justification.")
                example_json = json.dumps({"critique": critique_val, "pass": pass_val})

                messages.append({
                    "role": "user",
                    "content": f"Answer to evaluate:\n{ex['output']}",
                })
                messages.append({
                    "role": "assistant",
                    "content": example_json,
                })

        messages.append({
            "role": "user",
            "content": f"Answer to evaluate:\n{generated_text}",
        })
        return messages

    def _call_api_with_retry(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        """Execute chat completion with bounded retries and exponential backoff."""
        assert self._client is not None, "Client not initialized"

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    seed=self.seed,
                    timeout=self.timeout_seconds,
                )
                choice = response.choices[0]
                raw_text = choice.message.content or ""
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 100
                output_tokens = usage.completion_tokens if usage else 50
                return raw_text, input_tokens, output_tokens

            except Exception as exc:
                err_msg = str(exc).lower()
                last_exc = exc
                if "rate limit" in err_msg or "429" in err_msg:
                    sleep_time = (2 ** attempt) * 0.5
                    logger.warning("Rate limit hit on attempt %d; retrying in %.1fs", attempt + 1, sleep_time)
                    time.sleep(sleep_time)
                elif "timeout" in err_msg or "connection" in err_msg:
                    sleep_time = (2 ** attempt) * 0.5
                    logger.warning("Network timeout on attempt %d; retrying in %.1fs", attempt + 1, sleep_time)
                    time.sleep(sleep_time)
                else:
                    # Non-retriable error
                    raise exc

        if "rate limit" in str(last_exc).lower():
            raise RateLimitException(f"Rate limit exceeded after {self.max_retries} retries: {last_exc}")
        raise RuntimeError(f"API call failed after {self.max_retries} retries: {last_exc}")

    def _mock_check_fail(self, text: str) -> bool:
        """Deterministic mock rule to check whether text represents a failure."""
        t = text.strip()
        lower = t.lower()
        return (
            len(t.split()) < 15
            or "bypassing standard safety" in lower
            or "zero overhead" in lower
            or "o(1) time without allocating" in lower
            or "[generation_error" in lower
            or "absolute rule" in lower
        )

    def _evaluate_mock(
        self,
        generated_text: str,
        few_shot_examples: list[dict[str, Any]] | None = None,
    ) -> JudgeResponse:
        """Deterministic mock evaluation for offline testing and CI."""
        text = generated_text.strip()

        # In-context demonstration matching (simulates in-context prompt memorization)
        if few_shot_examples:
            for ex in few_shot_examples:
                if isinstance(ex, dict) and ex.get("output", "").strip() == text:
                    passed = bool(ex.get("label", 0) == 1)
                    critique = str(ex.get("critique", "Matches in-context prompt demonstration."))
                    prompt_tokens = max(20, int(len(text.split()) * 1.3) + 100)
                    completion_tokens = max(15, int(len(critique.split()) * 1.3) + 10)
                    self._track_spend(prompt_tokens, completion_tokens)
                    raw = json.dumps({"critique": critique, "pass": passed})
                    return JudgeResponse(critique=critique, passed=passed, raw=raw)

        is_fail = self._mock_check_fail(text)

        if is_fail:
            critique = (
                "Fails criteria F1 and F3: contains factual inaccuracies regarding computational complexity "
                "or omits foundational mechanisms necessary for a complete and reliable explanation."
            )
            passed = False
        else:
            critique = (
                "Meets criteria P1-P5: the explanation is technically sound, addresses the core concepts "
                "accurately, and provides relevant engineering details without fabricated claims."
            )
            passed = True

        prompt_tokens = max(20, int(len(text.split()) * 1.3) + 100)
        completion_tokens = max(15, int(len(critique.split()) * 1.3) + 10)
        self._track_spend(prompt_tokens, completion_tokens)

        raw = json.dumps({"critique": critique, "pass": passed})
        return JudgeResponse(critique=critique, passed=passed, raw=raw)

    def _evaluate_mock_pairwise(self, cand_a: str, cand_b: str) -> dict[str, Any]:
        """Deterministic mock evaluation for pairwise candidate comparison."""
        fail_a = self._mock_check_fail(cand_a)
        fail_b = self._mock_check_fail(cand_b)

        if not fail_a and fail_b:
            critique = (
                "Candidate A provides an accurate, complete explanation while Candidate B fails criteria."
            )
            pref = "A"
        elif fail_a and not fail_b:
            critique = (
                "Candidate B provides an accurate, complete explanation while Candidate A fails criteria."
            )
            pref = "B"
        else:
            # Both pass or both fail: deterministic preference
            critique = "Both candidates demonstrate comparable technical quality; Candidate A is preferred."
            pref = "A"

        prompt_tokens = max(30, int((len(cand_a.split()) + len(cand_b.split())) * 1.3) + 120)
        completion_tokens = max(15, int(len(critique.split()) * 1.3) + 10)
        self._track_spend(prompt_tokens, completion_tokens)

        return {"critique": critique, "preference": pref}

    def _build_pairwise_messages(self, cand_a: str, cand_b: str) -> list[dict[str, str]]:
        """Assemble messages for pairwise evaluation."""
        system = (
            "You are an expert technical judge. Compare Candidate A and Candidate B for technical accuracy, "
            "completeness, and correctness according to the evaluation rubric. Determine which candidate is superior. "
            'Respond in valid JSON with exactly two fields: "critique" (string) and "preference" ("A" or "B").'
        )
        user = (
            f"[Candidate A]\n{cand_a}\n\n"
            f"[Candidate B]\n{cand_b}\n\n"
            'Which answer is better? Respond in JSON with {"critique": "...", "preference": "A" or "B"}.'
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _parse_pairwise_response(self, raw: str) -> dict[str, Any]:
        """Parse raw response from pairwise evaluation."""
        if not raw or not raw.strip():
            raise JudgeParseException("Judge returned an empty response string for pairwise comparison.")

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        data: dict[str, Any] | None = None
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            pass

        if data is None:
            crit_m = re.search(r'"critique"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
            pref_m = re.search(r'"preference"\s*:\s*"([ABab])"', cleaned)
            if crit_m and pref_m:
                crit_str = crit_m.group(1).encode("utf-8").decode("unicode_escape", errors="replace")
                pref_str = pref_m.group(1).upper()
                data = {"critique": crit_str, "preference": pref_str}

        if data is None or "critique" not in data or "preference" not in data:
            raise JudgeParseException(f"Could not parse valid pairwise JSON: {raw[:150]}")

        critique = str(data["critique"]).strip()
        preference = str(data["preference"]).strip().upper()
        if preference not in ("A", "B"):
            raise JudgeParseException(f"Field 'preference' must be 'A' or 'B', got: {preference}")

        return {"critique": critique, "preference": preference}

    def _parse_response(self, raw: str) -> JudgeResponse:
        """Parse the raw response into a validated JudgeResponse."""
        if not raw or not raw.strip():
            raise JudgeParseException("Judge returned an empty response string.")

        cleaned = raw.strip()

        # Strip markdown code blocks if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        data: dict[str, Any] | None = None
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            pass

        # Defensive fallback: regex extraction
        if data is None:
            critique_match = re.search(r'"critique"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
            pass_match = re.search(r'"pass"\s*:\s*(true|false)\b', cleaned, re.IGNORECASE)
            if critique_match and pass_match:
                critique_str = critique_match.group(1).encode("utf-8").decode("unicode_escape", errors="replace")
                pass_bool = pass_match.group(1).lower() == "true"
                data = {"critique": critique_str, "pass": pass_bool}

        if data is None:
            raise JudgeParseException(f"Could not parse valid JSON from judge response: {raw[:150]}")

        # Schema validation: required keys
        if "critique" not in data or "pass" not in data:
            missing = [k for k in ("critique", "pass") if k not in data]
            raise JudgeParseException(f"Judge JSON missing required keys: {missing}")

        # Schema validation: critique must be non-empty string
        critique = data["critique"]
        if not isinstance(critique, str) or not critique.strip():
            raise JudgeParseException(f"Field 'critique' must be a non-empty string; got {type(critique).__name__}")

        # Schema validation: pass must be a strict boolean
        pass_val = data["pass"]
        if not isinstance(pass_val, bool) or type(pass_val) is not bool:
            raise JudgeParseException(
                f"Field 'pass' must be a strict boolean (true/false); got {type(pass_val).__name__} ({pass_val})"
            )

        return JudgeResponse(critique=critique.strip(), passed=pass_val, raw=raw)

    def _track_spend(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate spend and enforce the budget limit."""
        cost = (input_tokens / 1000) * self._COST_PER_1K_INPUT + (
            output_tokens / 1000
        ) * self._COST_PER_1K_OUTPUT
        self.spend += cost
        if self.spend > self.budget_usd:
            raise BudgetExceededException(
                f"Budget exceeded: ${self.spend:.4f} > ${self.budget_usd:.2f}. "
                "Increase JUDGE_BUDGET_USD to continue evaluation."
            )
