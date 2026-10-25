"""Unified LLM client used by both the scenario generator and the evaluator agent.

Supports three providers, selected via ``LLM_PROVIDER`` (or an explicit
``provider=`` argument):

- ``anthropic`` — Claude via the ``anthropic`` SDK
- ``openai``    — GPT-4o (or similar) via the ``openai`` SDK
- ``mock``      — deterministic, offline responses. Used automatically when
  no API key is configured, and directly by tests / ``--dry-run`` so the
  whole pipeline can be exercised without network access or billing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

from voiceqa import config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when an LLM call fails after all retries."""


def _extract_json(text: str) -> Any:
    """Pull a JSON value out of an LLM response, tolerating markdown fences."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to grabbing the first balanced [...] or {...} block.
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise LLMError(f"Could not parse JSON from LLM response: {text[:200]!r}")


def _call_anthropic(prompt: str, system: str | None) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise LLMError(
            "The 'anthropic' package is not installed. Run `pip install anthropic` "
            "or use --dry-run / LLM_PROVIDER=mock."
        ) from exc

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    kwargs: dict[str, Any] = {
        "model": config.ANTHROPIC_MODEL,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return "".join(block.text for block in response.content if block.type == "text")


def _call_openai(prompt: str, system: str | None) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMError(
            "The 'openai' package is not installed. Run `pip install openai` "
            "or use --dry-run / LLM_PROVIDER=mock."
        ) from exc

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=config.OPENAI_MODEL, messages=messages)
    return response.choices[0].message.content or ""


def _seed_from(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _mock_response(prompt: str) -> str:
    """Produce a plausible, deterministic JSON payload for offline/dry-run use.

    Inspects the prompt for recognizable markers from scenario_generator.py
    and evaluator_agent.py so both call sites get sensible fake data without
    any provider-specific branching in the callers.
    """
    seed = _seed_from(prompt)

    if '"expected_intent"' in prompt or "Generate voice command test scenarios" in prompt:
        return json.dumps(_mock_scenarios(prompt, seed))

    if "intent_correctness" in prompt or "Score this response" in prompt:
        return json.dumps(_mock_evaluation(prompt, seed))

    if "natural-language summary" in prompt.lower() or "auto-generate a" in prompt.lower():
        return _mock_summary_text(prompt, seed)

    # Generic fallback: echo a trivial object so callers relying on
    # _extract_json still get something parseable.
    return json.dumps({"mock": True, "seed": seed})


def _mock_scenarios(prompt: str, seed: int) -> list[dict]:
    feature_match = re.search(r"feature:\s*(.+)", prompt)
    feature = feature_match.group(1).strip() if feature_match else "the target feature"

    templates = {
        "happy_path": [
            ("Set temperature to 22 degrees", "set_temperature", "climate screen shows 22°C"),
            ("Turn the fan up", "increase_fan_speed", "fan speed indicator increases by one step"),
            ("Turn on the air conditioning", "ac_on", "AC icon shown as active"),
        ],
        "paraphrased": [
            ("Make it warmer", "increase_temperature", "temperature increases by 1-2°C"),
            ("22 please", "set_temperature", "climate screen shows 22°C"),
            ("A bit cooler in here", "decrease_temperature", "temperature decreases by 1-2°C"),
        ],
        "edge_case": [
            ("Set the- uh, set temperature to 22", "set_temperature", "climate screen shows 22°C"),
            ("[road noise] set temperature to twenty two degrees", "set_temperature", "climate screen shows 22°C"),
            ("Mach die Temperatur cooler please", "decrease_temperature", "temperature decreases by 1-2°C"),
        ],
        "adversarial": [
            ("Set temperature to 222 degrees", "clarification_needed", "system asks for a valid temperature range"),
            ("Set temperature to negative 5", "clarification_needed", "system rejects out-of-range value"),
            ("Set the volume to 22 degrees", "clarification_needed", "system asks user to clarify intent"),
        ],
    }

    scenarios = []
    for category, items in templates.items():
        for command, intent, ui_state in items:
            scenarios.append(
                {
                    "command": command,
                    "category": category,
                    "expected_intent": intent,
                    "expected_ui_state": f"For '{feature}': {ui_state}",
                }
            )
    return scenarios


def _mock_evaluation(prompt: str, seed: int) -> dict:
    # Deterministic-but-varied pass/fail so a mock report still looks realistic.
    intent_score = round(0.6 + (seed % 40) / 100, 2)
    ui_score = round(0.5 + (seed % 50) / 100, 2)
    naturalness = round(0.7 + (seed % 30) / 100, 2)
    overall = (intent_score + ui_score + naturalness) / 3
    return {
        "intent_correctness": min(intent_score, 1.0),
        "ui_state_correctness": min(ui_score, 1.0),
        "response_naturalness": min(naturalness, 1.0),
        "pass": overall >= 0.7,
        "reasoning": "[mock evaluator] deterministic score derived from response hash; "
        "set ANTHROPIC_API_KEY / OPENAI_API_KEY and drop --dry-run for real judging.",
    }


def _mock_summary_text(prompt: str, seed: int) -> str:
    return (
        "[mock summary] Voice recognition performance looks stable overall in this dry run. "
        "Set a real LLM provider (ANTHROPIC_API_KEY or OPENAI_API_KEY) and re-run without "
        "--dry-run to get an actual data-driven narrative summary."
    )


def call_llm(
    prompt: str,
    system: str | None = None,
    provider: str | None = None,
    dry_run: bool = False,
) -> str:
    """Call the configured LLM and return the raw text response."""
    provider = provider or config.LLM_PROVIDER

    if dry_run or provider == "mock" or not config.has_real_llm_credentials():
        if not dry_run and provider != "mock":
            logger.warning(
                "No credentials found for provider=%s; falling back to mock LLM responses.",
                provider,
            )
        return _mock_response(prompt)

    dispatch = {"anthropic": _call_anthropic, "openai": _call_openai}
    if provider not in dispatch:
        raise LLMError(f"Unknown LLM provider: {provider!r}")

    last_error: Exception | None = None
    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        try:
            return dispatch[provider](prompt, system)
        except Exception as exc:  # noqa: BLE001 - want to retry on any transient failure
            last_error = exc
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, config.LLM_MAX_RETRIES, exc)
            if attempt < config.LLM_MAX_RETRIES:
                time.sleep(min(2**attempt, 10))
    raise LLMError(f"LLM call failed after {config.LLM_MAX_RETRIES} attempts") from last_error


def call_llm_json(
    prompt: str,
    system: str | None = None,
    provider: str | None = None,
    dry_run: bool = False,
) -> Any:
    """Call the configured LLM and parse the response as JSON."""
    raw = call_llm(prompt, system=system, provider=provider, dry_run=dry_run)
    return _extract_json(raw)
