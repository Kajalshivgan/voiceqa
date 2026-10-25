"""Evaluator Agent — LLM-as-judge.

Compares expected vs. actual outcome and scores the result on
intent_correctness, ui_state_correctness, and response_naturalness. This is
the piece that goes beyond simple string matching: a real LLM judge can give
partial credit for "close but wrong" responses.

When no LLM credentials are configured (or ``dry_run=True``), falls back to
a deterministic rule-based scorer grounded in the actual scenario/response
data (word-overlap + intent match) rather than random mock output, so a
`--dry-run` report is still meaningfully tied to what the pipeline actually
did.
"""

from __future__ import annotations

import re

from voiceqa import config
from voiceqa.llm_client import LLMError, call_llm_json

SCORE_FIELDS = ("intent_correctness", "ui_state_correctness", "response_naturalness")


def build_prompt(scenario: dict, actual: dict) -> str:
    return f"""
Expected intent: {scenario['expected_intent']}
Expected UI state: {scenario['expected_ui_state']}
Actual UI hierarchy: {actual['ui_hierarchy']}
Actual spoken/text response: {actual['transcript']}

Score this response from 0-1 on:
- intent_correctness
- ui_state_correctness
- response_naturalness

Return JSON with those three scores, an overall "pass" boolean, and a short "reasoning" string.
Return ONLY the JSON object, no other text.
""".strip()


def _clamp01(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))


def _intent_score(scenario: dict, actual: dict) -> float:
    recognized = actual.get("recognized_intent")
    if recognized:
        return 1.0 if recognized == scenario["expected_intent"] else 0.0

    expected_words = _tokenize(scenario["expected_intent"])
    transcript_words = _tokenize(actual.get("transcript", ""))
    if not expected_words:
        return 0.5
    overlap = len(expected_words & transcript_words)
    return min(1.0, overlap / len(expected_words))


def _ui_score(scenario: dict, actual: dict) -> float:
    expected_tokens = _tokenize(scenario["expected_ui_state"])
    haystack_words = _tokenize(actual.get("ui_hierarchy", "")) | _tokenize(actual.get("screenshot", ""))
    if not expected_tokens:
        return 0.5
    hits = len(expected_tokens & haystack_words)
    return hits / len(expected_tokens)


def _naturalness_score(scenario: dict, actual: dict) -> float:
    transcript = (actual.get("transcript") or "").strip()
    if not transcript:
        return 0.0
    word_count = len(transcript.split())
    score = 0.85 if 2 <= word_count <= 30 else 0.5
    is_clarification = scenario["expected_intent"] == "clarification_needed"
    sounds_like_clarification = any(kw in transcript.lower() for kw in ("sorry", "didn't understand", "clarify", "did you mean"))
    if sounds_like_clarification and not is_clarification:
        score *= 0.6
    return score


def _fallback_score(scenario: dict, actual: dict) -> dict:
    intent = _intent_score(scenario, actual)
    ui = _ui_score(scenario, actual)
    naturalness = _naturalness_score(scenario, actual)
    overall = (intent + ui + naturalness) / 3
    return {
        "intent_correctness": round(intent, 2),
        "ui_state_correctness": round(ui, 2),
        "response_naturalness": round(naturalness, 2),
        "pass": overall >= 0.6,
        "reasoning": (
            f"[rule-based fallback] intent={intent:.2f} ui={ui:.2f} naturalness={naturalness:.2f} "
            "(no LLM credentials configured / --dry-run; set ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "for real LLM-as-judge scoring)"
        ),
    }


def evaluate_response(
    scenario: dict,
    actual: dict,
    provider: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Score one scenario's actual outcome against its expectations."""
    use_fallback = dry_run or not config.has_real_llm_credentials()
    if actual.get("manual_pass") is not None:
        # Tester confirmed the outcome by hand; no automated judging needed.
        score = 1.0 if actual["manual_pass"] else 0.0
        note = actual.get("transcript") or ""
        result = {
            "intent_correctness": score,
            "ui_state_correctness": score,
            "response_naturalness": score,
            "pass": bool(actual["manual_pass"]),
            "reasoning": "[manual confirmation] " + (note or ("tester confirmed" if score else "tester reported failure")),
        }
    elif use_fallback:
        result = _fallback_score(scenario, actual)
    else:
        prompt = build_prompt(scenario, actual)
        try:
            raw = call_llm_json(prompt, provider=provider, dry_run=False)
        except LLMError:
            result = _fallback_score(scenario, actual)
        else:
            if not isinstance(raw, dict):
                result = _fallback_score(scenario, actual)
            else:
                result = {field: _clamp01(raw.get(field, 0)) for field in SCORE_FIELDS}
                avg = sum(result.values()) / len(SCORE_FIELDS)
                result["pass"] = bool(raw.get("pass", avg >= 0.6))
                result["reasoning"] = str(raw.get("reasoning", ""))

    return {
        "scenario_id": scenario.get("id"),
        "command": scenario.get("command"),
        "category": scenario.get("category"),
        "expected_intent": scenario.get("expected_intent"),
        "expected_ui_state": scenario.get("expected_ui_state"),
        "transcript": actual.get("transcript"),
        "latency_ms": actual.get("latency_ms"),
        **result,
    }
