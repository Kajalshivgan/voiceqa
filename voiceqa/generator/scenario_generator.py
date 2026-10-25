"""Scenario Generator Agent.

Takes a plain-language feature description and produces a structured set of
voice command test cases spanning four categories: happy_path, paraphrased,
edge_case, and adversarial.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from voiceqa import config
from voiceqa.llm_client import LLMError, call_llm_json

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"command", "category", "expected_intent", "expected_ui_state"}
VALID_CATEGORIES = {"happy_path", "paraphrased", "edge_case", "adversarial"}


def build_prompt(feature_description: str, min_per_category: int = 3) -> str:
    return f"""
Generate voice command test scenarios for this feature: {feature_description}

Produce a JSON list of objects, each with:
- "command": the spoken text
- "category": one of "happy_path", "paraphrased", "edge_case", "adversarial"
- "expected_intent": the intent this should trigger
- "expected_ui_state": what the UI should show after this command

Include at least {min_per_category} examples per category. Cover realistic
variation within edge_case (background noise mentioned mid-command,
interruptions/false starts, code-switching between German and English) and
adversarial (similar-sounding but incorrect commands, out-of-range values,
ambiguous phrasing that should trigger clarification rather than a silent
guess).

Return ONLY the JSON list, no other text.
""".strip()


def validate_scenarios(scenarios: Any) -> list[dict]:
    """Validate the LLM output shape, raising ValueError with a clear message on failure."""
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Expected a non-empty JSON list of scenario objects.")

    cleaned = []
    for i, item in enumerate(scenarios):
        if not isinstance(item, dict):
            raise ValueError(f"Scenario #{i} is not an object: {item!r}")
        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            raise ValueError(f"Scenario #{i} is missing fields: {missing}")
        if item["category"] not in VALID_CATEGORIES:
            raise ValueError(
                f"Scenario #{i} has invalid category {item['category']!r}; "
                f"expected one of {VALID_CATEGORIES}"
            )
        cleaned.append(
            {
                "id": f"scn_{i:03d}",
                "command": str(item["command"]),
                "category": item["category"],
                "expected_intent": str(item["expected_intent"]),
                "expected_ui_state": str(item["expected_ui_state"]),
            }
        )
    return cleaned


def generate_scenarios(
    feature_description: str,
    min_per_category: int = 3,
    provider: str | None = None,
    dry_run: bool = False,
) -> list[dict]:
    prompt = build_prompt(feature_description, min_per_category)
    try:
        raw = call_llm_json(prompt, provider=provider, dry_run=dry_run)
    except LLMError as exc:
        raise LLMError(f"Scenario generation failed for feature={feature_description!r}: {exc}") from exc
    return validate_scenarios(raw)


def save_scenarios(scenarios: list[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scenarios, indent=2), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate voice command test scenarios via LLM.")
    parser.add_argument("--feature", required=True, help="Plain-language feature description.")
    parser.add_argument(
        "--output",
        default=str(config.SCENARIOS_DIR / "scenarios.json"),
        help="Where to write the generated scenarios JSON.",
    )
    parser.add_argument("--min-per-category", type=int, default=3)
    parser.add_argument("--provider", choices=["anthropic", "openai", "mock"], default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use deterministic mock scenarios instead of calling a real LLM.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    scenarios = generate_scenarios(
        args.feature,
        min_per_category=args.min_per_category,
        provider=args.provider,
        dry_run=args.dry_run,
    )
    output_path = save_scenarios(scenarios, args.output)

    by_category: dict[str, int] = {}
    for s in scenarios:
        by_category[s["category"]] = by_category.get(s["category"], 0) + 1

    print(f"Generated {len(scenarios)} scenarios -> {output_path}")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
