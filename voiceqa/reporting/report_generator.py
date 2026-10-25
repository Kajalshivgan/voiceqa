"""Aggregates per-scenario evaluation results into a categorized report,
persists them, and produces an auto-generated natural-language summary.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from voiceqa import config
from voiceqa.llm_client import LLMError, call_llm

RESULT_FIELDS = [
    "scenario_id",
    "command",
    "category",
    "expected_intent",
    "expected_ui_state",
    "transcript",
    "latency_ms",
    "intent_correctness",
    "ui_state_correctness",
    "response_naturalness",
    "pass",
    "reasoning",
]


def aggregate_results(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {"total": 0, "overall_pass_rate": 0.0, "by_category": {}, "latency_ms": {}}

    passed = sum(1 for r in results if r.get("pass"))

    by_category: dict[str, dict] = {}
    for r in results:
        cat = r.get("category", "unknown")
        bucket = by_category.setdefault(cat, {"total": 0, "pass": 0})
        bucket["total"] += 1
        if r.get("pass"):
            bucket["pass"] += 1
    for bucket in by_category.values():
        bucket["pass_rate"] = round(bucket["pass"] / bucket["total"], 4)

    latencies = [r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), (int, float))]
    latency_stats = {}
    if latencies:
        sorted_lat = sorted(latencies)
        latency_stats = {
            "min": min(sorted_lat),
            "max": max(sorted_lat),
            "mean": round(statistics.mean(sorted_lat), 1),
            "p50": sorted_lat[len(sorted_lat) // 2],
            "p95": sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))],
        }

    failures = [
        {"scenario_id": r.get("scenario_id"), "command": r.get("command"), "category": r.get("category"), "reasoning": r.get("reasoning")}
        for r in results
        if not r.get("pass")
    ]

    return {
        "total": total,
        "overall_pass_rate": round(passed / total, 4),
        "by_category": by_category,
        "latency_ms": latency_stats,
        "failures": failures,
    }


def _template_summary(summary: dict) -> str:
    if summary["total"] == 0:
        return "No results to summarize."

    by_cat = summary["by_category"]
    if not by_cat:
        return f"Overall pass rate: {summary['overall_pass_rate']:.0%} across {summary['total']} scenarios."

    best_cat = max(by_cat, key=lambda c: by_cat[c]["pass_rate"])
    worst_cat = min(by_cat, key=lambda c: by_cat[c]["pass_rate"])
    delta = (by_cat[best_cat]["pass_rate"] - by_cat[worst_cat]["pass_rate"]) * 100

    sentence = (
        f"Overall pass rate is {summary['overall_pass_rate']:.0%} across {summary['total']} scenarios. "
    )
    if delta > 1:
        sentence += (
            f"The '{worst_cat}' category lags {delta:.0f} points behind '{best_cat}' "
            f"({by_cat[worst_cat]['pass_rate']:.0%} vs {by_cat[best_cat]['pass_rate']:.0%}), "
            "suggesting that's where recognition/response quality is weakest."
        )
    else:
        sentence += "Pass rates are roughly even across categories."

    if summary.get("latency_ms"):
        sentence += f" Median response latency was {summary['latency_ms']['p50']}ms."

    return sentence


def generate_nl_summary(summary: dict, provider: str | None = None, dry_run: bool = False) -> str:
    """Auto-generate a natural-language summary of the categorized results."""
    if dry_run or not config.has_real_llm_credentials():
        return _template_summary(summary)

    prompt = f"""
Given this categorized voice-assistant test summary, write a short
(2-4 sentence) natural-language executive summary highlighting the most
notable failure pattern, referencing specific categories and pass rates.

Summary data:
{json.dumps(summary, indent=2)}

Return plain text only, no markdown, no JSON.
""".strip()
    try:
        return call_llm(prompt, provider=provider, dry_run=False).strip()
    except LLMError:
        return _template_summary(summary)


def save_results(
    results: list[dict],
    output_dir: str | Path,
    provider: str | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Write results.json, results.csv, and summary.json. Returns their paths."""
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    results_json_path = output_dir_p / "results.json"
    results_json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    results_csv_path = output_dir_p / "results.csv"
    with results_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    summary = aggregate_results(results)
    summary["nl_summary"] = generate_nl_summary(summary, provider=provider, dry_run=dry_run)

    summary_path = output_dir_p / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "results_json": str(results_json_path),
        "results_csv": str(results_csv_path),
        "summary_json": str(summary_path),
    }
