"""End-to-end smoke test: generate -> synthesize -> inject (mock) -> capture
-> evaluate (dry-run) -> report, with no network access and no optional
heavy dependencies required."""

import json

from voiceqa.generator.scenario_generator import generate_scenarios
from run_pipeline import run_pipeline


def test_full_pipeline_runs_end_to_end_in_mock_dry_run_mode(tmp_path):
    scenarios = generate_scenarios("voice-controlled climate control", dry_run=True)
    # Keep the smoke test fast: one scenario per category.
    seen_categories = set()
    subset = []
    for s in scenarios:
        if s["category"] not in seen_categories:
            subset.append(s)
            seen_categories.add(s["category"])

    results = run_pipeline(subset, mode="mock", dry_run=True, output_dir=tmp_path)

    assert len(results) == len(subset)
    for r in results:
        assert "pass" in r
        assert isinstance(r["latency_ms"], int)

    results_json = tmp_path / "results" / "results.json"
    summary_json = tmp_path / "results" / "summary.json"
    assert results_json.exists()
    assert summary_json.exists()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["total"] == len(subset)
    assert "nl_summary" in summary
