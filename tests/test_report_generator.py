import json

from voiceqa.reporting.report_generator import aggregate_results, save_results


RESULTS = [
    {"scenario_id": "scn_000", "command": "a", "category": "happy_path", "pass": True, "latency_ms": 200,
     "intent_correctness": 1.0, "ui_state_correctness": 1.0, "response_naturalness": 0.9, "reasoning": "ok"},
    {"scenario_id": "scn_001", "command": "b", "category": "happy_path", "pass": True, "latency_ms": 250,
     "intent_correctness": 1.0, "ui_state_correctness": 1.0, "response_naturalness": 0.9, "reasoning": "ok"},
    {"scenario_id": "scn_002", "command": "c", "category": "adversarial", "pass": False, "latency_ms": 400,
     "intent_correctness": 0.0, "ui_state_correctness": 0.3, "response_naturalness": 0.5, "reasoning": "wrong intent"},
]


def test_aggregate_results_computes_overall_and_per_category_rates():
    summary = aggregate_results(RESULTS)
    assert summary["total"] == 3
    assert summary["overall_pass_rate"] == round(2 / 3, 4)
    assert summary["by_category"]["happy_path"]["pass_rate"] == 1.0
    assert summary["by_category"]["adversarial"]["pass_rate"] == 0.0
    assert len(summary["failures"]) == 1


def test_aggregate_results_handles_empty_list():
    summary = aggregate_results([])
    assert summary["total"] == 0


def test_save_results_writes_expected_files(tmp_path):
    paths = save_results(RESULTS, tmp_path, dry_run=True)
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "summary.json").exists()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "nl_summary" in summary
    assert summary["total"] == 3
