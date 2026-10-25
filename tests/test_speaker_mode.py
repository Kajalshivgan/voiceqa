"""Speaker mode: audio played on the PC, outcome confirmed manually (no ADB)."""

import json

from run_pipeline import resolve_mode, run_pipeline
from voiceqa.generator.scenario_generator import generate_scenarios


def _two_scenarios():
    scenarios = generate_scenarios("voice-controlled climate control", dry_run=True)[:2]
    scenarios[1]["requires_adb"] = True
    return scenarios


def test_speaker_mode_uses_manual_verdict_and_skips_adb_only(tmp_path):
    played = []
    scenarios = _two_scenarios()

    results = run_pipeline(
        scenarios,
        mode="speaker",
        dry_run=True,
        output_dir=tmp_path,
        play_fn=played.append,
        confirm_fn=lambda scenario: (True, "temp changed"),
    )

    assert len(played) == 1  # the requires_adb scenario was skipped
    assert len(results) == 1
    assert results[0]["pass"] is True
    assert "manual confirmation" in results[0]["reasoning"]
    summary = json.loads((tmp_path / "results" / "summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 1


def test_speaker_mode_replay_then_fail(tmp_path):
    answers = iter([(False, "__replay__"), (False, "nothing happened")])
    played = []

    results = run_pipeline(
        _two_scenarios()[:1],
        mode="speaker",
        dry_run=True,
        output_dir=tmp_path,
        play_fn=played.append,
        confirm_fn=lambda scenario: next(answers),
    )

    assert len(played) == 2
    assert results[0]["pass"] is False


def test_auto_falls_back_to_speaker_without_adb(monkeypatch):
    monkeypatch.setattr("run_pipeline.AdbClient.is_available", staticmethod(lambda *a, **k: False))
    assert resolve_mode("auto") == "speaker"
