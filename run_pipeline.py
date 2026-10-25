"""Runs the full VoiceQA pipeline: synthesize -> inject -> capture -> evaluate -> report.

Examples:

    # No device, no API key, no heavy deps required:
    python run_pipeline.py --scenarios voiceqa/scenarios/climate.json --mock --dry-run

    # Mock target, real LLM-as-judge:
    python run_pipeline.py --scenarios voiceqa/scenarios/climate.json --mock

    # Real AAOS emulator/device:
    python run_pipeline.py --scenarios voiceqa/scenarios/climate.json --device emulator-5554

    # Pick a different voice:
    python run_pipeline.py --list-voices
    python run_pipeline.py --scenarios voiceqa/scenarios/assistant.json --speaker --voice zira

    # No ADB: play audio on this PC's speaker, confirm the result by hand:
    python run_pipeline.py --scenarios voiceqa/scenarios/climate.json --speaker

    # ADB if a device is connected, otherwise speaker + manual confirmation:
    python run_pipeline.py --scenarios voiceqa/scenarios/climate.json --auto
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from voiceqa import config
from voiceqa.capture.response_capture import capture_response
from voiceqa.evaluation.evaluator_agent import evaluate_response
from voiceqa.injection.adb_client import AdbClient, AdbError
from voiceqa.injection.audio_injector import AudioInjector
from voiceqa.mock_target.mock_assistant import MockClimateAssistant
from voiceqa.reporting.report_generator import save_results
from voiceqa.synthesis import variation
from voiceqa.synthesis.tts import list_voices, synthesize

logger = logging.getLogger(__name__)


def load_scenarios(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"No scenarios found in {path}")
    return data


def resolve_mode(mode: str, device_serial: str | None = None) -> str:
    """Resolve ``auto`` to ``adb`` when a device is reachable, else ``speaker``."""
    if mode != "auto":
        return mode
    try:
        if AdbClient.is_available() and AdbClient(serial=device_serial).devices():
            return "adb"
    except AdbError:
        pass
    return "speaker"


def run_pipeline(
    scenarios: list[dict],
    mode: str,
    device_serial: str | None = None,
    provider: str | None = None,
    dry_run: bool = False,
    output_dir: str | Path | None = None,
    progress_cb=None,
    play_fn=None,
    confirm_fn=None,
    voice: str | None = None,
) -> list[dict]:
    """``mode`` is one of ``mock``, ``adb``, ``speaker`` (play on this PC's
    speaker + manual confirmation) or ``auto`` (adb if a device is connected,
    otherwise speaker). Scenarios with ``"requires_adb": true`` are skipped
    when the resolved mode is not ``adb``."""
    output_dir_p = Path(output_dir) if output_dir else config.OUTPUT_DIR
    audio_dir = output_dir_p / "audio"
    capture_dir = output_dir_p / "capture"
    results_dir = output_dir_p / "results"

    mode = resolve_mode(mode, device_serial)
    print(f"Resolved mode: {mode}")

    adb_client = None
    if mode == "adb":
        adb_client = AdbClient(serial=device_serial)

    injector = AudioInjector(
        mode=mode,
        adb_client=adb_client,
        mock_assistant=MockClimateAssistant(),
        play_fn=play_fn,
        confirm_fn=confirm_fn,
    )

    results = []
    for i, scenario in enumerate(scenarios, start=1):
        scenario.setdefault("id", f"scn_{i:03d}")
        if scenario.get("requires_adb") and mode != "adb":
            print(f"[{i}/{len(scenarios)}] SKIP  {scenario['id']} (requires ADB, mode={mode})")
            continue

        raw_audio_path = str(audio_dir / f"{scenario['id']}_raw.wav")
        synthesize(scenario, raw_audio_path, voice=voice)

        policy = variation.CATEGORY_VARIATION.get(scenario["category"], {})
        noise_applied = policy.get("noise_db") is not None
        varied_audio_path = variation.apply_variation(scenario, raw_audio_path, str(audio_dir))

        injection_result = injector.inject(scenario, varied_audio_path, noise_applied=noise_applied)
        actual = capture_response(injection_result, str(capture_dir), adb_client=adb_client)
        evaluation = evaluate_response(scenario, actual, provider=provider, dry_run=dry_run)

        results.append(evaluation)
        if progress_cb:
            progress_cb(i, len(scenarios), evaluation)
        else:
            status = "PASS" if evaluation["pass"] else "FAIL"
            print(f"[{i}/{len(scenarios)}] {status}  {scenario['category']:<12} {scenario['command']!r}")

    save_results(results, results_dir, provider=provider, dry_run=dry_run)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full VoiceQA pipeline.")
    parser.add_argument("--scenarios", default=None, help="Path to a scenarios JSON file.")
    parser.add_argument("--device", default=None, help="ADB device serial. Implies real-device injection.")
    parser.add_argument("--mock", action="store_true", help="Use the mock intent-handling app instead of a real device.")
    parser.add_argument(
        "--speaker",
        action="store_true",
        help="No ADB: play each command on this PC's speaker and confirm the result manually.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use ADB if a device is connected, otherwise fall back to speaker + manual confirmation.",
    )
    parser.add_argument("--provider", choices=["anthropic", "openai", "mock"], default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the rule-based fallback evaluator instead of calling a real LLM judge.",
    )
    parser.add_argument("--voice", default=None, help="TTS voice: name substring or index (see --list-voices).")
    parser.add_argument("--list-voices", action="store_true", help="Show installed voices and exit.")
    parser.add_argument("--output-dir", default=None, help="Override the default output directory.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if args.list_voices:
        voices = list_voices()
        if not voices:
            print("No system voices found (is pyttsx3 installed?).")
        for i, name in enumerate(voices):
            print(f"{i}: {name}")
        return 0
    if not args.scenarios:
        parser.error("--scenarios is required")

    if sum([bool(args.mock), bool(args.speaker), bool(args.auto)]) > 1 or (args.device and (args.mock or args.speaker)):
        parser.error("--mock, --speaker, --auto and --device (with --mock/--speaker) are mutually exclusive")
    if args.auto:
        mode = "auto"
    elif args.speaker:
        mode = "speaker"
    else:
        mode = "adb" if args.device else "mock"

    scenarios = load_scenarios(args.scenarios)
    print(f"Loaded {len(scenarios)} scenarios from {args.scenarios}")
    print(f"Mode: {mode}" + (f" (device={args.device})" if args.device else " (no device required)"))
    if args.dry_run:
        print("Dry run: using rule-based fallback evaluator (no LLM calls).")

    start = time.time()
    results = run_pipeline(
        scenarios,
        mode=mode,
        device_serial=args.device,
        provider=args.provider,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        voice=args.voice,
    )
    elapsed = time.time() - start

    passed = sum(1 for r in results if r["pass"])
    if results:
        print(f"\nDone in {elapsed:.1f}s - {passed}/{len(results)} passed ({passed / len(results):.0%})")
    else:
        print("\nNo results.")
    output_dir_p = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR
    print(f"Results written to {output_dir_p / 'results'}")
    print("View the dashboard with: streamlit run voiceqa/reporting/dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
