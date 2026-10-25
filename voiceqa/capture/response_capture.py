"""Captures the system's response after each injected command: UI state
(screenshot + hierarchy dump) and the spoken/text reply, transcribed to text
for the evaluator.

Branches on the injector's mode:

- ``mock``: the mock assistant already produced the full outcome
  synchronously (ui_state + response_text). A lightweight placeholder
  "screenshot" (a JSON snapshot rendered as text) stands in for a real
  screen capture so downstream code always has a ``screenshot`` path to
  point at.
- ``adb``: pulls a real screenshot + UIAutomator dump from the device. Since
  reliably capturing the *device's own audio output* over adb is
  emulator/build-dependent (see injection/audio_injector.py), the transcript
  is recovered heuristically from on-screen text in the UI dump, which is
  how most AAOS assistant UIs surface their spoken reply anyway.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from voiceqa.injection.adb_client import AdbClient
from voiceqa.injection.audio_injector import InjectionResult


def _extract_text_from_ui_dump(ui_xml: str) -> str:
    """Heuristic: concatenate all non-empty text="..." attributes from a
    uiautomator dump, in document order, as a stand-in transcript of
    whatever the assistant rendered on screen."""
    texts = re.findall(r'text="([^"]*)"', ui_xml)
    return " ".join(t for t in texts if t.strip())


def _write_mock_screenshot(ui_state: dict, output_path: Path) -> str:
    """Write a small JSON 'screenshot' placeholder for the mock target.

    A real screenshot doesn't exist for a rule-based mock app; this keeps
    the ``actual`` dict shape identical between mock and adb modes so the
    evaluator agent doesn't need to know which mode produced it.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"ui_state": ui_state}, indent=2), encoding="utf-8")
    return str(output_path)


def capture_response(
    injection_result: InjectionResult,
    output_dir: str,
    adb_client: AdbClient | None = None,
) -> dict:
    """Standardize the outcome of one injected command into:
    {"screenshot": path, "ui_hierarchy": str, "transcript": str, "latency_ms": int}
    """
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)
    scn_id = injection_result.scenario_id

    if injection_result.mode == "mock":
        response = injection_result.mock_response or {}
        ui_state = response.get("ui_state", {})
        screenshot_path = _write_mock_screenshot(ui_state, output_dir_p / f"{scn_id}_ui_state.json")
        return {
            "screenshot": screenshot_path,
            "ui_hierarchy": json.dumps(ui_state),
            "transcript": response.get("response_text", ""),
            "latency_ms": response.get("latency_ms", injection_result.latency_ms),
            "recognized_intent": response.get("intent"),
            "perceived_command_text": injection_result.perceived_text,
        }

    if injection_result.mode == "speaker":
        return {
            "screenshot": "",
            "ui_hierarchy": "",
            "transcript": injection_result.extra.get("manual_note", ""),
            "latency_ms": injection_result.latency_ms,
            "recognized_intent": None,
            "perceived_command_text": None,
            "manual_pass": injection_result.extra.get("manual_pass"),
        }

    if adb_client is None:
        raise ValueError("adb_client is required to capture a response in 'adb' mode")

    screenshot_path = adb_client.screenshot(str(output_dir_p / f"{scn_id}_ui_state.png"))
    ui_hierarchy = adb_client.dump_ui()
    transcript = _extract_text_from_ui_dump(ui_hierarchy)

    return {
        "screenshot": screenshot_path,
        "ui_hierarchy": ui_hierarchy,
        "transcript": transcript,
        "latency_ms": injection_result.latency_ms,
        "recognized_intent": None,
        "perceived_command_text": None,
    }
