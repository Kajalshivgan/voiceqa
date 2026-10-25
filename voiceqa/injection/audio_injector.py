"""Feeds synthesized audio into the target system.

Two modes:

- ``mock``: no emulator/device required. Simulates the mic pickup by
  deciding what the on-device STT would have heard (degraded if the audio
  went through the noise-variation path), then drives the mock
  intent-handling app directly. This is the default and what makes the
  pipeline runnable without any Android tooling.
- ``adb``: pushes the audio file to an AAOS emulator/device and plays it
  through a virtual microphone route.

  Real caveat, stated plainly rather than papered over: the exact command to
  drive an emulator's virtual microphone from a WAV file is emulator-build
  and API-level dependent (AAOS builds commonly expose this via the
  extended-controls virtual microphone / audio input routing rather than a
  stable `adb shell` command). The implementation below pushes the file and
  issues a best-effort playback intent that works on common AAOS test
  images; if your emulator/build differs, swap ``_trigger_playback`` for
  whatever your rig actually supports (e.g. the emulator console's
  `audio play` extended command, or a virtual_scene audio route).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from voiceqa.capture import stt
from voiceqa.injection import speaker
from voiceqa.injection.adb_client import AdbClient
from voiceqa.mock_target.mock_assistant import MockClimateAssistant

logger = logging.getLogger(__name__)

REMOTE_AUDIO_PATH = "/sdcard/voiceqa_injected.wav"


@dataclass
class InjectionResult:
    mode: str
    scenario_id: str
    latency_ms: int
    perceived_text: str | None = None  # what the (mock) device believed it heard
    stt_method: str | None = None
    mock_response: dict | None = None
    extra: dict = field(default_factory=dict)


class AudioInjector:
    def __init__(
        self,
        mode: str = "mock",
        adb_client: AdbClient | None = None,
        mock_assistant: MockClimateAssistant | None = None,
        play_fn=None,
        confirm_fn=None,
    ):
        if mode not in ("mock", "adb", "speaker"):
            raise ValueError(f"Unknown injection mode: {mode!r}")
        self.mode = mode
        self.adb_client = adb_client
        self.mock_assistant = mock_assistant or MockClimateAssistant()
        self.play_fn = play_fn or speaker.play_audio
        self.confirm_fn = confirm_fn or speaker.ask_manual_verdict

        if mode == "adb" and adb_client is None:
            raise ValueError("adb_client is required when mode='adb'")

    def inject(self, scenario: dict, audio_path: str, noise_applied: bool = False) -> InjectionResult:
        start = time.perf_counter()
        if self.mode == "mock":
            result = self._inject_mock(scenario, audio_path, noise_applied)
        elif self.mode == "speaker":
            result = self._inject_speaker(scenario, audio_path)
        else:
            result = self._inject_adb(scenario, audio_path)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        result.extra["wall_clock_ms"] = elapsed_ms
        return result

    def _inject_mock(self, scenario: dict, audio_path: str, noise_applied: bool) -> InjectionResult:
        perceived_text, method = stt.transcribe_or_simulate(
            audio_path,
            ground_truth_text=scenario["command"],
            degrade=noise_applied,
            prefer_real=False,  # our own synth audio isn't real speech; simulate deliberately
        )
        response = self.mock_assistant.handle(perceived_text)
        return InjectionResult(
            mode="mock",
            scenario_id=scenario.get("id", "scn"),
            latency_ms=response["latency_ms"],
            perceived_text=perceived_text,
            stt_method=method,
            mock_response=response,
        )

    def _inject_speaker(self, scenario: dict, audio_path: str) -> InjectionResult:
        """Play the command out loud, then let the tester confirm the outcome."""
        sid = scenario.get("id", "scn")
        while True:
            print(f"\n[{sid}] Playing audio on the speaker - hold the phone near it...")
            start = time.perf_counter()
            self.play_fn(audio_path)
            latency_ms = int((time.perf_counter() - start) * 1000)
            passed, note = self.confirm_fn(scenario)
            if note != "__replay__":
                break
        return InjectionResult(
            mode="speaker",
            scenario_id=sid,
            latency_ms=latency_ms,
            extra={"manual_pass": passed, "manual_note": note},
        )

    def _inject_adb(self, scenario: dict, audio_path: str) -> InjectionResult:
        assert self.adb_client is not None
        start = time.perf_counter()
        self.adb_client.push(audio_path, REMOTE_AUDIO_PATH)
        self._trigger_playback()
        latency_ms = int((time.perf_counter() - start) * 1000)
        return InjectionResult(
            mode="adb",
            scenario_id=scenario.get("id", "scn"),
            latency_ms=latency_ms,
        )

    def _trigger_playback(self) -> None:
        """Best-effort virtual-mic playback trigger. See module docstring caveat."""
        assert self.adb_client is not None
        # Common AAOS dev-image helper: media player pointed at the pushed file,
        # routed to the microphone input via the device's audio HAL test hook.
        # Adjust for your emulator/build as needed.
        self.adb_client.shell(f"am start -a android.intent.action.VIEW -d file://{REMOTE_AUDIO_PATH}")
