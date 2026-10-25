"""A mock intent-handling app standing in for a real AAOS voice assistant.

Per the README: "Injects that audio into a target voice assistant (Android
Automotive OS emulator or a mock intent-handling app)". This module is that
mock app — a small rule-based NLU + state machine for an in-car climate
control feature, the same domain used in the README's examples. It lets the
whole pipeline (synthesis -> injection -> capture -> evaluation -> report)
run end to end without an emulator or a physical device.

It operates on already-transcribed text (the injector is responsible for
getting from audio to text, e.g. via Whisper), simulating the device's own
on-board STT + NLU + UI update in one step.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

MIN_TEMP_C = 16
MAX_TEMP_C = 30
MIN_FAN = 0
MAX_FAN = 5


@dataclass
class MockClimateAssistant:
    """Simulates an in-car climate-control voice assistant."""

    temperature_c: int = 21
    fan_speed: int = 2
    ac_on: bool = False
    history: list[dict] = field(default_factory=list)

    def _clamp_temp(self, value: int) -> int:
        return max(MIN_TEMP_C, min(MAX_TEMP_C, value))

    def _clamp_fan(self, value: int) -> int:
        return max(MIN_FAN, min(MAX_FAN, value))

    def _ui_state(self) -> dict:
        return {
            "temperature_c": self.temperature_c,
            "fan_speed": self.fan_speed,
            "ac_on": self.ac_on,
        }

    def _simulated_latency_ms(self, transcript: str) -> int:
        # Deterministic per-transcript "latency" so repeated runs/tests are stable,
        # while still varying enough to make the dashboard's latency chart meaningful.
        seed = int(hashlib.sha256(transcript.encode("utf-8")).hexdigest()[:6], 16)
        return 180 + (seed % 650)  # ~180ms - 830ms

    def handle(self, transcript: str) -> dict:
        """Process a transcribed voice command and return the simulated outcome.

        Returns: {"intent": str, "ui_state": dict, "response_text": str, "latency_ms": int}
        """
        text = transcript.strip().lower()
        latency_ms = self._simulated_latency_ms(transcript)

        intent, response_text = self._classify_and_apply(text)

        result = {
            "intent": intent,
            "ui_state": self._ui_state(),
            "response_text": response_text,
            "latency_ms": latency_ms,
            "transcript_in": transcript,
        }
        self.history.append(result)
        return result

    def _classify_and_apply(self, text: str) -> tuple[str, str]:
        # Adversarial / malformed: mismatched units, out-of-range values.
        if "volume" in text and "degree" in text:
            return "clarification_needed", "Did you mean set the temperature, or the volume?"

        temp_match = re.search(r"(-?\d+)\s*(?:degrees?)?", text)

        if "temperature" in text or "climate" in text:
            if temp_match:
                requested = int(temp_match.group(1))
                if requested < MIN_TEMP_C or requested > MAX_TEMP_C:
                    return (
                        "clarification_needed",
                        f"I can only set the temperature between {MIN_TEMP_C} and {MAX_TEMP_C} degrees.",
                    )
                self.temperature_c = self._clamp_temp(requested)
                return "set_temperature", f"Setting temperature to {self.temperature_c} degrees."

        if any(kw in text for kw in ("warmer", "wärmer", "warm it up", "heat it up")):
            self.temperature_c = self._clamp_temp(self.temperature_c + 2)
            return "increase_temperature", f"Making it warmer, now {self.temperature_c} degrees."

        if any(kw in text for kw in ("cooler", "colder", "kälter", "kühler", "cool it down")):
            self.temperature_c = self._clamp_temp(self.temperature_c - 2)
            return "decrease_temperature", f"Making it cooler, now {self.temperature_c} degrees."

        if temp_match and any(kw in text for kw in ("please", "set")):
            requested = int(temp_match.group(1))
            if requested < MIN_TEMP_C or requested > MAX_TEMP_C:
                return (
                    "clarification_needed",
                    f"I can only set the temperature between {MIN_TEMP_C} and {MAX_TEMP_C} degrees.",
                )
            self.temperature_c = self._clamp_temp(requested)
            return "set_temperature", f"Setting temperature to {self.temperature_c} degrees."

        if "fan" in text and any(kw in text for kw in ("up", "faster", "increase", "more")):
            self.fan_speed = self._clamp_fan(self.fan_speed + 1)
            return "increase_fan_speed", f"Fan speed now {self.fan_speed}."

        if "fan" in text and any(kw in text for kw in ("down", "slower", "decrease", "less")):
            self.fan_speed = self._clamp_fan(self.fan_speed - 1)
            return "decrease_fan_speed", f"Fan speed now {self.fan_speed}."

        if "air condition" in text or re.search(r"\bac\b", text):
            if "off" in text:
                self.ac_on = False
                return "ac_off", "Turning air conditioning off."
            self.ac_on = True
            return "ac_on", "Turning air conditioning on."

        return "clarification_needed", "Sorry, I didn't understand that. Could you rephrase?"

    def reset(self) -> None:
        self.temperature_c = 21
        self.fan_speed = 2
        self.ac_on = False
        self.history.clear()
