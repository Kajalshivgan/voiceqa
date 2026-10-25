"""Speaker + manual-confirmation helpers for when no ADB connection exists.

The synthesized command is played out loud on this computer's speaker, the
phone/head-unit under test picks it up through its own microphone, and the
tester confirms by hand whether the expected change actually happened.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


class PlaybackError(RuntimeError):
    pass


def play_audio(path: str) -> None:
    """Play a WAV file through the default output device and block until done."""
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(path, winsound.SND_FILENAME)
        return

    players = (["afplay"], ["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"])
    for player in players:
        if shutil.which(player[0]):
            subprocess.run(player + [path], check=False)
            return
    raise PlaybackError(f"No audio player found. Play this file manually: {path}")


def ask_manual_verdict(scenario: dict) -> tuple[bool, str]:
    """Ask the tester whether the phone did what the scenario expected.

    Returns ``(passed, note)``. Answering ``r`` is handled by the caller via
    the returned note sentinel ``"__replay__"``.
    """
    print(f"\n  Command spoken : {scenario['command']!r}")
    print(f"  Expected       : {scenario['expected_intent']} -> {scenario['expected_ui_state']}")
    while True:
        answer = input("  Did the change happen on the device? [y]es / [n]o / [r]eplay: ").strip().lower()
        if answer in ("y", "yes"):
            note = input("  Optional note (Enter to skip): ").strip()
            return True, note
        if answer in ("n", "no"):
            note = input("  What happened instead? (Enter to skip): ").strip()
            return False, note
        if answer in ("r", "replay"):
            return False, "__replay__"
        print("  Please answer y, n or r.")
