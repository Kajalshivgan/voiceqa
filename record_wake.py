"""Record your own "Hey Google" so the pipeline can use it instead of a computer voice.

Voice Match on the phone is trained on the phone owner's voice, so a
synthesized "Hey Google" is usually rejected. Run this once:

    pip install sounddevice
    python record_wake.py            # records 2 seconds from the default mic

The clip is saved to voiceqa/wake/hey_google.wav and is picked up automatically
(or point VOICEQA_WAKE_AUDIO / a scenario's "wake_audio" at another file).
"""

import argparse
import wave
from pathlib import Path

from voiceqa import config

SAMPLE_RATE = 22050


def main() -> int:
    parser = argparse.ArgumentParser(description="Record the wake-word clip.")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--output", default=config.WAKE_AUDIO)
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        print("Recording needs the 'sounddevice' package: pip install sounddevice")
        print("Or record a 16-bit WAV yourself and point VOICEQA_WAKE_AUDIO at it.")
        return 1

    input(f"Press Enter, then say 'Hey Google' clearly within {args.seconds:g}s... ")
    audio = sd.rec(int(args.seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())
    print(f"Saved {out}. Replay it to check it sounds right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
