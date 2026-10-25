"""Text-to-speech synthesis.

Primary backend is Coqui TTS (free, local, fully self-contained), per the
project README. Coqui has heavy native dependencies and is not always
installable on every platform/Python version, so this module degrades
gracefully:

1. Coqui TTS (``TTS`` package), if installed.
2. ``pyttsx3`` (lightweight, OS-native SAPI/NSSpeech/espeak), if installed.
3. A synthetic placeholder tone whose duration is estimated from word count —
   keeps the rest of the pipeline (variation, injection, capture, evaluation)
   fully runnable in environments where no real TTS engine is available.
"""

from __future__ import annotations

import json
from array import array
import logging
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

from voiceqa import config

logger = logging.getLogger(__name__)

_coqui_tts_instance = None


def _get_coqui_tts():
    global _coqui_tts_instance
    if _coqui_tts_instance is not None:
        return _coqui_tts_instance
    from TTS.api import TTS  # type: ignore

    _coqui_tts_instance = TTS(model_name=config.TTS_MODEL_NAME)
    return _coqui_tts_instance


_PYTTSX3_SCRIPT = """
import sys, pyttsx3
e = pyttsx3.init()
query = sys.argv[4].lower()
if query:
    voices = e.getProperty('voices')
    if query.isdigit() and int(query) < len(voices):
        e.setProperty('voice', voices[int(query)].id)
    else:
        for v in voices:
            if query in v.name.lower() or query in v.id.lower():
                e.setProperty('voice', v.id)
                break
        else:
            sys.exit('voice not found: ' + query)
e.setProperty('rate', int(e.getProperty('rate') * float(sys.argv[3])))
e.save_to_file(sys.argv[1], sys.argv[2])
e.runAndWait()
"""


def _pyttsx3_to_file(text: str, output_path: Path, speed: float, voice: str = "") -> None:
    """Run pyttsx3 in a fresh subprocess per utterance.

    pyttsx3's engine is a per-process singleton whose second ``runAndWait()``
    hangs on some platforms (seen on Windows/SAPI), so a single process can't
    synthesize more than one file reliably.
    """
    import pyttsx3  # type: ignore  # noqa: F401  (raises ImportError early if absent)

    result = subprocess.run(
        [sys.executable, "-c", _PYTTSX3_SCRIPT, text, str(output_path), str(speed), voice or ""],
        capture_output=True,
        timeout=60,
    )
    if b"voice not found" in result.stderr:
        raise ValueError(f"Unknown voice {voice!r}. Run `python run_pipeline.py --list-voices` to see the options.")
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(result.stderr.decode(errors="replace")[-300:] or "pyttsx3 produced no audio")


_LIST_VOICES_SCRIPT = """
import json, pyttsx3
print(json.dumps([v.name for v in pyttsx3.init().getProperty('voices')]))
"""


def list_voices() -> list[str]:
    """Names of the installed system voices (index in this list works as a voice id)."""
    try:
        result = subprocess.run([sys.executable, "-c", _LIST_VOICES_SCRIPT], capture_output=True, timeout=60)
        return json.loads(result.stdout.decode())
    except (ValueError, subprocess.SubprocessError, OSError):
        return []


def _estimate_duration_seconds(text: str, speed: float) -> float:
    # ~2.5 spoken words/second at speed=1.0, clamped to a sane floor.
    words = max(len(text.split()), 1)
    return max(words / 2.5 / speed, 0.4)


def _write_placeholder_tone(text: str, output_path: Path, speed: float) -> None:
    """Write a short sine-wave WAV standing in for real speech audio.

    Frequency is derived from the text so different commands are at least
    distinguishable by ear/waveform during manual debugging, without adding
    a numpy dependency for what is only a fallback path.
    """
    duration = _estimate_duration_seconds(text, speed)
    sample_rate = 22050
    freq = 180 + (sum(ord(c) for c in text) % 220)  # 180-400 Hz range
    n_samples = int(duration * sample_rate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        amplitude = 8000
        frames = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            # Slight amplitude envelope so it isn't a harsh buzz.
            envelope = min(1.0, t * 8) * min(1.0, (duration - t) * 8 + 0.05)
            value = int(amplitude * envelope * math.sin(2 * math.pi * freq * t))
            frames += struct.pack("<h", value)
        wav_file.writeframes(bytes(frames))

    logger.warning(
        "No real TTS engine available; wrote placeholder tone for %r -> %s. "
        "Install 'TTS' (Coqui) or 'pyttsx3' for real speech synthesis.",
        text,
        output_path,
    )


WAKE_WORD_PAUSE_SECONDS = 1.5


def _read_mono16(path: Path, target_rate: int) -> array:
    """Read a WAV as mono 16-bit samples, resampled (linear) to ``target_rate``."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        channels, rate = w.getnchannels(), w.getframerate()
        samples = array("h")
        samples.frombytes(w.readframes(w.getnframes()))
    if sys.byteorder == "big":
        samples.byteswap()
    if channels > 1:
        samples = array("h", (sum(samples[i : i + channels]) // channels for i in range(0, len(samples), channels)))
    if rate != target_rate and len(samples) > 1:
        n_out = int(len(samples) * target_rate / rate)
        ratio = (len(samples) - 1) / max(n_out - 1, 1)
        out = array("h")
        for i in range(n_out):
            pos = i * ratio
            lo = int(pos)
            hi = min(lo + 1, len(samples) - 1)
            out.append(int(samples[lo] + (samples[hi] - samples[lo]) * (pos - lo)))
        samples = out
    return samples


def _join_with_silence(first: Path, second: Path, output_path: Path, pause_seconds: float) -> None:
    """Concatenate two WAVs with a silent gap (stdlib only). ``first`` is
    converted to ``second``'s sample rate / mono 16-bit format if it differs."""
    with wave.open(str(second), "rb") as b:
        rate = b.getframerate()
        frames_b = b.readframes(b.getnframes())
        if b.getnchannels() != 1 or b.getsampwidth() != 2:
            raise ValueError("command audio must be mono 16-bit")
    head = _read_mono16(first, rate)
    if sys.byteorder == "big":
        head.byteswap()
    silence = b"\x00\x00" * int(rate * pause_seconds)
    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(head.tobytes() + silence + frames_b)


def synthesize(command: dict, output_path: str, speed: float = 1.0, voice: str | None = None) -> str:
    """Synthesize ``command['command']`` text to a WAV file at ``output_path``.

    If a recorded wake clip exists (``wake_audio`` on the scenario, or
    ``voiceqa/wake/hey_google.wav`` / ``VOICEQA_WAKE_AUDIO``), it is used in
    place of a synthesized wake word.

    If the scenario has a ``wake_word`` (e.g. "Hey Google"), it is spoken
    first, followed by a silent pause (``wake_pause`` seconds, default 1.5)
    so the assistant has time to wake up before the command starts.

    ``voice`` (a name substring or index from ``list_voices()``) selects the
    system voice; a scenario's own ``voice`` field wins, then this argument,
    then the ``VOICEQA_VOICE`` environment setting. Only pyttsx3 honours it.

    Mirrors the README's ``synthesize(command, output_path, speed)`` signature.
    Returns the output path as a string.
    """
    voice = command.get("voice") or voice or config.TTS_VOICE
    wake_word = command.get("wake_word")
    # A recording of the tester's own voice beats TTS for the wake word:
    # Voice Match on the phone is trained on the owner and rejects synthetic voices.
    wake_audio = command.get("wake_audio") or config.WAKE_AUDIO
    if wake_audio and not Path(wake_audio).is_file():
        wake_audio = ""
    if not wake_word and not wake_audio:
        return _synthesize_text(command["command"], output_path, speed, voice)

    out = Path(output_path)
    wake_path = out.with_name(out.stem + "_wake.wav")
    cmd_path = out.with_name(out.stem + "_cmd.wav")
    if wake_audio:
        wake_path = Path(wake_audio)
    else:
        _synthesize_text(wake_word, str(wake_path), speed, voice)
    _synthesize_text(command["command"], str(cmd_path), speed, voice)
    _join_with_silence(wake_path, cmd_path, out, command.get("wake_pause", WAKE_WORD_PAUSE_SECONDS))
    return str(out)


def _synthesize_text(text: str, output_path: str, speed: float = 1.0, voice: str = "") -> str:
    output_path_p = Path(output_path)
    output_path_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        tts = _get_coqui_tts()
        tts.tts_to_file(text=text, file_path=str(output_path_p), speed=speed)
        return str(output_path_p)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("Coqui TTS failed (%s); falling back to pyttsx3/placeholder.", exc)

    try:
        _pyttsx3_to_file(text, output_path_p, speed, voice)
        return str(output_path_p)
    except ImportError:
        pass
    except ValueError:
        raise  # bad --voice: tell the user instead of silently playing a beep
    except Exception as exc:  # noqa: BLE001
        logger.warning("pyttsx3 failed (%s); falling back to placeholder tone.", exc)

    _write_placeholder_tone(text, output_path_p, speed)
    return str(output_path_p)
