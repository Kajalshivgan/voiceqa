"""Accent/speed/background-noise variation layer.

This is what makes the generated test suite meaningfully harder than a
single clean recording per command (per the project README).

Primary backend is ``pydub``. ``pydub`` itself depends on the stdlib
``audioop`` module, which was removed in Python 3.13+, so on newer
interpreters (or wherever ``pydub``/``ffmpeg`` isn't available) this module
falls back to a small pure-stdlib WAV mixer/resampler built on ``wave`` so
the variation stage still works end to end.
"""

from __future__ import annotations

import logging
import random
import wave
from pathlib import Path

logger = logging.getLogger(__name__)


def _pydub_available() -> bool:
    try:
        import pydub  # noqa: F401
        import audioop  # noqa: F401  -- pydub needs this; absent on Python 3.13+
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------
# Noise sample generation (no external asset required)
# --------------------------------------------------------------------------


def generate_noise_sample(
    output_path: str,
    duration_seconds: float = 5.0,
    sample_rate: int = 22050,
    kind: str = "white",
) -> str:
    """Generate a synthetic noise WAV (white noise or a low-frequency 'road rumble')."""
    output_path_p = Path(output_path)
    output_path_p.parent.mkdir(parents=True, exist_ok=True)

    n_samples = int(duration_seconds * sample_rate)
    rng = random.Random(42)

    with wave.open(str(output_path_p), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        state = 0.0
        for _ in range(n_samples):
            white = rng.uniform(-1, 1)
            if kind == "road":
                # Cheap low-pass filter for a rumble-ish texture.
                state = 0.95 * state + 0.05 * white
                sample = state
            else:
                sample = white
            value = int(max(-1.0, min(1.0, sample)) * 6000)
            frames += value.to_bytes(2, "little", signed=True)
        wav_file.writeframes(bytes(frames))

    return str(output_path_p)


# --------------------------------------------------------------------------
# Pure-stdlib WAV helpers (fallback path)
# --------------------------------------------------------------------------


def _read_wav_samples(path: str) -> tuple[list[int], int, int]:
    with wave.open(path, "rb") as wf:
        n_frames = wf.getnframes()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        raw = wf.readframes(n_frames)
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV supported by the fallback mixer, got width={sample_width}")
    samples = [
        int.from_bytes(raw[i : i + 2], "little", signed=True) for i in range(0, len(raw), 2)
    ]
    return samples, frame_rate, sample_width


def _write_wav_samples(path: str, samples: list[int], frame_rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(frame_rate)
        frames = bytearray()
        for s in samples:
            clamped = max(-32768, min(32767, s))
            frames += clamped.to_bytes(2, "little", signed=True)
        wf.writeframes(bytes(frames))


def _db_to_amplitude_ratio(db: float) -> float:
    return 10 ** (db / 20)


def _fallback_add_background_noise(
    speech_path: str, noise_path: str, output_path: str, noise_db: float
) -> str:
    speech, speech_rate, _ = _read_wav_samples(speech_path)
    noise, noise_rate, _ = _read_wav_samples(noise_path)

    if noise_rate != speech_rate and noise:
        # Crude resample by nearest-neighbour index mapping.
        ratio = noise_rate / speech_rate
        noise = [noise[int(i * ratio) % len(noise)] for i in range(len(speech))]

    if not noise:
        noise = [0] * len(speech)
    if len(noise) < len(speech):
        noise = (noise * (len(speech) // len(noise) + 1))[: len(speech)]
    else:
        noise = noise[: len(speech)]

    gain = _db_to_amplitude_ratio(noise_db)
    mixed = [int(s + n * gain) for s, n in zip(speech, noise)]
    _write_wav_samples(output_path, mixed, speech_rate)
    return output_path


def _fallback_change_speed(input_path: str, output_path: str, speed: float) -> str:
    """Naive tape-speed style change: reinterpret the frame rate, then resample
    back to the original rate so the output plays at normal pitch reference
    but with altered duration/tempo characteristics baked into the samples,
    matching pydub's common `speedup` shortcut of reindexing frames."""
    samples, frame_rate, _ = _read_wav_samples(input_path)
    if speed <= 0:
        raise ValueError("speed must be > 0")

    new_length = max(1, int(len(samples) / speed))
    resampled = [samples[min(int(i * speed), len(samples) - 1)] for i in range(new_length)]
    _write_wav_samples(output_path, resampled, frame_rate)
    return output_path


# --------------------------------------------------------------------------
# Public API (mirrors README signatures)
# --------------------------------------------------------------------------


def add_background_noise(
    speech_path: str, noise_path: str, output_path: str, noise_db: float = -20
) -> str:
    if _pydub_available():
        from pydub import AudioSegment

        speech = AudioSegment.from_wav(speech_path)
        noise = AudioSegment.from_wav(noise_path) + noise_db
        noise = noise[: len(speech)]
        speech.overlay(noise).export(output_path, format="wav")
        return output_path

    logger.info("pydub/audioop unavailable; using stdlib fallback noise mixer.")
    return _fallback_add_background_noise(speech_path, noise_path, output_path, noise_db)


def change_speed(input_path: str, output_path: str, speed: float = 1.0) -> str:
    if speed == 1.0:
        import shutil

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, output_path)
        return output_path

    if _pydub_available():
        from pydub import AudioSegment

        audio = AudioSegment.from_wav(input_path)
        altered = audio._spawn(
            audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed)}
        ).set_frame_rate(audio.frame_rate)
        altered.export(output_path, format="wav")
        return output_path

    logger.info("pydub/audioop unavailable; using stdlib fallback speed changer.")
    return _fallback_change_speed(input_path, output_path, speed)


# --------------------------------------------------------------------------
# Scenario-driven variation policy
# --------------------------------------------------------------------------

CATEGORY_VARIATION = {
    "happy_path": {"speed": 1.0, "noise_db": None},
    "paraphrased": {"speed": 1.0, "noise_db": None},
    "edge_case": {"speed": 0.9, "noise_db": -18},
    "adversarial": {"speed": 1.1, "noise_db": None},
}


def apply_variation(
    scenario: dict,
    speech_path: str,
    output_dir: str,
    noise_sample_path: str | None = None,
) -> str:
    """Apply the variation policy appropriate for a scenario's category.

    Returns the path to the final (possibly unmodified) audio file.
    """
    policy = CATEGORY_VARIATION.get(scenario["category"], {"speed": 1.0, "noise_db": None})
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    current_path = speech_path
    scn_id = scenario.get("id", "scn")

    if policy["speed"] != 1.0:
        sped_path = str(output_dir_p / f"{scn_id}_speed.wav")
        current_path = change_speed(current_path, sped_path, speed=policy["speed"])

    if policy["noise_db"] is not None:
        if noise_sample_path is None:
            noise_sample_path = str(output_dir_p / "_noise_sample.wav")
            if not Path(noise_sample_path).exists():
                generate_noise_sample(noise_sample_path, duration_seconds=5.0, kind="road")
        noisy_path = str(output_dir_p / f"{scn_id}_noisy.wav")
        current_path = add_background_noise(
            current_path, noise_sample_path, noisy_path, noise_db=policy["noise_db"]
        )

    return current_path
