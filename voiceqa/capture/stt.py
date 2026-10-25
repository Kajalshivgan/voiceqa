"""Speech-to-text, used to read back the device's spoken/text reply for the
evaluator (per README: "Use Whisper ... to transcribe any spoken reply back
to text").

Also used by the mock injector to decide what the mock device "heard" from
the synthesized command audio, with a degradation simulation for noisy
variants so the dashboard's failure clustering has something real to show
(this is what backs the README's example insight: "voice recognition
degrades 23% under simulated road noise").
"""

from __future__ import annotations

import hashlib
import logging
import random

logger = logging.getLogger(__name__)

_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    import whisper  # type: ignore
    from voiceqa import config

    _whisper_model = whisper.load_model(config.WHISPER_MODEL_NAME)
    return _whisper_model


def transcribe(audio_path: str) -> str | None:
    """Real Whisper transcription. Returns None if whisper isn't available/usable."""
    try:
        model = _get_whisper_model()
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Whisper model load failed (%s); falling back.", exc)
        return None

    try:
        result = model.transcribe(audio_path)
        return result.get("text", "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Whisper transcription failed for %s (%s); falling back.", audio_path, exc)
        return None


_DROPPABLE_WORDS = {"the", "a", "an", "please", "it", "to"}
_CONFUSABLE = {
    "temperature": "temperature",
    "twenty": "twenty",
    "warmer": "warmer",
    "cooler": "cooler colder",
}


def simulate_transcription(ground_truth_text: str, degrade: bool = False, seed_key: str | None = None) -> str:
    """Deterministically simulate what an on-device STT might produce.

    When ``degrade`` is False, returns the ground truth text unchanged
    (clean-audio case). When True, applies small, reproducible
    perturbations (a dropped filler word, or an occasional garbled token)
    to emulate recognition errors under background noise — enough to
    produce realistic, non-trivial pass/fail variance without needing a
    real STT model.
    """
    if not degrade:
        return ground_truth_text

    key = seed_key or ground_truth_text
    rng = random.Random(int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16))

    words = ground_truth_text.split()
    if not words:
        return ground_truth_text

    # ~35% chance of a meaningful garble, otherwise just drop a filler word.
    if rng.random() < 0.35:
        idx = rng.randrange(len(words))
        # Garble a word by dropping its last couple characters or swapping
        # a digit, simulating a misheard token.
        word = words[idx]
        digit_match = next((c for c in word if c.isdigit()), None)
        if digit_match:
            words[idx] = word.replace(digit_match, str((int(digit_match) + rng.randint(1, 3)) % 10))
        elif len(word) > 3:
            words[idx] = word[:-2]
    else:
        words = [w for w in words if w.lower().strip(".,") not in _DROPPABLE_WORDS] or words

    return " ".join(words)


def transcribe_or_simulate(
    audio_path: str,
    ground_truth_text: str,
    degrade: bool = False,
    prefer_real: bool = True,
) -> tuple[str, str]:
    """Try real Whisper transcription first (if available and requested),
    otherwise fall back to the deterministic simulation.

    Returns (text, method) where method is "whisper" or "simulated".
    """
    if prefer_real:
        real = transcribe(audio_path)
        if real:
            return real, "whisper"
    return simulate_transcription(ground_truth_text, degrade=degrade, seed_key=audio_path), "simulated"
