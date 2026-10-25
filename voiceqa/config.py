"""Central configuration: environment variables, defaults, and shared paths."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- LLM provider settings -------------------------------------------------
# "anthropic" | "openai" | "mock"
# Falls back to "mock" automatically at call time if no key is configured,
# so the pipeline is runnable out of the box without any credentials.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

# --- TTS / STT settings -----------------------------------------------------
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "tts_models/en/ljspeech/tacotron2-DDC")
# System voice for pyttsx3: a name substring (e.g. "zira") or an index from --list-voices.
TTS_VOICE = os.getenv("VOICEQA_VOICE", "")
# Recorded wake-word clip ("Hey Google" in the tester's own voice), used instead of TTS if present.
WAKE_AUDIO = os.getenv("VOICEQA_WAKE_AUDIO", str(PROJECT_ROOT / "voiceqa" / "wake" / "hey_google.wav"))
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base")

# --- ADB / target settings --------------------------------------------------
ADB_BINARY = os.getenv("ADB_BINARY", "adb")
DEFAULT_DEVICE_SERIAL = os.getenv("VOICEQA_DEVICE", "")

# --- Paths -------------------------------------------------------------------
SCENARIOS_DIR = PROJECT_ROOT / "voiceqa" / "scenarios"
OUTPUT_DIR = Path(os.getenv("VOICEQA_OUTPUT_DIR", str(PROJECT_ROOT / "voiceqa" / "output")))
AUDIO_DIR = OUTPUT_DIR / "audio"
CAPTURE_DIR = OUTPUT_DIR / "capture"
RESULTS_DIR = OUTPUT_DIR / "results"

for _dir in (OUTPUT_DIR, AUDIO_DIR, CAPTURE_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def has_real_llm_credentials() -> bool:
    """True if a real (non-mock) LLM provider is actually usable."""
    if LLM_PROVIDER == "anthropic":
        return bool(ANTHROPIC_API_KEY)
    if LLM_PROVIDER == "openai":
        return bool(OPENAI_API_KEY)
    return False
