import wave

from voiceqa.synthesis.tts import synthesize
from voiceqa.synthesis import variation


def _write_tone(path, seconds=1.0, sample_rate=22050):
    import math
    import struct

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n = int(seconds * sample_rate)
        frames = bytearray()
        for i in range(n):
            t = i / sample_rate
            value = int(5000 * math.sin(2 * math.pi * 220 * t))
            frames += struct.pack("<h", value)
        wf.writeframes(bytes(frames))


def test_synthesize_produces_a_readable_wav(tmp_path):
    out = tmp_path / "out.wav"
    synthesize({"command": "set temperature to 22 degrees"}, str(out))
    assert out.exists()
    with wave.open(str(out), "rb") as wf:
        assert wf.getnframes() > 0


def test_generate_noise_sample_produces_valid_wav(tmp_path):
    path = variation.generate_noise_sample(str(tmp_path / "noise.wav"), duration_seconds=0.5)
    with wave.open(path, "rb") as wf:
        assert wf.getnframes() > 0


def test_add_background_noise_keeps_speech_length(tmp_path):
    speech_path = tmp_path / "speech.wav"
    noise_path = tmp_path / "noise.wav"
    _write_tone(speech_path, seconds=1.0)
    variation.generate_noise_sample(str(noise_path), duration_seconds=2.0)

    out_path = tmp_path / "mixed.wav"
    variation.add_background_noise(str(speech_path), str(noise_path), str(out_path), noise_db=-20)

    with wave.open(str(speech_path)) as sw, wave.open(str(out_path)) as ow:
        assert ow.getnframes() == sw.getnframes()


def test_change_speed_faster_produces_shorter_audio(tmp_path):
    speech_path = tmp_path / "speech.wav"
    _write_tone(speech_path, seconds=1.0)

    out_path = tmp_path / "fast.wav"
    variation.change_speed(str(speech_path), str(out_path), speed=1.25)

    with wave.open(str(speech_path)) as sw, wave.open(str(out_path)) as ow:
        assert ow.getnframes() < sw.getnframes()


def test_apply_variation_happy_path_is_passthrough_length(tmp_path):
    speech_path = tmp_path / "speech.wav"
    _write_tone(speech_path, seconds=1.0)

    scenario = {"id": "scn_000", "category": "happy_path"}
    out_path = variation.apply_variation(scenario, str(speech_path), str(tmp_path))

    with wave.open(str(speech_path)) as sw, wave.open(out_path) as ow:
        assert ow.getnframes() == sw.getnframes()
