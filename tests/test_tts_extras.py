"""Tests for TTS engine extras: list_voices, list_capabilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


def test_base_tts_has_list_voices():
    from revospeech.tts.base import BaseTTS

    assert hasattr(BaseTTS, "list_voices")
    assert hasattr(BaseTTS, "list_capabilities")


def test_base_list_voices_default_returns_empty():
    from revospeech.tts.base import BaseTTS
    from revospeech.tts.result import Audio

    class FakeTTS(BaseTTS):
        def synthesize(self, text, output_path=None, **kwargs):
            return Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=22050)

    engine = FakeTTS("test")
    assert engine.list_voices() == []


def test_list_capabilities_uses_manifest():
    from revospeech.tts.base import BaseTTS
    from revospeech.tts.result import Audio

    class FakeTTS(BaseTTS):
        def synthesize(self, text, output_path=None, **kwargs):
            return Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=22050)

    engine = FakeTTS("test-model")

    fake_manifest = MagicMock()
    fake_manifest.capabilities = ["voice-cloning", "streaming", "batch"]

    with patch("revospeech.registry.get", return_value=fake_manifest):
        result = engine.list_capabilities()

    assert result == ["voice-cloning", "streaming", "batch"]


def test_list_capabilities_missing_manifest():
    from revospeech.tts.base import BaseTTS
    from revospeech.tts.result import Audio

    class FakeTTS(BaseTTS):
        def synthesize(self, text, output_path=None, **kwargs):
            return Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=22050)

    engine = FakeTTS("test-model")

    with patch("revospeech.registry.get", side_effect=KeyError("not found")):
        result = engine.list_capabilities()

    assert result == []


def test_vits_list_voices_returns_speakers():
    """VitsTTS.list_voices should return PRODUCTION_SPEAKERS."""
    from revospeech.tts.vits_engine import PRODUCTION_SPEAKERS, VitsTTS

    # Sanity check the underlying constant.
    assert isinstance(PRODUCTION_SPEAKERS, list)
    assert len(PRODUCTION_SPEAKERS) > 0

    # Bind the real method to an instance built via __new__ (avoids model load).
    engine = VitsTTS.__new__(VitsTTS)
    assert engine.list_voices() == list(PRODUCTION_SPEAKERS)


def test_revovoice_list_voices_returns_empty():
    """RevoVoiceTTS.list_voices should return [] (zero-shot cloning)."""
    from revospeech.tts.revovoice_engine import RevoVoiceTTS

    engine = RevoVoiceTTS.__new__(RevoVoiceTTS)
    engine.model_name = "test"
    engine.device = "cpu"

    assert engine.list_voices() == []


def test_speed_parameter_is_accepted():
    """synthesize() should accept speed parameter without error."""
    from revospeech.tts.base import BaseTTS
    from revospeech.tts.result import Audio

    captured = {}

    class FakeTTS(BaseTTS):
        def synthesize(self, text, output_path=None, *, speed=1.0, **kwargs):
            captured["speed"] = speed
            return Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=22050)

    engine = FakeTTS("test")
    engine.synthesize("hello", speed=1.5)
    assert captured["speed"] == 1.5


def test_vits_speed_threaded_to_synthesize_ids():
    """VitsTTS.synthesize threads speed into _synthesize_ids via length_scale/speed."""
    from revospeech.tts.vits_engine import VitsTTS

    captured = {}

    def fake_load_speaker(self, speaker):
        captured["speaker"] = speaker
        return (MagicMock(), {"^": [1], "_": [0], "$": [2]}, {"inference": {}})

    import revospeech.tts.vits_engine as vits_mod

    def fake_phonemize(text, language="ms"):
        return [["a"]]

    def fake_synthesize_ids(self, sess, phoneme_ids, config, speed=1.0):
        captured["speed"] = speed
        return np.zeros(100, dtype=np.float32)

    engine = VitsTTS.__new__(VitsTTS)
    engine.manifest = MagicMock()
    engine.manifest.sample_rate = 22050

    with (
        patch.object(VitsTTS, "_load_speaker", fake_load_speaker),
        patch.object(vits_mod, "_phonemize_espeak", fake_phonemize),
        patch.object(VitsTTS, "_synthesize_ids", fake_synthesize_ids),
        patch.object(vits_mod, "_normalize_text_simple", side_effect=lambda x: x),
    ):
        engine.synthesize("hello", speed=1.5)

    assert captured.get("speed") == 1.5
