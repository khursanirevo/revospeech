"""Tests for ASR streaming and language detection methods."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class _FakeASR:
    """Minimal concrete subclass for testing BaseASR behavior."""

    def __init__(self, model_name: str = "test-model"):
        self.model_name = model_name
        self.device = "auto"

    # Mirror BaseASR methods under test
    def detect_language(self, audio_path):
        from revospeech.registry import get

        try:
            manifest = get(self.model_name, "asr")
            return manifest.language or "en"
        except KeyError:
            return "en"

    def list_languages(self):
        from revospeech.registry import get

        try:
            manifest = get(self.model_name, "asr")
            if manifest.languages:
                return list(manifest.languages)
            return [manifest.language or "en"]
        except KeyError:
            return ["en"]


def test_base_asr_has_stream_transcribe_method():
    from revospeech.asr.base import BaseASR

    assert hasattr(BaseASR, "stream_transcribe")
    assert hasattr(BaseASR, "detect_language")
    assert hasattr(BaseASR, "list_languages")


def test_stream_transcribe_default_raises():
    from revospeech.asr.base import BaseASR
    from revospeech.asr.result import Transcript

    class FakeASR(BaseASR):
        def transcribe(self, audio_path):
            return Transcript(text="", segments=[], language="en")

    engine = FakeASR("test-model")
    with pytest.raises(NotImplementedError, match="streaming"):
        engine.stream_transcribe("/dev/null")


def test_detect_language_default_uses_manifest():
    from revospeech.asr.base import BaseASR
    from revospeech.asr.result import Transcript

    class FakeASR(BaseASR):
        def transcribe(self, audio_path):
            return Transcript(text="", segments=[], language="en")

    engine = FakeASR("test-model")

    fake_manifest = MagicMock()
    fake_manifest.language = "fr"

    with patch("revospeech.registry.get", return_value=fake_manifest):
        result = engine.detect_language("/dev/null")

    assert result == "fr"


def test_detect_language_fallback_on_missing_manifest():
    from revospeech.asr.base import BaseASR
    from revospeech.asr.result import Transcript

    class FakeASR(BaseASR):
        def transcribe(self, audio_path):
            return Transcript(text="", segments=[], language="en")

    engine = FakeASR("test-model")

    with patch("revospeech.registry.get", side_effect=KeyError("not found")):
        result = engine.detect_language("/dev/null")

    assert result == "en"


def test_detect_language_empty_manifest_language_falls_back_to_en():
    from revospeech.asr.base import BaseASR
    from revospeech.asr.result import Transcript

    class FakeASR(BaseASR):
        def transcribe(self, audio_path):
            return Transcript(text="", segments=[], language="en")

    engine = FakeASR("test-model")

    fake_manifest = MagicMock()
    fake_manifest.language = ""

    with patch("revospeech.registry.get", return_value=fake_manifest):
        result = engine.detect_language("/dev/null")

    assert result == "en"


def test_list_languages_uses_manifest_list():
    from revospeech.asr.base import BaseASR
    from revospeech.asr.result import Transcript

    class FakeASR(BaseASR):
        def transcribe(self, audio_path):
            return Transcript(text="", segments=[], language="en")

    engine = FakeASR("test-model")

    fake_manifest = MagicMock()
    fake_manifest.language = "en"
    fake_manifest.languages = ["en", "es", "fr"]

    with patch("revospeech.registry.get", return_value=fake_manifest):
        result = engine.list_languages()

    assert result == ["en", "es", "fr"]


def test_list_languages_default_to_single_language():
    from revospeech.asr.base import BaseASR
    from revospeech.asr.result import Transcript

    class FakeASR(BaseASR):
        def transcribe(self, audio_path):
            return Transcript(text="", segments=[], language="en")

    engine = FakeASR("test-model")

    fake_manifest = MagicMock()
    fake_manifest.language = "de"
    fake_manifest.languages = []

    with patch("revospeech.registry.get", return_value=fake_manifest):
        result = engine.list_languages()

    assert result == ["de"]


def test_list_languages_missing_manifest_returns_en():
    from revospeech.asr.base import BaseASR
    from revospeech.asr.result import Transcript

    class FakeASR(BaseASR):
        def transcribe(self, audio_path):
            return Transcript(text="", segments=[], language="en")

    engine = FakeASR("test-model")

    with patch("revospeech.registry.get", side_effect=KeyError("not found")):
        result = engine.list_languages()

    assert result == ["en"]
