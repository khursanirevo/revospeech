"""Tests for revolab API backend skeleton."""

from __future__ import annotations

import base64

import pytest


def test_revolab_client_requires_api_key(monkeypatch):
    from revospeech.exceptions import RevosConfigError
    from revospeech.http_client import RevolabClient

    monkeypatch.setattr("revospeech.http_client.get_api_key", lambda *a, **kw: None)
    with pytest.raises(RevosConfigError, match="API key"):
        RevolabClient("https://api.example.com")


def test_revolab_client_requires_httpx(monkeypatch):
    # Mock httpx import failure
    import builtins

    from revospeech.exceptions import RevosConfigError
    from revospeech.http_client import RevolabClient

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    monkeypatch.setattr(
        "revospeech.http_client.get_api_key", lambda *a, **kw: "rv-test"
    )

    with pytest.raises(RevosConfigError, match="httpx"):
        RevolabClient("https://api.example.com")


def test_revolab_client_initializes_with_key(monkeypatch):
    """Smoke test: client initializes when key + httpx are available."""
    from revospeech.http_client import RevolabClient

    monkeypatch.setattr(
        "revospeech.http_client.get_api_key", lambda *a, **kw: "rv-test12345678"
    )
    try:
        client = RevolabClient("https://api.example.com")
    except ImportError:
        pytest.skip("httpx not installed")

    assert client.api_key == "rv-test12345678"
    assert client.endpoint == "https://api.example.com"
    client.close()


def test_mask_key(monkeypatch):
    from revospeech.http_client import RevolabClient

    monkeypatch.setattr(
        "revospeech.http_client.get_api_key", lambda *a, **kw: "rv-abcdefghijkl"
    )
    try:
        client = RevolabClient("https://api.example.com")
    except ImportError:
        pytest.skip("httpx not installed")

    # First 4 + last 4 of "rv-abcdefghijkl" = "rv-a" + "ijkl"
    assert client._mask_key() == "rv-a...ijkl"
    client.close()


def test_mask_key_short(monkeypatch):
    """Short keys should be fully masked."""
    from revospeech.http_client import RevolabClient

    monkeypatch.setattr("revospeech.http_client.get_api_key", lambda *a, **kw: "abc")
    try:
        client = RevolabClient("https://api.example.com")
    except ImportError:
        pytest.skip("httpx not installed")

    assert client._mask_key() == "***"
    client.close()


def test_revolab_asr_parse_response():
    """ASR response parser should handle the documented schema."""
    from revospeech.asr.revolab_engine import RevolabASR

    # Bypass __init__ to avoid network/client setup
    engine = RevolabASR.__new__(RevolabASR)
    engine.model_name = "test"
    engine.device = "cpu"

    fake_response = {
        "text": "hello world",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello", "confidence": 0.95},
            {"start": 1.0, "end": 2.0, "text": "world", "confidence": 0.93},
        ],
    }

    result = engine._parse_response(fake_response)
    assert result.text == "hello world"
    assert result.language == "en"
    assert len(result.segments) == 2
    assert result.segments[0].text == "hello"


def test_revolab_asr_parse_response_empty():
    from revospeech.asr.revolab_engine import RevolabASR

    engine = RevolabASR.__new__(RevolabASR)
    engine.model_name = "test"
    engine.device = "cpu"

    result = engine._parse_response({})
    assert result.text == ""
    assert result.segments == []
    assert result.language == "en"


def test_revolab_tts_fetch_audio_from_base64():
    from revospeech.tts.revolab_engine import RevolabTTS

    engine = RevolabTTS.__new__(RevolabTTS)
    engine.model_name = "test"
    engine.device = "cpu"

    audio_data = b"fake audio bytes"
    encoded = base64.b64encode(audio_data).decode()

    result = engine._fetch_audio({"audio_base64": encoded})
    assert result == audio_data


def test_revolab_tts_fetch_audio_malformed():
    from revospeech.exceptions import RevosEngineError
    from revospeech.tts.revolab_engine import RevolabTTS

    engine = RevolabTTS.__new__(RevolabTTS)
    engine.model_name = "test"
    engine.device = "cpu"

    with pytest.raises(RevosEngineError, match="Malformed"):
        engine._fetch_audio({"unexpected_key": "value"})
