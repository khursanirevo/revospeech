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


def _build_client_with_mock_http(monkeypatch):
    """Build a RevolabClient with a mock httpx Client for retry testing.

    Tests patch ``client._client.request`` to return canned responses.
    """
    from revospeech.http_client import RevolabClient

    monkeypatch.setattr(
        "revospeech.http_client.get_api_key", lambda *a, **kw: "rv-test12345678"
    )
    try:
        client = RevolabClient("https://api.example.com")
    except ImportError:
        pytest.skip("httpx not installed")

    class _MockResponse:
        def __init__(self, status_code, json_data=None, content=b""):
            self.status_code = status_code
            self._json = json_data
            self.content = content

        def json(self):
            return self._json or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise client._httpx.HTTPStatusError(
                    "mock error", request=None, response=self
                )

    client._MockResponse = _MockResponse
    return client


def test_request_success_returns_json(monkeypatch):
    """Successful POST returns parsed JSON."""
    client = _build_client_with_mock_http(monkeypatch)
    response = client._MockResponse(200, json_data={"text": "hi"})
    client._client.request = lambda *a, **kw: response

    result = client.post("/asr/transcribe", json={"audio": "x"})
    assert result == {"text": "hi"}
    client.close()


def test_request_401_raises_config_error(monkeypatch):
    """HTTP 401 → RevosConfigError with actionable suggestion."""
    from revospeech.exceptions import RevosConfigError

    client = _build_client_with_mock_http(monkeypatch)
    response = client._MockResponse(401)
    client._client.request = lambda *a, **kw: response

    with pytest.raises(RevosConfigError, match="401"):
        client.get("/asr/result")
    client.close()


def test_request_429_retries_then_raises(monkeypatch):
    """HTTP 429 retries max_retries times, then raises RevosEngineError."""
    from revospeech.exceptions import RevosEngineError

    client = _build_client_with_mock_http(monkeypatch)
    client.max_retries = 2  # speed up test
    response = client._MockResponse(429)
    client._client.request = lambda *a, **kw: response

    with pytest.raises(RevosEngineError, match="Rate limit"):
        client.post("/asr/transcribe")
    client.close()


def test_request_5xx_retries_then_raises(monkeypatch):
    """HTTP 500 retries, then raises RevosEngineError."""
    from revospeech.exceptions import RevosEngineError

    client = _build_client_with_mock_http(monkeypatch)
    client.max_retries = 2
    response = client._MockResponse(503)
    client._client.request = lambda *a, **kw: response

    with pytest.raises(RevosEngineError, match="Server error"):
        client.post("/asr/transcribe")
    client.close()


def test_request_network_error_retries(monkeypatch):
    """Network errors are retried, then raise RevosEngineError."""
    from revospeech.exceptions import RevosEngineError

    client = _build_client_with_mock_http(monkeypatch)
    client.max_retries = 2

    def always_fail(*a, **kw):
        raise client._httpx.HTTPError("network down")

    client._client.request = always_fail

    with pytest.raises(RevosEngineError, match="Network error"):
        client.post("/asr/transcribe")
    client.close()


def test_client_context_manager(monkeypatch):
    """Client supports `with` statement and closes on exit."""
    from revospeech.http_client import RevolabClient

    monkeypatch.setattr(
        "revospeech.http_client.get_api_key", lambda *a, **kw: "rv-test12345678"
    )
    try:
        with RevolabClient("https://api.example.com") as client:
            assert client.api_key == "rv-test12345678"
    except ImportError:
        pytest.skip("httpx not installed")


def test_get_raw_returns_bytes(monkeypatch):
    """get_raw returns response bytes for binary downloads."""
    client = _build_client_with_mock_http(monkeypatch)
    response = client._MockResponse(200, content=b"audio bytes")
    client._client.request = lambda *a, **kw: response

    result = client.get_raw("/audio/123")
    assert result == b"audio bytes"
    client.close()


def test_request_403_raises_config_error(monkeypatch):
    """HTTP 403 → RevosConfigError with permission suggestion."""
    from revospeech.exceptions import RevosConfigError

    client = _build_client_with_mock_http(monkeypatch)
    response = client._MockResponse(403)
    client._client.request = lambda *a, **kw: response

    with pytest.raises(RevosConfigError, match="403"):
        client.get("/asr/result")
    client.close()
