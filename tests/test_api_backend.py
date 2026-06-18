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


def test_get_raw_401_raises_config_error(monkeypatch):
    """get_raw HTTP 401 → RevosConfigError."""
    from revospeech.exceptions import RevosConfigError

    client = _build_client_with_mock_http(monkeypatch)
    response = client._MockResponse(401)
    client._client.request = lambda *a, **kw: response

    with pytest.raises(RevosConfigError, match="401"):
        client.get_raw("/audio/123")
    client.close()


def test_get_raw_403_raises_config_error(monkeypatch):
    """get_raw HTTP 403 → RevosConfigError with permission hint."""
    from revospeech.exceptions import RevosConfigError

    client = _build_client_with_mock_http(monkeypatch)
    response = client._MockResponse(403)
    client._client.request = lambda *a, **kw: response

    with pytest.raises(RevosConfigError, match="403"):
        client.get_raw("/audio/123")
    client.close()


def test_get_raw_429_retries_then_raises(monkeypatch):
    """get_raw HTTP 429 retries, then raises RevosEngineError."""
    from revospeech.exceptions import RevosEngineError

    client = _build_client_with_mock_http(monkeypatch)
    client.max_retries = 2
    response = client._MockResponse(429)
    client._client.request = lambda *a, **kw: response

    with pytest.raises(RevosEngineError, match="Rate limit"):
        client.get_raw("/audio/123")
    client.close()


def test_get_raw_5xx_retries_then_raises(monkeypatch):
    """get_raw HTTP 500 retries, then raises RevosEngineError."""
    from revospeech.exceptions import RevosEngineError

    client = _build_client_with_mock_http(monkeypatch)
    client.max_retries = 2
    response = client._MockResponse(503)
    client._client.request = lambda *a, **kw: response

    with pytest.raises(RevosEngineError, match="Server error"):
        client.get_raw("/audio/123")
    client.close()


def test_get_raw_network_error_retries(monkeypatch):
    """get_raw network errors retry, then raise RevosEngineError."""
    from revospeech.exceptions import RevosEngineError

    client = _build_client_with_mock_http(monkeypatch)
    client.max_retries = 2

    def always_fail(*a, **kw):
        raise client._httpx.HTTPError("network down")

    client._client.request = always_fail

    with pytest.raises(RevosEngineError, match="Network error"):
        client.get_raw("/audio/123")
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


def test_revolab_asr_transcribe_calls_api(monkeypatch, tmp_path):
    """transcribe() uploads audio and parses response."""
    import numpy as np
    import soundfile as sf

    from revospeech.asr.revolab_engine import RevolabASR

    wav = tmp_path / "test.wav"
    samples = np.zeros(16000, dtype=np.float32)
    sf.write(str(wav), samples, 16000)

    engine = RevolabASR.__new__(RevolabASR)
    engine.model_name = "revolab-asr-v1"
    engine.device = "cpu"

    class _MockClient:
        def __init__(self):
            self.calls = []

        def post(self, path, **kwargs):
            self.calls.append((path, kwargs))
            return {
                "text": "hello world",
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "hello", "confidence": 0.9}
                ],
            }

        def close(self):
            pass

    engine._client = _MockClient()

    result = engine.transcribe(str(wav), language="en", word_timestamps=True)
    assert result.text == "hello world"
    assert result.language == "en"
    assert len(result.segments) == 1
    assert engine._client.calls[0][0] == "/asr/transcribe"


def test_revolab_asr_transcribe_wraps_errors(tmp_path):
    """transcribe() wraps non-RevosEngineError exceptions."""
    import numpy as np
    import soundfile as sf

    from revospeech.asr.revolab_engine import RevolabASR
    from revospeech.exceptions import RevosEngineError

    wav = tmp_path / "test.wav"
    samples = np.zeros(16000, dtype=np.float32)
    sf.write(str(wav), samples, 16000)

    engine = RevolabASR.__new__(RevolabASR)
    engine.model_name = "test"
    engine.device = "cpu"

    class _MockClient:
        def post(self, path, **kwargs):
            raise RuntimeError("network gone")

        def close(self):
            pass

    engine._client = _MockClient()

    with pytest.raises(RevosEngineError, match="ASR API call failed"):
        engine.transcribe(str(wav))


def test_revolab_asr_close(monkeypatch):
    """close() delegates to underlying client."""
    from revospeech.asr.revolab_engine import RevolabASR

    engine = RevolabASR.__new__(RevolabASR)
    engine.model_name = "test"
    engine.device = "cpu"

    closed = [False]

    class _MockClient:
        def close(self):
            closed[0] = True

    engine._client = _MockClient()
    engine.close()
    assert closed[0]


# ---------------------------------------------------------------------------
# RevolabASR: __init__ branches + RevosEngineError pass-through
# ---------------------------------------------------------------------------
def test_revolab_asr_init_with_manifest_endpoint(monkeypatch):
    """__init__ reads api_endpoint from manifest when present."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register

    saved = dict(_models)
    _models.clear()
    register(
        ModelManifest(
            name="revolab-asr-v1",
            task="asr",
            backend="revolab",
            model_type="revolab",
            model_url="",
            sample_rate=16000,
            language="en",
            description="",
            mode="api",
            api_endpoint="https://custom.asr.api/v2",
        )
    )
    captured = {}
    monkeypatch.setattr(
        "revospeech.http_client.get_api_key", lambda *a, **kw: "rv-test12345678"
    )

    class _FakeClient:
        def __init__(self, *args, endpoint=None, **kwargs):
            captured["endpoint"] = endpoint

    monkeypatch.setattr("revospeech.asr.revolab_engine.RevolabClient", _FakeClient)
    try:
        from revospeech.asr.revolab_engine import RevolabASR

        RevolabASR("revolab-asr-v1")
    finally:
        _models.clear()
        _models.update(saved)
    assert captured["endpoint"] == "https://custom.asr.api/v2"


def test_revolab_asr_init_unknown_model_falls_back(monkeypatch):
    """__init__ falls back to default endpoint when model not registered."""
    captured = {}
    monkeypatch.setattr(
        "revospeech.http_client.get_api_key", lambda *a, **kw: "rv-test12345678"
    )

    class _FakeClient:
        def __init__(self, *args, endpoint=None, **kwargs):
            captured["endpoint"] = endpoint

    monkeypatch.setattr("revospeech.asr.revolab_engine.RevolabClient", _FakeClient)
    from revospeech.asr.revolab_engine import RevolabASR

    RevolabASR("totally-unknown-model")
    assert captured["endpoint"] == "https://api.revolab.ai/v1"


def test_revolab_asr_transcribe_passes_engine_error_through(tmp_path):
    """RevosEngineError from client.post is re-raised, not wrapped."""
    import numpy as np
    import soundfile as sf

    from revospeech.asr.revolab_engine import RevolabASR
    from revospeech.exceptions import RevosEngineError

    wav = tmp_path / "test.wav"
    samples = np.zeros(16000, dtype=np.float32)
    sf.write(str(wav), samples, 16000)

    class _MockClient:
        def post(self, path, **kwargs):
            raise RevosEngineError("rate limited", suggestion="retry later")

        def close(self):
            pass

    engine = RevolabASR.__new__(RevolabASR)
    engine.model_name = "test"
    engine.device = "cpu"
    engine._client = _MockClient()

    with pytest.raises(RevosEngineError, match="rate limited"):
        engine.transcribe(str(wav))


def test_revolab_tts_synthesize_calls_api(monkeypatch):
    """synthesize() posts text and fetches audio."""
    import base64 as b64
    import io

    import numpy as np
    import soundfile as sf

    from revospeech.tts.revolab_engine import RevolabTTS

    engine = RevolabTTS.__new__(RevolabTTS)
    engine.model_name = "revolab-tts-v1"
    engine.device = "cpu"

    buf = io.BytesIO()
    samples = np.zeros(22050, dtype=np.float32)
    sf.write(buf, samples, 22050, format="WAV")
    encoded = b64.b64encode(buf.getvalue()).decode()

    class _MockClient:
        def __init__(self):
            self.posts = []

        def post(self, path, **kwargs):
            self.posts.append((path, kwargs))
            return {"audio_base64": encoded, "sample_rate": 22050}

        def close(self):
            pass

    engine._client = _MockClient()

    audio = engine.synthesize("hello world")
    assert engine._client.posts[0][0] == "/tts/synthesize"
    assert audio.sample_rate == 22050
    assert len(audio.samples) == 22050


def test_revolab_tts_close():
    """close() delegates to underlying client."""
    from revospeech.tts.revolab_engine import RevolabTTS

    engine = RevolabTTS.__new__(RevolabTTS)
    engine.model_name = "test"
    engine.device = "cpu"

    closed = [False]

    class _MockClient:
        def close(self):
            closed[0] = True

    engine._client = _MockClient()
    engine.close()
    assert closed[0]


# ---------------------------------------------------------------------------
# RevolabTTS: __init__ branches, ref_audio path, error wrapping, audio_url fetch
# ---------------------------------------------------------------------------
def test_revolab_tts_init_with_manifest_endpoint(monkeypatch):
    """__init__ reads api_endpoint from manifest when present."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register
    from revospeech.tts.revolab_engine import RevolabTTS

    saved = dict(_models)
    _models.clear()
    register(
        ModelManifest(
            name="revolab-tts-v1",
            task="tts",
            backend="revolab",
            model_type="revolab",
            model_url="",
            sample_rate=22050,
            language="en",
            description="",
            mode="api",
            api_endpoint="https://custom.api/v2",
            files={},
        )
    )
    captured = {}
    monkeypatch.setattr(
        "revospeech.http_client.get_api_key", lambda *a, **kw: "rv-test12345678"
    )

    class _FakeClient:
        def __init__(self, *args, endpoint=None, **kwargs):
            captured["endpoint"] = endpoint

    monkeypatch.setattr("revospeech.tts.revolab_engine.RevolabClient", _FakeClient)
    try:
        RevolabTTS("revolab-tts-v1")
    finally:
        _models.clear()
        _models.update(saved)
    assert captured["endpoint"] == "https://custom.api/v2"


def test_revolab_tts_init_unknown_model_falls_back(monkeypatch):
    """__init__ falls back to default endpoint when model not registered."""
    from revospeech.tts.revolab_engine import RevolabTTS

    captured = {}
    monkeypatch.setattr(
        "revospeech.http_client.get_api_key", lambda *a, **kw: "rv-test12345678"
    )

    class _FakeClient:
        def __init__(self, *args, endpoint=None, **kwargs):
            captured["endpoint"] = endpoint

    monkeypatch.setattr("revospeech.tts.revolab_engine.RevolabClient", _FakeClient)
    RevolabTTS("totally-unknown-model")
    assert captured["endpoint"] == "https://api.revolab.ai/v1"


def test_revolab_tts_synthesize_with_ref_audio(tmp_path, monkeypatch):
    """synthesize() with ref_audio reads file and includes it in request."""
    import io

    import numpy as np
    import soundfile as sf

    from revospeech.tts.revolab_engine import RevolabTTS

    ref = tmp_path / "ref.wav"
    samples = np.zeros(16000, dtype=np.float32)
    sf.write(str(ref), samples, 16000, format="WAV")

    buf = io.BytesIO()
    out_samples = np.zeros(22050, dtype=np.float32)
    sf.write(buf, out_samples, 22050, format="WAV")
    encoded = base64.b64encode(buf.getvalue()).decode()

    captured = {}

    class _MockClient:
        def post(self, path, **kwargs):
            captured["kwargs"] = kwargs
            return {"audio_base64": encoded}

        def close(self):
            pass

    engine = RevolabTTS.__new__(RevolabTTS)
    engine.model_name = "test"
    engine.device = "cpu"
    engine._client = _MockClient()

    engine.synthesize("clone this", ref_audio=str(ref), ref_text="reference text")
    assert captured["kwargs"]["files"]["ref_audio"][0] == "ref.wav"
    assert captured["kwargs"]["data"]["ref_text"] == "reference text"


def test_revolab_tts_synthesize_saves_output(tmp_path):
    """synthesize() writes to output_path when provided."""
    import io

    import numpy as np
    import soundfile as sf

    from revospeech.tts.revolab_engine import RevolabTTS

    buf = io.BytesIO()
    samples = np.zeros(22050, dtype=np.float32)
    sf.write(buf, samples, 22050, format="WAV")
    encoded = base64.b64encode(buf.getvalue()).decode()

    class _MockClient:
        def post(self, path, **kwargs):
            return {"audio_base64": encoded}

        def close(self):
            pass

    engine = RevolabTTS.__new__(RevolabTTS)
    engine.model_name = "test"
    engine.device = "cpu"
    engine._client = _MockClient()

    out = tmp_path / "out.wav"
    engine.synthesize("hello", output_path=str(out))
    assert out.exists()


def test_revolab_tts_synthesize_wraps_unknown_exception():
    """Non-RevosEngineError exceptions are wrapped in RevosEngineError."""
    from revospeech.exceptions import RevosEngineError
    from revospeech.tts.revolab_engine import RevolabTTS

    class _MockClient:
        def post(self, path, **kwargs):
            raise ConnectionError("network down")

        def close(self):
            pass

    engine = RevolabTTS.__new__(RevolabTTS)
    engine.model_name = "test"
    engine.device = "cpu"
    engine._client = _MockClient()

    with pytest.raises(RevosEngineError, match="TTS API call failed"):
        engine.synthesize("hello")


def test_revolab_tts_synthesize_passes_engine_error_through():
    """RevosEngineError from client.post is re-raised, not wrapped."""
    from revospeech.exceptions import RevosEngineError
    from revospeech.tts.revolab_engine import RevolabTTS

    class _MockClient:
        def post(self, path, **kwargs):
            raise RevosEngineError("auth failed", suggestion="check key")

        def close(self):
            pass

    engine = RevolabTTS.__new__(RevolabTTS)
    engine.model_name = "test"
    engine.device = "cpu"
    engine._client = _MockClient()

    with pytest.raises(RevosEngineError, match="auth failed"):
        engine.synthesize("hello")


def test_revolab_tts_fetch_audio_from_url():
    """_fetch_audio uses get_raw when response has audio_url."""
    from revospeech.tts.revolab_engine import RevolabTTS

    fetched = []

    class _MockClient:
        def get_raw(self, url):
            fetched.append(url)
            return b"raw audio bytes"

    engine = RevolabTTS.__new__(RevolabTTS)
    engine.model_name = "test"
    engine.device = "cpu"
    engine._client = _MockClient()

    result = engine._fetch_audio({"audio_url": "https://cdn.example.com/a.wav"})
    assert result == b"raw audio bytes"
    assert fetched == ["https://cdn.example.com/a.wav"]
