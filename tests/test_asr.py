"""Tests for ASR engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from revospeech.asr.result import Segment, Transcript


def test_segment_creation():
    s = Segment(start=0.0, end=1.0, text="hello", confidence=0.95)
    assert s.text == "hello"
    assert s.confidence == 0.95


def test_transcript_creation():
    s = Segment(start=0.0, end=1.0, text="hello", confidence=0.95)
    t = Transcript(text="hello world", segments=[s], language="en")
    assert t.text == "hello world"
    assert len(t.segments) == 1


@patch("revospeech.asr.sherpa_engine.sherpa_onnx")
@patch("revospeech.asr.sherpa_engine.ensure_model")
@patch("revospeech.asr.sherpa_engine.get")
def test_asr_transcribe(mock_get, mock_ensure, mock_sherpa, sample_wav, tmp_path):
    """Test ASR transcribe with mocked sherpa_onnx."""
    from revospeech.asr.sherpa_engine import SherpaOnnxASR
    from revospeech.registry.manifest import ModelManifest

    # Setup manifest
    manifest = ModelManifest(
        name="test-asr",
        task="asr",
        backend="sherpa-onnx",
        model_type="transducer",
        model_url="http://example.com/model.tar.bz2",
        sample_rate=16000,
        language="en",
        description="Test",
        files={
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "joiner": "joiner.onnx",
            "tokens": "tokens.txt",
        },
    )
    mock_get.return_value = manifest
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for f in ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]:
        (model_dir / f).write_text("dummy")
    mock_ensure.return_value = model_dir

    # Mock recognizer
    result = MagicMock()
    result.text = "HELLO WORLD"
    result.timestamps = [0.0, 0.5]
    result.lang = "en"

    stream = MagicMock()
    stream.result = result

    recognizer = MagicMock()
    recognizer.create_stream.return_value = stream
    mock_sherpa_onnx = mock_sherpa
    mock_sherpa_onnx.OfflineRecognizer.from_transducer.return_value = recognizer

    # Run
    asr = SherpaOnnxASR("test-asr", device="cpu")
    transcript = asr.transcribe(sample_wav)

    assert isinstance(transcript, Transcript)
    assert transcript.text == "HELLO WORLD"
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "HELLO"
    assert transcript.language == "en"


@patch("revospeech.asr.sherpa_engine.SherpaOnnxASR")
def test_asr_factory(mock_cls):
    """Test the ASR() factory function."""
    from revospeech.asr import ASR

    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance

    # Need to register a manifest first
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import register

    register(
        ModelManifest(
            name="test",
            task="asr",
            backend="sherpa-onnx",
            model_type="transducer",
            model_url="",
            sample_rate=16000,
            language="en",
            description="",
        )
    )
    ASR("test")
    mock_cls.assert_called_once_with("test", "auto")


def test_asr_unsupported_backend():
    """Test that unsupported backend raises ValueError."""
    from revospeech.asr import ASR
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import register

    register(
        ModelManifest(
            name="bad",
            task="asr",
            backend="nonexistent",
            model_type="transducer",
            model_url="",
            sample_rate=16000,
            language="en",
            description="",
        )
    )
    with pytest.raises(ValueError, match="Supported backends: sherpa-onnx"):
        ASR("bad")


# ---------------------------------------------------------------------------
# ASR factory: api_key persistence, auto-select, did-you-mean, API-mode
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_asr_registry():
    """Clear registry before/after each test to isolate from built-in models."""
    from revospeech.registry.registry import _models

    saved = dict(_models)
    _models.clear()
    yield
    _models.clear()
    _models.update(saved)


def _register_sherpa_asr(name="test-asr", model_url=""):
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import register

    register(
        ModelManifest(
            name=name,
            task="asr",
            backend="sherpa-onnx",
            model_type="transducer",
            model_url=model_url,
            sample_rate=16000,
            language="en",
            description="",
        )
    )


def test_asr_with_api_key_persists(monkeypatch):
    """Passing api_key to ASR() persists it via set_api_key."""
    captured = []
    monkeypatch.setattr("revospeech.asr.set_api_key", lambda key: captured.append(key))
    _register_sherpa_asr()
    from revospeech.asr import ASR

    with patch("revospeech.asr.sherpa_engine.SherpaOnnxASR"):
        ASR("test-asr", api_key="rv-test12345")
    assert captured == ["rv-test12345"]


def test_asr_auto_select_no_models_registered():
    """No model_name + no ASR models → RevosModelError with catalog hint."""
    from revospeech.asr import ASR
    from revospeech.exceptions import RevosModelError

    with pytest.raises(RevosModelError, match="No ASR model specified") as exc:
        ASR()
    assert "catalog list" in exc.value.suggestion


def test_asr_auto_select_no_ready_with_available(monkeypatch):
    """No ready models, but some registered → suggestion lists them."""
    from revospeech.registry.status import ModelStatus

    def fake_list(**kw):
        if kw.get("status") == "ready":
            return []
        return [
            ModelStatus(
                name="not-ready-asr",
                task="asr",
                mode="local",
                status="needs-download",
                installed=False,
                size_mb=80.0,
                capabilities=[],
                languages=["en"],
            )
        ]

    monkeypatch.setattr("revospeech.asr.list_model_statuses", fake_list)

    from revospeech.asr import ASR
    from revospeech.exceptions import RevosModelError

    with pytest.raises(RevosModelError) as exc:
        ASR()
    assert "Download a model first" in exc.value.suggestion
    assert "not-ready-asr" in exc.value.suggestion


def test_asr_auto_select_smallest_ready(monkeypatch):
    """Auto-select picks the smallest ready ASR model by size_mb."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import register
    from revospeech.registry.status import ModelStatus

    register(
        ModelManifest(
            name="big-asr",
            task="asr",
            backend="sherpa-onnx",
            model_type="transducer",
            model_url="",
            sample_rate=16000,
            language="en",
            description="",
            size_mb=200.0,
        )
    )
    register(
        ModelManifest(
            name="small-asr",
            task="asr",
            backend="sherpa-onnx",
            model_type="transducer",
            model_url="",
            sample_rate=16000,
            language="en",
            description="",
            size_mb=50.0,
        )
    )

    fake_statuses = [
        ModelStatus(
            name="big-asr",
            task="asr",
            mode="local",
            status="ready",
            installed=True,
            size_mb=200.0,
            capabilities=[],
            languages=["en"],
        ),
        ModelStatus(
            name="small-asr",
            task="asr",
            mode="local",
            status="ready",
            installed=True,
            size_mb=50.0,
            capabilities=[],
            languages=["en"],
        ),
    ]
    monkeypatch.setattr(
        "revospeech.asr.list_model_statuses", lambda **kw: fake_statuses
    )

    captured = []
    monkeypatch.setattr(
        "revospeech.asr.sherpa_engine.SherpaOnnxASR",
        lambda *a, **kw: captured.append(a[0]) or MagicMock(),
    )

    from revospeech.asr import ASR

    ASR()
    assert captured == ["small-asr"]


def test_asr_unknown_model_with_did_you_mean():
    """Unknown ASR model triggers KeyError with did-you-mean hint."""
    _register_sherpa_asr("zipformer-v2")
    from revospeech.asr import ASR

    with pytest.raises(KeyError, match="not found"):
        ASR("zipformer")


def test_asr_api_mode_no_key_raises_config_error(monkeypatch):
    """API-mode ASR without key → RevosConfigError."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import register

    register(
        ModelManifest(
            name="api-asr",
            task="asr",
            backend="revolab",
            model_type="revolab",
            model_url="",
            sample_rate=16000,
            language="en",
            description="API ASR",
            mode="api",
        )
    )
    monkeypatch.setattr("revospeech.config.get_api_key", lambda *a, **kw: None)

    from revospeech.asr import ASR
    from revospeech.exceptions import RevosConfigError

    with pytest.raises(RevosConfigError, match="requires an API key"):
        ASR("api-asr")


def test_asr_api_mode_with_key_raises_not_implemented(monkeypatch):
    """API-mode ASR with key set → NotImplementedError."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import register

    register(
        ModelManifest(
            name="api-asr",
            task="asr",
            backend="revolab",
            model_type="revolab",
            model_url="",
            sample_rate=16000,
            language="en",
            description="API ASR",
            mode="api",
        )
    )
    monkeypatch.setattr(
        "revospeech.config.get_api_key", lambda *a, **kw: "rv-test12345"
    )

    from revospeech.asr import ASR

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        ASR("api-asr")
