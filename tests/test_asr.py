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
