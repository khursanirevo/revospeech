"""Tests for BytesIO in-memory audio support (US-025)."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from revospeech.asr.audio import read_waveform


def _make_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> io.BytesIO:
    """Create an in-memory WAV file (as BytesIO) from float32 samples."""
    buf = io.BytesIO()
    int_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
    buf.seek(0)
    return buf


def _write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    sf.write(str(path), samples, sr)


def test_read_waveform_accepts_bytesio():
    """read_waveform should accept an io.BytesIO of encoded WAV audio."""
    sr = 16000
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    buf = _make_wav_bytes(samples, sr)

    data, out_sr = read_waveform(buf)

    assert out_sr == sr
    assert data.dtype == np.float32
    assert len(data) == sr


def test_read_waveform_bytesio_resamples():
    """read_waveform should resample a BytesIO input to target_sr."""
    sr = 44100
    target_sr = 16000
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    buf = _make_wav_bytes(samples, sr)

    data, out_sr = read_waveform(buf, target_sr=target_sr)

    assert out_sr == target_sr
    assert len(data) == target_sr


def test_read_waveform_rejects_bad_type():
    """read_waveform should raise TypeError for unsupported inputs."""
    with pytest.raises(TypeError, match="Unsupported"):
        read_waveform(12345)  # type: ignore[arg-type]


def test_read_waveform_rejects_none():
    """read_waveform should raise TypeError for None input."""
    with pytest.raises(TypeError, match="Unsupported"):
        read_waveform(None)  # type: ignore[arg-type]


def test_read_waveform_path_still_works(tmp_path: Path):
    """Path-based input must continue to work after BytesIO support is added."""
    sr = 8000
    samples = np.zeros(sr, dtype=np.float32)
    wav_path = tmp_path / "test.wav"
    _write_wav(wav_path, samples, sr)

    data, out_sr = read_waveform(str(wav_path), target_sr=16000)

    assert out_sr == 16000
    # Resampling from 8000 -> 16000 doubles length
    assert len(data) >= sr

    # Also verify Path objects work
    data2, out_sr2 = read_waveform(wav_path, target_sr=16000)
    assert out_sr2 == 16000
    assert len(data2) == len(data)


def test_read_waveform_bytesio_handles_stereo():
    """read_waveform should downmix stereo BytesIO audio to mono."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 880 * t)
    stereo = np.column_stack([left, right])

    # Write stereo WAV to BytesIO directly via soundfile
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    buf.seek(0)

    data, out_sr = read_waveform(buf)

    assert out_sr == sr
    assert data.ndim == 1
    assert len(data) == sr


def test_read_waveform_generic_file_like():
    """read_waveform should accept any object with a read() method."""
    sr = 16000
    samples = np.zeros(sr, dtype=np.float32)
    buf = _make_wav_bytes(samples, sr)

    # Wrap in a generic file-like proxy to exercise the hasattr(audio, "read") path
    class GenericStream:
        def __init__(self, inner: io.BytesIO) -> None:
            self._inner = inner

        def read(self, size: int = -1) -> bytes:
            return self._inner.read(size)

        def seek(self, pos: int, whence: int = 0) -> int:
            return self._inner.seek(pos, whence)

        def tell(self) -> int:
            return self._inner.tell()

    stream = GenericStream(buf)
    data, out_sr = read_waveform(stream)  # type: ignore[arg-type]

    assert out_sr == sr
    assert len(data) == sr


@patch("revospeech.asr.sherpa_engine.sherpa_onnx")
@patch("revospeech.asr.sherpa_engine.ensure_model")
@patch("revospeech.asr.sherpa_engine.get")
def test_transcribe_accepts_bytesio(
    mock_get: Any,
    mock_ensure: Any,
    mock_sherpa: Any,
    tmp_path: Path,
):
    """SherpaOnnxASR.transcribe should accept BytesIO without TypeError."""
    from revospeech.asr.sherpa_engine import SherpaOnnxASR
    from revospeech.registry.manifest import ModelManifest

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
    for fname in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        (model_dir / fname).write_text("dummy")
    mock_ensure.return_value = model_dir

    result = MagicMock()
    result.text = "HELLO WORLD"
    result.timestamps = [0.0, 0.5]
    result.lang = "en"

    stream = MagicMock()
    stream.result = result

    recognizer = MagicMock()
    recognizer.create_stream.return_value = stream
    mock_sherpa.OfflineRecognizer.from_transducer.return_value = recognizer

    asr = SherpaOnnxASR("test-asr", device="cpu")

    # Build a BytesIO containing real WAV bytes
    samples = np.zeros(16000, dtype=np.float32)
    buf = _make_wav_bytes(samples, 16000)

    transcript = asr.transcribe(buf)

    # Verify the recognizer received the waveform
    recognizer.create_stream.assert_called_once()
    stream.accept_waveform.assert_called_once()
    # First positional arg is sample_rate, second is the samples array
    call_args = stream.accept_waveform.call_args
    passed_samples = call_args.args[1]
    assert isinstance(passed_samples, np.ndarray)
    assert passed_samples.dtype == np.float32

    # Verify we got the mocked transcript back
    assert transcript.text == "HELLO WORLD"
    assert transcript.language == "en"


@patch("revospeech.asr.sherpa_engine.sherpa_onnx")
@patch("revospeech.asr.sherpa_engine.ensure_model")
@patch("revospeech.asr.sherpa_engine.get")
def test_transcribe_path_still_works(
    mock_get: Any,
    mock_ensure: Any,
    mock_sherpa: Any,
    tmp_path: Path,
):
    """Path-based transcription must still work alongside BytesIO support."""
    from revospeech.asr.sherpa_engine import SherpaOnnxASR
    from revospeech.registry.manifest import ModelManifest

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
    for fname in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        (model_dir / fname).write_text("dummy")
    mock_ensure.return_value = model_dir

    result = MagicMock()
    result.text = "PATH INPUT"
    result.timestamps = []
    result.lang = "en"
    stream = MagicMock()
    stream.result = result
    recognizer = MagicMock()
    recognizer.create_stream.return_value = stream
    mock_sherpa.OfflineRecognizer.from_transducer.return_value = recognizer

    asr = SherpaOnnxASR("test-asr", device="cpu")

    samples = np.zeros(16000, dtype=np.float32)
    wav_path = tmp_path / "audio.wav"
    _write_wav(wav_path, samples, 16000)

    transcript = asr.transcribe(str(wav_path))

    assert transcript.text == "PATH INPUT"
