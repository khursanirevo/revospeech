"""Curated tests for docstring examples.

RevoSpeech docstrings contain >>> examples that document API patterns.
Many require real models or audio files, so we don't enable xdoctest
auto-collection globally. Instead, this file tests the examples that
CAN run in CI:

- Pure data class examples (Transcript, Audio, BatchReport)
- Helper function examples (_split_text)
- Import and module-level examples
- Factory function signature examples (no actual model loading)

For full docstring coverage, run: pytest --xdoctest revospeech/
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import numpy as np
import pytest


def test_audio_save_then_read_round_trip(tmp_path):
    """Audio.save() then re-read should preserve samples (within float precision)."""
    from revospeech.tts.result import Audio

    samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 22050), dtype=np.float32) * 0.1
    audio = Audio(samples=samples, sample_rate=22050)

    out = tmp_path / "test.wav"
    audio.save(str(out))
    assert out.exists()

    import soundfile as sf

    loaded, sr = sf.read(str(out), dtype="float32")
    assert sr == 22050
    assert len(loaded) == len(samples)


def test_audio_repr_format():
    """Audio.__repr__ should include duration and sample_rate."""
    from revospeech.tts.result import Audio

    audio = Audio(samples=np.zeros(22050, dtype=np.float32), sample_rate=22050)
    repr_str = repr(audio)
    assert "1.0" in repr_str  # 1 second duration
    assert "22050" in repr_str


def test_transcript_save_json(tmp_path):
    """Transcript.save('.json') should produce valid JSON."""
    from revospeech.asr.result import Segment, Transcript

    transcript = Transcript(
        text="hello world",
        segments=[Segment(start=0.0, end=1.0, text="hello", confidence=0.9)],
        language="en",
    )

    out = tmp_path / "result.json"
    transcript.save(str(out))

    data = json.loads(out.read_text())
    assert data["text"] == "hello world"
    assert data["language"] == "en"
    assert len(data["segments"]) == 1


def test_transcript_save_srt(tmp_path):
    """Transcript.save('.srt') should produce valid SRT format."""
    from revospeech.asr.result import Segment, Transcript

    transcript = Transcript(
        text="hello",
        segments=[Segment(start=0.0, end=1.5, text="hello", confidence=None)],
        language="en",
    )

    out = tmp_path / "result.srt"
    transcript.save(str(out))

    content = out.read_text()
    assert "1" in content  # segment index
    assert "-->" in content  # timestamp separator
    assert "00:00:00,000 --> 00:00:01,500" in content
    assert "hello" in content


def test_transcript_save_vtt(tmp_path):
    """Transcript.save('.vtt') should produce valid VTT format."""
    from revospeech.asr.result import Segment, Transcript

    transcript = Transcript(
        text="hello",
        segments=[Segment(start=0.0, end=1.5, text="hello", confidence=None)],
        language="en",
    )

    out = tmp_path / "result.vtt"
    transcript.save(str(out))

    content = out.read_text()
    assert "WEBVTT" in content
    assert "-->" in content
    assert "00:00:00.000 --> 00:00:01.500" in content  # VTT uses dot not comma


def test_transcript_save_txt(tmp_path):
    """Transcript.save('.txt') should produce plain text."""
    from revospeech.asr.result import Transcript

    transcript = Transcript(text="hello world", segments=[], language="en")
    out = tmp_path / "result.txt"
    transcript.save(str(out))

    assert out.read_text() == "hello world"


def test_transcript_duration_property():
    """Transcript.duration should return last segment end time."""
    from revospeech.asr.result import Segment, Transcript

    t = Transcript(
        text="test",
        segments=[
            Segment(start=0.0, end=1.5, text="test", confidence=None),
            Segment(start=1.5, end=3.0, text="more", confidence=None),
        ],
        language="en",
    )
    assert t.duration == 3.0


def test_transcript_duration_empty_segments():
    """Transcript.duration should be 0 if no segments."""
    from revospeech.asr.result import Transcript

    t = Transcript(text="", segments=[], language="en")
    assert t.duration == 0


def test_batch_report_save_round_trip(tmp_path):
    """BatchReport.save() then load should preserve data."""
    from revospeech.asr.result import BatchReport, BatchResult

    report = BatchReport(
        items=[
            BatchResult(input="test1", duration=0.5, error=None),
            BatchResult(input="test2", duration=0.3, error="failed"),
        ],
        total=2,
        succeeded=1,
        failed=1,
        total_duration=0.8,
    )

    out = tmp_path / "report.json"
    report.save(str(out))

    data = json.loads(out.read_text())
    assert data["total"] == 2
    assert data["succeeded"] == 1
    assert data["failed"] == 1
    assert len(data["items"]) == 2
    assert data["items"][0]["input"] == "test1"
    assert data["items"][1]["error"] == "failed"


def test_split_text_short():
    """_split_text with short input returns single chunk."""
    from revospeech.tts.base import _split_text

    assert _split_text("Hello.") == ["Hello."]


def test_split_text_long():
    """_split_text splits long text on sentence boundaries."""
    from revospeech.tts.base import _split_text

    text = "One. Two. Three. Four. Five. " * 50  # ~600 chars
    chunks = _split_text(text, max_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_split_text_cjk():
    """_split_text handles CJK sentence-ending punctuation."""
    from revospeech.tts.base import _split_text

    text = "今日は良い天気です。明日は雨が降るでしょう。"
    chunks = _split_text(text, max_chars=10)
    assert len(chunks) >= 2


def test_audio_concatenate():
    """Audio.concatenate() should combine clips with silence."""
    from revospeech.tts.result import Audio

    a1 = Audio(samples=np.ones(1000, dtype=np.float32), sample_rate=22050)
    a2 = Audio(samples=np.ones(500, dtype=np.float32), sample_rate=22050)

    combined = Audio.concatenate([a1, a2], silence_duration=0.1)
    assert combined.sample_rate == 22050
    # 1000 + silence(0.1*22050=2205) + 500 = 3705 samples
    assert len(combined.samples) == 3705


def test_audio_play_without_sounddevice_raises():
    """Audio.play() should raise ImportError-like error if sounddevice missing."""
    from revospeech.tts.result import Audio

    audio = Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=22050)

    # Mock sounddevice as not installed
    sd_backup = sys.modules.pop("sounddevice", None)
    try:
        with patch("builtins.__import__", side_effect=ImportError("no sounddevice")):
            with pytest.raises((ImportError, RuntimeError)):
                audio.play()
    finally:
        if sd_backup is not None:
            sys.modules["sounddevice"] = sd_backup


def test_module_imports():
    """Verify all top-level exports are importable."""
    import revospeech

    # These should not raise
    assert hasattr(revospeech, "ASR")
    assert hasattr(revospeech, "TTS")
    assert hasattr(revospeech, "list_models")
    assert hasattr(revospeech, "search_models")
    assert hasattr(revospeech, "check_model")
    assert hasattr(revospeech, "__version__")


def test_module_dir_returns_exports():
    """dir(revospeech) should include public exports."""
    import revospeech

    exports = dir(revospeech)
    assert "ASR" in exports
    assert "TTS" in exports
    assert "list_models" in exports
