"""Tests for new features added in TODO completion sprint."""

from __future__ import annotations

import json

import numpy as np
import pytest

from revospeech.registry.manifest import ModelManifest
from revospeech.registry.registry import _models, register


@pytest.fixture(autouse=True)
def clear_registry():
    _models.clear()
    yield
    _models.clear()


def _make_manifest(
    name="test", task="asr", backend="sherpa-onnx", model_type="transducer", **kw
):
    defaults = dict(
        name=name,
        task=task,
        backend=backend,
        model_type=model_type,
        model_url="http://example.com",
        sample_rate=16000,
        language="en",
        description="test",
        size_mb=10.0,
    )
    defaults.update(kw)
    return ModelManifest(**defaults)


# === suggest_models tests ===
def test_suggest_models_exact():
    register(_make_manifest("zipformer-v2"))
    from revospeech.registry.status import suggest_models

    result = suggest_models("zipformer-v2", "asr")
    assert "zipformer-v2" in result


def test_suggest_models_typo():
    register(_make_manifest("zipformer-v2"))
    from revospeech.registry.status import suggest_models

    result = suggest_models("zipformer", "asr")
    assert "zipformer-v2" in result


def test_suggest_models_no_match():
    register(_make_manifest("zipformer-v2"))
    from revospeech.registry.status import suggest_models

    result = suggest_models("xyzabc", "asr")
    assert result == []


# === ASR factory fuzzy error ===
def test_asr_factory_fuzzy_error():
    register(_make_manifest("zipformer-v2"))
    from revospeech.asr import ASR

    with pytest.raises(KeyError, match="Did you mean"):
        ASR("zipformer")


def test_asr_factory_no_ready_models():
    from revospeech.asr import ASR

    with pytest.raises(Exception, match="No ASR model"):
        ASR()


# === TTS factory fuzzy error ===
def test_tts_factory_fuzzy_error():
    register(
        _make_manifest(
            "revovoice", task="tts", backend="revovoice", model_type="diffusion"
        )
    )
    from revospeech.tts import TTS

    with pytest.raises(KeyError, match="Did you mean"):
        TTS("revovoice2")


# === Top-level __dir__ ===
def test_dir_includes_exports():
    import revospeech

    names = dir(revospeech)
    assert "ASR" in names
    assert "TTS" in names
    assert "list_models" in names
    assert "search_models" in names


# === search_models (top-level) ===
def test_search_models_matches_name():
    register(_make_manifest("zipformer-v2"))
    import revospeech

    results = revospeech.search_models("zipformer")
    assert any(getattr(m, "name", None) == "zipformer-v2" for m in results)


def test_search_models_no_match():
    register(_make_manifest("zipformer-v2"))
    import revospeech

    results = revospeech.search_models("zzzzznomatch")
    assert results == []


# === Transcript.save() ===
def test_transcript_save_txt(tmp_path):
    from revospeech.asr.result import Transcript

    t = Transcript(text="Hello world", segments=[], language="en")
    path = tmp_path / "out.txt"
    t.save(path)
    assert path.read_text() == "Hello world"


def test_transcript_save_json(tmp_path):
    from revospeech.asr.result import Segment, Transcript

    t = Transcript(
        text="Hello",
        segments=[Segment(start=0.0, end=1.0, text="Hello", confidence=0.9)],
        language="en",
    )
    path = tmp_path / "out.json"
    t.save(path)
    data = json.loads(path.read_text())
    assert data["text"] == "Hello"
    assert data["language"] == "en"
    assert len(data["segments"]) == 1


def test_transcript_save_srt(tmp_path):
    from revospeech.asr.result import Segment, Transcript

    t = Transcript(
        text="Hello world",
        segments=[Segment(start=0.0, end=2.0, text="Hello world", confidence=0.9)],
        language="en",
    )
    path = tmp_path / "out.srt"
    t.save(path)
    content = path.read_text()
    assert "1" in content
    assert "00:00:00,000 --> 00:00:02,000" in content
    assert "Hello world" in content


def test_transcript_save_vtt(tmp_path):
    from revospeech.asr.result import Segment, Transcript

    t = Transcript(
        text="Hello",
        segments=[Segment(start=0.0, end=1.0, text="Hello", confidence=0.9)],
        language="en",
    )
    path = tmp_path / "out.vtt"
    t.save(path)
    content = path.read_text()
    assert content.startswith("WEBVTT")


def test_transcript_save_unsupported(tmp_path):
    from revospeech.asr.result import Transcript

    t = Transcript(text="Hello", segments=[], language="en")
    with pytest.raises(ValueError, match="Unsupported transcript format"):
        t.save(tmp_path / "out.unsupported")


def test_transcript_duration():
    from revospeech.asr.result import Segment, Transcript

    t = Transcript(
        text="Hello",
        segments=[
            Segment(start=1.0, end=3.0, text="a", confidence=0.9),
            Segment(start=2.0, end=5.0, text="b", confidence=0.9),
        ],
        language="en",
    )
    assert t.duration == 4.0  # 5.0 - 1.0


def test_transcript_duration_empty():
    from revospeech.asr.result import Transcript

    t = Transcript(text="", segments=[], language="en")
    assert t.duration == 0.0


def test_transcript_repr():
    from revospeech.asr.result import Segment, Transcript

    t = Transcript(
        text="Hello world",
        segments=[Segment(0, 1, "Hello world", 0.9)],
        language="en",
    )
    r = repr(t)
    assert "Transcript" in r
    assert "Hello world" in r or "Hello worl" in r


# === Audio.__repr__ ===
def test_audio_repr():
    from revospeech.tts.result import Audio

    a = Audio(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000)
    r = repr(a)
    assert "Audio" in r
    assert "16000" in r
    assert "1.0s" in r


# === TTS synthesize_batch ===
def test_tts_synthesize_batch():
    from revospeech.tts.base import BaseTTS
    from revospeech.tts.result import Audio

    class FakeTTS(BaseTTS):
        def synthesize(self, text, output_path=None, **kw):
            return Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=16000)

    register(_make_manifest("test-tts", task="tts", backend="vits", model_type="vits"))
    engine = FakeTTS("test-tts")
    report = engine.synthesize_batch(["hello", "world"])
    assert report.total == 2
    assert report.succeeded == 2
    assert report.failed == 0


def test_tts_synthesize_batch_with_output(tmp_path):
    from revospeech.tts.base import BaseTTS
    from revospeech.tts.result import Audio

    class FakeTTS(BaseTTS):
        def synthesize(self, text, output_path=None, **kw):
            audio = Audio(
                samples=np.zeros(100, dtype=np.float32), sample_rate=16000
            )
            if output_path is not None:
                audio.save(str(output_path))
            return audio

    register(_make_manifest("test-tts", task="tts", backend="vits", model_type="vits"))
    engine = FakeTTS("test-tts")
    report = engine.synthesize_batch(["a", "b"], output_dir=str(tmp_path))
    assert (tmp_path / "audio_0.wav").exists()
    assert (tmp_path / "audio_1.wav").exists()
    assert report.succeeded == 2


def test_tts_synthesize_batch_with_error_continue():
    from revospeech.tts.base import BaseTTS
    from revospeech.tts.result import Audio

    class FakeTTS(BaseTTS):
        def synthesize(self, text, output_path=None, **kw):
            if text == "fail":
                raise RuntimeError("boom")
            return Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=16000)

    register(_make_manifest("test-tts", task="tts", backend="vits", model_type="vits"))
    engine = FakeTTS("test-tts")
    report = engine.synthesize_batch(["ok", "fail"])
    assert report.total == 2
    assert report.succeeded == 1
    assert report.failed == 1


def test_tts_synthesize_batch_with_error_raise():
    from revospeech.tts.base import BaseTTS

    class FakeTTS(BaseTTS):
        def synthesize(self, text, output_path=None, **kw):
            raise RuntimeError("always fail")

    register(_make_manifest("test-tts", task="tts", backend="vits", model_type="vits"))
    engine = FakeTTS("test-tts")
    with pytest.raises(RuntimeError, match="Batch synthesis failed"):
        engine.synthesize_batch(["a"], on_error="raise")
