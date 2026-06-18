"""Tests for BaseASR and BaseTTS abstract base classes.

US-026: Verifies abstract-method enforcement, default streaming behavior,
and the _split_text helper used by synthesize_long.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from revospeech.asr.base import BaseASR
from revospeech.tts.base import BaseTTS, _split_long_chunk, _split_text
from revospeech.tts.result import Audio


# ---------------------------------------------------------------------------
# Abstract instantiation enforcement
# ---------------------------------------------------------------------------
def test_base_asr_cannot_instantiate():
    """BaseASR is abstract and must refuse direct instantiation."""
    with pytest.raises(TypeError, match="abstract"):
        BaseASR("test-model")


def test_base_tts_cannot_instantiate():
    """BaseTTS is abstract and must refuse direct instantiation."""
    with pytest.raises(TypeError, match="abstract"):
        BaseTTS("test-model")


# ---------------------------------------------------------------------------
# Minimal concrete subclasses for behavior tests
# ---------------------------------------------------------------------------
class _ConcreteASR(BaseASR):
    """Minimal concrete ASR engine."""

    def transcribe(self, audio_path):  # type: ignore[no-untyped-def]
        from revospeech.asr.result import Transcript

        return Transcript(text="hello", language="en", duration=1.0)


class _ConcreteTTS(BaseTTS):
    """Minimal concrete TTS engine."""

    def synthesize(self, text, output_path=None, **kwargs):  # type: ignore[no-untyped-def]
        return Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=16000)


def test_concrete_asr_instantiates():
    """A subclass implementing transcribe() should instantiate fine."""
    engine = _ConcreteASR("my-model")
    assert engine.model_name == "my-model"
    assert engine.device == "auto"


def test_concrete_tts_instantiates():
    """A subclass implementing synthesize() should instantiate fine."""
    engine = _ConcreteTTS("my-model")
    assert engine.model_name == "my-model"
    assert engine.device == "auto"


def test_base_asr_init_custom_device():
    """Custom device should be stored on the engine."""
    engine = _ConcreteASR("m", device="cpu")
    assert engine.device == "cpu"


def test_base_tts_init_custom_device():
    """Custom device should be stored on the engine."""
    engine = _ConcreteTTS("m", device="cuda:0")
    assert engine.device == "cuda:0"


# ---------------------------------------------------------------------------
# Default streaming behavior raises NotImplementedError
# ---------------------------------------------------------------------------
def test_base_asr_stream_transcribe_default_raises():
    """BaseASR.stream_transcribe default raises NotImplementedError."""
    engine = _ConcreteASR("m")
    with pytest.raises(NotImplementedError, match="streaming"):
        engine.stream_transcribe("dummy.wav")


def test_base_tts_synthesize_streaming_default_raises():
    """BaseTTS.synthesize_streaming default raises NotImplementedError."""
    engine = _ConcreteTTS("m")
    with pytest.raises(NotImplementedError, match="streaming"):
        engine.synthesize_streaming("test text")


# ---------------------------------------------------------------------------
# _split_text behavior
# ---------------------------------------------------------------------------
def test_split_text_short_no_split():
    """A short text should return a single chunk unchanged."""
    text = "Hello world."
    assert _split_text(text) == [text]


def test_split_text_single_sentence_within_limit():
    """Single sentence under max_chars stays as one chunk."""
    assert _split_text("Hello.", max_chars=500) == ["Hello."]


def test_split_text_multiple_sentences_pack_into_chunks():
    """Multiple short sentences pack together until max_chars is reached."""
    text = "One. Two. Three. Four. Five."
    chunks = _split_text(text, max_chars=12)
    # Each chunk must be <= max_chars
    assert all(len(c) <= 12 for c in chunks)
    # All sentences should be present in order
    joined = " ".join(chunks)
    for word in ("One", "Two", "Three", "Four", "Five"):
        assert word in joined


def test_split_text_strips_whitespace():
    """Leading/trailing whitespace is stripped before splitting."""
    assert _split_text("  Hello.  ") == ["Hello."]


def test_split_text_preserves_sentence_order():
    """Sentence order is preserved across chunks."""
    text = ". ".join(f"Sentence {i}" for i in range(50))
    chunks = _split_text(text, max_chars=100)
    full = " ".join(chunks)
    # Verify order: "Sentence 0" before "Sentence 1" etc.
    pos = []
    for i in range(50):
        idx = full.find(f"Sentence {i}")
        assert idx >= 0, f"Sentence {i} missing from output"
        pos.append(idx)
    assert pos == sorted(pos), "Sentence order was not preserved"


# ---------------------------------------------------------------------------
# _split_long_chunk — comma/word fallback for oversized sentences
# ---------------------------------------------------------------------------
def test_split_long_chunk_at_commas():
    """A long chunk with commas should split at comma boundaries."""
    chunk = "first part, second part, third part, fourth part"
    parts = _split_long_chunk(chunk, max_chars=20)
    assert len(parts) >= 2
    assert all(len(p) <= 20 for p in parts)


def test_split_long_chunk_at_word_boundaries():
    """A long chunk without commas should split at word boundaries."""
    chunk = "word " * 30  # ~150 chars, no commas
    parts = _split_long_chunk(chunk.strip(), max_chars=20)
    assert len(parts) >= 2
    assert all(len(p) <= 20 for p in parts)


def test_split_long_chunk_single_word_too_long():
    """If a single word exceeds max_chars, it is returned as-is."""
    chunk = "a" * 100
    parts = _split_long_chunk(chunk, max_chars=20)
    # No word boundaries to split on — returns the single oversized chunk.
    assert parts == [chunk]


# ---------------------------------------------------------------------------
# synthesize_long integration with base class
# ---------------------------------------------------------------------------
def test_synthesize_long_calls_synthesize_per_chunk():
    """synthesize_long should call synthesize() once per chunk."""
    calls: list[str] = []
    engine = _ConcreteTTS("m")

    original = engine.synthesize

    def spy(text, output_path=None, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(text)
        return original(text, output_path, **kwargs)

    engine.synthesize = spy  # type: ignore[assignment]

    text = ". ".join(f"Sentence {i}" for i in range(30))
    engine.synthesize_long(text, max_chars=50)

    assert len(calls) > 1, "Expected multiple chunks -> multiple synthesize calls"


def test_synthesize_long_concatenates_audio():
    """synthesize_long should return concatenated Audio from all chunks."""
    engine = _ConcreteTTS("m")
    text = ". ".join(f"Sentence {i}" for i in range(10))
    result = engine.synthesize_long(text, max_chars=50)
    assert isinstance(result, Audio)
    # Each chunk produces 100 samples; with silence between, total > 100
    assert len(result.samples) > 100


def test_synthesize_long_saves_output(tmp_path: Path):
    """synthesize_long should write to output_path when given."""
    engine = _ConcreteTTS("m")
    out = tmp_path / "long_output.wav"
    engine.synthesize_long("Hello. World.", output_path=str(out))
    assert out.exists()
    # Verify it's a valid audio file
    import soundfile as sf

    data, sr = sf.read(str(out))
    assert sr == 16000
    assert len(data) > 0
