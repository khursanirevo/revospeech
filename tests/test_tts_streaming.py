"""Tests for TTS streaming synthesis."""

from __future__ import annotations

import numpy as np
import pytest


def test_synthesize_streaming_yields_per_chunk():
    """Streaming should yield one Audio per sentence chunk."""
    from revospeech.tts.base import BaseTTS, _split_text
    from revospeech.tts.result import Audio

    class FakeTTS(BaseTTS):
        def synthesize(self, text, output_path=None, **kwargs):
            return Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=22050)

        def synthesize_streaming(self, text, **kwargs):
            chunks = _split_text(text, max_chars=20)
            for chunk in chunks:
                yield self.synthesize(chunk, **kwargs)

    engine = FakeTTS("test")
    text = "First sentence. Second sentence. Third sentence."
    # Each sentence is ~14 chars; force per-sentence splitting.
    chunks = list(engine.synthesize_streaming(text))

    assert len(chunks) == 3
    for audio in chunks:
        assert audio.sample_rate == 22050
        assert len(audio.samples) == 100


def test_synthesize_streaming_short_text_single_chunk():
    from revospeech.tts.base import BaseTTS, _split_text
    from revospeech.tts.result import Audio

    class FakeTTS(BaseTTS):
        def synthesize(self, text, output_path=None, **kwargs):
            return Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=22050)

        def synthesize_streaming(self, text, **kwargs):
            for chunk in _split_text(text, max_chars=500):
                yield self.synthesize(chunk, **kwargs)

    engine = FakeTTS("test")
    chunks = list(engine.synthesize_streaming("Short text."))
    assert len(chunks) == 1


def test_synthesize_streaming_empty_text_yields_nothing():
    from revospeech.tts.base import BaseTTS, _split_text
    from revospeech.tts.result import Audio

    class FakeTTS(BaseTTS):
        def synthesize(self, text, output_path=None, **kwargs):
            return Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=22050)

        def synthesize_streaming(self, text, **kwargs):
            for chunk in _split_text(text, max_chars=500):
                yield self.synthesize(chunk, **kwargs)

    engine = FakeTTS("test")
    chunks = list(engine.synthesize_streaming(""))
    assert len(chunks) == 0


def test_base_synthesize_streaming_default_raises():
    """BaseTTS.synthesize_streaming should raise by default."""
    from revospeech.tts.base import BaseTTS
    from revospeech.tts.result import Audio

    class MinimalTTS(BaseTTS):
        def synthesize(self, text, output_path=None, **kwargs):
            return Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=22050)

        # No synthesize_streaming override

    engine = MinimalTTS("test")
    with pytest.raises(NotImplementedError, match="streaming"):
        list(engine.synthesize_streaming("test"))


def test_synthesize_streaming_passes_kwargs():
    """Streaming should forward kwargs to each synthesize call."""
    from revospeech.tts.base import BaseTTS, _split_text
    from revospeech.tts.result import Audio

    class FakeTTS(BaseTTS):
        def __init__(self, name):
            super().__init__(name)
            self.last_kwargs = None

        def synthesize(self, text, output_path=None, **kwargs):
            self.last_kwargs = kwargs
            return Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=22050)

        def synthesize_streaming(self, text, **kwargs):
            for chunk in _split_text(text, max_chars=500):
                yield self.synthesize(chunk, **kwargs)

    engine = FakeTTS("test")
    list(engine.synthesize_streaming("Hello.", speed=1.5, ref_audio="ref.wav"))
    assert engine.last_kwargs == {"speed": 1.5, "ref_audio": "ref.wav"}
