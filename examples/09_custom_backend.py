"""Subclass BaseASR to implement a custom backend.

RevoSpeech's factory dispatches by manifest.backend name. To add a new
backend without modifying the library: subclass BaseASR, register a
manifest with backend: "my-custom", and use ASR('my-model') once the
factory is extended (see AGENTS.md).

This example shows the subclass pattern directly.

Usage:
    python examples/09_custom_backend.py
"""

import numpy as np

from revospeech.asr.base import BaseASR
from revospeech.asr.result import Segment, Transcript


class DummyASR(BaseASR):
    """A no-op ASR that always returns a fixed transcript.

    Useful for testing pipelines without real model weights.
    """

    def transcribe(self, audio_path, language=None, word_timestamps=False):
        samples, sample_rate = self._read_audio(audio_path)
        duration = len(samples) / sample_rate
        return Transcript(
            text="(dummy transcript)",
            segments=[Segment(start=0.0, end=duration, text="(dummy)", confidence=1.0)],
            language=language or "en",
        )

    def _read_audio(self, path):
        # In a real backend this would call your model's preprocessing
        try:
            from revospeech.asr.audio import read_waveform

            return read_waveform(path)
        except Exception:
            # Fallback: assume 16kHz mono
            return np.zeros(16000, dtype=np.float32), 16000


# Use it directly
asr = DummyASR("dummy-model")
result = asr.transcribe("some_audio.wav")
print(f"Text: {result.text}")
print(f"Segments: {len(result.segments)}")
