"""Base class for util models (post-processors like speech restoration)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from revospeech.tts.result import Audio


class BaseUtil(ABC):
    """Abstract base for util models that transform audio.

    Util models are not main ASR/TTS engines — they post-process audio
    (e.g. denoise, dereverberate, bandwidth extension).
    """

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        if device == "auto":
            from revospeech.device import auto_detect_device

            device = auto_detect_device()
        self.device = device

    @abstractmethod
    def restore(self, audio: Audio) -> Audio:
        """Restore / enhance an Audio input, returning a new Audio."""
        ...

    def restore_file(self, input_path: str, output_path: str | None = None) -> Audio:
        """Convenience: read audio file, restore, optionally save."""
        import soundfile as sf

        samples, sr = sf.read(str(input_path), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        audio = Audio(samples=samples, sample_rate=sr)
        enhanced = self.restore(audio)
        if output_path:
            enhanced.save(output_path)
        return enhanced
