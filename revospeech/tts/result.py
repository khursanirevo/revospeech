"""TTS result type: Audio.

Example:
    >>> from revospeech.tts import TTS
    >>> audio = TTS().synthesize('Hello, world!')
    >>>
    >>> # Properties
    >>> audio.samples         # numpy array of float32
    >>> audio.sample_rate     # 22050 or similar
    >>> audio.duration        # len(samples) / sample_rate
    >>>
    >>> # Save to file
    >>> audio.save('out.wav')
    >>>
    >>> # Concatenate multiple clips
    >>> from revospeech.tts.result import Audio
    >>> combined = Audio.concatenate([audio1, audio2], silence_duration=0.2)
    >>>
    >>> # Playback (requires sounddevice)
    >>> audio.play()    # blocking
    >>> audio.play(block=False)  # non-blocking
    >>>
    >>> # Repr
    >>> repr(audio)  # Audio(duration=2.34s, sample_rate=22050)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class Audio:
    """Synthesized audio result from TTS."""

    samples: np.ndarray
    sample_rate: int

    def save(self, path: str | Path) -> None:
        """Save audio to a file.

        Args:
            path: Output file path (e.g., "output.wav").
        """
        sf.write(str(path), self.samples, self.sample_rate)

    @property
    def duration(self) -> float:
        """Audio duration in seconds."""
        return len(self.samples) / self.sample_rate

    @staticmethod
    def concatenate(
        segments: list[Audio],
        silence_duration: float = 0.1,
    ) -> Audio:
        """Concatenate multiple Audio segments into one.

        Args:
            segments: Audio segments to join. All must have the same
                sample rate.
            silence_duration: Seconds of silence between segments.

        Returns:
            A single Audio with all segments joined.

        Raises:
            ValueError: If segments is empty or sample rates differ.
        """
        if not segments:
            raise ValueError("Cannot concatenate empty segment list")

        rates = {s.sample_rate for s in segments}
        if len(rates) > 1:
            raise ValueError(
                f"All segments must have the same sample rate, got: {rates}"
            )

        sr = segments[0].sample_rate
        silence = np.zeros(int(sr * silence_duration), dtype=np.float32)

        parts = [segments[0].samples]
        for seg in segments[1:]:
            parts.append(silence)
            parts.append(seg.samples)

        return Audio(samples=np.concatenate(parts), sample_rate=sr)

    def play(self, block: bool = True) -> None:
        """Play audio through the system's audio output.

        Requires the optional ``sounddevice`` package.

        Args:
            block: If True (default), wait until playback finishes.
                If False, return immediately (playback continues in background).

        Raises:
            ImportError: If sounddevice is not installed.
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "Audio playback requires 'sounddevice'. Install it:\n"
                "  pip install sounddevice"
            )
        sd.play(self.samples, self.sample_rate, blocking=block)

    def __repr__(self) -> str:
        return (
            f"Audio(duration={self.duration:.1f}s, "
            f"sample_rate={self.sample_rate}Hz, samples={len(self.samples)})"
        )
