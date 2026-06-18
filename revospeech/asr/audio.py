"""Audio loading utility for ASR."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import IO, Any, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Supported audio input types: filesystem path or in-memory binary stream.
AudioInput = Union[str, Path, BytesIO, IO[bytes]]


def read_waveform(audio: AudioInput, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Read an audio file and return mono float32 samples at target sample rate.

    Accepts either a filesystem path (``str`` / ``pathlib.Path``) or an
    in-memory binary stream (``io.BytesIO`` or any file-like object with a
    ``read`` method). This enables transcription of audio that has not been
    persisted to disk, e.g. bytes received from a network upload.

    Args:
        audio: Path to the audio file (WAV, FLAC, etc.) or a binary file-like
            object (``io.BytesIO``) containing encoded audio.
        target_sr: Target sample rate. Audio will be resampled if different.

    Returns:
        Tuple of (samples as float32 ndarray, sample_rate).

    Raises:
        TypeError: If ``audio`` is neither a path-like nor a binary stream.
    """
    data, sr = _read_samples(audio)

    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)
        logger.debug("Converted multi-channel audio to mono")

    # Simple resampling by linear interpolation if needed
    if sr != target_sr:
        duration = len(data) / sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(data) - 1, target_len)
        data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
        sr = target_sr
        logger.debug("Resampled from %d Hz to %d Hz", sr, target_sr)

    return data, sr


def _read_samples(audio: AudioInput) -> tuple[np.ndarray, int]:
    """Dispatch to ``soundfile`` with a path or a file-like object."""
    if isinstance(audio, (str, Path)):
        return sf.read(str(audio), dtype="float32")
    if isinstance(audio, BytesIO) or _has_read(audio):
        # ``soundfile`` accepts any binary file-like object. Ensure the
        # cursor is at the start so previously-consumed streams are handled.
        try:
            audio.seek(0)
        except (OSError, ValueError):
            pass
        return sf.read(audio, dtype="float32")
    raise TypeError(
        f"Unsupported audio input type: {type(audio).__name__}. "
        "Expected str, pathlib.Path, or a binary file-like object (e.g. io.BytesIO)."
    )


def _has_read(obj: Any) -> bool:
    """Return True if ``obj`` looks like a binary file-like object."""
    return hasattr(obj, "read") and callable(getattr(obj, "read"))
