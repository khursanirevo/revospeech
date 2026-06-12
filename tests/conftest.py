"""Shared test fixtures for revos tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sample_wav(tmp_path: Path) -> str:
    """Create a small test WAV file (1-second 440Hz sine wave)."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), samples, sr)
    return str(wav_path)


