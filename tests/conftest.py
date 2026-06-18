"""Shared test fixtures and pytest configuration for revospeech tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "roundtrip: TTS->ASR round-trip tests (slow, require models)"
    )


def pytest_addoption(parser):
    """Add custom CLI options."""
    parser.addoption(
        "--run-roundtrip",
        action="store_true",
        default=False,
        help="Run round-trip tests (requires actual model weights)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip roundtrip tests unless explicitly requested."""
    if config.getoption("--run-roundtrip") or os.environ.get("REVOS_RUN_ROUNDTRIP"):
        return
    skip_roundtrip = pytest.mark.skip(
        reason="needs --run-roundtrip or REVOS_RUN_ROUNDTRIP=1"
    )
    for item in items:
        if "roundtrip" in item.keywords:
            item.add_marker(skip_roundtrip)


@pytest.fixture
def sample_wav(tmp_path: Path) -> str:
    """Create a small test WAV file (1-second 440Hz sine wave)."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), samples, sr)
    return str(wav_path)
