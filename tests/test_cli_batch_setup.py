"""Tests for CLI batch processing (US-023) and interactive setup (US-022)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from revospeech.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_wav(tmp_path: Path) -> str:
    """Create a small WAV file for transcribe tests."""
    import numpy as np
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = tmp_path / "test.wav"
    sf.write(str(p), samples, sr)
    return str(p)


# ---------------------------------------------------------------------------
# US-022: setup command
# ---------------------------------------------------------------------------


def test_setup_declines_all(runner: CliRunner):
    """Setup wizard should exit cleanly when user declines every prompt."""
    result = runner.invoke(cli, ["setup"], input="both\nn\nn\n")
    assert result.exit_code == 0, result.output
    assert "Welcome" in result.output
    assert "Setup complete" in result.output


def test_setup_asr_only(runner: CliRunner):
    """Setup wizard with task='asr' should show ASR section only."""
    with patch("revospeech.registry.status.list_model_statuses") as mock_list:
        mock_list.return_value = []
        result = runner.invoke(cli, ["setup"], input="asr\nn\nn\n")
    assert result.exit_code == 0, result.output
    assert "ASR" in result.output
    assert "TTS" not in result.output
    # list_model_statuses was called with task='asr' only
    mock_list.assert_called_with(task="asr")


def test_setup_prompts_for_api_key(runner: CliRunner):
    """Setup wizard should call set_api_key when user confirms."""
    with (
        patch("revospeech.config.set_api_key") as mock_set,
        patch("revospeech.registry.status.list_model_statuses", return_value=[]),
    ):
        result = runner.invoke(cli, ["setup"], input="both\nn\ny\nmy-secret-key\n")
    assert result.exit_code == 0, result.output
    mock_set.assert_called_once_with("my-secret-key")
    assert "API key saved" in result.output


# ---------------------------------------------------------------------------
# US-023 Task A: transcribe batch
# ---------------------------------------------------------------------------


@patch("revospeech.asr.ASR")
def test_transcribe_single_file_backward_compat(
    mock_asr_cls, runner: CliRunner, sample_wav: str
):
    """Single audio file should still use the simple transcribe path."""
    from revospeech.asr.result import Segment, Transcript

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = Transcript(
        text="HELLO WORLD",
        segments=[Segment(start=0.0, end=1.0, text="HELLO WORLD", confidence=0.9)],
        language="en",
    )
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav])
    assert result.exit_code == 0, result.output
    assert "HELLO WORLD" in result.output
    # Batch path should not be used for single file
    mock_asr.transcribe_batch.assert_not_called()
    mock_asr.transcribe.assert_called_once_with(sample_wav)


@patch("revospeech.asr.ASR")
def test_transcribe_batch_multiple_files(
    mock_asr_cls, runner: CliRunner, sample_wav: str
):
    """Multiple audio files should trigger batch transcription."""
    from revospeech.asr.result import BatchReport, BatchResult, Segment, Transcript

    transcript = Transcript(
        text="HELLO",
        segments=[Segment(start=0.0, end=0.5, text="HELLO", confidence=0.9)],
        language="en",
    )
    report = BatchReport(
        items=[
            BatchResult(input=sample_wav, result=transcript, duration=0.1),
            BatchResult(input=sample_wav, result=transcript, duration=0.1),
        ],
        total=2,
        succeeded=2,
        failed=0,
        total_duration=0.2,
    )

    mock_asr = MagicMock()
    mock_asr.transcribe_batch.return_value = report
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav, sample_wav])
    assert result.exit_code == 0, result.output
    mock_asr.transcribe_batch.assert_called_once()
    # Batch summary should be printed
    assert "Transcribed 2/2" in result.output
    # Each file's text should appear
    assert result.output.count("HELLO") >= 2


# ---------------------------------------------------------------------------
# US-023 Task B: synthesize --file-list
# ---------------------------------------------------------------------------


@patch("revospeech.tts.TTS")
def test_synthesize_file_list_batch(mock_tts_cls, runner: CliRunner):
    """--file-list should trigger synthesize_batch."""
    import numpy as np

    from revospeech.asr.result import BatchReport, BatchResult
    from revospeech.tts.result import Audio

    audio = Audio(samples=np.zeros(24000, dtype=np.float32), sample_rate=24000)
    report = BatchReport(
        items=[
            BatchResult(input="hello", result=audio, duration=0.1),
            BatchResult(input="world", result=audio, duration=0.1),
        ],
        total=2,
        succeeded=2,
        failed=0,
        total_duration=0.2,
    )

    mock_tts = MagicMock()
    mock_tts.synthesize_batch.return_value = report
    mock_tts_cls.return_value = mock_tts

    with runner.isolated_filesystem():
        Path("lines.txt").write_text("hello\nworld\n\n")
        result = runner.invoke(
            cli,
            [
                "synthesize",
                "-m",
                "test",
                "--file-list",
                "lines.txt",
                "-o",
                "out.wav",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_tts.synthesize_batch.assert_called_once()
    assert "Synthesized 2/2" in result.output


def test_synthesize_file_list_with_text_conflict(runner: CliRunner):
    """--file-list plus --text should raise a UsageError."""
    with runner.isolated_filesystem():
        Path("lines.txt").write_text("hello\n")
        result = runner.invoke(
            cli,
            [
                "synthesize",
                "-m",
                "test",
                "--file-list",
                "lines.txt",
                "--text",
                "hi",
                "-o",
                "out.wav",
            ],
        )
    assert result.exit_code != 0
    assert "cannot be combined" in result.output.lower()


def test_synthesize_file_list_empty(runner: CliRunner):
    """An empty --file-list should raise a UsageError."""
    with runner.isolated_filesystem():
        Path("empty.txt").write_text("\n\n   \n")
        result = runner.invoke(
            cli,
            [
                "synthesize",
                "-m",
                "test",
                "--file-list",
                "empty.txt",
                "-o",
                "out.wav",
            ],
        )
    assert result.exit_code != 0
    assert "empty" in result.output.lower()
