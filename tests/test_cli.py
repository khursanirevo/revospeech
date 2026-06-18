"""Tests for CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from revospeech.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner: CliRunner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "transcribe" in result.output
    assert "synthesize" in result.output


def test_cli_version(runner: CliRunner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.1" in result.output


def test_transcribe_help(runner: CliRunner):
    result = runner.invoke(cli, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--json" in result.output
    assert "--srt" in result.output


def test_synthesize_help(runner: CliRunner):
    result = runner.invoke(cli, ["synthesize", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--text" in result.output
    assert "--output" in result.output
    assert "--ref-audio" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_text_output(mock_asr_cls, runner: CliRunner, sample_wav):
    from revospeech.asr.result import Segment, Transcript

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = Transcript(
        text="HELLO WORLD",
        segments=[
            Segment(start=0.0, end=0.5, text="HELLO", confidence=0.9),
            Segment(start=0.5, end=1.0, text="WORLD", confidence=0.8),
        ],
        language="en",
    )
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav])
    assert result.exit_code == 0
    assert "HELLO WORLD" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_json_output(mock_asr_cls, runner: CliRunner, sample_wav):
    import json

    from revospeech.asr.result import Segment, Transcript

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = Transcript(
        text="HELLO",
        segments=[Segment(start=0.0, end=0.5, text="HELLO", confidence=0.9)],
        language="en",
    )
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", "--json", sample_wav])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["text"] == "HELLO"
    assert len(data["segments"]) == 1


@patch("revospeech.asr.ASR")
def test_transcribe_srt_output(mock_asr_cls, runner: CliRunner, sample_wav):
    from revospeech.asr.result import Segment, Transcript

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = Transcript(
        text="HELLO",
        segments=[Segment(start=0.0, end=1.5, text="HELLO", confidence=0.9)],
        language="en",
    )
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", "--srt", sample_wav])
    assert result.exit_code == 0
    assert "-->" in result.output
    assert "HELLO" in result.output


def test_synthesize_requires_text_or_file(runner: CliRunner):
    result = runner.invoke(cli, ["synthesize", "-m", "test", "-o", "out.wav"])
    assert result.exit_code != 0


@patch("revospeech.tts.TTS")
def test_synthesize_text_output(mock_tts_cls, runner: CliRunner):
    """Test synthesize command with text input and output."""
    import numpy as np

    from revospeech.tts.result import Audio

    mock_tts = MagicMock()
    samples = np.random.randn(24000).astype(np.float32) * 0.1
    mock_tts.synthesize.return_value = Audio(samples=samples, sample_rate=24000)
    mock_tts_cls.return_value = mock_tts

    with runner.isolated_filesystem():
        result = runner.invoke(
            cli, ["synthesize", "-m", "test", "-t", "Hello", "-o", "out.wav"]
        )
    assert result.exit_code == 0
    assert "Saved" in result.output


@patch("revospeech.tts.TTS")
def test_synthesize_from_file(mock_tts_cls, runner: CliRunner):
    """Test synthesize command reading from text file."""
    import numpy as np

    from revospeech.tts.result import Audio

    mock_tts = MagicMock()
    samples = np.random.randn(24000).astype(np.float32) * 0.1
    mock_tts.synthesize.return_value = Audio(samples=samples, sample_rate=24000)
    mock_tts_cls.return_value = mock_tts

    with runner.isolated_filesystem():
        with open("input.txt", "w") as f:
            f.write("Hello from file")

        result = runner.invoke(
            cli, ["synthesize", "-m", "test", "-f", "input.txt", "-o", "out.wav"]
        )
    assert result.exit_code == 0


def test_models_command(runner: CliRunner):
    """Test revos models lists registered models."""
    from revospeech.registry.registry import _models

    _models.clear()
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import register

    register(
        ModelManifest(
            name="test-model",
            task="asr",
            backend="sherpa-onnx",
            model_type="transducer",
            model_url="",
            sample_rate=16000,
            language="en",
            description="Test ASR model",
        )
    )

    result = runner.invoke(cli, ["models"])
    assert result.exit_code == 0
    assert "test-model" in result.output
    assert "asr" in result.output


def test_models_no_models(runner: CliRunner):
    """Test revos models when no models registered."""
    from revospeech.registry.registry import _models

    _models.clear()

    result = runner.invoke(cli, ["models"])
    assert result.exit_code == 0
    assert "No models found" in result.output


def test_info_command(runner: CliRunner):
    """Test revos info shows environment info."""
    result = runner.invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "Python" in result.output
    assert "Cache dir" in result.output


def test_search_command_with_results(runner: CliRunner):
    """revos search <query> shows matching models."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register

    _models.clear()
    register(
        ModelManifest(
            name="zipformer-v2",
            task="asr",
            backend="sherpa-onnx",
            model_type="transducer",
            model_url="",
            sample_rate=16000,
            language="en",
            description="Test ASR model",
            files={},
        )
    )
    try:
        result = runner.invoke(cli, ["search", "zip"])
        assert result.exit_code == 0
        assert "zipformer-v2" in result.output
    finally:
        _models.clear()


def test_search_command_no_results(runner: CliRunner):
    """revos search with no matches shows suggestion."""
    from revospeech.registry.registry import _models

    _models.clear()
    try:
        result = runner.invoke(cli, ["search", "nonexistent"])
        assert result.exit_code == 0
        assert "No models matching" in result.output
    finally:
        _models.clear()


def test_models_command_json_output(runner: CliRunner):
    """revos models --json outputs structured JSON."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register

    _models.clear()
    register(
        ModelManifest(
            name="test-model",
            task="asr",
            backend="sherpa-onnx",
            model_type="transducer",
            model_url="",
            sample_rate=16000,
            language="en",
            description="Test",
            files={},
        )
    )
    try:
        import json

        result = runner.invoke(cli, ["models", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert data[0]["name"] == "test-model"
        assert data[0]["task"] == "asr"
    finally:
        _models.clear()


def test_models_command_filter_by_task(runner: CliRunner):
    """revos models --task asr filters to only ASR models."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register

    _models.clear()
    register(
        ModelManifest(
            name="asr-model",
            task="asr",
            backend="sherpa-onnx",
            model_type="transducer",
            model_url="",
            sample_rate=16000,
            language="en",
            description="ASR",
            files={},
        )
    )
    register(
        ModelManifest(
            name="tts-model",
            task="tts",
            backend="vits",
            model_type="vits",
            model_url="",
            sample_rate=22050,
            language="en",
            description="TTS",
            files={},
        )
    )
    try:
        result = runner.invoke(cli, ["models", "--task", "asr"])
        assert result.exit_code == 0
        assert "asr-model" in result.output
        assert "tts-model" not in result.output
    finally:
        _models.clear()


def test_models_command_empty_with_suggestion(runner: CliRunner):
    """revos models shows suggestion when no models match."""
    from revospeech.registry.registry import _models

    _models.clear()
    try:
        result = runner.invoke(cli, ["models", "--task", "nonexistent"])
        assert result.exit_code == 0
        assert "No models found" in result.output
        assert "catalog list" in result.output
    finally:
        _models.clear()


def test_models_info_command_success(runner: CliRunner):
    """revos models-info <name> shows detailed info."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register

    _models.clear()
    register(
        ModelManifest(
            name="zipformer-v2",
            task="asr",
            backend="sherpa-onnx",
            model_type="transducer",
            model_url="",
            sample_rate=16000,
            language="en",
            description="Test",
            size_mb=80.0,
            capabilities=["word-timestamps"],
            languages=["en"],
            files={},
        )
    )
    try:
        result = runner.invoke(cli, ["models-info", "zipformer-v2"])
        assert result.exit_code == 0
        assert "zipformer-v2" in result.output
        assert "asr" in result.output
        assert "80 MB" in result.output
        assert "word-timestamps" in result.output
    finally:
        _models.clear()


def test_models_info_command_not_found(runner: CliRunner):
    """revos models-info <unknown> exits with error + suggestion."""
    from revospeech.registry.registry import _models

    _models.clear()
    try:
        result = runner.invoke(cli, ["models-info", "nonexistent-model"])
        assert result.exit_code == 1
        assert "Error" in result.output
        assert "revospeech models" in result.output
    finally:
        _models.clear()


@patch("revospeech.asr.ASR")
def test_transcribe_revos_engine_error(mock_asr_cls, runner: CliRunner, sample_wav):
    """transcribe surfaces RevosEngineError with formatted message."""
    from revospeech.exceptions import RevosEngineError

    mock_asr = MagicMock()
    mock_asr.transcribe.side_effect = RevosEngineError(
        "Model failed", suggestion="Check the model files"
    )
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav])
    assert result.exit_code == 1
    assert "Engine error" in result.output
    assert "Check the model files" in result.output


@patch("revospeech.tts.TTS")
def test_synthesize_revos_config_error(mock_tts_cls, runner: CliRunner):
    """synthesize surfaces RevosConfigError with formatted message."""
    from revospeech.exceptions import RevosConfigError

    mock_tts = MagicMock()
    mock_tts.synthesize.side_effect = RevosConfigError(
        "Missing API key", suggestion="Run 'revospeech config set-api-key'"
    )
    mock_tts_cls.return_value = mock_tts

    result = runner.invoke(
        cli, ["synthesize", "-m", "test", "-t", "hello", "-o", "out.wav"]
    )
    assert result.exit_code == 1
    assert "Configuration error" in result.output
    assert "set-api-key" in result.output
