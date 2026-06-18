"""Tests for CLI extras: config commands, --format flag, VTT output, and color."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from revospeech.cli.main import (
    _format_vtt_time,
    _status_text,
    _use_color,
    cli,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# US-013: config set-api-key / show-api-key
# ---------------------------------------------------------------------------


@patch("revospeech.config.get_api_key", return_value=None)
def test_config_show_api_key_not_set(mock_get: MagicMock, runner: CliRunner) -> None:
    """show-api-key reports 'not set' when no key is configured."""
    result = runner.invoke(cli, ["config", "show-api-key"])
    assert result.exit_code == 0
    assert "API key: not set" in result.output


@patch("revospeech.config.get_api_key", return_value="abcd1234efgh5678")
def test_config_show_api_key_masked(mock_get: MagicMock, runner: CliRunner) -> None:
    """show-api-key masks the key, showing first 4 and last 4 chars only."""
    result = runner.invoke(cli, ["config", "show-api-key"])
    assert result.exit_code == 0
    assert "abcd...5678" in result.output
    assert "(set)" in result.output
    # Never leak the middle of the key.
    assert "1234efgh" not in result.output


@patch("revospeech.config.get_api_key", return_value=None)
@patch("revospeech.config.set_api_key")
def test_config_set_api_key_new(
    mock_set: MagicMock, mock_get: MagicMock, runner: CliRunner
) -> None:
    """set-api-key saves a new key when none is configured."""
    # click.prompt with confirmation_prompt=True asks twice.
    result = runner.invoke(
        cli,
        ["config", "set-api-key"],
        input="secret-key\nsecret-key\n",
    )
    assert result.exit_code == 0, result.output
    assert "API key saved" in result.output
    mock_set.assert_called_once_with("secret-key")


@patch("revospeech.config.get_api_key", return_value="existing-key")
@patch("revospeech.config.set_api_key")
def test_config_set_api_key_overwrite_confirmed(
    mock_set: MagicMock, mock_get: MagicMock, runner: CliRunner
) -> None:
    """set-api-key overwrites an existing key when user confirms."""
    # First line: overwrite confirmation; next two: key + confirmation.
    result = runner.invoke(
        cli,
        ["config", "set-api-key"],
        input="y\nnew-key\nnew-key\n",
    )
    assert result.exit_code == 0, result.output
    assert "API key saved" in result.output
    mock_set.assert_called_once_with("new-key")


@patch("revospeech.config.get_api_key", return_value="existing-key")
@patch("revospeech.config.set_api_key")
def test_config_set_api_key_overwrite_aborted(
    mock_set: MagicMock, mock_get: MagicMock, runner: CliRunner
) -> None:
    """set-api-key aborts when user declines overwrite."""
    result = runner.invoke(
        cli,
        ["config", "set-api-key"],
        input="n\n",
    )
    assert result.exit_code == 0, result.output
    assert "Aborted." in result.output
    mock_set.assert_not_called()


# ---------------------------------------------------------------------------
# US-015: _status_text / _use_color
# ---------------------------------------------------------------------------


def test_use_color_disabled_when_no_color_set(monkeypatch) -> None:
    """NO_COLOR env var forces color off."""
    monkeypatch.setenv("NO_COLOR", "1")
    assert _use_color() is False


def test_use_color_disabled_when_not_tty(monkeypatch) -> None:
    """Non-TTY stdout disables color."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    # CliRunner output is not a TTY.
    import io

    monkeypatch.setattr("sys.stdout", io.StringIO())
    assert _use_color() is False


def test_status_text_plain_when_color_disabled(monkeypatch) -> None:
    """When color is disabled, output is plain 'icon status' text."""
    monkeypatch.setenv("NO_COLOR", "1")
    assert _status_text("ready") == "✓ ready"
    assert _status_text("needs-download") == "↓ needs-download"
    assert _status_text("needs-api-key") == "✗ needs-api-key"
    assert _status_text("unknown") == "? unknown"


def test_status_text_no_ansi_codes_when_color_disabled(monkeypatch) -> None:
    """No ANSI escape codes leak into output when color is off."""
    monkeypatch.setenv("NO_COLOR", "1")
    for status in ("ready", "needs-download", "needs-api-key"):
        text = _status_text(status)
        assert "\x1b[" not in text


# ---------------------------------------------------------------------------
# US-016: --format text|json|srt|vtt and deprecated --json / --srt
# ---------------------------------------------------------------------------


def _make_transcript():
    """Build a fake Transcript for ASR tests."""
    from revospeech.asr.result import Segment, Transcript

    return Transcript(
        text="HELLO WORLD",
        segments=[
            Segment(start=0.0, end=0.5, text="HELLO", confidence=0.9),
            Segment(start=0.5, end=1.0, text="WORLD", confidence=0.8),
        ],
        language="en",
    )


@patch("revospeech.asr.ASR")
def test_transcribe_format_text(
    mock_asr_cls: MagicMock, runner: CliRunner, sample_wav: str
) -> None:
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = _make_transcript()
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(
        cli, ["transcribe", "-m", "test", "--format", "text", sample_wav]
    )
    assert result.exit_code == 0, result.output
    assert "HELLO WORLD" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_format_json(
    mock_asr_cls: MagicMock, runner: CliRunner, sample_wav: str
) -> None:
    import json as _json

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = _make_transcript()
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(
        cli, ["transcribe", "-m", "test", "--format", "json", sample_wav]
    )
    assert result.exit_code == 0, result.output
    data = _json.loads(result.output)
    assert data["text"] == "HELLO WORLD"
    assert len(data["segments"]) == 2


@patch("revospeech.asr.ASR")
def test_transcribe_format_srt(
    mock_asr_cls: MagicMock, runner: CliRunner, sample_wav: str
) -> None:
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = _make_transcript()
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(
        cli, ["transcribe", "-m", "test", "--format", "srt", sample_wav]
    )
    assert result.exit_code == 0, result.output
    assert "1" in result.output
    assert "-->" in result.output
    assert "HELLO" in result.output
    # SRT uses a comma before milliseconds.
    assert "," in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_format_vtt(
    mock_asr_cls: MagicMock, runner: CliRunner, sample_wav: str
) -> None:
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = _make_transcript()
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(
        cli, ["transcribe", "-m", "test", "--format", "vtt", sample_wav]
    )
    assert result.exit_code == 0, result.output
    assert "WEBVTT" in result.output
    assert "-->" in result.output
    assert "HELLO" in result.output
    # VTT uses a dot before milliseconds.
    assert "00:00:00.000 --> 00:00:00.500" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_deprecated_json_flag(
    mock_asr_cls: MagicMock, runner: CliRunner, sample_wav: str
) -> None:
    """--json still works as a deprecated alias for --format json."""
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = _make_transcript()
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", "--json", sample_wav])
    assert result.exit_code == 0, result.output
    # JSON output is parseable.
    import json as _json

    data = _json.loads(result.output)
    assert data["text"] == "HELLO WORLD"


@patch("revospeech.asr.ASR")
def test_transcribe_deprecated_srt_flag(
    mock_asr_cls: MagicMock, runner: CliRunner, sample_wav: str
) -> None:
    """--srt still works as a deprecated alias for --format srt."""
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = _make_transcript()
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", "--srt", sample_wav])
    assert result.exit_code == 0, result.output
    assert "-->" in result.output
    assert "HELLO" in result.output


def test_transcribe_help_lists_format_option(runner: CliRunner) -> None:
    """--format/-fmt is documented in transcribe --help."""
    result = runner.invoke(cli, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "--format" in result.output
    assert "-fmt" in result.output
    assert "vtt" in result.output
    # Deprecated flags are still listed.
    assert "--json" in result.output
    assert "--srt" in result.output


def test_format_vtt_time_zero() -> None:
    assert _format_vtt_time(0.0) == "00:00:00.000"


def test_format_vtt_time_uses_dot_separator() -> None:
    assert _format_vtt_time(1.5) == "00:00:01.500"
    assert _format_vtt_time(3725.5) == "01:02:05.500"


def test_format_vtt_time_no_comma() -> None:
    """VTT timestamps must never contain a comma (SRT uses comma, VTT dot)."""
    assert "," not in _format_vtt_time(123.456)


# ---------------------------------------------------------------------------
# Color in models / models-info / search (plain-text path)
# ---------------------------------------------------------------------------


def test_models_command_plain_text_no_ansi(runner: CliRunner) -> None:
    """models command output has no ANSI codes when not a TTY."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register

    _models.clear()
    register(
        ModelManifest(
            name="color-test",
            task="asr",
            backend="sherpa-onnx",
            model_type="transducer",
            model_url="",
            sample_rate=16000,
            language="en",
            description="",
        )
    )
    os.environ["NO_COLOR"] = "1"
    try:
        result = runner.invoke(cli, ["models"])
    finally:
        os.environ.pop("NO_COLOR", None)
    assert result.exit_code == 0
    assert "\x1b[" not in result.output
    assert "color-test" in result.output
