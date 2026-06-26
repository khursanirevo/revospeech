"""Tests for CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
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
    from revospeech import __version__

    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


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


# ---------------------------------------------------------------------------
# CLI group: --verbose / --quiet log-level flags
# ---------------------------------------------------------------------------
def test_cli_verbose_flag_sets_debug_level(runner: CliRunner):
    """--verbose sets revospeech logger to DEBUG."""
    import logging

    logger = logging.getLogger("revospeech")
    original = logger.level
    try:
        runner.invoke(cli, ["-v", "info"])
        assert logger.level == logging.DEBUG
    finally:
        logger.setLevel(original)


def test_cli_quiet_flag_sets_warning_level(runner: CliRunner):
    """--quiet sets revospeech logger to WARNING."""
    import logging

    logger = logging.getLogger("revospeech")
    original = logger.level
    try:
        runner.invoke(cli, ["-q", "info"])
        assert logger.level == logging.WARNING
    finally:
        logger.setLevel(original)


# ---------------------------------------------------------------------------
# Color helpers — _status_text and _installed_status_text
# ---------------------------------------------------------------------------
def test_status_text_no_color_returns_plain():
    """Without color, _status_text returns plain '<icon> <status>'."""
    from revospeech.cli.main import _status_text

    assert _status_text("ready") == "✓ ready"
    assert _status_text("needs-download") == "↓ needs-download"
    assert _status_text("needs-api-key") == "✗ needs-api-key"
    assert _status_text("unknown") == "? unknown"


def test_installed_status_text_no_color():
    """Without color, _installed_status_text returns plain text."""
    from revospeech.cli.main import _installed_status_text

    assert _installed_status_text(True) == "✓ installed"
    assert _installed_status_text(False) == "↓ not installed"


# ---------------------------------------------------------------------------
# Utility helpers — _format_srt_time, _format_vtt_time, _get_version
# ---------------------------------------------------------------------------
def test_format_srt_time_zero():
    from revospeech.cli.main import _format_srt_time

    assert _format_srt_time(0) == "00:00:00,000"


def test_format_srt_time_with_millis():
    from revospeech.cli.main import _format_srt_time

    assert _format_srt_time(3661.5) == "01:01:01,500"


def test_format_vtt_time_zero():
    from revospeech.cli.main import _format_vtt_time

    assert _format_vtt_time(0) == "00:00:00.000"


def test_format_vtt_time_with_millis():
    from revospeech.cli.main import _format_vtt_time

    assert _format_vtt_time(3661.5) == "01:01:01.500"


def test_get_version_returns_known_version():
    """_get_version reads the installed distribution metadata."""
    from revospeech import __version__
    from revospeech.cli.main import _get_version

    assert _get_version() == __version__


def test_get_version_handles_missing_distribution(monkeypatch):
    """_get_version returns 'unknown' when importlib.metadata.version raises."""

    def boom(*a, **kw):
        raise ModuleNotFoundError("not installed")

    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", boom)
    from importlib import reload

    import revospeech.cli.main as cli_mod

    reload(cli_mod)
    assert cli_mod._get_version() == "unknown"


# ---------------------------------------------------------------------------
# models command — --download flow, filters, JSON output
# ---------------------------------------------------------------------------
def test_models_download_success(runner: CliRunner, monkeypatch):
    """revos models --download <name> triggers ensure_model and prints Done."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register

    _models.clear()
    register(
        ModelManifest(
            name="dl-model",
            task="asr",
            backend="sherpa-onnx",
            model_type="transducer",
            model_url="http://example.com/m.tar.bz2",
            sample_rate=16000,
            language="en",
            description="",
            files={},
        )
    )

    captured = []
    monkeypatch.setattr(
        "revospeech.registry.downloader.ensure_model",
        lambda manifest: captured.append(manifest.name),
    )

    try:
        result = runner.invoke(cli, ["models", "--download", "dl-model"])
        assert result.exit_code == 0
        assert "Downloading dl-model" in result.output
        assert "Done" in result.output
        assert captured == ["dl-model"]
    finally:
        _models.clear()


def test_models_download_unknown_name(runner: CliRunner):
    """revos models --download <unknown> exits 1 with suggestion."""
    from revospeech.registry.registry import _models

    _models.clear()
    try:
        result = runner.invoke(cli, ["models", "--download", "ghost"])
        assert result.exit_code == 1
        assert "not found" in result.output
        assert "catalog list" in result.output
    finally:
        _models.clear()


def test_models_filter_by_mode_and_ready(runner: CliRunner, monkeypatch):
    """--mode and --ready pass through to list_model_statuses."""
    from revospeech.registry.status import ModelStatus

    captured = []

    def fake_list(**kw):
        captured.append(kw)
        return [
            ModelStatus(
                name="x",
                task="asr",
                mode="local",
                status="ready",
                installed=True,
                size_mb=10.0,
                capabilities=[],
                languages=["en"],
            )
        ]

    monkeypatch.setattr("revospeech.registry.status.list_model_statuses", fake_list)

    result = runner.invoke(cli, ["models", "--mode", "local", "--ready"])
    assert result.exit_code == 0
    assert any(call.get("mode") == "local" for call in captured)
    assert any(call.get("status") == "ready" for call in captured)


def test_models_filter_by_status_explicit(runner: CliRunner, monkeypatch):
    """--status <value> is forwarded as status kwarg."""
    captured = []

    def fake_list(**kw):
        captured.append(kw)
        return []

    monkeypatch.setattr("revospeech.registry.status.list_model_statuses", fake_list)

    result = runner.invoke(cli, ["models", "--status", "needs-download"])
    assert result.exit_code == 0
    assert captured[-1].get("status") == "needs-download"


# ---------------------------------------------------------------------------
# info command — cache missing + API key present branches
# ---------------------------------------------------------------------------
def test_info_shows_empty_cache_when_absent(runner: CliRunner, monkeypatch):
    """info prints '(empty)' when CACHE_DIR does not exist."""
    monkeypatch.setattr("revospeech.registry.downloader.CACHE_DIR", "/nonexistent/path")
    monkeypatch.setattr("revospeech.config.get_api_key", lambda *a, **kw: None)
    monkeypatch.setattr(
        "revospeech.catalog.get_catalog_repo", lambda: "revolab/revospeech"
    )

    result = runner.invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "(empty)" in result.output
    assert "API key: not set" in result.output


def test_info_shows_masked_key_when_set(runner: CliRunner, monkeypatch):
    """info masks the API key when set."""
    monkeypatch.setattr("revospeech.registry.downloader.CACHE_DIR", "/nonexistent/path")
    monkeypatch.setattr(
        "revospeech.config.get_api_key", lambda *a, **kw: "rv-test12345"
    )
    monkeypatch.setattr(
        "revospeech.catalog.get_catalog_repo", lambda: "revolab/revospeech"
    )

    result = runner.invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "rv-t...2345" in result.output
    assert "(set)" in result.output


def test_info_computes_cache_size(runner: CliRunner, monkeypatch, tmp_path):
    """info computes total size of CACHE_DIR contents."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "model.bin").write_bytes(b"x" * 2048)

    monkeypatch.setattr("revospeech.registry.downloader.CACHE_DIR", str(cache))
    monkeypatch.setattr("revospeech.config.get_api_key", lambda *a, **kw: None)
    monkeypatch.setattr(
        "revospeech.catalog.get_catalog_repo", lambda: "revolab/revospeech"
    )

    result = runner.invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "MB" in result.output


# ---------------------------------------------------------------------------
# config set-api-key / show-api-key
# ---------------------------------------------------------------------------
def test_config_show_api_key_when_set(runner: CliRunner, monkeypatch):
    """config show-api-key masks and labels an existing key."""
    monkeypatch.setattr(
        "revospeech.config.get_api_key", lambda *a, **kw: "rv-test12345"
    )

    result = runner.invoke(cli, ["config", "show-api-key"])
    assert result.exit_code == 0
    assert "rv-t...2345" in result.output
    assert "(set)" in result.output


def test_config_show_api_key_when_unset(runner: CliRunner, monkeypatch):
    """config show-api-key prints 'not set' when no key configured."""
    monkeypatch.setattr("revospeech.config.get_api_key", lambda *a, **kw: None)

    result = runner.invoke(cli, ["config", "show-api-key"])
    assert result.exit_code == 0
    assert "not set" in result.output


def test_config_set_api_key_when_none_set(runner: CliRunner, monkeypatch):
    """config set-api-key prompts and saves when no existing key."""
    monkeypatch.setattr("revospeech.config.get_api_key", lambda *a, **kw: None)
    saved = []
    monkeypatch.setattr("revospeech.config.set_api_key", lambda k: saved.append(k))

    result = runner.invoke(
        cli,
        ["config", "set-api-key"],
        input="rv-newkey12345\nrv-newkey12345\n",
    )
    assert result.exit_code == 0
    assert saved == ["rv-newkey12345"]


def test_config_set_api_key_overwrite_confirmed(runner: CliRunner, monkeypatch):
    """config set-api-key asks to overwrite and proceeds on 'y'."""
    monkeypatch.setattr(
        "revospeech.config.get_api_key", lambda *a, **kw: "rv-oldkey12345"
    )
    saved = []
    monkeypatch.setattr("revospeech.config.set_api_key", lambda k: saved.append(k))

    result = runner.invoke(
        cli,
        ["config", "set-api-key"],
        input="y\nrv-newkey12345\nrv-newkey12345\n",
    )
    assert result.exit_code == 0
    assert saved == ["rv-newkey12345"]


def test_config_set_api_key_overwrite_aborted(runner: CliRunner, monkeypatch):
    """config set-api-key aborts when overwrite answer is not 'y'."""
    monkeypatch.setattr(
        "revospeech.config.get_api_key", lambda *a, **kw: "rv-oldkey12345"
    )
    saved = []
    monkeypatch.setattr("revospeech.config.set_api_key", lambda k: saved.append(k))

    result = runner.invoke(cli, ["config", "set-api-key"], input="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output
    assert saved == []


def test_config_set_api_key_save_error_exit_code(runner: CliRunner, monkeypatch):
    """Save failure surfaces as exit 1 with suggestion."""
    monkeypatch.setattr("revospeech.config.get_api_key", lambda *a, **kw: None)

    def boom(_key):
        raise RuntimeError("disk full")

    monkeypatch.setattr("revospeech.config.set_api_key", boom)

    result = runner.invoke(
        cli,
        ["config", "set-api-key"],
        input="rv-newkey12345\nrv-newkey12345\n",
    )
    assert result.exit_code == 1
    assert "Error" in result.output


# ---------------------------------------------------------------------------
# catalog list / pull / search / recommend
# ---------------------------------------------------------------------------
def _fake_catalog_manifest(**kw):
    from revospeech.registry.manifest import ModelManifest

    defaults = dict(
        name="catalog-model",
        task="asr",
        backend="sherpa-onnx",
        model_type="transducer",
        model_url="",
        sample_rate=16000,
        language="en",
        description="catalog desc",
        size_mb=80.0,
        capabilities=["streaming"],
        languages=["en"],
        tags=["english"],
        revision="v1.0",
    )
    defaults.update(kw)
    return ModelManifest(**defaults)


def test_catalog_list_empty(runner: CliRunner, monkeypatch):
    """catalog list when remote returns no entries prints 'No models found'."""
    monkeypatch.setattr("revospeech.catalog.get_catalog_repo", lambda: "org/repo")
    monkeypatch.setattr("revospeech.catalog.list_catalog", lambda task=None: [])
    monkeypatch.setattr("revospeech.catalog.catalog_installed_status", lambda: {})

    result = runner.invoke(cli, ["catalog", "list"])
    assert result.exit_code == 0
    assert "No models found" in result.output


def test_catalog_list_with_entries(runner: CliRunner, monkeypatch):
    """catalog list renders rows with installed indicator."""
    monkeypatch.setattr("revospeech.catalog.get_catalog_repo", lambda: "org/repo")
    fake_models = [
        _fake_catalog_manifest(name="alpha"),
        _fake_catalog_manifest(name="beta"),
    ]
    monkeypatch.setattr(
        "revospeech.catalog.list_catalog", lambda task=None: fake_models
    )
    monkeypatch.setattr(
        "revospeech.catalog.catalog_installed_status",
        lambda: {"alpha": True, "beta": False},
    )

    result = runner.invoke(cli, ["catalog", "list"])
    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "beta" in result.output
    assert "catalog pull" in result.output


def test_catalog_list_fetch_error_exits_one(runner: CliRunner, monkeypatch):
    """catalog list surfaces RuntimeError as exit 1 with suggestion."""

    def boom(_task=None):
        raise RuntimeError("network down")

    monkeypatch.setattr("revospeech.catalog.get_catalog_repo", lambda: "org/repo")
    monkeypatch.setattr("revospeech.catalog.list_catalog", boom)

    result = runner.invoke(cli, ["catalog", "list"])
    assert result.exit_code == 1
    assert "Error" in result.output
    assert "network connection" in result.output


def test_catalog_pull_success(runner: CliRunner, monkeypatch, tmp_path):
    """catalog pull success prints install path and usage hint."""
    monkeypatch.setattr("revospeech.catalog.get_catalog_repo", lambda: "org/repo")
    monkeypatch.setattr(
        "revospeech.catalog.pull_model", lambda name: tmp_path / "model.yaml"
    )

    result = runner.invoke(cli, ["catalog", "pull", "alpha"])
    assert result.exit_code == 0
    assert "Installed to" in result.output
    assert "TTS('alpha')" in result.output or "alpha" in result.output


def test_catalog_pull_not_found_exits_one(runner: CliRunner, monkeypatch):
    """catalog pull on unknown name raises KeyError → exit 1."""

    def boom(_name):
        raise KeyError("not in catalog")

    monkeypatch.setattr("revospeech.catalog.get_catalog_repo", lambda: "org/repo")
    monkeypatch.setattr("revospeech.catalog.pull_model", boom)

    result = runner.invoke(cli, ["catalog", "pull", "ghost"])
    assert result.exit_code == 1
    assert "Error" in result.output


def test_catalog_pull_runtime_error_exits_one(runner: CliRunner, monkeypatch):
    """catalog pull on network failure raises RuntimeError → exit 1."""

    def boom(_name):
        raise RuntimeError("network down")

    monkeypatch.setattr("revospeech.catalog.get_catalog_repo", lambda: "org/repo")
    monkeypatch.setattr("revospeech.catalog.pull_model", boom)

    result = runner.invoke(cli, ["catalog", "pull", "alpha"])
    assert result.exit_code == 1
    assert "Error" in result.output


def test_catalog_search_no_matches(runner: CliRunner, monkeypatch):
    """catalog search with no matches prints suggestion."""
    monkeypatch.setattr(
        "revospeech.catalog.list_catalog", lambda: [_fake_catalog_manifest()]
    )

    result = runner.invoke(cli, ["catalog", "search", "nonexistent"])
    assert result.exit_code == 0
    assert "No models found" in result.output


def test_catalog_search_matches(runner: CliRunner, monkeypatch):
    """catalog search filters by name and prints matches."""
    monkeypatch.setattr(
        "revospeech.catalog.list_catalog",
        lambda: [
            _fake_catalog_manifest(name="alpha-asr", description="english ASR"),
            _fake_catalog_manifest(name="beta-tts", description="english TTS"),
        ],
    )

    result = runner.invoke(cli, ["catalog", "search", "alpha"])
    assert result.exit_code == 0
    assert "alpha-asr" in result.output
    assert "beta-tts" not in result.output


def test_catalog_search_filter_by_language(runner: CliRunner, monkeypatch):
    """catalog search --language filters by languages list."""
    monkeypatch.setattr(
        "revospeech.catalog.list_catalog",
        lambda: [
            _fake_catalog_manifest(name="english-m", languages=["en"]),
            _fake_catalog_manifest(name="french-m", languages=["fr"]),
        ],
    )

    result = runner.invoke(cli, ["catalog", "search", "m", "--language", "fr"])
    assert result.exit_code == 0
    assert "french-m" in result.output
    assert "english-m" not in result.output


def test_catalog_recommend_with_results(runner: CliRunner, monkeypatch):
    """catalog recommend prints top 3 by size."""
    monkeypatch.setattr(
        "revospeech.catalog.recommend_models",
        lambda task=None, language=None: [
            _fake_catalog_manifest(name="a", size_mb=10.0),
            _fake_catalog_manifest(name="b", size_mb=20.0),
        ],
    )

    result = runner.invoke(cli, ["catalog", "recommend"])
    assert result.exit_code == 0
    assert "recommended" in result.output
    assert "a" in result.output and "b" in result.output
    assert "catalog pull" in result.output


def test_catalog_recommend_no_results(runner: CliRunner, monkeypatch):
    """catalog recommend prints 'No matching' when recommend_models returns []."""
    monkeypatch.setattr(
        "revospeech.catalog.recommend_models", lambda task=None, language=None: []
    )

    result = runner.invoke(cli, ["catalog", "recommend"])
    assert result.exit_code == 0
    assert "No matching" in result.output


def test_catalog_recommend_runtime_error_exits_one(runner: CliRunner, monkeypatch):
    """catalog recommend surfaces RuntimeError as exit 1."""

    def boom(task=None, language=None):
        raise RuntimeError("offline")

    monkeypatch.setattr("revospeech.catalog.recommend_models", boom)

    result = runner.invoke(cli, ["catalog", "recommend"])
    assert result.exit_code == 1
    assert "Error" in result.output


# ---------------------------------------------------------------------------
# transcribe single-file: exception handler coverage
# ---------------------------------------------------------------------------
@patch("revospeech.asr.ASR")
def test_transcribe_revos_audio_error(mock_asr_cls, runner: CliRunner, sample_wav):
    """transcribe surfaces RevosAudioError with formatted message."""
    from revospeech.exceptions import RevosAudioError

    mock_asr = MagicMock()
    mock_asr.transcribe.side_effect = RevosAudioError(
        "Bad audio", suggestion="Use 16kHz WAV"
    )
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav])
    assert result.exit_code == 1
    assert "Audio error" in result.output
    assert "Use 16kHz WAV" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_revos_model_error(mock_asr_cls, runner: CliRunner, sample_wav):
    """transcribe surfaces RevosModelError with formatted message."""
    from revospeech.exceptions import RevosModelError

    mock_asr = MagicMock()
    mock_asr.transcribe.side_effect = RevosModelError(
        "Model missing", suggestion="Run download"
    )
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav])
    assert result.exit_code == 1
    assert "Model error" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_generic_revos_error(mock_asr_cls, runner: CliRunner, sample_wav):
    """transcribe surfaces plain RevosError with formatted message."""
    from revospeech.exceptions import RevosError

    mock_asr = MagicMock()
    mock_asr.transcribe.side_effect = RevosError("Boom", suggestion="Retry")
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav])
    assert result.exit_code == 1
    assert "Error" in result.output
    assert "Retry" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_unexpected_exception(mock_asr_cls, runner: CliRunner, sample_wav):
    """transcribe surfaces unknown exceptions with verbose hint."""
    mock_asr = MagicMock()
    mock_asr.transcribe.side_effect = RuntimeError("kaboom")
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav])
    assert result.exit_code == 1
    assert "Unexpected error" in result.output
    assert "RuntimeError" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_vtt_output_single_file(mock_asr_cls, runner: CliRunner, sample_wav):
    """transcribe --format vtt emits WEBVTT-formatted output."""
    from revospeech.asr.result import Segment, Transcript

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = Transcript(
        text="HELLO",
        segments=[Segment(start=0.0, end=1.5, text="HELLO", confidence=0.9)],
        language="en",
    )
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(
        cli, ["transcribe", "-m", "test", "--format", "vtt", sample_wav]
    )
    assert result.exit_code == 0
    assert "WEBVTT" in result.output
    assert "00:00:00.000 --> 00:00:01.500" in result.output
    assert "HELLO" in result.output


# ---------------------------------------------------------------------------
# synthesize: exception handlers + synthesize_long path
# ---------------------------------------------------------------------------
@patch("revospeech.tts.TTS")
def test_synthesize_revos_engine_error(mock_tts_cls, runner: CliRunner):
    """synthesize surfaces RevosEngineError with formatted message."""
    from revospeech.exceptions import RevosEngineError

    mock_tts = MagicMock()
    mock_tts.synthesize.side_effect = RevosEngineError(
        "Inference failed", suggestion="Check device"
    )
    mock_tts_cls.return_value = mock_tts

    result = runner.invoke(
        cli, ["synthesize", "-m", "test", "-t", "hi", "-o", "out.wav"]
    )
    assert result.exit_code == 1
    assert "Engine error" in result.output
    assert "Check device" in result.output


@patch("revospeech.tts.TTS")
def test_synthesize_revos_model_error(mock_tts_cls, runner: CliRunner):
    """synthesize surfaces RevosModelError with formatted message."""
    from revospeech.exceptions import RevosModelError

    mock_tts = MagicMock()
    mock_tts.synthesize.side_effect = RevosModelError(
        "Model missing", suggestion="Download first"
    )
    mock_tts_cls.return_value = mock_tts

    result = runner.invoke(
        cli, ["synthesize", "-m", "test", "-t", "hi", "-o", "out.wav"]
    )
    assert result.exit_code == 1
    assert "Model error" in result.output


@patch("revospeech.tts.TTS")
def test_synthesize_revos_audio_error(mock_tts_cls, runner: CliRunner):
    """synthesize surfaces RevosAudioError with formatted message."""
    from revospeech.exceptions import RevosAudioError

    mock_tts = MagicMock()
    mock_tts.synthesize.side_effect = RevosAudioError(
        "Bad ref", suggestion="Use valid WAV"
    )
    mock_tts_cls.return_value = mock_tts

    result = runner.invoke(
        cli, ["synthesize", "-m", "test", "-t", "hi", "-o", "out.wav"]
    )
    assert result.exit_code == 1
    assert "Audio error" in result.output


@patch("revospeech.tts.TTS")
def test_synthesize_generic_revos_error(mock_tts_cls, runner: CliRunner):
    """synthesize surfaces plain RevosError with formatted message."""
    from revospeech.exceptions import RevosError

    mock_tts = MagicMock()
    mock_tts.synthesize.side_effect = RevosError("Boom", suggestion="Retry")
    mock_tts_cls.return_value = mock_tts

    result = runner.invoke(
        cli, ["synthesize", "-m", "test", "-t", "hi", "-o", "out.wav"]
    )
    assert result.exit_code == 1
    assert "Error" in result.output


@patch("revospeech.tts.TTS")
def test_synthesize_unexpected_exception(mock_tts_cls, runner: CliRunner):
    """synthesize surfaces unknown exceptions with verbose hint."""
    mock_tts = MagicMock()
    mock_tts.synthesize.side_effect = RuntimeError("kaboom")
    mock_tts_cls.return_value = mock_tts

    result = runner.invoke(
        cli, ["synthesize", "-m", "test", "-t", "hi", "-o", "out.wav"]
    )
    assert result.exit_code == 1
    assert "Unexpected error" in result.output
    assert "RuntimeError" in result.output


@patch("revospeech.tts.TTS")
def test_synthesize_long_path_for_long_text(mock_tts_cls, runner: CliRunner):
    """Text > 500 chars routes through tts.synthesize_long."""
    import numpy as np

    from revospeech.tts.result import Audio

    mock_tts = MagicMock()
    samples = np.zeros(24000, dtype=np.float32)
    mock_tts.synthesize_long.return_value = Audio(samples=samples, sample_rate=24000)
    mock_tts_cls.return_value = mock_tts

    long_text = "a" * 600
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["synthesize", "-m", "test", "-t", long_text, "-o", "out.wav"],
        )
    assert result.exit_code == 0
    assert "Saved" in result.output
    mock_tts.synthesize_long.assert_called_once()
    mock_tts.synthesize.assert_not_called()


# ---------------------------------------------------------------------------
# Batch mode: transcribe multiple files, synthesize --file-list
# ---------------------------------------------------------------------------
@patch("revospeech.asr.ASR")
def test_transcribe_batch_text_output(mock_asr_cls, runner: CliRunner, sample_wav):
    """Batch transcribe prints per-file text headers and summary."""
    from revospeech.asr.result import BatchReport, BatchResult, Segment, Transcript

    transcript = Transcript(
        text="HELLO",
        segments=[Segment(start=0.0, end=0.5, text="HELLO", confidence=0.9)],
        language="en",
    )
    item = BatchResult(input=sample_wav, result=transcript, duration=1.0)
    report = BatchReport(
        items=[item, item], total=2, succeeded=2, failed=0, total_duration=2.0
    )
    mock_asr = MagicMock()
    mock_asr.transcribe_batch.return_value = report
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav, sample_wav])
    assert result.exit_code == 0
    assert "Transcribed 2/2" in result.output
    assert "=== " in result.output
    assert "HELLO" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_batch_json_output(mock_asr_cls, runner: CliRunner, sample_wav):
    """Batch transcribe --format json emits per-file JSON objects."""
    import json

    from revospeech.asr.result import BatchReport, BatchResult, Segment, Transcript

    transcript = Transcript(
        text="HELLO",
        segments=[Segment(start=0.0, end=0.5, text="HELLO", confidence=0.9)],
        language="en",
    )
    item = BatchResult(input=sample_wav, result=transcript, duration=1.0)
    report = BatchReport(
        items=[item], total=1, succeeded=1, failed=0, total_duration=1.0
    )
    mock_asr = MagicMock()
    mock_asr.transcribe_batch.return_value = report
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(
        cli, ["transcribe", "-m", "test", "--format", "json", sample_wav, sample_wav]
    )
    assert result.exit_code == 0
    assert "Transcribed 1/1" in result.output
    parsed = json.loads(result.output.split("\n", 1)[1])
    assert parsed["text"] == "HELLO"


@patch("revospeech.asr.ASR")
def test_transcribe_batch_reports_failed_items(
    mock_asr_cls, runner: CliRunner, sample_wav
):
    """Failed items in batch report are printed to stderr."""
    from revospeech.asr.result import BatchReport, BatchResult

    item = BatchResult(input=sample_wav, error="Corrupt WAV", duration=0.0)
    report = BatchReport(
        items=[item], total=1, succeeded=0, failed=1, total_duration=0.0
    )
    mock_asr = MagicMock()
    mock_asr.transcribe_batch.return_value = report
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(
        cli,
        ["transcribe", "-m", "test", sample_wav, sample_wav],
    )
    assert result.exit_code == 0
    assert "Transcribed 0/1" in result.output


@patch("revospeech.tts.TTS")
def test_synthesize_file_list_batch(mock_tts_cls, runner: CliRunner):
    """--file-list runs synthesize_batch and prints status per line."""
    from revospeech.asr.result import BatchReport, BatchResult

    item_ok = BatchResult(input="hello", result=None, duration=0.2)
    item_fail = BatchResult(input="world", error="Boom", duration=0.0)
    report = BatchReport(
        items=[item_ok, item_fail],
        total=2,
        succeeded=1,
        failed=1,
        total_duration=0.5,
    )
    mock_tts = MagicMock()
    mock_tts.synthesize_batch.return_value = report
    mock_tts_cls.return_value = mock_tts

    with runner.isolated_filesystem():
        with open("list.txt", "w") as f:
            f.write("hello\nworld\n")

        result = runner.invoke(
            cli,
            ["synthesize", "-m", "test", "--file-list", "list.txt", "-o", "out/x.wav"],
        )
    assert result.exit_code == 0
    assert "Synthesized 1/2" in result.output
    assert "OK" in result.output
    assert "FAIL" in result.output


def test_synthesize_file_list_conflict_with_text(runner: CliRunner):
    """--file-list with --text is rejected as UsageError."""
    with runner.isolated_filesystem():
        with open("list.txt", "w") as f:
            f.write("hello\n")
        result = runner.invoke(
            cli,
            [
                "synthesize",
                "-m",
                "test",
                "-t",
                "hi",
                "--file-list",
                "list.txt",
                "-o",
                "out.wav",
            ],
        )
    assert result.exit_code != 0


def test_synthesize_file_list_empty(runner: CliRunner):
    """Empty --file-list is rejected."""
    with runner.isolated_filesystem():
        with open("empty.txt", "w") as f:
            f.write("\n")
        result = runner.invoke(
            cli,
            ["synthesize", "-m", "test", "--file-list", "empty.txt", "-o", "out.wav"],
        )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# setup wizard
# ---------------------------------------------------------------------------
def test_setup_decline_all(runner: CliRunner, monkeypatch):
    """setup wizard completes without installing when user declines."""
    from revospeech.registry.status import ModelStatus

    monkeypatch.setattr(
        "revospeech.registry.status.list_model_statuses",
        lambda task=None: [
            ModelStatus(
                name="alpha",
                task="asr",
                mode="local",
                status="ready",
                installed=True,
                size_mb=10.0,
                capabilities=[],
                languages=["en"],
            )
        ],
    )

    result = runner.invoke(cli, ["setup"], input="asr\nn\nn\n")
    assert result.exit_code == 0
    assert "Setup complete" in result.output


def test_setup_no_models_registered(runner: CliRunner, monkeypatch):
    """setup wizard handles missing models gracefully."""
    monkeypatch.setattr(
        "revospeech.registry.status.list_model_statuses", lambda task=None: []
    )

    result = runner.invoke(cli, ["setup"], input="asr\nn\nn\n")
    assert result.exit_code == 0
    assert "No asr models" in result.output
    assert "Setup complete" in result.output


# ---------------------------------------------------------------------------
# Color helpers — TTY branches via _use_color patch
# ---------------------------------------------------------------------------
def test_status_text_with_color(monkeypatch):
    """When _use_color is True, _status_text applies click.style coloring."""
    from revospeech.cli import main as cli_main

    monkeypatch.setattr(cli_main, "_use_color", lambda: True)
    assert cli_main._status_text("ready") == click.style("✓ ready", fg="green")
    assert cli_main._status_text("needs-download") == click.style(
        "↓ needs-download", fg="yellow"
    )
    assert cli_main._status_text("needs-api-key") == click.style(
        "✗ needs-api-key", fg="red"
    )
    assert cli_main._status_text("unknown") == "? unknown"


def test_installed_status_text_with_color(monkeypatch):
    """When _use_color is True, _installed_status_text applies click.style."""
    from revospeech.cli import main as cli_main

    monkeypatch.setattr(cli_main, "_use_color", lambda: True)
    assert cli_main._installed_status_text(True) == click.style(
        "✓ installed", fg="green"
    )
    assert cli_main._installed_status_text(False) == click.style(
        "↓ not installed", fg="yellow"
    )


# ---------------------------------------------------------------------------
# Batch transcribe: srt + vtt output formats
# ---------------------------------------------------------------------------
@patch("revospeech.asr.ASR")
def test_transcribe_batch_srt_output(mock_asr_cls, runner: CliRunner, sample_wav):
    """Batch transcribe --format srt emits per-file SRT blocks."""
    from revospeech.asr.result import BatchReport, BatchResult, Segment, Transcript

    transcript = Transcript(
        text="HELLO",
        segments=[Segment(start=0.0, end=1.5, text="HELLO", confidence=0.9)],
        language="en",
    )
    item = BatchResult(input=sample_wav, result=transcript, duration=1.0)
    report = BatchReport(
        items=[item], total=1, succeeded=1, failed=0, total_duration=1.0
    )
    mock_asr = MagicMock()
    mock_asr.transcribe_batch.return_value = report
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(
        cli, ["transcribe", "-m", "test", "--format", "srt", sample_wav, sample_wav]
    )
    assert result.exit_code == 0
    assert "===" in result.output
    assert "-->" in result.output
    assert "HELLO" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_batch_vtt_output(mock_asr_cls, runner: CliRunner, sample_wav):
    """Batch transcribe --format vtt emits per-file WebVTT blocks."""
    from revospeech.asr.result import BatchReport, BatchResult, Segment, Transcript

    transcript = Transcript(
        text="HELLO",
        segments=[Segment(start=0.0, end=1.5, text="HELLO", confidence=0.9)],
        language="en",
    )
    item = BatchResult(input=sample_wav, result=transcript, duration=1.0)
    report = BatchReport(
        items=[item], total=1, succeeded=1, failed=0, total_duration=1.0
    )
    mock_asr = MagicMock()
    mock_asr.transcribe_batch.return_value = report
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(
        cli, ["transcribe", "-m", "test", "--format", "vtt", sample_wav, sample_wav]
    )
    assert result.exit_code == 0
    assert "WEBVTT" in result.output
    assert "00:00:00.000 --> 00:00:01.500" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_batch_revos_engine_error(
    mock_asr_cls, runner: CliRunner, sample_wav
):
    """Batch transcribe surfaces RevosEngineError."""
    from revospeech.exceptions import RevosEngineError

    mock_asr = MagicMock()
    mock_asr.transcribe_batch.side_effect = RevosEngineError(
        "Inference failed", suggestion="Check device"
    )
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav, sample_wav])
    assert result.exit_code == 1
    assert "Engine error" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_batch_revos_audio_error(
    mock_asr_cls, runner: CliRunner, sample_wav
):
    """Batch transcribe surfaces RevosAudioError."""
    from revospeech.exceptions import RevosAudioError

    mock_asr = MagicMock()
    mock_asr.transcribe_batch.side_effect = RevosAudioError(
        "Bad audio", suggestion="Use 16kHz WAV"
    )
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav, sample_wav])
    assert result.exit_code == 1
    assert "Audio error" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_batch_revos_model_error(
    mock_asr_cls, runner: CliRunner, sample_wav
):
    """Batch transcribe surfaces RevosModelError."""
    from revospeech.exceptions import RevosModelError

    mock_asr = MagicMock()
    mock_asr.transcribe_batch.side_effect = RevosModelError(
        "Model missing", suggestion="Download"
    )
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav, sample_wav])
    assert result.exit_code == 1
    assert "Model error" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_batch_generic_revos_error(
    mock_asr_cls, runner: CliRunner, sample_wav
):
    """Batch transcribe surfaces plain RevosError."""
    from revospeech.exceptions import RevosError

    mock_asr = MagicMock()
    mock_asr.transcribe_batch.side_effect = RevosError("Boom", suggestion="Retry")
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav, sample_wav])
    assert result.exit_code == 1
    assert "Error" in result.output


@patch("revospeech.asr.ASR")
def test_transcribe_batch_unexpected_exception(
    mock_asr_cls, runner: CliRunner, sample_wav
):
    """Batch transcribe surfaces unexpected exceptions with verbose hint."""
    mock_asr = MagicMock()
    mock_asr.transcribe_batch.side_effect = RuntimeError("kaboom")
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav, sample_wav])
    assert result.exit_code == 1
    assert "Unexpected error" in result.output
    assert "RuntimeError" in result.output


# ---------------------------------------------------------------------------
# Batch synthesize: --file-list exception handlers
# ---------------------------------------------------------------------------
@patch("revospeech.tts.TTS")
def test_synthesize_file_list_revos_engine_error(mock_tts_cls, runner: CliRunner):
    """--file-list surfaces RevosEngineError."""
    from revospeech.exceptions import RevosEngineError

    mock_tts = MagicMock()
    mock_tts.synthesize_batch.side_effect = RevosEngineError(
        "Inference failed", suggestion="Check device"
    )
    mock_tts_cls.return_value = mock_tts

    with runner.isolated_filesystem():
        with open("list.txt", "w") as f:
            f.write("hello\n")
        result = runner.invoke(
            cli,
            ["synthesize", "-m", "test", "--file-list", "list.txt", "-o", "out/x.wav"],
        )
    assert result.exit_code == 1
    assert "Engine error" in result.output


@patch("revospeech.tts.TTS")
def test_synthesize_file_list_revos_model_error(mock_tts_cls, runner: CliRunner):
    """--file-list surfaces RevosModelError."""
    from revospeech.exceptions import RevosModelError

    mock_tts = MagicMock()
    mock_tts.synthesize_batch.side_effect = RevosModelError(
        "Model missing", suggestion="Download"
    )
    mock_tts_cls.return_value = mock_tts

    with runner.isolated_filesystem():
        with open("list.txt", "w") as f:
            f.write("hello\n")
        result = runner.invoke(
            cli,
            ["synthesize", "-m", "test", "--file-list", "list.txt", "-o", "out/x.wav"],
        )
    assert result.exit_code == 1
    assert "Model error" in result.output


@patch("revospeech.tts.TTS")
def test_synthesize_file_list_revos_audio_error(mock_tts_cls, runner: CliRunner):
    """--file-list surfaces RevosAudioError."""
    from revospeech.exceptions import RevosAudioError

    mock_tts = MagicMock()
    mock_tts.synthesize_batch.side_effect = RevosAudioError(
        "Bad audio", suggestion="Use 16kHz WAV"
    )
    mock_tts_cls.return_value = mock_tts

    with runner.isolated_filesystem():
        with open("list.txt", "w") as f:
            f.write("hello\n")
        result = runner.invoke(
            cli,
            ["synthesize", "-m", "test", "--file-list", "list.txt", "-o", "out/x.wav"],
        )
    assert result.exit_code == 1
    assert "Audio error" in result.output


@patch("revospeech.tts.TTS")
def test_synthesize_file_list_generic_revos_error(mock_tts_cls, runner: CliRunner):
    """--file-list surfaces plain RevosError."""
    from revospeech.exceptions import RevosError

    mock_tts = MagicMock()
    mock_tts.synthesize_batch.side_effect = RevosError("Boom", suggestion="Retry")
    mock_tts_cls.return_value = mock_tts

    with runner.isolated_filesystem():
        with open("list.txt", "w") as f:
            f.write("hello\n")
        result = runner.invoke(
            cli,
            ["synthesize", "-m", "test", "--file-list", "list.txt", "-o", "out/x.wav"],
        )
    assert result.exit_code == 1
    assert "Error" in result.output


@patch("revospeech.tts.TTS")
def test_synthesize_file_list_unexpected_exception(mock_tts_cls, runner: CliRunner):
    """--file-list surfaces unexpected exceptions with verbose hint."""
    mock_tts = MagicMock()
    mock_tts.synthesize_batch.side_effect = RuntimeError("kaboom")
    mock_tts_cls.return_value = mock_tts

    with runner.isolated_filesystem():
        with open("list.txt", "w") as f:
            f.write("hello\n")
        result = runner.invoke(
            cli,
            ["synthesize", "-m", "test", "--file-list", "list.txt", "-o", "out/x.wav"],
        )
    assert result.exit_code == 1
    assert "Unexpected error" in result.output
    assert "RuntimeError" in result.output


# ---------------------------------------------------------------------------
# catalog list edge: catalog_installed_status failure
# ---------------------------------------------------------------------------
def test_catalog_list_handles_installed_status_failure(runner: CliRunner, monkeypatch):
    """catalog list tolerates catalog_installed_status exceptions."""

    def boom():
        raise RuntimeError("local check failed")

    monkeypatch.setattr("revospeech.catalog.get_catalog_repo", lambda: "org/repo")
    monkeypatch.setattr(
        "revospeech.catalog.list_catalog",
        lambda task=None: [_fake_catalog_manifest()],
    )
    monkeypatch.setattr("revospeech.catalog.catalog_installed_status", boom)

    result = runner.invoke(cli, ["catalog", "list"])
    assert result.exit_code == 0
    assert "catalog-model" in result.output


# ---------------------------------------------------------------------------
# transcribe single-file RevosConfigError path (small gap on lines 157-158)
# ---------------------------------------------------------------------------
@patch("revospeech.asr.ASR")
def test_transcribe_revos_config_error(mock_asr_cls, runner: CliRunner, sample_wav):
    """transcribe surfaces RevosConfigError on ASR construction."""
    from revospeech.exceptions import RevosConfigError

    mock_asr_cls.side_effect = RevosConfigError(
        "Missing API key", suggestion="Run 'revos config set-api-key'"
    )

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav])
    assert result.exit_code == 1
    assert "Configuration error" in result.output
    assert "set-api-key" in result.output


# ---------------------------------------------------------------------------
# Remaining batch error paths + setup wizard interactive flows
# ---------------------------------------------------------------------------
@patch("revospeech.asr.ASR")
def test_transcribe_batch_revos_config_error(
    mock_asr_cls, runner: CliRunner, sample_wav
):
    """Batch transcribe surfaces RevosConfigError."""
    from revospeech.exceptions import RevosConfigError

    mock_asr_cls.side_effect = RevosConfigError(
        "Missing API key", suggestion="Run 'revos config set-api-key'"
    )

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav, sample_wav])
    assert result.exit_code == 1
    assert "Configuration error" in result.output


@patch("revospeech.tts.TTS")
def test_synthesize_file_list_revos_config_error(mock_tts_cls, runner: CliRunner):
    """--file-list surfaces RevosConfigError."""
    from revospeech.exceptions import RevosConfigError

    mock_tts_cls.side_effect = RevosConfigError(
        "Missing API key", suggestion="Run 'revos config set-api-key'"
    )

    with runner.isolated_filesystem():
        with open("list.txt", "w") as f:
            f.write("hello\n")
        result = runner.invoke(
            cli,
            ["synthesize", "-m", "test", "--file-list", "list.txt", "-o", "out/x.wav"],
        )
    assert result.exit_code == 1
    assert "Configuration error" in result.output


def test_setup_install_flow_success(runner: CliRunner, monkeypatch):
    """setup wizard install path downloads a model when user confirms."""
    from revospeech.registry.status import ModelStatus

    monkeypatch.setattr(
        "revospeech.registry.status.list_model_statuses",
        lambda task=None: [
            ModelStatus(
                name="alpha",
                task="asr",
                mode="local",
                status="needs-download",
                installed=False,
                size_mb=10.0,
                capabilities=[],
                languages=["en"],
            )
        ],
    )

    captured = []
    monkeypatch.setattr(
        "revospeech.registry.downloader.ensure_model",
        lambda manifest: captured.append(manifest.name),
    )
    # Fake registry: name "alpha" resolves to a dummy manifest.
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models

    saved_models = dict(_models)
    _models.clear()
    _models[("alpha", "asr")] = ModelManifest(
        name="alpha",
        task="asr",
        backend="sherpa-onnx",
        model_type="transducer",
        model_url="http://example.com/m.tar.bz2",
        sample_rate=16000,
        language="en",
        description="",
        files={},
    )

    try:
        result = runner.invoke(cli, ["setup"], input="asr\ny\nalpha\nn\n")
        assert result.exit_code == 0
        assert "Downloading alpha" in result.output
        assert "Done" in result.output
        assert captured == ["alpha"]
    finally:
        _models.clear()
        _models.update(saved_models)


def test_setup_install_unknown_model(runner: CliRunner, monkeypatch):
    """setup wizard install of unknown model prints not-found hint."""
    from revospeech.registry.registry import _models

    saved_models = dict(_models)
    _models.clear()

    try:
        result = runner.invoke(cli, ["setup"], input="asr\ny\nghost\nn\n")
        assert result.exit_code == 0
        assert "not found" in result.output
    finally:
        _models.clear()
        _models.update(saved_models)


def test_setup_install_runtime_error_is_caught(runner: CliRunner, monkeypatch):
    """setup wizard install errors are surfaced but don't crash."""

    def boom(_manifest):
        raise RuntimeError("disk full")

    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models

    saved_models = dict(_models)
    _models.clear()
    _models[("alpha", "asr")] = ModelManifest(
        name="alpha",
        task="asr",
        backend="sherpa-onnx",
        model_type="transducer",
        model_url="http://example.com/m.tar.bz2",
        sample_rate=16000,
        language="en",
        description="",
        files={},
    )
    monkeypatch.setattr("revospeech.registry.downloader.ensure_model", boom)

    try:
        result = runner.invoke(cli, ["setup"], input="asr\ny\nalpha\nn\n")
        assert result.exit_code == 0
        assert "Error" in result.output
        assert "disk full" in result.output
    finally:
        _models.clear()
        _models.update(saved_models)


def test_setup_set_api_key_flow(runner: CliRunner, monkeypatch):
    """setup wizard API key step saves when user confirms."""
    saved_keys = []
    monkeypatch.setattr(
        "revospeech.config.set_api_key", lambda key: saved_keys.append(key)
    )
    monkeypatch.setattr(
        "revospeech.registry.status.list_model_statuses", lambda task=None: []
    )

    result = runner.invoke(cli, ["setup"], input="asr\nn\ny\nrv-newkey12345\n")
    assert result.exit_code == 0
    assert saved_keys == ["rv-newkey12345"]
    assert "API key saved" in result.output


def test_catalog_search_filter_by_task(runner: CliRunner, monkeypatch):
    """catalog search --task filters out non-matching tasks."""
    monkeypatch.setattr(
        "revospeech.catalog.list_catalog",
        lambda: [
            _fake_catalog_manifest(name="alpha-asr", task="asr"),
            _fake_catalog_manifest(name="beta-tts", task="tts"),
        ],
    )

    # Query "en" matches the default tag "english" and language "en".
    result = runner.invoke(cli, ["catalog", "search", "en", "--task", "tts"])
    assert result.exit_code == 0
    assert "beta-tts" in result.output
    assert "alpha-asr" not in result.output
