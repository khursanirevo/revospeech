"""Tests for catalog CLI extras (US-028: installed indicator + recommend)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from revospeech.catalog import _CACHE_FILE
from revospeech.cli.main import cli
from revospeech.registry.registry import _models


@pytest.fixture(autouse=True)
def clear_registry():
    _models.clear()
    if _CACHE_FILE.exists():
        _CACHE_FILE.unlink()
    yield
    _models.clear()
    if _CACHE_FILE.exists():
        _CACHE_FILE.unlink()


@pytest.fixture
def runner():
    return CliRunner()


def test_catalog_recommend_runs(runner):
    """revos catalog recommend lists top models."""
    fake_model = MagicMock()
    fake_model.name = "test-model"
    fake_model.task = "asr"
    fake_model.size_mb = 100.0
    fake_model.language = "en"

    with patch("revospeech.catalog.recommend_models", return_value=[fake_model]):
        result = runner.invoke(cli, ["catalog", "recommend"])

    assert result.exit_code == 0
    assert "test-model" in result.output


def test_catalog_recommend_no_results(runner):
    """revos catalog recommend reports when nothing matches."""
    with patch("revospeech.catalog.recommend_models", return_value=[]):
        result = runner.invoke(cli, ["catalog", "recommend"])

    assert result.exit_code == 0
    assert "No matching" in result.output


def test_catalog_recommend_filters_by_task(runner):
    """revos catalog recommend -t asr passes task filter through."""
    with patch("revospeech.catalog.recommend_models") as mock_rec:
        mock_rec.return_value = []
        result = runner.invoke(cli, ["catalog", "recommend", "--task", "asr"])

    assert result.exit_code == 0
    args, kwargs = mock_rec.call_args
    assert kwargs.get("task") == "asr"


def test_catalog_recommend_filters_by_language(runner):
    """revos catalog recommend -l fr passes language filter through."""
    with patch("revospeech.catalog.recommend_models") as mock_rec:
        mock_rec.return_value = []
        result = runner.invoke(cli, ["catalog", "recommend", "--language", "fr"])

    assert result.exit_code == 0
    args, kwargs = mock_rec.call_args
    assert kwargs.get("language") == "fr"


def test_catalog_recommend_error_exit_code(runner):
    """revos catalog recommend surfaces RuntimeError as exit 1."""
    with patch(
        "revospeech.catalog.recommend_models",
        side_effect=RuntimeError("network down"),
    ):
        result = runner.invoke(cli, ["catalog", "recommend"])

    assert result.exit_code == 1
    assert "Error" in result.output


def test_catalog_list_shows_installed_indicator(runner):
    """revos catalog list marks installed models correctly."""
    fake_model = MagicMock()
    fake_model.name = "installed-model"
    fake_model.task = "asr"
    fake_model.backend = "sherpa-onnx"
    fake_model.language = "en"
    fake_model.revision = "latest"

    with (
        patch("revospeech.catalog.list_catalog", return_value=[fake_model]),
        patch(
            "revospeech.catalog.catalog_installed_status",
            return_value={"installed-model": True},
        ),
    ):
        result = runner.invoke(cli, ["catalog", "list"])

    assert result.exit_code == 0
    assert "installed-model" in result.output
    assert "installed" in result.output.lower()


def test_catalog_list_shows_not_installed(runner):
    """revos catalog list marks not-installed models correctly."""
    fake_model = MagicMock()
    fake_model.name = "remote-model"
    fake_model.task = "tts"
    fake_model.backend = "revovoice"
    fake_model.language = "multi"
    fake_model.revision = "latest"

    with (
        patch("revospeech.catalog.list_catalog", return_value=[fake_model]),
        patch(
            "revospeech.catalog.catalog_installed_status",
            return_value={"remote-model": False},
        ),
    ):
        result = runner.invoke(cli, ["catalog", "list"])

    assert result.exit_code == 0
    assert "remote-model" in result.output
    assert "not installed" in result.output.lower()


def test_catalog_list_handles_installed_status_error(runner):
    """revos catalog list still works when installed-status lookup fails."""
    fake_model = MagicMock()
    fake_model.name = "any-model"
    fake_model.task = "asr"
    fake_model.backend = "sherpa-onnx"
    fake_model.language = "en"
    fake_model.revision = "latest"

    with (
        patch("revospeech.catalog.list_catalog", return_value=[fake_model]),
        patch(
            "revospeech.catalog.catalog_installed_status",
            side_effect=RuntimeError("offline"),
        ),
    ):
        result = runner.invoke(cli, ["catalog", "list"])

    assert result.exit_code == 0
    assert "any-model" in result.output


def test_catalog_list_includes_status_header(runner):
    """revos catalog list table header includes a Status column."""
    with (
        patch("revospeech.catalog.list_catalog", return_value=[]),
        patch("revospeech.catalog.catalog_installed_status", return_value={}),
    ):
        result = runner.invoke(cli, ["catalog", "list"])

    # Empty list short-circuits to "No models found", so use a model
    assert result.exit_code == 0

    fake_model = MagicMock()
    fake_model.name = "m1"
    fake_model.task = "asr"
    fake_model.backend = "sherpa-onnx"
    fake_model.language = "en"
    fake_model.revision = "latest"

    with (
        patch("revospeech.catalog.list_catalog", return_value=[fake_model]),
        patch("revospeech.catalog.catalog_installed_status", return_value={}),
    ):
        result = runner.invoke(cli, ["catalog", "list"])

    assert result.exit_code == 0
    assert "Status" in result.output
