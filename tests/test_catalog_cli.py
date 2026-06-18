"""Tests for catalog CLI commands."""

from __future__ import annotations

from unittest.mock import patch

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


@patch("revospeech.catalog._download_raw")
@patch("revospeech.catalog._list_yaml_files")
def test_catalog_list_command(mock_list, mock_download, runner):
    """revos catalog list shows models from GitHub."""
    mock_list.return_value = ["revos/models/tts/revovoice.yaml"]

    manifest = (
        "name: revovoice\n"
        "task: tts\n"
        "backend: revovoice\n"
        "model_type: diffusion\n"
        "model_url: test\n"
        "sample_rate: 24000\n"
        "language: multilingual\n"
        "description: Test\n"
        "files: {}\n"
    )
    mock_download.return_value = manifest

    result = runner.invoke(cli, ["catalog", "list"])
    assert result.exit_code == 0
    assert "revovoice" in result.output
    assert "Fetching catalog" in result.output


@patch("revospeech.catalog._list_yaml_files")
def test_catalog_list_error(mock_list, runner):
    """revos catalog list handles errors gracefully."""
    mock_list.side_effect = Exception("network error")

    result = runner.invoke(cli, ["catalog", "list"])
    assert result.exit_code == 1
    assert "Error" in result.output


@patch("revospeech.catalog._download_raw")
@patch("revospeech.catalog._list_yaml_files")
def test_catalog_list_empty(mock_list, mock_download, runner):
    """revos catalog list shows message when no models found."""
    mock_list.return_value = []

    result = runner.invoke(cli, ["catalog", "list"])
    assert result.exit_code == 0
    assert "No models found" in result.output


@patch("revospeech.catalog._download_raw")
@patch("revospeech.catalog._list_yaml_files")
def test_catalog_list_filter_by_task(mock_list, mock_download, runner):
    """revos catalog list -t tts filters by task."""
    mock_list.return_value = ["revos/models/tts/revovoice.yaml"]

    manifest = (
        "name: revovoice\n"
        "task: tts\n"
        "backend: revovoice\n"
        "model_type: diffusion\n"
        "model_url: test\n"
        "sample_rate: 24000\n"
        "language: multilingual\n"
        "description: Test\n"
        "files: {}\n"
    )
    mock_download.return_value = manifest

    result = runner.invoke(cli, ["catalog", "list", "-t", "tts"])
    assert result.exit_code == 0
    assert "revovoice" in result.output


@patch("revospeech.catalog._download_raw")
@patch("revospeech.catalog._list_yaml_files")
def test_catalog_pull_command(mock_list, mock_download, runner, tmp_path):
    """revos catalog pull installs a model."""
    mock_list.return_value = ["revos/models/tts/revovoice.yaml"]

    manifest = (
        "name: revovoice\n"
        "task: tts\n"
        "backend: revovoice\n"
        "model_type: diffusion\n"
        "model_url: test\n"
        "sample_rate: 24000\n"
        "language: multilingual\n"
        "description: Test\n"
        "files: {}\n"
    )
    mock_download.side_effect = [manifest, manifest]

    models_dir = tmp_path / "models"
    with patch("revospeech.catalog._USER_MODELS_DIR", models_dir):
        result = runner.invoke(cli, ["catalog", "pull", "revovoice"])

    assert result.exit_code == 0
    assert "Installed" in result.output


@patch("revospeech.catalog._list_yaml_files")
def test_catalog_pull_not_found(mock_list, runner):
    """revos catalog pull handles missing model."""
    mock_list.return_value = []

    result = runner.invoke(cli, ["catalog", "pull", "nonexistent"])
    assert result.exit_code == 1
    assert "Error" in result.output


def test_info_shows_catalog_repo(runner):
    """revos info shows the configured catalog repo."""
    result = runner.invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "Catalog repo" in result.output


@patch("revospeech.catalog._download_raw")
@patch("revospeech.catalog._list_yaml_files")
def test_catalog_search_matches_name(mock_list, mock_download, runner):
    """revos catalog search finds models by name."""
    mock_list.return_value = ["revos/models/asr/zipformer.yaml"]
    mock_download.return_value = (
        "name: zipformer-v2\n"
        "task: asr\n"
        "backend: sherpa-onnx\n"
        "model_url: test\n"
        "sample_rate: 16000\n"
        "language: en\n"
        "description: Fast English ASR\n"
        "files: {}\n"
    )

    result = runner.invoke(cli, ["catalog", "search", "zip"])
    assert result.exit_code == 0
    assert "zipformer-v2" in result.output


@patch("revospeech.catalog._download_raw")
@patch("revospeech.catalog._list_yaml_files")
def test_catalog_search_no_matches(mock_list, mock_download, runner):
    """revos catalog search shows suggestion when nothing matches."""
    mock_list.return_value = ["revos/models/asr/zipformer.yaml"]
    mock_download.return_value = (
        "name: zipformer-v2\n"
        "task: asr\n"
        "backend: sherpa-onnx\n"
        "model_url: test\n"
        "sample_rate: 16000\n"
        "language: en\n"
        "description: Fast English ASR\n"
        "files: {}\n"
    )

    result = runner.invoke(cli, ["catalog", "search", "klingon"])
    assert result.exit_code == 0
    assert "No models found" in result.output


@patch("revospeech.catalog._download_raw")
@patch("revospeech.catalog._list_yaml_files")
def test_catalog_search_filter_by_task(mock_list, mock_download, runner):
    """revos catalog search --task filters results."""
    mock_list.return_value = [
        "revos/models/asr/zipformer.yaml",
        "revos/models/tts/vits.yaml",
    ]
    mock_download.side_effect = [
        (
            "name: zipformer-v2\n"
            "task: asr\n"
            "backend: sherpa-onnx\n"
            "model_url: test\n"
            "sample_rate: 16000\n"
            "language: en\n"
            "description: ASR\n"
            "files: {}\n"
        ),
        (
            "name: vits-en\n"
            "task: tts\n"
            "backend: vits\n"
            "model_url: test\n"
            "sample_rate: 22050\n"
            "language: en\n"
            "description: TTS\n"
            "files: {}\n"
        ),
    ]

    result = runner.invoke(cli, ["catalog", "search", "en", "--task", "tts"])
    assert result.exit_code == 0
    assert "vits-en" in result.output
    assert "zipformer-v2" not in result.output
