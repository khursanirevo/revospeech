"""Targeted tests for remaining coverage gaps across modules."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# revospeech.asr.result — BatchReport properties and save edge cases
# ---------------------------------------------------------------------------
def test_batch_report_errors_property():
    """BatchReport.errors returns items with non-None error."""
    from revospeech.asr.result import BatchReport, BatchResult

    report = BatchReport(
        items=[
            BatchResult(input="ok", result=None),
            BatchResult(input="bad", error="boom"),
        ]
    )
    errors = report.errors
    assert len(errors) == 1
    assert errors[0].input == "bad"


def test_batch_report_results_property_filters_none():
    """BatchReport.results skips items where result is None."""
    from revospeech.asr.result import BatchReport, BatchResult, Transcript

    t = Transcript(text="hi", segments=[], language="en")
    report = BatchReport(
        items=[
            BatchResult(input="ok", result=t),
            BatchResult(input="none", result=None),
        ]
    )
    results = report.results
    assert len(results) == 1
    assert results[0].text == "hi"


def test_batch_report_save_with_non_transcript_result(tmp_path: Path):
    """BatchReport.save handles items whose result lacks .text attribute."""
    from revospeech.asr.result import BatchReport, BatchResult

    report = BatchReport(
        items=[BatchResult(input="x", result=object())],  # type: ignore[arg-type]
        total=1,
        succeeded=1,
    )
    out = tmp_path / "report.json"
    report.save(out)
    data = json.loads(out.read_text())
    assert data["items"][0]["result"]["type"] == "object"


# ---------------------------------------------------------------------------
# revospeech.asr.audio — TypeError on unsupported input
# ---------------------------------------------------------------------------
def test_read_samples_raises_typeerror_for_int():
    """_read_samples rejects unsupported types with TypeError."""
    from revospeech.asr.audio import _read_samples

    with pytest.raises(TypeError, match="Unsupported audio input type"):
        _read_samples(42)  # type: ignore[arg-type]


def test_read_samples_raises_typeerror_for_list():
    """_read_samples rejects list input with TypeError."""
    from revospeech.asr.audio import _read_samples

    with pytest.raises(TypeError, match="Unsupported audio input type"):
        _read_samples([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# revospeech.asr.sherpa_engine — CUDA provider branch
# ---------------------------------------------------------------------------
@patch("revospeech.asr.sherpa_engine.sherpa_onnx")
@patch("revospeech.asr.sherpa_engine.ensure_model")
@patch("revospeech.asr.sherpa_engine.get")
def test_sherpa_engine_cuda_provider(mock_get, mock_ensure, mock_sherpa):
    """device='cuda' selects CUDA ExecutionProvider."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models

    manifest = ModelManifest(
        name="cuda-test",
        task="asr",
        backend="sherpa-onnx",
        model_type="transducer",
        model_url="",
        sample_rate=16000,
        language="en",
        description="",
        files={
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "joiner": "joiner.onnx",
            "tokens": "tokens.txt",
        },
    )
    saved = dict(_models)
    _models.clear()
    _models[("asr", "cuda-test")] = manifest
    try:
        mock_get.return_value = manifest
        mock_ensure.return_value = Path("/fake/path")
        mock_recognizer = MagicMock()
        mock_sherpa_obj = MagicMock()
        mock_sherpa_obj.OfflineRecognizer.from_transducer.return_value = mock_recognizer
        mock_sherpa.OfflineRecognizer = mock_sherpa_obj.OfflineRecognizer

        from revospeech.asr.sherpa_engine import SherpaOnnxASR

        engine = SherpaOnnxASR("cuda-test", device="cuda")
        assert engine.device == "cuda"
    finally:
        _models.clear()
        _models.update(saved)


# ---------------------------------------------------------------------------
# revospeech.registry.status — local manifest with files={} and existing dir
# ---------------------------------------------------------------------------
def test_compute_status_local_files_empty_with_existing_dir(
    monkeypatch, tmp_path: Path
):
    """When files={} but model_dir exists, status='ready'."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.status import _compute_status

    monkeypatch.setattr("revospeech.registry.status.CACHE_DIR", tmp_path)
    (tmp_path / "no-files-model").mkdir()
    manifest = ModelManifest(
        name="no-files-model",
        task="asr",
        backend="x",
        model_type="t",
        model_url="",
        sample_rate=16000,
        language="en",
        description="",
        files={},
    )
    status = _compute_status(manifest)
    assert status.installed is True
    assert status.status == "ready"


# ---------------------------------------------------------------------------
# revospeech.registry.registry — .yml extension loading
# ---------------------------------------------------------------------------
def test_load_manifests_from_dir_yml_extension(tmp_path: Path):
    """Both .yaml and .yml are loaded from the same directory."""
    import yaml

    from revospeech.registry.registry import _load_manifests_from_dir, get

    yaml_content = {
        "task": "asr",
        "backend": "sherpa-onnx",
        "model_type": "transducer",
        "model_url": "",
        "sample_rate": 16000,
        "language": "en",
        "description": "",
    }
    (tmp_path / "alpha.yaml").write_text(yaml.dump({**yaml_content, "name": "alpha"}))
    (tmp_path / "beta.yml").write_text(yaml.dump({**yaml_content, "name": "beta"}))

    from revospeech.registry.registry import _models

    saved = dict(_models)
    _models.clear()
    try:
        _load_manifests_from_dir(tmp_path)
        assert get("alpha", "asr").name == "alpha"
        assert get("beta", "asr").name == "beta"
    finally:
        _models.clear()
        _models.update(saved)


def test_load_manifests_from_dir_skips_broken_yml(tmp_path: Path):
    """Broken .yml file is skipped; valid sibling is loaded."""
    import yaml

    from revospeech.registry.registry import (
        _load_manifests_from_dir,
        _models,
        get,
    )

    (tmp_path / "broken.yml").write_text("not: valid: yaml: [")
    yaml_content = {
        "name": "sibling",
        "task": "asr",
        "backend": "sherpa-onnx",
        "model_type": "transducer",
        "model_url": "",
        "sample_rate": 16000,
        "language": "en",
        "description": "",
    }
    (tmp_path / "good.yml").write_text(yaml.dump(yaml_content))

    saved = dict(_models)
    _models.clear()
    try:
        _load_manifests_from_dir(tmp_path)
        assert get("sibling", "asr").name == "sibling"
    finally:
        _models.clear()
        _models.update(saved)


# ---------------------------------------------------------------------------
# revospeech.tts.result — Audio.__repr__
# ---------------------------------------------------------------------------
def test_audio_repr():
    """Audio.__repr__ includes duration, sample_rate, sample count."""
    import numpy as np

    from revospeech.tts.result import Audio

    samples = np.zeros(24000, dtype=np.float32)
    audio = Audio(samples=samples, sample_rate=24000)
    text = repr(audio)
    assert "Audio(" in text
    assert "24000Hz" in text
    assert "samples=24000" in text


# ---------------------------------------------------------------------------
# revospeech.usage — log rotation when file exceeds 10MB
# ---------------------------------------------------------------------------
def test_track_usage_rotates_large_log(tmp_path: Path):
    """track_usage rotates the log file when it exceeds _MAX_LOG_SIZE."""
    import revospeech.usage as usage_mod

    log_path = tmp_path / "usage.jsonl"
    log_path.write_text("x" * (usage_mod._MAX_LOG_SIZE + 100))

    rotated_path = tmp_path / "usage.jsonl.1"

    with patch("revospeech.usage._USAGE_LOG", log_path):
        usage_mod.track_usage(
            event="test_event",
            model_id="org/m",
            model_name="test",
            task="asr",
            hf_user=None,
            device="cpu",
        )

    assert rotated_path.exists(), "rotated file should exist"
    assert log_path.exists(), "new log should be recreated"
    new_contents = log_path.read_text()
    assert json.loads(new_contents)["event"] == "test_event"


def test_track_usage_replaces_existing_rotated_file(tmp_path: Path):
    """If rotated file already exists, it's overwritten during rotation."""
    import revospeech.usage as usage_mod

    log_path = tmp_path / "usage.jsonl"
    rotated_path = tmp_path / "usage.jsonl.1"

    log_path.write_text("x" * (usage_mod._MAX_LOG_SIZE + 100))
    rotated_path.write_text("old rotated content")

    with patch("revospeech.usage._USAGE_LOG", log_path):
        usage_mod.track_usage(
            event="after_rotate",
            model_id="org/m",
            model_name="test",
            task="asr",
            hf_user=None,
            device="cpu",
        )

    assert rotated_path.read_text() != "old rotated content"


# ---------------------------------------------------------------------------
# revospeech.http_client — retry exhaustion in get_raw
# ---------------------------------------------------------------------------
def test_revolab_client_get_raw_retry_exhaustion():
    """get_raw raises RevosEngineError after all retries are exhausted."""
    from revospeech.exceptions import RevosEngineError
    from revospeech.http_client import RevolabClient

    client = RevolabClient(
        endpoint="https://api.example.com/v1",
        api_key="rv-test12345",
        max_retries=2,
    )

    # Replace _httpx and _client so the real httpx is never used.
    fake_httpx = MagicMock()
    network_err = type("HTTPError", (Exception,), {})
    fake_httpx.HTTPError = network_err
    client._httpx = fake_httpx

    fake_inner_client = MagicMock()
    fake_inner_client.request.side_effect = network_err("boom")
    client._client = fake_inner_client

    with pytest.raises(RevosEngineError, match="Network error"):
        client.get_raw("https://example.com/file")


# ---------------------------------------------------------------------------
# revospeech.catalog — get_catalog_repo from config file
# ---------------------------------------------------------------------------
def test_get_catalog_repo_from_config_file(tmp_path: Path, monkeypatch):
    """get_catalog_repo returns value from config when env var not set."""
    monkeypatch.delenv("REVOS_CATALOG_REPO", raising=False)

    config_file = tmp_path / "config.yaml"
    config_file.write_text("catalog_repo: custom-org/custom-repo\n")

    import revospeech.config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_FILE", config_file)

    import importlib

    import revospeech.catalog as catalog_mod

    importlib.reload(catalog_mod)
    try:
        assert catalog_mod.get_catalog_repo() == "custom-org/custom-repo"
    finally:
        importlib.reload(catalog_mod)


def test_get_catalog_repo_falls_back_to_default(tmp_path: Path, monkeypatch):
    """get_catalog_repo returns DEFAULT_CATALOG_REPO when nothing is set."""
    monkeypatch.delenv("REVOS_CATALOG_REPO", raising=False)

    config_file = tmp_path / "config.yaml"  # does not exist

    import revospeech.config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_FILE", config_file)

    import importlib

    import revospeech.catalog as catalog_mod

    importlib.reload(catalog_mod)
    try:
        assert catalog_mod.get_catalog_repo() == catalog_mod.DEFAULT_CATALOG_REPO
    finally:
        importlib.reload(catalog_mod)


# ---------------------------------------------------------------------------
# revospeech.registry.downloader — _progress_hook branches
# ---------------------------------------------------------------------------
def test_progress_hook_with_total_size(capsys):
    """_progress_hook renders the percentage bar when total_size > 0."""
    from revospeech.registry.downloader import _progress_hook

    # block_num=5, block_size=1MB, total=5MB → 100%
    _progress_hook(5, 1024 * 1024, 1024 * 1024 * 5)
    captured = capsys.readouterr()
    assert "100%" in captured.err
    assert captured.err.endswith("\n")


def test_progress_hook_unknown_total_size(capsys):
    """_progress_hook falls back to MB counter when total_size == 0."""
    from revospeech.registry.downloader import _progress_hook

    _progress_hook(2, 1024 * 1024, 0)
    captured = capsys.readouterr()
    assert "Downloading" in captured.err
    assert "MB" in captured.err
