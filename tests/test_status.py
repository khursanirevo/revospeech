"""Tests for revospeech.registry.status module."""

from __future__ import annotations

import pytest

from revospeech.registry.manifest import ModelManifest
from revospeech.registry.registry import _models, register
from revospeech.registry.status import (
    ModelStatus,
    _compute_status,
    check_model,
    list_model_statuses,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    saved = dict(_models)
    _models.clear()
    yield
    _models.clear()
    _models.update(saved)


def _make_manifest(**kw):
    defaults = dict(
        name="test-model",
        task="asr",
        backend="sherpa-onnx",
        model_type="transducer",
        model_url="http://example.com/m.tar.bz2",
        sample_rate=16000,
        language="en",
        description="",
    )
    defaults.update(kw)
    return ModelManifest(**defaults)


# ---------------------------------------------------------------------------
# ModelStatus properties
# ---------------------------------------------------------------------------
def test_model_status_is_ready_true():
    ms = ModelStatus(
        name="x",
        task="asr",
        mode="local",
        status="ready",
        installed=True,
        size_mb=10.0,
        capabilities=[],
        languages=[],
    )
    assert ms.is_ready is True


def test_model_status_is_local_true():
    ms = ModelStatus(
        name="x",
        task="asr",
        mode="local",
        status="ready",
        installed=True,
        size_mb=None,
        capabilities=[],
        languages=[],
    )
    assert ms.is_local is True
    assert ms.is_api is False


def test_model_status_is_api_true():
    ms = ModelStatus(
        name="x",
        task="asr",
        mode="api",
        status="ready",
        installed=True,
        size_mb=None,
        capabilities=[],
        languages=[],
    )
    assert ms.is_api is True
    assert ms.is_local is False


# ---------------------------------------------------------------------------
# _compute_status — API mode
# ---------------------------------------------------------------------------
def test_compute_status_api_mode_with_key(monkeypatch):
    """API manifest + key present → status='ready', installed=True."""
    monkeypatch.setattr(
        "revospeech.registry.status.get_api_key", lambda *a, **kw: "rv-test12345"
    )
    manifest = _make_manifest(mode="api", backend="revolab", size_mb=0.0)
    ms = _compute_status(manifest)
    assert ms.status == "ready"
    assert ms.installed is True
    assert ms.size_mb is None  # size_mb=0 normalizes to None


def test_compute_status_api_mode_no_key(monkeypatch):
    """API manifest + no key → status='needs-api-key'."""
    monkeypatch.setattr(
        "revospeech.registry.status.get_api_key", lambda *a, **kw: None
    )
    manifest = _make_manifest(mode="api", backend="revolab")
    ms = _compute_status(manifest)
    assert ms.status == "needs-api-key"
    assert ms.installed is False


# ---------------------------------------------------------------------------
# _compute_status — local mode, files={} branch
# ---------------------------------------------------------------------------
def test_compute_status_local_no_files_installed(monkeypatch, tmp_path):
    """Local manifest with no files list: status='ready' when dir exists."""
    monkeypatch.setattr("revospeech.registry.status.CACHE_DIR", tmp_path)
    (tmp_path / "test-model").mkdir()
    manifest = _make_manifest(files={})
    ms = _compute_status(manifest)
    assert ms.status == "ready"
    assert ms.installed is True


def test_compute_status_local_no_files_missing(monkeypatch, tmp_path):
    """Local manifest with no files list: 'needs-download' when dir absent."""
    monkeypatch.setattr("revospeech.registry.status.CACHE_DIR", tmp_path)
    manifest = _make_manifest(files={})
    ms = _compute_status(manifest)
    assert ms.status == "needs-download"
    assert ms.installed is False


# ---------------------------------------------------------------------------
# check_model — ambiguous match
# ---------------------------------------------------------------------------
def test_check_model_ambiguous_raises():
    """Same model name registered under two tasks → KeyError with hint."""
    register(_make_manifest(name="dup", task="asr"))
    register(_make_manifest(name="dup", task="tts", backend="vits"))
    with pytest.raises(KeyError, match="ambiguous"):
        check_model("dup")


def test_check_model_no_task_not_found_raises():
    """Unknown model name without task hint → KeyError."""
    with pytest.raises(KeyError, match="not found"):
        check_model("ghost")


# ---------------------------------------------------------------------------
# list_model_statuses — filter combinations
# ---------------------------------------------------------------------------
def test_list_model_statuses_filter_by_mode():
    register(_make_manifest(name="local-m", mode="local"))
    register(_make_manifest(name="api-m", mode="api", backend="revolab"))
    result = list_model_statuses(task="asr", mode="api")
    assert len(result) == 1
    assert result[0].name == "api-m"


def test_list_model_statuses_filter_by_language():
    register(_make_manifest(name="en-m", languages=["en"]))
    register(_make_manifest(name="fr-m", languages=["fr"]))
    result = list_model_statuses(task="asr", language="fr")
    assert len(result) == 1
    assert result[0].name == "fr-m"


def test_list_model_statuses_filter_by_capability():
    register(_make_manifest(name="cap-m", capabilities=["streaming"]))
    register(_make_manifest(name="nocap-m", capabilities=[]))
    result = list_model_statuses(task="asr", capability="streaming")
    assert len(result) == 1
    assert result[0].name == "cap-m"


def test_list_model_statuses_filter_by_status(monkeypatch, tmp_path):
    """Filter by status returns only matching models."""
    monkeypatch.setattr("revospeech.registry.status.CACHE_DIR", tmp_path)
    # 'installed' is ready (dir exists), 'missing' is needs-download
    register(_make_manifest(name="installed", files={}))
    register(_make_manifest(name="missing", files={}))
    (tmp_path / "installed").mkdir()
    ready = list_model_statuses(task="asr", status="ready")
    assert any(m.name == "installed" for m in ready)
    assert all(m.name != "missing" for m in ready)
