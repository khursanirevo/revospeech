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


# ---------------------------------------------------------------------------
# revospeech.asr — auto-select when no ready models have size_mb
# ---------------------------------------------------------------------------
def test_asr_auto_select_no_size_info(monkeypatch):
    """ASR() with no args picks ready[0] when no model has size_mb."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models
    from revospeech.registry.status import ModelStatus

    fake_statuses = [
        ModelStatus(
            name="no-size-model",
            task="asr",
            mode="local",
            status="ready",
            installed=True,
            size_mb=None,
            capabilities=[],
            languages=["en"],
        )
    ]
    monkeypatch.setattr(
        "revospeech.asr.list_model_statuses", lambda **kw: fake_statuses
    )

    captured = []
    monkeypatch.setattr(
        "revospeech.asr.sherpa_engine.SherpaOnnxASR",
        lambda *a, **kw: captured.append(a[0]) or MagicMock(),
    )

    saved = dict(_models)
    _models.clear()
    _models[("asr", "no-size-model")] = ModelManifest(
        name="no-size-model",
        task="asr",
        backend="sherpa-onnx",
        model_type="transducer",
        model_url="",
        sample_rate=16000,
        language="en",
        description="",
    )
    try:
        from revospeech.asr import ASR

        ASR()
        assert captured == ["no-size-model"]
    finally:
        _models.clear()
        _models.update(saved)


# ---------------------------------------------------------------------------
# revospeech.tts — auto-select when no ready models have size_mb
# ---------------------------------------------------------------------------
def test_tts_auto_select_no_size_info(monkeypatch):
    """TTS() with no args picks ready[0] when no model has size_mb."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models
    from revospeech.registry.status import ModelStatus

    fake_statuses = [
        ModelStatus(
            name="no-size-tts",
            task="tts",
            mode="local",
            status="ready",
            installed=True,
            size_mb=None,
            capabilities=[],
            languages=["en"],
        )
    ]
    monkeypatch.setattr(
        "revospeech.tts.list_model_statuses", lambda **kw: fake_statuses
    )

    captured = []
    monkeypatch.setattr(
        "revospeech.tts.vits_engine.VitsTTS",
        lambda *a, **kw: captured.append(a[0]) or MagicMock(),
    )

    saved = dict(_models)
    _models.clear()
    _models[("tts", "no-size-tts")] = ModelManifest(
        name="no-size-tts",
        task="tts",
        backend="vits",
        model_type="vits",
        model_url="",
        sample_rate=22050,
        language="en",
        description="",
    )
    try:
        from revospeech.tts import TTS

        TTS()
        assert captured == ["no-size-tts"]
    finally:
        _models.clear()
        _models.update(saved)


# ---------------------------------------------------------------------------
# revospeech.asr.audio — OSError during seek is swallowed
# ---------------------------------------------------------------------------
def test_read_samples_swallows_seek_oserror():
    """_read_samples tolerates seek() raising OSError on file-like inputs."""
    from revospeech.asr.audio import _read_samples

    class FakeStream:
        def seek(self, offset):
            raise OSError("cannot seek")

        def read(self):
            return b""

    # soundfile.read will likely fail on an empty stream, but the seek
    # OSError should be swallowed before that. We just verify no OSError
    # propagates from seek().
    stream = FakeStream()
    try:
        _read_samples(stream)
    except OSError as e:
        if "cannot seek" in str(e):
            pytest.fail(f"seek OSError was not swallowed: {e}")
    except Exception:
        # Any non-OSError exception (e.g. soundfile decode error) is fine.
        pass


# ---------------------------------------------------------------------------
# revospeech.tts.result — Audio.play() with mocked sounddevice
# ---------------------------------------------------------------------------
def test_audio_play_with_sounddevice():
    """Audio.play invokes sounddevice.play when the package is importable."""
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock

    import numpy as np

    fake_sd = ModuleType("sounddevice")
    mock_play = MagicMock()
    fake_sd.play = mock_play
    with __import__("contextlib").suppress():
        # Replace any real sounddevice with the fake.
        saved = sys.modules.get("sounddevice")
        sys.modules["sounddevice"] = fake_sd
        try:
            from revospeech.tts.result import Audio

            samples = np.zeros(100, dtype=np.float32)
            audio = Audio(samples=samples, sample_rate=22050)
            audio.play()
            mock_play.assert_called_once()
            call_kwargs = mock_play.call_args
            assert call_kwargs.kwargs["blocking"] is True
        finally:
            if saved is not None:
                sys.modules["sounddevice"] = saved
            else:
                del sys.modules["sounddevice"]


def test_audio_play_raises_when_sounddevice_missing(monkeypatch):
    """Audio.play raises ImportError with install hint when pkg missing."""
    import sys

    import numpy as np

    # Force ImportError on `import sounddevice`.
    monkeypatch.setitem(sys.modules, "sounddevice", None)

    from revospeech.tts.result import Audio

    samples = np.zeros(100, dtype=np.float32)
    audio = Audio(samples=samples, sample_rate=22050)
    with pytest.raises(ImportError, match="sounddevice"):
        audio.play()


# ---------------------------------------------------------------------------
# revospeech.http_client — _request retry exhaustion
# ---------------------------------------------------------------------------
def test_revolab_client_request_retry_exhaustion():
    """_request raises RevosEngineError after all retries."""
    from revospeech.exceptions import RevosEngineError
    from revospeech.http_client import RevolabClient

    client = RevolabClient(
        endpoint="https://api.example.com/v1",
        api_key="rv-test12345",
        max_retries=2,
    )

    fake_httpx = MagicMock()
    network_err = type("HTTPError", (Exception,), {})
    fake_httpx.HTTPError = network_err
    client._httpx = fake_httpx

    fake_inner_client = MagicMock()
    fake_inner_client.request.side_effect = network_err("boom")
    client._client = fake_inner_client

    with pytest.raises(RevosEngineError, match="Network error"):
        client.post("/any-path")


def test_revolab_client_get_raw_zero_retries():
    """get_raw with max_retries=0 skips the loop and raises immediately."""
    from revospeech.exceptions import RevosEngineError
    from revospeech.http_client import RevolabClient

    client = RevolabClient(
        endpoint="https://api.example.com/v1",
        api_key="rv-test12345",
        max_retries=0,
    )

    fake_httpx = MagicMock()
    fake_httpx.HTTPError = type("HTTPError", (Exception,), {})
    client._httpx = fake_httpx
    client._client = MagicMock()

    with pytest.raises(RevosEngineError, match="Request failed after 0 retries"):
        client.get_raw("https://example.com/file")


def test_revolab_client_request_zero_retries():
    """_request with max_retries=0 skips the loop and raises immediately."""
    from revospeech.exceptions import RevosEngineError
    from revospeech.http_client import RevolabClient

    client = RevolabClient(
        endpoint="https://api.example.com/v1",
        api_key="rv-test12345",
        max_retries=0,
    )

    fake_httpx = MagicMock()
    fake_httpx.HTTPError = type("HTTPError", (Exception,), {})
    client._httpx = fake_httpx
    client._client = MagicMock()

    with pytest.raises(RevosEngineError, match="Request failed after 0 retries"):
        client.post("/any-path")


# ---------------------------------------------------------------------------
# revospeech.registry.downloader — tqdm hook via _download
# ---------------------------------------------------------------------------
def test_download_uses_tqdm_progress_hook(monkeypatch, tmp_path):
    """_download installs the tqdm hook when tqdm is available."""
    from revospeech.registry import downloader

    captured = {}

    class FakePbar:
        def __init__(self, *a, **kw):
            captured["created"] = True

        def update(self, n):
            captured.setdefault("updates", []).append(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            captured["closed"] = True

    fake_tqdm = MagicMock()
    fake_tqdm.tqdm = MagicMock(return_value=FakePbar())

    monkeypatch.setitem(__import__("sys").modules, "tqdm", fake_tqdm)

    # Capture urlretrieve hook installation.
    def fake_urlretrieve(url, dest, reporthook=None):
        # Simulate one block.
        if reporthook:
            reporthook(0, 1024, 1024)
            reporthook(1, 1024, 1024)
        Path(dest).write_bytes(b"x")

    monkeypatch.setattr(
        "revospeech.registry.downloader.urllib.request.urlretrieve",
        fake_urlretrieve,
    )

    dest = tmp_path / "out.bin"
    downloader._download("http://example.com/x", dest)
    assert captured.get("created") is True
    assert captured.get("closed") is True


# ---------------------------------------------------------------------------
# revospeech.tts.vits_engine — _get_fallback_phoneme_map continue branch
# ---------------------------------------------------------------------------
def test_vits_get_fallback_skips_missing_speaker_configs(tmp_path):
    """_get_fallback_phoneme_map skips speakers without model.onnx.json."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register
    from revospeech.registry.status import ModelStatus  # noqa: F401

    saved = dict(_models)
    _models.clear()
    register(
        ModelManifest(
            name="fallback-test",
            task="tts",
            backend="vits",
            model_type="vits",
            model_url="",
            sample_rate=22050,
            language="ms",
            description="",
            files={},
        )
    )
    try:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Only the third speaker has a config.
        for spk in ["sarah", "paan"]:
            (cache_dir / "speakers" / spk).mkdir(parents=True)
        target_dir = cache_dir / "speakers" / "anwar"
        target_dir.mkdir(parents=True)
        (target_dir / "model.onnx.json").write_text('{"phoneme_id_map": {"a": [1]}}')

        from revospeech.tts.vits_engine import VitsTTS

        engine = VitsTTS("fallback-test")
        engine._models_dir = cache_dir
        result = engine._get_fallback_phoneme_map(cache_dir)
        assert result == {"a": [1]}
    finally:
        _models.clear()
        _models.update(saved)


# ---------------------------------------------------------------------------
# revospeech.tts.vits_engine — cached fallback phoneme map early return
# ---------------------------------------------------------------------------
def test_vits_get_fallback_returns_cached():
    """_get_fallback_phoneme_map returns the cached value without re-reading disk."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register

    saved = dict(_models)
    _models.clear()
    register(
        ModelManifest(
            name="cached-fb",
            task="tts",
            backend="vits",
            model_type="vits",
            model_url="",
            sample_rate=22050,
            language="ms",
            description="",
            files={},
        )
    )
    try:
        from revospeech.tts.vits_engine import VitsTTS

        engine = VitsTTS("cached-fb")
        cached = {"x": [9]}
        engine._fallback_phoneme_map = cached

        result = engine._get_fallback_phoneme_map(Path("/nonexistent/path"))
        assert result is cached
    finally:
        _models.clear()
        _models.update(saved)


# ---------------------------------------------------------------------------
# revospeech.registry.status — manifest with files dict
# ---------------------------------------------------------------------------
def test_compute_status_with_files_dict_all_present(monkeypatch, tmp_path):
    """Status='ready' when all manifest.files are present on disk."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.status import _compute_status

    monkeypatch.setattr("revospeech.registry.status.CACHE_DIR", tmp_path)
    model_dir = tmp_path / "has-files"
    model_dir.mkdir()
    (model_dir / "encoder.onnx").write_bytes(b"x")
    (model_dir / "tokens.txt").write_text("a\n")

    manifest = ModelManifest(
        name="has-files",
        task="asr",
        backend="sherpa-onnx",
        model_type="transducer",
        model_url="",
        sample_rate=16000,
        language="en",
        description="",
        files={"encoder": "encoder.onnx", "tokens": "tokens.txt"},
    )
    status = _compute_status(manifest)
    assert status.installed is True
    assert status.status == "ready"


def test_compute_status_with_files_dict_some_missing(monkeypatch, tmp_path):
    """Status='needs-download' when any manifest.files entry is missing."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.status import _compute_status

    monkeypatch.setattr("revospeech.registry.status.CACHE_DIR", tmp_path)
    model_dir = tmp_path / "partial-files"
    model_dir.mkdir()
    (model_dir / "encoder.onnx").write_bytes(b"x")

    manifest = ModelManifest(
        name="partial-files",
        task="asr",
        backend="sherpa-onnx",
        model_type="transducer",
        model_url="",
        sample_rate=16000,
        language="en",
        description="",
        files={
            "encoder": "encoder.onnx",
            "tokens": "missing-tokens.txt",
        },
    )
    status = _compute_status(manifest)
    assert status.installed is False
    assert status.status == "needs-download"


# ---------------------------------------------------------------------------
# revospeech.tts.revovoice_engine — torch CPU fallback
# ---------------------------------------------------------------------------
def test_revovoice_engine_device_cpu_when_no_cuda(monkeypatch):
    """RevoVoice resolves device='cpu' when torch.cuda.is_available() is False."""
    import sys
    from types import ModuleType

    fake_torch = ModuleType("torch")
    fake_torch.cuda = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_omnivoice = ModuleType("omnivoice")
    fake_omnivoice.OmniVoice = MagicMock()
    monkeypatch.setitem(sys.modules, "omnivoice", fake_omnivoice)

    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register

    saved = dict(_models)
    _models.clear()
    register(
        ModelManifest(
            name="rv-test",
            task="tts",
            backend="revovoice",
            model_type="diffusion",
            model_url="org/rv",
            sample_rate=24000,
            language="en",
            description="",
            files={},
        )
    )
    try:
        from revospeech.tts.revovoice_engine import RevoVoiceTTS

        engine = RevoVoiceTTS("rv-test", device="auto")
        assert engine.device == "cpu"
    finally:
        _models.clear()
        _models.update(saved)


def test_revovoice_engine_warns_when_hf_user_missing(monkeypatch):
    """RevoVoice logs a warning when HF user cannot be identified."""
    import sys
    from types import ModuleType

    fake_torch = ModuleType("torch")
    fake_torch.cuda = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_omnivoice = ModuleType("omnivoice")
    fake_omnivoice.OmniVoice = MagicMock()
    monkeypatch.setitem(sys.modules, "omnivoice", fake_omnivoice)

    monkeypatch.setattr("revospeech.tts.revovoice_engine.get_hf_user", lambda: None)

    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register

    saved = dict(_models)
    _models.clear()
    register(
        ModelManifest(
            name="rv-no-hf",
            task="tts",
            backend="revovoice",
            model_type="diffusion",
            model_url="org/rv",
            sample_rate=24000,
            language="en",
            description="",
            files={},
        )
    )
    try:
        from revospeech.tts.revovoice_engine import RevoVoiceTTS

        engine = RevoVoiceTTS("rv-no-hf", device="cpu")
        assert engine.hf_user is None
    finally:
        _models.clear()
        _models.update(saved)


# ---------------------------------------------------------------------------
# revospeech.__init__ — version fallback path
# ---------------------------------------------------------------------------
def test_version_fallback_on_metadata_error():
    """__version__ falls back to '0.0.0-dev' when importlib.metadata fails."""
    import importlib
    import importlib.metadata

    original_version = importlib.metadata.version
    import revospeech

    try:
        importlib.metadata.version = lambda name: (_ for _ in ()).throw(
            Exception("metadata missing")
        )
        importlib.reload(revospeech)
        assert revospeech.__version__ == "0.0.0-dev"
    finally:
        importlib.metadata.version = original_version
        importlib.reload(revospeech)


# ---------------------------------------------------------------------------
# revospeech.catalog — _load_cached_catalog edge cases
# ---------------------------------------------------------------------------
def test_load_cached_catalog_repo_mismatch(tmp_path, monkeypatch):
    """Cache returns None when cached repo differs from requested repo."""
    import revospeech.catalog as catalog_mod

    cache_file = tmp_path / "catalog_cache.json"
    cache_file.write_text(
        json.dumps({"repo": "other/repo", "cached_at": 0, "manifests": []})
    )
    monkeypatch.setattr(catalog_mod, "_CACHE_FILE", cache_file)

    assert catalog_mod._load_cached_catalog("this/repo") is None


def test_load_cached_catalog_corrupted(tmp_path, monkeypatch):
    """Cache returns None when JSON is corrupted (JSONDecodeError swallowed)."""
    import revospeech.catalog as catalog_mod

    cache_file = tmp_path / "catalog_cache.json"
    cache_file.write_text("not valid json {{{")
    monkeypatch.setattr(catalog_mod, "_CACHE_FILE", cache_file)

    assert catalog_mod._load_cached_catalog("any/repo") is None


def test_load_cached_catalog_missing_key(tmp_path, monkeypatch):
    """Cache returns None when 'repo' key is absent (KeyError swallowed)."""
    import revospeech.catalog as catalog_mod

    cache_file = tmp_path / "catalog_cache.json"
    cache_file.write_text(json.dumps({"cached_at": 0}))
    monkeypatch.setattr(catalog_mod, "_CACHE_FILE", cache_file)

    assert catalog_mod._load_cached_catalog("any/repo") is None


def test_load_cached_catalog_oserror(tmp_path, monkeypatch):
    """Cache returns None when read raises OSError."""
    import revospeech.catalog as catalog_mod

    cache_file = tmp_path / "catalog_cache.json"
    cache_file.write_text(
        json.dumps({"repo": "any/repo", "cached_at": 0, "manifests": []})
    )

    def boom(*args, **kwargs):
        raise OSError("disk read failed")

    monkeypatch.setattr(catalog_mod, "_CACHE_FILE", cache_file)
    monkeypatch.setattr(Path, "read_text", boom)

    assert catalog_mod._load_cached_catalog("any/repo") is None


# ---------------------------------------------------------------------------
# revospeech.catalog — _urlopen_with_retry retries then succeeds
# ---------------------------------------------------------------------------
def test_urlopen_with_retry_succeeds_after_failure(monkeypatch):
    """_urlopen_with_retry sleeps and retries on transient URLError."""
    import urllib.error

    import revospeech.catalog as catalog_mod

    sleep_calls: list[float] = []
    monkeypatch.setattr(catalog_mod.time, "sleep", lambda s: sleep_calls.append(s))

    call_count = {"n": 0}

    def fake_urlopen(req, timeout):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise urllib.error.URLError("transient")
        resp = MagicMock()
        resp.read.return_value = b"ok"
        return resp

    monkeypatch.setattr(catalog_mod.urllib.request, "urlopen", fake_urlopen)

    data = catalog_mod._urlopen_with_retry("http://example.com", retries=3, backoff=0.0)
    assert data == b"ok"
    assert call_count["n"] == 2
    assert len(sleep_calls) == 1


def test_urlopen_with_retry_exhausts_attempts(monkeypatch):
    """_urlopen_with_retry raises RuntimeError after exhausting retries."""
    import urllib.error

    import revospeech.catalog as catalog_mod

    monkeypatch.setattr(catalog_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        catalog_mod.urllib.request,
        "urlopen",
        lambda req, timeout: (_ for _ in ()).throw(urllib.error.URLError("down")),
    )

    with pytest.raises(RuntimeError, match="Failed to fetch"):
        catalog_mod._urlopen_with_retry("http://example.com", retries=2, backoff=0.0)


# ---------------------------------------------------------------------------
# revospeech.catalog — list_catalog warning paths (cache + fetch)
# ---------------------------------------------------------------------------
def test_list_catalog_skips_broken_cached_entry(monkeypatch, tmp_path):
    """Cached entries that fail to parse are skipped with a warning."""
    import time

    import revospeech.catalog as catalog_mod

    cache_file = tmp_path / "catalog_cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "repo": "test/repo",
                "cached_at": time.time(),
                "manifests": [
                    {"path": "revos/models/asr/bad.yaml", "content": ": : :"},
                ],
            }
        )
    )
    monkeypatch.setattr(catalog_mod, "_CACHE_FILE", cache_file)
    monkeypatch.setenv("REVOS_CATALOG_REPO", "test/repo")

    results = catalog_mod.list_catalog()
    assert results == []


def test_list_catalog_skips_entry_that_fails_to_download(monkeypatch):
    """list_catalog skips entries whose _download_raw raises."""
    import revospeech.catalog as catalog_mod

    monkeypatch.setattr(
        catalog_mod, "_list_yaml_files", lambda repo, path: ["a.yaml", "b.yaml"]
    )

    good_manifest = (
        "name: ok-model\n"
        "task: asr\n"
        "backend: sherpa-onnx\n"
        "model_type: transducer\n"
        "model_url: ''\n"
        "sample_rate: 16000\n"
        "language: en\n"
        "description: ''\n"
        "files: {}\n"
    )

    def fake_download(repo, path):
        if path == "a.yaml":
            raise RuntimeError("network error")
        return good_manifest

    monkeypatch.setattr(catalog_mod, "_download_raw", fake_download)
    monkeypatch.setattr(catalog_mod, "_save_cached_catalog", lambda *a, **kw: None)
    monkeypatch.setenv("REVOS_CATALOG_REPO", "test/repo")

    results = catalog_mod.list_catalog()
    assert len(results) == 1
    assert results[0].name == "ok-model"


# ---------------------------------------------------------------------------
# revospeech.catalog — pull_model error paths
# ---------------------------------------------------------------------------
def test_pull_model_wraps_list_yaml_error(monkeypatch):
    """pull_model raises RuntimeError when _list_yaml_files fails."""
    import revospeech.catalog as catalog_mod

    monkeypatch.setattr(
        catalog_mod,
        "_list_yaml_files",
        lambda repo, path: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setenv("REVOS_CATALOG_REPO", "test/repo")

    with pytest.raises(RuntimeError, match="Cannot fetch catalog"):
        catalog_mod.pull_model("anything")


def test_pull_model_skips_unparseable_yaml(monkeypatch, tmp_path):
    """pull_model tolerates broken YAML entries and continues to the next."""
    import revospeech.catalog as catalog_mod

    monkeypatch.setattr(
        catalog_mod,
        "_list_yaml_files",
        lambda repo, path: ["bad.yaml", "good.yaml"],
    )

    good_manifest = (
        "name: target\n"
        "task: tts\n"
        "backend: revovoice\n"
        "model_type: diffusion\n"
        "model_url: org/m\n"
        "sample_rate: 24000\n"
        "language: en\n"
        "description: ''\n"
        "files: {}\n"
    )

    def fake_download(repo, path):
        if path == "bad.yaml":
            return ": : :"
        return good_manifest

    monkeypatch.setattr(catalog_mod, "_download_raw", fake_download)
    monkeypatch.setattr(catalog_mod, "_USER_MODELS_DIR", tmp_path / "models")
    monkeypatch.setenv("REVOS_CATALOG_REPO", "test/repo")

    result = catalog_mod.pull_model("target")
    assert result.name == "good.yaml"


# ---------------------------------------------------------------------------
# revospeech.tts.__init__ — revovoice backend dispatch
# ---------------------------------------------------------------------------
def test_tts_factory_dispatches_to_revovoice(monkeypatch):
    """TTS() routes to RevoVoiceTTS when backend='revovoice'."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register

    saved = dict(_models)
    _models.clear()
    register(
        ModelManifest(
            name="rv-dispatch",
            task="tts",
            backend="revovoice",
            model_type="diffusion",
            model_url="org/m",
            sample_rate=24000,
            language="en",
            description="",
            files={},
        )
    )
    try:
        captured = []
        monkeypatch.setattr(
            "revospeech.tts.revovoice_engine.RevoVoiceTTS",
            lambda *a, **kw: captured.append(a[0]) or MagicMock(),
        )

        from revospeech.tts import TTS

        TTS("rv-dispatch", auto_download=False)
        assert captured == ["rv-dispatch"]
    finally:
        _models.clear()
        _models.update(saved)


# ---------------------------------------------------------------------------
# revospeech.util — util model factory + manifest integration
# ---------------------------------------------------------------------------
def test_sidon_manifest_loads_with_util_task():
    """Sidon util manifest loads and exposes the right metadata."""
    from revospeech.registry import get
    from revospeech.registry.registry import _load_builtin_manifests, _models

    if not any(m.name == "sidon" and m.task == "util" for m in _models.values()):
        _load_builtin_manifests()

    m = get("sidon", "util")
    assert m.backend == "sidon"
    assert m.task == "util"
    assert "tts-postprocess" in m.tags
    assert m.files["predictor"].endswith("sidon-predictor.onnx")
    assert m.files["vocoder"].endswith("sidon-vocoder.onnx")


def test_util_factory_rejects_unknown_backend():
    """Util() raises ValueError on unsupported backends."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register

    saved = dict(_models)
    _models.clear()
    register(
        ModelManifest(
            name="bogus-util",
            task="util",
            backend="bogus",
            model_type="restoration",
            model_url="",
            sample_rate=48000,
            language="",
            description="",
            files={},
        )
    )
    try:
        from revospeech.util import Util

        with pytest.raises(ValueError, match="Unknown util backend"):
            Util("bogus-util")
    finally:
        _models.clear()
        _models.update(saved)


def test_util_factory_rejects_api_mode():
    """Util() raises NotImplementedError for API-mode manifests."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register

    saved = dict(_models)
    _models.clear()
    register(
        ModelManifest(
            name="api-util",
            task="util",
            backend="sidon",
            model_type="restoration",
            model_url="",
            sample_rate=48000,
            language="",
            description="",
            mode="api",
            api_endpoint="https://example.com",
            files={},
        )
    )
    try:
        from revospeech.util import Util

        with pytest.raises(NotImplementedError, match="API-mode util"):
            Util("api-util")
    finally:
        _models.clear()
        _models.update(saved)


def test_baseutil_requires_restore_method():
    """BaseUtil subclasses must implement restore()."""
    from revospeech.util.base import BaseUtil

    with pytest.raises(TypeError, match="abstract method"):
        BaseUtil("any")  # type: ignore[abstract]


def test_attach_post_processors_safe_with_no_ready(monkeypatch):
    """_attach_post_processors returns engine unchanged when no ready util."""
    import numpy as np

    from revospeech.tts import _attach_post_processors
    from revospeech.tts.base import BaseTTS
    from revospeech.tts.result import Audio

    class FakeEngine(BaseTTS):
        def __init__(self):
            super().__init__("fake")
            self._post_processors = []

        def synthesize(self, text, output_path=None, **kwargs):
            return Audio(samples=np.zeros(16000, dtype="float32"), sample_rate=16000)

    monkeypatch.setattr(
        "revospeech.registry.list_models",
        lambda: [],
    )

    engine = FakeEngine()
    wrapped = _attach_post_processors(engine, "cpu")
    assert wrapped is engine
    assert wrapped._post_processors == []


def test_sidonutil_resample_linear_is_identity_at_same_rate():
    """_resample_linear is a no-op when source and dest rates match."""
    import numpy as np

    from revospeech.util.sidon_engine import _resample_linear

    samples = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
    out = _resample_linear(samples, 16000, 16000)
    assert np.array_equal(out, samples)


def test_bundled_mel_frontend_asset_exists():
    """The bundled mel_frontend.onnx asset is shipped with the package."""
    from revospeech.util.sidon_engine import _bundled_mel_frontend_path

    path = _bundled_mel_frontend_path()
    assert path.exists(), f"Bundled mel_frontend.onnx missing at {path}"
    assert path.stat().st_size > 100_000  # ~1.1 MB


# ---------------------------------------------------------------------------
# revospeech.util — additional coverage: factory happy path, base helpers
# ---------------------------------------------------------------------------
def test_util_factory_dispatches_to_sidon(monkeypatch):
    """Util() routes to SidonUtil when backend='sidon' and status is ready."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register

    saved = dict(_models)
    _models.clear()
    register(
        ModelManifest(
            name="sidon-test",
            task="util",
            backend="sidon",
            model_type="restoration",
            model_url="org/m",
            sample_rate=48000,
            language="",
            description="",
            files={"predictor": "p.onnx", "vocoder": "v.onnx"},
        )
    )
    try:
        captured = {}

        class FakeUtil:
            def __init__(self, name, device="auto", auto_download=True):
                captured["name"] = name
                captured["device"] = device

        import revospeech.util.sidon_engine as sidon_mod

        monkeypatch.setattr(sidon_mod, "SidonUtil", FakeUtil)

        from revospeech.util import Util

        Util("sidon-test", device="cpu", auto_download=False)
        assert captured == {
            "name": "sidon-test",
            "device": "cpu",
        }
    finally:
        _models.clear()
        _models.update(saved)


def test_baseutil_restore_file_roundtrip(tmp_path):
    """BaseUtil.restore_file() reads audio, calls restore, writes output."""
    import numpy as np
    import soundfile as sf

    from revospeech.tts.result import Audio
    from revospeech.util.base import BaseUtil

    class Upper(BaseUtil):
        def restore(self, audio):
            return Audio(samples=audio.samples * 2, sample_rate=audio.sample_rate)

    # Write a test wav
    sr = 16000
    samples = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
    in_path = tmp_path / "in.wav"
    sf.write(str(in_path), samples, sr)

    out_path = tmp_path / "out.wav"
    u = Upper("dummy")
    result = u.restore_file(str(in_path), str(out_path))

    assert result.sample_rate == sr
    assert np.allclose(result.samples, samples * 2, atol=1e-3)
    # Output file should exist with the doubled samples
    out_samples, out_sr = sf.read(str(out_path), dtype="float32")
    assert out_sr == sr
    assert np.allclose(out_samples, samples * 2, atol=1e-3)


def test_attach_post_processors_wraps_when_ready(monkeypatch):
    """_attach_post_processors wraps synthesize() when a util is ready."""
    import numpy as np

    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register
    from revospeech.tts import _attach_post_processors
    from revospeech.tts.base import BaseTTS
    from revospeech.tts.result import Audio

    saved = dict(_models)
    _models.clear()
    register(
        ModelManifest(
            name="sidon",
            task="util",
            backend="sidon",
            model_type="restoration",
            model_url="org/m",
            sample_rate=48000,
            language="",
            description="",
            files={"predictor": "p.onnx", "vocoder": "v.onnx"},
            tags=["tts-postprocess"],
        )
    )

    class FakeTTS(BaseTTS):
        def __init__(self):
            super().__init__("fake")
            self._post_processors = []

        def synthesize(self, text, output_path=None, **kwargs):
            return Audio(samples=np.array([0.5], dtype=np.float32), sample_rate=16000)

    class FakeUtil:
        model_name = "sidon"

        def restore(self, audio):
            return Audio(
                samples=audio.samples * 3,
                sample_rate=audio.sample_rate,
            )

    try:
        # Force status check to return "ready"
        import revospeech.registry.status as status_mod

        class Ready:
            status = "ready"

        monkeypatch.setattr(status_mod, "check_model", lambda *a, **kw: Ready())

        import revospeech.util as util_mod
        from revospeech.util import Util  # noqa: F401

        monkeypatch.setattr(util_mod, "Util", lambda *a, **kw: FakeUtil())

        engine = FakeTTS()
        wrapped = _attach_post_processors(engine, "cpu")
        assert wrapped is engine
        assert len(wrapped._post_processors) == 1

        audio = wrapped.synthesize("hi")
        # Samples should be tripled by FakeUtil.restore
        assert np.allclose(audio.samples, np.array([1.5], dtype=np.float32))
    finally:
        _models.clear()
        _models.update(saved)


def test_sidonutil_init_loads_manifest():
    """SidonUtil constructor fetches the registered manifest."""
    from revospeech.registry.registry import _load_builtin_manifests, _models
    from revospeech.util.sidon_engine import SidonUtil

    if not any(m.name == "sidon" and m.task == "util" for m in _models.values()):
        _load_builtin_manifests()

    util = SidonUtil("sidon", device="cpu")
    assert util.manifest.name == "sidon"
    assert util.manifest.task == "util"
    assert util._predictor_session is None  # not loaded yet


def test_sidonutil_resample_linear_scales_length_proportionally():
    """_resample_linear halves sample count when downsampled 2x."""
    import numpy as np

    from revospeech.util.sidon_engine import _resample_linear

    samples = np.arange(16000, dtype=np.float32)
    out = _resample_linear(samples, 16000, 8000)
    assert len(out) == 8000
