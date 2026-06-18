"""Tests for TTS engine."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from revospeech.registry.manifest import ModelManifest
from revospeech.registry.registry import _models
from revospeech.tts.result import Audio


def test_audio_creation():
    samples = np.zeros(16000, dtype=np.float32)
    audio = Audio(samples=samples, sample_rate=16000)
    assert audio.sample_rate == 16000
    assert len(audio.samples) == 16000


def test_audio_save(tmp_path: Path):
    samples = np.random.randn(24000).astype(np.float32) * 0.1
    audio = Audio(samples=samples, sample_rate=24000)
    out_path = str(tmp_path / "test_out.wav")
    audio.save(out_path)

    import soundfile as sf

    data, sr = sf.read(out_path)
    assert sr == 24000
    assert len(data) == 24000


def test_audio_dataclass():
    samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    audio = Audio(samples=samples, sample_rate=16000)
    np.testing.assert_allclose(audio.samples, [0.1, 0.2, 0.3], atol=1e-6)


@pytest.fixture(autouse=True)
def clear_registry():
    _models.clear()
    yield
    _models.clear()


def _make_mock_omnivoice():
    """Create a mock omnivoice module with OmniVoice class."""
    mock_module = ModuleType("omnivoice")
    mock_cls = MagicMock()
    mock_model = MagicMock()
    audio_samples = np.random.randn(24000).astype(np.float32) * 0.1
    mock_model.generate.return_value = [audio_samples]
    mock_cls.from_pretrained.return_value = mock_model
    mock_module.OmniVoice = mock_cls
    return mock_module, mock_cls, mock_model


@patch("revospeech.hf_utils.get_hf_user", return_value=None)
def test_revovoice_engine_synthesize(mock_hf_user, tmp_path: Path):
    """Test RevoVoiceTTS.synthesize with mocked model."""
    from revospeech.registry.registry import register

    register(
        ModelManifest(
            name="test-tts",
            task="tts",
            backend="revovoice",
            model_type="diffusion",
            model_url="TestOrg/test-model",
            sample_rate=24000,
            language="en",
            description="Test TTS",
            files={},
        )
    )

    mock_module, mock_cls, mock_model = _make_mock_omnivoice()
    with patch.dict(sys.modules, {"omnivoice": mock_module}):
        from revospeech.tts.revovoice_engine import RevoVoiceTTS

        engine = RevoVoiceTTS("test-tts", device="cpu")
        result = engine.synthesize("Hello world")

    assert isinstance(result, Audio)
    assert result.sample_rate == 24000
    mock_model.generate.assert_called_once_with(text="Hello world", speed=1.0)


@patch("revospeech.hf_utils.get_hf_user", return_value=None)
def test_revovoice_engine_save_to_file(mock_hf_user, tmp_path: Path):
    """Test RevoVoiceTTS.synthesize saves to file when output_path given."""
    from revospeech.registry.registry import register

    register(
        ModelManifest(
            name="test-tts",
            task="tts",
            backend="revovoice",
            model_type="diffusion",
            model_url="TestOrg/test-model",
            sample_rate=24000,
            language="en",
            description="Test TTS",
            files={},
        )
    )

    mock_module, _, _ = _make_mock_omnivoice()
    with patch.dict(sys.modules, {"omnivoice": mock_module}):
        from revospeech.tts.revovoice_engine import RevoVoiceTTS

        engine = RevoVoiceTTS("test-tts", device="cpu")
        out_path = str(tmp_path / "output.wav")
        engine.synthesize("Hello", output_path=out_path)

    assert (tmp_path / "output.wav").exists()


@patch("revospeech.hf_utils.get_hf_user", return_value=None)
def test_revovoice_engine_gated_error(mock_hf_user):
    """Test that OSError for gated repo raises clear RuntimeError."""
    from revospeech.registry.registry import register

    register(
        ModelManifest(
            name="gated-tts",
            task="tts",
            backend="revovoice",
            model_type="diffusion",
            model_url="Revolab/revovoice",
            sample_rate=24000,
            language="en",
            description="Gated model",
            files={},
        )
    )

    mock_module = ModuleType("omnivoice")
    mock_cls = MagicMock()
    mock_cls.from_pretrained.side_effect = OSError("gated repo, please authenticate")
    mock_module.OmniVoice = mock_cls

    with patch.dict(sys.modules, {"omnivoice": mock_module}):
        from revospeech.tts.revovoice_engine import RevoVoiceTTS

        with pytest.raises(RuntimeError, match="HuggingFace authentication"):
            RevoVoiceTTS("gated-tts", device="cpu")


def _register_vits():
    from revospeech.registry.registry import register

    register(
        ModelManifest(
            name="test-vits",
            task="tts",
            backend="vits",
            model_type="vits",
            model_url="TestOrg/vits-test",
            sample_rate=22050,
            language="ms",
            description="Test VITS TTS",
            files={},
        )
    )


@patch(
    "revospeech.tts.vits_engine._phonemize_espeak",
    return_value=[["s", "a", "l", "a", "m"]],
)
def test_vits_engine_synthesize(mock_phonemize, tmp_path):
    mock_sess = MagicMock()
    mock_sess.run.return_value = [np.zeros((1, 1, 22050), dtype=np.float32)]
    mock_phoneme_map = {"^": [1], "$": [2], "s": [3], "a": [4], "l": [5]}

    _register_vits()
    with patch(
        "revospeech.tts.vits_engine.VitsTTS._load_speaker",
        return_value=(mock_sess, mock_phoneme_map, {}),
    ):
        from revospeech.tts.vits_engine import VitsTTS

        engine = VitsTTS("test-vits")
        result = engine.synthesize("Salam")

    assert isinstance(result, Audio)
    assert result.sample_rate == 22050
    mock_sess.run.assert_called_once()


@patch("revospeech.tts.vits_engine._phonemize_espeak", return_value=[])
def test_vits_engine_save_to_file(mock_phonemize, tmp_path):
    mock_sess = MagicMock()
    mock_sess.run.return_value = [np.zeros((1, 1, 8000), dtype=np.float32)]
    mock_phoneme_map = {"^": [1], "$": [2]}

    _register_vits()
    with patch(
        "revospeech.tts.vits_engine.VitsTTS._load_speaker",
        return_value=(mock_sess, mock_phoneme_map, {}),
    ):
        from revospeech.tts.vits_engine import VitsTTS

        engine = VitsTTS("test-vits")
        out_path = str(tmp_path / "vits_out.wav")
        engine.synthesize("test", output_path=out_path)

    assert (tmp_path / "vits_out.wav").exists()


@patch("revospeech.tts.vits_engine._phonemize_espeak", return_value=["a"])
def test_vits_engine_unknown_speaker(mock_phonemize):
    _register_vits()
    from revospeech.tts.vits_engine import VitsTTS

    engine = VitsTTS("test-vits")
    with pytest.raises(ValueError, match="Unknown speaker"):
        engine.synthesize("test", speaker="nonexistent")


def test_vits_factory_dispatch():
    _register_vits()
    from revospeech.tts import TTS
    from revospeech.tts.vits_engine import VitsTTS

    engine = TTS("test-vits")
    assert isinstance(engine, VitsTTS)


def test_tts_unsupported_backend():
    """Test that unsupported backend raises ValueError."""
    from revospeech.registry.registry import register

    register(
        ModelManifest(
            name="bad-backend",
            task="tts",
            backend="nonexistent",
            model_type="vits",
            model_url="",
            sample_rate=24000,
            language="en",
            description="",
        )
    )
    from revospeech.tts import TTS

    with pytest.raises(ValueError, match="Supported backends: revovoice, vits"):
        TTS("bad-backend")


# ---------------------------------------------------------------------------
# TTS factory: auto-select, API-mode, did-you-mean (covers tts/__init__.py)
# ---------------------------------------------------------------------------
def test_tts_with_api_key_persists(monkeypatch):
    """Passing api_key to TTS() should persist it via set_api_key."""
    captured = []
    monkeypatch.setattr("revospeech.tts.set_api_key", lambda key: captured.append(key))
    _register_vits()
    from revospeech.tts import TTS

    TTS("test-vits", api_key="rv-test12345")
    assert captured == ["rv-test12345"]


def test_tts_auto_select_no_models_registered():
    """No model_name + no registered TTS models → RevosModelError with catalog hint."""
    from revospeech.exceptions import RevosModelError
    from revospeech.tts import TTS

    with pytest.raises(RevosModelError, match="No TTS model specified") as exc:
        TTS()
    assert "catalog list" in exc.value.suggestion


def test_tts_auto_select_no_ready_models_with_available(monkeypatch):
    """No ready models, but some are registered → suggestion lists them."""
    from revospeech.registry.status import ModelStatus
    from revospeech.tts import TTS

    def fake_list(**kw):
        if kw.get("status") == "ready":
            return []
        return [
            ModelStatus(
                name="not-ready-tts",
                task="tts",
                mode="local",
                status="needs-download",
                installed=False,
                size_mb=50.0,
                capabilities=[],
                languages=["en"],
            )
        ]

    monkeypatch.setattr("revospeech.tts.list_model_statuses", fake_list)

    from revospeech.exceptions import RevosModelError

    with pytest.raises(RevosModelError) as exc:
        TTS()
    assert "Download a model first" in exc.value.suggestion
    assert "not-ready-tts" in exc.value.suggestion


def test_tts_auto_select_smallest_ready(monkeypatch):
    """Auto-select picks the smallest ready model by size_mb."""
    from revospeech.registry.registry import register
    from revospeech.registry.status import ModelStatus

    register(
        ModelManifest(
            name="big-tts",
            task="tts",
            backend="vits",
            model_type="vits",
            model_url="",
            sample_rate=22050,
            language="en",
            description="",
            size_mb=200.0,
            files={},
        )
    )
    register(
        ModelManifest(
            name="small-tts",
            task="tts",
            backend="vits",
            model_type="vits",
            model_url="",
            sample_rate=22050,
            language="en",
            description="",
            size_mb=50.0,
            files={},
        )
    )

    fake_statuses = [
        ModelStatus(
            name="big-tts",
            task="tts",
            mode="local",
            status="ready",
            installed=True,
            size_mb=200.0,
            capabilities=[],
            languages=["en"],
        ),
        ModelStatus(
            name="small-tts",
            task="tts",
            mode="local",
            status="ready",
            installed=True,
            size_mb=50.0,
            capabilities=[],
            languages=["en"],
        ),
    ]
    monkeypatch.setattr(
        "revospeech.tts.list_model_statuses",
        lambda **kw: fake_statuses,
    )

    captured = []
    monkeypatch.setattr(
        "revospeech.tts.vits_engine.VitsTTS",
        lambda *a, **kw: captured.append(a[0]) or MagicMock(),
    )

    from revospeech.tts import TTS

    TTS()
    assert captured == ["small-tts"]


def test_tts_unknown_model_with_did_you_mean():
    """Unknown model name triggers KeyError with did-you-mean hint."""
    _register_vits()
    from revospeech.tts import TTS

    with pytest.raises(KeyError, match="not found"):
        TTS("test-vit")


def test_tts_api_mode_no_key_raises_config_error(monkeypatch):
    """API-mode model without API key → RevosConfigError."""
    from revospeech.registry.registry import register

    register(
        ModelManifest(
            name="api-tts",
            task="tts",
            backend="revolab",
            model_type="revolab",
            model_url="",
            sample_rate=22050,
            language="en",
            description="API TTS",
            mode="api",
            files={},
        )
    )
    monkeypatch.setattr("revospeech.config.get_api_key", lambda *a, **kw: None)

    from revospeech.exceptions import RevosConfigError
    from revospeech.tts import TTS

    with pytest.raises(RevosConfigError, match="requires an API key"):
        TTS("api-tts")


def test_tts_api_mode_with_key_raises_not_implemented(monkeypatch):
    """API-mode model with key set → NotImplementedError (engine pending)."""
    from revospeech.registry.registry import register

    register(
        ModelManifest(
            name="api-tts",
            task="tts",
            backend="revolab",
            model_type="revolab",
            model_url="",
            sample_rate=22050,
            language="en",
            description="API TTS",
            mode="api",
            files={},
        )
    )
    monkeypatch.setattr(
        "revospeech.config.get_api_key", lambda *a, **kw: "rv-test12345"
    )

    from revospeech.tts import TTS

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        TTS("api-tts")


# ---------------------------------------------------------------------------
# RevoVoiceTTS: ImportError, device auto, ref_audio, streaming
# ---------------------------------------------------------------------------
def _register_revovoice():
    from revospeech.registry.registry import register

    register(
        ModelManifest(
            name="test-tts",
            task="tts",
            backend="revovoice",
            model_type="diffusion",
            model_url="TestOrg/test-model",
            sample_rate=24000,
            language="en",
            description="Test TTS",
            files={},
        )
    )


@patch("revospeech.hf_utils.get_hf_user", return_value=None)
def test_revovoice_engine_raises_without_omnivoice(mock_hf_user):
    """Missing omnivoice package → ImportError with install hint."""
    _register_revovoice()
    with patch.dict(sys.modules, {"omnivoice": None}):
        from revospeech.tts.revovoice_engine import RevoVoiceTTS

        with pytest.raises(ImportError, match="pip install revos"):
            RevoVoiceTTS("test-tts", device="cpu")


@patch("revospeech.hf_utils.get_hf_user", return_value=None)
def test_revovoice_engine_auto_device_cuda(mock_hf_user, monkeypatch):
    """device='auto' resolves to 'cuda' when CUDA is available."""
    _register_revovoice()
    mock_module, _, _ = _make_mock_omnivoice()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    with (
        patch.dict(sys.modules, {"omnivoice": mock_module, "torch": fake_torch}),
    ):
        from revospeech.tts.revovoice_engine import RevoVoiceTTS

        engine = RevoVoiceTTS("test-tts", device="auto")
    assert engine.device == "cuda"


@patch("revospeech.hf_utils.get_hf_user", return_value=None)
def test_revovoice_engine_auto_device_falls_back_when_no_torch(mock_hf_user):
    """device='auto' falls back to 'cpu' when torch is unavailable."""
    _register_revovoice()
    mock_module, _, _ = _make_mock_omnivoice()
    with patch.dict(sys.modules, {"omnivoice": mock_module, "torch": None}):
        from revospeech.tts.revovoice_engine import RevoVoiceTTS

        engine = RevoVoiceTTS("test-tts", device="auto")
    assert engine.device == "cpu"


@patch("revospeech.hf_utils.get_hf_user", return_value=None)
def test_revovoice_engine_synthesize_with_ref_audio(mock_hf_user, tmp_path: Path):
    """ref_audio and ref_text are forwarded to model.generate."""
    _register_revovoice()
    mock_module, _, mock_model = _make_mock_omnivoice()
    ref = tmp_path / "ref.wav"
    samples = np.zeros(16000, dtype=np.float32)
    import soundfile as sf

    sf.write(str(ref), samples, 16000, format="WAV")
    with patch.dict(sys.modules, {"omnivoice": mock_module}):
        from revospeech.tts.revovoice_engine import RevoVoiceTTS

        engine = RevoVoiceTTS("test-tts", device="cpu")
        engine.synthesize("hello", ref_audio=str(ref), ref_text="reference text")
    expected = {
        "text": "hello",
        "speed": 1.0,
        "ref_audio": str(ref),
        "ref_text": "reference text",
    }
    mock_model.generate.assert_called_once_with(**expected)


@patch("revospeech.hf_utils.get_hf_user", return_value=None)
def test_revovoice_engine_synthesize_non_list_result(mock_hf_user):
    """Non-list generate() output is coerced via np.array directly."""
    _register_revovoice()
    mock_module = ModuleType("omnivoice")
    mock_cls = MagicMock()
    mock_model = MagicMock()
    mock_model.generate.return_value = np.zeros(24000, dtype=np.float32)
    mock_cls.from_pretrained.return_value = mock_model
    mock_module.OmniVoice = mock_cls
    with patch.dict(sys.modules, {"omnivoice": mock_module}):
        from revospeech.tts.revovoice_engine import RevoVoiceTTS

        engine = RevoVoiceTTS("test-tts", device="cpu")
        result = engine.synthesize("hello")
    assert len(result.samples) == 24000


@patch("revospeech.hf_utils.get_hf_user", return_value=None)
def test_revovoice_engine_synthesize_streaming_yields_chunks(mock_hf_user):
    """synthesize_streaming yields one Audio per text chunk."""
    _register_revovoice()
    mock_module, _, _ = _make_mock_omnivoice()
    with patch.dict(sys.modules, {"omnivoice": mock_module}):
        from revospeech.tts.revovoice_engine import RevoVoiceTTS

        engine = RevoVoiceTTS("test-tts", device="cpu")
        text = ". ".join(
            f"This is sentence number {i} with extra words" for i in range(20)
        )
        chunks = list(engine.synthesize_streaming(text))
    assert len(chunks) > 1
    assert all(c.sample_rate == 24000 for c in chunks)
