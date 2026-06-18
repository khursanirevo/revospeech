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


# ---------------------------------------------------------------------------
# VitsTTS internals — _phonemize_espeak, _phonemes_to_ids, helpers
# ---------------------------------------------------------------------------
def test_phonemize_espeak_real_subprocess(monkeypatch):
    """_phonemize_espeak parses espeak-ng IPA output into phoneme lists."""
    from revospeech.tts.vits_engine import _phonemize_espeak

    fake_completed = MagicMock()
    fake_completed.stdout = "s a l a m\na p a   k a b a r\n"

    def fake_run(*args, **kwargs):
        return fake_completed

    monkeypatch.setattr("revospeech.tts.vits_engine.subprocess.run", fake_run)
    result = _phonemize_espeak("Salam apa khabar", "ms")
    assert result == [
        ["s", " ", "a", " ", "l", " ", "a", " ", "m"],
        [
            "a",
            " ",
            "p",
            " ",
            "a",
            " ",
            " ",
            " ",
            "k",
            " ",
            "a",
            " ",
            "b",
            " ",
            "a",
            " ",
            "r",
        ],
    ]


def test_phonemize_espeak_missing_binary(monkeypatch):
    """_phonemize_espeak raises RuntimeError when espeak-ng is missing."""
    from revospeech.tts.vits_engine import _phonemize_espeak

    def boom(*args, **kwargs):
        raise FileNotFoundError("espeak-ng not in PATH")

    monkeypatch.setattr("revospeech.tts.vits_engine.subprocess.run", boom)
    with pytest.raises(RuntimeError, match="espeak-ng not found"):
        _phonemize_espeak("hello")


def test_phonemize_espeak_strips_tie_bar(monkeypatch):
    """Tie bar characters are filtered out of phoneme output."""
    from revospeech.tts.vits_engine import _phonemize_espeak

    fake_completed = MagicMock()
    # "a‍b" — a-b with tie bar U+200D between
    fake_completed.stdout = "a‍b\n"
    monkeypatch.setattr(
        "revospeech.tts.vits_engine.subprocess.run",
        lambda *a, **k: fake_completed,
    )
    result = _phonemize_espeak("ab")
    assert result == [["a", "b"]]


def test_phonemize_espeak_skips_blank_lines(monkeypatch):
    """Blank lines in espeak output are skipped."""
    from revospeech.tts.vits_engine import _phonemize_espeak

    fake_completed = MagicMock()
    fake_completed.stdout = "\n\ns a l a m\n\n"
    monkeypatch.setattr(
        "revospeech.tts.vits_engine.subprocess.run",
        lambda *a, **k: fake_completed,
    )
    result = _phonemize_espeak("salam")
    assert result == [["s", " ", "a", " ", "l", " ", "a", " ", "m"]]


def test_phonemes_to_ids_basic():
    """_phonemes_to_ids emits BOS, phoneme IDs with PAD separators, EOS."""
    from revospeech.tts.vits_engine import _phonemes_to_ids

    id_map = {"^": [1], "$": [2], "_": [0], "a": [3], "b": [4]}
    ids = _phonemes_to_ids(["a", "b"], id_map)
    assert ids == [1, 3, 0, 4, 0, 2]


def test_phonemes_to_ids_missing_phoneme_skipped(caplog):
    """Unknown phonemes are skipped with a warning."""
    from revospeech.tts.vits_engine import _phonemes_to_ids

    id_map = {"^": [1], "$": [2], "_": [0], "a": [3]}
    with caplog.at_level("WARNING", logger="revospeech.tts.vits_engine"):
        ids = _phonemes_to_ids(["a", "z"], id_map)
    assert ids == [1, 3, 0, 2]
    assert any("Missing phoneme" in r.message for r in caplog.records)


def test_phonemes_to_ids_uses_default_bos_eos_when_missing():
    """Missing BOS/EOS in id_map fall back to [1] / [2]."""
    from revospeech.tts.vits_engine import _phonemes_to_ids

    id_map = {"_": [0], "a": [3]}
    ids = _phonemes_to_ids(["a"], id_map)
    assert ids == [1, 3, 0, 2]


def test_normalize_text_simple_with_revo_norm(monkeypatch):
    """_normalize_text_simple uses revo_norm when available."""
    from revospeech.tts import vits_engine

    fake_module = MagicMock()
    fake_module.normalize_text.return_value = "normalized"

    monkeypatch.setitem(sys.modules, "revo_norm", fake_module)
    assert vits_engine._normalize_text_simple("Salam") == "normalized"
    fake_module.normalize_text.assert_called_once_with("Salam", language="ms")


def test_normalize_text_simple_without_revo_norm(monkeypatch, caplog):
    """_normalize_text_simple falls back when revo_norm missing."""
    from revospeech.tts import vits_engine

    monkeypatch.setitem(sys.modules, "revo_norm", None)
    with caplog.at_level("WARNING", logger="revospeech.tts.vits_engine"):
        result = vits_engine._normalize_text_simple("Salam")
    assert result == "Salam"
    assert any("revo_norm not installed" in r.message for r in caplog.records)


def test_audio_float_to_int16_clipping():
    """_audio_float_to_int16 scales and clips into int16 range."""
    from revospeech.tts.vits_engine import _audio_float_to_int16

    samples = np.array([0.5, -0.5, 2.0, -2.0], dtype=np.float32)
    out = _audio_float_to_int16(samples)
    assert out.dtype == np.int16
    assert out.max() <= 32767
    assert out.min() >= -32768
    assert out.shape == samples.shape


def test_audio_float_to_int16_silent_input():
    """Silent input does not divide by zero."""
    from revospeech.tts.vits_engine import _audio_float_to_int16

    samples = np.zeros(4, dtype=np.float32)
    out = _audio_float_to_int16(samples)
    assert np.all(out == 0)


def test_vits_list_voices():
    """list_voices returns the production speakers list."""
    _register_vits()
    from revospeech.tts.vits_engine import PRODUCTION_SPEAKERS, VitsTTS

    engine = VitsTTS("test-vits")
    assert engine.list_voices() == list(PRODUCTION_SPEAKERS)


# ---------------------------------------------------------------------------
# VitsTTS._ensure_repo, _get_fallback_phoneme_map, _load_speaker, synthesize
# ---------------------------------------------------------------------------
def test_vits_ensure_repo_uses_existing_speakers_json(tmp_path, monkeypatch):
    """_ensure_repo short-circuits when speakers.json already exists."""
    _register_vits()
    monkeypatch.setattr("revospeech.hf_utils.get_hf_user", lambda *a, **kw: None)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

    cache_dir = tmp_path / "home" / ".cache" / "revospeech" / "test-vits"
    cache_dir.mkdir(parents=True)
    (cache_dir / "speakers.json").write_text("{}")

    from revospeech.tts.vits_engine import VitsTTS

    engine = VitsTTS("test-vits")
    result = engine._ensure_repo()
    assert result == cache_dir


def test_vits_ensure_repo_downloads_when_missing(tmp_path, monkeypatch):
    """_ensure_repo invokes download_gated_model when speakers.json absent."""
    _register_vits()
    monkeypatch.setattr("revospeech.hf_utils.get_hf_user", lambda *a, **kw: None)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

    captured = []

    def fake_download(url, dest):
        captured.append((url, str(dest)))
        dest_path = tmp_path / "home" / ".cache" / "revospeech" / "test-vits"
        dest_path.mkdir(parents=True, exist_ok=True)
        (dest_path / "speakers.json").write_text("{}")

    monkeypatch.setattr("revospeech.hf_utils.download_gated_model", fake_download)

    from revospeech.tts.vits_engine import VitsTTS

    engine = VitsTTS("test-vits")
    result = engine._ensure_repo()
    assert captured == [("TestOrg/vits-test", str(result))]


def test_vits_get_fallback_phoneme_map(tmp_path, monkeypatch):
    """_get_fallback_phoneme_map reads first available speaker config."""
    _register_vits()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

    cache_dir = tmp_path / "home" / ".cache" / "revospeech" / "test-vits"
    speakers_dir = cache_dir / "speakers" / "sarah"
    speakers_dir.mkdir(parents=True)
    (speakers_dir / "model.onnx.json").write_text(
        '{"phoneme_id_map": {"a": [1], "b": [2]}}'
    )

    from revospeech.tts.vits_engine import VitsTTS

    engine = VitsTTS("test-vits")
    engine._models_dir = cache_dir
    result = engine._get_fallback_phoneme_map(cache_dir)
    assert result == {"a": [1], "b": [2]}


def test_vits_get_fallback_phoneme_map_caches(tmp_path, monkeypatch):
    """_get_fallback_phoneme_map does not cache when no speaker config found."""
    _register_vits()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

    cache_dir = tmp_path / "home" / ".cache" / "revospeech" / "test-vits"
    cache_dir.mkdir(parents=True)

    from revospeech.tts.vits_engine import VitsTTS

    engine = VitsTTS("test-vits")
    engine._models_dir = cache_dir
    # No speakers dir → returns {} (without caching, since no raw map was found)
    first = engine._get_fallback_phoneme_map(cache_dir)
    assert first == {}
    assert engine._fallback_phoneme_map is None


def test_vits_load_speaker_file_not_found(tmp_path, monkeypatch):
    """_load_speaker raises FileNotFoundError when model.onnx is missing."""
    _register_vits()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

    cache_dir = tmp_path / "home" / ".cache" / "revospeech" / "test-vits"
    speakers_dir = cache_dir / "speakers" / "sarah"
    speakers_dir.mkdir(parents=True)
    # Create config but no model.onnx
    (speakers_dir / "model.onnx.json").write_text('{"inference": {}}')

    from revospeech.tts.vits_engine import VitsTTS

    engine = VitsTTS("test-vits")
    engine._models_dir = cache_dir
    with pytest.raises(FileNotFoundError, match="Speaker model not found"):
        engine._load_speaker("sarah")


@patch("revospeech.tts.vits_engine.ort.InferenceSession")
def test_vits_load_speaker_success(mock_session_cls, tmp_path, monkeypatch):
    """_load_speaker loads ONNX session and phoneme map from speaker dir."""
    _register_vits()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

    cache_dir = tmp_path / "home" / ".cache" / "revospeech" / "test-vits"
    speakers_dir = cache_dir / "speakers" / "sarah"
    speakers_dir.mkdir(parents=True)
    (speakers_dir / "model.onnx").write_bytes(b"fake-onnx")
    (speakers_dir / "model.onnx.json").write_text(
        '{"inference": {"noise_scale": 0.5}, '
        '"phoneme_id_map": {"a": [1], "^": [10], "$": [20], "_": [0]}}'
    )

    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    from revospeech.tts.vits_engine import VitsTTS

    engine = VitsTTS("test-vits")
    engine._models_dir = cache_dir
    sess, pmap, config = engine._load_speaker("sarah")
    assert sess is mock_session
    assert pmap == {"a": [1], "^": [10], "$": [20], "_": [0]}
    assert config["inference"]["noise_scale"] == 0.5

    # Calling again returns cached result.
    sess2, _, _ = engine._load_speaker("sarah")
    assert sess2 is mock_session
    mock_session_cls.assert_called_once()


@patch("revospeech.tts.vits_engine.ort.InferenceSession")
def test_vits_load_speaker_uses_fallback_map(mock_session_cls, tmp_path, monkeypatch):
    """_load_speaker falls back to sibling phoneme map when config lacks one."""
    _register_vits()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

    cache_dir = tmp_path / "home" / ".cache" / "revospeech" / "test-vits"

    # Target speaker has no phoneme_id_map.
    target_dir = cache_dir / "speakers" / "sarah"
    target_dir.mkdir(parents=True)
    (target_dir / "model.onnx").write_bytes(b"x")
    (target_dir / "model.onnx.json").write_text('{"inference": {}}')

    # Sibling speaker has the map.
    other_dir = cache_dir / "speakers" / "paan"
    other_dir.mkdir(parents=True)
    (other_dir / "model.onnx.json").write_text(
        '{"phoneme_id_map": {"^": [1], "$": [2], "a": [3]}}'
    )

    from revospeech.tts.vits_engine import VitsTTS

    engine = VitsTTS("test-vits")
    engine._models_dir = cache_dir
    sess, pmap, _ = engine._load_speaker("sarah")
    assert pmap == {"^": [1], "$": [2], "a": [3]}


@patch("revospeech.tts.vits_engine.ort.InferenceSession")
def test_vits_synthesize_full_pipeline(mock_session_cls, tmp_path, monkeypatch):
    """End-to-end synthesize pipeline with mocked espeak + ONNX session."""
    _register_vits()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

    cache_dir = tmp_path / "home" / ".cache" / "revospeech" / "test-vits"
    speakers_dir = cache_dir / "speakers" / "sarah"
    speakers_dir.mkdir(parents=True)
    (speakers_dir / "model.onnx").write_bytes(b"x")
    (speakers_dir / "model.onnx.json").write_text(
        '{"inference": {"noise_scale": 0.667, "length_scale": 1.0, '
        '"noise_w": 0.8}, "phoneme_id_map": '
        '{"^": [1], "$": [2], "_": [0], "a": [3], "b": [4]}}'
    )

    mock_session = MagicMock()
    mock_session.run.return_value = [np.zeros((1, 1, 8000), dtype=np.float32)]
    mock_session_cls.return_value = mock_session

    fake_completed = MagicMock()
    fake_completed.stdout = "a b\n"
    monkeypatch.setattr(
        "revospeech.tts.vits_engine.subprocess.run",
        lambda *a, **k: fake_completed,
    )
    monkeypatch.setitem(sys.modules, "revo_norm", None)

    from revospeech.tts.vits_engine import VitsTTS

    engine = VitsTTS("test-vits")
    engine._models_dir = cache_dir
    audio = engine.synthesize("ab")
    assert audio.sample_rate == 22050
    assert len(audio.samples) > 0
    mock_session.run.assert_called_once()


@patch("revospeech.tts.vits_engine.ort.InferenceSession")
def test_vits_synthesize_with_sentence_silence(mock_session_cls, tmp_path, monkeypatch):
    """sentence_silence > 0 inserts silence between sentences."""
    _register_vits()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

    cache_dir = tmp_path / "home" / ".cache" / "revospeech" / "test-vits"
    speakers_dir = cache_dir / "speakers" / "sarah"
    speakers_dir.mkdir(parents=True)
    (speakers_dir / "model.onnx").write_bytes(b"x")
    (speakers_dir / "model.onnx.json").write_text(
        '{"inference": {}, "phoneme_id_map": {"^": [1], "$": [2], "_": [0], "a": [3]}}'
    )

    mock_session = MagicMock()
    mock_session.run.return_value = [np.zeros((1, 1, 1000), dtype=np.float32)]
    mock_session_cls.return_value = mock_session

    fake_completed = MagicMock()
    fake_completed.stdout = "a\na\n"  # two sentences
    monkeypatch.setattr(
        "revospeech.tts.vits_engine.subprocess.run",
        lambda *a, **k: fake_completed,
    )
    monkeypatch.setitem(sys.modules, "revo_norm", None)

    from revospeech.tts.vits_engine import VitsTTS

    engine = VitsTTS("test-vits")
    engine._models_dir = cache_dir
    audio = engine.synthesize("a a", sentence_silence=0.5)
    # 2 sentences × (1000 sample chunk + 11025 sample silence gap)
    assert len(audio.samples) == 2 * (1000 + 11025)


@patch("revospeech.tts.vits_engine.ort.InferenceSession")
def test_vits_synthesize_no_sentences_returns_empty(
    mock_session_cls, tmp_path, monkeypatch
):
    """Empty phonemize output returns Audio with zero samples."""
    _register_vits()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

    cache_dir = tmp_path / "home" / ".cache" / "revospeech" / "test-vits"
    speakers_dir = cache_dir / "speakers" / "sarah"
    speakers_dir.mkdir(parents=True)
    (speakers_dir / "model.onnx").write_bytes(b"x")
    (speakers_dir / "model.onnx.json").write_text(
        '{"inference": {}, "phoneme_id_map": {"^": [1], "$": [2]}}'
    )

    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    fake_completed = MagicMock()
    fake_completed.stdout = ""  # no sentences
    monkeypatch.setattr(
        "revospeech.tts.vits_engine.subprocess.run",
        lambda *a, **k: fake_completed,
    )
    monkeypatch.setitem(sys.modules, "revo_norm", None)

    from revospeech.tts.vits_engine import VitsTTS

    engine = VitsTTS("test-vits")
    engine._models_dir = cache_dir
    audio = engine.synthesize("")
    assert len(audio.samples) == 0
    mock_session.run.assert_not_called()


@patch("revospeech.tts.vits_engine.ort.InferenceSession")
def test_vits_synthesize_streaming_yields_per_chunk(
    mock_session_cls, tmp_path, monkeypatch
):
    """synthesize_streaming splits long text and yields one Audio per chunk."""
    _register_vits()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

    cache_dir = tmp_path / "home" / ".cache" / "revospeech" / "test-vits"
    speakers_dir = cache_dir / "speakers" / "sarah"
    speakers_dir.mkdir(parents=True)
    (speakers_dir / "model.onnx").write_bytes(b"x")
    (speakers_dir / "model.onnx.json").write_text(
        '{"inference": {}, "phoneme_id_map": {"^": [1], "$": [2], "_": [0], "a": [3]}}'
    )

    mock_session = MagicMock()
    mock_session.run.return_value = [np.zeros((1, 1, 1000), dtype=np.float32)]
    mock_session_cls.return_value = mock_session

    fake_completed = MagicMock()
    fake_completed.stdout = "a a a\n"
    monkeypatch.setattr(
        "revospeech.tts.vits_engine.subprocess.run",
        lambda *a, **k: fake_completed,
    )
    monkeypatch.setitem(sys.modules, "revo_norm", None)

    from revospeech.tts.vits_engine import VitsTTS

    engine = VitsTTS("test-vits")
    engine._models_dir = cache_dir
    # Long text → multiple chunks via _split_text.
    text = ". ".join("aaa bbb ccc ddd eee fff ggg" for _ in range(30))
    chunks = list(engine.synthesize_streaming(text))
    assert len(chunks) > 1
    assert all(c.sample_rate == 22050 for c in chunks)
