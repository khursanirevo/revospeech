"""Tests for the extension protocol (subclassing BaseASR/BaseTTS).

Verifies that:
- Subclasses of BaseASR/BaseTTS instantiate correctly
- The factory dispatches based on manifest.backend
- Abstract methods must be implemented
- Invalid subclasses (missing abstract methods) raise clear errors
- Registry rejects malformed manifests
- Duplicate registration is idempotent
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Abstract method enforcement
# ---------------------------------------------------------------------------


def test_base_asr_requires_transcribe_implementation():
    """A class that doesn't implement transcribe() can't instantiate."""
    from revospeech.asr.base import BaseASR

    class IncompleteASR(BaseASR):
        pass  # missing transcribe()

    with pytest.raises(TypeError, match="abstract"):
        IncompleteASR("test")


def test_base_tts_requires_synthesize_implementation():
    """A class that doesn't implement synthesize() can't instantiate."""
    from revospeech.tts.base import BaseTTS

    class IncompleteTTS(BaseTTS):
        pass  # missing synthesize()

    with pytest.raises(TypeError, match="abstract"):
        IncompleteTTS("test")


# ---------------------------------------------------------------------------
# Complete subclasses behave correctly
# ---------------------------------------------------------------------------


def test_complete_asr_subclass_instantiates():
    from revospeech.asr.base import BaseASR
    from revospeech.asr.result import Transcript

    class MyASR(BaseASR):
        def transcribe(self, audio_path, **kwargs):
            return Transcript(text="hello", segments=[], language="en")

    engine = MyASR("test-model")
    assert engine.model_name == "test-model"
    assert engine.device == "auto"

    result = engine.transcribe("/dev/null")
    assert result.text == "hello"


def test_complete_tts_subclass_instantiates():
    from revospeech.tts.base import BaseTTS
    from revospeech.tts.result import Audio

    class MyTTS(BaseTTS):
        def synthesize(self, text, output_path=None, **kwargs):
            return Audio(samples=np.zeros(1000, dtype=np.float32), sample_rate=22050)

    engine = MyTTS("test-tts")
    assert engine.model_name == "test-tts"

    audio = engine.synthesize("hello")
    assert len(audio.samples) == 1000


def test_subclass_preserves_model_name_and_device():
    from revospeech.asr.base import BaseASR
    from revospeech.asr.result import Transcript

    class CustomASR(BaseASR):
        def transcribe(self, *args, **kwargs):
            return Transcript(text="", segments=[], language="en")

    engine = CustomASR("my-model", device="cpu")
    assert engine.model_name == "my-model"
    assert engine.device == "cpu"


# ---------------------------------------------------------------------------
# Factory dispatch with unknown backends
# ---------------------------------------------------------------------------


def _make_manifest(**overrides):
    """Build a ModelManifest with sensible defaults for tests."""
    from revospeech.registry.manifest import ModelManifest

    defaults = {
        "name": "test-model",
        "task": "asr",
        "backend": "sherpa-onnx",
        "model_type": "transducer",
        "model_url": "",
        "sample_rate": 16000,
        "language": "en",
        "description": "test model",
        "files": {},
    }
    defaults.update(overrides)
    return ModelManifest(**defaults)


def test_asr_factory_rejects_unknown_backend():
    """ASR factory should raise ValueError for unsupported backends."""
    from revospeech.registry.registry import _models, register

    manifest = _make_manifest(
        name="test-custom-asr",
        backend="custom-test-backend",
    )

    original = dict(_models)
    try:
        register(manifest)
        from revospeech.asr import ASR

        with pytest.raises(ValueError, match="Unsupported ASR backend"):
            ASR("test-custom-asr", auto_download=False)
    finally:
        _models.clear()
        _models.update(original)


def test_tts_factory_rejects_unknown_backend():
    """TTS factory should raise ValueError for unsupported backends."""
    from revospeech.registry.registry import _models, register

    manifest = _make_manifest(
        name="test-bad-tts",
        task="tts",
        backend="nonexistent-backend",
        model_type="vits",
        sample_rate=22050,
    )

    original = dict(_models)
    try:
        register(manifest)
        from revospeech.tts import TTS

        with pytest.raises(ValueError, match="Unsupported TTS backend"):
            TTS("test-bad-tts", auto_download=False)
    finally:
        _models.clear()
        _models.update(original)


# ---------------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------------


def test_register_requires_non_empty_name():
    """Register with empty name should fail."""
    from revospeech.exceptions import RevosModelError
    from revospeech.registry.registry import register

    bad_manifest = MagicMock()
    bad_manifest.name = ""
    bad_manifest.task = "asr"
    bad_manifest.backend = "sherpa-onnx"
    bad_manifest.model_type = "transducer"

    with pytest.raises(RevosModelError, match="missing required fields"):
        register(bad_manifest)


def test_register_with_empty_backend_fails():
    from revospeech.exceptions import RevosModelError
    from revospeech.registry.registry import register

    bad_manifest = MagicMock()
    bad_manifest.name = "test"
    bad_manifest.task = "asr"
    bad_manifest.backend = ""
    bad_manifest.model_type = "transducer"

    with pytest.raises(RevosModelError, match="missing required fields"):
        register(bad_manifest)


def test_register_with_empty_model_type_fails():
    from revospeech.exceptions import RevosModelError
    from revospeech.registry.registry import register

    bad_manifest = MagicMock()
    bad_manifest.name = "test"
    bad_manifest.task = "asr"
    bad_manifest.backend = "sherpa-onnx"
    bad_manifest.model_type = ""

    with pytest.raises(RevosModelError, match="missing required fields"):
        register(bad_manifest)


def test_register_duplicate_is_idempotent():
    """Registering the same name/task twice should silently replace."""
    from revospeech.registry.registry import _models, register

    m1 = _make_manifest(name="dup-test", backend="x", model_type="y")
    m2 = _make_manifest(name="dup-test", backend="x", model_type="y")

    original = dict(_models)
    try:
        register(m1)
        # Second register should not crash — replaces the first entry.
        register(m2)
        # _models is keyed by (task, name)
        assert ("asr", "dup-test") in _models
    finally:
        _models.clear()
        _models.update(original)


def test_registered_manifest_is_retrievable_via_get():
    """After register(), get(name, task) returns the manifest."""
    from revospeech.registry.registry import _models, get, register

    manifest = _make_manifest(name="retrievable-test")

    original = dict(_models)
    try:
        register(manifest)
        retrieved = get("retrievable-test", "asr")
        assert retrieved is manifest
        assert retrieved.name == "retrievable-test"
    finally:
        _models.clear()
        _models.update(original)


def test_get_unknown_model_raises_keyerror():
    from revospeech.registry.registry import get

    with pytest.raises(KeyError, match="not found"):
        get("this-model-does-not-exist-xyz", "asr")
