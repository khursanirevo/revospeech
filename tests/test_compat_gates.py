"""Backward-compatibility gate tests. These MUST pass after every change."""

from pathlib import Path


class TestManifestCompat:
    """Existing YAML manifests must load without error."""

    def test_load_asr_manifest(self):
        from revospeech.registry.manifest import load_manifest

        manifest = load_manifest(Path("revospeech/models/asr/zipformer_v2.yaml"))
        assert manifest.name == "zipformer-v2"
        assert manifest.task == "asr"
        assert manifest.backend == "sherpa-onnx"
        assert manifest.mode == "local"  # new field, should default
        assert manifest.is_local is True
        assert manifest.is_api is False

    def test_load_tts_manifest(self):
        from revospeech.registry.manifest import load_manifest

        manifest = load_manifest(Path("revospeech/models/tts/revovoice.yaml"))
        assert manifest.name == "revovoice"
        assert manifest.task == "tts"
        assert manifest.backend == "revovoice"
        assert manifest.mode == "local"
        assert manifest.is_local is True

    def test_new_fields_have_defaults(self):
        from revospeech.registry.manifest import load_manifest

        manifest = load_manifest(Path("revospeech/models/asr/zipformer_v2.yaml"))
        # All new fields should be present and correct type
        assert isinstance(manifest.capabilities, list)
        assert isinstance(manifest.languages, list)
        assert isinstance(manifest.tags, list)
        assert isinstance(manifest.size_mb, float)
        assert isinstance(manifest.license, str)
        assert isinstance(manifest.sha256, str)
        assert isinstance(manifest.min_ram_mb, int)
        assert isinstance(manifest.min_vram_mb, int)


class TestImportChain:
    """Core import chain must work without errors."""

    def test_import_revos(self):
        import revospeech

        assert hasattr(revospeech, "__version__")

    def test_import_asr_module(self):
        from revospeech.asr import ASR

        assert callable(ASR)

    def test_import_tts_module(self):
        from revospeech.tts import TTS

        assert callable(TTS)

    def test_import_exceptions(self):
        from revospeech.exceptions import (
            RevosAudioError,
            RevosConfigError,
            RevosEngineError,
            RevosError,
            RevosModelError,
        )

        assert issubclass(RevosConfigError, RevosError)
        assert issubclass(RevosModelError, RevosError)
        assert issubclass(RevosEngineError, RevosError)
        assert issubclass(RevosAudioError, RevosError)

    def test_import_config(self):
        from revospeech.config import get_api_key, set_api_key

        assert callable(get_api_key)
        assert callable(set_api_key)


class TestExceptionHierarchy:
    """Exceptions must have proper structure."""

    def test_exception_with_suggestion(self):
        from revospeech.exceptions import RevosConfigError

        e = RevosConfigError(
            "API key missing",
            suggestion="Run: revospeech config set-api-key",
        )
        assert "API key missing" in str(e)
        assert "revospeech config set-api-key" in str(e)

    def test_exception_without_suggestion(self):
        from revospeech.exceptions import RevosEngineError

        e = RevosEngineError("Inference failed")
        assert "Inference failed" in str(e)
        assert "Suggestion" not in str(e)
