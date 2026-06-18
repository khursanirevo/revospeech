"""Tests for auto-download and first-run experience."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_asr_auto_download_when_not_ready():
    """ASR() should auto-download when model status is needs-download."""
    from revospeech.registry import list_models

    # Find an ASR model that's local
    manifests = [m for m in list_models() if m.task == "asr" and m.is_local]
    if not manifests:
        pytest.skip("No local ASR model registered")
    manifest = manifests[0]

    with (
        patch("revospeech.asr.check_model") as mock_check,
        patch("revospeech.asr.ensure_model") as mock_ensure,
        patch("revospeech.asr.SherpaOnnxASR", create=True),
    ):
        # Simulate needs-download status
        mock_status = MagicMock()
        mock_status.status = "needs-download"
        mock_check.return_value = mock_status

        from revospeech.asr import ASR

        ASR(manifest.name, auto_download=True)
        mock_ensure.assert_called_once_with(manifest)


def test_asr_skip_download_when_disabled():
    """ASR() with auto_download=False should NOT call ensure_model."""
    from revospeech.registry import list_models

    manifests = [m for m in list_models() if m.task == "asr" and m.is_local]
    if not manifests:
        pytest.skip("No local ASR model registered")
    manifest = manifests[0]

    with (
        patch("revospeech.asr.check_model") as mock_check,
        patch("revospeech.asr.ensure_model") as mock_ensure,
        patch("revospeech.asr.SherpaOnnxASR", create=True),
    ):
        mock_status = MagicMock()
        mock_status.status = "needs-download"
        mock_check.return_value = mock_status

        from revospeech.asr import ASR

        try:
            ASR(manifest.name, auto_download=False)
        except Exception:
            # Engine construction may fail; we only verify
            # ensure_model was not called.
            pass
        mock_ensure.assert_not_called()


def test_tts_auto_download_when_not_ready():
    from revospeech.registry import list_models

    manifests = [m for m in list_models() if m.task == "tts" and m.is_local]
    if not manifests:
        pytest.skip("No local TTS model registered")
    manifest = manifests[0]

    with (
        patch("revospeech.tts.check_model") as mock_check,
        patch("revospeech.tts.ensure_model") as mock_ensure,
        patch("revospeech.tts.RevoVoiceTTS", create=True),
        patch("revospeech.tts.VitsTTS", create=True),
    ):
        mock_status = MagicMock()
        mock_status.status = "needs-download"
        mock_check.return_value = mock_status

        from revospeech.tts import TTS

        try:
            TTS(manifest.name, auto_download=True)
        except Exception:
            pass
        mock_ensure.assert_called_once()


def test_first_run_no_ready_models_suggests_catalog():
    """When no models are ready, suggestion should mention catalog or download."""
    from revospeech.asr import ASR
    from revospeech.exceptions import RevosModelError

    with patch("revospeech.asr.list_model_statuses") as mock_list:
        # First call: ready filter returns []
        # Second call: no filter returns [] too (simulate no models at all)
        mock_list.return_value = []

        with pytest.raises(RevosModelError) as exc_info:
            ASR(None)

        suggestion = exc_info.value.suggestion or ""
        assert "catalog" in suggestion.lower() or "download" in suggestion.lower()


def test_first_run_with_unready_models_lists_names():
    """When models exist but none ready, suggestion should list names."""
    from revospeech.asr import ASR
    from revospeech.exceptions import RevosModelError

    fake_model = MagicMock()
    fake_model.name = "fake-asr-model"

    def mock_list_statuses(**kwargs):
        if kwargs.get("status") == "ready":
            return []
        return [fake_model]

    with patch("revospeech.asr.list_model_statuses", side_effect=mock_list_statuses):
        with pytest.raises(RevosModelError) as exc_info:
            ASR(None)
        assert "fake-asr-model" in (exc_info.value.suggestion or "")
