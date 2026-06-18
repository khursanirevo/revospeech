"""Tests for HuggingFace utility helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from revospeech.hf_utils import get_hf_user, wrap_hf_error


def test_wrap_hf_error_403_returns_access_denied():
    err = wrap_hf_error(Exception("HTTP 403 Forbidden"), "owner/model")
    assert "Access denied" in str(err)
    assert "owner/model" in str(err)


def test_wrap_hf_error_401_returns_auth_required():
    err = wrap_hf_error(Exception("401 Unauthorized"), "owner/model")
    assert "authentication" in str(err).lower() or "requires" in str(err).lower()
    assert "huggingface-cli login" in str(err)


def test_wrap_hf_error_gated_model_returns_auth_required():
    err = wrap_hf_error(Exception("Access to gated model"), "owner/model")
    assert "huggingface-cli login" in str(err) or "huggingface.co" in str(err)


def test_wrap_hf_error_credentials_returns_auth_required():
    err = wrap_hf_error(Exception("Invalid credentials"), "owner/model")
    assert "huggingface-cli login" in str(err)


def test_wrap_hf_error_permission_returns_access_denied():
    err = wrap_hf_error(Exception("permission denied"), "owner/model")
    assert "Access denied" in str(err)


def test_wrap_hf_error_unknown_returns_original():
    original = Exception("Some random network error")
    err = wrap_hf_error(original, "owner/model")
    assert err is original


def test_get_hf_user_returns_name_when_authenticated():
    mock_api = MagicMock()
    mock_api.whoami.return_value = {"name": "testuser"}
    with patch("huggingface_hub.HfApi", return_value=mock_api):
        assert get_hf_user() == "testuser"


def test_get_hf_user_returns_none_on_failure():
    mock_api = MagicMock()
    mock_api.whoami.side_effect = Exception("not logged in")
    with patch("huggingface_hub.HfApi", return_value=mock_api):
        assert get_hf_user() is None


def test_get_hf_user_returns_none_when_not_dict():
    mock_api = MagicMock()
    mock_api.whoami.return_value = "not a dict"
    with patch("huggingface_hub.HfApi", return_value=mock_api):
        assert get_hf_user() is None


# ---------------------------------------------------------------------------
# download_gated_model — happy path + auth error wrapping
# ---------------------------------------------------------------------------
def test_download_gated_model_success(tmp_path):
    """Successful snapshot_download returns the local_dir path."""
    from revospeech.hf_utils import download_gated_model

    with patch("huggingface_hub.snapshot_download") as mock_dl:
        result = download_gated_model("owner/model", tmp_path)
    mock_dl.assert_called_once_with(repo_id="owner/model", local_dir=str(tmp_path))
    assert result == tmp_path


def test_download_gated_model_wraps_auth_error(tmp_path):
    """401 errors from snapshot_download are wrapped as RuntimeError."""
    from revospeech.hf_utils import download_gated_model

    with patch(
        "huggingface_hub.snapshot_download",
        side_effect=Exception("401 unauthorized"),
    ):
        with pytest.raises(RuntimeError, match="huggingface-cli login"):
            download_gated_model("owner/model", tmp_path)


def test_download_gated_model_wraps_permission_error(tmp_path):
    """403 errors from snapshot_download are wrapped as RuntimeError."""
    from revospeech.hf_utils import download_gated_model

    with patch(
        "huggingface_hub.snapshot_download",
        side_effect=Exception("403 Forbidden"),
    ):
        with pytest.raises(RuntimeError, match="Access denied"):
            download_gated_model("owner/model", tmp_path)


def test_download_gated_model_passes_through_unknown_error(tmp_path):
    """Non-auth errors are re-raised unchanged."""
    from revospeech.hf_utils import download_gated_model

    original = ConnectionError("network timeout")
    with patch("huggingface_hub.snapshot_download", side_effect=original):
        with pytest.raises(ConnectionError, match="network timeout"):
            download_gated_model("owner/model", tmp_path)
