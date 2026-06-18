"""Tests for HuggingFace utility helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
