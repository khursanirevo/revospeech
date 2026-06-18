"""Tests for revospeech.config module."""

from __future__ import annotations

import pytest

import revospeech.config as config_mod


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Redirect CONFIG_DIR and CONFIG_FILE to a tmp path."""
    config_dir = tmp_path / "revospeech"
    config_file = config_dir / "config.yaml"
    monkeypatch.setattr(config_mod, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", config_file)
    return config_file


def test_load_config_returns_empty_when_missing(isolated_config):
    assert config_mod.load_config() == {}


def test_save_then_load_round_trip(isolated_config):
    config_mod.save_config({"api_key": "rv-test", "other": "value"})
    loaded = config_mod.load_config()
    assert loaded["api_key"] == "rv-test"
    assert loaded["other"] == "value"


def test_save_config_sets_permissions(isolated_config):
    config_mod.save_config({"api_key": "rv-test"})
    mode = isolated_config.stat().st_mode & 0o777
    assert mode == 0o600


def test_load_config_invalid_yaml_returns_empty(isolated_config):
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text("not: valid: yaml: [")
    assert config_mod.load_config() == {}


def test_load_config_non_dict_yaml_returns_empty(isolated_config):
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text("- just\n- a\n- list\n")
    assert config_mod.load_config() == {}


def test_get_api_key_explicit_arg_wins(isolated_config, monkeypatch):
    monkeypatch.delenv(config_mod.ENV_VAR, raising=False)
    config_mod.save_config({"api_key": "from-file"})
    assert config_mod.get_api_key("from-arg") == "from-arg"


def test_get_api_key_env_var_beats_file(isolated_config, monkeypatch):
    monkeypatch.setenv(config_mod.ENV_VAR, "from-env")
    config_mod.save_config({"api_key": "from-file"})
    assert config_mod.get_api_key() == "from-env"


def test_get_api_key_falls_back_to_file(isolated_config, monkeypatch):
    monkeypatch.delenv(config_mod.ENV_VAR, raising=False)
    config_mod.save_config({"api_key": "from-file"})
    assert config_mod.get_api_key() == "from-file"


def test_get_api_key_returns_none_when_nothing_set(isolated_config, monkeypatch):
    monkeypatch.delenv(config_mod.ENV_VAR, raising=False)
    assert config_mod.get_api_key() is None


def test_set_api_key_preserves_existing_config(isolated_config, monkeypatch):
    monkeypatch.delenv(config_mod.ENV_VAR, raising=False)
    config_mod.save_config({"other_key": "preserve-me"})
    config_mod.set_api_key("rv-new")
    loaded = config_mod.load_config()
    assert loaded["api_key"] == "rv-new"
    assert loaded["other_key"] == "preserve-me"
