"""Centralized configuration for revos.

Manages the config file at ~/.config/revos/config.yaml and provides
API key resolution with a clear priority chain.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".config" / "revos"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
ENV_VAR = "REVOLAB_API_KEY"

__all__ = [
    "load_config",
    "save_config",
    "get_api_key",
    "set_api_key",
    "CONFIG_DIR",
    "CONFIG_FILE",
]


def load_config() -> dict:
    """Load config from ~/.config/revos/config.yaml.

    Returns:
        Parsed config dict, or empty dict if file is missing or invalid.
    """
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.warning("Failed to load config from %s", CONFIG_FILE)
        return {}


def save_config(data: dict) -> None:
    """Save config to ~/.config/revos/config.yaml.

    Creates the directory if needed. Sets file permissions to 0o600
    to protect any sensitive values (e.g. API keys).

    Args:
        data: Config dict to serialize as YAML.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = CONFIG_FILE.with_suffix(".tmp")
    try:
        with open(tmp_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
        tmp_path.rename(CONFIG_FILE)
        CONFIG_FILE.chmod(0o600)
    except Exception:
        logger.exception("Failed to save config to %s", CONFIG_FILE)
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def get_api_key(key: str | None = None) -> str | None:
    """Resolve API key in priority order.

    1. Explicit *key* argument
    2. Environment variable REVOLAB_API_KEY
    3. config.yaml -> api_key field

    Returns:
        The resolved key string, or None if nothing found.
    """
    if key:
        return key
    env_key = os.environ.get(ENV_VAR)
    if env_key:
        return env_key
    config = load_config()
    return config.get("api_key")


def set_api_key(key: str) -> None:
    """Save API key to config.yaml.

    Preserves any existing config values.

    Args:
        key: API key string to store.
    """
    config = load_config()
    config["api_key"] = key
    save_config(config)
    logger.info("API key saved to %s", CONFIG_FILE)
