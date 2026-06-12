"""Usage tracking for gated models.

Provides hooks to track which users load and use gated models.
Configure via ~/.config/revos/config.yaml or environment variables.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Type for usage callbacks
UsageCallback = Callable[[dict], None]

# Registered callbacks
_callbacks: list[UsageCallback] = []

# Thread safety locks
_callbacks_lock = threading.Lock()
_file_lock = threading.Lock()

# Local usage log path
_USAGE_LOG = Path.home() / ".cache" / "revos" / "usage.jsonl"
_MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB


def register_callback(callback: UsageCallback) -> None:
    """Register a callback to be called when a gated model is loaded.

    The callback receives a dict with:
        - model_id: HuggingFace model ID
        - model_name: revos model name
        - task: "asr" or "tts"
        - hf_user: HuggingFace username (or None)
        - device: "cpu" or "cuda"
        - timestamp: ISO 8601 UTC timestamp
        - event: "model_loaded" or "model_synthesized"

    Args:
        callback: Function that takes a usage dict.
    """
    with _callbacks_lock:
        _callbacks.append(callback)


def _log_to_local(usage: dict) -> None:
    """Append usage event to local JSONL log.

    Rotates the log file to ``usage.jsonl.1`` when it exceeds 10 MB.
    """
    _USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)

    if _USAGE_LOG.exists() and os.path.getsize(_USAGE_LOG) > _MAX_LOG_SIZE:
        rotated = _USAGE_LOG.with_suffix(".jsonl.1")
        if rotated.exists():
            rotated.unlink()
        os.replace(_USAGE_LOG, rotated)

    with open(_USAGE_LOG, "a") as f:
        f.write(json.dumps(usage, default=str) + "\n")
    os.chmod(_USAGE_LOG, 0o600)


def track_usage(
    event: str,
    model_id: str,
    model_name: str,
    task: str,
    hf_user: str | None,
    device: str,
    **extra: object,
) -> None:
    """Record a usage event and notify all registered callbacks.

    Args:
        event: Event type ("model_loaded", "model_synthesized").
        model_id: HuggingFace model ID.
        model_name: RevoS model name.
        task: "asr" or "tts".
        hf_user: HuggingFace username (or None).
        device: Device used ("cpu" or "cuda").
        **extra: Additional data to include.
    """
    usage = {
        "event": event,
        "model_id": model_id,
        "model_name": model_name,
        "task": task,
        "hf_user": hf_user,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **extra,
    }

    # Always log locally
    with _file_lock:
        _log_to_local(usage)

    # Notify registered callbacks
    with _callbacks_lock:
        callbacks_snapshot = list(_callbacks)
    for callback in callbacks_snapshot:
        try:
            callback(usage)
        except Exception as e:
            logger.warning("Usage callback failed: %s", e)

    logger.debug("Usage tracked: %s %s by %s", event, model_id, usage.get("hf_user"))


def get_usage_log() -> list[dict]:
    """Read all usage events from the local log.

    Returns:
        List of usage event dicts, most recent last.
    """
    if not _USAGE_LOG.exists():
        return []
    events = []
    with open(_USAGE_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
