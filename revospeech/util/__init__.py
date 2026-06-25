"""Util models: audio post-processors (speech restoration, etc.).

Public API:
    >>> from revospeech.util import Util
    >>> sidon = Util("sidon")
    >>> enhanced = sidon.restore(audio)

Util models are NOT main ASR/TTS engines — they transform audio outputs.
"""

from __future__ import annotations

import logging

from revospeech.registry import get
from revospeech.registry.status import check_model

from .base import BaseUtil

logger = logging.getLogger(__name__)


def Util(
    model_name: str = "sidon",
    *,
    device: str = "auto",
    auto_download: bool = True,
) -> BaseUtil:
    """Factory for util models.

    Args:
        model_name: Name of the util model (e.g. "sidon").
        device: "auto", "cpu", or "cuda".
        auto_download: Download model files if not yet cached.

    Returns:
        Engine instance (subclass of BaseUtil).
    """
    manifest = get(model_name, "util")

    if manifest.is_api:
        raise NotImplementedError(
            f"API-mode util model '{model_name}' is not yet supported."
        )

    if manifest.backend == "sidon":
        from .sidon_engine import SidonUtil

        if auto_download and manifest.model_url and not manifest.files == {}:
            status = check_model(model_name, task="util")
            if status.status == "needs-download":
                from revospeech.registry.downloader import ensure_model

                logger.info("Auto-downloading util model '%s'", model_name)
                try:
                    ensure_model(manifest)
                except Exception:
                    # Sidon uses HF snapshot_download via the engine itself.
                    logger.debug("ensure_model fell back to engine-side download")

        return SidonUtil(model_name, device=device)

    raise ValueError(
        f"Unknown util backend '{manifest.backend}' for model '{model_name}'. "
        f"Supported: 'sidon'."
    )


__all__ = ["Util", "BaseUtil"]
