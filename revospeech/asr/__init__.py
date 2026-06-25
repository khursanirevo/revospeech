"""RevoSpeech ASR — Automatic Speech Recognition.

Usage:
    >>> from revospeech.asr import ASR
    >>>
    >>> # Auto-select smallest ready ASR model
    >>> asr = ASR()
    >>>
    >>> # Or specify explicitly
    >>> asr = ASR('zipformer-v2')
    >>>
    >>> result = asr.transcribe('audio.wav')
    >>> print(result.text)
    >>> print(result.duration)  # audio duration in seconds
    >>>
    >>> # Save transcript in different formats
    >>> result.save('out.srt')  # or .vtt, .json, .txt
    >>>
    >>> # Batch processing
    >>> report = asr.transcribe_batch(['a.wav', 'b.wav'])
    >>> report.save('results.json')

BytesIO support for in-memory audio:
    >>> import io
    >>> buf = io.BytesIO(wav_bytes)
    >>> result = asr.transcribe(buf)
"""

from __future__ import annotations

import logging
from typing import overload

from revospeech.config import set_api_key
from revospeech.exceptions import RevosModelError
from revospeech.registry import get
from revospeech.registry.downloader import ensure_model
from revospeech.registry.status import check_model, list_model_statuses, suggest_models

from .base import BaseASR
from .result import Segment, Transcript

logger = logging.getLogger(__name__)


@overload
def ASR(
    model_name: None = None,
    *,
    device: str = "auto",
    api_key: str | None = None,
    auto_download: bool = True,
) -> BaseASR: ...


@overload
def ASR(
    model_name: str,
    *,
    device: str = "auto",
    api_key: str | None = None,
    auto_download: bool = True,
) -> BaseASR: ...


def ASR(
    model_name: str | None = None,
    device: str = "auto",
    api_key: str | None = None,
    *,
    auto_download: bool = True,
) -> BaseASR:
    """Create an ASR engine for the given model.

    Looks up the model manifest and dispatches to the appropriate backend.

    Args:
        model_name: Name of the ASR model (e.g. "zipformer-v2").
            If None, auto-selects the first ready local model.
        device: Compute device — "auto", "cpu", or "cuda".
        api_key: If provided, persists it via ``set_api_key`` before loading.
        auto_download: If True (default), automatically download local models
            that are not yet cached before constructing the engine.

    Returns:
        A BaseASR instance ready for transcription.

    Raises:
        KeyError: If the model is not registered (with did-you-mean hints).
        ValueError: If the model backend is not supported.
        RevosModelError: If auto-selection finds no ready models.
    """
    if api_key is not None:
        set_api_key(api_key)

    if model_name is None:
        ready = list_model_statuses(task="asr", mode="local", status="ready")
        if not ready:
            available = list_model_statuses(task="asr")
            if not available:
                suggestion = (
                    "No ASR models registered. Browse the catalog: "
                    "'revospeech catalog list'"
                    " then 'revospeech catalog pull <name>'."
                )
            else:
                names = [m.name for m in available[:5]]
                suggestion = (
                    f"Download a model first with "
                    f"'revospeech models --download <name>'. "
                    f"Available: {', '.join(names)}"
                    f"{' ...' if len(available) > 5 else ''}"
                )
            raise RevosModelError(
                "No ASR model specified and no ready local models found.",
                suggestion=suggestion,
            )
        # Pick the smallest by size_mb (None sorts as largest), else first.
        with_size = [m for m in ready if m.size_mb is not None]
        if with_size:
            chosen = min(with_size, key=lambda m: m.size_mb or float("inf"))
        else:
            chosen = ready[0]
        model_name = chosen.name

    try:
        manifest = get(model_name, "asr")
    except KeyError:
        suggestions = suggest_models(model_name, "asr")
        hint = ""
        if suggestions:
            hint = f" Did you mean: {', '.join(suggestions)}?"
        raise KeyError(
            f"ASR model '{model_name}' not found.{hint}"
            f" Run 'revospeech models' to list available models."
        ) from None

    # API-mode gate: validate key, then dispatch to the API engine.
    if manifest.is_api:
        from revospeech.config import get_api_key
        from revospeech.exceptions import RevosConfigError

        resolved_key = get_api_key()
        if not resolved_key:
            raise RevosConfigError(
                f"Model '{model_name}' requires an API key.",
                suggestion=(
                    "Set your API key: export REVOLAB_API_KEY=your-key"
                    " or run: revos config set-api-key"
                ),
            )

        if manifest.backend == "revolab":
            from .revolab_engine import RevolabASR

            return RevolabASR(model_name, device=device)

        raise ValueError(
            f"Unsupported API backend: '{manifest.backend}' for model "
            f"'{model_name}'. Supported API backends: revolab"
        )

    # Validate backend before attempting download — fail fast on typos.
    if manifest.backend != "sherpa-onnx":
        raise ValueError(
            f"Unsupported ASR backend: '{manifest.backend}' for model "
            f"'{model_name}'. Supported backends: sherpa-onnx"
        )

    # Auto-download: fetch local model files if not yet cached.
    if auto_download and manifest.is_local and manifest.model_url:
        status = check_model(model_name, task="asr")
        if status.status == "needs-download":
            logger.info("Model '%s' not downloaded. Auto-downloading...", model_name)
            ensure_model(manifest)

    from .sherpa_engine import SherpaOnnxASR

    return SherpaOnnxASR(model_name, device)


__all__ = ["ASR", "BaseASR", "Transcript", "Segment", "BatchReport", "BatchResult"]
