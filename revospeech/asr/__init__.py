"""RevoSpeech ASR — Automatic Speech Recognition.

Usage:
    from revospeech.asr import ASR

    asr = ASR('zipformer-v2')
    result = asr.transcribe('audio.wav')
    print(result.text)
"""

from __future__ import annotations

from revospeech.config import set_api_key
from revospeech.exceptions import RevosModelError
from revospeech.registry import get
from revospeech.registry.status import list_model_statuses, suggest_models

from .base import BaseASR
from .result import Segment, Transcript


def ASR(
    model_name: str | None = None,
    device: str = "auto",
    api_key: str | None = None,
) -> BaseASR:
    """Create an ASR engine for the given model.

    Looks up the model manifest and dispatches to the appropriate backend.

    Args:
        model_name: Name of the ASR model (e.g. "zipformer-v2").
            If None, auto-selects the first ready local model.
        device: Compute device — "auto", "cpu", or "cuda".
        api_key: If provided, persists it via ``set_api_key`` before loading.

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
            raise RevosModelError(
                "No ASR model specified and no ready local models found.",
                suggestion=(
                    "Download a model first: revospeech models download <name>"
                    " or list available models: revospeech models"
                ),
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

    # API-mode gate: validate key, then raise (API engine not yet implemented)
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
        raise NotImplementedError(
            f"API backend for '{model_name}' is not yet implemented. "
            f"Contributing guide: see CONTRIBUTING.md"
        )

    if manifest.backend == "sherpa-onnx":
        from .sherpa_engine import SherpaOnnxASR

        return SherpaOnnxASR(model_name, device)

    raise ValueError(
        f"Unsupported ASR backend: '{manifest.backend}' for model "
        f"'{model_name}'. Supported backends: sherpa-onnx"
    )


__all__ = ["ASR", "BaseASR", "Transcript", "Segment", "BatchReport", "BatchResult"]
