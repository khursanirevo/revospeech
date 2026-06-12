"""RevoSpeech ASR — Automatic Speech Recognition.

Usage:
    from revospeech.asr import ASR

    asr = ASR('zipformer-v2')
    result = asr.transcribe('audio.wav')
    print(result.text)
"""

from __future__ import annotations

from revospeech.registry import get

from .base import BaseASR
from .result import Segment, Transcript


def ASR(model_name: str, device: str = "auto") -> BaseASR:
    """Create an ASR engine for the given model.

    Looks up the model manifest and dispatches to the appropriate backend.

    Args:
        model_name: Name of the ASR model (e.g. "zipformer-v2").
        device: Compute device — "auto", "cpu", or "cuda".

    Returns:
        A BaseASR instance ready for transcription.

    Raises:
        KeyError: If the model is not registered.
        ValueError: If the model backend is not supported.
    """
    manifest = get(model_name, "asr")

    # API-mode gate: validate key, then raise (API engine not yet implemented)
    if manifest.is_api:
        from revospeech.config import get_api_key
        from revospeech.exceptions import RevosConfigError

        api_key = get_api_key()
        if not api_key:
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
