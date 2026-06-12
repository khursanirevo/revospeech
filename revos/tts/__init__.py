"""RevoS TTS — Text-to-Speech.

Usage:
    from revos.tts import TTS

    tts = TTS('revovoice')
    audio = tts.synthesize('Hello, world!')
    audio.save('output.wav')
"""

from __future__ import annotations

from revos.registry import get

from .base import BaseTTS
from .result import Audio


def TTS(model_name: str, device: str = "auto") -> BaseTTS:
    """Create a TTS engine for the given model.

    Looks up the model manifest and dispatches to the appropriate backend.

    Args:
        model_name: Name of the TTS model (e.g. "revovoice").
        device: Compute device — "auto", "cpu", or "cuda".

    Returns:
        A BaseTTS instance ready for synthesis.

    Raises:
        KeyError: If the model is not registered.
        ValueError: If the model backend is not supported.
    """
    manifest = get(model_name, "tts")

    # API-mode gate: validate key, then raise (API engine not yet implemented)
    if manifest.is_api:
        from revos.config import get_api_key
        from revos.exceptions import RevosConfigError

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

    if manifest.backend == "revovoice":
        from .revovoice_engine import RevoVoiceTTS

        return RevoVoiceTTS(model_name, device)

    raise ValueError(
        f"Unsupported TTS backend: '{manifest.backend}' for model "
        f"'{model_name}'. Supported backends: revovoice"
    )


__all__ = ["TTS", "BaseTTS", "Audio"]
