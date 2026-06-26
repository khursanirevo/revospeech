"""RevoSpeech TTS — Text-to-Speech.

Usage:
    >>> from revospeech.tts import TTS
    >>>
    >>> # Auto-select smallest ready TTS model
    >>> tts = TTS()
    >>>
    >>> # Or specify explicitly
    >>> tts = TTS('revovoice')
    >>>
    >>> # Basic synthesis
    >>> audio = tts.synthesize('Hello, how are you?')
    >>> audio.save('greeting.wav')
    >>>
    >>> # Voice cloning with reference audio
    >>> audio = tts.synthesize(
    ...     'This will sound like the reference speaker.',
    ...     ref_audio='speaker.wav',
    ...     ref_text='Sample of the speaker talking.',
    ... )
    >>>
    >>> # Long text auto-splits into chunks
    >>> audio = tts.synthesize_long(open('script.txt').read())
    >>>
    >>> # Batch synthesis with output directory
    >>> report = tts.synthesize_batch(
    ...     ['Hello.', 'World.'],
    ...     output_dir='output/',
    ... )
    >>>
    >>> # Playback (requires sounddevice)
    >>> audio = tts.synthesize('Play this immediately.')
    >>> audio.play()  # blocking

BytesIO round-trip with ASR:
    >>> import io
    >>> from revospeech.asr import ASR
    >>> buf = io.BytesIO()
    >>> tts.synthesize('Test sentence.').save(buf)
    >>> buf.seek(0)
    >>> ASR().transcribe(buf).text
"""

from __future__ import annotations

import logging
from typing import overload

from revospeech.config import set_api_key
from revospeech.exceptions import RevosModelError
from revospeech.registry import get
from revospeech.registry.downloader import ensure_model
from revospeech.registry.status import check_model, list_model_statuses, suggest_models

from .base import BaseTTS
from .result import Audio

logger = logging.getLogger(__name__)


@overload
def TTS(
    model_name: None = None,
    *,
    device: str = "auto",
    api_key: str | None = None,
    auto_download: bool = True,
    restore: bool = False,
) -> BaseTTS: ...


@overload
def TTS(
    model_name: str,
    *,
    device: str = "auto",
    api_key: str | None = None,
    auto_download: bool = True,
    restore: bool = False,
) -> BaseTTS: ...


def TTS(
    model_name: str | None = None,
    device: str = "auto",
    api_key: str | None = None,
    *,
    auto_download: bool = True,
    restore: bool = False,
) -> BaseTTS:
    """Create a TTS engine for the given model.

    Looks up the model manifest and dispatches to the appropriate backend.

    Args:
        model_name: Name of the TTS model (e.g. "revovoice").
            If None, auto-selects the first ready local model.
        device: Compute device — "auto", "cpu", or "cuda".
        api_key: If provided, persists it via ``set_api_key`` before loading.
        auto_download: If True (default), automatically download local models
            that are not yet cached before constructing the engine.
        restore: If True, apply any ready util model tagged
            ``tts-postprocess`` (e.g. Sidon speech restoration) to every
            ``synthesize()`` output. Off by default — opt in per-call.

    Returns:
        A BaseTTS instance ready for synthesis.

    Raises:
        KeyError: If the model is not registered (with did-you-mean hints).
        ValueError: If the model backend is not supported.
        RevosModelError: If auto-selection finds no ready models.
    """
    if api_key is not None:
        set_api_key(api_key)

    if model_name is None:
        ready = list_model_statuses(task="tts", mode="local", status="ready")
        if not ready:
            available = list_model_statuses(task="tts")
            if not available:
                suggestion = (
                    "No TTS models registered. Browse the catalog: "
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
                "No TTS model specified and no ready local models found.",
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
        manifest = get(model_name, "tts")
    except KeyError:
        suggestions = suggest_models(model_name, "tts")
        hint = ""
        if suggestions:
            hint = f" Did you mean: {', '.join(suggestions)}?"
        raise KeyError(
            f"TTS model '{model_name}' not found.{hint}"
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

    # Validate backend before attempting download — fail fast on typos.
    if manifest.backend not in ("revovoice", "vits"):
        raise ValueError(
            f"Unsupported TTS backend: '{manifest.backend}' for model "
            f"'{model_name}'. Supported backends: revovoice, vits"
        )

    # Auto-download: fetch local model files if not yet cached.
    if auto_download and manifest.is_local and manifest.model_url:
        status = check_model(model_name, task="tts")
        if status.status == "needs-download":
            logger.info("Model '%s' not downloaded. Auto-downloading...", model_name)
            ensure_model(manifest)

    if manifest.backend == "revovoice":
        from .revovoice_engine import RevoVoiceTTS

        engine: BaseTTS = RevoVoiceTTS(model_name, device)
    else:
        from .vits_engine import VitsTTS

        engine = VitsTTS(model_name, device)

    if restore:
        engine = _attach_post_processors(engine, device)
    return engine


def _attach_post_processors(engine: BaseTTS, device: str) -> BaseTTS:
    """Wrap engine.synthesize so any ready util tagged tts-postprocess runs after."""
    try:
        from revospeech.registry import list_models
        from revospeech.registry.status import check_model
        from revospeech.util import Util

        candidates = [
            m
            for m in list_models()
            if m.task == "util" and "tts-postprocess" in (m.tags or [])
        ]
        ready: list = []
        for m in candidates:
            try:
                if check_model(m.name, task=m.task).status == "ready":
                    ready.append(m)
            except Exception:
                continue
        if not ready:
            return engine

        post_processors = []
        for m in ready:
            try:
                post_processors.append(Util(m.name, device=device, auto_download=False))
                logger.info(
                    "Attached post-processor '%s' to TTS engine '%s'",
                    m.name,
                    engine.model_name,
                )
            except Exception as e:
                logger.warning("Failed to load util '%s': %s", m.name, e)
        if not post_processors:
            return engine

        setattr(engine, "_post_processors", post_processors)
        original_synthesize = engine.synthesize

        import inspect

        bound_sig = inspect.signature(original_synthesize)

        def synthesize_with_restore(*args, **kwargs):
            audio = original_synthesize(*args, **kwargs)
            try:
                bound = bound_sig.bind(*args, **kwargs)
                bound.apply_defaults()
                output_path = bound.arguments.get("output_path")
            except TypeError:
                output_path = kwargs.get("output_path")
            for proc in post_processors:
                try:
                    audio = proc.restore(audio)
                except Exception as e:
                    logger.warning(
                        "Post-processor '%s' failed: %s — returning unenhanced audio",
                        proc.model_name,
                        e,
                    )
                    return audio
            if output_path:
                audio.save(output_path)
            return audio

        engine.synthesize = synthesize_with_restore  # type: ignore[method-assign]
        return engine
    except Exception as e:
        logger.debug("Post-processor attach skipped: %s", e)
        return engine


__all__ = ["TTS", "BaseTTS", "Audio"]
