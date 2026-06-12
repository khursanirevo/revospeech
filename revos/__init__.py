"""RevoS — A unified Python library for speech AI.

Quick start:
    from revos.asr import ASR
    from revos.tts import TTS
"""

try:
    from importlib.metadata import version as _version

    __version__ = _version("revos")
except Exception:
    __version__ = "0.0.0-dev"


def __getattr__(name: str):
    """Lazy re-exports for convenience: from revos import ASR, TTS."""
    if name == "ASR":
        from revos.asr import ASR

        return ASR
    if name == "TTS":
        from revos.tts import TTS

        return TTS
    if name == "configure_logging":
        from revos.logging_config import configure_logging

        return configure_logging
    raise AttributeError(f"module 'revos' has no attribute {name!r}")


def list_models(
    task: str | None = None,
    mode: str | None = None,
    language: str | None = None,
    status: str | None = None,
    capability: str | None = None,
) -> list:
    """List available models with optional filters."""
    from revos.registry.status import list_model_statuses

    return list_model_statuses(
        task=task,
        mode=mode,
        language=language,
        status=status,
        capability=capability,
    )


def search_models(query: str) -> list:
    """Search models by name, tags, or description using fuzzy matching."""
    from difflib import SequenceMatcher

    from revos.registry.status import list_model_statuses

    query_lower = query.lower()
    all_models = list_model_statuses()

    scored = []
    for m in all_models:
        score = 0.0
        # Name match (highest weight)
        if query_lower in m.name.lower():
            score += 1.0
        else:
            score += SequenceMatcher(None, query_lower, m.name.lower()).ratio() * 0.7
        # Tag match
        for tag in getattr(m, "capabilities", []):
            if query_lower in tag.lower():
                score += 0.5
        # Language match
        for lang in getattr(m, "languages", []):
            if query_lower in lang.lower():
                score += 0.3
        if score > 0.2:
            scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]


__all__ = [
    "ASR",
    "TTS",
    "configure_logging",
    "list_models",
    "search_models",
]
