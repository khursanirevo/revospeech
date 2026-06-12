"""Custom exception hierarchy for the revos library."""

__all__ = [
    "RevosAudioError",
    "RevosConfigError",
    "RevosEngineError",
    "RevosError",
    "RevosModelError",
]


class RevosError(Exception):
    """Base exception for all revos errors."""

    def __init__(self, message: str, *, suggestion: str | None = None) -> None:
        self.message = message
        self.suggestion = suggestion
        super().__init__(message)

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message}\nSuggestion: {self.suggestion}"
        return self.message


class RevosConfigError(RevosError):
    """Missing API key, bad config, etc."""

    def __init__(self, message: str, *, suggestion: str | None = None) -> None:
        super().__init__(message, suggestion=suggestion)


class RevosModelError(RevosError):
    """Model not found, download failed, etc."""

    def __init__(self, message: str, *, suggestion: str | None = None) -> None:
        super().__init__(message, suggestion=suggestion)


class RevosEngineError(RevosError):
    """Inference failure, etc."""

    def __init__(self, message: str, *, suggestion: str | None = None) -> None:
        super().__init__(message, suggestion=suggestion)


class RevosAudioError(RevosError):
    """Unsupported format, corrupt file, etc."""

    def __init__(self, message: str, *, suggestion: str | None = None) -> None:
        super().__init__(message, suggestion=suggestion)
