"""Logging configuration for RevoSpeech.

Provides a simple way to control log output verbosity.

Usage:
    import revospeech
    revos.configure_logging("DEBUG")   # Verbose
    revos.configure_logging("WARNING") # Quiet
"""

from __future__ import annotations

import logging


def configure_logging(level: str = "WARNING") -> None:
    """Configure revos logging verbosity.

    Args:
        level: Log level string. One of:
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
            Defaults to "WARNING".
    """
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(
            f"Invalid log level: '{level}'. "
            f"Use DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )

    logger = logging.getLogger("revospeech")
    logger.setLevel(numeric)

    # Add handler only if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(numeric)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for existing_handler in logger.handlers:
            existing_handler.setLevel(numeric)
