"""Tests for revospeech.logging_config module."""

from __future__ import annotations

import logging

import pytest

from revospeech.logging_config import configure_logging


@pytest.fixture(autouse=True)
def reset_logger():
    """Clear revospeech logger handlers between tests."""
    logger = logging.getLogger("revospeech")
    original_level = logger.level
    original_handlers = list(logger.handlers)
    logger.handlers.clear()
    yield
    logger.handlers = original_handlers
    logger.setLevel(original_level)


def test_configure_logging_default_warning():
    configure_logging()
    logger = logging.getLogger("revospeech")
    assert logger.level == logging.WARNING


def test_configure_logging_debug():
    configure_logging("DEBUG")
    logger = logging.getLogger("revospeech")
    assert logger.level == logging.DEBUG


def test_configure_logging_info():
    configure_logging("INFO")
    assert logging.getLogger("revospeech").level == logging.INFO


def test_configure_logging_lowercase_input():
    configure_logging("error")
    assert logging.getLogger("revospeech").level == logging.ERROR


def test_configure_logging_invalid_raises():
    with pytest.raises(ValueError, match="Invalid log level"):
        configure_logging("LOUD")


def test_configure_logging_adds_handler_on_first_call():
    logger = logging.getLogger("revospeech")
    assert logger.handlers == []
    configure_logging("INFO")
    assert len(logger.handlers) == 1


def test_configure_logging_does_not_add_duplicate_handler():
    logger = logging.getLogger("revospeech")
    configure_logging("INFO")
    configure_logging("DEBUG")
    assert len(logger.handlers) == 1


def test_configure_logging_updates_existing_handler_level():
    logger = logging.getLogger("revospeech")
    configure_logging("INFO")
    configure_logging("ERROR")
    assert logger.handlers[0].level == logging.ERROR
