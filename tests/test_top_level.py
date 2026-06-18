"""Tests for the revospeech top-level API surface.

US-026: Verifies that the package-level imports, lazy attributes, and
__version__ behave as documented in revospeech/__init__.py.
"""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# __version__
# ---------------------------------------------------------------------------
def test_version_is_string():
    """__version__ must be a non-empty string."""
    import revospeech

    assert isinstance(revospeech.__version__, str)
    assert len(revospeech.__version__) > 0


def test_version_not_placeholder_when_installed():
    """If the package is installed, version should not be the dev fallback.

    This test is lenient: if the package is running in editable/develop mode
    without metadata, version falls back to '0.0.0-dev'. We accept either but
    verify the value is one of the known shapes.
    """
    import revospeech

    assert revospeech.__version__ in {"0.0.0-dev"} or any(
        c.isdigit() for c in revospeech.__version__
    ), f"Unexpected version value: {revospeech.__version__!r}"


# ---------------------------------------------------------------------------
# __dir__ and lazy __getattr__
# ---------------------------------------------------------------------------
def test_dir_returns_expected_exports():
    """__dir__ should list the documented public API names."""
    import revospeech

    exports = dir(revospeech)
    expected = {
        "ASR",
        "TTS",
        "configure_logging",
        "list_models",
        "search_models",
        "check_model",
    }
    missing = expected - set(exports)
    assert not missing, f"Missing from dir(revospeech): {missing}"


def test_all_contains_expected_names():
    """__all__ should list the documented public API names."""
    import revospeech

    assert hasattr(revospeech, "__all__")
    expected = {"ASR", "TTS", "list_models", "search_models"}
    assert expected.issubset(set(revospeech.__all__))


# ---------------------------------------------------------------------------
# Direct function exports
# ---------------------------------------------------------------------------
def test_list_models_is_callable():
    """list_models should be a callable at the top level."""
    import revospeech

    assert callable(revospeech.list_models)


def test_search_models_is_callable():
    """search_models should be a callable at the top level."""
    import revospeech

    assert callable(revospeech.search_models)


def test_list_models_returns_list():
    """list_models() should return a list (possibly empty if no manifests)."""
    import revospeech

    result = revospeech.list_models()
    assert isinstance(result, list)


def test_search_models_returns_list():
    """search_models() should return a list for any query."""
    import revospeech

    result = revospeech.search_models("nonexistent_query_xyz_123")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Lazy attribute access via __getattr__
# ---------------------------------------------------------------------------
def test_getattr_invalid_name_raises_attribute_error():
    """Accessing a non-existent attribute should raise AttributeError."""
    import revospeech

    with pytest.raises(AttributeError, match="nonexistent_thing"):
        _ = revospeech.nonexistent_thing  # type: ignore[attr-defined]


def test_getattr_invalid_name_message_contains_name():
    """The AttributeError message should mention the missing name."""
    import revospeech

    with pytest.raises(AttributeError) as exc_info:
        _ = revospeech.totally_bogus_attr  # type: ignore[attr-defined]
    assert "totally_bogus_attr" in str(exc_info.value)


def test_check_model_is_accessible():
    """check_model should be accessible via lazy import."""
    import revospeech

    # Accessing the attribute triggers lazy import; it should not raise.
    fn = revospeech.check_model
    assert callable(fn)


def test_configure_logging_is_accessible():
    """configure_logging should be accessible via lazy import."""
    import revospeech

    fn = revospeech.configure_logging
    assert callable(fn)


# ---------------------------------------------------------------------------
# Module reload safety
# ---------------------------------------------------------------------------
def test_reload_preserves_version():
    """Reloading revospeech should not corrupt __version__."""
    import revospeech

    v1 = revospeech.__version__
    importlib.reload(revospeech)
    v2 = revospeech.__version__
    assert v1 == v2


def test_check_model_missing_raises_key_error():
    """check_model on an unknown name should raise KeyError."""
    import revospeech

    with pytest.raises((KeyError, LookupError)):
        revospeech.check_model("this-model-does-not-exist-xyz-999")


# ---------------------------------------------------------------------------
# search_models scoring paths — name, capabilities, languages
# ---------------------------------------------------------------------------
def test_search_models_matches_by_capability():
    """Capabilities substring match contributes to the score."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register
    from revospeech.registry.status import ModelStatus

    saved = dict(_models)
    _models.clear()
    try:
        register(
            ModelManifest(
                name="weird-name-xyz",
                task="asr",
                backend="sherpa-onnx",
                model_type="transducer",
                model_url="",
                sample_rate=16000,
                language="en",
                description="",
                capabilities=["streaming"],
                languages=["en"],
                files={},
            )
        )

        import revospeech as rs

        # Stub list_model_statuses so search sees our model with capabilities.
        from revospeech.registry import status as status_mod

        fake = [
            ModelStatus(
                name="weird-name-xyz",
                task="asr",
                mode="local",
                status="ready",
                installed=True,
                size_mb=10.0,
                capabilities=["streaming"],
                languages=["en"],
            )
        ]
        orig = status_mod.list_model_statuses
        status_mod.list_model_statuses = lambda **kw: fake
        try:
            results = rs.search_models("stream")
            assert any(m.name == "weird-name-xyz" for m in results)
        finally:
            status_mod.list_model_statuses = orig
    finally:
        _models.clear()
        _models.update(saved)


def test_search_models_matches_by_language():
    """Languages substring match contributes to the score."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register
    from revospeech.registry.status import ModelStatus

    saved = dict(_models)
    _models.clear()
    try:
        register(
            ModelManifest(
                name="zzz-obscure",
                task="asr",
                backend="sherpa-onnx",
                model_type="transducer",
                model_url="",
                sample_rate=16000,
                language="en",
                description="",
                capabilities=[],
                languages=["fr"],
                files={},
            )
        )

        import revospeech as rs
        from revospeech.registry import status as status_mod

        fake = [
            ModelStatus(
                name="zzz-obscure",
                task="asr",
                mode="local",
                status="ready",
                installed=True,
                size_mb=10.0,
                capabilities=[],
                languages=["fr"],
            )
        ]
        orig = status_mod.list_model_statuses
        status_mod.list_model_statuses = lambda **kw: fake
        try:
            results = rs.search_models("fr")
            assert any(m.name == "zzz-obscure" for m in results)
        finally:
            status_mod.list_model_statuses = orig
    finally:
        _models.clear()
        _models.update(saved)


def test_search_models_partial_name_match_scores_higher_than_unrelated():
    """Direct name substring match scores higher than fuzzy unrelated names."""
    from revospeech.registry.manifest import ModelManifest
    from revospeech.registry.registry import _models, register
    from revospeech.registry.status import ModelStatus

    saved = dict(_models)
    _models.clear()
    try:
        register(
            ModelManifest(
                name="zipformer-v2",
                task="asr",
                backend="sherpa-onnx",
                model_type="transducer",
                model_url="",
                sample_rate=16000,
                language="en",
                description="",
                files={},
            )
        )
        register(
            ModelManifest(
                name="totally-unrelated",
                task="asr",
                backend="sherpa-onnx",
                model_type="transducer",
                model_url="",
                sample_rate=16000,
                language="en",
                description="",
                files={},
            )
        )

        import revospeech as rs
        from revospeech.registry import status as status_mod

        fake = [
            ModelStatus(
                name="zipformer-v2",
                task="asr",
                mode="local",
                status="ready",
                installed=True,
                size_mb=10.0,
                capabilities=[],
                languages=["en"],
            ),
            ModelStatus(
                name="totally-unrelated",
                task="asr",
                mode="local",
                status="ready",
                installed=True,
                size_mb=10.0,
                capabilities=[],
                languages=["en"],
            ),
        ]
        orig = status_mod.list_model_statuses
        status_mod.list_model_statuses = lambda **kw: fake
        try:
            results = rs.search_models("zip")
            names = [m.name for m in results]
            assert "zipformer-v2" in names
            if "totally-unrelated" in names:
                # If both matched, zipformer-v2 must rank first.
                assert names.index("zipformer-v2") < names.index("totally-unrelated")
        finally:
            status_mod.list_model_statuses = orig
    finally:
        _models.clear()
        _models.update(saved)
