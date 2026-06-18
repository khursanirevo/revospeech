"""Tests for catalog extras: installed status and recommendations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from revospeech.registry.manifest import ModelManifest


def _make_manifest(
    name: str,
    task: str = "asr",
    language: str = "en",
    languages: list[str] | None = None,
    size_mb: float = 100.0,
) -> ModelManifest:
    """Build a minimal ModelManifest for tests."""
    return ModelManifest(
        name=name,
        task=task,
        backend="x",
        model_type="transducer",
        model_url="",
        sample_rate=16000,
        language=language,
        description="",
        files={},
        languages=languages if languages is not None else [],
        size_mb=size_mb,
    )


def test_catalog_installed_status_uses_check_model():
    from revospeech.catalog import catalog_installed_status

    fake_model = MagicMock()
    fake_model.name = "fake-model"

    fake_status = MagicMock()
    fake_status.status = "ready"

    with (
        patch("revospeech.catalog.list_catalog", return_value=[fake_model]),
        patch("revospeech.catalog.check_model", return_value=fake_status),
    ):
        result = catalog_installed_status()

    assert result == {"fake-model": True}


def test_catalog_installed_status_handles_missing():
    from revospeech.catalog import catalog_installed_status

    fake_model = MagicMock()
    fake_model.name = "missing-model"

    with (
        patch("revospeech.catalog.list_catalog", return_value=[fake_model]),
        patch(
            "revospeech.catalog.check_model",
            side_effect=KeyError("not found"),
        ),
    ):
        result = catalog_installed_status()

    assert result == {"missing-model": False}


def test_catalog_installed_status_mixed():
    from revospeech.catalog import catalog_installed_status

    ready_model = MagicMock()
    ready_model.name = "ready-model"
    not_ready_model = MagicMock()
    not_ready_model.name = "not-ready-model"

    ready_status = MagicMock()
    ready_status.status = "ready"
    not_ready_status = MagicMock()
    not_ready_status.status = "needs-download"

    with (
        patch(
            "revospeech.catalog.list_catalog",
            return_value=[ready_model, not_ready_model],
        ),
        patch(
            "revospeech.catalog.check_model",
            side_effect=[ready_status, not_ready_status],
        ),
    ):
        result = catalog_installed_status()

    assert result == {"ready-model": True, "not-ready-model": False}


def test_recommend_models_filters_by_task():
    from revospeech.catalog import recommend_models

    asr_model = _make_manifest("asr-1", task="asr")
    tts_model = _make_manifest("tts-1", task="tts")

    with patch("revospeech.catalog.list_catalog", return_value=[asr_model, tts_model]):
        # recommend_models delegates task filtering to list_catalog, but
        # list_catalog is mocked here. Pass all and verify the mock passes through.
        result = recommend_models()

    names = [m.name for m in result]
    assert "asr-1" in names
    assert "tts-1" in names


def test_recommend_models_filters_by_task_via_mock():
    from revospeech.catalog import recommend_models

    asr_model = _make_manifest("asr-1", task="asr")
    # Simulate list_catalog filtering by task
    with patch("revospeech.catalog.list_catalog", return_value=[asr_model]) as mock_lc:
        result = recommend_models(task="asr")

    mock_lc.assert_called_once_with(task="asr")
    assert len(result) == 1
    assert result[0].name == "asr-1"


def test_recommend_models_sorts_by_size():
    from revospeech.catalog import recommend_models

    big = _make_manifest("big", size_mb=500)
    small = _make_manifest("small", size_mb=50)
    medium = _make_manifest("medium", size_mb=200)

    with patch("revospeech.catalog.list_catalog", return_value=[big, small, medium]):
        result = recommend_models()

    # Top 3, sorted by size ascending
    assert [m.name for m in result] == ["small", "medium", "big"]


def test_recommend_models_filters_by_language():
    from revospeech.catalog import recommend_models

    en_model = _make_manifest("en-1", task="asr", language="en", languages=["en"])
    fr_model = _make_manifest("fr-1", task="asr", language="fr", languages=["fr"])

    with patch("revospeech.catalog.list_catalog", return_value=[en_model, fr_model]):
        result = recommend_models(language="en")

    assert len(result) == 1
    assert result[0].name == "en-1"


def test_recommend_models_language_matches_via_language_field():
    from revospeech.catalog import recommend_models

    # languages list empty, but language field set
    en_model = _make_manifest("en-1", task="asr", language="en", languages=[])
    fr_model = _make_manifest("fr-1", task="asr", language="fr", languages=[])

    with patch("revospeech.catalog.list_catalog", return_value=[en_model, fr_model]):
        result = recommend_models(language="EN")  # case-insensitive

    assert len(result) == 1
    assert result[0].name == "en-1"


def test_recommend_models_limits_to_three():
    from revospeech.catalog import recommend_models

    # Start at 1 so no model has size_mb=0 (which sorts last as "unknown")
    models = [_make_manifest(f"m{i}", size_mb=(i + 1) * 100) for i in range(5)]

    with patch("revospeech.catalog.list_catalog", return_value=models):
        result = recommend_models()

    assert len(result) == 3
    # Should be the three smallest
    assert [m.name for m in result] == ["m0", "m1", "m2"]


def test_recommend_models_zero_size_sorted_last():
    from revospeech.catalog import recommend_models

    # size_mb=0 is treated as "unknown" and sorted last
    unknown = _make_manifest("unknown", size_mb=0.0)
    small = _make_manifest("small", size_mb=50)

    with patch("revospeech.catalog.list_catalog", return_value=[unknown, small]):
        result = recommend_models()

    assert [m.name for m in result] == ["small", "unknown"]
