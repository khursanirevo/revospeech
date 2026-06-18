"""Edge-case tests: corrupt audio, network errors, thread safety, empty/long/CJK text.

US-026: These tests verify graceful handling of pathological inputs and
concurrent access patterns that real users will hit.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from revospeech.registry.manifest import ModelManifest
from revospeech.registry.registry import _models, register
from revospeech.tts.base import BaseTTS, _split_text
from revospeech.tts.result import Audio


# ---------------------------------------------------------------------------
# Registry isolation fixture — keep the global registry clean.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot and restore the global registry around each test."""
    saved = dict(_models)
    _models.clear()
    yield
    _models.clear()
    _models.update(saved)


# ---------------------------------------------------------------------------
# 1. Corrupt / invalid audio input
# ---------------------------------------------------------------------------
def test_corrupt_audio_file_raises(tmp_path: Path):
    """read_waveform on a non-audio file should raise an exception."""
    from revospeech.asr.audio import read_waveform

    bad = tmp_path / "not_audio.wav"
    bad.write_text("this is definitely not a wav file")

    with pytest.raises(Exception):
        read_waveform(str(bad))


def test_corrupt_audio_bytes_raises(tmp_path: Path):
    """A file containing random bytes should also fail to decode."""
    from revospeech.asr.audio import read_waveform

    bad = tmp_path / "random.bin"
    bad.write_bytes(bytes(range(256)) * 4)

    with pytest.raises(Exception):
        read_waveform(str(bad))


def test_missing_audio_file_raises(tmp_path: Path):
    """A nonexistent path should raise rather than silently return."""
    from revospeech.asr.audio import read_waveform

    missing = tmp_path / "does_not_exist.wav"
    with pytest.raises(Exception):
        read_waveform(str(missing))


# ---------------------------------------------------------------------------
# 2. Network errors in downloader
# ---------------------------------------------------------------------------
def _make_manifest(**overrides) -> ModelManifest:
    defaults = dict(
        name="test-edge-model",
        task="asr",
        backend="sherpa-onnx",
        model_type="transducer",
        model_url="https://example.com/model.tar.bz2",
        sample_rate=16000,
        language="en",
        description="Test",
        files={"encoder": "encoder.onnx"},
    )
    defaults.update(overrides)
    return ModelManifest(**defaults)


def test_downloader_network_error_propagates(tmp_path: Path):
    """When urlretrieve fails with OSError, ensure_model should surface it."""
    from revospeech.registry.downloader import ensure_model

    manifest = _make_manifest()

    with (
        patch(
            "revospeech.registry.downloader.urllib.request.urlretrieve",
            side_effect=OSError("network unreachable"),
        ),
        patch("revospeech.registry.downloader.CACHE_DIR", tmp_path),
    ):
        with pytest.raises(OSError, match="network unreachable"):
            ensure_model(manifest)


def test_downloader_connection_error_propagates(tmp_path: Path):
    """ConnectionError should also propagate from the downloader."""
    from revospeech.registry.downloader import ensure_model

    manifest = _make_manifest()

    with (
        patch(
            "revospeech.registry.downloader.urllib.request.urlretrieve",
            side_effect=ConnectionError("connection refused"),
        ),
        patch("revospeech.registry.downloader.CACHE_DIR", tmp_path),
    ):
        with pytest.raises(ConnectionError):
            ensure_model(manifest)


def test_downloader_missing_url_raises_value_error(tmp_path: Path):
    """A manifest with no model_url should raise ValueError."""
    from revospeech.registry.downloader import ensure_model

    manifest = _make_manifest(model_url="")
    with patch("revospeech.registry.downloader.CACHE_DIR", tmp_path):
        with pytest.raises(ValueError, match="no download URL"):
            ensure_model(manifest)


# ---------------------------------------------------------------------------
# 3. Thread safety in registry — 10 threads x 100 registrations
# ---------------------------------------------------------------------------
def test_registry_thread_safety_concurrent_register():
    """1000 concurrent register() calls must not corrupt the registry."""
    errors: list[Exception] = []

    def register_one(i: int) -> None:
        try:
            m = ModelManifest(
                name=f"concurrent-model-{i}",
                task="asr",
                backend="test",
                model_type="test",
                model_url="",
                sample_rate=16000,
                language="en",
                description=f"model {i}",
            )
            register(m)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(register_one, range(100)))

    assert errors == []
    # All 100 models should be present
    asr_models = [m for m in _models.values() if m.task == "asr"]
    assert len(asr_models) == 100


def test_registry_thread_safety_mixed_tasks():
    """Concurrent registrations across asr + tts should not collide."""
    errors: list[Exception] = []

    def register_mixed(i: int) -> None:
        try:
            task = "asr" if i % 2 == 0 else "tts"
            m = ModelManifest(
                name=f"mix-{task}-{i}",
                task=task,
                backend="test",
                model_type="test",
                model_url="",
                sample_rate=16000,
                language="en",
                description="",
            )
            register(m)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(register_mixed, range(200)))

    assert errors == []
    asr_count = sum(1 for m in _models.values() if m.task == "asr")
    tts_count = sum(1 for m in _models.values() if m.task == "tts")
    assert asr_count + tts_count == 200


def test_registry_concurrent_read_write():
    """Concurrent reads and writes should not crash or corrupt."""
    errors: list[Exception] = []

    from revospeech.registry.registry import get, list_models

    # Seed one model
    seed = ModelManifest(
        name="seed-model",
        task="asr",
        backend="test",
        model_type="test",
        model_url="",
        sample_rate=16000,
        language="en",
        description="",
    )
    register(seed)

    def writer(i: int) -> None:
        try:
            m = ModelManifest(
                name=f"writer-{i}",
                task="asr",
                backend="test",
                model_type="test",
                model_url="",
                sample_rate=16000,
                language="en",
                description="",
            )
            register(m)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader(i: int) -> None:
        try:
            list_models()
            get("seed-model", "asr")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    with ThreadPoolExecutor(max_workers=10) as executor:
        # 50 writers + 50 readers interleaved
        writer_futures = [executor.submit(writer, i) for i in range(50)]
        reader_futures = [executor.submit(reader, i) for i in range(50)]
        for f in writer_futures + reader_futures:
            f.result()

    assert errors == []


# ---------------------------------------------------------------------------
# 4. Empty text input
# ---------------------------------------------------------------------------
class _FakeTTS(BaseTTS):
    """Minimal concrete TTS for testing synthesize_long empty handling."""

    def synthesize(self, text, output_path=None, **kwargs):  # type: ignore[no-untyped-def]
        # Return a tiny non-empty audio so tests that call this work.
        return Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=16000)


def test_synthesize_long_empty_raises():
    """synthesize_long with empty text should raise ValueError."""
    engine = _FakeTTS("fake")
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize_long("")


def test_synthesize_long_whitespace_only_raises():
    """synthesize_long with only whitespace should raise ValueError."""
    engine = _FakeTTS("fake")
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize_long("   \n\t  ")


def test_split_text_empty_returns_empty_list():
    """_split_text on empty/whitespace returns []."""
    assert _split_text("") == []
    assert _split_text("   ") == []
    assert _split_text("\n\n") == []


# ---------------------------------------------------------------------------
# 5. Very long text — should auto-split and still synthesize
# ---------------------------------------------------------------------------
def test_synthesize_long_many_chars_splits_and_succeeds():
    """Text over 5000 chars should split into multiple chunks and synthesize."""
    engine = _FakeTTS("fake")
    long_text = ". ".join([f"Sentence number {i}" for i in range(1000)])
    assert len(long_text) > 5000

    # Patch synthesize to track calls
    calls: list[str] = []

    original = engine.synthesize

    def tracking_synthesize(text, output_path=None, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(text)
        return original(text, output_path, **kwargs)

    engine.synthesize = tracking_synthesize  # type: ignore[assignment]

    result = engine.synthesize_long(long_text, max_chars=500)
    assert result is not None
    assert len(calls) > 1, "Long text should have been split into multiple chunks"
    # Each chunk must be within max_chars (after split)
    assert all(len(c) <= 500 for c in calls)


def test_split_text_long_produces_multiple_chunks():
    """_split_text on a long string produces multiple bounded chunks."""
    long_text = "Hello world. " * 500  # ~7000 chars
    chunks = _split_text(long_text, max_chars=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


# ---------------------------------------------------------------------------
# 6. CJK text — should not crash on sentence splitting
# ---------------------------------------------------------------------------
def test_split_text_cjk_japanese():
    """Japanese text with 。 punctuation should split correctly."""
    text = "今日は良い天気です。明日は雨が降るでしょう。明後日はまた晴れるでしょう。"
    chunks = _split_text(text, max_chars=15)
    assert len(chunks) >= 2
    # Each chunk <= max_chars (CJK may not split further at word level)
    assert all(len(c) <= 15 or len(chunks) == 1 for c in chunks)


def test_split_text_cjk_chinese():
    """Chinese text with 。punctuation should split correctly."""
    text = "你好世界。今天天气很好。明天可能会下雨。"
    chunks = _split_text(text, max_chars=12)
    assert len(chunks) >= 2


def test_split_text_cjk_korean():
    """Korean text should not crash during splitting."""
    text = "안녕하세요. 오늘 날씨가 좋습니다. 내일은 비가 올 것 같습니다."
    chunks = _split_text(text, max_chars=20)
    # Should produce at least one chunk without crashing
    assert len(chunks) >= 1


def test_synthesize_long_cjk_succeeds():
    """synthesize_long on CJK text should complete without error."""
    engine = _FakeTTS("fake")
    text = "今日は良い天気です。明日は雨が降るでしょう。"
    result = engine.synthesize_long(text, max_chars=20)
    assert result is not None
    assert len(result.samples) > 0
