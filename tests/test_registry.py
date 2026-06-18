"""Tests for the model registry."""

from pathlib import Path

import pytest
import yaml

from revospeech.registry.manifest import ModelManifest, load_manifest
from revospeech.registry.registry import _models, get, list_models, register


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the registry before each test."""
    _models.clear()
    yield
    _models.clear()


def test_register_and_get():
    m = ModelManifest(
        name="test-model",
        task="asr",
        backend="sherpa-onnx",
        model_type="transducer",
        model_url="http://example.com/model.tar.bz2",
        sample_rate=16000,
        language="en",
        description="Test model",
        files={"encoder": "encoder.onnx"},
    )
    register(m)
    assert get("test-model", "asr") is m


def test_get_missing_raises():
    with pytest.raises(KeyError, match="not found"):
        get("nonexistent", "asr")


def test_list_models_all():
    register(
        ModelManifest(
            name="a",
            task="asr",
            backend="x",
            model_type="transducer",
            model_url="",
            sample_rate=16000,
            language="en",
            description="",
        )
    )
    register(
        ModelManifest(
            name="b",
            task="tts",
            backend="x",
            model_type="vits",
            model_url="",
            sample_rate=24000,
            language="en",
            description="",
        )
    )
    assert len(list_models()) == 2
    assert len(list_models("asr")) == 1
    assert len(list_models("tts")) == 1
    assert list_models("asr")[0].name == "a"


def test_load_manifest(tmp_path: Path):
    manifest_data = {
        "name": "test-asr",
        "task": "asr",
        "backend": "sherpa-onnx",
        "model_type": "transducer",
        "model_url": "http://example.com/model.tar.bz2",
        "sample_rate": 16000,
        "language": "en",
        "description": "Test ASR model",
        "files": {"encoder": "encoder.onnx", "decoder": "decoder.onnx"},
    }
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml.dump(manifest_data))

    m = load_manifest(yaml_file)
    assert m.name == "test-asr"
    assert m.task == "asr"
    assert m.backend == "sherpa-onnx"
    assert m.sample_rate == 16000
    assert m.files["encoder"] == "encoder.onnx"


def test_register_overwrites():
    m1 = ModelManifest(
        name="x",
        task="asr",
        backend="a",
        model_type="transducer",
        model_url="",
        sample_rate=16000,
        language="en",
        description="v1",
    )
    m2 = ModelManifest(
        name="x",
        task="asr",
        backend="b",
        model_type="transducer",
        model_url="",
        sample_rate=16000,
        language="en",
        description="v2",
    )
    register(m1)
    register(m2)
    assert get("x", "asr").description == "v2"


# ---------------------------------------------------------------------------
# _load_manifests_from_dir — directory scanning + error tolerance
# ---------------------------------------------------------------------------
def test_load_manifests_from_dir_loads_yaml(tmp_path):
    """Valid YAML manifests in a directory are loaded."""
    from revospeech.registry.registry import _load_manifests_from_dir

    manifest_data = {
        "name": "from-disk",
        "task": "asr",
        "backend": "sherpa-onnx",
        "model_type": "transducer",
        "model_url": "http://example.com/m.tar.bz2",
        "sample_rate": 16000,
        "language": "en",
        "description": "",
    }
    (tmp_path / "good.yaml").write_text(yaml.dump(manifest_data))

    _load_manifests_from_dir(tmp_path)
    assert get("from-disk", "asr").sample_rate == 16000


def test_load_manifests_from_dir_loads_yml(tmp_path):
    """.yml extension is also picked up."""
    from revospeech.registry.registry import _load_manifests_from_dir

    manifest_data = {
        "name": "yml-model",
        "task": "asr",
        "backend": "sherpa-onnx",
        "model_type": "transducer",
        "model_url": "",
        "sample_rate": 16000,
        "language": "en",
        "description": "",
    }
    (tmp_path / "model.yml").write_text(yaml.dump(manifest_data))

    _load_manifests_from_dir(tmp_path)
    assert get("yml-model", "asr").backend == "sherpa-onnx"


def test_load_manifests_from_dir_skips_broken(tmp_path):
    """Broken YAML is logged and skipped, valid sibling is still loaded."""
    from revospeech.registry.registry import _load_manifests_from_dir

    (tmp_path / "broken.yaml").write_text("not: valid: yaml: [")

    good = {
        "name": "sibling",
        "task": "asr",
        "backend": "sherpa-onnx",
        "model_type": "transducer",
        "model_url": "",
        "sample_rate": 16000,
        "language": "en",
        "description": "",
    }
    (tmp_path / "good.yaml").write_text(yaml.dump(good))

    _load_manifests_from_dir(tmp_path)
    assert get("sibling", "asr").sample_rate == 16000


def test_load_manifests_from_dir_missing_dir_is_noop(tmp_path):
    """Non-existent directory is silently skipped."""
    from revospeech.registry.registry import _load_manifests_from_dir

    _load_manifests_from_dir(tmp_path / "does-not-exist")
    assert list_models() == []
