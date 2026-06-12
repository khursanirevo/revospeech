"""Model manifest dataclass and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelManifest:
    """Describes a model's metadata and file layout.

    Loaded from YAML manifests in revos/models/ or ~/.config/revospeech/models/.
    """

    name: str
    task: str  # "asr" or "tts"
    backend: str  # e.g. "sherpa-onnx"
    model_type: str  # e.g. "transducer", "vits", "kokoro"
    model_url: str
    sample_rate: int
    language: str
    description: str
    files: dict[str, str] = field(default_factory=dict)
    hf_private: bool = False
    revision: str = ""
    mode: str = "local"  # "local" | "api"
    api_endpoint: str = ""  # URL for API backend
    size_mb: float = 0.0  # download size in MB
    capabilities: list[str] = field(default_factory=list)  # e.g. streaming
    languages: list[str] = field(default_factory=list)  # e.g. ["en", "zh"]
    tags: list[str] = field(default_factory=list)  # e.g. ["fast", "multilingual"]
    license: str = ""  # model weight license
    sha256: str = ""  # for download verification
    min_ram_mb: int = 0  # minimum RAM requirement
    min_vram_mb: int = 0  # minimum VRAM requirement

    @property
    def is_local(self) -> bool:
        """Return True if this model runs locally."""
        return self.mode == "local"

    @property
    def is_api(self) -> bool:
        """Return True if this model is served via API."""
        return self.mode == "api"


def load_manifest(path: Path) -> ModelManifest:
    """Load a model manifest from a YAML file.

    Args:
        path: Path to the YAML manifest file.

    Returns:
        Populated ModelManifest instance.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    return ModelManifest(
        name=data["name"],
        task=data["task"],
        backend=data["backend"],
        model_type=data.get("model_type", ""),
        model_url=data.get("model_url", ""),
        sample_rate=data.get("sample_rate", 16000),
        language=data.get("language", ""),
        description=data.get("description", ""),
        files=data.get("files", {}),
        hf_private=data.get("hf_private", False),
        revision=data.get("revision", ""),
        mode=data.get("mode", "local"),
        api_endpoint=data.get("api_endpoint", ""),
        size_mb=data.get("size_mb", 0.0),
        capabilities=data.get("capabilities", []),
        languages=data.get("languages", []),
        tags=data.get("tags", []),
        license=data.get("license", ""),
        sha256=data.get("sha256", ""),
        min_ram_mb=data.get("min_ram_mb", 0),
        min_vram_mb=data.get("min_vram_mb", 0),
    )
