"""Model status and discovery helpers."""
from __future__ import annotations

from dataclasses import dataclass

from revos.config import get_api_key
from revos.registry.downloader import CACHE_DIR
from revos.registry.manifest import ModelManifest
from revos.registry.registry import get as _get_manifest
from revos.registry.registry import list_models as _list_manifests


@dataclass
class ModelStatus:
    name: str
    task: str
    mode: str  # "local" | "api"
    status: str  # "ready" | "needs-download" | "needs-api-key"
    installed: bool
    size_mb: float | None
    capabilities: list[str]
    languages: list[str]

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"

    @property
    def is_local(self) -> bool:
        return self.mode == "local"

    @property
    def is_api(self) -> bool:
        return self.mode == "api"


def _compute_status(manifest: ModelManifest) -> ModelStatus:
    """Compute the status of a model from its manifest."""
    if manifest.is_api:
        has_key = get_api_key() is not None
        return ModelStatus(
            name=manifest.name,
            task=manifest.task,
            mode=manifest.mode,
            status="ready" if has_key else "needs-api-key",
            installed=has_key,
            size_mb=manifest.size_mb if manifest.size_mb > 0 else None,
            capabilities=manifest.capabilities,
            languages=manifest.languages,
        )

    # Local model — check if files are cached
    model_dir = CACHE_DIR / manifest.name
    if manifest.files:
        installed = all((model_dir / f).exists() for f in manifest.files.values())
    else:
        # No files listed (e.g., revovoice manages its own download)
        installed = model_dir.exists()

    return ModelStatus(
        name=manifest.name,
        task=manifest.task,
        mode=manifest.mode,
        status="ready" if installed else "needs-download",
        installed=installed,
        size_mb=manifest.size_mb if manifest.size_mb > 0 else None,
        capabilities=manifest.capabilities,
        languages=manifest.languages,
    )


def check_model(name: str, task: str | None = None) -> ModelStatus:
    """Check the status of a specific model.

    When *task* is provided, look up the model directly. Otherwise iterate
    all registered manifests and return the first one whose name matches.
    """
    if task:
        manifest = _get_manifest(name, task)
    else:
        # Without a task hint we must search across all registered models.
        matches = [m for m in _list_manifests() if m.name == name]
        if not matches:
            raise KeyError(f"Model '{name}' not found in any task category")
        if len(matches) > 1:
            names = [f"{m.task}/{m.name}" for m in matches]
            raise KeyError(
                f"Model '{name}' is ambiguous — matches: {names}. "
                f"Specify the task argument to disambiguate."
            )
        manifest = matches[0]
    return _compute_status(manifest)


def list_model_statuses(
    task: str | None = None,
    mode: str | None = None,
    language: str | None = None,
    status: str | None = None,
    capability: str | None = None,
) -> list[ModelStatus]:
    """List models with optional filters. Returns ModelStatus objects."""
    manifests = _list_manifests(task=task)
    results = []
    for m in manifests:
        ms = _compute_status(m)
        if mode and ms.mode != mode:
            continue
        if language and language not in ms.languages:
            continue
        if status and ms.status != status:
            continue
        if capability and capability not in ms.capabilities:
            continue
        results.append(ms)
    return results


__all__ = ["ModelStatus", "check_model", "list_model_statuses"]
