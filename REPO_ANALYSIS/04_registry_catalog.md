# Registry & Catalog System — Detailed Analysis

## Overview

The registry and catalog subsystem provides three distinct capabilities:

1. **Model Manifest** (`manifest.py`) — a dataclass describing a model's metadata, loaded from YAML files.
2. **Model Registry** (`registry.py`) — an in-memory key-value store of manifests, auto-populated at import time from bundled and user directories.
3. **Model Downloader** (`downloader.py`) — downloads and caches model artifacts from URLs to the local filesystem.
4. **Remote Catalog** (`catalog.py`) — discovers and pulls YAML manifests from a GitHub-hosted repository, bridging remote discovery into the local registry.

---

## 1. File-by-File Analysis

---

### 1.1 `revos/registry/__init__.py`

**Purpose:** Public facade for the `revos.registry` package. Re-exports the five symbols that downstream consumers need.

**Re-exports:**

| Symbol | Source Module | Description |
|---|---|---|
| `ModelManifest` | `.manifest` | Dataclass for model metadata |
| `load_manifest` | `.manifest` | YAML -> ModelManifest loader |
| `register` | `.registry` | Add a manifest to the in-memory store |
| `get` | `.registry` | Look up a manifest by name + task |
| `list_models` | `.registry` | List registered manifests (optionally filtered) |
| `ensure_model` | `.downloader` | Download + cache a model if not present |

**`__all__` list:**
```python
__all__ = [
    "ModelManifest",
    "load_manifest",
    "register",
    "get",
    "list_models",
    "ensure_model",
]
```

**Design pattern:** Facade / re-export module. External code does `from revos.registry import get, ensure_model` without needing to know the internal module layout.

---

### 1.2 `revos/registry/manifest.py`

**Purpose:** Defines the `ModelManifest` dataclass and the YAML loader function. This is the single source of truth for what a model description looks like.

#### `ModelManifest` dataclass

```python
@dataclass
class ModelManifest:
    name: str                    # Unique model identifier, e.g. "zipformer-v2"
    task: str                    # Task domain: "asr" or "tts"
    backend: str                 # Engine/backend, e.g. "sherpa-onnx", "revovoice"
    model_type: str              # Architecture type, e.g. "transducer", "vits", "kokoro", "diffusion"
    model_url: str               # Download URL (http/https) or HF repo id
    sample_rate: int             # Audio sample rate in Hz (default: 16000)
    language: str                # Language code, e.g. "en", "multilingual"
    description: str             # Human-readable description
    files: dict[str, str]        # Logical name -> expected filename mapping
    hf_private: bool = False     # Whether model lives in a private HF repo
    revision: str = ""           # Pin to a specific commit hash or tag
```

**Field semantics:**

- `files` maps logical component names to physical filenames (e.g., `{"encoder": "encoder-epoch-99-avg-1.onnx"}`). The downloader uses these to verify that a cached model is complete and to locate model directory contents after extraction.
- `model_url` can be a direct HTTPS download link (for `.tar.bz2`, `.tar.gz`, `.zip` archives) or an HuggingFace repo identifier (e.g., `"Revolab/revovoice"`). When `hf_private=True`, the URL is treated as a private HF repo requiring authentication.
- `revision` is optional; when set, it pins the model to a specific commit hash or tag (relevant for HF-hosted models).

#### `load_manifest(path: Path) -> ModelManifest`

```python
def load_manifest(path: Path) -> ModelManifest:
```

Reads a YAML file, parses it with `yaml.safe_load`, and constructs a `ModelManifest`. Missing optional fields get defaults:
- `model_type` -> `""`
- `model_url` -> `""`
- `sample_rate` -> `16000`
- `language` -> `""`
- `description` -> `""`
- `files` -> `{}`
- `hf_private` -> `False`
- `revision` -> `""`

Only `name`, `task`, and `backend` are truly required (they have no defaults in `dataclass` fields, so `safe_load` must provide them or a `KeyError` will propagate).

---

### 1.3 `revos/registry/registry.py`

**Purpose:** In-memory registry that maps `(task, name)` tuples to `ModelManifest` instances. Auto-loads manifests from two directory trees on import.

#### Internal storage

```python
_models: dict[tuple[str, str], ModelManifest] = {}
```

Key is `(task: str, name: str)`. This means a model name must be unique within its task domain, but the same name could exist across different tasks.

#### `register(manifest: ModelManifest) -> None`

```python
def register(manifest: ModelManifest) -> None:
```

Inserts or overwrites the entry at key `(manifest.task, manifest.name)`. Logs at DEBUG level.

#### `get(name: str, task: str) -> ModelManifest`

```python
def get(name: str, task: str) -> ModelManifest:
```

Looks up by `(task, name)`. On miss, raises `KeyError` with a helpful message:
- If no models exist for the task at all: suggests adding a manifest in `~/.config/revos/models/{task}/`.
- Otherwise: lists all available model names for the task.

#### `list_models(task: str | None = None) -> list[ModelManifest]`

```python
def list_models(task: str | None = None) -> list[ModelManifest]:
```

Returns all manifests, optionally filtered by task. No ordering guarantee.

#### `_load_manifests_from_dir(directory: Path) -> None`

```python
def _load_manifests_from_dir(directory: Path) -> None:
```

Recursively scans `directory` for `*.yaml` and `*.yml` files using `rglob`. For each file, calls `load_manifest` then `register`. Failed loads are logged at WARNING level but do not halt scanning. This is fault-tolerant — one bad YAML does not prevent other manifests from loading.

#### `_load_builtin_manifests() -> None`

```python
def _load_builtin_manifests() -> None:
```

Loads from `revos/models/` (relative to the package source tree, computed as `Path(__file__).parent.parent / "models"`).

#### `_load_user_manifests() -> None`

```python
def _load_user_manifests() -> None:
```

Loads from `~/.config/revos/models/`.

#### Auto-load on import

```python
# Auto-load on import
_load_builtin_manifests()
_load_user_manifests()
```

Both are called at module level, meaning simply importing `revos.registry` (or any module that imports it) triggers manifest discovery and registration. The builtin directory is loaded first, then the user directory second, so user manifests can **override** builtins (since `register` overwrites on key collision).

**Design pattern:** Singleton registry with auto-discovery. Module-level side effects on import.

---

### 1.4 `revos/registry/downloader.py`

**Purpose:** Downloads model artifacts from URLs, extracts archives, and caches them on the local filesystem. Also verifies cached models are complete.

#### Constants

```python
CACHE_DIR = Path.home() / ".cache" / "revos"
```

All downloaded models are stored under `~/.cache/revos/<model_name>/`.

#### `_progress_hook(block_num: int, block_size: int, total_size: int) -> None`

A `urllib.request.urlretrieve` reporthook callback. Prints a 40-character ASCII progress bar to stderr with percentage and MB counters. Handles unknown total sizes (shows just MB downloaded).

#### `_download(url: str, dest: Path) -> None`

```python
def _download(url: str, dest: Path) -> None:
```

Validates the URL scheme is `https://` or `http://`, creates parent directories, then delegates to `urllib.request.urlretrieve` with the progress hook.

#### `_extract(archive_path: Path, dest_dir: Path) -> None`

```python
def _extract(archive_path: Path, dest_dir: Path) -> None:
```

Extracts archives based on extension:
- `.tar.bz2`, `.tar.gz`, `.tgz` -> `tarfile.open(...).extractall(dest_dir, filter="data")` (the `filter="data"` prevents path traversal in tar archives)
- `.zip` -> delegates to `_extract_zip_safe`
- Other -> plain `shutil.copy2` (treats as a single file)

#### `_extract_zip_safe(archive_path: Path, dest_dir: Path) -> None`

```python
def _extract_zip_safe(archive_path: Path, dest_dir: Path) -> None:
```

Extracts zip entries with path-traversal protection: each member's resolved destination is checked to ensure it starts with the resolved `dest_dir` prefix. Raises `ValueError` on unsafe paths.

#### `_find_model_dir(extract_dir: Path, manifest: ModelManifest) -> Path`

```python
def _find_model_dir(extract_dir: Path, manifest: ModelManifest) -> Path:
```

After extraction, archives may produce a subdirectory. This function locates the actual model directory:
1. Checks if all expected files exist directly in `extract_dir`.
2. Checks one level down (each immediate subdirectory).
3. Falls back to `extract_dir` if nothing matches.

#### `ensure_model(manifest: ModelManifest) -> Path`

```python
def ensure_model(manifest: ModelManifest) -> Path:
```

The main entry point. Flow:

1. **Cache check:** `CACHE_DIR / manifest.name` — if all files from `manifest.files.values()` exist, returns immediately.
2. **Download:** Validates `manifest.model_url` is non-empty, downloads to `model_dir / <archive_name>`.
3. **Extract (if archive):** Extracts to `model_dir / "_extracted"`, then moves files up to `model_dir` using `_find_model_dir`.
4. **Cleanup:** Removes `_extracted/` directory and the archive file.
5. **Return:** `model_dir` path.

Raises `ValueError` if `manifest.model_url` is empty.

---

### 1.5 `revos/catalog.py`

**Purpose:** Remote model catalog backed by a GitHub repository. Allows users to browse available models and pull new ones without upgrading the package.

#### Constants

```python
DEFAULT_CATALOG_REPO = "khursanirevo/revos"
_GITHUB_API = "https://api.github.com/repos"
_USER_MODELS_DIR = Path.home() / ".config" / "revos" / "models"
```

#### `get_catalog_repo() -> str`

```python
def get_catalog_repo() -> str:
```

Returns the catalog repository in `owner/repo` format. Resolution order:
1. `REVOS_CATALOG_REPO` environment variable
2. `~/.config/revos/config.yaml` -> `catalog_repo` key
3. Default: `"khursanirevo/revos"`

#### `_github_api_get(url: str) -> bytes`

```python
def _github_api_get(url: str) -> bytes:
```

Low-level GitHub API request helper. Sets `Accept: application/vnd.github.v3+json` and `User-Agent: revos-catalog` headers. Uses `urllib.request.urlopen` (no external HTTP dependencies).

#### `_list_yaml_files(repo: str, path: str) -> list[str]`

```python
def _list_yaml_files(repo: str, path: str) -> list[str]:
```

Recursively lists YAML files under a GitHub repo directory via the Contents API. Returns file paths relative to repo root (e.g., `"revos/models/asr/zipformer_v2.yaml"`). Recurses into subdirectories (like `asr/` and `tts/`).

#### `_download_raw(repo: str, path: str) -> str`

```python
def _download_raw(repo: str, path: str) -> str:
```

Downloads a file from GitHub as raw text via `raw.githubusercontent.com/{repo}/HEAD/{path}`. Returns UTF-8 decoded string.

#### `list_catalog(task: str | None = None) -> list[ModelManifest]`

```python
def list_catalog(task: str | None = None) -> list[ModelManifest]:
```

Fetches all YAML manifests from the remote catalog:
1. Gets the catalog repo from `get_catalog_repo()`.
2. Calls `_list_yaml_files` to discover all YAML files under `revos/models/`.
3. Optionally filters by task (e.g., only files under `revos/models/asr/`).
4. For each YAML file: downloads raw content, writes to a temp file, parses via `load_manifest`, cleans up temp file.
5. Returns list of `ModelManifest` instances.

Raises `RuntimeError` if the GitHub API call fails (network error, repo not found, etc.). Failed individual manifest loads are logged at WARNING but skipped.

#### `pull_model(name: str) -> Path`

```python
def pull_model(name: str) -> Path:
```

Downloads a single model manifest from the remote catalog and installs it locally:
1. Fetches all YAML files from the remote catalog.
2. Iterates through each, downloading and parsing to find the one with `manifest.name == name`.
3. Raises `KeyError` if not found.
4. Installs to `~/.config/revos/models/{task}/{filename}` — creates directories as needed.
5. Downloads the raw YAML content again and writes to the destination path.
6. Parses the installed file via `load_manifest` and calls `register()` to add it to the live in-memory registry.
7. Returns the installed `Path`.

Uses a deferred import (`from .registry import register`) to avoid circular imports at module level.

**Important note:** `pull_model` only downloads the **manifest YAML** file, not the actual model artifacts. The model files are downloaded later when `ensure_model` is called (typically at inference time).

---

### 1.6 `revos/models/asr/zipformer_v2.yaml`

```yaml
name: zipformer-v2
task: asr
backend: sherpa-onnx
model_type: transducer
model_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2"
sample_rate: 16000
language: en
description: "Zipformer small English transducer ASR model via sherpa-onnx"
files:
  encoder: "encoder-epoch-99-avg-1.onnx"
  decoder: "decoder-epoch-99-avg-1.onnx"
  joiner: "joiner-epoch-99-avg-1.onnx"
  tokens: "tokens.txt"
```

**Schema analysis:**
- Direct HTTPS download URL pointing to a `.tar.bz2` archive on GitHub releases.
- `files` maps four logical ONNX components to their expected filenames. The downloader verifies all four exist after extraction to confirm a valid cache.
- No `hf_private` or `revision` fields (they default to `False` and `""`).

---

### 1.7 `revos/models/tts/revovoice.yaml`

```yaml
name: revovoice
task: tts
backend: revovoice
model_type: diffusion
model_url: "Revolab/revovoice"
# revision: "a1b2c3d"    # Pin to a specific commit hash or tag
sample_rate: 24000
language: multilingual
description: "RevoVoice multilingual zero-shot TTS with voice cloning and design (600+ languages)"
hf_private: true
files: {}
```

**Schema analysis:**
- `model_url` is an HuggingFace repo identifier (`"Revolab/revovoice"`), not a direct download link. The `revovoice` backend engine handles HF-specific download logic internally.
- `hf_private: true` signals that this repo requires authentication.
- `revision` is commented out as an example — when uncommented, it would pin to a specific commit.
- `files` is empty (`{}`), meaning the downloader's cache verification is effectively skipped (no expected files to check). The revovoice engine manages its own file layout.

---

## 2. The YAML Model Config Schema

A model manifest YAML must contain at minimum: `name`, `task`, `backend`. All other fields have defaults.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | `str` | Yes | — | Unique identifier within task |
| `task` | `str` | Yes | — | `"asr"` or `"tts"` |
| `backend` | `str` | Yes | — | Engine name (e.g., `"sherpa-onnx"`, `"revovoice"`) |
| `model_type` | `str` | No | `""` | Architecture type (e.g., `"transducer"`, `"diffusion"`) |
| `model_url` | `str` | No | `""` | Download URL or HF repo id |
| `sample_rate` | `int` | No | `16000` | Audio sample rate in Hz |
| `language` | `str` | No | `""` | Language code or `"multilingual"` |
| `description` | `str` | No | `""` | Human-readable description |
| `files` | `dict[str, str]` | No | `{}` | Logical name -> expected filename |
| `hf_private` | `bool` | No | `False` | Whether model is in a private HF repo |
| `revision` | `str` | No | `""` | Pin to commit hash or tag |

---

## 3. How the Model Registry Works

### Registration

```python
manifest = load_manifest(Path("model.yaml"))  # Parse YAML
register(manifest)                              # Store in _models dict
```

The key is `(manifest.task, manifest.name)`. Registration overwrites on collision, so later registrations win. User manifests loaded after builtins take precedence.

### Lookup

```python
manifest = get("zipformer-v2", "asr")
```

Raises `KeyError` with a helpful message listing available alternatives if not found.

### Listing

```python
all_models = list_models()           # Everything
asr_models = list_models(task="asr") # Filtered by task
```

### Auto-Discovery

On `import revos.registry` (or any import chain that touches it), the module:
1. Scans `revos/models/` (bundled with the package) recursively for `.yaml`/`.yml` files.
2. Scans `~/.config/revos/models/` recursively for user-added manifests.
3. Each found file is parsed and registered.

---

## 4. How Model Downloading Works

### Sources

- **HTTPS URLs** — direct download of `.tar.bz2`, `.tar.gz`, `.tgz`, `.zip` archives, or single files.
- **HuggingFace repos** — identified by `model_url` as a repo id (e.g., `"Revolab/revovoice"`). Download is handled by the backend engine, not by `downloader.py` directly.

### Caching

All downloads go to `~/.cache/revos/<model_name>/`. The cache is considered valid when all files listed in `manifest.files` exist in that directory.

### Progress

`_progress_hook` displays a real-time ASCII progress bar on stderr during `urllib.request.urlretrieve` calls.

### Extraction

Archives are extracted to a temporary `_extracted/` subdirectory, then files are moved up to the model root. This handles archives that bundle files inside a top-level subdirectory. After extraction, the temporary directory and archive file are cleaned up.

### Security

- URL scheme validation (only `http://` and `https://`).
- Tar extraction uses `filter="data"` (Python 3.12+ path traversal protection).
- Zip extraction checks that every member resolves inside the destination directory.

---

## 5. How the Catalog System Works

### Remote Catalog Source

The catalog lives in the `revos/models/` directory of a GitHub repository (default: `khursanirevo/revos`). Team members add YAML manifests to the repo, and users discover them via the GitHub Contents API.

### Configuration

The catalog repo is resolved in priority order:
1. `REVOS_CATALOG_REPO` environment variable
2. `~/.config/revos/config.yaml` -> `catalog_repo` key
3. Hardcoded default: `khursanirevo/revos`

### Browse (`list_catalog`)

1. Queries GitHub Contents API for `revos/models/` directory.
2. Recursively discovers all `.yaml`/`.yml` files.
3. Downloads each file's raw content, parses into `ModelManifest`.
4. Returns the list (optionally filtered by task).

### Pull (`pull_model`)

1. Downloads all catalog YAML files and searches for one with matching `name`.
2. Installs the YAML to `~/.config/revos/models/{task}/{filename}`.
3. Registers the manifest in the live registry (so it's immediately usable without restarting).
4. Returns the installed file path.

**Key insight:** `pull_model` only installs the manifest, not the model artifacts. The actual model download happens lazily when `ensure_model` is called at inference time.

---

## 6. The Full Flow: Catalog Browse -> Pull -> Download -> Register

```
User runs: revos catalog list
    |
    v
list_catalog() -> GitHub API -> discovers YAML files -> parses manifests -> prints table
    |
User runs: revos catalog pull revovoice
    |
    v
pull_model("revovoice")
    -> fetches all catalog YAMLs from GitHub
    -> finds manifest with name=="revovoice"
    -> installs YAML to ~/.config/revos/models/tts/revovoice.yaml
    -> register(manifest)  # adds to in-memory _models dict
    |
User runs: revos tts -m revovoice -t "Hello"
    |
    v
get("revovoice", "tts")           # lookup in _models -> returns manifest
    |
    v
ensure_model(manifest)            # download artifacts if not cached
    -> check ~/.cache/revos/revovoice/ for expected files
    -> if missing: download from model_url
    -> extract archive -> move to cache dir -> cleanup
    -> return model directory path
    |
    v
TTS engine loads model files from returned path
    -> synthesizes audio -> outputs result
```

---

## 7. Design Patterns and Architecture Notes

### Patterns Used

1. **Singleton module-level registry** — `_models` dict at module scope, populated on import. No class needed; functions operate on module state.
2. **Auto-discovery on import** — manifest directories are scanned when `registry.py` is imported, making the registry "just work" without explicit initialization.
3. **Facade pattern** — `registry/__init__.py` re-exports a clean public API, hiding the internal module structure.
4. **Lazy import** — `catalog.py` uses `from .registry import register` inside `pull_model()` to avoid circular imports (catalog imports from registry.manifest, but needs registry.register at runtime).
5. **Cache-then-compute** — `ensure_model` checks for existing files before downloading; `list_catalog` is always remote (no local cache of the catalog itself).
6. **Fault-tolerant loading** — both `_load_manifests_from_dir` and `list_catalog` skip individual failures without aborting the entire operation.

### Data Flow Summary

```
YAML file --[load_manifest]--> ModelManifest --[register]--> _models dict
                                                              |
get(name, task) ----------------------------------------------> ModelManifest
                                                              |
ensure_model(manifest) ---------------------------------------> Path (cache dir)

GitHub repo --[list_catalog]--> [ModelManifest, ...]  (browse, no side effects)
GitHub repo --[pull_model]----> installed YAML + register()  (install + register)
```

### Key Filesystem Paths

| Path | Purpose |
|---|---|
| `revos/models/**/*.yaml` | Bundled manifests (shipped with the package) |
| `~/.config/revos/models/**/*.yaml` | User-installed manifests (from `pull_model` or manual) |
| `~/.config/revos/config.yaml` | User configuration (catalog_repo override) |
| `~/.cache/revos/<model_name>/` | Downloaded and extracted model artifacts |

### Consumers

- `revos/asr/__init__.py` — calls `get()` to resolve the ASR model manifest
- `revos/tts/revovoice_engine.py` — calls `get()` to resolve the TTS model manifest
- `revos/cli/main.py` — calls `list_models()`, `list_catalog()`, `pull_model()`, and `get_catalog_repo()` for the CLI commands
