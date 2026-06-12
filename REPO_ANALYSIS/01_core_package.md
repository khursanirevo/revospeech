# Core Package Analysis: `revos/`

This document covers the foundational modules of the **revos** package — the top-level `__init__.py`, logging configuration, device detection, usage tracking, and the PEP 561 type marker. It also includes a full breakdown of `pyproject.toml`.

---

## Table of Contents

1. [Package Overview (`pyproject.toml`)](#1-package-overview-pyprojecttoml)
2. [`revos/__init__.py`](#2-revos__init__py)
3. [`revos/logging_config.py`](#3-revoslogging_configpy)
4. [`revos/device.py`](#4-revosdevicepy)
5. [`revos/usage.py`](#5-revosusagepy)
6. [`revos/py.typed`](#6-revospytyped)
7. [Cross-Cutting Design Patterns](#7-cross-cutting-design-patterns)

---

## 1. Package Overview (`pyproject.toml`)

**File**: `/mnt/data/work/revos/pyproject.toml`

### Build System

| Field | Value |
|---|---|
| Build backend | `hatchling` (via `[build-system]`) |
| Build requirement | `hatchling` |

### Package Metadata

| Field | Value |
|---|---|
| Name | `revos` |
| Version | `0.1.0` |
| Description | "A unified Python library for speech AI — ASR and TTS using open models" |
| License | MIT |
| Minimum Python | `>=3.11` |
| Author | RevoS Team |

### Core Dependencies

| Package | Constraint | Purpose |
|---|---|---|
| `sherpa-onnx` | `>=1.10` | Primary inference engine for ASR and TTS |
| `sherpa-onnx-core` | (unpinned) | Core sherpa-onnx runtime |
| `onnxruntime` | `>=1.16` | ONNX model execution backend |
| `numpy` | (unpinned) | Numerical operations |
| `soundfile` | (unpinned) | Audio file I/O |
| `click` | `>=8.0` | CLI framework |
| `pyyaml` | (unpinned) | Configuration file parsing |
| `huggingface-hub` | `>=1.11.0` | Model catalog and download from HuggingFace |

### Optional Dependency Groups

| Group | Packages | Notes |
|---|---|---|
| `gpu` | `onnxruntime-gpu` | GPU-accelerated ONNX inference |
| `tts` | `omnivoice` | Additional TTS voice support |
| `all` | `onnxruntime-gpu`, `omnivoice` | Convenience: installs GPU + TTS |
| `dev` | `pytest>=7.0`, `pytest-cov`, `ruff` | Development tooling |

### Entry Points

| Type | Name | Target |
|---|---|---|
| Console script | `revos` | `revos.cli.main:cli` |
| Entry point group | `revos.models` | (empty — reserved for future plugin system) |

### Tool Configuration

- **pytest**: Test path is `tests/`; defines a `slow` marker for tests requiring model download and real inference.
- **ruff**: Targets Python 3.11, line length 88, lint rules `E`, `F`, `I`, `W` (pycodestyle errors, pyflakes, isort, whitespace warnings).
- **hatch**: Wheel build includes the `revos` package.

### Classifiers

Indicates alpha status (`Development Status :: 3 - Alpha`), supports Python 3.11/3.12/3.13, categorized under Speech/Audio and AI topics.

---

## 2. `revos/__init__.py`

**File**: `/mnt/data/work/revos/revos/__init__.py`

### Purpose

Top-level package initializer. Provides the public face of the `revos` library with a **lazy import** mechanism so that importing `revos` itself is near-instant — submodules (`asr`, `tts`, `logging_config`) are only loaded when their symbols are actually accessed.

### Key Elements

#### `__version__: str`
- Hard-coded version string `"0.1.0"`.
- Must be kept in sync with `pyproject.toml` version.

#### `__getattr__(name: str)`
Module-level `__getattr__` implementing **PEP 562** (module `__getattr__` and `__dir__`). Provides lazy re-exports for the following names:

| Name | Source | Description |
|---|---|---|
| `ASR` | `revos.asr.ASR` | Automatic Speech Recognition engine |
| `TTS` | `revos.tts.TTS` | Text-to-Speech engine |
| `configure_logging` | `revos.logging_config.configure_logging` | Logging verbosity control |

If an unrecognized name is requested, raises `AttributeError` with a descriptive message.

### Design Decisions

1. **Lazy imports via `__getattr__`**: Avoids eagerly importing `sherpa-onnx` and `onnxruntime` (which are heavy C-extension packages) at `import revos` time. Users pay load cost only for the components they use.
2. **No `__all__` defined**: The public API is implicitly defined by the `__getattr__` dispatch. Tools relying on `__all__` for introspection will not see the re-exports. This is a minor ergonomic trade-off.
3. **No `__dir__` override**: PEP 562 also allows `__dir__` for tab-completion support, but it is not implemented here. This means `revos.<TAB>` in a REPL will not suggest `ASR`, `TTS`, or `configure_logging`.

### Dependencies

None (stdlib only). Submodule imports are deferred to `__getattr__` calls.

### Public API Surface

```python
revos.__version__          # str: "0.1.0"
revos.ASR                  # -> revos.asr.ASR (lazy)
revos.TTS                  # -> revos.tts.TTS (lazy)
revos.configure_logging    # -> revos.logging_config.configure_logging (lazy)
```

---

## 3. `revos/logging_config.py`

**File**: `/mnt/data/work/revos/revos/logging_config.py`

### Purpose

Provides a single function to configure the `revos` package's logging verbosity. This is the sanctioned way for users to control log output from all `revos` submodules.

### Key Elements

#### `configure_logging(level: str = "WARNING") -> None`

Configures the `"revos"` logger (and all its children, e.g., `revos.asr`, `revos.tts`) to the specified level.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `level` | `str` | `"WARNING"` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

**Behavior:**
1. Resolves the string to a `logging` module constant via `getattr(logging, level.upper())`.
2. Validates that the resolved value is an `int` (i.e., a valid log level). If not, raises `ValueError` with a helpful message.
3. Sets the `"revos"` logger level.
4. If no handlers exist on the logger, creates a new `StreamHandler` with a formatted output:
   ```
   2026-06-12 10:30:00,123 - revos - INFO - message
   ```
5. If handlers already exist, updates all existing handlers to the new level (idempotent reconfiguration).

**Raises:**

| Exception | Condition |
|---|---|
| `ValueError` | Invalid log level string |

### Dependencies

- `logging` (stdlib)

### Public API Surface

```python
from revos.logging_config import configure_logging
configure_logging("DEBUG")
configure_logging("WARNING")
```

Also accessible via the lazy re-export:

```python
import revos
revos.configure_logging("DEBUG")
```

---

## 4. `revos/device.py`

**File**: `/mnt/data/work/revos/revos/device.py`

### Purpose

Auto-detects the available compute device (CPU or CUDA GPU) by inspecting `onnxruntime` provider availability. Used internally by ASR and TTS engines to select the inference device.

### Key Elements

#### `auto_detect_device() -> str`

Detects whether a CUDA GPU is available through the `onnxruntime` provider list.

**Returns:**

| Value | Condition |
|---|---|
| `"cuda"` | `CUDAExecutionProvider` is in `onnxruntime.get_available_providers()` |
| `"cpu"` | Otherwise (no CUDA, or `onnxruntime` not installed) |

**Behavior:**
1. Attempts to import `onnxruntime`.
2. Calls `onnxruntime.get_available_providers()` to enumerate available execution providers.
3. If `"CUDAExecutionProvider"` is present, logs info and returns `"cuda"`.
4. If `onnxruntime` is not installed, logs a warning and falls back to `"cpu"`.
5. Otherwise logs info and returns `"cpu"`.

### Design Decisions

1. **No global state / no caching**: The function performs detection fresh on every call. This is stateless and avoids stale results if the CUDA environment changes during a process's lifetime (rare but possible with dynamic library loading).
2. **Graceful degradation**: If `onnxruntime` is missing entirely, it warns but does not crash. Returns `"cpu"` as a safe default.
3. **Detection via onnxruntime, not torch**: Uses `onnxruntime` rather than `torch.cuda.is_available()`. This is consistent with the package's dependency on `onnxruntime` for inference and avoids requiring a PyTorch installation.

### Dependencies

- `logging` (stdlib)
- `onnxruntime` (optional import — handled with `try/except ImportError`)

### Public API Surface

```python
from revos.device import auto_detect_device
device = auto_detect_device()  # "cpu" or "cuda"
```

This module is used internally by ASR/TTS but is also importable as part of the public package.

---

## 5. `revos/usage.py`

**File**: `/mnt/data/work/revos/revos/usage.py`

### Purpose

Provides usage tracking for gated models. Records when models are loaded or used for synthesis, persists events to a local JSONL log, and supports a callback mechanism for external systems to receive usage events.

### Key Elements

#### Type Alias

```python
UsageCallback = Callable[[dict], None]
```

A type alias for callback functions that receive a usage event dict.

#### Module-Level State

| Variable | Type | Value | Purpose |
|---|---|---|---|
| `_callbacks` | `list[UsageCallback]` | `[]` | Registered usage callbacks |
| `_USAGE_LOG` | `Path` | `~/.cache/revos/usage.jsonl` | Local usage log file path |

#### `register_callback(callback: UsageCallback) -> None`

Registers a callback to be invoked on usage events. Callbacks are appended to a module-level list and persist for the lifetime of the process.

**Callback receives a dict with these keys:**

| Key | Type | Description |
|---|---|---|
| `model_id` | `str` | HuggingFace model ID |
| `model_name` | `str` | RevoS model name |
| `task` | `str` | `"asr"` or `"tts"` |
| `hf_user` | `str or None` | HuggingFace username |
| `device` | `str` | `"cpu"` or `"cuda"` |
| `timestamp` | `str` | ISO 8601 UTC timestamp |
| `event` | `str` | `"model_loaded"` or `"model_synthesized"` |

#### `_log_to_local(usage: dict) -> None` (internal)

Appends a usage event as a single JSON line to `~/.cache/revos/usage.jsonl`.

- Creates parent directories if they don't exist (`mkdir(parents=True, exist_ok=True)`).
- Sets file permissions to `0o600` (owner read/write only) for privacy.
- Uses append mode (`"a"`), so the file grows over time.

#### `track_usage(event: str, model_id: str, model_name: str, task: str, hf_user: dict | None, device: str, **extra: object) -> None`

Records a usage event. This is the primary entry point for tracking.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `event` | `str` | Event type (`"model_loaded"`, `"model_synthesized"`) |
| `model_id` | `str` | HuggingFace model ID |
| `model_name` | `str` | RevoS model name |
| `task` | `str` | `"asr"` or `"tts"` |
| `hf_user` | `dict or None` | HF user info dict; extracts `.get("name")` for logging |
| `device` | `str` | `"cpu"` or `"cuda"` |
| `**extra` | `object` | Additional data merged into the event dict |

**Behavior:**
1. Constructs a usage dict with the above fields plus a UTC ISO 8601 timestamp.
2. If `hf_user` is a dict, extracts `hf_user["name"]` for the `hf_user` field; otherwise `None`.
3. Merges any `**extra` kwargs into the dict.
4. Logs locally to the JSONL file via `_log_to_local()`.
5. Iterates over all registered callbacks, calling each with the usage dict.
6. Callbacks are called in registration order. Exceptions in callbacks are caught, logged as warnings, and do not propagate (fault isolation).

#### `get_usage_log() -> list[dict]`

Reads all usage events from the local JSONL log.

**Returns:** `list[dict]` — parsed JSON objects, one per line, in file order (oldest first, most recent last). Returns an empty list if the log file doesn't exist.

### Design Decisions

1. **JSONL format**: Each event is a self-contained JSON object on a single line. This is append-friendly, crash-safe, and easy to tail/parse incrementally.
2. **File permissions `0o600`**: Usage data may contain user identifiers. Restricting to owner-only access is a reasonable privacy default.
3. **Fault isolation for callbacks**: A failing callback does not prevent other callbacks from running or crash the caller. Logged as a warning.
4. **No file size management**: The log grows unbounded. There is no rotation or pruning. For long-running production use, an external log rotation solution or periodic cleanup would be needed.
5. **Module-level mutable state (`_callbacks`)**: Callbacks are stored in a module-level list. This is a simple approach but is not thread-safe. In a multi-threaded environment, concurrent `register_callback` calls could race. This is unlikely to matter in typical single-process usage.
6. **`hf_user` parameter type mismatch**: The `track_usage` parameter `hf_user` is typed as `dict | None`, but the logged field extracts `hf_user["name"]` as a string. The callback dict's `hf_user` key is therefore `str | None`, not `dict | None`. This is a type-level inconsistency between the parameter type and the stored value.

### Dependencies

- `logging` (stdlib)
- `json` (stdlib)
- `os` (stdlib)
- `datetime` (`datetime`, `timezone`)
- `pathlib` (`Path`)
- `typing` (`Callable`)

### Public API Surface

```python
from revos.usage import (
    register_callback,
    track_usage,
    get_usage_log,
)

register_callback(my_callback)              # Register a usage callback
track_usage("model_loaded", "model-id", ...)  # Record a usage event
events = get_usage_log()                    # Read all logged events
```

---

## 6. `revos/py.typed`

**File**: `/mnt/data/work/revos/revos/py.typed`

### Purpose

PEP 561 marker file. Its presence (even empty, 0 bytes) signals to type checkers (mypy, pyright, etc.) that this package supports inline type hints. When a user installs `revos`, type checkers will recognize the package as typed and use the annotations in the source files for type checking.

### Contents

Empty file (0 bytes). This is the standard PEP 561 convention — the file's existence is the signal, not its contents.

### Requirements Met

- Package is under `tool.hatch.build.targets.wheel.packages = ["revos"]`, so the wheel includes the `revos/` directory tree including `py.typed`.
- No `py.typed` content is needed since the package ships source (not stub-only).

---

## 7. Cross-Cutting Design Patterns

### Pattern: Lazy Imports (`__init__.py`)

The package uses PEP 562 `__getattr__` to defer all heavy imports. This means:
- `import revos` is fast (no C-extension loading).
- `from revos import ASR` triggers the actual import of `sherpa-onnx`, `onnxruntime`, etc.
- Each lazy import is a one-time cost per name (Python caches module attributes after first access).

### Pattern: Logger Per Module

Every module creates its own named logger:
```python
import logging
logger = logging.getLogger(__name__)
```

These loggers are children of the `"revos"` logger configured by `configure_logging()`. This follows the standard Python logging hierarchy convention and allows a single `configure_logging()` call to control all submodules.

### Pattern: Graceful Degradation (`device.py`)

The device detection wraps `onnxruntime` import in a try/except, falling back to `"cpu"` without crashing. This ensures the package can be imported and basic operations attempted even if `onnxruntime` is misconfigured.

### Pattern: Observer / Callback (`usage.py`)

The usage module implements a simple observer pattern:
- `register_callback()` subscribes an observer.
- `track_usage()` notifies all observers.
- `_log_to_local()` is effectively a built-in observer (always active).

### Pattern: Local-First Persistence (`usage.py`)

Usage data is always written to a local JSONL file first. External systems receive data via callbacks but the local log is the ground truth. This is a reliable "log first, notify second" approach.

### Potential Improvements Noted

1. **`__version__` duplication**: The version string `"0.1.0"` appears in both `__init__.py` and `pyproject.toml`. A common pattern is to read from `importlib.metadata.version("revos")` to keep a single source of truth.
2. **No `__dir__` in `__init__.py`**: Adding a `__dir__` override would improve REPL auto-completion for `revos.<TAB>`.
3. **Usage log rotation**: No built-in size limit or rotation for `~/.cache/revos/usage.jsonl`.
4. **Thread safety of `_callbacks`**: The callback list is not protected by a lock. Low risk in typical usage but worth noting.
5. **`hf_user` type inconsistency**: `track_usage()` accepts `dict | None` but stores `str | None` in the event. The type alias could be tightened.
