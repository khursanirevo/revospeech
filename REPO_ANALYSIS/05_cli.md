# CLI Analysis

## Overview

The RevoS CLI (`revos`) is a command-line interface built with **Click** (`click>=8.0`). It is registered as the console script entry point `revos = "revos.cli.main:cli"` in `pyproject.toml`. The CLI provides commands for speech recognition (ASR), text-to-speech (TTS), model management, environment introspection, and a remote model catalog.

There are **two files** in the CLI package:

| File | Lines | Purpose |
|---|---|---|
| `revos/cli/__init__.py` | 1 | Empty package marker (`"""RevoS CLI package."""`) |
| `revos/cli/main.py` | 251 | All CLI commands and helpers |

---

## File: `revos/cli/__init__.py`

**Purpose:** Trivial package initializer. Contains only a docstring (`"""RevoS CLI package."""`). No exports, no code. The actual CLI logic lives entirely in `main.py`.

---

## File: `revos/cli/main.py`

### Purpose and Responsibility

This is the sole CLI implementation file. It defines a Click command group with seven commands total (four top-level commands, one top-level group with two subcommands). It handles argument parsing, output formatting (plain text, JSON, SRT), user-facing error messages, and delegates all real work to the library modules.

### CLI Framework

**Click 8+** (`import click`). Uses:
- `@click.group()` for the root group and the `catalog` subgroup
- `@click.command()` for leaf commands
- `@click.option()` for named options
- `@click.argument()` for positional arguments
- `@click.version_option()` for `--version`
- `click.echo()` for all output (no `print()`)
- `click.UsageError` for validation errors
- `click.Path()` for path validation (with `exists=True` where appropriate)

**No Rich or console formatting library** is used. All output is plain text via `click.echo()`. Tables are rendered manually with fixed-width column formatting (f-string alignment like `{m.name:<20}`).

---

## Command Tree

```
revos
  |-- transcribe    Transcribe audio to text
  |-- synthesize    Synthesize speech from text
  |-- models        List locally available models
  |-- info          Show environment and configuration
  `-- catalog       Browse and pull remote models
       |-- list     List models in remote catalog
       `-- pull     Download and install a model
```

---

## Detailed Command Reference

### Root Group: `cli`

```python
@click.group()
@click.version_option()
def cli() -> None
```

- **Help text:** "RevoS -- A unified library for speech AI (ASR & TTS)."
- **`--version`** (provided by `@click.version_option()`): Prints the installed package version.
- **Entry point:** Registered in `pyproject.toml` as `revos = "revos.cli.main:cli"`.

---

### Command: `revos transcribe`

```python
@cli.command()
@click.option("--model", "-m", required=True, help="ASR model name (e.g. zipformer-v2)")
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--srt", "as_srt", is_flag=True, help="Output as SRT subtitles")
def transcribe(model: str, audio_path: str, as_json: bool, as_srt: bool) -> None
```

| Argument/Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `audio_path` | positional, `click.Path(exists=True)` | Yes | -- | Path to the audio file to transcribe |
| `--model` / `-m` | `str` | Yes | -- | ASR model name (e.g. `zipformer-v2`) |
| `--json` / `as_json` | flag (`bool`) | No | `False` | Output structured JSON |
| `--srt` / `as_srt` | flag (`bool`) | No | `False` | Output SRT subtitle format |

**Behavior:**
1. Imports `revos.asr.ASR` and instantiates `ASR(model)`.
2. Calls `asr.transcribe(audio_path)` which returns a `Transcript` object (with `.text`, `.segments`, `.language` fields).
3. Output mode selection:
   - `--json`: Prints a JSON object with keys `text`, `segments` (list of `{start, end, text, confidence}`), and `language`. Uses `indent=2` and `ensure_ascii=False`.
   - `--srt`: Prints standard SRT subtitle format. Each segment gets a numbered entry with `HH:MM:SS,mmm --> HH:MM:SS,mmm` timestamps and text.
   - Default (no flag): Prints just `result.text` as plain text.

**Library mapping:**
- `revos.asr.ASR.__init__(model_name: str)` -> looks up the model in the local registry
- `ASR.transcribe(audio_path: str) -> Transcript` -> delegates to the backend engine (e.g. `SherpaEngine.transcribe`)

**Error handling:** No explicit try/except. If the audio file does not exist, Click's `exists=True` validation catches it before the function runs. Model loading or transcription errors propagate as unhandled exceptions (Click will display the traceback).

---

### Command: `revos synthesize`

```python
@cli.command()
@click.option("--model", "-m", required=True, help="TTS model name (e.g. revovoice)")
@click.option("--text", "-t", help="Text to synthesize")
@click.option("--file", "-f", type=click.Path(exists=True), help="Text file to synthesize")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output audio path")
@click.option("--speed", default=1.0, help="Speech speed (default: 1.0)")
@click.option("--ref-audio", type=click.Path(exists=True), help="Reference audio for voice cloning")
@click.option("--ref-text", help="Transcription of reference audio")
def synthesize(
    model: str, text: str | None, file: str | None,
    output: str, speed: float, ref_audio: str | None, ref_text: str | None
) -> None
```

| Argument/Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `--model` / `-m` | `str` | Yes | -- | TTS model name (e.g. `revovoice`) |
| `--text` / `-t` | `str` | No* | `None` | Text to synthesize (inline string) |
| `--file` / `-f` | `str`, `click.Path(exists=True)` | No* | `None` | Path to a text file to synthesize |
| `--output` / `-o` | `str`, `click.Path()` | Yes | -- | Output audio file path |
| `--speed` | `float` | No | `1.0` | Speech speed multiplier |
| `--ref-audio` | `str`, `click.Path(exists=True)` | No | `None` | Reference audio for voice cloning |
| `--ref-text` | `str` | No | `None` | Transcription of the reference audio |

*Either `--text` or `--file` must be provided (validated at runtime).

**Behavior:**
1. Validates that at least one of `--text` or `--file` is provided; raises `click.UsageError` if neither is given.
2. If `--file` is provided (and `--text` is not), reads the file content and strips whitespace.
3. Imports `revos.tts.TTS` and instantiates `TTS(model)`.
4. **Auto-detection of long text:** If `len(text) > 500`, calls `tts.synthesize_long(...)`; otherwise calls `tts.synthesize(...)`. Both receive the same keyword arguments: `text, output, speed, ref_audio, ref_text`.
5. On success, prints: `"Saved {N} samples ({duration:.1f}s) to {output}"` using the returned `AudioResult` object's `.samples` and `.sample_rate`.

**Library mapping:**
- `revos.tts.TTS.__init__(model_name: str)` -> looks up model in local registry
- `TTS.synthesize(text, output, speed=1.0, ref_audio=None, ref_text=None) -> AudioResult` -> single-pass synthesis
- `TTS.synthesize_long(text, output, speed=1.0, ref_audio=None, ref_text=None) -> AudioResult` -> chunked synthesis for long text (internal chunking, concatenation)

**Error handling:**
- `click.UsageError` when neither `--text` nor `--file` is provided.
- `assert text is not None` after the file-read branch (defensive; should always pass due to prior validation).
- No try/except for synthesis errors -- they propagate as unhandled exceptions.

---

### Command: `revos models`

```python
@cli.command()
@click.option("--task", "-t", help="Filter by task type (asr or tts)")
def models(task: str | None) -> None
```

| Argument/Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `--task` / `-t` | `str` | No | `None` | Filter models by task type (`asr` or `tts`) |

**Behavior:**
1. Imports `revos.registry.list_models` and calls `list_models(task)`.
2. If no models are found, prints `"No models found."` and returns.
3. Otherwise, prints a formatted table with columns: `Name`, `Task`, `Backend`, `Language`. Uses fixed-width formatting with a 53-character separator line.

**Output format:**
```
Name                 Task   Backend         Language
-----------------------------------------------------
zipformer-v2         asr    sherpa-onnx     multilingual
revovoice            tts    onnxruntime     en
```

**Library mapping:**
- `revos.registry.list_models(task: str | None = None) -> list[ModelManifest]` -> returns locally registered models

---

### Command: `revos info`

```python
@cli.command()
def info() -> None
```

No arguments or options.

**Behavior:** Prints environment and configuration information. Gathers:

| Field | Source |
|---|---|
| RevoS version | `importlib.metadata.version("revos")`, fallback `"unknown"` |
| Python version | `sys.version.split()[0]` |
| Device | `revos.device.auto_detect_device()` |
| Models loaded | `len(revos.registry.list_models())` |
| Cache dir | Hardcoded `~/.cache/revos` |
| Catalog repo | `revos.catalog.get_catalog_repo()` |
| HuggingFace auth | `huggingface_hub.HfApi().whoami()`, falls back to `"not logged in"` |

**Error handling:**
- Version lookup: wrapped in try/except, returns `"unknown"` on failure.
- HuggingFace auth: wrapped in broad `except Exception`, prints `"not logged in"` if any error occurs.

---

### Subgroup: `revos catalog`

```python
@cli.group()
def catalog() -> None
```

- **Help text:** "Browse and pull models from the remote catalog."
- Container group for catalog-related subcommands.

---

#### Subcommand: `revos catalog list`

```python
@catalog.command("list")
@click.option("--task", "-t", help="Filter by task type (asr or tts)")
def catalog_list(task: str | None) -> None
```

| Argument/Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `--task` / `-t` | `str` | No | `None` | Filter catalog models by task type |

**Behavior:**
1. Prints `"Fetching catalog from {repo}..."` to show progress.
2. Calls `revos.catalog.list_catalog(task)` which fetches the remote model catalog.
3. On `RuntimeError`, prints the error to stderr and exits with code 1.
4. If no results, prints `"No models found in catalog."` and returns.
5. Otherwise prints a formatted table with columns: `Name`, `Task`, `Backend`, `Language`, `Version`. Uses a 65-character separator line.
6. After the table, prints a hint: `"Use 'revos catalog pull <name>' to install."`

**Library mapping:**
- `revos.catalog.get_catalog_repo() -> str` -> returns the GitHub repo URL
- `revos.catalog.list_catalog(task: str | None = None) -> list[ModelManifest]` -> fetches and parses remote catalog

**Error handling:**
- `RuntimeError` from `list_catalog()` is caught; error message printed to stderr via `click.echo(..., err=True)`, then `SystemExit(1)`.

---

#### Subcommand: `revos catalog pull`

```python
@catalog.command("pull")
@click.argument("model_name")
def catalog_pull(model_name: str) -> None
```

| Argument/Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `model_name` | positional `str` | Yes | -- | Name of the model to pull from the catalog |

**Behavior:**
1. Prints `"Pulling '{model_name}' from {repo}..."`.
2. Calls `revos.catalog.pull_model(model_name)` which downloads and installs the model locally.
3. On success, prints the installation path and a usage hint: `"Use: from revos.tts import TTS; TTS('{model_name}')"`.
4. On `KeyError` (model not found) or `RuntimeError` (download/install failure), prints the error to stderr and exits with code 1.

**Library mapping:**
- `revos.catalog.pull_model(name: str) -> Path` -> downloads model files and installs into local registry

**Error handling:**
- `KeyError` and `RuntimeError` are caught; error message printed to stderr via `click.echo(..., err=True)`, then `SystemExit(1)`.

---

## Helper Functions

### `_get_version() -> str`

```python
def _get_version() -> str:
    from importlib.metadata import version
    try:
        return version("revos")
    except Exception:
        return "unknown"
```

- Uses `importlib.metadata` (Python 3.8+) to read the installed package version.
- Broad `except Exception` catch for safety (package may not be installed, metadata may be missing).
- Returns `"unknown"` on any failure.

### `_format_srt_time(seconds: float) -> str`

```python
def _format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
```

- Formats a float number of seconds into the SRT timestamp format `HH:MM:SS,mmm`.
- Used exclusively by `revos transcribe --srt`.

---

## Import Strategy (Lazy Imports)

All library imports inside command functions are **lazy** (deferred). None of the heavy modules (`revos.asr`, `revos.tts`, `revos.registry`, `revos.catalog`, `revos.device`) are imported at module level. Only `click`, `json`, and `Path` are top-level imports. This means:

- `revos --help` and `revos --version` are fast (no model/engine loading).
- The first invocation of any command that touches a library module pays the import cost.
- This avoids import errors from optional dependencies (e.g., `sherpa-onnx`, `onnxruntime`) when running commands that don't need them.

---

## Output Formatting Summary

| Command | Default Output | Alternative Formats |
|---|---|---|
| `transcribe` | Plain text (`result.text`) | `--json` (structured JSON), `--srt` (subtitle format) |
| `synthesize` | Summary line with sample count and duration | None |
| `models` | Fixed-width column table (Name, Task, Backend, Language) | None |
| `info` | Key-value pairs with aligned labels | None |
| `catalog list` | Fixed-width column table (Name, Task, Backend, Language, Version) + hint | None |
| `catalog pull` | Status messages + installation path + usage hint | None |

No Rich library, no color, no progress bars. All output is plain monospaced text via `click.echo()`.

---

## Error Handling Summary

| Error Type | Where | Handling |
|---|---|---|
| Missing `--text`/`--file` | `synthesize` | `click.UsageError` (shows usage help) |
| Model not found in catalog | `catalog pull` | `KeyError` -> stderr message + `SystemExit(1)` |
| Download/install failure | `catalog pull` | `RuntimeError` -> stderr message + `SystemExit(1)` |
| Catalog fetch failure | `catalog list` | `RuntimeError` -> stderr message + `SystemExit(1)` |
| Package version unavailable | `info` | Returns `"unknown"` (broad `except Exception`) |
| HuggingFace not authenticated | `info` | Prints `"not logged in"` (broad `except Exception`) |
| File not found (audio/text/ref-audio) | Various | Click's `click.Path(exists=True)` validates before function body runs |
| Transcription/synthesis errors | `transcribe`, `synthesize` | **Unhandled** -- propagates as Python traceback |

Notable gaps: `transcribe` and `synthesize` have no try/except wrapping the core library calls. Any exception from the ASR/TTS engine will produce a raw Python traceback rather than a user-friendly error message. The `catalog` commands are better handled with explicit error catching.

---

## Entry Point Registration

From `pyproject.toml`:

```toml
[project.scripts]
revos = "revos.cli.main:cli"
```

This means after `pip install revos`, the `revos` command is available on PATH and invokes `cli()` from `revos.cli.main`. The `if __name__ == "__main__": cli()` guard also allows running via `python -m revos.cli.main`.
