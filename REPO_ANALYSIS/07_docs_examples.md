# 07 — Documentation & Examples Analysis

This document provides a thorough analysis of every documentation file and example
shipped with the RevoS repository.

---

## 1. README.md

### Purpose & Content Summary

The README is the primary entry point for the repository. It covers installation,
quick-start usage for both ASR and TTS, the CLI interface, the available model
catalog, custom model configuration, and a high-level project structure overview.

### Installation

Four install variants are documented:

| Variant | Command | What It Includes |
|---------|---------|-----------------|
| Core (ASR only) | `pip install revos` | ASR via sherpa-onnx |
| TTS support | `pip install revos[tts]` | Adds RevoVoice (requires PyTorch) |
| GPU support | `pip install revos[gpu]` | Installs `onnxruntime-gpu` instead of `onnxruntime` |
| Everything | `pip install revos[all]` | GPU + TTS combined |

`uv add revos` is also shown as an alternative.

**Critical install caveats documented:**

- `onnxruntime-gpu` **conflicts** with `onnxruntime`. Users must fully uninstall
  revos before switching to the GPU variant.
- The RevoVoice TTS model lives on a **private HuggingFace repository**.
  Users must run `huggingface-cli login` before using TTS, and request access
  from the model owner.

**Audio format support:** WAV, FLAC, OGG, and anything `libsndfile` handles.

### Quick Start

#### ASR Quick Start

```python
from revos.asr import ASR

asr = ASR('zipformer-v2')
result = asr.transcribe('meeting.wav')

result.text        # Full transcript string
result.language    # Detected language
result.segments    # List of Segment(start, end, text)
```

The public API uses factory functions (`ASR(...)`, `TTS(...)`) rather than
exposing engine classes directly. This is a deliberate design choice noted in
CONTRIBUTING.md.

#### TTS Quick Start

Two patterns are shown:

1. **Basic synthesis** -- `tts.synthesize('text')` returns an `Audio` object
   with `.save(path)` and `.duration`.
2. **Voice cloning** -- `tts.synthesize('text', ref_audio='speaker.wav', ref_text='...')`
   passes a reference sample and its transcript for zero-shot voice cloning.

#### CLI Quick Start

The CLI is built with Click and provides these commands:

| Command | Purpose |
|---------|---------|
| `revos transcribe -m <model> <audio>` | Transcribe audio to text |
| `revos transcribe --json <audio>` | JSON-formatted output |
| `revos transcribe --srt <audio>` | SRT subtitle output |
| `revos synthesize -m <model> -t "text" -o out.wav` | Synthesize speech |
| `revos synthesize -m <model> -f script.txt -o out.wav` | Synthesize from text file |
| `revos models` | List locally registered models |
| `revos catalog list` | Browse remote model catalog |
| `revos catalog pull <name>` | Install a remote model locally |
| `revos info` | Show environment info |

### Available Models

Two models ship with the repository:

| Model | Task | Backend | Languages | Access | Description |
|-------|------|---------|-----------|--------|-------------|
| `zipformer-v2` | ASR | sherpa-onnx | English | Open | Zipformer small transducer |
| `revovoice` | TTS | RevoVoice | 600+ | **Gated** | Zero-shot multilingual TTS + voice cloning |

Model manifests live in `revos/models/`:
- `revos/models/asr/zipformer_v2.yaml` -- Open, downloads from GitHub releases.
- `revos/models/tts/revovoice.yaml` -- Gated, requires HF login + approval.

### Gated Model Access Flow

Documented steps:
1. `huggingface-cli login` with a token from https://huggingface.co/settings/tokens
2. Visit the model's HF page and submit an access request.
3. Once approved, the model downloads automatically on first `TTS('revovoice')` use.

For team members: set `hf_private: true` in the YAML manifest to trigger auth
checks.

### Custom Models

Users can add models without code changes by placing a YAML manifest in
`~/.config/revos/models/`. The manifest must include:

- `name` -- unique identifier used in factory calls
- `task` -- `"asr"` or `"tts"`
- `backend` -- must match an existing backend (`sherpa-onnx`, `revovoice`)
- `model_type` -- e.g. `transducer`, `diffusion`
- `model_url` -- direct download link
- `revision` -- optional, pins to a specific HF commit hash or tag
- `sample_rate` -- expected sample rate
- `language` -- language code
- `description` -- human-readable
- `hf_private` -- boolean for gated models
- `files` -- dict mapping role to filename inside the archive

### Model Version Pinning

For HF-hosted models, the `revision` field pins to a commit hash or git tag:

```yaml
revision: "a1b2c3d"    # commit hash
revision: "v1.0.0"     # git tag
```

Without `revision`, the latest from the default branch is used.

### Remote Catalog

The catalog fetches available models directly from this repository's GitHub
`revos/models/` directory. Team members add YAML manifests; users discover them
without upgrading.

The catalog source is configurable:
- Environment variable: `REVOS_CATALOG_REPO="myorg/revos"`
- Config file: `~/.config/revos/config.yaml` with `catalog_repo: "myorg/revos"`

### Architecture Overview (from README)

```
revos/
├── revos/
│   ├── asr/           # ASR engine (sherpa-onnx backend)
│   ├── tts/           # TTS engine (RevoVoice backend)
│   ├── registry/      # Model manifest registry + downloader
│   ├── cli/           # Click CLI
│   ├── device.py      # GPU/CPU auto-detection
│   └── models/        # Bundled YAML manifests
├── tests/
├── pyproject.toml
├── AGENTS.md
└── CONTRIBUTING.md
```

---

## 2. AGENTS.md

### Purpose & Content Summary

AGENTS.md is the architecture and extension guide. It is aimed at both AI agents
and human contributors who want to extend RevoS with new models, backends, or
task types.

### Architecture

The architecture is layered:

```
CLI (click)
  -> Factory Functions (ASR/TTS)
    -> Base Classes (BaseASR/BaseTTS)
      -> Concrete Engines

         Model Registry (YAML manifests)
         Model Downloader (~/.cache/revos/)
         Remote Catalog (GitHub repo)
```

**Core design principle:** Adding a new model should require **zero changes to
core code** if the backend is already supported. Only a YAML manifest is needed.

Adding a new backend requires a new engine file plus registration in the factory.

### Task 1: Add a New Model (Same Backend)

Only requires a YAML manifest placed in either:
- `revos/models/{task}/` (bundled), or
- `~/.config/revos/models/{task}/` (user-local)

**Manifest field rules:**
- `name` must be unique within the same `task`.
- `files` dict keys must match what the engine expects.
- `model_url` must be a direct download link.
- The archive must extract to contain the files listed in `files`.

### Task 2: Add a New Backend for ASR

Steps:
1. Research the backend's Python API.
2. Create engine file at `revos/asr/{backend}_engine.py`.
3. Engine must inherit from `BaseASR` and implement `transcribe()`.
4. Register in factory function inside `revos/asr/__init__.py`.
5. Add optional dependency to `pyproject.toml`.
6. Add tests with mocked backend.
7. Add at least one YAML manifest.

**Engine rules:**
- Must return a `Transcript` object with `text`, `segments`, and `language`.
- Must use `revos.registry` for manifest lookup and `ensure_model()` for downloads.
- Must handle device selection (`"auto"`, `"cpu"`, `"cuda"`).
- Must **lazy-import** backend dependencies so the library works without them.
- Must raise `ImportError` with install instructions if the backend is missing.

### Task 3: Add a New Backend for TTS

Same pattern as ASR but the engine inherits from `BaseTTS` and implements
`synthesize()`. Must return an `Audio` object with `samples` (np.ndarray float32)
and `sample_rate` (int).

**Long text support:** `BaseTTS` provides `synthesize_long()` which:
- Automatically splits text into sentences (handles English and CJK punctuation).
- Synthesizes each chunk.
- Concatenates audio with configurable silence gaps.
- Falls back to comma/word boundaries for very long sentences.
- Uses `Audio.concatenate()` for joining segments.

Parameters: `max_chars=500`, `silence_duration=0.1`.

### Task 4: Add a New Task Type

For entirely new capabilities (diarization, VAD, speech enhancement):

1. Create task package at `revos/{task}/` with `__init__.py`, `base.py`,
   `result.py`, `{backend}_engine.py`.
2. Define task-specific result dataclasses.
3. Follow the `BaseASR`/`BaseTTS` pattern for the abstract base class.
4. Implement at least one backend engine.
5. Add CLI commands in `revos/cli/main.py`.
6. Add model manifests in `revos/models/{task}/`.
7. Add tests.

**Rule:** Each task gets its own package under `revos/`, with its own result
types and factory function matching the ASR/TTS pattern.

### Key File Locations Table

| Purpose | Location |
|---------|----------|
| ASR engine (sherpa-onnx) | `revos/asr/sherpa_engine.py` |
| TTS engine (RevoVoice) | `revos/tts/revovoice_engine.py` |
| ASR base class | `revos/asr/base.py` |
| TTS base class | `revos/tts/base.py` (includes `synthesize_long`) |
| ASR result types | `revos/asr/result.py` (Segment, Transcript) |
| TTS result types | `revos/tts/result.py` (Audio, Audio.concatenate) |
| Model registry | `revos/registry/registry.py` |
| Manifest loader | `revos/registry/manifest.py` (ModelManifest dataclass) |
| Model downloader | `revos/registry/downloader.py` |
| Remote catalog | `revos/catalog.py` |
| Device detection | `revos/device.py` |
| CLI entry point | `revos/cli/main.py` |
| Bundled manifests | `revos/models/{asr,tts}/*.yaml` |
| User manifests | `~/.config/revos/models/**/*.yaml` |
| Model cache | `~/.cache/revos/{model_name}/` |

### Remote Catalog (from AGENTS.md)

The catalog fetches YAML manifests from the GitHub repo's `revos/models/`
directory via the GitHub API. Workflow:

1. Team member adds YAML to `revos/models/{task}/` and pushes.
2. User runs `revos catalog list` to see available models.
3. User runs `revos catalog pull <name>` to install locally.

The catalog source is configurable via `REVOS_CATALOG_REPO` env var or
`~/.config/revos/config.yaml`.

### Current Backends

**ASR Backends:**

| Backend | Engine File | Dependencies | Notes |
|---------|------------|-------------|-------|
| sherpa-onnx | `revos/asr/sherpa_engine.py` | sherpa-onnx, onnxruntime | Zipformer transducer via ONNX |

**TTS Backends:**

| Backend | Engine File | Dependencies | Notes |
|---------|------------|-------------|-------|
| revovoice | `revos/tts/revovoice_engine.py` | omnivoice, torch | Diffusion-based zero-shot TTS, 600+ languages |

Important naming note: `omnivoice` is the pip package name; `revovoice` is the
model/backend name. This distinction appears in CONTRIBUTING.md.

### Testing Checklist

For any new model or backend addition:

- All tests pass: `uv run pytest tests/ -v`
- Factory function returns correct engine type
- Manifest loads and registers correctly
- CLI works: `uv run revos transcribe -m {model} test.wav`
- JSON output valid: `--json` flag
- SRT output valid: `--srt` flag
- GPU fallback to CPU works (`device="cpu"`)
- Base imports work without optional deps
- ImportErrors are helpful when optional backend is missing

---

## 3. CONTRIBUTING.md

### Purpose & Content Summary

The contribution guide covers the development workflow, code style requirements,
testing commands, and pointers to AGENTS.md for deeper architecture docs.

### Development Setup

```bash
git clone <repo-url>
cd revos
uv sync --extra dev        # Install dev dependencies
uv run ruff check revos/ tests/   # Lint
uv run pytest tests/ -v    # Test
uv build                   # Build package
```

### Adding a New Model (Zero Code Changes)

Reiterates the YAML manifest approach from README and AGENTS.md. Key details:

- Place manifests in `~/.config/revos/models/asr/` (user) or `revos/models/asr/` (bundled).
- Use `revision` field to pin HF model versions.
- Use `hf_private: true` for gated models.
- Remote catalog: models in `revos/models/` in the repo are auto-discoverable
  via `revos catalog list`.

### Adding a New Backend

Five-step process:
1. Create `revos/{task}/{backend}_engine.py` inheriting from base class.
2. Register in the factory function (`revos/{task}/__init__.py`).
3. Add optional dependency to `pyproject.toml`.
4. Add tests with mocked backend.
5. Add at least one YAML manifest.

Points to AGENTS.md for details.

### Development Workflow

1. Create feature branch: `git checkout -b feat/my-feature`
2. Make changes with tests.
3. Run lint and tests (`ruff check`, `pytest`).
4. Commit with clear messages.
5. Open a pull request.

### Code Style

- **Python 3.11+**, formatted by `ruff` (line length 88).
- **Lazy imports** for optional dependencies.
- **Factory functions** as the public API (not classes).
- **YAML manifests** for model configuration.

Critical naming note documented here: `omnivoice` is the pip package name;
`revovoice` is the model/backend name.

### Testing

Three testing modes:

```bash
uv run pytest tests/ -v                          # All tests
uv run pytest tests/ --cov=revos --cov-report=term-missing  # With coverage
uv run pytest tests/ -v -m "not slow"             # Exclude slow/integration tests
```

### Project Structure (from CONTRIBUTING.md)

```
revos/
  asr/           # ASR engine
  tts/           # TTS engine (includes synthesize_long)
  registry/      # Model manifest registry + downloader
  catalog.py     # Remote model catalog (GitHub-based)
  cli/           # Click CLI
  models/        # Bundled YAML manifests
tests/           # Test suite
```

Note: `catalog.py` is at the top level of the `revos/` package, not inside a
sub-package.

---

## 4. CHANGELOG.md

### Purpose & Content Summary

Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format with
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Version History

#### [0.1.0] - 2026-04-24 (Initial Release)

**Added:**

- ASR engine with sherpa-onnx backend (zipformer-v2 model)
- TTS engine with RevoVoice backend (revovoice model)
- Model registry with YAML manifests (bundled + user)
- Model downloader with security hardening:
  - Tarball filter (`filter="data"`)
  - Zip path traversal protection
  - URL scheme validation (only http/https allowed)
  - Usage log file permissions set to `0o600`
- CLI commands: `transcribe`, `synthesize`, `models`, `info`
- CLI commands: `catalog list`, `catalog pull`
- Remote model catalog fetching from GitHub repo
- `synthesize_long()` for automatic text splitting and audio concatenation
- `Audio.concatenate()` for joining audio segments
- `Audio.duration` property
- Model version pinning via `revision` field in YAML manifests
- Gated model access with clear error messages (401/403)
- Usage tracking for gated models (local JSONL log)
- Device auto-detection (GPU/CPU)
- HuggingFace authentication check
- Pre-commit hooks (ruff lint + format)
- CI workflow with coverage reporting
- MIT license

**Changed:**

- Renamed model from `omnivoice` to `revovoice` (both backend and model name).
- Catalog fetches from this GitHub repo instead of a separate HF repo.

**Security:**

- Zip path traversal protection in model extraction.
- URL scheme validation (only http/https).
- Usage log file permissions set to `0o600`.
- Tarball extraction uses `filter="data"`.

### Notable Observations

- Only one version entry exists (`0.1.0`), indicating this is a very young project.
- The `omnivoice` -> `revovoice` rename happened before or during the initial
  release, suggesting a branding decision was made late in development.
- Security is a first-class concern -- four security-related items in the
  initial release changelog.

---

## 5. LICENSE

### Purpose & Content Summary

Standard MIT License.

- Copyright holder: "RevoS Team"
- Copyright year: 2026
- Standard MIT terms: free to use, copy, modify, merge, publish, distribute,
  sublicense, and sell, with the condition that the copyright notice and
  permission notice are included in all copies or substantial portions.
- No additional restrictions, patent grants, or contributor license agreements.

---

## 6. examples/asr_transcribe.py

### Purpose

Demonstrates basic ASR (speech-to-text) usage.

### Full Source

```python
"""ASR example -- Transcribe audio to text.

Usage:
    uv run python examples/asr_transcribe.py audio.wav
"""

import sys
from revos.asr import ASR

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python asr_transcribe.py <audio.wav>")
        sys.exit(1)

    asr = ASR("zipformer-v2")
    result = asr.transcribe(sys.argv[1])

    print(f"Language: {result.language}")
    print(f"Text: {result.text}")
    print(f"\nSegments:")
    for seg in result.segments:
        print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
```

### API Usage Pattern

1. Import the factory: `from revos.asr import ASR`
2. Instantiate with model name: `ASR("zipformer-v2")`
3. Call `transcribe(audio_path)` with a file path.
4. Access results via `result.text`, `result.language`, and `result.segments`.
5. Each segment has `.start`, `.end` (seconds), and `.text`.

### Caveats

- Accepts the audio file path as a command-line argument (`sys.argv[1]`).
- No error handling beyond argument count check -- relies on the library to
  surface file-not-found or format errors.
- Hardcoded to the `zipformer-v2` model (not configurable via CLI arg).
- Uses `print()` rather than `logging` (acceptable for an example script).

---

## 7. examples/tts_synthesize.py

### Purpose

Demonstrates basic TTS (text-to-speech) synthesis and the `synthesize_long()`
method for longer text.

### Full Source

```python
"""TTS example -- Synthesize speech from text.

Usage:
    uv run python examples/tts_synthesize.py
"""

from revos.tts import TTS

if __name__ == "__main__":
    tts = TTS("revovoice")

    # Basic synthesis
    audio = tts.synthesize("Hello, how are you today?")
    audio.save("examples/output_hello.wav")
    print(f"Saved hello ({audio.duration:.1f}s)")

    # Long text (auto-splits and concatenates)
    long_text = (
        "RevoS is a unified Python library for speech AI. "
        "It supports automatic speech recognition and text to speech. "
        "You can use it to transcribe audio files or synthesize speech "
        "from text. It supports multiple languages and voice cloning."
    )
    audio = tts.synthesize_long(long_text)
    audio.save("examples/output_long.wav")
    print(f"Saved long text ({audio.duration:.1f}s)")
```

### API Usage Pattern

1. Import the factory: `from revos.tts import TTS`
2. Instantiate with model name: `TTS("revovoice")`
3. **Basic synthesis:** `tts.synthesize("text")` returns an `Audio` object.
4. **Save to file:** `audio.save("path.wav")`.
5. **Duration:** `audio.duration` returns length in seconds.
6. **Long text:** `tts.synthesize_long(text)` auto-splits and concatenates,
   returning a single `Audio` object.

### Caveats

- Writes output files into the `examples/` directory itself.
- The `revovoice` model requires HF authentication; the example will fail if
  the user has not logged in and been approved.
- No command-line arguments -- text is hardcoded.
- No reference audio is used here (see `tts_voice_cloning.py` for that pattern).

---

## 8. examples/tts_voice_cloning.py

### Purpose

Demonstrates zero-shot voice cloning using the RevoVoice TTS engine by providing
a reference audio sample.

### Full Source

```python
"""Voice cloning example -- Synthesize with a reference voice.

Usage:
    uv run python examples/tts_voice_cloning.py
"""

from revos.tts import TTS

if __name__ == "__main__":
    tts = TTS("revovoice")

    # Synthesize with a reference audio for voice cloning
    # Replace with your own reference audio file
    audio = tts.synthesize(
        "This will sound like the reference speaker.",
        ref_audio="examples/reference_speaker.wav",
        ref_text="This is a sample of the speaker talking.",
    )
    audio.save("examples/output_cloned.wav")
    print(f"Saved cloned voice ({audio.duration:.1f}s)")
```

### API Usage Pattern

1. Import and instantiate: `TTS("revovoice")`
2. Call `synthesize()` with **three** arguments:
   - `text` -- the text to synthesize in the cloned voice.
   - `ref_audio` -- path to a WAV file of the reference speaker.
   - `ref_text` -- transcript of what is said in `ref_audio`.
3. The engine performs zero-shot voice cloning: the output audio will sound
   like the reference speaker but say the new text.
4. Save and inspect duration as usual.

### Caveats

- The reference audio file `examples/reference_speaker.wav` is **not included**
  in the repository. Users must supply their own. The comment says "Replace
  with your own reference audio file."
- The `ref_text` must accurately match what is said in `ref_audio` for good
  cloning quality.
- Requires HF authentication (gated model).
- No error handling for missing reference file.
- This example depends on the RevoVoice model's zero-shot cloning capability,
  which is specific to that backend. Other TTS backends may not support
  `ref_audio`/`ref_text`.

---

## Cross-Cutting Patterns Across All Docs & Examples

### Consistent Design Principles

1. **Factory function pattern:** The public API is `ASR('model-name')` and
   `TTS('model-name')`, not direct class instantiation. Factory functions
   dispatch to the correct engine based on the manifest's `backend` field.

2. **YAML-driven configuration:** Models are defined declaratively in YAML,
   not in code. This enables adding models without touching Python source.

3. **Lazy imports for optional deps:** Backends lazy-import their heavy
   dependencies (PyTorch, ONNX Runtime) so the base install stays lightweight.

4. **Security-first downloader:** Model archives are extracted with path
   traversal protection, URL scheme validation, and tarball filtering.

5. **Graceful degradation:** GPU falls back to CPU. Missing optional deps
   produce clear `ImportError` messages with install instructions.

### Documentation Gaps Observed

- **No API reference:** There is no auto-generated or hand-written API
  reference documenting all public classes, methods, parameters, and return
  types. The quick-start snippets and AGENTS.md partially compensate, but
  there is no comprehensive reference.

- **No error handling guide:** The examples show happy-path usage. There is
  no documentation of common errors, error types, or recovery strategies.

- **No performance/latency guidance:** No documentation of expected latency,
  memory usage, or throughput for either ASR or TTS.

- **No deployment guide:** No guidance on running in production, Docker,
  serverless, or batch processing scenarios.

- **`synthesize_long()` tuning:** The `max_chars` and `silence_duration`
  parameters are mentioned in AGENTS.md but not explored in examples. No
  guidance on optimal values for different languages or use cases.

- **Missing reference audio:** The voice cloning example references a file
  that does not exist in the repo. A note about this is in the script's
  comment, but it could be more prominent.
