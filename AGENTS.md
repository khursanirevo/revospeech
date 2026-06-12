# AGENTS.md — Contributing Guide for AI Agents & Humans

This document describes how to extend the revospeech library with new models, backends, and tasks. Follow it precisely when adding support for a new speech AI model.

---

## Project Status (as of 2026-06-12)

### Completed (MVP Foundation — Phases 0-2)

The library has a hybrid local + API architecture ready for extension. Local inference works end-to-end. API backend support is scaffolded but engines are not yet implemented.

**What works now:**
- Local ASR via sherpa-onnx (Zipformer transducer)
- Local TTS via revovoice/omnivoice (diffusion, zero-shot voice cloning)
- Model registry with YAML manifests (11 optional fields including mode, capabilities, languages)
- Model discovery: `revospeech.models`, `revospeech.search("query")`, `revospeech.check_model("name")`
- Rich CLI: `revospeech models --ready`, `revospeech search zip`, `revospeech models-info zipformer-v2`
- Remote catalog with caching: `revospeech catalog list`, `revospeech catalog pull <name>`
- Config management: `revospeech config set-api-key`, env var `REVOLAB_API_KEY`
- Exception hierarchy with helpful suggestions
- Thread-safe registry and usage tracking
- CI: ruff check + format, pytest with coverage gate

**What's blocked:**
- revolab API engine implementation — waiting on API contract (endpoints, auth, request/response schema)
- Branch rename (`master` → `main` decision)

**Reference files:**
- `TODO.md` — full backlog (189 items across 9 phases)
- `PLAN.md` — Architect + Critic approved consensus plan with RALPLAN-DR
- `REPO_ANALYSIS/` — detailed codebase analysis (9 files, 4,343 lines)

---

## Architecture Overview

```
CLI (click) → Factory Functions (ASR/TTS) → Base Classes (BaseASR/BaseTTS) → Concrete Engines
                 ↓ mode dispatch
           manifest.mode == "api" → API Engine (check API key, call HTTP)
           manifest.mode == "local" → Local Engine (download weights, run inference)
                                    ↕
                              Model Registry (YAML manifests)
                                    ↕
                              Config (API keys, settings)
                              Model Downloader (~/.cache/revospeech/)
                              Remote Catalog (GitHub repo)
```

**Key principle:** Adding a new local model requires ZERO changes to core code if the backend is already supported. Only a new YAML manifest is needed. Adding an API backend requires a new engine file implementing BaseASR/BaseTTS + a YAML manifest with `mode: api`.

---

## How to Add the revolab API Backend (Next Major Task)

When API docs are provided, follow this process:

### 1. Define the API Contract

Create `revospeech/api_contract.py` (or similar) documenting:
- Base URL(s)
- Auth header format (e.g., `Authorization: Bearer {api_key}`)
- ASR endpoint: request format (audio upload), response format (transcript with timestamps)
- TTS endpoint: request format (text + params), response format (audio bytes or streaming chunks)
- Error response format (HTTP status codes, error body shape)
- Rate limits, pagination, max file sizes

### 2. Create API Engine Files

**ASR:** Create `revospeech/asr/revolab_engine.py` implementing `BaseASR`:

```python
"""revolab cloud API backend for ASR."""
from __future__ import annotations
import logging
import httpx
from .base import BaseASR
from .result import Segment, Transcript
from revospeechpeech.config import get_api_key
from revospeechpeech.exceptions import RevosConfigError, RevosEngineError

logger = logging.getLogger(__name__)

class RevolabASR(BaseASR):
    def __init__(self, model_name: str, device: str = "auto") -> None:
        super().__init__(model_name, device)
        from revospeechpeech.registry import get
        self.manifest = get(model_name, "asr")
        self.api_key = get_api_key()
        if not self.api_key:
            raise RevosConfigError(
                f"Model '{model_name}' requires an API key.",
                suggestion="Set your API key: export REVOLAB_API_KEY=your-key or run: revospeech config set-api-key"
            )
        self.base_url = self.manifest.api_endpoint or "https://api.revolab.ai/v1"
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0,
        )

    def transcribe(self, audio_path: str, **kwargs) -> Transcript:
        # Upload audio, parse response into Transcript
        # Map API response fields → Segment(start=, end=, text=, confidence=)
        raise NotImplementedError("Implement once API contract is defined")
```

**TTS:** Create `revospeech/tts/revolab_engine.py` implementing `BaseTTS` — same pattern.

### 3. Add YAML Manifests for API Models

```yaml
name: revolab-whisper
task: asr
mode: api                        # KEY: this triggers API dispatch
api_endpoint: "https://api.revolab.ai/v1"  # Optional: defaults to engine default
backend: revolab-api
capabilities:
  - word-timestamps
  - streaming
languages:
  - en
  - zh
  - es
  - multilingual
description: "revolab cloud Whisper API"
```

### 4. Register in Factory

The factories in `revospeech/asr/__init__.py` and `revospeech/tts/__init__.py` already check `manifest.is_api`. Currently they raise `NotImplementedError`. Update to load the API engine:

```python
# In ASR() factory, replace the NotImplementedError with:
if manifest.is_api:
    from .revolab_engine import RevolabASR
    return RevolabASR(model_name, device)
```

### 5. Add httpx as Optional Dependency

In `pyproject.toml`:
```toml
[project.optional-dependencies]
api = ["httpx>=0.27"]
```

### 6. Test with Mock API Server

Write tests using `pytest-httpx` or mock `httpx.Client` responses.

### Key Files to Modify

| File | Change |
|---|---|
| `revospeech/asr/revolab_engine.py` | NEW — API ASR engine |
| `revospeech/tts/revolab_engine.py` | NEW — API TTS engine |
| `revospeech/asr/__init__.py` | Replace NotImplementedError with engine import |
| `revospeech/tts/__init__.py` | Same |
| `revospeech/models/asr/revolab-whisper.yaml` | NEW — API model manifest |
| `revospeech/models/tts/revolab-voice.yaml` | NEW — API model manifest |
| `pyproject.toml` | Add `httpx` to `[api]` extras |

---

## Task 1: Add a New Local Model (Same Backend)

If the new model uses an already-supported backend (sherpa-onnx for ASR, revovoice for TTS), you only need a YAML manifest.

### Steps

1. **Create a YAML manifest** in `revospeech/models/{task}/` or `~/.config/revospeech/models/{task}/`:

```yaml
name: my-new-model
task: asr
mode: local                      # "local" (default) or "api"
backend: sherpa-onnx
model_type: transducer
model_url: "https://..."         # Download URL or HF repo id
sample_rate: 16000
language: en
description: "Human-readable description"
hf_private: false
files:
  encoder: "encoder.onnx"
  decoder: "decoder.onnx"
  joiner: "joiner.onnx"
  tokens: "tokens.txt"
# New optional fields (all have safe defaults):
# size_mb: 150.0
# capabilities: ["word-timestamps", "streaming"]
# languages: ["en", "zh"]
# tags: ["fast", "multilingual"]
# license: "Apache-2.0"
# sha256: "abc123..."
# min_ram_mb: 2048
# min_vram_mb: 0
```

2. **Test it:**
```python
from revospeechpeech.asr import ASR
asr = ASR('my-new-model')
result = asr.transcribe('test.wav')
print(result.text)
```

3. **Verify discovery:**
```bash
revospeech models-info my-new-model
revospeech search "english fast"
```

### Rules
- The `name` must be unique within the same `task`.
- The `files` dict keys must match what the engine expects.
- All new fields (`mode`, `capabilities`, etc.) are optional with safe defaults.
- Existing manifests without new fields still work unchanged.

---

## Task 2: Add a New Backend for ASR

To support a new inference backend (local or API).

### Steps

1. **Create engine file** at `revospeech/asr/{backend}_engine.py`:

```python
from .base import BaseASR
from .result import Segment, Transcript

class MyBackendASR(BaseASR):
    def __init__(self, model_name: str, device: str = "auto") -> None:
        super().__init__(model_name, device)
        from revospeechpeech.registry import get, ensure_model
        manifest = get(model_name, "asr")
        # For local: ensure_model(manifest), load weights
        # For API: get_api_key(), init httpx client

    def transcribe(self, audio_path: str, **kwargs) -> Transcript:
        # Run inference, return Transcript(text=, segments=, language=)
        ...

    # Optional: override stream_transcribe() for streaming support
    def stream_transcribe(self, audio_path, **kwargs):
        # Default raises NotImplementedError
        ...
```

2. **Register in factory** — edit `revospeech/asr/__init__.py`:

For local backends, add to the `manifest.mode == "local"` branch:
```python
if manifest.backend == "my-backend":
    from .my_backend_engine import MyBackendASR
    return MyBackendASR(model_name, device)
```

For API backends, add to the `manifest.is_api` branch:
```python
if manifest.backend == "my-api-backend":
    from .my_api_engine import MyApiASR
    return MyApiASR(model_name, device)
```

3. **Add dependency** to `pyproject.toml` as optional extra.
4. **Add tests** with mocked backend.
5. **Add a YAML manifest** for at least one model.

### Rules
- Must inherit from `BaseASR` and implement `transcribe()`.
- Must return a `Transcript` object.
- Must lazy-import backend dependencies (so the library works without them).
- For API backends: use `get_api_key()` from `revospeech.config`, raise `RevosConfigError` if missing.
- For local backends: use `ensure_model()` for downloads.

---

## Task 3: Add a New Backend for TTS

Same pattern as ASR, but inherit from `BaseTTS` and implement `synthesize()`.
Return `Audio(samples=np.ndarray, sample_rate=int)`.

`BaseTTS` provides `synthesize_long()` for long text — it auto-splits on sentence boundaries (Latin + CJK punctuation) and concatenates with silence gaps.

---

## Key File Locations

| Purpose | Location |
|---------|----------|
| ASR engine (sherpa-onnx) | `revospeech/asr/sherpa_engine.py` |
| TTS engine (RevoVoice) | `revospeech/tts/revovoice_engine.py` |
| ASR base class | `revospeech/asr/base.py` (includes `stream_transcribe` no-op) |
| TTS base class | `revospeech/tts/base.py` (includes `synthesize_long`, `synthesize_streaming` no-op) |
| ASR result types | `revospeech/asr/result.py` (Segment, Transcript) |
| TTS result types | `revospeech/tts/result.py` (Audio, Audio.concatenate) |
| ASR factory | `revospeech/asr/__init__.py` (dispatches by mode + backend) |
| TTS factory | `revospeech/tts/__init__.py` (dispatches by mode + backend) |
| Exception hierarchy | `revospeech/exceptions.py` (RevosError + 4 subclasses) |
| Config management | `revospeech/config.py` (API key resolution: arg > env > config.yaml) |
| Model status | `revospeech/registry/status.py` (ModelStatus, check_model, list_model_statuses) |
| Model registry | `revospeech/registry/registry.py` (thread-safe singleton dict) |
| Manifest loader | `revospeech/registry/manifest.py` (ModelManifest with 18 fields) |
| Model downloader | `revospeech/registry/downloader.py` (security-hardened extraction) |
| Remote catalog | `revospeech/catalog.py` (GitHub API with retry + cache) |
| Device detection | `revospeech/device.py` |
| Usage tracking | `revospeech/usage.py` (JSONL with rotation, thread-safe callbacks) |
| CLI entry point | `revospeech/cli/main.py` (Click: transcribe, synthesize, models, models-info, search, catalog, info, config) |
| Bundled manifests | `revospeech/models/{asr,tts}/*.yaml` |
| User manifests | `~/.config/revospeech/models/**/*.yaml` |
| Model cache | `~/.cache/revospeech/{model_name}/` |
| Config file | `~/.config/revospeech/config.yaml` |
| API key env var | `REVOLAB_API_KEY` |

---

## Manifest Schema (ModelManifest)

All fields except `name`, `task`, `backend` are optional with safe defaults.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | required | Unique identifier within task |
| `task` | str | required | `"asr"` or `"tts"` |
| `backend` | str | required | Engine name (e.g., `"sherpa-onnx"`, `"revovoice"`) |
| `mode` | str | `"local"` | `"local"` or `"api"` |
| `api_endpoint` | str | `""` | URL for API backend |
| `model_url` | str | `""` | Download URL or HF repo id |
| `model_type` | str | `""` | Architecture type |
| `sample_rate` | int | `16000` | Audio sample rate |
| `language` | str | `""` | Primary language |
| `description` | str | `""` | Human-readable description |
| `files` | dict | `{}` | Logical name → expected filename |
| `hf_private` | bool | `false` | Requires HF auth |
| `revision` | str | `""` | Pin to commit/tag |
| `size_mb` | float | `0.0` | Download size |
| `capabilities` | list | `[]` | e.g., `["streaming", "voice-cloning"]` |
| `languages` | list | `[]` | e.g., `["en", "zh"]` |
| `tags` | list | `[]` | e.g., `["fast", "multilingual"]` |
| `license` | str | `""` | Model weight license |
| `sha256` | str | `""` | Download integrity hash |
| `min_ram_mb` | int | `0` | RAM requirement |
| `min_vram_mb` | int | `0` | VRAM requirement |

Properties: `is_local` (mode == "local"), `is_api` (mode == "api")

---

## Exception Hierarchy

```
RevosError (base)
├── RevosConfigError    — missing API key, bad config
├── RevosModelError     — model not found, download failed
├── RevosEngineError    — inference failure
└── RevosAudioError     — unsupported format, corrupt file
```

All exceptions have a `suggestion` attribute with a fix instruction. CLI commands wrap these and show friendly messages on stderr.

---

## Remote Catalog

Users discover and install models from the GitHub repo's `revospeech/models/` directory via the GitHub API. Responses are cached for 1 hour in `~/.cache/revospeech/catalog_cache.json` (repo-aware — changing `REVOLAB_REPO` invalidates cache). Network calls have retry (3x) with exponential backoff and 10s timeout.

```bash
revospeech catalog list              # List models from GitHub (cached 1h)
revospeech catalog list -t tts       # Filter by task
revospeech catalog pull revovoice    # Install manifest locally
```

---

## Testing Checklist

When adding any new model or backend, verify:

- [ ] `uv run pytest tests/ -v` — all tests pass (currently 98 tests)
- [ ] `uv run ruff check revospeech/ tests/` — lint clean
- [ ] `uv run ruff format --check revospeech/ tests/` — format clean
- [ ] Gate tests pass: `uv run pytest tests/test_compat_gates.py -v`
- [ ] Factory function returns correct engine type
- [ ] Manifest loads and registers correctly
- [ ] `revospeech models` shows the new model with correct status
- [ ] `revospeech search` finds the model
- [ ] For API backends: missing API key → clear error with suggestion
- [ ] For local backends: `from revospeechpeech.asr import ASR` works without optional deps
- [ ] ImportErrors are helpful when optional backend is missing

---

## Current Backends

### ASR Backends

| Backend | Engine File | Dependencies | Mode | Notes |
|---------|------------|-------------|------|-------|
| sherpa-onnx | `revospeech/asr/sherpa_engine.py` | sherpa-onnx, onnxruntime | local | Zipformer transducer via ONNX |

### TTS Backends

| Backend | Engine File | Dependencies | Mode | Notes |
|---------|------------|-------------|------|-------|
| revovoice | `revospeech/tts/revovoice_engine.py` | omnivoice, torch | local | Diffusion zero-shot TTS, 600+ languages |

### Pending Backends

| Backend | Engine File | Dependencies | Mode | Status |
|---------|------------|-------------|------|--------|
| revolab-api | `revospeech/asr/revolab_engine.py` | httpx (optional) | api | Blocked — needs API contract |

---

## Contribution Pattern for Community

The library is designed so community members can PR new backends and models:

1. **New local model (easy):** Just add a YAML manifest to `revospeech/models/{task}/`. No code changes.
2. **New local backend (medium):** Create engine file implementing BaseASR/BaseTTS, add factory branch, add optional dep, add manifest + tests.
3. **New API backend (medium):** Same as local backend but set `mode: api` in manifest, use `get_api_key()` for auth, use `httpx` for HTTP.
4. **New task type (hard):** Create new package `revospeech/{task}/` with base, result, engine, factory. Add CLI commands. See existing ASR/TTS as reference.

The factory already handles mode dispatch. API key validation is built-in. Just plug in a new engine.
