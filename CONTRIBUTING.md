# Contributing to RevoS

Thank you for your interest in contributing! This guide covers everything you need.

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd revospeech
uv sync --extra dev

# Run checks
uv run ruff check revospeech/ tests/
uv run pytest tests/ -v

# Build
uv build
```

## Adding a New Model (Zero Code Changes)

If the model uses an existing backend (sherpa-onnx for ASR, revovoice for TTS), just add a YAML manifest:

```yaml
# ~/.config/revospeech/models/asr/my-model.yaml
name: my-model
task: asr
mode: local
backend: sherpa-onnx
model_type: transducer
model_url: "https://example.com/model.tar.bz2"
sample_rate: 16000
language: en
description: "My custom ASR model"
capabilities: ["word-timestamps", "streaming"]
languages: ["en"]
files:
  encoder: "encoder.onnx"
  decoder: "decoder.onnx"
  joiner: "joiner.onnx"
  tokens: "tokens.txt"
```

Then use it: `from revospeech.asr import ASR; ASR('my-model')`

### API Models

For cloud API backends, set `mode: api`:

```yaml
name: my-api-model
task: asr
mode: api
backend: my-api
api_endpoint: "https://api.example.com/v1"
description: "Cloud ASR via example API"
capabilities: ["streaming"]
languages: ["en", "multilingual"]
```

### Pinning Model Versions

For HuggingFace-hosted models, pin to a specific commit using the `revision` field:

```yaml
revision: "a1b2c3d"    # Pin to specific commit hash
# revision: "v1.0.0"   # Or use a git tag
```

For gated models, set `hf_private: true`.

### Remote Catalog

Models added to `revospeech/models/` in this repo are automatically available via the remote catalog:

```bash
revospeech catalog list           # Browse models from this repo
revospeech catalog pull <name>    # Install a model locally
```

### Manifest Schema

All fields except `name`, `task`, `backend` are optional with safe defaults:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | required | Unique identifier within task |
| `task` | str | required | `"asr"` or `"tts"` |
| `backend` | str | required | Engine name |
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

## Adding a New Backend

1. Create `revospeech/{task}/{backend}_engine.py` inheriting from the base class
2. Register in the factory function (`revospeech/{task}/__init__.py`)
3. Add optional dependency to `pyproject.toml`
4. Add tests with mocked backend
5. Add at least one YAML manifest

For API backends: use `get_api_key()` from `revospeech.config` for auth, set `mode: api` in manifest.

See [AGENTS.md](AGENTS.md) for detailed instructions.

## Model Discovery

Users can discover models via CLI and Python API:

```bash
# CLI
revospeech models                    # List all models with status
revospeech models --ready            # Only ready-to-use models
revospeech models --mode api         # Only API models
revospeech models-info zipformer-v2  # Detailed model info
revospeech search "english fast"     # Fuzzy search
```

```python
# Python
import revospeech
revospeech.list_models(task="asr", status="ready")
revospeech.search_models("english fast")
revospeech.check_model("zipformer-v2")
```

## Configuration

API keys and settings are managed via:

```bash
# CLI
revospeech config set-api-key       # Save API key (stored in ~/.config/revospeech/config.yaml)
export REVOLAB_API_KEY=rv-...  # Or use env var
```

Resolution order: constructor arg > `REVOLAB_API_KEY` env var > `~/.config/revospeech/config.yaml`

## Development Workflow

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Make changes with tests
3. Run lint and tests:
   ```bash
   uv run ruff check revospeech/ tests/
   uv run pytest tests/ -v
   ```
4. Commit with clear messages
5. Open a pull request

## Code Style

- Python 3.11+, formatted by ruff (line length 88)
- Lazy imports for optional dependencies (omnivoice, httpx)
- Factory functions as public API (not classes)
- YAML manifests for model configuration
- Custom exceptions from `revospeech.exceptions` (never bare `ValueError`/`KeyError`)

## Testing

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=revospeech --cov-report=term-missing

# Only fast tests (exclude real inference)
uv run pytest tests/ -v -m "not slow"
```

Markers: `slow` (real model download), `api` (requires API key), `roundtrip` (TTS → ASR validation)

## Project Structure

```
revospeech/
  asr/             # ASR engines (sherpa-onnx, future: revolab API)
  tts/             # TTS engines (revovoice, future: revolab API)
  registry/        # Model manifests, registry, downloader, status
  catalog.py       # Remote model catalog (GitHub-based, cached)
  config.py        # API key & configuration management
  exceptions.py    # Custom exception hierarchy
  cli/             # Click CLI
  models/          # Bundled YAML manifests
tests/             # Test suite
```

## Exception Hierarchy

```
RevosError (base)
├── RevosConfigError    — missing API key, bad config
├── RevosModelError     — model not found, download failed
├── RevosEngineError    — inference failure
└── RevosAudioError     — unsupported format, corrupt file
```

All exceptions have a `suggestion` attribute with a fix instruction.

## Need Help?

Check [AGENTS.md](AGENTS.md) for detailed architecture docs and extension guides.
