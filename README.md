# RevoS

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/revos/revos/ci.yml?branch=master)](.github/workflows/ci.yml)

A unified Python library for speech AI — ASR and TTS using open models.

## Installation

```bash
# Core (ASR support)
pip install revos

# With TTS support (RevoVoice — requires PyTorch)
pip install revos[tts]

# With GPU support
pip install revos[gpu]

# Everything (GPU + TTS)
pip install revos[all]

# Or with uv
uv add revos
```

### HuggingFace Login (Required for TTS)

> **Note:** The RevoVoice TTS model is hosted on a private HuggingFace repository. You **must** log in before using TTS.

```bash
pip install huggingface-hub
huggingface-cli login
```

Get your token at https://huggingface.co/settings/tokens

### Important Notes

- `revos[gpu]` and `revos[all]` install `onnxruntime-gpu`, which **conflicts** with `onnxruntime`. If you already have `revos` installed, uninstall it first before installing the GPU variant.
- Audio formats supported: WAV, FLAC, OGG, and any format supported by `libsndfile`.

## Quick Start

### ASR (Automatic Speech Recognition)

```python
from revos.asr import ASR

asr = ASR('zipformer-v2')
result = asr.transcribe('meeting.wav')

print(result.text)        # Full transcript
print(result.language)    # Detected language
for seg in result.segments:
    print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
```

### TTS (Text-to-Speech)

```python
from revos.tts import TTS

# Basic synthesis
tts = TTS('revovoice')
audio = tts.synthesize('Hello, how are you?')
audio.save('greeting.wav')

# Voice cloning (with reference audio)
audio = tts.synthesize(
    'This will sound like the reference speaker.',
    ref_audio='speaker.wav',
    ref_text='Sample of the speaker talking.',
)
audio.save('cloned.wav')
```

### Model Discovery

```python
import revos

# List all models with status
revos.list_models()

# Filter by task, mode, status
revos.list_models(task="asr", status="ready")
revos.list_models(mode="api")

# Fuzzy search
revos.search_models("english fast")

# Check a specific model
status = revos.check_model("zipformer-v2")
print(status.is_ready)
```

### CLI

```bash
# Transcribe audio
revos transcribe -m zipformer-v2 audio.wav

# JSON output
revos transcribe -m zipformer-v2 --json audio.wav

# SRT subtitles
revos transcribe -m zipformer-v2 --srt audio.wav

# Synthesize speech
revos synthesize -m revovoice -t "Hello, world!" -o output.wav

# From text file
revos synthesize -m revovoice -f script.txt -o audiobook.wav

# List available models (with status icons)
revos models
revos models --ready           # Only ready-to-use models
revos models --mode api        # Only API models
revos models --task asr        # Filter by task

# Detailed model info
revos models-info zipformer-v2

# Fuzzy search
revos search "english fast"

# Browse remote catalog
revos catalog list

# Pull a model from the catalog
revos catalog pull revovoice

# API key management
revos config set-api-key

# Show environment info
revos info
```

## Available Models

| Model | Task | Backend | Mode | Languages | Access | Description |
|-------|------|---------|------|-----------|--------|-------------|
| `zipformer-v2` | ASR | sherpa-onnx | local | English | Open | Zipformer small transducer model |
| `revovoice` | TTS | RevoVoice | local | 600+ | **Gated** | Zero-shot multilingual TTS with voice cloning |

### Model Directory

```
revos/models/
├── asr/
│   └── zipformer_v2.yaml    # Open — downloads from GitHub releases
└── tts/
    └── revovoice.yaml       # Gated — requires HF login + approval
```

### Gated Model Access

Some models (like `revovoice`) are hosted on private HuggingFace repositories
and require approval before use.

1. **Log in to HuggingFace:**
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```
   Get your token at https://huggingface.co/settings/tokens

2. **Request access:** Visit the model's HuggingFace page and submit an access request. The repo owner will review and approve.

3. **Use the model:** Once approved, the model will download automatically on first use:
   ```python
   from revos.tts import TTS
   tts = TTS('revovoice')  # Will prompt for HF login if not authenticated
   ```

> **For team members adding models:** If your model is gated, set `hf_private: true` in the YAML manifest. This tells RevoS to check HF authentication before downloading.

## Configuration

### API Keys

For cloud API backends, set your API key:

```bash
# Option 1: Environment variable
export REVOLAB_API_KEY=rv-your-key-here

# Option 2: CLI command (saves to ~/.config/revos/config.yaml)
revos config set-api-key
```

Resolution order: constructor arg > `REVOLAB_API_KEY` env var > `~/.config/revos/config.yaml`

### Catalog Source

Override the catalog source with:
```bash
export REVOS_CATALOG_REPO="myorg/revos"    # env var
```
Or in `~/.config/revos/config.yaml`:
```yaml
catalog_repo: "myorg/revos"
```

## Adding Custom Models

Add a YAML manifest to `~/.config/revos/models/`:

```yaml
# ~/.config/revos/models/asr/my-model.yaml
name: my-custom-model
task: asr
mode: local
backend: sherpa-onnx
model_type: transducer
model_url: "https://example.com/models/my-model.tar.bz2"
sample_rate: 16000
language: en
description: "My custom ASR model"
capabilities: ["word-timestamps"]
languages: ["en"]
files:
  encoder: "encoder.onnx"
  decoder: "decoder.onnx"
  joiner: "joiner.onnx"
  tokens: "tokens.txt"
```

Then use it: `from revos.asr import ASR; asr = ASR('my-custom-model')`

### API Models

```yaml
# ~/.config/revos/models/asr/my-api-model.yaml
name: my-api-model
task: asr
mode: api
backend: my-api
api_endpoint: "https://api.example.com/v1"
description: "Cloud ASR"
capabilities: ["streaming"]
languages: ["en"]
```

### Pinning Model Versions

For HuggingFace-hosted models, pin to a specific commit hash or tag using the `revision` field:

```yaml
revision: "a1b2c3d"       # Pin to specific commit hash
# revision: "v1.0.0"      # Or use a git tag
```

Without `revision`, the latest version from the default branch is used.

### Remote Catalog

The catalog fetches available models directly from this repository on GitHub. Team members add YAML manifests to `revos/models/` and users discover them without upgrading.

```bash
# Browse all available models from the repo
revos catalog list

# Install a model locally
revos catalog pull revovoice
```

## Documentation

- [AGENTS.md](AGENTS.md) — Architecture guide for AI agents and contributors
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute
- [TODO.md](TODO.md) — Full backlog and roadmap

## Project Structure

```
revos/
├── revos/
│   ├── asr/           # ASR engines (sherpa-onnx)
│   ├── tts/           # TTS engines (RevoVoice)
│   ├── registry/      # Model manifests, registry, downloader, status
│   ├── cli/           # Click CLI
│   ├── config.py      # API key & configuration management
│   ├── exceptions.py  # Custom exception hierarchy
│   ├── catalog.py     # Remote model catalog (GitHub, cached)
│   ├── device.py      # GPU/CPU auto-detection
│   └── models/        # Bundled YAML manifests
├── tests/
├── pyproject.toml
├── AGENTS.md
└── CONTRIBUTING.md
```

## License

MIT
