# RevoSpeech

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/khursanirevo/revospeech/ci.yml?branch=master)](.github/workflows/ci.yml)

A unified Python library for speech AI — ASR and TTS using open models.

## Quickstart

```bash
pip install revospeech
```

```python
from revospeech import ASR, TTS

# Transcribe
result = ASR().transcribe("audio.wav")
print(result.text)

# Synthesize
TTS().synthesize("Hello, world!").save("out.wav")
```

That's it. Models auto-download on first use.

## Installation

```bash
# Core (ASR support)
pip install revospeech

# With TTS support (RevoVoice — requires PyTorch)
pip install revospeech[tts]

# With GPU support
pip install revospeech[gpu]

# Everything (GPU + TTS)
pip install revospeech[all]

# Or with uv
uv add revospeech
```

### HuggingFace Login (Required for TTS)

> **Note:** The RevoVoice TTS model is hosted on a private HuggingFace repository. You **must** log in before using TTS.

```bash
pip install huggingface-hub
huggingface-cli login
```

Get your token at https://huggingface.co/settings/tokens

### Important Notes

- `revospeech[gpu]` and `revospeech[all]` install `onnxruntime-gpu`, which **conflicts** with `onnxruntime`. If you already have `revospeech` installed, uninstall it first before installing the GPU variant before installing the GPU variant.
- Audio formats supported: WAV, FLAC, OGG, and any format supported by `libsndfile`.

## Quick Start

### ASR (Automatic Speech Recognition)

```python
from revospeech.asr import ASR

# No args: auto-selects the first ready local ASR model
asr = ASR()
# Or pick one explicitly:
asr = ASR('zipformer-v2')

result = asr.transcribe('meeting.wav')

print(result.text)        # Full transcript
print(result.language)    # Detected language
for seg in result.segments:
    print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")

# Save transcript (format inferred from extension: .txt/.json/.srt/.vtt)
result.save('meeting.srt')

# Readable repr
print(result)   # Transcript(text="...", duration=12.3s, segments=5)

# Batch: transcribe many files in parallel
report = asr.transcribe_batch(['a.wav', 'b.wav', 'c.wav'], max_workers=4)
print(f"{report.succeeded}/{report.total} succeeded")
for item in report.items:
    if item.succeeded:
        print(item.result.text)
```

Passing an API key for cloud backends and disabling auto-download:

```python
# Persist the key for the session before loading the engine
asr = ASR('my-api-model', api_key='rv-...')

# Skip automatic downloads (raises if the model isn't already cached)
asr = ASR('zipformer-v2', auto_download=False)
```

### TTS (Text-to-Speech)

```python
from revospeech.tts import TTS

# No args: auto-selects the first ready local TTS model
tts = TTS()
# Or pick one explicitly:
tts = TTS('revovoice')

# Basic synthesis
audio = tts.synthesize('Hello, how are you?')
audio.save('greeting.wav')

# Play through the system output (requires `pip install sounddevice`)
audio.play()

# Readable repr
print(audio)   # Audio(duration=1.5s, sample_rate=22050Hz, samples=33075)

# Voice cloning (with reference audio)
audio = tts.synthesize(
    'This will sound like the reference speaker.',
    ref_audio='speaker.wav',
    ref_text='Sample of the speaker talking.',
)
audio.save('cloned.wav')

# Batch: synthesize many texts in parallel
report = tts.synthesize_batch(
    ['Hello.', 'How are you?', 'Goodbye.'],
    output_dir='clips/',   # writes clips/audio_0.wav, audio_1.wav, ...
)
print(f"{report.succeeded}/{report.total} succeeded")
```

### Model Discovery

```python
import revospeech
from revospeech.catalog import recommend_models

# List all models with status
revospeech.list_models()

# Filter by task, mode, status
revospeech.list_models(task="asr", status="ready")
revospeech.list_models(mode="api")

# Fuzzy search
revospeech.search_models("english fast")

# Recommend top 3 models by size (smallest = fastest)
recommend_models(task="asr", language="en")
recommend_models(task="tts")

# Check a specific model
status = revospeech.check_model("zipformer-v2")
print(status.is_ready)
```

### CLI

```bash
# Transcribe audio (plain text output by default)
revospeech transcribe -m zipformer-v2 audio.wav

# Choose output format
revospeech transcribe -m zipformer-v2 --format json audio.wav
revospeech transcribe -m zipformer-v2 --format srt audio.wav
revospeech transcribe -m zipformer-v2 --format vtt audio.wav
# (--json and --srt still work but are deprecated; prefer --format)

# Transcribe multiple files in one go (batch)
revospeech transcribe -m zipformer-v2 a.wav b.wav c.wav

# Synthesize speech
revospeech synthesize -m revovoice -t "Hello, world!" -o output.wav

# From text file
revospeech synthesize -m revovoice -f script.txt -o audiobook.wav

# Batch: synthesize one clip per line of inputs.txt into ./tts_output/
revospeech synthesize -m revovoice --file-list inputs.txt -o tts_output

# List available models (with status icons)
revospeech models
revospeech models --ready           # Only ready-to-use models
revospeech models --mode api        # Only API models
revospeech models --task asr        # Filter by task

# Detailed model info
revospeech models-info zipformer-v2

# Fuzzy search local registry
revospeech search "english fast"

# Browse remote catalog
revospeech catalog list
revospeech catalog list --task asr

# Search the remote catalog by name / language / task
revospeech catalog search "english" --task asr --language en

# Pull a model from the catalog
revospeech catalog pull revovoice

# API key management
revospeech config set-api-key       # Prompt and save key
revospeech config show-api-key      # Show masked key status

# Interactive first-time setup wizard (task, model, API key)
revospeech setup

# Show version, device, cache, and API key status
revospeech info

# Global flags
revospeech --verbose transcribe ... # Debug-level logging
revospeech --quiet   transcribe ... # Warnings only
```

## Examples

The `examples/` directory has numbered progressive examples:

| # | File | What it shows |
|---|------|---------------|
| 01 | `01_quickstart.py` | Minimal ASR + TTS |
| 02 | `02_pick_model.py` | Listing and choosing models |
| 03 | `03_transcribe_options.py` | Transcription options |
| 04 | `04_voice_cloning.py` | Voice cloning |
| 05 | `05_long_text.py` | Long text synthesis |
| 06 | `06_batch_processing.py` | Batch ASR + TTS |
| 07 | `07_api_backend.py` | API backend pattern |
| 08 | `08_roundtrip_test.py` | TTS -> ASR validation |
| 09 | `09_custom_backend.py` | Custom backend subclass |
| 10 | `10_streaming_tts.py` | Streaming synthesis |
| 11 | `11_batch_directory.py` | Directory batch |
| 12 | `12_cli_only.sh` | Pure CLI workflow |

Run any example with `python examples/01_quickstart.py`.

## Available Models

| Model | Task | Mode | Size | Languages | Quality Notes | Access |
|-------|------|------|------|-----------|---------------|--------|
| `zipformer-v2` | ASR | local | ~80 MB | English | Fast, CPU-friendly; supports word timestamps | Open |
| `revovoice` | TTS | local | ~1.2 GB | 600+ (multilingual) | High-quality zero-shot cloning, GPU recommended | **Gated** |

Run `revospeech models` for the live table with status indicators, or `revospeech models-info <name>` for full details (capabilities, min RAM/VRAM, license).

### Model Directory

```
revospeech/models/
├── asr/
│   └── zipformer_v2.yaml    # Open — downloads from GitHub releases
└── tts/
    └── revovoice.yaml       # Gated — requires HF login + approval
```

### Gated Model Access

Some models (like `revovoice`) are hosted on private HuggingFace repositories
and require approval before use. Approved models download automatically on first
use - you do not need to manually download them.

1. **Log in to HuggingFace:**
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```
   Get your token at https://huggingface.co/settings/tokens

2. **Request access:** Visit the model's HuggingFace page and submit an access request. The repo owner will review and approve.

3. **Use the model:** Once approved, the model will download automatically on first use:
   ```python
   from revospeech.tts import TTS
   tts = TTS('revovoice')  # Downloads on first call, then cached
   ```

Auto-download is on by default for local models. To skip it (e.g. in air-gapped
environments), pass `auto_download=False` and ensure the model is already cached:

```python
asr = ASR('zipformer-v2', auto_download=False)
```

> **For team members adding models:** If your model is gated, set `hf_private: true` in the YAML manifest. This tells RevoSpeech to check HF authentication before downloading.

## Local vs Cloud API

RevoSpeech supports two execution modes for both ASR and TTS:

| Aspect | Local | Cloud API |
|--------|-------|-----------|
| **Setup** | `pip install revospeech` + download model weights | `pip install revospeech` + set API key |
| **Cost** | Free (open models) | Per-use pricing |
| **Latency** | Depends on hardware (50ms–2s typical) | Network round-trip (~100–500ms) |
| **Privacy** | Audio never leaves your machine | Audio sent to revolab cloud |
| **Quality** | Limited by open models | State-of-the-art revolab models |
| **Languages** | Depends on downloaded model | 100+ languages, regularly updated |
| **Voice cloning** | Limited (depends on model) | High-quality zero-shot cloning |
| **Offline** | Yes | No (requires internet) |
| **GPU required** | Optional (CPU works, slower) | No (cloud handles compute) |

### Using Local Models

```python
from revospeech import ASR, TTS

# Local ASR — model auto-downloads on first use
asr = ASR("zipformer-v2")
result = asr.transcribe("audio.wav")

# Local TTS — gated models require HF login first
tts = TTS("revovoice")
audio = tts.synthesize("Hello, world!")
```

### Using revolab Cloud API

```bash
# Set API key once
revospeech config set-api-key
```

```python
from revospeech import ASR, TTS

# Same API, just a different model name
asr = ASR("revolab-asr-v1")  # uses cloud automatically
result = asr.transcribe("audio.wav")

tts = TTS("revolab-tts-v1")
audio = tts.synthesize("Hello, world!")
```

> **Note:** revolab API backends ship in a future release. The factory pattern is in place — once `revolab-asr-v1` and `revolab-tts-v1` manifests are registered, the same user code will work without changes.

### Choosing Between Modes

- **Pick Local** if: you need offline use, have privacy constraints, want zero per-use cost, or have a capable GPU
- **Pick API** if: you want state-of-the-art quality, don't want to manage model weights, need many languages, or have limited local hardware

Both modes use the same Python API. The only difference is the model name you pass to `ASR()` or `TTS()`.

## Configuration

### API Keys

For cloud API backends, set your API key:

```bash
# Option 1: Environment variable
export REVOLAB_API_KEY=rv-your-key-here

# Option 2: CLI command (saves to ~/.config/revospeech/config.yaml)
revospeech config set-api-key
```

Resolution order: constructor arg > `REVOLAB_API_KEY` env var > `~/.config/revospeech/config.yaml`

### Catalog Source

Override the catalog source with:
```bash
export REVOS_CATALOG_REPO="myorg/revospeech"    # env var
```
Or in `~/.config/revospeech/config.yaml`:
```yaml
catalog_repo: "myorg/revospeech"
```

## Adding Custom Models

Add a YAML manifest to `~/.config/revospeech/models/`:

```yaml
# ~/.config/revospeech/models/asr/my-model.yaml
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

Then use it: `from revospeech.asr import ASR; asr = ASR('my-custom-model')`

### API Models

```yaml
# ~/.config/revospeech/models/asr/my-api-model.yaml
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

The catalog fetches available models directly from this repository on GitHub. Team members add YAML manifests to `revospeech/models/` and users discover them without upgrading.

```bash
# Browse all available models from the repo
revospeech catalog list

# Install a model locally
revospeech catalog pull revovoice
```

## Documentation

- [AGENTS.md](AGENTS.md) — Architecture guide for AI agents and contributors
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute
- [TODO.md](TODO.md) — Full backlog and roadmap

## Project Structure

```
revospeech/
├── revospeech/
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
