# Model Catalog

This page lists models shipped with RevoSpeech. For live status (installed / needs-download), run `revospeech models`.

---

## ASR — Speech Recognition

| Model | Backend | Mode | Languages | Size | Capabilities | License | Notes |
|-------|---------|------|-----------|------|--------------|---------|-------|
| `zipformer-v2` | sherpa-onnx (Zipformer transducer) | local | English | ~80 MB | `word-timestamps` | Apache-2.0 | Fast, CPU-friendly, English-only |
| `revolab-asr` | revolab (cloud) | api | `ms`, `en` | — | `streaming`, code-switching | — | Cloud ASR via revolab API; handles Malaysian + EN code-switch; requires `REVOLAB_API_KEY` |

### Roadmap

- Whisper-based ONNX (multilingual) — planned
- Smaller/faster English-only model — planned
- Streaming ASR via sherpa-onnx OnlineRecognizer — planned
- Language auto-detection — planned

---

## TTS — Speech Synthesis

| Model | Backend | Languages | Size | Capabilities | License | Notes |
|-------|---------|-----------|------|--------------|---------|-------|
| [`revovoice`](models/revovoice.md) | RevoVoice (diffusion) | 600+ (multilingual) | ~1.2 GB | `voice-cloning`, `streaming` | MIT (code) | Zero-shot cloning; GPU recommended; gated HF repo |
| [`vits-ms`](models/vits-ms.md) | VITS | Malay (`ms`) | ~500 MB | `multi-speaker` | — | 3 production speakers: sarah, paan, anwar |

### Roadmap

- Kokoro TTS (if license permits) — under evaluation
- Language-specific models for major languages — planned

---

## Util — Speech Restoration

| Model | Backend | Languages | Size | Capabilities | Notes |
|-------|---------|-----------|------|--------------|-------|
| [`sidon`](models/sidon.md) | sidon (ONNX) | — | ~940 MB | `denoise`, `dereverberation`, `bandwidth-extension` | 3-stage pipeline (mel -> w2v-BERT 2.0 -> DAC); output 48 kHz; mel front-end bundled |

Use standalone via `revospeech restore`, or as TTS post-processing via
`TTS(..., restore=True)`. See [Speech Restoration](util.md).

---

## Discovery commands

```bash
revospeech models                      # local table with status icons
revospeech catalog list                # browse remote catalog
revospeech catalog recommend           # best-for-task recommendations
revospeech catalog search "english"    # fuzzy search by name/tag/language
revospeech models-info <name>          # full manifest details
```

## Adding your own model

See the [Extension guide](extension.md) for the three paths: YAML-only for
existing backends, new local backend engine, or new API backend.

## Benchmarks

Quantitative benchmarks (WER for ASR, MOS for TTS, RTF for both) will be
published here once a reproducible benchmark harness lands.

## See also

- [Extension guide](extension.md)
- [Registry API](api/registry.md)
- [Available Models in README](https://github.com/khursanirevo/revospeech#available-models)
