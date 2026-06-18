# Model Catalog

This page lists models shipped with RevoSpeech. For live status (installed / needs-download), run `revospeech models`.

---

## ASR — Speech Recognition

| Model | Backend | Languages | Size | Capabilities | License | Notes |
|-------|---------|-----------|------|--------------|---------|-------|
| `zipformer-v2` | sherpa-onnx (Zipformer transducer) | English | ~80 MB | `word-timestamps` | Apache-2.0 | Fast, CPU-friendly, English-only |

### Roadmap

- Whisper-based ONNX (multilingual) — planned
- Smaller/faster English-only model — planned
- Streaming ASR via sherpa-onnx OnlineRecognizer — planned
- Language auto-detection — planned

---

## TTS — Speech Synthesis

| Model | Backend | Languages | Size | Capabilities | License | Notes |
|-------|---------|-----------|------|--------------|---------|-------|
| `revovoice` | RevoVoice (diffusion) | 600+ (multilingual) | ~1.2 GB | `voice-cloning`, `streaming` | MIT (code) | Zero-shot cloning; GPU recommended; gated HF repo |
| `vits-ms` | VITS | Malay (`ms`) | ~500 MB | `multi-speaker` | — | 3 production speakers: sarah, paan, anwar |

### Roadmap

- Kokoro TTS (if license permits) — under evaluation
- Language-specific models for major languages — planned

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
published here once a reproducible benchmark harness lands. Tracked in
[TODO.md](https://github.com/khursanirevo/revospeech/blob/master/TODO.md) Phase 7.1.

## See also

- [Extension guide](extension.md)
- [Registry API](api/registry.md)
- [Available Models in README](https://github.com/khursanirevo/revospeech#available-models)
