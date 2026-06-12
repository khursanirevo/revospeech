# REPO_ANALYSIS — RevoS Library Full Analysis

> Auto-generated analysis of the `revos` repository at commit `fcd2f52` on 2026-06-12.

## Project Summary

**revos** is a unified Python library for speech AI — providing **ASR** (automatic speech recognition) and **TTS** (text-to-speech) using open ONNX models. It wraps `sherpa-onnx` for ASR and `omnivoice` for TTS behind a clean Python + CLI API.

| Metric | Value |
|---|---|
| Package version | 0.1.0 |
| Python | >=3.11 |
| Source lines | ~1,765 (Python) |
| Tests | ~100 across 13 files |
| Build system | hatchling |
| License | MIT |

## Architecture

```
revos/
├── __init__.py          # Lazy imports for ASR, TTS, configure_logging
├── logging_config.py    # Idempotent logging setup
├── device.py            # CUDA auto-detection via onnxruntime
├── usage.py             # JSONL telemetry with callback observer pattern
├── catalog.py           # Remote model catalog (GitHub Contents API)
├── asr/                 # Speech recognition
│   ├── base.py          #   ABC: BaseASR
│   ├── sherpa_engine.py #   Concrete: SherpaOnnxASR (Zipformer transducer)
│   ├── audio.py         #   read_waveform() — resample to 16kHz
│   └── result.py        #   Segment, Transcript dataclasses
├── tts/                 # Text-to-speech
│   ├── base.py          #   ABC: BaseTTS + synthesize_long template method
│   ├── revovoice_engine.py # Concrete: RevoVoiceTTS (OmniVoice wrapper)
│   └── result.py        #   Audio dataclass with save/concatenate
├── registry/            # Model management
│   ├── manifest.py      #   ModelManifest dataclass + YAML loader
│   ├── registry.py      #   Singleton in-memory dict, auto-discovery
│   └── downloader.py    #   Archive download with security hardening
├── cli/
│   └── main.py          #   Click CLI: transcribe, synthesize, models, info, catalog
└── models/              # Bundled model YAML manifests
    ├── asr/zipformer_v2.yaml
    └── tts/revovoice.yaml
```

## Analysis Files

| # | File | Topic | Lines |
|---|---|---|---|
| 01 | `01_core_package.md` | `__init__`, logging, device, usage, pyproject.toml | 407 |
| 02 | `02_asr_module.md` | ASR engine, audio I/O, sherpa-onnx integration | 352 |
| 03 | `03_tts_module.md` | TTS engine, voice cloning, long-form synthesis | 399 |
| 04 | `04_registry_catalog.md` | Model registry, downloader, remote catalog | 563 |
| 05 | `05_cli.md` | Click CLI — all 7 commands documented | 365 |
| 06 | `06_tests.md` | Full test suite analysis (~100 tests, coverage gaps) | 633 |
| 07 | `07_docs_examples.md` | README, AGENTS.md, examples, CHANGELOG | 684 |
| 08 | `08_ci_config.md` | CI/CD, publish workflow, pre-commit, pyproject | 562 |

## Key Findings Across All Analyses

### Notable Design Decisions
- **Factory functions** (`ASR()`, `TTS()`) with lazy backend dispatch — no heavy imports at package load
- **Template method** for `synthesize_long()` — split on sentence boundaries (Latin + CJK), synthesize chunks, concatenate
- **Security-first downloader** — path-traversal protection in archive extraction
- **OIDC Trusted Publishing** for PyPI releases (no stored secrets)

### Potential Issues
1. **CI branch mismatch** — CI watches `main` but git default branch is `master`
2. **Version duplication** — `"0.1.0"` hardcoded in both `__init__.py` and `pyproject.toml`
3. **No `__dir__` override** in `__init__.py` — lazy exports invisible to tab-completion
4. **Usage log has no rotation** — `~/.cache/revos/usage.jsonl` grows unbounded
5. **CLI error handling gap** — `transcribe`/`synthesize` commands show raw tracebacks on engine errors
6. **Thread safety** — registry `_models` dict and usage `_callbacks` list are unprotected globals
7. **Dead test fixtures** — `mock_recognizer` and `mock_tts_model` in conftest.py are never used
8. **Missing reference audio** — `tts_voice_cloning.py` example references non-existent `reference_speaker.wav`
9. **Build tool inconsistency** — CI uses `uv build`, publish uses `python -m build`

### What This Repo Does Well
- Clean layered architecture with clear separation of concerns
- Security-hardened downloader (path traversal, hash verification)
- Graceful GPU→CPU fallback
- YAML-driven model configuration (easy to add models)
- Remote catalog with GitHub-based browse and pull
- Comprehensive test suite with real integration tests gated behind `pytest.mark.slow`
