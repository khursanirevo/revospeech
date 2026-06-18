# Changelog

All notable changes to RevoSpeech are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Auto-download models on first use (`auto_download=True` on `ASR` / `TTS`).
- `revospeech setup` interactive first-run wizard.
- `revospeech config set-api-key` / `show-api-key` commands.
- `revospeech transcribe --format text|json|srt|vtt` output format option.
- `revospeech transcribe a.wav b.wav c.wav` multi-file batch on the CLI.
- `revospeech synthesize --file-list inputs.txt` batch synthesis from a text file.
- `revospeech catalog recommend [--task T] [--language L]` for curated picks.
- `revospeech catalog list` now shows an installed/not-installed `Status` column.
- Color CLI output (TTY-aware, respects `NO_COLOR`).
- `BytesIO` / file-like object support for in-memory audio in `ASR.transcribe`.
- `Audio.play()` method (requires optional `sounddevice`).
- `Transcript.save(path)` for `srt`, `vtt`, `json`, `txt` formats.
- `Transcript.duration` property.
- `BatchReport.save(path)` JSON export.
- `BatchReport.__repr__`, `BatchResult.__repr__`, `Audio.__repr__`.
- Progress bars for model downloads (tqdm) with size and ETA.
- Fuzzy "did you mean?" model name suggestions.
- Zero-config defaults: `ASR()` and `TTS()` auto-select the smallest ready model.
- `api_key=` parameter on `ASR` / `TTS` constructors.
- `Path` and `str` accepted interchangeably for all file arguments.
- Catalog helpers `catalog_installed_status()` and `recommend_models()`.
- Type overloads for `ASR()` and `TTS()` factory call patterns.
- `py.typed` marker for PEP 561 type-info distribution.
- `mypy` type-check step in CI.
- Round-trip tests (`tests/test_roundtrip.py`) with `pytest.mark.roundtrip`
  marker and `fuzzy_match` helper for TTS -> ASR self-validation.
- Extension protocol tests (`tests/test_extension_protocol.py`) covering
  `BaseASR` / `BaseTTS` subclassing, factory dispatch, and manifest registration.
- Additional Phase 6 test coverage: corrupt/invalid audio, network errors,
  thread safety, base classes, top-level imports, non-WAV formats, edge cases.
- Example scripts `01`-`12` (progressive, runnable).
- mkdocs-material documentation site with mkdocstrings and GitHub Pages deploy.
- `CODE_OF_CONDUCT.md`, `SECURITY.md`, GitHub issue/PR templates.

### Changed
- Catalog default repo updated from `khursanirevo/revospeech` to the revolab org.
- VITS phonemization rewritten to be Piper-compatible (`--ipa=3`, PAD insertion).
- RevoVoice production speaker list reduced to high-quality speakers.
- HuggingFace auth errors wrapped with helpful, actionable messages.

### Fixed
- Catalog `target_manifest` None-dereference bug.
- VITS audio quality issues caused by malformed phoneme sequences.
- Branch mismatch in CI (standardized on `master`).
- `hf_user` type inconsistency in `usage.py` (now `str | None`).

### Removed
- Low-quality speakers from the RevoVoice production list.
- Dead test fixtures (`mock_recognizer`, `mock_tts_model`) from `conftest.py`.
- Stale planning artifacts (`PLAN.md`, `REPO_ANALYSIS`).

## [0.1.1] - 2025

### Added
- VITS Malay TTS engine via ONNX Runtime.
- Batch ASR with parallel workers (`asr.transcribe_batch`).
- Shared HuggingFace auth utilities.
- VITS TTS engine tests.

### Changed
- Renamed PyPI package and internal module from `revos` to `revospeech`.
- Bumped version to 0.1.1.
- Updated CI workflows and project metadata for the rename.

### Fixed
- Updated paths and tests after the rename.
- Mocked `espeak-ng` in VITS tests for CI environments.

## [0.1.0] - 2026-04-24

### Added
- ASR engine with sherpa-onnx backend (zipformer-v2 model).
- TTS engine with RevoVoice backend (revovoice model).
- Model registry with YAML manifests (bundled + user).
- Model downloader with security hardening
  (tarball filter, zip path-traversal protection, URL validation).
- CLI: `revospeech transcribe`, `synthesize`, `models`, `info`.
- CLI: `revospeech catalog list`, `catalog pull`.
- Remote model catalog fetching from GitHub repo.
- Custom exception hierarchy (`RevosError`, `RevosConfigError`,
  `RevosModelError`, `RevosEngineError`, `RevosAudioError`) with
  `suggestion` attribute for fix instructions.
- Configuration module (`revospeech/config.py`): `get_api_key()`,
  `set_api_key()`, `load_config()`, `save_config()` — API key resolution
  order: constructor arg > `REVOLAB_API_KEY` env var >
  `~/.config/revospeech/config.yaml`.
- Model status system (`ModelStatus` dataclass, `check_model()`,
  `list_model_statuses()`) — ready / needs-download / needs-api-key.
- Model discovery API: `list_models()`, `search_models(query)`,
  `check_model(name)` with fuzzy search via `difflib.SequenceMatcher`.
- CLI: `revospeech models` rewritten with rich table, status icons
  (✓/↓/✗), filter flags (`--ready`, `--task`, `--mode`, `--status`, `--json`).
- CLI: `revospeech models-info <name>` and `revospeech search <query>`.
- Manifest schema extended with 11 optional fields: `mode`, `api_endpoint`,
  `size_mb`, `capabilities`, `languages`, `tags`, `license`, `sha256`,
  `min_ram_mb`, `min_vram_mb`, `quality_notes`.
- `ModelManifest.is_local` / `ModelManifest.is_api` properties.
- Factory dispatch for API mode: `ASR()` and `TTS()` validate API key
  when `mode: api`.
- `BaseASR.stream_transcribe()` and `BaseTTS.synthesize_streaming()`
  no-op stubs (raise `NotImplementedError`).
- Catalog caching with 1-hour TTL in
  `~/.cache/revospeech/catalog_cache.json` — repo-aware.
- Catalog network calls: retry (3x) with exponential backoff and 10s timeout.
- Thread-safe model registry (`threading.Lock`).
- Thread-safe usage tracking with log rotation (10MB max).
- `__version__` reads from `importlib.metadata` with fallback.
- `__all__` export list and `__dir__` override in `revospeech/__init__.py`.
- Backward-compatibility gate tests (`tests/test_compat_gates.py`).
- Integration tests for ASR, TTS, and catalog.
- pytest markers: `slow`, `api`, `roundtrip`.
- `synthesize_long()` for automatic text splitting and audio concatenation.
- `Audio.concatenate()` and `Audio.duration`.
- Model version pinning via `revision` field in YAML manifests.
- Gated model access with clear error messages (401/403).
- Usage tracking for gated models (local JSONL log).
- Device auto-detection (GPU/CPU).
- HuggingFace authentication check.
- Pre-commit hooks (ruff lint + format).
- CI workflow with coverage reporting (70% threshold).
- MIT license.

### Changed
- Renamed model from `omnivoice` to `revovoice` (backend and model name).
- Catalog fetches from this GitHub repo instead of a separate HF repo.
- CLI `transcribe` and `synthesize` wrapped in try/except for friendly errors.
- CI: added `ruff format --check`; publish workflow unified to `uv build`.

### Security
- Zip path-traversal protection in model extraction.
- URL scheme validation (only `http`/`https`).
- Usage log file permissions set to `0o600`.
- Tarball extraction uses `filter="data"`.

[Unreleased]: https://github.com/revolab/revospeech/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/revolab/revospeech/releases/tag/v0.1.1
[0.1.0]: https://github.com/revolab/revospeech/releases/tag/v0.1.0
