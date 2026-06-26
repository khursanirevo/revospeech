# Changelog

All notable changes to RevoSpeech are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-26

### Added
- Speech restoration util model (Sidon): denoise + dereverberation + bandwidth
  extension via a 3-stage ONNX pipeline (mel front-end -> w2v-BERT 2.0
  predictor -> DAC vocoder).
- New `util` task category in the manifest registry, parallel to `asr`/`tts`.
  `Util()` factory + `BaseUtil` ABC with `restore(audio)` interface.
- `revospeech restore -i in.wav -o out.wav` CLI subcommand for standalone use.
- `restore=` kwarg on `TTS()` (default off) and `revospeech synthesize --restore`
  opt-in flag to apply util post-processors to synthesis output.
- Mel front-end ONNX (1.1 MB) bundled with the wheel via hatch force-include.
- Cloud ASR backend wired up: `RevolabASR` engine now dispatched from the
  factory when `backend: revolab` (was a `NotImplementedError` stub).
- `revolab-asr` manifest (cloud ASR via private Qwen model).
- Size-confirmation prompt before downloads (TTY-aware; skip with
  `REVOSPEECH_YES=1`).
- Bare `revospeech` prints a setup hint; `revospeech info` suggests
  `revospeech setup` when no models are installed.
- README "First 60 seconds" quickstart + CLI command map (catalog vs models).
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
- Example scripts `01`-`11` (progressive, runnable).
- mkdocs-material documentation site with mkdocstrings and GitHub Pages deploy.
- `CODE_OF_CONDUCT.md`, GitHub issue/PR templates.

### Changed
- `TTS(restore=...)` default flipped to `False` (post-processing is opt-in).
- HuggingFace download routing: `hf_private` manifest flag or `org/repo` URL
  shorthand now route to `snapshot_download` instead of the HTTP downloader
  that rejected non-`https://` URLs. Fixes revovoice download.
- README rewritten to lead with `revospeech setup` and document the
  `catalog pull` vs `models --download` distinction.
- pyproject.toml description updated to mention speech restoration.
- Catalog default repo updated from `khursanirevo/revospeech` to the revolab org.
- VITS phonemization rewritten to be Piper-compatible (`--ipa=3`, PAD insertion).
- RevoVoice production speaker list reduced to high-quality speakers.
- HuggingFace auth errors wrapped with helpful, actionable messages.

### Fixed
- Catalog `target_manifest` None-dereference bug.
- VITS audio quality issues caused by malformed phoneme sequences.
- Branch mismatch in CI (standardized on `master`).
- `hf_user` type inconsistency in `usage.py` (now `str | None`).
- Click artifact at the end of Sidon-enhanced audio (DAC vocoder cutoff
  smoothed with a 5 ms fade).

### Removed
- `SECURITY.md` (use GitHub Issues for vulnerability reports).
- `TODO.md` (410-line internal planning doc; GitHub Issues is the venue).
- 3 orphan example scripts (`asr_transcribe.py`, `tts_synthesize.py`,
  `tts_voice_cloning.py`) that duplicated numbered examples.
- Low-quality speakers from the RevoVoice production list.
- Dead test fixtures (`mock_recognizer`, `mock_tts_model`) from `conftest.py`.

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

[Unreleased]: https://github.com/revolab/revospeech/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/revolab/revospeech/releases/tag/v0.2.0
[0.1.1]: https://github.com/revolab/revospeech/releases/tag/v0.1.1
[0.1.0]: https://github.com/revolab/revospeech/releases/tag/v0.1.0
