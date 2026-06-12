# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Custom exception hierarchy (`revospeech/exceptions.py`): `RevosError`, `RevosConfigError`, `RevosModelError`, `RevosEngineError`, `RevosAudioError` — all with `suggestion` attribute for fix instructions
- Configuration module (`revospeech/config.py`): `get_api_key()`, `set_api_key()`, `load_config()`, `save_config()` — API key resolution: constructor arg > `REVOLAB_API_KEY` env var > `~/.config/revospeech/config.yaml`
- Model status system (`revospeech/registry/status.py`): `ModelStatus` dataclass with `check_model()` and `list_model_statuses()` — shows ready/needs-download/needs-api-key per model
- Model discovery API: `revospeech.list_models()`, `revospeech.search_models(query)`, `revospeech.check_model(name)` — fuzzy search via `difflib.SequenceMatcher`
- CLI: `revospeech models` rewritten with rich table, status icons (✓/↓/✗), filter flags (`--ready`, `--task`, `--mode`, `--status`, `--json`)
- CLI: `revospeech models-info <name>` — detailed single-model view
- CLI: `revospeech search <query>` — fuzzy search across model names, tags, descriptions
- CLI: `revospeech config set-api-key` — save API key to config file
- Manifest schema extended with 11 new optional fields: `mode`, `api_endpoint`, `size_mb`, `capabilities`, `languages`, `tags`, `license`, `sha256`, `min_ram_mb`, `min_vram_mb`
- `ModelManifest.is_local` / `ModelManifest.is_api` properties
- Factory dispatch for API mode: `ASR()` and `TTS()` validate API key when `mode: api`
- `BaseASR.stream_transcribe()` no-op (raises NotImplementedError)
- `BaseTTS.synthesize_streaming()` no-op (raises NotImplementedError)
- Catalog caching with 1-hour TTL in `~/.cache/revospeech/catalog_cache.json` — repo-aware (changing `REVOLAB_CATALOG_REPO` invalidates cache)
- Catalog network calls: retry (3x) with exponential backoff and 10s timeout
- Thread-safe model registry (`threading.Lock`)
- Thread-safe usage tracking with log rotation (10MB max)
- `__version__` reads from `importlib.metadata` with fallback to `"0.0.0-dev"`
- `__all__` export list in `revospeech/__init__.py`
- Backward-compatibility gate tests (`tests/test_compat_gates.py`) — 10 tests covering manifest loading, import chain, exception hierarchy
- Integration tests for ASR, TTS, and catalog (`tests/test_integration.py`)
- pytest markers: `slow`, `api`, `roundtrip`

### Changed

- CLI `transcribe` and `synthesize` commands wrapped in try/except — friendly error messages instead of raw tracebacks
- `revospeech/catalog.py` uses `load_config()` from config module instead of inline YAML reading
- CI workflow: added `ruff format --check`, coverage threshold (70%)
- Publish workflow: unified to `uv build`

## [0.1.0] - 2026-04-24

### Added

- ASR engine with sherpa-onnx backend (zipformer-v2 model)
- TTS engine with RevoVoice backend (revovoice model)
- Model registry with YAML manifests (bundled + user)
- Model downloader with security hardening
  (tarball filter, zip path traversal protection, URL validation)
- CLI: `revospeech transcribe`, `revospeech synthesize`, `revospeech models`, `revospeech info`
- CLI: `revospeech catalog list`, `revospeech catalog pull`
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

### Changed

- Renamed model from omnivoice to revovoice (backend and model name)
- Catalog fetches from this GitHub repo instead of separate HF repo

### Security

- Zip path traversal protection in model extraction
- URL scheme validation (only http/https)
- Usage log file permissions set to 0o600
- Tarball extraction uses `filter="data"`
