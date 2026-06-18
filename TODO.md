# TODO — RevoSpeech Open Source Readiness

> **Goal**: Ship `revospeech` as revolab's open-source speech AI library.
> Users run models **locally** (weights from HuggingFace) or via **revolab cloud API** (API key).
> Architecture is extensible so the community can plug in their own providers.
> **Developer experience is the #1 priority.**

---

## Phase 0 — Architecture: Local + API + Extensible Provider System

### 0.1 Manifest Schema Extension

- [x] Add `mode` field to `ModelManifest`: `"local"` | `"api"` (default: `"local"` for backward compat)
- [x] Add `api_endpoint` field (optional, for API URL or self-hosted endpoints)
- [x] Update `manifest.py` `load_manifest()` to parse new fields with defaults
- [x] Add `is_local` and `is_api` helper properties to `ModelManifest`
- [x] Ensure existing YAML manifests work without any changes (new fields all optional with safe defaults)

### 0.2 Configuration & API Key Management

- [x] Create `revospeech/config.py` — centralized configuration
  - Resolve API key in priority order: constructor arg > env var (`REVOLAB_API_KEY`) > config file (`~/.config/revospeech/config.yaml`)
  - `get_api_key() -> str | None`
  - `set_api_key(key: str) -> None`
  - Load/save `~/.config/revospeech/config.yaml`
- [x] Validate API key presence before API calls — raise `RevosConfigError` with exact instructions
- [x] Add `revospeech config set-api-key` CLI command

### 0.3 Extensible Backend System

> **Decision:** The `revospeech/backends/` package was rejected during RALPLAN planning.
> Instead, backends are co-located with existing engines in `revospeech/asr/` and `revospeech/tts/`.
> The factory pattern already handles mode-based dispatch. Adding a new backend means:
> 1. Create `revospeech/{task}/{backend}_engine.py` implementing BaseASR/BaseTTS
> 2. Add a factory branch in `revospeech/{task}/__init__.py`
> 3. Add a YAML manifest with `mode: local` or `mode: api`
> See [AGENTS.md](AGENTS.md) for step-by-step instructions.

- [x] ~~Create `revospeech/backends/` package~~ — Rejected, use existing factory pattern
- [x] Factory dispatches by `manifest.mode` + `manifest.backend`
- [x] API mode validates API key via `get_api_key()`, raises `RevosConfigError` if missing
- [ ] Auto-discover backends via entry points (`revospeech.backends` group) — deferred to post-MVP

### 0.4 revolab Cloud API Backend

> Skeleton landed (commits f49b43e, 9febd97). Real field mappings remain BLOCKED on the API contract.

- [x] Create `revospeech/asr/revolab_engine.py` — skeleton with response parser ready for real fields
- [x] Create `revospeech/tts/revolab_engine.py` — skeleton with base64 + URL audio fetch
- [x] Handle auth, retries, rate limiting, streaming responses — `RevolabClient` in `revospeech/http_client.py` (httpx)
- [ ] Map API responses to existing `Transcript` and `Audio` result types (**BLOCKED** — waiting on API contract)
- [x] Add `httpx` as optional dependency (`revospeech[api]`)

### 0.5 ASR/TTS Factory Updates

- [x] Update `revospeech/asr/__init__.py` factory: check manifest mode → local engine or API backend
- [x] Update `revospeech/tts/__init__.py` factory: same logic
- [x] Resolve API key via `revospeech.config` before constructing API backend
- [x] User code stays identical — `ASR("model-name")` and `TTS("model-name")` work for both local and API models

---

## Phase 1 — Model Discovery & Registry UX

### 1.1 Rich Model Info in the Manifest

- [x] Add `size_mb: float` — download size
- [x] Add `capabilities: list[str]` — what the model supports (e.g., `["word-timestamps", "streaming", "voice-cloning"]`)
- [x] Add `languages: list[str]` — supported languages
- [x] Add `tags: list[str]` — for search/filter
- [x] Add `license: str` — model weight license
- [x] Add `sha256: str` — for download integrity verification
- [x] Add `min_ram_mb: int` / `min_vram_mb: int` — resource requirements
- [x] Add `quality_notes: str` — brief quality assessment
- [x] Validate manifests on registration — reject broken ones, warn on missing optional fields

### 1.2 Model Status — Know What's Usable Right Now

- [x] Add `ModelStatus` dataclass (`revospeech/registry/status.py`)
- [x] Add `revospeech.check_model(name) -> ModelStatus`
- [x] `is_installed` logic: check if all `manifest.files` exist in `~/.cache/revospeech/<name>/`
- [x] `status` logic: ready / needs-download / needs-api-key

### 1.3 Python API — List & Discover

- [x] Add `list_models()` to top-level `revospeech/__init__.py` with filter params: `task`, `mode`, `language`, `status`, `capability`
- [x] Add `search_models(query: str) -> list[ModelStatus]` — fuzzy search via `difflib.SequenceMatcher`
- [x] Add `check_model(name: str) -> ModelStatus`
- [x] Fuzzy matching: typo-tolerant model name lookup with "did you mean?" suggestions

### 1.4 CLI — `revospeech models`

- [x] Rewrite `revospeech models` with rich table: name, task, mode, status icon, size, language, capabilities
- [x] Status indicators: ✓ (ready), ↓ (needs download), ✗ (needs API key)
- [x] Add `--ready`, `--task`, `--mode`, `--status`, `--json` filter flags
- [x] Add `revospeech models-info <name>` — detailed single-model view
- [x] Add `revospeech search <query>` — fuzzy search across name, tags, description
- [x] Add `revospeech models --download <name>` — download a model without running inference
- [x] Color output when terminal supports it, plain when piped

### 1.5 Remote Catalog Improvements

- [x] Cache catalog responses locally (`~/.cache/revospeech/catalog_cache.json`, TTL: 1 hour)
- [x] Repo-aware cache (changing `REVOLAB_CATALOG_REPO` invalidates cache)
- [x] Add timeout (10s) and retry (3x exponential backoff) to catalog network calls
- [x] Show which catalog models are already installed locally
- [x] `revospeech catalog pull <name>` — progress bar with size + ETA
- [x] `revospeech catalog search <query>` — filter by language, task, size, capabilities
- [x] Model recommendations: "best for English ASR", "fastest TTS"

---

## Phase 2 — Code Quality Fixes

### 2.1 CI/CD

- [x] **Fix branch mismatch** — CI config uses `main` references but git default is `master`. Pick one and update all references.
- [x] Unify build tools — use `uv build` everywhere
- [x] Add `ruff format --check` to CI
- [x] Add coverage threshold (70% in CI)
- [x] Add `mypy` or `pyright` type check step to CI

### 2.2 Version Management

- [x] Use `importlib.metadata.version("revospeech")` as single source of truth in `__init__.py`
- [x] Remove hardcoded `"0.1.0"` from `__init__.py`
- [x] Add `__dir__` override so lazy imports show in tab-completion

### 2.3 Error Handling & Exceptions

- [x] Create custom exception hierarchy (`revospeech/exceptions.py`):
  ```
  RevosError (base)
  ├── RevosConfigError       — missing API key, bad config
  ├── RevosModelError        — model not found, download failed
  ├── RevosEngineError       — inference failure
  └── RevosAudioError        — unsupported format, corrupt file
  ```
- [x] CLI: wrap `transcribe` and `synthesize` in try/except — user-friendly errors
- [x] Every error includes `suggestion` attribute with fix instructions

### 2.4 Thread Safety

- [x] Add `threading.Lock` to `registry.py` `_models` dict
- [x] Add `threading.Lock` to `usage.py` `_callbacks` list and file writes
- [x] Add usage log rotation (10MB max)

### 2.5 Cleanup

- [x] Fix `hf_user` type inconsistency in `usage.py` (`str | None`)
- [x] Remove dead test fixtures (`mock_recognizer`, `mock_tts_model` in conftest.py)
- [x] Update catalog default repo from `khursanirevo/revospeech` to revolab org

---

## Phase 3 — Developer Experience (Top Priority)

### 3.1 Zero-Friction First Run

- [x] `ASR()` with no args should "just work" — auto-select a sensible default model
- [x] `TTS()` with no args should "just work" — same
- [x] First-run experience: detect no models → suggest `revospeech catalog list`
- [x] Add progress bars for model downloads with size and ETA
- [x] Add `revospeech setup` interactive command

### 3.2 Python API Ergonomics

- [x] `ASR()` / `TTS()` with no args: auto-select default model
- [x] `api_key` parameter on constructors
- [x] Auto-download models on first use with progress bar
- [x] Accept `Path` and `str` interchangeably for all file arguments
- [x] Accept file-like objects (BytesIO) for in-memory audio
- [x] `synthesize_long()` improvements: auto-detect long text, configurable chunk size

### 3.3 CLI Experience

- [x] `revospeech transcribe audio.wav` — zero-config, default model
- [x] Better output formatting: `--format text|json|srt|vtt`
- [x] `revospeech info` — show version, device, cache size, API key status
- [ ] Progress spinners for API calls (**BLOCKED** — waiting on API contract)
- [x] `--verbose` / `--quiet` flags

### 3.4 Batch Processing

- [x] Add `BatchResult` and `BatchReport` dataclasses
- [x] Add `asr.transcribe_batch(paths, max_workers=4, on_error="continue") -> BatchReport`
- [x] Add `tts.synthesize_batch(texts, max_workers=4, on_error="continue") -> BatchReport`
- [x] CLI: `revospeech transcribe *.wav`, `revospeech synthesize --file-list inputs.txt`
- [x] Batch report export: `report.save("report.json")`

### 3.5 Result Objects That Feel Right

- [x] Add `duration` property to `Transcript`
- [x] Add `save()` method to `Transcript` (`.srt`, `.vtt`, `.txt`, `.json`)
- [x] Add `play()` method to `Audio` (optional dependency)
- [x] Add `__repr__` to `Transcript` and `Audio`

### 3.6 Type Hints & Editor Support

- [x] Ensure all public APIs have complete type hints
- [x] `py.typed` marker exists — verify with `mypy`/`pyright`
- [x] Add `overload` signatures for common call patterns

---

## Phase 4 — Local Model Expansion

### 4.1 ASR Models

- [ ] Add Whisper-based ONNX model (multilingual, via sherpa-onnx)
- [ ] Add smaller/faster English-only ASR model
- [ ] Add streaming ASR support (real-time via sherpa-onnx online recognizer)
- [ ] Add language detection/auto-switch model

### 4.2 TTS Models

- [x] Add VITS-based model (lightweight, CPU-friendly) — `vits-ms` Malay multi-speaker via `revospeech/tts/vits_engine.py`
- [ ] Add Kokoro TTS model option (if license permits)
- [ ] Add language-specific models for major languages

---

## Phase 5 — revolab API Backend Features

> **Note:** All items in Phase 5 are (**BLOCKED** — waiting on API contract)
> until the revolab cloud API spec is finalized.

### 5.1 Core API Client

> Skeleton landed in `revospeech/http_client.py`. Remaining items (streaming TTS,
> file upload) need concrete endpoint shapes from the API contract.

- [x] HTTP client with retries, timeout, connection pooling — `RevolabClient` (httpx)
- [x] Request/response logging (debug level, API key masked) — `_mask_key()` + `logger.debug`
- [x] Error mapping: HTTP 401 → `RevosConfigError`, HTTP 429 → rate limit, HTTP 5xx → retry
- [ ] Streaming support for TTS (**BLOCKED** — waiting on API contract)
- [ ] File upload for ASR (large audio files) (**BLOCKED** — waiting on API contract)

### 5.2 API Feature Parity with Local

- [ ] Word-level timestamps
- [x] Voice cloning via `ref_audio` (local: `RevoVoiceTTS.synthesize(ref_audio=...)`)
- [ ] `synthesize_long()` for API
- [x] Voice listing: `TTS.list_voices()` (local: `BaseTTS.list_voices()` + VITS override)
- [ ] Speed/pitch controls

### 5.3 API Response → Same Result Types

- [ ] API ASR → `Transcript` with `Segment` objects
- [ ] API TTS → `Audio` with `samples`, `sample_rate`, `save()`
- [x] User code identical whether local or API (factory dispatch by `manifest.mode`)

---

## Phase 6 — Testing & Reliability

### 6.1 Cover Current Gaps

- [x] Corrupt/invalid audio input
- [x] Network errors in downloader
- [x] Thread safety tests for registry and usage modules
- [x] Base classes (`BaseASR`, `BaseTTS`) directly
- [x] `revospeech.__init__` top-level imports and version
- [x] Non-WAV audio formats
- [x] Empty text, very long text, CJK text

### 6.2 API Backend Tests

- [x] Mock revolab API responses (skeleton — `tests/test_api_backend.py`, real fields need contract)
- [x] Test API key resolution order (constructor > env var > config)
- [x] Test missing API key → clear error message
- [x] Test rate limiting / retry behavior (skeleton — `RevolabClient` retry loop tested)
- [ ] Test malformed API responses (**BLOCKED** — waiting on API contract)
- [ ] Test streaming TTS response handling (**BLOCKED** — waiting on API contract)

### 6.3 Discovery & Registry Tests

- [x] `list_models()` with all filter combinations
- [x] `search_models()` fuzzy matching — typos, partial names, tags
- [x] `check_model()` for each status
- [x] Manifest validation — missing required fields, invalid types

### 6.4 Integration Tests

- [x] Full flow: `catalog pull → local ASR/TTS → result`
- [ ] Full flow: `API key → API ASR/TTS → result` (**BLOCKED** — waiting on API contract)
- [ ] Hybrid: local ASR → API TTS (**BLOCKED** — waiting on API contract)

### 6.5 Round-Trip Tests (TTS → ASR Self-Validation)

- [x] Add `tests/test_roundtrip.py` with round-trip tests
- [x] `fuzzy_match(actual, expected, threshold=0.8)` helper
- [x] Test cases: simple English, longer text, multiple languages, voice cloning
- [ ] Local round-trip, API round-trip, hybrid round-trip (**BLOCKED** — waiting on API contract)
- [x] Add `pytest.mark.roundtrip` marker
- [x] Round-trip tests as quality benchmarks over time

### 6.6 Extension Protocol Tests

- [ ] Register a custom backend via entry point (**BLOCKED** — entry-point auto-discovery deferred to post-MVP)
- [x] Subclass `BaseASR` → verify factory dispatches
- [x] Subclass `BaseTTS` → verify factory dispatches
- [x] Invalid backend (missing abstract methods) → clear error

---

## Phase 7 — Documentation & Open Source Essentials

### 7.1 Documentation

**Layer 1 — Copy-paste examples (80% of users)**

- [x] `examples/01_quickstart.py` — 5 lines, `ASR()` + `TTS()` with defaults
- [x] `examples/02_pick_model.py` — `list_models()`, pass model name
- [x] `examples/03_transcribe_options.py` — `language=`, `word_timestamps=` kwargs
- [x] `examples/04_voice_cloning.py` — `ref_audio=`, `ref_text=` kwargs
- [x] `examples/05_long_text.py` — `synthesize_long()` for paragraphs
- [x] `examples/06_batch_processing.py` — batch methods
- [x] `examples/07_api_backend.py` — switch to revolab API with `api_key=`
- [x] `examples/08_roundtrip_test.py` — TTS → ASR → assert

**Layer 2 — Docstrings (15% of users)**

- [x] Every public method docstring starts with runnable example — factory funcs + BaseASR/BaseTTS methods + Transcript/Audio/BatchReport.save
- [x] Docstring examples tested with `pytest-doctest` or `xdoctest` — `tests/test_docstring_examples.py` + xdoctest dev dep

**Layer 3 — Full API reference (5% of users)**

- [x] Set up mkdocs-material + mkdocstrings
- [x] One page per class — split into `docs/api/{asr,tts,registry,cli,results}.md`
- [x] Cross-linked "See also" references — present on all api/* pages + top-level docs pages

**CLI discoverability**

- [x] `revospeech --help` shows all commands
- [x] Every CLI error suggests the fix — `_format_error()` surfaces `.suggestion`

**Docs Pages**

- [x] Quickstart — 5 lines, working in under 60 seconds
- [x] Local vs API guide — README + `docs/index.md`
- [x] Model catalog with benchmarks — `docs/models.md` catalogs current models + roadmap; benchmarks blocked on harness
- [x] Configuration guide — `docs/configuration.md`
- [x] Extension guide — `docs/extension.md`
- [x] CLI reference — `docs/cli.md` + `docs/api/cli.md`
- [x] Troubleshooting — `docs/troubleshooting.md`

### 7.2 Examples (Numbered, Progressive)

- [x] `examples/09_custom_backend.py`
- [x] `examples/10_streaming_tts.py`
- [x] `examples/11_batch_directory.py`
- [x] `examples/12_cli_only.sh`

### 7.3 Community Files

- [x] `CODE_OF_CONDUCT.md`
- [x] `SECURITY.md`
- [x] `.github/ISSUE_TEMPLATE/`
- [x] `.github/PULL_REQUEST_TEMPLATE.md`

### 7.4 README

- [x] Badges: Python versions, license, CI
- [x] 10-second quickstart at top — `## Quickstart` at README L9
- [x] "Local vs API" two-column comparison — `## Local vs Cloud API` at L298
- [x] Model table with download size, language, quality notes — L249, with size + quality notes
- [x] "Extensible" section with custom backend example — `## Adding Custom Models` at L382

---

## Phase 8 — Release

- [ ] Finalize package name and org namespace (**BLOCKED** — pending org migration decision)
- [ ] Update all references from `khursanirevo` to revolab org (**BLOCKED** — pending GitHub org creation)
- [x] Clean `pyproject.toml` metadata for PyPI — Windows classifier, tqdm core, playback extra
- [x] Test install from clean venv: `pip install revospeech[all]` — verified via `uv build` + clean venv install
- [x] Verify CLI entry point works post-install — `revospeech --version` returns `0.1.1`
- [ ] Publish to TestPyPI → verify → publish to PyPI (**BLOCKED** — pending release window)
- [x] Set up docs site (GitHub Pages or Read the Docs) — `.github/workflows/docs.yml` builds + deploys on master push

---

## Priority & Timeline

| Priority | What | Why | Effort |
|---|---|---|---|
| **P0** | Phase 0 (architecture) + Phase 1 (discovery) + Phase 2 (quality) | Core foundation | 4–6 weeks |
| **P1** | Phase 3 (DX) | Makes users love it | 2–3 weeks |
| **P2** | Phase 4 (more models) + Phase 5 (API backend) | Makes library useful | 3–4 weeks |
| **P3** | Phase 6 (testing) + Phase 7 (docs) | Completes the package | 2–3 weeks |
| **P4** | Phase 8 (release) | Ship it | 1 week |
| | **Total to v1.0** | | **12–17 weeks** |

### What "Done" Looks Like for v1.0

```python
# A new user installs and is productive in under 2 minutes:
pip install revospeech
python -c "from revospeech import TTS; TTS().synthesize('Hello world').save('out.wav')"

# They can always see what's available:
revospeech models
revospeech search "english fast"
revospeech models-info zipformer-v2

# That's it. The library handles model selection, download, and inference.
# Power users can: pick specific models, use revolab API, clone voices, extend backends.
```
