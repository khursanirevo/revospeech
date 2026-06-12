# PLAN — revos MVP Foundation (Phases 0-2)
# Status: CONSENSUS (Architect + Critic approved with conditions)

> RALPLAN consensus planning output. Architect and Critic reviews completed.
> See REPO_ANALYSIS/ for full codebase context. See TODO.md for complete backlog.

---

## RALPLAN-DR Summary

### Principles
1. **Single abstraction layer** — one way to add engines (extend existing ABCs, no parallel hierarchy)
2. **Backward compatibility** — existing code and YAML manifests keep working unchanged
3. **Progressive complexity** — simple things simple, complex things possible
4. **Production quality** — proper errors, thread safety, CI before open source

### Decision Drivers
1. Contributor clarity — one obvious way to extend the library
2. Minimal disruption — extend existing patterns, don't introduce new ones
3. Ship fast — defer complexity (keyring, entry_points) until proven demand

### Key Decisions

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Backend architecture | Co-locate API engines with local engines | Separate `revos/backends/` package | Avoids two-tier ABC confusion (Architect) |
| Extension mechanism | Convention-based (defer entry_points) | setuptools entry_points now | Premature for 2 backends, no community yet |
| HTTP client for API | `httpx` (optional dep, `revos[api]`) | `httpx` as core dep | Follows existing `omnivoice` optional pattern (Critic) |
| HTTP client for catalog/downloader | `urllib` + small retry/timeout helper | Migrate to `httpx` | Keeps core light, avoids dep bloat for local-only users (Critic) |
| Registry thread safety | Add `threading.Lock` to existing dict | Refactor to `Registry` class | Overengineering for 1.7K lines with zero concurrent usage (Critic) |
| Config key storage | env var + config.yaml | + keyring | Defer to post-MVP (both reviewers) |
| API key resolution | constructor > env var > config.yaml | + keyring | Three layers is enough for MVP |
| Branch name | `main` (match CI config) | master | CI already uses `main`, rename git default branch |
| revolab API contract | **Deferred** — build extension skeleton, not specific engines | Ship revolab engines now | User decision: prioritize extensibility/contribution DX over any specific API. Community PRs add backends and models. |

### Core Design Principle (User-confirmed)
> **Extensibility over specific implementations.** The hard problem is making it easy for users to extend and contribute. Community can PR new backends (API providers) and new HF models. If useful, revolab merges. Ship the contribution framework, not the engines.

### What changes from the plan
- Phase 0.5 (revolab API engines) becomes **Phase 0.5 (extension skeleton)** — ship a well-documented example engine + contribution guide instead of revolab-specific engines
- The `CONTRIBUTING.md` extension guide is promoted to Phase 0 (was Phase 7)
- A `contrib/` or `examples/` directory with a working example backend that contributors can copy

---

## Phase 0: Architecture

### 0.1 Manifest Extension
**Files:** `revos/registry/manifest.py`, `revos/models/asr/zipformer_v2.yaml`, `revos/models/tts/revovoice.yaml`

- Add optional fields to `ModelManifest` (all with safe defaults):
  - `mode: str = "local"` — `"local"` | `"api"`
  - `api_endpoint: str = ""`
  - `size_mb: float = 0.0`
  - `capabilities: list[str]` = field(default_factory=list)
  - `languages: list[str]` = field(default_factory=list)
  - `tags: list[str]` = field(default_factory=list)
  - `license: str = ""`
  - `sha256: str = ""`
  - `min_ram_mb: int = 0`
  - `min_vram_mb: int = 0`
- Update `load_manifest()` to parse new fields with `.get()` defaults
- Add `is_local` / `is_api` properties
- Update existing YAML manifests with new fields (enrichment, not required)
- **Acceptance:** existing YAML loads without error, all new fields get defaults

### 0.2 Config Module
**Files:** NEW `revos/config.py`, `revos/cli/main.py`

- `get_api_key() -> str | None` — resolution: constructor arg > `REVOLAB_API_KEY` env var > `~/.config/revos/config.yaml`
- `set_api_key(key) -> None` — writes to config.yaml (create if missing, `0o600` perms)
- `load_config() -> dict` / `save_config(data) -> None`
- CLI: `revos config set-api-key`, `revos config show`
- **Acceptance:** `get_api_key()` returns None when nothing set, returns key when env var set, returns key when config.yaml set

### 0.3 Exception Hierarchy
**Files:** NEW `revos/exceptions.py`

```
RevosError (base)
├── RevosConfigError       — missing API key, bad config
├── RevosModelError        — model not found, download failed
├── RevosEngineError       — inference failure
└── RevosAudioError        — unsupported format, corrupt file
```
- Every error includes `suggestion` attribute with fix instructions
- **Acceptance:** all public APIs raise RevosError subclasses, not bare ValueError/KeyError

### 0.4 Engine Base Updates
**Files:** `revos/asr/base.py`, `revos/tts/base.py`

- Add **concrete no-op methods** (NOT abstract) for API capabilities:
  - `BaseASR.stream_transcribe()` → raises NotImplementedError with "this engine does not support streaming"
  - `BaseTTS.synthesize_streaming()` → same pattern
- Existing engines (SherpaOnnxASR, RevoVoiceTTS) are unaffected — they inherit no-ops
- **Acceptance:** existing engines pass all current tests without modification

### 0.5 revolab API Engines
**Files:** NEW `revos/asr/revolab_engine.py`, NEW `revos/tts/revolab_engine.py`

- `RevolabASR(BaseASR)` — HTTP client calling revolab API for transcription
- `RevolabTTS(BaseTTS)` — HTTP client calling revolab API for synthesis
- Use `httpx` (optional, imported only in these engines)
- Retry (3x exponential backoff), timeout (30s), rate limit handling
- Map API responses to `Transcript` / `Audio` — **same result types as local**
- Resolve API key via config module, raise `RevosConfigError` with fix if missing
- **BLOCKED BY:** revolab API contract definition
- **Acceptance:** mock API server → engine → correct Transcript/Audio result

### 0.6 Factory Updates
**Files:** `revos/asr/__init__.py`, `revos/tts/__init__.py`

- Extend dispatch: `if manifest.mode == "api":` → load revolab_engine
- **Preserve lazy-import pattern** — import engines inside factory function, not at module level
- If `mode == "api"` and no API key: raise `RevosConfigError` with setup instructions
- **Acceptance:** `ASR("zipformer-v2")` still works (local), `ASR("revolab-whisper")` dispatches to API

---

## Phase 1: Model Discovery

### 1.1 Model Status
**Files:** NEW `revos/registry/status.py`

```python
@dataclass
class ModelStatus:
    name: str
    task: str
    mode: str             # "local" | "api"
    status: str           # "ready" | "needs-download" | "needs-api-key"
    installed: bool
    size_mb: float | None
    capabilities: list[str]
    languages: list[str]
```
- `check_model(name) -> ModelStatus` — checks cache dir for local, config for API
- `is_installed` logic: all `manifest.files` exist in `~/.cache/revos/<name>/`
- **Acceptance:** `check_model("zipformer-v2")` returns status="ready" when cached, "needs-download" when not

### 1.2 List & Search API
**Files:** `revos/registry/status.py`, `revos/__init__.py`

- `list_models(task=None, mode=None, language=None, status=None, capability=None) -> list[ModelStatus]`
- `search_models(query) -> list[ModelStatus]` — fuzzy match via `difflib.SequenceMatcher` (stdlib, no new dep)
- `check_model(name) -> ModelStatus`
- Expose via top-level `revos.__init__`
- **Acceptance:** `list_models(mode="local", status="ready")` returns only usable local models

### 1.3 CLI Discovery
**Files:** `revos/cli/main.py`

- Rewrite `revos models` with rich table: name, task, mode, status icon, size, language, capabilities
- Add `revos models info <name>` — detailed single-model view
- Add `revos search <query>` — fuzzy search
- Filter flags: `--ready`, `--task`, `--mode`, `--language`, `--json`
- Status icons: ✓ ready, ↓ needs download, ✗ needs API key
- **Acceptance:** `revos models` shows all registered models with correct status

### 1.4 Catalog Fixes
**Files:** `revos/catalog.py`

- Add timeout (10s) to `urllib.request.urlopen` calls
- Add retry helper: `_retry_urlopen(url, retries=3, backoff=1.0)` wrapping urllib
- Cache catalog responses in `~/.cache/revos/catalog_cache.json` (TTL: 1h)
- Cross-reference catalog with local registry to show installed status
- **Acceptance:** `list_catalog()` works behind slow network, returns cached results within TTL

---

## Phase 2: Code Quality

### 2.1 Thread Safety
**Files:** `revos/registry/registry.py`, `revos/usage.py`

- Add `threading.Lock` to `registry.py`: wrap `_models` dict access
- Add `threading.Lock` to `usage.py`: wrap `_callbacks` list and JSONL file writes
- Minimal change — no class refactor, just lock acquisition around mutations
- **Acceptance:** concurrent `register()` + `get()` calls don't raise

### 2.2 Version Management
**Files:** `revos/__init__.py`

- Reuse `_get_version()` pattern from `cli/main.py` lines 230-237 (already uses `importlib.metadata`)
- Remove hardcoded `"0.1.0"` from `__init__.py`
- `__version__` = `importlib.metadata.version("revos")` with `try/except` fallback to `"0.0.0-dev"`
- **Acceptance:** `revos.__version__` matches `pyproject.toml` version after `pip install`

### 2.3 CI/CD
**Files:** `.github/workflows/ci.yml`, `.github/workflows/publish.yml`

- Fix branch mismatch — **pick `main`** (modern convention, matches CI config), update git default branch
- Unify build tools: use `uv build` in publish.yml (currently `python -m build`)
- Add `ruff format --check` step after `ruff check`
- Add coverage threshold: `--cov-fail-under=80`
- Add `mypy` step: `uv run mypy revos/ --ignore-missing-imports`
- **Acceptance:** CI passes on `main` branch with all checks green

### 2.4 CLI Error Handling
**Files:** `revos/cli/main.py`

- Wrap `transcribe` and `synthesize` command bodies in try/except
- Map: `RevosConfigError` → "API key not set. Run: revos config set-api-key"
- Map: `RevosModelError` → "Model not found. Run: revos models"
- Map: `RevosEngineError` → "Inference failed: {details}"
- Map: `RevosAudioError` → "Audio error: {details}"
- All errors printed to stderr, exit code 1
- **Acceptance:** `revos transcribe nonexistent.wav` shows friendly error, not Python traceback

### 2.5 Cleanup
**Files:** `tests/conftest.py`, `revos/usage.py`, various

- Remove dead fixtures: `mock_recognizer`, `mock_tts_model` in conftest.py
- Fix `hf_user` type inconsistency in `usage.py`
- Add usage log rotation: max 10MB, rotate to `usage.jsonl.1`
- Update catalog default repo from `khursanirevo/revos` to revolab org
- **Acceptance:** `ruff check` + `ruff format --check` pass clean

---

## Backward Compatibility Gates

These tests must pass AFTER every phase change. Add them FIRST.

- [ ] Test: load `revos/models/asr/zipformer_v2.yaml` → all fields resolve, no KeyError
- [ ] Test: load `revos/models/tts/revovoice.yaml` → all fields resolve, no KeyError
- [ ] Test: `from revos.asr import ASR` → works without error (import chain)
- [ ] Test: `ASR("zipformer-v2")` → factory returns SherpaOnnxASR (no regression)
- [ ] Test: `from revos.tts import TTS` → works without error
- [ ] Test: `TTS("revovoice")` → factory returns RevoVoiceTTS (no regression)
- [ ] Test: `revos models` CLI → shows zipformer-v2 and revovoice
- [ ] Test: `revos transcribe --help` → works without import errors

---

## Execution Order

### Sprint 1 (Week 1-2): Foundation — no user-facing changes yet

| Order | Task | Dependencies | Parallelizable |
|---|---|---|---|
| 1 | Add backward-compat gate tests | None | — |
| 2 | 0.3 Exception hierarchy | None | Yes |
| 3 | 0.1 Manifest extension | None | Yes |
| 4 | 2.2 Version fix | None | Yes |
| 5 | 2.5 Cleanup | None | Yes |
| 6 | 0.2 Config module | 0.3 (uses exceptions) | Partially |

### Sprint 2 (Week 3-4): Architecture + Discovery

| Order | Task | Dependencies | Parallelizable |
|---|---|---|---|
| 7 | 0.4 Engine base updates | 0.1 | No |
| 8 | 0.6 Factory updates | 0.1, 0.2, 0.4 | No |
| 9 | 1.1 Model status | 0.1 | Yes |
| 10 | 1.4 Catalog fixes | None | Yes |
| 11 | 2.1 Thread safety | None | Yes |
| 12 | 1.2 List/search API | 1.1 | No |

### Sprint 3 (Week 5-6): CLI + API Engines + CI

| Order | Task | Dependencies | Parallelizable |
|---|---|---|---|
| 13 | 0.5 revolab API engines | 0.4, 0.6, revolab API contract | **BLOCKED** |
| 14 | 1.3 CLI discovery | 1.2 | No |
| 15 | 2.4 CLI error handling | 0.3 | Yes |
| 16 | 2.3 CI/CD fixes | None | Yes |

**Note:** Sprint 3 task 13 (revolab API engines) is blocked until the API contract is defined. All other tasks can proceed independently.

---

## Pre-Mortem

| Failure Scenario | Mitigation |
|---|---|
| API engines built against wrong contract → throwaway work | Pin down API contract before implementing engines. If API doesn't exist yet, ship Phase 0 without engines as "abstraction prep" |
| httpx install fails on target platforms | Make it optional (`revos[api]`). Local-only users never need it |
| Registry changes break auto-loading → KeyError on import | Backward-compat gate tests run after every change. Refactor is minimal (add Lock only) |

---

## Files Created/Modified Summary

**New files (5):**
- `revos/config.py`
- `revos/exceptions.py`
- `revos/asr/revolab_engine.py`
- `revos/tts/revolab_engine.py`
- `revos/registry/status.py`

**Modified files (12):**
- `revos/registry/manifest.py`
- `revos/registry/registry.py`
- `revos/registry/__init__.py`
- `revos/asr/__init__.py`
- `revos/asr/base.py`
- `revos/tts/__init__.py`
- `revos/tts/base.py`
- `revos/cli/main.py`
- `revos/catalog.py`
- `revos/__init__.py`
- `revos/usage.py`
- `pyproject.toml`
