# Test Suite Analysis

This document provides a thorough analysis of every test file in the `tests/` directory, covering what each file tests, every test function, fixtures, mocking patterns, parametrization, and coverage gaps.

---

## Table of Contents

1. [conftest.py -- Shared Fixtures](#1-conftestpy----shared-fixtures)
2. [test_asr.py -- ASR Engine Tests](#2-test_asrpy----asr-engine-tests)
3. [test_tts.py -- TTS Engine Tests](#3-test_ttspy----tts-engine-tests)
4. [test_tts_long.py -- Long-Form Synthesis Tests](#4-test_tts_longpy----long-form-synthesis-tests)
5. [test_audio.py -- Audio Loading Utility Tests](#5-test_audiopy----audio-loading-utility-tests)
6. [test_catalog.py -- Remote Catalog Tests](#6-test_catalogpy----remote-catalog-tests)
7. [test_catalog_cli.py -- Catalog CLI Tests](#7-test_catalog_clipy----catalog-cli-tests)
8. [test_cli.py -- CLI Command Tests](#8-test_clipy----cli-command-tests)
9. [test_device.py -- Device Detection Tests](#9-test_devicepy----device-detection-tests)
10. [test_downloader.py -- Model Downloader Tests](#10-test_downloaderpy----model-downloader-tests)
11. [test_integration.py -- Real Integration Tests](#11-test_integrationpy----real-integration-tests)
12. [test_registry.py -- Model Registry Tests](#12-test_registrypy----model-registry-tests)
13. [test_usage.py -- Usage Tracking Tests](#13-test_usagepy----usage-tracking-tests)
14. [Cross-Cutting Observations](#14-cross-cutting-observations)

---

## 1. conftest.py -- Shared Fixtures

**File**: `tests/conftest.py` (52 lines)
**Module tested**: None (provides shared pytest fixtures)
**Pattern**: Fixture definitions only, no test functions.

### Fixtures Defined

| Fixture | Scope | Description |
|---------|-------|-------------|
| `sample_wav` | function | Creates a 1-second 440Hz mono sine wave WAV at 16kHz in `tmp_path`. Returns the file path as a string. |
| `mock_recognizer` | function | Creates a `MagicMock` mimicking `sherpa_onnx.OfflineRecognizer`. The mock recognizer returns a stream whose result has text "HELLO WORLD TEST", timestamps `[0.0, 0.3, 0.6]`, language "en", and empty words list. |
| `mock_tts_model` | function | Creates a `MagicMock` mimicking an OmniVoice model. `generate()` returns a list with one 24000-sample random float32 array (representing ~1 second of 24kHz audio). |

### Usage Notes

- `sample_wav` is used by `test_asr.py` and `test_cli.py` (transcribe tests).
- `mock_recognizer` is defined but **not actually consumed** by any test file. The ASR tests in `test_asr.py` construct their own inline mocks instead of using this fixture.
- `mock_tts_model` is similarly **not consumed** by any test; the TTS tests build their own mocks via `_make_mock_omnivoice()` helpers.

---

## 2. test_asr.py -- ASR Engine Tests

**File**: `tests/test_asr.py` (130 lines)
**Module tested**: `revos.asr` (ASR engine, result types, factory function)
**Pattern**: Unit tests with heavy mocking of `sherpa_onnx`.

### Test Functions

| Function | What It Verifies |
|----------|-----------------|
| `test_segment_creation` | `Segment` dataclass stores text, confidence, start, end correctly. |
| `test_transcript_creation` | `Transcript` dataclass stores text, segments list, and language correctly. |
| `test_asr_transcribe` | Full `SherpaOnnxASR.transcribe()` pipeline with mocked sherpa_onnx. Patches `sherpa_onnx`, `ensure_model`, and `get`. Asserts the returned `Transcript` has correct text, segments (split from "HELLO WORLD" by timestamps), and language. |
| `test_asr_factory` | The `ASR()` factory function instantiates `SherpaOnnxASR` with the correct model name and device. Registers a manifest and patches the class. |
| `test_asr_unsupported_backend` | `ASR()` raises `ValueError` with message "Supported backends: sherpa-onnx" when the manifest specifies an unknown backend. |

### Fixtures Used

- `sample_wav` (from conftest) -- used in `test_asr_transcribe`
- `tmp_path` -- built-in pytest fixture for temporary directories

### Mocking Strategy

- Patches `revos.asr.sherpa_engine.sherpa_onnx` to avoid requiring the real sherpa-onnx library.
- Patches `revos.asr.sherpa_engine.ensure_model` to skip model downloads.
- Patches `revos.asr.sherpa_engine.get` to return a fake manifest.
- Manually creates mock recognizer, stream, and result objects with `MagicMock`.

### Parametrized Tests / Markers

None.

---

## 3. test_tts.py -- TTS Engine Tests

**File**: `tests/test_tts.py` (176 lines)
**Module tested**: `revos.tts` (TTS engine, result types, factory function)
**Pattern**: Unit tests with mocked `omnivoice` module.

### Test Functions

| Function | What It Verifies |
|----------|-----------------|
| `test_audio_creation` | `Audio` dataclass stores samples and sample_rate correctly. |
| `test_audio_save` | `Audio.save()` writes a valid WAV file that soundfile can read back with correct sample rate and length. |
| `test_audio_dataclass` | `Audio` samples array values are preserved exactly (via `np.testing.assert_allclose`). |
| `test_revovoice_engine_synthesize` | `RevoVoiceTTS.synthesize()` with mocked omnivoice returns an `Audio` object at 24kHz. Verifies `model.generate` is called with correct text and speed=1.0. |
| `test_revovoice_engine_save_to_file` | `RevoVoiceTTS.synthesize()` with `output_path` writes a WAV file to disk. |
| `test_revovoice_engine_gated_error` | When `OmniVoice.from_pretrained()` raises `OSError` (gated repo), `RevoVoiceTTS.__init__` wraps it in `RuntimeError` with message "HuggingFace authentication". |
| `test_tts_unsupported_backend` | `TTS()` factory raises `ValueError` with "Supported backends: revovoice" for an unknown backend. |

### Fixtures

- `clear_registry` (autouse) -- clears `_models` dict before/after each test to avoid cross-contamination.

### Helper

- `_make_mock_omnivoice()` -- creates a fake `omnivoice` module (`ModuleType`) with a mock `OmniVoice` class whose `from_pretrained()` returns a mock model that generates random audio.

### Mocking Strategy

- `sys.modules` is patched via `patch.dict` to inject the fake `omnivoice` module.
- `revos.tts.revovoice_engine._get_hf_user` is patched to return `None`.
- Registry is cleared via autouse fixture.

### Parametrized Tests / Markers

None.

---

## 4. test_tts_long.py -- Long-Form Synthesis Tests

**File**: `tests/test_tts_long.py` (269 lines)
**Module tested**: `revos.tts.base._split_text`, `revos.tts.result.Audio.concatenate`, `revos.tts.revovoice_engine.RevoVoiceTTS.synthesize_long`
**Pattern**: Unit tests split into three sections: text splitting, audio concatenation, and synthesize_long integration.

### Section 1: Text Splitting (`_split_text`)

| Function | What It Verifies |
|----------|-----------------|
| `test_split_text_short` | Short text ("Hello world.") is returned as a single-element list. |
| `test_split_text_empty` | Empty string and whitespace-only string return empty list. |
| `test_split_text_multiple_sentences` | Multi-sentence text splits into chunks; no chunk exceeds `max_chars`; reassembled text matches original. |
| `test_split_text_respects_max_chars` | Every chunk in the result stays within the `max_chars` limit. |
| `test_split_text_long_sentence_splits_at_commas` | A single long sentence exceeding `max_chars` falls back to splitting at commas. |
| `test_split_text_multilingual_punctuation` | Chinese period punctuation (。) is recognized as a sentence boundary. |
| `test_split_text_single_long_word` | A single unbreakable token longer than `max_chars` is returned as-is (fallback behavior). |

### Section 2: Audio Concatenation (`Audio.concatenate`)

| Function | What It Verifies |
|----------|-----------------|
| `test_audio_concatenate_basic` | Two 1-second audio segments concatenate with default 0.1s silence in between, producing 2.1 seconds total. |
| `test_audio_concatenate_no_silence` | With `silence_duration=0.0`, segments join directly (2 seconds total). |
| `test_audio_concatenate_single` | Single segment returns equivalent audio (1 second). |
| `test_audio_concatenate_empty_raises` | Empty list raises `ValueError` with "empty" in the message. |
| `test_audio_concatenate_mismatched_rates` | Mismatched sample rates (16kHz vs 24kHz) raises `ValueError` with "sample rate". |

### Section 3: Audio Duration Property

| Function | What It Verifies |
|----------|-----------------|
| `test_audio_duration` | `Audio.duration` returns `len(samples) / sample_rate` (5.0 seconds). |

### Section 4: synthesize_long Integration

| Function | What It Verifies |
|----------|-----------------|
| `test_synthesize_long` | `synthesize_long` splits text into multiple chunks, calls `model.generate` multiple times (>=2), and returns concatenated `Audio` at 24kHz. |
| `test_synthesize_long_saves_to_file` | `synthesize_long` with `output_path` writes the concatenated audio to a WAV file. Duration is > 0. |
| `test_synthesize_long_empty_raises` | `synthesize_long("")` raises `ValueError` with "empty". |

### Fixtures

- `clear_registry` (autouse) -- same pattern as test_tts.py.

### Mocking Strategy

- Same `_make_mock_omnivoice()` helper and `sys.modules` patching as test_tts.py.
- `model.generate.side_effect` returns different random audio arrays for each call to simulate multi-chunk synthesis.

### Parametrized Tests / Markers

None.

---

## 5. test_audio.py -- Audio Loading Utility Tests

**File**: `tests/test_audio.py` (57 lines)
**Module tested**: `revos.asr.audio.read_waveform`
**Pattern**: Unit tests using real WAV files generated in tmp_path.

### Test Functions

| Function | What It Verifies |
|----------|-----------------|
| `test_read_mono_wav` | Reading a mono 16kHz WAV returns correct sample rate, float32 dtype, and expected length (16000 samples). |
| `test_read_stereo_converts_to_mono` | A stereo WAV (left=440Hz, right=880Hz) is automatically downmixed to mono with correct length and 1D shape. |
| `test_read_resamples_to_target_sr` | A 44100Hz WAV is resampled to 16000Hz when `target_sr=16000` is passed; output length matches target rate. |

### Fixtures Used

- `tmp_path` -- built-in pytest fixture.

### Mocking Strategy

**No mocking.** Tests create real WAV files using soundfile and read them back. This is the most "real" unit test in the suite.

### Parametrized Tests / Markers

None.

---

## 6. test_catalog.py -- Remote Catalog Tests

**File**: `tests/test_catalog.py` (217 lines)
**Module tested**: `revos.catalog` (`get_catalog_repo`, `list_catalog`, `pull_model`, `_list_yaml_files`, `_download_raw`)
**Pattern**: Unit tests with mocked GitHub API responses.

### Test Functions

| Function | What It Verifies |
|----------|-----------------|
| `test_get_catalog_repo_default` | When no env var and no config file, `get_catalog_repo()` returns `DEFAULT_CATALOG_REPO`. Patches `Path.home()` to simulate missing config. |
| `test_get_catalog_repo_from_env` | `REVOS_CATALOG_REPO` env var overrides the default. |
| `test_get_catalog_repo_env_beats_config` | Env var takes precedence over config file (only env var is set in this test). |
| `test_list_catalog_github_error` | When `_list_yaml_files` raises an exception, `list_catalog()` raises `RuntimeError` with "Cannot fetch catalog". |
| `test_list_catalog_fetches_manifests` | `list_catalog()` fetches YAML files from GitHub, parses them into `ModelManifest` objects. Verifies two manifests (TTS and ASR) are returned with correct names. |
| `test_list_catalog_filters_by_task` | `list_catalog(task="tts")` returns only TTS models. |
| `test_pull_model_installs_locally` | `pull_model("revovoice")` downloads manifest YAML and installs it to `_USER_MODELS_DIR/tts/revovoice.yaml`. Verifies file exists on disk. |
| `test_pull_model_not_found` | `pull_model("nonexistent")` raises `KeyError` with "not found in catalog". |
| `test_list_yaml_files` | `_list_yaml_files()` correctly parses GitHub API JSON response (directory listing + file listing). Returns list of YAML file paths. |
| `test_download_raw` | `_download_raw()` fetches raw file content from GitHub via `urllib.request.urlopen`. Returns the decoded string content. |

### Fixtures

- `clear_registry` (autouse) -- clears `_models` before/after each test.

### Mocking Strategy

- Patches `urllib.request.urlopen` for low-level HTTP tests.
- Patches `revos.catalog._list_yaml_files` and `revos.catalog._download_raw` for higher-level catalog tests.
- Uses `patch.dict(os.environ, ...)` for environment variable tests.
- Patches `revos.catalog._USER_MODELS_DIR` to redirect file writes to `tmp_path`.

### Parametrized Tests / Markers

None.

---

## 7. test_catalog_cli.py -- Catalog CLI Tests

**File**: `tests/test_catalog_cli.py` (139 lines)
**Module tested**: `revos.cli.main.cli` (catalog subcommands: `list`, `pull`, `info`)
**Pattern**: CLI integration tests using Click's `CliRunner`.

### Test Functions

| Function | What It Verifies |
|----------|-----------------|
| `test_catalog_list_command` | `revos catalog list` exits 0, shows model names, and prints "Fetching catalog". Mocks `_download_raw` and `_list_yaml_files`. |
| `test_catalog_list_error` | `revos catalog list` exits 1 with "Error" in output when `_list_yaml_files` raises an exception. |
| `test_catalog_list_empty` | `revos catalog list` shows "No models found" when catalog is empty. |
| `test_catalog_list_filter_by_task` | `revos catalog list -t tts` filters output to only TTS models. |
| `test_catalog_pull_command` | `revos catalog pull revovoice` installs the model manifest locally. Verifies "Installed" in output. |
| `test_catalog_pull_not_found` | `revos catalog pull nonexistent` exits 1 with "Error" in output. |
| `test_info_shows_catalog_repo` | `revos info` shows "Catalog repo" in its output. |

### Fixtures

- `clear_registry` (autouse) -- clears `_models` before/after.
- `runner` -- returns a `CliRunner` instance.

### Mocking Strategy

- Same catalog function patches as `test_catalog.py`.
- Patches `_USER_MODELS_DIR` for pull tests.
- Uses `CliRunner.invoke()` for command execution.

### Parametrized Tests / Markers

None.

---

## 8. test_cli.py -- CLI Command Tests

**File**: `tests/test_cli.py` (198 lines)
**Module tested**: `revos.cli.main.cli` (top-level commands: `transcribe`, `synthesize`, `models`, `info`)
**Pattern**: CLI integration tests using Click's `CliRunner` with mocked engines.

### Test Functions

| Function | What It Verifies |
|----------|-----------------|
| `test_cli_help` | `revos --help` exits 0 and shows "transcribe" and "synthesize" in output. |
| `test_cli_version` | `revos --version` exits 0 and shows "0.1.0". |
| `test_transcribe_help` | `revos transcribe --help` shows `--model`, `--json`, `--srt` options. |
| `test_synthesize_help` | `revos synthesize --help` shows `--model`, `--text`, `--output`, `--ref-audio` options. |
| `test_transcribe_text_output` | `revos transcribe -m test <wav>` with mocked ASR prints "HELLO WORLD" to stdout. |
| `test_transcribe_json_output` | `revos transcribe -m test --json <wav>` produces valid JSON with `text`, `segments`, and correct structure. |
| `test_transcribe_srt_output` | `revos transcribe -m test --srt <wav>` produces SRT format (contains "-->" timestamp separator). |
| `test_synthesize_requires_text_or_file` | `revos synthesize` without `--text` or `--file` exits non-zero (validation error). |
| `test_synthesize_text_output` | `revos synthesize -m test -t Hello -o out.wav` with mocked TTS exits 0 and prints "Saved". |
| `test_synthesize_from_file` | `revos synthesize -m test -f input.txt -o out.wav` reads text from file and produces output. |
| `test_models_command` | `revos models` lists registered model names and tasks. |
| `test_models_no_models` | `revos models` shows "No models found" when registry is empty. |
| `test_info_command` | `revos info` shows Python version and "Cache dir". |

### Fixtures Used

- `runner` -- returns a `CliRunner` instance.
- `sample_wav` (from conftest) -- used by transcribe tests.

### Mocking Strategy

- Patches `revos.asr.ASR` and `revos.tts.TTS` factory functions to avoid loading real models.
- Mock instances return pre-built `Transcript` and `Audio` objects.
- Uses `runner.isolated_filesystem()` for synthesize file output tests.

### Parametrized Tests / Markers

None.

---

## 9. test_device.py -- Device Detection Tests

**File**: `tests/test_device.py` (25 lines)
**Module tested**: `revos.device.auto_detect_device`
**Pattern**: Unit tests with mocked `onnxruntime`.

### Test Functions

| Function | What It Verifies |
|----------|-----------------|
| `test_detect_cpu_when_no_cuda` | When `onnxruntime` only reports `CPUExecutionProvider`, returns "cpu". |
| `test_detect_cuda_when_available` | When `onnxruntime` reports both `CUDAExecutionProvider` and `CPUExecutionProvider`, returns "cuda". |
| `test_detect_cpu_on_import_error` | When `onnxruntime` cannot be imported, falls back to "cpu". |

### Fixtures Used

None.

### Mocking Strategy

- Patches `onnxruntime.get_available_providers` return value.
- Patches `builtins.__import__` to simulate `ImportError` for onnxruntime.

### Parametrized Tests / Markers

None.

---

## 10. test_downloader.py -- Model Downloader Tests

**File**: `tests/test_downloader.py` (205 lines)
**Module tested**: `revos.registry.downloader` (`ensure_model`, `_download`, `_extract`, `_find_model_dir`)
**Pattern**: Unit tests using real archive files (tar.bz2, zip) created in tmp_path.

### Test Functions

| Function | What It Verifies |
|----------|-----------------|
| `test_extract_tar_bz2` | `_extract` correctly extracts a tar.bz2 archive, producing the expected files in the destination directory. |
| `test_extract_zip` | `_extract` correctly extracts a zip archive. |
| `test_extract_zip_rejects_path_traversal` | `_extract` raises `ValueError` with "Unsafe path" when a zip contains `../../../etc/passwd` (path traversal attack). |
| `test_extract_single_file` | `_extract` handles non-archive files by copying them directly to the destination. |
| `test_find_model_dir_direct` | `_find_model_dir` returns the directory itself when model files are directly present. |
| `test_find_model_dir_subdir` | `_find_model_dir` finds model files in a subdirectory and returns that subdirectory path. |
| `test_ensure_model_cached` | `ensure_model` skips download when model files already exist in the cache directory. Verifies `_download` is not called. |
| `test_ensure_model_no_url` | `ensure_model` raises `ValueError` with "no download URL" when manifest has empty `model_url`. |
| `test_ensure_model_downloads_and_extracts` | Full flow: `ensure_model` calls `_download`, extracts the archive, and returns the directory with model files. |
| `test_download_creates_parent_dirs` | `_download` creates parent directories for the destination path. |
| `test_download_rejects_file_scheme` | `_download` raises `ValueError` with "Invalid model URL scheme" for `file://` URLs. |
| `test_download_rejects_ftp_scheme` | `_download` raises `ValueError` with "Invalid model URL scheme" for `ftp://` URLs. |

### Helper

- `_make_manifest(**overrides)` -- creates a `ModelManifest` with sensible defaults for ASR tests.

### Fixtures Used

- `tmp_path` -- built-in pytest fixture.

### Mocking Strategy

- Patches `revos.registry.downloader._download` to skip network calls.
- Patches `revos.registry.downloader.CACHE_DIR` to redirect to `tmp_path`.
- Patches `urllib.request.urlretrieve` for `_download` tests.
- **Notably, archive creation and extraction use real tarfile/zipfile operations** -- no mocking of the extraction logic itself.

### Parametrized Tests / Markers

None.

---

## 11. test_integration.py -- Real Integration Tests

**File**: `tests/test_integration.py` (295 lines)
**Module tested**: End-to-end ASR, TTS, and catalog functionality using real models and network access.
**Pattern**: True integration tests marked `@pytest.mark.slow`. Requires network access, real model downloads, and HuggingFace authentication for TTS.

### Module-Level Marker

```python
pytestmark = pytest.mark.slow
```

All tests in this file are marked `slow` and can be excluded via `pytest -m "not slow"`.

### Test Classes

#### TestASRIntegration (5 tests)

| Function | What It Verifies |
|----------|-----------------|
| `test_model_downloads_and_loads` | `ASR("zipformer-v2")` downloads the model and initializes without errors. Asserts `model_name` attribute. |
| `test_transcribe_sine_wave` | Full pipeline: create a 2-second 440Hz sine WAV, transcribe it with `zipformer-v2`, and verify the `Transcript` result structure (text is str, language is str, segments list with valid start/end/text). |
| `test_transcribe_to_json_via_cli` | CLI subprocess `revos transcribe -m zipformer-v2 --json <wav>` produces valid JSON with `text`, `segments`, `language` keys. Uses `subprocess.run` with 120s timeout. |
| `test_transcribe_to_srt_via_cli` | CLI subprocess `revos transcribe -m zipformer-v2 --srt <wav>` produces SRT output containing "-->". |
| `test_model_cached_on_second_load` | Second instantiation of `ASR("zipformer-v2")` uses the cache (no re-download). Results from two transcriptions are consistent (same text). |

#### TestTTSIntegration (7 tests)

| Function | What It Verifies |
|----------|-----------------|
| `test_model_loads` | `TTS("revovoice")` initializes without errors. |
| `test_synthesize_basic` | `tts.synthesize("Hello, world!")` returns `Audio` at 24kHz with non-empty samples, positive duration, and float32 dtype. |
| `test_synthesize_saves_to_file` | `tts.synthesize(..., output_path=...)` writes a valid WAV file readable by soundfile at 24kHz. |
| `test_synthesize_long_text` | `tts.synthesize_long(multi-sentence text)` splits, synthesizes, and concatenates. Duration > 1.0 second. |
| `test_synthesize_with_speed` | `tts.synthesize(text, speed=1.5)` produces shorter audio than `speed=1.0`. |
| `test_cli_synthesize` | CLI subprocess `revos synthesize -m revovoice -t "Hello from the CLI." -o out.wav` produces a valid WAV file. |
| `test_cli_synthesize_long_text` | CLI subprocess with long text auto-detects and uses `synthesize_long`. Output WAV is > 2 seconds. |

#### TestCatalogIntegration (3 tests)

| Function | What It Verifies |
|----------|-----------------|
| `test_catalog_list_real` | `list_catalog()` hits the real GitHub API and returns at least 2 models including "zipformer-v2" and "revovoice". |
| `test_catalog_list_filter_asr` | `list_catalog(task="asr")` returns only ASR models. |
| `test_catalog_pull_real` | `pull_model("zipformer-v2")` downloads and installs the manifest YAML to the user models directory. Verifies file exists. |

### Fixtures

- `check_hf_auth` (autouse, in `TestTTSIntegration`) -- attempts `HfApi().whoami()` and calls `pytest.skip()` if not authenticated.
- `tmp_path` -- for file operations.

### Mocking Strategy

**No mocking.** These are true end-to-end tests that download real models, run real inference, and hit real APIs. The only isolation is the `tmp_path` for file output and the `check_hf_auth` gate for TTS tests.

### Helper

- `_make_wav(path, duration, sr)` -- generates a test WAV file with a 440Hz sine wave.

### Parametrized Tests / Markers

- Module-level `pytestmark = pytest.mark.slow` applied to all tests.
- No individual parametrization.

---

## 12. test_registry.py -- Model Registry Tests

**File**: `tests/test_registry.py` (119 lines)
**Module tested**: `revos.registry.registry` (`register`, `get`, `list_models`) and `revos.registry.manifest` (`ModelManifest`, `load_manifest`)
**Pattern**: Pure unit tests with no mocking.

### Test Functions

| Function | What It Verifies |
|----------|-----------------|
| `test_register_and_get` | `register(m)` followed by `get(name, task)` returns the same manifest object (`is` identity). |
| `test_get_missing_raises` | `get("nonexistent", "asr")` raises `KeyError` with "not found". |
| `test_list_models_all` | `list_models()` returns all registered models; `list_models("asr")` and `list_models("tts")` filter by task correctly. |
| `test_load_manifest` | `load_manifest(yaml_file)` reads a YAML file and returns a `ModelManifest` with correct name, task, backend, sample_rate, and files dict. |
| `test_register_overwrites` | Registering a model with the same name overwrites the previous entry (verified by checking description field changes). |

### Fixtures

- `clear_registry` (autouse) -- clears `_models` before/after each test.

### Mocking Strategy

**No mocking.** Tests directly manipulate the global `_models` dict and use real YAML file I/O.

### Parametrized Tests / Markers

None.

---

## 13. test_usage.py -- Usage Tracking Tests

**File**: `tests/test_usage.py` (148 lines)
**Module tested**: `revos.usage` (`track_usage`, `register_callback`, `get_usage_log`)
**Pattern**: Unit tests with mocked log file paths.

### Test Functions

| Function | What It Verifies |
|----------|-----------------|
| `test_track_usage_writes_local` | `track_usage(...)` writes a JSONL event to the log file. Verifies event, model_name, task, hf_user, device, and timestamp fields. |
| `test_track_usage_with_hf_user` | When `hf_user` is a dict `{"name": ..., "fullname": ...}`, only the `name` field is extracted and stored. |
| `test_track_usage_calls_callbacks` | Registered callbacks are called with the usage dict. Verifies callback receives correct model_name. |
| `test_callback_exception_does_not_crash` | A failing callback (raises RuntimeError) does not prevent other callbacks from running, and the local log file is still written. |
| `test_get_usage_log_empty` | `get_usage_log()` returns empty list when log file does not exist. |
| `test_get_usage_log_reads_events` | `get_usage_log()` reads existing JSONL file and returns parsed events. |
| `test_get_usage_log_skips_blank_lines` | Blank lines and whitespace-only lines in the JSONL file are silently skipped. |

### Fixtures

- `clear_callbacks` (autouse) -- clears `_callbacks` before/after each test.

### Mocking Strategy

- Patches `revos.usage._USAGE_LOG` to redirect writes to `tmp_path`.
- Uses `MagicMock` for callback functions.

### Parametrized Tests / Markers

None.

---

## 14. Cross-Cutting Observations

### Test Distribution Summary

| File | Lines | Test Count | Pattern | Mocking Level |
|------|-------|------------|---------|---------------|
| conftest.py | 52 | 0 (fixtures) | Shared setup | N/A |
| test_asr.py | 130 | 5 | Unit | Heavy (sherpa_onnx, ensure_model, get) |
| test_tts.py | 176 | 7 | Unit | Heavy (omnivoice module injection) |
| test_tts_long.py | 269 | 13 | Unit | Heavy (omnivoice module injection) |
| test_audio.py | 57 | 3 | Unit | None (real WAV I/O) |
| test_catalog.py | 217 | 10 | Unit | Medium (GitHub API mocks) |
| test_catalog_cli.py | 139 | 7 | CLI integration | Medium (catalog + Click runner) |
| test_cli.py | 198 | 13 | CLI integration | Heavy (ASR/TTS factory mocks) |
| test_device.py | 25 | 3 | Unit | Medium (onnxruntime mocks) |
| test_downloader.py | 205 | 12 | Unit | Low (real archives, mocked network) |
| test_integration.py | 295 | 15 | E2E/Integration | None (real models + network) |
| test_registry.py | 119 | 5 | Unit | None (in-memory + YAML) |
| test_usage.py | 148 | 7 | Unit | Low (mocked file paths) |
| **Totals** | **~2,050** | **~100** | | |

### Test Pattern Breakdown

- **Unit tests**: 73 tests across test_asr, test_tts, test_tts_long, test_audio, test_catalog, test_device, test_downloader, test_registry, test_usage
- **CLI integration tests**: 20 tests across test_cli, test_catalog_cli
- **E2E/Integration tests**: 15 tests in test_integration (marked `slow`)
- **No parametrized tests** anywhere in the suite
- **No property-based/fuzz tests** (e.g., hypothesis)
- **No concurrency/parallel tests**

### Fixture Usage Patterns

1. **`clear_registry` (autouse)**: Used in test_tts.py, test_tts_long.py, test_catalog.py, test_catalog_cli.py, test_registry.py. Directly manipulates the global `_models` dict. This is a code smell indicating the registry is a global mutable singleton.
2. **`clear_callbacks` (autouse)**: Used in test_usage.py. Same pattern for `_callbacks`.
3. **`sample_wav` (conftest)**: Only used in test_asr.py and test_cli.py.
4. **`mock_recognizer` and `mock_tts_model` (conftest)**: Defined but **unused** -- dead fixtures. The ASR and TTS tests create their own inline mocks instead.

### Mocking Patterns

The test suite uses three primary mocking strategies:

1. **Module injection** (`patch.dict(sys.modules, ...)`) -- used to inject a fake `omnivoice` module since it is an optional/private dependency. This is the most complex mock pattern in the suite.
2. **Function/class patching** (`@patch("revos.asr.ASR")`) -- standard mock-the-factory pattern to avoid loading real models.
3. **Environment variable patching** (`patch.dict(os.environ, ...)`) -- for catalog repo configuration tests.

### Markers

- **`pytest.mark.slow`**: Only used in `test_integration.py` (module-level `pytestmark`). Allows excluding with `pytest -m "not slow"`.
- No other custom markers are defined or used (no `@pytest.mark.xfail`, `@pytest.mark.skipif`, etc. beyond the TTS auth check).

### Coverage Gaps

The following areas have **no test coverage** or have notable gaps:

1. **`revos.cli.commands.transcribe` / `revos.cli.commands.synthesize`**: The CLI command implementation modules are only tested indirectly through Click's `CliRunner`. Edge cases like invalid file paths, corrupt WAV files, missing model names, and permission errors on output files are not tested.

2. **Error handling in ASR/TTS pipelines**: No tests for:
   - Corrupt or empty audio files passed to `transcribe()`
   - Non-WAV file formats (MP3, FLAC, OGG)
   - Very long audio files (multi-hour)
   - GPU/CUDA-specific code paths
   - `ref_audio` (voice cloning) parameter in TTS
   - Speed parameter validation (negative, zero, extreme values)

3. **Catalog edge cases**:
   - Partial YAML parsing failures (malformed YAML)
   - Rate limiting or HTTP error codes from GitHub API
   - Concurrent catalog access
   - Disk full during `pull_model`

4. **Downloader edge cases**:
   - Network timeouts during `_download`
   - Corrupt archive files (partial downloads)
   - Disk space validation
   - Large model downloads (multi-GB)
   - Authentication-required URLs

5. **Registry edge cases**:
   - Thread safety of global `_models` dict
   - Manifest validation (missing required fields, invalid types)
   - Duplicate registration with different tasks

6. **Usage tracking edge cases**:
   - Log file rotation / large log files
   - Concurrent writes to the JSONL file
   - Malformed existing log entries

7. **Device detection edge cases**:
   - Other execution providers (TensorRT, CoreML, ROCM)
   - CUDA version compatibility checks
   - Multiple GPU selection

8. **`test_audio.py` gaps**:
   - Non-16kHz target sample rates
   - Very short (1-sample) or very long audio files
   - Non-WAV formats
   - Float64 input samples

9. **`test_tts_long.py` gaps**:
   - `synthesize_long` with very long text (thousands of characters)
   - `synthesize_long` with only whitespace (similar to empty but with content)
   - `_split_text` with newlines, tabs, and other whitespace
   - Concatenation of many (10+) audio segments

10. **No performance/load tests**: No tests verify latency, throughput, or memory usage under load.

11. **Dead fixtures in conftest.py**: `mock_recognizer` and `mock_tts_model` are defined but never used, suggesting they were created for tests that were refactored to use inline mocks instead.

12. **No test for `revos.__init__`** top-level imports or package version.

13. **No tests for the `revos.tts.base` `TTSEngine` base class** methods beyond `_split_text` (which is a private function).

14. **No tests for `revos.asr.base` `ASREngine` base class**.
