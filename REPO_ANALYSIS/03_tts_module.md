# TTS Module Analysis

## Overview

The `revos.tts` package provides text-to-speech synthesis backed by the RevoVoice diffusion model. It exposes a single public entry-point function (`TTS`), an abstract base class (`BaseTTS`) that defines the engine contract, a concrete engine (`RevoVoiceTTS`), and a value-object (`Audio`) that wraps synthesized waveforms.

---

## File-by-File Analysis

### 1. `revos/tts/__init__.py`

**Purpose:** Public API surface for the TTS package. Acts as the factory/dispatcher that resolves a model name to the correct backend engine.

**Key Symbols:**

```python
def TTS(model_name: str, device: str = "auto") -> BaseTTS
```

Factory function that:
1. Calls `revos.registry.get(model_name, "tts")` to retrieve a model manifest.
2. Inspects `manifest.backend` and dispatches to the matching engine class.
3. Currently only the `"revovoice"` backend is supported; anything else raises `ValueError`.

**Public exports:**

```python
__all__ = ["TTS", "BaseTTS", "Audio"]
```

**Internal dependencies:**
- `revos.registry.get` -- model manifest lookup
- `.base.BaseTTS` -- abstract base class (also re-exported)
- `.result.Audio` -- result dataclass (also re-exported)

**External dependencies:** None beyond the standard library.

**Design pattern:** Factory function (not a class). The naming convention `TTS(...)` mimics a constructor call but is actually a function that returns a `BaseTTS` subclass instance. This lets callers swap backends transparently without knowing the concrete class.

---

### 2. `revos/tts/base.py`

**Purpose:** Defines the abstract contract for all TTS engines and provides text-splitting utilities for long-form synthesis.

#### Constants

```python
_SENTENCE_RE: re.Pattern  # compiled regex: (?<=[.!?])\s+|(?<=[。！？])
DEFAULT_MAX_CHARS: int = 500
```

`_SENTENCE_RE` matches sentence boundaries for both Latin-script punctuation (`. ! ?` followed by whitespace) and CJK punctuation (`。！？` without requiring whitespace).

#### Module-level Functions

```python
def _split_text(text: str, max_chars: int = 500) -> list[str]
```

Splits text into chunks of at most `max_chars` characters. Strategy:
1. If total length <= `max_chars`, return the whole text as a single chunk.
2. Split on sentence boundaries via `_SENTENCE_RE`.
3. Greedily pack sentences into chunks up to `max_chars`.
4. Any remaining oversized chunks are handed off to `_split_long_chunk`.

```python
def _split_long_chunk(text: str, max_chars: int) -> list[str]
```

Fallback splitter for individual chunks that still exceed `max_chars`:
1. First attempt: split at comma/semicolon boundaries (both `, ;` and `，；`).
2. Last resort: split at word boundaries (whitespace-separated tokens).
3. Greedy packing into chunks of at most `max_chars`.

Both functions are private (underscore-prefixed) and used exclusively by `BaseTTS.synthesize_long`.

#### Class: `BaseTTS(ABC)`

```python
class BaseTTS(ABC):
    def __init__(self, model_name: str, device: str = "auto") -> None
```

Stores `model_name` and `device` as instance attributes.

**Abstract method:**

```python
@abstractmethod
def synthesize(
    self,
    text: str,
    output_path: str | None = None,
    *,
    speed: float = 1.0,
    ref_audio: str | None = None,
    ref_text: str | None = None,
) -> Audio
```

Every concrete TTS engine must implement `synthesize`. Parameters:
- `text` -- the text to convert to speech.
- `output_path` -- optional file path to save the audio (`.wav`).
- `speed` -- speech rate multiplier; `1.0` is normal speed.
- `ref_audio` -- path to a reference audio file for voice cloning.
- `ref_text` -- transcription of the reference audio (improves cloning quality).

**Concrete method (template method pattern):**

```python
def synthesize_long(
    self,
    text: str,
    output_path: str | None = None,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    silence_duration: float = 0.1,
    speed: float = 1.0,
    ref_audio: str | None = None,
    ref_text: str | None = None,
) -> Audio
```

Provides a complete implementation for long-form synthesis that any subclass inherits without override. The method:
1. Splits `text` into chunks via `_split_text(text, max_chars)`.
2. Raises `ValueError` if the text is empty.
3. Iterates over chunks, calling `self.synthesize(...)` for each one, forwarding `speed`, `ref_audio`, and `ref_text`.
4. Collects all `Audio` segments and joins them via `Audio.concatenate(segments, silence_duration)`.
5. If `output_path` is given, saves the concatenated result.

This is the **Template Method pattern**: the algorithm structure is fixed in the base class, but the actual synthesis step (`self.synthesize`) is delegated to the subclass.

**Internal dependencies:**
- `.result.Audio` -- return type and concatenation helper

**External dependencies:**
- `re` -- sentence splitting regex
- `abc.ABC`, `abc.abstractmethod` -- abstract base class

---

### 3. `revos/tts/result.py`

**Purpose:** Defines the `Audio` dataclass -- the universal value object returned by all TTS engines.

#### Class: `Audio` (dataclass)

```python
@dataclass
class Audio:
    samples: np.ndarray
    sample_rate: int
```

Fields:
- `samples` -- a NumPy array of floating-point audio samples (mono waveform).
- `sample_rate` -- integer sample rate in Hz.

**Methods:**

```python
def save(self, path: str) -> None
```
Writes the waveform to disk using `soundfile.write`. The file format is inferred from the extension (e.g., `.wav`).

```python
@property
def duration(self) -> float
```
Computed property. Returns `len(self.samples) / self.sample_rate` -- the audio duration in seconds.

```python
@staticmethod
def concatenate(
    segments: list[Audio],
    silence_duration: float = 0.1,
) -> Audio
```

Joins multiple `Audio` objects into one:
1. Validates that `segments` is non-empty.
2. Validates that all segments share the same `sample_rate` (raises `ValueError` otherwise).
3. Creates a silence gap of `silence_duration` seconds as `np.zeros(int(sr * silence_duration), dtype=np.float32)`.
4. Interleaves silence gaps between segments and concatenates via `np.concatenate`.
5. Returns a new `Audio(samples=..., sample_rate=sr)`.

The silence is inserted *between* segments (not before the first or after the last).

**Internal dependencies:** None.

**External dependencies:**
- `numpy` -- sample storage and concatenation
- `soundfile` -- file I/O

**Design decisions:**
- The dataclass is immutable in spirit (no mutation methods beyond `save`).
- The `concatenate` method enforces sample-rate homogeneity, which prevents subtle audio artifacts from rate mismatches.
- Silence duration is configurable, defaulting to 100 ms -- a natural pause between sentences in long-form speech.

---

### 4. `revos/tts/revovoice_engine.py`

**Purpose:** Concrete TTS engine that wraps the `omnivoice.OmniVoice` diffusion-based zero-shot TTS model (branded as "RevoVoice").

#### Module-level Helper

```python
def _get_hf_user() -> dict | None
```

Calls `huggingface_hub.HfApi().whoami()` to identify the currently authenticated HuggingFace user. Returns a dict with keys `"name"` and `"fullname"`, or `None` if not authenticated. Used during model loading to log identity and track gated-model access.

#### Class: `RevoVoiceTTS(BaseTTS)`

**Inheritance chain:** `RevoVoiceTTS -> BaseTTS -> ABC`

```python
def __init__(self, model_name: str, device: str = "auto") -> None
```

Initialization sequence:
1. Calls `super().__init__(model_name, device)`.
2. Attempts `from omnivoice import OmniVoice`. Raises `ImportError` with install instructions (`pip install revos[tts]`) if not available.
3. Fetches the model manifest via `revos.registry.get(model_name, "tts")` and extracts `model_url` (the HuggingFace model ID) and `revision`.
4. Resolves the device:
   - `"auto"` -> checks `torch.cuda.is_available()` -> `"cuda"` or `"cpu"`.
   - The final `device_map` is formatted as `"{device}:0"` for CUDA (index 0) or `"cpu"` as-is.
5. Identifies the HuggingFace user via `_get_hf_user()`. Logs the username or warns if unauthenticated.
6. Loads the model via `OmniVoice.from_pretrained(model_id, device_map=device_map, revision=revision)`.
7. Catches `OSError` from the HuggingFace Hub and translates it into user-friendly error messages:
   - Gated model / authentication failure -> instructions to run `huggingface-cli login` or set `HF_TOKEN`.
   - 403 / access denied -> instructions to request access on the model page.
8. Stores `self._sample_rate` from the manifest, `self._model_id` for tracking, and `self._model` (the loaded OmniVoice instance).
9. Calls `revos.usage.track_usage(...)` with event `"model_loaded"` to record model load events including the HF user identity and device.

**Key instance attributes (beyond BaseTTS):**
| Attribute | Type | Description |
|---|---|---|
| `_model` | `OmniVoice` | The loaded diffusion TTS model |
| `_sample_rate` | `int` | Sample rate from model manifest |
| `_model_id` | `str` | HuggingFace model repository ID |
| `hf_user` | `dict or None` | Authenticated HF user info |

**Method:**

```python
def synthesize(
    self,
    text: str,
    output_path: str | None = None,
    *,
    speed: float = 1.0,
    ref_audio: str | None = None,
    ref_text: str | None = None,
) -> Audio
```

Synthesis flow:
1. Builds a `kwargs` dict with `text` and `speed`.
2. If `ref_audio` is provided, adds `ref_audio` to kwargs. If `ref_text` is also provided, adds `ref_text` as well. These enable voice cloning by providing a reference speaker audio clip and its transcription.
3. Calls `self._model.generate(**kwargs)` to produce audio.
4. The OmniVoice model returns a `list[np.ndarray]`. The engine extracts `result[0]` and converts to `np.float32`.
5. Wraps the samples in `Audio(samples=..., sample_rate=self._sample_rate)`.
6. If `output_path` is given, calls `audio.save(output_path)`.
7. Returns the `Audio` object.

**Internal dependencies:**
- `revos.registry.get` -- manifest lookup
- `revos.usage.track_usage` -- usage telemetry
- `.base.BaseTTS` -- abstract base
- `.result.Audio` -- result wrapper

**External dependencies:**
- `omnivoice.OmniVoice` -- the diffusion TTS model (optional, gated behind `revos[tts]` extra)
- `numpy` -- array handling
- `huggingface_hub.HfApi` -- authentication check
- `torch` -- CUDA availability check (optional, gracefully falls back to CPU)

---

## Class Hierarchy

```
ABC
 └── BaseTTS
      └── RevoVoiceTTS
```

The factory function `TTS()` sits outside the hierarchy and returns a `BaseTTS` (actually a `RevoVoiceTTS`) instance.

---

## Data Flow: Text to Audio

```
User code
  │
  ▼
TTS("revovoice")                     # __init__.py factory
  │  Looks up manifest via registry
  │  Dispatches to RevoVoiceTTS
  ▼
RevoVoiceTTS.__init__()              # Loads OmniVoice model
  │
  ▼
tts.synthesize("Hello, world!")      # or tts.synthesize_long(...)
  │
  ├─ synthesize_long (if called):
  │    1. _split_text() splits into chunks
  │    2. For each chunk → self.synthesize()
  │    3. Audio.concatenate() joins segments with silence
  │
  ▼
RevoVoiceTTS.synthesize()
  │  1. Builds kwargs: text, speed, ref_audio?, ref_text?
  │  2. self._model.generate(**kwargs)
  │  3. Convert result list → np.ndarray (float32)
  │  4. Wrap in Audio(samples, sample_rate)
  │  5. Optionally save to disk
  ▼
Audio                                # Returned to caller
  │  .save(path) → soundfile.write()
  │  .duration → computed property
```

---

## Voice Cloning Support

Voice cloning is enabled through two optional parameters on both `synthesize` and `synthesize_long`:

- **`ref_audio: str | None`** -- Path to a reference audio file of the target speaker's voice. This is passed directly to the OmniVoice model's `generate()` method.
- **`ref_text: str | None`** -- Transcription of the reference audio. Improves cloning quality by giving the model the text alignment for the reference clip.

When `ref_audio` is provided (with or without `ref_text`), the OmniVoice model performs zero-shot voice cloning -- it adapts the synthesized speech to match the vocal characteristics of the reference speaker without fine-tuning. This is characteristic of diffusion-based TTS models.

The same reference parameters are forwarded through `synthesize_long`, ensuring that long-form synthesis maintains voice consistency across all chunks.

---

## Long-Form Synthesis (`synthesize_long`)

`BaseTTS.synthesize_long()` is a concrete method (not abstract) available on all engines. It implements a chunk-and-concatenate strategy:

1. **Splitting** (`_split_text`):
   - Respects sentence boundaries using a regex that handles both Latin (`. ! ?` + space) and CJK (`。！？`) punctuation.
   - Default chunk size: 500 characters (`DEFAULT_MAX_CHARS`).
   - Falls back to comma/semicolon splits, then word-boundary splits for oversized chunks.

2. **Per-chunk synthesis**:
   - Each chunk is synthesized via `self.synthesize(...)` with the same `speed`, `ref_audio`, and `ref_text` parameters.
   - This ensures voice consistency and speed uniformity across all chunks.

3. **Concatenation** (`Audio.concatenate`):
   - Joins segments with configurable silence (default 100 ms).
   - Validates sample-rate homogeneity.
   - Returns a single `Audio` object.

4. **Persistence**:
   - Optionally saves the final concatenated audio to `output_path`.
   - Logs the total duration.

---

## Design Patterns

| Pattern | Where | Description |
|---|---|---|
| Factory Function | `__init__.py::TTS()` | Returns engine instances by name without exposing concrete classes |
| Template Method | `BaseTTS.synthesize_long()` | Fixed algorithm (split-synthesize-concatenate) that delegates the synthesis step to subclasses |
| Strategy | `BaseTTS` subclasses | Different TTS backends implement the same `synthesize` interface |
| Value Object | `Audio` dataclass | Immutable-feeling data holder with computed properties and a static utility |
| Lazy Import | `RevoVoiceTTS.__init__()` | Heavy dependencies (`omnivoice`, `torch`, `huggingface_hub`) imported at construction time, not module load time |

---

## Error Handling

- **Missing dependency:** `RevoVoiceTTS.__init__` catches `ImportError` for `omnivoice` and re-raises with install instructions.
- **Gated model access:** Catches `OSError` from HuggingFace Hub and translates into actionable error messages for 401 (authentication) and 403 (authorization) scenarios.
- **Empty text:** `synthesize_long` raises `ValueError` if splitting yields no chunks.
- **Sample rate mismatch:** `Audio.concatenate` raises `ValueError` if segments have different sample rates.
- **Empty concatenation:** `Audio.concatenate` raises `ValueError` if the segment list is empty.

---

## Telemetry

`RevoVoiceTTS.__init__` calls `revos.usage.track_usage(...)` after successful model loading, recording:
- Event type: `"model_loaded"`
- Model ID and name
- Task type: `"tts"`
- HuggingFace user identity (if authenticated)
- Device (cpu/cuda)

This is a one-time event per engine instantiation, not per-synthesis call.
