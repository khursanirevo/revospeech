# ASR Module Analysis

## Overview

The `revos.asr` module provides Automatic Speech Recognition functionality. It follows a backend-dispatch pattern: a factory function (`ASR()`) selects the concrete engine based on a model manifest, while a shared abstract base class enforces a uniform interface. Currently only one backend is implemented -- sherpa-onnx for Zipformer transducer models.

**Canonical usage:**

```python
from revos.asr import ASR

asr = ASR("zipformer-v2")
result = asr.transcribe("audio.wav")
print(result.text)
```

---

## File-by-File Analysis

### 1. `revos/asr/__init__.py`

**Purpose:** Public API surface and factory dispatch for the entire ASR subsystem.

**Responsibility:** Re-exports the key symbols (`ASR`, `BaseASR`, `Transcript`, `Segment`) and implements the `ASR()` factory function that resolves a model name to a concrete backend.

**Key function:**

```python
def ASR(model_name: str, device: str = "auto") -> BaseASR
```

- Calls `revos.registry.get(model_name, "asr")` to fetch a `ModelManifest`.
- Inspects `manifest.backend`. If `"sherpa-onnx"`, lazily imports `SherpaOnnxASR` and returns an instance.
- Otherwise raises `ValueError` with a message listing supported backends.
- The import of `SherpaOnnxASR` is deferred inside the `if` block, so sherpa-onnx is only imported when actually needed.

**Re-exports (`__all__`):**

```python
__all__ = ["ASR", "BaseASR", "Transcript", "Segment"]
```

**Dependencies:**
- Internal: `revos.registry.get`, `.base.BaseASR`, `.result.Segment`, `.result.Transcript`
- External: none at import time (sherpa-onnx imported lazily)

**Design pattern:** Factory function with lazy backend loading. This avoids forcing users to install sherpa-onnx if they never use ASR, and makes it trivial to add new backends by extending the `if/elif` dispatch.

---

### 2. `revos/asr/base.py`

**Purpose:** Defines the abstract interface that all ASR engines must implement.

**Responsibility:** Establishes the contract for ASR backends via an ABC.

**Key class:**

```python
class BaseASR(ABC):
    """Base class for automatic speech recognition engines."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def transcribe(self, audio_path: str) -> Transcript:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Transcript object with text, segments, and language.
        """
        ...
```

**Instance attributes:**
- `self.model_name: str` -- the model identifier (e.g. `"zipformer-v2"`)
- `self.device: str` -- compute device selector (`"auto"`, `"cpu"`, or `"cuda"`)

**Dependencies:**
- Internal: `.result.Transcript`
- External: `abc` (stdlib)

**Design pattern:** Template Method / Strategy pattern. `BaseASR` is the strategy interface; concrete subclasses provide the `transcribe()` implementation. The `__init__` stores configuration shared across all backends.

---

### 3. `revos/asr/audio.py`

**Purpose:** Audio file I/O utility for the ASR subsystem.

**Responsibility:** Reads audio files from disk, converts to mono float32, and resamples to a target sample rate.

**Key function:**

```python
def read_waveform(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]
```

**Parameters:**
- `path: str` -- path to an audio file (WAV, FLAC, or any format supported by `soundfile`)
- `target_sr: int` -- target sample rate in Hz (default: 16000, the standard for speech models)

**Returns:**
- `tuple[np.ndarray, int]` -- `(samples, sample_rate)` where `samples` is a 1-D float32 NumPy array and `sample_rate` is an integer (equal to `target_sr` after resampling)

**Processing pipeline:**
1. Reads audio via `soundfile.read(path, dtype="float32")`, getting `(data, sr)`.
2. **Mono conversion:** If `data.ndim > 1` (stereo/multi-channel), collapses to mono by averaging channels (`data.mean(axis=1)`). Logs at DEBUG level.
3. **Resampling:** If the native sample rate `sr` differs from `target_sr`, performs linear-interpolation resampling:
   - Computes `duration = len(data) / sr`
   - Computes `target_len = int(duration * target_sr)`
   - Creates `indices = np.linspace(0, len(data) - 1, target_len)`
   - Interpolates: `np.interp(indices, np.arange(len(data)), data).astype(np.float32)`
   - Sets `sr = target_sr`
   - Logs at DEBUG level.

**Dependencies:**
- Internal: none
- External: `numpy` (as `np`), `soundfile` (as `sf`)

**Design decisions:**
- Resampling uses `np.interp` (linear interpolation), which is fast and dependency-free but less accurate than proper band-limited resampling (e.g., `librosa.resample` or `scipy.signal.resample`). This is a pragmatic tradeoff: it avoids adding `librosa`/`scipy` as dependencies and is adequate for most ASR use cases where the model is robust to minor resampling artifacts.
- The default `target_sr=16000` matches the standard sample rate for most speech recognition models (including Zipformer).

---

### 4. `revos/asr/result.py`

**Purpose:** Data classes for ASR transcription output.

**Responsibility:** Defines the structured result types returned by all ASR engines.

**Key classes:**

```python
@dataclass
class Segment:
    """A single transcription segment with timing and confidence."""

    start: float       # Start time in seconds
    end: float         # End time in seconds
    text: str          # Transcribed text of this segment
    confidence: float  # Confidence score (0.0 to 1.0; currently always 0.0 in sherpa-onnx)
```

```python
@dataclass
class Transcript:
    """Full transcription result from ASR."""

    text: str            # Complete transcribed text (all segments concatenated)
    segments: list[Segment]  # Individual segments with timing
    language: str        # Detected or configured language code (e.g. "en", "")
```

**Dependencies:**
- Internal: none
- External: `dataclasses` (stdlib)

**Design decisions:**
- These are plain `@dataclass` types with no methods, following a data-transfer-object (DTO) pattern. This keeps them serializable and easy to inspect.
- `confidence` is always set to `0.0` in the current sherpa-onnx backend because `sherpa_onnx`'s `OfflineRecognizer` does not expose per-word confidence scores. The field exists for forward compatibility with backends that do provide confidence.
- `language` is populated from `result.lang` in sherpa-onnx, but may be an empty string `""` if the model does not report language detection.

---

### 5. `revos/asr/sherpa_engine.py`

**Purpose:** Concrete ASR backend using the `sherpa-onnx` library for Zipformer transducer models.

**Responsibility:** Downloads (if needed) and loads a sherpa-onnx transducer model, accepts audio input, and produces `Transcript` results.

**Key class:**

```python
class SherpaOnnxASR(BaseASR):
    """ASR engine using sherpa-onnx OfflineRecognizer."""
```

#### Constructor

```python
def __init__(self, model_name: str, device: str = "auto") -> None
```

**Initialization sequence:**

1. Calls `super().__init__(model_name, device)` to store `self.model_name` and `self.device`.
2. **Device resolution:** If `self.device == "auto"`, calls `revos.device.auto_detect_device()` which checks for CUDA via `onnxruntime.get_available_providers()`. Sets `self.device` to `"cuda"` or `"cpu"`.
3. **Provider mapping:** Maps device to sherpa-onnx provider string: `"cuda"` -> `"cuda"`, everything else -> `"cpu"`.
4. **Manifest lookup:** Calls `revos.registry.get(model_name, "asr")` to get a `ModelManifest`.
5. **Model download:** Calls `revos.registry.ensure_model(manifest)` to download the model if not already cached locally. Returns a `Path` to the model directory.
6. **File path construction:** Extracts four file paths from `manifest.files` dict:
   - `encoder` -> encoder ONNX file
   - `decoder` -> decoder ONNX file
   - `joiner` -> joiner ONNX file
   - `tokens` -> tokens text file
7. **Recognizer creation:** Calls `sherpa_onnx.OfflineRecognizer.from_transducer(...)` with:
   ```python
   sherpa_onnx.OfflineRecognizer.from_transducer(
       encoder=encoder,    # str path to encoder ONNX
       decoder=decoder,    # str path to decoder ONNX
       joiner=joiner,      # str path to joiner ONNX
       tokens=tokens,      # str path to tokens file
       num_threads=2,      # hardcoded thread count
       sample_rate=manifest.sample_rate,  # from manifest (typically 16000)
       provider=provider,  # "cpu" or "cuda"
   )
   ```
8. **State storage:** Stores `self._sample_rate`, `self._model_id` for later use.
9. **Usage tracking:** If the model is private (`manifest.hf_private`) or has an HTTP URL, imports and calls `revos.usage.track_usage()` with event `"model_loaded"`.

**Instance attributes:**
- `self.model_name: str` (inherited)
- `self.device: str` (inherited, possibly resolved from `"auto"`)
- `self._recognizer: sherpa_onnx.OfflineRecognizer` -- the loaded ONNX transducer recognizer
- `self._sample_rate: int` -- sample rate from manifest
- `self._model_id: str` -- model URL from manifest

#### Transcribe method

```python
def transcribe(self, audio_path: str) -> Transcript
```

**Processing pipeline:**

1. **Audio loading:** Calls `read_waveform(audio_path, target_sr=self._sample_rate)` to get `(samples, sr)`.
2. **Stream creation:** `self._recognizer.create_stream()` creates a new offline recognition stream.
3. **Waveform injection:** `stream.accept_waveform(sr, samples)` feeds the audio into the stream.
4. **Decoding:** `self._recognizer.decode_stream(stream)` runs the transducer model inference.
5. **Result extraction:** Accesses `stream.result` which has `.text`, `.timestamps`, and `.lang`.
6. **Segment construction:** Two paths:
   - **With timestamps** (typical): Iterates over `text.split()` words paired with `result.timestamps`. Each word becomes a `Segment` with `start = timestamps[i]`, `end = timestamps[i+1]` (or `start + 0.1` for the last word), `confidence=0.0`.
   - **Without timestamps** (fallback): Creates a single `Segment(start=0.0, end=0.0, text=text, confidence=0.0)`.
7. **Transcript assembly:** Returns `Transcript(text=text, segments=segments, language=result.lang or "")`.

**Dependencies:**
- Internal: `revos.device.auto_detect_device`, `revos.registry.ensure_model`, `revos.registry.get`, `.audio.read_waveform`, `.base.BaseASR`, `.result.Segment`, `.result.Transcript`
- Conditional internal: `revos.usage.track_usage` (only imported for private/HTTP models)
- External: `sherpa_onnx`, `numpy` (transitively via `audio.py`)

---

## Class Hierarchy

```
ABC
 └── BaseASR                   (base.py)
      └── SherpaOnnxASR        (sherpa_engine.py)
```

The `ASR()` factory function in `__init__.py` acts as the public constructor, returning `BaseASR` (actually a `SherpaOnnxASR` instance).

---

## Data Flow: Raw Audio to Transcription Result

```
User code
  │
  ▼
ASR(model_name="zipformer-v2", device="auto")     # __init__.py factory
  │
  ├─ registry.get("zipformer-v2", "asr")           # lookup manifest
  ├─ manifest.backend == "sherpa-onnx"             # dispatch
  └─ SherpaOnnxASR("zipformer-v2", "auto")         # instantiate
       │
       ├─ auto_detect_device()                      # resolve "auto" -> "cpu"/"cuda"
       ├─ ensure_model(manifest)                    # download if needed
       └─ sherpa_onnx.OfflineRecognizer             # load ONNX model
            .from_transducer(...)
  │
  ▼
asr.transcribe("audio.wav")                         # sherpa_engine.py
  │
  ├─ read_waveform("audio.wav", target_sr=16000)   # audio.py
  │    ├─ sf.read(...) -> float32 array             # soundfile I/O
  │    ├─ mono conversion (if multi-channel)
  │    └─ linear-interpolation resampling (if needed)
  │
  ├─ stream = recognizer.create_stream()            # sherpa-onnx stream
  ├─ stream.accept_waveform(sr, samples)            # feed audio
  ├─ recognizer.decode_stream(stream)               # run inference
  │
  ├─ result = stream.result
  │    ├─ .text       -> "hello world"
  │    ├─ .timestamps -> [0.0, 0.5]
  │    └─ .lang       -> "en"
  │
  └─ build Segment list from word/timestamp pairs
       │
       ▼
  Transcript(
    text="hello world",
    segments=[
      Segment(start=0.0, end=0.5, text="hello", confidence=0.0),
      Segment(start=0.5, end=0.6, text="world", confidence=0.0),
    ],
    language="en",
  )
```

---

## Design Patterns

| Pattern | Where | Notes |
|---|---|---|
| **Factory function** | `__init__.py :: ASR()` | Decouples caller from concrete backend class; enables lazy imports. |
| **Strategy / Template Method** | `BaseASR` ABC | Defines the interface; backends plug in via `transcribe()`. |
| **DTO (Data Transfer Object)** | `Segment`, `Transcript` | Pure data dataclasses with no behavior; easy to serialize and inspect. |
| **Lazy import** | `__init__.py`, `sherpa_engine.py` | `sherpa_onnx` and `revos.usage` imported only when needed. |
| **Registry / Service Locator** | `revos.registry.get()` | Model manifests are looked up by name and task type, decoupling model discovery from engine code. |
| **Auto-detection** | `revos.device.auto_detect_device()` | Probes onnxruntime for CUDA availability at runtime. |

---

## Cross-Module Dependencies

```
revos.asr
  ├── revos.registry          (get, ensure_model, ModelManifest)
  ├── revos.device            (auto_detect_device)
  ├── revos.usage             (track_usage -- conditional import)
  ├── sherpa_onnx             (OfflineRecognizer)
  ├── numpy                   (audio resampling)
  ├── soundfile               (audio I/O)
  └── abc, dataclasses, logging (stdlib)
```

---

## Notable Design Decisions

1. **Word-level segmentation, not utterance-level:** The sherpa-onnx backend splits transcription into per-word `Segment` objects using `text.split()` paired with `result.timestamps`. This is a lower granularity than typical utterance-level segmentation but matches what sherpa-onnx provides. Each "segment" is actually a single word.

2. **Confidence always 0.0:** The `Segment.confidence` field is never populated with a real value because `sherpa_onnx.OfflineRecognizer` does not expose per-word confidence. The field is architectural forward-compatibility.

3. **Hardcoded `num_threads=2`:** The thread count for sherpa-onnx inference is fixed at 2. This is not configurable via the constructor or manifest. For production use on high-core-count machines, this may need to be made configurable.

4. **Linear-interpolation resampling:** The `read_waveform()` function uses `np.interp` rather than a proper band-limited resampler. This avoids heavy dependencies (scipy, librosa) but may introduce aliasing artifacts for large sample rate ratios.

5. **Conditional usage tracking:** `revos.usage.track_usage` is only imported and called for private models or HTTP-sourced models (`manifest.hf_private or manifest.model_url.startswith("http")`). This avoids overhead for standard open models loaded from local cache.

6. **Single-backend dispatch:** The factory currently only supports `"sherpa-onnx"`. Adding a new backend (e.g., Whisper via faster-whisper, or a cloud API) requires adding an `elif` branch in `ASR()` and a new `BaseASR` subclass. The architecture supports this cleanly.
