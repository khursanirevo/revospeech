# Util API

Speech restoration and audio post-processing models.

## `Util(model_name, *, device, auto_download)`

Factory for util models. Dispatches by manifest `backend`.

```python
from revospeech.util import Util, BaseUtil
```

### Args

| Name | Default | Description |
|---|---|---|
| `model_name` | `"sidon"` | Util model name |
| `device` | `"auto"` | `"auto"`, `"cpu"`, or `"cuda"` |
| `auto_download` | `True` | Download model files if not cached |

### Returns

A `BaseUtil` subclass instance.

### Example

```python
sidon = Util("sidon")
enhanced = sidon.restore(audio)  # audio: Audio -> Audio
```

## `BaseUtil`

Abstract base class. Subclasses implement `restore(audio) -> Audio`.

### `restore(audio)` (abstract)

Restore / enhance an `Audio` input. Returns `Audio`.

### `restore_file(input_path, output_path=None)`

Convenience wrapper: read audio file, call `restore`, optionally save.

```python
sidon = Util("sidon")
audio = sidon.restore_file("noisy.wav", "clean.wav")
```

## `SidonUtil`

Concrete implementation for the Sidon 3-stage ONNX pipeline. Constructed
indirectly via `Util("sidon")`.

### Pipeline

```
Audio@16kHz
  -> mel_frontend.onnx       (bundled with the package)
  -> sidon-predictor.onnx    (w2v-BERT 2.0)
  -> sidon-vocoder.onnx      (DAC decoder)
  -> Audio@48kHz
```

Input at sample rates other than 16 kHz is linear-resampled before processing.
Output is trimmed / faded at the tail to suppress vocoder cutoff clicks.

## Applying util models to TTS output

`TTS()` accepts a `restore=` flag (default `False`). When `True`, any ready
util model tagged `tts-postprocess` runs after each `synthesize()` call.

```python
from revospeech import TTS

tts = TTS("vits-ms", restore=True)
audio = tts.synthesize("Hello, world!")
# audio has been through Sidon restoration
```

The equivalent CLI flag is `revospeech synthesize --restore`.

## See also

- [Speech Restoration overview](../util.md)
- [TTS API](tts.md)
- [Registry API](registry.md) for manifest schema
