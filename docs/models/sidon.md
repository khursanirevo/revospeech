# sidon — Speech Restoration

Sidon is a 3-stage ONNX speech-restoration pipeline: denoise,
dereverberation, and bandwidth extension. Input is degraded 16 kHz speech;
output is clean 48 kHz wideband speech.

| | |
|---|---|
| **Backend** | `sidon` (ONNX Runtime) |
| **Mode** | local |
| **Task** | util |
| **Size** | ~940 MB (predictor + vocoder; mel front-end bundled) |
| **Input sample rate** | 16 kHz (resampled internally if different) |
| **Output sample rate** | 48 kHz |
| **Capabilities** | denoise, dereverberation, bandwidth-extension |

## Pipeline

```
audio@16kHz
  -> mel_frontend.onnx      (bundled, ~1.1 MB)
  -> sidon-predictor.onnx   (w2v-BERT 2.0, ~740 MB)
  -> sidon-vocoder.onnx     (DAC decoder, ~200 MB)
  -> audio@48kHz
```

Stages run as separate ONNX sessions. The mel front-end is shipped inside
the wheel (`revospeech/util/assets/mel_frontend.onnx`); the predictor and
vocoder download on first use.

## Parameters

`SidonUtil.restore()` has no tunable knobs — the pipeline is fixed. The
only contract is the I/O shape:

| | Type | Notes |
|---|---|---|
| `audio` (in) | `Audio` | Any sample rate; stereo is mixed to mono. Resampled to 16 kHz internally. |
| Return | `Audio` | Mono float32 at 48 kHz. Length matches the input duration scaled by `48000 / input_sr`. |

Internals visible in `revospeech/util/sidon_engine.py`:

| Constant | Value | Effect |
|---|---|---|
| `INPUT_SAMPLE_RATE` | 16000 | Internal pipeline rate. |
| `OUTPUT_SAMPLE_RATE` | 48000 | Vocoder output rate. |
| `HOP_LENGTH` | 160 | Frame advance for mel alignment. |
| Window length | 400 samples | Inputs shorter than this are zero-padded. |

A 5 ms linear fade is applied before the last non-zero sample to suppress
click artifacts from the DAC vocoder's cutoff. Trailing samples after the
fade are zeroed.

## Usage

Standalone:

```python
from revospeech.util import Util
from revospeech.tts.result import Audio

sidon = Util("sidon")                # auto-downloads on first call
audio = sidon.restore_file("noisy.wav", "clean.wav")
# Or restore an in-memory Audio:
# audio = sidon.restore(existing_audio)
```

CLI:

```bash
# Default model is 'sidon'
revospeech restore -i noisy.wav -o clean.wav

# Explicit model name
revospeech restore -m sidon -i noisy.wav -o clean.wav
```

As a TTS post-processor (opt-in, off by default):

```python
from revospeech import TTS

tts = TTS("vits-ms", restore=True)
audio = tts.synthesize("Hello, world!")
# audio is denoised + dereverberated + 48 kHz
audio.save("enhanced.wav")
```

```bash
revospeech synthesize -m vits-ms -t "Hello!" -o out.wav --restore
```

## Performance

- **Cold start:** ~7 s (ONNX session init + first inference)
- **Warm RTF:** ~0.5× (faster than real-time on CPU)
- **GPU:** Optional; set `Util("sidon", device="cuda")` to use CUDA EP

## Limitations

- **Single-speaker assumption.** Multi-speaker or heavily overlapping
  speech is not restored cleanly.
- **No tunable strength.** Restoration is all-or-nothing per call.
- **48 kHz output.** Downsample if your downstream pipeline expects 16 kHz
  ASR input.

## See also

- [Speech Restoration overview](../util.md)
- [Util API reference](../api/util.md)
- [Model catalog](../models.md)
