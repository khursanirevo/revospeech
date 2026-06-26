# Speech Restoration

RevoSpeech ships a **util** task category for audio post-processors that enhance
or restore speech. The first util model is **Sidon** — a 3-stage ONNX pipeline
that denoises, dereverberates, and bandwidth-extends speech audio.

## What Sidon does

| Input | Output |
|---|---|
| Noisy / reverberant 16 kHz speech | Clean, wideband 48 kHz speech |

Pipeline (all CPU via ONNX Runtime):

```
audio@16kHz
  -> mel_frontend.onnx      (bundled, ~1.1 MB)
  -> sidon-predictor.onnx   (w2v-BERT 2.0, ~740 MB)
  -> sidon-vocoder.onnx     (DAC decoder, ~200 MB)
  -> audio@48kHz
```

## Standalone use

CLI:

```bash
# Default model is 'sidon'
revospeech restore -i input.wav -o restored.wav

# Explicit model name
revospeech restore -m sidon -i input.wav -o restored.wav
```

Python:

```python
from revospeech.util import Util
from revospeech.tts.result import Audio

sidon = Util("sidon")
audio: Audio = sidon.restore_file("input.wav", "restored.wav")
```

## Apply to TTS output (opt-in)

Sidon can also run as a post-processor on TTS synthesis. This is **off by
default** — pass `restore=True` to opt in per call.

Python:

```python
from revospeech import TTS

tts = TTS("vits-ms", restore=True)
audio = tts.synthesize("Hello, world!")
# audio is now restored: denoised + dereverberated + 48 kHz
audio.save("enhanced.wav")
```

CLI:

```bash
revospeech synthesize -m vits-ms -t "Hello!" -o out.wav --restore
```

## Models

| Model | Backend | Size | Capabilities | Notes |
|---|---|---|---|---|
| `sidon` | sidon (ONNX) | ~940 MB | denoise, dereverberation, bandwidth-extension | Mel front-end bundled; predictor + vocoder download on first use |

## Performance

- Cold start: ~7 s (ONNX session init + first inference)
- Warm RTF: ~0.5× (faster than real-time on CPU)
- Output: 48 kHz mono float32

## See also

- [Util API reference](api/util.md)
- [CLI reference](cli.md) for the `restore` subcommand
- [Model catalog](models.md)
