# vits-ms — Malay Multi-Speaker TTS

VITS end-to-end neural TTS trained on Malay, served via ONNX Runtime with
Piper-compatible phonemization (`espeak-ng --ipa=3`).

| | |
|---|---|
| **Backend** | `vits` (ONNX Runtime, CPU) |
| **Mode** | local |
| **Languages** | Malay (`ms`) |
| **Size** | ~500 MB |
| **Sample rate** | 22050 Hz |
| **Capabilities** | multi-speaker |
| **License** | MIT (code) |

## Speakers

Three production-vetted speakers. Pick one via the `speaker=` kwarg on
`synthesize()`, or call `list_voices()` to enumerate at runtime.

| Speaker | Voice ID |
|---|---|
| Sarah (default) | `sarah` |
| Paan | `paan` |
| Anwar | `anwar` |

```python
from revospeech import TTS

tts = TTS("vits-ms")
print(tts.list_voices())  # ['sarah', 'paan', 'anwar']
```

Any other string raises `ValueError`. To add a new speaker, drop a
`speakers/<name>/model.onnx` + `model.onnx.json` pair into the cached repo
and extend `PRODUCTION_SPEAKERS` in `revospeech/tts/vits_engine.py`.

## Synthesis parameters

`VitsTTS.synthesize()` accepts:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | — | Malay text to synthesize. |
| `output_path` | `str \| Path \| None` | `None` | If set, writes WAV to this path. |
| `speaker` | `str` | `"sarah"` | One of `sarah`, `paan`, `anwar`. |
| `speed` | `float` | `1.0` | Speaking-rate multiplier. `2.0` ≈ 2× faster. |
| `sentence_silence` | `float` | `0.0` | Seconds of silence inserted between detected sentences. |

The following ONNX inference fields are read from each speaker's
`model.onnx.json` (`inference` block). Override them by editing the JSON
in the cached model directory — they are not exposed as kwargs because
they fall outside the happy path.

| Field | Default | Range | Effect |
|---|---|---|---|
| `noise_scale` | `0.667` | 0.0–1.0 | How much randomness in pitch/duration prediction. Higher = more expressive, less stable. |
| `length_scale` | `1.0` | >0.0 | Stretches (`>1.0`) or compresses (`<1.0`) duration. Multiplicative with `speed`. |
| `noise_w` | `0.8` | 0.0–1.0 | Stochasticity of the phoneme-duration predictor. |

Final duration scale applied to the model is `length_scale / speed`.

## Usage

Python:

```python
from revospeech import TTS

tts = TTS("vits-ms")  # auto-downloads on first call

# Default speaker (sarah)
tts.synthesize("Selamat pagi, apa khabar?", output_path="sarah.wav")

# Speaker swap + slower delivery
tts.synthesize("Terima kasih banyak-banyak.",
               speaker="paan", speed=0.9,
               output_path="paan.wav")

# Batch
report = tts.synthesize_batch(
    ["Satu.", "Dua.", "Tiga."],
    output_dir="outputs/",
)
```

CLI — the CLI exposes `--speed` only. To pick a non-default speaker, use
the Python API.

```bash
revospeech synthesize -m vits-ms -t "Selamat pagi." -o out.wav
revospeech synthesize -m vits-ms -t "Selamat pagi." --speed 1.2 -o out.wav
```

## Limitations

- **Malay only.** The espeak-ng voice is hard-pinned to `ms`; feeding
  English or Mandarin will produce garbled phonemes.
- **espeak-ng required.** Install via `sudo apt install espeak-ng`
  (Debian/Ubuntu) or `brew install espeak-ng` (macOS).
- **No voice cloning.** VITS is a fixed multi-speaker model. For zero-shot
  cloning use [revovoice](revovoice.md).

## See also

- [Model catalog](../models.md)
- [TTS API reference](../api/tts.md)
- [CLI reference](../cli.md)
