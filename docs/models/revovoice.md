# revovoice — Multilingual Zero-Shot TTS

RevoVoice is a diffusion-based TTS engine supporting 600+ languages with
zero-shot voice cloning from a short reference clip. The Python engine is
a thin wrapper around the `omnivoice` runtime.

| | |
|---|---|
| **Backend** | `revovoice` (diffusion) |
| **Mode** | local |
| **Languages** | 600+ (`multilingual`) |
| **Size** | ~1.2 GB |
| **Sample rate** | 24000 Hz |
| **Capabilities** | voice-cloning, streaming |
| **License** | MIT (code) |
| **Hardware** | GPU recommended (`min_vram_mb: 2048`) |

## How voices work

RevoVoice is **zero-shot** — there is no fixed voice registry. `list_voices()`
returns `[]`. Instead, you supply a reference clip and the model clones its
timbre, prosody, and recording conditions onto the target text.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | — | Text to synthesize. |
| `output_path` | `str \| Path \| None` | `None` | If set, writes WAV to this path. |
| `speed` | `float` | `1.0` | Speaking-rate multiplier. |
| `ref_audio` | `str \| None` | `None` | Path to a reference WAV. Required for cloning; omit for the model's default voice. |
| `ref_text` | `str \| None` | `None` | Optional transcription of `ref_audio`. Improves alignment when supplied; the engine auto-detects if omitted. |

For best clone quality:

- **Reference length**: 5–20 s of clean single-speaker speech.
- **Sample rate**: 16 kHz or higher; the engine resamples internally.
- **Match language**: Reference and target text should be the same language
  for natural prosody. Cross-lingual cloning works but loses fidelity.
- `ref_text` should be a verbatim transcript of `ref_audio`. Misalignment
  degrades clone similarity.

## Usage

Python:

```python
from revospeech import TTS

tts = TTS("revovoice")  # auto-downloads ~1.2 GB on first call

# Default voice (no cloning)
tts.synthesize("Hello, how are you?", output_path="default.wav")

# Voice cloning from a reference clip
tts.synthesize(
    "This will sound like the reference speaker.",
    ref_audio="speaker.wav",
    ref_text="Sample of the speaker talking.",  # optional but recommended
    output_path="cloned.wav",
)

# Long text auto-splits
audio = tts.synthesize_long(open("script.txt").read())
audio.save("long.wav")
```

CLI:

```bash
# Default voice
revospeech synthesize -m revovoice -t "Hello, how are you?" -o out.wav

# Voice cloning
revospeech synthesize -m revovoice \
    -t "This will sound like the reference speaker." \
    --ref-audio speaker.wav \
    --ref-text "Sample of the speaker talking." \
    -o cloned.wav
```

## Limitations

- **GPU recommended.** Runs on CPU but is slow; `min_vram_mb` is 2048.
- **Gated HF repo.** First-time download requires HuggingFace auth —
  run `revospeech config set-api-key` or export `HF_TOKEN`.
- **No multi-speaker preset.** Every voice is cloned on demand; there is
  no `speaker="X"` parameter.

## See also

- [Model catalog](../models.md)
- [TTS API reference](../api/tts.md)
- [vits-ms](vits-ms.md) for the CPU-friendly Malay alternative
