# Quickstart

## First 60 seconds

```bash
pip install revospeech
revospeech setup            # interactive: pick a task, download a model
revospeech info             # verify install state
```

`revospeech setup` walks you through picking a task (ASR / TTS / both),
shows you the available models with status icons, and installs one. It also
offers to configure an API key for cloud backends.

> **Note:** `ASR()` and `TTS()` with no arguments auto-select from models
> that are **already downloaded** — they don't auto-pick-and-download.
> Run `revospeech setup` first, or pass an explicit model name like
> `ASR('zipformer-v2')` to trigger auto-download.

## Install variants

```bash
pip install revospeech                # core (ASR + speech restoration)
pip install revospeech[tts]           # adds RevoVoice (requires PyTorch)
pip install revospeech[gpu]           # ONNX Runtime GPU
pip install revospeech[all]           # GPU + TTS + API + playback
```

## Transcribe audio

```python
from revospeech import ASR

# Explicit model name auto-downloads on first use
asr = ASR("zipformer-v2")
result = asr.transcribe("meeting.wav")

print(result.text)
print(f"Duration: {result.duration:.1f}s")

for seg in result.segments:
    print(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text}")

# Save transcript in different formats
result.save("meeting.srt")  # or .vtt, .json, .txt
```

## Synthesize speech

```python
from revospeech import TTS

tts = TTS("vits-ms")
audio = tts.synthesize("Hello, how are you?")
audio.save("greeting.wav")

# Apply speech restoration (denoise + dereverb + 48 kHz bandwidth)
tts = TTS("vits-ms", restore=True)
audio = tts.synthesize("Hello, how are you?")

# Playback (requires sounddevice)
audio.play()
```

## Voice cloning

```python
audio = tts.synthesize(
    "This will sound like the reference speaker.",
    ref_audio="speaker.wav",
    ref_text="Sample of the speaker talking.",
)
audio.save("cloned.wav")
```

## Long text

```python
# Auto-splits at sentence boundaries
audio = tts.synthesize_long(open("book.txt").read(), max_chars=500)
audio.save("audiobook.wav")
```

## Batch processing

```python
# Transcribe many files in parallel
report = asr.transcribe_batch(["a.wav", "b.wav", "c.wav"], max_workers=4)
print(f"{report.succeeded}/{report.total} succeeded")
report.save("report.json")

# Synthesize many texts
report = tts.synthesize_batch(
    ["Hello.", "World.", "Goodbye."],
    output_dir="output/",
)
```

## Find models

```bash
revospeech models               # list local models
revospeech catalog list         # browse remote catalog
revospeech catalog recommend    # get recommendations
revospeech search "english"     # fuzzy search
```

## See also

- [CLI reference](cli.md) — every command and flag
- [Configuration](configuration.md) — API keys, cache, catalog source
- [Extension guide](extension.md) — adding custom models and backends
- [API reference](api.md) — full Python API
