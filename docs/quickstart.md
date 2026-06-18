# Quickstart

## Install

```bash
pip install revospeech

# With TTS support
pip install revospeech[tts]

# Everything (GPU + TTS)
pip install revospeech[all]
```

## Transcribe audio

```python
from revospeech import ASR

# Auto-selects smallest ready model
asr = ASR()
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

tts = TTS()
audio = tts.synthesize("Hello, how are you?")
audio.save("greeting.wav")

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
