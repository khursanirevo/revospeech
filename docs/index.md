# RevoSpeech

A unified Python library for speech AI — ASR, TTS, and speech restoration
using open models.

## Features

- **Zero-config defaults**: `ASR()` and `TTS()` auto-select a ready model
- **Local or API**: run models locally, or use revolab cloud API
- **Auto-download**: models fetch themselves on first use
- **Speech restoration**: denoise + dereverb + bandwidth extension (Sidon)
- **BytesIO support**: in-memory pipelines without disk I/O
- **Voice cloning**: zero-shot with reference audio
- **Batch processing**: parallel transcription and synthesis
- **Typed**: full type hints + py.typed marker
- **Extensible**: add custom backends via subclassing

## First 60 seconds

```bash
pip install revospeech
revospeech setup            # interactive: pick a task, download a model
revospeech info             # verify install state
```

Then:

```bash
revospeech transcribe -m zipformer-v2 --format srt meeting.wav > meeting.srt
revospeech synthesize -m vits-ms -t "Hello, world!" -o out.wav
```

Or from Python:

```python
from revospeech import ASR, TTS

print(ASR("zipformer-v2").transcribe("audio.wav").text)
TTS("vits-ms").synthesize("Hello, world!").save("out.wav")
```

## Next steps

- [Quickstart guide](quickstart.md)
- [CLI reference](cli.md)
- [Speech restoration](util.md)
- [API reference](api.md)
- [Configuration](configuration.md)
