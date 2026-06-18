# RevoSpeech

A unified Python library for speech AI — ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) using open models.

## Features

- **Zero-config defaults**: `ASR()` and `TTS()` auto-select a ready model
- **Local or API**: run models locally, or use revolab cloud API
- **Auto-download**: models fetch themselves on first use
- **BytesIO support**: in-memory pipelines without disk I/O
- **Voice cloning**: zero-shot with reference audio
- **Batch processing**: parallel transcription and synthesis
- **Typed**: full type hints + py.typed marker
- **Extensible**: add custom backends via subclassing

## 10-second quickstart

```bash
pip install revospeech
```

```python
from revospeech import ASR, TTS

result = ASR().transcribe("audio.wav")
print(result.text)

TTS().synthesize("Hello, world!").save("out.wav")
```

## Next steps

- [Quickstart guide](quickstart.md)
- [CLI reference](cli.md)
- [API reference](api.md)
- [Configuration](configuration.md)
