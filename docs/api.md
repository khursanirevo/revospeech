# API Reference

Complete API reference organized by area. Click into each section for details.

## Areas

- [ASR (Speech Recognition)](api/asr.md) — `ASR()`, `BaseASR`, streaming, language detection
- [TTS (Speech Synthesis)](api/tts.md) — `TTS()`, `BaseTTS`, streaming, voice listing
- [Registry](api/registry.md) — Model manifests, listing, search, status
- [CLI](api/cli.md) — Command-line interface commands and flags
- [Result Types](api/results.md) — `Transcript`, `Audio`, `BatchReport`, `Segment`

## Quick API lookup

| What | Where |
|------|-------|
| Transcribe audio | `revospeech.ASR().transcribe(path)` |
| Synthesize speech | `revospeech.TTS().synthesize(text)` |
| List models | `revospeech.list_models()` |
| Search models | `revospeech.search_models(query)` |
| Check model status | `revospeech.check_model(name)` |
| Save transcript | `Transcript.save('out.srt')` |
| Play audio | `Audio.play()` |
