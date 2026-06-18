# Troubleshooting

## Model download fails

```bash
revospeech models-info <model-name>
```

Check status, then retry:

```bash
revospeech models --download <name>
```

## HuggingFace gated model

Some models (revovoice) are gated. You need:

1. Login: `huggingface-cli login`
2. Request access at the model's HF page
3. Wait for approval

Then:

```python
from revospeech.tts import TTS
tts = TTS('revovoice')  # will auto-download
```

## Missing sounddevice

`Audio.play()` requires `sounddevice`:

```bash
pip install sounddevice
```

## Slow first transcription

First run downloads the model (50-500MB). Subsequent runs use the cached version.

## Out of memory

Use a smaller model:

```bash
revospeech catalog recommend --task asr
```

Or run on CPU:

```python
asr = ASR("model-name", device="cpu")
```

## Bad transcription quality

- Check audio sample rate (most models want 16kHz mono)
- Use a model matching your language
- Check audio quality (background noise hurts accuracy)

## ONNX runtime conflicts

`revospeech[gpu]` and `revospeech[all]` install `onnxruntime-gpu`, which conflicts with `onnxruntime`. Uninstall first:

```bash
pip uninstall onnxruntime
pip install revospeech[gpu]
```

## Getting help

- [Open an issue](https://github.com/khursanirevo/revospeech/issues)
- [Email security reports to sani@khursani.dev](mailto:sani@khursani.dev)
