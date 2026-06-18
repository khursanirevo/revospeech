# Extension Guide

RevoSpeech is built so most new models need **zero core code changes** — just a YAML manifest. This page covers the three common extension scenarios.

---

## 1. Add a new model (existing backend)

If the model uses an already-supported backend (`sherpa-onnx` for ASR, `revovoice`/`vits` for TTS), you only need a manifest.

=== "ASR"

    ```yaml
    # revospeech/models/asr/my_whisper.yaml
    name: my-whisper
    task: asr
    mode: local
    backend: sherpa-onnx
    model_type: whisper
    model_url: "https://huggingface.co/you/whisper-model/resolve/main/model.onnx"
    sample_rate: 16000
    language: en
    description: "Custom Whisper model"
    size_mb: 250.0
    capabilities: [word-timestamps, streaming]
    languages: [en]
    tags: [english, whisper]
    license: "MIT"
    min_ram_mb: 2048
    files:
      encoder: "encoder.onnx"
      decoder: "decoder.onnx"
      tokens: "tokens.txt"
    ```

=== "TTS"

    ```yaml
    # revospeech/models/tts/my_vits.yaml
    name: my-vits
    task: tts
    mode: local
    backend: vits
    model_url: "https://huggingface.co/you/vits/resolve/main/vits.onnx"
    sample_rate: 22050
    language: en
    description: "Custom VITS model"
    size_mb: 150.0
    capabilities: []
    languages: [en]
    tags: [english]
    license: "MIT"
    files:
      model: "vits.onnx"
      tokens: "tokens.txt"
      lexicon: "lexicon.txt"
    ```

Drop the YAML into `revospeech/models/{task}/`. The registry picks it up automatically on next import.

---

## 2. Add a new backend (local inference engine)

For a brand-new local inference engine (e.g., ONNX Runtime, PyTorch):

1. **Create engine file** at `revospeech/asr/{backend}_engine.py` (or `revospeech/tts/`):

    ```python
    from .base import BaseASR
    from .result import Transcript

    class MyBackendASR(BaseASR):
        def __init__(self, model_name: str, device: str = "auto"):
            super().__init__(model_name, device)
            # Lazy-import heavy deps so library loads without them
            from revospeech.registry import get
            manifest = get(model_name, "asr")
            # Load weights via revospeech.downloader.ensure_model(...)
            ...

        def transcribe(self, audio_path) -> Transcript:
            ...
    ```

2. **Register in factory** — edit `revospeech/asr/__init__.py`, add a branch:

    ```python
    if manifest.backend == "my-backend":
        from .my_backend_engine import MyBackendASR
        return MyBackendASR(model_name, device=device)
    ```

3. **Add manifest** in `revospeech/models/{task}/` with `backend: my-backend`.

4. **Add tests** with mocked inference.

### Engine contract

- Subclass `BaseASR` / `BaseTTS` and implement `transcribe()` / `synthesize()`.
- Lazy-import backend dependencies so the library loads without them.
- Use `ensure_model()` from `revospeech.downloader` for weights.
- Raise `RevosEngineError` (with a `suggestion=`) on inference failures.

---

## 3. Add an API backend (cloud inference)

For cloud-hosted inference (yours or a third party):

1. **Create engine** at `revospeech/asr/{provider}_engine.py`:

    ```python
    from revospeech.config import get_api_key
    from revospeech.exceptions import RevosConfigError
    from .base import BaseASR

    class MyApiASR(BaseASR):
        def __init__(self, model_name: str, api_key: str | None = None):
            super().__init__(model_name, device="api")
            self.api_key = api_key or get_api_key()
            if not self.api_key:
                raise RevosConfigError(
                    "Missing API key",
                    suggestion="Set with: revospeech config set-api-key"
                )

        def transcribe(self, audio_path) -> Transcript:
            # POST to your endpoint, map response -> Transcript
            ...
    ```

2. **Register in factory** under `manifest.is_api` branch.

3. **Manifest** with `mode: api`:

    ```yaml
    name: my-cloud-asr
    task: asr
    mode: api
    backend: my-api
    api_endpoint: "https://api.example.com/v1/asr"
    language: en
    description: "Cloud ASR via Example"
    capabilities: [word-timestamps]
    license: "proprietary"
    ```

---

## Registering a model at runtime

For models you don't want in the repo (private, experimental, etc.):

```python
import revospeech
from revospeech.registry import register, ModelManifest

manifest = ModelManifest(name="my-private", task="tts", mode="local",
                         backend="vits", files={"model": "p.onnx"})
register(manifest)
```

Now `TTS("my-private")` works.

---

## See also

- [Adding Custom Models](https://github.com/khursanirevo/revospeech#adding-custom-models) in the README
- [Manifest reference](api/registry.md)
- [BaseASR / BaseTTS API](api/asr.md)
- [`AGENTS.md`](https://github.com/khursanirevo/revospeech/blob/master/AGENTS.md) — full contributor guide
