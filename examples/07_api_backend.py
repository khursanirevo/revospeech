"""Using revolab cloud API backend (when available).

Demonstrates the API key setup pattern. The API backend is not yet
implemented in this library — this example shows the future API surface
and how users will switch between local and API models.

Usage:
    python examples/07_api_backend.py
"""

from revospeech import set_api_key
from revospeech.asr import ASR
from revospeech.tts import TTS

# API key resolution order:
#   1. Constructor arg: ASR(..., api_key="rv-...")
#   2. Environment variable: export REVOLAB_API_KEY=rv-...
#   3. Config file: ~/.config/revospeech/config.yaml
#                 (set via: revospeech config set-api-key)

# Option A: pass key in constructor
try:
    asr = ASR("revolab-asr-v1", api_key="rv-your-key-here")
    print(f"ASR engine: {asr.model_name}")
except NotImplementedError as e:
    print(f"API backend not yet available: {e}")
except Exception as e:
    print(f"Could not init API ASR: {e}")

# Option B: set globally first, then construct
set_api_key("rv-your-key-here")
try:
    tts = TTS("revolab-tts-v1")
    print(f"TTS engine: {tts.model_name}")
except NotImplementedError as e:
    print(f"API backend not yet available: {e}")
except Exception as e:
    print(f"Could not init API TTS: {e}")

# Note: code stays identical whether model is local or API.
# Same .transcribe() / .synthesize() interface either way.
print("\nWhen the API backend ships, swapping 'revovoice' -> 'revolab-tts-v1'")
print("is the only change needed to move from local to cloud.")
