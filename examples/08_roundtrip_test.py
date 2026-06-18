"""Round-trip validation: TTS generates audio, ASR transcribes it back.

Useful for:
- Self-validating TTS quality (does ASR understand the synthesized speech?)
- Regression testing when changing TTS models/parameters
- Voice cloning quality benchmarking

Usage:
    python examples/08_roundtrip_test.py
"""

from difflib import SequenceMatcher

from revospeech.asr import ASR
from revospeech.tts import TTS


def fuzzy_match(actual: str, expected: str) -> float:
    """Return similarity ratio in [0, 1]."""
    return SequenceMatcher(
        None, actual.lower().strip(), expected.lower().strip()
    ).ratio()


# Synthesize a known sentence
tts = TTS()
expected = "The quick brown fox jumps over the lazy dog."
audio = tts.synthesize(expected, "roundtrip_input.wav")
print(f"Synthesized: {audio.duration:.2f}s")

# Transcribe the synthesized audio back
asr = ASR()
result = asr.transcribe("roundtrip_input.wav")
actual = result.text
print(f"Transcribed: {actual!r}")

# Score similarity
score = fuzzy_match(actual, expected)
print(f"Similarity: {score:.1%}")
if score >= 0.85:
    print("PASS: round-trip quality is good")
elif score >= 0.6:
    print("WARN: round-trip quality is acceptable but not great")
else:
    print("FAIL: TTS or ASR needs investigation")
