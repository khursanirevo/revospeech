"""Round-trip tests: TTS -> ASR self-validation.

These tests are slow and require real models. They are skipped by default.
Run them with:
    pytest --run-roundtrip tests/test_roundtrip.py
or:
    REVOS_RUN_ROUNDTRIP=1 pytest tests/test_roundtrip.py

Useful for:
- Tracking TTS/ASR quality changes over time
- Verifying voice cloning works end-to-end
- Detecting regressions when models are updated
"""

from __future__ import annotations

from difflib import SequenceMatcher

import pytest

# Marker for slow round-trip tests requiring real model weights.
ROUNDTRIP = pytest.mark.roundtrip


def fuzzy_match(actual: str, expected: str) -> float:
    """Return similarity ratio in [0, 1] using :class:`difflib.SequenceMatcher`.

    Inputs are lowercased and whitespace-normalized before comparison.
    """
    a = " ".join(actual.lower().split())
    e = " ".join(expected.lower().split())
    return SequenceMatcher(None, a, e).ratio()


# Test cases: (text, min_similarity_threshold)
ROUNDTRIP_CASES = [
    pytest.param(
        "Hello world.",
        0.85,
        id="simple-english",
    ),
    pytest.param(
        "The quick brown fox jumps over the lazy dog.",
        0.80,
        id="sentence-english",
    ),
    pytest.param(
        "Pack my box with five dozen liquor jugs.",
        0.80,
        id="pangram",
    ),
    pytest.param(
        "One. Two. Three. Four. Five.",
        0.70,
        id="numbers",
    ),
    pytest.param(
        "The weather is nice today. I think I'll go for a walk in the park.",
        0.75,
        id="paragraph",
    ),
]


@ROUNDTRIP
@pytest.mark.parametrize("text,min_score", ROUNDTRIP_CASES)
def test_roundtrip_english(text: str, min_score: float, tmp_path):
    """TTS generates audio, ASR transcribes it back; verify similarity."""
    from revospeech.asr import ASR
    from revospeech.tts import TTS

    # Synthesize
    tts = TTS()
    audio_path = tmp_path / "tts_output.wav"
    tts.synthesize(text, str(audio_path))

    # Transcribe back
    asr = ASR()
    result = asr.transcribe(str(audio_path))

    # Check similarity
    score = fuzzy_match(result.text, text)
    print(f"\nExpected: {text!r}")
    print(f"Actual:   {result.text!r}")
    print(f"Score:    {score:.1%} (min: {min_score:.0%})")
    assert score >= min_score, (
        f"Round-trip score {score:.1%} below threshold {min_score:.0%}"
    )


@ROUNDTRIP
def test_roundtrip_voice_cloning(tmp_path):
    """Voice cloning quality: synthesize with ref_audio, transcribe back."""
    from revospeech.asr import ASR
    from revospeech.tts import TTS

    # First generate a "reference" audio
    tts = TTS()
    ref_text = "This is a reference sample for voice cloning."
    ref_path = tmp_path / "ref.wav"
    tts.synthesize(ref_text, str(ref_path))

    # Now synthesize new text using the reference voice
    target_text = "Voice cloning should preserve intelligibility."
    out_path = tmp_path / "cloned.wav"
    tts.synthesize(
        target_text,
        str(out_path),
        ref_audio=str(ref_path),
        ref_text=ref_text,
    )

    # Transcribe back
    asr = ASR()
    result = asr.transcribe(str(out_path))
    score = fuzzy_match(result.text, target_text)
    print(f"\nCloned voice round-trip score: {score:.1%}")
    assert score >= 0.70, f"Voice cloning round-trip too low: {score:.1%}"


@ROUNDTRIP
def test_roundtrip_multilingual(tmp_path):
    """TTS and ASR should handle non-English text (if model supports it)."""
    from revospeech.asr import ASR
    from revospeech.tts import TTS

    # Use a multilingual-friendly sentence
    text = "Bonjour le monde."
    tts = TTS()
    audio_path = tmp_path / "multi.wav"

    try:
        tts.synthesize(text, str(audio_path))
    except Exception as e:
        pytest.skip(f"TTS model doesn't support this language: {e}")

    asr = ASR()
    result = asr.transcribe(str(audio_path))
    # Multilingual round-trip is harder, lower threshold
    score = fuzzy_match(result.text, text)
    print(f"\nMultilingual round-trip score: {score:.1%}")
    assert score >= 0.50, f"Multilingual round-trip too low: {score:.1%}"


def test_fuzzy_match_unit():
    """Unit test for the fuzzy_match helper (always runs, no model needed)."""
    assert fuzzy_match("hello world", "hello world") == 1.0
    assert fuzzy_match("Hello World", "hello world") == 1.0  # case insensitive
    assert fuzzy_match("  hello  world  ", "hello world") == 1.0  # whitespace
    assert fuzzy_match("hello", "world") < 0.5
    assert 0.5 < fuzzy_match("hello wrld", "hello world") < 1.0  # typo
