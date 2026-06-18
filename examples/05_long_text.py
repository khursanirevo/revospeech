"""Long text synthesis — auto-splits into chunks."""

from revospeech import TTS

tts = TTS()

long_text = """
RevoSpeech is a unified Python library for speech AI.
It supports both automatic speech recognition and text-to-speech.
The library is designed to be easy to use, with sensible defaults.
You can run models locally or via the revolab cloud API.
"""

audio = tts.synthesize_long(long_text)
audio.save("long_output.wav")
print(f"Generated {audio.duration:.1f}s audio from {len(long_text)} chars of text")
