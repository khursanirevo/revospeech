"""Voice cloning with RevoVoice TTS."""

from revospeech import TTS

tts = TTS("revovoice")

# Clone a voice from a reference audio sample
audio = tts.synthesize(
    "This is a cloned voice speaking new words.",
    ref_audio="audio/en_greeting_sarah.wav",
    ref_text="Hello, how are you doing today?",
)
audio.save("cloned_voice.wav")
print(f"Generated {audio.duration:.1f}s of cloned voice audio")
