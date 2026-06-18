"""Quickstart — ASR and TTS in 5 lines."""

from revospeech import ASR, TTS

# Transcribe audio (uses default model)
asr = ASR()
result = asr.transcribe("audio/en_news_ai.wav")
print(f"Transcript: {result.text}")

# Synthesize speech (uses default model)
tts = TTS()
audio = tts.synthesize("Hello, this is a quick start example.")
audio.save("output.wav")
print(f"Saved {audio.duration:.1f}s audio to output.wav")
