"""Pick a specific model from the catalog."""

from revospeech import ASR, list_models

# See what's available
for model in list_models():
    print(f"  {model.name:<25} {model.task:<6} {model.status}")

# Use a specific model
asr = ASR("zipformer-v2")
result = asr.transcribe("audio/en_news_ai.wav")
print(f"\nTranscript: {result.text}")
