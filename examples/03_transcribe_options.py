"""ASR with options — word timestamps and JSON export."""

from revospeech import ASR

asr = ASR("zipformer-v2")
result = asr.transcribe("audio/en_news_ai.wav")

# Access word-level timestamps
for seg in result.segments:
    print(f"  [{seg.start:.2f} - {seg.end:.2f}] {seg.text}")

# Save as different formats
result.save("transcript.txt")
result.save("transcript.json")
result.save("transcript.srt")
result.save("transcript.vtt")
print("Saved transcript in txt, json, srt, vtt formats")
