"""Batch processing — parallel transcription and synthesis.

Demonstrates:
- ASR.transcribe_batch(paths, max_workers=4) for parallel transcription
- TTS.synthesize_batch(texts, output_dir=...) for parallel synthesis
- BatchReport.save(path) for exporting results as JSON

Usage:
    python examples/06_batch_processing.py
"""

from pathlib import Path

from revospeech.asr import ASR
from revospeech.tts import TTS

# Batch transcription
asr = ASR()  # auto-selects smallest ready ASR model
audio_files = sorted(Path("audio_samples").glob("*.wav"))
if audio_files:
    report = asr.transcribe_batch([str(p) for p in audio_files], max_workers=4)
    succ, total, dur = report.succeeded, report.total, report.total_duration
    print(f"Transcribed {succ}/{total} files in {dur:.1f}s")
    report.save("transcription_report.json")
else:
    print("No .wav files in audio_samples/")

# Batch synthesis
tts = TTS()  # auto-selects smallest ready TTS model
texts = [
    "Hello, this is the first sentence.",
    "And here is the second one.",
    "Batch processing saves time when synthesizing many clips.",
]
report = tts.synthesize_batch(texts, output_dir="batch_output", max_workers=4)
succ, total, dur = report.succeeded, report.total, report.total_duration
print(f"Synthesized {succ}/{total} clips in {dur:.1f}s")
for item in report.items:
    status = "OK" if item.succeeded else f"FAIL: {item.error}"
    print(f"  [{status}] {item.input[:50]}")
