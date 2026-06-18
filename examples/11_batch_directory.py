"""Process a whole directory of audio files in batch.

Walks a directory, finds all .wav files, and transcribes them in parallel
using ASR.transcribe_batch(). Saves a JSON report.

Usage:
    python examples/11_batch_directory.py /path/to/audio_files
"""

import sys
from pathlib import Path

from revospeech.asr import ASR

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <audio_dir>")
    sys.exit(1)

audio_dir = Path(sys.argv[1])
if not audio_dir.is_dir():
    print(f"Not a directory: {audio_dir}")
    sys.exit(1)

audio_files = sorted(list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.flac")))
if not audio_files:
    print(f"No .wav or .flac files in {audio_dir}")
    sys.exit(0)

print(f"Found {len(audio_files)} audio files")

asr = ASR()
report = asr.transcribe_batch([str(p) for p in audio_files], max_workers=4)

print("\nResults:")
print(f"  Succeeded: {report.succeeded}/{report.total}")
print(f"  Failed:    {report.failed}/{report.total}")
print(f"  Duration:  {report.total_duration:.1f}s")

report.save("batch_directory_report.json")
print("\nReport saved to batch_directory_report.json")

for item in report.items[:5]:
    if item.succeeded:
        print(f"  OK   {Path(item.input).name}: {item.result.text[:60]!r}")
    else:
        print(f"  FAIL {Path(item.input).name}: {item.error}")

if len(report.items) > 5:
    print(f"  ... ({len(report.items) - 5} more)")
