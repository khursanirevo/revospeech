#!/usr/bin/env bash
# RevoSpeech CLI-only workflow — no Python code needed.
#
# This script demonstrates common tasks using just the `revospeech` CLI:
#   - List available models
#   - Download a model
#   - Transcribe an audio file
#   - Synthesize text to speech
#   - Browse the remote catalog
#   - Set up an API key
#
# Usage: bash examples/12_cli_only.sh

set -euo pipefail

echo "=== revospeech info ==="
revospeech info

echo
echo "=== Available models ==="
revospeech models

echo
echo "=== Ready-to-use models only ==="
revospeech models --ready

echo
echo "=== Search for English models ==="
revospeech search "english"

echo
echo "=== Remote catalog ==="
revospeech catalog list

echo
echo "=== Setup wizard ==="
# Uncomment to run interactively:
# revospeech setup

echo
echo "=== Transcription examples ==="
# JSON output:
# revospeech transcribe -m zipformer-v2 --format json audio.wav

# SRT subtitles:
# revospeech transcribe -m zipformer-v2 --format srt audio.wav > subtitles.srt

# Multi-file batch:
# revospeech transcribe -m zipformer-v2 audio1.wav audio2.wav audio3.wav

echo
echo "=== Synthesis examples ==="
# Quick text:
# revospeech synthesize -m revovoice -t "Hello, world!" -o hello.wav

# Long text from file:
# revospeech synthesize -m revovoice -f script.txt -o audiobook.wav

# Batch from text file (one text per line):
# revospeech synthesize -m revovoice --file-list inputs.txt -o output_dir/

echo
echo "=== Config ==="
# revospeech config set-api-key    # interactive
# revospeech config show-api-key

echo "Done. See README.md for more examples."
