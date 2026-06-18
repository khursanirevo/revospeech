# CLI Reference

The `revospeech` CLI provides commands for transcription, synthesis, model management, and configuration.

## Global flags

```bash
revospeech --verbose <command>   # DEBUG logging
revospeech --quiet <command>     # WARNING-only logging
```

## Commands

### transcribe

```bash
# Basic
revospeech transcribe -m zipformer-v2 audio.wav

# Output formats
revospeech transcribe audio.wav --format json
revospeech transcribe audio.wav --format srt
revospeech transcribe audio.wav --format vtt

# Multi-file batch
revospeech transcribe -m zipformer-v2 a.wav b.wav c.wav
```

### synthesize

```bash
# Quick text
revospeech synthesize -m revovoice -t "Hello!" -o out.wav

# From text file
revospeech synthesize -m revovoice -f script.txt -o book.wav

# Batch from file list (one text per line)
revospeech synthesize -m revovoice --file-list inputs.txt -o output/
```

### models

```bash
revospeech models                 # list models with status
revospeech models --ready         # only ready-to-use
revospeech models --task asr
revospeech models --download <name>  # pre-download a model
revospeech models-info <name>     # detailed info
```

### catalog

```bash
revospeech catalog list           # browse remote
revospeech catalog search <query> --task asr --language en
revospeech catalog recommend      # top 3 recommendations
revospeech catalog pull <name>    # install a model
```

### config

```bash
revospeech config set-api-key     # interactive prompt
revospeech config show-api-key    # shows masked key
```

### setup

```bash
revospeech setup                  # interactive first-run wizard
```

### info

```bash
revospeech info                   # version, device, cache size
```

### search

```bash
revospeech search "english fast"  # fuzzy model search
```
