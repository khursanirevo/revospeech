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

# Apply speech restoration post-processing (Sidon)
revospeech synthesize -m vits-ms -t "Hello!" -o out.wav --restore
```

`--restore` is opt-in. When set, any ready util model tagged
`tts-postprocess` (e.g. Sidon) runs after synthesis. See
[Speech Restoration](util.md).

### restore

```bash
# Restore / enhance speech (default model: sidon)
revospeech restore -i noisy.wav -o clean.wav

# Explicit model
revospeech restore -m sidon -i noisy.wav -o clean.wav
```

Applies denoise + dereverberation + bandwidth extension. Input at any sample
rate is resampled to 16 kHz internally; output is 48 kHz. See
[Speech Restoration](util.md).

### models

```bash
revospeech models                 # list models with status
revospeech models --ready         # only ready-to-use
revospeech models --task asr
revospeech models --download <name>  # pre-download a model
revospeech models-info <name>     # detailed info
```

Downloads over 50 MB prompt for confirmation in interactive terminals;
set `REVOSPEECH_YES=1` to skip the prompt in scripts/CI.

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

## See also

- [Quickstart](quickstart.md) — 60-second getting started
- [Configuration](configuration.md) — env vars, API keys, cache
- [API reference](api/cli.md) — CLI internals
