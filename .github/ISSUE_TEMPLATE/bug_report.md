---
name: Bug report
about: Report a bug to help us improve
title: "[BUG] "
labels: bug
assignees: ''
---

## Describe the bug

A clear and concise description of what the bug is.

## To reproduce

Steps to reproduce the behavior:

1. Run `...`
2. With input `...`
3. See error

If possible, include a minimal code snippet or command:

```bash
revospeech asr --model revovoice --input audio.wav
```

## Expected behavior

A clear and concise description of what you expected to happen.

## Actual behavior

A clear and concise description of what actually happened (error message,
wrong output, crash, hang, etc.).

## Environment

Please fill in the following:

- **OS**: [e.g. Ubuntu 22.04, macOS 14.2, Windows 11]
- **Python**: [e.g. 3.12.1]
- **revospeech version**: [output of `revospeech info`]
- **Models used**: [e.g. revovoice, zipformer-v2, piper-en_US-amy-medium]
- **Installation method**: [pip / uv / from source]
- **Hardware (if relevant)**: [CPU-only / GPU model / amount of RAM]

## Logs

Paste the relevant output from the terminal. Running with `--verbose` produces
detailed logs that are very helpful for debugging:

```text
$ revospeech --verbose ...
```

If the log is long, paste only the relevant portion or attach the full log as a
file. Please **do not** include sensitive information.

## Additional context

Add any other context about the problem here:

- Screenshots (if applicable)
- Related issues (#NNN)
- Whether this is a regression (worked in version X, broken in version Y)
- Any workaround you have found
