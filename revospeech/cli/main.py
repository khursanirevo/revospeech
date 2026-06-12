"""RevoS CLI — Command-line interface for speech AI.

Usage:
    revos transcribe --model zipformer-v2 audio.wav
    revos synthesize --model revovoice --text "Hello" -o output.wav
"""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group()
@click.version_option()
def cli() -> None:
    """RevoS — A unified library for speech AI (ASR & TTS)."""


@cli.command()
@click.option("--model", "-m", required=True, help="ASR model name (e.g. zipformer-v2)")
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--srt", "as_srt", is_flag=True, help="Output as SRT subtitles")
def transcribe(model: str, audio_path: str, as_json: bool, as_srt: bool) -> None:
    """Transcribe an audio file to text."""
    from revospeech.asr import ASR
    from revospeech.exceptions import (
        RevosAudioError,
        RevosConfigError,
        RevosEngineError,
        RevosError,
        RevosModelError,
    )

    try:
        asr = ASR(model)
        result = asr.transcribe(audio_path)

        if as_json:
            data = {
                "text": result.text,
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "confidence": seg.confidence,
                    }
                    for seg in result.segments
                ],
                "language": result.language,
            }
            click.echo(json.dumps(data, indent=2, ensure_ascii=False))
        elif as_srt:
            for i, seg in enumerate(result.segments, 1):
                start_ts = _format_srt_time(seg.start)
                end_ts = _format_srt_time(seg.end)
                click.echo(f"{i}")
                click.echo(f"{start_ts} --> {end_ts}")
                click.echo(seg.text)
                click.echo()
        else:
            click.echo(result.text)
    except RevosConfigError as e:
        click.echo(f"Configuration error: {e}", err=True)
        raise SystemExit(1)
    except RevosModelError as e:
        click.echo(f"Model error: {e}", err=True)
        raise SystemExit(1)
    except RevosEngineError as e:
        click.echo(f"Engine error: {e}", err=True)
        raise SystemExit(1)
    except RevosAudioError as e:
        click.echo(f"Audio error: {e}", err=True)
        raise SystemExit(1)
    except RevosError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option("--model", "-m", required=True, help="TTS model name (e.g. revovoice)")
@click.option("--text", "-t", help="Text to synthesize")
@click.option(
    "--file", "-f", type=click.Path(exists=True), help="Text file to synthesize"
)
@click.option(
    "--output", "-o", required=True, type=click.Path(), help="Output audio path"
)
@click.option("--speed", default=1.0, help="Speech speed (default: 1.0)")
@click.option(
    "--ref-audio",
    type=click.Path(exists=True),
    help="Reference audio for voice cloning",
)
@click.option("--ref-text", help="Transcription of reference audio")
def synthesize(
    model: str,
    text: str | None,
    file: str | None,
    output: str,
    speed: float,
    ref_audio: str | None,
    ref_text: str | None,
) -> None:
    """Synthesize speech from text."""
    from revospeech.exceptions import (
        RevosAudioError,
        RevosConfigError,
        RevosEngineError,
        RevosError,
        RevosModelError,
    )
    from revospeech.tts import TTS

    if text is None and file is None:
        raise click.UsageError("Either --text or --file must be provided")

    try:
        if text is None and file is not None:
            with open(file) as f:
                text = f.read().strip()

        assert text is not None

        tts = TTS(model)

        # Auto-detect long text and use synthesize_long
        if len(text) > 500:
            audio = tts.synthesize_long(
                text,
                output,
                speed=speed,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
        else:
            audio = tts.synthesize(
                text,
                output,
                speed=speed,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
        click.echo(
            f"Saved {len(audio.samples)} samples "
            f"({len(audio.samples) / audio.sample_rate:.1f}s) to {output}"
        )
    except RevosConfigError as e:
        click.echo(f"Configuration error: {e}", err=True)
        raise SystemExit(1)
    except RevosModelError as e:
        click.echo(f"Model error: {e}", err=True)
        raise SystemExit(1)
    except RevosEngineError as e:
        click.echo(f"Engine error: {e}", err=True)
        raise SystemExit(1)
    except RevosAudioError as e:
        click.echo(f"Audio error: {e}", err=True)
        raise SystemExit(1)
    except RevosError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option("--task", "-t", help="Filter by task (asr/tts)")
@click.option("--mode", "-m", help="Filter by mode (local/api)")
@click.option("--status", "-s", "status_filter", help="Filter by status")
@click.option("--ready", is_flag=True, help="Show only ready models")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def models(
    task: str | None,
    mode: str | None,
    status_filter: str | None,
    ready: bool,
    as_json: bool,
) -> None:
    """List available models."""
    from revospeech.registry.status import list_model_statuses

    kwargs: dict = {}
    if task:
        kwargs["task"] = task
    if mode:
        kwargs["mode"] = mode
    if status_filter:
        kwargs["status"] = status_filter
    if ready:
        kwargs["status"] = "ready"

    model_list = list_model_statuses(**kwargs)

    if not model_list:
        click.echo("No models found.")
        return

    if as_json:
        click.echo(
            json.dumps(
                [
                    {
                        "name": m.name,
                        "task": m.task,
                        "mode": m.mode,
                        "status": m.status,
                        "size_mb": m.size_mb,
                        "capabilities": m.capabilities,
                        "languages": m.languages,
                    }
                    for m in model_list
                ],
                indent=2,
            )
        )
        return

    # Status icons
    status_icon = {"ready": "✓", "needs-download": "↓", "needs-api-key": "✗"}

    # Table header
    click.echo(
        f"{'NAME':<20} {'TASK':<6} {'MODE':<7} {'STATUS':<16} {'SIZE':>8}  "
        f"{'LANG':<15} {'CAPABILITIES'}"
    )
    click.echo("-" * 90)

    for m in model_list:
        icon = status_icon.get(m.status, "?")
        size = f"{m.size_mb:.0f} MB" if m.size_mb else "—"
        langs = ",".join(m.languages[:3]) if m.languages else "—"
        caps = ",".join(m.capabilities[:3]) if m.capabilities else "—"
        status_text = f"{icon} {m.status}"
        click.echo(
            f"{m.name:<20} {m.task:<6} {m.mode:<7} {status_text:<16} "
            f"{size:>8}  {langs:<15} {caps}"
        )


@cli.command("models-info")
@click.argument("name")
def models_info(name: str) -> None:
    """Show detailed info for a model."""
    from revospeech.registry.status import check_model

    try:
        m = check_model(name)
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    status_icon = {"ready": "✓", "needs-download": "↓", "needs-api-key": "✗"}
    icon = status_icon.get(m.status, "?")
    size = f"{m.size_mb:.0f} MB" if m.size_mb else "—"
    langs = ", ".join(m.languages) if m.languages else "—"
    caps = ", ".join(m.capabilities) if m.capabilities else "—"

    click.echo(f"  Model:        {m.name}")
    click.echo(f"  Task:         {m.task}")
    click.echo(f"  Mode:         {m.mode}")
    click.echo(f"  Status:       {icon} {m.status}")
    click.echo(f"  Size:         {size}")
    click.echo(f"  Languages:    {langs}")
    click.echo(f"  Capabilities: {caps}")


@cli.command()
@click.argument("query")
def search(query: str) -> None:
    """Search models by name, tag, or language."""
    import revospeech

    results = revospeech.search_models(query)
    if not results:
        click.echo(f"No models matching '{query}'.")
        return

    status_icon = {"ready": "✓", "needs-download": "↓", "needs-api-key": "✗"}
    for m in results:
        icon = status_icon.get(m.status, "?")
        size = f"{m.size_mb:.0f} MB" if m.size_mb else "—"
        click.echo(
            f"  {icon} {m.name:<20} {m.task:<6} {m.mode:<7} {m.status:<16} {size}"
        )


@cli.command()
def info() -> None:
    """Show environment and configuration info."""
    import sys

    click.echo(f"RevoS version:   {_get_version()}")
    click.echo(f"Python:          {sys.version.split()[0]}")

    # Device
    from revospeech.device import auto_detect_device

    click.echo(f"Device:          {auto_detect_device()}")

    # Models
    from revospeech.registry import list_models

    click.echo(f"Models loaded:   {len(list_models())}")

    # Cache dir
    cache_dir = Path.home() / ".cache" / "revos"
    click.echo(f"Cache dir:       {cache_dir}")

    # Catalog repo
    from revospeech.catalog import get_catalog_repo

    click.echo(f"Catalog repo:    {get_catalog_repo()}")

    # HF auth
    try:
        from huggingface_hub import HfApi

        user = HfApi().whoami()
        click.echo(f"HuggingFace:     {user.get('name', 'unknown')}")
    except Exception:
        click.echo("HuggingFace:     not logged in")


@cli.group()
def catalog() -> None:
    """Browse and pull models from the remote catalog."""


@catalog.command("list")
@click.option("--task", "-t", help="Filter by task type (asr or tts)")
def catalog_list(task: str | None) -> None:
    """List models available in the remote catalog."""
    from revospeech.catalog import get_catalog_repo, list_catalog

    click.echo(f"Fetching catalog from {get_catalog_repo()}...")
    try:
        results = list_catalog(task)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    if not results:
        click.echo("No models found in catalog.")
        return

    click.echo(
        f"{'Name':<20} {'Task':<6} {'Backend':<15} "
        f"{'Language':<12} {'Version':<12}"
    )
    click.echo("-" * 65)
    for m in results:
        rev = m.revision or "latest"
        click.echo(
            f"{m.name:<20} {m.task:<6} {m.backend:<15} "
            f"{m.language:<12} {rev:<12}"
        )
    click.echo("\nUse 'revos catalog pull <name>' to install.")


@catalog.command("pull")
@click.argument("model_name")
def catalog_pull(model_name: str) -> None:
    """Pull a model from the catalog and install it locally."""
    from revospeech.catalog import get_catalog_repo, pull_model

    click.echo(f"Pulling '{model_name}' from {get_catalog_repo()}...")
    try:
        dest = pull_model(model_name)
    except (KeyError, RuntimeError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    click.echo(f"Installed to {dest}")
    click.echo(f"Use: from revospeech.tts import TTS; TTS('{model_name}')")


def _get_version() -> str:
    """Get revos version without triggering heavy imports."""
    from importlib.metadata import version

    try:
        return version("revos")
    except Exception:
        return "unknown"


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


if __name__ == "__main__":
    cli()
