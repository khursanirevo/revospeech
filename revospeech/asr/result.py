"""ASR result data classes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Segment:
    """A single transcription segment with timing and confidence."""

    start: float
    end: float
    text: str
    confidence: float


def _format_timestamp(seconds: float, separator: str = ",") -> str:
    """Format seconds as HH:MM:SS{separator}mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


@dataclass
class Transcript:
    """Full transcription result from ASR."""

    text: str
    segments: list[Segment]
    language: str

    @property
    def duration(self) -> float:
        """Total transcript duration in seconds based on segment timings."""
        if not self.segments:
            return 0.0
        return max(seg.end for seg in self.segments) - min(
            seg.start for seg in self.segments
        )

    def save(self, path: str | Path) -> None:
        """Save transcript to a file.

        Format is determined by file extension:
        - .txt: plain text
        - .json: structured JSON with text, segments, language
        - .srt: SubRip subtitle format
        - .vtt: WebVTT format

        Args:
            path: Output file path.
        """
        p = Path(path)
        ext = p.suffix.lower()

        if ext == ".txt":
            p.write_text(self.text, encoding="utf-8")
        elif ext == ".json":
            payload = {
                "text": self.text,
                "segments": [asdict(seg) for seg in self.segments],
                "language": self.language,
            }
            p.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        elif ext == ".srt":
            lines: list[str] = []
            for i, seg in enumerate(self.segments, start=1):
                start = _format_timestamp(seg.start, separator=",")
                end = _format_timestamp(seg.end, separator=",")
                lines.append(f"{i}")
                lines.append(f"{start} --> {end}")
                lines.append(seg.text.strip())
                lines.append("")
            p.write_text("\n".join(lines), encoding="utf-8")
        elif ext == ".vtt":
            lines = ["WEBVTT", ""]
            for seg in self.segments:
                start = _format_timestamp(seg.start, separator=".")
                end = _format_timestamp(seg.end, separator=".")
                lines.append(f"{start} --> {end}")
                lines.append(seg.text.strip())
                lines.append("")
            p.write_text("\n".join(lines), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported transcript format: {ext}")

    def __repr__(self) -> str:
        preview = self.text[:40]
        return (
            f"Transcript(text=\"{preview}...\", duration={self.duration:.1f}s, "
            f"segments={len(self.segments)})"
        )


@dataclass
class BatchResult:
    """Result for a single item in a batch."""

    input: str | Path
    result: Transcript | None = None
    error: str | None = None
    duration: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass
class BatchReport:
    """Aggregated results from a batch transcription."""

    items: list[BatchResult] = field(default_factory=list)
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    total_duration: float = 0.0

    @property
    def results(self) -> list[Transcript]:
        return [item.result for item in self.items if item.result is not None]

    @property
    def errors(self) -> list[BatchResult]:
        return [item for item in self.items if item.error is not None]
