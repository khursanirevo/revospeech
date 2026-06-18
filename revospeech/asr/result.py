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
            f'Transcript(text="{preview}...", duration={self.duration:.1f}s, '
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

    def __repr__(self) -> str:
        status = "ok" if self.succeeded else f"error: {self.error}"
        return f"BatchResult(input={str(self.input)!r}, {status}, {self.duration:.2f}s)"


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

    def save(self, path: str | Path) -> None:
        """Save batch report to JSON file."""
        data = {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "total_duration": self.total_duration,
            "items": [],
        }
        for item in self.items:
            item_data = {
                "input": str(item.input),
                "duration": item.duration,
                "error": item.error,
            }
            if item.result is not None:
                if hasattr(item.result, "text"):
                    item_data["result"] = {
                        "text": item.result.text,
                        "type": type(item.result).__name__,
                    }
                else:
                    item_data["result"] = {"type": type(item.result).__name__}
            data["items"].append(item_data)

        Path(path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def __repr__(self) -> str:
        return (
            f"BatchReport(total={self.total}, succeeded={self.succeeded}, "
            f"failed={self.failed}, {self.total_duration:.2f}s)"
        )
