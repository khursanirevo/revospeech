"""ASR result data classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Segment:
    """A single transcription segment with timing and confidence."""

    start: float
    end: float
    text: str
    confidence: float


@dataclass
class Transcript:
    """Full transcription result from ASR."""

    text: str
    segments: list[Segment]
    language: str


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
