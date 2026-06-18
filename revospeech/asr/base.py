"""Abstract base class for ASR engines."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .result import BatchReport, BatchResult, Transcript


class BaseASR(ABC):
    """Base class for automatic speech recognition engines."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device

    def stream_transcribe(self, audio_path: str, **kwargs):
        raise NotImplementedError(
            "This engine does not support streaming transcription"
        )

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> Transcript: ...

    def transcribe_batch(
        self,
        paths: list[str | Path],
        max_workers: int = 4,
        on_error: str = "continue",
    ) -> BatchReport:
        """Transcribe multiple audio files in parallel.

        Args:
            paths: List of audio file paths.
            max_workers: Number of parallel threads.
            on_error: "continue" (skip failures) or "raise" (fail fast).

        Returns:
            BatchReport with per-item results and summary.
        """
        results: dict[str, BatchResult] = {}

        def _process(path: str | Path) -> tuple[str, BatchResult]:
            t0 = time.perf_counter()
            try:
                transcript = self.transcribe(str(path))
                elapsed = time.perf_counter() - t0
                return str(path), BatchResult(
                    input=path, result=transcript, duration=elapsed
                )
            except Exception as e:
                elapsed = time.perf_counter() - t0
                return str(path), BatchResult(
                    input=path, error=str(e), duration=elapsed
                )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process, p): p for p in paths}

            for future in as_completed(futures):
                key, result = future.result()
                results[key] = result

                if on_error == "raise" and result.error is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(
                        f"Batch transcription failed for {key}: {result.error}"
                    )

        ordered = [results[str(p)] for p in paths if str(p) in results]
        return BatchReport(
            items=ordered,
            total=len(paths),
            succeeded=sum(1 for r in ordered if r.succeeded),
            failed=sum(1 for r in ordered if not r.succeeded),
            total_duration=sum(r.duration for r in ordered),
        )
