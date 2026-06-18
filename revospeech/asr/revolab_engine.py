"""Revolab cloud API backend for ASR.

SKELETON: actual endpoint paths and response schemas pending
API contract finalization. The infrastructure (auth, retries, error
mapping) is in place via RevolabClient.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from revospeech.asr.base import BaseASR
from revospeech.asr.result import Segment, Transcript
from revospeech.exceptions import RevosEngineError
from revospeech.http_client import RevolabClient

logger = logging.getLogger(__name__)


class RevolabASR(BaseASR):
    """ASR via revolab cloud API.

    Authenticates with API key (constructor or env var REVOLAB_API_KEY).
    Audio is uploaded to the API endpoint for transcription.
    """

    def __init__(self, model_name: str, device: str = "auto") -> None:
        super().__init__(model_name, device)
        from revospeech.registry import get

        try:
            manifest = get(model_name, "asr")
            endpoint = manifest.api_endpoint or "https://api.revolab.ai/v1"
        except KeyError:
            endpoint = "https://api.revolab.ai/v1"

        self._client = RevolabClient(endpoint=endpoint)

    def transcribe(
        self,
        audio_path: str | Path | Any,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> Transcript:
        """Transcribe audio via revolab API.

        SKELETON: endpoint path pending contract finalization.
        """
        audio_str = audio_path if isinstance(audio_path, str) else str(audio_path)

        logger.info("Uploading %s to revolab ASR API", audio_str)

        with open(audio_str, "rb") as f:
            files = {"audio": (Path(audio_str).name, f, "audio/wav")}
            data = {
                "language": language or "",
                "word_timestamps": str(word_timestamps).lower(),
            }
            try:
                response = self._client.post(
                    "/asr/transcribe",  # TODO: confirm actual path
                    files=files,
                    data=data,
                )
            except RevosEngineError:
                raise
            except Exception as e:
                raise RevosEngineError(
                    f"ASR API call failed: {e}",
                    suggestion="Check network connectivity and API status.",
                ) from e

        return self._parse_response(response)

    def _parse_response(self, response: dict) -> Transcript:
        """Parse API response into Transcript.

        SKELETON: schema pending contract.
        """
        # TODO: replace with actual schema once contract is finalized
        text = response.get("text", "")
        segments_data = response.get("segments", [])
        segments = [
            Segment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", ""),
                confidence=seg.get("confidence", 0.0),
            )
            for seg in segments_data
        ]
        return Transcript(
            text=text,
            segments=segments,
            language=response.get("language", "en"),
        )

    def close(self) -> None:
        self._client.close()
