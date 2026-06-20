"""Revolab cloud API backend for ASR."""

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

        Sends raw audio bytes to POST /recognize with Content-Type: audio/wav.
        """
        audio_str = audio_path if isinstance(audio_path, str) else str(audio_path)

        logger.info("Uploading %s to revolab ASR API", audio_str)

        try:
            with open(audio_str, "rb") as f:
                audio_bytes = f.read()
        except OSError as e:
            raise RevosEngineError(
                f"Cannot read audio file: {audio_str}: {e}",
                suggestion="Check the file path and permissions.",
            ) from e

        try:
            response = self._client.post(
                "/recognize",
                content=audio_bytes,
                headers={"Content-Type": "audio/wav"},
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

        Response: status, text, session_id, duration_s, confidence, rtf.
        """
        status = response.get("status", "")
        if status and status != "success":
            raise RevosEngineError(
                f"ASR API returned non-success status: {status}",
                suggestion="Retry the request or contact revolab support.",
            )

        text = response.get("text", "")
        duration = response.get("duration_s", 0.0)
        confidence = response.get("confidence", 0.0)

        segment = Segment(
            start=0.0,
            end=duration,
            text=text,
            confidence=confidence,
        )
        return Transcript(
            text=text,
            segments=[segment],
            language=response.get("language", ""),
        )

    def close(self) -> None:
        self._client.close()
