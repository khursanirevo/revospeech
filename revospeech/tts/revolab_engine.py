"""Revolab cloud API backend for TTS.

SKELETON: actual endpoint paths and response schemas pending
API contract finalization.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from revospeech.exceptions import RevosEngineError
from revospeech.http_client import RevolabClient
from revospeech.tts.base import BaseTTS
from revospeech.tts.result import Audio

logger = logging.getLogger(__name__)


class RevolabTTS(BaseTTS):
    """TTS via revolab cloud API."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        super().__init__(model_name, device)
        from revospeech.registry import get

        try:
            manifest = get(model_name, "tts")
            endpoint = manifest.api_endpoint or "https://api.revolab.ai/v1"
        except KeyError:
            endpoint = "https://api.revolab.ai/v1"

        self._client = RevolabClient(endpoint=endpoint)

    def synthesize(
        self,
        text: str,
        output_path: str | Path | None = None,
        *,
        speed: float = 1.0,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> Audio:
        """Synthesize speech via revolab API.

        SKELETON: endpoint path pending contract finalization.
        """
        data: dict[str, Any] = {
            "text": text,
            "speed": str(speed),
        }
        files: dict[str, Any] = {}

        if ref_audio:
            with open(ref_audio, "rb") as f:
                files["ref_audio"] = (Path(ref_audio).name, f.read(), "audio/wav")
            if ref_text:
                data["ref_text"] = ref_text

        try:
            response = self._client.post(
                "/tts/synthesize",  # TODO: confirm actual path
                data=data,
                files=files or None,
            )
        except RevosEngineError:
            raise
        except Exception as e:
            raise RevosEngineError(
                f"TTS API call failed: {e}",
                suggestion="Check network connectivity and API status.",
            ) from e

        # API returns audio URL or base64 — fetch and decode
        # SKELETON: actual response format pending contract
        audio_bytes = self._fetch_audio(response)

        import soundfile as sf

        samples, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        audio = Audio(samples=samples, sample_rate=sample_rate)
        if output_path:
            audio.save(output_path)
        return audio

    def _fetch_audio(self, response: dict) -> bytes:
        """Fetch audio bytes from API response.

        SKELETON: actual format (URL vs base64 vs inline) pending contract.
        """
        # TODO: replace with actual schema handling once contract is finalized
        if "audio_url" in response:
            return self._client.get_raw(response["audio_url"])
        if "audio_base64" in response:
            import base64

            return base64.b64decode(response["audio_base64"])
        raise RevosEngineError(
            "Malformed API response: no audio data.",
            suggestion="Contact revolab support.",
        )

    def close(self) -> None:
        self._client.close()
