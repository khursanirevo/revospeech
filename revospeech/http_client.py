"""HTTP client utilities for revolab cloud API.

Provides a shared httpx-based client with:
- Auth header injection from get_api_key()
- Retry on 5xx errors
- Timeout
- Request/response logging (API key masked)
- Error mapping: 401 -> RevosConfigError, 429 -> rate limit, 5xx -> retry
"""

from __future__ import annotations

import logging
from typing import Any

from revospeech.config import get_api_key
from revospeech.exceptions import RevosConfigError, RevosEngineError

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3


class RevolabClient:
    """HTTP client for revolab cloud API.

    Wraps httpx with auth, retries, and error mapping.

    Usage:
        client = RevolabClient(endpoint="https://api.revolab.ai/v1")
        response = client.post("/asr/transcribe", json={...})
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key or get_api_key() or ""
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise RevosConfigError(
                "Revolab API key required.",
                suggestion=(
                    "Set your API key: export REVOLAB_API_KEY=your-key"
                    " or run: revospeech config set-api-key"
                ),
            )

        try:
            import httpx
        except ImportError as e:
            raise RevosConfigError(
                "httpx is required for API backends.",
                suggestion="Install with: pip install 'revospeech[api]'",
            ) from e

        self._httpx = httpx
        self._client = httpx.Client(
            base_url=self.endpoint,
            timeout=timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def _mask_key(self) -> str:
        if len(self.api_key) < 8:
            return "***"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    def post(self, path: str, **kwargs: Any) -> dict:
        """POST request with retry on 5xx."""
        return self._request("POST", path, **kwargs)

    def get(self, path: str, **kwargs: Any) -> dict:
        """GET request with retry on 5xx."""
        return self._request("GET", path, **kwargs)

    def get_raw(self, path: str, **kwargs: Any) -> bytes:
        """GET request returning raw bytes (for binary downloads)."""
        url = path if path.startswith("http") else f"{self.endpoint}{path}"
        logger.debug("HTTP GET (raw) %s (auth=%s)", url, self._mask_key())

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.request("GET", url, **kwargs)

                if response.status_code == 401:
                    raise RevosConfigError(
                        "Invalid API key (HTTP 401).",
                        suggestion="Check your API key: revospeech config show-api-key",
                    )
                if response.status_code == 403:
                    raise RevosConfigError(
                        "API key lacks permission (HTTP 403).",
                        suggestion="Contact revolab support to upgrade your plan.",
                    )
                if response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        import time

                        wait = 2**attempt
                        logger.warning("Rate limited, retrying in %ds", wait)
                        time.sleep(wait)
                        continue
                    raise RevosEngineError(
                        "Rate limit exceeded.",
                        suggestion="Reduce request frequency or upgrade plan.",
                    )
                if 500 <= response.status_code < 600:
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            "HTTP %d, retrying (%d/%d)",
                            response.status_code,
                            attempt + 1,
                            self.max_retries,
                        )
                        continue
                    raise RevosEngineError(
                        f"Server error (HTTP {response.status_code}).",
                    )

                response.raise_for_status()
                return response.content
            except self._httpx.HTTPError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    continue
                raise RevosEngineError(f"Network error: {e}") from e

        raise RevosEngineError(
            f"Request failed after {self.max_retries} retries: {last_error}"
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        url = path if path.startswith("http") else f"{self.endpoint}{path}"
        logger.debug("HTTP %s %s (auth=%s)", method, url, self._mask_key())

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.request(method, url, **kwargs)

                if response.status_code == 401:
                    raise RevosConfigError(
                        "Invalid API key (HTTP 401).",
                        suggestion="Check your API key: revospeech config show-api-key",
                    )
                if response.status_code == 403:
                    raise RevosConfigError(
                        "API key lacks permission (HTTP 403).",
                        suggestion="Contact revolab support to upgrade your plan.",
                    )
                if response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        import time

                        wait = 2**attempt
                        logger.warning("Rate limited, retrying in %ds", wait)
                        time.sleep(wait)
                        continue
                    raise RevosEngineError(
                        "Rate limit exceeded.",
                        suggestion="Reduce request frequency or upgrade plan.",
                    )
                if 500 <= response.status_code < 600:
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            "HTTP %d, retrying (%d/%d)",
                            response.status_code,
                            attempt + 1,
                            self.max_retries,
                        )
                        continue
                    raise RevosEngineError(
                        f"Server error (HTTP {response.status_code}).",
                    )

                response.raise_for_status()
                return response.json()
            except self._httpx.HTTPError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    continue
                raise RevosEngineError(f"Network error: {e}") from e

        raise RevosEngineError(
            f"Request failed after {self.max_retries} retries: {last_error}"
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RevolabClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
