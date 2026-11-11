"""Async streaming Groq provider.

The SDK is imported lazily so retrieval and stand-in tests do not require a
credential or a network dependency.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator

from .base import ErrorKind, GenerationRequest, ModelProvider, ProviderError

# Rate-limit headers of interest (lower-cased for case-insensitive lookup)
_RATELIMIT_HEADERS = frozenset(
    [
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
        "retry-after",
    ]
)


class GroqProvider(ModelProvider):
    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GROQ_API_KEY_1")
        )
        self._model = model or os.getenv("GROQ_MODEL", "")
        if not self._api_key:
            raise ProviderError("GROQ_API_KEY is not configured", ErrorKind.PERMANENT)
        if not self._model:
            raise ProviderError("GROQ_MODEL is not configured", ErrorKind.PERMANENT)
        try:
            from groq import AsyncGroq
        except ImportError as exc:
            raise ProviderError(
                "The official groq package is not installed", ErrorKind.PERMANENT
            ) from exc
        self._client = AsyncGroq(api_key=self._api_key)
        # F-04: captured response headers from most recent stream() call
        self._last_headers: dict[str, str] = {}
        # F-05: captured token usage from most recent stream() call
        self._last_usage: dict | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def last_response_headers(self) -> dict[str, str]:
        """Return rate-limit headers captured from the most recent stream() call."""
        return self._last_headers

    @property
    def last_usage(self) -> dict | None:
        """Return token usage captured from the most recent stream() call."""
        return self._last_usage

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        # Reset per-call state
        self._last_headers = {}
        self._last_usage = None

        try:
            # F-04: capture the stream object so we can read headers afterwards
            stream_obj = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": request.prompt}],
                    max_tokens=request.max_output_tokens,
                    stream=True,
                    # gpt-oss-20b is a reasoning model; hide the reasoning chain so
                    # delta.content carries readable text instead of delta.reasoning.
                    reasoning_format="hidden",
                ),
                timeout=request.timeout_seconds,
            )

            async for chunk in stream_obj:
                # F-05: capture usage from the final chunk
                if getattr(chunk, "usage", None) is not None:
                    self._last_usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                    }

                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                text = getattr(delta, "content", None) if delta else None
                if text:
                    yield text

            # F-04: extract rate-limit headers from the completed stream response
            try:
                raw_headers = None
                if hasattr(stream_obj, "response") and hasattr(stream_obj.response, "headers"):
                    raw_headers = stream_obj.response.headers
                elif hasattr(stream_obj, "_response") and hasattr(stream_obj._response, "headers"):
                    raw_headers = stream_obj._response.headers

                if raw_headers is not None:
                    self._last_headers = {
                        k.lower(): v
                        for k, v in raw_headers.items()
                        if k.lower() in _RATELIMIT_HEADERS
                    }
            except AttributeError:
                pass

        except asyncio.TimeoutError as exc:
            raise ProviderError("Groq request timed out", ErrorKind.TIMEOUT) from exc
        except Exception as exc:
            raise self._classify(exc) from exc

    @staticmethod
    def _classify(exc: Exception) -> ProviderError:
        status = getattr(exc, "status_code", None)
        headers = getattr(exc, "headers", {}) or {}
        message = str(exc)
        if status == 429:
            lower = message.lower()
            kind = ErrorKind.DAILY_QUOTA if "day" in lower else ErrorKind.RATE_LIMITED
        elif status is not None and status >= 500:
            kind = ErrorKind.TRANSIENT
        else:
            kind = ErrorKind.PERMANENT
        retry_after = None
        try:
            retry_after = float(headers.get("retry-after")) if headers.get("retry-after") else None
        except (TypeError, ValueError):
            retry_after = None
        return ProviderError(
            f"Groq request failed: {message}",
            kind,
            status_code=status,
            retry_after=retry_after,
            headers=headers,
        )


def monotonic_now() -> float:
    return time.monotonic()
