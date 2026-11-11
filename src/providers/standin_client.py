"""Local replay provider with controllable reliability and streaming."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from .base import ErrorKind, GenerationRequest, ModelProvider, ProviderError


@dataclass
class StandinConfig:
    model: str = "stand-in-replay"
    base_delay_seconds: float = 0.02
    jitter_seconds: float = 0.01
    inter_chunk_seconds: float = 0.005
    chunk_size: int = 32
    transient_failure_rate: float = 0.0
    timeout_rate: float = 0.0
    permanent_failure_rate: float = 0.0
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    requests_per_day: int | None = None
    slowdown_per_concurrent: float = 0.01
    concurrency_limit: int | None = None
    calibration_status: str = "pending_real_groq_measurements"
    day_seconds: float = 86400.0


class ReplayStore:
    def __init__(self, path: Path) -> None:
        self.records: dict[str, dict[str, object]] = {}
        if path.exists():
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # Primary key: full question text
                if "question" in record:
                    self.records[str(record["question"])] = record
                # Fallback: also index by question_id when present
                if "question_id" in record:
                    self.records[str(record["question_id"])] = record

    def get(self, key: str) -> dict[str, object] | None:
        return self.records.get(key)


class StandinProvider(ModelProvider):
    def __init__(self, store: ReplayStore, config: StandinConfig | None = None) -> None:
        self.store = store
        self.config = config or StandinConfig()
        self._active = 0
        self._requests: list[float] = []
        self._tokens: list[tuple[float, int]] = []
        self._day_requests: int = 0
        self._day_start: float = time.monotonic()
        self._last_usage: dict[str, int] | None = None
        self._request_metadata: dict[str, dict] = {}
        self._attempts: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._semaphore = (
            asyncio.Semaphore(self.config.concurrency_limit)
            if self.config.concurrency_limit
            else None
        )

    @property
    def model_name(self) -> str:
        return self.config.model

    @property
    def last_usage(self) -> dict[str, int] | None:
        return self._last_usage

    def get_request_metadata(self, request_id: str) -> dict:
        """Return request-scoped metadata without removing it."""
        return self._request_metadata.get(request_id, {})

    def pop_request_metadata(self, request_id: str) -> dict:
        """Return and remove request-scoped metadata (prevents cross-request leakage)."""
        return self._request_metadata.pop(request_id, {})

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        record = self.store.get(request.question)
        if record is None:
            raise ProviderError(
                "No replay record found for question. The production replay file"
                " (data/replay/responses.jsonl) is empty. Record real Groq responses first.",
                ErrorKind.PERMANENT,
            )
        prompt_tokens = int(record.get("tokens_in", 0) or 0)
        completion_tokens = int(record.get("tokens_out", 0) or 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        self._last_usage = usage
        self._request_metadata[request.request_id] = {
            "usage": usage,
            "headers": {},
        }
        total_tokens = prompt_tokens + completion_tokens

        if self._semaphore:
            await self._semaphore.acquire()
        try:
            async with self._lock:
                self._check_limits(total_tokens)
                self._active += 1
            await self._failure_injection(request.timeout_seconds, request.request_id)
            delay = self.config.base_delay_seconds + random.random() * self.config.jitter_seconds
            delay += self._active * self.config.slowdown_per_concurrent
            await asyncio.sleep(delay)
            text = str(record["answer"])
            for offset in range(0, len(text), self.config.chunk_size):
                await asyncio.sleep(self.config.inter_chunk_seconds)
                yield text[offset : offset + self.config.chunk_size]
        finally:
            async with self._lock:
                self._active = max(0, self._active - 1)
            if self._semaphore:
                self._semaphore.release()

    def _check_limits(self, total_tokens: int) -> None:
        now = time.monotonic()
        # Reset daily quota if the day window has elapsed
        if now - self._day_start >= self.config.day_seconds:
            self._day_requests = 0
            self._day_start = now
        self._requests = [t for t in self._requests if now - t < 60]
        self._tokens = [(t, n) for t, n in self._tokens if now - t < 60]
        if self.config.requests_per_minute and len(self._requests) >= self.config.requests_per_minute:
            raise ProviderError("Stand-in request-per-minute limit reached", ErrorKind.RATE_LIMITED)
        if self.config.tokens_per_minute and sum(n for _, n in self._tokens) + total_tokens > self.config.tokens_per_minute:
            raise ProviderError("Stand-in token-per-minute limit reached", ErrorKind.RATE_LIMITED)
        if self.config.requests_per_day and self._day_requests >= self.config.requests_per_day:
            raise ProviderError("Stand-in daily quota exhausted", ErrorKind.DAILY_QUOTA)
        self._requests.append(now)
        self._tokens.append((now, total_tokens))
        self._day_requests += 1

    def reset_daily_quota(self) -> None:
        """Reset the daily request counter and window (for use in tests)."""
        self._day_requests = 0
        self._day_start = time.monotonic()

    async def _failure_injection(self, timeout_seconds: float, request_id: str = "") -> None:
        if request_id:
            attempt = self._attempts.get(request_id, 0)
            self._attempts[request_id] = attempt + 1
            import hashlib
            seed_bytes = f"{request_id}:{attempt}".encode()
            digest = hashlib.md5(seed_bytes).hexdigest()
            roll = int(digest[:8], 16) / 0xFFFFFFFF
        else:
            roll = random.random()

        if roll < self.config.permanent_failure_rate:
            raise ProviderError("Configured stand-in permanent failure", ErrorKind.PERMANENT)
        if roll < self.config.permanent_failure_rate + self.config.transient_failure_rate:
            raise ProviderError("Configured stand-in transient failure", ErrorKind.TRANSIENT)
        if roll < (
            self.config.permanent_failure_rate
            + self.config.transient_failure_rate
            + self.config.timeout_rate
        ):
            await asyncio.sleep(timeout_seconds + 0.01)
            raise ProviderError("Configured stand-in timeout", ErrorKind.TIMEOUT)
