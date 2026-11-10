"""Shared provider contracts and error classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Mapping, Protocol


class ErrorKind(str, Enum):
    TRANSIENT = "transient"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    DAILY_QUOTA = "daily_quota"
    PERMANENT = "permanent"


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        kind: ErrorKind,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.retry_after = retry_after
        self.headers = dict(headers or {})


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    question: str
    prompt: str
    max_output_tokens: int = 400
    timeout_seconds: float = 30.0
    retrieval_setting: str = "B"


@dataclass
class GenerationResult:
    text: str
    model: str
    tokens_in: int | None
    tokens_out: int | None
    first_token_at: float
    completed_at: float
    status_code: int | None = 200
    headers: dict[str, str] = field(default_factory=dict)
    retries: int = 0


class ModelProvider(Protocol):
    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Yield generated text pieces and raise ProviderError on failure."""

    @property
    def model_name(self) -> str:
        ...
