"""Shared request-level and stage diagnostic logging."""

from __future__ import annotations

import asyncio
import csv
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

RESULT_COLUMNS = [
    "request_id", "scenario", "source", "start_time", "first_token_ms",
    "total_ms", "status", "retries", "tokens_in", "tokens_out",
    "retrieval_setting", "answer_grade",
]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def now_iso_hires() -> str:
    """Return current UTC time as an ISO-8601 string with microsecond precision."""
    return datetime.now(timezone.utc).isoformat()


def now_iso() -> str:
    """Return current UTC time as ISO-8601 (microsecond precision, backward-compat name)."""
    return now_iso_hires()


def monotonic_ms() -> float:
    """Return time.monotonic() in milliseconds (float).

    Use for duration diffs between pipeline stages, NOT for wall-clock time.
    """
    return time.monotonic() * 1000.0


# ---------------------------------------------------------------------------
# DiagnosticEntry
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticEntry:
    """Typed container for per-request pipeline stage timing."""

    # Monotonic timestamps (ms since process start)
    arrival_ms: float = 0.0
    queue_exit_ms: float = 0.0          # same as arrival_ms until a real queue exists
    retrieval_start_ms: float = 0.0
    retrieval_end_ms: float = 0.0
    provider_start_ms: float = 0.0
    first_token_ms: float = 0.0
    provider_end_ms: float = 0.0
    verification_start_ms: float = 0.0
    verification_end_ms: float = 0.0
    final_response_ms: float = 0.0

    # Counters / metadata
    retries: int = 0
    decision: str = ""
    wall_clock_start: str = field(default_factory=now_iso_hires)

    @property
    def total_ms(self) -> float:
        """End-to-end latency from arrival to final response (ms)."""
        return self.final_response_ms - self.arrival_ms

    def to_dict(self, request_id: str = "") -> dict[str, Any]:
        d = asdict(self)
        d["total_ms"] = self.total_ms
        if request_id:
            d["request_id"] = request_id
        return d


# ---------------------------------------------------------------------------
# ExperimentLogger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.results_path = root / "results.csv"
        self.diagnostics_path = root / "diagnostics.jsonl"
        self.headers_path = root / "raw_headers.jsonl"
        self._lock: asyncio.Lock = asyncio.Lock()

    async def result(self, **values: Any) -> None:
        """Append one row to results.csv (async, lock-protected)."""
        row = {column: values.get(column, "") for column in RESULT_COLUMNS}
        async with self._lock:
            new_file = not self.results_path.exists()
            with self.results_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)

    async def diagnostic(self, **values: Any) -> None:
        """Append a free-form diagnostic record to diagnostics.jsonl (async, lock-protected)."""
        async with self._lock:
            with self.diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(values, ensure_ascii=False) + "\n")

    async def diagnostic_entry(self, request_id: str, entry: dict[str, Any]) -> None:
        """Append a structured DiagnosticEntry-shaped dict to diagnostics.jsonl (async, lock-protected).

        ``entry`` should be the output of ``DiagnosticEntry.to_dict()`` or any
        dict with the same keys.  ``request_id`` is merged in automatically.
        """
        payload = dict(entry)
        payload.setdefault("request_id", request_id)
        # Ensure total_ms is present even if entry came from a plain dict
        if "total_ms" not in payload and "arrival_ms" in payload and "final_response_ms" in payload:
            payload["total_ms"] = payload["final_response_ms"] - payload["arrival_ms"]
        async with self._lock:
            with self.diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    async def headers(self, request_id: str, headers: dict[str, str]) -> None:
        """Append raw request headers to raw_headers.jsonl (async, lock-protected)."""
        async with self._lock:
            with self.headers_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"request_id": request_id, "headers": headers}) + "\n")
