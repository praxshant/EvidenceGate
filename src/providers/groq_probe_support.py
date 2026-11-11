"""Helpers for bounded Groq probe output without making calls."""

from __future__ import annotations

import os


def probe_preflight(expected_calls: int, max_tokens: int) -> dict[str, object]:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_1")
    return {
        "configured": bool(api_key and os.getenv("GROQ_MODEL")),
        "model": os.getenv("GROQ_MODEL", ""),
        "expected_calls": expected_calls,
        "maximum_requested_output_tokens": expected_calls * max_tokens,
        "credentials_are_not_printed": True,
    }
