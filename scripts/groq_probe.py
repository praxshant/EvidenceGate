"""Print a quota-safe Groq probe preflight; run calls only with --run --confirm."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# F-03: fix import path when run directly from the command line
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.providers.groq_probe_support import probe_preflight  # noqa: E402


async def run_probe(
    api_key: str,
    model: str,
    num_calls: int,
    max_tokens: int,
) -> dict:
    """Execute bounded streaming probe calls against the Groq API.

    One JSONL line is appended to <project_root>/raw_headers.jsonl per call.
    Returns a summary dict with per-call measurements.
    """
    from groq import AsyncGroq  # type: ignore[import]

    client = AsyncGroq(api_key=api_key)
    project_root = Path(__file__).resolve().parents[1]
    jsonl_path = project_root / "raw_headers.jsonl"

    measurements: list[dict] = []

    for call_index in range(num_calls):
        call_result: dict = {
            "call_index": call_index,
            "model": model,
            "status": None,
            "wall_start": None,
            "first_token_monotonic": None,
            "completion_monotonic": None,
            "headers": {},
            "usage": None,
            "error": None,
        }

        wall_start = time.time()
        call_result["wall_start"] = wall_start
        start_mono = time.monotonic()
        first_token_mono: float | None = None
        last_usage: dict | None = None

        try:
            stream_obj = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say hello in one sentence."}],
                max_tokens=max_tokens,
                stream=True,
                reasoning_format="hidden",
            )

            async for chunk in stream_obj:
                # Capture first-token timing
                if first_token_mono is None:
                    choices = getattr(chunk, "choices", None)
                    if choices:
                        delta = getattr(choices[0], "delta", None)
                        content = getattr(delta, "content", None) if delta else None
                        if content:
                            first_token_mono = time.monotonic()

                # Capture usage from final chunk
                if getattr(chunk, "usage", None) is not None:
                    last_usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                    }

            completion_mono = time.monotonic()
            call_result["first_token_monotonic"] = (
                first_token_mono - start_mono if first_token_mono is not None else None
            )
            call_result["completion_monotonic"] = completion_mono - start_mono
            call_result["usage"] = last_usage
            call_result["status"] = "ok"

            # Capture response headers
            try:
                raw_headers = None
                if hasattr(stream_obj, "response") and hasattr(stream_obj.response, "headers"):
                    raw_headers = stream_obj.response.headers
                elif hasattr(stream_obj, "_response") and hasattr(
                    stream_obj._response, "headers"
                ):
                    raw_headers = stream_obj._response.headers

                if raw_headers is not None:
                    _of_interest = {
                        "x-ratelimit-limit-requests",
                        "x-ratelimit-remaining-requests",
                        "x-ratelimit-reset-requests",
                        "x-ratelimit-limit-tokens",
                        "x-ratelimit-remaining-tokens",
                        "x-ratelimit-reset-tokens",
                        "retry-after",
                    }
                    call_result["headers"] = {
                        k.lower(): v
                        for k, v in raw_headers.items()
                        if k.lower() in _of_interest
                    }
            except AttributeError:
                pass

        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            exc_headers = getattr(exc, "headers", {}) or {}
            if status_code == 429:
                lower = str(exc).lower()
                call_result["status"] = "daily_quota" if "day" in lower else "rate_limited"
            else:
                call_result["status"] = "error"
            call_result["error"] = str(exc)
            call_result["headers"] = dict(exc_headers) if exc_headers else {}

        measurements.append(call_result)

        # Append one JSONL line
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(call_result) + "\n")

        print(
            f"  call {call_index + 1}/{num_calls}: status={call_result['status']}"
            f"  first_token={call_result['first_token_monotonic']}"
            f"  total={call_result['completion_monotonic']}"
        )

    return {
        "num_calls": num_calls,
        "calls": measurements,
        "jsonl_path": str(jsonl_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Groq probe: preflight check and optional bounded live probe."
    )
    parser.add_argument("--expected-calls", type=int, default=3, help="Number of probe calls.")
    parser.add_argument(
        "--max-tokens", type=int, default=200, help="Max output tokens per call."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Enable network calls (also requires --confirm).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm you accept quota consumption. Required together with --run.",
    )
    args = parser.parse_args()

    preflight = probe_preflight(args.expected_calls, args.max_tokens)
    print(json.dumps(preflight, indent=2))

    if not preflight["configured"]:
        print(
            "\nGroq probe not run: GROQ_API_KEY and GROQ_MODEL environment variables are required."
        )
        return 2

    if not args.run:
        print(
            "\nPreflight only. This would send "
            f"{args.expected_calls} streaming request(s) to model '{preflight['model']}', "
            f"each requesting up to {args.max_tokens} tokens "
            f"({preflight['maximum_requested_output_tokens']} tokens total max).\n"
            "Re-run with --run --confirm after confirming quota and call budget."
        )
        return 0

    if not args.confirm:
        print(
            "\n--run was set but --confirm was not. "
            "Add --confirm to acknowledge that real API quota will be consumed."
        )
        return 1

    # Execute the real probe
    api_key = os.environ.get("GROQ_API_KEY") or os.environ["GROQ_API_KEY_1"]
    model = os.environ["GROQ_MODEL"]

    print(
        f"\nRunning {args.expected_calls} probe call(s) to model '{model}' "
        f"(max {args.max_tokens} tokens each)…"
    )

    summary = asyncio.run(run_probe(api_key, model, args.expected_calls, args.max_tokens))
    print("\nSummary:")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nRaw headers written to: {summary['jsonl_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
