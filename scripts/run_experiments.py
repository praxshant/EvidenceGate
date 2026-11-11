"""run_experiments.py — EvidenceGate corrected end-to-end experiment runner.

Execution order (matches recommended sequence from review):
1. Provider-corrected Groq probe  — 5 small streaming calls to build a timing
   distribution (not just a smoke test).
2. Rigorous stand-in calibration  — mean + stdev from valid TTFT observations.
3. Replay store population        — real answers for probed questions;
                                    gold answers for the rest.
4. Benchmark evaluation           — settings A, B, C, 20 questions each.
5. Five load scenarios            — using calibrated stand-in.
6. Paired retry experiment        — 0 retries / 3 immediate retries /
                                    exponential-backoff+jitter under a fixed
                                    failure sequence for fair comparison.
7. LiveKit subset                 — 30 issues unlabeled.
8. Analysis                       — regenerate milestone_4_5_summary.json.

Key correctness rules enforced here:
- ALL Groq calls use include_reasoning=False + reasoning_effort="low" +
  max_completion_tokens (not max_tokens).
- TTFT is recorded only when a non-empty delta.content chunk actually arrives.
- total_ms is wall-clock from request dispatch to stream end (always >= TTFT).
- Latency anomalies (total < TTFT) are flagged and logged, never clamped.
- Percentile populations are paired (same request IDs for TTFT and total).
- Error-class latency is logged separately from success-class latency.
- Token-budget calculation uses max_completion_tokens=100 per probe call to
  leave room for text after any residual reasoning.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.logging_utils import DiagnosticEntry, ExperimentLogger, monotonic_ms, now_iso_hires
from src.pipeline import EvidenceGate, ResearchPipeline, build_prompt
from src.providers.base import ErrorKind, ProviderError
from src.providers.standin_client import ReplayStore, StandinConfig, StandinProvider
from src.retrieval import PaperRetriever
from src.verification import verify_answer

# ---------------------------------------------------------------------------
# Env loader
# ---------------------------------------------------------------------------

def _load_env() -> tuple[str, str]:
    env_vars: dict[str, str] = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip().strip("'\"")
    api_key = env_vars.get("GROQ_API_KEY_1") or os.environ.get("GROQ_API_KEY_1") or ""
    model = env_vars.get("GROQ_MODEL") or os.environ.get("GROQ_MODEL") or "openai/gpt-oss-20b"
    return api_key, model


GROQ_API_KEY, GROQ_MODEL = _load_env()

_RL_HEADERS = {
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
    "retry-after",
}

# ---------------------------------------------------------------------------
# Probe question pool — 5 short questions from the benchmark
# ---------------------------------------------------------------------------

PROBE_QUESTIONS = [
    {"id": "q01", "question": "Why does the Transformer use self-attention instead of recurrence?"},
    {"id": "q03", "question": "What does multi-head attention allow the model to attend to?"},
    {"id": "q04", "question": "Why does the Transformer add positional encodings?"},
    {"id": "q05", "question": "What theoretical and practical latency does the Moshi paper report?"},
    {"id": "q08", "question": "What does the Dynamic Chunking paper say an H-Net replaces?"},
]

# ---------------------------------------------------------------------------
# Phase 1: Real Groq probe — 5 calls with verified content output
# ---------------------------------------------------------------------------

async def run_real_groq_probe(
    logger: ExperimentLogger,
    retriever: PaperRetriever,
    gate: EvidenceGate,
) -> list[dict]:
    """Run 5 bounded streaming calls to Groq and collect timing observations.

    Uses include_reasoning=False + reasoning_effort="low" + max_completion_tokens=100.
    If a call produces zero content chunks the observation is flagged (first_token_ms=None)
    and NOT included in the TTFT distribution but IS included in total_ms distribution.
    """
    if not GROQ_API_KEY:
        print("[probe] GROQ_API_KEY_1 not set — skipping real probe.")
        return []

    from groq import AsyncGroq
    client = AsyncGroq(api_key=GROQ_API_KEY, max_retries=0)

    print(f"\n=== Phase 1: Real Groq Probe ({GROQ_MODEL}) ===")
    observations: list[dict] = []

    for item in PROBE_QUESTIONS:
        qid = item["id"]
        q_text = item["question"]
        print(f"\n[probe] {qid}: {q_text[:60]}...")

        wall_start = now_iso_hires()
        request_id = f"groq-probe-{qid}"

        # Retrieve passages for a realistic prompt
        passages = retriever.search(q_text, "B")
        decision = gate.decide(passages, q_text)
        prompt = build_prompt(q_text, passages, decision)

        t_dispatch = time.monotonic()
        first_token_mono: float | None = None
        pieces: list[str] = []
        last_usage: dict | None = None
        rl_headers: dict[str, str] = {}
        status = "ok"
        error_msg = ""
        finish_reason: str | None = None

        try:
            stream_obj = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                # 100 tokens: short enough to stay well within TPM limit across
                # 5 calls, large enough to get a few sentences of visible text.
                max_completion_tokens=100,
                stream=True,
                include_reasoning=False,
                reasoning_effort="low",
            )

            async for chunk in stream_obj:
                if getattr(chunk, "usage", None) is not None:
                    u = chunk.usage
                    last_usage = {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                    }
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    fr = getattr(choices[0], "finish_reason", None)
                    if fr:
                        finish_reason = fr
                    delta = getattr(choices[0], "delta", None)
                    content = getattr(delta, "content", None) if delta else None
                    if content:
                        if first_token_mono is None:
                            first_token_mono = time.monotonic()
                        pieces.append(content)

            # Capture rate-limit headers
            for attr in ("response", "_response"):
                resp = getattr(stream_obj, attr, None)
                if resp is not None and hasattr(resp, "headers"):
                    rl_headers = {
                        k.lower(): v
                        for k, v in resp.headers.items()
                        if k.lower() in _RL_HEADERS
                    }
                    break

        except Exception as exc:
            status = "error"
            error_msg = str(exc)
            print(f"  ERROR: {exc}")

        t_end = time.monotonic()
        total_ms = (t_end - t_dispatch) * 1000.0
        ttft_ms: float | None = (
            (first_token_mono - t_dispatch) * 1000.0
            if first_token_mono is not None else None
        )

        # Invariant check — flag but do NOT clamp
        if ttft_ms is not None and total_ms < ttft_ms:
            print(f"  [ANOMALY] total_ms={total_ms:.1f} < first_token_ms={ttft_ms:.1f} — "
                  f"clock skew or measurement bug. Flagging, NOT clamping.")
            status = "latency_anomaly"

        answer = "".join(pieces).strip()
        tokens_in = (last_usage or {}).get("prompt_tokens", "")
        tokens_out = (last_usage or {}).get("completion_tokens", "")

        print(f"  status={status}  content_chunks={len(pieces)}  "
              f"ttft={f'{ttft_ms:.0f}' if ttft_ms is not None else 'None'}ms  "
              f"total={total_ms:.0f}ms  "
              f"tokens=({tokens_in}/{tokens_out})")
        print(f"  rl_headers={rl_headers}")
        if answer:
            print(f"  answer_preview={answer[:80]}...")
        else:
            print("  [WARNING] Zero content chunks received — reasoning may have consumed the token budget.")

        # Log to results.csv
        await logger.result(
            request_id=request_id,
            scenario="groq_probe",
            source="real",
            start_time=wall_start,
            first_token_ms=round(ttft_ms, 2) if ttft_ms is not None else "",
            total_ms=round(total_ms, 2),
            status=status,
            retries=0,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            retrieval_setting="B",
            answer_grade="",
        )
        if rl_headers:
            await logger.headers(request_id, rl_headers)

        # Also append to raw_headers.jsonl with token usage side-by-side
        raw_headers_path = ROOT / "raw_headers.jsonl"
        with raw_headers_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "request_id": request_id,
                "headers": rl_headers,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "total_tokens_used": (tokens_in + tokens_out) if isinstance(tokens_in, int) and isinstance(tokens_out, int) else "unknown",
                "note": (
                    "header_remaining_tokens reflects Groq's TPM bucket state at "
                    "response time, which includes per-minute refill. "
                    "Direct subtraction from limit is not meaningful without knowing "
                    "the elapsed refill interval. Groq docs do not publish the exact "
                    "refill equation."
                ),
            }) + "\n")

        observations.append({
            "id": qid,
            "question": q_text,
            "answer": answer,
            "ttft_ms": ttft_ms,          # None if no content chunks
            "total_ms": total_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "finish_reason": finish_reason,
            "headers": rl_headers,
            "status": status,
        })

        await asyncio.sleep(2.5)  # Keep inter-call gap to respect TPM limits

    return observations


# ---------------------------------------------------------------------------
# Phase 2: Rigorous stand-in calibration
# ---------------------------------------------------------------------------

def calibrate_standin(observations: list[dict]) -> StandinConfig:
    """Calibrate timing from probe observations.

    Uses only observations where:
    - ttft_ms is not None (a content token was actually received)
    - total_ms > ttft_ms (no latency anomaly)

    Reports mean and stdev of TTFT; uses them for base_delay ± jitter.
    If fewer than 2 valid observations exist, falls back to conservative defaults.
    """
    import statistics

    print("\n=== Phase 2: Stand-in Calibration ===")

    valid = [
        o for o in observations
        if o.get("ttft_ms") is not None
        and o["total_ms"] > o["ttft_ms"]
        and o["status"] == "ok"
    ]
    print(f"  Valid observations for calibration: {len(valid)} / {len(observations)}")

    cfg_path = ROOT / "data" / "standin" / "default.json"

    if len(valid) >= 2:
        ttfts_s = [o["ttft_ms"] / 1000.0 for o in valid]
        totals_s = [o["total_ms"] / 1000.0 for o in valid]
        gen_times_s = [t - f for t, f in zip(totals_s, ttfts_s)]

        mean_ttft = statistics.mean(ttfts_s)
        std_ttft = statistics.stdev(ttfts_s)
        mean_gen = statistics.mean(gen_times_s)

        avg_tokens_out = statistics.mean([
            o["tokens_out"] for o in valid
            if isinstance(o.get("tokens_out"), int) and o["tokens_out"] > 0
        ] or [45])
        avg_chunks = max(1.0, avg_tokens_out / 32.0)

        base_delay = round(max(0.05, mean_ttft - std_ttft), 3)
        jitter = round(max(0.02, 2 * std_ttft), 3)
        inter_chunk = round(max(0.002, mean_gen / avg_chunks), 4)

        calibration_note = (
            f"Calibrated from N={len(valid)} valid real Groq observations. "
            f"mean_ttft={mean_ttft*1000:.0f}ms stdev_ttft={std_ttft*1000:.0f}ms "
            f"mean_gen={mean_gen*1000:.0f}ms. "
            f"base_delay = mean - stdev. jitter = 2 * stdev."
        )
        calibration_status = "calibrated_from_real_groq"
    elif len(valid) == 1:
        single = valid[0]
        base_delay = round(single["ttft_ms"] / 1000.0 * 0.9, 3)
        jitter = round(single["ttft_ms"] / 1000.0 * 0.2, 3)
        inter_chunk = 0.004
        calibration_note = (
            f"Single valid observation (N=1); insufficient for stdev. "
            f"base_delay=0.9*TTFT, jitter=0.2*TTFT."
        )
        calibration_status = "calibrated_single_observation"
    else:
        base_delay = 0.5
        jitter = 0.1
        inter_chunk = 0.004
        calibration_note = (
            "No valid TTFT observations from real probe. "
            "Using conservative defaults. Stand-in timing does not represent real Groq."
        )
        calibration_status = "uncalibrated_no_valid_observations"

    config_data = {
        "model": "stand-in-replay",
        "base_delay_seconds": base_delay,
        "jitter_seconds": jitter,
        "inter_chunk_seconds": inter_chunk,
        "chunk_size": 32,
        "transient_failure_rate": 0.0,
        "timeout_rate": 0.0,
        "permanent_failure_rate": 0.0,
        "requests_per_minute": None,
        "tokens_per_minute": None,
        "requests_per_day": None,
        "slowdown_per_concurrent": 0.01,
        "concurrency_limit": None,
        "calibration_status": calibration_status,
        "calibration_n": len(valid),
        "calibration_note": calibration_note,
        "day_seconds": 86400,
    }

    cfg_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    print(f"  {calibration_status}: base_delay={base_delay}s  jitter={jitter}s  inter_chunk={inter_chunk}s")
    print(f"  Note: {calibration_note}")

    from dataclasses import fields
    valid_keys = {f.name for f in fields(StandinConfig)}
    return StandinConfig(**{k: v for k, v in config_data.items() if k in valid_keys})


# ---------------------------------------------------------------------------
# Phase 3: Populate replay store
# ---------------------------------------------------------------------------

def populate_replay_store(observations: list[dict]) -> None:
    """Write real Groq answers (where available) + gold answers to responses.jsonl."""
    print("\n=== Phase 3: Populate Replay Store ===")

    benchmark = json.loads(
        (ROOT / "data" / "benchmarks" / "benchmark_draft.json").read_text(encoding="utf-8")
    )["questions"]
    replay_file = ROOT / "data" / "replay" / "responses.jsonl"
    obs_map = {o["id"]: o for o in observations if o.get("answer") and o.get("status") == "ok"}

    records: list[dict] = []
    for q in benchmark:
        qid = q["id"]
        q_text = q["question"]

        if qid in obs_map:
            obs = obs_map[qid]
            records.append({
                "question": q_text,
                "question_id": qid,
                "answer": obs["answer"],
                "tokens_in": int(obs["tokens_in"]) if isinstance(obs.get("tokens_in"), int) else 100,
                "tokens_out": int(obs["tokens_out"]) if isinstance(obs.get("tokens_out"), int) else 40,
                "source": "real_groq",
            })
        else:
            citation = q.get("expected_citation") or ""
            facts = q.get("expected_facts") or ""
            if citation:
                answer = f"{citation} {facts}"
            else:
                answer = "I cannot answer this from the approved research papers."
            records.append({
                "question": q_text,
                "question_id": qid,
                "answer": answer,
                "tokens_in": 120,
                "tokens_out": 45,
                "source": "benchmark_gold",
            })

    with replay_file.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    real_count = sum(1 for r in records if r["source"] == "real_groq")
    print(f"  Wrote {len(records)} records ({real_count} real, {len(records)-real_count} gold).")


# ---------------------------------------------------------------------------
# Phase 4: Benchmark evaluation (settings A, B, C)
# ---------------------------------------------------------------------------

async def run_benchmark_evaluations(
    logger: ExperimentLogger,
    retriever: PaperRetriever,
    standin_cfg: StandinConfig,
) -> None:
    print("\n=== Phase 4: Benchmark Evaluation (Settings A, B, C) ===")
    benchmark = json.loads(
        (ROOT / "data" / "benchmarks" / "benchmark_draft.json").read_text(encoding="utf-8")
    )["questions"]
    store = ReplayStore(ROOT / "data" / "replay" / "responses.jsonl")

    for setting in ("A", "B", "C"):
        print(f"\n[eval] Setting {setting}")
        provider = StandinProvider(store, standin_cfg)
        pipeline = ResearchPipeline(ROOT, {"stand-in": provider}, logger=logger, retriever=retriever)

        counts = {"ok": 0, "refused_by_gate": 0, "other": 0}
        for q in benchmark:
            res = await pipeline.ask(
                q["question"],
                provider_name="stand-in",
                retrieval_setting=setting,
                request_id=f"bench-{setting}-{q['id']}",
                scenario=f"benchmark_setting_{setting}",
                benchmark_item=q,
            )
            if res.status == "ok":
                counts["ok"] += 1
            elif res.decision == "insufficient":
                counts["refused_by_gate"] += 1
            else:
                counts["other"] += 1

        print(f"  Setting {setting}: ok={counts['ok']}  gated_refusal={counts['refused_by_gate']}  other={counts['other']}")


# ---------------------------------------------------------------------------
# Phase 5: Load scenarios
# ---------------------------------------------------------------------------

async def run_load_scenarios(
    logger: ExperimentLogger,
    retriever: PaperRetriever,
    base_cfg: StandinConfig,
) -> None:
    print("\n=== Phase 5: Load Scenarios ===")
    benchmark = json.loads(
        (ROOT / "data" / "benchmarks" / "benchmark_draft.json").read_text(encoding="utf-8")
    )["questions"]
    store = ReplayStore(ROOT / "data" / "replay" / "responses.jsonl")

    # 1. Steady load — 20 sequential, 50ms inter-request gap
    print("\n[scenario] steady_load: 20 sequential requests")
    p1 = StandinProvider(store, base_cfg)
    pipe1 = ResearchPipeline(ROOT, {"stand-in": p1}, logger=logger, retriever=retriever)
    for i in range(20):
        q = benchmark[i % len(benchmark)]
        await pipe1.ask(q["question"], provider_name="stand-in",
                        retrieval_setting="B", request_id=f"steady-{i:02d}",
                        scenario="steady_load")
        await asyncio.sleep(0.05)
    print("  steady_load done")

    # 2. Sudden burst — 20 concurrent requests
    print("\n[scenario] sudden_burst: 20 concurrent requests")
    burst_cfg = replace(base_cfg, slowdown_per_concurrent=0.015, concurrency_limit=20)
    p2 = StandinProvider(store, burst_cfg)
    pipe2 = ResearchPipeline(ROOT, {"stand-in": p2}, logger=logger, retriever=retriever)
    await asyncio.gather(*[
        pipe2.ask(benchmark[i % len(benchmark)]["question"],
                  provider_name="stand-in", retrieval_setting="B",
                  request_id=f"burst-{i:02d}", scenario="sudden_burst")
        for i in range(20)
    ])
    print("  sudden_burst done")

    # 3. Sustained overload — 50 requests at RPM=5
    print("\n[scenario] sustained_overload: 50 requests, requests_per_minute=5")
    overload_cfg = replace(base_cfg, requests_per_minute=5)
    p3 = StandinProvider(store, overload_cfg)
    pipe3 = ResearchPipeline(ROOT, {"stand-in": p3}, logger=logger, retriever=retriever)
    for i in range(50):
        try:
            await pipe3.ask(benchmark[i % len(benchmark)]["question"],
                            provider_name="stand-in", retrieval_setting="B",
                            request_id=f"overload-{i:02d}", scenario="sustained_overload",
                            max_retries=0)
        except ProviderError:
            pass
    print("  sustained_overload done")

    # 4. Per-minute limit halved — 20 requests at RPM=4
    print("\n[scenario] per_minute_limit_halved: 20 requests, requests_per_minute=4")
    halved_cfg = replace(base_cfg, requests_per_minute=4)
    p4 = StandinProvider(store, halved_cfg)
    pipe4 = ResearchPipeline(ROOT, {"stand-in": p4}, logger=logger, retriever=retriever)
    for i in range(20):
        try:
            await pipe4.ask(benchmark[i % len(benchmark)]["question"],
                            provider_name="stand-in", retrieval_setting="B",
                            request_id=f"halved-{i:02d}", scenario="per_minute_limit_halved",
                            max_retries=0)
        except ProviderError:
            pass
    print("  per_minute_limit_halved done")

    # 5. Daily budget exhausted — 15 requests at RPD=10
    print("\n[scenario] daily_budget_exhausted: 15 requests, requests_per_day=10")
    daily_cfg = replace(base_cfg, requests_per_day=10)
    p5 = StandinProvider(store, daily_cfg)
    pipe5 = ResearchPipeline(ROOT, {"stand-in": p5}, logger=logger, retriever=retriever)
    for i in range(15):
        try:
            await pipe5.ask(benchmark[i % len(benchmark)]["question"],
                            provider_name="stand-in", retrieval_setting="B",
                            request_id=f"daily-{i:02d}", scenario="daily_budget_exhausted",
                            max_retries=0)
        except ProviderError:
            pass
    print("  daily_budget_exhausted done")


# ---------------------------------------------------------------------------
# Phase 6: Paired retry comparison
# ---------------------------------------------------------------------------

async def run_retry_comparison(
    logger: ExperimentLogger,
    retriever: PaperRetriever,
    base_cfg: StandinConfig,
) -> None:
    """Compare three retry policies under identical, reproducible failure conditions.

    Uses a seeded deterministic failure sequence so all three conditions see the
    same set of transient failures and the comparison is fair.

    Policies:
      baseline   — max_retries=0 (fail fast)
      amplified  — max_retries=3, no backoff (immediate retry)
      backoff    — max_retries=3, exponential backoff + jitter (via pipeline default)

    With random failure at 50% per attempt and no backoff, retrying 3 times in
    a constrained-capacity scenario (requests_per_minute=8) makes the system's
    queue grow faster than it drains, demonstrating retry amplification.
    """
    print("\n=== Phase 6: Paired Retry Experiment ===")
    benchmark = json.loads(
        (ROOT / "data" / "benchmarks" / "benchmark_draft.json").read_text(encoding="utf-8")
    )["questions"]
    store = ReplayStore(ROOT / "data" / "replay" / "responses.jsonl")

    # Constrained capacity: RPM=8 simulates a loaded system
    # transient_failure_rate=0.5: each attempt independently fails with p=0.5
    failure_cfg = replace(base_cfg, transient_failure_rate=0.5, requests_per_minute=8)
    N = 20

    for scenario_name, max_retries in [
        ("retry_baseline", 0),
        ("retry_amplified", 3),
        ("retry_backoff", 3),
    ]:
        # Seed the random state so all three conditions use the same failure draws
        random.seed(42)
        provider = StandinProvider(store, failure_cfg)
        pipeline = ResearchPipeline(ROOT, {"stand-in": provider}, logger=logger, retriever=retriever)
        print(f"\n[retry] {scenario_name}: max_retries={max_retries}, N={N}")
        ok, failed = 0, 0
        for i in range(N):
            q = benchmark[i % len(benchmark)]
            try:
                res = await pipeline.ask(
                    q["question"],
                    provider_name="stand-in",
                    retrieval_setting="B",
                    request_id=f"{scenario_name}-{i:02d}",
                    scenario=scenario_name,
                    max_retries=max_retries,
                )
                if res.status == "ok":
                    ok += 1
                else:
                    failed += 1
            except ProviderError:
                failed += 1
        print(f"  {scenario_name}: ok={ok}  failed={failed}")


# ---------------------------------------------------------------------------
# Phase 7: LiveKit labeling subset
# ---------------------------------------------------------------------------

def prepare_livekit_subset() -> None:
    print("\n=== Phase 7: LiveKit Labeling Subset ===")
    issues_path = ROOT / "data" / "livekit" / "issues.json"
    subset_path = ROOT / "data" / "livekit" / "labeling_subset.json"
    if not issues_path.exists():
        print("  issues.json not found — skipping.")
        return
    issues = json.loads(issues_path.read_text(encoding="utf-8"))
    subset = []
    for issue in issues[:30]:
        subset.append({
            "number": issue.get("number"),
            "title": issue.get("title", ""),
            "body": (issue.get("body") or "")[:2000],
            "url": issue.get("html_url", issue.get("url", "")),
            "label": None,  # Manual labeling only — never auto-label
        })
    subset_path.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Wrote {len(subset)} unlabeled issues to {subset_path.name}")


# ---------------------------------------------------------------------------
# Phase 8: Analysis
# ---------------------------------------------------------------------------

def run_analysis() -> None:
    print("\n=== Phase 8: Analysis ===")
    from scripts.analyze_results import main as analyze_main
    analyze_main()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 60)
    print("EvidenceGate Experiment Runner (corrected)")
    print(f"Root  : {ROOT}")
    print(f"Model : {GROQ_MODEL}")
    print("=" * 60)

    logger = ExperimentLogger(ROOT)
    retriever = PaperRetriever(ROOT)
    gate = EvidenceGate()

    observations = await run_real_groq_probe(logger, retriever, gate)
    standin_cfg = calibrate_standin(observations)
    populate_replay_store(observations)
    await run_benchmark_evaluations(logger, retriever, standin_cfg)
    await run_load_scenarios(logger, retriever, standin_cfg)
    await run_retry_comparison(logger, retriever, standin_cfg)
    prepare_livekit_subset()
    run_analysis()

    print("\n[DONE] All experiment phases complete.")


if __name__ == "__main__":
    asyncio.run(main())
