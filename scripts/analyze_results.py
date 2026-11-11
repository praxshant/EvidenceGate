"""Reproducible offline analysis for retrieval and available request records.

Latency reporting rules (enforced here, not assumed):
- TTFT percentiles are computed only over requests that produced a first token
  AND are computed on the same request population as total_ms.
- Error/rejection latency is reported as a separate statistic.
- No clamping of TTFT: if a row shows first_token_ms > total_ms it is flagged
  as a measurement anomaly rather than silently corrected.
- Real Groq percentiles use only rows where source == "real" AND first_token_ms
  is non-empty; total_ms percentiles use the same row IDs.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval import PaperRetriever, normalize_pdf_text

ROOT = Path(__file__).resolve().parents[1]
RESULT_COLUMNS = [
    "request_id", "scenario", "source", "start_time", "first_token_ms",
    "total_ms", "status", "retries", "tokens_in", "tokens_out",
    "retrieval_setting", "answer_grade",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_pdf_text(text)).strip().lower()


# ---------------------------------------------------------------------------
# Retrieval recall — split by answerability category
# ---------------------------------------------------------------------------

def retrieval_recall() -> dict[str, object]:
    """Return retrieval recall broken down by answerability category.

    The frozen benchmark has:
      - 10 fully answerable questions (q01–q10)
      -  5 partially supported questions (q11–q15)
      -  5 unanswerable questions (q16–q20)

    Retrieval recall is measured only over questions with gold evidence
    (the 10 answerable + 5 partially supported = 15).  Unanswerable questions
    have no gold passage so their retrieval output cannot be graded for recall.

    Recall is defined at question level: a question is recalled if at least one
    retrieved passage matches (same paper, same page, gold passage is a
    substring of passage text after normalization).  Passage-level recall would
    require multiple gold passages per question; the current benchmark has one
    gold passage per answerable question so question-level == passage-level here.
    """
    benchmark = json.loads(
        (ROOT / "data" / "benchmarks" / "benchmark_draft.json").read_text(encoding="utf-8")
    )["questions"]
    retriever = PaperRetriever(ROOT)

    result: dict[str, object] = {}

    for setting in ("A", "B", "C"):
        answerable_hits = 0
        answerable_total = 0
        partial_hits = 0
        partial_total = 0

        for q in benchmark:
            answerability = q["expected_answerability"]
            if answerability not in {"answerable", "partially_supported"}:
                continue

            gold_evidence = normalize(q["gold_supporting_passage"])
            passages = retriever.search(q["question"], setting)
            hit = any(
                p.paper == q["gold_paper"]
                and p.page == q["gold_page"]
                and gold_evidence in normalize(p.text)
                for p in passages
            )

            if answerability == "answerable":
                answerable_total += 1
                if hit:
                    answerable_hits += 1
            else:  # partially_supported
                partial_total += 1
                if hit:
                    partial_hits += 1

        total_with_gold = answerable_total + partial_total
        total_hits = answerable_hits + partial_hits

        result[setting] = {
            # Fully answerable (q01-q10)
            "answerable_questions": answerable_total,
            "answerable_recalled": answerable_hits,
            "answerable_recall_percent": (
                round(100 * answerable_hits / answerable_total, 1)
                if answerable_total else 0.0
            ),
            # Partially supported (q11-q15)
            "partially_supported_questions": partial_total,
            "partially_supported_recalled": partial_hits,
            "partially_supported_recall_percent": (
                round(100 * partial_hits / partial_total, 1)
                if partial_total else 0.0
            ),
            # Combined recall (all questions with gold evidence)
            "gold_evidence_questions": total_with_gold,
            "gold_evidence_recalled": total_hits,
            "gold_evidence_recall_percent": (
                round(100 * total_hits / total_with_gold, 1)
                if total_with_gold else 0.0
            ),
            "note": (
                "Recall is question-level (>=1 passage matches gold paper+page+text). "
                "Unanswerable questions (q16-q20) have no gold passage and are excluded."
            ),
        }

    return result


# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------

def percentile(values: list[float], quantile: float) -> float | str:
    if not values:
        return "not measured"
    values = sorted(values)
    index = min(len(values) - 1, round((len(values) - 1) * quantile))
    return round(values[index], 1)


def _flag_latency_anomalies(rows: list[dict[str, str]]) -> list[str]:
    """Return a list of anomaly descriptions for rows where total < first_token."""
    anomalies = []
    for r in rows:
        ttft_s = r.get("first_token_ms")
        total_s = r.get("total_ms")
        if ttft_s and total_s:
            ttft = float(ttft_s)
            total = float(total_s)
            if total < ttft:
                anomalies.append(
                    f"request_id={r.get('request_id')} scenario={r.get('scenario')} "
                    f"total_ms={total} < first_token_ms={ttft}"
                )
    return anomalies


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def request_table() -> list[dict[str, str]]:
    path = ROOT / "results.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and list(rows[0]) != RESULT_COLUMNS:
        raise ValueError("results.csv columns do not match the required schema")
    return rows


# ---------------------------------------------------------------------------
# Scenario summary with correct paired latency populations
# ---------------------------------------------------------------------------

def scenario_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Compute per-scenario performance statistics.

    Population rules:
    - Completed requests = rows where status == "ok".
    - TTFT and total_ms percentiles for completed requests are computed on the
      SAME set of rows (rows that have a non-empty first_token_ms AND status ok).
      This avoids mixing different populations that caused the TTFT > total
      inversion in the previous run.
    - Error latency (rate_limited, timeout, error) is reported separately.
    - Success rate = ok / total  (includes refused-correctly as "ok").
    - Gated_pipeline_success = (ok) / total.
    - Answerable_accuracy is not computable from the CSV schema alone (requires
      knowing which requests were for answerable questions); it is computed in
      the benchmark section instead.
    """
    scenarios = [
        "steady_load", "sudden_burst", "sustained_overload",
        "per_minute_limit_halved", "daily_budget_exhausted",
    ]
    output = []
    for scenario in scenarios:
        selected = [r for r in rows if r.get("scenario") == scenario]
        total = len(selected)
        if not total:
            output.append({
                "scenario": scenario,
                "requests": "not measured",
                "note": "No rows recorded for this scenario",
            })
            continue

        ok_rows = [r for r in selected if r.get("status") == "ok"]
        rate_limited_rows = [r for r in selected if r.get("status") == "rate_limited"]
        timeout_rows = [r for r in selected if r.get("status") == "timeout"]
        error_rows = [r for r in selected if r.get("status") not in {"ok", "rate_limited", "timeout"}]
        failed_rows = [r for r in selected if r.get("status") != "ok"]

        # Paired population: completed rows that have both first_token_ms and total_ms
        paired_rows = [
            r for r in ok_rows
            if r.get("first_token_ms") and r.get("total_ms")
        ]
        paired_ttft = [float(r["first_token_ms"]) for r in paired_rows]
        paired_total = [float(r["total_ms"]) for r in paired_rows]

        # All-ok total_ms (includes ok rows without TTFT, e.g. correctly refused)
        all_ok_total = [float(r["total_ms"]) for r in ok_rows if r.get("total_ms")]

        # Failed request latency
        failed_total = [float(r["total_ms"]) for r in failed_rows if r.get("total_ms")]

        output.append({
            "scenario": scenario,
            "requests_total": total,
            "requests_ok": len(ok_rows),
            "requests_rate_limited": len(rate_limited_rows),
            "requests_timeout": len(timeout_rows),
            "requests_error": len(error_rows),
            # Success = ok / total
            "gated_pipeline_success_percent": round(100 * len(ok_rows) / total, 1),
            "rate_limited_percent": round(100 * len(rate_limited_rows) / total, 1),
            "timeout_percent": round(100 * len(timeout_rows) / total, 1),
            # Latency over paired (same-population) completed requests with TTFT
            "paired_population_n": len(paired_rows),
            "ttft_p50_ms": percentile(paired_ttft, 0.50),
            "ttft_p95_ms": percentile(paired_ttft, 0.95),
            "total_p50_ms": percentile(paired_total, 0.50),
            "total_p95_ms": percentile(paired_total, 0.95),
            # All-ok end-to-end (broader population, may include no-TTFT refused)
            "all_ok_total_p50_ms": percentile(all_ok_total, 0.50),
            "all_ok_total_p95_ms": percentile(all_ok_total, 0.95),
            # Error response latency (rate-limit 429s, timeouts, etc.)
            "failed_duration_p50_ms": percentile(failed_total, 0.50),
            "failed_duration_p95_ms": percentile(failed_total, 0.95),
            "population_note": (
                "ttft and total percentiles use paired rows (same N). "
                "all_ok_total includes refused requests that have no TTFT."
            ),
        })
    return output


# ---------------------------------------------------------------------------
# Real Groq measurement summary (paired, same request IDs)
# ---------------------------------------------------------------------------

def real_groq_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    """Compute latency summary for real Groq calls using paired populations."""
    real_rows = [r for r in rows if r.get("source") == "real"]
    if not real_rows:
        return "not measured: no real Groq rows in results.csv"

    # Paired: rows that have both first_token_ms and total_ms
    paired = [r for r in real_rows if r.get("first_token_ms") and r.get("total_ms")]
    paired_ttft = [float(r["first_token_ms"]) for r in paired]
    paired_total = [float(r["total_ms"]) for r in paired]

    # All-real total_ms (may include requests without TTFT)
    all_total = [float(r["total_ms"]) for r in real_rows if r.get("total_ms")]

    # Token accounting
    tokens_in = [int(r["tokens_in"]) for r in real_rows if r.get("tokens_in")]
    tokens_out = [int(r["tokens_out"]) for r in real_rows if r.get("tokens_out")]

    # Latency anomaly check: flag but do not clamp
    anomalies = _flag_latency_anomalies(real_rows)

    return {
        "requests_real": len(real_rows),
        "requests_with_ttft": len(paired),
        # Paired population (same N for TTFT and total)
        "paired_ttft_p50_ms": percentile(paired_ttft, 0.50),
        "paired_ttft_p95_ms": percentile(paired_ttft, 0.95),
        "paired_total_p50_ms": percentile(paired_total, 0.50),
        "paired_total_p95_ms": percentile(paired_total, 0.95),
        # All-real total_ms
        "all_real_total_p50_ms": percentile(all_total, 0.50),
        "all_real_total_p95_ms": percentile(all_total, 0.95),
        "tokens_in_sum": sum(tokens_in) if tokens_in else "not measured",
        "tokens_out_sum": sum(tokens_out) if tokens_out else "not measured",
        "latency_anomalies": anomalies if anomalies else [],
        "population_note": (
            "paired_* uses only rows where both first_token_ms and total_ms are "
            "non-empty. all_real_total includes all real rows. "
            "latency_anomalies lists rows where total_ms < first_token_ms."
        ),
    }


# ---------------------------------------------------------------------------
# Benchmark accuracy summary (by answerability category)
# ---------------------------------------------------------------------------

def benchmark_accuracy_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    """Report pipeline accuracy over benchmark scenarios split by answerability.

    Answerable questions: q01-q10. Partially supported: q11-q15. Unanswerable: q16-q20.
    Success definitions:
      - Answerable/partial: status == "ok" (verified answer produced)
      - Unanswerable: status == "ok" with the pipeline refusing to guess
        (EvidenceGate returns insufficient; answer is a refusal string)
    """
    bench_rows = [
        r for r in rows
        if r.get("scenario", "").startswith("benchmark_setting_")
    ]
    if not bench_rows:
        return {"note": "No benchmark rows in results.csv"}

    settings_seen = sorted({r["scenario"].replace("benchmark_setting_", "") for r in bench_rows})
    result: dict[str, object] = {}

    # Answerable: q01-q10
    answerable_ids = {f"q{str(i).zfill(2)}" for i in range(1, 11)}
    # Partially supported: q11-q15
    partial_ids = {f"q{str(i).zfill(2)}" for i in range(11, 16)}
    # Unanswerable: q16-q20
    unanswerable_ids = {f"q{str(i).zfill(2)}" for i in range(16, 21)}

    for setting in settings_seen:
        setting_rows = [r for r in bench_rows if r.get("scenario") == f"benchmark_setting_{setting}"]

        def _acc(id_set: set[str]) -> dict:
            subset = [r for r in setting_rows if r.get("request_id", "").endswith(f"-{r.get('request_id','').split('-')[-1]}")]
            # Match by request_id suffix (bench-A-q01 → q01)
            subset = [r for r in setting_rows if r.get("request_id", "").split("-")[-1] in id_set]
            ok = [r for r in subset if r.get("status") == "ok"]
            return {
                "total": len(subset),
                "ok": len(ok),
                "accuracy_percent": round(100 * len(ok) / len(subset), 1) if subset else "not measured",
            }

        result[setting] = {
            "answerable_accuracy": _acc(answerable_ids),
            "partially_supported_accuracy": _acc(partial_ids),
            "unanswerable_refusal_rate": _acc(unanswerable_ids),
            "note": (
                "Accuracy = status=='ok' / total for that category. "
                "For unanswerable, 'ok' means EvidenceGate correctly refused."
            ),
        }

    return result


# ---------------------------------------------------------------------------
# Retry comparison summary
# ---------------------------------------------------------------------------

def retry_comparison_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    """Summarise the paired retry amplification experiment.

    Expects rows from scenarios 'retry_baseline' (0 retries) and
    'retry_amplified' (3 retries) under 60% transient failure rate.
    """
    baseline = [r for r in rows if r.get("scenario") == "retry_baseline"]
    amplified = [r for r in rows if r.get("scenario") == "retry_amplified"]
    backoff_rows = [r for r in rows if r.get("scenario") == "retry_backoff"]

    def _stats(subset: list[dict]) -> dict:
        if not subset:
            return {"note": "not measured"}
        ok = [r for r in subset if r.get("status") == "ok"]
        total_ms = [float(r["total_ms"]) for r in subset if r.get("total_ms")]
        retries = [int(r.get("retries") or 0) for r in subset]
        return {
            "requests": len(subset),
            "successes": len(ok),
            "success_rate_percent": round(100 * len(ok) / len(subset), 1),
            "total_attempts": sum(retries) + len(subset),
            "total_p50_ms": percentile(total_ms, 0.50),
            "total_p95_ms": percentile(total_ms, 0.95),
        }

    b_stats = _stats(baseline)
    a_stats = _stats(amplified)
    bk_stats = _stats(backoff_rows)

    result: dict[str, object] = {
        "retry_baseline_0_retries": b_stats,
        "retry_amplified_3_retries": a_stats,
        "retry_backoff_jitter": bk_stats,
    }

    if isinstance(b_stats, dict) and isinstance(a_stats, dict) and "total_attempts" in b_stats and "total_attempts" in a_stats:
        result["amplification_factor_attempts"] = round(
            a_stats["total_attempts"] / b_stats["total_attempts"], 2
        ) if b_stats["total_attempts"] else "n/a"

    return result


# ---------------------------------------------------------------------------
# Rate-limit header reconciliation log
# ---------------------------------------------------------------------------

def header_reconciliation() -> list[dict]:
    """Read raw_headers.jsonl and pair with results.csv token usage.

    Reports observed header values alongside actual token counts for each real
    request. Does NOT assert a specific refill formula — records facts only.
    """
    headers_path = ROOT / "raw_headers.jsonl"
    csv_rows = {r["request_id"]: r for r in request_table() if r.get("source") == "real"}

    if not headers_path.exists():
        return []

    records = []
    with headers_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rid = entry.get("request_id", "")
            csv_row = csv_rows.get(rid, {})
            tokens_in = int(csv_row.get("tokens_in") or 0)
            tokens_out = int(csv_row.get("tokens_out") or 0)
            total_tokens = tokens_in + tokens_out

            hdrs = entry.get("headers", {})
            rem_tokens = hdrs.get("x-ratelimit-remaining-tokens")
            lim_tokens = hdrs.get("x-ratelimit-limit-tokens")

            records.append({
                "request_id": rid,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "total_tokens_used": total_tokens,
                "header_remaining_tokens": rem_tokens,
                "header_limit_tokens": lim_tokens,
                "header_reset_tokens": hdrs.get("x-ratelimit-reset-tokens"),
                "header_remaining_requests": hdrs.get("x-ratelimit-remaining-requests"),
                "header_reset_requests": hdrs.get("x-ratelimit-reset-requests"),
                "note": (
                    "header_remaining_tokens reflects Groq's instantaneous TPM bucket "
                    "state at response time, which includes per-minute refill. "
                    "Direct subtraction from limit is not meaningful without knowing "
                    "the elapsed refill interval. Groq docs do not publish the exact "
                    "refill equation."
                ),
            })

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows = request_table()

    # Latency anomaly check across all rows
    all_anomalies = _flag_latency_anomalies(rows)
    if all_anomalies:
        print(f"[WARNING] {len(all_anomalies)} latency anomaly rows (total_ms < first_token_ms):")
        for a in all_anomalies:
            print(f"  {a}")

    standin_cfg_path = ROOT / "data" / "standin" / "default.json"
    standin_cal = "not calibrated"
    if standin_cfg_path.exists():
        cfg = json.loads(standin_cfg_path.read_text(encoding="utf-8"))
        standin_cal = {
            "calibration_status": cfg.get("calibration_status", "uncalibrated"),
            "base_delay_seconds": cfg.get("base_delay_seconds"),
            "jitter_seconds": cfg.get("jitter_seconds"),
            "inter_chunk_seconds": cfg.get("inter_chunk_seconds"),
            "calibration_n": cfg.get("calibration_n", "unknown"),
            "calibration_note": cfg.get("calibration_note", ""),
        }

    livekit_count = "not collected"
    lk_path = ROOT / "data" / "livekit" / "issues.json"
    if lk_path.exists():
        livekit_count = len(json.loads(lk_path.read_text(encoding="utf-8")))

    payload = {
        "retrieval_recall": retrieval_recall(),
        "benchmark_accuracy": benchmark_accuracy_summary(rows),
        "scenario_summary": scenario_summary(rows),
        "retry_comparison": retry_comparison_summary(rows),
        "real_groq_measurements": real_groq_summary(rows),
        "header_reconciliation": header_reconciliation(),
        "standin_calibration": standin_cal,
        "results_rows_total": len(rows),
        "latency_anomalies_total": len(all_anomalies),
        "livekit_records": livekit_count,
    }

    out = ROOT / "data" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "milestone_4_5_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
