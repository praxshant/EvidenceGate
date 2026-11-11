"""Reproducible offline analysis for retrieval and available request records."""

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


def retrieval_recall() -> dict[str, dict[str, int | float]]:
    benchmark = json.loads(
        (ROOT / "data" / "benchmarks" / "benchmark_draft.json").read_text(encoding="utf-8")
    )["questions"]
    retriever = PaperRetriever(ROOT)
    summary: dict[str, dict[str, int | float]] = {}
    for setting in ("A", "B", "C"):
        hits = 0
        eligible = 0
        for question in benchmark:
            if question["expected_answerability"] not in {"answerable", "partially_supported"}:
                continue
            eligible += 1
            evidence = normalize(question["gold_supporting_passage"])
            passages = retriever.search(question["question"], setting)
            if any(
                p.paper == question["gold_paper"]
                and p.page == question["gold_page"]
                and evidence in normalize(p.text)
                for p in passages
            ):
                hits += 1
        summary[setting] = {
            "gold_evidence_eligible": eligible,
            "gold_evidence_recalled": hits,
            "recall_percent": round(100 * hits / eligible, 1) if eligible else 0.0,
        }
    return summary


def percentile(values: list[float], quantile: float) -> float | str:
    if not values:
        return "not measured"
    values = sorted(values)
    index = min(len(values) - 1, round((len(values) - 1) * quantile))
    return round(values[index], 1)


def request_table() -> list[dict[str, str]]:
    path = ROOT / "results.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and list(rows[0]) != RESULT_COLUMNS:
        raise ValueError("results.csv columns do not match the required schema")
    return rows


def scenario_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    scenarios = [
        "steady_load", "sudden_burst", "sustained_overload",
        "per_minute_limit_halved", "daily_budget_exhausted",
    ]
    output = []
    for scenario in scenarios:
        selected = [r for r in rows if r.get("scenario") == scenario]
        total = len(selected)
        successes = sum(r.get("status") == "ok" for r in selected)
        rate_limited = sum(r.get("status") == "rate_limited" for r in selected)
        timeouts = sum(r.get("status") == "timeout" for r in selected)
        ttft = [float(r["first_token_ms"]) for r in selected if r.get("first_token_ms")]
        total_ms = [float(r["total_ms"]) for r in selected if r.get("total_ms")]
        output.append({
            "scenario": scenario,
            "requests": total or "not measured",
            "success_percent": round(100 * successes / total, 1) if total else "not measured",
            "rate_limited_percent": round(100 * rate_limited / total, 1) if total else "not measured",
            "timeout_percent": round(100 * timeouts / total, 1) if total else "not measured",
            "ttft_p50_ms": percentile(ttft, 0.50),
            "ttft_p95_ms": percentile(ttft, 0.95),
            "ttft_p99_ms": percentile(ttft, 0.99),
            "total_p50_ms": percentile(total_ms, 0.50),
            "total_p95_ms": percentile(total_ms, 0.95),
            "total_p99_ms": percentile(total_ms, 0.99),
        })
    return output


def main() -> None:
    rows = request_table()
    real_rows = [r for r in rows if r.get("source") == "real"]
    if real_rows:
        real_ttft = [float(r["first_token_ms"]) for r in real_rows if r.get("first_token_ms")]
        real_total = [float(r["total_ms"]) for r in real_rows if r.get("total_ms")]
        real_measurements: dict[str, object] = {
            "requests": len(real_rows),
            "ttft_p50_ms": percentile(real_ttft, 0.50),
            "ttft_p95_ms": percentile(real_ttft, 0.95),
            "total_p50_ms": percentile(real_total, 0.50),
            "total_p95_ms": percentile(real_total, 0.95),
        }
    else:
        real_measurements = "not measured: credentials are not configured"

    standin_cfg_path = ROOT / "data" / "standin" / "default.json"
    if standin_cfg_path.exists():
        cfg = json.loads(standin_cfg_path.read_text(encoding="utf-8"))
        standin_cal = {
            "calibration_status": cfg.get("calibration_status", "uncalibrated"),
            "base_delay_seconds": cfg.get("base_delay_seconds"),
            "inter_chunk_seconds": cfg.get("inter_chunk_seconds"),
            "jitter_seconds": cfg.get("jitter_seconds"),
        }
    else:
        standin_cal = "provisional: no recorded real Groq responses"

    payload = {
        "retrieval_recall": retrieval_recall(),
        "scenario_summary": scenario_summary(rows),
        "results_rows": len(rows),
        "livekit_records": len(json.loads(
            (ROOT / "data" / "livekit" / "issues.json").read_text(encoding="utf-8")
        )),
        "real_groq_measurements": real_measurements,
        "standin_calibration": standin_cal,
    }
    out = ROOT / "data" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "milestone_4_5_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
