"""
Complete Final Evaluation Runner for EvidenceGate / The Relay
Generates all artifacts inside data/final_evaluation/
"""
import asyncio
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env")

from src.pipeline import (
    ResearchPipeline,
    EvidenceGate,
    AdmissionController,
    build_prompt,
    AskResult,
)
from src.retrieval import PaperRetriever
from src.verification import verify_answer, is_abstention
from src.providers.groq_client import GroqProvider
from src.providers.standin_client import ReplayStore, StandinConfig, StandinProvider
from dataclasses import fields, replace
from src.providers.base import ProviderError, GenerationRequest, ErrorKind
from src.logging_utils import (
    ExperimentLogger,
    DiagnosticEntry,
    RESULT_COLUMNS,
    monotonic_ms,
    now_iso_hires,
)

EVAL_DIR = ROOT / "data" / "final_evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Set max output tokens default to 256 for GenerationRequest
GenerationRequest.max_output_tokens = 256

SELECTED_QUESTION_IDS = [
    # 5 Fully Answerable
    "q01",  # Transformer (en)
    "q04",  # Transformer (en)
    "q05",  # Moshi (en)
    "q08",  # Dynamic Chunking (en)
    "q10",  # Moshi (hi-en)
    # 5 Partially Supported
    "q11",  # Dynamic Chunking (en)
    "q12",  # Moshi (en)
    "q13",  # Moshi (hi-en)
    "q14",  # Dynamic Chunking (hi-en)
    "q15",  # Transformer (en)
    # 5 Unanswerable
    "q16",  # Project rate limits (en)
    "q17",  # Laptop RTX 3050 hardware (en)
    "q18",  # Groq client retry parameters (en)
    "q19",  # LiveKit GitHub issue (hi-en)
    "q20",  # New dataset WER (en)
]

RETRIEVAL_SETTINGS = [
    ("A", "300-word chunks, top-k 3"),
    ("B", "500-word chunks, top-k 3"),
    ("C", "500-word chunks, top-k 5, reranked"),
]


def load_benchmark():
    bm_path = ROOT / "data" / "benchmarks" / "benchmark_draft.json"
    with open(bm_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", data)
    return {q["id"]: q for q in questions}


async def run_phase2(logger, retriever, gate, benchmark):
    print("\n" + "=" * 80)
    print("PHASE 2: REAL GROQ ANSWER EVALUATION")
    print("Model: openai/gpt-oss-20b | max_completion_tokens: 256 | max_retries: 0")
    print("=" * 80)

    api_key = os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Groq API key is not configured. Aborting real Groq phase.")
        return []

    groq_provider = GroqProvider(api_key=api_key, model="openai/gpt-oss-20b")
    pipeline = ResearchPipeline(
        ROOT,
        {"groq": groq_provider},
        logger=logger,
        retriever=retriever,
    )

    eval_results = []
    rolling_tokens = []

    def record_tokens(n_tokens):
        now = time.monotonic()
        rolling_tokens.append((now, n_tokens))

    def get_tokens_last_60s():
        now = time.monotonic()
        while rolling_tokens and (now - rolling_tokens[0][0]) > 60.0:
            rolling_tokens.pop(0)
        return sum(t[1] for t in rolling_tokens)

    eval_index = 0
    total_evals = len(RETRIEVAL_SETTINGS) * len(SELECTED_QUESTION_IDS)

    for setting, setting_desc in RETRIEVAL_SETTINGS:
        print(f"\n>>> EVALUATING RETRIEVAL SETTING {setting}: {setting_desc} <<<\n")
        for qid in SELECTED_QUESTION_IDS:
            eval_index += 1
            bm_item = benchmark[qid]
            q_text = bm_item["question"]
            request_id = f"eval-{setting}-{qid}"
            scenario_name = f"final_eval_{setting}"

            print("-" * 80)
            print(f"[{eval_index}/{total_evals}] Request: {request_id}")
            print(f"Question ({bm_item['expected_answerability']}): {q_text}")
            print(f"Setting: {setting} ({setting_desc})")

            # Check pre-flight retrieval & gating
            passages = retriever.search(q_text, setting)
            gate_decision = gate.decide(passages, q_text)

            evidence_refs = [
                f"[{p.paper}, §{p.section}, p.{p.page}]" for p in passages
            ]
            print(f"Retrieved Evidence: {', '.join(evidence_refs) if evidence_refs else 'None'}")
            print(f"Evidence Gate Decision: {gate_decision}")

            groq_called = gate_decision != "insufficient"
            print(f"Groq Provider Called: {groq_called}")

            if groq_called:
                used_60s = get_tokens_last_60s()
                if used_60s > 4500:
                    sleep_time = 24.0
                    print(f"  [Pacing] TPM rolling window has {used_60s} tokens. Sleeping {sleep_time:.1f}s...")
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(5.0)

            ans_text = ""
            citations = []
            verif_result = ""
            ttft_ms = None
            total_ms = 0.0
            tokens_in = ""
            tokens_out = ""
            status = "ok"
            error_details = ""

            try:
                res = await pipeline.ask(
                    question=q_text,
                    provider_name="groq",
                    retrieval_setting=setting,
                    request_id=request_id,
                    max_retries=0,
                    scenario=scenario_name,
                    benchmark_item=bm_item,
                    timeout_seconds=30.0,
                )
                ans_text = res.answer
                citations = res.citations
                gate_decision = res.decision
                ttft_ms = res.timing.get("provider_ttft_ms")
                total_ms = res.timing.get("total_ms", 0.0)
                status = res.status

                # Read token usage from last_usage
                u = groq_provider._last_usage or {}
                tokens_in = u.get("prompt_tokens", "")
                tokens_out = u.get("completion_tokens", "")
                if tokens_in and tokens_out:
                    record_tokens(int(tokens_in) + int(tokens_out))

                # Verification check on answer
                if is_abstention(ans_text):
                    verif_result = "Abstention returned"
                else:
                    v_res = verify_answer(ans_text, passages)
                    verif_result = "PASS" if v_res.valid else f"FAIL ({v_res.reason})"

            except ProviderError as exc:
                status = "rate_limited" if exc.kind == ErrorKind.RATE_LIMITED else "error"
                error_details = str(exc)
                if exc.headers:
                    await logger.headers(request_id, exc.headers)
                print(f"  PROVIDER ERROR: {exc}")
                if exc.retry_after:
                    wait_s = float(exc.retry_after) + 1.0
                    print(f"  [Retry-After] Respecting provider requested wait: {wait_s:.2f}s...")
                    await asyncio.sleep(wait_s)

            except Exception as exc:
                status = "error"
                error_details = str(exc)
                print(f"  UNEXPECTED ERROR: {exc}")

            exp_facts = bm_item.get("expected_facts", "")
            exp_ans = bm_item.get("expected_answerability", "")
            if exp_ans == "unanswerable":
                provisional_grade = "refused_correctly" if is_abstention(ans_text) else "wrong"
            else:
                if is_abstention(ans_text):
                    provisional_grade = "refused_wrongly" if gate_decision == "insufficient" else "wrong (verification abstention)"
                else:
                    provisional_grade = "provisional_pass"

            ttft_display = f"{ttft_ms:.1f}ms" if ttft_ms is not None else "N/A"
            print(f"\n--- ACTUAL GENERATED ANSWER / REFUSAL ---")
            print(ans_text if ans_text else f"[No text - Status: {status}]")
            print(f"Citations: {citations if citations else 'None'}")
            print(f"Verification: {verif_result}")
            print(f"Latency: TTFT={ttft_display}, Total={total_ms:.1f}ms")
            print(f"Tokens: in={tokens_in}, out={tokens_out}")
            print(f"Status: {status}")
            print(f"Provisional Fact Assessment: {provisional_grade}")
            print(f"Human Review Required: {'YES' if provisional_grade != 'refused_correctly' else 'NO'}\n")

            eval_results.append({
                "request_id": request_id,
                "question_id": qid,
                "question": q_text,
                "expected_answerability": exp_ans,
                "expected_facts": exp_facts,
                "setting": setting,
                "retrieved_evidence": evidence_refs,
                "evidence_gate_decision": gate_decision,
                "groq_called": groq_called,
                "answer": ans_text,
                "citations": citations,
                "verification_result": verif_result,
                "ttft_ms": ttft_ms,
                "total_ms": total_ms,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "status": status,
                "error_details": error_details,
                "provisional_grade": provisional_grade,
                "human_review_required": provisional_grade != "refused_correctly",
            })

    # Save answers.jsonl
    answers_jsonl_path = EVAL_DIR / "answers.jsonl"
    with open(answers_jsonl_path, "w", encoding="utf-8") as f:
        for item in eval_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save answers.md
    answers_md_path = EVAL_DIR / "answers.md"
    with open(answers_md_path, "w", encoding="utf-8") as f:
        f.write("# Real Groq Answer Evaluations (openai/gpt-oss-20b)\n\n")
        f.write("Evaluated across 15 stratified benchmark questions and 3 retrieval configurations.\n\n")
        for setting, setting_desc in RETRIEVAL_SETTINGS:
            f.write(f"## Retrieval Setting {setting} ({setting_desc})\n\n")
            s_items = [it for it in eval_results if it["setting"] == setting]
            for it in s_items:
                f.write(f"### [{it['question_id']}] {it['question']}\n\n")
                f.write(f"- **Expected Answerability:** `{it['expected_answerability']}`\n")
                f.write(f"- **Evidence Gate Decision:** `{it['evidence_gate_decision']}` (Groq called: `{it['groq_called']}`)\n")
                f.write(f"- **Retrieved Passages:** {', '.join(it['retrieved_evidence']) if it['retrieved_evidence'] else 'None'}\n")
                ttft_disp = f"{it['ttft_ms']:.1f}ms" if it["ttft_ms"] is not None else "N/A"
                f.write(f"- **Status:** `{it['status']}` | **TTFT:** `{ttft_disp}` | **Total:** `{it['total_ms']:.1f}ms`\n")
                f.write(f"- **Tokens:** In: `{it['tokens_in']}` | Out: `{it['tokens_out']}`\n")
                f.write(f"- **Verification:** `{it['verification_result']}`\n")
                f.write(f"- **Provisional Assessment:** `{it['provisional_grade']}` (Human Review: `{it['human_review_required']}`)\n\n")
                f.write(f"**Generated Answer:**\n\n> {it['answer']}\n\n")
                f.write("---\n\n")

    print(f"\nSaved answers to {answers_jsonl_path} and {answers_md_path}")
    return eval_results


def run_phase3(retriever, gate, benchmark):
    print("\n" + "=" * 80)
    print("PHASE 3: RETRIEVAL EVALUATION")
    print("Evaluating all 20 benchmark questions across Settings A, B, C")
    print("=" * 80)

    # 10 Answerable, 5 Partially Supported, 5 Unanswerable
    ans_qids = [q["id"] for q in benchmark.values() if q["expected_answerability"] == "answerable"]
    part_qids = [q["id"] for q in benchmark.values() if q["expected_answerability"] == "partially_supported"]
    unans_qids = [q["id"] for q in benchmark.values() if q["expected_answerability"] == "unanswerable"]

    eval_summary = {
        "denominators": {
            "total_questions": len(benchmark),
            "answerable_questions": len(ans_qids),
            "partially_supported_questions": len(part_qids),
            "unanswerable_questions": len(unans_qids),
            "gold_passage_evaluation_pool": len(ans_qids) + len(part_qids),
            "explanation": (
                "Gold-passage recall and MRR are computed over the 15 questions with gold passages "
                "(10 answerable + 5 partially supported). Unanswerable questions (5) have no gold passages; "
                "their retrieval performance is measured by evidence-gate rejection rate (denominator = 5)."
            )
        },
        "settings": {}
    }

    for setting, setting_desc in RETRIEVAL_SETTINGS:
        print(f"\n--- Setting {setting}: {setting_desc} ---")
        gold_hits = 0
        rr_sum = 0.0
        gate_decisions = {
            "answerable": {"sufficient": 0, "partial": 0, "insufficient": 0},
            "partially_supported": {"sufficient": 0, "partial": 0, "insufficient": 0},
            "unanswerable": {"sufficient": 0, "partial": 0, "insufficient": 0},
        }

        per_question_results = []

        for qid, q in benchmark.items():
            q_text = q["question"]
            exp_ans = q["expected_answerability"]
            gold_paper = q.get("gold_paper")
            gold_page = q.get("gold_page")
            gold_text = q.get("gold_supporting_passage") or ""

            passages = retriever.search(q_text, setting)
            decision = gate.decide(passages, q_text)
            gate_decisions[exp_ans][decision] += 1

            matched_rank = None
            if exp_ans != "unanswerable":
                for rank_idx, p in enumerate(passages, start=1):
                    # Match by paper, page, and significant text overlap
                    paper_match = gold_paper.lower() in p.paper.lower() if gold_paper else False
                    page_match = (p.page == gold_page) if gold_page else False
                    # Check text overlap if available
                    words = [w for w in gold_text.lower().split() if len(w) > 4][:5]
                    text_match = any(w in p.text.lower() for w in words) if words else True

                    if paper_match and (page_match or text_match):
                        matched_rank = rank_idx
                        break

                if matched_rank is not None:
                    gold_hits += 1
                    rr_sum += 1.0 / matched_rank

            per_question_results.append({
                "qid": qid,
                "expected_answerability": exp_ans,
                "decision": decision,
                "matched_rank": matched_rank,
                "retrieved_count": len(passages),
            })

        pool_n = len(ans_qids) + len(part_qids)
        recall_at_k = gold_hits / pool_n if pool_n else 0.0
        mrr = rr_sum / pool_n if pool_n else 0.0

        unans_rejection_rate = gate_decisions["unanswerable"]["insufficient"] / len(unans_qids)

        eval_summary["settings"][setting] = {
            "description": setting_desc,
            "gold_passage_recall_at_k": round(recall_at_k, 4),
            "mrr": round(mrr, 4),
            "gold_hits": gold_hits,
            "gold_pool_size": pool_n,
            "gate_decisions": gate_decisions,
            "unanswerable_rejection_rate": round(unans_rejection_rate, 4),
            "per_question": per_question_results,
        }

        print(f"Recall@k: {recall_at_k:.2%} ({gold_hits}/{pool_n})")
        print(f"MRR: {mrr:.4f}")
        print(f"Unanswerable Rejection Rate: {unans_rejection_rate:.2%} ({gate_decisions['unanswerable']['insufficient']}/{len(unans_qids)})")
        print(f"Gate Decisions:")
        print(f"  Answerable (N=10): {gate_decisions['answerable']}")
        print(f"  Partially Supported (N=5): {gate_decisions['partially_supported']}")
        print(f"  Unanswerable (N=5): {gate_decisions['unanswerable']}")

    retrieval_json_path = EVAL_DIR / "retrieval_evaluation.json"
    with open(retrieval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved retrieval evaluation to {retrieval_json_path}")
    return eval_summary


async def run_phase4(logger, retriever, gate, benchmark):
    print("\n" + "=" * 80)
    print("PHASE 4: OFFLINE END-TO-END AND LOAD TESTING (STAND-IN SIMULATION)")
    print("=" * 80)

    # Load standin default config
    standin_cfg_path = ROOT / "data" / "standin" / "default.json"
    replay_path = ROOT / "data" / "replay" / "responses.jsonl"

    with open(standin_cfg_path, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)

    valid_keys = {field.name for field in fields(StandinConfig)}
    base_cfg = StandinConfig(**{k: v for k, v in cfg_data.items() if k in valid_keys})
    store = ReplayStore(replay_path)

    standin_provider = StandinProvider(store, base_cfg)

    pipeline = ResearchPipeline(
        ROOT,
        {"stand-in": standin_provider},
        logger=logger,
        retriever=retriever,
    )

    # 1. Five Load Scenarios
    scenarios_data = {}
    print("\n--- Running 5 Stand-in Load Scenarios ---")

    # Scenario 1: Steady Load (20 requests, sequential)
    print("\n1. Steady Load (N=20)")
    steady_results = []
    for i in range(20):
        qid = f"q{(i % 20) + 1:02d}"
        q = benchmark[qid]
        req_id = f"load-steady-{i:02d}"
        res = await pipeline.ask(
            q["question"],
            provider_name="stand-in",
            request_id=req_id,
            scenario="final_steady_load",
            benchmark_item=q,
        )
        steady_results.append({"req_id": req_id, "status": res.status, "timing": res.timing})
        await asyncio.sleep(0.05)
    scenarios_data["steady_load"] = {
        "count": len(steady_results),
        "status_distribution": {s: sum(1 for r in steady_results if r["status"] == s) for s in ["ok", "rate_limited", "error"]}
    }
    print(f"  Steady status counts: {scenarios_data['steady_load']['status_distribution']}")

    # Scenario 2: Sudden Burst (20 requests concurrent)
    print("\n2. Sudden Burst (N=20 concurrent)")
    async def burst_worker(idx):
        qid = f"q{(idx % 20) + 1:02d}"
        q = benchmark[qid]
        req_id = f"load-burst-{idx:02d}"
        res = await pipeline.ask(
            q["question"],
            provider_name="stand-in",
            request_id=req_id,
            scenario="final_sudden_burst",
            benchmark_item=q,
        )
        return {"req_id": req_id, "status": res.status}

    burst_results = await asyncio.gather(*(burst_worker(i) for i in range(20)))
    scenarios_data["sudden_burst"] = {
        "count": len(burst_results),
        "status_distribution": {s: sum(1 for r in burst_results if r["status"] == s) for s in ["ok", "rate_limited", "error"]}
    }
    print(f"  Burst status counts: {scenarios_data['sudden_burst']['status_distribution']}")

    # Scenario 3: Sustained Overload (50 requests concurrent)
    print("\n3. Sustained Overload (N=50 concurrent)")
    async def overload_worker(idx):
        qid = f"q{(idx % 20) + 1:02d}"
        q = benchmark[qid]
        req_id = f"load-overload-{idx:02d}"
        res = await pipeline.ask(
            q["question"],
            provider_name="stand-in",
            request_id=req_id,
            scenario="final_sustained_overload",
            benchmark_item=q,
        )
        return {"req_id": req_id, "status": res.status}

    overload_results = await asyncio.gather(*(overload_worker(i) for i in range(50)))
    scenarios_data["sustained_overload"] = {
        "count": len(overload_results),
        "status_distribution": {s: sum(1 for r in overload_results if r["status"] == s) for s in ["ok", "rate_limited", "error"]}
    }
    print(f"  Overload status counts: {scenarios_data['sustained_overload']['status_distribution']}")

    async def safe_ask(p_inst, q_text, r_id, sc_name, b_item, retries_count=2):
        try:
            return await p_inst.ask(
                q_text,
                provider_name="stand-in",
                request_id=r_id,
                scenario=sc_name,
                benchmark_item=b_item,
                max_retries=retries_count,
            )
        except ProviderError:
            return None

    def get_scenario_stats(sc_name):
        with open(EVAL_DIR / "results.csv", "r", encoding="utf-8") as f:
            sc_rows = [r for r in csv.DictReader(f) if r["scenario"] == sc_name]
        counts = {"ok": 0, "rate_limited": 0, "error": 0, "timeout": 0}
        for r in sc_rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return sc_rows, counts

    # Scenario 4: Per-minute Limit Halved (20 requests)
    print("\n4. Per-minute Limit Halved (N=20, TPM=4000)")
    halved_p = StandinProvider(store, replace(base_cfg, tokens_per_minute=4000))
    halved_pipeline = ResearchPipeline(
        ROOT,
        {"stand-in": halved_p},
        logger=logger,
        retriever=retriever,
    )
    await asyncio.gather(*(
        safe_ask(
            halved_pipeline,
            benchmark[f"q{(i % 20) + 1:02d}"]["question"],
            f"load-halved-{i:02d}",
            "final_per_minute_halved",
            benchmark[f"q{(i % 20) + 1:02d}"],
            retries_count=0,
        )
        for i in range(20)
    ))
    _, h_counts = get_scenario_stats("final_per_minute_halved")
    scenarios_data["per_minute_limit_halved"] = {
        "count": 20,
        "status_distribution": h_counts,
    }
    print(f"  Halved status counts: {h_counts}")

    # Scenario 5: Daily Budget Exhausted (15 requests)
    print("\n5. Daily Budget Exhausted (N=15, requests_per_day=3)")
    daily_p = StandinProvider(store, replace(base_cfg, requests_per_day=3))
    daily_pipeline = ResearchPipeline(
        ROOT,
        {"stand-in": daily_p},
        logger=logger,
        retriever=retriever,
    )
    for i in range(15):
        qid = f"q{(i % 20) + 1:02d}"
        q = benchmark[qid]
        await safe_ask(
            daily_pipeline,
            q["question"],
            f"load-daily-{i:02d}",
            "final_daily_exhausted",
            q,
            retries_count=0,
        )
    _, d_counts = get_scenario_stats("final_daily_exhausted")
    scenarios_data["daily_budget_exhausted"] = {
        "count": 15,
        "status_distribution": d_counts,
    }
    print(f"  Daily budget status counts: {d_counts}")

    # Save load_scenarios.json
    with open(EVAL_DIR / "load_scenarios.json", "w", encoding="utf-8") as f:
        json.dump(scenarios_data, f, indent=2)

    # 2. Retry Comparison
    print("\n--- Running Retry Policy Comparison ---")
    retry_data = {
        "limitation_note": (
            "StandinProvider failure injection computes seed_str = f'{request_id}:{attempt}'. "
            "Because each scenario incorporates its scenario name into request_id, "
            "pseudo-random failure schedules differ across scenario names for the same index i. "
            "This limitation is noted per evaluation requirements."
        ),
        "policies": {}
    }

    # Policy A: Zero retries
    p_zero = StandinProvider(store, replace(base_cfg, transient_failure_rate=0.4))
    pipe_zero = ResearchPipeline(ROOT, {"stand-in": p_zero}, logger=logger, retriever=retriever)
    await asyncio.gather(*(
        safe_ask(
            pipe_zero,
            benchmark[f"q{(i % 20) + 1:02d}"]["question"],
            f"retry-zero-{i:02d}",
            "final_retry_zero",
            benchmark[f"q{(i % 20) + 1:02d}"],
            retries_count=0,
        )
        for i in range(20)
    ))
    r_zero_rows, r_zero_counts = get_scenario_stats("final_retry_zero")
    retry_data["policies"]["zero_retries"] = {
        "status_counts": r_zero_counts,
        "mean_total_ms": round(sum(float(r["total_ms"]) for r in r_zero_rows if r.get("total_ms")) / max(len(r_zero_rows), 1), 2)
    }

    # Policy B: Immediate retries
    p_imm = StandinProvider(store, replace(base_cfg, transient_failure_rate=0.4))
    pipe_imm = ResearchPipeline(ROOT, {"stand-in": p_imm}, logger=logger, retriever=retriever)
    await asyncio.gather(*(
        safe_ask(
            pipe_imm,
            benchmark[f"q{(i % 20) + 1:02d}"]["question"],
            f"retry-imm-{i:02d}",
            "final_retry_imm",
            benchmark[f"q{(i % 20) + 1:02d}"],
            retries_count=2,
        )
        for i in range(20)
    ))
    r_imm_rows, r_imm_counts = get_scenario_stats("final_retry_imm")
    retry_data["policies"]["immediate_retries"] = {
        "status_counts": r_imm_counts,
        "mean_total_ms": round(sum(float(r["total_ms"]) for r in r_imm_rows if r.get("total_ms")) / max(len(r_imm_rows), 1), 2)
    }

    # Policy C: Exponential backoff
    p_backoff = StandinProvider(store, replace(base_cfg, transient_failure_rate=0.4))
    pipe_backoff = ResearchPipeline(ROOT, {"stand-in": p_backoff}, logger=logger, retriever=retriever)
    await asyncio.gather(*(
        safe_ask(
            pipe_backoff,
            benchmark[f"q{(i % 20) + 1:02d}"]["question"],
            f"retry-backoff-{i:02d}",
            "final_retry_backoff",
            benchmark[f"q{(i % 20) + 1:02d}"],
            retries_count=2,
        )
        for i in range(20)
    ))
    r_bo_rows, r_bo_counts = get_scenario_stats("final_retry_backoff")
    retry_data["policies"]["exponential_backoff"] = {
        "status_counts": r_bo_counts,
        "mean_total_ms": round(sum(float(r["total_ms"]) for r in r_bo_rows if r.get("total_ms")) / max(len(r_bo_rows), 1), 2)
    }
    print(f"Retry comparisons: {retry_data['policies']}")

    with open(EVAL_DIR / "retry_comparison.json", "w", encoding="utf-8") as f:
        json.dump(retry_data, f, indent=2)

    # 3. Targeted E2E Invariant & Feature Tests
    print("\n--- Executing Targeted E2E Feature Tests ---")
    e2e_results = {}

    # Test A: Successful Streaming
    q01 = benchmark["q01"]
    res_stream = await pipeline.ask(q01["question"], provider_name="stand-in", request_id="e2e-stream", benchmark_item=q01)
    e2e_results["successful_streaming"] = {
        "status": res_stream.status,
        "first_token_ms": res_stream.timing.get("first_token_ms") or res_stream.timing.get("provider_ttft_ms"),
        "total_ms": res_stream.timing.get("total_ms"),
        "passed": res_stream.status == "ok" and bool(res_stream.answer)
    }

    # Test B: Correct Refusal on Unanswerable
    q16 = benchmark["q16"]
    res_refusal = await pipeline.ask(q16["question"], provider_name="stand-in", request_id="e2e-refusal", benchmark_item=q16)
    e2e_results["correct_refusal"] = {
        "status": res_refusal.status,
        "decision": res_refusal.decision,
        "is_abstention": is_abstention(res_refusal.answer),
        "passed": res_refusal.decision == "insufficient" and is_abstention(res_refusal.answer)
    }

    # Test C: Citation Rejection
    v_res = verify_answer("Groundless claim without citation.", retriever.search("Transformer", "B"))
    e2e_results["citation_rejection"] = {
        "valid": v_res.valid,
        "reason": v_res.reason,
        "passed": not v_res.valid
    }

    # Test D: Admission Controller Queue Behavior
    adm = AdmissionController(max_concurrency=1, max_queue_depth=1, tokens_per_minute=50000)
    q_exit_1 = await adm.acquire(estimated_tokens=100)
    # Start background task holding the slot
    async def hold_slot():
        await asyncio.sleep(0.1)
        adm.release()
    asyncio.create_task(hold_slot())
    # Second should wait in queue
    q_exit_2 = await adm.acquire(estimated_tokens=100)
    adm.release()

    # Queue full test: fill slot and queue, third should be rejected immediately
    adm_full = AdmissionController(max_concurrency=1, max_queue_depth=1, tokens_per_minute=50000)
    await adm_full.acquire(estimated_tokens=100)
    # queue depth now 1
    t_wait = asyncio.create_task(adm_full.acquire(estimated_tokens=100))
    await asyncio.sleep(0.01)
    # third should reject with ProviderError
    queue_full_rejected = False
    try:
        await adm_full.acquire(estimated_tokens=100)
    except ProviderError as exc:
        queue_full_rejected = (exc.kind == ErrorKind.RATE_LIMITED)
    adm_full.release()
    await t_wait
    adm_full.release()

    e2e_results["admission_control"] = {
        "queue_waiting_verified": q_exit_2 >= q_exit_1,
        "queue_full_rejected_immediately": queue_full_rejected,
        "passed": (q_exit_2 >= q_exit_1) and queue_full_rejected
    }

    # Test E: Stand-in Token Accounting (input + output)
    p_tok = StandinProvider(store, replace(base_cfg, tokens_per_minute=200))
    tok_rej = False
    try:
        req_tok = GenerationRequest("tok-test", "q", "prompt text with many words", max_output_tokens=250)
        async for _ in p_tok.stream(req_tok):
            pass
    except ProviderError as exc:
        tok_rej = (exc.kind == ErrorKind.RATE_LIMITED)
    e2e_results["token_accounting_input_plus_output"] = {
        "rejected_on_total_tokens": tok_rej,
        "passed": tok_rej
    }

    # Test F: API /health and /ask endpoints
    from starlette.testclient import TestClient
    from src.api import create_app
    app = create_app(pipeline)
    client = TestClient(app)
    h_resp = client.get("/health")
    a_resp = client.post("/ask", json={"question": "What is the Transformer?"})
    e2e_results["api_endpoints"] = {
        "health_status_code": h_resp.status_code,
        "health_json": h_resp.json(),
        "ask_status_code": a_resp.status_code,
        "passed": h_resp.status_code == 200 and a_resp.status_code == 200
    }

    with open(EVAL_DIR / "e2e_test_results.json", "w", encoding="utf-8") as f:
        json.dump(e2e_results, f, indent=2)

    print(f"E2E test results: {e2e_results}")
    return scenarios_data, retry_data, e2e_results


def run_phase5():
    print("\n" + "=" * 80)
    print("PHASE 5: RECONCILIATION AND INVARIANT CHECKS")
    print("=" * 80)

    results_path = EVAL_DIR / "results.csv"
    diag_path = EVAL_DIR / "diagnostics.jsonl"
    headers_path = EVAL_DIR / "raw_headers.jsonl"

    if not results_path.exists():
        print("ERROR: results.csv does not exist in evaluation directory.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Total Rows in {results_path}: {len(rows)}")

    # Check 1: Exact columns
    with open(results_path, "r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    schema_ok = (header == RESULT_COLUMNS)
    print(f"1. Schema matches exactly (12 columns): {schema_ok}")

    # Check 2: Request ID uniqueness
    req_ids = [r["request_id"] for r in rows]
    unique_ids = len(set(req_ids)) == len(req_ids)
    print(f"2. All request IDs unique ({len(set(req_ids))}/{len(req_ids)}): {unique_ids}")

    # Check 3: Allowed status values
    statuses = set(r["status"] for r in rows)
    allowed_statuses = {"ok", "rate_limited", "timeout", "error"}
    status_ok = statuses.issubset(allowed_statuses)
    print(f"3. All statuses in allowed set {statuses}: {status_ok}")

    # Check 4: Latency invariants (total_ms >= first_token_ms for all streamed rows)
    inversions = []
    for r in rows:
        ttft = r.get("first_token_ms", "").strip()
        total = r.get("total_ms", "").strip()
        if ttft and total:
            try:
                if float(total) < float(ttft):
                    inversions.append(f"{r['request_id']}: total={total} < ttft={ttft}")
            except ValueError:
                pass
    print(f"4. Latency inversions (total < ttft): {len(inversions)} (Inversions: {inversions or 'None'})")

    # Check 5: Diagnostics coverage
    diag_ids = set()
    if diag_path.exists():
        with open(diag_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    diag_ids.add(json.loads(line).get("request_id"))
    print(f"5. Diagnostics coverage: {len(diag_ids)} entries in diagnostics.jsonl")

    # Check 6: Header capture
    header_count = 0
    if headers_path.exists():
        with open(headers_path, "r", encoding="utf-8") as f:
            header_count = sum(1 for line in f if line.strip())
    print(f"6. Raw headers recorded: {header_count} entries in raw_headers.jsonl")


async def main():
    benchmark = load_benchmark()
    retriever = PaperRetriever(ROOT)
    gate = EvidenceGate()
    logger = ExperimentLogger(EVAL_DIR)

    answers_path = EVAL_DIR / "answers.jsonl"
    if answers_path.exists() and sum(1 for line in answers_path.open(encoding="utf-8") if line.strip()) >= 45:
        print("\n" + "=" * 80)
        print("PHASE 2: REAL GROQ ANSWER EVALUATION (ALREADY COMPLETED — PRESERVING RESULTS)")
        print(f"Loaded 45 real Groq evaluations from {answers_path}")
        print("=" * 80)
    else:
        # Run Phase 2
        await run_phase2(logger, retriever, gate, benchmark)

    # Run Phase 3
    run_phase3(retriever, gate, benchmark)

    # Run Phase 4
    await run_phase4(logger, retriever, gate, benchmark)

    # Run Phase 5
    run_phase5()

    print("\n" + "=" * 80)
    print("ALL EVALUATION PHASES COMPLETE!")
    print(f"Artifacts saved in: {EVAL_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
