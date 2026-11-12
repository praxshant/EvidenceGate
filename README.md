# EvidenceGate: Evidence-Gated Voice-Agent Research Copilot

EvidenceGate (The Relay) is an evidence-gated Retrieval-Augmented Generation (RAG) system built to answer engineering questions strictly from three version-pinned academic research papers, enforce deterministic citation verification, and abstain when evidence is insufficient.

---

## 1. System Architecture

```
                    [ Incoming User Request ]
                                |
                                v
               [ Step 0: Admission Controller ]
              - Max Concurrency (In-flight Semaphore)
              - FIFO Queue (Bounded Wait & Capacity)
              - Rate / Token Accounting (TPM Bucket)
                                |
                                v
                [ Step 1: BM25 Passage Retrieval ]
              - Settings A (300w, stride 50, top-k 3)
              - Settings B (500w, stride 100, top-k 3)
              - Settings C (500w, stride 100, top-k 5 + Rerank)
                                |
                                v
                 [ Step 2: Evidence Gate ]
              - Token Overlap & BM25 Thresholding
              - Decisions: sufficient | partial | insufficient
                                |
           +--------------------+--------------------+
           | insufficient                            | sufficient / partial
           v                                         v
   [ Fast Abstention ]                     [ Unified Provider Interface ]
"I cannot answer this from..."          - Real Groq Client (`openai/gpt-oss-20b`)
                                          * Streaming TTFT & Token Logging
                                          * max_retries = 0 (App-level observation)
                                        - Stand-in Client (Deterministic Simulation)
                                          * Configurable Delay, Jitter, & Slowdown
                                          * Deterministic Hash-Based Failure Injection
                                                     |
                                                     v
                                       [ Step 3: Citation Verification ]
                                     - Deterministic Regex Bracket Match
                                     - Page/Section & Term Grounding
                                     - Abstention on Unverified Claims
                                                     |
                                                     v
                                         [ Final Verified Answer ]
                                                     |
                                                     v
                                      [ Experiment & Diagnostic Log ]
                                     - results.csv (12-column reconciled schema)
                                     - diagnostics.jsonl (Monotonic nanosecond timing)
                                     - raw_headers.jsonl (Provider rate-limit headers)
```

---

## 2. Pinned Research Papers Corpus

The pipeline operates strictly on three version-pinned arXiv research papers stored under `data/papers/` with page-aware PyMuPDF extractions under `data/processed/papers/`:

1. **Attention Is All You Need**: arXiv:1706.03762v7 (`attention_is_all_you_need_v7.pdf`, 15 pages)
2. **Moshi: a speech-text foundation model for real-time dialogue**: arXiv:2410.00037v1 (`moshi_v1.pdf`, 66 pages)
3. **Dynamic Chunking for Autoregressive Language Models (H-Net)**: arXiv:2507.07955v2 (`dynamic_chunking_v2.pdf`, 39 pages)

---

## 3. Evaluation Benchmark & Provenance

- **Benchmark File:** `data/benchmarks/benchmark_draft.json`
- **Benchmark SHA-256:** `1c4141474722a273a74319712f937e6e1eb8c354671fbc14bb51fc489f4b1352`
- **Provenance:** The benchmark was frozen and committed at `8b93800` prior to final evaluation runs.
- **Composition:** 20 stratified evaluation questions across the three papers:
  - 10 fully answerable questions (including multilingual Hindi/Hinglish questions).
  - 5 partially supported questions (testing nuanced boundaries and claims).
  - 5 strictly unanswerable questions (out-of-domain or unanswerable queries).

---

## 4. Final Evaluation Findings & Experimental Analysis

Artifacts are recorded under `data/final_evaluation/`.

### A. Real Groq Model Generation (15 Questions × 3 Retrieval Configurations = 45 Evaluations)
- **Model:** `openai/gpt-oss-20b` via Groq Cloud API (`max_retries = 0`).
- **Provider Attempts:** 39 actual Groq calls (6 evaluations were short-circuited by the Evidence Gate prior to calling Groq).
- **Successful Generations:** **14 successful responses** (7 in Setting A, 7 in Setting B, 0 in Setting C).
- **Setting C Stability Bottleneck:** Setting C (500 words, top-k 5) constructed large context prompts (~4,500 tokens), causing all 13 Groq provider streaming attempts to fail with connection timeouts / socket errors. Setting C is therefore identified as an unstable configuration under current provider payload limits, not a high-throughput mode.
- **Citation Verifier False Negatives:** Out of 7 provisional non-pass grades in Settings A and B, 4 were caused by the model outputting full-width Unicode citation brackets (e.g., `【Moshi, §Abstract, p.1】`) which the ASCII bracket regex rejected. Human review confirms the underlying generated answers were factually accurate.
- **Evidence Gate Under-conservatism:** 4 of 5 unanswerable questions (q16, q18, q19, q20) passed through the evidence gate because domain words in the query resulted in BM25 scores exceeding the 5.0 threshold, yielding `decision="partial"` or `decision="sufficient"`. The model subsequently handled refusal internally.

### B. Retrieval Performance Across 20 Benchmark Questions
- **Setting A (300w, top-k 3):** Recall@k = 1.000, MRR = 0.9000
- **Setting B (500w, top-k 3):** Recall@k = 1.000, MRR = 0.9667
- **Setting C (500w, top-k 5 + rerank):** Recall@k = 1.000, MRR = 0.9667

### C. Stand-in Concurrency & Load Scenarios (N=50 requests each)
1. **Steady Load (concurrency 2):** 50/50 completed, 0% drop rate, avg latency 632 ms.
2. **Sudden Burst (burst 10, concurrency 2):** 50/50 completed, 0% drop rate, avg latency 636 ms.
3. **Sustained Overload (concurrency 5, max queue 10):** 28 of 50 requests experienced queue waiting > 1 ms (median queue wait 516 ms, maximum queue wait 2,384 ms), demonstrating admission queue congestion under heavy load.
4. **Per-Minute Limit Halved (2,500 TPM):** 46 requests rate limited (429), verifying TPM throttling.
5. **Daily Budget Exhausted:** 50 requests rejected immediately with quota exhaustion errors.

### D. Retry Policies Comparison
- **Zero Retries (`max_retries=0`):** Fast terminal failure on transient faults; provides transparent measurement of provider availability.
- **Immediate Retries (`max_retries=2`):** Amplifies load during outages; causes cascading queue depth growth.
- **Exponential Backoff (`max_retries=2`):** Smooths load recovery and prevents provider stampedes.
*(Note: Because request IDs seed deterministic failure injection, failure schedules across retry policies are hash-dependent rather than identical).*

### E. LiveKit GitHub Issues Dataset
- Located at `data/livekit/labeling_subset.json` (30 sampled non-PR issues).
- All 30 issues are manually labeled and reviewed with explicit technical rationales as `unanswerable` relative to the version-pinned research paper corpus.

---

## 5. Verification & Reproduction

### Prerequisites
- Python 3.10+
- Dependencies: `pytest`, `pytest-asyncio`, `starlette`, `httpx`, `groq`, `pymupdf`

### Test Suite Execution
Run the complete test suite (36 tests verifying concurrency, admission control, token budgets, citation verifier, retries, and API endpoints):
```powershell
$env:PYTHONPATH = "."
python -m pytest -q .\tests --basetemp .\.pytest-temp
```

### Reproducing Final Evaluation
To run the full evaluation suite:
```powershell
$env:PYTHONPATH = "."
python .\scripts\run_final_evaluation.py
```

### Starting the API Service
```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
- Health probe: `GET /health`
- Evidence-gated question answering: `POST /ask` with `{"question": "Why does the Transformer use self-attention?"}`
