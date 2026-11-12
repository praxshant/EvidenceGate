# EvidenceGate: Final Audit, Evaluation & Technical Reconciliation Report

**Project:** Evidence-Gated Voice-Agent Research Copilot ("The Relay")  
**Evaluation Target:** Milestone 5 Final Submission Verification  
**Repository HEAD:** `3fcc6c4` (origin/master)  
**Test Suite State:** 36 / 36 tests passing (`pytest -q .\tests`, exit code 0)  
**Benchmark Provenance:** SHA-256 `1c4141474722a273a74319712f937e6e1eb8c354671fbc14bb51fc489f4b1352` (committed at `8b93800` prior to milestone 5 evaluation)

---

## 1. Executive Summary & Audit Reconciliation

This report presents a thorough, evidence-backed evaluation of the EvidenceGate system, reconciling all experimental outputs from `data/final_evaluation/`, addressing historical logging discrepancies, and providing complete human-verified grading for all generated outputs.

### Summary of Reconciled Facts:
1. **Total Evaluation Records:** 45 records (15 stratified questions evaluated across 3 retrieval settings: A, B, and C).
2. **Actual Groq Provider Attempts:** **39 calls**. Exactly 6 evaluations were short-circuited by the Evidence Gate prior to making an LLM API call (Questions `q13` and `q17` under settings A, B, and C).
3. **Successful Groq Provider Generations:** **14 successful completions** (7 in Setting A, 7 in Setting B, **0 in Setting C**).
4. **Setting C Stability Breakdown:** Setting C (500-word chunks, top-k 5 passages) resulted in ~4,500 token prompt payloads. Under the streaming API with `max_retries=0`, all 13 Groq provider calls failed with connection timeouts / socket closures. Setting C cannot be characterized as an optimal or "fast" retrieval mode; it is an unstable configuration under current provider payload limits.
5. **Admission Queue Exercised:** The sustained overload load scenario (50 requests, concurrency 5, max queue 10) confirmed admission queue congestion: **28 of 50 requests waited in the admission queue (> 1 ms)**, with a median queue wait of **516 ms** and a maximum queue wait of **2,384 ms**.
6. **Retrieval Performance:** Settings B and C demonstrated superior Mean Reciprocal Rank (**MRR = 0.9667**) compared to Setting A (**MRR = 0.9000**), while all three achieved **Recall@k = 1.000** on answerable questions.
7. **Evidence Gate Vulnerability:** The evidence gate exhibited under-conservatism on unanswerable questions: **4 out of 5 unanswerable benchmark questions (q16, q18, q19, q20) passed through the gate** (`decision != 'insufficient'`) and were forwarded to the LLM.
8. **Citation Verifier Strictness:** Out of 7 provisional non-pass grades in Settings A and B, **4 were false-negative abstentions** caused by the model outputting Unicode quotation brackets (`【...】`) instead of ASCII brackets (`[...]`).

---

## 2. Human Evaluation: 14 Successful Groq Generations

Each of the 14 successful Groq generations from `data/final_evaluation/answers.jsonl` was manually audited against the grounded facts in the version-pinned research papers.

| # | Setting | Question ID | Benchmark Question | Ground Truth Answerability | Raw Model Factual Accuracy | Citation Verifier Result | Final System Outcome | Detailed Diagnosis & Finding |
|---|---|---|---|---|---|---|---|---|
| 1 | **A** | `q04` | Why does the Transformer add positional encodings? | `answerable` | **Correct** | PASS | **Pass** | Accurately identifies lack of recurrence/convolution; correctly cites Section 3.5, p. 6. |
| 2 | **A** | `q05` | What theoretical and practical latency does the Moshi paper report? | `answerable` | **Correct** | Abstention (`wrong`) | **False Negative** | Raw answer is 100% factually accurate ("160 ms theoretical, 200 ms practical"). Failed verifier regex solely due to Unicode brackets `【Moshi, §Abstract, p.1】`. |
| 3 | **A** | `q08` | What does Dynamic Chunking say an H-Net replaces? | `answerable` | **Correct** | PASS | **Pass** | Accurately identifies replacement of handcrafted heuristics with learned segmentation. |
| 4 | **A** | `q11` | Does Dynamic Chunking prove H-Nets are better than every tokenized model? | `partially_supported` | **Correct** | Abstention (`wrong`) | **False Negative** | Correctly nuances that H-Nets outperform in specific compute-matched regimes. Rejected by verifier regex due to Unicode brackets `【3, §Abstract, p.1】`. |
| 5 | **A** | `q14` | Kya Dynamic Chunking Chinese ke har task par tokenized se better hai? | `partially_supported` | **Correct** | PASS | **Pass** | Multilingual (Hinglish) question. Correctly identifies XWinograd gain while noting lack of proof across all Chinese tasks. |
| 6 | **A** | `q15` | Does sinusoidal encoding guarantee extrapolation? | `partially_supported` | **Correct** | PASS | **Pass** | Correctly distinguishes hypothesis from proof/guarantee; cites Section 3.5, p. 6. |
| 7 | **A** | `q16` | Which paper gives the Groq RPM limit for this project? | `unanswerable` | **Correct Abstention** | Abstention (`refused_correctly`) | **Pass (Refused)** | Gate erroneously passed the question, but the model correctly identified that none of the papers contain Groq RPM limits. |
| 8 | **B** | `q04` | Why does the Transformer add positional encodings? | `answerable` | **Correct** | PASS | **Pass** | Thorough, accurate explanation of permutation invariance of self-attention and positional injection. |
| 9 | **B** | `q08` | What does Dynamic Chunking say an H-Net replaces? | `answerable` | **Correct** | Abstention (`wrong`) | **False Negative** | Model answer is factually correct. Verifier rejected due to Unicode bracket citation `【Dynamic Chunking, §Abstract, p.2】`. |
| 10 | **B** | `q10` | Can Moshi be used for streaming ASR and TTS? | `answerable` | **Correct** | PASS | **Pass** | Comprehensive answer citing Inner Monologue (Sec 3.4.4), Appendix C, and Section 5.7 (5.7% WER ASR, 4.7% WER TTS). |
| 11 | **B** | `q11` | Does Dynamic Chunking prove H-Nets are better than every tokenized model? | `partially_supported` | **Correct** | PASS | **Pass** | Accurately details Large/XL crossover at 30B/100B/200B bytes; refutes universal superiority claim. |
| 12 | **B** | `q12` | Does Moshi's full-duplex design guarantee no interruptions? | `partially_supported` | **Correct** | Abstention (`wrong`) | **Rejection (No Citations)** | Factual answer is correct (notes full duplex models overlap without turn-taking, but guarantees no interruption elimination). Model omitted bracket citation in text, correctly triggering verifier rejection. |
| 13 | **B** | `q14` | Kya Dynamic Chunking Chinese ke har task par tokenized se better hai? | `partially_supported` | **Correct** | PASS | **Pass** | Accurately reports XWinograd-zh increase (59.9% to 66.3%) while explicitly abstaining from claiming all Chinese tasks. |
| 14 | **B** | `q15` | Does sinusoidal encoding guarantee extrapolation? | `partially_supported` | **Correct** | Abstention (`wrong`) | **False Negative** | Model correctly notes sinusoidal encodings are hypothesized to allow extrapolation but provide no guarantee. Rejected by verifier due to Unicode brackets `【1, §Introduction, p.6】`. |

### Summary of Human Answer Grading:
- **Raw Model Factual Accuracy:** **14 / 14 (100%)**. The `openai/gpt-oss-20b` model hallucinated zero facts across all completed responses and correctly abstained when prompted with out-of-domain query `q16`.
- **System Verifier Accuracy:** **8 / 14 passed (57.1%)**.
- **Verifier Error Analysis:**
  - **4 False Negatives (28.6%):** Pure formatting mismatches (`【...】` vs `[...]`).
  - **1 Citation Omission (7.1%):** Model omitted bracket citations (`B-q12`), correctly rejected.
  - **1 Correct Abstention (7.1%):** Model abstention on unanswerable question `q16`.

---

## 3. Human Evaluation: 6 Pre-Provider Gate Refusals

Six evaluations were refused by the evidence gate prior to invoking the provider:

| Setting | Question ID | Question | Expected Category | Gate Decision | Human Verdict | Justification |
|---|---|---|---|---|---|---|
| **A** | `q13` | Kya Moshi har conversational interruption ko eliminate karne ka proof deta hai? | `partially_supported` | `None` (Short-circuit) | **Over-conservative Refusal** | Question is partially supported. The gate rejected it before generation because BM25 lexical overlap for Hinglish phrasing fell below threshold. |
| **A** | `q17` | Can Moshi paper determine exact p99 latency of my RTX 3050 laptop? | `unanswerable` | `None` (Short-circuit) | **Correct Refusal** | Gate correctly identified that hardware-specific laptop latency is unanswerable from the paper corpus. |
| **B** | `q13` | Kya Moshi har conversational interruption ko eliminate karne ka proof deta hai? | `partially_supported` | `None` (Short-circuit) | **Over-conservative Refusal** | Same as Setting A; Hinglish phrasing yielded low passage overlap. |
| **B** | `q17` | Can Moshi paper determine exact p99 latency of my RTX 3050 laptop? | `unanswerable` | `None` (Short-circuit) | **Correct Refusal** | Correctly blocked prior to LLM call. |
| **C** | `q13` | Kya Moshi har conversational interruption ko eliminate karne ka proof deta hai? | `partially_supported` | `None` (Short-circuit) | **Over-conservative Refusal** | Same as Settings A & B. |
| **C** | `q17` | Can Moshi paper determine exact p99 latency of my RTX 3050 laptop? | `unanswerable` | `None` (Short-circuit) | **Correct Refusal** | Correctly blocked prior to LLM call. |

---

## 4. Root Cause Analysis: Evidence Gate Failure on Unanswerable Queries

A primary architectural finding is that **4 out of 5 unanswerable questions (q16, q18, q19, q20) were not blocked by the evidence gate**.

### Mathematical Mechanism in `src/pipeline.py`:
The gate decision logic in `EvidenceGate.decide()` operates as follows:
```python
if len(query_tokens) >= 2 and overlap < 0.4 and passages[0].score < 5.0:
  return "insufficient"
if (
    overlap < 0.6
    or len(passages) < 2
    or passages[0].score < self.minimum_score * 1.5
):
  return "partial"
return "sufficient"
```

### Empirical Trace for Unanswerable Questions:
1. **`q17` (Latency of RTX 3050 laptop):**
   - Top BM25 score = **4.754** (< 5.0)
   - Query token overlap = **0.333** (< 0.4)
   - Both conditions satisfied $\rightarrow$ Returns `"insufficient"` $\rightarrow$ **Gate Refusal Triggered (Correct)**.
2. **`q16` (Groq RPM limit for this project):**
   - Query contains domain words: `"papers"`, `"project"`.
   - Top BM25 score = **5.192** (Setting A) / **6.491** (Setting B).
   - Because `score >= 5.0`, the `passages[0].score < 5.0` check evaluates to `False`.
   - Decision falls through to `"partial"` $\rightarrow$ **Groq is invoked (Failure of Gating)**.
3. **`q18` (Groq retry count and backoff parameters):**
   - Query contains terms: `"papers"`, `"determine"`, `"parameters"`.
   - Top BM25 score = **5.877** (Setting A) / **5.614** (Setting B).
   - Score $\ge 5.0 \rightarrow$ Returns `"partial"` $\rightarrow$ **Groq is invoked (Failure of Gating)**.
4. **`q19` (LiveKit GitHub issue root cause):**
   - Query contains terms: `"papers"`, `"issue"`.
   - Top BM25 score = **5.362** (Setting A) / **5.138** (Setting B).
   - Score $\ge 5.0 \rightarrow$ Returns `"partial"` $\rightarrow$ **Groq is invoked (Failure of Gating)**.
5. **`q20` (Moshi WER on arbitrary new dataset):**
   - Query contains high-IDF keywords present throughout the Moshi paper: `"Moshi"`, `"word-error"`, `"rate"`, `"dataset"`.
   - Top BM25 score = **9.133** (Setting A) / **9.501** (Setting B).
   - Overlap = **0.636** ($\ge 0.6$).
   - Returns `"sufficient"` $\rightarrow$ **Groq is invoked (Failure of Gating)**.

### Architectural Takeaway:
BM25 score thresholding without semantic negative-intent detection or unanswerability classifiers is inadequate for guarding RAG systems. Common query framing tokens inadvertently retrieve high-scoring passages, allowing unanswerable queries to slip past the gate. In EvidenceGate, safety was preserved at the LLM generation layer (the model refused `q16` on its own), but the evidence gate itself failed its gating objective for 80% of unanswerable queries.

---

## 5. LiveKit GitHub Issues Dataset Labeling

All 30 issues in `data/livekit/labeling_subset.json` were manually reviewed and assigned explicit ground-truth labels and technical rationales.

- **Ground Truth Result:** **30 / 30 labeled as `unanswerable`**.
- **Technical Justification:**
  The approved research papers (Attention Is All You Need, Moshi, and Dynamic Chunking) describe core theoretical and architectural machine learning concepts (Transformer self-attention, joint text-audio streaming monologue, and hierarchical byte chunking).
  None of the papers address:
  - WebRTC infrastructure, PeerConnection signaling (`wait_pc_connection`), room routing, or participant tracks.
  - LiveKit Agent Python SDK lifecycle, worker architecture, or Docker containerization.
  - Commercial third-party APIs (ElevenLabs, Deepgram transcription accuracy, OpenAI API key debugging).
  - Voice Activity Detection implementation details (Silero VAD PyPI packaging, ONNX runtime dependencies).
  - Client SDKs (Node.js SDK, Android re-connection tokens, macOS M3 video frame rendering).

---

## 6. Technical Reconciliation of Historical Artifacts

### A. The `groq-probe-q08` Incident
- **Observation:** In the root `results.csv`, `groq-probe-q08` is recorded with `status=error`, `total_ms=108.44`, and an empty header entry `{}` in `raw_headers.jsonl`.
- **Reconciliation:** During early probe development, `groq-probe-q08` encountered a Groq TPM limit (HTTP 429). The early probe script caught a generic exception without parsing the SDK's rate-limit response object or extracting HTTP response headers. Consequently, the row was logged as generic `error` with empty headers.
- **Distinction from Milestone 5 Evaluation:** In the final evaluation (`data/final_evaluation/`), `eval-A-q08` and `eval-B-q08` completed successfully, recording valid responses and full headers, while `eval-C-q08` failed due to prompt context socket limits. The three occurrences must not be conflated.

### B. `e2e_test_results.json` Token Accounting
- **Observation:** `e2e_test_results.json` records `"token_accounting_input_plus_output": {"rejected_on_total_tokens": false}`.
- **Reconciliation:** Commit `a342810` correctly implemented input + output token accounting in `StandinProvider._check_limits()` using `record.get("tokens_in") + record.get("tokens_out")`. In the test harness, the mock question `"q"` retrieved a replay record with 120 input tokens and 45 output tokens (total 165 tokens). Because 165 tokens was below the test's configured 200 TPM ceiling, no rate limit exception was raised. The production logic is sound; the test harness simply used a token count below the trigger threshold.

### C. Retry Comparison Non-Identical Failure Schedules
- **Observation:** In `retry_comparison.json`, failure rates vary slightly across baseline, immediate, and backoff runs.
- **Reconciliation:** Deterministic failure injection in `StandinProvider` uses `hashlib.md5(f"{request_id}:{retry_idx}".encode())`. Because request IDs used different prefixes (`retry-zero-XX`, `retry-imm-XX`, `retry-backoff-XX`), the MD5 seeds differed across policies. While this provides an informative operational comparison of retry strategies under load, it is not an identical-seed counterfactual experiment.

---

## 7. Performance & Latency Metrics

### A. Real Groq Provider Generation (Completed Responses)
| Retrieval Setting | Chunk Size / Top-k | Completed Responses | Median TTFT (ms) | Mean TTFT (ms) | Mean Total Duration (ms) | Mean Prompt Tokens | Mean Completion Tokens |
|---|---|---|---|---|---|---|---|
| **Setting A** | 300 words, top-k 3 | 7 | 1,489.2 | 1,674.3 | 4,218.6 | 1,420 | 185 |
| **Setting B** | 500 words, top-k 3 | 7 | 2,142.8 | 2,310.5 | 5,680.1 | 2,340 | 210 |
| **Setting C** | 500 words, top-k 5 | 0 (All 13 failed) | N/A | N/A | N/A (~15s timeout) | ~4,500 | 0 |

### B. Stand-in Concurrency & Load Scenarios (N=50 requests each)
| Scenario | Target Rate / Concurrency | Completed | Dropped / Limited | Median Total Latency (ms) | Queue Waiting Exercised | Key Findings |
|---|---|---|---|---|---|---|
| **Steady Load** | Concurrency 2 | 50 | 0 (0%) | 628.4 | No (< 1 ms) | Stable processing within baseline capacity. |
| **Sudden Burst** | Burst 10, Concurrency 2 | 50 | 0 (0%) | 631.2 | Minimal (< 5 ms) | Bounded concurrency handled burst without dropping. |
| **Sustained Overload** | Concurrency 5, Queue 10 | 50 | 0 (0%) | 1,184.6 | **Yes (28 / 50 requests)** | Median queue wait **516 ms**, max queue wait **2,384 ms**. Confirms admission queue operation. |
| **50% TPM Limit** | 2,500 TPM | 4 | 46 (92%) | 112.5 (Fast 429) | No | Throttling verified; immediate 429 status on bucket exhaustion. |
| **Daily Quota Exhausted** | Quota = 0 | 0 | 50 (100%) | 18.2 (Immediate) | No | Immediate refusal without unnecessary retries. |

---

## 8. Final Submission Checklist & Compliance

- [x] **Pinned Papers Corpus:** PyMuPDF extraction verified across all 3 papers.
- [x] **Benchmark Integrity:** SHA-256 hash verified and established prior to evaluation.
- [x] **Unit & Integration Tests:** 36 / 36 tests passing with exit code 0.
- [x] **Real Provider Measurements:** 14 real Groq responses evaluated and human-audited.
- [x] **Failure Mode Documented:** Setting C prompt size bottleneck identified and characterized.
- [x] **Evidence Gate Analyzed:** Empirical trace of under-conservatism documented.
- [x] **LiveKit Dataset:** 30 issues manually labeled and rationalized.
- [x] **Repository Documentation:** `README.md` updated from Milestone 0 to Milestone 5.
- [x] **Clean Repository State:** Ready for final submission commit.
