# Prompts and decision log

This log records important assistant prompts and decisions as they occur. It
does not replace the required unedited conversation history.

## Milestone 0

- Scope: complete paper/data/documentation preparation only; wait for explicit
  approval before implementing the inference pipeline.
- Decision: use the existing Conda Python 3.13 environment and install only
  narrowly required dependencies.
- Decision: use exact versioned arXiv PDF URLs and verify each version through
  its arXiv abstract page before extraction.
- Decision: use PyMuPDF so each extracted passage retains its original PDF
  page number.
- Decision: keep the benchmark in a separate JSON artifact and do not run it
  through the system before review.
- Decision: prepare the LiveKit collector but do not bulk collect or manually
  grade issues yet.

## Milestone 0 corrections

- Corrected benchmark citations after checking the extracted page text and
  original PDFs, including abstract-versus-introduction labels and
  PDF-preserved line-break hyphenation in quoted evidence.
- Expanded the draft benchmark from 18 to 20 questions with 10 answerable,
  5 partially supported, and 5 unanswerable questions, including four
  Hindi/Hinglish questions.
- Replaced the unused collector seed with an explicitly documented
  chronological cohort procedure and its sampling limitations.
- Made `GITHUB_TOKEN` optional but functional in the collector; no collection
  was run.
- Kept `results.csv`, `raw_headers.jsonl`, and `diagnostics.jsonl` trackable
  while continuing to ignore `.env`.

## Milestones 1–3 implementation

- Added a shared async provider contract with explicit transient, timeout,
  rate-limit, daily-quota, and permanent error classes.
- Added lazy official Groq SDK integration so offline retrieval and tests do
  not require credentials.
- Added page-aware BM25-style retrieval with configurable A/B/C chunk and
  top-k settings, deterministic evidence gating, buffered answer verification,
  and FastAPI `/health` and `/ask` endpoints.
- Added a replay-only stand-in with configurable latency, streaming chunks,
  failures, quotas, concurrency slowdown, and daily-budget exhaustion.
- Added shared `results.csv`, `diagnostics.jsonl`, and raw-header logging
  utilities plus offline tests.
- No Groq requests were made: the preflight found no configured
  `GROQ_API_KEY` or `GROQ_MODEL`, so calibration remains pending.

## Milestones 1–3 debugging and remediation pass

- Fixed F-01: Corrected status assignment before logging in `ResearchPipeline.ask()`.
- Fixed F-02: Unified ReplayStore indexing by question text with question_id fallback; added controlled ProviderError for empty replay files.
- Fixed F-03: Upgraded `scripts/groq_probe.py` from a hollow stub to a real, bounded streaming probe with latency, TTFT, token usage, header capture, and double-gated `--run --confirm` flags.
- Fixed F-04: Enabled response header extraction on successful Groq completions and logging to `raw_headers.jsonl`.
- Fixed F-05: Added `stream_options={"include_usage": True}` and unified `last_usage` token reporting for both Groq and stand-in providers into `results.csv`.
- Fixed F-06: Redesigned citation verification with `is_abstention()` detection, 3-tier section fallback, and deterministic claim token overlap checks.
- Fixed F-07: Implemented stateful sequential section boundary tracking in `PaperRetriever`, carrying active sections across continuation pages without guessing per-page.
- Fixed F-08: Replaced second-resolution timestamps in `diagnostics.jsonl` with microsecond ISO strings and monotonic millisecond stage diffs via `DiagnosticEntry`.
- Fixed F-09: Ensured all terminal failures (rate limits, timeouts, quota exhaustion) write exactly one row to `results.csv` before raising.
- Fixed F-10: Propagated `scenario` through `ResearchPipeline.ask()`, FastAPI `AskRequest`, and `ExperimentLogger.result()`.
- Fixed F-11: Added configurable `day_seconds` window and `reset_daily_quota()` to `StandinProvider`.
- Fixed F-12: Added `asyncio.Lock` synchronization to all CSV and JSONL write operations in `ExperimentLogger`.
- Fixed F-13: Added deterministic two-stage composite reranking (BM25 + query coverage + bigram bonus) for Retrieval Setting C.
- Fixed F-14: Injected project root into `sys.path` across all runnable scripts for seamless Windows execution without manual `PYTHONPATH`.
- Fixed F-15: Expanded the test suite from 6 superficial tests to 27 comprehensive integration and unit tests covering success, refusals, failures, rate-limiting, concurrency, stand-in simulation, and FastAPI endpoints. All 27 tests pass offline.

## Milestones 4–5 Initial Probe and Experiment Run

- Decision: Model target set strictly to `openai/gpt-oss-20b` on Groq using `GROQ_API_KEY_1` solely, without combining accounts.
- Decision: Prepared 30 LiveKit GitHub issues for manual evaluation in `data/livekit/labeling_subset.json` strictly without automated labeling (`label: null` across all 30 records).
- Implemented real Groq probe in `scripts/run_experiments.py` executing 5 short streaming queries (q01, q03, q04, q05, q08) with `max_completion_tokens=100`.
- Observed: q01, q03, q04, and q05 succeeded with real streaming content and captured rate-limit headers; q08 failed with HTTP 429 when prompt + requested completion tokens (6,933 used + 2,498 requested) exceeded the organization's 8,000 TPM limit.
- Calibrated stand-in simulation parameters from valid Groq probe observations (`base_delay=0.326s`, `jitter=1.603s`, `inter_chunk=0.0217s`).
- Populated replay store (`data/replay/responses.jsonl`) from real probe completions and ran the benchmark across Retrieval Settings A, B, and C, 5 load scenarios, and retry comparisons.

## Milestones 4–5 Audit, Invariant Enforcement, and Provider Corrections

- Observed mistake / discrepancy: Naive aggregate reporting produced an apparent median TTFT > median total latency anomaly.
- Candidate analysis & decision: Identified that mixing fast gate refusals (~35 ms, no provider streaming) and fast rate-limit failures with full streamed generations distorted aggregate medians. Mandated reporting paired latency populations (streamed completions vs gate refusals vs failures).
- Corrected GPT-OSS parameters: Removed unsupported `reasoning_format` parameter per Groq API specifications; applied supported `include_reasoning=False` and `reasoning_effort="low"`.
- Expanded test suite from 27 to 30 tests by adding latency invariant regression tests (`test_results_csv_latency_invariants`, `test_groq_provider_uses_correct_reasoning_params`, and `test_scenario_summary_no_ttft_total_inversion`).

## Remediation and Hardening Pass (Commits a342810, 533628e, b68944a)

- Candidate decision & remediation directive: Addressed 9 confirmed engineering and measurement defects prior to final submission.
- Fixed SDK Retries: Explicitly passed `max_retries=0` to `AsyncGroq()` in `GroqProvider`, `groq_probe.py`, and `run_experiments.py` to prevent SDK-internal silent retries from masking rate limits and distorting latency measurements.
- Fixed Retry State Bug: In `ResearchPipeline.ask()`, reset partial generation pieces (`pieces = []`) and first-token timestamps at the beginning of every attempt inside the retry loop so that aborted stream chunks do not contaminate retried outputs.
- Enforced Streaming Deadline: Added monotonic timeout deadline (`deadline_ms = monotonic_ms() + timeout_seconds * 1000.0`) checked across stream chunks and retry intervals.
- Corrected Retry-After Handling: Added `_parse_retry_after()` helper respecting provider-requested delays without arbitrary caps; delays exceeding remaining budget trigger `TIMEOUT` cleanly.
- Isolated Request Metadata: Added `_request_metadata` dictionary with `pop_request_metadata()` in `GroqProvider` and `StandinProvider` keyed by `request_id`, eliminating concurrent race conditions and memory leaks.
- Added Admission Control: Implemented `AdmissionController` in `src/pipeline.py` with `asyncio.Semaphore` concurrency limiting, bounded queue rejection, and rolling 60-second TPM token budget accounting.
- Fixed Stand-in Token Accounting: Corrected `StandinProvider._check_limits` to count combined prompt tokens and completion tokens (`total_tokens = prompt_tokens + completion_tokens`) against TPM limits rather than output-only tokens.
- Refined Benchmark Answer Grading: Grounded gate refusal grading on `expected_answerability` (`refused_correctly` for unanswerable questions vs `refused_wrongly` for answerable questions) and recorded detailed verification failure reasons in `diagnostics.jsonl`.
- Expanded Test Suite to 36 Tests: Added tests 31–36 in `tests/test_core.py` covering concurrent metadata isolation, streaming deadlines, retry chunk reset, refusal grading, queue rejection, and total token budget accounting. All 36 tests pass offline with exit code 0.
- Re-executed Complete Experiments: Generated 250 records in `results.csv` conforming strictly to the 12-column schema with zero latency inversions, complete diagnostics coverage on all 245 pipeline runs, and preserved raw evidence files.

## Final Experimental Execution and Answer Evaluation (data/final_evaluation/)

- Candidate directive: Execute real Groq evaluation across 15 stratified benchmark questions (5 answerable, 5 partially supported, 5 unanswerable) against all three retrieval settings (A, B, C = 45 evaluations) using `openai/gpt-oss-20b` with `max_retries=0` and pacing under 8,000 TPM.
- Preserved historical evidence: Stored complete execution artifacts in isolated timestamped directory `data/final_evaluation/` without modifying core source or overwriting historical root CSV/JSONL records.
- Completed Phase 2 Real Evaluations: Generated and saved all 45 outputs to `answers.md` and `answers.jsonl` with captured citations, verification results, and token measurements.
- Completed Phase 3 Retrieval Evaluation: Evaluated all 20 benchmark questions across settings A, B, and C; recorded Gold Recall@k (100%), Setting B/C MRR (0.9667), and gate decision breakdowns in `retrieval_evaluation.json`.
- Completed Phase 4 Stand-in Load & E2E Testing: Evaluated steady load, sudden burst, sustained overload, halved per-minute limit (rate-limiting verified), and daily budget exhaustion (quota exhaustion verified); evaluated retry policies and noted hash seed scope limitation; verified admission control queue behavior and citation rejection.
- Completed Phase 5 Reconciliation: Reconciled 233 unique records in `data/final_evaluation/results.csv` and `diagnostics.jsonl` with 100% schema conformance, zero latency inversions, and captured response headers in `raw_headers.jsonl`.

Future entries must distinguish assistant suggestions from the candidate's
decisions and must record mistakes only when they are actually observed.
