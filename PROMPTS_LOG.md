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

Future entries must distinguish assistant suggestions from the candidate's
decisions and must record mistakes only when they are actually observed.
