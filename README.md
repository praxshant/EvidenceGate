# Evidence-Gated Voice-Agent Research Copilot

This project is the Milestone 0 foundation for The Relay assignment. It will
answer engineering questions only from three version-pinned research papers,
attach verifiable citations, and abstain when the library is insufficient.

## Milestone 0 status

- The exact arXiv PDFs are stored under `data/papers/`.
- Page-aware PyMuPDF extraction is stored under `data/processed/papers/`.
- `data/papers/metadata.json` records identifiers, versions, URLs, and
  extraction checks.
- `data/benchmarks/benchmark_draft.json` contains 20 reviewable questions.
- The LiveKit collector is prepared at `scripts/fetch_livekit_issues.py`, but
  bulk collection is intentionally not run in Milestone 0.

## Proposed architecture

1. Normalize and validate the request.
2. Retrieve paper passages with BM25 and retain page/section metadata.
3. Apply deterministic evidence and scope checks.
4. Call either the Groq provider or the local controlled stand-in through one
   model interface.
5. Verify citations against retrieved passages before delivery.
6. Refuse, retry within bounded policy, or return the verified answer.

The initial implementation will use lightweight Python components, SQLite or
JSONL for experiment records, FastAPI for the local service surface, and
`asyncio` for bounded concurrency. No GPU-heavy model or fine-tuning is
planned.

The implementation is available under `src/`. Run offline checks with:

```powershell
$env:PYTHONPATH = "."
python -m pytest -q .\tests --basetemp .\.pytest-temp
python .\scripts\groq_probe.py
```

The Groq preflight currently reports whether `GROQ_API_KEY` and `GROQ_MODEL`
are configured and prints the bounded call/token budget without printing
credentials. In this environment they are not configured, so no real Groq
requests or calibration measurements have been made. The stand-in defaults
remain marked `pending_real_groq_measurements`.

## Milestone 0 commands

```powershell
# Verify paper versions and regenerate page-aware extraction.
python .\scripts\download_papers.py

# Dry run: confirms collection is intentionally disabled.
python .\scripts\fetch_livekit_issues.py

# Actual collection, only after review and approval.
python .\scripts\fetch_livekit_issues.py --collect --limit 100
```

The collector uses the oldest available chronological cohort of non-pull-request
issues returned by the API. It is reproducible in procedure but not
representative of all issues; the metadata records this limitation. An
optional `GITHUB_TOKEN` is used as a bearer token when present, but no token is
required by the script.

The required `results.csv`, `raw_headers.jsonl`, and `diagnostics.jsonl` files
are intentionally not ignored so that measured submission artifacts can be
tracked. They do not exist yet. Generated LiveKit issue data remains ignored
until its collection is approved. `.env` and credentials remain ignored.

## Planned later artifacts

The later milestones will add the pipeline, provider clients, stand-in,
experiments, tests, and the maximum two-page report. No measurements are
claimed until those experiments are run.
