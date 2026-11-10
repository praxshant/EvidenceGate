# Evidence-Gated Voice-Agent Research Copilot

This project is the Milestone 0 foundation for The Relay assignment. It will
answer engineering questions only from three version-pinned research papers,
attach verifiable citations, and abstain when the library is insufficient.

## Milestone 0 status

- The exact arXiv PDFs are stored under `data/papers/`.
- Page-aware PyMuPDF extraction is stored under `data/processed/papers/`.
- `data/papers/metadata.json` records identifiers, versions, URLs, and
  extraction checks.
- `data/benchmarks/benchmark_draft.json` contains 18 reviewable questions.
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

## Milestone 0 commands

```powershell
python .\scripts\download_papers.py
python .\scripts\fetch_livekit_issues.py
```

The collector command stops unless `--collect` is explicitly supplied. Do not
run bulk collection until the collection plan and benchmark are reviewed.

## Planned later artifacts

The later milestones will add the pipeline, provider clients, stand-in,
diagnostics, experiments, `results.csv`, raw headers, tests, and the maximum
two-page report. No measurements are claimed until those experiments are run.
