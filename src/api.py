"""FastAPI interface for the research pipeline."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .pipeline import ResearchPipeline
from .providers.base import ErrorKind, ProviderError


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    provider: str = "stand-in"
    retrieval_setting: str = "B"
    scenario: str = "interactive"


def create_app(pipeline: ResearchPipeline) -> FastAPI:
    app = FastAPI(title="EvidenceGate")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask")
    async def ask(request: AskRequest, response: Response) -> dict[str, object]:
        try:
            result = await pipeline.ask(
                request.question,
                provider_name=request.provider,
                retrieval_setting=request.retrieval_setting,
                scenario=request.scenario,
            )
        except ProviderError as exc:
            if exc.kind == ErrorKind.DAILY_QUOTA:
                raise HTTPException(status_code=503, detail="Daily quota exhausted") from exc
            elif exc.kind == ErrorKind.RATE_LIMITED:
                headers: dict[str, str] = {}
                if exc.retry_after is not None:
                    headers["Retry-After"] = str(int(exc.retry_after))
                raise HTTPException(status_code=429, detail=str(exc), headers=headers or None) from exc
            elif exc.kind == ErrorKind.TIMEOUT:
                raise HTTPException(status_code=504, detail=str(exc)) from exc
            elif exc.kind == ErrorKind.PERMANENT:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            else:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "request_id": result.request_id,
            "answer": result.answer,
            "citations": result.citations,
            "evidence": [
                {
                    "paper": p.paper,
                    "title": p.title,
                    "arxiv_id": p.arxiv_id,
                    "version": p.version,
                    "section": p.section,
                    "page": p.page,
                    "text": p.text,
                    "score": p.score,
                }
                for p in result.evidence
            ],
            "answerability": result.decision,
            "provider": result.provider,
            "timing": result.timing,
            "status": result.status,
            "scenario": request.scenario,
        }

    return app


def default_app() -> FastAPI:
    raise RuntimeError("Construct the app with configured providers; no provider is implicit.")
