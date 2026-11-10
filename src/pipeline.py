"""Evidence-gated question answering orchestration."""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import dataclass

from .logging_utils import DiagnosticEntry, ExperimentLogger, monotonic_ms, now_iso_hires
from .providers.base import ErrorKind, GenerationRequest, ModelProvider, ProviderError
from .retrieval import PaperRetriever, Passage
from .verification import VerificationResult, verify_answer


@dataclass
class AskResult:
    request_id: str
    answer: str
    citations: list[str]
    evidence: list[Passage]
    decision: str
    provider: str
    timing: dict[str, float]
    status: str


class EvidenceGate:
    def __init__(self, *, minimum_score: float = 2.0) -> None:
        self.minimum_score = minimum_score

    def decide(self, passages: list[Passage], question: str = "") -> str:
        if not passages or passages[0].score < self.minimum_score:
            return "insufficient"
        if question:
            from .retrieval import tokens
            query_tokens = tokens(question)
            if query_tokens:
                doc_tokens = set(tokens(passages[0].text))
                overlap = len(set(query_tokens) & doc_tokens) / len(set(query_tokens))
                # Incidental match: low query token coverage and no strong phrase boost
                if len(query_tokens) >= 2 and overlap < 0.4 and passages[0].score < 5.0:
                    return "insufficient"
                if overlap < 0.6 or len(passages) < 2 or passages[0].score < self.minimum_score * 1.5:
                    return "partial"
        elif len(passages) < 2 or passages[0].score < self.minimum_score * 1.5:
            return "partial"
        return "sufficient"


def build_prompt(question: str, passages: list[Passage], decision: str) -> str:
    evidence = "\n\n".join(
        f"Evidence {i}: [{p.paper}, §{p.section}, p.{p.page}]\n{p.text}"
        for i, p in enumerate(passages, 1)
    )
    return (
        "You answer only from the supplied research evidence. Do not use outside knowledge. "
        "Answer the supported part concisely, cite every material claim using the exact "
        "format [Paper short name, §section, p.N], and explicitly say what the evidence "
        "does not establish. If evidence is insufficient, refuse rather than guess.\n\n"
        f"Evidence decision: {decision}\nQuestion: {question}\n\n{evidence}"
    )


class ResearchPipeline:
    def __init__(
        self,
        root,
        providers: dict[str, ModelProvider],
        *,
        logger: ExperimentLogger | None = None,
        retriever: PaperRetriever | None = None,
    ) -> None:
        self.retriever = retriever or PaperRetriever(root)
        self.providers = providers
        self.logger = logger
        self.gate = EvidenceGate()

    async def ask(
        self,
        question: str,
        *,
        provider_name: str = "stand-in",
        retrieval_setting: str = "B",
        request_id: str | None = None,
        max_retries: int = 2,
        scenario: str = "interactive",
    ) -> AskResult:
        request_id = request_id or str(uuid.uuid4())
        arrival_ms = monotonic_ms()
        wall_clock_start = now_iso_hires()
        source_label = "real" if provider_name == "groq" else "stand-in"
        _logged = False

        # Step 1: Retrieval
        retrieval_start_ms = arrival_ms
        passages = self.retriever.search(question, retrieval_setting)
        retrieval_end_ms = monotonic_ms()
        retrieval_ms = retrieval_end_ms - retrieval_start_ms

        # Step 2: Evidence Gating
        decision = self.gate.decide(passages, question)

        # Early refusal branch: insufficient evidence
        if decision == "insufficient":
            final_response_ms = monotonic_ms()
            total_ms = final_response_ms - arrival_ms
            answer = "I cannot answer this from the approved research papers."
            status = "ok"

            if self.logger and not _logged:
                _logged = True
                await self.logger.result(
                    request_id=request_id,
                    scenario=scenario,
                    source=source_label,
                    start_time=wall_clock_start,
                    first_token_ms="",
                    total_ms=round(total_ms, 2),
                    status=status,
                    retries=0,
                    tokens_in="",
                    tokens_out="",
                    retrieval_setting=retrieval_setting,
                    answer_grade="refused_correctly",
                )
                diag = DiagnosticEntry(
                    arrival_ms=arrival_ms,
                    queue_exit_ms=arrival_ms,
                    retrieval_start_ms=retrieval_start_ms,
                    retrieval_end_ms=retrieval_end_ms,
                    provider_start_ms=retrieval_end_ms,
                    first_token_ms=0.0,
                    provider_end_ms=retrieval_end_ms,
                    verification_start_ms=retrieval_end_ms,
                    verification_end_ms=retrieval_end_ms,
                    final_response_ms=final_response_ms,
                    retries=0,
                    decision=decision,
                    wall_clock_start=wall_clock_start,
                )
                await self.logger.diagnostic_entry(request_id, diag.to_dict(request_id))

            timing = {
                "retrieval_ms": retrieval_ms,
                "total_ms": total_ms,
            }
            return AskResult(
                request_id, answer, [], passages, decision, provider_name, timing, status
            )

        provider = self.providers.get(provider_name)
        if provider is None:
            raise ValueError(f"Unknown provider: {provider_name}")

        request = GenerationRequest(
            request_id=request_id,
            question=question,
            prompt=build_prompt(question, passages, decision),
            retrieval_setting=retrieval_setting,
        )

        provider_start_ms = monotonic_ms()
        retries = 0
        pieces: list[str] = []
        first_token_ms_val: float | None = None

        # Step 3: Provider Generation with Bounded Retries and Terminal Failure Logging
        try:
            while True:
                try:
                    async for piece in provider.stream(request):
                        if first_token_ms_val is None:
                            first_token_ms_val = monotonic_ms()
                        pieces.append(piece)
                    break
                except ProviderError as exc:
                    if exc.kind in {ErrorKind.DAILY_QUOTA, ErrorKind.PERMANENT} or retries >= max_retries:
                        raise
                    retries += 1
                    backoff = exc.retry_after or ((0.05 * 2**retries) + random.random() * 0.05)
                    await asyncio.sleep(backoff)
        except ProviderError as exc:
            # F-09: Log terminal failures to results.csv
            final_response_ms = monotonic_ms()
            total_ms = final_response_ms - arrival_ms
            if exc.kind in {ErrorKind.RATE_LIMITED, ErrorKind.DAILY_QUOTA}:
                term_status = "rate_limited"
            elif exc.kind == ErrorKind.TIMEOUT:
                term_status = "timeout"
            else:
                term_status = "error"

            if self.logger and not _logged:
                _logged = True
                await self.logger.result(
                    request_id=request_id,
                    scenario=scenario,
                    source=source_label,
                    start_time=wall_clock_start,
                    first_token_ms="",
                    total_ms=round(total_ms, 2),
                    status=term_status,
                    retries=retries,
                    tokens_in="",
                    tokens_out="",
                    retrieval_setting=retrieval_setting,
                    answer_grade="",
                )
                diag = DiagnosticEntry(
                    arrival_ms=arrival_ms,
                    queue_exit_ms=arrival_ms,
                    retrieval_start_ms=retrieval_start_ms,
                    retrieval_end_ms=retrieval_end_ms,
                    provider_start_ms=provider_start_ms,
                    first_token_ms=first_token_ms_val or 0.0,
                    provider_end_ms=final_response_ms,
                    verification_start_ms=final_response_ms,
                    verification_end_ms=final_response_ms,
                    final_response_ms=final_response_ms,
                    retries=retries,
                    decision=decision,
                    wall_clock_start=wall_clock_start,
                )
                await self.logger.diagnostic_entry(request_id, diag.to_dict(request_id))
            raise

        provider_end_ms = monotonic_ms()
        raw_answer = "".join(pieces).strip()

        # Step 4: Verification
        verification_start_ms = monotonic_ms()
        verification: VerificationResult = verify_answer(raw_answer, passages)
        verification_end_ms = monotonic_ms()
        final_response_ms = monotonic_ms()

        # Determine status and final answer before logging (F-01, F-06)
        if verification.valid:
            status = "ok"
            answer = raw_answer
        elif verification.abstention:
            status = "ok"
            answer = raw_answer
        else:
            status = "error"
            answer = "I cannot provide a verified answer from the retrieved evidence."

        first_token_delay_ms = (
            (first_token_ms_val - provider_start_ms) if first_token_ms_val is not None else 0.0
        )
        total_ms = final_response_ms - arrival_ms

        timing = {
            "retrieval_ms": round(retrieval_ms, 2),
            "provider_first_token_ms": round(first_token_delay_ms, 2),
            "verification_ms": round(verification_end_ms - verification_start_ms, 2),
            "total_ms": round(total_ms, 2),
        }

        # F-05: Token accounting
        tokens_in: int | str = ""
        tokens_out: int | str = ""
        if hasattr(provider, "last_usage") and provider.last_usage:
            usage = provider.last_usage
            if isinstance(usage, dict):
                tokens_in = usage.get("prompt_tokens", "")
                tokens_out = usage.get("completion_tokens", "")

        # F-04: Header logging if available
        if hasattr(provider, "last_response_headers") and provider.last_response_headers:
            if self.logger:
                await self.logger.headers(request_id, provider.last_response_headers)

        if self.logger and not _logged:
            _logged = True
            await self.logger.result(
                request_id=request_id,
                scenario=scenario,
                source=source_label,
                start_time=wall_clock_start,
                first_token_ms=round(first_token_delay_ms, 2) if first_token_ms_val is not None else "",
                total_ms=round(total_ms, 2),
                status=status,
                retries=retries,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                retrieval_setting=retrieval_setting,
                answer_grade="",
            )
            diag = DiagnosticEntry(
                arrival_ms=arrival_ms,
                queue_exit_ms=arrival_ms,
                retrieval_start_ms=retrieval_start_ms,
                retrieval_end_ms=retrieval_end_ms,
                provider_start_ms=provider_start_ms,
                first_token_ms=first_token_ms_val or 0.0,
                provider_end_ms=provider_end_ms,
                verification_start_ms=verification_start_ms,
                verification_end_ms=verification_end_ms,
                final_response_ms=final_response_ms,
                retries=retries,
                decision=decision,
                wall_clock_start=wall_clock_start,
            )
            await self.logger.diagnostic_entry(request_id, diag.to_dict(request_id))

        return AskResult(
            request_id=request_id,
            answer=answer,
            citations=verification.citations,
            evidence=passages,
            decision=decision,
            provider=provider_name,
            timing=timing,
            status=status,
        )
