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


def _parse_retry_after(raw: object, default: float) -> float:
    """Safely parse Retry-After header/attribute without arbitrary caps."""
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return max(0.0, float(raw))
    if isinstance(raw, str):
        val = raw.strip()
        try:
            return max(0.0, float(val))
        except ValueError:
            if val.endswith("s"):
                try:
                    return max(0.0, float(val[:-1]))
                except ValueError:
                    pass
    return default


class AdmissionController:
    """Bounded queue, concurrency limiter, and token-aware admission control."""

    def __init__(
        self,
        *,
        max_concurrency: int = 20,
        max_queue_depth: int = 50,
        tokens_per_minute: int | None = None,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.max_queue_depth = max_queue_depth
        self.tokens_per_minute = tokens_per_minute
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._waiting_count = 0
        self._tokens_history: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    @property
    def queue_length(self) -> int:
        return self._waiting_count

    async def acquire(self, estimated_tokens: int = 2500) -> float:
        """Attempt admission into the processing pipeline.

        Returns queue_exit_ms (monotonic ms).
        Raises ProviderError(ErrorKind.RATE_LIMITED) if queue is full or TPM exceeded.
        """
        async with self._lock:
            # Check token budget over rolling 60-second window
            if self.tokens_per_minute:
                now_s = time.monotonic()
                self._tokens_history = [(t, n) for t, n in self._tokens_history if now_s - t < 60.0]
                current_tpm = sum(n for _, n in self._tokens_history)
                if current_tpm + estimated_tokens > self.tokens_per_minute:
                    raise ProviderError(
                        f"Admission control: estimated tokens ({estimated_tokens}) would exceed TPM limit "
                        f"({current_tpm}/{self.tokens_per_minute})",
                        ErrorKind.RATE_LIMITED,
                    )

            if self._waiting_count >= self.max_queue_depth:
                raise ProviderError(
                    f"Admission control: queue full ({self._waiting_count} >= {self.max_queue_depth})",
                    ErrorKind.RATE_LIMITED,
                )
            self._waiting_count += 1

        try:
            await self._semaphore.acquire()
            queue_exit_ms = monotonic_ms()
            async with self._lock:
                now_s = time.monotonic()
                self._tokens_history.append((now_s, estimated_tokens))
            return queue_exit_ms
        finally:
            async with self._lock:
                self._waiting_count = max(0, self._waiting_count - 1)

    def release(self) -> None:
        self._semaphore.release()


class ResearchPipeline:
    def __init__(
        self,
        root,
        providers: dict[str, ModelProvider],
        *,
        logger: ExperimentLogger | None = None,
        retriever: PaperRetriever | None = None,
        admission: AdmissionController | None = None,
    ) -> None:
        self.root = root
        self.retriever = retriever or PaperRetriever(root)
        self.providers = providers
        self.logger = logger
        self.gate = EvidenceGate()
        self.admission = admission or AdmissionController()
        self._benchmark_map: dict[str, dict] = {}
        try:
            import json
            bm_path = root / "data" / "benchmarks" / "benchmark_draft.json"
            if bm_path.exists():
                bm_data = json.loads(bm_path.read_text(encoding="utf-8"))
                for q in bm_data.get("questions", []):
                    self._benchmark_map[q["question"]] = q
                    if "id" in q:
                        self._benchmark_map[q["id"]] = q
        except Exception:
            pass

    async def ask(
        self,
        question: str,
        *,
        provider_name: str = "stand-in",
        retrieval_setting: str = "B",
        request_id: str | None = None,
        max_retries: int = 2,
        scenario: str = "interactive",
        benchmark_item: dict | None = None,
        timeout_seconds: float = 30.0,
    ) -> AskResult:
        request_id = request_id or str(uuid.uuid4())
        arrival_ms = monotonic_ms()
        wall_clock_start = now_iso_hires()
        source_label = "real" if provider_name == "groq" else "stand-in"
        _logged = False
        queue_exit_ms = arrival_ms

        bm_info = benchmark_item or self._benchmark_map.get(question) or self._benchmark_map.get(request_id or "")

        # Step 0: Admission control & Queueing
        try:
            queue_exit_ms = await self.admission.acquire(estimated_tokens=2500)
        except ProviderError as exc:
            final_response_ms = monotonic_ms()
            total_ms = final_response_ms - arrival_ms
            term_status = "rate_limited"
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
                    retries=0,
                    tokens_in="",
                    tokens_out="",
                    retrieval_setting=retrieval_setting,
                    answer_grade="",
                )
                diag = DiagnosticEntry(
                    arrival_ms=arrival_ms,
                    queue_exit_ms=queue_exit_ms,
                    retrieval_start_ms=queue_exit_ms,
                    retrieval_end_ms=queue_exit_ms,
                    provider_start_ms=queue_exit_ms,
                    first_token_ms=0.0,
                    provider_end_ms=final_response_ms,
                    verification_start_ms=final_response_ms,
                    verification_end_ms=final_response_ms,
                    final_response_ms=final_response_ms,
                    retries=0,
                    decision="rejected_at_admission",
                    failure_reason=str(exc),
                    wall_clock_start=wall_clock_start,
                )
                await self.logger.diagnostic_entry(request_id, diag.to_dict(request_id))
            raise

        try:
            # Step 1: Retrieval
            retrieval_start_ms = monotonic_ms()
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

                answer_grade = ""
                failure_reason = ""
                _load_scenarios = {
                    "steady_load", "sudden_burst", "sustained_overload",
                    "per_minute_limit_halved", "daily_budget_exhausted",
                    "retry_baseline", "retry_amplified", "retry_backoff",
                    "groq_probe",
                }
                if bm_info:
                    expected_ans = bm_info.get("expected_answerability", "")
                    if expected_ans == "unanswerable":
                        answer_grade = "refused_correctly"
                    else:
                        answer_grade = "refused_wrongly"
                        failure_reason = f"Gate refused answerable question (expected {expected_ans})"
                elif scenario in _load_scenarios:
                    answer_grade = ""
                else:
                    # Generic / unsupported query outside approved library
                    answer_grade = "refused_correctly"

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
                        answer_grade=answer_grade,
                    )
                    diag = DiagnosticEntry(
                        arrival_ms=arrival_ms,
                        queue_exit_ms=queue_exit_ms,
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
                        failure_reason=failure_reason,
                        wall_clock_start=wall_clock_start,
                    )
                    await self.logger.diagnostic_entry(request_id, diag.to_dict(request_id))

                timing = {
                    "queue_wait_ms": round(queue_exit_ms - arrival_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "total_ms": round(total_ms, 2),
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
                timeout_seconds=timeout_seconds,
            )

            deadline_ms = monotonic_ms() + (request.timeout_seconds * 1000.0)
            provider_start_ms = monotonic_ms()
            retries = 0
            pieces: list[str] = []
            first_token_ms_val: float | None = None

            # Step 3: Provider Generation with Bounded Retries, State Resets and Deadline
            try:
                while True:
                    remaining_ms = deadline_ms - monotonic_ms()
                    if remaining_ms <= 0:
                        raise ProviderError("Streaming deadline exceeded", ErrorKind.TIMEOUT)

                    # Reset partial output and first-token state on EVERY attempt
                    pieces = []
                    first_token_ms_val = None

                    try:
                        async for piece in provider.stream(request):
                            if first_token_ms_val is None:
                                first_token_ms_val = monotonic_ms()
                            pieces.append(piece)
                            if monotonic_ms() > deadline_ms:
                                raise ProviderError("Streaming deadline exceeded", ErrorKind.TIMEOUT)
                        break
                    except ProviderError as exc:
                        if exc.kind in {ErrorKind.DAILY_QUOTA, ErrorKind.PERMANENT} or retries >= max_retries:
                            raise
                        retries += 1
                        backoff = _parse_retry_after(
                            exc.retry_after,
                            default=((0.05 * 2**retries) + random.uniform(0.01, 0.05)),
                        )
                        # Respect deadline: check if backoff would exceed remaining time
                        if (monotonic_ms() + (backoff * 1000.0)) > deadline_ms:
                            raise ProviderError(
                                f"Retry delay ({backoff:.2f}s) would exceed streaming deadline",
                                ErrorKind.TIMEOUT,
                            )
                        await asyncio.sleep(backoff)
            except ProviderError as exc:
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
                        queue_exit_ms=queue_exit_ms,
                        retrieval_start_ms=retrieval_start_ms,
                        retrieval_end_ms=retrieval_end_ms,
                        provider_start_ms=provider_start_ms,
                        first_token_ms=0.0,
                        provider_end_ms=final_response_ms,
                        verification_start_ms=final_response_ms,
                        verification_end_ms=final_response_ms,
                        final_response_ms=final_response_ms,
                        retries=retries,
                        decision=decision,
                        failure_reason=str(exc),
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

            # Determine status and final answer
            if verification.valid:
                status = "ok"
                answer = raw_answer
            elif verification.abstention:
                status = "ok"
                answer = raw_answer
            else:
                status = "error"
                answer = "I cannot provide a verified answer from the retrieved evidence."

            # Latency semantics: measure end-to-end TTFT from request arrival
            e2e_ttft_ms = (first_token_ms_val - arrival_ms) if first_token_ms_val is not None else None
            provider_ttft_ms = (first_token_ms_val - provider_start_ms) if first_token_ms_val is not None else 0.0
            total_ms = final_response_ms - arrival_ms

            timing = {
                "queue_wait_ms": round(queue_exit_ms - arrival_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "provider_first_token_ms": round(provider_ttft_ms, 2),
                "verification_ms": round(verification_end_ms - verification_start_ms, 2),
                "total_ms": round(total_ms, 2),
            }

            # Retrieve request-scoped metadata (and clean up from provider)
            meta: dict = {}
            if hasattr(provider, "pop_request_metadata"):
                meta = provider.pop_request_metadata(request_id)
            elif hasattr(provider, "get_request_metadata"):
                meta = provider.get_request_metadata(request_id)

            usage = meta.get("usage")
            if usage is None and hasattr(provider, "last_usage") and provider.last_usage:
                usage = provider.last_usage

            tokens_in: int | str = ""
            tokens_out: int | str = ""
            if isinstance(usage, dict):
                tokens_in = usage.get("prompt_tokens", "")
                tokens_out = usage.get("completion_tokens", "")

            # Headers logging
            headers = meta.get("headers")
            if not headers and hasattr(provider, "last_response_headers"):
                headers = provider.last_response_headers
            if headers and self.logger:
                await self.logger.headers(request_id, headers)

            # Benchmark answer grading
            answer_grade = ""
            failure_reason = ""
            if bm_info:
                expected_ans = bm_info.get("expected_answerability", "")
                if expected_ans == "unanswerable":
                    if verification.abstention or status == "error":
                        answer_grade = "refused_correctly"
                    else:
                        answer_grade = "wrong"
                        failure_reason = "Model answered an unanswerable question"
                else:  # answerable or partially_supported
                    if verification.abstention:
                        answer_grade = "refused_wrongly"
                        failure_reason = "Model abstained on answerable question"
                    elif verification.valid and status == "ok":
                        answer_grade = "correct"
                    else:
                        answer_grade = "wrong"
                        failure_reason = f"Verification failed: {verification.reason}"
            elif scenario.startswith("benchmark_"):
                answer_grade = "correct" if (verification.valid and status == "ok") else "wrong"
                if answer_grade == "wrong":
                    failure_reason = f"Verification failed: {verification.reason}"

            if self.logger and not _logged:
                _logged = True
                await self.logger.result(
                    request_id=request_id,
                    scenario=scenario,
                    source=source_label,
                    start_time=wall_clock_start,
                    first_token_ms=round(e2e_ttft_ms, 2) if e2e_ttft_ms is not None else "",
                    total_ms=round(total_ms, 2),
                    status=status,
                    retries=retries,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    retrieval_setting=retrieval_setting,
                    answer_grade=answer_grade,
                )
                diag = DiagnosticEntry(
                    arrival_ms=arrival_ms,
                    queue_exit_ms=queue_exit_ms,
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
                    failure_reason=failure_reason,
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
        finally:
            self.admission.release()
