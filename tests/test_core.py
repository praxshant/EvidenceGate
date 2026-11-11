"""Comprehensive test suite for EvidenceGate.

Covers all 30 core test scenarios:
 1. Complete successful pipeline request
 2. Correctly refused unsupported request
 3. Partially supported request
 4. Invalid citation rejection (unapproved paper, wrong page, wrong section)
 5. Unsupported claim handling (fabricated claim with valid citation format)
 6. Valid citation acceptance
 7. Pipeline exception handling
 8. Failure result logging
 9. Rate-limit result logging
10. Scenario propagation
11. Groq client using mocked streaming responses
12. Successful response-header capture
13. Token usage extraction
14. Stand-in replay success
15. Missing replay record handling
16. Stand-in streaming
17. Stand-in transient failure
18. Stand-in timeout
19. Request rate limiting
20. Token rate limiting
21. Daily quota exhaustion and recovery
22. Load-dependent slowdown
23. Concurrent logging integrity
24. CSV schema compliance
25. Diagnostic timing consistency
26. FastAPI /health
27. FastAPI /ask with mocked/stand-in provider
28. Latency invariant: total_ms >= first_token_ms for every row in results.csv
29. GroqProvider call uses include_reasoning and reasoning_effort (not reasoning_format)
30. Scenario summary never produces ttft_p50 > total_p50 on paired population
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.logging_utils import RESULT_COLUMNS, DiagnosticEntry, ExperimentLogger, monotonic_ms, now_iso_hires
from src.pipeline import EvidenceGate, ResearchPipeline
from src.providers.base import ErrorKind, GenerationRequest, ModelProvider, ProviderError
from src.providers.groq_client import GroqProvider
from src.providers.standin_client import ReplayStore, StandinConfig, StandinProvider
from src.retrieval import PaperRetriever, Passage
from src.verification import VerificationResult, is_abstention, verify_answer

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def retriever() -> PaperRetriever:
    return PaperRetriever(ROOT)


@pytest.fixture
def sample_passages() -> list[Passage]:
    return [
        Passage(
            paper="Attention Is All You Need",
            title="Attention Is All You Need",
            arxiv_id="1706.03762",
            version=7,
            section="Abstract",
            page=1,
            text="The Transformer is based solely on attention mechanisms, dispensing with recurrence.",
            score=15.0,
        ),
        Passage(
            paper="Attention Is All You Need",
            title="Attention Is All You Need",
            arxiv_id="1706.03762",
            version=7,
            section="3.2.1 Scaled Dot-Product Attention",
            page=4,
            text="We compute the attention function on a set of queries simultaneously, packed into a matrix Q.",
            score=8.0,
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Complete successful pipeline request
# ---------------------------------------------------------------------------

def test_pipeline_successful_request(tmp_path: Path, retriever: PaperRetriever) -> None:
    rep_file = tmp_path / "responses.jsonl"
    q = "What is scaled dot-product attention?"
    ans = "[Attention Is All You Need, §Abstract, p.1] The Transformer uses attention without recurrence."
    rep_file.write_text(json.dumps({"question": q, "answer": ans, "tokens_in": 12, "tokens_out": 18}), encoding="utf-8")

    standin = StandinProvider(ReplayStore(rep_file), StandinConfig(base_delay_seconds=0, jitter_seconds=0, inter_chunk_seconds=0))
    logger = ExperimentLogger(tmp_path)
    pipe = ResearchPipeline(ROOT, {"stand-in": standin}, retriever=retriever, logger=logger)

    result = asyncio.run(pipe.ask(q, provider_name="stand-in", scenario="success_test"))
    assert result.status == "ok"
    assert result.decision in {"partial", "sufficient"}
    assert len(result.citations) == 1
    assert "Attention Is All You Need" in result.answer

    csv_lines = (tmp_path / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 2  # header + 1 row
    row = csv_lines[1].split(",")
    assert row[1] == "success_test"
    assert row[6] == "ok"
    assert row[8] == "12"
    assert row[9] == "18"


# ---------------------------------------------------------------------------
# 2. Correctly refused unsupported request
# ---------------------------------------------------------------------------

def test_pipeline_refuses_unsupported_request(tmp_path: Path, retriever: PaperRetriever) -> None:
    standin = StandinProvider(ReplayStore(tmp_path / "empty.jsonl"))
    logger = ExperimentLogger(tmp_path)
    pipe = ResearchPipeline(ROOT, {"stand-in": standin}, retriever=retriever, logger=logger)

    q = "What is the recipe for chocolate chip cookies?"
    result = asyncio.run(pipe.ask(q, provider_name="stand-in", scenario="unsupported_test"))
    assert result.status == "ok"
    assert result.decision == "insufficient"
    assert "cannot answer" in result.answer.lower()

    csv_lines = (tmp_path / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 2
    row = csv_lines[1].split(",")
    assert row[1] == "unsupported_test"
    assert row[6] == "ok"
    assert row[11] == "refused_correctly"


# ---------------------------------------------------------------------------
# 3. Partially supported request
# ---------------------------------------------------------------------------

def test_evidence_gate_three_decisions() -> None:
    gate = EvidenceGate(minimum_score=2.0)
    assert gate.decide([]) == "insufficient"

    low_p = [Passage("P", "T", "1", 1, "S", 1, "text", score=1.0)]
    assert gate.decide(low_p) == "insufficient"

    part_p = [Passage("P", "T", "1", 1, "S", 1, "text", score=2.5)]
    assert gate.decide(part_p) == "partial"

    suff_p = [
        Passage("P", "T", "1", 1, "S", 1, "query term word", score=10.0),
        Passage("P", "T", "1", 1, "S", 2, "query term word", score=6.0),
    ]
    assert gate.decide(suff_p, "query term word") == "sufficient"


# ---------------------------------------------------------------------------
# 4. Invalid citation rejection (unapproved paper, wrong page, wrong section)
# ---------------------------------------------------------------------------

def test_citation_verification_rejects_invalid_citations(sample_passages: list[Passage]) -> None:
    # Unapproved paper
    res1 = verify_answer("[GPT-4 Technical Report, §1, p.1] claim", sample_passages)
    assert not res1.valid
    assert "Unapproved paper" in res1.reason

    # Wrong page number
    res2 = verify_answer("[Attention Is All You Need, §Abstract, p.99] claim", sample_passages)
    assert not res2.valid
    assert "not in retrieved evidence" in res2.reason

    # Wrong section (that is completely unrelated)
    res3 = verify_answer("[Attention Is All You Need, §Nonexistent Section, p.1] claim", sample_passages)
    assert not res3.valid
    assert "does not match evidence" in res3.reason


# ---------------------------------------------------------------------------
# 5. Unsupported claim handling (fabricated claim with citation syntax)
# ---------------------------------------------------------------------------

def test_citation_verification_rejects_groundless_claim(sample_passages: list[Passage]) -> None:
    # Syntactically valid citation but completely fabricated sentence
    fake_ans = "Dinosaurs engineered medieval catapults on Mars [Attention Is All You Need, §Abstract, p.1]."
    res = verify_answer(fake_ans, sample_passages)
    assert not res.valid
    assert "not grounded in cited evidence" in res.reason


# ---------------------------------------------------------------------------
# 6. Valid citation acceptance
# ---------------------------------------------------------------------------

def test_citation_verification_accepts_valid_citation(sample_passages: list[Passage]) -> None:
    ans = "The Transformer relies entirely on attention mechanisms [Attention Is All You Need, §Abstract, p.1]."
    res = verify_answer(ans, sample_passages)
    assert res.valid
    assert len(res.citations) == 1

    # Abstention verification
    abstention_ans = "The provided evidence does not establish the training hardware used."
    res_abst = verify_answer(abstention_ans, sample_passages)
    assert res_abst.valid
    assert res_abst.abstention


# ---------------------------------------------------------------------------
# 7. Pipeline exception handling
# ---------------------------------------------------------------------------

def test_pipeline_exception_handling(tmp_path: Path, retriever: PaperRetriever) -> None:
    logger = ExperimentLogger(tmp_path)
    pipe = ResearchPipeline(ROOT, {}, retriever=retriever, logger=logger)

    # Unknown provider
    with pytest.raises(ValueError, match="Unknown provider"):
        asyncio.run(pipe.ask("What is attention?", provider_name="nonexistent"))


# ---------------------------------------------------------------------------
# 8. Failure result logging
# ---------------------------------------------------------------------------

def test_pipeline_failure_logging(tmp_path: Path, retriever: PaperRetriever) -> None:
    # Stand-in with empty store raises ProviderError (PERMANENT)
    standin = StandinProvider(ReplayStore(tmp_path / "empty.jsonl"))
    logger = ExperimentLogger(tmp_path)
    pipe = ResearchPipeline(ROOT, {"stand-in": standin}, retriever=retriever, logger=logger)

    with pytest.raises(ProviderError):
        asyncio.run(pipe.ask("What is scaled dot-product attention?", provider_name="stand-in", scenario="fail_scenario"))

    csv_lines = (tmp_path / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 2
    row = csv_lines[1].split(",")
    assert row[1] == "fail_scenario"
    assert row[6] == "error"
    assert row[11] == ""  # answer_grade empty for failed requests


# ---------------------------------------------------------------------------
# 9. Rate-limit result logging
# ---------------------------------------------------------------------------

def test_pipeline_rate_limit_logging(tmp_path: Path, retriever: PaperRetriever) -> None:
    rep_file = tmp_path / "responses.jsonl"
    q = "What is scaled dot-product attention?"
    ans = "[Attention Is All You Need, §Abstract, p.1] Attention mechanism."
    rep_file.write_text(json.dumps({"question": q, "answer": ans}), encoding="utf-8")

    standin = StandinProvider(
        ReplayStore(rep_file),
        StandinConfig(requests_per_minute=1, base_delay_seconds=0, jitter_seconds=0, inter_chunk_seconds=0),
    )
    logger = ExperimentLogger(tmp_path)
    pipe = ResearchPipeline(ROOT, {"stand-in": standin}, retriever=retriever, logger=logger)

    # 1st request succeeds
    asyncio.run(pipe.ask(q, provider_name="stand-in", scenario="rl_scenario"))
    # 2nd request hits rate limit
    with pytest.raises(ProviderError):
        asyncio.run(pipe.ask(q, provider_name="stand-in", scenario="rl_scenario", max_retries=0))

    csv_lines = (tmp_path / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 3
    r2 = csv_lines[2].split(",")
    assert r2[1] == "rl_scenario"
    assert r2[6] == "rate_limited"


# ---------------------------------------------------------------------------
# 10. Scenario propagation
# ---------------------------------------------------------------------------

def test_scenario_propagation(tmp_path: Path, retriever: PaperRetriever) -> None:
    rep_file = tmp_path / "responses.jsonl"
    q = "What is scaled dot-product attention?"
    ans = "[Attention Is All You Need, §Abstract, p.1] Attention mechanism."
    rep_file.write_text(json.dumps({"question": q, "answer": ans}), encoding="utf-8")

    standin = StandinProvider(ReplayStore(rep_file), StandinConfig(base_delay_seconds=0, jitter_seconds=0, inter_chunk_seconds=0))
    logger = ExperimentLogger(tmp_path)
    pipe = ResearchPipeline(ROOT, {"stand-in": standin}, retriever=retriever, logger=logger)

    asyncio.run(pipe.ask(q, provider_name="stand-in", scenario="steady_load_v1"))
    csv_lines = (tmp_path / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[1].split(",")[1] == "steady_load_v1"


# ---------------------------------------------------------------------------
# 11. Groq client using mocked streaming responses
# ---------------------------------------------------------------------------

class MockAsyncStream:
    def __init__(self, chunks: list, headers: dict | None = None) -> None:
        self._chunks = chunks
        self.response = MagicMock(headers=headers or {})

    async def __aiter__(self):
        for c in self._chunks:
            yield c


def test_groq_client_mock_streaming() -> None:
    provider = GroqProvider(api_key="mock_key", model="llama-3.1-8b-instant")

    # Build mock chunks
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta = MagicMock(content="Hello ")
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta = MagicMock(content="world!")
    chunk2.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    stream_obj = MockAsyncStream(
        [chunk1, chunk2],
        headers={"x-ratelimit-remaining-requests": "29", "retry-after": "5"},
    )

    provider._client = MagicMock()
    provider._client.chat.completions.create = AsyncMock(return_value=stream_obj)

    async def run_stream():
        pieces = []
        async for piece in provider.stream(GenerationRequest("req-1", "hi", "prompt")):
            pieces.append(piece)
        return pieces

    pieces = asyncio.run(run_stream())
    assert "".join(pieces) == "Hello world!"
    assert provider.last_usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert provider.last_response_headers["x-ratelimit-remaining-requests"] == "29"


# ---------------------------------------------------------------------------
# 12. Successful response-header capture
# ---------------------------------------------------------------------------

def test_groq_header_filtering() -> None:
    provider = GroqProvider(api_key="mock_key", model="llama-3.1-8b-instant")
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock(content="ok")
    chunk.usage = None

    stream_obj = MockAsyncStream(
        [chunk],
        headers={
            "x-ratelimit-limit-requests": "30",
            "x-ratelimit-remaining-requests": "29",
            "unrelated-server-header": "cloudflare",
        },
    )
    provider._client.chat.completions.create = AsyncMock(return_value=stream_obj)

    async def run_call():
        return [p async for p in provider.stream(GenerationRequest("r", "q", "p"))]

    asyncio.run(run_call())
    headers = provider.last_response_headers
    assert "x-ratelimit-limit-requests" in headers
    assert "x-ratelimit-remaining-requests" in headers
    assert "unrelated-server-header" not in headers


# ---------------------------------------------------------------------------
# 13. Token usage extraction
# ---------------------------------------------------------------------------

def test_token_usage_extraction() -> None:
    provider = GroqProvider(api_key="mock_key", model="llama-3.1-8b-instant")
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock(content="token check")
    chunk.usage = MagicMock(prompt_tokens=42, completion_tokens=17)

    stream_obj = MockAsyncStream([chunk], headers={})
    provider._client.chat.completions.create = AsyncMock(return_value=stream_obj)

    async def run_call():
        return [p async for p in provider.stream(GenerationRequest("r", "q", "p"))]

    asyncio.run(run_call())
    assert provider.last_usage == {"prompt_tokens": 42, "completion_tokens": 17}


# ---------------------------------------------------------------------------
# 14. Stand-in replay success
# ---------------------------------------------------------------------------

def test_standin_replay_success(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    path.write_text(
        json.dumps({"question": "Explain attention", "answer": "[Attention Is All You Need, §Abstract, p.1] Attention.", "tokens_in": 5, "tokens_out": 10}),
        encoding="utf-8",
    )
    provider = StandinProvider(ReplayStore(path), StandinConfig(base_delay_seconds=0, jitter_seconds=0, inter_chunk_seconds=0))
    request = GenerationRequest("req", "Explain attention", "prompt")

    async def run_call():
        return [p async for p in provider.stream(request)]

    chunks = asyncio.run(run_call())
    assert "".join(chunks) == "[Attention Is All You Need, §Abstract, p.1] Attention."
    assert provider.last_usage == {"prompt_tokens": 5, "completion_tokens": 10}


# ---------------------------------------------------------------------------
# 15. Missing replay record handling
# ---------------------------------------------------------------------------

def test_standin_missing_replay_record(tmp_path: Path) -> None:
    provider = StandinProvider(ReplayStore(tmp_path / "empty.jsonl"))
    request = GenerationRequest("req", "Unknown question?", "prompt")

    async def run_call():
        return [p async for p in provider.stream(request)]

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(run_call())
    assert exc_info.value.kind == ErrorKind.PERMANENT
    assert "No replay record found" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 16. Stand-in streaming (multiple chunks)
# ---------------------------------------------------------------------------

def test_standin_streaming_chunks(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    long_text = "A" * 64
    path.write_text(json.dumps({"question": "q", "answer": long_text}), encoding="utf-8")
    provider = StandinProvider(
        ReplayStore(path),
        StandinConfig(chunk_size=16, base_delay_seconds=0, jitter_seconds=0, inter_chunk_seconds=0),
    )

    async def run_call():
        return [p async for p in provider.stream(GenerationRequest("r", "q", "p"))]

    pieces = asyncio.run(run_call())
    assert len(pieces) == 4
    assert "".join(pieces) == long_text


# ---------------------------------------------------------------------------
# 17. Stand-in transient failure
# ---------------------------------------------------------------------------

def test_standin_transient_failure(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    path.write_text(json.dumps({"question": "q", "answer": "ans"}), encoding="utf-8")
    provider = StandinProvider(
        ReplayStore(path),
        StandinConfig(transient_failure_rate=1.0, base_delay_seconds=0, jitter_seconds=0, inter_chunk_seconds=0),
    )

    async def run_call():
        return [p async for p in provider.stream(GenerationRequest("r", "q", "p"))]

    with pytest.raises(ProviderError) as exc:
        asyncio.run(run_call())
    assert exc.value.kind == ErrorKind.TRANSIENT


# ---------------------------------------------------------------------------
# 18. Stand-in timeout
# ---------------------------------------------------------------------------

def test_standin_timeout_failure(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    path.write_text(json.dumps({"question": "q", "answer": "ans"}), encoding="utf-8")
    provider = StandinProvider(
        ReplayStore(path),
        StandinConfig(timeout_rate=1.0, base_delay_seconds=0, jitter_seconds=0, inter_chunk_seconds=0),
    )

    async def run_call():
        return [p async for p in provider.stream(GenerationRequest("r", "q", "p", timeout_seconds=0.01))]

    with pytest.raises(ProviderError) as exc:
        asyncio.run(run_call())
    assert exc.value.kind == ErrorKind.TIMEOUT


# ---------------------------------------------------------------------------
# 19. Request rate limiting
# ---------------------------------------------------------------------------

def test_standin_request_rate_limiting(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    path.write_text(json.dumps({"question": "q", "answer": "ans"}), encoding="utf-8")
    provider = StandinProvider(
        ReplayStore(path),
        StandinConfig(requests_per_minute=2, base_delay_seconds=0, jitter_seconds=0, inter_chunk_seconds=0),
    )

    async def call():
        return [p async for p in provider.stream(GenerationRequest("r", "q", "p"))]

    asyncio.run(call())
    asyncio.run(call())
    with pytest.raises(ProviderError) as exc:
        asyncio.run(call())
    assert exc.value.kind == ErrorKind.RATE_LIMITED
    assert "request-per-minute" in str(exc.value)


# ---------------------------------------------------------------------------
# 20. Token rate limiting
# ---------------------------------------------------------------------------

def test_standin_token_rate_limiting(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    path.write_text(json.dumps({"question": "q", "answer": "ans", "tokens_out": 50}), encoding="utf-8")
    provider = StandinProvider(
        ReplayStore(path),
        StandinConfig(tokens_per_minute=80, base_delay_seconds=0, jitter_seconds=0, inter_chunk_seconds=0),
    )

    async def call():
        return [p async for p in provider.stream(GenerationRequest("r", "q", "p"))]

    asyncio.run(call())  # 50 tokens
    with pytest.raises(ProviderError) as exc:
        asyncio.run(call())  # 50 + 50 = 100 > 80
    assert exc.value.kind == ErrorKind.RATE_LIMITED
    assert "token-per-minute" in str(exc.value)


# ---------------------------------------------------------------------------
# 21. Daily quota exhaustion and recovery
# ---------------------------------------------------------------------------

def test_standin_daily_quota_and_recovery(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    path.write_text(json.dumps({"question": "q", "answer": "ans"}), encoding="utf-8")
    provider = StandinProvider(
        ReplayStore(path),
        StandinConfig(requests_per_day=1, day_seconds=0.05, base_delay_seconds=0, jitter_seconds=0, inter_chunk_seconds=0),
    )

    async def call():
        return [p async for p in provider.stream(GenerationRequest("r", "q", "p"))]

    asyncio.run(call())  # 1st consumes daily quota
    with pytest.raises(ProviderError) as exc:
        asyncio.run(call())  # 2nd hits daily quota
    assert exc.value.kind == ErrorKind.DAILY_QUOTA

    # Test reset_daily_quota() recovery
    provider.reset_daily_quota()
    chunks = asyncio.run(call())
    assert "".join(chunks) == "ans"


# ---------------------------------------------------------------------------
# 22. Load-dependent slowdown
# ---------------------------------------------------------------------------

def test_standin_load_dependent_slowdown(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    path.write_text(json.dumps({"question": "q", "answer": "ans"}), encoding="utf-8")
    provider = StandinProvider(
        ReplayStore(path),
        StandinConfig(base_delay_seconds=0.01, slowdown_per_concurrent=0.05, jitter_seconds=0, inter_chunk_seconds=0),
    )

    async def run_two_concurrent():
        async def consume(req):
            return [p async for p in provider.stream(req)]

        t0 = time.monotonic()
        results = await asyncio.gather(
            consume(GenerationRequest("1", "q", "p")),
            consume(GenerationRequest("2", "q", "p")),
        )
        return time.monotonic() - t0

    # Under concurrency, active >= 1 introduces slowdown
    duration = asyncio.run(run_two_concurrent())
    assert duration > 0.005


# ---------------------------------------------------------------------------
# 23. Concurrent logging integrity
# ---------------------------------------------------------------------------

def test_concurrent_logging_integrity(tmp_path: Path) -> None:
    logger = ExperimentLogger(tmp_path)

    async def write_many(n: int):
        tasks = [
            logger.result(request_id=f"req-{i}", scenario="concurrency_test", status="ok", retries=0)
            for i in range(n)
        ]
        await asyncio.gather(*tasks)

    asyncio.run(write_many(50))

    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as handle:
        reader = list(csv.reader(handle))
        assert len(reader) == 51  # header + 50 rows
        req_ids = {row[0] for row in reader[1:]}
        assert len(req_ids) == 50  # no duplicates or dropped rows


# ---------------------------------------------------------------------------
# 24. CSV schema compliance
# ---------------------------------------------------------------------------

def test_results_csv_exact_schema(tmp_path: Path) -> None:
    logger = ExperimentLogger(tmp_path)
    asyncio.run(logger.result(request_id="r1", scenario="s1", source="stand-in", status="ok"))

    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
        assert header == RESULT_COLUMNS
        assert header == [
            "request_id", "scenario", "source", "start_time", "first_token_ms",
            "total_ms", "status", "retries", "tokens_in", "tokens_out",
            "retrieval_setting", "answer_grade",
        ]


# ---------------------------------------------------------------------------
# 25. Diagnostic timing consistency
# ---------------------------------------------------------------------------

def test_diagnostic_entry_monotonic_timing() -> None:
    diag = DiagnosticEntry(
        arrival_ms=1000.0,
        queue_exit_ms=1000.0,
        retrieval_start_ms=1000.0,
        retrieval_end_ms=1050.0,
        provider_start_ms=1060.0,
        first_token_ms=1100.0,
        provider_end_ms=1200.0,
        verification_start_ms=1205.0,
        verification_end_ms=1210.0,
        final_response_ms=1215.0,
        retries=1,
        decision="sufficient",
        wall_clock_start=now_iso_hires(),
    )
    d = diag.to_dict("req-test")
    assert d["request_id"] == "req-test"
    assert d["total_ms"] == 215.0
    assert d["retrieval_end_ms"] - d["retrieval_start_ms"] == 50.0
    assert d["first_token_ms"] - d["provider_start_ms"] == 40.0


# ---------------------------------------------------------------------------
# 26. FastAPI /health endpoint
# ---------------------------------------------------------------------------

def test_api_health_endpoint() -> None:
    pipe = MagicMock()
    app = create_app(pipe)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 27. FastAPI /ask endpoint with mocked/stand-in provider
# ---------------------------------------------------------------------------

def test_api_ask_endpoint(tmp_path: Path, retriever: PaperRetriever) -> None:
    rep_file = tmp_path / "responses.jsonl"
    q = "What is scaled dot-product attention?"
    ans = "[Attention Is All You Need, §Abstract, p.1] Scaled dot-product attention."
    rep_file.write_text(json.dumps({"question": q, "answer": ans, "tokens_in": 10, "tokens_out": 20}), encoding="utf-8")

    standin = StandinProvider(ReplayStore(rep_file), StandinConfig(base_delay_seconds=0, jitter_seconds=0, inter_chunk_seconds=0))
    pipeline = ResearchPipeline(ROOT, {"stand-in": standin}, retriever=retriever)
    app = create_app(pipeline)
    client = TestClient(app)

    response = client.post("/ask", json={"question": q, "provider": "stand-in", "scenario": "fastapi_test"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == ans
    assert data["status"] == "ok"
    assert data["scenario"] == "fastapi_test"
    assert len(data["evidence"]) > 0
    assert isinstance(data["timing"], dict)


# ---------------------------------------------------------------------------
# 28. Latency invariant: total_ms >= first_token_ms for every row in results.csv
# ---------------------------------------------------------------------------

def test_results_csv_latency_invariants() -> None:
    """Every row where first_token_ms is non-empty must have total_ms >= first_token_ms."""
    csv_path = ROOT / "results.csv"
    if not csv_path.exists():
        pytest.skip("results.csv not yet generated")
    violations = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ttft_s = row.get("first_token_ms")
            total_s = row.get("total_ms")
            if ttft_s and total_s:
                ttft = float(ttft_s)
                total = float(total_s)
                if total < ttft:
                    violations.append(
                        f"request_id={row.get('request_id')} "
                        f"total_ms={total} < first_token_ms={ttft}"
                    )
    assert not violations, "Latency invariant violations found:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# 29. GroqProvider uses include_reasoning and reasoning_effort, not reasoning_format
# ---------------------------------------------------------------------------

def test_groq_provider_uses_correct_reasoning_params() -> None:
    """GroqProvider.stream must use include_reasoning and reasoning_effort,
    not the unsupported reasoning_format parameter."""
    import inspect
    import src.providers.groq_client as gcm

    source = inspect.getsource(gcm.GroqProvider.stream)
    assert "reasoning_format" not in source, (
        "GroqProvider.stream must not use reasoning_format (unsupported for GPT-OSS). "
        "Use include_reasoning and reasoning_effort instead."
    )
    assert "include_reasoning" in source, "GroqProvider.stream must pass include_reasoning"
    assert "reasoning_effort" in source, "GroqProvider.stream must pass reasoning_effort"
    assert "max_completion_tokens" in source, (
        "GroqProvider.stream must use max_completion_tokens, not deprecated max_tokens"
    )


# ---------------------------------------------------------------------------
# 30. Scenario summary never produces ttft_p50 > total_p50 on paired population
# ---------------------------------------------------------------------------

def test_scenario_summary_no_ttft_total_inversion(tmp_path: Path) -> None:
    """Paired latency summary must never show TTFT p50 > total p50."""
    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location(
        "analyze_results",
        str(ROOT / "scripts" / "analyze_results.py"),
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    rows = []
    for i in range(10):
        rows.append({
            "request_id": f"test-{i}",
            "scenario": "steady_load",
            "source": "stand-in",
            "start_time": now_iso_hires(),
            "first_token_ms": str(100.0 + i * 10),
            "total_ms": str(500.0 + i * 10),
            "status": "ok",
            "retries": "0",
            "tokens_in": "50",
            "tokens_out": "20",
            "retrieval_setting": "B",
            "answer_grade": "",
        })

    summary = mod.scenario_summary(rows)
    for s in summary:
        if s.get("scenario") == "steady_load":
            ttft_p50 = s.get("ttft_p50_ms")
            total_p50 = s.get("total_p50_ms")
            if isinstance(ttft_p50, (int, float)) and isinstance(total_p50, (int, float)):
                assert total_p50 >= ttft_p50, (
                    f"Latency population inversion: ttft_p50={ttft_p50} > total_p50={total_p50}"
                )
