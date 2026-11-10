"""Lightweight page-aware BM25-style retrieval."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "can", "does", "for", "how", "is", "of", "the",
    "what", "why", "with", "paper", "report", "reported", "tell",
}

# Regex for numbered section headings (e.g. "1 Introduction", "2.1 Methods")
_SECTION_RE = re.compile(r"^\d+(\.\d+)*\s+[A-Za-z]")
# Exact-match keywords that are always section boundaries
_SECTION_KEYWORDS = {"Abstract", "Introduction", "Conclusion", "References", "Appendix"}
# Punctuation that disqualifies a line from being a section heading
_BAD_ENDINGS = (".", ",", "?")
# A line that is ONLY digits / dots is likely an equation or figure label
_FLOAT_ONLY_RE = re.compile(r"^[\d\.\s]+$")


@dataclass(frozen=True)
class Passage:
    paper: str
    title: str
    arxiv_id: str
    version: int
    section: str
    page: int
    text: str
    score: float


@dataclass(frozen=True)
class RetrievalSetting:
    name: str
    piece_size: int
    top_k: int
    rerank: bool = False


SETTINGS = {
    "A": RetrievalSetting("A", piece_size=300, top_k=3, rerank=False),
    "B": RetrievalSetting("B", piece_size=500, top_k=3, rerank=False),
    "C": RetrievalSetting("C", piece_size=500, top_k=5, rerank=True),
}


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


def normalize_pdf_text(text: str) -> str:
    return re.sub(r"-\s*\n\s*", "", text).replace("\u00ad", "")


def _is_valid_heading(line: str) -> bool:
    """Return True if *line* looks like a genuine section heading.

    Heuristics (F-07 requirement):
    - stripped length < 80 characters
    - does NOT end with '.', ',', or '?'
    - is NOT composed solely of digits / dots (equations, figure labels)
    - matches a known keyword or the numbered-section pattern
    """
    candidate = line.strip()
    if not candidate:
        return False
    if len(candidate) >= 80:
        return False
    if candidate.endswith(_BAD_ENDINGS):
        return False
    if _FLOAT_ONLY_RE.match(candidate):
        return False
    return candidate in _SECTION_KEYWORDS or bool(_SECTION_RE.match(candidate))


class PaperRetriever:
    def __init__(self, root: Path) -> None:
        self.root = root
        metadata = json.loads(
            (root / "data" / "papers" / "metadata.json").read_text(encoding="utf-8")
        )
        self._pages: list[dict[str, object]] = []

        for paper in metadata["papers"]:
            extracted = root / "data" / "processed" / "papers" / (
                Path(str(paper["filename"])).stem + ".json"
            )
            raw_pages = json.loads(extracted.read_text(encoding="utf-8"))["pages"]

            # --- F-07: stateful section tracking ---
            # Build page records first, then do a single sequential pass to
            # assign sections.  For each page we record the section that was
            # *active at the start of the page* (carry-forward), then update
            # current_section with any heading found on that page so it
            # propagates to subsequent pages.
            paper_pages: list[dict[str, object]] = []
            for page in raw_pages:
                text = normalize_pdf_text(str(page["text"])).strip()
                if not text:
                    continue
                paper_pages.append(
                    {
                        "paper": paper["short_name"],
                        "title": paper["short_name"],
                        "arxiv_id": paper["arxiv_id"],
                        "version": paper["version"],
                        "page": page["page"],
                        "text": text,
                        # section filled in below
                        "section": "Unknown",
                    }
                )

            current_section = "Unknown"
            for page_record in paper_pages:
                # Scan this page's lines for any new heading
                for line in str(page_record["text"]).splitlines():
                    if _is_valid_heading(line):
                        current_section = line.strip()
                        break  # first heading on the page wins
                page_record["section"] = current_section

            self._pages.extend(paper_pages)

        # --- F-13: lazy chunk cache keyed by piece_size ---
        # dict[piece_size -> list[tuple[page_metadata, doc_tokens]]]
        self._chunks_cache: dict[int, list[tuple[dict, list[str]]]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunks(text: str, piece_size: int) -> list[str]:
        words = text.split()
        return [" ".join(words[i: i + piece_size]) for i in range(0, len(words), piece_size)]

    def _get_documents(
        self, piece_size: int
    ) -> list[tuple[dict, list[str]]]:
        """Return (metadata, token_list) pairs for *piece_size*, cached."""
        if piece_size not in self._chunks_cache:
            docs: list[tuple[dict, list[str]]] = [
                (metadata | {"text": chunk}, tokens(chunk))
                for metadata in self._pages
                for chunk in self._chunks(str(metadata["text"]), piece_size)
            ]
            self._chunks_cache[piece_size] = docs
        return self._chunks_cache[piece_size]

    # ------------------------------------------------------------------
    # BM25 scoring (unchanged formula)
    # ------------------------------------------------------------------

    @staticmethod
    def _bm25_score(
        query: list[str],
        doc_tokens: list[str],
        document_frequency: dict[str, int],
        total: int,
    ) -> float:
        counts = {term: doc_tokens.count(term) for term in set(query)}
        score = 0.0
        for term, count in counts.items():
            if count:
                idf = math.log((total + 1) / (document_frequency.get(term, 0) + 1)) + 1
                score += idf * count / (count + 1.5)
        normalized_text = " ".join(doc_tokens)
        normalized_query = " ".join(query)
        if normalized_query in normalized_text:
            score += 12.0
        elif len(query) >= 2 and " ".join(query[:2]) in normalized_text:
            score += 3.0
        return score

    # ------------------------------------------------------------------
    # Reranking (F-13)
    # ------------------------------------------------------------------

    @staticmethod
    def _rerank(
        candidates: list[tuple[dict, list[str], float]],
        query: list[str],
        top_k: int,
    ) -> list[tuple[dict, list[str], float]]:
        """Two-stage rerank.

        candidates: list of (metadata, doc_tokens, bm25_score)
        Returns top_k items sorted by combined score (descending).
        """
        if not candidates:
            return []

        max_bm25 = max(score for _, _, score in candidates) or 1.0
        query_terms = set(query)
        query_bigrams: set[str] = set()
        for i in range(len(query) - 1):
            query_bigrams.add(query[i] + " " + query[i + 1])

        reranked: list[tuple[float, dict, list[str], float]] = []
        for metadata, doc_tokens, bm25 in candidates:
            bm25_norm = bm25 / max_bm25

            # Component 2: query term coverage
            doc_token_set = set(doc_tokens)
            coverage = (
                len(query_terms & doc_token_set) / len(query_terms)
                if query_terms
                else 0.0
            )

            # Component 3: exact bigram presence
            doc_text_tokenized = " ".join(doc_tokens)
            bigram_bonus = 1.0 if any(bg in doc_text_tokenized for bg in query_bigrams) else 0.0

            final_score = 0.5 * bm25_norm + 0.3 * coverage + 0.2 * bigram_bonus
            reranked.append((final_score, metadata, doc_tokens, bm25))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return [(meta, dtoks, bm25) for _, meta, dtoks, bm25 in reranked[:top_k]]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, question: str, setting: str = "B") -> list[Passage]:
        config = SETTINGS.get(setting)
        if config is None:
            raise ValueError(f"Unknown retrieval setting: {setting}")

        query = tokens(question)
        if not query:
            return []

        documents = self._get_documents(config.piece_size)

        # Build document-frequency table
        document_frequency: dict[str, int] = {}
        for _, doc_tokens in documents:
            for term in set(doc_tokens):
                document_frequency[term] = document_frequency.get(term, 0) + 1

        total = len(documents)

        # --- First-stage BM25 ---
        fetch_k = config.top_k * 3 if config.rerank else config.top_k
        scored_candidates: list[tuple[dict, list[str], float]] = []

        for metadata, doc_tokens in documents:
            score = self._bm25_score(query, doc_tokens, document_frequency, total)
            if score:
                scored_candidates.append((metadata, doc_tokens, score))

        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        top_candidates = scored_candidates[:fetch_k]

        # --- Second-stage reranking (setting C) ---
        if config.rerank:
            top_candidates = self._rerank(top_candidates, query, config.top_k)

        return [
            Passage(score=score, **metadata)
            for metadata, _doc_tokens, score in top_candidates[: config.top_k]
        ]
