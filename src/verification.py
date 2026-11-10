"""Deterministic citation checks against retrieved evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retrieval import Passage

CITATION_RE = re.compile(r"\[([^,\]]+),\s*§([^,]+),\s*p\.(\d+)\]")
# Detects any bracket group in the answer (used to rule out abstentions that
# happen to contain non-citation bracket text).
BRACKET_RE = re.compile(r"\[.+?\]")

APPROVED = {"Attention Is All You Need", "Moshi", "Dynamic Chunking"}

# Phrases that indicate the LLM is correctly refusing to answer because the
# retrieved evidence is insufficient.
_ABSTENTION_PHRASES = [
    "cannot answer",
    "cannot provide",
    "do not address",
    "does not address",
    "does not establish",
    "not establish",
    "not supported",
    "not in the",
    "insufficient evidence",
    "not addressed",
]

# Token splitter — mirrors the approach used in retrieval.py.
TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")

# Common English stopwords to filter when extracting key terms for the
# provenance check.  This list is intentionally small; the goal is to remove
# high-frequency words that carry no content signal.
_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "are", "was", "were",
    "has", "have", "had", "not", "but", "from", "they", "their", "its",
    "can", "which", "also", "been", "into", "more", "than", "one", "all",
    "any", "our", "each", "both", "who", "she", "him", "his", "her",
    "you", "your", "would", "could", "should", "will", "may", "might",
}


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    citations: list[str]
    reason: str
    abstention: bool = False


def is_abstention(answer: str) -> bool:
    """Return True when *answer* is a legitimate LLM refusal to answer.

    Conditions (both must hold):
    1. The answer contains no bracket pattern ``[...]`` — a bracketed citation
       in an otherwise-refusal-looking answer means it is NOT a pure abstention.
    2. The answer contains at least one of the recognised refusal phrases
       (case-insensitive).
    """
    if BRACKET_RE.search(answer):
        return False
    lower = answer.lower()
    return any(phrase in lower for phrase in _ABSTENTION_PHRASES)


def _extract_key_terms(text: str, max_terms: int = 5) -> list[str]:
    """Extract up to *max_terms* content-bearing lowercase words from *text*.

    Uses TOKEN_RE to split on alphabetic runs of ≥ 3 characters, then strips
    stopwords.  The result is used for the deterministic provenance check
    described in Finding F-06, section B.4.

    NOTE — Limitation: this is a bag-of-words surface overlap check, not
    semantic similarity.  It will pass through paraphrases that share no
    surface tokens with the source passage and will reject legitimate quotes
    that happen to share fewer than 2 tokens with the retrieved chunk boundary.
    Use it only as a low-cost sanity filter against completely fabricated
    claims.
    """
    tokens = [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
        if len(unique) == max_terms:
            break
    return unique


def _sentence_before_match(answer: str, match: re.Match) -> str:  # type: ignore[type-arg]
    """Return the sentence (or partial sentence) immediately before *match*.

    We walk back from the start of the match to find the preceding sentence
    boundary ('.' / '!' / '?') or the start of the string, then take up to
    200 characters of that segment plus the matched text itself.  This window
    is passed to _extract_key_terms so we capture the claim being made.
    """
    start = match.start()
    preceding = answer[:start]
    # Find the last sentence-ending punctuation before the citation.
    for sep in (".", "!", "?"):
        idx = preceding.rfind(sep)
        if idx != -1:
            preceding = preceding[idx + 1 :]
            break
    return (preceding + match.group(0))[-200:]


def verify_answer(answer: str, passages: list[Passage]) -> VerificationResult:
    """Verify *answer* against *passages* (retrieved evidence).

    Three answer categories are handled:

    A. **Abstentions** — no citations, recognised refusal phrase → valid=True,
       abstention=True.
    B. **Substantive answers with citations** — each citation is validated:
       paper approval, page presence, section match (3-tier fallback), and
       content-grounding (key-term overlap).
    C. **Partial answers** — cited claims that pass all checks make the whole
       answer valid even if other sentences are uncited.
    """
    # ── A. Abstention detection ───────────────────────────────────────────────
    if is_abstention(answer):
        return VerificationResult(
            valid=True,
            citations=[],
            reason="Legitimate abstention — no claim made",
            abstention=True,
        )

    # ── B. Citation-bearing answers ───────────────────────────────────────────
    matches = list(CITATION_RE.finditer(answer))
    if not matches:
        return VerificationResult(False, [], "No citations found")

    citations: list[str] = []
    for match in matches:
        paper, section, page = match.group(1), match.group(2), match.group(3)

        # 1. Approved-paper check (unchanged).
        if paper not in APPROVED:
            return VerificationResult(False, citations, f"Unapproved paper: {paper}")

        page_number = int(page)

        # 2. Page-number check (unchanged).
        candidates = [p for p in passages if p.paper == paper and p.page == page_number]
        if not candidates:
            return VerificationResult(False, citations, "Citation is not in retrieved evidence")

        # 3. Section matching — 3-tier fallback.
        section_exact = section in {c.section for c in candidates}
        section_substr = not section_exact and any(
            section.lower() in p.section.lower() for p in candidates
        )
        section_in_text = not section_exact and not section_substr and any(
            section.lower() in p.text.lower() for p in candidates
        )
        section_unknown = section == "Unknown" or any(c.section == "Unknown" for c in candidates)

        if not (section_exact or section_substr or section_in_text or section_unknown):
            return VerificationResult(
                False, citations, "Citation section does not match evidence"
            )

        # 4. Content-grounding (provenance) check — deterministic, bag-of-words.
        # NOTE — Limitation: surface token overlap only; paraphrases or
        # vocabulary mismatches may produce false negatives.  This is by design:
        # the check catches entirely fabricated citations, not partial rewording.
        claim_window = _sentence_before_match(answer, match)
        key_terms = _extract_key_terms(claim_window)
        if key_terms:
            grounded = False
            for candidate in candidates:
                passage_lower = candidate.text.lower()
                hits = sum(1 for t in key_terms if t in passage_lower)
                if hits >= min(2, len(key_terms)):
                    grounded = True
                    break
            if not grounded:
                return VerificationResult(
                    False,
                    citations,
                    "Claim not grounded in cited evidence",
                )

        citations.append(f"[{paper}, §{section}, p.{page}]")

    # ── C. Partial answers ────────────────────────────────────────────────────
    # All cited claims passed — the answer is valid even if some sentences are
    # uncited (the LLM is instructed to hedge uncited portions with abstention
    # language, which is acceptable per Finding F-06, section C).
    return VerificationResult(True, citations, "Citations resolve to retrieved evidence")
