"""Prepare a reproducible LiveKit issue collection without running it by default."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "livekit"
API_URL = "https://api.github.com/repos/livekit/agents/issues"
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
USER_RE = re.compile(r"(?<![\w-])@[A-Za-z0-9-]+")


def sanitize(text: str | None) -> str:
    text = text or ""
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    return USER_RE.sub("[REDACTED_USER]", text)


def fetch_page(page: int, per_page: int = 100) -> list[dict[str, object]]:
    params = f"?state=all&sort=created&direction=asc&per_page={per_page}&page={page}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "EvidenceGate/0.1",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        API_URL + params,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def collect(limit: int) -> None:
    records: list[dict[str, object]] = []
    page = 1
    while len(records) < limit:
        batch = fetch_page(page)
        if not batch:
            break
        records.extend(
            {
                "title": item.get("title", ""),
                "body": sanitize(item.get("body")),
                "labels": [label.get("name", "") for label in item.get("labels", [])],
                "created_at": item.get("created_at"),
                "url": item.get("html_url"),
                "number": item.get("number"),
            }
            for item in batch
            if "pull_request" not in item
        )
        page += 1
    records = records[:limit]
    key = json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "issues.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "source": "GitHub REST API",
        "endpoint": API_URL,
        "collection_utc": datetime.now(timezone.utc).isoformat(),
        "requested_limit": limit,
        "returned_records": len(records),
        "excluded_pull_requests": True,
        "sampling": {
            "method": (
                "first limit non-pull-request issues returned by the GitHub REST API "
                "with state=all, sort=created, direction=asc, and per_page=100"
            ),
            "cohort": "oldest available non-pull-request issues at collection time",
            "limitations": (
                "This is a chronological cohort, not a representative random sample. "
                "GitHub pagination and issue history can change between collection dates."
            ),
        },
        "content_sha256": hashlib.sha256(key).hexdigest(),
        "note": "The default Milestone 0 workflow only prepares this collector; it does not bulk collect.",
    }
    (OUTPUT / "collection_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Actually fetch issues.")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if not args.collect:
        raise SystemExit("Collector prepared. Re-run with --collect only after reviewing the collection plan.")
    collect(args.limit)
