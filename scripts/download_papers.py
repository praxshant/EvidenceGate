"""Download and extract the exact paper versions required by The Relay."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import fitz


ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "data" / "papers"
EXTRACTED_DIR = ROOT / "data" / "processed" / "papers"

PAPERS = [
    {
        "short_name": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "version": 7,
        "pdf_url": "https://arxiv.org/pdf/1706.03762v7",
        "filename": "attention_is_all_you_need_v7.pdf",
    },
    {
        "short_name": "Moshi",
        "arxiv_id": "2410.00037",
        "version": 1,
        "pdf_url": "https://arxiv.org/pdf/2410.00037v1",
        "filename": "moshi_v1.pdf",
    },
    {
        "short_name": "Dynamic Chunking",
        "arxiv_id": "2507.07955",
        "version": 2,
        "pdf_url": "https://arxiv.org/pdf/2507.07955v2",
        "filename": "dynamic_chunking_v2.pdf",
    },
]


def download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "EvidenceGate/0.1"})
    with urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def verify_arxiv_version(paper: dict[str, object]) -> bool:
    arxiv_id = str(paper["arxiv_id"])
    version = int(paper["version"])
    request = Request(
        f"https://arxiv.org/abs/{arxiv_id}v{version}",
        headers={"User-Agent": "EvidenceGate/0.1"},
    )
    with urlopen(request, timeout=30) as response:
        page = response.read().decode("utf-8", errors="replace")
    return f"{arxiv_id}v{version}" in page and str(paper["short_name"]).split()[0] in page


def extract_pdf(pdf_path: Path, output_path: Path) -> dict[str, object]:
    document = fitz.open(pdf_path)
    pages: list[dict[str, object]] = []
    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").replace("\r\n", "\n").replace("\r", "\n")
        pages.append(
            {
                "page": page_number,
                "text": text,
                "characters": len(text),
            }
        )
    payload = {
        "pdf": pdf_path.name,
        "page_count": len(document),
        "pages": pages,
        "extraction": "PyMuPDF page text extraction",
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "page_count": len(document),
        "nonempty_pages": sum(bool(page["text"].strip()) for page in pages),
        "characters": sum(int(page["characters"]) for page in pages),
    }


def main() -> int:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, object]] = []

    for paper in PAPERS:
        pdf_path = PAPERS_DIR / str(paper["filename"])
        extracted_path = EXTRACTED_DIR / f"{pdf_path.stem}.json"
        if not pdf_path.exists():
            print(f"Downloading {paper['pdf_url']}")
            download(str(paper["pdf_url"]), pdf_path)
        if pdf_path.stat().st_size < 10_000:
            raise RuntimeError(f"Downloaded PDF is unexpectedly small: {pdf_path}")
        version_verified = verify_arxiv_version(paper)
        if not version_verified:
            raise RuntimeError(f"Could not verify arXiv version for {paper['arxiv_id']}")
        stats = extract_pdf(pdf_path, extracted_path)
        metadata.append(
            {
                **paper,
                "local_pdf": str(pdf_path.relative_to(ROOT)),
                "version_verified": version_verified,
                **stats,
            }
        )

    (PAPERS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "source": "arXiv versioned PDF URLs",
                "papers": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
