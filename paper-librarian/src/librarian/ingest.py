"""PDF / arXiv fetching and text extraction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import arxiv
import pymupdf
import requests

from librarian.embed import Embedder
from librarian.store import Chunk, Paper, Store, library_home

Progress = Callable[[str], None]

# New-style (2310.06825v2) and old-style (hep-th/9901001) arXiv identifiers.
ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5}|[a-z][a-z.\-]*/\d{7})(v\d+)?", re.IGNORECASE)
ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:\.pdf)?/?$", re.IGNORECASE)

# Known section headings, mapped to canonical names.
CANONICAL_SECTIONS = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related work": "Related Work",
    "related works": "Related Work",
    "background": "Background",
    "preliminaries": "Background",
    "method": "Method",
    "methods": "Method",
    "methodology": "Method",
    "approach": "Method",
    "our approach": "Method",
    "proposed method": "Method",
    "experiments": "Experiments",
    "experimental setup": "Experiments",
    "experimental results": "Results",
    "evaluation": "Experiments",
    "results": "Results",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "conclusion and future work": "Conclusion",
    "acknowledgments": "Acknowledgments",
    "acknowledgements": "Acknowledgments",
    "references": "References",
    "bibliography": "References",
    "appendix": "Appendix",
}
# "3", "3.1", "3.", "IV.", "A." style numbering in front of a heading.
HEADING_NUMBER = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.|[A-H]\.?)\s+")
MAX_HEADING_WORDS = 8

# all-MiniLM-L6-v2 truncates at 256 word pieces, roughly 180-200 English words.
DEFAULT_CHUNK_WORDS = 180
DEFAULT_OVERLAP_WORDS = 30


@dataclass
class Segment:
    section: str
    page: int
    text: str


@dataclass
class ExtractedPdf:
    title: str
    authors: list[str]
    year: int | None
    segments: list[Segment]
    references_text: str = ""

    @property
    def abstract(self) -> str:
        return " ".join(s.text for s in self.segments if s.section == "Abstract").strip()


@dataclass
class IngestResult:
    paper: Paper
    duplicate: bool
    chunk_count: int = 0
    references_text: str = field(default="", repr=False)


def parse_arxiv_id(source: str) -> str | None:
    """Return the version-less arXiv id if `source` is an arXiv id or URL, else None."""
    source = source.strip()
    if m := ARXIV_URL.search(source):
        source = m.group(1)
    source = re.sub(r"^arxiv:", "", source, flags=re.IGNORECASE)
    m = ARXIV_ID.fullmatch(source)
    return m.group(1) if m else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- arXiv ---


def fetch_arxiv(arxiv_id: str, pdf_dir: Path) -> tuple[arxiv.Result, Path]:
    results = list(arxiv.Client(num_retries=3).results(arxiv.Search(id_list=[arxiv_id], max_results=1)))
    if not results:
        raise ValueError(f"arXiv paper not found: {arxiv_id}")
    result = results[0]
    if not result.pdf_url:
        raise ValueError(f"arXiv paper {arxiv_id} has no PDF link")

    pdf_dir.mkdir(parents=True, exist_ok=True)
    path = pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"
    if not path.exists():
        resp = requests.get(result.pdf_url, timeout=60)
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            raise ValueError(f"arXiv returned a non-PDF response for {arxiv_id}")
        path.write_bytes(resp.content)
    return result, path


# --- PDF extraction ---


def extract_pdf(path: Path) -> ExtractedPdf:
    with pymupdf.open(path) as doc:
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip() or _largest_text_on_first_page(doc) or path.stem
        authors = [a.strip() for a in re.split(r"[;,]| and ", meta.get("author") or "") if a.strip()]
        year = _year_from_pdf_date(meta.get("creationDate") or "")
        pages = [(i + 1, page.get_text(sort=True)) for i, page in enumerate(doc)]

    segments = split_sections(pages)
    references = " ".join(s.text for s in segments if s.section == "References")
    body = [s for s in segments if s.section not in ("References",)]
    return ExtractedPdf(title=title, authors=authors, year=year, segments=body, references_text=references)


def _largest_text_on_first_page(doc: pymupdf.Document) -> str:
    if doc.page_count == 0:
        return ""
    spans = [
        span
        for block in doc[0].get_text("dict")["blocks"]
        for line in block.get("lines", [])
        if tuple(line["dir"]) == (1.0, 0.0)  # skip rotated text such as arXiv's vertical margin stamp
        for span in line["spans"]
        if span["text"].strip()
    ]
    if not spans:
        return ""
    biggest = max(s["size"] for s in spans)
    return " ".join(s["text"].strip() for s in spans if abs(s["size"] - biggest) < 0.5)[:300]


def _year_from_pdf_date(value: str) -> int | None:
    # PDF dates look like "D:20170612..."; ignore anything implausible.
    m = re.match(r"(?:D:)?(\d{4})", value)
    return int(m.group(1)) if m and 1900 <= int(m.group(1)) <= 2100 else None


def heading_name(line: str) -> str | None:
    """Return a canonical section name if `line` looks like a section heading."""
    text = line.strip().rstrip(":").strip()
    if not text or len(text.split()) > MAX_HEADING_WORDS:
        return None
    numbered = HEADING_NUMBER.match(text)
    bare = HEADING_NUMBER.sub("", text).strip()
    if re.search(r"\s{3,}", bare):  # wide gaps mean merged columns (affiliations, table rows), not a heading
        return None
    key = re.sub(r"\s+", " ", bare).lower()
    if key in CANONICAL_SECTIONS:
        return CANONICAL_SECTIONS[key]
    if key.startswith("appendix"):
        return "Appendix"
    # Numbered, Title Case headings ("3 Model Architecture") keep their own name.
    if numbered and bare[:1].isupper() and not bare.endswith(".") and re.search(r"[A-Za-z]{3}", bare):
        return bare
    return None


def split_sections(pages: list[tuple[int, str]]) -> list[Segment]:
    segments: list[Segment] = []
    section, page_no, buf = "Front Matter", 1, []

    def flush():
        text = _clean(" ".join(buf))
        if text:
            segments.append(Segment(section=section, page=page_no, text=text))

    for number, text in pages:
        if buf:  # a section continuing onto a new page starts a new segment, so pages stay accurate
            flush()
            buf = []
        page_no = number
        for line in text.splitlines():
            name = heading_name(line)
            if name:
                flush()
                section, page_no, buf = name, number, []
            else:
                buf.append(line)
    flush()
    return segments


def _clean(text: str) -> str:
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    return re.sub(r"\s+", " ", text).strip()


def chunk_segments(
    segments: list[Segment], chunk_words: int = DEFAULT_CHUNK_WORDS, overlap_words: int = DEFAULT_OVERLAP_WORDS
) -> list[Chunk]:
    """Split segments into overlapping word windows. Chunks never span two sections."""
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must be in [0, chunk_words)")
    chunks: list[Chunk] = []
    step = chunk_words - overlap_words
    for seg in segments:
        words = seg.text.split()
        for start in range(0, len(words), step):
            window = words[start : start + chunk_words]
            # Skip a trailing window that is entirely overlap with the previous one.
            if start and len(window) <= overlap_words:
                break
            chunks.append(Chunk(seq=len(chunks), section=seg.section, page=seg.page, text=" ".join(window)))
    return chunks


# --- pipeline ---


def ingest(
    source: str,
    store: Store,
    embedder: Embedder,
    pdf_dir: Path | None = None,
    progress: Progress | None = None,
) -> IngestResult:
    """Add a paper from an arXiv id/URL or a local PDF path. Returns the existing paper if already present."""
    say = progress or (lambda _msg: None)
    pdf_dir = pdf_dir or library_home() / "pdfs"

    if arxiv_id := parse_arxiv_id(source):
        if existing := store.find_duplicate(arxiv_id=arxiv_id):
            return IngestResult(paper=existing, duplicate=True)
        say(f"Fetching arXiv:{arxiv_id}")
        result, path = fetch_arxiv(arxiv_id, pdf_dir)
        sha = sha256_file(path)
        if existing := store.find_duplicate(sha256=sha):
            return IngestResult(paper=existing, duplicate=True)
        say("Extracting text")
        extracted = extract_pdf(path)
        paper = Paper(
            id=f"arxiv:{arxiv_id}",
            title=re.sub(r"\s+", " ", result.title).strip(),
            source="arxiv",
            authors=[a.name for a in result.authors],
            year=result.published.year if result.published.year > 1 else None,
            abstract=re.sub(r"\s+", " ", result.summary).strip(),
            arxiv_id=arxiv_id,
            sha256=sha,
            path=str(path),
        )
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Not an arXiv id and no such file: {source}")
        sha = sha256_file(path)
        if existing := store.find_duplicate(sha256=sha):
            return IngestResult(paper=existing, duplicate=True)
        say(f"Extracting text from {path.name}")
        extracted = extract_pdf(path)
        paper = Paper(
            id=f"pdf:{sha[:16]}",
            title=extracted.title,
            source="pdf",
            authors=extracted.authors,
            year=extracted.year,
            abstract=extracted.abstract,
            sha256=sha,
            path=str(path),
        )

    chunks = chunk_segments(extracted.segments)
    if not chunks:
        raise ValueError(f"No text could be extracted from {path} (scanned PDF?)")
    say(f"Embedding {len(chunks)} chunks")
    embeddings = embedder.embed([c.text for c in chunks])

    # Paper and chunks go in one transaction so a failure never leaves a half-ingested paper.
    with store.db:
        store.add_paper(paper, commit=False)
        store.add_chunks(paper.id, chunks, embeddings, commit=False)
    say(f"Added {paper.title!r} ({len(chunks)} chunks)")
    return IngestResult(paper=paper, duplicate=False, chunk_count=len(chunks), references_text=extracted.references_text)
