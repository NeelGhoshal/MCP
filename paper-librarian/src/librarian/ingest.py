"""PDF / arXiv fetching and text extraction."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import arxiv
import pymupdf
import requests

from librarian.citations import INDENT, LinkResult, link_paper
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
# Names that are headings even when unnumbered and in plain type.
DISTINCTIVE_SECTIONS = {"Abstract", "References", "Acknowledgments"}
# "3", "3.1", "3.", "IV.", "A." style numbering in front of a heading. One or two digits per level,
# so years ("2018.") and decimals ("93.1") are not mistaken for section numbers.
HEADING_NUMBER = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVX]+\.|[A-H]\.?)\s+")
# page_text() prefixes lines set in bold or larger-than-body type with this marker.
EMPHASIS = "\x01"
BOLD_FONT = re.compile(r"bold|cmbx|medi|black|heavy|demi", re.IGNORECASE)
MAX_HEADING_WORDS = 8

# all-MiniLM-L6-v2 truncates at 256 word pieces, roughly 180-200 English words.
DEFAULT_CHUNK_WORDS = 180
DEFAULT_OVERLAP_WORDS = 30


@dataclass
class Segment:
    section: str
    page: int
    text: str
    raw: str = field(default="", repr=False)  # original lines; indented lines start with INDENT


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
    citations: LinkResult | None = None


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
        pages = [(i + 1, page_text(page)) for i, page in enumerate(doc)]

    segments = split_sections(pages)
    references = "\n".join(s.raw for s in segments if s.section == "References")
    body = [s for s in segments if s.section != "References"]
    return ExtractedPdf(title=title, authors=authors, year=year, segments=body, references_text=references)


COLUMN_GAP = 20  # points either side of the page centre a column block may cross
INDENT_MIN = 4  # points past the column's left edge that count as an indented line


def page_text(page: pymupdf.Page) -> str:
    """Page text in reading order, handling two-column layouts.

    Blocks are banded by full-width blocks (titles, wide figures); within a band the left column is
    read before the right. Lines indented past their column's left edge are prefixed with INDENT so
    hanging-indent reference lists can be split later.
    """
    mid = page.rect.width / 2
    blocks = []
    for block in page.get_text("dict")["blocks"]:
        lines = [ln for ln in block.get("lines", []) if tuple(ln["dir"]) == (1.0, 0.0)]
        if not lines:
            continue
        x0, y0, x1, _ = block["bbox"]
        if x0 < mid - COLUMN_GAP and x1 > mid + COLUMN_GAP:
            column = "full"
        else:
            column = "left" if x0 < mid else "right"
        blocks.append((column, y0, x0, lines))

    two_column = sum(1 for b in blocks if b[0] == "right") >= 3
    if two_column:
        ordered, band = [], []
        for b in sorted(blocks, key=lambda b: (b[1], b[2])):
            if b[0] == "full":
                ordered += sorted(band, key=lambda b: (b[0] != "left", b[1])) + [b]
                band = []
            else:
                band.append(b)
        ordered += sorted(band, key=lambda b: (b[0] != "left", b[1]))
    else:
        ordered = sorted(blocks, key=lambda b: (b[1], b[2]))

    left_edge = {}
    for column, _, x0, _ in ordered:
        left_edge[column] = min(left_edge.get(column, x0), x0)
    body_size = _body_font_size(blocks)

    out: list[str] = []
    for column, _, _, lines in ordered:
        for ln in _merge_same_baseline(lines):
            text = _line_text(ln["spans"])
            if text:
                indented = ln["bbox"][0] - left_edge[column] > INDENT_MIN
                emphasized = _is_emphasized(ln["spans"], body_size)
                out.append((EMPHASIS if emphasized else "") + (INDENT if indented else "") + text)
        out.append("")  # blank line between blocks
    return "\n".join(out)


def _merge_same_baseline(lines: list[dict]) -> list[dict]:
    """Some PDFs store one visual line as several ("3" and "Model Architecture"); rejoin them."""
    merged: list[dict] = []
    for ln in lines:
        prev = merged[-1] if merged else None
        if prev and abs(ln["bbox"][1] - prev["bbox"][1]) < 2 and abs(ln["bbox"][3] - prev["bbox"][3]) < 2:
            spans = sorted(prev["spans"] + ln["spans"], key=lambda s: s["bbox"][0])
            x0 = min(prev["bbox"][0], ln["bbox"][0])
            merged[-1] = {**prev, "spans": spans, "bbox": (x0, *prev["bbox"][1:])}
        else:
            merged.append(ln)
    return merged


def _line_text(spans: list[dict]) -> str:
    """Join spans, inserting a space where the layout leaves a gap instead of a space character."""
    parts, prev_x1 = [], None
    for span in spans:
        if prev_x1 is not None and span["bbox"][0] - prev_x1 > 1.0:
            parts.append(" ")
        parts.append(span["text"])
        prev_x1 = span["bbox"][2]
    return unicodedata.normalize("NFKC", "".join(parts)).strip()  # NFKC also expands ligatures


def _body_font_size(blocks: list) -> float:
    sizes: Counter[float] = Counter()
    for *_, lines in blocks:
        for ln in lines:
            for span in ln["spans"]:
                sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _is_emphasized(spans: list[dict], body_size: float) -> bool:
    visible = [s for s in spans if s["text"].strip()]
    if not visible:
        return False
    bold = all(s["flags"] & 16 or BOLD_FONT.search(s["font"]) for s in visible)
    larger = body_size > 0 and min(s["size"] for s in visible) >= body_size * 1.1
    return bold or larger


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
    title = " ".join(s["text"].strip() for s in spans if abs(s["size"] - biggest) < 0.5)
    return unicodedata.normalize("NFKC", title)[:300]


def _year_from_pdf_date(value: str) -> int | None:
    # PDF dates look like "D:20170612..."; ignore anything implausible.
    m = re.match(r"(?:D:)?(\d{4})", value)
    return int(m.group(1)) if m and 1900 <= int(m.group(1)) <= 2100 else None


def heading_name(line: str) -> str | None:
    """Return a section name if `line` looks like a section heading.

    Known names ("Introduction", "3 Methods") are accepted as plain text. Other numbered headings
    ("3 Model Architecture") must also be emphasized (bold or large type), which numbered body lines
    such as list items or reference continuations are not.
    """
    emphasized = line.startswith(EMPHASIS)
    text = line.lstrip(EMPHASIS).strip().rstrip(":").strip()
    if not text or len(text.split()) > MAX_HEADING_WORDS:
        return None
    numbered = HEADING_NUMBER.match(text)
    bare = HEADING_NUMBER.sub("", text).strip()
    if re.search(r"\s{3,}", bare):  # wide gaps mean merged columns (affiliations, table rows), not a heading
        return None
    key = re.sub(r"\s+", " ", bare).lower()
    if key in CANONICAL_SECTIONS:
        name = CANONICAL_SECTIONS[key]
        # An unnumbered plain-text "Method" or "Results" is usually a table header, not a heading.
        if numbered or emphasized or name in DISTINCTIVE_SECTIONS:
            return name
        return None
    if key.startswith("appendix"):
        return "Appendix"
    # Numbered, Title Case headings ("3 Model Architecture") keep their own name.
    if (
        numbered
        and emphasized
        and bare[:1].isupper()
        and not bare.endswith((".", ",", "-"))
        and ". " not in bare
        and re.search(r"[A-Za-z]{3}", bare)
    ):
        return bare
    return None


def split_sections(pages: list[tuple[int, str]]) -> list[Segment]:
    segments: list[Segment] = []
    section, page_no, buf = "Front Matter", 1, []

    def flush():
        text = _clean(" ".join(buf))
        if text:
            segments.append(Segment(section=section, page=page_no, text=text, raw="\n".join(buf)))

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
                buf.append(line.lstrip(EMPHASIS))
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

    say("Linking citations")
    # Paper, chunks and citations go in one transaction so a failure never leaves a half-ingested paper.
    with store.transaction():
        store.add_paper(paper, commit=False)
        store.add_chunks(paper.id, chunks, embeddings, commit=False)
        links = link_paper(store, paper, extracted.references_text, commit=False)
    say(
        f"Added {paper.title!r}: {len(chunks)} chunks, {links.references} references "
        f"({links.linked} in library), cited by {links.cited_by} library references"
    )
    return IngestResult(paper=paper, duplicate=False, chunk_count=len(chunks), citations=links)


def rebuild_citations(store: Store, progress: Progress | None = None) -> dict[str, LinkResult]:
    """Re-extract and re-link references for every paper, e.g. after improving reference parsing."""
    say = progress or (lambda _msg: None)
    results: dict[str, LinkResult] = {}
    with store.transaction():
        for paper in store.list_papers():
            if not paper.path or not Path(paper.path).is_file():
                say(f"Skipping {paper.id}: PDF not found at {paper.path}")
                continue
            results[paper.id] = link_paper(store, paper, extract_pdf(Path(paper.path)).references_text, commit=False)
            say(f"{paper.id}: {results[paper.id].references} references, {results[paper.id].linked} in library")
    return results
