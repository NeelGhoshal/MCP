"""Reference extraction and matching."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from librarian.store import Citation, Paper, Store

INDENT = "    "  # ingest.page_text() prefixes indented lines with this; defined here so both modules share it
BRACKET_MARKER = re.compile(r"^\s*\[(\d{1,3})\]\s*")
DOTTED_MARKER = re.compile(r"^\s*(\d{1,3})\.\s+(?=\S)")
MIN_BRACKET_MARKERS = 2  # "[n]" at a line start is distinctive
MIN_DOTTED_MARKERS = 3  # "n." also starts ordinary numbered lines
MIN_REFERENCE_CHARS = 15

ARXIV_IN_REFERENCE = re.compile(
    r"(?:arxiv(?:\s+preprint)?\s*:?\s*(?:arxiv\s*:\s*)?|abs/|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})",
    re.IGNORECASE,
)
MIN_TITLE_CHARS = 12  # shorter titles ("BERT") would match too many unrelated references
FUZZY_THRESHOLD = 0.92  # share of title characters found, in order, in the reference
WORD = re.compile(r"[a-z0-9]+")


# --- splitting ---


def split_references(raw: str) -> list[str]:
    """Split a References section into one string per entry.

    Tries, in order: "[1]" markers, "1." markers, then hanging indents (author-year styles, where an
    entry's first line starts at the column edge and its continuation lines are indented).
    """
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if (entries := _split_on_marker(lines, BRACKET_MARKER, MIN_BRACKET_MARKERS)) is not None:
        marker = BRACKET_MARKER
    elif (entries := _split_on_marker(lines, DOTTED_MARKER, MIN_DOTTED_MARKERS, sequential=True)) is not None:
        marker = DOTTED_MARKER
    else:
        entries, marker = _split_on_hanging_indent(lines), None
    cleaned = [_clean_reference(e, marker) for e in entries]
    return [e for e in cleaned if len(e) >= MIN_REFERENCE_CHARS]


def _split_on_marker(
    lines: list[str], marker: re.Pattern, min_markers: int, sequential: bool = False
) -> list[str] | None:
    starts = [i for i, ln in enumerate(lines) if marker.match(ln)]
    if len(starts) < min_markers:
        return None
    if sequential:
        # "1." also starts ordinary lines ("3. Results are..."), so require numbering that mostly counts up.
        numbers = [int(marker.match(lines[i]).group(1)) for i in starts]
        increasing = sum(1 for a, b in zip(numbers, numbers[1:]) if b == a + 1)
        if numbers[0] > 2 or increasing < 0.8 * (len(numbers) - 1):
            return None
    bounds = starts + [len(lines)]
    return ["\n".join(lines[a:b]) for a, b in zip(bounds, bounds[1:])]


def _split_on_hanging_indent(lines: list[str]) -> list[str]:
    entries: list[list[str]] = []
    for ln in lines:
        text = ln.strip()
        starts_entry = not ln.startswith(INDENT) and not text[:1].islower()
        if starts_entry or not entries:
            entries.append([text])
        else:
            entries[-1].append(text)
    return ["\n".join(e) for e in entries]


def _clean_reference(entry: str, marker: re.Pattern | None) -> str:
    text = marker.sub("", entry, count=1) if marker else entry
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip()


# --- matching ---


def compact(text: str) -> str:
    """Lowercase alphanumerics only, so "Pre-\\ntraining" and "Pretraining" compare equal."""
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", text).lower())


def reference_arxiv_id(reference: str) -> str | None:
    m = ARXIV_IN_REFERENCE.search(reference)
    return m.group(1) if m else None


class Matcher:
    """Matches reference strings to papers by arXiv id, then exact title, then fuzzy title."""

    def __init__(self, papers: list[Paper]):
        self.by_arxiv = {p.arxiv_id: p.id for p in papers if p.arxiv_id}
        self.titles = [
            (p.id, compact(p.title), {w for w in WORD.findall(p.title.lower()) if len(w) > 3})
            for p in papers
            if len(compact(p.title)) >= MIN_TITLE_CHARS
        ]

    def match(self, reference: str) -> str | None:
        if (arxiv_id := reference_arxiv_id(reference)) and arxiv_id in self.by_arxiv:
            return self.by_arxiv[arxiv_id]
        ref = compact(reference)
        ref_words = set(WORD.findall(reference.lower()))
        best: tuple[int, str] | None = None
        for paper_id, title, title_words in self.titles:
            if title in ref or self._fuzzy(title, title_words, ref, ref_words):
                # When several titles match, the longest one is the most specific.
                if best is None or len(title) > best[0]:
                    best = (len(title), paper_id)
        return best[1] if best else None

    @staticmethod
    def _fuzzy(title: str, title_words: set[str], ref: str, ref_words: set[str]) -> bool:
        # Cheap pre-filter: most of the title's longer words should appear in the reference.
        if title_words and len(title_words & ref_words) < 0.6 * len(title_words):
            return False
        blocks = SequenceMatcher(None, title, ref, autojunk=False).get_matching_blocks()
        found = sum(b.size for b in blocks if b.size >= 3)
        return found >= FUZZY_THRESHOLD * len(title)


# --- linking ---


@dataclass
class LinkResult:
    references: int  # entries parsed from this paper's References section
    linked: int  # of those, how many point at a paper in the library
    cited_by: int  # existing library references newly linked to this paper


def link_paper(store: Store, paper: Paper, references_text: str, commit: bool = True) -> LinkResult:
    """Store `paper`'s references, matched against the library, and link older references to it.

    Replaces any citations `paper` already had, so it can be re-run after improving extraction.
    """
    references = split_references(references_text)
    others = Matcher([p for p in store.list_papers() if p.id != paper.id])
    citations = [Citation(seq=i, raw_reference=r, matched_paper_id=others.match(r)) for i, r in enumerate(references)]

    this = Matcher([paper])
    backlinks = [c for c in store.unmatched_citations(exclude_paper_id=paper.id) if this.match(c.raw_reference)]

    with store.transaction(commit):
        store.delete_citations(paper.id, commit=False)
        store.add_citations(paper.id, citations, commit=False)
        for c in backlinks:
            store.set_citation_match(c.id, paper.id, commit=False)

    return LinkResult(
        references=len(citations),
        linked=sum(1 for c in citations if c.matched_paper_id),
        cited_by=len(backlinks),
    )
