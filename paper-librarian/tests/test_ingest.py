from datetime import datetime, timezone

import arxiv
import pymupdf
import pytest

from librarian import ingest as ingest_mod
from librarian.ingest import (
    Segment,
    chunk_segments,
    extract_pdf,
    heading_name,
    ingest,
    parse_arxiv_id,
    split_sections,
)
from librarian.store import Store

DIM = 4


class FakeEmbedder:
    dim = DIM

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "library.db", embedding_dim=DIM) as s:
        yield s


def make_pdf(path, pages, title=None, author=None):
    """pages: list of lists of (text, fontsize) lines."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 72
        for text, size in lines:
            page.insert_text((72, y), text, fontsize=size)
            y += size + 8
    meta = {}
    if title:
        meta["title"] = title
    if author:
        meta["author"] = author
    if meta:
        doc.set_metadata(meta)
    doc.save(path)
    doc.close()
    return path


def body(n_words, word="token"):
    words = [f"{word}{i}" for i in range(n_words)]
    return [(" ".join(words[i : i + 12]), 10) for i in range(0, n_words, 12)]


@pytest.fixture
def paper_pdf(tmp_path):
    return make_pdf(
        tmp_path / "paper.pdf",
        [
            [("A Study of Widgets", 20), ("Jane Doe", 11), ("Abstract", 12), ("We study widgets in depth.", 10),
             ("1 Introduction", 12), *body(30, "intro")],
            [("2 Widget Architecture", 12), *body(40, "arch"), ("References", 12),
             ("[1] A. Smith. Gadgets. 2019.", 10), ("[2] B. Lee. Gizmos. 2020.", 10)],
        ],
    )


@pytest.mark.parametrize(
    "source, expected",
    [
        ("1706.03762", "1706.03762"),
        ("1706.03762v5", "1706.03762"),
        ("arXiv:2310.06825", "2310.06825"),
        ("https://arxiv.org/abs/2310.06825v2", "2310.06825"),
        ("https://arxiv.org/pdf/2310.06825.pdf", "2310.06825"),
        ("hep-th/9901001", "hep-th/9901001"),
        ("papers/attention.pdf", None),
        ("C:\\papers\\1706.03762.pdf", None),
    ],
)
def test_parse_arxiv_id(source, expected):
    assert parse_arxiv_id(source) == expected


@pytest.mark.parametrize(
    "line, expected",
    [
        ("Abstract", "Abstract"),
        ("1 Introduction", "Introduction"),
        ("3.2 Methods", "Method"),
        ("IV. EXPERIMENTS", "Experiments"),
        ("References", "References"),
        ("Appendix B: Proofs", "Appendix"),
        ("3 Model Architecture", "Model Architecture"),
        ("We study widgets.", None),
        ("Model", None),
        ("2     Google Research      University of Toronto", None),
        ("2 The results show that this approach is clearly better than all baselines", None),
        ("2017", None),
    ],
)
def test_heading_name(line, expected):
    assert heading_name(line) == expected


def test_split_sections_tracks_pages():
    segs = split_sections([(1, "Abstract\nshort text\n1 Introduction\nintro text"), (2, "more intro\nReferences\n[1] x")])
    assert [(s.section, s.page) for s in segs] == [
        ("Abstract", 1), ("Introduction", 1), ("Introduction", 2), ("References", 2),
    ]


def test_chunking_overlap_and_section_boundaries():
    segs = [Segment("Method", 3, " ".join(f"w{i}" for i in range(25))), Segment("Results", 4, "r1 r2")]
    chunks = chunk_segments(segs, chunk_words=10, overlap_words=2)
    method = [c for c in chunks if c.section == "Method"]
    assert [len(c.text.split()) for c in method] == [10, 10, 9]
    assert method[1].text.split()[:2] == method[0].text.split()[-2:]
    assert chunks[-1].section == "Results" and chunks[-1].page == 4
    assert [c.seq for c in chunks] == list(range(len(chunks)))


def test_chunking_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_segments([], chunk_words=10, overlap_words=10)


def test_extract_pdf_falls_back_to_largest_text_for_title(paper_pdf):
    extracted = extract_pdf(paper_pdf)
    assert extracted.title == "A Study of Widgets"
    assert extracted.abstract == "We study widgets in depth."
    sections = {s.section for s in extracted.segments}
    assert {"Abstract", "Introduction", "Widget Architecture"} <= sections
    assert "References" not in sections
    assert "Gadgets" in extracted.references_text


def test_title_guess_ignores_rotated_text(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((20, 700), "arXiv:2105.00001v1 [cs.LG] 1 May 2021", fontsize=30, rotate=90)
    page.insert_text((72, 72), "Real Paper Title", fontsize=18)
    page.insert_text((72, 120), " ".join(f"w{i}" for i in range(20)), fontsize=10)
    doc.save(tmp_path / "stamp.pdf")
    doc.close()
    assert extract_pdf(tmp_path / "stamp.pdf").title == "Real Paper Title"


def test_extract_pdf_prefers_metadata(tmp_path):
    path = make_pdf(tmp_path / "m.pdf", [[("Big Heading", 24), *body(10)]], title="Real Title", author="A. One, B. Two")
    extracted = extract_pdf(path)
    assert extracted.title == "Real Title"
    assert extracted.authors == ["A. One", "B. Two"]


def test_ingest_local_pdf(store, paper_pdf):
    embedder = FakeEmbedder()
    messages = []
    result = ingest(str(paper_pdf), store, embedder, progress=messages.append)

    assert not result.duplicate
    assert result.paper.id.startswith("pdf:") and result.paper.source == "pdf"
    assert result.chunk_count > 0
    assert store.get_paper(result.paper.id).title == "A Study of Widgets"
    assert len(store.get_chunks(result.paper.id)) == result.chunk_count
    assert store.db.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0] == result.chunk_count
    assert any("Embedding" in m for m in messages)

    again = ingest(str(paper_pdf), store, embedder)
    assert again.duplicate and again.paper.id == result.paper.id
    assert embedder.calls == 1


def test_ingest_missing_file(store):
    with pytest.raises(FileNotFoundError):
        ingest("nope/missing.pdf", store, FakeEmbedder())


def test_ingest_failure_leaves_no_partial_paper(store, paper_pdf):
    class BadEmbedder(FakeEmbedder):
        def embed(self, texts):
            return [[1.0, 2.0]] * len(texts)  # wrong dimension

    with pytest.raises(ValueError):
        ingest(str(paper_pdf), store, BadEmbedder())
    assert store.list_papers() == []
    assert store.db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


def test_ingest_arxiv(store, paper_pdf, monkeypatch, tmp_path):
    fetched = []

    def fake_fetch(arxiv_id, pdf_dir):
        fetched.append(arxiv_id)
        result = arxiv.Result(
            entry_id=f"http://arxiv.org/abs/{arxiv_id}v1",
            title="Widgets:\n  A Study",
            summary="An  abstract\nfrom arXiv.",
            published=datetime(2021, 5, 1, tzinfo=timezone.utc),
            authors=[arxiv.Result.Author("Jane Doe")],
        )
        return result, paper_pdf

    monkeypatch.setattr(ingest_mod, "fetch_arxiv", fake_fetch)
    result = ingest("https://arxiv.org/abs/2105.00001v3", store, FakeEmbedder(), pdf_dir=tmp_path)

    paper = store.get_paper("arxiv:2105.00001")
    assert paper.title == "Widgets: A Study"
    assert paper.abstract == "An abstract from arXiv."
    assert paper.authors == ["Jane Doe"] and paper.year == 2021
    assert result.chunk_count > 0

    assert ingest("2105.00001", store, FakeEmbedder()).duplicate
    assert fetched == ["2105.00001"]  # duplicate check happens before any network call
