import sqlite3

import pytest

from librarian.store import Chunk, Citation, Paper, Store

DIM = 4


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "library.db", embedding_dim=DIM) as s:
        yield s


def make_paper(pid="arxiv:1706.03762", **kw):
    defaults = dict(
        title="Attention Is All You Need",
        source="arxiv",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        arxiv_id="1706.03762",
    )
    return Paper(id=pid, **(defaults | kw))


def test_add_and_get_paper(store):
    store.add_paper(make_paper())
    paper = store.get_paper("arxiv:1706.03762")
    assert paper.title == "Attention Is All You Need"
    assert paper.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert paper.added_at
    assert store.get_paper("missing") is None


def test_duplicates_rejected_and_found(store):
    store.add_paper(make_paper())
    with pytest.raises(sqlite3.IntegrityError):
        store.add_paper(make_paper(pid="arxiv:other"))
    assert store.find_duplicate(arxiv_id="1706.03762").id == "arxiv:1706.03762"
    assert store.find_duplicate(sha256="nope") is None


def test_source_must_be_valid(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.add_paper(make_paper(source="web"))


def test_chunks_fts_and_vectors(store):
    store.add_paper(make_paper())
    chunks = [
        Chunk(seq=0, section="Abstract", page=1, text="The dominant sequence transduction models"),
        Chunk(seq=1, section="Method", page=3, text="Scaled dot-product attention"),
    ]
    store.add_chunks("arxiv:1706.03762", chunks, embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]])

    assert [c.seq for c in store.get_chunks("arxiv:1706.03762")] == [0, 1]
    assert store.get_chunks("arxiv:1706.03762", section="method")[0].page == 3

    fts = store.db.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'attention'").fetchall()
    assert len(fts) == 1
    assert store.db.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0] == 2


def test_embedding_length_checked(store):
    store.add_paper(make_paper())
    with pytest.raises(ValueError):
        store.add_chunks("arxiv:1706.03762", [Chunk(seq=0, text="x")], embeddings=[[1, 2]])


def test_delete_cascades(store):
    store.add_paper(make_paper())
    store.add_chunks("arxiv:1706.03762", [Chunk(seq=0, text="attention")], embeddings=[[1, 0, 0, 0]])
    store.add_citations("arxiv:1706.03762", [Citation(seq=0, raw_reference="[1] Bahdanau et al.")])
    store.add_note("arxiv:1706.03762", "read section 3")

    assert store.delete_paper("arxiv:1706.03762")
    for table in ("chunks", "chunk_vectors", "citations", "notes"):
        assert store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0, table
    assert store.db.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'attention'").fetchone()[0] == 0


def test_citation_match_cleared_when_target_deleted(store):
    store.add_paper(make_paper())
    store.add_paper(make_paper(pid="pdf:abc", source="pdf", arxiv_id=None, sha256="abc", title="Other"))
    store.add_citations("pdf:abc", [Citation(seq=0, raw_reference="Vaswani 2017", matched_paper_id="arxiv:1706.03762")])
    store.delete_paper("arxiv:1706.03762")
    assert store.get_citations("pdf:abc")[0].matched_paper_id is None


def test_notes(store):
    store.add_paper(make_paper())
    note = store.add_note("arxiv:1706.03762", "key idea: multi-head attention")
    assert store.get_notes("arxiv:1706.03762") == [note]


def test_embedding_dim_mismatch_on_reopen(tmp_path):
    Store(tmp_path / "lib.db", embedding_dim=DIM).close()
    with pytest.raises(ValueError):
        Store(tmp_path / "lib.db", embedding_dim=8)
