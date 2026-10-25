"""SQLite metadata and vector index."""

from __future__ import annotations

import os
import sqlite3
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec

DEFAULT_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    id        TEXT PRIMARY KEY,          -- "arxiv:2310.06825" or "pdf:<sha256[:16]>"
    title     TEXT NOT NULL,
    authors   TEXT NOT NULL DEFAULT '',  -- "; "-separated
    year      INTEGER,
    abstract  TEXT NOT NULL DEFAULT '',
    source    TEXT NOT NULL CHECK (source IN ('arxiv', 'pdf')),
    arxiv_id  TEXT UNIQUE,
    sha256    TEXT UNIQUE,
    path      TEXT,
    added_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    paper_id  TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    seq       INTEGER NOT NULL,          -- order within the paper
    section   TEXT NOT NULL DEFAULT '',
    page      INTEGER,
    text      TEXT NOT NULL,
    UNIQUE (paper_id, seq)
);
CREATE INDEX IF NOT EXISTS chunks_paper ON chunks(paper_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, section, content='chunks', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, section) VALUES (new.id, new.text, new.section);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, section)
        VALUES ('delete', old.id, old.text, old.section);
    DELETE FROM chunk_vectors WHERE chunk_id = old.id;
END;

CREATE TABLE IF NOT EXISTS citations (
    id                INTEGER PRIMARY KEY,
    paper_id          TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    seq               INTEGER NOT NULL,
    raw_reference     TEXT NOT NULL,
    matched_paper_id  TEXT REFERENCES papers(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS citations_paper ON citations(paper_id);

CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY,
    paper_id    TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def library_home() -> Path:
    return Path(os.environ.get("LIBRARIAN_HOME", Path.home() / ".librarian"))


def default_db_path() -> Path:
    return library_home() / "library.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Paper:
    id: str
    title: str
    source: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    arxiv_id: str | None = None
    sha256: str | None = None
    path: str | None = None
    added_at: str = ""


@dataclass
class Chunk:
    seq: int
    text: str
    section: str = ""
    page: int | None = None


@dataclass
class Note:
    id: int
    paper_id: str
    text: str
    created_at: str


@dataclass
class Citation:
    seq: int
    raw_reference: str
    matched_paper_id: str | None = None
    id: int | None = None
    paper_id: str | None = None  # the citing paper


class Store:
    def __init__(self, db_path: str | Path | None = None, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        path = Path(db_path) if db_path is not None else default_db_path()
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.embedding_dim = embedding_dim
        self._init_schema()

    def _init_schema(self) -> None:
        with self.db:
            # vec0 must exist before the chunks_ad trigger references it.
            self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0("
                f"chunk_id INTEGER PRIMARY KEY, embedding float[{self.embedding_dim}])"
            )
            self.db.executescript(SCHEMA)
            row = self.db.execute("SELECT value FROM meta WHERE key = 'embedding_dim'").fetchone()
            if row is None:
                self.db.execute(
                    "INSERT INTO meta(key, value) VALUES ('embedding_dim', ?)", (str(self.embedding_dim),)
                )
            elif int(row["value"]) != self.embedding_dim:
                raise ValueError(
                    f"Library was built with embedding_dim={row['value']}, got {self.embedding_dim}"
                )

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # --- papers ---

    def transaction(self, commit: bool = True):
        """Commit on exit, or with commit=False join a transaction the caller manages via `with store.db:`."""
        return self.db if commit else nullcontext()

    def add_paper(self, paper: Paper, commit: bool = True) -> Paper:
        paper.added_at = paper.added_at or _now()
        with self.transaction(commit):
            self.db.execute(
                "INSERT INTO papers (id, title, authors, year, abstract, source, arxiv_id, sha256, path, added_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    paper.id, paper.title, "; ".join(paper.authors), paper.year, paper.abstract,
                    paper.source, paper.arxiv_id, paper.sha256, paper.path, paper.added_at,
                ),
            )
        return paper

    def get_paper(self, paper_id: str) -> Paper | None:
        row = self.db.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return _row_to_paper(row) if row else None

    def find_duplicate(self, arxiv_id: str | None = None, sha256: str | None = None) -> Paper | None:
        row = self.db.execute(
            "SELECT * FROM papers WHERE (arxiv_id IS NOT NULL AND arxiv_id = ?)"
            " OR (sha256 IS NOT NULL AND sha256 = ?)",
            (arxiv_id, sha256),
        ).fetchone()
        return _row_to_paper(row) if row else None

    def list_papers(self) -> list[Paper]:
        rows = self.db.execute("SELECT * FROM papers ORDER BY added_at DESC, id").fetchall()
        return [_row_to_paper(r) for r in rows]

    def delete_paper(self, paper_id: str) -> bool:
        with self.db:
            cur = self.db.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        return cur.rowcount > 0

    # --- chunks + vectors ---

    def add_chunks(
        self,
        paper_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
        commit: bool = True,
    ) -> list[int]:
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("chunks and embeddings must have the same length")
        ids: list[int] = []
        with self.transaction(commit):
            for i, chunk in enumerate(chunks):
                cur = self.db.execute(
                    "INSERT INTO chunks (paper_id, seq, section, page, text) VALUES (?, ?, ?, ?, ?)",
                    (paper_id, chunk.seq, chunk.section, chunk.page, chunk.text),
                )
                ids.append(cur.lastrowid)
                if embeddings is not None:
                    self._insert_vector(cur.lastrowid, embeddings[i])
        return ids

    def _insert_vector(self, chunk_id: int, embedding: list[float]) -> None:
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"expected embedding of length {self.embedding_dim}, got {len(embedding)}")
        self.db.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(embedding)),
        )

    def get_chunks(self, paper_id: str, section: str | None = None) -> list[Chunk]:
        sql = "SELECT seq, section, page, text FROM chunks WHERE paper_id = ?"
        params: list = [paper_id]
        if section is not None:
            sql += " AND section = ? COLLATE NOCASE"
            params.append(section)
        rows = self.db.execute(sql + " ORDER BY seq", params).fetchall()
        return [Chunk(seq=r["seq"], section=r["section"], page=r["page"], text=r["text"]) for r in rows]

    # --- citations ---

    def add_citations(self, paper_id: str, citations: list[Citation], commit: bool = True) -> None:
        with self.transaction(commit):
            self.db.executemany(
                "INSERT INTO citations (paper_id, seq, raw_reference, matched_paper_id) VALUES (?, ?, ?, ?)",
                [(paper_id, c.seq, c.raw_reference, c.matched_paper_id) for c in citations],
            )

    def get_citations(self, paper_id: str) -> list[Citation]:
        """References made by `paper_id`, in order."""
        rows = self.db.execute(
            "SELECT * FROM citations WHERE paper_id = ? ORDER BY seq", (paper_id,)
        ).fetchall()
        return [_row_to_citation(r) for r in rows]

    def get_cited_by(self, paper_id: str) -> list[Citation]:
        """References in other library papers that point at `paper_id`."""
        rows = self.db.execute(
            "SELECT * FROM citations WHERE matched_paper_id = ? ORDER BY paper_id, seq", (paper_id,)
        ).fetchall()
        return [_row_to_citation(r) for r in rows]

    def unmatched_citations(self, exclude_paper_id: str | None = None) -> list[Citation]:
        rows = self.db.execute(
            "SELECT * FROM citations WHERE matched_paper_id IS NULL AND paper_id IS NOT ?", (exclude_paper_id,)
        ).fetchall()
        return [_row_to_citation(r) for r in rows]

    def set_citation_match(self, citation_id: int, matched_paper_id: str | None, commit: bool = True) -> None:
        with self.transaction(commit):
            self.db.execute("UPDATE citations SET matched_paper_id = ? WHERE id = ?", (matched_paper_id, citation_id))

    def delete_citations(self, paper_id: str, commit: bool = True) -> None:
        with self.transaction(commit):
            self.db.execute("DELETE FROM citations WHERE paper_id = ?", (paper_id,))

    # --- notes ---

    def add_note(self, paper_id: str, text: str) -> Note:
        created_at = _now()
        with self.db:
            cur = self.db.execute(
                "INSERT INTO notes (paper_id, text, created_at) VALUES (?, ?, ?)", (paper_id, text, created_at)
            )
        return Note(id=cur.lastrowid, paper_id=paper_id, text=text, created_at=created_at)

    def get_notes(self, paper_id: str) -> list[Note]:
        rows = self.db.execute(
            "SELECT id, paper_id, text, created_at FROM notes WHERE paper_id = ? ORDER BY id", (paper_id,)
        ).fetchall()
        return [Note(**dict(r)) for r in rows]


def _row_to_citation(row: sqlite3.Row) -> Citation:
    return Citation(
        seq=row["seq"],
        raw_reference=row["raw_reference"],
        matched_paper_id=row["matched_paper_id"],
        id=row["id"],
        paper_id=row["paper_id"],
    )


def _row_to_paper(row: sqlite3.Row) -> Paper:
    return Paper(
        id=row["id"],
        title=row["title"],
        source=row["source"],
        authors=[a for a in row["authors"].split("; ") if a],
        year=row["year"],
        abstract=row["abstract"],
        arxiv_id=row["arxiv_id"],
        sha256=row["sha256"],
        path=row["path"],
        added_at=row["added_at"],
    )
