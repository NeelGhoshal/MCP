"""Hybrid (semantic + keyword) search over the library."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import sqlite_vec

from librarian.embed import Embedder
from librarian.store import Store

Mode = Literal["hybrid", "semantic", "keyword"]

RRF_K = 60  # standard reciprocal-rank-fusion constant; dampens the gap between rank 1 and rank 2
MIN_CANDIDATES = 50
SNIPPET_CHARS = 320
TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass
class SearchHit:
    paper_id: str
    title: str
    year: int | None
    section: str
    page: int | None
    snippet: str
    score: float
    matched_by: list[str] = field(default_factory=list)  # "semantic" and/or "keyword"
    chunk_id: int = field(default=0, repr=False)


@dataclass
class _Filters:
    year_from: int | None = None
    year_to: int | None = None
    paper_ids: list[str] | None = None

    def sql(self) -> tuple[str, list]:
        clauses, params = [], []
        if self.year_from is not None:
            clauses.append("p.year >= ?")
            params.append(self.year_from)
        if self.year_to is not None:
            clauses.append("p.year <= ?")
            params.append(self.year_to)
        if self.paper_ids:
            clauses.append(f"p.id IN ({','.join('?' * len(self.paper_ids))})")
            params.extend(self.paper_ids)
        return "".join(f" AND {c}" for c in clauses), params


def fts_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 query: each word quoted, OR-ed (bm25 rewards matching more of them)."""
    terms = [t for t in TOKEN.findall(text.lower()) if len(t) > 1]
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


def keyword_search(store: Store, query: str, k: int, filters: _Filters | None = None) -> list[int]:
    """Chunk ids ranked by BM25."""
    match = fts_query(query)
    if match is None:
        return []
    where, params = (filters or _Filters()).sql()
    rows = store.db.execute(
        "SELECT c.id FROM chunks_fts f"
        " JOIN chunks c ON c.id = f.rowid JOIN papers p ON p.id = c.paper_id"
        f" WHERE chunks_fts MATCH ?{where} ORDER BY f.rank LIMIT ?",
        [match, *params, k],
    ).fetchall()
    return [r[0] for r in rows]


def vector_search(store: Store, embedding: list[float], k: int, filters: _Filters | None = None) -> list[int]:
    """Chunk ids ranked by embedding distance.

    vec0 can't filter on paper columns inside the KNN scan, so with filters we over-fetch and filter after.
    """
    filters = filters or _Filters()
    where, params = filters.sql()
    pool = k if not where else max(k * 10, 200)
    rows = store.db.execute(
        "WITH knn AS ("
        "  SELECT chunk_id, distance FROM chunk_vectors WHERE embedding MATCH ? AND k = ?"
        ") SELECT knn.chunk_id FROM knn"
        " JOIN chunks c ON c.id = knn.chunk_id JOIN papers p ON p.id = c.paper_id"
        f" WHERE 1 = 1{where} ORDER BY knn.distance LIMIT ?",
        [sqlite_vec.serialize_float32(embedding), pool, *params, k],
    ).fetchall()
    return [r[0] for r in rows]


def rrf(rankings: dict[str, list[int]]) -> list[tuple[int, float, list[str]]]:
    """Reciprocal rank fusion: score = sum over lists of 1 / (RRF_K + rank)."""
    scores: dict[int, float] = {}
    sources: dict[int, list[str]] = {}
    for name, ids in rankings.items():
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            sources.setdefault(chunk_id, []).append(name)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(cid, scores[cid], sources[cid]) for cid in ordered]


def search(
    store: Store,
    query: str,
    embedder: Embedder | None = None,
    limit: int = 8,
    mode: Mode = "hybrid",
    per_paper: int = 3,
    year_from: int | None = None,
    year_to: int | None = None,
    paper_ids: list[str] | None = None,
) -> list[SearchHit]:
    """Search chunks across the library. At most `per_paper` hits are returned for any one paper."""
    if mode not in ("hybrid", "semantic", "keyword"):
        raise ValueError(f"unknown mode: {mode}")
    if mode != "keyword" and embedder is None:
        raise ValueError(f"mode={mode!r} needs an embedder")
    if not query.strip() or limit <= 0:
        return []

    filters = _Filters(year_from, year_to, paper_ids)
    k = max(limit * per_paper * 2, MIN_CANDIDATES)
    rankings: dict[str, list[int]] = {}
    if mode in ("hybrid", "semantic"):
        rankings["semantic"] = vector_search(store, embedder.embed([query])[0], k, filters)
    if mode in ("hybrid", "keyword"):
        rankings["keyword"] = keyword_search(store, query, k, filters)

    fused = rrf(rankings)
    rows = _load_chunks(store, [cid for cid, _, _ in fused])
    terms = [t for t in TOKEN.findall(query.lower()) if len(t) > 1]

    hits: list[SearchHit] = []
    taken: dict[str, int] = {}
    for chunk_id, score, matched_by in fused:
        row = rows[chunk_id]
        if taken.get(row["paper_id"], 0) >= per_paper:
            continue
        taken[row["paper_id"]] = taken.get(row["paper_id"], 0) + 1
        hits.append(
            SearchHit(
                paper_id=row["paper_id"],
                title=row["title"],
                year=row["year"],
                section=row["section"],
                page=row["page"],
                snippet=make_snippet(row["text"], terms),
                score=round(score, 5),
                matched_by=matched_by,
                chunk_id=chunk_id,
            )
        )
        if len(hits) == limit:
            break
    return hits


def _load_chunks(store: Store, chunk_ids: list[int]) -> dict:
    if not chunk_ids:
        return {}
    rows = store.db.execute(
        "SELECT c.id, c.paper_id, c.section, c.page, c.text, p.title, p.year"
        " FROM chunks c JOIN papers p ON p.id = c.paper_id"
        f" WHERE c.id IN ({','.join('?' * len(chunk_ids))})",
        chunk_ids,
    ).fetchall()
    return {r["id"]: r for r in rows}


def make_snippet(text: str, terms: list[str], width: int = SNIPPET_CHARS) -> str:
    """The window of `text` containing the most distinct query terms, or the start of the chunk."""
    if len(text) <= width:
        return text
    lower = text.lower()
    # Short words ("of", "the", "how") match everywhere, so they don't pick the window.
    meaningful = [t for t in terms if len(t) > 3] or terms
    hits = sorted(
        (m.start(), t) for t in meaningful for m in re.finditer(rf"\b{re.escape(t)}", lower)
    )
    centre, best = 0, 0
    for pos, _ in hits:
        distinct = len({t for p, t in hits if pos <= p < pos + width * 2 // 3})
        if distinct > best:
            centre, best = pos, distinct
    start = max(0, min(centre - width // 3, len(text) - width))
    # Snap to word boundaries.
    if start:
        start = text.find(" ", start) + 1
    end = text.rfind(" ", 0, start + width)
    end = end if end > start else start + width
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")
