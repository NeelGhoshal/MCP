"""MCP server and tool definitions."""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anyio.from_thread
import requests
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from librarian.embed import Embedder, SentenceTransformerEmbedder
from librarian.ingest import ingest, parse_arxiv_id
from librarian.search import search as run_search
from librarian.store import Chunk, Paper, Store

INSTRUCTIONS = """\
A local library of research papers (arXiv and PDFs), searchable by meaning and keyword.

- add_paper: add by arXiv id/URL or local PDF path (first use downloads a ~90 MB embedding model).
- search: find passages across the library; results cite paper, section and page.
- get_paper: metadata and section list, or the full text of one section.
- get_citations: a paper's references (marked when in the library) and which library papers cite it.
- compare_papers: relevant passages per paper per aspect, to build comparison tables.
- add_note / list_papers / remove_paper: manage the library.

Cite sources as (Title, Section, p.N) when answering from search results.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
MAX_OVERLAP_WORDS = 60


@dataclass
class AppState:
    store: Store
    embedder: Embedder
    lock: threading.Lock  # one SQLite connection is shared by tool threads; hold this while using it


def create_server(db_path: str | Path | None = None, embedder: Embedder | None = None) -> MCPServer[AppState]:
    # Static resources get no Context, so they read the running state from here instead.
    running: list[AppState] = []

    @asynccontextmanager
    async def lifespan(_server: MCPServer[AppState]) -> AsyncIterator[AppState]:
        emb = embedder or SentenceTransformerEmbedder()
        if isinstance(emb, SentenceTransformerEmbedder):
            emb.warm_up()
        store = Store(db_path, embedding_dim=emb.dim)
        app = AppState(store=store, embedder=emb, lock=threading.Lock())
        running.append(app)
        try:
            yield app
        finally:
            running.remove(app)
            store.close()

    mcp: MCPServer[AppState] = MCPServer("paper-librarian", instructions=INSTRUCTIONS, lifespan=lifespan)

    def state(ctx: Context) -> AppState:
        return ctx.request_context.lifespan_context

    @mcp.tool(annotations=ToolAnnotations(open_world_hint=True, idempotent_hint=True))
    def add_paper(source: str, ctx: Context) -> dict[str, Any]:
        """Add a paper to the library from an arXiv id or URL (e.g. "1706.03762") or a local PDF path.

        Downloads, extracts sections, indexes for search, and links its references to papers already
        in the library. Adding a paper that is already present returns it without re-processing.
        """
        app = state(ctx)
        steps = iter(range(1, 100))

        def progress(message: str) -> None:
            anyio.from_thread.run(ctx.report_progress, next(steps), None, message)

        try:
            with app.lock:
                result = ingest(source, app.store, app.embedder, progress=progress)
        except (FileNotFoundError, ValueError) as e:
            raise ToolError(str(e)) from e
        except requests.RequestException as e:
            raise ToolError(f"Download failed for {source!r}: {e}") from e

        out: dict[str, Any] = {"paper": _paper_summary(result.paper), "already_in_library": result.duplicate}
        if not result.duplicate:
            out["chunks_indexed"] = result.chunk_count
            if result.citations:
                out["references_found"] = result.citations.references
                out["references_in_library"] = result.citations.linked
                out["cited_by_in_library"] = result.citations.cited_by
        return out

    @mcp.tool(annotations=READ_ONLY)
    def search(
        query: str,
        ctx: Context,
        limit: int = 8,
        mode: Literal["hybrid", "semantic", "keyword"] = "hybrid",
        year_from: int | None = None,
        year_to: int | None = None,
        paper_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search passages across the library.

        "hybrid" (default) combines meaning-based and keyword matching; use "keyword" for exact terms
        such as dataset names. Filter by publication year or restrict to specific paper ids.
        """
        app = state(ctx)
        with app.lock:
            ids = [_resolve_id(app.store, pid) for pid in paper_ids] if paper_ids else None
            hits = run_search(
                app.store, query, app.embedder, limit=max(1, min(limit, 30)), mode=mode,
                year_from=year_from, year_to=year_to, paper_ids=ids,
            )
        return {
            "results": [
                {
                    "paper_id": h.paper_id,
                    "title": h.title,
                    "year": h.year,
                    "section": h.section,
                    "page": h.page,
                    "snippet": h.snippet,
                    "matched_by": h.matched_by,
                }
                for h in hits
            ]
        }

    @mcp.tool(annotations=READ_ONLY)
    def get_paper(paper_id: str, ctx: Context, section: str | None = None) -> dict[str, Any]:
        """Get a paper's metadata, abstract, section list and notes, or with `section`, that section's full text."""
        app = state(ctx)
        with app.lock:
            paper = _get(app.store, paper_id)
            chunks = app.store.get_chunks(paper.id)
            notes = app.store.get_notes(paper.id)

        sections = _sections(chunks)
        if section is None:
            return {
                **_paper_summary(paper),
                "abstract": paper.abstract,
                "path": paper.path,
                "sections": [{"name": name, "page": page} for name, page in sections],
                "notes": [{"text": n.text, "created_at": n.created_at} for n in notes],
            }

        wanted = _match_section(section, [name for name, _ in sections])
        if wanted is None:
            raise ToolError(f"No section {section!r} in {paper.id}. Sections: {', '.join(n for n, _ in sections)}")
        picked = [c for c in chunks if c.section == wanted]
        return {
            "paper_id": paper.id,
            "title": paper.title,
            "section": wanted,
            "pages": sorted({c.page for c in picked if c.page}),
            "text": join_chunks(picked),
        }

    @mcp.tool(annotations=READ_ONLY)
    def get_citations(paper_id: str, ctx: Context) -> dict[str, Any]:
        """List a paper's references (with the library paper each one points to, if any) and the
        library papers that cite it."""
        app = state(ctx)
        with app.lock:
            paper = _get(app.store, paper_id)
            refs = app.store.get_citations(paper.id)
            cited_by = app.store.get_cited_by(paper.id)
            titles = {p.id: p.title for p in app.store.list_papers()}
        return {
            "paper_id": paper.id,
            "title": paper.title,
            "references": [
                {"n": c.seq + 1, "reference": c.raw_reference, "in_library": c.matched_paper_id}
                for c in refs
            ],
            "references_in_library": sum(1 for c in refs if c.matched_paper_id),
            "cited_by": [
                {"paper_id": c.paper_id, "title": titles.get(c.paper_id), "as_reference": c.raw_reference}
                for c in cited_by
            ],
        }

    @mcp.tool(annotations=READ_ONLY)
    def compare_papers(
        paper_ids: list[str], aspects: list[str], ctx: Context, passages_per_aspect: int = 2
    ) -> dict[str, Any]:
        """Gather the most relevant passages from each paper for each aspect (e.g. "method",
        "datasets", "results"), so a comparison table can be written from the evidence."""
        if not 2 <= len(paper_ids) <= 8:
            raise ToolError("compare_papers needs between 2 and 8 paper ids")
        if not 1 <= len(aspects) <= 8:
            raise ToolError("compare_papers needs between 1 and 8 aspects")
        app = state(ctx)
        k = max(1, min(passages_per_aspect, 5))
        with app.lock:
            papers = [_get(app.store, pid) for pid in paper_ids]
            table: dict[str, Any] = {}
            for paper in papers:
                table[paper.id] = {"title": paper.title, "year": paper.year, "aspects": {}}
                for aspect in aspects:
                    hits = run_search(app.store, aspect, app.embedder, limit=k, per_paper=k, paper_ids=[paper.id])
                    table[paper.id]["aspects"][aspect] = [
                        {"section": h.section, "page": h.page, "passage": h.snippet} for h in hits
                    ]
        return {"papers": table}

    @mcp.tool()
    def add_note(paper_id: str, text: str, ctx: Context) -> dict[str, Any]:
        """Attach a note to a paper. Notes are returned by get_paper."""
        if not text.strip():
            raise ToolError("Note text is empty")
        app = state(ctx)
        with app.lock:
            paper = _get(app.store, paper_id)
            note = app.store.add_note(paper.id, text.strip())
        return {"paper_id": paper.id, "note_id": note.id, "created_at": note.created_at}

    @mcp.tool(annotations=READ_ONLY)
    def list_papers(ctx: Context) -> dict[str, Any]:
        """List every paper in the library, newest first."""
        app = state(ctx)
        with app.lock:
            papers = app.store.list_papers()
        return {"count": len(papers), "papers": [_paper_summary(p) for p in papers]}

    @mcp.tool(annotations=ToolAnnotations(destructive_hint=True, idempotent_hint=True, open_world_hint=False))
    def remove_paper(paper_id: str, ctx: Context) -> dict[str, Any]:
        """Remove a paper and its index, notes and references from the library. The PDF file is kept."""
        app = state(ctx)
        with app.lock:
            paper = _get(app.store, paper_id)
            app.store.delete_paper(paper.id)
        return {"removed": paper.id, "title": paper.title}

    @mcp.resource("library://papers", name="papers", mime_type="text/markdown")
    def papers_resource() -> str:
        """Every paper in the library as a Markdown list."""
        app = running[-1]
        with app.lock:
            papers = app.store.list_papers()
        if not papers:
            return "The library is empty. Add papers with the add_paper tool."
        lines = [f"# Library ({len(papers)} papers)", ""]
        for p in papers:
            authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
            lines.append(f"- **{p.title}** ({p.year or 'n.d.'}) — {authors or 'unknown authors'} `{p.id}`")
        return "\n".join(lines)

    @mcp.prompt()
    def literature_review(topic: str) -> str:
        """Write a short literature review on a topic from the papers in the library."""
        return (
            f"Write a literature review on: {topic}\n\n"
            "1. Call list_papers, then search for the topic and 2-3 related phrasings.\n"
            "2. Pick the relevant papers and call compare_papers on them with aspects such as "
            "problem, method, datasets, results and limitations.\n"
            "3. Use get_citations to see how the papers build on each other.\n"
            "4. Write the review: overview, a comparison table, how the work evolved, open problems.\n"
            "Cite every claim as (Title, Section, p.N). Say so where the library lacks coverage "
            "instead of filling gaps from memory."
        )

    return mcp


# --- helpers ---


def _resolve_id(store: Store, paper_id: str) -> str:
    """Accept full ids ("arxiv:1706.03762") or bare arXiv ids/URLs ("1706.03762")."""
    paper_id = paper_id.strip()
    if store.get_paper(paper_id):
        return paper_id
    if (arxiv_id := parse_arxiv_id(paper_id)) and store.get_paper(f"arxiv:{arxiv_id}"):
        return f"arxiv:{arxiv_id}"
    raise ToolError(f"No paper {paper_id!r} in the library. Use list_papers to see paper ids.")


def _get(store: Store, paper_id: str) -> Paper:
    return store.get_paper(_resolve_id(store, paper_id))


def _paper_summary(p: Paper) -> dict[str, Any]:
    return {"paper_id": p.id, "title": p.title, "authors": p.authors, "year": p.year, "source": p.source}


def _sections(chunks: list[Chunk]) -> list[tuple[str, int | None]]:
    """Distinct section names in reading order, with the page each starts on."""
    seen: dict[str, int | None] = {}
    for c in chunks:
        seen.setdefault(c.section, c.page)
    return list(seen.items())


def _match_section(query: str, names: list[str]) -> str | None:
    q = query.strip().lower()
    exact = [n for n in names if n.lower() == q]
    partial = [n for n in names if q in n.lower()]
    return (exact or partial or [None])[0]


def join_chunks(chunks: list[Chunk]) -> str:
    """Rebuild running text from consecutive chunks, dropping the words repeated by chunk overlap."""
    words: list[str] = []
    for chunk in chunks:
        new = chunk.text.split()
        overlap = next(
            (k for k in range(min(MAX_OVERLAP_WORDS, len(words), len(new)), 0, -1) if words[-k:] == new[:k]),
            0,
        )
        words.extend(new[overlap:])
    return " ".join(words)


def main() -> None:
    # Model-hub HTTP chatter would otherwise fill the host's MCP log on every start.
    for noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    create_server().run()


if __name__ == "__main__":
    main()
