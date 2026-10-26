"""End-to-end tests through a real MCP client connected in-process."""

import pytest
from mcp import Client

from librarian.server import create_server, join_chunks
from librarian.store import Chunk
from tests.test_ingest import FakeEmbedder, body, make_pdf

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pdfs(tmp_path):
    widgets = make_pdf(
        tmp_path / "widgets.pdf",
        [
            [("A Study of Widgets", 20), ("Abstract", 12), ("We study widgets in depth.", 10),
             ("1 Introduction", 12), *body(30, "intro")],
            [("2 Widget Architecture", 12), *body(40, "arch"), ("References", 12),
             ("[1] A. Smith. Gadgets and gizmos for everyone. 2019.", 10), ("[2] B. Lee. Gizmos. 2020.", 10)],
        ],
    )
    gadgets = make_pdf(tmp_path / "gadgets.pdf", [[("Gadgets and Gizmos for Everyone", 20), *body(25, "gadget")]])
    return widgets, gadgets


@pytest.fixture
async def client(tmp_path):
    server = create_server(tmp_path / "library.db", embedder=FakeEmbedder())
    async with Client(server, raise_exceptions=True) as c:
        yield c


async def call(client, tool, **args):
    result = await client.call_tool(tool, args)
    assert not result.is_error, result.content
    return result.structured_content


async def call_error(client, tool, **args) -> str:
    result = await client.call_tool(tool, args)
    assert result.is_error
    return result.content[0].text


async def test_lists_tools_with_annotations(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {
        "add_paper", "search", "get_paper", "get_citations", "compare_papers", "add_note", "list_papers", "remove_paper",
    }
    assert tools["search"].annotations.read_only_hint
    assert tools["remove_paper"].annotations.destructive_hint
    assert "paper_ids" in tools["search"].input_schema["properties"]


async def test_add_paper_and_progress(client, pdfs):
    messages = []

    async def on_progress(progress, total, message):
        messages.append(message)

    result = await client.call_tool("add_paper", {"source": str(pdfs[0])}, progress_callback=on_progress)
    out = result.structured_content
    assert out["paper"]["title"] == "A Study of Widgets" and not out["already_in_library"]
    assert out["chunks_indexed"] > 0 and out["references_found"] == 2
    assert any("Embedding" in (m or "") for m in messages)

    again = await call(client, "add_paper", source=str(pdfs[0]))
    assert again["already_in_library"] and "chunks_indexed" not in again


async def test_add_paper_errors_are_tool_errors(client, tmp_path):
    assert "no such file" in await call_error(client, "add_paper", source=str(tmp_path / "missing.pdf"))


async def test_list_search_and_get_paper(client, pdfs):
    added = await call(client, "add_paper", source=str(pdfs[0]))
    pid = added["paper"]["paper_id"]

    listed = await call(client, "list_papers")
    assert listed["count"] == 1 and listed["papers"][0]["paper_id"] == pid

    hits = (await call(client, "search", query="arch5 arch6", mode="keyword"))["results"]
    assert hits[0]["paper_id"] == pid and hits[0]["section"] == "Widget Architecture" and hits[0]["page"] == 2

    paper = await call(client, "get_paper", paper_id=pid)
    assert paper["abstract"] == "We study widgets in depth."
    assert [s["name"] for s in paper["sections"]] == ["Front Matter", "Abstract", "Introduction", "Widget Architecture"]

    section = await call(client, "get_paper", paper_id=pid, section="architecture")
    assert section["section"] == "Widget Architecture"
    assert section["text"].split() == [f"arch{i}" for i in range(40)]

    assert "Sections:" in await call_error(client, "get_paper", paper_id=pid, section="Nope")
    assert "list_papers" in await call_error(client, "get_paper", paper_id="arxiv:0000.00000")


async def test_citations_link_both_ways(client, pdfs):
    widgets = (await call(client, "add_paper", source=str(pdfs[0])))["paper"]["paper_id"]
    added = await call(client, "add_paper", source=str(pdfs[1]))
    gadgets = added["paper"]["paper_id"]
    assert added["cited_by_in_library"] == 1  # widgets' reference [1] now points at it

    refs = await call(client, "get_citations", paper_id=widgets)
    assert [r["in_library"] for r in refs["references"]] == [gadgets, None]
    cited = await call(client, "get_citations", paper_id=gadgets)
    assert [c["paper_id"] for c in cited["cited_by"]] == [widgets]


async def test_notes_compare_and_remove(client, pdfs):
    a = (await call(client, "add_paper", source=str(pdfs[0])))["paper"]["paper_id"]
    b = (await call(client, "add_paper", source=str(pdfs[1])))["paper"]["paper_id"]

    await call(client, "add_note", paper_id=a, text="  check section 2  ")
    assert (await call(client, "get_paper", paper_id=a))["notes"][0]["text"] == "check section 2"
    assert "empty" in await call_error(client, "add_note", paper_id=a, text="   ")

    compared = await call(client, "compare_papers", paper_ids=[a, b], aspects=["architecture", "gadgets"])
    assert set(compared["papers"]) == {a, b}
    assert compared["papers"][a]["aspects"]["architecture"]
    assert "between 2 and 8" in await call_error(client, "compare_papers", paper_ids=[a], aspects=["x"])

    assert (await call(client, "remove_paper", paper_id=a))["removed"] == a
    assert (await call(client, "list_papers"))["count"] == 1


async def test_resource_and_prompt(client, pdfs):
    empty = await client.read_resource("library://papers")
    assert "empty" in empty.contents[0].text

    await call(client, "add_paper", source=str(pdfs[0]))
    listing = (await client.read_resource("library://papers")).contents[0].text
    assert "A Study of Widgets" in listing and "(1 papers)" in listing

    prompt = await client.get_prompt("literature_review", {"topic": "widgets"})
    assert "widgets" in prompt.messages[0].content.text and "compare_papers" in prompt.messages[0].content.text


async def test_bare_arxiv_ids_resolve(client, pdfs, monkeypatch):
    added = await call(client, "add_paper", source=str(pdfs[0]))
    pid = added["paper"]["paper_id"]
    assert (await call(client, "get_paper", paper_id=f"  {pid} "))["paper_id"] == pid


def test_join_chunks_removes_overlap():
    chunks = [Chunk(seq=0, text="a b c d e f"), Chunk(seq=1, text="e f g h"), Chunk(seq=2, text="x y")]
    assert join_chunks(chunks) == "a b c d e f g h x y"
