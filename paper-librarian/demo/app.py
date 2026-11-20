"""Public demo: the real paper-librarian MCP server behind a small web UI.

The page calls tools through a genuine MCP client connected in-process and shows the JSON-RPC
messages, and an "Ask" mode lets an LLM (Groq) drive the same tools the way Claude would.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import sqlite_shim  # noqa: F401  (must precede librarian imports)
import anyio
import requests
from mcp import Client
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from librarian.ingest import parse_arxiv_id
from librarian.server import INSTRUCTIONS, create_server
from onnx_embed import OnnxEmbedder
from remote_mcp import build_public_mcp

HERE = Path(__file__).parent
DB_PATH = Path(os.environ.get("DEMO_DB", HERE / "data" / "library.db"))
GROQ_KEY = os.environ.get("GROQ_KEY") or os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_PAPERS = int(os.environ.get("DEMO_MAX_PAPERS", "12"))
MAX_STEPS = 6  # LLM turns that may call tools; one more turn is forced to answer
TOOL_RESULT_CHARS = 4000  # hard cap; Groq free tier allows ~8k tokens/minute

# Tools that would let one visitor change or deface the shared library for everyone else.
BLOCKED = {"remove_paper": "Removing papers is disabled in the public demo.",
           "add_note": "Notes are disabled in the public demo (the library is shared by all visitors)."}

# Hidden from the assistant: models tend to re-add papers already present. Still callable in the explorer.
ASK_HIDDEN = {"add_paper"}

state: dict[str, Any] = {}

# One model in memory, shared by the demo's in-process server and the public /mcp endpoint.
EMBEDDER = OnnxEmbedder()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
public_mcp, public_mcp_route = build_public_mcp(DB_PATH, EMBEDDER)


@asynccontextmanager
async def lifespan(_app: Starlette):
    server = create_server(DB_PATH, embedder=EMBEDDER)
    async with public_mcp.session_manager.run(), Client(server) as client:
        state["client"] = client
        state["tools"] = (await client.list_tools()).tools
        state["next_id"] = 1
        yield


def _rpc_id() -> int:
    state["next_id"] += 1
    return state["next_id"]


def _guard(name: str, args: dict[str, Any]) -> str | None:
    if name in BLOCKED:
        return BLOCKED[name]
    if name == "add_paper":
        if not parse_arxiv_id(str(args.get("source", ""))):
            return "The public demo only adds papers by arXiv id or URL (no local files on this server)."
    return None


async def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call one tool through the MCP client; return the JSON-RPC request/response pair as sent."""
    rid = _rpc_id()
    request = {"jsonrpc": "2.0", "id": rid, "method": "tools/call", "params": {"name": name, "arguments": args}}
    started = time.perf_counter()
    if (reason := _guard(name, args)) is not None:
        result = {"content": [{"type": "text", "text": reason}], "isError": True}
    else:
        if name == "add_paper":
            listed = await state["client"].call_tool("list_papers", {})
            if (listed.structured_content or {}).get("count", 0) >= MAX_PAPERS:
                arxiv = parse_arxiv_id(args["source"])
                present = any(p["paper_id"] == f"arxiv:{arxiv}" for p in listed.structured_content["papers"])
                if not present:
                    result = {"content": [{"type": "text", "text": f"The demo library is full ({MAX_PAPERS} papers)."}],
                              "isError": True}
                    return _pair(request, rid, result, started)
        res = await state["client"].call_tool(name, args)
        result = res.model_dump(mode="json", by_alias=True, exclude_none=True)
    return _pair(request, rid, result, started)


def _pair(request: dict, rid: int, result: dict, started: float) -> dict[str, Any]:
    return {"request": request, "response": {"jsonrpc": "2.0", "id": rid, "result": result},
            "ms": round((time.perf_counter() - started) * 1000)}


# --- rate limiting (per client IP, in memory) ---

_hits: dict[str, deque[float]] = defaultdict(deque)


def _limited(request: Request, per_minute: int) -> bool:
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    now, q = time.monotonic(), _hits[f"{request.url.path}:{ip}"]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= per_minute:
        return True
    q.append(now)
    return False


# --- routes ---


async def index(_request: Request):
    return FileResponse(HERE / "static" / "index.html")


async def tools(_request: Request):
    return JSONResponse({
        "server": {"name": "paper-librarian", "instructions": INSTRUCTIONS},
        "ask_enabled": bool(GROQ_KEY),
        "blocked": BLOCKED,
        "tools": [t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in state["tools"]],
    })


async def call(request: Request):
    if _limited(request, 40):
        return JSONResponse({"error": "Too many requests, wait a minute."}, status_code=429)
    body = await request.json()
    name, args = body.get("tool"), body.get("arguments") or {}
    if name not in {t.name for t in state["tools"]} or not isinstance(args, dict):
        return JSONResponse({"error": "Unknown tool or bad arguments"}, status_code=400)
    return JSONResponse(await _call(name, args))


SYSTEM = (
    "You are a research assistant connected to the paper-librarian MCP server. Answer only from what the "
    "tools return; never from memory. Use the tools (usually list_papers, then search) before answering. "
    "Keep answers concise and cite as (Title, Section, p.N). If the library lacks coverage, say so.\n\n"
    "Server instructions:\n" + INSTRUCTIONS
)


FINAL_NUDGE = "Stop calling tools. Answer now from the tool results above, noting any gaps."


def _shrink(value: Any, depth: int = 0) -> Any:
    """Trim long lists and strings so every field of a tool result survives the LLM's token budget
    (slicing the JSON text would drop whole trailing fields such as get_citations' cited_by)."""
    if isinstance(value, dict):
        return {k: _shrink(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        keep = 6 if depth else 10
        out = [_shrink(v, depth + 1) for v in value[:keep]]
        return out + [f"... {len(value) - keep} more"] if len(value) > keep else out
    if isinstance(value, str) and len(value) > 400:
        return value[:400] + "..."
    return value


def _groq(messages: list[dict], tool_defs: list[dict], final: bool = False) -> dict:
    for attempt in range(3):
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            json={"model": GROQ_MODEL, "temperature": 0.2, "max_completion_tokens": 1200,
                  **({"messages": [*messages, {"role": "user", "content": FINAL_NUDGE}]} if final else
                     {"messages": messages, "tools": tool_defs, "tool_choice": "auto"})},
            timeout=60,
        )
        if r.status_code != 429 or attempt == 2:
            break
        time.sleep(min(float(r.headers.get("retry-after", 5)), 15))
    if r.status_code != 200:
        raise RuntimeError(f"LLM error {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]


async def ask(request: Request):
    if not GROQ_KEY:
        return JSONResponse({"error": "Ask mode is not configured on this server."}, status_code=503)
    if _limited(request, 6):
        return JSONResponse({"error": "Rate limit: 6 questions per minute."}, status_code=429)
    question = str((await request.json()).get("question", "")).strip()[:1000]
    if not question:
        return JSONResponse({"error": "Empty question"}, status_code=400)

    tool_defs = [
        {"type": "function", "function": {"name": t.name, "description": t.description or "",
                                          "parameters": t.input_schema}}
        for t in state["tools"] if t.name not in BLOCKED and t.name not in ASK_HIDDEN
    ]
    messages: list[dict] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    trace: list[dict] = []
    try:
        for step in range(MAX_STEPS + 1):
            msg = await anyio.to_thread.run_sync(_groq, messages, tool_defs, step == MAX_STEPS)
            calls = msg.get("tool_calls") or []
            messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
            if not calls:
                trace.append({"type": "answer", "text": msg.get("content") or ""})
                break
            for tc in calls:
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                pair = await _call(fn["name"], args)
                trace.append({"type": "tool", "name": fn["name"], **pair})
                res = pair["response"]["result"]
                payload = res.get("structuredContent") or res.get("content")
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": json.dumps(_shrink(payload), separators=(",", ":"))[:TOOL_RESULT_CHARS]})
    except Exception as e:  # surface LLM/network failures in the UI instead of a bare 500
        trace.append({"type": "error", "text": str(e)})
    return JSONResponse({"model": GROQ_MODEL, "trace": trace})


async def health(_request: Request):
    return JSONResponse({"ok": True})


app = Starlette(
    routes=[
        Route("/", index),
        Route("/healthz", health),
        public_mcp_route,
        Route("/api/tools", tools),
        Route("/api/call", call, methods=["POST"]),
        Route("/api/ask", ask, methods=["POST"]),
        Mount("/static", StaticFiles(directory=HERE / "static"), name="static"),
    ],
    lifespan=lifespan,
)
