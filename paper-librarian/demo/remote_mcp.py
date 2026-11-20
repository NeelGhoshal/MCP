"""Public remote MCP endpoint: the real paper-librarian server over Streamable HTTP at /mcp.

Claude (claude.ai custom connectors, Claude Desktop, Claude Code) can connect to it by URL.
The endpoint has no auth, so it is read-only: tools that change the shared library are removed.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp
from starlette.routing import Route

from librarian.embed import Embedder
from librarian.server import create_server

# Anyone on the internet can call this endpoint; keep the shared library unchanged.
READ_ONLY_REMOVED = ("add_paper", "remove_paper", "add_note")


def build_public_mcp(db_path: Path, embedder: Embedder) -> tuple[MCPServer, Route]:
    """Return the server (enter `server.session_manager.run()` in the app lifespan) and its /mcp route."""
    server = create_server(db_path, embedder=embedder)
    for name in READ_ONLY_REMOVED:
        server.remove_tool(name)
    # Builds the session manager. Stateless: no session affinity needed, survives restarts cleanly.
    # host="0.0.0.0" skips the localhost-only Host-header check so the public hostname is accepted.
    server.streamable_http_app(stateless_http=True, host="0.0.0.0")
    # A plain Route (not Mount) so the URL is exactly /mcp, without a trailing-slash redirect.
    return server, Route("/mcp", endpoint=StreamableHTTPASGIApp(server.session_manager))
