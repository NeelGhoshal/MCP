"""MCP server and tool definitions."""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("paper-librarian")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
