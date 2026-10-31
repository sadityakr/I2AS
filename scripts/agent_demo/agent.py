"""Make one MCP call through ``mcp_bridge.py`` and print the decoded answer.

Usage:
    python scripts/agent_demo/agent.py tools                  # list tool names
    python scripts/agent_demo/agent.py call <tool> '<json>'   # tools/call
    python scripts/agent_demo/agent.py read <uri>             # resources/read
    python scripts/agent_demo/agent.py say "<narration>"      # log the agent's reasoning
"""

from __future__ import annotations

import json
import socket
import sys


def send(method: str, params: dict, port: int = 8899) -> dict:
    with socket.create_connection(("127.0.0.1", port), timeout=180) as sock:
        sock.sendall((json.dumps({"method": method, "params": params}) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = sock.recv(1 << 20)
            if not chunk:
                break
            data += chunk
    return json.loads(data)


def main(argv: list[str]) -> int:
    verb = argv[0]
    if verb == "tools":
        answer = send("tools/list", {})
        for tool in answer["result"]["tools"]:
            print(tool["name"], "-", (tool.get("description") or "").split("\n")[0][:110])
        return 0
    if verb == "schema":
        answer = send("tools/list", {})
        for tool in answer["result"]["tools"]:
            if tool["name"] == argv[1]:
                print(json.dumps(tool, indent=1))
        return 0
    if verb == "say":
        send("narrate", {"text": argv[1]})
        return 0
    if verb == "read":
        answer = send("resources/read", {"uri": argv[1]})
    else:
        args = json.loads(argv[2]) if len(argv) > 2 else {}
        answer = send("tools/call", {"name": argv[1], "arguments": args})
    result = answer.get("result", answer)
    if isinstance(result, dict) and "content" in result:
        for item in result["content"]:
            text = item.get("text", "")
            try:
                print(json.dumps(json.loads(text), indent=1)[:6000])
            except (ValueError, TypeError):
                print(text[:6000])
        if result.get("isError"):
            return 1
    elif isinstance(result, dict) and "contents" in result:
        for item in result["contents"]:
            print(item.get("text", "")[:6000])
    else:
        print(json.dumps(result, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
