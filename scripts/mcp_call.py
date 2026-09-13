"""Call one MCP tool against the running mcp_instance relay and print the
decoded result only (status-snapshot notifications that arrive meanwhile are
suppressed, not lost — they are still real traffic, just noise for a human
watching one call at a time).

Usage:
    python scripts/mcp_call.py tools/call list_procedures '{}'
    python scripts/mcp_call.py tools/call describe_procedure '{"procedure":"FieldSweep"}'
    python scripts/mcp_call.py resources/read i2as://status
"""

from __future__ import annotations

import argparse
import itertools
import json
import socket
import sys

_ids = itertools.count(1000)


def send(request: dict, port: int) -> list[dict]:
    with socket.create_connection(("127.0.0.1", port), timeout=35) as sock:
        sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if chunk.endswith(b"\n"):
                break
    return json.loads(b"".join(chunks).decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", help="tools/call, resources/read, ...")
    parser.add_argument("name", help="tool name or resource uri")
    parser.add_argument("args", nargs="?", default="{}", help="JSON arguments/params")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--raw", action="store_true", help="print the full JSON-RPC traffic")
    args = parser.parse_args()

    request_id = next(_ids)
    if args.method == "tools/call":
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": args.name, "arguments": json.loads(args.args)},
        }
    elif args.method == "resources/read":
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "resources/read",
            "params": {"uri": args.name},
        }
    else:
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": args.method,
            "params": json.loads(args.args),
        }

    messages = send(request, args.port)

    if args.raw:
        print(json.dumps(messages, indent=2))
        return 0

    for message in messages:
        if message.get("id") == request_id:
            result = message.get("result") or {}
            content = result.get("content") or []
            if content:
                for item in content:
                    text = item.get("text", "")
                    try:
                        print(json.dumps(json.loads(text), indent=2))
                    except ValueError:
                        print(text)
            else:
                print(json.dumps(result, indent=2))
            if result.get("isError"):
                print("(isError=true)", file=sys.stderr)
    events = [
        m for m in messages
        if m.get("method") == "notifications/message" and m.get("params", {}).get("logger") != "i2as.gateway"
    ]
    status_events = [
        m for m in messages
        if m.get("method") == "notifications/message"
        and (m.get("params", {}).get("data", {}).get("summary") not in ("StatusSnapshot",))
    ]
    for event in status_events:
        print("--- notification ---", file=sys.stderr)
        print(json.dumps(event["params"]["data"], indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
