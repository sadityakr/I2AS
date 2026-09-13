"""Send one line of JSON to the mcp_instance relay and print the response.

Usage:
    python scripts/mcp_send.py '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'
"""

from __future__ import annotations

import argparse
import json
import socket
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="One JSON-RPC request object, as a JSON string")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    request = json.loads(args.request)
    with socket.create_connection(("127.0.0.1", args.port), timeout=35) as sock:
        sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if chunk.endswith(b"\n"):
                break
    print(b"".join(chunks).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
