"""A long-lived MCP client for a running I2AS app, driven one call at a time.

Starts the real adapter (``python -m i2as.mcp``) against the app's gateway
descriptor, keeps the MCP session open, and serves a tiny line protocol on
127.0.0.1 so an agent working from a shell can make one tool call per
command (see ``agent.py``). Every request and its answer is appended to a
JSONL transcript with wall-clock timestamps — the record the demo video's
agent overlay is rendered from.

Usage:
    python scripts/agent_demo/mcp_bridge.py --transcript /path/transcript.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


class Adapter:
    """The adapter subprocess and every message it has sent."""

    def __init__(self, role: str, actor_id: str) -> None:
        self.messages: dict[int, dict[str, Any]] = {}
        self.notifications: list[dict[str, Any]] = []
        self._next = 0
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "i2as.mcp", "--role", role, "--actor-id", actor_id,
             "--framing", "shim", "--timeout", "60", "--log-level", "WARNING"],
            cwd=str(REPO_ROOT), env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            if not raw.strip():
                continue
            msg = json.loads(raw)
            with self._cv:
                if "id" in msg and msg["id"] is not None:
                    self.messages[msg["id"]] = msg
                else:
                    self.notifications.append(msg)
                    del self.notifications[:-200]
                self._cv.notify_all()

    def request(self, method: str, params: dict[str, Any] | None, timeout: float = 120.0) -> dict[str, Any]:
        with self._cv:
            self._next += 1
            rid = self._next
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps({"jsonrpc": "2.0", "id": rid, "method": method,
                                           "params": params or {}}) + "\n").encode())
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        with self._cv:
            while rid not in self.messages:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(method)
                self._cv.wait(left)
            return self.messages.pop(rid)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": method,
                                           "params": params or {}}) + "\n").encode())
        self.proc.stdin.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--role", default="session")
    parser.add_argument("--actor-id", default="claude-agent")
    parser.add_argument("--transcript", required=True)
    args = parser.parse_args()

    adapter = Adapter(args.role, args.actor_id)
    init = adapter.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                          "clientInfo": {"name": "claude-code", "version": "1"}})
    adapter.notify("notifications/initialized")
    transcript = Path(args.transcript)
    transcript.parent.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()

    def log(entry: dict[str, Any]) -> None:
        with log_lock, transcript.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": time.time(), **entry}) + "\n")

    log({"kind": "connected", "server": init.get("result", {}).get("serverInfo")})

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            line = self.rfile.readline()
            req = json.loads(line)
            if req.get("method") == "narrate":
                log({"kind": "narration", "text": req["params"]["text"]})
                answer: dict[str, Any] = {"result": "ok"}
            else:
                log({"kind": "call", "method": req["method"], "params": req.get("params")})
                try:
                    answer = adapter.request(req["method"], req.get("params"))
                except Exception as exc:  # noqa: BLE001 — reported to the caller
                    answer = {"error": repr(exc)}
                log({"kind": "result", "method": req["method"], "params": req.get("params"),
                     "answer": answer})
            self.wfile.write((json.dumps(answer) + "\n").encode())

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"bridge ready on {args.port}", flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
