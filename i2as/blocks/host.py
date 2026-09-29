"""The block host — the helper process a tier-2 block runs in.

``python -m i2as.blocks.host`` reads one JSON request per line on stdin and
answers one JSON line on stdout (``i2as.blocks.protocol``). The application
starts one host per block it needs, with a scrubbed environment and a
temporary working directory, hands a connector its one credential in the
``load`` request (never on the command line, never in the environment), and
kills the host when a call overruns its timeout. A renderer is loaded with the
network switched off: every socket it opens fails.

A block's own output on stdout would corrupt the protocol, so the real
stdout is kept for answers and ``sys.stdout`` is pointed at stderr before any
block is imported; whatever a block prints ends up in the host's stderr,
which the application logs.

This process imports only the standard library and ``i2as.blocks`` — no Qt,
no Station, no session layer — so a block running here can reach nothing of
the station by accident.
"""

from __future__ import annotations

import json
import socket
import sys
from typing import Any, TextIO

from i2as.blocks.discovery import KIND_CONNECTOR, KIND_RENDERER, connector_class, load_block_module
from i2as.blocks.protocol import (
    BlockProtocolError,
    dispatch_connector,
    dispatch_renderer,
    error_payload,
)


class NetworkDisabled(OSError):
    """Raised by every socket operation inside a renderer's host."""


#: The socket entry points a renderer's host switches off.
_SOCKET_NAMES: tuple[str, ...] = ("socket", "create_connection", "getaddrinfo")


def disable_network() -> dict[str, Any]:
    """Make every new socket fail — the renderer's host has no network.

    Returns:
        The originals, for ``restore_network()`` (the checker, which runs in
        the author's own process, restores them; the host never does).
    """

    def _refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise NetworkDisabled("a renderer has no network access")

    originals = {name: getattr(socket, name) for name in _SOCKET_NAMES}
    for name in _SOCKET_NAMES:
        setattr(socket, name, _refuse)
    return originals


def restore_network(originals: dict[str, Any]) -> None:
    """Undo ``disable_network()``."""
    for name, value in originals.items():
        setattr(socket, name, value)


class BlockHost:
    """Holds the one loaded block and answers requests for it."""

    def __init__(self) -> None:
        self.kind = ""
        self.block: Any = None

    def load(self, request: dict[str, Any]) -> dict[str, Any]:
        """Load the block a ``load`` request names.

        Raises:
            BlockProtocolError: Unknown kind, or already loaded.
        """
        if self.block is not None:
            raise BlockProtocolError("a block is already loaded in this host")
        kind = str(request.get("kind", ""))
        path = str(request.get("path", ""))
        if kind == KIND_RENDERER:
            disable_network()
            module = load_block_module(path)
            render = getattr(module, "render", None)
            if not callable(render):
                raise BlockProtocolError("the renderer defines no render(context) function")
            self.kind, self.block = kind, render
            return {"id": str(getattr(module, "NAME", "")), "kind": kind}
        if kind == KIND_CONNECTOR:
            module = load_block_module(path)
            cls = connector_class(module)
            settings = request.get("settings")
            self.block = cls(
                dict(settings) if isinstance(settings, dict) else {},
                str(request.get("credential") or ""),
            )
            self.kind = kind
            return {"id": cls.backend, "kind": kind, "contract_version": cls.contract_version}
        raise BlockProtocolError(f"unknown block kind {kind!r}")

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Answer one request; never raises."""
        request_id = request.get("id")
        try:
            op = request.get("op")
            if op == "load":
                result = self.load(request)
            elif op == "call":
                if self.block is None:
                    raise BlockProtocolError("no block is loaded")
                method = str(request.get("method", ""))
                args = request.get("args")
                args = dict(args) if isinstance(args, dict) else {}
                if self.kind == KIND_RENDERER:
                    result = dispatch_renderer(self.block, method, args)
                else:
                    result = dispatch_connector(self.block, method, args)
            elif op == "ping":
                result = "pong"
            else:
                raise BlockProtocolError(f"unknown op {op!r}")
        except Exception as exc:  # noqa: BLE001 - every failure becomes an answer
            return {"id": request_id, "ok": False, "error": error_payload(exc)}
        return {"id": request_id, "ok": True, "result": result}


def serve(stdin: TextIO, stdout: TextIO) -> int:
    """Answer requests until stdin closes.

    Args:
        stdin: Where requests arrive.
        stdout: Where answers go (the REAL stdout).

    Returns:
        The exit code.
    """
    host = BlockHost()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            answer: dict[str, Any] = {"id": None, "ok": False, "error": {"type": "BlockError", "message": "not JSON"}}
        else:
            answer = host.handle(request if isinstance(request, dict) else {})
        stdout.write(json.dumps(answer, ensure_ascii=True) + "\n")
        stdout.flush()
    return 0


def main() -> int:
    """Run the host on the process's stdin/stdout."""
    protocol_out = sys.stdout
    sys.stdout = sys.stderr  # a block's print() must never corrupt the protocol
    return serve(sys.stdin, protocol_out)


if __name__ == "__main__":
    raise SystemExit(main())
