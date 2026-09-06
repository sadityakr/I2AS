"""The in-repo stdio backend: newline-delimited JSON-RPC, stdlib only.

MCP's stdio transport is one JSON-RPC message per line on stdin and stdout,
UTF-8, no embedded newlines — a framing small enough that depending on a
package for it would make the tests depend on that package too. So this
module is the backend that always exists: it reads frames, hands each to
``McpAdapter``, writes back what the adapter returns, and writes out the
app's notifications as they arrive.

**One loop, two readable things.** The session's stdin and the **Gateway
server**'s socket are waited on together with ``selectors``, so a state
change reaches the session the moment the app emits it rather than the next
time the session happens to ask a question. Where a local socket cannot be
selected on — a named pipe on Windows — the loop wakes on a timer instead
and delivers the same notifications a little later; nothing else differs.

**stdout carries protocol and nothing else.** Every diagnostic goes to
stderr, which is where a client shows it. A stray ``print`` here would
corrupt the session's message stream, which is the reason the logging
standard's "never print" rule is not merely a preference in this file.
"""

from __future__ import annotations

import json
import logging
import os
import selectors
import sys
from collections.abc import Mapping
from typing import Any, BinaryIO

from i2as.mcp.adapter import McpAdapter

logger = logging.getLogger(__name__)

__all__ = ["MAX_FRAME_BYTES", "serve"]

#: The largest single stdin frame this backend will accept, in bytes. The
#: peer is the process that launched this one, not a network, so the cap is
#: a guard against a runaway writer rather than against an attacker.
MAX_FRAME_BYTES = 8 << 20

#: How long a wakeup waits when the gateway socket cannot be selected on.
_POLL_INTERVAL_S = 0.25

#: How much is read from stdin at once.
_CHUNK = 65536

PARSE_ERROR = -32700


def serve(
    adapter: McpAdapter,
    *,
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
) -> None:
    """Serve one MCP session over stdio until the client closes stdin.

    Args:
        adapter: The translation this loop frames for. It is expected to be
            open already (``McpAdapter.open()``), so the session's first
            request is answered without a connect in the middle of it.
        stdin: The byte stream to read frames from; defaults to the
            process's own.
        stdout: The byte stream to write frames to; defaults to the
            process's own.

    Raises:
        OSError: If stdin or stdout fails in a way that leaves no session to
            serve; a failure to write one message is logged and the loop
            continues.
    """
    source = stdin if stdin is not None else sys.stdin.buffer
    sink = stdout if stdout is not None else sys.stdout.buffer
    buffer = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(source.fileno(), selectors.EVENT_READ, "stdin")
    gateway_fd = adapter.client.fileno()
    if gateway_fd is not None:
        selector.register(gateway_fd, selectors.EVENT_READ, "gateway")
    timeout = None if gateway_fd is not None else _POLL_INTERVAL_S

    logger.info("MCP adapter serving over stdio")
    try:
        while True:
            for key, _ in selector.select(timeout):
                if key.data == "gateway":
                    _flush_notifications(adapter, sink)
                    continue
                chunk = os.read(key.fd, _CHUNK)
                if not chunk:
                    logger.info("MCP adapter: the client closed stdin")
                    return
                buffer.extend(chunk)
            if timeout is not None:
                # No selectable gateway: the wakeup was the timer, so ask
                # the connection whether anything arrived.
                _flush_notifications(adapter, sink)
            if not _consume(adapter, buffer, sink):
                return
    finally:
        selector.close()


def _consume(adapter: McpAdapter, buffer: bytearray, sink: BinaryIO) -> bool:
    """Answer every whole frame in *buffer*.

    Args:
        adapter: The translation.
        buffer: Bytes read but not yet answered; whole frames are removed.
        sink: Where to write the answers.

    Returns:
        ``True`` to keep serving; ``False`` when a frame exceeded the cap
        and the session cannot be trusted to be in step any more.
    """
    while True:
        newline = buffer.find(b"\n")
        if newline < 0:
            if len(buffer) > MAX_FRAME_BYTES:
                logger.error("MCP adapter: a frame exceeded %d bytes", MAX_FRAME_BYTES)
                return False
            return True
        frame = bytes(buffer[:newline])
        del buffer[: newline + 1]
        if not frame.strip():
            continue
        _answer(adapter, frame, sink)


def _answer(adapter: McpAdapter, frame: bytes, sink: BinaryIO) -> None:
    """Parse one frame, answer it, and write out anything it caused.

    Args:
        adapter: The translation.
        frame: One line of stdin, newline already stripped.
        sink: Where to write.
    """
    try:
        request = json.loads(frame.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        _write(sink, _parse_error(f"malformed JSON: {error}"))
        return
    if not isinstance(request, Mapping):
        _write(sink, _parse_error("a JSON-RPC message is an object"))
        return
    response = adapter.handle(request)
    if response is not None:
        _write(sink, response)
    _flush_notifications(adapter, sink)


def _flush_notifications(adapter: McpAdapter, sink: BinaryIO) -> None:
    """Write out every gateway notification waiting, translated.

    Args:
        adapter: The translation.
        sink: Where to write.
    """
    try:
        notifications = adapter.drain_notifications()
    except Exception:  # noqa: BLE001 — a lost event must not end the session
        logger.exception("MCP adapter could not read the gateway's notifications")
        return
    for notification in notifications:
        _write(sink, notification)


def _write(sink: BinaryIO, message: Mapping[str, Any]) -> None:
    """Write one JSON-RPC message and its newline.

    Args:
        sink: Where to write.
        message: The JSON-safe message.
    """
    try:
        line = json.dumps(message, default=str).encode("utf-8") + b"\n"
        sink.write(line)
        sink.flush()
    except (OSError, ValueError):
        logger.exception("MCP adapter could not write a message")


def _parse_error(message: str) -> dict[str, Any]:
    """Build the one error a frame that could not be parsed is answered with.

    Args:
        message: The explanation.

    Returns:
        A JSON-RPC error response with a null id, which is what the
        specification prescribes when the id could not be read.
    """
    return {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": PARSE_ERROR, "message": message},
    }
