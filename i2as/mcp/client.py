"""Finding the running app, and speaking its wire — stdlib only.

Two jobs, both deliberately small:

* **Find it.** The **Gateway server** publishes a descriptor beside its
  socket — the socket name, the owning pid, the wire's schema version and
  the per-launch token — so an adapter started by an editor with no
  arguments still knows where the app is and how to be admitted by it.
* **Speak to it.** ``GatewayClient`` is a newline-delimited JSON-RPC 2.0
  client: one ``hello`` at connect, one request at a time after that, and
  the server's notifications queued as they arrive so a caller reads them
  when it is ready rather than being interrupted.

Nothing here imports anything else from I2AS: this file runs in a
process that must not be able to reach an instrument (import contract C21),
and the descriptor's location is therefore re-derived from the two
documented environment locations rather than read from the module that owns
that rule for the app itself (``i2as.core.paths``). That duplication is
the price of the isolation and is deliberate; the two are kept in step by
``I2AS_LOG_DIR``, which is what a deployment actually sets.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "DESCRIPTOR_FILENAME",
    "DEFAULT_TIMEOUT_S",
    "SUPPORTED_SCHEMA",
    "GatewayError",
    "GatewayClient",
    "default_descriptor_path",
    "read_descriptor",
]

#: The descriptor the **Gateway server** writes while it is listening.
DESCRIPTOR_FILENAME = "gateway.json"

#: The wire schema this client speaks. The **Gateway server** stamps its own
#: into every descriptor, and a mismatch is refused rather than guessed at: a
#: client speaking a wire the app does not would be answered with errors whose
#: cause is invisible. Named here rather than imported for the same reason the
#: descriptor's location is — this process may not import the session layer —
#: and the two are kept in step by the descriptor itself, which says so the
#: moment they are not.
SUPPORTED_SCHEMA = 1

#: How long a single request may wait for its answer, in seconds. A tool
#: call is answered on the app's event loop, which is also the tick's, so a
#: tick that is busy inside a long ``measure()`` delays an answer without
#: losing it.
DEFAULT_TIMEOUT_S = 120.0

#: How much is read from the socket at once.
_CHUNK = 65536


class GatewayError(RuntimeError):
    """The gateway refused a request, or the connection failed.

    Attributes:
        code: The JSON-RPC error code the server sent, or ``None`` when the
            failure was in the transport rather than in an answer.
    """

    def __init__(self, message: str, code: int | None = None) -> None:
        """Build the error.

        Args:
            message: Operator-facing explanation.
            code: The JSON-RPC code, when there was one.
        """
        super().__init__(message)
        self.code = code


def default_descriptor_path() -> Path:
    """Resolve where the running app's descriptor is expected to be.

    Precedence, mirroring the app's own log-directory rule
    (``i2as.core.paths.log_directory()``):

    1. ``I2AS_GATEWAY_DESCRIPTOR``, the file itself.
    2. ``I2AS_LOG_DIR/gateway.json``.
    3. ``gateway.json`` under the per-user state root's ``logs/`` —
       ``%LOCALAPPDATA%\\I2AS\\logs`` on Windows, ``~/.local/state/i2as/logs``
       (or its ``XDG_STATE_HOME`` form) elsewhere and as the Windows fallback.

    Returns:
        The path (not guaranteed to exist).
    """
    explicit = os.environ.get("I2AS_GATEWAY_DESCRIPTOR")
    if explicit:
        return Path(explicit)
    log_dir = os.environ.get("I2AS_LOG_DIR")
    if log_dir:
        return Path(log_dir) / DESCRIPTOR_FILENAME
    # A deliberate copy of i2as.core.paths.user_state_dir(): this module
    # may import only the stdlib and i2as.core.events (import contract
    # C21 — the adapter must not be able to reach an instrument), so it
    # cannot import paths.py. Keep the two in step; test_mcp_adapter pins
    # them against each other.
    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / "I2AS" / "logs" / DESCRIPTOR_FILENAME
    xdg_state = os.environ.get("XDG_STATE_HOME")
    state_root = Path(xdg_state) if xdg_state else Path.home() / ".local" / "state"
    return state_root / "i2as" / "logs" / DESCRIPTOR_FILENAME


def read_descriptor(path: Path | str) -> dict[str, Any]:
    """Read and check one descriptor file.

    Args:
        path: The descriptor to read.

    Returns:
        Its contents: ``schema``, ``socket``, ``pid``, ``token`` and
        ``max_role``.

    Raises:
        GatewayError: If the file is missing, unreadable, not JSON, does not
            carry a socket name and a token, or was written by an app
            speaking a different wire schema. The message names the path,
            because "the app is not running" and "the app is somewhere else"
            look identical without it.
    """
    descriptor_file = Path(path)
    try:
        payload = json.loads(descriptor_file.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise GatewayError(
            f"no gateway descriptor at {descriptor_file}: the app is not "
            f"running, or its monitor.yaml does not set gateway_server: true"
        ) from error
    except (OSError, ValueError) as error:
        raise GatewayError(f"unreadable gateway descriptor {descriptor_file}: {error}") from error
    if not isinstance(payload, Mapping):
        raise GatewayError(f"{descriptor_file} is not a JSON object")
    if not payload.get("socket") or not payload.get("token"):
        raise GatewayError(f"{descriptor_file} carries no socket name and token")
    schema = payload.get("schema")
    if schema != SUPPORTED_SCHEMA:
        raise GatewayError(
            f"{descriptor_file} declares gateway wire schema {schema!r}; this "
            f"adapter speaks {SUPPORTED_SCHEMA}. Run the adapter shipped with "
            f"the application that is running."
        )
    return dict(payload)


class _UnixTransport:
    """A stream socket to a listening ``QLocalServer`` on POSIX."""

    def __init__(self, name: str, timeout: float) -> None:
        """Connect to the socket file.

        Args:
            name: The socket's path.
            timeout: Per-read timeout, in seconds.

        Raises:
            GatewayError: If the socket cannot be reached.
        """
        self._timeout = timeout
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(timeout)
        try:
            self._socket.connect(name)
        except OSError as error:
            raise GatewayError(f"cannot connect to {name}: {error}") from error

    def fileno(self) -> int | None:
        """Return the descriptor a selector can wait on."""
        return self._socket.fileno()

    def send(self, data: bytes) -> None:
        """Write bytes, all of them.

        Args:
            data: What to write.

        Raises:
            GatewayError: If the write fails.
        """
        try:
            self._socket.sendall(data)
        except OSError as error:
            raise GatewayError(f"gateway write failed: {error}") from error

    def recv(self, *, blocking: bool) -> bytes:
        """Read whatever is available.

        Args:
            blocking: Wait up to the timeout when ``True``; return ``b""``
                immediately when nothing is buffered and ``False``.

        Returns:
            The bytes read; ``b""`` when the peer closed, or when nothing
            waited and *blocking* is ``False``.

        Raises:
            GatewayError: If the read times out or fails.
        """
        self._socket.settimeout(self._timeout if blocking else 0.0)
        try:
            return self._socket.recv(_CHUNK)
        except (BlockingIOError, TimeoutError) as error:
            if blocking:
                raise GatewayError(
                    f"the gateway did not answer within {self._timeout:g} s"
                ) from error
            return b""
        except OSError as error:
            raise GatewayError(f"gateway read failed: {error}") from error

    def close(self) -> None:
        """Close the socket, ignoring a socket that is already gone."""
        try:
            self._socket.close()
        except OSError:
            pass


class _PipeTransport:
    """A named pipe to a listening ``QLocalServer`` on Windows.

    A local socket is a named pipe there, not a file in the filesystem, so
    it is opened rather than connected to and cannot be polled by a
    selector. The consequence is one the adapter handles rather than hides:
    ``fileno()`` returns ``None``, so notifications are delivered on the
    next request instead of the moment they arrive.
    """

    def __init__(self, name: str, timeout: float) -> None:
        """Open the pipe.

        Args:
            name: The pipe's name, with or without the ``\\\\.\\pipe\\``
                prefix.
            timeout: Unused; a pipe read blocks.

        Raises:
            GatewayError: If the pipe cannot be opened.
        """
        del timeout
        path = name if name.startswith("\\\\") else rf"\\.\pipe\{name}"
        try:
            self._handle = open(path, "r+b", buffering=0)  # noqa: SIM115
        except OSError as error:
            raise GatewayError(f"cannot open {path}: {error}") from error

    def fileno(self) -> int | None:
        """Return ``None``: a pipe is not selectable here."""
        return None

    def send(self, data: bytes) -> None:
        """Write bytes and flush them.

        Args:
            data: What to write.

        Raises:
            GatewayError: If the write fails.
        """
        try:
            self._handle.write(data)
            self._handle.flush()
        except OSError as error:
            raise GatewayError(f"gateway write failed: {error}") from error

    def recv(self, *, blocking: bool) -> bytes:
        """Read one chunk, blocking.

        Args:
            blocking: A non-blocking read is not available on a pipe, so a
                ``False`` here reads nothing rather than blocking anyway.

        Returns:
            The bytes read, or ``b""``.

        Raises:
            GatewayError: If the read fails.
        """
        if not blocking:
            return b""
        try:
            return self._handle.read(1)
        except OSError as error:
            raise GatewayError(f"gateway read failed: {error}") from error

    def close(self) -> None:
        """Close the pipe, ignoring one that is already gone."""
        try:
            self._handle.close()
        except OSError:
            pass


class GatewayClient:
    """One connection to a running app's **Gateway server**.

    Attributes:
        role: The role this connection asked for.
        actor_id: The identity it declared.
    """

    def __init__(
        self,
        socket_name: str,
        token: str,
        *,
        role: str = "observer",
        actor_id: str = "mcp",
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """Prepare a connection without opening it.

        Args:
            socket_name: The socket name from the descriptor.
            token: The secret from the descriptor.
            role: The role to declare at the handshake.
            actor_id: The identity to declare at the handshake — it lands on
                every verdict, run record and feed entry this connection
                causes, so it should name the session, not the tool.
            timeout: Per-read timeout in seconds.
        """
        self._socket_name = socket_name
        self._token = token
        self.role = role
        self.actor_id = actor_id
        self._timeout = timeout
        self._transport: _UnixTransport | _PipeTransport | None = None
        self._buffer = bytearray()
        self._notifications: list[dict[str, Any]] = []
        self._next_id = 0
        self.hello: dict[str, Any] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────

    def connect(self) -> dict[str, Any]:
        """Open the connection and perform the handshake.

        Returns:
            The ``hello`` result — the granted role, the actor id, the wire
            schema version and the deployment's ceiling.

        Raises:
            GatewayError: If the socket cannot be reached or the handshake
                is refused (a bad token, an unknown role, or a role above
                the app's ceiling).
        """
        transport_class = _PipeTransport if os.name == "nt" else _UnixTransport
        self._transport = transport_class(self._socket_name, self._timeout)
        self.hello = self.call(
            "hello",
            {"role": self.role, "actor_id": self.actor_id, "token": self._token},
        )
        return self.hello

    def close(self) -> None:
        """Close the connection. Idempotent."""
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def fileno(self) -> int | None:
        """Return the descriptor a selector may wait on, if there is one.

        Returns:
            The socket's file descriptor on POSIX; ``None`` on a platform
            whose local socket cannot be polled alongside stdin.
        """
        return self._transport.fileno() if self._transport is not None else None

    # ── Requests ──────────────────────────────────────────────────────

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Send one request and wait for the answer that carries its id.

        Notifications that arrive while waiting are queued, not dropped and
        not delivered here: the caller collects them with
        ``take_notifications()`` when it is ready to write them out.

        Args:
            method: The wire method.
            params: Its parameters.

        Returns:
            The response's ``result`` object.

        Raises:
            GatewayError: If the connection is closed, the server answers
                with an error, or the peer disconnects mid-request.
        """
        if self._transport is None:
            raise GatewayError("not connected to the gateway")
        self._next_id += 1
        request_id = self._next_id
        self._transport.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": dict(params or {}),
                }
            ).encode("utf-8")
            + b"\n"
        )
        while True:
            message = self._next_message(blocking=True)
            if message is None:
                raise GatewayError(
                    f"the gateway closed the connection while answering {method!r}"
                )
            if message.get("id") != request_id:
                self._queue(message)
                continue
            error = message.get("error")
            if isinstance(error, Mapping):
                raise GatewayError(
                    str(error.get("message", "gateway error")),
                    code=error.get("code"),
                )
            result = message.get("result")
            return dict(result) if isinstance(result, Mapping) else {}

    def receive_available(self) -> None:
        """Read whatever the socket has ready and queue it, without blocking.

        Called by a serving loop that woke because the socket became
        readable, so a notification is written out the moment it arrives
        rather than waiting for the next request.
        """
        if self._transport is None:
            return
        while True:
            message = self._next_message(blocking=False)
            if message is None:
                return
            self._queue(message)

    def take_notifications(self) -> list[dict[str, Any]]:
        """Return every queued notification and forget them.

        Returns:
            The notifications, oldest first.
        """
        queued, self._notifications = self._notifications, []
        return queued

    # ── Framing ───────────────────────────────────────────────────────

    def _queue(self, message: Mapping[str, Any]) -> None:
        """Keep a server-initiated message for the caller to collect.

        Args:
            message: The message read off the wire. A response to a request
                nobody is waiting for is dropped rather than queued — it can
                only be a stale answer to an abandoned call.
        """
        if "id" not in message and isinstance(message.get("method"), str):
            self._notifications.append(dict(message))

    def _next_message(self, *, blocking: bool) -> dict[str, Any] | None:
        """Return the next whole frame, reading more bytes if needed.

        Args:
            blocking: Whether to wait for bytes that have not arrived.

        Returns:
            The decoded message, or ``None`` when the peer closed (blocking)
            or nothing is ready (non-blocking).

        Raises:
            GatewayError: If the frame is not JSON, which means the two ends
                disagree about the wire and retrying cannot help.
        """
        assert self._transport is not None
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                frame = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                if not frame.strip():
                    continue
                try:
                    decoded = json.loads(frame.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as error:
                    raise GatewayError(f"malformed frame from the gateway: {error}") from error
                if isinstance(decoded, Mapping):
                    return dict(decoded)
                continue
            chunk = self._transport.recv(blocking=blocking)
            if not chunk:
                return None
            self._buffer.extend(chunk)
