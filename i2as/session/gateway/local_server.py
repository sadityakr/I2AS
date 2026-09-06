"""The **Gateway server**: the same ``submit()``, reached over a local socket.

**The transport standard.** An agent that runs in its own process must see
exactly the system an in-process client sees, and must be seen doing exactly
what an in-process client is seen doing. So this module adds a *transport*
and nothing else: every connection owns ONE ``Gateway``, built with the role
and actor id the connection declared at its handshake, and every request is
routed through that object's own ``call_tool()`` / mirror reads. No method
here reaches past a ``Gateway`` to the engine, no method here decides
authority, and no method here knows what an instrument is.

Three properties are the design:

* **No thread of its own.** ``QLocalServer`` and ``QLocalSocket`` are
  ordinary ``QObject``s on the GUI thread's event loop, and every command
  they carry reaches the engine the way every other client's does — posted
  through the **Orchestrator proxy** onto the instrument thread, answered by
  a verdict that comes back queued. The single hardware thread standard
  holds: this file adds no thread and touches no instrument. Nothing in it
  blocks, sleeps or waits.
* **Nothing raises into the loop.** A partial read, an oversized frame,
  malformed JSON, an unknown method, a bad argument and an unexpected
  failure are all answered as JSON-RPC errors on the connection that caused
  them. A client's mistake must never disturb a running measurement.
* **The handshake is the identity.** A connection is nothing until it says
  ``hello``; from then on it is one agent with one role, and the role it may
  claim is bounded by the deployment's ceiling (``role_within_ceiling()``).

The wire
--------

Newline-delimited `JSON-RPC 2.0 <https://www.jsonrpc.org/specification>`_,
UTF-8, one message per line, no embedded newlines. Requests carry an ``id``
and are answered; server-initiated messages are notifications and carry
none.

======================= =================================================
Method                  Answers
======================= =================================================
``hello``               ``{role, actor_id, schema, tools}`` — required first.
``tools/list``          ``{tools: [...]}``, the gateway's ``tool_schemas()``.
``tools/call``          The gateway's ``call_tool()`` answer, verbatim.
``status``              ``{status, state, attended, agent_gate}``.
``station``             ``{station: <StationInfo>}``.
``events/subscribe``    ``{subscribed: true}``; then notifications follow.
======================= =================================================

Two notification methods travel the other way once a connection has
subscribed: ``event``, carrying one ``StateChange`` or ``StatusSnapshot`` as
``{"event": {...}}``, and ``verdict``, carrying one ``Verdict`` as
``{"verdict": {...}}``. They are the engine's own messages, rendered by the
contract's own ``to_json()`` — this module defines no vocabulary of its own.

The token
---------

A per-launch random secret, written to the descriptor file
(``gateway.json``, beside the socket) with owner-only permissions and
required in ``hello``. It is not a security boundary against a hostile local
user — the socket's own owner-only permissions are that — it is what keeps
an unrelated process that stumbles onto the socket from acting as an agent.
The descriptor also carries the socket name, the owning pid and the schema
version, so a client finds the running app without being told where it is.

See ``README.md`` for the folder standard and ``GLOSSARY.md`` for the
**Gateway server** and **MCP adapter** vocabulary.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from PyQt6.QtNetwork import QLocalServer, QLocalSocket

from i2as.core.events import StateChange, StatusSnapshot, Verdict
from i2as.core.paths import log_directory
from i2as.session.agent_feed import AgentFeed
from i2as.session.gateway.gateway import (
    EngineClient,
    Gateway,
    event_stream,
    verdict_stream,
)
from i2as.session.gateway.roles import Role, role_within_ceiling
from i2as.session.gateway.tools import ToolContext

logger = logging.getLogger(__name__)

__all__ = [
    "SCHEMA_VERSION",
    "MAX_FRAME_BYTES",
    "DESCRIPTOR_FILENAME",
    "SOCKET_FILENAME",
    "GatewayServer",
    "descriptor_path",
    "default_socket_name",
]

#: Version of the descriptor file and of the wire vocabulary above. A client
#: that reads a schema it does not know refuses to connect rather than
#: guessing at the shape.
SCHEMA_VERSION = 1

#: The largest single frame the server will accept, in bytes. A client that
#: exceeds it gets one error and is disconnected: an unbounded buffer on the
#: GUI thread is a denial of service against the measurement, not merely a
#: bad request.
MAX_FRAME_BYTES = 1 << 20

#: The descriptor file, written beside the socket in the log directory.
DESCRIPTOR_FILENAME = "gateway.json"

#: The socket file's name on the platforms that give a local socket a path.
SOCKET_FILENAME = "gateway.sock"

# ── JSON-RPC 2.0's own error codes ────────────────────────────────────
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# ── This server's codes ───────────────────────────────────────────────
# Deliberately outside JSON-RPC's reserved -32768..-32000 band, which
# belongs to the protocol and to the frameworks that implement it.
BAD_TOKEN = -31001
ROLE_REFUSED = -31002
NOT_AUTHENTICATED = -31003
ALREADY_AUTHENTICATED = -31004
FRAME_TOO_LARGE = -31005


def descriptor_path() -> Path:
    """Return where the descriptor file lives on this installation.

    Returns:
        ``paths.log_directory()/gateway.json`` — the same directory the
        operational-status log is written to, because it is the one
        machine-local location every I2AS process already agrees on.
    """
    return log_directory() / DESCRIPTOR_FILENAME


def default_socket_name() -> str:
    """Return the local-socket name this installation listens on.

    Returns:
        A full path inside the log directory on the platforms where a local
        socket is a file, and a named-pipe name on Windows, where it is not.
    """
    if os.name == "nt":
        return f"i2as-gateway-{os.getpid()}"
    return str(log_directory() / SOCKET_FILENAME)


class _Connection:
    """One accepted client: its socket, its read buffer, its identity.

    Attributes:
        socket: The accepted ``QLocalSocket``.
        buffer: Bytes read but not yet terminated by a newline.
        gateway: The one ``Gateway`` this connection acts through, built at
            ``hello`` and ``None`` before it.
        subscribed: Whether the connection asked for event notifications.
    """

    def __init__(self, socket: QLocalSocket) -> None:
        """Wrap one accepted socket.

        Args:
            socket: The socket the server accepted.
        """
        self.socket = socket
        self.buffer = bytearray()
        self.gateway: Gateway | None = None
        self.subscribed = False


class GatewayServer(QLocalServer):
    """A local-socket front door onto one ``Gateway`` per connection.

    Lives on the GUI thread's event loop, as a CLIENT of the engine rather
    than beside it: under the single hardware thread standard the engine is on
    the instrument thread, so what this server holds — and hands every
    ``Gateway`` it builds — is the **Orchestrator proxy**, whose commands are
    posted across and whose two contract streams arrive queued. It holds no
    Station, no Orchestrator and no instrument: the only object it builds is a
    ``Gateway``, and the only thing it does with one is call the methods an
    in-process client calls.

    Attributes:
        token: The per-launch secret a client must present in ``hello``.
        max_role: The most authority this deployment hands out.
    """

    def __init__(
        self,
        engine: EngineClient,
        *,
        socket_name: str | None = None,
        descriptor: Path | str | None = None,
        token: str | None = None,
        max_role: Role | str = Role.OBSERVER,
        station_info: Any | None = None,
        tool_context: ToolContext | None = None,
        feed: AgentFeed | Callable[[], AgentFeed | None] | None = None,
        parent: Any | None = None,
    ) -> None:
        """Build the server without listening yet.

        Args:
            engine: The engine client every connection's ``Gateway`` is
                attached to — the **Orchestrator proxy** in the running app,
                because this server runs on the GUI thread and the engine may
                be on the instrument thread; anything satisfying
                ``EngineClient`` in a test.
            socket_name: The local-socket name to listen on; defaults to
                ``default_socket_name()``.
            descriptor: Where to write the descriptor file; defaults to
                ``descriptor_path()``.
            token: The secret to require in ``hello``; a fresh
                cryptographically random one is generated when omitted,
                which is the production path.
            max_role: The deployment's ceiling — a connection asking for a
                role that grants more is refused at the handshake.
            station_info: The station's declaration snapshot, or a
                zero-argument callable returning it, handed to every
                ``Gateway`` built here.
            tool_context: The collaborators the session tools read through,
                handed to every ``Gateway`` built here.
            feed: The experiment's **Agent feed**, or a zero-argument
                callable returning the open experiment's, so a connection
                opened later records into the experiment that is open then.
                ``None`` records nothing.
            parent: Optional Qt parent.

        Raises:
            ValueError: If *max_role* is not a known ``Role``.
        """
        super().__init__(parent)
        self._engine = engine
        self._socket_name = socket_name or default_socket_name()
        self._descriptor = Path(descriptor) if descriptor else descriptor_path()
        self.token = token or secrets.token_urlsafe(32)
        self.max_role = Role(max_role)
        self._station_info = station_info
        self._tool_context = tool_context
        self._feed = feed
        self._connections: dict[QLocalSocket, _Connection] = {}

        self.newConnection.connect(self._accept)
        # Under whichever names this client offers them: the server lives on
        # the GUI thread, so what it is handed is the **Orchestrator proxy**,
        # whose already-queued deliveries carry the contract's two streams as
        # `event` and `verdict`.
        event_stream(engine).connect(self._on_event)
        verdict_stream(engine).connect(self._on_verdict)

    # ── Lifecycle ─────────────────────────────────────────────────────

    @property
    def socket_name(self) -> str:
        """The local-socket name this server listens on."""
        return self._socket_name

    @property
    def descriptor(self) -> Path:
        """The descriptor file this server writes while it is listening."""
        return self._descriptor

    def start(self) -> bool:
        """Listen, and publish the descriptor a client finds the app by.

        Removes any stale socket left by a process that did not shut down
        cleanly, restricts the socket to this user, and writes the
        descriptor with owner-only permissions.

        Returns:
            ``True`` when the server is listening; ``False`` when Qt refused
            the name, in which case the failure is logged and nothing is
            written.
        """
        QLocalServer.removeServer(self._socket_name)
        self.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        if not self.listen(self._socket_name):
            logger.error(
                "Gateway server could not listen on %s: %s",
                self._socket_name,
                self.errorString(),
            )
            return False
        self._write_descriptor()
        logger.info(
            "Gateway server listening on %s (max role %r)",
            self.fullServerName(),
            self.max_role.value,
        )
        return True

    def stop(self) -> None:
        """Stop listening, drop every connection, and remove the descriptor.

        Idempotent: stopping a server that never started, or stopping twice,
        is a no-op.
        """
        for connection in list(self._connections.values()):
            connection.socket.disconnectFromServer()
            connection.socket.close()
        self._connections.clear()
        if self.isListening():
            self.close()
        QLocalServer.removeServer(self._socket_name)
        try:
            self._descriptor.unlink(missing_ok=True)
        except OSError:
            logger.warning("Gateway server could not remove %s", self._descriptor)
        logger.info("Gateway server stopped")

    def _write_descriptor(self) -> None:
        """Write the descriptor file with owner-only permissions.

        The name, the pid, the schema version and the token — everything a
        client needs to find this app and be admitted by it, and nothing
        about the experiment.
        """
        payload = {
            "schema": SCHEMA_VERSION,
            "socket": self.fullServerName() or self._socket_name,
            "pid": os.getpid(),
            "token": self.token,
            "max_role": self.max_role.value,
        }
        try:
            self._descriptor.parent.mkdir(parents=True, exist_ok=True)
            self._descriptor.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            os.chmod(self._descriptor, 0o600)
        except OSError:
            logger.exception("Gateway server could not write %s", self._descriptor)

    # ── Accepting and reading ─────────────────────────────────────────

    def _accept(self) -> None:
        """Take every pending connection and wire its two slots."""
        while self.hasPendingConnections():
            socket = self.nextPendingConnection()
            if socket is None:
                return
            connection = _Connection(socket)
            self._connections[socket] = connection
            socket.readyRead.connect(lambda s=socket: self._read(s))
            socket.disconnected.connect(lambda s=socket: self._drop(s))
            logger.info("Gateway server accepted a connection")

    def _drop(self, socket: QLocalSocket) -> None:
        """Forget one connection.

        Args:
            socket: The socket that disconnected.
        """
        self._connections.pop(socket, None)
        socket.deleteLater()
        logger.info("Gateway server connection closed")

    def _read(self, socket: QLocalSocket) -> None:
        """Drain one socket's readable bytes and answer every whole frame.

        The one place partial reads are handled: bytes accumulate in the
        connection's buffer until a newline completes a frame, and a buffer
        that grows past ``MAX_FRAME_BYTES`` without one ends the connection.

        Args:
            socket: The socket with bytes waiting.
        """
        connection = self._connections.get(socket)
        if connection is None:
            return
        try:
            connection.buffer.extend(bytes(socket.readAll()))
        except Exception:  # noqa: BLE001 — a read must never raise into the loop
            logger.exception("Gateway server read failed")
            return

        while True:
            newline = connection.buffer.find(b"\n")
            if newline < 0:
                if len(connection.buffer) > MAX_FRAME_BYTES:
                    self._fail_frame(connection)
                return
            frame = bytes(connection.buffer[:newline])
            del connection.buffer[: newline + 1]
            if len(frame) > MAX_FRAME_BYTES:
                self._fail_frame(connection)
                return
            self._handle(connection, frame)

    def _fail_frame(self, connection: _Connection) -> None:
        """Answer an oversized frame and end the connection.

        Args:
            connection: The offending connection.
        """
        connection.buffer.clear()
        self._send(
            connection,
            self._error(
                None,
                FRAME_TOO_LARGE,
                f"frame exceeds the {MAX_FRAME_BYTES}-byte cap",
            ),
        )
        connection.socket.disconnectFromServer()

    def _handle(self, connection: _Connection, frame: bytes) -> None:
        """Parse one frame, dispatch it, and write back exactly one answer.

        Args:
            connection: The connection the frame arrived on.
            frame: The raw bytes of one line, newline already stripped.
        """
        if not frame.strip():
            return
        try:
            request = json.loads(frame.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            self._send(
                connection, self._error(None, PARSE_ERROR, f"malformed JSON: {error}")
            )
            return
        if not isinstance(request, Mapping) or not isinstance(
            request.get("method"), str
        ):
            self._send(
                connection,
                self._error(
                    None,
                    INVALID_REQUEST,
                    "a request is a JSON object with a string 'method'",
                ),
            )
            return

        request_id = request.get("id")
        try:
            response = self._dispatch(connection, request)
        except Exception as error:  # noqa: BLE001 — never raise into the loop
            logger.exception("Gateway server request failed")
            response = self._error(
                request_id, INTERNAL_ERROR, f"{type(error).__name__}: {error}"
            )
        if request_id is None:
            # A notification: JSON-RPC forbids answering it.
            return
        response.setdefault("id", request_id)
        self._send(connection, response)

    # ── The methods ───────────────────────────────────────────────────

    def _dispatch(
        self, connection: _Connection, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Route one parsed request to the method that answers it.

        Args:
            connection: The connection asking.
            request: The parsed JSON-RPC request object.

        Returns:
            A JSON-RPC response object, less its ``id``.
        """
        method = str(request["method"])
        params = request.get("params") or {}
        if not isinstance(params, Mapping):
            return self._error(
                request.get("id"), INVALID_PARAMS, "'params' must be an object"
            )
        request_id = request.get("id")

        if method == "hello":
            return self._hello(connection, params, request_id)

        gateway = connection.gateway
        if gateway is None:
            return self._error(
                request_id,
                NOT_AUTHENTICATED,
                "say 'hello' with a role, an actor_id and the token from "
                "gateway.json before anything else",
            )

        if method == "tools/list":
            return self._ok(request_id, {"tools": gateway.tool_schemas()})
        if method == "tools/call":
            name = params.get("name")
            args = params.get("args") or {}
            if not isinstance(name, str) or not isinstance(args, Mapping):
                return self._error(
                    request_id,
                    INVALID_PARAMS,
                    "tools/call takes a string 'name' and an object 'args'",
                )
            return self._ok(request_id, gateway.call_tool(name, dict(args)))
        if method == "status":
            status = gateway.status()
            return self._ok(
                request_id,
                {
                    "status": status.to_json() if status is not None else None,
                    "state": gateway.state(),
                    "attended": gateway.attended(),
                    "agent_gate": gateway.agent_gate().value,
                },
            )
        if method == "station":
            return self._ok(request_id, {"station": gateway.station().to_json()})
        if method == "events/subscribe":
            connection.subscribed = True
            return self._ok(
                request_id,
                {"subscribed": True, "events": ["StateChange", "StatusSnapshot"]},
            )
        return self._error(request_id, METHOD_NOT_FOUND, f"no method {method!r}")

    def _hello(
        self,
        connection: _Connection,
        params: Mapping[str, Any],
        request_id: Any,
    ) -> dict[str, Any]:
        """Authenticate one connection and give it its ``Gateway``.

        Args:
            connection: The connection handshaking.
            params: ``{role, actor_id, token}``.
            request_id: The request's id, for the answer.

        Returns:
            The JSON-RPC response — the granted identity, or the refusal.
        """
        if connection.gateway is not None:
            return self._error(
                request_id, ALREADY_AUTHENTICATED, "this connection already said hello"
            )
        token = params.get("token")
        if not isinstance(token, str) or not hmac.compare_digest(token, self.token):
            logger.warning("Gateway server refused a connection: bad token")
            return self._error(
                request_id, BAD_TOKEN, "the token does not match this app's gateway.json"
            )
        actor_id = params.get("actor_id")
        if not isinstance(actor_id, str) or not actor_id.strip():
            return self._error(
                request_id, INVALID_PARAMS, "'actor_id' must be a non-empty string"
            )
        try:
            role = Role(params.get("role"))
        except ValueError:
            return self._error(
                request_id,
                ROLE_REFUSED,
                f"unknown role {params.get('role')!r}; the roles that exist are "
                f"{[member.value for member in Role]}",
            )
        if not role_within_ceiling(role, self.max_role):
            logger.warning(
                "Gateway server refused role %r above the %r ceiling",
                role.value,
                self.max_role.value,
            )
            return self._error(
                request_id,
                ROLE_REFUSED,
                f"this app hands out at most the {self.max_role.value!r} role; "
                f"{role.value!r} grants more and is refused",
            )

        connection.gateway = Gateway(
            self._engine,
            role,
            actor_id,
            station_info=self._station_info,
            tool_context=self._tool_context,
            feed=self._resolve_feed(),
        )
        logger.info(
            "Gateway server admitted %r under role %r", actor_id, role.value
        )
        return self._ok(
            request_id,
            {
                "schema": SCHEMA_VERSION,
                "role": role.value,
                "actor_id": actor_id,
                "max_role": self.max_role.value,
            },
        )

    def _resolve_feed(self) -> AgentFeed | None:
        """Return the **Agent feed** a connection opening now records into.

        Resolved per connection rather than once, so a client that connects
        after the physicist opened a new experiment writes into that
        experiment's folder.

        Returns:
            The feed, or ``None`` when this server was built without one.
        """
        feed = self._feed
        if callable(feed):
            try:
                feed = feed()
            except Exception:  # noqa: BLE001 — no trail is better than no connection
                logger.exception("Gateway server could not resolve the agent feed")
                return None
        return feed if isinstance(feed, AgentFeed) else None

    # ── Notifications ─────────────────────────────────────────────────

    def _on_event(self, event: Any) -> None:
        """Push one engine event to every subscribed connection.

        Only the two kinds an out-of-process client needs to follow the
        machine — the state machine's transitions and the per-tick snapshot
        — travel here; a client that wants readings reads them through a
        tool.

        Args:
            event: Anything on the engine's event stream.
        """
        if isinstance(event, (StateChange, StatusSnapshot)):
            self._broadcast("event", {"event": event.to_json()})

    def _on_verdict(self, verdict: Verdict) -> None:
        """Push one verdict to every subscribed connection.

        Args:
            verdict: Any verdict off the engine's stream — this connection's
                own, another agent's, or the operator's.
        """
        try:
            payload = verdict.to_json()
        except Exception:  # noqa: BLE001 — never raise into the engine's emit
            logger.exception("Gateway server could not render a verdict")
            return
        self._broadcast("verdict", {"verdict": payload})

    def _broadcast(self, method: str, params: Mapping[str, Any]) -> None:
        """Write one notification to every subscribed connection.

        Args:
            method: The notification method name.
            params: Its parameters.
        """
        for connection in list(self._connections.values()):
            if connection.subscribed:
                self._send(
                    connection,
                    {"jsonrpc": "2.0", "method": method, "params": dict(params)},
                )

    # ── Writing ───────────────────────────────────────────────────────

    def _send(self, connection: _Connection, payload: Mapping[str, Any]) -> None:
        """Write one JSON message and its newline, never raising.

        Args:
            connection: Whom to write to.
            payload: The JSON-safe message.
        """
        try:
            frame = json.dumps(payload, default=str).encode("utf-8") + b"\n"
            connection.socket.write(frame)
            connection.socket.flush()
        except Exception:  # noqa: BLE001 — a dead client must not disturb the app
            logger.exception("Gateway server write failed")

    @staticmethod
    def _ok(request_id: Any, result: Mapping[str, Any]) -> dict[str, Any]:
        """Build one JSON-RPC success response.

        Args:
            request_id: The request being answered.
            result: Its result object.

        Returns:
            The response object.
        """
        return {"jsonrpc": "2.0", "id": request_id, "result": dict(result)}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        """Build one JSON-RPC error response.

        Args:
            request_id: The request being answered, or ``None``.
            code: One of the codes this module declares.
            message: The operator-facing explanation.

        Returns:
            The response object.
        """
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
