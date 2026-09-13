"""The **HTTP MCP endpoint**: the same session, reachable by URL.

``python -m i2as.mcp`` serves one MCP session on stdio, and a client that
speaks stdio must be able to launch that process — which means it must be on
this machine, with this Python on its path. A client on the web (a hosted
assistant, a browser front end, a tunnel) cannot do that. MCP's other
standard transport, `Streamable HTTP
<https://modelcontextprotocol.io/specification/2025-06-18/basic/transports>`_,
inverts the arrangement: the server is reachable at one URL, and a client
needs that URL and a credential, nothing else.

This module is that transport, in the stdlib, on a thread of its own:

* **One endpoint, ``/mcp``.** A JSON-RPC request is a ``POST`` to it and is
  answered with one JSON object; a notification is answered ``202``. A
  ``GET`` opens a server-sent-events stream that carries the app's events
  as they happen — the same ``notifications/message`` objects the stdio
  backends write. A ``DELETE`` ends a session.
* **A key is an agent.** Every request presents an **access key**
  (``i2as/mcp/keys.py``), either as ``Authorization: Bearer <key>`` or, for
  a client that cannot set a header, in the path as ``/mcp/<key>``. The key
  names the actor id and the role, and the transport holds exactly one
  **Gateway server** connection per key, opened on first use and kept until
  the key is revoked or the server stops. Two web sessions under one key
  are one agent to the app, which is what the key means.
* **It reaches the app the way every other client does.** Nothing here
  touches the engine: each key's connection is a ``GatewayClient`` to the
  local socket, wrapped in the same ``McpAdapter`` the stdio backends use,
  so every tool call is the app's own ``tools/call``, judged at the same
  handshake against the same ceiling. Import contract C21 holds — this
  module imports the stdlib and ``i2as.mcp`` only.
* **The thread blocks; the app does not.** Requests are served on the
  HTTP server's own threads, which block on the socket while the GUI
  thread answers. The single hardware thread standard holds: this file
  adds no thread that touches an instrument.

Security, in three lines: a request without a valid key is ``401``; a
request with a browser ``Origin`` that is neither local nor the published
public URL is ``403`` (the DNS-rebinding rule the specification requires);
and the server binds to the loopback address unless told otherwise. What
the URL is *outside* this machine — a tunnel, a reverse proxy, a LAN
address — is deliberately not this module's business: it serves plain HTTP
on one port, and the operator publishes whatever address forwards to it.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from i2as.mcp.adapter import McpAdapter
from i2as.mcp.client import DEFAULT_TIMEOUT_S, GatewayClient, GatewayError
from i2as.mcp.keys import AccessKey, KeyStore

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PORT",
    "ENDPOINT_PATH",
    "MAX_BODY_BYTES",
    "SESSION_HEADER",
    "McpHttpServer",
]

#: The one path the transport serves, as the specification's example does.
ENDPOINT_PATH = "/mcp"

#: The port the GUI offers by default; any free port works.
DEFAULT_PORT = 8765

#: The header a session id travels in, both directions.
SESSION_HEADER = "Mcp-Session-Id"

#: The largest request body accepted. Matches the stdio shim's frame cap
#: in spirit: a tool call is a name and a few arguments.
MAX_BODY_BYTES = 1 << 20

#: How often the events stream looks for new app events, and how often it
#: writes a comment to keep an idle connection open through a proxy.
_STREAM_POLL_S = 0.25
_STREAM_KEEPALIVE_S = 15.0

#: Hostnames a browser ``Origin`` may name without being listed explicitly.
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]", "::1"})

#: JSON-RPC's parse-error code, for a body that is not a message.
_PARSE_ERROR = -32700


class _KeySession:
    """One key's connection to the app: adapter plus the lock that serialises it.

    ``GatewayClient`` answers one request at a time, so every use of the
    adapter under a key — a ``POST`` being answered, the events stream
    draining — takes ``lock`` first.
    """

    def __init__(self, key: AccessKey, adapter: McpAdapter) -> None:
        self.key = key
        self.adapter = adapter
        self.lock = threading.Lock()


class McpHttpServer:
    """Serve the app's MCP surface over Streamable HTTP.

    Attributes:
        host: The address the listener binds to.
        port: The port it listens on (the actual one, once started, when
            ``0`` was asked for).
        key_store: Where the keys a request may present live.
        public_url: The address the operator published for this endpoint
            outside this machine, or ``None``; used only to admit a browser
            ``Origin`` naming it.
    """

    def __init__(
        self,
        socket_name: str,
        token: str,
        key_store: KeyStore,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        public_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """Prepare the server without listening yet.

        Args:
            socket_name: The **Gateway server**'s socket name, as its
                descriptor publishes it.
            token: The per-launch token that socket requires in ``hello``.
            key_store: The keys a request may present.
            host: The address to bind; the loopback address by default,
                which is all a tunnel agent on this machine needs.
            port: The port to bind; ``0`` picks a free one.
            public_url: The externally reachable URL, if the operator has
                one.
            timeout: Seconds one gateway answer may take.
        """
        self._socket_name = socket_name
        self._token = token
        self.key_store = key_store
        self.host = host
        self.port = int(port)
        self.public_url = (public_url or "").strip() or None
        self._timeout = timeout
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._sessions: dict[str, _KeySession] = {}
        self._session_ids: dict[str, str] = {}
        self._lock = threading.Lock()
        self._stopping = threading.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────

    @property
    def listening(self) -> bool:
        """Whether the listener is up."""
        return self._httpd is not None

    @property
    def local_url(self) -> str:
        """The URL a client on this machine uses."""
        host = "127.0.0.1" if self.host in {"0.0.0.0", "", "::"} else self.host
        return f"http://{host}:{self.port}{ENDPOINT_PATH}"

    def start(self) -> None:
        """Bind and start serving on a daemon thread.

        Raises:
            OSError: If the address cannot be bound — most often another
                program on the port, which the caller shows the operator.
        """
        if self._httpd is not None:
            return
        self._stopping.clear()
        server = self
        handler = type("_BoundHandler", (_Handler,), {"server_state": server})
        httpd = ThreadingHTTPServer((self.host, self.port), handler)
        httpd.daemon_threads = True
        self.port = int(httpd.server_address[1])
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="i2as-mcp-http",
            daemon=True,
        )
        self._thread.start()
        logger.info("HTTP MCP endpoint listening at %s", self.local_url)

    def stop(self) -> None:
        """Stop listening and close every key's gateway connection. Idempotent."""
        self._stopping.set()
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._session_ids.clear()
        for session in sessions:
            self._close_quietly(session)
        if httpd is not None:
            logger.info("HTTP MCP endpoint stopped")

    def drop_key(self, name: str) -> None:
        """Close the connection held under the key called *name*, if any.

        Called when a key is revoked, so a client still holding it loses
        the app at once rather than at the next reconnect.

        Args:
            name: The key's label.
        """
        with self._lock:
            session = self._sessions.pop(name, None)
            stale = [sid for sid, owner in self._session_ids.items() if owner == name]
            for sid in stale:
                del self._session_ids[sid]
        if session is not None:
            self._close_quietly(session)

    def connections(self) -> list[dict[str, str]]:
        """Return one ``{"key", "actor_id", "role"}`` per key currently connected.

        Returns:
            The live connections, in no particular order.
        """
        with self._lock:
            return [
                {"key": name, "actor_id": s.key.actor_id, "role": s.key.role}
                for name, s in self._sessions.items()
            ]

    # ── Sessions (used by the handler) ────────────────────────────────

    def session_for(self, key: AccessKey) -> _KeySession:
        """Return the connection held under *key*, opening it on first use.

        Args:
            key: The key a request presented.

        Returns:
            The session.

        Raises:
            GatewayError: If the app refuses the connection — a role above
                the ceiling, or the socket gone.
        """
        with self._lock:
            session = self._sessions.get(key.name)
            if session is not None and session.key.digest == key.digest:
                return session
        client = GatewayClient(
            self._socket_name,
            self._token,
            role=key.role,
            actor_id=key.actor_id,
            timeout=self._timeout,
        )
        adapter = McpAdapter(client)
        adapter.open()
        fresh = _KeySession(key, adapter)
        with self._lock:
            existing = self._sessions.get(key.name)
            if existing is not None and existing.key.digest == key.digest:
                # Two first requests raced; keep the one already registered.
                fresh_to_close: _KeySession | None = fresh
                session = existing
            else:
                self._sessions[key.name] = fresh
                fresh_to_close = existing
                session = fresh
        if fresh_to_close is not None:
            self._close_quietly(fresh_to_close)
        return session

    def forget(self, session: _KeySession) -> None:
        """Drop a session whose connection failed, so the next request reconnects.

        Args:
            session: The session that raised.
        """
        with self._lock:
            if self._sessions.get(session.key.name) is session:
                del self._sessions[session.key.name]
        self._close_quietly(session)

    def new_session_id(self, key: AccessKey) -> str:
        """Mint the id an ``initialize`` answer carries.

        Args:
            key: The key the session belongs to.

        Returns:
            A random, ASCII-only id.
        """
        session_id = secrets.token_urlsafe(24)
        with self._lock:
            self._session_ids[session_id] = key.name
        return session_id

    def session_id_known(self, session_id: str) -> bool:
        """Whether *session_id* was minted here and not ended since."""
        with self._lock:
            return session_id in self._session_ids

    def end_session_id(self, session_id: str) -> bool:
        """Forget *session_id*; the key's connection stays for other sessions."""
        with self._lock:
            return self._session_ids.pop(session_id, None) is not None

    def origin_allowed(self, origin: str) -> bool:
        """Decide whether a browser ``Origin`` may reach this endpoint.

        Args:
            origin: The header's value.

        Returns:
            ``True`` for a local origin or the published public URL's
            origin; ``False`` otherwise. Requests without the header are
            not browsers and are not judged here.
        """
        parts = urlsplit(origin)
        if parts.hostname and parts.hostname.lower() in _LOCAL_HOSTS:
            return True
        if self.public_url:
            public = urlsplit(self.public_url)
            if (
                parts.scheme == public.scheme
                and (parts.hostname or "").lower() == (public.hostname or "").lower()
                and parts.port == public.port
            ):
                return True
        return False

    @property
    def stopping(self) -> threading.Event:
        """Set while ``stop()`` runs, so an events stream ends promptly."""
        return self._stopping

    @staticmethod
    def _close_quietly(session: _KeySession) -> None:
        try:
            session.adapter.close()
        except Exception:  # noqa: BLE001 — closing must not raise into the caller
            logger.debug("HTTP MCP endpoint: closing %r raised", session.key.name, exc_info=True)


class _Handler(BaseHTTPRequestHandler):
    """One HTTP request against the endpoint.

    ``server_state`` is bound per ``McpHttpServer.start()`` by subclassing,
    so the stdlib's handler-per-request construction reaches the owning
    server without a global.
    """

    server_state: McpHttpServer
    protocol_version = "HTTP/1.1"

    # ── Routing ───────────────────────────────────────────────────────

    def do_POST(self) -> None:  # noqa: N802 — the stdlib's naming
        self._serve("POST")

    def do_GET(self) -> None:  # noqa: N802
        self._serve("GET")

    def do_DELETE(self) -> None:  # noqa: N802
        self._serve("DELETE")

    def do_OPTIONS(self) -> None:  # noqa: N802
        # A browser's preflight. The answer says what the endpoint accepts;
        # the real request is still judged by key and origin.
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Authorization, Content-Type, Accept, Mcp-Session-Id, MCP-Protocol-Version, Last-Event-ID",
        )
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _serve(self, method: str) -> None:
        """Judge the request, then dispatch by method.

        Args:
            method: The HTTP method.
        """
        state = self.server_state
        path_key = self._route()
        if path_key is None and self.path.split("?", 1)[0] != ENDPOINT_PATH:
            self._json(HTTPStatus.NOT_FOUND, {"error": f"the MCP endpoint is {ENDPOINT_PATH}"})
            return

        origin = self.headers.get("Origin")
        if origin and not state.origin_allowed(origin):
            logger.warning("HTTP MCP endpoint refused origin %r", origin)
            self._json(HTTPStatus.FORBIDDEN, {"error": "this origin is not allowed"})
            return

        key = self._authenticate(path_key)
        if key is None:
            return

        if method == "POST":
            self._post(key)
        elif method == "GET":
            self._events(key)
        else:
            self._delete()

    def _route(self) -> str | None:
        """Return the key carried in the path, if the path is ``/mcp/<key>``."""
        path = self.path.split("?", 1)[0].rstrip("/")
        prefix = ENDPOINT_PATH + "/"
        if path.startswith(prefix) and "/" not in path[len(prefix):]:
            return path[len(prefix):]
        return None

    def _authenticate(self, path_key: str | None) -> AccessKey | None:
        """Resolve the request's key, answering ``401`` when there is none.

        Args:
            path_key: A key from the path, if the path carried one.

        Returns:
            The key, or ``None`` after the refusal has been written.
        """
        presented = path_key
        header = self.headers.get("Authorization", "")
        if header.lower().startswith("bearer "):
            presented = header[7:].strip()
        key = self.server_state.key_store.verify(presented or "")
        if key is None:
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("WWW-Authenticate", 'Bearer realm="i2as", error="invalid_token"')
            body = json.dumps(
                {"error": "present an I2AS access key as 'Authorization: Bearer <key>'"}
            ).encode("utf-8")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            logger.warning("HTTP MCP endpoint refused a request from %s: no valid key", self.client_address[0])
        return key

    # ── POST: one JSON-RPC message ────────────────────────────────────

    def _post(self, key: AccessKey) -> None:
        """Answer one JSON-RPC message.

        Args:
            key: The authenticated key.
        """
        state = self.server_state
        message = self._read_message()
        if message is None:
            return

        session_id = self.headers.get(SESSION_HEADER)
        if session_id and not state.session_id_known(session_id):
            # The specification's signal to start over with a new
            # initialize; a client that never sends an id is served too.
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown session; initialize again"})
            return

        try:
            session = state.session_for(key)
        except GatewayError as error:
            logger.warning("HTTP MCP endpoint: the app refused key %r: %s", key.name, error)
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": f"the I2AS gateway refused this key: {error}"},
            )
            return

        with session.lock:
            try:
                response = session.adapter.handle(message)
            except GatewayError as error:
                state.forget(session)
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": f"the I2AS gateway: {error}"})
                return

        extra: dict[str, str] = {}
        if message.get("method") == "initialize" and response is not None and "result" in response:
            extra[SESSION_HEADER] = state.new_session_id(key)
        if response is None:
            self._empty(HTTPStatus.ACCEPTED, extra)
            return
        self._json(HTTPStatus.OK, response, extra)

    def _read_message(self) -> dict[str, Any] | None:
        """Read and parse the body, writing the refusal when it is not one message.

        Returns:
            The JSON-RPC object, or ``None`` after answering.
        """
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Content-Length is required"})
            return None
        if length > MAX_BODY_BYTES:
            self._json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": f"a message may be at most {MAX_BODY_BYTES} bytes"},
            )
            return None
        raw = self.rfile.read(length) if length else b""
        try:
            message = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            self._json(
                HTTPStatus.BAD_REQUEST,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": _PARSE_ERROR, "message": f"malformed JSON: {error}"},
                },
            )
            return None
        if not isinstance(message, Mapping):
            self._json(
                HTTPStatus.BAD_REQUEST,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": _PARSE_ERROR,
                        "message": "one JSON-RPC message per request; batches are not served",
                    },
                },
            )
            return None
        return dict(message)

    # ── GET: the events stream ────────────────────────────────────────

    def _events(self, key: AccessKey) -> None:
        """Stream the app's events as server-sent events until the client leaves.

        Args:
            key: The authenticated key.
        """
        state = self.server_state
        accept = self.headers.get("Accept", "")
        if "text/event-stream" not in accept and "*/*" not in accept:
            self._json(
                HTTPStatus.METHOD_NOT_ALLOWED,
                {"error": "GET opens an event stream; send Accept: text/event-stream"},
            )
            return
        try:
            session = state.session_for(key)
        except GatewayError as error:
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": f"the I2AS gateway refused this key: {error}"},
            )
            return

        self.send_response(HTTPStatus.OK)
        self._cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        last_write = time.monotonic()
        try:
            while not state.stopping.is_set():
                with session.lock:
                    try:
                        notifications = session.adapter.drain_notifications()
                    except GatewayError:
                        state.forget(session)
                        return
                for notification in notifications:
                    self.wfile.write(
                        b"data: " + json.dumps(notification, default=str).encode("utf-8") + b"\n\n"
                    )
                if notifications:
                    self.wfile.flush()
                    last_write = time.monotonic()
                elif time.monotonic() - last_write > _STREAM_KEEPALIVE_S:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    last_write = time.monotonic()
                time.sleep(_STREAM_POLL_S)
        except (OSError, ValueError):
            # The client went away; that is how a stream ends.
            return

    # ── DELETE: end a session ─────────────────────────────────────────

    def _delete(self) -> None:
        """Forget the session the header names."""
        session_id = self.headers.get(SESSION_HEADER)
        if not session_id:
            self._json(HTTPStatus.BAD_REQUEST, {"error": f"send the {SESSION_HEADER} to end"})
            return
        if self.server_state.end_session_id(session_id):
            self._empty(HTTPStatus.NO_CONTENT)
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown session"})

    # ── Writing ───────────────────────────────────────────────────────

    def _cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin and self.server_state.origin_allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Expose-Headers", SESSION_HEADER)
            self.send_header("Vary", "Origin")

    def _json(self, status: HTTPStatus, payload: Mapping[str, Any], extra: Mapping[str, str] | None = None) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _empty(self, status: HTTPStatus, extra: Mapping[str, str] | None = None) -> None:
        self.send_response(status)
        self._cors_headers()
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — the stdlib's signature
        """Route the stdlib's access log through ``logging`` at debug level."""
        logger.debug("HTTP MCP endpoint %s - " + format, self.client_address[0], *args)
