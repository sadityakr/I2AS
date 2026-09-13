"""``GatewayController`` — lets a human turn the Agent gateway on and off,
and change who it hands out to, while the app is running.

**The config sets the ceiling; the operator sets the switch.** ``monitor.yaml``'s
``gateway_max_role`` (``read_gateway_config()``) still names the most
authority this setup will EVER hand an out-of-process client — that stays a
setup decision, made once, in the file, exactly as every safety limit is.
What used to require an edit and a restart is only *whether the door is open
at all* and *which role, up to that ceiling, walks through it*: this
controller is what the Monitor window's Connections menu drives to change
either, without touching ``monitor.yaml`` or restarting the app.

**Restart, not reconfigure, on a role change.** Changing the ceiling a
listening server hands out does not retroactively change the role an
already-connected session holds (``Gateway.role`` is fixed at its own
``hello``). Rather than leave a stale, half-updated ceiling in place, this
controller stops the old server (dropping every connection, so a tightened
ceiling is never silently kept by a session that connected under the looser
one) and starts a fresh one — the same "construct new, don't mutate live"
shape ``_build_gateway_server()`` already uses at startup.

**Remote access is the same door, reached by URL.** The controller also owns
the optional **HTTP MCP endpoint** (``i2as/mcp/http_server.py``) and the
**access keys** it admits (``i2as/mcp/keys.py``). The endpoint is a client
of the socket server — every key's connection says ``hello`` to it like any
other — so it can only be on while the socket server is, it is restarted
with it, and a key is refused at creation if its role outranks the ceiling
exactly as a socket client is refused at its handshake.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from i2as.mcp.http_server import DEFAULT_PORT, McpHttpServer
from i2as.mcp.keys import AccessKey, KeyStore
from i2as.session.agent_feed import AgentFeed
from i2as.session.gateway.local_server import GatewayServer
from i2as.session.gateway.roles import ROLE_LADDER, Role, role_within_ceiling
from i2as.session.gateway.tools import ToolContext

logger = logging.getLogger(__name__)

__all__ = ["GatewayController"]


class GatewayController:
    """Owns the (re)creation of one running app's Gateway server.

    Attributes:
        ceiling: The highest role this setup's ``monitor.yaml`` allows —
            never exceeded regardless of what is asked for here.
        server: The currently listening ``GatewayServer``, or ``None`` while
            the gateway is off.
        http_server: The listening ``McpHttpServer``, or ``None`` while
            remote access is off.
        key_store: The access keys the HTTP endpoint admits, or ``None``
            when no key file was given (remote access then cannot start).
    """

    def __init__(
        self,
        engine: Any,
        *,
        station_info: Callable[[], Any],
        tool_context: ToolContext,
        feed: Callable[[], AgentFeed | None],
        ceiling: Role | str,
        socket_name: str | None = None,
        descriptor: Path | str | None = None,
        key_store: KeyStore | Path | str | None = None,
    ) -> None:
        """Prepare the controller without starting anything.

        Args:
            engine: The engine client every connection's ``Gateway`` attaches
                to — the **Orchestrator proxy**, same as ``_build_gateway_server()``
                takes.
            station_info: The station's declaration snapshot, or a callable
                returning it.
            tool_context: The collaborators the session tools read through.
            feed: Resolves the open experiment's **Agent feed** at connection
                time.
            ceiling: The most authority ``monitor.yaml`` permits this setup
                to ever hand out. A role asked for above this is refused
                (``start()`` raises ``ValueError``) rather than silently
                clamped, so a caller's mistake is visible immediately.
            socket_name: The local-socket name to listen on; defaults to the
                installation's.
            descriptor: Where to write the descriptor file; defaults to the
                installation's.
            key_store: The access-key store the HTTP endpoint admits, or
                the path of its JSON file; ``None`` leaves remote access
                unavailable.
        """
        self._engine = engine
        self._station_info = station_info
        self._tool_context = tool_context
        self._feed = feed
        self.ceiling = Role(ceiling)
        self._socket_name = socket_name
        self._descriptor = descriptor
        self.server: GatewayServer | None = None
        if isinstance(key_store, (str, Path)):
            key_store = KeyStore(key_store)
        self.key_store: KeyStore | None = key_store
        self.http_server: McpHttpServer | None = None
        self._http_settings: dict[str, Any] | None = None

    @property
    def enabled(self) -> bool:
        """Whether the gateway is currently listening."""
        return self.server is not None

    @property
    def current_role(self) -> Role | None:
        """The role the listening server hands out, or ``None`` while off."""
        return self.server.max_role if self.server is not None else None

    def allowed_roles(self) -> list[Role]:
        """Return the roles this setup's ceiling permits choosing, ordered.

        Returns:
            ``ROLE_LADDER`` filtered to the roles at or below ``ceiling`` —
            what a role selector should offer.
        """
        return [role for role in ROLE_LADDER if role_within_ceiling(role, self.ceiling)]

    def start(self, max_role: Role | str) -> GatewayServer:
        """Start (or restart, at a new role) the gateway server.

        Args:
            max_role: The role to hand out — must be within ``ceiling``.

        Returns:
            The newly listening ``GatewayServer``.

        Raises:
            ValueError: If *max_role* outranks ``ceiling``, or names no
                known ``Role``.
        """
        role = Role(max_role)
        if not role_within_ceiling(role, self.ceiling):
            raise ValueError(
                f"role {role.value!r} exceeds this setup's ceiling "
                f"{self.ceiling.value!r} (set in monitor.yaml)"
            )
        # A fresh socket server means a fresh token: the HTTP endpoint's
        # connections must say hello again, so it is taken down with the
        # old server and brought back over the new one.
        http_settings = self._http_settings if self.http_server is not None else None
        self.stop()
        server = GatewayServer(
            self._engine,
            max_role=role,
            station_info=self._station_info,
            tool_context=self._tool_context,
            feed=self._feed,
            socket_name=self._socket_name,
            descriptor=self._descriptor,
        )
        server.start()
        self.server = server
        if http_settings is not None:
            try:
                self.start_http(**http_settings)
            except (OSError, RuntimeError):
                logger.exception("the HTTP MCP endpoint could not be restarted")
        return server

    def stop(self) -> None:
        """Stop the gateway server, if one is listening. Idempotent.

        The HTTP endpoint goes first: it cannot outlive the socket it
        connects through.
        """
        self.stop_http()
        if self.server is not None:
            self.server.stop()
            self.server = None

    def connections(self) -> list[dict[str, str]]:
        """Return every connected client's ``{"actor_id", "role"}``.

        Returns:
            An empty list while the gateway is off.
        """
        return self.server.connections() if self.server is not None else []

    # ── Remote access: the HTTP MCP endpoint ──────────────────────────

    @property
    def http_enabled(self) -> bool:
        """Whether the HTTP MCP endpoint is currently listening."""
        return self.http_server is not None

    def start_http(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        public_url: str | None = None,
    ) -> McpHttpServer:
        """Start (or restart, with new settings) the HTTP MCP endpoint.

        Args:
            host: The address to bind — the loopback address for a tunnel
                agent on this machine, ``0.0.0.0`` to be reachable on the
                lab network.
            port: The port to listen on.
            public_url: The URL the operator publishes for this endpoint
                outside this machine (whatever a tunnel or proxy hands
                out), used to admit a browser origin naming it and to
                render client configurations. ``None`` when there is none.

        Returns:
            The listening ``McpHttpServer``.

        Raises:
            RuntimeError: If the gateway is off (there is nothing for the
                endpoint to connect to) or no key store was configured.
            OSError: If the port cannot be bound.
        """
        if self.server is None:
            raise RuntimeError("turn the Agent gateway on before serving it over HTTP")
        if self.key_store is None:
            raise RuntimeError("this session has no access-key store configured")
        self.stop_http()
        http_server = McpHttpServer(
            self.server.fullServerName() or self.server.socket_name,
            self.server.token,
            self.key_store,
            host=host,
            port=port,
            public_url=public_url,
        )
        http_server.start()
        self.http_server = http_server
        self._http_settings = {"host": host, "port": http_server.port, "public_url": public_url}
        return http_server

    def stop_http(self) -> None:
        """Stop the HTTP MCP endpoint, if it is listening. Idempotent."""
        if self.http_server is not None:
            self.http_server.stop()
            self.http_server = None

    def keys(self) -> list[AccessKey]:
        """Return every issued access key, without secrets.

        Returns:
            The keys, oldest first; empty when no store is configured.
        """
        return self.key_store.keys() if self.key_store is not None else []

    def create_key(
        self, name: str, role: Role | str, *, actor_id: str | None = None
    ) -> tuple[AccessKey, str]:
        """Issue an access key for the HTTP endpoint.

        Args:
            name: The key's label, unique in the store.
            role: The role every connection under this key asks for — must
                be within ``ceiling``, the same rule a socket client meets
                at its handshake.
            actor_id: The identity stamped on what the key does; defaults
                to *name*.

        Returns:
            The key record and its secret, which is shown once.

        Raises:
            RuntimeError: If no key store is configured.
            ValueError: If the role outranks the ceiling, the name is
                blank or already taken.
        """
        if self.key_store is None:
            raise RuntimeError("this session has no access-key store configured")
        wanted = Role(role)
        if not role_within_ceiling(wanted, self.ceiling):
            raise ValueError(
                f"role {wanted.value!r} exceeds this setup's ceiling "
                f"{self.ceiling.value!r} (set in monitor.yaml)"
            )
        return self.key_store.create(name, actor_id=actor_id, role=wanted.value)

    def revoke_key(self, name: str) -> bool:
        """Delete an access key and drop any connection it holds open.

        Args:
            name: The key's label.

        Returns:
            ``True`` when a key was removed.
        """
        if self.key_store is None:
            return False
        removed = self.key_store.revoke(name)
        if self.http_server is not None:
            self.http_server.drop_key(name)
        return removed
