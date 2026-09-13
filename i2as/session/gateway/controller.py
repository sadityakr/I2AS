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
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
        """
        self._engine = engine
        self._station_info = station_info
        self._tool_context = tool_context
        self._feed = feed
        self.ceiling = Role(ceiling)
        self._socket_name = socket_name
        self._descriptor = descriptor
        self.server: GatewayServer | None = None

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
        return server

    def stop(self) -> None:
        """Stop the gateway server, if one is listening. Idempotent."""
        if self.server is not None:
            self.server.stop()
            self.server = None

    def connections(self) -> list[dict[str, str]]:
        """Return every connected client's ``{"actor_id", "role"}``.

        Returns:
            An empty list while the gateway is off.
        """
        return self.server.connections() if self.server is not None else []
