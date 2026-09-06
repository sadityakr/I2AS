"""The translation itself: one MCP request in, one gateway request out.

``McpAdapter`` is the whole adapter minus its framing. It holds one
``GatewayClient`` — one connection to the running app's **Gateway server**,
with one **Role** and one actor id — and answers every MCP method by making
the gateway request that already answers it. It parses no bytes, writes to
no stream and knows nothing about stdio, which is what lets the two serving
backends (``shim.py`` and ``sdk.py``) differ in framing alone.

**Every method is a forward.** ``tools/list`` is the gateway's own
``tools/list``, re-keyed; ``tools/call`` is the gateway's ``tools/call``,
verbatim in both directions; a resource read is a ``tools/call`` on one of
the three read tools the **Tool surface** already offers. This module
therefore contains no list of tools, no argument handling and no knowledge
of what an instrument is — a tool added to the app appears here without a
line changing.

**A refusal is a result, not an error.** The gateway answers a refused call
with a dict naming the rule that refused it, and that dict is rendered into
the tool result where the model can read it. A JSON-RPC error is reserved
for the cases where there is no answer at all: a method this adapter does
not serve, parameters it cannot read, a URI that is not a resource, and a
gateway connection that failed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from i2as.mcp.client import GatewayClient, GatewayError
from i2as.mcp.translate import (
    RESOURCE_TOOLS,
    discover_result,
    initialize_result,
    log_notifications,
    resource_contents,
    resources_list_result,
    tool_result,
    tools_list_result,
)

logger = logging.getLogger(__name__)

__all__ = ["McpAdapter", "METHOD_NOT_FOUND", "INVALID_PARAMS", "INTERNAL_ERROR",
           "RESOURCE_NOT_FOUND"]

# ── JSON-RPC 2.0's own error codes ────────────────────────────────────
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

#: MCP's own code for a URI the server does not serve.
RESOURCE_NOT_FOUND = -32002


class McpAdapter:
    """One MCP session over one **Gateway server** connection.

    Attributes:
        client: The connection to the running app.
    """

    def __init__(self, client: GatewayClient) -> None:
        """Wrap one gateway connection.

        Args:
            client: A ``GatewayClient``, connected or not — ``open()``
                connects it.
        """
        self.client = client

    # ── Lifecycle ─────────────────────────────────────────────────────

    def open(self) -> dict[str, Any]:
        """Connect to the app and subscribe to its events.

        Subscribing at open rather than on demand is what makes the event
        stream a property of the session: MCP has no way for a client to ask
        for notifications, so the adapter asks for them once and translates
        whatever arrives.

        Returns:
            The gateway's ``hello`` result — the granted role, the actor id,
            the wire schema version and the deployment's ceiling.

        Raises:
            GatewayError: If the app cannot be reached or the handshake is
                refused.
        """
        hello = self.client.connect()
        self.client.call("events/subscribe")
        logger.info(
            "MCP adapter connected as %r under role %r",
            hello.get("actor_id"),
            hello.get("role"),
        )
        return hello

    def close(self) -> None:
        """Close the gateway connection. Idempotent."""
        self.client.close()

    # ── Requests ──────────────────────────────────────────────────────

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        """Answer one MCP request.

        Args:
            request: A parsed JSON-RPC request or notification object.

        Returns:
            The JSON-RPC response to write back, or ``None`` when the
            message was a notification, which JSON-RPC forbids answering.
        """
        request_id = request.get("id")
        method = request.get("method")
        if not isinstance(method, str):
            if request_id is None:
                return None
            return self._error(
                request_id, INVALID_REQUEST, "a request needs a string 'method'"
            )
        params = request.get("params") or {}
        if not isinstance(params, Mapping):
            params = {}

        if request_id is None:
            # Notifications — `notifications/initialized`, `notifications/
            # cancelled` and anything else a client sends without an id.
            # There is nothing to do with them and nothing may be written
            # back.
            return None

        try:
            result = self._dispatch(method, params)
        except _McpError as error:
            return self._error(request_id, error.code, str(error))
        except GatewayError as error:
            logger.warning("MCP adapter: the gateway refused %r: %s", method, error)
            return self._error(
                request_id, INTERNAL_ERROR, f"the I2AS gateway: {error}"
            )
        except Exception as error:  # noqa: BLE001 — an editor's session must survive
            logger.exception("MCP adapter failed on %r", method)
            return self._error(
                request_id, INTERNAL_ERROR, f"{type(error).__name__}: {error}"
            )
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _dispatch(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """Route one method to the gateway request that answers it.

        Args:
            method: The MCP method name.
            params: Its parameters.

        Returns:
            The MCP result object.

        Raises:
            _McpError: For an unserved method, unreadable parameters or an
                unknown resource URI.
            GatewayError: If the gateway connection fails.
        """
        if method == "initialize":
            return initialize_result(params)
        if method == "server/discover":
            return discover_result()
        if method == "ping":
            return {}
        if method == "logging/setLevel":
            # Accepted and ignored: every notification this adapter sends is
            # one the app itself emitted, and dropping the app's events
            # because a client asked for less noise would hide a state
            # change from the session driving the cryostat.
            return {}
        if method == "tools/list":
            answer = self.client.call("tools/list")
            return tools_list_result(answer.get("tools") or [])
        if method == "tools/call":
            return self._call_tool(params)
        if method == "resources/list":
            return resources_list_result()
        if method == "resources/templates/list":
            return {"resultType": "complete", "ttlMs": 0, "cacheScope": "private",
                    "resourceTemplates": []}
        if method == "resources/read":
            return self._read_resource(params)
        raise _McpError(METHOD_NOT_FOUND, f"no method {method!r}")

    def _call_tool(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Forward one ``tools/call`` and render the gateway's answer.

        Args:
            params: MCP's ``{name, arguments}``. ``args`` is accepted as a
                synonym for ``arguments``, because that is the key the
                gateway's own wire uses and a hand-driven session reaches
                for it.

        Returns:
            The MCP tool result.

        Raises:
            _McpError: If the parameters are not a name and an object.
            GatewayError: If the gateway connection fails.
        """
        name = params.get("name")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = params.get("args") or {}
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            raise _McpError(
                INVALID_PARAMS,
                "tools/call takes a string 'name' and an object 'arguments'",
            )
        answer = self.client.call(
            "tools/call", {"name": name, "args": dict(arguments)}
        )
        return tool_result(answer)

    def _read_resource(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Answer one ``resources/read`` through the tool that serves it.

        Args:
            params: MCP's ``{uri}``.

        Returns:
            The MCP resource contents.

        Raises:
            _McpError: If the URI is missing or is not one this server
                serves.
            GatewayError: If the gateway connection fails.
        """
        uri = params.get("uri")
        tool_name = RESOURCE_TOOLS.get(uri) if isinstance(uri, str) else None
        if tool_name is None:
            raise _McpError(
                RESOURCE_NOT_FOUND,
                f"no resource {uri!r}; this server serves "
                f"{sorted(RESOURCE_TOOLS)}",
            )
        answer = self.client.call("tools/call", {"name": tool_name, "args": {}})
        if not answer.get("ok", False):
            # A read that the gateway refused is still an answer, and its
            # reason is what the session needs to see — so it travels as the
            # resource's content rather than as an error with the reason
            # stripped off.
            return resource_contents(uri, dict(answer))
        return resource_contents(uri, answer.get("result"))

    # ── Notifications ─────────────────────────────────────────────────

    def drain_notifications(self) -> list[dict[str, Any]]:
        """Return every gateway notification waiting, translated for MCP.

        Returns:
            Zero or more ``notifications/message`` objects, oldest first.
        """
        self.client.receive_available()
        return log_notifications(self.client.take_notifications())

    # ── Errors ────────────────────────────────────────────────────────

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        """Build one JSON-RPC error response.

        Args:
            request_id: The request being answered.
            code: The JSON-RPC or MCP error code.
            message: The explanation the session reads.

        Returns:
            The response object.
        """
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }


class _McpError(Exception):
    """One answerable failure, carrying the JSON-RPC code that names it.

    Attributes:
        code: The JSON-RPC or MCP error code.
    """

    def __init__(self, code: int, message: str) -> None:
        """Build the error.

        Args:
            code: The code to answer with.
            message: The explanation.
        """
        super().__init__(message)
        self.code = code
