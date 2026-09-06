"""Every MCP payload this adapter produces, as pure functions.

**The translation standard.** Nothing in this module describes a tool, an
instrument or a command in words of its own. A tool's name, description and
schema are the **Gateway server**'s, re-keyed from the contract's
``input_schema`` to MCP's ``inputSchema`` and nothing more; a tool's answer
is the **Agent gateway**'s own answer dict, rendered as JSON text; a
resource is one of the three read tools that surface already offers, named
by its URI here and read through ``tools/call`` like everything else. A tool
added to the app therefore reaches an external session with no change here,
which is the same argument the tool surface itself is built on.

Pure by design — no socket, no process, no state — so the two serving
backends (``shim.py`` and ``sdk.py``) produce byte-identical payloads and
the tests can check the shapes without either.

Two protocol eras, one server
-----------------------------

MCP's current revision (``2026-07-28``) is stateless: it retired the
``initialize`` / ``notifications/initialized`` exchange, made capability
discovery an ordinary ``server/discover`` request, and gave every cacheable
result three required fields — ``resultType``, ``ttlMs`` and ``cacheScope``.
The revisions before it are handshake-based, and the editors driving this
adapter still speak them.

This module renders both, because the two are not in conflict: a
handshake-era client opens with ``initialize`` and is answered with a
revision it knows (``HANDSHAKE_PROTOCOL_VERSION`` when it asked for one this
adapter does not), while a stateless client sends ``server/discover`` — or
sends nothing at all and calls ``tools/list`` straight away, which is what
"stateless" means. Every result carries the current revision's cacheability
fields whichever era asked for it, since an older client ignores fields it
does not know and a newer one requires them.

**Nothing here is cached, so nothing here claims to be.** ``ttlMs`` is 0 —
"immediately stale" — on every list: the tool surface grows when an
instrument connects, and the station's status changes every tick, so a
client that reused either would be reading a station that no longer exists.
``cacheScope`` is ``private`` for the same reason it would be on any
authenticated read: a connection is one actor with one **Role**, and its
answers are scoped to that authority, not shareable with the next client.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from i2as.core.events import event_from_json

__all__ = [
    "PROTOCOL_VERSIONS",
    "LATEST_PROTOCOL_VERSION",
    "HANDSHAKE_PROTOCOL_VERSION",
    "SERVER_NAME",
    "SERVER_VERSION",
    "INSTRUCTIONS",
    "CAPABILITIES",
    "STATUS_URI",
    "STATION_URI",
    "MANIFEST_URI",
    "RESOURCES",
    "RESOURCE_TOOLS",
    "negotiate_protocol_version",
    "initialize_result",
    "discover_result",
    "mcp_tool",
    "tools_list_result",
    "resources_list_result",
    "tool_result",
    "resource_contents",
    "log_notification",
    "log_notifications",
]

#: The protocol revisions this adapter speaks, newest first.
PROTOCOL_VERSIONS: tuple[str, ...] = (
    "2026-07-28",
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)

#: The newest revision this adapter speaks, advertised in ``server/discover``.
LATEST_PROTOCOL_VERSION = PROTOCOL_VERSIONS[0]

#: What an ``initialize`` is answered with when the client asked for a
#: revision this adapter does not know. Deliberately NOT
#: ``LATEST_PROTOCOL_VERSION``: a client that sent ``initialize`` is speaking
#: a handshake-based revision, and answering it with the stateless one would
#: name an era in which the request it just made does not exist.
HANDSHAKE_PROTOCOL_VERSION = "2025-11-25"

#: How this adapter names itself.
SERVER_NAME = "i2as"

#: The adapter's own version, which is the application's.
SERVER_VERSION = "0.1.0"

#: What this server offers. ``tools`` and ``resources`` are the two surfaces
#: it serves; ``logging`` is the channel the app's events travel on.
CAPABILITIES: dict[str, Any] = {"tools": {}, "resources": {}, "logging": {}}

#: The one paragraph an external session reads before it does anything. It
#: says what the surface IS and where the authority lives, and deliberately
#: describes no individual tool: the tools describe themselves.
INSTRUCTIONS = (
    "This server drives a running I2AS cryostat. Every tool is the "
    "application's own action surface, rendered from its command contract "
    "and the station's declaration, and every call is judged by the "
    "connection's role, the human's attendance and the kill switch before "
    "anything reaches an instrument. A refusal comes back as a normal "
    "result whose 'code' names the rule that refused; read it rather than "
    "retrying. Read i2as://status and i2as://station before acting, "
    "and validate or probe a run before starting one."
)

#: The three resources: the live snapshot, the station's declaration, and
#: the capability manifest.
STATUS_URI = "i2as://status"
STATION_URI = "i2as://station"
MANIFEST_URI = "i2as://manifest"

#: Which of the **Tool surface**'s own read tools answers each resource. A
#: resource is a URI over a tool that already exists, never a fourth way to
#: read the station: the same schema validation, the same role check and the
#: same trail in the **Agent feed** apply to a resource read as to any call.
RESOURCE_TOOLS: dict[str, str] = {
    STATUS_URI: "read_status",
    STATION_URI: "read_station_info",
    MANIFEST_URI: "read_manifest",
}

RESOURCES: tuple[dict[str, Any], ...] = (
    {
        "uri": STATUS_URI,
        "name": "status",
        "description": (
            "The engine's latest status snapshot: state, active run, "
            "attendance and the kill switch's setting."
        ),
        "mimeType": "application/json",
    },
    {
        "uri": STATION_URI,
        "name": "station",
        "description": (
            "The station's declaration: every instrument, what it reads and "
            "what it can be asked to do, with the configured bounds."
        ),
        "mimeType": "application/json",
    },
    {
        "uri": MANIFEST_URI,
        "name": "manifest",
        "description": (
            "The setup's capability manifest: the same declaration, with "
            "each instrument's capabilities resolved into its groups."
        ),
        "mimeType": "application/json",
    },
)


def _cacheable() -> dict[str, Any]:
    """Return the three fields the current revision requires on a list result.

    Returns:
        ``{"resultType": "complete", "ttlMs": 0, "cacheScope": "private"}`` —
        one whole answer, stale the moment it is given, scoped to the
        connection that asked. See this module's docstring for why nothing
        here is cacheable.
    """
    return {"resultType": "complete", "ttlMs": 0, "cacheScope": "private"}


def negotiate_protocol_version(requested: Any) -> str:
    """Choose the revision to answer one ``initialize`` with.

    Args:
        requested: Whatever the client put in ``protocolVersion``.

    Returns:
        The client's own revision when this adapter speaks it, else
        ``HANDSHAKE_PROTOCOL_VERSION`` — which is what the handshake
        prescribes: the server answers with a version it supports, and the
        client decides whether it can continue.
    """
    if isinstance(requested, str) and requested in PROTOCOL_VERSIONS:
        return requested
    return HANDSHAKE_PROTOCOL_VERSION


def initialize_result(params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the answer to one handshake-era ``initialize`` request.

    Args:
        params: The client's ``initialize`` parameters, read only for the
            protocol version it asked for.

    Returns:
        The result: the negotiated version, this server's capabilities, its
        identity and its instructions.
    """
    requested = (params or {}).get("protocolVersion")
    return {
        "protocolVersion": negotiate_protocol_version(requested),
        "capabilities": dict(CAPABILITIES),
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "instructions": INSTRUCTIONS,
    }


def discover_result() -> dict[str, Any]:
    """Build the answer to one stateless-era ``server/discover`` request.

    The same three facts ``initialize`` answers with — who this server is,
    what it offers and how to use it — with the single negotiated version
    replaced by the list of every revision this adapter speaks, because in a
    stateless session there is no handshake in which to agree on one.

    Returns:
        The discovery result: ``supportedVersions``, ``capabilities``,
        ``serverInfo``, ``instructions`` and the cacheability fields.
    """
    return {
        **_cacheable(),
        "supportedVersions": list(PROTOCOL_VERSIONS),
        "capabilities": dict(CAPABILITIES),
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "instructions": INSTRUCTIONS,
    }


def mcp_tool(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Re-key one gateway tool schema into MCP's tool shape.

    The whole translation is a rename: the contract publishes
    ``input_schema`` and MCP asks for ``inputSchema``. Nothing is added,
    dropped or rewritten, which is what keeps the surface an external
    session sees identical to the one every other client sees.

    Args:
        schema: One entry of the gateway's ``tool_schemas()`` —
            ``{name, description, input_schema}``.

    Returns:
        ``{name, description, inputSchema}``.
    """
    return {
        "name": str(schema.get("name", "")),
        "description": str(schema.get("description", "")),
        "inputSchema": dict(schema.get("input_schema") or {"type": "object"}),
    }


def tools_list_result(schemas: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Render the gateway's whole tool list as one ``tools/list`` result.

    Args:
        schemas: What the **Gateway server**'s ``tools/list`` answered with.

    Returns:
        ``{tools: [...], resultType, ttlMs, cacheScope}``, the tools in the
        order the gateway rendered them.
    """
    return {**_cacheable(), "tools": [mcp_tool(schema) for schema in schemas]}


def resources_list_result() -> dict[str, Any]:
    """Render the three resources as one ``resources/list`` result.

    Returns:
        ``{resources: [...], resultType, ttlMs, cacheScope}``.
    """
    return {**_cacheable(), "resources": [dict(one) for one in RESOURCES]}


def tool_result(answer: Mapping[str, Any]) -> dict[str, Any]:
    """Render one gateway tool answer as an MCP tool result.

    A refusal is NOT an MCP protocol error: the gateway answered, and its
    answer names the rule that refused. MCP's own guidance is the same one
    the **Verdict standard** follows — an error the caller can act on
    belongs in the result, where the model can read it, rather than in the
    envelope, where it cannot.

    Args:
        answer: The dict ``Gateway.call_tool()`` produced.

    Returns:
        ``{content: [one text block of the answer as JSON], isError,
        resultType}``.
    """
    return {
        "resultType": "complete",
        "content": [{"type": "text", "text": json.dumps(dict(answer), indent=2)}],
        "isError": not bool(answer.get("ok", False)),
    }


def resource_contents(uri: str, payload: Any) -> dict[str, Any]:
    """Render one resource read.

    Args:
        uri: The resource that was read.
        payload: Its JSON-safe content.

    Returns:
        ``{contents: [{uri, mimeType, text}], resultType}``.
    """
    return {
        "resultType": "complete",
        "contents": [
            {
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(payload, indent=2, default=str),
            }
        ],
    }


def _describe(notification: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Summarise one gateway notification for a log message.

    Args:
        notification: The gateway's ``event`` or ``verdict`` notification.

    Returns:
        The MCP logging level and the structured payload to send.
    """
    params = notification.get("params") or {}
    if notification.get("method") == "verdict":
        verdict = dict(params.get("verdict") or {})
        level = "info" if verdict.get("code") == "OK" else "warning"
        return level, {"verdict": verdict}
    payload = dict(params.get("event") or {})
    try:
        # Rebuilding through the contract is what makes the summary honest:
        # an event kind this adapter does not know raises here rather than
        # being described from whatever keys happened to be present.
        event = event_from_json(payload)
    except (KeyError, ValueError):
        return "info", {"event": payload}
    return "info", {"event": payload, "summary": type(event).__name__}


def log_notification(notification: Mapping[str, Any]) -> dict[str, Any] | None:
    """Translate one gateway notification into an MCP log message.

    MCP has no channel for an arbitrary server-pushed domain event, and
    inventing one would make this adapter a protocol of its own. The
    spec-defined ``notifications/message`` is the channel that exists, so a
    state change and a verdict travel as structured log data with the whole
    contract message in ``data`` — a client that wants the event reads it
    there, unmodified.

    Args:
        notification: A gateway ``event`` or ``verdict`` notification.

    Returns:
        The MCP notification, or ``None`` for anything else.
    """
    if notification.get("method") not in {"event", "verdict"}:
        return None
    level, data = _describe(notification)
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {"level": level, "logger": "i2as.gateway", "data": data},
    }


def log_notifications(
    notifications: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Translate a batch of gateway notifications, dropping the untranslatable.

    Args:
        notifications: What ``GatewayClient.take_notifications()`` returned.

    Returns:
        The MCP notifications, in order.
    """
    translated = (log_notification(one) for one in notifications)
    return [one for one in translated if one is not None]
