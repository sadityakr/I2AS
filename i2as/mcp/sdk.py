"""The optional backend: the same session, framed by the ``mcp`` package.

A deployment that installs the optional ``mcp`` extra gets its stdio framing
from the package that owns the protocol, so a revision of MCP that changes
how bytes are put on the wire is picked up by upgrading a dependency rather
than by editing this repository. Nothing an agent asks for changes: every
request is answered by the same ``McpAdapter``, so the two backends share
one dispatch table and one set of payloads.

**The shim is the reference backend.** It is the one the tests exercise
(``mcp`` is an extra, and the suite must run without it), it serves the
whole surface, and it is what runs when the package is absent. This module
is a framing swap on top of it, and it declines to serve rather than serve
partially: ``serve_with_sdk()`` checks, BEFORE a byte of stdin is read, that
the installed package exposes every name it uses, and returns ``False`` on
any mismatch so the caller falls back to the shim with the session still
intact. A version of the SDK this module was not written against therefore
costs a log line, never a broken session.

**The one visible difference**, and the reason the shim stays the default:
the SDK owns the serving loop, so the app's events are delivered at request
boundaries — after whatever the session next asks — instead of the instant
the app emits them. A session that watches a ramp wants the shim.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from i2as.mcp.adapter import McpAdapter

logger = logging.getLogger(__name__)

__all__ = ["serve_with_sdk", "sdk_unavailable_reason"]

#: The names this module calls on the installed package. Checked up front so
#: a mismatch is a fallback rather than a traceback halfway through a
#: session. ``Server`` takes its handlers as constructor arguments, each an
#: ``async (ctx, params) -> result`` callable.
_REQUIRED_SERVER_NAMES = ("Server",)
_REQUIRED_STDIO_NAMES = ("stdio_server",)
_REQUIRED_TYPE_NAMES = (
    "ListToolsResult",
    "CallToolResult",
    "ListResourcesResult",
    "ReadResourceResult",
)


def sdk_unavailable_reason() -> str | None:
    """Say why the ``mcp`` package cannot frame this session, if it cannot.

    Returns:
        ``None`` when the installed package exposes everything this module
        uses, else one sentence naming what is missing — the package itself,
        one of its modules, or a symbol this module was written against.
    """
    try:
        from mcp import server as mcp_server  # noqa: PLC0415 — the lazy import IS the point
        from mcp import types as mcp_types  # noqa: PLC0415
        from mcp.server import stdio as mcp_stdio  # noqa: PLC0415
    except ImportError as error:
        return f"the optional 'mcp' package is not installed ({error})"

    for module, names in (
        (mcp_server, _REQUIRED_SERVER_NAMES),
        (mcp_stdio, _REQUIRED_STDIO_NAMES),
        (mcp_types, _REQUIRED_TYPE_NAMES),
    ):
        missing = [name for name in names if not hasattr(module, name)]
        if missing:
            return (
                f"the installed 'mcp' package's {module.__name__} does not "
                f"expose {missing}"
            )
    return None


def serve_with_sdk(adapter: McpAdapter) -> bool:
    """Serve one MCP session over stdio through the ``mcp`` package.

    Args:
        adapter: The translation to serve, already opened.

    Returns:
        ``True`` when the session was served to completion; ``False`` when
        the installed package could not frame it, in which case nothing has
        been read from stdin and the caller may serve with the shim
        instead.
    """
    reason = sdk_unavailable_reason()
    if reason is not None:
        logger.info("MCP adapter: framing with the in-repo shim — %s", reason)
        return False
    try:
        asyncio.run(_serve(adapter))
    except TypeError:
        # The constructor or a handler signature is not the one this module
        # was written against. Raised before `stdio_server()` opens, so the
        # session is still untouched and the shim can take it.
        logger.warning(
            "MCP adapter: the installed 'mcp' package's Server takes "
            "different handlers; framing with the in-repo shim instead",
            exc_info=True,
        )
        return False
    return True


async def _serve(adapter: McpAdapter) -> None:
    """Build the SDK server and run it over stdio.

    Args:
        adapter: The translation every handler forwards to.

    Raises:
        TypeError: If the installed package's ``Server`` does not accept the
            handlers below, which ``serve_with_sdk()`` turns into a
            fallback.
    """
    from mcp import types as mcp_types  # noqa: PLC0415
    from mcp.server import Server  # noqa: PLC0415
    from mcp.server.stdio import stdio_server  # noqa: PLC0415

    def answer(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Ask the one dispatch table for one method's result.

        Args:
            method: The MCP method.
            params: Its parameters.

        Returns:
            The result object.

        Raises:
            RuntimeError: Carrying the adapter's own message when the
                adapter answered with a JSON-RPC error, so the SDK renders
                it as this request's error.
        """
        response = adapter.handle(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        ) or {}
        error = response.get("error")
        if error:
            raise RuntimeError(str(error.get("message", "the I2AS gateway failed")))
        return dict(response.get("result") or {})

    async def on_list_tools(ctx: Any, params: Any) -> Any:
        """Answer ``tools/list``."""
        await _push_events(ctx, adapter)
        return mcp_types.ListToolsResult.model_validate(answer("tools/list"))

    async def on_call_tool(ctx: Any, params: Any) -> Any:
        """Answer ``tools/call``."""
        await _push_events(ctx, adapter)
        call = {"name": params.name, "arguments": dict(params.arguments or {})}
        return mcp_types.CallToolResult.model_validate(answer("tools/call", call))

    async def on_list_resources(ctx: Any, params: Any) -> Any:
        """Answer ``resources/list``."""
        return mcp_types.ListResourcesResult.model_validate(answer("resources/list"))

    async def on_read_resource(ctx: Any, params: Any) -> Any:
        """Answer ``resources/read``."""
        read = {"uri": str(params.uri)}
        result = answer("resources/read", read)
        await _push_events(ctx, adapter)
        return mcp_types.ReadResourceResult.model_validate(result)

    server = Server(
        "i2as",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
        on_list_resources=on_list_resources,
        on_read_resource=on_read_resource,
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def _push_events(ctx: Any, adapter: McpAdapter) -> None:
    """Send the app's waiting events to the session, if this SDK can.

    Args:
        ctx: The SDK's per-request context.
        adapter: The translation holding the gateway connection.
    """
    send = getattr(ctx, "send_log_message", None) or getattr(
        getattr(ctx, "session", None), "send_log_message", None
    )
    notifications = adapter.drain_notifications()
    if send is None:
        if notifications:
            logger.warning(
                "MCP adapter: this 'mcp' package exposes no way to push a log "
                "message, so %d app event(s) were not delivered",
                len(notifications),
            )
        return
    for notification in notifications:
        params = notification["params"]
        await send(
            level=params["level"], data=params["data"], logger=params["logger"]
        )
