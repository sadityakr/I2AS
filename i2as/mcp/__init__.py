"""i2as.mcp — the **MCP adapter**: a translator, in its own process.

Run as ``python -m i2as.mcp``, this package speaks the Model Context
Protocol over stdio to an external agent session and newline-delimited
JSON-RPC over a local socket to the running app's **Gateway server**. It
translates between the two and does nothing else.

**Why it is a separate process, and why it imports almost nothing.** The
whole point of putting the transport out of process is that the thing
speaking to the outside world *cannot* touch the Station, the Orchestrator
or the session layer — not "does not", but *cannot*, mechanically. Import
contract C21 in ``pyproject.toml`` is what makes that true: the only I2AS
module this package may import is ``i2as.core.events``, the pure
contract. There is no code path from here to an instrument, and adding one
fails the build.

**Nothing here is hand-written per tool.** ``tools/list`` is the **Gateway
server**'s own ``tools/list`` re-keyed into MCP's shape, and each of the
three resources is a URI over one of the read tools that surface already
offers, so what an external session sees is what the **Agent gateway**
rendered from ``CommandName`` and the station declaration. A tool added to
the app appears here with no change to this package.

**Two backends, one translation.** ``adapter.py`` answers every MCP method
by making the gateway request that already answers it; ``shim.py`` frames
those answers over stdio with the stdlib, and ``sdk.py`` frames the same
answers through the ``mcp`` package when a deployment installed the optional
extra. The shim is the reference backend — it is what the tests exercise and
what runs when the package is absent — and ``sdk.py`` declines to serve
rather than serve partially, so a package it was not written against costs a
log line instead of a session.

See ``i2as/mcp/README.md`` for the folder standard and ``GLOSSARY.md``
for the **MCP adapter** and **Gateway server** vocabulary.
"""

from i2as.mcp.adapter import McpAdapter
from i2as.mcp.client import (
    DEFAULT_TIMEOUT_S,
    DESCRIPTOR_FILENAME,
    SUPPORTED_SCHEMA,
    GatewayClient,
    GatewayError,
    default_descriptor_path,
    read_descriptor,
)
from i2as.mcp.translate import (
    HANDSHAKE_PROTOCOL_VERSION,
    LATEST_PROTOCOL_VERSION,
    MANIFEST_URI,
    PROTOCOL_VERSIONS,
    RESOURCE_TOOLS,
    RESOURCES,
    STATION_URI,
    STATUS_URI,
    discover_result,
    initialize_result,
    log_notification,
    log_notifications,
    mcp_tool,
    negotiate_protocol_version,
    resource_contents,
    resources_list_result,
    tool_result,
    tools_list_result,
)

__all__ = [
    "McpAdapter",
    "GatewayClient",
    "GatewayError",
    "DESCRIPTOR_FILENAME",
    "DEFAULT_TIMEOUT_S",
    "SUPPORTED_SCHEMA",
    "default_descriptor_path",
    "read_descriptor",
    "PROTOCOL_VERSIONS",
    "LATEST_PROTOCOL_VERSION",
    "HANDSHAKE_PROTOCOL_VERSION",
    "RESOURCES",
    "RESOURCE_TOOLS",
    "STATUS_URI",
    "STATION_URI",
    "MANIFEST_URI",
    "mcp_tool",
    "tools_list_result",
    "resources_list_result",
    "tool_result",
    "resource_contents",
    "log_notification",
    "log_notifications",
    "initialize_result",
    "discover_result",
    "negotiate_protocol_version",
]
