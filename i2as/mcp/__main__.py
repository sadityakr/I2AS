"""``python -m i2as.mcp`` — the **MCP adapter** as its own process.

Finds the running app through the descriptor its **Gateway server** writes,
opens one connection under the role and identity this invocation declares,
and serves that connection as an MCP server on stdio until the client closes
it.

**It declares an identity, it does not choose an authority.** ``--role``
says what this session asks to be; whether it may be that is the app's
decision, taken at the handshake against the deployment's ceiling
(``monitor.yaml``'s ``gateway_max_role``). The default here is ``observer``
— the role that changes nothing — because a client launched by an editor
with no arguments should be able to look and not to act.

**Failing to connect is a failure.** There is no offline mode and no
reconnect loop: an adapter that came up without an app behind it would
publish a tool surface it cannot serve, and a session would discover that
one refused call at a time. Instead this exits non-zero with the reason on
stderr, which is where the client shows it.

**stdout is the protocol.** Logging goes to stderr, always, at the level
``--log-level`` names. See ``README.md`` for the folder standard and
``GLOSSARY.md`` for the **MCP adapter** vocabulary.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from i2as.mcp.adapter import McpAdapter
from i2as.mcp.client import (
    DEFAULT_TIMEOUT_S,
    GatewayClient,
    GatewayError,
    default_descriptor_path,
    read_descriptor,
)
from i2as.mcp.sdk import serve_with_sdk
from i2as.mcp.shim import serve

logger = logging.getLogger("i2as.mcp")

__all__ = ["build_parser", "main"]

#: The framing backends ``--framing`` chooses between. ``auto`` prefers the
#: ``mcp`` package and falls back to the shim; the other two say so
#: explicitly, which is what a bug report needs.
FRAMINGS = ("auto", "shim", "sdk")


def build_parser() -> argparse.ArgumentParser:
    """Build the command line.

    Returns:
        The parser, with every option defaulting from the environment so an
        editor's launcher can configure the adapter without arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m i2as.mcp",
        description=(
            "Serve a running I2AS application's agent gateway to an "
            "external session over the Model Context Protocol."
        ),
    )
    parser.add_argument(
        "--descriptor",
        default=os.environ.get("I2AS_GATEWAY_DESCRIPTOR"),
        help="The gateway.json to read; defaults to the app's log directory.",
    )
    parser.add_argument(
        "--role",
        default=os.environ.get("I2AS_MCP_ROLE", "observer"),
        help=(
            "The role to declare at the handshake: observer (default), debug "
            "or session. The app refuses anything above its own ceiling."
        ),
    )
    parser.add_argument(
        "--actor-id",
        default=os.environ.get("I2AS_MCP_ACTOR_ID", "mcp"),
        help=(
            "The identity stamped on every verdict, run record and agent-feed "
            "entry this session causes. Name the session, not the tool."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("I2AS_MCP_TIMEOUT", DEFAULT_TIMEOUT_S)),
        help="Seconds to wait for one gateway answer.",
    )
    parser.add_argument(
        "--framing",
        choices=FRAMINGS,
        default=os.environ.get("I2AS_MCP_FRAMING", "auto"),
        help="Which stdio backend to use (default: auto).",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("I2AS_MCP_LOG_LEVEL", "INFO"),
        help="Logging level for the diagnostics written to stderr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one adapter process.

    Args:
        argv: The command line, or ``None`` to read the process's own.

    Returns:
        The exit status: 0 when the session ended normally, 1 when the
        running application could not be found or refused this connection.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    descriptor = args.descriptor or default_descriptor_path()
    try:
        published = read_descriptor(descriptor)
    except GatewayError as error:
        logger.error("%s", error)
        return 1

    adapter = McpAdapter(
        GatewayClient(
            published["socket"],
            published["token"],
            role=args.role,
            actor_id=args.actor_id,
            timeout=args.timeout,
        )
    )
    try:
        adapter.open()
    except GatewayError as error:
        logger.error(
            "the I2AS gateway refused this connection: %s "
            "(this app hands out at most the %r role)",
            error,
            published.get("max_role", "observer"),
        )
        return 1

    try:
        if args.framing != "shim" and serve_with_sdk(adapter):
            return 0
        if args.framing == "sdk":
            logger.error("the 'mcp' package cannot frame this session")
            return 1
        serve(adapter)
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":  # pragma: no cover — the process entry point
    sys.exit(main())
