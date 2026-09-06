"""The command grammar of ``python -m i2as.ctl``.

Command grammar is API — an agent harness, a permission allowlist and the
integration tests all hard-code it — so subcommand names and their meanings
stay stable, and the two things a caller branches on are the exit code and
the ``detail.rule`` in the answer, never the prose.

The grammar
-----------

::

    python -m i2as.ctl [connection options] <subcommand> [arguments]

Connection options come BEFORE the subcommand, because they choose which
engine the subcommand runs against: ``--offline <config_dir>`` builds a
simulated station in this process, and without it the client talks to a
running application through the **Request spool**. ``--role`` and ``--actor``
declare the authority and the identity every command is stamped with.

Subcommands are the **Tool surface**, not a list of their own: ``tools``,
``schema`` and ``call`` reach every tool the station publishes, and the rest
are shorthands for the calls a person makes often enough to want a word for
— the reads (``status``, ``station``, ``manifest``, ``runs``, ``feed``) and
the four interventions (``pause``, ``resume``, ``abort``,
``emergency-standby``). A new command or capability therefore needs no code
here: it renders into ``tools`` and is reachable through ``call`` the moment
it is declared.

Output and exit codes
---------------------

One JSON object per invocation, on stdout, always — JSON in and JSON out is
the contract, so nothing has to be parsed out of prose. Three exit codes:

======  ===============================================================
   0    The tool answered and the answer is ``ok``: a ``Verdict`` of
        ``OK``, or a read that succeeded.
   1    The request was reached and REFUSED or FAILED — a ``BLOCKED_*``
        verdict, a schema violation, an unknown tool, an absent
        collaborator. The answer says which in ``detail.rule``.
   2    No engine was reached, so nothing was asked: no spool, a verdict
        that never arrived within ``--timeout``, or a config that would
        not build. Also argparse's own code for a usage error.
======  ===============================================================

Nothing here prompts. Authorisation is the harness's job, and a hung prompt
is the worst failure mode an agent can be given.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from i2as.core.logging_config import setup_logging
from i2as.ctl.client import (
    DEFAULT_SETTLE_TICKS,
    DEFAULT_TIMEOUT_S,
    CtlClient,
    CtlUnreachable,
    default_actor_id,
    open_client,
)
from i2as.session.gateway import Role
from i2as.troubleshoot.cli import resolve_config

logger = logging.getLogger(__name__)

__all__ = ["EXIT_OK", "EXIT_REFUSED", "EXIT_UNREACHABLE", "build_parser", "main"]

#: The tool answered and the answer is ``ok``.
EXIT_OK = 0

#: The request was reached and refused, or failed.
EXIT_REFUSED = 1

#: No engine was reached, so nothing was asked. Also argparse's usage code.
EXIT_UNREACHABLE = 2


# ══════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════


def _print(payload: dict[str, Any]) -> None:
    """Write one answer to stdout as JSON.

    Args:
        payload: The answer. Anything JSON cannot render is written as its
            ``repr()`` rather than failing the invocation.
    """
    sys.stdout.write(json.dumps(payload, indent=2, default=repr) + "\n")


def _stamped(client: CtlClient, payload: dict[str, Any]) -> dict[str, Any]:
    """Add the connection's identity to one answer.

    Every answer says which engine answered it and under whose authority,
    because a transcript of ctl invocations has to be readable without the
    argv that produced it.

    Args:
        client: The client that answered.
        payload: The answer.

    Returns:
        The answer with ``mode``, ``role`` and ``actor`` in front.
    """
    return {
        "mode": client.mode,
        "role": client.role.value,
        "actor": client.actor_id,
        **payload,
    }


def _failure(reason: str, rule: str, **detail: Any) -> dict[str, Any]:
    """Build the answer shape for something this client refused itself.

    Args:
        reason: The operator-readable explanation.
        rule: The machine-readable rule name, for ``detail.rule``.
        **detail: Anything else worth naming in ``detail``.

    Returns:
        The answer dict.
    """
    return {
        "ok": False,
        "code": "FAILED",
        "reason": reason,
        "detail": {"rule": rule, **detail},
    }


# ══════════════════════════════════════════════════════════════════════════
# Subcommands
# ══════════════════════════════════════════════════════════════════════════


def _call(
    client: CtlClient, tool: str, args: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    """Call one tool and turn its answer into an exit code and a payload.

    The single path every acting subcommand takes, so a shorthand and the
    equivalent ``call`` differ in nothing but the words typed.

    Args:
        client: The open client.
        tool: The tool's name.
        args: Its arguments.

    Returns:
        ``(exit code, answer)``.
    """
    answer = client.call(tool, args or {})
    return (EXIT_OK if answer.get("ok") else EXIT_REFUSED), _stamped(client, answer)


def _cmd_tools(args: argparse.Namespace, client: CtlClient) -> tuple[int, dict[str, Any]]:
    """List the tool surface this station publishes.

    Args:
        args: Parsed arguments; ``--schemas`` asks for the full JSON Schemas.
        client: The open client.

    Returns:
        ``(exit code, answer)``.
    """
    gateway = client.gateway
    if args.schemas:
        listing: list[dict[str, Any]] = gateway.tool_schemas()
    else:
        listing = [
            {
                "name": tool.name,
                "action_class": tool.action_class.value,
                "description": tool.description,
            }
            for tool in gateway.tools()
        ]
    return EXIT_OK, _stamped(
        client, {"ok": True, "code": "OK", "count": len(listing), "tools": listing}
    )


def _cmd_schema(
    args: argparse.Namespace, client: CtlClient
) -> tuple[int, dict[str, Any]]:
    """Show one tool's declaration: its description, schema and action class.

    Args:
        args: Parsed arguments; ``tool`` names the tool.
        client: The open client.

    Returns:
        ``(exit code, answer)``.
    """
    tool = client.gateway.tool(args.tool)
    if tool is None:
        return EXIT_REFUSED, _stamped(
            client,
            _failure(
                f"no tool named {args.tool!r}; `tools` lists this station's "
                f"whole surface",
                "unknown_tool",
                tool=args.tool,
            ),
        )
    return EXIT_OK, _stamped(
        client,
        {
            "ok": True,
            "code": "OK",
            "tool": tool.name,
            "action_class": tool.action_class.value,
            "permitted": client.gateway.permits(tool.command, tool.fixed_args) is None
            if tool.command is not None
            else None,
            "schema": tool.to_schema(),
        },
    )


def _cmd_call(args: argparse.Namespace, client: CtlClient) -> tuple[int, dict[str, Any]]:
    """Call any tool by name, with JSON arguments.

    Args:
        args: Parsed arguments; ``tool`` and ``--args``.
        client: The open client.

    Returns:
        ``(exit code, answer)``.
    """
    try:
        tool_args = _read_args(args.args)
    except ValueError as exc:
        return EXIT_REFUSED, _stamped(client, _failure(str(exc), "args"))
    return _call(client, args.tool, tool_args)


def _cmd_status(_args: argparse.Namespace, client: CtlClient) -> tuple[int, dict[str, Any]]:
    """Read the engine's latest status snapshot.

    Args:
        _args: Parsed arguments (unused).
        client: The open client.

    Returns:
        ``(exit code, answer)``.
    """
    return _call(client, "read_status")


def _cmd_station(
    _args: argparse.Namespace, client: CtlClient
) -> tuple[int, dict[str, Any]]:
    """Read the station's declaration snapshot.

    Args:
        _args: Parsed arguments (unused).
        client: The open client.

    Returns:
        ``(exit code, answer)``.
    """
    return _call(client, "read_station_info")


def _cmd_manifest(
    _args: argparse.Namespace, client: CtlClient
) -> tuple[int, dict[str, Any]]:
    """Read the capability manifest.

    Args:
        _args: Parsed arguments (unused).
        client: The open client.

    Returns:
        ``(exit code, answer)``.
    """
    return _call(client, "read_manifest")


def _cmd_runs(args: argparse.Namespace, client: CtlClient) -> tuple[int, dict[str, Any]]:
    """List the runs recorded in an experiment.

    Args:
        args: Parsed arguments; ``--experiment`` selects one.
        client: The open client.

    Returns:
        ``(exit code, answer)``.
    """
    return _call(client, "list_runs", _optional(experiment_id=args.experiment))


def _cmd_feed(args: argparse.Namespace, client: CtlClient) -> tuple[int, dict[str, Any]]:
    """Read the tail of an experiment's **Agent feed**.

    Args:
        args: Parsed arguments; ``--experiment`` and ``--last``.
        client: The open client.

    Returns:
        ``(exit code, answer)``.
    """
    return _call(
        client,
        "read_agent_feed",
        _optional(experiment_id=args.experiment, last=args.last),
    )


def _cmd_pause(_args: argparse.Namespace, client: CtlClient) -> tuple[int, dict[str, Any]]:
    """Pause the running procedure.

    Args:
        _args: Parsed arguments (unused).
        client: The open client.

    Returns:
        ``(exit code, answer)``.
    """
    return _call(client, "pause_procedure")


def _cmd_resume(
    _args: argparse.Namespace, client: CtlClient
) -> tuple[int, dict[str, Any]]:
    """Resume the paused procedure.

    Args:
        _args: Parsed arguments (unused).
        client: The open client.

    Returns:
        ``(exit code, answer)``.
    """
    return _call(client, "resume_procedure")


def _cmd_abort(_args: argparse.Namespace, client: CtlClient) -> tuple[int, dict[str, Any]]:
    """Abort the running procedure.

    Args:
        _args: Parsed arguments (unused).
        client: The open client.

    Returns:
        ``(exit code, answer)``.
    """
    return _call(client, "abort_procedure")


def _cmd_emergency_standby(
    args: argparse.Namespace, client: CtlClient
) -> tuple[int, dict[str, Any]]:
    """Take the station to a safe state, whatever it is doing.

    Permitted to every role, in every state, at every kill-switch setting:
    an actor that can see a problem must never be unable to make the station
    safe. The reason is required because it is what the record will say.

    Args:
        args: Parsed arguments; ``--reason``.
        client: The open client.

    Returns:
        ``(exit code, answer)``.
    """
    return _call(client, "emergency_standby", {"reason": args.reason})


def _optional(**values: Any) -> dict[str, Any]:
    """Drop the arguments the caller did not give.

    The tool schemas are closed and every declared parameter is required, so
    an omitted option must not travel as ``null``.

    Args:
        **values: Candidate arguments.

    Returns:
        Only the ones that are neither ``None`` nor empty.
    """
    return {key: value for key, value in values.items() if value not in (None, "")}


def _read_args(raw: str | None) -> dict[str, Any]:
    """Read a tool's arguments from the command line, a file, or stdin.

    Args:
        raw: The ``--args`` value: a JSON object, ``@path`` to read one from a
            file, ``-`` to read one from stdin, or ``None`` for no arguments.

    Returns:
        The arguments.

    Raises:
        ValueError: If the value is not a readable JSON object.
    """
    if raw is None:
        return {}
    text = raw
    if raw == "-":
        text = sys.stdin.read()
    elif raw.startswith("@"):
        try:
            with open(raw[1:], encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            raise ValueError(f"could not read arguments from {raw[1:]!r}: {exc}") from exc
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"--args is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("--args must be a JSON object of the tool's parameters")
    return payload


# ══════════════════════════════════════════════════════════════════════════
# The parser
# ══════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse command tree.

    Kept separate from :func:`main` so ``--help`` and the grammar itself can
    be tested without opening a client.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m i2as.ctl",
        description=(
            "I2AS reference client: drive the instrument through the agent "
            "gateway's tool surface, JSON in and JSON out."
        ),
    )
    parser.add_argument(
        "--offline",
        metavar="CONFIG",
        help=(
            "build a simulated station from this config directory (or shipped "
            "config name) in this process, instead of talking to a running "
            "application"
        ),
    )
    parser.add_argument(
        "--role",
        choices=[role.value for role in Role],
        default=Role.OBSERVER.value,
        help="the authority this connection declares (default: observer)",
    )
    parser.add_argument(
        "--actor",
        default="",
        help=f"the identity every command is stamped with (default: {default_actor_id()})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        metavar="SECONDS",
        help="how long a live request waits for its verdict (default: %(default)s)",
    )
    parser.add_argument(
        "--spool",
        metavar="DIR",
        help="the request spool to write into (default: the installation's)",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=DEFAULT_SETTLE_TICKS,
        help="offline: how many engine ticks each call runs (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p = sub.add_parser("tools", help="list every tool this station publishes")
    p.add_argument(
        "--schemas", action="store_true", help="full JSON Schemas, not just names"
    )
    p.set_defaults(func=_cmd_tools)

    p = sub.add_parser("schema", help="show one tool's schema and action class")
    p.add_argument("tool", help="tool name, as listed by `tools`")
    p.set_defaults(func=_cmd_schema)

    p = sub.add_parser("call", help="call any tool by name")
    p.add_argument("tool", help="tool name, as listed by `tools`")
    p.add_argument(
        "--args",
        metavar="JSON",
        help="the tool's arguments as a JSON object, @file, or - for stdin",
    )
    p.set_defaults(func=_cmd_call)

    p = sub.add_parser("status", help="the engine's latest status snapshot")
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("station", help="the station's declaration snapshot")
    p.set_defaults(func=_cmd_station)

    p = sub.add_parser("manifest", help="the capability manifest")
    p.set_defaults(func=_cmd_manifest)

    p = sub.add_parser("runs", help="the runs recorded in an experiment")
    p.add_argument("--experiment", default="", help="experiment id (default: the open one)")
    p.set_defaults(func=_cmd_runs)

    p = sub.add_parser("feed", help="the tail of an experiment's agent feed")
    p.add_argument("--experiment", default="", help="experiment id (default: the open one)")
    p.add_argument("--last", type=int, default=None, help="how many records to read")
    p.set_defaults(func=_cmd_feed)

    p = sub.add_parser("pause", help="pause the running procedure")
    p.set_defaults(func=_cmd_pause)

    p = sub.add_parser("resume", help="resume the paused procedure")
    p.set_defaults(func=_cmd_resume)

    p = sub.add_parser("abort", help="abort the running procedure")
    p.set_defaults(func=_cmd_abort)

    p = sub.add_parser(
        "emergency-standby", help="take the station to a safe state, from any state"
    )
    p.add_argument("--reason", required=True, help="why — it goes into the record")
    p.set_defaults(func=_cmd_emergency_standby)

    return parser


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None, *, client: CtlClient | None = None) -> int:
    """Run one ctl command and print its answer.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``). Exposed so tests
            drive the CLI in-process.
        client: An already-open client to run against, instead of opening
            one from the connection options. Exposed so several invocations
            can be driven against ONE in-process stack — a scenario is a
            sequence of commands against a station that remembers what the
            last one did, which a fresh station per invocation could not be.
            A client passed in is not closed here; its owner closes it.

    Returns:
        ``0`` when the answer is ``ok``, ``1`` when it was refused or failed,
        ``2`` when no engine could be reached.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    setup_logging()
    args = build_parser().parse_args(argv)

    owned = client is None
    if client is None:
        try:
            client = open_client(
                offline=resolve_config(args.offline) if args.offline else None,
                role=args.role,
                actor_id=args.actor,
                timeout_s=args.timeout,
                spool_root=args.spool,
                settle_ticks=args.ticks,
            )
        except (CtlUnreachable, SystemExit) as exc:
            # SystemExit is how the config resolver refuses a name it cannot
            # find; it is the same failure as an unreachable engine — nothing
            # was asked — and must leave by the same door, as one JSON answer.
            logger.error("ctl %s: %s", args.subcommand, exc)
            _print(_failure(str(exc), "unreachable"))
            return EXIT_UNREACHABLE

    try:
        code, payload = args.func(args, client)
    except CtlUnreachable as exc:
        logger.error("ctl %s: %s", args.subcommand, exc)
        code, payload = EXIT_UNREACHABLE, _stamped(
            client, _failure(str(exc), "unreachable")
        )
    except Exception as exc:  # noqa: BLE001 — one place turns any failure into an answer
        logger.exception("ctl %s failed", args.subcommand)
        code, payload = EXIT_REFUSED, _stamped(
            client,
            _failure(f"{type(exc).__name__}: {exc}", "unexpected_error"),
        )
    finally:
        if owned:
            client.close()

    _print(payload)
    return code
