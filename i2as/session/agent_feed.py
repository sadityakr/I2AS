"""The **Agent feed** — one experiment's accountability trail (L6).

**An autonomous actor must leave a trail a physicist can read months later.**
The control contract already names an ``Actor`` on every ``Command``,
``Verdict`` and ``Event``, but those live in memory and vanish with the
process. This module writes the non-operator half of that traffic to
``agent_actions.jsonl`` inside the experiment's own folder, so copying the
folder copies the evidence with it — the same rule the **Outbox** follows.

The record standard
-------------------

One JSON object per line, appended, never rewritten. Unlike the maintenance
log and the Outbox — journals of *entities*, where the last line naming an
id wins — this is a journal of *facts*: every line is a distinct thing that
happened, ordered by its ``seq``, and nothing here ever supersedes an
earlier line. A reader that wants the story reads the file in order.

Every record carries every key below; a value that does not apply to this
record kind is ``null``, never a missing key (the same rule
``core.operational_status`` follows, and for the same reason: a reader must
never have to guess whether an absent key means "no" or "old file").

======================= ============== =========================================
Field                   Type           Meaning
======================= ============== =========================================
``schema``              int            Record schema version, ``SCHEMA_VERSION``.
``ts``                  float          Epoch seconds the record was written.
``seq``                 int            Per-file counter, from 1, strictly increasing.
``experiment_id``       str            The owning experiment's store key.
``run_id``              str | null     The run in flight, when one is.
``record``              str            ``"command"``, ``"verdict"``, ``"event"``
                                       or ``"tool"``.
``actor``               obj            ``Actor.to_json()`` — kind, id, role.
``request_id``          str            The correlation id the two trails join on.
``command``             str | null     ``CommandName`` value; null on an event record.
``tool``                str | null     The **Tool spec**'s name; tool records only.
``args``                obj | null     The arguments; command and tool records only.
``event``               str | null     The event's ``kind``; event records only.
``detail``              obj | null     The event's payload, or a verdict's or a
                                       tool answer's ``detail``.
``verdict``             obj | null     ``{"code", "reason"}``; verdict and tool
                                       records only.
======================= ============== =========================================

What is recorded, and by whom
-----------------------------

* **Commands** are recorded by whoever submits them — the **Agent gateway**
  calls ``record_command()`` from its own ``submit()``, which is the only
  place the arguments are still in hand.
* **Tool calls that spend or change something** are recorded by the gateway
  as it answers them (``record_tool_call()``), because a session tool is
  answered inside the client rather than by the engine and so has no verdict
  record to be found under. WHICH calls those are is declared on the tool
  itself (``ToolSpec.recorded``), never decided by a branch here: a ``read``
  tool an agent polls every tick would drown the trail in observations,
  while a tool that queues a notebook entry or spends model tokens is
  exactly what this file exists to remember. A tool record carries no
  ``request_id`` — there is no engine request to join to — and its
  ``detail`` carries what the call cost when it cost anything.
* **Verdicts** and **state changes** are recorded off the engine's own
  ``verdict_emitted`` / ``event_emitted`` streams (``attach()``), so an
  actor that reaches the engine WITHOUT passing through the gateway still
  leaves a trail: the actor kind travels on the message itself, not on the
  connection. Such an action has a verdict record naming its actor, command
  and request id, but no command record — nothing invents one, because the
  arguments were never seen here.
* **Operator traffic is not recorded at all.** This file is the answer to
  "what did the machines do"; the physicist's own actions are the rest of
  the session record (``experiment.json``, the maintenance log, the
  operational-status log), and mixing the two would make the trail useless
  for the question it exists to answer.

Recording never raises into its caller. It sits on signal handlers in the
engine's emit path, so a full disk or a read-only folder must degrade to a
logged warning, never disturb a running measurement.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Protocol

from i2as.core.events import (
    ActorKind,
    Command,
    RunFinished,
    RunStarted,
    StateChange,
    Verdict,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SCHEMA_VERSION",
    "RECORD_COMMAND",
    "RECORD_VERDICT",
    "RECORD_EVENT",
    "RECORD_TOOL",
    "AgentFeed",
    "read_feed",
]

#: Version of the record shape documented above. Bump it when the shape
#: changes; adding a field is a bump, renaming or retyping one is forbidden.
#: Version 2 added ``tool`` and the ``"tool"`` record kind; every version-1
#: field kept its name and its type, so a version-1 reader still reads a
#: version-2 file and simply meets a record kind it does not know.
SCHEMA_VERSION = 2

#: A command a non-operator actor submitted, with its arguments.
RECORD_COMMAND = "command"

#: The engine's single answer to one such command.
RECORD_VERDICT = "verdict"

#: A state change an agent caused.
RECORD_EVENT = "event"

#: A **Tool spec** call the gateway answered itself, and what it answered.
RECORD_TOOL = "tool"


class _Signal(Protocol):
    """The one method this module needs from an engine signal."""

    def connect(self, slot: Any) -> Any:
        """Subscribe *slot* to this signal."""
        ...


class _Engine(Protocol):
    """The engine surface a feed attaches to — the two broadcast streams.

    Duck-typed exactly as ``gateway.EngineClient`` is, so this module needs
    no Qt import and a transport proxy substitutes for the Orchestrator.

    Attributes:
        verdict_emitted: One ``Verdict`` per submitted command.
        event_emitted: The engine's one event stream.
    """

    verdict_emitted: _Signal
    event_emitted: _Signal


def _append_line(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON line to ``path``, creating parent directories.

    Args:
        path: The JSONL file to append to.
        payload: JSON-serialisable object for the new line.

    Raises:
        OSError: If the directory cannot be created or the file written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_feed(
    path: str | Path, since_seq: int | None = None
) -> list[dict[str, Any]]:
    """Read an agent feed, oldest first, tolerating a corrupt line.

    The read side of the record standard. A line that is not readable JSON,
    or does not hold an object, is skipped with a WARNING rather than
    failing the read — one mangled line must never strand the rest of the
    trail, exactly as in ``store.py`` and ``maintenance_log.py``.

    Args:
        path: The feed file, normally from
            ``ExperimentStore.agent_feed_path()``.
        since_seq: When given, only records with a ``seq`` strictly greater
            than this are returned — how a client polls the trail without
            re-reading what it already has. A record whose ``seq`` is not an
            int is kept, since it cannot be compared away.

    Returns:
        One JSON-safe dict per well-formed record, in file order. ``[]``
        when the file does not exist or cannot be read.
    """
    file_path = Path(path)
    try:
        raw = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("Could not read the agent feed %s: %s", file_path, exc)
        return []

    records: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            logger.warning("Skipping corrupt agent-feed line %s:%d", file_path, number)
            continue
        if not isinstance(payload, dict):
            logger.warning("Skipping non-object agent-feed line %s:%d", file_path, number)
            continue
        if since_seq is not None:
            seq = payload.get("seq")
            if isinstance(seq, int) and not isinstance(seq, bool) and seq <= since_seq:
                continue
        records.append(payload)
    return records


class AgentFeed:
    """The append-only action trail for one experiment.

    One instance per experiment folder. Cheap to construct — the file is
    read once, on the first write, to pick the sequence number up where an
    earlier process left it, and is never held open.

    Attributes:
        experiment_id: The experiment this feed belongs to, stamped on every
            record.
    """

    def __init__(self, path: str | Path, experiment_id: str = "") -> None:
        """Bind to one feed file.

        Args:
            path: The journal file, from
                ``ExperimentStore.agent_feed_path()`` — the store owns where
                it sits, this class owns what is in it. Neither it nor its
                parent is created until something is actually recorded.
            experiment_id: The owning experiment's store key, stamped on
                every record so a feed copied out of its folder still says
                what it belongs to.
        """
        self._path = Path(path)
        self.experiment_id = str(experiment_id)
        self._seq: int | None = None
        self._run_id: str | None = None
        # Request ids already given a command record, so a command recorded
        # by the gateway and then seen again (a re-submit of the same
        # Command object, a second listener) appends once and only once.
        self._recorded_commands: set[str] = set()

    @property
    def path(self) -> Path:
        """The journal file this feed reads and appends to."""
        return self._path

    @property
    def run_id(self) -> str | None:
        """The run currently stamped on new records, or ``None``."""
        return self._run_id

    def set_run_id(self, run_id: str | None) -> None:
        """Set the run stamped on subsequent records.

        Maintained automatically from ``RunStarted``/``RunFinished`` once
        ``attach()`` has been called; exposed for a caller that drives the
        feed without an engine (a test, a replay).

        Args:
            run_id: The active run's manifest id, or ``None`` when idle.
        """
        self._run_id = str(run_id) if run_id else None

    # ── Wiring ────────────────────────────────────────────────────────

    def attach(self, engine: _Engine) -> None:
        """Subscribe to an engine's verdict and event streams.

        This is what makes the trail complete: the gateway supplies the
        command records (it alone still holds the arguments), and everything
        the engine says about a non-operator actor is picked up here,
        whether or not it came through a gateway.

        Args:
            engine: Anything exposing ``verdict_emitted`` and
                ``event_emitted`` — the Orchestrator today, a transport
                proxy later.
        """
        engine.verdict_emitted.connect(self.record_verdict)
        engine.event_emitted.connect(self.record_event)
        logger.info(
            "Agent feed attached for experiment %r: %s",
            self.experiment_id,
            self._path,
        )

    # ── Recording ─────────────────────────────────────────────────────

    def record_command(self, command: Command) -> None:
        """Record one command submitted by a non-operator actor.

        A no-op for an ``operator`` actor, and for a request id that already
        has a command record in this process.

        Args:
            command: The command as submitted, arguments included.
        """
        try:
            if command.actor.kind is ActorKind.OPERATOR:
                return
            if command.request_id in self._recorded_commands:
                return
            self._recorded_commands.add(command.request_id)
            self._append(
                record=RECORD_COMMAND,
                actor=command.actor.to_json(),
                request_id=command.request_id,
                command=command.name.value,
                args=dict(command.args),
            )
        except Exception:  # noqa: BLE001 — recording must never disturb a client
            logger.exception("agent feed: recording a command failed (non-fatal)")

    def record_tool_call(
        self,
        actor: Any,
        tool: str,
        args: dict[str, Any] | None = None,
        detail: dict[str, Any] | None = None,
        verdict: dict[str, Any] | None = None,
    ) -> None:
        """Record one tool call the gateway answered itself.

        A no-op for an ``operator`` actor, like every other recording method
        here. Called only for a tool that declares ``ToolSpec.recorded`` —
        see the record standard at the top of this module for why that is a
        declaration on the tool rather than a rule here.

        Args:
            actor: The ``Actor`` that called it.
            tool: The tool's name, as published by the **Tool surface**.
            args: The arguments it was called with.
            detail: The structured half of the answer, including what the
                call cost when it cost anything.
            verdict: ``{"code", "reason"}`` — the answer, in the same two
                fields a command's verdict is recorded under, so one reader
                reads both.
        """
        try:
            if getattr(actor, "kind", None) is ActorKind.OPERATOR:
                return
            self._append(
                record=RECORD_TOOL,
                actor=actor.to_json(),
                request_id="",
                tool=tool,
                args=dict(args or {}),
                detail=dict(detail) if detail else None,
                verdict=dict(verdict) if verdict else None,
            )
        except Exception:  # noqa: BLE001 — recording must never disturb a client
            logger.exception("agent feed: recording a tool call failed (non-fatal)")

    def record_verdict(self, verdict: Verdict) -> None:
        """Record the engine's answer to a non-operator actor's command.

        A no-op for an ``operator`` actor. Joined to its command record — if
        there is one — by ``request_id``.

        Args:
            verdict: Any verdict off the engine's stream.
        """
        try:
            if verdict.actor.kind is ActorKind.OPERATOR:
                return
            self._append(
                record=RECORD_VERDICT,
                actor=verdict.actor.to_json(),
                request_id=verdict.request_id,
                command=verdict.command.value,
                detail=dict(verdict.detail) if verdict.detail else None,
                verdict={"code": verdict.code.value, "reason": verdict.reason},
            )
        except Exception:  # noqa: BLE001 — recording must never disturb the engine
            logger.exception("agent feed: recording a verdict failed (non-fatal)")

    def record_event(self, event: Any) -> None:
        """Record one engine event, when it is one this trail carries.

        Two things happen here. ``RunStarted``/``RunFinished`` maintain the
        ``run_id`` every subsequent record is stamped with — the feed's own
        answer to "which run was this during" — regardless of who caused
        them. A ``StateChange`` whose actor is an ``agent`` is written as a
        record: the state machine moving because an autonomous actor asked
        is exactly the consequence a dispute is about.

        Args:
            event: Anything on the engine's event stream; anything else is
                ignored.
        """
        try:
            if isinstance(event, RunStarted):
                self.set_run_id(event.run_id)
            elif isinstance(event, RunFinished):
                self.set_run_id(None)
            if not isinstance(event, StateChange):
                return
            if event.actor.kind is not ActorKind.AGENT:
                return
            self._append(
                record=RECORD_EVENT,
                actor=event.actor.to_json(),
                request_id=event.request_id,
                event=StateChange.kind,
                detail={
                    "state": event.state,
                    "previous": event.previous,
                    "cause": event.cause,
                },
            )
        except Exception:  # noqa: BLE001 — recording must never disturb the engine
            logger.exception("agent feed: recording an event failed (non-fatal)")

    # ── The one write path ────────────────────────────────────────────

    def _append(
        self,
        *,
        record: str,
        actor: dict[str, Any],
        request_id: str,
        command: str | None = None,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
        event: str | None = None,
        detail: dict[str, Any] | None = None,
        verdict: dict[str, Any] | None = None,
    ) -> None:
        """Write one record, filling in every key of the standard.

        Args:
            record: Which record kind this is (``RECORD_*``).
            actor: The acting ``Actor``'s JSON form.
            request_id: The correlation id both trails join on.
            command: The ``CommandName`` value, where one applies.
            tool: The tool's name, on a tool record.
            args: The arguments, on a command or a tool record.
            event: The event's ``kind``, on an event record.
            detail: The event's payload, or the verdict's ``detail``.
            verdict: ``{"code", "reason"}``, on a verdict record.

        Raises:
            OSError: If the line cannot be written (caught by the public
                recording methods, which never raise at their caller).
        """
        _append_line(
            self._path,
            {
                "schema": SCHEMA_VERSION,
                "ts": time.time(),
                "seq": self._next_seq(),
                "experiment_id": self.experiment_id,
                "run_id": self._run_id,
                "record": record,
                "actor": actor,
                "request_id": request_id,
                "command": command,
                "tool": tool,
                "args": args,
                "event": event,
                "detail": detail,
                "verdict": verdict,
            },
        )

    def _next_seq(self) -> int:
        """Return the sequence number for the next record.

        Seeded once, from the highest ``seq`` already in the file, so a feed
        continued by a later process keeps counting instead of restarting at
        1 and making the trail unorderable.

        Returns:
            A number strictly greater than every ``seq`` written so far.
        """
        if self._seq is None:
            self._seq = max(
                (
                    payload["seq"]
                    for payload in read_feed(self._path)
                    if isinstance(payload.get("seq"), int)
                    and not isinstance(payload.get("seq"), bool)
                ),
                default=0,
            )
        self._seq += 1
        return self._seq
