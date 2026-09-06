"""The in-process client: one connection, one role, one actor identity.

``Gateway`` is what an autonomous client actually holds. It is deliberately
thin, because being thin is the design: the engine has two clients and they
must see the same system and be seen doing the same things, so this object
adds a permission check and an identity to the **Control contract** and
NOTHING else. It defines no vocabulary of its own, holds no instrument, and
opens no socket — in-process, no network, no thread.

Three jobs:

* **Stamp the identity.** Every command leaves here as
  ``Actor(kind="agent", id=..., role=...)``, so a verdict, a run record and
  the status bar all name the authority the action was taken under.
* **Authorize before forwarding.** ``roles.authorize()`` decides — the role
  matrix, the kill switch, **Attendance**, and the **run-ownership
  standard** judged against the owner this connection mirrors; a refusal
  is answered as a ``BLOCKED_ROLE`` verdict on the engine's OWN
  ``verdict_emitted`` stream, so the human's window sees the agent being
  refused exactly as it sees it being obeyed. A refusal never reaches the
  engine at all.
* **Leave a trail.** Every command this connection submits — forwarded or
  refused here — is written to the experiment's **Agent feed** when one is
  given, which is the only place the arguments are still in hand. The
  answering verdict is recorded by the feed itself, off the engine's own
  stream, so the two halves join on ``request_id``.
* **Mirror, never poll.** Every read is answered from the latest
  ``StatusSnapshot`` / ``StationInfo`` the engine broadcast, following the
  verdict standard's rule that a client answers reads locally and never
  calls into the engine for them. **Attendance** and the **Kill switch**
  come off that same mirror, which is what makes an agent's permission check
  read the same fact every other client sees.
* **Offer the rendered surface.** ``tools()`` / ``tool_schemas()`` publish
  the **Tool surface** ``tools.py`` renders from the same declarations, and
  ``call_tool()`` is the one entry point that validates a call against its
  **Tool spec**'s schema and then routes it — a command tool through
  ``submit()``, a session tool to its function. It never raises at its
  caller: an unknown tool, a schema violation and a missing collaborator are
  all answered with the same ``FAILED``-shaped dict, so an agent gets an
  answer to every call exactly as it gets a verdict for every command.

The engine is duck-typed on purpose (see ``EngineClient``): the Orchestrator
itself when a caller is on the instrument thread, and the
**Orchestrator proxy** when it is not — which, under the single hardware
thread standard, is every caller on the GUI thread. ``verdict_stream()`` and
``event_stream()`` are what make the two interchangeable: the engine declares
``verdict_emitted``/``event_emitted`` and a client adapter consumes them under
the contract's own names, ``verdict``/``event``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Protocol

from i2as.core.events import (
    Actor,
    ActorKind,
    AgentGate,
    Command,
    CommandName,
    StationInfo,
    StatusSnapshot,
    Verdict,
    VerdictCode,
)
from i2as.session.gateway.action_classes import ActionClass
from i2as.session.gateway.roles import (
    PERMISSION_MATRIX,
    Permission,
    Role,
    authorize,
)
from i2as.session.gateway.tools import (
    ToolContext,
    ToolError,
    ToolSpec,
    call_session_tool,
    feed_arguments,
    render_tools,
    validate_tool_args,
)
from i2as.session.agent_feed import AgentFeed
from i2as.session.eln.drafting import cost_line

logger = logging.getLogger(__name__)


class EngineClient(Protocol):
    """The engine surface a gateway needs — the control contract, nothing more.

    A ``Protocol`` rather than the Orchestrator itself, because the whole
    point of the contract is that a second implementation (the
    **Orchestrator proxy**, and a transport proxy later) is a drop-in. Both
    signals are duck-typed as objects with ``connect``/``emit``, so this
    module needs no Qt import, and both are reached through
    ``verdict_stream()``/``event_stream()`` rather than by attribute, because
    a client adapter consumes the two channels under the contract's own names
    (``verdict``/``event``) while the engine declares them as
    ``verdict_emitted``/``event_emitted``.

    Attributes:
        verdict_emitted: The stream carrying one ``Verdict`` per submitted
            command, named ``verdict`` on a client adapter.
        event_emitted: The engine's one event stream, named ``event`` on a
            client adapter.
    """

    verdict_emitted: Any
    event_emitted: Any

    def submit(self, command: Command) -> str:
        """Carry out one command and answer it with one verdict."""
        ...


#: The two names each contract stream answers to: what the engine declares,
#: then what a client adapter re-exposes it as. Same idiom as
#: ``StatusMirror.of()``'s lookup — ask for what the object offers rather
#: than requiring every client to know which of the two it is holding.
_STREAM_NAMES: dict[str, tuple[str, str]] = {
    "verdict": ("verdict_emitted", "verdict"),
    "event": ("event_emitted", "event"),
}


def _stream(engine: Any, channel: str) -> Any:
    """Return one contract stream of *engine*, under whichever name it uses.

    Args:
        engine: Anything satisfying ``EngineClient`` — the Orchestrator, an
            **Orchestrator proxy**, or a stand-in in a test.
        channel: ``"verdict"`` or ``"event"``.

    Returns:
        The signal-like object with ``connect``/``emit``.

    Raises:
        AttributeError: If the object offers neither name, which means it is
            not an engine client at all and would otherwise fail later, on a
            connection nobody made.
    """
    for name in _STREAM_NAMES[channel]:
        stream = getattr(engine, name, None)
        if stream is not None:
            return stream
    raise AttributeError(
        f"{type(engine).__name__} is not an engine client: it offers neither "
        f"of the {channel} stream's names "
        f"({' nor '.join(_STREAM_NAMES[channel])})"
    )


def verdict_stream(engine: Any) -> Any:
    """Return *engine*'s verdict stream.

    Args:
        engine: Anything satisfying ``EngineClient``.

    Returns:
        ``verdict_emitted`` on an engine, ``verdict`` on a client adapter.
    """
    return _stream(engine, "verdict")


def event_stream(engine: Any) -> Any:
    """Return *engine*'s event stream.

    Args:
        engine: Anything satisfying ``EngineClient``.

    Returns:
        ``event_emitted`` on an engine, ``event`` on a client adapter.
    """
    return _stream(engine, "event")


class Gateway:
    """One authorised, identified connection to the engine.

    Attributes:
        role: The authority this connection declared for itself.
        actor: The ``Actor`` stamped on every command it submits.
    """

    def __init__(
        self,
        engine: EngineClient,
        role: Role | str,
        actor_id: str,
        *,
        station_info: StationInfo | Any | None = None,
        tool_context: ToolContext | None = None,
        feed: AgentFeed | None = None,
    ) -> None:
        """Attach to an engine under a declared role.

        Args:
            engine: Anything satisfying ``EngineClient`` — the
                **Orchestrator proxy** for a caller on the GUI thread, the
                Orchestrator itself for one on the instrument thread.
            role: The ``Role`` this connection acts under, as the enum member
                or its string value.
            actor_id: A stable identifier for the client — an agent
                deployment name, a CLI invocation, a subagent's name. It goes
                into every verdict and every event this connection causes.
            station_info: The station's declaration snapshot, or a zero-
                argument callable returning it (e.g. ``station.station_info``).
                Optional: when omitted, the engine's own ``station_info()`` is
                used if it has one, and in either case the mirror is refreshed
                from every ``StationInfo`` the engine broadcasts. It is needed
                because a ``submit_vi_action``'s action class depends on the
                VI kind of the instrument it targets.
            tool_context: The collaborators the session tools read through —
                the experiment layer and the run catalog. Optional: without
                it every command tool still works and the three
                declaration-reading tools still answer from the mirror, while
                a tool whose collaborator is absent is refused by name.
            feed: The experiment's **Agent feed**, or ``None`` to record
                nothing. Given one, every command this connection submits is
                written to it before it is forwarded or refused. Attaching
                the same feed to the engine (``AgentFeed.attach()``) is what
                records the answering verdicts and the state changes; the
                two are separate because only this side ever sees a
                command's arguments.

        Raises:
            ValueError: If *role* is not a known ``Role``.
        """
        self._engine = engine
        self._feed = feed
        self.role = Role(role)
        self.actor = Actor(kind=ActorKind.AGENT, id=str(actor_id), role=self.role.value)

        self._status: StatusSnapshot | None = None
        self._station_info: StationInfo = self._initial_station_info(
            engine, station_info
        )
        # Sequence numbers: the highest the engine has been seen to use, and
        # this gateway's own counter for a refusal the engine never sees. The
        # local counter is kept at or above the engine's so a refusal is never
        # ordered before something that happened earlier.
        self._engine_seq = self._station_info.seq
        self._local_seq = self._engine_seq

        # The rendered tool surface, rebuilt whenever the mirrored
        # declaration is replaced (a connect or a disconnect rebuilds it), and
        # the one verdict a tool call is waiting on.
        self._tools: dict[str, ToolSpec] = {}
        self._tools_rendered_from: StationInfo | None = None
        self._awaiting_verdict = False
        self._awaited_verdicts: list[Verdict] = []
        # The context is this connection's: its own mirrors, and its own
        # identity, so a session tool that leaves something behind on disk
        # (an agent-written **Analysis recipe**) can stamp who left it
        # without being handed the actor call by call.
        self._tool_context = replace(
            tool_context or ToolContext(),
            status_source=self.status,
            station_source=self.station,
            actor=self.actor,
        )

        event_stream(engine).connect(self._observe_event)
        verdict_stream(engine).connect(self._observe_verdict)
        logger.info(
            "Gateway attached: actor %r under role %r", self.actor.id, self.role.value
        )

    # ── Construction helpers ──────────────────────────────────────────

    @staticmethod
    def _initial_station_info(
        engine: EngineClient, station_info: StationInfo | Any | None
    ) -> StationInfo:
        """Resolve the declaration snapshot to start the mirror from.

        Args:
            engine: The engine being attached to.
            station_info: The caller's snapshot, a callable returning one, or
                ``None``.

        Returns:
            The snapshot to start with — an empty ``StationInfo`` when
            neither the caller nor the engine can supply one, in which case
            the mirror fills in on the engine's next broadcast.
        """
        candidate = station_info
        if candidate is None:
            candidate = getattr(engine, "station_info", None)
        if callable(candidate):
            candidate = candidate()
        if isinstance(candidate, StationInfo):
            return candidate
        return StationInfo()

    # ── The mirror ────────────────────────────────────────────────────

    def _observe_event(self, event: Any) -> None:
        """Update the local mirror from one engine event.

        Guarded like every other reporting surface: a mirror that cannot be
        updated must never propagate an exception back into the engine's own
        emit.

        Args:
            event: Anything on the engine's event stream.
        """
        try:
            self._engine_seq = max(self._engine_seq, int(getattr(event, "seq", 0)))
            if isinstance(event, StatusSnapshot):
                self._status = event
            elif isinstance(event, StationInfo):
                self._station_info = event
        except Exception:  # noqa: BLE001 — mirroring must never disrupt the engine
            logger.exception("gateway mirror update failed (non-fatal)")

    def _observe_verdict(self, verdict: Verdict) -> None:
        """Keep the sequence mirror abreast of the verdict stream too.

        Args:
            verdict: Any verdict the engine emitted, for this client or another.
        """
        try:
            self._engine_seq = max(self._engine_seq, int(verdict.seq))
            if self._awaiting_verdict:
                self._awaited_verdicts.append(verdict)
        except Exception:  # noqa: BLE001 — mirroring must never disrupt the engine
            logger.exception("gateway verdict mirror update failed (non-fatal)")

    def _next_seq(self) -> int:
        """Return the sequence number to stamp on a locally emitted refusal.

        Derived from the engine's own counter wherever it is reachable — the
        highest ``seq`` seen on either stream — so a refusal the engine never
        saw still orders correctly against everything it did say.

        Returns:
            A number strictly greater than every sequence number seen so far.
        """
        self._local_seq = max(self._local_seq, self._engine_seq) + 1
        return self._local_seq

    # ── Reads: answered from the mirror, never by calling the engine ──

    def status(self) -> StatusSnapshot | None:
        """Return the latest status snapshot, or ``None`` before the first tick.

        Returns:
            The engine's most recent ``StatusSnapshot``.
        """
        return self._status

    def station(self) -> StationInfo:
        """Return the station's latest declaration snapshot.

        Returns:
            The most recent ``StationInfo`` — what every instrument reads,
            can be asked to do, and within which bounds.
        """
        return self._station_info

    def state(self) -> str:
        """Return the engine's current state name.

        Returns:
            The state machine's state, or ``""`` before the first snapshot.
        """
        return self._status.state if self._status is not None else ""

    def attended(self) -> bool:
        """Return whether a human is watching, per the engine's mirror.

        Returns:
            The published **Attendance** value; ``True`` (the restrictive
            reading for an agent) before the first snapshot arrives.
        """
        return self._status.attended if self._status is not None else True

    def run_owner(self) -> dict[str, Any] | None:
        """Return the **run owner** of the run in flight, per the mirror.

        Read exactly as **Attendance** and the **Kill switch** are: off the
        latest ``StatusSnapshot``, so this connection's ownership check reads
        the same fact every other client sees, and the engine still enforces
        the rule regardless of what a client makes of it.

        Returns:
            The owner's ``{"kind", "id"}``, or ``None`` when nothing is
            running or no snapshot has arrived yet.
        """
        run = self._status.run if self._status is not None else None
        if not run:
            return None
        owner = run.get("owner")
        return dict(owner) if isinstance(owner, Mapping) else None

    def agent_gate(self) -> AgentGate:
        """Return the **Kill switch**'s setting, per the engine's mirror.

        Returns:
            The published ``AgentGate``; ``ACTIVE`` before the first snapshot
            arrives, since the engine enforces the gate itself regardless.
        """
        if self._status is None:
            return AgentGate.ACTIVE
        return AgentGate(self._status.agent_gate)

    # ── The one write path ────────────────────────────────────────────

    def permits(
        self, name: CommandName | str, args: Mapping[str, Any] | None = None
    ) -> Verdict | None:
        """Answer whether this connection may submit that command, without doing it.

        The same decision ``submit()`` makes, exposed so a client can render
        its own tool list, or explain a refusal, without provoking one.

        Args:
            name: The command, as a ``CommandName`` or its string value.
            args: The command's arguments, JSON-safe.

        Returns:
            ``None`` when it would be forwarded, or the ``BLOCKED_ROLE``
            verdict that would refuse it.

        Raises:
            ValueError: If *name* is not a known ``CommandName``.
            TypeError: If *args* is not a mapping of JSON-safe values.
        """
        return self._authorize(self._command(name, args))

    def submit(
        self, name: CommandName | str, args: Mapping[str, Any] | None = None
    ) -> str:
        """Submit one command under this connection's identity.

        Stamps the actor, runs the permission check, and either forwards to
        the engine or answers the command here with a ``BLOCKED_ROLE``
        verdict on the engine's own ``verdict_emitted`` stream — so exactly
        one verdict answers every request either way, and the human's window
        sees an agent being refused exactly as it sees it being obeyed. A
        refused command never reaches the engine.

        Args:
            name: The command, as a ``CommandName`` or its string value.
            args: The command's arguments, JSON-safe and shaped by the
                target's ``ParamSpec``s — see ``Orchestrator.submit()`` for
                the per-command conventions.

        Returns:
            The request id the answering verdict carries back, whether the
            command was forwarded or refused here.

        Raises:
            ValueError: If *name* is not a known ``CommandName``.
            TypeError: If *args* is not a mapping of JSON-safe values.
        """
        command = self._command(name, args)
        self._record(command)
        refusal = self._authorize(command)
        if refusal is None:
            return self._engine.submit(command)
        self._emit(refusal)
        return command.request_id

    def _command(
        self, name: CommandName | str, args: Mapping[str, Any] | None
    ) -> Command:
        """Build one command stamped with this connection's actor.

        Args:
            name: The command name.
            args: Its arguments, or ``None``.

        Returns:
            The ``Command``, validated by the contract at construction.
        """
        return Command(name=name, actor=self.actor, args=dict(args or {}))

    def _record(self, command: Command) -> None:
        """Write one submitted command to the **Agent feed**, when there is one.

        Called before the permission check, so a refused command is in the
        trail exactly like an obeyed one — what an agent TRIED to do is as
        much a part of accountability as what it managed to do.

        Args:
            command: The command about to be authorized.
        """
        if self._feed is not None:
            self._feed.record_command(command)

    def _authorize(self, command: Command) -> Verdict | None:
        """Run the permission check for one command against the current mirror.

        Args:
            command: The command to judge.

        Returns:
            ``None`` when permitted, else the refusing verdict.
        """
        refusal = authorize(
            self.actor,
            command,
            self._station_info,
            self.attended(),
            self.agent_gate(),
            run_owner=self.run_owner(),
        )
        if refusal is None:
            return None
        # Numbered only now: a permitted command is numbered by the engine
        # that carries it out, so allocating here would leave a hole for
        # every command that was never refused.
        return replace(refusal, seq=self._next_seq())

    def _emit(self, verdict: Verdict) -> None:
        """Put one locally decided verdict onto the verdict stream.

        Whichever stream this gateway was attached to: the engine's own when
        a caller holds the engine, the **Orchestrator proxy**'s when it holds
        a proxy — which is the same broadcast every other client on that
        thread is already listening to, so the human's window still sees the
        refusal.

        Guarded: a listener that raises must not propagate back into the
        client that submitted the command.

        Args:
            verdict: The refusal to broadcast.
        """
        try:
            verdict_stream(self._engine).emit(verdict)
        except Exception:  # noqa: BLE001 — a signal failure must not raise at the client
            logger.exception("gateway verdict emit failed")

    # ── The tool surface ──────────────────────────────────────────────

    def tools(self) -> tuple[ToolSpec, ...]:
        """Return the **Tool surface** rendered for the station being mirrored.

        Re-rendered whenever the engine broadcasts a new declaration snapshot
        — a connect or a disconnect rebuilds it — so an instrument that
        appears brings its capability tools with it and nothing has to be
        told twice.

        Returns:
            Every command tool, capability tool and session tool, in render
            order.
        """
        if self._tools_rendered_from is not self._station_info:
            self._tools_rendered_from = self._station_info
            self._tools = {tool.name: tool for tool in render_tools(self._station_info)}
        return tuple(self._tools.values())

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return the tool list in the shape a tool-use API expects.

        The one rendering every client publishes, so a terminal client and a
        later tool-use adapter offer the same surface with no code of their
        own.

        Returns:
            One ``{"name", "description", "input_schema"}`` dict per tool.
        """
        return [tool.to_schema() for tool in self.tools()]

    def tool(self, name: str) -> ToolSpec | None:
        """Return one tool by name, or ``None`` when this surface has no such tool.

        Args:
            name: The tool's name.

        Returns:
            Its ``ToolSpec``, or ``None``.
        """
        self.tools()
        return self._tools.get(name)

    def call_tool(
        self, name: str, args: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call one tool by name and answer it, always.

        Three steps, in this order: the tool must exist, the arguments must
        satisfy its schema, and only then is it routed — a command tool
        through ``submit()`` (where the role matrix, the kill switch and the
        engine's own admission rules judge it) and a session tool to its
        function. Nothing raises at the caller: an unknown tool, a schema
        violation, an absent collaborator and an unexpected failure are all
        answered with a ``FAILED``-shaped dict whose ``detail`` names the rule
        that refused.

        Args:
            name: The tool's name, as published by ``tool_schemas()``.
            args: Its arguments, JSON-safe.

        Returns:
            A JSON-safe dict carrying ``tool``, ``ok`` and ``code``; a read
            tool adds ``result``, a command tool adds ``request_id`` and, once
            its verdict has arrived, ``verdict`` with the ``reason`` and
            ``detail`` the engine gave.
        """
        call_args = dict(args or {})
        tool = self.tool(name)
        if tool is None:
            return self._tool_failure(
                name,
                f"no tool named {name!r}; this station offers "
                f"{len(self._tools)} tools",
                {"rule": "unknown_tool"},
            )

        errors = validate_tool_args(call_args, tool.input_schema)
        if errors:
            return self._tool_failure(
                name,
                f"{name} refused: " + "; ".join(errors),
                {"rule": "schema", "errors": list(errors)},
            )

        if tool.is_command:
            return self._call_command_tool(tool, call_args)
        return self._call_session_tool(tool, call_args)

    def _call_command_tool(
        self, tool: ToolSpec, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Submit one command tool and report the verdict that answered it.

        Args:
            tool: The tool being called.
            args: Its validated arguments.

        Returns:
            The answer dict, carrying the request id and — since the engine
            answers synchronously on the submitting call — the verdict.
        """
        payload = {**tool.fixed_args, **args}
        self._awaiting_verdict = True
        self._awaited_verdicts.clear()
        try:
            request_id = self.submit(tool.command, payload)
        finally:
            self._awaiting_verdict = False
        # Everything the engine said while the command was in flight, not
        # merely the last of it: carrying out one command can answer another
        # (a queued action drained on the same call), and the answer to THIS
        # call is the one carrying its request id.
        verdict = next(
            (
                seen
                for seen in self._awaited_verdicts
                if seen.request_id == request_id
            ),
            None,
        )
        self._awaited_verdicts.clear()

        answer: dict[str, Any] = {"tool": tool.name, "request_id": request_id}
        if verdict is None:
            answer.update(
                {
                    "ok": False,
                    "code": "PENDING",
                    "reason": (
                        "the command was accepted and its verdict will arrive on "
                        "the engine's verdict stream"
                    ),
                    "detail": {},
                }
            )
            return answer
        answer.update(
            {
                "ok": verdict.ok,
                "code": verdict.code.value,
                "reason": verdict.reason,
                "detail": dict(verdict.detail or {}),
                "result": verdict.result,
                "verdict": verdict.to_json(),
            }
        )
        return answer

    def _call_session_tool(
        self, tool: ToolSpec, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Answer one session tool here, after the same authority checks.

        A session tool is not a command, so no ``Verdict`` answers it — but it
        is still an action of a declared **Action class**, so the kill switch
        and the permission matrix decide it exactly as they decide a command.

        Args:
            tool: The tool being called.
            args: Its validated arguments.

        Returns:
            The answer dict, carrying ``result`` on success.
        """
        refusal = self._authorize_session_tool(tool)
        if refusal is not None:
            return self._record_tool_call(tool, args, refusal)
        try:
            result = call_session_tool(tool, args, self._tool_context)
        except ToolError as error:
            return self._record_tool_call(
                tool, args, self._tool_failure(tool.name, str(error), error.detail)
            )
        except Exception as error:  # noqa: BLE001 — a tool never raises at its caller
            logger.exception("call_tool(%s) failed", tool.name)
            return self._record_tool_call(
                tool,
                args,
                self._tool_failure(
                    tool.name,
                    f"{tool.name} failed: {error}",
                    {"rule": "unexpected_error", "error": type(error).__name__},
                ),
            )
        return self._record_tool_call(
            tool,
            args,
            {
                "tool": tool.name,
                "ok": True,
                "code": VerdictCode.OK.value,
                "reason": "",
                "detail": {},
                "result": result,
            },
        )

    def _record_tool_call(
        self, tool: ToolSpec, args: Mapping[str, Any], answer: dict[str, Any]
    ) -> dict[str, Any]:
        """Write one answered session tool to the **Agent feed**, and pass it on.

        Only for a tool that declares ``ToolSpec.recorded``, and only when a
        feed was attached: a session tool is answered inside this client, so
        no verdict record would otherwise name it, while a ``read`` tool an
        agent polls every tick must not drown the trail (see the record
        standard in ``session/agent_feed.py``). The recorded ``detail`` is the
        answer's own, plus the cost line when the call spent model tokens —
        what an autonomous client spent, beside what it asked for. The
        arguments are recorded through ``tools.feed_arguments()``, which
        digests the ones too big to belong in an append-only file (a written
        recipe's source travels as its byte count and its SHA-256, and the
        file on disk is the copy).

        Args:
            tool: The tool that was called.
            args: The arguments it was called with.
            answer: The answer about to be returned, refusal or success.

        Returns:
            *answer*, unchanged, so this sits on the return path.
        """
        if self._feed is None or not tool.recorded:
            return answer
        self._feed.record_tool_call(
            self.actor,
            tool.name,
            feed_arguments(tool, args),
            detail={**dict(answer.get("detail") or {}), **cost_line(answer.get("result"))},
            verdict={
                "code": str(answer.get("code", "")),
                "reason": str(answer.get("reason", "")),
            },
        )
        return answer

    def _authorize_session_tool(self, tool: ToolSpec) -> dict[str, Any] | None:
        """Judge one session tool by the same table a command is judged by.

        Args:
            tool: The tool being called.

        Returns:
            ``None`` when it may run, else the ``BLOCKED_ROLE``-shaped answer
            that refuses it.
        """
        gate = self.agent_gate()
        if gate is AgentGate.REVOKED or (
            gate is AgentGate.READ_ONLY and tool.action_class is not ActionClass.READ
        ):
            return self._tool_refusal(
                tool,
                f"Agent access is {gate.value}: {tool.name!r} is a "
                f"{tool.action_class.value} tool and is refused.",
                {"rule": "kill_switch", "gate": gate.value},
            )
        permission = PERMISSION_MATRIX[tool.action_class][self.role]
        if permission is Permission.PERMITTED:
            return None
        if permission is Permission.UNATTENDED_ONLY:
            if not self.attended():
                return None
            return self._tool_refusal(
                tool,
                f"The {self.role.value!r} role may take "
                f"{tool.action_class.value} actions only while the experiment "
                f"is unattended; a human is watching, so {tool.name!r} is "
                f"refused — report it instead.",
                {"rule": "attendance", "attended": True},
            )
        return self._tool_refusal(
            tool,
            f"The {self.role.value!r} role does not grant "
            f"{tool.action_class.value} actions, so {tool.name!r} is refused.",
            {"rule": "role_matrix"},
        )

    def _tool_refusal(
        self, tool: ToolSpec, reason: str, detail: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Build the ``BLOCKED_ROLE``-shaped answer refusing one session tool.

        Args:
            tool: The tool refused.
            reason: The human-readable explanation.
            detail: The structured half; the role and action class are added.

        Returns:
            The answer dict.
        """
        logger.info("Gateway tool refusal: %s", reason)
        return {
            "tool": tool.name,
            "ok": False,
            "code": VerdictCode.BLOCKED_ROLE.value,
            "reason": reason,
            "detail": {
                **dict(detail),
                "role": self.role.value,
                "action_class": tool.action_class.value,
            },
        }

    @staticmethod
    def _tool_failure(
        name: str, reason: str, detail: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Build the one failure shape ``call_tool()`` ever answers with.

        Args:
            name: The tool that was called.
            reason: The human-readable explanation.
            detail: The structured half, naming the rule that refused.

        Returns:
            The answer dict.
        """
        logger.info("Gateway tool failure: %s", reason)
        return {
            "tool": name,
            "ok": False,
            "code": VerdictCode.FAILED.value,
            "reason": reason,
            "detail": dict(detail),
        }
