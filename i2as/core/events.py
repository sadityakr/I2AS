"""The control contract — the typed currency between the engine and its clients.

Two modules' worth of payload live here, both frozen and dependency-free so
that the Orchestrator (emitter) and every client (the GUI today, an agent
gateway later) can import them without dragging a layer along:

* ``ErrorEvent`` — the structured error/fault notification.
* The **station declaration snapshot**: ``StationInfo`` and its nested
  ``InstrumentInfo`` / ``MonitoredInfo`` / ``ControlInfo`` / ``GroupInfo``,
  the frozen picture of what every configured instrument reads, what it can
  be asked to do, within which bounds, and how those capabilities group.
  This module DEFINES that shape; ``Station.station_info()`` BUILDS it (only
  the Station holds both the VI declarations and the config the bounds come
  from) and ``core.capability_manifest.build_manifest()`` renders it. Keeping
  the definition here is what lets a client import the whole declaration
  without dragging the instrument stack in behind it.
* The **control contract**: ``Actor``, ``Command``, ``Verdict`` and the
  ``Event`` union. The Orchestrator has exactly two clients, the GUI and the
  agent, and they must see the same system and be seen doing the same things.
  So the boundary is one contract with two renderings rather than a proxy for
  one client and a gateway for the other: every action is a ``Command``
  answered by exactly one ``Verdict``, every consequence is an ``Event``, and
  every message names the ``Actor`` behind it, so a client reflects what the
  other one did for free.

Every contract type is a frozen dataclass whose fields are JSON-safe (``str``,
``int``, ``float``, ``bool``, ``None``, list/tuple/dict of those, enums
rendered as their values, and nested contract types) and carries
``to_json()`` / ``from_json()``. A dict round trip is the contract: the value
survives ``to_json()`` → ``json.dumps`` → ``json.loads`` → ``from_json()``
unchanged, which is what lets the same declaration cross a thread boundary
today and a process boundary later with no second contract. Every event also
carries a class-level ``kind`` discriminator, emitted into its JSON, so
``event_from_json()`` can rebuild the right type without the caller knowing
which one it holds.

``Command`` and ``Verdict`` here are the *control-contract* pair — a client's
request to the engine and the engine's answer. They are deliberately distinct
from ``core.plan.Command`` (one method call a procedure hands the Orchestrator
to dispatch on a VI) and ``core.conditions.Verdict`` (the enforcement decision
``decide()`` computes from a set of conditions); see ``GLOSSARY.md``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, ClassVar, Self


@dataclass(frozen=True)
class ErrorEvent:
    """One structured error/fault notification.

    Attributes:
        vi_name: The VI the event concerns, or ``None`` for a machine-wide
            event with no single originating instrument (e.g. an unhandled
            tick-boundary exception). May also be a comma-joined list of
            names when more than one VI is implicated (e.g. an EMERGENCY
            tripped by more than one instrument's safety flag).
        kind: The blast-radius tier this event belongs to:
            ``"fault"`` (a VI-scoped comm/stale/disconnected fault that
            quarantines only that VI), ``"run_failure"`` (an active run's
            claimed VI faulted — the run fails, the machine returns to
            IDLE), ``"safety"`` (a tripped safety flag — global EMERGENCY),
            ``"internal"`` (an unhandled tick-boundary exception —
            global ERROR, unknown blast radius), or ``"safety_hold"`` (a
            VI-scoped safety-hold enforcement action — the Orchestrator
            re-asserting or failing to re-assert ``standby()`` on a VI held
            by a hold-severity safety condition; scoped to that one VI,
            never a blast radius beyond it, unlike ``"safety"`` above).
        severity: ``"warning"``, ``"error"``, or ``"emergency"``.
        message: Human-readable description, suitable for direct display.
        timestamp: Unix time the event was created (``time.time()``).
    """

    vi_name: str | None
    kind: str
    severity: str
    message: str
    timestamp: float


# ══════════════════════════════════════════════════════════════════════
# The control contract
# ══════════════════════════════════════════════════════════════════════

# The JSON-safe scalar leaves every contract field bottoms out in.
_JSON_SCALARS = (str, int, float, bool)


def _jsonable(value: Any) -> Any:
    """Render one contract field as a JSON-safe value.

    Args:
        value: The field value. Enums render as their value, nested contract
            types through their own ``to_json()``, tuples as lists, mappings
            as plain dicts, and JSON scalars unchanged.

    Returns:
        A value made only of ``str``/``int``/``float``/``bool``/``None``/
        ``list``/``dict``, i.e. one ``json.dumps`` accepts.

    Raises:
        TypeError: If the value is of a type the contract cannot carry. This
            fires at construction (every contract type validates eagerly), so
            a non-serialisable payload fails at the boundary that built it
            rather than at the boundary that has to send it.
    """
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, _JSON_SCALARS):
        return value
    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        return to_json()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(
        f"control-contract fields must be JSON-safe; got {type(value).__name__}"
    )


def _checked_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a defensive, JSON-checked copy of a contract mapping field.

    Args:
        value: The mapping to copy, or ``None`` for an empty one.

    Returns:
        A new ``dict`` whose contents are known JSON-safe.

    Raises:
        TypeError: If the value is not a mapping, or holds a non-JSON-safe
            value (raised by ``_jsonable``).
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping, got {type(value).__name__}")
    return {str(key): _jsonable(item) for key, item in value.items()}


class _ContractMessage:
    """JSON round trip shared by every control-contract message.

    Mixed into the frozen dataclasses below. ``to_json()`` renders the
    declared fields plus, for events, the class-level ``kind`` discriminator;
    ``from_json()`` rebuilds from that dict, ignoring keys it does not declare
    so a newer producer never breaks an older consumer. Field coercion (a
    plain string back into its enum, a list back into a tuple, a dict back
    into a nested ``Actor``) happens in each class's ``__post_init__``, so the
    same coercion serves construction and deserialisation.
    """

    def to_json(self) -> dict[str, Any]:
        """Render this message as a JSON-safe dict.

        Returns:
            A dict of the declared fields, plus ``"kind"`` for event types.
            A type that declares ``kind`` as an ordinary FIELD (e.g.
            ``InstrumentInfo``, where it is the instrument's category) keeps
            its own value: the discriminator is written only for a type whose
            ``kind`` is the class-level tag, never over a field of that name.
        """
        declared = [f.name for f in fields(self)]
        payload = {name: _jsonable(getattr(self, name)) for name in declared}
        kind = getattr(type(self), "kind", None)
        if isinstance(kind, str) and "kind" not in declared:
            payload["kind"] = kind
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild a message from its ``to_json()`` dict.

        Args:
            payload: A mapping as produced by ``to_json()`` (or the result of
                sending one through ``json.dumps``/``json.loads``). Unknown
                keys — including the ``"kind"`` discriminator — are ignored.

        Returns:
            An instance of the class this is called on.

        Raises:
            TypeError: If a declared field is missing or carries a value the
                type rejects.
        """
        declared = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in declared})


# ── Actor ─────────────────────────────────────────────────────────────

class ActorKind(str, Enum):
    """Who is acting: a human at the GUI, an autonomous agent, or the system.

    A ``str`` enum so the value is JSON-safe as it stands and compares equal
    to its own wire form.
    """

    OPERATOR = "operator"
    AGENT = "agent"
    SYSTEM = "system"


@dataclass(frozen=True)
class Actor(_ContractMessage):
    """Who issued a command, or on whose behalf an event happened.

    Accountability is a value, not an ambient fact: every command and every
    event it causes names its actor, so the status bar can say "paused by
    agent", the queue can show who queued a run, and a run record can tell an
    agent's self-confirmation apart from the physicist's.

    Attributes:
        kind: ``operator``, ``agent`` or ``system`` (see ``ActorKind``).
            Accepts the enum member or its string value.
        id: Stable identifier for this actor — a username, an agent
            deployment name, or a subsystem name for ``system``.
        role: The authority this actor holds, free-form here; the agent
            gateway that grants roles owns their vocabulary.
    """

    kind: ActorKind
    id: str
    role: str = ""

    def __post_init__(self) -> None:
        """Coerce ``kind`` to its enum member and validate the string fields.

        Raises:
            ValueError: If ``kind`` is not a known ``ActorKind``.
            TypeError: If ``id`` or ``role`` is not a string.
        """
        object.__setattr__(self, "kind", ActorKind(self.kind))
        for name in ("id", "role"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"Actor.{name} must be a str")

    def ref(self) -> dict[str, str]:
        """Render this actor as the identity half of itself: kind and id.

        The shape every message that POINTS AT an actor rather than being
        issued by one carries — the **run owner** on a ``StatusSnapshot``'s
        run summary, the ``owner`` of an ownership refusal's detail, and
        ``RunFinished.overridden_owner`` (see GLOSSARY.md's *Run owner*).
        The role is deliberately absent: it is the authority a command was
        taken under, which belongs to the acting message, not to a pointer at
        who someone is.

        Returns:
            ``{"kind": ..., "id": ...}``, JSON-safe as it stands.
        """
        return {"kind": self.kind.value, "id": self.id}


#: The default actor for every entry point: the human at the GUI. Public
#: methods take ``actor=OPERATOR`` so no existing call site has to change.
OPERATOR = Actor(kind=ActorKind.OPERATOR, id="operator", role="operator")


class AgentGate(str, Enum):
    """The kill switch: how much of the engine autonomous actors may reach.

    A tri-state the human owns, pushed DOWN into the engine as a value (see
    ``Orchestrator.set_agent_gate()``) so that the single writer enforces it
    rather than trusting a client to police itself. It gates ``agent``
    actors ONLY: the ``operator`` and ``system`` paths are never consulted,
    which is the whole point of a kill switch — flipping it must never be
    able to lock the human out of their own instrument. ``emergency_standby``
    passes at every setting, because an actor that can see a problem must
    never be unable to make the station safe.

    A ``str`` enum so the value is JSON-safe as it stands and travels on a
    ``StatusSnapshot`` unchanged.

    Members:
        ACTIVE: Agents act normally; their role's permissions decide.
        READ_ONLY: Agents may take read-class actions only; anything that
            writes is refused naming the gate.
        REVOKED: Agents may take no action at all.
    """

    ACTIVE = "active"
    READ_ONLY = "read_only"
    REVOKED = "revoked"


class LifecycleState(str, Enum):
    """What an instrument is DOING, as an observed fact rather than a history.

    The **lifecycle-state standard**'s vocabulary (GLOSSARY.md's *Lifecycle
    state*): the VI layer keeps this as data
    (``BaseVirtualInstrument.lifecycle_state()``), the Station reports it
    (``Station.lifecycle_states()``), and it reaches every client on
    ``StatusSnapshot.instruments[vi_name]["lifecycle"]``. Declared HERE, in
    the contract, for the same reason every other shared vocabulary is: a
    client renders the fact without importing the instrument stack, and no
    client has to reconstruct it from whichever actions it happened to
    witness — a station stood down by a path that emits no per-VI action
    (an emergency's blanket ``standby_all()``) still reports the truth on
    the very next snapshot.

    A ``str`` enum, like ``AgentGate``, so the value is JSON-safe as it
    stands and travels on a ``StatusSnapshot`` unchanged.

    Members:
        IDLE: Not initiated — a freshly built VI, one whose ``disconnect()``
            hook has run, and any instrument that is not in the live
            registry at all.
        INITIATED: ``initiate()`` succeeded and nothing has stood the
            instrument down since. A measurement VI's
            ``initiate_measurement()`` — the arming half of its lifecycle —
            counts the same way.
        STANDBY: ``standby()`` succeeded; the instrument is at the safe idle
            state its own stand-down drives it to.
    """

    IDLE = "idle"
    INITIATED = "initiated"
    STANDBY = "standby"


def _as_actor(value: Actor | Mapping[str, Any]) -> Actor:
    """Coerce an actor field, which may arrive as its JSON dict.

    Args:
        value: An ``Actor`` or the mapping ``Actor.to_json()`` produced.

    Returns:
        An ``Actor``.

    Raises:
        TypeError: If the value is neither.
    """
    if isinstance(value, Actor):
        return value
    if isinstance(value, Mapping):
        return Actor.from_json(value)
    raise TypeError(f"expected an Actor, got {type(value).__name__}")


# ── Command ───────────────────────────────────────────────────────────

class CommandName(str, Enum):
    """Every Orchestrator command a client may issue.

    The command half of the control contract is enumerated exactly once,
    here: the GUI's action buttons and the agent's tool list are both
    rendered from this enum, so neither client can offer an action the other
    cannot see. Each value is the name of the ``Orchestrator`` method that
    implements it, which is what makes dispatch a lookup rather than a table.

    Read-only accessors are deliberately absent — a client answers those from
    its latest ``StatusSnapshot``, never by calling into the engine — as is
    ``shutdown()``, which is process lifecycle owned by ``main.py``.
    ``tests/test_conformance.py`` diffs this enum against the Orchestrator's
    public methods in both directions, with those exemptions named and
    justified there.
    """

    # Runs and the queue
    RUN_PROCEDURE = "run_procedure"
    QUEUE_PROCEDURE = "queue_procedure"
    RUN_QUEUE = "run_queue"
    PAUSE_PROCEDURE = "pause_procedure"
    RESUME_PROCEDURE = "resume_procedure"
    ABORT_PROCEDURE = "abort_procedure"

    # Instrument actions
    SUBMIT_VI_ACTION = "submit_vi_action"
    SUBMIT_GLOBAL_ACTION = "submit_global_action"
    STOP_RAMP = "stop_ramp"
    CONNECT_INSTRUMENT = "connect_instrument"
    DISCONNECT_INSTRUMENT = "disconnect_instrument"
    PING_INSTRUMENT = "ping_instrument"

    # Faults, safety and recovery
    EMERGENCY_STANDBY = "emergency_standby"
    ACKNOWLEDGE = "acknowledge"
    ACKNOWLEDGE_FAULT = "acknowledge_fault"
    RETRY_FAULT = "retry_fault"
    RECOVER_FROM_ERROR = "recover_from_error"

    # Monitoring and policy
    START_MONITORING = "start_monitoring"
    STOP_MONITORING = "stop_monitoring"
    SET_EXPERIMENT_ENVELOPE = "set_experiment_envelope"
    SET_ATTENDANCE = "set_attendance"
    SET_AGENT_GATE = "set_agent_gate"


@dataclass(frozen=True)
class Command(_ContractMessage):
    """One client's request for the engine to act.

    The control-contract request type — not ``core.plan.Command``, which is a
    procedure's request to dispatch one method on one VI.

    Attributes:
        name: Which command, from ``CommandName``. Accepts the enum member or
            its string value.
        actor: Who is asking. Defaults to the ``OPERATOR`` sentinel.
        args: The command's arguments, JSON-safe and shaped by the target's
            ``ParamSpec``s. Defensively copied and validated at construction.
        request_id: Correlation id; the ``Verdict`` answering this command and
            every event it causes carry it back. Defaults to a fresh hex UUID.
        issued_at: Unix time the command was created (``time.time()``).
    """

    name: CommandName
    actor: Actor = OPERATOR
    args: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    issued_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Coerce the enum, actor and args fields and validate them eagerly.

        Raises:
            ValueError: If ``name`` is not a known ``CommandName``.
            TypeError: If ``actor`` is not an ``Actor`` (or its dict), if
                ``args`` is not a mapping or holds a non-JSON-safe value, or
                if ``request_id``/``issued_at`` has the wrong type.
        """
        object.__setattr__(self, "name", CommandName(self.name))
        object.__setattr__(self, "actor", _as_actor(self.actor))
        object.__setattr__(self, "args", _checked_mapping(self.args))
        if not isinstance(self.request_id, str):
            raise TypeError("Command.request_id must be a str")
        if not isinstance(self.issued_at, (int, float)):
            raise TypeError("Command.issued_at must be a number")


# ── Verdict ───────────────────────────────────────────────────────────

class VerdictCode(str, Enum):
    """Why a command was accepted or refused.

    The machine-readable half of a verdict: a client decides what to do next
    from the code, never by parsing ``reason``, which exists for humans.

    Members:
        OK: The command was accepted and carried out.
        BLOCKED_STATE: The state machine forbids it right now.
        BLOCKED_CLAIM: An active run claims the instrument.
        BLOCKED_FAULT: The target instrument is faulted or offline.
        BLOCKED_LIMIT: A declared control limit rejected a parameter; the
            ``detail`` dict carries ``param``, ``value``, ``lo``, ``hi`` and
            ``limit_name``.
        BLOCKED_ENVELOPE: The active session envelope forbids it.
        BLOCKED_ROLE: The actor's role does not grant this action.
        FAILED: Accepted, attempted, and it failed.
    """

    OK = "OK"
    BLOCKED_STATE = "BLOCKED_STATE"
    BLOCKED_CLAIM = "BLOCKED_CLAIM"
    BLOCKED_FAULT = "BLOCKED_FAULT"
    BLOCKED_LIMIT = "BLOCKED_LIMIT"
    BLOCKED_ENVELOPE = "BLOCKED_ENVELOPE"
    BLOCKED_ROLE = "BLOCKED_ROLE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Verdict(_ContractMessage):
    """The engine's single answer to one ``Command``.

    Exactly one verdict answers every command, including every refusal — a
    broadcast signal a client may or may not be listening to is not an answer
    an agent can await. The control-contract answer type, not
    ``core.conditions.Verdict``, which is the enforcement decision computed
    from a set of system conditions.

    Attributes:
        request_id: The ``Command.request_id`` this answers.
        command: Which command was asked for.
        code: The machine-readable outcome (see ``VerdictCode``).
        actor: The actor of the command being answered.
        reason: Human-readable explanation, suitable for a banner. Never
            parsed by a client.
        detail: Optional structured explanation of the code — for
            ``BLOCKED_LIMIT`` the rejected ``param``, ``value``, ``lo``,
            ``hi`` and ``limit_name``. JSON-safe.
        result: Optional JSON-safe return value of the underlying call.
        seq: Monotonic sequence number of the emitting engine.
        ts: Unix time the verdict was created (``time.time()``).
    """

    request_id: str
    command: CommandName
    code: VerdictCode
    actor: Actor = OPERATOR
    reason: str = ""
    detail: dict[str, Any] | None = None
    result: Any = None
    seq: int = 0
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Coerce the enum, actor and payload fields and validate them.

        Raises:
            ValueError: If ``command`` or ``code`` is not a known member.
            TypeError: If a field carries a non-JSON-safe or wrongly typed
                value.
        """
        object.__setattr__(self, "command", CommandName(self.command))
        object.__setattr__(self, "code", VerdictCode(self.code))
        object.__setattr__(self, "actor", _as_actor(self.actor))
        if self.detail is not None:
            object.__setattr__(self, "detail", _checked_mapping(self.detail))
        object.__setattr__(self, "result", _jsonable(self.result))
        if not isinstance(self.request_id, str) or not isinstance(self.reason, str):
            raise TypeError("Verdict.request_id and Verdict.reason must be str")
        if not isinstance(self.seq, int) or isinstance(self.seq, bool):
            raise TypeError("Verdict.seq must be an int")

    @property
    def ok(self) -> bool:
        """Whether the command was accepted and carried out.

        Returns:
            ``True`` exactly when ``code`` is ``VerdictCode.OK``. Derived, not
            stored, so no verdict can claim success and a blocking code at
            once.
        """
        return self.code is VerdictCode.OK


# ── Events ────────────────────────────────────────────────────────────
#
# Everything the engine broadcasts. Each event is frozen, carries ``seq``
# and ``ts``, and — where it is the consequence of a command — the
# ``request_id`` and ``actor`` of that command, which is what lets one
# client see what the other one did. Each declares a class-level ``kind``
# discriminator, emitted into its JSON so ``event_from_json()`` can
# dispatch. (``ErrorEvent.kind`` above is a different thing: a blast-radius
# tier, not a type tag.)


@dataclass(frozen=True)
class StateChange(_ContractMessage):
    """The engine's state machine moved.

    Attributes:
        state: The new state's name (an ``OrchestratorState`` value).
        previous: The state left behind, or ``""`` for the first transition.
        cause: Short machine-readable reason for the transition (e.g.
            ``"run_started"``, ``"fault"``, ``"operator_abort"``).
        actor: Who caused it; the ``system`` actor for a transition the engine
            made on its own.
        request_id: The command that caused it, or ``""``.
        seq: Monotonic sequence number.
        ts: Unix time of the transition.
    """

    kind: ClassVar[str] = "state_change"

    state: str
    previous: str = ""
    cause: str = ""
    actor: Actor = OPERATOR
    request_id: str = ""
    seq: int = 0
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Coerce the actor field.

        Raises:
            TypeError: If ``actor`` is neither an ``Actor`` nor its dict.
        """
        object.__setattr__(self, "actor", _as_actor(self.actor))


@dataclass(frozen=True)
class StatusSnapshot(_ContractMessage):
    """Everything a client needs to answer a read without calling the engine.

    Emitted once per tick and on every state change; a client's mirror of it
    is what makes every read local. Its fields are the engine's whole read
    surface: the state machine, the run in flight, and one field per
    read-only accessor the Orchestrator exposes — named after that accessor,
    so "which snapshot field answers ``held_vi_names()``?" never needs a
    lookup table. ``tests/test_conformance.py`` diffs the two surfaces, so an
    accessor added to the engine without a field here fails the harness.

    ``instruments`` is the per-VI merge of the same information (availability,
    fault, offline reason, hold, override) for a client that renders one panel
    per instrument rather than one table per concern; it carries the live half
    of what ``StationInfo`` declares statically. One of its keys is answered
    by no other field: ``lifecycle`` — a ``LifecycleState`` value saying what
    that instrument is DOING (see GLOSSARY.md's *Lifecycle state*). It is
    here, rather than left for a client to infer from the actions it saw,
    because the engine stands the whole station down by paths that emit no
    per-VI action at all (an emergency's ``Station.standby_all()``); a client
    that reconstructed the state from action history would go on showing an
    instrument as running after the hardware was stood down.

    Attributes:
        state: The engine's current state name.
        run: The active run's summary (``run_id``, ``kind``, ``name``,
            ``progress``, ``step`` where available) plus ``owner``, the
            **run owner**'s ``Actor.ref()`` — who started this run, which is
            what every client shows and what the ownership rule is judged
            against. ``None`` when idle.
        instruments: ``{vi_name: {...}}`` of live per-instrument status —
            ``availability``, ``fault``, ``offline_reason``, ``held``,
            ``override_active`` and ``lifecycle`` (a ``LifecycleState``
            value; see above).
        is_monitoring: Whether the per-tick monitoring cycle is polling.
        pause_pending: Whether a pause is waiting for the current datapoint.
        active_run_kind: ``"procedure"``, or ``None`` while idle.
        override_active: Whether the EMERGENCY manual override is unlocked
            (the ``override_active(None)`` answer; the per-VI answers live in
            ``instruments[vi_name]["override_active"]``).
        manual_override_expires_at: Soonest-expiring override, or ``None``.
        held_vi_names: Every VI under a hold-severity condition.
        active_ramps: One JSON dict per ramp running as of the last tick.
        availabilities: ``{vi_name: availability dict}`` for every VI.
        vi_faults: ``{vi_name: fault dict}`` for every faulted VI.
        offline_reason: ``{vi_name: reason}`` for every offline VI.
        envelope_variables: ``{vi_name: envelope-variable dict}``.
        attended: Whether a human is watching this experiment. Pushed down
            by the session layer; read by a client's permission check.
        agent_gate: The kill switch's setting, one of ``AgentGate``'s values.
        seq: Monotonic sequence number.
        ts: Unix time the snapshot was taken.
    """

    kind: ClassVar[str] = "status_snapshot"

    state: str
    run: dict[str, Any] | None = None
    instruments: dict[str, Any] = field(default_factory=dict)
    is_monitoring: bool = False
    pause_pending: bool = False
    active_run_kind: str | None = None
    override_active: bool = False
    manual_override_expires_at: float | None = None
    held_vi_names: tuple[str, ...] = ()
    active_ramps: tuple[dict[str, Any], ...] = ()
    availabilities: dict[str, Any] = field(default_factory=dict)
    vi_faults: dict[str, Any] = field(default_factory=dict)
    offline_reason: dict[str, str] = field(default_factory=dict)
    envelope_variables: dict[str, Any] = field(default_factory=dict)
    attended: bool = True
    agent_gate: str = AgentGate.ACTIVE.value
    seq: int = 0
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Defensively copy and JSON-check every collection field.

        Raises:
            TypeError: If a mapping field is not a mapping of JSON-safe
                values, if a sequence field is not a sequence, or if a scalar
                field carries the wrong type.
            ValueError: If ``agent_gate`` is not a known ``AgentGate``.
        """
        if self.run is not None:
            object.__setattr__(self, "run", _checked_mapping(self.run))
        for name in (
            "instruments",
            "availabilities",
            "vi_faults",
            "offline_reason",
            "envelope_variables",
        ):
            object.__setattr__(self, name, _checked_mapping(getattr(self, name)))
        for name in ("is_monitoring", "pause_pending",
                     "override_active", "attended"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"StatusSnapshot.{name} must be a bool")
        object.__setattr__(self, "agent_gate", AgentGate(self.agent_gate).value)
        if self.active_run_kind is not None and not isinstance(self.active_run_kind, str):
            raise TypeError("StatusSnapshot.active_run_kind must be a str or None")
        if self.manual_override_expires_at is not None and not isinstance(
            self.manual_override_expires_at, (int, float)
        ):
            raise TypeError(
                "StatusSnapshot.manual_override_expires_at must be a number or None"
            )
        if isinstance(self.held_vi_names, (str, Mapping)):
            raise TypeError("StatusSnapshot.held_vi_names must be a sequence of names")
        object.__setattr__(
            self, "held_vi_names", tuple(str(name) for name in self.held_vi_names)
        )
        if isinstance(self.active_ramps, (str, Mapping)):
            raise TypeError("StatusSnapshot.active_ramps must be a sequence of dicts")
        object.__setattr__(
            self,
            "active_ramps",
            tuple(_checked_mapping(entry) for entry in self.active_ramps),
        )


# ── The station declaration snapshot ──────────────────────────────────
#
# ``StationInfo`` and its four nested types are the DECLARATION half of
# what a client renders: what each instrument reads, what it can be asked
# to do, within which bounds, and how those capabilities group. They are
# defined here, next to the rest of the contract, and BUILT by the Station
# (``Station.station_info()``), which is the only layer that holds both the
# VI declarations and the config the bounds come from. This module
# deliberately imports nothing from the Station or the VI layer: a client
# that only consumes the contract must never drag the instrument stack in
# behind it.


def _tuple_of(
    cls: type, values: Any, field_name: str
) -> tuple[Any, ...]:
    """Coerce a declaration field into a tuple of contract instances.

    The same coercion serves construction (the Station passes real
    instances) and deserialisation (``from_json`` passes the dicts a JSON
    round trip produced), which is what keeps the round trip exact.

    Args:
        cls: The nested contract type each entry must end up as.
        values: A sequence of ``cls`` instances or of their ``to_json()``
            mappings.
        field_name: Fully qualified field name, for the error message.

    Returns:
        A tuple of ``cls`` instances.

    Raises:
        TypeError: If ``values`` is not a sequence, or an entry is neither
            a ``cls`` nor a mapping.
    """
    if isinstance(values, (Mapping, str)) or not isinstance(values, (list, tuple)):
        raise TypeError(f"{field_name} must be a sequence of {cls.__name__}")
    coerced: list[Any] = []
    for entry in values:
        if isinstance(entry, cls):
            coerced.append(entry)
        elif isinstance(entry, Mapping):
            coerced.append(cls.from_json(entry))
        else:
            raise TypeError(
                f"{field_name} entries must be {cls.__name__} or its dict, "
                f"got {type(entry).__name__}"
            )
    return tuple(coerced)


def _checked_strings(owner: str, **values: Any) -> None:
    """Validate that every named declaration field is a plain string.

    Args:
        owner: The declaring type's name, for the error message.
        **values: Field name -> value.

    Raises:
        TypeError: If any value is not a ``str``.
    """
    for name, value in values.items():
        if not isinstance(value, str):
            raise TypeError(f"{owner}.{name} must be a str, got {value!r}")


@dataclass(frozen=True)
class MonitoredInfo(_ContractMessage):
    """One ``@monitored`` reading of one instrument, as declared.

    Attributes:
        name: The method name, which is also the reading's channel key
            wherever it is polled, logged or persisted.
        unit: SI unit label ("T", "K", "%"), ``""`` for a genuinely
            dimensionless, boolean or string reading.
        description: One human-readable sentence saying what the value is.
        group: The key of the ``GroupInfo`` this reading belongs to, or
            ``""`` when it belongs to none.
        returns: Name of the method's declared return type ("float",
            "str", "float | None"), or ``""`` when it declares none.
    """

    name: str
    unit: str = ""
    description: str = ""
    group: str = ""
    returns: str = ""

    def __post_init__(self) -> None:
        """Validate the declaration strings.

        Raises:
            TypeError: If any field is not a string.
            ValueError: If ``name`` is empty.
        """
        _checked_strings(
            "MonitoredInfo",
            name=self.name,
            unit=self.unit,
            description=self.description,
            group=self.group,
            returns=self.returns,
        )
        if not self.name:
            raise ValueError("MonitoredInfo.name must be a non-empty str")


@dataclass(frozen=True)
class ControlInfo(_ContractMessage):
    """One ``@control`` action of one instrument, as declared.

    Attributes:
        name: The method name, which is what a ``Command`` names to invoke
            it.
        scope: Capability scope — ``"measurement"`` or ``"operation"``.
        action_class: How much authority this action needs, as the VI
            declared it (the action-class declaration, see GLOSSARY.md's
            **Action class**): ``"read"``, ``"recovery"``, ``"run_control"``
            or ``"envelope"``. This is what the agent gateway classifies a
            ``submit_vi_action`` by; the GUI ignores it.
        panel: Declared default placement: ``True`` on the compact monitor
            card, ``False`` in the instrument front panel only. Display
            only, never a safety mechanism.
        group: The key of the ``GroupInfo`` this action belongs to, or
            ``""`` when it belongs to none.
        params: One JSON-rendered ``ParamSpec`` per signature parameter, in
            signature order. Each carries ``name``, ``declared`` (whether a
            ``ParamSpec`` is behind the rest, as opposed to only the
            signature — a renderer needs it to tell "declared nothing" from
            "declared none of it"), ``kind`` (the scalar type name),
            ``unit``, ``description``, ``default``, ``min``, ``max`` and
            ``choices`` (``None`` when the parameter is not enumerated).
            Flat scalars only: a group never crosses the boundary as a
            value.
    """

    name: str
    scope: str = "measurement"
    action_class: str = "run_control"
    panel: bool = True
    group: str = ""
    params: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Validate the strings and JSON-check the parameter renderings.

        Raises:
            TypeError: If a string field has the wrong type, if ``panel``
                is not a bool, or if ``params`` is not a sequence of
                JSON-safe mappings.
            ValueError: If ``name`` is empty.
        """
        _checked_strings(
            "ControlInfo",
            name=self.name,
            scope=self.scope,
            action_class=self.action_class,
            group=self.group,
        )
        if not self.name:
            raise ValueError("ControlInfo.name must be a non-empty str")
        if not self.action_class:
            raise ValueError("ControlInfo.action_class must be a non-empty str")
        if not isinstance(self.panel, bool):
            raise TypeError("ControlInfo.panel must be a bool")
        if isinstance(self.params, (Mapping, str)) or not isinstance(
            self.params, (list, tuple)
        ):
            raise TypeError("ControlInfo.params must be a sequence of dicts")
        object.__setattr__(
            self, "params", tuple(_checked_mapping(spec) for spec in self.params)
        )


@dataclass(frozen=True)
class GroupInfo(_ContractMessage):
    """One titled group of one instrument's capabilities, as declared.

    The contract rendering of ``core.plan.UIGroup``: presentation and
    description only, and its ``members`` order IS the render order and the
    workflow order an agent reads off the manifest.

    Attributes:
        key: Stable identity, the value a capability's ``group`` names.
        title: Human-readable heading.
        description: Optional sentence saying what the group is for.
        members: Ordered monitored/control method names.
    """

    key: str
    title: str
    description: str = ""
    members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the strings and freeze ``members`` into a tuple.

        Raises:
            TypeError: If a string field has the wrong type, or ``members``
                is not a sequence of strings.
            ValueError: If ``key`` or ``title`` is empty.
        """
        _checked_strings(
            "GroupInfo",
            key=self.key,
            title=self.title,
            description=self.description,
        )
        for name, value in (("key", self.key), ("title", self.title)):
            if not value:
                raise ValueError(f"GroupInfo.{name} must be a non-empty str")
        if isinstance(self.members, (Mapping, str)) or not isinstance(
            self.members, (list, tuple)
        ):
            raise TypeError("GroupInfo.members must be a sequence of str")
        for member in self.members:
            if not isinstance(member, str):
                raise TypeError(f"GroupInfo.members entries must be str, got {member!r}")
        object.__setattr__(self, "members", tuple(self.members))


@dataclass(frozen=True)
class InstrumentInfo(_ContractMessage):
    """Everything one configured instrument declares about itself.

    Every configured VI appears — live or offline — because an instrument
    that is currently unreachable still declares the same capabilities;
    ``availability`` is what says whether they can be used right now.

    Attributes:
        name: The VI's configured name (``"magnet_z"``), which every
            ``Command`` targeting it uses.
        vi_class: The VI class's name, e.g. ``"SuperconductingMagnetVI"``.
        role: The config registry's role for this VI — ``"system"`` or
            ``"measurement"`` (GLOSSARY.md's
            *vi_type (config/registry)*).
        kind: The VI class's own category — ``"magnet"``,
            ``"temperature"``, ``"measurement"`` …
            (GLOSSARY.md's *vi_type (class)*).
        availability: The Availability tags standing at snapshot time,
            sorted — empty for a fully usable instrument (see GLOSSARY.md's
            **Availability tag**). This is the ONE live field in an
            otherwise static declaration, which is why a snapshot is
            rebuilt on connect and disconnect.
        monitored: The instrument's readings, in declaration order.
        controls: The instrument's actions, in declaration order.
        limits: The configured bounds of the control-validation standard's
            declared limits, ``{method: {param: {"limit": name, "min": lo,
            "max": hi}}}``. ``min``/``max`` are ``null`` where that side is
            unbounded, and for an offline instrument, whose config-derived
            bounds were never computed.
        ui_groups: The instrument's declared groups, in declared order.
        safety_flags: ``{flag: severity}``, the VI's merged safety-flag
            manifest.
    """

    name: str
    vi_class: str = ""
    role: str = ""
    kind: str = ""
    availability: tuple[str, ...] = ()
    monitored: tuple[MonitoredInfo, ...] = ()
    controls: tuple[ControlInfo, ...] = ()
    limits: dict[str, Any] = field(default_factory=dict)
    ui_groups: tuple[GroupInfo, ...] = ()
    safety_flags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce every nested field, accepting instances or their dicts.

        Raises:
            TypeError: If a field has the wrong shape or carries a
                non-JSON-safe value.
            ValueError: If ``name`` is empty.
        """
        _checked_strings(
            "InstrumentInfo",
            name=self.name,
            vi_class=self.vi_class,
            role=self.role,
            kind=self.kind,
        )
        if not self.name:
            raise ValueError("InstrumentInfo.name must be a non-empty str")
        if isinstance(self.availability, (Mapping, str)) or not isinstance(
            self.availability, (list, tuple)
        ):
            raise TypeError("InstrumentInfo.availability must be a sequence of str")
        object.__setattr__(
            self, "availability", tuple(str(tag) for tag in self.availability)
        )
        object.__setattr__(
            self,
            "monitored",
            _tuple_of(MonitoredInfo, self.monitored, "InstrumentInfo.monitored"),
        )
        object.__setattr__(
            self,
            "controls",
            _tuple_of(ControlInfo, self.controls, "InstrumentInfo.controls"),
        )
        object.__setattr__(
            self,
            "ui_groups",
            _tuple_of(GroupInfo, self.ui_groups, "InstrumentInfo.ui_groups"),
        )
        object.__setattr__(self, "limits", _checked_mapping(self.limits))
        flags = _checked_mapping(self.safety_flags)
        for flag, severity in flags.items():
            if not isinstance(severity, str):
                raise TypeError(
                    f"InstrumentInfo.safety_flags[{flag!r}] must be a str severity"
                )
        object.__setattr__(self, "safety_flags", flags)


@dataclass(frozen=True)
class StationInfo(_ContractMessage):
    """What the station is, as declared — the static half of the picture.

    Built by the Station from the VI declarations and the config alone (no
    bus traffic, ever) and re-emitted on connect and disconnect. Both
    clients render *this*, never a hand-written description of any
    instrument — the GUI into instrument panels with titled group boxes,
    the agent gateway into its capability manifest (``core.
    capability_manifest.build_manifest()`` is that JSON rendering). Neither
    adapter carries a description of its own, which is what makes the
    interface translatable rather than merely mirrored.

    Attributes:
        setup: The setup's identity — the config directory's name — or
            ``""`` for a Station assembled without one.
        tick_interval_s: The configured monitor tick period, in seconds.
        instruments: One ``InstrumentInfo`` per configured VI, live and
            offline alike, in config order.
        seq: Monotonic sequence number, incremented on every rebuild.
        ts: Unix time the declaration was captured.
    """

    kind: ClassVar[str] = "station_info"

    setup: str = ""
    tick_interval_s: float = 0.0
    instruments: tuple[InstrumentInfo, ...] = ()
    seq: int = 0
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Validate the header fields and coerce ``instruments``.

        Raises:
            TypeError: If ``setup`` is not a string, ``tick_interval_s`` is
                not a number, or ``instruments`` is not a sequence of
                ``InstrumentInfo`` (or their dicts).
        """
        _checked_strings("StationInfo", setup=self.setup)
        if isinstance(self.tick_interval_s, bool) or not isinstance(
            self.tick_interval_s, (int, float)
        ):
            raise TypeError("StationInfo.tick_interval_s must be a number")
        object.__setattr__(self, "tick_interval_s", float(self.tick_interval_s))
        object.__setattr__(
            self,
            "instruments",
            _tuple_of(InstrumentInfo, self.instruments, "StationInfo.instruments"),
        )


@dataclass(frozen=True)
class Readings(_ContractMessage):
    """One poll of the monitored fields of every instrument.

    Attributes:
        values: ``{vi_name: {field_name: value}}`` as monitored this tick.
        seq: Monotonic sequence number.
        ts: Unix time of the poll.
    """

    kind: ClassVar[str] = "readings"

    values: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Defensively copy and JSON-check ``values``.

        Raises:
            TypeError: If it is not a mapping of JSON-safe values.
        """
        object.__setattr__(self, "values", _checked_mapping(self.values))


@dataclass(frozen=True)
class Datapoint(_ContractMessage):
    """One measured point of the run in flight.

    Attributes:
        run_id: The run this point belongs to.
        index: The point's ordinal within the run, from zero.
        values: The datapoint's columns, JSON-safe.
        seq: Monotonic sequence number.
        ts: Unix time the point was measured.
    """

    kind: ClassVar[str] = "datapoint"

    run_id: str
    index: int = 0
    values: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Defensively copy and JSON-check ``values``.

        Raises:
            TypeError: If it is not a mapping of JSON-safe values.
        """
        object.__setattr__(self, "values", _checked_mapping(self.values))


@dataclass(frozen=True)
class RunStarted(_ContractMessage):
    """A run reached successful setup and is now producing data.

    Attributes:
        run_id: Identifier of the run that started.
        manifest: The run manifest, JSON-safe.
        actor: Who started it.
        request_id: The command that started it, or ``""``.
        seq: Monotonic sequence number.
        ts: Unix time of the start.
    """

    kind: ClassVar[str] = "run_started"

    run_id: str
    manifest: dict[str, Any] = field(default_factory=dict)
    actor: Actor = OPERATOR
    request_id: str = ""
    seq: int = 0
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Coerce the actor and JSON-check the manifest.

        Raises:
            TypeError: If ``actor`` or ``manifest`` has the wrong shape.
        """
        object.__setattr__(self, "actor", _as_actor(self.actor))
        object.__setattr__(self, "manifest", _checked_mapping(self.manifest))


@dataclass(frozen=True)
class RunFinished(_ContractMessage):
    """A run ended, however it ended.

    Attributes:
        run_id: Identifier of the run that finished.
        status: How it ended (``"completed"``, ``"aborted"``, ``"failed"``).
        reason: Human-readable explanation, ``""`` for a clean completion.
        manifest: The run manifest as finished, JSON-safe.
        actor: Who ended it — the actor of the command that did, or the
            ``system`` actor for a run that ended on its own (completed,
            failed, contained). Paired with ``overridden_owner`` below, this
            is what makes a **takeover** legible on the event alone.
        overridden_owner: The **run owner** whose run this ended over their
            head, as ``{"kind": ..., "id": ...}``, or ``None`` — which is
            every ordinary ending. Written only when the ending command was
            accepted as a takeover (see GLOSSARY.md's *Takeover*).
        seq: Monotonic sequence number.
        ts: Unix time of the end.
    """

    kind: ClassVar[str] = "run_finished"

    run_id: str
    status: str = ""
    reason: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    actor: Actor = OPERATOR
    overridden_owner: dict[str, Any] | None = None
    seq: int = 0
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Coerce the actor and JSON-check the manifest and the owner.

        Raises:
            TypeError: If the manifest or the overridden owner is not a
                mapping of JSON-safe values, or ``actor`` is neither an
                ``Actor`` nor its dict.
        """
        object.__setattr__(self, "manifest", _checked_mapping(self.manifest))
        object.__setattr__(self, "actor", _as_actor(self.actor))
        if self.overridden_owner is not None:
            object.__setattr__(
                self, "overridden_owner", _checked_mapping(self.overridden_owner)
            )


@dataclass(frozen=True)
class QueueChanged(_ContractMessage):
    """The run queue's contents changed.

    Attributes:
        entries: One JSON-safe dict per queued run, in run order.
        actor: Who changed it.
        request_id: The command that changed it, or ``""``.
        seq: Monotonic sequence number.
        ts: Unix time of the change.
    """

    kind: ClassVar[str] = "queue_changed"

    entries: tuple[dict[str, Any], ...] = ()
    actor: Actor = OPERATOR
    request_id: str = ""
    seq: int = 0
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Freeze ``entries`` and coerce the actor.

        Raises:
            TypeError: If ``entries`` is not a sequence of JSON-safe dicts, or
                ``actor`` is neither an ``Actor`` nor its dict.
        """
        if isinstance(self.entries, Mapping) or isinstance(self.entries, str):
            raise TypeError("QueueChanged.entries must be a sequence of dicts")
        object.__setattr__(
            self, "entries", tuple(_checked_mapping(entry) for entry in self.entries)
        )
        object.__setattr__(self, "actor", _as_actor(self.actor))


#: Everything the engine broadcasts on its one event channel. A tagged union
#: rather than a base class: the members share no behaviour, only the
#: contract that each is frozen, JSON-safe and carries ``seq``/``ts``.
Event = (
    StateChange
    | StatusSnapshot
    | StationInfo
    | Readings
    | Datapoint
    | RunStarted
    | RunFinished
    | QueueChanged
)

# The discriminator table `event_from_json()` dispatches on. Built from the
# classes themselves so a new event type is registered by adding it here and
# nowhere else.
_EVENT_TYPES: dict[str, type] = {
    cls.kind: cls
    for cls in (
        StateChange,
        StatusSnapshot,
        StationInfo,
        Readings,
        Datapoint,
        RunStarted,
        RunFinished,
        QueueChanged,
    )
}


def event_from_json(payload: Mapping[str, Any]) -> Event:
    """Rebuild whichever event a JSON payload holds.

    Args:
        payload: A mapping as produced by an event's ``to_json()``, carrying
            the ``"kind"`` discriminator.

    Returns:
        The event, of the type ``"kind"`` names.

    Raises:
        KeyError: If the payload carries no ``"kind"``.
        ValueError: If ``"kind"`` names no known event type.
    """
    kind = payload["kind"]
    event_type = _EVENT_TYPES.get(kind)
    if event_type is None:
        raise ValueError(
            f"unknown event kind {kind!r}; known kinds: "
            f"{sorted(_EVENT_TYPES)}"
        )
    return event_type.from_json(payload)
