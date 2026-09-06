"""Orchestrator — cooperative state machine for I2AS.

The Orchestrator is single-threaded. The Qt event loop is the only
concurrency mechanism. It drives procedures via a state machine and
continually monitors the system.

Failure containment: every tick runs inside an exception boundary. PyQt6
aborts the whole process on an unhandled Python exception in a slot, which
for a cryostat controller would mean vanishing with the magnet still
ramping — so any unexpected exception instead closes the data file, stops
all ramps (hardware hold), and degrades to the ERROR state.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import logging
from dataclasses import replace
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from i2as.core import events as ev
from i2as.core.availability import TAG_POLICY, Availability
from i2as.core.conditions import Condition, Verdict, decide, envelope_conditions
from i2as.core.events import ErrorEvent
from i2as.core.exceptions import I2ASActionScopeError, I2ASSafetyError
from i2as.core.logging_config import LOGGER_ROOT, STATUS_LOGGER
from i2as.core.operational_status import build_operational_status
from i2as.core.plan import (
    Command,
    EnvelopeVariable,
    ExperimentEnvelope,
    ProbeSpec,
    Target,
)
from i2as.core.ramps import RampRecord, build_ramp_records
from i2as.core.request_spool import RequestSpool
from i2as.core.run_builder import build_procedure
from i2as.core.stall_detection import (
    StallConfig,
    StallState,
    apply_stall_verdict,
    no_motion_phases_from,
)
from i2as.core.station import FaultRecord, Station
from i2as.core.tiered_trend_logger import TieredTrendLogger
# Procedures will be imported/type-checked but for now we expect a BaseProcedure mock.
# We don't import BaseProcedure directly to avoid circular dependency.

logger = logging.getLogger(__name__)

#: The capability scope a MANUAL action is dispatched with (see the
#: capability-scope standard in GLOSSARY.md and
#: ``Station.execute_vi_action()``). A human at an instrument's front panel
#: holds the widest authority there is, so the Orchestrator grants that scope
#: explicitly here, in one named place, rather than leaving
#: ``execute_vi_action()``'s restrictive default to be the accident that
#: decides it. Every other caller of the direct action path gets
#: ``"measurement"`` unless it opts in the same way.
MANUAL_ACTION_SCOPE: str = "operation"

#: The actor every transition the engine makes on its own is attributed to —
#: a tick advancing the state machine, a tripped safety flag, a run ending.
#: The counterpart of ``events.OPERATOR``, which is who a public method
#: assumes is calling when nobody says otherwise.
SYSTEM_ACTOR: ev.Actor = ev.Actor(kind=ev.ActorKind.SYSTEM, id="orchestrator", role="engine")


@dataclasses.dataclass
class _PendingCommand:
    """The verdict owed for one submitted ``Command``, while it is in flight.

    Held for the duration of the ``submit()`` call (and, for a queued manual
    action, until the tick drains it) so the refusal and success emitters can
    complete the answer without every method having to build one. See the
    **verdict standard** in ``Orchestrator``'s docstring.

    Attributes:
        request_id: The ``Command.request_id`` being answered.
        command: Which command was asked for.
        actor: Who asked.
        resolved: Whether the one verdict this command is owed has been
            emitted. Set the moment it is, so nothing can answer twice.
        deferred: Whether the answer belongs to a later tick (a queued manual
            action), which is what stops ``submit()`` from closing it with an
            optimistic ``OK`` when the method returns.
        takeover: The **takeover** record, when this command was admitted
            over another actor's run ownership. It rides on the pending
            command rather than on the engine because the verdict that
            carries it is emitted AFTER the method has returned, by which
            time the acting context that decided it is gone.
    """

    request_id: str
    command: ev.CommandName
    actor: ev.Actor
    resolved: bool = False
    deferred: bool = False
    takeover: dict[str, Any] | None = None


#: The run-scoped commands an agent may take only on a run it OWNS — the
#: **run-ownership standard** (GLOSSARY.md's *Run owner*). Every one of them
#: is destructive of somebody's work: it ends the run. Safety and recovery
#: are deliberately absent — ``emergency_standby``, ``stop_ramp``,
#: ``pause_procedure`` and ``resume_procedure`` hold the station rather than
#: destroying a result, and an actor that can see a problem must never be
#: unable to make the station safe. ``run_queue`` is absent for the same
#: reason read backwards: it starts a run, it destroys none.
OWNER_SCOPED_COMMANDS: frozenset[str] = frozenset(
    {ev.CommandName.ABORT_PROCEDURE.value}
)

#: The two arguments an owner-scoped command carries to take a run over
#: (the refuse-then-override half of the run-ownership standard). The
#: ``command`` decorator absorbs them exactly as it absorbs ``actor``, so an
#: owner-scoped method keeps the signature it had and no caller of a method
#: that is not owner-scoped can pass them by accident.
OWNERSHIP_ARGUMENTS: tuple[str, ...] = ("override_owner", "reason")


def command(method: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a public ``Orchestrator`` method as a client-issuable command.

    The **operator sentinel** in one place: the wrapper adds the
    keyword-only ``actor`` argument every command carries (defaulting to
    ``events.OPERATOR``, so no existing call site changes) and holds it, with
    the command's own name, for the duration of the call — which is what lets
    ``_change_state()`` and every verdict record who asked and what for
    without threading an argument through the state machine.

    An ``actor`` left at the sentinel does not displace an actor already
    acting, so a command that calls another command internally (``run_
    procedure()`` queueing) stays attributed to whoever started it.

    **The run-ownership admission** runs here too, for the four commands in
    ``OWNER_SCOPED_COMMANDS``: it is in the wrapper rather than in
    ``submit()`` so that the spool, the local socket and a direct Python
    caller are all covered by the one check — the same reason the kill
    switch is enforced in the engine rather than in the client. It absorbs
    ``override_owner`` and ``reason`` (``OWNERSHIP_ARGUMENTS``) before the
    method sees them, refuses with ``BLOCKED_ROLE`` when the acting agent is
    not the owner, and otherwise holds the takeover for the duration of the
    call so the verdict and ``RunFinished`` can both name it.

    Args:
        method: The public method implementing the command.

    Returns:
        The wrapped method, whose signature is *method*'s plus a
        keyword-only ``actor``.
    """
    owner_scoped = method.__name__ in OWNER_SCOPED_COMMANDS

    @functools.wraps(method)
    def wrapper(self: Orchestrator, *args: Any, actor: ev.Actor = ev.OPERATOR, **kwargs: Any) -> Any:
        refusal: tuple[str, dict[str, Any]] | None = None
        takeover: dict[str, Any] | None = None
        if owner_scoped:
            override = bool(kwargs.pop("override_owner", False))
            stated = kwargs.pop("reason", "")
            refusal, takeover = self._run_owner_admission(
                method.__name__, actor, override, str(stated or "")
            )
        pending = self._pending
        if takeover is not None and pending is not None and (
            pending.command.value == method.__name__
        ):
            pending.takeover = takeover
        with self._acting_as(actor, method.__name__, takeover=takeover):
            if refusal is not None:
                reason, detail = refusal
                logger.info("Blocked by run ownership: %s", reason)
                self._action_blocked(reason, ev.VerdictCode.BLOCKED_ROLE, detail)
                return None
            return method(self, *args, **kwargs)

    parameters = list(inspect.signature(method).parameters.values())
    actor_param = inspect.Parameter(
        "actor", inspect.Parameter.KEYWORD_ONLY, default=ev.OPERATOR
    )
    insert_at = len(parameters)
    for index, parameter in enumerate(parameters):
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            insert_at = index
            break
    parameters.insert(insert_at, actor_param)
    wrapper.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
    return wrapper




def _json_safe(value: Any) -> Any:
    """Render an arbitrary runtime value as something the contract can carry.

    The contract types validate eagerly and reject anything that is not
    JSON-safe, which is right for a declaration and wrong for a measured
    datapoint: a numpy array or a frozenset of availability tags must not be
    able to stop the engine reporting. So everything the engine puts into an
    event goes through here first — arrays and numpy scalars via ``tolist()``,
    mappings and sequences element-wise, sets sorted for a stable rendering,
    and anything else as its ``str()``, which is lossy but never fatal.

    Args:
        value: Any runtime value.

    Returns:
        A value made only of ``str``/``int``/``float``/``bool``/``None``/
        ``list``/``dict``.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:  # noqa: BLE001 — degrade, never raise while reporting
            return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (frozenset, set)):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _verdict_for_exception(
    error: BaseException | None,
) -> tuple[ev.VerdictCode, dict[str, Any] | None]:
    """Classify one failed action into a verdict code and its structured detail.

    The mapping the verdict standard promises: a ``I2ASSafetyError``
    means nothing reached the instrument (a control-limit rejection or a VI
    interlock), so it is ``BLOCKED_LIMIT`` — with ``param``/``value``/``lo``/
    ``hi``/``limit_name`` in ``detail`` when the raiser filled them in, which
    is what lets a client decide from the code and the numbers and never from
    the prose. A capability outside the caller's scope is ``BLOCKED_ROLE``:
    the call is well-formed, the authority is not there. Anything else was
    attempted and failed.

    Args:
        error: The exception raised, or ``None`` for a failure with no
            exception behind it.

    Returns:
        ``(code, detail)``; *detail* is ``None`` unless the error carries the
        structured limit fields.
    """
    if isinstance(error, I2ASActionScopeError):
        return ev.VerdictCode.BLOCKED_ROLE, None
    if isinstance(error, I2ASSafetyError):
        if getattr(error, "param", None) is None:
            return ev.VerdictCode.BLOCKED_LIMIT, None
        return ev.VerdictCode.BLOCKED_LIMIT, {
            "param": error.param,
            "value": error.value,
            "lo": error.lo,
            "hi": error.hi,
            "limit_name": error.limit_name,
        }
    return ev.VerdictCode.FAILED, None


class OrchestratorState(Enum):
    """Possible states for the Orchestrator."""
    IDLE = "IDLE"
    INITIATING = "INITIATING"
    RAMPING = "RAMPING"
    INITIATION_GATE = "INITIATION_GATE"
    READING_GATE = "READING_GATE"
    MEASURING = "MEASURING"
    SWEEPING = "SWEEPING"
    STANDBY = "STANDBY"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    EMERGENCY = "EMERGENCY"


class Orchestrator(QObject):
    """State machine driving measurements and monitoring safety.

    **The verdict standard.** Every command a client submits is answered
    exactly once. ``submit(Command)`` returns the ``request_id`` immediately
    and the answer arrives on ``verdict_emitted`` as one ``events.Verdict``
    carrying that id — ``OK`` when the command was accepted and carried out,
    otherwise the code that says why it was not: ``BLOCKED_STATE`` (the state
    machine forbids it now), ``BLOCKED_CLAIM`` (an active run claims the
    instrument), ``BLOCKED_FAULT`` (the target instrument is not controllable
    — a communication fault or an active safety hold), ``BLOCKED_LIMIT`` (a
    declared control limit or a VI interlock refused it before any hardware
    call; ``detail`` carries ``param``/``value``/``lo``/``hi``/``limit_name``
    when the refusal names them), ``BLOCKED_ENVELOPE`` (the active experiment
    envelope forbids the value), ``BLOCKED_ROLE`` (the capability is outside
    the caller's scope, or the kill switch is closed against the submitting
    agent) or ``FAILED`` (attempted, and it failed — including
    an unknown command name or arguments that do not fit the method, neither
    of which ever raises at the caller).

    Two rules make the answer single and complete. Every refusal inside a
    command method goes through one of ``_action_blocked()`` /
    ``_action_failed()`` / ``_action_succeeded()``, which emit the legacy
    ``action_*`` signal AND close the pending verdict; and a method that
    returns having emitted no refusal is an acceptance, so silence is
    ``OK`` rather than nothing. A manual action is the one asynchronous case:
    ``submit_vi_action()`` QUEUES for the tick (the single hardware writer),
    so its verdict is emitted when the drain runs it — the pending request
    travels with the queued action — and carries the action's return value in
    ``Verdict.result``.

    The public methods stay the surface the GUI and the tests call directly;
    ``submit()`` is a dispatch table onto them, keyed by ``CommandName``
    (whose values ARE the method names). Each one gains a keyword-only
    ``actor`` (see the ``command`` decorator) that defaults to the operator
    sentinel and is recorded on every ``StateChange`` and ``Verdict`` the
    call produces.

    Signals:
        states_updated (dict): Full station state emitted every monitored tick.
        monitoring_changed (bool): Emitted when monitoring starts (True) or
            stops (False) — the source of truth for GUI state like the
            Monitor window's monitoring toggle.
        state_changed (str): Emitted when orchestrator state changes. Not
            run-scoped — fires regardless of run kind.
        procedure_progress (float): 0.0 to 1.0 progress of the current run.
        procedure_finished (): Emitted when a run ends cleanly.
        run_started (dict): Run manifest emitted once per run, after the
            procedure's ``initiate()`` succeeded and its plan was
            dispatched. Keys: ``run_id``, ``procedure`` (display name),
            ``kind`` ("run" for a procedure and "probe" for a probe run —
            the run's own ``run_kind`` attribute; ``apply_probe()`` is what
            sets it to "probe"), ``params`` (merged parameter values),
            ``data_file`` (HDF5 path, captured here because the procedure
            closes its file before the run ends), and ``started_utc``
            (ISO 8601). The session layer records runs from this signal; a
            run whose setup fails emits no manifest at all.
        run_finished (dict): The same manifest re-emitted exactly once when the
            run ends, with ``finished_utc``, terminal ``status`` (``done`` /
            ``aborted`` / ``failed``), ``reason`` (error text, empty for
            ``done``/``aborted``), and ``summary``
            (the dict ``procedure.run_summary()`` returned — duck-typed, ``{}``
            for a procedure that does not override it, and
            ``{}`` rather than propagating if the override raised) added.
        error_occurred (str): Emitted when ERROR or EMERGENCY state entered,
            or a run fails. Not run-scoped — fires regardless of run kind.
            Kept as a
            thin compat wrapper: every emission here has a matching, richer
            ``error_event`` emitted alongside it.
        error_event (ErrorEvent): Structured counterpart of
            ``error_occurred``/a VI-scoped fault (``core.events.
            ErrorEvent``): ``vi_name`` (the originating instrument, or
            ``None``/comma-joined for a machine-wide or multi-VI event),
            ``kind`` (``"fault"`` — VI-scoped, quarantines only that VI;
            ``"run_failure"`` — the active run's claimed VI faulted, the run
            fails and the machine returns to IDLE; ``"safety"`` — a tripped
            safety flag, global EMERGENCY; ``"internal"`` — an unhandled
            tick-boundary exception, global ERROR; ``"safety_hold"`` — a
            VI-scoped safety-hold enforcement action, either a routine
            re-assertion (``severity="warning"``) or an escalation once the
            hold proves unenforceable (``severity="error"``) — see
            ``_enforce_safety_holds()``), ``severity``
            (``"warning"``/``"error"``/``"emergency"``), ``message``, and
            ``timestamp``. A plain per-VI fault (``kind="fault"``,
            ``severity="warning"``) and a routine hold re-assertion
            (``kind="safety_hold"``, ``severity="warning"``) both fire ONLY
            this signal, deliberately NOT ``error_occurred`` — neither was
            an ``error_occurred``-worthy event before this plan and must not
            become banner-noisy in every window that still only listens to
            the compat signal (e.g. ProcedureWindow).
        action_blocked (str): Emitted if GUI action submitted while busy.
        action_succeeded (str, str): Emitted (vi_name, method_name) after a
            submit_vi_action() GUI action executes without raising — the
            source of truth for GUI state like InstrumentPanel's lifecycle
            toggle, which must reflect confirmed instrument state rather
            than an optimistic click.
        action_failed (str, str, str): Emitted (vi_name, method_name, reason)
            when a submitted GUI action raises — including a control-limits
            rejection or a VI safety guard (e.g. switch-heater mismatch).
            The reason string is the exception message, written by the VI to
            be shown to the user verbatim.
        status_message (str): Concise, human-readable milestone of the running
            PROCEDURE. Initiation is broken into one line per distinct setup
            action ("Ramping temperature to 300 K", "Ramping field to -1 T",
            "Arming DC resistance measurement"), followed by "Waiting N s at
            setpoint", "Measuring point 13/101", "Point 14/101: ramping field
            -> 0.55 T", etc. Labels/units come from each VI's setpoint metadata
            via the Station, so every procedure gets a status feed with no
            per-procedure code; consumed by the Procedure window's status log.
            Distinct from the per-tick detail stream on the Monitor log.
        verdict_emitted (events.Verdict): The single answer to one submitted
            ``Command`` — see the verdict standard above. Never emitted for a
            method called directly rather than through ``submit()``.
        event_emitted (events.Event): The engine's one event stream, carrying
            ``StateChange`` (every transition, with its cause and actor),
            ``StatusSnapshot`` (once per tick and on every state change),
            ``StationInfo`` (at construction and after connect/disconnect),
            ``Readings`` (each monitored poll), ``Datapoint`` (each measured
            point), ``RunStarted``/``RunFinished`` and ``QueueChanged``
            (after every queue mutation, carrying the whole queue as
            JSON-safe entries and the actor behind the change). Every
            payload is a copy, every event carries a monotonic ``seq``
            shared with the verdicts, and the existing per-purpose signals
            keep emitting unchanged alongside it.
    """

    verdict_emitted = pyqtSignal(object)  # events.Verdict — one per submitted Command
    event_emitted = pyqtSignal(object)  # events.Event — the engine's one event stream
    states_updated = pyqtSignal(dict)
    monitoring_changed = pyqtSignal(bool)
    state_changed = pyqtSignal(str)
    procedure_progress = pyqtSignal(float)
    procedure_finished = pyqtSignal()
    run_started = pyqtSignal(dict)  # run manifest at successful setup
    run_finished = pyqtSignal(dict)  # same manifest + finished_utc/status/reason
    error_occurred = pyqtSignal(str)
    error_event = pyqtSignal(object)  # ErrorEvent — structured error/fault payload
    action_blocked = pyqtSignal(str)
    action_succeeded = pyqtSignal(str, str)
    action_failed = pyqtSignal(str, str, str)
    instrument_reconnected = pyqtSignal(str)  # offline VI brought live via connect_instrument()
    instrument_disconnected = pyqtSignal(str)  # live VI released via disconnect_instrument()
    measurement_ready = pyqtSignal(dict)  # emitted after each measure() with last_datapoint
    operational_status = pyqtSignal(dict)  # per-tick runtime status record (troubleshooting)
    ramps_updated = pyqtSignal(list)  # list[RampRecord] — every ramp running right now
    status_message = pyqtSignal(str)  # concise, human-readable PROCEDURE milestone line

    def __init__(
        self,
        station: Station,
        tick_interval_ms: int = 3000,
        manual_override_timeout_s: float = 300.0,
        stall_seconds: float = 18.0,
        hold_enforcement_interval_s: float = 10.0,
        hold_enforcement_max_attempts: int = 3,
        run_catalog: Mapping[str, type] | None = None,
        next_procedure: Callable[[], Any] | None = None,
        queue_snapshot: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
        request_spool: RequestSpool | None = None,
    ) -> None:
        """Build the engine over a Station.

        Args:
            station: The L2 Station this engine is the sole writer to.
            tick_interval_ms: The cooperative tick period.
            manual_override_timeout_s: How long an ``acknowledge()`` unlocks
                manual control for before it re-locks itself.
            stall_seconds: How long a ramp may make no progress before the
                stall detector reports it.
            hold_enforcement_interval_s: Minimum gap between two ``standby()``
                re-assertions on the same held VI.
            hold_enforcement_max_attempts: Consecutive failed re-assertions
                before an unenforceable hold is escalated.
            run_catalog: ``{class name: procedure class}`` used to
                build a run from a ``Command``'s dict payload (see
                ``submit()``). Supplied by whoever owns discovery — the GUI,
                a test, or an agent gateway — because the engine may not
                import ``i2as.procedures`` (contract C5). Empty by
                default, which refuses a dict-payload run with a ``FAILED``
                verdict naming the missing class; passing a run object to
                ``run_procedure()`` directly needs no catalog at all.
            next_procedure: The **pull seam**: a callable returning the next
                run to start (a built procedure), or ``None``
                when nothing is waiting. ``run_queue()`` calls it once the
                engine's own handed-over runs are exhausted, so a client can
                hold the queue as data while the engine keeps sole authority
                over *when* a run starts (``session/run_queue.py``). Also
                settable afterwards as a plain attribute, which is what
                ``main.py`` does — the queue is built after the engine is.
            queue_snapshot: A callable returning that queue's waiting entries
                as JSON-safe dicts, merged into every ``QueueChanged`` event
                so one snapshot describes the whole queue. Also a settable
                attribute.
            request_spool: The **Request spool** this engine drains, or
                ``None`` (the default) for an engine with no file-based write
                path at all. Given one, the tick reads it at the same point it
                drains queued manual actions, and both broadcast streams are
                mirrored into it so an out-of-process client can answer its
                reads without asking. It adds no thread, no socket and no
                timer: the drain is a plain synchronous call inside the tick.
                Whoever supplies it also supplies its permission hook (see
                ``session/gateway/roles.py``'s ``authorize_spooled()``),
                because the role model lives in a layer this one may not
                import.
        """
        super().__init__()
        self._station = station
        self._state = OrchestratorState.IDLE
        self._procedure: Any = None
        self._run_catalog: dict[str, type] = dict(run_catalog or {})
        #: The pull seam (see the constructor): the engine ASKS for the next
        #: run rather than being pushed one. Public and settable, because the
        #: queue that supplies it is built after the engine.
        self.next_procedure: Callable[[], Any] | None = next_procedure
        #: That queue's waiting entries, as JSON-safe dicts, for QueueChanged.
        self.queue_snapshot: Callable[[], Sequence[Mapping[str, Any]]] | None = (
            queue_snapshot
        )

        # The control contract's bookkeeping (see the verdict standard in the
        # class docstring). _pending is the verdict owed for the command
        # currently in flight (None outside submit(), and outside the drain of
        # a queued action that carried one); _actor / _acting_command are the
        # actor and command name held for the duration of a decorated call,
        # read by _change_state() and by every verdict. _seq is the ONE
        # monotonic counter across events and verdicts, so a client can order
        # everything the engine said from a single stream.
        self._pending: _PendingCommand | None = None
        self._actor: ev.Actor | None = None
        self._acting_command: str | None = None
        # The **takeover** record of the command in flight, when it was
        # admitted over another actor's run ownership: {"owner", "reason"},
        # held for the duration of the call by _acting_as() so the verdict
        # and RunFinished both name it. None for every ordinary command.
        self._takeover: dict[str, Any] | None = None
        self._seq = 0
        # The last command the engine ACCEPTED (see _remember_accepted()),
        # carried onto every operational-status record so the runtime log
        # names who last got the engine to do something and under which
        # request id — the key that joins it to a client's own action trail.
        self._last_accepted: _PendingCommand | None = None
        # Index of the next datapoint within the active run, for Datapoint
        # events; reset at every run start.
        self._datapoint_index = 0
        self._procedure_queue: list[Any] = []
        # Who queued each run handed over as a live object, keyed by id() —
        # the queue holds the objects themselves (callers index into it), so
        # the actor rides alongside rather than inside. Entries are dropped
        # the moment their run leaves the queue.
        self._queued_actors: dict[int, ev.Actor] = {}
        self._gui_action_queue: list[dict[str, Any]] = []
        # The Request spool, when this engine has one: the file-based write
        # path a client outside the process submits through. It is drained by
        # the tick beside the manual-action queue and mirrors both broadcast
        # streams; nothing about it runs off the tick.
        self._request_spool: RequestSpool | None = request_spool
        if request_spool is not None:
            request_spool.ensure()
            self.verdict_emitted.connect(request_spool.record_verdict)
            self.event_emitted.connect(request_spool.record_event)
        self._active_system_vis: set[str] = set()

        self._wait_started = False
        self._wait_start_time = 0.0
        self._current_wait_time = 0.0
        self._standby_dispatched = False

        # Gate framework: procedure-declared waits that replace wait_s for the
        # RAMPING->MEASURING transition when the procedure declares any.
        # _first_measurement distinguishes the run's very first transition
        # (initiation_gates()) from every subsequent one (reading_gates()).
        self._pending_gates: list = []
        self._first_measurement = True

        # The single acknowledge()-driven override, time-boxed rather than a
        # standing bypass (GLOSSARY.md's **Hold acknowledge**). Both windows
        # are read by _manual_action_admissible() and reset by acknowledge();
        # _emergency_override_until additionally resets to None on every
        # fresh EMERGENCY entry (_enter_emergency()) and on the eventual
        # return to IDLE (_acknowledge_emergency()), so a new emergency
        # always starts locked.
        #
        # _emergency_override_until: set by acknowledge() while in EMERGENCY
        # and its condition is still active: unlocks submit_vi_action() for
        # manual front-panel recovery (e.g. cycling a switch heater by hand)
        # without leaving EMERGENCY. Procedures stay refused regardless —
        # their gates check self._state, not this window.
        self._emergency_override_until: float | None = None
        # _hold_override_until: condition.key -> expiry unix timestamp, one
        # entry per currently-acknowledged hold-severity condition (e.g.
        # "safety:coolant_low"). Pruned by expiry only (never merely because
        # the condition cleared) — see _tick_body()'s pruning step — so a
        # flapping flag doesn't force a fresh acknowledge on every re-trip
        # within the same window.
        self._hold_override_until: dict[str, float] = {}
        self._manual_override_timeout_s = float(manual_override_timeout_s)

        # Safety-hold enforcement (see _enforce_safety_holds()): the
        # level-triggered invariant that keeps every held, un-overridden VI
        # at standby for as long as its hold persists, not just at onset.
        # _hold_enforcement_last_s: vi_name -> the last unix time a standby()
        # RE-ASSERTION was attempted on it (rate limit; absent = never
        # attempted, so a freshly held VI fires immediately). _hold_
        # enforcement_attempts: vi_name -> consecutive raise count since the
        # last time standby_status() left "away". _hold_enforcement_
        # escalated: vi_names whose CRITICAL escalation has already fired
        # this episode, so it logs/emits once, not on every attempt past the
        # threshold. All three entries for a VI are cleared the instant it
        # stops being held-and-away — see _enforce_safety_holds()'s docstring.
        self._hold_enforcement_last_s: dict[str, float] = {}
        self._hold_enforcement_attempts: dict[str, int] = {}
        self._hold_enforcement_escalated: set[str] = set()
        self._hold_enforcement_interval_s = float(hold_enforcement_interval_s)
        self._hold_enforcement_max_attempts = int(hold_enforcement_max_attempts)

        # Operational-status reporting (runtime troubleshooting signal).
        # The setup name is resolved once here, from the fully built Station,
        # for the record standard's ``setup`` field.
        self._setup_name: str | None = station.setup_name()
        self._state_entered_at = time.time()
        self._prev_gaps: dict[str, float] = {}
        self._operational_status: dict = {}
        self._status_logger = logging.getLogger(STATUS_LOGGER)
        self._stall_state = StallState()
        # stall_seconds is converted to a tick count exactly once here, using
        # this Orchestrator's own tick_interval_ms — see StallConfig's
        # docstring for why the conversion cannot happen per-tick or in config
        # directly.
        self._stall_config = StallConfig(
            stall_seconds=stall_seconds, tick_interval_ms=tick_interval_ms
        )

        # Tiered trend-history writer: the disk-backed store of per-tick
        # station readings, downsampled live into the raw/3min/hourly tiers
        # (see TieredTrendLogger's docstring and GLOSSARY.md "Trend tier").
        # No arguments: its defaults resolve the i2as.trend_* loggers
        # already configured by setup_logging(), so the Orchestrator never
        # needs to know where trend logs live.
        self._tiered_trend_logger = TieredTrendLogger()

        # Session envelope (sample-specific bounds, narrower than config
        # limits) — set by the session layer, enforced here so it binds every
        # writer (GUI and agents alike). None = no envelope active.
        self._session_envelope: ExperimentEnvelope | None = None

        # Attendance and the kill switch: two session-owned policy VALUES
        # pushed down here for the same reason the envelope is, and enforced
        # at the same single choke point. Contract C12 forbids the engine
        # from reading the session layer, so a policy the session owns has to
        # arrive as a value or it cannot be enforced where every writer
        # passes. Attendance defaults to "a human is watching" (the
        # restrictive reading for an agent) and the gate to ACTIVE.
        self._attended: bool = True
        self._agent_gate: ev.AgentGate = ev.AgentGate.ACTIVE

        # Active run manifest: captured at run_started (the data file path is
        # gone by the time the run ends) and re-emitted once on run_finished.
        self._active_run_manifest: dict[str, Any] | None = None
        self._run_counter = 0

        # The **run owner**: the Actor that started the run in flight, taken
        # at _start_run() and cleared on every teardown path. A fact, never a
        # lease — nothing is reserved and nothing can be left locked — read
        # by _run_owner_admission() and published on every StatusSnapshot.
        self._run_owner: ev.Actor | None = None
        # The owner a run pulled off a queue inherits: whoever queued it,
        # remembered here between _take_queued()/_pull_next_run() and the
        # _start_run() that follows. None when the next run is not a queued
        # one, in which case the acting actor owns it.
        self._next_run_owner: ev.Actor | None = None

        # run_summary() hand-off: collected from self._procedure by
        # _emit_run_finished() for the "done" path, where self._procedure is
        # still set. The abort/fail/emergency paths clear self._procedure in
        # _abort_active_procedure() BEFORE calling _emit_run_finished(), so
        # that method caches the summary here first — _emit_run_finished()
        # prefers self._procedure when present, else falls back to this.
        self._pending_run_summary: dict[str, Any] = {}

        # Claims + admission gate: the active run's claimed_vi_names(),
        # captured once at _start_run() and cleared on EVERY teardown path
        # (finish, abort, fail, emergency — see _abort_active_procedure()/
        # _finish_run()). None while no run is active, or while the active
        # run claims everything (the default for every procedure).
        self._active_claims: set[str] | None = None

        # Condition-registry onset tracking (the System-Condition standard,
        # core/conditions.py): the set of Station condition keys
        # (comm:<vi_name>, safety:<flag>) active as of the last tick, used
        # to detect NEW conditions — a new comm condition on an unwatched
        # VI emits one warning error_event (not one per tick); a new
        # hold-severity condition dispatches one standby() to each VI it
        # affects (not one per tick a hold persists). Replaces the old
        # separate _known_fault_vis/_known_safety_holds trackers — one key
        # set now serves both, since a comm condition IS hold-severity
        # exactly like a safety-hold condition. Station is the source of
        # truth; this is only a transition-detection cache.
        self._known_condition_keys: set[str] = set()

        # Ramp tracking (see _publish_ramps): the running-ramp records as of
        # the last tick, and the per-tick memo of the Station snapshot they
        # and the operational-status record are both built from. The memo is
        # cleared at the top of every _tick_body() so a tick polls each VI's
        # ramp state exactly once no matter how many consumers read it.
        self._active_ramps: list[RampRecord] = []
        self._tick_ramp_info: dict[str, dict] | None = None

        self._pre_pause_state = OrchestratorState.IDLE
        self._paused_wait_elapsed = 0.0
        # Deferred pause (GLOSSARY.md's "Pause boundary"): set ONLY when a
        # pause is requested in MEASURING, where pausing would strand a
        # settled-but-unread point. The SWEEPING branch of _tick_body()
        # honours it once measure() has saved its datapoint. Every other
        # state pauses immediately, so this stays False there. Cleared on
        # every run teardown path so a request can never outlive its run.
        self._pause_requested = False
        # Last targets dispatched to the Station — re-dispatched on resume,
        # because pause_procedure() holds the hardware (which forgets its ramp).
        self._last_system_targets: dict[str, Target] = {}

        # Monitoring starts OFF: instruments are not polled until
        # start_monitoring() is called (typically from the Monitor window,
        # after the instruments have been initiated), so a fresh launch does
        # not immediately fire communication errors at not-yet-ready hardware.
        # The tick timer itself always runs — it is what processes GUI actions
        # (including "Initiate All") and drives the state machine.
        self._monitoring = False

        self._timer = QTimer(self)
        self._timer.setInterval(tick_interval_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        # The static half of the picture, if the Station declares one: emitted
        # once here and again after every connect/disconnect.
        self._emit_station_info()

    # ------------------------------------------------------------------
    # Public API — the control-contract port (the verdict standard)
    # ------------------------------------------------------------------

    def submit(self, command: ev.Command) -> str:
        """Carry out one client ``Command`` and answer it with one ``Verdict``.

        The engine port: the single entry point a client that speaks the
        control contract uses, as opposed to calling the public methods
        directly (which the GUI and the tests still do). Dispatch is a lookup,
        not a table — every ``CommandName``'s value IS the name of the method
        implementing it — so a command can never exist here without a method
        behind it, or a method without a command in front of it (conformance
        diffs the two).

        The answer arrives on ``verdict_emitted``, exactly once, carrying
        ``command.request_id``. It is emitted before this returns for every
        synchronous command; ``SUBMIT_VI_ACTION`` is the exception — it queues
        for the tick, the single hardware writer, so its verdict is emitted
        when the drain executes it, with the call's return value in
        ``Verdict.result``. Nothing here raises at the caller: an unknown
        name, a malformed payload, or arguments that do not fit the method are
        all answered ``FAILED``.

        **Argument conventions.** ``command.args`` is JSON, so the few
        commands whose methods take objects name them instead:

        * ``RUN_PROCEDURE`` / ``QUEUE_PROCEDURE`` take ``procedure`` (the
          class name, resolved through the ``run_catalog`` given at
          construction), plus ``params``, ``sample_info``, ``data_directory``,
          ``file_prefix`` and ``experiment_info`` — the arguments
          ``run_builder.build_procedure()`` assembles a run from — and the
          optional ``probe_spec`` (``{"n_points": .., "averaging": ..,
          "max_wait_s": ..}``, any key optional), which reduces the same run
          to a **probe run**: the same class and the same instruments, cut
          down to a few points so it costs minutes instead of hours. The run
          it starts declares ``kind == "probe"`` in its manifest, its
          ``RunStarted`` event and its data file.
        * ``SET_EXPERIMENT_ENVELOPE`` takes ``envelope``: the
          ``{vi_name: {min_value, max_value, state_key}}`` mapping
          ``ExperimentEnvelope.from_dict()`` reads, or ``null`` to clear it.
        * ``SUBMIT_VI_ACTION`` takes ``vi_name``, ``method_name`` and the
          capability's own parameters as FLAT scalars beside them (shaped by
          the control's ``ParamSpec``s), e.g.
          ``{"vi_name": "magnet_z", "method_name": "set_field",
          "target_T": 1.5}``.
        * Every other command's ``args`` are the method's own keyword
          arguments.

        **The kill switch.** Before anything is dispatched, a command whose
        actor is an ``agent`` is checked against the gate
        ``set_agent_gate()`` installed: unless the gate is ``ACTIVE`` the
        command is answered ``BLOCKED_ROLE`` with the gate named in
        ``detail``, and nothing runs. ``EMERGENCY_STANDBY`` is exempt at
        every setting, and an ``operator`` or ``system`` actor is never
        checked at all — a kill switch that could lock the human out of
        their own instrument would be a hazard rather than a safeguard.

        **Run ownership.** The four run-scoped commands in
        ``OWNER_SCOPED_COMMANDS`` are admitted a second time, by the
        ``command`` decorator: an ``agent`` that is not the **run owner** is
        refused ``BLOCKED_ROLE`` unless it carries ``override_owner`` and a
        ``reason``, in which case the action is recorded as a **takeover**.
        The check lives in the decorator rather than here so that a direct
        Python caller passes it too; see ``_run_owner_admission()`` for the
        rule in full. The two arguments are absorbed there and never reach
        the method.

        Args:
            command: The client's request.

        Returns:
            ``command.request_id``, immediately — the correlation id the
            verdict and every event this command causes carry back.
        """
        pending = _PendingCommand(
            request_id=command.request_id, command=command.name, actor=command.actor
        )
        previous, self._pending = self._pending, pending
        try:
            gate_refusal = self._agent_gate_refusal(command)
            if gate_refusal is not None:
                reason, detail = gate_refusal
                logger.info("Blocked by the agent gate: %s", reason)
                self._emit_verdict(
                    ev.VerdictCode.BLOCKED_ROLE, reason=reason, detail=detail
                )
                return command.request_id
            method_name = ev.CommandName(command.name).value
            method = getattr(self, method_name, None)
            if not callable(method):
                raise AttributeError(
                    f"no Orchestrator method implements {method_name!r}"
                )
            method(actor=command.actor, **self._command_arguments(command))
        except Exception as exc:  # noqa: BLE001 — a command never raises at its caller
            logger.exception("submit(%s) failed", command.name)
            self._emit_verdict(ev.VerdictCode.FAILED, reason=str(exc) or repr(exc))
        finally:
            # Silence is acceptance: a method that returned having emitted no
            # refusal did what was asked. A queued manual action is deferred
            # and answered by the drain instead.
            if not pending.deferred:
                self._emit_verdict(ev.VerdictCode.OK)
            self._pending = previous
        return command.request_id

    def _agent_gate_refusal(
        self, command: ev.Command
    ) -> tuple[str, dict[str, Any]] | None:
        """Decide whether the kill switch refuses this command before dispatch.

        Gates ``agent`` actors only, and never ``emergency_standby``. The
        engine is the enforcement point rather than the client that submitted
        the command, so a client cannot reach the instruments by declining to
        police itself.

        Args:
            command: The command about to be dispatched.

        Returns:
            ``(reason, detail)`` when the gate refuses it — *detail* naming
            the ``gate`` and the ``actor_kind`` — or ``None`` when the
            command may proceed to its method.
        """
        if command.actor.kind is not ev.ActorKind.AGENT:
            return None
        if self._agent_gate is ev.AgentGate.ACTIVE:
            return None
        if command.name is ev.CommandName.EMERGENCY_STANDBY:
            return None
        reason = (
            f"Agent access is {self._agent_gate.value}: "
            f"{command.name.value!r} from agent {command.actor.id!r} is refused. "
            "Only emergency standby passes while the kill switch is closed."
        )
        return reason, {
            "gate": self._agent_gate.value,
            "actor_kind": command.actor.kind.value,
            "actor_id": command.actor.id,
        }

    def _run_owner_admission(
        self, command_name: str, actor: ev.Actor, override: bool, reason: str
    ) -> tuple[tuple[str, dict[str, Any]] | None, dict[str, Any] | None]:
        """Judge one owner-scoped command against the **run-ownership standard**.

        The standard, in full (GLOSSARY.md's *Run owner* and *Takeover*):

        * The owner of a run is the ``Actor`` that started it. Ownership is a
          FACT, not a lease — nothing is reserved, nothing expires, nothing
          can be left locked, and there is no handover protocol.
        * **The human is never gated.** An ``operator`` or ``system`` actor
          is not checked at all, exactly as the kill switch does not check
          them: a rule that could stand between the physicist and their own
          cryostat would be a hazard rather than a safeguard.
        * **Safety is never gated.** Only the four commands in
          ``OWNER_SCOPED_COMMANDS`` reach here; emergency standby, stopping a
          ramp, pausing, resuming, the gate and attendance never do.
        * **Destructive run-scoped actions are owner-scoped for agents.** An
          ``agent`` that is not the owner is refused ``BLOCKED_ROLE`` with
          ``detail.rule == "run_owner"``, naming the owner and how to
          proceed.
        * **Refuse, then override.** The same command carrying
          ``override_owner`` and a non-empty ``reason`` is admitted as a
          **takeover** and recorded as one; an override with no reason is
          refused with ``detail.rule == "override_reason_required"``, because
          a takeover whose record does not say why is exactly the
          unaccountable act this rule exists to prevent. ``BLOCKED_ROLE``
          rather than ``FAILED`` for both: nothing was attempted, and the
          verdict standard reserves ``FAILED`` for what was.

        Args:
            command_name: The owner-scoped command being admitted.
            actor: Who is asking.
            override: Whether the caller asked to take the run over.
            reason: Why they are taking it over; required with *override*.

        Returns:
            ``(refusal, takeover)`` — exactly one of which is not ``None``,
            or both ``None`` when ownership has nothing to say (the human,
            the owner themselves, or no run at all). *refusal* is the
            ``(reason, detail)`` pair ``_action_blocked()`` takes; *takeover*
            is the ``{"owner", "reason"}`` record the verdict and
            ``RunFinished`` carry.
        """
        owner = self._run_owner
        if actor.kind is not ev.ActorKind.AGENT or owner is None:
            return None, None
        if owner.kind is actor.kind and owner.id == actor.id:
            return None, None

        detail: dict[str, Any] = {
            "owner": owner.ref(),
            "actor_id": actor.id,
            "command": command_name,
            "run_id": str((self._active_run_manifest or {}).get("run_id", "")),
        }
        if not override:
            return (
                (
                    f"This run is owned by {owner.kind.value} {owner.id!r}, and "
                    f"agent {actor.id!r} is not its owner, so {command_name!r} is "
                    f"refused. Ask the owner or the operator to do it; if you "
                    f"must do it yourself, re-send the same command with "
                    f"override_owner=true and a reason saying why — the takeover "
                    f"is recorded on the run and in the agent feed.",
                    {**detail, "rule": "run_owner"},
                ),
                None,
            )
        if not reason.strip():
            return (
                (
                    f"Taking over {owner.kind.value} {owner.id!r}'s run needs a "
                    f"reason: re-send {command_name!r} with override_owner=true "
                    f"and a non-empty reason saying why you are taking the run "
                    f"over.",
                    {**detail, "rule": "override_reason_required"},
                ),
                None,
            )
        logger.warning(
            "Takeover: agent %r is taking %s %r's run over with %s — %s",
            actor.id,
            owner.kind.value,
            owner.id,
            command_name,
            reason,
        )
        return None, {"owner": owner.ref(), "reason": reason}

    def _command_arguments(self, command: ev.Command) -> dict[str, Any]:
        """Convert one command's JSON ``args`` into the method's arguments.

        See ``submit()`` for the per-command conventions this implements.

        Args:
            command: The command being dispatched.

        Returns:
            The keyword arguments to call the implementing method with.

        Raises:
            KeyError: If a required argument is missing.
            TypeError: If an argument has the wrong shape.
            ValueError: If a named procedure class is not in the run
                catalog, or an envelope is malformed.
            I2ASError: If the procedure refuses to be built.
        """
        args = dict(command.args)
        name = command.name
        if name in (ev.CommandName.RUN_PROCEDURE, ev.CommandName.QUEUE_PROCEDURE):
            return {"procedure": self._build_run(args)}
        if name is ev.CommandName.SET_EXPERIMENT_ENVELOPE:
            envelope = args.get("envelope")
            return {
                "envelope": (
                    ExperimentEnvelope.from_dict(envelope) if envelope else None
                )
            }
        if name is ev.CommandName.SUBMIT_VI_ACTION:
            vi_name = args.pop("vi_name")
            method_name = args.pop("method_name")
            return {"vi_name": vi_name, "method_name": method_name, **args}
        return args

    def _build_run(self, args: dict[str, Any]) -> Any:
        """Build a procedure from a command's dict payload.

        The engine may not import ``i2as.procedures`` (contract C5), so a
        class name is resolved through the ``run_catalog`` whoever owns
        discovery handed the constructor. The run is assembled by
        ``core.run_builder.build_procedure()``, the headless construction
        path — so a run submitted as JSON is the same object the GUI and the
        queue build.

        Args:
            args: The command's arguments. The payload may carry
                ``probe_spec``, which reduces the built run to a probe (see
                ``ProbeSpec``).

        Returns:
            The ready procedure instance.

        Raises:
            ValueError: If no class of that name is in the run catalog, or a
                ``probe_spec`` is malformed.
            I2ASError: If the run refuses to be built (see
                ``run_builder.PROCEDURE_BUILD_ERRORS``).
        """
        class_name = str(args.get("procedure", ""))
        run_class = self._run_catalog.get(class_name)
        if run_class is None:
            raise ValueError(
                f"unknown procedure {class_name!r}: the run catalog holds "
                f"{sorted(self._run_catalog)}"
            )
        params = dict(args.get("params") or {})
        probe_spec = args.get("probe_spec")
        return build_procedure(
            run_class,
            station=self._station,
            params=params,
            sample_info=dict(args.get("sample_info") or {}),
            data_directory=str(args.get("data_directory") or ""),
            file_prefix=str(args.get("file_prefix") or ""),
            experiment_info=args.get("experiment_info"),
            probe=ProbeSpec.from_json(probe_spec) if probe_spec else None,
        )

    # ------------------------------------------------------------------
    # Public API — monitoring lifecycle
    # ------------------------------------------------------------------

    def is_monitoring(self) -> bool:
        """Return True while the per-tick monitoring cycle is active."""
        return self._monitoring

    @property
    def state(self) -> str:
        """Current state machine value (e.g. ``"IDLE"``, ``"EMERGENCY"``).

        The GUI's only sanctioned way to read current state: widgets whose
        visibility depends on state (e.g. the Acknowledge-Emergency button)
        must read this once at construction time to sync with a state
        entered before they existed — ``state_changed`` alone only reports
        *future* transitions.
        """
        return self._state.value

    @command
    def start_monitoring(self) -> bool:
        """Begin the per-tick monitoring cycle (state polling + stall detection).

        Idempotent. Until this is called, ticks process GUI actions and the
        state machine but touch no instrument — call it once the instruments
        have been initiated and are ready to be polled.

        Returns:
            True (monitoring is active when this returns).
        """
        if self._monitoring:
            return True
        self._monitoring = True
        logger.info("Monitoring started")
        self._emit_status("Monitoring started")
        # A client reads is_monitoring() off its mirror, and this is not a
        # state change, so refresh the snapshot here or the toggle shows the
        # old answer until the next tick.
        self._emit_status_snapshot()
        self.monitoring_changed.emit(True)
        return True

    @command
    def stop_monitoring(self) -> bool:
        """Stop the per-tick monitoring cycle (e.g. to debug an instrument).

        Refused (with an ``action_blocked`` signal) outside IDLE/ERROR for two
        independent reasons, either of which alone justifies the refusal:
        while a procedure runs or hardware ramps, the stall detector and stale
        detection that live in the monitoring cycle must keep running; and
        hold enforcement (see ``_enforce_safety_holds()``, GLOSSARY.md's
        **Hold enforcement**) is a level-triggered invariant re-checked every
        tick, so stopping the cycle would leave a held VI free to sit away
        from standby with nothing re-asserting it. Idempotent when already
        stopped.

        Returns:
            True if monitoring is stopped when this returns, False if the
            request was refused.
        """
        if not self._monitoring:
            return True
        if self._state not in (OrchestratorState.IDLE, OrchestratorState.ERROR):
            msg = (
                f"Cannot stop monitoring in state {self._state.name}: "
                "the stall detector and safety-hold enforcement must keep "
                "running while hardware is active."
            )
            logger.info("Blocked stop_monitoring: %s", msg)
            self._action_blocked(msg)
            return False
        self._monitoring = False
        logger.info("Monitoring stopped")
        self._emit_status("Monitoring stopped")
        self._emit_status_snapshot()  # see start_monitoring()
        self.monitoring_changed.emit(False)
        return True

    def shutdown(self) -> None:
        """Stop the tick timer permanently (application exit / test teardown).

        After this no tick ever fires again: no polling, no action
        processing, no state-machine advancement. Used by tests to guarantee
        a tick can never land while the GUI widget tree is being destroyed.
        Idempotent.
        """
        self._timer.stop()
        logger.info("Orchestrator shut down — tick timer stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @command
    def set_experiment_envelope(self, envelope: ExperimentEnvelope | None) -> None:
        """Install (or clear) the active experiment's session envelope.

        Called by the session layer on experiment start/close. Config
        ``init_params`` limits protect the instrument; the envelope protects
        the mounted sample with narrower per-experiment bounds. Enforcement
        happens here in the Orchestrator — the single choke point every writer
        goes through — in two places: every submitted ``Target`` for a bounded
        VI is validated before dispatch, every manual action on the direct
        action path is validated the same way (see
        ``_envelope_refusal()``), and every tick each bound with a
        ``state_key`` is checked against the VI's live reading (a violation
        enters EMERGENCY exactly like a tripped safety flag).

        Args:
            envelope: The bounds to enforce, or ``None`` to clear them.
        """
        self._session_envelope = envelope
        if envelope is None:
            logger.info("Session envelope cleared")
        else:
            logger.info("Session envelope set: %r", envelope)

    @command
    def set_attendance(self, attended: bool) -> None:
        """Record whether a human is watching the running experiment.

        The second session-owned policy value pushed DOWN into the engine,
        mirroring ``set_experiment_envelope()`` exactly and for the same
        reason: contract C12 forbids anything below the GUI from importing
        the session layer, so a policy the session record owns cannot be read
        at the place it has to be enforced — it has to arrive as a value.
        The session layer calls this from the experiment record's ``attended``
        flag, and the GUI's takeover toggle calls it directly.

        The engine itself does not restrict anything on attendance: it stores
        the value and mirrors it on every ``StatusSnapshot``, which is what
        lets a client's permission check (the agent gateway's role matrix)
        read the same fact every other client sees, rather than each client
        keeping a copy of its own. Defaults to ``True`` — assume a human is
        present until told otherwise, which is the restrictive reading for an
        agent asking to act alone.

        Args:
            attended: ``True`` when a human is watching, ``False`` when the
                experiment is running unattended.
        """
        self._attended = bool(attended)
        logger.info(
            "Attendance set: %s", "attended" if self._attended else "unattended"
        )
        self._emit_status_snapshot()

    @command
    def set_agent_gate(self, state: ev.AgentGate | str) -> None:
        """Set the kill switch: how much of the engine agents may reach.

        Pushed down as a value like attendance and the session envelope, and
        enforced HERE, in ``submit()``, before any command is dispatched —
        the single choke point every writer passes. A gate enforced in the
        client that submits the command would only bind a client that chose
        to be bound.

        The human path is never gated: only an ``agent`` actor is checked,
        and ``emergency_standby`` passes at every setting, so flipping this
        can never leave anyone — human or agent — unable to make the station
        safe.

        Args:
            state: An ``events.AgentGate`` member or its string value
                (``"active"``, ``"read_only"``, ``"revoked"``).

        Raises:
            ValueError: If *state* is not a known ``AgentGate``.
        """
        self._agent_gate = ev.AgentGate(state)
        logger.info("Agent gate set: %s", self._agent_gate.value)
        self._emit_status_snapshot()

    def envelope_variables(self) -> dict[str, EnvelopeVariable]:
        """Return each VI's enveloped quantity and the setup's bounds on it.

        The read surface the Start Experiment dialog's envelope editor
        pre-fills from, so the operator NARROWS the setup's own limits rather
        than composing an envelope from nothing. A pure passthrough to
        ``Station.envelope_variables()`` — the GUI talks only to the
        Orchestrator's public API.

        Returns:
            ``{vi_name: EnvelopeVariable}``; empty when no VI declares a
            setpoint capability.
        """
        return self._station.envelope_variables()

    @command
    def emergency_standby(self, reason: str) -> None:
        """Stand the whole station down NOW — permitted in every state.

        The unconditional safe-off path: an operator (or an agent) can always
        reach it, including from EMERGENCY and ERROR, which is exactly where
        every other manual route is refused and where making the machine safe
        matters most. It is deliberately NOT governed by
        ``_manual_action_admissible()``: an admission gate on the stop button
        is a gate on the wrong thing.

        Routes into the same emergency flow a tripped critical flag takes
        (aborting the active run, emitting ``run_finished``, entering
        EMERGENCY, and standing every VI down), so there is one shutdown
        implementation, not two. Re-entrant: calling it while already in
        EMERGENCY re-asserts the standby rather than refusing.

        Runs on the CALLER's stack, following ``stop_ramp()``/
        ``abort_procedure()``'s precedent — a safety action is never queued
        for the next tick. **Accepted latency:** the tick is synchronous and
        single-threaded, so a call arriving while a tick is mid-flight lands
        only after the current ``measure()`` returns. For a slow instrument
        that is seconds, bounded by one reading. This is a property of the
        cooperative single-thread design (no second thread, no blocking call
        in the tick path), not an oversight: the alternative — interrupting a
        bus transaction — is how GPIB races and half-written setpoints
        happen.

        Args:
            reason: Why the station is being stood down. Carried into the
                emergency's error message, its ``ErrorEvent`` and the single
                CRITICAL entry record ``_enter_emergency()`` writes (which
                also names the actor), so the record says who asked and why.
        """
        self._enter_emergency(f"emergency standby requested: {reason}")

    @command
    def run_procedure(self, procedure: Any) -> None:
        """Start a procedure immediately if IDLE or during a manual ramp; else queue it.

        Any exception during setup (initiate(), target dispatch) is contained:
        the partially-started run is cleaned up (data file closed, ramps
        stopped) and the Orchestrator degrades to ERROR instead of crashing
        the application.
        """
        manual_ramping = (
            self._state == OrchestratorState.RAMPING and self._procedure is None
        )
        if self._state != OrchestratorState.IDLE and not manual_ramping:
            self.queue_procedure(procedure)
            return

        # Cancel any manual ramp before starting the procedure (hardware
        # holds; the procedure's own targets take over immediately below).
        if manual_ramping:
            self._station.stop_ramps()
            logger.info("Manual ramp cancelled — procedure starting.")

        # A run without the per-tick stall detector and stale detection would
        # be blind to a quench or a dead controller, so monitoring is mandatory
        # while a procedure executes.
        if not self._monitoring:
            logger.info("run_procedure: monitoring was off — starting it (required during a run)")
            self.start_monitoring()

        self._start_run(procedure)


    def _start_run(self, procedure: Any) -> None:
        """Shared setup path for starting a run.

        Dispatches ``procedure.initiate()``'s plan (scope-checked via the
        procedure's own ``command_scope``, defaulting to "measurement"),
        enters INITIATING, and emits the run-started manifest. Any exception
        is contained to ERROR — the caller must have already confirmed
        permission to start (queueing, monitoring) before calling this.

        Args:
            procedure: The procedure to start.
        """
        self._procedure = procedure
        # The **run owner** (GLOSSARY.md's *Run owner*): whoever started this
        # run owns it. A run pulled off a queue is owned by whoever QUEUED
        # it — the decision to run it was theirs, and the actor draining the
        # queue may be the tick itself — so _next_run_owner, set when the run
        # left the queue, wins over the actor in flight.
        self._run_owner = self._next_run_owner or self._current_actor()
        self._next_run_owner = None
        # Claims + admission gate: captured here, duck-typed (never
        # importing BaseProcedure — contract C5) so a test
        # double without claimed_vi_names() behaves exactly like the
        # claim-everything default.
        claimed_vi_names = getattr(procedure, "claimed_vi_names", None)
        self._active_claims = claimed_vi_names() if callable(claimed_vi_names) else None
        self._standby_dispatched = False
        self._wait_started = False
        self._first_measurement = True
        self._pending_gates = []
        self._pause_requested = False
        try:
            plan = procedure.initiate()
            # The frozen-dataclass repr is the permanent record of exactly what
            # was requested — logged once, at INFO, on receipt.
            logger.info("Procedure plan (initiate): %r", plan)

            # Track active system VIs for stale monitoring, and as the run's
            # ramp scope (see _run_ramp_scope) — each step's targets are
            # accumulated into it as they are dispatched.
            self._active_system_vis = set(plan.targets.keys())

            # claim_commands (each claimed VI's own initiate()) go out FIRST,
            # before this plan's targets/commands — see PhasePlan.claim_commands
            # and BaseProcedure._claim_initiate_commands().
            allowed_scope = getattr(procedure, "command_scope", "measurement")
            self._station.send_measurement_commands(
                plan.claim_commands, allowed_scope=allowed_scope
            )
            self._dispatch_targets(plan.targets)
            self._station.send_measurement_commands(plan.commands, allowed_scope=allowed_scope)
            self._current_wait_time = plan.wait_s

            self._change_state(OrchestratorState.INITIATING)
            self._emit_run_started()
            self._emit_initiation_status(plan.targets, plan.commands)
        except Exception as exc:
            logger.exception("Procedure setup failed")
            self._fail_to_error(f"Could not start procedure: {exc}")

    def _current_gates(self) -> tuple:
        """Return the gates for the current RAMPING->MEASURING transition.

        ``initiation_gates()`` for the run's first transition,
        ``reading_gates()`` for every one after. Looked up defensively
        (``getattr`` with an empty-tuple default) so a duck-typed procedure
        test double without these methods behaves exactly like the no-op
        ``BaseProcedure`` default.
        """
        if self._procedure is None:
            return ()
        method_name = "initiation_gates" if self._first_measurement else "reading_gates"
        method = getattr(self._procedure, method_name, None)
        if method is None:
            return ()
        return tuple(method())

    def _dispatch_targets(self, targets: dict[str, Target]) -> None:
        """Forward targets to the Station, remembering them for resume.

        Raises:
            I2ASSafetyError: If a target for a bounded VI falls outside the
                active session envelope. Nothing is dispatched — the whole
                plan is rejected before any hardware is touched, and the tick
                boundary (or ``run_procedure``'s setup guard) contains the run
                to ERROR with the reason.
        """
        if self._session_envelope is not None:
            for vi_name, target in targets.items():
                message = self._session_envelope.check_target(vi_name, target.target)
                if message is not None:
                    raise I2ASSafetyError(message)
        self._last_system_targets = dict(targets)
        self._station.process_system_targets(targets)

    def _run_ramp_scope(self) -> set[str] | None:
        """Return the VIs whose ramps the active run owns (the ramp scope).

        The ramp-scope standard: a run waits for, holds, and stops the ramps
        IT started — never hardware someone else is moving. The owned set is
        ``_active_system_vis``, accumulated from every plan's targets, so it
        is precisely the VIs this run has ever sent a setpoint to.

        For every sweep procedure the distinction is invisible: a run can
        only start from IDLE, and a manual ramp leaves IDLE on the tick it
        starts, so nothing outside the scope can be ramping anyway. It
        matters for a procedure that commands no targets and narrows its
        ``claimed_vi_names()`` (see ``TimeSeries``): the operator keeps
        driving the cryostat from the front panel while the run records, and
        those manual ramps must neither stall the run's measurements nor be
        stopped when it ends.

        Returns:
            The run's owned VI names — possibly EMPTY, meaning the run owns
            no ramps at all (waits for nothing, stops nothing). ``None``
            when no run is active, which asks the whole-station question:
            the Orchestrator is then the only owner there is.
        """
        if self._procedure is None:
            return None
        return self._active_system_vis

    @command
    def queue_procedure(self, procedure: Any) -> None:
        """Queue a built procedure to run once the Orchestrator returns to IDLE.

        Args:
            procedure: The ready procedure instance to queue.
        """
        self._procedure_queue.append(procedure)
        self._queued_actors[id(procedure)] = self._current_actor()
        self._emit_queue_changed()


    def _take_queued(self, queue: list[Any]) -> Any:
        """Pop the front of *queue*, forgetting who queued it.

        Args:
            queue: ``_procedure_queue``.

        Returns:
            The run that was waiting at the front.
        """
        run = queue.pop(0)
        # Forgotten from the queue, remembered as the run's owner: the actor
        # that committed the station to this run is the one that owns it once
        # it starts (the run-ownership standard).
        self._next_run_owner = self._queued_actors.pop(id(run), None)
        return run

    def _pull_next_run(self) -> Any:
        """Ask the injected pull seam for the next run, or ``None``.

        Guarded: the queue lives outside the engine, so a client whose
        ``next_procedure()`` raises (a stale spec, a procedure that refuses
        the run, a catalog miss) must leave the engine IDLE and reportable,
        never degrade it to ERROR.

        Returns:
            A built procedure, or ``None`` when nothing is waiting or the
            callback failed.
        """
        if self.next_procedure is None:
            return None
        try:
            run = self.next_procedure()
        except Exception as exc:  # noqa: BLE001 — a client queue never kills the engine
            logger.exception("next_procedure() failed; queue not advanced")
            self._action_blocked(f"Could not start the next queued run: {exc}")
            return None
        # The **run owner** a queued run inherits, read off the built run
        # duck-typed (contract C5 keeps this module from importing the queue
        # that built it): whoever queued the spec owns the run it becomes.
        queued_by = getattr(run, "queued_by", None)
        self._next_run_owner = queued_by if isinstance(queued_by, ev.Actor) else None
        return run

    @command
    def run_queue(self) -> None:
        """Run the next queued procedure, if IDLE.

        **The engine pulls.** Runs handed over directly
        (``queue_procedure()``) drain first, then the injected
        ``next_procedure()`` seam is asked for one; a queue held as data
        outside the engine (``session/run_queue.py``) supplies it. Either way
        the engine, not the client, decides *when* a run starts — a client
        that watched ``state_changed`` for the IDLE state and started the
        next run itself would advance re-entrantly inside that emit and,
        seeing only a state name, could not tell a clean finish from a hold
        acknowledge.

        That last point is why this is called from exactly two places inside
        the engine, and never from the transition itself. Six transitions
        reach IDLE; only two chain:

        * ``abort_procedure()`` — chains. The operator ended THIS run, not
          the queue.
        * ``_finish_run()`` when the run ends — chains. The run completed,
          so the next one is due.
        * ``recover_from_error()`` — no chain. After an error the queue's
          assumptions may no longer hold; the operator restarts explicitly.
        * ``_acknowledge_emergency()`` — no chain. Acknowledging a critical
          flag is not a request to carry on measuring.
        * ``_fail_run_for_fault()`` — no chain. A run that failed for an
          instrument fault or a safety hold must not silently auto-continue.
        * ``_tick_body()``'s RAMPING branch when a MANUAL ramp completes — no
          chain. Nobody asked for a run; the operator was driving by hand.

        Refused outside IDLE with a reason; an empty queue is NOT a refusal —
        the queue simply ran to completion — so it returns quietly and the
        command is accepted.
        """
        if self._state != OrchestratorState.IDLE:
            message = (
                "Cannot start the next queued run: the Orchestrator is in "
                f"state {self._state.name}, starting requires IDLE"
            )
            logger.info("Blocked run_queue: %s", message)
            self._action_blocked(message)
            return
        if self._procedure_queue:
            procedure = self._take_queued(self._procedure_queue)
            self._emit_queue_changed()
            self.run_procedure(procedure)
            return
        run = self._pull_next_run()
        if run is None:
            self._next_run_owner = None
            return
        self._emit_queue_changed()
        self.run_procedure(run)

    def publish_queue(self, *, actor: ev.Actor = ev.OPERATOR) -> None:
        """Broadcast the run queue after a client changed it.

        The queue lives outside the engine, so the engine cannot see an add,
        a removal or a reorder happen; whoever made the change calls this and
        the resulting ``QueueChanged`` goes out on the one event stream,
        exactly like a change the engine made itself. Nothing else about the
        engine is touched — this starts nothing and refuses nothing, which is
        why it is a broadcast rather than a command.

        Args:
            actor: Who made the change, defaulting to the operator sentinel.
        """
        self._emit_queue_changed(actor)

    def _queue_entry(self, run: Any, kind: str) -> dict[str, Any]:
        """Render one directly-handed-over run as a queue entry.

        Shaped like ``session/run_queue.py``'s ``RunSpec.to_json()`` so one
        ``QueueChanged`` describes both halves of the queue in the same
        vocabulary. A run handed over as a live object carries no spec, so
        its ``spec_id`` is empty and the fields only a spec knows are blank.

        Args:
            run: The queued procedure.
            kind: ``"procedure"``.

        Returns:
            A JSON-safe dict.
        """
        actor = self._queued_actors.get(id(run), ev.OPERATOR)
        return {
            "spec_id": "",
            "kind": kind,
            "run_class": type(run).__name__,
            "params": {},
            "sample_info": {},
            "data_directory": "",
            "file_prefix": "",
            "actor": actor.to_json(),
            "queued_at": 0.0,
        }

    def _queue_entries(self) -> tuple[dict[str, Any], ...]:
        """Return every waiting run, in the order they will start.

        The runs handed over directly first, then whatever the injected
        ``queue_snapshot()`` reports.

        Returns:
            JSON-safe dicts, one per waiting run.
        """
        entries = [self._queue_entry(run, "procedure") for run in self._procedure_queue]
        if self.queue_snapshot is not None:
            try:
                entries += [dict(entry) for entry in self.queue_snapshot()]
            except Exception:  # noqa: BLE001 — reporting must never disrupt a run
                logger.exception("queue_snapshot() failed (non-fatal)")
        return tuple(entries)

    def _emit_queue_changed(self, actor: ev.Actor | None = None) -> None:
        """Broadcast the queue as it now stands.

        Args:
            actor: Who caused the change. ``None`` (the default) attributes it
                to the actor of the command in flight, which is right for
                every mutation the engine makes itself.
        """
        self._emit_event(
            ev.QueueChanged(
                entries=self._queue_entries(),
                actor=actor if actor is not None else self._current_actor(),
                request_id=self._pending.request_id if self._pending else "",
                seq=self._next_seq(),
            )
        )

    @property
    def pause_pending(self) -> bool:
        """True while a pause is waiting for the current datapoint to finish.

        The read half of the deferred-pause surface (GLOSSARY.md's "Pause
        boundary"): the GUI and tests ask this instead of reading the private
        flag. Only ever True for a pause requested in ``MEASURING`` — every
        other state pauses on the spot. Goes False again the moment the run
        enters PAUSED, or when the request is cancelled by
        ``resume_procedure()``/``abort_procedure()``.

        Both edges are broadcast: raising and withdrawing the request each
        emit a ``StatusSnapshot`` on the spot, so a client's status mirror
        can show ``{state: MEASURING, pause_pending: True}`` — the state the
        GUI renders as "Pausing" — for the whole interval between the click
        and the pause boundary, rather than learning of it a tick later when
        the state has already moved on.
        """
        return self._pause_requested

    @command
    def pause_procedure(self) -> None:
        """Pause the run, holding the hardware — but never mid-datapoint.

        Pausing stops the physical ramps, not just the schedule: a magnet PSU
        ramps autonomously to its last setpoint, so without a hardware hold
        "pause" would only stop the software while the field kept moving. The
        wait clock is frozen with it and restored on resume, so a point that
        was half-way through its settle still gets the settle it was declared
        to need. That hold is immediate from every state — mid-ramp, mid-wait,
        mid-gate — because holding the cryostat where it stands is the whole
        point of the control.

        ``MEASURING`` is the single exception, and the reason the **pause
        boundary** exists: pausing there would strand a point that is ramped,
        settled and gated but not yet read, and resuming would then take that
        reading after an arbitrarily long hold. So a pause requested in
        ``MEASURING`` is *deferred* — the flag is raised, ``measure()`` runs
        and saves its datapoint on the next tick, and the ``SWEEPING`` branch
        of ``_tick_body()`` honours the request the moment the point is
        complete. Resume then starts at the ramp to the next point.

        (The reading loop itself is never at risk: ``measure()`` runs
        synchronously inside one tick, so a click can only ever arrive
        between ticks, never between two readings of a datapoint.)
        """
        if self._procedure is None:
            self._action_blocked("Cannot pause: no run is active")
            return
        if self._state == OrchestratorState.MEASURING:
            # Mid-datapoint: defer to the pause boundary rather than strand a
            # settled, unread point.
            if self._pause_requested:
                return
            self._pause_requested = True
            self._emit_status("Pause requested - pausing after this point")
            # The request is a client-visible fact the moment it is accepted,
            # and no state change carries it: without this the mirror would
            # only learn of it on the next tick, by which time the state has
            # already left MEASURING and "pausing" has nothing left to say.
            self._emit_status_snapshot()
            return
        if self._state in (OrchestratorState.INITIATING, OrchestratorState.RAMPING,
                           OrchestratorState.INITIATION_GATE, OrchestratorState.READING_GATE,
                           OrchestratorState.SWEEPING, OrchestratorState.STANDBY):
            self._enter_paused()
            return
        self._action_blocked(f"Cannot pause while {self._state.value}")

    def _enter_paused(self, cause: str = "") -> None:
        """Enter PAUSED from the current state, holding all of the run's hardware.

        The single PAUSED-entry path, shared by ``pause_procedure()``'s
        immediate cases and by the pause boundary in ``_tick_body()``. Records
        the state to return to, freezes the wait clock, and stops the run's
        ramps (see ``pause_procedure()`` for why the hold is a hardware one,
        and GLOSSARY.md's **Ramp scope** for why only this run's).
        Clears any pending request — it has now been honoured.

        Args:
            cause: The ``StateChange`` cause, for a pause the tick honoured
                at the pause boundary rather than one a command asked for.
        """
        self._pause_requested = False
        self._pre_pause_state = self._state
        if self._wait_started:
            self._paused_wait_elapsed = time.time() - self._wait_start_time
        self._station.stop_ramps(self._run_ramp_scope())
        self._change_state(OrchestratorState.PAUSED, cause=cause)
        self._emit_status("Paused - hardware held")

    @command
    def resume_procedure(self) -> None:
        """Resume from PAUSED, or cancel a pause that has not landed yet.

        From PAUSED: restarts the held ramps and unfreezes the wait clock, and
        the run picks up where it was held — a pause taken mid-ramp finishes
        that ramp and measures its point; one taken at the pause boundary
        returns to SWEEPING, whose next act is the ramp to the next point.

        Called while a pause is merely *requested* (the run is still finishing
        the datapoint it was reading), this withdraws the request and the run
        carries on uninterrupted.
        """
        if self._state != OrchestratorState.PAUSED:
            if self._pause_requested:
                self._pause_requested = False
                self._emit_status("Pause request cancelled")
                # Symmetric with pause_procedure()'s deferred branch: the
                # withdrawal is a client-visible fact with no state change of
                # its own, so the "Pausing" indicator must be taken down by a
                # snapshot rather than left standing until the next tick.
                self._emit_status_snapshot()
                return
            self._action_blocked(f"Cannot resume while {self._state.value}")
            return
        # _enter_paused() held the hardware, which forgot its ramp — states
        # that were mid-ramp need their targets re-dispatched to continue.
        # SWEEPING (the pause boundary) is deliberately not among them: its
        # own change_sweep_step() dispatches the next point's targets.
        if self._pre_pause_state in (
            OrchestratorState.INITIATING,
            OrchestratorState.RAMPING,
            OrchestratorState.STANDBY,
        ) and self._last_system_targets:
            self._dispatch_targets(self._last_system_targets)
        if self._wait_started:
            self._wait_start_time = time.time() - self._paused_wait_elapsed
        self._change_state(self._pre_pause_state)
        self._emit_status("Resumed")

    @command
    def abort_procedure(self) -> None:
        """Abort the run: hold instruments where they are (no ramp-to-zero).

        Closes the data file (partial data preserved), sends the procedure's
        measurement safe-off commands, stops all active ramps with a hardware
        hold, and returns to IDLE. Ignored during EMERGENCY — the emergency
        flow owns cleanup there and is exited via acknowledge().

        Owner-scoped for agents (the run-ownership standard, see
        ``_run_owner_admission()``): an agent that did not start this run is
        refused unless it takes the run over deliberately, with a reason. The
        operator is never gated.
        """
        if self._state == OrchestratorState.EMERGENCY:
            logger.info("abort_procedure ignored during EMERGENCY")
            self._action_blocked(
                "Cannot abort during EMERGENCY — acknowledge the emergency first"
            )
            return
        self._abort_active_procedure()
        self._emit_run_finished("aborted")
        self._change_state(OrchestratorState.IDLE)
        self._emit_status("Aborted by user")
        self.run_queue()

    def active_ramps(self) -> list[RampRecord]:
        """Return the ramps running as of the last tick, ordered by VI name.

        The read half of the ramp-tracker surface (``ramps_updated`` is the
        push half — same payload, same objects). Returns the CACHED records
        built during the tick, so calling this from GUI code costs nothing
        and, crucially, touches no instrument: the tick is the only thing
        that polls hardware.

        Returns:
            One ``RampRecord`` per system VI whose ``ramp_status()`` was
            ``"RAMPING"`` at the last tick, ordered by VI name. Empty when
            nothing is ramping.
        """
        return list(self._active_ramps)

    @command
    def stop_ramp(self, vi_name: str) -> None:
        """Stop one VI's ramp, holding that instrument where it is.

        The per-instrument counterpart of ``abort_procedure()``: it stops a
        single ramp instead of tearing down the whole run, so an operator can
        cancel a manual field or temperature ramp started from the Monitor
        window without stopping anything else. Hardware is held where it is
        — there is no ramp-to-zero, exactly like an abort.

        Admission goes through ``_manual_action_admissible()``, the same
        predicate that governs every other manual action: a faulted or
        safety-held VI is refused, ERROR and (unless unlocked) EMERGENCY are
        refused, and — the case that matters here — a VI claimed by an active
        run is refused naming that run. Stopping one VI's ramp mid-run would
        strand the run waiting on a setpoint it can never reach, so aborting
        the run is the only correct way to stop a run's ramp.

        Unlike ``submit_vi_action()`` this is not queued for the next tick:
        a stop is a safety action and follows ``abort_procedure()``'s
        precedent of holding the hardware immediately. The state machine
        still owns the RAMPING → IDLE transition, which happens on the next
        tick once every remaining ramp has settled.

        Args:
            vi_name: The system VI whose ramp to stop.
        """
        admitted, reason, code = self._manual_action_admission(vi_name)
        if not admitted:
            logger.info("Blocked stop_ramp on %s: %s", vi_name, reason)
            self._action_blocked(reason, code)
            return
        try:
            self._station.stop_ramps({vi_name})
        except Exception as exc:  # noqa: BLE001 — every action gets a verdict
            logger.exception("stop_ramp failed on VI '%s'", vi_name)
            self._action_failed(vi_name, "stop_ramp", str(exc), error=exc)
            return
        logger.info("Ramp on '%s' stopped by user — hardware held", vi_name)
        self._action_succeeded(vi_name, "stop_ramp")
        self._emit_status(f"Ramp on {vi_name} stopped by user")
        # Drop the row from the tracker immediately rather than leaving a
        # stopped ramp on screen until the next tick rebuilds the snapshot.
        self._active_ramps = [r for r in self._active_ramps if r.vi_name != vi_name]
        self.ramps_updated.emit(list(self._active_ramps))





    @command
    def recover_from_error(self) -> None:
        """Return to IDLE after the user has reviewed an ERROR condition.

        The failed procedure was already cleaned up on ERROR entry. Queued
        procedures are NOT auto-started — after an error the queue's
        assumptions may no longer hold; the user restarts explicitly.
        """
        if self._state != OrchestratorState.ERROR:
            self._action_blocked(
                f"Nothing to recover from: not in ERROR (currently {self._state.value})"
            )
            return
        self._change_state(OrchestratorState.IDLE)

    @command
    def acknowledge(self) -> None:
        """Single GUI entry point: acknowledge EMERGENCY, or unlock held VIs.

        Grants manual control for ``self._manual_override_timeout_s`` seconds
        from this call, then automatically re-locks — never a standing
        bypass (GLOSSARY.md's **Hold acknowledge**). While in EMERGENCY,
        also runs ``_acknowledge_emergency()``'s existing two-stage
        state-recovery logic (return to IDLE once the triggering condition
        has cleared); the state itself is untouched otherwise — a hold-only
        acknowledge never fakes IDLE while a hold is still genuinely active,
        it only grants temporary manual access on top of whatever state is
        honestly still true. Re-acknowledging while already unlocked simply
        extends the window from this call; it does not stack, and (for the
        hold case) survives the condition flapping within the window — see
        ``_tick_body()``'s pruning step.

        Refused (with a reason) when nothing is held and the station isn't
        in EMERGENCY: there is nothing to acknowledge, and whoever pressed the
        button is owed that answer rather than silence.
        """
        now = time.time()
        if self._state == OrchestratorState.EMERGENCY:
            self._emergency_override_until = now + self._manual_override_timeout_s
            self._acknowledge_emergency()
            # The override window is a snapshot field and a hold-only
            # acknowledge changes no state, so refresh the mirror here or the
            # countdown label reads blank until the next tick.
            self._emit_status_snapshot()
            return
        held = self._held_vis()
        if not held:
            message = (
                "Nothing to acknowledge: no instrument is held and the "
                f"station is not in EMERGENCY (currently {self._state.value})"
            )
            logger.info("Blocked acknowledge: %s", message)
            self._action_blocked(message)
            return
        until = now + self._manual_override_timeout_s
        for condition in held.values():
            self._hold_override_until[condition.key] = until
        self._emit_status(
            "Holds acknowledged — manual control unlocked for "
            f"{self._manual_override_timeout_s:.0f}s: {', '.join(sorted(held))}"
        )
        self._emit_status_snapshot()  # see the EMERGENCY branch above

    def _acknowledge_emergency(self) -> None:
        """Acknowledge an EMERGENCY: unlock manual control, or return to IDLE.

        Implementation detail of ``acknowledge()`` — not called directly
        except by tests exercising EMERGENCY-only two-stage semantics in
        isolation.

        If the condition that triggered EMERGENCY (a critical safety flag —
        ``Station.active_critical_conditions()``, the System-Condition
        standard's own live registry, see ``core/conditions.py`` — or a
        session-envelope violation) is still active, acknowledging cannot
        return to IDLE (the next tick would bounce straight back), but it
        does unlock manual control of held VIs (via
        ``_manual_action_admissible()``'s override bypass) for front-panel
        recovery — e.g. cycling a switch heater by hand — while remaining in
        EMERGENCY. Starting a procedure stays refused throughout: those
        gates check the state itself, which is unchanged here.

        A merely hold-only flag never blocks this
        return: it was never why EMERGENCY was entered, and it keeps
        governing its concerned VIs identically in IDLE (the System-
        Condition standard applies in every state, not only EMERGENCY).

        Once the condition has cleared, acknowledging again (the same
        button, via ``acknowledge()``) returns to IDLE and relocks the
        override for the next emergency.
        """
        if self._state != OrchestratorState.EMERGENCY:
            return
        active = [c.kind for c in self._station.active_critical_conditions()]
        # A still-violated session envelope blocks the return to IDLE for the
        # same reason a critical safety flag does — the next tick would
        # bounce straight back.
        if self._session_envelope is not None:
            active.extend(self._session_envelope.check_state(self._station.cached_state))
        if active:
            self._emit_status(
                "Emergency acknowledged — front-panel manual control "
                f"unlocked. Condition still active: {', '.join(active)}"
            )
            return
        self._emergency_override_until = None
        self._change_state(OrchestratorState.IDLE)

    def held_vi_names(self) -> frozenset[str]:
        """Every VI currently under a hold-severity safety condition."""
        return frozenset(self._held_vis())

    def override_active(self, vi_name: str | None = None) -> bool:
        """True if a manual override currently admits action on *vi_name*.

        With ``vi_name=None``, True if the EMERGENCY override is currently
        unlocked. Tests and GUI code should assert/poll through this (or
        ``manual_override_expires_at()``) rather than reading
        ``_emergency_override_until``/``_hold_override_until`` directly.
        """
        now = time.time()
        if vi_name is None:
            return (
                self._emergency_override_until is not None
                and now < self._emergency_override_until
            )
        held = self._held_vis().get(vi_name)
        if held is None:
            return False
        return now < self._hold_override_until.get(held.key, 0.0)

    def manual_override_expires_at(self) -> float | None:
        """Soonest-expiring active override (EMERGENCY or hold), or ``None``."""
        candidates = list(self._hold_override_until.values())
        if self._emergency_override_until is not None:
            candidates.append(self._emergency_override_until)
        return min(candidates) if candidates else None

    def active_run_kind(self) -> str | None:
        """Return the active run's kind, or ``None`` if no run is active.

        The public, duck-type-free accessor GUI code uses to tell a run in
        flight from an idle station without reaching into ``self._procedure``
        or importing ``BaseProcedure`` (contracts C5/C8).

        Returns:
            ``"procedure"`` while a run is active, ``None`` otherwise.
        """
        return None if self._procedure is None else "procedure"


    def _active_run_label(self) -> str:
        """Return a human-readable ``"<kind> '<name>'"`` label for the active run.

        Used only to compose admission-refusal messages — never
        called with no active run.

        Returns:
            E.g. ``"procedure 'Field Sweep'"``.
        """
        procedure = self._procedure
        name = getattr(procedure, "name", "") or type(procedure).__name__
        return f"procedure {name!r}"

    def _held_vis(self) -> dict[str, Condition]:
        """Return ``{vi_name: Condition}`` for every VI currently held.

        Reads the Station's unified condition registry (the
        System-Condition standard, ``core/conditions.py``) through
        ``decide()`` — the same pure policy the tick pipeline uses to
        collapse the same information — rather than keeping a second,
        parallel bookkeeping structure. ``watched_vis``/``run_active`` are
        irrelevant to ``held_vis`` (only ``run_failure`` depends on them),
        so dummy values are passed.

        Returns:
            The alphabetically-first (by ``Condition.key``) hold-severity
            condition affecting each held VI, comm- or safety-origin alike.
        """
        return decide(
            self._station.conditions().values(), watched_vis=frozenset(), run_active=False
        ).held_vis

    def _manual_action_admissible(self, vi_name: str) -> tuple[bool, str]:
        """Return ``_manual_action_admission()`` without its verdict code.

        The two-value shape every caller that only renders a reason wants —
        ``stop_ramp()`` and the ramp tracker's stop policy, which must agree
        with the admission gate exactly (a row's Abort button can never look
        enabled for a stop the action would refuse).

        Args:
            vi_name: The VI the action targets.

        Returns:
            ``(True, "")`` when admitted; ``(False, reason)`` when refused.
        """
        admitted, reason, _code = self._manual_action_admission(vi_name)
        return admitted, reason

    def _manual_action_admission(
        self, vi_name: str, *, verb: str = "control"
    ) -> tuple[bool, str, ev.VerdictCode]:
        """Decide whether a manual action on *vi_name* may be admitted right now.

        The single admission predicate (the "Claims + admission gate"),
        shared verbatim by ``submit_vi_action()`` (what may be *queued*), the
        ``_tick_body()`` GUI-action drain gate (what may be *drained*) — they
        must agree, or a queued action could sit forever without a verdict —
        and ``disconnect_instrument()``, whose question is the same one:
        whoever owns this instrument right now decides who may touch it.

        Admission rules, in order:

        0. A VI carrying the Availability standard's ``not_responding`` tag
           (``i2as.core.availability`` — a comm-origin condition, an
           instrument fault, GLOSSARY.md's **Instrument fault**) is ALWAYS
           refused, regardless of state — including IDLE, and NEVER bypassed
           by the EMERGENCY manual override — until it recovers or
           ``retry_fault()`` succeeds. ``TAG_POLICY["not_responding"]
           .controllable`` is ``False`` and carries no override column, which
           is what makes the refusal unconditional; a safety hold (rule 0b)
           is a separate mechanism with its own override. Checked first, and
           here (not as a parallel check) so every caller
           (``submit_vi_action()``, the drain gate) inherits it for free.
        0b. A VI with an active safety-origin hold condition (the
            System-Condition standard: its ``safety_concerns()`` names a
            currently-tripped, non-tolerated flag — see ``Station.
            update_conditions()``) is refused next, ALSO regardless of
            state, UNLESS ``acknowledge()`` has unlocked it — either via the
            per-condition hold override (``_hold_override_until``) or the
            EMERGENCY override (``_emergency_override_until``, which admits
            every held VI once unlocked, not just the one that triggered
            EMERGENCY). Either override, once unlocked, ADMITS IMMEDIATELY
            here — this is what lets it also reach through claim-protection
            (rule 4) for a VI a paused procedure still claims, deliberately
            (GLOSSARY.md's **Hold acknowledge**; the known gap this creates
            for ``resume_procedure()`` is documented there). The hold
            override never admits while the station is in EMERGENCY or
            ERROR — those require their own (EMERGENCY) or no (ERROR)
            override; see the code comment below for why. A VI with no hold
            is entirely unaffected by this rule, whatever flags are tripped
            elsewhere.
        1. IDLE, or a manual ramp (RAMPING with no active run): always
           admitted.
        2. EMERGENCY: refused for EVERY VI, held or not, UNLESS the
           EMERGENCY override is unlocked — critical severity is
           station-wide scope by construction (the System-Condition
           standard): there is no "unconcerned VI" to admit once the whole
           station is in EMERGENCY. This is the inversion of the
           pre-System-Condition behavior, where a VI unconcerned with the
           tripped flag stayed operable.
        3. ERROR: always refused, naming the state.
        4. Otherwise a run is active. Admitted iff the active run's
           ``claimed_vi_names()`` is not "claim everything" (``None``) AND
           *vi_name* is not in it. A claimed VI, or a run that claims
           everything (every procedure today), is refused naming the owning
           run.

        Args:
            vi_name: The VI the action targets.
            verb: The verb every refusal reason is phrased with ("Cannot
                ``verb`` ``vi_name``: …"). Defaults to ``"control"``, which
                is what a front-panel action is doing; ``disconnect_
                instrument()`` passes its own so one predicate can answer for
                both without a client reading "cannot control" over a refused
                Disconnect. Only the wording changes — never the decision.

        Returns:
            ``(True, "", VerdictCode.OK)`` when admitted; otherwise
            ``(False, reason, code)`` with a human-readable reason naming why
            (and, for a claim refusal, the owning run) and the verdict code
            that rule refuses with: ``BLOCKED_FAULT`` for rules 0/0b (the
            instrument itself is not controllable), ``BLOCKED_CLAIM`` for
            rule 4, ``BLOCKED_STATE`` for the state rules. The code is
            produced HERE, next to the rule that decides, so a client's
            machine-readable answer can never drift from the prose one.
        """
        now = time.time()
        emergency_unlocked = (
            self._emergency_override_until is not None and now < self._emergency_override_until
        )
        admitted = (True, "", ev.VerdictCode.OK)
        held = self._held_vis().get(vi_name)
        if held is not None:
            not_responding = "not_responding" in self._station.availability(vi_name).tags
            if not_responding:
                if not TAG_POLICY["not_responding"].controllable:
                    return False, (
                        f"Cannot {verb} {vi_name}: instrument fault ({held.kind}) — "
                        f"{held.message}. Retry the instrument or wait for it to recover."
                    ), ev.VerdictCode.BLOCKED_FAULT
            else:
                hold_unlocked = now < self._hold_override_until.get(held.key, 0.0)
                if emergency_unlocked:
                    return admitted
                if hold_unlocked and self._state not in (
                    OrchestratorState.EMERGENCY, OrchestratorState.ERROR
                ):
                    return admitted
                return False, (
                    f"Cannot {verb} {vi_name}: safety hold active "
                    f"({held.kind}). Resolve the condition, or acknowledge to "
                    "unlock manual control."
                ), ev.VerdictCode.BLOCKED_FAULT
        if self._state == OrchestratorState.IDLE:
            return admitted
        manual_ramping = (
            self._state == OrchestratorState.RAMPING and self._procedure is None
        )
        if manual_ramping:
            return admitted
        if self._state == OrchestratorState.EMERGENCY:
            if emergency_unlocked:
                return admitted
            return False, (
                f"Cannot {verb} {vi_name}: EMERGENCY — acknowledge the "
                "emergency to unlock manual front-panel recovery."
            ), ev.VerdictCode.BLOCKED_STATE
        if self._state == OrchestratorState.ERROR:
            return False, (
                f"Cannot {verb} {vi_name}: procedure is running in state {self._state.name}"
            ), ev.VerdictCode.BLOCKED_STATE
        if self._procedure is None:
            # Defensive: no other non-IDLE, non-manual-ramp state should be
            # reachable with no active run. Refuse conservatively rather than
            # admit on an assumption that turned out false.
            return False, (
                f"Cannot {verb} {vi_name}: procedure is running in state {self._state.name}"
            ), ev.VerdictCode.BLOCKED_STATE
        if self._active_claims is None:
            return False, (
                f"Cannot {verb} {vi_name}: {self._active_run_label()} is running"
            ), ev.VerdictCode.BLOCKED_CLAIM
        if vi_name in self._active_claims:
            return False, (
                f"Cannot {verb} {vi_name}: claimed by running {self._active_run_label()}"
            ), ev.VerdictCode.BLOCKED_CLAIM
        return admitted

    def _envelope_refusal(
        self, vi_name: str, method_name: str, kwargs: Mapping[str, Any]
    ) -> str | None:
        """Return why the active envelope refuses this manual action, or ``None``.

        The direct action path's half of the session-envelope contract (see
        ``ExperimentEnvelope``): the envelope binds every writer, so a
        setpoint that would leave it is refused whether it arrives as a plan's
        ``Target`` (checked in ``_dispatch_targets()``) or as a manual action.
        The setpoint-parameter convention
        (``core.plan.SETPOINT_PARAM_PREFIX``) is what makes this generic — the
        Station reports which keyword argument of this capability carries the
        enveloped quantity, so no per-VI table lives here.

        Args:
            vi_name: The VI the action targets.
            method_name: The capability being called.
            kwargs: The action's keyword arguments.

        Returns:
            A refusal reason naming the violated envelope bound, or ``None``
            when no envelope is active, the VI is unbounded, the capability
            has no setpoint parameter, or the value is inside the bound.
        """
        if self._session_envelope is None:
            return None
        for param_name in self._station.setpoint_parameters(vi_name, method_name):
            if param_name not in kwargs:
                continue
            message = self._session_envelope.check_target(
                vi_name, kwargs[param_name]
            )
            if message is not None:
                return (
                    f"Cannot control {vi_name}: {method_name}("
                    f"{param_name}={kwargs[param_name]!r}) violates the "
                    f"{message}"
                )
        return None

    def _manual_action_verdict(
        self, vi_name: str, method_name: str, kwargs: Mapping[str, Any]
    ) -> tuple[bool, str, ev.VerdictCode]:
        """Return the full admission verdict for one manual action.

        ``_manual_action_admission()`` decides whether this VI may be
        controlled at all; this adds the value-level check the envelope owns.
        Shared verbatim by ``submit_vi_action()`` (what may be *queued*) and
        the ``_tick_body()`` drain gate (what may be *dispatched*), the same
        re-validation-at-drain discipline every other admission rule follows —
        an experiment (and therefore an envelope) can be opened or closed
        between the click and the tick.

        Args:
            vi_name: The VI the action targets.
            method_name: The capability being called.
            kwargs: The action's keyword arguments.

        Returns:
            ``(admitted, reason, code)``; *reason* is empty and *code* is
            ``OK`` when admitted, and ``BLOCKED_ENVELOPE`` when the value
            leaves the active experiment envelope.
        """
        admitted, reason, code = self._manual_action_admission(vi_name)
        if not admitted:
            return False, reason, code
        envelope_reason = self._envelope_refusal(vi_name, method_name, kwargs)
        if envelope_reason is not None:
            return False, envelope_reason, ev.VerdictCode.BLOCKED_ENVELOPE
        return True, "", ev.VerdictCode.OK

    @command
    def submit_vi_action(self, vi_name: str, method_name: str, **kwargs: Any) -> None:
        """Submit a GUI action to a specific VI.

        Admission is decided by ``_manual_action_verdict()``: a faulted
        or safety-held VI is always refused (the latter unless the
        EMERGENCY manual override is unlocked); IDLE / a manual ramp /
        EMERGENCY (for an unheld VI) always admit; ERROR always refuses;
        otherwise a run is active and the action is admitted iff *vi_name*
        is not one of the active run's claimed VIs. An admitted action is
        then checked against the active session envelope, which refuses a
        setpoint outside its bounds.

        Admitted actions are QUEUED, not dispatched: the tick is the single
        hardware writer, and it re-runs this same verdict before dispatching
        (see ``_tick_body()``). The dispatch itself goes through
        ``Station.execute_vi_action()``, whose own three checks refuse a
        private name, a method that is not a declared capability, and — for
        callers narrower than ``MANUAL_ACTION_SCOPE`` — an out-of-scope
        capability.
        """
        admitted, reason, code = self._manual_action_verdict(vi_name, method_name, kwargs)
        if not admitted:
            logger.info("Blocked action: %s", reason)
            self._action_blocked(reason, code)
            return

        # The pending verdict travels with the queued action: this command is
        # answered when the drain runs it, not when this method returns.
        pending = self._pending
        if pending is not None:
            pending.deferred = True
        self._gui_action_queue.append({
            "vi_name": vi_name,
            "method_name": method_name,
            "kwargs": kwargs,
            "pending": pending,
        })

    @command
    def submit_global_action(self, action: str) -> None:
        """Fan a global lifecycle action out into one queued action per VI.

        ``"initiate_all"`` / ``"standby_all"`` enqueue an ``initiate`` /
        ``standby`` for every registered VI onto the same GUI-action queue the
        per-panel lifecycle toggles use. Each then runs on the tick (the single
        hardware writer) and emits ``action_succeeded`` / ``action_failed`` —
        the per-VI verdict that flips each InstrumentPanel's lifecycle toggle.

        Calling ``station.initiate_all()`` / ``standby_all()`` directly here
        (the previous behaviour) ran the methods but emitted no verdict, so the
        toggles never updated and the click looked like it did nothing.

        Args:
            action: ``"initiate_all"`` or ``"standby_all"``. Anything else is
                refused with a reason naming the two that exist — an unknown
                action used to be silently ignored, which left the caller
                unable to tell a typo from a station that did nothing.
        """
        method = {"initiate_all": "initiate", "standby_all": "standby"}.get(action)
        if method is None:
            message = (
                f"Unknown global action {action!r}: expected 'initiate_all' "
                "or 'standby_all'"
            )
            logger.info("Blocked submit_global_action: %s", message)
            self._action_blocked(message, ev.VerdictCode.FAILED)
            return
        # Standby is also a safety action: if a run is in flight, abort it first
        # so the enqueued standby actions run once the Orchestrator is back in IDLE.
        if action == "standby_all" and self._state not in (
            OrchestratorState.IDLE,
            OrchestratorState.ERROR,
            OrchestratorState.EMERGENCY,
        ):
            self.abort_procedure()
        for vi_name in self._station.get_vi_names():
            self._gui_action_queue.append(
                {"vi_name": vi_name, "method_name": method, "kwargs": {}}
            )

    @command
    def connect_instrument(self, vi_name: str) -> None:
        """Bring an offline instrument online (the GUI's Connect action).

        The ``connect`` half of the connection-lifecycle standard (see
        ``BaseVirtualInstrument``) at the Orchestrator's public API — the one
        entry point for both cases that leave a VI offline: it never
        connected at startup, or the operator disconnected it deliberately.
        Delegates to ``Station.connect_instrument()`` and reports the verdict
        through the standard action signals: ``action_succeeded(vi_name,
        "connect")`` plus ``instrument_reconnected(vi_name)`` on success,
        ``action_failed`` with the reason otherwise.

        Allowed only in IDLE: a VI joining the station mid-procedure would
        bypass the run's safety review. Runs synchronously rather than via the
        GUI action queue — the queue dispatches to *registered* VIs, which an
        offline one is not, and everything is on the one thread anyway, so no
        tick can interleave with the connect (the single-writer guarantee
        holds).

        Args:
            vi_name: Name of the offline VI to connect.
        """
        if self._state != OrchestratorState.IDLE:
            msg = (
                f"Cannot connect {vi_name}: Orchestrator is in state "
                f"{self._state.name}, connecting requires IDLE"
            )
            logger.info("Blocked connect: %s", msg)
            self._action_blocked(msg)
            return
        ok, message = self._station.connect_instrument(vi_name)
        # Either outcome moves the station's offline registry — a success
        # takes the instrument out of it, a failure re-tags it — so both the
        # declaration and the live snapshot are republished FIRST, before any
        # per-VI notification: a client rebuilds that instrument's card when
        # it hears the notification and must not rebuild it from the previous
        # picture. Neither is a state change, so nothing else would refresh
        # them until the next tick.
        self._emit_station_info()
        self._emit_status_snapshot()
        if ok:
            logger.info("Connect succeeded for '%s'", vi_name)
            self.instrument_reconnected.emit(vi_name)
            self._action_succeeded(vi_name, "connect", result=message)
        else:
            logger.warning("Connect failed for '%s': %s", vi_name, message)
            self._action_failed(vi_name, "connect", message)

    @command
    def disconnect_instrument(self, vi_name: str) -> None:
        """Release a live instrument to its front panel (the GUI's Disconnect action).

        The ``disconnect`` half of the connection-lifecycle standard (see
        ``BaseVirtualInstrument``): delegates to
        ``Station.disconnect_instrument()``, which hands the instrument back
        WITHOUT standing it down — a magnet at field stays at field — and
        degrades the VI into the offline registry, exactly as if it had never
        connected. Reports through ``action_succeeded(vi_name, "disconnect")``
        plus ``instrument_disconnected(vi_name)``, or ``action_failed``.

        **Gated by the claim, not by the state.** Admission runs through
        ``_manual_action_admission()``, the single admission predicate every
        other manual route uses (the "Claims + admission gate"), rather than
        a blanket "IDLE only" of its own. Releasing an instrument is a manual
        action on that instrument, so it is refused for exactly the VIs whose
        front-panel controls are refused, and for the same reasons: a VI the
        active run CLAIMS is refused ``BLOCKED_CLAIM`` naming the run — that
        run would lose an instrument mid-procedure, escaping its safety
        review — while a VI it does not claim may be released mid-run, which
        is what the narrowed-claim procedures exist for (a ``TimeSeries``
        recording leaves the cryostat under manual control, and the operator
        may hand a magnet to its own front panel while it runs). Faults,
        safety holds, ERROR and EMERGENCY are answered by the same predicate,
        so this route can never disagree with the rest of the GUI about who
        owns an instrument. ``connect_instrument()`` stays IDLE-only by
        contrast: a VI the station does not hold has no claim to consult, and
        one JOINING mid-run has genuinely bypassed that run's safety review.

        The run's **ramp scope** (``_run_ramp_scope()``) is refused with it:
        a run that is physically moving a VI owns it for as long as that ramp
        is in flight, whether or not its ``claimed_vi_names()`` named it.

        A disconnected instrument is not a fault, so no ``ErrorEvent`` is
        raised: the operator asked for this.

        Args:
            vi_name: Name of the live VI to disconnect.
        """
        admitted, reason, code = self._manual_action_admission(
            vi_name, verb="disconnect"
        )
        if admitted and vi_name in (self._run_ramp_scope() or ()):
            admitted = False
            reason = (
                f"Cannot disconnect {vi_name}: ramped by running "
                f"{self._active_run_label()}"
            )
            code = ev.VerdictCode.BLOCKED_CLAIM
        if not admitted:
            logger.info("Blocked disconnect: %s", reason)
            self._action_blocked(reason, code)
            return
        ok, message = self._station.disconnect_instrument(vi_name)
        # Re-declared, and the snapshot refreshed with it, before any per-VI
        # notification — for the reasons ``connect_instrument()`` gives.
        self._emit_station_info()
        self._emit_status_snapshot()
        if ok:
            logger.info("Instrument '%s' disconnected by the operator", vi_name)
            self.instrument_disconnected.emit(vi_name)
            self._action_succeeded(vi_name, "disconnect", result=message)
        else:
            logger.warning("Disconnect failed for '%s': %s", vi_name, message)
            self._action_failed(vi_name, "disconnect", message)

    @command
    def ping_instrument(self, vi_name: str) -> None:
        """Send one instrument's identity query and report whether it answered.

        The connection check of the connection-lifecycle standard, at the
        Orchestrator's public API. ``BaseVirtualInstrument.ping()`` is
        harmless by construction — an identity query changes nothing — but it
        is still bus traffic, so it belongs to the engine, the sole writer,
        rather than to the client that asks for it. A client (the instrument
        front panel's "Check connection" button) submits this command and
        reads the answer off the verdict and the action signals; it never
        touches the VI.

        Runs synchronously rather than through the GUI action queue, for the
        reason ``connect_instrument()`` gives: the queue dispatches declared
        capabilities, and ``ping()`` is not one — it is the lifecycle probe
        that exists precisely for an instrument whose capabilities may not be
        reachable.

        Reports ``action_succeeded(vi_name, "ping")`` with ``result=True``
        when every driver answered, and ``action_failed(vi_name, "ping",
        reason)`` when one did not, when the query raised, or when no live VI
        of that name is registered.

        Args:
            vi_name: Name of the live VI to probe.
        """
        try:
            vi = self._station.get_vi(vi_name)
        except KeyError:
            message = f"Cannot check {vi_name}: no live instrument of that name"
            logger.info("Blocked ping_instrument: %s", message)
            self._action_failed(vi_name, "ping", message)
            return
        try:
            reachable = bool(vi.ping())
        except Exception as exc:  # noqa: BLE001 — any failure means "not reachable"
            logger.warning("Connection check for '%s' raised: %s", vi_name, exc)
            self._action_failed(vi_name, "ping", str(exc) or repr(exc), error=exc)
            return
        if reachable:
            logger.info("Connection check for '%s' succeeded", vi_name)
            self._action_succeeded(vi_name, "ping", result=True)
        else:
            logger.info("Connection check for '%s' found it unreachable", vi_name)
            self._action_failed(
                vi_name, "ping", f"{vi_name} did not answer an identity query"
            )

    def offline_reason(self, vi_name: str) -> str:
        """Return the current failure reason for an offline VI, GUI-safe.

        Args:
            vi_name: Name of the VI to look up.

        Returns:
            The offline record's human-readable reason, or ``""`` when the VI
            is not offline (e.g. it has just been reconnected).
        """
        try:
            return self._station.get_offline_info(vi_name).reason
        except KeyError:
            return ""

    def vi_faults(self) -> dict[str, FaultRecord]:
        """Return the Station's current runtime fault registry, GUI-safe.

        Returns:
            ``{vi_name: FaultRecord}`` for every VI with an active
            stale/disconnected fault.
        """
        return self._station.vi_faults()

    def availability(self, vi_name: str) -> Availability:
        """Return the Availability standard's unified record for one VI, GUI-safe.

        Args:
            vi_name: Name of a configured VI — live or offline.

        Returns:
            The :class:`~i2as.core.availability.Availability` record.

        Raises:
            KeyError: If `vi_name` is not a configured VI at all.
        """
        return self._station.availability(vi_name)

    def availabilities(self) -> dict[str, Availability]:
        """Return the Availability standard's unified record for every VI, GUI-safe.

        Returns:
            ``{vi_name: Availability}`` for every configured VI, live and
            offline alike.
        """
        return self._station.availabilities()

    @command
    def acknowledge_fault(self, vi_name: str) -> None:
        """Acknowledge a VI's active runtime fault (calms the Monitor UI).

        A no-op (logged) if the VI has no active fault — acknowledging
        something already clear is harmless. Emits ``action_succeeded`` so
        the Monitor's per-panel Acknowledge button gets the same confirmed-
        state feedback as every other GUI action.

        Args:
            vi_name: Name of the faulted VI.
        """
        if self._station.acknowledge_fault(vi_name):
            logger.info("Fault on '%s' acknowledged", vi_name)
            self._action_succeeded(vi_name, "acknowledge_fault")
        else:
            message = f"Nothing to acknowledge: '{vi_name}' has no active fault"
            logger.info("Blocked acknowledge_fault: %s", message)
            self._action_blocked(message)

    def _vi_claimed_by_active_run(self, vi_name: str) -> bool:
        """Return whether an active run currently claims *vi_name*.

        Mirrors the claim rule ``_manual_action_admissible()`` applies (its
        rule 4): no run active -> unclaimed; a claim-everything run
        (``_active_claims is None``) -> every VI counted claimed; otherwise
        membership in ``_active_claims``. Used to gate ``retry_fault()``'s
        disconnected-kind driver rebuild, which must not touch a VI a run
        is depending on.

        Args:
            vi_name: The VI to check.

        Returns:
            True if a run is active and claims this VI.
        """
        if self._procedure is None:
            return False
        return self._active_claims is None or vi_name in self._active_claims

    @command
    def retry_fault(self, vi_name: str) -> None:
        """Retry a VI's active runtime fault: reset counters, poll once — or,
        past the disconnect threshold, rebuild its driver session.

        The runtime counterpart of ``connect_instrument()``. For a
        ``"stale"`` fault this never rebuilds a driver (the VI is already
        live) — only ``Station.retry_fault()``'s counter-reset-and-repoll —
        and is not restricted to IDLE: an unclaimed VI's transient fault
        does not require aborting whatever run is in progress to retry it,
        and everything still runs on the one tick-driven thread so there is
        no concurrency hazard in doing this synchronously mid-run.

        A ``"disconnected"`` fault is different: ``Station.retry_fault()``
        closes and reopens the VI's driver session(s) in that case (see its
        docstring), which must never happen to a VI an active run currently
        claims — that would refresh hardware state out from under a run
        without going through its safety review, the same hazard
        ``disconnect_instrument()``'s claim gate and ``connect_instrument()``'s
        IDLE-only restriction exist to prevent. So the rebuild is refused (not just
        deferred) while the VI is claimed; the operator can retry again once
        the run releases it (ends, fails, or is aborted).

        Args:
            vi_name: Name of the faulted VI to retry.
        """
        condition = self._station.conditions().get(f"comm:{vi_name}")
        if condition is not None and condition.kind == "disconnected":
            if self._vi_claimed_by_active_run(vi_name):
                message = (
                    f"Cannot retry '{vi_name}': claimed by running "
                    f"{self._active_run_label()} — wait for the run to end, "
                    "fail, or be aborted before rebuilding its connection"
                )
                logger.info("Retry (rebuild) blocked for '%s': %s", vi_name, message)
                self._action_blocked(message, ev.VerdictCode.BLOCKED_CLAIM)
                return

        ok, message = self._station.retry_fault(vi_name)
        if ok:
            logger.info("Retry succeeded for faulted VI '%s'", vi_name)
            self._action_succeeded(vi_name, "retry_fault", result=message)
        else:
            logger.warning("Retry failed for faulted VI '%s': %s", vi_name, message)
            self._action_failed(vi_name, "retry_fault", message)

    # ------------------------------------------------------------------
    # Run manifests (consumed by the session layer)
    # ------------------------------------------------------------------

    def _emit_run_started(self) -> None:
        """Capture the active run's manifest and emit ``run_started``.

        Called exactly once per run, after ``initiate()`` succeeded and its
        plan was dispatched. The data file path must be captured here: the
        procedure closes its ``DataManager`` (and forgets the path) in
        ``standby()``/``abort()``, before the run-finished emission.
        Best-effort on the optional fields — a minimal procedure (e.g. a test
        mock) without the public accessors still gets a manifest.
        """
        procedure = self._procedure
        name = getattr(procedure, "name", "") or type(procedure).__name__
        slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
        self._run_counter += 1
        params: dict[str, Any] = {}
        get_params = getattr(procedure, "get_params", None)
        if callable(get_params):
            try:
                params = get_params()
            except Exception:  # noqa: BLE001 — manifest must never abort a run
                logger.exception("run manifest: get_params() failed")
        self._active_run_manifest = {
            "run_id": f"{time.strftime('%Y%m%d_%H%M%S')}_{self._run_counter:03d}_{slug}",
            "procedure": name,
            "kind": getattr(procedure, "run_kind", "run"),
            "params": params,
            "data_file": str(getattr(procedure, "data_filepath", None) or ""),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._datapoint_index = 0
        self.run_started.emit(dict(self._active_run_manifest))
        self._emit_event(
            ev.RunStarted(
                run_id=self._active_run_manifest["run_id"],
                manifest=_json_safe(self._active_run_manifest),
                actor=self._current_actor(),
                request_id=self._pending.request_id if self._pending else "",
                seq=self._next_seq(),
            )
        )

    def _collect_run_summary(self, procedure: Any) -> dict[str, Any]:
        """Return ``procedure.run_summary()``'s result, or ``{}`` on any problem.

        Duck-typed: looked up via ``getattr`` so this module never imports
        ``BaseProcedure`` (contract C5) — a procedure or a test double without
        ``run_summary()`` simply yields ``{}``. Guarded by a
        broad try/except plus a return-type check, so a broken or
        misbehaving override can never prevent the run from finishing.

        Args:
            procedure: The procedure to query (may be ``None``).

        Returns:
            The dict ``run_summary()`` returned, or ``{}`` if the method is
            absent, raises, or does not return a dict.
        """
        run_summary_fn = getattr(procedure, "run_summary", None)
        if not callable(run_summary_fn):
            return {}
        try:
            summary = run_summary_fn()
        except Exception:
            logger.exception("run_summary() raised")
            return {}
        if not isinstance(summary, dict):
            logger.warning(
                "run_summary() returned %r (expected a dict); ignoring", type(summary)
            )
            return {}
        return summary

    def _emit_run_finished(self, status: str, reason: str = "") -> None:
        """Emit ``run_finished`` for the active run, exactly once.

        Idempotent: the captured manifest is cleared on emission, so the
        overlapping cleanup paths (user abort, error containment, emergency
        entry) cannot double-report a run. A no-op when no run ever started
        (e.g. ``initiate()`` itself failed, or a manual-ramp-only session).

        Args:
            status: Terminal status — ``done``, ``aborted``, or ``failed``.
            reason: Error text for ``failed``; empty otherwise.
        """
        if self._active_run_manifest is None:
            return
        manifest = dict(self._active_run_manifest)
        self._active_run_manifest = None
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = status
        manifest["reason"] = reason
        # run_summary() hand-off: self._procedure is still set on
        # the "done" path (_finish_run() clears it AFTER this call); the
        # abort/fail/emergency paths already cleared it via
        # _abort_active_procedure(), which cached the summary into
        # self._pending_run_summary first — see that method's docstring.
        if self._procedure is not None:
            summary = self._collect_run_summary(self._procedure)
        else:
            summary = self._pending_run_summary
        self._pending_run_summary = {}
        manifest["summary"] = summary
        # The run is over, so nobody owns it any more (the run-ownership
        # standard: a fact about the run in flight, never a lease that
        # outlives it).
        self._run_owner = None
        self.run_finished.emit(manifest)
        takeover = self._takeover
        self._emit_event(
            ev.RunFinished(
                run_id=str(manifest.get("run_id", "")),
                status=status,
                reason=reason,
                manifest=_json_safe(manifest),
                actor=self._current_actor(),
                overridden_owner=(
                    dict(takeover["owner"]) if takeover is not None else None
                ),
                seq=self._next_seq(),
            )
        )

    # ------------------------------------------------------------------
    # Private / internal
    # ------------------------------------------------------------------

    def _change_state(self, new_state: OrchestratorState, cause: str = "") -> None:
        """Move the state machine, announcing the move on both channels.

        Args:
            new_state: The state being entered.
            cause: Short machine-readable reason, for the ``StateChange``
                event. Defaults to the command in flight, or ``"tick"`` for a
                transition the engine made on its own; callers whose reason is
                neither (``"emergency"``, ``"fault"``, ``"error"``,
                ``"procedure_finished"``, ``"pause_boundary"``) pass it here.
        """
        previous = self._state
        logger.info("Orchestrator state: %s -> %s", previous.name, new_state.name)
        self._state = new_state
        self._state_entered_at = time.time()
        self.state_changed.emit(self._state.value)
        self._emit_event(
            ev.StateChange(
                state=self._state.value,
                previous=previous.value,
                cause=cause or self._acting_command or "tick",
                actor=self._current_actor(),
                request_id=self._pending.request_id if self._pending else "",
                seq=self._next_seq(),
            )
        )
        self._emit_status_snapshot()

    # ------------------------------------------------------------------
    # The control contract: actors, verdicts, events
    # (the verdict standard — see the class docstring)
    # ------------------------------------------------------------------

    @contextmanager
    def _acting_as(
        self,
        actor: ev.Actor,
        command_name: str,
        takeover: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Hold *actor* and *command_name* for the duration of one command.

        The ``command`` decorator wraps every public command method in this,
        so ``_change_state()`` and every verdict can name who asked and what
        for without an argument threaded through the state machine. An actor
        left at the ``OPERATOR`` sentinel never displaces one already acting,
        so a command that internally calls another stays attributed to
        whoever started it.

        Args:
            actor: Who is acting.
            command_name: The command method's own name.
            takeover: The **takeover** record when this command was admitted
                over another actor's run ownership, or ``None`` (every
                ordinary command). Held the same way and for the same reason
                the actor is: the verdict and the ``RunFinished`` this
                command causes both have to name it, and neither is built
                where the admission ran.

        Yields:
            ``None`` — used only for its side effect.
        """
        previous_actor = self._actor
        previous_command = self._acting_command
        previous_takeover = self._takeover
        if actor is not ev.OPERATOR or previous_actor is None:
            self._actor = actor
        self._acting_command = command_name
        self._takeover = takeover
        try:
            yield
        finally:
            self._actor = previous_actor
            self._acting_command = previous_command
            self._takeover = previous_takeover

    def _current_actor(self) -> ev.Actor:
        """Return the actor to attribute what is happening right now to.

        Returns:
            The actor of the command in flight, or ``SYSTEM_ACTOR`` when the
            engine is acting on its own (a tick, a tripped flag, a run
            ending).
        """
        return self._actor if self._actor is not None else SYSTEM_ACTOR

    def _next_seq(self) -> int:
        """Return the next sequence number, shared by events and verdicts.

        Returns:
            A monotonically increasing integer, starting at 1.
        """
        self._seq += 1
        return self._seq

    def _emit_event(self, event: Any) -> None:
        """Emit one contract event, guarded.

        The event stream is a reporting surface: a listener that raises, or an
        event that cannot be built, must never degrade a running procedure to
        ERROR through the tick's exception boundary.

        Args:
            event: The ``events.Event`` to broadcast.
        """
        try:
            self.event_emitted.emit(event)
        except Exception:  # noqa: BLE001 — reporting must never disrupt a run
            logger.exception("event emit failed (non-fatal)")

    def _emit_verdict(
        self,
        code: ev.VerdictCode,
        reason: str = "",
        detail: dict[str, Any] | None = None,
        result: Any = None,
        pending: _PendingCommand | None = None,
    ) -> None:
        """Answer the command in flight, exactly once.

        A no-op when no command is in flight (a direct call from the GUI or a
        test), or when this command has already been answered — which is what
        makes "exactly one verdict per request id" a property of the code
        rather than a convention.

        Args:
            code: The machine-readable outcome.
            reason: Human-readable explanation, for a banner.
            detail: Structured explanation of the code, JSON-safe.
            result: The underlying call's return value, JSON-safe.
            pending: The command to answer; defaults to the one in flight.
        """
        pending = pending if pending is not None else self._pending
        if pending is None or pending.resolved:
            return
        pending.resolved = True
        if pending.takeover is not None:
            # A command admitted over another actor's run ownership answers
            # with the **takeover** in its detail, whatever else the detail
            # says: an accepted takeover that looked like an ordinary OK
            # would be exactly the silent act the rule exists to prevent.
            detail = {**(detail or {}), "takeover": dict(pending.takeover)}
        try:
            verdict = ev.Verdict(
                request_id=pending.request_id,
                command=pending.command,
                code=code,
                actor=pending.actor,
                reason=reason,
                detail=detail,
                result=_json_safe(result) if result is not None else None,
                seq=self._next_seq(),
            )
        except Exception:  # noqa: BLE001 — fall back to the payload-free answer
            logger.exception("verdict payload rejected; answering without it")
            verdict = ev.Verdict(
                request_id=pending.request_id,
                command=pending.command,
                code=ev.VerdictCode.FAILED,
                actor=pending.actor,
                reason=reason or "the verdict payload could not be built",
                seq=self._next_seq(),
            )
        self._remember_accepted(pending, code)
        try:
            self.verdict_emitted.emit(verdict)
        except Exception:  # noqa: BLE001 — a signal failure must not disrupt a run
            logger.exception("verdict emit failed")

    def _remember_accepted(
        self, pending: _PendingCommand, code: ev.VerdictCode
    ) -> None:
        """Hold the last ACCEPTED command, for the operational-status record.

        Accepted, not merely received: ``OK`` and ``FAILED`` both mean the
        engine took the command and acted (or tried to), while every
        ``BLOCKED_*`` code means it was refused before anything happened. A
        refusal therefore leaves the previous value standing, because the
        pair this feeds answers "who last got the engine to do something"
        and a refused command did nothing.

        Args:
            pending: The command just answered.
            code: The verdict code it was answered with.
        """
        if code in (ev.VerdictCode.OK, ev.VerdictCode.FAILED):
            self._last_accepted = pending

    def _action_blocked(
        self,
        reason: str,
        code: ev.VerdictCode = ev.VerdictCode.BLOCKED_STATE,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Refuse the action in flight: the compat signal AND the verdict.

        Args:
            reason: Human-readable refusal, shown verbatim in the GUI.
            code: Which blocking code the refusal is (default
                ``BLOCKED_STATE``, the state machine's own refusal).
            detail: The structured half of the refusal, naming the ``rule``
                that decided it — as ``_action_failed()`` carries a
                control-limit's numbers — so a client decides from the dict
                and never by parsing the prose. ``None`` for a refusal whose
                code says everything there is to say.
        """
        self.action_blocked.emit(reason)
        self._emit_verdict(code, reason=reason, detail=detail)

    def _action_failed(
        self,
        vi_name: str,
        method_name: str,
        reason: str,
        error: BaseException | None = None,
        pending: _PendingCommand | None = None,
    ) -> None:
        """Report a failed action: the compat signal AND the verdict.

        Args:
            vi_name: The VI the action targeted.
            method_name: The capability that was called.
            reason: The failure text, written to be shown verbatim.
            error: The exception behind it, when there was one — it decides
                the verdict code and carries the structured ``detail`` of a
                control-limit refusal.
            pending: The command to answer; defaults to the one in flight.
        """
        self.action_failed.emit(vi_name, method_name, reason)
        code, detail = _verdict_for_exception(error)
        self._emit_verdict(code, reason=reason, detail=detail, pending=pending)

    def _action_succeeded(
        self,
        vi_name: str,
        method_name: str,
        result: Any = None,
        pending: _PendingCommand | None = None,
    ) -> None:
        """Report a successful action: the compat signal AND the verdict.

        Args:
            vi_name: The VI the action targeted.
            method_name: The capability that was called.
            result: The call's return value, carried in ``Verdict.result``.
            pending: The command to answer; defaults to the one in flight.
        """
        self.action_succeeded.emit(vi_name, method_name)
        self._emit_verdict(ev.VerdictCode.OK, result=result, pending=pending)

    def _emit_station_info(self) -> None:
        """Emit the Station's static declaration on the event stream.

        Called once at construction and after every successful connect or
        disconnect — the only two things that change what the station IS.
        The Station builds the snapshot (``Station.station_info()``, from
        declarations and config alone, never the bus); the engine re-stamps
        it with the event stream's own ``seq`` so a client can order it
        against every other event and verdict. Reporting must never disrupt
        a run, so a failure to build the snapshot is logged, not raised.
        """
        try:
            declared = self._station.station_info()
            event = replace(declared, seq=self._next_seq())
        except Exception:  # noqa: BLE001 — reporting must never disrupt a run
            logger.exception("station_info() failed (non-fatal)")
            return
        self._emit_event(event)

    def station_info(self) -> ev.StationInfo:
        """Return the station's declaration snapshot as last published.

        The **priming read** of the status-mirror standard (GLOSSARY.md's
        *Status mirror*): a client's mirror is fed by the event stream, which
        broadcasts a ``StationInfo`` at construction and after every connect
        and disconnect — all of them emitted before a client that attaches
        later exists. So whoever builds the engine takes this read once, on
        the engine's own thread, to prime the mirror it hands the client;
        no client ever calls it, and no client ever needs to.

        Returns:
            The Station's current declaration, re-stamped with the event
            stream's ``seq`` exactly as the emitted one is.
        """
        return dataclasses.replace(
            self._station.station_info(), seq=self._next_seq()
        )

    def status_snapshot(self) -> ev.StatusSnapshot:
        """Return this moment's status snapshot.

        The second **priming read** of the status-mirror standard, and the
        counterpart of ``station_info()`` above: the live half of the picture,
        taken once when a client's mirror is built so it starts on the state
        the engine is already in rather than on a default. Every later value
        arrives on the event stream.

        Returns:
            The snapshot ``StatusSnapshot`` the next tick would emit.
        """
        return self._status_snapshot()

    def _run_status(self) -> dict[str, Any] | None:
        """Return the active run's summary for a ``StatusSnapshot``.

        Returns:
            ``{"id", "name", "kind", "owner", "progress", "step", "steps"}``
            for the run in flight — every optional key omitted rather than
            guessed — or ``None`` when no run is active. ``owner`` is the
            **run owner**'s ``Actor.ref()``, so every client can say whose
            run this is and judge the run-ownership standard against the same
            fact the engine enforces (``None`` only for a run started before
            an owner could be taken, which nothing here does).
        """
        if self._procedure is None and self._active_run_manifest is None:
            return None
        manifest = self._active_run_manifest or {}
        summary: dict[str, Any] = {
            "id": str(manifest.get("run_id", "")),
            "name": str(manifest.get("procedure", "")),
            "kind": self.active_run_kind() or "",
            "owner": self._run_owner.ref() if self._run_owner is not None else None,
        }
        get_progress = getattr(self._procedure, "get_progress", None)
        if callable(get_progress):
            try:
                summary["progress"] = float(get_progress())
            except Exception:  # noqa: BLE001 — a summary must never raise
                logger.debug("run summary: get_progress() failed", exc_info=True)
        get_position = getattr(self._procedure, "get_sweep_position", None)
        if callable(get_position):
            try:
                step, steps = get_position()
                summary["step"] = int(step)
                summary["steps"] = int(steps)
            except Exception:  # noqa: BLE001 — a summary must never raise
                logger.debug("run summary: get_sweep_position() failed", exc_info=True)
        return summary

    def _status_snapshot(self) -> ev.StatusSnapshot:
        """Build this moment's ``StatusSnapshot``.

        Every field is read from a cached or derived source — nothing here
        polls an instrument, because the tick is the only thing that does.
        Every payload is a fresh copy.

        Returns:
            The snapshot, ready to emit.
        """
        availabilities = {
            name: _json_safe(dataclasses.asdict(record))
            for name, record in self._station.availabilities().items()
        }
        faults = {
            name: _json_safe(dataclasses.asdict(record))
            for name, record in self.vi_faults().items()
        }
        offline_reasons: dict[str, str] = {}
        for name in availabilities:
            reason = self.offline_reason(name)
            if reason:
                offline_reasons[name] = reason
        held = self.held_vi_names()
        # The lifecycle-state standard (GLOSSARY.md's **Lifecycle state**):
        # what each instrument is DOING, read from the Station as a fact
        # rather than left for a client to infer from the actions it saw —
        # the emergency's blanket standby_all() dispatches none. A pure
        # cached read per VI, so it costs the tick nothing.
        lifecycles = self._station.lifecycle_states()
        instruments = {
            name: {
                "availability": availabilities[name],
                "fault": faults.get(name),
                "offline_reason": offline_reasons.get(name, ""),
                "held": name in held,
                "override_active": self.override_active(name),
                "lifecycle": lifecycles.get(name, ev.LifecycleState.IDLE.value),
            }
            for name in availabilities
        }
        return ev.StatusSnapshot(
            state=self._state.value,
            run=self._run_status(),
            instruments=instruments,
            is_monitoring=self._monitoring,
            pause_pending=self._pause_requested,
            active_run_kind=self.active_run_kind(),
            override_active=self.override_active(),
            manual_override_expires_at=self.manual_override_expires_at(),
            held_vi_names=tuple(sorted(held)),
            active_ramps=tuple(
                _json_safe(dataclasses.asdict(record)) for record in self._active_ramps
            ),
            availabilities=availabilities,
            vi_faults=faults,
            offline_reason=offline_reasons,
            envelope_variables={
                name: _json_safe(dataclasses.asdict(variable))
                for name, variable in self.envelope_variables().items()
            },
            attended=self._attended,
            agent_gate=self._agent_gate.value,
            seq=self._next_seq(),
        )

    def _emit_status_snapshot(self) -> None:
        """Emit this moment's ``StatusSnapshot``, guarded.

        Guarded exactly like ``_publish_ramps()``: the status mirror is a
        reporting surface, so a failure assembling it must never degrade a
        running procedure to ERROR through the tick's exception boundary.
        """
        try:
            snapshot = self._status_snapshot()
        except Exception:  # noqa: BLE001 — reporting must never disrupt a run
            logger.exception("status-snapshot assembly failed (non-fatal)")
            return
        self._emit_event(snapshot)

    def get_operational_status(self) -> dict:
        """Return the most recent operational-status record.

        The runtime troubleshooting signal: orchestrator state, elapsed time in
        it, and per-system-VI gap-to-target / rate / ETA / verdict. See
        ``i2as.core.operational_status`` for the schema; the same record is
        emitted on ``operational_status`` and appended to ``logs/status.jsonl``.
        """
        return dict(self._operational_status)

    def _update_operational_status(self, state: dict) -> None:
        """Build this tick's operational-status record, emit it, and log it.

        Called on EVERY tick, monitoring on or off, so that a gap in
        ``status.jsonl`` means the process stopped ticking and nothing else.
        Nothing is polled while the machine is quiet (monitoring off AND
        IDLE) — the same guard `_publish_ramps()` uses, for the same reason:
        a freshly launched app polls no instrument until its instruments are
        initiated, and reporting must not be the one thing that breaks that
        silence. A quiet tick therefore carries an empty instrument payload
        (``vis``; ``conditions`` reports whatever the registry last held,
        since nothing re-evaluated it) while every header field of the record
        standard is written as usual — see `i2as.core.operational_status`.

        The record also names the last command the engine ACCEPTED — its
        ``Actor`` and its ``request_id`` (see `_remember_accepted()`) — so a
        reader of ``status.jsonl`` can say who last got the engine to do
        something, and can join this log to that client's own action trail
        on the request id.

        Guarded: operational-status reporting is non-critical, so a failure here
        must never degrade a running procedure to ERROR via the tick boundary.

        Args:
            state: This tick's station state snapshot, or ``{}`` on a tick
                that polled nothing.
        """
        try:
            quiet = not self._monitoring and self._state == OrchestratorState.IDLE
            ramp_info: dict[str, dict] = {} if quiet else self._ramp_info()
            wait_target = self._current_wait_time if self._wait_started else None
            wait_elapsed = (
                time.time() - self._wait_start_time if self._wait_started else None
            )
            progress = None
            if self._procedure is not None and hasattr(self._procedure, "get_progress"):
                try:
                    progress = self._procedure.get_progress()
                except Exception:
                    progress = None
            # This tick's System-Condition standard registry (core/conditions.py),
            # mirroring how the unified condition pipeline below (~1602-1607)
            # builds its list: the Station's comm/safety conditions plus, when a
            # session envelope is active and we are not already in EMERGENCY, its
            # envelope conditions for this same state snapshot. Sorted by key so
            # the record is stable across ticks with the same conditions.
            conditions: list[Condition] = list(self._station.conditions().values())
            if self._session_envelope is not None and self._state != OrchestratorState.EMERGENCY:
                conditions.extend(
                    envelope_conditions(self._session_envelope.check_state(state), time.time())
                )
            conditions.sort(key=lambda c: c.key)
            manifest = self._active_run_manifest or {}
            record, self._prev_gaps = build_operational_status(
                orch_state=self._state.value,
                elapsed_in_state_s=time.time() - self._state_entered_at,
                state=state,
                ramp_info=ramp_info,
                prev_gaps=self._prev_gaps,
                wait_target_s=wait_target,
                wait_elapsed_s=wait_elapsed,
                progress=progress,
                # Postcondition gates are no longer a multi-tick wait phase
                # (evaluated once, immediately, as the run ends), so only the
                # initiation/reading gates can ever be "active" across ticks.
                active_gates=[g.name for g in self._pending_gates],
                conditions=conditions,
                run_id=manifest.get("run_id"),
                setup=self._setup_name,
                actor=(
                    self._last_accepted.actor.to_json()
                    if self._last_accepted is not None
                    else None
                ),
                request_id=(
                    self._last_accepted.request_id
                    if self._last_accepted is not None
                    else None
                ),
            )
            record, self._stall_state = apply_stall_verdict(
                record,
                self._stall_state,
                self._stall_config,
                no_motion_phases=no_motion_phases_from(ramp_info),
            )
            self._operational_status = record
            self.operational_status.emit(dict(record))  # the signal payload rule
            self._status_logger.info(json.dumps(record))
        except Exception:
            logger.exception("operational-status update failed (non-fatal)")

    # ------------------------------------------------------------------
    # Ramp tracking
    # ------------------------------------------------------------------

    def _ramp_info(self) -> dict[str, dict]:
        """Return this tick's ``Station.get_ramp_status()`` snapshot, polled once.

        Memoised for the duration of one tick (``_tick_body()`` clears the
        memo on entry) because two consumers want the same answer — the
        operational-status record and the ramp tracker — and each VI's ramp
        accessors are real instrument reads. Polling twice per tick would
        double that bus traffic for no new information.
        """
        if self._tick_ramp_info is None:
            self._tick_ramp_info = self._station.get_ramp_status()
        return self._tick_ramp_info

    def _publish_ramps(self) -> None:
        """Rebuild this tick's running-ramp records, cache them, and emit them.

        Guarded exactly like ``_update_operational_status()``: ramp tracking
        is a reporting surface, so a failure here must never degrade a
        running procedure to ERROR via the tick's exception boundary.

        Nothing is polled while the machine is quiet (monitoring off AND
        IDLE): a freshly launched app polls no instrument until its
        instruments are initiated, and the tracker must not be the one thing
        that breaks that silence. Any ramp — manual or run-driven — leaves
        IDLE on the tick it starts, so nothing that is actually ramping is
        ever missed by this guard.
        """
        try:
            if not self._monitoring and self._state == OrchestratorState.IDLE:
                records: list[RampRecord] = []
            else:
                run_active = self._procedure is not None
                records = build_ramp_records(
                    self._ramp_info(),
                    setpoint_meta=self._station.system_setpoint_meta,
                    # The SAME predicate stop_ramp() itself uses, so a row's
                    # Abort button can never look enabled for a stop the
                    # action would refuse (or vice versa).
                    stop_policy=self._manual_action_admissible,
                    run_label=self._active_run_label() if run_active else None,
                    run_claims=self._active_claims if run_active else None,
                )
            self._active_ramps = records
            self.ramps_updated.emit(list(records))
        except Exception:
            logger.exception("ramp-tracker update failed (non-fatal)")

    # ------------------------------------------------------------------
    # Concise status feed (Procedure-window status log)
    # ------------------------------------------------------------------

    def _emit_status(self, text: str) -> None:
        """Emit one concise milestone line to listeners and the status logger.

        Wrapped so a formatting or signal error can never abort a run: the tick
        runs inside an exception boundary that degrades to ERROR, and a
        cosmetic status line must not be able to trip it.

        Goes to ``status_message``/``i2as.procedure_status``. Neither is
        the ``i2as.status`` logger, which carries the machine-only JSONL
        operational-status stream and must stay pure JSON.
        """
        try:
            logging.getLogger(f"{LOGGER_ROOT}.procedure_status").info(text)
            self.status_message.emit(text)
        except Exception:  # noqa: BLE001 — status must never disrupt the run
            logger.exception("status emit failed")

    def _describe_system_target(self, vi_name: str, target: Target, *, verb: str) -> str:
        """Compose "<verb> <label> to <value> <unit>" for one system ramp target.

        Label and unit come from the VI's declarative setpoint metadata via the
        Station (e.g. magnet -> "field"/"T"), so any procedure's targets render
        without per-procedure code. Best-effort: degrades to the raw VI name
        rather than raising into the tick.
        """
        try:
            label, unit = self._station.system_setpoint_meta(vi_name)
            value = float(target.target)
            unit_suffix = f" {unit}" if unit else ""
            return f"{verb} {label} to {value:g}{unit_suffix}"
        except Exception:  # noqa: BLE001 — degrade, never raise into the tick
            return f"{verb} {vi_name}"

    def _describe_measurement_command(self, command: Command) -> str | None:
        """Compose "Arming/Disarming <label> measurement" for one ``Command``.

        Returns None if the command cannot be described.
        """
        try:
            label = self._station.measurement_label(command.vi_name)
            if command.method in ("standby", "disarm"):
                return f"Disarming {label} measurement"
            return f"Arming {label} measurement"
        except Exception:  # noqa: BLE001
            return None

    def _emit_setup_actions(
        self,
        system_targets: dict[str, Target],
        commands: Sequence[Command],
        *,
        verb: str,
    ) -> None:
        """Emit one status line per distinct setup action (ramps, then measurement).

        Used for both initiation and standby/parking so each thing being done
        (set temperature, ramp field, arm the measurement) shows separately.
        """
        for vi_name, target in system_targets.items():
            self._emit_status(self._describe_system_target(vi_name, target, verb=verb))
        for command in commands:
            line = self._describe_measurement_command(command)
            if line:
                self._emit_status(line)

    def _emit_initiation_status(
        self, system_targets: dict[str, Target], commands: Sequence[Command]
    ) -> None:
        """Emit the initiation header plus one line per distinct setup action.

        Initiation is a distinct, often slow phase: the procedure brings the
        NON-swept system state to its setpoints (target temperature in a field
        sweep, target field in a temperature sweep) AND moves the swept quantity
        to its start value AND arms the measurement instrument. Each is shown as
        its own line — none of these setpoints are reached yet, and a magnet or
        temperature ramp can take a long time.
        """
        try:
            name = getattr(self._procedure, "name", "") or type(self._procedure).__name__
            _, n = self._procedure.get_sweep_position()
            self._emit_status(f'Initiating "{name}" ({n} points)')
        except Exception:  # noqa: BLE001
            self._emit_status("Initiating procedure")
        self._emit_setup_actions(system_targets, commands, verb="Ramping")

    def _ramp_status_line(self, system_targets: dict[str, Target]) -> str:
        """Compose "Point i/n: ramping <label> -> <value> <unit>" for a sweep step.

        Describes the (usually single) target the sweep step ramps, using the
        VI's setpoint metadata via the Station. Best-effort.
        """
        try:
            i, n = self._procedure.get_sweep_position()
            parts = []
            for vi_name, target in system_targets.items():
                label, unit = self._station.system_setpoint_meta(vi_name)
                unit_suffix = f" {unit}" if unit else ""
                parts.append(f"{label} -> {float(target.target):g}{unit_suffix}")
            detail = "; ".join(parts) if parts else "next setpoint"
            return f"Point {i}/{n}: ramping {detail}"
        except Exception:  # noqa: BLE001 — degrade, never raise into the tick
            return "Ramping to next setpoint"

    def _measure_status_line(self) -> str:
        """Compose "Measuring point i/n" for the current point (best-effort)."""
        try:
            i, n = self._procedure.get_sweep_position()
            return f"Measuring point {i}/{n}"
        except Exception:  # noqa: BLE001
            return "Measuring"

    def _tick(self) -> None:
        """One cooperative cycle, inside the exception boundary.

        PyQt6 aborts the process on an unhandled exception in a slot; here
        that would mean dying with the magnet mid-ramp and the data file
        open. Anything unexpected instead cleans up and degrades to ERROR.
        """
        try:
            self._tick_body()
        except Exception as exc:  # noqa: BLE001 — boundary must be broad
            logger.exception("Unhandled exception in orchestrator tick")
            if self._state == OrchestratorState.EMERGENCY:
                # Already in the most severe state; just report.
                self._error(f"Internal error during EMERGENCY: {exc}")
                return
            self._fail_to_error(f"Internal error: {exc}")

    def _drain_request_spool(self) -> None:
        """Carry out every request the **Request spool** queued, this tick.

        The engine's file-based front door, drained beside the manual-action
        queue and for the same reason: the tick is the single writer, so a
        request that arrived from outside the process is admitted at the one
        point where nothing else is touching the hardware. It runs on the
        tick's own stack — no thread, no socket, no second timer.

        Every request is validated again HERE, not where it was written: the
        state, the claims, **Attendance** and the **Kill switch** may all have
        moved since the file was dropped, so the answer a request gets is the
        answer it would get if it had been submitted this instant. A request
        the permission model refuses is answered with its ``BLOCKED_ROLE``
        verdict on this engine's own ``verdict_emitted`` stream — the same
        place an obeyed command's verdict appears, so the human's window sees
        an agent being refused exactly as it sees it being obeyed — and never
        reaches ``submit()``.
        """
        spool = self._request_spool
        if spool is None:
            return
        requests = spool.take()
        if not requests:
            return
        # Only now: building the declaration snapshot costs something, and an
        # idle spool is the overwhelmingly common case.
        station_info = self.station_info()
        for request in requests:
            refusal = spool.authorize(
                request, station_info, self._attended, self._agent_gate
            )
            if refusal is not None:
                logger.info(
                    "Blocked spooled request %s: %s",
                    request.request_id,
                    refusal.reason,
                )
                self.verdict_emitted.emit(replace(refusal, seq=self._next_seq()))
                continue
            if request.command is not None:
                self.submit(request.command)

    def _tick_body(self) -> None:
        # Fresh tick: whatever ramp snapshot the last one memoised is stale.
        # See _ramp_info() — one poll per tick, shared by every consumer.
        self._tick_ramp_info = None

        # 1.+2. Monitor cycle — only while monitoring is active (see
        # start_monitoring()). While it is off, no instrument is polled at
        # all: a freshly launched app stays quiet until the instruments have
        # been initiated. run_procedure() auto-starts monitoring and
        # stop_monitoring() is refused outside IDLE/ERROR, so the stall
        # detector and stale detection below are guaranteed to run whenever
        # a procedure is active.
        state: dict = {}
        if self._monitoring:
            state = self._station.get_state()
            # The signal payload rule: what leaves the engine is a copy the
            # engine never touches again. Station.get_state() returns a fresh
            # outer dict but shares the inner per-VI dicts with its stale-value
            # cache, and a client on another thread must not be handed a
            # container the engine still holds.
            self.states_updated.emit(
                {name: dict(values) for name, values in state.items()}
            )
            self._emit_event(
                ev.Readings(values=_json_safe(state), seq=self._next_seq())
            )

            # One-line summary per tick (full per-method detail stays in the file log)
            parts = []
            for vi_name, vi_state in state.items():
                readable = {k: v for k, v in vi_state.items() if not k.startswith("_")}
                if readable:
                    kv = ", ".join(
                        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in readable.items()
                    )
                    parts.append(f"{vi_name}: {kv}")
            logger.debug("Monitor: %s", " | ".join(parts))

        # Operational-status record (runtime troubleshooting signal): assembled
        # from this tick's snapshot (empty while monitoring is off, which polls
        # nothing), emitted, and appended to the resolved log directory's
        # status.jsonl (see i2as.core.paths.log_directory()). Written on
        # EVERY tick, outside the monitoring branch above, so that silence in
        # that log unambiguously means the process is not ticking — an agent
        # gating on `troubleshoot status --max-age` can then tell a live run
        # from a log left behind by a process that died.
        self._update_operational_status(state)

        if self._monitoring:
            # Tiered trend-history sample: non-fatal, mirrors the
            # operational-status guard above — a logging failure here must
            # never fail the tick. record() is itself internally
            # exception-safe; this guard is belt-and-braces around
            # constructing its arguments (e.g. last_state_flat()).
            try:
                self._tiered_trend_logger.record(
                    self._station.last_state_flat(), time.time(), self._state.name
                )
            except Exception:
                logger.exception("tiered trend-history record failed (non-fatal)")

            run_active = self._state not in (
                OrchestratorState.IDLE,
                OrchestratorState.PAUSED,
                OrchestratorState.ERROR,
                OrchestratorState.EMERGENCY,
            )
            # The run's watched VIs: its system (target-receiving) VIs plus
            # its EXPLICIT claims — a claimed non-system VI (a monitor a
            # run depends on) faulting must fail the run too, not merely
            # warn while the run keeps trusting a dead instrument. Claims of
            # None (claim-everything, every plain procedure) deliberately do
            # NOT widen this beyond the system VIs: a procedure's unrelated
            # VI going stale was never a run-failure before and stays a
            # warning-only fault.
            watched_vis = set(self._active_system_vis) | (self._active_claims or set())

            # Safety check — reuses this tick's snapshot (no second hardware
            # poll), called exactly once regardless of how many decisions
            # below consult it. The active run's tolerated_safety_flags are
            # resolved once here and applied uniformly by
            # update_conditions(); a plain procedure (or an idle station)
            # tolerates nothing.
            safety = self._station.check_safety(state)
            tolerated: frozenset[str] = frozenset()
            if self._procedure is not None:
                tolerated = frozenset(
                    getattr(self._procedure, "tolerated_safety_flags", frozenset())
                )

            # ONE unified condition pipeline (the System-Condition standard,
            # core/conditions.py): get_state() above already recorded this
            # tick's comm conditions; update_conditions() refreshes the
            # safety-origin ones from this tick's check_safety() result,
            # honoring tolerated_safety_flags exactly as before. A
            # session-envelope violation is folded into the same list — same
            # snapshot, same consequence as a tripped safety flag — gated
            # exactly like today: not even computed once already EMERGENCY.
            self._station.update_conditions(safety, tolerated_flags=tolerated)
            conditions: list[Condition] = list(self._station.conditions().values())
            if self._session_envelope is not None and self._state != OrchestratorState.EMERGENCY:
                conditions.extend(
                    envelope_conditions(self._session_envelope.check_state(state), time.time())
                )

            # Onset diff: fires a one-shot warning for a NEW comm-origin
            # condition on an unwatched VI (a watched VI's fault is handled
            # as a run failure below instead, with a matching, more severe
            # event, so it must not ALSO get a warning here). This diff is
            # comm-origin ONLY — a hold-severity safety condition is no
            # longer handled by ONSET here at all: it is a LEVEL-TRIGGERED
            # invariant enforced every tick by _enforce_safety_holds() below
            # (over verdict.held_vis, once decide() has run), which re-issues
            # standby() for as long as the hold persists rather than once at
            # onset — see that method's docstring for why an edge-triggered
            # dispatch here left a hold that survived an acknowledge-then-
            # expire cycle permanently unenforced.
            by_key = {c.key: c for c in conditions}
            current_keys = set(by_key)
            new_keys = current_keys - self._known_condition_keys
            self._known_condition_keys = current_keys
            for key in sorted(new_keys):
                condition = by_key[key]
                if condition.origin == "comm":
                    vi_name = condition.source_vis[0]
                    if run_active and vi_name in watched_vis:
                        continue  # handled as a run failure below, this same tick
                    self._emit_fault_event(vi_name, condition.kind, condition.message)

            # Single verdict over every condition (comm + safety + envelope):
            # decide() partitions by severity — critical -> emergency,
            # hold -> held_vis, and (when a run is active) the
            # alphabetically-first watched VI landing in held_vis ->
            # run_failure. This collapses what used to be four separate
            # checks (EMERGENCY entry, envelope violation, held-claimed run
            # failure, stale-claimed run failure) into one policy call.
            verdict = decide(conditions, watched_vis=watched_vis, run_active=run_active)

            # Prune expired hold-override entries — by expiry only, never
            # because the condition disappeared from verdict.held_vis (that
            # would force a fresh acknowledge on every flap of the same
            # flag, defeating the point of the cooldown; see acknowledge()
            # and GLOSSARY.md's **Hold acknowledge**). _emergency_override_
            # until needs no equivalent tick-driven pruning: it is read live
            # via a timestamp comparison in _manual_action_admissible(), and
            # _enter_emergency() already resets it unconditionally on every
            # fresh EMERGENCY entry.
            if self._hold_override_until:
                now = time.time()
                self._hold_override_until = {
                    key: until for key, until in self._hold_override_until.items()
                    if now < until
                }

            if verdict.emergency and self._state != OrchestratorState.EMERGENCY:
                safety_conditions = [c for c in verdict.emergency if c.origin == "safety"]
                envelope_conds = [c for c in verdict.emergency if c.origin == "envelope"]
                reason_parts = []
                vi_names: set[str] = set()
                if safety_conditions:
                    reason_parts.append(", ".join(sorted({c.kind for c in safety_conditions})))
                    vi_names |= {vi for c in safety_conditions for vi in c.source_vis}
                if envelope_conds:
                    reason_parts.append("; ".join(c.message for c in envelope_conds))
                self._enter_emergency("; ".join(reason_parts), tuple(sorted(vi_names)))
                return  # emergency entry already cleaned up; nothing else this tick

            # Safety-hold enforcement (the level-triggered invariant — see
            # _enforce_safety_holds()'s docstring): runs AFTER the override
            # pruning above (so override_active() reads this tick's fresh
            # expiries, not a stale one) and AFTER the emergency block (an
            # EMERGENCY already issued its own blanket standby_all() and
            # returned above; there is no per-VI concerned subset left to
            # enforce). Runs BEFORE the run-failure check below, so a hold
            # that also fails a watched run is still enforced on this same
            # tick — the run ending must not silently stop the invariant
            # that keeps its now-unclaimed VI safe.
            self._enforce_safety_holds(verdict)

            # Run failure: the watched VI's hold — comm-origin (stale) or
            # safety-origin (a non-tolerated concern) — that decide() found.
            # Either way the run fails and returns to IDLE, not global
            # ERROR: the blast radius is this one VI, every other instrument
            # (including this one once it recovers) stays usable.
            if verdict.run_failure is not None:
                vi_name, condition = verdict.run_failure
                if condition.origin == "safety":
                    self._fail_run_for_fault(
                        vi_name, reason=f"safety hold on '{vi_name}' ({condition.kind})"
                    )
                else:
                    self._fail_run_for_fault(vi_name)
                return

        # 3. Client requests, then GUI actions. The Request spool is drained
        # FIRST so that a spooled submit_vi_action — which submit() queues for
        # the single writer rather than dispatching — is executed by the drain
        # immediately below, on this same tick rather than the next one.
        self._drain_request_spool()

        # Each queued action gets the SAME verdict
        # submit_vi_action() would give it right now, via the shared
        # _manual_action_admissible() predicate: the run may have
        # started/finished/changed claims since it was queued, and a claim
        # refusal during an active run only refuses the CLAIMED VIs, not the
        # whole queue — so admission is decided per action, not once for the
        # batch. Every action gets a verdict this tick; none is left queued.
        pending_actions = list(self._gui_action_queue)
        self._gui_action_queue.clear()
        for action in pending_actions:
            # The verdict this action owes its submitter, if it arrived
            # through submit() — see submit_vi_action()'s asynchronous case.
            pending = action.get("pending")
            admitted, reason, code = self._manual_action_verdict(
                action["vi_name"], action["method_name"], action["kwargs"]
            )
            if not admitted:
                logger.info("Blocked queued action on %s: %s", action["vi_name"], reason)
                self.action_blocked.emit(reason)
                self._emit_verdict(code, reason=reason, pending=pending)
                continue
            try:
                result = self._station.execute_vi_action(
                    action["vi_name"],
                    action["method_name"],
                    allowed_scope=MANUAL_ACTION_SCOPE,
                    **action["kwargs"]
                )
                self._action_succeeded(
                    action["vi_name"],
                    action["method_name"],
                    result=result,
                    pending=pending,
                )
            except Exception as e:
                # Every user action gets an explicit verdict: rejections
                # (limit violations, safety guards) and failures surface
                # to the GUI with the reason, never silently.
                logger.error("Error executing GUI action on %s: %s", action["vi_name"], e)
                self._action_failed(
                    action["vi_name"],
                    action["method_name"],
                    str(e),
                    error=e,
                    pending=pending,
                )
        if pending_actions:
            # A drained action may have started, retargeted, or stopped a
            # ramp, which the snapshot taken at the top of this tick predates
            # — so the tracker re-polls rather than showing the operator a
            # tick-old view of the click they just made. Only on ticks that
            # actually ran an action: an ordinary tick still polls once.
            self._tick_ramp_info = None

        # If a GUI action started (or restarted) a manual ramp, enter RAMPING.
        if self._state == OrchestratorState.IDLE and not self._station.check_ramps():
            self._change_state(OrchestratorState.RAMPING)

        # 4. State Machine matching
        if self._state == OrchestratorState.IDLE:
            pass
        elif self._state == OrchestratorState.INITIATING:
            self._change_state(OrchestratorState.RAMPING)
        elif self._state == OrchestratorState.RAMPING:
            # Scoped to the run's OWN ramps (see _run_ramp_scope): a manual
            # front-panel ramp on a VI this run neither targets nor claims
            # must not hold its next measurement. Every ramp still advances
            # inside this call, in or out of scope.
            if self._station.check_ramps(self._run_ramp_scope()):  # True = this run's ramps complete
                if self._procedure is None:
                    # Manual ramp from GUI — return to IDLE.
                    self._change_state(OrchestratorState.IDLE)
                else:
                    gates = self._current_gates()
                    if gates:
                        # Gates replace wait_s entirely for this transition.
                        self._pending_gates = list(gates)
                        gate_state = (
                            OrchestratorState.INITIATION_GATE
                            if self._first_measurement
                            else OrchestratorState.READING_GATE
                        )
                        self._change_state(gate_state)
                    else:
                        if not self._wait_started:
                            self._wait_started = True
                            self._wait_start_time = time.time()
                            if self._current_wait_time > 0:
                                self._emit_status(
                                    f"Waiting {self._current_wait_time:g} s at setpoint"
                                )

                        if time.time() - self._wait_start_time >= self._current_wait_time:
                            self._wait_started = False
                            self._first_measurement = False
                            self._change_state(OrchestratorState.MEASURING)
        elif self._state in (OrchestratorState.INITIATION_GATE, OrchestratorState.READING_GATE):
            # A gate can be waiting ON a ramp that is outside the run's ramp
            # scope — a VI commanded into standby rather than targeted still
            # ramps down, and a gate on its readback holds until it gets
            # there. Nothing but this call moves a ramp generator, so a
            # tick spent here must still advance them or the gate can never
            # come true.
            self._station.advance_ramps()
            self._pending_gates = [g for g in self._pending_gates if not g.step()]
            if not self._pending_gates:
                self._first_measurement = False
                self._change_state(OrchestratorState.MEASURING)
        elif self._state == OrchestratorState.MEASURING:
            # Ramps outside the run's scope keep moving while it measures —
            # a manual front-panel ramp must not advance at a third of its
            # rate just because two ticks of every measurement cycle are
            # spent in states that historically never saw a live ramp.
            self._station.advance_ramps()
            if self._procedure:
                self._emit_status(self._measure_status_line())
                self._procedure.measure()
                if hasattr(self._procedure, "get_progress"):
                    self.procedure_progress.emit(self._procedure.get_progress())
                last_datapoint = getattr(self._procedure, "last_datapoint", None)
                if last_datapoint:
                    self.measurement_ready.emit(dict(last_datapoint))
                    manifest = self._active_run_manifest or {}
                    self._emit_event(
                        ev.Datapoint(
                            run_id=str(manifest.get("run_id", "")),
                            index=self._datapoint_index,
                            values=_json_safe(last_datapoint),
                            seq=self._next_seq(),
                        )
                    )
                    self._datapoint_index += 1
            self._change_state(OrchestratorState.SWEEPING)
        elif self._state == OrchestratorState.SWEEPING:
            self._station.advance_ramps()  # see MEASURING above
            if self._pause_requested:
                # The pause boundary: a pause requested during MEASURING waits
                # here, where the point just read is saved and the next one has
                # not been asked for yet. Resume re-enters SWEEPING and steps
                # from here.
                self._enter_paused(cause="pause_boundary")
            elif self._procedure:
                step_plan = self._procedure.change_sweep_step()
                if step_plan is None:
                    # done, go to standby
                    self._change_state(OrchestratorState.STANDBY)
                else:
                    logger.info("Procedure plan (step): %r", step_plan)
                    # A step may target a VI initiate() never did, and from
                    # this moment the run owns that ramp — accumulate, so the
                    # ramp scope (see _run_ramp_scope) covers every VI this
                    # run has ever commanded, not just its opening plan's.
                    self._active_system_vis |= set(step_plan.targets)
                    self._dispatch_targets(step_plan.targets)
                    self._current_wait_time = step_plan.wait_s
                    self._change_state(OrchestratorState.RAMPING)
                    self._emit_status(self._ramp_status_line(step_plan.targets))
        elif self._state == OrchestratorState.STANDBY:
            if not self._standby_dispatched:
                # Wait for whatever ramp THIS RUN had in flight when SWEEPING
                # ended, then call standby() exactly once and dispatch
                # whatever targets it returns (e.g. ramp magnet to 0 T). A
                # manual ramp outside the run's scope is not waited for —
                # otherwise stopping a run that commands nothing would hang
                # until the operator's own ramp happened to land.
                if self._station.check_ramps(self._run_ramp_scope()):
                    self._emit_status("Sweep complete - closing data file")
                    if self._procedure and hasattr(self._procedure, "standby"):
                        plan = self._procedure.standby()
                        logger.info("Procedure plan (standby): %r", plan)
                        self._dispatch_targets(plan.targets)
                        allowed_scope = getattr(
                            self._procedure, "command_scope", "measurement"
                        )
                        self._station.send_measurement_commands(
                            plan.commands, allowed_scope=allowed_scope
                        )
                        if plan.targets or plan.commands:
                            self._emit_status("Parking hardware")
                            self._emit_setup_actions(plan.targets, plan.commands, verb="Ramping")
                    self._standby_dispatched = True
            else:
                # Wait for the ramp standby() itself just started (if any),
                # then finish: a procedure finishes as soon as it settles.
                if self._station.check_ramps(self._run_ramp_scope()):
                    self._finish_run()
        elif self._state == OrchestratorState.PAUSED:
            pass # Monitor continues, no ramp advancement
        elif self._state == OrchestratorState.ERROR:
            pass # Awaiting user interaction (recover_from_error)
        elif self._state == OrchestratorState.EMERGENCY:
            # Shutdown already ran once on entry (_enter_emergency).
            # Monitoring continues; awaiting acknowledge().
            pass

        # 5. Ramp tracker — last, so the run/claim context the records carry
        # is the state machine's verdict for this tick (a run that just
        # started or just ended) rather than the one it was in on entry.
        # Reads this tick's memoised snapshot; no extra poll.
        self._publish_ramps()

        # 6. The status mirror: one snapshot per tick, so a client that never
        # calls into the engine still sees every read refreshed at tick rate.
        # A tick that changed state has already emitted one from
        # _change_state(); this is the quiet tick's.
        self._emit_status_snapshot()

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    def _abort_active_procedure(self) -> None:
        """Clean up the running procedure: data file, measurement VI, ramps.

        Safe to call with no procedure active (stops manual ramps then).
        Each cleanup step is individually guarded so one failure (e.g. a dead
        instrument) cannot prevent the others. Also clears ``_active_claims``
        — the shared teardown path for user abort, ``_fail_to_error``,
        and ``_enter_emergency``, so a claim can never outlive its run.

        Caches ``procedure.run_summary()`` into
        ``self._pending_run_summary`` BEFORE clearing ``self._procedure``
        below: the subsequent ``_emit_run_finished()`` call on every one of
        these teardown paths (abort/fail/emergency) runs with
        ``self._procedure`` already ``None``, so it could not call
        ``run_summary()`` itself.
        """
        procedure = self._procedure
        self._pending_run_summary = self._collect_run_summary(procedure)
        if procedure is not None and hasattr(procedure, "abort"):
            try:
                commands = procedure.abort()
                if commands:
                    allowed_scope = getattr(procedure, "command_scope", "measurement")
                    self._station.send_measurement_commands(
                        commands, allowed_scope=allowed_scope
                    )
            except Exception:
                logger.exception("Procedure abort cleanup failed")
        try:
            # Hold hardware where it is: clearing generators alone would let
            # autonomous hardware (magnet PSU) keep ramping to its last setpoint.
            self._station.stop_ramps(self._run_ramp_scope())
        except Exception:
            logger.exception("Stopping ramps during abort failed")

        self._procedure = None
        self._run_owner = None
        self._active_claims = None
        self._active_system_vis.clear()
        self._standby_dispatched = False
        self._wait_started = False
        self._first_measurement = True
        self._pending_gates = []
        self._pause_requested = False
        self._last_system_targets = {}




    def _finish_run(self) -> None:
        """Declare the active run done: emit finished signals and return to IDLE.

        Called once the STANDBY wait settles.
        """
        self._emit_status("Procedure finished")
        self._emit_run_finished("done")
        self.procedure_finished.emit()
        self._procedure = None
        self._active_claims = None
        self._active_system_vis.clear()
        self._standby_dispatched = False
        self._pause_requested = False
        self._change_state(OrchestratorState.IDLE, cause="procedure_finished")
        self.run_queue()

    def _fail_to_error(self, message: str) -> None:
        """Contain a failure: clean up the run and degrade to ERROR.

        Reserved for unknown-blast-radius failures: an unhandled
        exception at the tick boundary, or a run whose ``initiate()``/setup
        itself raised (the run never got far enough to know which VI, if
        any, is to blame). A stale CLAIMED VI mid-run has a KNOWN, narrow
        blast radius and uses ``_fail_run_for_fault()`` instead — it does
        not degrade to global ERROR.
        """
        self._error(message)
        # A command that failed this way (a run whose setup raised) is
        # answered FAILED rather than closed with the optimistic OK a silent
        # return would earn it.
        self._emit_verdict(ev.VerdictCode.FAILED, reason=message)
        try:
            self._abort_active_procedure()
        except Exception:
            logger.exception("Cleanup while entering ERROR also failed")
        self._emit_run_finished("failed", reason=message)
        self._change_state(OrchestratorState.ERROR, cause="error")

    def _fail_run_for_fault(self, vi_name: str, reason: str | None = None) -> None:
        """Fail the active run because its claimed/watched VI faulted or was held.

        Unlike ``_fail_to_error()``, this does NOT degrade to global ERROR:
        the blast radius is known (one VI, already recorded as a
        comm- or safety-origin ``Condition`` in the Station's unified
        registry — the System-Condition standard, ``core/conditions.py``),
        so only the run ends — every other instrument, including this one
        once it recovers, stays usable.
        Deliberately does NOT call ``run_queue()`` afterward: a run failing
        for an instrument fault or safety hold must not silently
        auto-continue to the next queued run, the same conservative
        behavior the old global-ERROR path had.

        Args:
            vi_name: The claimed/watched VI that went stale, or was held by
                a safety concern, during the run.
            reason: Human-readable description of why the run is failing.
                ``None`` (the default) assumes a stale VI — every other
                caller (the safety-hold path) passes its own message.
        """
        if reason is None:
            message = (
                f"Run failed: active VI '{vi_name}' became stale. The instrument "
                "is quarantined; every other instrument stays usable."
            )
        else:
            message = f"Run failed: {reason}"
        self._error(message, vi_name=vi_name, kind="run_failure", severity="error")
        self._emit_verdict(ev.VerdictCode.FAILED, reason=message)
        try:
            self._abort_active_procedure()
        except Exception:
            logger.exception("Cleanup while failing run for VI fault also failed")
        self._emit_run_finished("failed", reason=message)
        self._change_state(OrchestratorState.IDLE, cause="fault")

    def _enter_emergency(self, reason: str, vi_names: tuple[str, ...] = ()) -> None:
        """One-shot emergency entry: clean up the run, then safe shutdown.

        The shutdown runs exactly once here (not every tick): repeating it
        each tick would, for a VI whose shutdown is itself a multi-tick
        sequence, restart that sequence every few seconds.

        **The single EMERGENCY-entry record.** This is the one path into
        EMERGENCY — a tripped critical condition observed by the tick and an
        operator's or agent's ``emergency_standby()`` alike — so the one
        CRITICAL log line for the entry is written HERE, naming the cause
        (the tripped condition, or the caller's reason) and the actor it is
        attributed to. CLAUDE.md reserves CRITICAL for safety events, and an
        entry into EMERGENCY is one however it was observed; writing it here
        rather than at each caller is what keeps the two routes from
        diverging (they did: the manual call logged CRITICAL, the tick's
        critical-flag entry only ERROR). Callers pass their reason and add no log line of
        their own, so the file log carries exactly one record per entry.

        Args:
            reason: Human-readable description of the tripped condition(s)
                (e.g. flag names or an envelope-violation message).
            vi_names: The VI(s) that originated the condition (from each
                critical ``Condition.source_vis``), so the reason and its
                ``ErrorEvent`` name the instrument. Empty when no per-VI
                attribution is available (e.g. a session-envelope
                violation, which is checked against a live reading rather
                than a VI-tagged safety flag).
        """
        actor = self._current_actor()
        message = f"EMERGENCY: safety condition triggered ({reason})"
        if vi_names:
            message += f" — instrument(s): {', '.join(vi_names)}"
        message += f" [actor {actor.kind.value}:{actor.id}]"
        self._error(
            message,
            vi_name=", ".join(vi_names) if vi_names else None,
            kind="safety",
            severity="emergency",
            log_level=logging.CRITICAL,
        )
        # Defense in depth alongside _manual_action_admissible()'s own
        # EMERGENCY/ERROR guard on the hold override: a fresh EMERGENCY
        # starts with a clean slate for BOTH override kinds, so a hold
        # acknowledged moments earlier (a hold ack still live on
        # magnet_z) cannot be mistaken for an EMERGENCY acknowledge once a
        # quench trips this same tick.
        self._emergency_override_until = None
        self._hold_override_until = {}
        try:
            self._abort_active_procedure()
        except Exception:
            logger.exception("Cleanup while entering EMERGENCY failed")
        self._emit_run_finished("failed", reason=f"EMERGENCY: {reason}")
        self._change_state(OrchestratorState.EMERGENCY, cause="emergency")
        # Always a blanket standby_all(): critical severity is station-wide
        # scope by construction (the System-Condition standard, core/
        # conditions.py) — there is no "concerned VI" subset to narrow the
        # shutdown to, whether the cause was a safety flag (vi_names
        # attributed) or a session-envelope violation (vi_names empty).
        try:
            self._station.standby_all()
            self._error("Emergency shutdown executed.")
        except Exception:
            logger.exception("standby_all during emergency entry failed")
            self._error("Emergency shutdown could not complete — check instruments.")

    def _error(
        self,
        message: str,
        *,
        vi_name: str | None = None,
        kind: str = "internal",
        severity: str = "error",
        log_level: int = logging.ERROR,
    ) -> None:
        """Report an error: log it, emit the compat + structured signals.

        Every call here emits BOTH ``error_occurred`` (compat, unchanged
        shape) and the richer ``error_event`` — see the class
        docstring's ``error_event`` entry for why a plain per-VI fault
        (``kind="fault"``, severity ``"warning"``) does NOT go through this
        method (see ``_emit_fault_event()`` instead).

        Args:
            message: Human-readable description.
            vi_name: The originating VI, if any (``None`` for a machine-wide
                event, or a comma-joined list for more than one VI).
            kind: ``"internal"`` (default, unhandled tick-boundary
                exception), ``"run_failure"``, ``"safety"``, or
                ``"safety_hold"`` (a VI-scoped safety-hold-enforcement
                escalation — see ``_enforce_safety_holds()``).
            severity: ``"error"`` (default) or ``"emergency"``.
            log_level: The ``logging`` level *message* is logged at
                (default ``logging.ERROR``). CLAUDE.md reserves
                ``CRITICAL`` for safety events; a caller reporting one
                (e.g. the hold-enforcement escalation) passes
                ``logging.CRITICAL`` here instead of also logging
                separately, so the file log carries the right severity in
                exactly one line rather than two near-duplicates.
        """
        logger.log(log_level, message)
        self.error_occurred.emit(message)
        try:
            self.error_event.emit(
                ErrorEvent(
                    vi_name=vi_name,
                    kind=kind,
                    severity=severity,
                    message=message,
                    timestamp=time.time(),
                )
            )
        except Exception:  # noqa: BLE001 — a signal-emit failure must never disrupt the run
            logger.exception("error_event emit failed in _error")
        # Also surface in the concise status log as a persistent history line.
        # logger.error above already wrote it to file, so emit the signal
        # directly (bypassing _emit_status's logger) to avoid double file
        # logging.
        try:
            self.status_message.emit(message)
        except Exception:  # noqa: BLE001 — status must never disrupt the run
            logger.exception("status emit failed in _error")

    def _emit_fault_event(self, vi_name: str, kind: str, message: str) -> None:
        """Emit a warning-severity ``ErrorEvent`` for an unclaimed VI's runtime fault.

        Deliberately does NOT call ``_error()`` — a stale/disconnected VI
        outside the active run's claim (or with no run active at all) is a
        per-instrument warning, not a run- or machine-wide error, and mere
        staleness never fired ``error_occurred`` before this plan. Keeping
        this a separate, quieter path preserves that for any window still
        only listening to the compat signal (ProcedureWindow) while giving
        fault-aware Monitor UI (instrument panels, banner) the structured
        event.

        Args:
            vi_name: The faulted VI.
            kind: ``"stale"`` or ``"disconnected"`` (the Station
                ``FaultRecord.kind``).
            message: Human-readable description of the fault.
        """
        logger.warning("VI fault: %s (%s) — %s", vi_name, kind, message)
        try:
            self.error_event.emit(
                ErrorEvent(
                    vi_name=vi_name,
                    kind="fault",
                    severity="warning",
                    message=message,
                    timestamp=time.time(),
                )
            )
        except Exception:  # noqa: BLE001 — a signal-emit failure must never disrupt the run
            logger.exception("error_event emit failed in _emit_fault_event")

    # ------------------------------------------------------------------
    # Safety-hold enforcement (the System-Condition standard's hold
    # invariant — GLOSSARY.md's **Safety hold**)
    # ------------------------------------------------------------------

    def _enforce_safety_holds(self, verdict: Verdict) -> None:
        """Keep every held, un-overridden VI at standby for as long as it is held.

        A hold-severity condition's enforcement is a LEVEL-TRIGGERED
        invariant, not a one-shot onset action: for as long as a VI appears
        in ``verdict.held_vis`` and its hold is not acknowledged
        (``override_active()``), that VI is driven to — and kept at — the
        safe idle state its own ``standby()`` defines, using
        ``BaseVirtualInstrument.standby_status()`` (the standby-provenance
        standard, see ``virtual_instruments/base.py``) as the evidence of
        whether it is already there. Replaces the old onset-only dispatch,
        which fired exactly once when a hold began and then never again —
        so a hold that outlived an acknowledge-then-expire cycle (the
        operator moved the VI away from standby while unlocked, then the
        override lapsed) left the VI at its new, unsafe position
        indefinitely: expiry revokes manual permission, it does not
        re-command hardware.

        Generality (CLAUDE.md's "standards over one-off code"): this reads
        only ``Condition.severity == "hold"`` and its ``affected_vis`` —
        both already resolved into ``verdict.held_vis`` by
        ``core.conditions.decide()`` — and never names a flag. No shipped
        VI declares a hold-severity flag today; one is a declaration on a
        category base plus a config threshold, inheriting this enforcement
        with zero Orchestrator change. The one thing that is NOT general: ``standby()`` is the
        only safe response this contract models. A future hold-severity
        flag that needs a different safe action belongs as a declaration
        on the VI itself (a new hook alongside ``standby_status()``), never
        as a special case in this method.

        Rate-limited via ``self._hold_enforcement_last_s`` to at most one
        ``standby()`` attempt per VI per ``self._hold_enforcement_
        interval_s`` — a VI with no recorded attempt yet fires immediately,
        which is what makes hold ONSET and hold RE-ASSERTION the same code
        path (a freshly held VI has never been enforced, so it reads
        ``"away"`` and is dispatched on this very tick). A VI whose
        ``standby()`` keeps raising past ``self._hold_enforcement_
        max_attempts`` is escalated exactly once per episode (see
        ``_emit_hold_enforcement_escalation``) — the invariant is then not
        merely unsatisfied, it is unsatisfiable (a driver failing silently,
        or hardware ignoring the command), which is a CRITICAL condition
        even though this method never transitions the state machine to
        EMERGENCY; that remains a separate decision made elsewhere.

        Args:
            verdict: This tick's ``core.conditions.decide()`` result — read
                for ``held_vis`` only.
        """
        held_names = set(verdict.held_vis)

        # Bookkeeping belongs only to a VI that is CURRENTLY held AND away;
        # drop it the instant either stops being true, so a later, unrelated
        # episode on the same VI starts its attempt count from zero instead
        # of inheriting a stale escalation marker or attempt tally.
        for vi_name in list(self._hold_enforcement_attempts) + list(
            self._hold_enforcement_last_s
        ):
            if vi_name not in held_names:
                self._hold_enforcement_attempts.pop(vi_name, None)
                self._hold_enforcement_last_s.pop(vi_name, None)
                self._hold_enforcement_escalated.discard(vi_name)

        now = time.time()
        for vi_name in sorted(held_names):
            if self.override_active(vi_name):
                # Acknowledged: the operator is deliberately in control.
                continue

            condition = verdict.held_vis[vi_name]
            try:
                vi = self._station.get_vi(vi_name)
            except Exception:
                logger.exception(
                    "held VI '%s' no longer resolves on the Station; "
                    "skipping enforcement this tick",
                    vi_name,
                )
                continue

            try:
                status = vi.standby_status()
            except Exception:
                # Conservative: a VI whose status accessor itself fails
                # (e.g. reaching ramp_status() on unreachable hardware) is
                # treated as "away" — re-issuing standby() is the safe
                # direction to guess wrong in.
                logger.exception(
                    "standby_status() raised on held VI '%s'; treating as 'away'",
                    vi_name,
                )
                status = "away"

            if status != "away":
                # Reached or converging: nothing to enforce this tick.
                self._hold_enforcement_attempts.pop(vi_name, None)
                self._hold_enforcement_last_s.pop(vi_name, None)
                self._hold_enforcement_escalated.discard(vi_name)
                continue

            last_attempt = self._hold_enforcement_last_s.get(vi_name)
            if last_attempt is not None and now - last_attempt < self._hold_enforcement_interval_s:
                continue
            self._hold_enforcement_last_s[vi_name] = now

            try:
                vi.standby()
            except Exception:
                logger.exception(
                    "standby() raised re-asserting the safety hold on VI '%s'", vi_name
                )
                attempts = self._hold_enforcement_attempts.get(vi_name, 0) + 1
                self._hold_enforcement_attempts[vi_name] = attempts
                if (
                    attempts >= self._hold_enforcement_max_attempts
                    and vi_name not in self._hold_enforcement_escalated
                ):
                    self._hold_enforcement_escalated.add(vi_name)
                    self._emit_hold_enforcement_escalation(vi_name, condition, attempts)
                continue

            # Success: standby() returning is not proof the VI reached safe
            # idle (see the docstring) so the attempt counter is
            # deliberately left untouched — only standby_status() leaving
            # "away", checked at the top of a later tick, resets it.
            self._emit_hold_enforcement_event(vi_name, condition)

    def _emit_hold_enforcement_event(self, vi_name: str, condition: Condition) -> None:
        """Announce a routine safety-hold re-assertion: a warning ``ErrorEvent``.

        Re-asserting standby moves hardware without the operator asking for
        it — correct for a hazard interlock, but it must never be silent:
        an operator who finds a field changed with no explanation learns to
        distrust the safety system. Modeled on ``_emit_fault_event()``:
        deliberately does NOT go through ``_error()``, since a single VI
        being kept at standby is VI-scoped, not machine-wide, and must not
        fire the compat ``error_occurred`` signal any more than a per-VI
        comm fault does.

        Args:
            vi_name: The VI just (re-)commanded to standby.
            condition: The hold-severity ``Condition`` holding it —
                ``condition.kind`` names the tripped flag,
                ``condition.message`` is its human-readable description.
        """
        message = (
            f"{vi_name}: driven to standby for the active safety hold "
            f"'{condition.kind}' ({condition.message})"
        )
        logger.warning("Safety-hold enforcement: %s", message)
        self._emit_status(message)
        try:
            self.error_event.emit(
                ErrorEvent(
                    vi_name=vi_name,
                    kind="safety_hold",
                    severity="warning",
                    message=message,
                    timestamp=time.time(),
                )
            )
        except Exception:  # noqa: BLE001 — a signal-emit failure must never disrupt the run
            logger.exception("error_event emit failed in _emit_hold_enforcement_event")

    def _emit_hold_enforcement_escalation(
        self, vi_name: str, condition: Condition, attempts: int
    ) -> None:
        """Escalate an unenforceable safety hold: CRITICAL log + louder ``ErrorEvent``.

        Reached once a held VI's ``standby()`` has raised
        ``self._hold_enforcement_max_attempts`` times in a row with no
        intervening tick where ``standby_status()`` left ``"away"``: the
        hold invariant is not merely unsatisfied, it is unsatisfiable — the
        driver is failing silently or the hardware is ignoring the command.
        Routed through ``_error()`` (louder than the routine per-attempt
        announcement above) so it also reaches the compat
        ``error_occurred`` signal, with ``log_level=logging.CRITICAL`` so
        the file log carries the severity CLAUDE.md reserves for safety
        events, in exactly one line rather than a second, near-duplicate
        one. Fires once per episode — ``_enforce_safety_holds()`` tracks
        that in ``self._hold_enforcement_escalated`` and clears it the
        instant the VI stops being held-and-away.

        Deliberately does NOT transition the state machine to EMERGENCY —
        that is a separate decision this standard does not make; this
        method's whole job is to make the failure visible, not to act on
        it further.

        Args:
            vi_name: The VI whose hold could not be enforced.
            condition: The hold-severity ``Condition`` holding it.
            attempts: The consecutive raise count that triggered escalation.
        """
        self._error(
            f"Safety-hold enforcement on '{vi_name}' failed {attempts} consecutive "
            f"times ({condition.kind}): standby() keeps raising and the hold is not "
            "being enforced. Check the instrument.",
            vi_name=vi_name,
            kind="safety_hold",
            severity="error",
            log_level=logging.CRITICAL,
        )
