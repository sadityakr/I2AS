"""Operational-status record — the runtime troubleshooting signal.

Sibling to the offline ``i2as.troubleshoot`` engine: that classifies
*communication* faults at setup (app closed); this classifies *progress* during
a live run (app open, reading live Orchestrator/Station state). They share the
``str, Enum`` fault-code + JSON-ready record shape so the troubleshoot layer can
consume both uniformly. This module is pure data assembly and holds no
references to the Orchestrator, Station, or Qt, so it is unit-testable in
isolation.

The record standard
-------------------

One record is assembled per Orchestrator tick, emitted on the
``operational_status`` signal and appended as one JSON line to
``status.jsonl``. The log is the contract: readers
(``i2as.troubleshoot.status_reader``) parse
these keys and nothing else, so a field is only ever *added*, never renamed,
retyped, or given a new meaning. A record always carries every key below; a
value that is not known on this tick is ``None`` (``null`` in the log), never
a missing key.

============================ ============ ===================================
Field                        Type         Meaning
============================ ============ ===================================
``schema``                   int          Record schema version, ``SCHEMA_VERSION``.
``ts``                       float        Epoch seconds when the record was assembled.
``seq``                      int          Process-wide record counter, from 1.
``run_id``                   str | null   Active run's manifest id, null when idle.
``experiment_id``            str | null   Session experiment the run belongs to.
``setup``                    str | null   Setup (config) name this station was built from.
``orch_state``               str          Orchestrator state name, e.g. ``"RAMPING"``.
``elapsed_in_state_s``       float        Seconds since that state was entered.
``wait``                     obj | null   ``{"target_s", "elapsed_s"}`` while settling.
``progress``                 float | null Procedure progress 0..1.
``verdict``                  str          Worst ``RunFaultCode`` this tick.
``alerts``                   list[str]    Human-readable advisory lines.
``vis``                      list[obj]    One ``VIHealth`` dict per system VI.
``active_gates``             list[str]    Pending initiation/reading gate names.
``conditions``               list[obj]    This tick's `Condition` registry.
``actor``                    obj | null   Actor of the last accepted command.
``request_id``               str | null   That command's correlation id.
============================ ============ ===================================

``actor``/``request_id`` name the last command the engine *accepted* — the
``Actor`` that issued it and the id its ``Verdict`` and events carry back —
and are ``null`` until one has been. A refused command (any ``BLOCKED_*``
verdict) never displaces them, because the pair answers "who last got the
engine to do something", and a command that was refused changed nothing. The
``request_id`` is what joins this log to the accountability trails on the
other side of the engine: the same id appears on the command, verdict and
event records of the **Agent feed**, so "the station started ramping at
03:12 and the last thing anyone asked for was request X" and "agent runner-7
asked for X" are one query rather than two guesses.

Schema history: version 1 is every record written before ``schema``/``ts``/
``seq``/``run_id``/``experiment_id``/``setup`` existed — it carried no time at
all, so an agent could not tell a live run from a log left by a process that
died three days ago. Version 1 records have no ``schema`` key; a reader that
needs the distinction treats an absent ``schema`` as 1. Version 2 added those
six; version 3 added ``actor``/``request_id``. Both additions are additive in
the sense the add-only rule means: a version-2 log simply reports the new
fields as ``None``, and stays readable.

``vis`` is the heavy part of the record and is empty on a tick that polled
nothing (monitoring off and IDLE — see
``Orchestrator._update_operational_status``); ``conditions`` on such a tick
carries whatever the condition registry last held, since nothing re-evaluated
it. The header fields above are written on *every* tick regardless, so a gap
in ``status.jsonl`` means the process stopped ticking and nothing else.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum

from i2as.core.conditions import Condition

# Version of the record shape documented in this module's docstring. Bump it
# only when the shape changes; adding a field is a bump, renaming or retyping
# one is forbidden outright.
SCHEMA_VERSION = 3

# Process-wide record counter behind next_sequence_number(). Module-level
# rather than per-Orchestrator because the guarantee readers rely on is "one
# strictly increasing sequence per writing process", and status.jsonl has one
# writer per process.
_sequence = itertools.count(1)


def next_sequence_number() -> int:
    """Return the next process-wide status-record sequence number.

    Starts at 1 and increases by one per call, so a reader can tell a
    genuinely new record from a re-read of the same one, and can detect a
    gap (a tick whose record never reached the log).

    Returns:
        The next sequence number for this process.
    """
    return next(_sequence)


class RunFaultCode(str, Enum):
    """Stable, machine-readable runtime health codes.

    ``str, Enum`` makes each member serialize as its plain string value, so a
    record is directly JSON-ready.

    * ``OK``              — ramping/settling normally, or idle.
    * ``VI_STALE``        — the instrument stopped updating (cached values).
    * ``VI_DISCONNECTED`` — repeated comms failures; assumed off the bus.
    * ``CRITICAL_FLAG``   — this instrument reported a safety flag its class
                            declares ``"critical"`` (the Safety-flag manifest
                            standard), which is what forces EMERGENCY. Never
                            names a particular flag: a magnet's ``quench``, a
                            pressure switch and an interlock all reach this
                            code by declaring the same severity.
    * ``RAMP_STALLED``    — a ramp made no progress for several ticks
                            (``i2as.core.stall_detection``).
    """

    OK = "OK"
    VI_STALE = "VI_STALE"
    VI_DISCONNECTED = "VI_DISCONNECTED"
    CRITICAL_FLAG = "CRITICAL_FLAG"
    RAMP_STALLED = "RAMP_STALLED"


# Higher = more severe; the record's overall verdict is the worst VI's code.
# Physical instrument faults (a critical flag, lost comms) outrank a
# progress stall.
_SEVERITY: dict[RunFaultCode, int] = {
    RunFaultCode.OK: 0,
    RunFaultCode.VI_STALE: 2,
    RunFaultCode.RAMP_STALLED: 3,
    RunFaultCode.CRITICAL_FLAG: 4,
    RunFaultCode.VI_DISCONNECTED: 4,
}

_SEVERITY_BY_VALUE: dict[str, int] = {code.value: sev for code, sev in _SEVERITY.items()}


def _worse(a: RunFaultCode, b: RunFaultCode) -> RunFaultCode:
    """Return the more severe of two codes."""
    return a if _SEVERITY[a] >= _SEVERITY[b] else b


def worst_code(code_values: list[str]) -> str:
    """Return the most severe of a list of code string values (OK if empty)."""
    if not code_values:
        return RunFaultCode.OK.value
    return max(code_values, key=lambda c: _SEVERITY_BY_VALUE.get(c, 0))


@dataclass
class VIHealth:
    """One system VI's ramp-progress facts and verdict within a single tick.

    ``@dataclass`` auto-generates ``__init__``/``__repr__``/``__eq__`` from the
    field list — this is a pure data record, so that is exactly the intent
    (mirrors ``troubleshoot.engine.ProbeResult``).
    """

    vi_name: str
    value: float | None        # current value in user units (T, K)
    target: float | None       # active ramp target, same units
    gap: float | None          # |value - target|
    closing: float | None      # gap decrease since last tick (+ = converging)
    rate: float | None          # user units per minute
    eta_s: float | None        # gap / rate, seconds to target at current rate
    ramp_status: str           # RAMPING / TARGET_REACHED / IDLE
    phase: str | None          # sub-phase (e.g. warmup/ramping) or None if N/A
    code: RunFaultCode
    detail: str = ""

    def as_dict(self) -> dict:
        """Return a JSON-ready plain dict (RunFaultCode becomes its string)."""
        data = asdict(self)
        data["code"] = self.code.value
        return data


def _condition_as_dict(condition: Condition) -> dict:
    """Render one `Condition` as the JSON-safe shape carried in the record.

    Args:
        condition: The condition to render.

    Returns:
        A plain dict with ``key``, ``origin``, ``severity``, ``kind``,
        ``message``, ``affected`` (``"all"`` for a station-wide condition,
        else a sorted list of VI names), ``since``, and ``acknowledged``.
    """
    return {
        "key": condition.key,
        "origin": condition.origin,
        "severity": condition.severity,
        "kind": condition.kind,
        "message": condition.message,
        "affected": (
            "all" if condition.affected_vis is None else sorted(condition.affected_vis)
        ),
        "since": condition.since,
        "acknowledged": condition.acknowledged,
    }


def build_operational_status(
    *,
    orch_state: str,
    elapsed_in_state_s: float,
    state: dict[str, dict],
    ramp_info: dict[str, dict],
    prev_gaps: dict[str, float],
    wait_target_s: float | None = None,
    wait_elapsed_s: float | None = None,
    progress: float | None = None,
    active_gates: list[str] | None = None,
    conditions: Sequence[Condition] = (),
    ts: float | None = None,
    seq: int | None = None,
    run_id: str | None = None,
    experiment_id: str | None = None,
    setup: str | None = None,
    actor: dict | None = None,
    request_id: str | None = None,
) -> tuple[dict, dict[str, float]]:
    """Assemble one operational-status record and the next-tick gap map.

    See the module docstring for the record standard — every field, its type,
    and the rule that an unknown value is ``None`` rather than a missing key.

    No hardware, no Qt, no I/O. The caller (Orchestrator) supplies the
    already-polled ``state`` snapshot and ``ramp_info`` (Station.get_ramp_status,
    which carries value/target/rate/ramp_status/phase per system VI) so this
    does not poll anything itself. The one piece of state it does touch is the
    process-wide sequence counter, and only when the caller does not supply
    ``seq``: stamping identity is part of assembling a record, and a caller
    that forgot would produce a log that cannot be ordered. Only unambiguous
    codes are set here; the heuristic stall verdict is layered on by
    ``i2as.core.stall_detection``.

    Args:
        orch_state: Orchestrator state name (e.g. ``"RAMPING"``).
        elapsed_in_state_s: Seconds since that state was entered.
        state: The station state snapshot ``{vi_name: {field: value, ...}}``,
            used for the ``_stale`` / ``_disconnected`` flags.
        ramp_info: ``{vi_name: {"value","target","rate","ramp_status","phase"}}``.
        prev_gaps: Per-VI gap from the previous tick, for the closing fact.
        wait_target_s / wait_elapsed_s: Settle-wait clock, if in a wait.
        progress: Procedure progress 0..1, if a procedure is running.
        active_gates: Names of the currently pending initiation/reading
            gates, if any (see ``i2as.core.gates.Gate``).
        conditions: This tick's System-Condition standard registry (see
            `i2as.core.conditions.Condition`) — the union of the
            Station's comm/safety/trend conditions and, when a session
            envelope is active, its envelope conditions. Defaults to empty,
            so callers that do not pass it (and old status.jsonl records
            written before this field existed) simply carry no conditions —
            additive and backward-compatible. Every advisory-severity
            condition also contributes one line to the returned record's
            ``alerts`` (see below) — the generic mechanism that makes a
            failing trend check visible to any reader of ``alerts`` (the
            status reader renders ``alerts``, not ``conditions``) without
            this module knowing anything about trend checks specifically; a
            hold/critical condition is already visible through its own
            enforcement (standby, EMERGENCY) and is not duplicated here.
        ts: Epoch seconds to stamp on the record; defaults to ``time.time()``.
        seq: Sequence number to stamp; defaults to the next process-wide
            number from `next_sequence_number()`.
        run_id: The active run's manifest id, or None when no run is active.
        experiment_id: The session experiment this run belongs to, or None
            when the engine has not been told of one. Null today on every
            record: the session layer's only push-down to the Orchestrator is
            the experiment envelope, which carries bounds and no identity.
            The field exists so a reader can join a record to an experiment
            the moment that channel does, and so the join key never has to be
            retrofitted into an existing schema version.
        setup: The setup (config) name this station was built from, from
            `Station.setup_name()`, or None for a Station built in-process
            without a config directory.
        actor: The `events.Actor` of the last command the engine accepted,
            already rendered as its JSON dict — this module is pure data
            assembly and does not import the contract to re-render one. None
            until a command has been accepted.
        request_id: That command's correlation id, the join key into the
            **Agent feed**. None alongside ``actor``.

    Returns:
        ``(record, new_gaps)`` — the JSON-ready record dict and the gap map to
        pass back as ``prev_gaps`` next tick.
    """
    vis: list[dict] = []
    new_gaps: dict[str, float] = {}
    verdict = RunFaultCode.OK

    # Which VI reported a flag its class declares "critical" — read off this
    # tick's condition registry rather than off any one instrument's status
    # word, so a magnet's quench, an interlock and a pressure switch all
    # reach RunFaultCode.CRITICAL_FLAG by declaring the same severity (the
    # Safety-flag manifest standard). Critical conditions are station-wide
    # (affected_vis is None by construction), so the attribution here is to
    # the VI that REPORTED the flag, which is what source_vis names.
    critical_by_vi: dict[str, str] = {}
    for condition in conditions:
        if condition.origin == "safety" and condition.severity == "critical":
            for source in condition.source_vis:
                critical_by_vi.setdefault(source, condition.kind)

    for vi_name, ramp in ramp_info.items():
        vi_state = state.get(vi_name, {})
        value = ramp.get("value")
        target = ramp.get("target")
        rate = ramp.get("rate")
        ramp_status = ramp.get("ramp_status", "IDLE")
        phase = ramp.get("phase")

        gap: float | None = None
        closing: float | None = None
        eta_s: float | None = None
        if value is not None and target is not None:
            gap = abs(value - target)
            new_gaps[vi_name] = gap
            prev = prev_gaps.get(vi_name)
            if prev is not None:
                closing = prev - gap
            if rate:
                eta_s = gap / (abs(rate) / 60.0)

        code = RunFaultCode.OK
        detail = ""
        if vi_state.get("_disconnected") or ramp.get("_disconnected"):
            code, detail = RunFaultCode.VI_DISCONNECTED, "no response from instrument"
        elif vi_state.get("_stale") or ramp.get("_stale"):
            code, detail = RunFaultCode.VI_STALE, "instrument stopped updating"
        elif vi_name in critical_by_vi:
            code, detail = (
                RunFaultCode.CRITICAL_FLAG,
                f"critical safety flag '{critical_by_vi[vi_name]}' reported",
            )

        vis.append(
            VIHealth(
                vi_name=vi_name,
                value=value,
                target=target,
                gap=gap,
                closing=closing,
                rate=rate,
                eta_s=eta_s,
                ramp_status=ramp_status,
                phase=phase,
                code=code,
                detail=detail,
            ).as_dict()
        )
        verdict = _worse(verdict, code)

    sorted_conditions = sorted(conditions, key=lambda c: c.key)
    # Advisory conditions have no enforcement of their own (see
    # i2as.core.conditions's severity ladder), so this is their only
    # visible trace short of reading status.jsonl directly — one alerts line
    # per condition, generic over origin (today: only "trend").
    advisory_alerts = [
        f"{c.key}: {c.message}" for c in sorted_conditions if c.severity == "advisory"
    ]

    record = {
        "schema": SCHEMA_VERSION,
        "ts": time.time() if ts is None else float(ts),
        "seq": next_sequence_number() if seq is None else int(seq),
        "run_id": run_id,
        "experiment_id": experiment_id,
        "setup": setup,
        "orch_state": orch_state,
        "elapsed_in_state_s": round(elapsed_in_state_s, 1),
        "wait": (
            {"target_s": round(wait_target_s, 1), "elapsed_s": round(wait_elapsed_s or 0.0, 1)}
            if wait_target_s is not None
            else None
        ),
        "progress": progress,
        "verdict": verdict.value,
        "alerts": advisory_alerts,
        "vis": vis,
        "active_gates": list(active_gates) if active_gates else [],
        "conditions": [_condition_as_dict(c) for c in sorted_conditions],
        "actor": dict(actor) if actor else None,
        "request_id": request_id,
    }
    return record, new_gaps
