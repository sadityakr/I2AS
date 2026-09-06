"""Composable scenario helpers for driving a simulated station into a target state.

The sim drivers already expose test-control knobs (``_simulate_error``,
``_simulate_quench``, ...); every existing test sets
these by hand and hand-writes its own ``qtbot.waitUntil`` convergence check.
This module names those recipes once so a test — or a throwaway exploration
script — can ask for "a safety hold" or "magnet_z disconnected" and get back a
station already in that state, then compose several in one test to ask "what
happens when X and Y are both true".

Input: an already-built ``station`` (``build_station(config_path)``) and
``orchestrator`` (``Orchestrator(station, ...)``, monitoring started) from
the caller's own fixtures, plus pytest-qt's ``qtbot`` for the convergence
wait. Each scenario function also takes the sim driver knob it drives.
Process: set the relevant driver attribute(s), then ``qtbot.waitUntil`` the
condition actually propagating through a real tick — never assert the flag
was set, since the point is to observe what the Orchestrator's tick loop
does with it, not to shortcut past it.
Output: the scenario functions mutate state and return ``None``; ``snapshot()``
returns a plain-dict view (Orchestrator state, held VIs, hold conditions,
comm faults) cheap to print or assert against.
"""

from __future__ import annotations

from typing import Any

from i2as.core.orchestrator import Orchestrator
from i2as.core.plan import PhasePlan, Target
from i2as.core.station import Station

_DEFAULT_TIMEOUT_MS = 3000


def snapshot(station: Station, orchestrator: Orchestrator) -> dict[str, Any]:
    """Return a plain-dict view of what the station currently allows/refuses.

    Args:
        station: The station under observation.
        orchestrator: Its Orchestrator.

    Returns:
        ``{"orchestrator_state": str, "held_vis": {vi_name: condition_key},
        "hold_conditions": [...], "faulted_vis": [...]}`` — cheap to print
        in an exploration script or assert against in a test.
    """
    held = orchestrator._held_vis()
    # get_operational_status()'s "conditions" key is populated by the tick
    # pipeline (_update_operational_status) — empty/absent before the first
    # tick, so a snapshot taken immediately after a scenario setup call
    # (before any tick has run) must not crash.
    status = orchestrator.get_operational_status()
    conditions = status.get("conditions", [])
    return {
        "orchestrator_state": orchestrator._state.name,
        "held_vis": {name: cond.key for name, cond in held.items()},
        "hold_conditions": [c for c in conditions if c["severity"] == "hold"],
        "faulted_vis": list(station.vi_faults()),
    }


HOLD_FLAG = "coolant_low"


class HoldFlag:
    """Handle over one declared hold-severity safety flag — see ``declare_hold_flag()``."""

    def __init__(self, source_vi: Any, flag: str) -> None:
        self._source_vi = source_vi
        self.flag = flag
        self.key = f"safety:{flag}"

    def trip(self) -> None:
        """Make the source VI report the flag from its next ``evaluate_safety()``."""
        self._source_vi._hold_flag_tripped = True

    def clear(self) -> None:
        """Stop reporting it; the hold lifts on the next tick, no acknowledge needed."""
        self._source_vi._hold_flag_tripped = False


def declare_hold_flag(
    monkeypatch: Any,
    station: Station,
    *,
    source_vi: str = "temperature",
    concerned_vis: tuple[str, ...] = ("magnet_z",),
    flag: str = HOLD_FLAG,
) -> HoldFlag:
    """Declare one hold-severity safety flag on a live station, for tests.

    No shipped VI declares a hold-severity flag (the only shipped flag is
    the magnet's critical ``quench``), so a test of the safety-hold half of
    the System-Condition standard declares one exactly the way a real setup
    does: the producer VI names it in ``safety_flags`` and reports it from
    ``evaluate_safety()``, and every concerned VI names it in
    ``safety_concerns()``. Both halves are class declarations, so they are
    installed on the classes and undone by ``monkeypatch``.

    Args:
        monkeypatch: pytest's fixture; owns undoing the declaration.
        station: The live station whose VIs carry the declaration.
        source_vi: Name of the VI that reports the flag.
        concerned_vis: Names of the VIs held while the flag is tripped.
        flag: The flag name.

    Returns:
        A ``HoldFlag`` handle whose ``trip()``/``clear()`` drive it.
    """
    source = station.get_vi(source_vi)
    source_cls = type(source)
    monkeypatch.setattr(
        source_cls,
        "safety_flags",
        {**source_cls.merged_safety_flags(), flag: "hold"},
        raising=False,
    )
    monkeypatch.setattr(
        source_cls,
        "evaluate_safety",
        lambda self, state, _f=flag: {_f: bool(getattr(self, "_hold_flag_tripped", False))},
    )
    for name in concerned_vis:
        monkeypatch.setattr(
            type(station.get_vi(name)), "safety_concerns", lambda self, _f=flag: {_f}
        )
    source._hold_flag_tripped = False
    return HoldFlag(source, flag)


def hold_flag_tripped(
    hold_flag: HoldFlag,
    orchestrator: Orchestrator,
    qtbot: Any,
    *,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
) -> None:
    """Trip a declared hold flag and wait for the resulting hold to land.

    Args:
        hold_flag: The handle ``declare_hold_flag()`` returned.
        orchestrator: The station's Orchestrator, monitoring already started.
        qtbot: pytest-qt's fixture, used to wait for a real tick.
        timeout_ms: ``qtbot.waitUntil`` timeout.

    Raises:
        TimeoutError: If no VI becomes held within ``timeout_ms`` — most
            likely because no VI names the flag in ``safety_concerns()``.
    """
    hold_flag.trip()

    def _tripped() -> bool:
        return any(
            cond.key == hold_flag.key for cond in orchestrator._held_vis().values()
        )

    qtbot.waitUntil(_tripped, timeout=timeout_ms)


def apply_quench(station: Station, *, magnet_vi: str = "magnet_z") -> None:
    """Trigger a simulated quench (no wait — see ``quench()``).

    ``quench`` is a critical-severity flag (station-wide by construction —
    see ``virtual_instruments/README.md``), so this is the "everything
    stops" scenario: EMERGENCY refuses manual control of every VI, held or
    not, once the next tick observes it.

    Args:
        station: The station; must have ``magnet_vi`` registered.
        magnet_vi: Name of the registered magnet VI to quench.
    """
    station.get_vi(magnet_vi)._driver._simulate_quench = True


def quench(
    station: Station,
    orchestrator: Orchestrator,
    qtbot: Any,
    *,
    magnet_vi: str = "magnet_z",
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
) -> None:
    """``apply_quench()`` then wait for the station to enter EMERGENCY.

    Args:
        station: The station; must have ``magnet_vi`` registered.
        orchestrator: Its Orchestrator, monitoring already started.
        qtbot: pytest-qt's fixture.
        magnet_vi: See ``apply_quench()``.
        timeout_ms: ``qtbot.waitUntil`` timeout.
    """
    from i2as.core.orchestrator import OrchestratorState

    apply_quench(station, magnet_vi=magnet_vi)
    qtbot.waitUntil(
        lambda: orchestrator._state == OrchestratorState.EMERGENCY, timeout=timeout_ms
    )


def apply_disconnect(
    station: Station, vi_name: str, *, driver_attr: str = "_driver"
) -> None:
    """Make a VI's driver fail every call (no wait — see ``disconnect()``).

    The same ``_simulate_error`` knob models both "the instrument dropped
    off the bus" and "a measurement instrument returned an error instead of
    data": every sim driver raises ``I2ASCommunicationError`` from
    every public method while it is set, regardless of the VI's role
    (system/measurement) — there is only one comm-fault primitive at
    the driver layer; what differs is which VI it's applied to.

    Only meaningful for a system VI: those are polled every tick
    (``Station.get_state()``) regardless of whether a run is active, so a
    comm error surfaces as a station-wide fault condition on its own. A
    measurement VI is *not* polled at idle — see ``measurement_error()``
    for "a measurement instrument returns an error instead of data".

    Args:
        station: The station; must have ``vi_name`` registered.
        vi_name: Name of the registered VI to fault.
        driver_attr: Name of the driver attribute on the VI — ``"_driver"``
            for a single-driver VI (the standard, e.g. every system VI).
    """
    getattr(station.get_vi(vi_name), driver_attr)._simulate_error = True


def disconnect(
    station: Station,
    orchestrator: Orchestrator,
    qtbot: Any,
    vi_name: str,
    *,
    driver_attr: str = "_driver",
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
) -> None:
    """``apply_disconnect()`` then wait for the fault to register.

    Args:
        station: The station; must have ``vi_name`` registered.
        orchestrator: Its Orchestrator, monitoring already started.
        qtbot: pytest-qt's fixture.
        vi_name: See ``apply_disconnect()``.
        driver_attr: See ``apply_disconnect()``.
        timeout_ms: ``qtbot.waitUntil`` timeout.
    """
    apply_disconnect(station, vi_name, driver_attr=driver_attr)
    qtbot.waitUntil(lambda: vi_name in station.vi_faults(), timeout=timeout_ms)


def measurement_error(
    station: Station,
    vi_name: str,
    *,
    driver_attr: str = "_meter",
) -> None:
    """Make a measurement VI's instrument fail on its next call.

    Unlike ``disconnect()``, this needs no wait: a measurement VI is only
    touched when a run actively arms/reads it (``initiate_measurement()``/
    ``take_reading()``), so the fault fires synchronously the moment the
    caller (a running procedure's ``sample()``, or a direct call in a test)
    next exercises it — there is no debounce buffer and no station-wide
    fault condition to converge on, only an exception at the call site.

    Args:
        station: The station; must have ``vi_name`` registered.
        vi_name: Name of the registered measurement VI to fault.
        driver_attr: Name of the driver attribute to fault. Defaults to
            ``"_meter"`` (the voltmeter ``take_reading()`` actually calls
            for the standard source+voltmeter pairing, e.g.
            ``dc_measurement``); pass ``"_source"``
            to instead fault current-source setup calls, or ``"_driver"``
            for a single-instrument measurement VI.
    """
    getattr(station.get_vi(vi_name), driver_attr)._simulate_error = True


class _HeldTargetProcedure:
    """Minimal procedure claiming one VI at a nonzero target, for scenario composition.

    Not a real measurement recipe — just enough of the ``BaseProcedure``
    surface (``initiate``/``standby``/``get_progress``) for
    ``Orchestrator.run_procedure()`` to accept it, so a scenario can ask
    "what happens with a procedure running" without pulling in a concrete
    procedure module (procedures may not import from ``tests/``, and this
    keeps the dependency the other way around).
    """

    name = "Scenario Procedure"

    def __init__(self, station: Station, vi_name: str, target: float) -> None:
        self._vi_name = vi_name
        self._target = float(target)

    def initiate(self) -> PhasePlan:
        return PhasePlan(
            targets={self._vi_name: Target(self._target)}, commands=(), wait_s=0.0
        )

    def standby(self) -> PhasePlan:
        return PhasePlan(targets={self._vi_name: Target(0.0)}, commands=(), wait_s=0.0)

    def get_progress(self) -> float:
        return 0.0


def running_procedure(
    station: Station,
    orchestrator: Orchestrator,
    *,
    vi_name: str = "magnet_z",
    target: float = 1.0,
) -> _HeldTargetProcedure:
    """Start a minimal procedure ramping ``vi_name`` toward ``target``.

    Use this to compose "a procedure is running" with another scenario
    (a safety hold, ``quench``, ``disconnect``): call this first, tick a
    few times so the run is genuinely in flight, then layer the second
    scenario on top and observe (``snapshot()``) whether the run survives,
    fails, or is refused outright.

    Args:
        station: The station; must have ``vi_name`` registered.
        orchestrator: Its Orchestrator; must be IDLE.
        vi_name: The VI the procedure ramps.
        target: The target value ``initiate()`` requests.

    Returns:
        The started ``_HeldTargetProcedure`` instance, in case the caller
        wants to inspect or abort it later.
    """
    procedure = _HeldTargetProcedure(station, vi_name, target)
    orchestrator.run_procedure(procedure)
    return procedure
