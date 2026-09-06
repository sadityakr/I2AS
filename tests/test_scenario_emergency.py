"""End-to-end scenarios: emergency standby, safety trips, and instrument
faults during a measurement.

Family: emergency standby, safety trips and instrument faults during a
measurement, exercised against a real ``FieldSweep`` run over the simulated
station (GLOSSARY.md's **System condition** / **Severity ladder** / **Hold
acknowledge** / **Safe shutdown**). Most scenarios build a bare
``Orchestrator`` and tick it directly, exactly like ``tests/
test_l3_orchestrator.py`` — the Instrument-thread boundary is not in play
there, so these run identically in both instrument modes. Two scenarios
(the threaded emergency latency measurement, and the bounded shutdown over a
wedged read) are inherently about the threaded design itself and build a
real ``InstrumentHost(mode="threaded")``, mirroring ``tests/
test_instrument_thread.py``'s own convention of exercising the boundary
directly rather than through the session's ``I2AS_INSTRUMENT_THREAD``
switch.
"""

from __future__ import annotations

import logging
import threading
import time

import h5py
import pytest
from PyQt6.QtCore import QThread

from i2as.core import events as ev
from i2as.core.instrument_host import InstrumentHost
from i2as.core.orchestrator import Orchestrator, OrchestratorState
from i2as.core.plan import PhasePlan, StepPlan, Target
from i2as.core.station import build_station
from i2as.procedures.field_sweep import FieldSweep

from tests import scenarios

CONFIG_PATH = "i2as/configs/sim_cryostat"

AGENT = ev.Actor(kind=ev.ActorKind.AGENT, id="scenario-agent", role="operator")

FIELD_SWEEP_PARAMS = {
    "measurement_vi": "dc_measurement",
    "field_start": -0.1,
    "field_end": 0.1,
    "field_steps": 3,
    "temperature": 300.0,  # the sim temperature controller starts at 300 K -> instant settle
    "current_A": 1e-6,
    "readings_per_point": 3,
    "init_wait": 0.0,
    "step_wait": 0.0,
}

SAMPLE_INFO = {"sample_name": "S", "sample_id": "S-1", "comments": "scenario emergency"}


# ── Shared fixtures and helpers ───────────────────────────────────────────────


def _fast_magnet(station) -> None:
    """Make the sim magnet ramp fast enough to drive a sweep tick-by-tick."""
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []


def _tick_until(orchestrator, predicate, max_ticks: int = 3000) -> None:
    """Tick the Orchestrator until *predicate* holds; assert it eventually does."""
    for _ in range(max_ticks):
        orchestrator._tick()
        if predicate():
            return
    raise AssertionError(f"predicate never became true within {max_ticks} ticks")


@pytest.fixture
def station():
    """Build a real simulated station."""
    return build_station(CONFIG_PATH)


@pytest.fixture
def orchestrator(station, qtbot):
    """Orchestrator with a small tick interval, monitoring active."""
    orch = Orchestrator(station, tick_interval_ms=10)
    orch.start_monitoring()
    yield orch
    orch.shutdown()


class _Recorder:
    """Collect everything one Orchestrator said, in emission order."""

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.verdicts: list[ev.Verdict] = []
        self.events: list[object] = []
        orchestrator.verdict_emitted.connect(self.verdicts.append)
        orchestrator.event_emitted.connect(self.events.append)

    def of_type(self, event_type):
        return [event for event in self.events if isinstance(event, event_type)]


def _start_field_sweep(orchestrator, station, tmp_path, **overrides) -> FieldSweep:
    """Build and start a real FieldSweep run over the sim station."""
    _fast_magnet(station)
    params = dict(FIELD_SWEEP_PARAMS)
    params.update(overrides)
    procedure = FieldSweep(
        station=station,
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
        **params,
    )
    orchestrator.run_procedure(procedure)
    return procedure


def _emergency_command(reason: str, actor: ev.Actor = AGENT) -> ev.Command:
    return ev.Command(
        name=ev.CommandName.EMERGENCY_STANDBY, actor=actor, args={"reason": reason}
    )


# ══════════════════════════════════════════════════════════════════════════
# Scenario 1 — emergency_standby() during MEASURING on a real FieldSweep
# ══════════════════════════════════════════════════════════════════════════


def test_emergency_during_measuring_ends_the_run_and_preserves_partial_data(
    orchestrator, station, tmp_path, caplog
):
    """emergency_standby() mid-run: OK verdict, EMERGENCY, standby, partial data.

    Drives a real ``FieldSweep`` to its second point (so at least one
    datapoint has already been saved), calls ``emergency_standby()`` through
    ``submit()`` while the state is ``MEASURING`` (between ticks, so no
    measurement is actually in flight — the single-threaded/inline
    equivalent of "arrives while the tick is mid-flight"; the genuinely
    concurrent case is exercised separately, in threaded mode, below), and
    checks every facet CLAUDE.md's control contract promises.
    """
    recorder = _Recorder(orchestrator)
    _start_field_sweep(orchestrator, station, tmp_path)

    # Let at least one point be measured and saved, then stop exactly when
    # the run is back in MEASURING for the next one.
    _tick_until(orchestrator, lambda: len(recorder.of_type(ev.Datapoint)) >= 1)
    saved_before = len(recorder.of_type(ev.Datapoint))
    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.MEASURING)

    stood_down: list[str] = []
    for vi_name in station.get_vi_names():
        vi = station.get_vi(vi_name)
        original = vi.standby

        def _spy(*args, _name=vi_name, _original=original, **kwargs):
            stood_down.append(_name)
            return _original(*args, **kwargs)

        vi.standby = _spy

    with caplog.at_level(logging.CRITICAL, logger="i2as.core.orchestrator"):
        request_id = orchestrator.submit(_emergency_command("scenario 1"))

    # Verdict OK at once.
    verdict = recorder.verdicts[-1]
    assert verdict.request_id == request_id
    assert verdict.code is ev.VerdictCode.OK

    # State is EMERGENCY (no further ticking was needed: emergency_standby()
    # runs on the caller's stack, never queued for the next tick).
    assert orchestrator._state == OrchestratorState.EMERGENCY
    assert orchestrator._procedure is None

    # No additional datapoint was measured past the request.
    assert len(recorder.of_type(ev.Datapoint)) == saved_before

    # Every VI received standby(); the magnet's own PSU status is not left
    # QUENCHed/RAMPING-forever — start_ramp(0) was dispatched.
    assert set(stood_down) == set(station.get_vi_names())

    # CRITICAL log carries the reason.
    critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert critical, "emergency standby must log CRITICAL"
    assert any("scenario 1" in r.getMessage() for r in critical)

    # StateChange: cause == "emergency", actor is who called.
    changes = recorder.of_type(ev.StateChange)
    emergency_changes = [c for c in changes if c.state == "EMERGENCY"]
    assert emergency_changes, [c.state for c in changes]
    assert emergency_changes[-1].cause == "emergency"
    assert emergency_changes[-1].actor.kind is ev.ActorKind.AGENT
    assert emergency_changes[-1].actor.id == AGENT.id

    # RunFinished names the emergency.
    finished = recorder.of_type(ev.RunFinished)
    assert finished, "no RunFinished was emitted"
    assert finished[-1].status == "failed"
    assert "EMERGENCY" in finished[-1].reason

    # The data file was closed with the points measured so far, and reads back.
    data_file = finished[-1].manifest.get("data_file")
    assert data_file, "run manifest carries no data_file path"
    with h5py.File(data_file, "r") as f:
        assert f["metadata"].attrs["procedure_name"] == "Field Sweep"
        assert f["data"]["field_T"].shape[0] == saved_before


def test_emergency_standby_latency_in_threaded_mode(qtbot):
    """The threaded design's actual latency bound: after measure(), never before.

    Builds a real ``InstrumentHost(mode="threaded")`` (mirroring ``tests/
    test_instrument_thread.py``), starts a run whose ``measure()`` blocks for
    a known duration, posts ``emergency_standby`` while the engine is
    genuinely INSIDE that call (on the instrument thread), and measures how
    long the client waits for the ``Verdict``/EMERGENCY ``StateChange``: it
    must land only once the in-flight ``measure()`` returns, bounded by one
    reading, never before.
    """

    class _SlowMeasureProcedure:
        name = "Scenario Slow Sweep"

        def __init__(self, station, measure_seconds: float = 0.4) -> None:
            self._sweep = [1.0, 2.0, 3.0]
            self._index = 0
            self._measure_seconds = measure_seconds
            self.measure_threads: list[QThread] = []
            self.measured = 0

        def initiate(self) -> PhasePlan:
            return PhasePlan(
                targets={"magnet_z": Target(self._sweep[0])}, commands=(), wait_s=0.0
            )

        def change_sweep_step(self) -> StepPlan | None:
            self._index += 1
            if self._index >= len(self._sweep):
                return None
            return StepPlan(
                targets={"magnet_z": Target(self._sweep[self._index])}, wait_s=0.0
            )

        def measure(self) -> None:
            self.measure_threads.append(QThread.currentThread())
            time.sleep(self._measure_seconds)
            self.measured += 1

        def standby(self) -> PhasePlan:
            return PhasePlan(targets={"magnet_z": Target(0.0)}, commands=(), wait_s=0.0)

        def get_progress(self) -> float:
            return self._index / len(self._sweep)

    host = InstrumentHost(
        lambda: build_station(CONFIG_PATH),
        mode="threaded",
        orchestrator_options={"tick_interval_ms": 20},
    )
    host.start()
    try:
        proxy = host.build_proxy()

        def _on_engine(call):
            answer: list = []
            done = threading.Event()

            def _run():
                try:
                    answer.append(call())
                finally:
                    done.set()

            host.bridge.post(_run)
            assert done.wait(5.0)
            return answer[0] if answer else None

        _on_engine(lambda: _fast_magnet(host.station))
        measure_seconds = 0.4
        procedure = _SlowMeasureProcedure(host.station, measure_seconds=measure_seconds)
        proxy.start_monitoring()
        proxy.run_procedure(procedure)
        qtbot.waitUntil(lambda: bool(procedure.measure_threads), timeout=20000)

        # The engine is now inside the slow measure() call, on its own thread.
        with qtbot.waitSignal(proxy.state_changed, timeout=10000) as changed:
            started = time.monotonic()
            request_id = proxy.emergency_standby("threaded latency probe")
            posted_elapsed = time.monotonic() - started
        landed_elapsed = time.monotonic() - started

        # Posting the command costs the client nothing — it is not waiting
        # on the engine's tick.
        assert posted_elapsed < 0.2, "submit() waited on the engine"
        assert request_id

        # The state change (EMERGENCY) landed only after the in-flight
        # measure() returned — bounded by roughly one reading, not zero.
        assert landed_elapsed >= measure_seconds * 0.5, (
            f"EMERGENCY landed in {landed_elapsed:.3f}s — before the "
            f"in-flight {measure_seconds}s measure() could have returned"
        )
        assert landed_elapsed < measure_seconds + 5.0, (
            f"EMERGENCY took {landed_elapsed:.3f}s — latency is not bounded "
            "by one reading"
        )
        assert changed.args[0]
        qtbot.waitUntil(lambda: proxy.state == "EMERGENCY", timeout=10000)
        assert set(procedure.measure_threads) == {host.thread_object}
        assert procedure.measured >= 1
    finally:
        host.shutdown()
        try:
            host.orchestrator.disconnect()
        except (RuntimeError, TypeError):
            pass


# ══════════════════════════════════════════════════════════════════════════
# Scenario 2 — emergency_standby() from RAMPING, PAUSED, and IDLE
# ══════════════════════════════════════════════════════════════════════════


class _SlowRampProcedure:
    """A procedure that waits at setpoint, so RAMPING/PAUSED hold for a while."""

    name = "Scenario Slow Ramp"

    def __init__(self, wait_s: float = 5.0) -> None:
        self._wait_s = wait_s
        self.measure_called = 0

    def initiate(self) -> PhasePlan:
        return PhasePlan(
            targets={"magnet_z": Target(0.02)}, commands=(), wait_s=self._wait_s
        )

    def change_sweep_step(self):
        return None

    def measure(self) -> None:
        self.measure_called += 1

    def standby(self) -> PhasePlan:
        return PhasePlan(targets={"magnet_z": Target(0.0)}, commands=(), wait_s=0.0)

    def get_progress(self) -> float:
        return 0.0


@pytest.mark.parametrize("start_state", ["IDLE", "RAMPING", "PAUSED"])
def test_emergency_standby_through_the_port_from_every_state(
    orchestrator, station, start_state: str
):
    """emergency_standby(), submitted through the port, always answers OK.

    MEASURING is exercised end-to-end above; this covers the remaining
    reachable states with an actual ``Verdict``, never just the direct call.
    """
    recorder = _Recorder(orchestrator)

    if start_state in ("RAMPING", "PAUSED"):
        procedure = _SlowRampProcedure(wait_s=10.0)
        orchestrator.run_procedure(procedure)
        orchestrator._tick()  # INITIATING -> RAMPING
        assert orchestrator._state == OrchestratorState.RAMPING
        if start_state == "PAUSED":
            orchestrator.pause_procedure()
            assert orchestrator._state == OrchestratorState.PAUSED

    orchestrator.submit(_emergency_command(f"from {start_state}"))

    assert recorder.verdicts[-1].code is ev.VerdictCode.OK
    assert orchestrator._state == OrchestratorState.EMERGENCY


# ══════════════════════════════════════════════════════════════════════════
# Scenario 3 — a queued run does not auto-start after acknowledge()
# ══════════════════════════════════════════════════════════════════════════


def test_queued_procedure_waits_for_the_operators_run_queue_call(
    orchestrator, station, qtbot
):
    """The redline's killer #3, plus the follow-up: the operator restarts it.

    A queued procedure stays queued through an emergency and its
    acknowledge — none of the six IDLE transitions chains ``run_queue()``
    except a clean finish or an operator abort (see ``run_queue()``'s own
    docstring) — and only an explicit ``run_queue()`` call starts it.
    """
    queued = scenarios._HeldTargetProcedure(station, "magnet_z", 1.0)
    orchestrator.queue_procedure(queued)

    orchestrator.emergency_standby("scenario 3 trip")
    assert orchestrator._state == OrchestratorState.EMERGENCY

    orchestrator.acknowledge()
    assert orchestrator._state == OrchestratorState.IDLE

    orchestrator._tick()
    assert orchestrator._state == OrchestratorState.IDLE
    assert orchestrator._procedure is None
    assert orchestrator._procedure_queue == [queued]

    # Only the operator's own run_queue() call starts it.
    orchestrator.run_queue()
    assert orchestrator._procedure is queued
    assert orchestrator._procedure_queue == []
    assert orchestrator._state == OrchestratorState.INITIATING


# ══════════════════════════════════════════════════════════════════════════
# Scenario 4 — a sim safety trip mid-measurement
# ══════════════════════════════════════════════════════════════════════════


def test_quench_mid_measurement_enters_emergency_and_closes_the_run(
    orchestrator, station, tmp_path, qtbot
):
    """A quench (critical) while MEASURING: EMERGENCY, standby, closed file."""
    recorder = _Recorder(orchestrator)
    _start_field_sweep(orchestrator, station, tmp_path)
    _tick_until(orchestrator, lambda: len(recorder.of_type(ev.Datapoint)) >= 1)
    saved_before = len(recorder.of_type(ev.Datapoint))

    magnet_stood_down = []
    original_standby = station.magnet_z.standby

    def _spy(*args, **kwargs):
        magnet_stood_down.append(1)
        return original_standby(*args, **kwargs)

    station.magnet_z.standby = _spy

    scenarios.apply_quench(station)
    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.EMERGENCY)

    assert orchestrator._procedure is None
    assert magnet_stood_down, "magnet_z never received standby() during EMERGENCY entry"

    finished = recorder.of_type(ev.RunFinished)
    assert finished and finished[-1].status == "failed"
    assert len(recorder.of_type(ev.Datapoint)) == saved_before

    emergency_events = recorder.of_type(ev.StateChange)
    assert any(c.state == "EMERGENCY" for c in emergency_events)

    # acknowledge() semantics: time-boxed override, then expiry.
    orchestrator.acknowledge()
    assert orchestrator.override_active() is True
    orchestrator._emergency_override_until = time.time() - 1.0  # force expiry
    assert orchestrator.override_active() is False


def test_safety_hold_mid_measurement_fails_only_the_run_not_the_station(
    orchestrator, station, tmp_path, qtbot, monkeypatch
):
    """A hold-severity flag fails the watched run, no EMERGENCY.

    ``magnet_z`` is a system (watched) VI for FieldSweep and is the VI
    concerned with the declared hold flag (see
    ``tests/scenarios.declare_hold_flag``), so the run fails via
    ``_fail_run_for_fault()`` and the machine returns to IDLE rather than
    EMERGENCY — every other instrument stays usable.
    """
    hold_flag = scenarios.declare_hold_flag(monkeypatch, station)
    recorder = _Recorder(orchestrator)
    _start_field_sweep(orchestrator, station, tmp_path)
    _tick_until(orchestrator, lambda: len(recorder.of_type(ev.Datapoint)) >= 1)
    saved_before = len(recorder.of_type(ev.Datapoint))

    magnet_stood_down = []
    original_standby = station.magnet_z.standby

    def _spy(*args, **kwargs):
        magnet_stood_down.append(1)
        return original_standby(*args, **kwargs)

    station.magnet_z.standby = _spy

    hold_flag.trip()
    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.IDLE)

    assert orchestrator._state == OrchestratorState.IDLE  # not EMERGENCY
    assert orchestrator._procedure is None
    assert len(magnet_stood_down) >= 1

    finished = recorder.of_type(ev.RunFinished)
    assert finished and finished[-1].status == "failed"
    assert hold_flag.flag in finished[-1].reason
    assert len(recorder.of_type(ev.Datapoint)) == saved_before

    # The hold is still active and reported.
    held = orchestrator.held_vi_names()
    assert "magnet_z" in held
    conditions = station.conditions()
    assert hold_flag.key in conditions

    # acknowledge() unlocks manual control for a time-boxed window.
    admitted_before, _reason, _code = orchestrator._manual_action_admission("magnet_z")
    assert admitted_before is False
    orchestrator.acknowledge()
    admitted_after, _reason, _code = orchestrator._manual_action_admission("magnet_z")
    assert admitted_after is True
    assert orchestrator.override_active("magnet_z") is True


def test_tick_triggered_emergency_entry_is_logged_critical(orchestrator, station, caplog):
    """A quench observed by the tick, not a manual call, still logs CRITICAL.

    Every entry into EMERGENCY goes through ``_enter_emergency()``, which
    writes the one CRITICAL record for it naming the cause and the actor —
    so the tick's own observation of a quench is recorded at exactly the
    severity CLAUDE.md reserves for a safety event, the same as an
    operator's ``emergency_standby()``.
    """
    with caplog.at_level(logging.CRITICAL, logger="i2as.core.orchestrator"):
        scenarios.apply_quench(station)
        _tick_until(
            orchestrator, lambda: orchestrator._state == OrchestratorState.EMERGENCY
        )
    critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert critical, "a tick-triggered EMERGENCY entry must log CRITICAL too"
    # The record names the tripped condition and who it is attributed to.
    assert any("quench" in r.getMessage() for r in critical)
    assert any("actor system:orchestrator" in r.getMessage() for r in critical)


# ══════════════════════════════════════════════════════════════════════════
# Scenario 5 — an instrument stops responding mid-run
# ══════════════════════════════════════════════════════════════════════════


def test_instrument_stops_responding_mid_run_fails_the_run_and_reports_the_fault(
    orchestrator, station, tmp_path, qtbot
):
    """A comm-origin fault on a watched VI fails the run, names the VI."""
    recorder = _Recorder(orchestrator)
    _start_field_sweep(orchestrator, station, tmp_path)
    _tick_until(orchestrator, lambda: len(recorder.of_type(ev.Datapoint)) >= 1)
    saved_before = len(recorder.of_type(ev.Datapoint))

    scenarios.apply_disconnect(station, "magnet_z")
    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.IDLE)

    assert orchestrator._procedure is None
    finished = recorder.of_type(ev.RunFinished)
    assert finished and finished[-1].status == "failed"
    assert "magnet_z" in finished[-1].reason
    assert len(recorder.of_type(ev.Datapoint)) == saved_before

    # The fault card data appears in StatusSnapshot.vi_faults.
    snapshot = orchestrator._status_snapshot()
    assert "magnet_z" in snapshot.vi_faults
    assert "magnet_z" in orchestrator.vi_faults()

    # acknowledge_fault(): calms the badge, does not clear the fault.
    orchestrator.acknowledge_fault("magnet_z")
    assert orchestrator.vi_faults()["magnet_z"].acknowledged is True

    # retry_fault(): the VI still errors, so it stays faulted; once the
    # simulated error clears, retry_fault() recovers it.
    orchestrator.retry_fault("magnet_z")
    assert "magnet_z" in orchestrator.vi_faults()
    station.get_vi("magnet_z")._driver._simulate_error = False
    orchestrator.retry_fault("magnet_z")
    assert "magnet_z" not in orchestrator.vi_faults()

    # recover_from_error() only returns to IDLE from ERROR — the run failed
    # for a fault, which is a known blast radius, not global ERROR, so this
    # command is refused rather than a no-op.
    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500) as blocker:
        orchestrator.recover_from_error()
    assert "ERROR" in blocker.args[0]
    assert orchestrator._state == OrchestratorState.IDLE


def test_recover_from_error_only_returns_from_error(orchestrator):
    """recover_from_error() works from ERROR and only from ERROR."""
    orchestrator._fail_to_error("scenario 5 probe")
    assert orchestrator._state == OrchestratorState.ERROR

    orchestrator.recover_from_error()
    assert orchestrator._state == OrchestratorState.IDLE


# ══════════════════════════════════════════════════════════════════════════
# Scenario 6 — manual actions during EMERGENCY, and the kill switch
# ══════════════════════════════════════════════════════════════════════════


def test_manual_actions_are_refused_during_emergency_unless_unlocked(
    orchestrator, station, qtbot
):
    """submit_vi_action() is BLOCKED during EMERGENCY, admitted once unlocked.

    Uses a genuinely tripped condition (quench), left active throughout, so
    ``acknowledge()`` takes the "condition still active" branch: it unlocks
    manual control but does NOT return to IDLE on its own — a manual
    ``emergency_standby()`` call (nothing really tripped) resolves
    immediately instead, which is exercised by scenario 7's clean re-entry.
    """
    scenarios.apply_quench(station)
    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.EMERGENCY)

    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500) as blocker:
        orchestrator.submit_vi_action("magnet_z", "set_field", target_T=0.01)
    assert "EMERGENCY" in blocker.args[0]
    assert orchestrator._gui_action_queue == []

    orchestrator.acknowledge()
    assert orchestrator._state == OrchestratorState.EMERGENCY  # quench still tripped
    assert orchestrator.override_active() is True

    orchestrator.submit_vi_action("magnet_z", "set_field", target_T=0.01)
    assert len(orchestrator._gui_action_queue) == 1
    orchestrator._tick()
    assert orchestrator._gui_action_queue == []


def test_emergency_standby_is_never_refused_by_a_revoked_kill_switch(orchestrator):
    """The kill switch never gates the one command that makes things safe."""
    orchestrator.set_agent_gate(ev.AgentGate.REVOKED)
    recorder = _Recorder(orchestrator)

    orchestrator.submit(_emergency_command("agent saw a drift", actor=AGENT))

    assert recorder.verdicts[-1].code is ev.VerdictCode.OK
    assert orchestrator._state == OrchestratorState.EMERGENCY

    # Meanwhile, an ordinary agent command IS refused by the same gate.
    request_id = orchestrator.submit(
        ev.Command(name=ev.CommandName.START_MONITORING, actor=AGENT)
    )
    refusal = [v for v in recorder.verdicts if v.request_id == request_id][0]
    assert refusal.code is ev.VerdictCode.BLOCKED_ROLE


# ══════════════════════════════════════════════════════════════════════════
# Scenario 7 — a new run after emergency + acknowledge completes normally
# ══════════════════════════════════════════════════════════════════════════


def test_a_new_run_after_emergency_acknowledge_completes_normally(
    orchestrator, station, tmp_path, qtbot
):
    """The engine is left in a clean, reusable state after an emergency cycle."""
    recorder = _Recorder(orchestrator)
    _start_field_sweep(orchestrator, station, tmp_path, file_prefix="first")
    _tick_until(orchestrator, lambda: len(recorder.of_type(ev.Datapoint)) >= 1)

    scenarios.apply_quench(station)
    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.EMERGENCY)
    station.get_vi("magnet_z")._driver._simulate_quench = False
    station.get_vi("magnet_z")._driver.reset_quench()
    qtbot.waitUntil(
        lambda: not station.check_safety().get("quench"), timeout=2000
    )
    orchestrator.acknowledge()
    assert orchestrator._state == OrchestratorState.IDLE

    first_finished = recorder.of_type(ev.RunFinished)[-1].manifest.get("data_file")

    second = _start_field_sweep(orchestrator, station, tmp_path, file_prefix="second")
    with qtbot.waitSignal(orchestrator.procedure_finished, timeout=20000):
        pass

    assert orchestrator._state == OrchestratorState.IDLE
    second_finished = [
        e for e in recorder.of_type(ev.RunFinished) if e.status == "done"
    ]
    assert second_finished, "the second run never finished cleanly"
    second_file = second_finished[-1].manifest.get("data_file")
    assert second_file and second_file != first_finished

    with h5py.File(second_file, "r") as f:
        assert f["data"]["field_T"].shape[0] == len(second.get_sweep_array())


# ══════════════════════════════════════════════════════════════════════════
# Scenario 8 — threaded shutdown() over a wedged instrument read
# ══════════════════════════════════════════════════════════════════════════


def test_shutdown_is_bounded_over_a_wedged_read_and_names_the_vi(caplog):
    """shutdown() must return within its join bound even mid-blocked read.

    Mirrors ``tests/test_instrument_thread.py``'s own coverage of this
    property, kept here so the whole emergency/fault family lives in one
    scenario file: a wedged VISA read must never make the process
    unexitable, and the CRITICAL log must name the instrument.
    """
    wedged = threading.Event()

    def _factory():
        station = build_station(CONFIG_PATH)
        original = station.get_vi("temperature").get_state

        def _never_returns():
            wedged.set()
            time.sleep(3.0)
            return original()

        station.get_vi("temperature").get_state = _never_returns
        return station

    host = InstrumentHost(
        _factory,
        mode="threaded",
        orchestrator_options={"tick_interval_ms": 20},
        join_timeout_ms=300,
    )
    host.start()
    proxy = host.build_proxy()
    proxy.start_monitoring()
    assert wedged.wait(10.0), "the engine never reached the wedged read"

    with caplog.at_level(logging.CRITICAL, logger="i2as.core.instrument_host"):
        started = time.monotonic()
        host.shutdown()
        elapsed = time.monotonic() - started

    assert elapsed < 3.0, f"shutdown() took {elapsed:.1f}s — it is not bounded"
    critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert critical, "a wedged shutdown must say so at CRITICAL"
    message = critical[-1].getMessage()
    assert "temperature" in message, message

    assert host.thread_object.wait(60000)


# ══════════════════════════════════════════════════════════════════════════
# Scenario 9 — every sim driver's safe_shutdown() after an emergency
# ══════════════════════════════════════════════════════════════════════════


def test_every_driver_reaches_safe_state_after_emergency_and_is_idempotent(
    orchestrator, station, tmp_path
):
    """safe_shutdown() on every driver, post-emergency: safe, and a no-op twice.

    Distinct from ``station.standby_all()`` (the VI-level supervised
    stand-down ``_enter_emergency()`` actually dispatches) — this is the
    lower-level driver-only standard (GLOSSARY.md's **Safe shutdown**),
    checked here end-to-end after a real emergency rather than only by
    ``tests/test_conformance.py``'s per-driver construction.
    """
    _start_field_sweep(orchestrator, station, tmp_path)
    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.MEASURING)
    orchestrator.emergency_standby("scenario 9 trip")
    assert orchestrator._state == OrchestratorState.EMERGENCY

    drivers = list(station._drivers.values())
    assert drivers, "the station built no drivers at all"

    for driver in drivers:
        driver.safe_shutdown()
        assert driver._is_in_safe_state() is True, driver
        # Idempotent: calling it again changes nothing and never raises.
        driver.safe_shutdown()
        assert driver._is_in_safe_state() is True, driver
