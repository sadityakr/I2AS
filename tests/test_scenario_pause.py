"""Pause/resume edge-case scenarios (scenario family s2).

End-to-end scenario tests for the **pause boundary** (GLOSSARY.md), layered
on top of the L3 behavior suite (``tests/test_l3_orchestrator.py``) and the
instrument-thread suite (``tests/test_instrument_thread.py``). Those files
already cover the mechanics of pause/resume in isolation; this file drives
whole scenarios — a real HDF5 file end to end, the control contract's
Verdict/actor surface, the agent feed, the status mirror, and the one
genuinely thread-only timing property — that no single existing test
exercises together.

Every engine-level test builds its own ``Orchestrator`` directly (as
``test_l3_orchestrator.py`` does), so it runs identically regardless of
``I2AS_INSTRUMENT_THREAD`` — that flag only matters to the one test in
the "threaded only" section, which is skipped under inline mode.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import h5py
import pytest

from i2as.core import events as ev
from i2as.core.data_manager import DataManager
from i2as.core.gates import Gate
from i2as.core.orchestrator import Orchestrator, OrchestratorState
from i2as.core.plan import PhasePlan, StepPlan, Target
from i2as.core.station import build_station
from i2as.core.status_mirror import StatusMirror
from i2as.session.agent_feed import (
    RECORD_COMMAND,
    RECORD_EVENT,
    RECORD_VERDICT,
    AgentFeed,
    read_feed,
)
from i2as.session.gateway import Gateway, Role

from tests.instrument_modes import build_host, instrument_mode, shutdown_host

CONFIG_PATH = "i2as/configs/sim_cryostat"

AGENT = ev.Actor(kind=ev.ActorKind.AGENT, id="scenario-s2-agent", role="session")


# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def station():
    """A real simulated station, built fresh per test."""
    return build_station(CONFIG_PATH)


@pytest.fixture
def orchestrator(station, qtbot):
    """A directly-built Orchestrator with monitoring on, torn down cleanly."""
    orch = Orchestrator(station, tick_interval_ms=10)
    orch.start_monitoring()
    yield orch
    orch.shutdown()


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


class _Recorder:
    """Collect every Verdict and Event an Orchestrator emits, in order."""

    def __init__(self, orchestrator) -> None:
        self.verdicts: list[ev.Verdict] = []
        self.events: list[object] = []
        orchestrator.verdict_emitted.connect(self.verdicts.append)
        orchestrator.event_emitted.connect(self.events.append)

    def of_type(self, event_type):
        return [event for event in self.events if isinstance(event, event_type)]


# ── The test procedure: a real 3-point sweep with a real HDF5 file ─────────


class PauseProbeProcedure:
    """A duck-typed 3-point ``magnet_z`` sweep with a genuine HDF5 output file.

    Deliberately minimal, like ``MockProcedure``/``ThreadProbeProcedure`` in
    the sibling suites — what it adds is a real ``DataManager`` so a scenario
    can check the file on disk rather than trust an in-memory counter, and an
    optional real ``measure_seconds`` delay for the scenarios that need a
    datapoint to genuinely take wall-clock time.

    Args:
        station: Ignored; taken so the run builds like any other.
        data_directory: Where the HDF5 file is written.
        measure_seconds: How long each ``measure()`` call sleeps.
        sweep: The magnet_z targets to visit; defaults to three points.
    """

    name = "Pause Probe"

    #: Every instance gets a distinct file_prefix — DataManager's filename
    #: only carries second resolution, so two instances built inside the same
    #: test (e.g. a queued second run) would otherwise collide on disk.
    _next_id = 0

    def __init__(
        self,
        station: Any,
        data_directory: Path | str,
        measure_seconds: float = 0.0,
        sweep: list[float] | None = None,
    ) -> None:
        self._station = station
        self._sweep = list(sweep) if sweep is not None else [1.0, 2.0, 3.0]
        self._index = 0
        self._measure_seconds = float(measure_seconds)
        self.measure_called = 0
        self.saved_indices: list[int] = []
        type(self)._next_id += 1
        self._data_manager: DataManager | None = DataManager(
            data_directory=str(data_directory),
            procedure_name=self.name,
            file_prefix=f"pause_probe_{type(self)._next_id}",
            procedure_params={},
            sample_info={"sample_name": "s", "sample_id": "1", "comments": ""},
            instrument_state={},
            system_targets={"magnet_z": self._sweep[0]},
            measurement_commands=[],
            data_config={"sweep_columns": {"point": "float"}},
            n_sweep_points=len(self._sweep),
        )

    @property
    def filepath(self) -> Path:
        """The HDF5 file's path — valid even after ``close()``."""
        return self._data_manager.filepath if self._data_manager else self._closed_path

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
        if self._measure_seconds:
            time.sleep(self._measure_seconds)
        if self._data_manager is not None:
            self._data_manager.save_datapoint(
                sweep_index=self._index,
                measured_data={"point": float(self._index)},
                station_snapshot={},
            )
        self.saved_indices.append(self._index)
        self.measure_called += 1

    def standby(self) -> PhasePlan:
        self._closed_path = self._data_manager.filepath
        self._data_manager.close()
        self._data_manager = None
        return PhasePlan(targets={"magnet_z": Target(0.0)}, commands=(), wait_s=0.0)

    def abort(self) -> tuple:
        """Close the file with whatever partial data was saved, like BaseProcedure."""
        if self._data_manager is not None:
            self._closed_path = self._data_manager.filepath
            self._data_manager.close()
            self._data_manager = None
        return ()

    def get_progress(self) -> float:
        return self._index / len(self._sweep)


def _read_points(filepath: Path) -> list[float]:
    """Read back the "point" sweep column of a closed PauseProbeProcedure file."""
    with h5py.File(filepath, "r") as f:
        return list(f["data"]["point"][:])


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1 — pause while RAMPING to the first point
# ═══════════════════════════════════════════════════════════════════════════


def test_pause_while_ramping_to_first_point_holds_and_resume_completes_the_sweep(
    orchestrator, station, tmp_path
):
    """Pausing mid-ramp holds the run's own hardware and leaves another VI's
    ramp alone; resuming re-dispatches the same target and the sweep still
    finishes every point.
    """
    # An independent ramp on a VI this run never targets, started manually —
    # exactly what an operator's own front-panel ramp looks like.
    station.temperature.set_temperature(target_K=250.0)
    assert station.temperature.ramp_status() == "RAMPING"

    procedure = PauseProbeProcedure(station, tmp_path)  # slow default magnet ramp
    orchestrator.run_procedure(procedure)
    orchestrator._tick()
    assert orchestrator._state == OrchestratorState.RAMPING

    orchestrator.pause_procedure()
    assert orchestrator._state == OrchestratorState.PAUSED
    assert station.magnet_z._driver.get_status() == "HOLD"

    orchestrator._tick()  # let the ramp tracker re-poll with the hold in place
    active = {r.vi_name for r in orchestrator.active_ramps()}
    assert "magnet_z" not in active, "the run's own ramp must be held, not listed as active"
    assert "temperature" in active, (
        "an unrelated VI's ramp must not be stopped by the pause"
    )
    assert station.temperature.ramp_status() == "RAMPING"  # never held

    # Speed the ramp up for the rest of the run — the point under test here is
    # that resume re-dispatches the SAME target, not how long a slow ramp
    # takes to land it.
    _fast_magnet(station)

    orchestrator.resume_procedure()
    assert orchestrator._state == OrchestratorState.RAMPING
    assert procedure._index == 0  # resumes the same point, not the next one

    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.IDLE)
    assert procedure.measure_called == 3
    assert procedure.saved_indices == [0, 1, 2]
    assert _read_points(procedure.filepath) == [0.0, 1.0, 2.0]


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2 — pause during MEASURING, a real >=1s datapoint
# ═══════════════════════════════════════════════════════════════════════════


def test_pause_during_a_slow_datapoint_defers_and_saves_it_before_pausing(
    orchestrator, station, tmp_path
):
    """A pause requested in MEASURING is answered OK at once, defers, and the
    point in flight (a real ~1s measurement) is saved before PAUSED lands.
    """
    _fast_magnet(station)
    procedure = PauseProbeProcedure(station, tmp_path, measure_seconds=1.0)
    recorder = _Recorder(orchestrator)
    orchestrator.run_procedure(procedure)

    _tick_until(
        orchestrator, lambda: orchestrator._state == OrchestratorState.MEASURING
    )
    assert procedure.measure_called == 0  # this point has not been read yet

    orchestrator.pause_procedure()
    assert orchestrator._state == OrchestratorState.MEASURING  # not paused yet
    assert orchestrator.pause_pending is True

    # Submitted again through the control contract (redundant with the direct
    # call above — a second pending request changes nothing new, covered again
    # precisely in the double-pause scenario) so the Verdict itself can be
    # checked: submit() answers synchronously, before any tick runs.
    command = ev.Command(name=ev.CommandName.PAUSE_PROCEDURE, actor=AGENT)
    request_id = orchestrator.submit(command)
    assert recorder.verdicts, "submit() must answer before returning"
    assert recorder.verdicts[-1].request_id == request_id
    assert recorder.verdicts[-1].code is ev.VerdictCode.OK
    assert orchestrator._state == OrchestratorState.MEASURING

    started = time.monotonic()
    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.PAUSED)
    elapsed = time.monotonic() - started
    assert elapsed >= 0.9, "the ~1s datapoint must actually have been read"
    assert procedure.measure_called == 1  # the point was read, not skipped
    assert procedure._index == 0  # the next point was never asked for
    assert orchestrator.pause_pending is False  # request honoured

    for _ in range(20):
        orchestrator._tick()
    assert orchestrator._state == OrchestratorState.PAUSED
    assert procedure.measure_called == 1  # nothing advances while paused

    orchestrator.resume_procedure()
    _tick_until(orchestrator, lambda: procedure.measure_called == 2)
    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.IDLE)

    assert procedure.saved_indices == [0, 1, 2]  # every point exactly once
    assert _read_points(procedure.filepath) == [0.0, 1.0, 2.0]


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3 — pause requested, then resumed before the boundary lands
# ═══════════════════════════════════════════════════════════════════════════


def test_resume_before_the_boundary_cancels_the_pause_and_the_run_never_pauses(
    orchestrator, station, tmp_path
):
    """Resuming while a MEASURING pause is still pending withdraws it cleanly."""
    _fast_magnet(station)
    procedure = PauseProbeProcedure(station, tmp_path)
    recorder = _Recorder(orchestrator)
    orchestrator.run_procedure(procedure)
    _tick_until(
        orchestrator, lambda: orchestrator._state == OrchestratorState.MEASURING
    )

    pause_id = orchestrator.submit(
        ev.Command(name=ev.CommandName.PAUSE_PROCEDURE, actor=AGENT)
    )
    assert orchestrator.pause_pending is True
    assert recorder.verdicts[-1].request_id == pause_id
    assert recorder.verdicts[-1].code is ev.VerdictCode.OK

    resume_id = orchestrator.submit(
        ev.Command(name=ev.CommandName.RESUME_PROCEDURE, actor=AGENT)
    )
    assert orchestrator.pause_pending is False
    assert recorder.verdicts[-1].request_id == resume_id
    assert recorder.verdicts[-1].code is ev.VerdictCode.OK
    assert orchestrator._state == OrchestratorState.MEASURING

    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.IDLE)
    assert procedure.measure_called == 3
    assert not recorder.of_type(ev.StateChange) or all(
        change.state != "PAUSED" for change in recorder.of_type(ev.StateChange)
    ), "the run must never have entered PAUSED"
    assert _read_points(procedure.filepath) == [0.0, 1.0, 2.0]


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4 — double pause, double resume
# ═══════════════════════════════════════════════════════════════════════════


def test_a_second_pause_while_paused_is_refused_and_changes_nothing(
    orchestrator, station, tmp_path
):
    """Pausing an already-PAUSED run is refused BLOCKED_STATE, not honoured twice."""
    _fast_magnet(station)
    procedure = PauseProbeProcedure(station, tmp_path)
    recorder = _Recorder(orchestrator)
    orchestrator.run_procedure(procedure)
    orchestrator._tick()  # INITIATING -> RAMPING
    orchestrator.pause_procedure()
    assert orchestrator._state == OrchestratorState.PAUSED
    pre_pause_state = orchestrator._pre_pause_state

    verdict_id = orchestrator.submit(
        ev.Command(name=ev.CommandName.PAUSE_PROCEDURE, actor=AGENT)
    )
    verdict = recorder.verdicts[-1]
    assert verdict.request_id == verdict_id
    assert verdict.code is ev.VerdictCode.BLOCKED_STATE
    assert "PAUSED" in verdict.reason
    assert orchestrator._state == OrchestratorState.PAUSED  # unchanged
    assert orchestrator._pre_pause_state == pre_pause_state  # unchanged

    orchestrator.abort_procedure()


def test_a_second_resume_while_running_is_refused_and_changes_nothing(
    orchestrator, station, tmp_path
):
    """Resuming a run that is not paused (and has no pending pause) is refused."""
    _fast_magnet(station)
    procedure = PauseProbeProcedure(station, tmp_path)
    recorder = _Recorder(orchestrator)
    orchestrator.run_procedure(procedure)
    orchestrator._tick()
    orchestrator.pause_procedure()
    assert orchestrator._state == OrchestratorState.PAUSED

    orchestrator.resume_procedure()
    assert orchestrator._state == OrchestratorState.RAMPING  # genuinely running again

    verdict_id = orchestrator.submit(
        ev.Command(name=ev.CommandName.RESUME_PROCEDURE, actor=AGENT)
    )
    verdict = recorder.verdicts[-1]
    assert verdict.request_id == verdict_id
    assert verdict.code is ev.VerdictCode.BLOCKED_STATE
    assert "RAMPING" in verdict.reason
    assert orchestrator._state == OrchestratorState.RAMPING  # unchanged

    orchestrator.abort_procedure()


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5 — pause in INITIATION_GATE and READING_GATE, resume completes it
# ═══════════════════════════════════════════════════════════════════════════


def test_pause_resume_in_initiation_gate_holds_then_the_gate_still_completes(
    orchestrator, station, tmp_path
):
    """A pause mid-INITIATION_GATE holds on the spot; resume finishes stepping
    the gate all the way to MEASURING, not just one more step of it.
    """
    procedure = PauseProbeProcedure(station, tmp_path)
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        return calls["n"] >= 4

    procedure.initiation_gates = lambda: (Gate("settle", check=check),)
    _fast_magnet(station)

    orchestrator.run_procedure(procedure)
    _tick_until(
        orchestrator, lambda: orchestrator._state == OrchestratorState.INITIATION_GATE
    )
    orchestrator._tick()
    n_before_pause = calls["n"]
    assert n_before_pause >= 1

    orchestrator.pause_procedure()
    assert orchestrator._state == OrchestratorState.PAUSED
    assert orchestrator._pre_pause_state == OrchestratorState.INITIATION_GATE
    orchestrator._tick()
    assert calls["n"] == n_before_pause  # nothing steps while paused

    orchestrator.resume_procedure()
    assert orchestrator._state == OrchestratorState.INITIATION_GATE

    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.MEASURING)
    assert calls["n"] == 4  # the gate ran to completion, picking up where it left off

    orchestrator.abort_procedure()


def test_pause_resume_in_reading_gate_holds_then_the_gate_still_completes(
    orchestrator, station, tmp_path
):
    """The READING_GATE counterpart: resume must reach MEASURING, not just tick once."""
    procedure = PauseProbeProcedure(station, tmp_path)
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        return calls["n"] >= 4

    procedure.reading_gates = lambda: (Gate("settle", check=check),)
    _fast_magnet(station)

    orchestrator.run_procedure(procedure)
    _tick_until(
        orchestrator, lambda: orchestrator._state == OrchestratorState.READING_GATE
    )
    orchestrator._tick()
    n_before_pause = calls["n"]

    orchestrator.pause_procedure()
    assert orchestrator._state == OrchestratorState.PAUSED

    orchestrator.resume_procedure()
    assert orchestrator._state == OrchestratorState.READING_GATE

    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.MEASURING)
    assert calls["n"] == 4
    assert n_before_pause < 4

    orchestrator.abort_procedure()


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6 — pause with no run active
# ═══════════════════════════════════════════════════════════════════════════


def test_pause_with_no_run_active_is_refused_and_emits_no_state_change(orchestrator):
    """Refused BLOCKED_STATE, naming the reason, with no side effect at all."""
    recorder = _Recorder(orchestrator)

    verdict_id = orchestrator.submit(
        ev.Command(name=ev.CommandName.PAUSE_PROCEDURE, actor=AGENT)
    )

    assert len(recorder.verdicts) == 1
    verdict = recorder.verdicts[0]
    assert verdict.request_id == verdict_id
    assert verdict.code is ev.VerdictCode.BLOCKED_STATE
    assert "no run is active" in verdict.reason
    assert not verdict.ok
    assert orchestrator._state == OrchestratorState.IDLE
    assert not recorder.of_type(ev.StateChange)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7 — abort while PAUSED
# ═══════════════════════════════════════════════════════════════════════════


def test_abort_while_paused_closes_the_file_with_partial_data_and_finishes_aborted(
    orchestrator, station, tmp_path
):
    """Abort from PAUSED: file closed with partial data, RunFinished(aborted), IDLE."""
    _fast_magnet(station)
    procedure = PauseProbeProcedure(station, tmp_path)
    recorder = _Recorder(orchestrator)
    orchestrator.run_procedure(procedure)
    _tick_until(
        orchestrator, lambda: orchestrator._state == OrchestratorState.MEASURING
    )
    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.SWEEPING)
    orchestrator.pause_procedure()
    assert orchestrator._state == OrchestratorState.PAUSED
    saved_before_abort = list(procedure.saved_indices)
    assert saved_before_abort == [0]  # one point measured, not three

    orchestrator.abort_procedure()

    assert orchestrator._state == OrchestratorState.IDLE
    finished = recorder.of_type(ev.RunFinished)
    assert finished and finished[-1].status == "aborted"

    points = _read_points(procedure.filepath)
    assert list(points) == [0.0]  # exactly the partial data, nothing more, nothing lost


def test_abort_while_paused_chains_into_a_queued_run_by_design(
    orchestrator, station, tmp_path
):
    """Documents an intentional design point the scenario brief did not expect.

    ``abort_procedure()`` is one of the two call sites that deliberately
    chains into ``run_queue()`` (see its own docstring: "The operator ended
    THIS run, not the queue") — so a procedure queued before an abort DOES
    start immediately, and the engine passes through IDLE only for the
    instant between the two. This is not a defect; it is written down here so
    "abort while paused never auto-starts the queue" is not silently assumed
    to hold everywhere it might be relied on. Direct handover (via
    ``queue_procedure()``) is the queue path this run itself exercises.
    """
    _fast_magnet(station)
    procedure = PauseProbeProcedure(station, tmp_path)
    recorder = _Recorder(orchestrator)
    orchestrator.run_procedure(procedure)
    orchestrator._tick()
    orchestrator.pause_procedure()
    assert orchestrator._state == OrchestratorState.PAUSED

    queued = PauseProbeProcedure(station, tmp_path, sweep=[5.0, 6.0])
    orchestrator.queue_procedure(queued)

    orchestrator.abort_procedure()

    states = [c.state for c in recorder.of_type(ev.StateChange)]
    assert "IDLE" in states, "abort still passes through IDLE on its way to the next run"
    # But it does not STAY there: run_queue() pulled the queued procedure
    # within the same abort_procedure() call.
    assert orchestrator._state != OrchestratorState.IDLE
    assert orchestrator._procedure is queued

    orchestrator.abort_procedure()  # clean up the second run too


def test_plain_abort_while_paused_with_an_empty_queue_settles_in_idle(
    orchestrator, station, tmp_path
):
    """With nothing queued, abort from PAUSED leaves the machine sitting in IDLE."""
    _fast_magnet(station)
    procedure = PauseProbeProcedure(station, tmp_path)
    orchestrator.run_procedure(procedure)
    orchestrator._tick()
    orchestrator.pause_procedure()
    assert orchestrator._state == OrchestratorState.PAUSED

    orchestrator.abort_procedure()

    assert orchestrator._state == OrchestratorState.IDLE


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 8 — actor accountability: agent pauses, operator resumes
# ═══════════════════════════════════════════════════════════════════════════


def test_agent_pause_and_operator_resume_carry_the_right_actor_and_agent_feed(
    orchestrator, station, tmp_path
):
    """StateChange carries each actor correctly; the agent feed keeps only the
    agent's half of the trail, exactly per the record standard.
    """
    _fast_magnet(station)
    procedure = PauseProbeProcedure(station, tmp_path)
    recorder = _Recorder(orchestrator)

    feed = AgentFeed(tmp_path / "agent_actions.jsonl", "20260903_s2")
    feed.attach(orchestrator)
    gateway = Gateway(orchestrator, Role.SESSION, "drift-watch", feed=feed)

    orchestrator.run_procedure(procedure)
    orchestrator._tick()
    assert orchestrator._state == OrchestratorState.RAMPING

    pause_request_id = gateway.submit(ev.CommandName.PAUSE_PROCEDURE)
    assert orchestrator._state == OrchestratorState.PAUSED

    changes = recorder.of_type(ev.StateChange)
    paused_change = [c for c in changes if c.state == "PAUSED"][-1]
    assert paused_change.actor.kind is ev.ActorKind.AGENT
    assert paused_change.actor.id == "drift-watch"
    assert paused_change.request_id == pause_request_id

    commands = [r for r in read_feed(feed.path) if r["record"] == RECORD_COMMAND]
    verdicts = [r for r in read_feed(feed.path) if r["record"] == RECORD_VERDICT]
    assert [c["request_id"] for c in commands] == [pause_request_id]
    assert commands[0]["command"] == "pause_procedure"
    assert [v["request_id"] for v in verdicts] == [pause_request_id]
    assert verdicts[0]["verdict"]["code"] == "OK"

    # The operator resumes directly (not through the agent gateway) — the
    # StateChange must name the operator, and the feed must record NOTHING
    # new for it: operator traffic never enters this trail.
    resume_request_id = orchestrator.submit(
        ev.Command(name=ev.CommandName.RESUME_PROCEDURE, actor=ev.OPERATOR)
    )
    resumed_changes = [c for c in recorder.of_type(ev.StateChange) if c.request_id == resume_request_id]
    assert resumed_changes and resumed_changes[-1].actor.kind is ev.ActorKind.OPERATOR

    after = read_feed(feed.path)
    assert len(after) == len(commands) + len(verdicts) + len(
        [r for r in read_feed(feed.path) if r["record"] == RECORD_EVENT]
    ), "the operator's resume must not add a new record to the agent feed"
    assert not any(
        r["request_id"] == resume_request_id for r in after
    ), "operator traffic must never appear in the agent feed"

    orchestrator.abort_procedure()


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 9 — the GUI mirror reflects each step within one snapshot
# ═══════════════════════════════════════════════════════════════════════════


def test_status_mirror_reflects_pause_pending_and_state_together(
    orchestrator, station, tmp_path
):
    """``StatusMirror``'s ``state``/``pause_pending()`` are always read from the
    SAME ``StatusSnapshot`` — so every step of a deferred pause, through
    PAUSED and back out through resume, is self-consistent (the test right
    below pins the other half: the request is broadcast the instant it is
    accepted, while the state is still ``MEASURING``).
    """
    _fast_magnet(station)
    procedure = PauseProbeProcedure(station, tmp_path)
    mirror = StatusMirror.for_engine(orchestrator)

    orchestrator.run_procedure(procedure)
    _tick_until(
        orchestrator, lambda: orchestrator._state == OrchestratorState.MEASURING
    )
    assert mirror.state == "MEASURING"
    assert mirror.pause_pending() is False

    orchestrator.pause_procedure()
    assert orchestrator.pause_pending is True  # the engine itself knows at once

    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.PAUSED)
    assert mirror.state == "PAUSED"
    assert mirror.pause_pending() is False  # honoured, so cleared together with it

    orchestrator.resume_procedure()
    assert mirror.state == "SWEEPING"
    assert mirror.pause_pending() is False

    orchestrator.abort_procedure()
    assert mirror.state == "IDLE"


def test_status_mirror_shows_pause_pending_while_still_in_measuring(
    orchestrator, station, tmp_path
):
    """The GUI-facing counterpart of the engine-level deferred-pause guarantee.

    ``Orchestrator.pause_pending`` is True the instant ``pause_procedure()``
    returns, and accepting the request pushes a fresh ``StatusSnapshot`` for
    it — so a mirror-driven "Pausing" indicator can appear while the window
    still reads MEASURING, which is the whole interval it has to describe.
    Withdrawing the request by resuming takes the indicator down again the
    same way, without waiting for a tick.
    """
    _fast_magnet(station)
    procedure = PauseProbeProcedure(station, tmp_path)
    mirror = StatusMirror.for_engine(orchestrator)

    orchestrator.run_procedure(procedure)
    _tick_until(
        orchestrator, lambda: orchestrator._state == OrchestratorState.MEASURING
    )

    orchestrator.pause_procedure()
    assert orchestrator.pause_pending is True

    assert mirror.state == "MEASURING"
    assert mirror.pause_pending() is True

    # Cancelling the request is broadcast on the spot too — still MEASURING,
    # no tick in between.
    orchestrator.resume_procedure()
    assert orchestrator.pause_pending is False
    assert mirror.state == "MEASURING"
    assert mirror.pause_pending() is False

    orchestrator.abort_procedure()


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 10 — threaded mode only: the click returns fast during a slow point
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    instrument_mode() != "threaded",
    reason="the sub-50ms responsive-click property is specific to threaded mode; "
    "inline mode is single-threaded and cannot demonstrate it",
)
def test_threaded_pause_click_returns_in_under_50ms_during_a_2s_datapoint(
    qtbot, tmp_path
):
    """The exit criterion of the instrument-thread design, pinned at the scenario level.

    A pause click posted while the engine is two seconds deep inside
    ``measure()`` must be acknowledged by the CLIENT almost instantly — the
    whole point of the thread — and PAUSED must still land only after that
    point was read and saved, never in the middle of it.
    """
    host = build_host(CONFIG_PATH, tick_interval_ms=50)
    try:
        proxy = host.build_proxy()

        def _fast_magnet_on_engine():
            host.station.magnet_z._default_ramp_rate = 6000.0
            host.station.magnet_z._ramp_segments = []

        host.bridge.post(_fast_magnet_on_engine)

        entered_measure: list[float] = []

        class _SlowProcedure(PauseProbeProcedure):
            def measure(self):
                entered_measure.append(time.monotonic())
                super().measure()

        procedure = _SlowProcedure(host.station, tmp_path, measure_seconds=2.0)
        proxy.start_monitoring()
        proxy.run_procedure(procedure)

        qtbot.waitUntil(lambda: bool(entered_measure), timeout=20000)

        started = time.monotonic()
        request_id = proxy.pause_procedure()
        click_latency = time.monotonic() - started
        assert click_latency < 0.05, f"pause click took {click_latency:.3f}s to return"
        assert request_id

        with qtbot.waitSignal(proxy.verdict, timeout=20000) as answered:
            pass
        assert answered.args[0].request_id == request_id
        assert answered.args[0].code is ev.VerdictCode.OK

        qtbot.waitUntil(lambda: proxy.state == "PAUSED", timeout=20000)
        assert procedure.measure_called >= 1  # the in-flight point was saved
    finally:
        shutdown_host(host)
