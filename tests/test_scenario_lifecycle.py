# ---
# description: |
#   End-to-end simulation scenarios for the measurement lifecycle: a run
#   started through the client boundary (the OrchestratorProxy, in both the
#   threaded and inline instrument modes), driven to completion, and checked
#   at every layer it touches — the control contract's event stream, the
#   HDF5 file core/data_reader.py reads back, the session layer's RunRecord,
#   the run queue's chaining rule, and the instrument thread's
#   responsiveness and shutdown guarantees.
# last_updated: 2026-09-03
# ---

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from PyQt6.QtCore import QTimer

from i2as.core import events as ev
from i2as.core.data_reader import list_columns, open_run
from i2as.core.instrument_host import InstrumentHost
from i2as.core.plan import PhasePlan, StepPlan
from i2as.core.run_builder import build_procedure
from i2as.core.station import build_station
from i2as.procedures.field_sweep import FieldSweep
from i2as.procedures.temperature_sweep import TemperatureSweep
from i2as.procedures.time_series import TimeSeries
from i2as.session.manager import ExperimentManager
from i2as.session.models import RUN_STATUS_DONE
from i2as.session.store import ExperimentStore, User, UserRoster
from tests.instrument_modes import instrument_mode, on_engine, shutdown_host

CONFIG_PATH = "i2as/configs/sim_cryostat"

SAMPLE_INFO = {"sample_name": "S", "sample_id": "S-1", "comments": ""}

#: The proven-minimal FieldSweep parameter set (tests/test_l3_orchestrator.py's
#: FIELD_SWEEP_PARAMS): a tiny +/-0.1 T sweep, three points, no settle waits.
FIELD_SWEEP_PARAMS: dict[str, Any] = {
    "measurement_vi": "dc_measurement",
    "field_start": -0.1,
    "field_end": 0.1,
    "field_steps": 3,
    "temperature": 300.0,
    "current_A": 1e-6,
    "readings_per_point": 3,
    "init_wait": 0.0,
    "step_wait": 0.0,
}

#: A three-point TemperatureSweep that never actually has to ramp (start ==
#: end == the sim temperature controller's resting value), measured with the DC source/meter.
TEMPERATURE_SWEEP_PARAMS: dict[str, Any] = {
    "measurement_vi": "dc_measurement",
    "temperature_start": 300.0,
    "temperature_end": 300.0,
    "temperature_steps": 3,
    "ramp_rate_K_per_min": 6000.0,
    "point_wait": 0.0,
    "current_A": 1e-6,
    "compliance_A": 1e-3,
    "voltmeter_range_V": 0.1,
    "readings_per_point": 5,
}

#: A three-point TimeSeries: elapsed-time only, no hardware commanded at all.
TIME_SERIES_PARAMS: dict[str, Any] = {
    "measurement_vi": "dc_measurement",
    "step_time_s": 0.01,
    "max_duration_s": 0.02,
    "end_condition": "time",
    "end_value": 300.0,
    "end_tolerance": 0.0,
    "current_A": 1e-6,
    "compliance_A": 1e-3,
    "voltmeter_range_V": 0.1,
    "readings_per_point": 5,
}


class _QueueRun:
    """A single-point, no-target run: instant, and keyword-buildable.

    Minimal on purpose (deliberately not a ``FieldSweep``): what scenario 3
    tests is the queue's ordering and chaining, never a procedure's own
    behaviour, so this commands no hardware at all — no ramp, no wait — and
    finishes on the very first tick that reaches MEASURING. Accepts
    ``build_procedure()``'s keyword contract (``station``, ``sample_info``,
    ``data_directory``, ``file_prefix``, ``experiment_info``, ``**params``)
    so it can be queued either as a live object or as a dict payload resolved
    through the run catalog.
    """

    command_scope = "measurement"

    def __init__(
        self,
        station: Any,
        name: str = "Queue Run",
        sample_info: Any = None,
        data_directory: str = "",
        file_prefix: str = "",
        experiment_info: Any = None,
        **params: Any,
    ) -> None:
        self.name = name
        self._done = False

    def initiate(self) -> PhasePlan:
        return PhasePlan(targets={}, commands=(), wait_s=0.0)

    def change_sweep_step(self) -> StepPlan | None:
        if self._done:
            return None
        self._done = True
        return StepPlan(targets={}, wait_s=0.0)

    def measure(self) -> None:
        pass

    def standby(self) -> PhasePlan:
        return PhasePlan(targets={}, commands=(), wait_s=0.0)

    def get_progress(self) -> float:
        return 1.0 if self._done else 0.0


RUN_CATALOG: dict[str, type] = {
    "FieldSweep": FieldSweep,
    "TemperatureSweep": TemperatureSweep,
    "TimeSeries": TimeSeries,
    "_QueueRun": _QueueRun,
}

#: A queued hop costs one event-loop turn; this bounds every wait generously
#: so a timeout means the boundary is broken, not that the machine was busy.
WAIT_MS = 15000


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def host(qtbot):
    """A started host, in this session's instrument mode, with the run catalog.

    ``instrument_modes.build_host()`` cannot be used unmodified here: it
    fixes ``orchestrator_options`` itself, and every scenario below needs a
    ``run_catalog`` on the engine. The magnet is sped up on the engine's own
    thread before any test touches it (``station.magnet_z`` — a plain
    attribute set, never a hardware call) so a +/-0.1 T sweep settles inside
    a handful of 10 ms ticks instead of the sim's realistic ramp rate.
    """
    built = InstrumentHost(
        lambda: build_station(CONFIG_PATH),
        mode=instrument_mode(),
        orchestrator_options={
            "tick_interval_ms": 10,
            "run_catalog": RUN_CATALOG,
        },
    )
    built.start()

    def _fast_magnet() -> None:
        built.station.magnet_z._default_ramp_rate = 6000.0
        built.station.magnet_z._ramp_segments = []

    on_engine(built, _fast_magnet, settle=False)

    yield built
    shutdown_host(built)


@pytest.fixture
def proxy(host):
    """The client adapter every scenario drives the run through."""
    return host.build_proxy()


def _build(station, cls, params, tmp_path, *, file_prefix="scenario", experiment_info=None):
    """Build a run the way the GUI does: on the client's thread, headlessly."""
    return build_procedure(
        cls,
        station=station,
        params=dict(params),
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
        file_prefix=file_prefix,
        experiment_info=experiment_info,
    )


def _run_to_completion(proxy, qtbot, procedure) -> tuple[list[Any], list[Any]]:
    """Start *procedure* through the proxy and wait for it to finish.

    Returns:
        ``(events, started_dicts)`` — every ``Event`` seen on the one event
        stream while the run was in flight, and every ``run_started`` dict
        (usually just one; ``events`` carries the ``RunStarted``/
        ``RunFinished`` objects too).
    """
    events: list[Any] = []
    proxy.event.connect(events.append)
    proxy.run_procedure(procedure)
    qtbot.waitUntil(
        lambda: any(isinstance(e, ev.RunFinished) for e in events), timeout=WAIT_MS
    )
    qtbot.waitUntil(lambda: proxy.state == "IDLE", timeout=WAIT_MS)
    return events, [e for e in events if isinstance(e, ev.RunStarted)]


# ══════════════════════════════════════════════════════════════════════════
# 1. FieldSweep, both entry points, to completion
# ══════════════════════════════════════════════════════════════════════════


def test_field_sweep_via_proxy_run_procedure_completes_and_is_readable(
    host, proxy, qtbot, tmp_path
):
    """A FieldSweep started through ``proxy.run_procedure`` runs to completion.

    Checks every claim of scenario 1 in one pass: exactly one RunStarted and
    one RunFinished("done"), the datapoint count equals the sweep length, the
    HDF5 file (read through ``core/data_reader.py``) has the right columns
    and point count, ``StatusSnapshot.run`` was non-null during the run and
    null after, and the final state is IDLE.
    """
    snapshots: list[ev.StatusSnapshot] = []
    proxy.status_snapshot_event.connect(snapshots.append)
    datapoints: list[ev.Datapoint] = []
    proxy.datapoint_event.connect(datapoints.append)

    procedure = _build(host.station, FieldSweep, FIELD_SWEEP_PARAMS, tmp_path)
    events, started = _run_to_completion(proxy, qtbot, procedure)

    assert len(started) == 1
    finished = [e for e in events if isinstance(e, ev.RunFinished)]
    assert len(finished) == 1
    assert finished[0].status == RUN_STATUS_DONE
    assert finished[0].run_id == started[0].run_id

    assert len(datapoints) == FIELD_SWEEP_PARAMS["field_steps"]

    data_file = finished[0].manifest["data_file"]
    with open_run(data_file) as handle:
        columns = {info.name for info in list_columns(handle)}
        n_points = handle.n_points
    assert {"field_T", "voltage_V"} <= columns
    assert n_points == FIELD_SWEEP_PARAMS["field_steps"]

    mid_run = [s for s in snapshots if s.seq >= started[0].seq and s.run is not None]
    assert mid_run, "StatusSnapshot.run was never populated while the run was live"
    assert snapshots[-1].run is None, "StatusSnapshot.run stayed populated after IDLE"

    assert proxy.state == "IDLE"


def test_field_sweep_via_submitted_command_with_the_run_catalog(
    host, proxy, qtbot, tmp_path
):
    """The same run, started as a JSON ``Command`` through the run catalog."""
    verdicts: list[ev.Verdict] = []
    events: list[Any] = []
    proxy.verdict.connect(verdicts.append)
    proxy.event.connect(events.append)

    command = ev.Command(
        name=ev.CommandName.RUN_PROCEDURE,
        actor=ev.Actor(kind=ev.ActorKind.AGENT, id="scenario-agent", role="operator"),
        args={
            "procedure": "FieldSweep",
            "params": dict(FIELD_SWEEP_PARAMS),
            "sample_info": SAMPLE_INFO,
            "data_directory": str(tmp_path),
            "file_prefix": "submit",
        },
    )
    request_id = proxy.submit(command)
    qtbot.waitUntil(lambda: len(verdicts) >= 1, timeout=WAIT_MS)

    assert request_id == command.request_id
    assert verdicts[0].request_id == request_id
    assert verdicts[0].code is ev.VerdictCode.OK

    qtbot.waitUntil(
        lambda: any(isinstance(e, ev.RunFinished) for e in events), timeout=WAIT_MS
    )
    qtbot.waitUntil(lambda: proxy.state == "IDLE", timeout=WAIT_MS)

    started = [e for e in events if isinstance(e, ev.RunStarted)]
    finished = [e for e in events if isinstance(e, ev.RunFinished)]
    assert len(started) == 1
    assert len(finished) == 1
    assert finished[0].status == RUN_STATUS_DONE
    assert started[0].actor.kind is ev.ActorKind.AGENT
    assert started[0].request_id == request_id


def test_field_sweep_run_record_completes_with_a_data_file_on_disk(
    host, proxy, qtbot, tmp_path
):
    """The session layer's RunRecord agrees with the HDF5 file it names."""
    store = ExperimentStore(tmp_path / "experiments")
    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe", email="jdoe@example.org"))
    manager = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=proxy,
        config_name="sim_cryostat",
        station=host.station,
        run_catalog=RUN_CATALOG,
    )
    record = manager.start_experiment("Scenario run", "jdoe", SAMPLE_INFO)

    recorded: list[dict] = []
    manager.run_recorded.connect(recorded.append)
    procedure = _build(
        host.station,
        FieldSweep,
        FIELD_SWEEP_PARAMS,
        tmp_path,
        experiment_info=manager.experiment_context(),
    )
    proxy.run_procedure(procedure)
    qtbot.waitUntil(
        lambda: any(r.get("status") == RUN_STATUS_DONE for r in recorded),
        timeout=WAIT_MS,
    )

    stored = store.load(record.experiment_id)
    assert len(stored.runs) == 1
    run = stored.runs[0]
    assert run.status == RUN_STATUS_DONE
    assert run.data_file
    assert Path(run.data_file).exists()


# ══════════════════════════════════════════════════════════════════════════
# 2. TemperatureSweep and TimeSeries, the same way
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "cls,params,axis_column,n_points",
    [
        pytest.param(
            TemperatureSweep,
            TEMPERATURE_SWEEP_PARAMS,
            "temperature_K",
            TEMPERATURE_SWEEP_PARAMS["temperature_steps"],
            id="TemperatureSweep",
        ),
        pytest.param(
            TimeSeries,
            TIME_SERIES_PARAMS,
            "elapsed_s",
            3,  # 0.02 // 0.01 + 1
            id="TimeSeries",
        ),
    ],
)
def test_other_procedures_run_to_completion_the_same_way(
    host, proxy, qtbot, tmp_path, cls, params, axis_column, n_points
):
    """TemperatureSweep and TimeSeries, driven to completion the same way.

    The same checks as the FieldSweep scenario, at each procedure's smallest
    working parameter set: one RunStarted, one RunFinished("done"), the
    datapoint count equal to the sweep length, the HDF5 file's axis column
    and point count, and a final state of IDLE.
    """
    snapshots: list[ev.StatusSnapshot] = []
    proxy.status_snapshot_event.connect(snapshots.append)
    datapoints: list[ev.Datapoint] = []
    proxy.datapoint_event.connect(datapoints.append)

    procedure = _build(host.station, cls, params, tmp_path)
    events, started = _run_to_completion(proxy, qtbot, procedure)

    assert len(started) == 1
    finished = [e for e in events if isinstance(e, ev.RunFinished)]
    assert len(finished) == 1
    assert finished[0].status == RUN_STATUS_DONE

    assert len(datapoints) == n_points

    data_file = finished[0].manifest["data_file"]
    with open_run(data_file) as handle:
        columns = {info.name for info in list_columns(handle)}
        found_points = handle.n_points
    assert {axis_column, "voltage_V"} <= columns
    assert found_points == n_points

    mid_run = [s for s in snapshots if s.seq >= started[0].seq and s.run is not None]
    assert mid_run, "StatusSnapshot.run was never populated while the run was live"
    assert snapshots[-1].run is None

    assert proxy.state == "IDLE"


# ══════════════════════════════════════════════════════════════════════════
# 3. The queue: chained via next_procedure, then empty
# ══════════════════════════════════════════════════════════════════════════


def test_three_queued_procedures_run_in_order_and_chain(host, proxy, qtbot):
    """Queue order and chaining both survive the client boundary.

    Three runs are queued; each finish pulls the next via the engine's own
    ``run_queue()`` chain (``_finish_run()`` -> IDLE -> ``run_queue()``) with
    no client re-queueing anything. A final ``run_queue()`` on the now-empty
    queue starts nothing.
    """
    changes: list[ev.QueueChanged] = []
    started: list[str] = []
    finished: list[str] = []
    proxy.queue_changed_event.connect(changes.append)
    proxy.run_started.connect(lambda m: started.append(m["procedure"]))
    proxy.run_finished.connect(lambda m: finished.append(m["procedure"]))

    proxy.queue_procedure(_QueueRun(host.station, name="Op A"))
    proxy.queue_procedure(_QueueRun(host.station, name="Proc B"))
    proxy.queue_procedure(_QueueRun(host.station, name="Proc C"))
    qtbot.waitUntil(lambda: len(changes) >= 3, timeout=WAIT_MS)
    assert [e["run_class"] for e in changes[-1].entries] == [
        "_QueueRun",
        "_QueueRun",
        "_QueueRun",
    ], "the queue must list every waiting run, in order"

    proxy.run_queue()
    qtbot.waitUntil(lambda: len(finished) == 3, timeout=WAIT_MS)
    qtbot.waitUntil(lambda: proxy.state == "IDLE", timeout=WAIT_MS)

    assert started == ["Op A", "Proc B", "Proc C"]
    assert finished == ["Op A", "Proc B", "Proc C"]
    assert all(c.actor.kind is ev.ActorKind.OPERATOR for c in changes)

    # Nothing left to pull: run_queue() on an empty queue starts nothing.
    blocked: list[str] = []
    proxy.action_blocked.connect(blocked.append)
    before_started, before_finished = list(started), list(finished)
    proxy.run_queue()
    qtbot.wait(300)
    assert started == before_started
    assert finished == before_finished
    assert blocked == [], "an empty queue is not a refusal"


def test_a_queue_changed_event_names_the_actor_who_queued_it(host, proxy, qtbot, tmp_path):
    """Accountability crosses the boundary: the queue shows who queued a run.

    Queued as a JSON command (the only way to attach a non-operator actor
    from a client, per ``proxy.queue_procedure()``'s own docstring), through
    the run catalog, exactly as an agent gateway would.
    """
    changes: list[ev.QueueChanged] = []
    proxy.queue_changed_event.connect(changes.append)
    agent = ev.Actor(kind=ev.ActorKind.AGENT, id="scenario-queue-agent", role="operator")

    command = ev.Command(
        name=ev.CommandName.QUEUE_PROCEDURE,
        actor=agent,
        args={
            "procedure": "FieldSweep",
            "params": dict(FIELD_SWEEP_PARAMS),
            "sample_info": SAMPLE_INFO,
            "data_directory": str(tmp_path),
        },
    )
    request_id = proxy.submit(command)
    qtbot.waitUntil(lambda: bool(changes), timeout=WAIT_MS)

    event = changes[-1]
    assert event.actor.kind is ev.ActorKind.AGENT
    assert event.actor.id == agent.id
    assert event.request_id == request_id
    assert event.entries[0]["run_class"] == "FieldSweep"
    assert event.entries[0]["actor"]["id"] == agent.id
    assert json.loads(json.dumps(event.to_json()))  # JSON-safe end to end

    # Drain it so the engine is left IDLE with an empty queue for teardown.
    proxy.run_queue()
    qtbot.waitUntil(lambda: proxy.state == "IDLE", timeout=WAIT_MS)


# ══════════════════════════════════════════════════════════════════════════
# 4. Two runs back to back: two files, two records, two run ids
# ══════════════════════════════════════════════════════════════════════════


def test_two_runs_back_to_back_produce_two_distinct_files_and_records(
    host, proxy, qtbot, tmp_path
):
    """A second run started right after the first gets its own everything."""
    store = ExperimentStore(tmp_path / "experiments")
    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe", email="jdoe@example.org"))
    manager = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=proxy,
        config_name="sim_cryostat",
        station=host.station,
        run_catalog=RUN_CATALOG,
    )
    record = manager.start_experiment("Two runs", "jdoe", SAMPLE_INFO)

    seen_run_ids: list[str] = []
    proxy.status_snapshot_event.connect(
        lambda s: seen_run_ids.append(s.run["id"]) if s.run is not None else None
    )

    def _run_once(prefix: str) -> tuple[str, str]:
        procedure = _build(
            host.station,
            FieldSweep,
            FIELD_SWEEP_PARAMS,
            tmp_path,
            file_prefix=prefix,
            experiment_info=manager.experiment_context(),
        )
        events, started = _run_to_completion(proxy, qtbot, procedure)
        finished = next(e for e in events if isinstance(e, ev.RunFinished))
        return finished.manifest["data_file"], started[0].run_id

    file1, run_id1 = _run_once("first")
    file2, run_id2 = _run_once("second")

    assert file1 != file2
    assert Path(file1).exists()
    assert Path(file2).exists()

    assert run_id1 != run_id2
    assert run_id1 in seen_run_ids
    assert run_id2 in seen_run_ids

    stored = store.load(record.experiment_id)
    assert len(stored.runs) == 2
    assert {r.status for r in stored.runs} == {RUN_STATUS_DONE}
    assert {r.data_file for r in stored.runs} == {file1, file2}
    assert {r.run_id for r in stored.runs} == {run_id1, run_id2}


# ══════════════════════════════════════════════════════════════════════════
# 5. Responsiveness: the GUI thread keeps firing through a slow measurement
# ══════════════════════════════════════════════════════════════════════════


class _SlowMeasureProcedure:
    """A single-point run whose ``measure()`` blocks the instrument thread.

    No system targets at all (like ``TimeSeries``): ``initiate()``/
    ``standby()`` ramp nothing, so the whole run is exactly one slow
    ``measure()`` and nothing else — the minimum shape needed to put a
    KNOWN, long, synchronous call on the instrument thread while a client-
    side timer is watched.
    """

    name = "Slow Measure"
    command_scope = "measurement"

    def __init__(self, station: Any, measure_seconds: float = 2.0) -> None:
        self._measure_seconds = float(measure_seconds)
        self._done = False
        self.measuring = False
        self.measured = 0

    def initiate(self) -> PhasePlan:
        return PhasePlan(targets={}, commands=(), wait_s=0.0)

    def change_sweep_step(self) -> StepPlan | None:
        if self._done:
            return None
        self._done = True
        return StepPlan(targets={}, wait_s=0.0)

    def measure(self) -> None:
        self.measuring = True
        time.sleep(self._measure_seconds)
        self.measured += 1

    def standby(self) -> PhasePlan:
        return PhasePlan(targets={}, commands=(), wait_s=0.0)

    def get_progress(self) -> float:
        return 1.0 if self._done else 0.0


def test_the_gui_thread_stays_responsive_through_a_slow_datapoint(qtbot):
    """The exit criterion of the whole change, hardcoded to THREADED mode.

    Regardless of which mode this test session runs in, this scenario is
    specifically about the instrument thread: a fresh ``InstrumentHost``
    built here with ``mode="threaded"``, exactly like
    ``tests/test_instrument_thread.py``'s own fixtures always do. A 50 ms
    ``QTimer`` on THIS (the GUI) thread must keep firing at least 20 times
    in 2 s while the sim's ``measure()`` blocks the instrument thread for
    that whole 2 s — the frozen-window symptom the thread was built to fix.
    """
    host = InstrumentHost(
        lambda: build_station(CONFIG_PATH),
        mode="threaded",
        orchestrator_options={"tick_interval_ms": 20},
    )
    host.start()
    try:
        proxy = host.build_proxy()
        procedure = _SlowMeasureProcedure(host.station, measure_seconds=2.0)
        proxy.start_monitoring()
        proxy.run_procedure(procedure)

        ticks: list[float] = []
        heartbeat = QTimer()
        heartbeat.setInterval(50)
        heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
        heartbeat.start()
        try:
            qtbot.waitUntil(lambda: procedure.measuring, timeout=WAIT_MS)
            ticks.clear()
            started = time.monotonic()
            qtbot.wait(2000)
            elapsed = time.monotonic() - started

            assert len(ticks) >= 20, (
                f"the GUI thread fired {len(ticks)} times in {elapsed:.2f} s — "
                "it is being blocked by the engine"
            )
            gaps = [b - a for a, b in zip(ticks, ticks[1:])]
            assert not gaps or max(gaps) < 0.5, (
                f"the GUI thread stalled for {max(gaps):.2f} s"
            )
            # Still ACCEPTING work, not merely repainting: a command posted
            # mid-measurement returns at once, answered later.
            request_id = proxy.pause_procedure()
            assert request_id
        finally:
            heartbeat.stop()

        qtbot.waitUntil(lambda: procedure.measured >= 1, timeout=WAIT_MS)
    finally:
        host.shutdown()


# ══════════════════════════════════════════════════════════════════════════
# 6. A run changes no declaration; seq strictly increases; events round-trip
# ══════════════════════════════════════════════════════════════════════════


def test_a_run_changes_no_declaration_and_every_event_round_trips_through_json(
    host, proxy, qtbot, tmp_path
):
    """The declaration is static; the event stream is the only thing that moves.

    A run neither adds nor removes an instrument, so ``StationInfo`` and the
    capability manifest built from it (``core.capability_manifest.
    build_manifest()``) must read identically before and after. Meanwhile
    every ``StatusSnapshot`` emitted during the run carries a strictly
    increasing ``seq`` (the engine's one monotonic counter, shared by every
    event type), and every event on the stream — of every type, not just
    ``StatusSnapshot`` — survives ``to_json()`` -> ``json.dumps`` ->
    ``json.loads`` -> ``event_from_json()`` unchanged, which is the contract
    that lets it cross a thread boundary today and a process boundary later.
    """
    from i2as.core.capability_manifest import build_manifest

    def _declaration() -> tuple[dict[str, Any], dict[str, Any]]:
        return on_engine(
            proxy,
            lambda: (
                host.station.station_info().to_json(),
                build_manifest(host.station),
            ),
        )

    before_info, before_manifest = _declaration()

    events: list[Any] = []
    proxy.event.connect(events.append)

    procedure = _build(host.station, FieldSweep, FIELD_SWEEP_PARAMS, tmp_path)
    _run_to_completion(proxy, qtbot, procedure)

    after_info, after_manifest = _declaration()
    assert after_info == before_info
    assert after_manifest == before_manifest

    assert events, "no events were observed during the run"
    snapshot_seqs = [e.seq for e in events if isinstance(e, ev.StatusSnapshot)]
    assert len(snapshot_seqs) >= 2, "not enough StatusSnapshot events to check seq"
    for earlier, later in zip(snapshot_seqs, snapshot_seqs[1:]):
        assert later > earlier, "StatusSnapshot.seq did not strictly increase"

    for event in events:
        payload = json.loads(json.dumps(event.to_json()))
        assert ev.event_from_json(payload) == event, (
            f"{type(event).__name__} did not round-trip through JSON"
        )


# ══════════════════════════════════════════════════════════════════════════
# 7. shutdown() after a completed run: no thread alive, the file reopens
# ══════════════════════════════════════════════════════════════════════════


def test_shutdown_after_a_completed_run_leaves_no_thread_and_closes_the_file(
    qtbot, tmp_path
):
    """The engine's own file handle and thread are both really gone.

    Hardcoded to THREADED mode, like scenario 5 — ``thread_object`` is the
    thing being asserted dead, so this is not a property the inline mode
    (which has none) can stand in for. The procedure's own ``standby()``
    closes the ``DataManager`` at the end of a normal run, so this is the
    sanity check that ``shutdown()`` neither leaves a stray handle open nor
    corrupts the file: reopening it read-only afterwards must simply work.
    """
    host = InstrumentHost(
        lambda: build_station(CONFIG_PATH),
        mode="threaded",
        orchestrator_options={"tick_interval_ms": 10, "run_catalog": RUN_CATALOG},
    )
    host.start()
    proxy = host.build_proxy()

    def _fast_magnet() -> None:
        host.station.magnet_z._default_ramp_rate = 6000.0
        host.station.magnet_z._ramp_segments = []

    on_engine(proxy, _fast_magnet, settle=False)

    procedure = _build(host.station, FieldSweep, FIELD_SWEEP_PARAMS, tmp_path)
    events, _started = _run_to_completion(proxy, qtbot, procedure)
    finished = next(e for e in events if isinstance(e, ev.RunFinished))
    data_file = finished.manifest["data_file"]

    thread_object = host.thread_object
    assert thread_object is not None
    assert not thread_object.isFinished()

    host.shutdown()

    assert thread_object.isFinished(), "the instrument thread outlived shutdown()"

    with open_run(data_file) as handle:
        columns = {info.name for info in list_columns(handle)}
        n_points = handle.n_points
    assert {"field_T", "voltage_V"} <= columns
    assert n_points == FIELD_SWEEP_PARAMS["field_steps"]
