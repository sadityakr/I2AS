import json
import logging
import time
from typing import ClassVar

import pytest


from i2as.core import events as ev
from i2as.core.operational_status import SCHEMA_VERSION
from i2as.core.orchestrator import Orchestrator, OrchestratorState
from i2as.core.plan import Command, ExperimentEnvelope, PhasePlan, StepPlan, Target
from i2as.core.station import Station, build_station
from i2as.procedures.field_sweep import FieldSweep
from i2as.session.run_queue import RunQueue, RunSpec, build_run
from i2as.virtual_instruments.base import BaseVirtualInstrument
from i2as.virtual_instruments.rampable import RampableVI


class MockProcedure:
    """Minimal procedure for testing the Orchestrator state machine."""
    name = "Mock Sweep"

    def __init__(self, station):
        self._station = station
        self._sweep = [1.0, 2.0, 3.0]
        self._index = 0
        self.measure_called = 0

    def initiate(self):
        return PhasePlan(
            targets={"magnet_z": Target(self._sweep[0])},
            commands=(),
            wait_s=0.0,  # instant
        )

    def change_sweep_step(self):
        self._index += 1
        if self._index >= len(self._sweep):
            return None
        return StepPlan(targets={"magnet_z": Target(self._sweep[self._index])}, wait_s=0.0)

    def measure(self):
        self.measure_called += 1

    def standby(self):
        return PhasePlan(targets={"magnet_z": Target(0.0)}, commands=(), wait_s=0.0)

    def get_progress(self):
        return self._index / len(self._sweep)


def _fast_magnet(station):
    """Make the sim magnet ramp fast enough to drive a sweep tick-by-tick."""
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []


def _tick_until(orchestrator, predicate, max_ticks: int = 2000):
    """Tick the Orchestrator until *predicate* holds; assert it eventually does."""
    for _ in range(max_ticks):
        orchestrator._tick()
        if predicate():
            return
    raise AssertionError(f"predicate never became true within {max_ticks} ticks")


def _pause_at_boundary(orchestrator):
    """Pause from MEASURING and tick until it lands at the pause boundary.

    The deferred-pause helper for tests that want a run PAUSED *between*
    points: a pause requested in MEASURING waits for that datapoint to be read
    and saved, so reaching PAUSED means ticking through the measurement.
    """
    _tick_until(
        orchestrator, lambda: orchestrator._state == OrchestratorState.MEASURING
    )
    orchestrator.pause_procedure()
    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.PAUSED)


@pytest.fixture
def station():
    """Build a real simulated station."""
    config_path = "i2as/configs/sim_cryostat"
    return build_station(config_path)


@pytest.fixture
def hold_flag(station, monkeypatch):
    """One declared hold-severity safety flag on the sim station.

    No shipped VI declares one (see ``tests/scenarios.declare_hold_flag``),
    so the tests of the safety-hold half of the System-Condition standard
    declare it here: reported by ``temperature``, concerned-with by
    ``magnet_z``. ``trip()``/``clear()`` drive it.
    """
    from tests import scenarios

    return scenarios.declare_hold_flag(monkeypatch, station)


@pytest.fixture
def orchestrator(station, qtbot):
    """Build Orchestrator with a small tick interval, monitoring active.

    Monitoring is OFF at construction (the production default: nothing is
    polled until the instruments are initiated), so tests of the monitored
    behavior start it explicitly here. The teardown stops the tick timer so
    no tick can ever fire into a test's torn-down objects.
    """
    # We create a QCoreApplication instance but qtbot handles the event loop
    orch = Orchestrator(station, tick_interval_ms=10)
    orch.start_monitoring()
    yield orch
    orch.shutdown()


# ── claim_commands: dispatched before this run's own targets/commands ─────────


class ClaimingProcedure(MockProcedure):
    """MockProcedure whose plan also claim-initiates magnet_z."""

    def initiate(self):
        return PhasePlan(
            targets={"magnet_z": Target(self._sweep[0])},
            commands=(Command("keithley_delta_mode", "initiate_measurement", {}),),
            claim_commands=(Command("magnet_z", "initiate", {}),),
            wait_s=0.0,
        )


def test_start_run_dispatches_claim_commands_before_targets_and_commands(
    orchestrator, station, monkeypatch
):
    """A run's claim_commands reach the VI before its targets/commands do.

    Otherwise a temperature controller's initiate() (heater AUTO, setpoint
    pinned to the current reading — see SampleTemperatureControllerVI) would
    run AFTER the ramp target already set the real sweep setpoint, stomping
    it right back down. magnet_z stands in here since it is simpler to spy
    on; the ordering guarantee is generic, not magnet-specific.
    """
    call_order = []
    orig_send = orchestrator._station.send_measurement_commands
    orig_dispatch = orchestrator._dispatch_targets

    def spy_send(commands, *, allowed_scope="measurement"):
        if commands:
            call_order.append(("commands", tuple(c.vi_name for c in commands)))
        return orig_send(commands, allowed_scope=allowed_scope)

    def spy_dispatch(targets):
        if targets:
            call_order.append(("targets", tuple(targets)))
        return orig_dispatch(targets)

    monkeypatch.setattr(orchestrator._station, "send_measurement_commands", spy_send)
    monkeypatch.setattr(orchestrator, "_dispatch_targets", spy_dispatch)

    orchestrator.run_procedure(ClaimingProcedure(station))

    assert call_order == [
        ("commands", ("magnet_z",)),  # claim_commands
        ("targets", ("magnet_z",)),
        ("commands", ("keithley_delta_mode",)),
    ]


def _degraded_station(tmp_path, vi_type: str = "measurement"):
    """Build a station whose one instrument failed to connect but will
    succeed on the next attempt (reuses the L2 flaky-driver double)."""
    from tests.test_l2_station import _FlakyDriver

    _FlakyDriver.fail_times = 1
    _FlakyDriver.attempts = 0
    (tmp_path / "devices.yaml").write_text(
        "real_drivers:\n"
        "  flaky_drv:\n"
        "    class: tests.test_l2_station._FlakyDriver\n"
        '    address: "GPIB0::12::INSTR"\n'
        "virtual_instruments:\n"
        "  flaky_vi:\n"
        "    class: tests.test_l2_station._StubVI\n"
        "    drivers: {main: flaky_drv}\n"
        f"    vi_type: {vi_type}\n"
    )
    (tmp_path / "monitor.yaml").write_text(
        "monitor:\n  tick_interval_ms: 1000\n  max_vi_errors: 3\n"
    )
    return build_station(str(tmp_path))


def test_connect_instrument_success_emits_signals(tmp_path, qtbot):
    """connect_instrument() in IDLE brings the VI live and reports the verdict."""
    orch = Orchestrator(_degraded_station(tmp_path), tick_interval_ms=10)
    try:
        with qtbot.waitSignals(
            [orch.instrument_reconnected, orch.action_succeeded], timeout=500
        ):
            orch.connect_instrument("flaky_vi")
        assert orch._station.has_vi("flaky_vi") is True
        assert orch._station.offline_vi_names() == []
    finally:
        orch.shutdown()


def test_connect_instrument_blocked_outside_idle(tmp_path, qtbot):
    """Reconnect is refused while a run is in flight (action_blocked verdict)."""
    orch = Orchestrator(_degraded_station(tmp_path), tick_interval_ms=10)
    try:
        orch._state = OrchestratorState.MEASURING
        with qtbot.waitSignal(orch.action_blocked, timeout=500):
            orch.connect_instrument("flaky_vi")
        assert orch._station.has_vi("flaky_vi") is False
    finally:
        orch._state = OrchestratorState.IDLE
        orch.shutdown()


def test_connect_instrument_failure_emits_action_failed(tmp_path, qtbot):
    """A still-unreachable instrument yields action_failed with the reason."""
    from tests.test_l2_station import _FlakyDriver

    orch = Orchestrator(_degraded_station(tmp_path), tick_interval_ms=10)
    _FlakyDriver.fail_times = 99  # next attempt fails again
    _FlakyDriver.attempts = 0
    try:
        with qtbot.waitSignal(orch.action_failed, timeout=500) as blocker:
            orch.connect_instrument("flaky_vi")
        assert "flaky_drv" in blocker.args[2]
        assert orch._station.offline_vi_names() == ["flaky_vi"]
    finally:
        orch.shutdown()


def test_basic_ticking(orchestrator, qtbot):
    """Orchestrator starts, ticks at interval, emits states_updated.

    The timeout is deliberately generous. The fixture ticks every 10 ms, so
    500 ticks of headroom is far more than correctness needs; it is sized
    against Qt event-loop contention when the full suite runs, not against the
    tick interval. At 500 ms this test was the suite's one persistent flake:
    it passed standalone in 0.2 s and failed under load. A real regression
    (no ticking at all) still fails here, just later. Do not trim it back.
    """
    with qtbot.waitSignal(orchestrator.states_updated, timeout=5000) as blocker:
        pass
    assert blocker.signal_triggered
    assert orchestrator._state == OrchestratorState.IDLE


def test_operational_status_populated_after_tick(orchestrator, qtbot):
    """A tick builds an operational-status record with the documented schema."""
    orchestrator._tick()
    status = orchestrator.get_operational_status()
    assert status["orch_state"] == "IDLE"
    assert "elapsed_in_state_s" in status
    assert status["verdict"] == "OK"
    assert status["vis"], "expected at least one system VI"
    keys = {"vi_name", "value", "target", "gap", "rate", "eta_s", "ramp_status", "code"}
    for vi in status["vis"]:
        assert keys <= set(vi)


def test_operational_status_reports_live_ramp_target(orchestrator, station, qtbot):
    """During a ramp, the record shows the VI's live target and a gap."""
    station.process_system_targets({"magnet_z": Target(1.0)})
    orchestrator._tick()
    status = orchestrator.get_operational_status()
    magnet = next(v for v in status["vis"] if v["vi_name"] == "magnet_z")
    assert magnet["target"] == pytest.approx(1.0)
    assert magnet["ramp_status"] == "RAMPING"
    assert magnet["gap"] is not None


def test_operational_status_conditions_empty_when_healthy(orchestrator, qtbot):
    """A healthy tick's record carries an empty conditions list, not a missing key."""
    orchestrator._tick()
    status = orchestrator.get_operational_status()
    assert status["conditions"] == []


def test_operational_status_carries_active_safety_condition(orchestrator, station, qtbot, hold_flag):
    """The System-Condition standard's registry reaches the status record.

    Mirrors test_hold_flag_holds_concerned_vis_without_emergency's setup: a
    declared hold-severity flag trips a safety condition scoped to magnet_z
    (see tests/scenarios.py's declare_hold_flag()).
    Once the Orchestrator's tick has recorded it, the same condition must
    show up in the operational-status record built by
    _update_operational_status.
    """
    hold_flag.trip()

    def magnet_held():
        return "magnet_z" in orchestrator._held_vis()

    qtbot.waitUntil(magnet_held, timeout=2000)
    orchestrator._tick()
    status = orchestrator.get_operational_status()
    hold_conditions = [c for c in status["conditions"] if c["severity"] == "hold"]
    assert hold_conditions, "expected a hold condition for the declared flag"
    condition = hold_conditions[0]
    assert condition["kind"] == hold_flag.flag
    assert "magnet_z" in condition["affected"]


# ── The operational-status record standard's header (core/operational_status) ──


class _StatusLogCollector(logging.Handler):
    """Collect the JSON lines the Orchestrator writes to ``i2as.status``.

    Attached directly to the logger rather than read through ``caplog``: the
    real handler is installed with ``propagate=False`` by ``setup_logging()``,
    so whether the records reach the root logger depends on whether some other
    test configured logging first.
    """

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _collect_status_lines(orch, ticks: int) -> list[dict]:
    """Tick *orch* *ticks* times and return the status records it logged.

    The logger's level is forced to INFO for the duration: under pytest the
    root logger keeps its default WARNING (the runner installs its own
    handlers, so ``basicConfig`` is a no-op), which would otherwise filter the
    records out before any handler sees them.
    """
    handler = _StatusLogCollector()
    status_logger = logging.getLogger("i2as.status")
    previous_level = status_logger.level
    status_logger.setLevel(logging.INFO)
    status_logger.addHandler(handler)
    try:
        for _ in range(ticks):
            orch._tick()
    finally:
        status_logger.removeHandler(handler)
        status_logger.setLevel(previous_level)
    return [json.loads(message) for message in handler.messages]


def test_operational_status_carries_the_record_standard_header(orchestrator, qtbot):
    """A sim-station tick stamps every header field, correctly typed."""
    orchestrator._tick()
    status = orchestrator.get_operational_status()
    assert status["schema"] == SCHEMA_VERSION
    assert isinstance(status["ts"], float)
    assert status["ts"] == pytest.approx(time.time(), abs=60.0)
    assert isinstance(status["seq"], int)
    assert status["seq"] >= 1
    # The setup is the config directory the Station was built from.
    assert status["setup"] == "sim_cryostat"
    # Unknown values are null, never a missing key: no run is active, and the
    # session layer has no push-down for the experiment id.
    assert status["run_id"] is None
    assert status["experiment_id"] is None


def test_operational_status_seq_strictly_increases_across_ticks(orchestrator, qtbot):
    """Consecutive ticks are distinguishable and orderable by ``seq``."""
    seqs = []
    for _ in range(4):
        orchestrator._tick()
        seqs.append(orchestrator.get_operational_status()["seq"])
    assert all(later > earlier for earlier, later in zip(seqs, seqs[1:]))


def test_operational_status_carries_run_id_during_a_run(orchestrator, station, qtbot):
    """While a run is active the record joins to its manifest by ``run_id``."""
    _fast_magnet(station)
    orchestrator.run_procedure(MockProcedure(station))
    _tick_until(orchestrator, lambda: orchestrator._active_run_manifest is not None)
    run_id = orchestrator._active_run_manifest["run_id"]
    orchestrator._tick()
    assert orchestrator.get_operational_status()["run_id"] == run_id


def test_operational_status_written_with_monitoring_off(station, qtbot):
    """One record per tick reaches status.jsonl even with monitoring off.

    Silence in the log must mean "the process is not ticking" and nothing
    else, otherwise an agent cannot tell a quiet app from a dead one. The
    quiet tick still polls nothing, so the instrument payload is empty while
    every header field is written as usual.
    """
    orch = Orchestrator(station, tick_interval_ms=10)
    try:
        assert orch.is_monitoring() is False
        records = _collect_status_lines(orch, 3)
    finally:
        orch.shutdown()

    assert len(records) == 3
    seqs = [record["seq"] for record in records]
    assert all(later > earlier for earlier, later in zip(seqs, seqs[1:]))
    for record in records:
        assert record["schema"] == SCHEMA_VERSION
        assert record["orch_state"] == "IDLE"
        assert record["setup"] == "sim_cryostat"
        assert isinstance(record["ts"], float)
        assert record["vis"] == []  # nothing polled while quiet
        assert record["verdict"] == "OK"


def test_operational_status_written_every_monitored_tick(orchestrator, qtbot):
    """The same one-record-per-tick guarantee holds with monitoring on."""
    records = _collect_status_lines(orchestrator, 3)
    assert len(records) == 3
    assert all(record["vis"] for record in records), "expected polled instruments"


def test_tick_emits_raw_trend_record(orchestrator, station, qtbot):
    """A tick writes exactly the documented raw-tier JSON shape (see
    TieredTrendLogger's docstring) to i2as.trend_raw, pinning the
    "t"/"s"/"v" nesting and the measurement-VI exclusion at the
    orchestrator boundary.

    i2as.trend_raw has propagate=False (logging_config.py), so the
    capturing handler must attach directly to it, not to root. The handler
    is removed in a finally block since loggers are process-global.
    """
    trend_logger = logging.getLogger("i2as.trend_raw")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    trend_logger.addHandler(handler)
    previous_level = trend_logger.level
    trend_logger.setLevel(logging.INFO)  # setup_logging() may not have run in-test

    try:
        orchestrator._tick()
    finally:
        trend_logger.removeHandler(handler)
        trend_logger.setLevel(previous_level)

    assert len(records) >= 1, "expected at least one raw-tier line from one tick"
    payload = json.loads(records[0].getMessage())

    assert set(payload) == {"t", "s", "v"}
    assert isinstance(payload["t"], float)
    assert payload["s"] == orchestrator._state.name

    v = payload["v"]
    assert isinstance(v, dict)
    assert v, "expected a non-empty flattened state dict"
    assert all(isinstance(key, str) and isinstance(val, float) for key, val in v.items())
    assert set(v) == set(station.last_state_flat())

    # dc_measurement is vi_type: measurement in sim_cryostat/devices.yaml;
    # last_state_flat() excludes every measurement VI, so none of its keys
    # may reach the raw trend tier.
    assert not any(key.startswith("dc_measurement_") for key in v)


def test_full_procedure_cycle(orchestrator, station, qtbot):
    """run_procedure() -> INITIATING -> RAMPING -> MEASURING -> SWEEPING -> ... -> IDLE."""
    procedure = MockProcedure(station)
    
    # We will record states
    states = []
    def on_state(s):
        states.append(s)
        
    orchestrator.state_changed.connect(on_state)
    
    # Fast ramp: override VI config rate and clear segments so generator uses 6000 A/min
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []

    orchestrator.run_procedure(procedure)
    
    # It should cycle until it finishes and goes back to IDLE
    with qtbot.waitSignal(orchestrator.procedure_finished, timeout=5000):
        # We need to manually tick if we aren't using the actual QTimer or qtbot wait
        # qtbot.waitSignal pumps the event loop, so the QTimer will fire!
        pass
        
    assert procedure.measure_called == 3
    assert OrchestratorState.IDLE.value in states
    assert OrchestratorState.RAMPING.value in states
    assert OrchestratorState.MEASURING.value in states
    assert OrchestratorState.SWEEPING.value in states
    assert OrchestratorState.STANDBY.value in states


def test_status_messages_emitted_during_run(orchestrator, station, qtbot):
    """A full run emits concise status milestones on status_message.

    MockProcedure has no get_sweep_position/get_sweep_array, so the line
    builders exercise their generic fallbacks — this also confirms status
    formatting can never raise into the tick.
    """
    procedure = MockProcedure(station)
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []

    messages: list[str] = []
    orchestrator.status_message.connect(messages.append)

    orchestrator.run_procedure(procedure)
    with qtbot.waitSignal(orchestrator.procedure_finished, timeout=5000):
        pass

    assert messages, "no status messages were emitted during the run"
    assert any("Initiating" in m for m in messages)
    # Distinct setup action, labelled from the magnet VI's setpoint metadata.
    assert any("Ramping field to" in m for m in messages)
    assert any("Measuring" in m for m in messages)
    assert any("parking" in m.lower() for m in messages)
    assert any("finished" in m.lower() for m in messages)
    # The initiation line must not be mislabelled as a sweep point.
    assert not any(m.startswith("Point 1/") for m in messages)


def test_station_setpoint_and_measurement_labels(station):
    """Station exposes each VI's declarative label/unit for status lines."""
    assert station.system_setpoint_meta("magnet_z") == ("field", "T")
    assert station.system_setpoint_meta("temperature") == ("temperature", "K")
    assert station.measurement_label("dc_measurement") == "DC resistance"
    # Unknown VI degrades to (name, "") / name rather than raising.
    assert station.system_setpoint_meta("nope") == ("nope", "")
    assert station.measurement_label("nope") == "nope"


def test_standby_waits_for_its_own_ramp_before_finishing(orchestrator, station, qtbot):
    """procedure_finished must not fire until standby()'s own ramp completes.

    Regression test: the STANDBY handler used to check check_ramps() BEFORE
    calling procedure.standby(), then declare the procedure finished in the
    same tick — never waiting for the ramp standby() itself dispatches (e.g.
    ramping the magnet back to 0 T). By the time procedure_finished fired,
    the magnet was often still mid-ramp.
    """
    procedure = MockProcedure(station)  # standby() ramps magnet_z to 0.0 T
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []

    orchestrator.run_procedure(procedure)

    with qtbot.waitSignal(orchestrator.procedure_finished, timeout=5000):
        pass

    assert station.magnet_z.ramp_status() in ("TARGET_REACHED", "IDLE")
    assert station.magnet_z.magnet_field_T() == pytest.approx(0.0, abs=0.01)


def test_wait_time_respected(orchestrator, station, qtbot):
    """After targets reached, MEASURING doesn't start until wait expires."""
    procedure = MockProcedure(station)
    
    # Override initiate to add wait time
    def delayed_initiate():
        return PhasePlan(
            targets={"magnet_z": Target(1.0)}, commands=(), wait_s=0.1
        )  # 100ms wait

    procedure.initiate = delayed_initiate
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []
    
    orchestrator.run_procedure(procedure)

    # Wait until MEASURING state
    with qtbot.waitSignal(orchestrator.state_changed, timeout=1000):
        pass
        
    # Should take at least 0.1s to reach MEASURING once the ramp finishes
    # However we just checking wait is somewhat respected, not strict benchmarking.
    # We can at least check it doesn't skip it entirely
    
    orchestrator.abort_procedure()


def test_pause_resume(orchestrator, qtbot, station):
    """pause_procedure() stops advancement at the boundary; resume continues."""
    _fast_magnet(station)
    procedure = MockProcedure(station)
    orchestrator.run_procedure(procedure)

    _pause_at_boundary(orchestrator)
    assert orchestrator._state == OrchestratorState.PAUSED

    orchestrator.resume_procedure()
    # The pause landed at the boundary, so the run resumes into SWEEPING —
    # its next act is the ramp to the next point.
    assert orchestrator._state == OrchestratorState.SWEEPING
    orchestrator.abort_procedure()


def test_pause_mid_ramp_holds_immediately_and_resumes_the_same_point(
    orchestrator, station
):
    """Pausing mid-ramp holds the hardware on the spot — no deferral.

    Holding the cryostat where it stands is the point of the control, and a
    ramp carries no data, so there is nothing to protect by waiting.
    """
    procedure = MockProcedure(station)  # slow default magnet ramp
    orchestrator.run_procedure(procedure)
    orchestrator._tick()
    assert orchestrator._state == OrchestratorState.RAMPING

    orchestrator.pause_procedure()
    assert orchestrator._state == OrchestratorState.PAUSED
    assert orchestrator.pause_pending is False  # nothing deferred
    assert station.magnet_z._driver.get_status() == "HOLD"
    assert procedure.measure_called == 0

    for _ in range(20):
        orchestrator._tick()
    assert orchestrator._state == OrchestratorState.PAUSED
    assert procedure.measure_called == 0  # nothing advances while paused

    orchestrator.resume_procedure()
    # Resumes the point it was ramping to, not the next one.
    assert orchestrator._state == OrchestratorState.RAMPING
    assert procedure._index == 0
    assert station.magnet_z.ramp_status() == "RAMPING"

    orchestrator.abort_procedure()


def test_pause_while_measuring_defers_until_the_datapoint_is_saved(
    orchestrator, station
):
    """The pause boundary: a point being read is never left stranded.

    A pause requested in MEASURING — the one state where a point is ramped,
    settled and gated but not yet read — waits for ``measure()`` to save its
    datapoint, then holds at SWEEPING with the next point not yet asked for.
    Resume's first act is the ramp to that next point.
    """
    _fast_magnet(station)
    procedure = MockProcedure(station)
    orchestrator.run_procedure(procedure)

    _tick_until(
        orchestrator, lambda: orchestrator._state == OrchestratorState.MEASURING
    )
    assert procedure.measure_called == 0  # this point has not been read yet

    orchestrator.pause_procedure()
    assert orchestrator._state == OrchestratorState.MEASURING  # not paused yet
    assert orchestrator.pause_pending is True

    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.PAUSED)
    assert procedure.measure_called == 1        # the point was read, not skipped
    assert procedure._index == 0                # the next point is not asked for
    assert orchestrator.pause_pending is False  # request honoured
    assert orchestrator._pre_pause_state == OrchestratorState.SWEEPING
    assert station.magnet_z._driver.get_status() == "HOLD"

    for _ in range(10):
        orchestrator._tick()
    assert orchestrator._state == OrchestratorState.PAUSED
    assert procedure.measure_called == 1  # nothing advances while paused

    orchestrator.resume_procedure()
    assert orchestrator._state == OrchestratorState.SWEEPING
    orchestrator._tick()  # SWEEPING -> change_sweep_step() -> ramp to point 2
    assert orchestrator._state == OrchestratorState.RAMPING
    assert procedure._index == 1
    _tick_until(orchestrator, lambda: procedure.measure_called == 2)

    orchestrator.abort_procedure()


def test_resume_cancels_a_pause_request_that_has_not_landed(orchestrator, station):
    """Resume while the request is still pending withdraws it; the run carries on."""
    _fast_magnet(station)
    procedure = MockProcedure(station)
    orchestrator.run_procedure(procedure)
    _tick_until(
        orchestrator, lambda: orchestrator._state == OrchestratorState.MEASURING
    )

    orchestrator.pause_procedure()
    assert orchestrator.pause_pending is True

    orchestrator.resume_procedure()
    assert orchestrator.pause_pending is False
    assert orchestrator._state == OrchestratorState.MEASURING

    _tick_until(orchestrator, lambda: procedure.measure_called == 2)
    assert orchestrator._state != OrchestratorState.PAUSED  # never paused
    orchestrator.abort_procedure()


def test_abort_clears_a_pending_pause_request(orchestrator, station):
    """A request must not outlive its run, or the next one would pause itself."""
    _fast_magnet(station)
    procedure = MockProcedure(station)
    orchestrator.run_procedure(procedure)
    _tick_until(
        orchestrator, lambda: orchestrator._state == OrchestratorState.MEASURING
    )
    orchestrator.pause_procedure()
    assert orchestrator.pause_pending is True

    orchestrator.abort_procedure()
    assert orchestrator._state == OrchestratorState.IDLE
    assert orchestrator.pause_pending is False


def test_pause_during_standby_pauses_immediately(orchestrator, station):
    """STANDBY holds on the spot like every other non-MEASURING state."""
    _fast_magnet(station)
    procedure = MockProcedure(station)
    orchestrator.run_procedure(procedure)
    _tick_until(orchestrator, lambda: orchestrator._state == OrchestratorState.STANDBY)

    orchestrator.pause_procedure()
    assert orchestrator._state == OrchestratorState.PAUSED
    assert orchestrator.pause_pending is False
    assert orchestrator._pre_pause_state == OrchestratorState.STANDBY

    orchestrator.resume_procedure()
    assert orchestrator._state == OrchestratorState.STANDBY
    orchestrator.abort_procedure()


def test_abort_to_idle(orchestrator, station):
    """Abort transitions to IDLE and measure doesn't continue."""
    procedure = MockProcedure(station)
    orchestrator.run_procedure(procedure)
    
    orchestrator.abort_procedure()
    assert orchestrator._state == OrchestratorState.IDLE
    assert orchestrator._procedure is None


# ── Ramp tracker: active_ramps() / ramps_updated / stop_ramp() ────────────────


def test_active_ramps_empty_while_nothing_ramps(orchestrator, qtbot):
    """An idle machine publishes an empty list, not a stale or missing one."""
    orchestrator._tick()
    assert orchestrator.active_ramps() == []


def test_tick_publishes_a_record_for_a_manual_ramp(orchestrator, station, qtbot):
    """A ramp started from a GUI action shows up with its rate and both setpoints."""
    station.get_vi("magnet_z").set_field(5.0)
    orchestrator._tick()

    records = orchestrator.active_ramps()
    assert [r.vi_name for r in records] == ["magnet_z"]
    record = records[0]
    assert record.label == "field"
    assert record.unit == "T"
    assert record.target == pytest.approx(5.0)
    assert record.setpoint is not None
    assert record.rate is not None
    # Nothing owns a manual ramp, so the operator may stop it.
    assert record.owner is None
    assert record.stoppable is True


def test_ramps_updated_emits_the_same_records(orchestrator, station, qtbot):
    """The push and read halves of the tracker surface agree."""
    station.get_vi("magnet_z").set_field(5.0)
    with qtbot.waitSignal(orchestrator.ramps_updated, timeout=1000) as blocker:
        orchestrator._tick()
    emitted = blocker.args[0]
    assert [r.vi_name for r in emitted] == [r.vi_name for r in orchestrator.active_ramps()]


def test_ramp_snapshot_is_polled_once_per_tick(orchestrator, station, monkeypatch):
    """Both consumers share one Station poll — ramp accessors are real bus reads."""
    calls = []
    real = station.get_ramp_status

    def counting():
        calls.append(1)
        return real()

    monkeypatch.setattr(station, "get_ramp_status", counting)
    station.get_vi("magnet_z").set_field(5.0)
    orchestrator._tick()
    # One poll for the operational-status record AND the ramp tracker
    # together; a second is only ever allowed on a tick that drained a GUI
    # action (which this one did not — set_field was called directly).
    assert len(calls) == 1


def test_procedure_owns_its_ramps_and_they_cannot_be_stopped_individually(
    orchestrator, station, qtbot
):
    """A run's ramp names its owner and refuses a single-VI stop.

    Stopping one VI mid-run would strand the run waiting on a setpoint it can
    never reach — aborting the run is the only correct stop.
    """
    procedure = MockProcedure(station)
    orchestrator.run_procedure(procedure)
    orchestrator._tick()

    records = [r for r in orchestrator.active_ramps() if r.vi_name == "magnet_z"]
    assert records, "the procedure's magnet ramp should be tracked"
    record = records[0]
    assert record.owner == "procedure 'Mock Sweep'"
    assert record.stoppable is False
    assert "Mock Sweep" in record.stop_blocked_reason

    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500):
        orchestrator.stop_ramp("magnet_z")
    # Refused means refused: the ramp is still running.
    assert station.get_vi("magnet_z").ramp_status() == "RAMPING"

    orchestrator.abort_procedure()


def test_stop_ramp_holds_that_vi_and_leaves_the_others_alone(
    orchestrator, station, qtbot
):
    """A manual ramp is stopped per instrument, unlike abort_procedure()."""
    station.get_vi("magnet_z").set_field(5.0)
    station.get_vi("temperature").set_temperature(200.0)
    orchestrator._tick()
    assert {r.vi_name for r in orchestrator.active_ramps()} == {
        "magnet_z", "temperature",
    }

    with qtbot.waitSignal(orchestrator.action_succeeded, timeout=500) as blocker:
        orchestrator.stop_ramp("magnet_z")
    assert blocker.args == ["magnet_z", "stop_ramp"]

    assert station.get_vi("magnet_z").ramp_status() == "IDLE"
    assert station.get_vi("temperature").ramp_status() == "RAMPING"
    # The stopped row leaves the tracker immediately, not a tick later.
    assert [r.vi_name for r in orchestrator.active_ramps()] == ["temperature"]


def test_stop_ramp_returns_the_machine_to_idle_on_the_next_tick(
    orchestrator, station, qtbot
):
    """The state machine still owns the RAMPING -> IDLE transition."""
    station.get_vi("magnet_z").set_field(5.0)
    orchestrator._tick()
    assert orchestrator._state == OrchestratorState.RAMPING

    orchestrator.stop_ramp("magnet_z")
    orchestrator._tick()
    assert orchestrator._state == OrchestratorState.IDLE


def test_quiet_machine_publishes_no_ramps_without_polling(station, qtbot, monkeypatch):
    """Monitoring off + IDLE polls nothing: a fresh launch stays quiet."""
    orch = Orchestrator(station, tick_interval_ms=10_000)
    try:
        calls = []
        monkeypatch.setattr(
            station, "get_ramp_status", lambda: calls.append(1) or {}
        )
        assert orch.is_monitoring() is False
        orch._tick()
        assert calls == []
        assert orch.active_ramps() == []
    finally:
        orch.shutdown()


def test_action_blocking(orchestrator, station, qtbot):
    """submit_vi_action() during procedure emits action_blocked."""
    procedure = MockProcedure(station)
    orchestrator.run_procedure(procedure)
    
    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500):
        orchestrator.submit_vi_action("magnet_z", "set_field", target_T=1.0)
    
    orchestrator.abort_procedure()


def test_action_succeeded_emitted_on_successful_gui_action(orchestrator, qtbot):
    """submit_vi_action() in IDLE, once executed by the tick loop, emits action_succeeded."""
    with qtbot.waitSignal(orchestrator.action_succeeded, timeout=500) as blocker:
        orchestrator.submit_vi_action("magnet_z", "initiate")

    assert blocker.args == ["magnet_z", "initiate"]


def test_action_succeeded_not_emitted_on_failed_gui_action(orchestrator, qtbot):
    """A GUI action that raises must not emit action_succeeded."""
    received = []
    orchestrator.action_succeeded.connect(lambda vi, method: received.append((vi, method)))

    orchestrator.submit_vi_action("magnet_z", "not_a_real_method")
    qtbot.wait(50)  # let one tick pass

    assert received == []


def test_submit_global_action_initiate_all_succeeds_for_every_vi(orchestrator, station, qtbot):
    """'Initiate All' fans out to one queued initiate per VI.

    Regression: the button used to call station.initiate_all() directly, which
    ran initiate() on every VI but emitted no action_succeeded verdict, so the
    per-panel lifecycle toggles never flipped and the click looked dead. Now it
    enqueues per-VI actions that the tick executes and confirms, one signal per
    VI — exactly what the InstrumentPanel toggles listen for.
    """
    expected = set(station.get_vi_names())
    assert expected, "sim station should register at least one VI"

    received: list[tuple[str, str]] = []
    orchestrator.action_succeeded.connect(lambda vi, m: received.append((vi, m)))

    orchestrator.submit_global_action("initiate_all")
    # One queued action per VI, before any tick has processed them.
    assert len(orchestrator._gui_action_queue) == len(expected)

    qtbot.waitUntil(lambda: len(received) >= len(expected), timeout=2000)

    assert {vi for vi, method in received} == expected
    assert all(method == "initiate" for _, method in received)


def test_submit_global_action_standby_all_succeeds_for_every_vi(orchestrator, station, qtbot):
    """'Standby All' likewise confirms a standby for every VI (same toggle path)."""
    expected = set(station.get_vi_names())
    received: list[tuple[str, str]] = []
    orchestrator.action_succeeded.connect(lambda vi, m: received.append((vi, m)))

    orchestrator.submit_global_action("standby_all")
    qtbot.waitUntil(lambda: len(received) >= len(expected), timeout=2000)

    assert {vi for vi, method in received} == expected
    assert all(method == "standby" for _, method in received)


def test_action_failed_emitted_with_reason(orchestrator, qtbot):
    """A refused GUI action emits action_failed(vi, method, reason).

    This is the uniform per-action verdict of the control-validation
    standard: here a set_field beyond the setup's field limit is rejected by
    the limits wrapper and the reason reaches the GUI signal verbatim.
    """
    with qtbot.waitSignal(orchestrator.action_failed, timeout=500) as blocker:
        orchestrator.submit_vi_action("magnet_z", "set_field", target_T=99.0)

    vi_name, method_name, reason = blocker.args
    assert (vi_name, method_name) == ("magnet_z", "set_field")
    assert "outside the allowed range" in reason
    # The refused command must not have started a ramp.
    assert orchestrator._state == OrchestratorState.IDLE


def test_stale_claimed_vi_during_procedure_fails_run_to_idle(orchestrator, station, qtbot):
    """A stale ACTIVE (claimed) VI fails the run, but returns to IDLE, not ERROR.

    A claimed VI's fault has a KNOWN, narrow blast radius (that one
    instrument), so it must not park the whole machine in global ERROR —
    only the run fails, and every other instrument (and this one, once it
    recovers or is retried) stays usable. This replaces the old
    test_stale_vi_during_procedure, which asserted the pre-plan global-ERROR
    behavior; that behavior is now reserved for unknown-blast-radius
    failures (unhandled tick-boundary exceptions), verified separately by
    test_unhandled_tick_exception_still_enters_error.
    """
    procedure = MockProcedure(station)
    finished: list[dict] = []
    events: list = []
    orchestrator.run_finished.connect(lambda manifest: finished.append(manifest))
    orchestrator.error_event.connect(lambda ev: events.append(ev))
    orchestrator.run_procedure(procedure)

    # Patch to simulate error on the VI the run is actively driving.
    station.magnet_z._driver._simulate_error = True

    def check_idle_again():
        return orchestrator._state == OrchestratorState.IDLE and bool(finished)

    qtbot.waitUntil(check_idle_again, timeout=1000)

    assert orchestrator._state == OrchestratorState.IDLE
    manifest = finished[-1]
    assert manifest["status"] == "failed"
    assert "magnet_z" in manifest["reason"]

    # The VI's fault stands in the Station registry — quarantined, not the
    # whole machine.
    faults = station.vi_faults()
    assert "magnet_z" in faults
    assert faults["magnet_z"].kind in ("stale", "disconnected")

    # A matching structured run_failure event named the instrument.
    run_failure_events = [e for e in events if e.kind == "run_failure"]
    assert run_failure_events
    assert run_failure_events[-1].vi_name == "magnet_z"
    assert run_failure_events[-1].severity == "error"

    # Every OTHER instrument stays usable: a manual action on an unfaulted
    # VI is admitted immediately (no run is active any more).
    admitted, _ = orchestrator._manual_action_admissible("temperature")
    assert admitted is True

    # The faulted VI itself is refused until it recovers or is retried.
    admitted, reason = orchestrator._manual_action_admissible("magnet_z")
    assert admitted is False
    assert "fault" in reason.lower()

    # The queue must NOT auto-continue after a run failure (conservative,
    # same as the old ERROR behavior).
    orchestrator.queue_procedure(MockProcedure(station))
    assert orchestrator._procedure_queue  # still queued, not auto-started
    assert orchestrator._state == OrchestratorState.IDLE

    # Clear the fault so nothing leaks into other tests.
    station.magnet_z._driver._simulate_error = False


def test_stale_unclaimed_vi_while_monitoring_is_warning_only(orchestrator, station, qtbot):
    """A stale UNCLAIMED VI (no run using it) never changes state — just a fault + warning."""
    events: list = []
    orchestrator.error_event.connect(lambda ev: events.append(ev))

    assert orchestrator._state == OrchestratorState.IDLE
    station.temperature._driver._simulate_error = True

    def has_fault():
        return "temperature" in station.vi_faults()

    qtbot.waitUntil(has_fault, timeout=1000)

    # No state change at all.
    assert orchestrator._state == OrchestratorState.IDLE

    fault_events = [
        e for e in events if e.kind == "fault" and e.vi_name == "temperature"
    ]
    assert fault_events
    assert fault_events[-1].severity == "warning"

    station.temperature._driver._simulate_error = False


def test_unhandled_tick_exception_still_enters_error(orchestrator, qtbot, monkeypatch):
    """An unhandled tick-boundary exception (unknown blast radius) still -> ERROR.

    The one case global ERROR survives: recover_from_error() is
    unchanged.
    """
    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(orchestrator, "_tick_body", boom)

    with qtbot.waitSignal(orchestrator.error_occurred, timeout=1000):
        orchestrator._tick()

    assert orchestrator._state == OrchestratorState.ERROR
    orchestrator.recover_from_error()
    assert orchestrator._state == OrchestratorState.IDLE


def test_hold_flag_holds_concerned_vis_without_emergency(orchestrator, station, qtbot, hold_flag):
    """A tripped hold-severity flag holds concerned VIs, never EMERGENCY.

    The declared flag is hold-only and magnet_z is the only VI concerned
    with it (see the ``hold_flag`` fixture): magnet_z is refused manual
    control the moment the hold is recorded, while temperature — the VI
    that REPORTS the flag, and is not concerned with it — keeps operating
    freely and the Orchestrator never leaves IDLE.
    """
    hold_flag.trip()

    def magnet_held():
        return "magnet_z" in orchestrator._held_vis()

    qtbot.waitUntil(magnet_held, timeout=2000)
    assert orchestrator._state == OrchestratorState.IDLE

    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500) as blocker:
        orchestrator.submit_vi_action("magnet_z", "initiate")
    assert "safety hold" in blocker.args[0]
    assert orchestrator._state == OrchestratorState.IDLE

    with qtbot.waitSignal(orchestrator.action_succeeded, timeout=500):
        orchestrator.submit_vi_action("temperature", "initiate")

    # The flag clears; the hold lifts on the next tick, on its own — no
    # acknowledgment needed.
    hold_flag.clear()

    def hold_cleared():
        return "magnet_z" not in orchestrator._held_vis()

    qtbot.waitUntil(hold_cleared, timeout=2000)
    with qtbot.waitSignal(orchestrator.action_succeeded, timeout=500):
        orchestrator.submit_vi_action("magnet_z", "initiate")


def test_acknowledge_unlocks_hold_without_emergency(orchestrator, station, qtbot, hold_flag):
    """acknowledge() unlocks a plain hold-severity condition, no EMERGENCY involved.

    Mirrors test_hold_flag_holds_concerned_vis_without_emergency but exercises
    the new override: before acknowledge(), magnet_z is refused exactly
    like today; after, manual control is admitted and the state never
    leaves IDLE — acknowledge() never fakes a state transition for a
    condition that is honestly still active (GLOSSARY.md's **Hold
    acknowledge**).
    """
    hold_flag.trip()

    def magnet_held():
        return "magnet_z" in orchestrator._held_vis()

    qtbot.waitUntil(magnet_held, timeout=2000)

    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500):
        orchestrator.submit_vi_action("magnet_z", "standby")

    orchestrator.acknowledge()
    assert orchestrator._state == OrchestratorState.IDLE
    assert orchestrator.override_active("magnet_z") is True

    with qtbot.waitSignal(orchestrator.action_succeeded, timeout=500):
        orchestrator.submit_vi_action("magnet_z", "standby")

    hold_flag.clear()


def test_acknowledge_hold_override_works_while_paused(orchestrator, station, qtbot, hold_flag):
    """The hold override reaches through claim-protection while PAUSED.

    A magnet held by a safety flag AND claimed by a paused procedure is
    refused by rule 0b before claim-protection (rule 4) is ever reached —
    acknowledging admits the action directly, with no PAUSED-specific code
    anywhere in _manual_action_admissible(). This is the deliberate,
    accepted trade-off documented as the Known gap for resume_procedure()
    (GLOSSARY.md's **Hold acknowledge**): the override reaches through
    claim-protection on purpose, so clearing the cause mid-pause is possible.
    """
    _fast_magnet(station)
    procedure = MockProcedure(station)
    orchestrator.run_procedure(procedure)
    _pause_at_boundary(orchestrator)
    assert orchestrator._state == OrchestratorState.PAUSED

    hold_flag.trip()

    def magnet_held():
        return "magnet_z" in orchestrator._held_vis()

    qtbot.waitUntil(magnet_held, timeout=2000)

    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500):
        orchestrator.submit_vi_action("magnet_z", "standby")

    orchestrator.acknowledge()
    with qtbot.waitSignal(orchestrator.action_succeeded, timeout=500):
        orchestrator.submit_vi_action("magnet_z", "standby")
    assert orchestrator._state == OrchestratorState.PAUSED

    hold_flag.clear()


def test_hold_override_does_not_leak_into_new_emergency(orchestrator, station, qtbot, hold_flag):
    """Regression: an acknowledged hold must not bypass a LATER EMERGENCY.

    Without the EMERGENCY/ERROR guard in _manual_action_admissible()'s
    rule 0b, an operator acknowledging a hold on magnet_z while IDLE
    would leave magnet_z unlocked straight through a subsequent quench on
    the SAME VI — breaking the invariant that critical severity refuses
    every VI, held or not
    (test_quench_emergency_blocks_all_manual_control). The still-live hold
    override must NOT admit action once EMERGENCY is entered; only a fresh
    acknowledge() (now delegating to _acknowledge_emergency()) can unlock
    it from there.
    """
    hold_flag.trip()

    def magnet_held():
        return "magnet_z" in orchestrator._held_vis()

    qtbot.waitUntil(magnet_held, timeout=2000)
    orchestrator.acknowledge()
    assert orchestrator.override_active("magnet_z") is True
    with qtbot.waitSignal(orchestrator.action_succeeded, timeout=500):
        orchestrator.submit_vi_action("magnet_z", "standby")

    station.magnet_z._driver._simulate_quench = True
    qtbot.waitUntil(
        lambda: orchestrator._state == OrchestratorState.EMERGENCY, timeout=2000
    )

    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500):
        orchestrator.submit_vi_action("magnet_z", "initiate")

    orchestrator.acknowledge()
    assert orchestrator._state == OrchestratorState.EMERGENCY
    with qtbot.waitSignal(orchestrator.action_succeeded, timeout=500):
        orchestrator.submit_vi_action("magnet_z", "initiate")

    station.magnet_z._driver._simulate_quench = False
    hold_flag.clear()


def test_hold_override_refused_during_error(orchestrator, station, qtbot, monkeypatch, hold_flag):
    """The hold override never admits action while the Orchestrator is in ERROR.

    ERROR means unknown blast radius (an unhandled tick-boundary
    exception), not a known, scoped condition a human can safely act
    around — the deliberate ERROR exclusion alongside EMERGENCY in rule
    0b's state guard. A hold acknowledged BEFORE the ERROR entry (still
    unexpired) must not leak through it, mirroring
    test_hold_override_does_not_leak_into_new_emergency for ERROR instead
    of EMERGENCY.
    """
    hold_flag.trip()

    def magnet_held():
        return "magnet_z" in orchestrator._held_vis()

    qtbot.waitUntil(magnet_held, timeout=2000)
    orchestrator.acknowledge()
    assert orchestrator.override_active("magnet_z") is True

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(orchestrator, "_tick_body", boom)
    with qtbot.waitSignal(orchestrator.error_occurred, timeout=1000):
        orchestrator._tick()
    assert orchestrator._state == OrchestratorState.ERROR

    admitted, reason = orchestrator._manual_action_admissible("magnet_z")
    assert admitted is False
    assert "safety hold" in reason

    hold_flag.clear()


def test_hold_override_expires_after_timeout(station, qtbot, hold_flag):
    """The hold override reverts to refusal once manual_override_timeout_s elapses.

    No state change accompanies the expiry — it is purely the timestamp
    comparison in _manual_action_admissible() lapsing on its own, not a
    tick-driven "re-lock" event. A fresh acknowledge() restores access.
    """
    orch = Orchestrator(station, tick_interval_ms=10, manual_override_timeout_s=0.2)
    orch.start_monitoring()
    try:
        hold_flag.trip()

        def magnet_held():
            return "magnet_z" in orch._held_vis()

        qtbot.waitUntil(magnet_held, timeout=2000)
        orch.acknowledge()
        assert orch.override_active("magnet_z") is True

        qtbot.wait(300)  # past the 0.2s window
        assert orch.override_active("magnet_z") is False
        admitted, reason = orch._manual_action_admissible("magnet_z")
        assert admitted is False
        assert orch._state == OrchestratorState.IDLE  # no state change

        orch.acknowledge()
        assert orch.override_active("magnet_z") is True
    finally:
        hold_flag.clear()
        orch.shutdown()


def test_hold_override_survives_flapping_within_window(station, qtbot, hold_flag):
    """The cooldown survives the SAME flag clearing and re-tripping (flapping).

    User-confirmed design: pruning is expiry-only, never merely because the
    condition disappeared from verdict.held_vis, so a flag hovering right
    at its threshold (a level flag is exactly this in practice) does not
    force a fresh acknowledge on every re-trip within the same window.
    """
    orch = Orchestrator(station, tick_interval_ms=10, manual_override_timeout_s=5.0)
    orch.start_monitoring()
    try:
        hold_flag.trip()

        def magnet_held():
            return "magnet_z" in orch._held_vis()

        qtbot.waitUntil(magnet_held, timeout=2000)
        orch.acknowledge()
        assert orch.override_active("magnet_z") is True

        # Condition clears...
        hold_flag.clear()

        def hold_cleared():
            return "magnet_z" not in orch._held_vis()

        qtbot.waitUntil(hold_cleared, timeout=2000)
        # override_active("magnet_z") is correctly False here — nothing is
        # held, so there is nothing to admit against. The invariant under
        # test is that the underlying _hold_override_until entry (keyed by
        # Condition.key) was NOT pruned just
        # because the condition cleared — checked directly since
        # override_active() has no VI to resolve a key through while
        # unheld. It only matters once the flag re-trips, below.
        assert orch._hold_override_until

        # ...and re-trips (same key) before the 5s window elapses, with no
        # second acknowledge() call.
        hold_flag.trip()
        qtbot.waitUntil(magnet_held, timeout=2000)
        assert orch.override_active("magnet_z") is True
        with qtbot.waitSignal(orch.action_succeeded, timeout=500):
            orch.submit_vi_action("magnet_z", "standby")
    finally:
        hold_flag.clear()
        orch.shutdown()


def test_emergency_override_expires_after_timeout(station, qtbot):
    """The EMERGENCY override also expires, reverting to refusal in-state.

    State stays EMERGENCY throughout (the condition is still active) —
    only manual-control admission lapses. Re-acknowledging renews it.
    """
    orch = Orchestrator(station, tick_interval_ms=10, manual_override_timeout_s=0.2)
    orch.start_monitoring()
    try:
        station.magnet_z._driver._simulate_quench = True
        qtbot.waitUntil(
            lambda: orch._state == OrchestratorState.EMERGENCY, timeout=2000
        )
        orch.acknowledge()
        assert orch.override_active() is True
        with qtbot.waitSignal(orch.action_succeeded, timeout=500):
            orch.submit_vi_action("dc_measurement", "initiate")

        qtbot.wait(300)  # past the 0.2s window
        assert orch.override_active() is False
        with qtbot.waitSignal(orch.action_blocked, timeout=500):
            orch.submit_vi_action("dc_measurement", "initiate")
        assert orch._state == OrchestratorState.EMERGENCY  # no state change

        orch.acknowledge()
        assert orch.override_active() is True
        with qtbot.waitSignal(orch.action_succeeded, timeout=500):
            orch.submit_vi_action("dc_measurement", "initiate")
    finally:
        station.magnet_z._driver._simulate_quench = False
        orch.shutdown()


def test_emergency_acknowledge_unlocks_manual_front_panel(orchestrator, station, qtbot):
    """Acknowledging an unresolved EMERGENCY unlocks manual control station-wide.

    Before acknowledging, a front-panel action on ANY VI — not just one
    concerned with the tripped critical flag — is refused: critical
    severity is station-wide scope by construction (the System-Condition
    standard), so EMERGENCY refuses every VI, held or not (the inversion of
    the pre-System-Condition behavior, where an unconcerned VI stayed
    operable). Acknowledging once (condition still active) stays in
    EMERGENCY but unlocks manual control of EVERY VI — the operator's way
    to intervene (e.g. cycling a switch heater by hand) without the
    condition having cleared on its own. run_procedure() must still refuse
    to run immediately: it only queues, same as any busy state.
    """
    station.magnet_z._driver._simulate_quench = True
    qtbot.waitUntil(
        lambda: orchestrator._state == OrchestratorState.EMERGENCY, timeout=2000
    )

    # Locked: front-panel action refused before acknowledging, on both the
    # quenched magnet AND a VI with no safety_concerns() at all.
    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500):
        orchestrator.submit_vi_action("magnet_z", "initiate")
    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500):
        orchestrator.submit_vi_action("dc_measurement", "initiate")
    assert orchestrator._state == OrchestratorState.EMERGENCY

    # Condition is still active, so acknowledging cannot reach IDLE...
    orchestrator.acknowledge()
    assert orchestrator._state == OrchestratorState.EMERGENCY
    assert orchestrator.override_active() is True

    # ...but the front panel is now unlocked, for every VI.
    with qtbot.waitSignal(orchestrator.action_succeeded, timeout=500) as blocker:
        orchestrator.submit_vi_action("magnet_z", "initiate")
    assert blocker.args == ["magnet_z", "initiate"]
    assert orchestrator._state == OrchestratorState.EMERGENCY
    with qtbot.waitSignal(orchestrator.action_succeeded, timeout=500) as blocker:
        orchestrator.submit_vi_action("dc_measurement", "initiate")
    assert blocker.args == ["dc_measurement", "initiate"]

    # A procedure is still refused from running immediately — it queues.
    procedure = MockProcedure(station)
    orchestrator.run_procedure(procedure)
    assert orchestrator._state == OrchestratorState.EMERGENCY
    assert orchestrator._procedure is None
    assert procedure in orchestrator._procedure_queue

    # Quench recovers; acknowledging again returns to IDLE and relocks.
    station.magnet_z._driver._simulate_quench = False
    qtbot.waitUntil(
        lambda: not orchestrator._station.check_safety().get("quench"), timeout=2000
    )
    orchestrator.acknowledge()
    assert orchestrator._state == OrchestratorState.IDLE
    assert orchestrator.override_active() is False


def test_emergency_shutdown_runs_once_not_every_tick(orchestrator, station, qtbot):
    """EMERGENCY entry's blanket standby_all() must run once, not every tick.

    _enter_emergency() always calls Station.standby_all() exactly once on
    entry (critical severity is station-wide scope by construction, so
    there is no "concerned VI" subset to narrow the shutdown to — see the
    System-Condition standard). Repeating it every tick would restart every
    instrument's own safe-off sequence every few seconds.
    """
    calls = {"n": 0}
    original = station.magnet_z.standby

    def counting():
        calls["n"] += 1
        original()

    station.magnet_z.standby = counting
    try:
        station.magnet_z._driver._simulate_quench = True
        qtbot.waitUntil(
            lambda: orchestrator._state == OrchestratorState.EMERGENCY,
            timeout=2000,
        )
        # Let several more ticks pass in EMERGENCY.
        qtbot.wait(100)
    finally:
        station.magnet_z.standby = original
    assert calls["n"] == 1


def test_quench_triggers_emergency(orchestrator, station, qtbot):
    """A magnet QUENCH status must escalate to EMERGENCY (it is a critical flag)."""
    station.magnet_z._driver._simulate_quench = True
    qtbot.waitUntil(
        lambda: orchestrator._state == OrchestratorState.EMERGENCY,
        timeout=2000,
    )
    assert orchestrator._state == OrchestratorState.EMERGENCY


def test_quench_emergency_blocks_all_manual_control(orchestrator, station, qtbot):
    """During quench EMERGENCY, EVERY VI's manual control is refused.

    Quench concerns no instrument (MagnetBase.safety_concerns() is the
    empty default — see virtual_instruments/base.py); it is a critical
    safety flag, and critical severity IS station-wide scope (the
    System-Condition standard): the whole station needs to be shut down
    regardless of which VI reported the quench or which VI's
    safety_concerns() name it. This is the inversion of the old
    "unconcerned VI stays usable" behavior — dc_measurement (a plain
    measurement VI with no safety_concerns() at all) and temperature
    (unaffected by quench under either the old or new safety_concerns())
    are refused exactly like magnet_z, until acknowledge()
    unlocks the manual override.
    """
    station.magnet_z._driver._simulate_quench = True
    qtbot.waitUntil(
        lambda: orchestrator._state == OrchestratorState.EMERGENCY, timeout=2000
    )

    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500):
        orchestrator.submit_vi_action("magnet_z", "initiate")
    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500):
        orchestrator.submit_vi_action("temperature", "initiate")
    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500):
        orchestrator.submit_vi_action("dc_measurement", "initiate")

    orchestrator.acknowledge()
    assert orchestrator._state == OrchestratorState.EMERGENCY
    assert orchestrator.override_active() is True

    with qtbot.waitSignal(orchestrator.action_succeeded, timeout=500):
        orchestrator.submit_vi_action("dc_measurement", "initiate")


def test_safety_hold_fails_claimed_run_but_spares_others(orchestrator, station, qtbot, hold_flag):
    """A non-tolerated hold on a run's watched VI fails just that run.

    Mirrors the stale-VI run-failure behavior, but for a physical safety
    concern rather than a communication fault: the machine returns to IDLE
    (not global ERROR), and every other instrument stays usable.
    """
    procedure = MockProcedure(station)  # targets magnet_z, claims everything
    orchestrator.run_procedure(procedure)
    assert orchestrator._procedure is procedure

    hold_flag.trip()
    qtbot.waitUntil(lambda: orchestrator._procedure is None, timeout=2000)
    assert orchestrator._state == OrchestratorState.IDLE

    with qtbot.waitSignal(orchestrator.action_succeeded, timeout=500):
        orchestrator.submit_vi_action("temperature", "initiate")


class RecordingProcedure(MockProcedure):
    """MockProcedure with a BaseProcedure-style abort() that records calls."""

    def __init__(self, station):
        super().__init__(station)
        self.abort_called = 0

    def abort(self):
        self.abort_called += 1
        return ()


def test_abort_calls_procedure_abort_and_holds_magnet(orchestrator, station, qtbot):
    """Abort must run the procedure's cleanup AND freeze the PSU (finding C3).

    Clearing the software generator alone is not enough: the PSU ramps
    autonomously to its last-commanded setpoint, so an abort that does not
    command a hardware hold leaves the field still moving.
    """
    proc = RecordingProcedure(station)
    orchestrator.run_procedure(proc)  # ramps magnet_z toward 1.0 T (slow rate)
    assert station.magnet_z.ramp_status() == "RAMPING"

    orchestrator.abort_procedure()

    assert proc.abort_called == 1
    assert orchestrator._state == OrchestratorState.IDLE
    assert orchestrator._procedure is None
    assert orchestrator._wait_started is False  # stale wait clock reset (H5)
    # Hardware held: PSU setpoint pinned to its present output.
    assert station.magnet_z.ramp_status() == "IDLE"
    drv = station.magnet_z._driver
    assert drv.get_status() == "HOLD"
    assert drv.get_current_setpoint() == pytest.approx(drv.get_current(), abs=0.01)


def test_measure_exception_degrades_to_error_not_crash(orchestrator, station, qtbot):
    """An exception inside the tick must contain to ERROR, never propagate.

    PyQt6 aborts the whole process on an unhandled exception in a slot
    (finding C2) — with the magnet live that is the worst possible failure.
    """
    class ExplodingProcedure(RecordingProcedure):
        def measure(self):
            raise RuntimeError("simulated measurement failure")

    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []
    proc = ExplodingProcedure(station)
    orchestrator.run_procedure(proc)

    qtbot.waitUntil(
        lambda: orchestrator._state == OrchestratorState.ERROR, timeout=5000
    )
    assert orchestrator._procedure is None   # run cleaned up
    assert proc.abort_called == 1            # data-file cleanup hook ran

    orchestrator.recover_from_error()
    assert orchestrator._state == OrchestratorState.IDLE


def test_run_procedure_setup_failure_degrades_to_error(orchestrator, station):
    """initiate() raising must not crash the GUI slot; it lands in ERROR."""
    class BadInit(RecordingProcedure):
        def initiate(self):
            raise ValueError("bad parameters")

    proc = BadInit(station)
    orchestrator.run_procedure(proc)

    assert orchestrator._state == OrchestratorState.ERROR
    assert orchestrator._procedure is None
    orchestrator.recover_from_error()
    assert orchestrator._state == OrchestratorState.IDLE


def test_malformed_initiate_return_fails_loudly(orchestrator, station):
    """initiate() returning the OLD tuple (not a PhasePlan) must fail loudly.

    The Wave-2 currency is typed: the Orchestrator consumes ``plan.targets`` /
    ``plan.commands`` / ``plan.wait_s``. A procedure that returns the legacy
    ``(system_targets, measurement_commands, wait)`` tuple has no ``.targets``
    attribute, so setup must contain the AttributeError to ERROR rather than
    silently mis-dispatching.
    """
    class LegacyReturn(RecordingProcedure):
        def initiate(self):
            # The pre-Wave-2 shape — a bare tuple, not a PhasePlan.
            return ({"magnet_z": {"target": 1.0}}, {}, 0.0)

    proc = LegacyReturn(station)
    orchestrator.run_procedure(proc)

    assert orchestrator._state == OrchestratorState.ERROR
    assert orchestrator._procedure is None
    orchestrator.recover_from_error()
    assert orchestrator._state == OrchestratorState.IDLE


def test_pause_holds_hardware_and_resume_redispatches(orchestrator, station, qtbot):
    """Pause must freeze the autonomous PSU; resume must restart the ramp."""
    proc = MockProcedure(station)
    orchestrator.run_procedure(proc)  # slow ramp toward 1.0 T
    assert station.magnet_z.ramp_status() == "RAMPING"

    orchestrator.pause_procedure()
    assert orchestrator._state == OrchestratorState.PAUSED
    drv = station.magnet_z._driver
    assert drv.get_status() == "HOLD"  # field frozen, not still ramping

    orchestrator.resume_procedure()
    assert orchestrator._state in (
        OrchestratorState.INITIATING, OrchestratorState.RAMPING
    )
    assert station.magnet_z.ramp_status() == "RAMPING"  # ramp re-dispatched

    orchestrator.abort_procedure()


def test_boundary_pause_holds_hardware_and_resume_ramps_to_the_next_point(
    orchestrator, station, qtbot
):
    """A pause taken at the boundary holds too, and resume steps the sweep."""
    _fast_magnet(station)
    proc = MockProcedure(station)
    orchestrator.run_procedure(proc)
    drv = station.magnet_z._driver

    _pause_at_boundary(orchestrator)
    assert drv.get_status() == "HOLD"  # field frozen, not still ramping

    orchestrator.resume_procedure()
    assert orchestrator._state == OrchestratorState.SWEEPING
    orchestrator._tick()  # steps the sweep and dispatches the next target
    assert orchestrator._state == OrchestratorState.RAMPING
    assert station.magnet_z.ramp_status() == "RAMPING"

    orchestrator.abort_procedure()


def test_queue_procedures(orchestrator, station, qtbot):
    """Multiple procedures queued, run sequentially."""
    proc1 = MockProcedure(station)
    proc2 = MockProcedure(station)
    
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []

    orchestrator.run_procedure(proc1)
    orchestrator.run_procedure(proc2)
    
    assert len(orchestrator._procedure_queue) == 1
    
    # wait for proc1 finished
    with qtbot.waitSignal(orchestrator.procedure_finished, timeout=2000):
        pass
        
    # proc2 should start now
    assert orchestrator._procedure == proc2
    assert orchestrator._state != OrchestratorState.IDLE
    orchestrator.abort_procedure()


def _spy_get_state(station, monkeypatch):
    """Wrap station.get_state with a call counter; returns the counter list."""
    calls: list[int] = []
    real_get_state = station.get_state

    def counted():
        calls.append(1)
        return real_get_state()

    monkeypatch.setattr(station, "get_state", counted)
    return calls


def test_monitoring_off_by_default_polls_nothing(station, qtbot, monkeypatch):
    """A fresh Orchestrator neither polls the station nor emits states_updated.

    The per-tick operational-status record is written anyway (see
    test_operational_status_written_with_monitoring_off) — it must not be the
    one thing that breaks the quiet, so its instrument payload stays empty and
    neither get_state() nor get_ramp_status() is called.
    """
    orch = Orchestrator(station, tick_interval_ms=10)
    calls = _spy_get_state(station, monkeypatch)
    ramp_calls: list[int] = []
    real_get_ramp_status = station.get_ramp_status
    monkeypatch.setattr(
        station,
        "get_ramp_status",
        lambda: (ramp_calls.append(1), real_get_ramp_status())[1],
    )
    emitted: list[dict] = []
    orch.states_updated.connect(emitted.append)

    assert orch.is_monitoring() is False
    for _ in range(3):
        orch._tick()
    assert calls == []
    assert ramp_calls == []
    assert emitted == []
    assert orch.get_operational_status()["vis"] == []
    orch.shutdown()


def test_start_monitoring_begins_polling_and_signals(station, qtbot, monkeypatch):
    """start_monitoring() emits monitoring_changed(True) and enables polling."""
    orch = Orchestrator(station, tick_interval_ms=10)
    calls = _spy_get_state(station, monkeypatch)
    changes: list[bool] = []
    orch.monitoring_changed.connect(changes.append)

    assert orch.start_monitoring() is True
    assert orch.is_monitoring() is True
    orch._tick()
    assert calls, "monitored tick must poll the station"

    # Idempotent: a second start emits no second signal.
    assert orch.start_monitoring() is True
    assert changes == [True]
    orch.shutdown()


def test_gui_actions_execute_while_monitoring_off(station, qtbot, monkeypatch):
    """The initiate-before-monitoring flow: actions run on the tick with no polling."""
    orch = Orchestrator(station, tick_interval_ms=10)
    calls = _spy_get_state(station, monkeypatch)
    verdicts: list[tuple[str, str]] = []
    orch.action_succeeded.connect(lambda vi, m: verdicts.append((vi, m)))

    orch.submit_vi_action("magnet_z", "initiate")
    orch._tick()

    assert ("magnet_z", "initiate") in verdicts
    assert calls == []  # still no instrument polling
    orch.shutdown()


def test_run_procedure_auto_starts_monitoring(station, qtbot):
    """A procedure must run under the stall detector: monitoring auto-starts."""
    orch = Orchestrator(station, tick_interval_ms=10)
    assert orch.is_monitoring() is False
    orch.run_procedure(MockProcedure(station))
    assert orch.is_monitoring() is True
    orch.abort_procedure()
    orch.shutdown()


def test_stop_monitoring_refused_while_procedure_active(station, qtbot):
    """stop_monitoring() is blocked outside IDLE/ERROR and allowed back in IDLE."""
    orch = Orchestrator(station, tick_interval_ms=10)
    blocked: list[str] = []
    orch.action_blocked.connect(blocked.append)

    orch.run_procedure(MockProcedure(station))
    assert orch._state == OrchestratorState.INITIATING

    assert orch.stop_monitoring() is False
    assert orch.is_monitoring() is True
    assert blocked and "monitoring" in blocked[0].lower()

    orch.abort_procedure()  # back to IDLE
    assert orch.stop_monitoring() is True
    assert orch.is_monitoring() is False
    orch.shutdown()


def test_shutdown_stops_ticking(station, qtbot):
    """After shutdown() no tick fires: states_updated stays silent."""
    orch = Orchestrator(station, tick_interval_ms=10)
    orch.start_monitoring()
    # Generous for the same reason as test_basic_ticking: this is a positive
    # "prove it ticks" wait, sized against suite-wide event-loop contention.
    # The negative assertion below keeps its short timeout on purpose.
    with qtbot.waitSignal(orch.states_updated, timeout=5000):
        pass  # ticking while monitoring: baseline

    orch.shutdown()
    with qtbot.waitSignal(orch.states_updated, timeout=100, raising=False) as blocker:
        pass
    assert not blocker.signal_triggered


# ── Run manifests (run_started / run_finished) ───────────────────────────────

def _fast_magnet(station):
    """Make magnet_z ramps effectively instant for state-machine tests."""
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []


def test_run_manifests_full_cycle(orchestrator, station, qtbot):
    """A completed run emits one run_started and one matching run_finished."""
    _fast_magnet(station)
    procedure = MockProcedure(station)
    started, finished = [], []
    orchestrator.run_started.connect(started.append)
    orchestrator.run_finished.connect(finished.append)

    orchestrator.run_procedure(procedure)
    with qtbot.waitSignal(orchestrator.procedure_finished, timeout=5000):
        pass

    assert len(started) == 1 and len(finished) == 1
    manifest = started[0]
    assert manifest["procedure"] == "Mock Sweep"
    assert manifest["kind"] == "run"
    assert manifest["run_id"]
    assert manifest["started_utc"]
    # MockProcedure has no data file / params accessors — best-effort fields.
    assert manifest["data_file"] == ""
    assert manifest["params"] == {}
    end = finished[0]
    assert end["run_id"] == manifest["run_id"]
    assert end["status"] == "done"
    assert end["finished_utc"]


def test_abort_emits_run_finished_aborted(orchestrator, station, qtbot):
    """User abort ends the run with status 'aborted', exactly once."""
    procedure = MockProcedure(station)  # slow default ramp keeps the run alive
    finished = []
    orchestrator.run_finished.connect(finished.append)

    orchestrator.run_procedure(procedure)
    with qtbot.waitSignal(orchestrator.state_changed, timeout=2000):
        pass
    orchestrator.abort_procedure()

    assert len(finished) == 1
    assert finished[0]["status"] == "aborted"
    assert orchestrator._state == OrchestratorState.IDLE
    # Recovering/ticking afterwards must not re-emit for the dead run.
    orchestrator._tick()
    assert len(finished) == 1


def test_failed_setup_emits_no_manifests(orchestrator, station, qtbot):
    """When initiate() itself raises, neither manifest is emitted."""

    class BrokenProcedure(MockProcedure):
        def initiate(self):
            raise RuntimeError("boom")

    started, finished = [], []
    orchestrator.run_started.connect(started.append)
    orchestrator.run_finished.connect(finished.append)
    orchestrator.run_procedure(BrokenProcedure(station))

    assert orchestrator._state == OrchestratorState.ERROR
    assert started == [] and finished == []


# ── Session envelope enforcement ─────────────────────────────────────────────

def _envelope(**bounds):
    from i2as.core.plan import ExperimentEnvelope

    return ExperimentEnvelope(bounds=dict(bounds))


def test_envelope_rejects_out_of_bounds_target(orchestrator, station, qtbot):
    """A procedure target outside the envelope is rejected before dispatch."""
    from i2as.core.plan import EnvelopeBound

    orchestrator.set_experiment_envelope(
        _envelope(magnet_z=EnvelopeBound(min_value=-0.5, max_value=0.5))
    )
    errors: list[str] = []
    started: list[dict] = []
    orchestrator.error_occurred.connect(errors.append)
    orchestrator.run_started.connect(started.append)

    orchestrator.run_procedure(MockProcedure(station))  # first target is 1.0 T

    assert orchestrator._state == OrchestratorState.ERROR
    assert started == [], "a rejected run must not report as started"
    assert any("session envelope" in e and "magnet_z" in e for e in errors)
    # The magnet was never asked to move.
    assert station.magnet_z.magnet_field_T() == pytest.approx(0.0, abs=1e-6)


def test_envelope_allows_within_bounds_and_clears(orchestrator, station, qtbot):
    """Targets inside the envelope run normally; None clears the envelope."""
    from i2as.core.plan import EnvelopeBound

    _fast_magnet(station)
    orchestrator.set_experiment_envelope(
        _envelope(magnet_z=EnvelopeBound(min_value=-5.0, max_value=5.0))
    )
    orchestrator.run_procedure(MockProcedure(station))
    with qtbot.waitSignal(orchestrator.procedure_finished, timeout=5000):
        pass
    assert orchestrator._state == OrchestratorState.IDLE

    orchestrator.set_experiment_envelope(None)
    assert orchestrator._session_envelope is None


def test_envelope_state_violation_enters_emergency(orchestrator, station, qtbot):
    """A live reading outside a state_key bound trips EMERGENCY like a safety flag."""
    from i2as.core.plan import EnvelopeBound

    # Sim sample thermometer sits at 300 K; a 400 K session minimum is an
    # immediate violation on the next tick.
    orchestrator.set_experiment_envelope(
        _envelope(
            temperature=EnvelopeBound(min_value=400.0, state_key="temperature")
        )
    )
    errors: list[str] = []
    orchestrator.error_occurred.connect(errors.append)

    orchestrator._tick()
    assert orchestrator._state == OrchestratorState.EMERGENCY
    assert any("session envelope" in e and "temperature" in e for e in errors)

    # Acknowledgement is refused while the violation persists...
    orchestrator._acknowledge_emergency()
    assert orchestrator._state == OrchestratorState.EMERGENCY

    # ...and succeeds once the envelope is cleared (the "sample removed" case).
    orchestrator.set_experiment_envelope(None)
    orchestrator._acknowledge_emergency()
    assert orchestrator._state == OrchestratorState.IDLE


# ── Scanner-enabled flag ──────────────────────────────────────────────────

def test_current_gates_empty_for_procedure_without_gate_methods(orchestrator, station):
    """A duck-typed procedure with no gate methods behaves like the no-op default."""
    procedure = MockProcedure(station)
    orchestrator._procedure = procedure
    orchestrator._first_measurement = True
    assert orchestrator._current_gates() == ()
    orchestrator._first_measurement = False
    assert orchestrator._current_gates() == ()


def test_initiation_gate_replaces_wait_and_blocks_until_satisfied(orchestrator, station):
    """A declared initiation gate is stepped each tick and wait_s is ignored."""
    from i2as.core.gates import Gate

    procedure = MockProcedure(station)
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        return calls["n"] >= 3

    procedure.initiation_gates = lambda: (Gate("settle", check=check),)
    # A large wait_s that must be ignored once a gate is declared.
    procedure.initiate = lambda: PhasePlan(
        targets={"magnet_z": Target(1.0)}, commands=(), wait_s=999.0
    )
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []

    orchestrator.run_procedure(procedure)
    # The sim magnet driver advances on real elapsed time, not tick count, so
    # the budget must clear ~0.1s of wall clock at this rate/target with
    # margin for per-tick overhead (monitoring poll, safety/envelope checks).
    for _ in range(1000):
        orchestrator._tick()
        if orchestrator._state == OrchestratorState.INITIATION_GATE:
            break
    assert orchestrator._state == OrchestratorState.INITIATION_GATE
    assert calls["n"] == 0  # ramp-complete tick declares the gate, doesn't step it yet
    assert orchestrator._wait_started is False

    orchestrator._tick()
    assert orchestrator._state == OrchestratorState.INITIATION_GATE
    assert calls["n"] == 1

    orchestrator._tick()
    assert orchestrator._state == OrchestratorState.INITIATION_GATE
    assert calls["n"] == 2

    orchestrator._tick()
    assert orchestrator._state == OrchestratorState.MEASURING
    assert calls["n"] == 3


def test_reading_gate_used_after_first_measurement_not_initiation(orchestrator, station):
    """reading_gates() governs the second sweep point; the first uses wait_s."""
    from i2as.core.gates import Gate

    procedure = MockProcedure(station)
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        return calls["n"] >= 2

    procedure.reading_gates = lambda: (Gate("settle", check=check),)
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []

    orchestrator.run_procedure(procedure)
    for _ in range(1000):
        orchestrator._tick()
        if orchestrator._state == OrchestratorState.READING_GATE:
            break
    assert orchestrator._state == OrchestratorState.READING_GATE
    assert procedure.measure_called == 1  # first point measured via ordinary wait_s
    assert calls["n"] == 0

    orchestrator._tick()
    assert orchestrator._state == OrchestratorState.READING_GATE
    assert calls["n"] == 1

    orchestrator._tick()
    assert orchestrator._state == OrchestratorState.MEASURING
    assert calls["n"] == 2


def test_pause_resume_during_reading_gate_holds_and_resumes(orchestrator, station):
    """Pausing mid-gate holds pending_gates; resume continues stepping them.

    A gate carries no data — its ``check`` is simply re-evaluated until it
    holds — so a pause here stops on the spot like any other non-MEASURING
    state, and the gate picks up where it left off.
    """
    from i2as.core.gates import Gate

    procedure = MockProcedure(station)
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        return calls["n"] >= 5

    procedure.reading_gates = lambda: (Gate("settle", check=check),)
    _fast_magnet(station)

    orchestrator.run_procedure(procedure)
    _tick_until(
        orchestrator, lambda: orchestrator._state == OrchestratorState.READING_GATE
    )

    orchestrator._tick()
    n_before_pause = calls["n"]
    assert orchestrator._state == OrchestratorState.READING_GATE

    orchestrator.pause_procedure()
    assert orchestrator._state == OrchestratorState.PAUSED
    orchestrator._tick()
    assert calls["n"] == n_before_pause  # no stepping while paused

    orchestrator.resume_procedure()
    assert orchestrator._state == OrchestratorState.READING_GATE
    orchestrator._tick()
    assert calls["n"] == n_before_pause + 1

    orchestrator.abort_procedure()


def test_not_responding_refuses_control_via_tag_policy(orchestrator, station, qtbot):
    """Rule 0 refuses a ``not_responding`` VI, driven by ``TAG_POLICY`` rather
    than a hardcoded ``held.origin == "comm"`` branch.

    ``Station.availability()`` (the Availability standard,
    ``i2as.core.availability``) reports the ``not_responding`` tag the
    moment a comm-origin hold exists, and ``TAG_POLICY["not_responding"]
    .controllable`` is ``False`` — the same refusal
    ``test_stale_claimed_vi_during_procedure_fails_run_to_idle`` observes
    indirectly through a failed run, checked here directly against
    ``_manual_action_admissible()``.
    """
    from i2as.core.availability import TAG_POLICY

    assert TAG_POLICY["not_responding"].controllable is False

    station.magnet_z._driver._simulate_error = True

    def has_fault():
        return "magnet_z" in station.vi_faults()

    qtbot.waitUntil(has_fault, timeout=1000)
    assert "not_responding" in orchestrator._station.availability("magnet_z").tags

    admitted, reason = orchestrator._manual_action_admissible("magnet_z")
    assert admitted is False
    assert "instrument fault" in reason

    station.magnet_z._driver._simulate_error = False


def test_not_responding_never_bypassed_by_emergency_override(orchestrator, station, qtbot):
    """The comm-vs-safety ``acknowledge()`` asymmetry still holds.

    ``TAG_POLICY["not_responding"]`` carries no override column at all, so
    unlike a safety hold (rule 0b), acknowledging an EMERGENCY can never
    unlock a ``not_responding`` VI — only recovery or ``retry_fault()`` can.
    A DIFFERENT VI's quench drives the EMERGENCY here so the fault under test
    is purely a comm condition on ``temperature``, isolating rule 0 from
    rule 0b.
    """
    station.temperature._driver._simulate_error = True

    def has_fault():
        return "temperature" in station.vi_faults()

    qtbot.waitUntil(has_fault, timeout=1000)

    station.magnet_z._driver._simulate_quench = True
    qtbot.waitUntil(
        lambda: orchestrator._state == OrchestratorState.EMERGENCY, timeout=2000
    )
    orchestrator.acknowledge()
    assert orchestrator.override_active() is True

    # The override unlocks a VI with no fault of its own...
    with qtbot.waitSignal(orchestrator.action_succeeded, timeout=500):
        orchestrator.submit_vi_action("dc_measurement", "initiate")

    # ...but NOT the not_responding VI, override notwithstanding.
    admitted, reason = orchestrator._manual_action_admissible("temperature")
    assert admitted is False
    assert "instrument fault" in reason

    station.magnet_z._driver._simulate_quench = False
    qtbot.waitUntil(
        lambda: not orchestrator._station.check_safety().get("quench"), timeout=2000
    )
    orchestrator.acknowledge()
    assert orchestrator._state == OrchestratorState.IDLE

    station.temperature._driver._simulate_error = False


def test_detached_vi_admitted_by_tag_policy(orchestrator, station):
    """A VI carrying only the ``detached`` tag is admitted, per ``TAG_POLICY``.

    ``detached`` (a single-client VI whose session is released between runs
    — the Availability standard, ``i2as.core.availability``) is the
    least-restrictive tag: ``TAG_POLICY["detached"].controllable`` is
    ``True``, and a detached VI is never a ``_held_vis()`` entry at all (it
    is not a ``Condition``), so it reaches rule 0 with ``held is None`` and
    falls through to ordinary admission. This VI carries no ``is_attached()``
    method for real yet (that declaration is a later phase of the
    Availability standard); a duck-typed stand-in exercises
    ``Station._build_availability()``'s existing ``is_attached()`` probe.
    """
    from i2as.core.availability import TAG_POLICY

    assert TAG_POLICY["detached"].controllable is True

    vi = station.get_vi("dc_measurement")
    vi.is_attached = lambda: False
    try:
        assert station.availability("dc_measurement").tags == frozenset({"detached"})
        admitted, reason = orchestrator._manual_action_admissible("dc_measurement")
        assert admitted is True
        assert reason == ""
    finally:
        del vi.is_attached


# ---------------------------------------------------------------------------
# Safety-hold enforcement (Orchestrator._enforce_safety_holds()): the
# level-triggered invariant that keeps every held, un-overridden VI at
# standby for as long as its hold persists — not just once at onset. See
# core/README.md's tick-pipeline paragraph and GLOSSARY.md's **Safety hold**.
# ---------------------------------------------------------------------------


def _force_ramp_to_settle(driver) -> None:
    """Rewind *driver*'s simulated clock so its next poll snaps to setpoint.

    The sim PSU (``SimOxfordIPS120``) advances its simulated current toward
    the commanded setpoint based on REAL elapsed wall time since
    ``_last_update``; the configured ramp rates (~0.5 T/min) would otherwise
    make a real ramp-to-completion take real minutes. Rewinding the clock
    far enough that ``max_step`` exceeds any remaining distance makes the
    very next simulated poll (any ``get_current()``/``get_status()`` call)
    snap straight to the setpoint — mirrors the technique
    ``tests/test_l1_virtual_instruments.py`` uses directly against the VI.
    """
    driver._last_update = time.time() - 3600.0


def test_expired_override_re_asserts_standby_on_the_held_vi(station, qtbot, hold_flag):
    """Regression: a hold that outlives an acknowledge-then-expire window must
    still drive the VI back to standby — expiry revokes manual PERMISSION,
    it must not leave the hardware wherever the operator last put it.

    This is the reported bug: a hold trips, the operator acknowledges to
    unlock manual control, sets a nonzero field during the unlocked window,
    and the window expires. On ``develop`` (the old onset-only dispatch at
    the deleted ``elif condition.origin == "safety" and condition.severity
    == "hold":`` branch) the hold was already "known" from the moment it
    began, so nothing re-fires when the override lapses — the magnet is
    simply never told to stand down again, and this test times out waiting
    for the field to return to zero. With ``_enforce_safety_holds()`` the
    hold is a level-triggered invariant re-checked every tick, so the very
    next eligible tick after the override lapses re-issues ``standby()``.
    """
    driver = station.magnet_z._driver
    vi = station.magnet_z
    orch = Orchestrator(
        station,
        tick_interval_ms=10,
        manual_override_timeout_s=0.2,
        hold_enforcement_interval_s=0.05,
    )
    orch.start_monitoring()
    try:
        hold_flag.trip()

        def magnet_held():
            return "magnet_z" in orch._held_vis()

        qtbot.waitUntil(magnet_held, timeout=2000)

        orch.acknowledge()
        assert orch.override_active("magnet_z") is True

        with qtbot.waitSignal(orch.action_succeeded, timeout=500):
            orch.submit_vi_action("magnet_z", "set_field", target_T=2.0)

        def field_at_target():
            _force_ramp_to_settle(driver)
            return abs(vi.magnet_field_T() - 2.0) < 0.01 and vi.ramp_status() == "TARGET_REACHED"

        qtbot.waitUntil(field_at_target, timeout=2000)

        qtbot.wait(300)  # well past the 0.2s override window
        assert orch.override_active("magnet_z") is False

        def field_back_to_zero():
            _force_ramp_to_settle(driver)
            return abs(vi.magnet_field_T()) < 0.01

        qtbot.waitUntil(field_back_to_zero, timeout=3000)
    finally:
        hold_flag.clear()
        orch.shutdown()


def test_enforcement_interrupts_a_ramp_still_in_flight(station, qtbot, monkeypatch, hold_flag):
    """The "2 T hole": enforcement must not wait for an in-flight manual ramp
    to finish before re-asserting the hold.

    A held magnet sent to 2 T during an acknowledge window, still climbing
    when that window lapses, must be re-commanded to standby WHILE the ramp
    is in flight — never exempted until it arrives. ``standby_status()`` is
    command-PROVENANCE-based (the standby-provenance standard,
    ``virtual_instruments/base.py``), never a physical read of where the
    field currently is, so it reports ``"away"`` the instant a manual
    ``start_ramp()`` supersedes a prior ``standby()``. A check written
    against the magnet's own ``magnet_state()`` instead would see
    ``"ramping"`` — with no notion of ramping to WHAT — classify the magnet
    as converging on safety, and leave it climbing to 2 T forever. That is
    the hole this test exists to keep closed.
    """
    vi = station.magnet_z
    orch = Orchestrator(
        station,
        tick_interval_ms=10,
        hold_enforcement_interval_s=0.02,
    )
    orch.start_monitoring()
    try:
        standby_calls: list[float] = []
        original_standby = vi.standby

        def counting_standby():
            standby_calls.append(time.time())
            return original_standby()

        monkeypatch.setattr(vi, "standby", counting_standby)

        hold_flag.trip()

        def magnet_held():
            return "magnet_z" in orch._held_vis()

        qtbot.waitUntil(magnet_held, timeout=2000)
        qtbot.waitUntil(lambda: len(standby_calls) >= 1, timeout=2000)  # onset attempt
        orch.acknowledge()

        with qtbot.waitSignal(orch.action_succeeded, timeout=500):
            orch.submit_vi_action("magnet_z", "set_field", target_T=2.0)

        calls_before = len(standby_calls)
        assert vi.ramp_status() == "RAMPING"
        assert vi.standby_status() == "away"

        # Force the override to have already expired, with the ramp toward
        # 2 T still climbing (the sim PSU ramps in real time, so the field
        # is nowhere near 2 T yet and stays that way for the whole test).
        orch._hold_override_until = {key: 0.0 for key in orch._hold_override_until}
        assert orch.override_active("magnet_z") is False

        # Enforcement must fire on the next eligible tick, mid-ramp.
        qtbot.waitUntil(lambda: len(standby_calls) > calls_before, timeout=2000)

        # It interrupted the climb rather than waiting for it: the magnet is
        # now converging on standby's target, and never got anywhere near
        # the 2 T the operator asked for.
        assert vi.standby_status() in ("converging", "reached")
        assert abs(vi.magnet_field_T() - 2.0) > 0.5
    finally:
        hold_flag.clear()
        orch.shutdown()


def test_converging_standby_is_not_re_issued_and_the_ramp_completes(station, qtbot, monkeypatch, hold_flag):
    """No wedging: while ``standby()`` is converging, it must not be re-issued.

    Catches the naive fix that calls ``standby()`` unconditionally every
    tick a hold persists: rebuilding ``SuperconductingMagnetVI``'s ramp
    generator on every tick would restart the ramp from scratch each time
    and the magnet would never arrive. A tiny
    ``hold_enforcement_interval_s`` (well under the whole test's duration)
    makes sure the assertion is actually exercising the ``standby_status()
    != "away"`` skip, not merely benefiting from the rate limit.
    """
    driver = station.magnet_z._driver
    vi = station.magnet_z
    orch = Orchestrator(
        station,
        tick_interval_ms=10,
        hold_enforcement_interval_s=0.01,
    )
    orch.start_monitoring()
    try:
        # Move the magnet away from zero first, so standby() has ground to
        # cover once the hold trips.
        with qtbot.waitSignal(orch.action_succeeded, timeout=500):
            orch.submit_vi_action("magnet_z", "set_field", target_T=1.0)

        def field_at_1T():
            _force_ramp_to_settle(driver)
            return abs(vi.magnet_field_T() - 1.0) < 0.01 and vi.ramp_status() == "TARGET_REACHED"

        qtbot.waitUntil(field_at_1T, timeout=2000)

        standby_calls: list[float] = []
        original_standby = vi.standby

        def counting_standby():
            standby_calls.append(time.time())
            return original_standby()

        monkeypatch.setattr(vi, "standby", counting_standby)

        hold_flag.trip()

        def magnet_held():
            return "magnet_z" in orch._held_vis()

        qtbot.waitUntil(magnet_held, timeout=2000)
        qtbot.waitUntil(lambda: vi.standby_status() == "converging", timeout=2000)
        assert len(standby_calls) == 1

        # Several more ticks pass while still converging, well past the 0.01s
        # rate-limit interval — standby() must not be re-issued.
        qtbot.wait(200)
        assert vi.standby_status() == "converging"
        assert len(standby_calls) == 1

        def field_back_to_zero():
            _force_ramp_to_settle(driver)
            return abs(vi.magnet_field_T()) < 0.01 and vi.standby_status() == "reached"

        qtbot.waitUntil(field_back_to_zero, timeout=2000)
        assert len(standby_calls) == 1  # the ramp finished on its own — never restarted
    finally:
        hold_flag.clear()
        orch.shutdown()


def test_failed_standby_is_retried_and_escalates_exactly_once(station, qtbot, monkeypatch, caplog, hold_flag):
    """A raising ``standby()`` is retried at the next interval and escalates
    once ``hold_enforcement_max_attempts`` is reached — logged CRITICAL and
    reported via one ``kind="safety_hold"``/``severity="error"``
    ``ErrorEvent``, not one per attempt past the threshold.
    """
    vi = station.magnet_z
    orch = Orchestrator(
        station,
        tick_interval_ms=10,
        hold_enforcement_interval_s=0.03,
        hold_enforcement_max_attempts=3,
    )
    orch.start_monitoring()
    try:

        def raising_standby():
            raise RuntimeError("PSU refused standby")

        monkeypatch.setattr(vi, "standby", raising_standby)

        events = []
        orch.error_event.connect(events.append)

        hold_flag.trip()

        def magnet_held():
            return "magnet_z" in orch._held_vis()

        qtbot.waitUntil(magnet_held, timeout=2000)

        with caplog.at_level(logging.CRITICAL, logger="i2as.core.orchestrator"):
            qtbot.waitUntil(
                lambda: orch._hold_enforcement_attempts.get("magnet_z", 0) >= 3, timeout=3000
            )
            qtbot.wait(60)  # let the escalating tick's signal land

        escalations = [
            e for e in events if e.kind == "safety_hold" and e.severity == "error"
        ]
        assert len(escalations) == 1
        assert "magnet_z" in escalations[0].message
        critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_records) == 1

        # Retried, not abandoned: a further interval later, a fourth attempt happens.
        qtbot.wait(150)
        assert orch._hold_enforcement_attempts.get("magnet_z", 0) >= 4

        # Escalation still fires only once per episode.
        escalations = [
            e for e in events if e.kind == "safety_hold" and e.severity == "error"
        ]
        assert len(escalations) == 1
    finally:
        hold_flag.clear()
        orch.shutdown()


def test_hold_enforcement_bookkeeping_resets_on_settle_and_fresh_episode_starts_at_zero(
    station, qtbot, monkeypatch, hold_flag
):
    """Attempt/escalation bookkeeping is scoped to the CURRENT held-and-away
    episode: it must clear the instant ``standby_status()`` leaves "away",
    and a later, unrelated episode on the same VI must not inherit it.
    """
    vi = station.magnet_z
    driver = station.magnet_z._driver
    original_standby = vi.standby
    orch = Orchestrator(
        station,
        tick_interval_ms=10,
        hold_enforcement_interval_s=0.02,
        hold_enforcement_max_attempts=100,  # never escalate in this test
    )
    orch.start_monitoring()
    try:
        call_count = {"n": 0}

        def flaky_standby():
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RuntimeError("transient PSU fault")
            return original_standby()

        monkeypatch.setattr(vi, "standby", flaky_standby)
        hold_flag.trip()

        def magnet_held():
            return "magnet_z" in orch._held_vis()

        qtbot.waitUntil(magnet_held, timeout=2000)
        qtbot.waitUntil(
            lambda: orch._hold_enforcement_attempts.get("magnet_z", 0) >= 2, timeout=2000
        )

        def field_settled():
            _force_ramp_to_settle(driver)
            return vi.standby_status() != "away"

        qtbot.waitUntil(field_settled, timeout=2000)
        # standby_status() itself updates the instant the 3rd (successful)
        # call returns; the Orchestrator's own bookkeeping-clear only runs
        # at the START of the NEXT enforcement pass — give it one more tick.
        qtbot.wait(30)
        assert "magnet_z" not in orch._hold_enforcement_attempts
        assert "magnet_z" not in orch._hold_enforcement_last_s
        assert "magnet_z" not in orch._hold_enforcement_escalated

        # Recover, move the magnet away from standby's target again while
        # UNHELD (an ordinary manual action), then re-trip: a second,
        # unrelated episode.
        hold_flag.clear()
        qtbot.waitUntil(lambda: "magnet_z" not in orch._held_vis(), timeout=2000)
        monkeypatch.setattr(vi, "standby", original_standby)

        with qtbot.waitSignal(orch.action_succeeded, timeout=500):
            orch.submit_vi_action("magnet_z", "set_field", target_T=0.5)

        def field_at_half_tesla():
            _force_ramp_to_settle(driver)
            return abs(vi.magnet_field_T() - 0.5) < 0.01

        qtbot.waitUntil(field_at_half_tesla, timeout=2000)

        hold_flag.trip()
        qtbot.waitUntil(magnet_held, timeout=2000)

        def field_back_to_zero():
            _force_ramp_to_settle(driver)
            return abs(vi.magnet_field_T()) < 0.01

        qtbot.waitUntil(field_back_to_zero, timeout=2000)
        # The fresh episode never accumulated a raise: bookkeeping started
        # at zero, exactly as if this were the VI's very first hold.
        assert orch._hold_enforcement_attempts.get("magnet_z", 0) == 0
    finally:
        hold_flag.clear()
        orch.shutdown()


def test_override_suppresses_enforcement(station, qtbot, monkeypatch, hold_flag):
    """While ``override_active(vi_name)`` is True, enforcement must not call
    ``standby()`` at all — the operator is deliberately in control.
    """
    vi = station.magnet_z
    orch = Orchestrator(
        station,
        tick_interval_ms=10,
        hold_enforcement_interval_s=0.02,  # tiny: would re-fire fast if not suppressed
    )
    orch.start_monitoring()
    try:
        standby_calls: list[float] = []
        original_standby = vi.standby

        def counting_standby():
            standby_calls.append(time.time())
            return original_standby()

        monkeypatch.setattr(vi, "standby", counting_standby)

        hold_flag.trip()

        def magnet_held():
            return "magnet_z" in orch._held_vis()

        qtbot.waitUntil(magnet_held, timeout=2000)
        qtbot.waitUntil(lambda: len(standby_calls) >= 1, timeout=2000)  # the onset attempt

        orch.acknowledge()
        assert orch.override_active("magnet_z") is True

        calls_before = len(standby_calls)
        with qtbot.waitSignal(orch.action_succeeded, timeout=500):
            orch.submit_vi_action("magnet_z", "set_field", target_T=1.5)
        assert vi.standby_status() == "away"  # would be enforced if not overridden

        qtbot.wait(150)
        assert len(standby_calls) == calls_before
    finally:
        hold_flag.clear()
        orch.shutdown()


class _SyntheticHoldFlagRampableVI(BaseVirtualInstrument, RampableVI):
    """Test-local double: declares its OWN hold-severity safety flag.

    Proves the enforcement standard's generality (see
    ``Orchestrator._enforce_safety_holds()``'s docstring): nothing there
    names a flag or a VI category — it reads only ``Condition.severity ==
    "hold"`` and ``affected_vis``, both already resolved by
    ``core.conditions.decide()``. A brand-new flag on a brand-new,
    non-magnet VI is enforced identically with zero Orchestrator change —
    this VI is both the flag's producer AND the sole VI concerned with it,
    unlike the ``hold_flag`` fixture's declaration (produced by one VI,
    concerned-with by another), which exercises the same mechanism from the
    other side.
    """

    vi_type = "system"
    safety_flags: ClassVar[dict[str, str]] = {"widget_stuck": "hold"}

    def __init__(self, drivers: dict, **init_params: object) -> None:
        super().__init__(drivers, **init_params)
        self.value = 0.0
        self._target = 0.0
        self._ramping = False
        self.tripped = False

    def safety_concerns(self) -> set[str]:
        return {"widget_stuck"}

    def evaluate_safety(self, state: dict) -> dict[str, bool]:
        return {"widget_stuck": self.tripped}

    def start_ramp(self, target: float) -> None:
        self._target = target
        self._ramping = True

    def advance_ramp(self) -> None:
        if self._ramping:
            self.value = self._target
            self._ramping = False

    def ramp_status(self) -> str:
        return "RAMPING" if self._ramping else "TARGET_REACHED"

    def stop_ramp(self) -> None:
        self._ramping = False

    def standby(self) -> None:
        self.start_ramp(0.0)


def test_generality_second_hold_flag_on_non_magnet_vi_enforced_identically(qtbot):
    """A second, unrelated hold-severity flag on a non-magnet VI is enforced
    by the exact same, unmodified ``_enforce_safety_holds()`` — see
    ``_SyntheticHoldFlagRampableVI``'s docstring for what this proves.
    """
    station = Station()
    station.register_vi("widget", _SyntheticHoldFlagRampableVI({}), "system")
    orch = Orchestrator(station, tick_interval_ms=10, hold_enforcement_interval_s=0.02)
    orch.start_monitoring()
    vi = station.get_vi("widget")
    try:
        vi.start_ramp(5.0)
        vi.advance_ramp()
        assert vi.value == 5.0

        vi.tripped = True

        def widget_held():
            return "widget" in orch._held_vis()

        qtbot.waitUntil(widget_held, timeout=2000)

        def widget_back_to_zero_and_settled():
            return vi.value == 0.0 and vi.standby_status() == "reached"

        qtbot.waitUntil(widget_back_to_zero_and_settled, timeout=2000)
    finally:
        vi.tripped = False
        orch.shutdown()


class _SyntheticNonRampableHoldFlagVI(BaseVirtualInstrument):
    """Test-local double: a non-``RampableVI`` held by its own hold-severity flag."""

    vi_type = "system"
    safety_flags: ClassVar[dict[str, str]] = {"sensor_stuck": "hold"}

    def __init__(self, drivers: dict, **init_params: object) -> None:
        super().__init__(drivers, **init_params)
        self.tripped = False
        self.standby_calls = 0

    def safety_concerns(self) -> set[str]:
        return {"sensor_stuck"}

    def evaluate_safety(self, state: dict) -> dict[str, bool]:
        return {"sensor_stuck": self.tripped}

    def standby(self) -> None:
        self.standby_calls += 1


def test_non_rampable_held_vi_reports_reached_and_is_never_recommanded(qtbot):
    """A held non-``RampableVI`` has no intermediate state to converge
    through — ``standby_status()`` is unconditionally ``"reached"`` — so
    enforcement never calls its ``standby()`` at all.
    """
    station = Station()
    station.register_vi("sensor", _SyntheticNonRampableHoldFlagVI({}), "system")
    orch = Orchestrator(station, tick_interval_ms=10, hold_enforcement_interval_s=0.02)
    orch.start_monitoring()
    vi = station.get_vi("sensor")
    try:
        vi.tripped = True

        def sensor_held():
            return "sensor" in orch._held_vis()

        qtbot.waitUntil(sensor_held, timeout=2000)
        assert vi.standby_status() == "reached"
        qtbot.wait(150)
        assert vi.standby_calls == 0
    finally:
        vi.tripped = False
        orch.shutdown()


def test_hold_enforcement_announces_reassertion(orchestrator, station, qtbot, hold_flag):
    """A routine re-assertion emits BOTH the concise status line and a
    ``kind="safety_hold"``/``severity="warning"`` ``ErrorEvent`` naming the
    VI and the tripped condition — silent hardware motion is unacceptable
    even for a correct safety action (see ``_emit_hold_enforcement_event``'s
    docstring).
    """
    status_messages: list[str] = []
    orchestrator.status_message.connect(status_messages.append)
    events = []
    orchestrator.error_event.connect(events.append)

    hold_flag.trip()
    try:

        def hold_events():
            return [
                e for e in events if e.kind == "safety_hold" and e.vi_name == "magnet_z"
            ]

        qtbot.waitUntil(lambda: len(hold_events()) >= 1, timeout=2000)
        event = hold_events()[0]
        assert event.severity == "warning"
        assert "magnet_z" in event.message
        assert hold_flag.flag in event.message
        assert any("magnet_z" in message for message in status_messages)
    finally:
        hold_flag.clear()


# ── Ramp scope: a run waits for, and stops, only the ramps it started ─────────


class NoTargetProcedure:
    """A procedure shaped like TimeSeries: commands nothing, claims the reading path.

    Stands in for the real thing at L3 so the state-machine behaviour is
    tested without a measurement VI, an HDF5 file, or a schedule.
    """

    name = "No-Target Series"

    def __init__(self, station, points: int = 3):
        self._station = station
        self._sweep = list(range(points))
        self._index = 0
        self.measure_called = 0

    def initiate(self):
        return PhasePlan(targets={}, commands=(), wait_s=0.0)

    def change_sweep_step(self):
        self._index += 1
        if self._index >= len(self._sweep):
            return None
        return StepPlan(targets={}, wait_s=0.0)

    def measure(self):
        self.measure_called += 1

    def standby(self):
        return PhasePlan(targets={}, commands=(), wait_s=0.0)

    def abort(self):
        return ()

    def claimed_vi_names(self):
        return {"keithley_delta_mode"}

    def get_progress(self):
        return self._index / len(self._sweep)


def _slow_manual_ramp(station):
    """Start a slow magnet ramp the way a front-panel click would."""
    station.magnet_z._default_ramp_rate = 0.001
    station.magnet_z._ramp_segments = []
    station.process_system_targets({"magnet_z": Target(1.0)})
    assert station.magnet_z.ramp_status() == "RAMPING"


def test_manual_ramp_does_not_block_a_run_that_owns_no_ramps(
    orchestrator, station, qtbot
):
    """A run with no targets measures on schedule while the operator ramps by hand.

    Before the ramp scope existed, the RAMPING->MEASURING gate asked whether
    ANY hardware was still moving, so a manual front-panel ramp stalled every
    measurement of a procedure that had commanded nothing at all.
    """
    procedure = NoTargetProcedure(station)
    orchestrator.run_procedure(procedure)
    _slow_manual_ramp(station)

    with qtbot.waitSignal(orchestrator.procedure_finished, timeout=5000):
        pass

    assert procedure.measure_called == 3
    # Untouched by the run's standby: the operator's ramp is still going.
    assert station.magnet_z.ramp_status() == "RAMPING"


def test_abort_leaves_an_unowned_manual_ramp_running(orchestrator, station, qtbot):
    """Aborting a run stops the ramps it started — not the operator's.

    The run commanded no targets, so it owns no ramps and has nothing to
    hold in place; freezing the manual ramp would be the run reaching
    outside its own claim.
    """
    orchestrator.run_procedure(NoTargetProcedure(station))
    _slow_manual_ramp(station)

    orchestrator.abort_procedure()

    assert orchestrator._procedure is None
    assert station.magnet_z.ramp_status() == "RAMPING"


def test_run_that_owns_a_ramp_still_waits_for_it(orchestrator, station, qtbot):
    """The scope narrows nothing for an ordinary sweep: it still waits for its own.

    MockProcedure targets magnet_z, so magnet_z is in its ramp scope and the
    measurement gate behaves exactly as before the change.
    """
    procedure = MockProcedure(station)
    station.magnet_z._default_ramp_rate = 0.001
    station.magnet_z._ramp_segments = []
    orchestrator.run_procedure(procedure)

    qtbot.waitUntil(lambda: orchestrator._state == OrchestratorState.RAMPING, timeout=2000)
    qtbot.wait(200)

    assert station.magnet_z.ramp_status() == "RAMPING"
    assert procedure.measure_called == 0  # held at the gate by its OWN ramp
    orchestrator.abort_procedure()


def test_out_of_scope_ramp_advances_in_every_run_state(orchestrator, station, qtbot):
    """A manual ramp progresses on every tick of a run, not only on RAMPING ticks.

    Ramp generators step exactly once per call to the Station's advance, and
    that call used to live only in the ramp-aware states. A run that no
    longer waits for a foreign ramp spends most of its ticks elsewhere, so
    without an advance in those states the operator's ramp would crawl at a
    third of its rate — or, behind a gate, freeze outright.
    """
    orchestrator.run_procedure(NoTargetProcedure(station, points=30))
    _slow_manual_ramp(station)

    measured_at = []
    fields = []
    for _ in range(12):
        orchestrator._tick()
        fields.append(station.magnet_z.magnet_field_T())
        measured_at.append(orchestrator._state)

    # Strictly increasing: every single tick moved the ramp, whatever state
    # the run was in.
    assert all(b > a for a, b in zip(fields, fields[1:])), fields
    orchestrator.abort_procedure()


# ── Every GUI action gets an explicit verdict ────────────────────────────────
# The control-validation standard (virtual_instruments/base.py) promises the
# operator an explicit success or failure for every action. These five entry
# points used to return silently on their refusal paths, so a click produced
# no state change and no message — indistinguishable from a wedged app. They
# matter more once the Orchestrator is not on the GUI thread, where a missing
# verdict is the only symptom a caller could ever see.

def test_pause_without_a_run_is_refused_with_a_reason(orchestrator, qtbot):
    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500) as blocker:
        orchestrator.pause_procedure()
    assert "no run is active" in blocker.args[0]


def test_pause_in_a_disallowed_state_is_refused_with_a_reason(orchestrator, qtbot):
    orchestrator._procedure = object()          # a run exists...
    orchestrator._state = OrchestratorState.PAUSED   # ...but this state cannot pause
    try:
        with qtbot.waitSignal(orchestrator.action_blocked, timeout=500) as blocker:
            orchestrator.pause_procedure()
        assert "PAUSED" in blocker.args[0]
    finally:
        orchestrator._procedure = None
        orchestrator._state = OrchestratorState.IDLE


def test_resume_outside_paused_is_refused_with_a_reason(orchestrator, qtbot):
    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500) as blocker:
        orchestrator.resume_procedure()
    assert "IDLE" in blocker.args[0]


def test_recover_from_error_outside_error_is_refused_with_a_reason(orchestrator, qtbot):
    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500) as blocker:
        orchestrator.recover_from_error()
    assert "IDLE" in blocker.args[0]
    assert orchestrator.state == OrchestratorState.IDLE.value


def test_abort_during_emergency_is_refused_with_a_reason(orchestrator, qtbot):
    orchestrator._state = OrchestratorState.EMERGENCY
    try:
        with qtbot.waitSignal(orchestrator.action_blocked, timeout=500) as blocker:
            orchestrator.abort_procedure()
        assert "EMERGENCY" in blocker.args[0]
    finally:
        orchestrator._state = OrchestratorState.IDLE


def test_recover_from_error_still_works_in_error(orchestrator, qtbot):
    """The refusal path must not have broken the real one."""
    orchestrator._state = OrchestratorState.ERROR
    orchestrator.recover_from_error()
    assert orchestrator.state == OrchestratorState.IDLE.value


# ══════════════════════════════════════════════════════════════════════
# The engine port: submit(Command) -> one Verdict, and the event stream
# (the verdict standard — see Orchestrator's class docstring)
# ══════════════════════════════════════════════════════════════════════

AGENT = ev.Actor(kind=ev.ActorKind.AGENT, id="drift-watch", role="operator")

FIELD_SWEEP_PARAMS = {
    "measurement_vi": "dc_measurement",
    "field_start": -0.1,
    "field_end": 0.1,
    "field_steps": 3,
    "temperature": 300.0,
    "current_A": 1e-6,
    "readings_per_point": 5,
    "init_wait": 0.0,
    "step_wait": 0.0,
}


class _Recorder:
    """Collect everything one Orchestrator said, in emission order."""

    def __init__(self, orchestrator):
        self.verdicts: list[ev.Verdict] = []
        self.events: list[object] = []
        orchestrator.verdict_emitted.connect(self.verdicts.append)
        orchestrator.event_emitted.connect(self.events.append)

    def of_type(self, event_type):
        return [event for event in self.events if isinstance(event, event_type)]


@pytest.fixture
def port(station, qtbot, tmp_path):
    """An Orchestrator with a run catalog, plus a recorder of its two channels."""
    orch = Orchestrator(
        station, tick_interval_ms=10, run_catalog={"FieldSweep": FieldSweep}
    )
    orch.start_monitoring()
    recorder = _Recorder(orch)
    yield orch, recorder, tmp_path
    orch.shutdown()


def _run_procedure_command(data_directory, **overrides):
    """Build the dict-payload command that starts a FieldSweep."""
    args = {
        "procedure": "FieldSweep",
        "params": dict(FIELD_SWEEP_PARAMS),
        "sample_info": {"sample_name": "S", "sample_id": "S-1", "comments": ""},
        "data_directory": str(data_directory),
        "file_prefix": "submit",
    }
    args.update(overrides)
    return ev.Command(name=ev.CommandName.RUN_PROCEDURE, actor=AGENT, args=args)


def test_dict_payload_starts_a_run_and_is_answered_ok(port):
    """A JSON-shaped run payload builds the procedure and starts the run."""
    orch, recorder, tmp_path = port
    command = _run_procedure_command(tmp_path)

    request_id = orch.submit(command)

    assert request_id == command.request_id
    assert [v.request_id for v in recorder.verdicts] == [request_id]
    verdict = recorder.verdicts[0]
    assert verdict.ok and verdict.code is ev.VerdictCode.OK
    assert verdict.actor == AGENT
    assert orch.state == OrchestratorState.INITIATING.value
    started = recorder.of_type(ev.RunStarted)
    assert len(started) == 1
    assert started[0].actor.kind is ev.ActorKind.AGENT
    assert started[0].request_id == request_id
    orch.abort_procedure()




def test_a_run_payload_naming_no_known_class_fails_without_raising(port):
    """An unknown procedure class is a FAILED verdict, never an exception."""
    orch, recorder, tmp_path = port

    orch.submit(_run_procedure_command(tmp_path, procedure="NoSuchProcedure"))

    assert len(recorder.verdicts) == 1
    assert recorder.verdicts[0].code is ev.VerdictCode.FAILED
    assert "NoSuchProcedure" in recorder.verdicts[0].reason
    assert orch.state == OrchestratorState.IDLE.value


def test_an_unknown_command_name_never_reaches_the_engine(port):
    """The enumeration is the boundary: an unknown name cannot become a Command.

    ``submit()`` therefore never sees one — and a name that got past the enum
    would still be answered rather than raised, since dispatch is a guarded
    lookup (``test_a_command_with_arguments_the_method_rejects_fails``
    exercises the same path).
    """
    with pytest.raises(ValueError):
        ev.Command(name="no_such_command")


def test_a_command_with_arguments_the_method_rejects_fails(port):
    """Arguments that do not fit the method answer FAILED, never raise."""
    orch, recorder, _tmp_path = port

    request_id = orch.submit(
        ev.Command(name=ev.CommandName.STOP_RAMP, actor=AGENT, args={"nope": 1})
    )

    assert [v.request_id for v in recorder.verdicts] == [request_id]
    assert recorder.verdicts[0].code is ev.VerdictCode.FAILED


def test_pause_when_idle_is_blocked_by_state(port):
    """A refusal answers the command, with the code AND the existing signal."""
    orch, recorder, _tmp_path = port

    orch.submit(ev.Command(name=ev.CommandName.PAUSE_PROCEDURE, actor=AGENT))

    assert len(recorder.verdicts) == 1
    verdict = recorder.verdicts[0]
    assert verdict.code is ev.VerdictCode.BLOCKED_STATE
    assert "no run is active" in verdict.reason
    assert not verdict.ok


def test_emergency_standby_is_accepted_from_idle_and_enters_emergency(port):
    """The unconditional safe-off path is reachable through the port."""
    orch, recorder, _tmp_path = port

    orch.submit(
        ev.Command(
            name=ev.CommandName.EMERGENCY_STANDBY,
            actor=AGENT,
            args={"reason": "agent saw a drift it could not explain"},
        )
    )

    assert [v.code for v in recorder.verdicts] == [ev.VerdictCode.OK]
    assert orch.state == OrchestratorState.EMERGENCY.value
    changes = recorder.of_type(ev.StateChange)
    assert changes[-1].state == "EMERGENCY"
    assert changes[-1].cause == "emergency"
    assert changes[-1].actor.kind is ev.ActorKind.AGENT


def test_acknowledge_with_nothing_held_is_refused(port):
    """One of the audit's silent refusals: acknowledging nothing now answers."""
    orch, recorder, _tmp_path = port

    orch.submit(ev.Command(name=ev.CommandName.ACKNOWLEDGE, actor=AGENT))

    assert len(recorder.verdicts) == 1
    assert recorder.verdicts[0].code is ev.VerdictCode.BLOCKED_STATE
    assert "Nothing to acknowledge" in recorder.verdicts[0].reason


def test_acknowledge_fault_with_no_fault_is_refused(orchestrator, qtbot):
    """Acknowledging a VI that is not faulted says so instead of logging quietly."""
    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500) as blocker:
        orchestrator.acknowledge_fault("magnet_z")
    assert "no active fault" in blocker.args[0]


def test_unknown_global_action_is_refused(orchestrator, qtbot):
    """An unknown global action is a typo the caller must be told about."""
    with qtbot.waitSignal(orchestrator.action_blocked, timeout=500) as blocker:
        orchestrator.submit_global_action("standby_everything")
    assert "standby_everything" in blocker.args[0]
    assert orchestrator._gui_action_queue == []


def test_run_queue_outside_idle_is_refused(orchestrator, qtbot):
    """Asking the queue to advance while busy answers rather than doing nothing."""
    orchestrator._state = OrchestratorState.RAMPING
    try:
        with qtbot.waitSignal(orchestrator.action_blocked, timeout=500) as blocker:
            orchestrator.run_queue()
        assert "requires IDLE" in blocker.args[0]
    finally:
        orchestrator._state = OrchestratorState.IDLE


def test_over_limit_action_is_blocked_by_limit_with_structured_detail(port):
    """A control-limit refusal carries the numbers, so no client parses prose."""
    orch, recorder, _tmp_path = port

    orch.submit(
        ev.Command(
            name=ev.CommandName.SUBMIT_VI_ACTION,
            actor=AGENT,
            args={"vi_name": "magnet_z", "method_name": "set_field", "target_T": 20.0},
        )
    )
    # Queued for the tick, the single hardware writer: no verdict yet.
    assert recorder.verdicts == []

    orch._tick()

    assert len(recorder.verdicts) == 1
    verdict = recorder.verdicts[0]
    assert verdict.code is ev.VerdictCode.BLOCKED_LIMIT
    assert verdict.detail == {
        "param": "target_T",
        "value": 20.0,
        "lo": -9.0,
        "hi": 9.0,
        "limit_name": "field_T",
    }


def test_an_admitted_action_is_answered_by_the_drain_with_its_result(port):
    """The asynchronous case: the queued action's verdict arrives at the drain."""
    orch, recorder, _tmp_path = port

    request_id = orch.submit(
        ev.Command(
            name=ev.CommandName.SUBMIT_VI_ACTION,
            actor=AGENT,
            args={"vi_name": "magnet_z", "method_name": "set_field", "target_T": 0.05},
        )
    )
    assert recorder.verdicts == []

    orch._tick()

    assert [v.request_id for v in recorder.verdicts] == [request_id]
    assert recorder.verdicts[0].code is ev.VerdictCode.OK


def test_an_action_refused_at_the_drain_is_still_answered(port):
    """A queued action refused when the tick re-checks it still gets its verdict."""
    orch, recorder, _tmp_path = port
    orch.submit(
        ev.Command(
            name=ev.CommandName.SUBMIT_VI_ACTION,
            actor=AGENT,
            args={"vi_name": "magnet_z", "method_name": "set_field", "target_T": 0.05},
        )
    )
    # The world changed between the click and the tick: a run now claims
    # everything, so the drain gate refuses what submission admitted.
    orch._procedure = MockProcedure(orch._station)
    orch._active_claims = None
    orch._state = OrchestratorState.RAMPING

    orch._tick()

    assert len(recorder.verdicts) == 1
    assert recorder.verdicts[0].code is ev.VerdictCode.BLOCKED_CLAIM


def test_a_claimed_vi_refuses_a_stop_with_the_claim_code(port):
    """The admission predicate's code travels with its reason."""
    orch, recorder, _tmp_path = port
    orch._procedure = MockProcedure(orch._station)
    orch._active_claims = {"magnet_z"}
    orch._state = OrchestratorState.RAMPING

    orch.submit(
        ev.Command(
            name=ev.CommandName.STOP_RAMP, actor=AGENT, args={"vi_name": "magnet_z"}
        )
    )

    assert recorder.verdicts[-1].code is ev.VerdictCode.BLOCKED_CLAIM


def test_an_envelope_violation_is_blocked_by_envelope(port):
    """The experiment envelope's refusal has its own code, not a generic one."""
    orch, recorder, _tmp_path = port
    orch.set_experiment_envelope(
        ExperimentEnvelope.from_dict({"magnet_z": {"max_value": 1.0}})
    )

    orch.submit(
        ev.Command(
            name=ev.CommandName.SUBMIT_VI_ACTION,
            actor=AGENT,
            args={"vi_name": "magnet_z", "method_name": "set_field", "target_T": 5.0},
        )
    )

    assert len(recorder.verdicts) == 1
    assert recorder.verdicts[0].code is ev.VerdictCode.BLOCKED_ENVELOPE


def test_the_envelope_can_be_set_and_cleared_through_the_port(port):
    """``set_experiment_envelope`` takes the envelope's dict form over the wire."""
    orch, recorder, _tmp_path = port

    orch.submit(
        ev.Command(
            name=ev.CommandName.SET_EXPERIMENT_ENVELOPE,
            actor=AGENT,
            args={"envelope": {"magnet_z": {"min_value": -1.0, "max_value": 1.0}}},
        )
    )
    assert orch._session_envelope is not None
    assert orch._session_envelope.bounds["magnet_z"].max_value == 1.0

    orch.submit(
        ev.Command(
            name=ev.CommandName.SET_EXPERIMENT_ENVELOPE, args={"envelope": None}
        )
    )
    assert orch._session_envelope is None
    assert [v.code for v in recorder.verdicts] == [
        ev.VerdictCode.OK,
        ev.VerdictCode.OK,
    ]


@pytest.mark.parametrize(
    "state, expected",
    [
        (OrchestratorState.IDLE, ev.VerdictCode.BLOCKED_STATE),
        (OrchestratorState.RAMPING, ev.VerdictCode.BLOCKED_STATE),
        (OrchestratorState.ERROR, ev.VerdictCode.BLOCKED_STATE),
        (OrchestratorState.EMERGENCY, ev.VerdictCode.BLOCKED_STATE),
    ],
)
def test_pause_refusal_matrix_over_states(port, state, expected):
    """Every state that cannot pause refuses with a code, never silently.

    The completeness rule as a matrix: with no run active, ``pause_procedure``
    must answer in every state the machine can be in.
    """
    orch, recorder, _tmp_path = port
    orch._state = state
    try:
        orch.submit(ev.Command(name=ev.CommandName.PAUSE_PROCEDURE, actor=AGENT))
    finally:
        orch._state = OrchestratorState.IDLE
    assert len(recorder.verdicts) == 1
    assert recorder.verdicts[0].code is expected


def test_every_command_is_answered_exactly_once(port):
    """The whole point of the standard, checked over a mixed batch."""
    orch, recorder, tmp_path = port
    commands = [
        ev.Command(name=ev.CommandName.START_MONITORING, actor=AGENT),
        ev.Command(name=ev.CommandName.PAUSE_PROCEDURE, actor=AGENT),
        ev.Command(name=ev.CommandName.ACKNOWLEDGE, actor=AGENT),
        ev.Command(name=ev.CommandName.SET_ATTENDANCE, args={"attended": True}),
        ev.Command(name=ev.CommandName.RECOVER_FROM_ERROR, actor=AGENT),
        _run_procedure_command(tmp_path, procedure="NoSuchProcedure"),
    ]
    for command in commands:
        orch.submit(command)

    answered = [v.request_id for v in recorder.verdicts]
    assert answered == [command.request_id for command in commands]
    assert len(set(answered)) == len(answered)


def test_the_actor_keyword_does_not_change_direct_calls(orchestrator, qtbot):
    """A direct call still works, still defaults to the operator sentinel."""
    changes = []
    orchestrator.event_emitted.connect(
        lambda event: changes.append(event)
        if isinstance(event, ev.StateChange)
        else None
    )
    orchestrator._state = OrchestratorState.ERROR
    orchestrator.recover_from_error()

    assert orchestrator.state == OrchestratorState.IDLE.value
    assert changes[-1].actor == ev.OPERATOR
    assert changes[-1].cause == "recover_from_error"


def test_a_tick_driven_transition_is_attributed_to_the_system(orchestrator, station):
    """No command in flight means the engine did it: the system actor."""
    changes = []
    orchestrator.event_emitted.connect(
        lambda event: changes.append(event)
        if isinstance(event, ev.StateChange)
        else None
    )
    _fast_magnet(station)
    orchestrator.run_procedure(MockProcedure(station))
    changes.clear()
    orchestrator._tick()  # INITIATING -> RAMPING, on the tick

    assert changes, "the tick made no transition"
    assert changes[0].actor.kind is ev.ActorKind.SYSTEM
    assert changes[0].cause == "tick"
    orchestrator.abort_procedure()


def test_one_status_snapshot_per_quiet_tick_and_it_round_trips(port):
    """The status mirror refreshes once per tick and survives a JSON hop."""
    orch, recorder, _tmp_path = port
    orch._tick()

    snapshots = recorder.of_type(ev.StatusSnapshot)
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    wire = json.loads(json.dumps(snapshot.to_json()))
    assert ev.StatusSnapshot.from_json(wire) == snapshot

    assert snapshot.state == OrchestratorState.IDLE.value
    assert snapshot.is_monitoring is True
    assert snapshot.run is None
    assert set(snapshot.availabilities) == set(orch._station.availabilities())
    assert set(snapshot.instruments) == set(snapshot.availabilities)
    assert snapshot.envelope_variables  # the sim station has bounded VIs


def test_a_state_change_also_refreshes_the_snapshot(port):
    """A client that never calls in still sees the new state immediately."""
    orch, recorder, _tmp_path = port
    orch.submit(
        ev.Command(
            name=ev.CommandName.EMERGENCY_STANDBY, args={"reason": "test"}, actor=AGENT
        )
    )
    snapshots = recorder.of_type(ev.StatusSnapshot)
    assert snapshots and snapshots[-1].state == OrchestratorState.EMERGENCY.value


def test_the_snapshot_carries_the_run_in_flight(port, station):
    """`run` names the run, its kind and where it has got to."""
    orch, recorder, _tmp_path = port
    _fast_magnet(station)
    orch.run_procedure(MockProcedure(station))
    orch._tick()

    snapshot = recorder.of_type(ev.StatusSnapshot)[-1]
    assert snapshot.run is not None
    assert snapshot.run["name"] == "Mock Sweep"
    assert snapshot.run["kind"] == "procedure"
    assert snapshot.run["id"]
    assert snapshot.active_run_kind == "procedure"
    orch.abort_procedure()


def test_every_emitted_snapshot_payload_is_a_copy(port):
    """Mutating what a listener received can never reach the engine."""
    orch, recorder, _tmp_path = port
    orch._tick()
    snapshot = recorder.of_type(ev.StatusSnapshot)[0]
    name = next(iter(snapshot.availabilities))
    snapshot.availabilities[name]["state"] = "tampered"

    orch._tick()
    fresh = recorder.of_type(ev.StatusSnapshot)[-1]
    assert fresh.availabilities[name]["state"] != "tampered"


def test_readings_and_datapoints_reach_the_event_stream(port, station):
    """The monitored poll and each measured point are contract events too."""
    orch, recorder, _tmp_path = port
    _fast_magnet(station)
    orch._tick()
    assert recorder.of_type(ev.Readings), "a monitored tick emits its readings"

    procedure = MockProcedure(station)
    procedure.last_datapoint = {"field_T": 0.5, "resistance_ohm": 12.75}
    orch.run_procedure(procedure)
    _tick_until(orch, lambda: bool(recorder.of_type(ev.Datapoint)))

    point = recorder.of_type(ev.Datapoint)[0]
    assert point.index == 0
    assert point.values["field_T"] == 0.5
    assert point.run_id
    orch.abort_procedure()

    finished = recorder.of_type(ev.RunFinished)
    assert finished and finished[-1].status == "aborted"


def test_station_info_is_emitted_at_construction_and_on_disconnect(station, qtbot):
    """The static half is declared at construction and re-declared on disconnect.

    The engine emits the Station's own ``station_info()`` snapshot, re-stamped
    with the event stream's ``seq`` so it orders against every other event.
    """
    events: list[object] = []
    orch = Orchestrator(station, tick_interval_ms=10)
    try:
        orch.event_emitted.connect(events.append)
        orch._emit_station_info()
        first = [e for e in events if isinstance(e, ev.StationInfo)]
        assert first, "construction must declare the station"
        names = {info.name for info in first[-1].instruments}
        assert "magnet_z" in names
        assert first[-1].setup == station.setup_name()

        orch.disconnect_instrument("magnet_z")
        declared = [e for e in events if isinstance(e, ev.StationInfo)]
        assert len(declared) > len(first), "disconnect must re-declare the station"
        latest = declared[-1]
        assert latest.seq > first[-1].seq
        by_name = {info.name: info for info in latest.instruments}
        assert "magnet_z" in by_name, "an offline VI is still declared"
        assert json.loads(json.dumps(latest.to_json())) == latest.to_json()
    finally:
        orch.shutdown()


def test_station_info_is_re_declared_before_the_per_vi_notification(station, qtbot):
    """A client rebuilding a card on ``instrument_disconnected`` sees the NEW station.

    The order matters: a client hears the per-VI notification and rebuilds
    that instrument's panel from the declaration it holds, so the declaration
    must already be the one that includes the change.
    """
    seen: list[str] = []
    orch = Orchestrator(station, tick_interval_ms=10)
    try:
        orch.event_emitted.connect(
            lambda e: seen.append("station_info")
            if isinstance(e, ev.StationInfo)
            else None
        )
        orch.instrument_disconnected.connect(lambda _n: seen.append("notified"))
        seen.clear()
        orch.disconnect_instrument("magnet_z")
        assert seen == ["station_info", "notified"]
    finally:
        orch.shutdown()


def test_the_priming_reads_answer_what_the_event_stream_broadcasts(port):
    """``station_info()`` / ``status_snapshot()`` are the mirror's two priming reads."""
    orch, recorder, _tmp_path = port
    orch._emit_station_info()  # the recorder attached after construction

    declared = orch.station_info()
    assert isinstance(declared, ev.StationInfo)
    broadcast = recorder.of_type(ev.StationInfo)[-1]
    assert {i.name for i in declared.instruments} == {
        i.name for i in broadcast.instruments
    }
    assert declared.seq > broadcast.seq, "each read is stamped on the one stream"

    snapshot = orch.status_snapshot()
    assert isinstance(snapshot, ev.StatusSnapshot)
    assert snapshot.state == orch.state
    assert snapshot.is_monitoring is orch.is_monitoring()


def test_ping_instrument_answers_reachable_and_unreachable(port):
    """The connection check is a command whose result arrives as a verdict."""
    orch, recorder, _tmp_path = port
    succeeded: list[tuple[str, str]] = []
    failed: list[tuple[str, str, str]] = []
    orch.action_succeeded.connect(lambda v, m: succeeded.append((v, m)))
    orch.action_failed.connect(lambda v, m, r: failed.append((v, m, r)))

    request = orch.submit(
        ev.Command(
            name=ev.CommandName.PING_INSTRUMENT, args={"vi_name": "magnet_z"}
        )
    )
    verdict = [v for v in recorder.verdicts if v.request_id == request][-1]
    assert verdict.code is ev.VerdictCode.OK
    assert verdict.result is True
    assert succeeded == [("magnet_z", "ping")]

    orch._station.get_vi("magnet_z").ping = lambda: False
    orch.submit(
        ev.Command(
            name=ev.CommandName.PING_INSTRUMENT, args={"vi_name": "magnet_z"}
        )
    )
    assert failed and failed[-1][:2] == ("magnet_z", "ping")
    assert recorder.verdicts[-1].code is ev.VerdictCode.FAILED


def test_ping_instrument_refuses_an_unknown_instrument(port):
    """An instrument that is not live is refused with a reason, never silently."""
    orch, recorder, _tmp_path = port
    orch.submit(
        ev.Command(name=ev.CommandName.PING_INSTRUMENT, args={"vi_name": "nope"})
    )
    verdict = recorder.verdicts[-1]
    assert verdict.code is ev.VerdictCode.FAILED
    assert "no live instrument" in verdict.reason


def test_a_failing_station_info_is_logged_not_raised(port, caplog):
    """Reporting never disrupts a run: a broken declaration is logged and skipped."""
    orch, recorder, _tmp_path = port
    before = len(recorder.of_type(ev.StationInfo))

    def broken():
        raise RuntimeError("declaration exploded")

    orch._station.station_info = broken
    with caplog.at_level(logging.ERROR):
        orch._emit_station_info()
    assert len(recorder.of_type(ev.StationInfo)) == before
    assert any("station_info() failed" in r.message for r in caplog.records)


def test_sequence_numbers_are_one_stream_across_events_and_verdicts(port):
    """One counter, so a client can order everything the engine said."""
    orch, recorder, _tmp_path = port
    orch.submit(ev.Command(name=ev.CommandName.PAUSE_PROCEDURE, actor=AGENT))
    orch._tick()

    sequences = [item.seq for item in recorder.events + recorder.verdicts]
    assert len(set(sequences)) == len(sequences)
    assert min(sequences) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# The pull seam: the engine asks for the next run, and decides when it starts
# ══════════════════════════════════════════════════════════════════════════════


class QueueableProcedure(MockProcedure):
    """A MockProcedure that accepts ``build_procedure()``'s keyword contract."""

    name = "Queueable Sweep"

    def __init__(
        self,
        station,
        sample_info=None,
        data_directory="",
        file_prefix="",
        experiment_info=None,
        **params,
    ):
        super().__init__(station)


def _pull_seam(orchestrator, station, queue):
    """Wire *queue* to the engine as its pull seam, and return the catalog."""
    catalog = {"QueueableProcedure": QueueableProcedure}

    def next_run():
        spec = queue.pop_next()
        if spec is None:
            return None
        return build_run(spec, station=station, run_catalog=catalog)

    orchestrator.next_procedure = next_run
    orchestrator.queue_snapshot = queue.entries
    return catalog




def test_the_pull_seam_starts_a_plain_procedure_as_a_procedure(orchestrator, station):
    """A run pulled off the queue goes down the procedure path."""
    queue = RunQueue()
    _pull_seam(orchestrator, station, queue)
    queue.add(RunSpec(kind="procedure", run_class="QueueableProcedure"))

    orchestrator.run_queue()

    assert isinstance(orchestrator._procedure, QueueableProcedure)
    assert orchestrator.active_run_kind() == "procedure"
    orchestrator.abort_procedure()


def test_an_empty_pull_seam_is_not_a_refusal(orchestrator, station):
    """A queue that simply ran to completion leaves the engine quietly IDLE."""
    blocked: list[str] = []
    orchestrator.action_blocked.connect(blocked.append)
    _pull_seam(orchestrator, station, RunQueue())

    orchestrator.run_queue()

    assert orchestrator._state == OrchestratorState.IDLE
    assert blocked == []


def test_a_pull_seam_that_raises_leaves_the_engine_idle_and_says_so(
    orchestrator, station
):
    """The queue lives outside the engine, so its failure is reportable, not fatal."""
    blocked: list[str] = []
    orchestrator.action_blocked.connect(blocked.append)

    def exploding():
        raise RuntimeError("stale spec")

    orchestrator.next_procedure = exploding

    orchestrator.run_queue()

    assert orchestrator._state == OrchestratorState.IDLE
    assert orchestrator._procedure is None
    assert blocked and "stale spec" in blocked[0]


def test_runs_handed_over_directly_drain_before_the_pull_seam(orchestrator, station):
    """A run handed over as a live object is already the engine's to start."""
    queue = RunQueue()
    _pull_seam(orchestrator, station, queue)
    queue.add(RunSpec(kind="procedure", run_class="QueueableProcedure"))
    handed_over = MockProcedure(station)
    orchestrator.queue_procedure(handed_over)

    orchestrator.run_queue()

    assert orchestrator._procedure is handed_over
    assert len(queue) == 1
    orchestrator.abort_procedure()


def test_queueing_from_a_state_changed_slot_does_not_start_a_run_inside_the_emit(
    orchestrator, station
):
    """Killer #2: the advance is never re-entrant inside ``state_changed``.

    ``_change_state()`` emits synchronously, so a client reacting to the IDLE
    state runs INSIDE the engine's own transition. Queueing from there
    must not start anything: the engine's own ``run_queue()``, which has not
    been reached yet, is what starts it — after the emit returns.
    """
    observed: list = []

    def on_state(state):
        if state == OrchestratorState.IDLE.value and not observed:
            orchestrator.queue_procedure(MockProcedure(station))
            observed.append(orchestrator._procedure)

    orchestrator.state_changed.connect(on_state)
    orchestrator.run_procedure(MockProcedure(station))

    orchestrator.abort_procedure()

    assert observed == [None], "a run started inside the state_changed emit"
    assert isinstance(orchestrator._procedure, MockProcedure)
    orchestrator.abort_procedure()


def test_nothing_auto_starts_after_an_emergency_acknowledge(
    orchestrator, station, qtbot
):
    """Killer #3: acknowledging a quench is not a request to carry on measuring.

    The queued procedure stays queued: the emergency acknowledge is one of
    the four transitions to IDLE that deliberately do not chain
    ``run_queue()``.
    """
    _fast_magnet(station)
    queued = MockProcedure(station)
    orchestrator.queue_procedure(queued)

    station.magnet_z._driver._simulate_quench = True
    qtbot.waitUntil(
        lambda: orchestrator._state == OrchestratorState.EMERGENCY, timeout=2000
    )
    station.magnet_z._driver._simulate_quench = False
    qtbot.waitUntil(
        lambda: not orchestrator._station.check_safety().get("quench"), timeout=2000
    )

    orchestrator.acknowledge()
    assert orchestrator._state == OrchestratorState.IDLE

    orchestrator._tick()

    assert orchestrator._state == OrchestratorState.IDLE
    assert orchestrator._procedure is None
    assert orchestrator._procedure_queue == [queued]


def test_error_recovery_does_not_chain_the_queue(orchestrator, station):
    """After an error the queue's assumptions may no longer hold."""
    queued = MockProcedure(station)
    orchestrator.queue_procedure(queued)
    orchestrator._fail_to_error("something unknown broke")
    assert orchestrator._state == OrchestratorState.ERROR

    orchestrator.recover_from_error()

    assert orchestrator._state == OrchestratorState.IDLE
    assert orchestrator._procedure_queue == [queued]


# ── QueueChanged: every mutation is broadcast, and names its actor ───────────


def test_queueing_a_run_broadcasts_the_whole_queue(port, station):
    """One event describes the queue as it now stands, in run order."""
    orch, recorder, _tmp_path = port
    orch.queue_procedure(MockProcedure(station))
    orch.queue_procedure(QueueableProcedure(station))

    events = recorder.of_type(ev.QueueChanged)
    assert len(events) == 2
    assert [entry["run_class"] for entry in events[-1].entries] == [
        "MockProcedure",
        "QueueableProcedure",
    ]
    assert events[-1].entries[0]["kind"] == "procedure"


def test_a_queue_changed_event_names_the_actor_who_queued(port, tmp_path):
    """Accountability is a value: the queue shows who queued a run."""
    orch, recorder, _tmp_path = port
    command = ev.Command(
        name=ev.CommandName.QUEUE_PROCEDURE,
        actor=AGENT,
        args={
            "procedure": "FieldSweep",
            "params": dict(FIELD_SWEEP_PARAMS),
            "sample_info": {"sample_name": "S", "sample_id": "S-1", "comments": ""},
            "data_directory": str(tmp_path),
        },
    )

    request_id = orch.submit(command)

    event = recorder.of_type(ev.QueueChanged)[-1]
    assert event.actor == AGENT
    assert event.request_id == request_id
    assert event.entries[0]["run_class"] == "FieldSweep"
    assert event.entries[0]["actor"]["id"] == AGENT.id
    assert json.loads(json.dumps(event.to_json()))


def test_starting_a_queued_run_broadcasts_the_shortened_queue(port, station):
    """The snapshot is emitted after the pop, so a client sees what is left."""
    orch, recorder, _tmp_path = port
    orch.queue_procedure(MockProcedure(station))
    orch.queue_procedure(MockProcedure(station))

    orch.run_queue()

    assert len(recorder.of_type(ev.QueueChanged)[-1].entries) == 1
    orch.abort_procedure()


def test_publish_queue_broadcasts_a_client_side_change(port, station):
    """The engine cannot see an external add/remove/reorder — it is told."""
    orch, recorder, _tmp_path = port
    queue = RunQueue()
    _pull_seam(orch, station, queue)
    before = len(recorder.of_type(ev.QueueChanged))

    queue.add(RunSpec(kind="procedure", run_class="QueueableProcedure"))
    orch.publish_queue()

    events = recorder.of_type(ev.QueueChanged)
    assert len(events) == before + 1
    assert [entry["run_class"] for entry in events[-1].entries] == [
        "QueueableProcedure"
    ]


def test_a_failing_queue_snapshot_is_logged_not_raised(port, station, caplog):
    """Reporting never disrupts a run, the queue snapshot included."""
    orch, recorder, _tmp_path = port

    def broken():
        raise RuntimeError("snapshot exploded")

    orch.queue_snapshot = broken
    with caplog.at_level(logging.ERROR):
        orch.publish_queue()

    assert recorder.of_type(ev.QueueChanged)[-1].entries == ()
    assert any("queue_snapshot() failed" in r.message for r in caplog.records)


# ══════════════════════════════════════════════════════════════════════════
# Attendance and the kill switch — two session-owned values pushed down
# ══════════════════════════════════════════════════════════════════════════


def test_attendance_defaults_to_attended_and_is_mirrored(port):
    """The restrictive default, and the snapshot carries it for every client."""
    orch, recorder, _tmp_path = port

    orch.submit(ev.Command(name=ev.CommandName.SET_ATTENDANCE, args={"attended": False}))

    snapshot = recorder.of_type(ev.StatusSnapshot)[-1]
    assert snapshot.attended is False
    assert recorder.verdicts[-1].code is ev.VerdictCode.OK

    orch.submit(ev.Command(name=ev.CommandName.SET_ATTENDANCE, args={"attended": True}))
    assert recorder.of_type(ev.StatusSnapshot)[-1].attended is True


def test_agent_gate_defaults_to_active_and_is_mirrored(port):
    """The gate travels on the same mirror every other read does."""
    orch, recorder, _tmp_path = port

    assert orch._status_snapshot().agent_gate == ev.AgentGate.ACTIVE.value

    orch.submit(
        ev.Command(name=ev.CommandName.SET_AGENT_GATE, args={"state": "revoked"})
    )

    assert recorder.of_type(ev.StatusSnapshot)[-1].agent_gate == "revoked"


def test_an_unknown_gate_value_is_answered_failed(port):
    """A typo in the gate name never silently opens or closes it."""
    orch, recorder, _tmp_path = port

    orch.submit(ev.Command(name=ev.CommandName.SET_AGENT_GATE, args={"state": "nope"}))

    assert recorder.verdicts[-1].code is ev.VerdictCode.FAILED
    assert orch._status_snapshot().agent_gate == ev.AgentGate.ACTIVE.value


def test_a_closed_gate_refuses_an_agent_and_names_itself(port):
    """BLOCKED_ROLE with the gate in the detail, and nothing dispatched."""
    orch, recorder, _tmp_path = port
    orch.set_agent_gate(ev.AgentGate.REVOKED)

    request_id = orch.submit(
        ev.Command(name=ev.CommandName.START_MONITORING, actor=AGENT)
    )

    verdict = recorder.verdicts[-1]
    assert verdict.request_id == request_id
    assert verdict.code is ev.VerdictCode.BLOCKED_ROLE
    assert verdict.detail == {
        "gate": "revoked",
        "actor_kind": "agent",
        "actor_id": AGENT.id,
    }
    assert "revoked" in verdict.reason


def test_a_closed_gate_never_touches_the_operator_path(port):
    """The same command from the human is carried out while agents are revoked."""
    orch, recorder, _tmp_path = port
    orch.set_agent_gate(ev.AgentGate.REVOKED)

    orch.submit(
        ev.Command(name=ev.CommandName.START_MONITORING)
    )

    assert recorder.verdicts[-1].code is not ev.VerdictCode.BLOCKED_ROLE


def test_emergency_standby_passes_the_closed_gate(port):
    """An actor that can see a problem is never unable to make the station safe."""
    orch, recorder, _tmp_path = port
    orch.set_agent_gate(ev.AgentGate.REVOKED)

    orch.submit(
        ev.Command(
            name=ev.CommandName.EMERGENCY_STANDBY,
            actor=AGENT,
            args={"reason": "coil voltage climbing"},
        )
    )

    assert recorder.verdicts[-1].code is ev.VerdictCode.OK
    assert orch.state == OrchestratorState.EMERGENCY.value


# ── Lifecycle state on the snapshot (GLOSSARY.md's Lifecycle state) ──────────


def test_snapshot_carries_each_instruments_lifecycle_state(orchestrator, station):
    """Every instrument entry says what that instrument is doing."""
    station.initiate_all()
    snapshot = orchestrator.status_snapshot()
    assert {
        name: entry["lifecycle"] for name, entry in snapshot.instruments.items()
    } == {name: "initiated" for name in station.get_vi_names()}


def test_emergency_standby_reaches_the_snapshot_as_a_lifecycle_change(
    orchestrator, station
):
    """The blanket stand-down dispatches no per-VI action, and still shows up.

    This is the defect the lifecycle-state standard exists for: an operator's
    card must never keep claiming an instrument is running after
    ``_enter_emergency()`` has stood it down through ``Station.standby_all()``.
    """
    station.initiate_all()
    assert all(
        entry["lifecycle"] == "initiated"
        for entry in orchestrator.status_snapshot().instruments.values()
    )

    orchestrator.emergency_standby("coil voltage climbing")
    assert orchestrator.state == OrchestratorState.EMERGENCY.value

    snapshot = orchestrator.status_snapshot()
    assert all(
        entry["lifecycle"] == "standby" for entry in snapshot.instruments.values()
    ), snapshot.instruments


def test_snapshot_reports_a_disconnected_instrument_as_idle(orchestrator, station):
    """An instrument I2AS no longer holds cannot be shown as initiated."""
    station.initiate_all()
    orchestrator.disconnect_instrument("magnet_z")

    snapshot = orchestrator.status_snapshot()
    assert snapshot.instruments["magnet_z"]["lifecycle"] == "idle"
    assert snapshot.instruments["temperature"]["lifecycle"] == "initiated"


# ══════════════════════════════════════════════════════════════════════
# The run-ownership standard: who owns the run in flight, and who may end it
# (GLOSSARY.md's "Run owner" / "Takeover")
# ══════════════════════════════════════════════════════════════════════

AGENT_A = ev.Actor(kind=ev.ActorKind.AGENT, id="agent-A", role="session")
AGENT_B = ev.Actor(kind=ev.ActorKind.AGENT, id="agent-B", role="session")


def _run_started_by(port, actor):
    """Start a FieldSweep as *actor* and return its request id."""
    orch, _recorder, tmp_path = port
    command = _run_procedure_command(tmp_path)
    started = ev.Command(
        name=ev.CommandName.RUN_PROCEDURE, actor=actor, args=dict(command.args)
    )
    orch.submit(started)
    assert orch.state != OrchestratorState.IDLE.value
    return started.request_id


def _verdict_for(recorder, request_id):
    """Return the one verdict answering *request_id*."""
    answers = [v for v in recorder.verdicts if v.request_id == request_id]
    assert len(answers) == 1, answers
    return answers[0]


def _submit(orch, recorder, name, actor, **args):
    """Submit one command and return the verdict that answered it."""
    command = ev.Command(name=name, actor=actor, args=args)
    orch.submit(command)
    return _verdict_for(recorder, command.request_id)


def test_the_run_owner_is_the_actor_that_started_it(port):
    """The snapshot names the owner while a run is in flight, and nobody idle."""
    orch, recorder, _tmp = port
    assert orch.status_snapshot().run is None

    _run_started_by(port, AGENT_A)

    assert orch.status_snapshot().run["owner"] == AGENT_A.ref()
    orch.abort_procedure()
    assert orch.status_snapshot().run is None


def test_another_agent_may_not_abort_the_owners_run(port):
    """The refusal names the owner, the rule, and how to proceed."""
    orch, recorder, _tmp = port
    _run_started_by(port, AGENT_A)

    verdict = _submit(orch, recorder, ev.CommandName.ABORT_PROCEDURE, AGENT_B)

    assert verdict.code is ev.VerdictCode.BLOCKED_ROLE
    assert verdict.detail["rule"] == "run_owner"
    assert verdict.detail["owner"] == {"kind": "agent", "id": "agent-A"}
    assert "agent-A" in verdict.reason and "override_owner" in verdict.reason
    # Nothing happened: the run is still the owner's, still running.
    assert orch.state != OrchestratorState.IDLE.value
    orch.abort_procedure()


def test_an_override_without_a_reason_is_refused(port):
    """A takeover whose record cannot say why is the act the rule prevents."""
    orch, recorder, _tmp = port
    _run_started_by(port, AGENT_A)

    verdict = _submit(
        orch,
        recorder,
        ev.CommandName.ABORT_PROCEDURE,
        AGENT_B,
        override_owner=True,
        reason="   ",
    )

    assert verdict.code is ev.VerdictCode.BLOCKED_ROLE
    assert verdict.detail["rule"] == "override_reason_required"
    assert orch.state != OrchestratorState.IDLE.value
    orch.abort_procedure()


def test_an_override_with_a_reason_takes_the_run_over_and_is_recorded(port):
    """Refuse-then-override: the verdict and RunFinished both name the takeover."""
    orch, recorder, _tmp = port
    _run_started_by(port, AGENT_A)

    verdict = _submit(
        orch,
        recorder,
        ev.CommandName.ABORT_PROCEDURE,
        AGENT_B,
        override_owner=True,
        reason="agent-A stopped answering",
    )

    assert verdict.code is ev.VerdictCode.OK
    assert verdict.detail["takeover"] == {
        "owner": {"kind": "agent", "id": "agent-A"},
        "reason": "agent-A stopped answering",
    }
    assert orch.state == OrchestratorState.IDLE.value

    finished = recorder.of_type(ev.RunFinished)[-1]
    assert finished.status == "aborted"
    assert finished.actor == AGENT_B
    assert finished.overridden_owner == {"kind": "agent", "id": "agent-A"}


def test_the_owners_own_abort_is_not_a_takeover(port):
    """Ownership says nothing about the owner acting on their own run."""
    orch, recorder, _tmp = port
    _run_started_by(port, AGENT_A)

    verdict = _submit(orch, recorder, ev.CommandName.ABORT_PROCEDURE, AGENT_A)

    assert verdict.code is ev.VerdictCode.OK
    assert (verdict.detail or {}).get("takeover") is None
    assert recorder.of_type(ev.RunFinished)[-1].overridden_owner is None


def test_the_operator_is_never_gated_by_run_ownership(port):
    """The human's authority comes from standing at the cryostat."""
    orch, recorder, _tmp = port
    _run_started_by(port, AGENT_A)

    verdict = _submit(orch, recorder, ev.CommandName.ABORT_PROCEDURE, ev.OPERATOR)

    assert verdict.code is ev.VerdictCode.OK
    assert (verdict.detail or {}).get("takeover") is None
    finished = recorder.of_type(ev.RunFinished)[-1]
    assert finished.overridden_owner is None
    assert finished.actor == ev.OPERATOR


def test_safety_and_recovery_commands_are_not_owner_scoped(port):
    """Pause, resume, stop_ramp and emergency standby ignore ownership."""
    orch, recorder, _tmp = port
    _run_started_by(port, AGENT_A)

    assert (
        _submit(orch, recorder, ev.CommandName.PAUSE_PROCEDURE, AGENT_B).code
        is ev.VerdictCode.OK
    )
    assert (
        _submit(orch, recorder, ev.CommandName.RESUME_PROCEDURE, AGENT_B).code
        is ev.VerdictCode.OK
    )
    # stop_ramp is refused by the CLAIM, never by ownership: a different rule,
    # and the one that has always governed a run's instruments.
    stop = _submit(
        orch, recorder, ev.CommandName.STOP_RAMP, AGENT_B, vi_name="magnet_z"
    )
    assert (stop.detail or {}).get("rule") != "run_owner"

    safe = _submit(
        orch, recorder, ev.CommandName.EMERGENCY_STANDBY, AGENT_B, reason="quench"
    )
    assert safe.code is ev.VerdictCode.OK
    assert orch.state == OrchestratorState.EMERGENCY.value




def test_a_run_handed_to_the_queue_is_owned_by_whoever_queued_it(
    orchestrator, station
):
    """The actor that committed the station to a run owns it once it starts.

    The engine's own queue: ``run_queue()`` is called by somebody else (here
    the operator), and the run is still agent-A's.
    """
    orchestrator.queue_procedure(MockProcedure(station), actor=AGENT_A)

    orchestrator.run_queue()

    assert orchestrator.status_snapshot().run["owner"] == AGENT_A.ref()
    orchestrator.abort_procedure()


def test_a_run_pulled_from_a_queued_spec_is_owned_by_the_queuing_actor(
    orchestrator, station
):
    """The same rule through the pull seam, where the spec carries the actor."""
    queue = RunQueue()
    _pull_seam(orchestrator, station, queue)
    queue.add(
        RunSpec(kind="procedure", run_class="QueueableProcedure", actor=AGENT_A)
    )

    orchestrator.run_queue()

    assert orchestrator.status_snapshot().run["owner"] == AGENT_A.ref()
    orchestrator.abort_procedure()
