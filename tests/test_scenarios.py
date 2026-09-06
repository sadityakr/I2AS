"""Behavior tests for tests/scenarios.py, doubling as its usage examples.

Each test asks one question a facility operator would ask about a hazardous
or degraded state ("what happens if a safety hold trips while a run is in
flight?", "does a quench really lock out everything?") using the scenario
helpers instead of hand-rolling driver flags + qtbot.waitUntil per test.
"""

from __future__ import annotations

import pytest

from i2as.core.orchestrator import Orchestrator, OrchestratorState
from i2as.core.station import build_station

from tests import scenarios


@pytest.fixture
def station():
    return build_station("i2as/configs/sim_cryostat")


@pytest.fixture
def orchestrator(station, qtbot):
    orch = Orchestrator(station, tick_interval_ms=10)
    orch.start_monitoring()
    yield orch
    orch.shutdown()


def test_hold_flag_holds_concerned_vis_and_fails_running_procedure(
    station, orchestrator, qtbot, monkeypatch
):
    """A procedure ramping a magnet, started under a standing safety hold, fails."""
    hold_flag = scenarios.declare_hold_flag(monkeypatch, station)
    scenarios.hold_flag_tripped(hold_flag, orchestrator, qtbot)

    run_finished = []
    orchestrator.run_finished.connect(lambda manifest: run_finished.append(manifest))
    scenarios.running_procedure(station, orchestrator, vi_name="magnet_z", target=1.0)

    def _run_ended() -> bool:
        return orchestrator._state == OrchestratorState.IDLE

    qtbot.waitUntil(_run_ended, timeout=2000)

    snap = scenarios.snapshot(station, orchestrator)
    assert "magnet_z" in snap["held_vis"]
    assert run_finished and run_finished[0]["status"] == "failed"
    assert hold_flag.flag in run_finished[0]["reason"]


def test_quench_blocks_manual_control_of_every_vi(station, orchestrator, qtbot):
    """Quench is critical severity: EMERGENCY refuses every VI, not just the quenched one."""
    scenarios.quench(station, orchestrator, qtbot, magnet_vi="magnet_z")

    snap = scenarios.snapshot(station, orchestrator)
    assert snap["orchestrator_state"] == "EMERGENCY"

    admitted, _reason = orchestrator._manual_action_admissible("temperature")
    assert admitted is False, "EMERGENCY must refuse an unrelated VI too"

    station.get_vi("magnet_z")._driver._simulate_quench = False


def test_disconnect_faults_the_named_vi(station, orchestrator, qtbot):
    """A disconnected instrument shows up as a comm fault, not a safety hold."""
    orchestrator._tick()  # one successful poll first, so the fault is a transition
    scenarios.disconnect(station, orchestrator, qtbot, "temperature")

    snap = scenarios.snapshot(station, orchestrator)
    assert "temperature" in snap["faulted_vis"]
    assert "not_responding" in orchestrator._station.availability("temperature").tags

    station.get_vi("temperature")._driver._simulate_error = False


def test_measurement_instrument_returns_error_instead_of_data(station, orchestrator, qtbot):
    """A measurement VI raises at the call site, with no station-wide fault at idle.

    Unlike a system VI (test_disconnect_faults_the_named_vi), a
    measurement VI is not polled every tick — the fault only exists once
    something actually tries to use it, mirroring how a running procedure's
    sample() would see it.
    """
    from i2as.core.exceptions import I2ASCommunicationError

    vi = station.get_vi("dc_measurement")
    vi.initiate_measurement(current_A=1e-6)  # arm while healthy
    scenarios.measurement_error(station, "dc_measurement", driver_attr="_meter")

    with pytest.raises(I2ASCommunicationError):
        vi.take_reading()

    snap = scenarios.snapshot(station, orchestrator)
    assert "dc_measurement" not in snap["faulted_vis"], (
        "a measurement VI's fault is not a station-wide condition until "
        "something reads it"
    )

    vi._meter._simulate_error = False
