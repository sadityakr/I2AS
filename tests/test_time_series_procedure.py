"""Behaviour tests for the TimeSeries procedure (L4).

Covers the three things that make it different from the two sweep
procedures: it commands no system hardware, it claims only the reading
path, and its axis is elapsed time with an end condition that can stop the
run before the schedule runs out.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from i2as.core.exceptions import I2ASConfigError
from i2as.core.plan import PhasePlan, StepPlan
from i2as.core.station import build_station
from i2as.procedures.time_series import TimeSeries

CONFIG_PATH = "i2as/configs/sim_cryostat"

SAMPLE_INFO = {
    "sample_name": "Test Sample",
    "sample_id": "T-TS-001",
    "comments": "automated test",
}

DELTA = {
    "measurement_vi": "dc_measurement",
    "current_A": 1e-6,
    "readings_per_point": 5,
    "voltmeter_range_V": 0.01,
    "compliance_V": 1.0,
    "delay_s": 0.01,
    "compliance_abort": True,
    "cold_switch": False,
}
DC = {
    "measurement_vi": "dc_measurement",
    "current_A": 1e-6,
    "compliance_A": 1e-3,
    "voltmeter_range_V": 0.1,
    "readings_per_point": 5,
}
MEAS = [pytest.param(DELTA, id="delta"), pytest.param(DC, id="dc")]

# A short, fast schedule: 4 points at 0 s cadence so no test ever sleeps.
FAST = {"step_time_s": 0.001, "max_duration_s": 0.003}


@pytest.fixture
def station():
    return build_station(CONFIG_PATH)


def _proc(station, tmp_path, meas=DC, **overrides):
    params = {**FAST, **meas, **overrides}
    return TimeSeries(
        station=station,
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
        **params,
    )


def _arm(station, proc):
    """Arm the measurement VI directly (normally done via the Orchestrator)."""
    station.get_vi(proc._measurement_vi).initiate_measurement(**proc._measurement_params)


def _fix_temperature(station, value: float):
    """Pin the VTI's monitored temperature so an end condition is deterministic."""
    station.temperature.temperature = lambda: value


# ── The schedule ─────────────────────────────────────────────────────────────


def test_schedule_is_elapsed_time_from_zero(station, tmp_path):
    proc = _proc(station, tmp_path, step_time_s=10.0, max_duration_s=60.0)
    assert proc.get_sweep_array() == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


def test_schedule_never_exceeds_the_maximum_duration(station, tmp_path):
    """A duration that is not a whole number of steps stops short, never over."""
    proc = _proc(station, tmp_path, step_time_s=10.0, max_duration_s=25.0)
    assert proc.get_sweep_array() == [0.0, 10.0, 20.0]


def test_non_positive_step_time_is_refused(station, tmp_path):
    with pytest.raises(I2ASConfigError, match="step_time_s"):
        _proc(station, tmp_path, step_time_s=0.0)


def test_axis_column_is_elapsed_seconds(station, tmp_path):
    proc = _proc(station, tmp_path)
    assert TimeSeries.axis_data_key() == "elapsed_s"
    assert TimeSeries.sweep_axis is None  # no linear/segments/CSV widget in the GUI
    assert "elapsed_s" in proc.get_data_keys()


# ── Commands nothing ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("meas", MEAS)
def test_initiate_commands_no_system_hardware(station, tmp_path, meas):
    """The opening plan arms the measurement VI and targets nothing else."""
    proc = _proc(station, tmp_path, meas)
    plan = proc.initiate()
    try:
        assert isinstance(plan, PhasePlan)
        assert plan.targets == {}
        assert plan.wait_s == pytest.approx(0.0)  # first reading on the next tick
        assert [c.vi_name for c in plan.commands] == [meas["measurement_vi"]]
        assert plan.commands[0].method == "initiate_measurement"
        # Narrowed claim (see test_claims_only_the_reading_path below): only
        # the measurement VI gets claim-initiated, never magnet_z or
        # temperature, which stay exactly as the operator left them.
        assert [c.vi_name for c in plan.claim_commands] == [meas["measurement_vi"]]
        assert plan.claim_commands[0].method == "initiate"
    finally:
        proc.standby()


def test_steps_and_standby_carry_no_targets(station, tmp_path):
    proc = _proc(station, tmp_path)
    proc.initiate()
    step = proc.change_sweep_step()
    assert isinstance(step, StepPlan)
    assert step.targets == {}
    assert proc.standby().targets == {}


def test_claims_only_the_reading_path(station, tmp_path):
    """Every non-measurement VI stays available for manual front-panel control."""
    proc = _proc(station, tmp_path, DC)
    claimed = proc.claimed_vi_names()
    assert claimed == {"dc_measurement"}
    for vi_name in ("magnet_z", "temperature"):
        assert vi_name not in claimed


# ── End condition: elapsed time ──────────────────────────────────────────────


def test_elapsed_time_run_stops_at_the_end_of_the_schedule(station, tmp_path):
    proc = _proc(station, tmp_path, step_time_s=1.0, max_duration_s=2.0)
    proc.initiate()
    assert proc.change_sweep_step() is not None  # point 2 of 3
    assert proc.change_sweep_step() is not None  # point 3 of 3
    assert proc.change_sweep_step() is None
    proc.standby()


# ── End condition: a watched channel ─────────────────────────────────────────


def test_watched_channel_stops_the_run_on_the_way_up(station, tmp_path):
    """Starting below the threshold, the run ends when the channel rises to it."""
    _fix_temperature(station, 10.0)
    proc = _proc(station, tmp_path, end_condition="temperature", end_value=50.0)
    proc.initiate()

    assert proc.change_sweep_step() is not None    # still cold
    _fix_temperature(station, 49.9)
    assert proc.change_sweep_step() is not None    # nearly there
    _fix_temperature(station, 50.1)
    assert proc.change_sweep_step() is None        # reached
    proc.standby()


def test_watched_channel_stops_the_run_on_the_way_down(station, tmp_path):
    """Direction comes from the baseline read at initiate(), with no extra parameter."""
    _fix_temperature(station, 300.0)
    proc = _proc(station, tmp_path, end_condition="temperature", end_value=100.0)
    proc.initiate()
    assert proc._approach_sign == pytest.approx(-1.0)

    _fix_temperature(station, 150.0)
    assert proc.change_sweep_step() is not None
    _fix_temperature(station, 99.0)
    assert proc.change_sweep_step() is None
    proc.standby()


def test_tolerance_stops_an_asymptotic_approach(station, tmp_path):
    """A channel that never quite crosses still ends the run within tolerance."""
    _fix_temperature(station, 10.0)
    proc = _proc(
        station,
        tmp_path,
        end_condition="temperature",
        end_value=300.0,
        end_tolerance=0.5,
    )
    proc.initiate()

    _fix_temperature(station, 299.4)
    assert proc.change_sweep_step() is not None
    _fix_temperature(station, 299.6)   # inside tolerance, never crosses
    assert proc.change_sweep_step() is None
    proc.standby()


def test_channel_already_past_the_threshold_yields_one_point(station, tmp_path):
    """Documented behaviour: one measurement, then the condition is already met."""
    _fix_temperature(station, 300.0)
    proc = _proc(station, tmp_path, end_condition="temperature", end_value=300.0)
    proc.initiate()
    assert proc.change_sweep_step() is None
    proc.standby()


def test_max_duration_caps_a_watched_run(station, tmp_path):
    """The schedule always applies, even when the channel never gets there."""
    _fix_temperature(station, 10.0)
    proc = _proc(
        station,
        tmp_path,
        step_time_s=1.0,
        max_duration_s=2.0,
        end_condition="temperature",
        end_value=300.0,
    )
    proc.initiate()
    assert proc.change_sweep_step() is not None
    assert proc.change_sweep_step() is not None
    assert proc.change_sweep_step() is None      # capped, condition never met
    proc.standby()


def test_watched_channel_without_a_vi_is_refused_at_construction(station, tmp_path):
    """A station missing the watched instrument fails now, not silently mid-run."""
    station._virtual_instruments.pop("temperature")
    station._vi_registry.pop("temperature")
    with pytest.raises(I2ASConfigError, match="temperature"):
        _proc(station, tmp_path, end_condition="temperature", end_value=4.0)


# ── Cadence ──────────────────────────────────────────────────────────────────


def test_step_wait_counts_from_the_scheduled_instant(station, tmp_path):
    """A slow reading is absorbed instead of pushing every later point back."""
    proc = _proc(station, tmp_path, step_time_s=100.0, max_duration_s=300.0)
    proc.initiate()
    proc._t0 -= 30.0  # pretend 30 s of the first interval has already gone

    assert proc.change_sweep_step().wait_s == pytest.approx(70.0, abs=1.0)
    proc.standby()


def test_step_wait_is_zero_once_the_instant_has_passed(station, tmp_path):
    proc = _proc(station, tmp_path, step_time_s=1.0, max_duration_s=10.0)
    proc.initiate()
    proc._t0 -= 60.0  # far behind schedule

    assert proc.change_sweep_step().wait_s == pytest.approx(0.0)
    proc.standby()


# ── Data ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("meas", MEAS)
def test_measure_saves_elapsed_time_and_measurement_columns(station, tmp_path, meas):
    proc = _proc(station, tmp_path, meas)
    proc.initiate()
    _arm(station, proc)
    proc.measure()
    filepath = proc._data_manager.filepath
    proc.standby()

    with h5py.File(filepath, "r") as f:
        elapsed = f["data"]["elapsed_s"][0]
        assert not np.isnan(elapsed)
        assert elapsed >= 0.0
        assert not np.isnan(f["data"]["voltage_V"][0, 0, 0])
        # System state is recorded even though the run commands none of it.
        assert "magnet_z_magnet_field_T" in f["data"]
