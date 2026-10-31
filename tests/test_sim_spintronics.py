"""The simulated spintronics rack: coupled sims, sample physics, bias sweep.

Covers what the conformance suite cannot see: that the sims on one
``SimEnvironment`` really are coupled (the field and temperature a p-MTJ and a
Hall bar respond to come from the magnet and the cryostat), that the sample
models produce the physics their docstrings promise (TMR magnitude and its
bias and temperature dependence, switching fields, a square anomalous-Hall
loop), that the cryostat's PID loop holds a setpoint once tuned, and that
``BiasSweep`` steps the measurement's own bias through the Station.
"""

from __future__ import annotations

import numpy as np
import pytest

import i2as.drivers.sim_cryostat_lakeshore_335 as cryostat_module
import i2as.drivers.sim_hall_keithley_2182a as hall_module
import i2as.drivers.sim_keithley_2400 as smu_module
from i2as.core import sim_environment, sim_samples
from i2as.core.station import build_station
from i2as.drivers.sim_cryostat_lakeshore_335 import SimCryostatLakeshore335
from i2as.drivers.sim_hall_keithley_2182a import OVERFLOW_READING_V, SimHallKeithley2182A
from i2as.drivers.sim_keithley_2400 import SimKeithley2400
from i2as.drivers.sim_keithley_6221 import SimKeithley6221
from i2as.procedures.bias_sweep import BiasSweep

SPIN_CONFIG = "i2as/configs/sim_spintronics"


@pytest.fixture(autouse=True)
def _instant_conversions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the sims' real conversion times so the physics tests run fast."""
    monkeypatch.setattr(smu_module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(hall_module.time, "sleep", lambda _s: None)


def _environment(name: str, temperature_K: float) -> sim_environment.SimEnvironment:
    """Return a fresh named environment pinned at *temperature_K*."""
    sim_environment._registry.pop(name, None)
    environment = sim_environment.get(name)
    environment.temperature_source = lambda: temperature_K
    return environment


def _resistance(smu: SimKeithley2400, bias_V: float, n: int = 20) -> float:
    smu.set_source_voltage(bias_V)
    return bias_V / float(np.mean([smu.measure_current() for _ in range(n)]))


def _sweep_loop(smu: SimKeithley2400, environment, bias_V: float, limit_T: float = 0.1):
    """Sweep +limit -> -limit -> +limit and return (fields, resistances)."""
    fields = np.r_[np.linspace(limit_T, -limit_T, 161), np.linspace(-limit_T, limit_T, 161)]
    resistances = []
    for field in fields:
        environment.applied_field_T = field
        resistances.append(_resistance(smu, bias_V, n=3))
    return fields, np.array(resistances)


def test_mtj_tmr_and_switching_at_10_k() -> None:
    """At 10 K and low bias: TMR ≈ 186 %, switching at ≈ +37 / −50 mT."""
    environment = _environment("t_mtj_cold", 10.0)
    smu = SimKeithley2400("SIM::K2400@t_mtj_cold")
    smu.set_current_compliance(0.01)
    smu.set_output(True)
    fields, resistances = _sweep_loop(smu, environment, 0.02)
    r_p, r_ap = resistances.min(), resistances.max()
    assert (r_ap - r_p) / r_p == pytest.approx(1.84, abs=0.08)
    high = resistances > 0.5 * (r_p + r_ap)
    down = fields[:161][high[:161]]
    up = fields[161:][~high[161:]]
    assert down.max() == pytest.approx(-0.050, abs=0.004)  # P -> AP on the way down
    assert up.min() == pytest.approx(0.037, abs=0.004)  # AP -> P on the way up


def test_mtj_tmr_falls_with_bias_and_temperature() -> None:
    """TMR halves near 0.5 V and is ≈ 120 % at room temperature."""
    mtj = sim_samples.PerpendicularMTJ(_environment("t_mtj_params", 300.0))
    assert mtj.tmr_zero_bias(300.0) == pytest.approx(1.20, abs=0.03)
    assert mtj.tmr_zero_bias(10.0) == pytest.approx(1.85, abs=0.03)
    assert mtj.coercivity(10.0, 0.6) < mtj.coercivity(10.0, 0.05)
    assert mtj.coercivity(300.0, 0.05) < mtj.coercivity(10.0, 0.05)


def test_mtj_iv_is_strongly_non_linear() -> None:
    """The tunnelling I–V bends upwards: dI/dV at 0.8 V is several times zero-bias."""
    environment = _environment("t_mtj_iv", 10.0)
    environment.applied_field_T = 0.1
    smu = SimKeithley2400("SIM::K2400@t_mtj_iv")
    smu.set_current_compliance(0.01)
    smu.set_output(True)
    assert _resistance(smu, 0.8) < 0.5 * _resistance(smu, 0.01)


def test_smu_clips_at_compliance_and_reports_it() -> None:
    environment = _environment("t_smu_cmpl", 10.0)
    environment.applied_field_T = 0.1
    smu = SimKeithley2400("SIM::K2400@t_smu_cmpl")
    smu.set_current_compliance(1e-4)
    smu.set_output(True)
    smu.set_source_voltage(0.8)
    assert smu.measure_current() == pytest.approx(1e-4)
    assert smu.is_in_compliance()


def test_hall_bar_square_loop_follows_the_6221_current() -> None:
    """The 2182A reads I x R_AHE: a square loop, ±2.4 Ω at 10 K, Hc ≈ 14 mT."""
    environment = _environment("t_hall", 10.0)
    source = SimKeithley6221("SIM::K6221@t_hall")
    meter = SimHallKeithley2182A("SIM::K2182A@t_hall")
    source.set_current(1e-4)
    fields = np.r_[np.linspace(0.04, -0.04, 161), np.linspace(-0.04, 0.04, 161)]
    r_xy = []
    for field in fields:
        environment.applied_field_T = field
        r_xy.append(np.mean([meter.get_voltage() for _ in range(5)]) / 1e-4)
    r_xy = np.array(r_xy)
    amplitude = 0.5 * (r_xy.max() - r_xy.min())
    assert amplitude == pytest.approx(2.42, abs=0.1)
    centre = 0.5 * (r_xy.max() + r_xy.min())
    down_switch = fields[:161][np.argmax(r_xy[:161] < centre)]
    assert down_switch == pytest.approx(-0.0145, abs=0.002)
    # No current, no Hall signal — only the thermal EMF and offsets.
    source.set_current(0.0)
    assert abs(meter.get_voltage()) < 5e-6


def test_hall_meter_overflows_on_too_small_a_range() -> None:
    _environment("t_hall_ovr", 10.0)
    source = SimKeithley6221("SIM::K6221@t_hall_ovr")
    meter = SimHallKeithley2182A("SIM::K2182A@t_hall_ovr")
    source.set_current(1e-3)
    meter.set_range(0.001)
    assert meter.get_voltage() == OVERFLOW_READING_V


def test_cryostat_holds_10_k_once_tuned(monkeypatch: pytest.MonkeyPatch) -> None:
    """From 15 K, the documented 10 K PID settles within ±10 mK in a few minutes."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(cryostat_module.time, "time", lambda: clock["t"])
    sim_environment._registry.pop("t_cryo", None)
    controller = SimCryostatLakeshore335("SIM::LS335?T0=15@t_cryo")
    controller._last_update = clock["t"]
    assert controller.get_heater_range() == "LOW"
    controller.set_proportional_band(1.667)
    controller.set_integral_action_time(0.0333)
    controller.set_derivative_action_time(0.0167)
    controller.set_setpoint(10.0)
    readings = []
    for _ in range(240):
        clock["t"] += 1.0
        readings.append(controller.get_temperature())
    tail = np.array(readings[-60:])
    assert abs(tail.mean() - 10.0) < 0.01
    assert tail.std() < 0.01
    # The sample, coupled through the environment, has followed.
    assert sim_environment.get("t_cryo").sample_temperature_K == pytest.approx(10.0, abs=0.02)


def test_bias_sweep_steps_the_bias_through_the_station(tmp_path) -> None:
    """Every point dispatches set_bias_voltage with that point's bias."""
    station = build_station(SPIN_CONFIG)
    proc = BiasSweep(
        station,
        {"sample_name": "S", "sample_id": "S-1", "comments": ""},
        str(tmp_path),
        measurement_vi="mtj_bias",
        bias_start=-0.2,
        bias_end=0.2,
        bias_steps=5,
        compliance_A=0.01,
        readings_per_point=1,
    )
    sent = []
    original = station.send_measurement_commands

    def _record(commands):
        sent.extend(commands)
        return original(commands)

    station.send_measurement_commands = _record  # type: ignore[method-assign]
    plan = proc.initiate()
    station.send_measurement_commands(plan.commands)
    sent.clear()
    biases = []
    while True:
        proc.measure()
        biases.append(sent[-1].kwargs["bias_V"])
        if proc.change_sweep_step() is None:
            break
    proc.standby()
    assert biases == pytest.approx([-0.2, -0.1, 0.0, 0.1, 0.2])
    assert all(command.method == "set_bias_voltage" for command in sent)


def test_bias_sweep_refuses_a_measurement_without_a_bias(tmp_path) -> None:
    from i2as.core.exceptions import I2ASConfigError

    station = build_station(SPIN_CONFIG)
    proc = BiasSweep(
        station,
        {"sample_name": "S", "sample_id": "S-1", "comments": ""},
        str(tmp_path),
        measurement_vi="hall",
    )
    with pytest.raises(I2ASConfigError, match="bias_V"):
        proc.initiate()


def test_bias_ceiling_refuses_a_breakdown_bias() -> None:
    """The setup's max_bias_V stops a bias that would destroy the barrier."""
    station = build_station(SPIN_CONFIG)
    vi = station.get_vi("mtj_bias")
    assert vi._limits["bias_V"] == (-1.0, 1.0)
