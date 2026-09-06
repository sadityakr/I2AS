"""Unit tests for SimLakeshore335 driver."""

from __future__ import annotations

import time
import pytest

from i2as.core.exceptions import I2ASCommunicationError
from i2as.drivers.sim_lakeshore_335 import SimLakeshore335


def test_idn():
    d = SimLakeshore335("SIM")
    assert d.get_idn() == "LSCI,MODEL335,SIM,1.0"


def test_setpoint():
    d = SimLakeshore335("SIM")
    assert d.get_setpoint() == 0.0
    d.set_setpoint(150.5)
    assert d.get_setpoint() == 150.5
    with pytest.raises(ValueError):
        d.set_setpoint(-5.0)


def test_heater_mode():
    d = SimLakeshore335("SIM")
    assert d.get_heater_mode() == "AUTO"  # Default
    d.set_heater_mode("MANUAL")
    assert d.get_heater_mode() == "MANUAL"
    d.set_heater_mode("AUTO")
    assert d.get_heater_mode() == "AUTO"
    with pytest.raises(ValueError):
        d.set_heater_mode("INVALID")


def test_heater_output_limits():
    d = SimLakeshore335("SIM")
    d.set_heater_mode("MANUAL")
    d.set_heater_range("HIGH")  # heater range must be on for output to apply
    d.set_heater_output(50.0)
    assert d.get_heater_output() == pytest.approx(50.0)

    d.set_heater_output(120.0)
    assert d.get_heater_output() == pytest.approx(99.9)
    d.set_heater_output(-10.0)
    assert d.get_heater_output() == pytest.approx(0.0)


def test_heater_range():
    d = SimLakeshore335("SIM")
    assert d.get_heater_range() == "OFF"  # power-up default
    for setting in ("LOW", "MEDIUM", "HIGH", "OFF"):
        d.set_heater_range(setting)
        assert d.get_heater_range() == setting
    with pytest.raises(ValueError):
        d.set_heater_range("INVALID")


def test_heater_stays_off_while_range_is_off():
    """Regression: even in AUTO mode with a valid setpoint, or with a manual
    output value commanded, the heater must deliver no power while its range
    is 'OFF' (the instrument's power-up default) — this was the reported bug."""
    d = SimLakeshore335("SIM")
    assert d.get_heater_range() == "OFF"

    # AUTO mode with a large setpoint error: still no power.
    d.set_heater_mode("AUTO")
    d.set_setpoint(300.0)
    d._temperature = 4.2
    assert d.get_heater_output() == pytest.approx(0.0)

    # MANUAL mode with a commanded output: still no power.
    d.set_heater_mode("MANUAL")
    d.set_heater_output(80.0)
    assert d.get_heater_output() == pytest.approx(0.0)

    # Turning the range on lets manual output apply.
    d.set_heater_range("HIGH")
    assert d.get_heater_output() == pytest.approx(80.0)


def test_pid_clamping():
    d = SimLakeshore335("SIM")
    
    # Proportional
    assert d.get_proportional_band() == pytest.approx(90.0)
    d.set_proportional_band(120.5)
    assert d.get_proportional_band() == pytest.approx(120.5)
    d.set_proportional_band(2000.0)
    assert d.get_proportional_band() == pytest.approx(1000.0)
    d.set_proportional_band(-5.0)
    assert d.get_proportional_band() == pytest.approx(0.0)

    # Integral
    assert d.get_integral_action_time() == pytest.approx(50.0)
    d.set_integral_action_time(150.0)
    assert d.get_integral_action_time() == pytest.approx(150.0)
    d.set_integral_action_time(1200.0)
    assert d.get_integral_action_time() == pytest.approx(1000.0)
    d.set_integral_action_time(-10.0)
    assert d.get_integral_action_time() == pytest.approx(0.0)

    # Derivative
    assert d.get_derivative_action_time() == pytest.approx(0.0)
    d.set_derivative_action_time(85.0)
    assert d.get_derivative_action_time() == pytest.approx(85.0)
    d.set_derivative_action_time(300.0)
    assert d.get_derivative_action_time() == pytest.approx(200.0)
    d.set_derivative_action_time(-5.0)
    assert d.get_derivative_action_time() == pytest.approx(0.0)


def test_auto_pid():
    d = SimLakeshore335("SIM")
    assert d.get_auto_pid() is False
    d.set_auto_pid(True)
    assert d.get_auto_pid() is True
    d.set_auto_pid(False)
    assert d.get_auto_pid() is False


def test_simulation_evolution():
    d = SimLakeshore335("SIM")
    d.set_heater_mode("MANUAL")
    d.set_heater_range("HIGH")  # heater range must be on for output to apply
    d.set_heater_output(50.0)
    # T_target = 4.2 + (50 / 99.9) * 295.8 = ~152.2 K
    d._temperature = 300.0
    d._last_update = time.time() - 3600.0  # 1 hour
    assert d.get_temperature() == pytest.approx(152.2, abs=0.5)


def test_error_injection():
    d = SimLakeshore335("SIM")
    d._simulate_error = True
    with pytest.raises(I2ASCommunicationError):
        d.get_temperature()
    with pytest.raises(I2ASCommunicationError):
        d.get_idn()


def test_sensor_curve_selection():
    d = SimLakeshore335("SIM")
    assert d.get_sensor_curve("A") == 22
    assert d.get_sensor_curve("B") == 2

    # Slots 21 and 23 are USER slots, so they must hold a curve before the
    # instrument will accept them (see test_user_curve_slot_must_be_loaded).
    d._loaded_user_curves.update({21, 23})

    d.set_sensor_curve(21, "A")
    assert d.get_sensor_curve("A") == 21

    d.set_sensor_curve(23, "B")
    assert d.get_sensor_curve("B") == 23

    with pytest.raises(ValueError):
        d.set_sensor_curve(60, "A")
    with pytest.raises(ValueError):
        d.set_sensor_curve(-1, "A")
    with pytest.raises(ValueError):
        d.set_sensor_curve(21, "C")
