"""Test suite for simulated drivers (Layer 0b)."""

import time

import pytest


class TestSimOxfordIPS120:
    """Tests for SimOxfordIPS120 magnet power supply."""

    def test_contract_single_string_init(self):
        """Driver accepts a single string argument."""
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        driver = SimOxfordIPS120("GPIB0::25::INSTR")
        assert driver is not None

    def test_initial_state(self):
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        assert d.get_current() == pytest.approx(0.0)
        assert d.get_current_setpoint() == pytest.approx(0.0)
        assert d.get_status() == "HOLD"

    def test_return_types(self):
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        assert isinstance(d.get_current(), float)
        assert isinstance(d.get_current_setpoint(), float)
        assert isinstance(d.get_status(), str)

    def test_ramp_starts_on_setpoint_change(self):
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        d.set_current_setpoint(10.0)
        assert d.get_status() == "RAMPING"

    def test_ramp_completes(self):
        """Ramp from 0 to a small target completes with enough time."""
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        d.set_ramp_rate(600.0)  # Very fast: 600 A/min for testing
        d.set_current_setpoint(1.0)
        # Simulate passage of time
        d._last_update = time.time() - 1.0  # 1 second ago
        d._update_simulation()
        assert d.get_current() == pytest.approx(1.0)
        assert d.get_status() == "HOLD"

    def test_ramp_direction_negative(self):
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        d._current = 10.0
        d.set_ramp_rate(600.0)
        d.set_current_setpoint(0.0)
        d._last_update = time.time() - 2.0
        d._update_simulation()
        assert d.get_current() == pytest.approx(0.0)

    def test_hold_when_no_ramp(self):
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        d._last_update = time.time() - 10.0
        d._update_simulation()
        assert d.get_current() == pytest.approx(0.0)
        assert d.get_status() == "HOLD"

    def test_setpoint_clamping_above_max(self):
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        d.set_current_setpoint(200.0)  # Above 90 A max
        assert d.get_current_setpoint() == pytest.approx(90.0)

    def test_setpoint_clamping_below_min(self):
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        d.set_current_setpoint(-200.0)  # Below -90 A min
        assert d.get_current_setpoint() == pytest.approx(-90.0)

    def test_simulate_error_raises(self):
        from i2as.core.exceptions import I2ASCommunicationError
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        d._simulate_error = True
        with pytest.raises(I2ASCommunicationError):
            d.get_current()

    def test_quench_status(self):
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        d._simulate_quench = True
        assert d.get_status() == "QUENCH"

    def test_clamped_refuses_ramp(self):
        """set_current_setpoint() raises while CLAMPED (mirrors real PSU denial)."""
        from i2as.core.exceptions import I2ASSafetyError
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        d._simulate_clamp = True
        with pytest.raises(I2ASSafetyError):
            d.set_current_setpoint(10.0)
        assert d.get_current_setpoint() == pytest.approx(0.0)  # write never applied

    def test_clear_clamp_allows_ramp_again(self):
        """clear_clamp() unclamps; a subsequent ramp then succeeds."""
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        d._simulate_clamp = True
        d.clear_clamp()
        assert d._simulate_clamp is False
        assert d.get_status() == "HOLD"
        d.set_current_setpoint(10.0)
        assert d.get_status() == "RAMPING"

    def test_clear_clamp_when_not_clamped_raises(self):
        """clear_clamp() is not a silent no-op when nothing is clamped."""
        from i2as.core.exceptions import I2ASSafetyError
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        with pytest.raises(I2ASSafetyError):
            d.clear_clamp()

    def test_get_status_never_clears_clamp(self):
        """Reading status must never have a write side effect (see driver docstring)."""
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        d._simulate_clamp = True
        for _ in range(5):
            d.get_status()
        assert d._simulate_clamp is True  # still clamped after repeated polling

    def test_no_ramp_when_setpoint_at_current(self):
        """No state transition when setpoint equals current (within 0.01 A)."""
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        d.set_current_setpoint(0.005)  # Difference < 0.01
        assert d.get_status() == "HOLD"

    def test_get_ramp_rate_reads_back_set_value(self):
        """get_ramp_rate() reflects the value passed to set_ramp_rate()."""
        from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

        d = SimOxfordIPS120("SIM")
        d.set_ramp_rate(12.5)
        assert d.get_ramp_rate() == pytest.approx(12.5)


class TestSimKeithley6221:
    """Tests for SimKeithley6221 current source."""

    def test_contract_single_string_init(self):
        from i2as.drivers.sim_keithley_6221 import SimKeithley6221

        d = SimKeithley6221("SIM")
        assert d is not None

    def test_source_enable_disable(self):
        from i2as.drivers.sim_keithley_6221 import SimKeithley6221

        d = SimKeithley6221("SIM")
        d.set_source_enabled(True)
        assert d.get_source_enabled() is True
        d.set_source_enabled(False)
        assert d.get_source_enabled() is False

    def test_configure_delta_mode(self):
        from i2as.drivers.sim_keithley_6221 import SimKeithley6221

        d = SimKeithley6221("SIM")
        d.configure_delta_mode(high_current=1e-6, n_readings=100, delay=0.01)
        # Should not raise

    def test_delta_readings_returns_list(self):
        from i2as.drivers.sim_keithley_6221 import SimKeithley6221

        d = SimKeithley6221("SIM")
        d.configure_delta_mode(high_current=1e-6, n_readings=50, delay=0.001)
        d.trigger_delta_mode()
        readings = d.get_delta_readings()
        assert isinstance(readings, list)
        assert len(readings) == 50

    def test_delta_readings_are_floats(self):
        from i2as.drivers.sim_keithley_6221 import SimKeithley6221

        d = SimKeithley6221("SIM")
        d.configure_delta_mode(high_current=1e-6, n_readings=10, delay=0.001)
        d.trigger_delta_mode()
        readings = d.get_delta_readings()
        assert all(isinstance(v, float) for v in readings)

    def test_current_get_set(self):
        from i2as.drivers.sim_keithley_6221 import SimKeithley6221

        d = SimKeithley6221("SIM")
        d.set_current(1e-3)
        assert d.get_current() == pytest.approx(1e-3)

    def test_delta_readings_with_paired_meter(self):
        """When a 2182A is paired, readings come from its get_voltage()."""
        from i2as.drivers.sim_keithley_2182a import SimKeithley2182A
        from i2as.drivers.sim_keithley_6221 import SimKeithley6221

        meter = SimKeithley2182A("SIM")
        meter._base_voltage = 2.5e-6
        meter._noise_std = 0.0  # Zero noise: deterministic

        source = SimKeithley6221("SIM")
        source._paired_meter = meter
        source.configure_delta_mode(high_current=1e-6, n_readings=5, delay=0.001)
        source.trigger_delta_mode()
        readings = source.get_delta_readings()
        assert all(v == pytest.approx(2.5e-6) for v in readings)

    def test_simulate_error_raises(self):
        from i2as.core.exceptions import I2ASCommunicationError
        from i2as.drivers.sim_keithley_6221 import SimKeithley6221

        d = SimKeithley6221("SIM")
        d._simulate_error = True
        with pytest.raises(I2ASCommunicationError):
            d.get_source_enabled()


class TestSimKeithley2182A:
    """Tests for SimKeithley2182A nanovoltmeter."""

    def test_contract_single_string_init(self):
        from i2as.drivers.sim_keithley_2182a import SimKeithley2182A

        d = SimKeithley2182A("SIM")
        assert d is not None

    def test_voltage_returns_float(self):
        from i2as.drivers.sim_keithley_2182a import SimKeithley2182A

        d = SimKeithley2182A("SIM")
        v = d.get_voltage()
        assert isinstance(v, float)

    def test_range_setting(self):
        from i2as.drivers.sim_keithley_2182a import SimKeithley2182A

        d = SimKeithley2182A("SIM")
        d.set_range(1.0)
        assert d.get_range() == pytest.approx(1.0)

    def test_voltage_has_noise(self):
        """Multiple voltage readings differ due to Gaussian noise."""
        from i2as.drivers.sim_keithley_2182a import SimKeithley2182A

        d = SimKeithley2182A("SIM")
        readings = [d.get_voltage() for _ in range(20)]
        # With non-zero noise_std, readings should not all be identical
        assert len(set(readings)) > 1

    def test_voltage_near_base(self):
        """Voltage readings are within a few standard deviations of base."""
        from i2as.drivers.sim_keithley_2182a import SimKeithley2182A

        d = SimKeithley2182A("SIM")
        readings = [d.get_voltage() for _ in range(50)]
        mean = sum(readings) / len(readings)
        # Mean should be within 10 noise_std of base_voltage
        assert abs(mean - d._base_voltage) < 10 * d._noise_std

    def test_simulate_error_raises(self):
        from i2as.core.exceptions import I2ASCommunicationError
        from i2as.drivers.sim_keithley_2182a import SimKeithley2182A

        d = SimKeithley2182A("SIM")
        d._simulate_error = True
        with pytest.raises(I2ASCommunicationError):
            d.get_voltage()
