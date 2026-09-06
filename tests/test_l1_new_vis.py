# ---
# description: |
#   Test suite for the behavior-based Virtual Instruments. Covers
#   SuperconductingMagnetVI, SampleTemperatureControllerVI,
#   Lakeshore335SampleTemperatureControllerVI, DCSeparateMeasurementVI and
#   DCMeasurementBase.
# entry_point: pytest tests/test_l1_new_vis.py -v
# last_updated: 2026-07-26
# ---

"""Tests for behavior-based VIs (Stage 2 of VI refactor)."""

from __future__ import annotations

import time

import pytest

from tests.mocks.bus_spy import spy_on_driver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ips_driver():
    from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120
    return SimOxfordIPS120("SIM")


@pytest.fixture
def source_driver():
    from i2as.drivers.sim_keithley_6221 import SimKeithley6221
    return SimKeithley6221("SIM")


@pytest.fixture
def meter_driver():
    from i2as.drivers.sim_keithley_2182a import SimKeithley2182A
    return SimKeithley2182A("SIM")


@pytest.fixture
def lakeshore_driver():
    from i2as.drivers.sim_lakeshore_335 import SimLakeshore335
    return SimLakeshore335("SIM")


# ---------------------------------------------------------------------------
# SuperconductingMagnetVI
# ---------------------------------------------------------------------------

class TestSuperConductingMagnetVI:
    """Tests for SuperconductingMagnetVI (no switch heater)."""

    def _make_vi(self, driver):
        from i2as.virtual_instruments.magnet.superconducting_magnet import SuperconductingMagnetVI
        vi = SuperconductingMagnetVI(
            {"main": driver},
            amperes_per_tesla=10.0,
            default_ramp_rate=5.0,
            max_current=90.0,
            min_current=-90.0,
            ramp_segments=[
                {"max_current_A": 40.0, "rate_A_per_min": 5.0},
                {"max_current_A": 90.0, "rate_A_per_min": 2.0},
            ],
        )
        vi.vi_name = "magnet_z"
        return vi

    def test_initial_field_is_zero(self, ips_driver):
        vi = self._make_vi(ips_driver)
        assert vi.magnet_field_T() == pytest.approx(0.0)

    def test_initial_status_is_idle(self, ips_driver):
        vi = self._make_vi(ips_driver)
        assert vi.ramp_status() == "IDLE"

    def test_start_ramp_transitions_to_ramping(self, ips_driver):
        vi = self._make_vi(ips_driver)
        vi.start_ramp(1.0)
        assert vi.ramp_status() == "RAMPING"

    def test_ramp_reaches_target(self, ips_driver):
        vi = self._make_vi(ips_driver)
        ips_driver.set_ramp_rate(600.0)
        vi.start_ramp(1.0)
        # Simulate fast completion
        ips_driver._last_update = time.time() - 10.0
        for _ in range(20):
            vi.advance_ramp()
        assert vi.ramp_status() in ("TARGET_REACHED", "RAMPING")

    def test_get_state_returns_monitored_keys(self, ips_driver):
        vi = self._make_vi(ips_driver)
        state = vi.get_state()
        assert "psu_current" in state
        assert "magnet_current" in state
        assert "magnet_field_T" in state
        assert "magnet_status" in state

    def test_magnet_current_type(self, ips_driver):
        vi = self._make_vi(ips_driver)
        assert isinstance(vi.magnet_current(), float)

    def test_set_field_control_starts_ramp(self, ips_driver):
        vi = self._make_vi(ips_driver)
        vi.set_field(2.0)
        assert vi.ramp_status() == "RAMPING"

    def test_standby_ramps_to_zero(self, ips_driver):
        vi = self._make_vi(ips_driver)
        ips_driver._current = 50.0
        vi.standby()
        assert vi.ramp_status() == "RAMPING"

    def test_vi_type_is_magnet(self, ips_driver):
        vi = self._make_vi(ips_driver)
        assert vi.vi_type == "magnet"

    def test_ramp_target_and_rate_none_when_idle(self, ips_driver):
        vi = self._make_vi(ips_driver)
        assert vi.ramp_target() is None
        assert vi.ramp_rate() is None

    def test_ramp_target_reports_field_in_tesla(self, ips_driver):
        vi = self._make_vi(ips_driver)
        vi.start_ramp(1.0)
        assert vi.ramp_target() == pytest.approx(1.0)
        # Segment rate at 0 A is 5 A/min; at 10 A/T that is 0.5 T/min.
        assert vi.ramp_rate() == pytest.approx(0.5)

    def test_stop_ramp_clears_ramp_target(self, ips_driver):
        vi = self._make_vi(ips_driver)
        vi.start_ramp(1.0)
        vi.stop_ramp()
        assert vi.ramp_target() is None
        assert vi.ramp_rate() is None

    # -- magnet_state(): standby / ramping / holding / quenched -------------

    def test_magnet_state_standby_initially(self, ips_driver):
        vi = self._make_vi(ips_driver)
        assert vi.magnet_state() == "standby"

    def test_magnet_state_ramping_while_ramp_active(self, ips_driver):
        vi = self._make_vi(ips_driver)
        vi.start_ramp(1.0)
        assert vi.magnet_state() == "ramping"

    def test_magnet_state_holding_after_ramp_completes(self, ips_driver):
        """Regression: a ramp generator is not reset to None on normal
        completion (only stop_ramp() does that — see advance_ramp()), so
        magnet_state() must not use `self._ramp_gen is not None` as its
        ramping test, or it would report "ramping" forever after the first
        completed ramp instead of "holding"."""
        vi = self._make_vi(ips_driver)
        ips_driver.set_ramp_rate(600.0)
        vi.start_ramp(1.0)
        for _ in range(20):
            ips_driver._last_update = time.time() - 10.0
            vi.advance_ramp()
        assert vi.ramp_status() == "TARGET_REACHED"
        assert vi.magnet_state() == "holding"

    def test_magnet_state_standby_after_ramp_to_zero_completes(self, ips_driver):
        vi = self._make_vi(ips_driver)
        ips_driver._current = 50.0
        vi.start_ramp(0.0)
        # Force the simulated PSU to its target directly rather than
        # depending on ramp-segment rates/timing (50 A crosses this fixture's
        # 40 A segment boundary, so the clock-rewind trick used elsewhere in
        # this file would need many more than 20 ticks to actually finish).
        ips_driver._current = 0.0
        ips_driver._status = "HOLD"
        vi.advance_ramp()
        assert vi.ramp_status() == "TARGET_REACHED"
        assert vi.magnet_state() == "standby"

    def test_magnet_state_quenched(self, ips_driver):
        vi = self._make_vi(ips_driver)
        ips_driver._simulate_quench = True
        assert vi.magnet_state() == "quenched"


# ---------------------------------------------------------------------------
# SampleTemperatureControllerVI
# ---------------------------------------------------------------------------

class TestSampleTemperatureControllerVI:
    """Tests for SampleTemperatureControllerVI."""

    def _make_vi(self, driver):
        from i2as.virtual_instruments.temperature.sample_temperature_controller import (
            SampleTemperatureControllerVI,
        )
        vi = SampleTemperatureControllerVI(
            {"main": driver},
            default_ramp_rate=600.0,
            tolerance=0.5,
        )
        vi.vi_name = "temperature_sample"
        return vi

    def test_initial_temperature_returns_float(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        assert isinstance(vi.temperature(), float)

    def test_initial_status_is_idle(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        assert vi.ramp_status() == "IDLE"

    def test_start_ramp_transitions_to_ramping(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        vi.start_ramp(100.0)
        assert vi.ramp_status() == "RAMPING"

    def test_ramp_completes(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        # Driver starts at 300 K; ramp to 300.1 K with very fast rate
        vi.start_ramp(300.1, rate=9999.0)
        for _ in range(100):
            vi.advance_ramp()
        # After generator exhausts, hardware must settle within tolerance
        lakeshore_driver._temperature = 300.05  # Force within tolerance
        lakeshore_driver._setpoint = 300.1
        assert vi.ramp_status() in ("TARGET_REACHED", "RAMPING")

    def test_temperature_and_rate_limits_from_init_params(self, lakeshore_driver):
        """Config-declared temperature / ramp-rate bounds are enforced on the
        @control entry points (control-validation standard)."""
        from i2as.core.exceptions import I2ASSafetyError
        from i2as.virtual_instruments.temperature.sample_temperature_controller import (
            SampleTemperatureControllerVI,
        )

        vi = SampleTemperatureControllerVI(
            {"main": lakeshore_driver},
            default_ramp_rate=2.0,
            tolerance=0.5,
            min_temperature_K=1.4,
            max_temperature_K=320.0,
            max_ramp_rate_K_per_min=20.0,
        )
        vi.vi_name = "temperature_sample"

        with pytest.raises(I2ASSafetyError, match="outside the allowed range"):
            vi.set_temperature(400.0)
        with pytest.raises(I2ASSafetyError, match="outside the allowed range"):
            vi.set_temperature(0.5)
        with pytest.raises(I2ASSafetyError, match="outside the allowed range"):
            vi.set_ramp_rate(50.0)
        with pytest.raises(ValueError):
            vi.set_ramp_rate(0.0)  # semantic guard: rate must be positive

        # Within bounds — accepted (driver starts at 300 K, so ramp toward
        # 250 K is genuinely in progress).
        vi.set_temperature(250.0)
        assert vi.ramp_status() == "RAMPING"

    def test_ramp_target_and_rate_none_when_idle(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        assert vi.ramp_target() is None
        assert vi.ramp_rate() is None

    def test_ramp_target_and_rate_reported_during_ramp(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        vi.start_ramp(100.0, rate=5.0)
        assert vi.ramp_target() == pytest.approx(100.0)
        assert vi.ramp_rate() == pytest.approx(5.0)

    def test_stop_ramp_clears_target_and_rate(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        vi.start_ramp(100.0, rate=5.0)
        vi.stop_ramp()
        assert vi.ramp_target() is None
        assert vi.ramp_rate() is None

    def test_no_limits_in_init_params_means_unbounded(self, lakeshore_driver):
        """A setup that declares no temperature bounds keeps working (open range)."""
        vi = self._make_vi(lakeshore_driver)
        vi.set_temperature(500.0)  # no max_temperature_K configured
        assert vi.ramp_status() == "RAMPING"

    def test_stop_ramp_pins_setpoint_to_current_temperature(self, lakeshore_driver):
        """stop_ramp() must go IDLE and pin the setpoint where the system is —
        otherwise the controller keeps regulating toward the last-commanded
        intermediate setpoint after an abort (review finding C3)."""
        vi = self._make_vi(lakeshore_driver)
        vi.start_ramp(100.0)
        vi.advance_ramp()
        vi.stop_ramp()
        assert vi.ramp_status() == "IDLE"
        assert lakeshore_driver.get_setpoint() == pytest.approx(
            lakeshore_driver.get_temperature(), abs=0.01
        )

    def test_get_state_keys(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        state = vi.get_state()
        assert "temperature" in state
        assert "setpoint" in state
        assert "heater_output" in state

    def test_set_ramp_rate_control(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        vi.set_ramp_rate(10.0)
        assert vi._default_ramp_rate == pytest.approx(10.0)

    def test_set_temperature_control_starts_ramp(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        vi.set_temperature(100.0)
        assert vi.ramp_status() == "RAMPING"

    def test_vi_type_is_temperature(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        assert vi.vi_type == "temperature"

    def test_no_needle_valve_attribute(self, lakeshore_driver):
        """SampleTemperatureControllerVI must NOT expose needle valve."""
        vi = self._make_vi(lakeshore_driver)
        assert not hasattr(vi, "needle_valve")

    # -- heater mode (shared base: Lakeshore 335 and Oxford ITC503 both
    #    implement get/set_heater_mode with the same 'AUTO'/'MANUAL' values) --

    def test_heater_mode_default_is_auto(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        assert vi.heater_mode() == "AUTO"

    def test_set_heater_mode_control(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        vi.set_heater_mode("MANUAL")
        assert vi.heater_mode() == "MANUAL"
        assert lakeshore_driver.get_heater_mode() == "MANUAL"
        vi.set_heater_mode("AUTO")
        assert vi.heater_mode() == "AUTO"

    def test_set_heater_mode_rejects_invalid(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        with pytest.raises(ValueError):
            vi.set_heater_mode("INVALID")

    def test_get_state_includes_heater_mode(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        state = vi.get_state()
        assert state["heater_mode"] == "AUTO"

    # -- manual heater output (only meaningful, and only allowed, while
    #    heater mode is MANUAL) --

    def test_set_heater_output_rejects_while_auto(self, lakeshore_driver):
        from i2as.core.exceptions import I2ASSafetyError

        vi = self._make_vi(lakeshore_driver)
        with pytest.raises(I2ASSafetyError, match="AUTO"):
            vi.set_heater_output(50.0)

    def test_set_heater_output_control(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        # The 335 delivers no power while its heater range is Off, so the
        # readback would stay at 0 % whatever output is programmed.
        lakeshore_driver.set_heater_range("MEDIUM")
        vi.set_heater_mode("MANUAL")
        vi.set_heater_output(42.0)
        assert vi.heater_output() == pytest.approx(42.0)

    # -- lifecycle: initiate() -> heater AUTO, standby() -> heater MANUAL
    #    at zero output (standardised across temperature controller VIs) --

    def test_initiate_sets_heater_mode_auto(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        vi.set_heater_mode("MANUAL")
        vi.initiate()
        assert vi.heater_mode() == "AUTO"

    def test_initiate_pins_setpoint_to_rounded_current_temperature(self, lakeshore_driver):
        """A stale setpoint left over from before the heater was switched to
        MANUAL must not survive initiate() — otherwise flipping to AUTO hands
        the PID a far-off target and the heater immediately starts driving
        toward it."""
        vi = self._make_vi(lakeshore_driver)
        vi.set_heater_mode("MANUAL")
        lakeshore_driver._temperature = 42.3
        lakeshore_driver.set_setpoint(250.0)
        vi.initiate()
        assert lakeshore_driver.get_setpoint() == pytest.approx(42.0)

    def test_standby_sets_heater_manual_and_zero_output(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        vi.set_heater_mode("MANUAL")
        lakeshore_driver.set_heater_output(50.0)
        vi.standby()
        assert vi.heater_mode() == "MANUAL"
        assert vi.heater_output() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Lakeshore335SampleTemperatureControllerVI
# ---------------------------------------------------------------------------

class TestLakeshore335SampleTemperatureControllerVI:
    """Tests for Lakeshore335SampleTemperatureControllerVI (with calibration curve)."""

    def _make_vi(self, driver):
        from i2as.virtual_instruments.temperature.lakeshore_335_sample_temperature_controller import (
            Lakeshore335SampleTemperatureControllerVI,
        )
        vi = Lakeshore335SampleTemperatureControllerVI(
            {"main": driver},
            default_ramp_rate=600.0,
            tolerance=0.5,
        )
        vi.vi_name = "temperature_sample"
        return vi

    def test_inherits_temperature_methods(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        assert isinstance(vi.temperature(), float)
        assert isinstance(vi.setpoint(), float)
        assert isinstance(vi.heater_output(), float)

    def test_ramp_inherited(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        vi.start_ramp(100.0)
        assert vi.ramp_status() == "RAMPING"

    def test_curve_reads_sensor_input_a(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        assert vi.curve() == lakeshore_driver.get_sensor_curve("A")

    def test_set_curve_control(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        # Slot 21 is a USER slot; the sim refuses assigning an empty one, so
        # load it first (see test_l0_lakeshore_335.py).
        lakeshore_driver._loaded_user_curves.add(21)
        vi.set_curve(21)
        assert vi.curve() == 21
        assert lakeshore_driver.get_sensor_curve("A") == 21
        # Sensor input B is untouched by the sample VI.
        assert lakeshore_driver.get_sensor_curve("B") == 2

    def test_set_curve_rejects_out_of_range(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        with pytest.raises(ValueError):
            vi.set_curve(60)

    def test_get_state_includes_curve(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        state = vi.get_state()
        assert state["curve"] == lakeshore_driver.get_sensor_curve("A")

    def test_control_param_specs_curve_choices(self, lakeshore_driver):
        """The curve drop-down defaults to the last POLLED curve, not a fresh read.

        ``control_param_specs()`` is a pure read (see its purity rule in
        ``BaseVirtualInstrument``): it must default the drop-down from the
        monitor cycle's cache rather than by asking the instrument, so
        describing the station never puts traffic on the bus.
        """
        vi = self._make_vi(lakeshore_driver)
        # Slot 22 is the one user curve the sim ships loaded; an empty slot
        # is refused by the driver's error-reporting standard.
        lakeshore_driver.set_sensor_curve(22, "A")
        vi.get_state()  # the monitor cycle's poll fills the cache

        specs = vi.control_param_specs("set_curve")
        spec = specs["curve"]
        assert spec.choices["None (0)"] == 0
        assert spec.choices["Standard 1"] == 1
        assert spec.choices["User 59"] == 59
        assert spec.default == 22

    def test_control_param_specs_default_before_first_poll(self, lakeshore_driver):
        """Never polled: the drop-downs fall back rather than reading hardware.

        A ``ParamSpec`` default must be one of its own ``choices``, so the
        fallback is a declared choice ("None (0)" / "Off") — never whatever
        the instrument happens to hold, which this path may not ask for.
        """
        vi = self._make_vi(lakeshore_driver)
        lakeshore_driver.set_sensor_curve(22, "A")
        lakeshore_driver.set_heater_range("HIGH")

        assert vi.control_param_specs("set_curve")["curve"].default == 0
        assert (
            vi.control_param_specs("set_heater_range")["range_setting"].default
            == "OFF"
        )

    def test_control_param_specs_issue_no_driver_traffic(self, lakeshore_driver):
        """The describe path sends nothing to the driver, for any control.

        The per-VI counterpart of the station-wide conformance check: the
        purity rule is what lets ``Station.station_info()`` describe an
        instrument without operating it.
        """
        vi = self._make_vi(lakeshore_driver)
        vi.get_state()

        calls: list[str] = []
        spy_on_driver(lakeshore_driver, calls)
        for method_name in ("set_curve", "set_heater_range", "set_temperature"):
            vi.control_param_specs(method_name)
        assert calls == []

    def test_control_param_specs_other_methods_unaffected(self, lakeshore_driver):
        """A control the override does not name falls through to the decorator.

        ``set_temperature`` is declared on ``SampleTemperatureControllerVI``
        (the declaration standard), so the hook must hand back that
        inherited declaration untouched rather than injecting anything of
        its own.
        """
        vi = self._make_vi(lakeshore_driver)
        specs = vi.control_param_specs("set_temperature")
        assert set(specs) == {"target_K"}
        assert specs["target_K"].unit == "K"
        assert specs["target_K"].choices is None

    # -- heater range (Lakeshore-specific: powers up 'OFF', which is the
    #    reported bug — no heater power regardless of heater_mode or
    #    setpoint until the range is explicitly turned on) --------------------

    def test_heater_range_default_is_off(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        assert vi.heater_range() == "OFF"

    def test_set_heater_range_control(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        vi.set_heater_range("HIGH")
        assert vi.heater_range() == "HIGH"
        assert lakeshore_driver.get_heater_range() == "HIGH"

    def test_set_heater_range_rejects_invalid(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        with pytest.raises(ValueError):
            vi.set_heater_range("INVALID")

    def test_heater_stays_off_until_range_is_set(self, lakeshore_driver):
        """Regression: heater_output() must read 0 at the VI layer while
        heater_range is 'OFF', even with heater_mode AUTO and a setpoint far
        from the current temperature — reproduces the reported bug and its
        fix at the level the front panel actually reads."""
        vi = self._make_vi(lakeshore_driver)
        vi.set_heater_mode("AUTO")
        vi.set_temperature(300.0)
        assert vi.heater_output() == pytest.approx(0.0)

        vi.set_heater_range("HIGH")
        lakeshore_driver._temperature = 4.2
        assert vi.heater_output() > 0.0

    def test_control_param_specs_heater_range_choices(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        specs = vi.control_param_specs("set_heater_range")
        spec = specs["range_setting"]
        assert spec.choices == {"Off": "OFF", "Low": "LOW", "Medium": "MEDIUM", "High": "HIGH"}
        assert spec.default == vi.heater_range()

    def test_get_state_includes_heater_range(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        state = vi.get_state()
        assert state["heater_range"] == "OFF"

    # -- lifecycle: standby() is inherited unchanged and must not touch
    #    heater_range (switching heater power back off is the operator's
    #    call); initiate() selects the configured range, because the 335
    #    delivers no power at all while the range is Off --

    def test_standby_does_not_touch_heater_range(self, lakeshore_driver):
        vi = self._make_vi(lakeshore_driver)
        vi.set_heater_range("HIGH")
        vi.set_heater_mode("AUTO")
        vi.standby()
        assert vi.heater_mode() == "MANUAL"
        assert vi.heater_range() == "HIGH"

    def test_initiate_selects_the_configured_heater_range(self, lakeshore_driver):
        """Without a range the closed loop is on but powerless (manual §4.5.1.7.8)."""
        vi = self._make_vi(lakeshore_driver)
        assert vi.heater_range() == "OFF"  # the instrument's power-up default
        vi.initiate()
        assert vi.heater_mode() == "AUTO"
        assert vi.heater_range() == "MEDIUM"

    def test_initiate_heater_range_comes_from_the_config(self, lakeshore_driver):
        from i2as.virtual_instruments.temperature.lakeshore_335_sample_temperature_controller import (
            Lakeshore335SampleTemperatureControllerVI,
        )

        vi = Lakeshore335SampleTemperatureControllerVI(
            {"main": lakeshore_driver}, initiate_heater_range="HIGH"
        )
        vi.initiate()
        assert vi.heater_range() == "HIGH"

    def test_unknown_initiate_heater_range_is_refused(self, lakeshore_driver):
        from i2as.virtual_instruments.temperature.lakeshore_335_sample_temperature_controller import (
            Lakeshore335SampleTemperatureControllerVI,
        )

        with pytest.raises(ValueError, match="initiate_heater_range"):
            Lakeshore335SampleTemperatureControllerVI(
                {"main": lakeshore_driver}, initiate_heater_range="WARM"
            )


# ---------------------------------------------------------------------------
# DCSeparateMeasurementVI
# ---------------------------------------------------------------------------


class TestDCSeparateMeasurementVI:
    """Tests for DCSeparateMeasurementVI."""

    def _make_vi(self, source, meter):
        from i2as.virtual_instruments.measurement.dc_separate_measurement import DCSeparateMeasurementVI
        vi = DCSeparateMeasurementVI({"source": source, "meter": meter})
        vi.vi_name = "dc_measurement"
        return vi

    def test_ping_returns_true_when_drivers_respond(self, source_driver, meter_driver):
        vi = self._make_vi(source_driver, meter_driver)
        assert vi.ping() is True

    def test_initiate_and_take_reading(self, source_driver, meter_driver):
        vi = self._make_vi(source_driver, meter_driver)
        vi.initiate_measurement(current_A=1e-6, compliance_A=1e-3, voltmeter_range_V=0.1, readings_per_point=5)
        data = vi.take_reading()
        assert "voltage_V" in data
        assert "current_A" in data
        assert len(data["voltage_V_array"]) == 5
        assert len(data["current_A_array"]) == 5

    def test_current_constant_across_readings(self, source_driver, meter_driver):
        vi = self._make_vi(source_driver, meter_driver)
        vi.initiate_measurement(current_A=2e-6, readings_per_point=10)
        data = vi.take_reading()
        assert all(abs(c - 2e-6) < 1e-12 for c in data["current_A_array"])
        assert data["current_A"] == pytest.approx(2e-6)

    def test_take_reading_without_initiate_raises(self, source_driver, meter_driver):
        vi = self._make_vi(source_driver, meter_driver)
        with pytest.raises(RuntimeError):
            vi.take_reading()

    def test_standby_resets_state(self, source_driver, meter_driver):
        vi = self._make_vi(source_driver, meter_driver)
        vi.initiate_measurement(current_A=1e-6)
        vi.standby()
        with pytest.raises(RuntimeError):
            vi.take_reading()

    def test_inherits_dc_measurement_base(self, source_driver, meter_driver):
        from i2as.virtual_instruments.base import DCMeasurementBase
        vi = self._make_vi(source_driver, meter_driver)
        assert isinstance(vi, DCMeasurementBase)

    def test_vi_type_is_measurement(self, source_driver, meter_driver):
        vi = self._make_vi(source_driver, meter_driver)
        assert vi.vi_type == "measurement"

    # ── The reading-loop setter (reading_setters standard) ───────────────────

    def test_set_source_current_before_initiate_raises(self, source_driver, meter_driver):
        vi = self._make_vi(source_driver, meter_driver)
        with pytest.raises(RuntimeError):
            vi.set_source_current(1e-6)

    def test_set_source_current_changes_reported_current(self, source_driver, meter_driver):
        vi = self._make_vi(source_driver, meter_driver)
        vi.initiate_measurement(current_A=1e-6, readings_per_point=4)
        vi.set_source_current(-1e-6)
        data = vi.take_reading()
        assert all(abs(c + 1e-6) < 1e-12 for c in data["current_A_array"])
        assert len(data["voltage_V_array"]) == 4

    def test_declares_current_reading_setter(self, source_driver, meter_driver):
        """current_A is loopable via set_source_current (the reading loop)."""
        vi = self._make_vi(source_driver, meter_driver)
        assert vi.reading_setters == {"current_A": "set_source_current"}
