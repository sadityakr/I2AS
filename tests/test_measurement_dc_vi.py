# ---
# description: |
#   Unit tests for DCSeparateMeasurementVI (L1 Virtual Instruments layer).
#   Tests initiate/take_reading lifecycle, data shape, standby, and ping.
# entry_point: pytest tests/test_measurement_dc_vi.py -v
# last_updated: 2026-04-19
# ---

import pytest
from i2as.drivers.sim_keithley_6221 import SimKeithley6221
from i2as.drivers.sim_keithley_2182a import SimKeithley2182A
from i2as.virtual_instruments.measurement.dc_separate_measurement import DCSeparateMeasurementVI


@pytest.fixture()
def vi() -> DCSeparateMeasurementVI:
    source = SimKeithley6221("SIM")
    meter = SimKeithley2182A("SIM")
    return DCSeparateMeasurementVI({"source": source, "meter": meter})


@pytest.fixture()
def vi_with_drivers():
    source = SimKeithley6221("SIM")
    meter = SimKeithley2182A("SIM")
    inst = DCSeparateMeasurementVI({"source": source, "meter": meter})
    return inst, source, meter


# ------------------------------------------------------------------
# take_reading before initiate must raise
# ------------------------------------------------------------------

def test_take_reading_before_initiate_raises(vi: DCSeparateMeasurementVI) -> None:
    with pytest.raises(RuntimeError, match="initiate_measurement\\(\\)"):
        vi.take_reading()


# ------------------------------------------------------------------
# initiate programs both instruments
# ------------------------------------------------------------------

def test_initiate_sets_source_current(vi_with_drivers) -> None:
    inst, source, _ = vi_with_drivers
    inst.initiate_measurement(current_A=2e-6, compliance_A=5e-3, voltmeter_range_V=0.01)
    assert source.get_current() == pytest.approx(2e-6)


def test_initiate_sets_compliance(vi_with_drivers) -> None:
    inst, source, _ = vi_with_drivers
    inst.initiate_measurement(current_A=1e-6, compliance_A=1e-2, voltmeter_range_V=0.1)
    assert source.get_compliance() == pytest.approx(1e-2)


def test_initiate_sets_voltmeter_range(vi_with_drivers) -> None:
    inst, _, meter = vi_with_drivers
    inst.initiate_measurement(current_A=1e-6, compliance_A=1e-3, voltmeter_range_V=0.05)
    assert meter.get_range() == pytest.approx(0.05)


# ------------------------------------------------------------------
# take_reading returns correct shape and constant current
# ------------------------------------------------------------------

def test_take_reading_returns_correct_n_points(vi: DCSeparateMeasurementVI) -> None:
    vi.initiate_measurement(current_A=1e-6, compliance_A=1e-3, voltmeter_range_V=0.1, readings_per_point=20)
    data = vi.take_reading()
    assert len(data["voltage_V_array"]) == 20
    assert len(data["current_A_array"]) == 20


def test_take_reading_current_array_is_constant(vi: DCSeparateMeasurementVI) -> None:
    current = 3e-6
    vi.initiate_measurement(current_A=current, compliance_A=1e-3, voltmeter_range_V=0.1, readings_per_point=15)
    data = vi.take_reading()
    assert all(c == pytest.approx(current) for c in data["current_A_array"])
    assert data["current_A"] == pytest.approx(current)


def test_take_reading_voltage_values_are_floats(vi: DCSeparateMeasurementVI) -> None:
    vi.initiate_measurement(current_A=1e-6, compliance_A=1e-3, voltmeter_range_V=0.1, readings_per_point=5)
    data = vi.take_reading()
    assert all(isinstance(v, float) for v in data["voltage_V_array"])


# ------------------------------------------------------------------
# standby zeros source and blocks take_reading
# ------------------------------------------------------------------

def test_standby_zeros_current(vi_with_drivers) -> None:
    inst, source, _ = vi_with_drivers
    inst.initiate_measurement(current_A=5e-6, compliance_A=1e-3, voltmeter_range_V=0.1)
    inst.standby()
    assert source.get_current() == pytest.approx(0.0)


def test_standby_blocks_subsequent_take_reading(vi: DCSeparateMeasurementVI) -> None:
    vi.initiate_measurement(current_A=1e-6, compliance_A=1e-3, voltmeter_range_V=0.1)
    vi.standby()
    with pytest.raises(RuntimeError):
        vi.take_reading()


# ------------------------------------------------------------------
# ping
# ------------------------------------------------------------------

def test_ping_returns_true_when_both_respond(vi: DCSeparateMeasurementVI) -> None:
    assert vi.ping() is True


def test_ping_returns_false_when_source_fails(vi_with_drivers) -> None:
    inst, source, _ = vi_with_drivers
    source._simulate_error = True
    assert inst.ping() is False


def test_ping_returns_false_when_meter_fails(vi_with_drivers) -> None:
    inst, _, meter = vi_with_drivers
    meter._simulate_error = True
    assert inst.ping() is False


# ------------------------------------------------------------------
# @control decoration (GUI discoverability)
# ------------------------------------------------------------------

def test_initiate_is_control(vi: DCSeparateMeasurementVI) -> None:
    assert getattr(vi.initiate_measurement, "_is_control", False) is True


def test_initiate_control_params_include_expected_args(vi: DCSeparateMeasurementVI) -> None:
    params = getattr(vi.initiate_measurement, "_control_params", {})
    assert "current_A" in params
    assert "compliance_A" in params
    assert "voltmeter_range_V" in params
