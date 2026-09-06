import logging

import pytest
import time
from i2as.core.exceptions import I2ASCommunicationError, I2ASSafetyError

# Assuming we have valid sim drivers from L0 test phase
from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120
from i2as.drivers.sim_lakeshore_335 import SimLakeshore335
from i2as.drivers.sim_keithley_6221 import SimKeithley6221
from i2as.drivers.sim_keithley_2182a import SimKeithley2182A

from i2as.virtual_instruments.base import BaseVirtualInstrument
from i2as.core.decorators import monitored, control

# 1. BaseVirtualInstrument subclass logging/state tests
class MockBaseVI(BaseVirtualInstrument):
    vi_type = "mock"
    def __init__(self):
        super().__init__({})
        self.calls = 0

    @monitored
    def mon_test(self):
        self.calls += 1
        return "monitored_val"

    @control
    def ctrl_test(self, val):
        self.calls += 1
        return val

    @control
    def error_test(self):
        raise ValueError("test error")

def test_base_vi_logging_and_state():
    vi = MockBaseVI()
    
    state = vi.get_state()
    assert "mon_test" in state
    assert state["mon_test"] == "monitored_val"
    assert vi.calls == 1


def test_vi_bus_traffic_is_logged_under_the_vi_logger_prefix(caplog):
    """Every wrapped call logs to ``<VI_LOGGER_PREFIX><vi_name>`` at DEBUG.

    The GUI log panel filters on that prefix to keep polling noise in the
    file, so the name is a contract between the base class and the panel.
    """
    from i2as.core.logging_config import VI_LOGGER_PREFIX

    vi = MockBaseVI()
    vi.vi_name = "probe"
    with caplog.at_level(logging.DEBUG, logger=f"{VI_LOGGER_PREFIX}probe"):
        vi.ctrl_test(3)
    names = {record.name for record in caplog.records}
    assert f"{VI_LOGGER_PREFIX}probe" in names
    assert f"{VI_LOGGER_PREFIX}probe" == "i2as.vi.probe"

def test_control_specs_and_panel_survive_subclass_wrapping():
    """__init_subclass__ must propagate _control_specs/_control_panel onto the
    limit+logging wrappers, or the GUI would see empty metadata on every VI."""
    from i2as.core.decorators import get_control_panel, get_control_specs
    from i2as.core.plan import ParamSpec

    spec = ParamSpec(type=float, default=0.0, unit="W", min=0.0, max=40.0)

    class SpecVI(BaseVirtualInstrument):
        vi_type = "mock"

        @control(params={"power_W": spec}, panel=False)
        def set_heater_power(self, power_W: float = 0.0):
            return power_W

    vi = SpecVI({})
    assert get_control_specs(vi.set_heater_power) == {"power_W": spec}
    assert get_control_panel(vi.set_heater_power) is False
    # Legacy bare @control: no specs, panel defaults True.
    mock = MockBaseVI()
    assert get_control_specs(mock.ctrl_test) == {}
    assert get_control_panel(mock.ctrl_test) is True


def test_control_specs_must_be_paramspec_instances():
    """A non-ParamSpec spec value fails at class creation, not on click."""
    with pytest.raises(TypeError, match="must be a ParamSpec"):
        class BadSpecVI(BaseVirtualInstrument):
            vi_type = "mock"

            @control(params={"power_W": {"type": float, "default": 0.0}})
            def set_heater_power(self, power_W: float = 0.0):
                return power_W


class LimitedVI(BaseVirtualInstrument):
    """A VI declaring one control limit, for the limit-wrapper tests below."""

    vi_type = "limited"
    control_limits = {"set_current": {"current_A": "max_current"}}

    def __init__(self):
        super().__init__({})
        self._limits["max_current"] = (-1e-3, 1e-3)

    @control
    def set_current(self, current_A):
        return current_A


def test_limit_wrapper_allows_a_value_inside_the_declared_range():
    """A control call within its declared limit reaches the method untouched."""
    vi = LimitedVI()
    vi.vi_name = "source"

    assert vi.set_current(5e-4) == 5e-4


def test_limit_wrapper_refusal_carries_structured_fields():
    """An out-of-range control call is refused with fields AND the same prose.

    Two assertions, deliberately together: a verdict is built from the
    structured fields, never by parsing the message, and the message itself is
    the operator's banner, so it must not drift when the fields are added.
    """
    vi = LimitedVI()
    vi.vi_name = "source"

    with pytest.raises(I2ASSafetyError) as excinfo:
        vi.set_current(0.05)
    err = excinfo.value

    assert str(err) == (
        "source.set_current: current_A=0.05 is outside the allowed range "
        "[-0.001, 0.001] for this setup (limit 'max_current' from the station "
        "config). Command refused."
    )
    assert err.param == "current_A"
    assert err.value == 0.05
    assert err.lo == -1e-3
    assert err.hi == 1e-3
    assert err.limit_name == "max_current"


def test_base_vi_error_pass_through():
    vi = MockBaseVI()
    with pytest.raises(ValueError):
        vi.error_test()

def test_communication_error_wrapping(monkeypatch):
    class FakeVisaIOError(Exception):
        pass

    import sys
    import types
    fake_pyvisa = types.ModuleType("pyvisa")
    fake_pyvisa.errors = types.ModuleType("errors")
    fake_pyvisa.errors.VisaIOError = FakeVisaIOError
    sys.modules["pyvisa"] = fake_pyvisa

    class MockCommVI(BaseVirtualInstrument):
        __module__ = "test"
        @monitored
        def fail(self):
            raise FakeVisaIOError("timeout")

    vi = MockCommVI({"main": None})
    with pytest.raises(I2ASCommunicationError):
        vi.fail()


# 3b. MeasurementInstrumentBase mean/error/array convention helpers
def test_mean_and_sem_multiple_samples():
    from i2as.virtual_instruments.base import MeasurementInstrumentBase

    mean, sem = MeasurementInstrumentBase.mean_and_sem([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    # stdev([1,2,3], ddof=1) == 1.0; sem == 1.0 / sqrt(3)
    assert sem == pytest.approx(1.0 / 3**0.5)


def test_mean_and_sem_single_sample_has_zero_error():
    from i2as.virtual_instruments.base import MeasurementInstrumentBase

    mean, sem = MeasurementInstrumentBase.mean_and_sem([5.0])
    assert mean == pytest.approx(5.0)
    assert sem == 0.0


def test_mean_and_sem_no_samples_is_nan():
    from i2as.virtual_instruments.base import MeasurementInstrumentBase
    import math

    mean, sem = MeasurementInstrumentBase.mean_and_sem([])
    assert math.isnan(mean)
    assert math.isnan(sem)


def test_quantity_columns_derives_array_mean_error_keys():
    from i2as.virtual_instruments.base import MeasurementInstrumentBase

    array_keys, scalar_columns = MeasurementInstrumentBase.quantity_columns(
        "voltage_V", "current_A"
    )
    assert array_keys == ["voltage_V_array", "current_A_array"]
    assert scalar_columns == {
        "voltage_V": "float",
        "voltage_V_error": "float",
        "current_A": "float",
        "current_A_error": "float",
    }


def test_quantity_columns_rejects_colliding_base_name():
    from i2as.core.exceptions import I2ASConfigError
    from i2as.virtual_instruments.base import MeasurementInstrumentBase

    with pytest.raises(I2ASConfigError, match="_array' or '_error'"):
        MeasurementInstrumentBase.quantity_columns("voltage_V_error")


# 4. Magnet VI tests
def test_magnet_vi_ramp_cycle():
    from i2as.virtual_instruments.magnet.superconducting_magnet import SuperconductingMagnetVI

    driver = SimOxfordIPS120("SIM")
    vi = SuperconductingMagnetVI({"main": driver}, default_ramp_rate=1200.0, amperes_per_tesla=10.0)

    vi.start_ramp(1.0)
    assert vi.ramp_status() == "RAMPING"

    for _ in range(50):
        vi.advance_ramp()
        driver._last_update = time.time() - 0.5
        driver._update_simulation()

    assert vi.ramp_status() in ("TARGET_REACHED", "IDLE")
    assert vi.magnet_field_T() == pytest.approx(1.0, abs=0.1)

def test_magnet_vi_ramp_segments():
    from i2as.virtual_instruments.magnet.superconducting_magnet import SuperconductingMagnetVI

    driver = SimOxfordIPS120("SIM")
    segments = [
        {"max_current_A": 20.0, "rate_A_per_min": 600.0},
        {"max_current_A": float('inf'), "rate_A_per_min": 100.0}
    ]
    vi = SuperconductingMagnetVI({"main": driver}, default_ramp_rate=5.0, amperes_per_tesla=10.0, ramp_segments=segments)

    vi.start_ramp(3.0)

    rates_used = set()
    for _ in range(100):
        vi.advance_ramp()
        driver._last_update = time.time() - 0.5
        driver._update_simulation()
        rates_used.add(driver._ramp_rate)
        if vi.ramp_status() == "TARGET_REACHED":
            break

    assert 600.0 in rates_used
    assert 100.0 in rates_used
    assert vi.magnet_field_T() == pytest.approx(3.0, abs=0.1)

def test_magnet_vi_safety_clamping():
    from i2as.virtual_instruments.magnet.superconducting_magnet import SuperconductingMagnetVI

    driver = SimOxfordIPS120("SIM")
    vi = SuperconductingMagnetVI({"main": driver}, max_current=50.0, min_current=-50.0)

    vi.start_ramp(6.0)
    for _ in range(10):
        vi.advance_ramp()

    assert driver.get_current_setpoint() <= 50.0

# 10. Temperature VI tests
def test_temperature_vi_ramp():
    from i2as.virtual_instruments.temperature.sample_temperature_controller import SampleTemperatureControllerVI
    driver = SimLakeshore335("SIM")

    vi = SampleTemperatureControllerVI({"main": driver}, default_ramp_rate=6000.0, tolerance=2.0)
    vi.start_ramp(200.0)

    for _ in range(20):
        time.sleep(0.01)
        vi.advance_ramp()
        driver._last_update = time.time() - 1.0
        driver._update_simulation()

    assert vi.ramp_status() in ("RAMPING", "TARGET_REACHED")

def test_temperature_vi_set_pid_forwards_to_driver_and_hides_from_card():
    """set_pid programs all three PID values on the driver; front-panel only.

    Covers both temperature VIs (the VTI VI inherits set_pid unchanged).
    """
    from i2as.core.decorators import get_control_panel
    from i2as.virtual_instruments.temperature.sample_temperature_controller import (
        SampleTemperatureControllerVI,
    )

    driver = SimLakeshore335("SIM")
    vi = SampleTemperatureControllerVI({"main": driver})

    vi.set_pid(p_K=25.0, i_min=2.5, d_min=0.5)
    assert driver.get_proportional_band() == pytest.approx(25.0)
    assert driver.get_integral_action_time() == pytest.approx(2.5)
    assert driver.get_derivative_action_time() == pytest.approx(0.5)

    # panel=False: shown in the instrument front panel, never on the card.
    assert get_control_panel(vi.set_pid) is False


# 16. Shared-instrument mode discipline (see GLOSSARY.md and
#     virtual_instruments/measurement/README.md): two measurement VIs can be
#     wired to the same physical Keithley 6221
#     (devices.yaml's real_drivers.keithley_6221), so a VI must never assume
#     the instrument was left in a compatible mode by whichever measurement
#     method ran previously.
def test_dc_separate_initiate_recovers_from_stale_delta_arm():
    """DC-separate VI's initiate() must not depend on a prior standby().

    Arms delta mode on a shared 6221 WITHOUT calling standby() (the Orchestrator
    normally would, but this proves DC-separate is safe even if it weren't —
    the shared-instrument mode discipline standard). initiate() must still
    leave the instrument correctly in plain DC mode at the requested current.
    """
    from i2as.virtual_instruments.measurement.dc_separate_measurement import (
        DCSeparateMeasurementVI,
    )

    source = SimKeithley6221("SIM")
    meter = SimKeithley2182A("SIM")
    source._paired_meter = meter

    # Another client left the shared source armed in delta mode.
    source.configure_and_start_delta(high_current=1e-6, n_readings=5, delay=0.001)
    assert source._mode == "DELTA"

    dc_vi = DCSeparateMeasurementVI({"source": source, "meter": meter})
    dc_vi.initiate_measurement(current_A=5e-6)

    assert source._mode == "DC"
    assert source.get_current() == pytest.approx(5e-6)


# 4b. Ramp-introspection standard: ramp_setpoint() — the NEXT setpoint
def test_rampable_default_ramp_setpoint_is_none():
    """RampableVI's optional hook has a safe default, like every other one."""
    from i2as.virtual_instruments.rampable import RampableVI

    class BareRampable(RampableVI):
        def start_ramp(self, target): ...
        def advance_ramp(self): ...
        def ramp_status(self): return "IDLE"
        def stop_ramp(self): ...

    assert BareRampable().ramp_setpoint() is None


def test_magnet_ramp_setpoint_is_the_segment_boundary_not_the_target():
    """Mid-ramp the magnet drives to the next segment boundary, not the end setpoint.

    This distinction is the whole reason the tracker shows two numbers: the
    PSU is heading for 2 T right now while the ramp finishes at 3 T.
    """
    from i2as.virtual_instruments.magnet.superconducting_magnet import (
        SuperconductingMagnetVI,
    )

    driver = SimOxfordIPS120("SIM")
    segments = [
        {"max_current_A": 20.0, "rate_A_per_min": 600.0},
        {"max_current_A": float("inf"), "rate_A_per_min": 100.0},
    ]
    vi = SuperconductingMagnetVI(
        {"main": driver},
        default_ramp_rate=5.0,
        amperes_per_tesla=10.0,
        ramp_segments=segments,
    )

    assert vi.ramp_setpoint() is None  # idle: nothing commanded

    vi.start_ramp(3.0)
    assert vi.ramp_setpoint() == pytest.approx(2.0)  # the 20 A boundary
    assert vi.ramp_target() == pytest.approx(3.0)

    for _ in range(100):
        vi.advance_ramp()
        driver._last_update = time.time() - 0.5
        driver._update_simulation()
        if vi.ramp_status() == "TARGET_REACHED":
            break
    # Past the boundary the setpoint becomes the end setpoint itself.
    assert vi.ramp_setpoint() == pytest.approx(3.0)


def test_magnet_stop_ramp_clears_the_setpoint():
    """Implementations MUST clear ramp_setpoint() in stop_ramp(), like the target."""
    from i2as.virtual_instruments.magnet.superconducting_magnet import (
        SuperconductingMagnetVI,
    )

    driver = SimOxfordIPS120("SIM")
    vi = SuperconductingMagnetVI({"main": driver}, amperes_per_tesla=10.0)
    vi.start_ramp(1.0)
    assert vi.ramp_setpoint() is not None

    vi.stop_ramp()
    assert vi.ramp_setpoint() is None
    assert vi.ramp_target() is None


def test_temperature_ramp_setpoint_trails_the_target():
    """A time-based ramp's commanded setpoint walks from start toward target."""
    from i2as.virtual_instruments.temperature.sample_temperature_controller import (
        SampleTemperatureControllerVI,
    )

    driver = SimLakeshore335("SIM")
    vi = SampleTemperatureControllerVI(
        {"main": driver}, default_ramp_rate=1.0, tolerance=2.0
    )
    start_T = driver.get_temperature()
    vi.start_ramp(start_T - 100.0)

    setpoint = vi.ramp_setpoint()
    assert setpoint is not None
    assert vi.ramp_target() == pytest.approx(start_T - 100.0)
    # A 1 K/min ramp has barely moved on its first tick, so the commanded
    # setpoint is still near the start and nowhere near the end setpoint.
    assert setpoint == pytest.approx(start_T, abs=1.0)
    assert setpoint != vi.ramp_target()

    vi.stop_ramp()
    assert vi.ramp_setpoint() is None


# 17. Standby-provenance standard: standby_status() (see BaseVirtualInstrument's
#     __init_subclass__ wrap of standby()/start_ramp()/stop_ramp())
def test_standby_status_fresh_rampable_vi_is_away():
    """Nothing has commanded standby() yet on a freshly built rampable VI."""
    from i2as.virtual_instruments.magnet.superconducting_magnet import (
        SuperconductingMagnetVI,
    )

    driver = SimOxfordIPS120("SIM")
    vi = SuperconductingMagnetVI({"main": driver}, amperes_per_tesla=10.0)

    assert vi.standby_status() == "away"


def test_standby_status_lifecycle_converging_then_reached():
    """standby() -> converging while the ramp-to-zero is still running, reached once it finishes."""
    from i2as.virtual_instruments.magnet.superconducting_magnet import (
        SuperconductingMagnetVI,
    )

    driver = SimOxfordIPS120("SIM")
    vi = SuperconductingMagnetVI(
        {"main": driver}, default_ramp_rate=1200.0, amperes_per_tesla=10.0
    )

    # Move away from the safe idle state first, so standby() has ground to cover.
    vi.start_ramp(1.0)
    for _ in range(50):
        vi.advance_ramp()
        driver._last_update = time.time() - 0.5
        driver._update_simulation()
    assert vi.standby_status() == "away"  # start_ramp, not standby, was last commanded

    vi.standby()
    assert vi.standby_status() == "converging"

    for _ in range(50):
        vi.advance_ramp()
        driver._last_update = time.time() - 0.5
        driver._update_simulation()
    assert vi.ramp_status() == "TARGET_REACHED"
    assert vi.standby_status() == "reached"


def test_standby_status_start_ramp_after_standby_is_away_even_mid_ramp():
    """start_ramp() invalidates standby provenance even while ramp_status() still reports RAMPING.

    Load-bearing: a check based on magnet_state() alone (rather than the
    _standby_commanded flag) would still see the field converging on some
    setpoint here and misreport "converging".
    """
    from i2as.virtual_instruments.magnet.superconducting_magnet import (
        SuperconductingMagnetVI,
    )

    driver = SimOxfordIPS120("SIM")
    vi = SuperconductingMagnetVI(
        {"main": driver}, default_ramp_rate=1200.0, amperes_per_tesla=10.0
    )

    vi.start_ramp(1.0)
    for _ in range(50):
        vi.advance_ramp()
        driver._last_update = time.time() - 0.5
        driver._update_simulation()

    vi.standby()
    assert vi.standby_status() == "converging"

    vi.start_ramp(0.5)
    assert vi.ramp_status() == "RAMPING"
    assert vi.standby_status() == "away"


def test_standby_status_stop_ramp_after_standby_is_away():
    from i2as.virtual_instruments.magnet.superconducting_magnet import (
        SuperconductingMagnetVI,
    )

    driver = SimOxfordIPS120("SIM")
    vi = SuperconductingMagnetVI(
        {"main": driver}, default_ramp_rate=1200.0, amperes_per_tesla=10.0
    )

    vi.start_ramp(1.0)
    for _ in range(50):
        vi.advance_ramp()
        driver._last_update = time.time() - 0.5
        driver._update_simulation()

    vi.standby()
    assert vi.standby_status() == "converging"

    vi.stop_ramp()
    assert vi.standby_status() == "away"


def test_standby_status_raising_standby_leaves_it_away():
    """A standby() that raises must not be mistaken for a converging/reached VI."""
    from i2as.virtual_instruments.base import BaseVirtualInstrument
    from i2as.virtual_instruments.rampable import RampableVI

    class RaisingStandbyRampable(BaseVirtualInstrument, RampableVI):
        vi_type = "mock"

        def __init__(self):
            super().__init__({})
            self._ramping = False

        def start_ramp(self, target):
            self._ramping = True

        def advance_ramp(self):
            pass

        def ramp_status(self):
            return "RAMPING" if self._ramping else "IDLE"

        def stop_ramp(self):
            self._ramping = False

        def standby(self):
            raise RuntimeError("standby failed")

    vi = RaisingStandbyRampable()
    with pytest.raises(RuntimeError):
        vi.standby()
    assert vi.standby_status() == "away"


def test_standby_status_non_rampable_vi_always_reached():
    """A measurement VI (or any non-RampableVI) has no intermediate state to converge through."""
    from i2as.virtual_instruments.measurement.dc_separate_measurement import (
        DCSeparateMeasurementVI,
    )

    source = SimKeithley6221("SIM")
    meter = SimKeithley2182A("SIM")
    source._paired_meter = meter
    vi = DCSeparateMeasurementVI({"source": source, "meter": meter})

    assert vi.standby_status() == "reached"
    vi.standby()
    assert vi.standby_status() == "reached"


def test_standby_status_inherited_standby_still_tracked():
    """A subclass that does NOT define its own standby() still gets correct provenance.

    Exercises the inherited-wrap path in __init_subclass__: only a directly
    defined standby()/start_ramp()/stop_ramp() is re-wrapped, so this
    subclass relies entirely on the wrap already applied to its parent.
    """
    from i2as.virtual_instruments.magnet.superconducting_magnet import (
        SuperconductingMagnetVI,
    )

    class UnmodifiedMagnetVI(SuperconductingMagnetVI):
        """Adds nothing; standby()/start_ramp()/stop_ramp() are all inherited."""

    driver = SimOxfordIPS120("SIM")
    vi = UnmodifiedMagnetVI(
        {"main": driver}, default_ramp_rate=1200.0, amperes_per_tesla=10.0
    )

    assert vi.standby_status() == "away"

    vi.start_ramp(1.0)
    for _ in range(50):
        vi.advance_ramp()
        driver._last_update = time.time() - 0.5
        driver._update_simulation()

    vi.standby()
    assert vi.standby_status() == "converging"

    for _ in range(50):
        vi.advance_ramp()
        driver._last_update = time.time() - 0.5
        driver._update_simulation()
    assert vi.standby_status() == "reached"


# 3c. The declaration standard: measurement_parameters reaches the controls
# One declaration on the VI has to render in two places — the procedure form
# and the instrument front panel — so MeasurementInstrumentBase installs it as
# the specs of the two controls whose parameters ARE measurement parameters.


def _minimal_measurement_vi(**namespace):
    """Build a throwaway MeasurementInstrumentBase subclass for spec checks.

    Args:
        **namespace: Class attributes and methods to place on the subclass.

    Returns:
        The freshly created class (never constructed or registered).
    """
    from i2as.virtual_instruments.base import MeasurementInstrumentBase

    return type("ThrowawayMeasurementVI", (MeasurementInstrumentBase,), namespace)


def test_measurement_parameters_install_on_arming_control():
    """A bare initiate_measurement inherits every measurement_parameters spec."""
    from i2as.core.decorators import control, get_control_specs
    from i2as.core.plan import ParamSpec

    specs = {
        "current_A": ParamSpec(type=float, default=1e-6, unit="A", description="I"),
        "n_readings": ParamSpec(type=int, default=10, description="N"),
    }

    @control
    def initiate_measurement(self, current_A=1e-6, n_readings=10):
        pass

    vi_cls = _minimal_measurement_vi(
        measurement_parameters=specs,
        initiate_measurement=initiate_measurement,
    )
    assert get_control_specs(vi_cls.initiate_measurement) == specs


def test_measurement_parameters_install_on_reading_setter():
    """A reading_setters setter inherits exactly its own single spec."""
    from i2as.core.decorators import control, get_control_specs
    from i2as.core.plan import ParamSpec

    current_spec = ParamSpec(type=float, default=1e-6, unit="A", description="I")
    specs = {
        "current_A": current_spec,
        "n_readings": ParamSpec(type=int, default=10, description="N"),
    }

    @control
    def initiate_measurement(self, current_A=1e-6, n_readings=10):
        pass

    @control
    def set_source_current(self, current_A=1e-6):
        pass

    vi_cls = _minimal_measurement_vi(
        measurement_parameters=specs,
        reading_setters={"current_A": "set_source_current"},
        initiate_measurement=initiate_measurement,
        set_source_current=set_source_current,
    )
    assert get_control_specs(vi_cls.set_source_current) == {"current_A": current_spec}


def test_explicit_control_specs_are_not_overwritten():
    """An explicit params= wins; the install only fills a control that declared none."""
    from i2as.core.decorators import control, get_control_specs
    from i2as.core.plan import ParamSpec

    declared = ParamSpec(type=float, default=0.0, unit="A", description="Declared")

    @control(params={"current_A": declared})
    def set_source_current(self, current_A=0.0):
        pass

    @control
    def initiate_measurement(self, current_A=1e-6):
        pass

    vi_cls = _minimal_measurement_vi(
        measurement_parameters={
            "current_A": ParamSpec(
                type=float, default=1e-6, unit="A", description="Installed"
            ),
        },
        reading_setters={"current_A": "set_source_current"},
        initiate_measurement=initiate_measurement,
        set_source_current=set_source_current,
    )
    assert get_control_specs(vi_cls.set_source_current) == {"current_A": declared}


def test_arming_signature_must_match_measurement_parameters():
    """A signature that drifts from measurement_parameters fails at import."""
    from i2as.core.decorators import control
    from i2as.core.plan import ParamSpec

    @control
    def initiate_measurement(self, current_A=1e-6, stray=1):
        pass

    with pytest.raises(ValueError, match="measurement_parameters"):
        _minimal_measurement_vi(
            measurement_parameters={
                "current_A": ParamSpec(type=float, default=1e-6, description="I"),
            },
            initiate_measurement=initiate_measurement,
        )


# 18. Lifecycle-state standard: lifecycle_state() (see BaseVirtualInstrument's
#     "Lifecycle-state standard" — the verbs own the fact, the read is pure,
#     and an observing VI may correct the cache from the monitor cycle)


def test_lifecycle_state_starts_idle_and_follows_the_verbs():
    """A fresh VI is idle; initiate() and standby() move it, in either order."""
    from i2as.virtual_instruments.magnet.superconducting_magnet import (
        SuperconductingMagnetVI,
    )

    driver = SimOxfordIPS120("SIM")
    vi = SuperconductingMagnetVI({"main": driver}, amperes_per_tesla=10.0)

    assert vi.lifecycle_state() == "idle"
    vi.initiate()
    assert vi.lifecycle_state() == "initiated"
    vi.standby()
    assert vi.lifecycle_state() == "standby"
    vi.initiate()
    assert vi.lifecycle_state() == "initiated"


def test_lifecycle_state_resets_to_idle_on_disconnect():
    """The release hook drops the fact with the session (the standard's rule 1)."""
    from i2as.virtual_instruments.magnet.superconducting_magnet import (
        SuperconductingMagnetVI,
    )

    driver = SimOxfordIPS120("SIM")
    vi = SuperconductingMagnetVI({"main": driver}, amperes_per_tesla=10.0)
    vi.initiate()
    assert vi.lifecycle_state() == "initiated"

    vi.disconnect()
    assert vi.lifecycle_state() == "idle"


def test_lifecycle_state_is_untouched_when_initiate_raises():
    """A refused initiate() must never leave a card claiming the instrument runs."""

    class RaisingInitiateVI(BaseVirtualInstrument):
        vi_type = "mock"

        def __init__(self):
            super().__init__({})

        def initiate(self):
            raise RuntimeError("could not initiate")

    vi = RaisingInitiateVI()
    with pytest.raises(RuntimeError):
        vi.initiate()
    assert vi.lifecycle_state() == "idle"


def test_lifecycle_state_inherited_verbs_are_still_tracked():
    """A subclass that overrides neither verb inherits the already-wrapped ones."""
    from i2as.virtual_instruments.magnet.superconducting_magnet import (
        SuperconductingMagnetVI,
    )

    class UnmodifiedMagnetVI(SuperconductingMagnetVI):
        """Adds nothing; initiate()/standby() are both inherited."""

    vi = UnmodifiedMagnetVI({"main": SimOxfordIPS120("SIM")}, amperes_per_tesla=10.0)

    assert vi.lifecycle_state() == "idle"
    vi.initiate()
    assert vi.lifecycle_state() == "initiated"
    vi.standby()
    assert vi.lifecycle_state() == "standby"


def test_measurement_vi_arming_counts_as_initiated():
    """initiate_measurement() is a measurement VI's own initiate (the standard).

    Its plain ``initiate()`` is only a connection check, so a VI armed by a
    procedure would otherwise still read "idle" while it is sourcing current.
    """
    from i2as.virtual_instruments.measurement.dc_separate_measurement import (
        DCSeparateMeasurementVI,
    )

    source = SimKeithley6221("SIM")
    meter = SimKeithley2182A("SIM")
    source._paired_meter = meter
    vi = DCSeparateMeasurementVI({"source": source, "meter": meter})

    assert vi.lifecycle_state() == "idle"
    vi.initiate_measurement(current_A=1e-6)
    assert vi.lifecycle_state() == "initiated"
    vi.standby()
    assert vi.lifecycle_state() == "standby"


def test_measurement_vi_plain_initiate_also_counts_as_initiated():
    """The connection-check ``initiate()`` still records the operator's verb."""
    from i2as.virtual_instruments.measurement.dc_separate_measurement import (
        DCSeparateMeasurementVI,
    )

    source = SimKeithley6221("SIM")
    meter = SimKeithley2182A("SIM")
    source._paired_meter = meter
    vi = DCSeparateMeasurementVI({"source": source, "meter": meter})

    vi.initiate()
    assert vi.lifecycle_state() == "initiated"


def test_observe_lifecycle_state_refreshes_the_cache_on_the_monitor_cycle():
    """A VI that can read the instrument corrects the cached value at poll time."""

    class ObservingVI(BaseVirtualInstrument):
        vi_type = "mock"

        def __init__(self):
            super().__init__({})
            self.output_on = False

        @monitored
        def output(self):
            return self.output_on

        def observe_lifecycle_state(self):
            return "initiated" if self.last_monitored("output") else "standby"

    vi = ObservingVI()
    vi.initiate()
    assert vi.lifecycle_state() == "initiated"

    # Somebody turned the output off at the instrument's own front panel.
    vi.get_state()
    assert vi.lifecycle_state() == "standby"

    vi.output_on = True
    vi.get_state()
    assert vi.lifecycle_state() == "initiated"


def test_observe_lifecycle_state_failures_never_break_the_monitor_cycle():
    """A raising or out-of-vocabulary observation is ignored, not propagated."""

    class BadObserverVI(BaseVirtualInstrument):
        vi_type = "mock"

        def __init__(self, answer):
            super().__init__({})
            self._answer = answer

        @monitored
        def reading(self):
            return 1.0

        def observe_lifecycle_state(self):
            if self._answer is RuntimeError:
                raise RuntimeError("cannot tell")
            return self._answer

    for answer in (RuntimeError, "armed"):
        vi = BadObserverVI(answer)
        vi.initiate()
        state = vi.get_state()
        assert state["reading"] == 1.0
        assert vi.lifecycle_state() == "initiated"
