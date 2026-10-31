# ---
# description: |
#   SMUDCMeasurementVI: DC mode on a single source-measure unit (e.g. Keithley
#   2400). Sources a fixed current (measuring voltage, protected by a voltage
#   compliance) or a fixed voltage (measuring current, protected by a current
#   compliance) and returns readings_per_point conversions of both, plus the
#   resistance of each. reading_setters let a procedure's reading loop step
#   the source level without re-arming.
# entry_point: Not run directly; instantiated by Station factory.
# dependencies:
#   - i2as.virtual_instruments.base (MeasurementInstrumentBase)
#   - i2as.core.decorators (control, monitored)
# input: |
#   drivers = {"smu": <source-measure unit driver>}
#   initiate_measurement(source_function, source_voltage_V, source_current_A,
#   compliance_A, compliance_V, nplc, readings_per_point) before take_reading().
# process: |
#   initiate_measurement() re-asserts the source function, compliance, level
#   and integration time, then switches the output on. take_reading() triggers
#   readings_per_point conversions. standby() zeroes the levels and switches
#   the output off.
# output: |
#   Mean/error/array triple for voltage_V, current_A and resistance_ohm, arrays
#   of length readings_per_point.
# last_updated: 2026-10-31
# ---

"""SMUDCMeasurementVI — DC mode on one source-measure unit (source I or V)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

from i2as.core.decorators import control, monitored
from i2as.core.plan import ParamSpec
from i2as.virtual_instruments.base import (
    EXCITATION_CURRENT_LIMIT,
    MeasurementInstrumentBase,
    _populate_excitation_current_limit,
)

_NOT_INITIATED = object()

#: ``control_limits`` name for a directly sourced voltage across the sample.
SOURCE_VOLTAGE_LIMIT: str = "source_voltage_V"
#: The config ``init_params`` key that sets it, volts (symmetric).
MAX_SOURCE_VOLTAGE_KEY: str = "max_source_voltage_V"

#: The two source functions, as the form offers them.
SOURCE_FUNCTIONS: dict[str, str] = {
    "Voltage (measure I)": "voltage",
    "Current (measure V)": "current",
}


def populate_source_voltage_limit(vi: MeasurementInstrumentBase, init_params: Mapping[str, Any]) -> None:
    """Set ``vi._limits[SOURCE_VOLTAGE_LIMIT]`` from ``max_source_voltage_V``.

    Symmetric, like the current ceiling; absent means unbounded (every
    shipped config declares it).
    """
    raw = init_params.get(MAX_SOURCE_VOLTAGE_KEY)
    ceiling = abs(float(raw)) if raw is not None else None
    vi._limits[SOURCE_VOLTAGE_LIMIT] = (-ceiling, ceiling) if ceiling is not None else (None, None)


def resistances(voltages: list[float], currents: list[float]) -> list[float]:
    """Return V/I per conversion, NaN where no current flowed."""
    return [v / i if i != 0.0 else math.nan for v, i in zip(voltages, currents)]


class SMUDCMeasurementVI(MeasurementInstrumentBase):
    """DC mode with one source-measure unit: source I or V, measure both.

    ``source_function`` picks what is held fixed:

    * ``"current"`` — source ``source_current_A``, measure the voltage; the
      ``compliance_V`` voltage compliance is what keeps a high-impedance or
      fragile sample (a tunnel barrier) from seeing more than that voltage.
    * ``"voltage"`` — source ``source_voltage_V``, measure the current,
      limited by ``compliance_A``. A tunnel junction's TMR is measured this
      way: the bias sets the tunnelling physics, so it is what is held.

    Both levels are bounded by the setup (the control-validation standard):
    ``max_source_current_A`` and ``max_source_voltage_V`` in the config.
    Every arming re-asserts the source function, compliance, level and
    integration time first (the shared-instrument mode discipline), because
    another method on the same instrument may have left it otherwise.

    Driver contract (``"smu"``): ``set_source_function``,
    ``set_source_voltage``, ``set_source_current``, ``set_current_compliance``,
    ``set_voltage_compliance``, ``set_nplc``, ``set_output``,
    ``measure() -> (V, I)``, ``get_idn``.
    """

    display_label: str = "DC mode (SMU)"
    selector_label: ClassVar[str] = "DC mode (2400)"

    reading_setters: ClassVar[dict[str, str]] = {
        "source_voltage_V": "set_source_voltage",
        "source_current_A": "set_source_current",
    }
    control_limits: ClassVar[dict[str, dict[str, str]]] = {
        "initiate_measurement": {
            "source_voltage_V": SOURCE_VOLTAGE_LIMIT,
            "source_current_A": EXCITATION_CURRENT_LIMIT,
        },
        "set_source_voltage": {"source_voltage_V": SOURCE_VOLTAGE_LIMIT},
        "set_source_current": {"source_current_A": EXCITATION_CURRENT_LIMIT},
    }

    _ARRAY_KEYS, _SCALAR_COLUMNS = MeasurementInstrumentBase.quantity_columns(
        "voltage_V", "current_A", "resistance_ohm"
    )
    measurement_data_keys: ClassVar[list[str]] = _ARRAY_KEYS
    measurement_scalar_columns: ClassVar[dict[str, str]] = _SCALAR_COLUMNS
    measurement_parameters: ClassVar[dict[str, ParamSpec]] = {
        "source_function": ParamSpec(
            type=str,
            default="voltage",
            choices=SOURCE_FUNCTIONS,
            description="What the source holds fixed: a voltage (current measured) or a current (voltage measured)",
        ),
        "source_voltage_V": ParamSpec(
            type=float, default=0.05, unit="V", description="Source voltage (used in voltage mode)"
        ),
        "source_current_A": ParamSpec(
            type=float, default=1e-5, unit="A", description="Source current (used in current mode)"
        ),
        "compliance_A": ParamSpec(
            type=float, default=1e-3, unit="A", description="Current compliance (voltage mode)"
        ),
        "compliance_V": ParamSpec(
            type=float, default=1.0, unit="V", description="Voltage compliance (current mode)"
        ),
        "nplc": ParamSpec(
            type=float, default=1.0, min=0.01, max=10.0,
            description="Integration time per conversion, in power-line cycles",
        ),
        "readings_per_point": ParamSpec(
            type=int, default=5, min=1, description="Conversions averaged per point"
        ),
    }

    def __init__(self, drivers: dict[str, object], **init_params: Any) -> None:
        """Bind the SMU and read the setup's current and voltage ceilings."""
        super().__init__(drivers, **init_params)
        self._smu = drivers["smu"]
        _populate_excitation_current_limit(self, init_params)
        populate_source_voltage_limit(self, init_params)
        self._function: object = _NOT_INITIATED
        self._readings_per_point = 5
        self._last_reading: dict[str, Any] | None = None

    def data_arrays(self, params: Mapping[str, Any]) -> dict[str, int]:
        """Return each array's per-point length (``readings_per_point``)."""
        n = int(params["readings_per_point"])
        return {key: n for key in self.measurement_data_keys}

    @control(panel=False, action_class="run_control")
    def initiate_measurement(
        self,
        source_function: str = "voltage",
        source_voltage_V: float = 0.05,
        source_current_A: float = 1e-5,
        compliance_A: float = 1e-3,
        compliance_V: float = 1.0,
        nplc: float = 1.0,
        readings_per_point: int = 5,
    ) -> None:
        """Re-assert the whole source configuration, then switch the output on.

        Args:
            source_function: ``"voltage"`` or ``"current"``.
            source_voltage_V: Level in voltage mode, volts.
            source_current_A: Level in current mode, amperes.
            compliance_A: Current compliance (voltage mode), amperes.
            compliance_V: Voltage compliance (current mode), volts.
            nplc: Integration time per conversion, power-line cycles.
            readings_per_point: Conversions ``take_reading()`` collects.

        Raises:
            ValueError: If *source_function* is neither mode.
        """
        if source_function not in SOURCE_FUNCTIONS.values():
            raise ValueError(f"source_function must be 'voltage' or 'current', got {source_function!r}")
        smu = self._smu
        smu.set_output(False)  # type: ignore[attr-defined]
        smu.set_nplc(float(nplc))  # type: ignore[attr-defined]
        if source_function == "voltage":
            smu.set_source_function("VOLT")  # type: ignore[attr-defined]
            smu.set_current_compliance(float(compliance_A))  # type: ignore[attr-defined]
            smu.set_source_voltage(float(source_voltage_V))  # type: ignore[attr-defined]
        else:
            smu.set_source_function("CURR")  # type: ignore[attr-defined]
            smu.set_voltage_compliance(float(compliance_V))  # type: ignore[attr-defined]
            smu.set_source_current(float(source_current_A))  # type: ignore[attr-defined]
        smu.set_output(True)  # type: ignore[attr-defined]
        self._function = source_function
        self._readings_per_point = int(readings_per_point)

    def take_reading(self) -> dict[str, list[float] | float]:
        """Collect ``readings_per_point`` conversions at the armed level.

        Raises:
            RuntimeError: If ``initiate_measurement()`` has not been called.
        """
        if self._function is _NOT_INITIATED:
            raise RuntimeError("initiate_measurement() must be called before take_reading().")
        voltages: list[float] = []
        currents: list[float] = []
        for _ in range(self._readings_per_point):
            voltage, current = self._smu.measure()  # type: ignore[attr-defined]
            voltages.append(float(voltage))
            currents.append(float(current))
        reading: dict[str, list[float] | float] = {}
        for name, samples in (
            ("voltage_V", voltages),
            ("current_A", currents),
            ("resistance_ohm", resistances(voltages, currents)),
        ):
            mean, error = self.mean_and_sem(samples)
            reading[f"{name}_array"] = samples
            reading[name] = mean
            reading[f"{name}_error"] = error
        return reading

    @control(action_class="run_control")
    def set_source_voltage(self, source_voltage_V: float) -> None:
        """Step the voltage level without re-arming.

        Programs the level whatever the source function, as the instrument
        does: it takes effect in voltage mode and is held for it otherwise.

        Args:
            source_voltage_V: The level, volts (may be negative).

        Raises:
            RuntimeError: If ``initiate_measurement()`` has not been called.
        """
        if self._function is _NOT_INITIATED:
            raise RuntimeError("initiate_measurement() must be called before set_source_voltage().")
        self._smu.set_source_voltage(float(source_voltage_V))  # type: ignore[attr-defined]

    @control(action_class="run_control")
    def set_source_current(self, source_current_A: float) -> None:
        """Step the current level without re-arming.

        Programs the level whatever the source function, as the instrument
        does: it takes effect in current mode and is held for it otherwise.

        Args:
            source_current_A: The level, amperes (may be negative).

        Raises:
            RuntimeError: If ``initiate_measurement()`` has not been called.
        """
        if self._function is _NOT_INITIATED:
            raise RuntimeError("initiate_measurement() must be called before set_source_current().")
        self._smu.set_source_current(float(source_current_A))  # type: ignore[attr-defined]

    def standby(self) -> None:
        """Zero both levels, switch the output off and reset the armed state."""
        smu = self._smu
        smu.set_source_voltage(0.0)  # type: ignore[attr-defined]
        smu.set_source_current(0.0)  # type: ignore[attr-defined]
        smu.set_output(False)  # type: ignore[attr-defined]
        self._function = _NOT_INITIATED
        self._last_reading = None

    @control(panel=False, action_class="read")
    def read_now(self) -> None:
        """Take one reading at the armed level and cache it for the monitored fields.

        Raises:
            RuntimeError: If ``initiate_measurement()`` has not been called.
        """
        self._last_reading = self.take_reading()

    @monitored(unit="ohm", description="Resistance from the last manual read")
    def last_resistance_ohm(self) -> float | None:
        """Return the mean resistance of the last ``read_now()``, or None."""
        if self._last_reading is None:
            return None
        return float(self._last_reading["resistance_ohm"])  # type: ignore[arg-type]
