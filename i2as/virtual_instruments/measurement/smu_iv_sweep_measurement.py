# ---
# description: |
#   SMUIVSweepMeasurementVI: one full I–V curve per procedure point on a
#   single source-measure unit (e.g. Keithley 2400). Steps the source linearly
#   from start to stop (voltage or current), measures both quantities at each
#   step and returns the curve as arrays, plus the low-bias resistance fitted
#   from it. Declares the (voltage, current) arrays as a per-point curve, so
#   the live plot draws each point's I–V.
# entry_point: Not run directly; instantiated by Station factory.
# dependencies:
#   - i2as.virtual_instruments.base (MeasurementInstrumentBase)
#   - i2as.virtual_instruments.measurement.smu_dc_measurement (limits, modes)
# input: |
#   drivers = {"smu": <source-measure unit driver>}
#   initiate_measurement(source_function, sweep_start_V, sweep_stop_V,
#   sweep_start_A, sweep_stop_A, points, compliance_A, compliance_V, nplc)
# process: |
#   initiate_measurement() re-asserts the source configuration at level 0 and
#   switches the output on. take_reading() walks the levels, reads each, and
#   parks the source back at 0. standby() switches the output off.
# output: |
#   voltage_V_array / current_A_array of length points (the curve), their
#   mean/SEM companions, and resistance_ohm(_error): the low-bias resistance.
# last_updated: 2026-10-31
# ---

"""SMUIVSweepMeasurementVI — a full I–V curve at every procedure point."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from i2as.core.decorators import control
from i2as.core.plan import ParamSpec
from i2as.virtual_instruments.base import (
    EXCITATION_CURRENT_LIMIT,
    MeasurementInstrumentBase,
    _populate_excitation_current_limit,
)
from i2as.virtual_instruments.measurement.smu_dc_measurement import (
    SOURCE_FUNCTIONS,
    SOURCE_VOLTAGE_LIMIT,
    populate_source_voltage_limit,
)

_NOT_INITIATED = object()

#: Fraction of the sweep's largest level used for the low-bias resistance fit.
LOW_BIAS_FRACTION = 0.1


class SMUIVSweepMeasurementVI(MeasurementInstrumentBase):
    """An I–V sweep as a measurement method: one whole curve per point.

    Where DC mode reads one operating point many times, this method reads
    many operating points once: at each procedure point it steps the source
    linearly over ``points`` levels from start to stop — volts in
    ``"voltage"`` mode, amperes in ``"current"`` mode — and returns the
    measured voltage and current at every level. A Time Series of this
    method records I–V against time; a Field Sweep records I–V against field
    (e.g. two points, +H and −H, give the parallel and antiparallel curves of
    a tunnel junction); a Temperature Sweep, I–V against temperature.

    Each curve is also reduced to the low-bias resistance — the inverse
    slope of a line fitted to the levels within ``LOW_BIAS_FRACTION`` of the
    largest — so a sweep can plot R(H) or R(T) beside the curves.

    Both sweep end points are bounded by the setup's ``max_source_voltage_V``
    and ``max_source_current_A`` (the control-validation standard). The
    source is parked at zero between curves.
    """

    display_label: str = "I–V sweep (SMU)"
    selector_label: ClassVar[str] = "IV sweep (2400)"

    control_limits: ClassVar[dict[str, dict[str, str]]] = {
        "initiate_measurement": {
            "sweep_start_V": SOURCE_VOLTAGE_LIMIT,
            "sweep_stop_V": SOURCE_VOLTAGE_LIMIT,
            "sweep_start_A": EXCITATION_CURRENT_LIMIT,
            "sweep_stop_A": EXCITATION_CURRENT_LIMIT,
        },
    }

    _ARRAY_KEYS, _SCALAR_COLUMNS = MeasurementInstrumentBase.quantity_columns(
        "voltage_V", "current_A"
    )
    measurement_data_keys: ClassVar[list[str]] = _ARRAY_KEYS
    measurement_scalar_columns: ClassVar[dict[str, str]] = {
        **_SCALAR_COLUMNS,
        "resistance_ohm": "float",
        "resistance_ohm_error": "float",
    }
    measurement_curve_keys: ClassVar[tuple[str, ...]] = ("voltage_V_array", "current_A_array")
    measurement_parameters: ClassVar[dict[str, ParamSpec]] = {
        "source_function": ParamSpec(
            type=str,
            default="voltage",
            choices=SOURCE_FUNCTIONS,
            description="What the sweep steps: the voltage (current measured) or the current (voltage measured)",
        ),
        "sweep_start_V": ParamSpec(type=float, default=-0.5, unit="V", description="First level (voltage mode)"),
        "sweep_stop_V": ParamSpec(type=float, default=0.5, unit="V", description="Last level (voltage mode)"),
        "sweep_start_A": ParamSpec(type=float, default=-1e-4, unit="A", description="First level (current mode)"),
        "sweep_stop_A": ParamSpec(type=float, default=1e-4, unit="A", description="Last level (current mode)"),
        "points": ParamSpec(type=int, default=41, min=3, description="Levels per I–V curve"),
        "compliance_A": ParamSpec(
            type=float, default=1e-3, unit="A", description="Current compliance (voltage mode)"
        ),
        "compliance_V": ParamSpec(
            type=float, default=1.0, unit="V", description="Voltage compliance (current mode)"
        ),
        "nplc": ParamSpec(
            type=float, default=1.0, min=0.01, max=10.0,
            description="Integration time per level, in power-line cycles",
        ),
    }

    def __init__(self, drivers: dict[str, object], **init_params: Any) -> None:
        """Bind the SMU and read the setup's current and voltage ceilings."""
        super().__init__(drivers, **init_params)
        self._smu = drivers["smu"]
        _populate_excitation_current_limit(self, init_params)
        populate_source_voltage_limit(self, init_params)
        self._function: object = _NOT_INITIATED
        self._levels: np.ndarray = np.zeros(0)

    def data_arrays(self, params: Mapping[str, Any]) -> dict[str, int]:
        """Return each array's per-point length (``points``)."""
        n = int(params["points"])
        return {key: n for key in self.measurement_data_keys}

    # panel=False: arming is a deliberate act — front panel and procedures only.
    @control(panel=False, action_class="run_control")
    def initiate_measurement(
        self,
        source_function: str = "voltage",
        sweep_start_V: float = -0.5,
        sweep_stop_V: float = 0.5,
        sweep_start_A: float = -1e-4,
        sweep_stop_A: float = 1e-4,
        points: int = 41,
        compliance_A: float = 1e-3,
        compliance_V: float = 1.0,
        nplc: float = 1.0,
    ) -> None:
        """Re-assert the source configuration at level 0 and switch the output on.

        Args:
            source_function: ``"voltage"`` or ``"current"``.
            sweep_start_V: First level in voltage mode, volts.
            sweep_stop_V: Last level in voltage mode, volts.
            sweep_start_A: First level in current mode, amperes.
            sweep_stop_A: Last level in current mode, amperes.
            points: Levels per curve.
            compliance_A: Current compliance (voltage mode), amperes.
            compliance_V: Voltage compliance (current mode), volts.
            nplc: Integration time per level, power-line cycles.

        Raises:
            ValueError: If *source_function* is neither mode or *points* < 3.
        """
        if source_function not in SOURCE_FUNCTIONS.values():
            raise ValueError(f"source_function must be 'voltage' or 'current', got {source_function!r}")
        if int(points) < 3:
            raise ValueError(f"an I–V sweep needs at least 3 points, got {points}")
        smu = self._smu
        smu.set_output(False)  # type: ignore[attr-defined]
        smu.set_nplc(float(nplc))  # type: ignore[attr-defined]
        if source_function == "voltage":
            smu.set_source_function("VOLT")  # type: ignore[attr-defined]
            smu.set_current_compliance(float(compliance_A))  # type: ignore[attr-defined]
            smu.set_source_voltage(0.0)  # type: ignore[attr-defined]
            self._levels = np.linspace(float(sweep_start_V), float(sweep_stop_V), int(points))
        else:
            smu.set_source_function("CURR")  # type: ignore[attr-defined]
            smu.set_voltage_compliance(float(compliance_V))  # type: ignore[attr-defined]
            smu.set_source_current(0.0)  # type: ignore[attr-defined]
            self._levels = np.linspace(float(sweep_start_A), float(sweep_stop_A), int(points))
        smu.set_output(True)  # type: ignore[attr-defined]
        self._function = source_function

    def take_reading(self) -> dict[str, list[float] | float]:
        """Sweep the levels once and return the curve and its low-bias resistance.

        Raises:
            RuntimeError: If ``initiate_measurement()`` has not been called.
        """
        if self._function is _NOT_INITIATED:
            raise RuntimeError("initiate_measurement() must be called before take_reading().")
        smu = self._smu
        setter = smu.set_source_voltage if self._function == "voltage" else smu.set_source_current  # type: ignore[attr-defined]
        voltages: list[float] = []
        currents: list[float] = []
        for level in self._levels:
            setter(float(level))
            voltage, current = smu.measure()  # type: ignore[attr-defined]
            voltages.append(float(voltage))
            currents.append(float(current))
        setter(0.0)
        reading: dict[str, list[float] | float] = {}
        for name, samples in (("voltage_V", voltages), ("current_A", currents)):
            mean, error = self.mean_and_sem(samples)
            reading[f"{name}_array"] = samples
            reading[name] = mean
            reading[f"{name}_error"] = error
        r, r_err = self.low_bias_resistance(voltages, currents)
        reading["resistance_ohm"] = r
        reading["resistance_ohm_error"] = r_err
        return reading

    @staticmethod
    def low_bias_resistance(voltages: list[float], currents: list[float]) -> tuple[float, float]:
        """Fit V = R I + V0 over the low-bias part of a curve.

        Args:
            voltages: The measured voltages, volts.
            currents: The measured currents, amperes.

        Returns:
            ``(R, standard error of R)`` in ohms; NaNs when fewer than three
            low-bias points carry current.
        """
        v = np.asarray(voltages, float)
        i = np.asarray(currents, float)
        reach = max(np.abs(v).max(), 1e-12)
        mask = np.abs(v) <= LOW_BIAS_FRACTION * reach + 1e-15
        if mask.sum() < 3:
            mask = np.argsort(np.abs(v))[:3]
        x, y = i[mask], v[mask]
        if np.ptp(x) == 0.0:
            return float("nan"), float("nan")
        coeffs, cov = np.polyfit(x, y, 1, cov=True) if len(x) > 3 else (np.polyfit(x, y, 1), None)
        error = float(np.sqrt(cov[0, 0])) if cov is not None else float("nan")
        return float(coeffs[0]), error

    def standby(self) -> None:
        """Zero both levels, switch the output off and reset the armed state."""
        smu = self._smu
        smu.set_source_voltage(0.0)  # type: ignore[attr-defined]
        smu.set_source_current(0.0)  # type: ignore[attr-defined]
        smu.set_output(False)  # type: ignore[attr-defined]
        self._function = _NOT_INITIATED
