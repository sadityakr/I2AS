# ---
# description: |
#   SMUVoltageBiasMeasurementVI: constant-voltage measurement with a single
#   source-measure unit (e.g. Keithley 2400) — the way a tunnel junction is
#   measured. initiate_measurement() programs the bias, the current
#   compliance and the integration time and switches the output on;
#   take_reading() collects readings_per_point current conversions at that
#   bias and derives the resistance of each. Declares reading_setters
#   {"bias_V": "set_bias_voltage"} so a bias sweep or a reading loop can
#   step the bias without re-arming.
# entry_point: Not run directly; instantiated by Station factory.
# dependencies:
#   - i2as.virtual_instruments.base (MeasurementInstrumentBase)
#   - i2as.core.decorators (control, monitored)
# input: |
#   drivers = {"smu": <source-measure unit driver>}
#   initiate_measurement(bias_V, compliance_A, nplc, readings_per_point)
#   must be called before the argument-less take_reading().
# process: |
#   initiate_measurement() programs the SMU and turns its output on.
#   take_reading() triggers readings_per_point current conversions.
#   set_bias_voltage() reprograms only the bias between readings.
#   standby() ramps nothing: it zeroes the bias and switches the output off.
# output: |
#   Mean/error/array triple per quantity: voltage_V, current_A and
#   resistance_ohm (= V / I per conversion), arrays of length
#   readings_per_point.
# last_updated: 2026-10-31
# ---

"""SMUVoltageBiasMeasurementVI — constant-voltage measurement with one SMU."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

from i2as.core.decorators import control, monitored
from i2as.core.plan import ParamSpec
from i2as.virtual_instruments.base import MeasurementInstrumentBase

_NOT_INITIATED = object()

#: ``control_limits`` limit name for the bias a voltage-biased measurement
#: applies across the sample — the quantity that can destroy a tunnel barrier.
BIAS_VOLTAGE_LIMIT: str = "bias_V"

#: The config ``init_params`` key that sets that limit, volts (symmetric).
MAX_BIAS_KEY: str = "max_bias_V"


class SMUVoltageBiasMeasurementVI(MeasurementInstrumentBase):
    """Constant-voltage (bias) measurement with a single source-measure unit.

    A tunnel junction is characterised at a fixed *voltage*, not a fixed
    current: the bias across the barrier is what sets the tunnelling physics
    (the non-linear I–V, the bias dependence of the TMR, spin-transfer
    torque), so it is the quantity this method holds and the current is the
    one it measures. One driver plays both roles::

        vi.initiate_measurement(bias_V=0.05, compliance_A=1e-4, nplc=1.0,
                                readings_per_point=5)
        data = vi.take_reading()
        # {"voltage_V": float, "voltage_V_error": float, "voltage_V_array": [...],
        #  "current_A": ..., "resistance_ohm": ..., (each as a triple)}

    The bias is bounded symmetrically by the setup's ``max_bias_V`` (the
    control-validation standard): a dielectric breakdown is irreversible, so
    the ceiling is the sample's, set in the config, and a mistyped bias is
    refused before it reaches the instrument.

    Driver contract
    ---------------
    ``"smu"`` driver must implement:
    * ``set_source_voltage(float)``, ``set_current_compliance(float)``,
      ``set_nplc(float)``, ``set_output(bool)``
    * ``measure_current() -> float``
    * ``get_idn() -> str``
    """

    display_label: str = "DC bias (SMU)"
    selector_label: ClassVar[str] = "Voltage bias (2400)"

    reading_setters: ClassVar[dict[str, str]] = {"bias_V": "set_bias_voltage"}

    control_limits: ClassVar[dict[str, dict[str, str]]] = {
        "initiate_measurement": {"bias_V": BIAS_VOLTAGE_LIMIT},
        "set_bias_voltage": {"bias_V": BIAS_VOLTAGE_LIMIT},
    }

    _ARRAY_KEYS, _SCALAR_COLUMNS = MeasurementInstrumentBase.quantity_columns(
        "voltage_V", "current_A", "resistance_ohm"
    )
    measurement_data_keys: ClassVar[list[str]] = _ARRAY_KEYS
    measurement_scalar_columns: ClassVar[dict[str, str]] = _SCALAR_COLUMNS
    measurement_parameters: ClassVar[dict[str, ParamSpec]] = {
        "bias_V": ParamSpec(
            type=float, default=0.05, unit="V", description="DC bias voltage across the sample"
        ),
        "compliance_A": ParamSpec(
            type=float, default=1e-4, unit="A", description="Current compliance"
        ),
        "nplc": ParamSpec(
            type=float,
            default=1.0,
            min=0.01,
            max=10.0,
            description="Integration time per conversion, in power-line cycles",
        ),
        "readings_per_point": ParamSpec(
            type=int, default=5, min=1, description="Current conversions averaged per point"
        ),
    }

    def __init__(self, drivers: dict[str, object], **init_params: Any) -> None:
        """Bind the SMU and read the setup's bias ceiling.

        Args:
            drivers: ``{"smu": <driver>}``.
            **init_params: ``max_bias_V`` bounds the applied bias to
                ``±max_bias_V``; absent means unbounded (every shipped config
                declares it).
        """
        super().__init__(drivers, **init_params)
        self._smu = drivers["smu"]
        raw = init_params.get(MAX_BIAS_KEY)
        ceiling = abs(float(raw)) if raw is not None else None
        self._limits[BIAS_VOLTAGE_LIMIT] = (
            (-ceiling, ceiling) if ceiling is not None else (None, None)
        )
        self._bias_V: object = _NOT_INITIATED
        self._readings_per_point: int = 5
        self._last_reading: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Measurement lifecycle
    # ------------------------------------------------------------------

    def data_arrays(self, params: Mapping[str, Any]) -> dict[str, int]:
        """Return each array's per-point length (``readings_per_point``)."""
        n = int(params["readings_per_point"])
        return {key: n for key in self.measurement_data_keys}

    @control(panel=False, action_class="run_control")
    def initiate_measurement(
        self,
        bias_V: float = 0.05,
        compliance_A: float = 1e-4,
        nplc: float = 1.0,
        readings_per_point: int = 5,
    ) -> None:
        """Program bias, compliance and integration time, then switch the output on.

        Args:
            bias_V: DC bias across the sample, volts.
            compliance_A: Current compliance, amperes.
            nplc: Integration time per conversion, power-line cycles.
            readings_per_point: Conversions ``take_reading()`` collects.
        """
        smu = self._smu
        smu.set_current_compliance(float(compliance_A))  # type: ignore[attr-defined]
        smu.set_nplc(float(nplc))  # type: ignore[attr-defined]
        smu.set_source_voltage(float(bias_V))  # type: ignore[attr-defined]
        smu.set_output(True)  # type: ignore[attr-defined]
        self._bias_V = float(bias_V)
        self._readings_per_point = int(readings_per_point)

    def take_reading(self) -> dict[str, list[float] | float]:
        """Collect ``readings_per_point`` current conversions at the armed bias.

        Returns:
            The mean/error/array triple for ``voltage_V``, ``current_A`` and
            ``resistance_ohm`` (``nan`` for a conversion that read zero).

        Raises:
            RuntimeError: If ``initiate_measurement()`` has not been called.
        """
        if self._bias_V is _NOT_INITIATED:
            raise RuntimeError("initiate_measurement() must be called before take_reading().")
        bias = float(self._bias_V)  # type: ignore[arg-type]
        voltages: list[float] = []
        currents: list[float] = []
        resistances: list[float] = []
        for _ in range(self._readings_per_point):
            current = float(self._smu.measure_current())  # type: ignore[attr-defined]
            voltages.append(bias)
            currents.append(current)
            resistances.append(bias / current if current != 0.0 else math.nan)
        reading: dict[str, list[float] | float] = {}
        for name, samples in (
            ("voltage_V", voltages),
            ("current_A", currents),
            ("resistance_ohm", resistances),
        ):
            mean, error = self.mean_and_sem(samples)
            reading[f"{name}_array"] = samples
            reading[name] = mean
            reading[f"{name}_error"] = error
        return reading

    @control(action_class="run_control")
    def set_bias_voltage(self, bias_V: float) -> None:
        """Step the bias without re-arming (the reading loop's setter).

        Args:
            bias_V: The new bias, volts (may be negative).

        Raises:
            RuntimeError: If ``initiate_measurement()`` has not been called.
        """
        if self._bias_V is _NOT_INITIATED:
            raise RuntimeError("initiate_measurement() must be called before set_bias_voltage().")
        self._smu.set_source_voltage(float(bias_V))  # type: ignore[attr-defined]
        self._bias_V = float(bias_V)

    def standby(self) -> None:
        """Zero the bias, switch the output off and reset the armed state."""
        self._smu.set_source_voltage(0.0)  # type: ignore[attr-defined]
        self._smu.set_output(False)  # type: ignore[attr-defined]
        self._bias_V = _NOT_INITIATED
        self._last_reading = None

    # ------------------------------------------------------------------
    # Manual bench read
    # ------------------------------------------------------------------

    @control(panel=False, action_class="read")
    def read_now(self) -> None:
        """Take one reading at the armed bias and cache it for the monitored fields.

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

    @monitored(unit="A", description="Current from the last manual read")
    def last_current_A(self) -> float | None:
        """Return the mean current of the last ``read_now()``, or None."""
        if self._last_reading is None:
            return None
        return float(self._last_reading["current_A"])  # type: ignore[arg-type]
