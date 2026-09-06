# ---
# description: |
#   DCSeparateMeasurementVI: behavior-based VI for DC resistance measurements
#   using a dedicated current source and a separate voltmeter.
#   initiate_measurement() arms the source with a fixed current, compliance, voltmeter
#   range and readings-per-point. take_reading() collects that many voltage
#   samples at the fixed current. Declares reading_setters
#   {"current_A": "set_source_current"} so the generic sweep procedure's
#   reading loop can measure a user-entered list of currents (e.g. +/- pairs
#   for thermal-offset cancellation) at every sweep point.
# entry_point: Not run directly; instantiated by Station factory.
# dependencies:
#   - i2as.virtual_instruments.base (DCMeasurementBase)
#   - i2as.core.decorators (control)
# input: |
#   drivers = {"source": <current source driver>, "meter": <voltmeter driver>}
#   initiate_measurement(current_A, compliance_A, voltmeter_range_V, readings_per_point)
#   must be called before the argument-less take_reading().
# process: |
#   initiate_measurement() stores measurement parameters and programs both instruments.
#   take_reading() acquires readings_per_point voltage samples and returns them
#   alongside a constant current array. set_source_current() reprograms only
#   the source current between readings (the reading loop's setter command).
# output: |
#   Mean/error/array triple per quantity: {"voltage_V": float, "voltage_V_error":
#   float, "voltage_V_array": list[float], "current_A": float,
#   "current_A_error": float, "current_A_array": list[float]}, arrays of
#   length readings_per_point.
# last_updated: 2026-07-17
# ---

"""DCSeparateMeasurementVI — DC measurement with separate current source + voltmeter."""

from __future__ import annotations

from typing import Any, ClassVar

from i2as.core.decorators import control, monitored
from i2as.virtual_instruments.base import (
    EXCITATION_CURRENT_LIMIT,
    DCMeasurementBase,
)

_NOT_INITIATED = object()


class DCSeparateMeasurementVI(DCMeasurementBase):
    """Virtual Instrument for DC resistance measurements with separate instruments.

    Uses two drivers:
    * ``"source"`` — current source (e.g. Keithley 6221).
    * ``"meter"``  — voltmeter (e.g. Keithley 2182A nanovoltmeter).

    Workflow::

        vi.initiate_measurement(current_A=1e-6, compliance_A=1e-3, voltmeter_range_V=0.1,
                    readings_per_point=50)
        data = vi.take_reading()
        # data = {"voltage_V": float, "voltage_V_error": float,
        #         "voltage_V_array": list[float](50,), "current_A": float,
        #         "current_A_error": float, "current_A_array": list[float](50,)}

    Bench-testing from the GUI front panel uses ``read_now()`` instead:
    after ``initiate_measurement()`` has armed the instruments, one manual
    read collects the same ``readings_per_point`` samples and surfaces them
    through the ``last_voltage_V`` / ``last_n_valid`` monitored fields, so an
    operator (or an agent holding nothing but read authority) can confirm a
    configured current yields sane readings before committing to a run.

    To swap to a single-instrument SMU, name that SMU's VI in the YAML
    config instead of this one. The procedure is unchanged.

    Driver contract
    ---------------
    ``"source"`` driver must implement:
    * ``set_current(float)``
    * ``set_compliance(float)``
    * ``get_idn() -> str``

    ``"meter"`` driver must implement:
    * ``get_voltage() -> float``
    * ``set_range(float)``
    * ``get_idn() -> str``
    """

    # Short drop-down name: a separate current source + nanovoltmeter, as
    # opposed to a single SMU doing both.
    selector_label: ClassVar[str] = "DC (6221 + 2182A)"

    # Reading-loop declaration: the source current can be reprogrammed between
    # readings without re-arming, so the generic sweep procedure lets the user
    # loop a list of currents (e.g. "1e-6, -1e-6" for +/- offset cancellation)
    # at every sweep point.
    reading_setters: ClassVar[dict[str, str]] = {"current_A": "set_source_current"}

    # Control-validation standard (see BaseVirtualInstrument): MERGED with the
    # base's excitation bound rather than replacing it, so the per-reading
    # setter is guarded by the same setup ceiling as arming is.
    control_limits: ClassVar[dict[str, dict[str, str]]] = {
        **DCMeasurementBase.control_limits,
        "set_source_current": {"current_A": EXCITATION_CURRENT_LIMIT},
    }

    def __init__(self, drivers: dict[str, object], **init_params: Any) -> None:
        super().__init__(drivers, **init_params)
        self._source = drivers["source"]
        self._meter = drivers["meter"]

        self._current_A: object = _NOT_INITIATED
        self._compliance_A: float = 1e-3
        self._voltmeter_range_V: float = 0.1
        self._readings_per_point: int = 10

        # Cache of the last read_now() datapoint, read by the two monitored
        # fields below. None until the first manual read.
        self._last_reading: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # DCMeasurementBase implementation
    # ------------------------------------------------------------------

    # panel=False: arming is a deliberate act — reachable from the front
    # panel and from procedures, never from the compact monitor card.
    @control(panel=False, action_class="run_control")
    def initiate_measurement(
        self,
        current_A: float = 1e-6,
        compliance_A: float = 1e-3,
        voltmeter_range_V: float = 0.1,
        readings_per_point: int = 10,
    ) -> None:
        """Arm both instruments and configure measurement parameters.

        Args:
            current_A: DC source current in Amperes.
            compliance_A: Current compliance in Amperes.
            voltmeter_range_V: Full-scale voltage range in Volts.
            readings_per_point: Number of voltage samples ``take_reading()``
                collects per datapoint.
        """
        self._current_A = float(current_A)
        self._compliance_A = float(compliance_A)
        self._voltmeter_range_V = float(voltmeter_range_V)
        self._readings_per_point = int(readings_per_point)

        source = self._source  # type: ignore[attr-defined]
        meter = self._meter    # type: ignore[attr-defined]
        source.set_compliance(self._compliance_A)
        source.set_current(self._current_A)
        # Pin the voltmeter single-shot so each READ? is one fresh triggered
        # conversion. The 2182A ships free-running and the driver used to
        # force this from its own __init__; under the connection-lifecycle
        # standard that is a setup command, so it belongs on the arming path.
        meter.set_continuous_initiation(False)
        meter.set_range(self._voltmeter_range_V)

    def take_reading(self) -> dict[str, list[float] | float]:
        """Acquire ``readings_per_point`` DC voltage samples at the fixed current.

        Returns:
            The mean/error/array triple for both quantities (``voltage_V``,
            ``voltage_V_error``, ``voltage_V_array``, ``current_A``,
            ``current_A_error``, ``current_A_array``), arrays of length
            ``readings_per_point`` (fixed at ``initiate_measurement()``).

        Raises:
            RuntimeError: If ``initiate_measurement()`` has not been called first.
        """
        if self._current_A is _NOT_INITIATED:
            raise RuntimeError("initiate_measurement() must be called before take_reading().")

        current = float(self._current_A)
        meter = self._meter  # type: ignore[attr-defined]

        voltages: list[float] = []
        currents: list[float] = []
        for _ in range(self._readings_per_point):
            voltages.append(float(meter.get_voltage()))
            currents.append(current)

        v_mean, v_error = self.mean_and_sem(voltages)
        c_mean, c_error = self.mean_and_sem(currents)
        return {
            "voltage_V_array": voltages,
            "voltage_V": v_mean,
            "voltage_V_error": v_error,
            "current_A_array": currents,
            "current_A": c_mean,
            "current_A_error": c_error,
        }

    # ------------------------------------------------------------------
    # Manual bench read — the one read-class capability of this VI
    # ------------------------------------------------------------------

    @monitored(
        unit="V",
        description="Most recent voltage sample from the last manual read",
    )
    def last_voltage_V(self) -> float | None:
        """Return the last sample of the last ``read_now()``, or None.

        Returns:
            The final voltage sample in Volts, or ``None`` before the first
            manual read.
        """
        if self._last_reading is None:
            return None
        samples = self._last_reading["voltage_V_array"]
        return float(samples[-1]) if samples else None

    # Dimensionless: a sample count, not a measured quantity.
    @monitored(
        unit="",
        description="Number of samples returned by the last manual read",
    )
    def last_n_valid(self) -> int | None:
        """Return how many samples the last ``read_now()`` collected, or None.

        Returns:
            The sample count, or ``None`` before the first manual read.
        """
        if self._last_reading is None:
            return None
        return len(self._last_reading["voltage_V_array"])

    # panel=False: a bench check belongs in the instrument front panel, not
    # on the compact monitor card. action_class="read": it observes the
    # sample at the excitation already armed and commands nothing new, which
    # is what makes it the read-class capability an observer may take.
    @control(panel=False, action_class="read")
    def read_now(self) -> None:
        """Take one manual reading and cache it for the monitored fields.

        The human- (and observer-) facing counterpart of ``take_reading()``,
        which stays procedure-only per the measurement-method standard:
        this collects one datapoint at the current already armed and stores
        it, so ``last_voltage_V`` / ``last_n_valid`` report it on the next
        monitor tick.

        Raises:
            RuntimeError: If ``initiate_measurement()`` has not been called
                first.
        """
        self._last_reading = self.take_reading()

    @control(action_class="run_control")
    def set_source_current(self, current_A: float) -> None:
        """Reprogram the source current without re-arming the measurement.

        The per-reading setter behind ``reading_setters["current_A"]``: keeps
        compliance, voltmeter range and readings-per-point as armed and changes
        only the source current (sign included). Subsequent ``take_reading()``
        calls report the new current in ``current_A``.

        Args:
            current_A: New DC source current in Amperes (may be negative).

        Raises:
            RuntimeError: If ``initiate_measurement()`` has not been called first.
        """
        if self._current_A is _NOT_INITIATED:
            raise RuntimeError("initiate_measurement() must be called before set_source_current().")
        self._current_A = float(current_A)
        self._source.set_current(self._current_A)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def standby(self) -> None:
        """Zero the current source and reset the initiated state."""
        self._source.set_current(0.0)  # type: ignore[attr-defined]
        self._current_A = _NOT_INITIATED
        self._last_reading = None
