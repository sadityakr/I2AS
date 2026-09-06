"""Simulated Lakeshore 335 temperature controller driver."""

from __future__ import annotations

import logging
import math
import time

from i2as.core.exceptions import (
    I2ASCommunicationError,
    I2ASInstrumentError,
)

log = logging.getLogger(__name__)

# Curve slots 1-20 are the instrument's built-in standard curves and 0 means
# "no curve"; 21-59 are USER slots, empty until somebody uploads a curve into
# them. Assigning an empty user slot is refused (execution error) — see
# set_sensor_curve().
_STANDARD_CURVE_SLOTS = frozenset(range(0, 21))


class SimLakeshore335:
    """Simulated Lakeshore 335 temperature controller.

    Matches the public API of the real Lakeshore335 driver.
    """

    def __init__(self, resource_string: str) -> None:
        """Initialise the simulated Lakeshore 335.

        Args:
            resource_string: VISA resource string (ignored).
        """
        _ = resource_string  # Explicitly ignored
        self._temperature: float = 300.0  # Kelvin (room-temperature start)
        self._setpoint: float = 0.0
        self._heater_output: float = 0.0
        self._heater_mode: str = "AUTO"
        self._heater_range: str = "OFF"  # matches the real instrument's power-up default
        self._proportional_band: float = 90.0
        self._integral_action_time: float = 50.0
        self._derivative_action_time: float = 0.0
        self._auto_pid: bool = False
        self._sensor_curves: dict[str, int] = {"A": 22, "B": 2}
        # Instrument-error model (the driver error-reporting standard,
        # drivers/README.md): which USER curve slots actually hold a curve.
        # 22 is loaded because input A is already assigned to it above.
        self._loaded_user_curves: set[int] = {22}
        
        # Simulation physics
        self._last_update: float = time.time()
        self._tau: float = 30.0  # thermal time constant in seconds
        self._simulate_error: bool = False
        # Connection-lifecycle standard: True once close() has released
        # the session; every command then fails (see _check_error).
        self._closed: bool = False

    def get_temperature(self) -> float:
        """Return the current simulated temperature in Kelvin."""
        self._check_error()
        self._update_simulation()
        return self._temperature

    def get_setpoint(self) -> float:
        """Return the simulated setpoint in Kelvin."""
        self._check_error()
        return self._setpoint

    def set_setpoint(self, setpoint: float) -> None:
        """Set the simulated temperature setpoint.

        Args:
            setpoint: Target temperature in Kelvin. Must be >= 0.
        """
        self._check_error()
        if setpoint < 0.0:
            raise ValueError(f"Setpoint must be >= 0 K, got {setpoint}")
        self._setpoint = setpoint

    def get_heater_output(self) -> float:
        """Return the simulated heater output percentage.

        Always 0% while the heater range is 'OFF' (the instrument's power-up
        default) — matching the real 335, where the commanded/computed
        output is preserved internally but no power reaches the heater until
        a non-'OFF' range is set.
        """
        self._check_error()
        self._update_simulation()
        if self._heater_range == "OFF":
            return 0.0
        return self._heater_output

    def set_heater_output(self, output: float) -> None:
        """Set the manual heater output percentage.

        Args:
            output: Percent of maximum power in [0.0, 99.9].
        """
        self._check_error()
        self._heater_output = max(0.0, min(99.9, output))

    def get_heater_mode(self) -> str:
        """Return the simulated heater control mode ('MANUAL' or 'AUTO')."""
        self._check_error()
        return self._heater_mode

    def set_heater_mode(self, mode: str) -> None:
        """Set the simulated heater control mode to 'MANUAL' or 'AUTO'.

        Args:
            mode: Must be 'MANUAL' or 'AUTO'.
        """
        self._check_error()
        if mode not in ("MANUAL", "AUTO"):
            raise ValueError(f"Heater mode must be 'MANUAL' or 'AUTO', got {mode}")
        self._heater_mode = mode

    def get_heater_range(self) -> str:
        """Return the simulated heater range ('OFF', 'LOW', 'MEDIUM', 'HIGH')."""
        self._check_error()
        return self._heater_range

    def set_heater_range(self, range_setting: str) -> None:
        """Set the simulated heater range, switching heater power on or off.

        Args:
            range_setting: One of 'OFF', 'LOW', 'MEDIUM', 'HIGH'.
        """
        self._check_error()
        if range_setting not in ("OFF", "LOW", "MEDIUM", "HIGH"):
            raise ValueError(
                f"Heater range must be one of 'OFF', 'LOW', 'MEDIUM', 'HIGH', "
                f"got {range_setting!r}"
            )
        self._heater_range = range_setting

    def get_proportional_band(self) -> float:
        """Return the proportional band."""
        self._check_error()
        return self._proportional_band

    def set_proportional_band(self, pb: float) -> None:
        """Set the proportional band."""
        self._check_error()
        self._proportional_band = max(0.0, min(1000.0, pb))

    def get_integral_action_time(self) -> float:
        """Return the integral action time."""
        self._check_error()
        return self._integral_action_time

    def set_integral_action_time(self, iat: float) -> None:
        """Set the integral action time."""
        self._check_error()
        self._integral_action_time = max(0.0, min(1000.0, iat))

    def get_derivative_action_time(self) -> float:
        """Return the derivative action time."""
        self._check_error()
        return self._derivative_action_time

    def set_derivative_action_time(self, dat: float) -> None:
        """Set the derivative action time."""
        self._check_error()
        self._derivative_action_time = max(0.0, min(200.0, dat))

    def get_auto_pid(self) -> bool:
        """Return whether Autotuning is active."""
        self._check_error()
        return self._auto_pid

    def set_auto_pid(self, enabled: bool) -> None:
        """Enable or disable Autotuning."""
        self._check_error()
        self._auto_pid = bool(enabled)

    def get_idn(self) -> str:
        """Return simulated identification string."""
        self._check_error()
        return "LSCI,MODEL335,SIM,1.0"

    def get_sensor_curve(self, sensor_input: str = "A") -> int:
        """Return the curve number assigned to the sensor input."""
        self._check_error()
        ch = str(sensor_input).upper()
        if ch not in ("A", "B"):
            raise ValueError(f"Sensor input must be 'A' or 'B', got {sensor_input}")
        return self._sensor_curves[ch]

    def set_sensor_curve(self, curve: int, sensor_input: str = "A") -> None:
        """Assign a temperature sensor curve to a sensor input.

        Models the instrument-error half of the driver error-reporting
        standard: assigning a USER curve slot (21-59) that holds no curve is
        an execution error — the instrument flags ``*ESR?`` bit 4 and leaves
        the input on the curve it already had. Nothing else on the bus
        changes, which is precisely why the real driver reads ``*ESR?`` after
        every state-changing write.

        Args:
            curve: Curve number (0 = None, 1-20 = Standard, 21-59 = User).
            sensor_input: Sensor input channel ('A' or 'B', default 'A').

        Raises:
            ValueError: If the input or curve number is outside the
                instrument's addressable range (a programming error, caught
                before anything reaches the bus).
            I2ASInstrumentError: ``ESR:0x10`` if the named user-curve slot
                holds no curve.
        """
        self._check_error()
        ch = str(sensor_input).upper()
        if ch not in ("A", "B"):
            raise ValueError(f"Sensor input must be 'A' or 'B', got {sensor_input}")
        if not (0 <= curve <= 59):
            raise ValueError(f"Curve number must be in [0, 59], got {curve}")
        if int(curve) not in _STANDARD_CURVE_SLOTS and int(curve) not in self._loaded_user_curves:
            # The assignment deliberately does NOT happen.
            context = f"set_sensor_curve({curve!r}, {sensor_input!r})"
            raise I2ASInstrumentError(
                f"Simulated Lakeshore 335 refused {context}: Execution error "
                f"(ESR:0x10) — user curve slot {int(curve)} holds no curve",
                code="ESR:0x10",
                instrument_message="Execution error",
                context=context,
                vi_name="SimLakeshore335",
            )
        self._sensor_curves[ch] = int(curve)

    # ------------------------------------------------------------------
    # Safe state (the safe-shutdown standard)
    # ------------------------------------------------------------------

    def safe_shutdown(self) -> None:
        """Take the simulated heater off; idempotent, never raises.

        Safe idle is heater range ``OFF`` with the manual output zeroed, so a
        later range change cannot resume heating at a leftover percentage.
        The setpoint is deliberately preserved — see the real driver.
        """
        log.info("SimLakeshore335: safe shutdown — heater range OFF, output 0 %%.")
        self._heater_range = "OFF"
        self._heater_output = 0.0

    def _is_in_safe_state(self) -> bool:
        """Return True when no power can reach the heater."""
        return self._heater_range == "OFF" and self._heater_output == 0.0

    # ------------------------------------------------------------------
    # Connection lifecycle (the connection-lifecycle standard)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the simulated bus session; the instrument is left untouched.

        Idempotent and never raises. Afterwards every command — including
        ``get_idn()`` — raises ``I2ASCommunicationError`` via
        :meth:`_check_error`, modelling a released session so a
        use-after-disconnect bug fails in a test instead of on hardware.
        A closed driver is never reopened in place: the Station builds a
        fresh instance when the operator reconnects.
        """
        self._closed = True

    # ------------------------------------------------------------------
    # Simulation & Internal helpers
    # ------------------------------------------------------------------

    def _check_error(self) -> None:
        if self._closed:
            raise I2ASCommunicationError(
                "SimLakeshore335: the session is closed — the driver was "
                "disconnected from I2AS",
                vi_name="SimLakeshore335",
            )
        if self._simulate_error:
            raise I2ASCommunicationError(
                "Simulated communication error on Lakeshore 335",
                vi_name="SimLakeshore335",
            )

    def _update_simulation(self) -> None:
        now = time.time()
        dt = now - self._last_update
        self._last_update = now
        if dt <= 0:
            return

        # _heater_output always tracks the commanded (MANUAL) or computed
        # (AUTO) value, independent of heater range — matching the real 335,
        # where MOUT/the PID's internal output survive a range change. What
        # the range gates is whether that value actually reaches the heater
        # (see get_heater_output() and the target computation below).
        if self._heater_mode == "AUTO":
            # In AUTO, heater output is simulated as proportional to difference
            error = self._setpoint - self._temperature
            self._heater_output = min(99.9, error * 10.0) if error > 0 else 0.0
        # MANUAL: self._heater_output already holds the value set_heater_output() commanded.

        if self._heater_range == "OFF":
            # Heater range 'OFF' (the instrument's power-up default) switches
            # power delivery off entirely, no matter the heater mode or
            # setpoint — matching the real 335's RANGE behaviour (see driver
            # docstring). This is the condition the reported bug hit.
            target = self._temperature
        elif self._heater_mode == "AUTO":
            target = self._setpoint
        else:
            # MANUAL: max power (99.9%) reaches 300 K, 0% sits at 4.2 K (base temp)
            target = 4.2 + (self._heater_output / 99.9) * 295.8

        self._temperature = (
            target + (self._temperature - target) * math.exp(-dt / self._tau)
        )
