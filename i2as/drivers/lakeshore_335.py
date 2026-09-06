"""Real Lakeshore 335 temperature controller driver (pure PyVISA)."""

from __future__ import annotations

import logging

import pyvisa

from i2as.core.exceptions import (
    I2ASCommunicationError,
    I2ASInstrumentError,
)

log = logging.getLogger(__name__)

# Standard Event Status Register bits (IEEE-488.2 ``*ESR?``, which the 335
# implements as its common interface). The 335 has no ``:SYST:ERR?`` queue,
# so this register IS its error reporting: a command it could not parse or
# could not carry out sets a bit here and nothing else on the bus changes.
# Reading the register clears it, which makes each check self-draining.
_ESR_QUERY_ERROR = 0x04
_ESR_EXECUTION_ERROR = 0x10
_ESR_COMMAND_ERROR = 0x20
_ESR_FAULT_BITS = _ESR_QUERY_ERROR | _ESR_EXECUTION_ERROR | _ESR_COMMAND_ERROR
_ESR_BIT_MEANINGS = (
    (_ESR_COMMAND_ERROR, "Command error"),
    (_ESR_EXECUTION_ERROR, "Execution error"),
    (_ESR_QUERY_ERROR, "Query error"),
)


class Lakeshore335:
    """Real Lakeshore 335 temperature controller.

    Reads temperature from input channel A and controls heater output 1.
    Exposes the temperature-controller driver API (excluding needle-valve
    methods, which are VTI-only), so SampleTemperatureControllerVI works
    with this driver without modification.

    Driver contract:
    1. It is a Python class.
    2. __init__ accepts a single VISA resource string.
    3. It is importable via i2as.drivers.lakeshore_335.
    """

    def __init__(self, resource_string: str) -> None:
        """Open the VISA resource and configure timeouts.

        Args:
            resource_string: VISA address, e.g. ``'GPIB0::12::INSTR'``.

        Raises:
            I2ASCommunicationError: If the resource cannot be opened.
        """
        self._rm = pyvisa.ResourceManager()
        try:
            self._instr = self._rm.open_resource(resource_string)
        except pyvisa.VisaIOError as exc:
            raise I2ASCommunicationError(
                f"Cannot open Lakeshore 335 at {resource_string}: {exc}",
                vi_name="Lakeshore335",
            ) from exc

        self._instr.timeout = 5_000
        self._instr.write_termination = "\n"
        self._instr.read_termination = "\n"

    # ------------------------------------------------------------------
    # Public API  (the subset SampleTemperatureControllerVI uses)
    # ------------------------------------------------------------------

    def get_temperature(self) -> float:
        """Return the current temperature from input channel A in Kelvin.

        Returns:
            Temperature in Kelvin.
        """
        raw = self._query("KRDG? A")
        try:
            return float(raw)
        except ValueError as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335: cannot parse temperature from {raw!r}: {exc}",
                vi_name="Lakeshore335",
            ) from exc

    def get_setpoint(self) -> float:
        """Return the temperature setpoint for output 1 in Kelvin.

        Returns:
            Setpoint in Kelvin.
        """
        raw = self._query("SETP? 1")
        try:
            return float(raw)
        except ValueError as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335: cannot parse setpoint from {raw!r}: {exc}",
                vi_name="Lakeshore335",
            ) from exc

    def set_setpoint(self, setpoint: float) -> None:
        """Set the temperature setpoint for output 1.

        Args:
            setpoint: Target temperature in Kelvin. Must be >= 0.

        Raises:
            ValueError: If setpoint is negative.
        """
        if setpoint < 0.0:
            raise ValueError(f"Setpoint must be >= 0 K, got {setpoint}")
        self._write(f"SETP 1,{setpoint:.4f}")
        import time
        time.sleep(0.05)  # Allow instrument to process setpoint update
        self._check_event_status(f"set_setpoint({setpoint!r})")

    def get_heater_output(self) -> float:
        """Return the heater output for output 1 as a percentage (0–100 %).

        Returns:
            Heater output percent.
        """
        raw = self._query("HTR? 1")
        try:
            return float(raw)
        except ValueError as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335: cannot parse heater output from {raw!r}: {exc}",
                vi_name="Lakeshore335",
            ) from exc

    def get_idn(self) -> str:
        """Return the instrument identification string."""
        return self._query("*IDN?").strip()

    def close(self) -> None:
        """Release the VISA session; the instrument is left exactly as it is.

        The driver half of the connection-lifecycle standard (see
        ``drivers/README.md``): returns the bus session, sends no
        instrument-state command — the heater keeps whatever range, mode and
        setpoint the operator left it on, because disconnecting is not a
        safe-off action (that is the VI's ``standby()``) — and never raises.
        ``GTL`` hands the front panel back to the operator, which is the whole
        point of disconnecting.
        """
        try:
            self._instr.control_ren(pyvisa.constants.VI_GPIB_REN_ADDRESS_GTL)
        except Exception as exc:  # noqa: BLE001 — best effort, close must not raise
            log.debug("Lakeshore 335: could not return to local: %s", exc)
        try:
            self._instr.close()
        except Exception as exc:  # noqa: BLE001 — best effort, close must not raise
            log.debug("Lakeshore 335: error closing VISA session: %s", exc)

    def set_heater_output(self, output: float) -> None:
        """Set the manual heater output percentage.

        Args:
            output: Percent of maximum power in [0.0, 99.9].
        """
        clamped = max(0.0, min(99.9, output))
        self._write(f"MOUT 1,{clamped:.2f}")
        self._check_event_status(f"set_heater_output({output!r})")

    def get_heater_mode(self) -> str:
        """Return the heater control mode ('MANUAL' or 'AUTO')."""
        raw = self._query("OUTMODE? 1")
        try:
            mode = int(raw.split(",")[0])
            if mode == 3:
                return "MANUAL"
            elif mode == 1:
                return "AUTO"
            else:
                return f"OTHER({mode})"
        except (ValueError, IndexError) as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335: cannot parse OUTMODE from {raw!r}: {exc}",
                vi_name="Lakeshore335",
            ) from exc

    def set_heater_mode(self, mode: str) -> None:
        """Set the heater control mode to 'MANUAL' or 'AUTO'.

        Args:
            mode: Must be 'MANUAL' or 'AUTO'.
        """
        if mode not in ("MANUAL", "AUTO"):
            raise ValueError(f"Heater mode must be 'MANUAL' or 'AUTO', got {mode}")

        # Avoid redundant writes that interrupt control loops
        try:
            if self.get_heater_mode() == mode:
                return
        except Exception:
            pass

        raw = self._query("OUTMODE? 1")
        try:
            parts = raw.split(",")
            input_ch = int(parts[1])
            powerup = int(parts[2])
        except (ValueError, IndexError) as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335: cannot parse OUTMODE from {raw!r}: {exc}",
                vi_name="Lakeshore335",
            ) from exc

        target_mode = 3 if mode == "MANUAL" else 1
        self._write(f"OUTMODE 1,{target_mode},{input_ch},{powerup}")
        import time
        time.sleep(0.2)  # Allow control loop to reinitialize and settle
        self._check_event_status(f"set_heater_mode({mode!r})")

    def get_heater_range(self) -> str:
        """Return the heater range for output 1 ('OFF', 'LOW', 'MEDIUM', or 'HIGH').

        The heater range is what actually switches the output on: even in
        Closed Loop PID (auto) mode with a valid setpoint, the heater
        delivers no power while its range is 'OFF' (the instrument's
        power-up default). Distinct from ``get_heater_mode()``, which only
        selects auto vs. manual control of the setpoint/output value.
        """
        raw = self._query("RANGE? 1")
        try:
            n = int(raw)
        except ValueError as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335: cannot parse RANGE from {raw!r}: {exc}",
                vi_name="Lakeshore335",
            ) from exc
        mapping = {0: "OFF", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}
        if n not in mapping:
            raise I2ASCommunicationError(
                f"Lakeshore 335: unexpected RANGE value {n}",
                vi_name="Lakeshore335",
            )
        return mapping[n]

    def set_heater_range(self, range_setting: str) -> None:
        """Set the heater range for output 1, switching heater power on or off.

        Args:
            range_setting: One of 'OFF', 'LOW', 'MEDIUM', 'HIGH'.

        Raises:
            ValueError: If ``range_setting`` is not one of the four values.
        """
        mapping = {"OFF": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        if range_setting not in mapping:
            raise ValueError(
                f"Heater range must be one of {sorted(mapping)}, got {range_setting!r}"
            )
        self._write(f"RANGE 1,{mapping[range_setting]}")
        self._check_event_status(f"set_heater_range({range_setting!r})")

    def get_proportional_band(self) -> float:
        """Return the proportional band (P value) for output 1."""
        raw = self._query("PID? 1")
        try:
            return float(raw.split(",")[0])
        except (ValueError, IndexError) as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335: cannot parse PID from {raw!r}: {exc}",
                vi_name="Lakeshore335",
            ) from exc

    def set_proportional_band(self, pb: float) -> None:
        """Set the proportional band (P value) for output 1.

        P is clamped to [0.0, 1000.0].
        """
        pb_clamped = max(0.0, min(1000.0, pb))
        raw = self._query("PID? 1")
        try:
            parts = raw.split(",")
            i_val = float(parts[1])
            d_val = float(parts[2])
        except (ValueError, IndexError) as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335: cannot parse PID from {raw!r}: {exc}",
                vi_name="Lakeshore335",
            ) from exc
        self._write(f"PID 1,{pb_clamped:.1f},{i_val:.1f},{d_val:.1f}")
        self._check_event_status(f"set_proportional_band({pb!r})")

    def get_integral_action_time(self) -> float:
        """Return the integral action time (I value) for output 1."""
        raw = self._query("PID? 1")
        try:
            return float(raw.split(",")[1])
        except (ValueError, IndexError) as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335: cannot parse PID from {raw!r}: {exc}",
                vi_name="Lakeshore335",
            ) from exc

    def set_integral_action_time(self, iat: float) -> None:
        """Set the integral action time (I value) for output 1.

        I is clamped to [0.0, 1000.0].
        """
        iat_clamped = max(0.0, min(1000.0, iat))
        raw = self._query("PID? 1")
        try:
            parts = raw.split(",")
            p_val = float(parts[0])
            d_val = float(parts[2])
        except (ValueError, IndexError) as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335: cannot parse PID from {raw!r}: {exc}",
                vi_name="Lakeshore335",
            ) from exc
        self._write(f"PID 1,{p_val:.1f},{iat_clamped:.1f},{d_val:.1f}")
        self._check_event_status(f"set_integral_action_time({iat!r})")

    def get_derivative_action_time(self) -> float:
        """Return the derivative action time (D value) for output 1."""
        raw = self._query("PID? 1")
        try:
            return float(raw.split(",")[2])
        except (ValueError, IndexError) as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335: cannot parse PID from {raw!r}: {exc}",
                vi_name="Lakeshore335",
            ) from exc

    def set_derivative_action_time(self, dat: float) -> None:
        """Set the derivative action time (D value) for output 1.

        D is clamped to [0.0, 200.0].
        """
        dat_clamped = max(0.0, min(200.0, dat))
        raw = self._query("PID? 1")
        try:
            parts = raw.split(",")
            p_val = float(parts[0])
            i_val = float(parts[1])
        except (ValueError, IndexError) as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335: cannot parse PID from {raw!r}: {exc}",
                vi_name="Lakeshore335",
            ) from exc
        self._write(f"PID 1,{p_val:.1f},{i_val:.1f},{dat_clamped:.1f}")
        self._check_event_status(f"set_derivative_action_time({dat!r})")

    def get_auto_pid(self) -> bool:
        """Return whether Autotuning is active on output 1."""
        raw = self._query("TUNEST? 1")
        try:
            active = int(raw.split(",")[0])
            return active == 1
        except (ValueError, IndexError) as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335: cannot parse TUNEST from {raw!r}: {exc}",
                vi_name="Lakeshore335",
            ) from exc

    def set_auto_pid(self, enabled: bool) -> None:
        """Enable or disable Autotuning on output 1."""
        if enabled:
            self._write("ATUNE 1,2")
        else:
            raw = self._query("OUTMODE? 1")
            self._write(f"OUTMODE 1,{raw}")
        self._check_event_status(f"set_auto_pid({enabled!r})")

    def get_sensor_curve(self, sensor_input: str = "A") -> int:
        """Return the curve number assigned to the sensor input.

        Args:
            sensor_input: Sensor input channel ('A' or 'B', default 'A').

        Returns:
            The assigned curve number.
        """
        ch = str(sensor_input).upper()
        if ch not in ("A", "B"):
            raise ValueError(f"Sensor input must be 'A' or 'B', got {sensor_input}")
        raw = self._query(f"INCRV? {ch}")
        try:
            return int(raw)
        except ValueError as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335: cannot parse curve from {raw!r}: {exc}",
                vi_name="Lakeshore335",
            ) from exc

    def set_sensor_curve(self, curve: int, sensor_input: str = "A") -> None:
        """Assign a temperature sensor curve to a sensor input.

        Args:
            curve: Curve number (0 = None, 1-20 = Standard, 21-59 = User).
            sensor_input: Sensor input channel ('A' or 'B', default 'A').
        """
        ch = str(sensor_input).upper()
        if ch not in ("A", "B"):
            raise ValueError(f"Sensor input must be 'A' or 'B', got {sensor_input}")
        if not (0 <= curve <= 59):
            raise ValueError(f"Curve number must be in [0, 59], got {curve}")
        self._write(f"INCRV {ch},{curve}")
        self._check_event_status(f"set_sensor_curve({curve!r}, {sensor_input!r})")

    # ------------------------------------------------------------------
    # Safe state (the safe-shutdown standard)
    # ------------------------------------------------------------------

    def safe_shutdown(self) -> None:
        """Unconditionally take the heater off; never raises.

        The 335's half of the **safe-shutdown standard** (see
        ``drivers/README.md``): idempotent, callable from any leftover state.

        Safe idle for this instrument is *no power to the heater*. The heater
        RANGE is what actually gates the output — even in closed-loop mode
        with a live setpoint, range ``OFF`` delivers nothing — so the range
        goes to ``OFF`` first, and the manual output value is zeroed behind
        it so a later range change cannot resume heating at whatever
        percentage was left commanded. The SETPOINT is deliberately left
        alone: it is the operator's stated intent, it heats nothing while the
        range is off, and clobbering it would lose information without
        making anything safer.

        Recovers from: an energised heater in either control mode, and a
        non-zero manual output left commanded.
        """
        log.info("Lakeshore 335: safe shutdown — heater range OFF, output 0 %%.")
        for cmd in ("RANGE 1,0", "MOUT 1,0.00"):
            try:
                self._write(cmd)
            except I2ASCommunicationError as exc:
                log.warning("Lakeshore 335: safe shutdown could not send %r: %s", cmd, exc)
        # Clear whatever the abandoned sequence flagged so it is not charged
        # to the next command (reading *ESR? clears it).
        try:
            self._query("*ESR?")
        except I2ASCommunicationError:
            pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_event_status(self, context: str) -> None:
        """Read ``*ESR?`` and raise if the 335 flagged the last command.

        The 335's half of the **driver error-reporting standard** (see
        ``drivers/README.md``). This instrument has no SCPI error queue, so
        its verification is the **status byte** the standard names as the
        alternative: the IEEE-488.2 Standard Event Status Register, where a
        command the instrument could not parse (command error) or could not
        carry out (execution error) sets a bit and nothing else on the bus
        changes. Reading the register clears it, so each check leaves a clean
        slate for the next command.

        A bus failure while reading the register is swallowed: the checker
        must never turn a working call into a failure of its own.

        Args:
            context: Human-readable description of the command just sent.

        Raises:
            I2ASInstrumentError: If a command/execution/query-error bit is
                set, with ``code`` naming the raw register value.
        """
        try:
            raw = self._query("*ESR?").strip()
        except I2ASCommunicationError:
            return
        try:
            bits = int(raw)
        except ValueError:
            log.warning("Lakeshore 335: unparseable *ESR? reply %r after %s", raw, context)
            return
        flagged = bits & _ESR_FAULT_BITS
        if not flagged:
            return
        names = ", ".join(name for bit, name in _ESR_BIT_MEANINGS if flagged & bit)
        code = f"ESR:0x{flagged:02X}"
        log.error("Lakeshore 335: %s after %s (*ESR? = %s)", names, context, raw)
        raise I2ASInstrumentError(
            f"Lakeshore 335 refused {context}: {names} ({code})",
            code=code,
            instrument_message=names,
            context=context,
            vi_name="Lakeshore335",
        )

    def _write(self, cmd: str) -> None:
        try:
            self._instr.write(cmd)
        except pyvisa.VisaIOError as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335 write failed ({cmd!r}): {exc}",
                vi_name="Lakeshore335",
            ) from exc

    def _query(self, cmd: str) -> str:
        try:
            return self._instr.query(cmd).strip()
        except pyvisa.VisaIOError as exc:
            raise I2ASCommunicationError(
                f"Lakeshore 335 query failed ({cmd!r}): {exc}",
                vi_name="Lakeshore335",
            ) from exc
