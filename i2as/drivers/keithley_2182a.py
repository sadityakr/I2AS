"""Real Keithley 2182A nanovoltmeter driver."""

from __future__ import annotations

import logging

import pyvisa

from i2as.core.exceptions import (
    I2ASCommunicationError,
    I2ASInstrumentError,
)

log = logging.getLogger(__name__)


# Cap on :SYST:ERR? reads when draining the queue. The 622x queue holds at
# most a handful of entries; the cap only guarantees termination if an
# instrument answers nonsense forever.
_ERROR_QUEUE_DRAIN_LIMIT = 32


def _parse_scpi_error(reply: str) -> tuple[str, str] | None:
    """Split a SCPI ``:SYST:ERR?`` reply into (code, message), or None if clean.

    A clean queue answers ``0,"No error"``. Anything else is a refusal the
    instrument recorded and nobody would otherwise see (the driver
    error-reporting standard, ``drivers/README.md``).

    Args:
        reply: The stripped instrument reply, e.g. ``'-221,"Settings conflict"'``.

    Returns:
        ``(code, message)`` with the code kept verbatim as a string and the
        message unquoted, or ``None`` when the queue reports no error. An
        unparseable reply is reported as an error with an empty code rather
        than assumed clean — the standard never guesses in the instrument's
        favour.
    """
    text = reply.strip()
    if not text:
        return None
    code, _, message = text.partition(",")
    code = code.strip()
    message = message.strip().strip('"')
    try:
        if int(code) == 0:
            return None
    except ValueError:
        return ("", text)
    return (code, message)


class Keithley2182A:
    """Real Keithley 2182A nanovoltmeter.

    Exposes the same public API as SimKeithley2182A.

    Note: In Keithley delta-mode measurements the 2182A is triggered and read
    by the 6221 via the trigger bus and serial relay — this driver is used only
    for standalone DC voltage readings and range setup.

    Driver contract:
    1. It is a Python class.
    2. __init__ accepts a single VISA resource string.
    3. It is importable via i2as.drivers.keithley_2182a.
    """

    def __init__(self, resource_string: str) -> None:
        """Open the VISA resource.

        Args:
            resource_string: VISA address, e.g. ``'GPIB0::7::INSTR'``.

        Raises:
            I2ASCommunicationError: If the resource cannot be opened.
        """
        self._rm = pyvisa.ResourceManager()
        try:
            self._instr = self._rm.open_resource(resource_string)
        except pyvisa.VisaIOError as exc:
            raise I2ASCommunicationError(
                f"Cannot open Keithley 2182A at {resource_string}: {exc}",
                vi_name="Keithley2182A",
            ) from exc

        self._instr.timeout = 5_000
        self._instr.write_termination = "\n"
        self._instr.read_termination = "\n"

    # ------------------------------------------------------------------
    # Public API  (matches SimKeithley2182A)
    # ------------------------------------------------------------------

    def set_continuous_initiation(self, enabled: bool) -> None:
        """Turn the instrument's free-running (continuous initiation) mode on/off.

        Live commissioning (2026-07-22, GPIB0::28) found the instrument ships
        with continuous initiation on (``:INIT:CONT?`` -> 1, ``:TRIG:COUN?``
        -> ~9.9e37 — free-running). ``READ?``'s implicit ``:INIT`` is then
        always rejected as "-213 Init ignored" (harmless — ``:FETCh?`` still
        returns a fresh conversion since one is always in flight) but it fills
        the error queue on every ``get_voltage()`` call. Pinning single-shot
        mode makes each ``READ?`` do exactly what it looks like it does: one
        trigger, one fresh reading, idle in between, zero spurious errors.

        This is an instrument-state command, so under the connection-lifecycle
        standard it belongs to the arming path (a measurement VI's
        ``initiate_measurement()``), never to driver construction — building
        the Station must leave every instrument exactly as the operator left
        it.

        Args:
            enabled: True to leave the instrument free-running, False (what
                every I2AS reading path wants) for single-shot.

        Raises:
            I2ASInstrumentError: If the 2182A queues an error for the
                write (the driver error-reporting standard).
        """
        self._write(":INIT:CONT " + ("ON" if enabled else "OFF"))
        self._check_error_queue(f"set_continuous_initiation({enabled!r})")

    def get_voltage(self) -> float:
        """Trigger and return a single DC voltage reading in Volts.

        Issues READ? which initiates a new measurement and fetches the result.

        Returns:
            Voltage in Volts.
        """
        raw = self._query("READ?")
        # Response may contain multiple comma-separated values; take channel 1.
        vals = [v.strip() for v in raw.split(",") if v.strip()]
        try:
            return float(vals[0])
        except (IndexError, ValueError) as exc:
            # Empty or garbage response — surface as a communication error so
            # the stale-value handling upstream applies instead of a crash.
            raise I2ASCommunicationError(
                f"Keithley 2182A: unparseable READ? response: {raw!r}",
                vi_name="Keithley2182A",
            ) from exc

    def set_range(self, range_v: float) -> None:
        """Set the DC voltage measurement range.

        Args:
            range_v: Full-scale voltage range in Volts
                     (e.g. 0.01 for 10 mV, 0.1 for 100 mV).

        Raises:
            I2ASInstrumentError: If the 2182A refuses the range (e.g.
                ``-222 "Parameter data out of range"`` above the channel's
                full scale) — the driver error-reporting standard. Without
                this poll the instrument silently keeps its previous range
                and every later reading is taken on the wrong scale.
        """
        self._write(f":SENS:VOLT:CHAN1:RANG {range_v:.4e}")
        self._check_error_queue(f"set_range({range_v!r})")

    def get_range(self) -> float:
        """Return the current DC voltage range setting in Volts."""
        raw = self._query(":SENS:VOLT:CHAN1:RANG?")
        try:
            return float(raw)
        except ValueError as exc:
            raise I2ASCommunicationError(
                f"Keithley 2182A: unparseable range response: {raw!r}",
                vi_name="Keithley2182A",
            ) from exc

    def get_idn(self) -> str:
        """Return the instrument identification string."""
        return self._query("*IDN?").strip()

    # ------------------------------------------------------------------
    # Connection lifecycle (the connection-lifecycle standard)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the VISA session; the instrument is left exactly as it is.

        The driver half of the connection-lifecycle standard (see
        ``drivers/README.md``): returns the bus session, sends no
        instrument-state command, and never raises — a disconnect must
        always succeed. ``GTL`` hands the front panel back to the operator,
        which is the whole point of disconnecting. A closed driver is never
        reopened in place; the Station builds a fresh instance to reconnect.
        """
        try:
            self._instr.control_ren(pyvisa.constants.VI_GPIB_REN_ADDRESS_GTL)
        except Exception as exc:  # noqa: BLE001 — best effort, close must not raise
            log.debug("Keithley 2182A: could not return to local: %s", exc)
        try:
            self._instr.close()
        except Exception as exc:  # noqa: BLE001 — best effort, close must not raise
            log.debug("Keithley 2182A: error closing VISA session: %s", exc)

    # ------------------------------------------------------------------
    # Safe state (the safe-shutdown standard)
    # ------------------------------------------------------------------

    def safe_shutdown(self) -> None:
        """Unconditionally put the voltmeter in its safe idle state.

        The 2182A's half of the **safe-shutdown standard** (see
        ``drivers/README.md``): idempotent, callable from any leftover state,
        never raises.

        The 2182A sources nothing, so "safe" here means *quiet*: single-shot
        instead of free-running (``:INIT:CONT OFF``), so it is not
        continuously triggering conversions and filling its error queue with
        ``-213 "Init ignored"`` against the next client of the bus. The
        measurement range is deliberately left alone — it is a measurement
        setting, not a hazard, and clobbering it would surprise an operator
        who set it from the front panel.

        Recovers from: a free-running instrument left mid-acquisition, and a
        non-empty error queue.
        """
        log.info("Keithley 2182A: safe shutdown — returning to single-shot.")
        try:
            self._write(":INIT:CONT OFF")
        except I2ASCommunicationError as exc:
            log.warning("Keithley 2182A: safe shutdown could not send :INIT:CONT OFF: %s", exc)
        self._drain_error_queue()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_error_queue(self, context: str) -> None:
        """Poll ``:SYST:ERR?`` and raise if the last command was refused.

        The 2182A's half of the **driver error-reporting standard** (see
        ``drivers/README.md``). Like every SCPI instrument here, the 2182A
        accepts a write it cannot execute and records the reason in its own
        error queue instead of failing the bus transaction, so without this
        poll a refused range change is invisible and every later reading is
        taken on a scale nobody chose.

        A bus failure while reading the queue is swallowed: the checker must
        never turn a working call into a failure of its own.

        Args:
            context: Human-readable description of the command just sent.

        Raises:
            I2ASInstrumentError: If the error queue is non-empty.
        """
        try:
            reply = self._query(":SYST:ERR?").strip()
        except I2ASCommunicationError:
            return
        error = _parse_scpi_error(reply)
        if error is None:
            return
        code, message = error
        log.error("Keithley 2182A: SCPI error after %s: %s", context, reply)
        raise I2ASInstrumentError(
            f"Keithley 2182A refused {context}: {code},{message!r}",
            code=code,
            instrument_message=message,
            context=context,
            vi_name="Keithley2182A",
        )

    def _drain_error_queue(self) -> None:
        """Read the error queue empty, discarding whatever it holds. Never raises."""
        for _ in range(_ERROR_QUEUE_DRAIN_LIMIT):
            try:
                reply = self._query(":SYST:ERR?").strip()
            except I2ASCommunicationError:
                return
            if _parse_scpi_error(reply) is None:
                return
            log.debug("Keithley 2182A: discarded queued SCPI error %s", reply)

    def _write(self, cmd: str) -> None:
        try:
            self._instr.write(cmd)
        except pyvisa.VisaIOError as exc:
            raise I2ASCommunicationError(
                f"Keithley 2182A write failed ({cmd!r}): {exc}",
                vi_name="Keithley2182A",
            ) from exc

    def _query(self, cmd: str) -> str:
        try:
            return self._instr.query(cmd).strip()
        except pyvisa.VisaIOError as exc:
            raise I2ASCommunicationError(
                f"Keithley 2182A query failed ({cmd!r}): {exc}",
                vi_name="Keithley2182A",
            ) from exc
