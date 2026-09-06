"""Simulated Keithley 6221 AC/DC Current Source driver."""

import logging
from typing import TYPE_CHECKING

from i2as.core.exceptions import (
    I2ASCommunicationError,
    I2ASInstrumentError,
)

log = logging.getLogger(__name__)

# The 622x's fixed current ranges (A, full scale). Delta mode pins the source
# to the smallest range that fits its configured high current and leaves
# autorange OFF, which is the whole mechanism behind the -221 incident this
# sim models — see _refuse_out_of_range().
_CURRENT_RANGES = (2e-9, 20e-9, 200e-9, 2e-6, 20e-6, 200e-6, 2e-3, 20e-3, 100e-3)


def _range_for(current: float) -> float:
    """Return the smallest 622x current range that holds *current*.

    Args:
        current: Absolute source current in Amperes.

    Returns:
        The full-scale value of the range the instrument would select.
    """
    for full_scale in _CURRENT_RANGES:
        if current <= full_scale:
            return full_scale
    return _CURRENT_RANGES[-1]

if TYPE_CHECKING:
    from i2as.drivers.sim_keithley_2182a import SimKeithley2182A


class SimKeithley6221:
    """Simulated Keithley 6221 AC/DC current source.

    Supports source enable/disable, current configuration, and delta-mode
    operation (typically paired with a SimKeithley2182A nanovoltmeter).

    This driver satisfies the three-rule driver contract:
    1. It is a Python class.
    2. __init__ accepts a single VISA resource string (ignored for simulation).
    3. It is importable via i2as.drivers.sim_keithley_6221.
    """

    def __init__(self, resource_string: str) -> None:
        """Initialise the simulated Keithley 6221.

        Args:
            resource_string: VISA address (e.g. 'GPIB0::22::INSTR'). Ignored.
        """
        _ = resource_string  # Explicitly ignored per driver contract

        self._source_enabled: bool = False
        self._current: float = 0.0         # Amperes
        self._compliance: float = 0.1      # Volts (voltage compliance limit)

        # Source function mode — "DC" (fixed current) or "DELTA" (bipolar
        # delta engine armed/running). Mirrors the real 6221's SCPI function
        # mode so tests can catch a VI that assumes a starting mode instead of
        # asserting its own (the "shared-instrument mode discipline" standard
        # in virtual_instruments/measurement/README.md). Private test/model
        # hook, not part of the public API parity check.
        self._mode: str = "DC"

        # Instrument-error model (the driver error-reporting standard,
        # drivers/README.md). The real 6221 records a refused command in its
        # SCPI error queue and answers the bus normally; the sim skips the
        # queue and raises the same typed error the real driver raises after
        # reading it, so a wrong command sequence fails in a test instead of
        # silently on hardware.
        #
        # Autorange OFF with a pinned range is exactly what delta mode leaves
        # behind (live commissioning 2026-07-22): a later source current
        # outside that leftover range is refused with -221 "Settings
        # conflict" and the source silently keeps its old value.
        self._autorange: bool = True
        self._fixed_range_A: float = _CURRENT_RANGES[-1]
        # True between configure_and_start_delta() and stop_delta_mode(): the
        # delta engine owns the source, so source-setting commands that do
        # not abort it first are refused.
        self._delta_armed: bool = False

        # Delta-mode configuration
        self._delta_high_current: float = 0.0
        self._delta_n_readings: int = 1
        self._delta_delay: float = 0.01    # seconds

        # Set externally to link the 2182A for realistic delta simulation
        self._paired_meter: "SimKeithley2182A | None" = None

        # Stored delta readings after trigger
        self._delta_readings: list[float] = []

        # Test control flags
        self._simulate_error: bool = False
        # Connection-lifecycle standard: True once close() has released
        # the session; every command then fails (see _check_error).
        self._closed: bool = False
        # When set (int), acquire_delta_readings() returns only this many
        # samples instead of the full n_readings — models the real 6221's
        # short-return path (compliance abort / repeated read failures) so
        # tests can exercise the VI's NaN-padding + n_valid contract. Private,
        # so it is not part of the public API parity check.
        self._delta_return_count: int | None = None

        # Mirrors the real driver's :SOUR:DELT:NVPR? check — set False to
        # model a 2182A that is powered off, uncabled, or left in GPIB mode
        # on its own front panel (unreachable via the 6221's RS-232 relay).
        # Private test hook, not part of the public API parity check.
        self._meter_present: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_source_enabled(self) -> bool:
        """Return True if the current source output is enabled."""
        self._check_error()
        return self._source_enabled

    def set_source_enabled(self, enabled: bool) -> None:
        """Enable or disable the current source output.

        Models the instrument-error half of the driver error-reporting
        standard: while the delta engine is armed it owns the output, and an
        ``OUTP`` written by hand is a settings conflict. The real driver is
        immune by construction, which is why its ordering is what it is —
        :meth:`stop_delta_mode` sends ``:SOUR:SWE:ABOR`` *before* ``OUTP
        OFF``, never the other way round. A caller that skips the abort is
        the wrong sequence this refusal exists to catch.

        Args:
            enabled: True to turn the output on.

        Raises:
            I2ASInstrumentError: ``-221 Settings conflict`` if the delta
                engine is armed.
        """
        if self._delta_armed:
            self._refuse(
                f"set_source_enabled({enabled!r})",
                "the delta engine owns the output; abort it first",
            )
        self._source_enabled = bool(enabled)

    def get_current(self) -> float:
        """Return the configured source current in Amperes."""
        self._check_error()
        return self._current

    def set_current(self, current: float) -> None:
        """Set the source current, recovering from any leftover mode first.

        Mirrors the real driver command for command: abort whatever engine is
        running (``:SOUR:SWE:ABOR``) and reassert autorange
        (``:SOUR:CURR:RANG:AUTO ON``) BEFORE writing the value. That order is
        what makes the call self-recovering: it is immune to the fixed range
        and armed engine another measurement VI sharing this same driver may
        have left behind (the "shared-instrument mode discipline" standard in
        ``virtual_instruments/measurement/README.md``). Reverse the order and
        the sim reproduces the -221 incident exactly.

        Args:
            current: Desired current in Amperes.
        """
        self._delta_armed = False
        self._mode = "DC"
        self._autorange = True
        self._fixed_range_A = _CURRENT_RANGES[-1]
        self._apply_current(f"set_current({current!r})", float(current))

    def set_compliance(self, compliance_v: float) -> None:
        """Set the voltage compliance limit.

        Deliberately NOT refused while the delta engine is armed: the real
        driver's own delta sequence writes ``:SOUR:CURR:COMP`` as part of
        programming delta mode, so compliance is demonstrably writable in
        that state and modelling a refusal here would be fiction.

        Args:
            compliance_v: Maximum output voltage in Volts.
        """
        self._compliance = float(compliance_v)

    def get_compliance(self) -> float:
        """Return the configured voltage compliance limit in Volts."""
        self._check_error()
        return self._compliance

    def configure_delta_mode(
        self, high_current: float, n_readings: int, delay: float
    ) -> None:
        """Configure delta-mode measurement parameters.

        In the real instrument this programs the 6221 to alternate between
        +I and -I while triggering the 2182A. Here we just store the config.

        Args:
            high_current: Peak current magnitude for delta mode (A).
            n_readings: Number of reading pairs to acquire.
            delay: Delay between source transitions (seconds).
        """
        self._delta_high_current = float(high_current)
        self._delta_n_readings = int(n_readings)
        self._delta_delay = float(delay)
        self._delta_readings = []

    def trigger_delta_mode(self) -> None:
        """Start a delta-mode measurement sweep.

        Generates n_readings voltage samples. If a paired 2182A is attached
        its get_voltage() method is called; otherwise zeros with noise are used.
        """
        import random
        readings: list[float] = []
        for _ in range(self._delta_n_readings):
            if self._paired_meter is not None:
                readings.append(self._paired_meter.get_voltage())
            else:
                # Simulate a tiny noisy resistance signal
                readings.append(random.gauss(1.5e-6, 1e-8))
        self._delta_readings = readings

    def get_delta_readings(self) -> list[float]:
        """Return the voltage readings from the last delta-mode sweep.

        Returns:
            List of float voltage readings (length == n_readings configured).
        """
        self._check_error()
        return list(self._delta_readings)

    def get_idn(self) -> str:
        """Return simulated *IDN? response string."""
        self._check_error()
        return "KEITHLEY,6221,SIM,1.0"

    # ------------------------------------------------------------------
    # Split delta lifecycle  (mirrors Keithley6221 real driver)
    # ------------------------------------------------------------------

    def configure_and_start_delta(
        self,
        high_current: float,
        n_readings: int,
        delay: float,
        compliance: float = 1.0,
        range_2182a: float = 0.01,
        compliance_abort: bool = True,
        cold_switch: bool = False,
    ) -> None:
        """Configure delta-mode and 'arm' the simulated engine.

        Stores all parameters; on the sim there is no hardware to arm.
        Call acquire_delta_readings() to collect samples. The signature must
        mirror the real Keithley6221 driver exactly (conformance parity check).

        Args:
            high_current: Peak delta current magnitude (A).
            n_readings: Number of readings per acquisition call.
            delay: Delay between source transitions (s).
            compliance: Voltage compliance limit (V) — stored but unused in sim.
            range_2182a: 2182A range (V) — stored but unused in sim.
            compliance_abort: Delta compliance-abort flag — stored but unused in sim.
            cold_switch: Delta cold-switch flag — stored but unused in sim.

        Raises:
            I2ASCommunicationError: If ``_meter_present`` has been set
                False, mirroring the real driver's NVPR-absent failure.
        """
        if not self._meter_present:
            raise I2ASCommunicationError(
                "Simulated Keithley 6221: no 2182A detected on the RS-232 "
                "serial relay.",
                vi_name="SimKeithley6221",
            )
        self._mode = "DELTA"
        # Delta mode pins the current range to the smallest one that fits its
        # high current and leaves autorange OFF, with nothing to undo it
        # afterwards. This side effect is the -221 incident's root cause, so
        # the sim models it rather than the tidy behaviour one might assume.
        self._autorange = False
        self._fixed_range_A = _range_for(abs(float(high_current)))
        self._delta_armed = True
        self._delta_high_current = float(high_current)
        self._delta_n_readings = int(n_readings)
        self._delta_delay = float(delay)
        self._delta_compliance = float(compliance)
        self._delta_range_2182a = float(range_2182a)
        self._delta_compliance_abort = bool(compliance_abort)
        self._delta_cold_switch = bool(cold_switch)
        self._delta_readings = []

    def acquire_delta_readings(
        self, n_readings: int, period: float = 0.01
    ) -> list[float]:
        """Collect *n_readings* simulated delta-voltage samples.

        Generates readings using the paired meter or noise, matching the
        behaviour of trigger_delta_mode() but as a split call.

        Args:
            n_readings: Number of readings to generate.
            period: Ignored in simulation.

        Returns:
            List of float voltage readings.
        """
        import random

        count = n_readings
        if self._delta_return_count is not None:
            # Model an early-terminated acquisition (returns fewer samples).
            count = min(int(self._delta_return_count), n_readings)

        readings: list[float] = []
        for _ in range(count):
            if self._paired_meter is not None:
                readings.append(self._paired_meter.get_voltage())  # type: ignore[attr-defined]
            else:
                readings.append(random.gauss(1.5e-6, 1e-8))
        self._delta_readings = readings
        return list(self._delta_readings)

    def stop_delta_mode(self) -> None:
        """Abort the simulated delta engine and reset source to a plain idle state.

        Mirrors the real driver's ``:SOUR:SWE:ABOR`` + ``OUTP OFF``. Leaves
        autorange off and the range still pinned, exactly as the real
        instrument does — undoing delta's range side effect is
        :meth:`set_current`'s and :meth:`safe_shutdown`'s job, and pretending
        otherwise here would hide the very bug this sim exists to catch.
        """
        self._delta_armed = False
        self._mode = "DC"
        self._current = 0.0
        self._source_enabled = False
        self._delta_readings = []

    def is_in_compliance(self) -> bool:
        """Return True if the simulated current source is in compliance."""
        self._check_error()
        if self._paired_meter is not None:
            # Sourced current * 1500 Ohm load compared against compliance
            v_est = self._current * 1500.0
            return abs(v_est) >= self._compliance
        return False

    def get_voltage(self) -> float:
        """Read voltage from the paired simulated meter."""
        self._check_error()
        if self._paired_meter is not None:
            self._paired_meter._base_voltage = self._current * 1500.0
            return self._paired_meter.get_voltage()
        import random
        return random.gauss(1.5e-6, 1e-8)

    def set_range(self, range_v: float) -> None:
        """Set range on the paired simulated meter."""
        self._check_error()
        if self._paired_meter is not None:
            self._paired_meter.set_range(range_v)

    def get_range(self) -> float:
        """Get range from the paired simulated meter."""
        self._check_error()
        if self._paired_meter is not None:
            return self._paired_meter.get_range()
        return 0.01

    # ------------------------------------------------------------------
    # Instrument-error model (the driver error-reporting standard)
    # ------------------------------------------------------------------

    def _refuse(self, context: str, why: str) -> None:
        """Raise the -221 the real instrument would queue for *context*.

        Args:
            context: The driver call being refused, e.g. ``"set_compliance(1.0)"``.
            why: Plain-English reason, appended for the reader; the ``code``
                and ``instrument_message`` carried on the error stay exactly
                what the real 622x reports.

        Raises:
            I2ASInstrumentError: Always.
        """
        raise I2ASInstrumentError(
            f"Simulated Keithley 6221 refused {context}: "
            f'-221,"Settings conflict" ({why})',
            code="-221",
            instrument_message="Settings conflict",
            context=context,
            vi_name="SimKeithley6221",
        )

    def _apply_current(self, context: str, current: float) -> None:
        """Apply a source current, refusing it if a pinned range cannot hold it.

        Models the exact silent-rejection path found on real hardware
        (2026-07-22): with autorange OFF and the range pinned by a previous
        delta run, a larger requested current is refused with -221 and the
        source keeps its OLD value — no exception on the bus, nothing in the
        log, visible only on the front panel or by polling the error queue.

        Args:
            context: The driver call being applied.
            current: Requested current in Amperes.

        Raises:
            I2ASInstrumentError: ``-221`` if the pinned range is too small.
        """
        if not self._autorange and abs(current) > self._fixed_range_A:
            # The source value deliberately does NOT change — that silence is
            # the bug being modelled.
            self._refuse(
                context,
                f"{abs(current):.3e} A exceeds the fixed "
                f"{self._fixed_range_A:.3e} A range left with autorange off",
            )
        self._current = float(current)

    # ------------------------------------------------------------------
    # Safe state (the safe-shutdown standard)
    # ------------------------------------------------------------------

    def safe_shutdown(self) -> None:
        """Put the simulated source in its safe idle state; never raises.

        Mirrors the real driver's sequence: abort the engine, reassert
        autorange, zero the current, output off, error state cleared.

        Safe idle for this instrument is: DC mode, delta engine disarmed,
        autorange on with no pinned range, zero source current, output off —
        the state :meth:`_is_in_safe_state` checks for.
        """
        log.info("SimKeithley6221: safe shutdown — engine aborted, output off.")
        self._delta_armed = False
        self._mode = "DC"
        self._autorange = True
        self._fixed_range_A = _CURRENT_RANGES[-1]
        self._current = 0.0
        self._source_enabled = False
        self._delta_readings = []

    def _is_in_safe_state(self) -> bool:
        """Return True when the sim is in the safe idle state defined above.

        The sim half of the safe-shutdown standard (``drivers/README.md``):
        the conformance test calls :meth:`safe_shutdown` and asserts this.
        """
        return (
            not self._delta_armed
            and self._mode == "DC"
            and self._autorange
            and self._current == 0.0
            and not self._source_enabled
        )

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

    def _check_error(self) -> None:
        """Raise I2ASCommunicationError if error simulation is active."""
        if self._closed:
            raise I2ASCommunicationError(
                "SimKeithley6221: the session is closed — the driver was "
                "disconnected from I2AS",
                vi_name="SimKeithley6221",
            )
        if self._simulate_error:
            raise I2ASCommunicationError(
                "Simulated communication error on Keithley 6221",
                vi_name="SimKeithley6221",
            )
