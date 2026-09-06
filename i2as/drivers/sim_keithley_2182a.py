"""Simulated Keithley 2182A Nanovoltmeter driver."""

import logging
import random

from i2as.core.exceptions import (
    I2ASCommunicationError,
    I2ASInstrumentError,
)

log = logging.getLogger(__name__)

# Full scale of the 2182A's largest channel-1 DC voltage range, plus the
# instrument's over-range headroom. A requested range above this is refused
# outright — see set_range().
_MAX_RANGE_V = 120.0


class SimKeithley2182A:
    """Simulated Keithley 2182A nanovoltmeter.

    Returns voltage readings with configurable Gaussian noise.

    This driver satisfies the three-rule driver contract:
    1. It is a Python class.
    2. __init__ accepts a single VISA resource string (ignored for simulation).
    3. It is importable via i2as.drivers.sim_keithley_2182a.
    """

    def __init__(self, resource_string: str) -> None:
        """Initialise the simulated Keithley 2182A.

        Args:
            resource_string: VISA address (e.g. 'GPIB0::7::INSTR'). Ignored.
        """
        _ = resource_string  # Explicitly ignored per driver contract

        self._base_voltage: float = 1.5e-6  # Volts — simulated signal level
        self._noise_std: float = 1e-8       # Gaussian noise standard deviation
        self._range: float = 0.1            # Volts measurement range
        # Free-running (continuous initiation) is the instrument's power-up
        # default; I2AS's reading paths pin it off when they arm.
        self._continuous_initiation: bool = True

        # Test control flags
        self._simulate_error: bool = False
        # Connection-lifecycle standard: True once close() has released
        # the session; every command then fails (see _check_error).
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_voltage(self) -> float:
        """Return a single voltage reading with Gaussian noise.

        Returns:
            Voltage in Volts (float).
        """
        self._check_error()
        return self._base_voltage + random.gauss(0.0, self._noise_std)

    def set_range(self, range_v: float) -> None:
        """Set the measurement voltage range.

        Models the instrument-error half of the driver error-reporting
        standard: a range above the largest the channel has is refused with
        ``-222 "Parameter data out of range"`` and the instrument keeps its
        previous range — silently, on real hardware, which is exactly why the
        real driver polls its error queue after this write.

        Args:
            range_v: Full-scale voltage range in Volts.

        Raises:
            I2ASInstrumentError: ``-222`` if *range_v* exceeds the
                channel's largest range.
        """
        self._check_error()
        if abs(float(range_v)) > _MAX_RANGE_V:
            # The stored range deliberately does NOT change.
            context = f"set_range({range_v!r})"
            raise I2ASInstrumentError(
                f"Simulated Keithley 2182A refused {context}: "
                f'-222,"Parameter data out of range" '
                f"({range_v!r} V is above the {_MAX_RANGE_V} V channel maximum)",
                code="-222",
                instrument_message="Parameter data out of range",
                context=context,
                vi_name="SimKeithley2182A",
            )
        self._range = float(range_v)

    def get_range(self) -> float:
        """Return the current voltage range setting in Volts."""
        self._check_error()
        return self._range

    def set_continuous_initiation(self, enabled: bool) -> None:
        """Turn the simulated free-running (continuous initiation) mode on/off.

        Mirrors the real driver's ``:INIT:CONT`` write. The flag is recorded
        so a test can assert the arming path pinned single-shot mode — and
        that nothing did so at Station build time.

        Args:
            enabled: True to leave the instrument free-running, False for
                single-shot.
        """
        self._check_error()
        self._continuous_initiation = bool(enabled)

    def get_idn(self) -> str:
        """Return simulated *IDN? response string."""
        self._check_error()
        return "KEITHLEY,2182A,SIM,1.0"

    # ------------------------------------------------------------------
    # Safe state (the safe-shutdown standard)
    # ------------------------------------------------------------------

    def safe_shutdown(self) -> None:
        """Put the simulated voltmeter in its safe idle state; never raises.

        Safe idle for a nanovoltmeter is *quiet*, not off: single-shot rather
        than free-running. The measurement range is deliberately left alone
        (a setting, not a hazard) — see the real driver's docstring.
        """
        log.info("SimKeithley2182A: safe shutdown — returning to single-shot.")
        self._continuous_initiation = False

    def _is_in_safe_state(self) -> bool:
        """Return True when the sim is idle (not free-running)."""
        return not self._continuous_initiation

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_error(self) -> None:
        """Raise I2ASCommunicationError if error simulation is active."""
        if self._closed:
            raise I2ASCommunicationError(
                "SimKeithley2182A: the session is closed — the driver was "
                "disconnected from I2AS",
                vi_name="SimKeithley2182A",
            )
        if self._simulate_error:
            raise I2ASCommunicationError(
                "Simulated communication error on Keithley 2182A",
                vi_name="SimKeithley2182A",
            )
