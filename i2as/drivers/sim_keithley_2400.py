"""Simulated Keithley 2400 SourceMeter wired to a perpendicular tunnel junction."""

from __future__ import annotations

import logging
import time

import numpy as np

from i2as.core import sim_environment, sim_samples
from i2as.core.exceptions import I2ASCommunicationError, I2ASInstrumentError

log = logging.getLogger(__name__)

#: The 2400's voltage-source full scale, volts (the 200 V range).
_MAX_SOURCE_V = 210.0
#: The 2400's current measurement ranges, amperes (full scale).
_CURRENT_RANGES = (1e-6, 10e-6, 100e-6, 1e-3, 10e-3, 100e-3, 1.0)
#: Allowed integration times, in power-line cycles.
_NPLC_LIMITS = (0.01, 10.0)


def _range_for(current_A: float) -> float:
    """Return the smallest 2400 current range that holds *current_A*."""
    magnitude = abs(current_A)
    for full_scale in _CURRENT_RANGES:
        if magnitude <= full_scale * 1.05:
            return full_scale
    return _CURRENT_RANGES[-1]


class SimKeithley2400:
    """Simulated Keithley 2400 SourceMeter sourcing voltage, measuring current.

    The instrument is modelled in its constant-voltage configuration (source
    ``VOLT``, sense ``CURR``, autorange on), which is how a tunnel junction is
    measured: the bias across the barrier is what sets the tunnelling physics,
    so it is the quantity held fixed. Its terminals are wired to the
    ``PerpendicularMTJ`` on the sample holder of the sim environment named by
    the resource string's ``@<name>`` suffix (the sim-coupling standard,
    ``core/sim_environment.py``); field and temperature come from that same
    environment.

    Instrument behaviour on top of the junction:

    * **Noise** per conversion: ``30 ppm of range + 60 ppm of reading`` rms
      at 1 PLC, scaling as ``1/sqrt(NPLC)``, plus a 0.4 pA offset.
    * **Conversion time**: ``NPLC / 50`` s per reading (50 Hz mains).
    * **Compliance**: the measured current clips at the current compliance
      and ``is_in_compliance()`` reports it, as the front-panel CMPL flag does.
    * **Output off**: no voltage reaches the sample; a reading returns only
      the offset and noise.

    Safe state (the safe-shutdown standard): output OFF with the source level
    at 0 V.
    """

    def __init__(self, resource_string: str) -> None:
        """Initialise the SourceMeter at its power-up state (output off, 0 V).

        Args:
            resource_string: e.g. ``"SIM::K2400@spin"``; only the optional
                ``@<environment>`` suffix is read.
        """
        self._environment = sim_environment.for_resource(resource_string)
        self._rng = np.random.default_rng(2400)
        self._output_on: bool = False
        self._source_voltage_V: float = 0.0
        self._compliance_A: float = 105e-6
        self._nplc: float = 1.0
        self._in_compliance: bool = False
        self._last_current_A: float = 0.0
        self._simulate_error: bool = False
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Source
    # ------------------------------------------------------------------

    def set_output(self, enabled: bool) -> None:
        """Switch the output relay on or off (``:OUTP ON|OFF``)."""
        self._check_error()
        self._output_on = bool(enabled)

    def get_output(self) -> bool:
        """Return whether the output is on."""
        self._check_error()
        return self._output_on

    def set_source_voltage(self, voltage_V: float) -> None:
        """Set the source voltage level (``:SOUR:VOLT:LEV``).

        Args:
            voltage_V: The voltage, volts.

        Raises:
            I2ASInstrumentError: ``-222`` if outside the ±210 V source range;
                the level is left unchanged.
        """
        self._check_error()
        if abs(float(voltage_V)) > _MAX_SOURCE_V:
            self._refuse(f"set_source_voltage({voltage_V!r})", "-222", "Parameter data out of range")
        self._source_voltage_V = float(voltage_V)

    def get_source_voltage(self) -> float:
        """Return the programmed source voltage, volts."""
        self._check_error()
        return self._source_voltage_V

    def set_current_compliance(self, compliance_A: float) -> None:
        """Set the current compliance (``:SENS:CURR:PROT``), amperes.

        Raises:
            I2ASInstrumentError: ``-222`` if outside ``(0, 1.05] A``.
        """
        self._check_error()
        value = float(compliance_A)
        if not 0.0 < value <= 1.05:
            self._refuse(f"set_current_compliance({compliance_A!r})", "-222", "Parameter data out of range")
        self._compliance_A = value

    def get_current_compliance(self) -> float:
        """Return the current compliance, amperes."""
        self._check_error()
        return self._compliance_A

    def set_nplc(self, nplc: float) -> None:
        """Set the integration time in power-line cycles (``:SENS:CURR:NPLC``).

        Raises:
            I2ASInstrumentError: ``-222`` outside ``[0.01, 10]``.
        """
        self._check_error()
        value = float(nplc)
        if not _NPLC_LIMITS[0] <= value <= _NPLC_LIMITS[1]:
            self._refuse(f"set_nplc({nplc!r})", "-222", "Parameter data out of range")
        self._nplc = value

    def get_nplc(self) -> float:
        """Return the integration time, in power-line cycles."""
        self._check_error()
        return self._nplc

    # ------------------------------------------------------------------
    # Measure
    # ------------------------------------------------------------------

    def measure_current(self) -> float:
        """Trigger one conversion and return the measured current (``:READ?``).

        Returns:
            The current in amperes, clipped at the compliance.
        """
        self._check_error()
        time.sleep(self._nplc / 50.0)
        if self._output_on:
            mtj = sim_samples.sample_for(self._environment, sim_samples.MTJ)
            true_current = mtj.current_A(self._source_voltage_V)
        else:
            true_current = 0.0
        full_scale = _range_for(true_current)
        scale = 1.0 / np.sqrt(self._nplc)
        noise_rms = (30e-6 * full_scale + 60e-6 * abs(true_current)) * scale
        reading = true_current + 0.4e-12 + noise_rms * float(self._rng.standard_normal())
        self._in_compliance = abs(reading) >= self._compliance_A
        if self._in_compliance:
            reading = float(np.copysign(self._compliance_A, reading))
        self._last_current_A = reading
        return reading

    def is_in_compliance(self) -> bool:
        """Return whether the last conversion hit the current compliance."""
        self._check_error()
        return self._in_compliance

    def get_idn(self) -> str:
        """Return the simulated ``*IDN?`` response."""
        self._check_error()
        return "KEITHLEY INSTRUMENTS INC.,MODEL 2400,SIM,C32"

    # ------------------------------------------------------------------
    # Safe state (the safe-shutdown standard)
    # ------------------------------------------------------------------

    def safe_shutdown(self) -> None:
        """Turn the output off and zero the source level; idempotent, never raises."""
        log.info("SimKeithley2400: safe shutdown — output OFF, 0 V.")
        self._output_on = False
        self._source_voltage_V = 0.0

    def _is_in_safe_state(self) -> bool:
        """Return True when no voltage can reach the sample."""
        return not self._output_on and self._source_voltage_V == 0.0

    # ------------------------------------------------------------------
    # Connection lifecycle (the connection-lifecycle standard)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the simulated session; idempotent, never raises."""
        self._closed = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refuse(self, context: str, code: str, message: str) -> None:
        raise I2ASInstrumentError(
            f'Simulated Keithley 2400 refused {context}: {code},"{message}"',
            code=code,
            instrument_message=message,
            context=context,
            vi_name="SimKeithley2400",
        )

    def _check_error(self) -> None:
        if self._closed:
            raise I2ASCommunicationError(
                "SimKeithley2400: the session is closed — the driver was "
                "disconnected from I2AS",
                vi_name="SimKeithley2400",
            )
        if self._simulate_error:
            raise I2ASCommunicationError(
                "Simulated communication error on Keithley 2400",
                vi_name="SimKeithley2400",
            )
