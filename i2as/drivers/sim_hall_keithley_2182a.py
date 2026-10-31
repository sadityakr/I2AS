"""Simulated Keithley 2182A wired across a perpendicular CoFeB Hall bar."""

from __future__ import annotations

import logging
import time

import numpy as np

from i2as.core import sim_environment, sim_samples
from i2as.drivers.sim_keithley_2182a import SimKeithley2182A

log = logging.getLogger(__name__)

#: What a 2182A returns for an over-range conversion.
OVERFLOW_READING_V = 9.9e37


class SimHallKeithley2182A(SimKeithley2182A):
    """Simulated 2182A nanovoltmeter reading the transverse (Hall) voltage.

    Same instrument, same public API as ``SimKeithley2182A``; what differs is
    what its input terminals are wired to. Under the sim-coupling standard
    (``core/sim_environment.py``) the voltmeter joins the environment named
    by its resource string's ``@<name>`` suffix and reads, on every
    conversion:

    * the ``PerpendicularHallBar`` on that environment's sample holder
      (``core/sim_samples.py``) — anomalous + ordinary Hall voltage,
      longitudinal pickup and thermal EMF;
    * the excitation current the sim 6221 on the same environment publishes;
    * the field and sample temperature the magnet and the cryostat publish.

    Instrument behaviour on top of the sample: white noise that grows with
    the selected range (``4 nV + 1.5e-7 x range`` rms at 1 PLC), a slowly
    drifting input offset, a 1-PLC conversion time, and the 2182A's
    over-range reading (``+9.9E37``) when the signal exceeds 120 % of the
    range — so a range chosen too small is visible in the data, as it is on
    the real instrument.
    """

    #: Conversion time of one 1-PLC reading, seconds (50 Hz mains).
    CONVERSION_S = 0.02

    def __init__(self, resource_string: str) -> None:
        """Initialise the voltmeter and join the resource string's environment.

        Args:
            resource_string: e.g. ``"SIM::K2182A_HALL@spin"``; only the
                optional ``@<environment>`` suffix is read.
        """
        super().__init__(resource_string)
        self._range = 0.01
        self._environment = sim_environment.for_resource(resource_string)
        self._rng = np.random.default_rng(2182)
        self._offset = sim_samples.OrnsteinUhlenbeck(12e-9, 60.0, self._rng)

    def get_voltage(self) -> float:
        """Return one Hall-voltage conversion, with the instrument's noise.

        Returns:
            The voltage in volts, or ``9.9e37`` when over range.
        """
        self._check_error()
        time.sleep(self.CONVERSION_S)
        hall_bar = sim_samples.sample_for(self._environment, sim_samples.HALL_BAR)
        signal = hall_bar.hall_voltage_V(self._environment.source_current_A)
        noise_rms = 4e-9 + 1.5e-7 * self._range
        reading = signal + self._offset.value() + noise_rms * float(self._rng.standard_normal())
        if abs(reading) > 1.2 * self._range:
            return OVERFLOW_READING_V
        return reading
