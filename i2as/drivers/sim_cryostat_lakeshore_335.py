"""Simulated Lakeshore 335 regulating a variable-temperature-insert cryostat."""

from __future__ import annotations

import logging
import math
import re
import time

import numpy as np

from i2as.core import sim_environment
from i2as.drivers.sim_lakeshore_335 import SimLakeshore335

log = logging.getLogger(__name__)

#: Heater power at 100 % output on each range, watts (25 Ω heater).
_RANGE_POWER_W = {"OFF": 0.0, "LOW": 0.5, "MEDIUM": 5.0, "HIGH": 50.0}
#: Temperature of the pumped helium the insert's flow gas cools towards, K.
_BASE_TEMPERATURE_K = 1.9
#: Integration step of the thermal model, seconds.
_STEP_S = 0.02
#: Longest gap the model integrates in one go; beyond it the cryostat is
#: treated as having sat in steady state (a stalled poll loop, a debugger).
_MAX_GAP_S = 600.0
#: The optional start-temperature field of the resource string.
_T0_PATTERN = re.compile(r"[?&]T0=([0-9.]+)", re.IGNORECASE)


def _heat_capacity(temperature_K: float) -> float:
    """Return the sample block's heat capacity, J/K (a ~50 g copper block)."""
    t = max(temperature_K, 0.5)
    return 0.01 + 0.004 * t + 2.0e-4 * t * t


def _cooling_conductance(temperature_K: float) -> float:
    """Return the block-to-flow-gas thermal conductance, W/K."""
    return 0.0045 + 6.5e-5 * max(temperature_K, 0.0)


class SimCryostatLakeshore335(SimLakeshore335):
    """A Lakeshore 335 on a VTI cryostat, with a real thermal model and PID.

    Same instrument and public API as ``SimLakeshore335``; what this sim adds
    is the cryostat it regulates, so temperature control looks and behaves as
    it does on a real insert:

    * **Thermal model.** A heater stage, a copper sample block (control
      sensor, input A) and the sample itself, each with its own lag. The
      block's heat capacity rises steeply with temperature and it leaks heat
      to flow gas at ~1.9 K whose cooling power fluctuates (an
      Ornstein–Uhlenbeck process, ±4 %, 15 s) — so a setpoint is held only by
      the loop working, and the heater output wanders as it does.
    * **A real PID loop**, ITC-style parameters as the I2AS temperature VI
      programs them: proportional band ``P`` (K), integral action time ``I``
      (min, 0 = off) and derivative action time ``D`` (min)::

          output % = (100 / P) * (e + (1/I)∫e dt + D de/dt)

      clamped to 0–100 % with anti-windup, times the range's power (Low
      0.5 W, Medium 5 W, High 50 W). The power-up values (``P`` 2 K,
      ``I`` 0.05 min, ``D`` 0) ring at 10 K and oscillate at 300 K; values
      that hold well are about ``1.7 K / 0.033 min / 0.017 min`` at 10 K on
      Low and ``5 K / 0.2 min / 0.25 min`` at 300 K on High. Medium cannot
      hold 300 K at all (the insert loses ~7 W there).
    * **Sensor noise** on every reading: ``0.4 mK + 20 ppm`` rms (a Cernox).

    The resource string may carry a start temperature, ``?T0=<kelvin>``
    (``"SIM::LS335?T0=15@spin"``): the cryostat then powers up regulating
    there on the Low range, as a station left cold overnight does. Without
    it the sim starts at 300 K with the heater off, like its parent. The
    sample temperature is published to the sim environment named by the
    ``@<name>`` suffix (the sim-coupling standard), so every sim on that
    holder sees the same, lagging, sample temperature.
    """

    def __init__(self, resource_string: str) -> None:
        """Initialise the controller and the cryostat it regulates.

        Args:
            resource_string: e.g. ``"SIM::LS335?T0=15@spin"``; only the
                optional ``?T0=`` field and ``@<environment>`` suffix are read.
        """
        super().__init__(resource_string)
        match = _T0_PATTERN.search(str(resource_string))
        start_K = float(match.group(1)) if match else 300.0
        self._rng = np.random.default_rng(335)
        self._heater_K = start_K
        self._block_K = start_K
        self._sensor_K = start_K
        self._sample_K = start_K
        self._temperature = start_K
        self._integral = 0.0
        self._previous_error: float | None = None
        self._flow_noise = 0.0
        self._proportional_band = 2.0
        self._integral_action_time = 0.05
        self._derivative_action_time = 0.0
        if match:
            # Left regulating overnight: closed loop on Low at the start point.
            self._setpoint = start_K
            self._heater_mode = "AUTO"
            self._heater_range = "LOW"
        self._environment = sim_environment.for_resource(resource_string)
        self._environment.temperature_source = self._sample_temperature

    # ------------------------------------------------------------------
    # Readings
    # ------------------------------------------------------------------

    def get_temperature(self) -> float:
        """Return the control sensor's reading, kelvin, with sensor noise."""
        self._check_error()
        self._update_simulation()
        noise = (4e-4 + 2e-5 * self._sensor_K) * float(self._rng.standard_normal())
        return self._sensor_K + noise

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _sample_temperature(self) -> float:
        """Return the sample's own temperature (what the devices see), kelvin."""
        self._update_simulation()
        return self._sample_K

    def _update_simulation(self) -> None:
        """Integrate the cryostat from the last update to now."""
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now
        if elapsed <= 0.0:
            return
        elapsed = min(elapsed, _MAX_GAP_S)
        steps = max(1, int(math.ceil(elapsed / _STEP_S)))
        dt = elapsed / steps
        for _ in range(steps):
            self._step(dt)
        self._temperature = self._sensor_K

    def _pid_output(self, dt: float) -> float:
        """Advance the PID loop one step and return the output, percent."""
        if self._heater_mode != "AUTO":
            return self._heater_output
        error = self._setpoint - self._sensor_K
        derivative = 0.0 if self._previous_error is None else (error - self._previous_error) / dt
        self._previous_error = error
        band = max(self._proportional_band, 1e-3)
        ti_s = self._integral_action_time * 60.0
        td_s = self._derivative_action_time * 60.0
        integral_term = self._integral / ti_s if ti_s > 0.0 else 0.0
        output = (100.0 / band) * (error + integral_term + td_s * derivative)
        # Anti-windup: integrate only while the output is not pinned in the
        # direction the error is pushing it.
        if ti_s > 0.0 and (0.0 < output < 100.0 or (output >= 100.0) != (error > 0.0)):
            self._integral += error * dt
        return min(99.9, max(0.0, output))

    def _step(self, dt: float) -> None:
        """Advance heater stage, block, sensor and sample by *dt* seconds."""
        self._heater_output = self._pid_output(dt)
        power = self._heater_output / 100.0 * _RANGE_POWER_W[self._heater_range]
        decay = math.exp(-dt / 15.0)
        self._flow_noise = self._flow_noise * decay + 0.04 * math.sqrt(1.0 - decay * decay) * float(
            self._rng.standard_normal()
        )
        block_c = _heat_capacity(self._block_K)
        link = block_c / (2.0 + 0.06 * self._block_K)
        to_block = link * (self._heater_K - self._block_K)
        self._heater_K += dt * (power - to_block) / (0.15 * _heat_capacity(self._heater_K))
        loss = _cooling_conductance(self._block_K) * (1.0 + self._flow_noise) * (
            self._block_K - _BASE_TEMPERATURE_K
        )
        self._block_K += dt * (to_block - loss) / block_c
        self._sensor_K += dt * (self._block_K - self._sensor_K) / (0.6 + 0.01 * self._block_K)
        self._sample_K += dt * (self._block_K - self._sample_K) / (3.0 + 0.08 * self._sample_K)
