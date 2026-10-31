"""Simulated devices on the sample holder — the physics coupled sims measure.

The **sim-coupling standard** (``core/sim_environment.py``) keeps every sim
driver modelling ONE instrument; what the instruments are wired to lives
here. Two devices from one perpendicular-anisotropy CoFeB/MgO stack sit on
the holder of the simulated spintronics rack (``configs/sim_spintronics``):

* ``PerpendicularMTJ`` — a CoFeB/MgO/CoFeB magnetic tunnel junction with a
  pinned (synthetic-antiferromagnet) reference layer and a free layer that
  switches under an out-of-plane field. A source-measure unit applies a
  constant bias and reads the tunnelling current: a strongly non-linear
  (sinh-like, Simmons/Brinkman) I–V, a tunnelling magnetoresistance that
  falls with bias and with temperature, and a free-layer loop whose
  coercivity shrinks with bias (Joule heating) and whose centre shifts with
  it (spin-transfer torque).
* ``PerpendicularHallBar`` — a Ta/CoFeB(1.1 nm)/MgO Hall cross. A current
  source drives it and a nanovoltmeter reads the transverse voltage: a
  square anomalous-Hall loop from a Preisach ensemble of grains, a small
  ordinary-Hall slope, a longitudinal pickup offset and a drifting thermal
  EMF.

Both read the applied field and the sample temperature from the
``SimEnvironment`` they are created in, carry their own magnetic memory, and
add the noise a real bench has: white instrument noise, 1/f-like drift
(Ornstein–Uhlenbeck processes), and stochastic switching fields. Every
parameter is a class attribute with its unit, so a test or a demo can read
the "true" values the analysis should recover.

This module is foundation code for the drivers (contract C2 allows drivers
to import ``i2as.core`` modules below the Station); it imports only numpy and
the standard library.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

#: Kinds ``sample_for`` knows how to build.
MTJ = "mtj"
HALL_BAR = "hall_bar"


class OrnsteinUhlenbeck:
    """A mean-reverting random walk — the 1/f-like drift of a real bench.

    Advanced by wall-clock time, so a drift seen across a 20-minute run looks
    the same whether the run reads ten times a second or once a minute.
    """

    def __init__(self, sigma: float, tau_s: float, rng: np.random.Generator) -> None:
        """Create the process at zero.

        Args:
            sigma: Stationary standard deviation.
            tau_s: Correlation time, in seconds.
            rng: The random generator the process draws from.
        """
        self.sigma = float(sigma)
        self.tau_s = float(tau_s)
        self._rng = rng
        self._value = 0.0
        self._last = time.monotonic()

    def value(self) -> float:
        """Return the process value now (advancing it by the elapsed time)."""
        now = time.monotonic()
        dt = max(0.0, now - self._last)
        self._last = now
        if dt > 0.0 and self.sigma > 0.0:
            decay = math.exp(-dt / self.tau_s)
            spread = self.sigma * math.sqrt(max(0.0, 1.0 - decay * decay))
            self._value = self._value * decay + spread * float(self._rng.standard_normal())
        return self._value


class PerpendicularMTJ:
    """A perpendicular CoFeB/MgO/CoFeB tunnel junction under bias and field.

    Magnetic state: the reference layer points up (+1) unless a field beyond
    ``REFERENCE_SWITCH_T`` flips it; the free layer is +1 (parallel, low
    resistance) or -1 (antiparallel, high resistance). The free layer switches
    up when the field exceeds ``h_up`` and down when it falls below
    ``h_down``; both thresholds are redrawn after every switch (thermally
    activated, Néel–Brown-like scatter) and depend on temperature and bias::

        Hc(T, V) = HC0 * (1 - sqrt(T / HC_T0)) * (1 - |V| / HC_V_SCALE)
        centre   = OFFSET_T - STT_T_PER_V * V
        h_up     = centre + Hc,   h_down = centre - Hc

    Transport (Simmons/Brinkman-like, slightly asymmetric barrier)::

        I_P(V)  = (V0 / R_P(T)) * sinh(V / V0) * (1 + ASYMMETRY * V)
        TMR(V)  = TMR(T) / (1 + (V / V_half(T))^2)
        I_AP(V) = I_P(V) / (1 + TMR(V))

    with ``TMR(T) = TMR0 * (1 - (T / TMR_T0)^1.5)`` (thermal magnons) and the
    parallel resistance rising slightly on cooling. Beyond ``BREAKDOWN_V`` the
    MgO barrier breaks down for good — the reason the measurement VI bounds
    the bias it may apply.
    """

    #: Zero-bias parallel resistance at 300 K, ohms (RA ≈ 6 Ω·µm², 60 nm pillar).
    R_P_300K = 2.05e3
    #: Fractional rise of R_P from 300 K down to 0 K.
    R_P_COLD_RISE = 0.045
    #: Brinkman non-linearity voltage scale, volts.
    V0 = 0.30
    #: Barrier asymmetry, per volt.
    ASYMMETRY = 0.04
    #: Zero-bias, zero-temperature TMR ratio (1.86 = 186 %).
    TMR0 = 1.86
    #: Magnon temperature scale for the TMR decay, kelvin (TMR(300 K) ≈ 120 %).
    TMR_T0 = 602.0
    #: Bias at which the TMR has halved, volts, at 10 K and at 300 K.
    V_HALF_COLD = 0.52
    V_HALF_WARM = 0.43
    #: Free-layer coercivity scale (T), its temperature scale (K) and its bias
    #: suppression scale (V). Hc(10 K) ≈ 45 mT, Hc(300 K) ≈ 22 mT at zero bias.
    HC0 = 0.050
    HC_T0 = 957.0
    HC_V_SCALE = 1.6
    #: Loop centre at zero bias (stray field of the reference stack), tesla.
    OFFSET_T = -0.0065
    #: Spin-transfer shift of the loop centre, tesla per volt.
    STT_T_PER_V = 0.012
    #: Thermal scatter of the switching fields, tesla, at 10 K and 300 K.
    SWITCH_SIGMA_COLD = 0.0004
    SWITCH_SIGMA_WARM = 0.0011
    #: Field that flips the pinned reference layer, tesla.
    REFERENCE_SWITCH_T = 0.65
    #: Bias magnitude beyond which the barrier breaks down, volts.
    BREAKDOWN_V = 1.35
    #: Resistance of a broken-down (shorted) junction, ohms.
    SHORTED_R = 48.0
    #: Per-reading resistance fluctuation (barrier 1/f noise integrated over
    #: one conversion), relative rms, parallel and antiparallel — the AP
    #: state is the noisier one in MgO junctions.
    WHITE_NOISE_P = 0.0012
    WHITE_NOISE_AP = 0.0028

    def __init__(self, environment: Any, seed: int = 20261031) -> None:
        """Create the junction, saturated parallel.

        Args:
            environment: The ``SimEnvironment`` providing field and temperature.
            seed: Seed for the junction's own random generator.
        """
        self._environment = environment
        self._rng = np.random.default_rng(seed)
        self.reference = 1
        self.free = 1
        self.broken = False
        self._bias_V = 0.0
        # Slow resistance drift (barrier charge trapping) and slower thermal
        # wander of the contact/lead resistance.
        self._drift = OrnsteinUhlenbeck(6e-4, 30.0, self._rng)
        self._lead = OrnsteinUhlenbeck(0.6, 120.0, self._rng)
        self._noise_up = float(self._rng.standard_normal())
        self._noise_down = float(self._rng.standard_normal())

    # -- environment --------------------------------------------------

    @property
    def field_T(self) -> float:
        """The applied out-of-plane field, tesla."""
        return float(self._environment.applied_field_T)

    @property
    def temperature_K(self) -> float:
        """The sample temperature, kelvin."""
        return float(self._environment.sample_temperature_K)

    # -- parameters as functions of T and V -----------------------------

    def tmr_zero_bias(self, temperature_K: float) -> float:
        """Return the zero-bias TMR ratio at *temperature_K*."""
        t = max(0.0, temperature_K)
        return self.TMR0 * max(0.05, 1.0 - (t / self.TMR_T0) ** 1.5)

    def v_half(self, temperature_K: float) -> float:
        """Return the bias at which the TMR halves, volts."""
        frac = min(1.0, max(0.0, (temperature_K - 10.0) / 290.0))
        return self.V_HALF_COLD + (self.V_HALF_WARM - self.V_HALF_COLD) * frac

    def resistance_p(self, temperature_K: float) -> float:
        """Return the zero-bias parallel resistance, ohms."""
        frac = 1.0 - min(1.0, max(0.0, temperature_K) / 300.0)
        return self.R_P_300K * (1.0 + self.R_P_COLD_RISE * frac)

    def coercivity(self, temperature_K: float, bias_V: float) -> float:
        """Return the free layer's coercive field, tesla."""
        thermal = max(0.05, 1.0 - math.sqrt(max(0.0, temperature_K) / self.HC_T0))
        joule = max(0.1, 1.0 - abs(bias_V) / self.HC_V_SCALE)
        return self.HC0 * thermal * joule

    def switching_fields(self, temperature_K: float, bias_V: float) -> tuple[float, float]:
        """Return ``(h_down, h_up)`` — this cycle's switching fields, tesla."""
        hc = self.coercivity(temperature_K, bias_V)
        centre = self.OFFSET_T - self.STT_T_PER_V * bias_V
        frac = min(1.0, max(0.0, (temperature_K - 10.0) / 290.0))
        sigma = self.SWITCH_SIGMA_COLD + (self.SWITCH_SIGMA_WARM - self.SWITCH_SIGMA_COLD) * frac
        return (
            centre - hc + sigma * self._noise_down,
            centre + hc + sigma * self._noise_up,
        )

    # -- dynamics ----------------------------------------------------------

    def _advance(self, bias_V: float) -> None:
        """Update the magnetic state for the present field, temperature and bias."""
        if abs(bias_V) > self.BREAKDOWN_V:
            self.broken = True
        field = self.field_T
        if field > self.REFERENCE_SWITCH_T:
            self.reference = 1
        elif field < -self.REFERENCE_SWITCH_T:
            self.reference = -1
        h_down, h_up = self.switching_fields(self.temperature_K, bias_V)
        if self.free < 0 and field > h_up:
            self.free = 1
            self._noise_up = float(self._rng.standard_normal())
        elif self.free > 0 and field < h_down:
            self.free = -1
            self._noise_down = float(self._rng.standard_normal())

    def _reading_state(self, bias_V: float) -> dict[str, Any]:
        """Draw everything one conversion shares: state, drift, noise.

        Advances the magnetic state for *bias_V*, then fixes for this one
        reading the drifting resistances, whether the free layer reads as
        parallel (random telegraph noise near a switching field at high
        temperature) and the per-reading noise factor — so a solver may
        evaluate the junction at many trial biases within one reading.
        """
        self._advance(bias_V)
        temperature = self.temperature_K
        parallel = self.free == self.reference
        if temperature > 150.0:
            # Random telegraph noise close to a switching field at high
            # temperature: the free layer hops thermally for one reading
            # without switching for good.
            h_down, h_up = self.switching_fields(temperature, bias_V)
            margin = abs(self.field_T - (h_up if self.free < 0 else h_down))
            if margin < 0.0012 and self._rng.random() < 0.22:
                parallel = not parallel
        white = self.WHITE_NOISE_P if parallel else self.WHITE_NOISE_AP
        return {
            "temperature": temperature,
            "parallel": parallel,
            "r_p": self.resistance_p(temperature) * (1.0 + self._drift.value()),
            "lead": 22.0 + self._lead.value(),
            "noise": 1.0 + white * float(self._rng.standard_normal()),
        }

    def _transport(self, v: float, state: dict[str, Any]) -> float:
        """Return the current at bias *v* for one reading's fixed *state*."""
        if self.broken:
            return v / (self.SHORTED_R + state["lead"])
        if v == 0.0:
            return 0.0
        r_p, temperature = state["r_p"], state["temperature"]
        i_p = (self.V0 / r_p) * math.sinh(v / self.V0) * (1.0 + self.ASYMMETRY * v)
        if state["parallel"]:
            i_junction = i_p
        else:
            tmr = self.tmr_zero_bias(temperature) / (1.0 + (v / self.v_half(temperature)) ** 2)
            i_junction = i_p / (1.0 + tmr)
        i_junction *= state["noise"]
        # Leads in series: a small correction, solved to first order.
        return v / (v / i_junction + state["lead"])

    def current_A(self, bias_V: float) -> float:
        """Apply a bias and return the tunnelling current (one reading).

        Args:
            bias_V: The voltage across the junction, volts.

        Returns:
            The current, amperes (sign follows the bias).
        """
        self._bias_V = float(bias_V)
        state = self._reading_state(self._bias_V)
        return self._transport(self._bias_V, state)

    def voltage_for_current(self, current_A: float, compliance_V: float) -> tuple[float, float]:
        """Drive a current through the junction and return what develops (one reading).

        The voltage the junction needs is found by bisection on the
        (monotonic) I–V of this reading. If it exceeds *compliance_V* the
        source clamps at the compliance and less current flows — exactly
        what protects a barrier from a mistyped current.

        Args:
            current_A: The sourced current, amperes.
            compliance_V: The source's voltage compliance, volts (> 0).

        Returns:
            ``(voltage_V, current_A)`` actually across and through the junction.
        """
        state = self._reading_state(self._bias_V)
        limit = abs(float(compliance_V))
        target = float(current_A)
        if abs(self._transport(math.copysign(limit, target), state)) <= abs(target):
            v = math.copysign(limit, target) if target != 0.0 else 0.0
        else:
            lo, hi = (0.0, limit) if target > 0 else (-limit, 0.0)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if self._transport(mid, state) < target:
                    lo = mid
                else:
                    hi = mid
            v = 0.5 * (lo + hi)
        self._bias_V = v
        self._advance(v)
        return v, self._transport(v, state)


class PerpendicularHallBar:
    """A Ta/CoFeB/MgO Hall cross with perpendicular anisotropy.

    A Preisach ensemble of ``N_GRAINS`` grains, each switching at
    ``Hc(T) * (1 + SPREAD * g)`` around an exchange-bias-free centre, gives the
    square-but-not-perfectly-square anomalous-Hall loop of a real PMA film.
    The transverse voltage under a current ``I`` is::

        V_xy = I * (R_AHE(T) * m + R_OHE * H + R_PICKUP(T)) + V_emf(t) + noise

    ``m`` is the ensemble's mean magnetisation, ``R_PICKUP`` the longitudinal
    voltage a slightly misaligned Hall cross picks up, ``V_emf`` a slowly
    drifting thermal EMF (which a current reversal would cancel).
    """

    N_GRAINS = 600
    #: Anomalous-Hall resistance at 10 K and 300 K, ohms.
    R_AHE_COLD = 2.42
    R_AHE_WARM = 1.86
    #: Ordinary-Hall slope, ohms per tesla.
    R_OHE = -0.11
    #: Longitudinal pickup at 10 K and 300 K, ohms (R_xx ≈ 1.1 kΩ, 0.03 % misalignment).
    R_PICKUP_COLD = 0.29
    R_PICKUP_WARM = 0.36
    #: Coercivity scale (T) and temperature scale (K): Hc(10 K) ≈ 14 mT, Hc(300 K) ≈ 5 mT.
    HC0 = 0.0160
    HC_T0 = 640.0
    #: Relative grain-to-grain spread of switching fields.
    SPREAD = 0.06
    #: Mean thermal EMF and its drift, volts.
    V_EMF = 0.6e-6
    V_EMF_DRIFT = 0.25e-6
    #: Transverse-resistance fluctuation (thermal gradients, contact noise):
    #: a correlated part and a per-reading part, ohms rms.
    R_FLUCTUATION = 1.5e-3
    R_WHITE = 1.0e-3

    def __init__(self, environment: Any, seed: int = 31102026) -> None:
        """Create the film saturated up.

        Args:
            environment: The ``SimEnvironment`` providing field and temperature.
            seed: Seed for the film's own random generator.
        """
        self._environment = environment
        self._rng = np.random.default_rng(seed)
        # A small skewed tail: a few grains pinned at defects switch late,
        # rounding the loop's approach to saturation.
        g = self._rng.standard_normal(self.N_GRAINS)
        tail = self._rng.exponential(0.12, self.N_GRAINS) * (self._rng.random(self.N_GRAINS) < 0.08)
        self._g = g * self.SPREAD + tail
        self._m = np.ones(self.N_GRAINS)
        self._emf = OrnsteinUhlenbeck(self.V_EMF_DRIFT, 90.0, self._rng)
        self._r_fluctuation = OrnsteinUhlenbeck(self.R_FLUCTUATION, 2.0, self._rng)

    @property
    def field_T(self) -> float:
        """The applied out-of-plane field, tesla."""
        return float(self._environment.applied_field_T)

    @property
    def temperature_K(self) -> float:
        """The sample temperature, kelvin."""
        return float(self._environment.sample_temperature_K)

    def _interp(self, cold: float, warm: float) -> float:
        frac = min(1.0, max(0.0, (self.temperature_K - 10.0) / 290.0))
        return cold + (warm - cold) * frac

    def coercivity(self, temperature_K: float) -> float:
        """Return the film's coercive field, tesla."""
        return self.HC0 * max(0.05, 1.0 - math.sqrt(max(0.0, temperature_K) / self.HC_T0))

    def magnetisation(self) -> float:
        """Advance the grains to the present field and return the mean ``m``."""
        field = self.field_T
        thresholds = self.coercivity(self.temperature_K) * (1.0 + self._g)
        self._m[field > thresholds] = 1.0
        self._m[field < -thresholds] = -1.0
        return float(self._m.mean())

    def hall_voltage_V(self, current_A: float) -> float:
        """Return the transverse voltage under *current_A* (noise excluded).

        Args:
            current_A: The current through the Hall bar, amperes.

        Returns:
            The Hall voltage, volts — including the thermal EMF, which is
            present whether or not current flows.
        """
        m = self.magnetisation()
        r_xy = (
            self._interp(self.R_AHE_COLD, self.R_AHE_WARM) * m
            + self.R_OHE * self.field_T
            + self._interp(self.R_PICKUP_COLD, self.R_PICKUP_WARM)
            + self._r_fluctuation.value()
            + self.R_WHITE * float(self._rng.standard_normal())
        )
        return current_A * r_xy + self.V_EMF + self._emf.value()


_FACTORIES = {MTJ: PerpendicularMTJ, HALL_BAR: PerpendicularHallBar}


def sample_for(environment: Any, kind: str) -> Any:
    """Return the device of *kind* on *environment*'s sample holder.

    Created on first use and stored in ``environment.samples``, so every sim
    wired to the same device (and the test inspecting it) sees one object.

    Args:
        environment: A ``SimEnvironment``.
        kind: ``MTJ`` or ``HALL_BAR``.

    Returns:
        The device model.

    Raises:
        KeyError: If *kind* names no known device.
    """
    if kind not in _FACTORIES:
        raise KeyError(f"no simulated sample kind {kind!r}; known: {sorted(_FACTORIES)}")
    samples = environment.samples
    if kind not in samples:
        samples[kind] = _FACTORIES[kind](environment)
    return samples[kind]


__all__ = [
    "HALL_BAR",
    "MTJ",
    "OrnsteinUhlenbeck",
    "PerpendicularHallBar",
    "PerpendicularMTJ",
    "sample_for",
]
