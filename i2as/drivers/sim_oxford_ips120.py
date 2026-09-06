"""Simulated Oxford IPS 120-10 Magnet Power Supply driver."""

import logging
import time

from i2as.core import sim_environment
from i2as.core.exceptions import (
    I2ASCommunicationError,
    I2ASInstrumentError,
    I2ASSafetyError,
)

log = logging.getLogger(__name__)


class SimOxfordIPS120:
    """Simulated Oxford IPS 120-10 magnet power supply.

    Models current ramping toward a setpoint at a configurable rate.
    Status transitions: HOLD -> RAMPING -> HOLD, and -> QUENCH if the switch
    heater is energised while the PSU and coil currents are mismatched.
    Persistent mode is heater-derived (heater OFF = persistent), mirroring
    the real Mercury iPS driver; ``set_persistent_mode()`` is a no-op.

    The sim-coupling standard (``drivers/README.md``): the PSU publishes its
    present output current to the ``SimEnvironment`` its resource string
    names (``"SIM::IPS_Z@imaging"`` joins ``imaging``; no suffix means a
    private world), so a sim that responds to the applied field — the sim
    camera — observes this magnet without either importing the other. The
    PSU knows only its current; the environment holds the coil constant.

    This driver satisfies the three-rule driver contract:
    1. It is a Python class.
    2. __init__ accepts a single VISA resource string (ignored for simulation).
    3. It is importable via i2as.drivers.sim_oxford_ips120.
    """

    # Physical limits of the real IPS 120-10
    MAX_CURRENT = 90.0   # Amperes
    MIN_CURRENT = -90.0  # Amperes

    def __init__(self, resource_string: str) -> None:
        """Initialise the simulated IPS 120.

        Args:
            resource_string: VISA address (e.g. 'GPIB0::25::INSTR'). Only its
                optional ``@<environment>`` suffix is read (the sim-coupling
                standard); the address itself is ignored.
        """
        # The shared sim world this PSU publishes its current into.
        self._environment = sim_environment.for_resource(resource_string)

        self._current: float = 0.0       # Current output in Amperes
        self._setpoint: float = 0.0      # Target current in Amperes
        self._ramp_rate: float = 5.0     # A/min
        self._status: str = "HOLD"       # "HOLD", "RAMPING", or "QUENCH"
        self._last_update: float = time.time()

        # Switch heater / persistent mode state.
        # The coil current is FROZEN at the PSU current whenever the heater
        # turns off (switch superconducting) and follows the PSU while the
        # heater is on (switch resistive). Persistent mode is heater-derived,
        # exactly like the real Mercury iPS driver.
        self._switch_heater_on: bool = False
        self._coil_current: float = 0.0    # Amps — frozen while heater is OFF

        # Test control flags
        self._simulate_error: bool = False   # Raises I2ASCommunicationError on any get_
        # Connection-lifecycle standard: True once close() has released
        # the session; every command then fails (see _check_error).
        self._closed: bool = False
        self._simulate_quench: bool = False  # Forces status to "QUENCH"
        self._simulate_clamp: bool = False   # Forces a hardware CLMP ("red Clamped") condition
        # A fresh PSU sits at zero: say so, so a world shared with an earlier
        # instance never carries that instance's last current forward.
        self._publish_current()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current(self) -> float:
        """Return the current magnet current in Amperes."""
        self._check_error()
        self._update_simulation()
        return self._current

    def get_current_setpoint(self) -> float:
        """Return the current setpoint in Amperes."""
        self._check_error()
        return self._setpoint

    def set_current_setpoint(self, setpoint: float) -> None:
        """Set the target current.

        Clamps the setpoint to [MIN_CURRENT, MAX_CURRENT].
        If the difference from the current value exceeds 0.01 A, transitions
        to RAMPING status. Ignored while quenched (a real PSU trips and
        requires a reset before accepting ramp commands again).

        Args:
            setpoint: Desired current in Amperes.

        Raises:
            I2ASSafetyError: If ``_simulate_clamp`` is set, mirroring the
                real driver's refusal to ramp while CLAMPED.
        """
        if self._simulate_clamp:
            raise I2ASSafetyError(
                "Sim Mercury iPS-M is CLAMPED (red 'Clamped' indicator). "
                "Ramping is refused until an operator calls clear_clamp().",
            )
        if self._status == "QUENCH":
            return
        clamped = max(self.MIN_CURRENT, min(self.MAX_CURRENT, setpoint))
        self._setpoint = clamped

        if abs(self._setpoint - self._current) > 0.01:
            self._status = "RAMPING"

    def hold(self) -> None:
        """Freeze the output where it is (mirror of the Mercury HOLD action)."""
        self._check_error()
        if self._status == "QUENCH":
            return
        self._update_simulation()
        self._setpoint = self._current
        self._status = "HOLD"

    def set_ramp_rate(self, rate: float) -> None:
        """Set the current ramp rate.

        Models the instrument-error half of the driver error-reporting
        standard: while the PSU is CLAMPED it answers a SIG: write with
        ``DENIED`` on the acknowledgement line and does not apply it. The
        real driver detects exactly that reply in ``_write()``; here it is
        raised directly, so a caller that never checked for a clamp finds
        out at the point of the write rather than by wondering why the ramp
        rate never changed.

        Args:
            rate: Ramp rate in A/min. Must be positive.

        Raises:
            ValueError: If *rate* is not positive (a programming error,
                caught before anything reaches the bus).
            I2ASInstrumentError: ``DENIED`` if the PSU is clamped.
        """
        if rate <= 0:
            raise ValueError(f"Ramp rate must be positive, got {rate}")
        if self._simulate_clamp:
            context = f"set_ramp_rate({rate!r})"
            raise I2ASInstrumentError(
                f"Simulated IPS 120 refused {context}: "
                f"STAT:SET:DEV:GRPZ:PSU:SIG:RCST:DENIED — the PSU is CLAMPED",
                code="DENIED",
                instrument_message="STAT:SET:DEV:GRPZ:PSU:SIG:RCST:DENIED",
                context=context,
                vi_name="SimOxfordIPS120",
            )
        self._ramp_rate = rate

    def get_ramp_rate(self) -> float:
        """Return the current ramp rate in Amperes per minute."""
        self._check_error()
        return self._ramp_rate

    def get_status(self) -> str:
        """Return the current status string.

        Returns:
            One of "HOLD", "RAMPING", or "QUENCH".
        """
        self._check_error()
        if self._simulate_quench:
            return "QUENCH"
        self._update_simulation()
        return self._status

    def get_idn(self) -> str:
        """Return simulated identification string."""
        self._check_error()
        return "OXFORD,IPS120,SIM,1.0"

    # ------------------------------------------------------------------
    # Switch heater / persistent mode API
    # ------------------------------------------------------------------

    def get_switch_heater_state(self) -> str:
        """Return 'ON' if the switch heater is energised, 'OFF' otherwise."""
        self._check_error()
        return "ON" if self._switch_heater_on else "OFF"

    def set_switch_heater(self, state: bool) -> None:
        """Energise (True) or de-energise (False) the persistent mode switch heater.

        Models the real physics of the switch:

        * Turning the heater ON while the PSU output differs from the frozen
          coil current QUENCHES the magnet — the stored coil current is forced
          through the now-resistive switch. This is exactly the failure the
          VI's ramp sequence must never trigger; the sim makes it loud so a
          wrong command order fails in tests instead of on hardware.
        * Turning the heater OFF freezes the coil current at the present PSU
          current (switch superconducting; coil current now circulates).

        Args:
            state: True to turn on, False to turn off.
        """
        if state == self._switch_heater_on:
            return
        self._update_simulation()
        if state:
            if abs(self._current - self._coil_current) > 0.05:
                self._quench()
                return
            self._switch_heater_on = True
        else:
            self._coil_current = self._current
            self._switch_heater_on = False

    def get_coil_current(self) -> float:
        """Return the coil current in Amperes.

        While the heater is ON the switch is resistive and the coil follows
        the PSU output; while OFF the coil current is frozen at the value it
        had when the heater was last turned off.
        """
        self._check_error()
        if self._switch_heater_on:
            self._update_simulation()
            return self._current
        return self._coil_current

    def get_persistent_mode(self) -> bool:
        """Return True when the magnet is in persistent mode.

        Heater-derived (switch heater OFF means the switch is superconducting
        and the coil holds its current) — mirrors the real Mercury iPS driver,
        which has no independent persistent-mode flag.
        """
        self._check_error()
        return not self._switch_heater_on

    def set_persistent_mode(self, persistent: bool) -> None:
        """No-op, mirroring the real Mercury iPS driver.

        Persistent mode is managed entirely through the switch heater and
        ramp commands; the VI layer sequences them.

        Args:
            persistent: Ignored.
        """
        _ = persistent

    def reset_quench(self) -> None:
        """Clear a quench (test helper — a real PSU needs a manual reset)."""
        self._status = "HOLD"
        self._simulate_quench = False
        self._setpoint = self._current

    def clear_clamp(self) -> None:
        """Explicitly unclamp the PSU, mirroring the real driver's method.

        Never called automatically by :meth:`get_status` or
        :meth:`set_current_setpoint` — see the real driver's docstring for
        why this must stay a deliberate, human-initiated action.

        Raises:
            I2ASSafetyError: If the PSU is not actually clamped.
        """
        if not self._simulate_clamp:
            raise I2ASSafetyError(
                "clear_clamp() called but sim Mercury iPS-M is not CLAMPED",
            )
        self._simulate_clamp = False
        self.hold()

    # ------------------------------------------------------------------
    # Internal simulation logic
    # ------------------------------------------------------------------

    def _quench(self) -> None:
        """Model a quench: coil energy dumps, currents collapse, PSU trips."""
        self._status = "QUENCH"
        self._current = 0.0
        self._coil_current = 0.0
        self._setpoint = 0.0
        self._switch_heater_on = False
        self._publish_current()

    def _publish_current(self) -> None:
        """Publish the present output current to the shared sim environment."""
        self._environment.psu_current_A = self._current

    def _update_simulation(self) -> None:
        """Advance simulated current toward setpoint based on elapsed real time."""
        now = time.time()
        dt_min = (now - self._last_update) / 60.0
        self._last_update = now

        if self._status != "RAMPING":
            return

        max_step = self._ramp_rate * dt_min
        remaining = self._setpoint - self._current
        if abs(remaining) <= max_step:
            self._current = self._setpoint
            self._status = "HOLD"
        else:
            direction = 1 if remaining > 0 else -1
            self._current += direction * max_step
        self._publish_current()

    # ------------------------------------------------------------------
    # Safe state (the safe-shutdown standard)
    # ------------------------------------------------------------------

    def safe_shutdown(self) -> None:
        """Freeze the simulated magnet where it is; idempotent, never raises.

        Safe idle for a superconducting magnet is HOLD, not zero field — see
        the real driver's docstring for why ramping down is a supervised
        operation and never a cleanup step. The switch heater is left
        untouched, because changing it across a PSU/coil current mismatch is
        what quenches the magnet.
        """
        log.info("SimOxfordIPS120: safe shutdown — HOLD (magnet stays at field).")
        self._update_simulation()
        self._setpoint = self._current
        if self._status == "RAMPING":
            self._status = "HOLD"

    def _is_in_safe_state(self) -> bool:
        """Return True when no ramp is running and the setpoint is where the PSU is."""
        return self._status != "RAMPING" and self._setpoint == self._current

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
                "SimOxfordIPS120: the session is closed — the driver was "
                "disconnected from I2AS",
                vi_name="SimOxfordIPS120",
            )
        if self._simulate_error:
            raise I2ASCommunicationError(
                "Simulated communication error on IPS 120",
                vi_name="SimOxfordIPS120",
            )
