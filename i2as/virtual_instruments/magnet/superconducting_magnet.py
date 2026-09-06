"""SuperconductingMagnetVI — behavior-based VI for any SC magnet PSU (no switch heater)."""

from __future__ import annotations

import logging
from typing import Any, Generator

from i2as.core.decorators import control, monitored
from i2as.core.plan import ParamSpec
from i2as.virtual_instruments.base import MagnetBase
from i2as.virtual_instruments.rampable import RampableVI

logger = logging.getLogger(__name__)


class SuperconductingMagnetVI(MagnetBase, RampableVI):
    """Virtual Instrument for a superconducting magnet power supply without switch heater.

    Ramp behaviour
    --------------
    The PSU ramps to a *current* setpoint continuously. This VI implements
    a status-driven ramp:

    1. ``start_ramp(target_T)`` converts to amperes, clamps, and creates a
       generator that walks through intermediate ramp segments.
    2. ``advance_ramp()`` calls ``next()`` on the generator each Orchestrator
       tick, sending a new setpoint when the driver reports ``"HOLD"``.
    3. ``ramp_status()`` reports ``"RAMPING"``/``"TARGET_REACHED"``/``"IDLE"``.

    Driver contract
    ---------------
    The ``"main"`` driver must implement:
    * ``get_current() -> float``         — output current in Amperes
    * ``get_status() -> str``            — "HOLD" | "RAMPING" | "QUENCH"
    * ``set_current_setpoint(float)``    — set target current
    * ``set_ramp_rate(float)``           — ramp rate in A/min

    Optionally:
    * ``hold()`` — freeze the output where it is (used by ``stop_ramp()``;
      without it the current output is re-sent as the setpoint instead).

    The physical mapping (e.g. whether the instrument takes rate + setpoint
    simultaneously or rate first then setpoint) is the driver's responsibility.
    """

    # Control-validation standard (see BaseVirtualInstrument): user-facing
    # set_field() is bounded by the setup's field limit, derived in __init__
    # from the config's max_current / min_current (or explicit
    # min_field_T / max_field_T keys when the config provides them).
    control_limits = {"set_field": {"target_T": "field_T"}}

    def __init__(self, drivers: dict[str, object], **init_params: Any) -> None:
        super().__init__(drivers, **init_params)
        self._driver = drivers["main"]

        self._amperes_per_tesla: float = float(init_params.get("amperes_per_tesla", 10.0))
        self._default_ramp_rate: float = float(init_params.get("default_ramp_rate", 5.0))
        self._ramp_segments: list[dict] = list(init_params.get("ramp_segments", []))
        self._max_current: float = float(init_params.get("max_current", 90.0))
        self._min_current: float = float(init_params.get("min_current", -90.0))

        # Field limit for set_field(): explicit config keys win; otherwise
        # derived from the setup's current limits.
        min_field = init_params.get("min_field_T")
        max_field = init_params.get("max_field_T")
        self._limits["field_T"] = (
            float(min_field) if min_field is not None
            else self._min_current / self._amperes_per_tesla,
            float(max_field) if max_field is not None
            else self._max_current / self._amperes_per_tesla,
        )

        self._ramp_gen: Generator | None = None
        self._ramp_exhausted: bool = True
        self._ramp_target_T: float | None = None
        #: The last setpoint commanded to the PSU, in tesla — the "next
        #: setpoint" of the ramp-introspection standard (RampableVI.
        #: ramp_setpoint). Distinct from _ramp_target_T whenever the
        #: generator stops at a ramp-segment boundary on the way there.
        self._ramp_setpoint_T: float | None = None

    # ------------------------------------------------------------------
    # RampableVI implementation
    # ------------------------------------------------------------------

    def start_ramp(self, target: float) -> None:
        """Begin ramping to *target* tesla.

        Args:
            target: Target field in tesla.
        """
        target_A = self._clamp_target_A(target * self._amperes_per_tesla)
        self._ramp_target_T = target_A / self._amperes_per_tesla
        # Cleared here, not in the generator: the first next() below sends the
        # first setpoint and records it, so a stale value from the previous
        # ramp is never reported for this one.
        self._ramp_setpoint_T = None

        self._ramp_gen = self._ramp_generator(target_A)
        self._ramp_exhausted = False
        try:
            next(self._ramp_gen)
        except StopIteration:
            self._ramp_exhausted = True

    def _clamp_target_A(self, target_A: float) -> float:
        """Clamp a target current to the setup's limits, loudly.

        Last-resort hardware protection for programmatic callers
        (procedures). User-facing set_field() already rejects loudly via the
        control-limits wrapper before reaching this point, so a silent clamp
        here would only ever hide a bug — hence the warning.
        """
        clamped_A = max(self._min_current, min(self._max_current, target_A))
        if clamped_A != target_A:
            logger.warning(
                "%s: requested %.4g A exceeds current limits [%.4g, %.4g] — "
                "clamped to %.4g A",
                self.vi_name or type(self).__name__,
                target_A, self._min_current, self._max_current, clamped_A,
            )
        return clamped_A

    def advance_ramp(self) -> None:
        """Advance the ramp generator by one tick."""
        if self._ramp_gen is None or self._ramp_exhausted:
            return
        try:
            next(self._ramp_gen)
        except StopIteration:
            self._ramp_exhausted = True

    def ramp_status(self) -> str:
        """Return current ramp state.

        Returns:
            ``"IDLE"``           — no generator active.
            ``"TARGET_REACHED"`` — generator exhausted and hardware reports HOLD.
            ``"RAMPING"``        — generator still running or hardware ramping.
        """
        if self._ramp_gen is None:
            return "IDLE"
        if self._ramp_exhausted:
            hw_status = self._driver.get_status()  # type: ignore[attr-defined]
            if hw_status == "HOLD":
                return "TARGET_REACHED"
            return "RAMPING"
        return "RAMPING"

    def stop_ramp(self) -> None:
        """Stop the ramp: kill the generator AND command the PSU to hold.

        Clearing the generator alone is not enough — the PSU is autonomous and
        keeps ramping to its last-commanded setpoint. If the driver exposes a
        ``hold()`` method it is called to freeze the output where it is;
        otherwise the current output value is re-sent as the setpoint.
        """
        self._ramp_gen = None
        self._ramp_exhausted = True
        driver = self._driver  # type: ignore[attr-defined]
        hold = getattr(driver, "hold", None)
        if callable(hold):
            hold()
        else:
            driver.set_current_setpoint(driver.get_current())
        self._ramp_target_T = None
        self._ramp_setpoint_T = None

    def ramp_target(self) -> float | None:
        """Return the active field target in tesla, or ``None`` when idle.

        Set when ``start_ramp`` begins (the clamped target, so it matches the
        field the magnet will actually reach) and cleared by ``stop_ramp``.
        """
        return self._ramp_target_T

    def ramp_rate(self) -> float | None:
        """Return the current ramp segment's rate in tesla/min, or ``None`` when idle.

        The magnet ramps through current-dependent segments, so this is the rate
        for the segment the output is currently in, read live and converted from
        the internal amperes/min to tesla/min for consistency with
        ``get_field()`` and ``ramp_target()``.
        """
        if self._ramp_target_T is None:
            return None
        curr_A = self._driver.get_current()  # type: ignore[attr-defined]
        target_A = self._ramp_target_T * self._amperes_per_tesla
        direction = 1 if target_A >= curr_A else -1
        return self._get_segment_rate(curr_A, direction) / self._amperes_per_tesla

    def nominal_ramp_rate(self) -> float | None:
        """Return the slowest configured ramp rate, in tesla/min.

        The declared rate a **duration estimate** is built from (see
        ``RampableVI.nominal_ramp_rate``): read from the configured ramp
        segments alone — no bus traffic, no active ramp — and reported as the
        SLOWEST of them, because a sweep crossing into a high-current segment
        pays that rate and an estimate must never be optimistic. A magnet
        configured without segments reports its ``default_ramp_rate``.
        Converted from the internal amperes/min to tesla/min for consistency
        with ``ramp_target()``.

        Returns:
            The nominal field ramp rate in tesla/min.
        """
        rates = [
            float(segment["rate_A_per_min"])
            for segment in self._ramp_segments
            if "rate_A_per_min" in segment
        ]
        slowest_A_per_min = min(rates) if rates else self._default_ramp_rate
        return slowest_A_per_min / self._amperes_per_tesla

    def ramp_setpoint(self) -> float | None:
        """Return the setpoint last commanded to the PSU, in tesla.

        The ramp walks to its target through the configured ``ramp_segments``,
        stopping at each boundary to change rate, so this is the boundary the
        PSU is driving to *now* — not necessarily ``ramp_target()``. Recorded
        by the generator as it commands each setpoint; no hardware read.
        """
        return self._ramp_setpoint_T

    def ramp_value(self) -> float | None:
        """Return the current field in tesla (the value the ramp drives)."""
        return self.magnet_field_T()

    # ------------------------------------------------------------------
    # Internal generator
    # ------------------------------------------------------------------

    def _ramp_generator(self, target_A: float) -> Generator:
        driver = self._driver  # type: ignore[attr-defined]

        while True:
            curr_A = driver.get_current()
            if abs(curr_A - target_A) <= 0.01:
                return

            status = driver.get_status()
            if status == "QUENCH":
                # Send no further setpoints; the Station safety check is
                # responsible for escalating a quench to EMERGENCY.
                return
            if status == "RAMPING":
                yield
                continue

            direction = 1 if target_A > curr_A else -1
            rate = self._get_segment_rate(curr_A, direction)

            next_boundary = target_A
            for seg in self._ramp_segments:
                limit = float(seg["max_current_A"])
                if limit == float('inf'):
                    continue
                if direction > 0:
                    if curr_A < limit - 0.001 < target_A:
                        next_boundary = min(next_boundary, limit)
                    if curr_A < -limit - 0.001 < target_A:
                        next_boundary = min(next_boundary, -limit)
                else:
                    if curr_A > limit + 0.001 > target_A:
                        next_boundary = max(next_boundary, limit)
                    if curr_A > -limit + 0.001 > target_A:
                        next_boundary = max(next_boundary, -limit)

            driver.set_ramp_rate(rate)
            driver.set_current_setpoint(next_boundary)
            self._ramp_setpoint_T = next_boundary / self._amperes_per_tesla
            yield

    # ------------------------------------------------------------------
    # Segment rate look-up
    # ------------------------------------------------------------------

    def _get_segment_rate(self, current_A: float, direction: int = 0) -> float:
        """Return ramp rate (A/min) for the given current magnitude.

        Args:
            current_A: Current operating current in amperes.
            direction: 1 if ramping up, -1 if down, 0 for static.
        """
        abs_I = abs(current_A + direction * 0.002)
        for segment in self._ramp_segments:
            if abs_I <= segment["max_current_A"]:
                return float(segment["rate_A_per_min"])
        return self._default_ramp_rate

    # ------------------------------------------------------------------
    # @monitored methods
    # ------------------------------------------------------------------

    @monitored(unit="A", description="Power-supply output current")
    def psu_current(self) -> float:
        """Return the PSU output current in amperes."""
        return self._driver.get_current()  # type: ignore[attr-defined]

    @monitored(
        unit="A",
        description="Current holding the field in the magnet coil",
    )
    def magnet_current(self) -> float:
        """Return the field-holding current in amperes.

        Equals the PSU output: this VI keeps the switch heater on throughout.
        """
        return self._driver.get_current()  # type: ignore[attr-defined]

    @monitored(unit="T", description="Magnetic field at the sample")
    def magnet_field_T(self) -> float:
        """Return the current magnetic field in tesla."""
        return self._driver.get_current() / self._amperes_per_tesla  # type: ignore[attr-defined]

    # Dimensionless: the PSU's own status word, not a measured quantity.
    @monitored(
        unit="",
        description=(
            "Power-supply status word as reported by the hardware: HOLD, "
            "RAMPING, QUENCH or CLAMPED"
        ),
    )
    def magnet_status(self) -> str:
        """Return the PSU status string (HOLD, RAMPING, QUENCH, or CLAMPED).

        The raw hardware report, read straight from the driver. Distinct
        from ``magnet_state()``, which is this VI's logical interpretation
        of that report plus the live current readings.
        """
        return self._driver.get_status()  # type: ignore[attr-defined]

    # Dimensionless: this VI's logical interpretation of the hardware
    # report plus the live current readings (see GLOSSARY's Magnet state).
    @monitored(
        unit="",
        description=(
            "Logical magnet state: standby, ramping, holding, quenched or "
            "clamped"
        ),
    )
    def magnet_state(self) -> str:
        """Return the logical magnet state.

        Returns:
            One of: "standby" (PSU ≈ 0 A), "ramping" (PSU status RAMPING, or
            a ramp generator not yet exhausted), "holding" (at target),
            "quenched" (safety condition), "clamped" (compliance — no driver
            in this codebase reports CLAMPED today; kept for a future one
            that does).
        """
        psu_status = self.magnet_status()

        if psu_status == "QUENCH":
            return "quenched"
        if psu_status == "CLAMPED":
            return "clamped"

        # PSU status is the source of truth for "at rest": a ramp generator
        # not yet exhausted (ramp_status() == "RAMPING") is OR'd in so a
        # tick where the PSU happens to read HOLD mid-ramp (e.g. between
        # ramp segments, or during a wait) doesn't get misclassified as
        # standby/holding.
        if self.ramp_status() == "RAMPING" or psu_status not in ("HOLD", "CLAMPED"):
            return "ramping"

        psu_A = self.psu_current()
        if abs(psu_A) <= 0.01:
            return "standby"

        return "holding"

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def evaluate_safety(self, state: dict) -> dict[str, bool]:
        """Flag a quench from the already-polled magnet status (no hardware poll)."""
        return {"quench": state.get("magnet_status") == "QUENCH"}

    # ------------------------------------------------------------------
    # @control methods
    # ------------------------------------------------------------------

    # The bound on target_T is NOT declared here: it is a setup property
    # read from the config through control_limits (see the
    # control-validation standard). The spec's default is the form seed —
    # zero field, the same target standby() drives to.
    @control(
        action_class="run_control",
        params={
            "target_T": ParamSpec(
                type=float,
                default=0.0,
                unit="T",
                description="Field to ramp the magnet to",
            ),
        },
    )
    def set_field(self, target_T: float) -> None:
        """Manually command a field ramp (GUI use; blocked during procedures).

        Args:
            target_T: Desired field in tesla.
        """
        self.start_ramp(target_T)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initiate(self) -> None:
        """Put the PSU in HOLD mode on startup."""

    def standby(self) -> None:
        """Ramp to zero field and hold."""
        self.start_ramp(0.0)
