"""StageAxisVI — one positioned axis of a sample stage, ramped like a setpoint."""

from __future__ import annotations

import logging
from typing import Any, Generator

from i2as.core.decorators import control, monitored
from i2as.core.exceptions import I2ASConfigError
from i2as.core.plan import ParamSpec
from i2as.virtual_instruments.base import StageBase
from i2as.virtual_instruments.rampable import RampableVI

logger = logging.getLogger(__name__)

#: The axes a two-axis stage driver positions.
_AXES = ("x", "y")


class StageAxisVI(StageBase, RampableVI):
    """Virtual Instrument for ONE axis of a sample stage.

    A stage move is a ramp: ``start_ramp(target_m)`` commands the axis to
    its new position, the driver moves it at the configured speed, and
    ``ramp_status()`` reports ``TARGET_REACHED`` once the axis is within
    ``tolerance_m`` of the target — so the Orchestrator waits for a move
    exactly as it waits for a field, the Ramps panel shows it, and the stall
    detector watches it, with no new machinery. The setpoint is one scalar
    per VI (the setpoint-parameter convention: ``set_position(target_m)``),
    which is why a two-axis stage is two of these over one driver, each
    declaring its ``axis`` in the config.

    Driver contract
    ---------------
    The ``"main"`` driver must implement:
    * ``move_to(x_m=None, y_m=None)`` — command a move; an omitted axis keeps
      its own move, so two axis VIs sharing one driver never cancel each
      other.
    * ``get_position() -> (x_m, y_m)``
    * ``is_moving(axis) -> bool``
    * ``stop(axis)`` — halt one axis where it is.
    * ``set_speed(speed_m_per_s)`` — applied once, at ``initiate()``.
    """

    # Control-validation standard (see BaseVirtualInstrument): the position
    # a user commands is bounded by the setup's travel range for this axis,
    # read in __init__ from min_position_m / max_position_m.
    control_limits = {"set_position": {"target_m": "position_m"}}

    def __init__(self, drivers: dict[str, object], **init_params: Any) -> None:
        """Bind one axis of the stage driver and read its setup limits.

        Args:
            drivers: ``{"main": <stage driver>}``.
            **init_params: ``axis`` (``"x"`` or ``"y"``, default ``"x"``);
                ``min_position_m`` / ``max_position_m`` — the travel this
                setup allows on the axis (absent means unbounded on that
                side, the same rule every other VI's limits follow);
                ``speed_m_per_s`` — the speed ``initiate()`` sets (default
                1e-3); ``tolerance_m`` — how close counts as arrived
                (default 1e-6).

        Raises:
            I2ASConfigError: If ``axis`` is not one of the driver's axes.
        """
        super().__init__(drivers, **init_params)
        self._driver = drivers["main"]

        axis = str(init_params.get("axis", "x")).lower()
        if axis not in _AXES:
            raise I2ASConfigError(
                f"{type(self).__name__}: axis must be one of {_AXES}, got {axis!r}"
            )
        self.axis = axis
        # The label the status line and the ramp tracker show for this
        # VI's setpoint: which axis, not just "position".
        self.setpoint_label = f"{axis} position"

        self._speed_m_per_s: float = float(init_params.get("speed_m_per_s", 1e-3))
        self._tolerance_m: float = float(init_params.get("tolerance_m", 1e-6))

        lo = init_params.get("min_position_m")
        hi = init_params.get("max_position_m")
        self._limits["position_m"] = (
            float(lo) if lo is not None else None,
            float(hi) if hi is not None else None,
        )

        self._ramp_gen: Generator | None = None
        self._ramp_exhausted: bool = True
        self._ramp_target_m: float | None = None
        #: The position last commanded to the driver — the "next setpoint"
        #: of the ramp-introspection standard. A move is commanded in one
        #: shot, so it equals the target for the whole ramp.
        self._ramp_setpoint_m: float | None = None

    # ------------------------------------------------------------------
    # RampableVI implementation
    # ------------------------------------------------------------------

    def start_ramp(self, target: float) -> None:
        """Command the axis to move to *target* metres.

        Args:
            target: Target position in metres.
        """
        target_m = self._clamp_target_m(float(target))
        self._ramp_target_m = target_m
        self._ramp_setpoint_m = None
        self._ramp_gen = self._ramp_generator(target_m)
        self._ramp_exhausted = False
        try:
            next(self._ramp_gen)
        except StopIteration:
            self._ramp_exhausted = True

    def _clamp_target_m(self, target_m: float) -> float:
        """Clamp a target to the setup's travel, loudly.

        Last-resort protection for programmatic callers (procedures);
        ``set_position()`` already rejects loudly via the control-limits
        wrapper, so a clamp here only ever hides a bug — hence the warning.

        Args:
            target_m: The requested position in metres.

        Returns:
            The position within the configured travel.
        """
        lo, hi = self._limits["position_m"]
        clamped = target_m
        if lo is not None:
            clamped = max(lo, clamped)
        if hi is not None:
            clamped = min(hi, clamped)
        if clamped != target_m:
            logger.warning(
                "%s: requested %.4g m exceeds the %s travel [%s, %s] — clamped to %.4g m",
                self.vi_name or type(self).__name__,
                target_m, self.axis, lo, hi, clamped,
            )
        return clamped

    def advance_ramp(self) -> None:
        """Advance the ramp generator by one tick."""
        if self._ramp_gen is None or self._ramp_exhausted:
            return
        try:
            next(self._ramp_gen)
        except StopIteration:
            self._ramp_exhausted = True

    def ramp_status(self) -> str:
        """Return the current ramp state.

        Returns:
            ``"IDLE"`` — no move active; ``"RAMPING"`` — the axis is still
            short of its target; ``"TARGET_REACHED"`` — within tolerance.
        """
        if self._ramp_gen is None:
            return "IDLE"
        if not self._ramp_exhausted:
            return "RAMPING"
        if self._ramp_target_m is None:
            return "IDLE"
        if abs(self.position() - self._ramp_target_m) <= self._tolerance_m:
            return "TARGET_REACHED"
        return "RAMPING"

    def stop_ramp(self) -> None:
        """Stop the move: kill the generator AND halt the axis where it is.

        The driver keeps moving to its last target on its own, so the axis
        is told to stop; its target is then wherever it is.
        """
        self._ramp_gen = None
        self._ramp_exhausted = True
        self._ramp_target_m = None
        self._ramp_setpoint_m = None
        self._driver.stop(self.axis)  # type: ignore[attr-defined]

    def ramp_target(self) -> float | None:
        """Return the active target position in metres, or ``None`` when idle."""
        return self._ramp_target_m

    def ramp_setpoint(self) -> float | None:
        """Return the position last commanded to the driver, in metres.

        A move is commanded in one shot, so this equals ``ramp_target()``
        from the first tick of the ramp; recorded by the generator, never
        read back from the hardware.
        """
        return self._ramp_setpoint_m

    def ramp_rate(self) -> float | None:
        """Return the speed of the active move in metres per minute, or ``None``."""
        if self._ramp_target_m is None:
            return None
        return self._speed_m_per_s * 60.0

    def nominal_ramp_rate(self) -> float | None:
        """Return the configured speed in metres per minute.

        The declared rate a **duration estimate** is built from: read from
        config alone, no bus traffic, no active move.
        """
        return self._speed_m_per_s * 60.0

    def ramp_value(self) -> float | None:
        """Return the present position in metres (the value the move drives)."""
        return self.position()

    def _ramp_generator(self, target_m: float) -> Generator:
        driver = self._driver  # type: ignore[attr-defined]
        driver.move_to(**{f"{self.axis}_m": target_m})
        self._ramp_setpoint_m = target_m
        while driver.is_moving(self.axis):
            yield

    # ------------------------------------------------------------------
    # @monitored methods
    # ------------------------------------------------------------------

    @monitored(unit="m", description="Position of this axis of the sample stage")
    def position(self) -> float:
        """Return the axis position in metres."""
        x_m, y_m = self._driver.get_position()  # type: ignore[attr-defined]
        return float(x_m if self.axis == "x" else y_m)

    # Dimensionless: a motion state word, not a measured quantity.
    @monitored(unit="", description="Whether this axis is moving or holding")
    def motion_state(self) -> str:
        """Return ``"moving"`` while the axis is short of its target, else ``"holding"``."""
        moving = self._driver.is_moving(self.axis)  # type: ignore[attr-defined]
        return "moving" if moving else "holding"

    # ------------------------------------------------------------------
    # @control methods
    # ------------------------------------------------------------------

    # The bound on target_m is NOT declared here: it is a setup property
    # read from the config through control_limits (the control-validation
    # standard). The spec's default is the form seed — the origin.
    @control(
        action_class="run_control",
        params={
            "target_m": ParamSpec(
                type=float,
                default=0.0,
                unit="m",
                description="Position to move this axis to",
            ),
        },
    )
    def set_position(self, target_m: float) -> None:
        """Manually command a move (GUI use; blocked during procedures).

        Args:
            target_m: Desired position in metres.
        """
        self.start_ramp(target_m)

    # A stop halts the axis where it is: it takes the stage OUT of motion,
    # which is what makes it the recovery-class action an observer may take.
    @control(action_class="recovery")
    def stop(self) -> None:
        """Halt this axis where it is and cancel its move."""
        self.stop_ramp()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initiate(self) -> None:
        """Apply the configured speed — the one setup command a stage needs."""
        self._driver.set_speed(self._speed_m_per_s)  # type: ignore[attr-defined]

    def standby(self) -> None:
        """Stop where it is.

        Safe idle for a sample stage is standing still, not the origin: a
        drive home is a move like any other and can collide with whatever
        the operator has in the way.
        """
        self.stop_ramp()
