"""Simulated XY sample stage driver — two axes with finite speed and travel."""

from __future__ import annotations

import logging
import time

from i2as.core.exceptions import (
    I2ASCommunicationError,
    I2ASInstrumentError,
)

log = logging.getLogger(__name__)

#: The two axes, in the order ``get_position()`` reports them.
_AXES = ("x", "y")


class SimXYStage:
    """Simulated two-axis sample stage.

    Each axis moves toward its own target at the common speed, advancing with
    wall-clock time exactly like the other sims (``SimOxfordIPS120``'s
    ramp), so a move is observed through repeated reads rather than through
    a tick call. The controller models what a real stage controller does at
    the bus:

    * ``move_to()`` takes each axis independently — an omitted axis keeps
      the target it already has — because a two-axis controller accepts one
      move command per axis and two callers (one per axis) must never cancel
      each other's moves.
    * A target beyond the travel limits is refused with the controller's own
      code, and nothing moves (the instrument-error half of the driver
      error-reporting standard).
    * ``stop()`` halts an axis where it is and pins its target there.

    This driver satisfies the three-rule driver contract:
    1. It is a Python class.
    2. __init__ accepts a single VISA resource string (ignored for simulation).
    3. It is importable via i2as.drivers.sim_xy_stage.
    """

    #: Travel of each axis, in metres (a 25 mm stage centred on zero).
    TRAVEL_MIN_M = -12.5e-3
    TRAVEL_MAX_M = 12.5e-3

    #: The speed range the controller accepts, in metres per second.
    MIN_SPEED_M_PER_S = 1e-6
    MAX_SPEED_M_PER_S = 50e-3

    def __init__(self, resource_string: str) -> None:
        """Initialise the simulated stage at the origin, at rest.

        Args:
            resource_string: VISA address (e.g. 'ASRL5::INSTR'). Ignored.
        """
        _ = resource_string  # Explicitly ignored per driver contract

        self._position: dict[str, float] = {axis: 0.0 for axis in _AXES}
        self._target: dict[str, float] = {axis: 0.0 for axis in _AXES}
        self._speed_m_per_s: float = 1e-3
        self._last_update: float = time.time()

        # Test control flags
        self._simulate_error: bool = False
        # Connection-lifecycle standard: True once close() has released
        # the session; every command then fails (see _check_error).
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def move_to(self, x_m: float | None = None, y_m: float | None = None) -> None:
        """Command one or both axes to move to a new position.

        Verified by explicit range check before anything moves: either
        target outside the travel is refused and NEITHER axis is commanded,
        so a caller never ends up with half a move.

        Args:
            x_m: New x target in metres, or ``None`` to leave x alone.
            y_m: New y target in metres, or ``None`` to leave y alone.

        Raises:
            I2ASInstrumentError: ``TRAVEL_LIMIT`` if a target lies
                outside ``[TRAVEL_MIN_M, TRAVEL_MAX_M]``.
        """
        self._check_error()
        requested = {"x": x_m, "y": y_m}
        for axis, value in requested.items():
            if value is None:
                continue
            if not self.TRAVEL_MIN_M <= float(value) <= self.TRAVEL_MAX_M:
                self._refuse(
                    f"move_to({axis}_m={value!r})",
                    "TRAVEL_LIMIT",
                    f"{axis} target must be within [{self.TRAVEL_MIN_M:g}, "
                    f"{self.TRAVEL_MAX_M:g}] m",
                )
        self._update_simulation()
        for axis, value in requested.items():
            if value is not None:
                self._target[axis] = float(value)

    def get_position(self) -> tuple[float, float]:
        """Return the present ``(x_m, y_m)`` position in metres."""
        self._check_error()
        self._update_simulation()
        return (self._position["x"], self._position["y"])

    def get_target(self) -> tuple[float, float]:
        """Return the ``(x_m, y_m)`` targets the axes are moving to."""
        self._check_error()
        return (self._target["x"], self._target["y"])

    def set_speed(self, speed_m_per_s: float) -> None:
        """Set the speed both axes move at.

        Args:
            speed_m_per_s: Speed in metres per second.

        Raises:
            I2ASInstrumentError: ``SPEED_RANGE`` if the value is outside
                the controller's range; the speed is left unchanged.
        """
        self._check_error()
        value = float(speed_m_per_s)
        if not self.MIN_SPEED_M_PER_S <= value <= self.MAX_SPEED_M_PER_S:
            self._refuse(
                f"set_speed({speed_m_per_s!r})",
                "SPEED_RANGE",
                f"speed must be within [{self.MIN_SPEED_M_PER_S:g}, "
                f"{self.MAX_SPEED_M_PER_S:g}] m/s",
            )
        self._update_simulation()
        self._speed_m_per_s = value

    def get_speed(self) -> float:
        """Return the speed in metres per second."""
        self._check_error()
        return self._speed_m_per_s

    def is_moving(self, axis: str | None = None) -> bool:
        """Return True while an axis is still short of its target.

        Args:
            axis: ``"x"`` or ``"y"`` to ask about one axis; ``None`` asks
                whether either is moving.

        Returns:
            Whether the axis (or any axis) is in motion.
        """
        self._check_error()
        self._update_simulation()
        axes = _AXES if axis is None else (self._axis(axis),)
        return any(self._position[a] != self._target[a] for a in axes)

    def stop(self, axis: str | None = None) -> None:
        """Halt an axis where it is, pinning its target to its position.

        Args:
            axis: ``"x"`` or ``"y"`` to stop one axis; ``None`` stops both.
        """
        self._check_error()
        self._update_simulation()
        for a in (_AXES if axis is None else (self._axis(axis),)):
            self._target[a] = self._position[a]

    def get_idn(self) -> str:
        """Return simulated identification string."""
        self._check_error()
        return "I2AS,SIMXYSTAGE,SIM,1.0"

    # ------------------------------------------------------------------
    # Internal simulation logic
    # ------------------------------------------------------------------

    def _axis(self, axis: str) -> str:
        """Validate an axis name.

        Args:
            axis: The axis name to check.

        Returns:
            The same name, lower-cased.

        Raises:
            ValueError: If it is not ``"x"`` or ``"y"`` (a programming
                error, caught before anything reaches the bus).
        """
        name = str(axis).lower()
        if name not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")
        return name

    def _update_simulation(self) -> None:
        """Advance every axis toward its target based on elapsed real time."""
        now = time.time()
        step = self._speed_m_per_s * (now - self._last_update)
        self._last_update = now
        for axis in _AXES:
            remaining = self._target[axis] - self._position[axis]
            if abs(remaining) <= step:
                self._position[axis] = self._target[axis]
            else:
                self._position[axis] += step if remaining > 0 else -step

    # ------------------------------------------------------------------
    # Safe state (the safe-shutdown standard)
    # ------------------------------------------------------------------

    def safe_shutdown(self) -> None:
        """Stop both axes where they are; idempotent, never raises.

        Safe idle for a sample stage is *standing still*, not the origin: a
        drive home is a move like any other and can collide with whatever an
        operator has in the way. The speed setting is left alone.
        """
        log.info("SimXYStage: safe shutdown — both axes stopped.")
        self._update_simulation()
        for axis in _AXES:
            self._target[axis] = self._position[axis]

    def _is_in_safe_state(self) -> bool:
        """Return True when no axis has a target away from its position."""
        return all(self._target[a] == self._position[a] for a in _AXES)

    # ------------------------------------------------------------------
    # Connection lifecycle (the connection-lifecycle standard)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the simulated session; the stage is left untouched.

        Idempotent and never raises. Afterwards every command — including
        ``get_idn()`` — raises ``I2ASCommunicationError`` via
        :meth:`_check_error`, modelling a released session so a
        use-after-disconnect bug fails in a test instead of on hardware.
        """
        self._closed = True

    def _refuse(self, context: str, code: str, reason: str) -> None:
        """Raise the typed refusal a real stage controller would report.

        Args:
            context: The driver call that was refused.
            code: The controller's own error code.
            reason: Why, in the controller's words.

        Raises:
            I2ASInstrumentError: Always.
        """
        raise I2ASInstrumentError(
            f"Simulated XY stage refused {context}: {code} — {reason}",
            code=code,
            instrument_message=f"{code}: {reason}",
            context=context,
            vi_name="SimXYStage",
        )

    def _check_error(self) -> None:
        """Raise I2ASCommunicationError if error simulation is active."""
        if self._closed:
            raise I2ASCommunicationError(
                "SimXYStage: the session is closed — the driver was "
                "disconnected from I2AS",
                vi_name="SimXYStage",
            )
        if self._simulate_error:
            raise I2ASCommunicationError(
                "Simulated communication error on XY stage",
                vi_name="SimXYStage",
            )
