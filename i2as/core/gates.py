"""Gate: the tick-driven wait primitive behind procedure initiation/reading gates."""

from __future__ import annotations

import time
from collections.abc import Callable

__all__ = ["Gate"]


class Gate:
    """A one-shot action optionally followed by a windowed stability check.

    Four shapes, selected by which constructor arguments are given:

    - Neither ``action`` nor ``check``: satisfied on the first ``step()``.
      Not a realistic use on its own, but not special-cased.
    - ``action`` only: runs the callable once and is satisfied on that same
      ``step()`` call.
    - ``check`` only: satisfied once ``check()`` has returned True for
      ``window_s`` seconds of continuous polling (a pure stability wait,
      e.g. "temperature within tolerance for the last 5 minutes").
    - Both: runs ``action`` once, then starts the stability clock — the
      action's effect only counts once observed via ``check``, so an
      orchestration-level command (e.g. "settle the reading path on this
      setting") can be followed by a wait for its effect to be confirmed.

    Attributes:
        name: Human-readable identifier, surfaced in operational-status
            reporting (``active_gates``) so a stuck run can be diagnosed by
            which gate it is waiting on.
    """

    def __init__(
        self,
        name: str,
        check: Callable[[], bool] | None = None,
        window_s: float = 0.0,
        action: Callable[[], None] | None = None,
    ) -> None:
        """Build a Gate.

        Args:
            name: Non-empty identifier for this gate.
            check: Optional zero-arg predicate polled once per ``step()``.
            window_s: Seconds ``check()`` must hold True continuously
                before the gate reports satisfied. Ignored if ``check`` is
                None. Must be non-negative.
            action: Optional zero-arg callable run exactly once, on the
                first ``step()`` call.

        Raises:
            ValueError: If ``name`` is empty or ``window_s`` is negative.
        """
        if not name:
            raise ValueError("Gate name must be non-empty")
        if window_s < 0:
            raise ValueError(f"Gate window_s must be non-negative, got {window_s}")

        self.name = name
        self._check = check
        self._window_s = float(window_s)
        self._action = action

        self._action_done = False
        self._stable_since: float | None = None

    def step(self) -> bool:
        """Advance the gate by one Orchestrator tick.

        Returns:
            True once the gate is satisfied (and on every subsequent call).
        """
        if not self._action_done:
            if self._action is not None:
                self._action()
            self._action_done = True
            if self._check is None:
                return True

        if self._check is None:
            return True

        if not self._check():
            self._stable_since = None
            return False

        now = time.time()
        if self._stable_since is None:
            self._stable_since = now
        return now - self._stable_since >= self._window_s

    def check_once(self) -> bool:
        """Evaluate this gate a single time, ignoring ``window_s`` entirely.

        Runs ``action`` once (if not already run) exactly like ``step()``,
        then reads ``check()`` a single time and returns it verbatim — no
        stability window, no holding, no timeout. For a caller that needs a
        gate's verdict at one instant rather than a wait. Call this instead
        of (never together with) ``step()`` on a given ``Gate`` instance.

        Returns:
            ``True`` if there is no ``check`` (action-only gate), else the
            single current ``check()`` reading.
        """
        if not self._action_done:
            if self._action is not None:
                self._action()
            self._action_done = True
        if self._check is None:
            return True
        return bool(self._check())
