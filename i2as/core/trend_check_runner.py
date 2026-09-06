"""trend_check_runner — schedules the trend-check standard on its own timer.

The Orchestrator is the state machine and the sole writer to hardware; a
trend check writes no hardware and touches no state machine, so it does not
belong there, and it does not need to: `Orchestrator._update_operational_status()`
already builds its per-tick record from `Station.conditions()`, so anything
that publishes into that registry reaches `status.jsonl` on the next tick
with the Orchestrator knowing nothing about it. This module is the ONE
Qt-aware piece the trend-check standard needs — a small, single-purpose
scheduler that lets the trend layer run itself on a slow cadence without
enlarging the Orchestrator. It holds a `Station` reference and nothing else
from the running application (no Orchestrator, no GUI).

Input: a `Station` to publish into, the `TrendCheck` declarations to
evaluate (see `i2as.core.trend_checks.declared_checks()`), and a refresh
interval in seconds. Process: on each timer firing, resolves the trend-history
log directory, evaluates every declared check against it (pure, via
`trend_checks.run_checks()`), and turns any failing check into an
advisory-severity `Condition`. Output: publishes those conditions into the
Station's unified condition registry via the public, origin-scoped
`Station.publish_conditions()`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from PyQt6.QtCore import QObject, QTimer

from i2as.core.paths import log_directory
from i2as.core.station import Station
from i2as.core.trend_checks import TrendCheck, conditions_for, run_checks

logger = logging.getLogger(__name__)


class TrendCheckRunner(QObject):
    """Owns the slow timer that evaluates and publishes trend checks.

    Trend-check windows are hours, so re-evaluating them every tick would be
    wasted file I/O on the one path this architecture forbids blocking work
    on. This timer runs independently of the Orchestrator's tick timer, at a
    much slower default cadence (60 s), mirroring the primary timer's own
    construction idiom (`QTimer(self)`, `setInterval`, `timeout.connect`,
    `start()`).

    Args:
        station: The `Station` to evaluate state from and publish
            trend-origin conditions into. Never an `Orchestrator` reference
            — this scheduler has no need to know orchestrator state.
        checks: The declared `TrendCheck`s to evaluate on every firing (see
            `i2as.core.trend_checks.declared_checks()`).
        refresh_interval_s: Seconds between evaluations. Floored to at least
            1 ms so a misconfigured non-positive value cannot spin the timer.
    """

    def __init__(
        self,
        station: Station,
        checks: Sequence[TrendCheck],
        *,
        refresh_interval_s: float = 60.0,
    ) -> None:
        super().__init__()
        self._station = station
        self._checks = tuple(checks)

        self._timer = QTimer(self)
        self._timer.setInterval(max(1, round(refresh_interval_s * 1000.0)))
        self._timer.timeout.connect(self._run)
        self._timer.start()

    def stop(self) -> None:
        """Stop the refresh timer (app exit / test teardown)."""
        self._timer.stop()

    def run_once(self) -> None:
        """Evaluate and publish every declared check immediately.

        The same body the timer calls on each firing, exposed directly so
        tests do not have to wait on a real `QTimer` interval.
        """
        self._run()

    def _run(self) -> None:
        """Evaluate this setup's declared trend checks and publish their verdicts.

        Guarded like the Orchestrator's own non-critical reporting paths: a
        failure here is a reporting problem, never a reason to disrupt a
        running measurement (which this scheduler cannot touch in the first
        place — it never calls into the Orchestrator or any driver/VI).
        """
        if not self._checks:
            return
        try:
            log_dir = log_directory()
            results = run_checks(self._checks, log_dir)
            conditions = conditions_for(self._checks, results, time.time())
            self._station.publish_conditions("trend", conditions)
        except Exception:
            logger.exception("trend-check refresh failed (non-fatal)")
