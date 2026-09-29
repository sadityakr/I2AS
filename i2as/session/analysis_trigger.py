"""The analysis trigger — whether a finished run is analysed, decided by analysis alone.

A finished run is handed to the analysis runner when the analysis section of
the general settings file says so, an experiment is open to hold the bundle,
and the run left a data file to read. That is the whole decision, and it
consults no notebook: whether the resulting bundle is ever published is the
publishing layer's business, taken later and separately.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from PyQt6.QtCore import QObject

from i2as.session.app_config import AnalysisSettings

logger = logging.getLogger(__name__)


class AnalysisTrigger(QObject):
    """Starts the automatic analysis of each finished run, when switched on.

    Connect ``on_run_finished`` to the engine's ``run_finished`` signal AFTER
    the ``ExperimentManager`` is connected, so the run is recorded before it is
    analysed.

    A ``QObject`` parented to the runner (when the runner is one) because Qt
    holds a slot's owner only weakly: a trigger nobody else referenced would
    be collected and its connection silently dropped.
    """

    def __init__(
        self,
        manager: Any,
        runner: Any,
        settings_source: Callable[[], AnalysisSettings],
        parent: QObject | None = None,
    ) -> None:
        """Wire the trigger.

        Args:
            manager: The ``ExperimentManager`` (the open experiment and the
                store the data path is resolved through).
            runner: The ``AnalysisRunner``, duck-typed on ``start()``.
            settings_source: Returns the current ``AnalysisSettings``.
            parent: The Qt owner; ``None`` uses the runner when it is a
                ``QObject``.
        """
        super().__init__(parent if parent is not None else (runner if isinstance(runner, QObject) else None))
        self._manager = manager
        self._runner = runner
        self._settings_source = settings_source

    def on_run_finished(self, event: Any) -> str:
        """Analyse one finished run if analysis is on. Never raises.

        Args:
            event: The run manifest dict the engine emits, or a
                ``RunFinished`` event carrying it.

        Returns:
            The bundle folder the analysis will write, or ``""`` when the run
            is not analysed (analysis off, no experiment, no data, a failed
            run, or the runner refused).
        """
        manifest = getattr(event, "manifest", event)
        if not isinstance(manifest, Mapping):
            return ""
        run_id = str(manifest.get("run_id", ""))
        try:
            settings = self._settings_source()
        except Exception:  # noqa: BLE001 - a settings failure must not reach the engine's signal
            logger.exception("Could not read the analysis settings")
            return ""
        if not (run_id and settings.enabled):
            return ""
        experiment = self._manager.current_experiment()
        if experiment is None:
            logger.debug("Run %r belongs to no experiment — not analysed", run_id)
            return ""
        data_file = str(manifest.get("data_file") or "")
        data_path = (
            str(self._manager.store.resolve_data_file(experiment.experiment_id, data_file))
            if data_file
            else ""
        )
        try:
            return str(self._runner.start(run_id, dict(manifest), data_path) or "")
        except Exception:  # noqa: BLE001 - never raise into the engine's signal
            logger.exception("Could not start the analysis of run %s", run_id)
            return ""
