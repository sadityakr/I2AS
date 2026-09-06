"""The analysis runner — one finished run, one worker process, one report.

**Recipes run in a separate process, never on a thread of this application.**
A recipe is user code: it may take seconds, it may import matplotlib, and it
may crash. The single hardware thread standard already puts every network
call and every analysis on the client side of the control contract; a
subprocess is the same idea one step further — the worker cannot reach an
instrument because it cannot import one, and a recipe that hangs costs a
timer, not the event loop.

Nothing here blocks. ``start()`` writes an ``AnalysisSpec`` into the run's
report directory and launches ``python -m i2as.analysis run --spec
<file>`` with ``QProcess``; the answer arrives later on ``finished``, bounded
by a ``QTimer`` from the settings' ``timeout_s``. One worker runs at a time
and further requests wait in a FIFO queue, for the same reason the **Outbox**
drains one job per firing: a slow analysis delays the next analysis, never the
GUI's next turn.

**Every ending produces an entry.** A report that ran becomes an **analysed
entry** through ``ElnPublisher.export_report()``; a recipe that raised, a
worker that timed out, one that never wrote a report, one that was cancelled
— each becomes a `failed` report, and the publisher parks the facts-only
entry instead (``park_facts_entry()``), carrying the reason. A run is never
silently left with nothing waiting for its human.

Both hand-offs PARK; neither publishes. Approval stays exactly where it was,
``ExperimentManager.approve_eln_draft()``.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QProcess, QTimer, pyqtSignal

from i2as.analysis.report import (
    REPORT_FAILED,
    REPORT_FILENAME,
    SPEC_FILENAME,
    AnalysisReport,
    AnalysisSpec,
)
from i2as.session.eln.drafting import manifest_from_run
from i2as.session.eln.settings import ElnSettings
from i2as.session.manager import ExperimentManager

logger = logging.getLogger(__name__)

#: Longest tail of the worker's stderr kept in a synthesized failure, so an
#: unreadable traceback still says what went wrong without carrying a
#: megabyte of output into a notebook entry.
_STDERR_TAIL_CHARS = 4000


@dataclass(frozen=True)
class _Request:
    """One queued analysis, already written to disk as a spec.

    Attributes:
        run_id: The run being analysed.
        recipe: The recipe name the spec names, or ``""`` for discovery's
            choice — carried so a synthesized failure can still say it.
        spec_path: The ``spec.json`` the worker is started with.
        output_dir: Where the worker writes its report and figures.
    """

    run_id: str
    recipe: str
    spec_path: Path
    output_dir: Path


class AnalysisRunner(QObject):
    """Runs one analysis worker at a time and hands its report to the publisher.

    Signals:
        analysis_started (str): The run id, when its worker actually starts.
        analysis_finished (str, dict): The run id and the report as its JSON
            dict, when a recipe ran to completion.
        analysis_failed (str, str): The run id and the failure text, for every
            other ending — a raising recipe, a timeout, a missing report, a
            cancellation.
    """

    analysis_started = pyqtSignal(str)
    analysis_finished = pyqtSignal(str, dict)
    analysis_failed = pyqtSignal(str, str)

    def __init__(
        self,
        manager: ExperimentManager,
        publisher: Any,
        settings_source: Callable[[], ElnSettings],
        python: str = sys.executable,
        parent: QObject | None = None,
    ) -> None:
        """Wire the runner to the session layer and the publisher.

        Args:
            manager: The session-layer façade — the open experiment, the run
                records, the store paths and the experiment context a spec is
                built from.
            publisher: The ELN publisher, duck-typed on
                ``export_report(run_id, report, report_dir)`` and
                ``park_facts_entry(run_id, warning)``. Never imported as a
                type here, so a test can hand in a stand-in.
            settings_source: Called for the current ``ElnSettings`` at the
                moment each analysis starts, so a settings change reaches the
                next run without re-wiring anything.
            python: The interpreter the worker is started with. Defaults to
                the running one, which is what makes the worker see the same
                installed I2AS.
            parent: Qt parent, if any.
        """
        super().__init__(parent)
        self._manager = manager
        self._publisher = publisher
        self._settings_source = settings_source
        self._python = python
        self._queue: list[_Request] = []
        self._active: _Request | None = None
        self._process: QProcess | None = None
        self._failure: str = ""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

    # ------------------------------------------------------------------
    # Read surface
    # ------------------------------------------------------------------

    def is_running(self, run_id: str = "") -> bool:
        """Whether an analysis is in flight.

        Args:
            run_id: Ask about one run. ``""`` asks about any.

        Returns:
            ``True`` while a worker is running or a request is queued behind
            one — from the caller's side both mean "the answer is not here
            yet".
        """
        in_flight = [request.run_id for request in self._pending()]
        return bool(run_id in in_flight if run_id else in_flight)

    def recipe_dirs(self) -> list[str]:
        """Return the extra recipe directories discovery should search.

        Returns:
            The open experiment's own recipes folder as a one-element list,
            or ``[]`` when no experiment is open. The folder need not exist —
            discovery tolerates a missing directory.
        """
        experiment = self._manager.current_experiment()
        if experiment is None:
            return []
        return [str(self._manager.store.recipes_dir(experiment.experiment_id))]

    # ------------------------------------------------------------------
    # Starting
    # ------------------------------------------------------------------

    def start(
        self,
        run_id: str,
        manifest: Mapping[str, Any] | None = None,
        data_path: str = "",
        recipe: str = "",
        options: Mapping[str, Any] | None = None,
    ) -> str:
        """Analyse one recorded run, later. Never blocks and never raises.

        Builds the spec from the record and the experiment context, writes it
        into the run's report directory, and either launches the worker or
        queues the request behind the one already running.

        Args:
            run_id: The run to analyse, in the open experiment.
            manifest: The Orchestrator's run manifest, when the caller has
                it. Merged OVER the record's own facts, because it is the
                fresher description of the same run.
            data_path: Absolute path of the run's data file. ``""`` resolves
                it from the record through the store.
            recipe: The recipe ``name`` to run. ``""`` falls back to the
                settings' per-procedure preference, then to discovery.
            options: Free-form recipe options, passed through to the report.

        Returns:
            The report directory as a string, or ``""`` when the analysis
            could not be started: no experiment open, no such run, or no data
            file to read (all logged, never raised).
        """
        experiment = self._manager.current_experiment()
        if experiment is None:
            logger.warning("No experiment is open — run %r cannot be analysed", run_id)
            return ""
        run = experiment.find_run(run_id) if run_id else None
        if run is None:
            logger.warning("No recorded run %r to analyse", run_id)
            return ""
        resolved = data_path or self._data_path(experiment.experiment_id, run)
        if not resolved:
            logger.warning("Run %s has no data file to analyse", run_id)
            return ""

        settings = self._settings_source()
        facts = {**manifest_from_run(run), **dict(manifest or {})}
        chosen = recipe or settings.analysis.recipes.get(str(facts.get("procedure", "")), "")
        output_dir = self._manager.store.report_dir(experiment.experiment_id, run_id)
        spec = AnalysisSpec(
            run_id=run_id,
            data_path=resolved,
            manifest=facts,
            experiment=self._experiment_facts(experiment),
            setup=dict(self._manager.experiment_context().get("setup") or {}),
            recipe=chosen,
            recipe_dirs=tuple(self.recipe_dirs()),
            output_dir=str(output_dir),
            options=dict(options or {}),
            include_fact_tables=settings.analysis.include_fact_tables,
            attach_data_file=settings.analysis.attach_data_file,
        )
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            # A stale report from an earlier analysis of this run must never be
            # mistaken for this one's answer.
            (output_dir / REPORT_FILENAME).unlink(missing_ok=True)
            spec_path = output_dir / SPEC_FILENAME
            spec_path.write_text(
                json.dumps(spec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            logger.error("Could not write the analysis spec for run %s: %s", run_id, exc)
            return ""

        self._queue.append(
            _Request(run_id=run_id, recipe=chosen, spec_path=spec_path, output_dir=output_dir)
        )
        self._start_next()
        return str(output_dir)

    def cancel(self, run_id: str = "") -> None:
        """Stop an analysis: kill its worker, or drop it from the queue.

        A cancelled analysis still ends in an entry — the facts-only one —
        because a run whose analysis was abandoned must not be left with
        nothing waiting for its human.

        Args:
            run_id: The run to cancel. ``""`` cancels everything in flight.
        """
        dropped = [
            request
            for request in self._queue
            if not run_id or request.run_id == run_id
        ]
        self._queue = [request for request in self._queue if request not in dropped]
        for request in dropped:
            self._finish(request, self._failed_report(request, "analysis cancelled"))
        active = self._active
        if active is not None and (not run_id or active.run_id == run_id):
            self._failure = "analysis cancelled"
            self._kill()

    # ------------------------------------------------------------------
    # The worker's life
    # ------------------------------------------------------------------

    def _pending(self) -> list[_Request]:
        """Return every request in flight, the active one first."""
        return ([self._active] if self._active is not None else []) + list(self._queue)

    def _start_next(self) -> None:
        """Launch the next queued worker, unless one is already running."""
        if self._active is not None or not self._queue:
            return
        request = self._queue.pop(0)
        self._active = request
        self._failure = ""
        process = QProcess(self)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        timeout_s = max(float(self._settings_source().analysis.timeout_s), 1.0)
        self._timer.start(int(timeout_s * 1000))
        logger.info(
            "Analysing run %s with recipe %s (timeout %.0f s)",
            request.run_id,
            request.recipe or "(discovered)",
            timeout_s,
        )
        process.start(
            self._python,
            ["-m", "i2as.analysis", "run", "--spec", str(request.spec_path)],
        )
        self.analysis_started.emit(request.run_id)

    def _kill(self) -> None:
        """Kill the running worker, if there is one. Never waits."""
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()

    def _on_timeout(self) -> None:
        """Bound a runaway recipe: kill it and let ``_on_finished`` report it."""
        request = self._active
        if request is None:
            return
        timeout_s = max(float(self._settings_source().analysis.timeout_s), 1.0)
        self._failure = f"analysis timed out after {timeout_s:.0f} s"
        logger.warning("Analysis of run %s timed out — killing the worker", request.run_id)
        self._kill()

    def _on_error(self, error: QProcess.ProcessError) -> None:
        """Record a worker that could not be started or crashed on the way in.

        Args:
            error: What Qt reported. ``FailedToStart`` is the one that never
                reaches ``finished`` on some platforms, so it is turned into a
                finish here.
        """
        request = self._active
        if request is None:
            return
        self._failure = self._failure or f"the analysis worker failed to run ({error.name})"
        if error == QProcess.ProcessError.FailedToStart:
            self._on_finished(-1, QProcess.ExitStatus.CrashExit)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        """Read the worker's report and hand it on. Never raises.

        Args:
            exit_code: The worker's exit code.
            exit_status: Whether it exited normally or was killed.
        """
        request = self._active
        if request is None:
            return
        self._timer.stop()
        stderr = self._drain_stderr()
        self._active = None
        process, self._process = self._process, None
        if process is not None:
            process.deleteLater()

        report = self._read_report(request)
        if report is None or self._failure:
            detail = self._failure or (
                f"the analysis worker wrote no report (exit code {exit_code}, "
                f"{exit_status.name}){chr(10) + stderr if stderr else ''}"
            )
            report = self._failed_report(request, detail)
        self._finish(request, report)
        self._start_next()

    def _finish(self, request: _Request, report: AnalysisReport) -> None:
        """Hand one finished analysis to the publisher and announce it.

        Args:
            request: The analysis that ended.
            report: Its report — real or synthesized.
        """
        if report.ok:
            self._call_publisher(
                "export_report", request.run_id, report, str(request.output_dir)
            )
            self.analysis_finished.emit(request.run_id, report.to_dict())
            return
        first_line = report.error.strip().splitlines()[0] if report.error.strip() else "analysis failed"
        self._call_publisher("park_facts_entry", request.run_id, warning=first_line)
        logger.warning("Analysis of run %s failed: %s", request.run_id, first_line)
        self.analysis_failed.emit(request.run_id, report.error)

    def _call_publisher(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Call one publisher method, containing every failure.

        The runner sits on a Qt signal path: a publisher that raises must not
        take the event loop down with it, and must not stop the next queued
        analysis from starting.

        Args:
            method: The publisher method's name.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        handler = getattr(self._publisher, method, None)
        if handler is None:
            logger.warning("The ELN publisher offers no %s() — nothing was parked", method)
            return
        try:
            handler(*args, **kwargs)
        except Exception:  # noqa: BLE001 - a notebook must never break the runner
            logger.exception("The ELN publisher raised in %s()", method)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _drain_stderr(self) -> str:
        """Return the tail of the worker's stderr, or ``""``."""
        if self._process is None:
            return ""
        text = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace")
        return text[-_STDERR_TAIL_CHARS:].strip()

    def _read_report(self, request: _Request) -> AnalysisReport | None:
        """Return the worker's report, or ``None`` when it wrote none.

        Args:
            request: The analysis that ended.

        Returns:
            The parsed report; ``None`` when the file is absent or unreadable
            (the caller synthesizes a failure from the worker's stderr).
        """
        path = request.output_dir / REPORT_FILENAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("No usable analysis report at %s: %s", path, exc)
            return None
        return AnalysisReport.from_dict(payload)

    def _failed_report(self, request: _Request, error: str) -> AnalysisReport:
        """Build the report an ending that produced none is reported as.

        Args:
            request: The analysis that ended.
            error: What went wrong, first line first.

        Returns:
            A ``failed`` report naming the run and the recipe that was asked
            for, so a failure is described in exactly the words a real report
            would use.
        """
        return AnalysisReport(
            run_id=request.run_id,
            recipe=request.recipe,
            status=REPORT_FAILED,
            error=error,
        )

    def _experiment_facts(self, experiment: Any) -> dict[str, Any]:
        """Return the experiment facts a recipe may cite.

        Args:
            experiment: The open ``ExperimentRecord``.

        Returns:
            ``experiment_id``, ``experiment_title``, ``sample_info``,
            ``findings`` and ``user_name`` — never a credential, never a live
            object.
        """
        context = self._manager.experiment_context().get("experiment") or {}
        return {
            "experiment_id": experiment.experiment_id,
            "experiment_title": experiment.title,
            "sample_info": dict(experiment.sample_info or {}),
            "findings": experiment.findings,
            "user_name": str(context.get("user_name", "")),
        }

    def _data_path(self, experiment_id: str, run: Any) -> str:
        """Return the absolute path of a run's data file, or ``""``.

        Args:
            experiment_id: The owning experiment's store key.
            run: The recorded run.

        Returns:
            The absolute path as a string; ``""`` when the run recorded none.
        """
        if not getattr(run, "data_file", ""):
            return ""
        return str(self._manager.store.resolve_data_file(experiment_id, run.data_file))
