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

**Every ending produces a bundle.** Each analysis writes into a folder of its
own, and when the worker ends — a report, a raising recipe, a timeout, a
missing report, a cancellation — the application SEALS that folder into an
**analysis bundle** (``i2as.analysis.bundle``): the worker's files hashed and
listed, its report parsed and capped, a ``failed`` status with the reason when
there was no usable report. The worker never writes the manifest itself.

**The runner knows no notebook.** A completed recipe bundle becomes the run's
**selected bundle** (``ExperimentManager.select_bundle``) and is announced on
``bundle_ready``; what happens to it next — a preview, a publish — is decided
elsewhere, from the bundle alone.

**An analysis script selects nothing.** ``start_script()`` runs one
exploratory **analysis script** (``i2as.analysis.scripts``) over one run, in
the same worker and the same queue, into the script's own folder, sealed as
the bundle ``script-<script_id>``. Whether it represents the run is a
separate, explicit decision (``select_analysis_bundle``), so exploring a run
never replaces what it already stands for.

**Every worker runs in a container** (``analysis_sandbox``): the settings'
``analysis.sandbox`` names the engine and the image; the sandbox maps the
spec's paths into the container when the analysis is started, and plans the
``<engine> run`` command — and the command that stops it — when it is
launched. No engine, no image, or an unmountable path is a ``failed`` report
naming why. Tests hand in a ``sandbox_factory`` that runs the worker as a
plain child process instead.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from i2as.analysis.bundle import (
    PRODUCER_RECIPE,
    PRODUCER_SCRIPT,
    BundleInput,
    Producer,
    new_bundle_id,
    script_bundle_id,
    seal_bundle,
)
from i2as.analysis.report import (
    REPORT_FAILED,
    REPORT_FILENAME,
    SPEC_FILENAME,
    AnalysisReport,
    AnalysisSpec,
    read_report_file,
)
from i2as.session.analysis_sandbox import (
    AnalysisSandbox,
    LaunchPlan,
    SandboxError,
    build_sandbox,
)
from i2as.session.app_config import AnalysisSettings, SandboxSettings
from i2as.session.manager import ExperimentManager
from i2as.session.run_facts import manifest_from_run

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
        sandbox: The backend that staged this analysis's inputs and will
            launch its worker.
        script_id: The **analysis script** this request runs, or ``""`` for a
            recipe analysis. A script's bundle is never selected here.
        bundle_id: The bundle this analysis is sealed as.
        experiment_id: The experiment the run belongs to.
        inputs: The run's identity, stamped into the bundle.
        actor: Who asked for the analysis.
    """

    run_id: str
    recipe: str
    spec_path: Path
    output_dir: Path
    sandbox: AnalysisSandbox
    script_id: str = ""
    bundle_id: str = ""
    experiment_id: str = ""
    inputs: tuple[BundleInput, ...] = ()
    actor: str = ""


class AnalysisRunner(QObject):
    """Runs one analysis worker at a time and seals what it leaves as a bundle.

    Signals:
        analysis_started (str): The run id, when its worker actually starts.
        analysis_finished (str, dict): The run id and the report as its JSON
            dict, when a recipe ran to completion.
        analysis_failed (str, str): The run id and the failure text, for every
            other ending of a recipe analysis — a raising recipe, a timeout, a
            missing report, a cancellation.
        script_finished (str, str, dict): The run id, the script id and the
            report as its JSON dict, for EVERY ending of an analysis script —
            ok or failed. The two signals above are never emitted for a
            script, so the analysis tab only ever hears about the run's recipe.
        bundle_ready (str, str, dict): The run id, the bundle id and the
            sealed bundle as its JSON dict, for EVERY ending, recipe or script.
    """

    analysis_started = pyqtSignal(str)
    analysis_finished = pyqtSignal(str, dict)
    analysis_failed = pyqtSignal(str, str)
    script_finished = pyqtSignal(str, str, dict)
    bundle_ready = pyqtSignal(str, str, dict)

    def __init__(
        self,
        manager: ExperimentManager,
        settings_source: Callable[[], AnalysisSettings],
        sandbox_factory: Callable[[SandboxSettings], AnalysisSandbox] = build_sandbox,
        parent: QObject | None = None,
    ) -> None:
        """Wire the runner to the session layer.

        Args:
            manager: The session-layer façade — the open experiment, the run
                records, the store paths, the experiment context a spec is
                built from, and the one writer of the run's bundle selection.
            settings_source: Called for the current ``AnalysisSettings`` at
                the moment each analysis starts, so a settings change reaches
                the next run without re-wiring anything.
            sandbox_factory: Builds the sandbox one analysis runs in from the
                ``analysis.sandbox`` settings. The default is the container;
                tests pass one that returns a ``SubprocessSandbox``.
            parent: Qt parent, if any.
        """
        super().__init__(parent)
        self._manager = manager
        self._settings_source = settings_source
        self._sandbox_factory = sandbox_factory
        self._queue: list[_Request] = []
        self._active: _Request | None = None
        self._stop_command: tuple[str, ...] = ()
        self._process: QProcess | None = None
        self._failure: str = ""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

    # ------------------------------------------------------------------
    # Read surface
    # ------------------------------------------------------------------

    def is_running(self, run_id: str = "", script_id: str = "") -> bool:
        """Whether an analysis is in flight.

        Args:
            run_id: Ask about one run. ``""`` asks about any analysis at all.
            script_id: With a ``run_id``, ask about one analysis script of it;
                ``""`` asks about the run's RECIPE analysis, which is what
                every caller that predates scripts means.

        Returns:
            ``True`` while a worker is running or a request is queued behind
            one — from the caller's side both mean "the answer is not here
            yet".
        """
        pending = self._pending()
        if not run_id:
            return bool(pending)
        return any(
            request.run_id == run_id and request.script_id == script_id
            for request in pending
        )

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
        actor: str = "",
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
            actor: Who asked (an agent id, ``"operator"``), stamped into the
                bundle; ``""`` for the automatic analysis of a finished run.

        Returns:
            The new bundle's folder as a string (its name is the bundle id),
            or ``""`` when the analysis could not be started: no experiment
            open, no such run, or no data file to read (all logged, never
            raised).
        """
        return self._enqueue(
            run_id,
            manifest=manifest,
            data_path=data_path,
            recipe=recipe,
            options=options,
            actor=actor,
        )

    def start_script(
        self,
        run_id: str,
        script_id: str,
        script_path: str | Path,
        options: Mapping[str, Any] | None = None,
        actor: str = "",
    ) -> str:
        """Run one **analysis script** over one recorded run, later. Never raises.

        The script's folder (``script_path``'s parent) is its output folder
        and its bundle (``script-<script_id>``): its spec, report, figures and
        captured output are written there, and the answer arrives on
        ``script_finished``. Nothing is selected.

        Args:
            run_id: The run to analyse, in the open experiment.
            script_id: The script's id — the key ``is_running()`` and
                ``script_finished`` use.
            script_path: The script file, already written into its folder.
            options: Free-form options, passed through to the script.
            actor: Who asked, stamped into the bundle.

        Returns:
            The script's output folder as a string, or ``""`` when nothing
            could be started (logged, never raised).
        """
        path = Path(script_path)
        return self._enqueue(
            run_id,
            options=options,
            script_id=script_id,
            script_path=path,
            output_dir=path.parent,
            actor=actor,
        )

    def _enqueue(
        self,
        run_id: str,
        *,
        manifest: Mapping[str, Any] | None = None,
        data_path: str = "",
        recipe: str = "",
        options: Mapping[str, Any] | None = None,
        script_id: str = "",
        script_path: Path | None = None,
        output_dir: Path | None = None,
        actor: str = "",
    ) -> str:
        """Build, stage and write one spec, then queue its worker.

        Args:
            run_id: The run to analyse.
            manifest: Fresher run facts, merged over the record's.
            data_path: The run's data file, or ``""`` to resolve it.
            recipe: The recipe name, or ``""``.
            options: Options passed through.
            script_id: The script's id, or ``""`` for a recipe analysis.
            script_path: The script file, for a script analysis.
            output_dir: Where the worker writes; ``None`` for a new bundle
                folder under the run's analysis folder.
            actor: Who asked.

        Returns:
            The output folder as a string, or ``""`` when nothing was queued.
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
        chosen = "" if script_path is not None else (
            recipe or settings.recipes.get(str(facts.get("procedure", "")), "")
        )
        if script_path is not None:
            bundle_id = script_bundle_id(script_id)
        else:
            bundle_id = new_bundle_id(chosen or "analysis")
        if output_dir is None:
            output_dir = self._manager.store.report_dir(experiment.experiment_id, run_id) / bundle_id
        sandbox = self._sandbox_factory(settings.sandbox)
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
            include_fact_tables=settings.include_fact_tables,
            attach_data_file=settings.attach_data_file,
            script_path=str(script_path) if script_path is not None else "",
        )
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            # A stale report or seal in a reused folder (a script run twice)
            # must never be mistaken for this analysis's answer.
            (output_dir / REPORT_FILENAME).unlink(missing_ok=True)
            (output_dir / "bundle.json").unlink(missing_ok=True)
            spec = sandbox.prepare(spec)
            spec_path = output_dir / SPEC_FILENAME
            spec_path.write_text(
                json.dumps(spec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except (OSError, SandboxError) as exc:
            logger.error("Could not prepare the analysis of run %s: %s", run_id, exc)
            return ""

        self._queue.append(
            _Request(
                run_id=run_id,
                recipe=chosen,
                spec_path=spec_path,
                output_dir=output_dir,
                sandbox=sandbox,
                script_id=script_id,
                bundle_id=bundle_id,
                experiment_id=experiment.experiment_id,
                inputs=(
                    BundleInput(
                        run_id=run_id,
                        data_file=str(getattr(run, "data_file", "") or ""),
                        params_digest=str(getattr(run, "params_digest", "") or ""),
                    ),
                ),
                actor=actor,
            )
        )
        self._start_next()
        return str(output_dir)

    def cancel(self, run_id: str = "") -> None:
        """Stop an analysis: kill its worker, or drop it from the queue.

        A cancelled analysis still ends in a (failed) bundle, so the run's
        analysis folder says what happened to it.

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
        """Launch the next queued worker, unless one is already running.

        A request whose sandbox cannot plan a launch is finished at once with
        a failed report naming why, and the next one is tried.
        """
        while self._active is None and self._queue:
            request = self._queue.pop(0)
            try:
                plan = request.sandbox.launch(request.spec_path)
            except SandboxError as exc:
                logger.error("Analysis of run %s cannot start: %s", request.run_id, exc)
                self._finish(request, self._failed_report(request, str(exc)))
                continue
            self._launch(request, plan)

    def _launch(self, request: _Request, plan: LaunchPlan) -> None:
        """Start one worker as its sandbox planned it.

        Args:
            request: The analysis to start.
            plan: The sandbox's ``LaunchPlan``.
        """
        self._active = request
        self._failure = ""
        self._stop_command = plan.stop_command
        process = QProcess(self)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        timeout_s = max(float(self._settings_source().timeout_s), 1.0)
        self._timer.start(int(timeout_s * 1000))
        if plan.environment is not None:
            environment = QProcessEnvironment()
            for name, value in plan.environment.items():
                environment.insert(name, value)
            process.setProcessEnvironment(environment)
        if plan.working_directory:
            process.setWorkingDirectory(plan.working_directory)
        logger.info(
            "Analysing run %s with %s (timeout %.0f s, sandbox %s)",
            request.run_id,
            f"script {request.script_id}" if request.script_id
            else f"recipe {request.recipe or '(discovered)'}",
            timeout_s,
            plan.backend,
        )
        process.start(plan.program, list(plan.arguments))
        if not request.script_id:
            self.analysis_started.emit(request.run_id)

    def _kill(self) -> None:
        """Kill the running worker, if there is one. Never waits.

        A container worker is stopped by name as well: killing the engine's
        ``run`` client does not always stop the container it started.
        """
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            if self._stop_command:
                QProcess.startDetached(self._stop_command[0], list(self._stop_command[1:]))
            self._process.kill()

    def _on_timeout(self) -> None:
        """Bound a runaway recipe: kill it and let ``_on_finished`` report it."""
        request = self._active
        if request is None:
            return
        timeout_s = max(float(self._settings_source().timeout_s), 1.0)
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
        self._stop_command = ()
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
        """Seal one finished analysis into its bundle and announce it.

        A completed recipe bundle becomes the run's selected bundle; a
        script's bundle is announced on ``script_finished`` and selects
        nothing.

        Args:
            request: The analysis that ended.
            report: Its report — real or synthesized.
        """
        bundle = self._seal(request, report)
        if request.script_id:
            self.script_finished.emit(request.run_id, request.script_id, report.to_dict())
        elif report.ok:
            if bundle is not None:
                self._select(request)
            self.analysis_finished.emit(request.run_id, report.to_dict())
        else:
            first_line = report.error.strip().splitlines()[0] if report.error.strip() else "analysis failed"
            logger.warning("Analysis of run %s failed: %s", request.run_id, first_line)
            self.analysis_failed.emit(request.run_id, report.error)
        if bundle is not None:
            self.bundle_ready.emit(request.run_id, request.bundle_id, bundle)

    def _seal(self, request: _Request, report: AnalysisReport) -> dict[str, Any] | None:
        """Seal the request's folder as its bundle. Never raises.

        Args:
            request: The analysis that ended.
            report: The parsed, capped report (or a synthesized failure).

        Returns:
            The sealed bundle as its JSON dict, or ``None`` when the folder
            could not be sealed (logged).
        """
        if request.script_id:
            producer = Producer(
                kind=PRODUCER_SCRIPT,
                name=request.script_id,
                digest=report.recipe_digest,
                actor=request.actor,
            )
        else:
            producer = Producer(
                kind=PRODUCER_RECIPE,
                name=report.recipe or request.recipe,
                digest=report.recipe_digest,
                actor=request.actor,
            )
        try:
            bundle = seal_bundle(
                request.output_dir,
                report.to_dict(),
                bundle_id=request.bundle_id,
                experiment_id=request.experiment_id,
                run_ids=(request.run_id,),
                producer=producer,
                inputs=request.inputs,
            )
        except OSError as exc:
            logger.error("Could not seal the bundle of run %s: %s", request.run_id, exc)
            return None
        return bundle.to_dict()

    def _select(self, request: _Request) -> None:
        """Make a completed recipe bundle the run's selection. Never raises.

        Only in the experiment the analysis was started in: a bundle that
        finishes after the physicist switched experiments stays on disk,
        selectable by hand, without touching the wrong record.

        Args:
            request: The analysis that completed.
        """
        experiment = self._manager.current_experiment()
        if experiment is None or experiment.experiment_id != request.experiment_id:
            return
        try:
            self._manager.select_bundle(request.run_id, request.bundle_id)
        except Exception:  # noqa: BLE001 - bookkeeping must never break the runner
            logger.exception("Selecting bundle %s for run %s failed", request.bundle_id, request.run_id)

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
        report = read_report_file(path)
        if report is None:
            logger.warning("No usable analysis report at %s", path)
        return report

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
