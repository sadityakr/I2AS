"""The publisher — what gets queued when, and the GUI-side drain that sends it.

Three responsibilities, and deliberately no fourth:

1. **Render and queue.** A run that finishes is rendered to its final entry
   text *at that moment* (``templates.py``) and appended to that experiment's
   **Outbox**. The same happens on a manual export. Queuing touches no
   network, so it costs a finished run nothing.
2. **Drain.** ``drain_once()`` sends at most one queued job, and is meant to
   be called from a **QTimer in the GUI process** — never from the
   Orchestrator's tick. This keeps the repository's one-thread promise intact:
   a slow upload delays the next upload, never a hardware write.
3. **Record.** When the backend confirms an entry, the publisher hands the
   reference to ``ExperimentManager.set_run_eln_link()``. It never edits a
   record or writes a file itself — the manager is the single writer of
   experiment state, exactly as the Orchestrator is the single writer to
   hardware.

**Publishing is opt-in and never silent.** With no user-level settings file
(the default) nothing is constructed and nothing leaves the machine. With
``auto_publish`` off, only an explicit export queues anything. And nothing is
ever published for a run that belongs to no experiment: an ad-hoc run has no
record to attach an entry to, and uploading it would be a surprise.

**An approved draft is not a second write path.** ``export_draft()`` queues
one ordinary outbox job whose title, body and tags come from an approved
**draft entry** instead of from the renderers, stamping the model and prompt
digest into the entry's metadata. It is the same journal, the same
idempotency, the same drain — the draft is data, and only the text differs.

**With the analysis stage on, a finished run is not queued at all.** It is
handed to the analysis stage instead — ``on_run_finished()`` emits
``analysis_requested`` and enqueues nothing — and what comes back is PARKED
as a **Pending entry** on the run record (``export_report()`` for an analysed
one, ``park_facts_entry()`` for the facts-only fallback when a recipe failed
or never ran). Approval is then the only door to the outbox, through
``ExperimentManager.approve_eln_draft()`` exactly as a model draft's is. That
is why ``auto_publish`` says nothing on this path: with analysis switched on,
nothing publishes until a human says so, and a failed analysis still leaves a
complete, correct entry waiting rather than losing the run.

**Backends are discovered, not listed.** ``discover_backends()`` walks the
package for ``ElnAdapter`` subclasses and keys them by their declared
``backend``, so a new backend module is selectable from the settings file the
moment its file exists — no registry to edit, exactly as drivers, VIs, and
procedures are discovered elsewhere in this repository.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from i2as.analysis.report import AnalysisReport
from i2as.session.eln.adapter import ElnAdapter
from i2as.session.eln.drafting import (
    SOURCE_ANALYSIS,
    SOURCE_FACTS,
    DraftEntry,
    manifest_from_run,
)
from i2as.session.eln.outbox import (
    DRAIN_IDLE,
    DRAIN_PUBLISHED,
    DRAIN_RETRY,
    JOB_PUBLISH_RUN,
    DrainResult,
    Outbox,
    OutboxJob,
)
from i2as.session.eln.settings import ElnSettings
from i2as.session.eln.templates import (
    render_analysed_body,
    render_analysed_title,
    render_prose_section,
    render_run_body,
    render_run_metadata,
    render_run_title,
)
from i2as.session.manager import ExperimentManager
from i2as.session.models import ElnLink, RunRecord

logger = logging.getLogger(__name__)

#: Publish state: everything queued has reached the notebook.
PUBLISH_SYNCED = "synced"

#: Publish state: jobs are queued and the last attempt did not fail.
PUBLISH_PENDING = "pending"

#: Publish state: the last attempt failed; the queue is retrying with backoff.
PUBLISH_OFFLINE = "offline"

#: Publish state: no usable settings — the track is switched off entirely.
PUBLISH_DISABLED = "disabled"


def discover_backends() -> dict[str, type[ElnAdapter]]:
    """Return every available ELN backend, keyed by its ``backend`` id.

    Walks ``i2as.session.eln`` for ``ElnAdapter`` subclasses rather than
    consulting a hand-maintained table, so adding a backend is adding a file.

    Returns:
        ``{backend_id: adapter_class}``; ids are the classes' own declared
        ``backend`` attributes.
    """
    import i2as.session.eln as package

    backends: dict[str, type[ElnAdapter]] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, ElnAdapter)
                and value is not ElnAdapter
                and value.__module__ == module.__name__
                and value.backend
            ):
                backends[value.backend] = value
    return backends


def _manifest_of(event: Any) -> dict[str, Any]:
    """Return the run manifest carried by whatever the caller passed.

    Accepts both shapes the application produces: a ``RunFinished`` contract
    event (whose ``manifest`` attribute holds it) and the Orchestrator's
    ``run_finished`` signal payload, which *is* the manifest dict.

    Args:
        event: A ``RunFinished`` event, or a plain manifest mapping.

    Returns:
        The manifest as a plain dict; empty when the argument is neither.
    """
    manifest = getattr(event, "manifest", None)
    if isinstance(manifest, dict):
        return dict(manifest)
    return dict(event) if isinstance(event, dict) else {}


class ElnPublisher(QObject):
    """Queues finished runs for the notebook and drains the queue off the tick.

    Signals:
        publish_state_changed (dict): ``{"state": str, "pending": int,
            "detail": str}`` — ``synced`` / ``pending`` / ``offline`` /
            ``disabled``, for a GUI status chip. Emitted after every drain and
            every enqueue.
        run_published (dict): ``{"run_id": str, "experiment_id": str,
            "eln_link": {...}}``, emitted once per run when the backend
            confirms its entry.
        analysis_requested (str, dict, str): ``run_id``, the run manifest and
            the absolute data path — emitted INSTEAD of queuing when the
            analysis stage is switched on. Whoever owns the analysis runner
            connects it; with nothing connected the run simply waits, and a
            manual export still queues it.
    """

    publish_state_changed = pyqtSignal(dict)
    run_published = pyqtSignal(dict)
    analysis_requested = pyqtSignal(str, dict, str)

    def __init__(
        self,
        manager: ExperimentManager,
        settings: ElnSettings | None = None,
        adapter: ElnAdapter | None = None,
    ) -> None:
        """Wire the publisher to one experiment manager.

        Args:
            manager: The session-layer façade. Supplies the open experiment,
                the store (for outbox paths and data-file resolution), the
                setup context the body is rendered from, and the single write
                path for the resulting ``ElnLink``.
            settings: The user-level ELN settings. ``None`` loads the
                defaults, which have publishing switched off.
            adapter: The backend to publish through. ``None`` builds one from
                ``settings.backend`` on first use; tests inject a
                ``SimElnAdapter``.
        """
        super().__init__()
        self._manager = manager
        self._settings = settings or ElnSettings()
        self._adapter = adapter
        self._outboxes: dict[str, Outbox] = {}
        self._state = PUBLISH_SYNCED if self._settings.enabled else PUBLISH_DISABLED
        self._timer = QTimer(self)
        self._timer.setInterval(max(1, round(self._settings.drain_interval_s * 1000.0)))
        self._timer.timeout.connect(self.drain_once)
        self._adopt_existing_outboxes()

    # ------------------------------------------------------------------
    # Read surface
    # ------------------------------------------------------------------

    @property
    def settings(self) -> ElnSettings:
        """The settings this publisher was built with."""
        return self._settings

    def pending_count(self) -> int:
        """Return how many jobs are queued across every known experiment."""
        return sum(len(outbox.pending()) for outbox in self._outboxes.values())

    def status(self) -> dict[str, Any]:
        """Return the current publish status, the shape the GUI chip renders.

        Returns:
            ``{"state": str, "pending": int, "detail": str}``.
        """
        return {"state": self._state, "pending": self.pending_count(), "detail": ""}

    # ------------------------------------------------------------------
    # The drain timer (owned here, started by whoever runs an event loop)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the drain timer. No-op when the track is not configured.

        Deliberately explicit rather than automatic in ``__init__``: the timer
        needs a running Qt event loop, so it is the application entry point —
        the GUI side — that decides when publishing goes live.
        """
        if not self._settings.is_configured:
            logger.info("ELN publishing is not configured — the drain timer stays off")
            return
        self._timer.start()
        logger.info(
            "ELN drain timer started (every %.0f s, backend=%s)",
            self._settings.drain_interval_s,
            self._settings.backend,
        )

    def stop(self) -> None:
        """Stop the drain timer (app exit / test teardown)."""
        self._timer.stop()

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def on_run_finished(
        self, event: Any, manifest: dict[str, Any] | None = None, data_path: str = ""
    ) -> str:
        """Queue a finished run for publication. Never raises.

        The auto-publish trigger. Connect it to the Orchestrator's
        ``run_finished`` signal, or call it with a ``RunFinished`` contract
        event; both carry the same manifest.

        Args:
            event: The ``RunFinished`` event, or the manifest dict the
                Orchestrator emits.
            manifest: The run manifest, when the caller has it separately.
                ``None`` takes it from ``event``.
            data_path: Absolute path of the run's data file. ``""`` resolves
                it from the recorded run through the store.

        With the analysis stage switched on this queues NOTHING: the run is
        handed to the analysis stage through ``analysis_requested`` and the
        entry it produces is parked for approval instead. ``auto_publish``
        says nothing there, because on that path nothing publishes until a
        human approves it.

        Returns:
            The queued job's id, or ``""`` when nothing was queued (publishing
            off, the analysis stage took the run, auto-publish off, no
            experiment open, or the job was already queued).
        """
        if not self._settings.enabled:
            return ""
        payload = manifest if manifest is not None else _manifest_of(event)
        run_id = str(payload.get("run_id", ""))
        if self._request_analysis(run_id, payload, data_path):
            return ""
        if not self._settings.auto_publish:
            logger.debug("Auto-publish is off — run left for a manual export")
            return ""
        return self._enqueue_run(run_id, payload, data_path)

    def _request_analysis(
        self, run_id: str, manifest: Mapping[str, Any], data_path: str
    ) -> bool:
        """Hand one finished run to the analysis stage, or decline to.

        The fork the whole track turns on. Analysing needs three things: the
        stage switched on, an open experiment to write the results into, and a
        data file to read. Missing any of them, the run takes today's path and
        is rendered from its facts.

        Args:
            run_id: The finished run.
            manifest: Its run manifest.
            data_path: The caller's data path, or ``""`` to resolve it.

        Returns:
            ``True`` when ``analysis_requested`` was emitted and the caller
            must not queue the run.
        """
        if not (run_id and self._settings.analysis.enabled):
            return False
        experiment = self._manager.current_experiment()
        if experiment is None:
            logger.debug("Run %r belongs to no experiment — not analysed", run_id)
            return False
        resolved = data_path or self._data_path_for(
            experiment.experiment_id, run_id, manifest
        )
        if not resolved:
            logger.warning("Run %s has no data file to analyse", run_id)
            return False
        logger.info("Run %s goes to the analysis stage before the notebook", run_id)
        self.analysis_requested.emit(run_id, dict(manifest), resolved)
        return True

    def _data_path_for(
        self, experiment_id: str, run_id: str, manifest: Mapping[str, Any]
    ) -> str:
        """Return a run's data path from its manifest, else from its record.

        Args:
            experiment_id: The owning experiment's store key.
            run_id: The run.
            manifest: The run manifest, whose ``data_file`` is preferred
                because it is what the Orchestrator captured at run start.

        Returns:
            The absolute path as a string, or ``""`` when the run has none.
        """
        data_file = str(manifest.get("data_file") or "")
        if data_file:
            return str(self._manager.store.resolve_data_file(experiment_id, data_file))
        experiment = self._manager.current_experiment()
        run = experiment.find_run(run_id) if experiment is not None else None
        return self._resolve_data_path(experiment_id, run)

    def export_run(self, run_id: str, data_path: str = "") -> str:
        """Queue one already-recorded run on demand — the manual "Publish now".

        Ignores ``auto_publish`` (the operator asked for this one explicitly)
        but not ``enabled``: with the track switched off there is nowhere to
        send it.

        Args:
            run_id: The run to publish, in the open experiment.
            data_path: Absolute path of the run's data file. ``""`` resolves
                it from the recorded run through the store.

        Returns:
            The queued job's id, or ``""`` when nothing was queued.
        """
        if not self._settings.enabled:
            return ""
        return self._enqueue_run(run_id, None, data_path)

    def export_draft(
        self, run_id: str, draft: DraftEntry | Mapping[str, Any], data_path: str = ""
    ) -> str:
        """Queue one run under an approved **draft entry**'s own text.

        The write half of the drafting track, and deliberately the SAME write
        half everything else uses: a draft becomes an ordinary outbox job
        whose title, body and tags come from the draft instead of from the
        renderers, with the drafting provenance stamped into the entry's
        metadata. It is idempotent by the same ``job_id``, so a run already
        queued is not queued again under a draft.

        Args:
            run_id: The run to publish, in the open experiment.
            draft: The approved draft — a ``DraftEntry``, or its JSON dict as
                the run record stores it (tolerantly loaded).
            data_path: Absolute path of the run's data file. ``""`` resolves
                it from the recorded run through the store.

        Returns:
            The queued job's id, or ``""`` when nothing was queued.
        """
        if not self._settings.enabled:
            return ""
        entry = draft if isinstance(draft, DraftEntry) else DraftEntry.from_dict(draft)
        return self._enqueue_run(run_id, None, data_path, draft=entry)

    def park_facts_entry(self, run_id: str, warning: str = "") -> bool:
        """Park today's facts-only entry on one run, waiting for approval.

        The fallback that loses nothing. When the analysis stage was asked and
        could not answer — a recipe raised, the worker timed out, no recipe
        matched — the run still gets a complete, correct entry: exactly the
        title and body a published run has always had, with the reason the
        analysis is missing said in the first section. It is PARKED, not
        queued: with the analysis stage on, approval is the only door to the
        notebook.

        Args:
            run_id: The run, in the open experiment.
            warning: One line saying why there is no analysis, rendered as
                escaped prose above the facts. ``""`` renders no such section.

        Returns:
            ``True`` when the entry was parked; ``False`` when there is no
            open experiment or no such run (logged, never raised).
        """
        prepared = self._entry_facts(run_id)
        if prepared is None:
            return False
        experiment, facts, resolved, context = prepared
        body = render_prose_section("Analysis unavailable", warning) + render_run_body(
            facts,
            experiment_id=experiment.experiment_id,
            experiment_title=experiment.title,
            setup=context.get("setup"),
            data_path=resolved,
            findings=experiment.findings,
        )
        entry = DraftEntry(
            title=render_run_title(facts, experiment.title),
            body_html=body,
            source=SOURCE_FACTS,
            attachments=[],
        )
        parked = bool(self._manager.set_pending_eln_draft(run_id, entry.to_dict()))
        if parked:
            logger.info("A facts-only entry for run %s is waiting for approval", run_id)
        return parked

    def export_report(
        self,
        run_id: str,
        report: AnalysisReport | Mapping[str, Any],
        report_dir: str | Path,
    ) -> bool:
        """Park one run's **analysed entry**, waiting for approval.

        The analysis stage's own hand-off. The report is rendered into the
        entry a physicist wants — prose, results, figures, provenance — its
        figures become attachments (their files live beside the report), and
        the raw data file stays attached only when the report asked for it.

        Parked whether the experiment is attended or not: with the analysis
        stage switched on, approval is the gate this design chose, and an
        unattended experiment's entry waits for the human who reads it later
        rather than publishing an unreviewed result.

        Args:
            run_id: The analysed run, in the open experiment.
            report: The **Analysis report**, or its ``report.json`` dict.
            report_dir: The directory the report and its figures were written
                to; every figure's ``file`` is relative to it.

        Returns:
            ``True`` when the entry was parked; ``False`` when there is no
            open experiment or no such run (logged, never raised).
        """
        rendered = (
            report if isinstance(report, AnalysisReport) else AnalysisReport.from_dict(report)
        )
        prepared = self._entry_facts(run_id)
        if prepared is None:
            return False
        experiment, facts, resolved, context = prepared
        directory = Path(report_dir)
        entry = DraftEntry(
            title=render_analysed_title(rendered, facts, experiment.title),
            body_html=render_analysed_body(
                rendered,
                facts,
                experiment_id=experiment.experiment_id,
                experiment_title=experiment.title,
                setup=context.get("setup"),
                data_path=resolved,
                findings=experiment.findings,
            ),
            tags=list(rendered.tags),
            attachments=[
                {"path": str(directory / figure.file), "comment": figure.caption or figure.file}
                for figure in rendered.figures
                if figure.file
            ],
            attach_data_file=rendered.attach_data_file,
            source=SOURCE_ANALYSIS,
            metadata={"recipe": rendered.recipe, "recipe_digest": rendered.recipe_digest},
        )
        parked = bool(self._manager.set_pending_eln_draft(run_id, entry.to_dict()))
        if parked:
            logger.info(
                "An analysed entry for run %s (recipe %s) is waiting for approval",
                run_id,
                rendered.recipe or "unnamed",
            )
        return parked

    def _entry_facts(
        self, run_id: str
    ) -> tuple[Any, dict[str, Any], str, dict[str, Any]] | None:
        """Gather what every parked entry is rendered from.

        Args:
            run_id: The run, in the open experiment.

        Returns:
            ``(experiment record, manifest-shaped facts, data path, experiment
            context)``, or ``None`` when no experiment is open or the run is
            unknown (logged, never raised).
        """
        experiment = self._manager.current_experiment()
        if experiment is None:
            logger.warning("No experiment is open — no entry can be parked for %r", run_id)
            return None
        run = experiment.find_run(run_id)
        if run is None:
            logger.warning("No recorded run %r — no entry can be parked", run_id)
            return None
        facts = self._manifest_from_record(run)
        resolved = self._resolve_data_path(experiment.experiment_id, run)
        return experiment, facts, resolved, self._manager.experiment_context()

    def reload_settings(self, settings: ElnSettings) -> None:
        """Swap the settings in, rebuild the backend, and re-arm the drain.

        What the **eLab setup** dialog calls after saving. The adapter is
        resolved afresh, because the URL, the key or the backend itself may
        have changed, and the drain timer follows the new settings: running
        while they are usable, stopped while they are not — so switching the
        track off stops the network the moment Save is pressed.

        Args:
            settings: The settings just saved.
        """
        self._settings = settings
        self._adapter = None
        self._timer.setInterval(max(1, round(settings.drain_interval_s * 1000.0)))
        self._timer.stop()
        if settings.is_configured:
            self._resolve_adapter()
            self.start()
        else:
            logger.info("ELN publishing is no longer configured — the drain timer is off")
        self._publish_state(DRAIN_IDLE)

    def _enqueue_run(
        self,
        run_id: str,
        manifest: dict[str, Any] | None,
        data_path: str,
        draft: DraftEntry | None = None,
    ) -> str:
        """Render one run and append it to its experiment's outbox.

        Args:
            run_id: The run to publish.
            manifest: The run manifest when one is at hand (the auto-publish
                path), else ``None`` — the recorded ``RunRecord`` is then the
                source of truth.
            data_path: Absolute data-file path, or ``""`` to resolve it.
            draft: An approved **draft entry** whose title, body and tags
                replace the rendered ones, or ``None`` for the rendered entry.

        Returns:
            The queued job's id, or ``""``.
        """
        experiment = self._manager.current_experiment()
        if experiment is None:
            logger.debug("Run %r belongs to no experiment — nothing published", run_id)
            return ""
        if not run_id:
            logger.warning("A run with no id cannot be published")
            return ""
        run = experiment.find_run(run_id)
        if run is None and manifest is None:
            logger.warning("No recorded run %r to export", run_id)
            return ""

        facts = dict(manifest or {})
        if run is not None:
            facts = {**self._manifest_from_record(run), **facts}
        resolved = data_path or self._resolve_data_path(experiment.experiment_id, run)
        context = self._manager.experiment_context()
        metadata = render_run_metadata(facts, experiment.experiment_id, resolved)
        if draft is not None:
            # The provenance of the pending entry travels with it, so the
            # notebook itself says which stage produced the text and — for a
            # model draft — from which prompt, or — for an analysed entry —
            # from which recipe source. The same accountability the Agent feed
            # keeps locally.
            metadata = {
                **metadata,
                "draft_model": draft.model,
                "draft_prompt_digest": draft.prompt_digest,
                "draft_source": draft.source,
            }
            for key in ("recipe", "recipe_digest"):
                value = draft.metadata.get(key)
                if value:
                    metadata[key] = value
        job = OutboxJob(
            job_id=f"{JOB_PUBLISH_RUN}:{run_id}",
            kind=JOB_PUBLISH_RUN,
            experiment_id=experiment.experiment_id,
            run_id=run_id,
            title=(draft.title if draft is not None and draft.title else None)
            or render_run_title(facts, experiment.title),
            body_html=(
                draft.body_html
                if draft is not None and draft.body_html
                else render_run_body(
                    facts,
                    experiment_id=experiment.experiment_id,
                    experiment_title=experiment.title,
                    setup=context.get("setup"),
                    data_path=resolved,
                    findings=experiment.findings,
                )
            ),
            tags=sorted(
                {*self._settings.tags, *(draft.tags if draft is not None else ())}
            ),
            template_id=self._settings.template_id,
            metadata=metadata,
            data_path=resolved,
            attach_data_file=draft.attach_data_file if draft is not None else True,
            attachments=[dict(item) for item in (draft.attachments if draft else [])],
            max_attachment_bytes=self._settings.max_attachment_bytes,
        )
        outbox = self._outbox_for(experiment.experiment_id)
        queued = outbox.enqueue(job)
        self._publish_state(DRAIN_IDLE if queued else self._state)
        return job.job_id if queued else ""

    @staticmethod
    def _manifest_from_record(run: RunRecord) -> dict[str, Any]:
        """Return a manifest-shaped dict built from a recorded run.

        Lets a manual export render exactly what auto-publish would, from the
        record alone — the Orchestrator's manifest is long gone by then. One
        owner (``drafting.manifest_from_run()``), so a published run and a
        drafted one describe a run in the same words.

        Args:
            run: The recorded run.

        Returns:
            The manifest-shaped facts, plus the **Params digest** the record
            keeps, which the analysed entry's provenance block names.
        """
        return {**manifest_from_run(run), "params_digest": getattr(run, "params_digest", "")}

    def _resolve_data_path(self, experiment_id: str, run: RunRecord | None) -> str:
        """Return the absolute path of a run's data file, or ``""``.

        Args:
            experiment_id: The owning experiment's store key.
            run: The recorded run, or ``None``.

        Returns:
            The absolute path as a string; ``""`` when the run has no data
            file recorded.
        """
        if run is None or not run.data_file:
            return ""
        return str(self._manager.store.resolve_data_file(experiment_id, run.data_file))

    # ------------------------------------------------------------------
    # Drain
    # ------------------------------------------------------------------

    def drain_once(self) -> DrainResult:
        """Send at most one queued job. Never raises — a GUI timer calls this.

        Returns:
            The ``DrainResult`` of the attempt, or an idle result when there
            is nothing due, no usable settings, or no backend to build.
        """
        if not self._settings.is_configured:
            return DrainResult(state=DRAIN_IDLE)
        adapter = self._resolve_adapter()
        if adapter is None:
            return DrainResult(state=DRAIN_IDLE)
        for experiment_id, outbox in list(self._outboxes.items()):
            result = outbox.drain(adapter)
            if result.state == DRAIN_IDLE:
                continue
            if result.state == DRAIN_PUBLISHED and result.entry is not None:
                self._record_link(experiment_id, result)
            self._publish_state(result.state, result.detail)
            return result
        self._publish_state(DRAIN_IDLE)
        return DrainResult(state=DRAIN_IDLE)

    def _record_link(self, experiment_id: str, result: DrainResult) -> None:
        """Hand a confirmed entry reference to the manager and announce it.

        Args:
            experiment_id: The owning experiment's store key.
            result: The successful drain result.
        """
        if result.entry is None:
            return
        link = ElnLink.from_dict(result.entry.to_dict())
        self._manager.set_run_eln_link(experiment_id, result.run_id, link)
        self.run_published.emit(
            {
                "run_id": result.run_id,
                "experiment_id": experiment_id,
                "eln_link": link.to_dict(),
            }
        )

    def _resolve_adapter(self) -> ElnAdapter | None:
        """Return the backend adapter, building it on first use.

        Returns:
            The adapter, or ``None`` when the configured backend is unknown
            or refuses to construct (logged once per attempt, never raised).
        """
        if self._adapter is not None:
            return self._adapter
        backends = discover_backends()
        adapter_cls = backends.get(self._settings.backend)
        if adapter_cls is None:
            logger.error(
                "Unknown ELN backend %r — available: %s",
                self._settings.backend,
                ", ".join(sorted(backends)) or "(none)",
            )
            return None
        try:
            self._adapter = adapter_cls(self._settings.to_dict(include_secret=True))
        except Exception:
            logger.exception("Could not build the %s ELN adapter", self._settings.backend)
            return None
        return self._adapter

    # ------------------------------------------------------------------
    # Outbox bookkeeping
    # ------------------------------------------------------------------

    def _outbox_for(self, experiment_id: str) -> Outbox:
        """Return (and remember) the outbox of one experiment.

        Args:
            experiment_id: The store key.

        Returns:
            That experiment's ``Outbox``.
        """
        outbox = self._outboxes.get(experiment_id)
        if outbox is None:
            outbox = Outbox(
                self._manager.store.outbox_path(experiment_id),
                retry_base_s=self._settings.retry_base_s,
                retry_max_s=self._settings.retry_max_s,
            )
            self._outboxes[experiment_id] = outbox
        return outbox

    def _adopt_existing_outboxes(self) -> None:
        """Pick up outboxes left on disk by an earlier run of the application.

        What makes the queue survive a restart: a job queued yesterday, on a
        day the notebook was down, drains today without anyone re-opening its
        experiment. Tolerates an unreadable store directory — nothing here may
        stop the application from starting.
        """
        try:
            experiment_ids = self._manager.store.list_experiments()
        except OSError as exc:
            logger.warning("Could not scan for ELN outboxes: %s", exc)
            return
        for experiment_id in experiment_ids:
            if self._manager.store.outbox_path(experiment_id).exists():
                self._outbox_for(experiment_id)
        pending = self.pending_count()
        if pending:
            logger.info("Resumed %d queued ELN job(s) from disk", pending)

    def _publish_state(self, drain_state: str, detail: str = "") -> None:
        """Recompute the publish state and emit it.

        Args:
            drain_state: The last drain's outcome, or ``DRAIN_IDLE``.
            detail: Failure text to carry to the GUI, or ``""``.
        """
        pending = self.pending_count()
        if not self._settings.enabled:
            state = PUBLISH_DISABLED
        elif drain_state == DRAIN_RETRY:
            state = PUBLISH_OFFLINE
        elif pending:
            state = PUBLISH_PENDING
        else:
            state = PUBLISH_SYNCED
        self._state = state
        self.publish_state_changed.emit(
            {"state": state, "pending": pending, "detail": detail}
        )

