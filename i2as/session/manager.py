"""ExperimentManager — the L6 façade and single writer of experiment state."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QObject, pyqtSignal

from i2as.core.events import OPERATOR, Actor, RunStarted
from i2as.core.orchestrator_proxy import OrchestratorProxy
from i2as.core.plan import ExperimentEnvelope, params_digest
from i2as.core.config import read_instrument_metadata
from i2as.core.station import Station
from i2as.session.models import (
    EXPERIMENT_STATUS_CLOSED,
    EXPERIMENT_STATUS_OPEN,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    SCHEMA_VERSION,
    ElnLink,
    ExperimentIndexEntry,
    ExperimentRecord,
    RunRecord,
    envelope_from_dict,
    envelope_to_dict,
)
from i2as.session.run_queue import (
    KIND_PROCEDURE,
    RunQueue,
    RunQueueHost,
    RunSpec,
    RunValidation,
)
from i2as.session.store import ExperimentStore, SessionStore, UserRoster

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class ExperimentManager(QObject):
    """The session layer's façade — the only object main/GUI (and later the
    Agent Gateway) talk to.

    Single-writer principle, one level up from the Orchestrator: exactly as
    all hardware writes flow through the Orchestrator, all experiment-record
    writes flow through this class. The GUI never edits records itself — it
    calls the lifecycle methods and renders the signals.

    Signals:
        experiment_changed (dict): The current experiment as a JSON-safe dict
            (``ExperimentRecord.to_dict()``), or ``{}`` when none is open.
            Emitted on start/close/resume and on every recorded mutation.
        run_recorded (dict): One ``RunRecord`` as a dict, emitted when a run
            is opened by a ``run_started`` manifest and again when its
            ``run_finished`` manifest completes it.
        store_health_changed (dict): ``{"ok": bool, "detail": str}``, emitted
            by ``_save_current()`` the first time a save fails (``ok=False``,
            ``detail`` the ``OSError`` text) and again the first time a save
            succeeds after failures (``ok=True``). One boolean of internal
            state; no retry machinery.
    """

    experiment_changed = pyqtSignal(dict)
    run_recorded = pyqtSignal(dict)
    store_health_changed = pyqtSignal(dict)

    def __init__(
        self,
        store: ExperimentStore,
        roster: UserRoster,
        orchestrator: OrchestratorProxy,
        config_name: str = "",
        config_path: str | None = None,
        session_store: SessionStore | None = None,
        station: Station | None = None,
        run_catalog: Mapping[str, type] | None = None,
    ) -> None:
        """Wire into the Orchestrator and resume any active experiment.

        Args:
            store: The experiment store (normally rooted in the data dir).
            roster: The setup-local user roster.
            orchestrator: The active Orchestrator; its run manifests drive run
                recording, and its ``set_experiment_envelope()`` receives the
                experiment's envelope.
            config_name: Identity of the active config, recorded on new
                experiments.
            config_path: Directory of the active config, read once for each
                VI's optional ``metadata:`` block (``read_instrument_metadata``).
                ``None`` (e.g. in unit tests) just means no instrument
                metadata is stamped — never an error.
            session_store: The Session-tier store, used to reconcile the
                active session's ``experiments`` index against its folder on
                ``start_experiment()``/``close_experiment()``/
                ``switch_experiment()`` (see ``_reconcile_session_index()``).
                ``None`` (e.g. in unit tests that only exercise the
                Experiment tier) simply skips index maintenance — every
                other feature works unchanged. When given, the session
                identity is derived from ``store.root``'s own two path
                segments (``sessions/<user_id>/<session_id>``), not passed
                separately, so there is no second source of truth to drift.
            station: The Station a queued run would drive — needed to build a
                run headlessly for ``validate_run()`` and to construct the one
                live object the engine pulls. ``None`` (a unit test that only
                exercises the experiment tier) leaves every other feature
                working; the two that need it say so.
            run_catalog: ``{class __name__: procedure class}``, the
                catalog a queued **run spec**'s class name is resolved
                through. Supplied by whoever owns discovery, because this
                package may not import ``i2as.procedures`` (contract C11).
        """
        super().__init__()
        self._store = store
        self._roster = roster
        self._orchestrator = orchestrator
        self._config_name = config_name
        self._instrument_metadata = (
            read_instrument_metadata(config_path) if config_path else {}
        )
        self._session_store = session_store
        self._experiment: ExperimentRecord | None = None
        self._queue_host = RunQueueHost(
            station=station,
            run_catalog=run_catalog,
            publish=self._publish_queue,
            experiment_info=self.experiment_context,
            envelope=self._current_envelope,
        )
        self._store_save_ok = True
        self._eln_publisher: Any | None = None

        orchestrator.run_started.connect(self._on_run_started)
        orchestrator.run_finished.connect(self._on_run_finished)
        # The one event stream, under whichever name this client carries it:
        # an `OrchestratorProxy` (what the application actually holds) renames
        # the engine's `event_emitted` to `event`, the same way it renames
        # `verdict_emitted` to `verdict`. The engine's name is tried FIRST,
        # because `event` is also `QObject`'s own virtual event handler and
        # every QObject therefore answers to it.
        event_stream = getattr(orchestrator, "event_emitted", None)
        if event_stream is None:
            event_stream = orchestrator.event
        event_stream.connect(self._on_engine_event)

        self._resume_active_experiment()

    # ------------------------------------------------------------------
    # Read surface
    # ------------------------------------------------------------------

    @property
    def store(self) -> ExperimentStore:
        """The underlying experiment store."""
        return self._store

    @property
    def roster(self) -> UserRoster:
        """The setup-local user roster."""
        return self._roster

    def current_experiment(self) -> ExperimentRecord | None:
        """Return the open experiment record, or ``None``."""
        return self._experiment

    def envelope_variables(self) -> dict[str, dict[str, Any]]:
        """Return each enveloped quantity and the setup's own bounds on it.

        The read side of the envelope this layer installs: the Start
        Experiment dialog pre-fills its envelope editor from these bounds, so
        the operator NARROWS what the setup already allows instead of composing
        an envelope from nothing. A passthrough to the Orchestrator's public
        API — the GUI has no Orchestrator of its own here, and this manager
        already owns the envelope's write side
        (``set_experiment_envelope()``), so the read side belongs beside it.

        Returns:
            ``{vi_name: rendered EnvelopeVariable}`` as the proxy's mirror
            answers it; empty when no VI declares a setpoint capability.
        """
        return self._orchestrator.envelope_variables()

    def experiment_context(self) -> dict[str, Any]:
        """Return the two-tier context dict stamped into every run's metadata.

        This is what the GUI passes as ``experiment_info`` when constructing a
        procedure, ending up whole (as one JSON blob) in
        ``/metadata/experiment_info``. Nests the two tiers this layer knows
        about — Setup (the config/instrument identity, true regardless of
        whether an experiment is open) and Experiment (this named group of
        runs, empty when none is open). The third tier, per-run measurement
        metadata, is stamped separately by the procedure itself.

        Returns:
            ``{"setup": {"config_name": ..., "instruments": {vi_name: {...}}},
            "experiment": {...} or {}}``. The experiment sub-dict, when
            present, has ``experiment_id``, ``experiment_title``, ``user_id``,
            ``user_name``, ``attended``, and ``eln_link`` (``{}`` until
            published).
        """
        setup = {
            "config_name": self._config_name,
            "instruments": dict(self._instrument_metadata),
        }
        if self._experiment is None:
            return {"setup": setup, "experiment": {}}
        user = self._roster.get(self._experiment.user_id)
        experiment = {
            "experiment_id": self._experiment.experiment_id,
            "experiment_title": self._experiment.title,
            "user_id": self._experiment.user_id,
            "user_name": user.name if user else "",
            "attended": self._experiment.attended,
            "eln_link": (
                self._experiment.eln_link.to_dict()
                if self._experiment.eln_link is not None
                else {}
            ),
        }
        return {"setup": setup, "experiment": experiment}

    # ------------------------------------------------------------------
    # Experiment lifecycle
    # ------------------------------------------------------------------

    def start_experiment(
        self,
        title: str,
        user_id: str,
        sample_info: dict[str, Any],
        envelope: ExperimentEnvelope | None = None,
        attended: bool = True,
        experiment_dirname: str | None = None,
    ) -> ExperimentRecord:
        """Open a new experiment and install its policy on the Orchestrator.

        Two session-owned policy values are pushed down as values here, for
        the same reason (contract C12): the **Session envelope** and
        **Attendance**. Both are re-installed by every other path that makes
        a record live — ``switch_experiment`` and the resume on construction.

        Args:
            title: Human title (also slugged into the experiment id when
                ``experiment_dirname`` is not given).
            user_id: Roster key of the person running the experiment.
            sample_info: The sample fields to snapshot onto the record.
            envelope: Optional per-experiment sample bounds, enforced by the
                Orchestrator for every writer until the experiment closes.
            attended: Initial attendance flag.
            experiment_dirname: Optional override for the experiment's
                folder name (and therefore its ``experiment_id``), directly
                under the session folder — flat only, no nesting. ``None``
                (the default) falls back to
                ``self._store.make_experiment_id(title, created)``.

        Returns:
            The persisted, now-active ``ExperimentRecord``.

        Raises:
            ValueError: If ``title`` is empty, another experiment is open,
                ``user_id`` is not in the roster, or ``experiment_dirname``
                is given but is empty, contains a path separator, is
                ``"."``/``".."``, or collides with an existing experiment
                folder in this session.
            OSError: If the record cannot be written.
        """
        if not title.strip():
            raise ValueError("Experiment title must not be empty")
        if self._experiment is not None:
            raise ValueError(
                f"Experiment {self._experiment.experiment_id!r} is still open; "
                "close it before starting a new one."
            )
        if self._roster.get(user_id) is None:
            raise ValueError(
                f"Unknown user {user_id!r} — add the user to the roster first."
            )
        created = _utc_now_iso()
        experiment_id = self._resolve_experiment_id(title, created, experiment_dirname)
        record = ExperimentRecord(
            experiment_id=experiment_id,
            title=title.strip(),
            user_id=user_id,
            sample_info=dict(sample_info),
            config_name=self._config_name,
            created_utc=created,
            status=EXPERIMENT_STATUS_OPEN,
            attended=attended,
            envelope=envelope_to_dict(envelope),
        )
        self._store.save(record)
        self._store.set_active(record.experiment_id)
        self._experiment = record
        self._orchestrator.set_experiment_envelope(envelope)
        self._orchestrator.set_attendance(record.attended)
        logger.info(
            "Experiment %s started (user=%s, attended=%s)",
            record.experiment_id,
            user_id,
            attended,
        )
        self._reconcile_session_index()
        self.experiment_changed.emit(record.to_dict())
        return record

    def _resolve_experiment_id(
        self, title: str, created_utc: str, experiment_dirname: str | None
    ) -> str:
        """Return the experiment id to use — auto-derived or user-chosen.

        Args:
            title: The experiment title (used for the auto-derived id).
            created_utc: ISO 8601 creation time (used for the auto-derived id).
            experiment_dirname: The caller's override, or ``None`` for the
                default auto-derived id.

        Returns:
            A valid, non-colliding experiment id.

        Raises:
            ValueError: If ``experiment_dirname`` is given but invalid (see
                ``start_experiment``'s docstring for the exact rules).
        """
        if experiment_dirname is None:
            return self._store.make_experiment_id(title, created_utc)
        candidate = experiment_dirname.strip()
        if not candidate:
            raise ValueError("Experiment folder name must not be empty")
        # Both separators are rejected on every platform, deliberately not via
        # os.sep/os.altsep: an experiment folder written on Linux is routinely
        # opened on a Windows analysis machine, where a backslash in the name
        # would split into a nested path. Keying off the host's separators let
        # "a\b" through on Linux (os.sep="/", os.altsep=None) while rejecting
        # it on Windows — the same name, two different verdicts.
        if "/" in candidate or "\\" in candidate:
            raise ValueError(
                f"Experiment folder name {experiment_dirname!r} must not contain "
                "a path separator — it names a single folder directly under "
                "the session, not a nested path"
            )
        if candidate in (".", ".."):
            raise ValueError(f"Experiment folder name {experiment_dirname!r} is not allowed")
        if candidate in self._store.list_experiments():
            raise ValueError(
                f"An experiment folder named {candidate!r} already exists in this session"
            )
        return candidate

    def close_experiment(self) -> None:
        """Close the open experiment and clear the envelope. No-op when none."""
        if self._experiment is None:
            return
        self._experiment.status = EXPERIMENT_STATUS_CLOSED
        self._experiment.closed_utc = _utc_now_iso()
        self._save_current()
        self._store.set_active(None)
        self._orchestrator.set_experiment_envelope(None)
        logger.info("Experiment %s closed", self._experiment.experiment_id)
        self._reconcile_session_index()
        self._experiment = None
        self.experiment_changed.emit({})

    def set_experiment_envelope(
        self, envelope: ExperimentEnvelope | None
    ) -> str:
        """Replace the open experiment's **Session envelope**. No-op when none.

        The write side of the envelope, and the counterpart to
        ``envelope_variables()``: the operator narrows the setup's limits at
        the experiment header, and the new bounds have to reach two places —
        the record (so they survive a restart and describe what this
        experiment was actually bounded by) and the Orchestrator (which is
        the only enforcement point). Both go through here for the same
        reason ``set_attended()`` does: this layer is the single writer for
        the record, and the engine cannot read it.

        Args:
            envelope: The new envelope, or ``None`` to clear it.

        Returns:
            The engine command's request id, or ``""`` when no experiment is
            open (nothing to bound, and nothing written).
        """
        if self._experiment is None:
            logger.warning("No experiment is open — the envelope was not applied")
            return ""
        self._experiment.envelope = envelope_to_dict(envelope)
        self._save_current()
        request_id = str(
            self._orchestrator.set_experiment_envelope(envelope) or ""
        )
        logger.info(
            "Experiment %s envelope %s",
            self._experiment.experiment_id,
            "cleared" if envelope is None else "updated",
        )
        self.experiment_changed.emit(self._experiment.to_dict())
        return request_id

    def set_findings(self, text: str) -> None:
        """Replace the experiment's free-text findings. No-op when none open.

        Args:
            text: The findings text (markdown).
        """
        if self._experiment is None:
            return
        self._experiment.findings = text
        self._save_current()
        self.experiment_changed.emit(self._experiment.to_dict())

    def set_attended(self, attended: bool) -> None:
        """Set the attendance flag. No-op when no experiment is open.

        **Attendance** is an input to the agent gateway's permission matrix
        (GLOSSARY.md, and ``session/gateway/README.md``): a ``debug`` role
        may take **Action class** ``recovery`` only while UNATTENDED; with a
        human present it diagnoses and reports instead. Recorded here so the
        flag survives a restart, and pushed down into the engine as a value
        by ``Orchestrator.set_attendance()``, since contract C12 stops the
        enforcement point from reading this record.

        Nothing is pushed down for a value the record already holds. That is
        safe because every path that makes a record live — ``start_experiment``,
        ``switch_experiment``, the resume on construction — installs its
        attendance on the engine the same way it installs the envelope, so
        the two can never be out of step to begin with.

        Args:
            attended: ``True`` when a human is present at the setup.
        """
        if self._experiment is None or self._experiment.attended == attended:
            return
        self._experiment.attended = attended
        self._save_current()
        self._orchestrator.set_attendance(attended)
        logger.info(
            "Experiment %s attendance: %s",
            self._experiment.experiment_id,
            "attended" if attended else "unattended",
        )
        self.experiment_changed.emit(self._experiment.to_dict())

    def set_queue(self, items: list[dict[str, Any]]) -> None:
        """Replace the open experiment's run queue. No-op when none is open.

        The queue is GUI-authored, opaque JSON — this layer stores and
        round-trips it but never interprets its shape (the GUI's
        ``QueueItemState`` is the only place that knows it; contract C11
        forbids this package from importing ``i2as.gui``).

        Args:
            items: The queue items, each an opaque JSON-safe dict.
        """
        if self._experiment is None:
            return
        self._experiment.queue = items
        self._save_current()

    def switch_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Switch to a different **open** experiment without closing the current one.

        Deactivates the current in-memory experiment by simply ceasing to
        track it — its own record is left exactly as last saved (still
        ``status == "open"`` on disk); ``close_experiment()``'s
        finalize-and-prompt-findings semantics are untouched and remain the
        only way to actually close an experiment. Re-installs the target's
        envelope on the Orchestrator the same way ``start_experiment``/
        ``_resume_active_experiment`` do, and updates the store's active
        pointer.

        Args:
            experiment_id: The store key of an open experiment to switch to.

        Returns:
            The newly active ``ExperimentRecord``.

        Raises:
            ValueError: If ``experiment_id`` is unknown, its record's
                ``status`` is not ``"open"``, or its ``schema_version`` is
                newer than this app's ``SCHEMA_VERSION`` — a future-format
                record must never become the live, mutable experiment of an
                older app.
        """
        record = self._store.load(experiment_id)
        if record is None:
            raise ValueError(f"Unknown experiment {experiment_id!r}")
        if record.status != EXPERIMENT_STATUS_OPEN:
            raise ValueError(
                f"Experiment {experiment_id!r} is not open (status={record.status!r})"
            )
        if record.schema_version > SCHEMA_VERSION:
            raise ValueError(
                f"Experiment {experiment_id!r} was written by a newer app "
                f"(schema_version={record.schema_version} > {SCHEMA_VERSION}); "
                "refusing to switch to it"
            )
        self._experiment = record
        self._store.set_active(record.experiment_id)
        self._orchestrator.set_experiment_envelope(envelope_from_dict(record.envelope))
        self._orchestrator.set_attendance(record.attended)
        logger.info("Switched to experiment %s", record.experiment_id)
        self._reconcile_session_index()
        self.experiment_changed.emit(record.to_dict())
        return record

    def current_data_dir(self) -> Path | None:
        """Return the open experiment's data folder, or ``None`` when none is open."""
        if self._experiment is None:
            return None
        return self._store.data_dir(self._experiment.experiment_id)

    def current_gui_state_path(self) -> Path | None:
        """Return the open experiment's GUI-state file path, or ``None`` when none is open."""
        if self._experiment is None:
            return None
        return self._store.gui_state_path(self._experiment.experiment_id)

    # ------------------------------------------------------------------
    # The run queue (see session/run_queue.py and GLOSSARY.md's Run queue)
    # ------------------------------------------------------------------

    @property
    def run_queue(self) -> RunQueue:
        """The ordered queue of runs waiting to start.

        Read-only in practice: mutate it through the methods below, which
        validate, log and broadcast. A client that only needs to render the
        queue reads ``queue_snapshot()`` (or the ``QueueChanged`` events the
        mutations emit) rather than holding this object.
        """
        return self._queue_host.queue

    @property
    def run_queue_host(self) -> RunQueueHost:
        """The queue plus its policy, for a client that owns the whole surface.

        The Procedure window's queue panel holds this rather than reaching
        back through the manager for each call; everything it offers is also
        available as a method here.
        """
        return self._queue_host

    def queue_snapshot(self) -> tuple[RunSpec, ...]:
        """Return every waiting **run spec**, in the order they will start."""
        return self._queue_host.snapshot()

    def queue_entries(self) -> tuple[dict[str, Any], ...]:
        """Return ``queue_snapshot()`` as JSON-safe dicts.

        Wired to ``Orchestrator.queue_snapshot`` so the engine can put the
        whole queue into every ``QueueChanged`` without knowing what a
        ``RunSpec`` is.
        """
        return self._queue_host.entries()

    def validate_run(
        self,
        procedure_cls: type,
        params: Mapping[str, Any],
        *,
        kind: str = KIND_PROCEDURE,
        sample_info: Mapping[str, Any] | None = None,
        data_directory: str = "",
        file_prefix: str = "",
        probe_spec: Mapping[str, Any] | None = None,
    ) -> RunValidation:
        """Decide whether a proposed run may be queued — free, and with no effect.

        The L6 entry point for **run validation**. What this layer adds over
        the bare check is the two things only it knows: the Station to build
        against and the OPEN EXPERIMENT'S envelope, both wired into the queue
        host at construction. The run is built headlessly and thrown away —
        nothing dispatched, no data file opened — so an operator (or an
        agent) learns at the moment of queueing that a value is out of
        bounds, instead of an hour later when the run would have started.

        Args:
            procedure_cls: The procedure class to check.
            params: The parameter values it would run with.
            kind: ``"procedure"``.
            sample_info: Sample metadata the run would record.
            data_directory: Directory the run would write into. Never created
                or written here.
            file_prefix: Filename prefix the run would use.
            probe_spec: A ``ProbeSpec``'s dict form to check the **probe
                run** variant instead of the full run.

        Returns:
            A ``RunValidation``; ``ok`` is True exactly when nothing was found.

        Raises:
            RuntimeError: If this manager was built without a Station, which
                makes a headless build impossible.
        """
        return self._queue_host.validate(
            procedure_cls,
            params,
            kind=kind,
            sample_info=sample_info,
            data_directory=data_directory,
            file_prefix=file_prefix,
            probe_spec=probe_spec,
        )

    def queue_run(
        self,
        procedure_cls: type,
        params: Mapping[str, Any],
        *,
        kind: str = KIND_PROCEDURE,
        sample_info: Mapping[str, Any] | None = None,
        data_directory: str = "",
        file_prefix: str = "",
        probe_spec: Mapping[str, Any] | None = None,
        actor: Actor = OPERATOR,
    ) -> tuple[RunSpec | None, RunValidation]:
        """Validate a proposed run and, if it passes, queue it.

        Validation happens HERE, at add time — a spec that fails never enters
        the queue, so nothing waiting in it is known to be unrunnable.

        Args:
            procedure_cls: The procedure class to queue.
            params: The parameter values it will run with.
            kind: ``"procedure"``.
            sample_info: Sample metadata to record with the run.
            data_directory: Directory the run writes into.
            file_prefix: Optional filename prefix.
            probe_spec: A ``ProbeSpec``'s dict form to queue the **probe
                run** variant of this run.
            actor: Who is queueing it.

        Returns:
            ``(spec, validation)`` — *spec* is the queued ``RunSpec``, or
            ``None`` when validation refused it and *validation.findings*
            says why.
        """
        return self._queue_host.add(
            procedure_cls,
            params,
            kind=kind,
            sample_info=sample_info,
            data_directory=data_directory,
            file_prefix=file_prefix,
            probe_spec=probe_spec,
            actor=actor,
        )

    def dequeue_run(self, spec_id: str, *, actor: Actor = OPERATOR) -> bool:
        """Remove one waiting run from the queue.

        Args:
            spec_id: The entry's ``RunSpec.spec_id``.
            actor: Who is removing it.

        Returns:
            True if an entry was removed.
        """
        return self._queue_host.remove(spec_id, actor=actor)

    def move_queued_run(
        self, spec_id: str, offset: int, *, actor: Actor = OPERATOR
    ) -> bool:
        """Move one waiting run within its own half of the queue.

        Args:
            spec_id: The entry's ``RunSpec.spec_id``.
            offset: Places to move it — negative towards the front.
            actor: Who is reordering it.

        Returns:
            True if the order changed.
        """
        return self._queue_host.move(spec_id, offset, actor=actor)

    def clear_run_queue(self, *, actor: Actor = OPERATOR) -> bool:
        """Empty the run queue.

        Args:
            actor: Who is clearing it.

        Returns:
            True if anything was removed.
        """
        return self._queue_host.clear(actor=actor)

    def next_run(self) -> Any:
        """Build and return the run the engine should start next.

        The **pull seam**'s other end, wired to
        ``Orchestrator.next_procedure``: the engine asks, this pops one spec
        and constructs the single live object it describes, stamped with the
        experiment context read HERE, at build time — so a run queued before
        an experiment was opened still belongs to the one open when it runs.

        Returns:
            A ready procedure, or ``None`` when the queue is
            empty or this manager has no Station/catalog to build with.

        Raises:
            KeyError: If the run catalog holds no class of the spec's name.
            I2ASError: If the run refuses to be built.
            TypeError: If the stored parameters no longer fit the signature.
            ValueError: If a parameter value is invalid.
        """
        return self._queue_host.next_run()

    def take_next_spec(self) -> Any:
        """Pop the next waiting spec, without building it.

        The client-thread half of the pull seam when the engine lives on the
        instrument thread: popping mutates this layer's queue, so it happens
        here, and what crosses to the engine is a frozen ``RunSpec``. See
        ``RunQueueHost.take_next_spec()``.

        Returns:
            The spec that just left the queue, or ``None`` when nothing is
            waiting.
        """
        return self._queue_host.take_next_spec()

    def build_spec(self, spec: Any) -> Any:
        """Build the live run a popped spec describes.

        The engine-thread half of the pull seam: it touches the Station, so it
        runs wherever the Station does. See ``RunQueueHost.build_spec()``.

        Args:
            spec: A spec ``take_next_spec()`` returned.

        Returns:
            A ready procedure.

        Raises:
            KeyError: If the run catalog holds no class of the spec's name.
            I2ASError: If the run refuses to be built.
            TypeError: If the stored parameters no longer fit the signature.
            ValueError: If a parameter value is invalid.
        """
        return self._queue_host.build_spec(spec)

    def _current_envelope(self) -> ExperimentEnvelope | None:
        """Return the open experiment's envelope, or ``None`` when none is open."""
        if self._experiment is None:
            return None
        return envelope_from_dict(self._experiment.envelope)

    def _publish_queue(self, actor: Actor) -> None:
        """Ask the Orchestrator to broadcast the queue as it now stands.

        The queue lives here, not in the engine, so the engine cannot see a
        change happen — but ``QueueChanged`` belongs on the one event stream
        every client already listens to, not on a second channel of this
        layer's own.

        Args:
            actor: Who caused the change.
        """
        self._orchestrator.publish_queue(actor=actor)

    # ------------------------------------------------------------------
    # Run recording (driven by the Orchestrator's manifests)
    # ------------------------------------------------------------------

    def _on_run_started(self, manifest: dict) -> None:
        """Open a ``RunRecord`` for a ``run_started`` manifest.

        Runs outside an experiment are not recorded — there is no record to
        attach them to (their HDF5 file still exists, unstamped).

        The **Params digest** is stamped here, from the manifest's own
        parameters, so what the run started with is fixed at the moment it
        started rather than recomputed from a record that may since have been
        amended.
        """
        if self._experiment is None:
            return
        raw_data_file = str(manifest.get("data_file", ""))
        data_file = (
            self._store.relativize_data_file(self._experiment.experiment_id, raw_data_file)
            if raw_data_file
            else ""
        )
        params = dict(manifest.get("params") or {})
        run = RunRecord(
            run_id=str(manifest.get("run_id", "")),
            procedure=str(manifest.get("procedure", "")),
            kind=str(manifest.get("kind", "run")),
            params=params,
            params_digest=params_digest(params),
            data_file=data_file,
            started_utc=str(manifest.get("started_utc", "")),
            status=RUN_STATUS_RUNNING,
        )
        self._experiment.runs.append(run)
        self._save_current()
        self.run_recorded.emit(run.to_dict())

    def _on_engine_event(self, event: object) -> None:
        """Stamp who started a run onto the record the manifest just opened.

        The manifest says WHAT ran; the contract's ``RunStarted`` event says
        who asked, and it is the only place that fact exists — so the record
        is completed from the event rather than from the manifest. The two
        arrive in that order (the engine emits the manifest signal first,
        then the event), which is what lets this find the record already
        there instead of racing it.

        Nothing else on the event stream concerns this layer; a run that
        started outside an experiment, or whose actor is already what the
        record says, writes nothing.

        Args:
            event: Anything on the Orchestrator's one event stream.
        """
        if not isinstance(event, RunStarted) or self._experiment is None:
            return
        run = self._experiment.find_run(event.run_id)
        if run is None or (run.actor == event.actor and not run.actor_legacy):
            return
        run.actor = event.actor
        run.actor_legacy = False
        self._save_current()
        self.run_recorded.emit(run.to_dict())

    def _on_run_finished(self, manifest: dict) -> None:
        """Complete the matching ``RunRecord`` from a ``run_finished`` manifest."""
        if self._experiment is None:
            return
        run = self._experiment.find_run(str(manifest.get("run_id", "")))
        if run is None:
            logger.warning(
                "run_finished for unknown run %r — ignored", manifest.get("run_id")
            )
            return
        run.finished_utc = str(manifest.get("finished_utc", ""))
        run.status = str(manifest.get("status", RUN_STATUS_FAILED))
        run.reason = str(manifest.get("reason", ""))
        self._save_current()
        self.run_recorded.emit(run.to_dict())

    def set_run_eln_link(self, experiment_id: str, run_id: str, link: ElnLink) -> bool:
        """Record the ELN entry one run was published to.

        The single-writer rule applied to the publishing track: the publisher
        never edits a record or touches the store itself, it hands the
        confirmed entry reference here. Called only after the backend has
        confirmed the entry, so a run that carries a link really is in the
        notebook.

        Works on any experiment in this store, not just the open one: an
        outbox job queued today may only reach the notebook next week, by
        which time its experiment is closed and something else is open. When
        the target IS the open experiment the in-memory record is updated and
        ``run_recorded`` is emitted; otherwise the record is loaded, amended,
        and saved without disturbing the live one.

        Args:
            experiment_id: The store key of the experiment owning the run.
            run_id: The run to stamp.
            link: The confirmed ELN entry reference.

        Returns:
            ``True`` when the link was recorded, ``False`` when the
            experiment or the run is unknown, or the record could not be
            written (all logged, never raised — a bookkeeping failure must
            not propagate into a GUI timer).
        """
        if self._experiment is not None and self._experiment.experiment_id == experiment_id:
            run = self._experiment.find_run(run_id)
            if run is None:
                logger.warning("No run %r in the open experiment to stamp an ELN link on", run_id)
                return False
            run.eln_link = link
            run.published = True
            self._save_current()
            self.run_recorded.emit(run.to_dict())
            logger.info("Run %s published to %s", run_id, link.url or link.entry_id)
            return True

        record = self._store.load(experiment_id)
        if record is None:
            logger.warning("Unknown experiment %r — cannot record its ELN link", experiment_id)
            return False
        run = record.find_run(run_id)
        if run is None:
            logger.warning("No run %r in experiment %r to stamp an ELN link on", run_id, experiment_id)
            return False
        if record.schema_version > SCHEMA_VERSION:
            logger.warning(
                "Refusing to stamp an ELN link on experiment %s: its schema_version=%d > %d",
                experiment_id,
                record.schema_version,
                SCHEMA_VERSION,
            )
            return False
        run.eln_link = link
        run.published = True
        try:
            self._store.save(record)
        except OSError as exc:
            logger.error("Could not record the ELN link for run %s: %s", run_id, exc)
            return False
        logger.info(
            "Run %s of closed experiment %s published to %s",
            run_id,
            experiment_id,
            link.url or link.entry_id,
        )
        return True

    # ------------------------------------------------------------------
    # The drafting approval gate
    # ------------------------------------------------------------------

    def attach_eln_publisher(self, publisher: Any | None) -> None:
        """Hold the ELN publisher an approved **draft entry** is enqueued through.

        The one seam between the manager and the publishing track, and it
        points the way the publisher does not: the publisher already holds
        this manager (it hands confirmed links to ``set_run_eln_link()``), so
        approval — a decision about a *record* — is taken here and the
        enqueue is delegated back. Duck-typed on ``export_draft(run_id,
        draft)`` rather than imported, so this module stays free of the ELN
        package and of the Qt object that owns its drain timer.

        Args:
            publisher: The ``ElnPublisher``, or ``None`` to detach. With none
                attached, ``approve_eln_draft()`` refuses by saying so.
        """
        self._eln_publisher = publisher
        logger.info(
            "ELN publisher %s for draft approval",
            "attached" if publisher is not None else "detached",
        )

    def set_pending_eln_draft(self, run_id: str, draft: Mapping[str, Any]) -> bool:
        """Park a **draft entry** on one run of the open experiment, unapproved.

        The single-writer rule applied to the drafting track: an agent that
        drafts an entry for an ATTENDED experiment may not publish it, so the
        draft is stored here — as JSON on the run record — until a human
        approves it with ``approve_eln_draft()``. Storing one replaces
        whatever was pending, because a draft is a proposal that can be
        redrawn at any time and only the newest is of interest.

        Args:
            run_id: The run the draft describes, in the open experiment.
            draft: The draft as its JSON dict (``DraftEntry.to_dict()``).

        Returns:
            ``True`` when it was stored, ``False`` when no experiment is open
            or the run is unknown (logged, never raised).
        """
        run = self._open_run(run_id, "park a draft on")
        if run is None:
            return False
        run.pending_eln_draft = dict(draft)
        self._save_current()
        self.run_recorded.emit(run.to_dict())
        logger.info("An ELN draft for run %s is waiting for approval", run_id)
        return True

    def pending_eln_draft(self, run_id: str) -> dict[str, Any]:
        """Return the **draft entry** waiting on one run, or ``{}``.

        Args:
            run_id: The run to read, in the open experiment.

        Returns:
            The pending draft's JSON dict, or ``{}`` when none is waiting (or
            no experiment is open, or the run is unknown).
        """
        run = self._open_run(run_id, "read a pending draft of")
        return {} if run is None else dict(run.pending_eln_draft)

    def discard_pending_eln_draft(self, run_id: str) -> bool:
        """Drop the **Pending entry** waiting on one run, publishing nothing.

        The other half of the approval gate: a proposal a human read and did
        not want. The run keeps its data and its record — only the proposed
        entry goes — so the run can be analysed or drafted again afterwards.

        Args:
            run_id: The run whose pending entry is discarded, in the open
                experiment.

        Returns:
            ``True`` when something was discarded; ``False`` when no
            experiment is open, the run is unknown, or nothing was pending
            (all logged, never raised: discarding is a GUI action).
        """
        run = self._open_run(run_id, "discard a pending entry of")
        if run is None:
            return False
        if not run.pending_eln_draft:
            logger.info("Run %s has no pending ELN entry to discard", run_id)
            return False
        run.pending_eln_draft = {}
        self._save_current()
        self.run_recorded.emit(run.to_dict())
        logger.info("Discarded the pending ELN entry for run %s", run_id)
        return True

    def approve_eln_draft(self, run_id: str) -> str:
        """Approve the **draft entry** waiting on one run and queue it.

        The human's half of the approval gate. The draft goes to the
        publisher's ``export_draft()``, which queues it as one ordinary
        outbox job; only once it is queued is the pending draft cleared, so a
        publisher that refused (publishing off, no experiment open) leaves the
        proposal exactly where it was, still approvable later.

        Args:
            run_id: The run whose pending draft is approved.

        Returns:
            The queued job's id, or ``""`` when nothing was queued — no
            experiment open, no such run, no draft pending, no publisher
            attached, or the publisher queued nothing (all logged, never
            raised: approval is a GUI action, and a bookkeeping failure must
            not propagate into it).
        """
        run = self._open_run(run_id, "approve a draft of")
        if run is None:
            return ""
        if not run.pending_eln_draft:
            logger.warning("Run %s has no pending ELN draft to approve", run_id)
            return ""
        if self._eln_publisher is None:
            logger.warning(
                "No ELN publisher is attached — the approved draft for run %s "
                "stays pending",
                run_id,
            )
            return ""
        draft = dict(run.pending_eln_draft)
        try:
            job_id = str(self._eln_publisher.export_draft(run_id, draft) or "")
        except Exception:  # noqa: BLE001 - approval must not raise into the GUI
            logger.exception("Queuing the approved draft for run %s failed", run_id)
            return ""
        if not job_id:
            logger.warning(
                "The publisher queued nothing for run %s — its draft stays pending",
                run_id,
            )
            return ""
        run.pending_eln_draft = {}
        self._save_current()
        self.run_recorded.emit(run.to_dict())
        logger.info("Approved the ELN draft for run %s: queued as %s", run_id, job_id)
        return job_id

    def _open_run(self, run_id: str, action: str) -> RunRecord | None:
        """Return one run of the OPEN experiment, or ``None`` with a warning.

        Args:
            run_id: The run to find.
            action: What the caller wanted to do, for the log line.

        Returns:
            The ``RunRecord``, or ``None`` when no experiment is open or the
            experiment has no such run.
        """
        if self._experiment is None:
            logger.warning("No experiment is open — cannot %s run %r", action, run_id)
            return None
        run = self._experiment.find_run(run_id)
        if run is None:
            logger.warning(
                "No run %r in the open experiment — cannot %s it", run_id, action
            )
        return run

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _current_session_identity(self) -> tuple[str, str] | None:
        """Return the ``(user_id, session_id)`` owning ``self._store``, or ``None``.

        ``None`` when no ``session_store`` was given at construction — the
        caller then knows to skip index maintenance entirely. Otherwise
        derived from ``self._store.root``'s own two path segments
        (``sessions/<user_id>/<session_id>``), never passed or cached
        separately, so this can never disagree with the store it describes.
        """
        if self._session_store is None:
            return None
        return self._store.root.parent.name, self._store.root.name

    def _reconcile_session_index(self) -> None:
        """Rebuild the active session's ``experiments`` index from its folder.

        Called after every ``start_experiment()``/``close_experiment()``/
        ``switch_experiment()`` — the three points an experiment becomes or
        stops being the one this manager is looking at. Rather than
        upserting just the one record that changed, this rescans
        ``self._store.list_experiments()`` (a live directory listing) and
        reads every ``experiment.json`` found there, replacing
        ``session.experiments`` wholesale. That is what makes moving an
        experiment folder by hand safe: an experiment folder moved OUT of
        this session (e.g. handed off to a different user's session to
        continue the project) drops out of the rebuilt list, and one moved
        IN is picked up, the next time any experiment in this session opens
        or closes — no separate "move" operation needs to touch the index
        itself. Each entry's ``user_id`` is copied verbatim from its
        ``ExperimentRecord`` — whoever originally ran that experiment stays
        on record regardless of which session folder it currently lives in;
        this method never rewrites it.

        Tolerates a missing/corrupt ``session.json``, an unreadable
        individual ``experiment.json`` (skipped, logged, the rest still
        reconcile), or a failed save — the index mirrors the experiment
        lifecycle, it must never be allowed to block it. No-op when this
        manager was built without a ``session_store``.
        """
        identity = self._current_session_identity()
        if identity is None:
            return
        user_id, session_id = identity
        session = self._session_store.load(user_id, session_id)
        if session is None:
            logger.warning(
                "Could not load session %s/%s to reconcile its experiment index",
                user_id,
                session_id,
            )
            return
        entries: list[ExperimentIndexEntry] = []
        for experiment_id in self._store.list_experiments():
            record = self._store.load(experiment_id)
            if record is None:
                logger.warning(
                    "Skipping unreadable experiment %r while reconciling "
                    "session %s/%s's index",
                    experiment_id,
                    user_id,
                    session_id,
                )
                continue
            entries.append(
                ExperimentIndexEntry(
                    experiment_id=record.experiment_id,
                    title=record.title,
                    user_id=record.user_id,
                    status=record.status,
                    created_utc=record.created_utc,
                    closed_utc=record.closed_utc,
                )
            )
        session.experiments = entries
        try:
            self._session_store.save(session)
        except OSError:
            logger.exception(
                "Could not save session %s/%s's reconciled experiment index",
                user_id,
                session_id,
            )

    def _save_current(self) -> None:
        """Persist the current record, tolerating write failures.

        A failed save must not crash a running measurement — it is logged,
        the in-memory record stays authoritative until the next save
        attempt, and ``store_health_changed`` tells the GUI so a stale disk
        copy is never silent (emitted once on the first failure, and once
        again on the first successful save after failures — no retry
        machinery, one boolean of internal state).

        A record whose ``schema_version`` is newer than this app's
        ``SCHEMA_VERSION`` is never written back — belt-and-suspenders for
        the read-only rule; such a record should never have become
        ``self._experiment`` in the first place (see ``switch_experiment``).
        """
        if self._experiment is None:
            return
        if self._experiment.schema_version > SCHEMA_VERSION:
            logger.warning(
                "Refusing to overwrite experiment %s: its schema_version=%d > %d",
                self._experiment.experiment_id,
                self._experiment.schema_version,
                SCHEMA_VERSION,
            )
            return
        try:
            self._store.save(self._experiment)
        except OSError as exc:
            logger.error("Could not save experiment %s: %s", self._experiment.experiment_id, exc)
            if self._store_save_ok:
                self._store_save_ok = False
                self.store_health_changed.emit({"ok": False, "detail": str(exc)})
            return
        if not self._store_save_ok:
            self._store_save_ok = True
            logger.info("Experiment %s save recovered", self._experiment.experiment_id)
            self.store_health_changed.emit({"ok": True, "detail": ""})

    def _resume_active_experiment(self) -> None:
        """Resume the store's active experiment on construction, if any.

        Runs left in ``running`` state (the app died mid-run) are marked
        failed — a record whose run cannot have survived the restart must not
        look like live work. The envelope stored on the record is re-installed
        on the Orchestrator.
        """
        active_id = self._store.get_active()
        if active_id is None:
            return
        record = self._store.load(active_id)
        if record is None or record.status != EXPERIMENT_STATUS_OPEN:
            logger.warning(
                "Active experiment %r missing or not open — clearing pointer",
                active_id,
            )
            try:
                self._store.set_active(None)
            except OSError:
                logger.exception("Could not clear the active-experiment pointer")
            return
        stale = [run for run in record.runs if run.status == RUN_STATUS_RUNNING]
        for run in stale:
            run.status = RUN_STATUS_FAILED
            run.reason = "application restarted while the run was in progress"
            run.finished_utc = run.finished_utc or _utc_now_iso()
        self._experiment = record
        if stale:
            self._save_current()
        self._orchestrator.set_experiment_envelope(
            envelope_from_dict(record.envelope)
        )
        self._orchestrator.set_attendance(record.attended)
        logger.info("Resumed experiment %s (%d runs)", record.experiment_id, len(record.runs))
        self.experiment_changed.emit(record.to_dict())
