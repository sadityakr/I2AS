"""ExperimentManager — the L6 façade and single writer of experiment state."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from i2as.core.events import OPERATOR, Actor, Command, CommandName, RunStarted, VerdictCode
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
    ElnBinding,
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
from i2as.core.run_naming import normalize_run_subfolder
from i2as.session.store import ExperimentStore, SessionStore, UserRoster

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping

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
    #: The folder every run is now written to (``""`` when none is open).
    run_folder_changed = pyqtSignal(str)
    #: A live session switch is about to re-root: save what belongs to the
    #: session being left (emitted once the engine has released the run folder).
    session_about_to_change = pyqtSignal()
    #: A live session switch completed: the new session folder.
    session_changed = pyqtSignal(str)
    #: A live session switch was refused or failed: the reason, for the operator.
    session_switch_failed = pyqtSignal(str)

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
        runs_held: str = "",
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
                folder IS ``store.root`` — the experiment store is rooted at
                the session folder — so there is no second source of truth
                to drift.
            station: The Station a queued run would drive — needed to build a
                run headlessly for ``validate_run()`` and to construct the one
                live object the engine pulls. ``None`` (a unit test that only
                exercises the experiment tier) leaves every other feature
                working; the two that need it say so.
            run_catalog: ``{class __name__: procedure class}``, the
                catalog a queued **run spec**'s class name is resolved
                through. Supplied by whoever owns discovery, because this
                package may not import ``i2as.procedures`` (contract C11).
            runs_held: Why this session's records must not be written —
                another running application holds the session folder. The
                session then opens with NO experiment adopted, every run is
                refused and every write to this session's records is refused,
                until another session is loaded. ``""`` (the default): not
                held.
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
        # Live session switching (request_session_switch): the extra "is
        # anything of this session still in flight?" checks other services
        # register (analysis, notebook publishing), and the switch waiting for
        # the engine's verdict on releasing the run folder.
        self._busy_checks: list[Callable[[], str]] = []
        self._pending_switch: tuple[str, Path, ExperimentStore] | None = None
        # Why this session is held ("" = not held): set when the session
        # folder is in use by another running application, so two stations
        # never write the same experiment records (see ``runs_held``).
        self._runs_held = runs_held

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
        # Verdicts, under the engine's name or the proxy's (see above): a live
        # session switch completes only on the engine's OK to release the run
        # folder.
        verdicts = getattr(orchestrator, "verdict_emitted", None)
        if verdicts is None:
            verdicts = getattr(orchestrator, "verdict", None)
        if verdicts is not None and hasattr(verdicts, "connect"):
            verdicts.connect(self._on_verdict)

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
            ``user_name``, ``attended``, and ``eln_link`` (the experiment's
            notebook page, ``{}`` while it has none).
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
                self._experiment.eln.entry.to_dict()
                if self._experiment.eln is not None and self._experiment.eln.entry is not None
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
            title: Human title (also slugged into the experiment id's label
                when ``experiment_dirname`` is not given).
            user_id: Roster key of the person running the experiment.
            sample_info: The sample fields to snapshot onto the record.
            envelope: Optional per-experiment sample bounds, enforced by the
                Orchestrator for every writer until the experiment closes.
            attended: Initial attendance flag.
            experiment_dirname: Optional label for the experiment's folder,
                directly under the session folder — flat only, no nesting.
                The folder is always ``NNN_<label>``: the session's next
                serial number, then this label slugged, or the title's when
                it is ``None`` (``ExperimentStore.make_experiment_id``).

        Returns:
            The persisted, now-active ``ExperimentRecord``.

        Raises:
            ValueError: If ``title`` is empty, another experiment is open,
                ``user_id`` is not in the roster, or ``experiment_dirname``
                is given but is empty, contains a path separator, or is
                ``"."``/``".."``.
            OSError: If the record cannot be written.
        """
        if self._runs_held:
            raise ValueError(f"This session is held: {self._runs_held}")
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
        self._install_run_folder(record.experiment_id)
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
            title: The experiment title (the label when no folder name is given).
            created_utc: ISO 8601 creation time (recorded on the experiment;
                the id itself is serial).
            experiment_dirname: The operator's folder label, or ``None``.

        Returns:
            A valid, non-colliding ``NNN_<label>`` experiment id.

        Raises:
            ValueError: If ``experiment_dirname`` is given but invalid (see
                ``start_experiment``'s docstring for the exact rules).
        """
        if experiment_dirname is None:
            return self._store.make_experiment_id(title)
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
        # The operator's name is the label; the serial number always comes
        # first, so every experiment in a session sorts in the order it was
        # started and no two can collide.
        return self._store.make_experiment_id(candidate)

    def close_experiment(self) -> None:
        """Close the open experiment and clear the envelope. No-op when none."""
        if self._experiment is None:
            return
        self._experiment.status = EXPERIMENT_STATUS_CLOSED
        self._experiment.closed_utc = _utc_now_iso()
        self._save_current()
        self._store.set_active(None)
        self._orchestrator.set_experiment_envelope(None)
        self._install_run_folder(None)
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
                ``status`` is not ``"open"``, something of the current
                experiment is still in flight (``session_busy_reason()``), or
                its ``schema_version`` is
                newer than this app's ``SCHEMA_VERSION`` — a future-format
                record must never become the live, mutable experiment of an
                older app.
        """
        if self._runs_held:
            raise ValueError(f"This session is held: {self._runs_held}")
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
        busy = self.session_busy_reason()
        if busy:
            raise ValueError(f"Cannot switch experiment now: {busy}")
        self._experiment = record
        self._store.set_active(record.experiment_id)
        self._orchestrator.set_experiment_envelope(envelope_from_dict(record.envelope))
        self._orchestrator.set_attendance(record.attended)
        self._install_run_folder(record.experiment_id)
        logger.info("Switched to experiment %s", record.experiment_id)
        self._reconcile_session_index()
        self.experiment_changed.emit(record.to_dict())
        return record

    def current_data_dir(self) -> Path | None:
        """Return the open experiment's data folder, or ``None`` when none is open."""
        if self._experiment is None:
            return None
        return self._store.data_dir(self._experiment.experiment_id)

    def run_folder(self) -> Path | None:
        """Return the folder every run is written to now, or ``None`` when none is open.

        The open experiment's ``data/`` folder, or the operator's run
        subfolder of it (``set_run_subfolder``).
        """
        data_dir = self.current_data_dir()
        if data_dir is None or self._experiment is None:
            return None
        subfolder = self._experiment.run_subfolder
        return data_dir / subfolder if subfolder else data_dir

    def set_run_subfolder(self, subfolder: str) -> str:
        """Choose the subfolder of the experiment's ``data/`` every run from now on writes to.

        Applies to every run placed afterwards — the operator's, a queued one,
        an agent's, a probe; a run already started keeps the file it has. The
        choice is stored on the experiment, so reopening it restores it.

        Args:
            subfolder: Relative to ``data/`` (``core.run_naming``'s rule);
                ``""`` for ``data/`` itself.

        Returns:
            The normalised subfolder.

        Raises:
            ValueError: If no experiment is open or the subfolder breaks the
                rule.
        """
        if self._runs_held:
            raise ValueError(f"This session is held: {self._runs_held}")
        if self._experiment is None:
            raise ValueError("No experiment is open")
        normalized = normalize_run_subfolder(subfolder)
        self._experiment.run_subfolder = normalized
        self._save_current()
        self._install_run_folder(self._experiment.experiment_id)
        logger.info("Run subfolder set: %r", normalized)
        return normalized

    # ------------------------------------------------------------------
    # Loading another session while running
    # ------------------------------------------------------------------

    def hold_runs(self, reason: str) -> None:
        """Hold this session: no experiment, no run, no record write, until another session loads.

        Prefer the constructor's ``runs_held`` — holding after construction
        comes after the active experiment was already adopted.

        Used when the session folder is in use by another running
        application: both would otherwise save the same experiment records,
        and the last save would silently drop the other's runs.

        Args:
            reason: Shown to the operator.
        """
        self._runs_held = reason
        logger.error("Runs held in this session: %s", reason)
        self._install_run_folder(None)
        if self._experiment is not None:
            self._experiment = None
            self.experiment_changed.emit({})

    def runs_held_reason(self) -> str:
        """Why runs are held in this session, or ``""`` when they are not."""
        return self._runs_held

    def add_busy_check(self, check: Callable[[], str]) -> None:
        """Register a "still in flight?" check consulted before a switch.

        Args:
            check: Returns ``""`` when idle, else a reason for the operator
                (e.g. ``"An analysis is running"``).
        """
        self._busy_checks.append(check)

    def session_busy_reason(self) -> str:
        """Return why the session or experiment cannot be switched now, or ``""``.

        A fast, client-side answer for the operator. It is NOT what makes a
        session switch safe — the engine's verdict on releasing the run
        folder is (``request_session_switch``) — and runs are filed by the
        folder they were placed in, never by whichever experiment is open.

        Returns:
            The first reason found, or ``""``.
        """
        if self._pending_switch is not None:
            return "A session switch is in progress"
        mirror = getattr(self._orchestrator, "status", None)
        run = getattr(mirror, "run", None)
        if callable(run):
            try:
                if run() is not None:
                    return "A run is in progress"
            except Exception:  # a mirror that cannot answer must not block the operator
                logger.debug("status mirror could not report the run", exc_info=True)
        for check in self._busy_checks:
            reason = check()
            if reason:
                return reason
        return ""

    def request_session_switch(self, folder: str | Path) -> None:
        """Load another session folder while the application runs.

        Two phases, so no run of the current session can start after the
        switch began and none can be filed into the new one:

        1. Here, on the caller's thread: the target is validated (a session
           folder, not the open one, not locked by another running
           application), nothing of this session may be in flight
           (``session_busy_reason``), and the target's lock is taken. Then
           the engine is asked to RELEASE the run folder — ``set_run_folder("",
           require_idle=True)`` — which it answers only after every command
           posted before it, and refuses if a run is active or queued in it.
        2. On the engine's OK verdict (``_on_verdict``): the run queue is
           parked (it is already saved with its experiment, so reopening that
           experiment brings it back), the registry names the new session
           (also ``sessions.json``, so ``i2as-ctl`` follows), the store is
           re-rooted, ``session_changed`` is emitted, and the new session's
           active experiment is adopted (``experiment_changed``).
           A refusal emits ``session_switch_failed`` and changes nothing.

        The open experiment stays open on disk; loading this session again
        resumes it. The caller saves its GUI state before calling.

        Args:
            folder: The session folder to load.

        Raises:
            ValueError: If session management is unavailable, *folder* is not
                a session or is the open one, or something is in flight.
            SessionLockedError: If another running application holds it.
        """
        if self._session_store is None:
            raise ValueError("Session management is not available")
        target = Path(folder).resolve()
        if target == self._store.root.resolve():
            raise ValueError(f"{target} is already the open session")
        busy = self.session_busy_reason()
        if busy:
            raise ValueError(f"Cannot load another session now: {busy}")
        if self._session_store.load(target) is None:
            raise ValueError(f"{target} is not a session folder")
        self._session_store.acquire_lock(target)
        command = Command(
            name=CommandName.SET_RUN_FOLDER,
            actor=OPERATOR,
            args={"data_directory": "", "require_idle": True},
        )
        self._pending_switch = (command.request_id, target, ExperimentStore(target))
        # A verdict that never comes (an engine wedged on a read) must not
        # leave every later switch refused: give up after a while.
        QTimer.singleShot(
            SWITCH_VERDICT_TIMEOUT_MS, lambda rid=command.request_id: self._abandon_switch(rid)
        )
        try:
            self._orchestrator.submit(command)
        except Exception:
            self._pending_switch = None
            self._session_store.release_lock(target)
            raise

    def _abandon_switch(self, request_id: str) -> None:
        """Give up a switch whose verdict never arrived; the session stays as it was."""
        pending = self._pending_switch
        if pending is None or pending[0] != request_id:
            return
        self._pending_switch = None
        assert self._session_store is not None
        self._session_store.release_lock(pending[1])
        self._reinstall_current_run_folder()
        self.session_switch_failed.emit(
            "Cannot load another session now: the station did not answer in time"
        )

    def _reinstall_current_run_folder(self) -> None:
        """Hand the engine this session's run folder again (after a refused or abandoned switch)."""
        current = self._experiment
        self._install_run_folder(current.experiment_id if current is not None else None)

    def _on_verdict(self, verdict: object) -> None:
        """Complete or abandon a pending session switch on the engine's verdict."""
        pending = self._pending_switch
        if pending is None or getattr(verdict, "request_id", None) != pending[0]:
            return
        self._pending_switch = None
        request_id, target, new_store = pending
        assert self._session_store is not None
        if getattr(verdict, "code", None) != VerdictCode.OK:
            self._session_store.release_lock(target)
            self._reinstall_current_run_folder()
            reason = getattr(verdict, "reason", "") or "the engine refused"
            logger.info("Session switch to %s refused: %s", target, reason)
            self.session_switch_failed.emit(f"Cannot load another session now: {reason}")
            return
        try:
            self._complete_session_switch(target, new_store)
        except Exception as exc:
            logger.exception("Session switch to %s failed", target)
            self._reinstall_current_run_folder()
            self.session_switch_failed.emit(f"Could not load {target}: {exc}")

    def _complete_session_switch(self, target: Path, new_store: ExperimentStore) -> None:
        """Re-root on the new session; the engine already refuses every run.

        Rolls back to the current session if the registry cannot be written
        (the only write before the store is swapped).
        """
        assert self._session_store is not None
        old_root = self._store.root
        # Last chance for the session being left to save what is its own
        # (the GUI's fields and queue, edited while the release was pending).
        self.session_about_to_change.emit()
        try:
            self._session_store.set_active(target)
        except (OSError, ValueError):
            self._session_store.release_lock(target)
            raise
        # Parked, not lost: the GUI saved it with its experiment.
        self._queue_host.clear()
        self._store = new_store
        self._experiment = None
        self._runs_held = ""  # the new session's lock is ours
        self._orchestrator.set_experiment_envelope(None)
        self._session_store.release_lock(old_root)
        logger.info("Session loaded: %s", target)
        # Announced BEFORE the new session's experiment is adopted, so every
        # listener drops what it cached for the old session first —
        # experiment ids repeat across sessions, and an "unchanged id" must
        # not look like no change.
        self.session_changed.emit(str(target))
        self._resume_active_experiment()
        if self._experiment is None:
            self.experiment_changed.emit({})
        self._reconcile_session_index()

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

    def _record_for(
        self, manifest: dict
    ) -> tuple[ExperimentStore, ExperimentRecord, bool] | None:
        """Find the record a run manifest belongs to, by the folder the run was placed in.

        A run is filed where the ENGINE placed it (the manifest's
        ``data_root``), never into whichever experiment happens to be open
        when the manifest arrives — so a run can never be misfiled by an
        experiment or session switch racing its start or finish.

        Args:
            manifest: A ``run_started``/``run_finished`` manifest.

        Returns:
            ``(store, record, is_current)``, or ``None`` when the run belongs
            to no experiment this layer can find.
        """
        root = str(manifest.get("data_root") or "")
        current = self._experiment
        if not root:
            # A run the engine did not place (no session layer at the time,
            # or a test double): the open experiment, as before.
            return (self._store, current, True) if current is not None else None
        if current is not None and _same_path(
            self._store.data_dir(current.experiment_id), root
        ):
            return self._store, current, True
        experiment_dir = Path(root).parent
        store = (
            self._store
            if _same_path(experiment_dir.parent, self._store.root)
            else ExperimentStore(experiment_dir.parent)
        )
        record = store.load(experiment_dir.name)
        if record is None:
            logger.warning("Run in %s belongs to no experiment record — not recorded", root)
            return None
        return store, record, False

    def _save_record(self, store: ExperimentStore, record: ExperimentRecord, is_current: bool) -> None:
        """Save a record found by ``_record_for`` — the open one through the usual path."""
        if is_current:
            self._save_current()
            return
        try:
            store.save(record)
        except OSError:
            logger.exception("Could not save run record of %s", record.experiment_id)

    def _on_run_started(self, manifest: dict) -> None:
        """Open a ``RunRecord`` for a ``run_started`` manifest.

        Runs outside an experiment are not recorded — there is no record to
        attach them to (their HDF5 file still exists, unstamped).

        The **Params digest** is stamped here, from the manifest's own
        parameters, so what the run started with is fixed at the moment it
        started rather than recomputed from a record that may since have been
        amended.
        """
        found = self._record_for(manifest)
        if found is None:
            return
        store, record, is_current = found
        raw_data_file = str(manifest.get("data_file", ""))
        data_file = (
            store.relativize_data_file(record.experiment_id, raw_data_file)
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
        record.runs.append(run)
        self._save_record(store, record, is_current)
        if is_current:
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
        if not isinstance(event, RunStarted):
            return
        found = self._record_for(dict(event.manifest or {}))
        if found is None:
            return
        store, record, is_current = found
        run = record.find_run(event.run_id)
        if run is None or (run.actor == event.actor and not run.actor_legacy):
            return
        run.actor = event.actor
        run.actor_legacy = False
        self._save_record(store, record, is_current)
        if is_current:
            self.run_recorded.emit(run.to_dict())

    def _on_run_finished(self, manifest: dict) -> None:
        """Complete the matching ``RunRecord`` from a ``run_finished`` manifest."""
        found = self._record_for(manifest)
        if found is None:
            return
        store, record, is_current = found
        run = record.find_run(str(manifest.get("run_id", "")))
        if run is None:
            logger.warning(
                "run_finished for unknown run %r — ignored", manifest.get("run_id")
            )
            return
        run.finished_utc = str(manifest.get("finished_utc", ""))
        run.status = str(manifest.get("status", RUN_STATUS_FAILED))
        run.reason = str(manifest.get("reason", ""))
        self._save_record(store, record, is_current)
        if is_current:
            self.run_recorded.emit(run.to_dict())

    # ------------------------------------------------------------------
    # Analysis bundle selection (no notebook involved)
    # ------------------------------------------------------------------

    def select_bundle(self, run_id: str, bundle_id: str) -> bool:
        """Choose the **analysis bundle** that represents one run.

        The analysis stage's own decision, and nothing else's: which of the
        run's bundles (a recipe's, a script's, a draft) stands for it wherever
        the run is presented, the notebook included. Only a completed bundle
        can be selected; ``""`` clears the selection, so the run is presented
        from its facts.

        Args:
            run_id: The run, in the open experiment.
            bundle_id: One of the run's bundles, or ``""``.

        Returns:
            ``True`` when the selection changed or was already that; ``False``
            when no experiment is open, the run is unknown, or the bundle does
            not exist or did not complete (all logged, never raised).
        """
        run = self._open_run(run_id, "select a bundle of")
        if run is None or self._experiment is None:
            return False
        if bundle_id:
            bundle = self._store.read_bundle(self._experiment.experiment_id, run_id, bundle_id)
            if bundle is None or not bundle.ok or not bundle.sealed:
                logger.warning("Run %s has no completed, sealed bundle %r to select", run_id, bundle_id)
                return False
        if run.selected_bundle == bundle_id:
            return True
        run.selected_bundle = bundle_id
        self._save_current()
        self.run_recorded.emit(run.to_dict())
        logger.info("Run %s is now represented by bundle %s", run_id, bundle_id or "(facts)")
        return True

    # ------------------------------------------------------------------
    # The notebook binding (one page per experiment)
    # ------------------------------------------------------------------

    def eln_binding(self) -> ElnBinding | None:
        """Return the open experiment's notebook binding, or ``None``."""
        return None if self._experiment is None else self._experiment.eln

    def link_eln(self, binding: ElnBinding) -> bool:
        """Link the open experiment to its notebook page.

        Replaces any earlier binding: re-linking the experiment to another
        page is a deliberate act, and the runs already published stay
        published where they went.

        Args:
            binding: The binding to install (the page, or ``create_pending``
                while a new page is being created).

        Returns:
            ``True`` when installed; ``False`` when no experiment is open.
        """
        if self._experiment is None:
            logger.warning("No experiment is open — nothing to link to a notebook page")
            return False
        self._experiment.eln = binding
        self._save_current()
        self.experiment_changed.emit(self._experiment.to_dict())
        logger.info(
            "Experiment %s linked to its notebook page (%s)",
            self._experiment.experiment_id,
            binding.entry.url if binding.entry else "page being created",
        )
        return True

    def unlink_eln(self) -> bool:
        """Remove the open experiment's notebook binding.

        Returns:
            ``True`` when a binding was removed.
        """
        if self._experiment is None or self._experiment.eln is None:
            return False
        self._experiment.eln = None
        self._save_current()
        self.experiment_changed.emit(self._experiment.to_dict())
        return True

    def set_eln_entry(
        self, experiment_id: str, link: ElnLink, session_root: Path | None = None
    ) -> bool:
        """Record the page the backend confirmed for one experiment.

        Works on a closed experiment too: a page queued for creation while the
        notebook was down may be confirmed after the experiment closed.

        Args:
            experiment_id: The experiment.
            link: The confirmed page.
            session_root: The session the experiment belongs to; ``None``
                for the open one (see ``_mutate_experiment``).

        Returns:
            ``True`` when recorded.
        """

        def _apply(record: ExperimentRecord) -> bool:
            binding = record.eln or ElnBinding()
            binding.entry = link
            binding.create_pending = False
            record.eln = binding
            return True

        return self._mutate_experiment(
            experiment_id, _apply, "record the notebook page of", session_root
        )

    def approve_eln_publishing(self, user_id: str) -> bool:
        """Approve publishing for the open experiment, from now on.

        The human's gate, taken once per experiment: after it, each publish
        appends its section without asking again.

        Args:
            user_id: Who approved.

        Returns:
            ``True`` when approved; ``False`` when nothing is linked.
        """
        if self._experiment is None or self._experiment.eln is None:
            logger.warning("No linked experiment to approve publishing for")
            return False
        binding = self._experiment.eln
        binding.publish_approved = True
        binding.approved_by = user_id
        binding.approved_utc = _utc_now_iso()
        self._save_current()
        self.experiment_changed.emit(self._experiment.to_dict())
        logger.info("Publishing approved for experiment %s by %s", self._experiment.experiment_id, user_id)
        return True

    def apply_eln_fields(
        self, values: Mapping[str, Any], snapshot: Mapping[str, Any] | None = None
    ) -> bool:
        """Apply fields read back from the notebook to the sample metadata.

        The read direction's single write: the values a person accepted (after
        seeing the difference) become the open experiment's ``sample_info``,
        which every LATER run stamps into its data file. Runs already written
        are facts and are never changed.

        Args:
            values: ``{sample_info key: value}`` to set.
            snapshot: What was read, with its sources, kept on the binding
                for provenance; ``None`` keeps the old snapshot.

        Returns:
            ``True`` when applied; ``False`` when no experiment is open.
        """
        if self._experiment is None:
            return False
        self._experiment.sample_info.update({str(k): v for k, v in values.items()})
        if snapshot is not None and self._experiment.eln is not None:
            self._experiment.eln.field_snapshot = dict(snapshot)
        self._save_current()
        self.experiment_changed.emit(self._experiment.to_dict())
        logger.info("Applied %d notebook field(s) to the sample metadata", len(values))
        return True

    def record_eln_publish(
        self,
        experiment_id: str,
        publish_id: str,
        run_bundles: Mapping[str, str],
        published_utc: str = "",
        session_root: Path | None = None,
    ) -> bool:
        """Record that one publish reached the notebook.

        Called by the publishing service only after the backend confirmed the
        appended section, so a run marked published really is on the page.

        Args:
            experiment_id: The experiment.
            publish_id: The publish.
            run_bundles: ``{run_id: bundle_id}`` it covered (``""`` for a run
                published from its facts).
            published_utc: When; ``""`` for now.
            session_root: The session the experiment belongs to; ``None``
                for the open one.

        Returns:
            ``True`` when recorded.
        """
        stamp = published_utc or _utc_now_iso()

        def _apply(record: ExperimentRecord) -> bool:
            for run_id, bundle_id in run_bundles.items():
                run = record.find_run(run_id)
                if run is None:
                    continue
                run.published = True
                run.eln_publish = {
                    "publish_id": publish_id,
                    "published_utc": stamp,
                    "bundle_id": bundle_id,
                }
            return True

        return self._mutate_experiment(experiment_id, _apply, "record a publish of", session_root)

    def _mutate_experiment(
        self,
        experiment_id: str,
        apply: Callable[[ExperimentRecord], bool],
        action: str,
        session_root: Path | None = None,
    ) -> bool:
        """Apply one change to an experiment record, open or closed.

        The open experiment is changed in memory, saved and announced; a
        closed one is loaded, changed and saved without disturbing the open
        one. A record written by a newer application is never changed.

        Args:
            experiment_id: The experiment.
            apply: Mutates the record; returns ``False`` to abandon.
            action: What is being done, for the log line.
            session_root: The session the experiment belongs to. ``None`` or
                the open session's folder: this session. Any other folder —
                a result arriving for a session loaded away from since
                (experiment ids repeat across sessions) — is changed in THAT
                session's store and never touches the open experiment.

        Returns:
            ``True`` when applied and saved.
        """
        store = self._store
        if session_root is not None and not _same_path(session_root, self._store.root):
            store = ExperimentStore(Path(session_root))
        elif self._runs_held:
            logger.warning("Session held — not allowed to %s experiment %s", action, experiment_id)
            return False
        elif self._experiment is not None and self._experiment.experiment_id == experiment_id:
            if not apply(self._experiment):
                return False
            self._save_current()
            self.experiment_changed.emit(self._experiment.to_dict())
            return True
        record = store.load(experiment_id)
        if record is None:
            logger.warning("Unknown experiment %r — cannot %s it", experiment_id, action)
            return False
        if record.schema_version > SCHEMA_VERSION:
            logger.warning(
                "Refusing to %s experiment %s: its schema_version=%d > %d",
                action,
                experiment_id,
                record.schema_version,
                SCHEMA_VERSION,
            )
            return False
        if not apply(record):
            return False
        try:
            store.save(record)
        except OSError as exc:
            logger.error("Could not %s experiment %s: %s", action, experiment_id, exc)
            return False
        return True

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

    def _install_run_folder(self, experiment_id: str | None) -> None:
        """Tell the engine where every run writes: this experiment's data folder.

        The third policy value pushed down beside the envelope and attendance
        (``Orchestrator.set_run_folder``), with the experiment's run
        subfolder and its run-number floor (one more than the highest run it
        ever recorded, so a deleted file never frees its number). ``None`` —
        no experiment open — installs ``""``, which refuses every run. An
        engine that predates the command (a test double) is skipped rather
        than failed.

        Args:
            experiment_id: The open experiment, or ``None``.
        """
        setter = getattr(self._orchestrator, "set_run_folder", None)
        if not callable(setter):
            return
        if self._pending_switch is not None:
            # The engine is releasing the folder for a session switch: nothing
            # may hand it a folder again until the switch completes (the new
            # session installs its own) or fails (this one is re-installed).
            return
        if experiment_id is None or self._runs_held:
            setter("")
            self.run_folder_changed.emit("")
            return
        record = self._experiment
        if record is None or record.experiment_id != experiment_id:
            record = self._store.load(experiment_id)
        subfolder = record.run_subfolder if record is not None else ""
        floor = (record.highest_run_number() if record is not None else 0) + 1
        setter(
            str(self._store.data_dir(experiment_id)),
            subfolder=subfolder,
            run_number_floor=floor,
        )
        self.run_folder_changed.emit(str(self.run_folder() or ""))

    def _current_session_folder(self) -> Path | None:
        """Return the session folder owning ``self._store``, or ``None``.

        ``None`` when no ``session_store`` was given at construction — the
        caller then knows to skip index maintenance entirely. Otherwise it is
        ``self._store.root`` itself: the experiment store is rooted AT the
        session folder, so this can never disagree with the store it
        describes.
        """
        if self._session_store is None:
            return None
        return self._store.root

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
        if self._runs_held:
            return
        folder = self._current_session_folder()
        if folder is None:
            return
        session = self._session_store.load(folder)
        if session is None:
            logger.warning(
                "Could not load session %s to reconcile its experiment index", folder
            )
            return
        entries: list[ExperimentIndexEntry] = []
        for experiment_id in self._store.list_experiments():
            record = self._store.load(experiment_id)
            if record is None:
                logger.warning(
                    "Skipping unreadable experiment %r while reconciling "
                    "session %s's index",
                    experiment_id,
                    folder,
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
            self._session_store.save(session, folder)
        except OSError:
            logger.exception(
                "Could not save session %s's reconciled experiment index", folder
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
        if self._runs_held:
            return  # never write records another station is saving
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
        # No experiment until one is resumed below: until then every run is
        # refused, because a run's data always belongs to an experiment.
        self._install_run_folder(None)
        if self._runs_held:
            # Another station holds this session: its records are not ours to
            # touch — not even to mark a run of theirs failed.
            logger.error("Session held, no experiment adopted: %s", self._runs_held)
            return
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
        self._install_run_folder(record.experiment_id)
        logger.info("Resumed experiment %s (%d runs)", record.experiment_id, len(record.runs))
        self.experiment_changed.emit(record.to_dict())


#: How long a session switch waits for the engine to release the run folder.
SWITCH_VERDICT_TIMEOUT_MS = 15000


def _same_path(a: str | Path, b: str | Path) -> bool:
    """Return whether two paths name the same folder (resolved; case per the OS)."""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)
