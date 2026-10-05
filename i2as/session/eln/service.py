"""The notebook service — the one place the application talks to an ELN.

**Nothing here blocks the GUI or touches the station.** The service lives on
the GUI thread, but every call into a block (a connector searching, reading,
creating a page, uploading, appending; a renderer laying out a section) runs
on the service's own WORKER THREAD, in the block's helper process
(``block_runner.py``), bounded by a timeout. Answers come back to the GUI
thread through a Qt signal, and only there are experiment records changed —
through the ``ExperimentManager``, the single writer of experiment state.

What it offers, all non-blocking:

* **Accounts** (the logged-in user's, from their profile): ``verify``,
  ``list_templates``, ``search``.
* **Linking** the open experiment to its ONE page — an existing page, or a
  new one created from the profile's template (queued, so it works offline) —
  with samples and resources linked to it: ``link_experiment``.
* **Reading back** the page's and the linked items' fields through the
  profile's read map, as a proposal a person applies: ``read_fields``.
* **Publishing** finished runs as one appended section: ``publish``. The first
  publish of an experiment needs a human's approval
  (``ExperimentManager.approve_eln_publishing``); later ones append without
  asking.
* **Status** for the GUI (``status_changed``): how much is queued, what needs
  attention and why, and ``retry`` / ``confirm_connector`` to act on it.

The service is built with the machine's publishing settings, the user profile
store and the credential store; with publishing switched off for the user it
does nothing and sends nothing.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from i2as.blocks.connector import KIND_ITEM, ElnEntryRef, ElnQuery, ElnRecord, ElnRef
from i2as.blocks.discovery import KIND_CONNECTOR, KIND_RENDERER
from i2as.blocks.profile import PAGE_ROLE, apply_read_map
from i2as.session.app_config import PublishingSettings
from i2as.session.credentials import SCOPE_ELN, CredentialStore, credential_key
from i2as.session.eln.block_runner import BlockRunner, SubprocessBlockRunner
from i2as.session.eln.ledger import Ledger
from i2as.session.eln.outbox import (
    ATTENTION_AUTH,
    ATTENTION_CHANGED,
    JOB_CREATE_ENTRY,
    STATE_NEEDS_ATTENTION,
    STATE_PENDING,
    Outbox,
)
from i2as.session.eln.publishing import (
    LEDGER_FILENAME,
    OUTBOX_FILENAME,
    PINNED_RENDERER,
    AccountUnavailable,
    BlockCatalog,
    JobExecutor,
    PublishError,
    create_entry_job,
    entry_of,
    link_items_job,
    link_of,
    new_publish_id,
    pin_blocks,
    pinned_profile,
    plan_publish,
    publish_job,
    render_plan,
    unpublished_runs,
    utc_now,
)
from i2as.session.models import ElnBinding, ExperimentRecord, LinkedItem, PinnedBlock
from i2as.session.user_profile import ElnAccount, UserProfileStore

logger = logging.getLogger(__name__)

#: Publish-status states, for a GUI chip.
STATUS_DISABLED = "disabled"
STATUS_SYNCED = "synced"
STATUS_PENDING = "pending"
STATUS_OFFLINE = "offline"
STATUS_ATTENTION = "attention"

RunnerFactory = Callable[..., BlockRunner]


class ElnService(QObject):
    """Talks to the notebook for the application, off the GUI thread.

    Signals:
        status_changed (dict): ``{"state", "pending", "attention": [...],
            "detail"}`` after every change.
        publish_finished (dict): ``{"experiment_id", "publish_id",
            "run_bundles", "url"}`` when a publish reached the page.
        publish_failed (dict): ``{"experiment_id", "publish_id", "reason"}``
            when a publish could not even be queued (the renderer failed).
        page_ready (dict): ``{"experiment_id", "entry"}`` when a new page
            was confirmed.

    Args:
        manager: The ``ExperimentManager``.
        profiles: The user profile store.
        credentials: The credential store.
        publishing: Returns the machine's ``PublishingSettings``.
        catalog: The block catalog (shipped + the user's blocks folder).
        runner_factory: Builds a ``BlockRunner`` (tests pass the in-process
            one).
        synchronous: Run worker tasks inline, on the calling thread (tests).
    """

    status_changed = pyqtSignal(dict)
    publish_finished = pyqtSignal(dict)
    publish_failed = pyqtSignal(dict)
    page_ready = pyqtSignal(dict)
    _delivered = pyqtSignal(object, object)

    def __init__(
        self,
        manager: Any,
        profiles: UserProfileStore,
        credentials: CredentialStore,
        publishing: Callable[[], PublishingSettings] = PublishingSettings,
        catalog: BlockCatalog | None = None,
        runner_factory: RunnerFactory = SubprocessBlockRunner,
        synchronous: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._profiles = profiles
        self._credentials = credentials
        self._publishing = publishing
        self._catalog = catalog or BlockCatalog()
        self._runner_factory = runner_factory
        self._synchronous = synchronous
        self._tasks: queue.Queue[tuple[Callable[[], Any], Callable[[Any], None] | None] | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._connectors: dict[tuple[str, str], tuple[BlockRunner, str, str]] = {}
        self._confirmed_entries: dict[str, ElnEntryRef] = {}
        self._outboxes: dict[str, Outbox] = {}
        # A drain for the NEW session is due once the one bound to the old
        # session (queued before a session switch) has finished.
        self._drain_again = False
        self._drain_queued = threading.Event()
        self._last_detail = ""
        self._offline = False
        self._delivered.connect(self._on_delivered)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.drain_soon)
        self._adopt_outboxes()

    # ------------------------------------------------------------------
    # Lifecycle and the worker
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread and the drain timer."""
        interval = max(self._publishing().drain_interval_s, 0.5)
        self._timer.start(int(interval * 1000))
        if not self._synchronous and self._worker is None:
            self._worker = threading.Thread(target=self._work, name="i2as-eln-worker", daemon=True)
            self._worker.start()
        self.drain_soon()

    def stop(self) -> None:
        """Stop the timer and the worker, and every helper process."""
        self._timer.stop()
        if self._worker is not None:
            self._tasks.put(None)
            self._worker.join(timeout=5)
            self._worker = None
        self._close_connectors()

    def _submit(self, task: Callable[[], Any], callback: Callable[[Any], None] | None = None) -> None:
        """Run ``task`` on the worker; deliver its result (or exception) to ``callback`` on the GUI thread."""
        if self._synchronous:
            try:
                result: Any = task()
            except Exception as exc:  # noqa: BLE001 - delivered, never raised into the GUI
                result = exc
            self._on_delivered(callback, result)
            return
        if self._worker is None:
            self.start()
        self._tasks.put((task, callback))

    def _work(self) -> None:
        """The worker thread's loop."""
        while True:
            item = self._tasks.get()
            if item is None:
                return
            task, callback = item
            try:
                result: Any = task()
            except Exception as exc:  # noqa: BLE001 - the worker must never die
                logger.exception("A notebook task failed")
                result = exc
            self._delivered.emit(callback, result)

    def _on_delivered(self, callback: Any, result: Any) -> None:
        """GUI thread: hand a worker result to its callback."""
        if callback is None:
            return
        try:
            callback(result)
        except Exception:  # noqa: BLE001 - a GUI callback must not break the service
            logger.exception("A notebook callback failed")

    # ------------------------------------------------------------------
    # Accounts and connectors
    # ------------------------------------------------------------------

    def current_user(self) -> str:
        """The user whose accounts the GUI is working with (the experiment's owner, or the guest)."""
        experiment = self._manager.current_experiment()
        return experiment.user_id if experiment is not None and experiment.user_id else "guest"

    def enabled(self, user_id: str = "") -> bool:
        """Whether publishing is switched on for a user."""
        return self._profiles.load(user_id or self.current_user()).eln.enabled

    def accounts(self, user_id: str = "") -> tuple[ElnAccount, ...]:
        """Return a user's notebook accounts."""
        return self._profiles.load(user_id or self.current_user()).eln.accounts

    def catalog(self) -> BlockCatalog:
        """The block catalog."""
        return self._catalog

    def _account(self, user_id: str, account_id: str) -> ElnAccount:
        account = self._profiles.load(user_id).eln.account(account_id)
        if account is None:
            raise AccountUnavailable(f"the notebook account '{account_id}' is not configured for {user_id}")
        return account

    def _connector(self, user_id: str, account_id: str) -> tuple[BlockRunner, str]:
        """Worker thread: return (runner, digest) for an account, reusing a live helper."""
        account = self._account(user_id, account_id)
        info = self._catalog.get(KIND_CONNECTOR, account.connector)
        if info is None:
            raise AccountUnavailable(f"the connector '{account.connector}' is not installed or not usable")
        secret = self._credentials.get(credential_key(SCOPE_ELN, account_id, user_id))
        signature = f"{info.digest}|{sorted(account.settings.items())!r}|{hash(secret)}"
        key = (user_id, account_id)
        cached = self._connectors.get(key)
        if cached is not None and cached[2] == signature:
            return cached[0], cached[1]
        if cached is not None:
            cached[0].close()
        runner = self._runner_factory(KIND_CONNECTOR, info.path, dict(account.settings), secret, label=f"{account.connector}:{account_id}")
        self._connectors[key] = (runner, info.digest, signature)
        return runner, info.digest

    def _close_connectors(self) -> None:
        for runner, _digest, _signature in self._connectors.values():
            runner.close()
        self._connectors.clear()

    def reload(self) -> None:
        """Forget cached helpers (a profile or key changed) and retry auth failures."""
        self._submit(self._close_connectors)
        for outbox in list(self._outboxes.values()):
            outbox.retry(reason=ATTENTION_AUTH)
        self.drain_soon()

    def _call(self, account_id: str, method: str, args: dict[str, Any], user_id: str = "") -> Any:
        """Worker thread: one connector call."""
        runner, _digest = self._connector(user_id or self.current_user(), account_id)
        return runner.call(method, args, self._publishing().block_timeout_s)

    def verify(self, account_id: str, callback: Callable[[Any], None], user_id: str = "") -> None:
        """Check an account's server and key; ``callback(ElnIdentity dict | Exception)``."""
        user = user_id or self.current_user()
        self._submit(lambda: self._call(account_id, "verify", {}, user), callback)

    def list_templates(self, account_id: str, callback: Callable[[Any], None], user_id: str = "") -> None:
        """List an account's templates; ``callback(list | Exception)``."""
        user = user_id or self.current_user()
        self._submit(lambda: self._call(account_id, "list_templates", {}, user), callback)

    def search(self, account_id: str, text: str, kind: str, callback: Callable[[Any], None], user_id: str = "") -> None:
        """Search pages (``entry``) or items; ``callback(list of hit dicts | Exception)``."""
        user = user_id or self.current_user()
        query = ElnQuery(text=text, kind=kind).to_dict()
        self._submit(lambda: self._call(account_id, "search", {"query": query}, user), callback)

    # ------------------------------------------------------------------
    # Linking the open experiment to its page
    # ------------------------------------------------------------------

    def link_experiment(
        self,
        account_id: str,
        profile_id: str = "",
        *,
        entry: ElnEntryRef | None = None,
        new_title: str = "",
        template_id: str = "",
        items: list[LinkedItem] | None = None,
    ) -> ElnBinding:
        """Link the open experiment to an existing page, or to a new one (queued).

        Args:
            account_id: The user's account to reach the page through.
            profile_id: The profile; ``""`` for the user's default.
            entry: An existing page to link, or ``None`` to create one.
            new_title: The new page's title (``""``: the experiment's title).
            template_id: The template for a new page (``""``: the profile's,
                then the user's default).
            items: Samples and resources to link to the page.

        Returns:
            The binding installed on the experiment.

        Raises:
            PublishError: No experiment is open, the account or a block is
                unusable.
        """
        experiment = self._manager.current_experiment()
        if experiment is None:
            raise PublishError("no experiment is open")
        user = experiment.user_id or "guest"
        eln_settings = self._profiles.load(user).eln
        account = eln_settings.account(account_id)
        if account is None:
            raise PublishError(f"the notebook account '{account_id}' is not configured")
        profile, connector_pin, renderer_pin, profile_pin = pin_blocks(
            self._catalog,
            self._manager.store.eln_dir(experiment.experiment_id),
            connector_id=account.connector,
            profile_id=profile_id or eln_settings.default_profile,
        )
        chosen_template = template_id or profile.template or eln_settings.default_template
        binding = ElnBinding(
            account_id=account.account_id,
            connector=connector_pin,
            renderer=renderer_pin,
            profile=profile_pin,
            template_id=chosen_template if entry is None else (entry.template_id or ""),
            entry=link_of(entry) if entry is not None else None,
            create_pending=entry is None,
            linked_items=list(items or []),
        )
        if not self._manager.link_eln(binding):
            raise PublishError("the experiment could not be linked")
        outbox = self._outbox(experiment.experiment_id)
        if entry is None:
            outbox.enqueue(create_entry_job(experiment, binding, user_id=user, title=new_title or experiment.title, fields={}))
        if binding.linked_items:
            outbox.enqueue(link_items_job(experiment, binding, binding.linked_items, user_id=user))
        self.drain_soon()
        return binding

    def confirm_connector(self) -> bool:
        """Pin the open experiment to its connector's CURRENT code, and retry what waited for it."""
        experiment = self._manager.current_experiment()
        binding = experiment.eln if experiment is not None else None
        if binding is None:
            return False
        account = self._profiles.load(experiment.user_id or "guest").eln.account(binding.account_id)
        info = self._catalog.get(KIND_CONNECTOR, account.connector) if account is not None else None
        if info is None:
            return False
        updated = replace(binding, connector=PinnedBlock(info.block_id, info.digest))
        self._manager.link_eln(updated)
        outbox = self._outbox(experiment.experiment_id)
        for job in outbox.jobs().values():
            if job.state == STATE_NEEDS_ATTENTION and job.attention == ATTENTION_CHANGED:
                outbox.record(replace(job, state=STATE_PENDING, attention="", next_due_utc="", payload={**job.payload, "connector_digest": info.digest}))
        self.drain_soon()
        return True

    # ------------------------------------------------------------------
    # Reading fields back
    # ------------------------------------------------------------------

    def read_fields(self, callback: Callable[[Any], None]) -> None:
        """Read the page's and linked items' fields through the profile's read map.

        ``callback`` receives ``{key: {"value", "unit", "source", "raw",
        "error", "current"}}`` — a proposal; nothing is applied — or an
        exception.
        """
        experiment = self._manager.current_experiment()
        binding = experiment.eln if experiment is not None else None
        if binding is None:
            self._on_delivered(callback, PublishError("the experiment is not linked to a notebook page"))
            return
        profile = pinned_profile(self._manager.store.eln_dir(experiment.experiment_id))
        user = experiment.user_id or "guest"
        current = dict(experiment.sample_info)
        entry = entry_of(binding)
        items = list(binding.linked_items)

        def _task() -> dict[str, Any]:
            records: dict[str, ElnRecord] = {}
            if entry is not None:
                records[PAGE_ROLE] = ElnRecord.from_dict(
                    self._call(binding.account_id, "get_record", {"ref": ElnRef(record_id=entry.entry_id).to_dict()}, user)
                )
            for item in items:
                if item.role and item.role not in records:
                    records[item.role] = ElnRecord.from_dict(
                        self._call(binding.account_id, "get_record", {"ref": ElnRef(kind=KIND_ITEM, record_id=item.item_id).to_dict()}, user)
                    )
            found = apply_read_map(profile, records)
            stamp = utc_now()
            for key, entry_value in found.items():
                entry_value["current"] = current.get(key)
                entry_value["fetched_utc"] = stamp
            return found

        self._submit(_task, callback)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, run_ids: list[str] | None = None) -> str:
        """Publish finished runs of the open experiment as one appended section.

        Args:
            run_ids: The runs; ``None`` for every finished run not yet
                published. A run already published may be named explicitly
                (its new section is appended; the old one stays).

        Returns:
            The publish id. Rendering and sending happen on the worker; the
            outcome arrives on ``publish_finished`` / ``publish_failed`` and in
            ``status_changed``.

        Raises:
            PublishError: Nothing can be published, and why (not linked, not
                approved, no account, nothing new).
        """
        experiment = self._manager.current_experiment()
        if experiment is None:
            raise PublishError("no experiment is open")
        binding = experiment.eln
        user = experiment.user_id or "guest"
        if not self.enabled(user):
            raise PublishError("publishing is switched off in your notebook settings")
        if binding is None:
            raise PublishError("the experiment is not linked to a notebook page")
        if not binding.publish_approved:
            raise PublishError("publishing for this experiment has not been approved yet")
        if self._profiles.load(user).eln.account(binding.account_id) is None:
            raise PublishError(f"the notebook account '{binding.account_id}' is not configured")
        if run_ids is None:
            runs = unpublished_runs(experiment)
        else:
            runs = [run for run in (experiment.find_run(r) for r in run_ids) if run is not None and run.status != "running"]
        if not runs:
            raise PublishError("there is nothing new to publish")
        store = self._manager.store
        eln_dir = store.eln_dir(experiment.experiment_id)
        renderer_path = eln_dir / PINNED_RENDERER
        if not renderer_path.is_file():
            raise PublishError("the experiment's renderer copy is missing; link the experiment again")
        user_record = self._manager.roster.get(user) if hasattr(self._manager, "roster") else None
        settings = self._publishing()
        # The worker gets a SNAPSHOT of the record: gathering the plan reads
        # and hashes every figure (to refuse one changed since sealing), which
        # is file I/O that must not run on the GUI thread.
        snapshot = ExperimentRecord.from_dict(experiment.to_dict())
        run_ids_chosen = [run.run_id for run in runs]
        publish_id = new_publish_id()
        experiment_id = experiment.experiment_id
        outbox = self._outbox(experiment_id)
        account_id = binding.account_id
        digest = binding.connector.digest
        user_name = getattr(user_record, "name", "") or user

        def _task() -> str:
            plan = plan_publish(
                snapshot,
                [run for run in (snapshot.find_run(r) for r in run_ids_chosen) if run is not None],
                read_bundle=store.read_bundle,
                bundle_dir=store.bundle_dir,
                profile=pinned_profile(eln_dir),
                user_name=user_name,
                config_name=snapshot.config_name,
                max_attachment_bytes=settings.max_attachment_bytes,
                publish_id=publish_id,
            )
            renderer = self._runner_factory(KIND_RENDERER, renderer_path, label=f"renderer:{experiment_id}")
            try:
                section, html = render_plan(plan, renderer, settings.block_timeout_s)
            finally:
                renderer.close()
            outbox.enqueue(publish_job(plan, section, html, user_id=user, account_id=account_id, connector_digest=digest))
            return plan.publish_id

        def _queued(result: Any) -> None:
            if isinstance(result, Exception):
                logger.warning("Publish %s of %s failed before it was queued: %s", publish_id, experiment_id, result)
                self.publish_failed.emit({"experiment_id": experiment_id, "publish_id": publish_id, "reason": str(result)})
            self._emit_status()
            self.drain_soon()

        self._submit(_task, _queued)
        logger.info("Publishing %d run(s) of %s as %s", len(runs), experiment_id, publish_id)
        return publish_id

    # ------------------------------------------------------------------
    # Draining the outboxes
    # ------------------------------------------------------------------

    def _outbox(self, experiment_id: str) -> Outbox:
        outbox = self._outboxes.get(experiment_id)
        if outbox is None:
            settings = self._publishing()
            outbox = Outbox(
                self._manager.store.eln_dir(experiment_id) / OUTBOX_FILENAME,
                retry_base_s=settings.retry_base_s,
                retry_max_s=settings.retry_max_s,
            )
            self._outboxes[experiment_id] = outbox
        return outbox

    def reset_session(self) -> None:
        """Forget the previous session's outboxes and pages; adopt the new session's.

        Called on ``ExperimentManager.session_changed``. Fresh containers, so
        a drain already bound to the old session keeps its own.
        """
        self._outboxes = {}
        self._confirmed_entries = {}
        self._adopt_outboxes()
        self._emit_status()
        if self._drain_queued.is_set():
            # The queued drain is bound to the old session: drain the new
            # one as soon as it has finished.
            self._drain_again = True
        else:
            self.drain_soon()

    def _adopt_outboxes(self) -> None:
        """Pick up journals left by an earlier run of the application."""
        try:
            experiment_ids = self._manager.store.list_experiments()
        except OSError:
            return
        for experiment_id in experiment_ids:
            if (self._manager.store.eln_dir(experiment_id) / OUTBOX_FILENAME).exists():
                self._outbox(experiment_id)

    def _entry_for(self, experiment_id: str) -> ElnEntryRef | None:
        """Worker thread: the page an experiment's record names (or one just created)."""
        confirmed = self._confirmed_entries.get(experiment_id)
        if confirmed is not None:
            return confirmed
        record = self._manager.store.load(experiment_id)
        return entry_of(record.eln if record is not None else None)

    def drain_soon(self) -> None:
        """Ask the worker to perform every due job (at most one drain queued).

        The drain is bound HERE, on the GUI thread, to the session open now:
        its store, outboxes and confirmed pages travel with it, and its
        results are written back to that session's records — so a session
        loaded while the drain runs can never receive another session's
        notebook outcomes (experiment ids repeat across sessions).
        """
        if self._drain_queued.is_set():
            return
        self._drain_queued.set()
        store = self._manager.store
        outboxes = dict(self._outboxes)
        confirmed = self._confirmed_entries
        self._submit(lambda: self._drain(store, outboxes, confirmed), self._drained)

    def _drain(
        self,
        store: Any = None,
        outboxes: dict[str, Outbox] | None = None,
        confirmed: dict[str, ElnEntryRef] | None = None,
    ) -> tuple[Any, list[Any]]:
        """Worker thread: perform every due job, oldest first, creations first.

        Args:
            store: The session's experiment store the drain is bound to
                (``None``: the one open now).
            outboxes: That session's outboxes (``None``: the live ones).
            confirmed: That session's confirmed pages (``None``: the live ones).

        Returns:
            ``(store, [(experiment_id, outcome), …])``.
        """
        self._drain_queued.clear()
        store = store if store is not None else self._manager.store
        outboxes = outboxes if outboxes is not None else dict(self._outboxes)
        confirmed = confirmed if confirmed is not None else self._confirmed_entries

        def entry_for(experiment_id: str) -> ElnEntryRef | None:
            known = confirmed.get(experiment_id)
            if known is not None:
                return known
            record = store.load(experiment_id)
            return entry_of(record.eln if record is not None else None)

        executor = JobExecutor(
            connector_for=self._connector,
            entry_for=entry_for,
            ledger_for=lambda experiment_id: Ledger(store.eln_dir(experiment_id) / LEDGER_FILENAME),
            timeout_s=self._publishing().block_timeout_s,
        )
        outcomes: list[Any] = []
        for experiment_id, outbox in list(outboxes.items()):
            due = sorted(outbox.due(), key=lambda job: (job.kind != JOB_CREATE_ENTRY, job.created_utc))
            for job in due:
                outcome = executor.run(outbox, job)
                if outcome.entry is not None:
                    confirmed[experiment_id] = outcome.entry
                outcomes.append((experiment_id, outcome))
        return store, outcomes

    def _drained(self, result: Any) -> None:
        """GUI thread: record what the drain achieved, in the session it ran for."""
        if self._drain_again:
            self._drain_again = False
            self.drain_soon()
        if isinstance(result, Exception):
            self._last_detail = str(result)
            self._emit_status()
            return
        store, outcomes = result
        session_root = store.root
        current = session_root == self._manager.store.root
        self._offline = False
        for experiment_id, outcome in outcomes:
            if outcome.entry is not None:
                self._manager.set_eln_entry(experiment_id, link_of(outcome.entry), session_root=session_root)
                if current:
                    self.page_ready.emit({"experiment_id": experiment_id, "entry": outcome.entry.to_dict()})
            if outcome.published is not None:
                published = outcome.published
                self._manager.record_eln_publish(
                    experiment_id,
                    published["publish_id"],
                    published["run_bundles"],
                    published["published_utc"],
                    session_root=session_root,
                )
                if current:
                    entry = self._confirmed_entries.get(experiment_id) or self._entry_for(experiment_id)
                    self.publish_finished.emit({**published, "experiment_id": experiment_id, "url": entry.url if entry else ""})
            if outcome.job.state == STATE_PENDING and outcome.job.last_error and not outcome.waiting:
                self._offline = True
                self._last_detail = outcome.job.last_error
        self._emit_status()

    def retry(self, experiment_id: str = "") -> int:
        """Put jobs needing attention back in the queue (one experiment, or all)."""
        outboxes = [self._outboxes[experiment_id]] if experiment_id in self._outboxes else list(self._outboxes.values())
        count = sum(outbox.retry() for outbox in outboxes)
        self.drain_soon()
        self._emit_status()
        return count

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self, experiment_id: str = "") -> dict[str, Any]:
        """Return what the GUI chip shows (for one experiment, or all)."""
        outboxes = (
            {experiment_id: self._outboxes[experiment_id]} if experiment_id in self._outboxes else ({} if experiment_id else self._outboxes)
        )
        pending = 0
        attention: list[dict[str, Any]] = []
        for eid, outbox in outboxes.items():
            for job in outbox.jobs().values():
                if job.state == STATE_PENDING:
                    pending += 1
                elif job.state == STATE_NEEDS_ATTENTION:
                    attention.append({"experiment_id": eid, "job_id": job.job_id, "kind": job.kind, "reason": job.attention, "error": job.last_error})
        if not self.enabled():
            state = STATUS_DISABLED
        elif attention:
            state = STATUS_ATTENTION
        elif pending and self._offline:
            state = STATUS_OFFLINE
        elif pending:
            state = STATUS_PENDING
        else:
            state = STATUS_SYNCED
        return {"state": state, "pending": pending, "attention": attention, "detail": self._last_detail if state in (STATUS_OFFLINE, STATUS_ATTENTION) else ""}

    def publish_history(self, experiment_id: str) -> list[dict[str, Any]]:
        """Return the publish jobs of one experiment, newest first, for the GUI."""
        outbox = self._outboxes.get(experiment_id)
        if outbox is None:
            return []
        jobs = [job for job in outbox.jobs().values() if job.kind == "publish"]
        return [
            {
                "publish_id": job.payload.get("publish_id", ""),
                "state": job.state,
                "runs": sorted((job.payload.get("run_bundles") or {}).keys()),
                "error": job.last_error,
                "created_utc": job.created_utc,
            }
            for job in sorted(jobs, key=lambda j: j.created_utc, reverse=True)
        ]

    def _emit_status(self) -> None:
        self.status_changed.emit(self.status())
