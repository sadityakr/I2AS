"""Typed records of the L6 Session Management layer.

Every class here is a plain ``@dataclass`` with the tolerant-parse contract:
``from_dict()`` accepts arbitrary junk and degrades to defaults instead of
raising, so a hand-edited or older ``experiment.json`` can never brick the
application. All models construct from defaults alone — both properties are
machine-checked by the session-model conformance tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from i2as.core.events import OPERATOR, Actor
from i2as.core.plan import EnvelopeBound, ExperimentEnvelope
from i2as.core.run_naming import normalize_run_subfolder

logger = logging.getLogger(__name__)

# Run lifecycle states (mirrors the Orchestrator's run-manifest statuses, plus
# the initial "running"). Exposed as constants so callers never hard-code them.
RUN_STATUS_RUNNING = "running"
RUN_STATUS_DONE = "done"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_ABORTED = "aborted"
_VALID_RUN_STATUSES = frozenset(
    {RUN_STATUS_RUNNING, RUN_STATUS_DONE, RUN_STATUS_FAILED, RUN_STATUS_ABORTED}
)

# Experiment lifecycle states.
EXPERIMENT_STATUS_OPEN = "open"
EXPERIMENT_STATUS_CLOSED = "closed"
_VALID_EXPERIMENT_STATUSES = frozenset(
    {EXPERIMENT_STATUS_OPEN, EXPERIMENT_STATUS_CLOSED}
)

# The on-disk format version stamped into experiment.json/gui_state.json/
# active.json. Bump only when the JSON shape changes in a way older code
# cannot tolerantly parse; a value greater than this on load means "written
# by a newer app" and the record is treated read-only (see store.py/manager.py).
# Bumped 1 -> 2 for Session.experiments: an older app's Session.to_dict()
# does not emit that field, so resaving a newer session.json without this
# bump would silently drop the experiment index.
# Bumped 2 -> 3 for ExperimentRecord.eln (the experiment's notebook binding):
# an older app would drop the binding, which page this experiment publishes
# to, on resave.
SCHEMA_VERSION = 3

# The fixed roster identity used when nobody has logged in (see
# i2as.main._ensure_guest_user_registered). A real roster entry, not a
# null-user sentinel, so ExperimentManager.start_experiment()'s
# roster-membership check never rejects it.
GUEST_USER_ID = "guest"
GUEST_USER_NAME = "Guest"


def _as_str(value: object, default: str = "") -> str:
    """Coerce a JSON value to ``str``, falling back to ``default`` on ``None``."""
    return default if value is None else str(value)

def _as_bool(value: object, default: bool) -> bool:
    """Return ``value`` if it is a bool, else ``default`` (defensive parse)."""
    return value if isinstance(value, bool) else default


def _as_int(value: object, default: int) -> int:
    """Coerce a JSON value to ``int``, falling back to ``default`` on junk.

    ``bool`` is explicitly rejected (it is never a legitimate revision number)
    even though it subclasses ``int`` in Python.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def _as_dict(value: object) -> dict[str, Any]:
    """Return ``value`` if it is a dict, else an empty dict (defensive parse)."""
    return dict(value) if isinstance(value, dict) else {}


def _as_actor(value: object) -> tuple[Actor, bool]:
    """Read a stored actor, saying whether it had to be invented.

    The tolerant-parse contract applied to accountability: a record written
    before actors were stamped, or one whose actor field is junk, must still
    load — but it must NOT quietly claim the physicist did it. So the pair
    is ``(actor, legacy)``: the ``OPERATOR`` sentinel with ``legacy`` true
    means "this record predates actor recording (or lost it), and nobody
    should read the operator here as a fact".

    Args:
        value: The record's stored ``actor`` value (any junk tolerated).

    Returns:
        ``(actor, legacy)`` — the parsed ``Actor`` and ``False``, or
        ``(OPERATOR, True)`` when there was nothing usable to parse.
    """
    if not isinstance(value, dict):
        return OPERATOR, True
    try:
        return Actor.from_json(value), False
    except (TypeError, ValueError) as exc:
        logger.warning("stored actor %r is invalid (%s); reading it as legacy", value, exc)
        return OPERATOR, True


def _as_dict_list(value: object) -> list[dict[str, Any]]:
    """Return a list of dicts from ``value``, tolerating junk (defensive parse).

    Non-list input yields ``[]``; non-dict items within the list are dropped
    rather than raising, so one bad queue entry cannot brick loading.
    """
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def envelope_to_dict(envelope: ExperimentEnvelope | None) -> dict[str, Any]:
    """Serialise a ``ExperimentEnvelope`` to the JSON form stored on records.

    Args:
        envelope: The typed envelope, or ``None`` for "no envelope".

    Returns:
        ``{vi_name: {"min_value": ..., "max_value": ..., "state_key": ...}}``,
        or ``{}`` for ``None``.
    """
    if envelope is None:
        return {}
    return {
        vi_name: {
            "min_value": bound.min_value,
            "max_value": bound.max_value,
            "state_key": bound.state_key,
        }
        for vi_name, bound in envelope.bounds.items()
    }


def envelope_from_dict(data: object) -> ExperimentEnvelope | None:
    """Rebuild a ``ExperimentEnvelope`` from its stored JSON form, tolerantly.

    Args:
        data: The dict written by ``envelope_to_dict()`` (or junk).

    Returns:
        The typed envelope, or ``None`` when ``data`` is empty, not a dict, or
        fails ``ExperimentEnvelope`` validation (logged at WARNING — a corrupt
        envelope must not brick loading, but silently *narrowing* it would be
        worse than none, so the whole envelope is dropped and the operator is
        told).
    """
    if not isinstance(data, dict) or not data:
        return None
    try:
        bounds = {
            str(vi_name): EnvelopeBound(
                min_value=entry.get("min_value"),
                max_value=entry.get("max_value"),
                state_key=str(entry.get("state_key") or ""),
            )
            for vi_name, entry in data.items()
            if isinstance(entry, dict)
        }
        if not bounds:
            return None
        return ExperimentEnvelope(bounds=bounds)
    except (TypeError, ValueError) as exc:
        logger.warning("session envelope in record is invalid (%s); dropping it", exc)
        return None


@dataclass
class User:
    """One person in the setup-local user roster.

    Identity, not authentication: the roster records who is measuring so runs
    and data files are attributable, and carries the optional link to the
    person's ELN identity for the publishing track.

    Attributes:
        user_id: Unique roster key (a short slug, e.g. ``"jdoe"``).
        name: Display name.
        email: Contact email (optional).
        orcid: ORCID iD (optional).
        eln_user_id: The person's backend-side ELN identity (optional).
    """

    user_id: str = ""
    name: str = ""
    email: str = ""
    orcid: str = ""
    eln_user_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "email": self.email,
            "orcid": self.orcid,
            "eln_user_id": self.eln_user_id,
        }

    @classmethod
    def from_dict(cls, data: object) -> User:
        """Build a ``User`` from a parsed dict, tolerating bad input."""
        if not isinstance(data, dict):
            return cls()
        return cls(
            user_id=_as_str(data.get("user_id")),
            name=_as_str(data.get("name")),
            email=_as_str(data.get("email")),
            orcid=_as_str(data.get("orcid")),
            eln_user_id=_as_str(data.get("eln_user_id")),
        )


@dataclass
class ElnLink:
    """Reference to one ELN page (entry) on one backend.

    Attributes:
        backend: ELN backend identifier (e.g. ``"elabftw"``).
        entry_id: The entry's id on that backend.
        url: Direct URL of the entry.
        template_id: The backend template the entry was created from.
    """

    backend: str = ""
    entry_id: str = ""
    url: str = ""
    template_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "backend": self.backend,
            "entry_id": self.entry_id,
            "url": self.url,
            "template_id": self.template_id,
        }

    @classmethod
    def from_dict(cls, data: object) -> ElnLink:
        """Build an ``ElnLink`` from a parsed dict, tolerating bad input."""
        if not isinstance(data, dict):
            return cls()
        return cls(
            backend=_as_str(data.get("backend")),
            entry_id=_as_str(data.get("entry_id")),
            url=_as_str(data.get("url")),
            template_id=_as_str(data.get("template_id")),
        )


@dataclass
class PinnedBlock:
    """One user block (connector, renderer, profile), pinned by its source digest.

    Attributes:
        block_id: The block's id (a connector's ``backend``, a renderer's or
            profile's name).
        digest: SHA-256 of the block's source when the experiment was linked.
            A block whose file has changed since then is not run for this
            experiment until the user confirms the new version.
    """

    block_id: str = ""
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {"id": self.block_id, "digest": self.digest}

    @classmethod
    def from_dict(cls, data: object) -> PinnedBlock:
        """Build a ``PinnedBlock`` from a parsed dict, tolerating bad input."""
        if not isinstance(data, dict):
            return cls()
        return cls(block_id=_as_str(data.get("id")), digest=_as_str(data.get("digest")))


@dataclass
class LinkedItem:
    """One ELN item (a sample, a resource, a page) linked to the experiment.

    Attributes:
        backend: The backend the item lives on.
        item_id: The item's id there.
        kind: ``"item"`` (a resource or sample database entry) or ``"entry"``.
        role: What the item is to this experiment (``"sample"``): the name
            the profile's read map uses for it.
        title: The item's title when it was linked.
        url: Where a human opens it.
    """

    backend: str = ""
    item_id: str = ""
    kind: str = "item"
    role: str = "sample"
    title: str = ""
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "backend": self.backend,
            "item_id": self.item_id,
            "kind": self.kind,
            "role": self.role,
            "title": self.title,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: object) -> LinkedItem:
        """Build a ``LinkedItem`` from a parsed dict, tolerating bad input."""
        if not isinstance(data, dict):
            return cls()
        return cls(
            backend=_as_str(data.get("backend")),
            item_id=_as_str(data.get("item_id")),
            kind=_as_str(data.get("kind"), "item"),
            role=_as_str(data.get("role"), "sample"),
            title=_as_str(data.get("title")),
            url=_as_str(data.get("url")),
        )


@dataclass
class ElnBinding:
    """How one experiment is connected to its ONE notebook page.

    Written when the experiment is linked (at start, or later), and only by
    the ``ExperimentManager``. The page belongs to the experiment; runs get no
    page of their own. Publishing appends sections to it.

    Attributes:
        account_id: The user's ELN account (per-user settings) the page is
            reached through.
        connector: The connector block, pinned.
        renderer: The renderer block, pinned.
        profile: The profile block, pinned; a copy is kept in the
            experiment's ``eln/profile.yaml``.
        template_id: The template a created page was made from.
        entry: The page, once it exists; ``None`` while its creation is still
            queued (``create_pending``).
        create_pending: A new page was asked for and the backend has not
            confirmed it yet.
        linked_items: Samples and resources linked to the experiment.
        field_snapshot: The last fields read back from the notebook, as
            ``{key: {"value", "unit", "source", "fetched_utc"}}``.
        publish_approved: A human approved publishing for this experiment;
            every later publish appends without asking again.
        approved_by: Who approved (a user id).
        approved_utc: When.
    """

    account_id: str = ""
    connector: PinnedBlock = field(default_factory=PinnedBlock)
    renderer: PinnedBlock = field(default_factory=PinnedBlock)
    profile: PinnedBlock = field(default_factory=PinnedBlock)
    template_id: str = ""
    entry: ElnLink | None = None
    create_pending: bool = False
    linked_items: list[LinkedItem] = field(default_factory=list)
    field_snapshot: dict[str, Any] = field(default_factory=dict)
    publish_approved: bool = False
    approved_by: str = ""
    approved_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "account_id": self.account_id,
            "connector": self.connector.to_dict(),
            "renderer": self.renderer.to_dict(),
            "profile": self.profile.to_dict(),
            "template_id": self.template_id,
            "entry": self.entry.to_dict() if self.entry else None,
            "create_pending": self.create_pending,
            "linked_items": [item.to_dict() for item in self.linked_items],
            "field_snapshot": dict(self.field_snapshot),
            "publish_approved": self.publish_approved,
            "approved_by": self.approved_by,
            "approved_utc": self.approved_utc,
        }

    @classmethod
    def from_dict(cls, data: object) -> ElnBinding:
        """Build an ``ElnBinding`` from a parsed dict, tolerating bad input."""
        if not isinstance(data, dict):
            return cls()
        raw_items = data.get("linked_items")
        return cls(
            account_id=_as_str(data.get("account_id")),
            connector=PinnedBlock.from_dict(data.get("connector")),
            renderer=PinnedBlock.from_dict(data.get("renderer")),
            profile=PinnedBlock.from_dict(data.get("profile")),
            template_id=_as_str(data.get("template_id")),
            entry=ElnLink.from_dict(data["entry"]) if isinstance(data.get("entry"), dict) else None,
            create_pending=_as_bool(data.get("create_pending"), False),
            linked_items=(
                [LinkedItem.from_dict(item) for item in raw_items if isinstance(item, dict)]
                if isinstance(raw_items, list)
                else []
            ),
            field_snapshot=_as_dict(data.get("field_snapshot")),
            publish_approved=_as_bool(data.get("publish_approved"), False),
            approved_by=_as_str(data.get("approved_by")),
            approved_utc=_as_str(data.get("approved_utc")),
        )


@dataclass
class RunRecord:
    """One procedure execution — one HDF5 file — inside an experiment.

    Created by the ``ExperimentManager`` from the Orchestrator's ``run_started``
    manifest and completed from ``run_finished``. ``procedure`` and ``params``
    are kept here so the experiment file answers "which procedure ran run N,
    with what parameters" without opening HDF5 — the basis for searching
    across runs by procedure/param.

    Attributes:
        run_id: The manifest's unique run id.
        procedure: The procedure's display name.
        kind: ``"run"`` for science runs; probe runs will carry ``"probe"``.
        params: Merged parameter values the run executed with.
        params_digest: The **Params digest** of ``params``, stamped when the
            run was opened (``core.plan.params_digest``). Stored rather than
            recomputed on read, so it fixes what the run actually started
            with even if the record is later amended, and so a confirmation
            record carrying the same digest proves the two agree without
            either holding a copy of the other's parameters. Empty on a
            record written before digests were stamped.
        data_file: Absolute path of the run's HDF5 file.
        actor: Who started the run, taken from the ``RunStarted`` event's
            own actor — so an agent-started run is distinguishable from the
            physicist's forever after, not only while the process lives.
        actor_legacy: ``True`` when ``actor`` was not read from the record
            but supplied as the ``OPERATOR`` sentinel because the file
            predates actor recording (or its actor field was unreadable). A
            reader must not treat a legacy actor as evidence of who acted.
        started_utc: ISO 8601 start time (UTC).
        finished_utc: ISO 8601 end time; empty while running.
        status: ``running`` → ``done`` / ``failed`` / ``aborted``.
        reason: Error text for a failed run; empty otherwise.
        published: Whether this run has been published to the experiment's
            notebook page yet (a section covering it was appended).
        eln_link: The per-run ELN entry an OLDER version of I2AS created for
            this run, kept as history and linked from the run's section on
            the experiment's page. New runs never get one.
        selected_bundle: The id of the **analysis bundle** that represents
            this run when it is published, or ``""`` when none is selected
            (the run is then published from its facts). Chosen by the
            analysis stage (the latest completed recipe bundle) or by
            ``select_analysis_bundle``; it knows nothing about any notebook.
        eln_publish: The last publish covering this run, as
            ``{"publish_id", "published_utc", "bundle_id"}``, or ``{}``.
    """

    run_id: str = ""
    procedure: str = ""
    kind: str = "run"
    params: dict[str, Any] = field(default_factory=dict)
    params_digest: str = ""
    data_file: str = ""
    actor: Actor = OPERATOR
    actor_legacy: bool = False
    started_utc: str = ""
    finished_utc: str = ""
    status: str = RUN_STATUS_RUNNING
    reason: str = ""
    published: bool = False
    eln_link: ElnLink | None = None
    selected_bundle: str = ""
    eln_publish: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "run_id": self.run_id,
            "procedure": self.procedure,
            "kind": self.kind,
            "params": dict(self.params),
            "params_digest": self.params_digest,
            "data_file": self.data_file,
            "actor": self.actor.to_json(),
            "actor_legacy": self.actor_legacy,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "status": self.status,
            "reason": self.reason,
            "published": self.published,
            "eln_link": self.eln_link.to_dict() if self.eln_link else None,
            "selected_bundle": self.selected_bundle,
            "eln_publish": dict(self.eln_publish),
        }

    @classmethod
    def from_dict(cls, data: object) -> RunRecord:
        """Build a ``RunRecord`` from a parsed dict, tolerating bad input.

        An unrecognised ``status`` degrades to ``failed`` (never silently back
        to ``running`` — a record whose status cannot be trusted must not look
        like live work). A record with no readable ``actor`` — every run
        written before actors were stamped — loads as the ``OPERATOR``
        sentinel with ``actor_legacy`` set, so "old file" stays
        distinguishable from "the physicist did it".
        """
        if not isinstance(data, dict):
            return cls()
        status = _as_str(data.get("status"), RUN_STATUS_RUNNING)
        if status not in _VALID_RUN_STATUSES:
            status = RUN_STATUS_FAILED
        actor, actor_legacy = _as_actor(data.get("actor"))
        return cls(
            run_id=_as_str(data.get("run_id")),
            procedure=_as_str(data.get("procedure")),
            kind=_as_str(data.get("kind"), "run"),
            params=_as_dict(data.get("params")),
            params_digest=_as_str(data.get("params_digest")),
            data_file=_as_str(data.get("data_file")),
            actor=actor,
            actor_legacy=actor_legacy or _as_bool(data.get("actor_legacy"), False),
            started_utc=_as_str(data.get("started_utc")),
            finished_utc=_as_str(data.get("finished_utc")),
            status=status,
            reason=_as_str(data.get("reason")),
            published=_as_bool(data.get("published"), False),
            eln_link=(
                ElnLink.from_dict(data["eln_link"])
                if isinstance(data.get("eln_link"), dict)
                else None
            ),
            selected_bundle=_as_str(data.get("selected_bundle")),
            eln_publish=_as_dict(data.get("eln_publish")),
        )


@dataclass
class ExperimentRecord:
    """A named group of runs on one sample toward one scientific question.

    The unit the session layer manages and (in the publishing track) mirrors
    to one ELN entry. Persisted as ``experiment.json`` inside the data
    directory, so the record archives with the data it describes.

    Attributes:
        experiment_id: Unique store key, ``NNN_<label>`` (serial number + slug, see
            ``ExperimentStore.make_experiment_id``).
        title: Human title (e.g. "Hall bar A3 — SOT switching vs T").
        user_id: Roster key of the person running the experiment.
        sample_info: The ``{sample_name, sample_id, comments}`` snapshot taken
            when the experiment was started.
        config_name: Identity of the active config at creation.
        created_utc: ISO 8601 creation time (UTC).
        closed_utc: ISO 8601 close time; empty while open.
        status: ``open`` or ``closed``.
        attended: The attendance flag — ``True`` when a human is present.
            An input to the agent gateway's permission matrix (GLOSSARY.md's
            **Attendance**): a ``debug`` role may take ``recovery`` actions
            only while unattended.
        envelope: The session envelope in its JSON form
            (``envelope_to_dict()``); ``{}`` means no envelope.
        run_subfolder: The operator's **run subfolder** inside the
            experiment's ``data/`` folder, where every run started from now
            on is written (``core.run_naming``'s rule); ``""`` — the default,
            and what a record written before it existed reads as — means
            ``data/`` itself. An invalid stored value reads as ``""``.
        runs: The experiment's runs, oldest first.
        findings: Free-text science notes (markdown).
        eln: The experiment's notebook binding (``ElnBinding``): which page
            it publishes to and how. ``None`` while it is not linked.
        queue: The GUI's run queue, as opaque JSON dicts. The session layer
            stores and round-trips this list but never interprets it — the
            GUI (``gui.form_autosave.QueueItemState``) is the only place that
            knows its shape (contract C11: session never imports gui).
        schema_version: The on-disk format version this record was loaded
            from (see ``SCHEMA_VERSION``). Absent on disk ⇒ ``1``. A value
            greater than the running app's ``SCHEMA_VERSION`` means the
            record was written by a newer app; callers must treat it as
            read-only rather than silently tolerant-parsing an unknown
            future shape.
    """

    experiment_id: str = ""
    title: str = ""
    user_id: str = ""
    sample_info: dict[str, Any] = field(default_factory=dict)
    config_name: str = ""
    created_utc: str = ""
    closed_utc: str = ""
    status: str = EXPERIMENT_STATUS_OPEN
    attended: bool = True
    envelope: dict[str, Any] = field(default_factory=dict)
    run_subfolder: str = ""
    runs: list[RunRecord] = field(default_factory=list)
    findings: str = ""
    eln: ElnBinding | None = None
    queue: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation.

        ``schema_version`` is always stamped as the running app's
        ``SCHEMA_VERSION`` (never ``self.schema_version``) — this is the
        write side of the format-version contract; a future-version record
        loaded read-only is never actually re-saved (see
        ``ExperimentManager._save_current``), so this constant stamp is safe.
        """
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "user_id": self.user_id,
            "sample_info": dict(self.sample_info),
            "config_name": self.config_name,
            "created_utc": self.created_utc,
            "closed_utc": self.closed_utc,
            "status": self.status,
            "attended": self.attended,
            "envelope": dict(self.envelope),
            "run_subfolder": self.run_subfolder,
            "runs": [run.to_dict() for run in self.runs],
            "findings": self.findings,
            "eln": self.eln.to_dict() if self.eln else None,
            "queue": [dict(item) for item in self.queue],
            "schema_version": SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, data: object) -> ExperimentRecord:
        """Build an ``ExperimentRecord`` from a parsed dict, tolerating bad input.

        An unrecognised ``status`` degrades to ``closed`` (a record whose
        state cannot be trusted must not resume as the live experiment).
        """
        if not isinstance(data, dict):
            return cls()
        status = _as_str(data.get("status"), EXPERIMENT_STATUS_OPEN)
        if status not in _VALID_EXPERIMENT_STATUSES:
            status = EXPERIMENT_STATUS_CLOSED
        raw_runs = data.get("runs")
        runs = (
            [RunRecord.from_dict(item) for item in raw_runs]
            if isinstance(raw_runs, list)
            else []
        )
        raw_binding = data.get("eln")
        eln = ElnBinding.from_dict(raw_binding) if isinstance(raw_binding, dict) else None
        raw_link = data.get("eln_link")
        if eln is None and isinstance(raw_link, dict) and raw_link.get("entry_id"):
            # Schema 2 kept a bare experiment-level link; it becomes the
            # binding's page, with no account yet (the user re-links it).
            eln = ElnBinding(entry=ElnLink.from_dict(raw_link))
        return cls(
            experiment_id=_as_str(data.get("experiment_id")),
            title=_as_str(data.get("title")),
            user_id=_as_str(data.get("user_id")),
            sample_info=_as_dict(data.get("sample_info")),
            config_name=_as_str(data.get("config_name")),
            created_utc=_as_str(data.get("created_utc")),
            closed_utc=_as_str(data.get("closed_utc")),
            status=status,
            attended=_as_bool(data.get("attended"), True),
            envelope=_as_dict(data.get("envelope")),
            run_subfolder=_as_run_subfolder(data.get("run_subfolder")),
            runs=runs,
            findings=_as_str(data.get("findings")),
            eln=eln,
            queue=_as_dict_list(data.get("queue")),
            # Absent on disk means "today's files" — version 1, not whatever
            # SCHEMA_VERSION the running app happens to define.
            schema_version=_as_int(data.get("schema_version"), 1),
        )

    def find_run(self, run_id: str) -> RunRecord | None:
        """Return the run with ``run_id``, or ``None``.

        Run ids are unique in a record written by this app; in an older
        record that reused a number, the NEWEST run with the id is the one
        returned — the one a ``run_finished`` manifest can still be about.

        Args:
            run_id: The manifest run id to look up.

        Returns:
            The matching ``RunRecord``, or ``None`` when absent.
        """
        for run in reversed(self.runs):
            if run.run_id == run_id:
                return run
        return None

    def highest_run_number(self) -> int:
        """Return the highest ``run-NNNN`` number this experiment ever recorded (0 if none)."""
        from i2as.core.run_naming import run_number_of

        numbers = [run_number_of(run.run_id) for run in self.runs]
        return max((number for number in numbers if number is not None), default=0)


def _as_run_subfolder(value: object) -> str:
    """Read a stored run subfolder, falling back to ``""`` when it breaks the rule.

    A hand-edited or corrupt value must not leave every run refused: the
    experiment simply writes to ``data/`` again, with a warning.
    """
    text = value if isinstance(value, str) else ""
    try:
        return normalize_run_subfolder(text)
    except ValueError as exc:
        logger.warning("Stored run subfolder %r ignored: %s", text, exc)
        return ""


@dataclass
class ExperimentIndexEntry:
    """One line of a session's authoritative experiment index.

    ``Session.experiments`` holds one of these per experiment folder inside
    the session, so a session answers "what experiments do I contain and
    where" from ``session.json`` alone — no directory scan, no opening every
    ``experiment.json``, for whatever reads the index. The index itself is
    rebuilt by ``ExperimentManager._reconcile_session_index()`` — which DOES
    do that scan and open every ``experiment.json`` — on
    ``start_experiment()``/``close_experiment()``/``switch_experiment()``
    (never edited elsewhere), so a folder moved by hand since the last
    reconciliation is reflected in full the next time any of those three
    fire.

    Deliberately carries no data-directory field: an experiment's data
    folder is always ``experiment_id`` + ``"/data"`` (the fixed convention
    ``ExperimentStore.data_dir()`` already computes), and storing a second
    copy of that path here would risk drifting from what the store actually
    uses.

    Attributes:
        experiment_id: The store key — also the experiment's directory name,
            directly under the session folder (flat; no nesting).
        title: Human title, mirrored from ``ExperimentRecord.title``.
        user_id: Roster key of the person who actually ran the experiment,
            mirrored from ``ExperimentRecord.user_id``. Authorship, not
            custody: reconciliation copies it verbatim and never rewrites it
            to match whichever session folder the experiment currently sits
            in, so moving an experiment into a different user's session (to
            hand a project off to whoever is continuing it) never rewrites
            who actually ran it.
        status: ``open`` or ``closed``.
        created_utc: ISO 8601 creation time (UTC), mirrored from the record.
        closed_utc: ISO 8601 close time; empty while open.
    """

    experiment_id: str = ""
    title: str = ""
    user_id: str = ""
    status: str = EXPERIMENT_STATUS_OPEN
    created_utc: str = ""
    closed_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "user_id": self.user_id,
            "status": self.status,
            "created_utc": self.created_utc,
            "closed_utc": self.closed_utc,
        }

    @classmethod
    def from_dict(cls, data: object) -> ExperimentIndexEntry:
        """Build an ``ExperimentIndexEntry`` from a parsed dict, tolerating bad input.

        An unrecognised ``status`` degrades to ``closed`` — same rule as
        ``ExperimentRecord.from_dict``: an index entry lying about "open"
        status is worse than none.
        """
        if not isinstance(data, dict):
            return cls()
        status = _as_str(data.get("status"), EXPERIMENT_STATUS_OPEN)
        if status not in _VALID_EXPERIMENT_STATUSES:
            status = EXPERIMENT_STATUS_CLOSED
        return cls(
            experiment_id=_as_str(data.get("experiment_id")),
            title=_as_str(data.get("title")),
            user_id=_as_str(data.get("user_id")),
            status=status,
            created_utc=_as_str(data.get("created_utc")),
            closed_utc=_as_str(data.get("closed_utc")),
        )


@dataclass
class Session:
    """The new middle tier between the measurement root and an experiment.

    A named, resumable folder holding multiple experiments — the one folder
    the operator chooses, anywhere on disk, with ``<experiment_id>/`` folders
    directly inside it (the layout in ``SessionStore``'s docstring). One user can own several sessions; a session is not
    identified by its owner alone. Persisted as ``session.json`` by
    ``SessionStore``.

    Attributes:
        session_id: The session folder's own name. Set from the folder
            whenever the record is loaded, so a moved or renamed session is
            still itself.
        user_id: Roster key of the session's owner (who created it). A
            record, not a path segment: any user may open any session
            folder.
        name: Display name, user-chosen at creation.
        default_experiment_dir: The saved default parent folder offered when
            starting a new experiment inside this session; user-editable.
            Empty string means "not set yet" (falls back to the session
            folder itself).
        last_open_experiment_id: The experiment id to auto-reopen on resume;
            empty string means none.
        experiments: This session's index of experiment folders — title,
            owner, status, and timestamps for lookup without opening every
            ``experiment.json``. Reconciled by ``ExperimentManager`` against
            a live directory listing on ``start_experiment()``/
            ``close_experiment()``/``switch_experiment()`` (see
            ``ExperimentManager._reconcile_session_index()``); never edited
            elsewhere. Because it is rebuilt from the folder each time
            rather than incrementally patched, an experiment folder moved
            into or out of this session by hand — e.g. handed off to a
            different user's session to continue the project — is picked up
            or dropped the next time any experiment in this session opens
            or closes, with no separate "move" step required. Each entry's
            ``user_id`` records who actually ran that experiment and is
            never rewritten by a move. A session created before this index
            existed starts with an empty list, but because reconciliation
            fully rebuilds from the folder rather than patching one entry,
            the very next start/close/switch in that session backfills every
            experiment folder already on disk — nothing is permanently
            stuck unindexed.
        created_utc: ISO 8601 creation time (UTC).
        last_opened_utc: ISO 8601 time this session was last made active.
        schema_version: The on-disk format version this record was loaded
            from (see ``SCHEMA_VERSION``). Absent on disk ⇒ ``1``. A value
            greater than the running app's ``SCHEMA_VERSION`` means the
            record was written by a newer app; callers must treat it as
            read-only rather than silently tolerant-parsing an unknown
            future shape (same contract as ``ExperimentRecord.schema_version``).
    """

    session_id: str = ""
    user_id: str = ""
    name: str = ""
    default_experiment_dir: str = ""
    last_open_experiment_id: str = ""
    experiments: list[ExperimentIndexEntry] = field(default_factory=list)
    created_utc: str = ""
    last_opened_utc: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation.

        ``schema_version`` is always stamped as the running app's
        ``SCHEMA_VERSION`` (never ``self.schema_version``) — same write-side
        contract as ``ExperimentRecord.to_dict()``.
        """
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "name": self.name,
            "default_experiment_dir": self.default_experiment_dir,
            "last_open_experiment_id": self.last_open_experiment_id,
            "experiments": [entry.to_dict() for entry in self.experiments],
            "created_utc": self.created_utc,
            "last_opened_utc": self.last_opened_utc,
            "schema_version": SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, data: object) -> Session:
        """Build a ``Session`` from a parsed dict, tolerating bad input."""
        if not isinstance(data, dict):
            return cls()
        experiments = [
            ExperimentIndexEntry.from_dict(item)
            for item in _as_dict_list(data.get("experiments"))
        ]
        return cls(
            session_id=_as_str(data.get("session_id")),
            user_id=_as_str(data.get("user_id")),
            name=_as_str(data.get("name")),
            default_experiment_dir=_as_str(data.get("default_experiment_dir")),
            last_open_experiment_id=_as_str(data.get("last_open_experiment_id")),
            experiments=experiments,
            created_utc=_as_str(data.get("created_utc")),
            last_opened_utc=_as_str(data.get("last_opened_utc")),
            # Absent on disk means "today's files" — version 1, not whatever
            # SCHEMA_VERSION the running app happens to define.
            schema_version=_as_int(data.get("schema_version"), 1),
        )


@dataclass
class MaintenanceLogEntry:
    """One revision of one maintenance-log entry (see ``session/maintenance_log.py``).

    Implements the **entry revision** model (GLOSSARY.md): every edit or
    deletion of a logical entry appends a *new* ``MaintenanceLogEntry`` sharing
    the same ``entry_id`` with an incremented ``revision`` rather than rewriting
    anything on disk. ``MaintenanceLogStore.entries()`` presents only the latest,
    non-deleted revision per ``entry_id``; ``revisions()`` returns the full
    history. ``created_utc`` is copied from the first revision and never
    changes, so entries keep a stable creation time across edits.

    Attributes:
        entry_id: Stable id shared by every revision of the same logical entry
            (a ``uuid4`` hex string, assigned on the first revision).
        kind: The declared log kind's key (e.g. ``"maintenance"``).
        values: The entry's field values, keyed by the kind's field names.
        source: Provenance of this entry — ``"manual"`` (a person via the
            GUI) or the name of whatever wrote it.
        run_id: The linked run id when the entry belongs to one; ``""``
            otherwise.
        created_utc: ISO 8601 creation time of the entry's first revision.
        revised_utc: ISO 8601 time this revision was written; ``""`` on the
            first revision.
        revised_by: Who made this revision; ``""`` on the first revision.
        revision: 1-based revision number, incrementing with every edit or
            deletion.
        deleted: ``True`` for a tombstone revision — the entry is hidden from
            ``MaintenanceLogStore.entries()`` but remains in its history.
    """

    entry_id: str = ""
    kind: str = ""
    values: dict[str, Any] = field(default_factory=dict)
    source: str = "manual"
    run_id: str = ""
    created_utc: str = ""
    revised_utc: str = ""
    revised_by: str = ""
    revision: int = 1
    deleted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "entry_id": self.entry_id,
            "kind": self.kind,
            "values": dict(self.values),
            "source": self.source,
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "revised_utc": self.revised_utc,
            "revised_by": self.revised_by,
            "revision": self.revision,
            "deleted": self.deleted,
        }

    @classmethod
    def from_dict(cls, data: object) -> MaintenanceLogEntry:
        """Build a ``MaintenanceLogEntry`` from a parsed dict, tolerating bad input."""
        if not isinstance(data, dict):
            return cls()
        return cls(
            entry_id=_as_str(data.get("entry_id")),
            kind=_as_str(data.get("kind")),
            values=_as_dict(data.get("values")),
            source=_as_str(data.get("source"), "manual"),
            run_id=_as_str(data.get("run_id")),
            created_utc=_as_str(data.get("created_utc")),
            revised_utc=_as_str(data.get("revised_utc")),
            revised_by=_as_str(data.get("revised_by")),
            revision=_as_int(data.get("revision"), 1),
            deleted=_as_bool(data.get("deleted"), False),
        )
