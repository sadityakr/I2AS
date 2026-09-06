"""The outbox — the journal that makes ELN publishing offline-first.

**Lab networks and notebook servers go down; measurements must not care.**
Nothing in this application ever publishes directly. A finished run is
*rendered* and *queued* — one line appended to a plain JSONL file inside the
experiment's own folder — and a separate, GUI-side drain later hands one
queued job at a time to an ``ElnAdapter``. If the notebook is unreachable, if
the credentials are wrong, if the machine is rebooted mid-day, the job is
still sitting in the file afterwards.

**The journal shape** is the entry-revision model this layer already uses for
the maintenance log: append-only, one JSON object per line, **the last line
naming a ``job_id`` wins**. A job is never rewritten in place and never
deleted, so a crash between two writes can lose at most the newest revision,
never the job. Reads tolerate a corrupt line (skipped with a WARNING) exactly
as ``store.py`` and ``maintenance_log.py`` do.

**Idempotency is by ``job_id``.** The publisher derives a job's id from what
it publishes (``"publish_run:<run_id>"``), so a duplicate ``RunFinished`` —
or a manual export of a run that is already queued — appends nothing and
publishes nothing twice. The same rule covers a *partially* published job:
the entry reference is journaled the instant ``create_entry`` succeeds, so a
retry after a failed attachment resumes at the attachment instead of creating
a second entry.

**Retry state is persisted, not held in memory**: every failure appends a
revision carrying ``attempts``, the error text, and the UTC time the job next
becomes due (exponential backoff from ``retry_base_s``, capped at
``retry_max_s``). There is no terminal "failed" state — an offline week is
exactly the case this exists for — so a job is retried forever, ever more
slowly, until it publishes or a human deletes the file.

``drain()`` **never raises into its caller.** It is wired to a GUI timer, and
a notebook outage must never take the event loop down with it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from i2as.session.eln.adapter import ElnAdapter, ElnEntryRef, ElnError

logger = logging.getLogger(__name__)

#: The one job kind today: publish one finished run as one ELN entry.
JOB_PUBLISH_RUN = "publish_run"

#: A job waiting to be published (the only non-terminal state).
JOB_STATE_PENDING = "pending"

#: A job whose entry exists and whose attachment step completed.
JOB_STATE_DONE = "done"

#: ``drain()`` found nothing due.
DRAIN_IDLE = "idle"

#: ``drain()`` published one job.
DRAIN_PUBLISHED = "published"

#: ``drain()`` tried one job, failed, and rescheduled it.
DRAIN_RETRY = "retry"


def _utc_now() -> datetime:
    """Return the current UTC time as an aware ``datetime``."""
    return datetime.now(timezone.utc)


def _parse_utc(text: str) -> datetime | None:
    """Parse an ISO 8601 timestamp, returning ``None`` on anything unparseable.

    Args:
        text: The stored timestamp string.

    Returns:
        An aware ``datetime``, or ``None`` when the field is empty or junk —
        which callers treat as "due now", never as an error.
    """
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        logger.warning("Unparseable outbox timestamp %r — treating the job as due", text)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _as_str(value: object, default: str = "") -> str:
    """Coerce a JSON value to ``str``, falling back to ``default`` on ``None``."""
    return default if value is None else str(value)


def _as_int(value: object, default: int) -> int:
    """Coerce a JSON value to ``int``, falling back to ``default``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_dict(value: object) -> dict[str, Any]:
    """Return ``value`` as a dict, or an empty dict when it is not one."""
    return dict(value) if isinstance(value, dict) else {}


def _as_str_list(value: object) -> list[str]:
    """Return ``value`` as a list of strings, or an empty list when it is not one."""
    return [str(item) for item in value if item is not None] if isinstance(value, list) else []


def _as_bool(value: object, default: bool) -> bool:
    """Return ``value`` if it is a bool, else ``default`` (defensive parse)."""
    return value if isinstance(value, bool) else default


def _as_attachments(value: object) -> list[dict[str, Any]]:
    """Return ``value`` as a list of ``{"path", "comment"}`` dicts.

    Args:
        value: Any parsed JSON value; anything but a mapping carrying a
            non-empty ``path`` is dropped rather than raised on — one mangled
            line must never strand the job it belongs to.

    Returns:
        The attachments, in order.
    """
    if not isinstance(value, list):
        return []
    attachments: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        path = _as_str(item.get("path"))
        if not path:
            continue
        attachments.append({"path": path, "comment": _as_str(item.get("comment"))})
    return attachments


@dataclass(frozen=True)
class OutboxJob:
    """One queued publish, rendered in full at enqueue time.

    Everything the drain needs is captured here, so a job that finally
    publishes a week later reproduces exactly what the run looked like when it
    finished — the drain never re-renders against state that has since moved
    on.

    Attributes:
        job_id: Stable identity, derived from what is published
            (``"publish_run:<run_id>"``). Two enqueues of the same id are one
            job; the last journaled line for an id is its current state.
        kind: What the job publishes (``"publish_run"`` today).
        created_utc: ISO 8601 time the job was first queued.
        experiment_id: The owning experiment's store key.
        run_id: The run this job publishes.
        title: The entry title.
        body_html: The rendered entry body.
        tags: Tags to set on the entry.
        template_id: Backend template to create from, or ``""`` for the
            backend default.
        metadata: JSON-safe metadata block for the entry.
        data_path: Absolute path of the run's data file, or ``""``.
        attach_data_file: Whether to attach ``data_path`` at all. ``True``
            (the default) is today's behaviour; an analysed entry whose report
            asked for figures alone sets it ``False`` and the data file is
            then named in the body's provenance block and nowhere else.
        attachments: Extra files to attach after the data file, in order, as
            ``{"path": absolute path, "comment": caption}`` — an analysed
            entry's figures. Each is subject to the same caps and the same
            link fallback as the data file.
        max_attachment_bytes: The caller's upload cap; a larger file is
            recorded as a link instead. ``0`` means no cap of the caller's own.
        state: ``"pending"`` or ``"done"``.
        attempts: How many publish attempts have failed so far.
        last_error: The most recent failure text; ``""`` when none.
        next_attempt_utc: ISO 8601 time the job next becomes due; ``""``
            means due now.
        entry: The created entry's ``ElnEntryRef.to_dict()``, journaled the
            instant ``create_entry`` succeeds so a retry never creates a
            second entry. Empty until then.
    """

    job_id: str = ""
    kind: str = JOB_PUBLISH_RUN
    created_utc: str = ""
    experiment_id: str = ""
    run_id: str = ""
    title: str = ""
    body_html: str = ""
    tags: list[str] = field(default_factory=list)
    template_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    data_path: str = ""
    attach_data_file: bool = True
    attachments: list[dict[str, Any]] = field(default_factory=list)
    max_attachment_bytes: int = 0
    state: str = JOB_STATE_PENDING
    attempts: int = 0
    last_error: str = ""
    next_attempt_utc: str = ""
    entry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "created_utc": self.created_utc,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "title": self.title,
            "body_html": self.body_html,
            "tags": list(self.tags),
            "template_id": self.template_id,
            "metadata": dict(self.metadata),
            "data_path": self.data_path,
            "attach_data_file": self.attach_data_file,
            "attachments": [dict(item) for item in self.attachments],
            "max_attachment_bytes": self.max_attachment_bytes,
            "state": self.state,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "next_attempt_utc": self.next_attempt_utc,
            "entry": dict(self.entry),
        }

    @classmethod
    def from_dict(cls, data: object) -> OutboxJob:
        """Build an ``OutboxJob`` from a parsed dict, tolerating bad input.

        Args:
            data: Any parsed JSON value; junk degrades to defaults.

        Returns:
            The job record.
        """
        if not isinstance(data, dict):
            return cls()
        defaults = cls()
        return cls(
            job_id=_as_str(data.get("job_id")),
            kind=_as_str(data.get("kind"), defaults.kind) or defaults.kind,
            created_utc=_as_str(data.get("created_utc")),
            experiment_id=_as_str(data.get("experiment_id")),
            run_id=_as_str(data.get("run_id")),
            title=_as_str(data.get("title")),
            body_html=_as_str(data.get("body_html")),
            tags=_as_str_list(data.get("tags")),
            template_id=_as_str(data.get("template_id")),
            metadata=_as_dict(data.get("metadata")),
            data_path=_as_str(data.get("data_path")),
            attach_data_file=_as_bool(data.get("attach_data_file"), defaults.attach_data_file),
            attachments=_as_attachments(data.get("attachments")),
            max_attachment_bytes=_as_int(data.get("max_attachment_bytes"), 0),
            state=_as_str(data.get("state"), defaults.state) or defaults.state,
            attempts=_as_int(data.get("attempts"), 0),
            last_error=_as_str(data.get("last_error")),
            next_attempt_utc=_as_str(data.get("next_attempt_utc")),
            entry=_as_dict(data.get("entry")),
        )

    def entry_ref(self) -> ElnEntryRef:
        """Return the journaled entry reference (empty when not yet created)."""
        return ElnEntryRef.from_dict(self.entry)

    def is_due(self, now: datetime) -> bool:
        """Whether this job is pending and its backoff has elapsed.

        Args:
            now: The current UTC time.

        Returns:
            ``True`` when the job should be attempted on this drain.
        """
        if self.state != JOB_STATE_PENDING:
            return False
        due = _parse_utc(self.next_attempt_utc)
        return due is None or due <= now


@dataclass(frozen=True)
class DrainResult:
    """What one ``drain()`` call did.

    Attributes:
        state: ``"idle"``, ``"published"``, or ``"retry"``.
        job_id: The job acted on, or ``""`` when idle.
        run_id: The run that job publishes, or ``""``.
        experiment_id: The owning experiment, or ``""``.
        entry: The entry reference after a successful publish, else ``None``.
        detail: Failure text on ``"retry"``; ``""`` otherwise.
        pending: How many jobs remain pending in this outbox afterwards.
    """

    state: str = DRAIN_IDLE
    job_id: str = ""
    run_id: str = ""
    experiment_id: str = ""
    entry: ElnEntryRef | None = None
    detail: str = ""
    pending: int = 0


class Outbox:
    """The append-only publish journal for one experiment.

    One instance per experiment folder. Cheap to construct — the file is read
    on demand, never held open, so an outbox written by an earlier process (or
    by hand) is picked up with no migration step.
    """

    def __init__(
        self, path: Path, retry_base_s: float = 30.0, retry_max_s: float = 3600.0
    ) -> None:
        """Bind to one outbox file.

        Args:
            path: The journal file, from ``ExperimentStore.outbox_path()``
                — the store owns where it sits, this class owns what is in
                it. Neither
                it nor its parent is created until a job is queued.
            retry_base_s: Delay before the first retry of a failed job.
            retry_max_s: Ceiling the doubling backoff never exceeds.
        """
        self._path = Path(path)
        self._retry_base_s = max(float(retry_base_s), 0.0)
        self._retry_max_s = max(float(retry_max_s), self._retry_base_s)

    @property
    def path(self) -> Path:
        """The journal file this outbox reads and appends to."""
        return self._path

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def jobs(self) -> list[OutboxJob]:
        """Return the current state of every job, in first-queued order.

        The last journaled line naming a ``job_id`` is that job's state; a
        line that is not readable JSON is skipped with a WARNING rather than
        failing the read — one mangled line must never strand the rest of the
        queue.

        Returns:
            One ``OutboxJob`` per distinct ``job_id``.
        """
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except OSError as exc:
            logger.warning("Could not read the ELN outbox %s: %s", self._path, exc)
            return []

        latest: dict[str, OutboxJob] = {}
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                logger.warning("Skipping corrupt ELN outbox line %s:%d", self._path, number)
                continue
            job = OutboxJob.from_dict(payload)
            if not job.job_id:
                logger.warning("Skipping ELN outbox line %s:%d with no job_id", self._path, number)
                continue
            latest[job.job_id] = job
        return list(latest.values())

    def get(self, job_id: str) -> OutboxJob | None:
        """Return one job's current state, or ``None`` when it is not queued.

        Args:
            job_id: The job's stable id.

        Returns:
            The job, or ``None``.
        """
        return next((job for job in self.jobs() if job.job_id == job_id), None)

    def pending(self) -> list[OutboxJob]:
        """Return every job still waiting to publish, due or not."""
        return [job for job in self.jobs() if job.state == JOB_STATE_PENDING]

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def enqueue(self, job: OutboxJob) -> bool:
        """Queue one job, unless its ``job_id`` is already known.

        The idempotency point of the whole track: a duplicate
        ``RunFinished``, or a manual export of an already-queued run, appends
        nothing.

        Args:
            job: The job to queue; its ``job_id`` must be non-empty.

        Returns:
            ``True`` when the job was appended, ``False`` when an identical
            id was already queued (published or not).

        Raises:
            ValueError: If ``job.job_id`` is empty.
        """
        if not job.job_id:
            raise ValueError("OutboxJob.job_id must be set before enqueue()")
        if self.get(job.job_id) is not None:
            logger.debug("ELN job %s already queued — not queued again", job.job_id)
            return False
        stamped = replace(
            job,
            created_utc=job.created_utc or _utc_now().isoformat(),
            state=JOB_STATE_PENDING,
        )
        if not self._append(stamped):
            return False
        logger.info(
            "Queued ELN job %s (run=%s, experiment=%s)",
            stamped.job_id,
            stamped.run_id,
            stamped.experiment_id,
        )
        return True

    def _append(self, job: OutboxJob) -> bool:
        """Append one journal line, never raising.

        Args:
            job: The revision to journal.

        Returns:
            ``True`` when the line reached disk; ``False`` (logged) otherwise.
            A failed append leaves the job at its previous journaled state,
            which is the safe direction: a job is retried, never lost.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(job.to_dict(), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("Could not write the ELN outbox %s: %s", self._path, exc)
            return False
        return True

    # ------------------------------------------------------------------
    # Draining
    # ------------------------------------------------------------------

    def drain(self, adapter: ElnAdapter, now: datetime | None = None) -> DrainResult:
        """Attempt at most one due job, and never raise.

        One job per call by design: the caller is a GUI timer, and the
        cooperative single-threaded philosophy of the tick loop applies here
        too — a slow upload delays the next job, never the event loop's next
        turn.

        Args:
            adapter: The backend to publish through.
            now: The current UTC time; defaults to the real clock (injected by
                tests to step over the backoff without sleeping).

        Returns:
            A ``DrainResult`` describing what happened.
        """
        moment = now or _utc_now()
        due = next((job for job in self.jobs() if job.is_due(moment)), None)
        if due is None:
            return DrainResult(state=DRAIN_IDLE, pending=len(self.pending()))
        try:
            published = self._publish(adapter, due)
        except ElnError as exc:
            return self._reschedule(due.job_id, str(exc), moment)
        except Exception as exc:  # an adapter that breaks its contract
            logger.exception("ELN adapter raised a non-ElnError on job %s", due.job_id)
            return self._reschedule(due.job_id, f"{type(exc).__name__}: {exc}", moment)
        self._append(replace(published, state=JOB_STATE_DONE, last_error=""))
        logger.info(
            "Published ELN job %s as entry %s",
            published.job_id,
            published.entry_ref().url or published.entry_ref().entry_id,
        )
        return DrainResult(
            state=DRAIN_PUBLISHED,
            job_id=published.job_id,
            run_id=published.run_id,
            experiment_id=published.experiment_id,
            entry=published.entry_ref(),
            pending=len(self.pending()) - 1,
        )

    def _publish(self, adapter: ElnAdapter, job: OutboxJob) -> OutboxJob:
        """Create the entry (unless already created) and attach the run's data.

        The entry reference is journaled between the two steps, which is what
        makes a retry after a failed attachment resume instead of creating a
        second entry. A retry *after* a successful upload whose response was
        lost may upload twice — the backends accept that, and a duplicate
        attachment is a far smaller problem than a duplicate entry.

        Args:
            adapter: The backend to publish through.
            job: The job to publish.

        Returns:
            The job carrying its entry reference.

        Raises:
            ElnError: Any backend failure; the caller reschedules.
        """
        ref = job.entry_ref()
        if not ref.entry_id:
            ref = adapter.create_entry(
                job.title,
                job.template_id or None,
                job.body_html,
                list(job.tags),
                dict(job.metadata),
            )
            job = replace(job, entry=ref.to_dict())
            self._append(job)
        self._attach_data(adapter, ref, job)
        return job

    def _attach_data(self, adapter: ElnAdapter, ref: ElnEntryRef, job: OutboxJob) -> None:
        """Attach the run's data file (when asked) and then every attachment.

        The data file comes first, and only when the job says so: an analysed
        entry whose report asked for figures alone leaves the raw file where
        it is. Every extra attachment follows in the order the job lists them,
        under exactly the same caps and the same link fallback — a figure that
        is missing or oversized is linked with the reason, so the entry always
        says where the file is even when the bytes stay on the measurement
        machine.

        Args:
            adapter: The backend to publish through.
            ref: The entry to attach to.
            job: The job being published.

        Raises:
            ElnError: An upload or a link was refused.
        """
        capabilities = adapter.capabilities
        caps = [
            limit
            for limit in (job.max_attachment_bytes, capabilities.max_attachment_bytes)
            if limit > 0
        ]
        limit = min(caps) if caps else 0
        if job.attach_data_file and job.data_path:
            self._attach_one(adapter, ref, job, Path(job.data_path), "Run data", limit)
        for attachment in job.attachments:
            path = str(attachment.get("path") or "")
            if not path:
                continue
            comment = str(attachment.get("comment") or "") or Path(path).name
            self._attach_one(adapter, ref, job, Path(path), comment, limit)

    def _attach_one(
        self,
        adapter: ElnAdapter,
        ref: ElnEntryRef,
        job: OutboxJob,
        path: Path,
        comment: str,
        limit: int,
    ) -> None:
        """Upload one file, or record where it lives when it cannot be uploaded.

        Args:
            adapter: The backend to publish through.
            ref: The entry to attach to.
            job: The job being published (for the log line).
            path: The file to attach.
            comment: The caption the notebook shows for it.
            limit: The effective size cap in bytes; ``0`` means uncapped.

        Raises:
            ElnError: The upload or the link was refused.
        """
        capabilities = adapter.capabilities
        size = path.stat().st_size if path.is_file() else -1
        if capabilities.attachments and size >= 0 and (limit == 0 or size <= limit):
            adapter.attach_file(ref, path, comment)
            return
        if capabilities.links:
            reason = (
                "file not found on the publishing machine"
                if size < 0
                else f"file is {size} bytes, above the {limit}-byte attachment cap"
            )
            adapter.attach_link(ref, str(path), f"{comment} — {reason}")
            return
        logger.warning(
            "Backend %s can neither attach nor link %s for job %s",
            adapter.backend,
            path,
            job.job_id,
        )

    def _reschedule(self, job_id: str, detail: str, now: datetime) -> DrainResult:
        """Journal one failed attempt and set the job's next due time.

        Re-reads the job from the journal rather than reusing the revision the
        drain started from: ``_publish`` may already have journaled a created
        entry, and that reference must survive the failure — it is what stops
        the retry from creating a second entry.

        Args:
            job_id: The job that failed.
            detail: The failure text.
            now: The current UTC time.

        Returns:
            A ``retry`` ``DrainResult``.
        """
        job = self.get(job_id) or OutboxJob(job_id=job_id)
        attempts = job.attempts + 1
        delay = min(self._retry_base_s * (2 ** (attempts - 1)), self._retry_max_s)
        retried = replace(
            job,
            attempts=attempts,
            last_error=detail,
            next_attempt_utc=(now + timedelta(seconds=delay)).isoformat(),
        )
        self._append(retried)
        logger.warning(
            "ELN job %s failed (attempt %d): %s — retrying in %.0f s",
            job.job_id,
            attempts,
            detail,
            delay,
        )
        return DrainResult(
            state=DRAIN_RETRY,
            job_id=job.job_id,
            run_id=job.run_id,
            experiment_id=job.experiment_id,
            detail=detail,
            pending=len(self.pending()),
        )
