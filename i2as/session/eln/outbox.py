"""The outbox — the journal that makes publishing offline-first and idempotent.

**Lab networks and notebook servers go down; measurements must not care.**
Nothing is ever sent to a notebook directly. A request (create the page,
publish a section, link items) becomes one **job**: one line appended to
``<experiment>/eln/outbox.jsonl``. The notebook service's worker thread later
performs due jobs one at a time. If the notebook is unreachable, the machine
is rebooted, or the application is closed, the job is still in the file.

**The journal shape**: append-only, one JSON object per line, **the last line
naming a ``job_id`` wins**. A job is never rewritten in place and never
deleted; a corrupt line is skipped with a WARNING.

**Idempotency is by ``job_id``**, derived from what the job does (the page
creation of an experiment, one publish id), so a repeated request appends
nothing. A job that fails half-way resumes where it stopped: the steps it
completed (the page created, each file uploaded, the section appended) are
journaled in its ``progress`` the moment each succeeds.

**Three states.** ``pending`` (due at ``next_due_utc``; a transient failure
pushes it back with exponential backoff, for as long as it takes), ``done``,
and ``needs_attention`` — a failure retrying will not fix (the key was
rejected, the request was refused, the connector crashed or changed). A job
needing attention waits for a person: ``retry()`` puts it back to pending.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JOB_CREATE_ENTRY = "create_entry"
JOB_PUBLISH = "publish"
JOB_LINK_ITEMS = "link_items"
JOB_KINDS: tuple[str, ...] = (JOB_CREATE_ENTRY, JOB_PUBLISH, JOB_LINK_ITEMS)

STATE_PENDING = "pending"
STATE_DONE = "done"
STATE_NEEDS_ATTENTION = "needs_attention"
JOB_STATES: tuple[str, ...] = (STATE_PENDING, STATE_DONE, STATE_NEEDS_ATTENTION)

#: Why a job needs attention (the GUI says what to do about each).
ATTENTION_AUTH = "auth"
ATTENTION_REFUSED = "refused"
ATTENTION_BLOCK = "block"
ATTENTION_CHANGED = "changed"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _parse(stamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class OutboxJob:
    """One unit of notebook work, as journaled.

    Attributes:
        job_id: Derived from what it does; the idempotency key.
        kind: One of ``JOB_KINDS``.
        experiment_id: The experiment it belongs to.
        user_id: Whose account it is performed with.
        account_id: That user's notebook account.
        state: ``pending`` / ``done`` / ``needs_attention``.
        attempts: Failed attempts so far.
        next_due_utc: When a pending job may run next.
        last_error: The last failure, for the GUI.
        attention: Why it needs attention (``auth``, ``refused``, ``block``,
            ``changed``), or ``""``.
        created_utc: When it was queued.
        payload: What to do (kind-specific, JSON-safe, never a secret).
        progress: What has already been done (kind-specific).
    """

    job_id: str
    kind: str
    experiment_id: str
    user_id: str = ""
    account_id: str = ""
    state: str = STATE_PENDING
    attempts: int = 0
    next_due_utc: str = ""
    last_error: str = ""
    attention: str = ""
    created_utc: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "experiment_id": self.experiment_id,
            "user_id": self.user_id,
            "account_id": self.account_id,
            "state": self.state,
            "attempts": self.attempts,
            "next_due_utc": self.next_due_utc,
            "last_error": self.last_error,
            "attention": self.attention,
            "created_utc": self.created_utc,
            "payload": dict(self.payload),
            "progress": dict(self.progress),
        }

    @classmethod
    def from_dict(cls, data: object) -> OutboxJob | None:
        """Load one journal line; ``None`` for anything unusable."""
        if not isinstance(data, dict) or not data.get("job_id") or data.get("kind") not in JOB_KINDS:
            return None
        state = data.get("state")
        attempts = data.get("attempts", 0)
        return cls(
            job_id=str(data["job_id"]),
            kind=str(data["kind"]),
            experiment_id=str(data.get("experiment_id") or ""),
            user_id=str(data.get("user_id") or ""),
            account_id=str(data.get("account_id") or ""),
            state=state if state in JOB_STATES else STATE_PENDING,
            attempts=attempts if isinstance(attempts, int) and not isinstance(attempts, bool) else 0,
            next_due_utc=str(data.get("next_due_utc") or ""),
            last_error=str(data.get("last_error") or ""),
            attention=str(data.get("attention") or ""),
            created_utc=str(data.get("created_utc") or ""),
            payload=dict(data["payload"]) if isinstance(data.get("payload"), dict) else {},
            progress=dict(data["progress"]) if isinstance(data.get("progress"), dict) else {},
        )

    def is_due(self, now: datetime | None = None) -> bool:
        """Whether this pending job may run now."""
        if self.state != STATE_PENDING:
            return False
        due = _parse(self.next_due_utc)
        return due is None or due <= (now or _utc_now())


class Outbox:
    """One experiment's journal of notebook jobs.

    Safe to use from the worker thread and the GUI thread at once: every read
    and append holds one lock.

    Args:
        path: ``<experiment>/eln/outbox.jsonl`` (created on first append).
        retry_base_s: First retry delay; doubles per attempt.
        retry_max_s: Ceiling for the doubling.
    """

    def __init__(self, path: Path, retry_base_s: float = 30.0, retry_max_s: float = 3600.0) -> None:
        self._path = Path(path)
        self.retry_base_s = retry_base_s
        self.retry_max_s = retry_max_s
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def jobs(self) -> dict[str, OutboxJob]:
        """Return every job's latest revision, in first-queued order."""
        with self._lock:
            return self._jobs()

    def _jobs(self) -> dict[str, OutboxJob]:
        jobs: dict[str, OutboxJob] = {}
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return {}
        except OSError as exc:
            logger.warning("Could not read the outbox %s: %s", self._path, exc)
            return {}
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                job = OutboxJob.from_dict(json.loads(line))
            except ValueError:
                job = None
            if job is None:
                logger.warning("Skipping unreadable outbox line %d in %s", number, self._path)
                continue
            jobs[job.job_id] = job
        return jobs

    def get(self, job_id: str) -> OutboxJob | None:
        return self.jobs().get(job_id)

    def _append(self, job: OutboxJob) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(job.to_dict(), ensure_ascii=False) + "\n")

    def enqueue(self, job: OutboxJob) -> bool:
        """Append a new job; a job id already present appends nothing.

        Returns:
            ``True`` when queued.
        """
        with self._lock:
            if job.job_id in self._jobs():
                return False
            self._append(replace(job, state=STATE_PENDING, created_utc=job.created_utc or _iso(_utc_now())))
            return True

    def record(self, job: OutboxJob) -> None:
        """Append one new revision of a job (progress, success, failure)."""
        with self._lock:
            self._append(job)

    def due(self, now: datetime | None = None) -> list[OutboxJob]:
        """Return the pending jobs that may run now, oldest first."""
        return [job for job in self.jobs().values() if job.is_due(now)]

    def failed(self, job: OutboxJob, error: str) -> OutboxJob:
        """Journal a transient failure and back off; returns the new revision."""
        attempts = job.attempts + 1
        delay = min(self.retry_base_s * (2 ** (attempts - 1)), self.retry_max_s)
        revision = replace(job, attempts=attempts, last_error=error[:2000], next_due_utc=_iso(_utc_now() + timedelta(seconds=delay)))
        self.record(revision)
        return revision

    def attention(self, job: OutboxJob, reason: str, error: str) -> OutboxJob:
        """Journal a failure that needs a person; returns the new revision."""
        revision = replace(job, state=STATE_NEEDS_ATTENTION, attention=reason, attempts=job.attempts + 1, last_error=error[:2000])
        self.record(revision)
        return revision

    def retry(self, job_ids: list[str] | None = None, reason: str = "") -> int:
        """Put jobs that need attention back to pending, due now.

        Args:
            job_ids: The jobs; ``None`` for every job needing attention.
            reason: Only those needing attention for this reason; ``""`` any.

        Returns:
            How many were put back.
        """
        count = 0
        for job in self.jobs().values():
            if job.state != STATE_NEEDS_ATTENTION:
                continue
            if job_ids is not None and job.job_id not in job_ids:
                continue
            if reason and job.attention != reason:
                continue
            self.record(replace(job, state=STATE_PENDING, attention="", next_due_utc=""))
            count += 1
        return count
