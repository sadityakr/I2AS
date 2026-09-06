"""Disk persistence for sessions, experiments, and the user roster (L6).

Every path helper here is PURE: it says where something belongs and creates
nothing, so pointing a store at a directory that does not exist yet (or is on
an unmounted drive) costs nothing until something is actually written. That
includes the analysis stage's folders (``analysis_dir`` / ``recipes_dir`` /
``report_dir``) — the analysis runner creates the report directory when it
writes a spec into it, and the recipe folder appears when a recipe is first
written.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from i2as.analysis.report import RECIPES_DIRNAME
from i2as.session.models import SCHEMA_VERSION, ExperimentRecord, Session, User

logger = logging.getLogger(__name__)

_EXPERIMENT_FILENAME = "experiment.json"
_SESSION_FILENAME = "session.json"
_ACTIVE_FILENAME = "active.json"
_GUI_STATE_FILENAME = "gui_state.json"
_OUTBOX_FILENAME = "outbox.jsonl"
_AGENT_FEED_FILENAME = "agent_actions.jsonl"
_DATA_DIRNAME = "data"
_ANALYSIS_DIRNAME = "analysis"


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Mirrors ``session.manager._utc_now_iso()``/``session.maintenance_log.
    _utc_now_iso()`` exactly; duplicated rather than imported to avoid a
    circular import (``manager.py`` imports this module).
    """
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: object) -> None:
    """Write ``payload`` as JSON to ``path`` atomically, creating parents.

    Args:
        path: Destination file.
        payload: JSON-serialisable object.

    Raises:
        OSError: If the directory cannot be created or the file written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(tmp_path, path)


def _read_json(path: Path) -> object | None:
    """Read JSON from ``path``, returning ``None`` on any failure (tolerant)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        logger.warning("session store: %s is not valid JSON (%s)", path, exc)
        return None


class ExperimentStore:
    """One-folder-per-experiment store rooted inside the data directory.

    Layout::

        <root>/
            active.json                     {"active": "<experiment_id>", ...}
            <experiment_id>/
                experiment.json
                gui_state.json              # GUI-authored, opaque to this store
                outbox.jsonl                # the ELN publish journal
                agent_actions.jsonl         # the Agent feed
                analysis/                   # the analysis stage's own folder
                    recipes/                # this experiment's recipe scripts
                    <run_id>/               # one run's report.json + figures
                data/                       # HDF5 files; sub-folders allowed
                    <sub-folders>/

    The store creates nothing on construction — directories appear on the
    first ``save()``, so pointing it at a data directory that does not exist
    yet (or is on an unmounted drive) costs nothing until an experiment is
    actually started.
    """

    def __init__(self, root: Path) -> None:
        """Remember the store root without touching the filesystem.

        Args:
            root: Directory holding the experiment folders (normally
                ``<measurement_root>/sessions/<user_id>/<session_id>``, one
                active ``SessionStore`` session's own folder).
        """
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The store's root directory."""
        return self._root

    def make_experiment_id(self, title: str, created_utc: str) -> str:
        """Derive a unique experiment id from the title and creation date.

        ``YYYYMMDD_<slug>`` with a ``_2``, ``_3`` … suffix on collision, so
        ids stay human-readable in the filesystem and unique in the store.

        Args:
            title: The experiment title (any text; slugged).
            created_utc: ISO 8601 creation time (its date part is used).

        Returns:
            A store-unique experiment id.
        """
        date_part = re.sub(r"[^0-9]", "", created_utc[:10]) or "00000000"
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "experiment"
        base = f"{date_part}_{slug}"
        candidate = base
        counter = 2
        existing = set(self.list_experiments())
        while candidate in existing:
            candidate = f"{base}_{counter}"
            counter += 1
        return candidate

    def list_experiments(self) -> list[str]:
        """Return every stored experiment id (sorted; [] when none/no root)."""
        if not self._root.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self._root.iterdir()
            if entry.is_dir() and (entry / _EXPERIMENT_FILENAME).is_file()
        )

    def load(self, experiment_id: str) -> ExperimentRecord | None:
        """Load one experiment record, tolerating a corrupt file.

        Args:
            experiment_id: The store key.

        Returns:
            The record, or ``None`` when missing/unreadable/not JSON. The
            record still loads (tolerant-parse) even when its
            ``schema_version`` is newer than this app's ``SCHEMA_VERSION``
            (logged at WARNING) — callers that must not silently re-save a
            future-format record check ``record.schema_version`` themselves
            (see ``ExperimentManager.switch_experiment``/``_save_current``).
        """
        data = _read_json(self._root / experiment_id / _EXPERIMENT_FILENAME)
        if data is None:
            return None
        record = ExperimentRecord.from_dict(data)
        if record.schema_version > SCHEMA_VERSION:
            logger.warning(
                "Experiment %s was written by a newer app (schema_version=%d > %d); "
                "loading read-only",
                experiment_id,
                record.schema_version,
                SCHEMA_VERSION,
            )
        return record

    def data_dir(self, experiment_id: str) -> Path:
        """Return the experiment's data folder (``<root>/<experiment_id>/data``).

        Args:
            experiment_id: The store key.

        Returns:
            The path, which may not exist yet — nothing here creates it; the
            data manager ``mkdir -p``s it lazily when a run actually saves.
        """
        return self._root / experiment_id / _DATA_DIRNAME

    def gui_state_path(self, experiment_id: str) -> Path:
        """Return the experiment's GUI-state file path.

        Args:
            experiment_id: The store key.

        Returns:
            ``<root>/<experiment_id>/gui_state.json`` (may not exist yet).
        """
        return self._root / experiment_id / _GUI_STATE_FILENAME

    def outbox_path(self, experiment_id: str) -> Path:
        """Return the experiment's ELN publish-journal file path.

        The **Outbox** lives inside the experiment folder, not in a global
        queue, so the folder stays the complete, portable record: copy it and
        its unpublished runs travel with it.

        Args:
            experiment_id: The store key.

        Returns:
            ``<root>/<experiment_id>/outbox.jsonl`` (may not exist yet —
            nothing is written until a run is actually queued).
        """
        return self._root / experiment_id / _OUTBOX_FILENAME

    def agent_feed_path(self, experiment_id: str) -> Path:
        """Return the experiment's **Agent feed** file path.

        The trail of everything a non-operator actor asked for and got lives
        inside the experiment folder for the same reason the **Outbox**
        does: the folder stays the complete, portable record, so copying it
        copies the accountability trail with it.

        Args:
            experiment_id: The store key.

        Returns:
            ``<root>/<experiment_id>/agent_actions.jsonl`` (may not exist yet
            — nothing is written until a non-operator actor acts).
        """
        return self._root / experiment_id / _AGENT_FEED_FILENAME

    def analysis_dir(self, experiment_id: str) -> Path:
        """Return the experiment's analysis folder.

        The analysis stage keeps everything it owns — the experiment's own
        recipe scripts and one folder of results per run — inside the
        experiment folder, for the same reason the **Outbox** and the **Agent
        feed** do: the folder stays the complete, portable record, so copying
        it copies the analysis that produced the entries with it.

        Args:
            experiment_id: The store key.

        Returns:
            ``<root>/<experiment_id>/analysis`` (may not exist yet — nothing
            here creates it; the analysis runner does, when it writes a spec).
        """
        return self._root / experiment_id / _ANALYSIS_DIRNAME

    def recipes_dir(self, experiment_id: str) -> Path:
        """Return the experiment's own **Analysis recipe** folder.

        The per-experiment half of recipe discovery: every ``*.py`` here is
        offered beside the package recipes, and one whose ``name`` matches a
        package recipe replaces it.

        Args:
            experiment_id: The store key.

        Returns:
            ``<root>/<experiment_id>/analysis/recipes`` (may not exist yet).
        """
        return self.analysis_dir(experiment_id) / RECIPES_DIRNAME

    def report_dir(self, experiment_id: str, run_id: str) -> Path:
        """Return where one run's analysis results are written.

        One folder per run, holding the worker's ``spec.json``, its
        ``report.json`` and every figure the recipe saved.

        Args:
            experiment_id: The store key.
            run_id: The analysed run.

        Returns:
            ``<root>/<experiment_id>/analysis/<run_id>`` (may not exist yet).
        """
        return self.analysis_dir(experiment_id) / run_id

    def relativize_data_file(self, experiment_id: str, path: str | Path) -> str:
        """Return ``path`` relative to the experiment's session folder, when inside it.

        The write side of the bundle-relative data-path rule: a run saved
        anywhere under ``<root>/<experiment_id>`` (normally inside ``data/``,
        sub-folders included) is stored relative so the whole folder can be
        copied or moved elsewhere and still resolve. A path outside the
        session folder (the physicist deliberately pointed Data Dir
        elsewhere) is stored absolute, unchanged.

        Args:
            experiment_id: The store key.
            path: The run's data file path, normally absolute.

        Returns:
            A POSIX-style bundle-relative string (e.g. ``"data/xyz.h5"`` or
            ``"data/heating_runs/xyz.h5"``) when ``path`` is inside
            ``<root>/<experiment_id>``, else the absolute path string
            unchanged.
        """
        session_folder = (self._root / experiment_id).resolve()
        resolved = Path(path).resolve()
        if resolved.is_relative_to(session_folder):
            return resolved.relative_to(session_folder).as_posix()
        return str(resolved)

    def resolve_data_file(self, experiment_id: str, stored: str) -> Path:
        """Resolve a stored ``data_file`` string back to a real path, tolerantly.

        The read side of the bundle-relative data-path rule. Resolution
        order: a relative stored path joins the session folder; an absolute
        path is used as-is when it still exists; a dangling absolute path
        (an old record whose session folder was moved) falls back to a
        recursive basename search under ``<root>/<experiment_id>/data``; if
        nothing is found there either, the original path is returned
        unchanged.

        Args:
            experiment_id: The store key.
            stored: The ``RunRecord.data_file`` string as read from disk.

        Returns:
            The best-effort real path to the data file.
        """
        candidate = Path(stored)
        if not candidate.is_absolute():
            return self._root / experiment_id / candidate
        if candidate.exists():
            return candidate
        match = next(self.data_dir(experiment_id).rglob(candidate.name), None)
        return match if match is not None else candidate

    def save(self, record: ExperimentRecord) -> None:
        """Persist ``record`` atomically under its ``experiment_id``.

        Args:
            record: The record to write; ``experiment_id`` must be non-empty.

        Raises:
            ValueError: If ``record.experiment_id`` is empty.
            OSError: If the file cannot be written.
        """
        if not record.experiment_id:
            raise ValueError("ExperimentRecord.experiment_id must be set before save()")
        path = self._root / record.experiment_id / _EXPERIMENT_FILENAME
        _write_json_atomic(path, record.to_dict())

    def get_active(self) -> str | None:
        """Return the persisted active experiment id, or ``None``."""
        data = _read_json(self._root / _ACTIVE_FILENAME)
        if isinstance(data, dict) and isinstance(data.get("active"), str):
            return data["active"] or None
        return None

    def set_active(self, experiment_id: str | None) -> None:
        """Persist (or clear) the active experiment pointer.

        Args:
            experiment_id: The id to resume on next start, or ``None`` to
                clear the pointer.

        Raises:
            OSError: If the pointer file cannot be written.
        """
        _write_json_atomic(
            self._root / _ACTIVE_FILENAME,
            {"active": experiment_id or "", "schema_version": SCHEMA_VERSION},
        )


class SessionStore:
    """One-folder-per-session store rooted at ``<measurement_root>/sessions``.

    The tier above ``ExperimentStore``: a session is a named, resumable,
    per-user folder holding multiple experiments (see ``GLOSSARY.md``'s
    **Session** for the tier and its filesystem layout).
    Sessions nest one level deeper than the store's own root, under their
    owner's ``user_id`` — ownership is structural (a directory), not just a
    field inside ``session.json`` that has to be read to be known. Each
    session's own ``ExperimentStore`` is rooted one level deeper still, at
    ``<root>/<user_id>/<session_id>``.

    Layout::

        <root>/                             <measurement_root>/sessions
            active.json                     {"active_user_id": ..., "active_session_id": ..., ...}
            <user_id>/
                <session_id>/
                    session.json
                    <experiment_id>/         an ExperimentStore rooted here

    This ``active.json`` tracks the one active *session* for the whole
    machine; it is a distinct file from ``ExperimentStore``'s own
    ``active.json``, which lives two levels deeper (inside a session
    folder) and tracks that session's active *experiment*. The two must
    never be confused.

    The store creates nothing on construction — directories appear on the
    first ``save()``, exactly like ``ExperimentStore``.
    """

    def __init__(self, root: Path) -> None:
        """Remember the store root without touching the filesystem.

        Args:
            root: Directory holding the session folders (normally
                ``<measurement_root>/sessions``).
        """
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The store's root directory."""
        return self._root

    def make_session_id(self, name: str, created_utc: str, user_id: str) -> str:
        """Derive a unique session id from the name, date, and owner.

        ``YYYYMMDD_<slug>`` with a ``_2``, ``_3`` … suffix on collision —
        the same scheme as ``ExperimentStore.make_experiment_id``, checked
        against ``list_sessions(user_id)`` instead of ``list_experiments()``.
        Collisions are scoped to one user's own folder, not the whole store:
        two different users picking the same name on the same day never
        fight over a shared suffix counter, since their paths never collide.

        Args:
            name: The session's display name (any text; slugged).
            created_utc: ISO 8601 creation time (its date part is used).
            user_id: Roster key of the intended owner.

        Returns:
            A store-unique (within ``user_id``'s own folder) session id.
        """
        date_part = re.sub(r"[^0-9]", "", created_utc[:10]) or "00000000"
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "session"
        base = f"{date_part}_{slug}"
        candidate = base
        counter = 2
        existing = set(self.list_sessions(user_id))
        while candidate in existing:
            candidate = f"{base}_{counter}"
            counter += 1
        return candidate

    def list_sessions(self, user_id: str) -> list[str]:
        """Return every session id owned by ``user_id`` (sorted; [] when none).

        Ownership is structural (a directory, ``<root>/<user_id>/``), so
        this is a cheap directory listing — no need to open every
        ``session.json`` to filter, unlike the flat, unnested layout this
        replaced.

        Args:
            user_id: Roster key whose sessions to list.

        Returns:
            Sorted session ids owned by ``user_id``.
        """
        user_dir = self._root / user_id
        if not user_dir.is_dir():
            return []
        return sorted(
            entry.name
            for entry in user_dir.iterdir()
            if entry.is_dir() and (entry / _SESSION_FILENAME).is_file()
        )

    def create_session(self, name: str, user_id: str) -> Session:
        """Create, save, and return a new ``Session`` owned by ``user_id``.

        Args:
            name: The session's display name.
            user_id: Roster key of the owner.

        Returns:
            The newly created, already-saved ``Session``.

        Raises:
            OSError: If the file cannot be written.
        """
        created = _utc_now_iso()
        session = Session(
            session_id=self.make_session_id(name, created, user_id),
            user_id=user_id,
            name=name,
            created_utc=created,
            last_opened_utc=created,
        )
        self.save(session)
        return session

    def load(self, user_id: str, session_id: str) -> Session | None:
        """Load one session record, tolerating a corrupt file.

        Args:
            user_id: Roster key of the owner (the session's path segment).
            session_id: The store key.

        Returns:
            The record, or ``None`` when missing/unreadable/not JSON. The
            record still loads (tolerant-parse) even when its
            ``schema_version`` is newer than this app's ``SCHEMA_VERSION``
            (logged at WARNING) — same contract as ``ExperimentStore.load``.
        """
        data = _read_json(self._root / user_id / session_id / _SESSION_FILENAME)
        if data is None:
            return None
        session = Session.from_dict(data)
        if session.schema_version > SCHEMA_VERSION:
            logger.warning(
                "Session %s was written by a newer app (schema_version=%d > %d); "
                "loading read-only",
                session_id,
                session.schema_version,
                SCHEMA_VERSION,
            )
        return session

    def save(self, session: Session) -> None:
        """Persist ``session`` atomically under its ``user_id``/``session_id``.

        Args:
            session: The record to write; ``user_id`` and ``session_id``
                must both be non-empty.

        Raises:
            ValueError: If ``session.user_id`` or ``session.session_id`` is
                empty.
            OSError: If the file cannot be written.
        """
        if not session.user_id:
            raise ValueError("Session.user_id must be set before save()")
        if not session.session_id:
            raise ValueError("Session.session_id must be set before save()")
        path = self._root / session.user_id / session.session_id / _SESSION_FILENAME
        _write_json_atomic(path, session.to_dict())

    def get_active(self) -> tuple[str, str] | None:
        """Return the persisted ``(user_id, session_id)`` active pair, or ``None``.

        Returns ``None`` for an unset pointer, a corrupt file, or a pointer
        written in the pre-per-user-nesting shape (``{"active": "..."}``,
        which has neither of the keys this reads) — all three tolerantly
        fall through to the caller's "unset" bootstrap branch (see
        ``i2as.main._resolve_active_session``).
        """
        data = _read_json(self._root / _ACTIVE_FILENAME)
        if not isinstance(data, dict):
            return None
        user_id = data.get("active_user_id")
        session_id = data.get("active_session_id")
        if isinstance(user_id, str) and user_id and isinstance(session_id, str) and session_id:
            return user_id, session_id
        return None

    def set_active(self, user_id: str, session_id: str) -> None:
        """Persist the active ``(user_id, session_id)`` pair.

        Args:
            user_id: Owner of the session to resume on next start.
            session_id: The session to resume on next start.

        Raises:
            OSError: If the pointer file cannot be written.
        """
        _write_json_atomic(
            self._root / _ACTIVE_FILENAME,
            {
                "active_user_id": user_id,
                "active_session_id": session_id,
                "schema_version": SCHEMA_VERSION,
            },
        )


class UserRoster:
    """The setup-local user roster, one JSON file.

    Identity, not authentication: users belong to the setup (they live next to
    the app settings, not inside one data directory).
    """

    def __init__(self, path: Path) -> None:
        """Remember the roster file path without touching the filesystem.

        Args:
            path: The ``users.json`` file location.
        """
        self._path = Path(path)

    def list_users(self) -> list[User]:
        """Return every roster user (tolerant: [] on a missing/corrupt file)."""
        data = _read_json(self._path)
        if not isinstance(data, list):
            return []
        users = [User.from_dict(item) for item in data]
        return [user for user in users if user.user_id]

    def get(self, user_id: str) -> User | None:
        """Return the user with ``user_id``, or ``None``.

        Args:
            user_id: The roster key to look up.
        """
        for user in self.list_users():
            if user.user_id == user_id:
                return user
        return None

    def add(self, user: User) -> None:
        """Add ``user`` to the roster (replacing any same-``user_id`` entry).

        Args:
            user: The user to store; ``user_id`` must be non-empty.

        Raises:
            ValueError: If ``user.user_id`` is empty.
            OSError: If the roster file cannot be written.
        """
        if not user.user_id:
            raise ValueError("User.user_id must be set before add()")
        users = [u for u in self.list_users() if u.user_id != user.user_id]
        users.append(user)
        _write_json_atomic(self._path, [u.to_dict() for u in users])
