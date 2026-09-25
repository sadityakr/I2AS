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

from i2as.analysis.report import RECIPES_DIRNAME, SCRIPTS_DIRNAME
from i2as.session.models import SCHEMA_VERSION, ExperimentRecord, Session, User

logger = logging.getLogger(__name__)

_EXPERIMENT_FILENAME = "experiment.json"
_SESSION_FILENAME = "session.json"
_ACTIVE_FILENAME = "active.json"
_GUI_STATE_FILENAME = "gui_state.json"
_OUTBOX_FILENAME = "outbox.jsonl"
_AGENT_FEED_FILENAME = "agent_actions.jsonl"
_DATA_DIRNAME = "data"

#: Digits of an experiment's serial number (``001_…``). A session with more
#: experiments than this still works; its numbers simply grow a digit.
EXPERIMENT_NUMBER_DIGITS = 3

#: An experiment folder's serial-number prefix: up to six digits, so a
#: date-named folder (``20260717_…``) is never read as experiment 20260717.
_SERIAL_PREFIX = re.compile(r"^(\d{1,6})_")

#: The machine-level registry of session folders, in the measurement root.
SESSIONS_REGISTRY_FILENAME = "sessions.json"

#: Where a session is created when nobody chose a folder (first launch), and
#: where the session folder dialog starts.
DEFAULT_SESSIONS_DIRNAME = "sessions"

#: How many recently active sessions the registry remembers.
MAX_RECENT_SESSIONS = 10
_ANALYSIS_DIRNAME = "analysis"


def is_plain_name(name: object) -> bool:
    """Whether a store key names exactly one folder directly under its root.

    Every experiment id (and session and run id) reaches the filesystem as a
    path segment, and some of them arrive from outside the application — an
    agent's tool call, a spool file. A key that could climb out of its root
    (``..``, a separator, a drive letter, an absolute path) or hide itself
    (a leading dot) is refused, so no caller can make the store read or
    write anywhere but the folder it names.

    Args:
        name: The candidate key.

    Returns:
        ``True`` for a non-empty single path segment that does not start with
        a dot and holds no separator, drive colon or control character.
    """
    if not isinstance(name, str) or not name or name.startswith("."):
        return False
    if any(ch in name for ch in ("/", "\\", ":")) or any(ord(ch) < 32 for ch in name):
        return False
    return Path(name).name == name


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
            <NNN_label>/                    one experiment, serially numbered
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
            root: Directory holding the experiment folders — the session
                folder the operator chose (``SessionStore``), which holds
                ``session.json`` beside the experiments.
        """
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The store's root directory."""
        return self._root

    def experiment_dir(self, experiment_id: str) -> Path:
        """Return one experiment's folder, refusing a key that is not a plain name.

        Every per-experiment path goes through here, so an experiment id that
        arrived from an agent can never name a folder outside this store.

        Args:
            experiment_id: The store key.

        Returns:
            ``<root>/<experiment_id>`` (may not exist yet).

        Raises:
            ValueError: If *experiment_id* is not a plain folder name
                (``is_plain_name``).
        """
        if not is_plain_name(experiment_id):
            raise ValueError(f"{experiment_id!r} is not an experiment id of this store")
        return self._root / experiment_id

    def next_experiment_number(self) -> int:
        """Return the serial number the next experiment in this session gets.

        One more than the highest ``NNN_`` prefix of any experiment folder
        here, so numbers are never reused — even after a folder is deleted
        or moved out — and the session's experiments sort in the order they
        were started.

        Returns:
            The next number, from 1.
        """
        numbers = [
            int(match.group(1))
            for name in self.list_experiments()
            if (match := _SERIAL_PREFIX.match(name))
        ]
        return max(numbers, default=0) + 1

    def make_experiment_id(self, label: str) -> str:
        """Return the next experiment id in this session: ``NNN_<slug>``.

        Every experiment folder in a session carries its serial number first,
        so a whole session reads — and an agent walks it — in the order it
        was measured. The label after it is for people: the operator's own
        folder name, or the experiment's title.

        Args:
            label: The operator's folder name or the experiment title (any
                text; slugged).

        Returns:
            A store-unique experiment id, e.g. ``"003_hall_bar_a3"``.
        """
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "experiment"
        return f"{self.next_experiment_number():0{EXPERIMENT_NUMBER_DIGITS}d}_{slug}"

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
        if not is_plain_name(experiment_id):
            logger.warning("Refusing to load experiment %r: not a plain id", experiment_id)
            return None
        data = _read_json(self.experiment_dir(experiment_id) / _EXPERIMENT_FILENAME)
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
        return self.experiment_dir(experiment_id) / _DATA_DIRNAME

    def gui_state_path(self, experiment_id: str) -> Path:
        """Return the experiment's GUI-state file path.

        Args:
            experiment_id: The store key.

        Returns:
            ``<root>/<experiment_id>/gui_state.json`` (may not exist yet).
        """
        return self.experiment_dir(experiment_id) / _GUI_STATE_FILENAME

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
        return self.experiment_dir(experiment_id) / _OUTBOX_FILENAME

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
        return self.experiment_dir(experiment_id) / _AGENT_FEED_FILENAME

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
        return self.experiment_dir(experiment_id) / _ANALYSIS_DIRNAME

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

    def script_dir(self, experiment_id: str, run_id: str, script_id: str) -> Path:
        """Return where one **analysis script** run over one run lives.

        One folder per script: the script itself, its spec, its report, its
        figures and what it printed. Kept apart from the run's own
        ``report.json`` so exploring a run never replaces the analysed entry
        its recipe produced.

        Args:
            experiment_id: The store key.
            run_id: The run the script analysed.
            script_id: The script's id.

        Returns:
            ``<root>/<experiment_id>/analysis/<run_id>/scripts/<script_id>``
            (may not exist yet).
        """
        return self.report_dir(experiment_id, run_id) / SCRIPTS_DIRNAME / script_id

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
        session_folder = self.experiment_dir(experiment_id).resolve()
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
        folder = self.experiment_dir(experiment_id)
        if not candidate.is_absolute():
            return folder / candidate
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
        path = self.experiment_dir(record.experiment_id) / _EXPERIMENT_FILENAME
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
    """The machine's registry of session folders, and the one way to create them.

    **A session is a folder the operator chooses** — anywhere on disk, picked
    in a folder dialog — and it is the only place on disk they choose. Its
    ``session.json`` names it; everything below it has one fixed shape, so
    a person, a script or an analysis agent can walk a whole session without
    being told where anything is:

    Layout of one session::

        <session folder>/                  chosen by the operator
            session.json                   name, owner, experiment index
            active.json                    the session's active experiment
            001_<experiment>/              an ExperimentStore is rooted at
            002_<experiment>/              the session folder itself

    The registry lives in the machine's measurement root, next to the user
    roster, because which session is active is a fact about this machine::

        <measurement_root>/
            sessions.json                  {"active": <folder>, "recent": [<folder>, …]}
            sessions/                      where a session is created when
                                           nobody chose one (first launch)

    Switching sessions is deferred until the next launch: the open
    ``ExperimentStore`` stays rooted where it started (see ``GLOSSARY.md``'s
    **Session**). The store creates nothing on construction.
    """

    def __init__(self, root: Path) -> None:
        """Remember the registry's folder without touching the filesystem.

        Args:
            root: The measurement root; ``sessions.json`` is read and written
                here.
        """
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The registry's folder (the measurement root)."""
        return self._root

    @property
    def registry_path(self) -> Path:
        """The registry file, ``<root>/sessions.json``."""
        return self._root / SESSIONS_REGISTRY_FILENAME

    def default_parent(self) -> Path:
        """Return where a session nobody chose a folder for is created.

        Returns:
            ``<root>/sessions`` — also where the folder dialog starts.
        """
        return self._root / DEFAULT_SESSIONS_DIRNAME

    @staticmethod
    def is_session_folder(folder: str | Path) -> bool:
        """Whether a folder is a session (holds a ``session.json``).

        Args:
            folder: The folder.

        Returns:
            ``True`` when ``<folder>/session.json`` exists.
        """
        return (Path(folder) / _SESSION_FILENAME).is_file()

    def make_session_folder(self, parent: str | Path, name: str, created_utc: str) -> Path:
        """Return a new, unused ``<parent>/YYYYMMDD_<slug>`` folder path.

        Args:
            parent: The folder the session goes into.
            name: The session's display name (slugged).
            created_utc: ISO 8601 creation time (its date part is used).

        Returns:
            A path that does not exist yet (``_2``, ``_3`` … on collision).
        """
        date_part = re.sub(r"[^0-9]", "", created_utc[:10]) or "00000000"
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "session"
        base = Path(parent) / f"{date_part}_{slug}"
        candidate, counter = base, 2
        while candidate.exists():
            candidate = base.with_name(f"{base.name}_{counter}")
            counter += 1
        return candidate

    def create_session(self, folder: str | Path, name: str, user_id: str) -> Session:
        """Make *folder* a new session, owned by *user_id*, and save it.

        The folder must be new or empty: a session's layout is fixed, and a
        folder that already holds other things would not be one.

        Args:
            folder: The session folder the operator chose.
            name: The session's display name.
            user_id: Roster key of the owner.

        Returns:
            The newly created, already-saved ``Session``.

        Raises:
            ValueError: If *folder* is already a session or is not empty.
            OSError: If the folder or file cannot be written.
        """
        path = Path(folder)
        if self.is_session_folder(path):
            raise ValueError(f"{path} is already a session; open it instead")
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"{path} is not empty; a new session needs an empty folder")
        created = _utc_now_iso()
        session = Session(
            session_id=path.name,
            user_id=user_id,
            name=name,
            created_utc=created,
            last_opened_utc=created,
        )
        self.save(session, path)
        return session

    def load(self, folder: str | Path) -> Session | None:
        """Load one session's record, tolerating a corrupt file.

        Args:
            folder: The session folder.

        Returns:
            The record, or ``None`` when missing/unreadable/not JSON. Its
            ``session_id`` is always the folder's own name, whatever the file
            says, so a session folder that was renamed or moved is still
            itself. A record from a newer app still loads (logged at
            WARNING) — same contract as ``ExperimentStore.load``.
        """
        path = Path(folder)
        data = _read_json(path / _SESSION_FILENAME)
        if data is None:
            return None
        session = Session.from_dict(data)
        session.session_id = path.name
        if session.schema_version > SCHEMA_VERSION:
            logger.warning(
                "Session %s was written by a newer app (schema_version=%d > %d); "
                "loading read-only",
                path,
                session.schema_version,
                SCHEMA_VERSION,
            )
        return session

    def save(self, session: Session, folder: str | Path) -> None:
        """Persist ``session`` atomically as ``<folder>/session.json``.

        Args:
            session: The record to write; ``user_id`` must be non-empty.
            folder: The session folder.

        Raises:
            ValueError: If ``session.user_id`` is empty.
            OSError: If the file cannot be written.
        """
        if not session.user_id:
            raise ValueError("Session.user_id must be set before save()")
        _write_json_atomic(Path(folder) / _SESSION_FILENAME, session.to_dict())

    def _registry(self) -> dict[str, object]:
        data = _read_json(self.registry_path)
        return data if isinstance(data, dict) else {}

    def get_active(self) -> Path | None:
        """Return the active session folder, or ``None``.

        Returns:
            The folder the registry names, when it still is a session;
            ``None`` for an unset pointer, a corrupt file, or a folder that
            is gone or no longer holds a ``session.json`` — all of which fall
            through to the caller's bootstrap branch.
        """
        active = self._registry().get("active")
        if not isinstance(active, str) or not active:
            return None
        folder = Path(active)
        return folder if self.is_session_folder(folder) else None

    def set_active(self, folder: str | Path) -> None:
        """Make *folder* the session the next launch opens, and put it first in recents.

        Args:
            folder: A session folder.

        Raises:
            ValueError: If *folder* is not a session.
            OSError: If the registry cannot be written.
        """
        path = Path(folder).resolve()
        if not self.is_session_folder(path):
            raise ValueError(f"{path} is not a session folder (no session.json)")
        recent = [str(path)] + [
            entry for entry in self._recent_entries() if entry != str(path)
        ]
        _write_json_atomic(
            self.registry_path,
            {
                "active": str(path),
                "recent": recent[:MAX_RECENT_SESSIONS],
                "schema_version": SCHEMA_VERSION,
            },
        )

    def resolve_active(self, user_id: str) -> Path:
        """Return the active session folder, creating one on first launch.

        The application must never refuse to start for lack of a session
        choice: when no session is active (first launch, a corrupt registry,
        a folder that was moved or deleted), a session named after
        *user_id* is created under ``default_parent()`` and made active.

        Args:
            user_id: Who owns a session created here.

        Returns:
            The active session folder.

        Raises:
            OSError: If a bootstrap session cannot be written.
        """
        active = self.get_active()
        if active is not None and self.load(active) is not None:
            return active
        folder = self.make_session_folder(self.default_parent(), user_id, _utc_now_iso())
        self.create_session(folder, name=user_id, user_id=user_id)
        self.set_active(folder)
        return folder.resolve()

    def _recent_entries(self) -> list[str]:
        recent = self._registry().get("recent")
        return [entry for entry in recent if isinstance(entry, str)] if isinstance(recent, list) else []

    def recent(self) -> list[Path]:
        """Return the recently active session folders that still exist, newest first.

        Returns:
            Up to ``MAX_RECENT_SESSIONS`` folders, each holding a
            ``session.json``.
        """
        return [
            Path(entry) for entry in self._recent_entries() if self.is_session_folder(entry)
        ]


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
