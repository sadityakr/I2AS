"""The **Request spool** — a file-based write path into a *running* engine.

The control contract already lets a second client submit commands, but only
in-process: an agent had to be built inside the application to reach
``Orchestrator.submit()`` at all. This module is the first rung of the
transport ladder that fixes that, and it adds **no thread, no socket and no
timer** — the one property that makes it safe in a codebase whose answer to
GPIB races is that there is exactly one writer.

How it works
------------

A client drops one JSON file into a request directory. The engine's own tick
picks it up at the same point it drains queued manual actions, validates it
again there, submits it through the ordinary ``submit()`` path, and deletes
it. The verdict that answers it — and every verdict the engine emits for
anyone — is appended to a JSONL sink the client tails for its own
``request_id``. Nothing here ever executes a command; the tick does, on its
own thread, exactly as it always did.

The spool directory
-------------------

Rooted at ``paths.log_directory()/spool/`` by default::

    spool/
      requests/<request_id>.json   one queued command, deleted when drained
      verdicts.jsonl               every verdict the engine emitted
      events.jsonl                 size-capped tail of state changes/status
      station.json                 the latest declaration snapshot

``requests/<request_id>.json`` is written **atomically** (a dot-prefixed
temporary file in the same directory, then ``rename``), so the tick can never
read a half-written request. Its shape is::

    {"schema": 1, "role": "session", "command": {<Command.to_json()>}}

and three rules make it self-checking:

* ``schema`` must equal ``SCHEMA_VERSION``;
* the file's stem must equal ``command.request_id`` — the correlation id is
  the file's name, which is what lets a request that cannot be parsed at all
  still be answered;
* ``role`` is the authority the request is judged under, stamped onto the
  command's actor. The file's declaration wins over anything inside the
  command, because a client that could write its own role into the actor and
  have it believed would not be judged at all.

Who may write to it
-------------------

Two rules, in this order, before the permission model is ever consulted:

1. **A spooled request is never the operator's.** Its actor kind must be
   ``agent`` or ``system``; an ``operator`` claim is refused ``BLOCKED_ROLE``.
   The operator's authority comes from standing at the cryostat, and a file
   on disk is not a human at a window.
2. **The declared role is capped.** The setup's ``monitor.yaml`` says how
   much authority this door may ever grant (``spool_max_role``, ``observer``
   by default); a request declaring more is refused. The cap is a property of
   the *transport*, not of the agent: a station that wants an unattended
   agent to run experiments through a file drop has to say so.

Only then does the command pass the **Agent gateway**'s own permission model
(the role matrix, **Attendance** and the **Kill switch**), through the
``authorizer`` hook this class is given — a hook rather than an import,
because the permission model lives in the session layer (L6) and nothing in
``core`` may import it. Whoever wires the spool supplies it.

Answers
-------

``verdicts.jsonl`` holds one JSON object per line, oldest first: every
``Verdict.to_json()`` the engine emitted while the spool was attached. A
request the spool could not read into a ``Command`` at all never reaches the
engine, so no ``Verdict`` exists for it; the spool writes the answer itself,
in the same shape with ``"command": null``, so a client still gets exactly
one answer per request it wrote.

``events.jsonl`` is a size-capped tail of ``StateChange`` and
``StatusSnapshot`` — enough for an out-of-process client to answer its reads
from a mirror instead of asking the engine, exactly as an in-process client
does. ``station.json`` holds the latest ``StationInfo`` on its own, rewritten
in place, so the declaration a client renders its tool surface from can never
scroll out of the capped tail.

Nothing in this module raises into the tick: a full disk, a read-only folder
or a mangled line degrades to a logged warning.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from i2as.core.events import (
    ActorKind,
    Command,
    StateChange,
    StationInfo,
    StatusSnapshot,
    Verdict,
    VerdictCode,
)
from i2as.core.paths import log_directory

logger = logging.getLogger(__name__)

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_MAX_ROLE",
    "DEFAULT_EVENT_CAP_BYTES",
    "SPOOL_DIRNAME",
    "Authorizer",
    "RequestSpool",
    "SpooledRequest",
    "spool_directory",
]

#: Version of the request-file shape documented above. Bump it when the shape
#: changes; a file declaring any other value is refused rather than guessed at.
SCHEMA_VERSION = 1

#: The most authority a spooled request may declare when the setup says
#: nothing. The safe end of the ladder: reads only.
DEFAULT_MAX_ROLE = "observer"

#: How large ``events.jsonl`` may grow before its oldest half is dropped. The
#: file is a mirror for a client that just started, not an archive — the
#: engine's own ``status.jsonl`` is the operational record.
DEFAULT_EVENT_CAP_BYTES = 256 * 1024

#: Directory name under the resolved log directory.
SPOOL_DIRNAME = "spool"

_REQUESTS_DIRNAME = "requests"
_VERDICTS_FILENAME = "verdicts.jsonl"
_EVENTS_FILENAME = "events.jsonl"
_STATION_FILENAME = "station.json"

#: How much of ``verdicts.jsonl`` a client reads when tailing it from the end.
_TAIL_BYTES = 256 * 1024

#: The permission hook a spool consults once a request has passed the two
#: transport rules above. Called with keyword arguments only — ``command``,
#: ``declared_role``, ``max_role``, ``station_info``, ``attendance``,
#: ``kill_switch`` and ``seq`` — and answering ``None`` to admit or the
#: ``Verdict`` that refuses. It is a hook rather than an import because the
#: permission model is the session layer's (see
#: ``session/gateway/roles.py``'s ``authorize_spooled()``), and nothing in
#: ``core`` may import upward to reach it.
Authorizer = Callable[..., "Verdict | None"]


def spool_directory() -> Path:
    """Resolve the spool root for this installation, without creating it.

    Returns:
        ``paths.log_directory()/spool`` — see
        :func:`i2as.core.paths.log_directory` for the precedence, which
        is overridable via ``I2AS_LOG_DIR``.
    """
    return log_directory() / SPOOL_DIRNAME


@dataclass(frozen=True)
class SpooledRequest:
    """One request file, read and validated, ready for the engine.

    Exactly one of ``command`` and ``refusal`` is meaningful to the caller:
    a request the spool itself refused carries the answering ``Verdict`` and
    must not be submitted.

    Attributes:
        request_id: The correlation id, which is also the file's name.
        role: The authority the file declared, already stamped onto
            ``command.actor``.
        command: The command to submit, or ``None`` when the spool refused it.
        refusal: The spool's own refusal verdict, or ``None`` when the
            request may go on to the permission model.
    """

    request_id: str
    role: str
    command: Command | None = None
    refusal: Verdict | None = None


class RequestSpool:
    """One spool directory, seen from both ends.

    The engine calls ``take()`` once per tick, ``authorize()`` on what comes
    back, and has ``record_verdict()`` / ``record_event()`` on its two
    broadcast streams. A client calls ``write_request()`` and then
    ``wait_for_verdict()``. The two halves never share anything but files.

    Attributes:
        max_role: The most authority a request may declare here.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        max_role: str = DEFAULT_MAX_ROLE,
        authorizer: Authorizer | None = None,
        event_cap_bytes: int = DEFAULT_EVENT_CAP_BYTES,
    ) -> None:
        """Bind to one spool directory.

        Args:
            root: The spool directory. Defaults to :func:`spool_directory`.
                Nothing is created here; the engine creates the request
                directory when it attaches, and a client creates it when it
                writes.
            max_role: The cap on a request's declared role — the setup's
                ``monitor.yaml`` ``spool_max_role``. Enforced by the
                *authorizer*, which owns the role vocabulary.
            authorizer: The permission hook described in :data:`Authorizer`.
                ``None`` means no permission model has been installed, and
                every request is refused: a door with no lock is not left
                open.
            event_cap_bytes: How large ``events.jsonl`` may grow before its
                oldest half is dropped.
        """
        self._root = Path(root) if root is not None else spool_directory()
        self.max_role = str(max_role)
        self._authorizer = authorizer
        self._event_cap_bytes = int(event_cap_bytes)

    # ── Where things are ──────────────────────────────────────────────

    @property
    def root(self) -> Path:
        """The spool directory."""
        return self._root

    @property
    def requests_dir(self) -> Path:
        """The directory queued request files are dropped into."""
        return self._root / _REQUESTS_DIRNAME

    @property
    def verdicts_path(self) -> Path:
        """The JSONL sink every verdict is appended to."""
        return self._root / _VERDICTS_FILENAME

    @property
    def events_path(self) -> Path:
        """The size-capped JSONL tail of state changes and status snapshots."""
        return self._root / _EVENTS_FILENAME

    @property
    def station_path(self) -> Path:
        """The latest declaration snapshot, rewritten in place."""
        return self._root / _STATION_FILENAME

    def ensure(self) -> bool:
        """Create the request directory, so a client can tell the spool is live.

        Called by whoever attaches the spool to an engine. The directory's
        existence is the client's "the app is running with the spool on"
        signal, which is why it is created eagerly rather than on the first
        drain.

        Returns:
            ``True`` when the directory exists afterwards, ``False`` when it
            could not be created (logged, never raised).
        """
        try:
            self.requests_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create the request spool %s: %s", self._root, exc)
            return False
        logger.info("Request spool open at %s (max role %r)", self._root, self.max_role)
        return True

    def is_open(self) -> bool:
        """Return whether a spool exists here for a client to write into.

        Returns:
            ``True`` when the request directory exists — the client-side test
            for "the application is running and its spool is enabled".
        """
        return self.requests_dir.is_dir()

    # ══════════════════════════════════════════════════════════════════
    # The engine's end
    # ══════════════════════════════════════════════════════════════════

    def take(self) -> list[SpooledRequest]:
        """Read, validate and remove every queued request file.

        Oldest first, by modification time then name, so requests are carried
        out in the order they were dropped. Every file is removed whether it
        was readable or not: a request that stayed on disk would be submitted
        again on the next tick, and a command repeated by a retry loop is
        exactly the failure mode a spool must not have.

        A file that cannot be read into a ``Command`` is answered ``FAILED``
        in the verdict sink here and does not appear in the result — the
        engine is never handed a request it could not have carried out.

        Returns:
            One :class:`SpooledRequest` per readable file, each either
            carrying a command to submit or the spool's own refusal.
        """
        directory = self.requests_dir
        try:
            entries = sorted(
                (path for path in directory.glob("*.json") if path.is_file()),
                key=lambda path: (path.stat().st_mtime_ns, path.name),
            )
        except OSError as exc:
            logger.warning("Could not list the request spool %s: %s", directory, exc)
            return []

        requests: list[SpooledRequest] = []
        for path in entries:
            request = self._take_one(path)
            if request is not None:
                requests.append(request)
        return requests

    def authorize(
        self,
        request: SpooledRequest,
        station_info: StationInfo,
        attendance: bool,
        kill_switch: Any,
        *,
        seq: int = 0,
    ) -> Verdict | None:
        """Put one taken request through the installed permission model.

        Args:
            request: The request, as returned by :meth:`take`.
            station_info: The station's declaration snapshot, which is where
                a ``submit_vi_action``'s action class comes from.
            attendance: Whether a human is watching.
            kill_switch: The engine's ``AgentGate``.
            seq: Sequence number to stamp on a refusal.

        Returns:
            ``None`` when the command may be submitted, or the ``Verdict``
            that refuses it.
        """
        if request.refusal is not None:
            return request.refusal
        if request.command is None:
            return None
        if self._authorizer is None:
            return Verdict(
                request_id=request.request_id,
                command=request.command.name,
                code=VerdictCode.BLOCKED_ROLE,
                actor=request.command.actor,
                reason=(
                    "The request spool has no permission model installed, so "
                    "no spooled request can be authorised; every request is "
                    "refused until one is wired in."
                ),
                detail={"rule": "spool_no_authorizer"},
                seq=seq,
            )
        try:
            return self._authorizer(
                command=request.command,
                declared_role=request.role,
                max_role=self.max_role,
                station_info=station_info,
                attendance=attendance,
                kill_switch=kill_switch,
                seq=seq,
            )
        except Exception as exc:  # noqa: BLE001 — a hook never fails the tick
            logger.exception("request spool authorizer failed")
            return Verdict(
                request_id=request.request_id,
                command=request.command.name,
                code=VerdictCode.FAILED,
                actor=request.command.actor,
                reason=f"The request spool's permission check failed: {exc}",
                detail={"rule": "spool_authorizer_error"},
                seq=seq,
            )

    def record_verdict(self, verdict: Any) -> None:
        """Append one verdict to the sink a client tails.

        Wired to the engine's ``verdict_emitted``. Every verdict is written,
        not only the ones answering spooled requests: a client watching a
        running experiment must see what the human's window sees.

        Args:
            verdict: The ``Verdict`` the engine emitted.
        """
        try:
            if not isinstance(verdict, Verdict):
                return
            self._append(self.verdicts_path, verdict.to_json())
        except Exception:  # noqa: BLE001 — a sink never disturbs the engine
            logger.exception("request spool: recording a verdict failed (non-fatal)")

    def record_event(self, event: Any) -> None:
        """Mirror one engine event into the spool's client-facing files.

        ``StateChange`` and ``StatusSnapshot`` append to the size-capped
        ``events.jsonl``; ``StationInfo`` replaces ``station.json``, so the
        declaration a client renders its tool surface from can never be
        dropped by the cap. Every other event is ignored.

        Args:
            event: Anything on the engine's event stream.
        """
        try:
            if isinstance(event, StationInfo):
                self._write_atomic(self.station_path, event.to_json())
                return
            if isinstance(event, (StateChange, StatusSnapshot)):
                self._append(self.events_path, event.to_json())
                self._cap(self.events_path)
        except Exception:  # noqa: BLE001 — a sink never disturbs the engine
            logger.exception("request spool: recording an event failed (non-fatal)")

    # ══════════════════════════════════════════════════════════════════
    # The client's end
    # ══════════════════════════════════════════════════════════════════

    def write_request(self, command: Command, role: str) -> Path:
        """Drop one command into the spool for the next tick to drain.

        Atomic: the payload is written to a dot-prefixed temporary file in
        the same directory and renamed into place, so the tick either sees
        the whole request or none of it. The temporary name starts with a dot
        and does not end in ``.json``, so a drain in flight never picks it up.

        Args:
            command: The command to submit, already stamped with its actor.
            role: The authority to declare, which is what the request is
                judged under.

        Returns:
            The path of the queued request file.

        Raises:
            OSError: If the spool directory cannot be created or written.
        """
        directory = self.requests_dir
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA_VERSION,
            "role": str(role),
            "command": command.to_json(),
        }
        target = directory / f"{command.request_id}.json"
        temporary = directory / f".{command.request_id}.{os.getpid()}.tmp"
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(target)
        logger.debug("Spooled %s as %s", command.name.value, target.name)
        return target

    def wait_for_verdict(
        self, request_id: str, timeout_s: float, poll_s: float = 0.05
    ) -> dict[str, Any] | None:
        """Wait for the answer to one request, tailing the sink from its end.

        A **client** call, never made from inside the tick: it sleeps. The
        sink is read backwards from its end each poll, so a long-lived
        experiment's history costs nothing to skip past.

        Args:
            request_id: The correlation id to wait for.
            timeout_s: How long to wait before giving up, in seconds.
            poll_s: Gap between reads, in seconds.

        Returns:
            The verdict record as a JSON-safe dict, or ``None`` if none
            arrived within the timeout.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            found = self.find_verdict(request_id)
            if found is not None:
                return found
            if time.monotonic() >= deadline:
                return None
            time.sleep(max(0.0, float(poll_s)))

    def find_verdict(self, request_id: str) -> dict[str, Any] | None:
        """Return the answer to one request if it is already in the sink.

        Args:
            request_id: The correlation id to look for.

        Returns:
            The newest verdict record carrying that id, or ``None``.
        """
        for record in reversed(self.read_verdicts()):
            if record.get("request_id") == request_id:
                return record
        return None

    def read_verdicts(self, max_bytes: int = _TAIL_BYTES) -> list[dict[str, Any]]:
        """Read the tail of the verdict sink, oldest first.

        Args:
            max_bytes: How much of the file's end to read.

        Returns:
            One dict per well-formed line, in file order; ``[]`` when the
            sink does not exist.
        """
        return _tail_records(self.verdicts_path, max_bytes)

    def read_events(self, max_bytes: int = _TAIL_BYTES) -> list[dict[str, Any]]:
        """Read the tail of the mirrored event stream, oldest first.

        Args:
            max_bytes: How much of the file's end to read.

        Returns:
            One dict per well-formed line, in file order; ``[]`` when the
            file does not exist.
        """
        return _tail_records(self.events_path, max_bytes)

    def latest_status(self) -> StatusSnapshot | None:
        """Return the newest mirrored status snapshot.

        Returns:
            The latest ``StatusSnapshot``, or ``None`` when the tail holds
            none (a freshly started app, or a spool that was just cleared).
        """
        for record in reversed(self.read_events()):
            if record.get("kind") == StatusSnapshot.kind:
                try:
                    return StatusSnapshot.from_json(record)
                except (TypeError, ValueError):
                    logger.warning("Skipping an unreadable status record in the spool")
        return None

    def latest_station(self) -> StationInfo | None:
        """Return the mirrored station declaration.

        Returns:
            The latest ``StationInfo``, or ``None`` when the app has not
            published one (or the file is unreadable).
        """
        try:
            raw = self.station_path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            payload = json.loads(raw)
            return StationInfo.from_json(payload)
        except (TypeError, ValueError) as exc:
            logger.warning("Could not read the spooled station declaration: %s", exc)
            return None

    # ══════════════════════════════════════════════════════════════════
    # Internals
    # ══════════════════════════════════════════════════════════════════

    def _take_one(self, path: Path) -> SpooledRequest | None:
        """Read, validate and remove one request file.

        Args:
            path: The request file.

        Returns:
            The request, or ``None`` when it was answered here (unreadable,
            wrong schema, or a command the contract rejects).
        """
        request_id = path.stem
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            self._remove(path)
            self._fail(request_id, f"The request file could not be read: {exc}", {})
            return None
        self._remove(path)

        try:
            payload = json.loads(raw)
        except ValueError as exc:
            self._fail(request_id, f"The request file is not valid JSON: {exc}", {})
            return None
        if not isinstance(payload, dict):
            self._fail(request_id, "The request file is not a JSON object.", {})
            return None

        schema = payload.get("schema", SCHEMA_VERSION)
        if schema != SCHEMA_VERSION:
            self._fail(
                request_id,
                f"The request declares schema {schema!r}; this engine reads "
                f"schema {SCHEMA_VERSION}.",
                {"schema": schema, "expected": SCHEMA_VERSION},
            )
            return None

        try:
            command = Command.from_json(payload.get("command") or {})
        except (TypeError, ValueError) as exc:
            self._fail(request_id, f"The request is not a valid command: {exc}", {})
            return None

        if command.request_id != request_id:
            self._fail(
                request_id,
                f"The request file is named {request_id!r} but its command "
                f"carries request_id {command.request_id!r}; the file's name "
                f"is the correlation id.",
                {"file_request_id": request_id, "command_request_id": command.request_id},
                command=command,
            )
            return None

        declared = payload.get("role")
        role = str(declared) if isinstance(declared, str) and declared else (
            command.actor.role
        )
        command = replace(command, actor=replace(command.actor, role=role))

        if command.actor.kind is not ActorKind.AGENT and (
            command.actor.kind is not ActorKind.SYSTEM
        ):
            return SpooledRequest(
                request_id=request_id,
                role=role,
                refusal=Verdict(
                    request_id=request_id,
                    command=command.name,
                    code=VerdictCode.BLOCKED_ROLE,
                    actor=command.actor,
                    reason=(
                        f"A spooled request may not claim actor kind "
                        f"{command.actor.kind.value!r}: the operator's authority "
                        f"comes from standing at the cryostat, not from a file "
                        f"on disk. Write it as 'agent' or 'system'."
                    ),
                    detail={
                        "rule": "spool_actor_kind",
                        "kind": command.actor.kind.value,
                    },
                ),
            )

        return SpooledRequest(request_id=request_id, role=role, command=command)

    def _fail(
        self,
        request_id: str,
        reason: str,
        detail: dict[str, Any],
        *,
        command: Command | None = None,
    ) -> None:
        """Answer a request the engine will never see, in the sink.

        No ``Verdict`` can exist for a request that never became a command,
        so the record is written in the verdict shape with ``"command":
        null`` — one answer per request written, which is what a waiting
        client depends on.

        Args:
            request_id: The correlation id, from the file's name.
            reason: The operator-readable explanation.
            detail: The structured half, naming the rule that refused.
            command: The command, when it parsed far enough to name one.
        """
        logger.warning("Refusing spooled request %s: %s", request_id, reason)
        self._append(
            self.verdicts_path,
            {
                "request_id": request_id,
                "command": command.name.value if command is not None else None,
                "code": VerdictCode.FAILED.value,
                "actor": command.actor.to_json() if command is not None else None,
                "reason": reason,
                "detail": {"rule": "malformed_request", **detail},
                "result": None,
                "seq": 0,
                "ts": time.time(),
            },
        )

    def _remove(self, path: Path) -> None:
        """Delete one request file, tolerating its absence.

        Args:
            path: The file to remove.
        """
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove the spooled request %s: %s", path, exc)

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        """Append one JSON line, creating the spool directory if needed.

        Args:
            path: The JSONL file.
            payload: The record to write.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=repr) + "\n")
        except OSError as exc:
            logger.warning("Could not append to the request spool %s: %s", path, exc)

    def _write_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        """Replace one file's whole contents, atomically.

        Args:
            path: The file to rewrite.
            payload: The record to write.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, default=repr), encoding="utf-8"
            )
            temporary.replace(path)
        except OSError as exc:
            logger.warning("Could not write the spool file %s: %s", path, exc)

    def _cap(self, path: Path) -> None:
        """Drop the oldest half of a JSONL file once it outgrows its cap.

        Args:
            path: The file to cap.
        """
        try:
            if path.stat().st_size <= self._event_cap_bytes:
                return
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            kept = lines[len(lines) // 2 :]
            self._replace_text(path, "\n".join(kept) + ("\n" if kept else ""))
            logger.debug("Capped %s to its newest %d records", path, len(kept))
        except OSError as exc:
            logger.warning("Could not cap the spool file %s: %s", path, exc)

    @staticmethod
    def _replace_text(path: Path, text: str) -> None:
        """Rewrite a file's whole contents atomically.

        Args:
            path: The file to rewrite.
            text: Its new contents.
        """
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)


def _tail_records(path: Path, max_bytes: int) -> list[dict[str, Any]]:
    """Read the last *max_bytes* of a JSONL file as records, oldest first.

    A partial first line (the read started mid-record) and any line that is
    not a JSON object are skipped, the same tolerance the **Agent feed**'s
    reader applies: one mangled line must never strand the rest.

    Args:
        path: The JSONL file.
        max_bytes: How much of the file's end to read.

    Returns:
        One dict per well-formed line, in file order; ``[]`` when the file
        does not exist or cannot be read.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
                handle.readline()  # drop the partial record we landed inside
            raw = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []

    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records
