"""Read and explain one experiment folder, after the fact.

The third reading mode of the troubleshoot toolbox, beside ``engine``'s
setup-time instrument checks and ``status_reader``'s live-run digest: this one
answers "what did this experiment actually do?" — which runs executed, who
started each one, in what order, how each ended, how long it took, where its
data file is, whether an incident report was filed alongside it, and what
safety envelope the experiment was running under. It is strictly read-only: it opens no
instruments, writes nothing, and never touches the running application.

**Why it reads JSON instead of importing the session layer.** Import-linter
contract C12 forbids ``i2as.troubleshoot`` from importing
``i2as.session``, so this module cannot reuse ``ExperimentStore``. It
parses the record files directly instead, depending only on their documented
on-disk shape — exactly as ``status_reader`` depends on the ``status.jsonl``
line format rather than on ``i2as.core``. The shapes it relies on are
owned by ``i2as/session/models.py`` (``ExperimentRecord``, ``RunRecord``)
and the folder layout by ``i2as/session/store.py``
(``ExperimentStore``/``SessionStore``); those two files are the contract, and
a change to either belongs here too. Parsing is tolerant in the same spirit as
the session layer's own ``from_dict()``: junk degrades to a default, it never
raises, because a half-written record must still produce a readable report.

The folder layout this walks::

    <measurement_root>/sessions/<user_id>/<session_id>/<experiment_id>/
        experiment.json     the record parsed here
        data/               HDF5 files a run's data_file points into
        incidents/*.md      incident reports filed against this experiment

Incident reports follow the setup-supervisor skill's naming — markdown named
``YYYY-MM-DD-<slug>.md`` inside an ``incidents/`` folder (see GLOSSARY.md,
**Incident report**). The skill writes them under the log directory today;
one copied or written next to the experiment it concerns is picked up here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SESSIONS_DIRNAME = "sessions"
_EXPERIMENT_FILENAME = "experiment.json"
_DATA_DIRNAME = "data"
_INCIDENTS_DIRNAME = "incidents"

# Run statuses in lifecycle order, so the summary line reads the same way
# every time regardless of which ones a given experiment happens to contain.
# Mirrors i2as.session.models RUN_STATUS_* (contract C12 keeps this
# package from importing them).
_RUN_STATUS_ORDER = ("running", "done", "failed", "aborted")


# ── Locating the experiment folder ────────────────────────────────────────────


def find_experiment_dirs(root: Path) -> list[Path]:
    """Return every experiment folder under a measurement root.

    Walks ``<root>/sessions/<user_id>/<session_id>/<experiment_id>/`` — the
    fixed depth ``SessionStore`` and ``ExperimentStore`` write — and keeps
    only folders that actually carry an ``experiment.json``, so a stray
    directory is never mistaken for an experiment.

    Args:
        root: The measurement root to search.

    Returns:
        Every experiment folder found, sorted by path (empty when the root
        or its ``sessions/`` folder does not exist).
    """
    sessions_dir = Path(root) / _SESSIONS_DIRNAME
    if not sessions_dir.is_dir():
        return []
    return sorted(
        path.parent
        for path in sessions_dir.glob(f"*/*/*/{_EXPERIMENT_FILENAME}")
        if path.is_file()
    )


def latest_experiment_dir(root: Path) -> Path | None:
    """Return the most recently modified experiment folder under ``root``.

    "Most recently modified" is the modification time of the folder's own
    ``experiment.json``, not of the folder: the record is rewritten every time
    a run starts or finishes, so it tracks activity, while a directory mtime
    also moves for reasons that have nothing to do with the experiment
    (a data file dropped in by hand, a filesystem copy).

    Args:
        root: The measurement root to search.

    Returns:
        The newest experiment folder, or ``None`` when there is none.
    """
    candidates = find_experiment_dirs(root)
    if not candidates:
        return None
    return max(candidates, key=lambda path: _mtime(path / _EXPERIMENT_FILENAME))


def _mtime(path: Path) -> float:
    """Return ``path``'s modification time, or ``0.0`` when it cannot be read."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# ── Tolerant parsing helpers ──────────────────────────────────────────────────


def _as_str(value: object, default: str = "") -> str:
    """Coerce a JSON value to ``str``, falling back to ``default`` on ``None``."""
    return default if value is None else str(value)


def _as_dict(value: object) -> dict[str, Any]:
    """Return ``value`` if it is a dict, else an empty dict (defensive parse)."""
    return dict(value) if isinstance(value, dict) else {}


# The actor kinds the control contract defines (mirrors core/events.py's
# ActorKind — contract C10 keeps this package out of the Orchestrator's
# modules, and the record is the contract here as everywhere else in the
# troubleshoot toolbox). Anything else in an actor field is unreadable.
_ACTOR_KINDS = ("operator", "agent", "system")

# What a record with no readable actor is called. Every run written before
# actors were stamped is one, and so is any file whose actor field is junk.
_ACTOR_UNKNOWN = "unknown (legacy record)"


def _actor_text(entry: dict[str, Any]) -> str:
    """Say who started a run, applying the session layer's own honesty rule.

    The record's ``actor`` is an `Actor` dict (``kind``, ``id``, ``role``) and
    ``actor_legacy`` flags one the writer could not read. A record with
    neither — every run written before actors were stamped — is not the
    physicist by default: ``session/models.py`` loads it as the operator
    sentinel *with the legacy flag set* precisely so "old file" never reads as
    "the physicist did it", and this reader has to reach the same verdict from
    the same bytes rather than printing a name the record does not support.

    Args:
        entry: One raw run record as read from ``experiment.json``.

    Returns:
        A short phrase naming the actor, or `_ACTOR_UNKNOWN` when the record
        carries no readable one.
    """
    actor = _as_dict(entry.get("actor"))
    kind = _as_str(actor.get("kind"))
    if entry.get("actor_legacy") or kind not in _ACTOR_KINDS:
        return _ACTOR_UNKNOWN
    identity = _as_str(actor.get("id"))
    # The operator sentinel is Actor("operator", id="operator", role="operator"),
    # so a plain human run would otherwise read "operator operator (role
    # operator)". Only the parts that say something new are printed.
    text = kind if identity in ("", kind) else f"{kind} {identity}"
    role = _as_str(actor.get("role"))
    return f"{text} (role {role})" if role and role != kind else text


def _parse_iso(value: str) -> datetime | None:
    """Parse an ISO 8601 timestamp, returning ``None`` on anything unparseable."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _duration_s(started_utc: str, finished_utc: str) -> float | None:
    """Return the run's wall-clock duration in seconds, or ``None``.

    Args:
        started_utc: ISO 8601 start time.
        finished_utc: ISO 8601 end time; empty while the run is still going.

    Returns:
        The elapsed seconds, or ``None`` when either timestamp is missing,
        unparseable, or the two disagree about carrying a timezone (a
        hand-edited record).
    """
    start = _parse_iso(started_utc)
    end = _parse_iso(finished_utc)
    if start is None or end is None:
        return None
    try:
        return (end - start).total_seconds()
    except TypeError:  # one aware, one naive
        return None


# ── Data files ────────────────────────────────────────────────────────────────


def resolve_data_file(experiment_dir: Path, stored: str) -> Path:
    """Resolve a stored ``data_file`` string to a real path, tolerantly.

    The read side of the bundle-relative data-path rule, mirroring
    ``ExperimentStore.resolve_data_file()``: a relative path joins the
    experiment folder; an absolute path is used as-is when it still exists;
    a dangling absolute path (a record whose folder was moved) falls back to
    a recursive basename search under ``data/``; failing that the original
    path is returned unchanged.

    Args:
        experiment_dir: The experiment folder.
        stored: The record's ``data_file`` string as read from disk.

    Returns:
        The best-effort real path to the data file.
    """
    candidate = Path(stored)
    if not candidate.is_absolute():
        return experiment_dir / candidate
    if candidate.exists():
        return candidate
    try:
        match = next((experiment_dir / _DATA_DIRNAME).rglob(candidate.name), None)
    except OSError:
        match = None
    return match if match is not None else candidate


# ── Incident reports ──────────────────────────────────────────────────────────


def find_incidents(experiment_dir: Path) -> list[dict[str, Any]]:
    """Collect the incident reports filed inside an experiment folder.

    A report is any markdown file under an ``incidents/`` folder, or any
    markdown file whose own name mentions an incident (the
    ``YYYY-MM-DD-<slug>.md`` the setup-supervisor skill writes, renamed by
    hand into the experiment folder). Its first ATX heading is used as the
    title, since the skill's template opens with ``# Incident: <symptom>``.

    Args:
        experiment_dir: The experiment folder to scan.

    Returns:
        One dict per report — ``path`` (relative, POSIX), ``title``,
        ``modified_utc``, ``size_bytes`` — sorted by path.
    """
    try:
        markdown = sorted(experiment_dir.rglob("*.md"))
    except OSError as exc:
        logger.warning("Could not scan %s for incident reports: %s", experiment_dir, exc)
        return []

    reports: list[dict[str, Any]] = []
    for path in markdown:
        relative = path.relative_to(experiment_dir)
        in_incidents_folder = _INCIDENTS_DIRNAME in relative.parts[:-1]
        if not in_incidents_folder and "incident" not in path.name.lower():
            continue
        reports.append(
            {
                "path": relative.as_posix(),
                "title": _incident_title(path),
                "modified_utc": _iso_utc(_mtime(path)),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
            }
        )
    return reports


def _incident_title(path: Path) -> str:
    """Return the report's first markdown heading, or its filename stem."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("#"):
                    return line.lstrip("#").strip()
    except OSError as exc:
        logger.warning("Could not read incident report %s: %s", path, exc)
    return path.stem


def _iso_utc(timestamp: float) -> str:
    """Render a POSIX timestamp as an ISO 8601 UTC string, seconds precision."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(
        timespec="seconds"
    )


# ── The report ────────────────────────────────────────────────────────────────


def unavailable(reason: str) -> dict[str, Any]:
    """Return the report payload for "there is nothing to report".

    Args:
        reason: One operator-readable sentence saying what was looked for
            and where.

    Returns:
        A payload with ``available`` false, shaped so a caller can render or
        serialise it exactly like a real report.
    """
    return {"available": False, "reason": reason}


def build_report(experiment_dir: Path) -> dict[str, Any]:
    """Build the JSON-ready report for one experiment folder.

    Args:
        experiment_dir: The experiment folder (the one holding
            ``experiment.json``).

    Returns:
        The report payload. ``available`` is false — with a ``reason`` — when
        the folder has no ``experiment.json`` or that file is not JSON;
        everything else degrades field by field rather than failing, so a
        partially written record still reports the runs it does contain.
    """
    experiment_dir = Path(experiment_dir)
    record_path = experiment_dir / _EXPERIMENT_FILENAME
    try:
        raw = record_path.read_text(encoding="utf-8")
    except OSError:
        return unavailable(f"No experiment record at {record_path}.")
    try:
        data = json.loads(raw)
    except ValueError as exc:
        return unavailable(f"{record_path} is not valid JSON ({exc}).")
    if not isinstance(data, dict):
        return unavailable(f"{record_path} does not hold an experiment record.")

    runs = _build_runs(experiment_dir, data.get("runs"))
    return {
        "available": True,
        "experiment_dir": str(experiment_dir),
        "experiment_id": _as_str(data.get("experiment_id")),
        "title": _as_str(data.get("title")),
        "user_id": _as_str(data.get("user_id")),
        "status": _as_str(data.get("status")),
        "config_name": _as_str(data.get("config_name")),
        "created_utc": _as_str(data.get("created_utc")),
        "closed_utc": _as_str(data.get("closed_utc")),
        "attended": data.get("attended"),
        "sample_info": _as_dict(data.get("sample_info")),
        "schema_version": data.get("schema_version"),
        "envelope": _build_envelope(data.get("envelope")),
        "runs": runs,
        "run_counts": _count_statuses(runs),
        "incidents": find_incidents(experiment_dir),
    }


def _build_runs(experiment_dir: Path, raw_runs: object) -> list[dict[str, Any]]:
    """Render every run record in stored order (oldest first).

    Args:
        experiment_dir: The experiment folder, for resolving data files.
        raw_runs: The record's ``runs`` value (any junk tolerated).

    Returns:
        One dict per run, numbered from 1, each carrying the run's kind,
        procedure, outcome, timestamps, duration, both the stored and the
        resolved data-file path plus whether that file is actually there, the
        starting actor as a short phrase (`_ACTOR_UNKNOWN` when the record
        carries no readable one), and the run's params digest (``""`` when it
        has none).
    """
    if not isinstance(raw_runs, list):
        return []

    runs: list[dict[str, Any]] = []
    for index, item in enumerate(raw_runs, start=1):
        entry = _as_dict(item)
        stored_data_file = _as_str(entry.get("data_file"))
        resolved = resolve_data_file(experiment_dir, stored_data_file) if stored_data_file else None
        started = _as_str(entry.get("started_utc"))
        finished = _as_str(entry.get("finished_utc"))
        runs.append(
            {
                "index": index,
                "run_id": _as_str(entry.get("run_id")),
                "kind": _as_str(entry.get("kind"), "run"),
                "procedure": _as_str(entry.get("procedure")),
                "status": _as_str(entry.get("status")),
                "reason": _as_str(entry.get("reason")),
                "started_utc": started,
                "finished_utc": finished,
                "duration_s": _duration_s(started, finished),
                "data_file": stored_data_file,
                "data_file_path": None if resolved is None else str(resolved),
                "data_file_exists": bool(resolved is not None and resolved.is_file()),
                "actor": _actor_text(entry),
                "params_digest": _as_str(entry.get("params_digest")),
            }
        )
    return runs


def _build_envelope(raw_envelope: object) -> list[dict[str, Any]]:
    """Flatten the stored session envelope into one entry per bounded VI.

    Args:
        raw_envelope: The record's ``envelope`` value — ``{vi_name:
            {min_value, max_value, state_key}}`` as written by
            ``session.models.envelope_to_dict()``.

    Returns:
        One dict per bound, sorted by VI name; empty when no envelope was
        stored.
    """
    bounds = _as_dict(raw_envelope)
    return [
        {
            "vi_name": vi_name,
            "min_value": _as_dict(entry).get("min_value"),
            "max_value": _as_dict(entry).get("max_value"),
            "state_key": _as_str(_as_dict(entry).get("state_key")),
        }
        for vi_name, entry in sorted(bounds.items())
        if isinstance(entry, dict)
    ]


def _count_statuses(runs: list[dict[str, Any]]) -> dict[str, int]:
    """Tally run outcomes, lifecycle order first, then anything unrecognised."""
    counts: dict[str, int] = {}
    for run in runs:
        status = run["status"] or "unknown"
        counts[status] = counts.get(status, 0) + 1
    ordered = {
        status: counts[status] for status in _RUN_STATUS_ORDER if status in counts
    }
    ordered.update(
        {status: count for status, count in sorted(counts.items()) if status not in ordered}
    )
    return ordered


# ── Rendering ─────────────────────────────────────────────────────────────────


def render_text(report: dict[str, Any]) -> str:
    """Render a report as a plain-text block for the CLI and for agents.

    Args:
        report: A ``build_report()`` (or ``unavailable()``) payload.

    Returns:
        The rendered block, without a trailing newline.
    """
    if not report.get("available"):
        return _as_str(report.get("reason"), "No experiment found.")

    lines = [
        f"Experiment: {report['experiment_id']}  ({report['status']})",
        f"Folder:     {report['experiment_dir']}",
    ]
    if report["title"]:
        lines.append(f"Title:      {report['title']}")
    lines.append(
        f"User:       {report['user_id'] or '-'}   "
        f"Config: {report['config_name'] or '-'}   "
        f"Created: {report['created_utc'] or '-'}"
        + (f"   Closed: {report['closed_utc']}" if report["closed_utc"] else "")
    )
    if report["sample_info"]:
        sample = ", ".join(f"{k}={v}" for k, v in sorted(report["sample_info"].items()))
        lines.append(f"Sample:     {sample}")

    lines.extend(_render_envelope(report["envelope"]))
    lines.extend(_render_runs(report["runs"], report["run_counts"]))
    lines.extend(_render_incidents(report["incidents"]))
    return "\n".join(lines)


def _render_envelope(envelope: list[dict[str, Any]]) -> list[str]:
    """Render the session envelope block (empty list when none was stored)."""
    if not envelope:
        return []
    lines = ["Envelope:"]
    for bound in envelope:
        state_key = f"  [{bound['state_key']}]" if bound["state_key"] else ""
        low = "unbounded" if bound["min_value"] is None else str(bound["min_value"])
        high = "unbounded" if bound["max_value"] is None else str(bound["max_value"])
        lines.append(f"  {bound['vi_name']}: {low} .. {high}{state_key}")
    return lines


def _render_runs(runs: list[dict[str, Any]], counts: dict[str, int]) -> list[str]:
    """Render the run list plus its one-line tally."""
    if not runs:
        return ["Runs: none recorded."]

    lines = [f"Runs ({len(runs)}):"]
    for run in runs:
        duration_str = _duration_text(run)
        lines.append(
            f"  {run['index']:>3} [{run['kind']}] {run['procedure'] or '-'}  "
            f"{run['status'] or '-'}  "
            f"{run['started_utc'] or '-'} -> {run['finished_utc'] or '-'}  "
            f"({duration_str})"
        )
        lines.append(f"      started by: {run['actor']}")
        if run["reason"]:
            lines.append(f"      reason: {run['reason']}")
        if run["data_file"]:
            lines.append(f"      data: {_data_file_text(run)}")
        else:
            lines.append("      data: (none recorded)")
    lines.append(
        "=> " + ", ".join(f"{count} {status}" for status, count in counts.items())
    )
    return lines


def _render_incidents(incidents: list[dict[str, Any]]) -> list[str]:
    """Render the incident-report block, or say plainly that there are none."""
    if not incidents:
        return ["Incident reports: none in this folder."]
    lines = [f"Incident reports ({len(incidents)}):"]
    for report in incidents:
        lines.append(f"  {report['path']}  ({report['modified_utc']})")
        lines.append(f"      {report['title']}")
    return lines


def _data_file_text(run: dict[str, Any]) -> str:
    """Render a run's data file: what the record stores, and where it actually is.

    The stored string leads, because a bundle-relative ``data/run_0001.h5``
    read under the folder printed at the top of the report is both shorter and
    exactly what the record says. The resolved path is appended only when the
    two differ — the dangling-absolute case, where the folder was moved and the
    file was found again by basename — since that is the one time the reader
    cannot derive it.
    """
    stored = run["data_file"]
    resolved = run["data_file_path"]
    text = stored if not Path(stored).is_absolute() or resolved == stored else (
        f"{stored} -> {resolved}"
    )
    return text if run["data_file_exists"] else f"{text}  [MISSING]"


def _duration_text(run: dict[str, Any]) -> str:
    """Say how long a run took, distinguishing "not finished" from "cannot tell".

    A run with no ``finished_utc`` is still going; one that has an end but no
    computable duration has timestamps that could not be parsed or compared,
    which is a different thing and must not read as a live run.
    """
    if run["duration_s"] is not None:
        return f"{run['duration_s']:.0f}s"
    return "still running" if not run["finished_utc"] else "duration unknown"
