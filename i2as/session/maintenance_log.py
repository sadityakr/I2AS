"""The maintenance-log framework (L6): declared log kinds and revisioned storage.

A setup keeps one human-facing log per declared **log kind**. A kind is a
key, a title and an ordered ``ParamSpec`` field schema; everything
downstream — storage, the entry-revision model, the table view, the add/edit
dialogs — is generic, so a new kind is one ``LogKindSpec`` here plus one
config line and no new store or GUI code.

See ``GLOSSARY.md`` for the **Maintenance log** / **Log kind** /
**Entry revision** definitions.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from i2as.core.plan import ParamSpec
from i2as.session.models import MaintenanceLogEntry

logger = logging.getLogger(__name__)

__all__ = [
    "LogKindSpec",
    "DECLARED_LOG_KINDS",
    "MaintenanceLogStore",
]


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _append_line(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON line to ``path``, creating parent directories.

    Args:
        path: The JSONL file to append to.
        payload: JSON-serialisable object for the new line.

    Raises:
        OSError: If the directory cannot be created or the file written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_lines_tolerant(path: Path) -> list[dict[str, Any]]:
    """Read every JSON line in ``path``, skipping corrupt ones with a warning.

    Args:
        path: The JSONL file to read.

    Returns:
        One dict per well-formed line, in file order. ``[]`` if the file is
        missing.
    """
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            data = json.loads(raw_line)
        except (TypeError, ValueError) as exc:
            logger.warning("%s: skipping corrupt line %d (%s)", path, lineno, exc)
            continue
        if not isinstance(data, dict):
            logger.warning("%s: skipping non-object line %d", path, lineno)
            continue
        records.append(data)
    return records


# ── Log kinds: declarations ─────────────────────────────────────────────────


@dataclass(frozen=True)
class LogKindSpec:
    """One declared maintenance-log table (GLOSSARY.md: **Log kind**).

    A log kind is a key, a title, and an ordered field schema reusing
    ``ParamSpec`` — the same currency the GUI already renders for procedure
    parameters. Everything downstream (storage, revision handling, table
    view, add/edit dialogs) is generic; adding a log kind for a new setup is
    one ``LogKindSpec`` plus one config line, never new store or GUI code.
    Validates eagerly at construction, mirroring ``core/plan.py``.

    Attributes:
        key: Stable identifier, e.g. ``"maintenance"``. Non-empty, a valid
            Python identifier, and lowercase (used verbatim in file paths).
        title: Human-readable table heading. Non-empty string.
        fields: Ordered, non-empty mapping of field name to ``ParamSpec``.
            Every ``ParamSpec`` already requires a type-matching ``default``
            at its own construction, so every declared field has a usable
            default automatically. Defensively copied.
        editable: Whether entries of this kind may be added/revised/deleted
            through ``MaintenanceLogStore.add_entry`` et al. ``False`` marks a
            read-only stream nothing in this layer writes.
    """

    key: str
    title: str
    fields: dict[str, ParamSpec]
    editable: bool = True

    def __post_init__(self) -> None:
        """Validate the declaration and defensively copy ``fields``.

        Raises:
            TypeError: If ``key``/``title`` is not a str, ``fields`` is not a
                dict, a fields key is not a str, a fields value is not a
                ``ParamSpec``, or ``editable`` is not a bool.
            ValueError: If ``key`` is empty, not a valid identifier, or not
                lowercase; if ``title`` is empty; or if ``fields`` is empty or
                a key is empty.
        """
        if not isinstance(self.key, str):
            raise TypeError(f"LogKindSpec.key must be a str, got {self.key!r}")
        if not self.key or not self.key.isidentifier() or self.key != self.key.lower():
            raise ValueError(
                f"LogKindSpec.key must be a non-empty lowercase identifier, "
                f"got {self.key!r}"
            )

        if not isinstance(self.title, str):
            raise TypeError(f"LogKindSpec.title must be a str, got {self.title!r}")
        if not self.title:
            raise ValueError("LogKindSpec.title must be a non-empty str")

        if not isinstance(self.fields, dict):
            raise TypeError(f"LogKindSpec.fields must be a dict, got {self.fields!r}")
        if not self.fields:
            raise ValueError(f"LogKindSpec({self.key!r}).fields must be a non-empty dict")
        for name, spec in self.fields.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"LogKindSpec({self.key!r}).fields key must be a non-empty str, "
                    f"got {name!r}"
                )
            if not isinstance(spec, ParamSpec):
                raise TypeError(
                    f"LogKindSpec({self.key!r}).fields[{name!r}] must be a ParamSpec, "
                    f"got {spec!r}"
                )
        object.__setattr__(self, "fields", dict(self.fields))

        if not isinstance(self.editable, bool):
            raise TypeError(f"LogKindSpec.editable must be a bool, got {self.editable!r}")


#: The one kind this framework ships: a free-form maintenance entry. A setup
#: adds its own kinds here — one ``LogKindSpec`` plus one config line — and
#: every table, dialog and revision path below stays generic.
_MAINTENANCE_KIND = LogKindSpec(
    key="maintenance",
    title="Maintenance log",
    fields={
        "performed_utc": ParamSpec(
            type=str,
            default="",
            widget_hint="datetime",
            description="When the work was carried out (UTC, ISO 8601)",
        ),
        "person": ParamSpec(
            type=str, default="", description="Who carried it out"
        ),
        "action": ParamSpec(
            type=str, default="", description="What was done"
        ),
        "notes": ParamSpec(
            type=str, default="", description="Free-text notes / corrections"
        ),
    },
    editable=True,
)

#: Registry of every declared log kind. Adding a kind for a new setup is one
#: entry here (plus a config reference) — no other code changes.
DECLARED_LOG_KINDS: dict[str, LogKindSpec] = {
    _MAINTENANCE_KIND.key: _MAINTENANCE_KIND,
}


def _coerce_field(kind: str, name: str, value: Any, spec: ParamSpec) -> Any:
    """Coerce one value against its field's ``ParamSpec``, or raise.

    Mirrors ``ParamSpec._matches_type``'s numeric nuance (an ``int`` is
    accepted where ``float`` is declared; ``bool`` never satisfies a numeric
    or ``str`` type) but additionally *coerces* an accepted ``int`` to
    ``float`` so stored values match the declared type exactly. A field
    declaring ``choices`` further restricts the type-matched value to one of ``spec.choices.values()``.

    Args:
        kind: The owning log kind's key, for error messages.
        name: The field name, for error messages.
        value: The candidate value.
        spec: The field's ``ParamSpec``.

    Returns:
        ``value`` coerced to ``spec.type``.

    Raises:
        ValueError: If ``value`` is not a legal instance of ``spec.type``, or
            the field declares ``choices`` and ``value`` is none of them.
    """
    if spec.type is bool:
        coerced: Any = value
        if not isinstance(value, bool):
            raise ValueError(f"{kind}.{name} must be a bool, got {value!r}")
    elif isinstance(value, bool):
        raise ValueError(f"{kind}.{name} must be a {spec.type.__name__}, got bool {value!r}")
    elif spec.type is float:
        if isinstance(value, (int, float)):
            coerced = float(value)
        else:
            raise ValueError(f"{kind}.{name} must be a real number, got {value!r}")
    elif spec.type is int:
        if isinstance(value, int):
            coerced = value
        else:
            raise ValueError(f"{kind}.{name} must be an int, got {value!r}")
    elif spec.type is str:
        if isinstance(value, str):
            coerced = value
        else:
            raise ValueError(f"{kind}.{name} must be a str, got {value!r}")
    else:
        raise ValueError(f"{kind}.{name} has unsupported field type {spec.type!r}")  # pragma: no cover

    if spec.choices is not None and coerced not in spec.choices.values():
        raise ValueError(
            f"{kind}.{name} must be one of {sorted(spec.choices.values())}, got {coerced!r}"
        )
    return coerced


def _coerce_values(spec: LogKindSpec, values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and coerce a partial values mapping against a log kind's fields.

    Missing fields take the field's ``ParamSpec.default``; unknown keys are
    rejected (never silently dropped, so a typo in a GUI dialog surfaces
    immediately instead of vanishing).

    Args:
        spec: The log kind's declaration.
        values: The candidate values (may be a partial subset of the fields).

    Returns:
        A full ``{field_name: coerced_value}`` dict covering every declared
        field.

    Raises:
        TypeError: If ``values`` is not a mapping.
        ValueError: If ``values`` names a field the kind does not declare, or
            a value cannot be coerced to its field's type.
    """
    if not isinstance(values, dict):
        raise TypeError(f"values for log kind {spec.key!r} must be a dict, got {values!r}")
    unknown = sorted(set(values) - set(spec.fields))
    if unknown:
        raise ValueError(f"log kind {spec.key!r} has no field(s) {unknown}")
    return {
        name: _coerce_field(spec.key, name, values[name], field_spec)
        if name in values
        else field_spec.default
        for name, field_spec in spec.fields.items()
    }


# ── MaintenanceLogStore: editable for humans, append-only on disk ──────────


class MaintenanceLogStore:
    """Per-setup, per-kind append-only storage with the entry-revision model.

    One JSONL file per kind per setup: ``<root>/<config_name>/<kind>.jsonl``.
    Editability comes from the entry-revision model (GLOSSARY.md: **Entry
    revision**) — ``add_entry``/``revise_entry``/``delete_entry`` all *append*
    a new ``MaintenanceLogEntry`` sharing the earlier one's ``entry_id``; nothing
    already on disk is ever rewritten. Readers (``entries``) see the latest,
    non-deleted revision per ``entry_id``; ``revisions`` exposes the full
    history. Writes are validated/coerced against the kind's ``ParamSpec``
    fields; reads tolerate a corrupt line (skipped with a WARNING, never
    raised) exactly like ``session/store.py``.
    """

    def __init__(self, root: Path | str, config_name: str) -> None:
        """Remember the store root without touching the filesystem.

        Args:
            root: Directory holding one subfolder per config (normally
                ``<data_dir>/maintenance``).
            config_name: Identity of the active config; entries of different
                configs never share a file.
        """
        self._root = Path(root)
        self._config_name = config_name

    def _path(self, kind: str) -> Path:
        """Return the JSONL path for ``kind`` (does not check it exists)."""
        return self._root / self._config_name / f"{kind}.jsonl"


    def _spec(self, kind: str) -> LogKindSpec:
        """Return the declared spec for ``kind``.

        Raises:
            ValueError: If ``kind`` is not in ``DECLARED_LOG_KINDS``.
        """
        spec = DECLARED_LOG_KINDS.get(kind)
        if spec is None:
            raise ValueError(
                f"unknown log kind {kind!r}; declared kinds are "
                f"{sorted(DECLARED_LOG_KINDS)}"
            )
        return spec

    def add_entry(
        self,
        kind: str,
        values: dict[str, Any],
        *,
        source: str = "manual",
        person: str = "",
        run_id: str = "",
    ) -> MaintenanceLogEntry:
        """Append a new entry (revision 1) to an editable log kind.

        Args:
            kind: The declared log kind's key.
            values: Field values (a subset is fine; missing fields take their
                declared default).
            source: Provenance — ``"manual"`` (default, a person) or the
                name of whatever wrote it.
            person: Convenience provenance value. If non-empty and the kind
                declares a ``"person"`` field that ``values`` does not already
                set, it is folded into the stored values under that key —
                lets a caller pass the person without duplicating the field
                name.
            run_id: Linked run id when the entry belongs to one.

        Returns:
            The new ``MaintenanceLogEntry`` (also appended to disk).

        Raises:
            ValueError: If ``kind`` is undeclared, not editable, names an
                undeclared field, or a value cannot be coerced.
        """
        spec = self._spec(kind)
        if not spec.editable:
            raise ValueError(f"log kind {kind!r} is not editable")
        merged = dict(values)
        if person and "person" in spec.fields and "person" not in merged:
            merged["person"] = person
        coerced = _coerce_values(spec, merged)
        entry = MaintenanceLogEntry(
            entry_id=uuid.uuid4().hex,
            kind=kind,
            values=coerced,
            source=source,
            run_id=run_id,
            created_utc=_utc_now_iso(),
            revision=1,
        )
        _append_line(self._path(kind), entry.to_dict())
        return entry

    def revise_entry(
        self, kind: str, entry_id: str, values: dict[str, Any], *, revised_by: str
    ) -> MaintenanceLogEntry:
        """Append a new revision of ``entry_id`` with ``values`` merged in.

        Fields not named in ``values`` keep the previous revision's value
        (partial edits — e.g. correcting only ``notes`` — do not need to
        restate the whole entry). ``source``/``run_id``/``created_utc`` carry
        forward from the entry's history unchanged.

        Args:
            kind: The declared log kind's key.
            entry_id: The entry to revise.
            values: The fields to change.
            revised_by: Who made this revision.

        Returns:
            The new ``MaintenanceLogEntry``.

        Raises:
            ValueError: If ``kind`` is undeclared, not editable, ``entry_id``
                has no history, an unknown field is named, or a value cannot
                be coerced.
        """
        spec = self._spec(kind)
        if not spec.editable:
            raise ValueError(f"log kind {kind!r} is not editable")
        history = self.revisions(kind, entry_id)
        if not history:
            raise ValueError(f"no entry {entry_id!r} in log kind {kind!r}")
        latest = history[-1]
        merged = {**latest.values, **values}
        coerced = _coerce_values(spec, merged)
        entry = MaintenanceLogEntry(
            entry_id=entry_id,
            kind=kind,
            values=coerced,
            source=latest.source,
            run_id=latest.run_id,
            created_utc=latest.created_utc,
            revised_utc=_utc_now_iso(),
            revised_by=revised_by,
            revision=latest.revision + 1,
            deleted=False,
        )
        _append_line(self._path(kind), entry.to_dict())
        return entry

    def delete_entry(self, kind: str, entry_id: str, *, revised_by: str) -> MaintenanceLogEntry:
        """Append a tombstone revision of ``entry_id`` (never removes history).

        Args:
            kind: The declared log kind's key.
            entry_id: The entry to delete.
            revised_by: Who deleted it.

        Returns:
            The new tombstone ``MaintenanceLogEntry`` (``deleted=True``).

        Raises:
            ValueError: If ``kind`` is undeclared, not editable, or
                ``entry_id`` has no history.
        """
        spec = self._spec(kind)
        if not spec.editable:
            raise ValueError(f"log kind {kind!r} is not editable")
        history = self.revisions(kind, entry_id)
        if not history:
            raise ValueError(f"no entry {entry_id!r} in log kind {kind!r}")
        latest = history[-1]
        entry = MaintenanceLogEntry(
            entry_id=entry_id,
            kind=kind,
            values=dict(latest.values),
            source=latest.source,
            run_id=latest.run_id,
            created_utc=latest.created_utc,
            revised_utc=_utc_now_iso(),
            revised_by=revised_by,
            revision=latest.revision + 1,
            deleted=True,
        )
        _append_line(self._path(kind), entry.to_dict())
        return entry


    def entries(self, kind: str) -> list[MaintenanceLogEntry]:
        """Return the latest, non-deleted revision of every entry, newest first.

        Args:
            kind: The declared log kind's key.

        Returns:
            One ``MaintenanceLogEntry`` per live ``entry_id``, sorted by
            ``created_utc`` descending (newest first). Corrupt lines are
            skipped with a WARNING log, never raised.

        Raises:
            ValueError: If ``kind`` is undeclared.
        """
        self._spec(kind)
        latest_by_id: dict[str, MaintenanceLogEntry] = {}
        for data in _read_lines_tolerant(self._path(kind)):
            entry = MaintenanceLogEntry.from_dict(data)
            current = latest_by_id.get(entry.entry_id)
            if current is None or entry.revision >= current.revision:
                latest_by_id[entry.entry_id] = entry
        visible = [entry for entry in latest_by_id.values() if not entry.deleted]
        visible.sort(key=lambda entry: entry.created_utc, reverse=True)
        return visible

    def revisions(self, kind: str, entry_id: str) -> list[MaintenanceLogEntry]:
        """Return the full revision history of one entry, oldest first.

        Args:
            kind: The declared log kind's key.
            entry_id: The entry to look up.

        Returns:
            Every revision (including tombstones) sorted by ``revision``
            ascending. ``[]`` if the entry has never been written.

        Raises:
            ValueError: If ``kind`` is undeclared.
        """
        self._spec(kind)
        history = [
            MaintenanceLogEntry.from_dict(data)
            for data in _read_lines_tolerant(self._path(kind))
        ]
        history = [entry for entry in history if entry.entry_id == entry_id]
        history.sort(key=lambda entry: entry.revision)
        return history

