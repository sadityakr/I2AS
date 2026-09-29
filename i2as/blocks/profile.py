"""Profiles — the YAML block that says how an experiment reads from and writes to its page.

A profile is data, not code: which connector it is written for, which renderer
lays out a section, which template a new page is made from, the **read map**
(notebook fields → the experiment's sample metadata) and the **write map**
(page fields ← values I2AS knows). See ``shipped/profiles/default.yaml`` for
the annotated format. Parsing is tolerant: a bad entry is dropped and named in
``Profile.errors`` rather than making the profile unusable, so the Settings
dialog and ``python -m i2as.blocks check`` can say exactly what to fix.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from i2as.blocks.connector import ElnRecord

#: The role naming the experiment's own page in a read map.
PAGE_ROLE = "page"

#: Value types a read-map entry may convert to.
READ_TYPES: tuple[str, ...] = ("text", "float", "int")

#: Sources a write-map entry may name.
WRITE_SOURCES: tuple[str, ...] = ("result", "experiment", "sample", "text")

#: The settings a profile may name; anything else is reported (a typo).
PROFILE_KEYS: tuple[str, ...] = ("profile", "connector", "renderer", "template", "read", "fields", "render")

#: The largest profile file read.
MAX_PROFILE_BYTES = 200_000


@dataclass(frozen=True)
class FieldRead:
    """One read-map entry: a notebook field that fills one sample-metadata key.

    Attributes:
        key: The ``sample_info`` key it fills.
        role: The linked item's role (``"sample"``) or ``"page"``.
        field: The field name on that record.
        value_type: ``text`` / ``float`` / ``int``.
        unit: The unit shown beside the value (the record's own unit wins).
    """

    key: str
    role: str
    field: str
    value_type: str = "text"
    unit: str = ""


@dataclass(frozen=True)
class Profile:
    """One parsed profile.

    Attributes:
        profile_id: The file's stem.
        connector: The connector it is written for, or ``""`` for any.
        renderer: The renderer's id.
        template: The template a new page is created from, or ``""``.
        read: The read map.
        fields: The write map, ``{page field: "<source>:<name>"}``.
        render: Options passed to the renderer.
        errors: What was dropped while parsing, one line each.
    """

    profile_id: str = "default"
    connector: str = ""
    renderer: str = "default"
    template: str = ""
    read: tuple[FieldRead, ...] = ()
    fields: dict[str, str] = field(default_factory=dict)
    render: dict[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()


def _plain(value: Any) -> Any:
    """Convert ruamel's round-trip containers to plain dicts and lists."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def parse_profile(text: str, profile_id: str = "default") -> Profile:
    """Parse profile YAML, tolerantly.

    Args:
        text: The YAML.
        profile_id: The profile's id.

    Returns:
        The profile; ``errors`` names every entry that was dropped.
    """
    errors: list[str] = []
    try:
        loaded = YAML(typ="safe").load(io.StringIO(text or "")) or {}
    except YAMLError as exc:
        return Profile(profile_id=profile_id, errors=(f"not valid YAML: {exc}",))
    data = _plain(loaded)
    if not isinstance(data, dict):
        return Profile(profile_id=profile_id, errors=("the profile must be a mapping",))
    for unknown in sorted(set(data) - set(PROFILE_KEYS)):
        errors.append(f"'{unknown}' is not a profile setting (known: {', '.join(PROFILE_KEYS)})")
    reads: list[FieldRead] = []
    for key, spec in (data.get("read") or {}).items() if isinstance(data.get("read"), dict) else []:
        entry = spec if isinstance(spec, dict) else {"from": spec}
        source = str(entry.get("from") or "")
        role, _, name = source.partition(":")
        value_type = str(entry.get("type") or "text")
        if not (role and name):
            errors.append(f"read.{key}: 'from' must be '<role>:<field>'")
            continue
        if value_type not in READ_TYPES:
            errors.append(f"read.{key}: type must be one of {', '.join(READ_TYPES)}")
            continue
        reads.append(FieldRead(str(key), role.strip(), name.strip(), value_type, str(entry.get("unit") or "")))
    fields: dict[str, str] = {}
    for name, source in (data.get("fields") or {}).items() if isinstance(data.get("fields"), dict) else []:
        text_source = str(source or "")
        kind, _, _rest = text_source.partition(":")
        if kind not in WRITE_SOURCES:
            errors.append(f"fields.{name}: source must start with one of {', '.join(WRITE_SOURCES)}")
            continue
        fields[str(name)] = text_source
    render = data.get("render") if isinstance(data.get("render"), dict) else {}
    return Profile(
        profile_id=profile_id,
        connector=str(data.get("connector") or ""),
        renderer=str(data.get("renderer") or "default"),
        template=str(data.get("template") or ""),
        read=tuple(reads),
        fields=fields,
        render=dict(render),
        errors=tuple(errors),
    )


def load_profile(path: str | Path) -> Profile:
    """Read and parse one profile file (id = the file's stem).

    Args:
        path: The YAML file.

    Returns:
        The profile; an unreadable file yields a profile whose ``errors`` say so.
    """
    file = Path(path)
    try:
        if file.stat().st_size > MAX_PROFILE_BYTES:
            return Profile(profile_id=file.stem, errors=("the profile file is too large",))
        text = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return Profile(profile_id=file.stem, errors=(f"unreadable: {exc}",))
    return parse_profile(text, file.stem)


def _convert(raw: str, value_type: str) -> Any:
    """Convert a notebook field's text to the read map's type (``None`` if it will not)."""
    text = raw.strip()
    if value_type == "text":
        return text
    try:
        number = float(text.replace(",", "."))
    except ValueError:
        return None
    if value_type == "int":
        return int(number) if number.is_integer() else None
    return number


def apply_read_map(profile: Profile, records: Mapping[str, ElnRecord]) -> dict[str, dict[str, Any]]:
    """Return what the read map finds in the linked records.

    Args:
        profile: The profile.
        records: ``{role: record}`` — the experiment's page under ``"page"``
            and each linked item under its role.

    Returns:
        ``{sample_info key: {"value", "unit", "source", "raw", "error"}}`` for
        every read-map entry; ``error`` says why a value is missing or could
        not be converted (``value`` is then ``None``).
    """
    found: dict[str, dict[str, Any]] = {}
    for read in profile.read:
        record = records.get(read.role)
        source = f"{read.role}:{read.field}"
        entry: dict[str, Any] = {"value": None, "unit": read.unit, "source": source, "raw": "", "error": ""}
        if record is None:
            entry["error"] = f"nothing is linked as '{read.role}'"
        elif read.field not in record.fields:
            entry["error"] = f"'{read.field}' is not a field of {record.title or read.role}"
        else:
            raw = record.fields[read.field]
            entry["raw"] = raw
            entry["unit"] = record.units.get(read.field) or read.unit
            entry["value"] = _convert(raw, read.value_type)
            if entry["value"] is None:
                entry["error"] = f"'{raw}' is not a {read.value_type}"
        found[read.key] = entry
    return found


def resolve_fields(
    profile: Profile,
    *,
    experiment: Mapping[str, Any],
    sample_info: Mapping[str, Any],
    results: Mapping[str, Any],
) -> dict[str, str]:
    """Return the page fields the write map sets, as text.

    Args:
        profile: The profile.
        experiment: ``title``, ``experiment_id``, ``user_name``.
        sample_info: The experiment's sample metadata.
        results: ``{result name: value}`` from the newest published runs.

    Returns:
        ``{page field: value}``; a source with no value is left out, so a
        field is never overwritten with nothing.
    """
    values: dict[str, str] = {}
    for name, source in profile.fields.items():
        kind, _, key = source.partition(":")
        value: Any = None
        if kind == "result":
            value = results.get(key)
        elif kind == "experiment":
            value = experiment.get(key)
        elif kind == "sample":
            value = sample_info.get(key)
        elif kind == "text":
            value = key
        if value is None or value == "":
            continue
        values[name] = f"{value:.6g}" if isinstance(value, float) else str(value)
    return values
