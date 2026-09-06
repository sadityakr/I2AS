"""The capability manifest — the JSON rendering of the station declaration.

Input: a built ``Station``. Process: read its ``StationInfo`` declaration
snapshot and render it as plain JSON, resolving each instrument's capabilities
into its declared UI groups (in declared order) with the ungrouped ones after.
Output: a JSON-safe ``dict``, plus the schema that describes it and a
validator that checks one against the other.

**Why a manifest at all.** An agent — and a GUI, and any future gateway —
must be able to answer "what can this system do?" without a human writing a
word of glue. That answer is *generated* from the same declarations that
drive everything else (``@monitored(unit=, description=, group=)``,
``@control(params=, scope=, action_class=, panel=, group=)``, ``control_limits``,
``ui_groups``, ``safety_flags``), never hand-maintained, because a
hand-maintained description drifts from the instrument in the first week.
This module writes no description of its own; every string in its output
came from a declaration on a VI or a value in a config.

**Three modules, three jobs.** ``core.events`` DEFINES the declaration's
shape (``StationInfo`` and its nested types) and depends on nothing.
``Station.station_info()`` BUILDS it, because only the Station holds both the
VI declarations and the config the bounds come from. This module RENDERS it:
it adds no facts, only an arrangement — the group structure a client draws
panels or schema objects from — and the schema that arrangement conforms to.

**No bus traffic, ever.** Building a manifest describes the station, it never
operates it. Every input is a declaration or a config value, which is what
lets this run against an instrument that is offline and lets a client ask for
the picture outside the tick loop. ``Station.station_info()``'s docstring and
``control_param_specs()``'s purity rule
(``virtual_instruments/base.py``) carry the standard;
``tests/test_conformance.py`` builds the whole thing against spied drivers
and fails on any driver call.

**The schema and its validator.** ``MANIFEST_SCHEMA`` is a JSON Schema
(draft 2020-12) document describing exactly the output of
``build_manifest()``. ``jsonschema`` is deliberately NOT a dependency of this
project — adding a runtime dependency to validate an internal, generated
document would be a poor trade — so ``validate_manifest()`` is a small
structural validator over the subset of JSON Schema the document actually
uses: ``$ref`` into ``$defs``, ``type`` (single or union), ``properties``,
``required``, ``additionalProperties`` (as a flag or as a schema, for open
maps), ``items`` and ``enum``. The schema stays a real, publishable JSON
Schema document, so a consumer that DOES have ``jsonschema`` can validate
with it directly; should ``jsonschema`` ever become a dependency for another
reason, this validator can be swapped for it with no change to the schema.

Run ``python -m i2as.core.capability_manifest <config_dir>`` to print the
manifest for a config directory. It builds a Station and nothing else — no
``QApplication``, no Orchestrator — so it works in a terminal, in CI, and in
a test.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from i2as.core.decorators import VALID_ACTION_CLASSES
from i2as.core.events import StationInfo

if TYPE_CHECKING:  # pragma: no cover — typing only, never imported at runtime
    from i2as.core.station import Station

logger = logging.getLogger(__name__)

#: Identifies this manifest's shape. Bumped only for a BREAKING change; a new
#: optional key is an additive change that leaves it alone, the same add-only
#: discipline the operational-status record follows.
MANIFEST_SCHEMA_ID = "i2as.capability_manifest/1"


def build_manifest(station: Station) -> dict[str, Any]:
    """Render a built Station's declaration as the capability manifest.

    The whole of ``station.station_info()``, rendered to JSON, with one thing
    added: each instrument's capabilities resolved into its declared UI
    groups. A group's ``monitored``/``controls`` name the members it declares,
    in declared order, and the instrument's ``ungrouped`` block names
    everything no group claims, after them. The full details stay in the
    instrument's own ``monitored``/``controls`` lists, ordered the same way —
    grouped first, ungrouped after — so a renderer can walk either the flat
    list or the group index and get the same order.

    Args:
        station: A built ``Station``. Only its ``station_info()`` is read, so
            this issues no instrument traffic and works just as well for a
            station whose instruments are all offline.

    Returns:
        A JSON-safe dict conforming to :data:`MANIFEST_SCHEMA`.
    """
    info: StationInfo = station.station_info()
    return {
        "schema": MANIFEST_SCHEMA_ID,
        "setup": info.setup,
        "tick_interval_s": info.tick_interval_s,
        "seq": info.seq,
        "ts": info.ts,
        "instruments": [_instrument_json(entry) for entry in info.instruments],
    }


def _instrument_json(instrument: Any) -> dict[str, Any]:
    """Render one ``InstrumentInfo``, resolving its groups.

    Args:
        instrument: One entry of ``StationInfo.instruments``.

    Returns:
        The instrument's JSON object, with ``monitored`` and ``controls``
        ordered grouped-first and the ``groups`` / ``ungrouped`` index over
        them.
    """
    payload = instrument.to_json()
    monitored_names = [entry.name for entry in instrument.monitored]
    control_names = [entry.name for entry in instrument.controls]

    groups: list[dict[str, Any]] = []
    claimed: list[str] = []
    for group in instrument.ui_groups:
        members = list(group.members)
        claimed.extend(members)
        groups.append(
            {
                "key": group.key,
                "title": group.title,
                "description": group.description,
                "monitored": [n for n in members if n in monitored_names],
                "controls": [n for n in members if n in control_names],
            }
        )

    ungrouped_monitored = [n for n in monitored_names if n not in claimed]
    ungrouped_controls = [n for n in control_names if n not in claimed]

    payload["groups"] = groups
    payload["ungrouped"] = {
        "monitored": ungrouped_monitored,
        "controls": ungrouped_controls,
    }
    payload["monitored"] = _ordered_by_group(
        payload["monitored"], claimed, ungrouped_monitored
    )
    payload["controls"] = _ordered_by_group(
        payload["controls"], claimed, ungrouped_controls
    )
    # ui_groups is the declaration; groups is this rendering of it, member by
    # member. Keeping both would be two sources for one fact.
    payload.pop("ui_groups", None)
    return payload


def _ordered_by_group(
    entries: Sequence[Mapping[str, Any]], claimed: Sequence[str], trailing: Sequence[str]
) -> list[dict[str, Any]]:
    """Order rendered capabilities grouped-first, in declared group order.

    Args:
        entries: The rendered ``MonitoredInfo``/``ControlInfo`` dicts, in
            declaration order.
        claimed: Every group member name, concatenated in declared group
            order — the order grouped capabilities take.
        trailing: The names no group claims, in declaration order.

    Returns:
        The same dicts, reordered. An entry named in ``claimed`` but absent
        from ``entries`` is skipped, which cannot happen for a VI the base
        class validated but keeps this a total function.
    """
    by_name = {str(entry["name"]): dict(entry) for entry in entries}
    ordered = [by_name[name] for name in claimed if name in by_name]
    ordered.extend(by_name[name] for name in trailing if name in by_name)
    return ordered


# ── The schema ────────────────────────────────────────────────────────────────

#: JSON Schema (draft 2020-12) for ``build_manifest()``'s output. A real,
#: publishable schema document: a consumer holding ``jsonschema`` can validate
#: against it directly, and ``validate_manifest()`` below checks the same
#: document with no dependency.
MANIFEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://i2as.invalid/schemas/capability_manifest.json",
    "title": "I2AS capability manifest",
    "description": (
        "Everything one I2AS station declares: each configured "
        "instrument's readings, actions, parameter specs, configured limits, "
        "capability groups and safety flags."
    ),
    "type": "object",
    "required": ["schema", "setup", "tick_interval_s", "seq", "ts", "instruments"],
    "additionalProperties": False,
    "properties": {
        "schema": {"type": "string", "enum": [MANIFEST_SCHEMA_ID]},
        "setup": {
            "type": "string",
            "description": "The setup's identity: its config directory's name.",
        },
        "tick_interval_s": {
            "type": "number",
            "description": "The monitor tick period, in seconds.",
        },
        "seq": {
            "type": "integer",
            "description": "Rebuild counter of the underlying StationInfo.",
        },
        "ts": {"type": "number", "description": "Unix time of the snapshot."},
        "instruments": {
            "type": "array",
            "items": {"$ref": "#/$defs/instrument"},
        },
    },
    "$defs": {
        "param": {
            "type": "object",
            "description": "One @control parameter, from its ParamSpec.",
            "required": [
                "name",
                "declared",
                "kind",
                "unit",
                "description",
                "default",
                "min",
                "max",
                "choices",
            ],
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "declared": {
                    "type": "boolean",
                    "description": (
                        "Whether a ParamSpec declares this parameter. False "
                        "means only the signature is known, so unit, "
                        "description, bounds and choices are absent rather "
                        "than declared empty."
                    ),
                },
                "kind": {
                    "type": "string",
                    "description": "Scalar type name: float, int, str or bool.",
                },
                "unit": {"type": "string"},
                "description": {"type": "string"},
                "default": {"description": "The declared default, a JSON scalar."},
                "min": {"type": ["number", "null"]},
                "max": {"type": ["number", "null"]},
                "choices": {
                    "type": ["object", "null"],
                    "description": "Label -> value for an enumerated parameter.",
                },
            },
        },
        "monitored": {
            "type": "object",
            "description": "One @monitored reading.",
            "required": ["name", "unit", "description", "group", "returns"],
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "unit": {
                    "type": "string",
                    "description": 'SI unit label, "" when dimensionless.',
                },
                "description": {"type": "string"},
                "group": {"type": "string"},
                "returns": {"type": "string"},
            },
        },
        "control": {
            "type": "object",
            "description": "One @control action.",
            "required": ["name", "scope", "action_class", "panel", "group", "params"],
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "scope": {"type": "string", "enum": ["measurement", "operation"]},
                "action_class": {
                    "type": "string",
                    "enum": list(VALID_ACTION_CLASSES),
                    "description": "How much authority this action needs, as "
                    "the instrument declares it.",
                },
                "panel": {"type": "boolean"},
                "group": {"type": "string"},
                "params": {"type": "array", "items": {"$ref": "#/$defs/param"}},
            },
        },
        "group": {
            "type": "object",
            "description": "One titled group of an instrument's capabilities.",
            "required": ["key", "title", "description", "monitored", "controls"],
            "additionalProperties": False,
            "properties": {
                "key": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "monitored": {"type": "array", "items": {"type": "string"}},
                "controls": {"type": "array", "items": {"type": "string"}},
            },
        },
        "limit": {
            "type": "object",
            "description": (
                "One control parameter's configured bounds. null means "
                "unbounded on that side, or not computed (an offline VI)."
            ),
            "required": ["limit", "min", "max"],
            "additionalProperties": False,
            "properties": {
                "limit": {"type": "string"},
                "min": {"type": ["number", "null"]},
                "max": {"type": ["number", "null"]},
            },
        },
        "instrument": {
            "type": "object",
            "required": [
                "name",
                "vi_class",
                "role",
                "kind",
                "availability",
                "monitored",
                "controls",
                "limits",
                "safety_flags",
                "groups",
                "ungrouped",
            ],
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "vi_class": {"type": "string"},
                "role": {
                    "type": "string",
                    "description": "Config registry role: system or "
                    "measurement.",
                },
                "kind": {
                    "type": "string",
                    "description": "VI class category: magnet, temperature, "
                    "measurement …",
                },
                "availability": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Availability tags standing at snapshot "
                    "time; empty means fully usable.",
                },
                "monitored": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/monitored"},
                },
                "controls": {"type": "array", "items": {"$ref": "#/$defs/control"}},
                "limits": {
                    "type": "object",
                    "description": "method -> parameter -> configured bounds.",
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": {"$ref": "#/$defs/limit"},
                    },
                },
                "safety_flags": {
                    "type": "object",
                    "description": "flag -> severity (advisory, hold, critical).",
                    "additionalProperties": {"type": "string"},
                },
                "groups": {"type": "array", "items": {"$ref": "#/$defs/group"}},
                "ungrouped": {
                    "type": "object",
                    "required": ["monitored", "controls"],
                    "additionalProperties": False,
                    "properties": {
                        "monitored": {"type": "array", "items": {"type": "string"}},
                        "controls": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    },
}


def validate_manifest(
    manifest: Mapping[str, Any], schema: Mapping[str, Any] | None = None
) -> list[str]:
    """Check a manifest against :data:`MANIFEST_SCHEMA`.

    Returns errors rather than raising, mirroring ``validate_config_dir()``:
    a caller wants every problem at once, not the first.

    Args:
        manifest: The document to check, typically ``build_manifest()``'s
            output.
        schema: The schema to check against. Defaults to
            :data:`MANIFEST_SCHEMA`; a caller passes one only to check a
            fragment against a ``$defs`` entry.

    Returns:
        Human-readable error strings, each naming the JSON path it concerns.
        An empty list means the manifest conforms.
    """
    root = schema if schema is not None else MANIFEST_SCHEMA
    errors: list[str] = []
    _check(manifest, root, MANIFEST_SCHEMA, "$", errors)
    return errors


# The JSON-Schema type names this validator understands, mapped to the Python
# check each one is. `bool` is excluded from the numeric types because it is an
# `int` subclass in Python and a flag is never a quantity.
_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, Mapping),
    "array": lambda v: isinstance(v, (list, tuple)),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _check(
    value: Any,
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    path: str,
    errors: list[str],
) -> None:
    """Validate one value against one schema node, appending any errors.

    Implements the JSON Schema subset ``MANIFEST_SCHEMA`` uses: ``$ref`` into
    ``$defs``, ``type`` (a name or a list of names), ``enum``,
    ``properties``, ``required``, ``additionalProperties`` (``False`` to
    close an object, or a schema every extra value must match), and
    ``items``. A keyword the schema does not use is not implemented, so
    adding one to the schema without teaching it here would silently not be
    checked — which is why the schema and this function are kept in one file.

    Args:
        value: The value under test.
        schema: The schema node it must satisfy.
        root: The whole schema document, for resolving ``$ref``.
        path: JSON path of ``value``, for the error messages.
        errors: The accumulating error list, appended to in place.
    """
    ref = schema.get("$ref")
    if isinstance(ref, str):
        resolved = _resolve_ref(ref, root)
        if resolved is None:
            errors.append(f"{path}: schema $ref {ref!r} does not resolve")
            return
        _check(value, resolved, root, path, errors)
        return

    declared = schema.get("type")
    if declared is not None:
        names = [declared] if isinstance(declared, str) else list(declared)
        if not any(_TYPE_CHECKS.get(name, lambda _v: False)(value) for name in names):
            errors.append(
                f"{path}: expected {'/'.join(names)}, got "
                f"{type(value).__name__}"
            )
            return

    allowed = schema.get("enum")
    if allowed is not None and value not in allowed:
        errors.append(f"{path}: {value!r} is not one of {allowed}")

    if isinstance(value, Mapping):
        _check_object(value, schema, root, path, errors)
    elif isinstance(value, (list, tuple)):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _check(item, item_schema, root, f"{path}[{index}]", errors)


def _check_object(
    value: Mapping[str, Any],
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    path: str,
    errors: list[str],
) -> None:
    """Validate an object's required keys, properties and extra keys.

    Args:
        value: The mapping under test.
        schema: Its schema node.
        root: The whole schema document, for resolving ``$ref``.
        path: JSON path of ``value``.
        errors: The accumulating error list, appended to in place.
    """
    properties = schema.get("properties") or {}
    for key in schema.get("required") or ():
        if key not in value:
            errors.append(f"{path}: missing required key {key!r}")

    for key, item in value.items():
        if key in properties:
            _check(item, properties[key], root, f"{path}.{key}", errors)
            continue
        extra = schema.get("additionalProperties")
        if extra is False:
            errors.append(f"{path}: unexpected key {key!r}")
        elif isinstance(extra, Mapping):
            _check(item, extra, root, f"{path}.{key}", errors)


def _resolve_ref(ref: str, root: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Resolve a local ``#/$defs/<name>`` reference.

    Args:
        ref: The reference string.
        root: The whole schema document.

    Returns:
        The referenced schema node, or ``None`` if it does not resolve.
    """
    prefix = "#/$defs/"
    if not ref.startswith(prefix):
        return None
    node = (root.get("$defs") or {}).get(ref[len(prefix) :])
    return node if isinstance(node, Mapping) else None


# ── Command-line entry point ──────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    """Print the capability manifest for one config directory.

    ``python -m i2as.core.capability_manifest <config_dir>``. Builds a
    Station and nothing else — no ``QApplication``, no Orchestrator — so the
    manifest can be read from a terminal, from CI, or from a test. The
    manifest itself is written to stdout (it IS this command's output, not a
    log line, which is why it is printed rather than logged, mirroring the
    troubleshoot CLI); a schema violation is reported on stderr and makes the
    command exit non-zero.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        The process exit status: 0 on success, 2 for a usage error, 1 for a
        build failure or a manifest that does not match its schema.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] in {"-h", "--help"}:
        sys.stderr.write(
            "usage: python -m i2as.core.capability_manifest <config_dir>\n"
        )
        return 2 if len(args) != 1 else 0

    from i2as.core.station import build_station

    try:
        station = build_station(args[0])
    except Exception as exc:  # noqa: BLE001 — a CLI reports, it does not traceback
        sys.stderr.write(f"cannot build a station from '{args[0]}': {exc}\n")
        return 1

    manifest = build_manifest(station)
    errors = validate_manifest(manifest)
    print(json.dumps(manifest, indent=2, sort_keys=False))
    if errors:
        for error in errors:
            sys.stderr.write(f"schema violation: {error}\n")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised through main()
    sys.exit(main())
