"""The **Tool surface**: what an autonomous client can call, rendered not written.

**The standard.** A tool is never hand-written for a command. Every command
tool is *rendered* from two declarations that already exist and are already
machine-checked:

* ``core.events.CommandName`` — the engine's command surface, whose values
  ARE the Orchestrator method names, so the tool list and the GUI's action
  buttons come from one enumeration and neither client can offer an action
  the other cannot see;
* the **Station info** declaration snapshot — every instrument's
  ``@control``s with their ``ParamSpec``s (kind, unit, description, default,
  choices) and the configured bounds of the **Control-validation
  standard**, which become the tool's JSON Schema.

The description of a command tool is the first paragraph of the Orchestrator
method's own docstring and each argument's description is that method's
Google ``Args:`` entry, so the text an agent reads is the text a person
reading the code reads. The description of a capability tool is the
capability's own row in ``action_classes.py`` — the sentence a physicist
reviewed. Nothing in this module describes an instrument or a command in
words of its own; a hand-maintained description drifts from the instrument in
the first week, which is the same argument the capability manifest is built
on.

**Three kinds of tool, one type.**

* **Command tools** — one per ``CommandName``, wrapping ``Gateway.submit()``.
* **Capability tools** — ``submit_vi_action`` is the one command whose
  arguments depend on what it targets, so it is rendered as one tool per
  ``(instrument, @control)`` the station declares, with that control's
  ``ParamSpec``s as the schema and the configured limit as the bound. An
  agent therefore reads "-9 to 9 T" off the tool, and is refused by the
  schema before a command is ever built.
* **Session tools** — hand-declared, because they are NOT commands: they read
  the experiment store, the run files, the operational log and the agent
  feed, they answer "may I run this, and how long will it take?" without
  dispatching anything, they draft and publish this experiment's notebook
  entries, and they read, write and run the **analysis recipes** a finished
  run is analysed with. Every one of them is ``ActionClass.READ`` except
  ``probe_run``, which really is a ``run_procedure`` with a ``ProbeSpec`` and
  is classified (and refused) as one; ``publish_eln_entry``, which puts a
  permanent record of this experiment into the outside world on the
  experiment's behalf; and ``write_analysis_recipe`` / ``run_analysis``, which
  put code on the measurement machine and start the process that executes it.
  All three are classified ``run_control``.

**Five tools reach the analysis stage**, and the trust boundary they sit on is
written down in this folder's README. A recipe is code, trusted like a
procedure: ``write_analysis_recipe`` compiles what it is given, stamps it with
a header naming the actor and the UTC time, and writes it into the
experiment's own ``analysis/recipes`` folder — it executes nothing. The
analysis worker does that, in a separate process that holds the run's data
file and reaches no instrument, and only when ``run_analysis`` (or a human's
button) starts it. Both are ``run_control`` and both declare ``recorded``, so
the **Agent feed** carries the call, the digest of the source written and the
run analysed, and the eLab tab shows the recipe before anybody runs it. The
other three — listing recipes, reading one, and reading a report — change
nothing and are ``read``.

**Two tools reach the ELN track**, and they divide exactly where the money
and the authority divide. ``draft_eln_entry`` renders one finished run's
facts into the draft prompt, asks one model, and returns the **draft entry**
AS DATA: it writes nothing, queues nothing, and is ``read`` because it
changes no state — but it spends model tokens, so it declares ``recorded``
and its cost line lands in the **Agent feed**. ``publish_eln_entry`` is the
only one that reaches the **Outbox**, and in an ATTENDED experiment it
refuses every agent by rule and parks the draft on the run record for a
human instead. Neither renders an entry itself and neither touches a
backend: the rendering is ``session/eln/``'s and the network is the
publisher's drain, off the tick.

**Arguments the engine translates.** ``Orchestrator.submit()`` documents four
commands whose JSON ``args`` are not simply the method's parameters — a
procedure travels as a class name plus its params, an envelope as a mapping,
the kill switch as its ``AgentGate`` value. Those are declared once in
``COMMAND_ARG_SCHEMAS`` below, each with the rationale for why it deviates,
and a command whose signature carries a type this module cannot render and
that has no entry there fails to render rather than being guessed at — the
same no-silent-default rule the classification tables follow.

**Validation before submission.** ``validate_tool_args()`` checks a call
against its tool's schema and answers with operator-facing messages that name
the bound and its unit. It reuses ``core.capability_manifest``'s structural
JSON Schema validator (``jsonschema`` is deliberately not a dependency of
this project) and adds the numeric-bound check on top, because a bound is
exactly what a tool schema exists to publish.

Conformance diffs the rendered list against ``CommandName`` and against every
shipped config's manifest in both directions, which is the third leg of the
three-way test: contract, engine, tool surface.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from i2as.analysis.report import REPORT_FILENAME, AnalysisReport
from i2as.core import data_reader
from i2as.core.capability_manifest import build_manifest, validate_manifest
from i2as.core.events import CommandName, StationInfo
from i2as.core.orchestrator import Orchestrator
from i2as.core.paths import log_directory
from i2as.session.eln.adapter import ElnError
from i2as.session.eln.drafting import (
    DraftRequest,
    draft_entry,
    manifest_from_run,
)
from i2as.session.eln.settings import AssistantSettings
from i2as.session.gateway.action_classes import (
    COMMAND_ACTION_CLASSES,
    ActionClass,
    classify_control,
)

logger = logging.getLogger(__name__)

#: Name of the per-experiment agent action feed: the append-only JSONL record
#: of what an agent asked for and what it was answered. Read through
#: ``ExperimentStore.agent_feed_path()`` where that exists, and from
#: ``<experiment folder>/agent_actions.jsonl`` otherwise — the layout the
#: store's own docstring documents.
AGENT_FEED_FILENAME = "agent_actions.jsonl"

#: Name of the per-tick operational-status log inside ``paths.log_directory()``.
OPERATIONAL_LOG_FILENAME = "status.jsonl"

#: Separator between an instrument and its capability in a capability tool's
#: name. A tool name must be a plain identifier for every tool-use API that
#: renders this list, so the two halves are joined rather than dotted; the
#: instrument and capability stay available as fields on the ``ToolSpec``, so
#: nothing ever has to parse the name back apart.
TOOL_NAME_SEPARATOR = "__"

#: The largest number of trailing records any log-tailing tool will return.
MAX_LOG_RECORDS = 500

#: The largest recipe source ``write_analysis_recipe`` will write, in bytes.
#: A recipe is one readable script a physicist reviews in the eLab tab before
#: anybody runs it; a file past this size is not that, and it would also be
#: quoted into the **Agent feed**'s digest line for as long as the experiment
#: folder lives.
MAX_RECIPE_BYTES = 200_000

#: Arguments too big to belong in the **Agent feed** verbatim, per tool. Each
#: named argument is replaced in the recorded call by its byte count and the
#: SHA-256 of its text, so the feed says exactly WHICH source was written —
#: and can be checked against the file on disk — without carrying a copy of
#: it in every line. The file itself is the record of the text.
FEED_DIGESTED_ARGS: dict[str, tuple[str, ...]] = {
    "write_analysis_recipe": ("source",),
}

#: The JSON Schema type each ``ParamSpec.type`` name and each scalar
#: annotation renders as.
JSON_TYPES: dict[str, str] = {
    "float": "number",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
}


class ToolError(Exception):
    """A tool call that cannot be answered — bad arguments, or a missing collaborator.

    Raised by the session-tool implementations and turned into a
    ``FAILED``-shaped result dict by ``Gateway.call_tool()``, which never
    raises at its caller: an agent receives an answer to every call it makes,
    exactly as the **Verdict standard** promises for every command.

    Attributes:
        detail: A JSON-safe structured explanation, so a client decides from
            the dict and never by parsing the message.
    """

    def __init__(self, message: str, detail: Mapping[str, Any] | None = None) -> None:
        """Build the error with its structured detail.

        Args:
            message: Operator-facing explanation.
            detail: The structured half; defaults to an empty dict.
        """
        super().__init__(message)
        self.detail: dict[str, Any] = dict(detail or {})


# ══════════════════════════════════════════════════════════════════════════
# The tool specification
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ToolSpec:
    """One callable a client is offered, and everything needed to route it.

    Frozen and JSON-safe: ``to_json()`` renders the whole declaration and
    ``to_schema()`` renders the ``{name, description, input_schema}`` shape a
    tool-use API expects, so a terminal client and a later tool-use adapter
    publish the same list with no rendering code of their own.

    Exactly one of ``command`` and ``session_function`` is set: a tool either
    wraps a ``CommandName`` (and is submitted, authorized and answered by a
    **Verdict** like any other command) or calls a session function (and is
    answered here). Nothing is both.

    Attributes:
        name: The tool's identifier, unique across the rendered surface.
        description: What it does, in the words of the declaration it was
            rendered from.
        input_schema: JSON Schema (draft 2020-12 subset) for the arguments —
            always an object schema, closed with ``additionalProperties:
            false`` so an unexpected key is refused rather than dropped.
        action_class: The **Action class** the permission matrix judges this
            tool by.
        command: The ``CommandName`` this tool submits, or ``None`` for a
            session tool.
        fixed_args: Arguments merged into every call of this tool before the
            command is built — how a capability tool carries its own
            ``vi_name`` and ``method_name`` without asking the caller for
            them.
        session_function: Key of the session function this tool calls, or
            ``""`` for a command tool.
        instrument: The VI this capability tool targets, or ``""``.
        capability: The ``@control`` this capability tool calls, or ``""``.
        recorded: Whether a call to this tool is written to the **Agent
            feed**. A command tool never needs it — the feed already records
            every command and its verdict — so this is how a SESSION tool,
            answered inside the client where no verdict exists, still leaves
            a trail. Declared per tool rather than derived from the action
            class, because the two questions are different: a ``read`` tool
            an agent polls every tick would drown the trail in observations,
            while a ``read`` tool that spends model tokens is exactly what
            the trail is for.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    action_class: ActionClass
    command: CommandName | None = None
    fixed_args: dict[str, Any] = field(default_factory=dict)
    session_function: str = ""
    instrument: str = ""
    capability: str = ""
    recorded: bool = False

    def __post_init__(self) -> None:
        """Validate the declaration and freeze copies of its mappings.

        Raises:
            TypeError: If ``input_schema`` or ``fixed_args`` is not a mapping.
            ValueError: If ``name`` or ``description`` is empty, if the schema
                is not an object schema, or if the tool is neither exactly a
                command tool nor exactly a session tool.
        """
        for label, value in (("name", self.name), ("description", self.description)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"ToolSpec.{label} must be a non-empty str")
        if not isinstance(self.input_schema, Mapping):
            raise TypeError("ToolSpec.input_schema must be a mapping")
        if self.input_schema.get("type") != "object":
            raise ValueError(
                f"ToolSpec({self.name!r}).input_schema must be an object schema"
            )
        if not isinstance(self.fixed_args, Mapping):
            raise TypeError("ToolSpec.fixed_args must be a mapping")
        object.__setattr__(self, "input_schema", json.loads(json.dumps(self.input_schema)))
        object.__setattr__(self, "fixed_args", dict(self.fixed_args))
        object.__setattr__(self, "action_class", ActionClass(self.action_class))
        if self.command is not None:
            object.__setattr__(self, "command", CommandName(self.command))
        if (self.command is None) == (not self.session_function):
            raise ValueError(
                f"ToolSpec({self.name!r}) must wrap exactly one of a CommandName "
                f"or a session function, not both and not neither"
            )

    @property
    def is_command(self) -> bool:
        """Whether this tool is submitted to the engine rather than answered here."""
        return self.command is not None

    def to_schema(self) -> dict[str, Any]:
        """Render the shape a tool-use API expects.

        Returns:
            ``{"name", "description", "input_schema"}`` — nothing else, so the
            same dict serves every client that publishes a tool list.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": json.loads(json.dumps(self.input_schema)),
        }

    def to_json(self) -> dict[str, Any]:
        """Render the whole declaration, routing included.

        Returns:
            A JSON-safe dict of every field, enums as their string values.
        """
        payload = self.to_schema()
        payload.update(
            {
                "action_class": self.action_class.value,
                "command": None if self.command is None else self.command.value,
                "fixed_args": dict(self.fixed_args),
                "session_function": self.session_function,
                "instrument": self.instrument,
                "capability": self.capability,
                "recorded": self.recorded,
            }
        )
        return payload


# ══════════════════════════════════════════════════════════════════════════
# Rendering command tools from the Orchestrator's declarations
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class WireArguments:
    """The JSON arguments of a command whose wire shape is not its signature.

    ``Orchestrator.submit()`` translates a handful of commands' JSON ``args``
    into the method's Python arguments — a procedure arrives as a class name
    plus its parameter values, an envelope as a mapping, the kill switch as
    its ``AgentGate`` value. This is that translation, declared from the tool
    surface's side, once.

    Attributes:
        properties: The JSON Schema ``properties`` block for the command.
        required: Argument names a caller must supply.
        rationale: Why this command deviates from its signature — what a
            reviewer reads to check the entry is still true.
    """

    properties: dict[str, Any]
    required: tuple[str, ...]
    rationale: str


#: A ``ProbeSpec``'s JSON form, reused by every tool that can reduce a run to
#: a **probe run**. Bounds mirror ``core.plan.ProbeSpec``'s own validation.
_PROBE_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Reduce the run to a probe run: the same procedure and the same "
        "instruments, subsampled until it costs minutes instead of hours. "
        "Omit for a full science run."
    ),
    "additionalProperties": False,
    "properties": {
        "n_points": {
            "type": "integer",
            "minimum": 1,
            "default": 3,
            "description": "Sweep points kept, always including the first and last.",
        },
        "averaging": {
            "type": "integer",
            "minimum": 1,
            "default": 1,
            "description": "Cap on every declared repeat count.",
        },
        "max_wait_s": {
            "type": "number",
            "minimum": 0,
            "default": 5.0,
            "unit": "s",
            "description": "Cap, in seconds, on every declared wait.",
        },
    },
}


def _run_properties(kind: str = "procedure", *, probe: bool = True) -> dict[str, Any]:
    """Build the shared ``properties`` block of a run command.

    Args:
        kind: The argument name carrying the class name.
        probe: Whether the command accepts a ``probe_spec``.

    Returns:
        The JSON Schema properties for that command's arguments.
    """
    properties: dict[str, Any] = {
        kind: {
            "type": "string",
            "description": (
                f"Class name of the {kind} to run, resolved through the "
                f"engine's run catalog."
            ),
        },
        "params": {
            "type": "object",
            "description": f"The {kind}'s own parameter values, as declared.",
        },
    }
    if kind == "procedure":
        properties.update(
            {
                "sample_info": {
                    "type": "object",
                    "description": "Sample metadata recorded with the run.",
                },
                "data_directory": {
                    "type": "string",
                    "description": "Directory the run writes its HDF5 file into.",
                },
                "file_prefix": {
                    "type": "string",
                    "description": "Optional filename prefix for the data file.",
                },
                "experiment_info": {
                    "type": ["object", "null"],
                    "description": "Experiment context stamped into the run manifest.",
                },
            }
        )
    if probe:
        properties["probe_spec"] = dict(_PROBE_SPEC_SCHEMA)
    return properties


#: The two arguments the four owner-scoped commands carry to take a run over
#: (the **run-ownership standard**, GLOSSARY.md's *Run owner* and *Takeover*).
#: They belong in this table rather than in the four signatures because the
#: ``command`` decorator absorbs them before the method is called, exactly as
#: it absorbs ``actor`` — so the wire shape and the Python signature differ,
#: which is what this table is for. Publishing them here is what lets the
#: CLI and the MCP adapter offer the override with no client
#: code of their own: they read it off the rendered schema.
_OWNERSHIP_PROPERTIES: dict[str, Any] = {
    "override_owner": {
        "type": "boolean",
        "default": False,
        "description": (
            "Take this run over even though another actor owns it. Leave it "
            "out on your own run. Doing this to somebody else's run is "
            "recorded as a takeover, on the run and in the agent feed; "
            "prefer asking the owner or the physicist first."
        ),
    },
    "reason": {
        "type": "string",
        "description": (
            "Why you are taking the run over. Required with override_owner, "
            "and refused if empty: it is what the takeover record says."
        ),
    },
}

#: The rationale every owner-scoped entry above carries, said once.
_OWNERSHIP_RATIONALE = (
    "The command decorator absorbs override_owner and reason (the "
    "run-ownership standard's refuse-then-override) before the method sees "
    "them, so the wire arguments are the method's own plus those two."
)

#: Commands whose JSON ``args`` are translated by ``Orchestrator.submit()``
#: rather than being the method's own parameters, plus the one command whose
#: parameter carries an enum. Every entry says why it deviates; a command
#: whose signature this module cannot render and that is absent here fails to
#: render, rather than being guessed at.
COMMAND_ARG_SCHEMAS: dict[CommandName, WireArguments] = {
    CommandName.RUN_PROCEDURE: WireArguments(
        properties=_run_properties("procedure", probe=True),
        required=("procedure",),
        rationale=(
            "The method takes a built procedure object; a client sends the "
            "class name plus what the engine builds it from."
        ),
    ),
    CommandName.QUEUE_PROCEDURE: WireArguments(
        properties=_run_properties("procedure", probe=True),
        required=("procedure",),
        rationale="As run_procedure — the same payload, queued instead of started.",
    ),
    CommandName.SET_EXPERIMENT_ENVELOPE: WireArguments(
        properties={
            "envelope": {
                "type": ["object", "null"],
                "description": (
                    "The session envelope as "
                    "{vi_name: {min_value, max_value, state_key}}, or null to "
                    "clear it."
                ),
            }
        },
        required=("envelope",),
        rationale=(
            "The method takes an ExperimentEnvelope; a client sends the "
            "mapping ExperimentEnvelope.from_dict() reads."
        ),
    ),
    CommandName.ABORT_PROCEDURE: WireArguments(
        properties=dict(_OWNERSHIP_PROPERTIES),
        required=(),
        rationale=_OWNERSHIP_RATIONALE,
    ),
    CommandName.SET_AGENT_GATE: WireArguments(
        properties={
            "state": {
                "type": "string",
                "enum": ["active", "read_only", "revoked"],
                "description": (
                    "The kill switch's setting: active (roles decide), "
                    "read_only (agents may only read) or revoked (nothing)."
                ),
            }
        },
        required=("state",),
        rationale=(
            "The parameter is an AgentGate enum, which travels as its string "
            "value; the enum's members are the schema's choices."
        ),
    ),
}

#: The one command rendered per capability instead of once: its arguments are
#: the target control's ``ParamSpec``s, which is the whole point of rendering
#: bounds into a tool schema.
COMMANDS_RENDERED_PER_CAPABILITY: frozenset[CommandName] = frozenset(
    {CommandName.SUBMIT_VI_ACTION}
)

#: Signature parameters that never appear in a tool schema: the receiver, and
#: the accountability keyword the gateway stamps itself.
_NON_ARGUMENT_PARAMETERS = frozenset({"self", "actor"})


def _plain(text: str) -> str:
    r"""Strip reStructuredText literal markup from a rendered description.

    The descriptions come from docstrings written for Sphinx; the tool
    surface's reader is an agent or a terminal, for which ``\`\`literal\`\```
    markers are noise rather than emphasis.

    Args:
        text: The docstring text.

    Returns:
        The same text with double-backtick markers removed.
    """
    return text.replace("``", "")


def _docstring_summary(doc: str) -> str:
    """Return the first paragraph of a docstring as one line.

    Args:
        doc: A dedented docstring, or ``""``.

    Returns:
        The first paragraph with its line breaks collapsed, or ``""``.
    """
    paragraph = doc.strip().split("\n\n", 1)[0]
    return _plain(" ".join(part.strip() for part in paragraph.splitlines() if part.strip()))


def _docstring_args(doc: str) -> dict[str, str]:
    """Return each ``Args:`` entry of a Google-style docstring.

    The per-argument descriptions of a command tool are the method's own, so
    the text an agent reads is the text a reader of the code reads.

    Args:
        doc: A dedented docstring, or ``""``.

    Returns:
        ``{argument name: description}``, continuation lines joined. Empty
        when the docstring declares no ``Args:`` section.
    """
    lines = doc.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Args:")
    except StopIteration:
        return {}

    entries: dict[str, list[str]] = {}
    current = ""
    entry_indent = 0
    for line in lines[start + 1 :]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            break
        match = re.match(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?:\s*(.*)$", line.strip())
        if match and (current == "" or indent <= entry_indent):
            current = match.group(1).lstrip("*")
            entry_indent = indent
            entries[current] = [match.group(2).strip()]
        elif current:
            entries[current].append(line.strip())
    return {
        name: _plain(" ".join(part for part in parts if part))
        for name, parts in entries.items()
    }


def _scalar_schema(annotation: Any, description: str) -> dict[str, Any] | None:
    """Render one signature parameter as a scalar JSON Schema, if it is one.

    Args:
        annotation: The parameter's annotation. Postponed evaluation makes it
            a string, which is exactly what is matched — this module never
            evaluates an annotation.
        description: The docstring's description of the parameter.

    Returns:
        The schema, or ``None`` when the annotation is not a plain scalar and
        the command therefore needs a ``COMMAND_ARG_SCHEMAS`` entry.
    """
    name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    json_type = JSON_TYPES.get(str(name).strip())
    if json_type is None:
        return None
    schema: dict[str, Any] = {"type": json_type}
    if description:
        schema["description"] = description
    return schema


def command_tool_spec(name: CommandName) -> ToolSpec:
    """Render one command's tool from the Orchestrator's own declarations.

    Args:
        name: The command to render.

    Returns:
        Its ``ToolSpec``: the description is the method's docstring summary,
        the schema is either the method's signature or the wire-argument
        entry that says why the two differ.

    Raises:
        ValueError: If the command is rendered per capability instead, if it
            has no **Action class**, or if its signature carries a type this
            module cannot render and it has no ``COMMAND_ARG_SCHEMAS`` entry.
    """
    if name in COMMANDS_RENDERED_PER_CAPABILITY:
        raise ValueError(
            f"{name.value!r} is rendered as one tool per capability; call "
            f"render_capability_tools() instead"
        )
    classified = COMMAND_ACTION_CLASSES.get(name)
    if classified is None:
        raise ValueError(
            f"{name.value!r} has no row in the gateway's action-class table, "
            f"so no role can be granted it and no tool may be offered for it"
        )

    method = getattr(Orchestrator, name.value)
    doc = inspect.getdoc(method) or ""
    described = _docstring_args(doc)

    wire = COMMAND_ARG_SCHEMAS.get(name)
    if wire is not None:
        properties = dict(wire.properties)
        required = list(wire.required)
    else:
        properties = {}
        required = []
        for parameter in inspect.signature(method).parameters.values():
            if parameter.name in _NON_ARGUMENT_PARAMETERS:
                continue
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            schema = _scalar_schema(
                parameter.annotation, described.get(parameter.name, "")
            )
            if schema is None:
                raise ValueError(
                    f"Orchestrator.{name.value}()'s {parameter.name!r} is "
                    f"annotated {parameter.annotation!r}, which is not a scalar "
                    f"this module renders; declare the command's wire "
                    f"arguments in COMMAND_ARG_SCHEMAS with its rationale"
                )
            properties[parameter.name] = schema
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter.name)

    return ToolSpec(
        name=name.value,
        description=_docstring_summary(doc) or name.value.replace("_", " "),
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        action_class=classified.action_class,
        command=name,
    )


def render_command_tools() -> tuple[ToolSpec, ...]:
    """Render one tool per ``CommandName``, in declaration order.

    Returns:
        Every command tool but the ones rendered per capability, which
        ``render_capability_tools()`` produces from the station instead.

    Raises:
        ValueError: If a command can be rendered neither from its signature
            nor from a wire-argument entry.
    """
    return tuple(
        command_tool_spec(member)
        for member in CommandName
        if member not in COMMANDS_RENDERED_PER_CAPABILITY
    )


# ══════════════════════════════════════════════════════════════════════════
# Rendering capability tools from the station declaration
# ══════════════════════════════════════════════════════════════════════════


def capability_tool_name(vi_name: str, method_name: str) -> str:
    """Return the tool name one instrument capability is offered under.

    Args:
        vi_name: The configured instrument name.
        method_name: The ``@control`` method name.

    Returns:
        ``"<vi_name>__<method_name>"``.
    """
    return f"{vi_name}{TOOL_NAME_SEPARATOR}{method_name}"


def _within(value: Any, lower: Any, upper: Any) -> bool:
    """Whether a declared default falls inside the bounds the tool publishes.

    Args:
        value: The declared default.
        lower: The published minimum, or ``None``.
        upper: The published maximum, or ``None``.

    Returns:
        ``True`` when the value is not a number, or is a number within both
        bounds. A non-numeric default has no bound to violate.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    if lower is not None and value < lower:
        return False
    return not (upper is not None and value > upper)


def _param_schema(
    spec: Mapping[str, Any], limits: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Render one ``ParamSpec`` (plus its configured bound) as a JSON Schema.

    The bound is the config's, not the declaration's, wherever the
    **Control-validation standard** declared one: the config is the single
    source of truth for a limit, so it is the number the tool publishes and
    the number a schema refusal names.

    Args:
        spec: One entry of ``ControlInfo.params`` — name, kind, unit,
            description, default, min, max, choices.
        limits: The configured ``{"limit", "min", "max"}`` for this parameter,
            or ``None`` when the control declares no limit on it.

    Returns:
        The parameter's schema.

    Raises:
        ValueError: If the declared ``kind`` is not a scalar type.
    """
    kind = str(spec.get("kind", ""))
    json_type = JSON_TYPES.get(kind)
    if json_type is None:
        raise ValueError(
            f"parameter {spec.get('name')!r} declares kind {kind!r}, which is "
            f"not one of {sorted(JSON_TYPES)}"
        )

    unit = str(spec.get("unit", ""))
    description = str(spec.get("description", ""))
    schema: dict[str, Any] = {"type": json_type}
    if unit:
        schema["unit"] = unit
    lower = spec.get("min")
    upper = spec.get("max")
    limit_name = ""
    if limits:
        limit_name = str(limits.get("limit", ""))
        if limits.get("min") is not None:
            lower = limits["min"]
        if limits.get("max") is not None:
            upper = limits["max"]
    if lower is not None:
        schema["minimum"] = lower
    if upper is not None:
        schema["maximum"] = upper
    choices = spec.get("choices")
    if isinstance(choices, Mapping) and choices:
        schema["enum"] = list(choices.values())
        schema["choice_labels"] = {str(k): v for k, v in choices.items()}
    default = spec.get("default")
    if "default" in spec and _within(default, lower, upper):
        schema["default"] = default
    elif "default" in spec:
        # A declaration's default can predate the configured limit that now
        # bounds it (a controller declaring 0 K on a cryostat configured from
        # 1.4 K up). Publishing it anyway would be a schema that refuses its
        # own default, so the bound wins and the default is simply not
        # offered: the config is the single source of truth for a limit.
        logger.debug(
            "dropping the out-of-bounds default %r declared for %r",
            default,
            spec.get("name"),
        )

    text = description
    if unit:
        text = f"{text} ({unit})" if text else unit
    if limit_name and (lower is not None or upper is not None):
        text = (
            f"{text}. Configured limit {limit_name}: "
            f"{lower if lower is not None else '-inf'} to "
            f"{upper if upper is not None else '+inf'}"
            f"{' ' + unit if unit else ''}."
        ).lstrip(". ")
    if text:
        schema["description"] = text
    return schema


def render_capability_tools(station_info: StationInfo) -> tuple[ToolSpec, ...]:
    """Render one tool per ``(instrument, @control)`` the station declares.

    Every configured instrument appears, live or offline alike, because an
    unreachable instrument still declares the same capabilities — what says
    whether they can be used right now is the availability on the snapshot,
    not the tool list.

    Args:
        station_info: The station's declaration snapshot.

    Returns:
        The capability tools, in the snapshot's own instrument and
        declaration order.

    Raises:
        ValueError: If a declared parameter carries a non-scalar kind.
        UnclassifiedActionError: If a declared capability has no row in the
            gateway's classification table — an action with no class is
            refused by name rather than offered without one.
    """
    tools: list[ToolSpec] = []
    for instrument in station_info.instruments:
        groups = {group.key: group for group in instrument.ui_groups}
        for control in instrument.controls:
            classified = classify_control(station_info, instrument.name, control.name)
            limits = instrument.limits.get(control.name) or {}
            properties = {
                str(spec["name"]): _param_schema(spec, limits.get(str(spec["name"])))
                for spec in control.params
            }
            group = groups.get(control.group)
            description = (
                f"{control.name} on {instrument.name} "
                f"({instrument.kind} instrument"
                + (f", {group.title}" if group is not None else "")
                + f"). {classified.rationale}"
            )
            tools.append(
                ToolSpec(
                    name=capability_tool_name(instrument.name, control.name),
                    description=description,
                    input_schema={
                        "type": "object",
                        "properties": properties,
                        # Every declared parameter is required: a control
                        # invoked with a parameter omitted would fall back to
                        # the method's default, and "ramp to 0 T because the
                        # argument was left out" is not an outcome an agent
                        # should be able to reach by accident.
                        "required": list(properties),
                        "additionalProperties": False,
                    },
                    action_class=classified.action_class,
                    command=CommandName.SUBMIT_VI_ACTION,
                    fixed_args={
                        "vi_name": instrument.name,
                        "method_name": control.name,
                    },
                    instrument=instrument.name,
                    capability=control.name,
                )
            )
    return tuple(tools)


# ══════════════════════════════════════════════════════════════════════════
# The session tools — hand-declared, because they are not commands
# ══════════════════════════════════════════════════════════════════════════

_RUN_SELECTOR: dict[str, Any] = {
    "run_id": {
        "type": "string",
        "description": "The run's id, as listed by list_runs.",
    },
    "experiment_id": {
        "type": "string",
        "description": "Experiment holding the run; defaults to the open one.",
    },
}

_LAST_RECORDS: dict[str, Any] = {
    "last": {
        "type": "integer",
        "minimum": 1,
        "maximum": MAX_LOG_RECORDS,
        "default": 20,
        "description": "How many trailing records to return.",
    }
}


def _read_tool(
    name: str,
    description: str,
    properties: Mapping[str, Any] | None = None,
    required: Sequence[str] = (),
) -> ToolSpec:
    """Declare one read-class session tool.

    Args:
        name: The tool's name, which is also its session-function key.
        description: What it answers.
        properties: Its argument schema's properties.
        required: Arguments a caller must supply.

    Returns:
        The ``ToolSpec``.
    """
    return ToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": dict(properties or {}),
            "required": list(required),
            "additionalProperties": False,
        },
        action_class=ActionClass.READ,
        session_function=name,
    )


#: The experiment an analysis tool works in; defaults to the open one, like
#: every other session tool that names an experiment.
_ANALYSIS_EXPERIMENT: dict[str, Any] = {
    "experiment_id": {
        "type": "string",
        "description": "Experiment to work in; defaults to the open one.",
    }
}

#: The session tools, in the order a client meets them: the live picture, the
#: stored runs, the "may I run this?" question, the two audit trails, the two
#: that reach the notebook, a probe run, and the five that read, write and run
#: the **Analysis recipe**s an experiment is analysed with. Every one is
#: ``read`` except ``probe_run``, which dispatches a real (if cheap) run and is
#: classified as the run control it is; ``publish_eln_entry``, which puts a
#: permanent record into the outside world; and ``write_analysis_recipe`` /
#: ``run_analysis``, which put code on the measurement machine and start the
#: process that executes it.
SESSION_TOOLS: tuple[ToolSpec, ...] = (
    _read_tool(
        "read_status",
        "The engine's latest status snapshot: state, active run, per-instrument "
        "status, ramps, faults, holds, attendance and the kill switch.",
    ),
    _read_tool(
        "read_station_info",
        "The station's declaration snapshot: every configured instrument, what "
        "it reads, what it can be asked to do, and within which bounds.",
    ),
    _read_tool(
        "read_manifest",
        "The capability manifest — the same declaration as read_station_info, "
        "with each instrument's capabilities resolved into its declared groups.",
    ),
    _read_tool(
        "list_runs",
        "Every run recorded in an experiment: id, procedure, kind, status, "
        "times and parameters.",
        {
            "experiment_id": {
                "type": "string",
                "description": "Experiment to list; defaults to the open one.",
            }
        },
    ),
    _read_tool(
        "read_run_columns",
        "The columns of one recorded run's data file, with their role, unit, "
        "dtype and shape.",
        _RUN_SELECTOR,
        ("run_id",),
    ),
    _read_tool(
        "read_run_slice",
        "A slice of one column of one recorded run, along its sweep axis.",
        {
            **_RUN_SELECTOR,
            "column": {"type": "string", "description": "The column to read."},
            "start": {"type": "integer", "description": "First sweep point."},
            "stop": {"type": "integer", "description": "Stop before this point."},
            "step": {"type": "integer", "minimum": 1, "description": "Stride."},
        },
        ("run_id", "column"),
    ),
    _read_tool(
        "read_run_stats",
        "The NaN-aware summary of one numeric column of one recorded run: "
        "count, min, max, mean and standard deviation over the written prefix.",
        {
            **_RUN_SELECTOR,
            "column": {"type": "string", "description": "The column to summarise."},
        },
        ("run_id", "column"),
    ),
    _read_tool(
        "read_run_metadata",
        "One recorded run's manifest as stored in its data file: procedure, "
        "parameters, sample info, run kind and timings.",
        _RUN_SELECTOR,
        ("run_id",),
    ),
    _read_tool(
        "validate_run",
        "Decide whether a proposed run may be queued, and how long it would "
        "take — the declared bounds, the headless build, the setup limits and "
        "the session envelope, plus the duration estimate and its assumptions. "
        "Dispatches nothing and opens no file.",
        {
            "procedure": {
                "type": "string",
                "description": "Class name of the procedure.",
            },
            "params": {"type": "object", "description": "The values it would run with."},
            "sample_info": {"type": "object", "description": "Sample metadata."},
            "data_directory": {
                "type": "string",
                "description": "Directory it would write into; never created here.",
            },
            "file_prefix": {"type": "string", "description": "Filename prefix."},
            "probe_spec": dict(_PROBE_SPEC_SCHEMA),
        },
        ("procedure",),
    ),
    _read_tool(
        "read_experiment",
        "One experiment record: title, sample, envelope, attendance, findings "
        "and its runs.",
        {
            "experiment_id": {
                "type": "string",
                "description": "Experiment to read; defaults to the open one.",
            }
        },
    ),
    _read_tool(
        "read_operational_log",
        "The tail of the per-tick operational-status log the engine writes: "
        "state, per-instrument health, ramp progress and stall alerts.",
        _LAST_RECORDS,
    ),
    _read_tool(
        "read_agent_feed",
        "The tail of this experiment's agent action feed: what an agent asked "
        "for and what it was answered.",
        {
            **_LAST_RECORDS,
            "experiment_id": {
                "type": "string",
                "description": "Experiment whose feed to read; defaults to the open one.",
            },
        },
    ),
    ToolSpec(
        name="draft_eln_entry",
        description=(
            "Draft one finished run into an electronic-lab-notebook entry. "
            "The run's own facts — procedure, parameters, per-column "
            "statistics, the station it ran on and the engine's state at run "
            "end — are rendered into a fixed prompt, one model is asked for "
            "prose, and the entry comes back as data: a title, a body with "
            "that prose above the same fact tables, tags, and what the draft "
            "cost. Writes nothing and publishes nothing; publish_eln_entry "
            "does that, and in an attended experiment a human approves it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                **_RUN_SELECTOR,
                "note": {
                    "type": "string",
                    "description": (
                        "A note to the drafter, carried into the prompt "
                        "verbatim — what to look at, or what is already known."
                    ),
                },
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
        action_class=ActionClass.READ,
        session_function="draft_eln_entry",
        recorded=True,
    ),
    ToolSpec(
        name="publish_eln_entry",
        description=(
            "Queue an approved draft entry for this experiment's electronic "
            "lab notebook. The run must belong to the OPEN experiment. While "
            "the experiment is attended this is refused for every agent and "
            "the draft is parked on the run record instead, for the human to "
            "approve; while it is unattended the entry is queued to the "
            "outbox and uploaded by the application's own drain."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "The run to publish, in the open experiment.",
                },
                "draft": {
                    "type": "object",
                    "description": "The draft entry, as draft_eln_entry returned it.",
                },
            },
            "required": ["run_id", "draft"],
            "additionalProperties": False,
        },
        action_class=ActionClass.RUN_CONTROL,
        session_function="publish_eln_entry",
        recorded=True,
    ),
    ToolSpec(
        name="probe_run",
        description=(
            "Run a procedure as a probe run: the same procedure driving the "
            "same instruments through the same code path, subsampled until it "
            "costs minutes instead of hours. Never science data — the file it "
            "writes declares run_kind 'probe'."
        ),
        input_schema={
            "type": "object",
            "properties": _run_properties("procedure", probe=True),
            "required": ["procedure", "probe_spec"],
            "additionalProperties": False,
        },
        action_class=COMMAND_ACTION_CLASSES[CommandName.RUN_PROCEDURE].action_class,
        command=CommandName.RUN_PROCEDURE,
    ),
    _read_tool(
        "list_analysis_recipes",
        "Every analysis recipe this experiment can be analysed with: the ones "
        "shipped with I2AS and the ones written into the experiment's own "
        "analysis/recipes folder, each with its name, description, the "
        "procedures it serves, its source path and the digest of its source. "
        "Also answers, under 'selected', which recipe would run for each "
        "procedure this experiment has already recorded a run of — the "
        "notebook settings' per-procedure preference first, then the recipe "
        "declaring that procedure, then a recipe that serves any procedure. "
        "Refused when no experiment is open and none was named.",
        _ANALYSIS_EXPERIMENT,
    ),
    _read_tool(
        "read_analysis_recipe",
        "The whole source of one analysis recipe, by name — an experiment "
        "recipe from this experiment's analysis/recipes folder, a shipped one "
        "from the package. Returns the name, the file it lives in, its source "
        "text, whether it is a 'package' or an 'experiment' recipe, and the "
        "digest of the file. A name no recipe answers to is refused "
        "(rule 'unknown_recipe'); list_analysis_recipes says which names there "
        "are.",
        {
            "name": {
                "type": "string",
                "description": "The recipe's name, as list_analysis_recipes gives it.",
            },
            **_ANALYSIS_EXPERIMENT,
        },
        ("name",),
    ),
    ToolSpec(
        name="write_analysis_recipe",
        description=(
            "Write one analysis recipe into this experiment's own "
            "analysis/recipes folder, as <name>.py. The source is checked to "
            "COMPILE and is stamped with a header naming who wrote it and "
            "when; it is not executed here. The analysis worker executes it "
            "later — in its own process, with no access to any instrument — "
            "when run_analysis is called or a human presses Run, and it is "
            "visible in the eLab tab, and recorded in this experiment's agent "
            "feed, before that happens. Returns the name, the path written, "
            "the file's size in bytes and its digest. Refused for a name that "
            "is not a plain Python identifier (rule 'invalid_name'), a source "
            "that does not compile ('syntax_error', with the line), a source "
            "over 200 kB ('too_large'), and an existing file unless overwrite "
            "is true ('exists')."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "The recipe's name: a plain Python identifier that "
                        "does not start with '_'. It names the file and it is "
                        "what run_analysis's 'recipe' takes."
                    ),
                },
                "source": {
                    "type": "string",
                    "description": (
                        "The recipe's whole source — a module defining an "
                        "AnalysisRecipe subclass. Capped at 200 kB."
                    ),
                },
                "overwrite": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Replace an existing recipe of this name. Without it "
                        "an existing file is refused rather than overwritten."
                    ),
                },
                **_ANALYSIS_EXPERIMENT,
            },
            "required": ["name", "source"],
            "additionalProperties": False,
        },
        action_class=ActionClass.RUN_CONTROL,
        session_function="write_analysis_recipe",
        recorded=True,
    ),
    ToolSpec(
        name="run_analysis",
        description=(
            "Analyse one recorded run: start the analysis worker on it, in a "
            "separate process that reads the run's data file and reaches no "
            "instrument. This returns as soon as the worker has been STARTED, "
            "not when it has finished — the result arrives later, so read it "
            "with read_analysis_report, which answers {'status': 'running'} "
            "until it is there. Refused while that run is already being "
            "analysed (rule 'already_running'), when nothing could be started "
            "('not_started': no experiment open, an unknown run, or a run with "
            "no data file), and when this connection was wired without an "
            "analysis runner."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "The recorded run to analyse.",
                },
                "recipe": {
                    "type": "string",
                    "description": (
                        "Recipe name to use; omit to let the runner choose the "
                        "one list_analysis_recipes reports under 'selected'."
                    ),
                },
                "options": {
                    "type": "object",
                    "description": (
                        "Options passed to the recipe verbatim, as scalars."
                    ),
                    "additionalProperties": {
                        "type": ["string", "number", "integer", "boolean", "null"]
                    },
                },
                **_ANALYSIS_EXPERIMENT,
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
        action_class=ActionClass.RUN_CONTROL,
        session_function="run_analysis",
        recorded=True,
    ),
    _read_tool(
        "read_analysis_report",
        "The analysis report of one recorded run: the recipe that ran, its "
        "summary paragraphs, derived values, figures and small tables, its "
        "warnings and — when the recipe failed — its error, plus the path of "
        "the report file. Answers {'status': 'running'} while the worker is "
        "still on that run and {'status': 'none'} when the run has not been "
        "analysed yet.",
        {
            "run_id": {
                "type": "string",
                "description": "The recorded run whose report to read.",
            },
            **_ANALYSIS_EXPERIMENT,
        },
        ("run_id",),
    ),
)


def render_tools(station_info: StationInfo) -> tuple[ToolSpec, ...]:
    """Render the whole tool surface for one station.

    Args:
        station_info: The station's declaration snapshot — where the
            capability tools, their parameters and their bounds come from.

    Returns:
        Every command tool, then every capability tool, then the session
        tools, each name unique across the three.

    Raises:
        ValueError: If a command or a declared parameter cannot be rendered,
            or if two tools would claim the same name.
        UnclassifiedActionError: If a declared capability has no action class.
    """
    tools = (
        *render_command_tools(),
        *render_capability_tools(station_info),
        *SESSION_TOOLS,
    )
    seen: set[str] = set()
    for tool in tools:
        if tool.name in seen:
            raise ValueError(f"two tools claim the name {tool.name!r}")
        seen.add(tool.name)
    logger.debug("Rendered %d tools for setup %r", len(tools), station_info.setup)
    return tools


# ══════════════════════════════════════════════════════════════════════════
# Argument validation
# ══════════════════════════════════════════════════════════════════════════


def _bound_errors(args: Mapping[str, Any], schema: Mapping[str, Any]) -> list[str]:
    """Check a call's scalar arguments against the schema's declared bounds.

    The structural validator borrowed from the capability manifest checks
    types, required keys, enums and unexpected keys; a numeric bound is the
    one thing it does not check and the one thing a tool schema exists to
    publish, so it is checked here — and the message names the bound, its
    unit and the configured limit it came from, because that is what an agent
    has to read to correct itself.

    Args:
        args: The call's arguments.
        schema: The tool's input schema.

    Returns:
        One operator-facing message per violated bound.
    """
    errors: list[str] = []
    properties = schema.get("properties") or {}
    for name, value in args.items():
        node = properties.get(name)
        if not isinstance(node, Mapping) or isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        unit = f" {node['unit']}" if node.get("unit") else ""
        lower, upper = node.get("minimum"), node.get("maximum")
        if lower is not None and value < lower:
            errors.append(
                f"{name}: {value}{unit} is below the minimum {lower}{unit}"
            )
        if upper is not None and value > upper:
            errors.append(
                f"{name}: {value}{unit} is above the maximum {upper}{unit}"
            )
    return errors


def validate_tool_args(
    args: Mapping[str, Any], schema: Mapping[str, Any]
) -> list[str]:
    """Check one tool call's arguments against the tool's schema.

    Args:
        args: What the caller supplied.
        schema: The tool's ``input_schema``.

    Returns:
        Human-readable error strings — empty when the call conforms. Returned
        rather than raised, mirroring ``validate_manifest()`` and
        ``validate_config_dir()``: a caller wants every problem at once.
    """
    errors = list(validate_manifest(args, schema))
    errors.extend(_bound_errors(args, schema))
    return errors


# ══════════════════════════════════════════════════════════════════════════
# The session tools' implementations
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ToolContext:
    """The collaborators the session tools read through.

    A gateway with no context still offers every command tool and answers
    ``read_status`` / ``read_station_info`` / ``read_manifest`` from its own
    mirror; a tool whose collaborator is absent is refused by name saying
    which one, never silently answered with nothing.

    Attributes:
        experiments: The L6 façade (``ExperimentManager``-shaped: ``store``,
            ``current_experiment()``, ``validate_run()``), duck-typed so a
            test can pass a stub. It is what supplies the Station to build
            against and the open experiment's envelope.
        run_catalog: ``{class __name__: procedure class}`` — the
            catalog a proposed run's class name is resolved through, supplied
            by whoever owns discovery because this package may not import
            ``i2as.procedures``.
        status_log_path: The operational-status log to tail. Defaults to
            ``paths.log_directory()/status.jsonl``, the file the engine writes.
        status_source: Zero-argument callable returning the latest
            ``StatusSnapshot``; the ``Gateway`` supplies its own mirror.
        station_source: Zero-argument callable returning the latest
            ``StationInfo``; the ``Gateway`` supplies its own mirror.
        draft_client: The **Draft client** ``draft_eln_entry`` asks
            (``complete(system, user, max_tokens)``), duck-typed so a test
            passes ``FakeDraftClient`` and no test reaches a network.
            ``None`` — the default — means this client cannot draft, and the
            tool refuses by name rather than building a model client of its
            own: whoever wires the gateway decides whether an LLM is in the
            loop at all.
        assistant_settings: The token cap and price table a draft is bounded
            and costed by. ``None`` uses ``AssistantSettings()``'s defaults.
        publisher: The ``ElnPublisher`` ``publish_eln_entry`` queues through
            (``export_draft(run_id, draft)``), duck-typed so this package
            imports neither it nor the Qt object that owns its drain timer.
            ``None`` means nothing can be published from here.
        analysis_runner: The **Analysis runner** ``run_analysis`` starts and
            ``read_analysis_report`` asks whether a run is still being
            analysed — duck-typed on ``start(run_id, recipe=..., options=...)``,
            ``is_running(run_id)`` and ``recipe_dirs()``, so this package
            imports neither it nor the Qt object that owns its worker
            process. ``None`` means nothing can be analysed from here, and
            the two tools that need it are refused by name.
        actor: The ``Actor`` of the connection calling the tool — the same one
            it stamps on every ``Command`` — set by the ``Gateway`` from its
            own identity, so a tool that leaves something behind on disk can
            stamp WHO left it. ``None`` outside a gateway (a direct
            ``call_session_tool()`` in a test), which stamps an unknown
            actor rather than refusing: the stamp is a record, not a
            permission.
    """

    experiments: Any | None = None
    run_catalog: Mapping[str, type] = field(default_factory=dict)
    status_log_path: Path | None = None
    status_source: Callable[[], Any] | None = None
    station_source: Callable[[], StationInfo] | None = None
    draft_client: Any | None = None
    assistant_settings: AssistantSettings | None = None
    publisher: Any | None = None
    analysis_runner: Any | None = None
    actor: Any | None = None

    def require_experiments(self, tool_name: str) -> Any:
        """Return the experiment façade, or refuse by name.

        Args:
            tool_name: The tool asking, for the message.

        Returns:
            The experiment manager.

        Raises:
            ToolError: If this gateway was built without one.
        """
        if self.experiments is None:
            raise ToolError(
                f"{tool_name} needs the experiment layer, and this gateway was "
                f"built without one",
                {"rule": "missing_collaborator", "collaborator": "experiments"},
            )
        return self.experiments

    def require_draft_client(self, tool_name: str) -> Any:
        """Return the **Draft client**, or refuse by name.

        Args:
            tool_name: The tool asking, for the message.

        Returns:
            The draft client.

        Raises:
            ToolError: If this gateway was built without one.
        """
        if self.draft_client is None:
            raise ToolError(
                f"{tool_name} needs a draft client, and this gateway was built "
                f"without one — drafting is off for this connection",
                {"rule": "missing_collaborator", "collaborator": "draft_client"},
            )
        return self.draft_client

    def require_publisher(self, tool_name: str) -> Any:
        """Return the ELN publisher, or refuse by name.

        Args:
            tool_name: The tool asking, for the message.

        Returns:
            The publisher.

        Raises:
            ToolError: If this gateway was built without one.
        """
        if self.publisher is None:
            raise ToolError(
                f"{tool_name} needs the ELN publisher, and this gateway was "
                f"built without one",
                {"rule": "missing_collaborator", "collaborator": "publisher"},
            )
        return self.publisher

    def require_analysis_runner(self, tool_name: str) -> Any:
        """Return the **Analysis runner**, or refuse by name.

        Args:
            tool_name: The tool asking, for the message.

        Returns:
            The analysis runner.

        Raises:
            ToolError: If this gateway was built without one.
        """
        if self.analysis_runner is None:
            raise ToolError(
                f"{tool_name} needs the analysis runner, and this gateway was "
                f"built without one — analysis is off for this connection",
                {"rule": "missing_collaborator", "collaborator": "analysis_runner"},
            )
        return self.analysis_runner

    def store(self, tool_name: str) -> Any:
        """Return the experiment store, or refuse by name.

        Args:
            tool_name: The tool asking, for the message.

        Returns:
            The ``ExperimentStore``.

        Raises:
            ToolError: If there is no experiment layer to take it from.
        """
        return self.require_experiments(tool_name).store

    def experiment_id(self, tool_name: str, requested: str = "") -> str:
        """Resolve which experiment a tool is about.

        Args:
            tool_name: The tool asking, for the message.
            requested: The caller's ``experiment_id``, or ``""`` for the open one.

        Returns:
            The experiment id.

        Raises:
            ToolError: If none was given and no experiment is open.
        """
        if requested:
            return requested
        record = self.require_experiments(tool_name).current_experiment()
        identity = getattr(record, "experiment_id", "") if record is not None else ""
        if not identity:
            raise ToolError(
                f"{tool_name} needs an experiment_id: none was given and no "
                f"experiment is open",
                {"rule": "no_experiment"},
            )
        return str(identity)


def _jsonable(value: Any) -> Any:
    """Render a value JSON-safe, turning NaN and infinities into ``null``.

    numpy arrays become nested lists; JSON has no NaN, so a non-finite float
    renders as ``null`` exactly as ``data_reader``'s own JSON round trip
    does.

    Args:
        value: Anything a reader returned.

    Returns:
        A JSON-safe rendering.
    """
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _tail_records(path: Path, count: int) -> list[dict[str, Any]]:
    """Return the last *count* JSON records of an append-only JSONL file.

    Read backwards in blocks rather than whole: these logs grow for as long
    as the app ticks, and every question asked of them needs only the last few
    records. A partial final line (the writer is mid-append) and a partial
    first line (the window began inside an earlier record) are both dropped,
    and an unparseable line is skipped rather than fatal.

    Implemented here rather than borrowed from the troubleshoot toolbox,
    because import-linter contract C9 keeps that toolbox a leaf that nothing
    in ``i2as`` imports. The JSONL format is the contract both read.

    Args:
        path: The file to tail.
        count: How many trailing records are wanted.

    Returns:
        Up to *count* parsed records, oldest first. A missing file gives ``[]``.
    """
    if not path.is_file():
        return []
    chunk_size = 65536
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        chunks: list[bytes] = []
        newlines = 0
        while position > 0 and newlines <= count:
            step = min(chunk_size, position)
            position -= step
            handle.seek(position)
            block = handle.read(step)
            newlines += block.count(b"\n")
            chunks.insert(0, block)
        blob = b"".join(chunks)

    ends_complete = blob.endswith(b"\n")
    lines = blob.decode("utf-8", errors="replace").splitlines()
    if lines and not ends_complete:
        lines.pop()
    if lines and position > 0:
        lines.pop(0)

    records: list[dict[str, Any]] = []
    for line in lines[-count:]:
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            logger.debug("skipping unparseable record in %s", path)
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _experiment_record(context: ToolContext, tool_name: str, experiment_id: str) -> Any:
    """Load one experiment record, refusing by name when it is not there.

    Args:
        context: The tool context.
        tool_name: The tool asking, for the message.
        experiment_id: The experiment to load.

    Returns:
        The ``ExperimentRecord``.

    Raises:
        ToolError: If this store holds no such experiment.
    """
    record = context.store(tool_name).load(experiment_id)
    if record is None:
        raise ToolError(
            f"no experiment {experiment_id!r} in this store",
            {"rule": "unknown_experiment", "experiment_id": experiment_id},
        )
    return record


def _run_record(context: ToolContext, tool_name: str, args: Mapping[str, Any]) -> Any:
    """Find one recorded run, refusing by name when it is not there.

    Args:
        context: The tool context.
        tool_name: The tool asking, for the message.
        args: The call's arguments, carrying ``run_id`` and optional
            ``experiment_id``.

    Returns:
        ``(experiment_id, ExperimentRecord, RunRecord)`` — the experiment as
        well as the run, because a tool that renders one run (a draft, an
        entry) needs the experiment's own title beside it.

    Raises:
        ToolError: If the experiment or the run cannot be found.
    """
    experiment_id = context.experiment_id(tool_name, str(args.get("experiment_id", "")))
    record = _experiment_record(context, tool_name, experiment_id)
    run_id = str(args["run_id"])
    run = record.find_run(run_id)
    if run is None:
        raise ToolError(
            f"experiment {experiment_id!r} has no run {run_id!r}",
            {"rule": "unknown_run", "experiment_id": experiment_id, "run_id": run_id},
        )
    return experiment_id, record, run


def _open_run_file(context: ToolContext, tool_name: str, args: Mapping[str, Any]) -> Any:
    """Open one recorded run's data file, resolved through the store.

    Path resolution goes through ``ExperimentStore.resolve_data_file()`` and
    never through a path the caller supplied: a read tool that accepted an
    arbitrary path would be a file reader wearing an instrument's name.

    Args:
        context: The tool context.
        tool_name: The tool asking, for the message.
        args: The call's arguments.

    Returns:
        An open ``RunHandle``, which the caller closes.

    Raises:
        ToolError: If the run has no data file, or the file cannot be read.
    """
    experiment_id, _record, run = _run_record(context, tool_name, args)
    if not run.data_file:
        raise ToolError(
            f"run {run.run_id!r} recorded no data file",
            {"rule": "no_data_file", "run_id": run.run_id},
        )
    path = context.store(tool_name).resolve_data_file(experiment_id, run.data_file)
    try:
        return data_reader.open_run(path)
    except (OSError, ValueError) as error:
        raise ToolError(
            f"cannot read run {run.run_id!r} at {path}: {error}",
            {"rule": "unreadable_run", "run_id": run.run_id, "path": str(path)},
        ) from error


def _tool_read_status(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``read_status`` from the client's own mirror.

    Args:
        args: Unused; the tool takes none.
        context: The tool context, whose ``status_source`` is the mirror.

    Returns:
        The latest ``StatusSnapshot`` as JSON, or ``None`` before the first tick.
    """
    snapshot = context.status_source() if context.status_source else None
    return None if snapshot is None else snapshot.to_json()


def _tool_read_station_info(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``read_station_info`` from the client's own mirror.

    Args:
        args: Unused; the tool takes none.
        context: The tool context, whose ``station_source`` is the mirror.

    Returns:
        The latest ``StationInfo`` as JSON.
    """
    return _station(context).to_json()


def _tool_read_manifest(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``read_manifest`` by rendering the mirrored declaration.

    ``build_manifest()`` reads nothing but its argument's ``station_info()``,
    so the mirror stands in for the Station and the manifest is produced with
    no bus traffic and no Station of our own.

    Args:
        args: Unused; the tool takes none.
        context: The tool context.

    Returns:
        The capability manifest.
    """
    snapshot = _station(context)
    return build_manifest(_MirroredStation(snapshot))


class _MirroredStation:
    """The one method ``build_manifest()`` reads, backed by a mirrored snapshot.

    Attributes:
        _snapshot: The declaration snapshot to render.
    """

    def __init__(self, snapshot: StationInfo) -> None:
        """Hold the snapshot to render.

        Args:
            snapshot: The station's declaration snapshot.
        """
        self._snapshot = snapshot

    def station_info(self) -> StationInfo:
        """Return the mirrored declaration snapshot.

        Returns:
            The snapshot this stands for.
        """
        return self._snapshot


def _station(context: ToolContext) -> StationInfo:
    """Return the mirrored station declaration, or refuse by name.

    Args:
        context: The tool context.

    Returns:
        The latest ``StationInfo``.

    Raises:
        ToolError: If no mirror was wired in.
    """
    if context.station_source is None:
        raise ToolError(
            "this gateway has no station declaration to read",
            {"rule": "missing_collaborator", "collaborator": "station"},
        )
    return context.station_source()


def _tool_list_runs(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``list_runs`` from the experiment store.

    Args:
        args: The call's arguments, with an optional ``experiment_id``.
        context: The tool context.

    Returns:
        ``{"experiment_id": ..., "runs": [run record dicts]}``.

    Raises:
        ToolError: If the experiment cannot be resolved or loaded.
    """
    experiment_id = context.experiment_id(
        "list_runs", str(args.get("experiment_id", ""))
    )
    record = context.store("list_runs").load(experiment_id)
    if record is None:
        raise ToolError(
            f"no experiment {experiment_id!r} in this store",
            {"rule": "unknown_experiment", "experiment_id": experiment_id},
        )
    return {
        "experiment_id": experiment_id,
        "runs": [run.to_dict() for run in record.runs],
    }


def _tool_read_run_columns(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``read_run_columns`` over the run's data file.

    Args:
        args: The call's arguments.
        context: The tool context.

    Returns:
        ``{"columns": [ColumnInfo dicts]}``.

    Raises:
        ToolError: If the run or its file cannot be found.
    """
    with _open_run_file(context, "read_run_columns", args) as handle:
        return {
            "columns": [
                info.to_json() for info in data_reader.list_columns(handle)
            ]
        }


def _tool_read_run_slice(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``read_run_slice`` over the run's data file.

    Args:
        args: The call's arguments, with the column and optional bounds.
        context: The tool context.

    Returns:
        ``{"column": ..., "values": [...]}`` with non-finite values as ``null``.

    Raises:
        ToolError: If the run, its file, or the column is not there.
    """
    column = str(args["column"])
    with _open_run_file(context, "read_run_slice", args) as handle:
        try:
            values = data_reader.read_slice(
                handle,
                column,
                args.get("start"),
                args.get("stop"),
                args.get("step"),
            )
        except (KeyError, ValueError) as error:
            raise ToolError(
                f"cannot read column {column!r}: {error}",
                {"rule": "bad_column", "column": column},
            ) from error
    return {"column": column, "values": _jsonable(values)}


def _tool_read_run_stats(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``read_run_stats`` over the run's data file.

    Args:
        args: The call's arguments, with the column to summarise.
        context: The tool context.

    Returns:
        The column's ``Stats`` as JSON.

    Raises:
        ToolError: If the run, its file, or the column is not there.
    """
    column = str(args["column"])
    with _open_run_file(context, "read_run_stats", args) as handle:
        try:
            stats = data_reader.summary_stats(handle, column)
        except (KeyError, ValueError) as error:
            raise ToolError(
                f"cannot summarise column {column!r}: {error}",
                {"rule": "bad_column", "column": column},
            ) from error
    return stats.to_json()


def _tool_read_run_metadata(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``read_run_metadata`` over the run's data file.

    Args:
        args: The call's arguments.
        context: The tool context.

    Returns:
        The run's metadata under the canonical keys.

    Raises:
        ToolError: If the run or its file cannot be found.
    """
    with _open_run_file(context, "read_run_metadata", args) as handle:
        return _jsonable(data_reader.read_metadata(handle))


def _tool_validate_run(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``validate_run`` through the experiment layer.

    Args:
        args: The call's arguments, naming the class and its params.
        context: The tool context.

    Returns:
        The ``RunValidation`` as JSON: findings, the duration estimate, and
        the assumptions behind it.

    Raises:
        ToolError: If the class is not in the run catalog, or the experiment
            layer cannot build headlessly.
    """
    experiments = context.require_experiments("validate_run")
    class_name = str(args["procedure"])
    run_class = context.run_catalog.get(class_name)
    if run_class is None:
        raise ToolError(
            f"unknown procedure {class_name!r}: the run catalog holds "
            f"{sorted(context.run_catalog)}",
            {"rule": "unknown_run_class", "procedure": class_name},
        )
    try:
        validation = experiments.validate_run(
            run_class,
            dict(args.get("params") or {}),
            sample_info=dict(args.get("sample_info") or {}),
            data_directory=str(args.get("data_directory", "")),
            file_prefix=str(args.get("file_prefix", "")),
            probe_spec=dict(args.get("probe_spec") or {}),
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise ToolError(
            f"cannot validate {class_name!r}: {error}",
            {"rule": "validation_failed", "procedure": class_name},
        ) from error
    return validation.to_json()


def _tool_read_experiment(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``read_experiment`` from the store.

    Args:
        args: The call's arguments, with an optional ``experiment_id``.
        context: The tool context.

    Returns:
        The experiment record as a dict.

    Raises:
        ToolError: If the experiment cannot be resolved or loaded.
    """
    experiment_id = context.experiment_id(
        "read_experiment", str(args.get("experiment_id", ""))
    )
    record = context.store("read_experiment").load(experiment_id)
    if record is None:
        raise ToolError(
            f"no experiment {experiment_id!r} in this store",
            {"rule": "unknown_experiment", "experiment_id": experiment_id},
        )
    return record.to_dict()


def _tool_read_operational_log(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``read_operational_log`` by tailing status.jsonl.

    Args:
        args: The call's arguments, with an optional ``last``.
        context: The tool context, whose ``status_log_path`` may override the
            resolved log directory.

    Returns:
        ``{"path": ..., "records": [...]}``, oldest record first.
    """
    path = context.status_log_path or (log_directory() / OPERATIONAL_LOG_FILENAME)
    count = int(args.get("last", 20))
    return {"path": str(path), "records": _tail_records(Path(path), count)}


def _tool_read_agent_feed(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``read_agent_feed`` by tailing the experiment's action feed.

    The feed's path comes from ``ExperimentStore.agent_feed_path()`` where
    that exists, and otherwise from the store's documented layout —
    ``<root>/<experiment_id>/agent_actions.jsonl``, beside ``outbox.jsonl``.

    Args:
        args: The call's arguments, with optional ``experiment_id`` and ``last``.
        context: The tool context.

    Returns:
        ``{"experiment_id": ..., "path": ..., "records": [...]}``, oldest first.

    Raises:
        ToolError: If the experiment cannot be resolved.
    """
    experiment_id = context.experiment_id(
        "read_agent_feed", str(args.get("experiment_id", ""))
    )
    store = context.store("read_agent_feed")
    getter = getattr(store, "agent_feed_path", None)
    if callable(getter):
        path = Path(getter(experiment_id))
    else:
        path = Path(store.root) / experiment_id / AGENT_FEED_FILENAME
    count = int(args.get("last", 20))
    return {
        "experiment_id": experiment_id,
        "path": str(path),
        "records": _tail_records(path, count),
    }


def _draft_stats(
    context: ToolContext, args: Mapping[str, Any], run: Any
) -> dict[str, dict[str, Any]]:
    """Summarise every numeric column of one run, tolerating an unreadable file.

    Deliberately best-effort, unlike ``read_run_stats``: a run that recorded
    no data file, or whose file cannot be opened, still deserves a draft —
    the prompt then says the statistics were unavailable and the model is
    told to say so, which is a better answer than refusing to draft an
    aborted run at all.

    Args:
        context: The tool context.
        args: The call's arguments, naming the run.
        run: The recorded run.

    Returns:
        ``{column: Stats.to_json()}``, empty when there was nothing to read.
    """
    if not getattr(run, "data_file", ""):
        logger.info(
            "Run %s recorded no data file — drafting without statistics",
            getattr(run, "run_id", ""),
        )
        return {}
    stats: dict[str, dict[str, Any]] = {}
    try:
        with _open_run_file(context, "draft_eln_entry", args) as handle:
            for info in data_reader.list_columns(handle):
                if info.dtype == "str":
                    continue
                try:
                    stats[info.name] = data_reader.summary_stats(
                        handle, info.name
                    ).to_json()
                except (KeyError, ValueError):
                    logger.warning(
                        "Column %r of run %s could not be summarised for the draft",
                        info.name,
                        getattr(run, "run_id", ""),
                    )
    except (ToolError, OSError, ValueError, RuntimeError) as error:
        # Deliberately wider than `read_run_stats`'s refusal: a half-written or
        # corrupt file (which the HDF5 layer reports as a RuntimeError) must
        # cost the draft its statistics, never the whole draft.
        logger.warning(
            "Drafting run %s without statistics: %s",
            getattr(run, "run_id", ""),
            error,
        )
        return {}
    return stats


def _tool_draft_eln_entry(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``draft_eln_entry`` through the ELN track's drafting module.

    Assembles the facts (the recorded run, its column statistics, the station
    and status mirrors, the experiment and its setup) and hands them to
    ``draft_entry()``. This function contains no prompt text and no rendering
    of its own: the prompt is the draft prompt standard's, the body is
    ``templates.py``'s, and what comes back is returned unchanged as data.

    Args:
        args: The call's arguments — the run, optionally its experiment, and
            an optional note carried into the prompt.
        context: The tool context.

    Returns:
        The **draft entry** as its JSON dict, cost line included.

    Raises:
        ToolError: If the run cannot be found, no draft client was wired in,
            or the model could not be reached.
    """
    experiment_id, record, run = _run_record(context, "draft_eln_entry", args)
    client = context.require_draft_client("draft_eln_entry")
    experiments = context.require_experiments("draft_eln_entry")

    station = context.station_source() if context.station_source else None
    snapshot = context.status_source() if context.status_source else None
    setup = experiments.experiment_context().get("setup") or {}
    data_path = ""
    if run.data_file:
        data_path = str(
            context.store("draft_eln_entry").resolve_data_file(
                experiment_id, run.data_file
            )
        )
    request = DraftRequest(
        run_id=run.run_id,
        experiment_id=experiment_id,
        manifest=manifest_from_run(run),
        stats=_draft_stats(context, args, run),
        station=station.to_json() if station is not None else {},
        status=snapshot.to_json() if snapshot is not None else {},
        experiment_title=getattr(record, "title", ""),
        setup=dict(setup),
        data_path=data_path,
        template_id=_template_id(context),
        operator_note=str(args.get("note", "")),
    )
    try:
        return draft_entry(request, client, context.assistant_settings).to_dict()
    except ElnError as error:
        raise ToolError(
            f"the draft for run {run.run_id!r} could not be written: {error}",
            {"rule": "draft_failed", "run_id": run.run_id},
        ) from error


def _template_id(context: ToolContext) -> str:
    """Return the notebook template a published entry would be created from.

    Args:
        context: The tool context.

    Returns:
        The publisher's configured ``template_id``, or ``""`` when no
        publisher is wired in — a fact about the entry, never a reason to
        refuse a draft.
    """
    settings = getattr(context.publisher, "settings", None)
    return str(getattr(settings, "template_id", "") or "")


def _tool_publish_eln_entry(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``publish_eln_entry``, or refuse it because a human must approve.

    The approval rule, and the one place it is enforced: while the open
    experiment is ATTENDED an agent may draft but never publish, so the draft
    is parked on the run record (through the manager, the single writer of
    experiment state) and the call is refused with ``approval_required`` — a
    refusal that leaves the work where the human will find it, rather than
    discarding it. Unattended, the draft goes to the publisher's outbox,
    which is the same journal a manual export uses.

    Args:
        args: The call's arguments — the run and the approved draft.
        context: The tool context.

    Returns:
        ``{"run_id", "experiment_id", "job_id", "queued"}`` on success.

    Raises:
        ToolError: If the run is unknown, the experiment is attended, no
            publisher was wired in, or nothing was queued.
    """
    experiment_id, record, run = _run_record(context, "publish_eln_entry", args)
    experiments = context.require_experiments("publish_eln_entry")
    draft = dict(args.get("draft") or {})

    if getattr(record, "attended", False):
        stored = bool(experiments.set_pending_eln_draft(run.run_id, draft))
        whereabouts = (
            "It is waiting on the run record for approval."
            if stored
            else "It could NOT be parked on the run record, and is lost."
        )
        raise ToolError(
            f"experiment {experiment_id!r} is attended: a human approves an "
            f"ELN entry before it is published, so the draft for run "
            f"{run.run_id!r} was not queued. {whereabouts}",
            {
                "rule": "approval_required",
                "attended": True,
                "experiment_id": experiment_id,
                "run_id": run.run_id,
                "pending": stored,
            },
        )

    publisher = context.require_publisher("publish_eln_entry")
    job_id = str(publisher.export_draft(run.run_id, draft) or "")
    if not job_id:
        raise ToolError(
            f"nothing was queued for run {run.run_id!r}: ELN publishing is "
            f"switched off, or the run is already queued",
            {"rule": "not_queued", "run_id": run.run_id},
        )
    return {
        "run_id": run.run_id,
        "experiment_id": experiment_id,
        "job_id": job_id,
        "queued": True,
    }


# ══════════════════════════════════════════════════════════════════════════
# The analysis tools: the recipes an experiment is analysed with, and the
# reports the analysis worker leaves behind
# ══════════════════════════════════════════════════════════════════════════


def _analysis_discovery(tool_name: str) -> Any:
    """Import the recipe-discovery module, refusing by name when it is absent.

    Imported here rather than at module import time so this package still
    loads — and every other tool still answers — in an installation without
    the analysis stage.

    Args:
        tool_name: The tool asking, for the message.

    Returns:
        The ``i2as.analysis.discovery`` module.

    Raises:
        ToolError: If the analysis package cannot be imported.
    """
    try:
        from i2as.analysis import discovery
    except ImportError as error:
        raise ToolError(
            f"{tool_name} needs i2as.analysis, which this installation "
            f"does not provide: {error}",
            {"rule": "missing_collaborator", "collaborator": "analysis"},
        ) from error
    return discovery


def _store_directory(
    context: ToolContext, tool_name: str, method_name: str, *args: Any
) -> Path:
    """Ask the experiment store for one of its directories, by name.

    Directories are the store's to lay out — an analysis tool never joins a
    path of its own, the same rule ``ExperimentStore.resolve_data_file()``
    already imposes on every run file.

    Args:
        context: The tool context.
        tool_name: The tool asking, for the message.
        method_name: The store method to call (``recipes_dir``/``report_dir``).
        *args: Its arguments.

    Returns:
        The directory, which this function does not create.

    Raises:
        ToolError: If the store does not offer that method.
    """
    store = context.store(tool_name)
    method = getattr(store, method_name, None)
    if not callable(method):
        raise ToolError(
            f"{tool_name} needs ExperimentStore.{method_name}(), which this "
            f"store does not offer",
            {"rule": "missing_collaborator", "collaborator": f"store.{method_name}"},
        )
    return Path(method(*args))


def _discovered_recipes(
    context: ToolContext, tool_name: str, experiment_id: str
) -> tuple[Path, tuple[Any, ...]]:
    """Discover every recipe available to one experiment.

    Args:
        context: The tool context.
        tool_name: The tool asking, for the message.
        experiment_id: The experiment whose own recipes are added to the
            package's.

    Returns:
        ``(recipes_dir, recipes)`` — the experiment's recipe folder (which may
        not exist yet) and the ``RecipeInfo``s, package recipes first.

    Raises:
        ToolError: If the analysis package or the store method is absent.
    """
    discovery = _analysis_discovery(tool_name)
    recipes_dir = _store_directory(context, tool_name, "recipes_dir", experiment_id)
    return recipes_dir, tuple(discovery.discover_recipes([recipes_dir]))


def _recipe_preferences(context: ToolContext) -> dict[str, str]:
    """Return the notebook settings' per-procedure recipe preference.

    Read through ``getattr`` at every step: a gateway wired without a
    publisher, or one whose publisher predates the analysis settings, has no
    preference rather than a failure — the preference decides which of
    several recipes runs, never whether the question can be answered.

    Args:
        context: The tool context.

    Returns:
        ``{procedure: recipe name}``, empty when nothing declares one.
    """
    settings = getattr(context.publisher, "settings", None)
    preferences = getattr(getattr(settings, "analysis", None), "recipes", None)
    if not isinstance(preferences, Mapping):
        return {}
    return {str(key): str(value) for key, value in preferences.items() if value}


def _actor_stamp(context: ToolContext) -> str:
    """Render the calling actor as ``<kind> '<id>'`` for a written-file header.

    Args:
        context: The tool context, carrying the connection's ``Actor``.

    Returns:
        The two words a header line names the writer with; an unknown actor
        renders as ``unknown '?'`` rather than as nothing, so a stamp is
        never silently empty.
    """
    actor = context.actor
    kind = getattr(getattr(actor, "kind", None), "value", None) or "unknown"
    identity = str(getattr(actor, "id", "") or "?")
    return f"{kind} {identity!r}"


def _digest(text: str) -> str:
    """Return the SHA-256 of a text, the same digest a ``RecipeInfo`` carries.

    Args:
        text: The text to digest.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tool_list_analysis_recipes(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``list_analysis_recipes``.

    Args:
        args: The call's arguments — optionally the experiment.
        context: The tool context.

    Returns:
        ``{"recipes", "recipes_dir", "selected"}`` — every discovered recipe,
        the folder this experiment's own recipes live in, and the recipe that
        would run for each procedure the experiment has recorded a run of.

    Raises:
        ToolError: If the experiment cannot be resolved or the analysis
            package is absent.
    """
    tool_name = "list_analysis_recipes"
    experiment_id = context.experiment_id(tool_name, str(args.get("experiment_id", "")))
    record = _experiment_record(context, tool_name, experiment_id)
    recipes_dir, recipes = _discovered_recipes(context, tool_name, experiment_id)
    discovery = _analysis_discovery(tool_name)
    preferences = _recipe_preferences(context)

    selected: dict[str, str] = {}
    for run in getattr(record, "runs", ()) or ():
        procedure = str(getattr(run, "procedure", "") or "")
        if not procedure or procedure in selected:
            continue
        chosen = discovery.recipe_for(
            procedure, recipes, preferences.get(procedure, "")
        )
        if chosen is not None:
            selected[procedure] = chosen.name
    return {
        "recipes": [info.to_dict() for info in recipes],
        "recipes_dir": str(recipes_dir),
        "selected": selected,
    }


def _tool_read_analysis_recipe(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``read_analysis_recipe``.

    An experiment recipe is read from the experiment's own folder and a
    package recipe from the file the discovery reported, so no caller-supplied
    path is ever opened — the rule every run-file read already follows.

    Args:
        args: The call's arguments — the recipe name and optionally the
            experiment.
        context: The tool context.

    Returns:
        ``{"name", "path", "source", "origin", "digest"}``.

    Raises:
        ToolError: If no recipe answers to that name, or its file cannot be
            read.
    """
    tool_name = "read_analysis_recipe"
    experiment_id = context.experiment_id(tool_name, str(args.get("experiment_id", "")))
    name = str(args["name"])
    recipes_dir, recipes = _discovered_recipes(context, tool_name, experiment_id)
    info = next((item for item in recipes if item.name == name), None)
    if info is None:
        raise ToolError(
            f"no analysis recipe named {name!r} for experiment "
            f"{experiment_id!r}; list_analysis_recipes says which there are",
            {"rule": "unknown_recipe", "name": name, "experiment_id": experiment_id},
        )
    path = (
        recipes_dir / f"{name}.py"
        if info.origin == "experiment"
        else Path(info.source_path)
    )
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ToolError(
            f"the recipe {name!r} at {path} cannot be read: {error}",
            {"rule": "unreadable_recipe", "name": name, "path": str(path)},
        ) from error
    return {
        "name": info.name,
        "path": str(path),
        "source": source,
        "origin": info.origin,
        "digest": info.digest,
    }


def _tool_write_analysis_recipe(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``write_analysis_recipe``: put one recipe in the experiment folder.

    The source is compiled but never executed here — writing a recipe and
    running one are two actions, and the second is ``run_analysis``, in the
    analysis worker's own process. What this leaves behind is a stamped file:
    a header naming the actor and the UTC time above the source the caller
    sent, so the folder itself says who wrote what, beside the **Agent feed**
    record of the same call.

    Args:
        args: The call's arguments — the name, the source, whether an existing
            file may be replaced, and optionally the experiment.
        context: The tool context.

    Returns:
        ``{"name", "path", "bytes", "digest"}`` — the digest is of the file as
        written, header included, which is the digest a ``RecipeInfo`` for it
        will carry.

    Raises:
        ToolError: If the name is not a plain identifier, the source does not
            compile or is too large, the file exists and may not be replaced,
            or it cannot be written.
    """
    tool_name = "write_analysis_recipe"
    experiment_id = context.experiment_id(tool_name, str(args.get("experiment_id", "")))
    name = str(args["name"])
    source = str(args["source"])

    if not name.isidentifier() or name.startswith("_"):
        raise ToolError(
            f"{name!r} is not a usable recipe name: it must be a plain Python "
            f"identifier that does not start with an underscore",
            {"rule": "invalid_name", "name": name},
        )
    size = len(source.encode("utf-8"))
    if size > MAX_RECIPE_BYTES:
        raise ToolError(
            f"the recipe {name!r} is {size} bytes, over the "
            f"{MAX_RECIPE_BYTES}-byte limit a reviewable recipe is held to",
            {"rule": "too_large", "name": name, "bytes": size,
             "limit": MAX_RECIPE_BYTES},
        )
    try:
        compile(source, f"<{name}.py>", "exec")
    except SyntaxError as error:
        raise ToolError(
            f"the recipe {name!r} does not compile: {error.msg} "
            f"(line {error.lineno})",
            {
                "rule": "syntax_error",
                "name": name,
                "line": int(error.lineno or 0),
                "message": str(error.msg),
            },
        ) from error

    recipes_dir = _store_directory(context, tool_name, "recipes_dir", experiment_id)
    path = recipes_dir / f"{name}.py"
    if path.exists() and not bool(args.get("overwrite", False)):
        raise ToolError(
            f"the recipe {name!r} already exists at {path}; call it again with "
            f"overwrite true to replace it",
            {"rule": "exists", "name": name, "path": str(path)},
        )

    stamped_at = datetime.now(timezone.utc).isoformat()
    header = (
        f"# Written by {_actor_stamp(context)} via {tool_name} at {stamped_at}\n"
        f"# The analysis worker executes this file in its own process; it is "
        f"visible\n"
        f"# in the eLab tab, and recorded in the agent feed, before anybody "
        f"runs it.\n\n"
    )
    content = header + source
    try:
        recipes_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as error:
        raise ToolError(
            f"the recipe {name!r} could not be written to {path}: {error}",
            {"rule": "unwritable_recipe", "name": name, "path": str(path)},
        ) from error
    logger.info(
        "analysis recipe %r written to %s by %s", name, path, _actor_stamp(context)
    )
    return {
        "name": name,
        "path": str(path),
        "bytes": len(content.encode("utf-8")),
        "digest": _digest(content),
    }


def _tool_run_analysis(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``run_analysis``: start the analysis worker on one recorded run.

    Starting is all this does. The worker is a separate process with the data
    file and no instrument, and its report arrives on disk later — which is
    why the answer carries the path the report WILL be at and why
    ``read_analysis_report`` is the tool that reads it.

    Args:
        args: The call's arguments — the run, optionally its experiment, the
            recipe and the options.
        context: The tool context.

    Returns:
        ``{"run_id", "started", "report_path", "recipe"}``.

    Raises:
        ToolError: If the run is unknown, no runner was wired in, that run is
            already being analysed, or the runner started nothing.
    """
    tool_name = "run_analysis"
    _experiment_id, _record, run = _run_record(context, tool_name, args)
    runner = context.require_analysis_runner(tool_name)
    recipe = str(args.get("recipe", ""))
    options = dict(args.get("options") or {})

    if runner.is_running(run.run_id):
        raise ToolError(
            f"run {run.run_id!r} is being analysed already; read its report "
            f"with read_analysis_report rather than starting a second worker",
            {"rule": "already_running", "run_id": run.run_id},
        )
    report_dir = str(runner.start(run.run_id, recipe=recipe, options=options) or "")
    if not report_dir:
        raise ToolError(
            f"nothing was started for run {run.run_id!r}: no experiment is "
            f"open, the run is not known to the runner, or it wrote no data "
            f"file",
            {"rule": "not_started", "run_id": run.run_id},
        )
    return {
        "run_id": run.run_id,
        "started": True,
        "report_path": str(Path(report_dir) / REPORT_FILENAME),
        "recipe": recipe,
    }


def _tool_read_analysis_report(args: Mapping[str, Any], context: ToolContext) -> Any:
    """Answer ``read_analysis_report``: the report of one run, or why there is none.

    Args:
        args: The call's arguments — the run and optionally its experiment.
        context: The tool context.

    Returns:
        The **analysis report** as its dict, plus ``report_path``; or
        ``{"status": "running"}`` while the worker is still on that run, or
        ``{"status": "none"}`` when nothing has been analysed yet.

    Raises:
        ToolError: If the run is unknown or the report file cannot be read.
    """
    tool_name = "read_analysis_report"
    experiment_id, _record, run = _run_record(context, tool_name, args)
    runner = context.analysis_runner
    if runner is not None and runner.is_running(run.run_id):
        return {"status": "running", "run_id": run.run_id}

    path = (
        _store_directory(context, tool_name, "report_dir", experiment_id, run.run_id)
        / REPORT_FILENAME
    )
    if not path.is_file():
        return {"status": "none", "run_id": run.run_id}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ToolError(
            f"the analysis report for run {run.run_id!r} at {path} cannot be "
            f"read: {error}",
            {"rule": "unreadable_report", "run_id": run.run_id, "path": str(path)},
        ) from error
    if not isinstance(payload, Mapping):
        raise ToolError(
            f"the analysis report for run {run.run_id!r} at {path} is not an "
            f"object",
            {"rule": "unreadable_report", "run_id": run.run_id, "path": str(path)},
        )
    report = AnalysisReport.from_dict(payload).to_dict()
    report["report_path"] = str(path)
    return report


def feed_arguments(tool: ToolSpec, args: Mapping[str, Any]) -> dict[str, Any]:
    """Render one tool call's arguments for the **Agent feed**.

    Everything a call was made with is recorded, except an argument named in
    ``FEED_DIGESTED_ARGS``, which is replaced by its byte count and the
    SHA-256 of its text. The feed is an append-only record that lives as long
    as the experiment does, and a recipe's whole source in every line would
    make it unreadable while adding nothing the file itself does not already
    hold — the digest is what proves WHICH text was written.

    Args:
        tool: The tool that was called.
        args: The arguments it was called with.

    Returns:
        The arguments as the feed should carry them.

    """
    recorded = dict(args)
    for name in FEED_DIGESTED_ARGS.get(tool.name, ()):
        if name not in recorded:
            continue
        text = str(recorded.pop(name))
        recorded[f"{name}_bytes"] = len(text.encode("utf-8"))
        recorded[f"{name}_digest"] = _digest(text)
    return recorded


#: One implementation per session tool answered here, keyed by
#: ``ToolSpec.session_function``. ``probe_run`` is deliberately absent: it
#: wraps ``run_procedure`` and is submitted like any other command.
SESSION_TOOL_FUNCTIONS: dict[str, Callable[[Mapping[str, Any], ToolContext], Any]] = {
    "read_status": _tool_read_status,
    "read_station_info": _tool_read_station_info,
    "read_manifest": _tool_read_manifest,
    "list_runs": _tool_list_runs,
    "read_run_columns": _tool_read_run_columns,
    "read_run_slice": _tool_read_run_slice,
    "read_run_stats": _tool_read_run_stats,
    "read_run_metadata": _tool_read_run_metadata,
    "validate_run": _tool_validate_run,
    "read_experiment": _tool_read_experiment,
    "read_operational_log": _tool_read_operational_log,
    "read_agent_feed": _tool_read_agent_feed,
    "draft_eln_entry": _tool_draft_eln_entry,
    "publish_eln_entry": _tool_publish_eln_entry,
    "list_analysis_recipes": _tool_list_analysis_recipes,
    "read_analysis_recipe": _tool_read_analysis_recipe,
    "write_analysis_recipe": _tool_write_analysis_recipe,
    "run_analysis": _tool_run_analysis,
    "read_analysis_report": _tool_read_analysis_report,
}


def call_session_tool(
    spec: ToolSpec, args: Mapping[str, Any], context: ToolContext
) -> Any:
    """Run one session tool and return its JSON-safe result.

    Args:
        spec: The tool being called.
        args: Its validated arguments.
        context: The collaborators to read through.

    Returns:
        Whatever the tool answers, JSON-safe.

    Raises:
        ToolError: If the tool has no implementation, or the call cannot be
            answered.
    """
    implementation = SESSION_TOOL_FUNCTIONS.get(spec.session_function)
    if implementation is None:
        raise ToolError(
            f"tool {spec.name!r} declares the session function "
            f"{spec.session_function!r}, which is not implemented",
            {"rule": "unimplemented_tool", "tool": spec.name},
        )
    return implementation(args, context)
