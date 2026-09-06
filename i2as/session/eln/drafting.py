"""LLM drafting — turning one finished run into a **draft entry**, as data.

**A draft is data, not a privileged path.** Everything here is reached through
the same **Tool surface**, judged by the same **Action class** matrix, and
recorded in the same **Agent feed** as any other action an autonomous client
takes. Nothing in this module publishes anything: it renders facts into a
prompt, asks one model for prose, and returns a ``DraftEntry`` that the caller
may show a human or hand to the **Outbox** — the write half stays exactly
where it already was.

``DraftEntry`` is also the shape of every **Pending entry**, not only of an
LLM draft. The run record's ``pending_eln_draft`` slot holds whatever is
waiting for a human: a model draft (``source="model"``), an analysed entry
built from an **Analysis report** (``source="analysis"``), or the facts-only
fallback parked when the analysis stage failed or was never asked
(``source="facts"``). One shape, one approval gate, one enqueue — the stage
that produced the entry is a field on it, never a second write path. An entry
may carry its own ``attachments`` (an analysed entry's figures) and its own
``metadata`` (the recipe and its digest), both travelling with it into the
outbox job.

The draft prompt standard
-------------------------

The prompt is a *standard*, in the same sense the driver contract and the ELN
adapter contract are: written down here, deterministic, and machine-checked.

1. **Two halves, both plain text.** ``DRAFT_SYSTEM_PROMPT`` is a constant —
   who the assistant is and what it may not do — and
   ``render_draft_prompt(request)`` renders the facts. Neither contains
   markup, a URL, or anything read from the environment.
2. **Deterministic.** The same ``DraftRequest`` renders byte-identical text:
   every mapping is emitted in sorted key order, floats through ``repr``, and
   no clock, path, or random value is ever read. ``prompt_digest`` is the
   SHA-256 of the two halves joined, so two drafts of the same run are
   provably the same question — and a changed prompt is visible as a changed
   digest in the entry that came out of it.
3. **Facts in a fixed order**, each under a bare uppercase heading: ``RUN``,
   ``PARAMETERS``, ``COLUMN STATISTICS``, ``STATION``, ``STATE AT RUN END``,
   ``OPERATOR NOTE``. Nothing is summarised, rounded, or interpreted on the
   way in — the model reads what the run recorded.
4. **The answer shape is two markers**, ``TITLE:`` on one line and ``SUMMARY:``
   before the prose, because a marker parses tolerantly where JSON does not:
   a completion missing either marker still yields a usable draft (the run's
   own rendered title, and the whole text as the summary) instead of an
   error.
5. **The facts are rendered, not delegated.** The body a draft carries is
   ``templates.render_draft_body()``: the model's prose *above* the same
   escaped, self-contained tables a published run gets. So a reviewer checks
   every drafted sentence against the numbers printed beneath it, and a model
   that says nothing useful still produces a complete, correct entry.

The **Draft client** contract
-----------------------------

One method, ``complete(system, user, max_tokens) -> CompletionResult``, so the
model is one injectable collaborator and every test runs against
``FakeDraftClient`` with no network. ``AnthropicDraftClient`` is the real one;
its SDK is an optional dependency (``pip install i2as[assistant]``),
imported lazily so that a checkout without it imports this module, renders
prompts, and runs every test unchanged — and gets one clear ``ElnError`` at
the moment a client is constructed, never a stack trace from an import.

**The key is never logged and never reaches an entry.** It comes from the
user-level settings file's ``assistant.api_key`` (or its
``I2AS_ASSISTANT_APIKEY`` override), is redacted by ``AssistantSettings``
in ``repr()`` and ``to_dict()``, and is passed to the vendor SDK and nowhere
else. An empty key deliberately means "let the SDK resolve credentials from
the environment", which is how an installation keeps it out of every file.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from i2as.session.eln.adapter import ElnError
from i2as.session.eln.settings import AssistantSettings
from i2as.session.eln.templates import render_draft_body, render_run_title

logger = logging.getLogger(__name__)

#: Marker the completion puts its one-line title on.
TITLE_MARKER = "TITLE:"

#: Marker the completion puts its prose after.
SUMMARY_MARKER = "SUMMARY:"

#: The four fields a draft reports what it cost in — the **cost line**. One
#: tuple, so ``DraftEntry.cost_line()`` and the **Agent feed** record of the
#: tool call that produced it are provably the same four fields.
COST_FIELDS: tuple[str, ...] = ("model", "input_tokens", "output_tokens", "cost_usd")

#: The tag every drafted entry carries, so a notebook can find the entries
#: that began as machine drafts however they were later edited.
DRAFT_TAG = "draft"

#: ``DraftEntry.source`` of an entry an LLM drafted from a run's facts.
SOURCE_MODEL = "model"

#: ``DraftEntry.source`` of an entry the analysis stage produced from a
#: recipe's **Analysis report**.
SOURCE_ANALYSIS = "analysis"

#: ``DraftEntry.source`` of the facts-only entry parked when analysis is off
#: for this entry, failed, or timed out — the fallback that loses nothing.
SOURCE_FACTS = "facts"

#: The system half of the draft prompt standard. A constant, so it is part of
#: every draft's ``prompt_digest`` and a change to it is visible in the record.
DRAFT_SYSTEM_PROMPT = (
    "You are drafting one electronic-lab-notebook entry for a cryostat "
    "measurement run. You are given only the facts the run recorded: its "
    "procedure and parameters, per-column summary statistics, the station it "
    "ran on, and the engine's state when it finished.\n"
    "\n"
    "Rules:\n"
    "- Describe only what the facts show. Never invent a number, a unit, a "
    "sample, or an instrument that is not listed.\n"
    "- Do not state a physical conclusion the statistics do not support. "
    "Where the data is ambiguous, say so plainly.\n"
    "- Say explicitly if something looks wrong: a column with no finite "
    "values, a run that did not finish, a fault or a hold at run end.\n"
    "- Write plain prose in short paragraphs. No markup, no lists, no "
    "headings, no tables — the facts are tabulated for the reader already.\n"
    "- A human approves this entry before it is published. Write for that "
    "reviewer.\n"
    "\n"
    "Answer in exactly this shape, and nothing else:\n"
    f"{TITLE_MARKER} <one line naming the run in the notebook's index>\n"
    f"{SUMMARY_MARKER}\n"
    "<your paragraphs>"
)


@dataclass(frozen=True)
class CompletionResult:
    """One model completion and what it cost in tokens.

    Attributes:
        text: The generated text, joined across the completion's text blocks.
        model: The model that actually answered, as the vendor reported it —
            never the model that was asked for, so a substitution is visible.
        input_tokens: Tokens the request consumed.
        output_tokens: Tokens the completion generated.
    """

    text: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class DraftClient(Protocol):
    """The one model call this package makes — the **Draft client** contract.

    Deliberately one synchronous method with no streaming, no tools and no
    conversation: a draft is one question and one answer. Implementations
    raise ``ElnError`` on any failure, exactly as an **ELN adapter** does, so
    a caller has one exception type to catch whichever collaborator failed.
    """

    def complete(self, system: str, user: str, max_tokens: int) -> CompletionResult:
        """Answer one prompt.

        Args:
            system: The system half of the prompt.
            user: The user half — the run's facts.
            max_tokens: Cap on the generated tokens.

        Returns:
            The completion and its token counts.

        Raises:
            ElnError: The model could not be reached, or refused.
        """
        ...


@dataclass(frozen=True)
class DraftRequest:
    """Everything one draft is written from — the facts, and nothing else.

    In-memory only: unlike ``DraftEntry`` this is never persisted, because it
    is rebuilt from the run's own record and the client's mirrors whenever a
    draft is asked for.

    Attributes:
        run_id: The run being drafted.
        experiment_id: The owning experiment's store key.
        manifest: The run's manifest-shaped facts — ``procedure``, ``kind``,
            ``params``, the timestamps, the terminal ``status`` and ``reason``
            (``manifest_from_run()`` builds it from a ``RunRecord``).
        stats: ``{column: Stats.to_json()}``, the NaN-aware summary
            ``core.data_reader.summary_stats()`` gives each numeric column.
        station: The **Station info** snapshot as JSON — what the run ran on.
        status: The latest ``StatusSnapshot`` as JSON at run end, or ``{}``
            when the client had none.
        experiment_title: The owning experiment's title, or ``""``.
        setup: The setup tier of ``ExperimentManager.experiment_context()``.
        data_path: Where the run's data file lives, or ``""``.
        template_id: The backend template the entry would be created from.
        operator_note: The operator's own note to the drafter, or ``""``.
    """

    run_id: str = ""
    experiment_id: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    station: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)
    experiment_title: str = ""
    setup: dict[str, Any] = field(default_factory=dict)
    data_path: str = ""
    template_id: str = ""
    operator_note: str = ""


@dataclass(frozen=True)
class DraftEntry:
    """One drafted notebook entry, plus what drafting it cost.

    JSON-safe and tolerantly loaded, because it is persisted: an attended
    experiment parks it on the run record as ``pending_eln_draft`` until a
    human approves it, and that record may be read back by a later process.

    It is also the shape EVERY **Pending entry** takes, not only a model
    draft: an analysed entry and a facts-only fallback are parked in the same
    slot and say which they are in ``source``, so the approval gate has one
    kind of thing to approve and the notebook one kind of thing to queue.

    Attributes:
        title: The entry title, plain text.
        body_html: The rendered body — the drafted prose above the run's own
            escaped, self-contained fact tables (``render_draft_body()``), or
            the analysed entry's own body.
        tags: Tags to set on the entry; a model draft always carries
            ``DRAFT_TAG``.
        model: The model that answered, as the vendor reported it.
        input_tokens: Tokens the prompt consumed.
        output_tokens: Tokens the completion generated.
        cost_usd: What those tokens cost at the settings' price table, or
            ``0.0`` when the model has no price row (never a guess).
        prompt_digest: SHA-256 of the exact prompt that produced this draft,
            so two drafts of one run are provably the same question.
        attachments: Extra files to attach to the entry, in order, as
            ``{"path": absolute path, "comment": caption}``. Empty for a
            model draft; an analysed entry lists its figures here.
        attach_data_file: Whether the run's raw data file is attached too.
            ``True`` is what a published run and a model draft have always
            done; an analysed entry whose report asked for figures alone sets
            it ``False``.
        source: Which stage produced this entry — ``"model"`` (an LLM draft),
            ``"analysis"`` (an analysed entry) or ``"facts"`` (the facts-only
            fallback). The entry that reaches the notebook says so.
        metadata: Extra JSON-safe provenance stamped into the entry's
            metadata block when it is queued (an analysed entry's ``recipe``
            and ``recipe_digest``). Never a credential and never a live
            object.
    """

    title: str = ""
    body_html: str = ""
    tags: list[str] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    prompt_digest: str = ""
    attachments: list[dict[str, Any]] = field(default_factory=list)
    attach_data_file: bool = True
    source: str = SOURCE_MODEL
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "title": self.title,
            "body_html": self.body_html,
            "tags": list(self.tags),
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "prompt_digest": self.prompt_digest,
            "attachments": [dict(item) for item in self.attachments],
            "attach_data_file": self.attach_data_file,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: object) -> DraftEntry:
        """Build a ``DraftEntry`` from a parsed dict, tolerating bad input.

        Args:
            data: Any parsed JSON value; junk degrades to defaults.

        Returns:
            The draft record.
        """
        if not isinstance(data, dict):
            return cls()
        return cls(
            title=_as_str(data.get("title")),
            body_html=_as_str(data.get("body_html")),
            tags=_as_str_list(data.get("tags")),
            model=_as_str(data.get("model")),
            input_tokens=_as_int(data.get("input_tokens")),
            output_tokens=_as_int(data.get("output_tokens")),
            cost_usd=_as_float(data.get("cost_usd")),
            prompt_digest=_as_str(data.get("prompt_digest")),
            attachments=_as_attachments(data.get("attachments")),
            attach_data_file=_as_bool(data.get("attach_data_file"), True),
            source=_as_str(data.get("source"), SOURCE_MODEL) or SOURCE_MODEL,
            metadata=dict(data["metadata"]) if isinstance(data.get("metadata"), dict) else {},
        )

    def cost_line(self) -> dict[str, Any]:
        """Return the four cost fields the **Agent feed** records for a draft.

        Returns:
            ``{"model", "input_tokens", "output_tokens", "cost_usd"}`` — what
            an autonomous client spent, in the trail beside what it asked for.
        """
        return {field_name: getattr(self, field_name) for field_name in COST_FIELDS}


def cost_line(result: object) -> dict[str, Any]:
    """Return the cost line a tool result carries, or ``{}``.

    The read side of ``DraftEntry.cost_line()``, for a caller holding the
    JSON dict rather than the record — the **Agent gateway**, stamping what a
    call spent into the **Agent feed** without having to know which tools
    spend anything. A result that carries no cost fields costs nothing to
    record, which is exactly the answer for every tool that spends no tokens.

    Args:
        result: Any tool result; anything but a mapping yields ``{}``.

    Returns:
        ``{"model", "input_tokens", "output_tokens", "cost_usd"}`` when the
        result carries all four, else ``{}`` — never a partial line, which
        would read as a cost of zero rather than as no cost at all.
    """
    if not isinstance(result, Mapping) or not all(
        field_name in result for field_name in COST_FIELDS
    ):
        return {}
    return {field_name: result[field_name] for field_name in COST_FIELDS}


def _as_str(value: object, default: str = "") -> str:
    """Coerce a JSON value to ``str``, falling back to ``default`` on ``None``."""
    return default if value is None else str(value)


def _as_int(value: object, default: int = 0) -> int:
    """Coerce a JSON value to ``int``, falling back to ``default`` on junk."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce a JSON value to ``float``, falling back to ``default`` on junk."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool) -> bool:
    """Return ``value`` if it is a bool, else ``default`` (defensive parse)."""
    return value if isinstance(value, bool) else default


def _as_str_list(value: object) -> list[str]:
    """Return ``value`` as a list of strings, or ``[]`` when it is not a list."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _as_attachments(value: object) -> list[dict[str, Any]]:
    """Return ``value`` as a list of ``{"path", "comment"}`` dicts.

    Args:
        value: Any parsed JSON value; anything but a list of mappings with a
            non-empty ``path`` is dropped rather than raised on — a mangled
            attachment must cost the entry its file, never its approval.

    Returns:
        The attachments, in order.
    """
    if not isinstance(value, list):
        return []
    attachments: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        path = _as_str(item.get("path"))
        if not path:
            continue
        attachments.append({"path": path, "comment": _as_str(item.get("comment"))})
    return attachments


def manifest_from_run(run: Any) -> dict[str, Any]:
    """Return manifest-shaped facts built from one recorded run.

    The one place a ``RunRecord`` becomes the dict every renderer here reads,
    so a draft, a manual export and an auto-published run all describe a run
    with the same words. Duck-typed on the record rather than importing it,
    because this module sits below the manager that owns it.

    Args:
        run: A ``RunRecord``-shaped object.

    Returns:
        The manifest-shaped facts.
    """
    return {
        "run_id": getattr(run, "run_id", ""),
        "procedure": getattr(run, "procedure", ""),
        "kind": getattr(run, "kind", ""),
        "params": dict(getattr(run, "params", {}) or {}),
        "data_file": getattr(run, "data_file", ""),
        "started_utc": getattr(run, "started_utc", ""),
        "finished_utc": getattr(run, "finished_utc", ""),
        "status": getattr(run, "status", ""),
        "reason": getattr(run, "reason", ""),
    }


def _scalar(value: object) -> str:
    """Render one fact for the prompt, deterministically.

    Args:
        value: Any JSON-safe value from a manifest, a statistic, or a
            snapshot.

    Returns:
        Its text form: ``repr`` for a float (so no precision is lost or
        invented), sorted ``key=value`` pairs for a mapping, comma-joined
        items for a sequence, ``str`` otherwise, and ``"none"`` for ``None``.
    """
    if value is None:
        return "none"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={_scalar(value[key])}" for key in sorted(value, key=str))
    if isinstance(value, (list, tuple)):
        return ", ".join(_scalar(item) for item in value)
    return str(value)


def _lines(pairs: list[tuple[str, object]]) -> list[str]:
    """Render ``(label, value)`` pairs as ``label: value`` prompt lines.

    Args:
        pairs: Ordered label/value pairs.

    Returns:
        One line per pair.
    """
    return [f"{label}: {_scalar(value)}" for label, value in pairs]


def render_draft_prompt(request: DraftRequest) -> str:
    """Render the user half of the draft prompt — the run's facts, in order.

    Deterministic by construction: every mapping is walked in sorted key
    order, floats go through ``repr``, and nothing is read from the clock, the
    filesystem or the environment. See the draft prompt standard at the top of
    this module.

    Args:
        request: The facts to draft from.

    Returns:
        The prompt text.
    """
    manifest = dict(request.manifest)
    blocks: list[str] = ["RUN", *_lines(
        [
            ("run_id", request.run_id or manifest.get("run_id")),
            ("experiment_id", request.experiment_id),
            ("experiment_title", request.experiment_title),
            ("procedure", manifest.get("procedure")),
            ("kind", manifest.get("kind")),
            ("started_utc", manifest.get("started_utc")),
            ("finished_utc", manifest.get("finished_utc")),
            ("status", manifest.get("status")),
            ("reason", manifest.get("reason")),
            ("entry_template", request.template_id),
        ]
    )]

    params = manifest.get("params")
    params = params if isinstance(params, Mapping) else {}
    blocks.append("")
    blocks.append("PARAMETERS")
    blocks.extend(
        _lines([(str(key), params[key]) for key in sorted(params, key=str)])
        or ["none recorded"]
    )

    blocks.append("")
    blocks.append("COLUMN STATISTICS")
    stat_lines: list[str] = []
    for column in sorted(request.stats, key=str):
        summary = request.stats[column]
        if not isinstance(summary, Mapping):
            continue
        fields = ("count", "min", "max", "mean", "std", "first", "last")
        rendered = ", ".join(f"{name}={_scalar(summary.get(name))}" for name in fields)
        stat_lines.append(f"{column}: {rendered}")
    blocks.extend(stat_lines or ["none available"])

    blocks.append("")
    blocks.append("STATION")
    blocks.extend(_lines([("setup", request.station.get("setup"))]))
    instruments = request.station.get("instruments")
    declared = [item for item in instruments or [] if isinstance(item, Mapping)]
    for item in sorted(declared, key=lambda entry: str(entry.get("name", ""))):
        blocks.append(
            f"instrument: {item.get('name', '')} "
            f"(kind={item.get('kind', '')}, class={item.get('vi_class', '')}, "
            f"availability={_scalar(item.get('availability'))})"
        )

    blocks.append("")
    blocks.append("STATE AT RUN END")
    status = request.status
    blocks.extend(
        _lines(
            [
                ("state", status.get("state")),
                ("faulted_instruments", sorted(status.get("vi_faults") or {})),
                ("held_instruments", status.get("held_vi_names")),
                ("offline_instruments", sorted(status.get("offline_reason") or {})),
            ]
        )
        if status
        else ["no status snapshot was available"]
    )

    blocks.append("")
    blocks.append("OPERATOR NOTE")
    blocks.append(request.operator_note or "none")
    return "\n".join(blocks)


def prompt_digest(system: str, user: str) -> str:
    """Return the SHA-256 fingerprint of one exact prompt.

    Covers both halves, so a changed system prompt is as visible as a changed
    fact — the digest answers "was this drafted from the same question?", not
    merely "from the same run?".

    Args:
        system: The system half.
        user: The user half.

    Returns:
        The hex digest.
    """
    payload = f"{system}\n\n{user}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_completion(text: str) -> tuple[str, str]:
    """Split one completion into its title and its prose.

    Tolerant by design (see the draft prompt standard): a completion missing
    either marker still yields a usable draft rather than an error, because a
    model that ignored the shape has usually still written the summary.

    Args:
        text: The completion text.

    Returns:
        ``(title, summary)``. ``title`` is ``""`` when no ``TITLE:`` line was
        found, and ``summary`` is the whole text when no ``SUMMARY:`` marker
        was.
    """
    title = ""
    body_lines: list[str] = []
    seen_summary = False
    for line in text.splitlines():
        stripped = line.strip()
        if not title and stripped.upper().startswith(TITLE_MARKER):
            title = stripped[len(TITLE_MARKER):].strip()
            continue
        if not seen_summary and stripped.upper().startswith(SUMMARY_MARKER):
            seen_summary = True
            remainder = stripped[len(SUMMARY_MARKER):].strip()
            if remainder:
                body_lines.append(remainder)
            continue
        body_lines.append(line)
    return title, "\n".join(body_lines).strip()


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    prices: Mapping[str, Mapping[str, float]],
) -> float:
    """Return what one completion cost, at the settings' price table.

    Args:
        model: The model that answered.
        input_tokens: Tokens the prompt consumed.
        output_tokens: Tokens the completion generated.
        prices: ``{model: {"input": usd_per_mtok, "output": usd_per_mtok}}``
            from ``AssistantSettings.prices``.

    Returns:
        The cost in US dollars, or ``0.0`` with a WARNING when the model has
        no row — an unpriced model reports no cost rather than a guessed one.
    """
    row = prices.get(model)
    if not isinstance(row, Mapping):
        logger.warning(
            "No price row for model %r — reporting a draft cost of 0.0 USD", model
        )
        return 0.0
    per_million = float(row.get("input", 0.0)) * input_tokens + float(
        row.get("output", 0.0)
    ) * output_tokens
    return per_million / 1_000_000.0


def draft_entry(
    request: DraftRequest,
    client: DraftClient,
    settings: AssistantSettings | None = None,
) -> DraftEntry:
    """Draft one notebook entry for one finished run.

    The whole drafting path in one call: render the prompt, ask the model
    once, parse the answer tolerantly, and render the body with the drafted
    prose above the run's own escaped fact tables. Publishes nothing and
    writes nothing.

    Args:
        request: The facts to draft from.
        client: The **Draft client** to ask.
        settings: The assistant settings supplying the token cap and the
            price table. ``None`` uses the defaults.

    Returns:
        The ``DraftEntry``, carrying the prompt digest and the cost line.

    Raises:
        ElnError: The model could not be reached, or refused.
    """
    resolved = settings or AssistantSettings()
    user_prompt = render_draft_prompt(request)
    digest = prompt_digest(DRAFT_SYSTEM_PROMPT, user_prompt)
    completion = client.complete(
        DRAFT_SYSTEM_PROMPT, user_prompt, resolved.max_tokens
    )
    title, summary = parse_completion(completion.text)
    body = render_draft_body(
        summary,
        request.manifest,
        stats=request.stats,
        experiment_id=request.experiment_id,
        experiment_title=request.experiment_title,
        setup=request.setup,
        station=request.station,
        data_path=request.data_path,
        operator_note=request.operator_note,
    )
    entry = DraftEntry(
        title=title or render_run_title(request.manifest, request.experiment_title),
        body_html=body,
        tags=_draft_tags(request),
        model=completion.model or resolved.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cost_usd=cost_usd(
            completion.model or resolved.model,
            completion.input_tokens,
            completion.output_tokens,
            resolved.prices,
        ),
        prompt_digest=digest,
    )
    logger.info(
        "Drafted an ELN entry for run %s (%d in / %d out tokens, %.4f USD)",
        request.run_id,
        entry.input_tokens,
        entry.output_tokens,
        entry.cost_usd,
    )
    return entry


def _draft_tags(request: DraftRequest) -> list[str]:
    """Return the tags a drafted entry carries, deterministically.

    The notebook's own standing tags are the publisher's business (they come
    from the settings at enqueue time); these are the two a draft adds about
    itself, so an entry that began as a machine draft stays findable.

    Args:
        request: The facts being drafted from.

    Returns:
        ``[DRAFT_TAG]`` plus the procedure name when the run recorded one.
    """
    procedure = str(request.manifest.get("procedure") or "")
    return [DRAFT_TAG, procedure] if procedure else [DRAFT_TAG]


class FakeDraftClient:
    """An in-memory **Draft client** — the workhorse of every drafting test.

    The ``sim_`` rule applied to the model: it answers from a canned script,
    records every prompt it was given, and models the failure mode that
    matters (an unreachable model) as an ``ElnError``. No network, no SDK, no
    key.

    Attributes:
        calls: One ``(system, user, max_tokens)`` tuple per completion asked
            for, so a test can assert on the exact prompt that was sent.
        offline: When ``True``, every call raises ``ElnError``.
    """

    def __init__(
        self,
        text: str = "",
        model: str = "fake-model",
        input_tokens: int = 1000,
        output_tokens: int = 200,
        offline: bool = False,
    ) -> None:
        """Build the fake with the answer it will always give.

        Args:
            text: The completion text to return. Empty yields a minimal
                well-formed answer in the standard's own shape.
            model: The model id to report.
            input_tokens: Prompt tokens to report.
            output_tokens: Completion tokens to report.
            offline: Start unreachable.
        """
        self._text = text or (
            f"{TITLE_MARKER} Drafted run\n{SUMMARY_MARKER}\nThe run completed."
        )
        self._model = model
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.offline = offline
        self.calls: list[tuple[str, str, int]] = []

    def complete(self, system: str, user: str, max_tokens: int) -> CompletionResult:
        """Answer from the canned script, recording the prompt.

        Args:
            system: The system half of the prompt.
            user: The user half.
            max_tokens: Cap on the generated tokens.

        Returns:
            The canned completion and its declared token counts.

        Raises:
            ElnError: When ``offline`` is set.
        """
        self.calls.append((system, user, max_tokens))
        if self.offline:
            raise ElnError("the fake draft client is offline")
        return CompletionResult(
            text=self._text,
            model=self._model,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )


class AnthropicDraftClient:
    """The real **Draft client**: one message to the vendor's Messages API.

    Its SDK is an optional dependency, declared as the ``assistant`` extra and
    imported lazily inside ``__init__`` — so a checkout without it imports
    this module and runs every test unchanged, and an installation that turns
    drafting on without installing it gets one clear ``ElnError`` naming the
    command that fixes it.
    """

    def __init__(self, settings: AssistantSettings | None = None) -> None:
        """Build the vendor client from the assistant settings.

        Args:
            settings: The assistant settings — the model, the token cap, and
                the API key. ``None`` uses the defaults, whose empty key means
                the SDK resolves credentials from the environment itself.

        Raises:
            ElnError: The vendor SDK is not installed, or the client could not
                be constructed (a malformed base URL, an unusable key).
        """
        try:
            import anthropic
        except ImportError as error:  # the optional extra is not installed
            raise ElnError(
                "LLM drafting needs the 'anthropic' package, which is an "
                "optional dependency: install it with "
                "`pip install i2as[assistant]`."
            ) from error

        self._settings = settings or AssistantSettings()
        try:
            self._client = (
                anthropic.Anthropic(api_key=self._settings.api_key)
                if self._settings.api_key
                else anthropic.Anthropic()
            )
        except Exception as error:  # the SDK raises its own types
            raise ElnError(f"could not build the drafting client: {error}") from error
        logger.info("Drafting client ready (model=%s)", self._settings.model)

    @property
    def settings(self) -> AssistantSettings:
        """The assistant settings this client was built with."""
        return self._settings

    def complete(self, system: str, user: str, max_tokens: int) -> CompletionResult:
        """Ask the model once and return its answer with the token counts.

        Args:
            system: The system half of the prompt.
            user: The user half — the run's facts.
            max_tokens: Cap on the generated tokens.

        Returns:
            The completion, the model that actually answered, and the tokens
            each half consumed.

        Raises:
            ElnError: Any vendor failure — unreachable, refused, rate-limited
                or malformed — mapped to this package's one exception type so
                a caller has exactly one thing to catch.
        """
        try:
            message = self._client.messages.create(
                model=self._settings.model,
                max_tokens=max(int(max_tokens), 1),
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as error:  # the SDK raises its own types
            raise ElnError(f"the drafting model could not be reached: {error}") from error

        text = "".join(
            str(getattr(block, "text", ""))
            for block in getattr(message, "content", [])
            if getattr(block, "type", "") == "text"
        )
        usage = getattr(message, "usage", None)
        return CompletionResult(
            text=text,
            model=_as_str(getattr(message, "model", ""), self._settings.model),
            input_tokens=_as_int(getattr(usage, "input_tokens", 0)),
            output_tokens=_as_int(getattr(usage, "output_tokens", 0)),
        )
