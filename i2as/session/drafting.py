"""LLM drafting — one model's prose about one finished run, kept as an analysis bundle.

**A draft is an analysis product, not a notebook entry.** The assistant reads
the facts a run recorded (its procedure and parameters, per-column statistics,
the station, the engine's state at run end), is asked once for a short
summary, and what comes back is written as an **analysis bundle** of kind
``draft`` in the run's analysis folder: a ``summary.md`` plus the sealed
``bundle.json`` naming the model, the prompt digest and the cost. It publishes
nothing and knows no notebook: the bundle is selected, previewed and published
exactly like a recipe's.

The draft prompt standard
-------------------------

1. **Two halves, both plain text.** ``DRAFT_SYSTEM_PROMPT`` is a constant and
   ``render_draft_prompt(request)`` renders the facts; neither contains markup,
   a URL, or anything read from the environment.
2. **Deterministic.** The same ``DraftRequest`` renders byte-identical text,
   and ``prompt_digest`` is the SHA-256 of both halves, so a changed prompt is
   visible as a changed digest in the bundle that came out of it.
3. **Facts in a fixed order** under bare uppercase headings: ``RUN``,
   ``PARAMETERS``, ``COLUMN STATISTICS``, ``STATION``, ``STATE AT RUN END``,
   ``OPERATOR NOTE``.
4. **The answer shape is two markers**, ``TITLE:`` and ``SUMMARY:``, parsed
   tolerantly: a completion missing either still yields a usable draft.

The **Draft client** contract is one method, ``complete(system, user,
max_tokens) -> CompletionResult``; ``FakeDraftClient`` answers every test, and
``AnthropicDraftClient`` is the real one (the optional ``assistant`` extra,
imported lazily). Any failure is a ``DraftError``.

**The key is never logged and never written into a bundle.** It is the
user's own, read from the credential store (``i2as.session.credentials``)
when a client is built, and passed to the vendor SDK and nowhere else.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from i2as.analysis.bundle import PRODUCER_DRAFT, Bundle, BundleInput, Producer, seal_bundle

logger = logging.getLogger(__name__)

_REDACTED = "***"

#: The model a draft is written by when the settings file names none. Chosen
#: as the vendor's current general-purpose default; a setup that wants a
#: cheaper or a newer one sets ``assistant.model`` and, if it is not in the
#: price table below, ``assistant.prices``.
DEFAULT_ASSISTANT_MODEL = "claude-opus-5"

#: Largest number of tokens a single draft may generate. A drafted summary is
#: a handful of paragraphs; the cap is what stops a runaway completion from
#: costing an unbounded amount. Deliberately several times what the prose
#: itself needs: on the current default model the vendor's reasoning is on by
#: default and is generated — and billed — against this same cap, so a cap
#: sized for the prose alone would truncate the draft rather than bound it.
DEFAULT_ASSISTANT_MAX_TOKENS = 8192

#: List price per one million tokens, per model, in US dollars — the source
#: the reported ``cost_usd`` of a draft is computed from.
#:
#: Source: Anthropic's published API pricing (https://www.anthropic.com/pricing),
#: as of 2026-06-24. These are LIST prices: an account on partner or negotiated
#: rates overrides the whole table from the settings file's ``assistant.prices``,
#: which is why the numbers live in settings rather than in the drafting code.
#: A model with no row here reports ``cost_usd`` of 0.0 and logs a WARNING —
#: never a guessed price.
DEFAULT_MODEL_PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5": {"input": 5.0, "output": 25.0},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}


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


def _as_prices(value: object) -> dict[str, dict[str, float]]:
    """Coerce a JSON value to a per-model price table, dropping malformed rows.

    Args:
        value: Any parsed JSON value. A non-mapping, or a row that names
            neither an input nor an output price, degrades to the default
            table and a skipped row respectively — a mangled price must never
            stop a draft, it must only stop the draft claiming a cost.

    Returns:
        ``{model: {"input": usd_per_mtok, "output": usd_per_mtok}}``.
    """
    if not isinstance(value, dict):
        return {model: dict(row) for model, row in DEFAULT_MODEL_PRICES.items()}
    table: dict[str, dict[str, float]] = {}
    for model, row in value.items():
        if not isinstance(row, dict):
            logger.warning("Ignoring malformed price row for model %r", model)
            continue
        table[str(model)] = {
            "input": _as_float(row.get("input"), 0.0),
            "output": _as_float(row.get("output"), 0.0),
        }
    return table


@dataclass(frozen=True)
class AssistantSettings:
    """The drafting assistant's settings — a section of each user's profile.

    Every field has a working default and the record parses tolerantly. The
    key is NOT stored with these settings: it lives in the credential store
    and is put into ``api_key`` only in memory, when a client is built;
    ``repr()`` and ``to_dict()`` redact it.

    Attributes:
        enabled: Master switch for drafting. ``False`` (the default) means no
            **Draft client** is built and no model is ever called.
        model: The model id a draft is written by.
        api_key: The API key, redacted from ``repr``/``to_dict``. Empty means
            "let the vendor SDK resolve credentials from the environment",
            which is how an installation keeps the key out of every file.
        max_tokens: Cap on one draft's generated tokens.
        prices: ``{model: {"input": usd_per_mtok, "output": usd_per_mtok}}``,
            the table a draft's ``cost_usd`` is computed from. Defaults to
            ``DEFAULT_MODEL_PRICES``; an account on other rates replaces it.
    """

    enabled: bool = False
    model: str = DEFAULT_ASSISTANT_MODEL
    api_key: str = ""
    max_tokens: int = DEFAULT_ASSISTANT_MAX_TOKENS
    prices: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            model: dict(row) for model, row in DEFAULT_MODEL_PRICES.items()
        }
    )

    def __repr__(self) -> str:
        """Return a repr with the API key redacted (never log the key)."""
        return (
            f"AssistantSettings(enabled={self.enabled!r}, model={self.model!r}, "
            f"api_key={_REDACTED if self.api_key else ''!r}, "
            f"max_tokens={self.max_tokens!r}, prices={sorted(self.prices)!r})"
        )

    def to_dict(self, include_secret: bool = False) -> dict[str, Any]:
        """Return a JSON-safe dict representation.

        Args:
            include_secret: When ``True``, the real ``api_key`` is included —
                for writing the settings file back, never for logging.

        Returns:
            A JSON-serialisable dict of every setting.
        """
        return {
            "enabled": self.enabled,
            "model": self.model,
            "api_key": (
                self.api_key if include_secret else (_REDACTED if self.api_key else "")
            ),
            "max_tokens": self.max_tokens,
            "prices": {model: dict(row) for model, row in self.prices.items()},
        }

    @classmethod
    def from_dict(cls, data: object) -> AssistantSettings:
        """Build ``AssistantSettings`` from a parsed dict, tolerating bad input.

        Args:
            data: Any parsed JSON value; junk degrades to defaults.

        Returns:
            The settings record. A redacted ``api_key`` read back from a
            ``to_dict()`` dump is treated as "no key".
        """
        if not isinstance(data, dict):
            return cls()
        defaults = cls()
        api_key = _as_str(data.get("api_key"))
        if api_key == _REDACTED:
            api_key = ""
        return cls(
            enabled=_as_bool(data.get("enabled"), defaults.enabled),
            model=_as_str(data.get("model"), defaults.model) or defaults.model,
            api_key=api_key,
            max_tokens=_as_int(data.get("max_tokens"), defaults.max_tokens),
            prices=_as_prices(data.get("prices")),
        )


class DraftError(RuntimeError):
    """The drafting model could not be reached, refused, or is not installed."""


#: Marker the completion puts its one-line title on.
TITLE_MARKER = "TITLE:"

#: Marker the completion puts its prose after.
SUMMARY_MARKER = "SUMMARY:"

#: The four fields a draft reports what it cost in — the **cost line**.
COST_FIELDS: tuple[str, ...] = ("model", "input_tokens", "output_tokens", "cost_usd")

#: The tag every draft bundle carries.
DRAFT_TAG = "draft"

#: The system half of the draft prompt standard. A constant, so it is part of
#: every draft's ``prompt_digest`` and a change to it is visible in the record.
DRAFT_SYSTEM_PROMPT = (
    "You are drafting a short summary of one cryostat "
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
    "- A human reviews this summary before it is used. Write for that "
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
    """The one model call drafting makes — the **Draft client** contract.

    One synchronous method, no streaming, no tools, no conversation: a draft
    is one question and one answer. Any failure is a ``DraftError``.
    """

    def complete(self, system: str, user: str, max_tokens: int) -> CompletionResult:
        """Answer one prompt.

        Raises:
            DraftError: The model could not be reached, or refused.
        """
        ...


@dataclass(frozen=True)
class DraftRequest:
    """Everything one draft is written from — the facts, and nothing else.

    In-memory only: never persisted, because it
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
    operator_note: str = ""


@dataclass(frozen=True)
class Draft:
    """One model's answer about one run, and what it cost.

    Attributes:
        title: A one-line title for the run.
        summary: The prose, as plain paragraphs.
        tags: Tags the draft proposes (``DRAFT_TAG`` and the procedure).
        model: The model that answered, as the vendor reported it.
        input_tokens: Tokens the prompt consumed.
        output_tokens: Tokens the completion generated.
        cost_usd: What those tokens cost at the price table, or ``0.0`` when
            the model has no price row (never a guess).
        prompt_digest: SHA-256 of the exact prompt.
    """

    title: str = ""
    summary: str = ""
    tags: tuple[str, ...] = ()
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    prompt_digest: str = ""

    def cost_line(self) -> dict[str, Any]:
        """Return the four cost fields the **Agent feed** records."""
        return {name: getattr(self, name) for name in COST_FIELDS}


def cost_line(result: object) -> dict[str, Any]:
    """Return the cost line a tool result carries, or ``{}``.

    The read side of ``Draft.cost_line()``, for a caller holding the
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


def draft_summary(
    request: DraftRequest,
    client: DraftClient,
    settings: AssistantSettings | None = None,
) -> Draft:
    """Ask the model once for a summary of one run.

    Args:
        request: The facts to draft from.
        client: The **Draft client** to ask.
        settings: The token cap and the price table; ``None`` for defaults.

    Returns:
        The draft, carrying its prompt digest and its cost line.

    Raises:
        DraftError: The model could not be reached, or refused.
    """
    resolved = settings or AssistantSettings()
    user_prompt = render_draft_prompt(request)
    digest = prompt_digest(DRAFT_SYSTEM_PROMPT, user_prompt)
    completion = client.complete(DRAFT_SYSTEM_PROMPT, user_prompt, resolved.max_tokens)
    title, summary = parse_completion(completion.text)
    procedure = str(request.manifest.get("procedure") or "")
    model = completion.model or resolved.model
    draft = Draft(
        title=title or f"{procedure or 'Run'} {request.run_id}".strip(),
        summary=summary,
        tags=(DRAFT_TAG, procedure) if procedure else (DRAFT_TAG,),
        model=model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cost_usd=cost_usd(model, completion.input_tokens, completion.output_tokens, resolved.prices),
        prompt_digest=digest,
    )
    logger.info(
        "Drafted a summary of run %s (%d in / %d out tokens, %.4f USD)",
        request.run_id,
        draft.input_tokens,
        draft.output_tokens,
        draft.cost_usd,
    )
    return draft


def write_draft_bundle(
    folder: str | Path,
    draft: Draft,
    *,
    bundle_id: str,
    experiment_id: str,
    run: Any,
    actor: str = "",
) -> Bundle:
    """Write one draft as an analysis bundle and seal it.

    The prose goes into ``summary.md`` and into the bundle's ``summary``
    paragraphs; the model, prompt digest and cost into its producer and
    options. Written by the application (a draft is not worker output), then
    sealed exactly as a worker's folder is.

    Args:
        folder: The new bundle folder (created).
        draft: The draft.
        bundle_id: Its id.
        experiment_id: The owning experiment.
        run: The ``RunRecord`` drafted about.
        actor: Who asked.

    Returns:
        The sealed bundle.

    Raises:
        OSError: The folder could not be written.
    """
    directory = Path(folder)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.md").write_text(draft.summary + "\n", encoding="utf-8")
    paragraphs = [part.strip() for part in draft.summary.split("\n\n") if part.strip()]
    claims = {
        "status": "ok",
        "summary": paragraphs,
        "tags": list(draft.tags),
        "options": {
            "title": draft.title,
            "model": draft.model,
            "input_tokens": draft.input_tokens,
            "output_tokens": draft.output_tokens,
            "cost_usd": draft.cost_usd,
        },
    }
    run_id = str(getattr(run, "run_id", ""))
    return seal_bundle(
        directory,
        claims,
        bundle_id=bundle_id,
        experiment_id=experiment_id,
        run_ids=(run_id,),
        producer=Producer(
            kind=PRODUCER_DRAFT,
            name=draft.model,
            digest=draft.prompt_digest,
            actor=actor,
            model=draft.model,
            prompt_digest=draft.prompt_digest,
        ),
        inputs=(
            BundleInput(
                run_id=run_id,
                data_file=str(getattr(run, "data_file", "") or ""),
                params_digest=str(getattr(run, "params_digest", "") or ""),
            ),
        ),
    )


class FakeDraftClient:
    """An in-memory **Draft client** — the workhorse of every drafting test.

    The ``sim_`` rule applied to the model: it answers from a canned script,
    records every prompt it was given, and models the failure mode that
    matters (an unreachable model) as an ``DraftError``. No network, no SDK, no
    key.

    Attributes:
        calls: One ``(system, user, max_tokens)`` tuple per completion asked
            for, so a test can assert on the exact prompt that was sent.
        offline: When ``True``, every call raises ``DraftError``.
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
            DraftError: When ``offline`` is set.
        """
        self.calls.append((system, user, max_tokens))
        if self.offline:
            raise DraftError("the fake draft client is offline")
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
    drafting on without installing it gets one clear ``DraftError`` naming the
    command that fixes it.
    """

    def __init__(self, settings: AssistantSettings | None = None) -> None:
        """Build the vendor client from the assistant settings.

        Args:
            settings: The assistant settings — the model, the token cap, and
                the API key. ``None`` uses the defaults, whose empty key means
                the SDK resolves credentials from the environment itself.

        Raises:
            DraftError: The vendor SDK is not installed, or the client could not
                be constructed (a malformed base URL, an unusable key).
        """
        try:
            import anthropic
        except ImportError as error:  # the optional extra is not installed
            raise DraftError(
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
            raise DraftError(f"could not build the drafting client: {error}") from error
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
            DraftError: Any vendor failure — unreachable, refused, rate-limited
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
            raise DraftError(f"the drafting model could not be reached: {error}") from error

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
