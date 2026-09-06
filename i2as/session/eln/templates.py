"""Body renderers — what I2AS writes *into* an ELN entry.

Two cooperating levels of "template" exist, and this module is the second:

1. **Backend-side entry templates** (categories, team defaults) stay the
   lab's: an adapter creates an entry *from* a template id chosen in settings,
   so an existing eLabFTW template keeps working untouched.
2. **I2AS-side body renderers** — this module. Plain Python functions, no
   template-language dependency, producing a deterministic, self-contained
   HTML fragment: no ``<script>``, no ``<link>``, no ``<img>``, no external
   URL of any kind, so the entry renders identically in the notebook, in an
   export, and in a test snapshot. Every value is HTML-escaped.

The unit rendered is one **run**: the run id and its experiment, the procedure
and its parameters, the setup (config identity and instruments) it ran on, the
timestamps, the outcome, and where the data file lives. That is exactly the
content of the Orchestrator's run manifest plus the session layer's experiment
context, which is why the publisher can render a job at the moment the run
finishes and queue the finished text — an outbox job never has to re-render
later against state that has since moved on.

Three bodies exist, built from the same shared row builders so a run reads
identically in all of them: ``render_run_body()`` for a published run,
``render_draft_body()`` for a **draft entry**, which puts a drafted summary
above the facts it was drafted from, and ``render_analysed_body()`` for an
**analysed entry**, which puts one recipe's **Analysis report** — its prose,
its derived values, its figures and its tables — above a compact provenance
block naming the run, the data file and the recipe that produced the numbers.
The drafted prose is model output and the analysed prose is user-recipe
output; both are escaped like any other value — ``render_prose_section()`` is
the one door untrusted text comes through, and it can introduce no markup at
all.

An analysed entry names its figures by file name and never embeds one: a
figure travels as an **Outbox** attachment, so the body stays self-contained
and renders identically in the notebook, in an export, and in a test
snapshot. The run's full fact tables are appended only when the report asks
for them (``include_fact_tables``) — the point of the analysis stage is that
what reaches the notebook is the result, not the raw facts.
"""

from __future__ import annotations

import html
import logging
from collections.abc import Mapping
from typing import Any

from i2as.analysis.report import AnalysisReport

logger = logging.getLogger(__name__)

_MAX_CELL_CHARS = 400


def _text(value: object) -> str:
    """Return an HTML-escaped, length-capped rendering of one scalar value.

    Args:
        value: Any manifest value.

    Returns:
        Escaped text; ``"—"`` for ``None``/empty, truncated with an ellipsis
        beyond ``_MAX_CELL_CHARS`` so one pathological parameter cannot
        produce a megabyte of entry body.
    """
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        raw = repr(value)
    elif isinstance(value, (list, tuple)):
        raw = ", ".join(str(item) for item in value)
    elif isinstance(value, Mapping):
        raw = ", ".join(f"{key}={value[key]}" for key in sorted(value, key=str))
    else:
        raw = str(value)
    if len(raw) > _MAX_CELL_CHARS:
        raw = raw[:_MAX_CELL_CHARS] + "…"
    return html.escape(raw)


def _rows(pairs: list[tuple[str, object]]) -> str:
    """Render ``(label, value)`` pairs as HTML table rows.

    Args:
        pairs: Ordered label/value pairs.

    Returns:
        The concatenated ``<tr>`` markup.
    """
    return "".join(
        f"<tr><th align='left'>{html.escape(str(label))}</th>"
        f"<td>{_text(value)}</td></tr>"
        for label, value in pairs
    )


def _table(caption: str, pairs: list[tuple[str, object]]) -> str:
    """Render a captioned two-column table, or an empty string when no rows.

    Args:
        caption: Section heading.
        pairs: Ordered label/value pairs; an empty list renders nothing.

    Returns:
        The ``<h3>`` + ``<table>`` markup.
    """
    if not pairs:
        return ""
    return (
        f"<h3>{html.escape(caption)}</h3>"
        f"<table border='1' cellpadding='4' cellspacing='0'>{_rows(pairs)}</table>"
    )


def render_run_title(
    manifest: Mapping[str, Any], experiment_title: str = ""
) -> str:
    """Render the ELN entry title for one run.

    Args:
        manifest: The run manifest (``procedure``, ``run_id``, ``started_utc``).
        experiment_title: The owning experiment's title, prefixed when given.

    Returns:
        A single-line plain-text title (not HTML — backends set titles as text).
    """
    procedure = str(manifest.get("procedure") or "Run")
    started = str(manifest.get("started_utc") or "")
    stem = f"{procedure} — {started}" if started else procedure
    return f"{experiment_title} — {stem}" if experiment_title else stem


def _run_rows(
    manifest: Mapping[str, Any], data_path: str = ""
) -> list[tuple[str, object]]:
    """Return the ``Run`` table's label/value pairs for one manifest.

    Shared by every body renderer here, so a run reads identically in a
    published entry and in a draft.

    Args:
        manifest: The run manifest.
        data_path: Where the run's data file lives, or ``""``.

    Returns:
        The ordered label/value pairs.
    """
    return [
        ("Run id", manifest.get("run_id")),
        ("Procedure", manifest.get("procedure")),
        ("Kind", manifest.get("kind")),
        ("Started (UTC)", manifest.get("started_utc")),
        ("Finished (UTC)", manifest.get("finished_utc")),
        ("Outcome", manifest.get("status")),
        ("Reason", manifest.get("reason")),
        ("Data file", data_path),
    ]


def _experiment_rows(
    experiment_id: str, experiment_title: str
) -> list[tuple[str, object]]:
    """Return the ``Experiment`` table's rows, or an empty list when unknown.

    Args:
        experiment_id: The owning experiment's id, or ``""``.
        experiment_title: The owning experiment's title, or ``""``.

    Returns:
        The ordered label/value pairs; empty when neither is known, which
        renders nothing at all.
    """
    if not (experiment_id or experiment_title):
        return []
    return [("Experiment id", experiment_id), ("Title", experiment_title)]


def _param_rows(manifest: Mapping[str, Any]) -> list[tuple[str, object]]:
    """Return the ``Parameters`` table's rows, sorted by key.

    Args:
        manifest: The run manifest, whose ``params`` are rendered.

    Returns:
        The ordered label/value pairs; empty when the run declared none.
    """
    params = manifest.get("params")
    if not isinstance(params, Mapping) or not params:
        return []
    return [(str(key), params[key]) for key in sorted(params, key=str)]


def _setup_rows(setup: Mapping[str, Any] | None) -> list[tuple[str, object]]:
    """Return the ``Setup`` table's rows, sorted by instrument name.

    Args:
        setup: The setup tier of ``ExperimentManager.experiment_context()``
            (``config_name`` plus per-VI ``instruments``), or ``None``.

    Returns:
        The ordered label/value pairs.
    """
    setup_map = dict(setup or {})
    rows: list[tuple[str, object]] = []
    config_name = setup_map.get("config_name")
    if config_name:
        rows.append(("Config", config_name))
    instruments = setup_map.get("instruments")
    if isinstance(instruments, Mapping):
        for vi_name in sorted(instruments, key=str):
            rows.append((str(vi_name), instruments[vi_name]))
    return rows


def render_prose_section(caption: str, text: str) -> str:
    """Render free text as escaped paragraphs under a heading.

    The one door untrusted prose comes through: a drafted summary is model
    output, so it is escaped and split on blank lines into ``<p>`` elements
    rather than inserted as markup. Nothing here can introduce a tag, a link
    or a script into an entry body. A URL the prose happens to mention
    survives as inert text — escaping is what removes every way it could be
    fetched or followed, so the body still renders identically in the
    notebook, in an export, and in a test snapshot.

    Args:
        caption: Section heading.
        text: The prose; blank input renders nothing.

    Returns:
        The ``<h3>`` + ``<p>`` markup, or ``""`` when *text* is empty.
    """
    paragraphs = [block.strip() for block in text.strip().split("\n\n")]
    rendered = "".join(
        f"<p>{html.escape(block)}</p>" for block in paragraphs if block
    )
    if not rendered:
        return ""
    return f"<h3>{html.escape(str(caption))}</h3>{rendered}"


def render_stats_section(stats: Mapping[str, Mapping[str, Any]]) -> str:
    """Render per-column summary statistics as one table.

    Deterministic: columns and their statistics are emitted in sorted order,
    so the same statistics always render byte-identical HTML.

    Args:
        stats: ``{column: Stats.to_json()}`` — the NaN-aware summary
            ``core.data_reader.summary_stats()`` produces per column.

    Returns:
        The ``<h3>`` + ``<table>`` markup, or ``""`` when there is nothing to
        summarise.
    """
    columns = [name for name in sorted(stats, key=str) if isinstance(stats[name], Mapping)]
    if not columns:
        return ""
    fields = ("count", "min", "max", "mean", "std", "first", "last")
    header = "".join(f"<th align='left'>{html.escape(name)}</th>" for name in ("Column", *fields))
    rows = "".join(
        "<tr>"
        + f"<th align='left'>{html.escape(str(name))}</th>"
        + "".join(f"<td>{_text(stats[name].get(field))}</td>" for field in fields)
        + "</tr>"
        for name in columns
    )
    return (
        "<h3>Column statistics</h3>"
        f"<table border='1' cellpadding='4' cellspacing='0'>"
        f"<tr>{header}</tr>{rows}</table>"
    )


def render_run_body(
    manifest: Mapping[str, Any],
    experiment_id: str = "",
    experiment_title: str = "",
    setup: Mapping[str, Any] | None = None,
    data_path: str = "",
    findings: str = "",
) -> str:
    """Render one run's ELN entry body.

    Deterministic: parameters and instruments are emitted in sorted key order,
    so the same manifest always renders byte-identical HTML.

    Args:
        manifest: The Orchestrator's run manifest — ``run_id``, ``procedure``,
            ``kind``, ``params``, ``started_utc``, ``finished_utc``,
            ``status``, ``reason``.
        experiment_id: The owning experiment's id, or ``""``.
        experiment_title: The owning experiment's title, or ``""``.
        setup: The setup tier of ``ExperimentManager.experiment_context()``
            (``config_name`` plus per-VI ``instruments``), or ``None``.
        data_path: Where the run's data file lives, rendered as plain text
            (the file itself is attached separately by the publisher).
        findings: Optional free-text science notes to include verbatim
            (escaped).

    Returns:
        A self-contained HTML fragment.
    """
    parts: list[str] = [
        _table("Run", _run_rows(manifest, data_path)),
        _table("Experiment", _experiment_rows(experiment_id, experiment_title)),
        _table("Parameters", _param_rows(manifest)),
        _table("Setup", _setup_rows(setup)),
    ]
    if findings:
        parts.append(f"<h3>Findings</h3><p>{_text(findings)}</p>")
    parts.append("<p><em>Published by I2AS.</em></p>")
    return "".join(part for part in parts if part)


def render_draft_body(
    summary: str,
    manifest: Mapping[str, Any],
    stats: Mapping[str, Mapping[str, Any]] | None = None,
    experiment_id: str = "",
    experiment_title: str = "",
    setup: Mapping[str, Any] | None = None,
    station: Mapping[str, Any] | None = None,
    data_path: str = "",
    operator_note: str = "",
) -> str:
    """Render one **draft entry**'s body: a drafted summary over the facts.

    The same self-contained, escaped, deterministic HTML ``render_run_body()``
    produces, in the order a reviewer reads it: the drafted prose first, then
    the facts it was drafted from, so a human approving the entry can check
    every sentence against the tables below it. The summary is escaped like
    any other value — it is model output, not markup.

    Args:
        summary: The drafted prose. Escaped, never inserted as markup.
        manifest: The run manifest.
        stats: ``{column: Stats.to_json()}``, or ``None``.
        experiment_id: The owning experiment's id, or ``""``.
        experiment_title: The owning experiment's title, or ``""``.
        setup: The setup tier of ``ExperimentManager.experiment_context()``.
        station: The **Station info** snapshot as JSON (``setup`` and
            ``instruments``), or ``None`` — what the run actually ran on.
        data_path: Where the run's data file lives.
        operator_note: The operator's own note passed into the draft, if any.

    Returns:
        A self-contained HTML fragment.
    """
    parts: list[str] = [
        render_prose_section("Summary (drafted, unreviewed)", summary),
        _table("Run", _run_rows(manifest, data_path)),
        _table("Experiment", _experiment_rows(experiment_id, experiment_title)),
        _table("Parameters", _param_rows(manifest)),
        render_stats_section(dict(stats or {})),
        _table("Setup", _setup_rows(setup) + _station_rows(station)),
    ]
    if operator_note:
        parts.append(f"<h3>Operator note</h3><p>{_text(operator_note)}</p>")
    parts.append(
        "<p><em>Drafted by I2AS; reviewed by a human before publication.</em></p>"
    )
    return "".join(part for part in parts if part)


def _station_rows(station: Mapping[str, Any] | None) -> list[tuple[str, object]]:
    """Return the station-declaration rows appended to the ``Setup`` table.

    Args:
        station: The **Station info** snapshot as JSON, or ``None``.

    Returns:
        The setup name and one row per declared instrument, sorted by name.
    """
    snapshot = dict(station or {})
    rows: list[tuple[str, object]] = []
    name = snapshot.get("setup")
    if name:
        rows.append(("Setup", name))
    instruments = snapshot.get("instruments")
    if isinstance(instruments, (list, tuple)):
        declared = [item for item in instruments if isinstance(item, Mapping)]
        for item in sorted(declared, key=lambda entry: str(entry.get("name", ""))):
            rows.append(
                (
                    str(item.get("name", "")),
                    f"{item.get('kind', '') or 'instrument'} "
                    f"({item.get('vi_class', '') or 'unknown class'})",
                )
            )
    return rows


def render_run_metadata(
    manifest: Mapping[str, Any], experiment_id: str = "", data_path: str = ""
) -> dict[str, Any]:
    """Render the JSON-safe metadata block stored alongside the entry body.

    The machine-readable twin of the body: the same facts as flat scalars, so
    a later search in the notebook can filter by run id, procedure, or
    outcome without parsing HTML.

    Args:
        manifest: The run manifest.
        experiment_id: The owning experiment's id, or ``""``.
        data_path: Where the run's data file lives.

    Returns:
        A flat, JSON-safe dict.
    """
    return {
        "run_id": str(manifest.get("run_id") or ""),
        "experiment_id": experiment_id,
        "procedure": str(manifest.get("procedure") or ""),
        "kind": str(manifest.get("kind") or ""),
        "started_utc": str(manifest.get("started_utc") or ""),
        "finished_utc": str(manifest.get("finished_utc") or ""),
        "status": str(manifest.get("status") or ""),
        "reason": str(manifest.get("reason") or ""),
        "data_file": data_path,
        "source": "i2as",
    }


def _as_report(report: AnalysisReport | Mapping[str, Any]) -> AnalysisReport:
    """Return *report* as an ``AnalysisReport``, loading a dict tolerantly.

    Args:
        report: The report, or its ``report.json`` dict as the analysis
            worker wrote it.

    Returns:
        The report record; junk degrades to an empty ``ok`` report rather
        than raising, because a body must always render.
    """
    return report if isinstance(report, AnalysisReport) else AnalysisReport.from_dict(report)


def _file_name(path: str) -> str:
    """Return the file name of *path*, tolerating either separator.

    Args:
        path: A file path written on any platform, or ``""``.

    Returns:
        The last path segment, or ``""``.
    """
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def _results_table(results: tuple[Any, ...]) -> str:
    """Render the report's derived values as one table.

    Args:
        results: The report's ``ResultValue`` rows, in the order they should
            be listed.

    Returns:
        The ``<h3>`` + ``<table>`` markup, or ``""`` when there are none.
    """
    if not results:
        return ""
    header = "".join(
        f"<th align='left'>{html.escape(name)}</th>"
        for name in ("Quantity", "Value", "Uncertainty", "Note")
    )
    rows = "".join(
        "<tr>"
        f"<th align='left'>{_text(result.name)}</th>"
        f"<td>{_text(_value_with_unit(result.value, result.unit))}</td>"
        f"<td>{_text('± ' + repr(result.uncertainty) if result.uncertainty is not None else None)}</td>"
        f"<td>{_text(result.note)}</td>"
        "</tr>"
        for result in results
    )
    return (
        "<h3>Results</h3>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        f"<tr>{header}</tr>{rows}</table>"
    )


def _value_with_unit(value: object, unit: str) -> object:
    """Return one result's value with its unit appended, or the bare value.

    Args:
        value: The result's value (a JSON scalar).
        unit: The SI unit symbol, or ``""`` for a dimensionless or textual
            value.

    Returns:
        The value ready for ``_text()`` — display scaling (mK, µA) is a GUI
        concern and never happens here.
    """
    if not unit:
        return value
    rendered = repr(value) if isinstance(value, float) else str(value)
    return f"{rendered} {unit}"


def _figures_section(figures: tuple[Any, ...]) -> str:
    """Render the report's figures as a captioned list naming each file.

    Never an ``<img>``: the figure travels as an attachment on the entry, and
    the body stays self-contained, so it renders identically in the notebook,
    in an export, and in a test snapshot.

    Args:
        figures: The report's ``FigureRef`` rows, in order.

    Returns:
        The ``<h3>`` + ``<ul>`` markup, or ``""`` when there are none.
    """
    if not figures:
        return ""
    items = "".join(
        f"<li>{_text(figure.caption or figure.file)} "
        f"(attached as {_text(figure.file)})</li>"
        for figure in figures
    )
    return f"<h3>Figures</h3><ul>{items}</ul>"


def _spec_table(table: Any) -> str:
    """Render one of the report's ``TableSpec``s.

    Args:
        table: The ``TableSpec`` — a caption, column headings and bounded
            rows of JSON scalars.

    Returns:
        The ``<h3>`` + ``<table>`` markup, or ``""`` when the table is empty.
    """
    if not table.columns and not table.rows:
        return ""
    header = "".join(f"<th align='left'>{_text(name)}</th>" for name in table.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{_text(cell)}</td>" for cell in row) + "</tr>"
        for row in table.rows
    )
    truncated = "<p><em>Truncated to the report's row and column caps.</em></p>" if table.truncated else ""
    return (
        f"<h3>{_text(table.caption or 'Table')}</h3>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        f"<tr>{header}</tr>{rows}</table>{truncated}"
    )


def _warnings_section(warnings: tuple[str, ...]) -> str:
    """Render the report's non-fatal warnings as a list.

    Args:
        warnings: The warnings the recipe or the framework appended.

    Returns:
        The ``<h3>`` + ``<ul>`` markup, or ``""`` when there are none.
    """
    if not warnings:
        return ""
    items = "".join(f"<li>{_text(warning)}</li>" for warning in warnings)
    return f"<h3>Analysis warnings</h3><ul>{items}</ul>"


def _provenance_rows(
    report: AnalysisReport,
    manifest: Mapping[str, Any],
    data_path: str,
    experiment_id: str = "",
    experiment_title: str = "",
) -> list[tuple[str, object]]:
    """Return the compact provenance block's label/value pairs.

    What the notebook needs in order to find this result again: which
    experiment and run, which procedure, which parameters, which data file,
    and which recipe (by name and by source digest) turned the one into the
    other. The experiment rows lead, because an analysed entry that does not
    carry the fact tables would otherwise never name its experiment.

    Args:
        report: The analysis report.
        manifest: The run manifest the analysis ran against.
        data_path: Where the run's data file lives.
        experiment_id: The owning experiment's id, or ``""`` to omit the row.
        experiment_title: The owning experiment's title, or ``""``.

    Returns:
        The ordered label/value pairs.
    """
    return [
        *_experiment_rows(experiment_id, experiment_title),
        ("Run id", manifest.get("run_id")),
        ("Procedure", manifest.get("procedure")),
        ("Kind", manifest.get("kind")),
        ("Params digest", manifest.get("params_digest")),
        ("Started (UTC)", manifest.get("started_utc")),
        ("Finished (UTC)", manifest.get("finished_utc")),
        ("Outcome", manifest.get("status")),
        ("Data file", _file_name(data_path)),
        ("Recipe", report.recipe),
        ("Recipe digest", report.recipe_digest),
    ]


def render_analysed_title(
    report: AnalysisReport | Mapping[str, Any],
    facts: Mapping[str, Any],
    experiment_title: str = "",
) -> str:
    """Render an **analysed entry**'s title.

    The same index line a published run gets, so the notebook's list reads
    the same however an entry was produced — with a failed analysis said
    plainly in the title rather than hidden in the body.

    Args:
        report: The analysis report, or its ``report.json`` dict.
        facts: The run manifest (``procedure``, ``started_utc``).
        experiment_title: The owning experiment's title, prefixed when given.

    Returns:
        A single-line plain-text title (not HTML — backends set titles as
        text).
    """
    rendered = _as_report(report)
    stem = render_run_title(facts, experiment_title)
    return stem if rendered.ok else f"{stem} — analysis failed"


def render_analysed_body(
    report: AnalysisReport | Mapping[str, Any],
    facts: Mapping[str, Any],
    *,
    experiment_id: str = "",
    experiment_title: str = "",
    setup: Mapping[str, Any] | None = None,
    data_path: str = "",
    findings: str = "",
) -> str:
    """Render an **analysed entry**'s body: the result over its provenance.

    The order a physicist reads it in: what the run showed (the recipe's
    prose), the numbers it derived, the figures it saved, its own tables, any
    warnings, and last a compact provenance block naming the run, the data
    file and the recipe. The run's full fact tables follow ONLY when the
    report asked for them — the point of the analysis stage is that what
    reaches the notebook is the result, not the raw facts.

    A failed report renders its error under its own heading, so a failure is
    visible exactly where the result would have been, never silently empty.

    Deterministic, escaped and length-capped like every other body here, and
    self-contained: a figure is named, never embedded (it travels as an
    attachment on the entry).

    Args:
        report: The analysis report, or its ``report.json`` dict.
        facts: The run manifest — ``run_id``, ``procedure``, ``kind``,
            ``params``, ``params_digest``, ``started_utc``, ``finished_utc``,
            ``status``, ``reason``, and optionally ``summary_stats``
            (``{column: Stats.to_json()}``) for the appended fact tables.
        experiment_id: The owning experiment's id, or ``""``.
        experiment_title: The owning experiment's title, or ``""``.
        setup: The setup tier of ``ExperimentManager.experiment_context()``.
        data_path: Where the run's data file lives, rendered as plain text.
        findings: The experiment's free-text science notes, included verbatim
            (escaped).

    Returns:
        A self-contained HTML fragment.
    """
    rendered = _as_report(report)
    parts: list[str] = []
    if not rendered.ok:
        parts.append(
            render_prose_section(
                "Analysis failed", rendered.error or "The analysis produced no result."
            )
        )
    parts.append(render_prose_section("Summary", "\n\n".join(rendered.summary)))
    parts.append(_results_table(rendered.results))
    parts.append(_figures_section(rendered.figures))
    parts.extend(_spec_table(table) for table in rendered.tables)
    parts.append(_warnings_section(rendered.warnings))
    if findings:
        parts.append(f"<h3>Findings</h3><p>{_text(findings)}</p>")
    parts.append(
        _table(
            "Provenance",
            _provenance_rows(
                rendered, facts, data_path, experiment_id, experiment_title
            ),
        )
    )
    if rendered.include_fact_tables:
        stats = facts.get("summary_stats")
        parts.extend(
            [
                _table("Run", _run_rows(facts, data_path)),
                _table("Experiment", _experiment_rows(experiment_id, experiment_title)),
                _table("Parameters", _param_rows(facts)),
                render_stats_section(dict(stats) if isinstance(stats, Mapping) else {}),
                _table("Setup", _setup_rows(setup)),
            ]
        )
    parts.append(
        "<p><em>Analysed by I2AS; reviewed by a human before publication.</em></p>"
    )
    return "".join(part for part in parts if part)
