"""The analysis report — what a recipe returns and what a notebook entry is built from.

The **analysis report standard**: a recipe (``base.AnalysisRecipe``) turns one
finished run into exactly one ``AnalysisReport``, and everything downstream —
the notebook renderer, the procedure window's preview, the agent's
``read_analysis_report`` tool — reads that report and nothing else. Every
type here is a frozen dataclass whose ``to_dict()`` is JSON-safe and whose
``from_dict()`` loads tolerantly (missing keys default, junk values are
coerced or dropped, never raised on), because a report crosses a process
boundary as ``report.json`` and lives on for as long as the experiment folder
does.

``AnalysisSpec`` is the other half of that boundary: the request the analysis
worker (``python -m i2as.analysis run --spec <file>``) is started with.
It names the data file, the run manifest, the experiment and setup facts, the
recipe, where to look for experiment recipes and where to write. It never
carries a credential and never a live object.

Sizes are bounded here, not only at render time: a recipe that emits a
thousand-row table gets its table truncated (and a warning appended) before
the report is written, so a report is always small enough to preview, to
journal and to hand to a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Terminal status of a report whose recipe ran to completion.
REPORT_OK = "ok"

#: Terminal status of a report whose recipe raised, timed out, or could not
#: be loaded. Such a report still carries ``recipe``, ``error`` and whatever
#: ``warnings`` accumulated, so the failure is visible where the result would
#: have been.
REPORT_FAILED = "failed"

#: Name of the report file the worker writes into the run's analysis
#: directory.
REPORT_FILENAME = "report.json"

#: Name of the spec file the runner writes beside the report.
SPEC_FILENAME = "spec.json"

#: Sub-directory of an experiment's analysis folder holding the experiment's
#: own recipe scripts (``<experiment>/analysis/recipes/*.py``).
RECIPES_DIRNAME = "recipes"

#: Largest number of rows kept in one ``TableSpec``; longer tables are
#: truncated and the report gains a warning.
MAX_TABLE_ROWS = 50

#: Largest number of columns kept in one ``TableSpec``.
MAX_TABLE_COLUMNS = 12

#: Largest number of figures kept in one report.
MAX_FIGURES = 8

#: Largest number of result rows kept in one report.
MAX_RESULTS = 40

#: Largest number of summary paragraphs kept in one report.
MAX_SUMMARY_PARAGRAPHS = 12

#: Longest text kept for any single string field (a paragraph, a caption, a
#: note); longer text is cut and marked with an ellipsis.
MAX_TEXT_CHARS = 2000

#: The wildcard a recipe declares in ``procedures`` to serve every procedure.
ANY_PROCEDURE = "*"


def _text(value: object, limit: int = MAX_TEXT_CHARS) -> str:
    """Coerce ``value`` to a bounded string."""
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _texts(value: object, limit: int) -> list[str]:
    """Coerce ``value`` to a bounded list of bounded strings."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [_text(item) for item in list(value)[:limit] if item is not None]


def _json_scalar(value: object) -> Any:
    """Return ``value`` if it is a JSON scalar, else its ``str``.

    Floats that JSON cannot carry (NaN, ±inf) become ``None``, the same rule
    the data reader applies to its statistics.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    try:  # numpy scalars and the like
        item = value.item()  # type: ignore[attr-defined]
    except (AttributeError, ValueError, TypeError):
        return _text(value)
    return _json_scalar(item) if item is not value else _text(value)


def _optional_float(value: object) -> float | None:
    """Coerce to ``float`` or ``None`` (never raise)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _bool(value: object, default: bool) -> bool:
    """Return ``value`` if it is a bool, else ``default``."""
    return value if isinstance(value, bool) else default


def _dict(value: object) -> dict[str, Any]:
    """Return ``value`` as a dict with string keys, or ``{}``."""
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}


@dataclass(frozen=True)
class ResultValue:
    """One derived number (or short text) the analysis produced.

    Attributes:
        name: What it is (``"Critical field"``).
        value: The value — a JSON scalar; numbers stay numbers so a reader
            can format them, text stays text.
        unit: SI unit symbol, ``""`` for a dimensionless or textual value.
            Display scaling (mT, µA) is the renderer's business.
        uncertainty: One-sigma uncertainty in the same unit, or ``None``.
        note: One short sentence of context (method, caveat), or ``""``.
    """

    name: str
    value: Any = None
    unit: str = ""
    uncertainty: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {
            "name": self.name,
            "value": _json_scalar(self.value),
            "unit": self.unit,
            "uncertainty": self.uncertainty,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: object) -> ResultValue:
        """Load tolerantly from a dict; junk becomes defaults."""
        payload = _dict(data)
        return cls(
            name=_text(payload.get("name", "")),
            value=_json_scalar(payload.get("value")),
            unit=_text(payload.get("unit", "")),
            uncertainty=_optional_float(payload.get("uncertainty")),
            note=_text(payload.get("note", "")),
        )


@dataclass(frozen=True)
class FigureRef:
    """One figure the recipe saved, referenced by file name.

    The file lives in the report's output directory; ``file`` is its name
    relative to that directory, never an absolute path, so a report copied
    with its experiment folder still finds its figures.

    Attributes:
        file: File name relative to the output directory (``"overview.png"``).
        caption: The caption the notebook shows under it.
        width_px: Rendered width in pixels, or ``0`` for the renderer's
            default.
    """

    file: str
    caption: str = ""
    width_px: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {"file": self.file, "caption": self.caption, "width_px": self.width_px}

    @classmethod
    def from_dict(cls, data: object) -> FigureRef:
        """Load tolerantly from a dict."""
        payload = _dict(data)
        width = payload.get("width_px", 0)
        return cls(
            file=_text(payload.get("file", "")),
            caption=_text(payload.get("caption", "")),
            width_px=int(width) if isinstance(width, int) and not isinstance(width, bool) else 0,
        )


@dataclass(frozen=True)
class TableSpec:
    """A small table for the notebook — bounded rows and columns.

    Attributes:
        caption: What the table shows.
        columns: Column headings.
        rows: Rows of JSON scalars, each as long as ``columns``.
        truncated: ``True`` when rows or columns were cut to the caps.
    """

    caption: str
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    truncated: bool = False

    @classmethod
    def build(
        cls, caption: str, columns: list[str] | tuple[str, ...], rows: list[list[Any]] | list[tuple[Any, ...]]
    ) -> TableSpec:
        """Build a table, applying the caps and marking truncation.

        Args:
            caption: What the table shows.
            columns: Column headings.
            rows: The rows; each is cut or padded to ``len(columns)``.

        Returns:
            The bounded table.
        """
        cols = tuple(_text(c, 200) for c in list(columns)[:MAX_TABLE_COLUMNS])
        cut_cols = len(columns) > MAX_TABLE_COLUMNS
        kept = list(rows)[:MAX_TABLE_ROWS]
        cut_rows = len(rows) > MAX_TABLE_ROWS
        width = len(cols)
        fixed = tuple(
            tuple(_json_scalar(cell) for cell in (list(row)[:width] + [None] * (width - len(row))))
            for row in kept
        )
        return cls(caption=_text(caption), columns=cols, rows=fixed, truncated=cut_cols or cut_rows)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {
            "caption": self.caption,
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: object) -> TableSpec:
        """Load tolerantly from a dict, re-applying the caps."""
        payload = _dict(data)
        columns = payload.get("columns")
        rows = payload.get("rows")
        table = cls.build(
            payload.get("caption", ""),
            [str(c) for c in columns] if isinstance(columns, list) else [],
            [list(r) for r in rows if isinstance(r, (list, tuple))] if isinstance(rows, list) else [],
        )
        if _bool(payload.get("truncated"), False) and not table.truncated:
            table = TableSpec(table.caption, table.columns, table.rows, truncated=True)
        return table


@dataclass(frozen=True)
class AnalysisReport:
    """Everything one analysis produced for one run — the recipe's whole answer.

    Attributes:
        run_id: The run analysed.
        recipe: The recipe's ``name``.
        recipe_digest: SHA-256 of the recipe's source text, so the entry
            says which code produced it.
        status: ``REPORT_OK`` or ``REPORT_FAILED``.
        error: The failure, one line plus traceback, when ``status`` is
            ``failed``; ``""`` otherwise.
        summary: Plain-text paragraphs, in order — what the run showed.
        results: Derived values, in the order they should be listed.
        figures: Saved figures, in the order they should appear.
        tables: Small tables.
        tags: Extra notebook tags this analysis proposes.
        include_fact_tables: Append the run's full fact tables (parameters,
            per-column statistics, setup) below the analysis.
        attach_data_file: Attach the raw data file to the entry.
        warnings: Non-fatal notes from the recipe or the framework (a table
            was truncated, a column was missing, matplotlib was absent).
        options: The options the recipe was run with (echoed from the spec).
        started_utc: When the worker started the recipe (ISO 8601), or ``""``.
        duration_s: Wall time the recipe took, or ``0.0``.
    """

    run_id: str = ""
    recipe: str = ""
    recipe_digest: str = ""
    status: str = REPORT_OK
    error: str = ""
    summary: tuple[str, ...] = ()
    results: tuple[ResultValue, ...] = ()
    figures: tuple[FigureRef, ...] = ()
    tables: tuple[TableSpec, ...] = ()
    tags: tuple[str, ...] = ()
    include_fact_tables: bool = False
    attach_data_file: bool = False
    warnings: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)
    started_utc: str = ""
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether the recipe ran to completion."""
        return self.status == REPORT_OK

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form, caps applied."""
        return {
            "run_id": self.run_id,
            "recipe": self.recipe,
            "recipe_digest": self.recipe_digest,
            "status": self.status,
            "error": _text(self.error, 20000),
            "summary": [_text(p) for p in self.summary[:MAX_SUMMARY_PARAGRAPHS]],
            "results": [r.to_dict() for r in self.results[:MAX_RESULTS]],
            "figures": [f.to_dict() for f in self.figures[:MAX_FIGURES]],
            "tables": [t.to_dict() for t in self.tables],
            "tags": list(self.tags),
            "include_fact_tables": self.include_fact_tables,
            "attach_data_file": self.attach_data_file,
            "warnings": [_text(w) for w in self.warnings],
            "options": {str(k): _json_scalar(v) for k, v in self.options.items()},
            "started_utc": self.started_utc,
            "duration_s": float(self.duration_s),
        }

    @classmethod
    def from_dict(cls, data: object) -> AnalysisReport:
        """Load tolerantly from a dict (a ``report.json``); junk becomes defaults."""
        payload = _dict(data)
        status = payload.get("status")
        results = payload.get("results")
        figures = payload.get("figures")
        tables = payload.get("tables")
        duration = _optional_float(payload.get("duration_s"))
        return cls(
            run_id=_text(payload.get("run_id", ""), 200),
            recipe=_text(payload.get("recipe", ""), 200),
            recipe_digest=_text(payload.get("recipe_digest", ""), 128),
            status=REPORT_FAILED if status == REPORT_FAILED else REPORT_OK,
            error=_text(payload.get("error", ""), 20000),
            summary=tuple(_texts(payload.get("summary"), MAX_SUMMARY_PARAGRAPHS)),
            results=tuple(
                ResultValue.from_dict(r) for r in (results if isinstance(results, list) else [])[:MAX_RESULTS]
            ),
            figures=tuple(
                FigureRef.from_dict(f) for f in (figures if isinstance(figures, list) else [])[:MAX_FIGURES]
            ),
            tables=tuple(TableSpec.from_dict(t) for t in (tables if isinstance(tables, list) else [])),
            tags=tuple(_texts(payload.get("tags"), 50)),
            include_fact_tables=_bool(payload.get("include_fact_tables"), False),
            attach_data_file=_bool(payload.get("attach_data_file"), False),
            warnings=tuple(_texts(payload.get("warnings"), 100)),
            options=_dict(payload.get("options")),
            started_utc=_text(payload.get("started_utc", ""), 64),
            duration_s=duration if duration is not None and duration >= 0 else 0.0,
        )


@dataclass(frozen=True)
class AnalysisSpec:
    """The request the analysis worker is started with.

    Attributes:
        run_id: The run to analyse.
        data_path: Absolute path of the run's HDF5 file.
        manifest: The run manifest (procedure, kind, params, status, times).
        experiment: The experiment facts a recipe may cite — ``experiment_id``,
            ``experiment_title``, ``sample_info``, ``findings``, ``user_name``.
        setup: The setup context (``config_name``, ``instruments``).
        recipe: The recipe ``name`` to run; ``""`` lets discovery choose by
            procedure.
        recipe_dirs: Extra directories holding experiment recipes, absolute.
        output_dir: Where the report and its figures are written, absolute.
        options: Free-form recipe options (JSON scalars), passed through.
        include_fact_tables: Default for the report's flag when the recipe
            does not set it.
        attach_data_file: Default for the report's flag when the recipe does
            not set it.
    """

    run_id: str
    data_path: str
    manifest: dict[str, Any] = field(default_factory=dict)
    experiment: dict[str, Any] = field(default_factory=dict)
    setup: dict[str, Any] = field(default_factory=dict)
    recipe: str = ""
    recipe_dirs: tuple[str, ...] = ()
    output_dir: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    include_fact_tables: bool = False
    attach_data_file: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {
            "run_id": self.run_id,
            "data_path": self.data_path,
            "manifest": dict(self.manifest),
            "experiment": dict(self.experiment),
            "setup": dict(self.setup),
            "recipe": self.recipe,
            "recipe_dirs": list(self.recipe_dirs),
            "output_dir": self.output_dir,
            "options": dict(self.options),
            "include_fact_tables": self.include_fact_tables,
            "attach_data_file": self.attach_data_file,
        }

    @classmethod
    def from_dict(cls, data: object) -> AnalysisSpec:
        """Load tolerantly from a dict (a ``spec.json``)."""
        payload = _dict(data)
        dirs = payload.get("recipe_dirs")
        return cls(
            run_id=_text(payload.get("run_id", ""), 200),
            data_path=_text(payload.get("data_path", ""), 4096),
            manifest=_dict(payload.get("manifest")),
            experiment=_dict(payload.get("experiment")),
            setup=_dict(payload.get("setup")),
            recipe=_text(payload.get("recipe", ""), 200),
            recipe_dirs=tuple(str(d) for d in dirs) if isinstance(dirs, list) else (),
            output_dir=_text(payload.get("output_dir", ""), 4096),
            options=_dict(payload.get("options")),
            include_fact_tables=_bool(payload.get("include_fact_tables"), False),
            attach_data_file=_bool(payload.get("attach_data_file"), False),
        )
