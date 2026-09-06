"""The recipe contract — the written standard every analysis recipe implements.

**The recipe contract.** A *recipe* is a plain Python class turning ONE
finished run into ONE ``AnalysisReport``. Nothing else. The rules a new
recipe must satisfy — all machine-checked by the analysis conformance tests
in ``tests/test_conformance.py``:

1. **One class, three declarations, one method.** Subclass ``AnalysisRecipe``
   and declare a ``name`` (a unique snake_case identifier, the id the panel,
   the settings file and the agent's tools use), a one-line ``description``,
   and ``procedures`` — the tuple of procedure names the recipe serves (class
   name or display name; matched case- and punctuation-insensitively),
   or ``(ANY_PROCEDURE,)`` for a recipe that serves every run. Then implement
   ``analyse(self, run, context) -> AnalysisReport``. A recipe is instantiated
   with no arguments, so it holds no constructor state; anything it needs
   comes from ``run`` and ``context``. A recipe that should be PREFERRED over
   another one matching the run equally well also declares a ``priority``:
   discovery orders package recipes by it (highest first, then by name), so
   the shipped ``generic_sweep`` — priority 10 — is what a run nobody
   configured a recipe for gets, and ``facts_only`` — priority 0 — has to be
   asked for.
2. **It reads the run through the run-source vocabulary and nothing else.**
   ``run`` is a ``i2as.core.data_reader.RunSource`` — ``n_points``,
   ``list_columns()``, ``read_slice()``, ``summary_stats()``,
   ``read_metadata()`` — exactly what an agent's run-reading tools see. A
   recipe never opens the file itself and never reaches for a path other than
   ``context.output_dir``.
3. **It may import the data reader, the report types, this module, numpy,
   h5py, matplotlib (lazily, through ``context``) and the standard library.**
   It must NEVER import the Station, the Orchestrator, a driver, a virtual
   instrument, a procedure, the session layer or the GUI — mechanically
   enforced by import contract C22, and true by construction anyway: a recipe
   runs in the **analysis worker**, a separate process started as ``python -m
   i2as.analysis run --spec <file>``, which has no Station to touch and no
   engine to talk to. A recipe therefore CANNOT command hardware, cannot
   write an experiment record, and cannot publish anything. It reads a file
   and returns a value.
4. **Figures and tables are made through the context, never by hand.**
   ``context.pyplot()`` returns ``matplotlib.pyplot`` with the headless
   ``Agg`` backend already selected; ``context.figure(name, fig, caption)``
   saves the figure as ``<output_dir>/<name>.png`` at
   ``FIGURE_DPI`` dots per inch, closes it, and returns the ``FigureRef`` to
   put in the report; ``context.table(caption, columns, rows)`` builds a
   capped ``TableSpec``. matplotlib is the OPTIONAL ``analysis`` extra: both
   helpers raise ``AnalysisError`` naming ``pip install i2as[analysis]``
   when it is absent, and a well-behaved recipe catches that, appends the
   message to its warnings and returns a report without a figure rather than
   failing outright.
5. **Sizes are bounded.** ``report.py``'s caps (``MAX_FIGURES``,
   ``MAX_TABLE_ROWS``, ``MAX_RESULTS``, ``MAX_SUMMARY_PARAGRAPHS``,
   ``MAX_TEXT_CHARS``) apply to every report, and a table built through
   ``context.table()`` truncates itself and says so. A report is meant to be
   previewed in a panel, pasted into a notebook entry and handed to a model:
   keep it to a few paragraphs, a handful of figures, and small tables.
   Bulk data stays in the HDF5 file, which the report can ask to attach.
6. **Failure is data, not an exception.** A recipe MAY raise: the runner
   catches everything and returns a ``REPORT_FAILED`` report carrying the
   traceback, so a broken recipe costs a visible failure, never a lost run.
   A recipe that merely could not do part of its job appends to
   ``context.warnings`` and returns an ``ok`` report.

Two column conventions are shared by the shipped recipes and by the scaffold
template, and live here so a new recipe inherits them: ``choose_x_column()``
(what to plot against) and ``measured_columns()`` (what to plot).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from i2as.analysis.report import AnalysisReport, FigureRef, TableSpec
from i2as.core.data_reader import (
    ROLE_IMAGE,
    ROLE_LOOP_AXIS,
    ROLE_RAW_BLOCK,
    TIMESTAMP_COLUMN,
    ColumnInfo,
    RunSource,
)
from i2as.core.exceptions import I2ASError

logger = logging.getLogger(__name__)

#: Resolution every saved figure is written at.
FIGURE_DPI = 120

#: The one file type a figure is saved as.
FIGURE_SUFFIX = ".png"

#: The install command an ``AnalysisError`` names when matplotlib is missing.
#: matplotlib is the OPTIONAL ``analysis`` extra, so a checkout without it
#: imports, lints and tests unchanged and a recipe asking for a figure gets
#: one clear sentence instead of an ImportError traceback.
MATPLOTLIB_INSTALL_HINT = "pip install i2as[analysis]"

#: The element types a recipe can plot or summarise.
NUMERIC_DTYPES: tuple[str, ...] = ("float64", "int64")

#: Column names that mean "time since the run started", best first. The
#: fallback x axis when the run declares no sweep axis of its own — an
#: elapsed-time run (``TimeSeries``) writes ``elapsed_s``, and every run
#: writes ``unix_time``.
ELAPSED_COLUMNS: tuple[str, ...] = ("elapsed_s", "time_s", "unix_time")

#: Manifest keys that may name the run's own x axis, best first. A procedure
#: declares its axis column as ``default_x_key`` / ``sweep_data_keys``; when
#: whoever built the spec put either on the manifest, it wins over the
#: file-derived guess below.
X_COLUMN_MANIFEST_KEYS: tuple[str, ...] = (
    "default_x_key",
    "sweep_data_key",
    "sweep_data_keys",
)

#: A legal recipe/figure name: a lowercase-ish Python identifier that does not
#: start with an underscore, so it is safe as both a module name and a file
#: name.
_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class AnalysisError(I2ASError):
    """Any failure raised by the analysis stage itself.

    Raised by the context helpers (matplotlib absent, a bad figure name) and
    by the scaffold (a bad recipe name, a file already there). A recipe's own
    exception is NOT wrapped in this — the runner reports whatever came out —
    so this type means "the analysis framework refused", not "the science
    went wrong".
    """


def _import_pyplot() -> Any:
    """Import ``matplotlib.pyplot`` with the headless backend selected.

    Returns:
        The ``matplotlib.pyplot`` module, with the ``Agg`` backend selected
        before it was first imported, so a recipe draws in a worker process
        that has no display.

    Raises:
        AnalysisError: If matplotlib is not installed, naming the install
            command for the optional ``analysis`` extra.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise AnalysisError(
            "matplotlib is not installed, so this analysis cannot draw a "
            f"figure; install the optional analysis extra: {MATPLOTLIB_INSTALL_HINT}"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass
class AnalysisContext:
    """Everything a recipe is given besides the run itself.

    The run's own numbers come from the ``RunSource``; this carries the
    surrounding facts (which run, which procedure, which experiment, which
    setup), where output goes, the options the caller asked for, and the two
    helpers that build the parts of a report a recipe cannot build alone.

    Recipes may append to ``warnings``; the runner merges them into the
    report it returns, so a note survives even when the recipe forgot to copy
    it into its own report.

    Attributes:
        run_id: The run being analysed.
        manifest: The run manifest — ``procedure``, ``kind``, ``params``,
            ``status``, ``started_utc``/``finished_utc``, ``data_file``.
        experiment: The experiment facts a recipe may cite —
            ``experiment_id``, ``experiment_title``, ``sample_info``,
            ``findings``, ``user_name``.
        setup: The setup context — ``config_name``, ``instruments``.
        output_dir: Where figures (and the report) are written. Created on
            first use by ``figure()``; a recipe writes nothing outside it.
        options: Free-form recipe options passed through from the spec.
        warnings: Non-fatal notes; a recipe appends, the runner collects.
    """

    run_id: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    experiment: dict[str, Any] = field(default_factory=dict)
    setup: dict[str, Any] = field(default_factory=dict)
    output_dir: Path = field(default_factory=Path)
    options: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def pyplot(self) -> Any:
        """Return ``matplotlib.pyplot``, headless and ready to draw.

        The import is lazy and the ``Agg`` backend is selected before pyplot
        is first imported, because the worker has no display and matplotlib
        is an optional extra.

        Returns:
            The ``matplotlib.pyplot`` module.

        Raises:
            AnalysisError: If matplotlib is not installed; the message names
                ``pip install i2as[analysis]``.
        """
        return _import_pyplot()

    def figure(self, name: str, fig: Any, caption: str = "", width_px: int = 0) -> FigureRef:
        """Save one matplotlib figure into the output directory.

        The figure is written as ``<output_dir>/<name>.png`` at
        ``FIGURE_DPI``, then closed — a recipe never has to remember to close
        one, and a worker that drew ten figures does not hold ten open.

        Args:
            name: File stem, a plain identifier (no directory separator, no
                extension). It is the reader's handle on the figure, so make
                it descriptive: ``"overview"``, ``"critical_field_fit"``.
            fig: The ``matplotlib.figure.Figure`` to save.
            caption: The caption the notebook shows under it.
            width_px: Rendered width in pixels, or ``0`` for the renderer's
                default.

        Returns:
            The ``FigureRef`` to put in the report; its ``file`` is the name
            RELATIVE to the output directory, so the report travels with its
            experiment folder.

        Raises:
            AnalysisError: If ``name`` is not a safe file stem, if matplotlib
                is not installed, or if the file could not be written.
        """
        if not _SAFE_NAME.match(name):
            raise AnalysisError(
                f"figure name {name!r} must be a plain identifier (letters, "
                f"digits and underscores, starting with a letter) — it becomes "
                f"a file name inside the report directory"
            )
        plt = self.pyplot()
        path = self.output_dir / f"{name}{FIGURE_SUFFIX}"
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        except OSError as exc:
            raise AnalysisError(f"could not save figure {name!r} to {path}: {exc}") from exc
        finally:
            plt.close(fig)
        logger.debug("analysis: saved figure %s", path)
        return FigureRef(file=path.name, caption=caption, width_px=width_px)

    def table(
        self,
        caption: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
    ) -> TableSpec:
        """Build one capped table for the report.

        Args:
            caption: What the table shows.
            columns: Column headings.
            rows: The rows; each is cut or padded to ``len(columns)``.

        Returns:
            The ``TableSpec``, truncated to the report caps and marked
            ``truncated`` when it was.
        """
        return TableSpec.build(caption, list(columns), [list(row) for row in rows])


class AnalysisRecipe:
    """One analysis of one run — the class every recipe subclasses.

    See this module's docstring for the full recipe contract. A recipe is
    instantiated with no arguments and asked exactly once for one report.

    Attributes:
        name: Unique snake_case id, e.g. ``"field_sweep_overview"``. A recipe
            with an empty name is an intermediate base and is not discovered.
        procedures: The procedures this recipe serves — class names or
            display names, matched case- and punctuation-insensitively — or
            ``(ANY_PROCEDURE,)`` for every run.
        description: One line, shown in the panel and in the agent's recipe
            list.
        priority: Selection order among recipes that match a run equally
            well; higher is preferred, ties broken by ``name``. Leave it at
            ``0`` unless the recipe is meant to be somebody's default.
    """

    name: str = ""
    procedures: tuple[str, ...] = ()
    description: str = ""
    priority: int = 0

    def analyse(self, run: RunSource, context: AnalysisContext) -> AnalysisReport:
        """Turn one finished run into one report.

        Args:
            run: The run to read, as a ``RunSource``.
            context: The surrounding facts, the output directory, the options
                and the figure/table helpers.

        Returns:
            The report — prose, derived values, figures and small tables.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement analyse(run, context) and "
            f"return an AnalysisReport"
        )


def is_numeric(info: ColumnInfo) -> bool:
    """Whether a column holds numbers a recipe can plot or summarise.

    Args:
        info: The column declaration.

    Returns:
        ``True`` for a float or integer column, ``False`` for a string column
        (the ``timestamp``) or anything else.
    """
    return info.dtype in NUMERIC_DTYPES


def choose_x_column(run: RunSource, manifest: Mapping[str, Any] | None = None) -> ColumnInfo | None:
    """Return the column a run should be plotted against, or ``None``.

    The axis convention, best first:

    1. The manifest's own declaration (``default_x_key`` / ``sweep_data_key``
       / the first of ``sweep_data_keys``) — what the procedure declared as
       its axis — when that column exists in the run and is numeric.
    2. The run file's own last-declared sweep column. The writer's
       ``data_config['sweep_columns']`` lists the system read-backs first and
       the procedure's own axis read-back last, so the last numeric entry
       that the file actually has is the swept quantity.
    3. An elapsed-time column (``ELAPSED_COLUMNS``), for a run with no swept
       setpoint at all.
    4. ``None`` — plot against the point index.

    Args:
        run: The run to inspect.
        manifest: The run manifest, or ``None``.

    Returns:
        The chosen column, or ``None`` when the run declares no usable axis.
    """
    by_name = {info.name: info for info in run.list_columns() if is_numeric(info)}
    if not by_name:
        return None

    for key in X_COLUMN_MANIFEST_KEYS:
        declared = (manifest or {}).get(key)
        if isinstance(declared, (list, tuple)):
            declared = declared[0] if declared else None
        if isinstance(declared, str) and declared in by_name:
            return by_name[declared]

    metadata = run.read_metadata()
    raw = metadata.get("raw") if isinstance(metadata.get("raw"), Mapping) else {}
    data_config = raw.get("data_config") if isinstance(raw, Mapping) else None
    if isinstance(data_config, Mapping):
        declared_sweeps = data_config.get("sweep_columns")
        if isinstance(declared_sweeps, Mapping):
            candidates = [
                name
                for name in declared_sweeps
                if str(name) in by_name and str(name) != TIMESTAMP_COLUMN
            ]
            if candidates:
                return by_name[str(candidates[-1])]

    for name in ELAPSED_COLUMNS:
        if name in by_name:
            return by_name[name]
    return None


def measured_columns(run: RunSource, exclude: Sequence[str] = ()) -> tuple[ColumnInfo, ...]:
    """Return the numeric columns worth plotting against the x axis.

    Everything numeric except: the ``timestamp`` counter and the run's other
    clocks (``ELAPSED_COLUMNS``), which are axes rather than readings; the
    reading loop's own axis columns, which are labels rather than readings;
    the raw diagnostic blocks, which are for diagnostics rather than for an
    overview; and whatever the caller excluded — normally the x column
    itself.

    Args:
        run: The run to inspect.
        exclude: Column names to leave out.

    Returns:
        The columns, in the source's own name order.
    """
    skipped = {TIMESTAMP_COLUMN, *ELAPSED_COLUMNS, *exclude}
    return tuple(
        info
        for info in run.list_columns()
        if is_numeric(info)
        and info.name not in skipped
        and info.role not in (ROLE_LOOP_AXIS, ROLE_RAW_BLOCK, ROLE_IMAGE)
    )


def axis_label(info: ColumnInfo | None) -> str:
    """Return the axis label for one column.

    Args:
        info: The column, or ``None`` for the point-index fallback.

    Returns:
        ``"field_T (T)"`` for a column declaring a unit, the bare name when it
        declares none, ``"point index"`` for ``None``.
    """
    if info is None:
        return "point index"
    return f"{info.name} ({info.unit})" if info.unit else info.name


__all__ = [
    "ELAPSED_COLUMNS",
    "FIGURE_DPI",
    "FIGURE_SUFFIX",
    "MATPLOTLIB_INSTALL_HINT",
    "NUMERIC_DTYPES",
    "X_COLUMN_MANIFEST_KEYS",
    "AnalysisContext",
    "AnalysisError",
    "AnalysisRecipe",
    "axis_label",
    "choose_x_column",
    "is_numeric",
    "measured_columns",
]
