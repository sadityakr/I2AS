"""Analysis scripts — exploratory code, run once over one run, in the worker.

A **recipe** is a reviewed, reusable class: it is discovered, chosen by
procedure, and its report becomes the run's analysed entry. An **analysis
script** is the step before that: a plain Python file an agent (or a
physicist at a shell) runs once over one finished run to find out what the
data says. It runs in the same analysis worker, under the same rules — it
reads the run through ``RunSource``, it writes only into its own output
directory, it cannot import the Station and it reaches no instrument — and it
answers with the same **analysis report**, so everything downstream (the
agent's ``read_analysis_script_result``, the eLab preview, the notebook
renderer) needs no second format.

**The script contract.** The file is executed as a module body, with these
names already bound:

- ``run`` — the ``RunSource`` of the run being analysed;
- ``context`` — the ``AnalysisContext`` (``pyplot()``, ``manifest``,
  ``options``, ``output_dir``, ``warnings``);
- ``report`` — a ``ScriptReport`` builder: ``report.summary(text)``,
  ``report.value(name, value, unit, uncertainty, note)``,
  ``report.figure(name, fig, caption)``, ``report.table(caption, columns,
  rows)``, ``report.tag(text)``, ``report.warn(text)``;
- ``manifest`` and ``options`` — shorthands for ``context.manifest`` and
  ``context.options``;
- ``np`` — numpy.

Everything the script prints is captured (capped at ``MAX_STDOUT_CHARS``) and
written beside the report as ``stdout.txt``, because printing is how
exploratory code talks. A script may also bind ``report`` to an
``AnalysisReport`` of its own making; that report is then used as it stands.

A script that raises answers, like a recipe, with a ``failed`` report
carrying the traceback — and its captured output is still written, since the
lines printed before the failure are usually what explains it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import logging
import time
import traceback
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from i2as.analysis.base import AnalysisContext, AnalysisRecipe
from i2as.analysis.report import (
    ANY_PROCEDURE,
    REPORT_FAILED,
    AnalysisReport,
    AnalysisSpec,
    FigureRef,
    ResultValue,
    TableSpec,
)
from i2as.core.data_reader import RunSource, open_run

logger = logging.getLogger(__name__)

#: What a script's report names as its ``recipe``: this prefix, then the
#: script file's stem, so a parked entry says it came from a script and which.
SCRIPT_RECIPE_PREFIX = "script:"

#: The file a script's captured standard output is written to, beside its
#: ``report.json``.
STDOUT_FILENAME = "stdout.txt"

#: The most captured output kept. Past it, the rest is counted and dropped
#: and a closing line says how much — enough to explain a result, never a
#: megabyte of loop output in an agent's context.
MAX_STDOUT_CHARS = 20_000


class _CappedBuffer(io.TextIOBase):
    """A text sink that keeps the first ``limit`` characters and counts the rest."""

    def __init__(self, limit: int) -> None:
        """Build an empty buffer.

        Args:
            limit: How many characters to keep.
        """
        super().__init__()
        self._limit = limit
        self._parts: list[str] = []
        self._kept = 0
        self.dropped = 0

    def writable(self) -> bool:
        """Answer that this stream accepts writes."""
        return True

    def write(self, text: str) -> int:
        """Keep what fits, count what does not.

        Args:
            text: The text written.

        Returns:
            ``len(text)`` — every character is accepted, kept or counted.
        """
        room = self._limit - self._kept
        if room > 0:
            self._parts.append(text[:room])
            self._kept += min(room, len(text))
        self.dropped += max(0, len(text) - max(room, 0))
        return len(text)

    def text(self) -> str:
        """Return what was kept, with a closing line when anything was dropped."""
        kept = "".join(self._parts)
        if self.dropped:
            kept += f"\n[... {self.dropped} more characters of output dropped]\n"
        return kept


class ScriptReport:
    """The report builder a script is handed as ``report``.

    Each method appends one part; ``build()`` turns them into the
    ``AnalysisReport`` the worker writes. Figures and tables go through the
    context's own helpers, so they are saved, named and capped exactly as a
    recipe's are.
    """

    def __init__(self, context: AnalysisContext) -> None:
        """Build an empty report around one context.

        Args:
            context: The analysis context — where figures are saved.
        """
        self._context = context
        self._summary: list[str] = []
        self._results: list[ResultValue] = []
        self._figures: list[FigureRef] = []
        self._tables: list[TableSpec] = []
        self._tags: list[str] = []
        self._warnings: list[str] = []

    def summary(self, text: str) -> None:
        """Add one paragraph of prose — what the run showed.

        Args:
            text: The paragraph.
        """
        self._summary.append(str(text))

    def value(
        self,
        name: str,
        value: Any,
        unit: str = "",
        uncertainty: float | None = None,
        note: str = "",
    ) -> None:
        """Add one derived value.

        Args:
            name: What it is (``"Zero-field resistance"``).
            value: The value; a numpy scalar is turned into a Python number.
            unit: SI unit symbol, ``""`` for none.
            uncertainty: One-sigma uncertainty in the same unit, or ``None``.
            note: One short sentence of context.
        """
        if isinstance(value, np.generic):
            value = value.item()
        self._results.append(
            ResultValue(
                name=str(name),
                value=value,
                unit=str(unit),
                uncertainty=None if uncertainty is None else float(uncertainty),
                note=str(note),
            )
        )

    def figure(self, name: str, fig: Any, caption: str = "") -> FigureRef:
        """Save one matplotlib figure and add it.

        Args:
            name: File stem — a plain identifier.
            fig: The figure.
            caption: What it shows.

        Returns:
            The saved ``FigureRef``.
        """
        ref = self._context.figure(name, fig, caption=caption)
        self._figures.append(ref)
        return ref

    def table(self, caption: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        """Add one small table, capped like a recipe's.

        Args:
            caption: What it shows.
            columns: Column headings.
            rows: The rows.
        """
        self._tables.append(self._context.table(caption, columns, rows))

    def tag(self, text: str) -> None:
        """Propose one notebook tag.

        Args:
            text: The tag.
        """
        self._tags.append(str(text))

    def warn(self, text: str) -> None:
        """Record one non-fatal note.

        Args:
            text: The note.
        """
        self._warnings.append(str(text))

    def build(self) -> AnalysisReport:
        """Return the report assembled so far.

        Returns:
            An ``ok`` ``AnalysisReport``; the worker stamps the provenance.
        """
        return AnalysisReport(
            summary=tuple(self._summary),
            results=tuple(self._results),
            figures=tuple(self._figures),
            tables=tuple(self._tables),
            tags=tuple(self._tags),
            warnings=tuple(self._warnings),
        )


def script_recipe_name(script_path: str | Path) -> str:
    """Return the ``recipe`` a script's report is stamped with.

    Args:
        script_path: The script file.

    Returns:
        ``"script:<stem>"``.
    """
    return f"{SCRIPT_RECIPE_PREFIX}{Path(script_path).stem}"


def _write_stdout(output_dir: Path, text: str) -> None:
    """Write the captured output beside the report; a failure is only logged.

    Args:
        output_dir: The script's output directory.
        text: The captured output.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / STDOUT_FILENAME).write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.warning("analysis: could not write the script's output: %s", exc)


def execute_script(source: str, filename: str, run: Any, context: AnalysisContext) -> AnalysisReport:
    """Execute one script body over one open run — the core both paths share.

    Runs the script with the names the script contract binds, captures what
    it prints into ``<output_dir>/stdout.txt`` (written even when the script
    raises), and returns what it built. It does NOT catch the script's
    exception: ``run_script()`` turns it into a failed report, and a
    ``ScriptRecipe`` lets the runner do so, exactly as for any recipe.

    Args:
        source: The script's text.
        filename: The name tracebacks cite.
        run: The open ``RunSource``.
        context: The analysis context.

    Returns:
        The report the script built, or the ``AnalysisReport`` it bound to
        ``report`` itself.

    Raises:
        SyntaxError: If the source does not compile.
        Exception: Whatever the script raised.
    """
    code = compile(source, filename, "exec")
    builder = ScriptReport(context)
    buffer = _CappedBuffer(MAX_STDOUT_CHARS)
    namespace: dict[str, Any] = {
        "__name__": "__i2as_analysis_script__",
        "run": run,
        "context": context,
        "report": builder,
        "manifest": context.manifest,
        "options": context.options,
        "np": np,
    }
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, namespace)  # noqa: S102 — executing the script IS this function's job
    finally:
        _write_stdout(context.output_dir, buffer.text())
    produced = namespace.get("report")
    return produced if isinstance(produced, AnalysisReport) else builder.build()


class ScriptRecipe(AnalysisRecipe):
    """A recipe whose ``analyse`` is a script body — how a script is SAVED.

    The bridge from exploring to keeping: a script that proved its worth is
    written into an experiment's ``analysis/recipes`` folder as a subclass of
    this, with the script's text as ``SCRIPT``, and from then on it is an
    ordinary recipe — discovered, chosen by procedure, digested, reviewed in
    the eLab tab, and run by the worker like any other. The body runs under
    the script contract (this module's docstring), so the text that was
    explored is the text that is kept, unchanged.

    Attributes:
        SCRIPT: The script's text.
    """

    SCRIPT: str = ""

    def analyse(self, run: RunSource, context: AnalysisContext) -> AnalysisReport:
        """Run ``SCRIPT`` over the run.

        Args:
            run: The run.
            context: The context.

        Returns:
            The report the script built.
        """
        return execute_script(self.SCRIPT, f"<recipe {self.name}>", run, context)


def render_script_recipe(
    name: str,
    source: str,
    *,
    procedures: Sequence[str] = (),
    description: str = "",
) -> str:
    """Render a saved script as a recipe module's source.

    The script's text is embedded as a raw triple-quoted string when it can
    be (so the file reads as the script it is), and as a ``repr`` otherwise —
    either way the module compiles and ``SCRIPT`` holds the text byte for
    byte.

    Args:
        name: The recipe's name — a plain identifier.
        source: The script's text.
        procedures: The procedures it serves; empty means every procedure.
        description: Its one-line description.

    Returns:
        The module source.
    """
    class_name = "".join(part.capitalize() for part in name.split("_") if part) + "Recipe"
    summary = description or "Saved analysis script"
    return (
        f"{(name + ' — an analysis script saved as a recipe.')!r}\n\n"
        "from i2as.analysis.scripts import ScriptRecipe\n\n\n"
        f"class {class_name}(ScriptRecipe):\n"
        f"    {summary!r}\n\n"
        f"    name = {name!r}\n"
        f"    procedures = {tuple(procedures) or (ANY_PROCEDURE,)!r}\n"
        f"    description = {summary!r}\n"
        f"    SCRIPT = {_script_literal(source)}\n"
    )


def _script_literal(source: str) -> str:
    """Return a Python literal whose value is exactly *source*.

    A raw triple-quoted string keeps the script readable in the saved file;
    it is used only when it provably round-trips, and ``repr`` otherwise.

    Args:
        source: The script's text.

    Returns:
        The literal's source text.
    """
    literal = f"r'''{source}'''"
    namespace: dict[str, Any] = {}
    try:
        exec(compile(f"value = {literal}\n", "<script literal>", "exec"), namespace)  # noqa: S102
    except SyntaxError:
        return repr(source)
    return literal if namespace.get("value") == source else repr(source)


def run_script(spec: AnalysisSpec) -> AnalysisReport:
    """Run the analysis script one spec names and return its report.

    Never raises, exactly like ``runner.run_spec()``: a script that is not
    there, does not compile or raises answers with a ``failed`` report.

    Args:
        spec: The request; its ``script_path`` names the script.

    Returns:
        The report, stamped with ``run_id``, ``recipe`` (``script:<stem>``),
        ``recipe_digest`` (SHA-256 of the script's text), ``options``,
        ``started_utc`` and ``duration_s``.
    """
    started_utc = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    name = script_recipe_name(spec.script_path)
    output_dir = Path(spec.output_dir) if spec.output_dir else Path(spec.script_path).parent
    digest = ""

    def _stamped(report: AnalysisReport, warnings: tuple[str, ...] = ()) -> AnalysisReport:
        return dataclasses.replace(
            report,
            run_id=spec.run_id or report.run_id,
            recipe=name,
            recipe_digest=digest,
            warnings=tuple(report.warnings)
            + tuple(note for note in warnings if note not in report.warnings),
            include_fact_tables=report.include_fact_tables or spec.include_fact_tables,
            attach_data_file=report.attach_data_file or spec.attach_data_file,
            options=dict(spec.options),
            started_utc=started_utc,
            duration_s=time.monotonic() - clock,
        )

    def _failed(error: str, warnings: tuple[str, ...] = ()) -> AnalysisReport:
        logger.warning("analysis: script %s failed: %s", name, error.splitlines()[0] if error else "")
        return _stamped(AnalysisReport(status=REPORT_FAILED, error=error), warnings)

    try:
        source = Path(spec.script_path).read_text(encoding="utf-8")
    except OSError as exc:
        return _failed(f"no readable analysis script at {spec.script_path!r}: {exc}")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()

    data_path = Path(spec.data_path)
    if not spec.data_path or not data_path.is_file():
        return _failed(f"no readable run file at {spec.data_path!r}")

    context = AnalysisContext(
        run_id=spec.run_id,
        manifest=dict(spec.manifest),
        experiment=dict(spec.experiment),
        setup=dict(spec.setup),
        output_dir=output_dir,
        options=dict(spec.options),
        warnings=[],
    )
    logger.info("analysis: running script %s on run %s", name, spec.run_id)
    try:
        with open_run(data_path) as run:
            report = execute_script(source, str(spec.script_path), run, context)
    except SyntaxError as exc:
        return _failed(f"the script does not compile: {exc.msg} (line {exc.lineno})")
    except Exception as exc:  # noqa: BLE001 — a script failure is data, never a crash
        return _failed(
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            tuple(context.warnings),
        )
    return _stamped(report, tuple(context.warnings))


__all__ = [
    "MAX_STDOUT_CHARS",
    "SCRIPT_RECIPE_PREFIX",
    "STDOUT_FILENAME",
    "ScriptRecipe",
    "ScriptReport",
    "execute_script",
    "render_script_recipe",
    "run_script",
    "script_recipe_name",
]
