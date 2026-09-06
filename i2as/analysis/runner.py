"""Running one recipe against one run — in process, and never raising.

The half of the analysis worker that does the work. ``__main__.py`` is the
process; this is what it calls, and it is deliberately importable on its own
so a test (or a future in-process caller) exercises exactly the code the
worker runs, with no subprocess in the way.

**A failure is a report, not an exception.** ``run_spec()`` catches
everything a recipe can do to it — raising, returning junk, taking a
column that is not there — and answers with a ``REPORT_FAILED`` report whose
``error`` carries the traceback. The three framework-level failures (no
recipe of that name, no readable data file, a recipe that will not import)
answer the same way, with one clear sentence instead of a traceback. The
caller therefore has exactly one thing to read in every case: a report. That
is what lets the notebook fall back to a facts-only entry instead of losing
the run, and what lets the worker exit 0 even when the analysis failed.

Every report leaves here stamped with what produced it: the ``run_id``, the
``recipe`` name, the ``recipe_digest`` (SHA-256 of the recipe source, so an
entry says which code ran even after the file was edited), ``started_utc``,
``duration_s`` and the ``options`` it was run with.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from i2as.analysis.base import AnalysisContext
from i2as.analysis.discovery import RecipeInfo, discover_recipes, load_recipe, recipe_for
from i2as.analysis.report import (
    REPORT_FAILED,
    REPORT_FILENAME,
    AnalysisReport,
    AnalysisSpec,
)
from i2as.core.data_reader import open_run

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Returns:
        e.g. ``"2026-09-05T14:03:11.482913+00:00"``.
    """
    return datetime.now(timezone.utc).isoformat()


def _failed_report(
    spec: AnalysisSpec,
    error: str,
    *,
    started_utc: str,
    duration_s: float,
    info: RecipeInfo | None = None,
    warnings: tuple[str, ...] = (),
) -> AnalysisReport:
    """Build the report a failure answers with.

    Args:
        spec: The request being served.
        error: The failure text — one line, plus a traceback when there was
            one.
        started_utc: When the attempt started.
        duration_s: How long it took to fail.
        info: The recipe that was chosen, when one was.
        warnings: Anything the recipe or the framework noted before failing.

    Returns:
        A ``REPORT_FAILED`` report carrying the failure and the provenance.
    """
    logger.warning("analysis: run %s failed: %s", spec.run_id, error.splitlines()[0] if error else "")
    return AnalysisReport(
        run_id=spec.run_id,
        recipe=info.name if info else spec.recipe,
        recipe_digest=info.digest if info else "",
        status=REPORT_FAILED,
        error=error,
        warnings=warnings,
        include_fact_tables=spec.include_fact_tables,
        attach_data_file=spec.attach_data_file,
        options=dict(spec.options),
        started_utc=started_utc,
        duration_s=duration_s,
    )


def _resolve_recipe(spec: AnalysisSpec) -> tuple[RecipeInfo | None, str]:
    """Choose the recipe this spec asks for.

    Args:
        spec: The request — its ``recipe`` name when it names one, its
            ``recipe_dirs`` for the experiment's own scripts, and its
            manifest's ``procedure`` for the by-procedure choice.

    Returns:
        ``(info, error)``: the chosen recipe and ``""``, or ``None`` and one
        sentence naming what was available.
    """
    recipes = discover_recipes(spec.recipe_dirs)
    known = ", ".join(sorted(info.name for info in recipes)) or "none"
    if spec.recipe and not any(info.name == spec.recipe for info in recipes):
        return None, f"unknown recipe {spec.recipe!r}; available recipes: {known}"
    procedure = str(spec.manifest.get("procedure", ""))
    info = recipe_for(procedure, recipes, preferred=spec.recipe)
    if info is None:
        return None, (
            f"no recipe serves procedure {procedure!r} and none accepts every "
            f"procedure; available recipes: {known}"
        )
    return info, ""


def run_spec(spec: AnalysisSpec) -> AnalysisReport:
    """Run the analysis one spec asks for and return its report.

    Never raises. A recipe that fails, a data file that is not there, a
    recipe name nothing answers to — each becomes a ``REPORT_FAILED`` report
    whose ``error`` says what happened.

    Args:
        spec: The request: the run, its data file, its manifest, the
            experiment and setup facts, the recipe (or ``""`` to choose by
            procedure), the extra recipe directories, the output directory
            and the options.

    Returns:
        The report — ``ok`` or ``failed``, always stamped with ``run_id``,
        ``recipe``, ``recipe_digest``, ``started_utc``, ``duration_s`` and
        ``options``. The spec's ``include_fact_tables``/``attach_data_file``
        are applied as defaults wherever the recipe left them ``False``.
    """
    started_utc = _utc_now()
    clock = time.monotonic()
    info: RecipeInfo | None = None
    try:
        info, error = _resolve_recipe(spec)
        if info is None:
            return _failed_report(
                spec, error, started_utc=started_utc, duration_s=time.monotonic() - clock
            )

        data_path = Path(spec.data_path)
        if not spec.data_path or not data_path.is_file():
            return _failed_report(
                spec,
                f"no readable run file at {spec.data_path!r}",
                started_utc=started_utc,
                duration_s=time.monotonic() - clock,
                info=info,
            )

        recipe = load_recipe(info)
        output_dir = Path(spec.output_dir) if spec.output_dir else data_path.parent
        context = AnalysisContext(
            run_id=spec.run_id,
            manifest=dict(spec.manifest),
            experiment=dict(spec.experiment),
            setup=dict(spec.setup),
            output_dir=output_dir,
            options=dict(spec.options),
            warnings=[],
        )
        logger.info("analysis: running recipe %s on run %s", info.name, spec.run_id)
        with open_run(data_path) as run:
            report = recipe.analyse(run, context)
        if not isinstance(report, AnalysisReport):
            return _failed_report(
                spec,
                f"recipe {info.name!r} returned {type(report).__name__}, not an AnalysisReport",
                started_utc=started_utc,
                duration_s=time.monotonic() - clock,
                info=info,
                warnings=tuple(context.warnings),
            )
    except Exception as exc:  # noqa: BLE001 — a recipe failure is data, never a crash
        return _failed_report(
            spec,
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            started_utc=started_utc,
            duration_s=time.monotonic() - clock,
            info=info,
        )

    warnings = tuple(report.warnings) + tuple(
        note for note in context.warnings if note not in report.warnings
    )
    return dataclasses.replace(
        report,
        run_id=spec.run_id or report.run_id,
        recipe=info.name,
        recipe_digest=info.digest,
        warnings=warnings,
        include_fact_tables=report.include_fact_tables or spec.include_fact_tables,
        attach_data_file=report.attach_data_file or spec.attach_data_file,
        options=dict(spec.options),
        started_utc=started_utc,
        duration_s=time.monotonic() - clock,
    )


def write_report(report: AnalysisReport, output_dir: Path) -> Path:
    """Write one report as ``report.json``, atomically.

    Atomic because whoever reads it (the runner in the application process,
    an agent's ``read_analysis_report``) may look while the worker is still
    writing: the file is written beside its destination and renamed over it,
    so a reader sees either the previous report or the whole new one.

    Args:
        report: The report to write.
        output_dir: The run's analysis directory; created if needed.

    Returns:
        The path written.

    Raises:
        OSError: If the directory could not be created or the file written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / REPORT_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    os.replace(temporary, path)
    logger.debug("analysis: wrote %s", path)
    return path


def read_report(output_dir: Path) -> AnalysisReport | None:
    """Read back the report in one analysis directory.

    Args:
        output_dir: The run's analysis directory.

    Returns:
        The report, or ``None`` when there is none yet or the file is not
        readable JSON (logged at WARNING) — a caller polling for a worker's
        answer must not have to catch anything.
    """
    path = Path(output_dir) / REPORT_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("analysis: cannot read %s: %s", path, exc)
        return None
    return AnalysisReport.from_dict(payload)


__all__ = ["read_report", "run_spec", "write_report"]
