"""generic_sweep — an overview of any run: one figure, one stats table, one paragraph.

The recipe that serves every procedure, so a run is never left without an
analysed entry. It makes no assumption about the physics: it finds the run's
own axis (the axis convention in ``base.choose_x_column``), plots every other
numeric column against it as stacked subplots sharing that axis, tabulates
each measured column's min/max/mean in the column's declared unit, and writes
one paragraph saying how many points were taken, over what range, for how
long, and how the run ended.

Two degenerate cases are part of the contract, not accidents:

- **A run with no written points** (aborted before the first datapoint) still
  yields an ``ok`` report: a warning, the paragraph, no figure and no table.
- **A checkout without matplotlib** still yields an ``ok`` report: the
  ``AnalysisError`` naming ``pip install i2as[analysis]`` is appended to
  the warnings and everything that is not a figure is produced as usual.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import numpy as np

from i2as.analysis.base import (
    AnalysisContext,
    AnalysisError,
    AnalysisRecipe,
    axis_label,
    choose_x_column,
    measured_columns,
)
from i2as.analysis.report import ANY_PROCEDURE, AnalysisReport, ResultValue
from i2as.core.data_reader import RunSource

logger = logging.getLogger(__name__)

#: Largest number of stacked subplots in the overview figure. Beyond this the
#: figure stops being readable, so the extra columns are named in a warning
#: and left to the statistics table.
MAX_SUBPLOTS = 6

#: Largest number of series drawn in one subplot. A measurement column
#: carrying reading-loop axes becomes one series per loop combination.
MAX_SERIES_PER_SUBPLOT = 4

#: Height in inches of one subplot, and the figure's width.
SUBPLOT_HEIGHT_IN = 1.9
FIGURE_WIDTH_IN = 7.5


def _parse_time(value: object) -> datetime | None:
    """Parse an ISO 8601 timestamp, or return ``None``.

    Args:
        value: The candidate — a string from the manifest or the file's
            metadata, or anything else.

    Returns:
        The parsed datetime, or ``None`` when the value is absent or not a
        timestamp.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _duration_s(manifest: Mapping[str, Any], metadata: Mapping[str, Any]) -> float | None:
    """Return the run's wall duration in seconds, or ``None``.

    The manifest's ``started_utc``/``finished_utc`` are preferred (they bound
    the whole run, setup included); the file's own ``start_time``/``end_time``
    are the fallback for a report built from the file alone.

    Args:
        manifest: The run manifest.
        metadata: The run file's ``read_metadata()``.

    Returns:
        The duration in seconds, or ``None`` when neither pair is complete.
    """
    for start_key, end_key, source in (
        ("started_utc", "finished_utc", manifest),
        ("start_time", "end_time", metadata),
    ):
        start = _parse_time(source.get(start_key))
        end = _parse_time(source.get(end_key))
        if start is not None and end is not None:
            return max(0.0, (end - start).total_seconds())
    return None


def _series(values: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Flatten one column's readings into labelled 1-D series.

    A scalar column is one series; a column carrying reading-loop axes after
    the sweep axis becomes one series per loop combination, labelled by its
    flattened index.

    Args:
        values: The column as read, leading axis = sweep point.

    Returns:
        ``(label suffix, 1-D values)`` pairs, capped at
        ``MAX_SERIES_PER_SUBPLOT``.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim <= 1:
        return [("", array)]
    flat = array.reshape(array.shape[0], -1)
    return [
        (f"[{index}]", flat[:, index])
        for index in range(min(flat.shape[1], MAX_SERIES_PER_SUBPLOT))
    ]


def _span(values: np.ndarray) -> tuple[float, float] | None:
    """Return the first and last finite value of an axis, or ``None``.

    Args:
        values: The axis values, in storage order.

    Returns:
        ``(first, last)``, or ``None`` when no value is finite.
    """
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return None
    return float(finite[0]), float(finite[-1])


def _format_number(value: float) -> float | None:
    """Return a JSON-safe number, mapping a non-finite statistic to ``None``.

    Args:
        value: The statistic.

    Returns:
        The float, or ``None`` when it is NaN or infinite — a column with no
        finite value has no minimum, and "no value" is not zero.
    """
    return float(value) if np.isfinite(value) else None


class GenericSweepRecipe(AnalysisRecipe):
    """Every measured column against the run's own axis, plus its statistics."""

    name = "generic_sweep"
    procedures = (ANY_PROCEDURE,)
    description = "Overview of any run: every measured column against the sweep axis, with statistics"
    # The default answer for a run nobody configured a recipe for.
    priority = 10

    def analyse(self, run: RunSource, context: AnalysisContext) -> AnalysisReport:
        """Build the overview report for one run.

        Args:
            run: The finished run.
            context: The manifest, the output directory and the helpers.

        Returns:
            An ``ok`` report — with a figure and a statistics table when the
            run has points and matplotlib is installed, and with a warning
            explaining the absence when it does not.
        """
        metadata = run.read_metadata()
        n_points = run.n_points
        x_info = choose_x_column(run, context.manifest)
        y_infos = measured_columns(run, exclude=[x_info.name] if x_info is not None else [])

        warnings: list[str] = []
        figures: list[Any] = []
        tables: list[Any] = []
        results: list[ResultValue] = []
        x_span: tuple[float, float] | None = None

        if n_points == 0:
            warnings.append(
                "the run wrote no points, so there is nothing to plot or summarise"
            )
        elif not y_infos:
            warnings.append("the run has no measured columns to plot")
        else:
            x_values = (
                np.asarray(run.read_slice(x_info.name), dtype=float)
                if x_info is not None
                else np.arange(n_points, dtype=float)
            )
            plotted = y_infos[:MAX_SUBPLOTS]
            if len(y_infos) > MAX_SUBPLOTS:
                warnings.append(
                    f"{len(y_infos)} measured columns; the figure shows the first "
                    f"{MAX_SUBPLOTS} ({', '.join(info.name for info in plotted)})"
                )
            try:
                figures.append(self._overview_figure(run, context, x_info, x_values, plotted))
            except AnalysisError as exc:
                warnings.append(str(exc))

            tables.append(self._stats_table(run, context, y_infos))
            results.extend(self._axis_results(x_info, x_values))
            x_span = _span(x_values)

        summary = self._summary(context.manifest, metadata, n_points, x_info, y_infos, x_span)
        return AnalysisReport(
            summary=(summary,),
            results=tuple(results),
            figures=tuple(figures),
            tables=tuple(tables),
            warnings=tuple(warnings),
        )

    # ── Parts of the report ───────────────────────────────────────────────

    def _overview_figure(
        self,
        run: RunSource,
        context: AnalysisContext,
        x_info: Any,
        x_values: np.ndarray,
        columns: tuple[Any, ...],
    ) -> Any:
        """Draw and save the stacked-subplot overview.

        Args:
            run: The run to read the columns from.
            context: The context whose ``pyplot()``/``figure()`` are used.
            x_info: The x column, or ``None`` for the point index.
            x_values: The x values already read.
            columns: The columns to plot, one subplot each.

        Returns:
            The ``FigureRef`` for the saved figure.

        Raises:
            AnalysisError: If matplotlib is not installed or the figure could
                not be saved.
        """
        plt = context.pyplot()
        fig, axes = plt.subplots(
            len(columns),
            1,
            sharex=True,
            figsize=(FIGURE_WIDTH_IN, max(2.4, SUBPLOT_HEIGHT_IN * len(columns))),
            squeeze=False,
        )
        for axis, info in zip([row[0] for row in axes], columns, strict=True):
            series = _series(run.read_slice(info.name))
            for suffix, values in series:
                length = min(len(x_values), len(values))
                axis.plot(
                    x_values[:length],
                    values[:length],
                    marker=".",
                    markersize=3,
                    linewidth=1.0,
                    label=f"{info.name}{suffix}" if suffix else None,
                )
            axis.set_ylabel(axis_label(info), fontsize="small")
            axis.grid(True, alpha=0.3)
            if any(suffix for suffix, _ in series):
                axis.legend(fontsize="x-small", ncol=2)
        axes[-1][0].set_xlabel(axis_label(x_info))
        fig.suptitle(f"{context.manifest.get('procedure', 'Run')} — {context.run_id}".strip(" —"))
        fig.tight_layout()
        return context.figure(
            "overview",
            fig,
            caption=f"Measured columns against {axis_label(x_info)}.",
        )

    def _stats_table(
        self, run: RunSource, context: AnalysisContext, columns: tuple[Any, ...]
    ) -> Any:
        """Build the per-column statistics table.

        Args:
            run: The run to summarise.
            context: The context whose ``table()`` applies the caps.
            columns: The measured columns to summarise.

        Returns:
            The ``TableSpec``: one row per column, min/max/mean over the
            finite values, in the column's declared unit.
        """
        rows: list[list[Any]] = []
        for info in columns:
            stats = run.summary_stats(info.name)
            rows.append(
                [
                    info.name,
                    info.unit,
                    _format_number(stats.min),
                    _format_number(stats.max),
                    _format_number(stats.mean),
                    stats.count,
                ]
            )
        return context.table(
            "Measured columns: range and mean over the finite values.",
            ["Column", "Unit", "Min", "Max", "Mean", "Finite values"],
            rows,
        )

    def _axis_results(self, x_info: Any, x_values: np.ndarray) -> list[ResultValue]:
        """Return the derived values describing the axis the run covered.

        Args:
            x_info: The x column, or ``None`` for the point index.
            x_values: The x values.

        Returns:
            The axis start and end as ``ResultValue`` rows, or an empty list
            when the axis carries no finite value.
        """
        finite = x_values[np.isfinite(x_values)] if x_values.size else x_values
        if x_info is None or finite.size == 0:
            return []
        unit = x_info.unit
        return [
            ResultValue(
                name=f"{x_info.name} at first point",
                value=float(finite[0]),
                unit=unit,
                note="read back from the run's own axis column",
            ),
            ResultValue(
                name=f"{x_info.name} at last point",
                value=float(finite[-1]),
                unit=unit,
                note="read back from the run's own axis column",
            ),
        ]

    def _summary(
        self,
        manifest: Mapping[str, Any],
        metadata: Mapping[str, Any],
        n_points: int,
        x_info: Any,
        y_infos: tuple[Any, ...],
        x_span: tuple[float, float] | None = None,
    ) -> str:
        """Write the one paragraph the notebook entry leads with.

        Args:
            manifest: The run manifest.
            metadata: The run file's metadata.
            n_points: How many points the run actually wrote.
            x_info: The x column, or ``None``.
            y_infos: The measured columns.
            x_span: The axis's first and last finite value, or ``None``.

        Returns:
            The paragraph: procedure, points, axis and its span, duration and
            terminal status.
        """
        procedure = str(manifest.get("procedure") or metadata.get("procedure") or "run")
        status = str(manifest.get("status") or metadata.get("status") or "unknown")
        duration = _duration_s(manifest, metadata)
        parts = [
            f"{procedure} recorded {n_points} point(s) of "
            f"{len(y_infos)} measured column(s) against {axis_label(x_info)}."
        ]
        if x_span is not None:
            unit = f" {x_info.unit}" if x_info is not None and x_info.unit else ""
            parts.append(f"The axis ran from {x_span[0]:.6g}{unit} to {x_span[1]:.6g}{unit}.")
        if duration is not None:
            parts.append(f"The run took {duration:.0f} s.")
        parts.append(f"It ended with status {status!r}.")
        reason = str(manifest.get("reason") or "")
        if reason:
            parts.append(f"Reason: {reason}.")
        return " ".join(parts)
