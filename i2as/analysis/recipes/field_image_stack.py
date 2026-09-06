"""field_image_stack — a Field Imaging run: montage, difference images, the loop.

The recipe that serves ``FieldImaging``: it reads the run's image block
through ``run.read_image()`` and answers the three questions a domain-imaging
run is taken to answer — what the sample looked like at each field (a
montage of at most ``MAX_PANELS`` frames, evenly subsampled), what changed
relative to the **reference frame** (difference images against frame 0,
which the saturation pre-step made a known, fully magnetised state), and
how the ROI mean followed the field (the hysteresis loop, with a coercive-
field estimate from where the normalised loop crosses zero).

Three degenerate cases are part of the contract, not accidents:

- **A run with no written points** still yields an ``ok`` report: a warning,
  the paragraph, no figure.
- **A run with no image block** (the procedure ran another measurement
  method) still yields an ``ok`` report with the loop from its scalar
  column, if it has one, and a warning about the missing frames.
- **A checkout without matplotlib** still yields an ``ok`` report: the
  ``AnalysisError`` naming ``pip install i2as[analysis]`` becomes a
  warning and every derived value is produced as usual.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
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
from i2as.analysis.report import AnalysisReport, ResultValue
from i2as.core.data_reader import ROLE_IMAGE, ColumnInfo, RunSource

logger = logging.getLogger(__name__)

#: Largest number of frames shown in the montage (and in the difference
#: montage); a longer run is subsampled evenly, first and last kept.
MAX_PANELS = 12

#: Panels per montage row, and the size of one panel in inches.
PANELS_PER_ROW = 4
PANEL_IN = 1.9

#: The scalar column the loop is taken from when the run has it; otherwise
#: the first measured column.
PREFERRED_LOOP_COLUMN = "roi_mean"

#: How much larger than its typical point-to-point step a loop's full swing
#: must be before its zero crossings are read. Normalising to the extremes
#: makes any non-constant series span [-1, 1], so the swing is judged
#: against the median step instead: a switching loop steps little except at
#: the switch, noise steps about as much as it swings.
_MIN_SWING_TO_STEP_RATIO = 3.0


def _panel_indices(n_points: int, max_panels: int = MAX_PANELS) -> list[int]:
    """Return at most *max_panels* sweep indices, evenly spread, ends kept.

    Args:
        n_points: How many points the run wrote.
        max_panels: The cap.

    Returns:
        Strictly increasing indices; all of them when the run fits.
    """
    if n_points <= 0:
        return []
    if n_points <= max_panels:
        return list(range(n_points))
    return sorted({int(round(v)) for v in np.linspace(0, n_points - 1, max_panels)})


def _normalised_loop(values: np.ndarray) -> np.ndarray | None:
    """Map a loop's scalar onto ``[-1, 1]`` from its finite extremes.

    Args:
        values: The scalar per sweep point.

    Returns:
        ``(v - mid) / half_swing``, or ``None`` when fewer than two finite
        values exist or the swing is zero.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return None
    mid = 0.5 * (low + high)
    half = 0.5 * (high - low)
    return (values - mid) / half


def coercive_fields(field: np.ndarray, values: np.ndarray) -> list[float]:
    """Estimate every field at which the normalised loop crosses zero.

    The loop is normalised to ``[-1, 1]`` from its extremes; each sign
    change between consecutive finite points is interpolated linearly to
    the field where the normalised value is zero. A series whose full swing
    is not at least ``_MIN_SWING_TO_STEP_RATIO`` times its median
    point-to-point step — noise, or a loop that never switched — yields no
    crossing.

    Args:
        field: The field at each sweep point, in tesla.
        values: The loop's scalar at each point.

    Returns:
        The crossing fields in sweep order, possibly empty.
    """
    loop = _normalised_loop(np.asarray(values, dtype=float))
    if loop is None:
        return []
    field = np.asarray(field, dtype=float)
    keep = np.isfinite(loop) & np.isfinite(field)
    loop, field = loop[keep], field[keep]
    if loop.size < 2:
        return []
    raw = np.asarray(values, dtype=float)[keep]
    swing = float(raw.max() - raw.min())
    step = float(np.median(np.abs(np.diff(raw))))
    if swing <= _MIN_SWING_TO_STEP_RATIO * step:
        return []
    crossings: list[float] = []
    for i in range(loop.size - 1):
        a, b = loop[i], loop[i + 1]
        if a == 0.0 and (i == 0 or loop[i - 1] != 0.0):
            crossings.append(float(field[i]))
        elif a * b < 0.0:
            crossings.append(float(field[i] + (0.0 - a) * (field[i + 1] - field[i]) / (b - a)))
    return crossings


class FieldImageStackRecipe(AnalysisRecipe):
    """Frames against field, differences against the reference frame, the loop."""

    name = "field_image_stack"
    procedures = ("FieldImaging",)
    description = (
        "Field Imaging run: montage of frames against field, difference images "
        "against the reference frame, and the ROI-mean hysteresis loop with a "
        "coercive-field estimate"
    )

    def analyse(self, run: RunSource, context: AnalysisContext) -> AnalysisReport:
        """Build the image-stack report for one run.

        Args:
            run: The finished run.
            context: The manifest, the output directory and the helpers.

        Returns:
            An ``ok`` report — three figures and the coercive-field results
            when the run has frames, a loop and matplotlib; warnings where
            it lacks any of them.
        """
        metadata = run.read_metadata()
        n_points = run.n_points
        columns = run.list_columns()
        image_infos = [info for info in columns if info.role == ROLE_IMAGE]
        x_info = choose_x_column(run, context.manifest)
        loop_info = self._loop_column(run, x_info)

        warnings: list[str] = []
        figures: list[Any] = []
        results: list[ResultValue] = []
        n_panels = 0
        crossings: list[float] = []

        if n_points == 0:
            warnings.append("the run wrote no points, so there is nothing to show")
        else:
            x_values = (
                np.asarray(run.read_slice(x_info.name), dtype=float)
                if x_info is not None
                else np.arange(n_points, dtype=float)
            )
            if image_infos:
                image = image_infos[0]
                if len(image_infos) > 1:
                    warnings.append(
                        f"the run has {len(image_infos)} image blocks; showing "
                        f"{image.name!r}"
                    )
                if any(len(axis) > 1 for axis in self._loop_axes(metadata)):
                    warnings.append(
                        "the run looped its reading; frames are shown for the "
                        "first loop index only"
                    )
                indices = _panel_indices(n_points)
                n_panels = len(indices)
                try:
                    frames = [run.read_image(image.name, i) for i in indices]
                    figures.append(
                        self._montage(context, image, x_info, x_values, indices, frames)
                    )
                    figures.append(
                        self._differences(context, image, x_info, x_values, indices, frames)
                    )
                except AnalysisError as exc:
                    warnings.append(str(exc))
            else:
                warnings.append("the run has no image block, so there are no frames to show")

            if loop_info is not None:
                loop_values = np.asarray(run.read_slice(loop_info.name), dtype=float)
                loop_values = loop_values.reshape(loop_values.shape[0], -1)[:, 0]
                if x_info is not None:
                    crossings = coercive_fields(x_values, loop_values)
                results.extend(self._loop_results(x_info, loop_info, crossings))
                try:
                    figures.append(
                        self._loop_figure(context, x_info, loop_info, x_values, loop_values, crossings)
                    )
                except AnalysisError as exc:
                    if not any(str(exc) == w for w in warnings):
                        warnings.append(str(exc))
                if x_info is not None and not crossings:
                    warnings.append(
                        f"{loop_info.name} never switched cleanly along the sweep, "
                        f"so no coercive field could be read off the loop"
                    )
            else:
                warnings.append("the run has no scalar column to draw a loop from")

        summary = self._summary(
            context.manifest, metadata, n_points, x_info, loop_info, n_panels, crossings
        )
        return AnalysisReport(
            summary=(summary,),
            results=tuple(results),
            figures=tuple(figures),
            warnings=tuple(warnings),
        )

    # ── Column choices ───────────────────────────────────────────────────

    @staticmethod
    def _loop_column(run: RunSource, x_info: ColumnInfo | None) -> ColumnInfo | None:
        """Return the scalar column the loop is drawn from, or ``None``.

        Args:
            run: The run.
            x_info: The axis column, excluded from the candidates.

        Returns:
            ``roi_mean`` when the run has it, else the first measured column.
        """
        candidates = measured_columns(run, exclude=[x_info.name] if x_info is not None else [])
        for info in candidates:
            if info.name == PREFERRED_LOOP_COLUMN:
                return info
        return candidates[0] if candidates else None

    @staticmethod
    def _loop_axes(metadata: Mapping[str, Any]) -> list[list[Any]]:
        """Return the reading loop's value lists from the run's parameters."""
        params = metadata.get("params") if isinstance(metadata.get("params"), Mapping) else {}
        axes: list[list[Any]] = []
        for key in ("loop1_values", "loop2_values"):
            values = params.get(key) if isinstance(params, Mapping) else None
            if isinstance(values, (list, tuple)):
                axes.append(list(values))
        return axes

    # ── Figures ──────────────────────────────────────────────────────────

    @staticmethod
    def _grid(plt: Any, n_panels: int) -> tuple[Any, list[Any]]:
        """Return a figure and its flat list of axes for *n_panels* panels."""
        rows = max(1, int(np.ceil(n_panels / PANELS_PER_ROW)))
        cols = min(PANELS_PER_ROW, max(1, n_panels))
        fig, axes = plt.subplots(
            rows, cols, figsize=(PANEL_IN * cols, PANEL_IN * rows), squeeze=False
        )
        flat = [axis for row in axes for axis in row]
        for axis in flat:
            axis.set_xticks([])
            axis.set_yticks([])
        for axis in flat[n_panels:]:
            axis.set_visible(False)
        return fig, flat

    @staticmethod
    def _panel_title(x_info: ColumnInfo | None, x_values: np.ndarray, index: int) -> str:
        """Return one panel's title: the axis value at *index*, or the index."""
        if x_info is None:
            return f"point {index}"
        value = x_values[index]
        unit = f" {x_info.unit}" if x_info.unit else ""
        return f"{value:.3g}{unit}"

    def _montage(
        self,
        context: AnalysisContext,
        image: ColumnInfo,
        x_info: ColumnInfo | None,
        x_values: np.ndarray,
        indices: list[int],
        frames: list[np.ndarray],
    ) -> Any:
        """Draw and save the montage of frames against the axis.

        Raises:
            AnalysisError: If matplotlib is not installed or the figure could
                not be saved.
        """
        plt = context.pyplot()
        stack = np.stack(frames)
        finite = stack[np.isfinite(stack)]
        vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        fig, axes = self._grid(plt, len(frames))
        for axis, index, frame in zip(axes[: len(frames)], indices, frames, strict=True):
            axis.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            axis.set_title(self._panel_title(x_info, x_values, index), fontsize="small")
        fig.suptitle(f"{image.name} against {axis_label(x_info)}", fontsize="medium")
        fig.tight_layout()
        return context.figure(
            "montage",
            fig,
            caption=(
                f"{len(frames)} of {int(x_values.size)} frames of {image.name!r} "
                f"({image.unit}) against {axis_label(x_info)}, on a common grey scale."
            ),
        )

    def _differences(
        self,
        context: AnalysisContext,
        image: ColumnInfo,
        x_info: ColumnInfo | None,
        x_values: np.ndarray,
        indices: list[int],
        frames: list[np.ndarray],
    ) -> Any:
        """Draw and save the difference images against the reference frame.

        Raises:
            AnalysisError: If matplotlib is not installed or the figure could
                not be saved.
        """
        plt = context.pyplot()
        reference = frames[0]
        differences = [frame - reference for frame in frames]
        stack = np.stack(differences)
        finite = np.abs(stack[np.isfinite(stack)])
        limit = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
        fig, axes = self._grid(plt, len(frames))
        for axis, index, difference in zip(axes[: len(frames)], indices, differences, strict=True):
            axis.imshow(
                difference, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest"
            )
            axis.set_title(self._panel_title(x_info, x_values, index), fontsize="small")
        fig.suptitle(
            f"{image.name} minus the reference frame "
            f"({self._panel_title(x_info, x_values, indices[0])})",
            fontsize="medium",
        )
        fig.tight_layout()
        return context.figure(
            "difference",
            fig,
            caption=(
                f"Each frame minus the reference frame (the first, at "
                f"{self._panel_title(x_info, x_values, indices[0])}); red is brighter "
                f"than the reference, blue darker, on a symmetric scale of "
                f"±{limit:.3g} {image.unit}."
            ),
        )

    def _loop_figure(
        self,
        context: AnalysisContext,
        x_info: ColumnInfo | None,
        loop_info: ColumnInfo,
        x_values: np.ndarray,
        loop_values: np.ndarray,
        crossings: list[float],
    ) -> Any:
        """Draw and save the loop of the scalar column against the axis.

        Raises:
            AnalysisError: If matplotlib is not installed or the figure could
                not be saved.
        """
        plt = context.pyplot()
        fig, axis = plt.subplots(figsize=(5.5, 3.6))
        length = min(len(x_values), len(loop_values))
        axis.plot(x_values[:length], loop_values[:length], marker="o", markersize=3, linewidth=1.0)
        if length > 1:
            axis.annotate(
                "",
                xy=(x_values[length - 1], loop_values[length - 1]),
                xytext=(x_values[length - 2], loop_values[length - 2]),
                arrowprops={"arrowstyle": "->", "color": "C0"},
            )
        for crossing in crossings:
            axis.axvline(crossing, color="C3", linestyle="--", linewidth=0.8)
        axis.set_xlabel(axis_label(x_info))
        axis.set_ylabel(axis_label(loop_info))
        axis.grid(True, alpha=0.3)
        fig.suptitle(f"{loop_info.name} against {axis_label(x_info)}", fontsize="medium")
        fig.tight_layout()
        return context.figure(
            "loop",
            fig,
            caption=(
                f"{loop_info.name} against {axis_label(x_info)} in sweep order"
                + (
                    "; dashed lines mark where the normalised loop crosses zero."
                    if crossings
                    else "."
                )
            ),
        )

    # ── Derived values and the paragraph ────────────────────────────────

    @staticmethod
    def _loop_results(
        x_info: ColumnInfo | None, loop_info: ColumnInfo, crossings: list[float]
    ) -> list[ResultValue]:
        """Return the coercive-field results read off the loop.

        Args:
            x_info: The axis column, or ``None``.
            loop_info: The loop's scalar column.
            crossings: The zero-crossing fields, in sweep order.

        Returns:
            One ``ResultValue`` per crossing, plus the mean coercive field
            and the loop width when the loop crossed at least twice.
        """
        if x_info is None or not crossings:
            return []
        unit = x_info.unit
        results = [
            ResultValue(
                name=f"Coercive field (crossing {k + 1})",
                value=crossing,
                unit=unit,
                note=(
                    f"{x_info.name} where the normalised {loop_info.name} loop "
                    f"crosses zero, by linear interpolation"
                ),
            )
            for k, crossing in enumerate(crossings)
        ]
        if len(crossings) >= 2:
            magnitudes = [abs(c) for c in crossings[:2]]
            results.append(
                ResultValue(
                    name="Coercive field",
                    value=float(np.mean(magnitudes)),
                    unit=unit,
                    uncertainty=float(abs(magnitudes[0] - magnitudes[1]) / 2.0),
                    note="mean magnitude of the first two crossings; half their difference as the uncertainty",
                )
            )
            results.append(
                ResultValue(
                    name="Loop width",
                    value=float(abs(crossings[1] - crossings[0])),
                    unit=unit,
                    note="separation of the first two crossings",
                )
            )
        return results

    @staticmethod
    def _summary(
        manifest: Mapping[str, Any],
        metadata: Mapping[str, Any],
        n_points: int,
        x_info: ColumnInfo | None,
        loop_info: ColumnInfo | None,
        n_panels: int,
        crossings: list[float],
    ) -> str:
        """Write the one paragraph the notebook entry leads with."""
        procedure = str(manifest.get("procedure") or metadata.get("procedure") or "run")
        status = str(manifest.get("status") or metadata.get("status") or "unknown")
        parts = [f"{procedure} recorded {n_points} point(s) against {axis_label(x_info)}."]
        if n_panels:
            parts.append(
                f"The montage shows {n_panels} of them, and each frame is compared "
                f"with the reference frame taken at the first point."
            )
        else:
            parts.append("The run holds no frames.")
        if loop_info is not None and n_points:
            parts.append(f"The loop is {loop_info.name} against {axis_label(x_info)}.")
        if crossings:
            unit = f" {x_info.unit}" if x_info is not None and x_info.unit else ""
            listed = ", ".join(f"{c:.3g}{unit}" for c in crossings)
            parts.append(f"The normalised loop crosses zero at {listed}.")
        parts.append(f"It ended with status {status!r}.")
        return " ".join(parts)
