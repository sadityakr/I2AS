"""sweep_builder — piecewise, CSV-custom, and hysteresis sweep construction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Any

from i2as.core.plan import ConditionalGroup, ParamSpec, When


@dataclass
class SweepSegment:
    """One contiguous piece of a piecewise sweep: evenly spaced values from
    ``start`` to ``end`` at approximately ``step`` spacing.

    A ``dataclass`` is used instead of a plain dict so a segment is typo-proof
    (``segment.step`` instead of ``segment["step"]``) — ``@dataclass`` auto-
    generates ``__init__``, ``__repr__``, and ``__eq__`` from the three fields
    declared below, which is why the class body has no methods of its own.

    Attributes:
        start: First value of this segment, in the same units as the sweep.
        end: Last value of this segment.
        step: Requested spacing between points. The actual spacing used is
            ``(end - start) / n`` for the smallest ``n`` that keeps spacing at
            or below this value, so the segment lands on ``end`` exactly.
    """

    start: float
    end: float
    step: float


def build_piecewise_sweep(segments: list[SweepSegment]) -> list[float]:
    """Build a sweep with a different step size per sub-range.

    Example — coarse steps everywhere except a fine region near zero::

        build_piecewise_sweep([
            SweepSegment(start=1.0, end=0.1, step=0.1),      # 1 T -> 0.1 T, 0.1 T steps
            SweepSegment(start=0.1, end=-0.1, step=0.01),    # 0.1 T -> -0.1 T, 10 mT steps
            SweepSegment(start=-0.1, end=-1.0, step=0.1),    # -0.1 T -> -1 T, 0.1 T steps
        ])

    Segments must be contiguous: segment ``i``'s ``start`` must equal segment
    ``i - 1``'s ``end``. The shared boundary point between two segments is
    included only once. A single segment reduces to an ordinary linear sweep.

    Args:
        segments: Ordered, contiguous list of ``SweepSegment``.

    Returns:
        Flat list of sweep values, in the order the segments were given.
        Empty list if *segments* is empty.

    Raises:
        ValueError: If any segment has a non-positive ``step``, or if
            segments are not contiguous.
    """
    if not segments:
        return []

    points: list[float] = []
    for i, seg in enumerate(segments):
        if seg.step <= 0:
            raise ValueError(f"Segment {i} step must be positive, got {seg.step}")
        if i > 0 and seg.start != segments[i - 1].end:
            raise ValueError(
                f"Segment {i} start ({seg.start}) does not match segment "
                f"{i - 1} end ({segments[i - 1].end}); segments must be contiguous"
            )
        span = seg.end - seg.start
        n_steps = max(1, round(abs(span) / seg.step))
        actual_step = span / n_steps
        points.extend(seg.start + k * actual_step for k in range(n_steps))

    points.append(segments[-1].end)
    return points


def load_custom_sweep_csv(file_path: str) -> list[float]:
    """Load a sweep array from a single-column CSV file, one numeric value per row.

    Blank rows are skipped. Intended for arbitrary, non-uniform field (or other
    swept-variable) lists that cannot be expressed as start/end/step segments.

    Args:
        file_path: Path to the CSV file.

    Returns:
        List of parsed float values, in file order.

    Raises:
        ValueError: If a non-blank row has more than one column, if a value
            cannot be parsed as a float, or if the file has no values at all.
        FileNotFoundError: If *file_path* does not exist.
    """
    values: list[float] = []
    with open(file_path, encoding="utf-8", newline="") as f:
        for row_num, row in enumerate(csv.reader(f), start=1):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            if len(row) != 1:
                raise ValueError(
                    f"{file_path}: row {row_num} must contain exactly one "
                    f"column, found {len(row)}"
                )
            try:
                values.append(float(row[0].strip()))
            except ValueError as exc:
                raise ValueError(
                    f"{file_path}: row {row_num} value {row[0]!r} is not a number"
                ) from exc

    if not values:
        raise ValueError(f"{file_path}: no sweep values found")
    return values


def apply_hysteresis(values: list[float]) -> list[float]:
    """Extend a one-directional sweep into a forward+backward hysteresis loop.

    Appends the reverse of *values*, excluding its last element, so the
    turning point at the end of the forward leg is not measured twice.
    E.g. ``[-1, 0, 1]`` becomes ``[-1, 0, 1, 0, -1]``.

    Args:
        values: A one-directional sweep array (e.g. from
            ``build_piecewise_sweep()`` or ``load_custom_sweep_csv()``).

    Returns:
        The forward sweep followed by the backward sweep. Returned unchanged
        if *values* has fewer than 2 points (nothing to reverse).
    """
    if len(values) < 2:
        return list(values)
    return list(values) + list(reversed(values[:-1]))


@dataclass(frozen=True)
class SweepAxis:
    """Declares the one quantity a Procedure sweeps over.

    A ``dataclass`` again (see ``SweepSegment`` above for why): a plain,
    typo-proof bundle of fields with no behaviour of its own. ``frozen=True``
    makes it immutable — a Procedure class attribute is shared by every
    instance, so nothing should be able to mutate it after the class is
    defined.

    Declaring one ``SweepAxis`` as a Procedure's ``sweep_axis`` class
    attribute is enough to get a working ``_build_sweep_array()`` (linear,
    segments, or CSV, all with optional hysteresis) for free from
    ``BaseProcedure`` — see ``core/procedure.py`` — and a matching
    mode-selector widget for free in the GUI — see
    ``gui/sweep_axis_widget.py``. No procedure-specific GUI code and no
    per-procedure sweep-array code is needed either way.

    Attributes:
        key: Parameter-name prefix, e.g. ``"field"``. Generates the hidden
            parameters ``{key}_mode``, ``{key}_start``, ``{key}_end``,
            ``{key}_steps``, ``{key}_segments``, ``{key}_csv_path``,
            ``{key}_hysteresis``.
        unit: Physical unit shown in the GUI, e.g. ``"T"``.
        data_key: The HDF5/data-dict column name for the measured value at
            each sweep point, e.g. ``"field_T"``.
        description: Human-readable label, e.g. ``"Magnetic field"``.
        default_start: Default value of ``{key}_start``.
        default_end: Default value of ``{key}_end``.
        default_steps: Default value of ``{key}_steps``.
    """

    key: str
    unit: str
    data_key: str
    description: str
    default_start: float = 0.0
    default_end: float = 1.0
    default_steps: int = 101


#: The three shapes a sweep axis's points can take, as the ``{key}_mode``
#: choices: the label a form shows, and the value a run carries.
SWEEP_MODES: dict[str, str] = {"Linear": "linear", "Segments": "segments", "CSV": "csv"}


def sweep_axis_param_specs(axis: SweepAxis) -> dict[str, ParamSpec]:
    """Return the seven hidden ``ParamSpec``s a declared ``SweepAxis`` adds.

    Every value ``build_axis_sweep()`` reads is declared here, as data, so
    every surface — the GUI form, the agent's ``describe_procedure``, the
    validator — sees the same sweep the same way and none has to
    special-case it:

    * ``{key}_mode`` — structural, one of ``SWEEP_MODES``; which of the three
      blocks below is part of the form (see ``sweep_axis_form_blocks()``).
    * ``{key}_start`` / ``{key}_end`` / ``{key}_steps`` — the linear sweep.
    * ``{key}_segments`` — the piecewise sweep as a TABLE (``type=list``):
      one ``{start, end, step}`` row per contiguous segment, exactly the
      shape ``build_piecewise_sweep()`` consumes.
    * ``{key}_csv_path`` — the file of values, ``widget_hint="file"``.
    * ``{key}_hysteresis`` — sweep back to the start after the end.

    Args:
        axis: The Procedure's declared sweep axis.

    Returns:
        Dict of ``{param_name: ParamSpec}`` for the seven axis parameters.
    """
    k = axis.key
    lower_desc = axis.description[0].lower() + axis.description[1:]
    span = abs(axis.default_end - axis.default_start)
    default_step = span / max(axis.default_steps - 1, 1) if span else 1.0
    return {
        f"{k}_mode": ParamSpec(
            type=str,
            default="linear",
            choices=dict(SWEEP_MODES),
            structural=True,
            description=(
                f"How the {lower_desc} points are generated: linear "
                f"(start/end/steps), segments (a breakpoint table), or csv "
                f"(a file of values)"
            ),
        ),
        f"{k}_start": ParamSpec(
            type=float,
            default=axis.default_start,
            unit=axis.unit,
            description=f"Starting {lower_desc}",
        ),
        f"{k}_end": ParamSpec(
            type=float,
            default=axis.default_end,
            unit=axis.unit,
            description=f"Ending {lower_desc}",
        ),
        f"{k}_steps": ParamSpec(
            type=int,
            default=axis.default_steps,
            min=2,
            description=f"Number of {lower_desc} steps",
        ),
        f"{k}_segments": ParamSpec(
            type=list,
            default=[],
            columns={
                "start": ParamSpec(
                    type=float,
                    default=axis.default_start,
                    unit=axis.unit,
                    description=f"Where this segment's {lower_desc} starts",
                ),
                "end": ParamSpec(
                    type=float,
                    default=axis.default_end,
                    unit=axis.unit,
                    description=(
                        f"Where this segment's {lower_desc} ends; the next "
                        f"segment starts here"
                    ),
                ),
                "step": ParamSpec(
                    type=float,
                    default=default_step,
                    unit=axis.unit,
                    description="Spacing between points within this segment",
                ),
            },
            description=(
                f"The {lower_desc} sweep as contiguous segments, each with "
                f"its own step, in sweep order"
            ),
        ),
        f"{k}_csv_path": ParamSpec(
            type=str,
            default="",
            widget_hint="file",
            description=f"Path to the CSV of {lower_desc} values, in csv mode",
        ),
        f"{k}_hysteresis": ParamSpec(
            type=bool,
            default=False,
            description=(
                f"Sweep the {lower_desc} back down to the start after reaching "
                f"the end"
            ),
        ),
    }


def sweep_axis_form_blocks(
    axis: SweepAxis, *, key: str = "sweep", title: str = "Sweep"
) -> tuple[ConditionalGroup, ...]:
    """Declare a sweep axis's parameters as guarded blocks of one group.

    The form half of the axis, in the **conditional-parameter standard**'s
    own terms: the mode selector unconditionally, then ONE of the three
    value blocks guarded on the mode it belongs to, then hysteresis. Every
    block shares *key*, so they merge into the one Sweep group beside a
    procedure's own ``sweep_parameters``, and a client that can resolve a
    form can render — and reflect — a sweep of any shape with no widget of
    its own.

    Args:
        axis: The Procedure's declared sweep axis.
        key: The rendered group's key. ``"sweep"`` puts the axis in the
            Sweep column.
        title: The rendered group's heading.

    Returns:
        The blocks, in render order.
    """
    k = axis.key
    specs = sweep_axis_param_specs(axis)
    mode = f"{k}_mode"
    return (
        ConditionalGroup(key=key, title=title, params={mode: specs[mode]}),
        ConditionalGroup(
            key=key,
            title=title,
            params={name: specs[name] for name in (f"{k}_start", f"{k}_end", f"{k}_steps")},
            when=When({mode: ("linear",)}),
        ),
        ConditionalGroup(
            key=key,
            title=title,
            params={f"{k}_segments": specs[f"{k}_segments"]},
            when=When({mode: ("segments",)}),
        ),
        ConditionalGroup(
            key=key,
            title=title,
            params={f"{k}_csv_path": specs[f"{k}_csv_path"]},
            when=When({mode: ("csv",)}),
        ),
        ConditionalGroup(
            key=key, title=title, params={f"{k}_hysteresis": specs[f"{k}_hysteresis"]}
        ),
    )


def build_axis_sweep(axis: SweepAxis, params: dict[str, Any]) -> list[float]:
    """Build a sweep array for *axis* from a Procedure's ``self._params``.

    Reads ``params["{axis.key}_mode"]`` to pick the sweep shape:

    - ``"linear"`` (default): evenly spaced from ``{key}_start`` to
      ``{key}_end`` in ``{key}_steps`` points.
    - ``"segments"``: ``build_piecewise_sweep()`` over ``{key}_segments``
      (a list of ``SweepSegment`` or ``{"start", "end", "step"}`` dicts —
      normalized to plain dicts in *params* in place, since ``SweepSegment``
      is not JSON-serializable and *params* is later saved as HDF5 metadata).
    - ``"csv"``: ``load_custom_sweep_csv()`` over ``{key}_csv_path``.

    Then, if ``params["{key}_hysteresis"]`` is truthy, wraps the result with
    ``apply_hysteresis()``.

    Args:
        axis: The Procedure's declared sweep axis.
        params: The Procedure's ``self._params`` dict (mutated in place to
            normalize ``{key}_segments``, if present).

    Returns:
        List of sweep point values.
    """
    k = axis.key
    mode = params.get(f"{k}_mode", "linear")

    if mode == "csv":
        path = str(params.get(f"{k}_csv_path") or "").strip()
        if not path:
            raise ValueError(
                f"{k}: csv sweep mode selected but {k}_csv_path names no file"
            )
        try:
            base = load_custom_sweep_csv(path)
        except OSError as exc:
            raise ValueError(f"{k}: cannot read the sweep CSV {path!r}: {exc}") from exc
    elif mode == "segments":
        segments = [
            seg if isinstance(seg, SweepSegment) else SweepSegment(**seg)
            for seg in params.get(f"{k}_segments", [])
        ]
        if not segments:
            raise ValueError(
                f"{k}: segments sweep mode selected but {k}_segments holds no segment"
            )
        params[f"{k}_segments"] = [
            {"start": s.start, "end": s.end, "step": s.step} for s in segments
        ]
        base = build_piecewise_sweep(segments)
    else:
        start = float(params[f"{k}_start"])
        end = float(params[f"{k}_end"])
        steps = max(int(params[f"{k}_steps"]), 1)
        if steps == 1:
            base = [start]
        else:
            base = [start + i * (end - start) / (steps - 1) for i in range(steps)]

    if params.get(f"{k}_hysteresis", False):
        base = apply_hysteresis(base)

    return base
