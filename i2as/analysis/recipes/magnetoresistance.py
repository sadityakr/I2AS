"""magnetoresistance — R(B) from a field sweep, fitted to extract R0 and the MR.

The first physics recipe: it turns a ``FieldSweep`` run into the numbers a
transport notebook entry leads with — the zero-field resistance, the
magnetoresistance coefficient and the MR at the largest field reached — with
the fit drawn over the data and its residuals beneath.

**Where the resistance comes from**, best first:

1. A column that already IS a resistance: unit ``Ω``/``Ohm``/``V/A``, or a
   name starting ``resistance``/``R_`` (option ``resistance_column`` names
   one explicitly).
2. A voltage column (``voltage_V``, or the first column in ``V``) divided by
   the current. The current is the run's own ``current_A`` column (or the
   first column in ``A``), else the reading loop's values when the loop
   parameter is a current (``loop1_values``/``loop2_values``, from the
   manifest or the run file's own parameter record), else the
   ``current_A`` option. When one point holds readings at two or more
   distinct currents — a current-reversal loop — R is the SLOPE of V
   against I at that point, so a thermal offset cancels; with one current it
   is V/I averaged over the loop.

**When it runs.** Only when asked: by name (``run_analysis`` with
``recipe="magnetoresistance"``, the eLab tab's recipe list), or as a
procedure's configured recipe (``analysis.recipes`` in the settings file).
Discovery's default for a field sweep stays ``generic_sweep``.

**The fit** is linear least squares over every finite point (optionally only
``|B| <= fit_range_T``), with an odd term kept separate so a Hall or
misalignment pickup does not bias the even MR:

- ``model="quadratic"`` (default): ``R(B) = R0 + c1·B + c2·B²`` — ordinary
  (Lorentz) MR at low field;
- ``model="abs_linear"``: ``R(B) = R0 + c1·B + c2·|B|`` — linear MR.

One-sigma uncertainties come from the residual variance and the normal
matrix. The MR is reported from the even term alone:
``MR(B) = c2·f(B)/R0 · 100 %`` at the largest ``|B|`` fitted.

A run this recipe cannot read as R(B) — no field axis, no voltage or
resistance, no current, fewer points than parameters — is not a failure: the
report is the ``generic_sweep`` overview, with a warning saying why the fit
was not made.
"""

from __future__ import annotations

import dataclasses
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
from i2as.analysis.recipes.generic_sweep import GenericSweepRecipe
from i2as.analysis.report import ANY_PROCEDURE, AnalysisReport, ResultValue
from i2as.core.data_reader import ColumnInfo, RunSource

logger = logging.getLogger(__name__)

#: The fit models the ``model`` option accepts, each with the function of B
#: its even coefficient multiplies and the unit that coefficient carries.
MODELS: dict[str, tuple[str, str]] = {
    "quadratic": ("B²", "Ω/T²"),
    "abs_linear": ("|B|", "Ω/T"),
}

#: The model used when the ``model`` option names none, or an unknown one.
DEFAULT_MODEL = "quadratic"

#: Units that mark a column as a resistance already.
RESISTANCE_UNITS: tuple[str, ...] = ("Ω", "ohm", "ohms", "v/a")

#: Units that mark a column as a field axis.
FIELD_UNITS: tuple[str, ...] = ("T", "mT", "Oe", "G")

#: Fewest points a fit of three parameters is attempted on. Four leaves one
#: degree of freedom for the uncertainty.
MIN_FIT_POINTS = 4

FIGURE_WIDTH_IN = 7.0
FIGURE_HEIGHT_IN = 5.5


class MagnetoresistanceRecipe(AnalysisRecipe):
    """R(B) from a field sweep, fitted for R0, the MR coefficient and the MR."""

    name = "magnetoresistance"
    # Asked for, never assumed: it serves any run (and falls back to the
    # overview on one that is not a field sweep) at a priority below
    # generic_sweep, so discovery never picks it on its own. An agent names
    # it, the eLab tab offers it, or the settings make it a procedure's
    # recipe with {"FieldSweep": "magnetoresistance"}.
    procedures = (ANY_PROCEDURE,)
    priority = 5
    description = (
        "Magnetoresistance of a field sweep: R(B) fitted to R0 + c1·B + c2·B² "
        "(or c2·|B|), with R0, the MR coefficient and MR% at the largest field"
    )

    def analyse(self, run: RunSource, context: AnalysisContext) -> AnalysisReport:
        """Fit one field sweep's R(B), or fall back to the overview.

        Args:
            run: The finished run.
            context: The manifest, the options, the output directory and the
                helpers.

        Returns:
            An ``ok`` report: the fit, its figure and its parameter table; or
            the ``generic_sweep`` overview with a warning naming why no fit
            was made.
        """
        options = context.options
        model = str(options.get("model") or DEFAULT_MODEL)
        warnings: list[str] = []
        if model not in MODELS:
            warnings.append(
                f"unknown model {model!r}; fitted {DEFAULT_MODEL!r} instead "
                f"(models: {', '.join(MODELS)})"
            )
            model = DEFAULT_MODEL

        if run.n_points == 0:
            return self._fallback(run, context, "the run wrote no points")
        field_info = choose_x_column(run, context.manifest)
        if field_info is None or not _is_field(field_info):
            return self._fallback(run, context, "the run has no magnetic-field axis to fit against")
        field = np.asarray(run.read_slice(field_info.name), dtype=float).reshape(run.n_points, -1)
        field = field[:, 0] if field.shape[1] == 1 else np.nanmean(field, axis=1)
        field = field * _to_tesla(field_info.unit)

        resistance, source, reason = _resistance(run, context, field_info.name)
        if resistance is None:
            return self._fallback(run, context, reason)

        mask = np.isfinite(field) & np.isfinite(resistance)
        fit_range = _optional_positive(options.get("fit_range_T"))
        if fit_range is not None:
            mask &= np.abs(field) <= fit_range
        if int(mask.sum()) < MIN_FIT_POINTS:
            return self._fallback(
                run,
                context,
                f"only {int(mask.sum())} finite point(s) to fit; the fit needs "
                f"at least {MIN_FIT_POINTS}",
            )

        fit = _fit(field[mask], resistance[mask], model)
        figures: list[Any] = []
        try:
            figures.append(self._figure(context, field, resistance, mask, fit, field_info))
        except AnalysisError as exc:
            warnings.append(str(exc))

        return AnalysisReport(
            summary=(self._summary(context, fit, source, fit_range),),
            results=tuple(self._results(fit, source)),
            figures=tuple(figures),
            tables=(self._table(context, fit),),
            tags=("magnetoresistance",),
            warnings=tuple(warnings),
        )

    # ── Parts of the report ───────────────────────────────────────────────

    def _fallback(self, run: RunSource, context: AnalysisContext, reason: str) -> AnalysisReport:
        """Answer with the overview, saying why no fit was made.

        Args:
            run: The run.
            context: The context.
            reason: Why the fit was not made.

        Returns:
            The ``generic_sweep`` report with one more warning.
        """
        logger.info("magnetoresistance: no fit for run %s: %s", context.run_id, reason)
        overview = GenericSweepRecipe().analyse(run, context)
        return dataclasses.replace(
            overview,
            warnings=(f"no magnetoresistance fit: {reason}", *overview.warnings),
        )

    def _figure(
        self,
        context: AnalysisContext,
        field: np.ndarray,
        resistance: np.ndarray,
        mask: np.ndarray,
        fit: _Fit,
        field_info: ColumnInfo,
    ) -> Any:
        """Draw R(B) with the fit over it and the residuals beneath.

        Args:
            context: The context whose ``pyplot()``/``figure()`` are used.
            field: Every point's field, in T.
            resistance: Every point's resistance, in Ω.
            mask: Which points were fitted.
            fit: The fit.
            field_info: The field column, for the axis label.

        Returns:
            The saved ``FigureRef``.

        Raises:
            AnalysisError: If matplotlib is absent or the figure cannot be saved.
        """
        plt = context.pyplot()
        fig, (top, bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        top.plot(field[mask], resistance[mask], ".", markersize=4, label="data (fitted)")
        if not mask.all():
            top.plot(field[~mask], resistance[~mask], ".", color="0.6", markersize=4, label="data (not fitted)")
        grid = np.linspace(float(fit.field.min()), float(fit.field.max()), 200)
        top.plot(grid, fit.predict(grid), "-", linewidth=1.5, label=f"fit ({fit.model})")
        top.set_ylabel("R (Ω)")
        top.grid(True, alpha=0.3)
        top.legend(fontsize="small")
        bottom.axhline(0.0, color="0.5", linewidth=0.8)
        bottom.plot(fit.field, fit.residuals, ".", markersize=4)
        bottom.set_ylabel("residual (Ω)")
        bottom.set_xlabel("B (T)" if field_info.unit == "T" else f"B (T, from {axis_label(field_info)})")
        bottom.grid(True, alpha=0.3)
        fig.suptitle(f"Magnetoresistance — {context.run_id}".strip(" —"))
        fig.tight_layout()
        return context.figure(
            "magnetoresistance_fit",
            fig,
            caption=f"R(B) with the {fit.label} fit; residuals below.",
        )

    def _results(self, fit: _Fit, source: str) -> list[ResultValue]:
        """Return the derived values the entry lists.

        Args:
            fit: The fit.
            source: How the resistance was obtained.

        Returns:
            R0, the even and odd coefficients, the MR at the largest field,
            the fit quality and the points used.
        """
        even_symbol, even_unit = MODELS[fit.model]
        return [
            ResultValue(
                name="Zero-field resistance R0",
                value=fit.r0,
                unit="Ω",
                uncertainty=fit.sigma[0],
                note=f"intercept of the {fit.label} fit; R from {source}",
            ),
            ResultValue(
                name=f"MR coefficient c2 (× {even_symbol})",
                value=fit.even,
                unit=even_unit,
                uncertainty=fit.sigma[2],
                note="even-in-field term",
            ),
            ResultValue(
                name="Odd-in-field term c1 (× B)",
                value=fit.odd,
                unit="Ω/T",
                uncertainty=fit.sigma[1],
                note="Hall or contact-misalignment pickup; excluded from the MR",
            ),
            ResultValue(
                name=f"MR at |B| = {fit.b_max:.4g} T",
                value=fit.mr_percent,
                unit="%",
                uncertainty=fit.mr_sigma_percent,
                note="(R_even(B_max) − R0)/R0 from the fit",
            ),
            ResultValue(name="Fit R²", value=fit.r_squared, note="coefficient of determination"),
            ResultValue(name="Points fitted", value=int(fit.field.size)),
        ]

    def _table(self, context: AnalysisContext, fit: _Fit) -> Any:
        """Build the fit-parameter table.

        Args:
            context: The context whose ``table()`` applies the caps.
            fit: The fit.

        Returns:
            One row per fitted parameter, with its one-sigma uncertainty.
        """
        even_symbol, even_unit = MODELS[fit.model]
        rows = [
            ["R0", _g(fit.r0), _g(fit.sigma[0]), "Ω"],
            ["c1 (× B)", _g(fit.odd), _g(fit.sigma[1]), "Ω/T"],
            [f"c2 (× {even_symbol})", _g(fit.even), _g(fit.sigma[2]), even_unit],
        ]
        return context.table(
            f"Fit of R(B) = {fit.label}: parameters and one-sigma uncertainties.",
            ["Parameter", "Value", "± 1σ", "Unit"],
            rows,
        )

    def _summary(
        self,
        context: AnalysisContext,
        fit: _Fit,
        source: str,
        fit_range: float | None,
    ) -> str:
        """Write the paragraph the entry leads with.

        Args:
            context: The context, for the procedure name.
            fit: The fit.
            source: How the resistance was obtained.
            fit_range: The ``|B|`` limit applied, or ``None``.

        Returns:
            One paragraph: what was fitted, over what, and what came out.
        """
        procedure = str(context.manifest.get("procedure") or "The field sweep")
        window = (
            f" within |B| ≤ {fit_range:.4g} T" if fit_range is not None else ""
        )
        return (
            f"{procedure}: R(B) (from {source}) was fitted to {fit.label} over "
            f"{fit.field.size} points between {fit.field.min():.4g} T and "
            f"{fit.field.max():.4g} T{window}. R0 = {fit.r0:.6g} ± {fit.sigma[0]:.2g} Ω, "
            f"and the magnetoresistance reaches {fit.mr_percent:.4g} ± "
            f"{fit.mr_sigma_percent:.2g} % at |B| = {fit.b_max:.4g} T "
            f"(fit R² = {fit.r_squared:.4f}). The odd-in-field term is "
            f"{fit.odd:.3g} ± {fit.sigma[1]:.2g} Ω/T."
        )


# ── The fit ───────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class _Fit:
    """One least-squares fit of R(B) and everything derived from it."""

    model: str
    field: np.ndarray
    coefficients: np.ndarray
    sigma: tuple[float, float, float]
    residuals: np.ndarray
    r_squared: float

    @property
    def r0(self) -> float:
        """The intercept, Ω."""
        return float(self.coefficients[0])

    @property
    def odd(self) -> float:
        """The odd-in-field coefficient, Ω/T."""
        return float(self.coefficients[1])

    @property
    def even(self) -> float:
        """The even-in-field (MR) coefficient."""
        return float(self.coefficients[2])

    @property
    def label(self) -> str:
        """The model as a formula."""
        return f"R0 + c1·B + c2·{MODELS[self.model][0]}"

    @property
    def b_max(self) -> float:
        """The largest |B| fitted, T."""
        return float(np.max(np.abs(self.field)))

    def _even_basis(self, field: np.ndarray | float) -> np.ndarray:
        values = np.asarray(field, dtype=float)
        return values**2 if self.model == "quadratic" else np.abs(values)

    def predict(self, field: np.ndarray) -> np.ndarray:
        """Evaluate the fitted R at the given fields.

        Args:
            field: Fields, T.

        Returns:
            The fitted resistance, Ω.
        """
        return self.r0 + self.odd * field + self.even * self._even_basis(field)

    @property
    def mr_percent(self) -> float:
        """The even MR at ``b_max``, percent of R0."""
        if self.r0 == 0.0:
            return float("nan")
        return float(self.even * self._even_basis(self.b_max) / self.r0 * 100.0)

    @property
    def mr_sigma_percent(self) -> float:
        """The MR's one-sigma uncertainty, from c2's and R0's (uncorrelated)."""
        if self.r0 == 0.0:
            return float("nan")
        basis = float(self._even_basis(self.b_max))
        relative = np.hypot(
            self.sigma[2] / self.even if self.even else 0.0,
            self.sigma[0] / self.r0,
        )
        return float(abs(self.even * basis / self.r0 * 100.0) * relative)


def _fit(field: np.ndarray, resistance: np.ndarray, model: str) -> _Fit:
    """Fit ``R = R0 + c1·B + c2·g(B)`` by linear least squares.

    Args:
        field: The fitted points' field, T.
        resistance: Their resistance, Ω.
        model: A key of ``MODELS``.

    Returns:
        The fit, with one-sigma uncertainties from the residual variance.
    """
    even = field**2 if model == "quadratic" else np.abs(field)
    design = np.column_stack([np.ones_like(field), field, even])
    coefficients, *_ = np.linalg.lstsq(design, resistance, rcond=None)
    residuals = resistance - design @ coefficients
    rss = float(residuals @ residuals)
    dof = max(field.size - design.shape[1], 1)
    try:
        covariance = rss / dof * np.linalg.inv(design.T @ design)
        sigma = tuple(float(np.sqrt(max(v, 0.0))) for v in np.diag(covariance))
    except np.linalg.LinAlgError:
        sigma = (float("nan"),) * 3
    tss = float(np.sum((resistance - resistance.mean()) ** 2))
    r_squared = 1.0 - rss / tss if tss > 0 else 1.0
    return _Fit(
        model=model,
        field=field,
        coefficients=coefficients,
        sigma=sigma,  # type: ignore[arg-type]
        residuals=residuals,
        r_squared=float(r_squared),
    )


# ── Reading R off the run ─────────────────────────────────────────────────


def _resistance(
    run: RunSource, context: AnalysisContext, field_name: str
) -> tuple[np.ndarray | None, str, str]:
    """Return one resistance per point, and how it was obtained.

    Args:
        run: The run.
        context: The context — options and manifest parameters.
        field_name: The field column, never taken as a reading.

    Returns:
        ``(R per point, source description, "")``, or ``(None, "", reason)``
        when the run holds nothing a resistance can be made from.
    """
    options = context.options
    columns = {info.name: info for info in measured_columns(run, exclude=[field_name])}
    n = run.n_points

    def pick(explicit: str, predicate: Any) -> ColumnInfo | None:
        if explicit and explicit in columns:
            return columns[explicit]
        return next((info for info in columns.values() if predicate(info)), None)

    r_info = pick(str(options.get("resistance_column") or ""), _is_resistance)
    if r_info is not None:
        values = _per_point(run.read_slice(r_info.name), n)
        return np.nanmean(values, axis=1), f"the {r_info.name!r} column", ""

    v_info = pick(
        str(options.get("voltage_column") or ""),
        lambda info: info.name == "voltage_V" or (info.unit == "V" and not _is_error(info)),
    )
    if v_info is None:
        return None, "", "the run has neither a resistance nor a voltage column"
    voltage = _per_point(run.read_slice(v_info.name), n)

    i_info = pick(
        str(options.get("current_column") or ""),
        lambda info: info.name == "current_A" or (info.unit == "A" and not _is_error(info)),
    )
    if i_info is not None:
        current = _per_point(run.read_slice(i_info.name), n)
        source = f"{v_info.name!r} / {i_info.name!r}"
    else:
        # The manifest's parameters when the caller passed them; the run
        # file's own record of them otherwise — the file always has them.
        params = context.manifest.get("params") or run.read_metadata().get("params")
        loop_current = _loop_currents(params, voltage.shape[1])
        fixed = _optional_nonzero(options.get("current_A"))
        if loop_current is not None:
            current = np.broadcast_to(loop_current, voltage.shape)
            source = f"{v_info.name!r} / the reading loop's currents"
        elif fixed is not None:
            current = np.full(voltage.shape, fixed)
            source = f"{v_info.name!r} / current_A = {fixed:g} A (option)"
        else:
            return None, "", (
                f"the run has a voltage column ({v_info.name!r}) but no current: "
                f"no current column, no current reading loop, and no current_A option"
            )
    if current.shape[1] != voltage.shape[1]:
        current = np.broadcast_to(np.nanmean(current, axis=1, keepdims=True), voltage.shape)
    return _v_over_i(voltage, current), source, ""


def _v_over_i(voltage: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return R per point: the V–I slope where a point spans two currents, else V/I.

    Args:
        voltage: ``(n, k)`` voltages.
        current: ``(n, k)`` currents.

    Returns:
        ``(n,)`` resistances; NaN where a point has no usable reading.
    """
    out = np.full(voltage.shape[0], np.nan)
    for index in range(voltage.shape[0]):
        v, i = voltage[index], current[index]
        ok = np.isfinite(v) & np.isfinite(i) & (i != 0)
        if not ok.any():
            continue
        v, i = v[ok], i[ok]
        if np.ptp(i) > 0:
            out[index] = float(np.polyfit(i, v, 1)[0])
        else:
            out[index] = float(np.mean(v / i))
    return out


def _per_point(values: Any, n: int) -> np.ndarray:
    """Reshape one column to ``(n, readings per point)``.

    Args:
        values: The column as read — leading axis the sweep point.
        n: The number of points.

    Returns:
        A float array with the loop axes flattened.
    """
    array = np.asarray(values, dtype=float)
    return array.reshape(n, -1) if array.size else np.full((n, 1), np.nan)


def _loop_currents(params: object, k: int) -> np.ndarray | None:
    """Return the reading loop's current per reading, when a loop sets a current.

    Only a loop whose parameter NAMES a current counts (``loop1_parameter`` /
    ``loop2_parameter`` containing ``current``): a loop over numeric channel
    numbers must never be divided into a voltage.

    Args:
        params: The run's procedure parameters.
        k: Readings per point (``n_loop1 × n_loop2``, flattened row-major).

    Returns:
        ``(k,)`` currents in the flattened loop order, or ``None``.
    """
    if not isinstance(params, Mapping):
        return None
    values1 = _floats(params.get("loop1_values"))
    values2 = _floats(params.get("loop2_values"))
    n1, n2 = len(values1) or 1, len(values2) or 1
    if n1 * n2 != k:
        return None
    if values1 and "current" in str(params.get("loop1_parameter") or "").lower():
        return np.repeat(np.asarray(values1), n2)
    if values2 and "current" in str(params.get("loop2_parameter") or "").lower():
        return np.tile(np.asarray(values2), n1)
    return None


def _floats(value: object) -> list[float]:
    """Return a list of floats, or ``[]`` for anything that is not one."""
    if not isinstance(value, (list, tuple)):
        return []
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return []


def _is_field(info: ColumnInfo) -> bool:
    """Whether a column is a magnetic field."""
    return info.unit in FIELD_UNITS or "field" in info.name.lower()


def _to_tesla(unit: str) -> float:
    """Return the factor that turns a field in *unit* into tesla."""
    return {"mT": 1e-3, "Oe": 1e-4, "G": 1e-4}.get(unit, 1.0)


def _is_resistance(info: ColumnInfo) -> bool:
    """Whether a column is a resistance already."""
    name = info.name.lower()
    return not _is_error(info) and (
        info.unit.lower() in RESISTANCE_UNITS
        or name.startswith("resistance")
        or info.name.startswith("R_")
    )


def _is_error(info: ColumnInfo) -> bool:
    """Whether a column is the uncertainty of another."""
    return info.name.endswith("_error")


def _optional_positive(value: object) -> float | None:
    """Return a positive float option, or ``None``."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _optional_nonzero(value: object) -> float | None:
    """Return a non-zero float option, or ``None``."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number != 0 and np.isfinite(number) else None


def _g(value: float) -> float | None:
    """Return a JSON-safe number for a table cell."""
    return float(f"{value:.6g}") if np.isfinite(value) else None
