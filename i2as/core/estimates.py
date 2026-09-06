"""How long a run will take — the duration-estimate standard.

Pure policy, in the shape of ``conditions.py`` and ``availability.py``: plain
functions over declared values, with no Station, no Orchestrator, no Qt and no
hardware anywhere near them. Everything this module needs arrives as an
argument, which is what lets an estimate be computed in a test, in a script,
or in a client with no GUI — and what keeps it free of consequence, since
estimating a run must never dispatch, open, or claim anything.

**The standard.** An estimate is assembled from exactly two contributions, and
each side owns the half it actually knows:

* **The run** contributes its per-point cost — how many points it will take,
  the one-off setup wait, the settle wait before each later point, and how
  long one measurement lasts — through the single hook
  ``estimate_step_seconds() -> StepCost`` (see ``BaseProcedure``). A procedure
  therefore never has to know how fast a magnet moves.
* **The setup** contributes its nominal ramp rates, as
  ``{vi_name: units_per_minute}`` (``Station.nominal_ramp_rates()``). Combined
  with the run's own ``planned_targets()`` — the setpoints it declares it will
  command — that is what turns a sweep into a time.

**Assumptions are explicit, never silently zero.** Anything the model could
not derive — a VI with no declared rate, a measurement whose duration nothing
declares, a run with no cost model at all — is named in the returned
``DurationEstimate.assumptions`` instead of quietly counting as instant. An
estimate a reader cannot qualify is worse than no estimate, because it looks
like a promise.

Two modelling choices are part of the standard, both stated as assumptions on
every estimate:

* **Concurrent ramps.** The Orchestrator dispatches a point's targets together
  and waits for all of them, so the ramp time of a run is the time of its
  SLOWEST instrument, not the sum across instruments.
* **The first approach is not counted.** Where the hardware stands when the
  run starts cannot be known without reading it, so the ramp from there to the
  first setpoint is left out. Every ramp between the run's own declared
  setpoints is counted in full.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from i2as.core.plan import DurationEstimate, StepCost

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

#: The phase keys every estimate reports, in the order a run pays them. A
#: client renders them; it never switches on them (a future phase is additive).
PHASE_SETUP = "setup"
PHASE_RAMP = "ramp"
PHASE_SETTLE = "settle"
PHASE_MEASURE = "measure"

#: The two modelling assumptions every estimate carries (see the module
#: docstring): they are properties of the model, not of one run, so they are
#: stated on every estimate rather than re-derived per caller.
_CONCURRENCY_ASSUMPTION = (
    "ramps on different instruments run concurrently, so the ramp time is the "
    "slowest instrument's, not the sum"
)
_FIRST_APPROACH_ASSUMPTION = (
    "the ramp from wherever the hardware stands now to the run's first "
    "setpoint is not counted — it cannot be known without reading hardware"
)


def estimate_duration(
    run: Any, ramp_rates: Mapping[str, float] | None = None
) -> DurationEstimate:
    """Estimate how long *run* will take, and say what the answer assumed.

    Takes the BUILT run rather than a class and a parameter dict: the sweep a
    run will actually walk, the waits it will actually pay and the setpoints it
    will actually command are properties of the built instance (which is
    exactly what ``session.run_queue.validate_run()`` already has in hand,
    since it builds the run headlessly to check it). Purely declarative — it
    reads the run's own declarations and touches no instrument.

    Duck-typed on purpose: this module sits below L4 and must not import
    ``BaseProcedure`` (contracts C5/C6 keep the layers apart), so a run that
    declares neither hook simply gets an estimate that says so.

    Args:
        run: A built procedure. Read through two optional
            declarations — ``estimate_step_seconds()`` for its per-point cost
            and ``planned_targets()`` for the setpoints it will command.
        ramp_rates: ``{vi_name: units_per_minute}`` for the setup the run will
            drive, normally ``Station.nominal_ramp_rates()``. A VI missing
            from it contributes no ramp time and one explicit assumption.

    Returns:
        The ``DurationEstimate``: a total, the per-phase breakdown
        (``setup`` / ``ramp`` / ``settle`` / ``measure``), and every
        assumption behind it — never an empty assumption list.
    """
    assumptions: list[str] = []
    cost = _step_cost(run, assumptions)
    ramp_s = _ramp_seconds(run, dict(ramp_rates or {}), assumptions)
    later_points = max(cost.points - 1, 0)
    phases = {
        PHASE_SETUP: cost.setup_s,
        PHASE_RAMP: ramp_s,
        PHASE_SETTLE: cost.settle_s * later_points,
        PHASE_MEASURE: cost.measure_s * cost.points,
    }
    return DurationEstimate(
        total_s=sum(phases.values()),
        phases=phases,
        assumptions=tuple(assumptions),
    )


def _step_cost(run: Any, assumptions: list[str]) -> StepCost:
    """Read the run's own per-point cost model, or state why there is none.

    Args:
        run: The built run.
        assumptions: Collected assumptions, appended to in place.

    Returns:
        The run's ``StepCost``, or an all-zero one when it declares none or
        its hook misbehaved.
    """
    name = type(run).__name__
    hook = getattr(run, "estimate_step_seconds", None)
    if not callable(hook):
        assumptions.append(
            f"{name} declares no cost model (no estimate_step_seconds()), so "
            f"only its ramps are estimated"
        )
        return StepCost()
    try:
        cost = hook()
    except Exception:  # noqa: BLE001 — an estimate never raises at its caller
        logger.exception("estimate_duration: %s.estimate_step_seconds() failed", name)
        assumptions.append(
            f"{name}'s cost model could not be read, so only its ramps are "
            f"estimated"
        )
        return StepCost()
    if not isinstance(cost, StepCost):
        assumptions.append(
            f"{name}.estimate_step_seconds() did not return a StepCost, so its "
            f"per-point cost is not counted"
        )
        return StepCost()
    assumptions.extend(cost.assumptions)
    return cost


def _ramp_seconds(
    run: Any, ramp_rates: dict[str, float], assumptions: list[str]
) -> float:
    """Estimate the run's total ramping time from its declared setpoints.

    Per VI: the summed distance between consecutive declared setpoints,
    divided by that VI's nominal rate. Across VIs: the maximum, since a
    point's targets are dispatched together and waited for together.

    Args:
        run: The built run.
        ramp_rates: ``{vi_name: units_per_minute}``.
        assumptions: Collected assumptions, appended to in place.

    Returns:
        Seconds of ramping, ``0.0`` when the run declares no setpoints.
    """
    name = type(run).__name__
    planned = getattr(run, "planned_targets", None)
    if not callable(planned):
        assumptions.append(
            f"{name} declares no setpoints (no planned_targets()), so no ramp "
            f"time is counted"
        )
        return 0.0
    try:
        targets: Mapping[str, Any] = planned()
    except Exception:  # noqa: BLE001 — an estimate never raises at its caller
        logger.exception("estimate_duration: %s.planned_targets() failed", name)
        assumptions.append(
            f"{name}'s setpoints could not be read, so no ramp time is counted"
        )
        return 0.0

    assumptions.append(_CONCURRENCY_ASSUMPTION)
    assumptions.append(_FIRST_APPROACH_ASSUMPTION)
    slowest = 0.0
    for vi_name, values in sorted(targets.items()):
        numbers = [float(value) for value in values]
        distance = sum(
            abs(numbers[index + 1] - numbers[index])
            for index in range(len(numbers) - 1)
        )
        if distance <= 0:
            continue
        rate = ramp_rates.get(vi_name)
        if not rate or rate <= 0:
            assumptions.append(
                f"{vi_name} declares no ramp rate, so the {distance:g} units it "
                f"would travel are not counted"
            )
            continue
        slowest = max(slowest, distance / rate * 60.0)
    return slowest
