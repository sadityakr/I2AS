# ---
# description: |
#   Tests for i2as.core.estimates — the duration-estimate standard. Covers
#   the two contributions (the run's StepCost hook, the setup's nominal ramp
#   rates), scaling with sweep length and with rate, the per-phase breakdown,
#   the explicit-assumptions rule, and validate_run() returning the estimate.
# last_updated: 2026-09-03
# ---

import pytest

from i2as.core.estimates import (
    PHASE_MEASURE,
    PHASE_RAMP,
    PHASE_SETTLE,
    PHASE_SETUP,
    estimate_duration,
)
from i2as.core.plan import DurationEstimate, ProbeSpec, StepCost
from i2as.core.procedure import BaseProcedure
from i2as.core.run_builder import build_procedure
from i2as.core.station import build_station
from i2as.procedures.field_sweep import FieldSweep
from i2as.session.run_queue import validate_run

CONFIG_PATH = "i2as/configs/sim_cryostat"

SAMPLE_INFO = {"sample_name": "S", "sample_id": "S-1", "comments": ""}

PARAMS = {
    "measurement_vi": "dc_measurement",
    "field_start": -1.0,
    "field_end": 1.0,
    "field_steps": 21,
    "readings_per_point": 100,
    "init_wait": 300.0,
    "step_wait": 5.0,
}


@pytest.fixture
def station():
    return build_station(CONFIG_PATH)


def _sweep(station, tmp_path, probe=None, **overrides):
    return build_procedure(
        FieldSweep,
        station=station,
        params={**PARAMS, **overrides},
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
        probe=probe,
    )


# ── The setup's half: nominal ramp rates ─────────────────────────────────────

def test_the_station_declares_a_nominal_rate_per_ramping_vi(station):
    """Rates come from config alone, in each VI's own user units per minute."""
    rates = station.nominal_ramp_rates()

    # The sim magnet's slowest segment is 1 A/min at 10 A/T.
    assert rates["magnet_z"] == pytest.approx(0.1)
    assert rates["temperature"] == pytest.approx(2.0)
    assert "dc_measurement" not in rates  # not a ramping system VI


def test_a_magnet_reports_its_slowest_segment_never_its_fastest(station):
    """An estimate built from a magnet's rate must never be optimistic."""
    magnet = station.magnet_z
    segment_rates = [
        segment["rate_A_per_min"] / magnet._amperes_per_tesla
        for segment in magnet._ramp_segments
    ]

    assert magnet.nominal_ramp_rate() == pytest.approx(min(segment_rates))


# ── The estimate itself ──────────────────────────────────────────────────────

def test_the_estimate_breaks_the_run_into_its_phases(station, tmp_path):
    """Every phase is derived from a declaration, and they sum to the total."""
    estimate = estimate_duration(
        _sweep(station, tmp_path), station.nominal_ramp_rates()
    )

    assert isinstance(estimate, DurationEstimate)
    # -1 T -> 1 T at 0.1 T/min is 20 minutes of ramping; 300 s of initial
    # settle; 20 later points at 5 s. The shipped DC VI declares no
    # seconds-valued measurement parameter, so no measurement time is
    # counted — the honest answer, and the one the assumptions state.
    assert estimate.phases[PHASE_RAMP] == pytest.approx(1200.0)
    assert estimate.phases[PHASE_SETUP] == pytest.approx(300.0)
    assert estimate.phases[PHASE_SETTLE] == pytest.approx(100.0)
    assert estimate.phases[PHASE_MEASURE] == pytest.approx(0.0)
    assert estimate.total_s == pytest.approx(sum(estimate.phases.values()))


def test_the_estimate_scales_with_sweep_length(station, tmp_path):
    """More points is more settling and more measuring, over the same ramp."""
    short = estimate_duration(
        _sweep(station, tmp_path, field_steps=11), station.nominal_ramp_rates()
    )
    long = estimate_duration(
        _sweep(station, tmp_path, field_steps=101), station.nominal_ramp_rates()
    )

    assert long.total_s > short.total_s
    assert long.phases[PHASE_SETTLE] == pytest.approx(10 * short.phases[PHASE_SETTLE])
    assert long.phases[PHASE_RAMP] == pytest.approx(short.phases[PHASE_RAMP])


def test_the_estimate_scales_with_the_ramp_rate(station, tmp_path):
    """Halve the rate the setup declares and the ramp takes twice as long."""
    rates = station.nominal_ramp_rates()
    run = _sweep(station, tmp_path)

    normal = estimate_duration(run, rates)
    slow = estimate_duration(run, {**rates, "magnet_z": rates["magnet_z"] / 2})

    assert slow.phases[PHASE_RAMP] == pytest.approx(2 * normal.phases[PHASE_RAMP])


def test_a_hysteresis_loop_costs_its_whole_path(station, tmp_path):
    """Ramp time follows the declared setpoints, not the endpoints."""
    rates = station.nominal_ramp_rates()

    one_way = estimate_duration(_sweep(station, tmp_path), rates)
    loop = estimate_duration(
        _sweep(station, tmp_path, field_hysteresis=True), rates
    )

    assert loop.phases[PHASE_RAMP] == pytest.approx(2 * one_way.phases[PHASE_RAMP])


def test_a_probes_estimate_is_the_probes_own(station, tmp_path):
    """The point of a probe is that it is cheap, and the estimate shows it."""
    rates = station.nominal_ramp_rates()

    full = estimate_duration(_sweep(station, tmp_path), rates)
    probe = estimate_duration(
        _sweep(station, tmp_path, probe=ProbeSpec(max_wait_s=0.0)), rates
    )

    assert probe.total_s < full.total_s
    assert probe.phases[PHASE_SETUP] == 0.0


# ── Assumptions are explicit, never silently zero ────────────────────────────

def test_every_estimate_names_its_assumptions(station, tmp_path):
    """An unqualified number would read as a promise; these never are."""
    estimate = estimate_duration(
        _sweep(station, tmp_path), station.nominal_ramp_rates()
    )

    assert estimate.assumptions
    joined = " ".join(estimate.assumptions)
    assert "concurrently" in joined
    assert "not counted" in joined
    # The measurement-time model states what it did with the selected VI's
    # declarations, whether or not that VI declares a per-sample delay.
    assert "no per-sample delay" in joined


def test_a_vi_with_no_declared_rate_is_named_not_ignored(station, tmp_path):
    """Ramp time nobody can derive is stated, never counted as instant."""
    estimate = estimate_duration(_sweep(station, tmp_path), ramp_rates={})

    assert estimate.phases[PHASE_RAMP] == 0.0
    assert any("magnet_z declares no ramp rate" in a for a in estimate.assumptions)


def test_a_run_with_no_cost_model_says_so(station):
    """A run that declares neither hook still gets an honest estimate."""

    class BareRun:
        pass

    estimate = estimate_duration(BareRun(), {"magnet_z": 1.0})

    assert estimate.total_s == 0.0
    assert len(estimate.assumptions) == 2
    assert all("BareRun" in assumption for assumption in estimate.assumptions)


def test_a_broken_cost_hook_never_raises_at_the_caller(station):
    """Estimating is a read: a misbehaving run degrades, it does not explode."""

    class BrokenRun:
        def estimate_step_seconds(self):
            raise RuntimeError("boom")

        def planned_targets(self):
            return {"magnet_z": [0.0, 1.0]}

    estimate = estimate_duration(BrokenRun(), {"magnet_z": 1.0})

    assert estimate.phases[PHASE_RAMP] == pytest.approx(60.0)
    assert any("could not be read" in assumption for assumption in estimate.assumptions)


def test_a_cost_hook_returning_the_wrong_type_is_reported(station):
    """The hook is typed; a run that ignores that is named in the assumptions."""

    class SloppyRun:
        def estimate_step_seconds(self):
            return 42.0

    estimate = estimate_duration(SloppyRun(), {})

    assert estimate.total_s == 0.0
    assert any("did not return a StepCost" in a for a in estimate.assumptions)


# ── The hook every procedure inherits ────────────────────────────────────────

def test_the_step_cost_comes_from_the_hooks_the_run_itself_uses(station, tmp_path):
    """The cost model reads the same waits the tick loop will pay."""
    run = _sweep(station, tmp_path)

    cost = run.estimate_step_seconds()

    assert isinstance(cost, StepCost)
    assert (cost.points, cost.setup_s, cost.settle_s) == (21, 300.0, 5.0)
    # No seconds-valued measurement parameter on the selected VI -> nothing
    # to count, and the assumption says so rather than implying zero cost.
    assert cost.measure_s == pytest.approx(0.0)
    assert any("no per-sample delay" in a for a in cost.assumptions)


def test_the_default_hook_is_honest_about_what_it_does_not_know(station, tmp_path):
    """A procedure with no declared waits says so instead of implying zero cost."""

    class BareSweep(BaseProcedure):
        name = "Bare"

        def _build_sweep_array(self):
            return [0.0, 1.0, 2.0]

    run = BareSweep(
        station=station,
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
    )

    cost = run.estimate_step_seconds()

    assert (cost.points, cost.setup_s, cost.settle_s, cost.measure_s) == (
        3,
        0.0,
        0.0,
        0.0,
    )
    assert any("BareSweep" in assumption for assumption in cost.assumptions)


# ── validate_run returns it ──────────────────────────────────────────────────

def test_validate_run_answers_with_the_estimate(station, tmp_path):
    """One call answers "may I run this?" and "how long will it take?"."""
    result = validate_run(
        FieldSweep, PARAMS, station=station, data_directory=str(tmp_path)
    )

    assert result.ok
    assert result.estimate is not None
    assert result.duration_estimate_s == result.estimate.total_s
    assert result.estimate.total_s > 0
    assert result.estimate.assumptions


def test_validate_run_estimates_the_probe_when_one_is_asked_for(station, tmp_path):
    """A probe is validated and estimated as what would actually run."""
    full = validate_run(
        FieldSweep, PARAMS, station=station, data_directory=str(tmp_path)
    )
    probe = validate_run(
        FieldSweep,
        PARAMS,
        station=station,
        data_directory=str(tmp_path),
        probe_spec={"n_points": 3, "max_wait_s": 0.0},
    )

    assert probe.ok
    assert probe.duration_estimate_s < full.duration_estimate_s




