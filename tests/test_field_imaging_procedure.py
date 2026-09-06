"""FieldImaging: the imaging sweep, headless, on the sim_imaging station.

Builds the run the way every client does (``run_builder.build_procedure``),
checks the plan objects it hands the Orchestrator — the saturation
pre-step in particular — and drives one short run through a real
Orchestrator to a data file whose frames come back through ``read_image``.
"""

from __future__ import annotations

import numpy as np
import pytest

from i2as.core.data_reader import ROLE_IMAGE, open_run
from i2as.core.exceptions import I2ASConfigError
from i2as.core.gates import Gate
from i2as.core.orchestrator import Orchestrator, OrchestratorState
from i2as.core.plan import Command, PhasePlan, StepPlan, Target
from i2as.core.run_builder import build_procedure
from i2as.core.station import build_station
from i2as.procedures.field_imaging import FieldImaging

IMAGING_CONFIG = "i2as/configs/sim_imaging"
CRYOSTAT_CONFIG = "i2as/configs/sim_cryostat"

SAMPLE_INFO = {"sample_name": "Domain sample", "sample_id": "IMG-001", "comments": ""}

FAST = {
    "field_start": -1.0,
    "field_end": 1.0,
    "field_steps": 5,
    "saturation_field_T": -1.5,
    "init_wait": 0.0,
    "step_wait": 0.0,
    "exposure_s": 0.01,
    "binning": 1,
    "frames_per_step": 2,
}


@pytest.fixture
def station():
    station = build_station(IMAGING_CONFIG)
    # Fast enough to drive a sweep tick-by-tick.
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []
    return station


def _build(station, tmp_path, **params):
    return build_procedure(
        FieldImaging,
        station=station,
        params={**FAST, **params},
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
    )


# ── Roles and parameters ─────────────────────────────────────────────────────


def test_roles_resolve_to_the_magnet_and_both_stage_axes(station, tmp_path):
    proc = _build(station, tmp_path)
    assert proc.role_vi("field_vi") == "magnet_z"
    assert proc.role_vi("stage_x_vi") == "stage_x"
    assert proc.role_vi("stage_y_vi") == "stage_y"
    assert proc._measurement_vi == "camera"
    assert proc.get_params()["saturation_field_T"] == -1.5


def test_stage_roles_are_told_apart_by_axis(station):
    groups = FieldImaging.get_param_groups(station)
    specs = {name: spec for group in groups for name, spec in group.params.items()}
    assert list(specs["stage_x_vi"].choices) == ["stage_x"]
    assert list(specs["stage_y_vi"].choices) == ["stage_y"]


def test_a_stage_axis_that_is_not_a_candidate_is_refused(station, tmp_path):
    with pytest.raises(I2ASConfigError):
        _build(station, tmp_path, stage_x_vi="stage_y")


def test_builds_on_a_setup_without_a_stage(tmp_path):
    """The stage roles are optional: no stage, no stage targets, no error."""
    station = build_station(CRYOSTAT_CONFIG)
    proc = build_procedure(
        FieldImaging,
        station=station,
        params={"field_steps": 3, "init_wait": 0.0, "step_wait": 0.0},
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
    )
    assert proc.role_vi("stage_x_vi") == ""
    assert proc.role_vi("stage_y_vi") == ""
    targets = proc._initial_system_targets()
    assert set(targets) == {"magnet_z"}
    assert proc._measurement_vi == "dc_measurement"


# ── The plans ────────────────────────────────────────────────────────────────


def test_initiate_ramps_to_saturation_positions_the_stage_and_arms(station, tmp_path):
    proc = _build(station, tmp_path, stage_x_m=1e-3, stage_y_m=-2e-3)
    plan = proc.initiate()
    assert isinstance(plan, PhasePlan)
    assert plan.targets["magnet_z"] == Target(-1.5)
    assert plan.targets["stage_x"] == Target(1e-3)
    assert plan.targets["stage_y"] == Target(-2e-3)
    arm = [c for c in plan.commands if c.method == "initiate_measurement"]
    assert arm == [
        Command("camera", "initiate_measurement", {"exposure_s": 0.01, "binning": 1, "frames_per_step": 2})
    ]
    assert plan.wait_s == 0.0
    proc.abort()


def test_planned_targets_cover_saturation_and_the_sweep(station, tmp_path):
    proc = _build(station, tmp_path)
    planned = proc.planned_targets()
    assert planned["magnet_z"][0] == -1.5
    assert planned["magnet_z"][1:] == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert planned["stage_x"] == [0.0]


def test_the_saturation_pre_step_is_two_initiation_gates(station, tmp_path):
    proc = _build(station, tmp_path)
    proc.initiate()
    gates = proc.initiation_gates()
    assert [g.name for g in gates] == ["from_saturation", "settle"]
    assert all(isinstance(g, Gate) for g in gates)
    magnet = station.get_vi("magnet_z")
    # The first gate's action commands the first sweep field through the
    # magnet's own set_field; its check waits for the magnet to get there.
    assert magnet.ramp_status() == "IDLE"
    first = gates[0]
    assert first.check_once() is False
    assert magnet.ramp_target() == pytest.approx(-1.0)
    assert magnet.ramp_status() == "RAMPING"
    proc.abort()


def test_without_saturation_there_are_no_gates_and_the_first_field_is_the_start(
    station, tmp_path
):
    proc = _build(station, tmp_path, saturate=False)
    assert proc.initiation_gates() == ()
    plan = proc.initiate()
    assert plan.targets["magnet_z"] == Target(-1.0)
    proc.abort()


def test_step_and_standby_plans(station, tmp_path):
    proc = _build(station, tmp_path)
    proc.initiate()
    step = proc.change_sweep_step()
    assert isinstance(step, StepPlan)
    assert step.targets == {"magnet_z": Target(-0.5)}
    plan = proc.standby()
    assert plan.targets == {}
    assert plan.commands[0] == Command("camera", "standby", {})
    assert plan.commands[-1] == Command("magnet_z", "standby", {})


def test_the_estimate_counts_the_exposures(station, tmp_path):
    proc = _build(station, tmp_path, exposure_s=0.02, frames_per_step=3)
    cost = proc.estimate_step_seconds()
    assert cost.points == 5
    assert cost.measure_s == pytest.approx(0.06)
    assert any("exposure_s" in a for a in cost.assumptions)


# ── A run through the Orchestrator ───────────────────────────────────────────


def test_a_run_writes_frames_the_reader_serves(station, tmp_path, qtbot):
    proc = _build(station, tmp_path)
    orch = Orchestrator(station, tick_interval_ms=10)
    states: list[str] = []
    orch.state_changed.connect(states.append)
    orch.run_procedure(proc)

    with qtbot.waitSignal(orch.procedure_finished, timeout=30000):
        pass

    assert orch._state == OrchestratorState.IDLE
    assert OrchestratorState.INITIATION_GATE.value in states, "the saturation pre-step ran"
    assert not station.get_vi("camera")._camera.is_armed(), "standby disarmed the camera"

    (data_file,) = tmp_path.glob("*.h5")
    with open_run(data_file) as run:
        assert run.n_points == 5
        columns = {info.name: info for info in run.list_columns()}
        assert columns["frame"].role == ROLE_IMAGE
        assert columns["frame"].shape[-2:] == (128, 128)
        for name in ("roi_mean", "roi_mean_error", "roi_std", "field_T", "stage_x_position"):
            assert name in columns
        reference = run.read_image("frame", 0)
        last = run.read_image("frame", 4)
        assert reference.shape == (128, 128) and last.shape == (128, 128)
        assert np.isfinite(reference).all()
        # From saturation at -1.5 T the sweep went -1 ... +1 T: the sample
        # switched, so the last frame is brighter than the reference frame
        # and the ROI mean follows.
        assert last.mean() > 1.5 * reference.mean()
        roi_mean = run.read_slice("roi_mean")
        assert roi_mean[-1] > 1.5 * roi_mean[0]
        assert np.all(np.diff(roi_mean) >= -1.0)
        field = run.read_slice("field_T")
        assert field == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0], abs=0.05)
        params = run.read_metadata()["params"]
        assert params["saturation_field_T"] == -1.5
        assert params["measurement_vi"] == "camera"
