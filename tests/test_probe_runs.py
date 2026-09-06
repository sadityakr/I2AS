# ---
# description: |
#   Tests for probe runs — the cheap reduced variant of a requested run.
#   Covers the ProbeSpec contract (frozen, validated, JSON round trip), the
#   reduction rules BaseProcedure.apply_probe()/SweepMeasureProcedure apply
#   (sweep subsampling, wait caps, averaging caps, run_kind), the run_builder
#   probe keyword, and the probe tag reaching the HDF5 file and data_reader.
# last_updated: 2026-09-03
# ---

import json

import pytest

from i2as.core import events as ev
from i2as.core.data_reader import list_columns, open_run, read_metadata, summary_stats
from i2as.core.orchestrator import Orchestrator
from i2as.core.plan import ProbeSpec
from i2as.core.procedure import BaseProcedure
from i2as.core.run_builder import build_procedure
from i2as.core.station import build_station
from i2as.procedures.field_sweep import FieldSweep

CONFIG_PATH = "i2as/configs/sim_cryostat"

SAMPLE_INFO = {"sample_name": "S", "sample_id": "S-1", "comments": ""}

FULL_PARAMS = {
    "measurement_vi": "dc_measurement",
    "field_start": -1.0,
    "field_end": 1.0,
    "field_steps": 21,
    "temperature": 300.0,
    "current_A": 1e-6,
    "readings_per_point": 50,
    "init_wait": 300.0,
    "step_wait": 30.0,
}


@pytest.fixture
def station():
    return build_station(CONFIG_PATH)


def _build(station, tmp_path, probe=None, **overrides):
    """Build a FieldSweep, optionally reduced to a probe."""
    params = {**FULL_PARAMS, **overrides}
    return build_procedure(
        FieldSweep,
        station=station,
        params=params,
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
        file_prefix="probe",
        probe=probe,
    )


# ── The ProbeSpec contract ───────────────────────────────────────────────────

def test_probe_spec_is_frozen_and_json_safe():
    """A spec is immutable and survives a full JSON round trip."""
    spec = ProbeSpec(n_points=5, averaging=2, max_wait_s=1.5)

    with pytest.raises(Exception):
        spec.n_points = 9  # type: ignore[misc]

    assert ProbeSpec.from_json(json.loads(json.dumps(spec.to_json()))) == spec


def test_probe_spec_defaults_to_first_middle_last():
    """The default reduction keeps three points and one reading."""
    assert ProbeSpec() == ProbeSpec(n_points=3, averaging=1, max_wait_s=5.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_points": 0},
        {"averaging": 0},
        {"max_wait_s": -1.0},
        {"n_points": True},
        {"max_wait_s": float("inf")},
    ],
)
def test_probe_spec_refuses_impossible_reductions(kwargs):
    """A spec that could not describe a real run is refused at construction."""
    with pytest.raises((TypeError, ValueError)):
        ProbeSpec(**kwargs)


def test_probe_spec_from_json_ignores_unknown_keys():
    """A newer producer's extra key never breaks an older consumer."""
    assert ProbeSpec.from_json({"n_points": 2, "future_key": 7}) == ProbeSpec(
        n_points=2
    )


# ── The reduction rules ──────────────────────────────────────────────────────

def test_probe_keeps_first_middle_and_last_sweep_points(station, tmp_path):
    """Rule 1: the sweep is subsampled, extremes kept — that is where limits bite."""
    full = _build(station, tmp_path).get_sweep_array()
    probe = _build(station, tmp_path, probe=ProbeSpec()).get_sweep_array()

    assert len(full) == 21
    assert probe == [full[0], full[10], full[-1]]


def test_probe_point_cap_is_honoured_and_never_extends(station, tmp_path):
    """n_points is a cap: a short sweep is left exactly as it was."""
    probe = _build(station, tmp_path, probe=ProbeSpec(n_points=5)).get_sweep_array()
    assert len(probe) == 5

    short = _build(
        station, tmp_path, probe=ProbeSpec(n_points=9), field_steps=3
    ).get_sweep_array()
    assert len(short) == 3


def test_probe_caps_declared_waits_but_never_raises_them(station, tmp_path):
    """Rule 2: every seconds-valued declared parameter is capped, never raised."""
    params = _build(
        station, tmp_path, probe=ProbeSpec(max_wait_s=2.0), step_wait=0.5
    ).get_params()

    assert params["init_wait"] == 2.0  # 300 s -> the cap
    assert params["step_wait"] == 0.5  # already cheaper, left alone


def test_probe_caps_the_measurement_averaging(station, tmp_path):
    """Rule 3: a repeat count is found by declaration and cut to `averaging`."""
    params = _build(station, tmp_path, probe=ProbeSpec(averaging=2)).get_params()

    assert params["readings_per_point"] == 2
    assert params["current_A"] == 1e-6  # not a repeat count: untouched


def test_probe_declares_itself_and_records_its_spec(station, tmp_path):
    """Rule 4: the run says it is a probe, and says which reduction made it."""
    run = _build(station, tmp_path, probe=ProbeSpec(n_points=2))

    assert run.run_kind == "probe"
    assert run.get_params()["probe_spec"] == ProbeSpec(n_points=2).to_json()
    assert FieldSweep.run_kind == "run"  # the class is untouched


def test_a_run_built_without_a_probe_spec_is_unchanged(station, tmp_path):
    """The default path builds the run exactly as requested."""
    run = _build(station, tmp_path)

    assert run.run_kind == "run"
    assert "probe_spec" not in run.get_params()
    assert run.get_params()["readings_per_point"] == 50


def test_probe_targets_still_reach_the_extremes(station, tmp_path):
    """The reduced run still declares the setpoints the full run would command."""
    probe = _build(station, tmp_path, probe=ProbeSpec())

    fields = probe.planned_targets()["magnet_z"]
    assert min(fields) == pytest.approx(-1.0)
    assert max(fields) == pytest.approx(1.0)


def test_apply_probe_refuses_anything_but_a_probe_spec(station, tmp_path):
    """The hook is typed: a dict is not a spec."""
    run = _build(station, tmp_path)

    with pytest.raises(TypeError):
        run.apply_probe({"n_points": 2})


def test_build_procedure_refuses_to_probe_a_class_that_cannot(station, tmp_path):
    """A run class with no apply_probe() is refused, not silently un-probed."""

    class NotAProcedure:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    with pytest.raises(ValueError, match="cannot be probed"):
        build_procedure(
            NotAProcedure,
            station=station,
            params={},
            sample_info=SAMPLE_INFO,
            data_directory=str(tmp_path),
            probe=ProbeSpec(),
        )


def test_base_procedure_subsampling_keeps_the_ends(station, tmp_path):
    """The subsample helper is the rule, independent of any procedure."""
    values = list(range(10))

    assert BaseProcedure._probe_subsample(values, 3) == [0, 4, 9]
    assert BaseProcedure._probe_subsample(values, 1) == [0]
    assert BaseProcedure._probe_subsample([], 3) == []


# ── The probe reaches the data file ──────────────────────────────────────────

def test_probe_data_file_is_tagged_and_read_back_as_a_probe(station, tmp_path):
    """A probe writes a real HDF5 file that identifies itself as a probe."""
    run = _build(station, tmp_path, probe=ProbeSpec(n_points=2))
    station.send_measurement_commands(run.initiate().commands)
    run.measure()
    path = run.data_filepath
    run.standby()

    with open_run(path) as handle:
        metadata = read_metadata(handle)

    assert metadata["run_kind"] == "probe"
    assert metadata["params"]["probe_spec"]["n_points"] == 2


def test_a_science_run_file_is_tagged_run(station, tmp_path):
    """The same read on an unreduced run says "run", never an empty kind."""
    run = _build(station, tmp_path, field_steps=2)
    station.send_measurement_commands(run.initiate().commands)
    run.measure()
    path = run.data_filepath
    run.standby()

    with open_run(path) as handle:
        assert read_metadata(handle)["run_kind"] == "run"


# ── The engine port: a probe submitted as JSON, start to file ────────────────

def _tick_until(orchestrator, predicate, max_ticks: int = 4000):
    """Tick the engine until *predicate* holds; assert that it eventually does."""
    for _ in range(max_ticks):
        orchestrator._tick()
        if predicate():
            return
    raise AssertionError("the run never reached the expected state")


def test_a_probe_submitted_as_a_command_runs_and_writes_a_probe_file(
    station, qtbot, tmp_path
):
    """The whole path: submit a probe, tick it out, read the file back.

    The one test that exercises every layer the probe standard touches — the
    control contract's ``probe_spec`` argument, the headless build, the
    engine's run manifest, the HDF5 file's ``run_kind``, and the reader an
    analyst would use.
    """
    # The sim magnet ramps at a realistic rate; a tick-by-tick test cannot
    # wait for it, so speed it up (the same treatment the engine suite gives).
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []

    orchestrator = Orchestrator(
        station, tick_interval_ms=10, run_catalog={"FieldSweep": FieldSweep}
    )
    orchestrator.start_monitoring()
    verdicts: list[ev.Verdict] = []
    events: list[object] = []
    orchestrator.verdict_emitted.connect(verdicts.append)
    orchestrator.event_emitted.connect(events.append)

    command = ev.Command(
        name=ev.CommandName.RUN_PROCEDURE,
        actor=ev.Actor(kind=ev.ActorKind.AGENT, id="drift-watch", role="operator"),
        args={
            "procedure": "FieldSweep",
            "params": {**FULL_PARAMS, "field_steps": 51},
            "sample_info": SAMPLE_INFO,
            "data_directory": str(tmp_path),
            "file_prefix": "probe",
            "probe_spec": {"n_points": 3, "averaging": 2, "max_wait_s": 0.0},
        },
    )
    try:
        request_id = orchestrator.submit(command)

        assert request_id == command.request_id
        assert [v.request_id for v in verdicts] == [request_id]
        assert verdicts[0].code is ev.VerdictCode.OK

        started = [e for e in events if isinstance(e, ev.RunStarted)]
        assert len(started) == 1
        assert started[0].manifest["kind"] == "probe"
        assert started[0].request_id == request_id

        _tick_until(
            orchestrator, lambda: any(isinstance(e, ev.RunFinished) for e in events)
        )
        finished = [e for e in events if isinstance(e, ev.RunFinished)][0]
        assert finished.status == "done"
        data_file = finished.manifest["data_file"]
    finally:
        orchestrator.shutdown()

    with open_run(data_file) as handle:
        metadata = read_metadata(handle)
        columns = {info.name for info in list_columns(handle)}
        stats = summary_stats(handle, "field_T")

    assert metadata["run_kind"] == "probe"
    assert metadata["params"]["probe_spec"] == {
        "n_points": 3,
        "averaging": 2,
        "max_wait_s": 0.0,
    }
    assert metadata["params"]["readings_per_point"] == 2
    assert {"field_T", "voltage_V"} <= columns
    # The reduced point count, end to end: 51 requested, 3 measured, and the
    # extremes of the requested sweep still reached.
    assert stats.count == 3
    assert (stats.min, stats.max) == pytest.approx((-1.0, 1.0))
