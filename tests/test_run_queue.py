# ---
# description: |
#   Tests for i2as.session.run_queue — the run queue as data. Covers the
#   RunSpec contract (frozen, JSON round trip, actor), RunQueue ordering,
#   build_run's construction shape, and validate_run's three checks (declared ParamSpec
#   bounds, the headless build, control_limits + the session envelope).
# last_updated: 2026-09-03
# ---

import json

import pytest

from i2as.core.events import OPERATOR, Actor, ActorKind
from i2as.core.plan import EnvelopeBound, ExperimentEnvelope
from i2as.core.station import build_station
from i2as.procedures.field_sweep import FieldSweep
from i2as.session.run_queue import (
    FINDING_BUILD_REFUSED,
    FINDING_CONTROL_LIMIT,
    FINDING_ENVELOPE,
    FINDING_PARAM_BOUNDS,
    FINDING_UNKNOWN_PARAM,
    KIND_PROCEDURE,
    RunFinding,
    RunQueue,
    RunQueueHost,
    RunSpec,
    RunValidation,
    build_run,
    validate_run,
)

CONFIG_PATH = "i2as/configs/sim_cryostat"

AGENT = Actor(kind=ActorKind.AGENT, id="drift-watch", role="operator")

SAMPLE_INFO = {"sample_name": "S", "sample_id": "S-1", "comments": ""}

FAST_PARAMS = {
    "measurement_vi": "dc_measurement",
    "field_start": -0.1,
    "field_end": 0.1,
    "field_steps": 3,
    "temperature": 300.0,
    "current_A": 1e-6,
    "readings_per_point": 5,
    "init_wait": 0.0,
    "step_wait": 0.0,
}


@pytest.fixture
def station():
    return build_station(CONFIG_PATH)


def _spec(kind=KIND_PROCEDURE, run_class="FieldSweep", **overrides):
    """Build a RunSpec with sensible defaults for queue-ordering tests."""
    return RunSpec(kind=kind, run_class=run_class, **overrides)


# ── RunSpec: the contract ────────────────────────────────────────────────────

def test_run_spec_is_frozen_and_json_safe():
    """A spec is immutable and survives a full JSON round trip."""
    spec = RunSpec(
        kind=KIND_PROCEDURE,
        run_class="FieldSweep",
        params={"field_start": -1.0, "segments": [{"end": 1.0}]},
        sample_info=SAMPLE_INFO,
        data_directory="/data",
        file_prefix="run1",
        actor=AGENT,
    )

    with pytest.raises(Exception):
        spec.run_class = "Other"  # type: ignore[misc]

    restored = RunSpec.from_json(json.loads(json.dumps(spec.to_json())))
    assert restored == spec
    assert restored.actor == AGENT


def test_run_spec_defaults_to_the_operator_and_a_fresh_id():
    """Nobody has to name the operator, and every spec is identifiable."""
    first, second = _spec(), _spec()

    assert first.actor == OPERATOR
    assert first.spec_id != second.spec_id
    assert first.queued_at > 0


def test_run_spec_copies_its_params_defensively():
    """A caller mutating the dict it passed cannot reach into the queue."""
    params = {"field_start": -1.0}
    spec = RunSpec(kind=KIND_PROCEDURE, run_class="FieldSweep", params=params)
    params["field_start"] = 99.0

    assert spec.params == {"field_start": -1.0}


def test_run_spec_refuses_an_unknown_kind():
    """Only 'procedure' exists — a third kind fails loudly."""
    with pytest.raises(ValueError, match="kind"):
        RunSpec(kind="calibration", run_class="FieldSweep")


def test_run_spec_refuses_a_non_json_parameter():
    """A spec is a contract type: it will not carry what a client cannot parse."""
    with pytest.raises(TypeError, match="JSON-safe"):
        RunSpec(kind=KIND_PROCEDURE, run_class="FieldSweep", params={"vi": object()})


# ── RunQueue: ordering ───────────────────────────────



def test_queue_preserves_add_order_within_a_kind():
    """Two procedures run in the order they were added."""
    queue = RunQueue()
    first = queue.add(_spec())
    second = queue.add(_spec())

    assert [s.spec_id for s in queue.snapshot()] == [first.spec_id, second.spec_id]


def test_remove_drops_one_entry_and_reports_whether_it_did():
    """Removing is idempotent-safe: an unknown id is False, not an error."""
    queue = RunQueue()
    spec = queue.add(_spec())

    assert queue.remove(spec.spec_id) is True
    assert queue.remove(spec.spec_id) is False
    assert len(queue) == 0


def test_move_reorders_within_the_bucket_and_clamps_at_the_ends():
    """A move never interleaves the two kinds and never falls off an end."""
    queue = RunQueue()
    first, second = queue.add(_spec()), queue.add(_spec())

    assert queue.move(second.spec_id, -1) is True
    assert [s.spec_id for s in queue.snapshot()] == [second.spec_id, first.spec_id]
    assert queue.move(second.spec_id, -1) is False
    assert queue.move("nope", -1) is False




def test_clear_empties_the_queue_and_reports_whether_it_did():
    """Clearing an empty queue changes nothing, so it broadcasts nothing."""
    queue = RunQueue()
    queue.add(_spec())

    assert queue.clear() is True
    assert queue.clear() is False
    assert queue.snapshot() == ()


def test_a_duplicate_spec_id_is_refused():
    """Ids are how remove/move address an entry, so they must stay unique."""
    queue = RunQueue()
    spec = queue.add(_spec())

    with pytest.raises(ValueError, match="already queued"):
        queue.add(spec)


def test_entries_are_json_safe_dicts_in_run_order():
    """The snapshot a QueueChanged event carries is plain JSON."""
    queue = RunQueue()
    queue.add(_spec(run_class="FieldSweep"))
    queue.add(_spec(run_class="TemperatureSweep"))

    entries = queue.entries()

    assert json.loads(json.dumps(entries))
    assert [entry["run_class"] for entry in entries] == [
        "FieldSweep",
        "TemperatureSweep",
    ]


# ── build_run: the pull seam's other half ────────────────────────────────────

def test_build_run_constructs_the_procedure_the_spec_names(station, tmp_path):
    """A spec plus a catalog yields exactly one live procedure."""
    spec = RunSpec(
        kind=KIND_PROCEDURE,
        run_class="FieldSweep",
        params=FAST_PARAMS,
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
        file_prefix="queued",
    )

    run = build_run(spec, station=station, run_catalog={"FieldSweep": FieldSweep})

    assert isinstance(run, FieldSweep)
    assert run._file_prefix == "queued"


def test_build_run_refuses_a_class_the_catalog_does_not_hold(station, tmp_path):
    """Discovery is injected, so an unknown name is a caller-facing KeyError."""
    spec = RunSpec(kind=KIND_PROCEDURE, run_class="NoSuchProcedure")

    with pytest.raises(KeyError, match="NoSuchProcedure"):
        build_run(spec, station=station, run_catalog={"FieldSweep": FieldSweep})


def test_build_run_stamps_the_experiment_open_at_build_time(station, tmp_path):
    """A queued run belongs to the experiment open when it actually starts."""
    spec = RunSpec(
        kind=KIND_PROCEDURE,
        run_class="FieldSweep",
        params=FAST_PARAMS,
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
    )

    run = build_run(
        spec,
        station=station,
        run_catalog={"FieldSweep": FieldSweep},
        experiment_info={"experiment": {"experiment_id": "exp-1"}},
    )

    assert run._experiment_info["experiment"]["experiment_id"] == "exp-1"


# ── validate_run: refused at add time, not at start time ─────────────────────

def test_a_valid_run_produces_no_findings(station, tmp_path):
    """The happy path: nothing found, nothing dispatched, no file written."""
    result = validate_run(
        FieldSweep, FAST_PARAMS, station=station, data_directory=str(tmp_path)
    )

    assert result.ok
    assert result.findings == ()
    assert list(tmp_path.iterdir()) == []


def test_a_setpoint_outside_the_setup_limit_is_refused(station, tmp_path):
    """control_limits: a field beyond the magnet's configured range is caught."""
    params = dict(FAST_PARAMS, field_end=50.0)

    result = validate_run(
        FieldSweep, params, station=station, data_directory=str(tmp_path)
    )

    assert not result.ok
    codes = {finding.code for finding in result.findings}
    assert FINDING_CONTROL_LIMIT in codes
    assert any("magnet_z" in message for message in result.messages())


def test_a_setpoint_outside_the_session_envelope_is_refused(station, tmp_path):
    """The envelope narrows the setup limit, and narrows it at queue time."""
    envelope = ExperimentEnvelope(
        bounds={"magnet_z": EnvelopeBound(min_value=-0.05, max_value=0.05)}
    )

    result = validate_run(
        FieldSweep,
        FAST_PARAMS,
        station=station,
        data_directory=str(tmp_path),
        envelope=envelope,
    )

    assert not result.ok
    assert {f.code for f in result.findings} == {FINDING_ENVELOPE}


def test_a_declared_bound_is_checked_before_anything_is_built(station, tmp_path):
    """ParamSpec bounds are the run's own declaration, checked first."""
    spec = FieldSweep.parameters["field_steps"]
    assert spec.min is not None, "field_steps declares a minimum to violate"
    params = dict(FAST_PARAMS, field_steps=spec.min - 1)

    result = validate_run(
        FieldSweep, params, station=station, data_directory=str(tmp_path)
    )

    assert not result.ok
    assert any(
        finding.code == FINDING_PARAM_BOUNDS and finding.param == "field_steps"
        for finding in result.findings
    )


def test_an_undeclared_parameter_is_reported(station, tmp_path):
    """A stale saved queue entry naming a renamed parameter is caught here."""

    class _Strict:
        name = "Strict"
        parameters = {"good": object()}

        def __init__(self, station, sample_info, data_directory, file_prefix="",
                     experiment_info=None, good=1):
            self.good = good

    result = validate_run(
        _Strict, {"typo": 1}, station=station, data_directory=str(tmp_path)
    )

    assert any(finding.code == FINDING_UNKNOWN_PARAM for finding in result.findings)


def test_a_procedure_that_absorbs_extra_parameters_is_not_told_they_are_unknown(
    station, tmp_path
):
    """A generic sweep takes its measurement VI's parameters alongside its own."""
    result = validate_run(
        FieldSweep, FAST_PARAMS, station=station, data_directory=str(tmp_path)
    )

    assert not [f for f in result.findings if f.code == FINDING_UNKNOWN_PARAM]


def test_a_procedure_that_refuses_the_run_becomes_a_finding(station, tmp_path):
    """A refusing __init__ is a finding, never an exception at the caller."""
    params = dict(FAST_PARAMS, measurement_vi="no_such_vi")

    result = validate_run(
        FieldSweep, params, station=station, data_directory=str(tmp_path)
    )

    assert not result.ok
    assert any(f.code == FINDING_BUILD_REFUSED for f in result.findings)


def test_validation_renders_as_json(station, tmp_path):
    """Findings and the duration estimate are both JSON-safe."""
    result = validate_run(
        FieldSweep,
        dict(FAST_PARAMS, field_end=50.0),
        station=station,
        data_directory=str(tmp_path),
    )

    payload = json.loads(json.dumps(result.to_json()))
    assert payload["ok"] is False
    assert payload["findings"][0]["code"] == FINDING_CONTROL_LIMIT
    # The run still built, so it still has an estimate — and the headline
    # number is the breakdown's own total, never a second stored copy.
    assert payload["duration_estimate_s"] == payload["estimate"]["total_s"]
    assert payload["estimate"]["assumptions"]


def test_a_refused_build_has_no_estimate(station, tmp_path):
    """Nothing was built, so there is nothing to estimate — and it says so."""
    result = validate_run(
        FieldSweep,
        dict(FAST_PARAMS, measurement_vi="no_such_vi"),
        station=station,
        data_directory=str(tmp_path),
    )

    assert [f.code for f in result.findings] == [FINDING_BUILD_REFUSED]
    assert result.estimate is None
    assert result.duration_estimate_s is None


def test_run_validation_rejects_a_bare_string_finding():
    """Findings are structured so a client never has to parse prose."""
    with pytest.raises(TypeError, match="RunFinding"):
        RunValidation(findings=("something went wrong",))

    assert RunValidation(findings=(RunFinding("x", "y"),)).ok is False


# ── The pre-dispatch declaration ─────────────────────────────────────────────

def test_planned_targets_names_every_setpoint_the_sweep_would_command(
    station, tmp_path
):
    """planned_targets() is derived from the same hooks the plans dispatch."""
    run = build_run(
        RunSpec(
            kind=KIND_PROCEDURE,
            run_class="FieldSweep",
            params=FAST_PARAMS,
            sample_info=SAMPLE_INFO,
            data_directory=str(tmp_path),
        ),
        station=station,
        run_catalog={"FieldSweep": FieldSweep},
    )

    planned = run.planned_targets()

    assert "magnet_z" in planned
    assert min(planned["magnet_z"]) == pytest.approx(-0.1)
    assert max(planned["magnet_z"]) == pytest.approx(0.1)


def test_a_run_that_declares_no_targets_is_validated_on_its_parameters_alone(
    station, tmp_path
):
    """The default declaration is {} — honest for a run that only reads."""

    class _ReadOnlyRun:
        name = "Read Only"
        parameters: dict = {}

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    result = validate_run(
        _ReadOnlyRun, {}, station=station, data_directory=str(tmp_path)
    )

    assert result.ok


# ── Probe runs: a reduced variant, queued and built like any other ───────────

def test_a_spec_carries_its_probe_reduction_through_a_json_round_trip():
    """A probe entry stays JSON-safe end to end, like every other spec field."""
    spec = RunSpec(
        kind=KIND_PROCEDURE,
        run_class="FieldSweep",
        params=FAST_PARAMS,
        probe_spec={"n_points": 2, "averaging": 1, "max_wait_s": 0.0},
    )

    restored = RunSpec.from_json(json.loads(json.dumps(spec.to_json())))

    assert restored.probe_spec == spec.probe_spec


def test_a_spec_refuses_a_malformed_probe_reduction():
    """A reduction that could not describe a run is refused when it is queued."""
    with pytest.raises((TypeError, ValueError)):
        RunSpec(
            kind=KIND_PROCEDURE, run_class="FieldSweep", probe_spec={"n_points": 0}
        )




def test_build_run_reduces_a_probe_spec_to_the_cheap_variant(station, tmp_path):
    """The pull seam builds the probe the spec asked for, not the full run."""
    spec = RunSpec(
        kind=KIND_PROCEDURE,
        run_class="FieldSweep",
        params={**FAST_PARAMS, "field_steps": 21},
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
        probe_spec={"n_points": 3},
    )

    run = build_run(spec, station=station, run_catalog={"FieldSweep": FieldSweep})

    assert run.run_kind == "probe"
    assert len(run.get_sweep_array()) == 3


def test_validate_run_checks_the_probe_variant_when_one_is_asked_for(
    station, tmp_path
):
    """A probe is validated as what would actually run — the reduced run."""
    result = validate_run(
        FieldSweep,
        {**FAST_PARAMS, "field_steps": 21},
        station=station,
        data_directory=str(tmp_path),
        probe_spec={"n_points": 3, "max_wait_s": 0.0},
    )

    assert result.ok


def test_queueing_a_probe_stores_its_reduction_on_the_spec(station, tmp_path):
    """What waits in the queue is the probe itself, reduction and all."""
    host = RunQueueHost(
        station=station, run_catalog={"FieldSweep": FieldSweep}
    )

    spec, validation = host.add(
        FieldSweep,
        FAST_PARAMS,
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
        probe_spec={"n_points": 2},
    )

    assert validation.ok
    assert spec is not None and spec.probe_spec == {
        "n_points": 2,
        "averaging": 1,
        "max_wait_s": 5.0,
    }
    assert host.next_run().run_kind == "probe"






