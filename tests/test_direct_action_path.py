# ---
# description: |
#   The direct action path's fence: the five refusals that stand between a
#   manual (GUI or agent) call and an instrument, and the unconditional
#   emergency-standby route that is permitted in every state.
# last_updated: 2026-09-02
# ---

"""Direct-action-path tests: what a manual action may command, and what it may not.

The direct action path is ``Station.execute_vi_action()`` and the
``Orchestrator`` manual-action queue that feeds it (see GLOSSARY.md's
**Direct action path**). Five independent checks refuse a call before any
hardware command is sent, and each gives its own reason:

1. a private (underscore-prefixed) name,
2. a public method that is not a declared capability,
3. a capability outside the caller's capability scope,
4. a value outside the setup's ``control_limits``,
5. a setpoint outside the active experiment's session envelope.

The first four are the Station's; the fifth is the Orchestrator's, because the
envelope lives there. This module asserts all five, that their reasons are
pairwise distinct, and that each reaches the GUI as a verdict rather than
silence.
"""

from __future__ import annotations

import logging

import pytest

from i2as.core.exceptions import (
    I2ASActionScopeError,
    I2ASPrivateActionError,
    I2ASSafetyError,
    I2ASUndeclaredActionError,
)
from i2as.core.orchestrator import (
    MANUAL_ACTION_SCOPE,
    Orchestrator,
    OrchestratorState,
)
from i2as.core.decorators import control
from i2as.core.plan import EnvelopeBound, ExperimentEnvelope
from i2as.core.station import build_station
from i2as.virtual_instruments.base import BaseVirtualInstrument


class _HousekeepingVI(BaseVirtualInstrument):
    """Test-local double declaring one operation-scope capability.

    No shipped VI declares ``scope="operation"``, so the capability-scope
    fence (refusal 3 below) gets its subject here, exactly as a setup with
    an instrument-housekeeping action would declare it.
    """

    vi_type = "system"

    def __init__(self, drivers: dict, **init_params: object) -> None:
        super().__init__(drivers, **init_params)
        self.mode = 0

    @control(scope="operation", action_class="recovery")
    def set_housekeeping_mode(self, mode: int) -> None:
        """Select an instrument-housekeeping mode.

        Args:
            mode: The mode index to select.
        """
        self.mode = int(mode)


@pytest.fixture
def station():
    """A real simulated station (sim_cryostat), plus one operation-scope VI."""
    built = build_station("i2as/configs/sim_cryostat")
    built.register_vi("housekeeping", _HousekeepingVI({}), "system")
    return built


@pytest.fixture
def orchestrator(station, qtbot):
    """An Orchestrator over the sim station, ticked by hand in these tests."""
    orch = Orchestrator(station, tick_interval_ms=10)
    yield orch
    orch.shutdown()


# ── The five refusals (the audit's Phase 0 exit test) ─────────────────────────


def test_refuses_private_name(station):
    """An underscore-prefixed name is never a capability."""
    with pytest.raises(I2ASPrivateActionError) as exc:
        station.execute_vi_action("magnet_z", "_ramp_generator")
    assert "private name" in str(exc.value)


def test_refuses_non_control_method(station):
    """A public method without @control is not dispatchable either.

    ``start_ramp`` is the case that motivates the check: it is public, it
    moves the magnet, and it is the primitive ``set_field`` calls — so
    dispatching it directly would have walked straight past the field limit
    ``set_field``'s ``control_limits`` wrapper enforces.
    """
    with pytest.raises(I2ASUndeclaredActionError) as exc:
        station.execute_vi_action("magnet_z", "start_ramp", target=99.0)
    assert "not a declared capability" in str(exc.value)
    assert station.magnet_z.ramp_status() == "IDLE"


def test_refuses_out_of_scope_capability(station):
    """An operation-scope capability is refused for a measurement-scope caller."""
    with pytest.raises(I2ASActionScopeError) as exc:
        station.execute_vi_action("housekeeping", "set_housekeeping_mode", mode=1)
    assert "requires operation-scope access" in str(exc.value)


def test_refuses_value_outside_control_limits(station):
    """A value beyond the setup's configured limit is refused before dispatch."""
    with pytest.raises(I2ASSafetyError) as exc:
        station.execute_vi_action("magnet_z", "set_field", target_T=99.0)
    assert "outside the allowed range" in str(exc.value)


def test_refuses_setpoint_outside_session_envelope(orchestrator):
    """A manual setpoint outside the active envelope is refused, naming the bound."""
    orchestrator.set_experiment_envelope(
        ExperimentEnvelope(bounds={"magnet_z": EnvelopeBound(min_value=-0.5, max_value=0.5)})
    )
    blocked: list[str] = []
    orchestrator.action_blocked.connect(blocked.append)

    orchestrator.submit_vi_action("magnet_z", "set_field", target_T=2.0)

    assert blocked, "an out-of-envelope manual action must be refused"
    assert "session envelope" in blocked[0]
    assert "above the session maximum 0.5" in blocked[0]
    assert not orchestrator._gui_action_queue, "the action must not be queued"


def test_the_five_refusal_reasons_are_pairwise_distinct(station, orchestrator):
    """Each of the five refusals says something different.

    A fence whose refusals all read alike is a fence nobody can debug: the
    operator (or an agent choosing what to do next) must be able to tell "that
    is not a capability" from "that value is too large" without guessing.
    """
    reasons: list[str] = []

    for call in (
        lambda: station.execute_vi_action("magnet_z", "_ramp_generator"),
        lambda: station.execute_vi_action("magnet_z", "start_ramp", target=99.0),
        lambda: station.execute_vi_action("housekeeping", "set_housekeeping_mode", mode=1),
        lambda: station.execute_vi_action("magnet_z", "set_field", target_T=99.0),
    ):
        with pytest.raises(I2ASSafetyError) as exc:
            call()
        reasons.append(str(exc.value))

    orchestrator.set_experiment_envelope(
        ExperimentEnvelope(bounds={"magnet_z": EnvelopeBound(min_value=-0.5, max_value=0.5)})
    )
    blocked: list[str] = []
    orchestrator.action_blocked.connect(blocked.append)
    orchestrator.submit_vi_action("magnet_z", "set_field", target_T=2.0)
    reasons.append(blocked[0])

    assert len(reasons) == 5
    assert len(set(reasons)) == 5, f"refusal reasons repeat: {reasons}"


# ── The refusals reach the operator ───────────────────────────────────────────


def test_station_refusal_surfaces_as_action_failed(orchestrator, station, qtbot):
    """A Station-level refusal on the drained action becomes action_failed(reason)."""
    failures: list[tuple[str, str, str]] = []
    orchestrator.action_failed.connect(
        lambda vi, method, reason: failures.append((vi, method, reason))
    )

    orchestrator.submit_vi_action("magnet_z", "start_ramp", target=99.0)
    orchestrator._tick()

    assert failures, "a refused action must get an explicit verdict"
    vi_name, method_name, reason = failures[0]
    assert (vi_name, method_name) == ("magnet_z", "start_ramp")
    assert "not a declared capability" in reason


def test_envelope_refusal_is_re_checked_at_drain(orchestrator, station):
    """An envelope installed AFTER the click still refuses the queued action.

    The re-validation-at-drain discipline: an experiment can open between the
    click and the tick, and the envelope must bind the action that is about to
    execute, not the one that was submitted.
    """
    orchestrator.submit_vi_action("magnet_z", "set_field", target_T=2.0)
    assert orchestrator._gui_action_queue

    orchestrator.set_experiment_envelope(
        ExperimentEnvelope(bounds={"magnet_z": EnvelopeBound(min_value=-0.5, max_value=0.5)})
    )
    blocked: list[str] = []
    orchestrator.action_blocked.connect(blocked.append)

    orchestrator._tick()

    assert blocked and "session envelope" in blocked[0]
    assert station.magnet_z.ramp_status() == "IDLE", "no ramp may have started"


def test_manual_path_grants_operation_scope(orchestrator, station, qtbot):
    """A human at the front panel may still use a wider-scope capability.

    ``execute_vi_action()``'s default scope is the restrictive
    ``"measurement"``; the Orchestrator opts into ``MANUAL_ACTION_SCOPE`` for
    manual actions, which is what keeps today's front-panel behaviour intact.
    """
    assert MANUAL_ACTION_SCOPE == "operation"
    succeeded: list[tuple[str, str]] = []
    orchestrator.action_succeeded.connect(lambda vi, m: succeeded.append((vi, m)))

    orchestrator.submit_vi_action("housekeeping", "set_housekeeping_mode", mode=1)
    orchestrator._tick()

    assert succeeded == [("housekeeping", "set_housekeeping_mode")]


def test_lifecycle_actions_still_dispatch(station):
    """initiate/standby carry no @control but remain capabilities of this path."""
    station.execute_vi_action("magnet_z", "initiate")
    station.execute_vi_action("magnet_z", "standby")


# ── The read side: what an envelope may narrow ───────────────────────────────


def test_envelope_variables_name_each_setpoint_and_its_setup_bounds(
    orchestrator, station
):
    """Each enveloped quantity reports its capability and the config's bounds.

    The source the Start Experiment dialog's editor pre-fills from, exposed to
    the GUI through the Orchestrator (which is all the GUI may talk to) and
    identical to the Station's own answer.
    """
    variables = orchestrator.envelope_variables()

    assert variables == station.envelope_variables()
    magnet = variables["magnet_z"]
    assert (magnet.method_name, magnet.param_name) == ("set_field", "target_T")
    assert (magnet.config_min, magnet.config_max) == station.magnet_z.limit_bounds(
        "field_T"
    )
    assert magnet.unit_suffix == "T"
    assert "keithley_dc_mode" not in variables, (
        "a measurement VI commands no setpoint an envelope bounds"
    )


def test_setpoint_parameters_are_reported_only_for_capabilities(station):
    """``setpoint_parameters()`` answers for a @control, and only for one."""
    assert station.setpoint_parameters("magnet_z", "set_field") == ("target_T",)
    assert station.setpoint_parameters("magnet_z", "start_ramp") == ()
    assert station.setpoint_parameters("magnet_z", "no_such_method") == ()
    assert station.setpoint_parameters("no_such_vi", "set_field") == ()


def test_envelope_ignores_non_setpoint_parameters(orchestrator, station):
    """A bounded VI's NON-setpoint controls are untouched by the envelope.

    The envelope bounds a quantity, not a VI: with a temperature envelope
    active, the controller's rate control still passes.
    """
    orchestrator.set_experiment_envelope(
        ExperimentEnvelope(
            bounds={"temperature": EnvelopeBound(min_value=-1.0, max_value=1.0)}
        )
    )
    succeeded: list[tuple[str, str]] = []
    orchestrator.action_succeeded.connect(lambda vi, m: succeeded.append((vi, m)))

    orchestrator.submit_vi_action(
        "temperature", "set_ramp_rate", rate_K_per_min=2.0
    )
    orchestrator._tick()

    assert succeeded == [("temperature", "set_ramp_rate")]


# ── The excitation-current fence ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "vi_name, method_name, param_name",
    [
        ("dc_measurement", "initiate_measurement", "current_A"),
        ("dc_measurement", "set_source_current", "current_A"),
    ],
)
def test_excitation_current_is_bounded_both_ways(
    station, vi_name: str, method_name: str, param_name: str
):
    """Every way of commanding excitation current is bounded, in both signs.

    Arming a measurement and reprogramming the current between readings are
    two different entry points to the same hazard, so both carry the limit —
    and the bound is symmetric, because current reversal is routine.
    """
    ceiling = station.get_vi(vi_name)._limits["source_current_A"][1]
    assert ceiling == pytest.approx(0.105)
    for value in (ceiling * 2, -ceiling * 2):
        with pytest.raises(I2ASSafetyError) as exc:
            station.execute_vi_action(vi_name, method_name, **{param_name: value})
        assert "outside the allowed range" in str(exc.value)


# ── emergency_standby(): the unconditional S0 path ───────────────────────────


def _drive_to_measuring(orchestrator, station):
    """Run a trivial sweep far enough to reach MEASURING."""
    from i2as.core.plan import PhasePlan, StepPlan, Target

    class _Sweep:
        name = "Emergency standby probe"

        def __init__(self):
            self.measure_called = 0

        def initiate(self):
            return PhasePlan(targets={"magnet_z": Target(0.05)}, commands=(), wait_s=0.0)

        def change_sweep_step(self):
            return StepPlan(targets={"magnet_z": Target(0.05)}, wait_s=0.0)

        def measure(self):
            self.measure_called += 1

        def standby(self):
            return PhasePlan(targets={"magnet_z": Target(0.0)}, commands=(), wait_s=0.0)

    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []
    orchestrator.run_procedure(_Sweep())
    for _ in range(200):
        orchestrator._tick()
        if orchestrator._state == OrchestratorState.MEASURING:
            return
    raise AssertionError("never reached MEASURING")


@pytest.mark.parametrize("start_state", ["IDLE", "MEASURING", "PAUSED", "EMERGENCY"])
def test_emergency_standby_is_permitted_in_every_state(
    orchestrator, station, qtbot, start_state: str
):
    """The stop button always works — including from PAUSED and EMERGENCY.

    Those last two are precisely the states where every other manual route is
    refused, and where making the machine safe matters most.
    """
    orchestrator.start_monitoring()
    if start_state == "MEASURING":
        _drive_to_measuring(orchestrator, station)
    elif start_state == "PAUSED":
        _drive_to_measuring(orchestrator, station)
        orchestrator.pause_procedure()
        for _ in range(200):
            orchestrator._tick()
            if orchestrator._state == OrchestratorState.PAUSED:
                break
        assert orchestrator._state == OrchestratorState.PAUSED
    elif start_state == "EMERGENCY":
        orchestrator.emergency_standby("first trip")
        assert orchestrator._state == OrchestratorState.EMERGENCY

    orchestrator.emergency_standby("operator pressed the panic button")

    assert orchestrator._state == OrchestratorState.EMERGENCY
    assert orchestrator._procedure is None


def test_emergency_standby_logs_critical_with_the_reason(orchestrator, caplog):
    """The reason lands in a CRITICAL log record, per the logging levels."""
    with caplog.at_level(logging.CRITICAL, logger="i2as.core.orchestrator"):
        orchestrator.emergency_standby("coolant level below the safe line")

    critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert critical, "an emergency standby must be logged at CRITICAL"
    assert any("coolant level below the safe line" in r.getMessage() for r in critical)


def test_emergency_standby_reason_reaches_the_error_signal(orchestrator, qtbot):
    """The reason is carried into the emergency's own error message too."""
    errors: list[str] = []
    orchestrator.error_occurred.connect(errors.append)

    orchestrator.emergency_standby("sample stage jammed")

    assert any("sample stage jammed" in message for message in errors)


def test_emergency_standby_stands_every_vi_down(orchestrator, station, qtbot):
    """It routes into the same standby_all() a tripped critical flag takes."""
    stood_down: list[str] = []
    for vi_name in station.get_vi_names():
        vi = station.get_vi(vi_name)
        original = vi.standby

        def _spy(*args, _name=vi_name, _original=original, **kwargs):
            stood_down.append(_name)
            return _original(*args, **kwargs)

        vi.standby = _spy

    orchestrator.emergency_standby("cryostat vacuum lost")

    assert set(stood_down) == set(station.get_vi_names())
