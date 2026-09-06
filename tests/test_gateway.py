"""The agent gateway's permission model (L6).

Two halves, tested against a real simulated station's declaration snapshot
rather than a hand-built one, so a classification that names a capability the
station does not have fails here rather than in front of an agent:

* the classification (`action_classes.py`) — which class an action IS;
* the matrix (`roles.py`) — who may take an action of that class.
"""

from __future__ import annotations

import pytest

from i2as.core import events as ev
from i2as.core.station import build_station
from i2as.session.gateway import (
    ActionClass,
    Permission,
    Role,
    UnclassifiedActionError,
    authorize,
    classify_command,
)

OBSERVER = ev.Actor(kind=ev.ActorKind.AGENT, id="watcher", role=Role.OBSERVER.value)
DEBUG = ev.Actor(kind=ev.ActorKind.AGENT, id="fixer", role=Role.DEBUG.value)
SESSION = ev.Actor(kind=ev.ActorKind.AGENT, id="runner", role=Role.SESSION.value)


@pytest.fixture(scope="module")
def station_info():
    """The declaration snapshot of a real simulated station."""
    return build_station("i2as/configs/sim_cryostat").station_info()


def _command(name, actor=SESSION, **args):
    return ev.Command(name=name, actor=actor, args=args)


def _authorize(command, station_info, *, attended=True, gate=ev.AgentGate.ACTIVE):
    return authorize(command.actor, command, station_info, attended, gate)


# ── Classification ────────────────────────────────────────────────────────


def test_a_command_is_classified_from_the_table(station_info):
    """Every command but submit_vi_action reads its class straight off a row."""
    assert (
        classify_command(_command(ev.CommandName.RUN_PROCEDURE), station_info).action_class
        is ActionClass.RUN_CONTROL
    )
    assert (
        classify_command(_command(ev.CommandName.PAUSE_PROCEDURE), station_info).action_class
        is ActionClass.RECOVERY
    )
    assert (
        classify_command(
            _command(ev.CommandName.SET_EXPERIMENT_ENVELOPE), station_info
        ).action_class
        is ActionClass.ENVELOPE
    )


def test_a_vi_action_is_classified_from_its_target(station_info):
    """The one command whose class depends on what it points at.

    And the class comes from the target's own declaration (the action-class
    declaration on the ``@control``), not from a table in the gateway.
    """
    ramp = _command(
        ev.CommandName.SUBMIT_VI_ACTION,
        vi_name="magnet_z",
        method_name="set_field",
        target_T=0.1,
    )
    assert classify_command(ramp, station_info).action_class is ActionClass.RUN_CONTROL

    read = _command(
        ev.CommandName.SUBMIT_VI_ACTION,
        vi_name="dc_measurement",
        method_name="read_now",
    )
    assert classify_command(read, station_info).action_class is ActionClass.READ

    lifecycle = _command(
        ev.CommandName.SUBMIT_VI_ACTION, vi_name="magnet_z", method_name="standby"
    )
    assert classify_command(lifecycle, station_info).action_class is ActionClass.RECOVERY


def test_an_unclassified_capability_is_refused_by_name(station_info):
    """No silent default: the refusal names the capability that has no row."""
    command = _command(
        ev.CommandName.SUBMIT_VI_ACTION,
        vi_name="magnet_z",
        method_name="no_such_capability",
    )
    with pytest.raises(UnclassifiedActionError, match="no_such_capability"):
        classify_command(command, station_info)

    verdict = _authorize(command, station_info)
    assert verdict is not None
    assert verdict.code is ev.VerdictCode.BLOCKED_ROLE
    assert "no_such_capability" in verdict.reason
    assert verdict.detail["rule"] == "unclassified_action"


def test_an_unconfigured_instrument_is_refused_by_name(station_info):
    """An action aimed at an instrument this station does not have."""
    command = _command(
        ev.CommandName.SUBMIT_VI_ACTION, vi_name="ghost", method_name="set_field"
    )
    with pytest.raises(UnclassifiedActionError, match="ghost"):
        classify_command(command, station_info)


# ── The matrix ────────────────────────────────────────────────────────────


def test_observer_is_refused_run_control_with_a_structured_reason(station_info):
    """The lowest role reads and nothing else; the refusal is machine-readable."""
    command = _command(ev.CommandName.RUN_PROCEDURE, actor=OBSERVER)

    verdict = _authorize(command, station_info)

    assert verdict is not None
    assert verdict.code is ev.VerdictCode.BLOCKED_ROLE
    assert verdict.actor == OBSERVER
    assert verdict.request_id == command.request_id
    assert verdict.detail["rule"] == "role_matrix"
    assert verdict.detail["role"] == "observer"
    assert verdict.detail["action_class"] == "run_control"
    assert verdict.detail["rationale"]
    assert not verdict.ok


def test_observer_may_read(station_info):
    """A read-class capability is permitted to every role."""
    command = _command(
        ev.CommandName.SUBMIT_VI_ACTION,
        actor=OBSERVER,
        vi_name="dc_measurement",
        method_name="read_now",
    )

    assert _authorize(command, station_info) is None


def test_debug_takes_recovery_only_while_unattended(station_info):
    """The attendance rule: with a human present the agent reports instead."""
    command = _command(ev.CommandName.PAUSE_PROCEDURE, actor=DEBUG)

    refused = _authorize(command, station_info, attended=True)
    assert refused is not None
    assert refused.code is ev.VerdictCode.BLOCKED_ROLE
    assert refused.detail["rule"] == "attendance"
    assert refused.detail["attended"] is True
    assert "unattended" in refused.reason

    assert _authorize(command, station_info, attended=False) is None


def test_debug_is_refused_run_control_even_unattended(station_info):
    """Attendance widens recovery only — it never grants run control."""
    command = _command(ev.CommandName.ABORT_PROCEDURE, actor=DEBUG)

    verdict = _authorize(command, station_info, attended=False)

    assert verdict is not None
    assert verdict.detail["rule"] == "role_matrix"
    assert verdict.detail["action_class"] == "run_control"


def test_session_runs_the_experiment_but_never_changes_the_rules(station_info):
    """Run control yes; the envelope, attendance and the gate are the human's."""
    assert _authorize(_command(ev.CommandName.RUN_PROCEDURE), station_info) is None
    assert (
        _authorize(_command(ev.CommandName.PAUSE_PROCEDURE), station_info) is None
    )

    for name in (
        ev.CommandName.SET_EXPERIMENT_ENVELOPE,
        ev.CommandName.SET_ATTENDANCE,
        ev.CommandName.SET_AGENT_GATE,
    ):
        verdict = _authorize(_command(name), station_info)
        assert verdict is not None, f"{name.value} must be refused to every role"
        assert verdict.detail["action_class"] == "envelope"
        assert verdict.detail["rule"] == "role_matrix"


def test_an_unknown_role_is_refused_by_name(station_info):
    """A role that is not in the enum grants nothing, rather than defaulting."""
    stranger = ev.Actor(kind=ev.ActorKind.AGENT, id="x", role="superuser")
    command = _command(ev.CommandName.PAUSE_PROCEDURE, actor=stranger)

    verdict = _authorize(command, station_info)

    assert verdict is not None
    assert verdict.detail == {"rule": "unknown_role", "role": "superuser"}


# ── The kill switch ───────────────────────────────────────────────────────


def test_a_revoked_gate_refuses_even_the_session_role(station_info):
    """The gate subtracts; it can never be talked round by a role."""
    command = _command(ev.CommandName.RUN_PROCEDURE)

    verdict = _authorize(command, station_info, gate=ev.AgentGate.REVOKED)

    assert verdict is not None
    assert verdict.detail["rule"] == "kill_switch"
    assert verdict.detail["gate"] == "revoked"


def test_a_read_only_gate_leaves_reads_alone(station_info):
    """read_only is the middle rung: observe, but write nothing."""
    read = _command(
        ev.CommandName.SUBMIT_VI_ACTION,
        vi_name="dc_measurement",
        method_name="read_now",
    )
    assert _authorize(read, station_info, gate=ev.AgentGate.READ_ONLY) is None

    write = _command(ev.CommandName.PAUSE_PROCEDURE)

    refused = _authorize(write, station_info, gate=ev.AgentGate.READ_ONLY)
    assert refused is not None
    assert refused.detail["gate"] == "read_only"

    assert _authorize(write, station_info, gate=ev.AgentGate.REVOKED) is not None


def test_the_human_path_is_never_judged_here(station_info):
    """An operator or system actor passes every check, at every gate setting."""
    for actor in (ev.OPERATOR, ev.Actor(kind=ev.ActorKind.SYSTEM, id="tick")):
        for gate in ev.AgentGate:
            command = _command(ev.CommandName.SET_AGENT_GATE, actor=actor)
            assert authorize(actor, command, station_info, True, gate) is None


def test_emergency_standby_is_permitted_to_every_role_at_every_gate(station_info):
    """Nobody who can see a problem may be unable to make the station safe."""
    for actor in (OBSERVER, DEBUG, SESSION):
        for gate in ev.AgentGate:
            for attended in (True, False):
                command = _command(
                    ev.CommandName.EMERGENCY_STANDBY,
                    actor=actor,
                    reason="coil voltage climbing",
                )
                assert (
                    authorize(actor, command, station_info, attended, gate) is None
                ), f"{actor.role} refused emergency standby at gate {gate.value}"


def test_the_matrix_is_the_only_thing_that_decides(station_info):
    """Spot-check that the table drives the answer, cell by cell."""
    from i2as.session.gateway import PERMISSION_MATRIX

    assert PERMISSION_MATRIX[ActionClass.RECOVERY][Role.DEBUG] is (
        Permission.UNATTENDED_ONLY
    )
    assert PERMISSION_MATRIX[ActionClass.ENVELOPE][Role.SESSION] is Permission.REFUSED
    assert PERMISSION_MATRIX[ActionClass.READ][Role.OBSERVER] is Permission.PERMITTED


# ══════════════════════════════════════════════════════════════════════════
# The Gateway object: one connection, end to end against a sim station
# ══════════════════════════════════════════════════════════════════════════


class _Recorder:
    """Collect everything one engine said, in emission order."""

    def __init__(self, engine):
        self.verdicts: list[ev.Verdict] = []
        self.events: list[object] = []
        engine.verdict_emitted.connect(self.verdicts.append)
        engine.event_emitted.connect(self.events.append)

    def of_type(self, event_type):
        return [event for event in self.events if isinstance(event, event_type)]


@pytest.fixture
def engine(qtbot):
    """A real Orchestrator over a real simulated station."""
    from i2as.core.orchestrator import Orchestrator

    station = build_station("i2as/configs/sim_cryostat")
    orch = Orchestrator(station, tick_interval_ms=10)
    yield orch, station
    orch.shutdown()


def _gateway(engine, role, actor_id="agent-1"):
    from i2as.session.gateway import Gateway

    orch, station = engine
    return Gateway(orch, role, actor_id, station_info=station.station_info)


def test_the_gateway_stamps_its_identity_on_every_command(engine):
    """A forwarded command names the agent and the role it acted under."""
    orch, _station = engine
    recorder = _Recorder(orch)
    gateway = _gateway(engine, Role.SESSION, actor_id="runner-7")

    request_id = gateway.submit(ev.CommandName.START_MONITORING)

    verdict = recorder.verdicts[-1]
    assert verdict.request_id == request_id
    assert verdict.code is ev.VerdictCode.OK
    assert verdict.actor == ev.Actor(
        kind=ev.ActorKind.AGENT, id="runner-7", role="session"
    )
    assert orch.is_monitoring()


def test_a_refused_command_never_reaches_the_engine(engine):
    """The refusal is answered on the engine's own stream, and nothing runs."""
    orch, _station = engine
    recorder = _Recorder(orch)
    gateway = _gateway(engine, Role.OBSERVER)

    request_id = gateway.submit(ev.CommandName.START_MONITORING)

    assert [v.request_id for v in recorder.verdicts] == [request_id]
    verdict = recorder.verdicts[-1]
    assert verdict.code is ev.VerdictCode.BLOCKED_ROLE
    assert verdict.detail["rule"] == "role_matrix"
    assert verdict.seq > 0
    assert not orch.is_monitoring()


def test_permits_answers_without_provoking_anything(engine):
    """A client can render its own tool list without submitting to find out."""
    orch, _station = engine
    recorder = _Recorder(orch)
    gateway = _gateway(engine, Role.OBSERVER)

    assert gateway.permits(ev.CommandName.RUN_QUEUE) is not None
    assert recorder.verdicts == []


def test_the_gateway_reads_from_its_mirror(engine, qtbot):
    """Every read is answered locally, from what the engine broadcast."""
    orch, _station = engine
    gateway = _gateway(engine, Role.SESSION)

    assert gateway.status() is None
    assert gateway.attended() is True
    assert gateway.agent_gate() is ev.AgentGate.ACTIVE
    assert gateway.station().instruments

    orch.set_attendance(False)
    orch.set_agent_gate(ev.AgentGate.READ_ONLY)

    assert gateway.attended() is False
    assert gateway.agent_gate() is ev.AgentGate.READ_ONLY
    assert gateway.state() == "IDLE"


def test_unattended_widens_debug_to_recovery_end_to_end(engine):
    """The attendance value pushed into the engine changes what the agent may do."""
    orch, _station = engine
    recorder = _Recorder(orch)
    gateway = _gateway(engine, Role.DEBUG)

    orch.set_attendance(True)
    gateway.submit(ev.CommandName.START_MONITORING)
    assert recorder.verdicts[-1].code is ev.VerdictCode.BLOCKED_ROLE
    assert recorder.verdicts[-1].detail["rule"] == "attendance"
    assert not orch.is_monitoring()

    orch.set_attendance(False)
    gateway.submit(ev.CommandName.START_MONITORING)
    assert recorder.verdicts[-1].code is ev.VerdictCode.OK
    assert orch.is_monitoring()


def test_the_kill_switch_refuses_the_agent_and_leaves_the_human_alone(engine):
    """The same command: refused from the gateway, carried out from the GUI path."""
    orch, _station = engine
    recorder = _Recorder(orch)
    gateway = _gateway(engine, Role.SESSION)
    orch.set_agent_gate(ev.AgentGate.REVOKED)

    gateway.submit(ev.CommandName.START_MONITORING)

    assert recorder.verdicts[-1].code is ev.VerdictCode.BLOCKED_ROLE
    assert recorder.verdicts[-1].detail["gate"] == "revoked"
    assert not orch.is_monitoring()

    orch.submit(ev.Command(name=ev.CommandName.START_MONITORING))

    assert recorder.verdicts[-1].code is ev.VerdictCode.OK
    assert orch.is_monitoring()


def test_the_engine_gates_an_agent_that_skips_the_gateway(engine):
    """The gateway is the front door; the engine is the authority behind it."""
    orch, _station = engine
    recorder = _Recorder(orch)
    orch.set_agent_gate(ev.AgentGate.REVOKED)

    orch.submit(ev.Command(name=ev.CommandName.START_MONITORING, actor=SESSION))

    assert recorder.verdicts[-1].code is ev.VerdictCode.BLOCKED_ROLE
    assert recorder.verdicts[-1].detail["gate"] == "revoked"
    assert not orch.is_monitoring()


def test_an_observer_can_always_stand_the_station_down(engine):
    """The exit criterion: emergency standby under a closed gate, from the lowest role."""
    orch, _station = engine
    recorder = _Recorder(orch)
    gateway = _gateway(engine, Role.OBSERVER)
    orch.set_agent_gate(ev.AgentGate.REVOKED)

    gateway.submit(
        ev.CommandName.EMERGENCY_STANDBY, {"reason": "coil voltage climbing"}
    )

    assert recorder.verdicts[-1].code is ev.VerdictCode.OK
    assert orch.state == "EMERGENCY"


def test_a_gateway_refusal_orders_after_what_the_engine_said(engine):
    """A locally emitted verdict still sorts correctly against the engine's stream."""
    orch, _station = engine
    recorder = _Recorder(orch)
    gateway = _gateway(engine, Role.OBSERVER)
    orch.set_attendance(False)
    engine_seq = max(getattr(e, "seq", 0) for e in recorder.events)

    gateway.submit(ev.CommandName.RUN_QUEUE)

    assert recorder.verdicts[-1].seq > engine_seq


# ── The ceiling ───────────────────────────────────────────────────────────


def test_the_ceiling_is_read_off_the_matrix_not_a_second_ordering():
    """A deployment's maximum role follows the table that already exists."""
    from i2as.session.gateway import PERMISSION_MATRIX, role_within_ceiling

    assert role_within_ceiling(Role.OBSERVER, Role.SESSION)
    assert role_within_ceiling(Role.DEBUG, Role.SESSION)
    assert role_within_ceiling(Role.SESSION, Role.SESSION)
    assert not role_within_ceiling(Role.SESSION, Role.DEBUG)
    assert not role_within_ceiling(Role.DEBUG, Role.OBSERVER)
    # Every role is within itself, and observer — which grants read alone —
    # is within every other, straight from the matrix's own cells.
    for role in Role:
        assert role_within_ceiling(role, role)
        assert role_within_ceiling(Role.OBSERVER, role)
    assert set(PERMISSION_MATRIX) == set(ActionClass)


# ══════════════════════════════════════════════════════════════════════════
# The engine client: the same gateway over the engine or over its proxy
# ══════════════════════════════════════════════════════════════════════════


def test_the_two_contract_streams_are_found_under_either_name():
    """``EngineClient`` is satisfied by the engine AND by a client adapter.

    The engine declares ``verdict_emitted``/``event_emitted``; the
    **Orchestrator proxy** consumes those two channels and re-exposes them
    under the contract's own names, ``verdict``/``event``. A gateway must not
    have to know which of the two it is holding — which it did, and building
    the **Gateway server** over the proxy raised at construction because of
    it.
    """
    from i2as.session.gateway.gateway import event_stream, verdict_stream

    class _Engine:
        verdict_emitted = "engine verdicts"
        event_emitted = "engine events"

    class _Client:
        verdict = "client verdicts"
        event = "client events"

    assert verdict_stream(_Engine()) == "engine verdicts"
    assert event_stream(_Engine()) == "engine events"
    assert verdict_stream(_Client()) == "client verdicts"
    assert event_stream(_Client()) == "client events"


def test_an_object_that_is_not_an_engine_client_is_refused_by_name():
    """Neither name means it is not a client at all — say so at the wiring."""
    from i2as.session.gateway.gateway import event_stream, verdict_stream

    with pytest.raises(AttributeError, match="verdict_emitted nor verdict"):
        verdict_stream(object())
    with pytest.raises(AttributeError, match="event_emitted nor event"):
        event_stream(object())


def test_a_gateway_over_the_proxy_submits_and_is_answered(qtbot, engine):
    """The whole point of the duck typing: a gateway on the client side.

    Inline here, so the answering verdict is synchronous; across the
    instrument thread the same wiring is what ``tests/test_gateway_server.py``
    drives over a real socket.
    """
    from i2as.core.orchestrator_proxy import OrchestratorProxy
    from i2as.session.gateway import Gateway

    orch, station = engine
    proxy = OrchestratorProxy(orch)
    verdicts: list[ev.Verdict] = []
    proxy.verdict.connect(verdicts.append)
    gateway = Gateway(proxy, Role.SESSION, "runner-7", station_info=station.station_info)

    request_id = gateway.submit(ev.CommandName.START_MONITORING)

    assert [v.request_id for v in verdicts] == [request_id]
    assert verdicts[-1].code is ev.VerdictCode.OK
    assert orch.is_monitoring()

    # And a refusal, which the gateway itself puts on the same stream.
    refused = Gateway(
        proxy, Role.OBSERVER, "watcher", station_info=station.station_info
    ).submit(ev.CommandName.STOP_MONITORING)

    assert verdicts[-1].request_id == refused
    assert verdicts[-1].code is ev.VerdictCode.BLOCKED_ROLE
    assert orch.is_monitoring()


# ── Run ownership: the same rule, mirrored at the front door ──────────────


OWNER = {"kind": "agent", "id": "agent-A"}
INTRUDER = ev.Actor(kind=ev.ActorKind.AGENT, id="agent-B", role=Role.SESSION.value)


def _owner_scoped(name=ev.CommandName.ABORT_PROCEDURE, actor=INTRUDER, **args):
    return ev.Command(name=name, actor=actor, args=args)


def test_a_non_owner_agent_is_refused_by_the_ownership_rule(station_info):
    """The gateway refuses in the engine's words, naming the owner."""
    verdict = authorize(
        INTRUDER,
        _owner_scoped(),
        station_info,
        True,
        ev.AgentGate.ACTIVE,
        run_owner=OWNER,
    )

    assert verdict is not None
    assert verdict.code is ev.VerdictCode.BLOCKED_ROLE
    assert verdict.detail["rule"] == "run_owner"
    assert verdict.detail["owner"] == OWNER
    assert "override_owner" in verdict.reason


def test_every_owner_scoped_command_is_judged_the_same_way(station_info):
    """All four, and nothing else: a recovery action is untouched by ownership."""
    from i2as.session.gateway.roles import OWNER_SCOPED_COMMANDS

    for name in OWNER_SCOPED_COMMANDS:
        verdict = authorize(
            INTRUDER,
            _owner_scoped(name, key="needle_valve"),
            station_info,
            True,
            ev.AgentGate.ACTIVE,
            run_owner=OWNER,
        )
        assert verdict is not None and verdict.detail["rule"] == "run_owner", name

    for name in (ev.CommandName.PAUSE_PROCEDURE, ev.CommandName.RESUME_PROCEDURE):
        assert (
            authorize(
                INTRUDER,
                _owner_scoped(name),
                station_info,
                True,
                ev.AgentGate.ACTIVE,
                run_owner=OWNER,
            )
            is None
        ), name


def test_the_owner_and_an_unowned_run_are_not_refused(station_info):
    """Ownership says nothing about the owner, or about a station at rest."""
    owner_actor = ev.Actor(kind=ev.ActorKind.AGENT, id="agent-A", role=Role.SESSION.value)
    assert (
        authorize(
            owner_actor,
            _owner_scoped(actor=owner_actor),
            station_info,
            True,
            ev.AgentGate.ACTIVE,
            run_owner=OWNER,
        )
        is None
    )
    assert (
        authorize(
            INTRUDER, _owner_scoped(), station_info, True, ev.AgentGate.ACTIVE
        )
        is None
    )


def test_an_override_needs_a_reason_here_too(station_info):
    """Refuse-then-override, with the same rule name the engine uses."""
    verdict = authorize(
        INTRUDER,
        _owner_scoped(override_owner=True),
        station_info,
        True,
        ev.AgentGate.ACTIVE,
        run_owner=OWNER,
    )
    assert verdict is not None
    assert verdict.detail["rule"] == "override_reason_required"

    assert (
        authorize(
            INTRUDER,
            _owner_scoped(override_owner=True, reason="agent-A stopped answering"),
            station_info,
            True,
            ev.AgentGate.ACTIVE,
            run_owner=OWNER,
        )
        is None
    )


def test_a_debug_agents_override_is_refused_by_the_matrix_not_by_ownership(
    station_info,
):
    """Order is the model: authority it never had is refused on that ground.

    A ``debug`` role may not take run-control actions at all, so it is told
    that — not whose run it is, which would imply the override would work.
    """
    debug = ev.Actor(kind=ev.ActorKind.AGENT, id="fixer", role=Role.DEBUG.value)
    verdict = authorize(
        debug,
        _owner_scoped(actor=debug, override_owner=True, reason="taking over"),
        station_info,
        True,
        ev.AgentGate.ACTIVE,
        run_owner=OWNER,
    )

    assert verdict is not None
    assert verdict.detail["rule"] == "role_matrix"


def test_the_gateway_mirrors_ownership_from_the_status_snapshot(qtbot, engine, tmp_path):
    """End to end on one engine: agent B is refused locally, and it is in the feed."""
    from i2as.session.agent_feed import AgentFeed, read_feed
    from i2as.session.gateway import Gateway

    orch, station = engine
    feed = AgentFeed(tmp_path / "agent_actions.jsonl", "exp-1")
    feed.attach(orch)
    agent_a = _gateway(engine, Role.SESSION, "agent-A")
    agent_b = Gateway(
        orch, Role.SESSION, "agent-B", station_info=station.station_info, feed=feed
    )

    from tests.test_l3_orchestrator import QueueableProcedure

    orch.run_procedure(QueueableProcedure(station), actor=agent_a.actor)
    orch._emit_status_snapshot()
    assert agent_b.run_owner() == {"kind": "agent", "id": "agent-A"}

    answer = agent_b.call_tool("abort_procedure")

    assert answer["ok"] is False
    assert answer["code"] == "BLOCKED_ROLE"
    assert answer["detail"]["rule"] == "run_owner"
    assert orch.state != "IDLE"

    records = read_feed(feed.path)
    # Both halves of the trail: what agent B asked for, and what it was told.
    commands = [r for r in records if r["record"] == "command"]
    verdicts = [r for r in records if r["record"] == "verdict"]
    assert commands[-1]["actor"]["id"] == "agent-B"
    assert commands[-1]["command"] == "abort_procedure"
    assert verdicts[-1]["detail"]["rule"] == "run_owner"

    orch.abort_procedure()
