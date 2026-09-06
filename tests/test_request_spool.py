# ---
# description: |
#   The Request spool (core/request_spool.py) and the tick drain that reads it.
#   Covers the request-file format and its atomic write, the two transport
#   rules (no operator claim, the configured role cap), the malformed-file
#   answer, the verdict sink and the mirrored event tail, and the fact that an
#   engine built without a spool never looks at one.
# last_updated: 2026-09-03
# ---

from __future__ import annotations

import json

import pytest

from i2as.core import events as ev
from i2as.core.orchestrator import Orchestrator
from i2as.core.request_spool import (
    SCHEMA_VERSION,
    RequestSpool,
    spool_directory,
)
from i2as.core.config import read_request_spool_config
from i2as.core.station import build_station
from i2as.session.gateway import Role, authorize_spooled

CONFIG_PATH = "i2as/configs/sim_cryostat"


def _command(name, *, role=Role.SESSION, kind=ev.ActorKind.AGENT, actor_id="ctl-1", **args):
    """Build one command as a spooling client would."""
    return ev.Command(
        name=name,
        actor=ev.Actor(kind=kind, id=actor_id, role=role.value if isinstance(role, Role) else role),
        args=args,
    )


@pytest.fixture
def spool(tmp_path):
    """A spool wired to the real permission model, capped at ``session``."""
    return RequestSpool(
        tmp_path / "spool", max_role=Role.SESSION.value, authorizer=authorize_spooled
    )


@pytest.fixture
def engine(qtbot, spool):
    """A real Orchestrator over a real simulated station, draining *spool*."""
    station = build_station(CONFIG_PATH)
    orchestrator = Orchestrator(station, tick_interval_ms=10, request_spool=spool)
    yield orchestrator
    orchestrator.shutdown()


def _verdicts_for(spool, request_id):
    """Every verdict record in the sink carrying *request_id*."""
    return [
        record
        for record in spool.read_verdicts()
        if record.get("request_id") == request_id
    ]


# ══════════════════════════════════════════════════════════════════════════
# The file format
# ══════════════════════════════════════════════════════════════════════════


def test_a_request_is_written_atomically_and_named_for_its_request_id(spool):
    """The correlation id IS the file name, and nothing partial is ever visible."""
    command = _command(ev.CommandName.START_MONITORING)

    path = spool.write_request(command, Role.SESSION.value)

    assert path.name == f"{command.request_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["role"] == "session"
    assert payload["command"]["name"] == "start_monitoring"
    # The temporary file is dot-prefixed and does not end in .json, so a drain
    # running while it is being written cannot pick it up.
    assert list(spool.requests_dir.glob("*.json")) == [path]


def test_the_spool_root_follows_the_log_directory(monkeypatch, tmp_path):
    """One installation rule for every machine-local directory."""
    monkeypatch.setenv("I2AS_LOG_DIR", str(tmp_path / "logs"))

    assert spool_directory() == tmp_path / "logs" / "spool"


def test_a_spool_is_open_only_once_its_request_directory_exists(tmp_path):
    """The client's 'is the app running with the spool on?' test."""
    spool = RequestSpool(tmp_path / "spool")

    assert spool.is_open() is False
    assert spool.ensure() is True
    assert spool.is_open() is True


# ══════════════════════════════════════════════════════════════════════════
# The drain
# ══════════════════════════════════════════════════════════════════════════


def test_a_dropped_request_is_drained_by_the_next_tick(engine, spool):
    """The exit criterion: one file in, one verdict out, and the file is gone."""
    command = _command(ev.CommandName.START_MONITORING)
    spool.write_request(command, Role.SESSION.value)

    engine._tick()

    answers = _verdicts_for(spool, command.request_id)
    assert len(answers) == 1
    assert answers[0]["code"] == "OK"
    assert answers[0]["command"] == "start_monitoring"
    assert answers[0]["actor"]["kind"] == "agent"
    assert answers[0]["actor"]["role"] == "session"
    assert list(spool.requests_dir.glob("*.json")) == []


def test_a_second_tick_does_not_answer_the_same_request_twice(engine, spool):
    """A request left on disk would be a retry loop; the drain removes it."""
    command = _command(ev.CommandName.START_MONITORING)
    spool.write_request(command, Role.SESSION.value)

    engine._tick()
    engine._tick()

    assert len(_verdicts_for(spool, command.request_id)) == 1


def test_requests_are_drained_oldest_first(engine, spool):
    """The order they were dropped in is the order they are carried out in."""
    first = _command(ev.CommandName.START_MONITORING)
    second = _command(ev.CommandName.STOP_MONITORING)
    spool.write_request(first, Role.SESSION.value)
    spool.write_request(second, Role.SESSION.value)

    engine._tick()

    ordered = [
        record["request_id"]
        for record in spool.read_verdicts()
        if record["request_id"] in {first.request_id, second.request_id}
    ]
    assert ordered == [first.request_id, second.request_id]


def test_a_spooled_vi_action_is_executed_on_the_same_tick(engine, spool):
    """A queued manual action is drained by the step immediately below."""
    command = _command(
        ev.CommandName.SUBMIT_VI_ACTION,
        vi_name="magnet_z",
        method_name="standby",
    )
    spool.write_request(command, Role.SESSION.value)

    engine._tick()

    answers = _verdicts_for(spool, command.request_id)
    assert len(answers) == 1
    assert answers[0]["code"] == "OK"


def test_an_engine_without_a_spool_never_looks_at_one(qtbot, tmp_path):
    """Off by default: the files stay where they are and no answer appears."""
    spool = RequestSpool(tmp_path / "spool", authorizer=authorize_spooled)
    command = _command(ev.CommandName.START_MONITORING)
    spool.write_request(command, Role.SESSION.value)
    station = build_station(CONFIG_PATH)
    orchestrator = Orchestrator(station, tick_interval_ms=10)
    try:
        orchestrator._tick()
    finally:
        orchestrator.shutdown()

    assert list(spool.requests_dir.glob("*.json")) == [
        spool.requests_dir / f"{command.request_id}.json"
    ]
    assert spool.read_verdicts() == []


# ══════════════════════════════════════════════════════════════════════════
# Who may write to it
# ══════════════════════════════════════════════════════════════════════════


def test_an_operator_claim_is_refused(engine, spool):
    """A file on disk is not a human standing at the cryostat."""
    command = _command(
        ev.CommandName.START_MONITORING, kind=ev.ActorKind.OPERATOR, role="operator"
    )
    spool.write_request(command, "operator")

    engine._tick()

    answer = _verdicts_for(spool, command.request_id)[0]
    assert answer["code"] == "BLOCKED_ROLE"
    assert answer["detail"]["rule"] == "spool_actor_kind"
    assert answer["detail"]["kind"] == "operator"


def test_a_system_actor_is_admitted(engine, spool):
    """The other kind a spooled request may claim."""
    command = _command(ev.CommandName.START_MONITORING, kind=ev.ActorKind.SYSTEM)
    spool.write_request(command, Role.SESSION.value)

    engine._tick()

    assert _verdicts_for(spool, command.request_id)[0]["code"] == "OK"


def test_a_role_above_the_cap_is_refused(qtbot, tmp_path):
    """The transport's own bound, checked before the permission matrix."""
    spool = RequestSpool(
        tmp_path / "spool", max_role=Role.OBSERVER.value, authorizer=authorize_spooled
    )
    station = build_station(CONFIG_PATH)
    orchestrator = Orchestrator(station, tick_interval_ms=10, request_spool=spool)
    command = _command(ev.CommandName.START_MONITORING, role=Role.SESSION)
    spool.write_request(command, Role.SESSION.value)
    try:
        orchestrator._tick()
    finally:
        orchestrator.shutdown()

    answer = _verdicts_for(spool, command.request_id)[0]
    assert answer["code"] == "BLOCKED_ROLE"
    assert answer["detail"]["rule"] == "spool_role_cap"
    assert answer["detail"]["max_role"] == "observer"
    assert answer["detail"]["role"] == "session"


def test_the_cap_never_refuses_emergency_standby(qtbot, tmp_path):
    """An actor that can see a problem must always be able to make it safe."""
    spool = RequestSpool(
        tmp_path / "spool", max_role=Role.OBSERVER.value, authorizer=authorize_spooled
    )
    station = build_station(CONFIG_PATH)
    orchestrator = Orchestrator(station, tick_interval_ms=10, request_spool=spool)
    command = _command(
        ev.CommandName.EMERGENCY_STANDBY, role=Role.SESSION, reason="spool test"
    )
    spool.write_request(command, Role.SESSION.value)
    try:
        orchestrator._tick()
    finally:
        orchestrator.shutdown()

    assert _verdicts_for(spool, command.request_id)[0]["code"] == "OK"


def test_the_declared_role_wins_over_the_actors_own(engine, spool):
    """A client that could write its own authority into the actor is not judged."""
    command = _command(ev.CommandName.RUN_QUEUE, role=Role.SESSION)
    spool.write_request(command, Role.OBSERVER.value)

    engine._tick()

    answer = _verdicts_for(spool, command.request_id)[0]
    assert answer["code"] == "BLOCKED_ROLE"
    assert answer["detail"]["rule"] == "role_matrix"
    assert answer["detail"]["role"] == "observer"


def test_a_spool_with_no_permission_model_refuses_everything(qtbot, tmp_path):
    """A door with no lock is not left open."""
    spool = RequestSpool(tmp_path / "spool")
    station = build_station(CONFIG_PATH)
    orchestrator = Orchestrator(station, tick_interval_ms=10, request_spool=spool)
    command = _command(ev.CommandName.START_MONITORING)
    spool.write_request(command, Role.SESSION.value)
    try:
        orchestrator._tick()
    finally:
        orchestrator.shutdown()

    answer = _verdicts_for(spool, command.request_id)[0]
    assert answer["code"] == "BLOCKED_ROLE"
    assert answer["detail"]["rule"] == "spool_no_authorizer"


# ══════════════════════════════════════════════════════════════════════════
# Malformed requests
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "contents",
    [
        "{not json at all",
        "[1, 2, 3]",
        json.dumps({"schema": 99, "role": "session", "command": {"name": "run_queue"}}),
        json.dumps({"schema": SCHEMA_VERSION, "role": "session", "command": {"name": "no_such_command"}}),
        json.dumps({"schema": SCHEMA_VERSION, "role": "session"}),
    ],
)
def test_a_malformed_request_is_answered_failed_and_removed(engine, spool, contents):
    """One answer per request written, even for one that never became a command."""
    spool.requests_dir.mkdir(parents=True, exist_ok=True)
    path = spool.requests_dir / "broken-request.json"
    path.write_text(contents, encoding="utf-8")

    engine._tick()

    assert not path.exists()
    answer = _verdicts_for(spool, "broken-request")[0]
    assert answer["code"] == "FAILED"
    assert answer["detail"]["rule"] == "malformed_request"
    assert answer["command"] is None


def test_a_request_whose_id_disagrees_with_its_file_name_is_refused(engine, spool):
    """The file's name is the correlation id, so the two may not disagree."""
    spool.requests_dir.mkdir(parents=True, exist_ok=True)
    command = _command(ev.CommandName.START_MONITORING)
    path = spool.requests_dir / "some-other-id.json"
    path.write_text(
        json.dumps(
            {"schema": SCHEMA_VERSION, "role": "session", "command": command.to_json()}
        ),
        encoding="utf-8",
    )

    engine._tick()

    answer = _verdicts_for(spool, "some-other-id")[0]
    assert answer["code"] == "FAILED"
    assert answer["command"] == "start_monitoring"


# ══════════════════════════════════════════════════════════════════════════
# The mirror a client reads
# ══════════════════════════════════════════════════════════════════════════


def test_the_engine_mirrors_its_declaration_and_its_status(engine, spool):
    """An out-of-process client answers reads from files, never by asking."""
    engine._tick()

    station_info = spool.latest_station()
    status = spool.latest_status()

    assert station_info is not None
    assert station_info.setup == "sim_cryostat"
    assert status is not None
    assert status.state


def test_the_event_tail_is_size_capped(qtbot, tmp_path):
    """A mirror for a client that just started, not an archive."""
    spool = RequestSpool(tmp_path / "spool", event_cap_bytes=4096)
    station = build_station(CONFIG_PATH)
    orchestrator = Orchestrator(station, tick_interval_ms=10, request_spool=spool)
    try:
        for _ in range(60):
            orchestrator._tick()
    finally:
        orchestrator.shutdown()

    assert spool.events_path.stat().st_size <= 4096 * 2
    assert spool.latest_status() is not None


def test_every_verdict_reaches_the_sink_not_only_spooled_ones(engine, spool):
    """A client watching a running experiment sees what the window sees."""
    engine.submit(ev.Command(name=ev.CommandName.START_MONITORING))

    records = spool.read_verdicts()

    assert [record["actor"]["kind"] for record in records] == ["operator"]


# ══════════════════════════════════════════════════════════════════════════
# The setup property
# ══════════════════════════════════════════════════════════════════════════


def test_the_spool_is_off_and_capped_at_observer_by_default():
    """A setup that has not asked for the door does not have one."""
    settings = read_request_spool_config(CONFIG_PATH)

    assert settings == {"enabled": False, "max_role": "observer"}


def test_a_setup_may_declare_the_spool_and_its_cap(tmp_path):
    """Both are setup properties, so both live in monitor.yaml."""
    (tmp_path / "monitor.yaml").write_text(
        "monitor:\n"
        "  tick_interval_ms: 3000\n"
        "  request_spool: true\n"
        "  spool_max_role: session\n",
        encoding="utf-8",
    )

    settings = read_request_spool_config(str(tmp_path))

    assert settings == {"enabled": True, "max_role": "session"}
