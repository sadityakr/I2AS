# ---
# description: |
#   Tests for the Reference client (i2as/ctl/): the command grammar, the
#   JSON answer shape and the three exit codes, in both modes — an --offline
#   client over a real simulated station, and a live client writing into a
#   Request spool that a test-ticked engine drains.
# last_updated: 2026-09-03
# ---

from __future__ import annotations

import json

import pytest

from i2as.core.orchestrator import Orchestrator
from i2as.core.request_spool import RequestSpool
from i2as.core.station import build_station
from i2as.ctl.cli import EXIT_OK, EXIT_REFUSED, EXIT_UNREACHABLE, build_parser, main
from i2as.ctl.client import CtlUnreachable, default_actor_id, open_client
from i2as.ctl.discovery import discover_run_catalog
from i2as.session.gateway import Role, authorize_spooled
from i2as.session.models import GUEST_USER_ID
from i2as.session.store import SessionStore

CONFIG_PATH = "i2as/configs/sim_cryostat"


@pytest.fixture(autouse=True)
def quiet_logging(monkeypatch):
    """Leave the process's logging alone.

    ``main()`` configures logging like any entry point; under pytest's stream
    capture that binds a handler to a stream the next test closes, which
    turns every later log line into noise. Logging setup is not what these
    tests are about.
    """
    monkeypatch.setattr("i2as.ctl.cli.setup_logging", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def isolated_installation(tmp_path, monkeypatch):
    """Point every machine-local directory at this test's own tmp_path.

    The client resolves its spool from the log directory and its session
    store from the measurement root, both of which are real installation
    directories; a test must never write into the developer's.
    """
    monkeypatch.setenv("I2AS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("I2AS_MEASUREMENT_ROOT", str(tmp_path / "data"))


def _run(capsys, argv, **kwargs):
    """Run one ctl invocation and return ``(exit code, parsed answer)``.

    Args:
        capsys: pytest's capture fixture.
        argv: The argument list, without the program name.
        **kwargs: Passed to ``main()`` (``client=`` for a shared stack).

    Returns:
        The exit code and the JSON object printed on stdout.
    """
    code = main(list(argv), **kwargs)
    printed = capsys.readouterr().out
    return code, json.loads(printed)


# ══════════════════════════════════════════════════════════════════════════
# The grammar
# ══════════════════════════════════════════════════════════════════════════


def test_the_subcommand_grammar_is_the_documented_one():
    """Command grammar is API: a harness and an allowlist hard-code it."""
    parser = build_parser()
    actions = [
        action
        for action in parser._actions  # noqa: SLF001 — argparse exposes no reader
        if getattr(action, "choices", None) and hasattr(action.choices, "keys")
    ]

    assert set(actions[0].choices) == {
        "tools",
        "schema",
        "call",
        "status",
        "station",
        "manifest",
        "runs",
        "feed",
        "pause",
        "resume",
        "abort",
        "emergency-standby",
    }


def test_the_declared_role_defaults_to_the_narrowest_one():
    """A client that says nothing about its authority gets reads and nothing more."""
    args = build_parser().parse_args(["status"])

    assert args.role == Role.OBSERVER.value
    assert args.offline is None  # live is the default; --offline is the exception


def test_an_unknown_role_is_refused_by_the_parser():
    """The role vocabulary is closed, so a typo never becomes a weaker check."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--role", "admin", "status"])


def test_the_default_actor_names_the_shell_and_the_machine():
    """An unattributed action is exactly what the agent feed exists to prevent."""
    actor = default_actor_id()

    assert actor.startswith("ctl:")
    assert "@" in actor


def test_discovery_finds_the_shipped_procedures():
    """The run catalog an entry point owns, so a run can travel as a class name."""
    catalog = discover_run_catalog()

    assert "FieldSweep" in catalog
    assert all(isinstance(cls, type) for cls in catalog.values())


# ══════════════════════════════════════════════════════════════════════════
# Offline: a whole stack in this process
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def offline(qtbot):
    """One offline client, shared across the invocations of a scenario."""
    client = open_client(offline=CONFIG_PATH, role=Role.SESSION.value, actor_id="ctl-test")
    yield client
    client.close()


def test_the_tool_surface_is_published_whole(capsys, offline):
    """The subcommands are the tool surface, not a list of their own."""
    code, answer = _run(capsys, ["tools"], client=offline)

    names = {tool["name"] for tool in answer["tools"]}
    assert code == EXIT_OK
    assert answer["mode"] == "offline"
    assert answer["count"] == len(answer["tools"]) > 40
    assert {"run_procedure", "read_status", "magnet_z__set_field"} <= names


def test_a_tools_schema_carries_the_configured_bound(capsys, offline):
    """An agent reads the bound off the tool, and the bound is the setup's."""
    code, answer = _run(capsys, ["schema", "magnet_z__set_field"], client=offline)

    target = answer["schema"]["input_schema"]["properties"]["target_T"]
    assert code == EXIT_OK
    assert (target["minimum"], target["maximum"]) == (-9.0, 9.0)
    assert answer["action_class"] == "run_control"


def test_an_unknown_tool_is_refused_by_name(capsys, offline):
    """Refused, not crashed — and the caller branches on the rule, not the prose."""
    code, answer = _run(capsys, ["schema", "no_such_tool"], client=offline)

    assert code == EXIT_REFUSED
    assert answer["detail"]["rule"] == "unknown_tool"


def test_arguments_that_are_not_a_json_object_are_refused(capsys, offline):
    """JSON in and JSON out: --args is an object of the tool's parameters."""
    code, answer = _run(capsys, ["call", "read_status", "--args", "[1, 2]"], client=offline)

    assert code == EXIT_REFUSED
    assert answer["detail"]["rule"] == "args"


def test_arguments_outside_a_tools_bound_are_refused_before_a_command_exists(
    capsys, offline
):
    """The schema refuses it, naming the bound, and no Command is ever built."""
    code, answer = _run(
        capsys,
        ["call", "magnet_z__set_field", "--args", '{"target_T": 99.0}'],
        client=offline,
    )

    assert code == EXIT_REFUSED
    assert answer["detail"]["rule"] == "schema"
    assert "request_id" not in answer


def test_the_reads_answer_from_the_mirror(capsys, offline):
    """Status, declaration and manifest, without asking the engine anything."""
    status_code, status = _run(capsys, ["status"], client=offline)
    station_code, station = _run(capsys, ["station"], client=offline)
    manifest_code, manifest = _run(capsys, ["manifest"], client=offline)

    assert (status_code, station_code, manifest_code) == (EXIT_OK, EXIT_OK, EXIT_OK)
    assert status["result"]["state"] == "IDLE"
    assert {i["name"] for i in station["result"]["instruments"]} >= {"magnet_z"}
    assert manifest["result"]["setup"] == "sim_cryostat"


def test_a_command_is_answered_by_its_verdict(capsys, offline):
    """One request, one verdict, exit 0 — the whole contract in one invocation."""
    code, answer = _run(capsys, ["call", "start_monitoring"], client=offline)

    assert code == EXIT_OK
    assert answer["code"] == "OK"
    assert answer["verdict"]["request_id"] == answer["request_id"]
    assert answer["verdict"]["actor"] == {
        "kind": "agent",
        "id": "ctl-test",
        "role": "session",
    }


def test_a_queued_capability_action_is_settled_before_the_answer_is_printed(
    capsys, offline
):
    """submit_vi_action is carried out by the tick, so the client runs the tick."""
    _run(capsys, ["call", "start_monitoring"], client=offline)

    code, answer = _run(
        capsys,
        ["call", "magnet_z__set_field", "--args", '{"target_T": 0.05}'],
        client=offline,
    )

    assert code == EXIT_OK
    assert answer["code"] == "OK"  # not PENDING: the tick ran here
    assert answer["verdict"]["request_id"] == answer["request_id"]


def test_a_role_that_does_not_grant_the_action_gets_a_structured_refusal(capsys):
    """The permission matrix, reported as a rule a caller can branch on."""
    client = open_client(offline=CONFIG_PATH, role=Role.OBSERVER.value, actor_id="watcher")
    try:
        code, answer = _run(capsys, ["pause"], client=client)
    finally:
        client.close()

    assert code == EXIT_REFUSED
    assert answer["code"] == "BLOCKED_ROLE"
    assert answer["detail"]["rule"] == "role_matrix"
    assert answer["detail"]["role"] == "observer"


def test_a_session_tool_with_no_open_experiment_refuses_by_name(capsys, offline):
    """A client never opens an experiment; with none open it says so."""
    code, answer = _run(capsys, ["runs"], client=offline)

    assert code == EXIT_REFUSED
    assert answer["detail"]["rule"] == "no_experiment"


def test_a_config_that_will_not_build_is_exit_two(capsys):
    """Nothing was asked, so it is not a refusal — it is 'I never got to ask'."""
    code, answer = _run(capsys, ["--offline", "/no/such/config", "status"])

    assert code == EXIT_UNREACHABLE
    assert answer["detail"]["rule"] == "unreachable"


def test_the_shorthands_and_the_equivalent_call_are_the_same_request(capsys, offline):
    """A shorthand differs from `call` in nothing but the words typed."""
    short_code, short = _run(capsys, ["emergency-standby", "--reason", "test"], client=offline)
    long_code, long_form = _run(
        capsys,
        ["call", "emergency_standby", "--args", '{"reason": "test"}'],
        client=offline,
    )

    assert (short_code, long_code) == (EXIT_OK, EXIT_OK)
    assert short["tool"] == long_form["tool"] == "emergency_standby"
    assert short["code"] == long_form["code"] == "OK"


# ══════════════════════════════════════════════════════════════════════════
# Live: one request file, one verdict line
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def ticking_spool(qtbot, tmp_path, monkeypatch):
    """Build a spool a test-ticked engine drains, standing in for a running app.

    The application's tick is what drains the spool; a test has no event
    loop, so the client's wait runs it. Everything else — the request file,
    the verdict line, the mirrored declaration — is the real thing. An active
    session is created too, because that is what a live client reads its
    ``runs`` and ``feed`` out of.
    """
    SessionStore(tmp_path / "data" / "sessions").set_active(
        GUEST_USER_ID,
        SessionStore(tmp_path / "data" / "sessions")
        .create_session(name=GUEST_USER_ID, user_id=GUEST_USER_ID)
        .session_id,
    )
    engines = []

    def _build(max_role: str = Role.SESSION.value):
        root = tmp_path / f"spool-{max_role}"
        spool = RequestSpool(root, max_role=max_role, authorizer=authorize_spooled)
        engine = Orchestrator(
            build_station(CONFIG_PATH), tick_interval_ms=10, request_spool=spool
        )
        engines.append(engine)
        engine._tick()  # noqa: SLF001 — a first status for the client to mirror

        original = RequestSpool.wait_for_verdict

        def _ticking(self, request_id, timeout_s, poll_s=0.05):
            engine._tick()  # noqa: SLF001 — the running application's own tick
            return original(self, request_id, 0.0, 0.0)

        monkeypatch.setattr(RequestSpool, "wait_for_verdict", _ticking)
        return root, spool, engine

    yield _build
    for engine in engines:
        engine.shutdown()


@pytest.fixture
def live(ticking_spool):
    """A live spool whose cap is wide enough for run control."""
    return ticking_spool()


def test_a_live_read_is_answered_from_the_spools_mirror(capsys, live):
    """No engine is disturbed: station.json and events.jsonl are the answer."""
    root, _spool, _engine = live

    code, answer = _run(capsys, ["--spool", str(root), "status"])

    assert code == EXIT_OK
    assert answer["mode"] == "live"
    assert answer["result"]["state"] == "IDLE"


def test_a_live_command_is_a_request_file_answered_by_a_verdict_line(capsys, live):
    """The whole live path: one file in, one verdict out, exit 0."""
    root, spool, _engine = live

    code, answer = _run(
        capsys,
        ["--spool", str(root), "--role", "session", "call", "start_monitoring"],
        client=None,
    )

    assert code == EXIT_OK
    assert answer["code"] == "OK"
    assert answer["verdict"]["actor"]["kind"] == "agent"
    # The request file is gone: the tick drains what it carries out.
    assert list(spool.requests_dir.glob("*.json")) == []
    assert any(
        record["request_id"] == answer["request_id"] for record in spool.read_verdicts()
    )


def test_a_role_above_the_spools_cap_is_refused_at_the_engine(capsys, ticking_spool):
    """The transport's cap subtracts, and the refusal names the rule."""
    root, _spool, _engine = ticking_spool(Role.OBSERVER.value)

    code, answer = _run(
        capsys,
        ["--spool", str(root), "--role", "session", "call", "start_monitoring"],
    )

    assert code == EXIT_REFUSED
    assert answer["code"] == "BLOCKED_ROLE"
    assert answer["detail"]["rule"] == "spool_role_cap"
    assert answer["detail"]["max_role"] == "observer"


def test_no_spool_at_all_is_exit_two(capsys, tmp_path):
    """The application is not running, which is not the same as being refused."""
    code, answer = _run(capsys, ["--spool", str(tmp_path / "absent"), "status"])

    assert code == EXIT_UNREACHABLE
    assert answer["detail"]["rule"] == "unreachable"
    assert "request_spool" in answer["reason"]


def test_a_verdict_that_never_arrives_is_exit_two(capsys, tmp_path, qtbot):
    """A spool nobody drains times out; it is still 'I never got to ask'."""
    root = tmp_path / "quiet"
    RequestSpool(root).ensure()

    code, answer = _run(
        capsys,
        ["--spool", str(root), "--role", "session", "--timeout", "0", "call", "start_monitoring"],
    )

    assert code == EXIT_UNREACHABLE
    assert answer["detail"]["rule"] == "unreachable"


def test_a_live_client_is_refused_before_it_writes_anything(capsys, live):
    """The front door: an observer's run-control command never becomes a file."""
    root, spool, _engine = live

    code, answer = _run(capsys, ["--spool", str(root), "abort"])

    assert code == EXIT_REFUSED
    assert answer["code"] == "BLOCKED_ROLE"
    assert list(spool.requests_dir.glob("*.json")) == []


def test_validate_run_out_of_process_refuses_naming_the_station(capsys, live):
    """A live client has no Station, and says so rather than answering 'no findings'."""
    root, _spool, _engine = live

    code, answer = _run(
        capsys,
        ["--spool", str(root), "call", "validate_run", "--args", '{"procedure": "FieldSweep"}'],
    )

    assert code == EXIT_REFUSED
    assert answer["detail"] == {"rule": "missing_collaborator", "collaborator": "station"}


def test_open_client_raises_the_one_unreachable_error(tmp_path):
    """CtlUnreachable is the only 'never got to ask', so exit 2 has one source."""
    with pytest.raises(CtlUnreachable):
        open_client(spool_root=tmp_path / "nothing-here")
