# ---
# description: |
#   The agent-gateway exit scenario, driven end to end through the Reference
#   client's own main(): validate, probe-run, start and abort a FieldSweep on
#   a simulated station, with the envelope, attendance, role and kill-switch
#   refusals each asserted by its structured reason, and every request the
#   client made present in the experiment's agent feed with its verdict.
# last_updated: 2026-09-03
# ---

from __future__ import annotations

import json

import pytest

from i2as.core import events as ev
from i2as.core.station import build_station
from i2as.ctl.cli import EXIT_OK, EXIT_REFUSED, main
from i2as.ctl.client import CtlClient, open_client
from i2as.session.gateway import Role
from i2as.session.models import User

CONFIG_PATH = "i2as/configs/sim_cryostat"

SAMPLE_INFO = {"sample_name": "S", "sample_id": "S-1", "comments": ""}

FIELD_SWEEP_PARAMS = {
    "measurement_vi": "dc_measurement",
    "field_start": -1.0,
    "field_end": 1.0,
    "field_steps": 21,
    "temperature": 300.0,
    "current_A": 1e-6,
    "readings_per_point": 3,
    "init_wait": 300.0,
    "step_wait": 30.0,
}


@pytest.fixture(autouse=True)
def isolated_installation(tmp_path, monkeypatch):
    """Keep the whole scenario inside this test's own tmp_path."""
    monkeypatch.setenv("I2AS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("I2AS_MEASUREMENT_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr("i2as.ctl.cli.setup_logging", lambda *a, **k: None)


@pytest.fixture
def fast_station(monkeypatch):
    """Build the client's station with a magnet that ramps at test speed.

    The sim magnet ramps at a realistic rate, and a tick-by-tick scenario
    cannot wait for it — the same treatment the engine and gateway suites
    give it. Nothing else about the station changes.
    """
    from i2as.ctl import client as ctl_client

    def _build(config_path: str):
        station = build_station(config_path)
        station.magnet_z._default_ramp_rate = 6000.0
        station.magnet_z._ramp_segments = []
        return station

    monkeypatch.setattr(ctl_client, "build_station", _build)


@pytest.fixture
def scenario(qtbot, fast_station):
    """A session-role client with an experiment the physicist has opened.

    The client never opens the experiment: that is `envelope`-class work,
    which the permission matrix grants to no role. It is opened here through
    the session layer, which is what the human's window does.
    """
    client = open_client(
        offline=CONFIG_PATH, role=Role.SESSION.value, actor_id="ctl-scenario"
    )
    manager = client.experiments
    manager.roster.add(User(user_id="jdoe", name="J. Doe", email="jdoe@example.org"))
    manager.start_experiment("Phase E", "jdoe", dict(SAMPLE_INFO))
    yield client, manager
    client.close()


def _run(capsys, client, argv):
    """Drive one ctl invocation against the scenario's stack.

    Args:
        capsys: pytest's capture fixture.
        client: The open client every invocation shares, so the station
            remembers what the last command did.
        argv: The argument list.

    Returns:
        ``(exit code, answer)``.
    """
    code = main(list(argv), client=client)
    return code, json.loads(capsys.readouterr().out)


def _tick_until(client, predicate, max_ticks: int = 4000):
    """Run the engine's tick until *predicate* holds.

    Args:
        client: The client whose engine to tick.
        predicate: Zero-argument callable.
        max_ticks: Give up after this many ticks.

    Raises:
        AssertionError: If the predicate never holds.
    """
    for _ in range(max_ticks):
        client.pump(1)
        if predicate():
            return
    raise AssertionError("the engine never reached the expected state")


# ══════════════════════════════════════════════════════════════════════════
# The scenario
# ══════════════════════════════════════════════════════════════════════════


def test_a_client_validates_probes_runs_and_aborts_a_field_sweep(capsys, scenario):
    """The Phase E exit scenario, one ctl invocation per step.

    Everything a client does to a real experiment, in order and through the
    published grammar: ask whether the run is allowed and how long it would
    take, prove it on the instruments as a probe run, start the real one, and
    stop it. Each step is answered by exactly one verdict carrying the
    request id the invocation printed.
    """
    client, manager = scenario
    data_dir = str(manager.current_data_dir())

    code, monitoring = _run(capsys, client, ["call", "start_monitoring"])
    assert (code, monitoring["code"]) == (EXIT_OK, "OK")

    # 1. Validate: may I run this, and how long would it take?
    code, validated = _run(
        capsys,
        client,
        [
            "call",
            "validate_run",
            "--args",
            json.dumps(
                {
                    "procedure": "FieldSweep",
                    "params": {**FIELD_SWEEP_PARAMS, "field_steps": 5},
                    "sample_info": dict(SAMPLE_INFO),
                    "data_directory": data_dir,
                }
            ),
        ],
    )
    assert code == EXIT_OK
    assert validated["result"]["ok"] is True
    assert validated["result"]["duration_estimate_s"] > 0
    assert validated["result"]["estimate"]["assumptions"]

    # 2. Probe first: the same procedure on the same instruments, subsampled.
    code, probed = _run(
        capsys,
        client,
        [
            "call",
            "probe_run",
            "--args",
            json.dumps(
                {
                    "procedure": "FieldSweep",
                    "params": {**FIELD_SWEEP_PARAMS, "field_steps": 51},
                    "sample_info": dict(SAMPLE_INFO),
                    "data_directory": data_dir,
                    "file_prefix": "probe",
                    "probe_spec": {"n_points": 3, "averaging": 2, "max_wait_s": 0.0},
                }
            ),
        ],
    )
    assert (code, probed["code"]) == (EXIT_OK, "OK")

    _tick_until(
        client, lambda: manager.current_experiment().runs[-1].status == "done"
    )
    code, listed = _run(capsys, client, ["runs"])
    assert code == EXIT_OK
    assert [run["kind"] for run in listed["result"]["runs"]] == ["probe"]

    # 3. The real run, started and then stopped.
    code, started = _run(
        capsys,
        client,
        [
            "call",
            "run_procedure",
            "--args",
            json.dumps(
                {
                    "procedure": "FieldSweep",
                    "params": {**FIELD_SWEEP_PARAMS, "field_steps": 5},
                    "sample_info": dict(SAMPLE_INFO),
                    "data_directory": data_dir,
                    "file_prefix": "sweep",
                }
            ),
        ],
    )
    assert (code, started["code"]) == (EXIT_OK, "OK")
    _tick_until(client, lambda: client.gateway.state() != "IDLE")

    code, aborted = _run(capsys, client, ["abort"])
    assert (code, aborted["code"]) == (EXIT_OK, "OK")
    _tick_until(client, lambda: client.gateway.state() == "IDLE")
    assert manager.current_experiment().runs[-1].status == "aborted"


def test_the_four_refusals_each_carry_their_own_structured_reason(capsys, scenario):
    """Envelope, attendance, role and kill switch, each named by its rule.

    A client branches on ``detail.rule``, never on the prose, so each of the
    four ways authority can run out has to be distinguishable in the answer
    itself — and none of them is a crash or a silent no-op.
    """
    client, _manager = scenario
    engine = client.engine

    # Envelope is nobody's: not even the session role may widen the bounds it
    # is judged by.
    code, envelope = _run(
        capsys,
        client,
        ["call", "set_experiment_envelope", "--args", '{"envelope": null}'],
    )
    assert code == EXIT_REFUSED
    assert envelope["code"] == "BLOCKED_ROLE"
    assert envelope["detail"]["rule"] == "role_matrix"
    assert envelope["detail"]["action_class"] == "envelope"

    # Role: an observer reads, and takes no recovery or run-control action.
    watcher = CtlClient(
        mode=client.mode,
        engine=engine,
        role=Role.OBSERVER,
        actor_id="ctl-watcher",
        experiments=client.experiments,
    )
    code, refused = _run(capsys, watcher, ["pause"])
    assert code == EXIT_REFUSED
    assert refused["detail"]["rule"] == "role_matrix"
    assert refused["detail"]["role"] == "observer"
    assert refused["detail"]["action_class"] == "recovery"

    # Attendance: a debug agent may recover only while nobody is watching.
    engine.set_attendance(True)
    client.pump(1)
    debugger = CtlClient(
        mode=client.mode,
        engine=engine,
        role=Role.DEBUG,
        actor_id="ctl-debug",
        experiments=client.experiments,
    )
    code, attended = _run(capsys, debugger, ["pause"])
    assert code == EXIT_REFUSED
    assert attended["detail"]["rule"] == "attendance"
    assert attended["detail"]["role"] == "debug"

    # Kill switch: the human closes it and the session role loses everything
    # but reads — while emergency standby still passes, at every setting.
    engine.set_agent_gate(ev.AgentGate.READ_ONLY)
    client.pump(1)
    code, gated = _run(capsys, client, ["pause"])
    assert code == EXIT_REFUSED
    assert gated["detail"]["rule"] == "kill_switch"

    code, still_readable = _run(capsys, client, ["status"])
    assert code == EXIT_OK
    assert still_readable["result"]["agent_gate"] == "read_only"

    code, safe = _run(
        capsys, client, ["emergency-standby", "--reason", "the scenario says so"]
    )
    assert (code, safe["code"]) == (EXIT_OK, "OK")


def test_every_request_is_in_the_feed_with_its_verdict(capsys, scenario):
    """The accountability half: what was asked, and what it was told.

    The feed is joined on ``request_id`` — one command record per request the
    client made, forwarded or refused, and the single verdict that answered
    it — so the trail can be read without the engine that produced it.
    """
    client, _manager = scenario

    _, allowed = _run(capsys, client, ["call", "start_monitoring"])
    _, refused = _run(
        capsys,
        client,
        ["call", "set_experiment_envelope", "--args", '{"envelope": null}'],
    )

    code, feed = _run(capsys, client, ["feed", "--last", "200"])
    records = feed["result"]["records"]
    commands = {r["request_id"]: r for r in records if r["record"] == "command"}
    verdicts = {r["request_id"]: r for r in records if r["record"] == "verdict"}

    assert code == EXIT_OK
    # What an agent TRIED is as much of the trail as what it managed.
    assert {allowed["request_id"], refused["request_id"]} <= set(commands)
    assert set(commands) <= set(verdicts)
    assert verdicts[allowed["request_id"]]["verdict"]["code"] == "OK"
    assert verdicts[refused["request_id"]]["verdict"]["code"] == "BLOCKED_ROLE"
    # Every record names who acted, and it is the client's own actor id.
    assert {r["actor"]["id"] for r in commands.values()} == {"ctl-scenario"}
    assert commands[allowed["request_id"]]["command"] == "start_monitoring"


def test_the_feed_records_a_reading_client_asking_nothing_of_the_instrument(
    capsys, scenario
):
    """Reads are not actions: asking is not acting, and the trail says so."""
    client, _manager = scenario

    _run(capsys, client, ["status"])
    _run(capsys, client, ["manifest"])
    code, feed = _run(capsys, client, ["feed", "--last", "200"])

    assert code == EXIT_OK
    assert [r for r in feed["result"]["records"] if r["record"] == "command"] == []
