# ---
# description: |
#   End-to-end scenario tests for the family "a measurement driven by an
#   agent", run through every client surface the Agent gateway offers: the
#   in-process Gateway (threaded, over a real InstrumentHost), role and
#   kill-switch and attendance refusals mid-run, i2as.ctl offline, the
#   Request spool, the local socket and the MCP shim over it, and two agents
#   at once. Each surface drives the same story —
#   read status, read manifest, validate_run, probe_run, run a FieldSweep,
#   watch it, then stop it — so a defect that shows up on one surface and not
#   another is a real inconsistency, not a fluke of one harness. Two of them
#   also drive the run-ownership standard: a second actor is refused the
#   first's run, and takes it over on the record.
# last_updated: 2026-09-04
# ---

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from PyQt6.QtNetwork import QLocalSocket

from i2as.core import events as ev
from i2as.core.data_reader import open_run
from i2as.core.instrument_host import InstrumentHost
from i2as.core.orchestrator import Orchestrator
from i2as.core.request_spool import RequestSpool
from i2as.core.station import build_station
from i2as.ctl.cli import EXIT_OK, EXIT_REFUSED, main as ctl_main
from i2as.ctl.client import open_client
from i2as.mcp import translate as mcp_translate
from i2as.procedures.field_sweep import FieldSweep
from i2as.session.agent_feed import AgentFeed, read_feed
from i2as.session.gateway import (
    Gateway,
    GatewayServer,
    Role,
    ToolContext,
    authorize_spooled,
)
from i2as.session.gateway.gateway import event_stream, verdict_stream
from i2as.session.gateway.local_server import ROLE_REFUSED
from i2as.session.manager import ExperimentManager
from i2as.session.models import User
from i2as.session.store import ExperimentStore, UserRoster
from tests.instrument_modes import shutdown_host

CONFIG_PATH = "i2as/configs/sim_cryostat"

TOKEN = "test-token-not-a-secret"  # noqa: S105 — a fixture value, not a secret

SAMPLE_INFO = {"sample_name": "S", "sample_id": "S-1", "comments": ""}

#: A FieldSweep that reaches a non-IDLE state on the first tick or two; the
#: settle waits (init_wait/step_wait) are real wall-clock seconds inside the
#: Orchestrator, so every scenario here either aborts the run shortly after
#: it starts (mirroring tests/test_phase_e_scenario.py and
#: tests/test_gateway_server.py) or drives it as a probe run, whose
#: ``probe_spec`` caps every declared wait near zero.
FAST_PARAMS = {
    "measurement_vi": "dc_measurement",
    "field_start": -1.0,
    "field_end": 1.0,
    "field_steps": 5,
    "temperature": 300.0,
    "current_A": 1e-6,
    "readings_per_point": 3,
    "init_wait": 300.0,
    "step_wait": 30.0,
}

PROBE_PARAMS = {**FAST_PARAMS, "field_steps": 51}

PROBE_SPEC = {"n_points": 3, "averaging": 2, "max_wait_s": 0.0}

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fast_station(config_path: str = CONFIG_PATH):
    """Build a sim station whose magnet ramps at test speed."""
    station = build_station(config_path)
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []
    return station


def _tick_until(engine, predicate, max_ticks: int = 4000) -> None:
    """Tick *engine* until *predicate* holds; assert that it eventually does."""
    for _ in range(max_ticks):
        engine._tick()
        if predicate():
            return
    raise AssertionError("the engine never reached the expected state")


class _Recorder:
    """Collect everything one engine (or client adapter) said, in order."""

    def __init__(self, client) -> None:
        self.verdicts: list[ev.Verdict] = []
        self.events: list[object] = []
        verdict_stream(client).connect(self.verdicts.append)
        event_stream(client).connect(self.events.append)

    def verdict_for(self, request_id: str) -> ev.Verdict | None:
        return next((v for v in self.verdicts if v.request_id == request_id), None)


def _run_procedure_args(data_dir: str, *, prefix: str = "sweep") -> dict:
    return {
        "procedure": "FieldSweep",
        "params": dict(FAST_PARAMS),
        "sample_info": dict(SAMPLE_INFO),
        "data_directory": data_dir,
        "file_prefix": prefix,
    }


def _probe_run_args(data_dir: str, *, prefix: str = "probe") -> dict:
    return {
        "procedure": "FieldSweep",
        "params": dict(PROBE_PARAMS),
        "sample_info": dict(SAMPLE_INFO),
        "data_directory": data_dir,
        "file_prefix": prefix,
        "probe_spec": dict(PROBE_SPEC),
    }


# ══════════════════════════════════════════════════════════════════════════
# 1. In-process Gateway, threaded mode: the full story
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def threaded_stack(qtbot, tmp_path):
    """A real InstrumentHost in threaded mode — the wiring the app itself uses.

    The Gateway attaches to the **Orchestrator proxy**, exactly as
    ``i2as.main`` wires the Gateway server, so
    a command tool answers ``PENDING`` until its verdict crosses back over
    the instrument thread — the asynchronous half of the client boundary a
    plain-Orchestrator fixture never exercises.
    """
    host = InstrumentHost(
        _fast_station,
        mode="threaded",
        orchestrator_options={
            "tick_interval_ms": 20,
            "run_catalog": {"FieldSweep": FieldSweep},
        },
    )
    host.start()
    proxy = host.build_proxy()

    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe", email="jdoe@example.org"))
    store = ExperimentStore(tmp_path / "experiments")
    manager = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=proxy,
        config_name="sim_cryostat",
        station=host.station,
        run_catalog={"FieldSweep": FieldSweep},
    )
    experiment = manager.start_experiment("Scenario 1", "jdoe", dict(SAMPLE_INFO))
    feed = AgentFeed(
        store.agent_feed_path(experiment.experiment_id), experiment.experiment_id
    )
    feed.attach(host.orchestrator)

    yield host, proxy, manager, feed, tmp_path

    shutdown_host(host)


def test_the_full_story_through_an_in_process_gateway_threaded(qtbot, threaded_stack):
    """Read, validate, probe, run, watch, abort — over the real instrument thread.

    Every request lands in the agent feed joined to its verdict by
    ``request_id``; the probe run's file declares ``run_kind`` "probe" and
    the real run's record carries "run"; the ``StateChange`` the run causes
    reaches the GUI-side proxy naming the agent that caused it; and both run
    records name an agent actor.
    """
    host, proxy, manager, feed, tmp_path = threaded_stack
    data_dir = str(manager.current_data_dir())
    recorder = _Recorder(proxy)
    gateway = Gateway(
        proxy,
        Role.SESSION,
        "agent-scn1",
        station_info=host.station.station_info,
        tool_context=ToolContext(
            experiments=manager,
            run_catalog={"FieldSweep": FieldSweep},
            status_log_path=tmp_path / "status.jsonl",
        ),
        feed=feed,
    )

    def _await_verdict(request_id: str, timeout: int = 15000) -> ev.Verdict:
        qtbot.waitUntil(
            lambda: recorder.verdict_for(request_id) is not None, timeout=timeout
        )
        return recorder.verdict_for(request_id)

    # 1. Read status.
    qtbot.waitUntil(lambda: gateway.status() is not None, timeout=5000)
    status = gateway.call_tool("read_status")
    assert status["ok"] is True
    assert status["result"]["state"] == "IDLE"

    # 2. Read the manifest.
    manifest = gateway.call_tool("read_manifest")
    assert manifest["ok"] is True
    assert manifest["result"]["setup"] == "sim_cryostat"

    monitoring = gateway.call_tool("start_monitoring")
    assert _await_verdict(monitoring["request_id"]).code is ev.VerdictCode.OK

    # 3. validate_run: may I run this, and how long would it take?
    validated = gateway.call_tool(
        "validate_run",
        {
            "procedure": "FieldSweep",
            "params": dict(FAST_PARAMS),
            "sample_info": dict(SAMPLE_INFO),
            "data_directory": data_dir,
        },
    )
    assert validated["ok"] is True
    assert validated["result"]["ok"] is True
    assert validated["result"]["duration_estimate_s"] > 0

    # 4. probe_run: the same procedure, subsampled.
    probed = gateway.call_tool("probe_run", _probe_run_args(data_dir))
    probe_verdict = _await_verdict(probed["request_id"])
    assert probe_verdict.code is ev.VerdictCode.OK, probe_verdict

    qtbot.waitUntil(
        lambda: bool(manager.current_experiment().runs)
        and manager.current_experiment().runs[-1].status == "done",
        timeout=15000,
    )
    probe_record = manager.current_experiment().runs[-1]
    assert probe_record.kind == "probe"
    assert probe_record.actor.kind is ev.ActorKind.AGENT
    probe_meta = gateway.call_tool(
        "read_run_metadata", {"run_id": probe_record.run_id}
    )
    assert probe_meta["result"]["run_kind"] == "probe"

    # 5. Run the real FieldSweep.
    started = gateway.call_tool("run_procedure", _run_procedure_args(data_dir))
    run_verdict = _await_verdict(started["request_id"])
    assert run_verdict.code is ev.VerdictCode.OK, run_verdict

    # 6. Watch: the StateChange the run caused reaches the GUI side, naming
    # the agent, not the transport.
    qtbot.waitUntil(
        lambda: any(
            isinstance(event, ev.StateChange)
            and event.request_id == started["request_id"]
            and event.actor.kind is ev.ActorKind.AGENT
            and event.actor.id == "agent-scn1"
            for event in recorder.events
        ),
        timeout=10000,
    )

    # 7. Abort — real settle waits are real wall-clock seconds, so the run is
    # stopped rather than waited out (as in tests/test_phase_e_scenario.py).
    aborted = gateway.call_tool("abort_procedure")
    abort_verdict = _await_verdict(aborted["request_id"])
    assert abort_verdict.code is ev.VerdictCode.OK, abort_verdict
    qtbot.waitUntil(
        lambda: manager.current_experiment().runs[-1].status == "aborted",
        timeout=15000,
    )
    run_record = manager.current_experiment().runs[-1]
    assert run_record.kind == "run"
    assert run_record.actor.kind is ev.ActorKind.AGENT

    # Every request this gateway made is in the feed, joined by request_id.
    records = read_feed(feed.path)
    commands = {r["request_id"]: r for r in records if r["record"] == "command"}
    verdicts = {r["request_id"]: r for r in records if r["record"] == "verdict"}
    for request_id in (
        monitoring["request_id"],
        probed["request_id"],
        started["request_id"],
        aborted["request_id"],
    ):
        assert request_id in commands, request_id
        assert request_id in verdicts, request_id
        assert commands[request_id]["actor"]["id"] == "agent-scn1"


# ══════════════════════════════════════════════════════════════════════════
# 2. Role refusals at each step
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def bare_engine(qtbot):
    """A plain Orchestrator over a real sim station, ticked by hand."""
    station = build_station(CONFIG_PATH)
    orch = Orchestrator(station, tick_interval_ms=10)
    yield orch, station
    orch.shutdown()


def _gw(engine, role, actor_id="agent"):
    orch, station = engine
    return Gateway(orch, role, actor_id, station_info=station.station_info)


def test_role_refusals_at_each_step(bare_engine):
    """observer reads and is refused control; debug's recovery needs unattended."""
    orch, _station = bare_engine
    observer = _gw(bare_engine, Role.OBSERVER, "watcher")
    debugger = _gw(bare_engine, Role.DEBUG, "fixer")
    session = _gw(bare_engine, Role.SESSION, "runner")

    # observer reads everything.
    for name in ("read_status", "read_station_info", "read_manifest"):
        answer = observer.call_tool(name)
        assert answer["ok"] is True, (name, answer)

    # observer is refused both probe_run and run_procedure.
    for name, args in (
        ("run_procedure", {"procedure": "FieldSweep"}),
        ("probe_run", {"procedure": "FieldSweep", "probe_spec": {}}),
    ):
        answer = observer.call_tool(name, args)
        assert answer["ok"] is False, (name, answer)
        assert answer["code"] == "BLOCKED_ROLE"
        assert answer["detail"]["rule"] == "role_matrix"

    # debug takes recovery only while unattended: pause / resume / stop_ramp.
    orch.set_attendance(True)
    for name, args in (
        ("pause_procedure", {}),
        ("resume_procedure", {}),
        ("stop_ramp", {"vi_name": "magnet_z"}),
    ):
        answer = debugger.call_tool(name, args)
        assert answer["ok"] is False, (name, answer)
        assert answer["detail"]["rule"] == "attendance"
        assert answer["detail"]["action_class"] == "recovery"

    orch.set_attendance(False)
    for name, args in (
        ("pause_procedure", {}),
        ("resume_procedure", {}),
        ("stop_ramp", {"vi_name": "magnet_z"}),
    ):
        answer = debugger.call_tool(name, args)
        # Permitted by the matrix now; whatever the engine's own state
        # machine says next (an idle station has no run to pause) is not a
        # role refusal.
        assert answer["code"] != "BLOCKED_ROLE", (name, answer)

    # session runs the experiment...
    assert session.permits(ev.CommandName.RUN_PROCEDURE) is None
    assert session.permits(ev.CommandName.PAUSE_PROCEDURE) is None
    # ...but never the envelope, attendance or the kill switch.
    refusals = {
        "set_experiment_envelope": {"envelope": None},
        "set_attendance": {"attended": True},
        "set_agent_gate": {"state": "active"},
    }
    for name, args in refusals.items():
        answer = session.call_tool(name, args)
        assert answer["ok"] is False, (name, answer)
        assert answer["code"] == "BLOCKED_ROLE"
        assert answer["detail"]["rule"] == "role_matrix"
        assert answer["detail"]["action_class"] == "envelope"


# ══════════════════════════════════════════════════════════════════════════
# 3. Kill switch mid-run
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def running_engine(qtbot):
    """A plain Orchestrator with a run catalog, magnet sped up for tests."""
    station = _fast_station()
    orch = Orchestrator(
        station, tick_interval_ms=10, run_catalog={"FieldSweep": FieldSweep}
    )
    yield orch, station
    orch.shutdown()


def test_kill_switch_mid_run(running_engine, tmp_path):
    """read_only leaves the agent nothing but reads; revoked leaves it nothing;
    the operator's identical abort is never gated; emergency standby always is.
    """
    orch, station = running_engine
    started_events: list[ev.RunStarted] = []
    orch.event_emitted.connect(
        lambda e: started_events.append(e) if isinstance(e, ev.RunStarted) else None
    )
    gateway = Gateway(orch, Role.SESSION, "runner", station_info=station.station_info)

    started = gateway.call_tool(
        "run_procedure", _run_procedure_args(str(tmp_path))
    )
    assert started["code"] == "OK", started
    _tick_until(orch, lambda: orch.state != "IDLE")
    (run_started,) = started_events
    data_file = run_started.manifest["data_file"]

    # read_only: the matrix's middle rung. pause_procedure is RECOVERY, not
    # READ, so it is refused even though the gate does not close reads.
    orch.set_agent_gate(ev.AgentGate.READ_ONLY)
    paused = gateway.call_tool("pause_procedure")
    assert paused["ok"] is False
    assert paused["detail"]["rule"] == "kill_switch"
    assert paused["detail"]["gate"] == "read_only"
    still_reads = gateway.call_tool("read_status")
    assert still_reads["ok"] is True

    # revoked: the agent's abort is refused...
    orch.set_agent_gate(ev.AgentGate.REVOKED)
    agent_abort = gateway.call_tool("abort_procedure")
    assert agent_abort["ok"] is False
    assert agent_abort["detail"]["rule"] == "kill_switch"
    assert agent_abort["detail"]["gate"] == "revoked"
    assert orch.state != "IDLE"

    # ...the operator's identical abort, submitted the way the GUI does, is
    # never gated by an agent's kill switch.
    orch.submit(ev.Command(name=ev.CommandName.ABORT_PROCEDURE))
    _tick_until(orch, lambda: orch.state == "IDLE")

    # The run's data file was closed: its end_time attribute is stamped.
    with open_run(Path(data_file)) as handle:
        assert handle.read_metadata()["end_time"]

    # Emergency standby still passes at every kill-switch setting.
    safe = gateway.call_tool("emergency_standby", {"reason": "scenario 3"})
    assert safe["code"] != "BLOCKED_ROLE"
    assert safe["ok"] is True
    assert orch.state == "EMERGENCY"


# ══════════════════════════════════════════════════════════════════════════
# 4. Attendance flips mid-run
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def experiment_engine(qtbot, tmp_path):
    """A plain Orchestrator plus a real ExperimentManager, magnet sped up."""
    station = _fast_station()
    orch = Orchestrator(
        station, tick_interval_ms=10, run_catalog={"FieldSweep": FieldSweep}
    )
    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe", email="jdoe@example.org"))
    store = ExperimentStore(tmp_path / "experiments")
    manager = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=orch,
        config_name="sim_cryostat",
        station=station,
        run_catalog={"FieldSweep": FieldSweep},
    )
    manager.start_experiment("Scenario 4", "jdoe", dict(SAMPLE_INFO))
    yield orch, station, manager
    orch.shutdown()


def test_attendance_flips_mid_run(experiment_engine, tmp_path):
    """A debug agent may pause while unattended and is refused resume once watched."""
    orch, station, manager = experiment_engine
    session = Gateway(orch, Role.SESSION, "runner", station_info=station.station_info)
    debugger = Gateway(orch, Role.DEBUG, "fixer", station_info=station.station_info)

    started = session.call_tool(
        "run_procedure", _run_procedure_args(str(manager.current_data_dir()))
    )
    assert started["code"] == "OK", started
    _tick_until(orch, lambda: orch.state != "IDLE")

    manager.set_attended(False)
    paused = debugger.call_tool("pause_procedure")
    assert paused["code"] == "OK", paused
    _tick_until(orch, lambda: orch.state == "PAUSED")

    manager.set_attended(True)
    resumed = debugger.call_tool("resume_procedure")
    assert resumed["ok"] is False
    assert resumed["detail"]["rule"] == "attendance"
    assert orch.state == "PAUSED", "the refusal must not have moved the run"

    # The experiment record and the engine's own mirror agree.
    assert manager.current_experiment().attended is True
    assert debugger.attended() is True

    # The operator's own resume is never subject to the matrix.
    orch.submit(ev.Command(name=ev.CommandName.RESUME_PROCEDURE))
    _tick_until(orch, lambda: orch.state != "PAUSED")
    orch.submit(ev.Command(name=ev.CommandName.ABORT_PROCEDURE))
    _tick_until(orch, lambda: orch.state == "IDLE")


# ══════════════════════════════════════════════════════════════════════════
# 5. i2as.ctl offline mode
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _isolated_installation_env(tmp_path, monkeypatch):
    """Keep every scenario's machine-local paths inside its own tmp_path.

    Applies to every test in this module (offline ctl and the request spool
    both resolve installation directories from these variables), and is a
    no-op for scenarios that never read them.
    """
    monkeypatch.setenv("I2AS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("I2AS_MEASUREMENT_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr("i2as.ctl.cli.setup_logging", lambda *a, **k: None)


@pytest.fixture
def ctl_client(qtbot, monkeypatch):
    """A session-role offline client with an experiment the physicist opened."""
    from i2as.ctl import client as ctl_client_module

    monkeypatch.setattr(ctl_client_module, "build_station", lambda path: _fast_station(path))
    client = open_client(
        offline=CONFIG_PATH, role=Role.SESSION.value, actor_id="ctl-scenario"
    )
    manager = client.experiments
    manager.roster.add(User(user_id="jdoe", name="J. Doe", email="jdoe@example.org"))
    manager.start_experiment("Scenario 5", "jdoe", dict(SAMPLE_INFO))
    # The gateway is rebuilt lazily against the newly opened experiment (see
    # CtlClient.gateway); force that rebuild and give it one tick so the
    # first read this test makes already has a status snapshot to answer
    # from, exactly as _open_offline() does before the very first command.
    _ = client.gateway
    client.pump(1)
    yield client, manager
    client.close()


def _ctl(capsys, client, argv):
    code = ctl_main(list(argv), client=client)
    return code, json.loads(capsys.readouterr().out)


def test_ctl_offline_the_full_story(capsys, ctl_client):
    """tools, status, validate_run, probe_run, a refused run as observer, a
    successful run as session, and the feed showing every request.
    """
    from i2as.ctl.client import CtlClient

    client, manager = ctl_client
    data_dir = str(manager.current_data_dir())

    code, tools = _ctl(capsys, client, ["tools"])
    assert code == EXIT_OK
    assert {"run_procedure", "probe_run", "read_status"} <= {
        t["name"] for t in tools["tools"]
    }

    code, status = _ctl(capsys, client, ["status"])
    assert (code, status["result"]["state"]) == (EXIT_OK, "IDLE")

    code, monitoring = _ctl(capsys, client, ["call", "start_monitoring"])
    assert (code, monitoring["code"]) == (EXIT_OK, "OK")

    code, validated = _ctl(
        capsys,
        client,
        [
            "call",
            "validate_run",
            "--args",
            json.dumps(
                {
                    "procedure": "FieldSweep",
                    "params": dict(FAST_PARAMS),
                    "sample_info": dict(SAMPLE_INFO),
                    "data_directory": data_dir,
                }
            ),
        ],
    )
    assert code == EXIT_OK
    assert validated["result"]["ok"] is True

    code, probed = _ctl(
        capsys,
        client,
        ["call", "probe_run", "--args", json.dumps(_probe_run_args(data_dir))],
    )
    assert (code, probed["code"]) == (EXIT_OK, "OK")

    watcher = CtlClient(
        mode=client.mode,
        engine=client.engine,
        role=Role.OBSERVER,
        actor_id="ctl-watcher",
        experiments=client.experiments,
    )
    code, refused = _ctl(
        capsys,
        watcher,
        [
            "call",
            "run_procedure",
            "--args",
            json.dumps(_run_procedure_args(data_dir)),
        ],
    )
    assert code == EXIT_REFUSED
    assert refused["code"] == "BLOCKED_ROLE"
    assert refused["detail"]["rule"] == "role_matrix"
    assert refused["detail"]["role"] == "observer"

    code, started = _ctl(
        capsys,
        client,
        ["call", "run_procedure", "--args", json.dumps(_run_procedure_args(data_dir))],
    )
    assert (code, started["code"]) == (EXIT_OK, "OK")
    code, aborted = _ctl(capsys, client, ["abort"])
    assert (code, aborted["code"]) == (EXIT_OK, "OK")

    code, feed = _ctl(capsys, client, ["feed", "--last", "200"])
    assert code == EXIT_OK
    request_ids = {
        r["request_id"] for r in feed["result"]["records"] if r["record"] == "command"
    }
    assert {
        monitoring["request_id"],
        probed["request_id"],
        started["request_id"],
        aborted["request_id"],
    } <= request_ids


def test_ctl_offline_refuses_a_second_client_the_owners_run(capsys, ctl_client):
    """The **run-ownership standard** through the terminal client.

    Two invocations of `i2as.ctl` are two actors, exactly as two agents
    are: the second is refused the first's run by name, reads the override
    off the published tool schema, and is then obeyed — on the record.
    """
    from i2as.ctl.client import CtlClient

    client, manager = ctl_client
    data_dir = str(manager.current_data_dir())

    code, schema = _ctl(capsys, client, ["schema", "abort_procedure"])
    assert code == EXIT_OK
    properties = schema["schema"]["input_schema"]["properties"]
    assert set(properties) == {"override_owner", "reason"}
    assert properties["override_owner"]["type"] == "boolean"
    assert schema["schema"]["input_schema"]["required"] == []

    code, started = _ctl(
        capsys,
        client,
        ["call", "run_procedure", "--args", json.dumps(_run_procedure_args(data_dir))],
    )
    assert (code, started["code"]) == (EXIT_OK, "OK")

    other = CtlClient(
        mode=client.mode,
        engine=client.engine,
        role=Role.SESSION,
        actor_id="ctl-someone-else",
        experiments=client.experiments,
    )
    code, refused = _ctl(capsys, other, ["call", "abort_procedure"])
    assert code == EXIT_REFUSED
    assert refused["code"] == "BLOCKED_ROLE"
    assert refused["detail"]["rule"] == "run_owner"
    assert refused["detail"]["owner"]["id"] == "ctl-scenario"
    assert _ctl(capsys, client, ["status"])[1]["result"]["state"] != "IDLE"

    code, took_over = _ctl(
        capsys,
        other,
        [
            "call",
            "abort_procedure",
            "--args",
            json.dumps(
                {"override_owner": True, "reason": "the physicist asked me to"}
            ),
        ],
    )
    assert (code, took_over["code"]) == (EXIT_OK, "OK")
    assert took_over["detail"]["takeover"] == {
        "owner": {"kind": "agent", "id": "ctl-scenario"},
        "reason": "the physicist asked me to",
    }
    assert _ctl(capsys, client, ["status"])[1]["result"]["state"] == "IDLE"


# ══════════════════════════════════════════════════════════════════════════
# 6. Request spool, live mode
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def spooled_engine(qtbot, tmp_path):
    """A plain Orchestrator draining a Request spool capped at observer."""
    spool = RequestSpool(
        tmp_path / "spool", max_role=Role.OBSERVER.value, authorizer=authorize_spooled
    )
    station = build_station(CONFIG_PATH)
    orch = Orchestrator(station, tick_interval_ms=10, request_spool=spool)
    yield orch, spool
    orch.shutdown()


def _spool_command(name, *, role, kind=ev.ActorKind.AGENT, actor_id="ctl-1", **args):
    return ev.Command(
        name=name,
        actor=ev.Actor(
            kind=kind, id=actor_id, role=role.value if isinstance(role, Role) else role
        ),
        args=args,
    )


def test_request_spool_role_cap_then_raised(spooled_engine):
    """A request above the cap is refused; raising the cap drains it next tick;
    an operator-kind file is refused regardless of the cap.
    """
    orch, spool = spooled_engine

    above_cap = _spool_command(
        ev.CommandName.RUN_PROCEDURE, role=Role.SESSION, procedure="FieldSweep"
    )
    spool.write_request(above_cap, Role.SESSION.value)
    orch._tick()

    (answer,) = [
        r for r in spool.read_verdicts() if r["request_id"] == above_cap.request_id
    ]
    assert answer["code"] == "BLOCKED_ROLE"
    assert answer["detail"]["rule"] == "spool_role_cap"
    assert answer["detail"]["max_role"] == "observer"
    assert list(spool.requests_dir.glob("*.json")) == []

    # Raise the cap: the setup's own property, changed like monitor.yaml's.
    spool.max_role = Role.SESSION.value
    now_allowed = _spool_command(
        ev.CommandName.START_MONITORING, role=Role.SESSION
    )
    spool.write_request(now_allowed, Role.SESSION.value)
    orch._tick()

    drained = [
        r for r in spool.read_verdicts() if r["request_id"] == now_allowed.request_id
    ]
    assert len(drained) == 1, "exactly one verdict line for the drained request"
    assert drained[0]["code"] == "OK"

    # An operator-kind claim is refused whatever the cap says: a file on
    # disk is not a human standing at the cryostat.
    operator_claim = _spool_command(
        ev.CommandName.STOP_MONITORING, role="operator", kind=ev.ActorKind.OPERATOR
    )
    spool.write_request(operator_claim, "operator")
    orch._tick()

    (op_answer,) = [
        r for r in spool.read_verdicts() if r["request_id"] == operator_claim.request_id
    ]
    assert op_answer["code"] == "BLOCKED_ROLE"
    assert op_answer["detail"]["rule"] == "spool_actor_kind"


# ══════════════════════════════════════════════════════════════════════════
# 7. The local socket, then the MCP shim as a subprocess against it
# ══════════════════════════════════════════════════════════════════════════


class _SocketClient:
    """A minimal newline-delimited JSON-RPC client over ``QLocalSocket``."""

    def __init__(self, qtbot, socket_name: str) -> None:
        self._qtbot = qtbot
        self._next_id = 0
        self._buffer = bytearray()
        self.messages: list[dict] = []
        self.socket = QLocalSocket()
        self.socket.readyRead.connect(self._read)
        self.socket.connectToServer(socket_name)
        assert self.socket.waitForConnected(2000), self.socket.errorString()

    def _read(self) -> None:
        self._buffer.extend(bytes(self.socket.readAll()))
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                return
            line = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            if line.strip():
                self.messages.append(json.loads(line))

    def call(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        request_id = self._next_id
        payload = (
            json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
            ).encode("utf-8")
            + b"\n"
        )
        self.socket.write(payload)
        self.socket.flush()
        self._qtbot.waitUntil(
            lambda: any(m.get("id") == request_id for m in self.messages), timeout=5000
        )
        return next(m for m in self.messages if m.get("id") == request_id)

    def result(self, method: str, params: dict | None = None) -> dict:
        response = self.call(method, params)
        assert "error" not in response, response
        return response["result"]

    def hello(self, role, actor_id="agent-1", token=TOKEN) -> dict:
        return self.call(
            "hello",
            {
                "role": role.value if isinstance(role, Role) else role,
                "actor_id": actor_id,
                "token": token,
            },
        )

    @property
    def notifications(self) -> list[dict]:
        return [m for m in self.messages if "id" not in m]


@pytest.fixture
def served(qtbot, tmp_path):
    """A listening GatewayServer over a real Orchestrator and sim station."""
    station = _fast_station()
    orch = Orchestrator(
        station, tick_interval_ms=10, run_catalog={"FieldSweep": FieldSweep}
    )
    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe", email="jdoe@example.org"))
    store = ExperimentStore(tmp_path / "experiments")
    manager = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=orch,
        config_name="sim_cryostat",
        station=station,
        run_catalog={"FieldSweep": FieldSweep},
    )
    manager.start_experiment("Scenario 7", "jdoe", dict(SAMPLE_INFO))
    server = GatewayServer(
        orch,
        socket_name=str(tmp_path / "gateway.sock"),
        descriptor=tmp_path / "gateway.json",
        token=TOKEN,
        max_role=Role.SESSION,
        station_info=station.station_info,
        tool_context=ToolContext(
            experiments=manager,
            run_catalog={"FieldSweep": FieldSweep},
            status_log_path=tmp_path / "status.jsonl",
        ),
    )
    assert server.start()
    yield server, orch, manager, tmp_path
    server.stop()
    orch.shutdown()


def test_local_socket_the_full_story_and_a_ceiling_refusal(qtbot, served):
    """hello, probe_run, a StateChange notification naming the agent, and a
    role above the deployment's ceiling refused at the handshake.
    """
    server, orch, manager, tmp_path = served
    client = _SocketClient(qtbot, server.socket_name)
    hello = client.hello(Role.SESSION, "socket-agent")
    assert "error" not in hello, hello
    assert client.result("events/subscribe")["subscribed"] is True

    orch.start_monitoring()
    probed = client.result(
        "tools/call",
        {"name": "probe_run", "args": _probe_run_args(str(manager.current_data_dir()))},
    )
    assert probed["code"] == "OK", probed

    qtbot.waitUntil(
        lambda: any(m.get("method") == "event" for m in client.notifications),
        timeout=5000,
    )
    agent_events = [
        m
        for m in client.notifications
        if m["method"] == "event"
        and m["params"]["event"].get("actor", {}).get("kind") == "agent"
    ]
    assert agent_events, client.notifications

    # A role above the ceiling is refused before it can connect at all.
    server.max_role = Role.OBSERVER
    ceiling_client = _SocketClient(qtbot, server.socket_name)
    refused = ceiling_client.hello(Role.SESSION, "too-eager")
    assert refused["error"]["code"] == ROLE_REFUSED
    assert "observer" in refused["error"]["message"]


class _AdapterProcess:
    """``python -m i2as.mcp`` running for real, over its own stdio."""

    def __init__(self, descriptor: Path, *, role: str, actor_id: str) -> None:
        self.messages: list[dict] = []
        self._next_id = 0
        self.process = subprocess.Popen(  # noqa: S603 — our own interpreter
            [
                sys.executable,
                "-m",
                "i2as.mcp",
                "--descriptor",
                str(descriptor),
                "--role",
                role,
                "--actor-id",
                actor_id,
                "--log-level",
                "WARNING",
            ],
            cwd=str(REPO_ROOT),
            env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if line.strip():
                self.messages.append(json.loads(line))

    def send(self, message: dict) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
        self.process.stdin.flush()

    def request(self, qtbot, orchestrator, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        request_id = self._next_id
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})

        def step() -> bool:
            orchestrator._tick()
            return any(m.get("id") == request_id for m in self.messages)

        qtbot.waitUntil(step, timeout=20000)
        return next(m for m in self.messages if m.get("id") == request_id)

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover — a wedged child
            self.process.kill()
            self.process.wait(timeout=10)


def test_mcp_shim_subprocess_against_the_running_app(qtbot, served):
    """initialize, tools/list equal to the gateway's own, one tools/call."""
    server, orch, _manager, tmp_path = served
    process = _AdapterProcess(
        tmp_path / "gateway.json", role="session", actor_id="mcp-scenario"
    )
    try:
        handshake = process.request(
            qtbot,
            orch,
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
        )
        assert handshake["result"]["protocolVersion"] == "2025-06-18"

        listed = process.request(qtbot, orch, "tools/list")
        gateway = next(iter(server._connections.values())).gateway
        expected = [mcp_translate.mcp_tool(one) for one in gateway.tool_schemas()]
        assert listed["result"]["tools"] == expected

        called = process.request(
            qtbot, orch, "tools/call", {"name": "read_status", "arguments": {}}
        )
        result = called["result"]
        assert result["isError"] is False
        answer = json.loads(result["content"][0]["text"])
        assert answer["ok"] is True
        assert answer["tool"] == "read_status"
    finally:
        process.close()


# ══════════════════════════════════════════════════════════════════════════
# 9. Two agents at once
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def two_agent_engine(qtbot):
    """A plain Orchestrator with a run catalog, magnet sped up, one shared feed."""
    station = _fast_station()
    orch = Orchestrator(
        station, tick_interval_ms=10, run_catalog={"FieldSweep": FieldSweep}
    )
    yield orch, station
    orch.shutdown()


def test_two_agents_at_once(two_agent_engine, tmp_path):
    """Two Gateways, two actor ids, one run — and the **run-ownership
    standard** between them: agent A's run is agent A's, agent B is refused
    by name and told how to proceed, and the override it then sends is
    recorded as a takeover everywhere the story is told.

    This scenario asserted the opposite until 2026-09-04: either agent could
    end the other's run, and nothing anywhere said which of them had. That
    was the defect the standard was written for.
    """
    orch, station = two_agent_engine
    feed = AgentFeed(tmp_path / "agent_actions.jsonl", "exp-1")
    feed.attach(orch)
    changes: list[ev.StateChange] = []
    finished: list[ev.RunFinished] = []
    orch.event_emitted.connect(
        lambda e: changes.append(e) if isinstance(e, ev.StateChange) else None
    )
    orch.event_emitted.connect(
        lambda e: finished.append(e) if isinstance(e, ev.RunFinished) else None
    )
    agent_a = Gateway(
        orch, Role.SESSION, "agent-A", station_info=station.station_info, feed=feed
    )
    agent_b = Gateway(
        orch, Role.SESSION, "agent-B", station_info=station.station_info, feed=feed
    )

    started = agent_a.call_tool(
        "run_procedure", _run_procedure_args(str(tmp_path))
    )
    assert started["code"] == "OK", started
    _tick_until(orch, lambda: orch.state != "IDLE")

    # Reading is nobody's exclusive right, and the run's owner is published.
    read = agent_b.call_tool("read_status")
    assert read["ok"] is True
    assert read["result"]["run"]["owner"] == {"kind": "agent", "id": "agent-A"}
    assert agent_b.run_owner() == {"kind": "agent", "id": "agent-A"}

    # Ending it is not. Agent B is refused by name, and the run carries on.
    refused = agent_b.call_tool("abort_procedure")
    assert refused["code"] == "BLOCKED_ROLE", refused
    assert refused["detail"]["rule"] == "run_owner"
    assert refused["detail"]["owner"] == {"kind": "agent", "id": "agent-A"}
    assert "override_owner" in refused["reason"]
    assert orch.state != "IDLE"

    # An override with no reason is refused too: a takeover whose record
    # cannot say why is the act the standard exists to prevent.
    unexplained = agent_b.call_tool("abort_procedure", {"override_owner": True})
    assert unexplained["detail"]["rule"] == "override_reason_required"
    assert orch.state != "IDLE"

    # Refuse, then override — deliberately, with a reason, on the record.
    aborted = agent_b.call_tool(
        "abort_procedure",
        {"override_owner": True, "reason": "agent-A stopped answering"},
    )
    assert aborted["code"] == "OK", aborted
    assert aborted["detail"]["takeover"] == {
        "owner": {"kind": "agent", "id": "agent-A"},
        "reason": "agent-A stopped answering",
    }
    _tick_until(orch, lambda: orch.state == "IDLE")

    # The run's own ending names both actors: who ended it, and over whom.
    assert finished[-1].status == "aborted"
    assert finished[-1].actor.id == "agent-B"
    assert finished[-1].overridden_owner == {"kind": "agent", "id": "agent-A"}

    # Every command is in the feed under its own actor id — including the two
    # that were refused, because what an agent TRIED to do is accountability
    # too — and the takeover's verdict record carries the takeover.
    records = read_feed(feed.path)
    commands = {r["request_id"]: r for r in records if r["record"] == "command"}
    verdicts = {r["request_id"]: r for r in records if r["record"] == "verdict"}
    assert commands[started["request_id"]]["actor"]["id"] == "agent-A"
    assert commands[refused["request_id"]]["actor"]["id"] == "agent-B"
    assert verdicts[refused["request_id"]]["detail"]["rule"] == "run_owner"
    assert commands[aborted["request_id"]]["args"] == {
        "override_owner": True,
        "reason": "agent-A stopped answering",
    }
    assert verdicts[aborted["request_id"]]["detail"]["takeover"]["owner"] == {
        "kind": "agent",
        "id": "agent-A",
    }

    # The panel-facing StateChange the abort caused names the second agent.
    abort_changes = [
        change for change in changes if change.request_id == aborted["request_id"]
    ]
    assert abort_changes, changes
    assert abort_changes[0].actor.id == "agent-B"
    assert abort_changes[0].actor.role == "session"
