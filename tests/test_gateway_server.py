# ---
# description: |
#   Tests for the Gateway server (session/gateway/local_server.py): the
#   newline-delimited JSON-RPC transport that carries one Gateway per
#   connection over a QLocalServer on the GUI thread's event loop. Covers the
#   token/role handshake, the rendered tool list, the Phase E scenario driven
#   entirely over the socket against a sim station, the four structured
#   refusals, the agent feed, malformed framing, and the agent-actor state
#   change the GUI sees.
# last_updated: 2026-09-03
# ---

from __future__ import annotations

import json

import pytest
from PyQt6.QtNetwork import QLocalSocket

from i2as.core import events as ev
from i2as.core.orchestrator import Orchestrator
from i2as.core.station import build_station
from i2as.procedures.field_sweep import FieldSweep
from i2as.session.agent_feed import AgentFeed
from i2as.session.gateway import (
    Gateway,
    GatewayServer,
    Role,
    ToolContext,
)
from i2as.session.gateway.local_server import (
    BAD_TOKEN,
    METHOD_NOT_FOUND,
    NOT_AUTHENTICATED,
    PARSE_ERROR,
    ROLE_REFUSED,
    SCHEMA_VERSION,
    SOCKET_FILENAME,
    default_socket_name,
)
from i2as.session.manager import ExperimentManager
from i2as.session.models import User
from i2as.session.store import ExperimentStore, UserRoster

CONFIG_PATH = "i2as/configs/sim_cryostat"

TOKEN = "test-token-not-a-secret"

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


class SocketClient:
    """A newline-delimited JSON-RPC client over ``QLocalSocket``.

    Deliberately hand-written rather than reusing any production helper: the
    point of these tests is the wire, so nothing but ``json`` and a socket
    stands between the assertions and the bytes.
    """

    def __init__(self, qtbot, socket_name: str) -> None:
        """Connect to a listening server.

        Args:
            qtbot: The pytest-qt fixture, used to spin the real event loop.
            socket_name: The name the server is listening on.
        """
        self._qtbot = qtbot
        self._next_id = 0
        self._buffer = bytearray()
        self.messages: list[dict] = []
        self.socket = QLocalSocket()
        self.socket.readyRead.connect(self._read)
        self.socket.connectToServer(socket_name)
        assert self.socket.waitForConnected(2000), self.socket.errorString()

    def _read(self) -> None:
        """Accumulate bytes and decode every whole line."""
        self._buffer.extend(bytes(self.socket.readAll()))
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                return
            line = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            if line.strip():
                self.messages.append(json.loads(line))

    def write(self, raw: bytes) -> None:
        """Write raw bytes, exactly as given.

        Args:
            raw: The bytes to put on the wire.
        """
        self.socket.write(raw)
        self.socket.flush()

    def call(self, method: str, params: dict | None = None) -> dict:
        """Send one request and return the response that answers it.

        Args:
            method: The JSON-RPC method.
            params: Its parameters.

        Returns:
            The response object.
        """
        self._next_id += 1
        request_id = self._next_id
        self.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            ).encode("utf-8")
            + b"\n"
        )
        self._qtbot.waitUntil(
            lambda: any(m.get("id") == request_id for m in self.messages),
            timeout=5000,
        )
        return next(m for m in self.messages if m.get("id") == request_id)

    def result(self, method: str, params: dict | None = None) -> dict:
        """Call and assert success, returning the ``result``.

        Args:
            method: The JSON-RPC method.
            params: Its parameters.

        Returns:
            The response's ``result`` object.
        """
        response = self.call(method, params)
        assert "error" not in response, response
        return response["result"]

    def hello(self, role: Role | str, actor_id: str = "agent-1", token: str = TOKEN):
        """Perform the handshake.

        Args:
            role: The role to claim.
            actor_id: The identity to claim.
            token: The secret to present.

        Returns:
            The response object.
        """
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
        """Every server-initiated message received so far."""
        return [m for m in self.messages if "id" not in m]


def _tick_until(orchestrator, predicate, max_ticks: int = 4000) -> None:
    """Tick the engine until *predicate* holds; assert that it eventually does."""
    for _ in range(max_ticks):
        orchestrator._tick()
        if predicate():
            return
    raise AssertionError("the engine never reached the expected state")


@pytest.fixture
def served(qtbot, tmp_path):
    """A listening ``GatewayServer`` over a real Orchestrator and sim station."""
    station = build_station(CONFIG_PATH)
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []
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
    manager.start_experiment("Transport", "jdoe", dict(SAMPLE_INFO))
    feed = AgentFeed(tmp_path / "agent_actions.jsonl", "exp-1")
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
        feed=feed,
    )
    assert server.start()
    yield server, orch, manager, feed, tmp_path
    server.stop()
    orch.shutdown()


def _client(qtbot, served, role=Role.SESSION, actor_id="agent-1"):
    """Connect and hand back a client that has already said hello."""
    server = served[0]
    client = SocketClient(qtbot, server.socket_name)
    response = client.hello(role, actor_id)
    assert "error" not in response, response
    return client


# ══════════════════════════════════════════════════════════════════════════
# The descriptor and the handshake
# ══════════════════════════════════════════════════════════════════════════


def test_the_descriptor_names_the_socket_the_pid_and_the_token(served):
    """A client finds the running app by reading one file it owns alone."""
    server, _orch, _manager, _feed, tmp_path = served

    descriptor = json.loads((tmp_path / "gateway.json").read_text())

    assert descriptor["schema"] == SCHEMA_VERSION
    assert descriptor["socket"] == server.socket_name
    assert descriptor["token"] == TOKEN
    assert descriptor["max_role"] == "session"
    assert descriptor["pid"] > 0
    assert (tmp_path / "gateway.json").stat().st_mode & 0o077 == 0


def test_a_bad_token_is_refused_at_the_handshake(qtbot, served):
    """The front door: no token, no gateway, no tools."""
    server = served[0]
    client = SocketClient(qtbot, server.socket_name)

    refusal = client.hello(Role.OBSERVER, token="wrong")

    assert refusal["error"]["code"] == BAD_TOKEN
    assert client.call("tools/list")["error"]["code"] == NOT_AUTHENTICATED


def test_a_role_above_the_ceiling_is_refused_at_the_handshake(qtbot, tmp_path):
    """A deployment that hands out `observer` hands out nothing more."""
    station = build_station(CONFIG_PATH)
    orch = Orchestrator(station, tick_interval_ms=10)
    server = GatewayServer(
        orch,
        socket_name=str(tmp_path / "ceiling.sock"),
        descriptor=tmp_path / "ceiling.json",
        token=TOKEN,
        max_role=Role.OBSERVER,
        station_info=station.station_info,
    )
    assert server.start()
    try:
        client = SocketClient(qtbot, server.socket_name)

        refused = client.hello(Role.SESSION)
        permitted = client.hello(Role.OBSERVER)

        assert refused["error"]["code"] == ROLE_REFUSED
        assert "observer" in refused["error"]["message"]
        assert permitted["result"]["role"] == "observer"
    finally:
        server.stop()
        orch.shutdown()


def test_an_unknown_role_is_refused_by_name(qtbot, served):
    """No silent default at the handshake either."""
    server = served[0]
    client = SocketClient(qtbot, server.socket_name)

    refusal = client.hello("superuser")

    assert refusal["error"]["code"] == ROLE_REFUSED
    assert "superuser" in refusal["error"]["message"]


# ══════════════════════════════════════════════════════════════════════════
# Framing
# ══════════════════════════════════════════════════════════════════════════


def test_malformed_frames_are_answered_with_json_rpc_errors(qtbot, served):
    """Nothing a client can write is allowed to raise into the event loop."""
    server, orch, *_ = served
    client = _client(qtbot, served)

    client.write(b"{not json at all}\n")
    qtbot.waitUntil(
        lambda: any(m.get("error", {}).get("code") == PARSE_ERROR for m in client.messages),
        timeout=5000,
    )

    assert client.call("no/such/method")["error"]["code"] == METHOD_NOT_FOUND
    # A JSON value that is not a request object, and a request with no method.
    client.write(b"[1, 2, 3]\n")
    assert client.call("status")["result"]["state"] in {"", "IDLE"}
    # The engine is untouched by all of it.
    orch._tick()
    assert orch.state == "IDLE"


def test_a_frame_split_across_reads_is_reassembled(qtbot, served):
    """Partial reads are the normal case on a stream, not an error case."""
    client = _client(qtbot, served)
    frame = json.dumps(
        {"jsonrpc": "2.0", "id": 99, "method": "station", "params": {}}
    ).encode("utf-8")

    client.write(frame[:10])
    client.write(frame[10:])
    client.write(b"\n")
    qtbot.waitUntil(
        lambda: any(m.get("id") == 99 for m in client.messages), timeout=5000
    )

    answer = next(m for m in client.messages if m.get("id") == 99)
    names = {i["name"] for i in answer["result"]["station"]["instruments"]}
    assert "magnet_z" in names


def test_an_oversized_frame_is_refused_and_the_connection_dropped(qtbot, served):
    """An unbounded buffer on the GUI thread would be a hazard, not a bug."""
    server = served[0]
    client = SocketClient(qtbot, server.socket_name)
    client.hello(Role.OBSERVER)

    client.write(b"x" * (1 << 21))
    qtbot.waitUntil(
        lambda: any("error" in m and m.get("id") is None for m in client.messages),
        timeout=5000,
    )

    assert any(m.get("error", {}).get("code") == -31005 for m in client.messages)


# ══════════════════════════════════════════════════════════════════════════
# The surface is the gateway's, unchanged
# ══════════════════════════════════════════════════════════════════════════


def test_tools_list_is_exactly_the_gateways_tool_schemas(qtbot, served):
    """The transport renders nothing of its own — it forwards the surface."""
    _server, orch, _manager, _feed, _tmp = served
    client = _client(qtbot, served)
    reference = Gateway(
        orch, Role.SESSION, "reference", station_info=orch.station_info
    )

    tools = client.result("tools/list")["tools"]

    assert tools == reference.tool_schemas()
    assert tools


def test_status_and_station_answer_from_the_gateways_own_mirror(qtbot, served):
    """Two reads that never call into the engine, over the wire."""
    _server, orch, *_ = served
    client = _client(qtbot, served)
    orch._tick()

    status = client.result("status")
    station = client.result("station")

    assert status["state"] == "IDLE"
    assert status["attended"] is True
    assert status["agent_gate"] == "active"
    assert status["status"]["state"] == "IDLE"
    assert {i["name"] for i in station["station"]["instruments"]} >= {"magnet_z"}


# ══════════════════════════════════════════════════════════════════════════
# The Phase E scenario, driven entirely over the socket
# ══════════════════════════════════════════════════════════════════════════


def test_the_scenario_runs_over_the_socket_against_a_sim_station(qtbot, served):
    """Validate, probe, run and abort a FieldSweep without an in-process client."""
    _server, orch, manager, _feed, tmp_path = served
    client = _client(qtbot, served)
    orch.start_monitoring()

    validated = client.result(
        "tools/call",
        {
            "name": "validate_run",
            "args": {
                "procedure": "FieldSweep",
                "params": {**FULL_PARAMS, "field_steps": 5},
                "sample_info": dict(SAMPLE_INFO),
                "data_directory": str(manager.current_data_dir()),
            },
        },
    )
    assert validated["ok"] is True
    assert validated["result"]["duration_estimate_s"] > 0

    probed = client.result(
        "tools/call",
        {
            "name": "probe_run",
            "args": {
                "procedure": "FieldSweep",
                "params": {**FULL_PARAMS, "field_steps": 51},
                "sample_info": dict(SAMPLE_INFO),
                "data_directory": str(manager.current_data_dir()),
                "file_prefix": "probe",
                "probe_spec": {"n_points": 3, "averaging": 2, "max_wait_s": 0.0},
            },
        },
    )
    assert probed["code"] == "OK", probed

    _tick_until(orch, lambda: manager.current_experiment().runs[-1].status == "done")
    listed = client.result("tools/call", {"name": "list_runs", "args": {}})
    assert listed["result"]["runs"][-1]["kind"] == "probe"

    started = client.result(
        "tools/call",
        {
            "name": "run_procedure",
            "args": {
                "procedure": "FieldSweep",
                "params": {**FULL_PARAMS, "field_steps": 51},
                "sample_info": dict(SAMPLE_INFO),
                "data_directory": str(manager.current_data_dir()),
                "file_prefix": "sweep",
            },
        },
    )
    assert started["code"] == "OK", started
    for _ in range(20):
        orch._tick()

    aborted = client.result("tools/call", {"name": "abort_procedure", "args": {}})
    assert aborted["code"] == "OK", aborted
    _tick_until(orch, lambda: orch.state == "IDLE")


def test_the_four_refusals_arrive_with_structured_reasons(qtbot, served):
    """Envelope, role, attendance and kill switch, each named by its rule."""
    _server, orch, *_ = served
    session = _client(qtbot, served, Role.SESSION, "runner")
    observer = _client(qtbot, served, Role.OBSERVER, "watcher")
    debugger = _client(qtbot, served, Role.DEBUG, "fixer")

    envelope = session.result(
        "tools/call",
        {"name": "set_experiment_envelope", "args": {"envelope": None}},
    )
    assert envelope["code"] == "BLOCKED_ROLE"
    assert envelope["detail"]["rule"] == "role_matrix"
    assert envelope["detail"]["action_class"] == "envelope"

    role = observer.result(
        "tools/call", {"name": "run_procedure", "args": {"procedure": "FieldSweep"}}
    )
    assert role["code"] == "BLOCKED_ROLE"
    assert role["detail"]["rule"] == "role_matrix"
    assert role["detail"]["role"] == "observer"

    orch.set_attendance(True)
    attendance = debugger.result(
        "tools/call", {"name": "pause_procedure", "args": {}}
    )
    assert attendance["code"] == "BLOCKED_ROLE"
    assert attendance["detail"]["rule"] == "attendance"

    orch.set_agent_gate(ev.AgentGate.REVOKED)
    killed = session.result("tools/call", {"name": "read_status", "args": {}})
    assert killed["code"] == "BLOCKED_ROLE"
    assert killed["detail"]["rule"] == "kill_switch"
    assert killed["detail"]["gate"] == "revoked"


def test_every_request_over_the_socket_lands_in_the_agent_feed(qtbot, served):
    """The trail does not care which transport the command arrived on."""
    _server, _orch, _manager, feed, _tmp = served
    client = _client(qtbot, served, Role.OBSERVER, "watcher")

    refused = client.result(
        "tools/call", {"name": "run_procedure", "args": {"procedure": "FieldSweep"}}
    )
    permitted = client.result(
        "tools/call",
        {"name": "emergency_standby", "args": {"reason": "socket test"}},
    )

    records = [
        json.loads(line)
        for line in feed.path.read_text().splitlines()
        if line.strip()
    ]
    commands = {
        record["request_id"]: record
        for record in records
        if record["record"] == "command"
    }
    assert refused["request_id"] in commands
    assert permitted["request_id"] in commands
    assert commands[refused["request_id"]]["actor"] == {
        "kind": "agent",
        "id": "watcher",
        "role": "observer",
    }
    assert commands[refused["request_id"]]["args"]["procedure"] == "FieldSweep"


# ══════════════════════════════════════════════════════════════════════════
# What the GUI sees, and what a subscriber sees
# ══════════════════════════════════════════════════════════════════════════


def test_a_command_over_the_socket_shows_as_an_agent_on_the_gui_side(qtbot, served):
    """The exit criterion: the window names the agent, not the transport."""
    _server, orch, manager, _feed, _tmp = served
    changes: list[ev.StateChange] = []
    orch.event_emitted.connect(
        lambda event: changes.append(event)
        if isinstance(event, ev.StateChange)
        else None
    )
    client = _client(qtbot, served, Role.SESSION, "runner")
    orch.start_monitoring()

    answer = client.result(
        "tools/call",
        {
            "name": "run_procedure",
            "args": {
                "procedure": "FieldSweep",
                "params": {**FULL_PARAMS, "field_steps": 5},
                "sample_info": dict(SAMPLE_INFO),
                "data_directory": str(manager.current_data_dir()),
                "file_prefix": "gui",
            },
        },
    )
    assert answer["code"] == "OK", answer

    agent_changes = [
        change
        for change in changes
        if change.actor.kind is ev.ActorKind.AGENT
        and change.request_id == answer["request_id"]
    ]
    assert agent_changes, changes
    assert agent_changes[0].actor.id == "runner"
    assert agent_changes[0].actor.role == "session"
    client.result("tools/call", {"name": "abort_procedure", "args": {}})
    _tick_until(orch, lambda: orch.state == "IDLE")


def test_a_subscriber_receives_state_changes_and_verdicts(qtbot, served):
    """`events/subscribe` turns the connection into a listener too."""
    _server, orch, *_ = served
    client = _client(qtbot, served)

    assert client.result("events/subscribe")["subscribed"] is True
    client.result("tools/call", {"name": "pause_procedure", "args": {}})
    orch._tick()
    qtbot.waitUntil(
        lambda: any(m.get("method") == "event" for m in client.notifications),
        timeout=5000,
    )

    verdicts = [m for m in client.notifications if m["method"] == "verdict"]
    events = [m for m in client.notifications if m["method"] == "event"]
    assert verdicts
    assert verdicts[0]["params"]["verdict"]["command"] == "pause_procedure"
    assert {e["params"]["event"]["kind"] for e in events} <= {
        "state_change",
        "status_snapshot",
    }
    rebuilt = ev.event_from_json(events[0]["params"]["event"])
    assert isinstance(rebuilt, (ev.StateChange, ev.StatusSnapshot))


def test_a_connection_that_never_subscribed_receives_nothing(qtbot, served):
    """Notifications are opt-in: a polling client is not made to read them."""
    _server, orch, *_ = served
    client = _client(qtbot, served)

    client.result("tools/call", {"name": "pause_procedure", "args": {}})
    orch._tick()
    client.result("status")

    assert client.notifications == []


def test_stopping_removes_the_descriptor_and_the_socket(qtbot, served):
    """A stale descriptor would point a client at a process that is gone."""
    server, _orch, _manager, _feed, tmp_path = served
    _client(qtbot, served)

    server.stop()

    assert not (tmp_path / "gateway.json").exists()
    assert not server.isListening()


def test_the_app_stops_the_server_when_it_quits():
    """``main()`` wires ``stop()`` to the application's own shutdown.

    Read off the source rather than by running ``main()``, which needs a
    real application, a real station and a real window. What is being
    asserted is one wiring decision: a descriptor that outlives its process
    names a socket that is gone and a token that means nothing, so an
    adapter reading it reports "cannot connect" where it should report "the
    app is not running".
    """
    import ast
    from pathlib import Path

    import i2as.main

    tree = ast.parse(Path(i2as.main.__file__).read_text(encoding="utf-8"))
    wired = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "aboutToQuit"
        and any(
            isinstance(argument, ast.Attribute) and argument.attr == "stop"
            for argument in node.args
        )
    ]

    assert wired, "main() must stop the Gateway server when the app quits"


# ══════════════════════════════════════════════════════════════════════════
# The application's own wiring, across the instrument thread
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def app_wired_server(qtbot, tmp_path):
    """The server ``main()`` builds: over the proxy, engine on its own thread.

    Exercises ``i2as.main._build_gateway_server()`` itself rather than a
    hand-assembled equivalent, because the wiring IS what is under test: the
    object handed to the server is the **Orchestrator proxy**, never the
    engine, and under the single hardware thread standard the engine is on
    the instrument thread while the server is on this one.
    """
    from i2as.core.instrument_host import InstrumentHost
    from i2as.core.config import read_gateway_config
    from i2as.main import _build_gateway_server

    host = InstrumentHost(
        lambda: build_station(CONFIG_PATH),
        mode="threaded",
        orchestrator_options={
            "tick_interval_ms": 50,
            "run_catalog": {"FieldSweep": FieldSweep},
        },
    )
    host.start()
    proxy = host.build_proxy()

    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe", email="jdoe@example.org"))
    manager = ExperimentManager(
        store=ExperimentStore(tmp_path / "experiments"),
        roster=roster,
        orchestrator=proxy,
        config_name="sim_cryostat",
        station=host.station,
        run_catalog={"FieldSweep": FieldSweep},
    )
    manager.start_experiment("Transport", "jdoe", dict(SAMPLE_INFO))

    # The setup's own answer, read the way main() reads it.
    (tmp_path / "monitor.yaml").write_text(
        "monitor:\n"
        "  tick_interval_ms: 1000\n"
        "  gateway_server: true\n"
        "  gateway_max_role: session\n"
    )
    server = _build_gateway_server(
        proxy,
        read_gateway_config(str(tmp_path)),
        station_info=host.station.station_info,
        tool_context=ToolContext(
            experiments=manager, run_catalog={"FieldSweep": FieldSweep}
        ),
        feed=lambda: None,
        socket_name=str(tmp_path / "gateway.sock"),
        descriptor=tmp_path / "gateway.json",
        token=TOKEN,
    )
    yield server, host, proxy
    server.stop()
    host.shutdown()


def test_the_app_wiring_serves_a_client_with_the_engine_on_its_own_thread(
    qtbot, app_wired_server
):
    """The wiring main() uses builds a listening server and answers a client.

    The regression this pins: the server was built over an object whose two
    contract streams are named ``verdict``/``event``, and connecting to
    ``verdict_emitted``/``event_emitted`` raised at construction — a failure
    no test saw, because nothing built the server through ``main()``. One
    ``tools/list`` over the real socket is the proof that the whole path
    holds: config gate, proxy, handshake, rendered surface.
    """
    server, host, proxy = app_wired_server

    assert server is not None
    assert server.isListening()
    # The engine really is elsewhere; the server really is here.
    assert host.thread_object is not None
    assert host.orchestrator.thread() is host.thread_object
    assert proxy.bridge is not None

    client = SocketClient(qtbot, server.socket_name)
    assert "error" not in client.hello(Role.SESSION, "agent-1")

    tools = client.result("tools/list")["tools"]

    assert tools, "the wired server rendered no tools"
    assert {"start_monitoring", "pause_procedure"} <= {
        tool["name"] for tool in tools
    }


def test_a_setup_that_does_not_ask_for_a_server_gets_none(tmp_path):
    """The config gate, in the wiring function: silence opens no socket."""
    from i2as.core.config import read_gateway_config
    from i2as.main import _build_gateway_server

    (tmp_path / "monitor.yaml").write_text("monitor:\n  tick_interval_ms: 1000\n")

    assert (
        _build_gateway_server(
            object(),
            read_gateway_config(str(tmp_path)),
            station_info=lambda: None,
            tool_context=ToolContext(),
            feed=lambda: None,
        )
        is None
    )


def test_default_socket_name_is_a_file_on_posix_and_a_named_pipe_on_windows(
    monkeypatch, tmp_path
):
    """The socket lives in the log directory, or is a per-process pipe name."""
    import os

    from i2as.session.gateway import local_server

    monkeypatch.setenv("I2AS_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(local_server.os, "name", "posix")
    assert default_socket_name() == str(tmp_path / SOCKET_FILENAME)
    monkeypatch.setattr(local_server.os, "name", "nt")
    assert default_socket_name() == f"i2as-gateway-{os.getpid()}"
