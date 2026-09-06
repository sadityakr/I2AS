# ---
# description: |
#   Tests for the MCP adapter (i2as/mcp/): the pure translation from the
#   Gateway server's wire into MCP payloads, the adapter's dispatch against a
#   small in-repo JSON-RPC fake, the import allowlist that keeps the adapter
#   process unable to reach an instrument, and the stdio framing itself —
#   driven by running `python -m i2as.mcp` as a real subprocess against a
#   real GatewayServer over a real socket.
# last_updated: 2026-09-03
# ---

from __future__ import annotations

import ast
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from i2as.core.orchestrator import Orchestrator
from i2as.core.station import build_station
from i2as.mcp import translate
from i2as.mcp.adapter import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    RESOURCE_NOT_FOUND,
    McpAdapter,
)
from i2as.mcp.client import GatewayError, default_descriptor_path, read_descriptor
from i2as.mcp.sdk import sdk_unavailable_reason, serve_with_sdk
from i2as.procedures.field_sweep import FieldSweep
from i2as.session.gateway import GatewayServer, Role, ToolContext
from i2as.session.manager import ExperimentManager
from i2as.session.models import User
from i2as.session.store import ExperimentStore, UserRoster

CONFIG_PATH = "i2as/configs/sim_cryostat"


# ── the descriptor's default location tracks core.paths ───────────────────────


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"XDG_STATE_HOME": "xdg-state"},
        {"I2AS_LOG_DIR": "explicit-logs"},
    ],
    ids=["home-default", "xdg-state-home", "i2as-log-dir"],
)
def test_default_descriptor_path_matches_the_log_directory_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    """The adapter's copy of the per-user state rule stays in step with core.paths.

    Contract C21 keeps ``i2as.mcp.client`` from importing ``paths.py``,
    so it carries a copy of ``user_state_dir()``; this pins the copy to the
    original for every branch a POSIX host can take (the Windows branch is
    the same three lines in both, read side by side).
    """
    from i2as.core import paths

    for name in ("I2AS_GATEWAY_DESCRIPTOR", "I2AS_LOG_DIR", "XDG_STATE_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, str(tmp_path / value))
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    assert default_descriptor_path() == paths.log_directory() / "gateway.json"

TOKEN = "test-token-not-a-secret"

SAMPLE_INFO = {"sample_name": "S", "sample_id": "S-1", "comments": ""}

REPO_ROOT = Path(__file__).resolve().parents[1]


# ══════════════════════════════════════════════════════════════════════════
# A JSON-RPC fake standing in for the Gateway server
# ══════════════════════════════════════════════════════════════════════════


class FakeGatewayClient:
    """A ``GatewayClient`` that answers from a table instead of a socket.

    Small on purpose: the adapter's job is to forward, so what these tests
    need to see is exactly which wire request each MCP method produced.

    Attributes:
        calls: Every ``(method, params)`` the adapter sent, in order.
        answers: What to answer each wire method with.
        queued: Notifications ``drain_notifications()`` will pick up.
    """

    def __init__(self, answers: dict | None = None) -> None:
        """Build the fake.

        Args:
            answers: Wire method to result object; a method with no entry is
                answered with an empty result.
        """
        self.calls: list[tuple[str, dict]] = []
        self.answers = dict(answers or {})
        self.queued: list[dict] = []
        self.closed = False
        self.raise_on: str | None = None

    def connect(self) -> dict:
        """Pretend to connect."""
        return {"role": "session", "actor_id": "fake", "schema": 1}

    def close(self) -> None:
        """Record the close."""
        self.closed = True

    def fileno(self) -> int | None:
        """No selectable descriptor."""
        return None

    def call(self, method: str, params: dict | None = None) -> dict:
        """Record and answer one wire request.

        Args:
            method: The wire method.
            params: Its parameters.

        Returns:
            The configured answer.

        Raises:
            GatewayError: When ``raise_on`` names this method.
        """
        self.calls.append((method, dict(params or {})))
        if self.raise_on == method:
            raise GatewayError("the app went away", code=-31099)
        return dict(self.answers.get(method, {}))

    def receive_available(self) -> None:
        """Nothing arrives out of band in the fake."""

    def take_notifications(self) -> list[dict]:
        """Hand over and forget the queued notifications."""
        queued, self.queued = self.queued, []
        return queued


def _adapter(answers: dict | None = None) -> tuple[McpAdapter, FakeGatewayClient]:
    """Build an adapter over a fake client."""
    client = FakeGatewayClient(answers)
    return McpAdapter(client), client


def _request(method: str, params: dict | None = None, request_id: int = 1) -> dict:
    """Build one JSON-RPC request object."""
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}


# ══════════════════════════════════════════════════════════════════════════
# The translation is a rename, not a rewrite
# ══════════════════════════════════════════════════════════════════════════


def test_a_tool_is_re_keyed_and_nothing_else():
    """``input_schema`` becomes ``inputSchema``; no word is invented."""
    schema = {
        "name": "start_procedure",
        "description": "Start one procedure.",
        "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
    }

    rendered = translate.mcp_tool(schema)

    assert rendered == {
        "name": "start_procedure",
        "description": "Start one procedure.",
        "inputSchema": schema["input_schema"],
    }


def test_the_tool_list_is_the_gateways_own_list_in_order():
    """Every tool the gateway published appears once, in the same order."""
    schemas = [
        {"name": "a", "description": "A", "input_schema": {"type": "object"}},
        {"name": "b", "description": "B", "input_schema": {"type": "object"}},
    ]

    result = translate.tools_list_result(schemas)

    assert [tool["name"] for tool in result["tools"]] == ["a", "b"]
    assert result["resultType"] == "complete"
    assert result["ttlMs"] == 0
    assert result["cacheScope"] == "private"


def test_a_refusal_is_a_result_that_says_so_not_a_protocol_error():
    """The gateway's whole answer survives; only ``isError`` is added."""
    refusal = {
        "tool": "start_procedure",
        "ok": False,
        "code": "BLOCKED_ROLE",
        "verdict": {"detail": {"rule": "role_matrix"}},
    }

    result = translate.tool_result(refusal)

    assert result["isError"] is True
    assert json.loads(result["content"][0]["text"]) == refusal


def test_a_successful_call_is_not_an_error():
    """``ok`` true means ``isError`` false."""
    assert translate.tool_result({"ok": True, "result": 3})["isError"] is False


def test_the_handshake_answers_a_revision_the_client_named():
    """A known revision is echoed; an unknown one gets the handshake era's."""
    assert translate.negotiate_protocol_version("2025-06-18") == "2025-06-18"
    assert (
        translate.negotiate_protocol_version("1999-01-01")
        == translate.HANDSHAKE_PROTOCOL_VERSION
    )
    assert translate.HANDSHAKE_PROTOCOL_VERSION in translate.PROTOCOL_VERSIONS


def test_discovery_advertises_every_revision_this_adapter_speaks():
    """``server/discover`` lists the versions rather than negotiating one."""
    result = translate.discover_result()

    assert result["supportedVersions"] == list(translate.PROTOCOL_VERSIONS)
    assert result["supportedVersions"][0] == translate.LATEST_PROTOCOL_VERSION
    assert set(result["capabilities"]) == {"tools", "resources", "logging"}


def test_a_gateway_notification_travels_as_a_log_message():
    """A state change reaches the session whole, in ``data``."""
    event = {"kind": "StateChange", "from_state": "IDLE", "to_state": "RAMPING"}

    translated = translate.log_notification({"method": "event", "params": {"event": event}})

    assert translated is not None
    assert translated["method"] == "notifications/message"
    assert translated["params"]["data"]["event"] == event


def test_a_refused_verdict_is_logged_as_a_warning():
    """A verdict that is not OK is not an ``info``."""
    translated = translate.log_notification(
        {"method": "verdict", "params": {"verdict": {"code": "BLOCKED_ROLE"}}}
    )

    assert translated is not None
    assert translated["params"]["level"] == "warning"


def test_anything_that_is_not_an_event_or_a_verdict_is_dropped():
    """The adapter invents no MCP notification of its own."""
    assert translate.log_notification({"method": "something_else"}) is None
    assert translate.log_notifications([{"method": "x"}, {"method": "y"}]) == []


# ══════════════════════════════════════════════════════════════════════════
# The adapter forwards, and answers everything
# ══════════════════════════════════════════════════════════════════════════


def test_opening_subscribes_so_events_are_a_property_of_the_session():
    """``open()`` says hello and asks for the event stream, once."""
    adapter, client = _adapter()

    adapter.open()

    assert [method for method, _ in client.calls] == ["events/subscribe"]


def test_tools_list_is_the_gateways_tools_list():
    """One MCP request produces exactly one wire request."""
    adapter, client = _adapter(
        {"tools/list": {"tools": [{"name": "a", "description": "A", "input_schema": {}}]}}
    )

    response = adapter.handle(_request("tools/list"))

    assert client.calls == [("tools/list", {})]
    assert [tool["name"] for tool in response["result"]["tools"]] == ["a"]


def test_tools_call_forwards_mcps_arguments_as_the_wires_args():
    """The one key MCP and the gateway spell differently."""
    adapter, client = _adapter({"tools/call": {"ok": True, "result": None}})

    adapter.handle(
        _request("tools/call", {"name": "set_field", "arguments": {"field_t": 1.0}})
    )

    assert client.calls == [
        ("tools/call", {"name": "set_field", "args": {"field_t": 1.0}})
    ]


def test_tools_call_without_a_name_is_an_invalid_params_error():
    """Unreadable parameters are the envelope's problem, not the model's."""
    adapter, client = _adapter()

    response = adapter.handle(_request("tools/call", {"arguments": {}}))

    assert response["error"]["code"] == INVALID_PARAMS
    assert client.calls == []


def test_each_resource_is_read_through_the_tool_that_already_serves_it():
    """A resource is a URI over a read tool, never a fourth way to read."""
    adapter, client = _adapter({"tools/call": {"ok": True, "result": {"state": "IDLE"}}})

    for uri, tool in translate.RESOURCE_TOOLS.items():
        client.calls.clear()
        response = adapter.handle(_request("resources/read", {"uri": uri}))
        assert client.calls == [("tools/call", {"name": tool, "args": {}})]
        contents = response["result"]["contents"][0]
        assert contents["uri"] == uri
        assert json.loads(contents["text"]) == {"state": "IDLE"}


def test_a_refused_resource_read_still_carries_its_reason():
    """The reason a read was refused is what the session needs to see."""
    adapter, _ = _adapter(
        {"tools/call": {"ok": False, "code": "FAILED", "reason": "no station"}}
    )

    response = adapter.handle(_request("resources/read", {"uri": translate.STATUS_URI}))

    assert json.loads(response["result"]["contents"][0]["text"])["reason"] == "no station"


def test_an_unknown_uri_is_a_resource_not_found():
    """A URI this server does not serve names the ones it does."""
    adapter, _ = _adapter()

    response = adapter.handle(_request("resources/read", {"uri": "i2as://nope"}))

    assert response["error"]["code"] == RESOURCE_NOT_FOUND
    assert "i2as://status" in response["error"]["message"]


def test_an_unknown_method_is_a_method_not_found():
    """The adapter serves what it declares and refuses the rest by name."""
    adapter, _ = _adapter()

    response = adapter.handle(_request("prompts/list"))

    assert response["error"]["code"] == METHOD_NOT_FOUND


def test_a_notification_is_never_answered():
    """JSON-RPC forbids answering a message with no id."""
    adapter, _ = _adapter()

    assert adapter.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_a_broken_gateway_connection_is_an_internal_error_not_a_crash():
    """The session survives the app going away, and is told why."""
    adapter, client = _adapter()
    client.raise_on = "tools/list"

    response = adapter.handle(_request("tools/list"))

    assert response["error"]["code"] == INTERNAL_ERROR
    assert "the app went away" in response["error"]["message"]


def test_initialize_and_discover_both_answer_the_same_server():
    """Two protocol eras, one identity and one set of instructions."""
    adapter, _ = _adapter()

    handshake = adapter.handle(_request("initialize", {"protocolVersion": "2025-06-18"}))
    stateless = adapter.handle(_request("server/discover"))

    assert handshake["result"]["protocolVersion"] == "2025-06-18"
    assert handshake["result"]["serverInfo"] == stateless["result"]["serverInfo"]
    assert handshake["result"]["instructions"] == stateless["result"]["instructions"]


# ══════════════════════════════════════════════════════════════════════════
# The isolation is mechanical
# ══════════════════════════════════════════════════════════════════════════


def test_the_adapter_imports_only_the_contract_from_i2as():
    """Import contract C21, read off the modules themselves.

    The contract in ``pyproject.toml`` enumerates the forbidden modules,
    which a module added to ``i2as/core/`` later would not be in. This
    reads the adapter's own imports instead, so the allowlist cannot be
    outgrown.
    """
    allowed = {"i2as.core.events", "i2as.mcp"}
    offenders: list[str] = []
    for path in sorted(Path(translate.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if not name.startswith("i2as"):
                    continue
                if any(name == one or name.startswith(one + ".") for one in allowed):
                    continue
                offenders.append(f"{path.name}: {name}")

    assert not offenders, (
        "the MCP adapter runs in a process that must not be able to reach an "
        "instrument; it may import only i2as.core.events. Offending "
        f"imports: {offenders}"
    )


def test_the_optional_package_is_absent_and_the_shim_serves():
    """The suite runs without the extra, and says so rather than skipping.

    The ``mcp`` package is an optional extra precisely so that the transport
    is testable everywhere. If this ever fails it means the extra crept into
    the test environment, and the shim — the backend that always exists —
    would stop being the one the tests exercise.
    """
    reason = sdk_unavailable_reason()

    assert reason is not None
    assert "not installed" in reason


def test_the_sdk_backend_declines_rather_than_serving_partially():
    """``serve_with_sdk()`` reads no stdin when it cannot frame the session."""
    adapter, client = _adapter()

    assert serve_with_sdk(adapter) is False
    assert client.calls == []


# ══════════════════════════════════════════════════════════════════════════
# The descriptor
# ══════════════════════════════════════════════════════════════════════════


def test_a_missing_descriptor_says_the_app_is_not_running(tmp_path):
    """The message names the path, and the setting that publishes it."""
    with pytest.raises(GatewayError) as error:
        read_descriptor(tmp_path / "gateway.json")

    assert "gateway_server" in str(error.value)


def test_a_descriptor_from_a_different_wire_is_refused(tmp_path):
    """A schema mismatch is named, not guessed at."""
    descriptor = tmp_path / "gateway.json"
    descriptor.write_text(
        json.dumps({"schema": 99, "socket": "s", "token": "t"}), encoding="utf-8"
    )

    with pytest.raises(GatewayError) as error:
        read_descriptor(descriptor)

    assert "schema 99" in str(error.value)


# ══════════════════════════════════════════════════════════════════════════
# The stdio framing, driven as a real subprocess
# ══════════════════════════════════════════════════════════════════════════


class AdapterProcess:
    """``python -m i2as.mcp`` running for real, driven over its stdio.

    Deliberately a subprocess and not an in-process call: the point of this
    adapter is that it is a SEPARATE process which cannot reach an
    instrument, and the only way to test that claim honestly is to make the
    bytes cross a real pipe into a real interpreter.

    Attributes:
        messages: Every JSON message the adapter has written, in order.
    """

    def __init__(self, descriptor: Path, *, role: str, actor_id: str) -> None:
        """Launch the adapter against one descriptor.

        Args:
            descriptor: The ``gateway.json`` to find the app by.
            role: The role to declare at the handshake.
            actor_id: The identity to declare.
        """
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
        """Collect every whole line the adapter writes, until it closes."""
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if line.strip():
                self.messages.append(json.loads(line))

    def send(self, message: dict) -> None:
        """Write one JSON-RPC message to the adapter.

        Args:
            message: The message to send.
        """
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
        self.process.stdin.flush()

    def request(self, qtbot, orchestrator, method: str, params: dict | None = None) -> dict:
        """Send one request and spin the app's loop until it is answered.

        The app answers on the very event loop this test is running on, so
        waiting for the answer means driving that loop — and ticking the
        engine while waiting, so the app behaves as it does in the field.

        Args:
            qtbot: The pytest-qt fixture.
            orchestrator: The engine to tick while waiting.
            method: The MCP method.
            params: Its parameters.

        Returns:
            The response object carrying this request's id.
        """
        self._next_id += 1
        request_id = self._next_id
        self.send(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        )
        self.wait_for(
            qtbot,
            orchestrator,
            lambda: any(m.get("id") == request_id for m in self.messages),
        )
        return next(m for m in self.messages if m.get("id") == request_id)

    def wait_for(self, qtbot, orchestrator, predicate, timeout: int = 20000) -> None:
        """Tick the engine and spin the loop until *predicate* holds.

        Args:
            qtbot: The pytest-qt fixture.
            orchestrator: The engine to tick.
            predicate: What is being waited for.
            timeout: Milliseconds to wait.
        """

        def step() -> bool:
            orchestrator._tick()
            if self.process.poll() is not None:
                raise AssertionError(
                    "the adapter exited: " + self._stderr()
                )
            return predicate()

        qtbot.waitUntil(step, timeout=timeout)

    def _stderr(self) -> str:
        """Return whatever the adapter wrote to stderr, for a failure message."""
        assert self.process.stderr is not None
        return self.process.stderr.read().decode("utf-8", "replace")

    def close(self) -> None:
        """Close stdin and wait for the adapter to exit."""
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover — a wedged child
            self.process.kill()
            self.process.wait(timeout=10)


@pytest.fixture
def app(qtbot, tmp_path):
    """A listening ``GatewayServer`` over a real Orchestrator and sim station."""
    station = build_station(CONFIG_PATH)
    orchestrator = Orchestrator(
        station, tick_interval_ms=10, run_catalog={"FieldSweep": FieldSweep}
    )
    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe", email="jdoe@example.org"))
    store = ExperimentStore(tmp_path / "experiments")
    manager = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
        station=station,
        run_catalog={"FieldSweep": FieldSweep},
    )
    manager.start_experiment("Transport", "jdoe", dict(SAMPLE_INFO))
    server = GatewayServer(
        orchestrator,
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
    yield server, orchestrator, tmp_path / "gateway.json"
    server.stop()
    orchestrator.shutdown()


@pytest.fixture
def adapter_process(app):
    """The adapter subprocess, launched against the listening app."""
    _, _, descriptor = app
    process = AdapterProcess(descriptor, role="session", actor_id="mcp-test")
    yield process
    process.close()


def test_the_adapter_answers_an_initialize_over_real_pipes(
    qtbot, app, adapter_process
):
    """The handshake crosses a pipe, a socket, and back."""
    _, orchestrator, _ = app

    response = adapter_process.request(
        qtbot,
        orchestrator,
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    )

    assert response["result"]["protocolVersion"] == "2025-06-18"
    assert response["result"]["serverInfo"]["name"] == "i2as"
    assert "kill switch" in response["result"]["instructions"]


def test_the_subprocess_publishes_the_running_apps_own_tool_surface(
    qtbot, app, adapter_process
):
    """``tools/list`` is the station's rendered surface, not a written one."""
    server, orchestrator, _ = app
    adapter_process.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    response = adapter_process.request(qtbot, orchestrator, "tools/list")

    gateway = next(iter(server._connections.values())).gateway
    expected = [translate.mcp_tool(one) for one in gateway.tool_schemas()]
    assert response["result"]["tools"] == expected
    assert any(tool["name"] == "read_status" for tool in expected)


def test_a_tool_call_over_stdio_reaches_the_engine(qtbot, app, adapter_process):
    """One call crosses both wires and comes back as an MCP tool result."""
    _, orchestrator, _ = app

    response = adapter_process.request(
        qtbot,
        orchestrator,
        "tools/call",
        {"name": "read_status", "arguments": {}},
    )

    result = response["result"]
    assert result["isError"] is False
    answer = json.loads(result["content"][0]["text"])
    assert answer["ok"] is True
    assert answer["tool"] == "read_status"


def test_a_refused_call_comes_back_as_a_readable_result(qtbot, app, adapter_process):
    """A tool the surface does not have is answered, not raised."""
    _, orchestrator, _ = app

    response = adapter_process.request(
        qtbot, orchestrator, "tools/call", {"name": "no_such_tool", "arguments": {}}
    )

    assert response["result"]["isError"] is True
    answer = json.loads(response["result"]["content"][0]["text"])
    assert answer["detail"]["rule"] == "unknown_tool"


def test_a_resource_read_over_stdio_returns_the_station(qtbot, app, adapter_process):
    """``i2as://station`` is the declaration the app is running."""
    _, orchestrator, _ = app

    response = adapter_process.request(
        qtbot, orchestrator, "resources/read", {"uri": translate.STATION_URI}
    )

    payload = json.loads(response["result"]["contents"][0]["text"])
    assert payload["instruments"]


def test_the_apps_events_arrive_as_mcp_notifications(qtbot, app, adapter_process):
    """A tick the session never asked about still reaches it."""
    _, orchestrator, _ = app
    adapter_process.request(qtbot, orchestrator, "initialize", {})

    adapter_process.wait_for(
        qtbot,
        orchestrator,
        lambda: any(
            message.get("method") == "notifications/message"
            for message in adapter_process.messages
        ),
    )

    notification = next(
        message
        for message in adapter_process.messages
        if message.get("method") == "notifications/message"
    )
    assert notification["params"]["logger"] == "i2as.gateway"
    assert "event" in notification["params"]["data"]


def test_a_malformed_frame_on_stdin_is_a_parse_error(qtbot, app, adapter_process):
    """Rubbish on stdin is answered, and the session keeps working."""
    _, orchestrator, _ = app

    # Written raw rather than through `send()`: the point is bytes that are
    # not a JSON message at all, which `send()` could not produce.
    assert adapter_process.process.stdin is not None
    adapter_process.process.stdin.write(b"{not json at all\n")
    adapter_process.process.stdin.flush()

    adapter_process.wait_for(
        qtbot,
        orchestrator,
        lambda: any(
            message.get("error", {}).get("code") == -32700
            for message in adapter_process.messages
        ),
    )
    assert adapter_process.request(qtbot, orchestrator, "ping")["result"] == {}


def test_the_adapter_refuses_to_start_without_a_running_app(tmp_path):
    """No app, no session: it exits non-zero and says why."""
    completed = subprocess.run(  # noqa: S603 — our own interpreter
        [sys.executable, "-m", "i2as.mcp", "--descriptor", str(tmp_path / "gone.json")],
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        timeout=60,
    )

    assert completed.returncode == 1
    assert b"gateway_server" in completed.stderr


def test_a_role_above_the_ceiling_is_refused_before_the_session_opens(app, qtbot):
    """The app's ceiling is enforced at the handshake, in another process."""
    server, _, descriptor = app
    server.max_role = Role.OBSERVER

    process = AdapterProcess(descriptor, role="session", actor_id="too-eager")
    try:
        qtbot.waitUntil(lambda: process.process.poll() is not None, timeout=20000)
    finally:
        stderr = process.process.stderr.read() if process.process.stderr else b""
        process.process.stdin.close()

    assert process.process.returncode == 1
    assert b"observer" in stderr


def test_the_server_name_and_resource_uris_are_the_application_name():
    """The identifiers an MCP client matches on are spelled once, in translate."""
    assert translate.SERVER_NAME == "i2as"
    assert translate.STATUS_URI == "i2as://status"
    assert translate.STATION_URI == "i2as://station"
    assert translate.MANIFEST_URI == "i2as://manifest"
    assert translate.STATUS_URI in translate.INSTRUCTIONS
    assert translate.STATION_URI in translate.INSTRUCTIONS
