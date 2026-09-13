# ---
# description: |
#   Tests for remote access — the HTTP MCP endpoint (i2as/mcp/http_server.py),
#   the access keys it admits (i2as/mcp/keys.py), the controller methods
#   that own both (session/gateway/controller.py), and the client-config
#   text the Connections dialog renders. The endpoint is driven with the
#   stdlib http.client against a real GatewayServer over a sim station, so
#   what is asserted is the wire a web client sees.
# last_updated: 2026-09-13
# ---

from __future__ import annotations

import http.client
import json
import os
import stat
import threading
from typing import Any

import pytest

from i2as.core.orchestrator import Orchestrator
from i2as.core.station import build_station
from i2as.mcp.http_server import SESSION_HEADER, McpHttpServer
from i2as.mcp.keys import KEY_PREFIX, KeyStore, digest_key
from i2as.procedures.field_sweep import FieldSweep
from i2as.session.agent_feed import AgentFeed
from i2as.session.gateway import GatewayController, GatewayServer, Role, ToolContext
from i2as.session.manager import ExperimentManager
from i2as.session.models import User
from i2as.session.store import ExperimentStore, UserRoster

CONFIG_PATH = "i2as/configs/sim_cryostat"
TOKEN = "test-token-not-a-secret"
SAMPLE_INFO = {"sample_name": "S", "sample_id": "S-1", "comments": ""}


# ══════════════════════════════════════════════════════════════════════════
# The key store
# ══════════════════════════════════════════════════════════════════════════


def test_a_key_is_shown_once_and_only_its_digest_is_kept(tmp_path):
    store = KeyStore(tmp_path / "keys.json")

    key, secret = store.create("chatgpt", role="session")

    assert secret.startswith(KEY_PREFIX)
    assert key.digest == digest_key(secret)
    assert key.actor_id == "chatgpt"
    assert key.hint == secret[: len(KEY_PREFIX) + 4]
    on_disk = (tmp_path / "keys.json").read_text(encoding="utf-8")
    assert secret not in on_disk
    assert key.digest in on_disk


def test_verify_finds_the_key_by_secret_and_nothing_else(tmp_path):
    store = KeyStore(tmp_path / "keys.json")
    _, secret = store.create("a", role="observer")
    store.create("b", role="session")

    assert store.verify(secret).name == "a"
    assert store.verify(secret + "x") is None
    assert store.verify("") is None
    assert store.verify(None) is None  # type: ignore[arg-type]


def test_keys_persist_and_revoke_forgets_them(tmp_path):
    path = tmp_path / "keys.json"
    _, secret = KeyStore(path).create("laptop", role="observer")

    reopened = KeyStore(path)
    assert [key.name for key in reopened.keys()] == ["laptop"]
    assert reopened.verify(secret) is not None

    assert reopened.revoke("laptop") is True
    assert reopened.revoke("laptop") is False
    assert KeyStore(path).verify(secret) is None


def test_names_are_unique_and_never_blank(tmp_path):
    store = KeyStore(tmp_path / "keys.json")
    store.create("x", role="observer")

    with pytest.raises(ValueError):
        store.create("x", role="observer")
    with pytest.raises(ValueError):
        store.create("   ", role="observer")


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_the_key_file_is_owner_only(tmp_path):
    store = KeyStore(tmp_path / "keys.json")
    store.create("x", role="observer")

    mode = stat.S_IMODE((tmp_path / "keys.json").stat().st_mode)
    assert mode == 0o600


def test_a_corrupt_file_is_treated_as_empty(tmp_path):
    path = tmp_path / "keys.json"
    path.write_text("not json", encoding="utf-8")

    assert KeyStore(path).keys() == []


# ══════════════════════════════════════════════════════════════════════════
# The endpoint over a real Gateway server
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def served(qtbot, tmp_path):
    """A GatewayServer over a sim station, an HTTP endpoint over it, one key."""
    station = build_station(CONFIG_PATH)
    orch = Orchestrator(station, tick_interval_ms=10, run_catalog={"FieldSweep": FieldSweep})
    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe", email="jdoe@example.org"))
    manager = ExperimentManager(
        store=ExperimentStore(tmp_path / "experiments"),
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
    store = KeyStore(tmp_path / "keys.json")
    key, secret = store.create("web", role="session")
    http_server = McpHttpServer(
        server.fullServerName() or server.socket_name,
        TOKEN,
        store,
        host="127.0.0.1",
        port=0,
        public_url="https://lab.example.ngrok.app/mcp",
    )
    http_server.start()
    yield http_server, secret, store, server, orch
    http_server.stop()
    server.stop()
    orch.shutdown()


def request(
    qtbot,
    http_server: McpHttpServer,
    method: str,
    path: str = "/mcp",
    *,
    body: Any | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """Make one HTTP request off the GUI thread while the Qt loop spins.

    The endpoint answers a request by asking the Gateway server, which
    lives on the test's own (Qt) thread — so the request must be made from
    another thread and waited for with the event loop running.
    """
    outcome: dict[str, Any] = {}

    def go() -> None:
        connection = http.client.HTTPConnection("127.0.0.1", http_server.port, timeout=10)
        try:
            payload = json.dumps(body).encode("utf-8") if body is not None else None
            sent = dict(headers or {})
            if payload is not None:
                sent.setdefault("Content-Type", "application/json")
            connection.request(method, path, body=payload, headers=sent)
            response = connection.getresponse()
            outcome["status"] = response.status
            outcome["headers"] = {k.lower(): v for k, v in response.getheaders()}
            outcome["body"] = response.read()
        except Exception as error:  # noqa: BLE001 — surfaced by the assertion below
            outcome["error"] = error
        finally:
            connection.close()

    thread = threading.Thread(target=go, daemon=True)
    thread.start()
    qtbot.waitUntil(lambda: bool(outcome), timeout=10000)
    assert "error" not in outcome, outcome.get("error")
    return outcome["status"], outcome["headers"], outcome["body"]


def rpc(method: str, params: dict | None = None, request_id: int | None = 1) -> dict:
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if request_id is not None:
        message["id"] = request_id
    return message


def bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


def test_no_key_is_401_with_a_challenge(qtbot, served):
    http_server, _secret, *_ = served

    status, headers, _ = request(qtbot, http_server, "POST", body=rpc("tools/list"))

    assert status == 401
    assert headers["www-authenticate"].startswith("Bearer")


def test_a_wrong_key_is_401(qtbot, served):
    http_server, secret, *_ = served

    status, _, _ = request(
        qtbot, http_server, "POST", body=rpc("tools/list"), headers=bearer(secret + "z")
    )

    assert status == 401


def test_initialize_mints_a_session_id_and_tools_list_is_the_gateways(qtbot, served):
    http_server, secret, *_ = served

    status, headers, body = request(
        qtbot,
        http_server,
        "POST",
        body=rpc("initialize", {"protocolVersion": "2025-06-18"}),
        headers=bearer(secret),
    )
    assert status == 200
    assert headers["content-type"] == "application/json"
    session_id = headers[SESSION_HEADER.lower()]
    assert session_id
    assert json.loads(body)["result"]["serverInfo"]["name"] == "i2as"

    status, _, body = request(
        qtbot,
        http_server,
        "POST",
        body=rpc("tools/list", request_id=2),
        headers={**bearer(secret), SESSION_HEADER: session_id},
    )
    assert status == 200
    names = {tool["name"] for tool in json.loads(body)["result"]["tools"]}
    assert "read_status" in names

    # The key is the agent: the socket server sees one connection under
    # its actor id and role.
    assert http_server.connections() == [{"key": "web", "actor_id": "web", "role": "session"}]
    assert served[3].connections() == [{"actor_id": "web", "role": "session"}]


def test_a_notification_is_202_and_an_unknown_session_is_404(qtbot, served):
    http_server, secret, *_ = served

    status, _, body = request(
        qtbot,
        http_server,
        "POST",
        body=rpc("notifications/initialized", request_id=None),
        headers=bearer(secret),
    )
    assert status == 202
    assert body == b""

    status, _, _ = request(
        qtbot,
        http_server,
        "POST",
        body=rpc("tools/list"),
        headers={**bearer(secret), SESSION_HEADER: "never-issued"},
    )
    assert status == 404


def test_the_key_may_travel_in_the_path_for_clients_without_headers(qtbot, served):
    http_server, secret, *_ = served

    status, _, body = request(qtbot, http_server, "POST", f"/mcp/{secret}", body=rpc("ping"))

    assert status == 200
    assert json.loads(body) == {"jsonrpc": "2.0", "id": 1, "result": {}}


def test_a_tool_call_is_answered_by_the_app(qtbot, served):
    http_server, secret, *_ = served

    status, _, body = request(
        qtbot,
        http_server,
        "POST",
        body=rpc("tools/call", {"name": "read_status", "arguments": {}}),
        headers=bearer(secret),
    )

    assert status == 200
    result = json.loads(body)["result"]
    assert result["isError"] is False
    assert "content" in result


def test_a_browser_origin_is_judged(qtbot, served):
    http_server, secret, *_ = served

    status, _, _ = request(
        qtbot,
        http_server,
        "POST",
        body=rpc("ping"),
        headers={**bearer(secret), "Origin": "https://evil.example"},
    )
    assert status == 403

    for origin in ("http://localhost:5173", "https://lab.example.ngrok.app"):
        status, headers, _ = request(
            qtbot,
            http_server,
            "POST",
            body=rpc("ping"),
            headers={**bearer(secret), "Origin": origin},
        )
        assert status == 200, origin
        assert headers["access-control-allow-origin"] == origin


def test_malformed_and_batch_bodies_are_400(qtbot, served):
    http_server, secret, *_ = served

    outcome: dict[str, Any] = {}

    def go() -> None:
        connection = http.client.HTTPConnection("127.0.0.1", http_server.port, timeout=10)
        connection.request("POST", "/mcp", body=b"{not json", headers=bearer(secret))
        outcome["status"] = connection.getresponse().status
        connection.close()

    thread = threading.Thread(target=go, daemon=True)
    thread.start()
    qtbot.waitUntil(lambda: bool(outcome), timeout=10000)
    assert outcome["status"] == 400

    status, _, body = request(
        qtbot, http_server, "POST", body=[rpc("ping")], headers=bearer(secret)
    )
    assert status == 400
    assert "batches" in json.loads(body)["error"]["message"]


def test_other_paths_are_404(qtbot, served):
    http_server, secret, *_ = served

    status, _, _ = request(qtbot, http_server, "POST", "/other", body=rpc("ping"), headers=bearer(secret))

    assert status == 404


def test_the_events_stream_carries_the_apps_notifications(qtbot, served):
    http_server, secret, _store, _server, orch = served
    # Open the key's connection first, so the stream drains a live one.
    assert request(qtbot, http_server, "POST", body=rpc("ping"), headers=bearer(secret))[0] == 200

    received: list[dict] = []
    stop = threading.Event()

    def listen() -> None:
        connection = http.client.HTTPConnection("127.0.0.1", http_server.port, timeout=10)
        connection.request(
            "GET", "/mcp", headers={**bearer(secret), "Accept": "text/event-stream"}
        )
        response = connection.getresponse()
        received.append({"status": response.status, "type": response.getheader("Content-Type")})
        while not stop.is_set():
            line = response.fp.readline()
            if not line:
                break
            if line.startswith(b"data: "):
                received.append(json.loads(line[6:]))
        connection.close()

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    qtbot.waitUntil(lambda: bool(received), timeout=10000)
    assert received[0] == {"status": 200, "type": "text/event-stream"}

    # A tick of the engine emits events the gateway forwards to every
    # subscribed connection, and the stream writes each as one SSE event.
    def tick_and_check() -> bool:
        orch._tick()
        return len(received) > 1

    qtbot.waitUntil(tick_and_check, timeout=10000)
    stop.set()
    notification = received[1]
    assert notification["method"] == "notifications/message"
    assert notification["params"]["logger"] == "i2as.gateway"


def test_revoking_a_key_drops_its_connection_and_refuses_it_after(qtbot, served):
    http_server, secret, store, server, _orch = served
    assert request(qtbot, http_server, "POST", body=rpc("ping"), headers=bearer(secret))[0] == 200
    assert server.connections()

    store.revoke("web")
    http_server.drop_key("web")

    status, _, _ = request(qtbot, http_server, "POST", body=rpc("ping"), headers=bearer(secret))
    assert status == 401
    assert http_server.connections() == []
    qtbot.waitUntil(lambda: server.connections() == [], timeout=5000)


def test_a_key_above_the_ceiling_is_refused_by_the_app_not_served(qtbot, served):
    http_server, _secret, store, server, _orch = served
    server.max_role = Role.OBSERVER  # tightened after the key was issued
    _, ambitious = store.create("ambitious", role="session")

    status, _, body = request(qtbot, http_server, "POST", body=rpc("ping"), headers=bearer(ambitious))

    assert status == 503
    assert "refused" in json.loads(body)["error"]


def test_stop_closes_every_connection(qtbot, served):
    http_server, secret, _store, server, _orch = served
    assert request(qtbot, http_server, "POST", body=rpc("ping"), headers=bearer(secret))[0] == 200

    http_server.stop()

    assert not http_server.listening
    qtbot.waitUntil(lambda: server.connections() == [], timeout=5000)


# ══════════════════════════════════════════════════════════════════════════
# The controller owns both
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def controller(qtbot, tmp_path):
    station = build_station(CONFIG_PATH)
    orch = Orchestrator(station, tick_interval_ms=10, run_catalog={"FieldSweep": FieldSweep})
    control = GatewayController(
        orch,
        station_info=station.station_info,
        tool_context=ToolContext(experiments=None, run_catalog={"FieldSweep": FieldSweep}),
        feed=lambda: None,
        ceiling=Role.DEBUG,
        socket_name=str(tmp_path / "gateway.sock"),
        descriptor=tmp_path / "gateway.json",
        key_store=tmp_path / "keys.json",
    )
    yield control
    control.stop()
    orch.shutdown()


def test_remote_access_needs_the_gateway_on(controller):
    with pytest.raises(RuntimeError):
        controller.start_http(port=0)


def test_create_key_respects_the_ceiling(controller):
    with pytest.raises(ValueError):
        controller.create_key("too-much", Role.SESSION)

    key, secret = controller.create_key("ok", "debug", actor_id="laptop-agent")
    assert key.role == "debug"
    assert key.actor_id == "laptop-agent"
    assert controller.keys()[0].name == "ok"
    assert controller.key_store.verify(secret) is not None


def test_the_endpoint_follows_the_socket_server_through_a_restart(qtbot, controller):
    controller.start(Role.OBSERVER)
    first = controller.start_http(port=0)
    port = first.port
    assert controller.http_enabled

    controller.start(Role.DEBUG)  # a new socket server, a new token

    assert controller.http_enabled
    assert controller.http_server is not first
    assert controller.http_server.port == port

    controller.stop()
    assert not controller.http_enabled
    assert controller.server is None


def test_revoke_key_drops_the_live_connection(qtbot, controller):
    controller.start(Role.OBSERVER)
    http_server = controller.start_http(port=0)
    _, secret = controller.create_key("web", Role.OBSERVER)
    assert request(qtbot, http_server, "POST", body=rpc("ping"), headers=bearer(secret))[0] == 200
    assert http_server.connections()

    assert controller.revoke_key("web") is True

    assert http_server.connections() == []
    assert request(qtbot, http_server, "POST", body=rpc("ping"), headers=bearer(secret))[0] == 401


# ══════════════════════════════════════════════════════════════════════════
# The client-config text
# ══════════════════════════════════════════════════════════════════════════


def test_client_configs_carry_the_url_and_key_in_each_clients_shape():
    from i2as.gui.connections_dialog import CLIENT_CHOICES, render_client_config

    url = "https://lab.example.ngrok.app/mcp"
    claude = json.loads(render_client_config(CLIENT_CHOICES[0], url, "i2as_abc"))
    assert claude["mcpServers"]["i2as"] == {
        "type": "http",
        "url": url,
        "headers": {"Authorization": "Bearer i2as_abc"},
    }

    chatgpt = render_client_config(CLIENT_CHOICES[1], url, "i2as_abc")
    assert f"{url}/i2as_abc" in chatgpt
    assert "No authentication" in chatgpt

    openwebui = render_client_config(CLIENT_CHOICES[2], url, "i2as_abc")
    assert url in openwebui and "i2as_abc" in openwebui

    generic = render_client_config(CLIENT_CHOICES[3], url)
    assert "Authorization: Bearer <your key>" in generic


# ══════════════════════════════════════════════════════════════════════════
# The dialog applies and persists the remote-access settings
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Redirect the app QSettings factory to a throwaway INI file."""
    from PyQt6.QtCore import QSettings

    from i2as.gui import app_settings

    ini_path = tmp_path / "i2as_test_settings.ini"
    monkeypatch.setattr(
        app_settings,
        "get_settings",
        lambda: QSettings(str(ini_path), QSettings.Format.IniFormat),
    )
    return ini_path


def test_the_dialog_switches_remote_access_on_and_persists_it(qtbot, controller, isolated_settings):
    from PyQt6.QtWidgets import QCheckBox, QLineEdit, QPlainTextEdit, QPushButton, QSpinBox

    from i2as.gui import app_settings
    from i2as.gui.connections_dialog import ConnectionsDialog

    controller.create_key("web", Role.OBSERVER)
    dialog = ConnectionsDialog(controller)
    qtbot.addWidget(dialog)

    dialog.findChild(QCheckBox, "connections_enabled_checkbox").setChecked(True)
    dialog.findChild(QCheckBox, "connections_remote_checkbox").setChecked(True)
    dialog.findChild(QSpinBox, "connections_port_spin").setValue(18765)
    dialog.findChild(QLineEdit, "connections_public_url_edit").setText("https://lab.example.ngrok.app")
    dialog.findChild(QPushButton, "connections_save_btn").click()

    assert controller.enabled
    assert controller.http_enabled
    assert controller.http_server.port == 18765
    assert controller.http_server.public_url == "https://lab.example.ngrok.app"
    assert app_settings.remote_access_enabled() is True
    assert app_settings.remote_access_port() == 18765
    assert app_settings.remote_access_public_url() == "https://lab.example.ngrok.app"

    config = dialog.findChild(QPlainTextEdit, "connections_client_config").toPlainText()
    assert "https://lab.example.ngrok.app/mcp" in config
    assert "<your key>" in config  # the secret was not created in this dialog

    dialog.findChild(QCheckBox, "connections_remote_checkbox").setChecked(False)
    dialog.findChild(QPushButton, "connections_save_btn").click()
    assert not controller.http_enabled
    assert app_settings.remote_access_enabled() is False
