"""Stand up a real, long-running I2AS instance (sim station) with its
Gateway server listening, plus the real ``python -m i2as.mcp`` adapter
connected to it — and a tiny local relay so a human (or another process) can
send one real MCP JSON-RPC request at a time and see the real response.

This is not a scenario runner: it performs no procedure calls itself. It
only assembles the same collaborators the desktop app wires (Station,
Orchestrator, ExperimentManager, GatewayServer, AnalysisRunner,
ElnPublisher) against a real Qt event loop — so the Orchestrator's own
QTimer ticks for real, exactly as it does in the shipped app — and keeps
them alive until killed.

Usage:
    python scripts/mcp_instance.py [--port 8765]

Then, from another process, send one line of JSON per request to
127.0.0.1:<port> and read one line of JSON back (an array of every MCP
message — the answer plus any notifications — that arrived while waiting).
"""

from __future__ import annotations

import argparse
import json
import socketserver
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QApplication

from i2as.core.orchestrator import Orchestrator
from i2as.core.procedure_catalog import build_procedure_infos
from i2as.core.station import build_station
from i2as.procedures.field_sweep import FieldSweep
from i2as.session.agent_feed import AgentFeed
from i2as.session.analysis_runner import AnalysisRunner
from i2as.session.eln.drafting import FakeDraftClient
from i2as.session.eln.publisher import ElnPublisher
from i2as.session.eln.settings import AnalysisSettings, ElnSettings
from i2as.session.eln.sim_eln import SimElnAdapter
from i2as.session.gateway import Gateway, Role, ToolContext
from i2as.session.gateway.local_server import GatewayServer
from i2as.session.manager import ExperimentManager
from i2as.session.models import User
from i2as.session.store import ExperimentStore, UserRoster

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "i2as/configs/sim_cryostat"
SAMPLE_INFO = {"sample_name": "S", "sample_id": "S-1", "comments": ""}
TOKEN = "instance-token-not-a-secret"


class _AdapterBridge:
    """Owns the real MCP adapter subprocess and every message it has sent."""

    def __init__(self, descriptor: Path) -> None:
        self.messages: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "i2as.mcp",
                "--descriptor",
                str(descriptor),
                "--role",
                "session",
                "--actor-id",
                "live-agent",
                "--framing",
                "shim",
                "--log-level",
                "INFO",
            ],
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if line.strip():
                with self._lock:
                    self.messages.append(json.loads(line))

    def _pump_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            sys.stderr.write("[adapter] " + line.decode("utf-8", "replace"))
            sys.stderr.flush()

    def send(self, request: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        self.process.stdin.flush()

    def messages_from(self, index: int) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.messages[index:])

    def message_count(self) -> int:
        with self._lock:
            return len(self.messages)


def _make_handler(bridge: _AdapterBridge):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            raw = self.rfile.readline()
            if not raw.strip():
                return
            try:
                request = json.loads(raw)
            except ValueError as error:
                self.wfile.write((json.dumps({"error": str(error)}) + "\n").encode("utf-8"))
                return
            baseline = bridge.message_count()
            bridge.send(request)
            request_id = request.get("id")
            deadline = threading.Event()
            import time

            start = time.monotonic()
            collected: list[dict[str, Any]] = []
            while time.monotonic() - start < 30.0:
                collected = bridge.messages_from(baseline)
                if request_id is None:
                    break
                if any(m.get("id") == request_id for m in collected):
                    break
                if bridge.process.poll() is not None:
                    collected.append({"error": "adapter process exited"})
                    break
                time.sleep(0.05)
            self.wfile.write((json.dumps(collected) + "\n").encode("utf-8"))

    return Handler


def build_instance(tmp_path: Path):
    app = QApplication.instance() or QApplication([])

    station = build_station(CONFIG_PATH)
    station.magnet_z._default_ramp_rate = 6000.0  # fast ramps for an interactive live demo
    station.magnet_z._ramp_segments = []
    catalog = {FieldSweep.__name__: FieldSweep}
    station.declare_procedures(build_procedure_infos(station, catalog))

    orchestrator = Orchestrator(station, tick_interval_ms=250, run_catalog=catalog)

    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe", email="jdoe@example.org"))
    store = ExperimentStore(tmp_path / "experiments")
    manager = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
        station=station,
        run_catalog=catalog,
    )
    experiment = manager.start_experiment("Live MCP session", "jdoe", dict(SAMPLE_INFO))
    manager.set_attended(False)  # let the agent publish without a human parking every draft

    settings = ElnSettings(
        enabled=True,
        backend="sim_eln",
        base_url="https://sim.example",
        api_key="k",
        retry_base_s=0.0,
        retry_max_s=0.0,
        analysis=AnalysisSettings(enabled=True, timeout_s=120.0),
    )
    eln_adapter = SimElnAdapter({})
    publisher = ElnPublisher(manager, settings, adapter=eln_adapter)
    manager.attach_eln_publisher(publisher)
    orchestrator.run_finished.connect(publisher.on_run_finished)
    publisher.start()

    analysis_runner = AnalysisRunner(manager, publisher, lambda: publisher.settings)

    feed = AgentFeed(store.agent_feed_path(experiment.experiment_id), experiment.experiment_id)
    feed.attach(orchestrator)

    tool_context = ToolContext(
        experiments=manager,
        run_catalog=catalog,
        status_log_path=tmp_path / "status.jsonl",
        publisher=publisher,
        analysis_runner=analysis_runner,
        draft_client=FakeDraftClient(),
    )

    descriptor = tmp_path / "gateway.json"
    server = GatewayServer(
        orchestrator,
        socket_name=str(tmp_path / "gateway.sock"),
        descriptor=descriptor,
        token=TOKEN,
        max_role=Role.SESSION,
        station_info=station.station_info,
        tool_context=tool_context,
        feed=feed,
    )
    assert server.start(), "GatewayServer failed to start"

    return app, server, orchestrator, manager, eln_adapter, descriptor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-dir", default=None, help="Where to keep experiments/gateway files")
    args = parser.parse_args()

    tmp_path = Path(args.state_dir) if args.state_dir else Path(tempfile.mkdtemp(prefix="i2as_live_"))
    tmp_path.mkdir(parents=True, exist_ok=True)

    app, server, orchestrator, manager, eln_adapter, descriptor = build_instance(tmp_path)
    bridge = _AdapterBridge(descriptor)

    relay = socketserver.ThreadingTCPServer(("127.0.0.1", args.port), _make_handler(bridge))
    relay.daemon_threads = True
    threading.Thread(target=relay.serve_forever, daemon=True).start()

    print(f"state_dir={tmp_path}")
    print(f"descriptor={descriptor}")
    print(f"relay=127.0.0.1:{args.port}")
    print("ready", flush=True)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
