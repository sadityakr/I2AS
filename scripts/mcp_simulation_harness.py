from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QApplication

from i2as.core.orchestrator import Orchestrator
from i2as.core.procedure_catalog import build_procedure_infos
from i2as.core.station import build_station
from i2as.procedures.field_sweep import FieldSweep
from i2as.session.agent_feed import AgentFeed
from i2as.session.analysis_runner import AnalysisRunner
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
TOKEN = "test-token-not-a-secret"


class AdapterProcess:
    """Run the adapter as a real subprocess over the in-repo stdio shim."""

    def __init__(self, descriptor: Path, *, role: str, actor_id: str) -> None:
        self.messages: list[dict[str, Any]] = []
        self._next_id = 0
        self._lock = threading.Lock()
        self.process = subprocess.Popen(
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
                "--framing",
                "shim",
                "--log-level",
                "WARNING",
            ],
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        assert self.process.stdout is not None
        for raw_line in self.process.stdout:
            if raw_line.strip():
                message = json.loads(raw_line.decode("utf-8"))
                with self._lock:
                    self.messages.append(message)

    def send(self, message: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        self.process.stdin.flush()

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        tick: callable | None = None,
        timeout_s: float = 20.0,
    ) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if tick is not None:
                tick()
            with self._lock:
                for message in self.messages:
                    if message.get("id") == request_id:
                        return message
            if self.process.poll() is not None:
                raise RuntimeError(f"adapter exited unexpectedly: {self._stderr()}")
            time.sleep(0.05)
        raise TimeoutError(f"timed out waiting for {method!r}")

    def _stderr(self) -> str:
        assert self.process.stderr is not None
        return self.process.stderr.read().decode("utf-8", errors="replace")

    def close(self) -> None:
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


class HarnessReport(dict):
    pass


def ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def build_harness(tmp_path: Path) -> tuple[QApplication, GatewayServer, Orchestrator, ExperimentManager, AdapterProcess]:
    app = ensure_qapp()

    station = build_station(CONFIG_PATH)
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []

    catalog = {FieldSweep.__name__: FieldSweep}
    station.declare_procedures(build_procedure_infos(station, catalog))
    orchestrator = Orchestrator(station, tick_interval_ms=10, run_catalog=catalog)

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
    experiment = manager.start_experiment("MCP Simulation", "jdoe", dict(SAMPLE_INFO))

    settings = ElnSettings(
        enabled=True,
        backend="sim_eln",
        base_url="https://sim.example",
        api_key="k",
        retry_base_s=0.0,
        retry_max_s=0.0,
        analysis=AnalysisSettings(enabled=True, timeout_s=120.0),
    )
    adapter = SimElnAdapter({})
    publisher = ElnPublisher(manager, settings, adapter=adapter)
    manager.attach_eln_publisher(publisher)
    orchestrator.run_finished.connect(publisher.on_run_finished)

    analysis_runner = AnalysisRunner(manager, publisher, lambda: publisher.settings)

    feed = AgentFeed(store.agent_feed_path(experiment.experiment_id), experiment.experiment_id)
    feed.attach(orchestrator)

    gateway = Gateway(
        orchestrator,
        Role.SESSION,
        "sim-harness",
        station_info=station.station_info,
        tool_context=ToolContext(
            experiments=manager,
            run_catalog=catalog,
            status_log_path=tmp_path / "status.jsonl",
            publisher=publisher,
            analysis_runner=analysis_runner,
        ),
        feed=feed,
    )

    descriptor = tmp_path / "gateway.json"
    server = GatewayServer(
        orchestrator,
        socket_name=str(tmp_path / "gateway.sock"),
        descriptor=descriptor,
        token=TOKEN,
        max_role=Role.SESSION,
        station_info=station.station_info,
        tool_context=ToolContext(
            experiments=manager,
            run_catalog=catalog,
            status_log_path=tmp_path / "status.jsonl",
            publisher=publisher,
            analysis_runner=analysis_runner,
        ),
        feed=feed,
    )
    assert server.start(), "GatewayServer failed to start"

    adapter_proc = AdapterProcess(descriptor, role="session", actor_id="sim-harness")

    return app, server, orchestrator, manager, adapter_proc


def call_tool(
    proc: AdapterProcess,
    name: str,
    params: dict[str, Any] | None = None,
    *,
    tick: callable,
) -> tuple[bool, dict[str, Any]]:
    response = proc.request("tools/call", {"name": name, "arguments": params or {}}, tick=tick)
    result = response.get("result") or {}
    content = result.get("content") or []
    text = content[0].get("text", "{}") if content else "{}"
    envelope = json.loads(text)
    payload = envelope.get("result") if isinstance(envelope.get("result"), dict) else envelope
    return not result.get("isError", False) and envelope.get("ok", True), payload


def wait_for(predicate: callable, *, tick: callable, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        tick()
        if predicate():
            return True
        time.sleep(0.05)
    return False


def run_harness() -> dict[str, Any]:
    tmp_base = Path(tempfile.mkdtemp(prefix="i2as_mcp_harness_"))
    app, server, orchestrator, manager, proc = build_harness(tmp_base)

    report: dict[str, Any] = {
        "environment": {
            "platform": sys.platform,
            "python": sys.version,
            "framing": "shim",
            "repo_root": str(REPO_ROOT),
        },
        "steps": [],
        "problems": [],
        "risks": [],
    }

    def tick() -> None:
        orchestrator._tick()
        app.processEvents()

    try:
        init = proc.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "sim-harness", "version": "0"},
            },
            tick=tick,
        )
        report["steps"].append({
            "name": "initialize",
            "passed": bool(init.get("result")),
            "details": init,
        })

        tools = proc.request("tools/list", tick=tick)
        tool_names = [tool.get("name") for tool in (tools.get("result") or {}).get("tools", [])]
        report["steps"].append({
            "name": "tools/list",
            "passed": bool(tool_names),
            "details": {"tool_count": len(tool_names), "tools": tool_names},
        })

        ok, listed = call_tool(proc, "list_procedures", tick=tick)
        report["steps"].append({
            "name": "list_procedures",
            "passed": ok and any(entry.get("procedure") == "FieldSweep" for entry in listed.get("procedures", [])),
            "details": listed,
        })

        ok, default_form = call_tool(
            proc,
            "describe_procedure",
            {"procedure": "FieldSweep"},
            tick=tick,
        )
        group_keys = [group.get("key") for group in default_form.get("groups", [])]
        report["steps"].append({
            "name": "describe_procedure(default)",
            "passed": ok and bool(group_keys),
            "details": {
                "group_keys": group_keys,
                "selections": default_form.get("selections", {}),
            },
        })

        ok, conditional_form = call_tool(
            proc,
            "describe_procedure",
            {
                "procedure": "FieldSweep",
                "selections": {
                    "measurement_vi": "dc_measurement",
                    "loop1_parameter": "dc_measurement.current_A",
                },
            },
            tick=tick,
        )
        exposed_loop_values = any(
            "loop1_values" in str(param.get("name", ""))
            for group in conditional_form.get("groups", [])
            for param in group.get("params", [])
        )
        report["steps"].append({
            "name": "describe_procedure(selections)",
            "passed": ok and exposed_loop_values,
            "details": conditional_form,
        })

        data_dir = str(manager.current_data_dir())
        run_args = {
            "procedure": "FieldSweep",
            "params": {
                "measurement_vi": "dc_measurement",
                "field_start": -1.0,
                "field_end": 1.0,
                "field_steps": 5,
                "temperature": 300.0,
                "current_A": 1e-6,
                "readings_per_point": 3,
                "init_wait": 0.0,
                "step_wait": 0.0,
            },
            "sample_info": dict(SAMPLE_INFO),
            "data_directory": data_dir,
            "file_prefix": "harness",
        }

        ok, validated = call_tool(proc, "validate_run", run_args, tick=tick)
        report["steps"].append({
            "name": "validate_run",
            "passed": ok and validated.get("ok", False),
            "details": validated,
        })

        ok, probed = call_tool(
            proc,
            "probe_run",
            {**run_args, "probe_spec": {"n_points": 3, "averaging": 1, "max_wait_s": 0.0}},
            tick=tick,
        )
        report["steps"].append({
            "name": "probe_run",
            "passed": ok and probed.get("code") == "OK",
            "details": probed,
        })

        probe_done = wait_for(
            lambda: len(manager.current_experiment().runs) >= 1 and manager.current_experiment().runs[0].status == "done",
            tick=tick,
            timeout_s=30.0,
        )
        report["steps"].append({
            "name": "probe_run completion",
            "passed": probe_done,
            "details": {"runs": [run.to_dict() for run in manager.current_experiment().runs]},
        })

        ok, started = call_tool(proc, "run_procedure", run_args, tick=tick)
        report["steps"].append({
            "name": "run_procedure",
            "passed": ok and started.get("code") == "OK",
            "details": started,
        })

        run_done = wait_for(
            lambda: len(manager.current_experiment().runs) >= 2 and manager.current_experiment().runs[-1].status == "done",
            tick=tick,
            timeout_s=45.0,
        )
        report["steps"].append({
            "name": "run_procedure completion",
            "passed": run_done,
            "details": {
                "run_count": len(manager.current_experiment().runs),
                "latest_status": manager.current_experiment().runs[-1].status if manager.current_experiment().runs else None,
            },
        })

        run_id = manager.current_experiment().runs[-1].run_id
        ok, analysis_started = call_tool(proc, "run_analysis", {"run_id": run_id}, tick=tick)
        report["steps"].append({
            "name": "run_analysis",
            "passed": ok and analysis_started.get("started") is True,
            "details": analysis_started,
        })

        report_data: dict[str, Any] | None = None
        report_ready = wait_for(
            lambda: bool(report_data and report_data.get("status") == "ok"),
            tick=tick,
            timeout_s=60.0,
        )
        if not report_ready:
            ok, report_data = call_tool(proc, "read_analysis_report", {"run_id": run_id}, tick=tick)
        else:
            ok, report_data = call_tool(proc, "read_analysis_report", {"run_id": run_id}, tick=tick)
        report["steps"].append({
            "name": "read_analysis_report",
            "passed": ok and report_data.get("status") == "ok",
            "details": report_data,
        })

        ok, status_payload = call_tool(proc, "read_status", tick=tick)
        report["steps"].append({
            "name": "read_status",
            "passed": ok and status_payload.get("state") is not None,
            "details": status_payload,
        })

        ok, station_payload = call_tool(proc, "read_station_info", tick=tick)
        report["steps"].append({
            "name": "read_station_info",
            "passed": ok and bool(station_payload.get("instruments")),
            "details": station_payload,
        })

        ok, manifest_payload = call_tool(proc, "read_manifest", tick=tick)
        report["steps"].append({
            "name": "read_manifest",
            "passed": ok and bool(manifest_payload.get("setup")),
            "details": manifest_payload,
        })

    except Exception as exc:
        report["problems"].append({"error": f"{type(exc).__name__}: {exc}"})

    finally:
        proc.close()
        server.stop()
        orchestrator.shutdown()
        app.processEvents()

    tool_names = [step for step in report["steps"] if step["name"] == "tools/list"]
    if tool_names:
        names = set(tool_names[0].get("details", {}).get("tools", []))
        missing = {"list_procedures", "describe_procedure", "run_analysis"} - names
        if missing:
            report["risks"].append(
                f"MCP surface is missing expected tools: {', '.join(sorted(missing))}"
            )

    default_step = next((step for step in report["steps"] if step["name"] == "describe_procedure(default)"), None)
    if default_step and not default_step.get("passed", False):
        report["risks"].append("Dynamic procedure form discovery did not produce the expected default shape.")

    conditional_step = next((step for step in report["steps"] if step["name"] == "describe_procedure(selections)"), None)
    if conditional_step and not conditional_step.get("passed", False):
        report["risks"].append("Dynamic procedure form expansion did not react to structural selections.")

    report["summary"] = {
        "status": "pass" if not report["problems"] and not report["risks"] else "warning",
        "steps_passed": sum(1 for step in report["steps"] if step.get("passed", False)),
        "steps_total": len(report["steps"]),
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an MCP simulation harness against the I2AS adapter over stdio.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON report.")
    args = parser.parse_args()

    report = run_harness()
    print(json.dumps(report, indent=2 if args.pretty else None))
    return 0 if report["summary"].get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
