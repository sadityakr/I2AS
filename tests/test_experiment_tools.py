"""The agent's experiment lifecycle tools: start_experiment, add_finding, close_experiment.

An agent opens, annotates and closes experiments through the same
``ExperimentManager`` the operator's dialogs use, under three rules this file
pins: an agent-opened experiment is always ATTENDED and owned by the
station's user; an envelope may only narrow the setup's limits; and every
call is recorded in the trail of the experiment it acted on — including the
call that opened it and the one that closed it.
"""

from __future__ import annotations

import pytest

from i2as.core.orchestrator import Orchestrator
from i2as.core.station import build_station
from i2as.procedures.field_sweep import FieldSweep
from i2as.session.agent_feed import AgentFeed, read_feed
from i2as.session.gateway import Gateway, Role, ToolContext
from i2as.session.manager import ExperimentManager
from i2as.session.models import User
from i2as.session.store import ExperimentStore, UserRoster

CONFIG_PATH = "i2as/configs/sim_cryostat"


@pytest.fixture
def lab(qtbot, tmp_path):
    """An engine, an experiment manager with no experiment open, and a gateway factory."""
    station = build_station(CONFIG_PATH)
    orch = Orchestrator(station, tick_interval_ms=10, run_catalog={"FieldSweep": FieldSweep})
    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    store = ExperimentStore(tmp_path / "experiments")
    manager = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=orch,
        config_name="sim_cryostat",
        station=station,
        run_catalog={"FieldSweep": FieldSweep},
    )
    feeds: dict[str, AgentFeed] = {}

    def feed_source():
        record = manager.current_experiment()
        if record is None:
            return None
        return feeds.setdefault(
            record.experiment_id,
            AgentFeed(store.agent_feed_path(record.experiment_id), record.experiment_id),
        )

    context = ToolContext(
        experiments=manager,
        run_catalog={"FieldSweep": FieldSweep},
        status_log_path=tmp_path / "status.jsonl",
        current_user=lambda: "jdoe",
    )
    orch.start_monitoring()

    def build(role=Role.SESSION):
        gateway = Gateway(
            orch, role, "agent-7", station_info=station.station_info,
            tool_context=context, feed=feed_source,
        )
        qtbot.waitUntil(lambda: gateway.status() is not None, timeout=3000)
        return gateway

    build.station = station
    yield build, manager, store
    orch.shutdown()


def test_an_agent_opens_an_attended_experiment_owned_by_the_station_user(lab) -> None:
    build, manager, _ = lab
    answer = build().call_tool(
        "start_experiment",
        {
            "title": "p-MTJ TMR",
            "sample_name": "pMTJ-B7",
            "sample_id": "W12-B7",
            "envelope": {"magnet_z": {"min_value": -0.5, "max_value": 0.5}},
        },
    )
    assert answer["ok"], answer
    record = manager.current_experiment()
    assert record.title == "p-MTJ TMR"
    assert record.user_id == "jdoe"
    assert record.attended is True
    assert record.envelope["magnet_z"]["max_value"] == 0.5
    again = build().call_tool("start_experiment", {"title": "second", "sample_name": "x"})
    assert again["detail"]["rule"] == "experiment_open"


def test_an_envelope_may_only_narrow(lab) -> None:
    build, manager, _ = lab
    answer = build().call_tool(
        "start_experiment",
        {"title": "t", "sample_name": "s", "envelope": {"magnet_z": {"max_value": 20.0}}},
    )
    assert answer["detail"]["rule"] == "envelope_wider_than_setup"
    assert manager.current_experiment() is None


def test_findings_are_appended_and_stamped(lab) -> None:
    build, manager, _ = lab
    gateway = build()
    gateway.call_tool("start_experiment", {"title": "t", "sample_name": "s"})
    manager.set_findings("Operator note.")
    assert gateway.call_tool("add_finding", {"text": "TMR = 182 % at 10 K."})["ok"]
    assert gateway.call_tool("add_finding", {"text": "Hc = 43 mT."})["ok"]
    findings = manager.current_experiment().findings.splitlines()
    assert findings[0] == "Operator note."
    assert "agent 'agent-7'" in findings[1] and findings[1].endswith("TMR = 182 % at 10 K.")
    assert findings[2].endswith("Hc = 43 mT.")


def test_close_and_the_trail_holds_both_ends(lab) -> None:
    build, manager, store = lab
    gateway = build()
    opened = gateway.call_tool("start_experiment", {"title": "t", "sample_name": "s"})
    experiment_id = opened["result"]["experiment_id"]
    closed = gateway.call_tool("close_experiment", {})
    assert closed["ok"], closed
    assert manager.current_experiment() is None
    assert gateway.call_tool("add_finding", {"text": "late"})["detail"]["rule"] == "no_experiment"
    tools = [r.get("tool") for r in read_feed(store.agent_feed_path(experiment_id))]
    assert tools[0] == "start_experiment" and tools[-1] == "close_experiment"


def test_close_is_refused_while_a_run_is_in_flight(lab, qtbot) -> None:
    build, manager, _ = lab
    gateway = build()
    gateway.call_tool("start_experiment", {"title": "t", "sample_name": "s"})
    started = gateway.call_tool(
        "run_procedure",
        {"procedure": "FieldSweep", "params": {"field_steps": 3, "init_wait": 30.0, "set_temperature": False},
         "sample_info": {"sample_name": "s", "sample_id": "", "comments": ""}},
    )
    assert started["code"] in ("OK", "PENDING"), started
    qtbot.waitUntil(lambda: gateway.status().state != "IDLE", timeout=3000)
    assert gateway.call_tool("close_experiment", {})["detail"]["rule"] == "engine_busy"
    assert manager.current_experiment() is not None


def test_an_observer_cannot_open_or_close(lab) -> None:
    build, manager, _ = lab
    answer = build(Role.OBSERVER).call_tool("start_experiment", {"title": "t", "sample_name": "s"})
    assert not answer["ok"]
    assert manager.current_experiment() is None


def test_an_agent_queued_run_is_visible_in_the_queue_panel(lab, qtbot) -> None:
    """A run an agent hands to the engine is listed, labelled with who queued it."""
    from i2as.gui.queue_panel import QueuePanel

    build, manager, _ = lab
    gateway = build()
    gateway.call_tool("start_experiment", {"title": "t", "sample_name": "s"})
    panel = QueuePanel(
        build.station, gateway._engine, get_experiment_info=lambda: {},
        queue_host=manager.run_queue_host, procedure_classes=[FieldSweep],
    )
    qtbot.addWidget(panel)
    for prefix in ("first", "RH_0.05V"):
        gateway.call_tool(
            "queue_procedure" if prefix != "first" else "run_procedure",
            {"procedure": "FieldSweep", "file_prefix": prefix,
             "params": {"field_steps": 3, "init_wait": 30.0, "set_temperature": False},
             "sample_info": {"sample_name": "s", "sample_id": "", "comments": ""}},
        )
    qtbot.waitUntil(lambda: panel._queue_list.count() >= 1, timeout=3000)
    text = panel._queue_list.item(0).text()
    assert "[RH_0.05V] Field Sweep" in text and "queued by agent 'agent-7'" in text
