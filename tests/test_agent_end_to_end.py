"""End-to-end, as an agent: each shipped example driven THROUGH THE GATEWAY.

``test_analysis_end_to_end.py`` proves the transport half — a finished run's
manifest in, an analysed entry out. This is the other half of the same story:
nothing here calls the engine, the runner or the publisher directly. An
``Actor`` of kind ``agent`` under the ``session`` role reads the manifest,
validates the run, rehearses it as a probe, runs it, waits for ``RunFinished``,
asks for the analysis and reads the report back, all through
``Gateway.call_tool()`` — the same surface the MCP server publishes. What it
cannot do is approve: the analysed entry is parked on the run record until a
human approves it, and only then does the sim notebook see one entry with the
figures attached. The agent feed is asserted alongside, because an agent that
leaves no trail is the one thing this path must never allow.

One test per shipped example — the transport example (``sim_cryostat``,
Field Sweep) and the imaging example (``sim_imaging``, Field Imaging) — with
the same story, so the agent API is shown to be indifferent to what the
instruments are.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import i2as
from i2as.analysis.report import AnalysisReport
from i2as.core import events as ev
from i2as.core.orchestrator import Orchestrator
from i2as.core.station import build_station
from i2as.procedures.field_imaging import FieldImaging
from i2as.procedures.field_sweep import FieldSweep
from i2as.session.agent_feed import (
    RECORD_COMMAND,
    RECORD_TOOL,
    RECORD_VERDICT,
    AgentFeed,
    read_feed,
)
from i2as.session.analysis_runner import AnalysisRunner
from i2as.session.eln.outbox import DRAIN_PUBLISHED
from i2as.session.eln.publisher import ElnPublisher
from i2as.session.eln.settings import AnalysisSettings, ElnSettings
from i2as.session.eln.sim_eln import SimElnAdapter
from i2as.session.gateway import Gateway, Role, ToolContext
from i2as.session.gateway.gateway import event_stream
from i2as.session.manager import ExperimentManager
from i2as.session.models import User
from i2as.session.store import ExperimentStore, UserRoster

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX subprocess semantics assumed"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = Path(i2as.__file__).parent / "configs"

ACTOR_ID = "agent-e2e"
SAMPLE_INFO = {"sample_name": "S", "sample_id": "S-1", "comments": ""}

#: A probe that costs three points and no settle time at all.
PROBE_SPEC = {"n_points": 3, "averaging": 1, "max_wait_s": 0.0}
#: The probe standard caps EVERY seconds-valued parameter, and on the imaging
#: example that includes the camera's exposure, which the setup's config
#: bounds from below; a cap equal to the exposure leaves it alone.
IMAGING_PROBE_SPEC = {**PROBE_SPEC, "max_wait_s": 0.01}

#: How long one run may take on the sim; the ramps are sped up below, and
#: every declared wait is zero, so a run is a few dozen ticks.
RUN_TIMEOUT_MS = 30_000
#: How long the analysis worker (a separate process) may take.
ANALYSIS_TIMEOUT_MS = 60_000


@dataclass(frozen=True)
class Example:
    """One shipped example, as the agent leg sees it."""

    config_name: str
    procedure_cls: type
    params: dict[str, Any]
    recipe: str
    #: Figure files the recipe is expected to draw, in report order; empty
    #: means "at least one, whatever it is called".
    figures: tuple[str, ...] = field(default_factory=tuple)
    #: The probe reduction the rehearsal asks for.
    probe_spec: dict[str, Any] = field(default_factory=lambda: dict(PROBE_SPEC))


EXAMPLES = [
    Example(
        config_name="sim_cryostat",
        procedure_cls=FieldSweep,
        params={
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
        recipe="generic_sweep",
    ),
    Example(
        config_name="sim_imaging",
        procedure_cls=FieldImaging,
        params={
            "measurement_vi": "camera",
            "field_start": -1.0,
            "field_end": 1.0,
            "field_steps": 5,
            "saturation_field_T": -1.5,
            "init_wait": 0.0,
            "step_wait": 0.0,
            "exposure_s": 0.01,
            "binning": 1,
            "frames_per_step": 2,
        },
        recipe="field_image_stack",
        figures=("montage.png",),
        probe_spec=IMAGING_PROBE_SPEC,
    ),
]


@dataclass
class Wired:
    """Everything the production wiring builds, for the assertions."""

    gateway: Gateway
    orchestrator: Orchestrator
    manager: ExperimentManager
    publisher: ElnPublisher
    adapter: SimElnAdapter
    runner: AnalysisRunner
    feed: AgentFeed
    experiment_id: str
    finished: list[ev.RunFinished]


def _wire(example: Example, tmp_path: Path) -> Wired:
    """The wiring ``i2as.main`` does, over one example's sim station.

    One deliberate difference: the publisher's ``analysis_requested`` is NOT
    connected to the runner. In the application a finished run is analysed
    automatically; here the agent is the one asking, through ``run_analysis``,
    which is the tool the leg exists to exercise.
    """
    station = build_station(str(CONFIGS_DIR / example.config_name))
    # The sim magnet ramps at a realistic rate; a test cannot wait for it, so
    # it is sped up the way every engine-level suite does.
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []
    catalog = {example.procedure_cls.__name__: example.procedure_cls}
    orchestrator = Orchestrator(station, tick_interval_ms=10, run_catalog=catalog)

    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    store = ExperimentStore(tmp_path / "experiments")
    manager = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=orchestrator,
        config_name=example.config_name,
        station=station,
        run_catalog=catalog,
    )
    experiment = manager.start_experiment("Agent leg", "jdoe", dict(SAMPLE_INFO))

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
    runner = AnalysisRunner(manager, publisher, lambda: publisher.settings)

    feed = AgentFeed(
        store.agent_feed_path(experiment.experiment_id), experiment.experiment_id
    )
    feed.attach(orchestrator)
    finished: list[ev.RunFinished] = []
    event_stream(orchestrator).connect(
        lambda event: finished.append(event) if isinstance(event, ev.RunFinished) else None
    )

    gateway = Gateway(
        orchestrator,
        Role.SESSION,
        ACTOR_ID,
        station_info=station.station_info,
        tool_context=ToolContext(
            experiments=manager,
            run_catalog=catalog,
            status_log_path=tmp_path / "status.jsonl",
            publisher=publisher,
            analysis_runner=runner,
        ),
        feed=feed,
    )
    return Wired(
        gateway=gateway,
        orchestrator=orchestrator,
        manager=manager,
        publisher=publisher,
        adapter=adapter,
        runner=runner,
        feed=feed,
        experiment_id=experiment.experiment_id,
        finished=finished,
    )


@pytest.fixture(params=EXAMPLES, ids=lambda e: f"{e.config_name}/{e.procedure_cls.__name__}")
def wired(request, tmp_path, qtbot, monkeypatch):
    """One shipped example, wired as the application wires it."""
    # The analysis worker is a real subprocess importing the package.
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    parts = _wire(request.param, tmp_path)
    yield request.param, parts
    parts.runner.cancel()
    parts.publisher.stop()
    parts.orchestrator.shutdown()


def _run_args(example: Example, data_dir: str, *, prefix: str, probe: bool = False) -> dict:
    args = {
        "procedure": example.procedure_cls.__name__,
        "params": dict(example.params),
        "sample_info": dict(SAMPLE_INFO),
        "data_directory": data_dir,
        "file_prefix": prefix,
    }
    if probe:
        args["probe_spec"] = dict(example.probe_spec)
    return args


def _wait_for_run_done(wired: Wired, qtbot, run_index: int) -> str:
    """Wait for the ``run_index``-th recorded run to end, and return its id."""
    manager = wired.manager

    def _done() -> bool:
        runs = manager.current_experiment().runs
        return len(runs) > run_index and runs[run_index].status == "done"

    qtbot.waitUntil(_done, timeout=RUN_TIMEOUT_MS)
    run_id = manager.current_experiment().runs[run_index].run_id
    qtbot.waitUntil(
        lambda: any(e.run_id == run_id for e in wired.finished), timeout=RUN_TIMEOUT_MS
    )
    return run_id


def test_an_agent_runs_analyses_and_parks_an_entry_a_human_publishes(wired, qtbot):
    """read_manifest → validate_run → probe_run → run_procedure → RunFinished →
    run_analysis → read_analysis_report → parked → approved → published."""
    example, parts = wired
    gateway, manager, adapter = parts.gateway, parts.manager, parts.adapter
    data_dir = str(manager.current_data_dir())

    # 1. What is this station? The manifest names the setup and its VIs.
    manifest = gateway.call_tool("read_manifest")
    assert manifest["ok"] is True, manifest
    assert manifest["result"]["setup"] == example.config_name
    instruments = {entry["name"] for entry in manifest["result"]["instruments"]}
    assert example.params["measurement_vi"] in instruments

    # 2. May I run this? Nothing is dispatched.
    validated = gateway.call_tool(
        "validate_run",
        {
            "procedure": example.procedure_cls.__name__,
            "params": dict(example.params),
            "sample_info": dict(SAMPLE_INFO),
            "data_directory": data_dir,
        },
    )
    assert validated["ok"] is True, validated
    assert validated["result"]["ok"] is True, validated["result"]
    assert validated["result"]["duration_estimate_s"] >= 0

    # 3. Rehearse it as a probe: same procedure, same instruments, three points.
    probed = gateway.call_tool("probe_run", _run_args(example, data_dir, prefix="probe", probe=True))
    assert probed["code"] == "OK", probed
    probe_id = _wait_for_run_done(parts, qtbot, 0)
    probe_record = manager.current_experiment().find_run(probe_id)
    assert probe_record.kind == "probe"
    assert probe_record.actor.kind is ev.ActorKind.AGENT
    assert probe_record.actor.id == ACTOR_ID
    assert manager.pending_eln_draft(probe_id) == {}, "a probe is never notebook material"

    # 4. The real run, and its RunFinished on the event stream.
    started = gateway.call_tool("run_procedure", _run_args(example, data_dir, prefix="sweep"))
    assert started["code"] == "OK", started
    run_id = _wait_for_run_done(parts, qtbot, 1)
    run_record = manager.current_experiment().find_run(run_id)
    assert run_record.kind == "run"
    assert run_record.actor.kind is ev.ActorKind.AGENT
    ended = next(e for e in parts.finished if e.run_id == run_id)
    assert ended.status == "done", ended
    assert ended.manifest["procedure"] == example.procedure_cls.name

    # With analysis on, run end parks nothing and queues nothing on its own:
    # the notebook waits for the analysis, and the analysis waits to be asked.
    assert parts.publisher.pending_count() == 0
    assert manager.pending_eln_draft(run_id) == {}
    assert adapter.entries == {}

    # 5. The agent asks for the analysis, then reads the report back.
    analysis = gateway.call_tool("run_analysis", {"run_id": run_id})
    assert analysis["ok"] is True, analysis
    assert analysis["result"]["started"] is True
    with qtbot.waitSignal(parts.runner.analysis_finished, timeout=ANALYSIS_TIMEOUT_MS):
        pass
    report_answer = gateway.call_tool("read_analysis_report", {"run_id": run_id})
    assert report_answer["ok"] is True, report_answer
    assert report_answer["result"]["status"] == "ok", report_answer["result"]
    assert report_answer["result"]["recipe"] == example.recipe
    report = AnalysisReport.from_dict(report_answer["result"])
    report_dir = manager.store.report_dir(parts.experiment_id, run_id)
    figure_names = [figure.file for figure in report.figures]
    assert figure_names, "every shipped recipe draws at least one figure"
    for expected in example.figures:
        assert expected in figure_names, figure_names
    for name in figure_names:
        assert (report_dir / name).stat().st_size > 0

    # 6. The analysed entry is parked on the run, not published: approval is
    # the human's, and there is no tool for it.
    assert gateway.tool("approve_eln_draft") is None
    pending = manager.pending_eln_draft(run_id)
    assert pending["source"] == "analysis"
    assert [Path(a["path"]).name for a in pending["attachments"]] == figure_names
    assert parts.publisher.pending_count() == 0
    assert adapter.entries == {}

    # 7. The human approves in the eLab tab; the ordinary drain publishes it.
    assert manager.approve_eln_draft(run_id)
    assert parts.publisher.drain_once().state == DRAIN_PUBLISHED
    assert len(adapter.entries) == 1
    (entry,) = adapter.entries.values()
    assert entry["body_html"] == pending["body_html"]
    assert sorted(Path(u["path"]).name for u in adapter.uploads) == sorted(figure_names)
    assert manager.pending_eln_draft(run_id) == {}
    assert manager.current_experiment().find_run(run_id).eln_link is not None

    # 8. The trail: every command, its verdict and the recorded tool call name
    # the agent — the same actor the run records carry.
    records = read_feed(parts.feed.path)
    assert records
    for record in records:
        assert record["actor"] == {"kind": "agent", "id": ACTOR_ID, "role": "session"}, record
    by_request = {r["request_id"] for r in records if r["record"] == RECORD_COMMAND}
    assert {probed["request_id"], started["request_id"]} <= by_request
    answered = {r["request_id"] for r in records if r["record"] == RECORD_VERDICT}
    assert {probed["request_id"], started["request_id"]} <= answered
    tool_calls = [r for r in records if r["record"] == RECORD_TOOL]
    assert [r["tool"] for r in tool_calls] == ["run_analysis"]
    assert tool_calls[0]["args"] == {"run_id": run_id}
    assert tool_calls[0]["verdict"]["code"] == "OK"
