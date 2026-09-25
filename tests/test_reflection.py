# ---
# description: |
#   The reflection standard, end to end: a verdict echoes the command it
#   answers; the run manifest names its class and owner and the mirror keeps
#   it; the procedure form, the instrument cards, the agent panel and the
#   takeover strip all render an action whoever took it, so the GUI is in the
#   state it would be in had the operator set it — and the operator's own
#   Run Now goes through the same door as an agent's.
# last_updated: 2026-09-13
# ---

"""The reflection standard (contract, mirror, GUI; both instrument modes).

Built the way the application builds it: a real sim station behind an
``InstrumentHost`` with a run catalog, the ``OrchestratorProxy`` a window is
handed, a ``Gateway`` attached to that proxy exactly as ``i2as.main`` attaches
the gateway server. The suite runs unchanged in either instrument mode
(``tests/instrument_modes.py``).
"""

from __future__ import annotations

import json

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QCheckBox, QComboBox, QLineEdit

from i2as.core import events as ev
from i2as.core.instrument_host import InstrumentHost
from i2as.core.plan import ParamSpec
from i2as.core.station import build_station
from i2as.core.status_mirror import StatusMirror
from i2as.gui import param_form
from i2as.gui.agent_panel import AgentAction, action_from_feed_record
from i2as.gui.instrument_panel import InstrumentPanel
from i2as.gui.monitor_window import MonitorWindow
from i2as.gui.procedure_window import ProcedureWindow
from i2as.procedures.field_sweep import FieldSweep
from i2as.session.agent_feed import SCHEMA_VERSION, AgentFeed
from i2as.session.gateway import Gateway, Role
from tests.instrument_modes import (
    JOIN_TIMEOUT_MS,
    instrument_mode,
    settled,
    shutdown_host,
    tick_engine,
)

CONFIG_PATH = "i2as/configs/sim_cryostat"

AGENT = ev.Actor(kind=ev.ActorKind.AGENT, id="runner-7", role="session")

#: A FieldSweep the sim station runs, with values distinct from the form's
#: defaults so a reflected form is provably the run's and not the draft's.
RUN_PARAMS = {
    "measurement_vi": "dc_measurement",
    "field_mode": "linear",
    "field_start": -0.2,
    "field_end": 0.2,
    "field_steps": 5,
    "field_hysteresis": True,
    "temperature": 250.0,
    "current_A": 2e-6,
    "readings_per_point": 3,
    "init_wait": 0.0,
    "step_wait": 0.0,
}

SAMPLE_INFO = {"sample_name": "S", "sample_id": "S-1", "comments": ""}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Redirect the app QSettings factory to a throwaway INI file."""
    from i2as.gui import app_settings

    ini_path = tmp_path / "i2as_test_settings.ini"
    monkeypatch.setattr(
        app_settings,
        "get_settings",
        lambda: QSettings(str(ini_path), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr(
        app_settings, "autosave_file_path", lambda user_id=None: tmp_path / "auto.json"
    )
    return ini_path


@pytest.fixture
def instrument_host(qtbot):
    """A started host over the sim station, with a run catalog, in this mode."""
    host = InstrumentHost(
        lambda: build_station(CONFIG_PATH),
        mode=instrument_mode(),
        orchestrator_options={
            "tick_interval_ms": 50,
            "run_catalog": {"FieldSweep": FieldSweep},
        },
        join_timeout_ms=JOIN_TIMEOUT_MS,
    )
    host.start()
    yield host
    shutdown_host(host)


@pytest.fixture
def station(instrument_host):
    return instrument_host.station


@pytest.fixture
def orchestrator(instrument_host):
    """The client adapter the windows are handed, as ``main.py`` hands it."""
    return instrument_host.build_proxy()


@pytest.fixture
def procedure_win(station, orchestrator, tmp_path, qtbot):
    """A ProcedureWindow whose runs write into the test's tmp directory."""
    win = ProcedureWindow(
        station,
        orchestrator,
        get_sample_info=lambda: dict(SAMPLE_INFO),
        get_data_dir=lambda: str(tmp_path),
    )
    qtbot.addWidget(win)
    win.show()
    return win


class _Recorder:
    """Collect everything one client said, in delivery order."""

    def __init__(self, client):
        self.verdicts: list[ev.Verdict] = []
        self.events: list = []
        client.verdict.connect(self.verdicts.append)
        client.event.connect(self.events.append)

    def of_type(self, event_type):
        return [event for event in self.events if isinstance(event, event_type)]

    def verdict_for(self, request_id):
        answers = [v for v in self.verdicts if v.request_id == request_id]
        assert len(answers) == 1, answers
        return answers[0]


def _run_command(data_dir, actor=AGENT, **overrides):
    args = {
        "procedure": "FieldSweep",
        "params": dict(RUN_PARAMS),
        "sample_info": dict(SAMPLE_INFO),
        "data_directory": str(data_dir),
        "file_prefix": "reflect",
    }
    args.update(overrides)
    return ev.Command(name=ev.CommandName.RUN_PROCEDURE, actor=actor, args=args)


def _start_run(orchestrator, data_dir, actor=AGENT):
    """Start a FieldSweep through the contract and wait for its RunStarted."""
    recorder = _Recorder(orchestrator)
    command = _run_command(data_dir, actor=actor)
    orchestrator.submit(command)
    settled(orchestrator, rounds=2)
    started = recorder.of_type(ev.RunStarted)
    assert len(started) == 1, recorder.events
    return command, started[0], recorder


def _end_run(orchestrator):
    orchestrator.abort_procedure()
    settled(orchestrator, rounds=3)


def _form_text(win, name):
    widget = win._params_panel._param_inputs[name]
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QCheckBox):
        return str(widget.isChecked())
    return param_form.get_widget_raw(widget)


# ── The contract: the answer echoes the question ──────────────────────────────


def test_a_verdict_echoes_the_arguments_of_the_command_it_answers(orchestrator):
    """The one message every client already receives now says what was asked."""
    recorder = _Recorder(orchestrator)
    command = ev.Command(
        name=ev.CommandName.SET_AGENT_GATE, args={"state": ev.AgentGate.ACTIVE.value}
    )
    orchestrator.submit(command)
    settled(orchestrator)

    verdict = recorder.verdict_for(command.request_id)
    assert verdict.ok
    assert verdict.args == {"state": "active"}
    assert verdict.actor == ev.OPERATOR


def test_a_deferred_instrument_action_verdict_still_carries_its_arguments(
    orchestrator,
):
    """The verdict a later tick emits echoes the command the tick long forgot."""
    recorder = _Recorder(orchestrator)
    command = ev.Command(
        name=ev.CommandName.SUBMIT_VI_ACTION,
        actor=AGENT,
        args={"vi_name": "magnet_z", "method_name": "initiate"},
    )
    orchestrator.submit(command)
    tick_engine(orchestrator, times=3)
    settled(orchestrator, rounds=2)

    verdict = recorder.verdict_for(command.request_id)
    assert verdict.args == {"vi_name": "magnet_z", "method_name": "initiate"}
    assert verdict.actor == AGENT


def test_a_gateway_refusal_echoes_the_arguments_too(orchestrator, station):
    """A refusal is read in full: what was asked for, not only that it was refused."""
    recorder = _Recorder(orchestrator)
    gateway = Gateway(orchestrator, Role.OBSERVER, "watcher", station_info=station.station_info)
    request_id = gateway.submit(
        ev.CommandName.SUBMIT_VI_ACTION,
        {"vi_name": "magnet_z", "method_name": "set_field", "target_T": 0.4},
    )
    settled(orchestrator)

    verdict = recorder.verdict_for(request_id)
    assert verdict.code is ev.VerdictCode.BLOCKED_ROLE
    assert verdict.args == {
        "vi_name": "magnet_z",
        "method_name": "set_field",
        "target_T": 0.4,
    }


def test_the_verdict_round_trips_its_arguments_through_json():
    """The contract's JSON round trip covers the new field like every other."""
    verdict = ev.Verdict(
        request_id="r-1",
        command=ev.CommandName.SUBMIT_VI_ACTION,
        code=ev.VerdictCode.OK,
        actor=AGENT,
        args={"vi_name": "magnet_z", "method_name": "set_field", "target_T": 1.5},
    )
    wire = json.loads(json.dumps(verdict.to_json()))
    assert ev.Verdict.from_json(wire) == verdict


# ── The run manifest and the mirror ───────────────────────────────────────────


def test_the_run_manifest_names_its_class_and_its_owner(orchestrator, tmp_path):
    """What ``run_procedure`` took, and whose run it is, travel on the event."""
    _command, started, _recorder = _start_run(orchestrator, tmp_path)
    try:
        assert started.manifest["procedure_class"] == "FieldSweep"
        assert started.manifest["procedure"] == FieldSweep.name
        assert started.manifest["owner"] == {"kind": "agent", "id": "runner-7"}
        assert started.manifest["params"]["field_start"] == -0.2
        assert started.manifest["params"]["measurement_vi"] == "dc_measurement"
    finally:
        _end_run(orchestrator)


def test_the_mirror_keeps_the_run_in_flight_and_clears_it_when_it_ends(qtbot):
    """A client built mid-run can still ask what exactly is running."""
    mirror = StatusMirror()
    seen: list = []
    mirror.run_manifest_updated.connect(seen.append)
    assert mirror.run_manifest() is None

    manifest = {"run_id": "r-1", "procedure_class": "FieldSweep", "params": {"a": 1}}
    mirror.on_event(ev.RunStarted(run_id="r-1", manifest=manifest))
    assert mirror.run_manifest() == manifest
    assert seen == [manifest]
    mirror.run_manifest()["params"]["a"] = 2  # a copy, never the mirror's own
    assert mirror.run_manifest()["params"]["a"] == 1

    mirror.on_event(ev.RunFinished(run_id="r-1", status="completed", manifest=manifest))
    assert mirror.run_manifest() is None
    assert seen[-1] is None


# ── The setters are the inverses of the collectors ────────────────────────────


@pytest.mark.parametrize(
    ("spec", "value"),
    [
        (ParamSpec(type=float, default=1.0), 2.5),
        (ParamSpec(type=int, default=1), 7),
        (ParamSpec(type=str, default="x"), "hello"),
        (ParamSpec(type=bool, default=False), True),
        (ParamSpec(type=str, default="a", choices={"A": "a", "B": "b"}), "b"),
        (ParamSpec(type=int, default=1, choices={"one": 1, "two": 2}), 2),
        (ParamSpec(type=str, default="", widget_hint="array"), "1e-6, -1e-6"),
    ],
)
def test_set_widget_value_is_the_inverse_of_collect_value(spec, value, qtbot):
    widget = param_form.build_param_widget("p", spec)
    qtbot.addWidget(widget)
    param_form.set_widget_value(widget, spec, value)
    assert param_form.collect_value(widget, spec) == value


def test_a_value_no_choice_maps_to_leaves_the_combobox_alone(qtbot):
    spec = ParamSpec(type=str, default="a", choices={"A": "a", "B": "b"})
    widget = param_form.build_param_widget("p", spec)
    qtbot.addWidget(widget)
    param_form.set_widget_value(widget, spec, "zzz")
    assert param_form.collect_value(widget, spec) == "a"


def test_a_table_widget_is_the_inverse_of_its_collector(qtbot):
    """A list ParamSpec renders as a table whose rows round-trip typed."""
    spec = ParamSpec(
        type=list,
        default=[],
        columns={
            "start": ParamSpec(type=float, default=0.0, unit="T"),
            "end": ParamSpec(type=float, default=1.0, unit="T"),
            "step": ParamSpec(type=float, default=0.1, unit="T"),
        },
    )
    widget = param_form.build_param_widget("field_segments", spec)
    qtbot.addWidget(widget)
    rows = [
        {"start": -1.0, "end": 0.0, "step": 0.5},
        {"start": 0.0, "end": 0.2, "step": 0.1},
    ]
    param_form.set_widget_value(widget, spec, rows)
    assert param_form.collect_value(widget, spec) == rows
    # The session cache carries the cells as JSON and restores them.
    raw = param_form.get_widget_raw(widget)
    param_form.set_widget_raw(widget, "")
    assert param_form.collect_value(widget, spec) == []
    param_form.set_widget_raw(widget, raw)
    assert param_form.collect_value(widget, spec) == rows
    # A new row continues the previous one.
    widget.add_row()
    assert param_form.collect_value(widget, spec)[-1] == {"start": 0.2, "end": 1.0, "step": 0.1}


def test_the_sweep_column_follows_the_mode_and_keeps_its_place(procedure_win):
    """The axis is guarded blocks: the mode picks the fields, in place."""
    panel = procedure_win._params_panel
    panel.select_procedure_by_name(FieldSweep.name)
    order_before = [
        panel._param_hbox.itemAt(i).widget().title()
        for i in range(panel._param_hbox.count())
        if panel._param_hbox.itemAt(i).widget() is not None
    ]
    assert order_before[0] == "Sweep"
    assert "field_start" in panel._param_inputs
    assert "field_segments" not in panel._param_inputs

    panel.apply_values({"field_mode": "segments"})

    assert "field_segments" in panel._param_inputs
    assert "field_start" not in panel._param_inputs
    order_after = [
        panel._param_hbox.itemAt(i).widget().title()
        for i in range(panel._param_hbox.count())
        if panel._param_hbox.itemAt(i).widget() is not None
    ]
    assert order_after == order_before
    assert panel.collect_values()["field_mode"] == "segments"
    # The segments table costs the form no width the linear range did not:
    # the table shrinks before the columns overflow. Measured on a settled
    # layout at a width where the linear form fits.
    from PyQt6.QtTest import QTest

    procedure_win.resize(1600, 800)
    QTest.qWait(150)
    segments_overflow = panel._param_scroll.horizontalScrollBar().maximum()
    panel.apply_values({"field_mode": "linear"})
    QTest.qWait(150)
    linear_overflow = panel._param_scroll.horizontalScrollBar().maximum()
    assert segments_overflow <= linear_overflow == 0


def test_a_segments_run_is_reflected_into_the_table(procedure_win):
    """A run's piecewise sweep lands in the form as rows, and collects back."""
    panel = procedure_win._params_panel
    panel.select_procedure_by_name(FieldSweep.name)
    segments = [
        {"start": -0.5, "end": 0.0, "step": 0.25},
        {"start": 0.0, "end": 0.1, "step": 0.05},
    ]
    panel.apply_values(
        {"field_mode": "segments", "field_segments": segments, "field_hysteresis": True}
    )
    collected = panel.collect_values()
    assert collected["field_mode"] == "segments"
    assert collected["field_segments"] == segments
    assert collected["field_hysteresis"] is True
    assert "field_csv_path" not in collected

    panel.apply_values({"field_mode": "csv", "field_csv_path": "C:/sweep.csv"})
    collected = panel.collect_values()
    assert collected["field_mode"] == "csv" and collected["field_csv_path"] == "C:/sweep.csv"
    assert "field_segments" not in collected


def test_apply_values_is_the_inverse_of_collect_values(procedure_win):
    """The declared form takes a run's values back, cascade included."""
    panel = procedure_win._params_panel
    panel.select_procedure_by_name(FieldSweep.name)
    values = {
        **RUN_PARAMS,
        "loop1_parameter": "dc_measurement.current_A",
        "loop1_values": "1e-6, -1e-6",
    }

    panel.apply_values(values)

    collected = panel.collect_values()
    assert collected is not None
    for name, value in values.items():
        assert collected[name] == value, name
    # The cascade settled: the loop slot's value field exists and was filled.
    assert "loop1_values" in panel._param_inputs
    assert panel.current_selections()["loop1_parameter"] == "dc_measurement.current_A"


# ── The procedure window reflects the run in flight, whoever started it ───────


def test_an_agent_started_run_is_shown_in_the_form(procedure_win, orchestrator, tmp_path):
    """The form holds the agent's parameters as if the operator had typed them."""
    panel = procedure_win._params_panel
    assert panel.reflection_text() == ""
    field_start_before = _form_text(procedure_win, "field_start") if "field_start" in panel._param_inputs else None
    assert field_start_before != "-0.2"

    _command, started, _recorder = _start_run(orchestrator, tmp_path)
    try:
        assert panel.current_class() is FieldSweep
        collected = panel.collect_values()
        assert collected["field_start"] == -0.2
        assert collected["field_end"] == 0.2
        assert collected["field_steps"] == 5
        assert collected["field_hysteresis"] is True
        assert collected["measurement_vi"] == "dc_measurement"
        assert collected["temperature"] == 250.0
        assert collected["current_A"] == 2e-6
        assert "runner-7" in panel.reflection_text()
        assert started.run_id in panel.reflection_text()
        assert panel._reflection_label.isVisible()
        log = procedure_win._status_log.toPlainText()
        assert "Run started by agent 'runner-7'" in log
    finally:
        _end_run(orchestrator)

    assert panel.reflection_text().startswith("Last run:")
    assert "aborted" in panel.reflection_text()


def test_a_window_opened_mid_run_shows_the_run_not_a_draft(
    station, orchestrator, tmp_path, qtbot
):
    """The mirror kept the manifest, so a late window reflects it at construction."""
    _command, _started, _recorder = _start_run(orchestrator, tmp_path)
    try:
        win = ProcedureWindow(
            station,
            orchestrator,
            get_sample_info=lambda: dict(SAMPLE_INFO),
            get_data_dir=lambda: str(tmp_path),
        )
        qtbot.addWidget(win)
        panel = win._params_panel
        assert panel.current_class() is FieldSweep
        assert panel.collect_values()["field_start"] == -0.2
        assert "runner-7" in panel.reflection_text()
    finally:
        _end_run(orchestrator)


def test_run_now_goes_through_the_one_door_as_a_json_command(
    procedure_win, orchestrator
):
    """The operator's own run is a Command answered by a Verdict, like an agent's."""
    recorder = _Recorder(orchestrator)
    panel = procedure_win._params_panel
    panel.select_procedure_by_name(FieldSweep.name)
    panel.apply_values({"init_wait": 0.0, "step_wait": 0.0, "field_steps": 3})

    procedure_win._on_run_now()
    settled(orchestrator, rounds=2)
    try:
        # Inline, the verdict lands inside _on_run_now() itself; threaded, a
        # turn later. Either way exactly one operator run_procedure verdict.
        (verdict,) = [
            v
            for v in recorder.verdicts
            if v.command is ev.CommandName.RUN_PROCEDURE
        ]
        request_id = verdict.request_id
        assert verdict.ok
        assert verdict.actor == ev.OPERATOR
        assert verdict.command is ev.CommandName.RUN_PROCEDURE
        assert verdict.args["procedure"] == "FieldSweep"
        assert verdict.args["params"]["field_steps"] == 3
        assert verdict.args["sample_info"] == SAMPLE_INFO
        (started,) = recorder.of_type(ev.RunStarted)
        assert started.request_id == request_id
        assert started.manifest["owner"] == {"kind": "operator", "id": "operator"}
        assert "operator" in panel.reflection_text()
        assert procedure_win._pending_run_request is None
    finally:
        _end_run(orchestrator)


def test_a_refused_run_now_is_shown_in_the_banner(procedure_win):
    """The window's own refusal arrives as a verdict and is shown, not lost."""
    procedure_win._pending_run_request = "req-x"
    procedure_win._on_verdict(
        ev.Verdict(
            request_id="req-x",
            command=ev.CommandName.RUN_PROCEDURE,
            code=ev.VerdictCode.FAILED,
            reason="unknown procedure 'Nope'",
        )
    )
    assert procedure_win._banner.isVisible()
    assert "unknown procedure" in procedure_win._banner._label.text()
    assert procedure_win._pending_run_request is None


# ── Instrument cards reflect accepted actions ─────────────────────────────────


def _set_field_verdict(code, target, actor=AGENT, vi_name="magnet_z"):
    return ev.Verdict(
        request_id="r-sf",
        command=ev.CommandName.SUBMIT_VI_ACTION,
        code=code,
        actor=actor,
        args={"vi_name": vi_name, "method_name": "set_field", "target_T": target},
    )


@pytest.fixture
def magnet_card(orchestrator, qtbot):
    card = InstrumentPanel(
        "magnet_z", orchestrator, orchestrator.status, panel_controls=["set_field"]
    )
    qtbot.addWidget(card)
    return card


def test_an_accepted_agent_setpoint_lands_in_the_card_s_own_field(magnet_card):
    field = magnet_card._control_inputs["set_field"]["target_T"]
    assert isinstance(field, QLineEdit)
    magnet_card.on_verdict(_set_field_verdict(ev.VerdictCode.OK, 0.42))
    assert field.text() == "0.42"


def test_a_refused_setpoint_and_another_instrument_s_leave_the_card_alone(magnet_card):
    field = magnet_card._control_inputs["set_field"]["target_T"]
    before = field.text()
    magnet_card.on_verdict(_set_field_verdict(ev.VerdictCode.BLOCKED_LIMIT, 9.0))
    magnet_card.on_verdict(_set_field_verdict(ev.VerdictCode.OK, 0.1, vi_name="magnet_y"))
    magnet_card.on_verdict(
        ev.Verdict(request_id="r", command=ev.CommandName.PAUSE_PROCEDURE, code=ev.VerdictCode.OK)
    )
    assert field.text() == before


def test_the_monitor_window_fans_verdicts_out_to_every_card(station, orchestrator, qtbot):
    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    card = next(p for p in win._panels if p.vi_name == "magnet_z")
    if "set_field" not in card._control_inputs:
        card._open_front_panel()
        card = card._front_panel.panel
    field = card._control_inputs["set_field"]["target_T"]

    win._on_verdict_for_agents(_set_field_verdict(ev.VerdictCode.OK, 0.31))

    assert field.text() == "0.31"


# ── The agent panel says what was asked ───────────────────────────────────────


def test_an_agent_row_carries_the_arguments_inline_and_in_full(station, orchestrator, qtbot):
    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    panel = win._agent_panel

    panel.on_verdict(_set_field_verdict(ev.VerdictCode.OK, 0.42))
    (row,) = panel.row_texts()
    assert "magnet_z.set_field(target_T=0.42) → OK" in row

    panel.on_verdict(
        ev.Verdict(
            request_id="r-run",
            command=ev.CommandName.RUN_PROCEDURE,
            code=ev.VerdictCode.OK,
            actor=AGENT,
            args={"procedure": "FieldSweep", "params": dict(RUN_PARAMS)},
        )
    )
    run_row = panel.row_texts()[-1]
    assert "run_procedure(procedure=FieldSweep, params.measurement_vi=dc_measurement" in run_row
    assert "more)" in run_row
    action = panel.actions()[-1]
    assert "params.field_start = -0.2" in action.args_detail()


def test_seeding_from_an_older_feed_joins_the_arguments_by_request_id(tmp_path):
    """A schema-2 file's verdict records have no args; the command record does."""
    path = tmp_path / "agent_actions.jsonl"
    command = {
        "schema": 2, "ts": 1.0, "seq": 1, "experiment_id": "e", "run_id": None,
        "record": "command", "actor": AGENT.to_json(), "request_id": "req-9",
        "command": "submit_vi_action", "tool": None,
        "args": {"vi_name": "magnet_z", "method_name": "set_field", "target_T": 0.5},
        "event": None, "detail": None, "verdict": None,
    }
    verdict = {
        **command, "seq": 2, "record": "verdict", "args": None,
        "verdict": {"code": "OK", "reason": ""},
    }
    path.write_text(
        json.dumps(command) + "\n" + json.dumps(verdict) + "\n", encoding="utf-8"
    )
    from i2as.session.agent_feed import read_feed

    records = read_feed(path)
    joined = {"req-9": command["args"]}
    rows = [action_from_feed_record(r, joined) for r in records]
    assert rows[0] is None  # a command record is not a row
    assert isinstance(rows[1], AgentAction)
    assert "target_T=0.5" in rows[1].text()


def test_a_recorded_verdict_carries_its_arguments(tmp_path):
    """Schema 3: the feed's verdict record echoes the arguments itself."""
    feed = AgentFeed(tmp_path / "agent_actions.jsonl", "exp")
    feed.record_verdict(_set_field_verdict(ev.VerdictCode.OK, 0.5))
    from i2as.session.agent_feed import read_feed

    (record,) = read_feed(feed.path)
    assert record["schema"] == SCHEMA_VERSION == 3
    assert record["args"] == {"vi_name": "magnet_z", "method_name": "set_field", "target_T": 0.5}


# ── The takeover strip says what is running ───────────────────────────────────


def test_the_strip_names_the_procedure_and_lists_its_parameters(station, orchestrator, qtbot):
    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    strip = win._takeover_strip
    manifest = {
        "run_id": "r-1",
        "procedure": "Field Sweep",
        "procedure_class": "FieldSweep",
        "params": {"field_start": -0.1, "field_end": 0.1},
        "owner": {"kind": "agent", "id": "agent-A"},
    }
    win._mirror.on_event(
        ev.StatusSnapshot(
            state="RAMPING",
            run={"id": "r-1", "name": "Field Sweep", "kind": "procedure",
                 "owner": {"kind": "agent", "id": "agent-A"}},
        )
    )
    win._mirror.on_event(ev.RunStarted(run_id="r-1", manifest=manifest))

    assert strip._run_owner_label.text() == "▶ Field Sweep, run owned by agent-A"
    assert "field_start = -0.1" in strip._run_owner_label.toolTip()

    win._mirror.on_event(ev.RunFinished(run_id="r-1", status="completed", manifest=manifest))
    win._mirror.on_event(ev.StatusSnapshot(state="IDLE"))
    assert strip._run_owner_label.text() == ""


# ── The live app attaches one feed per experiment ─────────────────────────────


def test_the_app_opens_one_attached_feed_per_experiment(orchestrator, tmp_path):
    """Two connections share one feed, and the feed hears the engine's answers."""
    from i2as.main import ExperimentFeeds
    from i2as.session.manager import ExperimentManager
    from i2as.session.models import User
    from i2as.session.store import ExperimentStore, UserRoster

    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    manager = ExperimentManager(
        store=ExperimentStore(tmp_path / "experiments"),
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
    )
    feeds = ExperimentFeeds(manager, orchestrator)
    assert feeds.current() is None
    manager.start_experiment("Reflect", "jdoe", dict(SAMPLE_INFO))

    first = feeds.current()
    second = feeds.current()
    assert first is second and first is not None

    command = ev.Command(
        name=ev.CommandName.SET_AGENT_GATE, actor=AGENT, args={"state": "active"}
    )
    orchestrator.submit(command)
    settled(orchestrator, rounds=2)
    from i2as.session.agent_feed import read_feed

    records = read_feed(first.path)
    verdicts = [r for r in records if r["record"] == "verdict"]
    assert len(verdicts) == 1
    assert verdicts[0]["request_id"] == command.request_id
    assert verdicts[0]["args"] == {"state": "active"}


# ── An agent reads the same numbers the front panel shows ─────────────────────


def test_read_readings_answers_the_front_panel_s_numbers(orchestrator, station):
    """The reading values travel on one event, and the tool reads that event."""
    gateway = Gateway(orchestrator, Role.OBSERVER, "watcher", station_info=station.station_info)
    assert "read_readings" in {tool.name for tool in gateway.tools()}
    assert gateway.call_tool("read_readings")["result"] is None  # nothing polled yet

    orchestrator.start_monitoring()
    tick_engine(orchestrator, times=2)
    settled(orchestrator, rounds=2)

    answer = gateway.call_tool("read_readings")["result"]
    assert answer is not None and answer["values"]
    magnet = answer["values"]["magnet_z"]
    info = next(i for i in station.station_info().instruments if i.name == "magnet_z")
    declared = {reading.name for reading in info.monitored}
    assert declared & set(magnet), (declared, set(magnet))
    assert answer["seq"] > 0


def test_the_readings_resource_is_the_readings_tool():
    from i2as.mcp import READINGS_URI, RESOURCE_TOOLS, RESOURCES

    assert RESOURCE_TOOLS[READINGS_URI] == "read_readings"
    assert any(resource["uri"] == READINGS_URI for resource in RESOURCES)


# ── The table kind reaches the agent's surface and the validator ──────────────


def test_describe_procedure_declares_the_segments_table(orchestrator, station):
    """An agent reads the sweep as data: the mode's choices and the table's columns."""
    from i2as.core.plan import blocks_from_json, resolve_form
    from i2as.core.procedure_catalog import build_procedure_infos

    info = next(
        p
        for p in build_procedure_infos(station, {"FieldSweep": FieldSweep})
        if p.class_name == "FieldSweep"
    )
    blocks = blocks_from_json([block.to_json() for block in info.form])
    groups = {g.key: g for g in resolve_form(blocks, {"field_mode": "segments"})}
    sweep = groups["sweep"].params
    assert sweep["field_mode"].structural
    assert sweep["field_segments"].type is list
    assert list(sweep["field_segments"].columns) == ["start", "end", "step"]
    assert sweep["field_segments"].columns["start"].unit == "T"
    assert "field_start" not in sweep


def test_validate_run_refuses_a_malformed_segments_table(station):
    from i2as.session.run_queue import FINDING_PARAM_BOUNDS, validate_run

    result = validate_run(
        FieldSweep,
        {**RUN_PARAMS, "field_mode": "segments", "field_segments": [{"start": 0.0}]},
        station=station,
        sample_info=dict(SAMPLE_INFO),
        data_directory="C:/nowhere",
    )
    assert not result.ok
    assert any(
        f.code == FINDING_PARAM_BOUNDS and f.param == "field_segments"
        for f in result.findings
    )
