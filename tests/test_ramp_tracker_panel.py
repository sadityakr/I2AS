"""Behavior tests for RampTrackerPanel / RampRow."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QMessageBox, QScrollArea

from i2as.core.orchestrator import Orchestrator
from i2as.core.ramps import ACTIVE_RAMP_STATUS, RampRecord
from i2as.core.station import build_station
from i2as.gui import ramp_tracker_panel as ramp_tracker_module
from i2as.gui.monitor_window import MonitorWindow
from i2as.gui.ramp_tracker_panel import RampRow, RampTrackerPanel

CONFIG_PATH = "i2as/configs/sim_cryostat"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Redirect the app QSettings factory to a throwaway INI file.

    Same dependency seam as ``test_gui.py``'s fixture: MonitorWindow persists
    splitter state (including this quadrant's new bottom-right splitter), and
    a pytest run must never read or overwrite the user's real saved layout.
    """
    from i2as.gui import app_settings

    ini_path = tmp_path / "i2as_test_settings.ini"
    monkeypatch.setattr(
        app_settings,
        "get_settings",
        lambda: QSettings(str(ini_path), QSettings.Format.IniFormat),
    )
    return ini_path


@pytest.fixture
def station():
    return build_station(CONFIG_PATH)


@pytest.fixture
def orchestrator(station, qtbot):
    orch = Orchestrator(station, tick_interval_ms=50)
    yield orch
    orch.shutdown()


@pytest.fixture
def panel(orchestrator, qtbot):
    widget = RampTrackerPanel(orchestrator)
    qtbot.addWidget(widget)
    widget.show()
    return widget


def _record(vi_name: str = "magnet_z", **overrides) -> RampRecord:
    """A ramping-magnet record, with per-field overrides."""
    fields = {
        "vi_name": vi_name,
        "label": "field",
        "unit": "T",
        "value": 0.25,
        "setpoint": 4.0,
        "target": 5.0,
        "rate": 0.5,
        "phase": None,
        "owner": None,
        "stoppable": True,
        "stop_blocked_reason": "",
        "stale": False,
    }
    fields.update(overrides)
    return RampRecord(**fields)


def _fully_inside(viewport, widget) -> bool:
    """Return True if *widget* is visible AND fully inside *viewport*'s width.

    Mirrors ``test_gui.py``'s ``_fully_inside_param_viewport`` idiom
    (horizontal-only): a QScrollArea legitimately clips content vertically,
    but a widget pushed off-screen to the *right* — the bug class this guards
    against — never should be.
    """
    if not widget.isVisible():
        return False
    top_left = widget.mapTo(viewport, widget.rect().topLeft())
    return top_left.x() >= 0 and top_left.x() + widget.width() <= viewport.width()


# ── Empty state ───────────────────────────────────────────────────────────────


def test_empty_state_shown_until_something_ramps(panel):
    assert panel.row_names() == []
    assert panel.findChild(type(panel._empty_label), "ramp_tracker_empty_label").isVisible()

    panel.on_ramps_updated([_record()])
    assert not panel._empty_label.isVisible()

    panel.on_ramps_updated([])
    assert panel._empty_label.isVisible()


# ── Row content: rate, next setpoint, end setpoint ────────────────────────────


def test_row_shows_value_next_setpoint_end_setpoint_and_rate(panel):
    """The three numbers the operator asked for, plus where the ramp is now."""
    panel.on_ramps_updated([_record()])
    row = panel._rows["magnet_z"]

    assert row.findChild(type(row._progress_label), "ramp_magnet_z_progress_label")
    assert "0.25 T" in row._progress_label.text()
    assert "4 T" in row._progress_label.text()  # the NEXT setpoint
    assert "End setpoint 5 T" in row._detail_label.text()
    assert "0.5 T/min" in row._detail_label.text()


def test_row_title_names_both_the_quantity_and_the_instrument(panel):
    """Two magnets both ramp "field"; only the VI name says which coil moves."""
    panel.on_ramps_updated([_record()])
    assert panel._rows["magnet_z"].title() == "field · magnet_z"


def test_row_renders_a_vi_with_no_introspection_hooks(panel):
    """A rampable VI exposing nothing still gets a working row, never a crash."""
    panel.on_ramps_updated(
        [
            _record(
                vi_name="rotator",
                label="rotator",
                unit="",
                value=None,
                setpoint=None,
                target=None,
                rate=None,
            )
        ]
    )
    row = panel._rows["rotator"]
    assert row.title() == "rotator"
    assert row._progress_label.text() == "— → —"
    assert "rate unknown" in row._detail_label.text()


def test_row_shows_the_ramp_sub_phase_when_the_vi_has_one(panel):
    panel.on_ramps_updated([_record(phase="warmup")])
    assert "warmup" in panel._rows["magnet_z"]._detail_label.text()


def test_row_flags_a_stale_reading(panel):
    panel.on_ramps_updated([_record(stale=True)])
    assert "reading stale" in panel._rows["magnet_z"]._detail_label.text()


# ── Reconciliation across ticks ───────────────────────────────────────────────


def test_rows_are_updated_in_place_not_rebuilt(panel):
    """A rebuild every tick would make the Abort button unclickable."""
    panel.on_ramps_updated([_record()])
    row = panel._rows["magnet_z"]

    panel.on_ramps_updated([_record(value=1.5, setpoint=5.0)])
    assert panel._rows["magnet_z"] is row  # same widget object
    assert "1.5 T" in row._progress_label.text()


def test_finished_ramp_loses_its_row(panel):
    panel.on_ramps_updated([_record(), _record(vi_name="magnet_y")])
    assert panel.row_names() == ["magnet_z", "magnet_y"]

    panel.on_ramps_updated([_record(vi_name="magnet_y")])
    assert panel.row_names() == ["magnet_y"]


def test_rows_follow_the_records_order(panel):
    panel.on_ramps_updated(
        [_record(vi_name="magnet_y"), _record(vi_name="temperature_sample")]
    )
    assert panel.row_names() == ["magnet_y", "temperature_sample"]


# ── Abort button ──────────────────────────────────────────────────────────────


def test_abort_confirms_then_stops_only_that_ramp(panel, orchestrator, monkeypatch):
    called = []
    monkeypatch.setattr(orchestrator, "stop_ramp", lambda name: called.append(name))
    monkeypatch.setattr(
        ramp_tracker_module.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )

    panel.on_ramps_updated([_record()])
    panel._rows["magnet_z"]._abort_btn.click()
    assert called == ["magnet_z"]


def test_abort_declined_does_nothing(panel, orchestrator, monkeypatch):
    called = []
    monkeypatch.setattr(orchestrator, "stop_ramp", lambda name: called.append(name))
    monkeypatch.setattr(
        ramp_tracker_module.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )

    panel.on_ramps_updated([_record()])
    panel._rows["magnet_z"]._abort_btn.click()
    assert called == []


def test_owned_ramp_shows_its_run_and_disables_abort(panel):
    """The record's own verdict drives the button — the panel decides nothing."""
    panel.on_ramps_updated(
        [
            _record(
                owner="procedure 'Field Sweep'",
                stoppable=False,
                stop_blocked_reason="Cannot control magnet_z: claimed by running "
                "procedure 'Field Sweep'",
            )
        ]
    )
    row = panel._rows["magnet_z"]
    # isHidden(), not isVisible(): this asserts the panel's own show/hide
    # decision, without depending on when Qt gets round to realising a child
    # just inserted into an already-shown layout.
    assert not row._owner_label.isHidden()
    assert "Field Sweep" in row._owner_label.text()
    assert not row._abort_btn.isEnabled()
    assert "claimed by running" in row._abort_btn.toolTip()


def test_manual_ramp_hides_the_owner_line_and_enables_abort(panel):
    panel.on_ramps_updated([_record()])
    row = panel._rows["magnet_z"]
    assert row._owner_label.isHidden()
    assert row._abort_btn.isEnabled()


def test_a_row_that_becomes_owned_updates_its_button(panel):
    """A manual ramp adopted by a starting run flips to refused in place."""
    panel.on_ramps_updated([_record()])
    row = panel._rows["magnet_z"]
    assert row._abort_btn.isEnabled()

    panel.on_ramps_updated(
        [_record(owner="procedure 'Field Sweep'", stoppable=False, stop_blocked_reason="no")]
    )
    assert panel._rows["magnet_z"] is row
    assert not row._abort_btn.isEnabled()


# ── MonitorWindow integration ─────────────────────────────────────────────────


def test_monitor_window_hosts_the_ramps_subpanel_over_the_agents_one(
    station, orchestrator, qtbot
):
    """The bottom-right quadrant is a vertical splitter: Ramps over Agents."""
    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    win.resize(1280, 900)
    win.show()
    qtbot.waitExposed(win)

    splitter = win._agents_splitter
    assert splitter.count() == 2
    assert splitter.widget(0).objectName() == "ramps_quadrant"
    assert splitter.widget(1).objectName() == "agents_quadrant"
    assert win._ramp_tracker is not None


def test_ramp_rows_are_fully_on_screen_at_a_realistic_width(
    station, orchestrator, qtbot
):
    """A row (Abort button included) must never be laid out off to the right."""
    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    win.resize(1280, 900)
    win.show()
    qtbot.waitExposed(win)

    # A REAL ramp, not an injected record: the orchestrator's live tick
    # republishes the (empty) truth every 50 ms and would otherwise delete an
    # injected row out from under the assertions below.
    orchestrator.start_monitoring()
    station.get_vi("magnet_z").set_field(5.0)
    with qtbot.waitSignal(orchestrator.ramps_updated, timeout=2000):
        pass
    # The row exists the moment the signal lands, but Qt lays it out on the
    # next event-loop pass — and geometry, not existence, is what this test
    # is about. Subsequent ticks update the same row in place, so waiting
    # cannot make it disappear.
    qtbot.waitUntil(lambda: win.findChild(RampRow, "ramp_row_magnet_z") is not None)
    row = win.findChild(RampRow, "ramp_row_magnet_z")
    qtbot.waitUntil(row.isVisible)

    scroll = win.findChild(QScrollArea, "ramps_scroll")
    assert scroll is not None
    assert scroll.horizontalScrollBar().maximum() == 0
    assert _fully_inside(scroll.viewport(), row)
    abort_btn = win.findChild(type(row._abort_btn), "ramp_magnet_z_abort_btn")
    assert _fully_inside(scroll.viewport(), abort_btn)


def test_monitor_window_forwards_ramps_updated_to_the_panel(
    station, orchestrator, qtbot
):
    """The window is the receiver (destruction-order rule); the panel is fed by it."""
    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)

    orchestrator.ramps_updated.emit([_record()])
    assert win._ramp_tracker.row_names() == ["magnet_z"]


def test_live_manual_ramp_reaches_the_panel_through_a_real_tick(
    station, orchestrator, qtbot
):
    """End to end: a ramp started on the instrument card appears in the tracker."""
    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    orchestrator.start_monitoring()

    station.get_vi("magnet_z").set_field(5.0)
    assert station.get_vi("magnet_z").ramp_status() == ACTIVE_RAMP_STATUS
    with qtbot.waitSignal(orchestrator.ramps_updated, timeout=2000):
        pass

    assert "magnet_z" in win._ramp_tracker.row_names()
    row = win._ramp_tracker._rows["magnet_z"]
    assert "End setpoint 5 T" in row._detail_label.text()
