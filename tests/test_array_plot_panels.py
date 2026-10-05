"""GUI side of the monitored-kind standard: array history, image/waterfall panels,
the kind-aware plot quadrant, and the Procedure window's XY/Image plot panel."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from PyQt6.QtCore import QSettings

from i2as.core.events import InstrumentInfo, MonitoredInfo, StationInfo
from i2as.core.station import build_station
from i2as.gui.array_plot_panels import ImagePlotPanel, WaterfallPlotPanel, waterfall_image
from i2as.gui.live_plot_panel import LivePlotPanel
from i2as.gui.monitor_history import ArrayHistory
from i2as.gui.trends_quadrant import TrendsQuadrant, array_fields_from_station_info

CONFIG_PATH = "i2as/configs/sim_cryostat"

IMAGE_INFO = MonitoredInfo(
    name="last_frame", unit="counts", description="A frame", kind="image",
    shape=(4, 3), period_s=1.0,
)
TRACE_INFO = MonitoredInfo(
    name="spectrum", unit="V", description="A spectrum", kind="trace",
    shape=(5,), period_s=1.0, axis=(400.0, 800.0, "nm"),
)
FIELDS = {"camera_last_frame": IMAGE_INFO, "spec_spectrum": TRACE_INFO}


@pytest.fixture
def station():
    return build_station(CONFIG_PATH)


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    from i2as.gui import app_settings

    ini_path = tmp_path / "settings.ini"
    monkeypatch.setattr(
        app_settings, "get_settings", lambda: QSettings(str(ini_path), QSettings.Format.IniFormat)
    )
    return ini_path


# ----------------------------------------------------------------------
# ArrayHistory
# ----------------------------------------------------------------------


class TestWaterfallImage:
    def test_a_gap_stays_empty_and_rows_are_one_period(self):
        now = 100.0
        entries = [(40.0, np.zeros(5)), (41.0, np.ones(5)), (99.0, np.full(5, 2.0))]
        image, rect = waterfall_image(
            entries, now=now, window_s=60.0, period_s=1.0, max_rows=600, axis=None
        )
        assert image.shape == (60, 5)
        assert image[0, 0] == 0.0 and image[1, 0] == 1.0 and image[59, 0] == 2.0
        assert np.isnan(image[2:59]).all()  # the run in between is a visible gap
        assert rect == (-0.5, -60.0, 5.0, 60.0)

    def test_pixels_are_centred_on_the_declared_axis(self):
        _image, rect = waterfall_image(
            [(0.0, np.zeros(5))], now=1.0, window_s=60.0, period_s=1.0,
            max_rows=600, axis=(400.0, 800.0, "nm"),
        )
        x, _y, width, _height = rect
        assert x == pytest.approx(350.0)  # 400 - dx/2, dx = 100
        assert width == pytest.approx(500.0)

    def test_rows_coarsen_to_the_history_cap(self):
        image, _rect = waterfall_image(
            [(0.0, np.zeros(3))], now=1.0, window_s=3600.0, period_s=1.0,
            max_rows=600, axis=None,
        )
        assert image.shape == (600, 3)


class TestArrayHistory:
    def test_an_image_keeps_its_newest_frame_only(self):
        history = ArrayHistory()
        for t in range(5):
            history.record({"camera": {"last_frame": np.full((4, 4), float(t))}}, timestamp=t)
        entries = history.window("camera_last_frame", 100.0, now=5.0)
        assert len(entries) == 1 and entries[0][1][0, 0] == 4.0
        assert history.nbytes == 4 * 4 * 8

    def test_a_byte_budget_evicts_the_oldest_trace_rows_across_keys(self):
        row = np.zeros(100)  # 800 bytes
        history = ArrayHistory(max_entries=600, max_bytes=8 * 800)
        for t in range(10):
            history.record({"a": {"x": row.copy()}, "b": {"y": row.copy()}}, timestamp=float(t))
        assert history.nbytes <= 8 * 800
        a = history.window("a_x", 100.0, now=10.0)
        b = history.window("b_y", 100.0, now=10.0)
        assert a[-1][0] == 9.0 and b[-1][0] == 9.0  # the newest always survive
        assert len(a) + len(b) == 8

    def test_keys_are_flattened_like_the_scalar_history(self):
        history = ArrayHistory()
        history.record({"camera": {"last_frame": np.zeros((2, 2))}}, timestamp=1.0)
        assert history.keys() == ["camera_last_frame"]
        timestamp, frame = history.latest("camera_last_frame")
        assert timestamp == 1.0
        assert frame.shape == (2, 2)
        assert history.latest("missing") is None

    def test_each_key_keeps_only_its_newest_entries(self):
        history = ArrayHistory(max_entries=3)
        for t in range(5):
            history.record({"spec": {"spectrum": np.full(5, float(t))}}, timestamp=float(t))
        entries = history.window("spec_spectrum", window_s=100.0, now=4.0)
        assert [t for t, _ in entries] == [2.0, 3.0, 4.0]

    def test_window_reaches_back_only_so_far(self):
        history = ArrayHistory()
        for t in (0.0, 50.0, 90.0):
            history.record({"spec": {"spectrum": np.zeros(5)}}, timestamp=t)
        assert [t for t, _ in history.window("spec_spectrum", 45.0, now=100.0)] == [90.0]

    def test_a_cap_below_one_is_refused(self):
        with pytest.raises(ValueError):
            ArrayHistory(max_entries=0)


def test_array_fields_come_from_the_declaration():
    info = StationInfo(
        instruments=(
            InstrumentInfo(
                name="camera",
                monitored=(MonitoredInfo(name="exposure", unit="s"), IMAGE_INFO),
            ),
        )
    )
    assert array_fields_from_station_info(info) == {"camera_last_frame": IMAGE_INFO}


# ----------------------------------------------------------------------
# Monitor panels
# ----------------------------------------------------------------------


class TestImagePlotPanel:
    def test_offers_only_image_fields_and_draws_the_newest_frame(self, qtbot):
        history = ArrayHistory()
        panel = ImagePlotPanel(history, "image_0", FIELDS)
        qtbot.addWidget(panel)
        assert panel.selected_key() == "camera_last_frame"
        assert panel.set_selected_key("spec_spectrum") is None  # not offered
        assert panel.selected_key() == "camera_last_frame"
        assert panel._view.frame() is None

        history.record({"camera": {"last_frame": np.zeros((4, 3))}}, timestamp=1.0)
        history.record({"camera": {"last_frame": np.ones((4, 3))}}, timestamp=2.0)
        panel.refresh()
        np.testing.assert_array_equal(panel._view.frame(), np.ones((4, 3)))

    def test_settings_entry_round_trips(self, qtbot):
        panel = ImagePlotPanel(ArrayHistory(), "image_0", FIELDS)
        qtbot.addWidget(panel)
        assert panel.settings_entry() == {"kind": "image", "key": "camera_last_frame"}

    def test_a_station_without_image_fields_says_so(self, qtbot):
        panel = ImagePlotPanel(ArrayHistory(), "image_0", {})
        qtbot.addWidget(panel)
        assert panel.selected_key() is None
        assert "No image field" in panel._status.text()


class TestWaterfallPlotPanel:
    def test_traces_land_on_the_real_time_axis_newest_last(self, qtbot):
        history = ArrayHistory()
        now = time.time()
        for age, level in ((30.0, 0.0), (20.0, 1.0), (10.0, 2.0)):
            history.record({"spec": {"spectrum": np.full(5, level)}}, timestamp=now - age)
        panel = WaterfallPlotPanel(history, "waterfall_0", FIELDS)
        qtbot.addWidget(panel)
        panel.refresh()

        stack = panel._view.frame()
        assert stack.shape[1] == 5
        filled = stack[~np.isnan(stack[:, 0]), 0]
        np.testing.assert_array_equal(filled, [0.0, 1.0, 2.0])

    def test_windows_are_limited_to_what_the_history_can_hold(self, qtbot):
        panel = WaterfallPlotPanel(ArrayHistory(max_entries=120), "waterfall_0", FIELDS)
        qtbot.addWidget(panel)
        # period 1 s x 120 traces = 2 min: only "1 min" can be filled.
        items = [panel._window_selector.itemText(i) for i in range(panel._window_selector.count())]
        assert items == ["1 min"]

    def test_settings_entry_carries_the_window(self, qtbot):
        panel = WaterfallPlotPanel(ArrayHistory(), "waterfall_0", FIELDS)
        qtbot.addWidget(panel)
        panel.apply_settings_entry({"kind": "waterfall", "key": "spec_spectrum", "window_s": 60.0})
        assert panel.settings_entry() == {
            "kind": "waterfall", "key": "spec_spectrum", "window_s": 60.0,
        }
        # A window the history cannot fill is not restored.
        panel.apply_settings_entry({"kind": "waterfall", "key": "spec_spectrum", "window_s": 3600.0})
        assert panel.selected_window_s() == 60.0


# ----------------------------------------------------------------------
# The kind-aware quadrant
# ----------------------------------------------------------------------


class TestPlotQuadrant:
    def test_array_kinds_are_offered_only_when_declared(self, qtbot, station, tmp_path):
        plain = TrendsQuadrant(station, log_dir=tmp_path)
        qtbot.addWidget(plain)
        assert plain._add_buttons["image"].isHidden()
        assert plain._add_buttons["waterfall"].isHidden()
        plain._on_add_clicked("image")
        assert all(p.kind == "trend" for p in plain.panels().values())

        rich = TrendsQuadrant(station, log_dir=tmp_path, array_fields=FIELDS)
        qtbot.addWidget(rich)
        assert not rich._add_buttons["image"].isHidden()
        assert not rich._add_buttons["waterfall"].isHidden()

    def test_panels_of_every_kind_share_the_grid_and_the_cap(self, qtbot, station, tmp_path):
        quadrant = TrendsQuadrant(station, log_dir=tmp_path, array_fields=FIELDS)
        qtbot.addWidget(quadrant)
        quadrant._add_buttons["image"].click()
        quadrant._add_buttons["waterfall"].click()
        kinds = [panel.kind for panel in quadrant.panels().values()]
        assert kinds == ["trend", "trend", "image", "waterfall"]
        assert list(quadrant.panels()) == ["trend_0", "trend_1", "image_2", "waterfall_3"]
        assert not any(button.isEnabled() for button in quadrant._add_buttons.values())

    def test_arrays_reach_the_array_panels(self, qtbot, station, tmp_path):
        quadrant = TrendsQuadrant(station, log_dir=tmp_path, array_fields=FIELDS)
        qtbot.addWidget(quadrant)
        panel_id = quadrant._add_panel("image")
        quadrant.on_arrays_updated({"camera": {"last_frame": np.full((4, 3), 7.0)}})
        frame = quadrant.panels()[panel_id]._view.frame()
        np.testing.assert_array_equal(frame, np.full((4, 3), 7.0))

    def test_layout_with_kinds_round_trips(self, qtbot, station, tmp_path, settings_file):
        first = TrendsQuadrant(station, log_dir=tmp_path, array_fields=FIELDS)
        qtbot.addWidget(first)
        first._add_panel("waterfall")
        first.save_settings()
        saved = json.loads(QSettings(str(settings_file), QSettings.Format.IniFormat).value(
            "MonitorWindow/trends"
        ))
        assert [entry["kind"] for entry in saved] == ["trend", "trend", "waterfall"]

        second = TrendsQuadrant(station, log_dir=tmp_path, array_fields=FIELDS)
        qtbot.addWidget(second)
        second.restore_settings()
        assert [p.kind for p in second.panels().values()] == ["trend", "trend", "waterfall"]

    def test_a_legacy_layout_restores_as_trends_and_skips_what_cannot_draw(
        self, qtbot, station, tmp_path, settings_file
    ):
        QSettings(str(settings_file), QSettings.Format.IniFormat).setValue(
            "MonitorWindow/trends",
            json.dumps([
                {"key": "a", "window_s": 3600.0},  # saved before panel kinds
                {"kind": "image", "key": "camera_last_frame"},  # no camera here
                {"kind": "hologram", "key": "x"},  # unknown kind
            ]),
        )
        quadrant = TrendsQuadrant(station, log_dir=tmp_path)
        qtbot.addWidget(quadrant)
        quadrant.restore_settings()
        assert [p.kind for p in quadrant.panels().values()] == ["trend"]


# ----------------------------------------------------------------------
# Procedure window: XY / Image plot panel
# ----------------------------------------------------------------------


def _panel(qtbot) -> LivePlotPanel:
    panel = LivePlotPanel(
        "Plot 1", "#336699",
        x_selector_name="x", y_selector_name="y", plot_object_name="live_plot",
        loop1_selector_name="l1", loop2_selector_name="l2",
    )
    qtbot.addWidget(panel)
    panel.show()
    return panel


class TestLivePlotPanelKinds:
    def test_image_kind_needs_an_image_block(self, qtbot):
        panel = _panel(qtbot)
        assert panel.selected_kind() == "xy"
        panel.set_kind("image")
        assert panel.selected_kind() == "xy"  # nothing to draw yet

        panel.set_available_image_blocks({"frame": "counts"})
        panel.set_kind("image")
        assert panel.selected_kind() == "image"
        assert panel._image_selector.isVisible()
        assert not panel._x_selector.isVisible()

        panel.set_available_image_blocks({})  # a procedure that records no frames
        assert panel.selected_kind() == "xy"

    def test_draws_the_latest_or_the_picked_point(self, qtbot):
        panel = _panel(qtbot)
        panel.set_available_image_blocks({"frame": "counts"})
        panel.set_kind("image")
        points = [{"frame": np.full((3, 2), float(k))} for k in range(3)]
        panel.redraw(points)
        np.testing.assert_array_equal(panel._image_view.frame(), np.full((3, 2), 2.0))

        panel._point_selector.setValue(1)  # the first point
        np.testing.assert_array_equal(panel._image_view.frame(), np.full((3, 2), 0.0))

    def test_loop_selectors_pick_the_reading(self, qtbot):
        panel = _panel(qtbot)
        panel.set_available_image_blocks({"frame": "counts"})
        panel.set_kind("image")
        panel.set_available_loop_labels(({0: "A1", 1: "A2"}, None))
        grid = [[np.full((2, 2), 10.0)], [np.full((2, 2), 20.0)]]  # (n_loop1=2, n_loop2=1)
        panel.redraw([{"frame": grid}])
        np.testing.assert_array_equal(panel._image_view.frame(), np.full((2, 2), 10.0))
        panel._loop_selectors[0].setCurrentIndex(1)
        np.testing.assert_array_equal(panel._image_view.frame(), np.full((2, 2), 20.0))

    def test_clear_empties_the_image(self, qtbot):
        panel = _panel(qtbot)
        panel.set_available_image_blocks({"frame": "counts"})
        panel.set_kind("image")
        panel.redraw([{"frame": np.zeros((2, 2))}])
        panel.clear()
        assert panel._image_view.frame() is None


def test_every_registered_kind_builds_a_protocol_panel(qtbot, station, tmp_path):
    """The registry is the extension point: each entry's factory yields a PlotPanel."""
    from i2as.gui.plot_panel import PANEL_KINDS, PlotPanel

    quadrant = TrendsQuadrant(station, log_dir=tmp_path, array_fields=FIELDS)
    qtbot.addWidget(quadrant)
    for kind in PANEL_KINDS:
        panel_id, panel = quadrant._create_panel(kind)
        assert isinstance(panel, PlotPanel)
        assert panel.kind == kind and panel.panel_id == panel_id
        assert panel.settings_entry()["kind"] == kind
