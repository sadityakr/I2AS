"""Tests for TrendPlotPanel — reusable time-series trend plot panel."""

import json
import time
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QComboBox, QPushButton

from i2as.gui.monitor_history import MonitorHistory
from i2as.gui.trend_plot_panel import TIME_WINDOWS, TrendPlotPanel


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write one JSONL record per line to ``path`` (matches trend_history's on-disk format)."""
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _bucket_record(t: float, key_stats: dict[str, dict]) -> dict:
    """One aggregate-tier (3min/hourly) JSONL record."""
    return {"t": t, "n": sum(s["count"] for s in key_stats.values()), "v": key_stats}


def _stats(min_v: float, max_v: float, mean: float, std: float, count: int) -> dict:
    return {"min": min_v, "max": max_v, "mean": mean, "std": std, "count": count}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def history():
    """Empty MonitorHistory with generous retention for synthetic test data."""
    return MonitorHistory(retention_s=90000.0, tick_interval_s=1.0)


def _record_series(hist: MonitorHistory, key_state: dict, timestamp: float) -> None:
    """Record one snapshot into ``hist`` at an explicit timestamp.

    Args:
        hist: The MonitorHistory to record into.
        key_state: Nested ``{vi_name: {field_name: value}}`` state dict.
        timestamp: Unix timestamp to record the point under.
    """
    hist.record(key_state, timestamp=timestamp)


# ── Construction ──────────────────────────────────────────────────────────────

def test_construction_objectnames_and_default_window(qtbot, history):
    """Combos and remove button exist with expected objectNames; window default is 1 h."""
    panel = TrendPlotPanel(history, panel_id="p0")
    qtbot.addWidget(panel)

    y_selector = panel.findChild(QComboBox, "trend_y_selector_p0")
    window_selector = panel.findChild(QComboBox, "trend_window_selector_p0")
    remove_button = panel.findChild(QPushButton, "trend_remove_button_p0")

    assert y_selector is not None
    assert window_selector is not None
    assert remove_button is not None
    assert window_selector.currentText() == "1 h"


# ── refresh() population and redraw ──────────────────────────────────────────

def test_refresh_populates_y_combo_and_draws_selected_series(qtbot, history):
    """refresh() after recording data populates the Y combo and draws the curve."""
    now = time.time()
    for i in range(5):
        _record_series(
            history,
            {"magnet_z": {"field_T": float(i)}, "temp_a": {"temperature_K": 4.0 + i}},
            timestamp=now - (4 - i) * 10,
        )

    panel = TrendPlotPanel(history, panel_id="p1")
    qtbot.addWidget(panel)
    panel.refresh()

    y_selector = panel.findChild(QComboBox, "trend_y_selector_p1")
    combo_items = [y_selector.itemText(i) for i in range(y_selector.count())]
    assert combo_items == history.keys()
    assert combo_items == ["magnet_z_field_T", "temp_a_temperature_K"]

    # Panel selects the first key by default when nothing was chosen yet.
    assert panel.selected_key() == "magnet_z_field_T"

    expected_times, expected_values = history.series(
        "magnet_z_field_T", window_s=panel.selected_window_s()
    )
    x_data, y_data = panel._curve.getData()
    assert list(x_data) == expected_times
    assert list(y_data) == expected_values
    assert len(x_data) == 5


def test_refresh_with_no_keys_leaves_curve_empty(qtbot, history):
    """refresh() on an empty history clears the curve and leaves the combo empty."""
    panel = TrendPlotPanel(history, panel_id="p_empty")
    qtbot.addWidget(panel)
    panel.refresh()

    assert panel.selected_key() is None
    x_data, y_data = panel._curve.getData()
    assert x_data is None or len(x_data) == 0
    assert y_data is None or len(y_data) == 0


# ── Time-window filtering ────────────────────────────────────────────────────

def test_window_filtering_changes_plotted_point_count(qtbot, history):
    """Points spread over a long range; the 15 min window selects a known subset."""
    now = time.time()
    # 10 points: 5 within the last 15 minutes, 5 from 2-6 hours ago.
    recent_offsets = [0, 60, 120, 300, 600]           # within last 15 min (900s)
    old_offsets = [7200, 10800, 14400, 18000, 21600]  # 2h-6h ago

    for offset in old_offsets + recent_offsets:
        _record_series(
            history, {"magnet_z": {"field_T": 1.0}}, timestamp=now - offset
        )

    panel = TrendPlotPanel(history, panel_id="p2")
    qtbot.addWidget(panel)
    panel.refresh()
    panel.set_selected_key("magnet_z_field_T")

    # Default window is 1 h: recent (5) + old points within 3600s (0 of the
    # "old" set, since the closest is 7200s) -> exactly the 5 recent points.
    panel.refresh()
    x_data, _ = panel._curve.getData()
    assert len(x_data) == 5

    # Switch to 15 min: still the same 5 recent points (all <= 600s old).
    window_selector = panel.findChild(QComboBox, "trend_window_selector_p2")
    window_selector.setCurrentText("15 min")
    x_data, _ = panel._curve.getData()
    assert len(x_data) == 5

    # Switch to 24 h: now all 10 points are included.
    window_selector.setCurrentText("24 h")
    x_data, _ = panel._curve.getData()
    assert len(x_data) == 10

    # Switch to 6 h: old_offsets max is 21600s = 6h exactly (>= cutoff keeps it);
    # use a value strictly inside 6h that excludes the farthest old point.
    window_selector.setCurrentText("6 h")
    x_data, _ = panel._curve.getData()
    expected_times, _ = history.series("magnet_z_field_T", window_s=21600.0)
    assert len(x_data) == len(expected_times)


# ── Selection round-trip ─────────────────────────────────────────────────────

def test_set_selected_key_roundtrip_and_survives_refresh(qtbot, history):
    """set_selected_key/selected_key round-trip; selection preserved across refresh()."""
    now = time.time()
    _record_series(history, {"magnet_z": {"field_T": 1.0}}, timestamp=now)
    _record_series(history, {"temp_a": {"temperature_K": 4.2}}, timestamp=now)

    panel = TrendPlotPanel(history, panel_id="p3")
    qtbot.addWidget(panel)
    panel.refresh()

    panel.set_selected_key("temp_a_temperature_K")
    assert panel.selected_key() == "temp_a_temperature_K"

    # A new key appears; refresh() must preserve the still-valid selection.
    _record_series(history, {"level_he": {"level_pct": 80.0}}, timestamp=now)
    panel.refresh()
    assert panel.selected_key() == "temp_a_temperature_K"

    y_selector = panel.findChild(QComboBox, "trend_y_selector_p3")
    combo_items = [y_selector.itemText(i) for i in range(y_selector.count())]
    assert "level_he_level_pct" in combo_items


def test_selected_window_s_roundtrip(qtbot, history):
    """set_selected_window_s/selected_window_s round-trip over all TIME_WINDOWS."""
    panel = TrendPlotPanel(history, panel_id="p4")
    qtbot.addWidget(panel)

    for _label, seconds in TIME_WINDOWS:
        panel.set_selected_window_s(seconds)
        assert panel.selected_window_s() == seconds


# ── Remove button signal ─────────────────────────────────────────────────────

def test_remove_button_emits_remove_requested_with_panel_id(qtbot, history):
    """Clicking the remove button emits remove_requested(panel_id)."""
    panel = TrendPlotPanel(history, panel_id="p5")
    qtbot.addWidget(panel)
    remove_button = panel.findChild(QPushButton, "trend_remove_button_p5")

    with qtbot.waitSignal(panel.remove_requested, timeout=1000) as blocker:
        remove_button.click()

    assert blocker.args == ["p5"]


# ── Long windows (7 d / 1 y) read from disk ──────────────────────────────────

def test_time_windows_includes_7d_and_1y_after_24h():
    """TIME_WINDOWS gains ("7 d", 604800.0) and ("1 y", 31536000.0) after "24 h"."""
    labels = [label for label, _ in TIME_WINDOWS]
    assert labels.index("24 h") == labels.index("7 d") - 1
    assert labels.index("7 d") == labels.index("1 y") - 1
    assert dict(TIME_WINDOWS)["7 d"] == 604800.0
    assert dict(TIME_WINDOWS)["1 y"] == 31536000.0


def test_window_selector_offers_7d_and_1y(qtbot, history):
    """The window selector combo lists the two new long-window options."""
    panel = TrendPlotPanel(history, panel_id="p6")
    qtbot.addWidget(panel)

    window_selector = panel.findChild(QComboBox, "trend_window_selector_p6")
    items = [window_selector.itemText(i) for i in range(window_selector.count())]
    assert "7 d" in items
    assert "1 y" in items


def test_selecting_7d_reads_3min_tier_from_disk(qtbot, history, tmp_path):
    """Selecting "7 d" plots data read from the 3-min tier on disk, not RAM."""
    now = time.time()
    key = "temperature_sample_temperature_K"
    _write_jsonl(
        tmp_path / "trend_history_3min.jsonl",
        [
            _bucket_record(now - 3600, {key: _stats(4.0, 4.2, 4.1, 0.05, 60)}),
            _bucket_record(now - 1800, {key: _stats(4.1, 4.3, 4.2, 0.05, 60)}),
        ],
    )

    # RAM history has the key too, but with different data — proves the
    # long window is NOT served from RAM.
    history.record({"temperature_sample": {"temperature_K": 99.0}}, timestamp=now)

    panel = TrendPlotPanel(history, panel_id="p7", log_dir=tmp_path)
    qtbot.addWidget(panel)
    panel.refresh()
    panel.set_selected_key(key)
    window_selector = panel.findChild(QComboBox, "trend_window_selector_p7")
    window_selector.setCurrentText("7 d")

    x_data, y_data = panel._curve.getData()
    assert len(x_data) == 2
    assert list(y_data) == [4.1, 4.2]


def test_selecting_1y_reads_hourly_tier_from_disk(qtbot, history, tmp_path):
    """Selecting "1 y" plots data read from the hourly tier on disk."""
    now = time.time()
    key = "temperature_sample_temperature_K"
    _write_jsonl(
        tmp_path / "trend_history_hourly.jsonl",
        [
            _bucket_record(now - 86400 * 30, {key: _stats(3.9, 4.4, 4.15, 0.1, 3600)}),
        ],
    )

    # RAM history must know the key too, so the Y combo offers it — the
    # live path is what makes a key selectable, regardless of window.
    history.record({"temperature_sample": {"temperature_K": 4.2}}, timestamp=now)

    panel = TrendPlotPanel(history, panel_id="p8", log_dir=tmp_path)
    qtbot.addWidget(panel)
    panel.refresh()
    panel.set_selected_key(key)
    window_selector = panel.findChild(QComboBox, "trend_window_selector_p8")
    window_selector.setCurrentText("1 y")

    x_data, y_data = panel._curve.getData()
    assert len(x_data) == 1
    assert list(y_data) == [4.15]


def test_selecting_1h_still_reads_ram_and_ignores_missing_files(qtbot, history, tmp_path):
    """"1 h" reads MonitorHistory in RAM regardless of whether any disk file exists."""
    now = time.time()
    key = "magnet_z_field_T"
    history.record({"magnet_z": {"field_T": 1.5}}, timestamp=now)

    # log_dir points at an empty, otherwise-unused directory: no trend
    # history files exist here at all.
    panel = TrendPlotPanel(history, panel_id="p9", log_dir=tmp_path)
    qtbot.addWidget(panel)
    panel.refresh()
    panel.set_selected_key(key)
    window_selector = panel.findChild(QComboBox, "trend_window_selector_p9")
    window_selector.setCurrentText("1 h")

    x_data, y_data = panel._curve.getData()
    assert len(x_data) == 1
    assert list(y_data) == [1.5]


def test_disk_read_failure_or_empty_renders_empty_plot(qtbot, history, tmp_path):
    """A missing/empty disk store on a long window renders an empty curve, never raises."""
    key = "temperature_sample_temperature_K"
    history.record({"temperature_sample": {"temperature_K": 4.2}}, timestamp=time.time())

    empty_dir = tmp_path / "does_not_exist"
    panel = TrendPlotPanel(history, panel_id="p10", log_dir=empty_dir)
    qtbot.addWidget(panel)
    panel.refresh()
    panel.set_selected_key(key)
    window_selector = panel.findChild(QComboBox, "trend_window_selector_p10")
    window_selector.setCurrentText("1 y")

    x_data, y_data = panel._curve.getData()
    assert x_data is None or len(x_data) == 0
    assert y_data is None or len(y_data) == 0


def test_measurement_vi_key_plottable_live_but_empty_on_disk_window(qtbot, history, tmp_path):
    """Measurement-VI asymmetry: live in RAM, empty on a disk-backed window."""
    now = time.time()
    key = "lockin1_x_voltage_V"  # measurement-VI-shaped key: never persisted to disk tiers
    history.record({"lockin1": {"x_voltage_V": 0.003}}, timestamp=now)

    # Disk store exists (other keys persisted) but never carries this key,
    # exactly as Station.last_state_flat() excludes measurement VIs.
    _write_jsonl(
        tmp_path / "trend_history_hourly.jsonl",
        [_bucket_record(now - 3600, {"magnet_z_field_T": _stats(1.0, 1.0, 1.0, 0.0, 60)})],
    )

    panel = TrendPlotPanel(history, panel_id="p11", log_dir=tmp_path)
    qtbot.addWidget(panel)
    panel.refresh()

    # Live: plottable and appears in the combo with RAM data at a RAM window.
    panel.set_selected_key(key)
    window_selector = panel.findChild(QComboBox, "trend_window_selector_p11")
    window_selector.setCurrentText("1 h")
    x_data, y_data = panel._curve.getData()
    assert len(x_data) == 1
    assert list(y_data) == [0.003]

    # Same key, long (disk-backed) window: empty, not an error.
    window_selector.setCurrentText("1 y")
    x_data, y_data = panel._curve.getData()
    assert x_data is None or len(x_data) == 0
    assert y_data is None or len(y_data) == 0
