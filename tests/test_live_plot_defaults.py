"""LivePlotPanel's axis defaults apply on first population and never clobber a choice."""

from __future__ import annotations

from i2as.gui.live_plot_panel import LivePlotPanel

KEYS = ["unix_time", "field_T", "voltage_V", "current_A", "resistance_ohm"]


def test_declared_defaults_apply_when_nothing_is_selected(qtbot) -> None:
    panel = LivePlotPanel("Plot 1", "#1f77b4", "x_sel", "y_sel", "plot")
    qtbot.addWidget(panel)
    panel.set_available_keys(KEYS, "field_T", "resistance_ohm")
    assert panel._x_selector.currentText() == "field_T"
    assert panel._y_selector.currentText() == "resistance_ohm"


def test_a_still_valid_choice_survives_and_an_invalid_one_falls_back(qtbot) -> None:
    panel = LivePlotPanel("Plot 1", "#1f77b4", "x_sel", "y_sel", "plot")
    qtbot.addWidget(panel)
    panel.set_available_keys(KEYS, "field_T", "resistance_ohm")
    panel._y_selector.setCurrentText("current_A")
    panel.set_available_keys(["unix_time", "applied_bias_V", "current_A"], "applied_bias_V", "voltage_V")
    assert panel._x_selector.currentText() == "applied_bias_V"  # field_T gone -> default
    assert panel._y_selector.currentText() == "current_A"  # kept
    panel.set_available_keys(["unix_time", "x"], "missing", None)
    assert panel._x_selector.currentText() == "unix_time"  # default absent -> first key
