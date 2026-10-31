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


def test_curve_mode_draws_the_newest_point_and_fades_earlier_ones(qtbot) -> None:
    """With an (x, y) curve pair on both axes, each datapoint is a whole curve."""
    keys = ["unix_time", "field_T", "resistance_ohm", "voltage_V_array", "current_A_array"]
    panel = LivePlotPanel("Plot 1", "#1f77b4", "x_sel", "y_sel", "plot")
    qtbot.addWidget(panel)
    panel.set_available_keys(keys, "field_T", "resistance_ohm")
    panel.set_curve_default("voltage_V_array", "current_A_array")
    panel.set_available_keys(keys, "voltage_V_array", "current_A_array")
    assert panel._x_selector.currentText() == "voltage_V_array"  # forced: a new curve method
    points = [
        {"voltage_V_array": [[[-1.0, 0.0, 1.0]]], "current_A_array": [[[-k, 0.0, k]]]}
        for k in (1.0, 2.0, 3.0)
    ]
    panel.redraw(points)
    x, y = panel._curve.getData()
    assert list(x) == [-1.0, 0.0, 1.0] and list(y) == [-3.0, 0.0, 3.0]
    assert list(panel._history_curves[0].getData()[1]) == [-2.0, 0.0, 2.0]
    # Back to a scalar axis: one value per point, no curves left behind.
    panel._x_selector.setCurrentText("field_T")
    assert len(panel._curve.getData()[0] if panel._curve.getData()[0] is not None else []) == 0
    panel._y_selector.setCurrentText("resistance_ohm")
    assert panel._history_curves[0].getData()[0] is None or len(panel._history_curves[0].getData()[0]) == 0
