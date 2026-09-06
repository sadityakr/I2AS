import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidgetItem

from i2as.core.sweep_builder import SweepAxis
from i2as.gui.sweep_axis_widget import SweepAxisWidget


@pytest.fixture
def axis():
    return SweepAxis(
        key="field",
        unit="T",
        data_key="field_T",
        description="Magnetic field",
        default_start=-1.0,
        default_end=1.0,
        default_steps=101,
    )


@pytest.fixture
def widget(axis, qtbot):
    w = SweepAxisWidget(axis)
    qtbot.addWidget(w)
    return w


def _set_row(table, row, value, step):
    table.setItem(row, 0, QTableWidgetItem(str(value)))
    table.setItem(row, 1, QTableWidgetItem(str(step)))


def test_param_keys(widget):
    assert widget.param_keys() == {
        "field_mode",
        "field_start",
        "field_end",
        "field_steps",
        "field_segments",
        "field_csv_path",
        "field_hysteresis",
    }


def test_default_mode_is_linear_with_axis_defaults(widget, axis):
    params = widget.get_params()
    assert params["field_mode"] == "linear"
    assert params["field_start"] == pytest.approx(axis.default_start)
    assert params["field_end"] == pytest.approx(axis.default_end)
    assert params["field_steps"] == axis.default_steps
    assert params["field_segments"] == []
    assert params["field_csv_path"] == ""
    assert params["field_hysteresis"] is False


def test_switching_mode_changes_stack_page(widget):
    widget._mode_combo.setCurrentIndex(1)  # Segments
    assert widget._stack.currentIndex() == 1
    widget._mode_combo.setCurrentIndex(2)  # CSV
    assert widget._stack.currentIndex() == 2


def test_linear_mode_reads_edited_fields(widget):
    widget._start_input.setText("-0.5")
    widget._end_input.setText("0.5")
    widget._steps_input.setText("5")
    params = widget.get_params()
    assert params["field_start"] == pytest.approx(-0.5)
    assert params["field_end"] == pytest.approx(0.5)
    assert params["field_steps"] == 5


def test_linear_mode_unparseable_field_raises(widget):
    widget._start_input.setText("not_a_number")
    with pytest.raises(ValueError, match="start"):
        widget.get_params()


def test_segments_mode_without_rows_raises(widget):
    widget._mode_combo.setCurrentIndex(1)  # Segments
    with pytest.raises(ValueError, match="at least two breakpoints"):
        widget.get_params()


def test_segments_mode_single_row_raises(widget):
    widget._mode_combo.setCurrentIndex(1)
    widget._add_segment_row()
    _set_row(widget._segments_table, 0, -0.1, 0.04)
    with pytest.raises(ValueError, match="at least two breakpoints"):
        widget.get_params()


def test_segments_mode_returns_paired_breakpoints(widget):
    widget._mode_combo.setCurrentIndex(1)
    widget._add_segment_row()
    _set_row(widget._segments_table, 0, -0.1, 0.04)
    widget._add_segment_row()
    _set_row(widget._segments_table, 1, -0.02, 0.02)
    widget._add_segment_row()
    _set_row(widget._segments_table, 2, 0.02, "")  # last row: step disabled/unused

    params = widget.get_params()
    assert params["field_mode"] == "segments"
    assert params["field_segments"] == [
        {"start": -0.1, "end": -0.02, "step": 0.04},
        {"start": -0.02, "end": 0.02, "step": 0.02},
    ]


def test_add_segment_row_disables_last_row_step_cell(widget):
    widget._add_segment_row()
    assert not (widget._segments_table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable)
    widget._add_segment_row()
    assert widget._segments_table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
    assert not (widget._segments_table.item(1, 1).flags() & Qt.ItemFlag.ItemIsEditable)


def test_segments_mode_bad_value_cell_raises(widget):
    widget._mode_combo.setCurrentIndex(1)
    widget._add_segment_row()
    _set_row(widget._segments_table, 0, "oops", 0.04)
    widget._add_segment_row()
    _set_row(widget._segments_table, 1, 0.02, "")
    with pytest.raises(ValueError, match="row 1"):
        widget.get_params()


def test_segments_mode_bad_step_cell_raises(widget):
    widget._mode_combo.setCurrentIndex(1)
    widget._add_segment_row()
    _set_row(widget._segments_table, 0, -0.1, "oops")
    widget._add_segment_row()
    _set_row(widget._segments_table, 1, 0.02, "")
    with pytest.raises(ValueError, match="row 1"):
        widget.get_params()


def test_csv_mode_without_path_raises(widget):
    widget._mode_combo.setCurrentIndex(2)  # CSV
    with pytest.raises(ValueError, match="no file chosen"):
        widget.get_params()


def test_csv_mode_with_path_returns_it(widget, tmp_path):
    csv_file = tmp_path / "fields.csv"
    csv_file.write_text("1.0\n0.0\n-1.0\n")
    widget._mode_combo.setCurrentIndex(2)
    widget._csv_input.setText(str(csv_file))

    params = widget.get_params()
    assert params["field_mode"] == "csv"
    assert params["field_csv_path"] == str(csv_file)


def test_hysteresis_checkbox_included_regardless_of_mode(widget):
    widget._hysteresis_checkbox.setChecked(True)
    assert widget.get_params()["field_hysteresis"] is True

    widget._mode_combo.setCurrentIndex(2)  # CSV — unrelated to hysteresis
    widget._csv_input.setText("dummy.csv")
    assert widget.get_params()["field_hysteresis"] is True


def test_remove_segment_row(widget):
    widget._mode_combo.setCurrentIndex(1)
    widget._add_segment_row()
    widget._add_segment_row()
    widget._segments_table.setCurrentCell(0, 0)
    widget._remove_segment_row()
    assert widget._segments_table.rowCount() == 1
