"""SweepAxisWidget — mode-selector sweep-shape editor for a SweepAxis."""

from __future__ import annotations

from typing import Any

import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from i2as.core.sweep_builder import SweepAxis
from i2as.gui.theme import BG_BASE, BG_ELEVATED, TEXT_MUTED, TEXT_PRIMARY

# Row order matches the QComboBox item order; index <-> mode string.
_MODES = ["linear", "segments", "csv"]
_MODE_LABELS = ["Linear", "Segments", "CSV"]
# Each row is one breakpoint. Row i's Value and row i+1's Value become one
# SweepSegment's start/end; row i's "Step to next" is that segment's step.
# The last row's step is unused (no following breakpoint) and disabled in
# the UI rather than shown as a live, ignorable input.
_SEGMENT_COLUMNS = ["Value", "Step to next"]
# Max width (px) for the single-value Linear/CSV input fields, so a short
# number doesn't stretch the Sweep column wide and leave it looking empty.
_FIELD_MAX_WIDTH = 150
# Width (px) of each Segments-table column. Values here are short numbers
# (e.g. "-0.10"), so a narrow fixed width keeps the table compact instead of
# defaulting to Qt's much wider 100px-per-column split.
_SEGMENT_COLUMN_WIDTH = 70
# Shown (as a tooltip and placeholder) on the last row's Step cell so its
# disabled state reads as intentional rather than a broken/unresponsive field.
_LAST_ROW_STEP_TOOLTIP = "No step: add another breakpoint to set this segment's step."


class SweepAxisWidget(QWidget):
    """Sweep-shape editor for one declared ``SweepAxis``.

    Args:
        axis: The Procedure's declared sweep axis.
        parent: Optional Qt parent widget.
    """

    def __init__(self, axis: SweepAxis, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._axis = axis
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        k = self._axis.key
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        mode_row = QHBoxLayout()
        self._mode_combo = QComboBox()
        self._mode_combo.setObjectName(f"sweep_{k}_mode_combo")
        self._mode_combo.addItems(_MODE_LABELS)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        root.addLayout(mode_row)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_linear_page())
        self._stack.addWidget(self._build_segments_page())
        self._stack.addWidget(self._build_csv_page())
        root.addWidget(self._stack)

        self._hysteresis_checkbox = QCheckBox("Hysteresis")
        self._hysteresis_checkbox.setObjectName(f"sweep_{k}_hysteresis_checkbox")
        self._hysteresis_checkbox.setToolTip(
            "Run the sweep forward then back to its start (forward + backward "
            f"{self._axis.description.lower()} loop)."
        )
        root.addWidget(self._hysteresis_checkbox)

    def _build_linear_page(self) -> QWidget:
        k = self._axis.key
        page = QWidget()
        form = QFormLayout(page)
        form.setSpacing(4)

        self._start_input = QLineEdit(str(self._axis.default_start))
        self._start_input.setObjectName(f"sweep_{k}_start_input")
        self._end_input = QLineEdit(str(self._axis.default_end))
        self._end_input.setObjectName(f"sweep_{k}_end_input")
        self._steps_input = QLineEdit(str(self._axis.default_steps))
        self._steps_input.setObjectName(f"sweep_{k}_steps_input")

        # Cap the field width so short values (e.g. "-1.0") don't stretch the
        # whole Sweep column wide and leave it looking empty. Wide enough for
        # a full-precision number; the column then hugs its content.
        for field in (self._start_input, self._end_input, self._steps_input):
            field.setMaximumWidth(_FIELD_MAX_WIDTH)

        form.addRow("Start:", self._start_input)
        form.addRow("End:", self._end_input)
        form.addRow("Steps:", self._steps_input)
        return page

    def _build_segments_page(self) -> QWidget:
        k = self._axis.key
        page = QWidget()
        col = QVBoxLayout(page)
        col.setSpacing(4)

        self._segments_table = QTableWidget(0, len(_SEGMENT_COLUMNS))
        self._segments_table.setObjectName(f"sweep_{k}_segments_table")
        self._segments_table.setHorizontalHeaderLabels(list(_SEGMENT_COLUMNS))
        self._segments_table.verticalHeader().setVisible(False)
        header = self._segments_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        for column in range(len(_SEGMENT_COLUMNS)):
            self._segments_table.setColumnWidth(column, _SEGMENT_COLUMN_WIDTH)
        self._segments_table.setMaximumWidth(
            len(_SEGMENT_COLUMNS) * _SEGMENT_COLUMN_WIDTH + 24
        )
        self._segments_table.cellClicked.connect(self._on_segment_cell_clicked)
        col.addWidget(self._segments_table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton()
        add_btn.setObjectName(f"sweep_{k}_add_segment_btn")
        add_btn.setIcon(qta.icon("fa5s.plus", color=TEXT_PRIMARY))
        add_btn.setToolTip("Add breakpoint")
        add_btn.clicked.connect(self._add_segment_row)
        remove_btn = QPushButton()
        remove_btn.setObjectName(f"sweep_{k}_remove_segment_btn")
        remove_btn.setIcon(qta.icon("fa5s.minus", color=TEXT_PRIMARY))
        remove_btn.setToolTip("Remove selected breakpoint")
        remove_btn.clicked.connect(self._remove_segment_row)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        col.addLayout(btn_row)
        return page

    def _build_csv_page(self) -> QWidget:
        k = self._axis.key
        page = QWidget()
        row = QHBoxLayout(page)
        self._csv_input = QLineEdit()
        self._csv_input.setObjectName(f"sweep_{k}_csv_input")
        self._csv_input.setPlaceholderText("Path to single-column CSV file")
        browse_btn = QPushButton("Browse...")
        browse_btn.setObjectName(f"sweep_{k}_csv_browse_btn")
        browse_btn.clicked.connect(self._on_browse_csv)
        row.addWidget(self._csv_input)
        row.addWidget(browse_btn)
        return page

    # ------------------------------------------------------------------
    # Slot handlers
    # ------------------------------------------------------------------

    def _on_mode_changed(self, index: int) -> None:
        self._stack.setCurrentIndex(index)

    def _on_segment_cell_clicked(self, row: int, column: int) -> None:
        """Open the clicked cell for editing on a single click, not just a double click.

        Qt's default double-click-to-edit trigger has proven unreliable on a
        cell's very first click in this environment, silently doing nothing.
        A single click editing directly (like a spreadsheet) sidesteps that.
        """
        item = self._segments_table.item(row, column)
        if item is not None and item.flags() & Qt.ItemFlag.ItemIsEditable:
            self._segments_table.editItem(item)

    def _add_segment_row(self) -> None:
        """Append a blank breakpoint row (Value, Step to next).

        Segment contiguity is automatic in the 2-column breakpoint model —
        row i's Value doubles as the previous segment's End — so, unlike a
        3-column Start/End/Step table, there is nothing to carry forward.
        """
        row = self._segments_table.rowCount()
        self._segments_table.insertRow(row)
        self._segments_table.setItem(row, 0, QTableWidgetItem(""))
        self._segments_table.setItem(row, 1, QTableWidgetItem(""))
        self._refresh_step_column_state()

    def _remove_segment_row(self) -> None:
        row = self._segments_table.currentRow()
        if row >= 0:
            self._segments_table.removeRow(row)
            self._refresh_step_column_state()

    def _refresh_step_column_state(self) -> None:
        """Disable the last row's Step cell: there is no following breakpoint for it to reach.

        Every other row's Step cell is (re-)enabled, since a row that used to
        be last (and got disabled) may no longer be after an insert/remove.
        The disabled cell also gets a muted color and a tooltip explaining why
        it won't accept input, so it reads as intentional rather than broken.
        """
        last_row = self._segments_table.rowCount() - 1
        for row in range(self._segments_table.rowCount()):
            item = self._segments_table.item(row, 1)
            if item is None:
                continue
            if row == last_row:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setText("")
                item.setToolTip(_LAST_ROW_STEP_TOOLTIP)
                item.setBackground(QBrush(QColor(BG_BASE)))
                item.setForeground(QBrush(QColor(TEXT_MUTED)))
            else:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                item.setToolTip("")
                item.setBackground(QBrush(QColor(BG_ELEVATED)))
                item.setForeground(QBrush(QColor(TEXT_PRIMARY)))

    def _on_browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select sweep CSV file", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self._csv_input.setText(path)

    # ------------------------------------------------------------------
    # Public API consumed by ProcedureWindow
    # ------------------------------------------------------------------

    def param_keys(self) -> set[str]:
        """Return the set of hidden parameter names this widget owns.

        Used by ProcedureWindow to skip these in the generic flat-field
        collection loop (they're read via ``get_params()`` instead).
        """
        k = self._axis.key
        return {
            f"{k}_mode",
            f"{k}_start",
            f"{k}_end",
            f"{k}_steps",
            f"{k}_segments",
            f"{k}_csv_path",
            f"{k}_hysteresis",
        }

    def get_params(self) -> dict[str, Any]:
        """Read the current widget state into a sweep_axis parameter dict.

        Only the active mode's own inputs are validated strictly — fields on
        an inactive page are read on a best-effort basis (falling back to the
        axis defaults) so an unrelated half-filled tab never blocks a run.

        Returns:
            Dict of ``{axis.key}_``-prefixed values matching
            ``sweep_builder.sweep_axis_param_specs()``.

        Raises:
            ValueError: If the active mode's required input is missing or
                cannot be parsed.
        """
        k = self._axis.key
        mode = _MODES[self._mode_combo.currentIndex()]
        result: dict[str, Any] = {f"{k}_mode": mode}

        result[f"{k}_start"] = self._parse_float(
            self._start_input.text(), self._axis.default_start, required=mode == "linear",
            field_label=f"{self._axis.description} start",
        )
        result[f"{k}_end"] = self._parse_float(
            self._end_input.text(), self._axis.default_end, required=mode == "linear",
            field_label=f"{self._axis.description} end",
        )
        result[f"{k}_steps"] = self._parse_int(
            self._steps_input.text(), self._axis.default_steps, required=mode == "linear",
            field_label=f"{self._axis.description} steps",
        )

        result[f"{k}_segments"] = self._read_segments_table() if mode == "segments" else []

        if mode == "csv":
            path = self._csv_input.text().strip()
            if not path:
                raise ValueError("CSV sweep mode selected but no file chosen.")
            result[f"{k}_csv_path"] = path
        else:
            result[f"{k}_csv_path"] = ""

        result[f"{k}_hysteresis"] = self._hysteresis_checkbox.isChecked()
        return result

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_float(text: str, default: float, *, required: bool, field_label: str) -> float:
        try:
            return float(text)
        except ValueError:
            if required:
                raise ValueError(f"Cannot parse '{text}' as a number for '{field_label}'.") from None
            return default

    @staticmethod
    def _parse_int(text: str, default: int, *, required: bool, field_label: str) -> int:
        try:
            return int(text)
        except ValueError:
            if required:
                raise ValueError(f"Cannot parse '{text}' as an integer for '{field_label}'.") from None
            return default

    def _read_segments_table(self) -> list[dict[str, float]]:
        """Read the 2-column breakpoint table and pair consecutive rows into segments.

        Row i's Value and row i+1's Value become one SweepSegment's
        start/end; row i's Step is that segment's step. The last row
        contributes only its Value (its Step cell is disabled in the UI).

        Returns:
            List of ``{"start": ..., "end": ..., "step": ...}`` dicts, one
            per consecutive breakpoint pair — the same shape
            ``build_piecewise_sweep`` already consumes.

        Raises:
            ValueError: Fewer than two breakpoints, or a Value/Step cell
                that isn't a number.
        """
        rows = []
        for row in range(self._segments_table.rowCount()):
            value_item = self._segments_table.item(row, 0)
            step_item = self._segments_table.item(row, 1)
            value_text = value_item.text().strip() if value_item is not None else ""
            step_text = step_item.text().strip() if step_item is not None else ""
            if value_text == "" and step_text == "":
                continue
            rows.append((row, value_text, step_text))

        if len(rows) < 2:
            raise ValueError(
                "Segments sweep mode selected but at least two breakpoints are needed."
            )

        values: list[float] = []
        for row, value_text, _step_text in rows:
            try:
                values.append(float(value_text))
            except ValueError:
                raise ValueError(f"Breakpoint row {row + 1}: Value must be a number.") from None

        segments: list[dict[str, float]] = []
        for i, (row, _value_text, step_text) in enumerate(rows[:-1]):
            try:
                step = float(step_text)
            except ValueError:
                raise ValueError(
                    f"Breakpoint row {row + 1}: Step to next must be a number."
                ) from None
            segments.append({"start": values[i], "end": values[i + 1], "step": step})

        return segments
