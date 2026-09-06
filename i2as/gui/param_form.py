"""param_form — the one ParamSpec -> Qt-widget mapping for the procedure form.

This is the ONLY place that names Qt widget classes for procedure parameters.
Layer L4 (procedures) declares parameters as ``ParamSpec`` value objects and
never mentions a widget; the GUI's ProcedureWindow owns the surrounding layout
and the ``SweepAxisWidget``; and this module bridges the two. Adding a new input
kind (a new ParamSpec shape) means adding a branch here and nowhere else.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QHeaderView,
)
import qtawesome as qta

from i2as.core.plan import ParamGroup, ParamSpec
from i2as.gui.theme import TEXT_PRIMARY

__all__ = [
    "LoopValuesWidget",
    "build_param_widget",
    "build_param_tooltip",
    "build_form_layout",
    "build_group_box",
    "collect_value",
    "get_widget_raw",
    "set_widget_raw",
]


class LoopValuesWidget(QWidget):
    """A structured array/list editor widget for parameter loop values.

    Renders a single-column table of values with Add/Remove buttons.
    """

    def __init__(self, default_val: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self.set_raw(default_val)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Single column table: "Value"
        self._table = QTableWidget(0, 1)
        self._table.setObjectName("loop_values_table")
        self._table.setHorizontalHeaderLabels(["Value"])
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setMaximumHeight(120)  # keep it compact
        self._table.cellClicked.connect(self._on_cell_clicked)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        
        self._add_btn = QPushButton()
        self._add_btn.setObjectName("loop_values_add_btn")
        self._add_btn.setIcon(qta.icon("fa5s.plus", color=TEXT_PRIMARY))
        self._add_btn.setToolTip("Add loop value")
        self._add_btn.setMaximumWidth(32)
        self._add_btn.clicked.connect(self._add_row)
        
        self._remove_btn = QPushButton()
        self._remove_btn.setObjectName("loop_values_remove_btn")
        self._remove_btn.setIcon(qta.icon("fa5s.minus", color=TEXT_PRIMARY))
        self._remove_btn.setToolTip("Remove selected loop value")
        self._remove_btn.setMaximumWidth(32)
        self._remove_btn.clicked.connect(self._remove_row)

        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._remove_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _on_cell_clicked(self, row: int, column: int) -> None:
        item = self._table.item(row, column)
        if item is not None and item.flags() & Qt.ItemFlag.ItemIsEditable:
            self._table.editItem(item)

    def _add_row(self) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(""))

    def _remove_row(self) -> None:
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)
        elif self._table.rowCount() > 0:
            self._table.removeRow(self._table.rowCount() - 1)

    def get_raw(self) -> str:
        vals = []
        for r in range(self._table.rowCount()):
            item = self._table.item(r, 0)
            if item is not None:
                text = item.text().strip()
                if text:
                    vals.append(text)
        return ", ".join(vals)

    def set_raw(self, raw: str) -> None:
        self._table.setRowCount(0)
        if not raw:
            return
        # Split by comma to get individual values/ranges
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        for part in parts:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(part))



def build_param_widget(param_name: str, spec: ParamSpec) -> QWidget:
    """Create the input widget for one parameter, chosen by its ``ParamSpec``.

    * ``spec.choices`` (label -> value dict) -> ``QComboBox`` of the labels,
      preselected to the label whose value equals ``spec.default``.
    * ``spec.type is bool`` -> ``QCheckBox``, checked to ``spec.default``.
    * otherwise -> ``QLineEdit`` seeded with ``str(spec.default)``.

    Args:
        param_name: The parameter name (used only for error/debug context).
        spec: The parameter's ``ParamSpec`` declaration.

    Returns:
        The constructed widget (not yet registered or laid out).
    """
    if spec.choices:
        combo = QComboBox()
        for label in spec.choices:
            combo.addItem(str(label))
        for label, value in spec.choices.items():
            if value == spec.default:
                combo.setCurrentText(str(label))
                break
        return combo
    if spec.type is bool:
        check = QCheckBox()
        check.setChecked(bool(spec.default))
        return check
    if spec.widget_hint == "datetime" and spec.type is str:
        # Still a QLineEdit (the type is str, an ISO 8601 string) — the hint
        # only changes the placeholder shown when the field is empty, so a
        # the operator knows the expected format without needing a
        # dedicated date-picker widget yet.
        field = QLineEdit(str(spec.default))
        field.setPlaceholderText("YYYY-MM-DDTHH:MM:SS+00:00 (UTC)")
        return field
    if spec.widget_hint == "array":
        default_str = ""
        if spec.default:
            if isinstance(spec.default, (list, tuple)):
                default_str = ", ".join(str(x) for x in spec.default)
            else:
                default_str = str(spec.default)
        return LoopValuesWidget(default_str)
    return QLineEdit(str(spec.default))


def build_param_tooltip(spec: ParamSpec) -> str:
    """Build the hover-tooltip text for one ``ParamSpec``.

    Assembles, in order and skipping any part whose source data is absent:
    the ``description`` sentence (a period is appended if missing), then
    ``Default: {default} {unit}.``, then a ``Range: ...`` line built from
    ``min``/``max`` (one-sided phrasing — ``Min: ...`` / ``Max: ...`` — when
    only one bound is declared).

    Args:
        spec: A single parameter's ``ParamSpec`` (``type``, ``default``, and
            optionally ``unit``, ``min``, ``max``, ``description``).

    Returns:
        A single plain-text string, parts joined with a space.
    """
    unit = spec.unit
    parts: list[str] = []

    description = spec.description
    if description:
        parts.append(description if description.endswith(".") else f"{description}.")

    unit_suffix = f" {unit}" if unit else ""
    parts.append(f"Default: {spec.default}{unit_suffix}.")

    has_min = spec.min is not None
    has_max = spec.max is not None
    if has_min and has_max:
        parts.append(f"Range: {spec.min} to {spec.max}{unit_suffix}.")
    elif has_min:
        parts.append(f"Min: {spec.min}{unit_suffix}.")
    elif has_max:
        parts.append(f"Max: {spec.max}{unit_suffix}.")

    return " ".join(parts)


def build_form_layout(
    params: Mapping[str, ParamSpec],
    label_overrides: Mapping[str, str] | None = None,
    wrap: bool = False,
) -> tuple[QFormLayout, dict[str, QWidget]]:
    """Build a ``QFormLayout`` of input rows for a name -> ``ParamSpec`` mapping.

    Each row's widget is chosen per parameter by ``build_param_widget``: a
    drop-down for ``choices``, a checkbox for ``bool``, else a text field.

    The label IS the canonical parameter name — the same key used in the
    procedure code and stored under ``/metadata/procedure_params`` in the HDF5
    output — plus its unit: ``f"{param_name} ({unit}):"`` when the spec declares
    a non-empty ``unit``, else ``f"{param_name}:"``. The prose ``description``
    (plus default and min/max range) moves into a tooltip set on both the input
    field and its form label. Each field's ``objectName`` is
    ``f"param_{param_name}_input"``.

    ``label_overrides`` lets the caller show a *prettier* visible label than the
    canonical parameter name WITHOUT changing the name the value is collected
    under (or its HDF5 metadata key). The ProcedureWindow uses this for the
    reading-loop column, where a parameter name is namespaced (kept prefixed so
    a choice cannot collide with a measurement/system parameter) but the
    visible row label is the bare choice. Only the label changes; the
    ``objectName``, the collected key, and the metadata key are untouched.

    Args:
        params: A parameter-group mapping (name -> ``ParamSpec``).
        label_overrides: Optional ``{param_name: visible_label}`` map; a param
            present here uses ``visible_label`` (plus its unit) instead of its
            canonical name for the row label. The unit suffix is still appended.
        wrap: When True, sets ``WrapLongRows`` so a row too wide for its column
            drops its field beneath the label (lowering the column's minimum
            width). Default False keeps every row inline.

    Returns:
        ``(form, widgets)``: the populated ``QFormLayout`` (not yet attached to a
        parent widget) and a ``{param_name: QWidget}`` registry of its inputs.
    """
    overrides = label_overrides or {}
    form = QFormLayout()
    form.setSpacing(4)
    if wrap:
        # WrapLongRows drops a row's field beneath its label only when the row
        # is too narrow to fit both side by side. It leaves short rows inline
        # but lowers the layout's minimum width to ~max(label, field) instead of
        # label+field — letting a capped column compress without a horizontal
        # scrollbar, and letting a wrapped field use the column's full width
        # (so long values are no longer clipped). Used for the Measurement /
        # reading-loop columns, which the ProcedureWindow must fit four-across.
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    widgets: dict[str, QWidget] = {}
    for param_name, spec in params.items():
        unit = spec.unit
        display_name = overrides.get(param_name, param_name)
        label_text = f"{display_name} ({unit}):" if unit else f"{display_name}:"
        field = build_param_widget(param_name, spec)
        field.setObjectName(f"param_{param_name}_input")
        tooltip = build_param_tooltip(spec)
        field.setToolTip(tooltip)
        widgets[param_name] = field
        form.addRow(label_text, field)
        row_label = form.labelForField(field)
        if row_label is not None:
            row_label.setToolTip(tooltip)
    return form, widgets


def build_group_box(
    group: ParamGroup, wrap: bool = False
) -> tuple[QGroupBox, dict[str, QWidget]]:
    """Build one titled ``QGroupBox`` panel for a ``ParamGroup``.

    The box title is ``group.title``; its layout is the ``QFormLayout`` produced
    by ``build_form_layout(group.params)`` — the same rows the flat form renders.

    Args:
        group: The ``ParamGroup`` to render.
        wrap: Forwarded to ``build_form_layout`` (WrapLongRows when True).

    Returns:
        ``(box, widgets)``: the ``QGroupBox`` and its ``{param_name: QWidget}``
        input registry (so the caller can read the fields back on collect).
    """
    box = QGroupBox(group.title)
    form, widgets = build_form_layout(group.params, wrap=wrap)
    box.setLayout(form)
    return box, widgets


def collect_value(widget: QWidget, spec: ParamSpec) -> Any:
    """Read one input widget's current value, typed per its ``ParamSpec``.

    Inverse of ``build_param_widget``: a combobox returns the *mapped* value for
    its selected label, a checkbox returns a ``bool``, and a text field returns
    its stripped text coerced by ``spec.type``.

    Args:
        widget: A widget created by ``build_param_widget``.
        spec: The parameter's ``ParamSpec``.

    Returns:
        The collected value (mapped choice value, bool, or ``spec.type(text)``).

    Raises:
        ValueError: If a text field's contents cannot be parsed as ``spec.type``.
        TypeError: If ``spec.type`` rejects the text (e.g. wrong argument type).
    """
    if spec.choices:
        # A combobox can only hold labels this module added, so the lookup is safe.
        return spec.choices[widget.currentText()]
    if spec.type is bool:
        return widget.isChecked()
    if isinstance(widget, LoopValuesWidget):
        return spec.type(widget.get_raw())
    raw = widget.text().strip()
    return spec.type(raw)


def get_widget_raw(widget: QWidget) -> str:
    """Read a parameter widget's display value as a string (for caching).

    Uniform string form so session persistence never has to branch on the
    concrete widget type: combobox -> current label, checkbox -> ``"True"`` /
    ``"False"``, line-edit -> its text.

    Args:
        widget: A widget created by ``build_param_widget``.

    Returns:
        The widget's current value as a string.
    """
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QCheckBox):
        return str(widget.isChecked())
    if isinstance(widget, LoopValuesWidget):
        return widget.get_raw()
    return widget.text() if isinstance(widget, QLineEdit) else ""


def set_widget_raw(widget: QWidget, raw: str) -> None:
    """Restore a parameter widget from a cached display string.

    Inverse of ``get_widget_raw``; a value that no longer matches any combobox
    item (or an unparseable checkbox string) is ignored so a stale cache can
    never crash form restoration.

    Args:
        widget: A widget created by ``build_param_widget``.
        raw: The cached string previously returned by ``get_widget_raw``.
    """
    if isinstance(widget, QComboBox):
        if raw in (widget.itemText(i) for i in range(widget.count())):
            widget.setCurrentText(raw)
    elif isinstance(widget, QCheckBox):
        widget.setChecked(raw == "True")
    elif isinstance(widget, LoopValuesWidget):
        widget.set_raw(raw)
    elif isinstance(widget, QLineEdit):
        widget.setText(raw)
