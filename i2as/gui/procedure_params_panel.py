"""ProcedureParamsPanel — procedure selector, parameter form, and param cache."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from i2as.core.plan import ParamGroup, ParamSpec
from i2as.core.procedure import BaseProcedure
from i2as.gui import param_form
from i2as.gui.form_autosave import FormAutosaveState
from i2as.gui.sweep_axis_widget import SweepAxisWidget
from i2as.gui.theme import (
    BTN_CLASS_PRIMARY,
    BTN_CLASS_SECONDARY,
    TEXT_ON_ACCENT,
    TEXT_PRIMARY,
)

if TYPE_CHECKING:  # the GUI holds a Station only as a type (contract C19)
    from i2as.core.station import Station

# Max width (px) for the Sweep parameter column. Its inputs are narrow, so this
# stops it expanding to an equal third and crowding out the Measurement column.
_SWEEP_COLUMN_MAX_WIDTH = 260
# Max width (px) for the Reading loop column (two slot drop-downs plus their
# values inputs — pick checkboxes or a comma-separated text field).
_READING_LOOP_COLUMN_MAX_WIDTH = 320


class ProcedureParamsPanel(QWidget):
    """The parameter quadrant: selector, prefix field, and auto-generated form.

    ObjectNames (``params_quadrant``, ``procedure_selector``,
    ``add_to_queue_btn``, ``run_now_btn``, ``file_prefix_input``,
    ``param_scroll``, and the ``param_*_input`` fields built by
    ``param_form``) are preserved API — tests rely on them.

    Signals:
        add_to_queue_requested: The "Add to Queue" button was clicked.
        run_now_requested: The "Run Now" button was clicked.
        structure_changed: The form was (re)built for the given procedure
            class — the hosting window refreshes its plot axis selectors
            (including the reading-loop selectors, since a loop slot's
            parameter choice and pick checkboxes are all structural).

    Args:
        station: The active Station instance.
        procedures: The discovered concrete procedure classes, in menu order.
        parent: Optional Qt parent widget.
    """

    add_to_queue_requested = pyqtSignal()
    run_now_requested = pyqtSignal()
    structure_changed = pyqtSignal(object)  # type[BaseProcedure]

    def __init__(
        self,
        station: Station,
        procedures: list[type[BaseProcedure]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._station = station
        self._procedures = procedures

        # Per-procedure last-typed flat parameter text, keyed by procedure name.
        self._procedure_params: dict[str, dict[str, str]] = {}
        self._current_procedure_name: str = ""
        # Parameter widgets, keyed by parameter name. Type varies with the
        # spec: QLineEdit (plain number/text), QComboBox (enumerated choices),
        # or QCheckBox (bool). Built and read via i2as.gui.param_form.
        self._param_inputs: dict[str, QWidget] = {}
        # The parameter groups currently rendered (from get_param_groups), the
        # non-sweep group boxes keyed by ParamGroup.key, and the container's
        # horizontal layout — together they let a structural change re-render
        # only the group(s) that changed and leave the rest (and the axis
        # widget) untouched.
        self._current_groups: list[ParamGroup] = []
        self._group_boxes: dict[str, QGroupBox] = {}
        self._param_hbox: QHBoxLayout | None = None
        # The composite "Measurement" column: the method drop-down on top and
        # the selected VI's parameter sub-form below it, inside ONE QGroupBox
        # (rather than a box per group). On a method change only the sub-form
        # container is rebuilt; the box and the drop-down keep their instances.
        # Kept separate from _group_boxes, which holds the independent columns
        # (System, Reading loop, any generic group).
        self._measurement_box: QGroupBox | None = None
        self._measurement_params_layout: QVBoxLayout | None = None
        self._measurement_params_container: QWidget | None = None
        self._measurement_params_key: str | None = None
        # Sweep-axis editor for the selected procedure, if it declares one
        self._axis_widget: SweepAxisWidget | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setObjectName("params_quadrant")
        col = QVBoxLayout(self)
        col.setSpacing(8)
        col.setContentsMargins(0, 0, 4, 0)

        # ── Procedure selector + Add/Run buttons (one row) ─────────────
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Procedure:"))
        self._proc_selector = QComboBox()
        self._proc_selector.setObjectName("procedure_selector")
        for cls in self._procedures:
            self._proc_selector.addItem(getattr(cls, "name", cls.__name__))
        self._proc_selector.currentIndexChanged.connect(self._on_procedure_selected)
        sel_row.addWidget(self._proc_selector)
        sel_row.addStretch()

        add_btn = QPushButton("Add to Queue")
        add_btn.setObjectName("add_to_queue_btn")
        add_btn.setProperty("class", BTN_CLASS_SECONDARY)
        add_btn.setIcon(qta.icon("fa5s.plus", color=TEXT_PRIMARY))
        add_btn.setToolTip("Add the current procedure and parameters to the run queue")
        add_btn.clicked.connect(self.add_to_queue_requested)
        run_now_btn = QPushButton("Run Now")
        run_now_btn.setObjectName("run_now_btn")
        run_now_btn.setProperty("class", BTN_CLASS_PRIMARY)
        run_now_btn.setIcon(qta.icon("fa5s.play", color=TEXT_ON_ACCENT))
        run_now_btn.setToolTip("Run the current procedure immediately")
        run_now_btn.clicked.connect(self.run_now_requested)
        sel_row.addWidget(add_btn)
        sel_row.addWidget(run_now_btn)
        col.addLayout(sel_row)

        # ── Filename prefix (carried into the queue per entry) ─────────
        prefix_row = QHBoxLayout()
        prefix_row.addWidget(QLabel("Filename prefix:"))
        self._file_prefix_input = QLineEdit()
        self._file_prefix_input.setObjectName("file_prefix_input")
        self._file_prefix_input.setPlaceholderText(
            "optional — defaults to procedure name"
        )
        self._file_prefix_input.setToolTip(
            "Prepended to the saved HDF5 filename, still followed by a unique "
            "timestamp. Captured when a procedure is added to the queue, so "
            "each queued entry can carry its own filename."
        )
        prefix_row.addWidget(self._file_prefix_input)
        col.addLayout(prefix_row)

        # ── Parameters (scroll area; rebuilt on selector change) ──────
        self._param_scroll = QScrollArea()
        self._param_scroll.setObjectName("param_scroll")
        self._param_scroll.setWidgetResizable(True)
        col.addWidget(self._param_scroll)

    def initialize_selection(self) -> None:
        """Render the first procedure's form.

        Called by the hosting window AFTER it has connected to this panel's
        signals, so the initial ``structure_changed`` emission reaches the
        window's plot-selector refresh (mirroring the pre-extraction
        construction order).
        """
        if self._procedures:
            self._on_procedure_selected(0)

    # ------------------------------------------------------------------
    # Form construction
    # ------------------------------------------------------------------

    def _build_param_form(self, cls: type[BaseProcedure]) -> QWidget:
        """Auto-generate the parameter columns from the procedure's ParamGroups.

        Reads the ordered groups from ``cls.get_param_groups(station)`` and lays
        them out as side-by-side columns. Three group-key conventions are
        composed specially so the form fits on screen without horizontal scroll:

        * ``sweep`` — the capped-width Sweep column, always hosting the
          ``SweepAxisWidget`` (linear / segments / CSV, with hysteresis) when
          ``cls`` declares a ``sweep_axis``, plus any flat ``sweep`` fields.
        * ``measurement_select`` + ``measurement:<vi>`` — folded into ONE
          "Measurement" column (method drop-down on top, the selected VI's
          parameter sub-form below), instead of two separate columns that would
          overflow off-screen. See ``_build_measurement_box``.
        * ``reading_loop`` — the two generic loop slots (a loopable-parameter
          drop-down each, with per-choice pick checkboxes or a value-list text
          field). See ``_build_reading_loop_box``.

        Any OTHER group key becomes its own generic column (so future procedures
        are not constrained by these conventions). Widget construction, labels,
        and tooltips all come from ``i2as.gui.param_form`` (the one place
        that maps a ``ParamSpec`` to a Qt widget); this method only arranges the
        boxes and records each input in ``self._param_inputs``.

        Args:
            cls: The BaseProcedure subclass to introspect.

        Returns:
            A QWidget containing the columns.
        """
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setSpacing(8)
        hbox.setContentsMargins(0, 0, 0, 0)
        self._param_inputs.clear()
        self._group_boxes = {}
        self._measurement_box = None
        self._measurement_params_layout = None
        self._measurement_params_container = None
        self._measurement_params_key = None
        self._param_hbox = hbox
        self._axis_widget = None

        groups = cls.get_param_groups(self._station, self.current_selections())
        self._current_groups = groups
        groups_by_key = {group.key: group for group in groups}

        sweep_box = QGroupBox("Sweep")
        # The sweep inputs are narrow (single values / a compact 2-col table),
        # so cap the column width instead of letting it expand to an equal
        # third — otherwise it sits half-empty and pushes the Measurement
        # column out of view. Maximum policy lets it hug even narrower content;
        # the hard cap overrides the hidden Segments table's wide size hint.
        # System/Measurement (default Expanding) absorb the freed width.
        sweep_box.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        sweep_box.setMaximumWidth(_SWEEP_COLUMN_MAX_WIDTH)
        sweep_col = QVBoxLayout(sweep_box)
        sweep_col.setSpacing(4)
        if cls.sweep_axis is not None:
            self._axis_widget = SweepAxisWidget(cls.sweep_axis)
            sweep_col.addWidget(self._axis_widget)
        sweep_group = groups_by_key.get("sweep")
        if sweep_group is not None:
            form, widgets = param_form.build_form_layout(sweep_group.params)
            self._register_group_widgets(sweep_group, widgets)
            sweep_col.addLayout(form)
        sweep_col.addStretch()
        hbox.addWidget(sweep_box)

        # Remaining groups, in get_param_groups order. The measurement pair is
        # emitted once (at the measurement_select position) as the composite
        # Measurement column; its measurement:<vi> partner is skipped.
        for group in groups:
            key = group.key
            if key == "sweep" or key.startswith("measurement:"):
                continue
            if key == "measurement_select":
                params_group = next(
                    (g for g in groups if g.key.startswith("measurement:")), None
                )
                hbox.addWidget(self._build_measurement_box(group, params_group))
                continue
            if key == "reading_loop":
                box = self._build_reading_loop_box(group)
            else:
                # System / generic columns wrap long rows too, so all four
                # columns compress to fit the params quadrant without a
                # horizontal scrollbar.
                box, widgets = param_form.build_group_box(group, wrap=True)
                self._register_group_widgets(group, widgets)
            hbox.addWidget(box)
            self._group_boxes[key] = box

        return container

    def _build_measurement_box(
        self, select_group: ParamGroup, params_group: ParamGroup | None
    ) -> QGroupBox:
        """Build the composite "Measurement" column (selector on top, params below).

        The method drop-down (labelled "Method:") sits at the top, a separator
        beneath it, then the selected VI's parameter sub-form. Only the sub-form
        is rebuilt on a method change (see ``_rerender_groups``); the box and the
        drop-down keep their instances. The drop-down keeps the objectName
        ``param_measurement_vi_input`` so collection and tests still find it.

        Args:
            select_group: The ``measurement_select`` group (the ``measurement_vi``
                selector spec).
            params_group: The ``measurement:<vi>`` group for the currently
                selected VI, or ``None`` if that VI declares no parameters.

        Returns:
            The "Measurement" ``QGroupBox``.
        """
        box = QGroupBox("Measurement")
        vbox = QVBoxLayout(box)
        vbox.setSpacing(6)

        spec = select_group.params["measurement_vi"]
        combo = param_form.build_param_widget("measurement_vi", spec)
        combo.setObjectName("param_measurement_vi_input")
        combo.setToolTip(param_form.build_param_tooltip(spec))
        if isinstance(combo, QComboBox):
            # Let the combo shrink so the longest method label does not force the
            # whole column wide (the pre-fix bug), and carry the vi_name in a
            # per-item tooltip so two VIs sharing a selector_label stay
            # distinguishable. The collected value is still the vi_name.
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            # Pin a small minimum so the widest method label does not floor the
            # whole column's width (Expanding still lets it grow to fill the
            # column); the full label is always readable once dropped down.
            combo.setMinimumWidth(90)
            for i in range(combo.count()):
                vi_name = spec.choices.get(combo.itemText(i))
                if vi_name:
                    combo.setItemData(i, vi_name, Qt.ItemDataRole.ToolTipRole)
        method_form = QFormLayout()
        method_form.setSpacing(4)
        method_form.addRow("Method:", combo)
        self._register_group_widgets(select_group, {"measurement_vi": combo})
        vbox.addLayout(method_form)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        vbox.addWidget(line)

        self._measurement_params_layout = QVBoxLayout()
        self._measurement_params_layout.setContentsMargins(0, 0, 0, 0)
        vbox.addLayout(self._measurement_params_layout)
        if params_group is not None:
            self._install_measurement_params(params_group)
        vbox.addStretch()

        self._measurement_box = box
        return box

    def _install_measurement_params(self, params_group: ParamGroup) -> None:
        """Render the selected VI's parameter sub-form into the Measurement column.

        Records the sub-form's container and its group key so a later method
        change can remove exactly this sub-form and drop its widgets from the
        registry, leaving the rest of the Measurement column untouched.

        Args:
            params_group: The ``measurement:<vi>`` group to render.
        """
        assert self._measurement_params_layout is not None
        self._measurement_params_key = params_group.key
        form, widgets = param_form.build_form_layout(params_group.params, wrap=True)
        self._register_group_widgets(params_group, widgets)
        container = QWidget()
        container.setLayout(form)
        self._measurement_params_container = container
        self._measurement_params_layout.addWidget(container)

    def _build_reading_loop_box(self, group: ParamGroup) -> QGroupBox:
        """Build the Reading loop column: the two generic loop slots.

        Each slot is a ``{slot}_parameter`` drop-down plus its values input —
        per-choice ``{slot}_pick_{value}`` checkboxes for an enumerated
        parameter (each row label prettified to the bare choice via
        ``label_overrides``; the parameter key keeps its namespaced name for
        collection and HDF5 metadata) or a ``{slot}_values`` text field
        otherwise. All widgets are structural, so their wiring comes from
        ``_register_group_widgets``.

        Args:
            group: The ``reading_loop`` group.

        Returns:
            The capped-width ``QGroupBox``.
        """
        box = QGroupBox(group.title)
        box.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        box.setMaximumWidth(_READING_LOOP_COLUMN_MAX_WIDTH)
        label_overrides = {
            name: name.split("_pick_", 1)[1]
            for name in group.params
            if "_pick_" in name
        }
        form, widgets = param_form.build_form_layout(group.params, label_overrides, wrap=True)
        self._register_group_widgets(group, widgets)
        box.setLayout(form)
        return box

    # ------------------------------------------------------------------
    # Structural re-render (a structural param changes which groups exist)
    # ------------------------------------------------------------------

    def _register_group_widgets(
        self, group: ParamGroup, widgets: dict[str, QWidget]
    ) -> None:
        """Record a group's input widgets and wire up any structural ones.

        Args:
            group: The ``ParamGroup`` the widgets belong to.
            widgets: ``{param_name: QWidget}`` from ``param_form``.
        """
        for name, widget in widgets.items():
            self._param_inputs[name] = widget
            if group.params[name].structural:
                self._connect_structural(widget, group.params[name])

    def _connect_structural(self, widget: QWidget, spec: ParamSpec) -> None:
        """Connect a structural param widget's change signal to a re-render."""
        if isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._on_structural_changed)
        elif isinstance(widget, QCheckBox):
            widget.stateChanged.connect(self._on_structural_changed)
        elif isinstance(widget, QLineEdit):
            widget.editingFinished.connect(self._on_structural_changed)

    def current_selections(self) -> dict[str, Any]:
        """Return the current values of every rendered structural parameter.

        Fed back into ``get_param_groups`` so the form re-derives from the user's
        structural choices (e.g. which measurement VI is selected).
        """
        selections: dict[str, Any] = {}
        for group in self._current_groups:
            for name, spec in group.params.items():
                if not spec.structural:
                    continue
                widget = self._param_inputs.get(name)
                if widget is None:
                    continue
                try:
                    selections[name] = param_form.collect_value(widget, spec)
                except (ValueError, TypeError):
                    pass
        return selections

    def _on_structural_changed(self, *_args: Any) -> None:
        """Re-derive the form after a structural parameter changed.

        Caches the currently-typed values, re-derives the groups from the new
        selections, rebuilds ONLY the group boxes whose key changed (leaving the
        sweep/system boxes and the axis widget as the same widget instances),
        then re-applies cached values and signals the structure change so the
        hosting window refreshes its plot axis selectors.
        """
        index = self._proc_selector.currentIndex()
        if index < 0 or index >= len(self._procedures):
            return
        cls = self._procedures[index]
        self.cache_current_params()
        self._rerender_groups(cls, self.current_selections())
        self._apply_cached_params(self._current_procedure_name)
        self.structure_changed.emit(cls)

    def _rerender_groups(
        self, cls: type[BaseProcedure], selections: dict[str, Any]
    ) -> None:
        """Diff current vs new groups by key; rebuild only the boxes that changed.

        Args:
            cls: The selected procedure class.
            selections: Current structural-parameter values.
        """
        old_by_key = {group.key: group for group in self._current_groups}
        new_groups = cls.get_param_groups(self._station, selections)
        new_by_key = {group.key: group for group in new_groups}

        # (1) Measurement sub-form: swap the selected VI's params IN PLACE inside
        # the Measurement column. The box, the method drop-down, and every other
        # column keep their widget instances — this is the diff that makes
        # switching the method (Delta ↔ DC) cheap and non-destructive.
        new_meas_key = next(
            (k for k in new_by_key if k.startswith("measurement:")), None
        )
        if (
            self._measurement_box is not None
            and new_meas_key != self._measurement_params_key
        ):
            if self._measurement_params_key is not None:
                old_group = old_by_key.get(self._measurement_params_key)
                if old_group is not None:
                    for name in old_group.params:
                        self._param_inputs.pop(name, None)
            if (
                self._measurement_params_container is not None
                and self._measurement_params_layout is not None
            ):
                self._measurement_params_layout.removeWidget(
                    self._measurement_params_container
                )
                self._measurement_params_container.setParent(None)
                self._measurement_params_container.deleteLater()
                self._measurement_params_container = None
            self._measurement_params_key = None
            if new_meas_key is not None:
                self._install_measurement_params(new_by_key[new_meas_key])

        # (2) Independent columns (System, Reading loop, any generic group)
        # diff by key AND parameter set. The measurement_select /
        # measurement:* keys are NOT here — they live in the Measurement
        # column handled above.
        def _is_column_key(key: str) -> bool:
            return (
                key != "sweep"
                and key != "measurement_select"
                and not key.startswith("measurement:")
            )

        for key in list(self._group_boxes):
            old_group = old_by_key.get(key)
            # A column is stale when its key vanished OR its key persists but
            # its parameter set changed (e.g. the Reading loop group gains or
            # loses the loopN_pick_*/loopN_values fields when a slot's chosen
            # parameter changes). A changed column is dropped here and rebuilt
            # by the append loop below.
            stale = key not in new_by_key or (
                old_group is not None
                and set(old_group.params) != set(new_by_key[key].params)
            )
            if stale:
                box = self._group_boxes.pop(key)
                if old_group is not None:
                    for name in old_group.params:
                        self._param_inputs.pop(name, None)
                if self._param_hbox is not None:
                    self._param_hbox.removeWidget(box)
                box.setParent(None)
                box.deleteLater()

        # A newly-appearing independent column is appended at the end of the row.
        # (Only structural params re-render the form; none of the shipped
        # procedures add/remove an independent column this way, so exact
        # re-insertion order is not exercised — the Measurement swap above is.)
        for key in new_by_key:
            if not _is_column_key(key) or key in self._group_boxes:
                continue
            group = new_by_key[key]
            if key == "reading_loop":
                box = self._build_reading_loop_box(group)
            else:
                box, widgets = param_form.build_group_box(group, wrap=True)
                self._register_group_widgets(group, widgets)
            if self._param_hbox is not None:
                self._param_hbox.addWidget(box)
            self._group_boxes[key] = box

        self._current_groups = new_groups

    # ------------------------------------------------------------------
    # Selection + collection
    # ------------------------------------------------------------------

    def _on_procedure_selected(self, index: int) -> None:
        """Rebuild the parameter form and signal the change when procedure changes.

        Args:
            index: Index of the selected procedure in the dropdown.
        """
        if index < 0 or index >= len(self._procedures):
            return
        # Preserve the outgoing procedure's typed values before rebuilding.
        self.cache_current_params()
        cls = self._procedures[index]
        param_widget = self._build_param_form(cls)
        self._param_scroll.setWidget(param_widget)
        self.structure_changed.emit(cls)
        self._current_procedure_name = getattr(cls, "name", cls.__name__)
        # Re-apply any previously-typed flat values for the incoming procedure.
        self._apply_cached_params(self._current_procedure_name)

    def current_class(self) -> type[BaseProcedure] | None:
        """Return the currently selected procedure class, or None."""
        index = self._proc_selector.currentIndex()
        if index < 0 or index >= len(self._procedures):
            return None
        return self._procedures[index]

    def file_prefix(self) -> str:
        """Return the stripped filename-prefix field value."""
        return self._file_prefix_input.text().strip()

    def collect_values(self) -> dict[str, Any] | None:
        """Read and validate all form inputs into a parameter dict.

        Gathers across ALL currently-rendered groups (Sweep flat params, System,
        the measurement-method selector, and the selected VI's params), rather
        than ``cls.parameters`` — a generic procedure's measurement params are
        station-dependent and never appear in the class's static parameters.

        Returns:
            The merged parameter values on success, or ``None`` if a field
            cannot be parsed (a warning dialog is shown).
        """
        if self.current_class() is None:
            return None

        axis_keys = self._axis_widget.param_keys() if self._axis_widget is not None else set()

        param_values: dict[str, Any] = {}
        for group in self._current_groups:
            for param_name, spec in group.params.items():
                if param_name in axis_keys:
                    continue
                widget = self._param_inputs.get(param_name)
                if widget is None:
                    continue
                try:
                    # param_form owns the ParamSpec -> value mapping (choices ->
                    # mapped value, bool -> checkbox, else parse text by spec.type).
                    param_values[param_name] = param_form.collect_value(widget, spec)
                except (ValueError, TypeError):
                    # Only a text field can fail to parse; choices/bool never raise.
                    raw = widget.text().strip() if hasattr(widget, "text") else ""
                    QMessageBox.warning(
                        self,
                        "Invalid Parameter",
                        f"Cannot parse '{raw}' as {spec.type.__name__} for parameter '{param_name}'.",
                    )
                    return None

        if self._axis_widget is not None:
            try:
                param_values.update(self._axis_widget.get_params())
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid Sweep Parameters", str(exc))
                return None

        return param_values

    # ------------------------------------------------------------------
    # Session persistence (parameter cache + selection)
    # ------------------------------------------------------------------

    def cache_current_params(self) -> None:
        """Store the current form's raw field text, keyed by (group, param).

        Raw text (not validated values) is cached so persistence never triggers
        the "Invalid Parameter" dialog. Keys are ``"{group.key}::{param_name}"``
        so a parameter that exists in more than one structural variant (e.g.
        ``voltmeter_range_V``, a drop-down under the delta VI but a text field
        under the DC VI) is cached separately per variant — switching Delta → DC
        → Delta restores each side's typed value. The cache accumulates across
        selections (entries are never dropped), so a value typed under one
        measurement VI survives switching away and back. The ``measurement_vi``
        selection itself is cached the same way. Sweep-axis widget state is not
        cached to the form (queue entries preserve full params).
        """
        if not (self._current_procedure_name and self._param_inputs):
            return
        cache = self._procedure_params.setdefault(self._current_procedure_name, {})
        for group in self._current_groups:
            for name in group.params:
                widget = self._param_inputs.get(name)
                if widget is not None:
                    cache[f"{group.key}::{name}"] = param_form.get_widget_raw(widget)

    def _apply_cached_params(self, procedure_name: str) -> None:
        """Fill the current form's fields with cached values, keyed by (group, param).

        Restores structural selections first (with signals blocked) and
        re-derives the groups once, so a cached ``measurement_vi`` selects the
        right measurement group before its parameter values are restored —
        without a recursive signal cascade that could cross-contaminate a
        shared parameter name (e.g. ``voltmeter_range_V``) between variants.
        """
        cached = self._procedure_params.get(procedure_name)
        if not cached:
            return

        # 1. Restore structural values (signals blocked), then re-render once so
        #    the correct variant groups are present.
        restored_structural = False
        for group in self._current_groups:
            for name, spec in group.params.items():
                if not spec.structural:
                    continue
                widget = self._param_inputs.get(name)
                raw = cached.get(f"{group.key}::{name}")
                if widget is not None and raw is not None:
                    widget.blockSignals(True)
                    param_form.set_widget_raw(widget, str(raw))
                    widget.blockSignals(False)
                    restored_structural = True
        if restored_structural:
            cls = self.current_class()
            if cls is not None:
                self._rerender_groups(cls, self.current_selections())

        # 2. Restore all values on the now-final set of groups.
        for group in self._current_groups:
            for name in group.params:
                widget = self._param_inputs.get(name)
                raw = cached.get(f"{group.key}::{name}")
                if widget is not None and raw is not None:
                    widget.blockSignals(True)
                    param_form.set_widget_raw(widget, str(raw))
                    widget.blockSignals(False)

    def procedure_by_name(self, name: str) -> type[BaseProcedure] | None:
        """Return the discovered procedure class whose name matches, or None."""
        for cls in self._procedures:
            if getattr(cls, "name", cls.__name__) == name:
                return cls
        return None

    def select_procedure_by_name(self, name: str) -> None:
        """Select ``name`` in the dropdown and rebuild its form."""
        for i, cls in enumerate(self._procedures):
            if getattr(cls, "name", cls.__name__) == name:
                if self._proc_selector.currentIndex() == i:
                    self._on_procedure_selected(i)  # signal won't fire; rebuild directly
                else:
                    self._proc_selector.setCurrentIndex(i)  # fires _on_procedure_selected
                return

    def restore_session(self, session_state: FormAutosaveState) -> None:
        """Apply a loaded session's parameter cache and procedure selection."""
        self._procedure_params = {
            name: dict(values)
            for name, values in session_state.procedure_params.items()
        }
        # Suppress caching the current default values over the restored ones
        # while we switch the selector.
        self._current_procedure_name = ""
        if session_state.selected_procedure:
            self.select_procedure_by_name(session_state.selected_procedure)
        else:
            self._on_procedure_selected(self._proc_selector.currentIndex())

    def export_session_state(self, state: FormAutosaveState) -> None:
        """Write this panel's selection and parameter cache into ``state``."""
        self.cache_current_params()
        state.selected_procedure = self._current_procedure_name
        state.procedure_params = {
            name: dict(values) for name, values in self._procedure_params.items()
        }

    def reset(self) -> None:
        """Clear the cached params and reset the form to defaults."""
        self._procedure_params.clear()
        self._current_procedure_name = ""  # suppress caching stale values
        if self._procedures:
            self._on_procedure_selected(self._proc_selector.currentIndex())
