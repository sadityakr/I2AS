"""InstrumentPanel — auto-generated per-VI monitor panel."""

from __future__ import annotations

import logging
from typing import Any

import qtawesome as qta
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from i2as.core.events import (
    ControlInfo,
    GroupInfo,
    InstrumentInfo,
    LifecycleState,
)
from i2as.core.orchestrator_proxy import OrchestratorProxy
from i2as.core.plan import ParamSpec
from i2as.gui.lifecycle_toggle import ConnectionButton, LifecycleToggleButton
from i2as.gui.param_form import (
    build_param_tooltip,
    build_param_widget,
    collect_value,
)
from i2as.core.status_mirror import StatusMirror
from i2as.gui.theme import BTN_CLASS_DANGER, BTN_CLASS_SECONDARY, TEXT_PRIMARY

logger = logging.getLogger(__name__)

#: The four scalar types a ``ParamSpec`` may declare, keyed by the name the
#: declaration snapshot renders them under. Rebuilding the spec from its JSON
#: is what lets a panel render the typed widget the VI declared without ever
#: touching the VI (the status-mirror standard — see ``gui/README.md``).
_PARAM_TYPES: dict[str, type] = {
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
}


def _spec_from_json(param: dict[str, Any]) -> ParamSpec | None:
    """Rebuild one control parameter's ``ParamSpec`` from its declaration.

    Args:
        param: One entry of ``ControlInfo.params`` — ``name``, ``declared``,
            ``kind``, ``unit``, ``description``, ``default``, ``min``,
            ``max``, ``choices``.

    Returns:
        The reconstructed spec, or ``None`` when the parameter declares none
        (``declared`` false) or names a type no spec can carry — in both
        cases the panel falls back to the plain text input the signature
        alone justifies.
    """
    if not param.get("declared"):
        return None
    param_type = _PARAM_TYPES.get(str(param.get("kind", "")))
    if param_type is None:
        return None
    try:
        return ParamSpec(
            type=param_type,
            default=param.get("default"),
            unit=str(param.get("unit", "")),
            description=str(param.get("description", "")),
            min=param.get("min"),
            max=param.get("max"),
            choices=param.get("choices") or None,
        )
    except (TypeError, ValueError):
        logger.warning(
            "InstrumentPanel: declaration for %r could not be rebuilt as a "
            "ParamSpec; falling back to a plain input",
            param.get("name"),
        )
        return None


class InstrumentPanel(QGroupBox):
    """Auto-generated display + control panel for one Virtual Instrument.

    Created by MonitorWindow for each configured VI. The layout is derived
    entirely from the station's **declaration snapshot** (``InstrumentInfo``
    — GLOSSARY.md's *Station info*), never from the VI object: the panel
    lives on the client side of the boundary and holds nothing the engine
    owns. No hardcoded per-instrument widget lists.

    Args:
        vi_name: The VI's registered name (e.g. ``"temperature"``).
        orchestrator: OrchestratorProxy every action is submitted to, and whose
            ``states_updated`` signal drives value updates.
        mirror: The status mirror this panel reads from — its declaration
            supplies the layout, its snapshot the fault row. Built from the
            engine when none is given (the inline construction path).
        parent: Optional Qt parent widget.
        panel_controls: Optional allowlist of @control method names to show
            (a setup's ``monitor.yaml`` ``panels:`` entry for this VI). When
            None, each control's own declared ``panel=`` default decides.
            Display-only — a hidden control stays fully functional in the
            instrument's front panel.
        show_front_panel_button: Whether the header carries the icon that
            opens the :class:`InstrumentFrontPanel`. False inside the front
            panel itself (a front panel must not open another one).
        type_tag: Optional role label (e.g. "Measurement") shown next
            to the VI name, so non-system cards in the instrument grid are
            recognisable at a glance.
        extra_widget: Optional caller-built widget appended below the control
            rows — the hook for role-specific additions without this class
            learning any per-role logic.
        grouped: Whether to render the VI's declared ``ui_groups`` as titled
            boxes (see ``_build_capability_rows()``). False for the compact
            monitor card, which stays flat by decision — a card is a glance,
            not a workflow — and True for the instrument front panel. A VI
            that declares no groups renders identically either way.
    """

    def __init__(
        self,
        vi_name: str,
        orchestrator: OrchestratorProxy,
        mirror: StatusMirror | None = None,
        parent: QWidget | None = None,
        panel_controls: list[str] | None = None,
        show_front_panel_button: bool = True,
        type_tag: str | None = None,
        extra_widget: QWidget | None = None,
        grouped: bool = False,
    ) -> None:
        super().__init__(parent)  # no native title — see module docstring
        # objectNames are API (gui-edit skill): MonitorWindow finds this card
        # by name when a disconnect swaps it for an offline one.
        self.setObjectName(f"{vi_name}_panel")
        self._vi_name = vi_name
        self._orchestrator = orchestrator
        self._mirror = (
            mirror if mirror is not None else StatusMirror.of(orchestrator)
        )
        # An instrument the declaration does not name renders as an empty
        # card rather than raising: a panel is a report, and reporting must
        # never be the thing that fails.
        self._info = self._mirror.instrument_info(vi_name) or InstrumentInfo(
            name=vi_name
        )
        self._panel_controls = panel_controls
        self._grouped = grouped
        self._show_front_panel_button = show_front_panel_button
        self._type_tag = type_tag
        self._extra_widget = extra_widget
        self._front_panel: QWidget | None = None  # lazily created

        # Maps field name → value label widget
        self._value_labels: dict[str, QLabel] = {}
        # Maps method_name → {param_name → input widget}. Widgets built by
        # param_form when the control declares ParamSpecs (combo/checkbox/
        # line-edit), plain QLineEdits otherwise.
        self._control_inputs: dict[str, dict[str, QWidget]] = {}
        # Maps method_name → {param_name → the declaration entry it was built
        # from}, so submitting reads the same declaration the widget was
        # rendered from instead of asking the VI a second time.
        self._control_params: dict[str, dict[str, dict[str, Any]]] = {}
        # Maps method_name → the control's QPushButton, so a runtime fault
        # can disable every @control row (inputs above, buttons here) in
        # one pass — mirroring OfflineInstrumentPanel's "deliberately
        # control-free" idiom for a VI that DID connect but has since faulted.
        self._control_buttons: dict[str, QPushButton] = {}
        # Current status ("ok"/"stale"/"disconnected"). Drives the QSS
        # `status` property; tracked so styling is only re-applied on change.
        self._status = "ok"
        # Whether the fault row is currently shown — tracked so it is only
        # toggled (and controls only enabled/disabled) on an actual
        # fault/recovery transition, not every tick.
        self._faulted = False

        self._build_layout()

        # Keep panels readable: never let the grid squeeze a panel below the
        # width of its content, and give it a natural minimum height from the
        # assembled layout's sizeHint.
        self.setMinimumWidth(300)
        self.setMinimumHeight(self.sizeHint().height())

        orchestrator.states_updated.connect(self._on_states_updated)
        orchestrator.action_succeeded.connect(self._on_action_succeeded)
        # The lifecycle toggle renders the mirror, so a card built for an
        # instrument that is ALREADY initiated (a reconnect, a window opened
        # mid-experiment) opens showing that, not "Initiate". Refreshed
        # afterwards by on_status_snapshot(), which the owning window calls.
        self._sync_lifecycle()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        outer = QVBoxLayout()
        outer.setSpacing(4)
        outer.setContentsMargins(8, 8, 8, 8)

        # ── Header: name/status label + lifecycle toggle, one compact row ──
        header_row = QHBoxLayout()
        self._name_label = QLabel(f"<b>{self._vi_name}</b>")
        self._name_label.setObjectName(f"{self._vi_name}_name_label")
        self._name_label.setProperty("class", "panel_name_label")
        # The header's elastic element: the two lifecycle controls and the
        # front-panel icon hold their natural widths, so a long VI name in a
        # narrow grid column is what gives way — never a clipped button. The
        # floor keeps enough characters to tell two cards apart; widening the
        # instruments splitter shows the rest.
        self._name_label.setMinimumWidth(70)
        header_row.addWidget(self._name_label)
        if self._type_tag:
            tag_lbl = QLabel(self._type_tag)
            tag_lbl.setObjectName(f"{self._vi_name}_type_tag")
            tag_lbl.setProperty("class", "secondary_label")
            header_row.addWidget(tag_lbl)
        header_row.addStretch()
        if self._show_front_panel_button:
            fp_btn = QPushButton()
            fp_btn.setObjectName(f"{self._vi_name}_front_panel_btn")
            fp_btn.setIcon(qta.icon("fa5s.sliders-h", color=TEXT_PRIMARY))
            fp_btn.setToolTip(
                "Open the instrument front panel (all monitored values and "
                "controls, including ones hidden from this card)"
            )
            fp_btn.clicked.connect(self._open_front_panel)
            header_row.addWidget(fp_btn)
        # The two axes of the connection-lifecycle standard, side by side:
        # Disconnect (who owns the instrument) next to Initiate/Standby
        # (what the instrument is doing). Every card carries both, whatever
        # its role — that uniformity IS the standard.
        self._connection = ConnectionButton(
            self._vi_name,
            "disconnect",
            self._submit_disconnect,
            parent=self,
            compact=True,
        )
        header_row.addWidget(self._connection)
        self._lifecycle = LifecycleToggleButton(self._vi_name, self._submit_lifecycle, parent=self)
        header_row.addWidget(self._lifecycle)
        outer.addLayout(header_row)

        # ── Fault row (runtime fault registry) ────────────────
        # Hidden until _on_states_updated() detects an active fault via
        # Orchestrator.vi_faults() — the RUNTIME sibling of
        # OfflineInstrumentPanel's build-time fault card.
        self._fault_row = QWidget()
        fault_layout = QHBoxLayout(self._fault_row)
        fault_layout.setContentsMargins(0, 2, 0, 2)
        self._fault_label = QLabel("")
        self._fault_label.setObjectName(f"{self._vi_name}_fault_badge")
        self._fault_label.setProperty("class", "secondary_label")
        self._fault_label.setWordWrap(True)
        fault_layout.addWidget(self._fault_label, stretch=1)
        self._ack_fault_btn = QPushButton("Acknowledge")
        self._ack_fault_btn.setObjectName(f"{self._vi_name}_ack_fault_btn")
        self._ack_fault_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._ack_fault_btn.setToolTip("Acknowledge this fault (calms the banner)")
        self._ack_fault_btn.clicked.connect(self._on_acknowledge_fault_clicked)
        fault_layout.addWidget(self._ack_fault_btn)
        self._retry_fault_btn = QPushButton("Retry")
        self._retry_fault_btn.setObjectName(f"{self._vi_name}_retry_fault_btn")
        self._retry_fault_btn.setProperty("class", BTN_CLASS_DANGER)
        self._retry_fault_btn.setToolTip(
            "Reset the error counter and poll this instrument once more"
        )
        self._retry_fault_btn.clicked.connect(self._on_retry_fault_clicked)
        fault_layout.addWidget(self._retry_fault_btn)
        self._fault_row.setVisible(False)
        outer.addWidget(self._fault_row)

        # ── Monitored fields and control methods ──────────────────────
        self._build_capability_rows(outer)

        if self._extra_widget is not None:
            outer.addWidget(self._extra_widget)

        # Absorb any extra vertical space so control rows never expand beyond
        # their natural height when the panel is stretched by the grid.
        outer.addStretch()

        self.setLayout(outer)

    def _build_capability_rows(self, outer: QVBoxLayout) -> None:
        """Lay out this instrument's readings and actions, grouped or flat.

        Two renderings of the same declaration, and which one applies is the
        `grouped` flag, not a property of the instrument:

        * **Flat** (the compact monitor card, and any VI that declares no
          groups): every reading first, then every visible control, both in
          NAME order. That order is what the card has always shown, and a
          VI with no ``ui_groups`` renders here identically either way.
        * **Grouped** (the instrument front panel): one titled ``QGroupBox``
          per declared group, in DECLARED order, holding that group's
          members in the group's own order — which is the workflow order the
          VI author wrote down — followed by whatever no group claims, in
          name order as before.

        A member a group names but the declaration does not carry is
        skipped: a group is presentation, and it must never be able to
        invent a capability.

        Args:
            outer: The panel's outer layout, appended to in place.
        """
        monitored = sorted(info.name for info in self._info.monitored)
        controls = {control.name: control for control in self._info.controls}
        claimed: set[str] = set()

        if self._grouped:
            for group in self._info.ui_groups:
                box = self._build_group_box(group, monitored, controls, claimed)
                if box is not None:
                    outer.addWidget(box)

        for method_name in monitored:
            if method_name not in claimed:
                outer.addLayout(self._build_monitored_row(method_name))

        for name in sorted(controls):
            if name in claimed or not self._control_visible(controls[name]):
                continue
            outer.addWidget(self._build_control_row(controls[name]))

    def _build_group_box(
        self,
        group: GroupInfo,
        monitored: list[str],
        controls: dict[str, ControlInfo],
        claimed: set[str],
    ) -> QGroupBox | None:
        """Build one declared group's titled box, in the group's own order.

        Args:
            group: The group's declaration.
            monitored: Every declared reading name.
            controls: Every declared control, keyed by name.
            claimed: Names already rendered; extended in place with the ones
                this box takes, so the ungrouped pass skips them.

        Returns:
            The assembled box, or ``None`` when the group claims nothing
            this panel renders (every member hidden by a ``panels:``
            allowlist, say) — an empty titled box is noise.
        """
        box = QGroupBox(group.title)
        box.setObjectName(f"{self._vi_name}_group_{group.key}")
        if group.description:
            box.setToolTip(group.description)
        inner = QVBoxLayout(box)
        inner.setSpacing(4)
        inner.setContentsMargins(8, 4, 8, 4)
        members = 0
        for member in group.members:
            if member in monitored:
                inner.addLayout(self._build_monitored_row(member))
            elif member in controls:
                if not self._control_visible(controls[member]):
                    continue
                inner.addWidget(self._build_control_row(controls[member]))
            else:
                logger.warning(
                    "InstrumentPanel: %s group %r names %r, which %s does not "
                    "declare; skipped",
                    self._vi_name,
                    group.key,
                    member,
                    self._vi_name,
                )
                continue
            claimed.add(member)
            members += 1
        if members == 0:
            box.deleteLater()
            return None
        return box

    def _build_monitored_row(self, method_name: str) -> QHBoxLayout:
        """Build one ``@monitored`` reading's label + value row.

        Args:
            method_name: The reading's declared name, which is also the key
                its value arrives under in the state dict.

        Returns:
            The assembled row; its value label is registered for updates.
        """
        row = QHBoxLayout()
        display_name = method_name.replace("_", " ")
        lbl = QLabel(f"{display_name}:")
        lbl.setMinimumWidth(130)
        val = QLabel("—")
        val.setObjectName(f"{self._vi_name}_{method_name}_value")
        val.setProperty("class", "value_readout")
        self._value_labels[method_name] = val
        row.addWidget(lbl)
        row.addWidget(val)
        row.addStretch()
        return row

    @property
    def vi_name(self) -> str:
        """Return the registered name of the VI this card renders."""
        return self._vi_name

    def close_front_panel(self) -> None:
        """Close this VI's front-panel window, if open.

        Called before the card is replaced by an offline one on disconnect —
        the mirror of ``OfflineInstrumentPanel.close_details()``. A front
        panel left open would keep offering controls for an instrument
        I2AS no longer holds.
        """
        if self._front_panel is not None:
            self._front_panel.close()
            self._front_panel = None

    def _open_front_panel(self) -> None:
        """Lazily create and show this VI's full front-panel window."""
        if self._front_panel is None:
            # Local import: InstrumentFrontPanel embeds an InstrumentPanel,
            # so a module-level import would be circular.
            from i2as.gui.instrument_front_panel import InstrumentFrontPanel

            self._front_panel = InstrumentFrontPanel(
                self._vi_name,
                self._orchestrator,
                self._mirror,
                parent=self.window(),
            )
        self._front_panel.show()
        self._front_panel.raise_()
        self._front_panel.activateWindow()

    def _control_visible(self, control: ControlInfo) -> bool:
        """Decide whether one @control appears on this compact card.

        A config allowlist (``panels:`` in ``monitor.yaml``) wins when given;
        otherwise the control's own declared ``panel`` flag decides.

        Args:
            control: The control's declaration.

        Returns:
            True when the control's row should be built.
        """
        if self._panel_controls is not None:
            return control.name in self._panel_controls
        return control.panel

    def _build_param_input(
        self, method_name: str, param: dict[str, Any]
    ) -> tuple[QLabel, QWidget]:
        """Build one parameter's label + input widget.

        A parameter the declaration marks ``declared`` carries a
        ``ParamSpec``, rebuilt here from its JSON (``_spec_from_json()``) and
        handed to ``param_form.build_param_widget`` (drop-down for
        ``choices``, checkbox for ``bool``, else a line edit) with the unit
        in its label and the description/default/range in a tooltip. A
        parameter known only from its signature keeps the legacy plain
        ``QLineEdit`` seeded from that signature's default.

        Args:
            method_name: The @control method the parameter belongs to.
            param: The parameter's entry in ``ControlInfo.params``.

        Returns:
            ``(label, field)``, the field's objectName already set.
        """
        param_name = str(param["name"])
        spec = _spec_from_json(param)
        if spec is not None:
            label_text = (
                f"{param_name} ({spec.unit}):" if spec.unit else f"{param_name}:"
            )
            lbl = QLabel(label_text)
            field = build_param_widget(param_name, spec)
            tooltip = build_param_tooltip(spec)
            lbl.setToolTip(tooltip)
            field.setToolTip(tooltip)
            if isinstance(field, QLineEdit):
                field.setMaximumWidth(90)
        else:
            lbl = QLabel(f"{param_name}:")
            field = QLineEdit()
            field.setPlaceholderText(param_name)
            field.setMaximumWidth(90)
            default = param.get("default")
            if default is not None:
                field.setText(str(default))
        field.setObjectName(f"{self._vi_name}_{method_name}_{param_name}_input")
        return lbl, field

    def _build_control_row(self, control: ControlInfo) -> QWidget:
        """Build one @control method's widget block: button + input widgets.

        Layout scales with the parameter count: up to two parameters stay
        inline next to the button (compact, like ``set_temperature``); more
        stack beneath the button as a labelled grid — one column, or two once
        the count exceeds ten — so a many-knob control (a measurement VI's
        arming action, say) stays editable instead of stretching off-screen in a
        single row.

        Args:
            control: The control's declaration — its name and its parameters
                in signature order.

        Returns:
            A QWidget containing the assembled block.
        """
        method_name = control.name
        container = QWidget()
        # Fixed vertical policy prevents the container from expanding when the
        # parent panel is given extra height, which would make the button thin.
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        btn_label = method_name.replace("_", " ").title()
        btn = QPushButton(btn_label)
        btn.setObjectName(f"{self._vi_name}_{method_name}_btn")

        inputs: dict[str, QWidget] = {}
        widgets = [
            (str(param["name"]), *self._build_param_input(method_name, param))
            for param in control.params
        ]
        for param_name, _lbl, field in widgets:
            inputs[param_name] = field

        if len(widgets) <= 2:
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 2, 0, 2)
            row.addWidget(btn)
            for _param_name, lbl, field in widgets:
                row.addWidget(lbl)
                row.addWidget(field)
            row.addStretch()
        else:
            outer = QVBoxLayout(container)
            outer.setContentsMargins(0, 2, 0, 2)
            outer.setSpacing(2)
            btn_row = QHBoxLayout()
            btn_row.addWidget(btn)
            btn_row.addStretch()
            outer.addLayout(btn_row)
            grid = QGridLayout()
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(2)
            columns = 2 if len(widgets) > 10 else 1
            for idx, (_param_name, lbl, field) in enumerate(widgets):
                grid_row, grid_col = divmod(idx, columns)
                grid.addWidget(lbl, grid_row, grid_col * 2)
                grid.addWidget(field, grid_row, grid_col * 2 + 1)
            # Let the field columns absorb the slack so the grid hugs the left.
            grid.setColumnStretch(columns * 2 - 1, 1)
            outer.addLayout(grid)

        self._control_inputs[method_name] = inputs
        self._control_buttons[method_name] = btn
        self._control_params[method_name] = {
            str(param["name"]): dict(param) for param in control.params
        }

        btn.clicked.connect(
            lambda checked=False, mn=method_name: self._submit_control(mn)
        )
        return container

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def on_status_snapshot(self) -> None:
        """Re-render everything this card reads off a fresh ``StatusSnapshot``.

        Called by the window that owns this card whenever its
        ``StatusMirror`` absorbs a snapshot — once per tick and on every
        state change. The window is the connection receiver on purpose (the
        gui-edit skill's destruction-order rule): Qt severs a receiver's
        connections at the start of its own destruction, so routing a
        tick-rate signal through the window is what keeps it out of a
        partially destroyed child tree.
        """
        self._sync_lifecycle()

    def _sync_lifecycle(self) -> None:
        """Render the Initiate/Standby toggle from the mirror's lifecycle state.

        The lifecycle-state standard's client half (GLOSSARY.md's **Lifecycle
        state**): the card RENDERS what the engine says the instrument is
        doing; it never reconstructs it from the actions it happened to see.
        That is what makes a stand-down nobody clicked — an emergency's
        blanket ``Station.standby_all()``, an agent standing the VI down
        through the gateway, an operator initiating from the CLI — reach this
        card at all. ``set_initiated()`` is a no-op when nothing changed, so
        calling this every snapshot costs no repaint.
        """
        lifecycle = self._mirror.lifecycle_state(self._vi_name)
        self._lifecycle.set_initiated(lifecycle == LifecycleState.INITIATED.value)

    def _on_states_updated(self, full_state: dict) -> None:
        """Update displayed values and border from the station state dict.

        Args:
            full_state: ``{vi_name: {field: value, ...}}`` from Orchestrator.
        """
        vi_state = full_state.get(self._vi_name, {})
        is_stale = vi_state.get("_stale", False)
        is_disconnected = vi_state.get("_disconnected", False)

        for method_name, label in self._value_labels.items():
            value = vi_state.get(method_name)
            if value is None:
                label.setText("—")
            elif isinstance(value, float):
                label.setText(f"{value:.5g}")
            else:
                label.setText(str(value))

        if is_disconnected:
            new_status = "disconnected"
        elif is_stale:
            new_status = "stale"
        else:
            new_status = "ok"

        # Only restyle when the status actually changes. Restyling every tick
        # would force a needless full repolish of the panel each 3 s.
        if new_status != self._status:
            self._status = new_status
            self.setProperty("status", new_status)
            self._name_label.setProperty("status", new_status)
            # Qt only re-evaluates property-based QSS selectors (e.g.
            # QGroupBox[status="stale"]) after an unpolish/polish cycle;
            # setProperty alone does not repaint.
            for widget in (self, self._name_label):
                widget.style().unpolish(widget)
                widget.style().polish(widget)

            if new_status == "disconnected":
                # "NOT RESPONDING", not "DISCONNECTED": since the
                # connection-lifecycle standard made Disconnect an operator
                # verb, that word on a card means "the operator released this
                # instrument" (the offline card's badge). This badge is the
                # opposite situation — I2AS still holds the instrument and
                # it has stopped answering. The underlying
                # Condition/FaultRecord kind stays "disconnected" (the
                # established Instrument-fault vocabulary, see GLOSSARY.md);
                # only the operator-facing wording is disambiguated.
                self._name_label.setText(f"<b>{self._vi_name}</b>  [NOT RESPONDING]")
            elif new_status == "stale":
                self._name_label.setText(f"<b>{self._vi_name}</b>  [stale]")
            else:
                self._name_label.setText(f"<b>{self._vi_name}</b>")

        self._sync_fault_row()

    def _sync_fault_row(self) -> None:
        """Show/hide the fault row and enable/disable controls from the fault registry.

        Visibility follows the Availability standard's unified record
        (``StatusMirror.availability_tags()``, ``i2as.core.availability``):
        the row shows exactly while this VI carries the ``not_responding``
        tag, the same "faulted" state the QSS status border already reflects.
        The row's message still reads the mirror's ``vi_fault()`` for
        ``kind``/``message``/``acknowledged`` — fields the unified record
        does not carry, since acknowledge/retry are comm-specific actions
        (see GLOSSARY.md's **Instrument fault**). Reads every tick, but only
        actually toggles widget state on a fault/recovery TRANSITION —
        matching the existing status-border repolish discipline (never
        restyle unless something changed).
        """
        is_faulted = "not_responding" in self._mirror.availability_tags(self._vi_name)
        if is_faulted != self._faulted:
            self._faulted = is_faulted
            self._fault_row.setVisible(is_faulted)
            for btn in self._control_buttons.values():
                btn.setEnabled(not is_faulted)
            for inputs in self._control_inputs.values():
                for widget in inputs.values():
                    widget.setEnabled(not is_faulted)
        if is_faulted:
            fault = self._mirror.vi_fault(self._vi_name)
            if fault is not None:
                self._fault_label.setText(
                    f"Fault ({fault.get('kind', '')}): {fault.get('message', '')}"
                )
                self._ack_fault_btn.setEnabled(not fault.get("acknowledged", False))

    # ------------------------------------------------------------------
    # Fault row actions (runtime fault registry)
    # ------------------------------------------------------------------

    def _on_acknowledge_fault_clicked(self) -> None:
        """Acknowledge this VI's active fault — calms the Monitor banner."""
        self._orchestrator.acknowledge_fault(self._vi_name)

    def _on_retry_fault_clicked(self) -> None:
        """Retry this VI's active fault: reset the error counter, poll once."""
        self._orchestrator.retry_fault(self._vi_name)

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _submit_control(self, method_name: str) -> None:
        """Read input fields and submit a @control action to the Orchestrator.

        Args:
            method_name: The @control method to call.
        """
        inputs = self._control_inputs.get(method_name, {})
        declarations = self._control_params.get(method_name, {})

        kwargs: dict[str, Any] = {}
        for param_name, field in inputs.items():
            declaration = declarations.get(param_name, {})
            spec = _spec_from_json(declaration)
            if spec is not None:
                # Spec-built widget: an emptied line edit falls back to the
                # method's own default (a ParamSpec always declares one); an
                # unparseable entry aborts the submit with an explicit verdict
                # instead of sending a wrong-typed value onward.
                if isinstance(field, QLineEdit) and not field.text().strip():
                    continue
                try:
                    kwargs[param_name] = collect_value(field, spec)
                except (ValueError, TypeError):
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self,
                        "Invalid Parameter",
                        f"'{field.text().strip()}' is not a valid "
                        f"{spec.type.__name__} for '{param_name}' "
                        f"({method_name}).",
                    )
                    return
                continue

            raw = field.text().strip()
            param_type = _PARAM_TYPES.get(str(declaration.get("kind", "")), str)
            has_default = declaration.get("default") is not None

            if not raw:
                if not has_default:
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self,
                        "Missing Parameter",
                        f"'{param_name}' is required for {method_name}.",
                    )
                    return
                continue  # omit; let the method use its own default

            try:
                kwargs[param_name] = param_type(raw)
            except (ValueError, TypeError):
                logger.warning(
                    "InstrumentPanel: could not coerce '%s' to %s for param '%s'",
                    raw,
                    param_type,
                    param_name,
                )
                kwargs[param_name] = raw

        self._orchestrator.submit_vi_action(self._vi_name, method_name, **kwargs)

    def _submit_lifecycle(self, action: str) -> None:
        """Submit an initiate or standby action for this VI.

        Args:
            action: ``"initiate"`` or ``"standby"``.
        """
        self._orchestrator.submit_vi_action(self._vi_name, action)

    def _submit_disconnect(self) -> None:
        """Ask the Orchestrator to release this instrument to its front panel.

        Runs through ``Orchestrator.disconnect_instrument()`` rather than the
        GUI action queue: a disconnect removes the VI from the live registry,
        which the queue (built to dispatch to *registered* VIs) cannot express.
        The verdict arrives on ``instrument_disconnected`` / ``action_failed``,
        and MonitorWindow — not this panel — swaps the card, because a card
        cannot replace itself.
        """
        self._orchestrator.disconnect_instrument(self._vi_name)

    def _on_action_succeeded(self, vi_name: str, method_name: str) -> None:
        """Flip the lifecycle toggle the moment this card's own action lands.

        An OPTIMISTIC flip, no longer the truth: the toggle's state is the
        lifecycle the ``StatusSnapshot`` carries (see ``_sync_lifecycle()``),
        and the next snapshot confirms this or corrects it. It is kept
        because a verdict arrives before the snapshot that reports its
        consequence, and a button that answers the click it just received is
        worth one tick of optimism.

        Args:
            vi_name: The VI the confirmed action was submitted for.
            method_name: The confirmed method name.
        """
        if vi_name != self._vi_name:
            return
        if method_name == "initiate":
            self._lifecycle.set_initiated(True)
        elif method_name == "standby":
            self._lifecycle.set_initiated(False)
