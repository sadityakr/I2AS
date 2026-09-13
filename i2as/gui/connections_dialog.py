"""The **Connections dialog** — turning the Agent gateway on and off from the GUI.

One modal form over a ``GatewayController`` (``i2as/session/gateway/controller.py``),
opened from the Monitor window's Connections menu. Unlike the eLab setup
dialog it sits beside, this one has a live side effect on Save: the gateway
is actually started or stopped right there, not merely written to a
settings file for the next launch to pick up — though it IS also written to
``i2as/gui/app_settings.py`` (``gateway_enabled`` / ``gateway_max_role``) so
the choice survives a restart too.

**The role selector never shows more than the ceiling.** ``controller.allowed_roles()``
is monitor.yaml's ``gateway_max_role`` filtered down by
``role_within_ceiling()`` — this dialog cannot construct a choice the
controller would refuse, so Save never fails for exceeding the ceiling.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from i2as.gui import app_settings
from i2as.gui.theme import BTN_CLASS_PRIMARY, BTN_CLASS_SECONDARY

logger = logging.getLogger(__name__)

#: How often the connected-agents list refreshes while the dialog is open.
_REFRESH_INTERVAL_MS = 2000


class ConnectionsDialog(QDialog):
    """The Connections dialog: on/off and role ceiling for the Agent gateway.

    Named widgets (``findChild`` objectNames are API): the enabled toggle
    ``connections_enabled_checkbox``, the role selector
    ``connections_role_combo``, the read-only descriptor line
    ``connections_descriptor_label``, the connected-agents list
    ``connections_list``, the ``connections_save_btn`` /
    ``connections_close_btn`` buttons and the ``connections_status_label``.

    Args:
        controller: The ``GatewayController`` this dialog edits and reads
            live connection state from.
        parent: Optional Qt parent widget.
    """

    def __init__(self, controller: Any, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connections")
        self.setObjectName("connections_dialog")
        self.setMinimumWidth(480)
        self._controller = controller

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.addWidget(self._build_gateway_group())
        root.addWidget(self._build_agents_group())

        self._status_label = QLabel("")
        self._status_label.setObjectName("connections_status_label")
        self._status_label.setProperty("class", "secondary_label")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        root.addLayout(self._build_buttons())

        self._load_from_controller()
        self._refresh_agents()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(_REFRESH_INTERVAL_MS)
        self._refresh_timer.timeout.connect(self._refresh_agents)
        self._refresh_timer.start()

    # ── Layout ────────────────────────────────────────────────────────

    def _build_gateway_group(self) -> QGroupBox:
        group = QGroupBox("Agent gateway")
        form = QFormLayout(group)

        self._enabled_checkbox = QCheckBox("Accept agent connections")
        self._enabled_checkbox.setObjectName("connections_enabled_checkbox")
        self._enabled_checkbox.setToolTip(
            "Whether this running app listens for out-of-process agents at all"
        )
        form.addRow(self._enabled_checkbox)

        self._role_combo = QComboBox()
        self._role_combo.setObjectName("connections_role_combo")
        self._role_combo.setToolTip(
            "The most authority a connecting agent may claim — capped by "
            "this setup's monitor.yaml and never adjustable above it"
        )
        for role in self._controller.allowed_roles():
            self._role_combo.addItem(role.value, role)
        form.addRow("Role ceiling:", self._role_combo)

        self._descriptor_label = QLabel("—")
        self._descriptor_label.setObjectName("connections_descriptor_label")
        self._descriptor_label.setWordWrap(True)
        self._descriptor_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("Descriptor:", self._descriptor_label)

        return group

    def _build_agents_group(self) -> QGroupBox:
        group = QGroupBox("Connected agents")
        layout = QVBoxLayout(group)
        self._agents_list = QListWidget()
        self._agents_list.setObjectName("connections_list")
        self._agents_list.setToolTip("Every session past its hello, right now")
        layout.addWidget(self._agents_list)
        return group

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch(1)

        close_btn = QPushButton("Close")
        close_btn.setObjectName("connections_close_btn")
        close_btn.setProperty("class", BTN_CLASS_SECONDARY)
        close_btn.clicked.connect(self.reject)
        row.addWidget(close_btn)

        save_btn = QPushButton("Save")
        save_btn.setObjectName("connections_save_btn")
        save_btn.setProperty("class", BTN_CLASS_PRIMARY)
        save_btn.clicked.connect(self._on_save)
        row.addWidget(save_btn)

        return row

    # ── State ─────────────────────────────────────────────────────────

    def _load_from_controller(self) -> None:
        """Show the controller's live state, not a stale persisted guess."""
        self._enabled_checkbox.setChecked(self._controller.enabled)
        current = self._controller.current_role
        if current is not None:
            index = self._role_combo.findData(current)
            if index >= 0:
                self._role_combo.setCurrentIndex(index)
        self._update_descriptor_label()

    def _update_descriptor_label(self) -> None:
        server = self._controller.server
        if server is None:
            self._descriptor_label.setText("gateway is off")
        else:
            self._descriptor_label.setText(str(server.descriptor))

    def _refresh_agents(self) -> None:
        """Repaint the connected-agents list from the controller, live."""
        self._agents_list.clear()
        for connection in self._controller.connections():
            item = QListWidgetItem(f"{connection['actor_id']}  ({connection['role']})")
            self._agents_list.addItem(item)
        if not self._controller.connections():
            placeholder = QListWidgetItem("(none connected)")
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._agents_list.addItem(placeholder)

    # ── Save ──────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        """Apply the form live, then persist it for the next launch."""
        enabled = self._enabled_checkbox.isChecked()
        role = self._role_combo.currentData()

        try:
            if enabled:
                self._controller.start(role)
            else:
                self._controller.stop()
        except ValueError as error:
            logger.exception("Connections dialog could not apply the Gateway settings")
            self._status_label.setText(f"Could not apply: {error}")
            return

        app_settings.set_gateway_enabled(enabled)
        if role is not None:
            app_settings.set_gateway_max_role(role.value)

        self._update_descriptor_label()
        self._refresh_agents()
        self._status_label.setText(
            "Gateway is on." if enabled else "Gateway is off."
        )
