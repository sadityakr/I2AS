"""The **Connections dialog** — the Agent gateway, and its reach, from the GUI.

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

**Remote access is a second group on the same form.** It turns the **HTTP
MCP endpoint** on and off, chooses where it binds, records the public URL
the operator's own tunnel or proxy hands out (I2AS runs no tunnel: what
forwards to this machine is the operator's choice and their account), and
issues and revokes the **access keys** a web client presents. A key's
secret is shown exactly once, in the box that creates it. The "client
config" box renders, for the selected client, the URL and key in the
shape that client's settings expect — so configuring a client is a paste.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from i2as.gui import app_settings
from i2as.gui.theme import BTN_CLASS_DANGER, BTN_CLASS_PRIMARY, BTN_CLASS_SECONDARY

logger = logging.getLogger(__name__)

#: How often the connected-agents list refreshes while the dialog is open.
_REFRESH_INTERVAL_MS = 2000

#: The bind addresses the dialog offers, label → address.
BIND_CHOICES: tuple[tuple[str, str], ...] = (
    ("This computer only (127.0.0.1)", "127.0.0.1"),
    ("Every network interface (0.0.0.0)", "0.0.0.0"),
)

#: The clients the config box knows how to render for.
CLIENT_CHOICES: tuple[str, ...] = (
    "Claude Code (.mcp.json)",
    "ChatGPT (developer mode connector)",
    "Open WebUI (external tool)",
    "Generic (URL and header)",
)

#: What the config box shows in place of a secret it no longer has.
KEY_PLACEHOLDER = "<your key>"


def render_client_config(client: str, url: str, key: str = KEY_PLACEHOLDER) -> str:
    """Render one client's configuration for the endpoint.

    Pure, so a test can check the text without a window.

    Args:
        client: One of ``CLIENT_CHOICES``.
        url: The endpoint URL — the public one when there is one.
        key: The secret, or the placeholder when it is not at hand.

    Returns:
        The text to paste into the client.
    """
    base = url.rstrip("/")
    if client.startswith("Claude Code"):
        document = {
            "mcpServers": {
                "i2as": {
                    "type": "http",
                    "url": base,
                    "headers": {"Authorization": f"Bearer {key}"},
                }
            }
        }
        return json.dumps(document, indent=2)
    if client.startswith("ChatGPT"):
        return (
            "ChatGPT connectors cannot send a header, so the key travels in the URL.\n"
            "Settings → Apps & Connectors → Create (developer mode on):\n"
            f"  MCP server URL:   {base}/{key}\n"
            "  Authentication:   No authentication\n"
            "Anyone holding that URL holds the key: share it as you would a password."
        )
    if client.startswith("Open WebUI"):
        return (
            "Admin Settings → External Tools → Add (type: MCP, Streamable HTTP):\n"
            f"  URL:     {base}\n"
            "  Auth:    Bearer\n"
            f"  Token:   {key}"
        )
    return f"URL:     {base}\nHeader:  Authorization: Bearer {key}"


class ConnectionsDialog(QDialog):
    """The Connections dialog: on/off and role ceiling for the Agent gateway,
    plus the HTTP endpoint and its access keys.

    Named widgets (``findChild`` objectNames are API): the enabled toggle
    ``connections_enabled_checkbox``, the role selector
    ``connections_role_combo``, the read-only descriptor line
    ``connections_descriptor_label``, the connected-agents list
    ``connections_list``, the ``connections_save_btn`` /
    ``connections_close_btn`` buttons and the ``connections_status_label``.
    Remote access adds ``connections_remote_checkbox``,
    ``connections_bind_combo``, ``connections_port_spin``,
    ``connections_public_url_edit``, ``connections_local_url_label``,
    ``connections_keys_list``, ``connections_new_key_btn``,
    ``connections_revoke_key_btn``, ``connections_client_combo`` and
    ``connections_client_config``.

    Args:
        controller: The ``GatewayController`` this dialog edits and reads
            live connection state from.
        parent: Optional Qt parent widget.
    """

    def __init__(self, controller: Any, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connections")
        self.setObjectName("connections_dialog")
        self.setMinimumWidth(560)
        self._controller = controller
        #: The secret of the key created most recently in THIS dialog, so
        #: the config box can render it once; never persisted.
        self._last_secret: str | None = None

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.addWidget(self._build_gateway_group())
        root.addWidget(self._build_remote_group())
        root.addWidget(self._build_agents_group())

        self._status_label = QLabel("")
        self._status_label.setObjectName("connections_status_label")
        self._status_label.setProperty("class", "secondary_label")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        root.addLayout(self._build_buttons())

        self._load_from_controller()
        self._refresh_agents()
        self._refresh_keys()
        self._refresh_urls()
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

    def _build_remote_group(self) -> QGroupBox:
        group = QGroupBox("Remote access (MCP over HTTP)")
        layout = QVBoxLayout(group)
        form = QFormLayout()
        layout.addLayout(form)

        self._remote_checkbox = QCheckBox("Serve the gateway over HTTP")
        self._remote_checkbox.setObjectName("connections_remote_checkbox")
        self._remote_checkbox.setToolTip(
            "Publish the same tools at a URL, for clients that cannot launch "
            "python -m i2as.mcp themselves — a web assistant, a tunnel, another PC"
        )
        form.addRow(self._remote_checkbox)

        self._bind_combo = QComboBox()
        self._bind_combo.setObjectName("connections_bind_combo")
        self._bind_combo.setToolTip(
            "Where to listen. This computer only is enough for a tunnel agent "
            "running here; every interface makes the lab network able to reach it"
        )
        for label, address in BIND_CHOICES:
            self._bind_combo.addItem(label, address)
        form.addRow("Listen on:", self._bind_combo)

        self._port_spin = QSpinBox()
        self._port_spin.setObjectName("connections_port_spin")
        self._port_spin.setRange(1, 65535)
        self._port_spin.setToolTip("The TCP port; any free port works")
        form.addRow("Port:", self._port_spin)

        self._local_url_label = QLabel("—")
        self._local_url_label.setObjectName("connections_local_url_label")
        self._local_url_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("Local URL:", self._local_url_label)

        self._public_url_edit = QLineEdit()
        self._public_url_edit.setObjectName("connections_public_url_edit")
        self._public_url_edit.setPlaceholderText("https://… (what your tunnel or proxy forwards to the local URL)")
        self._public_url_edit.setToolTip(
            "Optional. I2AS runs no tunnel: start ngrok, cloudflared, Tailscale "
            "or a reverse proxy yourself, point it at the local URL, and paste the "
            "address it gives you here so client configs use it"
        )
        self._public_url_edit.textChanged.connect(lambda _text: self._refresh_urls())
        form.addRow("Public URL:", self._public_url_edit)

        keys_row = QHBoxLayout()
        self._keys_list = QListWidget()
        self._keys_list.setObjectName("connections_keys_list")
        self._keys_list.setToolTip("Every access key issued; a key is an agent with a role")
        self._keys_list.setMaximumHeight(96)
        self._keys_list.currentRowChanged.connect(lambda _row: self._refresh_urls())
        keys_row.addWidget(self._keys_list, 1)
        key_buttons = QVBoxLayout()
        self._new_key_btn = QPushButton("New key…")
        self._new_key_btn.setObjectName("connections_new_key_btn")
        self._new_key_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._new_key_btn.clicked.connect(self._on_new_key)
        key_buttons.addWidget(self._new_key_btn)
        self._revoke_key_btn = QPushButton("Revoke")
        self._revoke_key_btn.setObjectName("connections_revoke_key_btn")
        self._revoke_key_btn.setProperty("class", BTN_CLASS_DANGER)
        self._revoke_key_btn.clicked.connect(self._on_revoke_key)
        key_buttons.addWidget(self._revoke_key_btn)
        key_buttons.addStretch(1)
        keys_row.addLayout(key_buttons)
        form.addRow("Access keys:", keys_row)

        client_row = QHBoxLayout()
        self._client_combo = QComboBox()
        self._client_combo.setObjectName("connections_client_combo")
        for client in CLIENT_CHOICES:
            self._client_combo.addItem(client)
        self._client_combo.currentIndexChanged.connect(lambda _index: self._refresh_urls())
        client_row.addWidget(self._client_combo, 1)
        copy_btn = QPushButton("Copy")
        copy_btn.setObjectName("connections_copy_config_btn")
        copy_btn.setProperty("class", BTN_CLASS_SECONDARY)
        copy_btn.clicked.connect(self._on_copy_config)
        client_row.addWidget(copy_btn)
        form.addRow("Client config:", client_row)

        self._client_config = QPlainTextEdit()
        self._client_config.setObjectName("connections_client_config")
        self._client_config.setReadOnly(True)
        self._client_config.setMaximumHeight(120)
        layout.addWidget(self._client_config)

        return group

    def _build_agents_group(self) -> QGroupBox:
        group = QGroupBox("Connected agents")
        layout = QVBoxLayout(group)
        self._agents_list = QListWidget()
        self._agents_list.setObjectName("connections_list")
        self._agents_list.setToolTip("Every session past its hello, right now")
        self._agents_list.setMaximumHeight(96)
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

        http_server = getattr(self._controller, "http_server", None)
        self._remote_checkbox.setChecked(http_server is not None)
        host = http_server.host if http_server is not None else app_settings.remote_access_host()
        index = self._bind_combo.findData(host)
        self._bind_combo.setCurrentIndex(index if index >= 0 else 0)
        self._port_spin.setValue(
            http_server.port if http_server is not None else app_settings.remote_access_port()
        )
        public_url = (
            http_server.public_url if http_server is not None else app_settings.remote_access_public_url()
        )
        self._public_url_edit.setText(public_url or "")

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

    def _refresh_keys(self) -> None:
        """Repaint the keys list from the controller."""
        selected = self._selected_key_name()
        self._keys_list.clear()
        for key in self._controller.keys():
            item = QListWidgetItem(f"{key.name}  ({key.role})  {key.hint}…")
            item.setData(Qt.ItemDataRole.UserRole, key.name)
            self._keys_list.addItem(item)
            if key.name == selected:
                self._keys_list.setCurrentItem(item)
        if self._keys_list.count() == 0:
            placeholder = QListWidgetItem("(no keys yet — a client needs one to connect)")
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._keys_list.addItem(placeholder)
        self._refresh_urls()

    def _selected_key_name(self) -> str | None:
        item = self._keys_list.currentItem()
        if item is None:
            return None
        name = item.data(Qt.ItemDataRole.UserRole)
        return str(name) if name else None

    def _endpoint_url(self) -> str:
        """The URL a client should use: the public one when given, else local."""
        public = self._public_url_edit.text().strip()
        if public:
            return public if public.rstrip("/").endswith("/mcp") else public.rstrip("/") + "/mcp"
        host = self._bind_combo.currentData() or "127.0.0.1"
        host = "127.0.0.1" if host == "0.0.0.0" else host
        return f"http://{host}:{self._port_spin.value()}/mcp"

    def _refresh_urls(self) -> None:
        """Re-render the local URL line and the client config box."""
        host = self._bind_combo.currentData() or "127.0.0.1"
        host = "127.0.0.1" if host == "0.0.0.0" else host
        self._local_url_label.setText(f"http://{host}:{self._port_spin.value()}/mcp")
        key = self._last_secret or KEY_PLACEHOLDER
        self._client_config.setPlainText(
            render_client_config(self._client_combo.currentText(), self._endpoint_url(), key)
        )

    # ── Keys ──────────────────────────────────────────────────────────

    def _on_new_key(self) -> None:
        """Ask for a name and a role, issue the key, and show its secret once."""
        name, ok = QInputDialog.getText(
            self, "New access key", "Name for this key (also its actor id, e.g. chatgpt):"
        )
        if not ok or not name.strip():
            return
        roles = [role.value for role in self._controller.allowed_roles()]
        role, ok = QInputDialog.getItem(
            self, "New access key", "Role this key connects with:", roles, len(roles) - 1, False
        )
        if not ok:
            return
        try:
            key, secret = self._controller.create_key(name.strip(), role)
        except (ValueError, RuntimeError) as error:
            self._status_label.setText(f"Could not create the key: {error}")
            return
        self._last_secret = secret
        self._refresh_keys()
        for row in range(self._keys_list.count()):
            item = self._keys_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == key.name:
                self._keys_list.setCurrentItem(item)
        self._refresh_urls()
        QApplication.clipboard().setText(secret)
        QMessageBox.information(
            self,
            "Access key created",
            f"Key '{key.name}' (role {key.role}) is copied to the clipboard and shown "
            "below — it will not be shown again.\n\n"
            f"{secret}\n\n"
            "The client config box now includes it.",
        )
        self._status_label.setText(f"Key '{key.name}' created.")

    def _on_revoke_key(self) -> None:
        """Delete the selected key after a confirmation."""
        name = self._selected_key_name()
        if name is None:
            self._status_label.setText("Select a key to revoke.")
            return
        answer = QMessageBox.question(
            self,
            "Revoke access key",
            f"Revoke '{name}'? A client holding it loses the station immediately.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self._controller.revoke_key(name):
            self._status_label.setText(f"Key '{name}' revoked.")
        self._last_secret = None
        self._refresh_keys()

    def _on_copy_config(self) -> None:
        QApplication.clipboard().setText(self._client_config.toPlainText())
        self._status_label.setText("Client config copied to the clipboard.")

    # ── Save ──────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        """Apply the form live, then persist it for the next launch."""
        enabled = self._enabled_checkbox.isChecked()
        role = self._role_combo.currentData()
        remote = self._remote_checkbox.isChecked() and enabled
        host = str(self._bind_combo.currentData() or "127.0.0.1")
        port = int(self._port_spin.value())
        public_url = self._public_url_edit.text().strip() or None

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

        remote_note = ""
        try:
            if remote:
                self._controller.start_http(host=host, port=port, public_url=public_url)
            else:
                self._controller.stop_http()
        except (OSError, RuntimeError) as error:
            logger.exception("Connections dialog could not apply the remote-access settings")
            remote_note = f" Remote access could not start: {error}"
            remote = False
        app_settings.set_remote_access_enabled(remote)
        app_settings.set_remote_access_host(host)
        app_settings.set_remote_access_port(port)
        app_settings.set_remote_access_public_url(public_url)

        self._update_descriptor_label()
        self._refresh_agents()
        self._refresh_urls()
        summary = "Gateway is on." if enabled else "Gateway is off."
        if enabled:
            summary += " Remote access is on." if remote else " Remote access is off."
        self._status_label.setText(summary + remote_note)
