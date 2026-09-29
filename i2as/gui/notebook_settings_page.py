"""The **Electronic notebook page** — one person's notebook accounts and blocks.

The Settings dialog's page over the logged-in user's own profile
(``i2as.session.user_profile``): whether they publish, their notebook account
(which connector, and that connector's settings — the form is RENDERED from
the connector block's ``settings_schema``, so a connector a user wrote gets
its own form with no GUI code), the API key (stored in the system keyring,
never shown, never pre-filled), the profile and template a new experiment
uses, and the list of installed blocks with a button to open the folder
where a user's own blocks live.

Test connection and Fetch templates go through the notebook service, on its
worker thread: the page never waits on the network.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from i2as.blocks.discovery import BLOCK_KINDS, KIND_CONNECTOR, KIND_DIRS, KIND_PROFILE
from i2as.gui.theme import BTN_CLASS_PRIMARY, BTN_CLASS_SECONDARY
from i2as.session.credentials import SCOPE_ELN, credential_key
from i2as.session.user_profile import ElnAccount, ElnUserSettings

logger = logging.getLogger(__name__)

#: Placeholder of the API-key field (it is never pre-filled).
KEY_PLACEHOLDER = "leave blank to keep the stored key"

#: The one account this page edits (a user may add more by editing the profile).
DEFAULT_ACCOUNT_ID = "lab"

#: The account form's first row for the connector's own settings (after the
#: switch and the connector choice).
_FIRST_FIELD_ROW = 2


class NotebookSettingsPage(QWidget):
    """The Electronic notebook page.

    Named widgets: ``settings_eln_enabled_checkbox``,
    ``settings_eln_connector_combo``, ``settings_eln_field_<name>`` (one per
    connector setting), ``settings_eln_key_edit``, ``settings_eln_key_label``,
    ``settings_eln_profile_combo``, ``settings_eln_template_combo``,
    ``settings_eln_test_btn``, ``settings_eln_templates_btn``,
    ``settings_eln_blocks_text``, ``settings_eln_open_blocks_btn``,
    ``settings_eln_save_btn``, ``settings_eln_status_label``.

    Args:
        profiles: The ``UserProfileStore``.
        credentials: The ``CredentialStore``.
        user_id: Whose profile this page edits.
        catalog: The ``BlockCatalog``.
        service: The ``ElnService`` (Test / Fetch templates), or ``None``.
        parent: Optional Qt parent.
    """

    def __init__(
        self,
        profiles: Any,
        credentials: Any,
        user_id: str,
        catalog: Any,
        service: Any | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("notebook_settings_page")
        self._profiles = profiles
        self._credentials = credentials
        self._user_id = user_id or "guest"
        self._catalog = catalog
        self._service = service
        self._field_widgets: dict[str, QWidget] = {}
        self._settings = profiles.load(self._user_id).eln

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        # The groups scroll rather than squeeze: a connector may declare more
        # settings than the dialog has room for.
        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)
        who = QLabel(f"Notebook settings of <b>{self._user_id}</b> (saved in their own profile).")
        who.setProperty("class", "secondary_label")
        body.addWidget(who)
        body.addWidget(self._build_account_group())
        body.addWidget(self._build_defaults_group())
        body.addWidget(self._build_blocks_group())
        body.addStretch(1)
        scroll = QScrollArea()
        scroll.setObjectName("settings_eln_scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        root.addWidget(scroll, stretch=1)

        self._status = QLabel("")
        self._status.setObjectName("settings_eln_status_label")
        self._status.setProperty("class", "secondary_label")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        row = QHBoxLayout()
        row.addStretch(1)
        save = QPushButton("Save")
        save.setObjectName("settings_eln_save_btn")
        save.setProperty("class", BTN_CLASS_PRIMARY)
        save.clicked.connect(self.save)
        row.addWidget(save)
        root.addLayout(row)
        self._load()

    # ── Layout ────────────────────────────────────────────────────────

    def _build_account_group(self) -> QGroupBox:
        group = QGroupBox("Notebook account")
        layout = QVBoxLayout(group)
        # ONE form, so the connector's own settings line up with the rest:
        # rows 0-1 are fixed, the connector's rows follow, the key comes last.
        top = QFormLayout()
        self._form = top
        self._enabled = QCheckBox("Publish finished runs to my notebook")
        self._enabled.setObjectName("settings_eln_enabled_checkbox")
        top.addRow(self._enabled)
        self._connector = QComboBox()
        self._connector.setObjectName("settings_eln_connector_combo")
        for block_id, info in sorted(self._catalog.all(KIND_CONNECTOR).items()):
            label = f"{info.display_name} ({info.source})" if info.usable else f"{block_id} — unusable: {info.error}"
            self._connector.addItem(label, block_id)
        self._connector.currentIndexChanged.connect(self._rebuild_fields)
        top.addRow("Connector:", self._connector)
        layout.addLayout(top)
        key_form = top
        self._key = QLineEdit()
        self._key.setObjectName("settings_eln_key_edit")
        self._key.setEchoMode(QLineEdit.EchoMode.Password)
        self._key.setPlaceholderText(KEY_PLACEHOLDER)
        key_form.addRow("API key:", self._key)
        self._key_label = QLabel("")
        self._key_label.setObjectName("settings_eln_key_label")
        self._key_label.setProperty("class", "secondary_label")
        key_form.addRow("", self._key_label)
        buttons = QHBoxLayout()
        self._test_btn = QPushButton("Save && test connection")
        self._test_btn.setObjectName("settings_eln_test_btn")
        self._test_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._test_btn.clicked.connect(self.test_connection)
        self._test_btn.setEnabled(self._service is not None)
        buttons.addWidget(self._test_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return group

    def _build_defaults_group(self) -> QGroupBox:
        group = QGroupBox("New experiments")
        form = QFormLayout(group)
        self._profile = QComboBox()
        self._profile.setObjectName("settings_eln_profile_combo")
        for block_id, info in sorted(self._catalog.all(KIND_PROFILE).items()):
            self._profile.addItem(f"{block_id} ({info.source})", block_id)
        form.addRow("Profile:", self._profile)
        template_row = QHBoxLayout()
        self._template = QComboBox()
        self._template.setObjectName("settings_eln_template_combo")
        self._template.setEditable(True)
        self._template.setToolTip("The template a new page is created from, when the profile names none.")
        template_row.addWidget(self._template, stretch=1)
        self._templates_btn = QPushButton("Fetch templates")
        self._templates_btn.setObjectName("settings_eln_templates_btn")
        self._templates_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._templates_btn.clicked.connect(self.fetch_templates)
        self._templates_btn.setEnabled(self._service is not None)
        template_row.addWidget(self._templates_btn)
        form.addRow("Template:", template_row)
        return group

    def _build_blocks_group(self) -> QGroupBox:
        group = QGroupBox("Blocks (connectors, renderers, profiles)")
        layout = QVBoxLayout(group)
        hint = QLabel(
            "Your own blocks live in the folder below; check one with "
            "<code>python -m i2as.blocks check &lt;file&gt;</code>. A block runs in a "
            "helper process, never inside the station."
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "secondary_label")
        layout.addWidget(hint)
        self._blocks_text = QPlainTextEdit()
        self._blocks_text.setObjectName("settings_eln_blocks_text")
        self._blocks_text.setReadOnly(True)
        self._blocks_text.setMaximumHeight(110)
        layout.addWidget(self._blocks_text)
        row = QHBoxLayout()
        open_btn = QPushButton("Open my blocks folder")
        open_btn.setObjectName("settings_eln_open_blocks_btn")
        open_btn.setProperty("class", BTN_CLASS_SECONDARY)
        open_btn.clicked.connect(self.open_blocks_folder)
        row.addWidget(open_btn)
        row.addStretch(1)
        layout.addLayout(row)
        self._refresh_blocks()
        return group

    # ── Form ↔ profile ────────────────────────────────────────────────

    def _account(self) -> ElnAccount:
        return self._settings.account(DEFAULT_ACCOUNT_ID) or self._settings.account() or ElnAccount(account_id=DEFAULT_ACCOUNT_ID, connector=str(self._connector.currentData() or "elabftw"))

    def _load(self) -> None:
        account = self._account()
        self._enabled.setChecked(self._settings.enabled)
        index = self._connector.findData(account.connector)
        self._connector.setCurrentIndex(max(index, 0))
        self._rebuild_fields(values=account.settings)
        index = self._profile.findData(self._settings.default_profile)
        self._profile.setCurrentIndex(max(index, 0))
        if self._settings.default_template:
            self._template.addItem(self._settings.default_template, self._settings.default_template)
            self._template.setCurrentIndex(0)
        self._refresh_key_label()

    def _schema(self) -> dict[str, Any]:
        info = self._catalog.all(KIND_CONNECTOR).get(str(self._connector.currentData() or ""))
        return dict(info.settings_schema) if info is not None else {}

    def _rebuild_fields(self, *_args: Any, values: dict[str, Any] | None = None) -> None:
        """Render the connector's settings form from its ``settings_schema``."""
        current = values if values is not None else self.connector_settings()
        for _ in range(len(self._field_widgets)):
            self._form.removeRow(_FIRST_FIELD_ROW)
        self._field_widgets.clear()
        row = _FIRST_FIELD_ROW
        for name, spec in (self._schema().get("properties") or {}).items():
            if not isinstance(spec, dict):
                continue
            kind = spec.get("type")
            value = current.get(name, spec.get("default"))
            widget: QWidget
            if kind == "boolean":
                box = QCheckBox()
                box.setChecked(bool(value))
                widget = box
            elif kind in ("number", "integer"):
                spin = QDoubleSpinBox() if kind == "number" else QSpinBox()
                spin.setRange(-1e9 if kind == "number" else -(2**31), 1e9 if kind == "number" else 2**31 - 1)
                try:
                    spin.setValue(float(value) if kind == "number" else int(value))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    pass
                widget = spin
            else:
                edit = QLineEdit(str(value or ""))
                widget = edit
            widget.setObjectName(f"settings_eln_field_{name}")
            widget.setToolTip(str(spec.get("description") or ""))
            self._field_widgets[name] = widget
            self._form.insertRow(row, f"{spec.get('title') or name}:", widget)
            row += 1

    def connector_settings(self) -> dict[str, Any]:
        """Return the connector settings as the form shows them."""
        values: dict[str, Any] = {}
        for name, widget in self._field_widgets.items():
            if isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                values[name] = widget.value()
            elif isinstance(widget, QLineEdit):
                values[name] = widget.text().strip()
        return values

    def _refresh_key_label(self) -> None:
        stored = self._credentials.has(credential_key(SCOPE_ELN, self._account().account_id, self._user_id))
        self._key_label.setText(
            f"A key is stored ({self._credentials.backend_name})." if stored else "No key is stored yet."
        )

    def _refresh_blocks(self) -> None:
        lines: list[str] = []
        for kind in BLOCK_KINDS:
            for block_id, info in sorted(self._catalog.all(kind).items()):
                state = "ok" if info.usable else f"UNUSABLE: {info.error}"
                lines.append(f"{kind:9}  {block_id:18}  {info.source:7}  {state}")
        self._blocks_text.setPlainText("\n".join(lines))

    def edited_settings(self) -> ElnUserSettings:
        """Return the user's notebook settings as the form edits them."""
        account = replace(
            self._account(),
            connector=str(self._connector.currentData() or "elabftw"),
            settings=self.connector_settings(),
            label=self._account().label or "Lab notebook",
        )
        others = tuple(a for a in self._settings.accounts if a.account_id != account.account_id)
        # An item picked from the list stands for its id; typed text is an id.
        index = self._template.currentIndex()
        picked = index >= 0 and self._template.itemText(index) == self._template.currentText()
        data = self._template.itemData(index) if picked else None
        template = str(data) if data else self._template.currentText().strip()
        return replace(
            self._settings,
            enabled=self._enabled.isChecked(),
            accounts=(account, *others),
            default_account=account.account_id,
            default_profile=str(self._profile.currentData() or "default"),
            default_template=template,
        )

    # ── Actions ───────────────────────────────────────────────────────

    def save(self) -> bool:
        """Write the profile (and a newly typed key); returns ``True`` on success."""
        settings = self.edited_settings()
        try:
            self._profiles.update(self._user_id, lambda profile: replace(profile, eln=settings))
            key = self._key.text()
            if key:
                self._credentials.set(credential_key(SCOPE_ELN, settings.default_account, self._user_id), key)
                self._key.clear()
        except (OSError, ValueError, RuntimeError) as exc:
            self._status.setText(f"Not saved: {exc}")
            return False
        self._settings = settings
        self._refresh_key_label()
        if self._service is not None:
            self._service.reload()
        self._status.setText("Saved.")
        return True

    def test_connection(self) -> None:
        """Save, then check the server and the key (answer arrives later)."""
        if self._service is None or not self.save():
            return
        self._status.setText("Testing the connection…")
        self._service.verify(self._settings.default_account, self._on_verified, user_id=self._user_id)

    def _on_verified(self, result: Any) -> None:
        if isinstance(result, Exception):
            self._status.setText(f"Connection failed: {result}")
        else:
            identity = result if isinstance(result, dict) else {}
            name = identity.get("name", "")
            team = identity.get("team", "")
            self._status.setText(f"Connected as {name}{f' ({team})' if team else ''}.")

    def fetch_templates(self) -> None:
        """Save, then list the account's templates into the combo."""
        if self._service is None or not self.save():
            return
        self._status.setText("Fetching templates…")
        self._service.list_templates(self._settings.default_account, self._on_templates, user_id=self._user_id)

    def _on_templates(self, result: Any) -> None:
        if isinstance(result, Exception):
            self._status.setText(f"Could not fetch templates: {result}")
            return
        current = self._template.currentText()
        self._template.clear()
        self._template.addItem("", "")
        for item in result if isinstance(result, list) else []:
            self._template.addItem(f"{item.get('name', '')} [{item.get('template_id', '')}]", str(item.get("template_id", "")))
        index = self._template.findData(current)
        self._template.setCurrentIndex(max(index, 0))
        self._status.setText(f"{max(self._template.count() - 1, 0)} template(s).")

    def open_blocks_folder(self) -> None:
        """Create (if needed) and open the user's blocks folder."""
        root = getattr(self._catalog, "user_root", None)
        if root is None:
            return
        for kind in BLOCK_KINDS:
            (Path(root) / KIND_DIRS[kind]).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))
