"""The **eLab setup dialog** — where a person configures their own notebook.

One modal form over ``ElnSettings``, opened from the Monitor window's User
menu and from the procedure window's **eLab tab**. It edits the user-level
settings file (``session/eln/settings.py``: never a shipped config, never
git-tracked) and hands the edited record back through the ``on_save``
callable, which is what writes it and reloads the publisher — the dialog
itself writes no file and holds no publisher, so it stays a pure form.

**The key is never shown.** The API-key field is a password field with a
"leave blank to keep the stored key" placeholder and is never pre-filled with
the stored key; nothing here renders it into a label, a tooltip or a log
line, exactly as ``ElnSettings`` redacts it from ``repr()``. A blank field
means "keep what is stored", so a person can edit the URL without ever
handling their own key.

**Unshown fields survive.** ``settings_from_form()`` builds its answer with
``dataclasses.replace`` on the settings it was given, so the drafting
assistant's block, the price table, the retry timings and anything a later
release adds are carried through untouched by a dialog that never displayed
them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from i2as.gui.theme import BTN_CLASS_PRIMARY, BTN_CLASS_SECONDARY
from i2as.session.eln.adapter import ElnError
from i2as.session.eln.settings import ElnSettings

logger = logging.getLogger(__name__)

#: Placeholder of the API-key field. The field is never pre-filled with the
#: stored key, so this line is what tells the reader that leaving it alone
#: keeps the key they already have.
KEY_PLACEHOLDER = "leave blank to keep the stored key"

#: One megabyte, the unit the attachment cap is edited in (the setting itself
#: is bytes, SI-in-the-API with the display conversion here in the GUI).
_BYTES_PER_MB = 1024 * 1024

#: Defaults the Analysis group shows when the settings record predates the
#: analysis block (a mid-migration settings file, or a parallel build).
_ANALYSIS_DEFAULTS = {
    "enabled": False,
    "timeout_s": 120.0,
    "include_fact_tables": False,
    "attach_data_file": False,
}


def persist_eln_settings(settings: ElnSettings, publisher: Any | None = None) -> bool:
    """Write the edited settings and tell the publisher to re-read them.

    The one place the two halves of "Save" live together, so every opener of
    this dialog (the Monitor window's menu, the eLab tab's button) does the
    same thing. Both halves are optional at import time: an installation
    whose session layer predates ``save_eln_settings``/``reload_settings``
    logs a warning and keeps the edited record in memory rather than failing
    the click.

    Args:
        settings: The record ``ElnSettingsDialog.settings_from_form()``
            returned.
        publisher: The ``ElnPublisher`` to reload, or ``None`` when no
            publisher is wired.

    Returns:
        ``True`` when the settings file was written, ``False`` otherwise
        (logged, never raised — saving settings is a GUI action).
    """
    written = False
    try:
        from i2as.session.eln.settings import save_eln_settings
    except ImportError:
        logger.warning("This build cannot write ELN settings — nothing was saved")
    else:
        try:
            path = save_eln_settings(settings)
        except OSError:
            logger.exception("Writing the ELN settings file failed")
        else:
            written = True
            logger.info("ELN settings written to %s", path)

    reload_settings = getattr(publisher, "reload_settings", None)
    if callable(reload_settings):
        try:
            reload_settings(settings)
        except Exception:  # noqa: BLE001 - a reload must not raise into Qt
            logger.exception("Reloading the ELN publisher's settings failed")
    return written


class ElnSettingsDialog(QDialog):
    """The **eLab setup dialog**: one form over the user-level ELN settings.

    Named widgets (``findChild`` objectNames are API): the enabled toggle
    ``eln_enabled_checkbox``, the backend selector ``eln_backend_combo``, the
    URL ``eln_base_url_input``, the key ``eln_api_key_input``, the team
    ``eln_team_id_input``, the template ``eln_template_combo``, TLS
    verification ``eln_verify_tls_checkbox``, auto-publish
    ``eln_auto_publish_checkbox``, tags ``eln_tags_input``, the attachment cap
    ``eln_attachment_cap_input``, the Analysis group ``eln_analysis_group``
    with ``eln_analysis_enabled_checkbox``, ``eln_analysis_timeout_input``,
    ``eln_analysis_facts_checkbox`` and ``eln_analysis_attach_checkbox``, the
    four buttons ``eln_test_btn`` / ``eln_fetch_templates_btn`` /
    ``eln_save_btn`` / ``eln_cancel_btn``, and the one status line
    ``eln_status_label``.

    Args:
        settings: The settings to edit. Never mutated: the dialog answers
            with a new record built by ``dataclasses.replace`` on this one,
            so every field it does not show survives a save.
        on_save: Called with the edited settings when Save is pressed, before
            the dialog accepts. Whoever opens the dialog owns writing the
            file and reloading the publisher (``persist_eln_settings``).
        adapters: ``{backend_id: adapter_class}`` the backend combo offers
            and "Test connection" builds from. ``None`` discovers them
            (``session/eln/publisher.py``'s ``discover_backends()``), which
            is what the application passes implicitly.
        parent: Optional Qt parent widget.
    """

    def __init__(
        self,
        settings: ElnSettings,
        *,
        on_save: Callable[[ElnSettings], None],
        adapters: Mapping[str, type] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("eLab notebook")
        self.setObjectName("eln_settings_dialog")
        self.setMinimumWidth(520)
        self._settings = settings
        self._on_save = on_save
        self._adapters: dict[str, type] = dict(
            adapters if adapters is not None else _discover_adapters()
        )

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.addWidget(self._build_notebook_group())
        root.addWidget(self._build_analysis_group())

        self._status_label = QLabel("")
        self._status_label.setObjectName("eln_status_label")
        self._status_label.setProperty("class", "secondary_label")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        root.addLayout(self._build_buttons())
        self._load_from(settings)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_notebook_group(self) -> QGroupBox:
        """Build the notebook half of the form.

        Returns:
            The "Notebook" group box, fully populated.
        """
        box = QGroupBox("Notebook")
        form = QFormLayout(box)

        self._enabled_checkbox = QCheckBox("Publish runs to an electronic lab notebook")
        self._enabled_checkbox.setObjectName("eln_enabled_checkbox")
        self._enabled_checkbox.setToolTip(
            "Master switch. Off means nothing ever leaves this machine."
        )
        form.addRow("Enabled:", self._enabled_checkbox)

        self._backend_combo = QComboBox()
        self._backend_combo.setObjectName("eln_backend_combo")
        self._backend_combo.setToolTip("Which notebook backend to publish through")
        for backend in sorted(self._adapters):
            self._backend_combo.addItem(backend, backend)
        form.addRow("Backend:", self._backend_combo)

        self._base_url_input = QLineEdit()
        self._base_url_input.setObjectName("eln_base_url_input")
        self._base_url_input.setPlaceholderText("https://elab.example.org")
        form.addRow("Base URL:", self._base_url_input)

        self._api_key_input = QLineEdit()
        self._api_key_input.setObjectName("eln_api_key_input")
        self._api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_input.setPlaceholderText(KEY_PLACEHOLDER)
        self._api_key_input.setToolTip(
            "Type a key to replace the stored one. Left blank, the stored key "
            "is kept and is never displayed."
        )
        form.addRow("API key:", self._api_key_input)

        self._team_id_input = QLineEdit()
        self._team_id_input.setObjectName("eln_team_id_input")
        self._team_id_input.setPlaceholderText("optional")
        form.addRow("Team id:", self._team_id_input)

        self._template_combo = QComboBox()
        self._template_combo.setObjectName("eln_template_combo")
        self._template_combo.setEditable(True)
        self._template_combo.setToolTip(
            "The backend template new entries are created from. "
            "Use “Fetch templates” to list what the backend offers."
        )
        form.addRow("Template:", self._template_combo)

        self._verify_tls_checkbox = QCheckBox("Verify the server certificate")
        self._verify_tls_checkbox.setObjectName("eln_verify_tls_checkbox")
        self._verify_tls_checkbox.setToolTip(
            "Only turn this off deliberately, for a lab instance with a "
            "self-signed certificate."
        )
        form.addRow("TLS:", self._verify_tls_checkbox)

        self._auto_publish_checkbox = QCheckBox("Publish a run when it finishes")
        self._auto_publish_checkbox.setObjectName("eln_auto_publish_checkbox")
        form.addRow("Auto-publish:", self._auto_publish_checkbox)

        self._tags_input = QLineEdit()
        self._tags_input.setObjectName("eln_tags_input")
        self._tags_input.setPlaceholderText("comma separated")
        form.addRow("Tags:", self._tags_input)

        self._attachment_cap_input = QSpinBox()
        self._attachment_cap_input.setObjectName("eln_attachment_cap_input")
        self._attachment_cap_input.setRange(0, 100_000)
        self._attachment_cap_input.setSuffix(" MB")
        self._attachment_cap_input.setToolTip(
            "A data file larger than this is recorded as a link instead of "
            "uploaded."
        )
        form.addRow("Attachment cap:", self._attachment_cap_input)
        return box

    def _build_analysis_group(self) -> QGroupBox:
        """Build the Analysis half of the form.

        Shown even when the settings record carries no ``analysis`` block
        yet: the group then displays the documented defaults and is skipped
        by ``settings_from_form()``, so an older record is never rewritten
        with a block it does not know.

        Returns:
            The "Analysis" group box, fully populated.
        """
        box = QGroupBox("Analysis")
        box.setObjectName("eln_analysis_group")
        form = QFormLayout(box)

        self._analysis_enabled_checkbox = QCheckBox(
            "Analyse a finished run before its entry is written"
        )
        self._analysis_enabled_checkbox.setObjectName("eln_analysis_enabled_checkbox")
        self._analysis_enabled_checkbox.setToolTip(
            "A finished run is analysed by a recipe first; the entry it "
            "produces waits for your approval in the eLab tab."
        )
        form.addRow("Analysis on:", self._analysis_enabled_checkbox)

        self._analysis_timeout_input = QDoubleSpinBox()
        self._analysis_timeout_input.setObjectName("eln_analysis_timeout_input")
        self._analysis_timeout_input.setRange(1.0, 3600.0)
        self._analysis_timeout_input.setDecimals(0)
        self._analysis_timeout_input.setSuffix(" s")
        self._analysis_timeout_input.setToolTip(
            "How long one recipe may run before the worker is stopped."
        )
        form.addRow("Timeout:", self._analysis_timeout_input)

        self._analysis_facts_checkbox = QCheckBox(
            "Append the run's own fact tables below the analysis"
        )
        self._analysis_facts_checkbox.setObjectName("eln_analysis_facts_checkbox")
        form.addRow("Fact tables:", self._analysis_facts_checkbox)

        self._analysis_attach_checkbox = QCheckBox("Attach the raw data file")
        self._analysis_attach_checkbox.setObjectName("eln_analysis_attach_checkbox")
        form.addRow("Data file:", self._analysis_attach_checkbox)
        return box

    def _build_buttons(self) -> QHBoxLayout:
        """Build the dialog's four buttons.

        Returns:
            The button row's layout.
        """
        row = QHBoxLayout()

        self._test_btn = QPushButton("Test connection")
        self._test_btn.setObjectName("eln_test_btn")
        self._test_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._test_btn.setToolTip(
            "Ask the backend who these credentials belong to. Nothing is "
            "written to the notebook."
        )
        self._test_btn.clicked.connect(self.test_connection)
        row.addWidget(self._test_btn)

        self._fetch_btn = QPushButton("Fetch templates")
        self._fetch_btn.setObjectName("eln_fetch_templates_btn")
        self._fetch_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._fetch_btn.setToolTip("List the entry templates the backend offers")
        self._fetch_btn.clicked.connect(self.fetch_templates)
        row.addWidget(self._fetch_btn)

        row.addStretch()

        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("eln_save_btn")
        self._save_btn.setProperty("class", BTN_CLASS_PRIMARY)
        self._save_btn.setToolTip("Write these settings and reload the publisher")
        self._save_btn.clicked.connect(self._on_save_clicked)
        row.addWidget(self._save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("eln_cancel_btn")
        cancel_btn.setProperty("class", BTN_CLASS_SECONDARY)
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(cancel_btn)
        return row

    # ------------------------------------------------------------------
    # Form <-> settings
    # ------------------------------------------------------------------

    def _load_from(self, settings: ElnSettings) -> None:
        """Fill every field from ``settings`` — except the key, ever.

        Args:
            settings: The record to display.
        """
        self._enabled_checkbox.setChecked(bool(settings.enabled))
        index = self._backend_combo.findData(settings.backend)
        if index < 0 and settings.backend:
            self._backend_combo.addItem(settings.backend, settings.backend)
            index = self._backend_combo.findData(settings.backend)
        if index >= 0:
            self._backend_combo.setCurrentIndex(index)
        self._base_url_input.setText(settings.base_url)
        self._api_key_input.clear()
        self._team_id_input.setText(settings.team_id)
        self._template_combo.setCurrentText(settings.template_id)
        self._verify_tls_checkbox.setChecked(bool(settings.verify_tls))
        self._auto_publish_checkbox.setChecked(bool(settings.auto_publish))
        self._tags_input.setText(", ".join(settings.tags))
        self._attachment_cap_input.setValue(
            max(0, round(settings.max_attachment_bytes / _BYTES_PER_MB))
        )

        analysis = getattr(settings, "analysis", None)
        self._analysis_enabled_checkbox.setChecked(
            bool(getattr(analysis, "enabled", _ANALYSIS_DEFAULTS["enabled"]))
        )
        self._analysis_timeout_input.setValue(
            float(getattr(analysis, "timeout_s", _ANALYSIS_DEFAULTS["timeout_s"]))
        )
        self._analysis_facts_checkbox.setChecked(
            bool(
                getattr(
                    analysis,
                    "include_fact_tables",
                    _ANALYSIS_DEFAULTS["include_fact_tables"],
                )
            )
        )
        self._analysis_attach_checkbox.setChecked(
            bool(
                getattr(
                    analysis, "attach_data_file", _ANALYSIS_DEFAULTS["attach_data_file"]
                )
            )
        )
        if analysis is None:
            self._status_label.setText(
                "This build stores no analysis settings yet — the Analysis "
                "group shows the defaults and is not saved."
            )

    def settings_from_form(self) -> ElnSettings:
        """Return the edited settings, preserving every field not shown.

        Built with ``dataclasses.replace`` on the record the dialog was given,
        so the drafting assistant's block, the price table and the retry
        timings survive a save that never displayed them. A blank API-key
        field keeps the stored key.

        Returns:
            The new ``ElnSettings``.
        """
        typed_key = self._api_key_input.text().strip()
        tags = tuple(
            part.strip() for part in self._tags_input.text().split(",") if part.strip()
        )
        updated = replace(
            self._settings,
            enabled=self._enabled_checkbox.isChecked(),
            backend=str(self._backend_combo.currentData() or self._settings.backend),
            base_url=self._base_url_input.text().strip().rstrip("/"),
            api_key=typed_key or self._settings.api_key,
            team_id=self._team_id_input.text().strip(),
            template_id=self._template_id(),
            verify_tls=self._verify_tls_checkbox.isChecked(),
            auto_publish=self._auto_publish_checkbox.isChecked(),
            tags=tags,
            max_attachment_bytes=self._attachment_cap_input.value() * _BYTES_PER_MB,
        )

        analysis = getattr(self._settings, "analysis", None)
        if analysis is None:
            return updated
        return replace(
            updated,
            analysis=replace(
                analysis,
                enabled=self._analysis_enabled_checkbox.isChecked(),
                timeout_s=float(self._analysis_timeout_input.value()),
                include_fact_tables=self._analysis_facts_checkbox.isChecked(),
                attach_data_file=self._analysis_attach_checkbox.isChecked(),
            ),
        )

    def _template_id(self) -> str:
        """Return the template the form names, as the backend's own id.

        The combo is editable and, once "Fetch templates" has filled it,
        shows ``"<id> — <name>"`` per row. A row that is selected therefore
        answers with its stored id; anything typed by hand answers with
        itself, because a backend id is exactly what a person would type.

        Returns:
            The template id, or ``""`` for the backend's default.
        """
        index = self._template_combo.currentIndex()
        text = self._template_combo.currentText().strip()
        if index >= 0 and text == self._template_combo.itemText(index):
            stored = self._template_combo.itemData(index)
            if stored:
                return str(stored)
        return text

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _build_adapter(self) -> Any:
        """Build the selected backend's adapter from the form's own values.

        Returns:
            The adapter instance.

        Raises:
            ElnError: No adapter class is available for the selected backend,
                or the class refused the settings mapping.
        """
        settings = self.settings_from_form()
        adapter_class = self._adapters.get(settings.backend)
        if adapter_class is None:
            raise ElnError(f"No adapter is available for backend {settings.backend!r}")
        try:
            return adapter_class(settings.to_dict(include_secret=True))
        except ElnError:
            raise
        except Exception as exc:  # noqa: BLE001 - any refusal is one failure here
            raise ElnError(f"Could not build the {settings.backend} adapter: {exc}")

    def test_connection(self) -> str:
        """Ask the backend who the credentials belong to, and show the answer.

        The call is bounded by the settings' own ``timeout_s`` (the adapter
        applies it), which is why a busy cursor is honest here: the dialog
        waits, but only for that long.

        Returns:
            The identity string the backend reported, or ``""`` on failure
            (the failure is shown in the status line, never raised).
        """
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            identity = str(self._build_adapter().verify())
        except ElnError as exc:
            self._status_label.setText(f"Connection failed: {exc}")
            logger.warning("eLab connection test failed: %s", exc)
            return ""
        finally:
            QApplication.restoreOverrideCursor()
        self._status_label.setText(f"Connected as {identity}")
        logger.info("eLab connection test succeeded: %s", identity)
        return identity

    def fetch_templates(self) -> int:
        """Fill the template combo from the backend's template list.

        Returns:
            How many templates were listed, or ``0`` on failure (shown in the
            status line, never raised).
        """
        current = self._template_combo.currentText().strip()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            templates = list(self._build_adapter().list_templates())
        except ElnError as exc:
            self._status_label.setText(f"Could not list templates: {exc}")
            logger.warning("eLab template listing failed: %s", exc)
            return 0
        finally:
            QApplication.restoreOverrideCursor()

        self._template_combo.clear()
        for template in templates:
            template_id = str(getattr(template, "template_id", ""))
            name = str(getattr(template, "name", "")) or template_id
            self._template_combo.addItem(f"{template_id} — {name}", template_id)
        index = self._template_combo.findData(current)
        if index >= 0:
            self._template_combo.setCurrentIndex(index)
        else:
            self._template_combo.setCurrentText(current)
        self._status_label.setText(f"{len(templates)} template(s) available")
        return len(templates)

    def _on_save_clicked(self) -> None:
        """Hand the edited settings to ``on_save`` and close."""
        settings = self.settings_from_form()
        try:
            self._on_save(settings)
        except Exception:  # noqa: BLE001 - a save must not raise into Qt
            logger.exception("Saving the ELN settings failed")
            self._status_label.setText("Saving failed — see the log")
            return
        self.accept()


def _discover_adapters() -> dict[str, type]:
    """Return every ELN backend this installation offers, keyed by its id.

    Imported lazily so the dialog module stays importable (and testable with
    injected adapters) in a build whose publisher module is unavailable.

    Returns:
        ``{backend_id: adapter_class}``; empty when discovery is impossible.
    """
    try:
        from i2as.session.eln.publisher import discover_backends

        return dict(discover_backends())
    except Exception:  # noqa: BLE001 - a dialog never fails to open over this
        logger.exception("Could not discover ELN backends")
        return {}
