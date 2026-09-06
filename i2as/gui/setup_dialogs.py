"""Modal dialogs for the Setup tier: user login and read-only instrument info."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from i2as.gui.experiment_dialogs import UserPickerWidget
from i2as.session.store import UserRoster


class LoginDialog(QDialog):
    """Pick (or create) who is using the app.

    Identity only — no password. Whatever is selected becomes
    ``app_settings.current_user_id()``, which switches which form-autosave
    file ``ExperimentInfoPanel``'s sample fields and the run queue restore from.
    """

    def __init__(
        self,
        roster: UserRoster,
        current_user_id: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Log In")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Who's using I2AS?"))

        self._user_picker = UserPickerWidget(roster)
        self._user_picker.reload(select_user_id=current_user_id)
        layout.addWidget(self._user_picker)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setEnabled(self._user_picker.has_users())
        self._user_picker.selection_changed_signal().connect(self._update_ok_enabled)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_ok_enabled(self) -> None:
        self._ok_button.setEnabled(self._user_picker.has_users())

    def selected_user_id(self) -> str:
        """Return the picked roster id. Only meaningful after ``exec()`` accepts."""
        return self._user_picker.selected_user_id()


class InstrumentInfoDialog(QDialog):
    """Read-only view of each VI's optional ``devices.yaml`` metadata block.

    Editing happens in the Config Editor, not here — this dialog only
    displays what ``read_instrument_metadata()`` already parsed.
    """

    def __init__(
        self,
        instrument_metadata: dict[str, dict[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Instrument Info")
        self.resize(480, 420)

        outer = QVBoxLayout(self)

        if not instrument_metadata:
            empty_label = QLabel(
                "No instrument has a metadata: block in the active config's "
                "devices.yaml yet. Add one and reopen this dialog — see the "
                "Config Editor."
            )
            empty_label.setWordWrap(True)
            outer.addWidget(empty_label)
            outer.addStretch()
        else:
            scroll = QScrollArea()
            scroll.setObjectName("instrument_info_scroll")
            scroll.setWidgetResizable(True)
            scroll.setWidget(self._build_list(instrument_metadata))
            outer.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        outer.addWidget(buttons)

    def _build_list(self, instrument_metadata: dict[str, dict[str, str]]) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        for vi_name in sorted(instrument_metadata):
            fields = instrument_metadata[vi_name]
            layout.addWidget(QLabel(f"<b>{vi_name}</b>"))
            for field_name in sorted(fields):
                layout.addWidget(QLabel(f"  {field_name}: {fields[field_name]}"))
        layout.addStretch()
        return box
