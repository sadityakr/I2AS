"""session_dialogs — dialog for picking/creating the active Session (L6 tier)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from i2as.session.store import SessionStore

_SESSION_ID_ROLE = Qt.ItemDataRole.UserRole


class ResumeSessionDialog(QDialog):
    """Pick an existing Session to resume, or create a new one.

    Every session owned by ``user_id`` is listed via
    ``SessionStore.list_sessions()`` — sessions live under
    ``sessions/<user_id>/``, so "nobody logged in yet" must already have
    been resolved to the fixed Guest identity by the caller before this
    dialog is built (see ``i2as.session.models.GUEST_USER_ID``).
    ``selected_session_id()`` is only meaningful after ``exec()`` returns
    ``Accepted``. Switching sessions is deferred-until-restart (see
    ``GLOSSARY.md``'s **Session**) — this dialog only picks or creates the
    record; the caller persists it as
    active and tells the operator to restart.
    """

    def __init__(
        self,
        store: SessionStore,
        user_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resume Session")
        self._store = store
        self._user_id = user_id

        self._list = QListWidget()
        self._list.setObjectName("resume_session_list")
        self._populate()

        new_row = QHBoxLayout()
        self._new_name_input = QLineEdit()
        self._new_name_input.setObjectName("new_session_name_input")
        self._new_name_input.setPlaceholderText("New session name…")
        self._new_name_input.textChanged.connect(self._update_create_enabled)
        new_row.addWidget(self._new_name_input, 1)
        self._create_btn = QPushButton("Create")
        self._create_btn.setObjectName("create_session_btn")
        self._create_btn.setEnabled(False)
        self._create_btn.clicked.connect(self._on_create_clicked)
        new_row.addWidget(self._create_btn)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._list.itemSelectionChanged.connect(self._update_ok_enabled)
        self._list.itemDoubleClicked.connect(lambda _item: self.accept())

        layout = QVBoxLayout(self)
        layout.addWidget(self._list)
        layout.addLayout(new_row)
        layout.addWidget(buttons)

    def _populate(self) -> None:
        """List every session owned by ``user_id``."""
        for session_id in self._store.list_sessions(self._user_id):
            session = self._store.load(self._user_id, session_id)
            if session is None:
                continue
            label = f"{session.name} ({session.created_utc[:10]})"
            item = QListWidgetItem(label)
            item.setData(_SESSION_ID_ROLE, session_id)
            self._list.addItem(item)

    def _update_ok_enabled(self) -> None:
        self._ok_button.setEnabled(bool(self._list.selectedItems()))

    def _update_create_enabled(self) -> None:
        self._create_btn.setEnabled(bool(self._new_name_input.text().strip()))

    def _on_create_clicked(self) -> None:
        """Create a new session owned by ``user_id``, select it, and accept."""
        name = self._new_name_input.text().strip()
        if not name:
            return
        session = self._store.create_session(name=name, user_id=self._user_id)
        item = QListWidgetItem(f"{session.name} ({session.created_utc[:10]})")
        item.setData(_SESSION_ID_ROLE, session.session_id)
        self._list.addItem(item)
        self._list.setCurrentItem(item)
        self.accept()

    def selected_session_id(self) -> str | None:
        """Return the chosen session id. Only meaningful after ``accept()``.

        Returns:
            The selected session's store id, or ``None`` if nothing is
            selected.
        """
        items = self._list.selectedItems()
        if not items:
            return None
        return str(items[0].data(_SESSION_ID_ROLE) or "") or None
