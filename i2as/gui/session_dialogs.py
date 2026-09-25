"""session_dialogs — choosing the session folder, the one place on disk the operator picks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from i2as.session.store import SessionStore

_FOLDER_ROLE = Qt.ItemDataRole.UserRole


class SessionFolderDialog(QDialog):
    """Open an existing session folder, or create a new one.

    A session is a folder the operator chooses anywhere on disk; everything
    below it — experiments, run files, analysis output — has one fixed shape
    (``SessionStore``). The dialog shows the session in use, the recently
    used ones, and two ways to pick another:

    - **Open Folder…** picks a folder that already holds a ``session.json``;
    - **New Session** takes a name and a folder: an empty folder becomes the
      session itself, and any other folder gets a new ``YYYYMMDD_<name>``
      folder created inside it.

    ``selected_folder()`` is only meaningful after ``exec()`` returns
    ``Accepted``. Switching is deferred until the next launch (see
    ``GLOSSARY.md``'s **Session**): the caller persists the choice as active
    and tells the operator.
    """

    def __init__(
        self,
        store: SessionStore,
        user_id: str,
        current_folder: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the dialog.

        Args:
            store: The machine's session registry.
            user_id: Who owns a session created here.
            current_folder: The session in use, shown at the top.
            parent: Qt parent.
        """
        super().__init__(parent)
        self.setWindowTitle("Session Folder")
        self._store = store
        self._user_id = user_id
        self._selected: Path | None = None

        current = QLabel(self._describe(current_folder) if current_folder else "No session in use")
        current.setObjectName("current_session_label")
        current.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._list = QListWidget()
        self._list.setObjectName("recent_sessions_list")
        for folder in store.recent():
            item = QListWidgetItem(self._describe(folder))
            item.setData(_FOLDER_ROLE, str(folder))
            item.setToolTip(str(folder))
            self._list.addItem(item)
        self._list.itemSelectionChanged.connect(self._update_ok_enabled)
        self._list.itemDoubleClicked.connect(lambda _item: self._accept_selected())

        open_btn = QPushButton("Open Folder…")
        open_btn.setObjectName("open_session_folder_btn")
        open_btn.clicked.connect(self._on_open_folder)

        self._new_name_input = QLineEdit()
        self._new_name_input.setObjectName("new_session_name_input")
        self._new_name_input.setPlaceholderText("New session name…")
        self._new_name_input.textChanged.connect(self._update_create_enabled)
        self._create_btn = QPushButton("New Session…")
        self._create_btn.setObjectName("create_session_btn")
        self._create_btn.setEnabled(False)
        self._create_btn.setToolTip("Choose the folder the new session is created in")
        self._create_btn.clicked.connect(self._on_create_clicked)
        new_row = QHBoxLayout()
        new_row.addWidget(self._new_name_input, 1)
        new_row.addWidget(self._create_btn)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setEnabled(False)
        buttons.accepted.connect(self._accept_selected)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>In use</b>"))
        layout.addWidget(current)
        layout.addWidget(QLabel("<b>Recent sessions</b>"))
        layout.addWidget(self._list)
        layout.addWidget(open_btn)
        layout.addLayout(new_row)
        layout.addWidget(buttons)

    def _describe(self, folder: Path) -> str:
        """Return ``"<name> — <folder>"`` for one session folder."""
        session = self._store.load(folder)
        name = session.name if session is not None and session.name else Path(folder).name
        return f"{name} — {folder}"

    def _update_ok_enabled(self) -> None:
        self._ok_button.setEnabled(bool(self._list.selectedItems()))

    def _update_create_enabled(self) -> None:
        self._create_btn.setEnabled(bool(self._new_name_input.text().strip()))

    def _accept_selected(self) -> None:
        items = self._list.selectedItems()
        if not items:
            return
        self._selected = Path(str(items[0].data(_FOLDER_ROLE)))
        self.accept()

    def _pick_directory(self, caption: str) -> Path | None:
        """Ask for a folder; a seam the tests replace.

        Args:
            caption: The dialog's caption.

        Returns:
            The chosen folder, or ``None`` when the operator cancelled.
        """
        chosen = QFileDialog.getExistingDirectory(self, caption, str(self._store.default_parent()))
        return Path(chosen) if chosen else None

    def _on_open_folder(self) -> None:
        """Accept a folder that is already a session, or say why not."""
        folder = self._pick_directory("Open Session Folder")
        if folder is None:
            return
        if not self._store.is_session_folder(folder):
            QMessageBox.warning(
                self,
                "Open Session",
                f"{folder} is not a session folder (it has no session.json).\n"
                f"Use New Session to make a session there.",
            )
            return
        self._selected = folder
        self.accept()

    def _on_create_clicked(self) -> None:
        """Create a session in the chosen folder, or in a new folder inside it."""
        name = self._new_name_input.text().strip()
        if not name:
            return
        chosen = self._pick_directory("Choose Folder for the New Session")
        if chosen is None:
            return
        folder = self.new_session_folder(chosen, name)
        try:
            self._store.create_session(folder, name=name, user_id=self._user_id)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "New Session", str(exc))
            return
        self._selected = folder
        self.accept()

    def new_session_folder(self, chosen: Path, name: str) -> Path:
        """Return the folder a new session named *name* goes into, given the pick.

        Args:
            chosen: The folder the operator picked.
            name: The new session's name.

        Returns:
            *chosen* itself when it is empty or new, else a fresh
            ``YYYYMMDD_<name>`` folder inside it.
        """
        if not chosen.exists() or not any(chosen.iterdir()):
            return chosen
        return self._store.make_session_folder(
            chosen, name, datetime.now(timezone.utc).isoformat()
        )

    def selected_folder(self) -> Path | None:
        """Return the chosen session folder. Only meaningful after ``accept()``."""
        return self._selected
