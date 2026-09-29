"""Notebook dialogs — linking an experiment to its page, and reading fields back.

``LinkNotebookDialog`` is shown as soon as an experiment is started (when the
user publishes to a notebook), and from the Analysis tab later: link an
EXISTING page (searched on the notebook) or CREATE a new one from the
profile's template (queued, so it works offline), optionally with the sample
and resource items the experiment uses. "Not now" leaves the experiment
unlinked; it can be linked any time.

``ReadFieldsDialog`` reads the page's and the linked items' fields through
the profile's read map and shows each value beside what the experiment has
now; only the values the person ticks are applied to the sample metadata,
which every LATER run stamps into its data file.

Both talk to the notebook through the ``ElnService``, whose worker thread
does the network: the dialogs never wait on it.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from i2as.blocks.connector import KIND_ENTRY, KIND_ITEM, ElnEntryRef
from i2as.blocks.discovery import KIND_PROFILE
from i2as.gui.theme import BTN_CLASS_SECONDARY
from i2as.session.eln.publishing import PublishError
from i2as.session.models import LinkedItem

MODE_EXISTING = "existing"
MODE_CREATE = "create"
MODE_LATER = "later"


class LinkNotebookDialog(QDialog):
    """Link the open experiment to an existing notebook page, or create one.

    Named widgets: ``link_account_combo``, ``link_profile_combo``,
    ``link_mode_existing``, ``link_mode_create``, ``link_mode_later``,
    ``link_page_search_edit``, ``link_page_search_btn``, ``link_page_list``,
    ``link_title_edit``, ``link_item_search_edit``, ``link_item_search_btn``,
    ``link_item_list``, ``link_status_label``.

    Args:
        service: The ``ElnService``.
        experiment: The open ``ExperimentRecord``.
        parent: Optional Qt parent.
    """

    def __init__(self, service: Any, experiment: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Notebook page for this experiment")
        self.setObjectName("link_notebook_dialog")
        self.setMinimumSize(620, 560)
        self._service = service
        self._experiment = experiment
        user = experiment.user_id or "guest"
        self._user = user

        root = QVBoxLayout(self)
        intro = QLabel(
            f"Experiment <b>{experiment.title or experiment.experiment_id}</b> publishes to ONE "
            "notebook page. Link a page that already exists, or create a new one."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        self._account = QComboBox()
        self._account.setObjectName("link_account_combo")
        for account in service.accounts(user):
            self._account.addItem(account.label or account.account_id, account.account_id)
        form.addRow("Account:", self._account)
        self._profile = QComboBox()
        self._profile.setObjectName("link_profile_combo")
        for block_id in sorted(service.catalog().all(KIND_PROFILE)):
            self._profile.addItem(block_id, block_id)
        form.addRow("Profile:", self._profile)
        root.addLayout(form)

        self._modes = QButtonGroup(self)
        self._existing = QRadioButton("Link an existing page")
        self._existing.setObjectName("link_mode_existing")
        self._create = QRadioButton("Create a new page")
        self._create.setObjectName("link_mode_create")
        self._later = QRadioButton("Not now")
        self._later.setObjectName("link_mode_later")
        for button in (self._existing, self._create, self._later):
            self._modes.addButton(button)
        self._create.setChecked(True)

        root.addWidget(self._existing)
        search_row = QHBoxLayout()
        self._page_search = QLineEdit()
        self._page_search.setObjectName("link_page_search_edit")
        self._page_search.setPlaceholderText("Search pages by title")
        search_row.addWidget(self._page_search, stretch=1)
        self._page_search_btn = QPushButton("Search")
        self._page_search_btn.setObjectName("link_page_search_btn")
        self._page_search_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._page_search_btn.clicked.connect(lambda: self._search(KIND_ENTRY))
        search_row.addWidget(self._page_search_btn)
        root.addLayout(search_row)
        self._pages = QListWidget()
        self._pages.setObjectName("link_page_list")
        self._pages.setMaximumHeight(110)
        self._pages.itemSelectionChanged.connect(lambda: self._existing.setChecked(True))
        root.addWidget(self._pages)

        root.addWidget(self._create)
        title_row = QFormLayout()
        self._title = QLineEdit(experiment.title or experiment.experiment_id)
        self._title.setObjectName("link_title_edit")
        title_row.addRow("New page title:", self._title)
        root.addLayout(title_row)

        items_label = QLabel("Samples and resources used (linked to the page; their fields can be read back):")
        items_label.setWordWrap(True)
        root.addWidget(items_label)
        item_row = QHBoxLayout()
        self._item_search = QLineEdit()
        self._item_search.setObjectName("link_item_search_edit")
        self._item_search.setPlaceholderText("Search samples / resources")
        item_row.addWidget(self._item_search, stretch=1)
        self._item_search_btn = QPushButton("Search")
        self._item_search_btn.setObjectName("link_item_search_btn")
        self._item_search_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._item_search_btn.clicked.connect(lambda: self._search(KIND_ITEM))
        item_row.addWidget(self._item_search_btn)
        root.addLayout(item_row)
        self._items = QListWidget()
        self._items.setObjectName("link_item_list")
        self._items.setMaximumHeight(110)
        root.addWidget(self._items)

        root.addWidget(self._later)
        self._status = QLabel("")
        self._status.setObjectName("link_status_label")
        self._status.setWordWrap(True)
        self._status.setProperty("class", "secondary_label")
        root.addWidget(self._status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        if self._account.count() == 0:
            self._status.setText("No notebook account is configured — Settings → Electronic notebook.")
            self._existing.setEnabled(False)
            self._create.setEnabled(False)
            self._later.setChecked(True)

    # ── Searching (answers arrive later) ──────────────────────────────

    def _search(self, kind: str) -> None:
        text = (self._page_search if kind == KIND_ENTRY else self._item_search).text().strip()
        account = str(self._account.currentData() or "")
        if not account:
            return
        self._status.setText("Searching…")
        self._service.search(account, text, kind, lambda result: self._on_results(kind, result), user_id=self._user)

    def _on_results(self, kind: str, result: Any) -> None:
        if isinstance(result, Exception):
            self._status.setText(f"Search failed: {result}")
            return
        target = self._pages if kind == KIND_ENTRY else self._items
        target.clear()
        for hit in result if isinstance(result, list) else []:
            label = hit.get("title", "")
            if hit.get("category"):
                label += f"  [{hit['category']}]"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, hit)
            if kind == KIND_ITEM:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
            target.addItem(item)
        self._status.setText(f"{target.count()} found.")

    # ── Result ────────────────────────────────────────────────────────

    def mode(self) -> str:
        """``existing``, ``create`` or ``later``."""
        if self._later.isChecked():
            return MODE_LATER
        return MODE_EXISTING if self._existing.isChecked() else MODE_CREATE

    def selected_items(self) -> list[LinkedItem]:
        """The ticked samples and resources."""
        chosen: list[LinkedItem] = []
        for index in range(self._items.count()):
            item = self._items.item(index)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            hit = item.data(Qt.ItemDataRole.UserRole) or {}
            ref = hit.get("ref") or {}
            category = str(hit.get("category") or "")
            chosen.append(
                LinkedItem(
                    item_id=str(ref.get("record_id", "")),
                    kind="item",
                    role="sample" if category.lower() in ("", "sample", "samples") else category.lower(),
                    title=str(hit.get("title", "")),
                    url=str(hit.get("url", "")),
                )
            )
        return chosen

    def accept(self) -> None:
        """Link (or create) and close; stay open with the reason on failure."""
        mode = self.mode()
        if mode == MODE_LATER:
            super().accept()
            return
        entry: ElnEntryRef | None = None
        if mode == MODE_EXISTING:
            current = self._pages.currentItem()
            if current is None:
                self._status.setText("Pick a page from the search results first.")
                return
            hit = current.data(Qt.ItemDataRole.UserRole) or {}
            entry = ElnEntryRef(entry_id=str((hit.get("ref") or {}).get("record_id", "")), url=str(hit.get("url", "")))
        try:
            self._service.link_experiment(
                str(self._account.currentData() or ""),
                str(self._profile.currentData() or ""),
                entry=entry,
                new_title=self._title.text().strip(),
                items=self.selected_items(),
            )
        except PublishError as exc:
            self._status.setText(str(exc))
            return
        super().accept()


class ReadFieldsDialog(QDialog):
    """Show the notebook's field values beside the experiment's; apply the ticked ones.

    Named widgets: ``read_fields_table``, ``read_fields_status_label``,
    ``read_fields_apply_btn``.

    Args:
        service: The ``ElnService``.
        manager: The ``ExperimentManager`` (the one writer of the values).
        parent: Optional Qt parent.
    """

    def __init__(self, service: Any, manager: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Read fields from the notebook")
        self.setObjectName("read_fields_dialog")
        self.setMinimumSize(640, 360)
        self._manager = manager
        self._found: dict[str, Any] = {}
        root = QVBoxLayout(self)
        self._table = QTableWidget(0, 5)
        self._table.setObjectName("read_fields_table")
        self._table.setHorizontalHeaderLabels(["Apply", "Key", "Now", "From the notebook", "Source"])
        root.addWidget(self._table, stretch=1)
        self._status = QLabel("Reading…")
        self._status.setObjectName("read_fields_status_label")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self._apply = buttons.addButton("Apply ticked values", QDialogButtonBox.ButtonRole.AcceptRole)
        self._apply.setObjectName("read_fields_apply_btn")
        self._apply.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        service.read_fields(self.show_result)

    def show_result(self, result: Any) -> None:
        """Fill the table with what was read (or say why nothing was)."""
        if isinstance(result, Exception):
            self._status.setText(f"Could not read the notebook: {result}")
            return
        self._found = dict(result) if isinstance(result, dict) else {}
        self._table.setRowCount(len(self._found))
        for row, (key, entry) in enumerate(sorted(self._found.items())):
            check = QTableWidgetItem()
            usable = entry.get("value") is not None
            check.setFlags(
                (Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled) if usable else Qt.ItemFlag.NoItemFlags
            )
            changed = usable and entry.get("value") != entry.get("current")
            check.setCheckState(Qt.CheckState.Checked if changed else Qt.CheckState.Unchecked)
            check.setData(Qt.ItemDataRole.UserRole, key)
            self._table.setItem(row, 0, check)
            self._table.setItem(row, 1, QTableWidgetItem(key))
            self._table.setItem(row, 2, QTableWidgetItem("" if entry.get("current") is None else str(entry.get("current"))))
            shown = entry.get("error") or f"{entry.get('value')} {entry.get('unit') or ''}".strip()
            self._table.setItem(row, 3, QTableWidgetItem(shown))
            self._table.setItem(row, 4, QTableWidgetItem(str(entry.get("source", ""))))
        self._table.resizeColumnsToContents()
        self._apply.setEnabled(bool(self._found))
        self._status.setText("Tick the values to copy into the sample metadata. Runs already recorded keep theirs.")

    def chosen_values(self) -> dict[str, Any]:
        """The ticked ``{key: value}``."""
        values: dict[str, Any] = {}
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                key = str(item.data(Qt.ItemDataRole.UserRole))
                values[key] = self._found[key].get("value")
        return values

    def accept(self) -> None:
        """Apply the ticked values through the manager, keeping what was read as provenance."""
        values = self.chosen_values()
        if values:
            snapshot = {
                key: {k: entry.get(k) for k in ("value", "unit", "source", "fetched_utc")}
                for key, entry in self._found.items()
            }
            self._manager.apply_eln_fields(values, snapshot)
        super().accept()
