"""The **Settings dialog** — one window, one page per section of the settings file.

Opened from the Monitor window's Settings menu. A list on the left names the
pages; the stack on the right shows one. Each page edits one section of the
general settings file (``i2as.session.app_config``) — or, for the
Electronic notebook page, the logged-in user's own profile — and has its own
Save, because the pages have different side effects: Connections starts and
stops the gateway live, Analysis checks the container engine before it lets
the stage go on, Electronic notebook stores the API key in the keyring.

**Adding a section.** Add its dataclass to ``AppConfig``, write a page
widget that takes the ``AppConfigStore``, and add one entry to
``SettingsDialog._build_pages()``. Nothing else changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from i2as.gui.analysis_settings_page import AnalysisSettingsPage
from i2as.gui.connections_page import ConnectionsPage
from i2as.gui.notebook_settings_page import NotebookSettingsPage
from i2as.gui.theme import BTN_CLASS_SECONDARY
from i2as.session.analysis_sandbox import check_engine
from i2as.session.app_config import AnalysisSettings, SandboxSettings

#: Page keys, in sidebar order — what ``SettingsDialog(page=...)`` accepts.
PAGE_CONNECTIONS = "connections"
PAGE_ANALYSIS = "analysis"
PAGE_NOTEBOOK = "notebook"

#: What the Connections page shows when no gateway is wired in.
NO_GATEWAY_TEXT = "This session has no Agent gateway wired in."


class SettingsDialog(QDialog):
    """The Settings dialog: a sidebar of pages over the general settings file.

    Named widgets: ``settings_page_list`` (the sidebar), ``settings_pages``
    (the stack), ``settings_close_btn``, and each page's own names —
    ``connections_page`` (or ``settings_connections_unavailable``) and
    ``analysis_settings_page``.

    Args:
        store: The ``AppConfigStore`` every page reads and saves.
        gateway_controller: The ``GatewayController`` the Connections page
            drives, or ``None`` when there is none.
        on_analysis_saved: Called after the Analysis page saves.
        engine_checker: Passed to the Analysis page; a test's stand-in.
        notebook: ``(profiles, credentials, user_id, catalog, service)`` for
            the Electronic notebook page, or ``None`` to leave it out.
        page: The page to open on (``PAGE_CONNECTIONS``, ``PAGE_ANALYSIS`` or
            ``PAGE_NOTEBOOK``).
        parent: Optional Qt parent widget.
    """

    def __init__(
        self,
        store: Any,
        *,
        gateway_controller: Any | None = None,
        on_analysis_saved: Callable[[AnalysisSettings], None] | None = None,
        engine_checker: Callable[[SandboxSettings], Any] = check_engine,
        notebook: tuple[Any, Any, str, Any, Any] | None = None,
        page: str = PAGE_CONNECTIONS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setObjectName("settings_dialog")
        self.setMinimumSize(780, 560)
        self._store = store
        self._gateway_controller = gateway_controller
        self._on_analysis_saved = on_analysis_saved
        self._engine_checker = engine_checker
        self._notebook = notebook

        root = QVBoxLayout(self)
        body = QHBoxLayout()
        root.addLayout(body, stretch=1)

        self._page_list = QListWidget()
        self._page_list.setObjectName("settings_page_list")
        self._page_list.setFixedWidth(160)
        body.addWidget(self._page_list)

        self._stack = QStackedWidget()
        self._stack.setObjectName("settings_pages")
        body.addWidget(self._stack, stretch=1)

        self._keys: list[str] = []
        self._pages: dict[str, QWidget] = {}
        self._build_pages()
        self._page_list.currentRowChanged.connect(self._stack.setCurrentIndex)

        row = QHBoxLayout()
        row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.setObjectName("settings_close_btn")
        close_btn.setProperty("class", BTN_CLASS_SECONDARY)
        close_btn.clicked.connect(self.accept)
        row.addWidget(close_btn)
        root.addLayout(row)

        self.show_page(page)

    def _build_pages(self) -> None:
        """Add every page, in sidebar order."""
        if self._gateway_controller is not None:
            connections: QWidget = ConnectionsPage(self._gateway_controller)
        else:
            connections = QLabel(NO_GATEWAY_TEXT)
            connections.setObjectName("settings_connections_unavailable")
            connections.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._add_page(PAGE_CONNECTIONS, "Connections", connections)
        self._add_page(
            PAGE_ANALYSIS,
            "Analysis",
            AnalysisSettingsPage(
                self._store,
                on_saved=self._on_analysis_saved,
                engine_checker=self._engine_checker,
            ),
        )
        if self._notebook is not None:
            profiles, credentials, user_id, catalog, service = self._notebook
            self._add_page(
                PAGE_NOTEBOOK,
                "Electronic notebook",
                NotebookSettingsPage(profiles, credentials, user_id, catalog, service),
            )

    def _add_page(self, key: str, title: str, widget: QWidget) -> None:
        self._keys.append(key)
        self._pages[key] = widget
        self._page_list.addItem(title)
        self._stack.addWidget(widget)

    def show_page(self, key: str) -> None:
        """Select one page by key; an unknown key selects the first.

        Args:
            key: ``PAGE_CONNECTIONS`` or ``PAGE_ANALYSIS``.
        """
        row = self._keys.index(key) if key in self._keys else 0
        self._page_list.setCurrentRow(row)
        self._stack.setCurrentIndex(row)

    def page(self, key: str) -> QWidget | None:
        """Return one page's widget, or ``None``."""
        return self._pages.get(key)

    def done(self, result: int) -> None:
        """Stop a running image build before the dialog goes away."""
        analysis = self._pages.get(PAGE_ANALYSIS)
        if isinstance(analysis, AnalysisSettingsPage):
            analysis.stop_build()
        super().done(result)
