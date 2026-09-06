"""MonitorWindow — main I2AS monitor window (composition shell)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from i2as.core.events import ErrorEvent
from i2as.core.orchestrator import OrchestratorState
from i2as.core.orchestrator_proxy import OrchestratorProxy
from i2as.core.config import read_instrument_metadata
from i2as.gui import app_settings  # import the module (not the function) so tests can monkeypatch the factory
from i2as.gui import form_autosave  # module import keeps save/load monkeypatchable
from i2as.gui import window_geometry
from i2as.gui.eln_settings_dialog import ElnSettingsDialog, persist_eln_settings
from i2as.gui.experiment_info_panel import ExperimentInfoPanel
from i2as.gui.agent_panel import AgentPanel
from i2as.gui.instrument_panel import InstrumentPanel
from i2as.gui.log_panel import LogPanel
from i2as.gui.notification_banner import NotificationBanner
from i2as.gui.offline_panel import OfflineInstrumentPanel
from i2as.gui.open_experiment_dialog import OpenExperimentDialog
from i2as.gui.ramp_tracker_panel import RampTrackerPanel
from i2as.gui.session_dialogs import ResumeSessionDialog
from i2as.core.status_mirror import StatusMirror
from i2as.gui.setup_dialogs import InstrumentInfoDialog, LoginDialog
from i2as.gui.theme import (
    BANNER_SEVERITY_ERROR,
    BANNER_SEVERITY_WARNING,
    BTN_CLASS_PRIMARY,
    BTN_CLASS_SECONDARY,
    TEXT_ON_ACCENT,
    TEXT_PRIMARY,
)
from i2as.gui.takeover_strip import TakeoverStrip
from i2as.gui.trends_quadrant import TrendsQuadrant
from i2as.gui.widget_lifecycle import hold_window, release_window, retire_widget
from i2as.session.manager import ExperimentManager
from i2as.session.models import GUEST_USER_ID
from i2as.session.store import SessionStore

if TYPE_CHECKING:
    from i2as.core.station import Station

logger = logging.getLogger(__name__)

# QSettings keys for persisted window/layout state.
_GEOMETRY_KEY = "MonitorWindow/geometry"
# Namespaced distinctly from any earlier layout scheme's settings (the
# pre-dock splitter-grid MonitorWindow used a plain "MonitorWindow/main_splitter"
# key; QSplitter.restoreState() restores orientation/child-count from the
# blob too, so applying its leftover registry value here would silently
# reshape this splitter to match a layout that no longer exists).
_MAIN_SPLITTER_KEY = "MonitorWindow/quadrant_main_splitter"
_LEFT_SPLITTER_KEY = "MonitorWindow/quadrant_left_splitter"
_RIGHT_SPLITTER_KEY = "MonitorWindow/quadrant_right_splitter"
_AGENTS_SPLITTER_KEY = "MonitorWindow/quadrant_agents_splitter"

# The ramp tracker's minimum width: a pane that displays data gets a real
# minimum so a splitter drag cannot crush a ramp row's numbers out of
# readability, and it must stay small enough that the right quadrant column
# never outgrows half the window — otherwise the main 50/50 quadrant split is
# forced open on a small screen.
_RAMPS_MIN_WIDTH = 140

# The bottom-right quadrant's vertical split: the Ramps sub-panel on top, the
# Agents sub-panel underneath. Vertical rather than two columns because an
# agent row is one wide line of text — time, actor, command, verdict,
# reason — and a column would wrap every one of them. The minimum is what
# keeps two rows and the filter visible at the smallest drag. Stretch factors
# as well as initial sizes: setSizes() alone is a one-shot hint that Qt
# re-divides evenly on the next resize.
_AGENTS_MIN_HEIGHT = 90
_BOTTOM_RIGHT_VERTICAL_SIZES = [220, 220]
_BOTTOM_RIGHT_VERTICAL_STRETCH = (1, 1)

# Orchestrator state names that colour the status bar (dynamic 'level' property).
_ACTIVE_STATES = frozenset({
    OrchestratorState.INITIATING.value,
    OrchestratorState.RAMPING.value,
    OrchestratorState.INITIATION_GATE.value,
    OrchestratorState.READING_GATE.value,
    OrchestratorState.MEASURING.value,
    OrchestratorState.SWEEPING.value,
    OrchestratorState.PAUSED.value,
})
_ERROR_STATES = frozenset({
    OrchestratorState.ERROR.value,
    OrchestratorState.EMERGENCY.value,
})


class MonitorWindow(QMainWindow):
    """Main window: live instrument monitor, sample info, global controls, and log.

    A slim page tab bar in the header switches between two pages held in a
    central QStackedWidget. Page 1 (Monitor) is the fixed 2x2 quadrant grid
    built from nested QSplitters: top-left is a scrollable 2-column grid of
    :class:`InstrumentPanel` cards for EVERY VI — system cards first, then
    measurement cards tagged by role — top-right is the
    :class:`TrendsQuadrant`, bottom-left is the :class:`ExperimentInfoPanel`,
    and bottom-right splits vertically into the :class:`RampTrackerPanel`
    over the :class:`AgentPanel`. The header carries the
    :class:`TakeoverStrip`. Every splitter boundary is draggable; nothing in
    the grid can be closed, detached, or floated. Page 2 (Logs) hosts the
    :class:`LogPanel` and nothing else.

    Args:
        station: The active Station instance.
        orchestrator: The active Orchestrator instance.
        parent: Optional Qt parent widget.
        active_config_path: Path of the currently-active config, or None —
            the source the User menu's "Instrument Info…" reads each VI's
            ``metadata:`` block from.
        startup_warning: Startup config-fallback warning to surface, or None.
        session_manager: Optional ExperimentManager (L6), forwarded to
            ExperimentInfoPanel and used for attribution prefills.
        eln_publisher: Optional ``ElnPublisher``, passed through to the
            procedure window's **eLab tab** and edited by the User menu's
            "eLab notebook…" action. ``None`` leaves both inert.
        analysis_runner: Optional ``AnalysisRunner``, passed through to the
            procedure window's **eLab tab**. ``None`` disables its "Run
            analysis" button.
        session_store: Optional SessionStore (the L6 Session tier above
            ``session_manager``), used by the User menu's "Resume Session…"
            action to list/create sessions and persist the active one.
            Switching is deferred-until-restart and takes effect on the
            next launch (see ``GLOSSARY.md``'s **Session**) —
            ``session_manager``'s own ``ExperimentStore`` is never rebound
            live.
        panels_config: The active config's ``panels:`` block
            (``i2as.core.config.read_panels_config()``): per-VI allowlists of the
            controls shown on the compact instrument cards. None/empty means
            every VI keeps its declared ``panel=`` defaults.
    """

    def __init__(
        self,
        station: Station,
        orchestrator: OrchestratorProxy,
        parent: QWidget | None = None,
        active_config_path: str | None = None,
        startup_warning: str | None = None,
        session_manager: ExperimentManager | None = None,
        eln_publisher: Any | None = None,
        analysis_runner: Any | None = None,
        session_store: SessionStore | None = None,
        panels_config: dict[str, list[str]] | None = None,
        mirror: StatusMirror | None = None,
    ) -> None:
        super().__init__(parent)
        self._station = station
        self._orchestrator = orchestrator
        # The status-mirror standard: every read below is a mirror read. One
        # mirror is shared by this window and every panel it builds, so they
        # all answer from the same event. Whoever builds the engine hands one
        # in already primed; the fallback is the inline construction path
        # (tests, and any caller with the engine on its own thread).
        self._mirror = mirror if mirror is not None else StatusMirror.of(orchestrator)
        self._panels_config = dict(panels_config or {})
        self._procedure_window = None  # lazily created
        #: Always built: every setup can ramp something, and the tracker
        #: shows its own empty state otherwise.
        self._ramp_tracker: RampTrackerPanel | None = None

        # Session layer (L6, optional — absent in unit tests). experiment_context()
        # stamps built procedures; the experiment start/close/attendance/findings
        # controls live on the ExperimentInfoPanel, which owns session_manager directly.
        self._session_manager = session_manager
        # The ELN/analysis pair: held only to hand on to the procedure
        # window's eLab tab and to open the eLab setup dialog. This window
        # neither publishes nor analyses anything itself.
        self._eln_publisher = eln_publisher
        self._analysis_runner = analysis_runner
        # The L6 Session tier above session_manager, used only by the User
        # menu's Resume Session… action (see _open_resume_session_dialog) —
        # session_manager's own ExperimentStore stays fixed for this run
        # regardless of what the operator picks here (deferred-until-restart).
        self._session_store = session_store
        # Tracks the experiment_id last seen by _on_session_experiment_changed,
        # so that handler (and the initial-resume check just below) only acts
        # on an actual open/switch transition — never on a same-experiment
        # re-emit (attendance/findings edits).
        self._last_session_experiment_id: str | None = None

        # The active config directory, held only so "Instrument Info…" can
        # read its ``metadata:`` blocks, and the startup fallback warning the
        # banner shows once the UI exists.
        self._active_config_path = active_config_path
        self._startup_warning = startup_warning

        # Who's logged in (Setup tier, User menu). Identity only — governs which
        # form-autosave file the session *content* below is loaded from/saved to,
        # so switching users switches what's remembered instead of one person's
        # fields overwriting another's. None means nobody has logged in yet (or
        # this is a unit test), and everything falls back to the original
        # shared last_session.json.
        self._current_user_id = app_settings.current_user_id()

        # Persistent session *content* (sample metadata, procedure params, run
        # queue) — a second persistence tier separate from the QSettings window
        # state. Loaded here and applied to the fields once they exist; re-saved
        # on close and by the User menu. When the ExperimentManager already has an
        # experiment resumed from a previous run (crash/close recovery), its own
        # gui_state.json — not the per-user AppData file — is the right source:
        # this is what makes a resumed session's own sample fields/queue reappear
        # rather than whoever's per-user autosave happens to be current. A
        # resumed experiment with no gui_state.json yet (never saved before the
        # app last stopped) starts from a blank FormAutosaveState, not the per-user
        # file, so ExperimentInfoPanel's own data-dir forcing (see
        # experiment_info_panel.py) is the one source of truth for its Data Dir.
        resumed_experiment = (
            self._session_manager.current_experiment()
            if self._session_manager is not None
            else None
        )
        if resumed_experiment is not None:
            self._last_session_experiment_id = resumed_experiment.experiment_id
            resumed_gui_state_path = self._session_manager.current_gui_state_path()
            self._session = (
                form_autosave.load(resumed_gui_state_path)
                if resumed_gui_state_path is not None and resumed_gui_state_path.exists()
                else form_autosave.FormAutosaveState()
            )
        else:
            self._session = form_autosave.load(
                app_settings.autosave_file_path(self._current_user_id)
            )

        self.setWindowTitle("I2AS — Monitor")
        window_geometry.restore_or_center(self, _GEOMETRY_KEY, fraction=0.9)

        self._build_ui()
        self._session_info.apply_session(self._session)
        self._build_menu()
        self._connect_signals()
        self._restore_monitor_state()
        # Sync state-dependent widgets (the ACKNOWLEDGE button and its
        # countdown) against whatever state the Orchestrator is already in:
        # state_changed only reports FUTURE transitions, and an EMERGENCY
        # (or a pre-existing hold — _refresh_ack_controls() checks
        # held_vi_names() live, independent of the state passed in) may
        # already be active by the time this window is constructed.
        self._on_state_changed(self._mirror.state)

        # Attach the log handler after the UI exists (LogPanel guards against
        # a duplicate if the window is ever reconstructed in-process).
        self._log_panel.attach()

        # Surface a startup config fallback (a bad active config was skipped)
        # and/or instruments that failed to connect (degraded build). One
        # combined banner: show_message replaces, so two calls would hide the
        # first message.
        startup_notes: list[str] = []
        if self._startup_warning:
            startup_notes.append(
                f"Config fallback in effect — {self._startup_warning}"
            )
        offline_names = self._station.offline_vi_names()
        if offline_names:
            startup_notes.append(
                f"{len(offline_names)} instrument(s) offline: "
                f"{', '.join(offline_names)}. Everything else is operational — "
                "open the instrument's details (sliders icon) to retry."
            )
        if startup_notes:
            self._banner.show_message(
                " | ".join(startup_notes), BANNER_SEVERITY_WARNING
            )

        # The window-liveness standard (gui/widget_lifecycle.py): this window
        # owns the reference that keeps it alive, so no garbage-collection
        # pass can destroy it — and the pyqtgraph scenes in its Trends
        # quadrant — while it is on screen. Released in closeEvent().
        hold_window(self)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        """Build the two operator menus: User and Procedures.

        There is no View menu: every quadrant is always visible and nothing
        can be hidden, so there is nothing to toggle. Trend plots are added
        via the button inside the Trends quadrant itself.
        """
        menu_bar = self.menuBar()

        # User menu is added first so it sits leftmost (menu order follows
        # addMenu() call order). Setup-tier concerns: who's logged in, and the
        # per-user form-autosave content that follows (sample info, params,
        # queue) — a "Session" label here would collide with i2as.session
        # (L6, the experiment layer), so this menu is named for what it is.
        user_menu = menu_bar.addMenu("User")
        login_action = QAction("Log in as…", self)
        login_action.setToolTip(
            "Pick who's using I2AS — switches which saved sample info, "
            "parameters, and queue are loaded"
        )
        login_action.triggered.connect(self._open_login_dialog)
        user_menu.addAction(login_action)

        # L6 session (experiment) surfaces — distinct from the per-user
        # form-autosave content below: these operate on ExperimentRecord
        # folders under the active Session's own folder, not last_session.json.
        load_session_action = QAction("Load Session…", self)
        load_session_action.setToolTip(
            "Switch to another open session (experiment)"
        )
        load_session_action.triggered.connect(self._open_load_session_dialog)
        user_menu.addAction(load_session_action)

        resume_session_action = QAction("Resume Session…", self)
        resume_session_action.setToolTip(
            "Pick or create the Session (folder holding multiple experiments) "
            "to use — applies fully on next launch"
        )
        resume_session_action.triggered.connect(self._open_resume_session_dialog)
        user_menu.addAction(resume_session_action)

        # The notebook account is a property of the PERSON, like the login
        # above and unlike a config: an API key must never travel with a
        # config directory (session/eln/settings.py), so the eLab setup
        # dialog belongs in this Setup-tier menu.
        eln_action = QAction("eLab notebook…", self)
        eln_action.setToolTip(
            "Notebook address, credentials, and whether a finished run is "
            "analysed before its entry is written"
        )
        eln_action.triggered.connect(self._open_eln_settings)
        user_menu.addAction(eln_action)

        # Setup-tier, read-only: what the active config says each instrument
        # is. It lives beside the login because it describes the rack the
        # person in front of the app is using, and it writes nothing.
        instrument_info_action = QAction("Instrument Info…", self)
        instrument_info_action.setToolTip(
            "View each instrument's identity metadata from devices.yaml"
        )
        instrument_info_action.triggered.connect(self._open_instrument_info)
        user_menu.addAction(instrument_info_action)

        user_menu.addSeparator()
        new_session_action = QAction("New Session", self)
        new_session_action.setToolTip(
            "Clear sample info, parameters, and queue and start a fresh session"
        )
        new_session_action.triggered.connect(self._on_new_session)
        user_menu.addAction(new_session_action)
        save_session_action = QAction("Save Session Now", self)
        save_session_action.setToolTip("Write the current session to disk immediately")
        save_session_action.triggered.connect(self._save_session)
        user_menu.addAction(save_session_action)

        proc_menu = menu_bar.addMenu("Procedures")
        open_action = QAction("Open Procedures…", self)
        open_action.setShortcut("Ctrl+P")
        open_action.triggered.connect(self._open_procedures)
        proc_menu.addAction(open_action)

    def _open_procedures(self) -> None:
        """Lazily create and show the ProcedureWindow."""
        if self._procedure_window is None:
            from i2as.gui.procedure_window import ProcedureWindow
            self._procedure_window = ProcedureWindow(
                self._station,
                self._orchestrator,
                get_sample_info=self.get_sample_info,
                get_data_dir=self.get_data_dir_for_run,
                initial_session=self._session,
                get_experiment_info=self.get_experiment_info,
                queue_host=(
                    self._session_manager.run_queue_host
                    if self._session_manager is not None
                    else None
                ),
                mirror=self._mirror,
                session_manager=self._session_manager,
                eln_publisher=self._eln_publisher,
                analysis_runner=self._analysis_runner,
            )
        self._procedure_window.show()
        self._procedure_window.raise_()
        self._procedure_window.activateWindow()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        content_widget = QWidget()
        root = QVBoxLayout(content_widget)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        self._system_vi_names = [
            n for n in self._station.get_vi_names()
            if self._station.get_vi_type(n) in {"system"}
        ]
        measurement_vis = [
            n for n in self._station.get_vi_names()
            if self._station.get_vi_type(n) == "measurement"
        ]

        # ── Header ────────────────────────────────────────────────────
        root.addLayout(self._build_header())

        # ── Notification banner (hidden until a warning/error arrives) ─
        self._banner = NotificationBanner()
        root.addWidget(self._banner)

        # ── Acknowledge (single home; moved off ProcedureWindow) ────────
        # Unified control for both EMERGENCY and a plain hold-severity
        # System condition — see Orchestrator.acknowledge() and GLOSSARY.md's
        # **Hold acknowledge**. Right-aligned in the top bar, next to the
        # countdown that reports how long the override it grants stays
        # unlocked.
        self._in_emergency = False
        ack_row = QHBoxLayout()
        ack_row.addStretch()
        self._ack_countdown_label = QLabel("")
        self._ack_countdown_label.setObjectName("ack_countdown_label")
        self._ack_countdown_label.setVisible(False)
        ack_row.addWidget(self._ack_countdown_label)
        self._ack_btn = QPushButton("Acknowledge emergency")
        self._ack_btn.setObjectName("ack_emergency_btn")
        self._ack_btn.setVisible(False)
        self._ack_btn.clicked.connect(self._on_ack_clicked)
        ack_row.addWidget(self._ack_btn)
        root.addLayout(ack_row)

        # Tracks the last per-VI fault warning message shown on the banner,
        # so states_updated can dismiss it once every fault clears without
        # stomping on an unrelated banner message.
        self._last_fault_message: str | None = None

        # Tracks the last hold-severity condition message shown on the
        # banner (see _refresh_ack_controls()), so it can be dismissed once
        # every hold condition clears without stomping on an unrelated
        # banner message (e.g. a fault warning that appeared since).
        self._last_hold_message: str | None = None

        # ── Fixed 2x2 quadrant grid (Page 1 — Monitor) ───────────────
        top_left = self._build_instruments_quadrant(measurement_vis)
        self._trends = TrendsQuadrant(self._station, parent=self)
        self._session_info = ExperimentInfoPanel(session_manager=self._session_manager)
        bottom_right = self._build_bottom_right_quadrant()

        self._left_splitter = QSplitter(Qt.Orientation.Vertical)
        self._left_splitter.setObjectName("left_splitter")
        self._left_splitter.setChildrenCollapsible(False)
        self._left_splitter.addWidget(top_left)
        self._left_splitter.addWidget(self._session_info)
        self._left_splitter.setSizes([750, 250])

        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setObjectName("right_splitter")
        self._right_splitter.setChildrenCollapsible(False)
        self._right_splitter.addWidget(self._trends)
        self._right_splitter.addWidget(bottom_right)
        self._right_splitter.setSizes([700, 300])

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setObjectName("main_splitter")
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(self._left_splitter)
        self._main_splitter.addWidget(self._right_splitter)
        self._main_splitter.setSizes([600, 600])

        # ── Page 2 — Logs ─────────────────────────────────────────────
        # The application LogPanel is created here and composed into the
        # Logs page (moved off the bottom-right quadrant); MonitorWindow
        # still owns its attach()/detach() lifecycle (see __init__/closeEvent).
        self._log_panel = LogPanel()
        self._logs_page = QWidget()
        self._logs_page.setObjectName("logs_page")
        logs_layout = QVBoxLayout(self._logs_page)
        logs_layout.setContentsMargins(4, 4, 4, 4)
        logs_layout.setSpacing(4)
        logs_layout.addWidget(QLabel("<b>Logs</b>"))
        logs_layout.addWidget(self._log_panel)

        # ── Page switcher: a QStackedWidget driven by the header tab bar ──
        self._page_stack = QStackedWidget()
        self._page_stack.setObjectName("page_stack")
        self._page_stack.addWidget(self._main_splitter)  # page 0: Monitor
        self._page_stack.addWidget(self._logs_page)  # page 1: Logs
        root.addWidget(self._page_stack)
        self._page_tab_bar.currentChanged.connect(self._on_page_changed)

        # ── Content widget is the central widget directly (no outer scroll) ──
        self.setCentralWidget(content_widget)

        # ── Status bar ────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        # The last state announced, kept because the label is re-rendered
        # from it whenever a fresh status snapshot lands (a pause requested
        # mid-datapoint changes no state — see _refresh_state_label()).
        self._state_name = OrchestratorState.IDLE.value
        self._state_label = QLabel(f"State: {self._state_name}")
        self._status_bar.addWidget(self._state_label)
        # Current status-bar 'level' ("", "active", "error"); tracked so the
        # dynamic-property restyle only fires when the level actually changes.
        self._status_level = ""

    def _build_header(self) -> QHBoxLayout:
        """Build the top toolbar with title and global action buttons.

        Returns:
            A QHBoxLayout containing the header widgets.
        """
        row = QHBoxLayout()

        title = QLabel("<b>I2AS</b>  — Instrument Monitor")
        row.addWidget(title)

        self._current_user_label = QLabel()
        self._current_user_label.setObjectName("current_user_label")
        self._sync_current_user_label()
        row.addWidget(self._current_user_label)

        # Slim page switcher: Page 1 (Monitor, the quadrant grid, unchanged)
        # / Page 2 (Logs). Not connected here — the pages
        # it switches between are built later in _build_ui(); the connection
        # is made once both exist, at the end of _build_ui().
        self._page_tab_bar = QTabBar()
        self._page_tab_bar.setObjectName("page_tab_bar")
        self._page_tab_bar.addTab("Monitor")
        self._page_tab_bar.addTab("Logs")
        self._page_tab_bar.setExpanding(False)
        row.addWidget(self._page_tab_bar)

        row.addStretch()

        # The takeover strip: the kill switch, the attendance toggle and the
        # "agents active" indicator, in the header because taking the machine
        # back must never be somewhere you have to go and find. Its own
        # controls are never gated — see takeover_strip.py.
        self._takeover_strip = TakeoverStrip(
            self._orchestrator,
            self._mirror,
            self._session_manager,
            parent=self,
        )
        row.addWidget(self._takeover_strip)

        # Monitoring toggle: the Orchestrator polls no instrument until
        # monitoring is started (typically after "Initiate All" has brought
        # the instruments up), and can be stopped again in IDLE to debug an
        # instrument by hand. Checked state mirrors the Orchestrator via
        # monitoring_changed — never set optimistically from the click alone.
        self._monitoring_btn = QPushButton()
        self._monitoring_btn.setObjectName("monitoring_btn")
        self._monitoring_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._monitoring_btn.setCheckable(True)
        self._monitoring_btn.clicked.connect(self._on_monitoring_clicked)
        self._sync_monitoring_btn()
        row.addWidget(self._monitoring_btn)

        initiate_all_btn = QPushButton("Initiate All")
        initiate_all_btn.setObjectName("initiate_all_btn")
        initiate_all_btn.setProperty("class", BTN_CLASS_PRIMARY)
        initiate_all_btn.setIcon(qta.icon("fa5s.play", color=TEXT_ON_ACCENT))
        initiate_all_btn.setToolTip("Bring every instrument to its operating state")
        initiate_all_btn.clicked.connect(
            lambda: self._orchestrator.submit_global_action("initiate_all")
        )

        standby_all_btn = QPushButton("Standby All")
        standby_all_btn.setObjectName("standby_all_btn")
        standby_all_btn.setProperty("class", BTN_CLASS_SECONDARY)
        standby_all_btn.setIcon(qta.icon("fa5s.power-off", color=TEXT_PRIMARY))
        standby_all_btn.setToolTip("Return every instrument to a safe standby state")
        standby_all_btn.clicked.connect(
            lambda: self._orchestrator.submit_global_action("standby_all")
        )

        row.addWidget(initiate_all_btn)
        row.addWidget(standby_all_btn)
        return row

    def _on_monitoring_clicked(self, checked: bool) -> None:
        """Start or stop monitoring from the header toggle.

        The Orchestrator may refuse a stop (outside IDLE/ERROR the tick loop
        must keep running both so hold enforcement can keep re-asserting
        standby on any held VI — GLOSSARY.md's **Hold enforcement** — and so
        the stall detector and stale detection keep watching active hardware;
        the refusal reason arrives on ``action_blocked`` and shows in the
        banner), so the
        button is re-synced from the confirmed state rather than left at the
        clicked position.

        Args:
            checked: The button's new checked state after the click.
        """
        if checked:
            self._orchestrator.start_monitoring()
        else:
            self._orchestrator.stop_monitoring()
        self._sync_monitoring_btn()

    def _sync_monitoring_btn(self) -> None:
        """Mirror the Orchestrator's confirmed monitoring state onto the toggle."""
        monitoring = self._mirror.is_monitoring()
        btn = self._monitoring_btn
        btn.setChecked(monitoring)
        btn.setText("Stop Monitoring" if monitoring else "Start Monitoring")
        btn.setIcon(
            qta.icon("fa5s.eye-slash" if monitoring else "fa5s.eye", color=TEXT_PRIMARY)
        )
        btn.setToolTip(
            "Stop polling instrument state (allowed only while idle — e.g. to "
            "debug an instrument by hand)"
            if monitoring
            else "Start polling instrument state each tick (do this once the "
            "instruments have been initiated)"
        )

    def _build_instruments_quadrant(self, measurement_vis: list[str]) -> QWidget:
        """Build the top-left quadrant: a scrollable 2-column grid of ALL VI cards.

        System VIs come first (config order, untagged), then measurement VIs
        as tagged cards — full citizens of the instrument grid since the Other
        Devices section was retired. Panels are built once and kept in
        self._panels for the lifetime of the window — recreating them would
        drop their Orchestrator signal connections.

        Args:
            measurement_vis: Names of measurement VIs, rendered tagged.

        Returns:
            A QWidget containing the title, and a QScrollArea of InstrumentPanels.
        """
        container = QWidget()
        container.setObjectName("instruments_quadrant")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)
        outer.addWidget(QLabel("<b>Instruments</b>"))

        entries: list[tuple[str, str | None]] = [
            (n, None) for n in self._system_vi_names
        ]
        entries += [(n, "Measurement") for n in measurement_vis]

        self._panels: list[InstrumentPanel] = []
        self._offline_cards: dict[str, OfflineInstrumentPanel] = {}
        grid_container = QWidget()
        grid = QGridLayout(grid_container)
        grid.setSpacing(6)
        self._instruments_grid = grid
        for idx, (vi_name, type_tag) in enumerate(entries):
            panel = self._make_live_panel(vi_name, type_tag)
            self._panels.append(panel)
            row, col = divmod(idx, 2)
            grid.addWidget(panel, row, col)

        # Offline instruments (degraded build) render after the live cards as
        # control-free fault cards; a successful reconnect swaps the card for
        # a live panel in place (_on_instrument_reconnected).
        tag_by_type = {"measurement": "Measurement"}
        for offset, vi_name in enumerate(self._station.offline_vi_names()):
            card = OfflineInstrumentPanel(
                vi_name,
                self._orchestrator,
                self._mirror,
                parent=self,
                type_tag=tag_by_type.get(self._role_of(vi_name)),
            )
            card.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self._offline_cards[vi_name] = card
            row, col = divmod(len(entries) + offset, 2)
            grid.addWidget(card, row, col)

        scroll = QScrollArea()
        scroll.setObjectName("instruments_scroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(grid_container)
        outer.addWidget(scroll)
        return container

    def _make_live_panel(
        self, vi_name: str, type_tag: str | None
    ) -> InstrumentPanel:
        """Construct one live VI card (shared by initial build and reconnect).

        Args:
            vi_name: The registered VI's name.
            type_tag: Role label for tagged cards ("Measurement"), None for
                system cards.

        Returns:
            The wired InstrumentPanel, size policy applied.
        """
        panel = InstrumentPanel(
            vi_name,
            self._orchestrator,
            self._mirror,
            parent=self,
            panel_controls=self._panels_config.get(vi_name),
            type_tag=type_tag,
        )
        panel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        return panel

    def _role_of(self, vi_name: str) -> str:
        """Return one configured VI's config-registry role, from the declaration.

        Args:
            vi_name: The VI's configured name.

        Returns:
            ``"system"`` or ``"measurement"``, or ``""``
            when the declaration names no such instrument.
        """
        info = self._mirror.instrument_info(vi_name)
        return info.role if info is not None else ""

    def _on_instrument_reconnected(self, vi_name: str) -> None:
        """Swap an offline card for a live InstrumentPanel in place.

        The card that leaves goes out through ``retire_widget()`` (the
        card-retirement standard, gui/widget_lifecycle.py): hidden and out of
        the grid before its deferred delete, so it cannot paint over its
        replacement in the meantime.

        Args:
            vi_name: The VI just brought live by
                Orchestrator.connect_instrument().
        """
        # Popped before the replacement is built: the pop is what makes a
        # re-entrant reconnect signal (a second emission while this one is
        # still running) a no-op instead of a second swap.
        card = self._offline_cards.pop(vi_name, None)
        if card is None:
            return
        tag_by_type = {"measurement": "Measurement"}
        panel = self._make_live_panel(vi_name, tag_by_type.get(self._role_of(vi_name)))
        self._panels.append(panel)
        self._instruments_grid.replaceWidget(card, panel)
        panel.show()
        card.close_details()
        retire_widget(card, self._instruments_grid)
        logger.info("Offline card for '%s' replaced by live panel", vi_name)

    def _on_instrument_disconnected(self, vi_name: str) -> None:
        """Swap a live InstrumentPanel for an offline card in place.

        The exact inverse of ``_on_instrument_reconnected()`` — the GUI half
        of the connection-lifecycle standard's "a disconnected instrument
        degrades exactly like one that never connected". The card is swapped
        rather than restyled because the live panel's controls, monitored
        values and lifecycle toggle all describe an instrument I2AS no
        longer holds; showing them greyed out would invite clicks that can
        only be refused.

        The panel that leaves goes out through ``retire_widget()`` (the
        card-retirement standard, gui/widget_lifecycle.py) — deferred, because
        this runs inside the click signal of the Disconnect button on the very
        card being retired.

        Args:
            vi_name: The VI just released by
                Orchestrator.disconnect_instrument().
        """
        panel = next((p for p in self._panels if p.vi_name == vi_name), None)
        if panel is None:
            return
        # Dropped from the panel list before the replacement is built, the
        # mirror of the pop in _on_instrument_reconnected(): a re-entrant
        # disconnect signal then finds no panel and returns.
        self._panels.remove(panel)
        tag_by_type = {"measurement": "Measurement"}
        card = OfflineInstrumentPanel(
            vi_name,
            self._orchestrator,
            self._mirror,
            parent=self,
            type_tag=tag_by_type.get(self._role_of(vi_name)),
        )
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._offline_cards[vi_name] = card
        self._instruments_grid.replaceWidget(panel, card)
        card.show()
        panel.close_front_panel()
        retire_widget(panel, self._instruments_grid)
        logger.info("Live panel for '%s' replaced by offline card", vi_name)

    def _build_bottom_right_quadrant(self) -> QWidget:
        """Build the bottom-right quadrant: Ramps over Agents.

        The always-present ``RampTrackerPanel`` on top (every ramp running
        right now, each with its own Abort), the ``AgentPanel`` underneath
        ("what did the machines do"). Stacked rather than side by side
        because each agent row is a single wide line that a column would only
        wrap.

        Returns:
            A vertical QSplitter holding the Ramps and Agents sub-panels.
        """
        column = QSplitter(Qt.Orientation.Vertical)
        column.setObjectName("agents_splitter")
        column.setChildrenCollapsible(False)
        column.addWidget(self._build_ramps_subpanel())
        column.addWidget(self._build_agents_subpanel())
        column.setSizes(list(_BOTTOM_RIGHT_VERTICAL_SIZES))
        column.setStretchFactor(0, _BOTTOM_RIGHT_VERTICAL_STRETCH[0])
        column.setStretchFactor(1, _BOTTOM_RIGHT_VERTICAL_STRETCH[1])
        self._agents_splitter = column
        return column

    def _build_agents_subpanel(self) -> QWidget:
        """Build the bottom-right quadrant's bottom sub-panel: the Agent panel.

        Always built, like the ramp tracker: a setup with no agent shows the
        panel's own empty state, and "nothing has acted on my cryostat but
        me" is an answer the physicist should be able to read off the window
        rather than infer from an absent widget. The panel connects to no
        engine signal itself — the window forwards the two it filters (see
        ``_connect_signals``), which is the destruction-order rule.

        Returns:
            A QWidget containing the title and the panel.
        """
        container = QWidget()
        container.setObjectName("agents_quadrant")
        container.setMinimumHeight(_AGENTS_MIN_HEIGHT)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)
        outer.addWidget(QLabel("<b>Agents</b>"))
        self._agent_panel = AgentPanel(
            session_manager=self._session_manager, parent=self
        )
        outer.addWidget(self._agent_panel)
        return container

    def _build_ramps_subpanel(self) -> QWidget:
        """Build the bottom-right quadrant's top sub-panel: the ramp tracker.

        Always built — a setup with no rampable VI simply shows the panel's
        own "No ramps running." empty state, and every setup has at least
        one system VI in practice. ``ramps_updated`` is connected on the
        window, not here (see ``_connect_signals``).

        Returns:
            A QWidget containing the title and the scrolled tracker.
        """
        container = QWidget()
        container.setObjectName("ramps_quadrant")
        # Layout rule: a pane that displays data gets a real minimum so the
        # splitter can never crush a ramp row's numbers out of readability.
        container.setMinimumWidth(_RAMPS_MIN_WIDTH)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)
        outer.addWidget(QLabel("<b>Ramps</b>"))

        self._ramp_tracker = RampTrackerPanel(self._orchestrator, parent=self)
        ramps_scroll = QScrollArea()
        ramps_scroll.setObjectName("ramps_scroll")
        ramps_scroll.setWidgetResizable(True)
        ramps_scroll.setWidget(self._ramp_tracker)
        outer.addWidget(ramps_scroll)
        return container


    def _on_page_changed(self, index: int) -> None:
        """Switch the central QStackedWidget's page.

        Args:
            index: The tab bar's new current index (0 = Monitor, 1 = Logs).
        """
        self._page_stack.setCurrentIndex(index)


    # ------------------------------------------------------------------
    # Public sample-info accessors (used by ProcedureWindow)
    # ------------------------------------------------------------------

    def get_sample_info(self) -> dict[str, str]:
        """Return the current sample info as a dict.

        Returns:
            Dict with keys ``sample_name``, ``sample_id``, ``comments``.
        """
        return self._session_info.get_sample_info()

    def get_data_dir(self) -> str:
        """Return the configured data directory path.

        Returns:
            Absolute path string; falls back to the open experiment's own
            data folder, or (no experiment open)
            ``i2as.core.paths.measurement_root()``, if the field is
            empty (``ExperimentInfoPanel.get_data_dir``). Not enforced — a
            plain read, used by autosave. Callers that actually start a run
            use ``get_data_dir_for_run()`` instead.
        """
        return self._session_info.get_data_dir()

    def get_data_dir_for_run(self) -> str | None:
        """Return the data dir, enforcing hard containment before a run starts.

        Experiment-directory containment is hard, not a warning (see
        ``GLOSSARY.md``'s **Session**): when an experiment is open and the
        configured Data Dir resolves outside that experiment's folder, the
        run is refused rather than merely noted.
        No experiment open is unaffected (session-less legacy state).

        Returns:
            The data directory path, or ``None`` if it was rejected (a
            warning dialog has already been shown; the caller must abort).
        """
        if not self._session_info.is_data_dir_contained():
            QMessageBox.warning(
                self,
                "Data directory outside experiment",
                "The data directory is outside the open experiment's folder. "
                "Choose a directory inside it before starting a run.",
            )
            return None
        return self.get_data_dir()

    def get_experiment_info(self) -> dict[str, str]:
        """Return the session layer's experiment context for procedure stamping.

        Returns:
            ``ExperimentManager.experiment_context()`` (experiment id/title, user
            identity), or ``{}`` when no session layer is wired or no
            experiment is open.
        """
        if self._session_manager is None:
            return {}
        return self._session_manager.experiment_context()

    # ------------------------------------------------------------------
    # Session persistence (content tier: sample info, procedure params, queue)
    # ------------------------------------------------------------------

    def _collect_session_state(self) -> form_autosave.FormAutosaveState:
        """Build a FormAutosaveState from the current UI, preserving procedure data.

        The Sample Info fields are read live. The procedure selection,
        parameters, and queue come from the open ProcedureWindow if there is
        one; otherwise the values loaded at startup are preserved unchanged.
        """
        info = self.get_sample_info()
        state = form_autosave.FormAutosaveState(
            sample_name=info["sample_name"],
            sample_id=info["sample_id"],
            comments=info["comments"],
            data_dir=self.get_data_dir(),
            selected_procedure=self._session.selected_procedure,
            procedure_params=self._session.procedure_params,
            queue=self._session.queue,
        )
        if self._procedure_window is not None:
            self._procedure_window.export_session_state(state)
        return state

    def _save_session(self) -> None:
        """Persist the current session to disk, tolerating write failures.

        When an experiment is open, GUI state follows the session bundle —
        ``session_manager.current_gui_state_path()`` inside the session
        folder — instead of the per-user AppData file, and the run queue is
        additionally promoted into the experiment record itself via
        ``set_queue`` so it survives independently of the autosave file. With
        no experiment open, behavior is unchanged (the per-user AppData
        file).
        """
        self._session = self._collect_session_state()
        session_manager = self._session_manager
        is_session_open = (
            session_manager is not None and session_manager.current_experiment() is not None
        )
        if is_session_open:
            path = session_manager.current_gui_state_path()
        else:
            path = app_settings.autosave_file_path(self._current_user_id)
        try:
            form_autosave.save(self._session, path)
        except OSError as exc:
            logger.warning("MonitorWindow: could not save session: %s", exc)
        if is_session_open:
            session_manager.set_queue([item.to_dict() for item in self._session.queue])

    def _on_new_session(self) -> None:
        """Clear the session to defaults after user confirmation."""
        reply = QMessageBox.question(
            self,
            "New Session",
            "Clear the current session (sample info, parameters, and queue) "
            "and start fresh?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._session = form_autosave.FormAutosaveState()
        self._session_info.apply_session(self._session)
        if self._procedure_window is not None:
            self._procedure_window.reset_session()
        self._save_session()

    def _open_eln_settings(self) -> None:
        """Open the **eLab setup dialog** over the publisher's settings.

        The same dialog the procedure window's **eLab tab** opens, over the
        same record. Saving writes the user-level settings file and hands
        the new record to the publisher; with no publisher wired there is
        nothing to edit, and the window says so rather than showing a form
        that could not be applied.
        """
        settings = getattr(self._eln_publisher, "settings", None)
        if settings is None:
            QMessageBox.information(
                self,
                "eLab notebook",
                "No electronic lab notebook is wired into this session.",
            )
            return

        def _save(edited: Any) -> None:
            """Write the edited settings and reload the publisher.

            Args:
                edited: The ``ElnSettings`` the dialog's form produced.
            """
            persist_eln_settings(edited, self._eln_publisher)
            if self._procedure_window is not None:
                self._procedure_window.reload_analysis_panel()

        ElnSettingsDialog(settings, on_save=_save, parent=self).exec()

    def _open_login_dialog(self) -> None:
        """Open LoginDialog and switch to the picked user, if any."""
        if self._session_manager is None:
            QMessageBox.information(
                self, "Log In", "Session management is not available."
            )
            return
        dialog = LoginDialog(
            self._session_manager.roster, self._current_user_id, self
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self._switch_user(dialog.selected_user_id())

    def _switch_user(self, user_id: str) -> None:
        """Save the outgoing user's fields and load the incoming user's own.

        Only the Session Info panel's sample fields, data dir, and app
        settings persistence follow the switch; a ProcedureWindow already
        open keeps its in-memory queue/params from before the switch (it
        re-reads whoever is current the next time it is built).

        Args:
            user_id: The roster id to switch to.
        """
        self._save_session()
        self._current_user_id = user_id
        app_settings.set_current_user_id(user_id)
        self._session = form_autosave.load(app_settings.autosave_file_path(user_id))
        self._session_info.apply_session(self._session)
        self._sync_current_user_label()

    def _open_load_session_dialog(self) -> None:
        """Open OpenExperimentDialog and switch to the picked experiment, if any."""
        if self._session_manager is None:
            QMessageBox.information(
                self, "Load Session", "Session management is not available."
            )
            return
        dialog = OpenExperimentDialog(self._session_manager, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        experiment_id = dialog.selected_experiment_id()
        if experiment_id:
            self._switch_experiment(experiment_id)

    def _open_resume_session_dialog(self) -> None:
        """Open ResumeSessionDialog and persist the picked/created session as active.

        Deferred-until-restart, same precedent the old sessions-root relocate
        action used: ``session_manager``'s own ``ExperimentStore`` stays fixed
        for the rest of this run regardless of what is picked here (see
        ``GLOSSARY.md``'s **Session**).
        """
        if self._session_store is None:
            QMessageBox.information(
                self, "Resume Session", "Session management is not available."
            )
            return
        user_id = self._current_user_id or GUEST_USER_ID
        dialog = ResumeSessionDialog(self._session_store, user_id, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        session_id = dialog.selected_session_id()
        if not session_id:
            return
        self._session_store.set_active(user_id, session_id)
        self._status_bar.showMessage(
            "Session updated — applies fully on next launch", 5000
        )

    def _switch_experiment(self, experiment_id: str) -> None:
        """Save the outgoing session's fields and load the incoming session's own.

        Mirrors ``_switch_user``'s save-outgoing/load-incoming shape one
        level up, at the L6 session-record level: (1) persist the current
        GUI state to wherever it is currently targeted, (2) ask
        ``ExperimentManager`` to switch — a ``ValueError`` (unknown id, or the
        target is not open) surfaces as a warning and aborts before anything
        else changes, (3) load the target's own ``gui_state.json`` (a default
        ``FormAutosaveState`` when it has none yet) and apply it to the Session
        Info panel. A ``ProcedureWindow`` already open keeps its in-memory
        queue/params from before the switch — the same documented limitation
        ``_switch_user`` has — it re-reads whoever is current only the next
        time it is (re)built.

        Args:
            experiment_id: The store key of an open experiment to switch to.
        """
        if self._session_manager is None:
            return
        self._save_session()
        try:
            self._session_manager.switch_experiment(experiment_id)
        except ValueError as exc:
            QMessageBox.warning(self, "Could not switch session", str(exc))
            return
        gui_state_path = self._session_manager.current_gui_state_path()
        self._session = (
            form_autosave.load(gui_state_path)
            if gui_state_path is not None and gui_state_path.exists()
            else form_autosave.FormAutosaveState()
        )
        self._session_info.apply_session(self._session)

    def _sync_current_user_label(self) -> None:
        """Reflect the current login in the header label."""
        if not self._current_user_id:
            self._current_user_label.setText("Not logged in")
            return
        name = self._current_user_id
        if self._session_manager is not None:
            user = self._session_manager.roster.get(self._current_user_id)
            if user is not None and user.name:
                name = user.name
        self._current_user_label.setText(f"Logged in as {name}")

    def _open_instrument_info(self) -> None:
        """Open a read-only view of each VI's devices.yaml metadata block."""
        metadata = (
            read_instrument_metadata(self._active_config_path)
            if self._active_config_path
            else {}
        )
        InstrumentInfoDialog(metadata, self).exec()

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._orchestrator.monitoring_changed.connect(
            lambda _on: self._sync_monitoring_btn()
        )
        self._orchestrator.instrument_reconnected.connect(
            self._on_instrument_reconnected
        )
        self._orchestrator.instrument_disconnected.connect(
            self._on_instrument_disconnected
        )
        self._orchestrator.state_changed.connect(self._on_state_changed)
        self._orchestrator.error_occurred.connect(self._on_error)
        self._orchestrator.error_event.connect(self._on_error_event)
        self._orchestrator.action_blocked.connect(self._on_action_blocked)
        self._orchestrator.action_failed.connect(self._on_action_failed)
        self._orchestrator.action_succeeded.connect(self._on_action_confirmed)
        # Separate from InstrumentPanel's own states_updated connections
        # (each panel connects itself in its constructor) — this slot only
        # feeds the Trends quadrant.
        self._orchestrator.states_updated.connect(self._on_states_updated)
        # ramps_updated likewise fires every tick, so it routes through this
        # window rather than connecting the tracker directly (gui-edit
        # skill's destruction-order rule).
        self._orchestrator.ramps_updated.connect(self._on_ramps_updated)
        # The held-VI set and the override window are mirror reads with no
        # state transition to piggyback on — a hold-only condition never
        # changes the engine's state — so the ack controls refresh when the
        # mirror does, which is once per tick and on every state change.
        self._mirror.status_updated.connect(self._on_status_snapshot)
        # The Agent panel is a FILTER of these two streams, and this window is
        # the receiver for both — a panel that connected itself would keep a
        # live connection into a tree Qt is already tearing down (the
        # destruction-order rule).
        # Both channels under whichever name this client carries them: the
        # engine's own ``verdict_emitted``/``event_emitted``, which the proxy
        # renames to ``verdict``/``event`` because a client CONSUMES the
        # contract rather than relaying it. The engine's names are tried
        # first, because ``event`` is also QObject's own virtual handler and
        # every QObject answers to it.
        verdict_stream = getattr(self._orchestrator, "verdict_emitted", None)
        if verdict_stream is None:
            verdict_stream = self._orchestrator.verdict
        verdict_stream.connect(self._on_verdict_for_agents)
        event_stream = getattr(self._orchestrator, "event_emitted", None)
        if event_stream is None:
            event_stream = self._orchestrator.event
        event_stream.connect(self._on_event_for_agents)
        self._agent_panel.agents_active_changed.connect(
            self._takeover_strip.set_agents_active
        )
        if self._session_manager is not None:
            self._session_manager.experiment_changed.connect(
                self._on_session_experiment_changed
            )
            self._session_manager.store_health_changed.connect(
                self._on_store_health_changed
            )

    def _on_session_experiment_changed(self, record: dict) -> None:
        """Load a newly opened/switched session's gui_state.json, if it has one.

        Connected to ``ExperimentManager.experiment_changed`` for every path
        that can bring a *different* experiment live — Start Experiment,
        ``switch_experiment``, and the resume-on-construction case already
        handled once in ``__init__``. A brand-new experiment (just started
        via Start Experiment) has no ``gui_state.json`` yet — doing nothing
        in that case is what stops a default ``FormAutosaveState`` from wiping
        the sample fields the physicist just typed. A same-experiment
        re-emit (attendance/findings edits) is ignored outright.

        Args:
            record: ``ExperimentRecord.to_dict()``, or ``{}`` when none open.
        """
        experiment_id = record.get("experiment_id", "") if record else ""
        if not experiment_id or experiment_id == self._last_session_experiment_id:
            self._last_session_experiment_id = experiment_id or None
            return
        self._last_session_experiment_id = experiment_id
        if self._session_manager is None:
            return
        gui_state_path = self._session_manager.current_gui_state_path()
        if gui_state_path is None or not gui_state_path.exists():
            return
        self._session = form_autosave.load(gui_state_path)
        self._session_info.apply_session(self._session)

    def _on_store_health_changed(self, info: dict) -> None:
        """Surface a session-record save failure/recovery via the banner + status bar.

        ``ok=False`` shows a persistent banner error — the physicist should
        know before losing work that the record is not reaching disk.
        ``ok=True`` clears it (the banner's own dismiss, not another
        message) and confirms recovery as a routine status-bar note instead
        of a second banner.

        Args:
            info: ``{"ok": bool, "detail": str}`` from
                ``ExperimentManager.store_health_changed``.
        """
        if info.get("ok"):
            self._banner.dismiss()
            self._status_bar.showMessage("Session record saving recovered", 5000)
            return
        detail = info.get("detail", "")
        self._banner.show_message(
            f"Session record is NOT being saved: {detail}", BANNER_SEVERITY_ERROR
        )

    def _on_states_updated(self, state: dict) -> None:
        """Forward the per-tick state snapshot to the Trends quadrant.

        The WINDOW (not the child panels) is the connection receiver on
        purpose: Qt severs a receiver's connections at the start of its own
        destruction, so routing the tick through the window guarantees the
        Orchestrator's still-running timer can never reach a partially
        destroyed child tree. Connecting the panels directly re-introduced a
        teardown race (RuntimeError/segfault on a deleted plot curve when a
        tick landed mid-destruction under pytest-qt).

        Args:
            state: ``{vi_name: {field: value, ...}}`` from the Orchestrator.
        """
        self._trends.on_states_updated(state)

        # Calm a shown fault-warning banner once every runtime fault has
        # cleared — but only if THIS banner is the one showing
        # (never steal a dismiss from an unrelated message, e.g. the
        # save-health error).
        if self._last_fault_message is not None and not self._mirror.vi_faults():
            self._banner.dismiss()
            self._last_fault_message = None

    def _on_error_event(self, event: ErrorEvent) -> None:
        """Show a per-VI fault warning on the banner.

        Only ``kind="fault"``/``severity="warning"`` events are handled
        here — everything more severe (``run_failure``, ``safety``,
        ``internal``) already reaches the banner via the compat
        ``error_occurred`` -> ``_on_error`` path, which fires alongside
        every such ``error_event`` (see ``Orchestrator._error()``).

        Args:
            event: The structured error/fault payload.
        """
        if event.severity != "warning" or event.kind != "fault":
            return
        message = f"{event.vi_name}: {event.message}" if event.vi_name else event.message
        if message == self._last_fault_message:
            return
        self._last_fault_message = message
        self._banner.show_message(message, BANNER_SEVERITY_WARNING)


    def _on_ramps_updated(self, records: list) -> None:
        """Forward this tick's running-ramp records to the Ramps sub-panel.

        Routed through the window for the same teardown-race reason as
        ``_on_states_updated`` — ``ramps_updated`` fires every tick, and
        LAST (``_publish_ramps()`` is the final step of ``_tick_body()``),
        after that tick's ``update_conditions()``/``decide()`` and any state
        transition it triggered.

        Args:
            records: ``list[i2as.core.ramps.RampRecord]`` from the
                Orchestrator (typed loosely here because a Qt ``list``
                signal payload carries no element type).
        """
        if self._ramp_tracker is not None:
            self._ramp_tracker.on_ramps_updated(records)

    def _on_status_snapshot(self, _snapshot: object) -> None:
        """Refresh the state-dependent header controls from the fresh mirror.

        Also re-renders every instrument card's lifecycle toggle.

        Args:
            _snapshot: The ``StatusSnapshot`` the mirror just absorbed; the
                controls read the mirror rather than the payload, so that
                one slot serves every read they make.
        """
        self._refresh_ack_controls()
        # A requested-but-not-yet-honoured pause changes no state, so the
        # state label has to follow the snapshot as well as state_changed.
        self._refresh_state_label()
        # The kill switch and attendance are values ANY client can change, so
        # the strip re-reads the mirror rather than trusting its own last
        # click; the activity count decays with time, so it is recomputed
        # here too rather than only when an agent acts.
        self._takeover_strip.sync_from_mirror()
        self._takeover_strip.set_agents_active(
            self._agent_panel.active_agent_count()
        )
        # Whose run is in flight (GLOSSARY.md's **Run owner**) — forwarded to
        # the panel from here rather than read there, for the same
        # destruction-order reason every other per-tick payload is.
        self._agent_panel.set_run_owner(self._mirror.run_owner())
        # Each instrument card's Initiate/Standby toggle renders the
        # lifecycle state this snapshot carries (GLOSSARY.md's **Lifecycle
        # state**), so a stand-down nobody clicked — an emergency's blanket
        # standby_all(), an agent through the gateway, the CLI — reaches the
        # card. Routed through this window rather than connected per panel:
        # the mirror emits at tick rate, and the destruction-order rule wants
        # the window as the receiver.
        for panel in self._panels:
            panel.on_status_snapshot()

    def _on_verdict_for_agents(self, verdict: object) -> None:
        """Forward one verdict to the panels that render verdicts.

        Two of them, for opposite halves of the same contract: the Agent
        panel keeps the non-operator ones, and the experiment header keeps
        the one answering its own Apply click.

        Args:
            verdict: Anything off the client's ``verdict`` stream.
        """
        self._agent_panel.on_verdict(verdict)
        self._session_info.on_verdict(verdict)

    def _on_event_for_agents(self, event: object) -> None:
        """Forward one event to the Agent panel.

        Args:
            event: Anything off the client's ``event`` stream; the panel keeps
                the non-operator ``StateChange``s and ignores the rest.
        """
        self._agent_panel.on_event(event)


    def _on_state_changed(self, state_name: str) -> None:
        """Update the status bar label and colour level when state changes.

        The status bar background is driven by a dynamic ``level`` QSS property
        (``""``/``"active"``/``"error"``). The restyle only fires when the level
        actually changes (same repolish pattern as the InstrumentPanel border).

        Args:
            state_name: The new state name string (e.g. ``"IDLE"``).
        """
        self._state_name = state_name
        self._refresh_state_label()
        logger.debug("MonitorWindow: orchestrator state → %s", state_name)

        self._in_emergency = state_name == OrchestratorState.EMERGENCY.value
        self._refresh_ack_controls()

        if state_name in _ERROR_STATES:
            level = "error"
        elif state_name in _ACTIVE_STATES:
            level = "active"
        else:
            level = ""

        if level != self._status_level:
            self._status_level = level
            self._status_bar.setProperty("level", level)
            # Repolish the child label too: descendant selectors like
            # QStatusBar[level="error"] QLabel are resolved per-widget, so
            # repolishing only the status bar leaves the label's old colour.
            for widget in (self._status_bar, self._state_label):
                widget.style().unpolish(widget)
                widget.style().polish(widget)

    def _refresh_state_label(self) -> None:
        """Render the status bar's state label, including a requested pause.

        A pause asked for while the run is MEASURING is *deferred* to the
        pause boundary (GLOSSARY.md's **Pause boundary**), so for the length
        of that datapoint the state is still MEASURING and the only thing
        that has changed is a flag on the status snapshot. Rendering it as
        ``MEASURING · Pausing`` is what tells the operator their click was
        taken — otherwise the window looks identical before and after it, and
        the pause reads as ignored until the state finally moves.

        Text only: no dynamic property, no new colour. The status-bar level
        still follows the STATE (a pending pause is not an error, and the run
        is still active), so nothing here can drift from the palette.
        """
        pausing = " · Pausing" if self._mirror.pause_pending() else ""
        self._state_label.setText(f"State: {self._state_name}{pausing}")

    def _on_ack_clicked(self) -> None:
        """Acknowledge, then refresh immediately rather than waiting for the
        next tick's ``ramps_updated`` — otherwise the countdown label stays
        blank for up to one tick interval after the click, which reads as
        the click not having registered.
        """
        self._orchestrator.acknowledge()
        self._refresh_ack_controls()

    def _refresh_ack_controls(self) -> None:
        """Sync the ACKNOWLEDGE button and its "Acknowledged (mm:ss)" countdown.

        Called on every ``state_changed`` AND every tick's ``ramps_updated``
        (not just state transitions) — a hold-severity condition can appear
        or clear mid-IDLE with no state transition at all, so the button
        must not wait for one to show up or disappear; ``ramps_updated``,
        not ``states_updated``, is the tick-driven trigger deliberately,
        since it fires AFTER that tick's condition computation (see
        ``_on_ramps_updated()``'s docstring) — using ``states_updated``
        would show the previous tick's held-VI set. The countdown is a
        plain top-bar label rather than a popup: it never steals focus, and
        it reports the SAME override window every subsequent action either
        succeeds or is refused against (Orchestrator.acknowledge()), so a
        refusal after it reads 00:00 is never a silent surprise.
        """
        held = self._mirror.held_vi_names()
        self._ack_btn.setVisible(self._in_emergency or bool(held))
        self._ack_btn.setText(
            "Acknowledge emergency" if self._in_emergency else "Acknowledge & unlock"
        )
        self._refresh_hold_banner(held)
        expires_at = self._mirror.manual_override_expires_at()
        if expires_at is None:
            self._ack_countdown_label.setVisible(False)
            return
        remaining = max(0.0, expires_at - time.time())
        minutes, seconds = divmod(int(remaining), 60)
        self._ack_countdown_label.setText(
            f"Acknowledged ({minutes:02d}:{seconds:02d}) — held VIs return to standby at 00:00"
        )
        self._ack_countdown_label.setVisible(True)

    def _refresh_hold_banner(self, held: frozenset[str]) -> None:
        """Show why the ACKNOWLEDGE & UNLOCK button appeared, on the banner.

        A plain hold-severity condition (one that never escalated to
        EMERGENCY) drives ``_ack_btn``'s visibility but never goes
        through ``Orchestrator._error()`` — that path is reserved for
        ``internal``/``run_failure``/EMERGENCY-severity events (see
        ``_on_error()``) — so nothing else ever puts its description on the
        banner. This fills that gap from the same public
        ``get_operational_status()`` conditions list that
        ``held_vi_names()`` is derived from, without duplicating or
        pre-empting the EMERGENCY message path.

        Args:
            held: The currently held VI names, as returned by
                ``Orchestrator.held_vi_names()`` (only used to short-circuit
                when nothing is held).
        """
        if self._in_emergency or not held:
            if self._last_hold_message is not None:
                self._banner.dismiss()
                self._last_hold_message = None
            return

        conditions = self._mirror.get_operational_status().get("conditions", [])
        hold_conditions = [c for c in conditions if c.get("severity") == "hold"]
        if not hold_conditions:
            if self._last_hold_message is not None:
                self._banner.dismiss()
                self._last_hold_message = None
            return

        message = "; ".join(
            f"{c['message']} — affecting {', '.join(c.get('affected', []))}"
            if c.get("affected")
            else c["message"]
            for c in hold_conditions
        )
        if message == self._last_hold_message:
            return
        self._last_hold_message = message
        self._banner.show_message(message, BANNER_SEVERITY_WARNING)

    def _on_error(self, message: str) -> None:
        """Show a non-modal error banner when ERROR or EMERGENCY is entered.

        Replaces the old blocking ``QMessageBox.critical`` so repeated error
        signals no longer stack modal dialogs over the GUI.

        Args:
            message: Human-readable error description.
        """
        logger.error("MonitorWindow: %s", message)
        self._banner.show_message(message, BANNER_SEVERITY_ERROR)

    def _on_action_blocked(self, message: str) -> None:
        """Show a non-modal warning banner when the Orchestrator blocks an action.

        Args:
            message: Human-readable reason the action was blocked.
        """
        self._banner.show_message(message, BANNER_SEVERITY_WARNING)

    def _on_action_failed(self, vi_name: str, method_name: str, reason: str) -> None:
        """Show a non-modal error banner when a submitted GUI action raises.

        This is the uniform failure verdict of the control-validation
        standard: limit rejections and VI safety-interlock refusals arrive
        here with the reason string the VI wrote for the user.

        Args:
            vi_name: The VI the action targeted.
            method_name: The @control method that was called.
            reason: The exception message explaining why it was refused.
        """
        self._banner.show_message(
            f"{vi_name}.{method_name} failed: {reason}", BANNER_SEVERITY_ERROR
        )

    def _on_action_confirmed(self, vi_name: str, method_name: str) -> None:
        """Confirm a successful GUI action with a transient status-bar message.

        A self-expiring status-bar message (not the banner) on purpose:
        success is routine and should not demand a dismissal click, while
        failures (banner) must. ``showMessage`` temporarily overlays the
        permanent state label and restores it automatically.

        Args:
            vi_name: The VI the action targeted.
            method_name: The @control method that completed.
        """
        self._status_bar.showMessage(f"{vi_name}.{method_name} ✓ done", 4000)

    # ------------------------------------------------------------------
    # Window lifecycle + layout persistence
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt override)
        """Abort any active run, detach the log handler, persist window state.

        Closing the window used to leave live hardware exactly as it was —
        no abort, no standby — if a run was still active (SWEEPING, PAUSED,
        even mid-STANDBY before its safe-off commands finished dispatching):
        found live 2026-07-22 while chasing a Keithley 6221 that stayed
        armed/output-on across an app restart. ``abort_procedure()`` is
        documented safe to call with nothing running (a no-op past the state
        check) and already no-ops during EMERGENCY (that flow owns its own
        cleanup), so it is safe to call unconditionally here whenever state
        is not already IDLE.

        Detaching the log handler prevents it from writing to the destroyed
        ``QTextEdit`` after the window is gone (RuntimeError on a dead widget).
        Accepting the close is also what releases this window's own strong
        reference (the window-liveness standard, gui/widget_lifecycle.py): a
        closed window paints nothing, so it is safe to collect from here on.
        Splitter proportions and trend selections are saved automatically
        here (no separate "Save layout" action, unlike the old dock-state
        save/restore) — there is nothing else for the user to arrange since
        panels can't be hidden, closed, or moved out of their quadrant.

        Args:
            event: The Qt close event.
        """
        if self._mirror.state != OrchestratorState.IDLE.value:
            logger.warning(
                "MonitorWindow closing while orchestrator state=%s — aborting "
                "the active run to leave hardware safed.",
                self._mirror.state,
            )
            self._orchestrator.abort_procedure()
        self._save_session()
        self._log_panel.detach()
        settings = app_settings.get_settings()
        settings.setValue(_GEOMETRY_KEY, self.saveGeometry())
        settings.setValue(_MAIN_SPLITTER_KEY, self._main_splitter.saveState())
        settings.setValue(_LEFT_SPLITTER_KEY, self._left_splitter.saveState())
        settings.setValue(_RIGHT_SPLITTER_KEY, self._right_splitter.saveState())
        settings.setValue(_AGENTS_SPLITTER_KEY, self._agents_splitter.saveState())
        self._trends.save_settings()
        super().closeEvent(event)
        if event.isAccepted():
            release_window(self)

    def _restore_splitter_state(self) -> None:
        """Restore each quadrant splitter's saved proportions, defensively.

        ``QSplitter.restoreState()`` restores orientation and child count
        from the saved blob, not just sizes — applying a blob saved by a
        differently-shaped splitter (e.g. a stale value some other settings
        key never got cleared) would silently reshape this one. The
        orientation/count are captured before restoring and checked after;
        a mismatch reverts the restore rather than leaving a corrupted
        layout. A missing key or a ``restoreState()`` failure both silently
        keep the default proportions set in ``_build_ui()``.
        """
        settings = app_settings.get_settings()
        for splitter, key in (
            (self._main_splitter, _MAIN_SPLITTER_KEY),
            (self._left_splitter, _LEFT_SPLITTER_KEY),
            (self._right_splitter, _RIGHT_SPLITTER_KEY),
            (self._agents_splitter, _AGENTS_SPLITTER_KEY),
        ):
            state = settings.value(key)
            if state is None:
                continue
            expected_orientation = splitter.orientation()
            expected_count = splitter.count()
            expected_sizes = splitter.sizes()
            try:
                splitter.restoreState(state)
            except (TypeError, ValueError) as exc:
                logger.debug("MonitorWindow: could not restore %s: %s", key, exc)
                continue
            if splitter.orientation() != expected_orientation or splitter.count() != expected_count:
                logger.warning(
                    "MonitorWindow: %s restoreState() reshaped the splitter "
                    "(likely a stale settings value) — reverting to defaults.",
                    key,
                )
                splitter.setOrientation(expected_orientation)
                splitter.setSizes(expected_sizes)

    def _restore_monitor_state(self) -> None:
        """Restore trend panels and splitter proportions from QSettings, defensively.

        Called once at the end of ``__init__``, after the UI and menu are
        built. The saved trend count/keys are applied first (recreating the
        matching set of trend panels), then splitter proportions are
        restored. A missing key, wrong type, or corrupt JSON all silently
        fall back to the DEFAULT layout already built by ``_build_ui()``.
        """
        self._trends.restore_settings()
        self._restore_splitter_state()
