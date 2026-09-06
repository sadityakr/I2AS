"""InstrumentFrontPanel — the full-capability window for one VI."""

from __future__ import annotations

import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from i2as.core.events import InstrumentInfo
from i2as.core.orchestrator_proxy import OrchestratorProxy
from i2as.gui.instrument_panel import InstrumentPanel
from i2as.core.status_mirror import StatusMirror
from i2as.gui.theme import TEXT_PRIMARY


class InstrumentFrontPanel(QWidget):
    """Child window rendering one VI's complete monitored + control surface.

    Built entirely from the station's declaration snapshot, like the
    :class:`InstrumentPanel` it embeds: this window holds no VI object, and
    its one hardware action — the connection check — is submitted as a
    command like every other.

    Args:
        vi_name: The VI's registered name.
        orchestrator: OrchestratorProxy every action is submitted to.
        mirror: The status mirror the declaration is read from; built from
            the engine when none is given (the inline construction path).
        parent: The owning widget. The window is parented (so it is destroyed
            with the application) but flagged ``Qt.WindowType.Window`` so it
            floats as a real window.
    """

    def __init__(
        self,
        vi_name: str,
        orchestrator: OrchestratorProxy,
        mirror: StatusMirror | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setObjectName(f"{vi_name}_front_panel")
        self.setWindowTitle(f"{vi_name} — Instrument Front Panel")
        self.setMinimumSize(420, 320)
        self._vi_name = vi_name
        self._orchestrator = orchestrator
        self._mirror = (
            mirror if mirror is not None else StatusMirror.of(orchestrator)
        )
        info = self._mirror.instrument_info(vi_name) or InstrumentInfo(name=vi_name)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        # ── Connection check row: a harmless identity query, never an
        # arming/initiate action — the front panel's bench-test equivalent
        # of the old Other Devices "Check" button. It is bus traffic, so it
        # goes to the engine as a PING_INSTRUMENT command and the answer
        # comes back on the action signals, never by calling the VI. ──────
        check_row = QHBoxLayout()
        check_btn = QPushButton("Check connection")
        check_btn.setObjectName(f"{vi_name}_check_btn")
        check_btn.setIcon(qta.icon("fa5s.plug", color=TEXT_PRIMARY))
        check_btn.setToolTip(
            f"Send an identity query to test the {vi_name} connection "
            "(does not initiate or arm anything)"
        )
        self._check_status = QLabel("")
        self._check_status.setObjectName(f"{vi_name}_check_status")
        self._check_status.setProperty("class", "secondary_label")
        check_btn.clicked.connect(self._on_check)
        check_row.addWidget(check_btn)
        check_row.addWidget(self._check_status)
        check_row.addStretch()
        outer.addLayout(check_row)

        # The allowlist override shows EVERY control, regardless of each
        # control's panel= default or the setup's monitor.yaml allowlist.
        all_controls = [control.name for control in info.controls]
        self._panel = InstrumentPanel(
            vi_name,
            orchestrator,
            self._mirror,
            parent=self,
            panel_controls=all_controls,
            show_front_panel_button=False,
            grouped=True,
        )

        scroll = QScrollArea()
        scroll.setObjectName(f"{vi_name}_front_panel_scroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._panel)
        outer.addWidget(scroll)

        orchestrator.action_succeeded.connect(self._on_check_answered)
        orchestrator.action_failed.connect(self._on_check_refused)
        # This WINDOW is the receiver for the tick-rate mirror signal that
        # re-renders its embedded card, exactly as MonitorWindow is for the
        # cards in the instrument grid (the gui-edit skill's
        # destruction-order rule). It is what puts the lifecycle-state
        # standard's rendering on the front panel too, so the toggle here
        # cannot disagree with the one on the card.
        self._mirror.status_updated.connect(self._on_status_snapshot)

    def _on_status_snapshot(self, _snapshot: object) -> None:
        """Forward a fresh snapshot to the embedded panel.

        Args:
            _snapshot: The ``StatusSnapshot`` the mirror just absorbed; the
                panel reads the mirror rather than the payload.
        """
        self._panel.on_status_snapshot()

    def _on_check(self) -> None:
        """Submit the connection check; the verdict arrives on a signal."""
        self._check_status.setText("Checking…")
        self._orchestrator.ping_instrument(self._vi_name)

    def _on_check_answered(self, vi_name: str, method_name: str) -> None:
        """Report a reachable instrument inline.

        Args:
            vi_name: The VI the confirmed action was submitted for.
            method_name: The confirmed method name.
        """
        if vi_name == self._vi_name and method_name == "ping":
            self._check_status.setText("Connected")

    def _on_check_refused(
        self, vi_name: str, method_name: str, _reason: str
    ) -> None:
        """Report an unreachable instrument inline.

        Args:
            vi_name: The VI the failed action was submitted for.
            method_name: The failed method name.
            _reason: The engine's reason; the inline label keeps the short
                wording the button has always shown, and the full reason
                reaches the operator through the window's banner.
        """
        if vi_name == self._vi_name and method_name == "ping":
            self._check_status.setText("Not reachable")
