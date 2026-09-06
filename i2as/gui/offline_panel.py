"""OfflineInstrumentPanel — grid card and detail window for an offline VI."""

from __future__ import annotations

from dataclasses import dataclass

import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from i2as.core.orchestrator_proxy import OrchestratorProxy
from i2as.core.status_mirror import StatusMirror
from i2as.gui.lifecycle_toggle import ConnectionButton
from i2as.gui.theme import TEXT_PRIMARY

_HINT_TEXT = (
    "Check that the instrument is powered on and its cable and address match "
    "the config, then try to connect. For a deeper diagnosis run:\n"
    "python -m i2as.troubleshoot check"
)

_OPERATOR_HINT_TEXT = (
    "You released this instrument, so it is free for its own front panel or "
    "vendor software. Nothing was changed on it. Press Connect to hand it "
    "back to I2AS — then Initiate to bring it to its operating state."
)

_RECONNECT_FAILED_HINT_TEXT = (
    "You released this instrument, so it was free for its own front panel or "
    "vendor software — but the last attempt to hand it back to I2AS "
    "failed on hardware (see the reason above). Check that it is powered on "
    "and its cable and address match the config, then press Connect again. "
    "For a deeper diagnosis run:\n"
    "python -m i2as.troubleshoot check"
)


@dataclass(frozen=True)
class _OfflineWording:
    """The card/detail-window wording one offline availability tag set selects.

    the Availability record's ``tags`` (that standard's closed vocabulary,
    ``i2as.core.availability``) can hold ``"connect_failed"``,
    ``"operator"``, or both at once — an operator-released VI whose reconnect
    then failed on hardware. Each combination gets its own row so the
    two-tag case can say BOTH things are true, rather than a bool picking one
    over the other.

    Attributes:
        badge: Header badge text (``"[OFFLINE]"`` / ``"[DISCONNECTED]"``).
        note: One-line card note under the reason.
        header: Detail-window header sentence; ``{vi_name}`` is substituted.
        hint: Detail-window diagnosis/recovery hint.
        window_title: Detail-window title; ``{vi_name}`` is substituted.
    """

    badge: str
    note: str
    header: str
    hint: str
    window_title: str


_WORDING: dict[frozenset[str], _OfflineWording] = {
    frozenset({"connect_failed"}): _OfflineWording(
        badge="[OFFLINE]",
        note="Not connected at startup — all other instruments are unaffected.",
        header="{vi_name} failed to connect at startup.",
        hint=_HINT_TEXT,
        window_title="{vi_name} — Instrument Offline",
    ),
    frozenset({"operator"}): _OfflineWording(
        badge="[DISCONNECTED]",
        note="Released to its front panel — all other instruments are unaffected.",
        header="{vi_name} is disconnected — I2AS is not holding it.",
        hint=_OPERATOR_HINT_TEXT,
        window_title="{vi_name} — Instrument Disconnected",
    ),
    frozenset({"operator", "connect_failed"}): _OfflineWording(
        badge="[DISCONNECTED]",
        note=(
            "Released to its front panel, and the reconnect attempt then "
            "failed — all other instruments are unaffected."
        ),
        header=(
            "{vi_name} is disconnected — you released it, and the last "
            "attempt to hand it back failed on hardware."
        ),
        hint=_RECONNECT_FAILED_HINT_TEXT,
        window_title="{vi_name} — Instrument Disconnected",
    ),
}

_DEFAULT_WORDING = _WORDING[frozenset({"connect_failed"})]


def _wording_for(tags: frozenset[str]) -> _OfflineWording:
    """Return the `_OfflineWording` an offline VI's tag set selects.

    Args:
        tags: An Availability record's ``tags`` value — a non-empty subset
            of the Availability standard's absence tags,
            ``{"connect_failed", "operator"}`` (`i2as.core.availability`).

    Returns:
        The exact-match wording for `tags`; falls back to the
        `connect_failed`-only wording for any combination the offline
        registry does not actually produce (defensive).
    """
    return _WORDING.get(tags, _DEFAULT_WORDING)


class OfflineInstrumentPanel(QGroupBox):
    """Instrument-grid card for a VI I2AS does not currently hold.

    Covers every offline tag combination with one card, because the Station
    degrades them identically (see the connection-lifecycle standard on
    ``BaseVirtualInstrument``): an instrument that never connected, one the
    operator deliberately released, or both at once (a released instrument
    whose reconnect then failed on hardware). Only the label, the note and
    the hint differ, selected from ``info.tags`` via a tag-keyed mapping
    (``_wording_for()``), so the operator can tell "something is wrong" from
    "I did this" — or both.

    Control-free apart from the one action that applies here: Connect — the
    offline half of the ConnectionButton pair whose live-card half reads
    Disconnect. Everything else on the station keeps working around it.

    Args:
        vi_name: The VI's configured name.
        orchestrator: OrchestratorProxy handling the connect request.
        mirror: The status mirror this card reads its tags and its offline
            reason from; built from the engine when none is given (the
            inline construction path).
        parent: Optional Qt parent widget.
        type_tag: Optional role label (e.g. "Measurement"), mirroring
            the live cards so the grid stays recognisable.
    """

    def __init__(
        self,
        vi_name: str,
        orchestrator: OrchestratorProxy,
        mirror: StatusMirror | None = None,
        parent: QWidget | None = None,
        type_tag: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._vi_name = vi_name
        self._orchestrator = orchestrator
        self._mirror = (
            mirror if mirror is not None else StatusMirror.of(orchestrator)
        )
        self._details: OfflineFrontPanel | None = None  # lazily created
        self._tags = self._mirror.availability_tags(vi_name)
        wording = _wording_for(self._tags)
        self.setObjectName(f"{vi_name}_offline_card")

        outer = QVBoxLayout()
        outer.setSpacing(4)
        outer.setContentsMargins(8, 8, 8, 8)

        header_row = QHBoxLayout()
        self._name_label = QLabel(f"<b>{vi_name}</b>  {wording.badge}")
        self._name_label.setObjectName(f"{vi_name}_offline_name_label")
        self._name_label.setProperty("class", "panel_name_label")
        header_row.addWidget(self._name_label)
        if type_tag:
            tag_lbl = QLabel(type_tag)
            tag_lbl.setObjectName(f"{vi_name}_offline_type_tag")
            tag_lbl.setProperty("class", "secondary_label")
            header_row.addWidget(tag_lbl)
        header_row.addStretch()
        details_btn = QPushButton()
        details_btn.setObjectName(f"{vi_name}_offline_details_btn")
        details_btn.setIcon(qta.icon("fa5s.sliders-h", color=TEXT_PRIMARY))
        details_btn.setToolTip(
            "Open the offline-instrument details (full reason and the "
            "Connect action)"
        )
        details_btn.clicked.connect(self._open_details)
        header_row.addWidget(details_btn)
        outer.addLayout(header_row)

        reason_lbl = QLabel(self._mirror.offline_reason(vi_name))
        reason_lbl.setObjectName(f"{vi_name}_offline_reason")
        reason_lbl.setProperty("class", "secondary_label")
        reason_lbl.setWordWrap(True)
        outer.addWidget(reason_lbl)

        self._note_lbl = QLabel(wording.note)
        self._note_lbl.setObjectName(f"{vi_name}_offline_note")
        self._note_lbl.setProperty("class", "secondary_label")
        self._note_lbl.setWordWrap(True)
        outer.addWidget(self._note_lbl)

        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName(f"{vi_name}_offline_card_status")
        self._status_lbl.setProperty("class", "secondary_label")
        self._status_lbl.setWordWrap(True)
        outer.addWidget(self._status_lbl)

        # Connect lives in the body, not the header: an offline card has room
        # to spare where a live card's header does not, so the one action that
        # applies here gets its word rather than the live card's compact icon.
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 2, 0, 2)
        self._connect_btn = ConnectionButton(
            vi_name, "connect", self._on_connect_clicked, parent=self
        )
        action_row.addWidget(self._connect_btn)
        action_row.addStretch()
        outer.addLayout(action_row)

        outer.addStretch()
        self.setLayout(outer)

        orchestrator.action_failed.connect(self._on_action_failed)
        self.setMinimumWidth(300)
        self.setMinimumHeight(self.sizeHint().height())

        for widget in (self, self._name_label):
            widget.setProperty("status", "offline")
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def _on_connect_clicked(self) -> None:
        """Submit the connect request; the verdict arrives via signals."""
        self._status_lbl.setText("Connecting…")
        self._orchestrator.connect_instrument(self._vi_name)

    def _on_action_failed(self, vi_name: str, method_name: str, reason: str) -> None:
        """Report a failed connect attempt on the card itself.

        Success needs no handler here: MonitorWindow replaces this card with
        a live panel the moment ``instrument_reconnected`` fires. A failed
        reconnect of an ALREADY-offline VI can change its tags (the
        Availability standard's ``connect_instrument()`` bug fix: a failed
        reconnect of an operator-disconnected VI ADDS ``connect_failed``
        rather than overwriting ``operator`` — see
        ``i2as.core.station``), so the badge/note are re-derived here
        rather than staying fixed at construction.
        """
        if vi_name != self._vi_name or method_name != "connect":
            return
        self._status_lbl.setText(f"Still not reachable: {reason}")
        self._refresh_wording()

    def _refresh_wording(self) -> None:
        """Re-select badge/note from the offline registry's current tags."""
        self._tags = self._mirror.availability_tags(self._vi_name)
        wording = _wording_for(self._tags)
        self._name_label.setText(f"<b>{self._vi_name}</b>  {wording.badge}")
        self._note_lbl.setText(wording.note)

    def _open_details(self) -> None:
        """Lazily create and show this VI's offline detail window."""
        if self._details is None:
            self._details = OfflineFrontPanel(
                self._vi_name,
                self._orchestrator,
                mirror=self._mirror,
                parent=self.window(),
                tags=self._tags,
            )
        self._details.show()
        self._details.raise_()
        self._details.activateWindow()

    def close_details(self) -> None:
        """Close the detail window, if open (called before the card is
        replaced by a live panel on successful reconnect)."""
        if self._details is not None:
            self._details.close()
            self._details = None


class OfflineFrontPanel(QWidget):
    """Detail window for one offline VI: full reason, hint, Connect.

    The offline counterpart of :class:`InstrumentFrontPanel`. The connect
    request goes through ``Orchestrator.connect_instrument()`` (IDLE-gated);
    the verdict comes back via ``instrument_reconnected`` / ``action_failed``
    and is reported inline.

    Args:
        vi_name: The offline VI's configured name.
        orchestrator: OrchestratorProxy handling the connect request.
        mirror: The status mirror the live failure reason and the tags are
            read from; built from the engine when none is given.
        parent: The owning widget (parented, but flagged as a real window).
        tags: The offline VI's Availability tags (`i2as.core.availability`
            — a subset of ``{"connect_failed", "operator"}``), selecting the
            title/header/hint via ``_wording_for()``. The action and the
            code path are identical regardless of which tags apply.
    """

    def __init__(
        self,
        vi_name: str,
        orchestrator: OrchestratorProxy,
        mirror: StatusMirror | None = None,
        parent: QWidget | None = None,
        tags: frozenset[str] = frozenset({"connect_failed"}),
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._vi_name = vi_name
        self._orchestrator = orchestrator
        self._mirror = (
            mirror if mirror is not None else StatusMirror.of(orchestrator)
        )
        wording = _wording_for(tags)
        self.setObjectName(f"{vi_name}_offline_front_panel")
        self.setWindowTitle(wording.window_title.format(vi_name=vi_name))
        self.setMinimumSize(420, 220)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self._header_lbl = QLabel(wording.header.format(vi_name=vi_name))
        self._header_lbl.setObjectName(f"{vi_name}_offline_detail_header")
        outer.addWidget(self._header_lbl)

        self._reason_lbl = QLabel("")
        self._reason_lbl.setObjectName(f"{vi_name}_offline_detail_reason")
        self._reason_lbl.setWordWrap(True)
        outer.addWidget(self._reason_lbl)
        self._refresh_reason()

        self._hint_lbl = QLabel(wording.hint)
        self._hint_lbl.setObjectName(f"{vi_name}_offline_detail_hint")
        self._hint_lbl.setProperty("class", "secondary_label")
        self._hint_lbl.setWordWrap(True)
        outer.addWidget(self._hint_lbl)

        action_row = QHBoxLayout()
        self._connect_btn = ConnectionButton(
            vi_name,
            "connect",
            self._on_connect_clicked,
            parent=self,
            # Keep this window's long-standing objectName (objectNames are
            # API) and keep it distinct from the card's own Connect button.
            object_name=f"{vi_name}_reconnect_btn",
        )
        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName(f"{vi_name}_reconnect_status")
        self._status_lbl.setProperty("class", "secondary_label")
        action_row.addWidget(self._connect_btn)
        action_row.addWidget(self._status_lbl)
        action_row.addStretch()
        outer.addLayout(action_row)
        outer.addStretch()

        orchestrator.instrument_reconnected.connect(self._on_reconnected)
        orchestrator.action_failed.connect(self._on_action_failed)

    def _refresh_reason(self) -> None:
        """Show the offline registry's current reason."""
        self._reason_lbl.setText(self._mirror.offline_reason(self._vi_name))

    def _on_connect_clicked(self) -> None:
        """Submit the connect request; the verdict arrives via signals."""
        self._status_lbl.setText("Connecting…")
        self._orchestrator.connect_instrument(self._vi_name)

    def _on_reconnected(self, vi_name: str) -> None:
        """Report success; MonitorWindow swaps the card and closes us."""
        if vi_name != self._vi_name:
            return
        self._status_lbl.setText("Connected — instrument is live.")
        self._connect_btn.setEnabled(False)

    def _on_action_failed(self, vi_name: str, method_name: str, reason: str) -> None:
        """Report a failed connect attempt inline, with the fresh reason and wording.

        A reconnect attempt on an already-offline VI can change its tags (see
        ``OfflineInstrumentPanel._on_action_failed()``'s docstring), so the
        title/header/hint are re-derived here too, not just the reason text.
        """
        if vi_name != self._vi_name or method_name != "connect":
            return
        self._status_lbl.setText("Still not reachable.")
        self._refresh_reason()
        self._refresh_wording()

    def _refresh_wording(self) -> None:
        """Re-select title/header/hint from the offline registry's current tags."""
        tags = self._mirror.availability_tags(self._vi_name)
        wording = _wording_for(tags)
        self.setWindowTitle(wording.window_title.format(vi_name=self._vi_name))
        self._header_lbl.setText(wording.header.format(vi_name=self._vi_name))
        self._hint_lbl.setText(wording.hint)
