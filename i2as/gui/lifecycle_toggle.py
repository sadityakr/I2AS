"""Per-instrument lifecycle controls: Initiate/Standby and Connect/Disconnect."""

from __future__ import annotations

from collections.abc import Callable

import qtawesome as qta
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from i2as.gui.theme import BTN_CLASS_PRIMARY, BTN_CLASS_SECONDARY, TEXT_ON_ACCENT, TEXT_PRIMARY


class LifecycleToggleButton(QWidget):
    """A single Initiate/Standby toggle with a status glow dot.

    Args:
        vi_name: The VI's registered name, used to scope objectNames.
        on_toggle: Called with ``"initiate"`` or ``"standby"`` (the opposite
            of the current displayed state) when the button is clicked.
        parent: Optional Qt parent widget.
    """

    def __init__(
        self,
        vi_name: str,
        on_toggle: Callable[[str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._vi_name = vi_name
        self._on_toggle = on_toggle
        self._initiated = False
        # Hold the natural width: in the two-column instrument grid the header
        # is contested (name + front-panel icon + Disconnect + this), and Qt
        # would otherwise shrink the button below its size hint and clip
        # "Initiate" (gui-edit: never cap an icon button below its size hint).
        # The VI-name label beside it is the one that gives way instead.
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self._dot = QLabel("●")
        self._dot.setObjectName(f"{vi_name}_lifecycle_dot")
        self._dot.setProperty("class", "lifecycle_dot")
        self._dot.setProperty("status", "standby")
        row.addWidget(self._dot)

        self._btn = QPushButton()
        self._btn.setObjectName(f"{vi_name}_lifecycle_btn")
        row.addWidget(self._btn)

        self._btn.clicked.connect(self._on_click)
        self._render()

    def _on_click(self) -> None:
        self._on_toggle("standby" if self._initiated else "initiate")

    def is_initiated(self) -> bool:
        """Return the currently displayed state (True = initiated)."""
        return self._initiated

    def set_initiated(self, initiated: bool) -> None:
        """Update the displayed state. No-op if already showing that state.

        Args:
            initiated: True to show the "Standby" / green-dot state, False
                for "Initiate" / red-dot.
        """
        if initiated == self._initiated:
            return
        self._initiated = initiated
        self._render()

    def _render(self) -> None:
        if self._initiated:
            self._btn.setText("Standby")
            self._btn.setProperty("class", BTN_CLASS_SECONDARY)
            self._btn.setIcon(qta.icon("fa5s.power-off", color=TEXT_PRIMARY))
            self._btn.setToolTip(f"Return {self._vi_name} to a safe standby state")
            self._dot.setProperty("status", "initiated")
        else:
            self._btn.setText("Initiate")
            self._btn.setProperty("class", BTN_CLASS_PRIMARY)
            self._btn.setIcon(qta.icon("fa5s.play", color=TEXT_ON_ACCENT))
            self._btn.setToolTip(f"Bring {self._vi_name} to its operating state")
            self._dot.setProperty("status", "standby")

        # Qt only re-evaluates property-based QSS selectors after an
        # unpolish/polish cycle (same pattern InstrumentPanel's status border uses).
        for widget in (self._btn, self._dot):
            widget.style().unpolish(widget)
            widget.style().polish(widget)


class ConnectionButton(QPushButton):
    """The Connect or Disconnect button every instrument card carries.

    The CONNECTION axis of the connection-lifecycle standard (see
    ``BaseVirtualInstrument``), sitting next to the Initiate/Standby toggle so
    the two axes read as the pair they are: Initiate/Standby changes what the
    instrument is *doing*, Connect/Disconnect changes who *owns* it.

    Deliberately one-way rather than a toggle. A live card can only ever offer
    Disconnect and an offline card only Connect, because the card itself is
    replaced when the connection state flips — so a toggle would have to track
    a state it can never actually be in.

    Args:
        vi_name: The VI's registered name, used to scope the objectName and
            to word the tooltip.
        direction: ``"disconnect"`` (shown on a live card) or ``"connect"``
            (shown on an offline card).
        on_click: Called with no arguments when the button is pressed. It only
            SUBMITS the request; the displayed state never changes optimistically.
        parent: Optional Qt parent widget.
        object_name: Override for the button's objectName (objectNames are
            API — see the gui-edit skill). Defaults to
            ``"{vi_name}_{direction}_btn"``; the offline detail window passes
            its long-standing ``"{vi_name}_reconnect_btn"`` so its own name
            survives, and so the card's button and the window's button stay
            distinguishable by ``findChild``.
        compact: Render icon-only, for the monitor card's header row. The
            instrument grid is two narrow columns, and a labelled button there
            squeezes the VI name and the Initiate/Standby toggle down to
            ellipses — so on the card this matches the front-panel icon button
            beside it and carries its meaning in the tooltip, while every
            roomier surface (the offline card's body, the detail window) keeps
            the word.

    Raises:
        ValueError: If *direction* is not "connect" or "disconnect".
    """

    def __init__(
        self,
        vi_name: str,
        direction: str,
        on_click: Callable[[], None],
        parent: QWidget | None = None,
        object_name: str | None = None,
        compact: bool = False,
    ) -> None:
        super().__init__(parent)
        if direction not in ("connect", "disconnect"):
            raise ValueError(
                f"ConnectionButton direction must be 'connect' or "
                f"'disconnect', got {direction!r}"
            )
        self._vi_name = vi_name
        self._direction = direction
        self.setObjectName(object_name or f"{vi_name}_{direction}_btn")
        # Same reasoning as LifecycleToggleButton: hold the natural width and
        # let the VI-name label absorb a narrow card instead.
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        if direction == "disconnect":
            if not compact:
                self.setText("Disconnect")
            self.setProperty("class", BTN_CLASS_SECONDARY)
            self.setIcon(qta.icon("fa5s.unlink", color=TEXT_PRIMARY))
            self.setToolTip(
                f"Disconnect {vi_name} — release it so it can be used from "
                f"its own front panel or vendor software. Nothing on the "
                f"instrument changes: press Standby first if you want it "
                f"idle. I2AS stops polling it until you press Connect."
            )
        else:
            if not compact:
                self.setText("Connect")
            self.setProperty("class", BTN_CLASS_PRIMARY)
            self.setIcon(qta.icon("fa5s.link", color=TEXT_ON_ACCENT))
            self.setToolTip(
                f"Connect {vi_name} — hand it back to I2AS: reopen its "
                f"connection and check its identity. Nothing on the "
                f"instrument changes; press Initiate afterwards to bring it "
                f"to its operating state."
            )

        self.clicked.connect(lambda _checked=False: on_click())

    def direction(self) -> str:
        """Return the direction this button submits ("connect"/"disconnect")."""
        return self._direction
