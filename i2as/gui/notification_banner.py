"""NotificationBanner — inline non-modal warning/error strip."""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from i2as.gui.theme import BANNER_SEVERITY_ERROR, BANNER_SEVERITY_WARNING

logger = logging.getLogger(__name__)

_VALID_SEVERITIES = frozenset({BANNER_SEVERITY_WARNING, BANNER_SEVERITY_ERROR})


class NotificationBanner(QWidget):
    """A hidden-by-default inline strip for non-modal warning/error messages.

    Styling is driven entirely from ``theme.py`` via a dynamic ``severity``
    QSS property (``"warning"`` / ``"error"``). The widget sets
    ``WA_StyledBackground`` so the stylesheet can paint its background — a bare
    QWidget subclass otherwise ignores QSS ``background-color``.

    Args:
        parent: Optional Qt parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("notification_banner")
        # A plain QWidget only paints its QSS background when this attribute is set.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Current base message (without the "(N×) " prefix) and its repeat count.
        self._base_message = ""
        self._count = 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 6, 6)
        layout.setSpacing(8)

        self._label = QLabel("")
        self._label.setObjectName("banner_label")
        self._label.setWordWrap(True)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(self._label, stretch=1)

        self._dismiss_btn = QPushButton("✕")  # ✕
        self._dismiss_btn.setObjectName("banner_dismiss_btn")
        self._dismiss_btn.setToolTip("Dismiss this notification")
        self._dismiss_btn.setMaximumWidth(32)
        self._dismiss_btn.clicked.connect(self.dismiss)
        layout.addWidget(self._dismiss_btn, alignment=Qt.AlignmentFlag.AlignTop)

        self.hide()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_message(self, message: str, severity: str) -> None:
        """Display a notification, or bump the counter if it repeats.

        If the banner is already visible showing the same message, the repeat
        counter increments and the text gains a ``"(N×) "`` prefix instead of a
        second banner stacking up. A different message (or a hidden banner)
        resets the counter and re-styles for the given severity.

        Args:
            message: The human-readable notification text.
            severity: ``"warning"`` or ``"error"``. Unknown values fall back to
                ``"warning"`` (logged).
        """
        if severity not in _VALID_SEVERITIES:
            logger.warning("NotificationBanner: unknown severity %r; using 'warning'", severity)
            severity = BANNER_SEVERITY_WARNING

        if self.isVisible() and message == self._base_message:
            self._count += 1
        else:
            self._base_message = message
            self._count = 1
            self._apply_severity(severity)

        self._render()
        self.show()

    def dismiss(self) -> None:
        """Hide the banner and reset its message/counter state."""
        self._base_message = ""
        self._count = 0
        self.hide()

    @property
    def count(self) -> int:
        """Return the current repeat count of the active message (0 if hidden)."""
        return self._count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _render(self) -> None:
        """Update the label text, applying the ``(N×) `` prefix when repeated."""
        prefix = f"({self._count}×) " if self._count > 1 else ""
        self._label.setText(f"{prefix}{self._base_message}")

    def _apply_severity(self, severity: str) -> None:
        """Set the ``severity`` QSS property and repolish so styling updates.

        Qt only re-evaluates property-based QSS selectors after an
        unpolish/polish cycle (same dynamic-property repolish pattern the
        InstrumentPanel status border uses). Repolishing the parent alone is
        NOT enough: descendant selectors like ``[severity="error"] QLabel``
        are resolved per-widget, so each affected child must be repolished
        too or it keeps its old colour.

        Args:
            severity: ``"warning"`` or ``"error"``.
        """
        self.setProperty("severity", severity)
        for widget in (self, self._label, self._dismiss_btn):
            widget.style().unpolish(widget)
            widget.style().polish(widget)
