"""Alerts — every warning and error the station raises, each in its own row until it is over.

**Why not one banner.** A single banner holds one message: a new warning
replaced the one before it, and a cleared fault calling ``dismiss()`` could
erase a save failure that arrived in between. So a fault could hide a stall,
and clearing one condition could silence another. Here every alert has a
**key** naming its cause, and nothing but that key touches it:

* **Condition alerts** (an instrument fault, a hold, an EMERGENCY, a save
  failure) are *resolved* by the code that watches the condition, when the
  condition is over — ``resolve(key)``, or ``sync_group(prefix, current)``
  for a family whose members come and go. The operator cannot dismiss them:
  an alert that disappeared while its cause was still there would be a lie.
* **Event alerts** (an action refused or failed, an engine error, a startup
  note) are things that *happened*. They stay until the operator dismisses
  them, or — for the routine ones, like a refusal — until they are
  ``EXPIRY_S`` old. A repeat of the same event does not add a row: it bumps
  the row's count and time.

Alerts are ordered most severe first, then newest first. The model is a
plain ``QObject`` with one ``changed`` signal, so the band that draws it,
the window that feeds it and the tests that check it are independent.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import qtawesome as qta
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from i2as.gui.theme import alert_text_color

#: Severities, most severe first. ``info`` is for facts worth a row but not
#: a problem (an acknowledged override counting down).
SEVERITY_EMERGENCY = "emergency"
SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"
SEVERITIES: tuple[str, ...] = (
    SEVERITY_EMERGENCY,
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    SEVERITY_INFO,
)

#: How long an expiring event alert (a refusal) stays after it last happened.
EXPIRY_S = 60.0

#: The most alerts kept. Past it the oldest dismissible ones are dropped:
#: condition alerts are never dropped, because their cause is still there.
MAX_ALERTS = 50

#: Rows the band shows before folding the rest behind "Show N more".
VISIBLE_ROWS = 4


@dataclass(frozen=True)
class Alert:
    """One thing the operator should know about.

    Attributes:
        key: The cause, e.g. ``"fault:magnet_z"`` — the only handle anything
            resolves or dismisses it by.
        severity: One of ``SEVERITIES``.
        source: What it is about (an instrument, ``station``, ``session``).
        message: What happened, in one sentence.
        first_seen: When it was first raised (epoch seconds).
        last_seen: When it was last raised.
        count: How many times it was raised.
        action_label: The one thing to do about it, or ``""``.
        action: Called when the action button is pressed.
        action_name: The action button's objectName (tests, muscle memory).
        dismissible: Whether the operator may remove it; ``False`` for a
            condition, which only its own watcher resolves.
        expires: Whether it disappears ``EXPIRY_S`` after ``last_seen``.
    """

    key: str
    severity: str
    source: str
    message: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    count: int = 1
    action_label: str = ""
    action: Callable[[], object] | None = field(default=None, compare=False)
    action_name: str = ""
    dismissible: bool = True
    expires: bool = False

    @property
    def rank(self) -> int:
        """Position of the severity in ``SEVERITIES`` (lower is more severe)."""
        return SEVERITIES.index(self.severity) if self.severity in SEVERITIES else len(SEVERITIES)


class AlertCenter(QObject):
    """The model: alerts by key, resolved by their cause or dismissed by the operator.

    Signals:
        changed (): The set of alerts, or one of them, changed.
    """

    changed = pyqtSignal()

    def __init__(self, clock: Callable[[], float] = time.time, parent: QObject | None = None) -> None:
        """Build an empty centre.

        Args:
            clock: Seconds since the epoch; injected by tests.
            parent: Qt parent.
        """
        super().__init__(parent)
        self._clock = clock
        self._alerts: dict[str, Alert] = {}
        self._expiry_timer = QTimer(self)
        self._expiry_timer.setInterval(1000)
        self._expiry_timer.timeout.connect(self.expire)
        self._expiry_timer.start()

    # ── Reading ───────────────────────────────────────────────────────────

    def alerts(self) -> list[Alert]:
        """Return every alert, most severe first, then newest first."""
        return sorted(self._alerts.values(), key=lambda a: (a.rank, -a.last_seen))

    def get(self, key: str) -> Alert | None:
        """Return the alert with *key*, or ``None``."""
        return self._alerts.get(key)

    def keys(self, prefix: str = "") -> list[str]:
        """Return the keys, optionally only those starting with *prefix*."""
        return [key for key in self._alerts if key.startswith(prefix)]

    def counts(self) -> dict[str, int]:
        """Return ``{severity: number of alerts}`` for every severity present."""
        result: dict[str, int] = {}
        for alert in self._alerts.values():
            result[alert.severity] = result.get(alert.severity, 0) + 1
        return result

    def __len__(self) -> int:
        return len(self._alerts)

    # ── Raising ───────────────────────────────────────────────────────────

    def raise_alert(
        self,
        key: str,
        severity: str,
        source: str,
        message: str,
        *,
        action_label: str = "",
        action: Callable[[], object] | None = None,
        action_name: str = "",
        dismissible: bool = True,
        expires: bool = False,
    ) -> Alert:
        """Raise the alert *key*, or refresh it if it is already there.

        A refresh keeps the first-seen time and, when the message is the
        same, counts the repeat; a new message replaces the old one (the
        condition's description changed) without counting.

        Args:
            key: The cause.
            severity: One of ``SEVERITIES``.
            source: What it is about.
            message: What happened.
            action_label: The one action offered, or ``""``.
            action: Called by the action button.
            action_name: The action button's objectName.
            dismissible: Whether the operator may remove it.
            expires: Whether it disappears ``EXPIRY_S`` after it last happened.

        Returns:
            The alert as it now stands.

        Raises:
            ValueError: If *severity* is unknown.
        """
        alert = self._upsert(
            key,
            severity,
            source,
            message,
            action_label=action_label,
            action=action,
            action_name=action_name,
            dismissible=dismissible,
            expires=expires,
        )
        self._trim()
        self.changed.emit()
        return alert

    def _upsert(
        self,
        key: str,
        severity: str,
        source: str,
        message: str,
        *,
        action_label: str = "",
        action: Callable[[], object] | None = None,
        action_name: str = "",
        dismissible: bool = True,
        expires: bool = False,
    ) -> Alert:
        """Insert or refresh one alert without emitting; see ``raise_alert``."""
        if severity not in SEVERITIES:
            raise ValueError(f"unknown alert severity {severity!r}")
        now = self._clock()
        existing = self._alerts.get(key)
        if existing is None:
            alert = Alert(
                key=key,
                severity=severity,
                source=source,
                message=message,
                first_seen=now,
                last_seen=now,
                action_label=action_label,
                action=action,
                action_name=action_name,
                dismissible=dismissible,
                expires=expires,
            )
        else:
            alert = replace(
                existing,
                severity=severity,
                source=source,
                message=message,
                last_seen=now,
                count=existing.count + (1 if message == existing.message else 0),
                action_label=action_label,
                action=action,
                action_name=action_name,
                dismissible=dismissible,
                expires=expires,
            )
        self._alerts[key] = alert
        return alert

    def update_message(self, key: str, message: str) -> None:
        """Change an alert's text without counting it as a repeat (a countdown).

        Args:
            key: The alert.
            message: Its new text.
        """
        alert = self._alerts.get(key)
        if alert is None or alert.message == message:
            return
        self._alerts[key] = replace(alert, message=message)
        self.changed.emit()

    # ── Ending ────────────────────────────────────────────────────────────

    def resolve(self, key: str) -> bool:
        """Remove the alert *key* because its cause is over.

        Args:
            key: The cause.

        Returns:
            ``True`` when there was such an alert.
        """
        if self._alerts.pop(key, None) is None:
            return False
        self.changed.emit()
        return True

    def sync_group(self, prefix: str, current: Mapping[str, Mapping[str, object]]) -> None:
        """Make the alerts under *prefix* exactly *current*.

        For a family of condition alerts whose members come and go (one per
        faulted instrument, one per hold condition): every key in *current*
        is raised with its keyword arguments, and every key under *prefix*
        not in it is resolved — so a condition that ended cannot linger and
        one that is still there cannot be lost.

        Args:
            prefix: The family, e.g. ``"fault:"``.
            current: ``{key: raise_alert keyword arguments}``; every key must
                start with *prefix*.

        Raises:
            ValueError: If a key does not start with *prefix*.
        """
        for key in current:
            if not key.startswith(prefix):
                raise ValueError(f"{key!r} is not in the {prefix!r} group")
        stale = [key for key in self._alerts if key.startswith(prefix) and key not in current]
        touched = bool(stale)
        for key in stale:
            del self._alerts[key]
        for key, spec in current.items():
            existing = self._alerts.get(key)
            # A condition that is still there, unchanged, is not raised again:
            # its row keeps its time, and nothing is redrawn.
            if existing is not None and all(
                getattr(existing, name) == value
                for name, value in spec.items()
                if name != "action"
            ):
                continue
            self._upsert(key, **spec)  # type: ignore[arg-type]
            touched = True
        if touched:
            self._trim()
            self.changed.emit()

    def dismiss(self, key: str) -> bool:
        """Remove the alert *key* at the operator's request, if it may be.

        Args:
            key: The alert.

        Returns:
            ``True`` when it was removed; ``False`` for a condition alert or
            an unknown key.
        """
        alert = self._alerts.get(key)
        if alert is None or not alert.dismissible:
            return False
        return self.resolve(key)

    def dismiss_all(self) -> int:
        """Remove every dismissible alert.

        Returns:
            How many were removed.
        """
        removable = [key for key, alert in self._alerts.items() if alert.dismissible]
        for key in removable:
            del self._alerts[key]
        if removable:
            self.changed.emit()
        return len(removable)

    def expire(self) -> None:
        """Drop every expiring alert last seen more than ``EXPIRY_S`` ago."""
        now = self._clock()
        old = [
            key
            for key, alert in self._alerts.items()
            if alert.expires and now - alert.last_seen > EXPIRY_S
        ]
        for key in old:
            del self._alerts[key]
        if old:
            self.changed.emit()

    def _trim(self) -> None:
        """Keep at most ``MAX_ALERTS``, dropping the oldest dismissible ones."""
        excess = len(self._alerts) - MAX_ALERTS
        if excess <= 0:
            return
        removable = sorted(
            (alert for alert in self._alerts.values() if alert.dismissible),
            key=lambda a: a.last_seen,
        )
        for alert in removable[:excess]:
            del self._alerts[alert.key]


#: Icon per severity, drawn in the row's text colour.
_ICONS: dict[str, str] = {
    SEVERITY_EMERGENCY: "fa5s.radiation",
    SEVERITY_ERROR: "fa5s.times-circle",
    SEVERITY_WARNING: "fa5s.exclamation-triangle",
    SEVERITY_INFO: "fa5s.info-circle",
}


class AlertRow(QWidget):
    """One alert: icon, source, message, time and count, its action, and ✕ when dismissible."""

    def __init__(self, alert: Alert, center: AlertCenter, parent: QWidget | None = None) -> None:
        """Draw one alert.

        Args:
            alert: The alert.
            center: The model, for dismissing.
            parent: Qt parent.
        """
        super().__init__(parent)
        self.alert = alert
        self.setObjectName("alert_row")
        self.setProperty("severity", alert.severity)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 3, 6, 3)
        row.setSpacing(8)

        color = alert_text_color(alert.severity)
        icon = QLabel()
        icon.setPixmap(qta.icon(_ICONS.get(alert.severity, _ICONS[SEVERITY_INFO]), color=color).pixmap(14, 14))
        row.addWidget(icon)
        source = QLabel(f"<b>{alert.source}</b>")
        source.setObjectName("alert_source")
        row.addWidget(source)
        self.message_label = QLabel(alert.message)
        self.message_label.setObjectName("alert_message")
        self.message_label.setWordWrap(False)
        self.message_label.setToolTip(alert.message)
        self.message_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self.message_label, 1)
        when = time.strftime("%H:%M:%S", time.localtime(alert.last_seen))
        stamp = QLabel(f"{alert.count}× · {when}" if alert.count > 1 else when)
        stamp.setObjectName("alert_time")
        row.addWidget(stamp)
        self.action_button: QPushButton | None = None
        if alert.action_label and alert.action is not None:
            self.action_button = QPushButton(alert.action_label)
            self.action_button.setObjectName(alert.action_name or "alert_action_btn")
            self.action_button.setProperty("class", "alert_action")
            self.action_button.clicked.connect(lambda _checked=False, act=alert.action: act())
            row.addWidget(self.action_button)
        self.dismiss_button: QPushButton | None = None
        if alert.dismissible:
            self.dismiss_button = QPushButton("✕")
            self.dismiss_button.setObjectName("alert_dismiss_btn")
            self.dismiss_button.setProperty("class", "alert_dismiss")
            self.dismiss_button.setToolTip("Dismiss")
            self.dismiss_button.clicked.connect(lambda: center.dismiss(alert.key))
            row.addWidget(self.dismiss_button)


class AlertBand(QWidget):
    """The Monitor window's third band: one row per alert, hidden when there are none.

    At most ``VISIBLE_ROWS`` rows are shown — the most severe first — and the
    rest fold behind "Show N more"; "Dismiss all" removes every dismissible
    alert. Rebuilt from the model on every ``changed``, so what is drawn is
    always exactly what the model holds.
    """

    def __init__(self, center: AlertCenter, parent: QWidget | None = None) -> None:
        """Build the band over one model.

        Args:
            center: The alerts to draw.
            parent: Qt parent.
        """
        super().__init__(parent)
        self.setObjectName("alert_band")
        self._center = center
        self._expanded = False
        self._rows: list[AlertRow] = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(1)
        self._footer = QWidget()
        footer = QHBoxLayout(self._footer)
        footer.setContentsMargins(8, 0, 6, 0)
        self._more_button = QPushButton("")
        self._more_button.setObjectName("alert_more_btn")
        self._more_button.setProperty("class", "alert_link")
        self._more_button.clicked.connect(self._toggle_expanded)
        footer.addWidget(self._more_button)
        footer.addStretch()
        self._dismiss_all_button = QPushButton("Dismiss all")
        self._dismiss_all_button.setObjectName("alert_dismiss_all_btn")
        self._dismiss_all_button.setProperty("class", "alert_link")
        self._dismiss_all_button.clicked.connect(center.dismiss_all)
        footer.addWidget(self._dismiss_all_button)
        center.changed.connect(self.refresh)
        self.refresh()

    @property
    def rows(self) -> list[AlertRow]:
        """The rows drawn now, top to bottom."""
        return list(self._rows)

    def row_for(self, key: str) -> AlertRow | None:
        """Return the drawn row of alert *key*, or ``None`` when it is not drawn."""
        return next((row for row in self._rows if row.alert.key == key), None)

    def _toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        self.refresh()

    def refresh(self) -> None:
        """Redraw from the model."""
        for row in self._rows:
            self._layout.removeWidget(row)
            row.hide()
            row.deleteLater()
        self._rows = []
        self._layout.removeWidget(self._footer)
        alerts = self._center.alerts()
        shown = alerts if self._expanded else alerts[:VISIBLE_ROWS]
        for alert in shown:
            row = AlertRow(alert, self._center, self)
            self._layout.addWidget(row)
            # Shown now, not on the layout's queued show: a caller that
            # raised an alert may look at its row before the event loop runs.
            row.show()
            self._rows.append(row)
        hidden = len(alerts) - len(shown)
        dismissible = sum(1 for alert in alerts if alert.dismissible)
        show_footer = hidden > 0 or self._expanded and len(alerts) > VISIBLE_ROWS or dismissible > 1
        if show_footer:
            self._more_button.setText(
                f"Show {hidden} more" if hidden > 0 else "Show fewer"
            )
            self._more_button.setVisible(hidden > 0 or len(alerts) > VISIBLE_ROWS)
            self._dismiss_all_button.setVisible(dismissible > 1)
            self._layout.addWidget(self._footer)
            self._footer.show()
        else:
            self._footer.hide()
        self.setVisible(bool(alerts))


__all__ = [
    "EXPIRY_S",
    "MAX_ALERTS",
    "SEVERITIES",
    "SEVERITY_EMERGENCY",
    "SEVERITY_ERROR",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "VISIBLE_ROWS",
    "Alert",
    "AlertBand",
    "AlertCenter",
    "AlertRow",
]
