"""RampTrackerPanel — live list of running ramps, each with an Abort button."""

from __future__ import annotations

import logging
import re

import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from i2as.core.orchestrator_proxy import OrchestratorProxy
from i2as.core.ramps import RampRecord
from i2as.gui.theme import BTN_CLASS_DANGER, TEXT_PRIMARY

logger = logging.getLogger(__name__)

__all__ = ["RampRow", "RampTrackerPanel"]


def _slug(vi_name: str) -> str:
    """Return a lowercase, underscore-joined objectName fragment for *vi_name*.

    Args:
        vi_name: A registered VI name (e.g. ``"temperature"``).

    Returns:
        e.g. ``"temperature"`` — used to scope every widget objectName a row
        creates so two rows never collide.
    """
    return re.sub(r"[^a-z0-9]+", "_", vi_name.lower()).strip("_")


def _row_title(record: RampRecord) -> str:
    """Return a row's group-box title: its setpoint label and its VI name.

    Args:
        record: The ramp the row shows.

    Returns:
        e.g. ``"setpoint · temperature"``, or just the VI name when the VI
        declares no distinct setpoint label.
    """
    if record.label and record.label != record.vi_name:
        return f"{record.label} · {record.vi_name}"
    return record.vi_name


def _format_value(value: float | None, unit: str) -> str:
    """Render one ramp quantity for display, with its unit.

    Display formatting is a GUI concern (the SI-units-in-APIs rule): the
    record carries plain SI floats and this is the only place they become
    text. ``%.4g`` keeps a 0.0005 T step and a 290 K setpoint both readable
    in the same narrow column.

    Args:
        value: The quantity in the record's user units, or ``None`` when the
            VI does not expose it.
        unit: The unit to append (may be empty).

    Returns:
        e.g. ``"1.234 T"``, or ``"—"`` when *value* is ``None``.
    """
    if value is None:
        return "—"
    return f"{value:.4g} {unit}".strip()


class RampRow(QGroupBox):
    """One running ramp: value → next setpoint, end setpoint, rate, and Abort.

    Built generically from a ``RampRecord`` — this class contains no
    per-instrument logic. It is created once per ramping VI and updated in
    place on every subsequent tick.

    Args:
        record: The ramp this row shows. Its ``vi_name`` fixes the row's
            identity (and every objectName) for the row's whole lifetime;
            every other field is re-read on each ``update_record()``.
        orchestrator: The active Orchestrator — ``stop_ramp()`` is the only
            method this row calls.
        parent: Optional Qt parent widget.
    """

    def __init__(
        self,
        record: RampRecord,
        orchestrator: OrchestratorProxy,
        parent: QWidget | None = None,
    ) -> None:
        # Title carries BOTH the setpoint label and the VI name: a two-magnet
        # setup shows two rows whose label is "field", and only the VI name
        # tells the operator which coil is moving.
        super().__init__(_row_title(record), parent)
        self._vi_name = record.vi_name
        self._orchestrator = orchestrator
        self._slug = _slug(record.vi_name)
        self.setObjectName(f"ramp_row_{self._slug}")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        text_column = QVBoxLayout()
        text_column.setContentsMargins(0, 0, 0, 0)
        text_column.setSpacing(2)

        #: "<current value> → <next setpoint>": where the instrument is and
        #: where it is being driven to right now.
        self._progress_label = QLabel("")
        self._progress_label.setObjectName(f"ramp_{self._slug}_progress_label")
        self._progress_label.setProperty("class", "value_readout")
        # Wrapped, like every other label here: this sub-panel is a quarter of
        # a quarter, so a row must stay readable at ~200 px rather than push
        # its own Abort button off the right edge (the off-screen-to-the-right
        # bug class gui/README.md calls out).
        self._progress_label.setWordWrap(True)
        text_column.addWidget(self._progress_label)

        #: End setpoint + rate (+ sub-phase, for a VI that has phases).
        self._detail_label = QLabel("")
        self._detail_label.setObjectName(f"ramp_{self._slug}_detail_label")
        self._detail_label.setProperty("class", "secondary_label")
        self._detail_label.setWordWrap(True)
        text_column.addWidget(self._detail_label)

        #: Which run owns this ramp; hidden entirely for a manual ramp.
        self._owner_label = QLabel("")
        self._owner_label.setObjectName(f"ramp_{self._slug}_owner_label")
        self._owner_label.setProperty("class", "secondary_label")
        self._owner_label.setWordWrap(True)
        self._owner_label.hide()
        text_column.addWidget(self._owner_label)

        outer.addLayout(text_column, stretch=1)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setObjectName(f"ramp_{self._slug}_abort_btn")
        self._abort_btn.setProperty("class", BTN_CLASS_DANGER)
        self._abort_btn.setIcon(qta.icon("fa5s.ban", color=TEXT_PRIMARY))
        self._abort_btn.clicked.connect(self._on_abort_clicked)
        # Top-aligned so the button stays next to the first line of a text
        # column that may wrap to three.
        outer.addWidget(self._abort_btn, alignment=Qt.AlignmentFlag.AlignTop)

        self.update_record(record)

    @property
    def vi_name(self) -> str:
        """Return the VI name this row is bound to (fixed for its lifetime)."""
        return self._vi_name

    def update_record(self, record: RampRecord) -> None:
        """Re-render this row from a fresh record for the same VI.

        Args:
            record: The current ``RampRecord`` for this row's VI.
        """
        unit = record.unit
        value_text = _format_value(record.value, unit)
        setpoint_text = _format_value(record.setpoint, unit)
        self._progress_label.setText(f"{value_text} → {setpoint_text}")
        self._progress_label.setToolTip(
            f"{record.vi_name}: now at {value_text}, driving to the next "
            f"setpoint {setpoint_text}"
        )

        details = [f"End setpoint {_format_value(record.target, unit)}"]
        if record.rate is not None:
            rate_unit = f"{unit}/min" if unit else "/min"
            details.append(f"{record.rate:.4g} {rate_unit}")
        else:
            details.append("rate unknown")
        if record.phase:
            details.append(record.phase)
        if record.stale:
            # Not a colour change: a stale ramp read is a "these numbers are
            # not current" caveat, and the panel deliberately introduces no
            # new status colours (the instrument cards already own that
            # vocabulary for the same VI).
            details.append("reading stale")
        self._detail_label.setText(" · ".join(details))

        if record.owner:
            self._owner_label.setText(f"Owned by {record.owner}")
            self._owner_label.show()
        else:
            self._owner_label.clear()
            self._owner_label.hide()

        self._abort_btn.setEnabled(record.stoppable)
        self._abort_btn.setToolTip(
            f"Stop the {record.label or record.vi_name} ramp on "
            f"{record.vi_name} — hold the instrument where it is now"
            if record.stoppable
            else record.stop_blocked_reason
        )

    def _on_abort_clicked(self) -> None:
        """Confirm, then ask the Orchestrator to stop this one ramp."""
        answer = QMessageBox.question(
            self,
            f"Abort ramp on {self._vi_name}",
            f"Stop the ramp on {self._vi_name}? The instrument is held "
            f"where it is now — nothing else is stopped.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._orchestrator.stop_ramp(self._vi_name)


class RampTrackerPanel(QWidget):
    """The live list of running ramps — one ``RampRow`` per ramping VI.

    Args:
        orchestrator: The active Orchestrator, forwarded to every row.
            ``ramps_updated`` is deliberately NOT connected here: it fires
            every tick, so ``MonitorWindow`` receives it and forwards to
            ``on_ramps_updated()`` (the destruction-order rule — see
            ``gui/README.md``).
        parent: Optional Qt parent widget.
    """

    def __init__(self, orchestrator: OrchestratorProxy, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ramp_tracker_panel")
        self._orchestrator = orchestrator
        self._rows: dict[str, RampRow] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        self._empty_label = QLabel("No ramps running.")
        self._empty_label.setObjectName("ramp_tracker_empty_label")
        self._empty_label.setProperty("class", "secondary_label")
        outer.addWidget(self._empty_label)

        #: Rows are inserted before the trailing stretch so they stack from
        #: the top instead of spreading over the quadrant's full height.
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        outer.addLayout(self._rows_layout)
        outer.addStretch()

    def on_ramps_updated(self, records: list[RampRecord]) -> None:
        """Reconcile the visible rows against the ramps running right now.

        Forwarded from ``MonitorWindow`` on every ``ramps_updated`` tick.
        Rows are added, updated in place, and removed rather than rebuilt:
        a rebuild every 3 s would make the Abort button unclickable and
        every objectName transient.

        Args:
            records: The ramps running as of this tick, ordered by VI name
                (``Orchestrator.active_ramps()``'s ordering).
        """
        seen: set[str] = set()
        for position, record in enumerate(records):
            seen.add(record.vi_name)
            row = self._rows.get(record.vi_name)
            if row is None:
                row = RampRow(record, self._orchestrator, parent=self)
                self._rows[record.vi_name] = row
                self._rows_layout.insertWidget(position, row)
            else:
                row.update_record(record)

        for vi_name in [name for name in self._rows if name not in seen]:
            row = self._rows.pop(vi_name)
            self._rows_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()

        self._empty_label.setVisible(not self._rows)

    def row_names(self) -> list[str]:
        """Return the VI names currently shown, in layout order.

        The panel's read surface for tests and for any caller that needs to
        know what is on screen without walking the widget tree.
        """
        return [
            row.vi_name
            for row in (
                self._rows_layout.itemAt(i).widget() for i in range(self._rows_layout.count())
            )
            if isinstance(row, RampRow)
        ]
