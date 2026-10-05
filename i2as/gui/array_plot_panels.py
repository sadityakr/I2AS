"""Monitor-window plot panels for the array @monitored kinds: image and waterfall.

The monitored-kind standard (``core/decorators.py``) lets a VI declare a
``@monitored`` field as ``kind="image"`` (a 2-D frame) or ``kind="trace"``
(a 1-D array). Those values never enter the scalar state or the trend
history; they arrive on ``monitored_arrays_updated`` and are kept, RAM only,
in an :class:`~i2as.gui.monitor_history.ArrayHistory`. The panels here draw
them:

* :class:`ImagePlotPanel` — the newest frame of one ``image`` field.
* :class:`WaterfallPlotPanel` — the recent ``trace`` values of one field
  stacked over time (newest at the top), so a drifting spectrum or line
  profile reads at a glance.

Both follow the same **plot-panel protocol** as ``TrendPlotPanel``, which is
all ``TrendsQuadrant`` relies on, so the quadrant hosts every kind the same
way:

* ``kind`` — the panel kind's registry key (``"trend"``, ``"image"``,
  ``"waterfall"``);
* ``remove_requested(str)`` — emitted with the ``panel_id``;
* ``refresh()`` — redraw from the shared history;
* ``selected_key()`` / ``set_selected_key(key)``;
* ``settings_entry()`` / ``apply_settings_entry(entry)`` — the JSON-safe
  dict the quadrant persists for this panel.

Which keys a panel offers comes from the station's declaration
(``MonitoredInfo.kind``), never from guessing at an array's shape.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import qtawesome as qta
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from i2as.core.events import MonitoredInfo
from i2as.gui.image_view import ImageView
from i2as.gui.monitor_history import ArrayHistory
from i2as.gui.theme import TEXT_PRIMARY

logger = logging.getLogger(__name__)

# (label, seconds) — how far back a waterfall reaches. Bounded by
# ArrayHistory's per-key entry cap, not by disk: array fields are RAM only.
WATERFALL_WINDOWS: list[tuple[str, float]] = [
    ("1 min", 60.0),
    ("5 min", 300.0),
    ("15 min", 900.0),
    ("1 h", 3600.0),
]
_DEFAULT_WATERFALL_WINDOW = "5 min"


class _ArrayPlotPanel(QGroupBox):
    """The shared frame of an array panel: a key selector, a remove button, a status line.

    Subclasses set ``kind`` and ``data_kind`` (the ``MonitoredInfo.kind``
    they draw), add any extra header widgets in ``_build_extra_header()``
    and implement ``_redraw()``.

    Args:
        history: The shared ``ArrayHistory``. Not owned by this panel.
        panel_id: Host-assigned identifier (e.g. ``"plot_2"``), used for
            objectNames and echoed on ``remove_requested``.
        fields: ``{flat_key: MonitoredInfo}`` of every declared array field
            on the station; the panel offers those of its ``data_kind``.
        parent: Optional Qt parent widget.
    """

    kind: ClassVar[str] = ""
    data_kind: ClassVar[str] = ""
    title_prefix: ClassVar[str] = ""
    #: Row 0 drawn at the top with square pixels (a camera frame), or rows
    #: bottom-up with each axis stretched to the panel (a waterfall).
    frame_like: ClassVar[bool] = True

    remove_requested = pyqtSignal(str)

    def __init__(
        self,
        history: ArrayHistory,
        panel_id: str,
        fields: Mapping[str, MonitoredInfo],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(f"{self.title_prefix} — {panel_id}", parent)
        self._history = history
        self._panel_id = panel_id
        self._fields = {
            key: info for key, info in fields.items() if info.kind == self.data_kind
        }
        self.setMinimumSize(260, 180)

        vlay = QVBoxLayout(self)
        top_row = QHBoxLayout()

        self._key_selector = QComboBox()
        self._key_selector.setObjectName(f"{self.kind}_key_selector_{panel_id}")
        self._key_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._key_selector.addItems(sorted(self._fields))
        self._key_selector.currentTextChanged.connect(self._on_key_changed)
        top_row.addWidget(self._key_selector, 1)

        self._build_extra_header(top_row)

        remove_button = QPushButton()
        remove_button.setObjectName(f"{self.kind}_remove_button_{panel_id}")
        remove_button.setIcon(qta.icon("fa5s.trash", color=TEXT_PRIMARY))
        remove_button.setToolTip("Remove this plot")
        remove_button.clicked.connect(lambda: self.remove_requested.emit(self._panel_id))
        top_row.addWidget(remove_button)
        vlay.addLayout(top_row)

        self._view = ImageView(
            object_name=f"{self.kind}_view_{panel_id}",
            invert_y=self.frame_like,
            lock_aspect=self.frame_like,
        )
        vlay.addWidget(self._view, 1)

        self._status = QLabel()
        self._status.setObjectName(f"{self.kind}_status_{panel_id}")
        vlay.addWidget(self._status)

        self._on_key_changed(self._key_selector.currentText())

    # ------------------------------------------------------------------
    # Plot-panel protocol
    # ------------------------------------------------------------------

    @property
    def panel_id(self) -> str:
        """The host-assigned identifier of this panel."""
        return self._panel_id

    def refresh(self) -> None:
        """Redraw from the shared history."""
        self._redraw()

    def selected_key(self) -> str | None:
        """Return the selected field's flat key, or ``None`` when none is declared."""
        text = self._key_selector.currentText()
        return text or None

    def set_selected_key(self, key: str) -> None:
        """Select a field by flat key; a no-op for a key this panel does not offer.

        Args:
            key: The flat key to select.
        """
        if self._key_selector.findText(key) >= 0:
            self._key_selector.setCurrentText(key)

    def settings_entry(self) -> dict[str, Any]:
        """Return the JSON-safe dict the host persists for this panel."""
        return {"kind": self.kind, "key": self.selected_key()}

    def apply_settings_entry(self, entry: Mapping[str, Any]) -> None:
        """Restore this panel from a persisted entry, ignoring what does not fit.

        Args:
            entry: A dict as produced by ``settings_entry()``.
        """
        key = entry.get("key")
        if isinstance(key, str) and key:
            self.set_selected_key(key)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _build_extra_header(self, row: QHBoxLayout) -> None:
        """Add kind-specific widgets to the header row (default: none)."""

    def _redraw(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_key_changed(self, key: str) -> None:
        info = self._fields.get(key)
        self._key_selector.setToolTip(info.description if info else "")
        if not self._fields:
            self._status.setText(f"No {self.data_kind} field is declared on this station.")
        self._view.clear()
        self._redraw()


class ImagePlotPanel(_ArrayPlotPanel):
    """The newest frame of one ``@monitored(kind="image")`` field."""

    kind: ClassVar[str] = "image"
    data_kind: ClassVar[str] = "image"
    title_prefix: ClassVar[str] = "Image"

    def _redraw(self) -> None:
        key = self.selected_key()
        if key is None:
            return
        info = self._fields[key]
        self._view.set_labels("column (px)", "row (px)", info.unit)
        latest = self._history.latest(key)
        if latest is None:
            self._status.setText("Waiting for the first frame…")
            return
        timestamp, frame = latest
        self._view.set_frame(np.asarray(frame))
        self._status.setText(f"Frame of {time.strftime('%H:%M:%S', time.localtime(timestamp))}")


class WaterfallPlotPanel(_ArrayPlotPanel):
    """The recent values of one ``@monitored(kind="trace")`` field, stacked over time.

    Rows are drawn evenly spaced between the oldest and newest trace in the
    window — the traces arrive on the field's declared period, so the
    spacing is close to uniform; the Y axis reads "seconds ago". The X axis
    is the trace's declared physical axis when it has one, its sample index
    otherwise.
    """

    kind: ClassVar[str] = "waterfall"
    data_kind: ClassVar[str] = "trace"
    title_prefix: ClassVar[str] = "Waterfall"
    frame_like: ClassVar[bool] = False

    def _build_extra_header(self, row: QHBoxLayout) -> None:
        self._window_selector = QComboBox()
        self._window_selector.setObjectName(f"waterfall_window_selector_{self._panel_id}")
        self._window_selector.setToolTip("How far back the waterfall reaches")
        for label, _seconds in WATERFALL_WINDOWS:
            self._window_selector.addItem(label)
        self._window_selector.setCurrentText(_DEFAULT_WATERFALL_WINDOW)
        self._window_selector.currentTextChanged.connect(lambda _text: self._redraw())
        row.addWidget(self._window_selector)

    def selected_window_s(self) -> float:
        """Return the selected window in seconds."""
        return dict(WATERFALL_WINDOWS)[self._window_selector.currentText()]

    def settings_entry(self) -> dict[str, Any]:
        """Return the JSON-safe dict the host persists for this panel."""
        entry = super().settings_entry()
        entry["window_s"] = self.selected_window_s()
        return entry

    def apply_settings_entry(self, entry: Mapping[str, Any]) -> None:
        """Restore the key and the window from a persisted entry.

        Args:
            entry: A dict as produced by ``settings_entry()``.
        """
        window_s = entry.get("window_s")
        for label, seconds in WATERFALL_WINDOWS:
            if seconds == window_s:
                self._window_selector.setCurrentText(label)
        super().apply_settings_entry(entry)

    def _redraw(self) -> None:
        key = self.selected_key()
        if key is None:
            return
        info = self._fields[key]
        x_label = "sample"
        if info.axis is not None:
            x_label = f"{info.axis[2]}" if info.axis[2] else "x"
        self._view.set_labels(x_label, "seconds ago", info.unit)

        now = time.time()
        entries = self._history.window(key, self.selected_window_s(), now=now)
        if not entries:
            self._status.setText("Waiting for the first trace…")
            return
        stack = np.vstack([np.asarray(trace, dtype=np.float64) for _t, trace in entries])
        length = stack.shape[1]
        if info.axis is not None:
            x0, x1 = info.axis[0], info.axis[1]
        else:
            x0, x1 = 0.0, float(length - 1)
        oldest_age = now - entries[0][0]
        newest_age = now - entries[-1][0]
        height = max(oldest_age - newest_age, 1e-6)
        # One row per trace, oldest at the bottom: y runs from -oldest_age
        # up to -newest_age, so "seconds ago" reads as a negative offset.
        width = (x1 - x0) if x1 != x0 else float(length)
        self._view.set_frame(stack, rect=(x0, -oldest_age, width, height))
        self._status.setText(f"{len(entries)} trace(s), newest {newest_age:.0f} s ago")
