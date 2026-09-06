"""TrendPlotPanel — reusable time-series trend plot panel for the Monitor window.

Each panel owns one Y-variable selector, one time-window selector, and a
themed ``pyqtgraph`` curve plotted against wall-clock time. It reads from a
shared :class:`~i2as.gui.monitor_history.MonitorHistory` ring buffer that
the host (the Monitor window) feeds from ``Orchestrator.states_updated``; this
widget never talks to the Orchestrator directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyqtgraph as pg
import qtawesome as qta
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from i2as.core import trend_history
from i2as.core.paths import log_directory
from i2as.gui.monitor_history import MonitorHistory
from i2as.gui.theme import PLOT_SERIES, TEXT_PRIMARY

logger = logging.getLogger(__name__)

# (label, window in seconds) — module constant so the Monitor window and
# tests can both reference the same option list without duplicating it.
# Windows beyond "24 h" exceed MonitorHistory's in-RAM retention and are
# served from the disk-backed tiered trend-history store instead (see
# _redraw()'s RAM/disk split, keyed off this same list via _is_disk_window()).
TIME_WINDOWS: list[tuple[str, float]] = [
    ("15 min", 900.0),
    ("1 h", 3600.0),
    ("6 h", 21600.0),
    ("24 h", 86400.0),
    ("7 d", 604800.0),
    ("1 y", 31536000.0),
]
_DEFAULT_WINDOW_LABEL = "1 h"

# Windows at or below this many seconds are always served from the in-RAM
# MonitorHistory ring buffer (its own default retention); anything longer is
# read from disk via trend_history.read_window(), which picks the tier
# itself (trend_history.pick_tier() — tier choice does not belong in the
# GUI, see GLOSSARY.md "Trend tier").
_RAM_WINDOW_CEILING_S = 86400.0


class TrendPlotPanel(QGroupBox):
    """A single time-series trend panel: a Y-variable selector, a time-window
    selector, a remove button, and a themed ``pyqtgraph`` curve plotted
    against wall-clock time.

    This mirrors the *widget extraction* pattern used by ``LivePlotPanel``:
    a self-contained ``QGroupBox`` that owns its selectors and curve, so the
    Monitor window can host several of these side by side without duplicating
    plot-construction code. Unlike ``LivePlotPanel`` (which plots one
    instrument value against another), this panel always plots a single
    variable against real time, using a ``DateAxisItem`` so the X axis shows
    human-readable clock ticks instead of raw Unix timestamps.

    Attributes:
        remove_requested: Signal emitted with this panel's ``panel_id`` when
            the user clicks the remove button, so the host can drop the panel
            from its layout and bookkeeping.

    Args:
        history: The shared ``MonitorHistory`` ring buffer this panel reads
            from. Not owned by this widget; the host keeps recording into it.
        panel_id: A host-assigned identifier for this panel instance (e.g.
            ``"trend_0"``), used to build objectNames and echoed back on
            ``remove_requested`` so the host knows which panel to drop.
        series_index: Index into ``theme.PLOT_SERIES`` used to pick this
            panel's curve colour, the same convention ``LivePlotPanel`` uses
            for its two plots.
        parent: Optional Qt parent widget.
        log_dir: Directory containing the tiered trend-history JSONL store,
            used only for windows longer than the in-RAM ``MonitorHistory``
            can serve (``"7 d"``, ``"1 y"``). ``None`` (the default) resolves
            it via ``i2as.core.paths.log_directory()``; tests
            pass an explicit ``tmp_path`` instead.
    """

    remove_requested = pyqtSignal(str)

    def __init__(
        self,
        history: MonitorHistory,
        panel_id: str,
        series_index: int = 0,
        parent: QWidget | None = None,
        log_dir: Path | None = None,
    ) -> None:
        super().__init__(f"Trend — {panel_id}", parent)

        self._history = history
        self._panel_id = panel_id
        self._known_keys: list[str] = []
        self._log_dir = log_dir if log_dir is not None else log_directory()

        self.setMinimumSize(260, 180)

        vlay = QVBoxLayout(self)

        top_row = QHBoxLayout()

        self._y_selector = QComboBox()
        self._y_selector.setObjectName(f"trend_y_selector_{panel_id}")
        # AdjustToContents sizes both the closed combo and its dropdown to
        # the widest key currently in the list — without it, Qt's default
        # policy truncates long flat keys (e.g. "temperature_sample_heater_output")
        # in both places, leaving no way to tell which variable is selected.
        self._y_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._y_selector.currentTextChanged.connect(self._redraw)
        # Stretch factor 1: this combo claims all leftover row width, pushing
        # the time-window selector and remove button to the right instead of
        # them sitting immediately after it at their own natural size.
        top_row.addWidget(self._y_selector, 1)

        self._window_selector = QComboBox()
        self._window_selector.setObjectName(f"trend_window_selector_{panel_id}")
        self._window_selector.setToolTip("Time window shown on this trend")
        for label, _seconds in TIME_WINDOWS:
            self._window_selector.addItem(label)
        self._window_selector.setCurrentText(_DEFAULT_WINDOW_LABEL)
        self._window_selector.currentTextChanged.connect(self._redraw)
        top_row.addWidget(self._window_selector)

        self._remove_button = QPushButton()
        self._remove_button.setObjectName(f"trend_remove_button_{panel_id}")
        self._remove_button.setIcon(qta.icon("fa5s.trash", color=TEXT_PRIMARY))
        self._remove_button.setToolTip("Remove this trend plot")
        self._remove_button.clicked.connect(
            lambda: self.remove_requested.emit(self._panel_id)
        )
        top_row.addWidget(self._remove_button)

        vlay.addLayout(top_row)

        self._plot_widget = pg.PlotWidget(
            axisItems={"bottom": pg.DateAxisItem(orientation="bottom")}
        )
        self._plot_widget.setObjectName(f"trend_plot_{panel_id}")
        self._plot_widget.setMinimumHeight(140)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        series_color = PLOT_SERIES[series_index % len(PLOT_SERIES)]
        pen = pg.mkPen(series_color, width=2)
        self._curve = self._plot_widget.plot(
            [], [], pen=pen, symbol="o", symbolSize=5, symbolBrush=series_color, symbolPen=None
        )
        vlay.addWidget(self._plot_widget)

        self._update_y_label()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Repopulate the Y selector from the history and redraw the curve.

        Repopulating preserves the current selection when it is still valid;
        if nothing is selected yet, the combo is left on its first entry once
        keys exist. Safe to call on every Orchestrator tick.
        """
        keys = self._history.keys()
        if keys != self._known_keys:
            self._known_keys = keys
            prev = self._y_selector.currentText()
            self._y_selector.blockSignals(True)
            self._y_selector.clear()
            self._y_selector.addItems(keys)
            if prev in keys:
                self._y_selector.setCurrentText(prev)
            elif keys:
                self._y_selector.setCurrentIndex(0)
            self._y_selector.blockSignals(False)
        self._redraw()

    def selected_key(self) -> str | None:
        """Return the currently selected Y-variable key.

        Returns:
            The selected flat key, or ``None`` when the combo is empty.
        """
        text = self._y_selector.currentText()
        return text if text else None

    def set_selected_key(self, key: str) -> None:
        """Select a Y-variable key and redraw.

        Args:
            key: The flat key to select. No-op if ``key`` is not currently a
                selectable item in the combo.
        """
        if self._y_selector.findText(key) < 0:
            return
        self._y_selector.setCurrentText(key)

    def selected_window_s(self) -> float:
        """Return the currently selected time window, in seconds.

        Returns:
            The window duration in seconds corresponding to the selected
            label in ``TIME_WINDOWS``.
        """
        label = self._window_selector.currentText()
        for window_label, seconds in TIME_WINDOWS:
            if window_label == label:
                return seconds
        return dict(TIME_WINDOWS)[_DEFAULT_WINDOW_LABEL]

    def set_selected_window_s(self, window_s: float) -> None:
        """Select a time window by its value in seconds and redraw.

        Args:
            window_s: The window duration in seconds. No-op if it does not
                match one of ``TIME_WINDOWS``' values.
        """
        for label, seconds in TIME_WINDOWS:
            if seconds == window_s:
                self._window_selector.setCurrentText(label)
                return

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        """Fetch the windowed series for the selected key and update the curve."""
        key = self.selected_key()
        # Belt-and-suspenders alongside AdjustToContents: a hover tooltip
        # with the full key, in case the panel is ever narrower than the
        # combo would like to be.
        self._y_selector.setToolTip(key or "Variable to plot on this trend")
        if not key or key not in self._known_keys:
            self._curve.setData([], [])
            return
        window_s = self.selected_window_s()
        if window_s <= _RAM_WINDOW_CEILING_S:
            times, values = self._history.series(key, window_s=window_s)
        else:
            times, values = self._read_disk_series(key, window_s)
        self._curve.setData(times, values)
        self._update_y_label()

    def _read_disk_series(self, key: str, window_s: float) -> tuple[list[float], list[float]]:
        """Read a disk-backed series for a window beyond MonitorHistory's retention.

        Delegates tier selection entirely to ``trend_history.read_window()``
        (that decision belongs in the store, not the GUI). Synchronous by
        design: the disk reads involved (~3,840 lines for a week, ~8,760 for
        a year) are small enough to read on the tick/redraw path without a
        thread, which this codebase's single-threaded cooperative scheduling
        forbids anyway.

        Args:
            key: The flat key to plot.
            window_s: The requested window length in seconds.

        Returns:
            ``(times, values)`` parallel lists, empty on any read failure or
            if the key has no persisted data in the window — including a
            measurement-VI key, which ``Station.last_state_flat()`` excludes
            from every tier, so it plots live but is always empty here (see
            GLOSSARY.md "Trend history").
        """
        try:
            series = trend_history.read_window(self._log_dir, [key], window_s)[key]
        except Exception:
            logger.exception(
                "trend_plot_panel[%s]: disk read failed for key=%r window_s=%s",
                self._panel_id, key, window_s,
            )
            return [], []
        if not series:
            return [], []
        times = [t for t, _ in series]
        values = [v for _, v in series]
        return times, values

    def _update_y_label(self) -> None:
        """Set the Y-axis label to the currently selected key string."""
        key = self.selected_key()
        self._plot_widget.setLabel("left", key if key else "")
