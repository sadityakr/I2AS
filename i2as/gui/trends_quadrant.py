"""TrendsQuadrant — the Trends panel grid and its MonitorHistory."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import qtawesome as qta
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from i2as.core import trend_history
from i2as.core.paths import log_directory
from i2as.gui import app_settings  # import the module (not the function) so tests can monkeypatch the factory
from i2as.gui.monitor_history import MonitorHistory
from i2as.gui.theme import TEXT_PRIMARY

if TYPE_CHECKING:  # the GUI holds a Station only as a type (contract C19)
    from i2as.core.station import Station
from i2as.gui.trend_plot_panel import TrendPlotPanel

logger = logging.getLogger(__name__)

_MIN_TREND_PANELS = 1
_MAX_TREND_PANELS = 4
_DEFAULT_TREND_PANEL_COUNT = 2

# Default key-selection hints applied to the default trend panels, in
# creation order, once MonitorHistory has keys (first key whose flat name
# contains the hint substring; falls back to the first available key). A
# harmless heuristic: a setup with no matching key simply gets the first key
# it does have, and the operator picks from there.
_DEFAULT_TREND_KEY_HINTS = ("temperature",)

# QSettings key for the persisted trend-panel list. Kept identical to the
# pre-extraction MonitorWindow key so existing saved layouts still restore.
_TRENDS_KEY = "MonitorWindow/trends"


class TrendsQuadrant(QWidget):
    """The Trends quadrant: an auto-gridded set of TrendPlotPanels.

    Owns the shared :class:`MonitorHistory` ring buffer (Qt-free by design,
    see monitor_history.py) feeding all panels. The hosting window connects
    the Orchestrator's ``states_updated`` signal to :meth:`on_states_updated`.

    Args:
        station: The active Station instance (VI names inform the
            default-trend-key picking).
        parent: Optional Qt parent widget.
        log_dir: Directory containing the tiered trend-history JSONL store,
            as resolved by ``i2as.core.paths.log_directory()``.
            ``None`` (the default) resolves it via that function; tests pass
            an explicit ``tmp_path`` instead. Used both for startup
            rehydration (this class) and passed down to each
            ``TrendPlotPanel`` for disk-backed long-window reads.
    """

    def __init__(
        self,
        station: Station,
        parent: QWidget | None = None,
        log_dir: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._station = station
        self._log_dir = log_dir if log_dir is not None else log_directory()
        self.setObjectName("trends_quadrant")

        # Shared ring-buffer history feeding all Trend plot panels.
        self._history = MonitorHistory()
        self._rehydrate_history()

        self._trend_panels: dict[str, TrendPlotPanel] = {}
        self._trend_series_counter = 0
        # Keys the restore path still wants applied once MonitorHistory has
        # data for them (a fresh panel's Y combo is empty until the first
        # states_updated tick, so set_selected_key() at restore time is a
        # harmless no-op that we retry from on_states_updated).
        self._pending_trend_keys: dict[str, str] = {}
        # Same retry pattern for the DEFAULT (non-restored) trend panels'
        # opportunistic default key selection.
        self._default_trend_key_hints: dict[str, str] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("<b>Trends</b>"))
        toolbar.addStretch()
        self._add_trend_btn = QPushButton("Add trend plot")
        self._add_trend_btn.setObjectName("add_trend_btn")
        self._add_trend_btn.setIcon(qta.icon("fa5s.plus", color=TEXT_PRIMARY))
        self._add_trend_btn.setToolTip(f"Add a trend plot (up to {_MAX_TREND_PANELS})")
        self._add_trend_btn.clicked.connect(self._on_trend_add_clicked)
        toolbar.addWidget(self._add_trend_btn)
        outer.addLayout(toolbar)

        self._trends_grid_container = QWidget()
        self._trends_grid = QGridLayout(self._trends_grid_container)
        self._trends_grid.setSpacing(6)

        scroll = QScrollArea()
        scroll.setObjectName("trends_scroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._trends_grid_container)
        outer.addWidget(scroll)

        self._build_default_trend_panels()
        self._update_trend_add_action_state()

    @property
    def history(self) -> MonitorHistory:
        """The shared MonitorHistory ring buffer feeding all trend panels."""
        return self._history

    def _rehydrate_history(self) -> None:
        """Replay the raw trend-history tier into ``self._history`` at startup.

        Reads the raw tier for a window equal to ``MonitorHistory``'s own
        retention and feeds each record's value mapping through
        ``record_flat()``, oldest-first, so the ring buffer is pre-populated
        as if the app had been running the whole time. Never fatal: a
        missing or corrupt log directory must degrade to an empty history,
        not a failed GUI startup.
        """
        try:
            records = trend_history.read_tier(
                self._log_dir, "raw", window_s=self._history.retention_s
            )
            for timestamp, mapping in records:
                self._history.record_flat(mapping, timestamp)
        except Exception:
            logger.exception(
                "trends_quadrant: rehydration from %s failed; starting with empty history",
                self._log_dir,
            )
            return

        if records:
            logger.info(
                "trends_quadrant: rehydrated %d raw-tier record(s) from %s",
                len(records),
                self._log_dir,
            )
        else:
            logger.info(
                "trends_quadrant: no raw-tier history found at %s; starting empty",
                self._log_dir,
            )

    # ------------------------------------------------------------------
    # Panel management
    # ------------------------------------------------------------------

    def _build_default_trend_panels(self) -> None:
        """(Re)create exactly the default number of trend panels.

        Replaces any existing trend panels, then creates
        ``_DEFAULT_TREND_PANEL_COUNT`` fresh ones, with the opportunistic
        default-key hints (applied once MonitorHistory has data — see
        :meth:`on_states_updated`). A panel with no hint of its own simply
        keeps the first key the setup has.
        """
        for panel_id in list(self._trend_panels.keys()):
            self._remove_trend_panel_widget(panel_id)
        self._trend_series_counter = 0
        self._default_trend_key_hints.clear()

        for _ in range(_DEFAULT_TREND_PANEL_COUNT):
            self._add_trend_panel()

        for panel_id, hint in zip(self._trend_panels.keys(), _DEFAULT_TREND_KEY_HINTS):
            self._default_trend_key_hints[panel_id] = hint

    def _create_trend_panel(self) -> tuple[str, TrendPlotPanel]:
        """Create and register a new TrendPlotPanel.

        Registers the panel in ``self._trend_panels`` but does NOT place it
        in the grid — callers call ``_relayout_trend_grid()``.

        Returns:
            ``(panel_id, panel)`` for the caller.
        """
        panel_id = f"trend_{self._next_trend_panel_index()}"
        panel = TrendPlotPanel(
            self._history,
            panel_id,
            series_index=self._trend_series_counter,
            parent=self,
            log_dir=self._log_dir,
        )
        self._trend_series_counter += 1
        panel.remove_requested.connect(self._on_trend_remove_requested)

        self._trend_panels[panel_id] = panel
        return panel_id, panel

    def _relayout_trend_grid(self) -> None:
        """Rebuild the trend grid: current panels arranged in a ceil(sqrt(N)) grid.

        Recomputed from scratch on every add/remove — cheap at N<=4 and
        avoids tracking incremental grid positions separately from
        ``self._trend_panels``' insertion order.
        """
        grid = self._trends_grid
        while grid.count():
            grid.takeAt(0)  # widgets are reparented into the grid on addWidget; not deleted here

        panels = list(self._trend_panels.values())
        if not panels:
            return
        columns = math.ceil(math.sqrt(len(panels)))
        for idx, panel in enumerate(panels):
            row, col = divmod(idx, columns)
            grid.addWidget(panel, row, col)

    def _add_trend_panel(self) -> str:
        """Create, place, and grid-arrange a new trend panel.

        Returns:
            The new panel's ``panel_id``.
        """
        panel_id, _panel = self._create_trend_panel()
        self._relayout_trend_grid()
        self._update_trend_add_action_state()
        return panel_id

    def _next_trend_panel_index(self) -> int:
        """Return the smallest non-negative integer not already used in a panel_id.

        Returns:
            An index such that ``f"trend_{index}"`` is not already in use, so
            panel_ids never collide after panels are added and removed.
        """
        used: set[int] = set()
        for panel_id in self._trend_panels:
            try:
                used.add(int(panel_id.rsplit("_", 1)[-1]))
            except ValueError:
                continue
        index = 0
        while index in used:
            index += 1
        return index

    def _on_trend_add_clicked(self) -> None:
        """Add a trend panel via the quadrant's Add button, up to the cap."""
        if len(self._trend_panels) >= _MAX_TREND_PANELS:
            return
        self._add_trend_panel()

    def _on_trend_remove_requested(self, panel_id: str) -> None:
        """Remove a trend panel, never dropping below the minimum.

        Args:
            panel_id: The panel_id echoed back by TrendPlotPanel.remove_requested.
        """
        if len(self._trend_panels) <= _MIN_TREND_PANELS:
            return
        self._remove_trend_panel_widget(panel_id)
        self._relayout_trend_grid()
        self._update_trend_add_action_state()

    def _remove_trend_panel_widget(self, panel_id: str) -> None:
        """Unconditionally drop a trend panel's widget and bookkeeping.

        Does not relayout the grid — callers that need the grid consistent
        immediately after (as opposed to before a batch of further adds)
        call ``_relayout_trend_grid()`` themselves.

        Args:
            panel_id: The panel_id to remove. No-op if not present.
        """
        panel = self._trend_panels.pop(panel_id, None)
        if panel is not None:
            self._trends_grid.removeWidget(panel)
            panel.setParent(None)
            panel.deleteLater()
        self._pending_trend_keys.pop(panel_id, None)
        self._default_trend_key_hints.pop(panel_id, None)

    def _update_trend_add_action_state(self) -> None:
        """Enable/disable the "Add trend plot" button based on the current panel count."""
        self._add_trend_btn.setEnabled(len(self._trend_panels) < _MAX_TREND_PANELS)

    # ------------------------------------------------------------------
    # Live updates
    # ------------------------------------------------------------------

    def on_states_updated(self, state: dict) -> None:
        """Record a state snapshot into MonitorHistory and refresh trend panels.

        Args:
            state: ``{vi_name: {field: value, ...}}`` from the Orchestrator.
        """
        self._history.record(state)
        for panel_id, panel in self._trend_panels.items():
            panel.refresh()

            pending_key = self._pending_trend_keys.get(panel_id)
            if pending_key is not None:
                panel.set_selected_key(pending_key)
                if panel.selected_key() == pending_key:
                    del self._pending_trend_keys[panel_id]
                continue

            hint = self._default_trend_key_hints.get(panel_id)
            if hint is not None:
                keys = self._history.keys()
                if keys:
                    panel.set_selected_key(self._pick_default_trend_key(hint, keys))
                    del self._default_trend_key_hints[panel_id]

    def _pick_default_trend_key(self, hint: str, keys: list[str]) -> str:
        """Pick the best default trend key for a hint substring (e.g. "temperature").

        Flat keys are ``{vi_name}_{field_name}``, and several VI names
        themselves contain hint words (e.g. ``temperature``,
        ``temperature_sample``), which would make a plain substring search
        match a boring setting field (``temperature_sample_heater_output``)
        before the actual reading. This strips the known vi_name prefix first
        so the hint is matched against the FIELD name, falling back to a
        plain substring match over the whole key, and finally the first key,
        if nothing more specific matches.

        Args:
            hint: Substring to look for (e.g. ``"temperature"``).
            keys: Non-empty, sorted list of MonitorHistory flat keys.

        Returns:
            The chosen flat key.
        """
        for key in keys:
            for vi_name in self._station.get_vi_names():
                prefix = f"{vi_name}_"
                if key.startswith(prefix) and hint in key[len(prefix):]:
                    return key
        for key in keys:
            if hint in key:
                return key
        return keys[0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_settings(self) -> None:
        """Persist the ordered list of trend panels' selected key and window."""
        data = [
            {"key": panel.selected_key(), "window_s": panel.selected_window_s()}
            for panel in self._trend_panels.values()
        ]
        app_settings.get_settings().setValue(_TRENDS_KEY, json.dumps(data))

    def restore_settings(self) -> None:
        """Restore the trend panels from QSettings, defensively.

        A missing key, wrong type, or corrupt JSON all silently keep the
        DEFAULT panels already built in ``__init__``.
        """
        raw_trends = app_settings.get_settings().value(_TRENDS_KEY)
        parsed = None
        if raw_trends:
            try:
                parsed = json.loads(raw_trends)
            except (TypeError, ValueError):
                parsed = None
        if isinstance(parsed, list) and parsed:
            self._apply_trend_restore(parsed)

    def _apply_trend_restore(self, entries: list) -> None:
        """Replace the current trend panels with ones matching saved entries.

        Args:
            entries: Parsed JSON list of ``{"key": ..., "window_s": ...}``
                dicts, already validated to be a non-empty list.
        """
        valid_entries = [e for e in entries if isinstance(e, dict)][:_MAX_TREND_PANELS]
        if not valid_entries:
            return

        for panel_id in list(self._trend_panels.keys()):
            self._remove_trend_panel_widget(panel_id)
        self._default_trend_key_hints.clear()

        for entry in valid_entries:
            panel_id = self._add_trend_panel()
            panel = self._trend_panels[panel_id]

            window_s = entry.get("window_s")
            if isinstance(window_s, (int, float)) and not isinstance(window_s, bool):
                panel.set_selected_window_s(float(window_s))

            key = entry.get("key")
            if isinstance(key, str) and key:
                panel.set_selected_key(key)  # no-op now if history is still empty
                self._pending_trend_keys[panel_id] = key
