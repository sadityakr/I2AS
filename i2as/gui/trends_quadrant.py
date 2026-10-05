"""TrendsQuadrant — the Monitor window's plot grid and the histories feeding it.

The quadrant hosts plot panels of several **kinds**, one per value kind a
``@monitored`` field can declare (the monitored-kind standard,
``core/decorators.py``):

==========  ===========================  ==================================
Panel kind  Draws                        Fed by
==========  ===========================  ==================================
trend       a scalar over time           ``states_updated`` ->
                                         ``MonitorHistory`` (+ disk tiers)
image       the newest 2-D frame         ``monitored_arrays_updated`` ->
                                         ``ArrayHistory`` (RAM only)
waterfall   a 1-D trace stacked in time  ``monitored_arrays_updated`` ->
                                         ``ArrayHistory`` (RAM only)
==========  ===========================  ==================================

The quadrant knows no kind by name: every panel follows the
:class:`~i2as.gui.plot_panel.PlotPanel` protocol, and each kind's label,
data kind, feed and factory come from :data:`~i2as.gui.plot_panel.PANEL_KINDS`
— adding a kind changes nothing here (see ``gui/plot_panel.py``).

``PlotsQuadrant`` was ``TrendsQuadrant`` when it hosted trends only; the old
name stays as an alias, and the module, objectNames and QSettings key keep
theirs so saved layouts and existing callers are untouched.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from i2as.core.events import MonitoredInfo
from i2as.core.paths import log_directory
from i2as.gui import app_settings  # import the module (not the function) so tests can monkeypatch the factory
from i2as.gui.monitor_history import ArrayHistory, MonitorHistory
from i2as.gui.plot_panel import PANEL_KINDS, PanelContext, PlotPanel
from i2as.gui.theme import TEXT_PRIMARY

if TYPE_CHECKING:  # the GUI holds a Station only as a type (contract C19)
    from i2as.core.station import Station

logger = logging.getLogger(__name__)

_MIN_TREND_PANELS = 1
# The cap counts every panel, whatever its kind: it is a screen-space limit.
_MAX_TREND_PANELS = 4
_DEFAULT_TREND_PANEL_COUNT = 2

# Default key-selection hints applied to the default trend panels, in
# creation order, once MonitorHistory has keys (first key whose flat name
# contains the hint substring; falls back to the first available key). A
# harmless heuristic: a setup with no matching key simply gets the first key
# it does have, and the operator picks from there.
_DEFAULT_TREND_KEY_HINTS = ("temperature",)

# QSettings key for the persisted panel list. Kept identical to the
# pre-extraction MonitorWindow key so existing saved layouts still restore
# (an entry without a "kind" is a trend, which is all they ever held).
_TRENDS_KEY = "MonitorWindow/trends"


def array_fields_from_station_info(station_info: Any) -> dict[str, MonitoredInfo]:
    """Collect every declared array field of a station, by flat key.

    Args:
        station_info: An ``events.StationInfo`` (duck-typed: anything with
            ``instruments``, each with ``name`` and ``monitored``).

    Returns:
        ``{f"{vi_name}_{field}": MonitoredInfo}`` for every ``@monitored``
        field whose kind is not ``"scalar"`` — the keys
        ``ArrayHistory`` records them under.
    """
    fields: dict[str, MonitoredInfo] = {}
    for instrument in getattr(station_info, "instruments", ()):
        for info in instrument.monitored:
            if info.kind != "scalar":
                fields[f"{instrument.name}_{info.name}"] = info
    return fields


class PlotsQuadrant(QWidget):
    """The Monitor's plot quadrant: an auto-gridded set of plot panels of any kind.

    Owns the shared :class:`MonitorHistory` (scalars, Qt-free by design, see
    monitor_history.py) and :class:`ArrayHistory` (images and traces) that
    feed the panels. The hosting window connects the Orchestrator's
    ``states_updated`` signal to :meth:`on_states_updated` and
    ``monitored_arrays_updated`` to :meth:`on_arrays_updated`.

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
        array_fields: ``{flat_key: MonitoredInfo}`` of the station's
            declared image/trace fields (see
            :func:`array_fields_from_station_info`). ``None`` or empty means
            the station declares none, and only trend panels are offered.
    """

    def __init__(
        self,
        station: Station,
        parent: QWidget | None = None,
        log_dir: Path | None = None,
        array_fields: Mapping[str, MonitoredInfo] | None = None,
    ) -> None:
        super().__init__(parent)
        self._station = station
        self._log_dir = log_dir if log_dir is not None else log_directory()
        self._array_fields: dict[str, MonitoredInfo] = dict(array_fields or {})
        self.setObjectName("trends_quadrant")

        # Shared ring-buffer history feeding all trend panels.
        self._history = MonitorHistory()
        self._rehydrate_history()
        # Shared RAM-only history feeding the image and waterfall panels.
        self._array_history = ArrayHistory()
        # What every panel factory gets (gui/plot_panel.py).
        self._context = PanelContext(
            history=self._history,
            array_history=self._array_history,
            array_fields=self._array_fields,
            log_dir=Path(self._log_dir),
            parent=self,
        )

        # Every panel, of every kind, in grid order.
        self._panels: dict[str, PlotPanel] = {}
        # Keys the restore path still wants applied once the panel offers
        # them (a fresh trend panel's Y combo is empty until the first
        # states_updated tick, so set_selected_key() at restore time is a
        # harmless no-op that is retried on each refresh of its feed).
        self._pending_trend_keys: dict[str, str] = {}
        # Same retry pattern for the DEFAULT (non-restored) trend panels'
        # opportunistic default key selection.
        self._default_trend_key_hints: dict[str, str] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("<b>Plots</b>"))
        toolbar.addStretch()
        toolbar.addWidget(QLabel("Add:"))
        self._add_buttons: dict[str, QPushButton] = {}
        for kind, spec in PANEL_KINDS.items():
            button = QPushButton(spec.label)
            button.setObjectName(f"add_{kind}_btn")
            button.setIcon(qta.icon(spec.icon, color=TEXT_PRIMARY))
            button.setToolTip(
                f"Add a {spec.label.lower()} plot (up to {_MAX_TREND_PANELS} plots)"
            )
            button.clicked.connect(lambda _checked=False, k=kind: self._on_add_clicked(k))
            button.setVisible(self._kind_available(kind))
            toolbar.addWidget(button)
            self._add_buttons[kind] = button
        self._add_trend_btn = self._add_buttons["trend"]
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
        self._update_add_buttons_state()

    @property
    def history(self) -> MonitorHistory:
        """The shared MonitorHistory ring buffer feeding all trend panels."""
        return self._history

    @property
    def array_history(self) -> ArrayHistory:
        """The shared ArrayHistory feeding the image and waterfall panels."""
        return self._array_history

    def panels(self) -> dict[str, PlotPanel]:
        """Return ``{panel_id: panel}`` for every panel, in grid order."""
        return dict(self._panels)

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

    def _kind_available(self, kind: str) -> bool:
        """Whether a panel kind has anything to draw on this station.

        Args:
            kind: A ``PANEL_KINDS`` key.

        Returns:
            ``True`` for a scalar kind (every station has scalars) and for
            an array kind at least one declared field has.
        """
        data_kind = PANEL_KINDS[kind].data_kind
        if data_kind == "scalar":
            return True
        return any(info.kind == data_kind for info in self._array_fields.values())

    def _build_default_trend_panels(self) -> None:
        """(Re)create exactly the default number of trend panels.

        Replaces any existing panels, then creates
        ``_DEFAULT_TREND_PANEL_COUNT`` fresh trend ones, with the
        opportunistic default-key hints (applied once MonitorHistory has
        data — see :meth:`on_states_updated`). A panel with no hint of its
        own simply keeps the first key the setup has.
        """
        for panel_id in list(self._panels.keys()):
            self._remove_panel_widget(panel_id)
        self._context.series_counter = 0
        self._default_trend_key_hints.clear()

        for _ in range(_DEFAULT_TREND_PANEL_COUNT):
            self._add_panel("trend")

        for panel_id, hint in zip(self._panels.keys(), _DEFAULT_TREND_KEY_HINTS):
            self._default_trend_key_hints[panel_id] = hint

    def _create_panel(self, kind: str) -> tuple[str, PlotPanel]:
        """Create and register a new panel of ``kind`` through its registry factory.

        Registers the panel in ``self._panels`` but does NOT place it in the
        grid — callers call ``_relayout_trend_grid()``.

        Args:
            kind: A ``PANEL_KINDS`` key.

        Returns:
            ``(panel_id, panel)`` for the caller.

        Raises:
            ValueError: If ``kind`` is not registered.
        """
        spec = PANEL_KINDS.get(kind)
        if spec is None:
            raise ValueError(f"unknown plot panel kind {kind!r}")
        panel_id = f"{kind}_{self._next_panel_index()}"
        panel = spec.factory(self._context, panel_id)
        panel.remove_requested.connect(self._on_remove_requested)
        self._panels[panel_id] = panel
        return panel_id, panel

    def _relayout_trend_grid(self) -> None:
        """Rebuild the grid: current panels arranged in a ceil(sqrt(N)) grid.

        Recomputed from scratch on every add/remove — cheap at N<=4 and
        avoids tracking incremental grid positions separately from
        ``self._panels``' insertion order.
        """
        grid = self._trends_grid
        while grid.count():
            grid.takeAt(0)  # widgets are reparented into the grid on addWidget; not deleted here

        panels = list(self._panels.values())
        if not panels:
            return
        columns = math.ceil(math.sqrt(len(panels)))
        for idx, panel in enumerate(panels):
            row, col = divmod(idx, columns)
            grid.addWidget(panel, row, col)

    def _add_panel(self, kind: str) -> str:
        """Create, place, and grid-arrange a new panel of ``kind``.

        Args:
            kind: A ``PANEL_KINDS`` key.

        Returns:
            The new panel's ``panel_id``.
        """
        panel_id, panel = self._create_panel(kind)
        self._relayout_trend_grid()
        self._update_add_buttons_state()
        self._refresh_panel(panel_id, panel)
        return panel_id

    def _next_panel_index(self) -> int:
        """Return the smallest non-negative integer not already used in a panel_id.

        Indices are shared across kinds, so ``trend_0`` and ``image_0``
        never coexist and a panel_id names one panel however it is spelled.

        Returns:
            An index not already used by any panel's ``panel_id`` suffix.
        """
        used: set[int] = set()
        for panel_id in self._panels:
            try:
                used.add(int(panel_id.rsplit("_", 1)[-1]))
            except ValueError:
                continue
        index = 0
        while index in used:
            index += 1
        return index

    def _on_add_clicked(self, kind: str) -> None:
        """Add a panel of ``kind`` via its Add button, up to the cap.

        Args:
            kind: A ``PANEL_KINDS`` key.
        """
        if len(self._panels) >= _MAX_TREND_PANELS or not self._kind_available(kind):
            return
        self._add_panel(kind)

    def _on_trend_add_clicked(self) -> None:
        """Add a trend panel, up to the cap (kept for existing callers)."""
        self._on_add_clicked("trend")

    def _on_remove_requested(self, panel_id: str) -> None:
        """Remove a panel, never dropping below the minimum.

        Args:
            panel_id: The panel_id echoed back by the panel's remove_requested.
        """
        if len(self._panels) <= _MIN_TREND_PANELS:
            return
        self._remove_panel_widget(panel_id)
        self._relayout_trend_grid()
        self._update_add_buttons_state()

    def _remove_panel_widget(self, panel_id: str) -> None:
        """Unconditionally drop a panel's widget and bookkeeping.

        Does not relayout the grid — callers that need the grid consistent
        immediately after (as opposed to before a batch of further adds)
        call ``_relayout_trend_grid()`` themselves.

        Args:
            panel_id: The panel_id to remove. No-op if not present.
        """
        panel = self._panels.pop(panel_id, None)
        if panel is not None:
            self._trends_grid.removeWidget(panel)
            panel.setParent(None)
            panel.deleteLater()
        self._pending_trend_keys.pop(panel_id, None)
        self._default_trend_key_hints.pop(panel_id, None)

    def _update_add_buttons_state(self) -> None:
        """Enable/disable the Add buttons based on the current panel count."""
        room = len(self._panels) < _MAX_TREND_PANELS
        for button in self._add_buttons.values():
            button.setEnabled(room)

    # ------------------------------------------------------------------
    # Live updates
    # ------------------------------------------------------------------

    def on_states_updated(self, state: dict) -> None:
        """Record a state snapshot into MonitorHistory and refresh the ``"states"``-fed panels.

        Args:
            state: ``{vi_name: {field: value, ...}}`` from the Orchestrator.
        """
        self._history.record(state)
        self._refresh_feed("states")

    def on_arrays_updated(self, arrays: dict) -> None:
        """Record an array payload into ArrayHistory and refresh the ``"arrays"``-fed panels.

        Args:
            arrays: ``{vi_name: {field: ndarray}}`` from the Orchestrator's
                ``monitored_arrays_updated`` — only the fields read that
                tick.
        """
        self._array_history.record(arrays)
        self._refresh_feed("arrays")

    def _refresh_feed(self, feed: str) -> None:
        """Refresh every panel whose kind is fed by ``feed``.

        Args:
            feed: ``"states"`` or ``"arrays"``.
        """
        for panel_id, panel in list(self._panels.items()):
            if PANEL_KINDS[panel.kind].feed == feed:
                self._refresh_panel(panel_id, panel)

    def _refresh_panel(self, panel_id: str, panel: PlotPanel) -> None:
        """Redraw one panel, then apply any key it was still waiting to offer.

        A failing redraw costs that panel one redraw, never the Monitor.

        Args:
            panel_id: The panel's id.
            panel: The panel.
        """
        try:
            panel.refresh()
        except Exception:
            logger.exception("plots_quadrant: %s failed to redraw", panel_id)
            return

        pending_key = self._pending_trend_keys.get(panel_id)
        if pending_key is not None:
            panel.set_selected_key(pending_key)
            if panel.selected_key() == pending_key:
                del self._pending_trend_keys[panel_id]
            return

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
        """Persist the ordered list of panels, each as its own settings entry."""
        data = [panel.settings_entry() for panel in self._panels.values()]
        app_settings.get_settings().setValue(_TRENDS_KEY, json.dumps(data))

    def restore_settings(self) -> None:
        """Restore the panels from QSettings, defensively.

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
        """Replace the current panels with ones matching saved entries.

        An entry without a ``"kind"`` is a trend (every layout saved before
        panel kinds existed). An entry whose kind is unknown, or has nothing
        to draw on this station (an image panel saved on a setup that had a
        camera), is skipped.

        Args:
            entries: Parsed JSON list of entry dicts (see each panel's
                ``settings_entry()``), already validated to be a non-empty
                list.
        """
        valid_entries = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("kind", "trend") in PANEL_KINDS
            and self._kind_available(entry.get("kind", "trend"))
        ][:_MAX_TREND_PANELS]
        if not valid_entries:
            return

        for panel_id in list(self._panels.keys()):
            self._remove_panel_widget(panel_id)
        self._default_trend_key_hints.clear()

        for entry in valid_entries:
            kind = entry.get("kind", "trend")
            panel_id = self._add_panel(kind)
            panel = self._panels[panel_id]
            panel.apply_settings_entry(entry)

            # A key the panel does not offer yet (a trend before the first
            # tick) is retried on each refresh of its feed until it sticks.
            key = entry.get("key")
            if isinstance(key, str) and key and panel.selected_key() != key:
                self._pending_trend_keys[panel_id] = key


#: The pre-kinds name, kept so existing callers and tests keep working.
TrendsQuadrant = PlotsQuadrant
