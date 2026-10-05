"""The plot-panel extension point: the protocol every Monitor plot panel follows, and the kind registry.

The Monitor's plot quadrant (``gui/trends_quadrant.py``) hosts panels of
several kinds — a scalar trend, a live image, a waterfall of traces — and
knows none of them by name. It knows only:

* :class:`PlotPanel` — what a panel must offer the host;
* :class:`PanelKind` — one registry entry: how the kind is offered (label,
  icon), what it draws (``data_kind``, a ``MonitoredInfo.kind``), which
  history feeds it (``feed``) and how to build one (``factory``);
* :data:`PANEL_KINDS` — the registry, in Add-button order.

**Adding a panel kind** is a panel class that satisfies :class:`PlotPanel`
plus one :data:`PANEL_KINDS` entry. The grid, the Add buttons, the shared
panel cap, the live feeds and the persisted layout then work for it with no
change to the quadrant.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from PyQt6.QtWidgets import QWidget

from i2as.core.events import MonitoredInfo
from i2as.gui.monitor_history import ArrayHistory, MonitorHistory


@runtime_checkable
class PlotPanel(Protocol):
    """What the plot quadrant needs from a panel of any kind.

    Implementations are ``QWidget`` subclasses; ``remove_requested`` is a
    ``pyqtSignal(str)`` emitted with the panel's ``panel_id``.
    """

    kind: str
    remove_requested: Any

    @property
    def panel_id(self) -> str:
        """The host-assigned identifier, unique among the host's panels."""
        ...

    def refresh(self) -> None:
        """Redraw from the shared history this kind's ``feed`` names."""
        ...

    def selected_key(self) -> str | None:
        """The flat key drawn, or ``None`` when there is nothing to pick."""
        ...

    def set_selected_key(self, key: str) -> None:
        """Select a flat key; a no-op for a key not (yet) offered."""
        ...

    def settings_entry(self) -> dict[str, Any]:
        """The JSON-safe dict persisted for this panel; carries ``"kind"``."""
        ...

    def apply_settings_entry(self, entry: Mapping[str, Any]) -> None:
        """Restore from a persisted entry, ignoring what does not fit."""
        ...


@dataclass
class PanelContext:
    """Everything a panel factory may need from its host, in one place.

    Attributes:
        history: The scalar ``MonitorHistory`` (the ``"states"`` feed).
        array_history: The ``ArrayHistory`` (the ``"arrays"`` feed).
        array_fields: ``{flat_key: MonitoredInfo}`` of the station's
            declared image/trace fields.
        log_dir: The trend-history store, for disk-backed trend windows.
        parent: The Qt parent for new panels.
        series_counter: How many colour-series indices have been handed out
            (a trend panel takes the next one).
    """

    history: MonitorHistory
    array_history: ArrayHistory
    array_fields: dict[str, MonitoredInfo]
    log_dir: Path
    parent: QWidget | None = None
    series_counter: int = field(default=0)

    def next_series_index(self) -> int:
        """Hand out the next colour-series index."""
        index = self.series_counter
        self.series_counter += 1
        return index


@dataclass(frozen=True)
class PanelKind:
    """One entry of the panel-kind registry.

    Attributes:
        label: The Add button's caption.
        icon: The Add button's qtawesome icon name.
        data_kind: The ``MonitoredInfo.kind`` the panel draws. A kind whose
            data kind the station declares no field of is not offered.
        feed: Which history update refreshes it: ``"states"`` (every scalar
            tick) or ``"arrays"`` (every array payload).
        factory: ``(context, panel_id) -> PlotPanel``.
    """

    label: str
    icon: str
    data_kind: str
    feed: Literal["states", "arrays"]
    factory: Callable[[PanelContext, str], PlotPanel]


def _trend(context: PanelContext, panel_id: str) -> PlotPanel:
    from i2as.gui.trend_plot_panel import TrendPlotPanel

    return TrendPlotPanel(
        context.history,
        panel_id,
        series_index=context.next_series_index(),
        parent=context.parent,
        log_dir=context.log_dir,
    )


def _image(context: PanelContext, panel_id: str) -> PlotPanel:
    from i2as.gui.array_plot_panels import ImagePlotPanel

    return ImagePlotPanel(
        context.array_history, panel_id, context.array_fields, parent=context.parent
    )


def _waterfall(context: PanelContext, panel_id: str) -> PlotPanel:
    from i2as.gui.array_plot_panels import WaterfallPlotPanel

    return WaterfallPlotPanel(
        context.array_history, panel_id, context.array_fields, parent=context.parent
    )


#: The panel kinds the Monitor offers, in Add-button order. Keys are each
#: panel class's ``kind`` and the ``"kind"`` of a persisted layout entry.
PANEL_KINDS: dict[str, PanelKind] = {
    "trend": PanelKind("Trend", "fa5s.chart-line", "scalar", "states", _trend),
    "image": PanelKind("Image", "fa5s.image", "image", "arrays", _image),
    "waterfall": PanelKind("Waterfall", "fa5s.water", "trace", "arrays", _waterfall),
}
