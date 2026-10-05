"""LivePlotPanel — reusable live plot panel for ProcedureWindow: XY or image.

A panel has a **kind**, picked per panel with its kind selector:

* ``"xy"`` — one scalar column against another, over the run's datapoints
  (the panel as it always was);
* ``"image"`` — one frame of an image block (the image-block standard,
  ``core/plan.py``'s ``ImageBlock``), for the latest datapoint or any
  earlier one picked with the Point selector.

Both kinds share the Loop selectors: an XY plot indexes each scalar
column's ``(n_loop1, n_loop2)`` grid with them, an image plot picks the
frame of that reading. The image is drawn by the same
:class:`~i2as.gui.image_view.ImageView` the Monitor's image panels use.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from i2as.core.data_reader import select_frame
from i2as.gui.image_view import ImageView

logger = logging.getLogger(__name__)

#: The plot kinds a LivePlotPanel offers, ``(data, label)`` in selector order.
PLOT_KINDS: tuple[tuple[str, str], ...] = (("xy", "XY"), ("image", "Image"))


def _to_float(raw: Any) -> float:
    """Convert a datapoint value to a plottable float, or NaN when absent.

    Every plottable column is already a scalar: computing it — the mean of
    a quantity's raw samples — is the measurement method's responsibility
    (``MeasurementInstrumentBase.mean_and_sem``), never the plot panel's.
    ``*_array`` columns are excluded from the plottable key list, so this
    function does no array reduction; a non-scalar reaching it is a bug
    upstream, not something to paper over here.

    Args:
        raw: The raw value pulled from a datapoint dict (may be ``None``
            or a scalar).

    Returns:
        The value as a float, or NaN when the value is ``None``.
    """
    return float("nan") if raw is None else float(raw)


class LivePlotPanel(QGroupBox):
    """A live plot panel: a kind selector, per-kind selectors, Loop selectors, and the plot.

    In ``"xy"`` mode it shows X/Y axis selectors and a themed pyqtgraph
    curve; in ``"image"`` mode an image-block selector, a Point selector
    (``latest`` follows the run) and an :class:`ImageView`. The Image kind
    is offered only while the selected procedure records image blocks (see
    ``set_available_image_blocks``).

    This is a *widget extraction* — the pattern of pulling a repeated block of
    UI (here, ProcedureWindow's near-identical Plot 1 / Plot 2) into one
    reusable widget class so the two instances stay in lock-step and the parent
    shrinks. Each panel keeps a reference to the last datapoint history handed to
    ``redraw`` so that changing an axis selector can immediately redraw without
    the parent re-supplying the data. The two reading-loop selectors (one per
    loop slot) are hidden by default; call ``set_available_loop_labels`` to
    show them for looped measurements. Axis keys stay the PLAIN column names —
    each measurement column is a real ``(n_loop1, n_loop2)`` grid in every
    datapoint, and the selected loop indices (0 when a slot is inactive) are
    used to index directly into it at draw time.

    Args:
        title: Group-box title (e.g. ``"Plot 1"``).
        series_color: Hex colour for the pen and symbols (from ``PLOT_SERIES``).
        x_selector_name: objectName for the X-axis combo (must match the legacy
            name, e.g. ``"x1_axis_selector"``, so ``findChild`` keeps working).
        y_selector_name: objectName for the Y-axis combo (e.g. ``"y_axis_selector"``).
        plot_object_name: objectName for the PlotWidget (e.g. ``"live_plot"``).
        loop1_selector_name: objectName for the slot-1 Loop combo
            (e.g. ``"plot1_loop1_selector"``).
        loop2_selector_name: objectName for the slot-2 Loop combo
            (e.g. ``"plot1_loop2_selector"``).
        parent: Optional Qt parent widget.
    """

    def __init__(
        self,
        title: str,
        series_color: str,
        x_selector_name: str,
        y_selector_name: str,
        plot_object_name: str,
        loop1_selector_name: str = "",
        loop2_selector_name: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)

        # Last datapoint history handed to redraw(); a selector change redraws
        # against this without the parent re-supplying it.
        self._datapoints: list[dict] = []

        # Declared image blocks of the selected procedure: {name: unit}.
        self._image_blocks: dict[str, str] = {}

        vlay = QVBoxLayout(self)

        axis_row = QHBoxLayout()
        self._kind_selector = QComboBox()
        self._kind_selector.setObjectName(f"{plot_object_name}_kind_selector")
        self._kind_selector.setToolTip("What this panel plots")
        for kind, label in PLOT_KINDS:
            self._kind_selector.addItem(label, kind)
        self._kind_selector.currentIndexChanged.connect(self._on_kind_changed)
        axis_row.addWidget(self._kind_selector)

        # XY-mode selectors.
        self._x_label = QLabel("X axis:")
        axis_row.addWidget(self._x_label)
        self._x_selector = QComboBox()
        self._x_selector.setObjectName(x_selector_name)
        self._x_selector.currentTextChanged.connect(self._redraw)
        axis_row.addWidget(self._x_selector)

        self._y_label = QLabel("Y axis:")
        axis_row.addWidget(self._y_label)
        self._y_selector = QComboBox()
        self._y_selector.setObjectName(y_selector_name)
        self._y_selector.currentTextChanged.connect(self._redraw)
        axis_row.addWidget(self._y_selector)

        # Image-mode selectors: which block, and which datapoint (0 = the
        # latest, followed as the run goes on; k = the k-th point).
        self._image_label = QLabel("Image:")
        axis_row.addWidget(self._image_label)
        self._image_selector = QComboBox()
        self._image_selector.setObjectName(f"{plot_object_name}_image_selector")
        self._image_selector.currentTextChanged.connect(self._redraw)
        axis_row.addWidget(self._image_selector)
        self._point_label = QLabel("Point:")
        axis_row.addWidget(self._point_label)
        self._point_selector = QSpinBox()
        self._point_selector.setObjectName(f"{plot_object_name}_point_selector")
        self._point_selector.setSpecialValueText("latest")
        self._point_selector.setRange(0, 0)
        self._point_selector.setToolTip("Which datapoint's frame to show; 'latest' follows the run")
        self._point_selector.valueChanged.connect(self._redraw)
        axis_row.addWidget(self._point_selector)

        # Reading-loop selectors, one per loop slot: each picks WHICH reading
        # of the datapoint is plotted (slot 1 labels A1, A2, ...; slot 2
        # labels B1, B2, ...). Hidden until set_available_loop_labels() is
        # called with a non-None map for the slot.
        self._loop_labels_ui: list[QLabel] = []
        self._loop_selectors: list[QComboBox] = []
        for ordinal, object_name in (
            ("1", loop1_selector_name or "loop1_selector"),
            ("2", loop2_selector_name or "loop2_selector"),
        ):
            label = QLabel(f"Loop {ordinal}:")
            axis_row.addWidget(label)
            selector = QComboBox()
            selector.setObjectName(object_name)
            selector.currentIndexChanged.connect(self._redraw)
            axis_row.addWidget(selector)
            label.setVisible(False)
            selector.setVisible(False)
            self._loop_labels_ui.append(label)
            self._loop_selectors.append(selector)

        axis_row.addStretch()
        vlay.addLayout(axis_row)

        self._stack = QStackedWidget()
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setObjectName(plot_object_name)
        self._plot_widget.setMinimumHeight(150)
        self._plot_widget.setLabel("bottom", "Field (T)")
        self._plot_widget.setLabel("left", "Value")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        pen = pg.mkPen(series_color, width=2)
        self._curve = self._plot_widget.plot(
            [], [], pen=pen, symbol="o", symbolSize=5,
            symbolBrush=series_color, symbolPen=series_color,
        )
        self._stack.addWidget(self._plot_widget)
        self._image_view = ImageView(object_name=f"{plot_object_name}_image")
        self._stack.addWidget(self._image_view)
        vlay.addWidget(self._stack)

        self._apply_kind_visibility()
        self._refresh_kind_availability()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_available_keys(
        self, keys: list[str], default_x: str, default_y: str | None
    ) -> None:
        """Repopulate both axis selectors, preserving a still-valid selection.

        Safe to call before a run starts. When a selector's current choice is
        still present in the new key list it is kept, so the user's axis choices
        survive switching procedures. ``blockSignals`` prevents a redraw storm
        while the combos are being rebuilt.

        Args:
            keys: The full list of selectable axis keys.
            default_x: The X-axis key to fall back to when the current X choice
                is not in ``keys``.
            default_y: The Y-axis key to fall back to when the current Y choice
                is not in ``keys``, or ``None`` for no Y default.
        """
        if not keys:
            return

        for sel in (self._x_selector, self._y_selector):
            prev = sel.currentText()
            sel.blockSignals(True)
            sel.clear()
            sel.addItems(keys)
            sel.setCurrentText(prev if prev in keys else keys[0])
            sel.blockSignals(False)

        # Apply sensible defaults only when the current selection is blank/unknown.
        if self._x_selector.currentText() not in keys:
            self._x_selector.setCurrentText(default_x)
        if (
            default_y is not None
            and self._y_selector.currentText() not in keys
            and default_y in keys
        ):
            self._y_selector.setCurrentText(default_y)

    def set_available_image_blocks(self, blocks: Mapping[str, str]) -> None:
        """Set the image blocks the Image kind can draw, preserving a still-valid choice.

        With no blocks the Image kind is disabled and a panel in Image mode
        falls back to XY, so a procedure that records no frames never shows
        an empty image panel.

        Args:
            blocks: ``{block_name: pixel_unit}`` — the selected procedure's
                ``live_plot_image_blocks()``.
        """
        self._image_blocks = dict(blocks)
        prev = self._image_selector.currentText()
        self._image_selector.blockSignals(True)
        self._image_selector.clear()
        self._image_selector.addItems(list(self._image_blocks))
        if prev in self._image_blocks:
            self._image_selector.setCurrentText(prev)
        self._image_selector.blockSignals(False)
        self._refresh_kind_availability()
        self._redraw()

    def selected_kind(self) -> str:
        """Return the panel's current kind, ``"xy"`` or ``"image"``."""
        return str(self._kind_selector.currentData())

    def set_kind(self, kind: str) -> None:
        """Switch the panel's kind; a no-op for an unknown or unavailable kind.

        Args:
            kind: ``"xy"`` or ``"image"``.
        """
        index = self._kind_selector.findData(kind)
        if index < 0 or not self._kind_item_enabled(index):
            return
        self._kind_selector.setCurrentIndex(index)

    def set_available_loop_labels(
        self, label_maps: tuple[dict[int, str] | None, dict[int, str] | None]
    ) -> None:
        """Show/hide/enable the two Loop selectors, one per reading-loop slot.

        Per slot: ``None`` means the selection offers no loop at all — the
        selector is hidden. ``{}`` means a loop is possible but that slot is
        off/static/invalid — visible but disabled. With two or more entries
        the selector is enabled: each item shows the display text (e.g.
        ``"A1 = Mux-Ch1"``) and carries the 0-based axis index as its item
        data, which ``_redraw`` uses to index into that slot's grid axis.

        Args:
            label_maps: One ordered ``{axis_index: display_text}`` map (or
                ``{}`` / ``None``) per loop slot, in slot order.
        """
        for label_widget, selector, labels in zip(
            self._loop_labels_ui, self._loop_selectors, label_maps
        ):
            if labels is None:
                label_widget.setVisible(False)
                selector.setVisible(False)
                continue
            label_widget.setVisible(True)
            selector.setVisible(True)
            if len(labels) < 2:
                selector.setEnabled(False)
                selector.blockSignals(True)
                selector.clear()
                selector.blockSignals(False)
            else:
                selector.setEnabled(True)
                prev = selector.currentData()
                selector.blockSignals(True)
                selector.clear()
                for index, display in labels.items():
                    selector.addItem(display, index)
                found = selector.findData(prev)
                selector.setCurrentIndex(found if found >= 0 else 0)
                selector.blockSignals(False)
        self._redraw()

    def redraw(self, datapoints: list[dict]) -> None:
        """Store the datapoint history and redraw from it.

        Args:
            datapoints: Full datapoint history (each entry an enriched dict).
        """
        self._datapoints = datapoints
        # The Point selector ranges over the points so far; 0 stays "latest".
        self._point_selector.blockSignals(True)
        self._point_selector.setMaximum(len(datapoints))
        self._point_selector.blockSignals(False)
        self._redraw()

    def clear(self) -> None:
        """Empty the curve and the image (does not touch the selectors)."""
        self._datapoints = []
        self._curve.setData([], [])
        self._image_view.clear()
        self._point_selector.blockSignals(True)
        self._point_selector.setRange(0, 0)
        self._point_selector.blockSignals(False)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _kind_item_enabled(self, index: int) -> bool:
        item = self._kind_selector.model().item(index)  # type: ignore[attr-defined]
        return bool(item.isEnabled())

    def _refresh_kind_availability(self) -> None:
        """Enable the Image kind only while there is an image block to draw."""
        index = self._kind_selector.findData("image")
        item = self._kind_selector.model().item(index)  # type: ignore[attr-defined]
        item.setEnabled(bool(self._image_blocks))
        if not self._image_blocks and self.selected_kind() == "image":
            self._kind_selector.setCurrentIndex(self._kind_selector.findData("xy"))

    def _on_kind_changed(self, _index: int) -> None:
        self._apply_kind_visibility()
        self._redraw()

    def _apply_kind_visibility(self) -> None:
        """Show the selectors and the view of the current kind only."""
        image = self.selected_kind() == "image"
        for widget in (self._x_label, self._x_selector, self._y_label, self._y_selector):
            widget.setVisible(not image)
        for widget in (
            self._image_label, self._image_selector, self._point_label, self._point_selector,
        ):
            widget.setVisible(image)
        self._stack.setCurrentWidget(self._image_view if image else self._plot_widget)

    def _loop_indices(self) -> tuple[int, int, list[str]]:
        """Return the selected index per loop slot, and the display text of any pick.

        Returns:
            ``(i1, i2, qualifiers)`` — each index 0 (the trivial index into
            a length-1 axis) when that slot is inactive or not picked.
        """
        indices: list[int] = []
        qualifiers: list[str] = []
        for selector in self._loop_selectors:
            if (
                selector.isVisible()
                and selector.isEnabled()
                and selector.currentData() is not None
            ):
                indices.append(int(selector.currentData()))
                qualifiers.append(selector.currentText())
            else:
                indices.append(0)
        return indices[0], indices[1], qualifiers

    def _redraw(self) -> None:
        """Redraw the current kind from the stored datapoint history."""
        if self.selected_kind() == "image":
            self._redraw_image()
        else:
            self._redraw_xy()

    def _redraw_image(self) -> None:
        """Draw the selected image block's frame for the selected point and reading."""
        block = self._image_selector.currentText()
        unit = self._image_blocks.get(block, "")
        i1, i2, _qualifiers = self._loop_indices()
        self._image_view.set_labels("column (px)", "row (px)", unit)
        if not block or not self._datapoints:
            self._image_view.clear()
            return
        point = self._point_selector.value()
        datapoint = self._datapoints[-1] if point == 0 else self._datapoints[point - 1]
        raw = datapoint.get(block)
        if raw is None:
            self._image_view.clear()
            return
        try:
            frame = select_frame(np.asarray(raw, dtype=np.float64), i1, i2)
        except (IndexError, ValueError):
            logger.debug("LivePlotPanel: no frame for %r at loop (%d, %d)", block, i1, i2)
            self._image_view.clear()
            return
        self._image_view.set_frame(frame)

    def _redraw_xy(self) -> None:
        """Redraw the curve from the stored datapoint history and relabel axes."""
        x_key = self._x_selector.currentText()
        y_key = self._y_selector.currentText()
        i1, i2, qualifiers = self._loop_indices()

        def _lookup(dp: dict, key: str):
            # Every measurement column is a real (n_loop1, n_loop2) grid;
            # sweep-only columns (unix_time, system state, sweep axis) are
            # plain scalars with no grid to index.
            raw = dp.get(key)
            if raw is None or not hasattr(raw, "__len__"):
                return raw
            try:
                return raw[i1][i2]
            except (IndexError, TypeError):
                return None

        xs = []
        ys = []
        for dp in self._datapoints:
            xs.append(_to_float(_lookup(dp, x_key)))
            ys.append(_to_float(_lookup(dp, y_key)))

        self._curve.setData(xs, ys)
        x_label = x_key.replace("_", " ")
        y_label = y_key.replace("_", " ")
        qualifier_text = ", ".join(qualifiers)
        if qualifier_text:
            x_label += f" ({qualifier_text})"
            y_label += f" ({qualifier_text})"
        self._plot_widget.setLabel("bottom", x_label)
        self._plot_widget.setLabel("left", y_label)
