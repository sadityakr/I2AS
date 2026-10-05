"""ImageView — the one 2-D renderer every image-like plot panel shares.

A themed ``pyqtgraph`` plot holding an ``ImageItem`` and a colour bar. It
knows nothing about where a frame comes from: the Monitor's image panel
hands it a live ``@monitored(kind="image")`` frame, the waterfall panel a
stack of ``kind="trace"`` rows, the Procedure window's image panel a frame
out of a run's image block. Keeping the rendering in one widget is what
makes "an image plot" one thing in the GUI rather than three.

Frames are given row-major — ``frame[row, col]``, the way every image block
and every ``kind="image"`` field is declared ``(height_px, width_px)`` — and
drawn with row 0 at the top, as a camera frame is read.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget

# The colour map every image panel uses: perceptually uniform, readable in
# both themes and for the common colour-vision deficiencies.
_COLORMAP = "viridis"


class ImageView(QWidget):
    """A 2-D image plot with a colour bar, auto-levelled to each frame.

    Args:
        object_name: objectName for the inner ``GraphicsLayoutWidget``, so
            tests and ``findChild`` can reach it.
        invert_y: Draw row 0 at the top (``True``, the camera convention)
            or at the bottom (``False``, e.g. a waterfall whose newest row
            is drawn last).
        lock_aspect: Keep square pixels (``True``, a camera frame) or let
            each axis stretch to the panel (``False``, a waterfall whose
            axes have unrelated units).
        parent: Optional Qt parent widget.
    """

    def __init__(
        self,
        object_name: str = "",
        invert_y: bool = True,
        lock_aspect: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._canvas = pg.GraphicsLayoutWidget()
        if object_name:
            self._canvas.setObjectName(object_name)
        self._canvas.setMinimumHeight(140)
        layout.addWidget(self._canvas)

        self._plot = self._canvas.addPlot()
        self._plot.setAspectLocked(lock_aspect)
        self._plot.invertY(invert_y)
        self._image = pg.ImageItem(axisOrder="row-major")
        self._plot.addItem(self._image)
        self._colorbar = pg.ColorBarItem(
            colorMap=pg.colormap.get(_COLORMAP), interactive=False, width=12
        )
        self._colorbar.setImageItem(self._image, insert_in=self._plot)
        self._frame: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frame(
        self,
        frame: np.ndarray,
        rect: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Draw one frame, re-levelling the colour bar to its finite range.

        Args:
            frame: A 2-D array, ``frame[row, col]``. NaN pixels are drawn
                transparent and ignored by the levels.
            rect: ``(x, y, width, height)`` in data units the frame spans,
                for physical axes; ``None`` spans pixel indices.

        Raises:
            ValueError: If ``frame`` is not two-dimensional.
        """
        array = np.asarray(frame, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"ImageView needs a 2-D frame, got shape {array.shape}")
        self._frame = array
        self._image.setImage(array, autoLevels=False)
        finite = array[np.isfinite(array)]
        if finite.size:
            low, high = float(finite.min()), float(finite.max())
            if high <= low:
                high = low + 1.0
            self._colorbar.setLevels((low, high))
        if rect is not None:
            self._image.setRect(*rect)
        else:
            self._image.setRect(0.0, 0.0, float(array.shape[1]), float(array.shape[0]))

    def clear(self) -> None:
        """Remove the frame (the axes and colour bar stay)."""
        self._frame = None
        self._image.clear()

    def frame(self) -> np.ndarray | None:
        """Return the frame currently drawn, or ``None``.

        Returns:
            The last array given to ``set_frame()``, or ``None`` after
            ``clear()`` / before the first frame.
        """
        return self._frame

    def set_labels(self, bottom: str = "", left: str = "", colorbar: str = "") -> None:
        """Label the axes and the colour bar.

        Args:
            bottom: X-axis label.
            left: Y-axis label.
            colorbar: Colour-bar label — typically the pixel unit.
        """
        self._plot.setLabel("bottom", bottom)
        self._plot.setLabel("left", left)
        self._colorbar.setLabel("right", colorbar)
