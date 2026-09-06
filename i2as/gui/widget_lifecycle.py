"""Widget lifetimes: the window-liveness and card-retirement standards.

Two rules about *when* a Qt widget's C++ object is destroyed. Both exist
because PyQt destroys the C++ object as soon as the last Python reference to
a parentless widget goes away, and Python's cyclic garbage collector chooses
that moment on its own — including from inside an allocation made by Qt's own
paint path.

Window liveness
    A top-level window holds a strong Python reference to itself from
    construction until it is closed, instead of relying on whoever built it
    to keep one. A ``MonitorWindow`` whose only remaining reference was a
    reference cycle used to be destroyed by an arbitrary generational GC pass
    — one that could (and did) land in the middle of that same window's
    ``paintEvent``: Qt logged "Cannot destroy paint device that is being
    painted", pyqtgraph then painted an ``AxisItem`` whose C++ object had just
    been freed ("wrapped C/C++ object of type AxisItem has been deleted"), and
    the process segfaulted on the freed item. A window that keeps itself alive
    can only be destroyed at a point the code picked: its ``closeEvent``.

Card retirement
    A widget swapped out of a live layout is hidden and taken out of that
    layout, and any pyqtgraph plot it owns is closed, BEFORE ``deleteLater()``
    is scheduled. A widget merely dropped from a layout stays a visible child
    of its parent and keeps painting — over its replacement — until the
    deferred delete is delivered, which is one full event-loop iteration away
    at best, and never while no event loop is running at all. The retired
    widget keeps its Qt parent on purpose: a card is normally retired from
    inside a signal emitted by one of its own children (the Disconnect button
    on the card being swapped away), so its destruction must stay deferred.
    Unparenting it would hand ownership back to Python and let the C++ object
    be destroyed the moment the last Python reference goes — with that
    child's signal still on the stack.

Both helpers are deliberately Qt-only: they take widgets, not I2AS
objects, so any GUI module can use them without new layer dependencies.
"""

from __future__ import annotations

import logging

import pyqtgraph as pg
from PyQt6.QtWidgets import QLayout, QWidget

logger = logging.getLogger(__name__)

# Strong references to the shown top-level windows, in creation order. The
# only thing that removes an entry is release_window() (i.e. a closeEvent).
_HELD_WINDOWS: list[QWidget] = []


def hold_window(window: QWidget) -> None:
    """Hold a strong Python reference to one top-level window.

    The window-liveness half of this module's standard: call it at the end of
    a top-level window's ``__init__`` and pair it with
    :func:`release_window` in that window's ``closeEvent``. Holding the
    reference here means the window's lifetime no longer depends on its
    creator keeping a local variable alive, so no garbage-collection pass can
    destroy it (and the pyqtgraph scenes it owns) while it is on screen.

    Args:
        window: The top-level window to hold. Holding one twice is a no-op.
    """
    if any(held is window for held in _HELD_WINDOWS):
        return
    _HELD_WINDOWS.append(window)
    logger.debug(
        "widget_lifecycle: holding %s (%d window(s) held)",
        type(window).__name__,
        len(_HELD_WINDOWS),
    )


def release_window(window: QWidget) -> None:
    """Drop the strong reference held for one top-level window.

    Called from the window's ``closeEvent`` once the close is accepted: a
    closed window is hidden, so nothing paints it any more and Python is free
    to collect it whenever it likes.

    Args:
        window: The window to release. Releasing one that is not held is a
            no-op.
    """
    for index, held in enumerate(_HELD_WINDOWS):
        if held is window:
            del _HELD_WINDOWS[index]
            logger.debug(
                "widget_lifecycle: released %s (%d window(s) held)",
                type(window).__name__,
                len(_HELD_WINDOWS),
            )
            return


def held_windows() -> tuple[QWidget, ...]:
    """Return the currently held top-level windows.

    Returns:
        The held windows in creation order. Tests assert on this to check
        that a window is held while open and released on close; production
        code has no reason to read it.
    """
    return tuple(_HELD_WINDOWS)


def retire_widget(widget: QWidget, layout: QLayout | None = None) -> None:
    """Retire one widget out of a live layout, in the order Qt needs.

    The card-retirement half of this module's standard, and the only
    supported way to remove a card (or any other widget) from a layout that
    stays on screen:

    1. hide it, so Qt schedules no further paints for it;
    2. take it out of ``layout``, so the layout stops sizing it;
    3. close every pyqtgraph ``PlotWidget`` it owns, so no graphics item is
       left waiting to be painted from a scene that is about to go away;
    4. only then schedule ``deleteLater()``.

    The widget keeps its Qt parent (see this module's docstring): retirement
    usually runs inside a signal emitted by one of the widget's own children,
    and the deferred delete is what makes that safe.

    Args:
        widget: The widget to retire. It must not be used again afterwards.
        layout: The layout the widget currently sits in, if any. Passing the
            layout is what makes step 2 possible; ``None`` skips it (for a
            widget already removed or never laid out).
    """
    widget.hide()
    if layout is not None:
        layout.removeWidget(widget)
    for plot_widget in widget.findChildren(pg.PlotWidget):
        _close_plot(plot_widget)
    if isinstance(widget, pg.PlotWidget):
        _close_plot(widget)
    widget.deleteLater()


def _close_plot(plot_widget: pg.PlotWidget) -> None:
    """Tear down one pyqtgraph plot so nothing in it can be painted again.

    ``PlotWidget.close()`` clears the plot item, empties the scene and drops
    the central item; without it the scene's items outlive the widget's
    removal from the layout and are painted again on the next event-loop
    turn.

    Args:
        plot_widget: The plot to close. Failures are logged, never raised:
            retirement must not leave a half-retired card behind.
    """
    try:
        plot_widget.plotItem.clear()
        plot_widget.close()
    except RuntimeError as exc:  # already-deleted C++ object
        logger.debug("widget_lifecycle: plot already gone (%s)", exc)
