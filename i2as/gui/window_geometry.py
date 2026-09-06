"""Shared window-geometry persistence helpers for the GUI windows."""

from __future__ import annotations

from PyQt6.QtWidgets import QApplication, QWidget

from i2as.gui import app_settings  # import the module (not the function) so tests can monkeypatch the factory


def geometry_on_screen(window: QWidget) -> bool:
    """Return True if the window frame overlaps an attached screen enough to see.

    ``restoreGeometry`` reports success even when it places the window on a
    screen that no longer exists, so this guards against an invisible window:
    it requires at least a 100x100 overlap with some screen's available area.

    Args:
        window: The window whose frame geometry is checked.

    Returns:
        True if at least one attached screen shows a 100x100 region of the
        window.
    """
    frame = window.frameGeometry()
    for screen in QApplication.screens():
        overlap = screen.availableGeometry().intersected(frame)
        if overlap.width() >= 100 and overlap.height() >= 100:
            return True
    return False


def restore_or_center(window: QWidget, settings_key: str, fraction: float) -> None:
    """Restore the saved window geometry, or size to a screen fraction, centered.

    Geometry is persisted with ``QSettings``, which on Windows is backed by
    the registry (``HKCU\\Software\\I2AS\\I2AS``). No saved geometry,
    a restore failure, or geometry that landed off-screen (e.g. saved on a
    monitor that is no longer attached — the usual cause of a window that
    "does not appear") all fall back to a centered default sized to
    ``fraction`` of the primary screen's available area.

    Args:
        window: The window to move/resize.
        settings_key: The QSettings key holding the saved geometry blob.
        fraction: Fallback size as a fraction of the available screen area
            (e.g. ``0.9``).
    """
    settings = app_settings.get_settings()
    saved = settings.value(settings_key)
    if saved is not None and window.restoreGeometry(saved) and geometry_on_screen(window):
        return
    screen = QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        width = int(available.width() * fraction)
        height = int(available.height() * fraction)
        window.resize(width, height)
        window.move(
            available.x() + (available.width() - width) // 2,
            available.y() + (available.height() - height) // 2,
        )


def save_geometry(window: QWidget, settings_key: str) -> None:
    """Persist the window's current geometry blob under ``settings_key``.

    Args:
        window: The window whose geometry is saved.
        settings_key: The QSettings key to write.
    """
    app_settings.get_settings().setValue(settings_key, window.saveGeometry())
