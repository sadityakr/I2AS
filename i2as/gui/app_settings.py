"""app_settings — QSettings factory used as a test seam.

Dependency seam: a single indirection point (this factory) that tests
monkeypatch so GUI tests never touch the real registry. Windows import the
*module* and call ``app_settings.get_settings()`` rather than importing the
function directly, so that ``monkeypatch.setattr(app_settings, "get_settings",
...)`` is seen at every call site.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings, QStandardPaths

_ORGANISATION = "I2AS"
_APPLICATION = "I2AS"

_SESSION_FILENAME = "last_session.json"
_SESSIONS_SUBDIR = "sessions"
_ACTIVE_CONFIG_NAME_KEY = "ActiveConfig/name"
_ACTIVE_CONFIG_SOURCE_KEY = "ActiveConfig/source"
_CURRENT_USER_KEY = "CurrentUser/user_id"


def get_settings() -> QSettings:
    """Return the application's QSettings store.

    Returns:
        A ``QSettings`` scoped to the I2AS organisation and application. In
        production this is the platform-native store (the Windows registry);
        GUI tests monkeypatch this function to return an INI-file store instead.
    """
    return QSettings(_ORGANISATION, _APPLICATION)


def autosave_file_path(user_id: str | None = None) -> Path:
    """Return the path to a persistent form-autosave JSON file.

    The file lives in the platform per-installation application-data
    directory (``%APPDATA%/I2AS/`` on Windows), separate from both the
    registry and the user's measurement data directory. This is the second
    persistence tier: ``get_settings()`` holds machine-specific window/dock
    *chrome*, while this file holds portable session *content* (sample
    metadata, procedure params, run queue). Like ``get_settings``, GUI tests
    monkeypatch this function to redirect it into a throwaway directory.

    Args:
        user_id: When given (someone is logged in — see ``current_user_id``),
            returns that person's own autosave file
            (``%APPDATA%/I2AS/sessions/<user_id>.json``), so switching
            users switches what's remembered instead of one person's fields
            overwriting another's. ``None`` (nobody logged in yet, or a
            caller that predates the login feature) returns the original
            shared ``last_session.json``.

    Returns:
        The absolute ``Path`` of the session JSON file. The parent directory is
        not guaranteed to exist yet; ``form_autosave.save`` creates it on first write.
    """
    base = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation
    )
    if user_id:
        return Path(base) / _SESSIONS_SUBDIR / f"{user_id}.json"
    return Path(base) / _SESSION_FILENAME


def user_config_dir() -> Path:
    """Return the directory holding the user's editable config copies.

    ``%APPDATA%/I2AS/configs`` on Windows. Separate from the shipped,
    read-only configs in the repo. Monkeypatchable test seam.

    Returns:
        The ``Path`` of the user config directory (may not exist yet).
    """
    base = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation
    )
    return Path(base) / "configs"


def shipped_config_dir() -> Path:
    """Return the repo's read-only shipped-config directory (``i2as/configs``).

    Resolved relative to the package so it is independent of the current working
    directory.

    Returns:
        The ``Path`` of the shipped config directory.
    """
    return Path(__file__).resolve().parents[1] / "configs"


def config_active() -> tuple[str, str] | None:
    """Return the saved active config's ``(name, source)`` identity, or None.

    The active config is machine-level (which cryostat this install controls),
    so it lives in QSettings rather than the per-session JSON file. Identity
    (name + source) is stored rather than a resolved absolute path, so the
    saved selection stays valid across clones/worktrees: the caller re-derives
    the actual directory via ``shipped_config_dir()``/``user_config_dir()`` at
    load time instead of trusting a path that may no longer exist.

    Returns:
        A ``(name, source)`` tuple (``source`` is ``"shipped"`` or ``"user"``),
        or None when no config has been selected yet.
    """
    settings = get_settings()
    name = settings.value(_ACTIVE_CONFIG_NAME_KEY)
    source = settings.value(_ACTIVE_CONFIG_SOURCE_KEY)
    if not name or not source:
        return None
    return (str(name), str(source))


def set_config_active(name: str, source: str) -> None:
    """Persist a config's ``(name, source)`` identity as active for next launch.

    Args:
        name: The config's directory name (``ConfigEntry.name``).
        source: ``"shipped"`` or ``"user"`` (``ConfigEntry.source``).
    """
    settings = get_settings()
    settings.setValue(_ACTIVE_CONFIG_NAME_KEY, name)
    settings.setValue(_ACTIVE_CONFIG_SOURCE_KEY, source)


def current_user_id() -> str | None:
    """Return the roster id of whoever is currently "logged in", or None.

    Machine-level like the active config: persists across restarts until the
    User menu's "Log in as…" switches it. Identity only — no password, no
    session token; the roster (``i2as.session.store.UserRoster``) is the
    source of truth for whether the id still exists.

    Returns:
        The roster ``user_id``, or ``None`` when nobody has logged in yet.
    """
    value = get_settings().value(_CURRENT_USER_KEY)
    return str(value) if value else None


def set_current_user_id(user_id: str | None) -> None:
    """Persist (or clear) who is currently logged in.

    Args:
        user_id: The roster id to remember, or ``None`` to log out (falls
            back to the shared ``last_session.json`` on next read).
    """
    settings = get_settings()
    if user_id:
        settings.setValue(_CURRENT_USER_KEY, user_id)
    else:
        settings.remove(_CURRENT_USER_KEY)
