"""app_settings — QSettings factory used as a test seam, and the settings-file store.

Dependency seam: a single indirection point (this factory) that tests
monkeypatch so GUI tests never touch the real registry. Windows import the
*module* and call ``app_settings.get_settings()`` rather than importing the
function directly, so that ``monkeypatch.setattr(app_settings, "get_settings",
...)`` is seen at every call site.

QSettings keeps machine chrome (window geometry, the active config, who is
logged in). What a person CHOOSES about the installation — connections,
analysis — lives in the general settings file, reached through
``config_store()``; see ``i2as.session.app_config``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QSettings, QStandardPaths

from i2as.core.paths import user_state_dir
from i2as.session.app_config import (
    AppConfigStore,
    ConnectionSettings,
    app_config_path,
    legacy_analysis_block,
    read_app_config_file,
)

logger = logging.getLogger(__name__)

_ORGANISATION = "I2AS"
_APPLICATION = "I2AS"

_SESSION_FILENAME = "last_session.json"
_SESSIONS_SUBDIR = "sessions"
_ACTIVE_CONFIG_NAME_KEY = "ActiveConfig/name"
_ACTIVE_CONFIG_SOURCE_KEY = "ActiveConfig/source"
_CURRENT_USER_KEY = "CurrentUser/user_id"
# Legacy QSettings keys: read once, to migrate them into the settings file.
_GATEWAY_ENABLED_KEY = "Gateway/enabled"
_GATEWAY_MAX_ROLE_KEY = "Gateway/max_role"
_REMOTE_ENABLED_KEY = "RemoteAccess/enabled"
_REMOTE_HOST_KEY = "RemoteAccess/host"
_REMOTE_PORT_KEY = "RemoteAccess/port"
_REMOTE_PUBLIC_URL_KEY = "RemoteAccess/public_url"


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


# ── The general settings file ────────────────────────────────────────────
#
# Connections (and analysis) live in the general settings file
# (``i2as.session.app_config``), not in QSettings. The functions below keep
# their old names so every caller is unchanged; they now read and write the
# one shared ``AppConfigStore``.

#: The process-wide store, created on first use. Tests reset it to ``None``.
_CONFIG_STORE: AppConfigStore | None = None


def config_store() -> AppConfigStore:
    """Return the process-wide general settings store, creating it once.

    On creation, a file with no ``connections`` section takes the values the
    Connections dialog used to keep in QSettings, and a file with no
    ``analysis`` section the block ``eln-settings.json`` used to keep; when
    either was migrated the file is written at once, so the copy is
    permanent and the old places are never read again.

    Returns:
        The store every GUI page, the ELN publisher and the analysis runner
        share.
    """
    global _CONFIG_STORE
    if _CONFIG_STORE is None:
        path = app_config_path()
        raw = read_app_config_file(path) or {}
        store = AppConfigStore(path)
        config = store.current
        migrated = "analysis" not in raw and legacy_analysis_block() is not None
        if "connections" not in raw:
            legacy = _legacy_connections()
            if legacy is not None:
                config = replace(config, connections=legacy)
                migrated = True
        if migrated:
            try:
                store.save(config)
            except OSError:
                logger.exception("Could not write the migrated settings to %s", path)
        _CONFIG_STORE = store
    return _CONFIG_STORE


def _legacy_connections() -> ConnectionSettings | None:
    """Return the connection values QSettings held before the settings file, or ``None``."""
    settings = get_settings()
    keys = (
        _GATEWAY_ENABLED_KEY,
        _GATEWAY_MAX_ROLE_KEY,
        _REMOTE_ENABLED_KEY,
        _REMOTE_HOST_KEY,
        _REMOTE_PORT_KEY,
        _REMOTE_PUBLIC_URL_KEY,
    )
    if not any(settings.contains(key) for key in keys):
        return None
    defaults = ConnectionSettings()
    enabled = (
        bool(settings.value(_GATEWAY_ENABLED_KEY, False, type=bool))
        if settings.contains(_GATEWAY_ENABLED_KEY)
        else None
    )
    try:
        port = int(settings.value(_REMOTE_PORT_KEY, defaults.remote_port))
    except (TypeError, ValueError):
        port = defaults.remote_port
    return ConnectionSettings(
        gateway_enabled=enabled,
        gateway_max_role=str(settings.value(_GATEWAY_MAX_ROLE_KEY) or ""),
        remote_enabled=bool(settings.value(_REMOTE_ENABLED_KEY, False, type=bool)),
        remote_host=str(settings.value(_REMOTE_HOST_KEY) or defaults.remote_host),
        remote_port=port if 1 <= port <= 65535 else defaults.remote_port,
        remote_public_url=str(settings.value(_REMOTE_PUBLIC_URL_KEY) or ""),
    )


def _update_connections(**changes: object) -> None:
    """Write some connection fields to the settings file.

    Args:
        **changes: ``ConnectionSettings`` field names and their new values.
    """
    store = config_store()
    config = store.current
    store.save(replace(config, connections=replace(config.connections, **changes)))


def gateway_enabled() -> bool | None:
    """Return whether the Settings dialog last left the Gateway server on.

    This is what lets an operator turn the Agent gateway on or off from the
    GUI and have that choice survive a restart, independent of
    ``monitor.yaml``'s ``gateway_server`` flag, which only seeds the very
    first launch.

    Returns:
        ``True``/``False`` once the Settings dialog has set it; ``None`` when
        it never has, so the caller falls back to the active config's own
        default.
    """
    return config_store().connections().gateway_enabled


def set_gateway_enabled(enabled: bool) -> None:
    """Persist whether the Gateway server should listen on next launch.

    Args:
        enabled: The Connections page's on/off toggle.
    """
    _update_connections(gateway_enabled=bool(enabled))


def gateway_max_role() -> str | None:
    """Return the Gateway role ceiling the Settings dialog last set.

    Returns:
        The ``Role`` value string (e.g. ``"session"``), or ``None`` when the
        dialog has never set one — the caller falls back to the active
        config's ``gateway_max_role``.
    """
    return config_store().connections().gateway_max_role or None


def set_gateway_max_role(role: str) -> None:
    """Persist the Gateway role ceiling for next launch.

    Args:
        role: A ``Role`` value string, as chosen on the Connections page.
    """
    _update_connections(gateway_max_role=str(role))


# ── Remote access: the HTTP MCP endpoint ─────────────────────────────────


def access_keys_path() -> Path:
    """Return the file the HTTP endpoint's access keys live in.

    ``%LOCALAPPDATA%/I2AS/mcp_keys.json`` on Windows — the per-installation
    state root beside the gateway descriptor, because a key is a credential
    for THIS machine's app and must not travel with a synced profile.
    Monkeypatchable test seam.

    Returns:
        The ``Path`` (may not exist yet; the store creates it on first key).
    """
    return user_state_dir() / "mcp_keys.json"


def remote_access_enabled() -> bool:
    """Return whether the Settings dialog last left the HTTP endpoint on.

    Returns:
        ``True`` once the dialog has switched it on; ``False`` otherwise —
        remote access is never on by default.
    """
    return config_store().connections().remote_enabled


def set_remote_access_enabled(enabled: bool) -> None:
    """Persist whether the HTTP endpoint should listen on next launch.

    Args:
        enabled: The Connections page's toggle.
    """
    _update_connections(remote_enabled=bool(enabled))


def remote_access_host() -> str:
    """Return the address the HTTP endpoint binds; the loopback address by default."""
    return config_store().connections().remote_host


def set_remote_access_host(host: str) -> None:
    """Persist the bind address for the HTTP endpoint.

    Args:
        host: ``"127.0.0.1"`` or ``"0.0.0.0"``.
    """
    _update_connections(remote_host=str(host))


def remote_access_port() -> int:
    """Return the HTTP endpoint's port; ``8765`` until the dialog sets one."""
    return config_store().connections().remote_port


def set_remote_access_port(port: int) -> None:
    """Persist the HTTP endpoint's port.

    Args:
        port: The TCP port.
    """
    _update_connections(remote_port=int(port))


def remote_access_public_url() -> str | None:
    """Return the public URL the operator published for the endpoint, if any."""
    return config_store().connections().remote_public_url or None


def set_remote_access_public_url(url: str | None) -> None:
    """Persist (or clear) the public URL the operator's tunnel hands out.

    Args:
        url: The externally reachable ``https://…`` address, or ``None``.
    """
    _update_connections(remote_public_url=(url or "").strip())
