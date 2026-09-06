"""I2AS installation-path resolution.

Machine-local directories — where logs go on *this* installation, where a
user's own settings live — are a deployment property, not a source-tree one,
and resolving them is neither logging's job nor any single caller's. Keeping
every such rule here means a caller asks for a directory instead of
reimplementing the precedence, and the rules stay readable side by side.

The **per-user path standard**: exactly two per-user roots exist, and every
per-user file in I2AS hangs off one of them —

* ``user_config_dir()`` — what the user CHOOSES and would carry to another
  machine: editable config copies, ELN settings.
  ``%APPDATA%\\I2AS`` on Windows; ``$XDG_CONFIG_HOME/i2as``, else
  ``~/.config/i2as``, elsewhere.
* ``user_state_dir()`` — what the application ACCUMULATES on this machine and
  the user would not miss: logs, the trend-history store, the gateway
  descriptor. ``%LOCALAPPDATA%\\I2AS`` on Windows;
  ``$XDG_STATE_HOME/i2as``, else ``~/.local/state/i2as``, elsewhere.

On Windows with the platform variable unset, both fall back to the XDG shape
under the home directory rather than inventing a third location. A module
that cannot import this one (``i2as.mcp.client``, held to stdlib and
``core.events`` by contract C21) carries a copy of the state-dir rule and
says so beside it.

This module is import-linter contract C1 foundation: it must import nothing
else from the ``i2as`` package, stdlib only.
"""

from __future__ import annotations

import os
from pathlib import Path

_APP_DIR_WINDOWS = "I2AS"
_APP_DIR_XDG = "i2as"


def _xdg_dir(env_var: str, *default_parts: str) -> Path:
    """Resolve one XDG base directory for this application.

    Args:
        env_var: The XDG variable to honour, e.g. ``"XDG_CONFIG_HOME"``.
        default_parts: The path under the home directory the specification
            names as that variable's default, e.g. ``(".config",)``.

    Returns:
        ``$<env_var>/i2as`` when the variable is set and non-empty, else
        ``~/<default_parts>/i2as``.
    """
    base = os.environ.get(env_var)
    if base:
        return Path(base) / _APP_DIR_XDG
    return Path.home().joinpath(*default_parts) / _APP_DIR_XDG


def user_config_dir() -> Path:
    """Resolve the per-user CONFIG root without creating it.

    The home of what the user chooses and would carry to another machine —
    editable config copies, ELN settings. See the module docstring's
    per-user path standard.

    Returns:
        ``%APPDATA%\\I2AS`` on Windows (``os.name == "nt"``) when
        ``APPDATA`` is set; otherwise ``$XDG_CONFIG_HOME/i2as``, else
        ``~/.config/i2as``. Not guaranteed to exist; nothing is created.
    """
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / _APP_DIR_WINDOWS
    return _xdg_dir("XDG_CONFIG_HOME", ".config")


def user_state_dir() -> Path:
    """Resolve the per-user STATE root without creating it.

    The home of what the application accumulates on this machine — logs,
    the trend-history store, the gateway descriptor. See the module
    docstring's per-user path standard.

    Returns:
        ``%LOCALAPPDATA%\\I2AS`` on Windows (``os.name == "nt"``) when
        ``LOCALAPPDATA`` is set; otherwise ``$XDG_STATE_HOME/i2as``, else
        ``~/.local/state/i2as``. Not guaranteed to exist; nothing is
        created.
    """
    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / _APP_DIR_WINDOWS
    return _xdg_dir("XDG_STATE_HOME", ".local", "state")


def log_directory() -> Path:
    """Resolve the I2AS log directory without creating it.

    Precedence:

    1. ``I2AS_LOG_DIR`` environment variable, if set and non-empty.
    2. ``user_state_dir() / "logs"`` — ``%LOCALAPPDATA%\\I2AS\\logs`` on
       Windows, ``~/.local/state/i2as/logs`` (or its ``XDG_STATE_HOME``
       form) elsewhere and as the Windows fallback.

    This is a pure function: it only resolves and returns a path, it never
    creates the directory or any file in it. ``setup_logging()`` is
    responsible for the ``mkdir(parents=True, exist_ok=True)``.

    No migration of existing log files is performed when the resolved
    location changes (e.g. moving off ``I2AS_LOG_DIR`` or between
    machines). Logs are disposable operational telemetry, not data of
    record: the new location simply starts empty. Do not write a migrator
    for this.

    Returns:
        The resolved log directory path (not guaranteed to exist).
    """
    env_dir = os.environ.get("I2AS_LOG_DIR")
    if env_dir:
        return Path(env_dir)
    return user_state_dir() / "logs"


def _app_config_path() -> Path:
    """Resolve the machine-level settings file's path (not guaranteed to exist).

    Returns:
        ``%ProgramData%\\I2AS\\App-config.yaml`` on Windows
        (``os.name == "nt"``), or ``/etc/i2as/App-config.yaml`` on other
        platforms.
    """
    if os.name == "nt":
        program_data = os.environ.get("ProgramData", r"C:\ProgramData")
        return Path(program_data) / "I2AS" / "App-config.yaml"
    return Path("/etc/i2as/App-config.yaml")


def _read_measurement_root_setting(config_path: Path) -> str | None:
    """Read the ``measurement_root`` key out of the machine settings file.

    ``App-config.yaml`` has exactly one key for now, so this is a tiny
    single-key line parser rather than a general YAML parser: it does not
    pull in a PyYAML dependency for a stdlib-only contract C1 module (see
    the module docstring). It reads the first ``measurement_root: <value>``
    line, splits on the first colon, and strips surrounding whitespace and
    matching quotes from the value. Comments (``#``) and any other key are
    ignored, so the format stays forward-compatible with the file growing a
    second key later.

    Args:
        config_path: Path to the settings file to read.

    Returns:
        The value of ``measurement_root``, or ``None`` if the file is
        missing, unreadable, or has no such key.
    """
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition(":")
        if not sep or key.strip() != "measurement_root":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        return value

    return None


def measurement_root() -> Path:
    """Resolve the fixed I2AS measurement root without creating it.

    The measurement root is a machine-level, admin-set value — deliberately
    not GUI-editable or per-user — so it stays fixed across a station's
    lifetime rather than drifting via live settings.

    Precedence:

    1. ``I2AS_MEASUREMENT_ROOT`` environment variable, if set and
       non-empty.
    2. The ``measurement_root`` key in a machine-level settings file,
       ``%ProgramData%\\I2AS\\App-config.yaml`` on Windows
       (``os.name == "nt"``) or ``/etc/i2as/App-config.yaml`` on other
       platforms. A missing file, a missing key, or a blank value all fall
       through to step 3.
    3. No fallback: raises ``RuntimeError``.

    This is a pure function: it only resolves and returns a path, it never
    creates the directory, the settings file, or anything else.

    Returns:
        The resolved measurement root path (not guaranteed to exist).

    Raises:
        RuntimeError: Neither the environment variable nor the settings
            file resolves a measurement root, naming the exact settings
            file path that was checked.
    """
    env_dir = os.environ.get("I2AS_MEASUREMENT_ROOT")
    if env_dir:
        return Path(env_dir)

    config_path = _app_config_path()
    setting = _read_measurement_root_setting(config_path)
    if setting:
        return Path(setting)

    raise RuntimeError(
        "No measurement root configured. Set the I2AS_MEASUREMENT_ROOT "
        "environment variable, or create "
        f"{config_path} with a 'measurement_root: <path>' line."
    )
