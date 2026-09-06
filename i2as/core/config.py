"""Config readers — the YAML-only answers to "what does this setup declare?".

The **config-reader standard**: every ``read_*()`` function here parses one
of a config directory's two files (``devices.yaml``, ``monitor.yaml``),
returns a plain, fully defaulted Python value, and never raises — a missing
file, malformed YAML or an absent block reads as the defaults, so the GUI,
the troubleshoot CLI and the reference client can ask a config anything
before (or without) a Station being built. None of them imports a driver or
VI class, and none of them instantiates anything; building a Station from a
config is ``i2as.core.station.build_station()``'s job, and validating
one is ``validate_config_dir()``'s, both of which live beside the Station
because they need the classes it wires.

This module is the ONE place a user reads to learn the config surface: each
block's shape is documented on its reader, and each reader's defaults table
sits beside it. Import contract C24 keeps it free of everything above the
foundation, so the reference client and the MCP-side tooling can import it
without pulling the engine in.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from i2as.core.request_spool import DEFAULT_MAX_ROLE

logger = logging.getLogger(__name__)


def read_instrument_metadata(config_path: str) -> dict[str, dict[str, str]]:
    """Read each VI's optional descriptive ``metadata:`` block, GUI-safe.

    A setup property, like everything else in ``devices.yaml``: free-text
    identity (manufacturer, model, role, notes — whatever the setup wants to
    record about what a VI physically is) for display and for stamping onto
    every run's metadata. Parses YAML only — never imports a driver/VI class
    or instantiates anything, so it is safe to call from the GUI thread on a
    config that may describe unreachable hardware.

    Args:
        config_path: Path to the config directory containing ``devices.yaml``.

    Returns:
        ``{vi_name: {field: value, ...}}``, string-coerced. Empty for a VI
        with no ``metadata:`` block, and ``{}`` entirely if the config
        directory, file, or YAML is unreadable — never raises.
    """
    try:
        from ruamel.yaml import YAML  # type: ignore
    except ImportError:
        return {}

    devices_file = Path(config_path) / "devices.yaml"
    try:
        with devices_file.open("r", encoding="utf-8") as f:
            devices_config = dict(YAML().load(f) or {})
    except OSError:
        return {}
    except Exception:  # noqa: BLE001 — malformed YAML must not break the GUI
        return {}

    virtual_instruments = devices_config.get("virtual_instruments") or {}
    if not isinstance(virtual_instruments, dict):
        return {}

    result: dict[str, dict[str, str]] = {}
    for vi_name, vi_cfg in virtual_instruments.items():
        if not isinstance(vi_cfg, dict):
            continue
        metadata = vi_cfg.get("metadata")
        if isinstance(metadata, dict) and metadata:
            result[str(vi_name)] = {str(k): str(v) for k, v in metadata.items()}

    return result


# Defaults applied by read_safety_config() for every key the config omits —
# these apply even when the whole ``safety:`` block is absent
# (Orchestrator.acknowledge()'s override window, the safety-hold enforcement
# invariant, and stall detection are not opt-in features; every setup gets
# them).
#
# stall_seconds mirrors 18.0 s (the previous hardcoded 6-tick threshold at
# this module's 3000 ms reference tick interval — see
# i2as.core.stall_detection.StallConfig) as a wall-clock default so an
# omitted key preserves that setup's prior behaviour exactly; setups with a
# shorter tick_interval_ms now get a genuinely equivalent wall-clock
# threshold instead of a shorter one, which is the units-bug fix this default
# exists to deliver.
_SAFETY_DEFAULTS: dict[str, float] = {
    "manual_override_timeout_s": 300.0,
    "stall_seconds": 18.0,
    # Minimum seconds between standby() re-assertion attempts on the same
    # held VI (Orchestrator._enforce_safety_holds() — GLOSSARY.md's
    # **Safety hold**): keeps a persistently-held VI from being re-commanded
    # every tick.
    "hold_enforcement_interval_s": 10.0,
    # Consecutive failed standby() attempts on the same held VI before the
    # Orchestrator escalates (CRITICAL log + ErrorEvent) instead of quietly
    # retrying forever.
    "hold_enforcement_max_attempts": 3,
}

# Defaults applied by read_trends_config() for every key the config omits —
# always defaulted like _SAFETY_DEFAULTS, never {} on an absent block, so the
# trend-check scheduler (i2as.core.trend_check_runner) always has a
# refresh cadence to run on even for a setup that has never touched this
# block. A separate ``trends:`` block rather than folding into ``safety:``:
# a trend check is never enforcement (it can only ever publish an advisory
# Condition — see core/trend_checks.py), whereas every existing safety:
# key (manual_override_timeout_s, stall_seconds) times something that
# feeds an enforcement or run-fault path. Keeping the blocks separate keeps
# that distinction legible in devices.yaml, at the cost of one more block
# name to remember.
#
# The checks themselves are the `checks:` list — one entry per declared
# check, built by i2as.core.trend_checks.declared_checks() — and default
# to none: a setup that declares no check runs no check. Which channel is
# watched and what band it must stay in are setup facts (a controller's
# noise floor, a sample's safe range), so no default check ships here; the
# example config declares one.
_TREND_DEFAULTS: dict[str, Any] = {
    "refresh_interval_s": 60.0,
    "store_live_stale_ticks": 10.0,
    "checks": [],
}


def _load_devices_yaml(config_path: str) -> dict[str, Any] | None:
    """Parse ``devices.yaml`` under *config_path*, GUI-safe.

    Shared by every ``read_*`` reader in this module: YAML-parse only, never
    imports a driver/VI class or instantiates anything, so it is safe to call
    from the GUI thread on a config that may describe unreachable hardware.

    Args:
        config_path: Path to the config directory containing ``devices.yaml``.

    Returns:
        The parsed mapping, or ``None`` if ruamel.yaml is unavailable, the
        file is missing/unreadable, or the YAML is malformed.
    """
    try:
        from ruamel.yaml import YAML  # type: ignore
    except ImportError:
        return None

    devices_file = Path(config_path) / "devices.yaml"
    try:
        with devices_file.open("r", encoding="utf-8") as f:
            return dict(YAML().load(f) or {})
    except OSError:
        return None
    except Exception:  # noqa: BLE001 — malformed YAML must not break the GUI
        return None


def _load_monitor_yaml(config_path: str) -> dict[str, Any] | None:
    """Parse ``monitor.yaml`` under *config_path*, GUI-safe.

    Mirrors ``_load_devices_yaml`` for the display-side config file: YAML-parse
    only, never instantiates anything, never raises.

    Args:
        config_path: Path to the config directory containing ``monitor.yaml``.

    Returns:
        The parsed mapping, or ``None`` if ruamel.yaml is unavailable, the
        file is missing/unreadable, or the YAML is malformed.
    """
    try:
        from ruamel.yaml import YAML  # type: ignore
    except ImportError:
        return None

    monitor_file = Path(config_path) / "monitor.yaml"
    try:
        with monitor_file.open("r", encoding="utf-8") as f:
            return dict(YAML().load(f) or {})
    except OSError:
        return None
    except Exception:  # noqa: BLE001 — malformed YAML must not break the GUI
        return None


#: Orchestrator's own construction default (core/orchestrator.py's
#: `tick_interval_ms: int = 3000` parameter) — mirrored here, not imported,
#: since this module answers config questions for every layer and must not
#: depend on the engine (contract C24). ``build_station()`` reads it too.
DEFAULT_TICK_INTERVAL_MS = 3000


def read_tick_interval_ms(config_path: str) -> int:
    """Read ``monitor.yaml``'s ``tick_interval_ms``, GUI-safe, defaulted.

    A general config accessor, not built for one caller: any code that needs
    to reason in wall-clock seconds about a setup's tick cadence without
    constructing a full ``Orchestrator`` (e.g. the troubleshoot CLI's
    ``trends`` subcommand, evaluating `trend_store_live` from outside the
    running app) reads it through here rather than re-parsing
    ``monitor.yaml`` itself.

    Args:
        config_path: Path to the config directory containing ``monitor.yaml``.

    Returns:
        The configured tick interval in milliseconds, or
        ``DEFAULT_TICK_INTERVAL_MS`` if the file is missing, unreadable, or
        omits the key — never raises.
    """
    monitor_config = _load_monitor_yaml(config_path)
    if monitor_config is None:
        return DEFAULT_TICK_INTERVAL_MS
    mon = monitor_config.get("monitor")
    if not isinstance(mon, dict):
        return DEFAULT_TICK_INTERVAL_MS
    try:
        return int(mon.get("tick_interval_ms", DEFAULT_TICK_INTERVAL_MS))
    except (TypeError, ValueError):
        return DEFAULT_TICK_INTERVAL_MS


def read_request_spool_config(config_path: str) -> dict[str, Any]:
    """Read ``monitor.yaml``'s **Request spool** settings, GUI-safe, defaulted.

    Whether this setup offers a file-based write path into the running
    application, and how much authority that door may grant, are properties
    of the SETUP — a shared rig in a student lab and a single-user
    development machine want different answers — so they live in the config
    like every other limit. Expected shape::

        monitor:
          request_spool: true
          spool_max_role: session

    ``request_spool`` is ``false`` by default: an installation that has not
    asked for the door does not have one. ``spool_max_role`` is
    ``observer`` by default, the safe end of the role ladder, so turning the
    spool on without a second thought grants reads and nothing more.

    Args:
        config_path: Path to the config directory containing ``monitor.yaml``.

    Returns:
        ``{"enabled": bool, "max_role": str}``, fully defaulted. Never
        raises: a missing, unreadable or malformed file reads as "off".
    """
    settings: dict[str, Any] = {"enabled": False, "max_role": DEFAULT_MAX_ROLE}
    monitor_config = _load_monitor_yaml(config_path)
    if monitor_config is None:
        return settings
    block = monitor_config.get("monitor")
    if not isinstance(block, dict):
        return settings
    settings["enabled"] = bool(block.get("request_spool", False))
    max_role = block.get("spool_max_role", DEFAULT_MAX_ROLE)
    if isinstance(max_role, str) and max_role:
        settings["max_role"] = max_role
    return settings


def read_instrument_thread(config_path: str) -> bool:
    """Read ``monitor.yaml``'s ``instrument_thread`` flag, GUI-safe, defaulted.

    The setup's own answer to "does the instrument stack get its own thread?"
    — a property of the machine (how patient its instruments are, whether its
    VISA layer has been exercised under a second thread), so it lives in the
    config like every other setup property.
    ``core/instrument_host.py``'s ``resolve_mode()`` turns it into a mode, and
    lets ``I2AS_INSTRUMENT_THREAD`` override it for one launch.

    Defaulted to ``True``: the instrument thread is the standard, so a setup
    that says nothing inherits it, and a setup that wants the temporary
    ``inline`` mode back says ``instrument_thread: false`` deliberately.

    Args:
        config_path: Path to the config directory containing ``monitor.yaml``.

    Returns:
        ``False`` only when the file says so explicitly; ``True`` when it asks
        for the thread, omits the key, or is missing or unreadable — never
        raises.
    """
    monitor_config = _load_monitor_yaml(config_path)
    if monitor_config is None:
        return True
    mon = monitor_config.get("monitor")
    if not isinstance(mon, dict):
        return True
    return bool(mon.get("instrument_thread", True))


def read_panels_config(config_path: str) -> dict[str, list[str]]:
    """Read the optional ``panels:`` block of ``monitor.yaml``, GUI-safe.

    Which controls a setup's operators use day-to-day is a display property
    of the setup, so it lives in the config: each entry allowlists the
    controls shown on that VI's compact monitor card, overriding the
    ``panel=`` defaults the VI's ``@control`` declarations carry. A VI absent
    from the block keeps its declared defaults; every control, listed or
    not, remains available in the instrument's front panel. Display-only —
    hiding a control never disables it, and safety stays with
    ``control_limits``.

    Expected shape::

        panels:
          temperature:
            controls: [set_temperature]

    Args:
        config_path: Path to the config directory containing ``monitor.yaml``.

    Returns:
        ``{vi_name: [control_method_name, ...]}`` for every well-formed
        entry (a ``controls:`` list of strings). ``{}`` when the block is
        absent or the file/YAML is unreadable — never raises.
    """
    monitor_config = _load_monitor_yaml(config_path)
    if monitor_config is None:
        return {}
    block = monitor_config.get("panels")
    if not isinstance(block, dict):
        return {}
    result: dict[str, list[str]] = {}
    for vi_name, entry in block.items():
        if not isinstance(entry, dict):
            continue
        controls = entry.get("controls")
        if isinstance(controls, list):
            result[str(vi_name)] = [str(name) for name in controls]
    return result


#: The **Gateway server**'s defaults: off, and — when a setup does switch it
#: on — handing out no more than the role that reads and changes nothing.
#: Both are deliberately the most restrictive value: opening a process to
#: autonomous clients is a decision a setup makes explicitly, in its config,
#: exactly like every safety limit.
_GATEWAY_DEFAULTS: dict[str, Any] = {
    "gateway_server": False,
    "gateway_max_role": "observer",
}


def read_gateway_config(config_path: str) -> dict[str, Any]:
    """Read ``monitor.yaml``'s gateway keys, GUI-safe, always defaulted.

    Whether this setup accepts out-of-process clients, and how much authority
    it hands one, is a property of the setup rather than of the code — the
    same rule every limit follows — so both live in the config beside the
    tick interval. Like ``read_safety_config()``, an absent block means "use
    the defaults", and the defaults are the closed door.

    Expected shape::

        monitor:
          tick_interval_ms: 3000
          gateway_server: true
          gateway_max_role: session

    Args:
        config_path: Path to the config directory containing ``monitor.yaml``.

    Returns:
        ``_GATEWAY_DEFAULTS`` with any declared override merged in:
        ``gateway_server`` as a bool and ``gateway_max_role`` as a string.
        Falls back to the defaults untouched if the file or YAML is
        unreadable or the keys are absent — never raises.
    """
    merged = dict(_GATEWAY_DEFAULTS)
    monitor_config = _load_monitor_yaml(config_path)
    if monitor_config is None:
        return merged
    block = monitor_config.get("monitor")
    if not isinstance(block, dict):
        return merged
    if "gateway_server" in block:
        merged["gateway_server"] = bool(block["gateway_server"])
    if "gateway_max_role" in block:
        merged["gateway_max_role"] = str(block["gateway_max_role"])
    return merged


def read_safety_config(config_path: str) -> dict[str, float]:
    """Read the optional ``safety:`` block, GUI-safe, always defaulted.

    An absent ``safety:`` block does NOT mean "feature disabled" —
    ``Orchestrator.acknowledge()``'s override window applies to every setup, so this always returns
    ``_SAFETY_DEFAULTS`` merged with whatever the config overrides, never
    ``{}``.

    Expected shape::

        safety:
          manual_override_timeout_s: 300.0
          stall_seconds: 18.0
          hold_enforcement_interval_s: 10.0
          hold_enforcement_max_attempts: 3

    Args:
        config_path: Path to the config directory containing ``devices.yaml``.

    Returns:
        ``_SAFETY_DEFAULTS`` (``manual_override_timeout_s``,
        ``stall_seconds``, ``hold_enforcement_interval_s``,
        ``hold_enforcement_max_attempts``)
        with any declared overrides merged in. Falls back to the defaults
        untouched if the config directory/file/YAML is unreadable or the
        block is absent/malformed — never raises.
    """
    merged = dict(_SAFETY_DEFAULTS)
    devices_config = _load_devices_yaml(config_path)
    if devices_config is None:
        return merged
    block = devices_config.get("safety")
    if not isinstance(block, dict):
        return merged
    merged.update(block)
    return merged


def read_trends_config(config_path: str) -> dict[str, Any]:
    """Read the optional ``trends:`` block, GUI-safe, always defaulted.

    Mirrors ``read_safety_config()``'s pattern exactly: the trend-check
    scheduler (`i2as.core.trend_check_runner.TrendCheckRunner`) always
    needs a refresh cadence, so an absent block means "use the defaults",
    never "feature disabled". See ``_TREND_DEFAULTS``'s comment for why
    this is its own block rather than an extension of ``safety:``.

    Expected shape::

        trends:
          refresh_interval_s: 60.0
          store_live_stale_ticks: 10
          checks:                      # the Trend check standard's declarations
            - key: temperature_temperature
              low: 1.0
              high: 320.0
              window_s: 3600.0

    Each ``checks:`` entry is handed as-is to
    ``i2as.core.trend_checks.declared_checks()``, which builds and
    validates the check (an unknown field or a band with ``low >= high`` is
    a config error raised THERE, where the check kinds are known).

    Args:
        config_path: Path to the config directory containing ``devices.yaml``.

    Returns:
        ``_TREND_DEFAULTS`` with any declared overrides merged in;
        ``"checks"`` is always a list (``[]`` when absent). Falls back to
        the defaults untouched if the config directory/file/YAML is
        unreadable or the block is absent/malformed, and to ``[]`` when
        ``checks:`` is not a list — never raises.
    """
    merged: dict[str, Any] = dict(_TREND_DEFAULTS)
    merged["checks"] = []
    devices_config = _load_devices_yaml(config_path)
    if devices_config is None:
        return merged
    block = devices_config.get("trends")
    if not isinstance(block, dict):
        return merged
    merged.update(block)
    if not isinstance(merged["checks"], list):
        logger.warning(
            "trends.checks in %s must be a list, got %r — running no trend checks",
            config_path, merged["checks"],
        )
        merged["checks"] = []
    return merged
