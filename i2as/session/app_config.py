"""The general settings file — one JSON document, one section per concern.

**The standard place for an installation's settings.** Everything a person
chooses about how THIS installation behaves — which agents may connect, how
finished runs are analysed — lives in one file, ``settings.json`` in the
per-user config directory (``%APPDATA%\\I2AS`` on Windows), one top-level
section per concern. The Settings dialog (``i2as.gui.settings_dialog``) edits
it, one page per section. A new concern adds a section here and a page there,
and nothing else moves.

Sections today:

- ``connections`` (``ConnectionSettings``): the Agent gateway's on/off switch
  and role ceiling, and the HTTP MCP endpoint (remote access).
- ``analysis`` (``AnalysisSettings``): whether a finished run is analysed
  automatically, the worker's timeout, the report defaults, the
  per-procedure recipe preference, and the container the worker runs in
  (``SandboxSettings``).
- ``publishing`` (``PublishingSettings``): how the notebook service behaves
  on this machine — retry timings, the upload cap, and how long a block
  (a connector or renderer in its helper process) may take per call.

**What is NOT here.** Credentials, and anything a person chooses for
themselves. Notebook accounts, the drafting assistant and the session list
are per user (``i2as.session.user_profile``); API keys are in the system
keyring (``i2as.session.credentials``); the HTTP endpoint's access keys stay
in their own key store. A settings file that carries no secret can be read,
diffed and pasted into a bug report without a second thought.

**Tolerant, like every settings file here.** A missing file is the normal
first-launch case and yields the defaults; a malformed one logs a WARNING and
also yields the defaults; a junk field degrades to its own default. Startup
never fails on this file.

**Migration.** Before this file existed, the analysis switches lived in the
``analysis`` block of ``eln-settings.json`` and the connection switches in the
Qt settings store. ``load_app_config()`` reports a file with no ``analysis``
section as needing that block copied over (``legacy_analysis_block()``); the
GUI's ``app_settings.config_store()`` does the same for the Qt keys, then
writes the file once so the copy is permanent. The legacy ``sandbox`` block
(``local``/``venv``) is not carried over: analysis now runs in a container
only.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from i2as.core.paths import user_config_dir

logger = logging.getLogger(__name__)

#: Environment variable pointing at an explicit settings file (tests, and
#: installations that keep user state somewhere unusual).
APP_CONFIG_PATH_ENV_VAR = "I2AS_SETTINGS"

_APP_CONFIG_FILENAME = "settings.json"

#: The image the analysis worker runs in unless the settings name another. It
#: is built locally from the Dockerfile shipped in ``i2as/analysis/container``
#: (the Settings dialog's "Build image" button, or
#: ``python -m i2as.session.analysis_sandbox build-image``); a lab that needs
#: more libraries builds its own image ``FROM`` this one and names it here.
DEFAULT_ANALYSIS_IMAGE = "i2as-analysis:latest"

#: The container engines the analysis sandbox knows how to drive. Both take
#: the same ``run`` flags; ``podman`` is here for a lab that runs rootless.
CONTAINER_ENGINES: tuple[str, ...] = ("docker", "podman")

#: The Gateway's HTTP endpoint defaults: loopback only, a fixed port.
DEFAULT_REMOTE_HOST = "127.0.0.1"
DEFAULT_REMOTE_PORT = 8765


def app_config_path() -> Path:
    """Resolve the general settings file without creating it.

    Precedence:

    1. ``I2AS_SETTINGS``, if set and non-empty.
    2. ``settings.json`` under ``i2as.core.paths.user_config_dir()``.

    Returns:
        The resolved path (not guaranteed to exist).
    """
    env_path = os.environ.get(APP_CONFIG_PATH_ENV_VAR)
    if env_path:
        return Path(env_path)
    return user_config_dir() / _APP_CONFIG_FILENAME


# ── Tolerant parsing ──────────────────────────────────────────────────────
# The same rules as i2as.session.eln.settings, kept here so this module does
# not import the ELN package (whose __init__ imports the publisher, which
# imports this module).


def _as_bool(value: object, default: bool) -> bool:
    """Return ``value`` if it is a bool, else ``default``."""
    return value if isinstance(value, bool) else default


def _as_float(value: object, default: float) -> float:
    """Coerce a JSON value to ``float``, falling back to ``default``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int) -> int:
    """Coerce a JSON value to ``int``, falling back to ``default``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: object, default: str = "") -> str:
    """Coerce a JSON value to ``str``, falling back to ``default`` on ``None``."""
    return default if value is None else str(value)


def _as_recipes(value: object) -> dict[str, str]:
    """Coerce a JSON value to a ``{procedure: recipe name}`` map, dropping bad rows."""
    if not isinstance(value, dict):
        return {}
    recipes: dict[str, str] = {}
    for procedure, recipe in value.items():
        if recipe is None or isinstance(recipe, (list, tuple, dict, set)):
            logger.warning("Ignoring malformed analysis recipe row for %r", procedure)
            continue
        recipes[str(procedure)] = str(recipe)
    return recipes


# ── Sections ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SandboxSettings:
    """The container the analysis worker runs in — ``analysis.sandbox``.

    Attributes:
        engine: The container engine's command: ``docker`` or ``podman``, or
            a full path to either. Found on ``PATH`` when it is a bare name.
        image: The image to run. Never pulled: a missing image fails the
            analysis at once, naming the image, rather than fetching
            something from a registry nobody chose.
        memory: The container's memory cap, in the engine's own syntax
            (``4g``, ``512m``); ``""`` sets no cap.
        cpus: The container's CPU cap; ``0`` sets no cap.
        pids_limit: The most processes the container may hold, so a fork
            bomb in a script is a failed analysis rather than a hung machine.
    """

    engine: str = "docker"
    image: str = DEFAULT_ANALYSIS_IMAGE
    memory: str = "4g"
    cpus: float = 2.0
    pids_limit: int = 256

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "engine": self.engine,
            "image": self.image,
            "memory": self.memory,
            "cpus": self.cpus,
            "pids_limit": self.pids_limit,
        }

    @classmethod
    def from_dict(cls, data: object) -> SandboxSettings:
        """Build ``SandboxSettings`` from a parsed dict, tolerating bad input.

        A legacy block (``backend``/``python`` from before containers) has
        none of these keys and so parses as the defaults.

        Args:
            data: Any parsed JSON value; junk degrades to defaults.

        Returns:
            The settings record.
        """
        if not isinstance(data, dict):
            return cls()
        defaults = cls()
        return cls(
            engine=_as_str(data.get("engine"), defaults.engine).strip() or defaults.engine,
            image=_as_str(data.get("image"), defaults.image).strip() or defaults.image,
            memory=_as_str(data.get("memory"), defaults.memory).strip(),
            cpus=max(_as_float(data.get("cpus"), defaults.cpus), 0.0),
            pids_limit=max(_as_int(data.get("pids_limit"), defaults.pids_limit), 0),
        )


@dataclass(frozen=True)
class AnalysisSettings:
    """The analysis stage's section — ``analysis``.

    Attributes:
        enabled: Master switch. ``False`` (the default) means a finished run is
            rendered from its facts and queued as it always was; ``True``
            means it is analysed first, in a container, and nothing reaches
            the notebook until a human approves the result.
        timeout_s: How long one analysis may run before its container is
            killed and its report synthesized as failed.
        include_fact_tables: Default for a report's own flag — append the
            run's full fact tables below the analysis.
        attach_data_file: Default for a report's own flag — attach the raw
            data file to the entry.
        recipes: ``{procedure class name: recipe name}`` — which recipe a
            procedure prefers. A procedure with no row lets discovery choose.
        sandbox: The container the worker runs in (``SandboxSettings``).
    """

    enabled: bool = False
    timeout_s: float = 120.0
    include_fact_tables: bool = False
    attach_data_file: bool = False
    recipes: dict[str, str] = field(default_factory=dict)
    sandbox: SandboxSettings = field(default_factory=SandboxSettings)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "enabled": self.enabled,
            "timeout_s": self.timeout_s,
            "include_fact_tables": self.include_fact_tables,
            "attach_data_file": self.attach_data_file,
            "recipes": dict(self.recipes),
            "sandbox": self.sandbox.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object) -> AnalysisSettings:
        """Build ``AnalysisSettings`` from a parsed dict, tolerating bad input.

        Args:
            data: Any parsed JSON value; junk degrades to defaults, and a
                malformed recipe row is dropped rather than raised on.

        Returns:
            The settings record.
        """
        if not isinstance(data, dict):
            return cls()
        defaults = cls()
        return cls(
            enabled=_as_bool(data.get("enabled"), defaults.enabled),
            timeout_s=_as_float(data.get("timeout_s"), defaults.timeout_s),
            include_fact_tables=_as_bool(
                data.get("include_fact_tables"), defaults.include_fact_tables
            ),
            attach_data_file=_as_bool(data.get("attach_data_file"), defaults.attach_data_file),
            recipes=_as_recipes(data.get("recipes")),
            sandbox=SandboxSettings.from_dict(data.get("sandbox")),
        )


@dataclass(frozen=True)
class ConnectionSettings:
    """The Agent gateway and its HTTP endpoint — ``connections``.

    Attributes:
        gateway_enabled: Whether the gateway listens. ``None`` means the
            Settings dialog has never set it, so the active config's
            ``monitor.yaml`` ``gateway_server`` flag decides.
        gateway_max_role: The role ceiling the dialog last chose, or ``""``
            when it never has (``monitor.yaml``'s ``gateway_max_role`` then
            decides). Never above that ceiling: the controller refuses it.
        remote_enabled: Whether the HTTP MCP endpoint listens. Off by default.
        remote_host: The address it binds.
        remote_port: The TCP port it binds.
        remote_public_url: The address the operator's own tunnel or proxy
            hands out, or ``""``.
    """

    gateway_enabled: bool | None = None
    gateway_max_role: str = ""
    remote_enabled: bool = False
    remote_host: str = DEFAULT_REMOTE_HOST
    remote_port: int = DEFAULT_REMOTE_PORT
    remote_public_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "gateway_enabled": self.gateway_enabled,
            "gateway_max_role": self.gateway_max_role,
            "remote_enabled": self.remote_enabled,
            "remote_host": self.remote_host,
            "remote_port": self.remote_port,
            "remote_public_url": self.remote_public_url,
        }

    @classmethod
    def from_dict(cls, data: object) -> ConnectionSettings:
        """Build ``ConnectionSettings`` from a parsed dict, tolerating bad input.

        Args:
            data: Any parsed JSON value; junk degrades to defaults.

        Returns:
            The settings record.
        """
        if not isinstance(data, dict):
            return cls()
        defaults = cls()
        enabled = data.get("gateway_enabled")
        port = _as_int(data.get("remote_port"), defaults.remote_port)
        return cls(
            gateway_enabled=enabled if isinstance(enabled, bool) else None,
            gateway_max_role=_as_str(data.get("gateway_max_role")).strip(),
            remote_enabled=_as_bool(data.get("remote_enabled"), defaults.remote_enabled),
            remote_host=_as_str(data.get("remote_host")).strip() or defaults.remote_host,
            remote_port=port if 1 <= port <= 65535 else defaults.remote_port,
            remote_public_url=_as_str(data.get("remote_public_url")).strip(),
        )


@dataclass(frozen=True)
class PublishingSettings:
    """The ``publishing`` section: how the notebook service behaves on this machine.

    Attributes:
        retry_base_s: First retry delay after a transient failure; doubles
            per attempt.
        retry_max_s: Ceiling for that doubling.
        max_attachment_bytes: The largest file uploaded to a page.
        block_timeout_s: How long one call into a block's helper process may
            take before the process is killed.
        drain_interval_s: How often the service looks for due work.
    """

    retry_base_s: float = 30.0
    retry_max_s: float = 3600.0
    max_attachment_bytes: int = 50 * 1024 * 1024
    block_timeout_s: float = 60.0
    drain_interval_s: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "retry_base_s": self.retry_base_s,
            "retry_max_s": self.retry_max_s,
            "max_attachment_bytes": self.max_attachment_bytes,
            "block_timeout_s": self.block_timeout_s,
            "drain_interval_s": self.drain_interval_s,
        }

    @classmethod
    def from_dict(cls, data: object) -> PublishingSettings:
        """Build from a parsed dict, tolerating bad input."""
        if not isinstance(data, dict):
            return cls()
        defaults = cls()
        return cls(
            retry_base_s=max(_as_float(data.get("retry_base_s"), defaults.retry_base_s), 1.0),
            retry_max_s=max(_as_float(data.get("retry_max_s"), defaults.retry_max_s), 1.0),
            max_attachment_bytes=max(_as_int(data.get("max_attachment_bytes"), defaults.max_attachment_bytes), 0),
            block_timeout_s=max(_as_float(data.get("block_timeout_s"), defaults.block_timeout_s), 1.0),
            drain_interval_s=max(_as_float(data.get("drain_interval_s"), defaults.drain_interval_s), 0.5),
        )


@dataclass(frozen=True)
class AppConfig:
    """The whole general settings file: one field per section.

    Attributes:
        connections: The ``connections`` section.
        analysis: The ``analysis`` section.
        publishing: The ``publishing`` section.
    """

    connections: ConnectionSettings = field(default_factory=ConnectionSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    publishing: PublishingSettings = field(default_factory=PublishingSettings)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation, one key per section."""
        return {
            "connections": self.connections.to_dict(),
            "analysis": self.analysis.to_dict(),
            "publishing": self.publishing.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object) -> AppConfig:
        """Build ``AppConfig`` from a parsed dict, tolerating bad input.

        Args:
            data: Any parsed JSON value; junk degrades to defaults.

        Returns:
            The settings record.
        """
        if not isinstance(data, dict):
            return cls()
        return cls(
            connections=ConnectionSettings.from_dict(data.get("connections")),
            analysis=AnalysisSettings.from_dict(data.get("analysis")),
            publishing=PublishingSettings.from_dict(data.get("publishing")),
        )


# ── Reading and writing ───────────────────────────────────────────────────


def read_app_config_file(path: Path | None = None) -> dict[str, Any] | None:
    """Return the settings file's parsed top-level object, or ``None``.

    Args:
        path: The file; ``None`` uses ``app_config_path()``.

    Returns:
        The parsed dict; ``None`` when the file is missing, unreadable, or
        not a JSON object (the latter two logged).
    """
    settings_path = app_config_path() if path is None else path
    try:
        raw = settings_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("Could not read the settings file %s: %s", settings_path, exc)
        return None
    try:
        data = json.loads(raw)
    except ValueError as exc:
        logger.warning("Malformed settings file %s: %s", settings_path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Settings file %s is not a JSON object — using defaults", settings_path)
        return None
    return data


#: Environment variable that pointed at the old ``eln-settings.json``.
LEGACY_ELN_SETTINGS_ENV_VAR = "I2AS_ELN_SETTINGS"


def legacy_eln_settings_path() -> Path:
    """Return where the old ``eln-settings.json`` lives (it is only ever migrated from).

    Returns:
        ``I2AS_ELN_SETTINGS`` when set, else ``eln-settings.json`` in the
        per-user config directory.
    """
    override = os.environ.get(LEGACY_ELN_SETTINGS_ENV_VAR)
    return Path(override) if override else user_config_dir() / "eln-settings.json"


def legacy_analysis_block(path: Path | None = None) -> dict[str, Any] | None:
    """Return the ``analysis`` block ``eln-settings.json`` used to carry.

    Its ``sandbox`` sub-block is dropped: the ``local``/``venv`` backends it
    named no longer exist, and a stale interpreter path must not be mistaken
    for a container setting.

    Args:
        path: The old ELN settings file; ``None`` uses its usual place.

    Returns:
        The block as a dict, or ``None`` when there is none to migrate.
    """
    if path is None:
        path = legacy_eln_settings_path()
    eln_path = path
    migrated = eln_path.with_name(eln_path.name + ".migrated")
    if not eln_path.exists() and migrated.exists():
        # The user-profile migration may already have moved the file aside
        # (it keeps it as *.migrated); its analysis block is still ours.
        eln_path = migrated
    try:
        data = json.loads(eln_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    block = data.get("analysis") if isinstance(data, dict) else None
    if not isinstance(block, dict):
        return None
    return {key: value for key, value in block.items() if key != "sandbox"}


def load_app_config(path: Path | None = None) -> AppConfig:
    """Load the general settings, never raising.

    A file with no ``analysis`` section takes the legacy block from
    ``eln-settings.json`` instead, so a person who had switched analysis on
    before this file existed keeps it on.

    Args:
        path: The file; ``None`` uses ``app_config_path()``.

    Returns:
        The parsed settings.
    """
    data = read_app_config_file(path) or {}
    if "analysis" not in data:
        legacy = legacy_analysis_block()
        if legacy is not None:
            data = {**data, "analysis": legacy}
    return AppConfig.from_dict(data)


def save_app_config(config: AppConfig, path: Path | None = None) -> Path:
    """Write the general settings back, atomically.

    A temporary file in the same directory, then a replace, so a crash
    mid-write leaves the previous settings intact.

    Args:
        config: The record to write.
        path: The file; ``None`` uses ``app_config_path()``.

    Returns:
        The path written.

    Raises:
        OSError: If the directory cannot be created or the file written. The
            caller (a settings page) reports it.
    """
    settings_path = app_config_path() if path is None else Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = settings_path.with_name(settings_path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(tmp_path, settings_path)
    logger.info("Wrote the settings to %s", settings_path)
    return settings_path


class AppConfigStore:
    """The one in-memory copy of the settings file, shared by everyone who reads it.

    The GUI, the ELN publisher and the analysis runner all hold the SAME
    store, so a change saved from the Settings dialog is what the next
    finished run is analysed with — nobody re-reads the file, and nobody
    holds a stale copy. Listeners are told after every save.

    Args:
        path: The file; ``None`` uses ``app_config_path()`` (resolved once).
        config: A starting record instead of reading the file — for tests.
    """

    def __init__(self, path: Path | None = None, config: AppConfig | None = None) -> None:
        self._path = app_config_path() if path is None else Path(path)
        self._config = load_app_config(self._path) if config is None else config
        self._listeners: list[Callable[[AppConfig], None]] = []
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """The file this store reads and writes."""
        return self._path

    @property
    def current(self) -> AppConfig:
        """The settings as last loaded or saved."""
        return self._config

    def analysis(self) -> AnalysisSettings:
        """The ``analysis`` section — the shape a ``settings_source`` callable returns."""
        return self._config.analysis

    def connections(self) -> ConnectionSettings:
        """The ``connections`` section."""
        return self._config.connections

    def publishing(self) -> PublishingSettings:
        """The ``publishing`` section."""
        return self._config.publishing

    def save(self, config: AppConfig) -> None:
        """Write *config*, adopt it, and tell every listener.

        Args:
            config: The new settings.

        Raises:
            OSError: If the file cannot be written; the in-memory copy is
                left unchanged so memory never claims what the disk does not.
        """
        with self._lock:
            save_app_config(config, self._path)
            self._config = config
        for listener in list(self._listeners):
            try:
                listener(config)
            except Exception:  # noqa: BLE001 - one listener must not stop the rest
                logger.exception("A settings listener raised")

    def subscribe(self, listener: Callable[[AppConfig], None]) -> None:
        """Call *listener* with the new settings after every ``save()``.

        Args:
            listener: The callable.
        """
        self._listeners.append(listener)


__all__ = [
    "APP_CONFIG_PATH_ENV_VAR",
    "CONTAINER_ENGINES",
    "DEFAULT_ANALYSIS_IMAGE",
    "DEFAULT_REMOTE_HOST",
    "DEFAULT_REMOTE_PORT",
    "AnalysisSettings",
    "AppConfig",
    "AppConfigStore",
    "ConnectionSettings",
    "PublishingSettings",
    "SandboxSettings",
    "app_config_path",
    "legacy_analysis_block",
    "legacy_eln_settings_path",
    "load_app_config",
    "read_app_config_file",
    "save_app_config",
]
