"""config_catalog — discover and version I2AS config directories.

Two tiers, mirroring the session-state design: the shipped configs under
``i2as/configs`` are read-only baselines (git-tracked, a safe fallback that
the GUI can never corrupt), and user edits live as copies under a writable
user-config directory. Editing a shipped config is a copy-on-edit ``fork``; each
save of a user config is kept as a named version so recovery never depends on a
single file staying intact.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_DEVICES = "devices.yaml"
_MONITOR = "monitor.yaml"
_VERSIONS_DIR = ".versions"
_VERSION_META = "meta.json"


@dataclass(frozen=True)
class ConfigEntry:
    """A discovered config directory.

    Attributes:
        name: The directory name (shown in the picker).
        path: The directory ``build_station`` loads.
        read_only: True for shipped baselines (not editable in place).
        source: ``"shipped"`` or ``"user"``.
    """

    name: str
    path: Path
    read_only: bool
    source: str


@dataclass(frozen=True)
class ConfigVersion:
    """One saved snapshot of a user config.

    Attributes:
        version_id: Sortable timestamp id and the snapshot directory name.
        label: Optional user-given name for the version.
        timestamp: ISO-8601 creation time.
        path: The snapshot directory (holds devices.yaml + monitor.yaml).
    """

    version_id: str
    label: str
    timestamp: str
    path: Path


def _is_config_dir(path: Path) -> bool:
    """True if ``path`` is a directory holding a devices.yaml."""
    return path.is_dir() and (path / _DEVICES).is_file()


def _iter_config_dirs(base: Path) -> list[Path]:
    """Return immediate sub-directories of ``base`` that are config dirs, sorted."""
    if not base.is_dir():
        return []
    return sorted(
        (d for d in base.iterdir() if not d.name.startswith(".") and _is_config_dir(d)),
        key=lambda d: d.name.lower(),
    )


class ConfigCatalog:
    """Discovery + versioning over shipped and user config directories."""

    def __init__(self, shipped_dir: Path | str, user_dir: Path | str) -> None:
        """Initialise the catalog.

        Args:
            shipped_dir: The git-tracked ``i2as/configs`` directory.
            user_dir: A writable directory for user config copies. Created on
                first write; it need not exist yet.
        """
        self._shipped_dir = Path(shipped_dir)
        self._user_dir = Path(user_dir)

    # -- discovery ----------------------------------------------------------

    def list_configs(self) -> list[ConfigEntry]:
        """Return all configs: shipped (read-only) first, then user copies."""
        entries = [
            ConfigEntry(name=d.name, path=d, read_only=True, source="shipped")
            for d in _iter_config_dirs(self._shipped_dir)
        ]
        entries += [
            ConfigEntry(name=d.name, path=d, read_only=False, source="user")
            for d in _iter_config_dirs(self._user_dir)
        ]
        return entries

    def get_by_path(self, path: Path | str) -> ConfigEntry | None:
        """Return the entry whose directory matches ``path``, or None."""
        target = Path(path).resolve()
        for entry in self.list_configs():
            if entry.path.resolve() == target:
                return entry
        return None

    def user_config_dir(self, name: str) -> Path:
        """Return the (possibly non-existent) directory for user config ``name``."""
        return self._user_dir / name

    # -- copy-on-edit -------------------------------------------------------

    def fork_shipped(self, shipped_name: str, new_name: str | None = None) -> ConfigEntry:
        """Copy a shipped config into an editable user config.

        Args:
            shipped_name: Name of the shipped config to copy.
            new_name: Destination name under the user dir; defaults to
                ``shipped_name``.

        Returns:
            The new editable ``ConfigEntry``.

        Raises:
            FileNotFoundError: If the shipped config does not exist.
            FileExistsError: If a user config of the destination name exists.
        """
        src = self._shipped_dir / shipped_name
        if not _is_config_dir(src):
            raise FileNotFoundError(f"shipped config '{shipped_name}' not found")
        dst = self._user_dir / (new_name or shipped_name)
        if dst.exists():
            raise FileExistsError(f"user config '{dst.name}' already exists")
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / _DEVICES, dst / _DEVICES)
        if (src / _MONITOR).is_file():
            shutil.copy2(src / _MONITOR, dst / _MONITOR)
        entry = ConfigEntry(name=dst.name, path=dst, read_only=False, source="user")
        # Seed the version history with the forked-from state.
        self.save_version(
            dst.name,
            (dst / _DEVICES).read_text(encoding="utf-8"),
            (dst / _MONITOR).read_text(encoding="utf-8") if (dst / _MONITOR).is_file() else None,
            label=f"forked from {shipped_name}",
        )
        return entry

    # -- read / save --------------------------------------------------------

    def read_active(self, name: str) -> tuple[str, str]:
        """Return the (devices.yaml, monitor.yaml) text of a user config.

        Args:
            name: The user config name.

        Returns:
            A ``(devices_text, monitor_text)`` tuple; ``monitor_text`` is ``""``
            if the file is absent.

        Raises:
            FileNotFoundError: If the config or its devices.yaml is missing.
        """
        config_dir = self._user_dir / name
        devices = config_dir / _DEVICES
        if not devices.is_file():
            raise FileNotFoundError(f"user config '{name}' has no {_DEVICES}")
        monitor = config_dir / _MONITOR
        return (
            devices.read_text(encoding="utf-8"),
            monitor.read_text(encoding="utf-8") if monitor.is_file() else "",
        )

    def save_version(
        self,
        name: str,
        devices_text: str,
        monitor_text: str | None = None,
        label: str = "",
        version_id: str | None = None,
    ) -> ConfigVersion:
        """Write the active config and snapshot it as a new named version.

        The active ``devices.yaml`` (and ``monitor.yaml`` when ``monitor_text``
        is given) is written first, then copied into ``.versions/<id>/`` with a
        ``meta.json``. The newest version therefore always matches the active
        files.

        Args:
            name: The user config name (created if it does not exist).
            devices_text: New devices.yaml content.
            monitor_text: New monitor.yaml content; when None, an existing
                monitor.yaml is preserved and snapshotted as-is.
            label: Optional user-facing version name.
            version_id: Snapshot id; defaults to a timestamp. A collision is
                resolved by appending a counter.

        Returns:
            The created ``ConfigVersion``.
        """
        config_dir = self._user_dir / name
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / _DEVICES).write_text(devices_text, encoding="utf-8")
        if monitor_text is not None:
            (config_dir / _MONITOR).write_text(monitor_text, encoding="utf-8")

        now = datetime.now()
        base_id = version_id or now.strftime("%Y%m%d-%H%M%S")
        versions_root = config_dir / _VERSIONS_DIR
        versions_root.mkdir(parents=True, exist_ok=True)
        vid = base_id
        counter = 2
        while (versions_root / vid).exists():
            vid = f"{base_id}-{counter}"
            counter += 1

        snapshot = versions_root / vid
        snapshot.mkdir()
        shutil.copy2(config_dir / _DEVICES, snapshot / _DEVICES)
        if (config_dir / _MONITOR).is_file():
            shutil.copy2(config_dir / _MONITOR, snapshot / _MONITOR)
        timestamp = now.isoformat(timespec="seconds")
        (snapshot / _VERSION_META).write_text(
            json.dumps({"label": label, "timestamp": timestamp}), encoding="utf-8"
        )
        return ConfigVersion(
            version_id=vid, label=label, timestamp=timestamp, path=snapshot
        )

    def list_versions(self, name: str) -> list[ConfigVersion]:
        """Return a user config's saved versions, newest first.

        Args:
            name: The user config name.

        Returns:
            A list of ``ConfigVersion`` sorted by ``version_id`` descending; an
            empty list if the config has no version history.
        """
        versions_root = self._user_dir / name / _VERSIONS_DIR
        if not versions_root.is_dir():
            return []
        versions: list[ConfigVersion] = []
        for snapshot in versions_root.iterdir():
            if not (snapshot / _DEVICES).is_file():
                continue
            label, timestamp = "", ""
            meta = snapshot / _VERSION_META
            if meta.is_file():
                try:
                    parsed = json.loads(meta.read_text(encoding="utf-8"))
                    label = str(parsed.get("label", ""))
                    timestamp = str(parsed.get("timestamp", ""))
                except (ValueError, OSError):
                    pass
            versions.append(
                ConfigVersion(
                    version_id=snapshot.name,
                    label=label,
                    timestamp=timestamp,
                    path=snapshot,
                )
            )
        return sorted(versions, key=lambda v: v.version_id, reverse=True)

    def restore_version(self, name: str, version_id: str) -> None:
        """Copy a saved version's files over the active config.

        Args:
            name: The user config name.
            version_id: The version to restore.

        Raises:
            FileNotFoundError: If the version does not exist.
        """
        snapshot = self._user_dir / name / _VERSIONS_DIR / version_id
        if not (snapshot / _DEVICES).is_file():
            raise FileNotFoundError(f"version '{version_id}' of '{name}' not found")
        config_dir = self._user_dir / name
        shutil.copy2(snapshot / _DEVICES, config_dir / _DEVICES)
        if (snapshot / _MONITOR).is_file():
            shutil.copy2(snapshot / _MONITOR, config_dir / _MONITOR)
