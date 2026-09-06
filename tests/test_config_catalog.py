"""Unit tests for the Qt-free config catalog (i2as.core.config_catalog).

Exercises discovery, copy-on-edit fork, and named-version save/list/restore
against throwaway shipped/user directories under tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from i2as.core.config_catalog import ConfigCatalog


def _make_shipped(base: Path, name: str, devices: str = "real_drivers: {}\n") -> Path:
    """Create a minimal shipped config directory and return it."""
    d = base / name
    d.mkdir(parents=True)
    (d / "devices.yaml").write_text(devices, encoding="utf-8")
    (d / "monitor.yaml").write_text("monitor:\n  tick_interval_ms: 3000\n", encoding="utf-8")
    return d


@pytest.fixture
def catalog(tmp_path):
    """A catalog over a shipped dir (one config) and an empty user dir."""
    shipped = tmp_path / "shipped"
    user = tmp_path / "user"
    _make_shipped(shipped, "sim_cryostat")
    return ConfigCatalog(shipped, user)


# ── discovery ─────────────────────────────────────────────────────────────────

def test_list_configs_finds_shipped_read_only(catalog):
    """Shipped configs are discovered and marked read-only."""
    entries = catalog.list_configs()
    assert len(entries) == 1
    assert entries[0].name == "sim_cryostat"
    assert entries[0].read_only is True
    assert entries[0].source == "shipped"


def test_list_configs_skips_non_config_dirs(tmp_path):
    """A directory without devices.yaml is not treated as a config."""
    shipped = tmp_path / "shipped"
    (shipped / "not_a_config").mkdir(parents=True)
    _make_shipped(shipped, "real_one")
    catalog = ConfigCatalog(shipped, tmp_path / "user")
    assert [e.name for e in catalog.list_configs()] == ["real_one"]


def test_missing_user_dir_is_fine(catalog):
    """A not-yet-created user dir yields no user configs, no error."""
    assert all(e.source == "shipped" for e in catalog.list_configs())


# ── fork (copy-on-edit) ───────────────────────────────────────────────────────

def test_fork_shipped_creates_editable_user_copy(catalog):
    """Forking a shipped config yields an editable user copy with a seed version."""
    entry = catalog.fork_shipped("sim_cryostat")
    assert entry.read_only is False
    assert entry.source == "user"
    assert (entry.path / "devices.yaml").is_file()
    # Now discoverable as a user config alongside the shipped one.
    sources = sorted(e.source for e in catalog.list_configs())
    assert sources == ["shipped", "user"]
    # Seeded with one version.
    assert len(catalog.list_versions("sim_cryostat")) == 1


def test_fork_shipped_missing_source_raises(catalog):
    """Forking a non-existent shipped config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        catalog.fork_shipped("nope")


def test_fork_shipped_existing_dest_raises(catalog):
    """Forking onto an existing user config raises FileExistsError."""
    catalog.fork_shipped("sim_cryostat")
    with pytest.raises(FileExistsError):
        catalog.fork_shipped("sim_cryostat")


def test_shipped_config_untouched_by_fork(catalog):
    """The shipped baseline is never modified by editing its fork."""
    original = (catalog._shipped_dir / "sim_cryostat" / "devices.yaml").read_text()
    catalog.fork_shipped("sim_cryostat")
    catalog.save_version("sim_cryostat", "real_drivers: {edited: true}\n")
    assert (catalog._shipped_dir / "sim_cryostat" / "devices.yaml").read_text() == original


# ── versioning ────────────────────────────────────────────────────────────────

def test_save_version_snapshots_and_lists_newest_first(catalog):
    """Each save writes the active file and adds a listable version."""
    catalog.fork_shipped("sim_cryostat")
    # Far-future ids so they sort above the seed version's real-time id.
    catalog.save_version("sim_cryostat", "real_drivers: {v: 1}\n", label="v1", version_id="99991231-000001")
    catalog.save_version("sim_cryostat", "real_drivers: {v: 2}\n", label="v2", version_id="99991231-000002")

    versions = catalog.list_versions("sim_cryostat")
    # seed + v1 + v2, newest first
    assert versions[0].version_id == "99991231-000002"
    assert versions[0].label == "v2"
    # Active file reflects the latest save.
    devices, _ = catalog.read_active("sim_cryostat")
    assert "v: 2" in devices


def test_version_id_collision_gets_counter(catalog):
    """Two saves with the same explicit id do not overwrite each other."""
    catalog.fork_shipped("sim_cryostat")
    catalog.save_version("sim_cryostat", "a: 1\n", version_id="dup")
    catalog.save_version("sim_cryostat", "a: 2\n", version_id="dup")
    ids = {v.version_id for v in catalog.list_versions("sim_cryostat")}
    assert "dup" in ids
    assert "dup-2" in ids


def test_restore_version_overwrites_active(catalog):
    """Restoring a version copies its files back over the active config."""
    catalog.fork_shipped("sim_cryostat")
    catalog.save_version("sim_cryostat", "real_drivers: {v: 1}\n", version_id="20260101-000001")
    catalog.save_version("sim_cryostat", "real_drivers: {v: 2}\n", version_id="20260101-000002")

    catalog.restore_version("sim_cryostat", "20260101-000001")
    devices, _ = catalog.read_active("sim_cryostat")
    assert "v: 1" in devices


def test_restore_unknown_version_raises(catalog):
    """Restoring a non-existent version raises FileNotFoundError."""
    catalog.fork_shipped("sim_cryostat")
    with pytest.raises(FileNotFoundError):
        catalog.restore_version("sim_cryostat", "nope")


def test_save_version_preserves_monitor_when_omitted(catalog):
    """Saving with monitor_text=None keeps the existing monitor.yaml."""
    catalog.fork_shipped("sim_cryostat")
    catalog.save_version("sim_cryostat", "real_drivers: {v: 9}\n")
    _, monitor = catalog.read_active("sim_cryostat")
    assert "tick_interval_ms" in monitor
