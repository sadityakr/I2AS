"""Unit tests for validate_config_dir — the config editor's dry-run gate.

Confirms it reports importable-class and driver-reference problems without
instantiating any driver or VI (so validating a real-hardware config is safe).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from i2as.core.exceptions import I2ASConfigError
from i2as.core.config import read_instrument_metadata
from i2as.core.station import build_station_with_fallback, validate_config_dir

_SIM_CONFIG = "i2as/configs/sim_cryostat"

_GOOD_DEVICES = """\
real_drivers:
  ips_x:
    class: i2as.drivers.sim_oxford_ips120.SimOxfordIPS120
    address: "SIM::IPS_X"
virtual_instruments:
  magnet_z:
    class: i2as.virtual_instruments.magnet.superconducting_magnet.SuperconductingMagnetVI
    drivers: {main: ips_x}
    vi_type: system
"""

_MONITOR = "monitor:\n  tick_interval_ms: 3000\n"


def _write(base: Path, devices: str, monitor: str = _MONITOR) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    (base / "devices.yaml").write_text(devices, encoding="utf-8")
    (base / "monitor.yaml").write_text(monitor, encoding="utf-8")
    return base


def test_valid_config_has_no_errors(tmp_path):
    """A well-formed config validates clean."""
    d = _write(tmp_path / "cfg", _GOOD_DEVICES)
    assert validate_config_dir(str(d)) == []


def test_missing_files_reported(tmp_path):
    """A directory with no YAML files reports both missing."""
    (tmp_path / "empty").mkdir()
    errors = validate_config_dir(str(tmp_path / "empty"))
    assert any("devices.yaml" in e for e in errors)
    assert any("monitor.yaml" in e for e in errors)


def test_unknown_driver_reference_reported(tmp_path):
    """A VI referencing an undefined driver is flagged."""
    devices = _GOOD_DEVICES.replace("main: ips_x", "main: does_not_exist")
    d = _write(tmp_path / "cfg", devices)
    errors = validate_config_dir(str(d))
    assert any("does_not_exist" in e for e in errors)


def test_unimportable_class_reported(tmp_path):
    """A non-importable driver class is flagged."""
    devices = _GOOD_DEVICES.replace(
        "i2as.drivers.sim_oxford_ips120.SimOxfordIPS120",
        "i2as.drivers.nope.DoesNotExist",
    )
    d = _write(tmp_path / "cfg", devices)
    errors = validate_config_dir(str(d))
    assert any("ips_x" in e for e in errors)


def test_malformed_yaml_reported(tmp_path):
    """Unparseable devices.yaml is reported, not raised."""
    d = _write(tmp_path / "cfg", "real_drivers: {: broken\n")
    errors = validate_config_dir(str(d))
    assert any("devices.yaml" in e for e in errors)


# ── build_station_with_fallback ───────────────────────────────────────────────

def test_fallback_first_valid_wins():
    """The first valid config is used and there are no warnings."""
    station, used, warnings = build_station_with_fallback([_SIM_CONFIG])
    assert used == _SIM_CONFIG
    assert warnings == []
    assert station.get_vi_names()


def test_fallback_skips_invalid_uses_next(tmp_path):
    """An invalid first candidate is skipped (with a warning) for the next."""
    bad = _write(tmp_path / "bad", "real_drivers: {: broken\n")
    station, used, warnings = build_station_with_fallback([str(bad), _SIM_CONFIG])
    assert used == _SIM_CONFIG
    assert warnings and "bad" in warnings[0]
    assert station.get_vi_names()


def test_fallback_all_invalid_raises(tmp_path):
    """If nothing is usable, it raises rather than returning a broken station."""
    bad = _write(tmp_path / "bad", "real_drivers: {: broken\n")
    with pytest.raises(I2ASConfigError):
        build_station_with_fallback([str(bad)])


# ── read_instrument_metadata ────────────────────────────────────────────────

_DEVICES_WITH_METADATA = """\
real_drivers:
  ips_x:
    class: i2as.drivers.sim_oxford_ips120.SimOxfordIPS120
    address: "SIM::IPS_X"
virtual_instruments:
  magnet_z:
    class: i2as.virtual_instruments.magnet.superconducting_magnet.SuperconductingMagnetVI
    drivers: {main: ips_x}
    vi_type: system
    metadata:
      role: X-axis magnet
      manufacturer: Oxford Instruments
      model: IPS120-10
  magnet_y:
    class: i2as.virtual_instruments.magnet.superconducting_magnet.SuperconductingMagnetVI
    drivers: {main: ips_x}
    vi_type: system
"""


def test_read_instrument_metadata_returns_only_vis_with_a_block(tmp_path):
    """A VI with a metadata: block is included; one without is omitted."""
    d = _write(tmp_path / "cfg", _DEVICES_WITH_METADATA)
    result = read_instrument_metadata(str(d))
    assert result == {
        "magnet_z": {
            "role": "X-axis magnet",
            "manufacturer": "Oxford Instruments",
            "model": "IPS120-10",
        }
    }


def test_read_instrument_metadata_from_sim_cryostat():
    """The shipped sim_cryostat config carries metadata for every VI."""
    result = read_instrument_metadata(_SIM_CONFIG)
    assert result  # non-empty
    assert set(result.keys()) <= set(build_station_with_fallback([_SIM_CONFIG])[0].get_vi_names())
    for vi_meta in result.values():
        assert "role" in vi_meta


def test_read_instrument_metadata_never_instantiates_anything(tmp_path):
    """A metadata-only read of a config with an unimportable class does not raise."""
    devices = """\
virtual_instruments:
  ghost:
    class: i2as.does.not.exist.Ghost
    drivers: {}
    vi_type: system
    metadata:
      role: Deliberately unimportable
"""
    d = _write(tmp_path / "cfg", devices)
    assert read_instrument_metadata(str(d)) == {"ghost": {"role": "Deliberately unimportable"}}


def test_read_instrument_metadata_tolerates_missing_directory():
    """A nonexistent config path returns {} rather than raising."""
    assert read_instrument_metadata("/no/such/config/path") == {}


def test_read_instrument_metadata_tolerates_malformed_yaml(tmp_path):
    """Unparseable devices.yaml returns {} rather than raising."""
    d = _write(tmp_path / "cfg", "real_drivers: {: broken\n")
    assert read_instrument_metadata(str(d)) == {}


def test_read_instrument_metadata_ignores_empty_metadata_block(tmp_path):
    """A VI with an empty metadata: {} block is omitted, not included as {}."""
    devices = """\
virtual_instruments:
  magnet_z:
    class: i2as.virtual_instruments.magnet.superconducting_magnet.SuperconductingMagnetVI
    drivers: {}
    vi_type: system
    metadata: {}
"""
    d = _write(tmp_path / "cfg", devices)
    assert read_instrument_metadata(str(d)) == {}
