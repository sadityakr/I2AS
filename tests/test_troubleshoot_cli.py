"""Troubleshoot CLI tests — the command grammar and exit codes are API for skills."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from i2as.troubleshoot import cli
from tests.test_troubleshoot_engine import (
    _THIS,
    FakeInstrument,
    FakeResourceManager,
    make_config,
)

# Captured before the autouse isolated_transcript fixture below patches
# cli._transcript_dir per-test, so a test can restore the real resolver
# delegation and check it against I2AS_LOG_DIR.
_REAL_TRANSCRIPT_DIR = cli._transcript_dir


@pytest.fixture(autouse=True)
def isolated_transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send every invocation's transcript into the test's tmp dir."""
    transcript_dir = tmp_path / "logs"
    monkeypatch.setattr(cli, "_transcript_dir", lambda: transcript_dir)
    return transcript_dir


@pytest.fixture()
def fake_bus(monkeypatch: pytest.MonkeyPatch) -> FakeResourceManager:
    rm = FakeResourceManager(
        {
            "GPIB0::7::INSTR": FakeInstrument(reply="KEITHLEY,2182A,123,C01"),
            "ASRL10::INSTR": FakeInstrument(reply="IPS120-10 Version 3.07"),
        }
    )
    monkeypatch.setattr(cli, "_rm_factory", lambda: rm)
    return rm


_TRENDS_BLOCK = """
trends:
  checks:
    - key: temperature_temperature
      low: 1.0
      high: 320.0
      window_s: 3600.0
"""


@pytest.fixture()
def sim_config(tmp_path: Path) -> str:
    """A minimal config declaring one trend check, as the shipped example does."""
    path = make_config(
        tmp_path / "cfg",
        {
            "meter": {
                "class": "i2as.drivers.sim_keithley_2182a.SimKeithley2182A",
                "address": "SIM::CLI",
            }
        },
    )
    with (Path(path) / "devices.yaml").open("a", encoding="utf-8") as handle:
        handle.write(_TRENDS_BLOCK)
    return path


@pytest.fixture()
def sim_config_without_trends(tmp_path: Path) -> str:
    """The same minimal config with no ``trends:`` block at all."""
    return make_config(
        tmp_path / "cfg_plain",
        {
            "meter": {
                "class": "i2as.drivers.sim_keithley_2182a.SimKeithley2182A",
                "address": "SIM::CLI",
            }
        },
    )


def _json_out(capsys: pytest.CaptureFixture) -> dict:
    return json.loads(capsys.readouterr().out)


# ── scan / probe ──────────────────────────────────────────────────────────────


def test_scan_lists_resources_json(fake_bus, capsys) -> None:
    assert cli.main(["scan", "--json"]) == 0
    payload = _json_out(capsys)
    assert payload["resources"] == ["ASRL10::INSTR", "GPIB0::7::INSTR"]
    assert "probes" not in payload


def test_scan_probe_skips_serial_by_default(fake_bus, capsys) -> None:
    assert cli.main(["scan", "--probe", "--json"]) == 0
    probes = _json_out(capsys)["probes"]
    assert [p["address"] for p in probes] == ["GPIB0::7::INSTR"]
    assert probes[0]["code"] == "OK"


def test_scan_probe_serial_opt_in(fake_bus, capsys) -> None:
    assert cli.main(["scan", "--probe-serial", "--json"]) == 0
    probes = _json_out(capsys)["probes"]
    assert [p["address"] for p in probes] == ["ASRL10::INSTR", "GPIB0::7::INSTR"]


def test_probe_ok_exit_zero(fake_bus, capsys) -> None:
    assert cli.main(["probe", "GPIB0::7::INSTR", "--json"]) == 0
    assert _json_out(capsys)["idn"] == "KEITHLEY,2182A,123,C01"


def test_probe_missing_address_exit_one(fake_bus, capsys) -> None:
    assert cli.main(["probe", "GPIB0::99::INSTR", "--json"]) == 1
    assert _json_out(capsys)["code"] == "OPEN_FAILED"


# ── check ─────────────────────────────────────────────────────────────────────


def test_check_sim_config_green(sim_config, capsys) -> None:
    assert cli.main(["check", "--config", sim_config, "--no-bus", "--json"]) == 0
    payload = _json_out(capsys)
    assert payload["ok"] is True
    assert payload["results"][0]["code"] == "OK"


def test_check_failing_driver_exit_one(tmp_path, capsys) -> None:
    config = make_config(
        tmp_path / "bad",
        {"dead": {"class": f"{_THIS}.OpenFailsDriver", "address": "GPIB0::2::INSTR"}},
    )
    assert cli.main(["check", "--config", config, "--no-bus", "--json"]) == 1
    assert _json_out(capsys)["results"][0]["code"] == "OPEN_FAILED"


def test_check_resolves_shipped_config_by_name(capsys) -> None:
    """--config sim_cryostat resolves against the shipped configs folder."""
    assert cli.main(["check", "--config", "sim_cryostat", "--no-bus", "--json"]) == 0
    payload = _json_out(capsys)
    assert payload["ok"] is True
    assert "sim_cryostat" in payload["config"]


def test_unknown_config_name_is_a_clean_error() -> None:
    with pytest.raises(SystemExit):
        cli.main(["check", "--config", "no-such-setup", "--no-bus"])


# ── methods / idn / read / write ──────────────────────────────────────────────


def test_methods_reports_read_write_classification(sim_config, capsys) -> None:
    assert cli.main(["methods", "meter", "--config", sim_config, "--json"]) == 0
    methods = {m["name"]: m for m in _json_out(capsys)["methods"]}
    assert methods["get_voltage"]["read_only"] is True
    assert methods["set_range"]["read_only"] is False


def test_idn_command(sim_config, capsys) -> None:
    assert cli.main(["idn", "meter", "--config", sim_config, "--json"]) == 0
    assert _json_out(capsys)["result"] == "KEITHLEY,2182A,SIM,1.0"


def test_read_calls_getter(sim_config, capsys) -> None:
    assert cli.main(["read", "meter", "get_voltage", "--config", sim_config, "--json"]) == 0
    assert isinstance(_json_out(capsys)["result"], float)


def test_read_refuses_writing_method(sim_config, capsys) -> None:
    """The CLI-level read/write split: 'read' never reaches a set_* method."""
    exit_code = cli.main(
        ["read", "meter", "set_range", "0.1", "--config", sim_config, "--json"]
    )
    assert exit_code == 1
    assert "changes instrument state" in _json_out(capsys)["error"]


def test_write_allows_setter_with_coercion(sim_config, capsys) -> None:
    assert cli.main(
        ["write", "meter", "set_range", "0.1", "--config", sim_config, "--json"]
    ) == 0
    payload = _json_out(capsys)
    assert payload["method"] == "set_range"
    assert payload["args"] == ["0.1"]


def test_adhoc_bench_via_class_and_address(capsys) -> None:
    """TARGET + --address benches a driver with no config entry (driver dev)."""
    assert cli.main(
        ["idn", f"{_THIS}.AlwaysUpDriver", "--address", "GPIB0::30::INSTR", "--json"]
    ) == 0
    assert _json_out(capsys)["result"] == "ACME,MODEL1,SN42,9.9"


class FlakyDriver:
    """Fails every second read — the intermittent/timing fault signature."""

    _calls = 0

    def __init__(self, resource_string: str) -> None:
        type(self)._calls = 0

    def get_value(self) -> float:
        type(self)._calls += 1
        if type(self)._calls % 2 == 0:
            raise OSError("timeout")
        return 1.0


def test_read_repeat_all_ok(sim_config, capsys) -> None:
    exit_code = cli.main(
        ["read", "meter", "get_voltage", "--config", sim_config,
         "--repeat", "3", "--interval", "0", "--json"]
    )
    assert exit_code == 0
    payload = _json_out(capsys)
    assert payload["failures"] == 0
    assert len(payload["outcomes"]) == 3


def test_read_repeat_exposes_intermittency(capsys) -> None:
    exit_code = cli.main(
        ["read", "tests.test_troubleshoot_cli.FlakyDriver", "get_value",
         "--address", "SIM::FLAKY", "--repeat", "4", "--interval", "0", "--json"]
    )
    assert exit_code == 1
    payload = _json_out(capsys)
    assert payload["failures"] == 2
    assert [o["ok"] for o in payload["outcomes"]] == [True, False, True, False]


# ── query / send ──────────────────────────────────────────────────────────────


def test_query_without_raw_handle_is_clean_error(sim_config, capsys) -> None:
    exit_code = cli.main(
        ["query", "meter", "*IDN?", "--config", sim_config, "--json"]
    )
    assert exit_code == 1
    assert "raw VISA handle" in _json_out(capsys)["error"]


# ── trends ────────────────────────────────────────────────────────────────────


def _write_trend_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


@pytest.fixture()
def isolated_trend_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh, empty log directory, wired through I2AS_LOG_DIR."""
    log_dir = tmp_path / "trend_logs"
    log_dir.mkdir()
    monkeypatch.setenv("I2AS_LOG_DIR", str(log_dir))
    return log_dir


def test_trends_indeterminate_with_no_store_on_disk(
    sim_config, isolated_trend_log_dir: Path, capsys
) -> None:
    """No trend-history files at all: every check is "cannot tell", never a failure."""
    exit_code = cli.main(["trends", "--config", sim_config, "--json"])
    assert exit_code == 0
    payload = _json_out(capsys)
    assert payload["ok"] is True
    names = {r["name"] for r in payload["results"]}
    assert names == {"temperature_temperature_within_band", "trend_store_live"}
    assert all(r["passed"] is None for r in payload["results"])


def test_trends_with_no_declared_checks_runs_only_the_store_liveness_check(
    sim_config_without_trends, isolated_trend_log_dir: Path, capsys
) -> None:
    """A setup that declares no ``trends.checks`` runs none; the CLI-only check remains."""
    exit_code = cli.main(["trends", "--config", sim_config_without_trends, "--json"])
    assert exit_code == 0
    payload = _json_out(capsys)
    assert [r["name"] for r in payload["results"]] == ["trend_store_live"]


def test_trends_fails_when_the_temperature_leaves_its_band(
    sim_config, isolated_trend_log_dir: Path, capsys
) -> None:
    now = time.time()
    records = [
        {"t": now - i * 60, "v": {"temperature_temperature": v}}
        for i, v in enumerate([4.0, 4.6, 400.0, 4.6])
    ]
    _write_trend_jsonl(isolated_trend_log_dir / "trend_history_raw.jsonl", records)

    exit_code = cli.main(["trends", "--config", sim_config, "--json"])

    assert exit_code == 1
    payload = _json_out(capsys)
    assert payload["ok"] is False
    by_name = {r["name"]: r for r in payload["results"]}
    assert by_name["temperature_temperature_within_band"]["passed"] is False
    assert by_name["temperature_temperature_within_band"]["evidence"]["max"] == 400.0


def test_trends_store_live_fails_when_stale(
    sim_config, isolated_trend_log_dir: Path, capsys
) -> None:
    now = time.time()
    stale_t = now - 10_000.0
    path = isolated_trend_log_dir / "trend_history_raw.jsonl"
    path.write_text(
        json.dumps({"t": stale_t, "v": {"temperature_temperature": 4.2}}) + "\n",
        encoding="utf-8",
    )
    os.utime(path, (stale_t, stale_t))

    exit_code = cli.main(["trends", "--config", sim_config, "--json"])

    assert exit_code == 1
    payload = _json_out(capsys)
    by_name = {r["name"]: r for r in payload["results"]}
    assert by_name["trend_store_live"]["passed"] is False


def test_trends_window_override_applies_to_every_declared_check(
    sim_config, isolated_trend_log_dir: Path, capsys
) -> None:
    now = time.time()
    records = [{"t": now - i, "v": {"temperature_temperature": 4.2}} for i in range(5)]
    _write_trend_jsonl(isolated_trend_log_dir / "trend_history_raw.jsonl", records)

    exit_code = cli.main(["trends", "--config", sim_config, "--window", "600", "--json"])

    assert exit_code == 0
    payload = _json_out(capsys)
    by_name = {r["name"]: r for r in payload["results"]}
    assert by_name["temperature_temperature_within_band"]["evidence"]["window_s"] == 600.0


def test_trends_invalid_window_is_a_clean_error(sim_config) -> None:
    with pytest.raises(SystemExit):
        cli.main(["trends", "--config", sim_config, "--window", "nonsense"])


def test_trends_human_output_lists_every_check(
    sim_config, isolated_trend_log_dir: Path, capsys
) -> None:
    exit_code = cli.main(["trends", "--config", sim_config])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "temperature_temperature_within_band" in out
    assert "trend_store_live" in out


def test_trends_human_output_renders_key_summary_evidence_readably(
    sim_config, isolated_trend_log_dir: Path, capsys
) -> None:
    """A "cannot tell" verdict's evidence cites a KeySummary; its fields must render, not a repr.

    With no trend-history files on disk, `no_data_outcome()` stores raw
    `KeySummary` objects in `CheckResult.evidence` (see its docstring), and
    the human CLI path must render their fields inline rather than Python's
    default dataclass `repr()` — see `_render_check_result()`'s docstring.
    """
    exit_code = cli.main(["trends", "--config", sim_config])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "KeySummary(" not in out
    assert "persisted=False" in out


# ── transcript ────────────────────────────────────────────────────────────────


def test_transcript_appends_one_jsonl_line_per_invocation(
    sim_config, isolated_transcript: Path, capsys
) -> None:
    cli.main(["idn", "meter", "--config", sim_config, "--json"])
    cli.main(["read", "meter", "set_range", "1", "--config", sim_config, "--json"])
    lines = (
        (isolated_transcript / "troubleshoot.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["ok"] is True and first["argv"][0] == "idn"
    assert second["ok"] is False and "error" in second["payload"]
    assert "ts" in first


def test_transcript_dir_resolves_through_log_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sim_config, capsys
) -> None:
    """`_transcript_dir()` delegates to log_directory(), so I2AS_LOG_DIR
    steers both the transcript and the `status` command's default log path."""
    monkeypatch.setattr(cli, "_transcript_dir", _REAL_TRANSCRIPT_DIR)
    monkeypatch.setenv("I2AS_LOG_DIR", str(tmp_path))

    cli.main(["idn", "meter", "--config", sim_config, "--json"])

    assert (tmp_path / "troubleshoot.jsonl").exists()
    assert cli._transcript_dir() == tmp_path


def test_qsettings_scope_matches_the_gui_and_the_application_name():
    """Three modules spell the settings scope; they must agree, and be I2AS.

    ``troubleshoot.cli`` mirrors ``gui.app_settings`` because contract C10
    keeps it out of the GUI, and ``main`` names the process the same way so
    the desktop and the settings store show one application.
    """
    from i2as import main as app_main
    from i2as.gui import app_settings

    assert (cli._QSETTINGS_ORG, cli._QSETTINGS_APP) == (
        app_settings._ORGANISATION,
        app_settings._APPLICATION,
    ) == ("I2AS", "I2AS")
    assert app_main.APPLICATION_NAME == app_settings._APPLICATION
