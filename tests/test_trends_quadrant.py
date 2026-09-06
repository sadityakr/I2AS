"""Tests for TrendsQuadrant — startup rehydration from the raw trend-history tier."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from i2as.core.station import build_station
from i2as.gui.trends_quadrant import TrendsQuadrant

CONFIG_PATH = "i2as/configs/sim_cryostat"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write one JSONL record per line to ``path`` (matches trend_history's on-disk format)."""
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _raw_record(t: float, values: dict[str, float]) -> dict:
    return {"t": t, "v": values}


@pytest.fixture
def station():
    """Real simulated station from sim_cryostat config."""
    return build_station(CONFIG_PATH)


# ── Successful rehydration ───────────────────────────────────────────────────

def test_rehydrates_history_from_raw_tier(qtbot, station, tmp_path):
    """Constructing with a raw-tier JSONL present replays its records into history.

    Keys land in history exactly as they were written: rehydration is a
    verbatim replay of the store, never a rewriting of it.
    """
    now = time.time()
    key = "magnet_z_magnet_field_T"
    _write_jsonl(
        tmp_path / "trend_history_raw.jsonl",
        [
            _raw_record(now - 20, {key: 1.0}),
            _raw_record(now - 10, {key: 1.5}),
            _raw_record(now, {key: 2.0}),
        ],
    )

    quadrant = TrendsQuadrant(station, log_dir=tmp_path)
    qtbot.addWidget(quadrant)

    assert key in quadrant.history.keys()
    times, values = quadrant.history.series(key)
    assert values == [1.0, 1.5, 2.0]
    assert times == [now - 20, now - 10, now]


def test_rehydration_is_oldest_first_regardless_of_file_order(qtbot, station, tmp_path):
    """Records replay oldest-first, matching read_tier's contract."""
    now = time.time()
    key = "level_he_level_pct"
    rotated = tmp_path / "trend_history_raw.jsonl.2026-07-24"
    live = tmp_path / "trend_history_raw.jsonl"
    _write_jsonl(rotated, [_raw_record(now - 3600, {key: 70.0})])
    _write_jsonl(live, [_raw_record(now, {key: 65.0})])

    quadrant = TrendsQuadrant(station, log_dir=tmp_path)
    qtbot.addWidget(quadrant)

    times, values = quadrant.history.series(key)
    assert values == [70.0, 65.0]
    assert times[0] < times[1]


# ── Non-fatal degradation: missing / corrupt store ───────────────────────────

def test_missing_log_dir_leaves_usable_empty_history(qtbot, station, tmp_path):
    """A log_dir that does not exist must not raise, and leaves an empty, usable history."""
    missing_dir = tmp_path / "does_not_exist"

    quadrant = TrendsQuadrant(station, log_dir=missing_dir)
    qtbot.addWidget(quadrant)

    assert quadrant.history.keys() == []
    # Usable: recording into it afterwards still works normally.
    quadrant.history.record({"magnet_z": {"get_field": 0.5}}, timestamp=time.time())
    assert "magnet_z_get_field" in quadrant.history.keys()


def test_corrupt_raw_file_leaves_usable_empty_history(qtbot, station, tmp_path):
    """A corrupt raw-tier file must not raise; corrupt lines are skipped, not fatal."""
    (tmp_path / "trend_history_raw.jsonl").write_text(
        "{not valid json at all\n{\"t\": \"also not a number\", \"v\": {}}\n",
        encoding="utf-8",
    )

    quadrant = TrendsQuadrant(station, log_dir=tmp_path)
    qtbot.addWidget(quadrant)

    assert quadrant.history.keys() == []


def test_rehydration_failure_is_caught_and_logged(qtbot, station, tmp_path, monkeypatch, caplog):
    """Any exception during rehydration is caught, logged, and never propagates."""
    import i2as.gui.trends_quadrant as trends_quadrant_module

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated rehydration failure")

    monkeypatch.setattr(trends_quadrant_module.trend_history, "read_tier", _boom)

    with caplog.at_level("ERROR"):
        quadrant = TrendsQuadrant(station, log_dir=tmp_path)
    qtbot.addWidget(quadrant)

    assert quadrant.history.keys() == []
    assert any("rehydration" in rec.message.lower() for rec in caplog.records)
