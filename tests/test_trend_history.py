"""Tests for i2as.core.trend_history."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import pytest

from i2as.core.trend_history import (
    TIERS,
    _iter_lines_reverse,
    _order_tier_files_newest_first,
    find_crossings,
    persisted_keys,
    pick_tier,
    read_tier,
    read_window,
    summarize,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write one JSONL record per line to ``path``."""
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _raw_record(t: float, values: dict[str, float]) -> dict:
    return {"t": t, "v": values}


def _bucket_record(t: float, key_stats: dict[str, dict]) -> dict:
    return {"t": t, "n": sum(s["count"] for s in key_stats.values()), "v": key_stats}


def _stats(min_v, max_v, mean, std, count) -> dict:
    return {"min": min_v, "max": max_v, "mean": mean, "std": std, "count": count}


# --------------------------------------------------------------------------
# read_tier: file discovery and merging
# --------------------------------------------------------------------------


def test_read_tier_merges_live_and_rotated_files_oldest_first(tmp_path: Path) -> None:
    live = tmp_path / "trend_history_raw.jsonl"
    rotated = tmp_path / "trend_history_raw.jsonl.2026-07-24"

    now = 2_000_000.0
    _write_jsonl(rotated, [_raw_record(now - 100, {"a": 1.0})])
    _write_jsonl(live, [_raw_record(now - 50, {"a": 2.0}), _raw_record(now, {"a": 3.0})])

    records = read_tier(tmp_path, "raw", window_s=200.0, now=now)

    assert [t for t, _ in records] == [now - 100, now - 50, now]
    assert [v["a"] for _, v in records] == [1.0, 2.0, 3.0]


def test_read_tier_skips_corrupt_truncated_and_blank_lines(tmp_path: Path) -> None:
    live = tmp_path / "trend_history_raw.jsonl"
    now = 2_000_000.0
    good1 = json.dumps(_raw_record(now - 10, {"a": 1.0}))
    good2 = json.dumps(_raw_record(now, {"a": 2.0}))
    corrupt = "{not json at all"
    truncated = json.dumps(_raw_record(now - 5, {"a": 1.5}))[:20]  # cut mid-object
    blank = ""

    live.write_text(
        "\n".join([good1, corrupt, truncated, blank, good2]) + "\n", encoding="utf-8"
    )

    records = read_tier(tmp_path, "raw", window_s=200.0, now=now)

    assert [t for t, _ in records] == [now - 10, now]
    assert [v["a"] for _, v in records] == [1.0, 2.0]


def test_read_tier_excludes_sync_conflict_copy_filenames(tmp_path: Path) -> None:
    live = tmp_path / "trend_history_raw.jsonl"
    conflict = tmp_path / "trend_history_raw-DESKTOP-ABC123.jsonl"
    conflict2 = tmp_path / "trend_history_raw (conflicted copy 2026-07-25).jsonl"

    now = 2_000_000.0
    _write_jsonl(live, [_raw_record(now, {"a": 1.0})])
    _write_jsonl(conflict, [_raw_record(now, {"a": 999.0})])
    _write_jsonl(conflict2, [_raw_record(now, {"a": 888.0})])

    records = read_tier(tmp_path, "raw", window_s=200.0, now=now)

    assert len(records) == 1
    assert records[0][1]["a"] == 1.0


def test_read_tier_missing_dir_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    assert read_tier(missing, "raw", window_s=200.0) == []


def test_read_tier_missing_files_returns_empty(tmp_path: Path) -> None:
    assert read_tier(tmp_path, "raw", window_s=200.0) == []


def test_read_tier_filters_to_window(tmp_path: Path) -> None:
    live = tmp_path / "trend_history_raw.jsonl"
    now = 2_000_000.0
    _write_jsonl(
        live,
        [
            _raw_record(now - 1000, {"a": 1.0}),  # outside window
            _raw_record(now - 50, {"a": 2.0}),  # inside
            _raw_record(now, {"a": 3.0}),  # inside (boundary)
        ],
    )

    records = read_tier(tmp_path, "raw", window_s=100.0, now=now)

    assert [v["a"] for _, v in records] == [2.0, 3.0]


def test_read_tier_partial_result_when_older_files_absent(tmp_path: Path) -> None:
    """A window extending past retention (only some files present) is partial, not an error."""
    live = tmp_path / "trend_history_3min.jsonl"
    now = 2_000_000.0
    _write_jsonl(live, [_bucket_record(now, {"a": _stats(1.0, 1.0, 1.0, 0.0, 1)})])
    # No rotated backups exist at all, even though the window asks for far more.
    records = read_tier(tmp_path, "3min", window_s=999_999.0, now=now)
    assert len(records) == 1


def test_read_tier_stops_scanning_once_past_the_window_lower_bound(tmp_path: Path) -> None:
    """The reverse scan must not read records strictly older than the window needs.

    Regression guard for the defect this fix addresses: read_tier() used to
    read and parse every line of every tier file before applying the window
    filter, so cost scaled with total retention rather than the requested
    window. A file whose oldest records fall outside the window, followed
    by a corrupt line that would raise if ever parsed, proves the scan
    stopped before reaching it.
    """
    live = tmp_path / "trend_history_raw.jsonl"
    now = 2_000_000.0
    lines = [
        "{not json at all — must never be reached by the reverse scan",
        json.dumps(_raw_record(now - 10_000.0, {"a": -1.0})),  # far outside window
        json.dumps(_raw_record(now - 50.0, {"a": 1.0})),  # inside window
        json.dumps(_raw_record(now, {"a": 2.0})),  # inside window
    ]
    live.write_text("\n".join(lines) + "\n", encoding="utf-8")

    records = read_tier(tmp_path, "raw", window_s=100.0, now=now)

    assert [t for t, _ in records] == [now - 50.0, now]
    assert [v["a"] for _, v in records] == [1.0, 2.0]


# --------------------------------------------------------------------------
# Reverse-chunked file reading (the mechanism read_tier() scans backward with)
# --------------------------------------------------------------------------


def test_iter_lines_reverse_yields_newest_line_first(tmp_path: Path) -> None:
    path = tmp_path / "lines.jsonl"
    # newline="" so pathlib writes a literal "\n" on Windows too — this test
    # asserts on raw line text, unlike read_tier() callers, which parse each
    # line through _parse_line()'s .strip() and would not notice a stray \r.
    path.write_text("first\nsecond\nthird\n", encoding="utf-8", newline="")

    assert list(_iter_lines_reverse(path)) == ["third", "second", "first"]


def test_iter_lines_reverse_no_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "lines.jsonl"
    path.write_text("first\nsecond\nthird", encoding="utf-8", newline="")

    assert list(_iter_lines_reverse(path)) == ["third", "second", "first"]


def test_iter_lines_reverse_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert list(_iter_lines_reverse(path)) == []


def test_iter_lines_reverse_missing_file_returns_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jsonl"

    assert list(_iter_lines_reverse(missing)) == []


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 1000])
def test_iter_lines_reverse_reassembles_line_straddling_a_chunk_boundary(chunk_size, tmp_path: Path) -> None:
    """A record split across a tail-read chunk must be reassembled, not dropped or corrupted.

    This is the classic bug in reverse-line readers: with a small
    ``chunk_size``, every record here is guaranteed to straddle at least one
    chunk boundary (each line is far longer than the chunk), yet every line
    must still come back byte-for-byte and in the right (newest-first)
    order.
    """
    lines = ["lineA_with_some_length", "lineB_also_fairly_long", "lineC_the_last_one"]
    path = tmp_path / "lines.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")

    result = list(_iter_lines_reverse(path, chunk_size=chunk_size))

    assert result == list(reversed(lines))


def test_read_tier_record_straddling_a_small_chunk_boundary_is_not_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration-level version of the chunk-boundary guard, through read_tier() itself.

    Forces read_tier()'s actual scan (not just the lower-level line reader)
    to split every record across a chunk boundary, by shrinking the module's
    chunk size to a few bytes, and checks the public contract still holds:
    same records, same ascending order, nothing dropped or corrupted.
    """
    import i2as.core.trend_history as trend_history

    monkeypatch.setattr(trend_history, "_REVERSE_READ_CHUNK_BYTES", 4)

    live = tmp_path / "trend_history_raw.jsonl"
    now = 2_000_000.0
    records_in = [
        _raw_record(now - 2.0, {"a": 1.0, "b": 2.0, "c": 3.0}),
        _raw_record(now - 1.0, {"a": 4.0, "b": 5.0, "c": 6.0}),
        _raw_record(now, {"a": 7.0, "b": 8.0, "c": 9.0}),
    ]
    _write_jsonl(live, records_in)

    records = read_tier(tmp_path, "raw", window_s=100.0, now=now)

    assert [t for t, _ in records] == [now - 2.0, now - 1.0, now]
    assert [v for _, v in records] == [rec["v"] for rec in records_in]


def test_order_tier_files_newest_first(tmp_path: Path) -> None:
    spec = TIERS["raw"]
    live = tmp_path / spec.filename
    older = tmp_path / f"{spec.filename}.2026-07-01"
    newer_rotated = tmp_path / f"{spec.filename}.2026-07-24"
    for p in (live, older, newer_rotated):
        p.write_text("", encoding="utf-8")

    ordered = _order_tier_files_newest_first([older, live, newer_rotated], spec)

    assert ordered == [live, newer_rotated, older]


# --------------------------------------------------------------------------
# pick_tier
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "window_s,expected",
    [
        (86400.0, "raw"),
        (86401.0, "3min"),
        (604800.0, "3min"),
        (604801.0, "hourly"),
    ],
)
def test_pick_tier_boundaries(window_s: float, expected: str) -> None:
    assert pick_tier(window_s) == expected


def test_tiers_table_has_expected_filenames() -> None:
    assert TIERS["raw"].filename == "trend_history_raw.jsonl"
    assert TIERS["3min"].filename == "trend_history_3min.jsonl"
    assert TIERS["hourly"].filename == "trend_history_hourly.jsonl"


# --------------------------------------------------------------------------
# summarize: raw tier
# --------------------------------------------------------------------------


def test_summarize_raw_arithmetic(tmp_path: Path) -> None:
    live = tmp_path / "trend_history_raw.jsonl"
    now = 2_000_000.0
    values = [1.0, 2.0, 3.0, 4.0]
    _write_jsonl(
        live, [_raw_record(now - (len(values) - i) * 1.0, {"a": v}) for i, v in enumerate(values)]
    )

    result = summarize(tmp_path, ["a"], window_s=100.0, now=now)
    s = result["a"]

    expected_mean = sum(values) / len(values)
    expected_std = math.sqrt(sum((v - expected_mean) ** 2 for v in values) / len(values))

    assert s.persisted is True
    assert s.tier == "raw"
    assert s.count == 4
    assert s.min == 1.0
    assert s.max == 4.0
    assert s.mean == pytest.approx(expected_mean)
    assert s.std == pytest.approx(expected_std)


# --------------------------------------------------------------------------
# summarize: aggregate tier, count-weighted mean regression guard
# --------------------------------------------------------------------------


def test_summarize_aggregate_count_weighted_mean_differs_from_naive(tmp_path: Path) -> None:
    """Unequal bucket counts: count-weighted mean must differ from mean-of-means.

    This is the regression guard for the entire reason per-key ``count`` is
    written by TieredTrendLogger: a plain mean-of-means is wrong once
    buckets have unequal sample counts.
    """
    live = tmp_path / "trend_history_3min.jsonl"
    now = 2_000_000.0
    bucket_a = _stats(min_v=0.0, max_v=2.0, mean=1.0, std=1.0, count=10)
    bucket_b = _stats(min_v=4.0, max_v=6.0, mean=5.0, std=1.0, count=100)
    _write_jsonl(
        live,
        [
            _bucket_record(now - 360.0, {"a": bucket_a}),
            _bucket_record(now - 180.0, {"a": bucket_b}),
        ],
    )

    result = summarize(tmp_path, ["a"], window_s=90000.0, now=now)
    s = result["a"]

    naive_mean_of_means = (bucket_a["mean"] + bucket_b["mean"]) / 2  # = 3.0
    expected_weighted_mean = (
        bucket_a["mean"] * bucket_a["count"] + bucket_b["mean"] * bucket_b["count"]
    ) / (bucket_a["count"] + bucket_b["count"])  # = 510/110 = 4.636...

    assert s.persisted is True
    assert s.tier == "3min"
    assert s.count == 110
    assert s.min == 0.0
    assert s.max == 6.0
    assert s.mean == pytest.approx(expected_weighted_mean)
    assert s.mean != pytest.approx(naive_mean_of_means)
    assert expected_weighted_mean == pytest.approx(510.0 / 110.0)


def _population_stats(values: list[float]) -> tuple[float, float, int]:
    """Mean, population std, and count of ``values`` computed directly."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return mean, math.sqrt(variance), n


def test_summarize_aggregate_std_is_exact_via_law_of_total_variance(tmp_path: Path) -> None:
    """Cross-bucket ``std`` recombination is exact, not an approximation.

    Two unequal-size groups of raw values (3 and 7) are pre-aggregated into
    two buckets exactly as TieredTrendLogger would flush them. summarize()
    must reproduce the *true* population std of all 10 raw values combined,
    to full float precision, via the law of total variance recovered from
    each bucket's (mean, std, count) — not the smaller number a naive
    pooled-within-bucket-only estimate (ignoring the spread between bucket
    means) would give.
    """
    group_a = [10.0, 12.0, 14.0]  # 3 values
    group_b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # 7 values
    combined = group_a + group_b

    mean_a, std_a, count_a = _population_stats(group_a)
    mean_b, std_b, count_b = _population_stats(group_b)
    true_mean, true_std, true_count = _population_stats(combined)

    live = tmp_path / "trend_history_3min.jsonl"
    now = 2_000_000.0
    bucket_a = _stats(min(group_a), max(group_a), mean_a, std_a, count_a)
    bucket_b = _stats(min(group_b), max(group_b), mean_b, std_b, count_b)
    _write_jsonl(
        live,
        [
            _bucket_record(now - 360.0, {"a": bucket_a}),
            _bucket_record(now - 180.0, {"a": bucket_b}),
        ],
    )

    result = summarize(tmp_path, ["a"], window_s=90000.0, now=now)
    s = result["a"]

    assert s.count == true_count
    assert s.min == min(combined)
    assert s.max == max(combined)
    assert s.mean == pytest.approx(true_mean, rel=1e-12)
    assert s.std == pytest.approx(true_std, rel=1e-12)

    # Regression guard: the old (superseded) pooled-within-bucket-only
    # estimate ignores the spread *between* bucket means and therefore
    # underestimates the true combined spread whenever those means differ.
    naive_pooled_variance = (count_a * std_a**2 + count_b * std_b**2) / (count_a + count_b)
    naive_pooled_std = math.sqrt(naive_pooled_variance)
    assert naive_pooled_std < s.std
    assert naive_pooled_std != pytest.approx(s.std)


# --------------------------------------------------------------------------
# persisted flag
# --------------------------------------------------------------------------


def test_summarize_persisted_false_for_never_written_key(tmp_path: Path) -> None:
    live = tmp_path / "trend_history_raw.jsonl"
    now = 2_000_000.0
    _write_jsonl(live, [_raw_record(now, {"a": 1.0})])

    result = summarize(tmp_path, ["a", "never_persisted_key"], window_s=100.0, now=now)

    assert result["a"].persisted is True
    assert result["never_persisted_key"].persisted is False
    assert result["never_persisted_key"].count == 0
    assert result["never_persisted_key"].mean is None


def test_summarize_persisted_true_but_empty_window_distinguishes_from_never_written(
    tmp_path: Path,
) -> None:
    live = tmp_path / "trend_history_raw.jsonl"
    now = 2_000_000.0
    # "a" was written, but far outside the requested window.
    _write_jsonl(live, [_raw_record(now - 100_000.0, {"a": 1.0})])

    result = summarize(tmp_path, ["a"], window_s=10.0, now=now)
    s = result["a"]

    assert s.persisted is True  # known to the store...
    assert s.count == 0  # ...but nothing in this window
    assert s.mean is None


def test_persisted_keys_empty_for_missing_dir(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    assert persisted_keys(missing, "raw") == set()


def test_persisted_keys_reads_recent_records(tmp_path: Path) -> None:
    live = tmp_path / "trend_history_raw.jsonl"
    now = 2_000_000.0
    _write_jsonl(live, [_raw_record(now, {"a": 1.0, "b": 2.0})])

    keys = persisted_keys(tmp_path, "raw")

    assert keys == {"a", "b"}


# --------------------------------------------------------------------------
# read_window
# --------------------------------------------------------------------------


def test_read_window_raw_tier_scalar_and_missing_key(tmp_path: Path) -> None:
    live = tmp_path / "trend_history_raw.jsonl"
    now = 2_000_000.0
    _write_jsonl(live, [_raw_record(now, {"a": 1.5})])

    series = read_window(tmp_path, ["a", "missing"], window_s=100.0, now=now)

    assert series["a"] == [(now, 1.5)]
    assert series["missing"] == []


def test_read_window_aggregate_tier_uses_mean(tmp_path: Path) -> None:
    live = tmp_path / "trend_history_3min.jsonl"
    now = 2_000_000.0
    _write_jsonl(live, [_bucket_record(now, {"a": _stats(1.0, 3.0, 2.0, 0.5, 60)})])

    series = read_window(tmp_path, ["a"], window_s=200000.0, now=now)

    assert series["a"] == [(now, 2.0)]


# --------------------------------------------------------------------------
# find_crossings
# --------------------------------------------------------------------------


def _write_raw_series(path: Path, now: float, values: list[float]) -> None:
    records = [_raw_record(now - (len(values) - i) * 1.0, {"a": v}) for i, v in enumerate(values)]
    _write_jsonl(path, records)


def test_find_crossings_below(tmp_path: Path) -> None:
    live = tmp_path / "trend_history_raw.jsonl"
    now = 2_000_000.0
    # 5, 4, 3, 2 (threshold 3): crosses below when value goes from >=3 to <3.
    _write_raw_series(live, now, [5.0, 4.0, 3.0, 2.0])

    crossings = find_crossings(tmp_path, "a", threshold=3.0, window_s=100.0, direction="below", now=now)

    assert len(crossings) == 1
    assert crossings[0] == now - 1.0  # the sample that landed at value 2.0


def test_find_crossings_above(tmp_path: Path) -> None:
    live = tmp_path / "trend_history_raw.jsonl"
    now = 2_000_000.0
    _write_raw_series(live, now, [1.0, 2.0, 3.0, 4.0])

    crossings = find_crossings(tmp_path, "a", threshold=3.0, window_s=100.0, direction="above", now=now)

    assert len(crossings) == 1
    assert crossings[0] == now - 1.0


def test_find_crossings_both(tmp_path: Path) -> None:
    live = tmp_path / "trend_history_raw.jsonl"
    now = 2_000_000.0
    _write_raw_series(live, now, [1.0, 4.0, 1.0, 4.0])

    crossings = find_crossings(tmp_path, "a", threshold=3.0, window_s=100.0, direction="both", now=now)

    assert len(crossings) == 3


def test_find_crossings_invalid_direction_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        find_crossings(tmp_path, "a", threshold=3.0, window_s=100.0, direction="sideways")


def test_find_crossings_missing_dir_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    assert find_crossings(missing, "a", threshold=3.0, window_s=100.0) == []


# --------------------------------------------------------------------------
# Performance regression guard: a check evaluation must not scale with
# tier retention. This pins the requirement that motivated read_tier()'s
# reverse-chunked scan (see its docstring): TrendCheckRunner shares the
# Orchestrator's event loop on a QTimer, and single-threaded cooperative
# scheduling means slow work here delays the next tick — magnet ramps and
# safety polls — so it must never block on the full retained store.
# --------------------------------------------------------------------------


#  Realistic-length flat state keys (f"{vi_name}_{monitored_method}", as
# Station.last_state_flat() actually produces — see e.g.
# a trend check's declared keys), not the
# short synthetic "key_0" style: file size (and therefore I/O cost) scales
# with key-name length too, and a short-name fixture understated it.
_REALISTIC_KEY_NAMES = [
    "temperature_sample_temperature", "temperature_temperature",
    "magnet_z_get_field", "magnet_z_magnet_current", "magnet_z_persistent_switch_state",
    "coolant_monitor_reservoir_level", "coolant_monitor_secondary_level",
    "pressure_gauge_still_pressure", "pressure_gauge_condenser_pressure",
    "source_measure_unit_output_voltage", "source_measure_unit_output_current",
    "lockin_amplifier_x_reading", "lockin_amplifier_y_reading",
    "temperature_probe_secondary_temperature", "flow_controller_mass_flow_rate",
    "heater_power_output_watts", "vacuum_gauge_chamber_pressure",
    "rotator_angle_degrees", "compressor_return_pressure",
    "compressor_water_flow_rate", "turbo_pump_rotation_speed_rpm",
    "ups_battery_charge_percent", "chiller_output_temperature",
    "gas_handling_system_manifold_pressure", "still_heater_power_output",
    "mixing_chamber_heater_power", "sorb_pump_temperature",
    "cold_plate_temperature", "radiation_shield_temperature",
    "sample_rotator_position_degrees",
]


def _build_realistic_raw_tier(log_dir: Path, *, n_keys: int, now: float) -> list[str]:
    """Write a raw tier at the shipped spec: 3 s cadence, 3 days, 3 rotated files.

    Matches ``TIERS["raw"]`` (``interval_s=3.0``, ``retention_s=3 * 86400``):
    86,400 total lines split across a live file and two dated rotated
    siblings, the same file count and layout ``_tier_files()`` and
    ``_order_tier_files_newest_first()`` handle in production.

    Values are drawn from a small pre-serialized pool (97 distinct
    ``"v"``-dicts, cycled) rather than calling ``random.uniform()`` and
    ``json.dumps()`` fresh per line: at 86,400 lines x 30 keys that naive
    approach cost ~4 s of pure Python-level generation, dwarfing the ~0.5 s
    the fixture is meant to measure a check evaluation *against* — an
    expensive fixture that makes the timing assertion below hard to trust
    and slows every run of this test. The pool is prime-sized so it never
    aligns with any periodic structure in the timestamps; what varies line
    to line, and is real, is ``t`` and the file the line lands in — exactly
    what ``read_tier()``'s windowing and file selection scan.

    Returns:
        The ``n_keys`` flat state key names written to every record.
    """
    keys = _REALISTIC_KEY_NAMES[:n_keys]
    n_lines = 86_400
    cadence_s = 3.0
    lines_per_file = n_lines // 3
    rng = random.Random(0)

    pool_size = 97
    value_pool = [
        json.dumps({k: round(rng.uniform(0.0, 100.0), 4) for k in keys}) for _ in range(pool_size)
    ]

    def write_file(path: Path, start_index: int, count: int) -> None:
        lines = []
        for i in range(count):
            idx = start_index + i
            t = now - (n_lines - idx) * cadence_s
            lines.append(f'{{"t": {json.dumps(t)}, "v": {value_pool[idx % pool_size]}}}')
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_file(log_dir / "trend_history_raw.jsonl.2026-08-01", 0, lines_per_file)
    write_file(log_dir / "trend_history_raw.jsonl.2026-08-02", lines_per_file, lines_per_file)
    write_file(
        log_dir / "trend_history_raw.jsonl",
        2 * lines_per_file,
        n_lines - 2 * lines_per_file,
    )
    return keys


# Budget for one check's worth of evaluation (summarize() + read_window(),
# exactly what trend_checks.run_check() does) over a 1 h window on the
# 86,400-line, 30-keys/record fixture below — the worst-case key count this
# repository's synthetic-store measurements have used. Key names are
# realistic-length flat state keys (see _REALISTIC_KEY_NAMES), not short
# placeholders: name length feeds file size and was part of the original
# measurement, so a fixture using short names would understate the cost.
#
# Measured on THIS reverse-scan implementation (both read_tier() and
# persisted_keys() windowed): ~220 ms at 10 keys/record, ~460 ms at 30
# keys/record.
#
# Measured on the two full-file-parse defects this branch fixed, at 30
# keys/record:
#   - read_tier() alone unfixed (persisted_keys() fixed): the original
#     regression this branch exists for — every line of every raw-tier file
#     parsed before the window filter — 3.0-4.7 s (see the module's other
#     history; not independently re-measured here since it dominates by a
#     wide margin regardless).
#   - persisted_keys() alone unfixed (read_tier() fixed): a sibling defect
#     found in review — read_text().splitlines() materialised the whole
#     file before its own 500-line parse cap applied — 687 ms combined
#     summarize()+read_window() (vs. 458 ms fixed).
#
# 1000 ms sits above the fixed implementation with >2x headroom (absorbing
# slow-CI variance) and below the dominant read_tier regression by 3-4.7x,
# so a revert of the primary defect fails unambiguously; it also sits below
# the persisted_keys()-only regression (687 ms), though that narrower
# margin makes it the harder of the two failure modes to guarantee catching
# under heavy CI noise — the budget is chosen so the common, larger
# regression is never missed, not for micro-benchmark precision.
_CHECK_EVALUATION_BUDGET_S = 1.0


def test_check_evaluation_does_not_scale_with_tier_retention(tmp_path: Path) -> None:
    """A 1 h check evaluation must cost roughly the window, not the 3-day retention.

    Regression guard for two sibling defects this branch fixed: read_tier()
    used to parse every line of every raw-tier file before applying the
    window filter, and persisted_keys() used to materialise a whole file
    with read_text().splitlines() before applying its own line-count cap —
    both scaling with total retention rather than the request. See
    ``_CHECK_EVALUATION_BUDGET_S``'s comment for the measured numbers this
    budget is calibrated against. Evaluating a 1 h window should now touch
    roughly the ~1,200 lines the window needs (plus persisted_keys()'s
    separately bounded tail scan), not all 86,400, and stay well under
    budget.
    """
    from i2as.core.trend_history import read_window, summarize

    now = 2_000_000.0
    keys = _build_realistic_raw_tier(tmp_path, n_keys=30, now=now)

    start = time.perf_counter()
    summarize(tmp_path, keys, window_s=3600.0, now=now)
    read_window(tmp_path, keys, window_s=3600.0, now=now)
    elapsed = time.perf_counter() - start

    assert elapsed < _CHECK_EVALUATION_BUDGET_S, (
        f"check evaluation over a 1 h window took {elapsed:.3f} s against an "
        f"86,400-line raw tier; budget is {_CHECK_EVALUATION_BUDGET_S} s — "
        f"this smells like a regression to a full-tier-file parse instead of "
        f"read_tier()'s reverse windowed scan"
    )
