"""Tests for i2as.core.trend_checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from i2as.core.conditions import Condition
from i2as.core.trend_checks import (
    CheckOutcome,
    CheckResult,
    TrendCheck,
    channel_within_band,
    conditions_for,
    declared_checks,
    no_data_outcome,
    run_check,
    run_checks,
    to_condition,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _raw_record(t: float, values: dict[str, float]) -> dict:
    return {"t": t, "v": values}


# One trivial throwaway predicate/check, exercising the run_check()/
# run_checks() machinery generically — the shipped check kind
# (channel_within_band) gets its own dedicated predicate tests below.
def _stability_predicate(key: str, std_limit: float):
    def predicate(summaries, series, window_s):
        del series, window_s  # unused: this fixture predicate only needs the summary
        no_data = no_data_outcome(summaries, [key])
        if no_data is not None:
            return no_data
        summary = summaries[key]
        evidence = {"std": summary.std, "count": summary.count}
        if summary.std <= std_limit:
            return CheckOutcome(
                True, f"std={summary.std:.3g} <= {std_limit} over {summary.count} samples", evidence
            )
        return CheckOutcome(
            False, f"std={summary.std:.3g} > {std_limit} over {summary.count} samples", evidence
        )

    return predicate


def _fixture_check(
    name: str = "temp_stable", key: str = "temp", std_limit: float = 0.5, severity: str = "advisory"
) -> TrendCheck:
    return TrendCheck(
        name=name,
        keys=(key,),
        window_s=3600.0,
        severity=severity,
        predicate=_stability_predicate(key, std_limit),
    )


# ── TrendCheck validation ───────────────────────────────────────────────────


def test_trend_check_rejects_empty_name():
    with pytest.raises(ValueError, match="name"):
        TrendCheck(name="", keys=("a",), window_s=1.0, severity="advisory", predicate=lambda s, sr, w: None)


def test_trend_check_rejects_empty_keys():
    with pytest.raises(ValueError, match="keys"):
        TrendCheck(name="x", keys=(), window_s=1.0, severity="advisory", predicate=lambda s, sr, w: None)


def test_trend_check_rejects_non_positive_window():
    with pytest.raises(ValueError, match="window_s"):
        TrendCheck(name="x", keys=("a",), window_s=0.0, severity="advisory", predicate=lambda s, sr, w: None)


def test_trend_check_rejects_unknown_severity():
    with pytest.raises(ValueError, match="severity"):
        TrendCheck(name="x", keys=("a",), window_s=1.0, severity="urgent", predicate=lambda s, sr, w: None)


# ── no_data_outcome ──────────────────────────────────────────────────────────


def test_no_data_outcome_none_when_data_present(tmp_path: Path):
    now = 2_000_000.0
    _write_jsonl(tmp_path / "trend_history_raw.jsonl", [_raw_record(now, {"temp": 1.0})])
    check = _fixture_check()
    result = run_check(check, tmp_path, now=now)
    assert result.passed is not None


def test_no_data_outcome_never_persisted_is_indeterminate(tmp_path: Path):
    # No files at all: the key was never persisted.
    check = _fixture_check()
    result = run_check(check, tmp_path, now=2_000_000.0)
    assert result.passed is None
    assert "never persisted" in result.message


def test_no_data_outcome_persisted_but_empty_window_is_indeterminate(tmp_path: Path):
    now = 2_000_000.0
    # The key is persisted (appears in the tier's files) but has no samples
    # inside the requested window — write one sample far outside window_s.
    _write_jsonl(
        tmp_path / "trend_history_raw.jsonl", [_raw_record(now - 10_000_000.0, {"temp": 1.0})]
    )
    check = _fixture_check()
    result = run_check(check, tmp_path, now=now)
    assert result.passed is None
    assert "no samples" in result.message


# ── run_check / run_checks ───────────────────────────────────────────────────


def test_run_check_passes_for_stable_data(tmp_path: Path):
    now = 2_000_000.0
    records = [_raw_record(now - i, {"temp": 10.0}) for i in range(5)]
    _write_jsonl(tmp_path / "trend_history_raw.jsonl", records)
    check = _fixture_check(std_limit=0.5)
    result = run_check(check, tmp_path, now=now)
    assert result.passed is True
    assert result.name == "temp_stable"
    assert result.evidence["count"] == 5


def test_run_check_fails_for_unstable_data(tmp_path: Path):
    now = 2_000_000.0
    records = [_raw_record(now - i, {"temp": v}) for i, v in enumerate([0.0, 20.0, 0.0, 20.0])]
    _write_jsonl(tmp_path / "trend_history_raw.jsonl", records)
    check = _fixture_check(std_limit=0.5)
    result = run_check(check, tmp_path, now=now)
    assert result.passed is False


def test_run_checks_evaluates_every_check_uniformly(tmp_path: Path):
    now = 2_000_000.0
    _write_jsonl(
        tmp_path / "trend_history_raw.jsonl",
        [_raw_record(now, {"a": 1.0, "b": 100.0})],
    )
    checks = [_fixture_check(name="a_stable", key="a"), _fixture_check(name="b_stable", key="b")]
    results = run_checks(checks, tmp_path, now=now)
    assert [r.name for r in results] == ["a_stable", "b_stable"]


# ── to_condition / conditions_for ────────────────────────────────────────────


def test_to_condition_none_for_passing_result():
    check = _fixture_check()
    result = CheckResult(name=check.name, passed=True, message="ok", evidence={})
    assert to_condition(check, result, since=100.0) is None


def test_to_condition_none_for_indeterminate_result():
    check = _fixture_check()
    result = CheckResult(name=check.name, passed=None, message="cannot tell", evidence={})
    assert to_condition(check, result, since=100.0) is None


def test_to_condition_builds_advisory_condition_for_failure():
    check = _fixture_check(name="temp_stable")
    result = CheckResult(name="temp_stable", passed=False, message="unstable", evidence={"std": 9.0})
    condition = to_condition(check, result, since=100.0)
    assert isinstance(condition, Condition)
    assert condition.key == "trend:temp_stable"
    assert condition.origin == "trend"
    assert condition.severity == "advisory"
    assert condition.affected_vis is None
    assert condition.message == "unstable"
    assert condition.since == 100.0


def test_conditions_for_only_includes_failures(tmp_path: Path):
    now = 2_000_000.0
    _write_jsonl(
        tmp_path / "trend_history_raw.jsonl",
        [_raw_record(now - i, {"stable": 5.0, "unstable": v}) for i, v in enumerate([0.0, 50.0])],
    )
    checks = [
        _fixture_check(name="stable_check", key="stable"),
        _fixture_check(name="unstable_check", key="unstable"),
    ]
    results = run_checks(checks, tmp_path, now=now)
    conditions = conditions_for(checks, results, since=now)
    assert [c.key for c in conditions] == ["trend:unstable_check"]


# ── declared_checks: built from the config's trends.checks list ──────────────

# A fully-merged trends: config, mirroring what config.read_trends_config()
# actually hands declared_checks() at runtime.
_BAND_ENTRY = {"key": "temperature_temperature", "low": 1.0, "high": 320.0, "window_s": 3600.0}
_FULL_TRENDS_CONFIG: dict[str, object] = {
    "refresh_interval_s": 60.0,
    "store_live_stale_ticks": 10.0,
    "checks": [_BAND_ENTRY],
}


def test_declared_checks_builds_one_check_per_config_entry():
    checks = declared_checks(_FULL_TRENDS_CONFIG)
    assert [c.name for c in checks] == ["temperature_temperature_within_band"]
    assert checks[0].keys == ("temperature_temperature",)
    assert checks[0].window_s == 3600.0
    assert all(c.severity == "advisory" for c in checks)


def test_declared_checks_is_empty_when_the_config_declares_none():
    assert declared_checks({}) == ()
    assert declared_checks({"checks": []}) == ()


def test_declared_checks_honours_name_kind_and_severity_overrides():
    entry = {**_BAND_ENTRY, "kind": "channel_within_band", "name": "sample_ok", "severity": "hold"}
    (check,) = declared_checks({"checks": [entry]})
    assert check.name == "sample_ok"
    assert check.severity == "hold"


def test_declared_checks_preserves_declaration_order():
    second = {**_BAND_ENTRY, "key": "magnet_z_magnet_field_T", "low": -9.0, "high": 9.0}
    checks = declared_checks({"checks": [_BAND_ENTRY, second]})
    assert [c.keys[0] for c in checks] == ["temperature_temperature", "magnet_z_magnet_field_T"]


@pytest.mark.parametrize(
    "entry, match",
    [
        ("not-a-mapping", "must be a mapping"),
        ({**_BAND_ENTRY, "kind": "no_such_kind"}, "unknown kind"),
        ({**_BAND_ENTRY, "typo": 1.0}, "unknown field"),
        ({"key": "temperature_temperature", "low": 1.0, "window_s": 60.0}, "high"),
        ({**_BAND_ENTRY, "low": 5.0, "high": 5.0}, "low must be below high"),
        ({**_BAND_ENTRY, "window_s": 0.0}, "window_s must be positive"),
    ],
    ids=["not-mapping", "unknown-kind", "unknown-field", "missing-high", "inverted-band", "zero-window"],
)
def test_declared_checks_refuses_a_malformed_entry(entry, match):
    """A malformed check is a config error worth stopping on, not a check that never runs."""
    with pytest.raises(ValueError, match=match):
        declared_checks({"checks": [entry]})


# ── channel_within_band ──────────────────────────────────────────────────────


def _band_check(**overrides) -> TrendCheck:
    return channel_within_band(**{**_BAND_ENTRY, **overrides})


def test_channel_within_band_passes_for_readings_inside_the_band(tmp_path: Path):
    now = 2_000_000.0
    records = [
        _raw_record(now - i, {"temperature_temperature": 4.2}) for i in range(0, 3600, 100)
    ]
    _write_jsonl(tmp_path / "trend_history_raw.jsonl", records)
    result = run_check(_band_check(), tmp_path, now=now)
    assert result.passed is True
    assert result.evidence["min"] == pytest.approx(4.2)
    assert result.evidence["max"] == pytest.approx(4.2)
    assert result.evidence["count"] == len(records)
    assert result.evidence["window_s"] == 3600.0
    assert "stayed within" in result.message


def test_channel_within_band_fails_when_the_maximum_leaves_the_band(tmp_path: Path):
    now = 2_000_000.0
    values = [4.0, 4.6, 400.0, 4.6, 4.0]
    records = [
        _raw_record(now - i * 60, {"temperature_temperature": v}) for i, v in enumerate(values)
    ]
    _write_jsonl(tmp_path / "trend_history_raw.jsonl", records)
    result = run_check(_band_check(), tmp_path, now=now)
    assert result.passed is False
    assert result.evidence["max"] == pytest.approx(400.0)
    assert result.evidence["high"] == 320.0
    assert "left the band" in result.message


def test_channel_within_band_fails_when_the_minimum_leaves_the_band(tmp_path: Path):
    now = 2_000_000.0
    records = [_raw_record(now - i * 60, {"temperature_temperature": v}) for i, v in enumerate([4.0, 0.5])]
    _write_jsonl(tmp_path / "trend_history_raw.jsonl", records)
    result = run_check(_band_check(), tmp_path, now=now)
    assert result.passed is False
    assert result.evidence["min"] == pytest.approx(0.5)


def test_channel_within_band_bounds_are_inclusive(tmp_path: Path):
    now = 2_000_000.0
    records = [_raw_record(now - i * 60, {"temperature_temperature": v}) for i, v in enumerate([1.0, 320.0])]
    _write_jsonl(tmp_path / "trend_history_raw.jsonl", records)
    assert run_check(_band_check(), tmp_path, now=now).passed is True


def test_channel_within_band_indeterminate_on_empty_window(tmp_path: Path):
    result = run_check(_band_check(), tmp_path, now=2_000_000.0)
    assert result.passed is None


def test_channel_within_band_reports_the_window_actually_queried(tmp_path: Path):
    """A caller overriding window_s gets evidence citing that window, not the declared one."""
    import dataclasses

    now = 2_000_000.0
    _write_jsonl(
        tmp_path / "trend_history_raw.jsonl", [_raw_record(now, {"temperature_temperature": 4.2})]
    )
    check = dataclasses.replace(_band_check(), window_s=600.0)
    result = run_check(check, tmp_path, now=now)
    assert result.passed is True
    assert result.evidence["window_s"] == 600.0


def test_channel_within_band_default_name_and_overrides():
    assert _band_check().name == "temperature_temperature_within_band"
    assert _band_check(name="sample_ok", severity="advisory").name == "sample_ok"
    with pytest.raises(ValueError, match="low must be below high"):
        channel_within_band("k", 2.0, 1.0, 60.0)
