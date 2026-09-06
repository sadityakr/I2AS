"""Tests for i2as.core.trend_check_runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from i2as.core.conditions import decide
from i2as.core.station import Station, build_station
from i2as.core.trend_check_runner import TrendCheckRunner
from i2as.core.trend_checks import CheckOutcome, TrendCheck


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _raw_record(t: float, values: dict[str, float]) -> dict:
    return {"t": t, "v": values}


def _fixed_verdict_check(name: str, passed: bool) -> TrendCheck:
    """A check whose predicate ignores its data and always returns *passed*.

    Deliberately not exercising real data-driven judgement here — that is
    `test_trend_checks.py`'s job — this module tests the SCHEDULING and
    PUBLICATION path, so a fixed verdict isolates it from summarize()'s
    behaviour.
    """
    return TrendCheck(
        name=name,
        keys=("temperature_sample_temperature",),
        window_s=3600.0,
        severity="advisory",
        predicate=lambda summaries, series, window_s: CheckOutcome(
            passed, f"fixed verdict: {passed}", {}
        ),
    )


@pytest.fixture
def sim_station() -> Station:
    config_path = Path(__file__).parent.parent / "i2as" / "configs" / "sim_cryostat"
    return build_station(str(config_path))


def test_run_once_with_no_checks_is_a_no_op(sim_station: Station, qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("I2AS_LOG_DIR", str(tmp_path))
    runner = TrendCheckRunner(sim_station, [], refresh_interval_s=60.0)
    runner.run_once()
    assert sim_station.conditions() == {}
    runner.stop()


def test_run_once_publishes_advisory_condition_for_a_failing_check(
    sim_station: Station, qtbot, tmp_path, monkeypatch
):
    monkeypatch.setenv("I2AS_LOG_DIR", str(tmp_path))
    checks = [_fixed_verdict_check("always_fails", passed=False)]
    runner = TrendCheckRunner(sim_station, checks, refresh_interval_s=60.0)

    runner.run_once()

    conditions = sim_station.conditions()
    assert "trend:always_fails" in conditions
    condition = conditions["trend:always_fails"]
    assert condition.origin == "trend"
    assert condition.severity == "advisory"
    runner.stop()


def test_advisory_trend_condition_leaves_decide_verdict_empty(
    sim_station: Station, qtbot, tmp_path, monkeypatch
):
    """The required proof that advisory really means no enforcement.

    A failing trend check reaches Station.conditions() as an advisory
    Condition, and decide() over the full condition set leaves held_vis
    empty, emergency empty, and run_failure None for it — even when the
    watched VI is exactly the one the check concerns.
    """
    monkeypatch.setenv("I2AS_LOG_DIR", str(tmp_path))
    checks = [_fixed_verdict_check("sample_temperature_stable", passed=False)]
    runner = TrendCheckRunner(sim_station, checks, refresh_interval_s=60.0)

    runner.run_once()
    assert "trend:sample_temperature_stable" in sim_station.conditions()

    verdict = decide(
        sim_station.conditions().values(),
        watched_vis={"temperature"},
        run_active=True,
    )
    assert verdict.held_vis == {}
    assert verdict.emergency == ()
    assert verdict.run_failure is None
    runner.stop()


def test_run_once_clears_condition_once_check_passes(sim_station: Station, qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("I2AS_LOG_DIR", str(tmp_path))
    failing = [_fixed_verdict_check("flaky", passed=False)]
    runner = TrendCheckRunner(sim_station, failing, refresh_interval_s=60.0)
    runner.run_once()
    assert "trend:flaky" in sim_station.conditions()

    # Swap in a check with the same name that now passes — the runner's
    # published set is a complete refresh, not a delta, so the condition
    # clears the moment its cause clears.
    runner._checks = (_fixed_verdict_check("flaky", passed=True),)
    runner.run_once()
    assert "trend:flaky" not in sim_station.conditions()
    runner.stop()


def test_trend_refresh_does_not_disturb_safety_conditions(
    sim_station: Station, qtbot, tmp_path, monkeypatch
):
    """The 60 s trend cadence and the per-tick safety cadence never wipe each other."""
    monkeypatch.setenv("I2AS_LOG_DIR", str(tmp_path))
    from tests import scenarios

    hold_flag = scenarios.declare_hold_flag(monkeypatch, sim_station)
    hold_flag.trip()
    safety = sim_station.check_safety(sim_station.get_state())
    assert safety[hold_flag.flag] is True
    sim_station.update_conditions(safety, tolerated_flags=frozenset())
    assert hold_flag.key in sim_station.conditions()

    checks = [_fixed_verdict_check("always_fails", passed=False)]
    runner = TrendCheckRunner(sim_station, checks, refresh_interval_s=60.0)
    runner.run_once()

    assert hold_flag.key in sim_station.conditions()
    assert "trend:always_fails" in sim_station.conditions()

    # The per-tick safety refresh must not wipe the trend condition either.
    sim_station.update_conditions(safety, tolerated_flags=frozenset())
    assert "trend:always_fails" in sim_station.conditions()
    runner.stop()
