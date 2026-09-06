"""Tests for the deterministic per-VI stall detector."""

from __future__ import annotations

from i2as.core.operational_status import build_operational_status
from i2as.core.stall_detection import (
    StallConfig,
    StallState,
    apply_stall_verdict,
    no_motion_phases_from,
)


def _run_ticks(
    values, target, rate, *, phase=None, ramp_status="RAMPING", config=None,
    no_motion_phases=frozenset(),
):
    """Feed measured values through build + stall detection; return the per-tick records.

    One system VI "m" ramping toward *target*; each entry in *values* is its
    measured value on that tick. prev_gaps and StallState are threaded across
    ticks exactly as the Orchestrator does, and the VI's ``no_motion_phases``
    declaration rides in the ramp snapshot exactly as
    ``Station.get_ramp_status()`` carries it.
    """
    prev_gaps: dict[str, float] = {}
    stall_state = StallState()
    records = []
    for v in values:
        ramp_info = {
            "m": {"value": v, "target": target, "rate": rate,
                  "ramp_status": ramp_status, "phase": phase,
                  "no_motion_phases": no_motion_phases},
        }
        record, prev_gaps = build_operational_status(
            orch_state="RAMPING", elapsed_in_state_s=1.0, state={"m": {}},
            ramp_info=ramp_info, prev_gaps=prev_gaps,
        )
        record, stall_state = apply_stall_verdict(
            record, stall_state, config, no_motion_phases=no_motion_phases_from(ramp_info)
        )
        records.append(record)
    return records


def test_converging_ramp_never_stalls():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]  # gap shrinks each tick
    records = _run_ticks(values, target=10.0, rate=1.0)
    assert all(r["verdict"] == "OK" for r in records)
    assert all(not r["alerts"] for r in records)


def test_flat_ramp_stalls_after_threshold():
    values = [5.0] * 9  # gap frozen: tick0 has no delta, then 8 non-closing ticks
    records = _run_ticks(values, target=10.0, rate=1.0)
    # Below threshold (default stall_seconds=18.0 / tick_interval_ms=3000 -> 6 ticks):
    # still quiet at the 4th record.
    assert records[3]["verdict"] == "OK"
    assert not records[3]["alerts"]
    # Past threshold: stalled, with a per-VI code and a human alert.
    assert records[-1]["verdict"] == "RAMP_STALLED"
    stalled = records[-1]["vis"][0]
    assert stalled["code"] == "RAMP_STALLED"
    assert "not closing" in stalled["detail"]
    assert any("stalled" in a for a in records[-1]["alerts"])


def test_declared_no_motion_phase_never_stalls():
    """A phase the VI declares in ``no_motion_phases`` is an expected pause."""
    values = [5.0] * 12  # frozen gap, but the value is meant to hold in this phase
    records = _run_ticks(
        values, target=10.0, rate=1.0, phase="warmup", no_motion_phases=frozenset({"warmup"})
    )
    assert all(r["verdict"] == "OK" for r in records)
    assert all(not r["alerts"] for r in records)


def test_undeclared_phase_is_judged_like_any_other():
    """The same phase name with no declaration behind it is a stall.

    The detector knows no instrument: "warmup" means nothing until a VI
    declares it, so a VI that did not declare it is judged on progress.
    """
    values = [5.0] * 12
    records = _run_ticks(values, target=10.0, rate=1.0, phase="warmup")
    assert records[-1]["verdict"] == "RAMP_STALLED"


def test_no_motion_phases_from_reads_the_snapshot():
    snapshot = {
        "a": {"no_motion_phases": frozenset({"warmup"})},
        "b": {"no_motion_phases": ()},
        "c": {},
    }
    assert no_motion_phases_from(snapshot) == {
        "a": frozenset({"warmup"}), "b": frozenset(), "c": frozenset(),
    }


def test_reached_ramp_not_flagged():
    values = [10.0] * 12  # at target, not RAMPING
    records = _run_ticks(values, target=10.0, rate=1.0, ramp_status="TARGET_REACHED")
    assert all(r["verdict"] == "OK" for r in records)


def test_progress_resets_the_stall_counter():
    # Flat for a few ticks, then one real step of progress, then flat again —
    # the counter resets, so it never reaches the threshold.
    values = [5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0]
    records = _run_ticks(values, target=10.0, rate=1.0)
    assert all(r["verdict"] == "OK" for r in records)


def test_seconds_to_ticks_conversion_matches_wall_clock_across_tick_intervals():
    # Regression guard for the units bug: the same stall_seconds must trigger
    # at (approximately) the same wall-clock elapsed time regardless of the
    # setup's tick_interval_ms, even though the tick COUNT to get there
    # necessarily differs (6 ticks at 1000 ms vs 2 ticks at 3000 ms).
    fast_config = StallConfig(stall_seconds=6.0, tick_interval_ms=1000)
    slow_config = StallConfig(stall_seconds=6.0, tick_interval_ms=3000)
    assert fast_config.stall_ticks == 6
    assert slow_config.stall_ticks == 2

    values = [5.0] * 9  # frozen gap throughout

    fast_records = _run_ticks(values, target=10.0, rate=1.0, config=fast_config)
    slow_records = _run_ticks(values, target=10.0, rate=1.0, config=slow_config)

    fast_stall_tick = next(i for i, r in enumerate(fast_records) if r["verdict"] == "RAMP_STALLED")
    slow_stall_tick = next(i for i, r in enumerate(slow_records) if r["verdict"] == "RAMP_STALLED")

    fast_wall_clock_s = fast_stall_tick * (fast_config.tick_interval_ms / 1000.0)
    slow_wall_clock_s = slow_stall_tick * (slow_config.tick_interval_ms / 1000.0)
    assert abs(fast_wall_clock_s - slow_wall_clock_s) <= max(
        fast_config.tick_interval_ms, slow_config.tick_interval_ms
    ) / 1000.0


def test_seconds_to_ticks_conversion_floors_at_one_tick():
    # A tick interval longer than the configured stall_seconds must still
    # yield at least one tick, not zero (which would never trigger).
    config = StallConfig(stall_seconds=1.0, tick_interval_ms=3000)
    assert config.stall_ticks == 1
