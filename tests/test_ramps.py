"""Unit tests for the RampRecord value object and build_ramp_records()."""

from __future__ import annotations

import pytest

from i2as.core.ramps import ACTIVE_RAMP_STATUS, RampRecord, build_ramp_records


def _meta(vi_name: str) -> tuple[str, str]:
    """Stand-in for Station.system_setpoint_meta."""
    return {
        "magnet_z": ("field", "T"),
        "temperature_sample": ("temperature", "K"),
    }.get(vi_name, (vi_name, ""))


def _always_stoppable(vi_name: str) -> tuple[bool, str]:
    """Stand-in admission predicate that admits everything."""
    return True, ""


def _never_stoppable(vi_name: str) -> tuple[bool, str]:
    """Stand-in admission predicate that refuses everything, with a reason."""
    return False, f"Cannot control {vi_name}: claimed by running procedure 'X'"


def _snapshot(**overrides) -> dict[str, dict]:
    """One ramping magnet, with per-field overrides."""
    entry = {
        "value": 0.25,
        "setpoint": 4.0,
        "target": 5.0,
        "rate": 0.5,
        "ramp_status": ACTIVE_RAMP_STATUS,
        "phase": None,
    }
    entry.update(overrides)
    return {"magnet_z": entry}


# ── Filtering: only a RAMPING VI is a running ramp ────────────────────────────


def test_ramping_vi_becomes_a_record():
    records = build_ramp_records(
        _snapshot(), setpoint_meta=_meta, stop_policy=_always_stoppable
    )
    assert len(records) == 1
    record = records[0]
    assert record.vi_name == "magnet_z"
    assert record.label == "field"
    assert record.unit == "T"
    assert record.value == pytest.approx(0.25)
    assert record.setpoint == pytest.approx(4.0)
    assert record.target == pytest.approx(5.0)
    assert record.rate == pytest.approx(0.5)


@pytest.mark.parametrize("status", ["IDLE", "TARGET_REACHED", ""])
def test_non_ramping_vi_is_dropped(status):
    """A ramp that has arrived, or was never started, is not a running ramp."""
    records = build_ramp_records(
        _snapshot(ramp_status=status),
        setpoint_meta=_meta,
        stop_policy=_always_stoppable,
    )
    assert records == []


def test_records_are_ordered_by_vi_name():
    snapshot = {
        name: {
            "value": 1.0,
            "setpoint": 2.0,
            "target": 3.0,
            "rate": 1.0,
            "ramp_status": ACTIVE_RAMP_STATUS,
            "phase": None,
        }
        for name in ("temperature_sample", "magnet_z", "rotator")
    }
    records = build_ramp_records(
        snapshot, setpoint_meta=_meta, stop_policy=_always_stoppable
    )
    assert [r.vi_name for r in records] == ["magnet_z", "rotator", "temperature_sample"]


# ── Next setpoint vs end setpoint ─────────────────────────────────────────────


def test_setpoint_and_target_are_carried_separately():
    """The whole point of the tracker: NEXT setpoint != END setpoint mid-ramp."""
    record = build_ramp_records(
        _snapshot(setpoint=4.0, target=5.0),
        setpoint_meta=_meta,
        stop_policy=_always_stoppable,
    )[0]
    assert record.setpoint != record.target


def test_missing_introspection_hooks_degrade_to_none():
    """A VI exposing no setpoint/rate yields None, never a KeyError."""
    record = build_ramp_records(
        {"rotator": {"ramp_status": ACTIVE_RAMP_STATUS}},
        setpoint_meta=_meta,
        stop_policy=_always_stoppable,
    )[0]
    assert record.value is None
    assert record.setpoint is None
    assert record.target is None
    assert record.rate is None
    assert record.phase is None
    assert record.label == "rotator"
    assert record.unit == ""


def test_non_numeric_reading_degrades_to_none():
    """A misbehaving VI's junk value must not raise into the tick."""
    record = build_ramp_records(
        _snapshot(value="not a number"),
        setpoint_meta=_meta,
        stop_policy=_always_stoppable,
    )[0]
    assert record.value is None


def test_stale_entry_is_flagged():
    record = build_ramp_records(
        _snapshot(_stale=True), setpoint_meta=_meta, stop_policy=_always_stoppable
    )[0]
    assert record.stale is True


def test_phase_is_carried_as_text():
    record = build_ramp_records(
        _snapshot(phase="warmup"), setpoint_meta=_meta, stop_policy=_always_stoppable
    )[0]
    assert record.phase == "warmup"


# ── Ownership and stoppability ────────────────────────────────────────────────


def test_no_run_means_no_owner():
    record = build_ramp_records(
        _snapshot(), setpoint_meta=_meta, stop_policy=_always_stoppable
    )[0]
    assert record.owner is None
    assert record.stoppable is True
    assert record.stop_blocked_reason == ""


def test_claim_everything_run_owns_every_ramp():
    """run_claims=None is the claim-everything case (every plain procedure)."""
    record = build_ramp_records(
        _snapshot(),
        setpoint_meta=_meta,
        stop_policy=_never_stoppable,
        run_label="procedure 'Field Sweep'",
        run_claims=None,
    )[0]
    assert record.owner == "procedure 'Field Sweep'"


def test_narrow_claims_leave_unclaimed_ramps_unowned():
    """A run claiming only some VIs does not own the others' ramps."""
    record = build_ramp_records(
        _snapshot(),
        setpoint_meta=_meta,
        stop_policy=_always_stoppable,
        run_label="procedure 'Field Sweep'",
        run_claims={"dc_measurement"},
    )[0]
    assert record.owner is None


def test_refusal_reason_is_carried_verbatim():
    record = build_ramp_records(
        _snapshot(), setpoint_meta=_meta, stop_policy=_never_stoppable
    )[0]
    assert record.stoppable is False
    assert "claimed by running procedure 'X'" in record.stop_blocked_reason


# ── Value object ──────────────────────────────────────────────────────────────


def test_record_is_frozen_and_json_safe():
    record = build_ramp_records(
        _snapshot(), setpoint_meta=_meta, stop_policy=_always_stoppable
    )[0]
    with pytest.raises(Exception):
        record.vi_name = "other"  # frozen dataclass
    as_dict = record.as_dict()
    assert as_dict["vi_name"] == "magnet_z"
    assert set(as_dict) == {f.name for f in RampRecord.__dataclass_fields__.values()}
