"""Tests for the operational-status record builder."""

from __future__ import annotations

import json
import time

import pytest

from i2as.core.conditions import Condition
from i2as.core.operational_status import (
    SCHEMA_VERSION,
    RunFaultCode,
    build_operational_status,
    next_sequence_number,
)


def test_normal_ramp_reports_gap_eta_and_ok():
    state = {"magnet_z": {"get_field": 0.5, "magnet_status": "RAMPING"}}
    ramp_info = {
        "magnet_z": {"value": 0.5, "target": 1.0, "rate": 0.5, "ramp_status": "RAMPING"},
    }
    record, gaps = build_operational_status(
        orch_state="RAMPING",
        elapsed_in_state_s=12.0,
        state=state,
        ramp_info=ramp_info,
        prev_gaps={},
    )
    assert record["orch_state"] == "RAMPING"
    assert record["verdict"] == "OK"
    vi = record["vis"][0]
    assert vi["vi_name"] == "magnet_z"
    assert vi["gap"] == pytest.approx(0.5)
    # eta = gap / (rate/60) = 0.5 / (0.5/60) = 60 s
    assert vi["eta_s"] == pytest.approx(60.0)
    assert vi["closing"] is None  # no previous gap
    assert gaps["magnet_z"] == pytest.approx(0.5)


def test_closing_is_gap_decrease_from_prev_tick():
    ramp_info = {"t": {"value": 48.0, "target": 50.0, "rate": 2.0, "ramp_status": "RAMPING"}}
    record, _ = build_operational_status(
        orch_state="RAMPING", elapsed_in_state_s=5.0, state={"t": {}},
        ramp_info=ramp_info, prev_gaps={"t": 3.0},
    )
    vi = record["vis"][0]
    assert vi["gap"] == pytest.approx(2.0)
    assert vi["closing"] == pytest.approx(1.0)  # 3.0 -> 2.0, converging


def test_stale_flag_sets_verdict():
    state = {"t": {"_stale": True}}
    ramp_info = {
        "t": {"value": None, "target": None, "rate": None,
              "ramp_status": "IDLE", "_stale": True},
    }
    record, _ = build_operational_status(
        orch_state="RAMPING", elapsed_in_state_s=1.0, state=state,
        ramp_info=ramp_info, prev_gaps={},
    )
    assert record["verdict"] == RunFaultCode.VI_STALE.value
    assert record["vis"][0]["code"] == "VI_STALE"


def test_disconnected_outranks_stale():
    state = {"t": {"_stale": True, "_disconnected": True}}
    ramp_info = {"t": {"value": None, "target": None, "rate": None, "ramp_status": "IDLE"}}
    record, _ = build_operational_status(
        orch_state="RAMPING", elapsed_in_state_s=1.0, state=state,
        ramp_info=ramp_info, prev_gaps={},
    )
    assert record["verdict"] == "VI_DISCONNECTED"


def test_a_critical_flag_sets_the_verdict_on_the_vi_that_reported_it():
    """An active critical-severity condition IS the CRITICAL_FLAG verdict.

    Read off this tick's condition registry, never off an instrument's own
    status word: the code names the declared severity, so any setup's
    critical flag reaches it (see the Safety-flag manifest standard).
    """
    ramp_info = {"magnet_z": {"value": 1.0, "target": 2.0, "rate": 0.5, "ramp_status": "RAMPING"}}
    quench = Condition(
        key="safety:quench",
        origin="safety",
        severity="critical",
        kind="quench",
        source_vis=("magnet_z",),
        affected_vis=None,
        message="Safety flag 'quench' is tripped",
        since=1.0,
    )
    record, _ = build_operational_status(
        orch_state="RAMPING", elapsed_in_state_s=1.0, state={},
        ramp_info=ramp_info, prev_gaps={}, conditions=[quench],
    )
    assert record["verdict"] == "CRITICAL_FLAG"
    assert record["vis"][0]["code"] == "CRITICAL_FLAG"
    assert "quench" in record["vis"][0]["detail"]


def test_a_hold_flag_is_not_a_critical_verdict():
    """Only critical severity produces the code — a hold is enforced elsewhere."""
    ramp_info = {"magnet_z": {"value": 1.0, "target": 2.0, "rate": 0.5, "ramp_status": "RAMPING"}}
    hold = Condition(
        key="safety:coolant_low",
        origin="safety",
        severity="hold",
        kind="coolant_low",
        source_vis=("magnet_z",),
        affected_vis=frozenset({"magnet_z"}),
        message="Safety flag 'coolant_low' is tripped",
        since=1.0,
    )
    record, _ = build_operational_status(
        orch_state="RAMPING", elapsed_in_state_s=1.0, state={},
        ramp_info=ramp_info, prev_gaps={}, conditions=[hold],
    )
    assert record["verdict"] == "OK"


def test_wait_block_present_only_during_wait():
    r1, _ = build_operational_status(
        orch_state="RAMPING", elapsed_in_state_s=1.0, state={}, ramp_info={},
        prev_gaps={}, wait_target_s=30.0, wait_elapsed_s=5.0,
    )
    assert r1["wait"] == {"target_s": 30.0, "elapsed_s": 5.0}
    r2, _ = build_operational_status(
        orch_state="RAMPING", elapsed_in_state_s=1.0, state={}, ramp_info={}, prev_gaps={},
    )
    assert r2["wait"] is None


def test_record_is_json_serializable():
    ramp_info = {"m": {"value": 0.0, "target": 1.0, "rate": 0.5, "ramp_status": "RAMPING"}}
    record, _ = build_operational_status(
        orch_state="RAMPING", elapsed_in_state_s=1.0, state={"m": {}},
        ramp_info=ramp_info, prev_gaps={},
    )
    json.dumps(record)  # must not raise (the record is written to status.jsonl)


def test_conditions_defaults_to_empty_list():
    record, _ = build_operational_status(
        orch_state="IDLE", elapsed_in_state_s=0.0, state={}, ramp_info={}, prev_gaps={},
    )
    assert record["conditions"] == []


def test_conditions_station_wide_reports_affected_all():
    condition = Condition(
        key="envelope:field too high",
        origin="envelope",
        severity="critical",
        kind="envelope",
        source_vis=(),
        affected_vis=None,
        message="field too high",
        since=100.0,
    )
    record, _ = build_operational_status(
        orch_state="EMERGENCY", elapsed_in_state_s=0.0, state={}, ramp_info={}, prev_gaps={},
        conditions=[condition],
    )
    assert record["conditions"] == [
        {
            "key": "envelope:field too high",
            "origin": "envelope",
            "severity": "critical",
            "kind": "envelope",
            "message": "field too high",
            "affected": "all",
            "since": 100.0,
            "acknowledged": False,
        }
    ]


def test_conditions_scoped_reports_sorted_affected_vis():
    condition = Condition(
        key="safety:coolant_low",
        origin="safety",
        severity="hold",
        kind="coolant_low",
        source_vis=("coolant_monitor",),
        affected_vis=frozenset({"magnet_z", "magnet_x"}),
        message="coolant low",
        since=50.0,
        acknowledged=True,
    )
    record, _ = build_operational_status(
        orch_state="RAMPING", elapsed_in_state_s=1.0, state={}, ramp_info={}, prev_gaps={},
        conditions=[condition],
    )
    entry = record["conditions"][0]
    assert entry["affected"] == ["magnet_x", "magnet_z"]
    assert entry["acknowledged"] is True


def test_conditions_are_sorted_by_key():
    c_b = Condition(
        key="safety:b", origin="safety", severity="advisory", kind="b",
        source_vis=(), affected_vis=None, message="b", since=1.0,
    )
    c_a = Condition(
        key="safety:a", origin="safety", severity="advisory", kind="a",
        source_vis=(), affected_vis=None, message="a", since=1.0,
    )
    record, _ = build_operational_status(
        orch_state="IDLE", elapsed_in_state_s=0.0, state={}, ramp_info={}, prev_gaps={},
        conditions=[c_b, c_a],
    )
    assert [c["key"] for c in record["conditions"]] == ["safety:a", "safety:b"]


def test_conditions_are_json_serializable():
    condition = Condition(
        key="comm:magnet_z", origin="comm", severity="hold", kind="stale",
        source_vis=("magnet_z",), affected_vis=frozenset({"magnet_z"}),
        message="stale", since=1.0,
    )
    record, _ = build_operational_status(
        orch_state="RAMPING", elapsed_in_state_s=1.0, state={}, ramp_info={}, prev_gaps={},
        conditions=[condition],
    )
    json.dumps(record)  # must not raise


# ── The record standard's header fields (schema 2) ────────────────────────────


def _minimal_record(**kwargs):
    """Build one record from an empty station snapshot, for header assertions."""
    record, _ = build_operational_status(
        orch_state="IDLE", elapsed_in_state_s=0.0, state={}, ramp_info={},
        prev_gaps={}, **kwargs,
    )
    return record


def test_header_fields_are_always_present_and_typed():
    """Every header field of the record standard is present on every record."""
    record = _minimal_record()
    assert record["schema"] == SCHEMA_VERSION
    assert isinstance(record["ts"], float)
    assert record["ts"] == pytest.approx(time.time(), abs=60.0)
    assert isinstance(record["seq"], int)
    assert record["seq"] >= 1


def test_unknown_header_values_are_null_never_missing():
    """A value the caller does not know is None — a reader never sees a gap."""
    record = _minimal_record()
    for field in ("run_id", "experiment_id", "setup", "actor", "request_id"):
        assert field in record
        assert record[field] is None


def test_header_values_are_carried_through_when_supplied():
    record = _minimal_record(
        run_id="20260902_120000_001_mock_sweep",
        experiment_id="exp-7",
        setup="sim_cryostat",
    )
    assert record["run_id"] == "20260902_120000_001_mock_sweep"
    assert record["experiment_id"] == "exp-7"
    assert record["setup"] == "sim_cryostat"


def test_seq_strictly_increases_across_records():
    """Consecutive records carry strictly increasing sequence numbers."""
    seqs = [_minimal_record()["seq"] for _ in range(5)]
    assert all(later > earlier for earlier, later in zip(seqs, seqs[1:]))


def test_next_sequence_number_is_process_wide_and_starts_at_one():
    first = next_sequence_number()
    assert first >= 1
    assert next_sequence_number() == first + 1


def test_explicit_ts_and_seq_override_the_defaults():
    """Callers may stamp their own identity (replaying a log, say)."""
    record = _minimal_record(ts=1_000_000.5, seq=42)
    assert record["ts"] == pytest.approx(1_000_000.5)
    assert record["seq"] == 42


def test_record_is_json_serialisable_with_the_header():
    """The record is written as one JSON line, so it must serialise as-is."""
    record = _minimal_record(run_id="r1", setup="sim_cryostat")
    round_tripped = json.loads(json.dumps(record))
    assert round_tripped["schema"] == SCHEMA_VERSION
    assert round_tripped["run_id"] == "r1"
    assert round_tripped["experiment_id"] is None


# ── Who last got the engine to act ───────────────────────────────────────────
#
# The pair exists so status.jsonl can be joined to a client's own action trail
# on the request id — that join is the point, so the tests hold the pair's
# presence, its null-when-none rule, and that it survives the JSON line.


def test_the_last_accepted_command_is_carried_through_when_supplied():
    actor = {"kind": "agent", "id": "runner-7", "role": "session"}
    record = _minimal_record(actor=actor, request_id="3f2a9c1b")

    assert record["actor"] == actor
    assert record["request_id"] == "3f2a9c1b"


def test_the_last_accepted_command_survives_the_json_line():
    """The record is written as one line, so the join key must round-trip."""
    record = _minimal_record(
        actor={"kind": "agent", "id": "runner-7", "role": "session"},
        request_id="3f2a9c1b",
    )
    round_tripped = json.loads(json.dumps(record))

    assert round_tripped["actor"]["id"] == "runner-7"
    assert round_tripped["request_id"] == "3f2a9c1b"


def test_the_actor_is_copied_not_aliased():
    """A record is a snapshot; mutating the caller's dict must not rewrite it."""
    actor = {"kind": "agent", "id": "runner-7", "role": "session"}
    record = _minimal_record(actor=actor)
    actor["id"] = "someone-else"

    assert record["actor"]["id"] == "runner-7"
