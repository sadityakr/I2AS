"""Tests for the runtime status reader and `troubleshoot status`."""

from __future__ import annotations

import json
import time

from i2as.troubleshoot import status_reader
from i2as.troubleshoot.cli import main


def _write_log(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _rec(state, verdict, vis, alerts=None):
    return {
        "orch_state": state, "elapsed_in_state_s": 5.0, "verdict": verdict,
        "alerts": alerts or [], "progress": None, "vis": vis,
    }


def _vi(name, value, target, gap, code, ramp_status="RAMPING"):
    return {
        "vi_name": name, "value": value, "target": target, "gap": gap,
        "rate": 1.0, "eta_s": gap * 60.0, "ramp_status": ramp_status,
        "phase": None, "code": code,
    }


def _condition(
    key="safety:coolant_low", severity="hold", message="coolant low",
    affected="magnet_z", acknowledged=False,
):
    return {
        "key": key, "origin": "safety", "severity": severity, "kind": "coolant_low",
        "message": message, "affected": [affected] if affected != "all" else "all",
        "since": 1.0, "acknowledged": acknowledged,
    }


def test_read_records_missing_file(tmp_path):
    assert status_reader.read_records(tmp_path / "nope.jsonl") == []


def test_read_records_last_window(tmp_path):
    p = tmp_path / "status.jsonl"
    _write_log(p, [_rec("IDLE", "OK", []) for _ in range(5)])
    assert len(status_reader.read_records(p)) == 5
    assert len(status_reader.read_records(p, last=2)) == 2


def test_summarize_empty_is_unavailable():
    assert status_reader.summarize([]) == {"available": False}


def test_summarize_reports_closing_trend():
    recs = [
        _rec("RAMPING", "OK", [_vi("m", 8.0, 10.0, 2.0, "OK")]),
        _rec("RAMPING", "OK", [_vi("m", 9.0, 10.0, 1.0, "OK")]),
    ]
    d = status_reader.summarize(recs)
    assert d["available"] is True
    assert d["orch_state"] == "RAMPING"
    assert d["trends"]["m"] == "closing"


def test_render_stalled_shows_alert_and_code_help():
    rec = _rec(
        "RAMPING", "RAMP_STALLED", [_vi("temp", 48.0, 50.0, 2.0, "RAMP_STALLED")],
        alerts=["temp: ramp stalled (gap 2 not closing for 6 ticks)"],
    )
    text = status_reader.render_text(status_reader.summarize([rec]))
    assert "RAMP_STALLED" in text
    assert "ramp stalled" in text
    assert "not following" in text  # from CODE_HELP's RAMP_STALLED triage note


def test_render_no_log_message():
    assert "No operational-status log" in status_reader.render_text({"available": False})


def test_cli_status_ok_exits_zero(tmp_path, capsys):
    p = tmp_path / "status.jsonl"
    _write_log(p, [_rec("IDLE", "OK", [])])
    rc = main(["status", "--log", str(p), "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "OK"


def test_cli_status_stalled_exits_one(tmp_path):
    p = tmp_path / "status.jsonl"
    _write_log(p, [_rec("RAMPING", "RAMP_STALLED",
                        [_vi("m", 1.0, 5.0, 4.0, "RAMP_STALLED")],
                        alerts=["m: ramp stalled"])])
    assert main(["status", "--log", str(p)]) == 1


def test_cli_status_missing_log_exits_one(tmp_path):
    assert main(["status", "--log", str(tmp_path / "none.jsonl")]) == 1


def test_summarize_carries_conditions_through():
    rec = _rec("RAMPING", "OK", [])
    rec["conditions"] = [_condition()]
    d = status_reader.summarize([rec])
    assert d["conditions"] == [_condition()]


def test_summarize_defaults_conditions_to_empty_for_old_log():
    rec = _rec("IDLE", "OK", [])  # no "conditions" key at all
    assert "conditions" not in rec
    d = status_reader.summarize([rec])
    assert d["conditions"] == []


def test_render_text_shows_active_conditions_section():
    rec = _rec("RAMPING", "OK", [_vi("m", 1.0, 2.0, 1.0, "OK")])
    rec["conditions"] = [_condition(severity="hold", message="coolant low", affected="magnet_z")]
    text = status_reader.render_text(status_reader.summarize([rec]))
    assert "Active conditions:" in text
    assert "HOLD: coolant low (affects: magnet_z)" in text


def test_render_text_marks_station_wide_and_acknowledged_conditions():
    rec = _rec("EMERGENCY", "OK", [])
    rec["conditions"] = [
        _condition(
            key="envelope:field too high", severity="critical",
            message="field too high", affected="all", acknowledged=True,
        )
    ]
    text = status_reader.render_text(status_reader.summarize([rec]))
    assert "CRITICAL: field too high (affects: all instruments) [acknowledged]" in text


def test_render_text_omits_active_conditions_section_when_none():
    rec = _rec("IDLE", "OK", [])  # old-log style, no conditions field
    text = status_reader.render_text(status_reader.summarize([rec]))
    assert "Active conditions:" not in text


def test_cli_status_json_includes_conditions(tmp_path, capsys):
    p = tmp_path / "status.jsonl"
    rec = _rec("IDLE", "OK", [])
    rec["conditions"] = [_condition()]
    _write_log(p, [rec])
    rc = main(["status", "--log", str(p), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["conditions"] == [_condition()]


# ── --max-age: a log left by a dead process must not read as a live run ───────


def _fresh(records_ts: float, **kwargs):
    """One OK record stamped with *records_ts* (the record standard's ``ts``)."""
    record = _rec("IDLE", "OK", [])
    record.update({"schema": 2, "ts": records_ts, "seq": 1, **kwargs})
    return record


def test_cli_status_max_age_fresh_log_exits_zero(tmp_path, capsys):
    p = tmp_path / "status.jsonl"
    _write_log(p, [_fresh(time.time())])
    assert main(["status", "--log", str(p), "--max-age", "60"]) == 0


def test_cli_status_max_age_stale_log_exits_one(tmp_path, capsys):
    """A three-day-old record must not yield a confident, healthy exit 0."""
    p = tmp_path / "status.jsonl"
    _write_log(p, [_fresh(time.time() - 3 * 86400.0)])
    assert main(["status", "--log", str(p), "--max-age", "30"]) == 1
    err = capsys.readouterr().err
    assert "stale" in err
    assert "not ticking" in err


def test_cli_status_max_age_missing_log_exits_one(tmp_path, capsys):
    assert main(["status", "--log", str(tmp_path / "none.jsonl"), "--max-age", "30"]) == 1
    assert "no operational-status record" in capsys.readouterr().err


def test_cli_status_max_age_rejects_a_log_without_a_timestamp(tmp_path, capsys):
    """A schema-1 record carries no ``ts``, so its age cannot be vouched for."""
    p = tmp_path / "status.jsonl"
    _write_log(p, [_rec("RAMPING", "OK", [])])  # no ts key at all
    assert main(["status", "--log", str(p), "--max-age", "30"]) == 1
    assert "no timestamp" in capsys.readouterr().err


def test_cli_status_without_max_age_ignores_age(tmp_path):
    """The freshness gate is opt-in; existing callers are unaffected."""
    p = tmp_path / "status.jsonl"
    _write_log(p, [_fresh(time.time() - 3 * 86400.0)])
    assert main(["status", "--log", str(p)]) == 0


def test_cli_status_max_age_json_carries_the_reason(tmp_path, capsys):
    p = tmp_path / "status.jsonl"
    _write_log(p, [_fresh(time.time() - 600.0)])
    assert main(["status", "--log", str(p), "--json", "--max-age", "30"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["stale"] is True
    assert payload["max_age_s"] == 30.0
    assert payload["age_s"] > 500.0
    assert "stale_reason" in payload


def test_summarize_carries_the_record_standard_header():
    digest = status_reader.summarize([_fresh(1234.5, run_id="r1", setup="sim_cryostat")])
    assert digest["schema"] == 2
    assert digest["ts"] == 1234.5
    assert digest["seq"] == 1
    assert digest["run_id"] == "r1"
    assert digest["setup"] == "sim_cryostat"
    assert digest["experiment_id"] is None


def test_summarize_reports_schema_one_for_a_headerless_record():
    digest = status_reader.summarize([_rec("IDLE", "OK", [])])
    assert digest["schema"] == 1
    assert digest["ts"] is None


# ── Who last got the engine to act ───────────────────────────────────────────
#
# status.jsonl says what the station is doing; the actor and request id say
# who last asked for it, and the request id is the join into that client's own
# action trail. The reader has to carry both through verbatim, and to stay
# silent rather than guess when the log predates them.

_AGENT = {"kind": "agent", "id": "runner-7", "role": "session"}


def test_summarize_carries_the_last_accepted_command():
    digest = status_reader.summarize(
        [_fresh(1234.5, actor=_AGENT, request_id="3f2a9c1b")]
    )
    assert digest["actor"] == _AGENT
    assert digest["request_id"] == "3f2a9c1b"


def test_summarize_reports_no_command_for_a_log_that_predates_the_field():
    digest = status_reader.summarize([_fresh(1234.5)])
    assert digest["actor"] is None
    assert digest["request_id"] is None


def test_render_names_the_actor_and_the_request_id():
    """Both halves printed verbatim: the request id is a join key, not decoration."""
    text = status_reader.render_text(
        status_reader.summarize([_fresh(1234.5, actor=_AGENT, request_id="3f2a9c1b")])
    )
    assert "agent 'runner-7'" in text
    assert "role session" in text
    assert "3f2a9c1b" in text


def test_render_says_nothing_when_no_command_has_been_accepted():
    text = status_reader.render_text(status_reader.summarize([_fresh(1234.5)]))
    assert "Last accepted command" not in text


def test_cli_status_json_carries_the_last_accepted_command(tmp_path, capsys):
    """`troubleshoot status --json` is how an agent reads it back."""
    log = tmp_path / "status.jsonl"
    _write_log(log, [_fresh(time.time(), actor=_AGENT, request_id="3f2a9c1b")])

    assert main(["status", "--log", str(log), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["actor"]["id"] == "runner-7"
    assert payload["request_id"] == "3f2a9c1b"


# ── Tail-from-end reads ───────────────────────────────────────────────────────


def _write_bulk_log(path, count: int) -> None:
    """Write *count* realistically sized records, one JSON line each."""
    vis = [_vi(f"vi_{i}", 1.0, 2.0, 1.0, "OK") for i in range(6)]
    with path.open("w", encoding="utf-8") as handle:
        for seq in range(count):
            record = _rec("RAMPING", "OK", vis)
            record.update({"schema": 2, "ts": 1000.0 + seq, "seq": seq})
            handle.write(json.dumps(record) + "\n")


def test_tail_read_matches_a_full_read_on_a_multi_megabyte_log(tmp_path):
    """The tail path returns exactly what reading the whole file would."""
    p = tmp_path / "status.jsonl"
    _write_bulk_log(p, 4000)
    assert p.stat().st_size > 2_000_000, "expected a multi-MB synthetic log"

    full = status_reader.read_records(p)
    assert len(full) == 4000
    for window in (1, 5, 37):
        assert status_reader.read_records(p, last=window) == full[-window:]


def test_tail_read_stops_at_the_start_of_a_short_log(tmp_path):
    """Asking for more records than the file holds returns all of them."""
    p = tmp_path / "status.jsonl"
    _write_bulk_log(p, 3)
    assert len(status_reader.read_records(p, last=10)) == 3


def test_tail_read_skips_a_truncated_final_line(tmp_path):
    """A record half-written when the reader arrives is not a record yet."""
    p = tmp_path / "status.jsonl"
    _write_bulk_log(p, 3)
    with p.open("a", encoding="utf-8") as handle:
        handle.write('{"schema": 2, "ts": 1003.0, "seq": 3, "orch_st')

    assert [r["seq"] for r in status_reader.read_records(p, last=2)] == [1, 2]
    # The newest COMPLETE record is what a digest must describe.
    assert status_reader.summarize(status_reader.read_records(p, last=2))["seq"] == 2


def test_tail_read_handles_a_log_of_one_truncated_line(tmp_path):
    p = tmp_path / "status.jsonl"
    p.write_text('{"schema": 2, "ts": 1.0, "seq"', encoding="utf-8")
    assert status_reader.read_records(p, last=5) == []


def test_tail_read_on_an_empty_log(tmp_path):
    p = tmp_path / "status.jsonl"
    p.write_text("", encoding="utf-8")
    assert status_reader.read_records(p, last=5) == []
