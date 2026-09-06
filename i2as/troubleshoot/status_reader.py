"""Read and explain the runtime operational-status log.

This is the runtime sibling of ``troubleshoot.engine``'s setup-time checks: it
answers "what is the running measurement doing, and is it stuck?" by reading the
log the Orchestrator writes, never by touching the live app. It depends only on
the JSONL record format, not on ``i2as.core``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Bytes read per backward step when tailing. One tick's record is a few
# hundred bytes to a few kB, so a 64 kB step reaches the default five-record
# window in a single read on any realistic station.
_TAIL_CHUNK = 65536

# Plain-English meaning + first thing to check, per runtime fault code. Keyed by
# the string code as it appears in status.jsonl (the log is the contract), so
# this table stays independent of i2as.core.operational_status.
CODE_HELP: dict[str, str] = {
    "OK": "Normal — ramping/settling on schedule, or idle.",
    "VI_STALE": (
        "The instrument stopped returning fresh readings (values are cached). "
        "Check its connection and that it is powered and not hung."
    ),
    "VI_DISCONNECTED": (
        "Repeated communication failures — treated as off the bus. Check power, "
        "cabling, and address; run `troubleshoot check` with the app closed."
    ),
    "CRITICAL_FLAG": (
        "This instrument reported a safety flag its class declares critical. "
        "The run should be in EMERGENCY; find what tripped (the record's "
        "conditions list names the flag) before doing anything else."
    ),
    "RAMP_STALLED": (
        "A ramp has not moved toward its target for several ticks. The setpoint "
        "is being sent but the value is not following, so suspect a "
        "controller/PID limit, a saturated heater, a thermal load, or the "
        "instrument not accepting setpoints."
    ),
}


def tail_lines(path: Path, count: int) -> list[str]:
    """Return the last *count* complete lines of a file, reading from the end.

    status.jsonl is append-only and grows for as long as the app ticks, while
    every question this module answers ("what is it doing right now?") needs
    only its last few records. Reading the whole file to keep five lines makes
    the cheap question cost O(file), which on a long run is tens of MB per
    invocation. So seek to the end and step backwards in `_TAIL_CHUNK` blocks
    until enough newlines are in hand.

    Two fragments are dropped, both of which would otherwise parse as
    corrupt records:

    * a partial FINAL line — the writer is mid-append (a record is one
      `write()` of one line, but nothing guarantees the reader observes it
      whole), so a file not ending in a newline has an incomplete last line
      that is not a record yet;
    * a partial FIRST line — the backward window almost always begins in the
      middle of some earlier record; that fragment is only kept when the walk
      reached the start of the file, where it is a whole line.

    Args:
        path: The file to tail.
        count: Number of complete trailing lines wanted (must be positive).

    Returns:
        Up to *count* complete lines, oldest first, blank lines dropped.
    """
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        chunks: list[bytes] = []
        newlines = 0
        # One more newline than records wanted: it proves the window's first
        # line is complete rather than a fragment of an earlier record.
        while position > 0 and newlines <= count:
            step = min(_TAIL_CHUNK, position)
            position -= step
            handle.seek(position)
            chunk = handle.read(step)
            newlines += chunk.count(b"\n")
            chunks.insert(0, chunk)
        blob = b"".join(chunks)

    ends_complete = blob.endswith(b"\n")
    # errors="replace" cannot corrupt a returned line: the only place a
    # multi-byte character can be split is the window's leading edge, which
    # is either dropped below or the true start of the file.
    lines = blob.decode("utf-8", errors="replace").splitlines()
    if lines and not ends_complete:
        lines.pop()
    if lines and position > 0:
        lines.pop(0)
    return [line for line in lines if line.strip()][-count:]


def read_records(log_path: str | Path, last: int | None = None) -> list[dict]:
    """Return parsed JSONL records from status.jsonl (the last *last* if given).

    Missing file → empty list. Unparseable lines are skipped, not fatal.

    Args:
        log_path: Path to status.jsonl.
        last: Number of trailing records wanted. A positive value is read by
            tailing from the end of the file (`tail_lines`) rather than
            parsing the whole log; None (or a non-positive value, kept for
            callers relying on Python's slice semantics) reads it all.

    Returns:
        The parsed records, oldest first.
    """
    path = Path(log_path)
    if not path.exists():
        return []
    if last is not None and last > 0:
        lines = tail_lines(path, last)
    else:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if last is not None:
            lines = lines[-last:]
    records: list[dict] = []
    for ln in lines:
        try:
            records.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return records


def _trend_word(gaps: list[float]) -> str:
    """Classify a sequence of gaps as closing / widening / flat / unknown."""
    if len(gaps) < 2:
        return "unknown"
    if gaps[-1] < gaps[0] - 1e-9:
        return "closing"
    if gaps[-1] > gaps[0] + 1e-9:
        return "widening"
    return "flat"


def summarize(records: list[dict]) -> dict:
    """Fold a window of records into a digest (latest record is authoritative).

    ``conditions`` is read from the latest record and defaults to ``[]`` when
    absent — status.jsonl written before the System-Condition standard was
    carried into it stays readable, never an error (the log is the contract).
    The record standard's header fields (``schema``, ``ts``, ``seq``,
    ``run_id``, ``experiment_id``, ``setup``) are carried through the same
    way; ``schema`` defaults to 1 (records predating the field) and the rest
    to None. So are ``actor`` and ``request_id`` — who last got the engine to
    do something, and the id that joins this log to that client's own action
    trail (the agent feed's records carry the same request id).

    Args:
        records: The window of parsed records, oldest first.

    Returns:
        The digest dict rendered by `render_text` and printed by
        ``troubleshoot status --json``.
    """
    if not records:
        return {"available": False}
    latest = records[-1]
    # Header fields of the record standard (see
    # i2as.core.operational_status). `.get` throughout: a schema-1 log,
    # written before these existed, must stay readable and simply reports
    # them as None — the log is the contract, and old logs are part of it.
    gaps: dict[str, list[float]] = {}
    for rec in records:
        for vi in rec.get("vis", []):
            g = vi.get("gap")
            if g is not None:
                gaps.setdefault(vi["vi_name"], []).append(g)
    return {
        "available": True,
        "schema": latest.get("schema", 1),
        "ts": latest.get("ts"),
        "seq": latest.get("seq"),
        "run_id": latest.get("run_id"),
        "experiment_id": latest.get("experiment_id"),
        "setup": latest.get("setup"),
        "actor": latest.get("actor"),
        "request_id": latest.get("request_id"),
        "orch_state": latest.get("orch_state"),
        "elapsed_in_state_s": latest.get("elapsed_in_state_s"),
        "verdict": latest.get("verdict"),
        "alerts": latest.get("alerts", []),
        "progress": latest.get("progress"),
        "vis": latest.get("vis", []),
        "trends": {name: _trend_word(g) for name, g in gaps.items()},
        "window": len(records),
        "conditions": latest.get("conditions", []),
    }


def _last_command_text(digest: dict) -> str:
    """Say who last got the engine to act, and under which request id.

    The pair is the join key between this log and a client's own action trail
    (an agent's feed carries the same ``request_id`` on the command it sent,
    the verdict it got back, and the state changes that followed), so both
    halves are printed verbatim rather than prettified. Empty string when the
    record names no command — a log written before the field existed, or a
    process that has not accepted one yet.

    Args:
        digest: A `summarize()` digest.

    Returns:
        One line, or ``""`` when there is nothing to say.
    """
    actor = digest.get("actor")
    request_id = digest.get("request_id")
    if not isinstance(actor, dict) and not request_id:
        return ""
    who = "unknown"
    if isinstance(actor, dict):
        kind = actor.get("kind") or "unknown"
        who = f"{kind} {actor.get('id')!r}" if actor.get("id") else str(kind)
        if actor.get("role"):
            who = f"{who} (role {actor['role']})"
    return f"Last accepted command: by {who}, request {request_id or '-'}"


def render_text(digest: dict) -> str:
    """Render a digest as a plain-English block for the CLI and for agents."""
    if not digest.get("available"):
        return (
            "No operational-status log found (is the app running? status.jsonl "
            "is expected in the resolved log directory — see "
            "i2as.core.paths.log_directory(), overridable via "
            "I2AS_LOG_DIR)."
        )

    lines: list[str] = []
    lines.append(
        f"State: {digest['orch_state']}  "
        f"({digest['elapsed_in_state_s']}s in state)   Verdict: {digest['verdict']}"
    )
    if digest.get("progress") is not None:
        lines.append(f"Procedure progress: {digest['progress'] * 100:.0f}%")

    last_command = _last_command_text(digest)
    if last_command:
        lines.append(last_command)

    if digest["alerts"]:
        lines.append("Alerts:")
        lines.extend(f"  ! {a}" for a in digest["alerts"])

    if digest.get("conditions"):
        lines.append("Active conditions:")
        for c in digest["conditions"]:
            affected = c.get("affected")
            affected_str = "all instruments" if affected == "all" else ", ".join(affected)
            ack_str = " [acknowledged]" if c.get("acknowledged") else ""
            lines.append(
                f"  {c['severity'].upper()}: {c['message']} "
                f"(affects: {affected_str}){ack_str}"
            )

    lines.append("Instruments:")
    for vi in digest["vis"]:
        name = vi["vi_name"]
        trend = digest["trends"].get(name, "")
        gap = vi.get("gap")
        if vi.get("target") is not None and vi.get("value") is not None and gap is not None:
            eta = vi.get("eta_s")
            eta_str = f", ~{eta:.0f}s to target" if eta else ""
            lines.append(
                f"  {name}: {vi['value']:.4g} -> {vi['target']:.4g} "
                f"(gap {gap:.3g}, {vi.get('ramp_status')}, {trend}{eta_str}) [{vi['code']}]"
            )
        else:
            lines.append(f"  {name}: {vi.get('ramp_status')} [{vi['code']}]")

    codes = {vi["code"] for vi in digest["vis"]}
    if digest["verdict"] != "OK":
        codes.add(digest["verdict"])
    problem_codes = sorted(c for c in codes if c != "OK")
    if problem_codes:
        lines.append("What the codes mean:")
        lines.extend(f"  {c}: {CODE_HELP.get(c, 'Unknown code.')}" for c in problem_codes)

    return "\n".join(lines)
