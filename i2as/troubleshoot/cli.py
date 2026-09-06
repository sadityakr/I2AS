"""Troubleshoot CLI — one-shot diagnostic commands for agents and humans.

Command grammar is API: the setup-supervisor skills and permission allowlists
hard-code it, so subcommand names and their meanings must stay stable.

Read/write split for permission gating: ``scan``, ``probe``, ``check``,
``bench-l0``, ``methods``, ``idn``, ``read``, ``status``, and ``session``
never change instrument state and are safe to allowlist (``status`` only
reads a log file, ``session`` only an experiment folder). ``write`` calls
state-changing driver methods and ``query`` / ``send`` transmit arbitrary raw
bytes (a raw query can mutate state too), so those three should stay behind a
permission prompt.

There are deliberately no interactive prompts: authorization is the
harness's job, and a hung prompt is the worst failure mode for an agent.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import i2as
from i2as.core.logging_config import setup_logging
from i2as.core.paths import log_directory, measurement_root, user_config_dir
from i2as.core.config import read_tick_interval_ms, read_trends_config
from i2as.core.trend_checks import CheckResult, declared_checks, run_checks
from i2as.troubleshoot import engine, session_report, status_reader
from i2as.troubleshoot.engine import (
    DriverBench,
    L0BenchResult,
    ProbeResult,
    bench_l0,
    check_config,
    check_trend_store_live,
    probe_address,
    scan_bus,
)

logger = logging.getLogger(__name__)

# Mirrors i2as/gui/app_settings.py (_ORGANISATION/_APPLICATION/
# _ACTIVE_CONFIG_NAME_KEY/_ACTIVE_CONFIG_SOURCE_KEY). Duplicated because
# contract C10 keeps this package out of i2as.gui — if app_settings
# changes these, change them here too.
_QSETTINGS_ORG = "I2AS"
_QSETTINGS_APP = "I2AS"
_ACTIVE_CONFIG_NAME_KEY = "ActiveConfig/name"
_ACTIVE_CONFIG_SOURCE_KEY = "ActiveConfig/source"

# Test seam: tests monkeypatch these two module attributes.
_rm_factory = engine.open_resource_manager


def _transcript_dir() -> Path:
    """Directory for the JSONL invocation transcript.

    Delegates to ``i2as.core.paths.log_directory()`` (see its
    docstring for the resolution precedence, overridable via
    ``I2AS_LOG_DIR``).
    """
    return log_directory()


# ── Config resolution ─────────────────────────────────────────────────────────


def _shipped_config_dir() -> Path:
    return Path(i2as.__file__).parent / "configs"


def _user_config_dir() -> Path:
    """The user's editable config copies: ``configs/`` under the per-user config root.

    Delegates to ``i2as.core.paths.user_config_dir()`` (the per-user
    path standard) so this CLI and the application resolve the same
    directory.
    """
    return user_config_dir() / "configs"


def _read_active_config() -> str | None:
    """Return the app's saved active-config directory, or None if unavailable.

    Resolved from the saved ``(name, source)`` identity rather than a stored
    path, so it stays correct across clones/worktrees (see app_settings.py
    ``config_active``/``set_config_active`` for the rationale).
    """
    try:
        from PyQt6.QtCore import QSettings
    except ImportError:
        return None
    settings = QSettings(_QSETTINGS_ORG, _QSETTINGS_APP)
    name = settings.value(_ACTIVE_CONFIG_NAME_KEY)
    source = settings.value(_ACTIVE_CONFIG_SOURCE_KEY)
    if not name or not source:
        return None
    base_dir = _user_config_dir() if str(source) == "user" else _shipped_config_dir()
    return str(base_dir / str(name))


def resolve_config(value: str | None) -> str:
    """Resolve a --config argument to a config directory path.

    Resolution order:

    1. ``value`` as a directory path (absolute or relative).
    2. ``value`` as a bare name under the shipped configs, then user configs.
    3. With no value: the machine's saved active config (QSettings), falling
       back to the shipped ``sim_cryostat``.

    Args:
        value: The --config argument, or None.

    Returns:
        Path string of an existing config directory.

    Raises:
        SystemExit: Via argparse-style error if nothing resolves.
    """
    if value:
        candidates = [
            Path(value),
            _shipped_config_dir() / value,
            _user_config_dir() / value,
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return str(candidate)
        raise SystemExit(
            f"error: config '{value}' not found (tried "
            f"{[str(c) for c in candidates]})"
        )

    active = _read_active_config()
    if active and Path(active).is_dir():
        return active
    return str(_shipped_config_dir() / "sim_cryostat")


# ── Bench construction ────────────────────────────────────────────────────────


def _make_bench(args: argparse.Namespace) -> DriverBench:
    """Build the bench from TARGET: config alias, or dotted class + --address."""
    if getattr(args, "address", None):
        return DriverBench.from_class(args.target, args.address)
    return DriverBench.from_config(resolve_config(args.config), args.target)


# ── Rendering ─────────────────────────────────────────────────────────────────


def _print_json(payload: dict[str, Any]) -> None:
    # default=repr: driver methods may return values JSON cannot encode.
    print(json.dumps(payload, indent=2, default=repr))


def _print_probe_table(results: list[ProbeResult]) -> None:
    for r in results:
        name = r.alias or r.address
        extra = r.idn or r.detail
        print(f"{name:<20} {r.address:<22} {r.code.value:<20} {extra}")
        if r.idn and r.detail:
            print(f"{'':<20} {'':<22} {'':<20} {r.detail}")


def _summarize(results: list[ProbeResult]) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.code.value] = counts.get(r.code.value, 0) + 1
    return ", ".join(f"{n} {code}" for code, n in sorted(counts.items()))


# ── Subcommand implementations ────────────────────────────────────────────────
# Each returns (ok, payload): ok drives the exit code, payload goes to stdout
# (--json) and to the transcript either way.


def _cmd_scan(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    rm = _rm_factory()
    resources = scan_bus(rm)
    payload: dict[str, Any] = {"resources": resources}

    probes: list[ProbeResult] = []
    if args.probe or args.probe_serial:
        for address in resources:
            if address.upper().startswith("ASRL") and not args.probe_serial:
                continue  # unknown-baud serial probing is opt-in
            probes.append(probe_address(rm, address, idn_command=args.idn_command))
        payload["probes"] = [p.as_dict() for p in probes]

    if args.json:
        _print_json(payload)
    else:
        for address in resources:
            print(address)
        if probes:
            print()
            _print_probe_table(probes)
    return True, payload


def _cmd_probe(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    result = probe_address(_rm_factory(), args.address, idn_command=args.idn_command)
    if args.json:
        _print_json(result.as_dict())
    else:
        _print_probe_table([result])
    return result.ok, result.as_dict()


def _cmd_check(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    config = resolve_config(args.config)
    rm = None
    if not args.no_bus:
        try:
            rm = _rm_factory()
        except Exception as exc:  # noqa: BLE001 — degrade to no bus scan
            logger.warning("Bus scan unavailable, checking without it: %s", exc)
    results = check_config(config, rm=rm)
    ok = all(r.ok for r in results)
    payload = {
        "config": config,
        "ok": ok,
        "results": [r.as_dict() for r in results],
    }
    if args.json:
        _print_json(payload)
    else:
        print(f"Config: {config}")
        _print_probe_table(results)
        print(f"=> {_summarize(results)}")
    return ok, payload


def _print_l0_bench_table(results: list[L0BenchResult]) -> None:
    for r in results:
        idn_mark = "OK  " if r.idn_ok else "FAIL"
        print(f"{r.alias:<20} idn={idn_mark}  {r.idn or r.detail}")
        if r.getter:
            getter_mark = "OK  " if r.getter_ok else "FAIL"
            print(f"{'':<20} {r.getter}()={getter_mark}  {r.getter_value}")
        elif r.idn_ok:
            print(f"{'':<20} (no extra zero-arg getter found besides get_idn)")


def _cmd_bench_l0(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    """L0 bench: idn + one passive getter for every driver in a config.

    The automated half of the commissioning skill's L0 rung — zero
    excitation, no approval needed. Run after `check` is green; a human
    still has to eyeball whether the returned values are physically
    plausible (this only proves communication and parsing).
    """
    config = resolve_config(args.config)
    results = bench_l0(config)
    ok = all(r.ok for r in results)
    payload = {
        "config": config,
        "ok": ok,
        "results": [r.as_dict() for r in results],
    }
    if args.json:
        _print_json(payload)
    else:
        print(f"Config: {config}")
        _print_l0_bench_table(results)
        n_ok = sum(1 for r in results if r.ok)
        print(f"=> {n_ok}/{len(results)} passed L0")
    return ok, payload


def _cmd_methods(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    bench = _make_bench(args)
    try:
        methods = bench.list_methods()
    finally:
        bench.close()
    payload = {"target": args.target, "methods": [m.as_dict() for m in methods]}
    if args.json:
        _print_json(payload)
    else:
        for m in methods:
            marker = "read " if m.read_only else "WRITE"
            print(f"[{marker}] {m.name}{m.signature}  — {m.doc}")
    return True, payload


def _call_and_report(
    args: argparse.Namespace, method: str, call_args: list[str], allow_write: bool
) -> tuple[bool, dict[str, Any]]:
    bench = _make_bench(args)
    try:
        result = bench.call(method, call_args, allow_write=allow_write)
    finally:
        bench.close()
    payload = {
        "target": args.target,
        "method": method,
        "args": call_args,
        "result": result,
    }
    if args.json:
        _print_json(payload)
    else:
        print(repr(result))
    return True, payload


def _cmd_idn(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    return _call_and_report(args, "get_idn", [], allow_write=False)


def _cmd_read(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    if args.repeat <= 1:
        return _call_and_report(args, args.method, args.args, allow_write=False)
    return _cmd_read_repeated(args)


def _cmd_read_repeated(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    """Repeat a read to expose intermittent faults (timing/settling class).

    A read that passes some repeats and fails others — especially when the
    failure rate drops at longer --interval values — is the signature of a
    too-short waiting time rather than a dead instrument. Failures are
    collected, not aborted on, because the failure *pattern* is the datum.
    """
    import time

    bench = _make_bench(args)
    outcomes: list[dict[str, Any]] = []
    failures = 0
    try:
        for i in range(args.repeat):
            if i > 0 and args.interval > 0:
                time.sleep(args.interval)
            try:
                value = bench.call(args.method, args.args, allow_write=False)
                outcomes.append({"i": i, "ok": True, "value": value})
            except Exception as exc:  # noqa: BLE001 — per-iteration capture is the point
                failures += 1
                outcomes.append(
                    {"i": i, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                )
    finally:
        bench.close()

    ok = failures == 0
    payload = {
        "target": args.target,
        "method": args.method,
        "args": args.args,
        "repeat": args.repeat,
        "interval_s": args.interval,
        "failures": failures,
        "outcomes": outcomes,
    }
    if args.json:
        _print_json(payload)
    else:
        for o in outcomes:
            print(f"[{o['i']:>4}] {'ok   ' if o['ok'] else 'FAIL '} "
                  f"{o.get('value', o.get('error'))!r}")
        print(f"=> {failures}/{args.repeat} failed at interval {args.interval}s")
    return ok, payload


def _cmd_write(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    return _call_and_report(args, args.method, args.args, allow_write=True)


def _cmd_query(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    bench = _make_bench(args)
    try:
        response = bench.query(args.command)
    finally:
        bench.close()
    payload = {"target": args.target, "command": args.command, "response": response}
    if args.json:
        _print_json(payload)
    else:
        print(response)
    return True, payload


def _cmd_send(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    bench = _make_bench(args)
    try:
        bench.send(args.command)
    finally:
        bench.close()
    payload = {"target": args.target, "command": args.command}
    if args.json:
        _print_json(payload)
    else:
        print("sent")
    return True, payload


# ── Parser ────────────────────────────────────────────────────────────────────


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        help="config directory path or name (default: the saved active "
        "config, falling back to the shipped sim_cryostat)",
    )


def _add_target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "target",
        help="driver alias from the config's real_drivers — or, together "
        "with --address, a dotted driver class path (driver development)",
    )
    parser.add_argument(
        "--address",
        help="VISA resource string; makes TARGET a dotted class path instead "
        "of a config alias",
    )
    _add_config_arg(parser)


def _judge_freshness(
    digest: dict[str, Any], max_age_s: float
) -> tuple[float | None, str | None]:
    """Judge a status digest against a ``--max-age`` freshness limit.

    Freshness is judged from the newest record's own ``ts``, never from the
    file's mtime: a log rotation or a file copy moves the mtime without a
    single new record being written, and the question here is whether the app
    is still *ticking*. Mirrors `engine.check_trend_store_live`, which asks
    the same question of the trend-history store.

    Args:
        digest: The `status_reader.summarize()` digest to judge.
        max_age_s: Maximum tolerated age, in seconds, of the newest record.

    Returns:
        ``(age_s, reason)`` — the newest record's age in seconds where that
        can be computed, and a one-line operator-readable reason the log
        fails the gate, or None when it is fresh.
    """
    if not digest.get("available"):
        return None, "no operational-status record: the app has not written one here."
    ts = digest.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return None, (
            "the newest record carries no timestamp (schema 1 log), so its age "
            "cannot be checked — treat it as stale."
        )
    age_s = time.time() - float(ts)
    if age_s > max_age_s:
        return age_s, (
            f"the newest record is {age_s:.0f}s old (limit {max_age_s:.0f}s) — "
            "the app is not ticking, so this state is history, not live."
        )
    return age_s, None


def _cmd_status(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    """Summarize the running app's operational-status log (works while it runs).

    Reads status.jsonl from the resolved log directory (see
    ``i2as.core.paths.log_directory()``; written by the
    Orchestrator each tick) and reports the current state, per-instrument
    ramp progress and trend, and any stall alerts. Exit 0 only when a log
    exists and its verdict is OK, so an agent can gate on the exit code.
    This is the one troubleshoot command that reads the LIVE app rather than
    opening instruments with the app closed.

    With ``--max-age`` the exit code also gates on freshness. Without it, a
    log left behind by a process that died days ago still reads as a
    confident "RAMPING, ~1400 s to target" and exits 0, so anything gating on
    the exit code should pass ``--max-age``.
    """
    log_path = args.log or (_transcript_dir() / "status.jsonl")
    records = status_reader.read_records(log_path, last=args.last)
    digest = status_reader.summarize(records)
    stale_reason: str | None = None
    if args.max_age is not None:
        age_s, stale_reason = _judge_freshness(digest, args.max_age)
        digest["max_age_s"] = args.max_age
        digest["age_s"] = None if age_s is None else round(age_s, 1)
        digest["stale"] = stale_reason is not None
        if stale_reason:
            digest["stale_reason"] = stale_reason
    if args.json:
        _print_json(digest)
    else:
        print(status_reader.render_text(digest))
        if stale_reason:
            print(f"stale: {stale_reason}", file=sys.stderr)
    ok = bool(digest.get("available")) and digest.get("verdict") == "OK"
    return ok and stale_reason is None, digest


_WINDOW_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "": 1.0}


def _parse_window(value: str) -> float:
    """Parse a `--window` value like ``"8h"``, ``"90m"``, ``"2d"``, or a bare seconds number.

    Args:
        value: The raw `--window` string.

    Returns:
        The window length in seconds.

    Raises:
        SystemExit: If `value` does not match ``<number><unit>`` with unit in
            ``s``/``m``/``h``/``d`` (or no unit, meaning seconds).
    """
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhdSMHD]?)\s*", value)
    if not match:
        raise SystemExit(
            f"error: invalid --window value {value!r} (expected e.g. '8h', '90m', '3600')"
        )
    amount = float(match.group(1))
    return amount * _WINDOW_UNITS[match.group(2).lower()]


def _format_evidence_value(value: Any) -> str:
    """Render one evidence value for the human-readable CLI path.

    A plain value (number, string) renders as-is. A dataclass instance
    (e.g. `trend_history.KeySummary`, which `no_data_outcome()` stores
    directly in a `CheckResult.evidence` mapping) renders its fields
    inline instead of falling back to its default `repr()` — the
    ``ClassName(field=value, ...)`` dump that is otherwise indistinguishable
    from noise to a human reading the CLI at 3 AM. Generic over any
    dataclass so a future check's evidence never needs a name-specific
    branch here — see `_render_check_result()`'s docstring.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = ", ".join(f"{k}={v!r}" for k, v in dataclasses.asdict(value).items())
        return f"{{{fields}}}"
    return str(value)


def _render_check_result(result: CheckResult) -> str:
    """One human-readable line for a `CheckResult`, evidence included.

    An agent reading this at 3 AM instead of a log needs the numbers behind
    the verdict, not just the verdict — this repository's first principle is
    that claims are traceable to their source. Evidence values render
    through `_format_evidence_value()` rather than plain `str()`/`f"{v}"`,
    so a dataclass value (e.g. the `KeySummary` a "no data" verdict cites)
    prints its fields instead of a bare `KeySummary(...)` repr; `--json`
    output is unaffected (`dataclasses.asdict()` already flattens it there).
    """
    marker = {True: "PASS", False: "FAIL", None: "N/A "}[result.passed]
    evidence = ", ".join(f"{k}={_format_evidence_value(v)}" for k, v in result.evidence.items())
    line = f"[{marker}] {result.name} — {result.message}"
    if evidence:
        line += f"\n{'':<9} evidence: {evidence}"
    return line


def _cmd_trends(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    """Evaluate this setup's declared trend checks plus the pull-only store-liveness check.

    Runs `i2as.core.trend_checks.declared_checks()` (the same standard
    the in-process `TrendCheckRunner` publishes from) against the resolved
    trend-history log directory, and separately evaluates
    `trend_store_live` — the one check that is pull-only by design (see
    `declared_checks()`'s docstring): it reads the store's file state
    directly, from this separate CLI process, so it can catch a running
    application that has wedged, which no check scheduled inside that same
    process ever could.

    `--window`, when given, overrides every declared check's `window_s`
    uniformly (`dataclasses.replace`, not a per-check branch) — an ad hoc
    "was everything fine over the last N hours" query distinct from each
    check's own configured default window.

    Returns:
        ``(ok, payload)`` where `ok` is `False` if any check's `passed` is
        `False` — an indeterminate (`None`) result does not fail the
        command, since "cannot tell" is not itself a problem (see
        `i2as.core.trend_checks`'s module docstring).
    """
    config = resolve_config(args.config)
    trends_config = read_trends_config(config)
    log_dir = log_directory()
    now = time.time()

    checks = declared_checks(trends_config)
    if args.window:
        window_s = _parse_window(args.window)
        checks = tuple(dataclasses.replace(check, window_s=window_s) for check in checks)
    results = list(run_checks(checks, log_dir, now=now))

    tick_interval_ms = read_tick_interval_ms(config)
    stale_seconds = trends_config["store_live_stale_ticks"] * tick_interval_ms / 1000.0
    results.append(check_trend_store_live(log_dir, stale_seconds, now=now))

    ok = not any(r.passed is False for r in results)
    payload: dict[str, Any] = {
        "config": config,
        "log_directory": str(log_dir),
        "ok": ok,
        "results": [dataclasses.asdict(r) for r in results],
    }
    if args.json:
        _print_json(payload)
    else:
        print(f"Config: {config}")
        print(f"Trend-history log directory: {log_dir}")
        for result in results:
            print(_render_check_result(result))
        n_pass = sum(1 for r in results if r.passed is True)
        n_fail = sum(1 for r in results if r.passed is False)
        n_indet = sum(1 for r in results if r.passed is None)
        print(f"=> {n_pass} pass, {n_fail} fail, {n_indet} indeterminate")
    return ok, payload


def _resolve_experiment_dir(explicit: str | None) -> tuple[Path | None, str | None]:
    """Resolve which experiment folder to report on.

    Args:
        explicit: The EXPERIMENT_DIR positional argument, or None.

    Returns:
        ``(directory, reason)`` — the folder to report on, or None together
        with one operator-readable sentence saying what was looked for and
        where. An explicitly named folder is taken as given (its record is
        parsed, and complains for itself if absent); with no argument, the
        newest experiment under the measurement root wins.
    """
    if explicit:
        return Path(explicit), None
    try:
        root = measurement_root()
    except RuntimeError as exc:
        return None, str(exc)
    directory = session_report.latest_experiment_dir(root)
    if directory is None:
        return None, (
            f"No experiment found under {root} (looked for "
            f"sessions/<user_id>/<session_id>/<experiment_id>/experiment.json)."
        )
    return directory, None


def _cmd_session(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    """Report on one finished or in-progress experiment folder (read-only).

    The after-the-fact sibling of ``status``: where that one explains what the
    running app is doing right now, this reads an experiment's own record off
    disk and lists its runs in order — kind, procedure, outcome, timestamps,
    duration, data file — plus the session envelope it ran under and any
    incident report filed in the folder. It opens no instruments and writes
    nothing.

    Exit 0 means a report was produced; exit 1 means there was nothing to
    report on (no experiment resolved, or its ``experiment.json`` missing or
    unparseable). A failed run is *content* of a successful report, not a
    failure of the command — an agent gating on the exit code is asking "did
    I get the record?", and reads the outcomes out of the payload.
    """
    directory, reason = _resolve_experiment_dir(args.experiment_dir)
    if directory is None:
        report = session_report.unavailable(reason or "No experiment found.")
    else:
        report = session_report.build_report(directory)
    if args.json:
        _print_json(report)
    else:
        print(session_report.render_text(report))
    return bool(report.get("available")), report


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse command tree (kept separate for --help testing)."""
    parser = argparse.ArgumentParser(
        prog="python -m i2as.troubleshoot",
        description="I2AS troubleshoot toolbox: diagnose instruments and "
        "configs while the main application is closed.",
    )
    # A parent parser lets every subcommand accept --json in its natural
    # trailing position (e.g. "check --json"); add_help=False stops it from
    # stealing the subparsers' -h.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", help="machine-readable JSON output"
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p = sub.add_parser("scan", help="list VISA resources on the bus",
                       parents=[common])
    p.add_argument("--probe", action="store_true",
                   help="also identify every non-serial resource")
    p.add_argument("--probe-serial", action="store_true",
                   help="probe serial (ASRL) resources too — opt-in, since "
                        "bytes at a wrong baud rate can confuse instruments")
    p.add_argument("--idn-command", default="*IDN?",
                   help="identify query for probes (default '*IDN?'; use 'V' "
                        "for pre-SCPI Oxford instruments)")
    p.set_defaults(func=_cmd_scan)

    p = sub.add_parser("probe", parents=[common], help="raw identify query to one bare address")
    p.add_argument("address", help="VISA resource string")
    p.add_argument("--idn-command", default="*IDN?")
    p.set_defaults(func=_cmd_probe)

    p = sub.add_parser("check", parents=[common], help="preflight every driver in a config")
    _add_config_arg(p)
    p.add_argument("--no-bus", action="store_true",
                   help="skip the bus-presence scan (probe by opening only)")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("bench-l0", parents=[common],
                       help="L0 bench for every driver in a config: idn + one "
                            "passive getter (zero excitation, no approval needed)")
    _add_config_arg(p)
    p.set_defaults(func=_cmd_bench_l0)

    p = sub.add_parser("status", parents=[common],
                       help="summarize the RUNNING app's operational-status log")
    p.add_argument(
        "--log",
        help="path to status.jsonl (default: status.jsonl in the resolved "
        "log directory — see i2as.core.paths.log_directory(), "
        "overridable via I2AS_LOG_DIR)",
    )
    p.add_argument("--last", type=int, default=5,
                   help="recent records to fold in for the gap trend (default 5)")
    p.add_argument(
        "--max-age", type=float, default=None, metavar="SECONDS",
        help="fail (exit 1) unless the newest record is younger than this. "
        "A few tick intervals is the useful value — 30 at the default 3 s "
        "tick. Off by default; pass it whenever you gate on the exit code, "
        "or a log from a process that died days ago reads as a live run",
    )
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("session", parents=[common],
                       help="report on one experiment folder: its runs, their "
                            "outcomes and data files, its envelope, and any "
                            "incident reports filed beside it")
    p.add_argument(
        "experiment_dir",
        nargs="?",
        metavar="EXPERIMENT_DIR",
        help="experiment folder to report on (default: the most recently "
        "modified experiment under the measurement root — see "
        "i2as.core.paths.measurement_root(), overridable via "
        "I2AS_MEASUREMENT_ROOT)",
    )
    p.set_defaults(func=_cmd_session)

    p = sub.add_parser("trends", parents=[common],
                       help="evaluate the trend checks this setup's devices.yaml "
                            "declares under trends.checks (each a channel kept within "
                            "a band over a window) plus the pull-only store-liveness "
                            "check")
    _add_config_arg(p)
    p.add_argument(
        "--window",
        help="override every declared check's window uniformly, e.g. '8h', '90m', "
        "'3600' (default: each check's own configured window)",
    )
    p.set_defaults(func=_cmd_trends)

    p = sub.add_parser("methods", parents=[common], help="list a driver's public methods")
    _add_target_args(p)
    p.set_defaults(func=_cmd_methods)

    p = sub.add_parser("idn", parents=[common], help="identify one configured instrument")
    _add_target_args(p)
    p.set_defaults(func=_cmd_idn)

    p = sub.add_parser("read", parents=[common], help="call a read-only driver method (get_*)")
    _add_target_args(p)
    p.add_argument("method")
    p.add_argument("args", nargs="*", help="method arguments (coerced by type hints)")
    p.add_argument("--repeat", type=int, default=1,
                   help="repeat the read N times to expose intermittent faults")
    p.add_argument("--interval", type=float, default=0.2,
                   help="seconds between repeats (default 0.2); a failure rate "
                        "that drops at longer intervals points to timing")
    p.set_defaults(func=_cmd_read)

    p = sub.add_parser("write", parents=[common], help="call a state-changing driver method")
    _add_target_args(p)
    p.add_argument("method")
    p.add_argument("args", nargs="*")
    p.set_defaults(func=_cmd_write)

    p = sub.add_parser("query", parents=[common], help="raw command with reply (state may change!)")
    _add_target_args(p)
    p.add_argument("command", help="raw command string, e.g. '*IDN?'")
    p.set_defaults(func=_cmd_query)

    p = sub.add_parser("send", parents=[common], help="raw command, no reply (state may change!)")
    _add_target_args(p)
    p.add_argument("command")
    p.set_defaults(func=_cmd_send)

    return parser


# ── Transcript ────────────────────────────────────────────────────────────────


def _append_transcript(argv: list[str], ok: bool, payload: dict[str, Any]) -> None:
    """Append one JSONL line describing this invocation (best-effort)."""
    try:
        directory = _transcript_dir()
        directory.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "argv": argv,
            "ok": ok,
            "payload": payload,
        }
        with (directory / "troubleshoot.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=repr) + "\n")
    except Exception as exc:  # noqa: BLE001 — a broken transcript must not fail the command
        logger.warning("Could not append troubleshoot transcript: %s", exc)


# ── Entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Run one troubleshoot command.

    Args:
        argv: Argument list (defaults to sys.argv[1:]). Exposed so tests call
            the CLI in-process.

    Returns:
        0 if the command fully succeeded, 1 on any fault or error.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    setup_logging()
    args = build_parser().parse_args(argv)

    try:
        ok, payload = args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — one place turns any failure into exit 1
        message = f"{type(exc).__name__}: {exc}"
        if args.json:
            _print_json({"error": message})
        else:
            print(f"error: {message}", file=sys.stderr)
        logger.error("troubleshoot %s failed: %s", args.subcommand, message)
        _append_transcript(argv, False, {"error": message})
        return 1

    _append_transcript(argv, ok, payload)
    return 0 if ok else 1
