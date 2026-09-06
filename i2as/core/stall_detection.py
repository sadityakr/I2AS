"""Stall detection — deterministic per-VI ramp-progress verdicts.

The record from ``build_operational_status`` carries the *facts* (gap, closing,
elapsed). This layer makes the *judgement*: has a ramp stopped making progress?
It is pure arithmetic over the record plus a small carried counter, so it is
fully unit-testable — a scripted sequence of ticks can assert the alert fires
at exactly the right tick and stays quiet through a phase the VI declares as
a deliberate pause.

Which phases are deliberate pauses is the VI's own physics, not this
module's: each rampable VI declares them in ``RampableVI.no_motion_phases``
(the no-motion phase declaration), ``Station.get_ramp_status()`` carries the
declaration beside the live phase, and the caller hands them in per VI. This
module knows no instrument.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from i2as.core.operational_status import RunFaultCode, worst_code


@dataclass
class StallConfig:
    """Tunable thresholds. Defaults are deliberately lenient (quiet beats eager).

    A stall detector that cries wolf gets ignored, so v1 favours late detection
    over false alarms; tighten these against real runs once "that one was
    stuck" / "that one was fine" feedback exists.

    ``stall_seconds`` is the threshold in wall-clock seconds — the config unit,
    because a setup's ``tick_interval_ms`` ranges from 1000 to 3000 ms across
    shipped setups, so a threshold counted in ticks would silently mean a
    different duration on each one. ``stall_ticks`` is derived from it exactly
    once here, at construction, via ``tick_interval_ms``: the per-tick
    judgement below only ever compares against a tick count, never re-derives
    seconds itself. The conversion is floored at 1 tick — a ``stall_seconds``
    shorter than one tick interval must still fire eventually rather than
    silently disabling the check (an unfloored division would otherwise round
    to 0 ticks, which never triggers).
    """

    noise_floor: float = 1e-3       # gap must shrink by more than this to count as progress
    stall_seconds: float = 18.0     # wall-clock seconds of non-closing progress before RAMP_STALLED
    tick_interval_ms: int = 3000    # this setup's tick period, for the seconds->ticks conversion
    stall_ticks: int = field(init=False)

    def __post_init__(self) -> None:
        raw_ticks = round(self.stall_seconds * 1000.0 / self.tick_interval_ms)
        self.stall_ticks = max(1, raw_ticks)


@dataclass
class StallState:
    """State carried across ticks: per-VI consecutive non-closing tick counts."""

    stuck_ticks: dict[str, int] = field(default_factory=dict)


def no_motion_phases_from(ramp_info: Mapping[str, Mapping]) -> dict[str, frozenset[str]]:
    """Pull each VI's no-motion phase declaration out of a ramp-status snapshot.

    Args:
        ramp_info: ``Station.get_ramp_status()``'s ``{vi_name: {...}}``, whose
            per-VI entries carry ``"no_motion_phases"``.

    Returns:
        ``{vi_name: frozenset_of_phases}``; a VI whose entry lacks the key
        declares none.
    """
    return {
        name: frozenset(info.get("no_motion_phases") or ())
        for name, info in ramp_info.items()
    }


def apply_stall_verdict(
    record: dict,
    state: StallState,
    config: StallConfig | None = None,
    no_motion_phases: Mapping[str, Collection[str]] | None = None,
) -> tuple[dict, StallState]:
    """Layer heuristic stall verdicts onto an operational-status record.

    Pure over ``record`` (mutated in place — it is freshly built each tick) and
    the carried ``state``. Reads no hardware and no clock of its own.

    Args:
        record: The dict from ``build_operational_status``.
        state: Per-VI non-closing tick counts from the previous tick.
        config: Thresholds; defaults if omitted.
        no_motion_phases: Per-VI ``{vi_name: phases}`` — each VI's
            ``RampableVI.no_motion_phases`` declaration, as carried in the
            ramp-status snapshot (see ``no_motion_phases_from()``). A VI
            absent from the mapping declares none, so every RAMPING tick of
            it is judged.

    Returns:
        ``(record, new_state)`` — the record with per-VI codes upgraded to
        RAMP_STALLED where warranted, an ``"alerts"`` list, and the overall
        ``"verdict"`` recomputed; and the StallState for the next tick.
    """
    config = config or StallConfig()
    no_motion_phases = no_motion_phases or {}
    new_stuck = dict(state.stuck_ticks)
    alerts: list[str] = list(record.get("alerts", []))

    for vi in record.get("vis", []):
        name = vi["vi_name"]
        ramping = vi.get("ramp_status") == "RAMPING"
        phase = vi.get("phase")
        if not ramping or phase in no_motion_phases.get(name, ()):
            # Not ramping, or in an expected no-motion phase: reset, don't judge.
            new_stuck[name] = 0
            continue

        closing = vi.get("closing")
        if closing is None:
            # First tick with a gap — no delta yet, so nothing to judge.
            new_stuck.setdefault(name, 0)
            continue

        if closing > config.noise_floor:
            new_stuck[name] = 0                       # meaningful progress
        else:
            new_stuck[name] = new_stuck.get(name, 0) + 1

        if new_stuck[name] >= config.stall_ticks and vi.get("code") == RunFaultCode.OK.value:
            vi["code"] = RunFaultCode.RAMP_STALLED.value
            gap = vi.get("gap")
            gap_str = f"{gap:.3g}" if gap is not None else "?"
            vi["detail"] = f"gap {gap_str} not closing for {new_stuck[name]} ticks"
            alerts.append(f"{name}: ramp stalled ({vi['detail']})")

    codes = [vi.get("code", RunFaultCode.OK.value) for vi in record.get("vis", [])]
    record["verdict"] = worst_code(codes)
    record["alerts"] = alerts
    return record, StallState(stuck_ticks=new_stuck)
