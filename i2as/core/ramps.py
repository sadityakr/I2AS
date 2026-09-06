"""RampRecord — structured running-ramp payload (core, dependency-free)."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["ACTIVE_RAMP_STATUS", "RampRecord", "build_ramp_records"]

#: The one ``ramp_status()`` value that means "this ramp is running now".
#: ``TARGET_REACHED`` is a ramp that has arrived (the VI holds at setpoint
#: while a procedure measures) and ``IDLE`` is no ramp at all — neither is
#: something the operator can stop, so neither becomes a record.
ACTIVE_RAMP_STATUS = "RAMPING"


@dataclass(frozen=True)
class RampRecord:
    """One running ramp, as the ramp tracker and any other consumer sees it.

    Every numeric field is in the VI's own user units (the units ``unit``
    names): tesla for a magnet, kelvin for a temperature controller, degrees
    for an angle — the SI-units-in-APIs rule, with display formatting left
    to the GUI.

    Attributes:
        vi_name: The registered VI name (e.g. ``"magnet_z"``).
        label: The VI's declared setpoint label (e.g. ``"field"``), from
            ``Station.system_setpoint_meta()``; falls back to ``vi_name``.
        unit: The setpoint unit (e.g. ``"T"``), possibly empty.
        value: The current reading the ramp is driving, or ``None`` if the
            VI does not expose ``ramp_value()``.
        setpoint: The NEXT setpoint — the intermediate value the hardware is
            driving to right now (``RampableVI.ramp_setpoint()``), or
            ``None``. For a magnet this is the ramp-segment boundary it is
            heading for; for a temperature controller, this tick's advanced
            setpoint. Equals ``target`` on the final step, and for a VI whose
            generator commands its target in one shot.
        target: The END setpoint the ramp finishes at
            (``RampableVI.ramp_target()``), or ``None``.
        rate: The active ramp rate in user units per MINUTE, or ``None``.
        phase: The ramp sub-phase (``"warmup"``, ``"matching"``, …) for a VI
            that has them, else ``None``.
        owner: A human label for the run that owns this ramp — e.g.
            ``"procedure 'Field Sweep'"`` — or ``None`` for a manual ramp
            started from the Monitor window.
        stoppable: Whether ``Orchestrator.stop_ramp(vi_name)`` would be
            admitted right now. A ramp owned by a run is not: stopping one
            VI mid-run would strand the run waiting on a setpoint it can
            never reach, so aborting the run is the only correct stop.
        stop_blocked_reason: Why not, when ``stoppable`` is False — the
            verdict text straight from the admission predicate, suitable for
            direct display as a tooltip. Empty when ``stoppable``.
        stale: True when the Station could not read this VI's ramp state
            this tick (a communication error); the numbers are unknown
            rather than current.
    """

    vi_name: str
    label: str
    unit: str
    value: float | None
    setpoint: float | None
    target: float | None
    rate: float | None
    phase: str | None
    owner: str | None
    stoppable: bool
    stop_blocked_reason: str
    stale: bool

    def as_dict(self) -> dict[str, Any]:
        """Return this record as a plain JSON-safe dict (field names unchanged)."""
        return asdict(self)


def build_ramp_records(
    ramp_info: Mapping[str, Mapping[str, Any]],
    *,
    setpoint_meta: Callable[[str], tuple[str, str]],
    stop_policy: Callable[[str], tuple[bool, str]],
    run_label: str | None = None,
    run_claims: Collection[str] | None = None,
) -> list[RampRecord]:
    """Turn one ramp-status snapshot into the list of running-ramp records.

    Pure: it reads the snapshot it is given and calls the two lookups, and
    touches no hardware. Only VIs whose ``ramp_status`` is
    ``ACTIVE_RAMP_STATUS`` produce a record — a ramp that has reached its
    target, or was never started, is not a running ramp.

    Args:
        ramp_info: One ``Station.get_ramp_status()`` snapshot.
        setpoint_meta: ``Station.system_setpoint_meta`` — maps a VI name to
            its ``(label, unit)``.
        stop_policy: The Orchestrator's manual-action admission predicate
            (``_manual_action_admissible``) — maps a VI name to
            ``(admitted, reason)``, which become ``stoppable`` /
            ``stop_blocked_reason``. Passing the same predicate the action
            itself uses is what keeps the button's enabled state and the
            action's verdict from ever disagreeing.
        run_label: A human label for the active run (e.g.
            ``"procedure 'Field Sweep'"``), or ``None`` when no run is
            active — every ramp is then a manual one.
        run_claims: The active run's claimed VI names, or ``None`` for the
            claim-everything case (every plain procedure). Ignored entirely
            when *run_label* is ``None``.

    Returns:
        One ``RampRecord`` per running ramp, ordered by VI name.
    """
    records: list[RampRecord] = []
    for vi_name in sorted(ramp_info):
        info = ramp_info[vi_name]
        if str(info.get("ramp_status", "")) != ACTIVE_RAMP_STATUS:
            continue
        label, unit = setpoint_meta(vi_name)
        stoppable, reason = stop_policy(vi_name)
        owner = None
        if run_label is not None and (run_claims is None or vi_name in run_claims):
            owner = run_label
        records.append(
            RampRecord(
                vi_name=vi_name,
                label=label,
                unit=unit,
                value=_as_float(info.get("value")),
                setpoint=_as_float(info.get("setpoint")),
                target=_as_float(info.get("target")),
                rate=_as_float(info.get("rate")),
                phase=str(info["phase"]) if info.get("phase") is not None else None,
                owner=owner,
                stoppable=stoppable,
                stop_blocked_reason="" if stoppable else reason,
                stale=bool(info.get("_stale", False)),
            )
        )
    return records


def _as_float(value: Any) -> float | None:
    """Return *value* as a float, or ``None`` if it is absent or not numeric.

    A VI that does not implement an introspection hook returns ``None``, and
    a VI misbehaving under a fault could return anything at all; neither may
    raise into the tick that builds these records.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
