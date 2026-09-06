"""The System-Condition standard: origin x severity, scope follows severity.

I2AS has exactly four producers of "something is wrong":

- ``"safety"`` — a Virtual Instrument's ``evaluate_safety()`` (see
  `virtual_instruments/base.py` and the **Control-validation standard** in
  `GLOSSARY.md`).
- ``"comm"`` — the Station's poller marking a VI stale or disconnected (see
  `GLOSSARY.md`'s **Instrument fault**).
- ``"envelope"`` — ``ExperimentEnvelope.check_state()`` protecting the sample
  under experiment-scoped bounds narrower than the config limits (see
  `GLOSSARY.md`'s **Session envelope**).
- ``"trend"`` — a failing named judgement over the trend-history store (see
  `core/trend_checks.py` and `GLOSSARY.md`'s **Trend check**), published on
  a slow independent cadence rather than every tick. Every trend check
  declared so far is ``"advisory"`` severity, making this origin the first
  real producer of that reserved rung (see the severity list below).

Each origin emits `Condition` objects with one of three severities:

- ``"advisory"`` — reported, no enforcement. Every ``"trend"``-origin
  condition uses this rung today; no other origin has published one yet.
- ``"hold"`` — scoped to the condition's `affected_vis`: those VIs go to
  standby on onset, their manual controls are refused while the condition
  persists, and a run watching one of them fails.
- ``"critical"`` — station-wide by construction (`affected_vis` is always
  `None`): EMERGENCY, standby-all, every manual control refused until
  acknowledged.

This module is the *pure policy* layer of the standard: it holds the
`Condition`/`Verdict` value objects and the deterministic `decide()`
function, and imports nothing beyond the Python standard library. The four
producers and the enforcement they trigger (standby, EMERGENCY entry, run
failure) live above this module (Station, Orchestrator, and — for
``"trend"`` — `core/trend_check_runner.py`) and are out of scope here — see
`core/README.md` for how those layers consume `decide()`'s `Verdict`.

A tripped safety flag can be *tolerated* by a run's
``tolerated_safety_flags`` (see `GLOSSARY.md`'s **Tolerated safety flags**);
that tolerance is applied by the producer at `Condition` construction time
(a tolerated hold-severity safety flag is simply never constructed, or is
constructed as ``"advisory"`` instead), never inside `decide()` — this
module never knows which run is active when it builds a `Condition`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

SEVERITIES = ("advisory", "hold", "critical")
ORIGINS = ("comm", "safety", "envelope", "trend")


@dataclass(frozen=True)
class Condition:
    """One "something is wrong" signal, as origin x severity.

    Immutable and hashable (frozen dataclass over hashable field types —
    `affected_vis` is a `frozenset` rather than a `set` for exactly this
    reason), so a `Condition` can be stored in sets/dict keys and compared
    by value.

    Attributes:
        key: Stable identity for this condition, e.g. ``"comm:magnet_z"``,
            ``"safety:coolant_low"``, ``"envelope:<bound-description>"``,
            ``"trend:temperature_temperature_within_band"``. Non-empty; used to sort
            and to deduplicate.
        origin: One of `ORIGINS` — which producer reported this.
        severity: One of `SEVERITIES` — determines enforcement scope.
        kind: A short discriminator within the origin: ``"stale"`` or
            ``"disconnected"`` for comm; the tripped flag name for safety;
            ``"envelope"`` for envelope; the check name for trend.
        source_vis: The VI name(s) that reported this condition, in
            reporting order. Empty for envelope (the envelope is not a VI)
            and for trend (a check reads state keys, not a specific VI).
        affected_vis: The VI names this condition scopes enforcement to.
            Must be `None` (station-wide) for a critical condition, and a
            non-empty `frozenset` for a hold condition. Advisory conditions
            may be either, since they trigger no enforcement.
        message: Human-readable description, surfaced to the operator.
        since: Unix timestamp this condition was first observed.
        acknowledged: Whether an operator has acknowledged this condition.
            Does not change enforcement — see `GLOSSARY.md`'s **Fault
            acknowledge** for the analogous existing behavior.

    Raises:
        ValueError: If `__post_init__` finds the fields inconsistent — see
            its docstring for the exact rules.
    """

    key: str
    origin: str
    severity: str
    kind: str
    source_vis: tuple[str, ...]
    affected_vis: frozenset[str] | None
    message: str
    since: float
    acknowledged: bool = False

    def __post_init__(self) -> None:
        """Validate the invariants that make scope follow from severity.

        Raises:
            ValueError: If `key` is empty; if `severity` is not one of
                `SEVERITIES`; if `origin` is not one of `ORIGINS`; if
                `severity` is ``"critical"`` and `affected_vis` is not
                `None`; if `severity` is ``"hold"`` and `affected_vis` is
                not a non-empty `frozenset`.
        """
        if not self.key:
            raise ValueError("Condition.key must be non-empty")
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"Condition.severity must be one of {SEVERITIES}, got {self.severity!r}"
            )
        if self.origin not in ORIGINS:
            raise ValueError(f"Condition.origin must be one of {ORIGINS}, got {self.origin!r}")
        if self.severity == "critical" and self.affected_vis is not None:
            raise ValueError(
                f"critical condition {self.key!r} must have affected_vis=None "
                "(critical is station-wide by definition), "
                f"got {self.affected_vis!r}"
            )
        if self.severity == "hold" and not self.affected_vis:
            raise ValueError(
                f"hold condition {self.key!r} must have a non-empty affected_vis frozenset, "
                f"got {self.affected_vis!r}"
            )


@dataclass(frozen=True)
class Verdict:
    """The enforcement decision `decide()` computes from a set of conditions.

    Attributes:
        held_vis: Maps a held VI name to the `Condition` holding it. Built
            only from hold-severity conditions.
        emergency: Every critical-severity condition, sorted by `key`.
            Non-empty means EMERGENCY is demanded station-wide.
        run_failure: ``(vi_name, condition)`` for the alphabetically-first
            watched VI that is held, or `None` if no watched VI is held (or
            no run is active). Computed independently of `emergency` —
            `decide()` does not suppress `run_failure` just because
            `emergency` is also non-empty. Precedence between the two
            (EMERGENCY takes priority over a plain run failure) is an
            executor-level policy applied by the caller (the Orchestrator),
            not by this module.
    """

    held_vis: Mapping[str, Condition]
    emergency: tuple[Condition, ...]
    run_failure: tuple[str, Condition] | None


def decide(
    conditions: Iterable[Condition],
    *,
    watched_vis: AbstractSet[str],
    run_active: bool,
) -> Verdict:
    """Turn a set of conditions into an enforcement verdict.

    Pure and deterministic: the same `conditions` (as a collection of
    values, order-independent) with the same `watched_vis`/`run_active`
    always yields an equal `Verdict`. No I/O, no clock, no hardware.

    Args:
        conditions: Every currently-active condition, from any origin and
            severity, in any order.
        watched_vis: The VI names the active run is watching (e.g. the
            claimed set — see `GLOSSARY.md`'s **Claim**). Ignored when
            `run_active` is False.
        run_active: Whether a run is currently active. `run_failure` is
            always `None` when this is False.

    Returns:
        A `Verdict` with `held_vis`, `emergency`, and `run_failure` derived
        as follows:

        - `emergency`: every critical-severity condition, sorted by `key`.
        - `held_vis`: iterate hold-severity conditions sorted by `key`; for
          each, map every VI in its `affected_vis` to that condition unless
          the VI is already mapped — the first (sorted-key) condition
          affecting a VI wins.
        - `run_failure`: `None` if `run_active` is False or no watched VI
          is in `held_vis`; otherwise the alphabetically-first watched VI
          present in `held_vis`, paired with its condition.

        advisory conditions contribute to none of the three fields.
    """
    ordered = sorted(conditions, key=lambda c: c.key)

    emergency = tuple(c for c in ordered if c.severity == "critical")

    held_vis: dict[str, Condition] = {}
    for condition in ordered:
        if condition.severity != "hold":
            continue
        assert condition.affected_vis is not None  # enforced by __post_init__
        for vi_name in condition.affected_vis:
            held_vis.setdefault(vi_name, condition)

    run_failure: tuple[str, Condition] | None = None
    if run_active:
        held_watched = sorted(vi_name for vi_name in watched_vis if vi_name in held_vis)
        if held_watched:
            first = held_watched[0]
            run_failure = (first, held_vis[first])

    return Verdict(held_vis=held_vis, emergency=emergency, run_failure=run_failure)


def envelope_conditions(violations: list[str], now: float) -> list[Condition]:
    """Build one critical `Condition` per envelope violation.

    The envelope origin's producer (`ExperimentEnvelope.check_state()`, see
    `GLOSSARY.md`'s **Session envelope**) reports violations as plain
    description strings rather than typed objects; this is the adapter that
    lifts them into the `Condition` currency the rest of the standard uses.
    An envelope violation is always station-wide and unconditional — it
    protects the sample, so it is always critical severity, never tolerated.

    Args:
        violations: Human-readable violation descriptions, one per breached
            bound.
        now: Unix timestamp to stamp every resulting `Condition.since` with.

    Returns:
        One `Condition` per violation, in the same order, each with
        ``key=f"envelope:{violation}"``, ``origin="envelope"``,
        ``severity="critical"``, ``kind="envelope"``, ``source_vis=()``,
        ``affected_vis=None``, and ``message=violation``.
    """
    return [
        Condition(
            key=f"envelope:{violation}",
            origin="envelope",
            severity="critical",
            kind="envelope",
            source_vis=(),
            affected_vis=None,
            message=violation,
            since=now,
        )
        for violation in violations
    ]
