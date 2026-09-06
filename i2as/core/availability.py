"""The Availability standard: one state, one tag vocabulary, one policy table.

I2AS has exactly four reasons a Virtual Instrument cannot be used the
way a fully live one can:

- ``"connect_failed"`` — the hardware could not be reached, either at
  build time or on a failed reconnect attempt.
- ``"operator"`` — the operator released it via the connection-lifecycle
  standard (see ``virtual_instruments/base.py``), freeing it for its own
  front panel or the vendor's software.
- ``"not_responding"`` — the VI is held (still in the live registry) but
  its poll has gone stale or disconnected: a comm-origin ``Condition``
  (see ``core/conditions.py`` and GLOSSARY.md's **Instrument fault**).
- ``"detached"`` — a single-client VI that released its session between
  runs and reacquires it on demand; usable, just not holding the bus right
  now.

Each tag maps to exactly one of four mutually exclusive **states**
(``AVAILABILITY_STATES``): ``"live"`` (no tags at all), ``"absent"``
(``connect_failed`` and/or ``operator`` — not in the live registry),
``"faulted"`` (``not_responding`` — live but not answering), or
``"detached"`` (``detached`` — live, session released, reacquires on arm).
State is always *derived* from tags by `state_for()`, never set by hand, so
the two can never disagree — enforced by `Availability.__post_init__`.

Both a set of tags and a single state exist because they answer different
questions: tags can co-occur (e.g. ``{"operator", "connect_failed"}`` — an
operator-released VI whose reconnect then fails on hardware, see
`core/station.py`'s `connect_instrument()`), while state is the single
mutually-exclusive answer a caller branches on.

This module is the *pure policy* layer of the standard: it holds the
`TagPolicy`/`Availability` value objects and the deterministic
`state_for()`/`decide_availability()` functions, and imports nothing beyond
the Python standard library. The producers that discover which tags apply —
the Station's offline registry, its unified condition registry, and a VI's
own attachment state — and the enforcement built on top of the resulting
policy live above this module (Station, Orchestrator) and are out of scope
here.
"""

from __future__ import annotations

from dataclasses import dataclass

AVAILABILITY_TAGS = ("connect_failed", "operator", "not_responding", "detached")
AVAILABILITY_STATES = ("live", "detached", "faulted", "absent")

# Most-restrictive-first: when a VI carries more than one tag, the winning
# policy is the first of these found present. Restrictiveness is judged by
# what the tag forbids relative to a fully live VI:
#   - "connect_failed" and "operator" both mean *absent* — not enumerable,
#     not controllable at all — the most restrictive state there is. Between
#     the two, "connect_failed" leads because it names a hardware fact (the
#     instrument truly could not be reached) rather than an operator choice,
#     and the policy row the two share is identical either way, so this only
#     affects which reason a caller would report first if it only wanted one.
#   - "not_responding" is next: the VI is still enumerable, but every other
#     column is the most restrictive of the tags that keep a VI live
#     (uncontrollable, fails a claimed run, raises an ErrorEvent).
#   - "detached" is last: fully enumerable AND controllable, the least
#     restrictive tag — it just means the session is not held this instant.
TAG_PRECEDENCE: tuple[str, ...] = (
    "connect_failed",
    "operator",
    "not_responding",
    "detached",
)


@dataclass(frozen=True)
class TagPolicy:
    """The declared policy one availability tag selects.

    Attributes:
        tag: The `AVAILABILITY_TAGS` member this row governs.
        state: The `AVAILABILITY_STATES` member this tag implies.
        enumerable: Whether the VI still appears in VI-enumerating listings
            while this tag holds.
        controllable: Whether manual control actions are admitted while
            this tag holds.
        fails_claimed_run: Whether a run watching this VI fails while this
            tag holds (mirrors `i2as.core.conditions.decide()`'s
            `run_failure`, for the tags that route through a `Condition`).
        raises_error_event: Whether onset of this tag raises an ErrorEvent.
        recovery: How this tag clears — ``"connect"`` (an explicit Connect
            action), ``"retry"`` (the VI polls successfully again), or
            ``""`` (automatic, no operator action needed).
    """

    tag: str
    state: str
    enumerable: bool
    controllable: bool
    fails_claimed_run: bool
    raises_error_event: bool
    recovery: str


TAG_POLICY: dict[str, TagPolicy] = {
    "connect_failed": TagPolicy(
        tag="connect_failed",
        state="absent",
        enumerable=False,
        controllable=False,
        fails_claimed_run=False,
        raises_error_event=False,
        recovery="connect",
    ),
    "operator": TagPolicy(
        tag="operator",
        state="absent",
        enumerable=False,
        controllable=False,
        fails_claimed_run=False,
        raises_error_event=False,
        recovery="connect",
    ),
    "not_responding": TagPolicy(
        tag="not_responding",
        state="faulted",
        enumerable=True,
        controllable=False,
        fails_claimed_run=True,
        raises_error_event=True,
        recovery="retry",
    ),
    "detached": TagPolicy(
        tag="detached",
        state="detached",
        enumerable=True,
        controllable=True,
        fails_claimed_run=False,
        raises_error_event=False,
        recovery="",
    ),
}


def state_for(tags: frozenset[str]) -> str:
    """Return the `AVAILABILITY_STATES` member implied by a set of tags.

    Args:
        tags: The availability tags currently held, drawn from
            `AVAILABILITY_TAGS`.

    Returns:
        ``"live"`` if `tags` is empty; otherwise the `TagPolicy.state` of
        the highest-precedence tag present, per `TAG_PRECEDENCE`.
    """
    if not tags:
        return "live"
    for tag in TAG_PRECEDENCE:
        if tag in tags:
            return TAG_POLICY[tag].state
    return "live"


def decide_availability(tags: frozenset[str]) -> TagPolicy | None:
    """Return the winning `TagPolicy` for a set of tags, or `None` if none apply.

    The one resolver: when several tags apply at once, `TAG_PRECEDENCE`
    picks the single policy row that governs enforcement.

    Args:
        tags: The availability tags currently held, drawn from
            `AVAILABILITY_TAGS`.

    Returns:
        The `TagPolicy` of the highest-precedence tag present, or `None` if
        `tags` is empty (the VI is live and unconstrained).
    """
    if not tags:
        return None
    for tag in TAG_PRECEDENCE:
        if tag in tags:
            return TAG_POLICY[tag]
    return None


@dataclass(frozen=True)
class Availability:
    """The unified answer to "why can't I use this instrument?" for one VI.

    Assembled by the Station (`core/station.py`'s `availability()` /
    `availabilities()`) from its existing sources of truth — the offline
    registry, the unified condition registry, and the VI's own attachment
    state — never stored as a fourth registry of its own.

    Attributes:
        vi_name: The VI's configured name.
        vi_type: The registry vi_type from config (``"system"``,
            ``"measurement"``).
        state: One of `AVAILABILITY_STATES`, always equal to
            ``state_for(tags)``.
        tags: The availability tags currently held, a subset of
            `AVAILABILITY_TAGS`. Empty for a fully live VI.
        reason: Human-readable description, suitable for direct display in
            the GUI. Empty for a fully live VI.
        since: Unix timestamp this availability state was first observed.

    Raises:
        ValueError: If `__post_init__` finds the fields inconsistent — see
            its docstring for the exact rules.
    """

    vi_name: str
    vi_type: str
    state: str
    tags: frozenset[str]
    reason: str
    since: float

    def __post_init__(self) -> None:
        """Validate that tags are in-vocabulary and state agrees with tags.

        Raises:
            ValueError: If `tags` contains a value outside
                `AVAILABILITY_TAGS`; if `state` is not one of
                `AVAILABILITY_STATES`; if `state` disagrees with
                ``state_for(tags)``.
        """
        for tag in self.tags:
            if tag not in AVAILABILITY_TAGS:
                raise ValueError(
                    f"Availability.tags must be drawn from {AVAILABILITY_TAGS}, "
                    f"got {tag!r}"
                )
        if self.state not in AVAILABILITY_STATES:
            raise ValueError(
                f"Availability.state must be one of {AVAILABILITY_STATES}, "
                f"got {self.state!r}"
            )
        expected = state_for(self.tags)
        if self.state != expected:
            raise ValueError(
                f"Availability.state {self.state!r} disagrees with "
                f"state_for(tags)={expected!r} for tags {self.tags!r}"
            )
