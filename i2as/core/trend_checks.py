"""trend_checks — named temporal judgements over the trend-history store.

A trend check turns a window of `trend_history.summarize()` evidence into a
pass/fail/indeterminate verdict, e.g. "has the sample temperature been
stable for the last hour". Qt-free and `Station`-free by design, like its
neighbour `trend_history.py`: this module imports nothing from PyQt6, the
Orchestrator, or the Station, so a check is unit-testable against a synthetic
JSONL directory with no hardware and no GUI.

The standard this module defines: a check is a `TrendCheck` DECLARATION (a
name, the state keys and window it reads, its severity, and a predicate
function), not a hand-written function with its own control flow. `run_checks`
is the one runner that evaluates any number of declarations uniformly — it
never special-cases a check by name — and `to_condition`/`conditions_for` are
the one adapter that turns a failing check into the `Condition` currency the
rest of I2AS's System-Condition standard (`core/conditions.py`) already
understands. The checks a setup runs are DECLARED IN ITS CONFIG: `declared_checks()`
builds one `TrendCheck` per entry of the `trends.checks:` list that
`i2as.core.config.read_trends_config()` hands it, each through a builder such as
`channel_within_band()` — so which channel is watched and where its band
lies are setup facts in `devices.yaml`, never constants in this module. A
new kind of judgement is one builder function plus one `CHECK_KINDS` row.

No data in a requested window is not a failure of the thing being measured,
and it is not a pass either: it is "cannot tell", which this module
represents as `CheckOutcome.passed=None`/`CheckResult.passed=None`, distinct
from `True`/`False`. `summarize()` already tells the two "no data" cases
apart (`persisted=False` — never persisted, e.g. a measurement-VI key never
reaches disk — versus `persisted=True, count=0` — persisted but nothing
landed in this window) and never raises; a predicate that ignores this and
divides by a zero count, or claims a definite pass/fail from an empty window,
is a defect. `no_data_outcome()` below is the shared helper every predicate
should call first. `to_condition()` never publishes a `Condition` for an
indeterminate result: the System-Condition standard's registry holds active
PROBLEMS, and "cannot tell" is not one.

A predicate takes two extra arguments beyond `summaries`, both derived from
the SAME evaluation `run_check()` already did — never re-declared or
re-decided by the predicate itself:

- The windowed series (`{key: [(t, value), ...]}` from
  `trend_history.read_window()`), for rate-of-change reasoning: `KeySummary`
  carries `first_t`/`last_t` but never the values sampled at those times, so
  a check that needs "how much did this change" (a boil-off rate, a
  ramp-completion ETA) cannot answer it from `summaries` alone, and adding
  those values to `KeySummary` itself would leak a rate-specific need into
  an aggregate-statistics type. A predicate that only needs aggregates
  (e.g. `channel_within_band`) simply does not read this argument. On
  the `3min`/`hourly` tiers a series' values are bucket *means*, not true
  instantaneous samples (see `KeySummary.tier`); a predicate presenting one
  as if it were a raw reading on a window long enough to leave the raw tier
  is a defect.
- The window, in seconds, THIS evaluation actually queried. A predicate must
  report this rather than a value closed over at `declared_checks()` time:
  a caller (the troubleshoot CLI's `--window` override) may evaluate a
  `TrendCheck` with a different `window_s` than the one it was declared
  with (`dataclasses.replace(check, window_s=...)`), and evidence citing a
  stale window would misstate what was actually evaluated.

`run_check()` computes `summarize()` and `read_window()` over the same
`(keys, window_s, now)` once, from the same tier `pick_tier()` picks, and
hands both plus `check.window_s` to every predicate uniformly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from i2as.core.conditions import SEVERITIES, Condition
from i2as.core.trend_history import KeySummary, read_window, summarize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckOutcome:
    """A predicate's raw verdict, before the check's identity is attached.

    Attributes:
        passed: `True`/`False` for a definite verdict, or `None` when the
            window had no data to judge — see the module docstring's "no
            data" policy. Distinct from `False` so a caller can tell "fine"
            from "cannot tell".
        message: Human- and agent-readable verdict, citing the evidence
            numbers directly (this repository's first principle is that
            claims are traceable to their source).
        evidence: The `KeySummary` fields (or numbers derived from them) the
            verdict rested on.
    """

    passed: bool | None
    message: str
    evidence: dict[str, object]


WindowedSeries = Mapping[str, list[tuple[float, float]]]

Predicate = Callable[[Mapping[str, KeySummary], WindowedSeries, float], CheckOutcome]


@dataclass(frozen=True)
class TrendCheck:
    """One declared temporal judgement over the trend-history store.

    Attributes:
        name: Stable identity, e.g. ``"temperature_temperature_within_band"``. Used to
            key its published `Condition` (``f"trend:{name}"``) and to match
            a `CheckResult` back to its declaration.
        keys: Flat state keys to summarize, as in `Station.last_state_flat()`
            (e.g. ``"temperature_temperature"``).
        window_s: Trailing window length in seconds, passed to
            `trend_history.summarize()`. Tier selection is not this check's
            decision — `trend_history.pick_tier()` is the single home for
            that.
        severity: A `i2as.core.conditions.SEVERITIES` rung. Every check
            shipped in this branch declares ``"advisory"`` — see
            `to_condition()`'s docstring for why nothing here forces that as
            a type-level restriction.
        predicate: Pure function from (``{key: KeySummary}``, the windowed
            series `{key: [(t, value), ...]}` — see `WindowedSeries` — and
            the actual window queried in seconds; see the module docstring)
            to a `CheckOutcome`, all three covering exactly the `keys` above
            over the window `run_check()` actually evaluated (normally
            `window_s`, but a caller may override it — see the module
            docstring). Knows nothing about Qt, the Orchestrator, or the
            Station.

    Raises:
        ValueError: If `__post_init__` finds the fields invalid — see its
            docstring.
    """

    name: str
    keys: tuple[str, ...]
    window_s: float
    severity: str
    predicate: Predicate

    def __post_init__(self) -> None:
        """Validate the declaration shape.

        Raises:
            ValueError: If `name` or `keys` is empty, if `window_s` is not
                positive, or if `severity` is not one of
                `i2as.core.conditions.SEVERITIES`.
        """
        if not self.name:
            raise ValueError("TrendCheck.name must be non-empty")
        if not self.keys:
            raise ValueError(f"TrendCheck {self.name!r}: keys must be non-empty")
        if self.window_s <= 0:
            raise ValueError(f"TrendCheck {self.name!r}: window_s must be positive")
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"TrendCheck {self.name!r}: severity must be one of {SEVERITIES}, "
                f"got {self.severity!r}"
            )


@dataclass(frozen=True)
class CheckResult:
    """One `TrendCheck`'s verdict for one evaluation.

    Attributes:
        name: The originating `TrendCheck.name`.
        passed: See `CheckOutcome.passed` — `True`/`False`/`None`
            (indeterminate).
        message: See `CheckOutcome.message`.
        evidence: See `CheckOutcome.evidence`.
    """

    name: str
    passed: bool | None
    message: str
    evidence: dict[str, object]


def no_data_outcome(
    summaries: Mapping[str, KeySummary], keys: Sequence[str]
) -> CheckOutcome | None:
    """Return the shared "cannot tell" outcome if any of `keys` has no data.

    Every predicate should call this first: a key `summarize()` never
    persisted (`persisted=False`, e.g. a measurement-VI key that never
    reaches disk) or persisted but empty in this window (`persisted=True`,
    `count=0`) means the check cannot judge anything, which is neither a
    pass nor a failure of the thing being measured (see the module
    docstring's "no data" policy).

    Args:
        summaries: This evaluation's `{key: KeySummary}`, as passed to a
            `TrendCheck.predicate`.
        keys: The keys the predicate actually needs data for.

    Returns:
        A `CheckOutcome(passed=None, ...)` naming which key(s) had no data,
        or `None` if every key has at least one sample — proceed with the
        real judgement.
    """
    missing = [key for key in keys if summaries[key].count == 0]
    if not missing:
        return None
    never_persisted = [key for key in missing if not summaries[key].persisted]
    if never_persisted:
        detail = f"never persisted to disk: {', '.join(sorted(never_persisted))}"
    else:
        detail = f"no samples in the requested window: {', '.join(sorted(missing))}"
    return CheckOutcome(
        passed=None,
        message=f"Cannot tell — {detail}.",
        evidence={key: summaries[key] for key in missing},
    )


def run_check(check: TrendCheck, log_dir: Path, now: float | None = None) -> CheckResult:
    """Evaluate one declared check against a trend-history log directory.

    Args:
        check: The declaration to evaluate.
        log_dir: Directory containing the trend-history JSONL files, as
            resolved by `i2as.core.paths.log_directory()`.
        now: Reference "now" timestamp; defaults to `time.time()` inside
            `trend_history.summarize()`.

    Returns:
        This check's `CheckResult`.
    """
    summaries = summarize(log_dir, check.keys, check.window_s, now=now)
    series = read_window(log_dir, check.keys, check.window_s, now=now)
    outcome = check.predicate(summaries, series, check.window_s)
    return CheckResult(
        name=check.name, passed=outcome.passed, message=outcome.message, evidence=outcome.evidence
    )


def run_checks(
    checks: Sequence[TrendCheck], log_dir: Path, now: float | None = None
) -> list[CheckResult]:
    """Evaluate every declared check against one trend-history log directory.

    The standard's runner: a uniform loop over whatever `checks` is passed,
    with no per-check branch. A caller (the CLI, a scheduled refresh) is
    what decides which checks to evaluate; this function never reads
    `declared_checks()` itself.

    Args:
        checks: The declarations to evaluate, in any order.
        log_dir: Directory containing the trend-history JSONL files.
        now: Reference "now" timestamp, shared by every check in this call
            so a single evaluation is judged against one consistent instant.

    Returns:
        One `CheckResult` per check, in the same order as `checks`.
    """
    return [run_check(check, log_dir, now=now) for check in checks]


def to_condition(check: TrendCheck, result: CheckResult, since: float) -> Condition | None:
    """Build the `Condition` a failing check publishes, or `None`.

    Only a definite failure (`result.passed is False`) becomes a `Condition`
    — a pass publishes nothing (there is nothing wrong to report) and an
    indeterminate result (`result.passed is None`) publishes nothing either,
    per the module docstring's "no data" policy: the System-Condition
    registry holds active problems, and "cannot tell" is not one.

    `severity` is read off `check.severity` rather than hardcoded, so this
    adapter itself never has to change if a future check is ever promoted
    off `"advisory"` — every check declared in this branch happens to be
    `"advisory"` (see the review standard this work is held to: a trend
    check must never be able to interrupt the operator), which is a fact
    about what `declared_checks()` currently returns, not a restriction
    this function enforces.

    Args:
        check: The declaration `result` came from (for `severity`/`name`).
        result: This evaluation's verdict.
        since: Unix timestamp to stamp a fresh `Condition.since` with. If
            the same key is already active in the Station's registry,
            `Station.publish_conditions()` -> `_upsert_condition()` preserves
            the PRIOR `since`/`acknowledged` instead — see their docstrings.

    Returns:
        A `Condition` keyed ``f"trend:{check.name}"``, origin ``"trend"``,
        `affected_vis=None` (advisory conditions trigger no enforcement, so
        no VI needs naming), or `None` if `result.passed` is not `False`.
    """
    if result.passed is not False:
        return None
    return Condition(
        key=f"trend:{check.name}",
        origin="trend",
        severity=check.severity,
        kind=check.name,
        source_vis=(),
        affected_vis=None,
        message=result.message,
        since=since,
    )


def conditions_for(
    checks: Sequence[TrendCheck], results: Sequence[CheckResult], since: float
) -> list[Condition]:
    """Build every `Condition` this evaluation's failing checks publish.

    Args:
        checks: The declarations `results` were evaluated from.
        results: `run_checks(checks, ...)`'s output.
        since: Passed through to `to_condition()`.

    Returns:
        One `Condition` per check with `passed is False`, in `results`
        order.

    Raises:
        KeyError: If a `result.name` does not match any `check.name`.
    """
    by_name = {check.name: check for check in checks}
    conditions: list[Condition] = []
    for result in results:
        condition = to_condition(by_name[result.name], result, since)
        if condition is not None:
            conditions.append(condition)
    return conditions


# ── channel_within_band ──────────────────────────────────────────────────────
# The one shipped check KIND. Its instances are not written here: a setup
# declares them in its devices.yaml `trends.checks:` list, which
# read_trends_config() hands to declared_checks() below, so which channel is
# watched and what "within band" means are setup facts in the config, never
# constants in this module.


def channel_within_band(
    key: str,
    low: float,
    high: float,
    window_s: float,
    *,
    name: str | None = None,
    severity: str = "advisory",
) -> TrendCheck:
    """Declare a check that one flat state key stayed inside ``[low, high]``.

    Passes when every sample in the trailing window lies within the band —
    judged from `summarize()`'s exact ``min``/``max`` over the window, never
    re-derived from raw rows (which would disagree with `summarize()` at
    tier boundaries; see the module docstring). Fails when the window's
    minimum dipped below ``low`` or its maximum rose above ``high``, and
    reports both bounds and both extremes as evidence so the reader can see
    by how much. The window is read off the predicate's own ``window_s``
    argument (the window THIS evaluation actually queried), never closed
    over, so a caller overriding `TrendCheck.window_s` never gets evidence
    citing a stale window.

    Args:
        key: The flat state key to watch, as in `Station.last_state_flat()`
            (``"<vi_name>_<monitored_method>"``, e.g.
            ``"temperature_temperature"``).
        low: Lower bound of the band, inclusive, in the key's own unit.
        high: Upper bound of the band, inclusive, same unit.
        window_s: Trailing window in seconds the check defaults to.
        name: Stable identity for the check; defaults to
            ``f"{key}_within_band"``.
        severity: A `i2as.core.conditions.SEVERITIES` rung; defaults to
            ``"advisory"`` — a trend check reports and never enforces.

    Returns:
        The `TrendCheck` declaration, ready for `run_check()`.

    Raises:
        ValueError: If ``low >= high``, or via `TrendCheck.__post_init__`
            for an empty key, a non-positive window or an unknown severity.
    """
    if not low < high:
        raise ValueError(
            f"channel_within_band({key!r}): low must be below high, got low={low!r} high={high!r}"
        )

    def predicate(
        summaries: Mapping[str, KeySummary], series: WindowedSeries, window_s: float
    ) -> CheckOutcome:
        # `series` is unused: this check needs only the aggregate summary.
        no_data = no_data_outcome(summaries, [key])
        if no_data is not None:
            return no_data
        summary = summaries[key]
        # min/max are non-None here: no_data_outcome already ruled out
        # count == 0, and summarize() always populates them together.
        below = summary.min < low  # type: ignore[operator]
        above = summary.max > high  # type: ignore[operator]
        evidence: dict[str, object] = {
            "min": summary.min,
            "max": summary.max,
            "low": low,
            "high": high,
            "count": summary.count,
            "window_s": window_s,
        }
        verdict = "left" if (below or above) else "stayed within"
        message = (
            f"{key} {verdict} the band [{low:g}, {high:g}] over "
            f"{window_s / 3600.0:.2g} h: min={summary.min:.4g}, max={summary.max:.4g}, "
            f"n={summary.count}."
        )
        return CheckOutcome(passed=not (below or above), message=message, evidence=evidence)

    return TrendCheck(
        name=name or f"{key}_within_band",
        keys=(key,),
        window_s=float(window_s),
        severity=severity,
        predicate=predicate,
    )


#: The builder behind each ``kind:`` a ``trends.checks:`` entry may name.
#: ``kind`` is optional in the config and defaults to the one shipped kind;
#: a new kind of check is one builder function plus one row here.
CHECK_KINDS: Mapping[str, Callable[..., TrendCheck]] = {
    "channel_within_band": channel_within_band,
}

_ENTRY_FIELDS = frozenset({"kind", "key", "low", "high", "window_s", "name", "severity"})


def declared_checks(config: Mapping[str, object]) -> tuple[TrendCheck, ...]:
    """Build every trend check this setup declares in its config.

    The standard's single registration point: the trend-check scheduler
    (`i2as.core.trend_check_runner`), the troubleshoot CLI's `trends`
    subcommand, and the conformance test all call this once and iterate
    whatever it returns — none of them special-case a check by name. The
    checks themselves are declared in the setup's `devices.yaml`::

        trends:
          checks:
            - key: temperature_temperature   # a Station.last_state_flat() key
              low: 1.0                       # band, in the key's own unit
              high: 320.0
              window_s: 3600.0
              # kind: channel_within_band    # the default, and the only shipped kind
              # name: sample_temperature_in_band  # default "<key>_within_band"
              # severity: advisory           # default; a trend check never enforces

    so watching a different channel, or adding a second band, is a config
    edit (`CLAUDE.md`: constants and limits in config, not code). Adding a
    new KIND of judgement means one builder function beside
    `channel_within_band()` and one row in `CHECK_KINDS` — no other module
    in this standard changes.

    `trend_store_live` is deliberately NOT declared here: it is pull-only by
    design (see its own module, `i2as.troubleshoot.engine`) — it exists
    to catch a wedged or crashed application, and a check scheduled on this
    same process's own `QTimer` (`i2as.core.trend_check_runner`) cannot
    fire once that process is the thing that is hung, so it would report
    healthy in exactly the scenario it is built for. It is evaluated only
    from the troubleshoot CLI, a separate process reading the store's file
    state from outside.

    Args:
        config: This setup's ``trends:`` block as merged by
            `i2as.core.config.read_trends_config()`; its ``"checks"``
            entry is the list above (``[]`` when the setup declares none).

    Returns:
        One `TrendCheck` per ``checks:`` entry, in declaration order.

    Raises:
        ValueError: If an entry is not a mapping, names an unknown ``kind``
            or an unknown field, or fails its builder's own validation —
            a malformed check is a config error worth stopping on, not a
            check that silently never runs.
    """
    entries = config.get("checks") or ()
    checks: list[TrendCheck] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"trends.checks[{index}] must be a mapping, got {entry!r}")
        unknown = set(entry) - _ENTRY_FIELDS
        if unknown:
            raise ValueError(
                f"trends.checks[{index}] has unknown field(s) {sorted(unknown)}; "
                f"allowed: {sorted(_ENTRY_FIELDS)}"
            )
        kind = str(entry.get("kind", "channel_within_band"))
        builder = CHECK_KINDS.get(kind)
        if builder is None:
            raise ValueError(
                f"trends.checks[{index}]: unknown kind {kind!r}; known: {sorted(CHECK_KINDS)}"
            )
        try:
            checks.append(builder(**{k: v for k, v in entry.items() if k != "kind"}))
        except TypeError as exc:  # a missing or misnamed required field
            raise ValueError(f"trends.checks[{index}] ({kind}): {exc}") from exc
    return tuple(checks)
