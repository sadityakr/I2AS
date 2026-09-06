"""Tests for the System-Condition standard's pure policy core."""

from __future__ import annotations

import pytest
from i2as.core.conditions import (
    ORIGINS,
    SEVERITIES,
    Condition,
    decide,
    envelope_conditions,
)


_UNSET = object()


def _condition(
    key: str,
    *,
    origin: str = "safety",
    severity: str = "hold",
    kind: str = "flag",
    source_vis: tuple[str, ...] = (),
    affected_vis: frozenset[str] | None | object = _UNSET,
    message: str = "",
    since: float = 0.0,
    acknowledged: bool = False,
) -> Condition:
    """Build a Condition with sensible defaults, overridable per test.

    ``affected_vis`` defaults to ``frozenset({"vi"})`` for a hold-severity
    condition (the common case in these tests) and to ``None`` otherwise,
    but an explicitly passed value (including ``None``) is always honored —
    needed so validation-failure tests can pass ``affected_vis=None`` to a
    hold condition without the default silently filling it back in.
    """
    if affected_vis is _UNSET:
        affected_vis = frozenset({"vi"}) if severity == "hold" else None
    return Condition(
        key=key,
        origin=origin,
        severity=severity,
        kind=kind,
        source_vis=source_vis,
        affected_vis=affected_vis,
        message=message,
        since=since,
        acknowledged=acknowledged,
    )


# ----------------------------------------------------------------------
# Condition validation
# ----------------------------------------------------------------------


def test_valid_hold_condition_constructs():
    c = _condition("safety:coolant_low", severity="hold", affected_vis=frozenset({"monitor_1"}))
    assert c.severity == "hold"
    assert c.affected_vis == frozenset({"monitor_1"})


def test_valid_critical_condition_constructs():
    c = _condition("envelope:field_too_high", severity="critical", affected_vis=None)
    assert c.severity == "critical"
    assert c.affected_vis is None


def test_valid_advisory_condition_constructs_with_either_affected_vis():
    assert _condition("a1", severity="advisory", affected_vis=None).affected_vis is None
    c2 = _condition("a2", severity="advisory", affected_vis=frozenset({"x"}))
    assert c2.affected_vis == frozenset({"x"})


def test_bad_severity_raises_value_error():
    with pytest.raises(ValueError, match="severity"):
        _condition("bad", severity="urgent", affected_vis=None)


def test_bad_origin_raises_value_error():
    with pytest.raises(ValueError, match="origin"):
        Condition(
            key="bad",
            origin="hardware",
            severity="hold",
            kind="x",
            source_vis=(),
            affected_vis=frozenset({"vi"}),
            message="",
            since=0.0,
        )


def test_empty_key_raises_value_error():
    with pytest.raises(ValueError, match="key"):
        _condition("", severity="advisory", affected_vis=None)


def test_critical_with_affected_vis_raises_value_error():
    with pytest.raises(ValueError, match="critical"):
        _condition("bad", severity="critical", affected_vis=frozenset({"vi"}))


def test_hold_without_affected_vis_raises_value_error():
    with pytest.raises(ValueError, match="hold"):
        _condition("bad", severity="hold", affected_vis=None)


def test_hold_with_empty_affected_vis_raises_value_error():
    with pytest.raises(ValueError, match="hold"):
        _condition("bad", severity="hold", affected_vis=frozenset())


def test_severities_and_origins_are_the_documented_tuples():
    assert SEVERITIES == ("advisory", "hold", "critical")
    assert ORIGINS == ("comm", "safety", "envelope", "trend")


def test_condition_is_frozen_and_hashable():
    c = _condition("safety:coolant_low", severity="hold", affected_vis=frozenset({"monitor_1"}))
    with pytest.raises(AttributeError):
        c.acknowledged = True  # type: ignore[misc]
    # Hashable: usable in a set/dict key, and equal conditions hash equal.
    c2 = _condition("safety:coolant_low", severity="hold", affected_vis=frozenset({"monitor_1"}))
    assert c == c2
    assert hash(c) == hash(c2)
    assert {c, c2} == {c}


# ----------------------------------------------------------------------
# decide()
# ----------------------------------------------------------------------


def test_decide_empty_input_yields_empty_verdict():
    verdict = decide([], watched_vis=set(), run_active=False)
    assert verdict.held_vis == {}
    assert verdict.emergency == ()
    assert verdict.run_failure is None


def test_decide_hold_expands_to_every_affected_vi():
    c = _condition(
        "comm:magnet_z", origin="comm", kind="stale", severity="hold",
        affected_vis=frozenset({"magnet_z", "magnet_y"}),
    )
    verdict = decide([c], watched_vis=set(), run_active=False)
    assert verdict.held_vis == {"magnet_z": c, "magnet_y": c}


def test_decide_overlapping_holds_first_sorted_key_wins():
    # "hold:a" sorts before "hold:b"; both affect "shared".
    c_a = _condition("hold:a", affected_vis=frozenset({"shared", "only_a"}))
    c_b = _condition("hold:b", affected_vis=frozenset({"shared", "only_b"}))
    verdict = decide([c_b, c_a], watched_vis=set(), run_active=False)
    assert verdict.held_vis["shared"] is c_a
    assert verdict.held_vis["only_a"] is c_a
    assert verdict.held_vis["only_b"] is c_b


def test_decide_critical_conditions_become_sorted_emergency():
    c_z = _condition("envelope:z", severity="critical", affected_vis=None)
    c_a = _condition("envelope:a", severity="critical", affected_vis=None)
    verdict = decide([c_z, c_a], watched_vis=set(), run_active=False)
    assert verdict.emergency == (c_a, c_z)


def test_decide_advisory_conditions_produce_no_enforcement():
    c = _condition("safety:minor", severity="advisory", affected_vis=None)
    verdict = decide([c], watched_vis={"anything"}, run_active=True)
    assert verdict.held_vis == {}
    assert verdict.emergency == ()
    assert verdict.run_failure is None


def test_decide_run_failure_requires_run_active():
    c = _condition("comm:magnet_z", affected_vis=frozenset({"magnet_z"}))
    verdict = decide([c], watched_vis={"magnet_z"}, run_active=False)
    assert verdict.held_vis == {"magnet_z": c}
    assert verdict.run_failure is None


def test_decide_run_failure_picks_alphabetically_first_watched_held_vi():
    c = _condition("comm:multi", affected_vis=frozenset({"z_vi", "a_vi", "m_vi"}))
    verdict = decide([c], watched_vis={"z_vi", "a_vi", "m_vi"}, run_active=True)
    assert verdict.run_failure == ("a_vi", c)


def test_decide_run_failure_none_when_watched_disjoint_from_held():
    c = _condition("comm:magnet_z", affected_vis=frozenset({"magnet_z"}))
    verdict = decide([c], watched_vis={"temperature"}, run_active=True)
    assert verdict.held_vis == {"magnet_z": c}
    assert verdict.run_failure is None


def test_decide_mixed_critical_and_hold_both_reported():
    hold = _condition("comm:magnet_z", affected_vis=frozenset({"magnet_z"}))
    critical = _condition("envelope:over_bound", severity="critical", affected_vis=None)
    verdict = decide([hold, critical], watched_vis={"magnet_z"}, run_active=True)
    assert verdict.emergency == (critical,)
    assert verdict.held_vis == {"magnet_z": hold}
    assert verdict.run_failure == ("magnet_z", hold)


def test_decide_is_order_independent():
    c1 = _condition("hold:a", affected_vis=frozenset({"vi_a"}))
    c2 = _condition("hold:b", affected_vis=frozenset({"vi_b"}))
    v_forward = decide([c1, c2], watched_vis={"vi_a", "vi_b"}, run_active=True)
    v_backward = decide([c2, c1], watched_vis={"vi_a", "vi_b"}, run_active=True)
    assert v_forward == v_backward


# ----------------------------------------------------------------------
# envelope_conditions()
# ----------------------------------------------------------------------


def test_envelope_conditions_field_mapping():
    now = 1234.5
    result = envelope_conditions(["field exceeds 1.0 T bound"], now)
    assert len(result) == 1
    c = result[0]
    assert c.key == "envelope:field exceeds 1.0 T bound"
    assert c.origin == "envelope"
    assert c.severity == "critical"
    assert c.kind == "envelope"
    assert c.source_vis == ()
    assert c.affected_vis is None
    assert c.message == "field exceeds 1.0 T bound"
    assert c.since == now


def test_envelope_conditions_one_per_violation_same_order():
    result = envelope_conditions(["first violation", "second violation"], 0.0)
    assert [c.message for c in result] == ["first violation", "second violation"]


def test_envelope_conditions_empty_list_yields_empty_list():
    assert envelope_conditions([], 0.0) == []


def test_envelope_conditions_are_valid_conditions_usable_by_decide():
    conditions = envelope_conditions(["over bound"], 100.0)
    verdict = decide(conditions, watched_vis=set(), run_active=False)
    assert verdict.emergency == tuple(conditions)
