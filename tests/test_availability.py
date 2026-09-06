"""Tests for the Availability standard's policy core and its real mechanisms."""

from __future__ import annotations

import pytest
from i2as.core.availability import (
    AVAILABILITY_STATES,
    AVAILABILITY_TAGS,
    TAG_POLICY,
    TAG_PRECEDENCE,
    Availability,
    decide_availability,
    state_for,
)
from i2as.core.conditions import decide
from i2as.core.orchestrator import Orchestrator
from i2as.core.station import Station, build_station
from tests.test_connection_lifecycle import _DetachableDriver, _DetachWhenIdleNoOverrideVI
from tests.test_l3_orchestrator import _degraded_station

SIM_CONFIG = "i2as/configs/sim_cryostat"


# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------


def test_availability_tags_is_the_closed_vocabulary():
    assert AVAILABILITY_TAGS == ("connect_failed", "operator", "not_responding", "detached")


def test_availability_states_are_the_four_mutually_exclusive_situations():
    assert AVAILABILITY_STATES == ("live", "detached", "faulted", "absent")


def test_tag_policy_covers_exactly_availability_tags():
    assert set(TAG_POLICY.keys()) == set(AVAILABILITY_TAGS)
    for tag, policy in TAG_POLICY.items():
        assert policy.tag == tag


def test_tag_policy_states_are_all_in_vocabulary():
    for policy in TAG_POLICY.values():
        assert policy.state in AVAILABILITY_STATES


def test_tag_precedence_is_a_permutation_of_availability_tags():
    assert sorted(TAG_PRECEDENCE) == sorted(AVAILABILITY_TAGS)
    assert len(TAG_PRECEDENCE) == len(set(TAG_PRECEDENCE))


def test_tag_policy_table_matches_the_declared_rows():
    assert TAG_POLICY["connect_failed"].state == "absent"
    assert TAG_POLICY["connect_failed"].enumerable is False
    assert TAG_POLICY["connect_failed"].controllable is False
    assert TAG_POLICY["connect_failed"].fails_claimed_run is False
    assert TAG_POLICY["connect_failed"].raises_error_event is False
    assert TAG_POLICY["connect_failed"].recovery == "connect"

    assert TAG_POLICY["operator"].state == "absent"
    assert TAG_POLICY["operator"].enumerable is False
    assert TAG_POLICY["operator"].controllable is False
    assert TAG_POLICY["operator"].fails_claimed_run is False
    assert TAG_POLICY["operator"].raises_error_event is False
    assert TAG_POLICY["operator"].recovery == "connect"

    assert TAG_POLICY["not_responding"].state == "faulted"
    assert TAG_POLICY["not_responding"].enumerable is True
    assert TAG_POLICY["not_responding"].controllable is False
    assert TAG_POLICY["not_responding"].fails_claimed_run is True
    assert TAG_POLICY["not_responding"].raises_error_event is True
    assert TAG_POLICY["not_responding"].recovery == "retry"

    assert TAG_POLICY["detached"].state == "detached"
    assert TAG_POLICY["detached"].enumerable is True
    assert TAG_POLICY["detached"].controllable is True
    assert TAG_POLICY["detached"].fails_claimed_run is False
    assert TAG_POLICY["detached"].raises_error_event is False
    assert TAG_POLICY["detached"].recovery == ""


# ----------------------------------------------------------------------
# state_for()
# ----------------------------------------------------------------------


def test_state_for_empty_tags_is_live():
    assert state_for(frozenset()) == "live"


@pytest.mark.parametrize(
    "tag,expected_state",
    [
        ("connect_failed", "absent"),
        ("operator", "absent"),
        ("not_responding", "faulted"),
        ("detached", "detached"),
    ],
)
def test_state_for_single_tag(tag, expected_state):
    assert state_for(frozenset({tag})) == expected_state


def test_state_for_operator_and_connect_failed_resolves_to_absent():
    # The bug-fix scenario: an operator-released VI whose reconnect then
    # fails on hardware. Both tags imply "absent" so the state agrees
    # regardless of which wins precedence.
    assert state_for(frozenset({"operator", "connect_failed"})) == "absent"


def test_state_for_not_responding_and_detached_prefers_the_more_restrictive_tag():
    # Per TAG_PRECEDENCE, "not_responding" is more restrictive than
    # "detached" and wins.
    assert state_for(frozenset({"not_responding", "detached"})) == "faulted"


# ----------------------------------------------------------------------
# decide_availability()
# ----------------------------------------------------------------------


def test_decide_availability_empty_tags_is_none():
    assert decide_availability(frozenset()) is None


def test_decide_availability_single_tag_returns_its_policy():
    policy = decide_availability(frozenset({"not_responding"}))
    assert policy is TAG_POLICY["not_responding"]


def test_decide_availability_operator_and_connect_failed_prefers_connect_failed():
    policy = decide_availability(frozenset({"operator", "connect_failed"}))
    assert policy is TAG_POLICY["connect_failed"]


def test_decide_availability_not_responding_and_detached_prefers_not_responding():
    policy = decide_availability(frozenset({"not_responding", "detached"}))
    assert policy is TAG_POLICY["not_responding"]


# ----------------------------------------------------------------------
# Availability construction/validation
# ----------------------------------------------------------------------


def test_availability_live_vi_constructs():
    a = Availability(
        vi_name="magnet_z",
        vi_type="system",
        state="live",
        tags=frozenset(),
        reason="",
        since=0.0,
    )
    assert a.state == "live"
    assert a.tags == frozenset()


def test_availability_absent_vi_constructs():
    a = Availability(
        vi_name="magnet_z",
        vi_type="system",
        state="absent",
        tags=frozenset({"operator"}),
        reason="Disconnected by the operator",
        since=123.0,
    )
    assert a.state == "absent"


def test_availability_rejects_unknown_tag():
    with pytest.raises(ValueError):
        Availability(
            vi_name="magnet_z",
            vi_type="system",
            state="live",
            tags=frozenset({"bogus_tag"}),
            reason="",
            since=0.0,
        )


def test_availability_rejects_unknown_state():
    with pytest.raises(ValueError):
        Availability(
            vi_name="magnet_z",
            vi_type="system",
            state="bogus_state",
            tags=frozenset(),
            reason="",
            since=0.0,
        )


def test_availability_rejects_state_disagreeing_with_tags():
    with pytest.raises(ValueError):
        Availability(
            vi_name="magnet_z",
            vi_type="system",
            state="live",
            tags=frozenset({"operator"}),
            reason="",
            since=0.0,
        )


def test_availability_rejects_absent_state_with_no_tags():
    with pytest.raises(ValueError):
        Availability(
            vi_name="magnet_z",
            vi_type="system",
            state="absent",
            tags=frozenset(),
            reason="",
            since=0.0,
        )


# ============================================================================
# Tag policy — mechanism checks
# ============================================================================
# TAG_POLICY's rows are DESCRIPTIONS of enforcement that lives elsewhere
# (registry membership, i2as.core.conditions.decide(), the Orchestrator's
# onset diff, the Availability standard's recovery actions) — the table
# itself enforces nothing, by design (see core/availability.py's module
# docstring: "the enforcement... live above this module... out of scope
# here"). These tests exist so a row that no longer matches that real
# mechanism fails CI instead of quietly becoming decoration.
#
# `.controllable` already has this kind of coverage for "not_responding" and
# "detached" in tests/test_l3_orchestrator.py
# (test_not_responding_refuses_control_via_tag_policy,
# test_detached_vi_admitted_by_tag_policy) — not duplicated here. The
# "absent" tags (operator, connect_failed) are covered below, against the
# mechanism that actually stands in for _manual_action_admissible() once a
# VI is not live at all: Station.execute_vi_action() has no VI to dispatch
# to.


@pytest.fixture
def station() -> Station:
    """A full sim station, built the way the app builds it."""
    return build_station(SIM_CONFIG)


@pytest.fixture
def orchestrator(station, qtbot):
    """An Orchestrator over the sim station, monitoring started.

    ``qtbot`` is required even where a test never waits on a signal: it is
    what guarantees a ``QApplication`` exists for the ``Orchestrator``
    (a ``QObject``) to construct against.
    """
    orch = Orchestrator(station, tick_interval_ms=10)
    orch.start_monitoring()
    yield orch
    orch.shutdown()


# ----------------------------------------------------------------------
# enumerable: membership in Station.get_vi_names() while the tag holds
# ----------------------------------------------------------------------


def test_enumerable_false_for_connect_failed_vi(tmp_path):
    """A VI whose driver never connected at build time is absent from the roster."""
    degraded = _degraded_station(tmp_path)
    assert "connect_failed" in degraded.availability("flaky_vi").tags
    assert (
        "flaky_vi" in degraded.get_vi_names()
    ) == TAG_POLICY["connect_failed"].enumerable


def test_enumerable_false_for_operator_disconnected_vi(station: Station):
    """An operator-released VI is absent from the roster, just like connect_failed."""
    ok, _ = station.disconnect_instrument("temperature")
    assert ok
    assert "operator" in station.availability("temperature").tags
    assert ("temperature" in station.get_vi_names()) == TAG_POLICY["operator"].enumerable


def test_enumerable_true_for_not_responding_vi(station: Station):
    """A faulted-but-live VI still appears in the roster — it is still held."""
    station.magnet_z._driver._simulate_error = True
    try:
        station.get_state()  # records the comm-origin condition
        assert "not_responding" in station.availability("magnet_z").tags
        assert (
            "magnet_z" in station.get_vi_names()
        ) == TAG_POLICY["not_responding"].enumerable
    finally:
        station.magnet_z._driver._simulate_error = False


def test_enumerable_true_for_detached_vi(station: Station):
    """A detach_when_idle VI mid-detach still appears in the roster."""
    vi = station.get_vi("dc_measurement")
    vi.is_attached = lambda: False
    try:
        assert "detached" in station.availability("dc_measurement").tags
        assert (
            "dc_measurement" in station.get_vi_names()
        ) == TAG_POLICY["detached"].enumerable
    finally:
        del vi.is_attached


# ----------------------------------------------------------------------
# controllable: the "absent" tags (not_responding/detached are covered in
# tests/test_l3_orchestrator.py directly against
# Orchestrator._manual_action_admissible())
# ----------------------------------------------------------------------


def test_controllable_false_for_absent_vi_cannot_be_dispatched_to(station: Station):
    """An absent VI cannot be dispatched to at all — there is nothing to admit.

    Unlike ``not_responding``/``detached`` (live, held VIs refused or
    admitted by ``Orchestrator._manual_action_admissible()``), an absent VI
    is not in the live registry, so there is no admission call to make for
    it — ``Station.execute_vi_action()`` (what a GUI action ultimately
    dispatches to) simply has no VI to find.
    """
    assert TAG_POLICY["connect_failed"].controllable is False
    assert TAG_POLICY["operator"].controllable is False

    ok, _ = station.disconnect_instrument("temperature")
    assert ok
    with pytest.raises(KeyError):
        station.execute_vi_action("temperature", "initiate")


# ----------------------------------------------------------------------
# fails_claimed_run: agreement with i2as.core.conditions.decide()
# ----------------------------------------------------------------------


def test_fails_claimed_run_true_for_not_responding(station: Station):
    """A not_responding VI produces a hold Condition that decide() fails a watched run on."""
    station.magnet_z._driver._simulate_error = True
    try:
        station.get_state()
        assert "not_responding" in station.availability("magnet_z").tags

        verdict = decide(
            station.conditions().values(), watched_vis=frozenset({"magnet_z"}), run_active=True
        )
        failed = verdict.run_failure is not None and verdict.run_failure[0] == "magnet_z"
        assert failed == TAG_POLICY["not_responding"].fails_claimed_run
    finally:
        station.magnet_z._driver._simulate_error = False


def test_fails_claimed_run_false_for_operator_and_detached(station: Station):
    """operator/detached never produce a Condition, so decide() cannot fail a run on them.

    Neither tag has a producer in the System-Condition standard
    (``core/conditions.py``): ``operator`` is an ``OfflineInstrument`` tag,
    ``detached`` is read straight off ``is_attached()`` — so a run
    "watching" either VI can never land in ``decide()``'s ``held_vis``.
    """
    ok, _ = station.disconnect_instrument("temperature")
    assert ok
    vi = station.get_vi("dc_measurement")
    vi.is_attached = lambda: False
    try:
        for vi_name, tag in (("temperature", "operator"), ("dc_measurement", "detached")):
            verdict = decide(
                station.conditions().values(), watched_vis=frozenset({vi_name}), run_active=True
            )
            failed = verdict.run_failure is not None and verdict.run_failure[0] == vi_name
            assert failed == TAG_POLICY[tag].fails_claimed_run, tag
    finally:
        del vi.is_attached


def test_fails_claimed_run_false_for_connect_failed(tmp_path):
    """connect_failed never produces a Condition either — same argument as operator."""
    degraded = _degraded_station(tmp_path)
    verdict = decide(
        degraded.conditions().values(), watched_vis=frozenset({"flaky_vi"}), run_active=True
    )
    failed = verdict.run_failure is not None and verdict.run_failure[0] == "flaky_vi"
    assert failed == TAG_POLICY["connect_failed"].fails_claimed_run


# ----------------------------------------------------------------------
# raises_error_event: agreement with the Orchestrator's onset diff
# (core/orchestrator.py's _tick_body — new comm-origin keys fire
# _emit_fault_event(); operator/connect_failed/detached never appear as a
# NEW condition key at all, so the onset diff never sees them)
# ----------------------------------------------------------------------


def test_raises_error_event_true_for_not_responding_onset(station: Station, orchestrator):
    """A newly-onset not_responding fault raises an ErrorEvent this same tick."""
    events: list = []
    orchestrator.error_event.connect(lambda ev: events.append(ev))

    station.magnet_z._driver._simulate_error = True
    try:
        orchestrator._tick()
        raised = any(ev.vi_name == "magnet_z" for ev in events)
        assert raised == TAG_POLICY["not_responding"].raises_error_event
    finally:
        station.magnet_z._driver._simulate_error = False


def test_raises_error_event_false_for_operator_and_detached_onset(
    station: Station, orchestrator
):
    """Disconnecting a VI, or detaching one, raises no ErrorEvent for it.

    Neither ever becomes a Condition, so the onset diff (which only ever
    inspects ``Station.conditions()``) never sees a new key for it.
    """
    events: list = []
    orchestrator.error_event.connect(lambda ev: events.append(ev))

    ok, _ = station.disconnect_instrument("temperature")
    assert ok
    orchestrator._tick()
    raised = any(ev.vi_name == "temperature" for ev in events)
    assert raised == TAG_POLICY["operator"].raises_error_event

    events.clear()
    vi = station.get_vi("dc_measurement")
    vi.is_attached = lambda: False
    try:
        orchestrator._tick()
        raised = any(ev.vi_name == "dc_measurement" for ev in events)
        assert raised == TAG_POLICY["detached"].raises_error_event
    finally:
        del vi.is_attached


def test_raises_error_event_false_for_connect_failed_onset(tmp_path, qtbot):
    """A VI offline from the very first build tick never raises an ErrorEvent for it."""
    degraded = _degraded_station(tmp_path)
    orch = Orchestrator(degraded, tick_interval_ms=10)
    orch.start_monitoring()
    events: list = []
    orch.error_event.connect(lambda ev: events.append(ev))
    try:
        orch._tick()
        raised = any(ev.vi_name == "flaky_vi" for ev in events)
        assert raised == TAG_POLICY["connect_failed"].raises_error_event
    finally:
        orch.shutdown()


# ----------------------------------------------------------------------
# recovery: the named action is the one that actually clears the tag
# ----------------------------------------------------------------------


def test_recovery_connect_clears_operator_tag(station: Station, orchestrator, qtbot):
    """recovery="connect" for operator: Orchestrator.connect_instrument() clears it."""
    assert TAG_POLICY["operator"].recovery == "connect"

    ok, _ = station.disconnect_instrument("temperature")
    assert ok
    assert "operator" in station.availability("temperature").tags

    with qtbot.waitSignal(orchestrator.instrument_reconnected, timeout=500):
        orchestrator.connect_instrument("temperature")

    assert station.availability("temperature").tags == frozenset()


def test_recovery_connect_clears_connect_failed_tag(tmp_path, qtbot):
    """recovery="connect" for connect_failed too — the same Connect action."""
    from tests.test_l2_station import _FlakyDriver

    assert TAG_POLICY["connect_failed"].recovery == "connect"

    degraded = _degraded_station(tmp_path)
    orch = Orchestrator(degraded, tick_interval_ms=10)
    try:
        assert "connect_failed" in degraded.availability("flaky_vi").tags
        _FlakyDriver.fail_times = 0  # the cable is back in for this attempt
        _FlakyDriver.attempts = 0

        with qtbot.waitSignal(orch.instrument_reconnected, timeout=500):
            orch.connect_instrument("flaky_vi")

        assert degraded.availability("flaky_vi").tags == frozenset()
    finally:
        orch.shutdown()


def test_recovery_retry_clears_not_responding_tag(station: Station, orchestrator):
    """recovery="retry" for not_responding: Orchestrator.retry_fault() clears it."""
    assert TAG_POLICY["not_responding"].recovery == "retry"

    station.magnet_z._driver._simulate_error = True
    station.get_state()
    assert "not_responding" in station.availability("magnet_z").tags

    station.magnet_z._driver._simulate_error = False  # the instrument recovered
    orchestrator.retry_fault("magnet_z")

    assert "not_responding" not in station.availability("magnet_z").tags


def test_recovery_automatic_for_detached_tag_needs_no_operator_action():
    """recovery="" for detached: ordinary re-use clears it, no dedicated action.

    Unlike ``connect_instrument()``/``retry_fault()``, nothing named
    "recover" or "retry" exists for a detach_when_idle VI — ``_attach()``
    (what ``initiate_measurement()`` calls to arm) reacquires the session
    as a side effect of ordinary use.
    """
    assert TAG_POLICY["detached"].recovery == ""

    driver = _DetachableDriver("SIM::TEST")
    vi = _DetachWhenIdleNoOverrideVI({"main": driver})
    vi._detach()
    assert vi.is_attached() is False

    vi._attach()

    assert vi.is_attached() is True
