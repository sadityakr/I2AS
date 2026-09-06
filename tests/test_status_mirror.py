"""StatusMirror tests — the GUI's local copy of the engine's picture.

The status-mirror standard has three properties worth pinning: it answers
every read the engine exposes, it answers them from the last event it saw
rather than by calling in, and nothing it hands out is a live container the
mirror (or the engine) can mutate afterwards.
"""

import dataclasses

import pytest

from i2as.core import events as ev
from i2as.core.orchestrator import Orchestrator
from i2as.core.station import build_station
from i2as.core.status_mirror import StatusMirror

CONFIG_PATH = "i2as/configs/sim_cryostat"


@pytest.fixture
def station():
    """Real simulated station from sim_cryostat config."""
    return build_station(CONFIG_PATH)


@pytest.fixture
def engine(station, qtbot):
    """Orchestrator with a short tick, torn down with its timer stopped."""
    orch = Orchestrator(station, tick_interval_ms=50)
    yield orch
    orch.shutdown()


@pytest.fixture
def mirror(engine):
    """A mirror primed from, and attached to, the engine."""
    return StatusMirror.for_engine(engine)


def test_for_engine_primes_from_the_two_priming_reads(engine):
    """A mirror starts on the state the engine is already in, not a default."""
    engine._state = engine._state.__class__("EMERGENCY")
    mirror = StatusMirror.for_engine(engine)
    assert mirror.state == "EMERGENCY"
    assert {info.name for info in mirror.station_info().instruments}
    assert mirror.instrument_info("magnet_z") is not None


def test_every_snapshot_field_has_a_mirror_read(mirror, engine):
    """One mirror read per ``StatusSnapshot`` field, named after the accessor.

    The engine's read surface and the snapshot are already diffed by
    conformance; this closes the third side of the triangle, so a field added
    to the snapshot cannot arrive without a way for a widget to read it.
    """
    housekeeping = {"seq", "ts", "instruments"}
    fields = {f.name for f in dataclasses.fields(ev.StatusSnapshot)} - housekeeping
    missing = {name for name in fields if not hasattr(mirror, name)}
    assert not missing, f"StatusSnapshot fields with no mirror read: {sorted(missing)}"


def test_reads_follow_the_event_stream(mirror, engine):
    """Every read tracks the last snapshot the engine broadcast."""
    engine.start_monitoring()
    engine._tick()

    assert mirror.is_monitoring() is True
    assert mirror.state == engine.state
    assert set(mirror.availabilities()) == set(engine.availabilities())
    assert mirror.active_run_kind() is None
    assert mirror.held_vi_names() == frozenset(engine.held_vi_names())

    engine.stop_monitoring()
    engine._tick()
    assert mirror.is_monitoring() is False


def test_state_change_and_station_info_reach_their_signals(mirror, engine, qtbot):
    """The mirror re-broadcasts each event type it absorbs."""
    with qtbot.waitSignal(mirror.state_changed, timeout=500) as change:
        engine._change_state(engine._state.__class__("ERROR"), cause="test")
    assert isinstance(change.args[0], ev.StateChange)
    assert change.args[0].state == "ERROR"

    with qtbot.waitSignal(mirror.station_updated, timeout=500) as declared:
        engine._emit_station_info()
    assert isinstance(declared.args[0], ev.StationInfo)
    assert mirror.station_info() is declared.args[0]


def test_a_disconnect_updates_the_declaration_the_mirror_holds(mirror, engine):
    """The declaration a panel builds from follows connect/disconnect."""
    before = mirror.instrument_info("magnet_z")
    assert before is not None
    engine.disconnect_instrument("magnet_z")
    after = mirror.instrument_info("magnet_z")
    assert after is not None, "an offline instrument is still declared"
    assert after is not before
    assert "operator" in after.availability


def test_operational_status_is_mirrored_from_its_own_stream(mirror, engine):
    """The per-tick troubleshooting record travels on its own signal."""
    assert mirror.get_operational_status() == {}
    engine._tick()
    record = mirror.get_operational_status()
    assert record, "a tick delivers a record"
    assert record == engine.get_operational_status()


def test_nothing_the_mirror_hands_out_is_a_live_container(mirror, engine):
    """Every read returns a copy: a caller's edit can never reach the mirror."""
    engine.start_monitoring()
    engine._tick()

    availabilities = mirror.availabilities()
    availabilities.clear()
    assert mirror.availabilities(), "the map handed out was the mirror's own"

    record = mirror.availability("magnet_z")
    record["tags"] = ["tampered"]
    assert mirror.availability_tags("magnet_z") != frozenset({"tampered"})

    status = mirror.get_operational_status()
    status["tampered"] = True
    assert "tampered" not in mirror.get_operational_status()


def test_unknown_instruments_answer_empty_rather_than_raising(mirror):
    """A read about an instrument the snapshot never carried is not an error."""
    assert mirror.availability("nope") == {}
    assert mirror.availability_tags("nope") == frozenset()
    assert mirror.vi_fault("nope") is None
    assert mirror.offline_reason("nope") == ""
    assert mirror.instrument_info("nope") is None
    assert mirror.lifecycle_state("nope") == ev.LifecycleState.IDLE.value


def test_lifecycle_state_is_mirrored_from_the_snapshot(mirror, engine, station):
    """The card's fact travels the contract, not the action history.

    Stands the station down the way an emergency does — straight through
    ``Station.standby_all()``, dispatching no per-VI action — and requires
    the next snapshot to say so.
    """
    station.initiate_all()
    engine._emit_status_snapshot()
    assert mirror.lifecycle_state("magnet_z") == "initiated"

    station.standby_all()
    engine._emit_status_snapshot()
    assert mirror.lifecycle_state("magnet_z") == "standby"
    assert all(
        mirror.lifecycle_state(name) == "standby" for name in station.get_vi_names()
    )
