"""The **Agent feed** (L6): one experiment's non-operator action trail.

The accountability question this answers is "what did the machines ask for,
what were they told, and what happened as a result" — so the tests are
written against that question rather than against the file format: a command
and its verdict must join on the request id, the physicist's own actions must
stay out, and an agent that skips the gateway must still be in the file.
"""

from __future__ import annotations

import json

import pytest

from i2as.core import events as ev
from i2as.core.station import build_station
from i2as.session.agent_feed import (
    RECORD_COMMAND,
    RECORD_EVENT,
    RECORD_VERDICT,
    SCHEMA_VERSION,
    AgentFeed,
    read_feed,
)
from i2as.session.gateway import Gateway, Role
from i2as.session.store import ExperimentStore

AGENT = ev.Actor(kind=ev.ActorKind.AGENT, id="runner-7", role="session")
OBSERVER = ev.Actor(kind=ev.ActorKind.AGENT, id="watcher", role="observer")


class _Signal:
    """The smallest thing that satisfies the connect/emit duck type."""

    def __init__(self) -> None:
        self._slots: list = []

    def connect(self, slot) -> None:
        self._slots.append(slot)

    def emit(self, payload) -> None:
        for slot in self._slots:
            slot(payload)


class _FakeEngine:
    """An engine that answers every command OK, with no hardware behind it."""

    def __init__(self) -> None:
        self.verdict_emitted = _Signal()
        self.event_emitted = _Signal()
        self.submitted: list[ev.Command] = []
        self._seq = 0

    def station_info(self) -> ev.StationInfo:
        return ev.StationInfo()

    def submit(self, command: ev.Command) -> str:
        self.submitted.append(command)
        self._seq += 1
        self.verdict_emitted.emit(
            ev.Verdict(
                request_id=command.request_id,
                command=command.name,
                code=ev.VerdictCode.OK,
                actor=command.actor,
                seq=self._seq,
            )
        )
        return command.request_id


@pytest.fixture
def feed(tmp_path) -> AgentFeed:
    """A feed on a throwaway experiment folder."""
    store = ExperimentStore(tmp_path / "session")
    return AgentFeed(store.agent_feed_path("20260903_demo"), "20260903_demo")


def _records(feed: AgentFeed, record: str | None = None) -> list[dict]:
    """Every record in the feed, optionally of one kind only."""
    return [
        entry
        for entry in read_feed(feed.path)
        if record is None or entry["record"] == record
    ]


# ── Where the file lives ──────────────────────────────────────────────────


def test_the_store_owns_where_the_feed_sits(tmp_path):
    """The trail lives inside the experiment folder, like the outbox."""
    store = ExperimentStore(tmp_path / "session")

    path = store.agent_feed_path("20260903_demo")

    assert path == tmp_path / "session" / "20260903_demo" / "agent_actions.jsonl"
    assert not path.exists(), "nothing is created until something is recorded"


# ── A gateway submission: one command, one verdict, one request id ────────


def test_a_gateway_submission_writes_one_command_and_one_verdict(feed):
    """The exit criterion: the two halves of one action, joined by request id."""
    engine = _FakeEngine()
    feed.attach(engine)
    gateway = Gateway(engine, Role.SESSION, "runner-7", feed=feed)

    request_id = gateway.submit(
        ev.CommandName.RUN_PROCEDURE, {"procedure": "FieldSweep"}
    )

    commands = _records(feed, RECORD_COMMAND)
    verdicts = _records(feed, RECORD_VERDICT)
    assert len(commands) == 1
    assert len(verdicts) == 1
    assert commands[0]["request_id"] == verdicts[0]["request_id"] == request_id
    assert commands[0]["command"] == "run_procedure"
    assert commands[0]["args"] == {"procedure": "FieldSweep"}
    assert commands[0]["actor"] == AGENT.to_json()
    assert verdicts[0]["verdict"] == {"code": "OK", "reason": ""}


def test_every_record_carries_the_whole_standard(feed):
    """A key that does not apply is null, never absent (the record standard)."""
    engine = _FakeEngine()
    feed.attach(engine)
    Gateway(engine, Role.SESSION, "runner-7", feed=feed).submit(
        ev.CommandName.START_MONITORING
    )

    for entry in read_feed(feed.path):
        assert set(entry) == {
            "schema",
            "ts",
            "seq",
            "experiment_id",
            "run_id",
            "record",
            "actor",
            "request_id",
            "command",
            "tool",
            "args",
            "event",
            "detail",
            "verdict",
        }
        assert entry["schema"] == SCHEMA_VERSION
        assert entry["experiment_id"] == "20260903_demo"
        assert entry["ts"] > 0


def test_a_refused_command_is_still_in_the_trail(feed):
    """What an agent tried is as much a part of accountability as what it did."""
    engine = _FakeEngine()
    feed.attach(engine)
    gateway = Gateway(engine, Role.OBSERVER, "watcher", feed=feed)

    request_id = gateway.submit(ev.CommandName.RUN_PROCEDURE)

    assert engine.submitted == [], "a refusal never reaches the engine"
    commands = _records(feed, RECORD_COMMAND)
    verdicts = _records(feed, RECORD_VERDICT)
    assert [entry["request_id"] for entry in commands] == [request_id]
    assert [entry["request_id"] for entry in verdicts] == [request_id]
    assert verdicts[0]["verdict"]["code"] == "BLOCKED_ROLE"
    assert verdicts[0]["detail"]["rule"] == "role_matrix"


def test_permits_provokes_no_record(feed):
    """Asking whether a command would be allowed is not taking an action."""
    engine = _FakeEngine()
    feed.attach(engine)
    gateway = Gateway(engine, Role.OBSERVER, "watcher", feed=feed)

    gateway.permits(ev.CommandName.RUN_PROCEDURE)

    assert read_feed(feed.path) == []


# ── Who is recorded, and who is not ───────────────────────────────────────


def test_the_physicists_own_actions_are_not_recorded(feed):
    """This file answers "what did the machines do"; the operator is elsewhere."""
    engine = _FakeEngine()
    feed.attach(engine)

    engine.submit(ev.Command(name=ev.CommandName.START_MONITORING))
    feed.record_command(ev.Command(name=ev.CommandName.ABORT_PROCEDURE))

    assert read_feed(feed.path) == []


def test_an_agent_that_skips_the_gateway_is_still_recorded(feed):
    """The actor kind travels on the message, not on the connection."""
    engine = _FakeEngine()
    feed.attach(engine)

    request_id = engine.submit(
        ev.Command(name=ev.CommandName.STOP_RAMP, actor=AGENT, args={"vi_name": "magnet_z"})
    )

    verdicts = _records(feed, RECORD_VERDICT)
    assert [entry["request_id"] for entry in verdicts] == [request_id]
    assert verdicts[0]["actor"]["kind"] == "agent"
    assert verdicts[0]["command"] == "stop_ramp"
    assert _records(feed, RECORD_COMMAND) == [], (
        "no command record is invented: the arguments were never seen here"
    )


def test_a_state_change_an_agent_caused_is_recorded(feed):
    """The consequence, not only the request — joined on the same request id."""
    engine = _FakeEngine()
    feed.attach(engine)

    engine.event_emitted.emit(
        ev.StateChange(
            state="RAMPING",
            previous="IDLE",
            cause="run_procedure",
            actor=AGENT,
            request_id="req-1",
            seq=4,
        )
    )
    engine.event_emitted.emit(
        ev.StateChange(state="IDLE", previous="RAMPING", cause="tick", seq=5)
    )

    events = _records(feed, RECORD_EVENT)
    assert len(events) == 1, "an operator/system transition is not this file's business"
    assert events[0]["event"] == "state_change"
    assert events[0]["request_id"] == "req-1"
    assert events[0]["detail"] == {
        "state": "RAMPING",
        "previous": "IDLE",
        "cause": "run_procedure",
    }


def test_records_are_stamped_with_the_run_in_flight(feed):
    """"Which run was this during" is answered by the record itself."""
    engine = _FakeEngine()
    feed.attach(engine)

    engine.event_emitted.emit(ev.RunStarted(run_id="20260903_120000_001_sweep"))
    engine.submit(ev.Command(name=ev.CommandName.PAUSE_PROCEDURE, actor=AGENT))
    engine.event_emitted.emit(ev.RunFinished(run_id="20260903_120000_001_sweep"))
    engine.submit(ev.Command(name=ev.CommandName.START_MONITORING, actor=AGENT))

    during, after = _records(feed, RECORD_VERDICT)
    assert during["run_id"] == "20260903_120000_001_sweep"
    assert after["run_id"] is None


# ── The journal discipline ────────────────────────────────────────────────


def test_the_sequence_continues_where_an_earlier_process_left_off(feed):
    """A feed reopened later keeps counting, so the trail stays orderable."""
    engine = _FakeEngine()
    feed.attach(engine)
    engine.submit(ev.Command(name=ev.CommandName.START_MONITORING, actor=AGENT))

    reopened = AgentFeed(feed.path, feed.experiment_id)
    reopened.record_command(ev.Command(name=ev.CommandName.STOP_MONITORING, actor=AGENT))

    assert [entry["seq"] for entry in read_feed(feed.path)] == [1, 2]


def test_read_feed_polls_from_a_sequence_number(feed):
    """A client follows the trail without re-reading what it already has."""
    engine = _FakeEngine()
    feed.attach(engine)
    for _ in range(3):
        engine.submit(ev.Command(name=ev.CommandName.START_MONITORING, actor=AGENT))

    assert [entry["seq"] for entry in read_feed(feed.path, since_seq=1)] == [2, 3]
    assert read_feed(feed.path, since_seq=99) == []


def test_a_corrupt_line_never_strands_the_rest_of_the_trail(feed):
    """One mangled line is skipped, exactly as in the store and the outbox."""
    engine = _FakeEngine()
    feed.attach(engine)
    engine.submit(ev.Command(name=ev.CommandName.START_MONITORING, actor=AGENT))
    with feed.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")
    engine.submit(ev.Command(name=ev.CommandName.STOP_MONITORING, actor=AGENT))

    records = read_feed(feed.path)

    assert [entry["command"] for entry in records] == [
        "start_monitoring",
        "stop_monitoring",
    ]


def test_recording_never_raises_at_the_engine(tmp_path):
    """A feed that cannot be written degrades to a log line, never an exception."""
    unwritable = tmp_path / "not_a_directory"
    unwritable.write_text("", encoding="utf-8")
    feed = AgentFeed(unwritable / "agent_actions.jsonl", "20260903_demo")
    engine = _FakeEngine()
    feed.attach(engine)

    engine.submit(ev.Command(name=ev.CommandName.START_MONITORING, actor=AGENT))

    assert read_feed(feed.path) == []


def test_every_line_is_json_and_the_file_is_append_only(feed):
    """The record standard: one JSON object per line, nothing ever rewritten."""
    engine = _FakeEngine()
    feed.attach(engine)
    engine.submit(ev.Command(name=ev.CommandName.START_MONITORING, actor=AGENT))
    first = feed.path.read_text(encoding="utf-8")
    engine.submit(ev.Command(name=ev.CommandName.STOP_MONITORING, actor=AGENT))

    text = feed.path.read_text(encoding="utf-8")

    assert text.startswith(first), "an existing line is never rewritten"
    assert [isinstance(json.loads(line), dict) for line in text.splitlines()] == [
        True,
        True,
    ]


# ── Against the real engine ───────────────────────────────────────────────


@pytest.fixture
def orchestrator(qtbot):
    """A real Orchestrator over a real simulated station."""
    from i2as.core.orchestrator import Orchestrator

    station = build_station("i2as/configs/sim_cryostat")
    orch = Orchestrator(station, tick_interval_ms=10)
    yield orch, station
    orch.shutdown()


def test_the_real_engine_fills_the_trail_end_to_end(orchestrator, feed):
    """A real command through a real gateway leaves both halves in the file."""
    orch, station = orchestrator
    feed.attach(orch)
    gateway = Gateway(
        orch, Role.SESSION, "runner-7", station_info=station.station_info, feed=feed
    )

    request_id = gateway.submit(ev.CommandName.START_MONITORING)

    commands = _records(feed, RECORD_COMMAND)
    verdicts = _records(feed, RECORD_VERDICT)
    assert [entry["request_id"] for entry in commands] == [request_id]
    assert [entry["request_id"] for entry in verdicts] == [request_id]
    assert verdicts[0]["verdict"]["code"] == "OK"
    assert orch.is_monitoring()


def test_the_two_trails_join_on_the_request_id(orchestrator, feed):
    """The join the whole scheme rests on: one id, two independent records.

    The Agent feed says an agent asked for something and what it was told; the
    operational-status record says what the station was doing and who last got
    it to act. Neither is derived from the other — one is written by the
    session layer's feed, the other assembled by the engine's own tick — so
    the request id being the same value in both is what lets a physicist read
    "the station started monitoring at 03:12" and "runner-7 asked for it" as
    one fact.
    """
    orch, station = orchestrator
    feed.attach(orch)
    gateway = Gateway(
        orch, Role.SESSION, "runner-7", station_info=station.station_info, feed=feed
    )

    request_id = gateway.submit(ev.CommandName.START_MONITORING)
    orch._tick()

    status = orch.get_operational_status()
    assert status["request_id"] == request_id
    assert status["actor"] == {"kind": "agent", "id": "runner-7", "role": "session"}

    trail = [entry for entry in read_feed(feed.path) if entry["request_id"] == request_id]
    assert [entry["record"] for entry in trail] == [RECORD_COMMAND, RECORD_VERDICT]


def test_a_refused_command_does_not_claim_the_station(orchestrator, feed):
    """A refusal changed nothing, so it never displaces who last acted."""
    orch, station = orchestrator
    feed.attach(orch)
    gateway = Gateway(
        orch, Role.SESSION, "runner-7", station_info=station.station_info, feed=feed
    )

    accepted = gateway.submit(ev.CommandName.START_MONITORING)
    orch.set_agent_gate(ev.AgentGate.REVOKED)
    refused = gateway.submit(ev.CommandName.STOP_MONITORING)
    orch._tick()

    assert refused != accepted
    assert orch.get_operational_status()["request_id"] == accepted
