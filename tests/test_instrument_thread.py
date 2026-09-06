"""The instrument thread — the boundary, exercised across a real ``QThread``.

Every test here builds an ``InstrumentHost(mode="threaded")`` over the sim
station, so the Station, the Orchestrator, the tick timer and every driver
live on the instrument thread while the test itself is the GUI-side client.
What is being proved is the boundary, not the engine: the engine's own
behaviour has its own suites, and each of these asks only "does that
behaviour still hold when the two sides are a queued hop apart?"

Four properties, one per crossing (GLOSSARY.md's **Instrument thread**):

* a command is POSTED — it returns at once with the ``request_id`` its
  ``Verdict`` will carry, and the verdict arrives later on the client's own
  event loop;
* an engine signal is DELIVERED QUEUED — the client's event loop keeps
  running while the engine is inside a slow ``measure()``, which is the whole
  reason the thread exists;
* a mutable payload crosses as a COPY;
* the run queue crosses by a snapshot PUSH one way and a marshalled POP the
  other.

Plus the two things a thread must never do: hang on shutdown, and lose the
actor on the way across.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pytest
from PyQt6.QtCore import QCoreApplication, QThread

from i2as.core import events as ev
from i2as.core.instrument_host import InstrumentHost
from i2as.core.orchestrator import OrchestratorState
from i2as.core.plan import PhasePlan, StepPlan, Target
from i2as.core.station import build_station

CONFIG_PATH = "i2as/configs/sim_cryostat"

#: Every wait in this file is bounded, and generously: a queued hop costs one
#: event-loop turn, so a timeout this long means the boundary is broken, not
#: that the machine was busy.
WAIT_MS = 5000


# ── The runs these tests drive ───────────────────────────────────────────────


class ThreadProbeProcedure:
    """A three-point sweep whose ``measure()`` can be made slow.

    Deliberately minimal — the engine's procedure handling is tested at L4;
    what this exists for is to put a KNOWN, long, synchronous call on the
    instrument thread so the client side can be observed while it runs.

    Args:
        station: The Station, ignored; taken so the run builds like any other.
        measure_seconds: How long each ``measure()`` blocks the engine's tick.
    """

    name = "Thread Probe"

    def __init__(self, station: Any, measure_seconds: float = 0.0) -> None:
        self._sweep = [1.0, 2.0, 3.0]
        self._index = 0
        self._measure_seconds = float(measure_seconds)
        self.measured = 0
        self.measure_threads: list[QThread] = []

    def initiate(self) -> PhasePlan:
        """Ramp to the first point."""
        return PhasePlan(
            targets={"magnet_z": Target(self._sweep[0])}, commands=(), wait_s=0.0
        )

    def change_sweep_step(self) -> StepPlan | None:
        """Advance to the next point, or end the sweep."""
        self._index += 1
        if self._index >= len(self._sweep):
            return None
        return StepPlan(
            targets={"magnet_z": Target(self._sweep[self._index])}, wait_s=0.0
        )

    def measure(self) -> None:
        """Read one point, slowly if asked, recording which thread ran it."""
        self.measure_threads.append(QThread.currentThread())
        if self._measure_seconds:
            time.sleep(self._measure_seconds)
        self.measured += 1

    def standby(self) -> PhasePlan:
        """Ramp back to zero."""
        return PhasePlan(targets={"magnet_z": Target(0.0)}, commands=(), wait_s=0.0)

    def get_progress(self) -> float:
        """Fraction of the sweep completed."""
        return self._index / len(self._sweep)


def _fast_magnet(station: Any) -> None:
    """Make the sim magnet ramp fast enough to drive a sweep in a few ticks."""
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def build_threaded_host(qtbot):
    """Return a factory for started threaded hosts, all shut down after the test.

    A factory rather than a host so a test can choose the tick interval, the
    join bound, or a station factory that has been tampered with — and still
    be sure the instrument thread is joined before pytest tears the test's
    objects down underneath it.
    """
    hosts: list[InstrumentHost] = []

    def _build(
        station_factory=None,
        *,
        tick_interval_ms: int = 50,
        **kwargs: Any,
    ) -> InstrumentHost:
        host = InstrumentHost(
            station_factory or (lambda: build_station(CONFIG_PATH)),
            mode="threaded",
            orchestrator_options={"tick_interval_ms": tick_interval_ms},
            **kwargs,
        )
        hosts.append(host)
        host.start()
        return host

    yield _build
    for host in reversed(hosts):
        host.shutdown()
        # Queued deliveries already posted to THIS thread outlive the engine
        # that emitted them, and land on objects pytest is about to collect.
        # Cutting the engine's connections and then draining is what keeps a
        # torn-down test from segfaulting the next one.
        try:
            host.orchestrator.disconnect()
        except (RuntimeError, TypeError):
            pass
        for _ in range(3):
            QCoreApplication.processEvents()


@pytest.fixture
def host(build_threaded_host):
    """A started threaded host over the sim station."""
    return build_threaded_host()


@pytest.fixture
def proxy(host):
    """The client adapter, carrying the host's bridge."""
    return host.build_proxy()


def _ask_engine(host: InstrumentHost, call, timeout_s: float = 5.0) -> Any:
    """Run *call* on the instrument thread and return what it produced.

    The test-side counterpart of the engine's own marshalling: several tests
    have to observe or tamper with engine-side state, and doing that from the
    client's thread would be the very race the thread exists to remove.

    Args:
        host: The threaded host whose engine thread should run the call.
        call: A zero-argument callable.
        timeout_s: How long to wait for the answer.

    Returns:
        The callable's return value.

    Raises:
        AssertionError: If the engine does not answer within *timeout_s*.
    """
    answer: list[Any] = []
    done = threading.Event()

    def _run() -> None:
        try:
            answer.append(call())
        finally:
            done.set()

    host.bridge.post(_run)
    assert done.wait(timeout_s), "the instrument thread never ran the call"
    return answer[0] if answer else None


def _tick_on_engine(host: InstrumentHost, ticks: int = 1) -> None:
    """Fire *ticks* engine ticks on the instrument thread and wait for them.

    The **tick helper**: a test that used to call ``orchestrator._tick()``
    directly is calling into the engine from the wrong thread the moment the
    stack moves, so it goes through here instead — the tick runs where the
    engine lives, and the caller still gets the synchronous "that tick has
    happened" it was relying on.

    Args:
        host: The threaded host.
        ticks: How many ticks to run.
    """
    engine = host.orchestrator
    for _ in range(ticks):
        _ask_engine(host, engine._tick)


def _wait_for_client_state(qtbot, proxy, state: OrchestratorState) -> None:
    """Wait until the CLIENT's mirror reports *state*.

    Deliberately the mirror and not the engine: the engine reaching a state
    and the client learning of it are two different moments once a queued hop
    separates them, and every one of these tests is about the second.

    Args:
        qtbot: pytest-qt's fixture.
        proxy: The client adapter.
        state: The state to wait for.
    """
    qtbot.waitUntil(lambda: proxy.state == state.value, timeout=20000)


# ── Affinity: exactly one thread touches the instruments ─────────────────────


def test_the_whole_stack_is_built_on_the_instrument_thread(build_threaded_host):
    """The station factory RUNS in the thread, which is what opens VISA there.

    Building the Station on the client's thread and moving it afterwards
    would leave every pyvisa session owned by a thread that never touches it
    again — the one thing a VISA layer is entitled to refuse.
    """
    built_on: list[QThread] = []

    def _factory():
        built_on.append(QThread.currentThread())
        return build_station(CONFIG_PATH)

    host = build_threaded_host(_factory)
    assert built_on == [host.thread_object]
    assert host.thread_object is not QThread.currentThread()


def test_the_engine_and_its_tick_timer_belong_to_the_instrument_thread(host):
    """Affinity, asserted rather than assumed: the timer fires over there."""
    assert host.orchestrator.thread() is host.thread_object
    assert host.orchestrator._timer.thread() is host.thread_object
    assert host.bridge.engine_thread is host.thread_object
    assert host.bridge.client_thread is QThread.currentThread()
    assert host.bridge.on_engine_thread() is False
    assert _ask_engine(host, host.bridge.on_engine_thread) is True


def test_a_station_companion_is_built_and_stopped_on_the_engines_thread(
    build_threaded_host,
):
    """A companion holds the Station, so it must live where the Station does."""
    built_on: list[QThread] = []

    class _Companion:
        """A station companion recording where it was built and stopped."""

        def __init__(self, station: Any) -> None:
            built_on.append(QThread.currentThread())
            self.stopped_on: QThread | None = None

        def stop(self) -> None:
            """Record the thread that stopped this companion."""
            self.stopped_on = QThread.currentThread()

    host = build_threaded_host(station_companions=(_Companion,))
    (companion,) = host.companions
    assert built_on == [host.thread_object]
    host.shutdown()
    assert companion.stopped_on is host.thread_object


# ── Commands: posted, answered by exactly one verdict ────────────────────────


#: One call per ``CommandName`` the contract can carry as JSON, in an order
#: that leaves the engine usable for the next one — the four object-carrying
#: commands are excluded because they forward without a verdict, which is a
#: property of the contract and not of the thread.
_COMMANDS: tuple[tuple[str, tuple[Any, ...], dict[str, Any]], ...] = (
    ("start_monitoring", (), {}),
    ("set_attendance", (False,), {}),
    ("set_agent_gate", (ev.AgentGate.READ_ONLY,), {}),
    ("submit_vi_action", ("magnet_z", "set_field"), {"target_T": 0.05}),
    ("submit_global_action", ("initiate",), {}),
    ("stop_ramp", ("magnet_z",), {}),
    ("ping_instrument", ("magnet_z",), {}),
    ("disconnect_instrument", ("temperature",), {}),
    ("connect_instrument", ("temperature",), {}),
    ("acknowledge_fault", ("temperature",), {}),
    ("retry_fault", ("temperature",), {}),
    ("set_experiment_envelope", (None,), {}),
    ("run_queue", (), {}),
    ("pause_procedure", (), {}),
    ("resume_procedure", (), {}),
    ("abort_procedure", (), {}),
    ("recover_from_error", (), {}),
    ("stop_monitoring", (), {}),
    ("emergency_standby", ("thread suite",), {}),
    ("acknowledge", (), {}),
)


def test_every_command_crosses_the_thread_and_comes_back_as_one_verdict(
    proxy, qtbot
):
    """The posted crossing, for every command the contract can carry.

    Each call returns at once with the ``request_id`` its verdict will carry
    — that id is generated on THIS thread, which is the only reason a posted
    submission can answer synchronously — and each verdict arrives on this
    thread's event loop, within an explicit bound. Whether the engine accepts
    or refuses is not this test's business; that it answers, exactly once, is.
    """
    covered = {name for name, _, _ in _COMMANDS}
    forwarded = {
        ev.CommandName.RUN_PROCEDURE.value,
        ev.CommandName.QUEUE_PROCEDURE.value,
    }
    assert covered | forwarded == {name.value for name in ev.CommandName}

    for name, args, kwargs in _COMMANDS:
        with qtbot.waitSignal(proxy.verdict, timeout=WAIT_MS) as answered:
            request_id = getattr(proxy, name)(*args, **kwargs)
        verdict = answered.args[0]
        assert verdict.request_id == request_id, name
        assert verdict.command.value == name


def test_a_command_returns_before_the_engine_has_run_it(host, proxy, qtbot):
    """Posted, not called: the client is not waiting on the engine's tick.

    The engine is held inside a slow call, and the client still gets its
    ``request_id`` back immediately — the proof that a submission costs the
    client an event-loop post and nothing else.
    """
    released = threading.Event()
    entered = threading.Event()

    def _slow_tick() -> None:
        entered.set()
        released.wait(10.0)

    host.bridge.post(_slow_tick)
    assert entered.wait(5.0), "the engine never picked up the blocking call"

    started = time.monotonic()
    request_id = proxy.start_monitoring()
    elapsed = time.monotonic() - started
    assert request_id
    assert elapsed < 0.5, "submit() waited on the engine"
    assert host.orchestrator.is_monitoring() is False  # not run yet

    released.set()
    with qtbot.waitSignal(proxy.verdict, timeout=WAIT_MS) as answered:
        pass
    assert answered.args[0].request_id == request_id


# ── The reason the thread exists ─────────────────────────────────────────────


def test_the_client_event_loop_keeps_running_through_a_slow_measure(
    host, proxy, qtbot
):
    """The frozen-GUI detector, and the exit criterion of the whole change.

    A 50 ms timer on the CLIENT thread counts its own firings while a
    ``measure()`` blocks the instrument thread for two seconds. Single-
    threaded, that timer would fire zero times during the measurement — the
    frozen window operators actually complained about. Threaded, it must keep
    firing throughout, and a click posted mid-measurement must be accepted at
    once.
    """
    from PyQt6.QtCore import QTimer

    _ask_engine(host, lambda: _fast_magnet(host.station))
    procedure = ThreadProbeProcedure(host.station, measure_seconds=2.0)
    proxy.start_monitoring()
    proxy.run_procedure(procedure)

    ticks: list[float] = []
    heartbeat = QTimer()
    heartbeat.setInterval(50)
    heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
    heartbeat.start()
    try:
        # Wait for the engine to be INSIDE the slow measure, then watch the
        # client thread for a second of it.
        qtbot.waitUntil(lambda: bool(procedure.measure_threads), timeout=20000)
        ticks.clear()
        started = time.monotonic()
        qtbot.wait(1000)
        elapsed = time.monotonic() - started

        # A responsive client thread fires ~20 times a second. Ten is a floor
        # a loaded CI machine still clears and a frozen one cannot.
        assert len(ticks) >= 10, (
            f"the client thread fired {len(ticks)} times in {elapsed:.2f} s — "
            "it is being blocked by the engine"
        )
        gaps = [b - a for a, b in zip(ticks, ticks[1:])]
        assert max(gaps) < 0.5, f"the client thread stalled for {max(gaps):.2f} s"

        # And it is still ACCEPTING work, not merely repainting.
        assert proxy.pause_procedure()
    finally:
        heartbeat.stop()

    assert procedure.measure_threads[0] is host.thread_object


def test_the_measurement_runs_on_the_instrument_thread(host, proxy, qtbot):
    """Where the hardware call happens, asserted from inside the call itself."""
    _ask_engine(host, lambda: _fast_magnet(host.station))
    procedure = ThreadProbeProcedure(host.station)
    proxy.start_monitoring()
    proxy.run_procedure(procedure)
    qtbot.waitUntil(lambda: procedure.measured >= 1, timeout=20000)
    assert set(procedure.measure_threads) == {host.thread_object}


# ── The pause boundary, end to end across the thread ─────────────────────────


def test_the_pause_boundary_holds_across_the_thread(host, proxy, qtbot):
    """A pause clicked mid-datapoint is acknowledged at once and lands after it.

    The **pause boundary** is the engine's rule; what the thread adds is that
    the click is acknowledged on the client's thread while the engine is
    still inside the point, and the PAUSED state arrives afterwards as an
    event rather than as a return value.
    """
    _ask_engine(host, lambda: _fast_magnet(host.station))
    procedure = ThreadProbeProcedure(host.station, measure_seconds=1.0)
    proxy.start_monitoring()
    proxy.run_procedure(procedure)
    qtbot.waitUntil(lambda: bool(procedure.measure_threads), timeout=20000)

    # Clicked while the engine is inside measure(): acknowledged immediately,
    # by a client that is not waiting for the point to finish.
    started = time.monotonic()
    request_id = proxy.pause_procedure()
    assert time.monotonic() - started < 0.5
    assert request_id

    with qtbot.waitSignal(proxy.state_changed, timeout=WAIT_MS) as changed:
        pass
    _wait_for_client_state(qtbot, proxy, OrchestratorState.PAUSED)
    assert changed.args[0]

    # The point that was in flight was read and saved; the pause landed after
    # it, never in the middle of it.
    assert procedure.measured >= 1
    assert proxy.state == "PAUSED"


# ── Quench, then the queued procedure that must not start ────────────────────


def test_a_quench_stops_the_run_and_no_queued_procedure_starts(host, proxy, qtbot):
    """The emergency path, with the run queue on the other side of the boundary.

    A quench is a critical-severity flag: the engine enters EMERGENCY on the
    tick that observes it. The waiting queue then must NOT be pulled — and
    with the queue living on the client's thread, "must not be pulled" now
    also means the engine never marshals a question across to it.
    """
    _ask_engine(host, lambda: _fast_magnet(host.station))
    pulls: list[str] = []

    def _take_next_spec() -> Any:
        pulls.append("pull")
        return None

    proxy.install_run_queue(
        next_run=lambda: None,
        queue_entries=lambda: [{"name": "Waiting", "actor": {"kind": "operator", "id": "op"}}],
        take_next_spec=_take_next_spec,
        build_spec=lambda spec: None,
    )

    procedure = ThreadProbeProcedure(host.station)
    proxy.start_monitoring()
    proxy.run_procedure(procedure)
    qtbot.waitUntil(lambda: bool(procedure.measure_threads), timeout=20000)

    _ask_engine(host, lambda: setattr(
        host.station.get_vi("magnet_z")._driver, "_simulate_quench", True
    ))
    _wait_for_client_state(qtbot, proxy, OrchestratorState.EMERGENCY)

    qtbot.wait(300)  # give any errant pull time to happen
    assert pulls == [], "the queue was pulled after a quench"
    assert proxy.state == "EMERGENCY"


# ── The actor survives the crossing ──────────────────────────────────────────


def test_an_agent_command_arrives_named_as_an_agent(host, proxy, qtbot):
    """Accountability is a value, and a queued hop must not launder it.

    An agent-actor command is submitted on the engine's thread (which is
    where the agent gateway's engine reference lives), and the ``StateChange``
    it causes is read on the CLIENT's side — where the status bar would read
    it — still naming the agent.
    """
    agent = ev.Actor(kind=ev.ActorKind.AGENT, id="thread-suite-agent", role="operator")
    changes: list[ev.StateChange] = []
    proxy.state_change_event.connect(changes.append)

    command = ev.Command(
        name=ev.CommandName.EMERGENCY_STANDBY,
        actor=agent,
        args={"reason": "agent asked"},
    )
    with qtbot.waitSignal(proxy.verdict, timeout=WAIT_MS) as answered:
        host.bridge.post(lambda: host.orchestrator.submit(command))

    assert answered.args[0].actor.kind == ev.ActorKind.AGENT
    qtbot.waitUntil(lambda: bool(changes), timeout=WAIT_MS)
    emergency = [c for c in changes if c.state == "EMERGENCY"]
    assert emergency, [c.state for c in changes]
    assert emergency[-1].actor.kind == ev.ActorKind.AGENT
    assert emergency[-1].actor.id == "thread-suite-agent"
    assert emergency[-1].request_id == command.request_id


# ── Payload ownership ────────────────────────────────────────────────────────


def test_a_mutable_payload_crosses_as_a_copy(host, proxy, qtbot):
    """The signal payload rule: what the client receives, the engine has let go.

    Qt hands a queued receiver the very object that was emitted, so the copy
    cannot happen at delivery — it has to happen at the EMIT, and this is what
    checks that it did. Two properties, and the second is the one that bites:
    the payload shares no container with ``Station``'s stale-value cache, and
    it does not change under the client while the engine keeps polling.

    The **event stream**'s own payloads need no copy and get none: every event
    is a frozen dataclass, so sharing one is sharing a value.
    """
    from copy import deepcopy

    received: list[dict] = []
    proxy.states_updated.connect(received.append)
    proxy.start_monitoring()
    qtbot.waitUntil(lambda: bool(received), timeout=WAIT_MS)

    payload = received[0]
    cache = host.station._last_known_state
    assert payload is not cache
    for vi_name, values in payload.items():
        assert values is not cache.get(vi_name), vi_name

    # The engine goes on polling; the client's copy must not move under it.
    frozen = deepcopy(payload)
    qtbot.waitUntil(lambda: len(received) >= 4, timeout=WAIT_MS)
    assert payload == frozen, "the engine mutated a payload it had handed away"

    records: list[dict] = []
    proxy.operational_status.connect(records.append)
    qtbot.waitUntil(lambda: bool(records), timeout=WAIT_MS)
    assert records[0] is not host.orchestrator._operational_status

    events: list[Any] = []
    proxy.event.connect(events.append)
    qtbot.waitUntil(lambda: bool(events), timeout=WAIT_MS)
    assert all(
        type(event).__dataclass_params__.frozen for event in events
    ), "an event that is not frozen cannot be shared across the boundary"


# ── The run queue's two crossings ────────────────────────────────────────────


def test_the_queue_snapshot_is_pushed_and_the_pop_is_marshalled(host, proxy, qtbot):
    """The queue-crossing rule, both halves, with the threads named.

    ``queue_entries`` is read on the CLIENT's thread — it walks a list the
    client is free to mutate — and the engine reads the pushed copy. Popping
    the next spec is marshalled back to the client's thread for the same
    reason, while BUILDING the run happens on the engine's, because it
    touches the Station.
    """
    entries_read_on: list[QThread] = []
    popped_on: list[QThread] = []
    built_on: list[QThread] = []
    procedure = ThreadProbeProcedure(host.station)

    def _queue_entries() -> list[dict[str, Any]]:
        entries_read_on.append(QThread.currentThread())
        return [{"name": "Thread Probe", "position": 1}]

    def _take_next_spec() -> Any:
        popped_on.append(QThread.currentThread())
        return "spec" if len(popped_on) == 1 else None

    def _build_spec(spec: Any) -> Any:
        built_on.append(QThread.currentThread())
        return procedure

    proxy.install_run_queue(
        next_run=lambda: None,
        queue_entries=_queue_entries,
        take_next_spec=_take_next_spec,
        build_spec=_build_spec,
    )

    client = QThread.currentThread()
    changes: list[ev.QueueChanged] = []
    proxy.queue_changed_event.connect(changes.append)
    with qtbot.waitSignal(proxy.queue_changed_event, timeout=WAIT_MS):
        proxy.publish_queue()
    assert set(entries_read_on) == {client}
    assert list(changes[-1].entries) == [{"name": "Thread Probe", "position": 1}]

    _ask_engine(host, lambda: _fast_magnet(host.station))
    proxy.start_monitoring()
    proxy.run_queue()
    qtbot.waitUntil(lambda: bool(built_on), timeout=WAIT_MS)
    assert set(popped_on) == {client}, "the pop must run where the queue lives"
    assert set(built_on) == {host.thread_object}, "the build touches the Station"


def test_a_threaded_client_must_install_both_halves_of_the_pull_seam(proxy):
    """Half a seam is a race, so it is refused where it is written."""
    with pytest.raises(ValueError, match="take_next_spec and build_spec"):
        proxy.install_run_queue(next_run=lambda: None, queue_entries=lambda: [])


# ── Shutdown: bounded, never a hang ──────────────────────────────────────────


def test_shutdown_is_bounded_when_an_instrument_read_never_returns(
    build_threaded_host, qtbot, caplog
):
    """A wedged VISA read must not make the application unexitable.

    A sim driver's read is made to block forever, the engine is left inside
    it, and ``shutdown()`` is called. It must return within its own join
    bound, log ``CRITICAL``, and NAME the instrument it was reading —
    ``Station.polling_vi()`` is what makes an anonymous hang a diagnosable
    one.
    """
    wedged = threading.Event()

    def _factory():
        station = build_station(CONFIG_PATH)
        original = station.get_vi("temperature").get_state

        def _never_returns():
            wedged.set()
            time.sleep(3.0)
            return original()

        station.get_vi("temperature").get_state = _never_returns
        return station

    host = build_threaded_host(_factory, tick_interval_ms=20, join_timeout_ms=300)
    proxy = host.build_proxy()
    proxy.start_monitoring()
    assert wedged.wait(10.0), "the engine never reached the wedged read"

    with caplog.at_level(logging.CRITICAL, logger="i2as.core.instrument_host"):
        started = time.monotonic()
        host.shutdown()
        elapsed = time.monotonic() - started

    assert elapsed < 3.0, f"shutdown() took {elapsed:.1f} s — it is not bounded"
    assert not host.thread_object.isFinished()
    critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert critical, "a wedged shutdown must say so at CRITICAL"
    message = critical[-1].getMessage()
    assert "temperature" in message, message
    assert "300 ms" in message or "300" in message

    # The wedged thread outlives shutdown() by design; wait it out so pytest
    # does not tear this test's objects down underneath it.
    assert host.thread_object.wait(60000)


def test_shutdown_is_idempotent_and_safe_before_start():
    """A caller unsure whether the host is up may simply shut it down."""
    host = InstrumentHost(lambda: build_station(CONFIG_PATH), mode="threaded")
    host.shutdown()
    host.shutdown()
    assert host.thread_object is None


def test_shutdown_stops_the_timer_on_the_thread_that_owns_it(host, qtbot):
    """The stop and the quit travel together, so the timer is never orphaned.

    Qt refuses to stop a timer from another thread (it prints
    ``Timers cannot be stopped from another thread`` and leaks it), so this
    is the difference between a clean exit and a warning-strewn one.
    """
    timer = host.orchestrator._timer
    assert timer.isActive()
    host.shutdown()
    assert host.thread_object.isFinished()
    assert not timer.isActive()


def test_the_event_loop_survives_a_raising_marshalled_call(host, proxy, qtbot):
    """The thread-level exception boundary, from the outside.

    A call that raises on the instrument thread must not take the event loop
    with it: a dead loop is a ``shutdown()`` that can never be delivered.
    """
    def _explode() -> None:
        raise RuntimeError("boom on the instrument thread")

    host.bridge.post(_explode)
    with qtbot.waitSignal(proxy.verdict, timeout=WAIT_MS) as answered:
        request_id = proxy.start_monitoring()
    assert answered.args[0].request_id == request_id
    host.shutdown()
    assert host.thread_object.isFinished()


# ── The client never reads across the boundary ───────────────────────────────


def test_the_mirror_is_primed_from_the_engines_own_thread(host, proxy):
    """``client_state()`` is captured where the engine was built, once."""
    station_info, snapshot, operational = host.client_state()
    assert isinstance(station_info, ev.StationInfo)
    assert isinstance(snapshot, ev.StatusSnapshot)
    assert isinstance(operational, dict)
    assert proxy.station_info() is proxy.status.station_info()
    assert proxy.state == "IDLE"
    assert proxy.is_monitoring() is False


def test_the_gui_builds_and_drives_through_the_proxy_across_the_thread(
    host, proxy, qtbot
):
    """Both windows, on the client's thread, against an engine that is not.

    The transparency claim, tested where it can actually fail: the widgets
    are handed exactly what they are handed inline, and a click still reaches
    the engine — only now it arrives one event-loop hop later.
    """
    from i2as.gui.monitor_window import MonitorWindow
    from i2as.gui.procedure_window import ProcedureWindow

    monitor = MonitorWindow(host.station, proxy, mirror=proxy.status)
    qtbot.addWidget(monitor)
    monitor.show()
    assert monitor._state_label.text().endswith("IDLE")

    procedure = ProcedureWindow(
        host.station,
        proxy,
        get_sample_info=lambda: {"sample_name": "s", "sample_id": "1", "comments": ""},
        get_data_dir=lambda: "C:/CryoData",
        mirror=proxy.status,
    )
    qtbot.addWidget(procedure)
    procedure.show()

    monitor._monitoring_btn.click()
    qtbot.waitUntil(lambda: host.orchestrator.is_monitoring() is True, timeout=WAIT_MS)
    qtbot.waitUntil(
        lambda: monitor._monitoring_btn.text() == "Stop Monitoring", timeout=WAIT_MS
    )


def test_the_instrument_thread_carries_the_application_name(host):
    """The QThread is named so a debugger or a thread dump can find it."""
    assert host.thread_object is not None
    assert host.thread_object.objectName() == "i2as-instrument"
