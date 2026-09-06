# ---
# description: |
#   Tests for the test harness itself: the bound on the tick-helper family in
#   ``tests/instrument_modes.py``. A helper that waits on the engine used to
#   wait forever — a fast tick with a window attached refills the client's
#   event loop faster than a drain can empty it, and the run sat at 100 % CPU
#   until it was killed. Every wait is bounded now, and these are the tests
#   that say so. Runs unchanged in both instrument modes:
#
#       pytest tests/test_instrument_modes.py
#       I2AS_INSTRUMENT_THREAD=0 pytest tests/test_instrument_modes.py
# last_updated: 2026-09-04
# ---

"""The settle bound on the tick helpers (both instrument modes)."""

from __future__ import annotations

import time

import pytest
from PyQt6.QtCore import QObject, QTimer

from tests.instrument_modes import (
    DEFAULT_SETTLE_TIMEOUT_S,
    SETTLE_TIMEOUT_ENV_VAR,
    EngineNotSettled,
    build_host,
    drain,
    instrument_mode,
    on_engine,
    settle_timeout_s,
    settled,
    shutdown_host,
)

CONFIG_PATH = "i2as/configs/sim_cryostat"

#: The bound the bounded tests ask for. Short enough that a failing bound is
#: a fast test, long enough that a loaded machine does not trip it early.
SHORT_BOUND_S: float = 0.5

#: How far past the bound a drain may still return. Not zero: the deadline is
#: read between passes of the event loop, so a drain overruns it by whatever
#: the pass in progress had already queued (see ``drain()``'s docstring).
SLACK_S: float = 5.0

#: The same allowance where the queue is refilled by a real engine, which can
#: hand one pass a backlog far larger than the flood fixture's.
ENGINE_SLACK_S: float = 30.0

#: What one delivery costs the client in the "engine outruns the client"
#: fixture — the stand-in for a visible MonitorWindow's repaint.
SLOW_CONSUMER_S: float = 0.02

#: The tick interval that outruns it, as in the defect report (10-50 ms).
FAST_TICK_MS: int = 10

#: The tick interval the GUI suites use, for the quiet host.
NORMAL_TICK_MS: int = 200


class _EventFlood(QObject):
    """A source that keeps this thread's event loop permanently fed.

    A zero-interval timer is re-armed by every pass of the event loop, so the
    queue is never empty, and the sleep in the slot makes the consumer slower
    than the producer: the shape of the defect with no station in it, and
    cheap enough that the bound can be timed sharply against it.
    """

    def __init__(self, work_s: float = 0.005) -> None:
        super().__init__()
        self._work_s = work_s
        self.deliveries = 0
        self._timer = QTimer(self)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._work)

    def start(self) -> None:
        """Start flooding this thread's event loop."""
        self._timer.start()

    def stop(self) -> None:
        """Stop flooding it."""
        self._timer.stop()

    def _work(self) -> None:
        """Deliver one unit of slow work."""
        self.deliveries += 1
        time.sleep(self._work_s)


class _OutrunClient:
    """A host whose engine emits faster than this thread can consume.

    The engine ticks every 10 ms and every snapshot costs the client 20 ms,
    so the client's queue is refilled faster than a drain can empty it:
    threaded that is the instrument thread outrunning the GUI's, inline it is
    the tick timer firing inside every pass of the drain. Either way
    ``drain()`` never reaches an empty queue, which is what used to mean it
    never returned.

    A test MUST call ``stop()`` as soon as it has what it needs: pytest-qt
    runs the loop dry in its own teardown hook, *before* fixture finalizers,
    so a consumer left slow makes every backlogged delivery cost 20 ms there.

    Args:
        host: The started host, whose proxy is slowed down here.
    """

    def __init__(self, host) -> None:
        self.host = host
        self.proxy = host.build_proxy()
        self.slow = True
        self.proxy.states_updated.connect(self._consume)
        self.proxy.start_monitoring()

    def _consume(self, _states: dict) -> None:
        """Spend longer on one snapshot than the engine spends producing it."""
        if self.slow:
            time.sleep(SLOW_CONSUMER_S)

    def stop(self) -> None:
        """Let the client catch up again. Idempotent."""
        self.slow = False


@pytest.fixture
def flood(qtbot):
    """An event loop this thread can never run dry, stopped on teardown."""
    source = _EventFlood()
    source.start()
    yield source
    source.stop()


@pytest.fixture
def outrun_client(qtbot):
    """A host that outruns its client, per :class:`_OutrunClient`."""
    host = build_host(CONFIG_PATH, tick_interval_ms=FAST_TICK_MS)
    busy = _OutrunClient(host)
    qtbot.wait(200)  # let monitoring start and the snapshots begin
    yield busy
    busy.stop()
    shutdown_host(host)


@pytest.fixture
def quiet_host(qtbot):
    """A started host on the suite's usual tick, idle and not monitoring.

    Yields:
        The ``OrchestratorProxy`` the client side holds.
    """
    host = build_host(CONFIG_PATH, tick_interval_ms=NORMAL_TICK_MS)
    yield host.build_proxy()
    shutdown_host(host)


# ── The default bound and its environment override ────────────────────────────


def test_the_default_bound_is_ten_seconds(monkeypatch):
    """With nothing set, every helper waits ``DEFAULT_SETTLE_TIMEOUT_S``."""
    monkeypatch.delenv(SETTLE_TIMEOUT_ENV_VAR, raising=False)
    assert settle_timeout_s() == DEFAULT_SETTLE_TIMEOUT_S == 10.0


def test_the_environment_widens_the_default_and_a_call_overrides_it(monkeypatch):
    """CI widens every helper at once; one call can still say what it needs."""
    monkeypatch.setenv(SETTLE_TIMEOUT_ENV_VAR, "30")
    assert settle_timeout_s() == 30.0
    assert settle_timeout_s(0.5) == 0.5


@pytest.mark.parametrize("raw", ["", "   ", "soon", "0", "-3"])
def test_an_unusable_environment_value_falls_back_to_the_default(monkeypatch, raw):
    """A bound that is not a positive number is no bound; take the default."""
    monkeypatch.setenv(SETTLE_TIMEOUT_ENV_VAR, raw)
    assert settle_timeout_s() == DEFAULT_SETTLE_TIMEOUT_S


# ── drain() ───────────────────────────────────────────────────────────────────


def test_drain_returns_at_once_when_the_queue_runs_dry(qtbot):
    """The ordinary case is unchanged: nothing queued, back immediately."""
    started = time.monotonic()
    drain()
    assert time.monotonic() - started < SLACK_S


def test_drain_raises_when_the_queue_never_runs_dry(flood):
    """The defect, bounded: a spin forever becomes a failure with a diagnosis."""
    started = time.monotonic()
    with pytest.raises(EngineNotSettled) as excinfo:
        drain(timeout_s=SHORT_BOUND_S)
    elapsed = time.monotonic() - started
    assert SHORT_BOUND_S <= elapsed < SHORT_BOUND_S + SLACK_S
    message = str(excinfo.value)
    assert "drained" in message and "events" in message
    assert SETTLE_TIMEOUT_ENV_VAR in message
    assert flood.deliveries > 0


def test_drain_stops_at_max_events_rather_than_failing(flood):
    """A busy client is not a broken one: enough deliveries is a clean return."""
    delivered = drain(timeout_s=SHORT_BOUND_S, max_events=5)
    assert delivered >= 5


def test_drain_is_bounded_when_the_engine_outruns_the_client(outrun_client):
    """The reported defect end to end: it ends, and it says what it saw.

    The diagnosis is the point — the engine's state and its tick interval are
    what a frozen process never told anyone.
    """
    started = time.monotonic()
    with pytest.raises(EngineNotSettled) as excinfo:
        drain(client=outrun_client.proxy, timeout_s=SHORT_BOUND_S)
    elapsed = time.monotonic() - started
    outrun_client.stop()
    assert SHORT_BOUND_S <= elapsed < ENGINE_SLACK_S
    message = str(excinfo.value)
    assert "engine state" in message
    assert f"tick interval {FAST_TICK_MS} ms" in message


# ── settled() and on_engine() ─────────────────────────────────────────────────


def test_settled_returns_at_once_on_a_settled_engine(quiet_host):
    """An engine with nothing in flight answers immediately, as it always did."""
    started = time.monotonic()
    settled(quiet_host)
    settled(quiet_host, timeout_s=SHORT_BOUND_S)
    assert time.monotonic() - started < SLACK_S


def test_on_engine_still_carries_the_result_back(quiet_host):
    """The bound is the only thing that changed: the call still answers."""
    assert on_engine(quiet_host, lambda: 6 * 7, timeout_s=SHORT_BOUND_S) == 42


def test_settled_is_bounded_when_the_engine_outruns_the_client(outrun_client):
    """``settled()`` on a client that cannot catch up fails fast, never hangs.

    Threaded, the round trip's drain is the half that cannot finish, so the
    bound fires there. Inline there is no round trip to wait out —
    ``settled()`` is a documented no-op in that mode — and the only contract
    is that it returns at once; the bound on inline's drain is
    ``test_drain_is_bounded_when_the_engine_outruns_the_client``'s.
    """
    started = time.monotonic()
    if instrument_mode() == "threaded":
        with pytest.raises(EngineNotSettled):
            settled(outrun_client.proxy, timeout_s=SHORT_BOUND_S)
    else:
        settled(outrun_client.proxy, timeout_s=SHORT_BOUND_S)
    elapsed = time.monotonic() - started
    outrun_client.stop()
    assert elapsed < ENGINE_SLACK_S
