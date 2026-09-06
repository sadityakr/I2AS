"""Running a GUI suite against either instrument mode.

The **instrument-thread flag** decides whether the Station and the
Orchestrator live on their own thread — the default — or on the caller's
(``i2as.core.instrument_host``). Everything above the boundary is
supposed not to care — same proxy, same primed mirror, same signals — and the
only way to keep that true is to run the same GUI suite both ways. This module
is what makes that a matter of an environment variable:

    pytest tests/test_gui.py                                # threaded
    I2AS_INSTRUMENT_THREAD=0 pytest tests/test_gui.py   # inline

Not a test file. It gives a suite three things:

* ``instrument_mode()`` — what the environment asks for, once per session;
* ``build_host()`` / ``shutdown_host()`` — a host in that mode, torn down so
  that no tick and no queued delivery can land in a half-destroyed widget
  tree;
* the **tick helper** family — ``on_engine()``, ``set_on_engine()`` and
  ``tick_engine()``. A GUI test that reaches past the client boundary (forcing
  a state, calling ``_tick()``, setting a private the engine only writes from
  inside a tick) is calling into the engine, and in threaded mode that is a
  call onto another thread. These run it where the engine lives, wait for it,
  and then drain the client's own event loop so the queued consequences —
  the mirror update, the widget repaint — have landed by the time the helper
  returns. Inline they are a direct call plus a drain, so a test reads the
  same either way.

**Every wait here is bounded.** A tick helper waits on an engine that may
never answer — a fast tick with a window attached can keep the engine and the
client's own event loop busy for as long as the test is willing to watch — so
``drain()``, ``on_engine()`` and ``settled()`` all carry a deadline: the drain
stops when the queue runs dry, when ``max_events`` deliveries have landed, or
when ``timeout_s`` is up, and the last of those raises
:class:`EngineNotSettled` naming what it drained, the engine's state and the
tick interval. A wedged suite is then a failing test with a diagnosis instead
of a process at 100 % CPU that has to be killed.

The default bound is ``I2AS_TEST_SETTLE_TIMEOUT_S`` seconds (10 s when the
variable is unset), so a slow CI machine can widen every helper at once::

    I2AS_TEST_SETTLE_TIMEOUT_S=30 pytest tests/test_gui.py

and any single call that legitimately needs longer says so itself::

    settled(orchestrator, timeout_s=30)
    on_engine(orchestrator, slow_call, timeout_s=30)

A test that builds its OWN station and drives it synchronously
(``station.get_state()`` in a loop) is single-threaded by construction and
stays that way; the mode governs the shared fixtures, which is where the
windows are built.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from PyQt6.QtCore import QCoreApplication, QEvent, QEventLoop, QObject, QThread

from i2as.core.instrument_host import InstrumentHost, resolve_mode
from i2as.core.station import build_station

#: The environment variable that widens every settle bound in one go, for a
#: machine slow enough that the default is a false failure. A per-call
#: ``timeout_s`` overrides it.
SETTLE_TIMEOUT_ENV_VAR: str = "I2AS_TEST_SETTLE_TIMEOUT_S"

#: How long a marshalled call — and, separately, the drain that follows it —
#: may take before the suite calls it a failure.
DEFAULT_SETTLE_TIMEOUT_S: float = 10.0

#: How many deliveries one ``drain()`` may land before it calls the job done
#: and returns. Not a failure: an engine that keeps its client's event loop
#: permanently fed has still had every consequence of the test's own action
#: delivered long before this, and stopping is what keeps a busy engine from
#: turning a healthy drain into a timeout.
DEFAULT_MAX_DRAIN_EVENTS: int = 20_000

#: How long one pass of the client's event loop may run before ``drain()``
#: looks at its bounds again. Small, because the bounds are only as sharp as
#: the pass that has to finish before they are read.
DRAIN_SLICE_MS: int = 20

#: How long ``shutdown_host()`` gives the instrument thread to stop.
JOIN_TIMEOUT_MS: int = 5000


class EngineNotSettled(TimeoutError):
    """The engine did not go quiet within the settle bound.

    Raised by ``drain()`` (the client's event loop never ran dry) and by
    ``on_engine()`` (the instrument thread never ran the posted call). Either
    way the test that hit it was about to hang: the message names how far the
    helper got, what the engine was doing, and which knob widens the bound.
    """


def settle_timeout_s(timeout_s: float | None = None) -> float:
    """Resolve the settle bound: the call's own value, else the environment.

    Args:
        timeout_s: What the caller asked for, or ``None`` to take the default.

    Returns:
        The bound in seconds. ``None`` resolves ``SETTLE_TIMEOUT_ENV_VAR``,
        and falls back to ``DEFAULT_SETTLE_TIMEOUT_S`` when it is unset,
        unparseable, or not positive.
    """
    if timeout_s is not None:
        return float(timeout_s)
    raw = os.environ.get(SETTLE_TIMEOUT_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_SETTLE_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_SETTLE_TIMEOUT_S
    return value if value > 0 else DEFAULT_SETTLE_TIMEOUT_S


class _EventCounter(QObject):
    """Counts every event delivered on this thread while a drain runs.

    Installed on the application object, which is Qt's documented way of
    seeing every event delivered to every object on that thread — the only
    way ``drain()`` can say "the queue ran dry" rather than "I called
    ``processEvents()`` three times and hoped".
    """

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """Count one delivery and let it through untouched.

        Args:
            a0: The receiving object.
            a1: The event being delivered.

        Returns:
            ``False``, always: this filter observes, it never eats an event.
        """
        self.count += 1
        return False


def _engine_diagnosis(client: Any) -> str:
    """Describe what the engine was doing, for a bound that ran out.

    Args:
        client: The proxy (or engine) the test holds, or ``None``.

    Returns:
        A phrase naming the engine's state and tick interval, each replaced
        by ``unknown`` when this side cannot read it.
    """
    if client is None:
        return "engine state unknown, tick interval unknown"
    state = getattr(client, "state", None)
    if not isinstance(state, str):
        state = getattr(getattr(engine_of(client), "_state", None), "value", None)
    # The tick period is read from the stall config's plain copy of it, never
    # from the engine's QTimer: the timer belongs to the instrument thread.
    interval = getattr(
        getattr(engine_of(client), "_stall_config", None), "tick_interval_ms", None
    )
    tick = "unknown" if interval is None else f"{interval} ms"
    return f"engine state {state or 'unknown'}, tick interval {tick}"


def instrument_mode() -> str:
    """Return the mode this test session runs in.

    Reads ``I2AS_INSTRUMENT_THREAD`` alone — a test session has no
    ``monitor.yaml`` to consult, and CI selects the mode by environment.
    Unset means ``threaded``, exactly as it does for the application.

    Returns:
        ``"inline"`` or ``"threaded"``.
    """
    return resolve_mode(None)


def build_host(
    config_path: str, *, tick_interval_ms: int = 50, **kwargs: Any
) -> InstrumentHost:
    """Build and start a host over the sim station, in this session's mode.

    Args:
        config_path: The config directory to build the Station from.
        tick_interval_ms: The engine's tick interval.
        **kwargs: Passed through to ``InstrumentHost``.

    Returns:
        The started host.
    """
    host = InstrumentHost(
        lambda: build_station(config_path),
        mode=instrument_mode(),
        orchestrator_options={"tick_interval_ms": tick_interval_ms},
        join_timeout_ms=JOIN_TIMEOUT_MS,
        **kwargs,
    )
    host.start()
    return host


def shutdown_host(host: InstrumentHost) -> None:
    """Stop a host and make sure nothing it emitted is still in flight.

    Three steps, and each one has cost a segfault at some point: stop the
    engine (so no further tick fires into a widget tree qtbot is about to
    destroy), cut the engine's connections (so a delivery already posted to
    this thread cannot re-emit through an object that is gone), and drain.

    Args:
        host: The host to stop. Safe on a host that was never started.
    """
    host.shutdown()
    engine: Any = None
    try:
        engine = host.orchestrator
        engine.disconnect()
    except (RuntimeError, TypeError):
        pass  # never started, or already disconnected
    drain(client=engine)


def drain(
    rounds: int = 3,
    *,
    client: Any = None,
    timeout_s: float | None = None,
    max_events: int | None = None,
) -> int:
    """Let every queued delivery already posted to this thread land, bounded.

    Runs the client's event loop in short passes until one of three things is
    true: the queue ran dry (and at least *rounds* passes have happened, since
    a delivery can post another), *max_events* deliveries have landed, or the
    deadline is up. Only the last is a failure — an engine ticking fast with a
    window attached refills this queue faster than it can be emptied, and
    before the bound existed that turned a drain into a permanent spin at
    100 % CPU with no Python-level clue as to why.

    The deadline is read between passes, never inside one: Qt delivers a
    whole queue before it hands control back, so a drain overruns its bound
    by whatever the pass in progress had already taken on. The bound is what
    makes a drain end and say why, not a promise about when.

    Args:
        rounds: The minimum number of passes to run, as before.
        client: The proxy (or engine) the test holds, used only to describe
            the engine in the timeout message.
        timeout_s: How long to keep draining before giving up. ``None`` takes
            the default (``SETTLE_TIMEOUT_ENV_VAR``, else
            ``DEFAULT_SETTLE_TIMEOUT_S``).
        max_events: How many deliveries are enough. ``None`` takes
            ``DEFAULT_MAX_DRAIN_EVENTS``.

    Returns:
        How many deliveries landed during this drain (``0`` when there is no
        application object to drain, and when the count cannot be taken).

    Raises:
        EngineNotSettled: If the queue neither ran dry nor reached
            *max_events* within the deadline.
    """
    app = QCoreApplication.instance()
    if app is None:
        return 0
    limit = settle_timeout_s(timeout_s)
    cap = DEFAULT_MAX_DRAIN_EVENTS if max_events is None else int(max_events)
    counter = _EventCounter()
    # An event filter only sees the thread it was installed from; off the
    # application's own thread there is no count to take, so the pass budget
    # (`rounds`) is the whole bound, exactly as it was before.
    counting = QThread.currentThread() is app.thread()
    if counting:
        app.installEventFilter(counter)
    deadline = time.monotonic() + limit
    passes = 0
    try:
        while True:
            before = counter.count
            QCoreApplication.processEvents(
                QEventLoop.ProcessEventsFlag.AllEvents, DRAIN_SLICE_MS
            )
            passes += 1
            dry = not counting or counter.count == before
            if (passes >= rounds and dry) or counter.count >= cap:
                return counter.count
            if time.monotonic() >= deadline:
                raise EngineNotSettled(
                    f"the client's event loop did not run dry within {limit:g} s: "
                    f"drained {counter.count} events, {_engine_diagnosis(client)}. "
                    f"Pass a longer timeout_s, or widen the default with "
                    f"{SETTLE_TIMEOUT_ENV_VAR}."
                )
    finally:
        if counting:
            app.removeEventFilter(counter)


def engine_of(client: Any) -> Any:
    """Return the Orchestrator behind *client*.

    Args:
        client: An ``OrchestratorProxy`` or an ``Orchestrator``.

    Returns:
        The engine itself.
    """
    return getattr(client, "_engine", client)


def on_engine(
    client: Any,
    call: Callable[[], Any],
    *,
    settle: bool = True,
    timeout_s: float | None = None,
) -> Any:
    """Run *call* where the engine lives, wait for it, then drain this thread.

    The **tick helper**'s general form. Inline this is ``call()`` followed by
    a drain; threaded it posts the call across, waits for it, and then drains
    — so that by the time it returns, the consequences the engine emitted have
    been delivered to this thread's widgets and mirror.

    Both waits are bounded by *timeout_s*, each in its own right: the engine
    gets that long to run the call, and the drain that follows gets that long
    to run this thread's queue dry.

    Args:
        client: The proxy (or engine) the test holds.
        call: A zero-argument callable to run on the engine's thread.
        settle: Whether to drain this thread's event loop afterwards. Pass
            ``False`` for a call that emits nothing — a bare attribute set —
            where draining would only give the engine's tick timer a chance
            to fire and undo what was just forced.
        timeout_s: The bound for each wait. ``None`` takes the default
            (``SETTLE_TIMEOUT_ENV_VAR``, else ``DEFAULT_SETTLE_TIMEOUT_S``);
            pass a number for a call that legitimately takes longer.

    Returns:
        Whatever *call* returned.

    Raises:
        EngineNotSettled: If the engine does not run the call in time, or the
            drain that follows does not finish in time.
        BaseException: Whatever *call* raised, re-raised here.
    """
    limit = settle_timeout_s(timeout_s)
    bridge = getattr(client, "bridge", None)
    if bridge is None or bridge.on_engine_thread():
        result = call()
        if settle:
            drain(client=client, timeout_s=limit)
        return result

    answer: list[Any] = []
    failure: list[BaseException] = []
    done = threading.Event()

    def _run() -> None:
        try:
            answer.append(call())
        except BaseException as exc:  # noqa: BLE001 — carried back to the test
            failure.append(exc)
        finally:
            done.set()

    bridge.post(_run)
    if not done.wait(limit):
        raise EngineNotSettled(
            f"the instrument thread did not run the call within {limit:g} s: "
            f"{_engine_diagnosis(client)}. Pass a longer timeout_s, or widen "
            f"the default with {SETTLE_TIMEOUT_ENV_VAR}."
        )
    if settle:
        drain(client=client, timeout_s=limit)
    if failure:
        raise failure[0]
    return answer[0] if answer else None


def set_on_engine(
    client: Any, name: str, value: Any, *, timeout_s: float | None = None
) -> None:
    """Set one engine attribute from a test, on the engine's own thread.

    The forcing pattern GUI tests use to stand a scenario up without driving
    the whole state machine to it. Written through here rather than as a bare
    assignment because ``client`` is the proxy: assigning to it would set an
    attribute on the proxy and quietly change nothing.

    Args:
        client: The proxy (or engine) the test holds.
        name: The engine attribute to set.
        value: What to set it to.
        timeout_s: The bound on the wait, as in ``on_engine()``.
    """
    engine = engine_of(client)
    on_engine(
        client, lambda: setattr(engine, name, value), settle=False, timeout_s=timeout_s
    )


def tick_engine(client: Any, times: int = 1, *, timeout_s: float | None = None) -> None:
    """Fire *times* engine ticks where the engine lives, and wait for them.

    What replaces a bare ``orchestrator._tick()`` in a test that has moved
    behind the client boundary.

    Args:
        client: The proxy (or engine) the test holds.
        times: How many ticks to run.
        timeout_s: The bound on each tick, as in ``on_engine()``. Pass a
            longer one for a tick that does something slow — a datapoint the
            sim takes seconds over.
    """
    engine = engine_of(client)
    for _ in range(times):
        on_engine(client, engine._tick, timeout_s=timeout_s)


@contextmanager
def ticks_paused(client: Any, *, timeout_s: float | None = None) -> Iterator[None]:
    """Hold the engine's tick timer for the body of the ``with``.

    A GUI test that FORCES engine state — ``_state``, an active run — and then
    clicks is describing a moment, not a trajectory: the next tick would
    advance the state machine straight back out of it. Inline that never
    happened, because nothing ran the event loop between the two lines;
    threaded, the engine ticks on its own clock and would. Stopping the timer
    makes the moment hold in both modes.

    Args:
        client: The proxy (or engine) the test holds.
        timeout_s: The bound on stopping and restarting the timer, as in
            ``on_engine()``.

    Yields:
        Nothing; the timer is stopped for the duration.
    """
    engine = engine_of(client)
    on_engine(client, engine._timer.stop, settle=False, timeout_s=timeout_s)
    try:
        yield
    finally:
        on_engine(client, engine._timer.start, settle=False, timeout_s=timeout_s)


def settled(client: Any, rounds: int = 1, *, timeout_s: float | None = None) -> None:
    """Wait until everything already sent to the engine has come back.

    A GUI action crosses the boundary twice — the command out, the event that
    proves it happened back — and inline both hops are the same call stack, so
    a test may assert immediately. Threaded they are two event-loop turns, and
    a test that asserts immediately is asserting before the answer exists.
    This closes exactly that gap: it posts a no-op behind everything already
    queued for the engine, waits for it (so every earlier command has run),
    then drains this thread (so every event they emitted has been delivered).

    A no-op inline, deliberately: there is nothing in flight, and running the
    event loop there would only let the tick timer fire in the middle of a
    test that never expected one.

    Each round trip is bounded: an engine that never comes back — a fast tick
    with a window attached, a ``measure()`` that blocks — raises
    :class:`EngineNotSettled` rather than spinning here forever.

    Args:
        client: The proxy the test holds.
        rounds: How many round trips to wait out, for an effect that takes
            more than one.
        timeout_s: The bound on each round trip, as in ``on_engine()``.
            ``None`` takes the default (``SETTLE_TIMEOUT_ENV_VAR``, else
            ``DEFAULT_SETTLE_TIMEOUT_S``).

    Raises:
        EngineNotSettled: If a round trip does not complete within the bound.
    """
    if getattr(client, "bridge", None) is None:
        return
    for _ in range(rounds):
        on_engine(client, lambda: None, timeout_s=timeout_s)
