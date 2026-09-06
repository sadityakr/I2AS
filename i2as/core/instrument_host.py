"""InstrumentHost — who builds the instrument stack, and where it lives.

One object owns the Station and the Orchestrator and decides which thread they
are built on. Two modes:

* ``threaded`` — the default, and the one the single hardware thread standard
  describes: the instrument stack lives on its own ``QThread``, the
  **instrument thread**, so a slow ``measure()`` can no longer freeze the
  window.
* ``inline`` — the same design on one thread: everything is constructed on the
  caller's thread, and the client is still handed a ``StatusMirror``-primed
  ``OrchestratorProxy`` rather than the engine itself.

``inline`` and ``threaded`` differ in where ``start()`` runs, and in nothing
the client can see: the same proxy, the same primed mirror, the same signals.

**``inline`` is temporary.** It stays for one release after ``threaded``
became the default, as the way back for a setup whose VISA layer misbehaves
under a second thread (``I2AS_INSTRUMENT_THREAD=0``, or
``instrument_thread: false`` in that setup's ``monitor.yaml``). It is removed
one release after the flip if no hardware regression has been filed against
the threaded default; until then every behaviour here holds in both modes and
the GUI suite runs both ways.

**The single hardware thread standard.** In ``threaded`` mode exactly one
thread ever touches a driver, a VI, the Station, the Orchestrator or the
DataManager. That is enforced by construction rather than by a lock:

* the *station factory* runs inside the thread, so every pyvisa
  ``ResourceManager`` and serial port is opened by the thread that will use
  it, and the Orchestrator (and its tick ``QTimer``) is constructed there too,
  which is what gives the timer that thread's affinity;
* a **station companion** — an object that holds the Station and runs on its
  own timer, ``TrendCheckRunner`` being the one today — is built inside the
  thread by the same call and stopped there;
* every client call crosses through the :class:`ThreadBridge`: a client
  ``submit()`` is *posted* (queued, non-blocking, so it returns at once with
  the ``request_id`` the client itself generated), and the engine's few
  callbacks into client-owned data are *asked* (queued, with a bounded wait
  for the result);
* every engine signal reaches the client through Qt's auto connection, which
  is a queued connection precisely because emitter and receiver live on
  different threads — the payload is delivered on the client's event loop,
  one hop later, never inside the engine's emit.

**Shutdown is bounded, never a hang.** ``shutdown()`` posts the stop onto the
engine's own event loop (the tick timer is stopped by the thread that owns
it), then ``quit()``s and joins with a timeout. A wedged VISA read cannot make
the application unexitable: the join gives up, a ``CRITICAL`` record names
what the engine was reading, and the process exits leaving the instrument
exactly where a process kill would leave it.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from i2as.core import events as ev
from i2as.core.orchestrator import Orchestrator

if TYPE_CHECKING:
    from i2as.core.orchestrator_proxy import OrchestratorProxy
    from i2as.core.station import Station

logger = logging.getLogger(__name__)

#: The modes ``InstrumentHost`` knows.
MODES: tuple[str, ...] = ("inline", "threaded")

#: The environment variable that overrides the config file's choice, for CI
#: (which runs the suite in both modes, ``0`` being the explicit inline leg)
#: and for a one-off launch.
THREAD_ENV_VAR: str = "I2AS_INSTRUMENT_THREAD"

#: The setting that selects the mode, named in log lines and refusals so an
#: operator is told exactly which knob they turned.
THREAD_FLAG: str = f"monitor.yaml `instrument_thread` ({THREAD_ENV_VAR})"

#: How long ``start()`` waits for the thread to finish building the stack.
#: Generous: opening a rack of VISA sessions is slow, and a build that is
#: merely slow must not be reported as a build that failed.
DEFAULT_BUILD_TIMEOUT_S: float = 120.0

#: How long ``shutdown()`` waits for the instrument thread to stop before it
#: gives up and says so. Long enough for a settled instrument read to return,
#: short enough that a wedged one does not hold the window open.
DEFAULT_JOIN_TIMEOUT_MS: int = 5000

#: How long the engine waits for a client-thread callback to answer. Bounded
#: because the alternative — Qt's ``BlockingQueuedConnection`` — has no
#: timeout at all, and an engine that cannot get an answer must carry on
#: rather than stop ticking.
DEFAULT_CALLBACK_TIMEOUT_S: float = 10.0

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def resolve_mode(configured: bool | None = None) -> str:
    """Decide which mode to build in, config first and the environment last.

    The **instrument-thread flag**: ``monitor.yaml``'s ``instrument_thread``
    is the setup's own choice (read with
    ``i2as.core.config.read_instrument_thread()``), and
    ``I2AS_INSTRUMENT_THREAD`` overrides it for one launch — which is how
    CI runs the same suite in both modes without editing a config.

    ``threaded`` is the default: a setup that says nothing gets the instrument
    thread, and only an explicit ``false`` (or ``I2AS_INSTRUMENT_THREAD=0``)
    asks for the temporary ``inline`` mode back.

    Args:
        configured: What the config file says, or ``None`` when it says
            nothing. ``None`` and ``True`` both mean ``threaded``; only
            ``False`` means ``inline``.

    Returns:
        ``"threaded"`` or ``"inline"``.
    """
    override = os.environ.get(THREAD_ENV_VAR)
    if override is not None:
        text = override.strip().lower()
        if text in _TRUE_VALUES:
            return "threaded"
        if text in _FALSE_VALUES:
            return "inline"
        logger.warning(
            "Ignoring %s=%r: expected one of %s",
            THREAD_ENV_VAR,
            override,
            sorted(_TRUE_VALUES | _FALSE_VALUES),
        )
    return "inline" if configured is False else "threaded"


class _Ask:
    """One question posted to another thread, with a place for the answer.

    Callable so it can travel the same courier as a fire-and-forget call: the
    receiving thread simply calls it, and the asking thread waits on ``done``.

    Args:
        call: The zero-argument callable to run on the receiving thread.
    """

    __slots__ = ("call", "done", "error", "result")

    def __init__(self, call: Callable[[], Any]) -> None:
        self.call = call
        self.result: Any = None
        self.error: BaseException | None = None
        self.done = threading.Event()

    def __call__(self) -> None:
        """Run the question on the receiving thread and release the asker."""
        try:
            self.result = self.call()
        except BaseException as exc:  # noqa: BLE001 — carried back, not swallowed
            self.error = exc
        finally:
            self.done.set()


class _Courier(QObject):
    """Runs a callable on the thread this object was created on.

    The whole thread crossing, in one object: ``dispatch`` is connected to
    ``_run`` on the courier itself, so Qt's auto connection makes an emit from
    another thread a QUEUED delivery onto this courier's event loop, and an
    emit from its own thread a direct call. One courier is created on the
    instrument thread (for client → engine calls) and one on the client thread
    (for the engine's few callbacks into client-owned data).

    Args:
        parent: Optional Qt parent.
    """

    dispatch = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.dispatch.connect(self._run)

    @pyqtSlot(object)
    def _run(self, call: Callable[[], Any]) -> None:
        """Run one marshalled call, guarded.

        Args:
            call: The zero-argument callable that crossed the boundary. A
                raising call is logged and dropped: a marshalled call has no
                caller left to raise at, and killing the receiving thread's
                event loop would be far worse than losing one call.
        """
        try:
            call()
        except Exception:  # noqa: BLE001 — nothing to raise at on this side
            logger.exception("a marshalled call raised on the receiving thread")


class ThreadBridge:
    """How calls cross between a client thread and the instrument thread.

    Held by the :class:`~i2as.core.orchestrator_proxy.OrchestratorProxy`,
    which is the only thing that needs to know a boundary exists. Two
    directions, deliberately asymmetric:

    * ``post()`` — client → engine, queued and non-blocking. Every command
      goes this way, which is why a client never waits on an engine that is
      forty seconds deep inside ``measure()``.
    * ``ask()`` — engine → client, queued with a bounded wait for the answer.
      Used only where the engine genuinely needs a value back out of data the
      client owns, and never for anything a snapshot could carry.

    Args:
        engine: The courier living on the instrument thread.
        client: The courier living on the client thread.
        callback_timeout_s: How long ``ask()`` waits for an answer.
    """

    def __init__(
        self,
        engine: _Courier,
        client: _Courier,
        *,
        callback_timeout_s: float = DEFAULT_CALLBACK_TIMEOUT_S,
    ) -> None:
        self._engine = engine
        self._client = client
        self._callback_timeout_s = callback_timeout_s

    @property
    def engine_thread(self) -> QThread:
        """The thread that owns the Station, the Orchestrator and the drivers."""
        return self._engine.thread()

    @property
    def client_thread(self) -> QThread:
        """The thread that built the host and holds the proxy."""
        return self._client.thread()

    def on_engine_thread(self) -> bool:
        """Return whether the caller is already on the instrument thread."""
        return QThread.currentThread() is self.engine_thread

    def post(self, call: Callable[[], Any]) -> None:
        """Run *call* on the instrument thread, without waiting for it.

        Args:
            call: A zero-argument callable. Called directly when the caller is
                already on the instrument thread, which is what makes a
                command submitted from inside the engine behave exactly as it
                did before the thread existed.
        """
        self._engine.dispatch.emit(call)

    def ask(self, call: Callable[[], Any], *, default: Any = None) -> Any:
        """Run *call* on the client thread and wait, bounded, for its answer.

        Args:
            call: A zero-argument callable evaluated on the client thread.
            default: What to return if the client does not answer in time.

        Returns:
            The callable's return value, or *default* on timeout.

        Raises:
            BaseException: Whatever *call* raised, re-raised on the asking
                thread so the engine's own guards (which already treat a
                failing client callback as "the queue did not advance") see
                it.
        """
        question = _Ask(call)
        self._client.dispatch.emit(question)
        if not question.done.wait(self._callback_timeout_s):
            logger.critical(
                "The client thread did not answer an engine callback within "
                "%.1f s; carrying on without it. The client's event loop is "
                "blocked, or the application is shutting down.",
                self._callback_timeout_s,
            )
            return default
        if question.error is not None:
            raise question.error
        return question.result


class _EngineThread(QThread):
    """The instrument thread: builds the stack, then runs its event loop.

    ``run()`` is what makes the thread affinity right — everything the host
    builds is constructed here, so every QObject among them (the Orchestrator
    and its tick timer first of all) belongs to this thread, and every VISA
    session is opened by the thread that will use it.

    Args:
        host: The host whose stack this thread builds and owns.
    """

    def __init__(self, host: InstrumentHost) -> None:
        super().__init__()
        self._host = host

    def run(self) -> None:
        """Build the stack, then serve its event loop until ``quit()``."""
        try:
            self._host._build()
        except BaseException as exc:  # noqa: BLE001 — reported back to start()
            logger.critical(
                "The instrument stack could not be built on the instrument "
                "thread; the application has no engine",
                exc_info=True,
            )
            self._host._build_error = exc
            self._host._built.set()
            return
        self._host._built.set()
        logger.info("Instrument thread running")
        # The thread-level exception boundary. The tick has its own boundary
        # and degrades to ERROR; this one is the outer guarantee that the
        # event loop itself survives, because a dead loop is a shutdown() that
        # can never be delivered and an application that cannot exit.
        while True:
            try:
                self.exec()
                return
            except BaseException:  # noqa: BLE001 — boundary must be broad
                logger.critical(
                    "The instrument thread's event loop raised; restarting it "
                    "so shutdown can still reach the engine",
                    exc_info=True,
                )


class InstrumentHost(QObject):
    """Owns the instrument stack and hands a client its adapter.

    Args:
        station_factory: Builds the Station. A callable rather than a built
            Station because in ``threaded`` mode it must run *inside* the
            thread that will own every instrument handle.
        mode: ``"inline"`` or ``"threaded"``. The application never relies
            on this argument's default — ``resolve_mode()`` decides for it,
            and answers ``threaded`` unless a setup or the environment asks
            otherwise; the default here keeps a test that builds a host
            directly on the thread it already runs on.
        orchestrator_options: Keyword arguments for the ``Orchestrator``
            constructor beyond the Station — tick interval, safety timings,
            the run catalog. May be a callable taking the built Station,
            for the common case where those values come from the config the
            Station build itself resolved.
        station_companions: Factories for objects that hold the Station and
            must therefore live wherever it does — the **station companion**
            rule. Each is called with the built Station on the engine's own
            thread; whatever it returns is kept alive by the host and, if it
            offers ``stop()``, stopped on that same thread at shutdown.
            ``TrendCheckRunner`` is the one companion today.
        join_timeout_ms: How long ``shutdown()`` waits for the instrument
            thread to stop before giving up with a ``CRITICAL`` record.
        build_timeout_s: How long ``start()`` waits for the thread to build
            the stack.
        callback_timeout_s: How long the engine waits for a client-thread
            callback (see :class:`ThreadBridge`).
        parent: Optional Qt parent.

    Raises:
        ValueError: If *mode* is not one of ``MODES``.
    """

    def __init__(
        self,
        station_factory: Callable[[], Station],
        *,
        mode: str = "inline",
        orchestrator_options: (
            Mapping[str, Any] | Callable[[Station], Mapping[str, Any]] | None
        ) = None,
        station_companions: Sequence[Callable[[Station], Any]] = (),
        join_timeout_ms: int = DEFAULT_JOIN_TIMEOUT_MS,
        build_timeout_s: float = DEFAULT_BUILD_TIMEOUT_S,
        callback_timeout_s: float = DEFAULT_CALLBACK_TIMEOUT_S,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if mode not in MODES:
            raise ValueError(
                f"unknown InstrumentHost mode {mode!r}; expected one of "
                f"{list(MODES)}"
            )
        self._mode = mode
        self._station_factory = station_factory
        self._orchestrator_options = orchestrator_options or {}
        self._station_companions = tuple(station_companions)
        self._join_timeout_ms = join_timeout_ms
        self._build_timeout_s = build_timeout_s
        self._callback_timeout_s = callback_timeout_s

        self._station: Station | None = None
        self._orchestrator: Orchestrator | None = None
        self._companions: tuple[Any, ...] = ()
        self._client_state: (
            tuple[ev.StationInfo, ev.StatusSnapshot, dict[str, Any]] | None
        ) = None

        self._thread: _EngineThread | None = None
        self._bridge: ThreadBridge | None = None
        self._engine_courier: _Courier | None = None
        self._client_courier: _Courier | None = None
        self._built = threading.Event()
        self._build_error: BaseException | None = None
        self._stopped = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build the Station and the Orchestrator, and capture the client state.

        Idempotent: a second call is a no-op, so a caller that is unsure
        whether the host is up may simply start it.

        In ``threaded`` mode this returns once the instrument thread has
        finished building — the client needs the primed ``client_state()``
        before it can show a window, and a build is a one-off cost at launch,
        not something the running application ever waits on.

        Raises:
            RuntimeError: If the instrument thread does not finish building
                within ``build_timeout_s``.
            BaseException: Whatever the station factory or the Orchestrator
                constructor raised, re-raised on the caller's thread so a bad
                config fails at launch exactly as it does inline.
        """
        if self._orchestrator is not None:
            return
        if self._mode == "threaded":
            self._start_threaded()
        else:
            self._build()
        logger.info("Instrument host started in %s mode", self._mode)

    def _start_threaded(self) -> None:
        """Start the instrument thread and wait for it to build the stack.

        Raises:
            RuntimeError: If the build does not finish within the timeout.
            BaseException: Whatever the build raised, re-raised here.
        """
        self._client_courier = _Courier()  # created here → client affinity
        self._thread = _EngineThread(self)
        self._thread.setObjectName("i2as-instrument")
        self._thread.start()
        if not self._built.wait(self._build_timeout_s):
            raise RuntimeError(
                "The instrument thread did not finish building the station "
                f"within {self._build_timeout_s:.0f} s. Set {THREAD_FLAG} to "
                "false to build on the caller's thread instead."
            )
        if self._build_error is not None:
            raise self._build_error

    def _build(self) -> None:
        """Build the whole stack on whichever thread calls this.

        The one construction path both modes take: in ``inline`` mode the
        caller's thread calls it, in ``threaded`` mode ``_EngineThread.run()``
        does. Everything built here — the Station's driver sessions, the
        Orchestrator, its tick timer, the station companions and the engine's
        courier — therefore belongs to that one thread.
        """
        self._station = self._station_factory()
        options = self._orchestrator_options
        if callable(options):
            options = options(self._station)
        self._orchestrator = Orchestrator(self._station, **dict(options))
        self._companions = tuple(
            build(self._station) for build in self._station_companions
        )
        if self._mode == "threaded":
            assert self._client_courier is not None  # set before the thread ran
            self._engine_courier = _Courier()  # engine-thread affinity
            self._bridge = ThreadBridge(
                self._engine_courier,
                self._client_courier,
                callback_timeout_s=self._callback_timeout_s,
            )
        # The mirror's priming values, taken HERE — on whichever thread the
        # engine was built on — so no client ever reads across the boundary.
        self._client_state = (
            self._orchestrator.station_info(),
            self._orchestrator.status_snapshot(),
            self._orchestrator.get_operational_status(),
        )

    def shutdown(self) -> None:
        """Stop the engine's tick timer and, in threaded mode, join the thread.

        Idempotent, and safe before ``start()``. The stop always happens on
        the thread that owns the timer: inline that is the caller's thread,
        threaded it is posted onto the instrument thread's own event loop,
        together with the ``quit()`` that ends it.

        The join is bounded. If it expires the instrument thread is wedged —
        almost always inside a VISA read that will not return — and this logs
        ``CRITICAL`` naming what it was reading, then returns so the
        application can still exit. The instrument is left exactly where a
        process kill would leave it.
        """
        if self._stopped:
            return
        self._stopped = True
        if self._orchestrator is None:
            return
        if self._mode != "threaded" or self._bridge is None:
            self._stop_engine()
            return
        thread = self._thread
        if thread is None:
            self._stop_engine()
            return
        # Stop and quit travel together, as ONE posted call, so the timer is
        # always stopped by the thread that owns it BEFORE that thread's event
        # loop ends. Quitting from this side instead would race: `quit()` can
        # take effect on the next loop iteration, leaving the posted stop
        # undelivered and the timer to be torn down from the wrong thread.
        self._bridge.post(lambda: (self._stop_engine(), thread.quit()))
        if not thread.wait(self._join_timeout_ms):
            logger.critical(
                "The instrument thread did not stop within %d ms: %s. "
                "Exiting anyway — the instrument is left exactly where it is, "
                "as a process kill would leave it.",
                self._join_timeout_ms,
                self._wedged_description(),
            )

    def _stop_engine(self) -> None:
        """Stop the tick timer and every station companion. Engine-thread only."""
        for companion in self._companions:
            stop = getattr(companion, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:  # noqa: BLE001 — teardown must never raise
                    logger.exception("a station companion refused to stop")
        if self._orchestrator is not None:
            self._orchestrator.shutdown()

    def _wedged_description(self) -> str:
        """Name what the instrument thread is stuck on, for the CRITICAL log.

        Read from the client thread while the engine is wedged, so it names
        only values the engine publishes rather than calling into it: the read
        the Station is in the middle of, and the engine's last known state.

        Returns:
            A human-readable phrase for the log line.
        """
        parts: list[str] = []
        station = self._station
        if station is not None:
            in_flight = station.polling_vi()
            parts.append(
                f"the read of '{in_flight}' has not returned"
                if in_flight
                else "no instrument read was in flight"
            )
        engine = self._orchestrator
        if engine is not None:
            parts.append(f"engine state {engine.state}")
        return "; ".join(parts) or "no diagnostic available"

    # ------------------------------------------------------------------
    # What the host hands out
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """The mode this host was constructed for."""
        return self._mode

    @property
    def bridge(self) -> ThreadBridge | None:
        """How calls cross the boundary, or ``None`` when there is none.

        ``None`` in ``inline`` mode, which is what keeps the proxy's
        marshalling a single ``if`` rather than a second code path.
        """
        return self._bridge

    @property
    def thread_object(self) -> QThread | None:
        """The instrument thread, or ``None`` in ``inline`` mode.

        Named ``thread_object`` because ``QObject.thread()`` already means
        "the thread this object lives on", which for the host is the client's.
        """
        return self._thread

    @property
    def companions(self) -> tuple[Any, ...]:
        """The station companions this host built, in declaration order."""
        return self._companions

    @property
    def station(self) -> Station:
        """The Station this host built.

        It lives on the engine's thread. A client holds it only for the pure
        declaration reads the GUI has not finished moving onto ``StationInfo``
        (``tests/test_conformance.py`` ratchets that list down), never to
        touch an instrument.

        Returns:
            The Station.

        Raises:
            RuntimeError: If the host has not been started.
        """
        if self._station is None:
            raise RuntimeError("InstrumentHost.start() has not been called")
        return self._station

    @property
    def orchestrator(self) -> Orchestrator:
        """The Orchestrator this host built.

        Returns:
            The engine.

        Raises:
            RuntimeError: If the host has not been started.
        """
        if self._orchestrator is None:
            raise RuntimeError("InstrumentHost.start() has not been called")
        return self._orchestrator

    def client_state(
        self,
    ) -> tuple[ev.StationInfo, ev.StatusSnapshot, dict[str, Any]]:
        """Return what a client's status mirror is primed with.

        The station's first declaration, the status at start, and the latest
        operational-status record — all captured on the engine's own thread
        at ``start()``. Everything after them arrives on the event stream.

        Returns:
            ``(station_info, status_snapshot, operational_status)``.

        Raises:
            RuntimeError: If the host has not been started.
        """
        if self._client_state is None:
            raise RuntimeError("InstrumentHost.start() has not been called")
        return self._client_state

    def build_proxy(self, parent: QObject | None = None) -> OrchestratorProxy:
        """Build the client adapter for this host's engine.

        Args:
            parent: Optional Qt parent for the proxy.

        Returns:
            An :class:`~i2as.core.orchestrator_proxy.OrchestratorProxy`
            with a mirror already primed from ``client_state()`` and, in
            ``threaded`` mode, this host's :class:`ThreadBridge`.

        Raises:
            RuntimeError: If the host has not been started.
        """
        # Local import: the proxy imports this module for typing, so a
        # module-level import here would be circular.
        from i2as.core.orchestrator_proxy import OrchestratorProxy

        return OrchestratorProxy.for_host(self, parent=parent)
