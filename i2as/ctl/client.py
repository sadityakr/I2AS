"""The two ways this client reaches an engine, behind one object.

``--offline`` builds a whole instrument stack in this process and hands the
**Agent gateway** the real engine; **live** (the default) talks to an
application that is already running, through the **Request spool**. Both end
in the same place — a ``Gateway`` with a rendered **Tool surface** — so every
subcommand is written once and neither mode is a special case above this
module.

What differs, and why
---------------------

*Offline* is a complete stack: the Station is built from a config directory,
the Orchestrator over it, and the session layer over that, so ``validate_run``
can build a run headlessly and a probe run really runs. Its tick has no event
loop to fire it — a one-shot command is not an application — so the client
drives the engine's own tick by hand, which is also what makes a command's
verdict available before the process exits.

*Live* is a **mirror plus a letterbox**. Reads are answered from the files the
running engine mirrors into the spool (``station.json`` for the declaration,
``events.jsonl`` for the status) and from the session store on disk, exactly
as an in-process client answers them from its own mirror. Writes become one
request file, and the answer is the verdict the running tick appends to
``verdicts.jsonl``. Nothing here reaches into the running process.

Two things a client may never do, in either mode
------------------------------------------------

* **Open or close an experiment.** That is `envelope`-class work, which the
  permission matrix grants to no role: the physicist opens the experiment,
  and the client works inside the one that is open. A client with no open
  experiment still reads the station and submits commands; the session tools
  that need a record refuse by name.
* **Widen its own authority.** The declared role travels on the request and
  is judged at the engine — in live mode capped a second time by the setup's
  ``spool_max_role`` — never by this process.
"""

from __future__ import annotations

import getpass
import logging
import socket
from pathlib import Path
from typing import Any

from i2as.core import events as ev
from i2as.core.instrument_host import InstrumentHost
from i2as.core.paths import measurement_root
from i2as.core.request_spool import RequestSpool, spool_directory
from i2as.core.config import read_safety_config, read_tick_interval_ms
from i2as.core.station import Station, build_station
from i2as.ctl.discovery import discover_run_catalog
from i2as.session.agent_feed import AgentFeed
from i2as.session.gateway import Gateway, Role, ToolContext
from i2as.session.gateway.tools import ToolError
from i2as.session.manager import ExperimentManager
from i2as.session.models import GUEST_USER_ID, GUEST_USER_NAME, User
from i2as.session.store import ExperimentStore, SessionStore, UserRoster

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_SETTLE_TICKS",
    "MODE_LIVE",
    "MODE_OFFLINE",
    "CtlClient",
    "CtlUnreachable",
    "SpoolEngine",
    "default_actor_id",
    "open_client",
]

#: How long a live request waits for its verdict before the client gives up.
DEFAULT_TIMEOUT_S = 10.0

#: How many ticks an offline client runs to settle a command the engine
#: queued for the tick (``submit_vi_action`` is the one that does).
DEFAULT_SETTLE_TICKS = 2

MODE_OFFLINE = "offline"
MODE_LIVE = "live"

#: The Qt application object an offline client builds its engine under, held
#: for the process lifetime (see ``_ensure_qt_application``).
_QT_APPLICATION: Any = None


class CtlUnreachable(Exception):
    """No engine could be reached, so no command was even attempted.

    The one failure this client distinguishes from a refusal: a missing
    spool, a verdict that never arrived, or a config that would not build.
    The CLI answers it with exit code 2, so "the instrument said no" and "I
    never got to ask" are never confused for one another.
    """


def default_actor_id() -> str:
    """Return the actor id an invocation stamps on its commands by default.

    Every verdict, feed record and event this client causes names it, so it
    has to say WHICH shell on WHICH machine — an unattributed action in the
    trail is the failure the **Agent feed** exists to prevent.

    Returns:
        ``ctl:<user>@<host>``, degrading to ``ctl:unknown@unknown`` when the
        platform will not say.
    """
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 — an unnamed user is not a failure
        user = "unknown"
    try:
        host = socket.gethostname()
    except Exception:  # noqa: BLE001 — an unnamed host is not a failure
        host = "unknown"
    return f"ctl:{user or 'unknown'}@{host or 'unknown'}"


# ══════════════════════════════════════════════════════════════════════════
# The live engine: the Request spool, seen as an EngineClient
# ══════════════════════════════════════════════════════════════════════════


class _Signal:
    """The little of a Qt signal the gateway actually binds to.

    The ``Gateway`` connects to ``verdict_emitted`` and ``event_emitted`` and
    the engine emits on them; that is the whole protocol. A live client has
    no Qt event loop and wants none, so the two streams are plain callback
    lists rather than a reason to drag ``QObject`` into a shell command.
    """

    def __init__(self) -> None:
        self._slots: list[Any] = []

    def connect(self, slot: Any) -> None:
        """Subscribe *slot* to this stream.

        Args:
            slot: A one-argument callable.
        """
        self._slots.append(slot)

    def emit(self, payload: Any) -> None:
        """Deliver *payload* to every subscriber, in subscription order.

        A raising subscriber is logged and skipped: one client's bad slot
        must not swallow another's answer.

        Args:
            payload: The verdict or event to deliver.
        """
        for slot in list(self._slots):
            try:
                slot(payload)
            except Exception:  # noqa: BLE001 — one slot never breaks the rest
                logger.exception("ctl: a signal subscriber failed")


class SpoolEngine:
    """A running application's engine, reached through the **Request spool**.

    Satisfies the gateway's ``EngineClient`` shape without a socket, a thread
    or a Qt event loop: ``station_info()`` reads the declaration the engine
    mirrored, and ``submit()`` writes one request file and blocks — this is a
    client, not a tick — until the running engine's own tick answers it in
    ``verdicts.jsonl``, then republishes that verdict on the stream the
    gateway is listening to. Waiting HERE rather than in the CLI is what lets
    the gateway report a live command exactly as it reports an offline one.

    Attributes:
        timed_out: Whether the last submit gave up waiting for its verdict.
        last_record: The raw verdict record the last submit read back, or
            ``None`` — kept because the spool answers a request it could not
            parse with a record that is deliberately not a ``Verdict``.
    """

    def __init__(
        self,
        spool: RequestSpool,
        role: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """Bind to one spool directory.

        Args:
            spool: The spool the running application drains.
            role: The authority declared on every request file written here.
                Capped at the engine by the setup's ``spool_max_role``.
            timeout_s: How long to wait for a verdict before giving up.
        """
        self.verdict_emitted = _Signal()
        self.event_emitted = _Signal()
        self.timed_out = False
        self.last_record: dict[str, Any] | None = None
        self._spool = spool
        self._role = str(role)
        self._timeout_s = float(timeout_s)

    def station_info(self) -> ev.StationInfo:
        """Return the declaration snapshot the running engine mirrored.

        Returns:
            The mirrored ``StationInfo``, or an empty one when the
            application has not published a declaration yet — an empty
            station renders the command tools and no capability tools, which
            is the truthful answer rather than a failure.
        """
        return self._spool.latest_station() or ev.StationInfo()

    def prime(self) -> None:
        """Publish the mirrored declaration and status onto the streams.

        Called once the gateway has subscribed, because a gateway that has
        heard nothing assumes the restrictive reading of **Attendance** and
        cannot render a capability tool. The values come from the spool's
        mirror, so no engine is disturbed to answer a read.
        """
        station = self._spool.latest_station()
        if station is not None:
            self.event_emitted.emit(station)
        status = self._spool.latest_status()
        if status is not None:
            self.event_emitted.emit(status)

    def submit(self, command: ev.Command) -> str:
        """Write one request and wait for the running tick's answer.

        Args:
            command: The command to spool, actor already stamped.

        Returns:
            ``command.request_id``, whether or not a verdict arrived.

        Raises:
            CtlUnreachable: If the spool cannot be written at all.
        """
        self.timed_out = False
        self.last_record = None
        try:
            self._spool.write_request(command, self._role)
        except OSError as exc:
            raise CtlUnreachable(
                f"could not write to the request spool at {self._spool.root}: {exc}"
            ) from exc
        logger.info(
            "Spooled %s as %s", command.name.value, command.request_id
        )
        record = self._spool.wait_for_verdict(command.request_id, self._timeout_s)
        if record is None:
            self.timed_out = True
            logger.error(
                "No verdict for %s within %.1f s", command.request_id, self._timeout_s
            )
            return command.request_id
        self.last_record = record
        try:
            self.verdict_emitted.emit(ev.Verdict.from_json(record))
        except (TypeError, ValueError):
            # The spool answers an unparseable request with a record that is
            # deliberately not a Verdict (no command to name). Keep it as the
            # raw answer rather than inventing one.
            logger.warning("The spool's answer to %s is not a verdict", command.request_id)
        return command.request_id


# ══════════════════════════════════════════════════════════════════════════
# The live experiment façade: the session store, read from another process
# ══════════════════════════════════════════════════════════════════════════


class _StoredExperiments:
    """The read half of the experiment façade, over the store on disk.

    A live client is a second process, so it cannot hold the running
    application's ``ExperimentManager`` — but the store is a folder of files
    on the same machine, and reading it is how the session tools answer
    ``list_runs``, ``read_run_*``, ``read_experiment`` and ``read_agent_feed``
    without asking the engine for anything. Read-only by construction:
    nothing here writes, and the one method that would need the Station
    refuses by name.

    Attributes:
        store: The ``ExperimentStore`` of the active session.
    """

    def __init__(self, store: ExperimentStore) -> None:
        """Bind to one experiment store.

        Args:
            store: The store of the session the application is running under.
        """
        self.store = store

    def current_experiment(self) -> Any:
        """Return the store's active experiment record, or ``None``.

        Returns:
            The ``ExperimentRecord`` the store points at, re-read on every
            call because the running application owns it and may have moved
            on since the last one.
        """
        active = self.store.get_active()
        return self.store.load(active) if active else None

    def validate_run(self, *_args: Any, **_kwargs: Any) -> Any:
        """Refuse to validate a run out of process, by name.

        Validation builds the run headlessly against the Station, and a live
        client has no Station — the running application has it. The refusal
        names the collaborator rather than answering "no findings", which
        would read as approval.

        Raises:
            ToolError: Always.
        """
        raise ToolError(
            "validate_run needs the Station, and a live client has none: run "
            "it with --offline <config_dir> to validate against a simulated "
            "station, or ask the running application through its own window.",
            {"rule": "missing_collaborator", "collaborator": "station"},
        )


# ══════════════════════════════════════════════════════════════════════════
# The client
# ══════════════════════════════════════════════════════════════════════════


class CtlClient:
    """One ctl connection: a gateway, and whatever had to be built for it.

    Attributes:
        mode: ``"offline"`` or ``"live"``.
        role: The authority every command declares.
        actor_id: The identity every command is stamped with.
        engine: The object the gateway submits through — the Orchestrator
            offline, a :class:`SpoolEngine` live.
        experiments: The experiment façade the session tools read through, or
            ``None`` when there is no session to read.
    """

    def __init__(
        self,
        *,
        mode: str,
        engine: Any,
        role: Role,
        actor_id: str,
        experiments: Any | None = None,
        run_catalog: dict[str, type] | None = None,
        host: InstrumentHost | None = None,
        settle_ticks: int = DEFAULT_SETTLE_TICKS,
    ) -> None:
        """Assemble the client around an engine that is already reachable.

        Built by :func:`open_client` rather than directly; the arguments are
        what the two modes differ in.

        Args:
            mode: ``"offline"`` or ``"live"``.
            engine: The gateway's engine.
            role: The declared authority.
            actor_id: The declared identity.
            experiments: The experiment façade, or ``None``.
            run_catalog: ``{class name: class}`` for ``validate_run``.
            host: The instrument host to shut down on ``close()``, offline.
            settle_ticks: How many ticks an offline client runs to let the
                engine carry out a command it queued for its tick.
        """
        self.mode = mode
        self.role = role
        self.actor_id = actor_id
        self.engine = engine
        self.experiments = experiments
        self._run_catalog = dict(run_catalog or {})
        self._host = host
        self._settle_ticks = max(0, int(settle_ticks))
        self._gateway: Gateway | None = None
        self._gateway_experiment: str | None = None
        # Offline only: every verdict the engine emits, by request id, so a
        # command the tick carries out later can still be reported by the
        # invocation that submitted it.
        self._verdicts: dict[str, ev.Verdict] = {}
        if mode == MODE_OFFLINE:
            engine.verdict_emitted.connect(self._remember_verdict)

    # ── The gateway ───────────────────────────────────────────────────

    @property
    def gateway(self) -> Gateway:
        """The **Agent gateway** this client acts through.

        Rebuilt when the open experiment changes, because the **Agent feed**
        a gateway records into is one experiment's file: a client that kept
        the first feed it saw would write one experiment's actions into
        another's trail.

        Returns:
            The gateway, built on first use.
        """
        current = self._current_experiment_id()
        if self._gateway is None or current != self._gateway_experiment:
            self._gateway = self._build_gateway(current)
            self._gateway_experiment = current
        return self._gateway

    def _current_experiment_id(self) -> str | None:
        """Return the open experiment's id, or ``None``.

        Returns:
            The id, or ``None`` when nothing is open or there is no session
            layer to ask.
        """
        if self.experiments is None:
            return None
        try:
            record = self.experiments.current_experiment()
        except Exception:  # noqa: BLE001 — an unreadable store is not an error here
            logger.exception("ctl: could not read the open experiment")
            return None
        identity = getattr(record, "experiment_id", "") if record is not None else ""
        return str(identity) or None

    def _build_gateway(self, experiment_id: str | None) -> Gateway:
        """Build a gateway bound to *experiment_id*'s feed.

        Args:
            experiment_id: The open experiment, or ``None``.

        Returns:
            The gateway.
        """
        feed: AgentFeed | None = None
        if experiment_id and self.experiments is not None:
            path = self.experiments.store.agent_feed_path(experiment_id)
            feed = AgentFeed(path, experiment_id)
            # Offline the engine is here, so the feed can record the verdicts
            # and state changes that answer what it recorded being asked.
            if self.mode == MODE_OFFLINE:
                feed.attach(self.engine)
        gateway = Gateway(
            self.engine,
            self.role,
            self.actor_id,
            tool_context=ToolContext(
                experiments=self.experiments, run_catalog=self._run_catalog
            ),
            feed=feed,
        )
        if isinstance(self.engine, SpoolEngine):
            self.engine.prime()
        return gateway

    # ── Acting ────────────────────────────────────────────────────────

    def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call one tool and return the answer, settled.

        The gateway answers every call; this adds the one thing a *process*
        has to do that an in-process agent does not — offline, run the ticks
        that carry out a command the engine queued for its tick, so the
        invocation reports the verdict rather than "pending".

        Args:
            name: The tool's name.
            args: Its arguments, JSON-safe.

        Returns:
            The gateway's answer dict, with ``code`` and ``verdict`` filled in
            from the settled verdict where one arrived.
        """
        answer = self.gateway.call_tool(name, args or {})
        if self.mode == MODE_OFFLINE:
            answer = self._settle(answer)
        elif answer.get("code") == "PENDING":
            answer = self._settle_live(answer)
        return answer

    def _remember_verdict(self, verdict: Any) -> None:
        """Record one verdict off the engine's stream, by request id.

        Args:
            verdict: The engine's ``Verdict``.
        """
        if isinstance(verdict, ev.Verdict):
            self._verdicts[verdict.request_id] = verdict

    def _settle(self, answer: dict[str, Any]) -> dict[str, Any]:
        """Tick an offline engine until the answered command has its verdict.

        ``submit_vi_action`` is the one command the engine queues for the
        tick rather than dispatching, so its verdict does not exist yet when
        ``submit()`` returns. Everything else is already answered and this
        returns unchanged after a single tick, which is also what advances a
        run the call just started.

        Args:
            answer: The gateway's answer dict.

        Returns:
            The answer, with the verdict merged in when one arrived.
        """
        request_id = str(answer.get("request_id") or "")
        if not request_id:
            return answer
        for _ in range(self._settle_ticks):
            self.pump(1)
            if request_id in self._verdicts:
                break
        if answer.get("code") != "PENDING":
            return answer
        verdict = self._verdicts.get(request_id)
        if verdict is None:
            return answer
        answer.update(
            {
                "ok": verdict.ok,
                "code": verdict.code.value,
                "reason": verdict.reason,
                "detail": dict(verdict.detail or {}),
                "result": verdict.result,
                "verdict": verdict.to_json(),
            }
        )
        return answer

    def _settle_live(self, answer: dict[str, Any]) -> dict[str, Any]:
        """Report what a live submit came back with when no verdict arrived.

        Two things can leave a live command unanswered: the wait timed out
        (the application is not ticking, or the spool is not enabled), or the
        engine answered with a record that is deliberately not a ``Verdict``.
        They are different failures and the answer says which.

        Args:
            answer: The gateway's ``PENDING`` answer dict.

        Returns:
            The answer, resolved.

        Raises:
            CtlUnreachable: If the verdict never arrived at all.
        """
        engine = self.engine
        if isinstance(engine, SpoolEngine) and engine.last_record is not None:
            record = engine.last_record
            answer.update(
                {
                    "ok": False,
                    "code": str(record.get("code") or "FAILED"),
                    "reason": str(record.get("reason") or ""),
                    "detail": dict(record.get("detail") or {}),
                    "verdict": record,
                }
            )
            return answer
        raise CtlUnreachable(
            f"no verdict for {answer.get('request_id')} arrived in the request "
            f"spool: the application may not be running, may not be ticking, "
            f"or may have the spool disabled (monitor.yaml request_spool)."
        )

    def pump(self, ticks: int = 1) -> None:
        """Run the offline engine's own tick, by hand, *ticks* times.

        A one-shot command has no Qt event loop, so the timer the engine
        starts never fires; the tick is driven here instead. Live mode has
        nothing to pump — the application's own tick is doing it — and this
        does nothing.

        Args:
            ticks: How many ticks to run.
        """
        if self.mode != MODE_OFFLINE:
            return
        for _ in range(max(0, int(ticks))):
            self.engine._tick()  # noqa: SLF001 — the engine's tick, driven by hand

    def close(self) -> None:
        """Release whatever this client built. Idempotent."""
        if self._host is not None:
            self._host.shutdown()
            self._host = None


# ══════════════════════════════════════════════════════════════════════════
# Building one
# ══════════════════════════════════════════════════════════════════════════


def open_client(
    *,
    offline: str | None = None,
    role: str = Role.OBSERVER.value,
    actor_id: str = "",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    spool_root: str | Path | None = None,
    settle_ticks: int = DEFAULT_SETTLE_TICKS,
) -> CtlClient:
    """Open a client in whichever mode the arguments ask for.

    Args:
        offline: A config directory to build a simulated station from, or
            ``None`` for live mode.
        role: The declared **Role** (``observer`` / ``debug`` / ``session``).
        actor_id: The declared identity; defaults to
            :func:`default_actor_id`.
        timeout_s: How long a live request waits for its verdict.
        spool_root: The spool directory, live; defaults to the installation's.
        settle_ticks: How many ticks an offline client runs per call.

    Returns:
        The client, ready to act.

    Raises:
        CtlUnreachable: If the offline config will not build a station, or no
            spool is open for a live client to write into.
        ValueError: If *role* is not a known ``Role``.
    """
    declared = Role(role)
    identity = actor_id or default_actor_id()
    if offline is not None:
        return _open_offline(offline, declared, identity, settle_ticks)
    return _open_live(declared, identity, timeout_s, spool_root)


def _open_offline(
    config_path: str, role: Role, actor_id: str, settle_ticks: int
) -> CtlClient:
    """Build a whole instrument stack in this process.

    Args:
        config_path: The config directory to build from.
        role: The declared role.
        actor_id: The declared identity.
        settle_ticks: How many ticks a call runs.

    Returns:
        The client.

    Raises:
        CtlUnreachable: If the station will not build.
    """
    _ensure_qt_application()
    catalog = discover_run_catalog()

    def _build() -> Station:
        return build_station(config_path)

    def _options(_station: Station) -> dict[str, Any]:
        safety = read_safety_config(config_path)
        return {
            "tick_interval_ms": read_tick_interval_ms(config_path),
            "manual_override_timeout_s": safety["manual_override_timeout_s"],
            "stall_seconds": safety["stall_seconds"],
            "hold_enforcement_interval_s": safety["hold_enforcement_interval_s"],
            "hold_enforcement_max_attempts": safety["hold_enforcement_max_attempts"],
            "run_catalog": catalog,
        }

    host = InstrumentHost(_build, mode="inline", orchestrator_options=_options)
    try:
        host.start()
    except Exception as exc:  # noqa: BLE001 — one place turns a bad config into exit 2
        raise CtlUnreachable(
            f"could not build a station from {config_path!r}: {exc}"
        ) from exc

    engine = host.orchestrator
    experiments = _open_session_layer(engine, host.station, config_path, catalog)
    client = CtlClient(
        mode=MODE_OFFLINE,
        engine=engine,
        role=role,
        actor_id=actor_id,
        experiments=experiments,
        run_catalog=catalog,
        host=host,
        settle_ticks=settle_ticks,
    )
    # One tick before anything is asked: the engine publishes its first
    # status here, which is where the gateway's mirror gets **Attendance**
    # and the **Kill switch** from.
    _ = client.gateway  # built here, so it is subscribed before the first tick
    client.pump(1)
    logger.info("ctl offline over %s as %r", config_path, actor_id)
    return client


def _open_live(
    role: Role, actor_id: str, timeout_s: float, spool_root: str | Path | None
) -> CtlClient:
    """Attach to a running application through its **Request spool**.

    Args:
        role: The declared role.
        actor_id: The declared identity.
        timeout_s: How long a request waits for its verdict.
        spool_root: The spool directory, or ``None`` for the installation's.

    Returns:
        The client.

    Raises:
        CtlUnreachable: If no spool is open.
    """
    root = Path(spool_root) if spool_root is not None else spool_directory()
    spool = RequestSpool(root)
    if not spool.is_open():
        raise CtlUnreachable(
            f"no request spool at {root}: the application is not running, or "
            f"its setup has not enabled one (monitor.yaml request_spool: true). "
            f"Use --offline <config_dir> to work against a simulated station."
        )
    engine = SpoolEngine(spool, role.value, timeout_s=timeout_s)
    client = CtlClient(
        mode=MODE_LIVE,
        engine=engine,
        role=role,
        actor_id=actor_id,
        experiments=_stored_experiments(),
        # The catalog is discovered live too, so a proposed run is refused for
        # the reason that is actually true — no Station out of process — and
        # not as an unknown class name.
        run_catalog=discover_run_catalog(),
    )
    logger.info("ctl live over %s as %r", root, actor_id)
    return client


def _ensure_qt_application() -> None:
    """Make sure a Qt application object exists before the engine is built.

    The Orchestrator is a ``QObject`` that starts a ``QTimer``; without an
    application object Qt refuses the timer and says so on stderr. A *core*
    application is enough — a client has no window and must never need a
    display — and the reference is held at module level because an
    application object that is garbage-collected takes the timer with it.
    """
    global _QT_APPLICATION
    from PyQt6.QtCore import QCoreApplication

    existing = QCoreApplication.instance()
    if existing is None:
        _QT_APPLICATION = QCoreApplication([])
    else:
        _QT_APPLICATION = existing


def _open_session_layer(
    engine: Any, station: Station, config_path: str, catalog: dict[str, type]
) -> ExperimentManager | None:
    """Build the session layer over an offline engine, or answer ``None``.

    The same wiring the application does — the roster, the Session tier's
    active session, and the experiment store nested inside it — so an offline
    client reads and records in exactly the place the application would. The
    manager resumes the store's active experiment on construction; it never
    opens one, because opening an experiment is the physicist's.

    Args:
        engine: The Orchestrator.
        station: The Station a validated run is built against.
        config_path: The config the station was built from.
        catalog: The run catalog.

    Returns:
        The manager, or ``None`` when the measurement root cannot be
        resolved (an installation that has never been configured), which
        leaves every command tool working and the session tools refusing by
        name.
    """
    try:
        root = measurement_root()
    except Exception as exc:  # noqa: BLE001 — no data root is not a failure to start
        logger.warning("ctl: no measurement root, so no session layer: %s", exc)
        return None
    roster = UserRoster(root / "users.json")
    if roster.get(GUEST_USER_ID) is None:
        roster.add(User(user_id=GUEST_USER_ID, name=GUEST_USER_NAME))
    sessions = SessionStore(root / "sessions")
    active = sessions.get_active()
    if active is None or sessions.load(*active) is None:
        session = sessions.create_session(name=GUEST_USER_ID, user_id=GUEST_USER_ID)
        active = (GUEST_USER_ID, session.session_id)
        sessions.set_active(*active)
    user_id, session_id = active
    return ExperimentManager(
        store=ExperimentStore(root / "sessions" / user_id / session_id),
        roster=roster,
        orchestrator=engine,
        config_name=Path(config_path).name,
        config_path=config_path,
        session_store=sessions,
        station=station,
        run_catalog=catalog,
    )


def _stored_experiments() -> _StoredExperiments | None:
    """Open the read-only experiment façade a live client answers reads from.

    Returns:
        The façade over the active session's store, or ``None`` when there is
        no measurement root or no active session to read.
    """
    try:
        root = measurement_root()
    except Exception as exc:  # noqa: BLE001 — no data root is not a failure to start
        logger.warning("ctl: no measurement root, so no session reads: %s", exc)
        return None
    active = SessionStore(root / "sessions").get_active()
    if active is None:
        logger.info("ctl: no active session, so the session tools have nothing to read")
        return None
    user_id, session_id = active
    return _StoredExperiments(ExperimentStore(root / "sessions" / user_id / session_id))
