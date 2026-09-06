"""I2AS application entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication

from i2as.core.config_catalog import ConfigCatalog
from i2as.core.instrument_host import InstrumentHost, resolve_mode
from i2as.core.logging_config import setup_logging
from i2as.core.paths import measurement_root
from i2as.core.request_spool import RequestSpool
from i2as.core.config import (
    read_gateway_config,
    read_instrument_thread,
    read_panels_config,
    read_request_spool_config,
    read_safety_config,
    read_trends_config,
)
from i2as.core.station import Station, build_station_with_fallback
from i2as.core.trend_check_runner import TrendCheckRunner
from i2as.core.trend_checks import declared_checks
from i2as.gui import app_settings
from i2as.gui.monitor_window import MonitorWindow
from i2as.gui.procedure_discovery import discover_procedures
from i2as.gui.theme import PLOT_AXIS, PLOT_BG, build_stylesheet
from i2as.session.agent_feed import AgentFeed
from i2as.session.eln.publisher import ElnPublisher
from i2as.session.gateway import authorize_spooled
from i2as.session.eln.settings import load_eln_settings
from i2as.session.gateway import GatewayServer, ToolContext
from i2as.session.manager import ExperimentManager
from i2as.session.models import GUEST_USER_ID, GUEST_USER_NAME, User
from i2as.session.store import ExperimentStore, SessionStore, UserRoster

logger = logging.getLogger(__name__)

#: The name Qt reports for this process, and the one the desktop shows. The
#: QSettings scope (``gui/app_settings.py``) spells the same name on its own,
#: because it must stay stable across launches; a test pins the two together.
APPLICATION_NAME = "I2AS"


def build_parser() -> argparse.ArgumentParser:
    """Build the application's command line.

    Returns:
        The parser. ``--config`` picks the config for this launch only; the
        saved active config (``gui/app_settings.py``) stays the persistent
        choice and is what a launch without the option opens.
    """
    parser = argparse.ArgumentParser(
        prog="i2as", description="Start the I2AS desktop application."
    )
    parser.add_argument(
        "--config",
        metavar="NAME",
        default=None,
        help=(
            "Config directory name to open for this launch — a shipped config "
            "(e.g. sim_imaging) or a user copy of one. Overrides the saved "
            "active config without replacing it; the startup fallback chain "
            "still applies behind it."
        ),
    )
    return parser


def resolve_config_name(name: str) -> tuple[Path, str]:
    """Resolve a config directory name to its directory and source.

    Shipped configs are searched first, then the user's copies, mirroring the
    identity ``app_settings.config_active()`` stores.

    Args:
        name: A config directory name such as ``"sim_imaging"``.

    Returns:
        ``(path, source)`` with ``source`` ``"shipped"`` or ``"user"``.

    Raises:
        LookupError: If no config of that name exists in either place.
    """
    searched = (
        (app_settings.shipped_config_dir(), "shipped"),
        (app_settings.user_config_dir(), "user"),
    )
    for base_dir, source in searched:
        candidate = base_dir / name
        if candidate.is_dir():
            return candidate, source
    raise LookupError(
        f"no config named {name!r} (looked in "
        + ", ".join(str(base_dir) for base_dir, _ in searched)
        + ")"
    )


def _chain_for(name: str, source: str) -> list[str]:
    """Return one config and, for a user copy, its never-edited shipped baseline."""
    base_dir = (
        app_settings.user_config_dir()
        if source == "user"
        else app_settings.shipped_config_dir()
    )
    chain = [str(base_dir / name)]
    if source == "user":
        shipped_baseline = app_settings.shipped_config_dir() / name
        if shipped_baseline.is_dir():
            chain.append(str(shipped_baseline))
    return chain


def _startup_candidates(config: str | None = None) -> list[str]:
    """Return the ordered config candidates for startup, safest last.

    The config named on the command line, if any, is tried first; then the
    saved active config; each followed by its shipped namesake (the
    never-edited baseline) when it is a user copy; the always-loadable
    ``sim_cryostat`` is the final guarantee. Order-preserving de-dup.

    Args:
        config: The ``--config`` value, a config directory name, or ``None``.

    Returns:
        A list of config directory paths, most-preferred first.

    Raises:
        LookupError: If ``config`` names a config that exists nowhere.
    """
    candidates: list[str] = []
    if config is not None:
        requested, source = resolve_config_name(config)
        candidates.extend(_chain_for(requested.name, source))
    active = app_settings.config_active()
    if active is not None:
        candidates.extend(_chain_for(*active))
    candidates.append(str(app_settings.shipped_config_dir() / "sim_cryostat"))

    seen: set[str] = set()
    ordered: list[str] = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def _ensure_guest_user_registered(roster: UserRoster) -> None:
    """Ensure the fixed Guest roster identity exists (idempotent).

    Runs unconditionally, every launch — the same "must never fail to start
    for lack of an explicit choice" principle ``_resolve_active_session``
    already follows one level down: nobody having logged in must not make
    ``ExperimentManager.start_experiment()``'s roster-membership check
    reject ``user_id=GUEST_USER_ID`` the first time it's used. Never
    clobbers an existing Guest entry (e.g. an admin who gave it a different
    display name) — only adds it when entirely absent.

    Args:
        roster: The setup-local user roster.
    """
    if roster.get(GUEST_USER_ID) is None:
        roster.add(User(user_id=GUEST_USER_ID, name=GUEST_USER_NAME))


def _resolve_active_session(store: SessionStore) -> tuple[str, str]:
    """Return the active ``(user_id, session_id)``, auto-creating a bootstrap session if needed.

    The app must never fail to start for lack of an explicit session choice
    (the startup-wiring rule behind the Session tier — see ``GLOSSARY.md``'s
    **Session**): when the store's active pointer is unset, points at a
    session record that fails to load, or is in the pre-per-user-nesting
    shape (first-ever launch, a corrupt pointer, or an older install), a
    bootstrap session is created and activated on the spot, owned by
    whoever is logged in (or ``GUEST_USER_ID`` when nobody is).

    Args:
        store: The ``SessionStore`` rooted at ``measurement_root() / "sessions"``.

    Returns:
        The active session's ``(user_id, session_id)`` — either the pair
        already pointed to, or a freshly created bootstrap session's.
    """
    active = store.get_active()
    if active is not None and store.load(*active) is not None:
        return active
    user_id = app_settings.current_user_id() or GUEST_USER_ID
    session = store.create_session(name=user_id, user_id=user_id)
    store.set_active(user_id, session.session_id)
    return user_id, session.session_id


def _open_experiment_feed(manager: ExperimentManager) -> AgentFeed | None:
    """Return the **Agent feed** of whichever experiment is open right now.

    Resolved on demand rather than once at startup, so a client that
    connects after the physicist opened a new experiment leaves its trail in
    that experiment's folder — the feed belongs to the experiment, not to
    the process.

    Args:
        manager: The session layer's experiment façade.

    Returns:
        The feed for the open experiment, or ``None`` when none is open (in
        which case a connection records nothing rather than inventing a
        folder to record into).
    """
    experiment = manager.current_experiment()
    if experiment is None:
        return None
    return AgentFeed(
        manager.store.agent_feed_path(experiment.experiment_id),
        experiment.experiment_id,
    )




def _build_gateway_server(
    engine: Any,
    gateway_config: Mapping[str, Any],
    *,
    station_info: Callable[[], Any],
    tool_context: ToolContext,
    feed: Callable[[], AgentFeed | None],
    socket_name: str | None = None,
    descriptor: Path | str | None = None,
    token: str | None = None,
) -> GatewayServer | None:
    """Build and start the **Gateway server**, if this setup asks for one.

    Factored out of ``main()`` so the wiring itself is testable: the object
    handed in is the **Orchestrator proxy**, not the engine, because this
    server lives on the GUI thread and under the single hardware thread
    standard the engine does not. The proxy satisfies the gateway's
    ``EngineClient`` — ``submit()`` posts the command across, and the two
    contract streams arrive queued under their client-side names.

    Args:
        engine: The engine client every connection's ``Gateway`` attaches to.
        gateway_config: ``read_gateway_config()``'s answer for this setup —
            ``gateway_server`` gates the whole thing, ``gateway_max_role``
            caps what any connection may claim.
        station_info: The station's declaration snapshot, or a callable
            returning it.
        tool_context: The collaborators the session tools read through.
        feed: Resolves the open experiment's **Agent feed** at connection
            time, so a client that connects later records into whichever
            experiment is open then.
        socket_name: The local-socket name to listen on; defaults to the
            installation's.
        descriptor: Where to write the descriptor file; defaults to the
            installation's.
        token: The secret to require in ``hello``; a fresh random one when
            omitted, which is the production path.

    Returns:
        The listening server, or ``None`` when this setup does not ask for
        one or names a ``gateway_max_role`` that is not a **Role** — an
        unknown ceiling closes the door rather than guessing at it, and the
        window opens regardless.
    """
    if not gateway_config["gateway_server"]:
        return None
    try:
        server = GatewayServer(
            engine,
            max_role=gateway_config["gateway_max_role"],
            station_info=station_info,
            tool_context=tool_context,
            feed=feed,
            socket_name=socket_name,
            descriptor=descriptor,
            token=token,
        )
    except ValueError:
        logger.exception(
            "Gateway server not started: monitor.yaml declares the unknown "
            "gateway_max_role %r",
            gateway_config["gateway_max_role"],
        )
        return None
    server.start()
    return server


def main(
    argv: Sequence[str] | None = None,
    *,
    on_station_built: Callable[[Station], None] | None = None,
) -> None:
    """Start the I2AS application.

    Args:
        argv: The command line (see ``build_parser``), or ``None`` to read
            the process's own. Bare ``main()`` is what the ``i2as`` console
            script calls.
        on_station_built: Optional hook run once, immediately after the
            Station is built and before the Monitor window is shown.
            Monitoring is off at that point (the production default), so a
            hook that sets sim-driver test-control attributes (e.g.
            ``scripts/run_scenario.py``, driving ``tests.scenarios``' apply
            functions) lands them before anything polls the hardware. Never
            used in normal `python -m i2as.main` startup.
    """
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        candidates = _startup_candidates(args.config)
    except LookupError as error:
        parser.error(str(error))

    setup_logging()

    app = QApplication(sys.argv)
    app.setApplicationName(APPLICATION_NAME)
    app.setApplicationVersion("0.1.0")
    app.setStyleSheet(build_stylesheet())
    pg.setConfigOptions(background=PLOT_BG, foreground=PLOT_AXIS, antialias=True)

    # The config catalog: used at startup only — to resolve the identity of
    # whichever candidate actually loaded, so the next launch starts there
    # even from a different clone or worktree.
    catalog = ConfigCatalog(
        app_settings.shipped_config_dir(), app_settings.user_config_dir()
    )

    # The run catalog: {class __name__: class} for every discovered
    # procedure. Discovery lives up here because neither the engine
    # (contract C5) nor the session layer (C11) may import
    # i2as.procedures — whoever owns discovery hands the catalog down, so
    # a client that speaks the control contract can name a run by class and
    # the run queue can resolve a stored spec back to its class.
    run_catalog: dict[str, type] = {cls.__name__: cls for cls in discover_procedures()}

    # The instrument stack is built by the InstrumentHost, not here: which
    # THREAD owns the Station and the Orchestrator is the host's decision, and
    # `inline` mode is exactly today's behaviour with that decision named. The
    # station factory therefore carries the whole build — including the config
    # fallback, whose resolved path the Orchestrator's own safety settings
    # come from, which is why the options are a callable too.
    build: dict[str, Any] = {}

    def _build_station() -> Station:
        """Build the Station from the first usable startup config."""
        station, used_path, warnings = build_station_with_fallback(candidates)
        build["used_path"] = used_path
        build["warnings"] = warnings
        if on_station_built is not None:
            on_station_built(station)
        return station

    def _orchestrator_options(_station: Station) -> dict[str, Any]:
        """Read the engine's settings from the config the build resolved."""
        safety_config = read_safety_config(build["used_path"])
        # The Request spool (GLOSSARY.md's **Request spool**): off unless the
        # setup asks for it, and capped at the role the setup declares. The
        # permission hook is wired HERE because the role model is the session
        # layer's and the engine may not import it (contract C12) — the same
        # reason the run catalog is handed down rather than discovered below.
        spool_config = read_request_spool_config(build["used_path"])
        request_spool = (
            RequestSpool(
                max_role=spool_config["max_role"], authorizer=authorize_spooled
            )
            if spool_config["enabled"]
            else None
        )
        return {
            "tick_interval_ms": 3000,
            "manual_override_timeout_s": safety_config["manual_override_timeout_s"],
            "stall_seconds": safety_config["stall_seconds"],
            "hold_enforcement_interval_s": safety_config[
                "hold_enforcement_interval_s"
            ],
            "hold_enforcement_max_attempts": safety_config[
                "hold_enforcement_max_attempts"
            ],
            "run_catalog": run_catalog,
            "request_spool": request_spool,
        }

    # The instrument-thread flag. Read from the config the app is ABOUT to
    # load rather than the one it ends up with, because which thread owns the
    # stack has to be decided before anything is built; a fallback to another
    # config does not change the mode. `I2AS_INSTRUMENT_THREAD` overrides
    # it for one launch, which is how CI runs the GUI suite in both modes.
    mode = resolve_mode(read_instrument_thread(candidates[0]))

    # Trend-check standard (core/trend_checks.py, GLOSSARY.md's **Trend
    # check**): a small, single-purpose scheduler independent of the
    # Orchestrator — it holds only a Station, never an Orchestrator — that
    # evaluates this setup's declared checks on its own slow timer and
    # publishes failing ones as advisory-severity conditions. It is a STATION
    # COMPANION: it holds the Station, so the host builds it wherever the
    # Station lives and stops it there too.
    def _build_trend_check_runner(station: Station) -> TrendCheckRunner:
        """Build the trend-check scheduler beside the Station it publishes into."""
        trends_config = read_trends_config(build["used_path"])
        return TrendCheckRunner(
            station,
            declared_checks(trends_config),
            refresh_interval_s=trends_config["refresh_interval_s"],
        )

    app.instrument_host = InstrumentHost(
        _build_station,
        mode=mode,
        orchestrator_options=_orchestrator_options,
        station_companions=(_build_trend_check_runner,),
    )
    app.instrument_host.start()
    # Process lifecycle, owned here: the host stops the tick timer on the
    # thread that owns it and, in threaded mode, joins the instrument thread
    # with a bounded wait, so a wedged instrument read can never leave the
    # application unexitable.
    app.aboutToQuit.connect(app.instrument_host.shutdown)
    station = app.instrument_host.station
    used_path = build["used_path"]
    warnings = build["warnings"]

    # Persist the config that actually loaded (by identity, not path) so the
    # next launch starts there even from a different clone/worktree — unless
    # the command line chose it, which is a choice for this launch alone.
    used_entry = catalog.get_by_path(used_path)
    if used_entry is not None and args.config is None:
        app_settings.set_config_active(used_entry.name, used_entry.source)
    if warnings:
        for warning in warnings:
            logger.warning("Startup config fallback: %s", warning)
    for offline_name in station.offline_vi_names():
        logger.warning(
            "Instrument offline at startup: %s (%s)",
            offline_name,
            station.get_offline_info(offline_name).reason,
        )

    # The control contract, rendered for this process's clients: one typed
    # method per command, the engine's signals re-exposed, and every read
    # answered from a status mirror the host primed on the engine's own
    # thread. Nothing above this line hands the engine itself to anyone.
    orchestrator = app.instrument_host.build_proxy()
    mirror = orchestrator.status

    # Session layer (L6 + the Session tier above it). measurement_root() is
    # the fixed, machine-level, admin-set root (never derived from the Data
    # Directory form field, which is itself now *derived from* the open
    # experiment — see i2as.core.paths.measurement_root()). SessionStore
    # owns the Session tier: sessions/<user_id>/<session_id>/ folders one
    # level above experiments, nested per owner so ownership is structural,
    # not just a field inside session.json. _ensure_guest_user_registered()
    # guarantees the fixed Guest roster identity exists before anything looks
    # it up; _resolve_active_session() auto-creates a bootstrap session on
    # first-ever launch (or a corrupt/legacy-shape pointer) so the app never
    # refuses to start for lack of an explicit session choice, owned by
    # whoever is logged in or Guest otherwise. ExperimentStore is then rooted
    # two levels deeper, inside that one active session's own folder —
    # switching sessions (User menu, Resume Session…) only updates
    # SessionStore's active pointer and takes effect on the next launch;
    # ExperimentManager keeps this ExperimentStore for the process lifetime:
    # it is always constructed with one real, already-resolved
    # ExperimentStore and grows no live rebind machinery (GLOSSARY.md's
    # Session). The user roster relocates to
    # measurement_root()/"users.json", alongside "sessions/".
    roster = UserRoster(measurement_root() / "users.json")
    _ensure_guest_user_registered(roster)
    session_store = SessionStore(measurement_root() / "sessions")
    active_user_id, active_session_id = _resolve_active_session(session_store)
    session_manager = ExperimentManager(
        store=ExperimentStore(measurement_root() / "sessions" / active_user_id / active_session_id),
        roster=roster,
        orchestrator=orchestrator,
        config_name=used_entry.name if used_entry is not None else Path(used_path).name,
        config_path=used_path,
        session_store=session_store,
        station=station,
        run_catalog=run_catalog,
    )

    # The pull seam (GLOSSARY.md's **Run queue**): the engine ASKS the session
    # layer's queue for the next run rather than holding one of its own, and
    # reads the waiting entries from it for every QueueChanged event. Wired
    # here because this is the one place that owns both objects — the queue
    # cannot exist before the engine it broadcasts through, and the engine
    # must not import the session layer (contract C12). It is installed
    # through the proxy, like every other call a client makes here.
    orchestrator.install_run_queue(
        next_run=session_manager.next_run,
        queue_entries=session_manager.queue_entries,
        take_next_spec=session_manager.take_next_spec,
        build_spec=session_manager.build_spec,
    )

    # The Gateway server (i2as/session/gateway/local_server.py,
    # GLOSSARY.md's **Gateway server**): a local socket carrying the same
    # Gateway an in-process client holds, one per connection, so an agent in
    # its own process is authorised, fed and seen exactly as an in-process
    # one. Off unless this setup's monitor.yaml turns it on, and never
    # handing out more than the role that file names — opening a station to
    # autonomous clients is a setup decision, so it lives in the config like
    # every limit. It is a QLocalServer on THIS event loop, and it is built
    # over the PROXY: the server runs on the GUI thread while the engine runs
    # on the instrument thread, so a frame is parsed in a slot that posts its
    # command across like every other client — no thread of its own, and no
    # second writer to the bus. Wired here because this is the one place that
    # owns both the proxy and the session layer; attached to `app` so its
    # ownership is explicit, like every other QObject built in main().
    gateway_config = read_gateway_config(used_path)
    app.gateway_server = _build_gateway_server(
        orchestrator,
        gateway_config,
        station_info=station.station_info,
        tool_context=ToolContext(experiments=session_manager, run_catalog=run_catalog),
        feed=lambda: _open_experiment_feed(session_manager),
    )
    if app.gateway_server is not None:
        # Stopping on quit is what keeps the descriptor honest: a gateway.json
        # left behind names a socket that is gone and a token that means
        # nothing, and an adapter reading it reports "cannot connect" instead
        # of "the app is not running". The socket itself is reclaimed either
        # way — start() removes a stale one — so this exists for the
        # descriptor's sake.
        app.aboutToQuit.connect(app.gateway_server.stop)

    # Read once and shared with the publisher below: the drafting model,
    # key, token cap and price table live in the same user-level settings
    # file the notebook's do, and two reads could disagree.
    eln_settings = load_eln_settings()

    # ELN publishing (i2as/session/eln/): entirely opt-in and entirely
    # GUI-side. With no user-level settings file — the default — the
    # publisher is built, finds nothing configured, and does nothing: the
    # drain timer never starts and on_run_finished() returns immediately, so
    # a setup that has no notebook carries no footprint. The timer lives HERE,
    # in the application entry point, rather than in the Orchestrator, for the
    # same reason all network I/O does: it must never share the tick that
    # writes to hardware. Attached to `app` so its ownership is explicit — a QObject with no Python reference is eligible
    # for GC regardless of Qt-side parenting.
    app.eln_publisher = ElnPublisher(session_manager, eln_settings)
    orchestrator.run_finished.connect(app.eln_publisher.on_run_finished)
    # The other direction of the same seam: the manager holds the publisher so
    # that approving a **draft entry** parked on a run record can queue it.
    session_manager.attach_eln_publisher(app.eln_publisher)
    app.eln_publisher.start()

    # Analysis before the notebook: with analysis on, a finished run is
    # analysed by a recipe in its OWN process first, and the entry that
    # analysis produced waits on the run for a human's approval in the eLab
    # tab. The runner lives here beside the publisher for the same reason the
    # drain timer does — a recipe is user code that may take seconds, so it
    # belongs on the client side of the control contract, never on the tick.
    app.analysis_runner = None
    try:
        from i2as.session.analysis_runner import AnalysisRunner
    except ImportError:
        logger.warning("This build has no analysis runner — analysis stays off")
    else:
        app.analysis_runner = AnalysisRunner(
            session_manager,
            app.eln_publisher,
            lambda: app.eln_publisher.settings,
        )
        requested = getattr(app.eln_publisher, "analysis_requested", None)
        if requested is not None:
            requested.connect(app.analysis_runner.start)

    monitor = MonitorWindow(
        station,
        orchestrator,
        active_config_path=used_path,
        startup_warning="; ".join(warnings) if warnings else None,
        session_manager=session_manager,
        eln_publisher=app.eln_publisher,
        analysis_runner=app.analysis_runner,
        session_store=session_store,
        panels_config=read_panels_config(used_path),
        mirror=mirror,
    )
    monitor.show()

    # The Orchestrator's tick timer starts in __init__, but monitoring starts
    # OFF: no instrument is polled (and no communication errors can fire)
    # until the user starts monitoring from the Monitor window's header
    # toggle, normally after "Initiate All" has brought the instruments up.
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
