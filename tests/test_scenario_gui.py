# ---
# description: |
#   End-to-end GUI scenarios for a running measurement: click storm during a
#   long datapoint, disconnecting an instrument mid-run, a fault during a run,
#   an emergency from the GUI, the agent panel showing an agent's own story,
#   the session envelope, "Probe first" from the queue, the ELN approval
#   flow, shutdown while measuring, and window-liveness. Built the real
#   MonitorWindow/ProcedureWindow against the OrchestratorProxy over an
#   InstrumentHost on ``sim_cryostat``, so the whole file runs unchanged in
#   both instrument modes (``tests/instrument_modes.py``):
#
#       pytest tests/test_scenario_gui.py
#       I2AS_INSTRUMENT_THREAD=0 pytest tests/test_scenario_gui.py
#
#   Passing scenarios are regression tests. A genuine defect is recorded as
#   ``@pytest.mark.xfail(strict=True, reason="DEFECT: ...")`` with its
#   reproduction in the test body, never fixed here (rules of engagement —
#   this suite tests, it does not patch product code). A defect that HAS been
#   fixed loses its xfail and stays as the regression test for the fix.
# last_updated: 2026-09-04
# ---

"""GUI scenarios for a running measurement (both instrument modes)."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QApplication, QPushButton

from i2as.core import events as ev
from i2as.core.instrument_host import InstrumentHost
from i2as.core.plan import PhasePlan, StepPlan, Target
from i2as.core.request_spool import RequestSpool
from i2as.core.station import build_station
from i2as.gui.monitor_window import MonitorWindow
from i2as.gui.procedure_window import ProcedureWindow
from i2as.procedures.field_sweep import FieldSweep
from i2as.session.eln.publisher import ElnPublisher
from i2as.session.eln.settings import ElnSettings
from i2as.session.eln.sim_eln import SimElnAdapter
from i2as.session.gateway import Gateway, Role, ToolContext, authorize_spooled
from i2as.session.manager import ExperimentManager
from i2as.session.models import User
from i2as.session.store import ExperimentStore, UserRoster
from tests import scenarios
from tests.instrument_modes import (
    JOIN_TIMEOUT_MS,
    build_host,
    engine_of,
    instrument_mode,
    on_engine,
    settled,
    shutdown_host,
    tick_engine,
)

CONFIG_PATH = "i2as/configs/sim_cryostat"

SAMPLE_INFO = {"sample_name": "S", "sample_id": "S-1", "comments": ""}

#: A FieldSweep param set fast enough for a real run inside the ~20 s bound —
#: real field steps, real code path, but no realistic thermal/settle waits.
FAST_FIELD_SWEEP_PARAMS = {
    "measurement_vi": "dc_measurement",
    "field_start": -0.05,
    "field_end": 0.05,
    "field_steps": 3,
    "temperature": 300.0,  # the sim temperature controller starts at 300 K -> instant settle
    "current_A": 1e-6,
    "readings_per_point": 2,
    "init_wait": 0.0,
    "step_wait": 0.0,
}

SCREENSHOT_DIR = (
    Path(__file__).resolve().parent.parent
    / "tmp"
    / "scenario_gui"
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)


def _screenshot(widget, name: str) -> Path:
    """Save an offscreen pixmap of *widget* for human inspection.

    Args:
        widget: The window/widget to grab.
        name: Filename (without extension).

    Returns:
        The path written to.
    """
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_DIR / f"{name}.png"
    widget.grab().save(str(path))
    return path


def _fast_magnet(station) -> None:
    """Make the sim magnet ramp fast enough to drive a sweep in a few ticks."""
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Redirect the app QSettings factory to a throwaway INI file (see test_gui.py)."""
    from i2as.gui import app_settings

    ini_path = tmp_path / "i2as_test_settings.ini"
    monkeypatch.setattr(
        app_settings,
        "get_settings",
        lambda: QSettings(str(ini_path), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr(
        app_settings, "autosave_file_path", lambda user_id=None: tmp_path / "auto.json"
    )
    return ini_path


@pytest.fixture
def instrument_host(qtbot):
    """A started host over the sim station, in this session's instrument mode."""
    host = build_host(CONFIG_PATH, tick_interval_ms=200)
    yield host
    shutdown_host(host)


@pytest.fixture
def station(instrument_host):
    return instrument_host.station


@pytest.fixture
def orchestrator(instrument_host):
    """The client adapter the windows are handed, as ``main.py`` hands it."""
    return instrument_host.build_proxy()


@pytest.fixture
def catalog_host(qtbot):
    """A host whose ENGINE also carries a run catalog, for agent run/probe tools.

    ``Gateway.call_tool("probe_run"/"run_procedure", ...)`` resolves the class
    name through ``Orchestrator._run_catalog`` (see ``orchestrator.py``'s
    ``_build_run``), which is separate from the session-layer queue's own
    catalog — so this is a second host construction rather than reusing
    ``build_host()``, which does not expose ``run_catalog``.
    """
    host = InstrumentHost(
        lambda: build_station(CONFIG_PATH),
        mode=instrument_mode(),
        orchestrator_options={
            "tick_interval_ms": 200,
            "run_catalog": {"FieldSweep": FieldSweep},
        },
        join_timeout_ms=JOIN_TIMEOUT_MS,
    )
    host.start()
    yield host
    shutdown_host(host)


@pytest.fixture
def session_manager(tmp_path, station, orchestrator):
    """A real ExperimentManager wired for queued FieldSweep runs."""
    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    manager = ExperimentManager(
        store=ExperimentStore(tmp_path / "experiments"),
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
        station=station,
        run_catalog={"FieldSweep": FieldSweep},
    )
    orchestrator.install_run_queue(
        next_run=manager.next_run,
        queue_entries=manager.queue_entries,
        take_next_spec=manager.take_next_spec,
        build_spec=manager.build_spec,
    )
    return manager


@pytest.fixture
def monitor_win(station, orchestrator, session_manager, qtbot):
    """MonitorWindow shown via qtbot, closed on teardown.

    The window-liveness standard (``gui/widget_lifecycle.py``) keeps a
    shown window out of the garbage collector's reach until ``.close()``
    runs — deliberately so, but it means a fixture that never closes one
    leaves it (and its trend plots) alive and painting for the rest of the
    process, one per test in a whole-file run. With several scenarios here
    driving a real tick loop (unlike ``test_gui.py``'s bare-signal
    fixtures), that accumulation is what starves later tests' event-loop
    waits inside pyqtgraph's own paint code — closing here is what keeps
    each test's window from outliving it.
    """
    win = MonitorWindow(station, orchestrator, session_manager=session_manager)
    qtbot.addWidget(win)
    win.show()
    yield win
    win.close()


@pytest.fixture
def procedure_win(station, orchestrator, session_manager, tmp_path, qtbot):
    """ProcedureWindow wired to the same session layer as ``monitor_win``, closed on teardown."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    win = ProcedureWindow(
        station,
        orchestrator,
        get_sample_info=lambda: dict(SAMPLE_INFO),
        get_data_dir=lambda: str(data_dir),
        get_experiment_info=session_manager.experiment_context,
        queue_host=session_manager.run_queue_host,
    )
    qtbot.addWidget(win)
    win.show()
    yield win
    win.close()


class _SlowDatapointProcedure:
    """A three-point sweep whose ``measure()`` can be made slow.

    Deliberately minimal, mirroring ``test_instrument_thread.py``'s
    ``ThreadProbeProcedure`` — just enough of the procedure surface to put a
    known, long, synchronous call on the instrument thread so the GUI thread
    can be observed while it runs.
    """

    name = "Slow Datapoint Probe"

    def __init__(self, measure_seconds: float = 2.0) -> None:
        self._sweep = [0.01, 0.02, 0.03]
        self._index = 0
        self._measure_seconds = float(measure_seconds)
        self.measured = 0
        self.measuring = False

    def initiate(self) -> PhasePlan:
        return PhasePlan(
            targets={"magnet_z": Target(self._sweep[0])}, commands=(), wait_s=0.0
        )

    def change_sweep_step(self) -> StepPlan | None:
        self._index += 1
        if self._index >= len(self._sweep):
            return None
        return StepPlan(
            targets={"magnet_z": Target(self._sweep[self._index])}, wait_s=0.0
        )

    def measure(self) -> None:
        self.measuring = True
        if self._measure_seconds:
            time.sleep(self._measure_seconds)
        self.measured += 1
        self.measuring = False

    def standby(self) -> PhasePlan:
        return PhasePlan(targets={"magnet_z": Target(0.0)}, commands=(), wait_s=0.0)

    def get_progress(self) -> float:
        return self._index / len(self._sweep)


class _MagnetOnlyClaimProcedure(_SlowDatapointProcedure):
    """The same sweep, claiming only the magnet it drives.

    A narrowed **Claim** (GLOSSARY.md), as ``TimeSeries`` and the operations
    declare one: everything the run does not name stays under manual control
    for the length of the run — including being released to its own front
    panel.
    """

    name = "Magnet-Only Claim Probe"

    def claimed_vi_names(self) -> set[str]:
        return {"magnet_z"}


# ══════════════════════════════════════════════════════════════════════════
# Scenario 1 — click storm during a long datapoint
# ══════════════════════════════════════════════════════════════════════════


def test_click_storm_during_a_long_datapoint(
    monitor_win, procedure_win, station, orchestrator, qtbot
):
    """Clicking through the GUI mid-datapoint never freezes it; Pause lands after the point.

    Threaded mode is the standard being tested: a 50 ms GUI-thread heartbeat
    keeps firing (>= 20 in the 2 s the datapoint takes) while the engine
    thread sleeps inside ``measure()``, a screenshot taken mid-datapoint
    shows live instrument values (not "--"), and Pause's verdict arrives at
    once even though PAUSED itself only lands once the in-flight point is
    saved. Inline mode collapses engine and GUI onto one thread by design
    (the temporary Inline mode, GLOSSARY.md), so the 2 s ``measure()`` call
    IS the GUI thread being busy for 2 s — there is nothing to click through
    until it returns, so this leg only asserts the final state (state
    machine correctness, not responsiveness) exactly as the module docstring
    describes.
    """
    on_engine(orchestrator, lambda: _fast_magnet(station))
    orchestrator.start_monitoring()
    procedure = _SlowDatapointProcedure(measure_seconds=2.0)
    orchestrator.run_procedure(procedure)

    if instrument_mode() == "inline":
        qtbot.waitUntil(lambda: procedure.measured == 3, timeout=15000)
        qtbot.waitUntil(lambda: orchestrator.state == "IDLE", timeout=5000)
        return

    ticks: list[float] = []
    heartbeat = QTimer()
    heartbeat.setInterval(50)
    heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
    heartbeat.start()
    try:
        qtbot.waitUntil(lambda: procedure.measuring, timeout=10000)

        # Click through the Monitor window's tabs.
        monitor_win._page_tab_bar.setCurrentIndex(1)
        QApplication.processEvents()
        monitor_win._page_tab_bar.setCurrentIndex(0)
        QApplication.processEvents()

        # A screenshot mid-datapoint renders live values, not a frozen "--".
        shot = _screenshot(monitor_win, "01_click_storm_mid_datapoint")

        # Toggle the takeover strip.
        strip = monitor_win._takeover_strip
        strip._radios[ev.AgentGate.READ_ONLY.value].click()
        QApplication.processEvents()
        strip._radios[ev.AgentGate.ACTIVE.value].click()
        QApplication.processEvents()

        # Open the Procedure window (already built; bring it to front) and
        # keep clicking through the wait to prove the heartbeat survives it.
        procedure_win.raise_()
        procedure_win.activateWindow()

        ticks.clear()
        started = time.monotonic()
        qtbot.wait(1200)
        elapsed = time.monotonic() - started
        assert len(ticks) >= 15, (
            f"the GUI-thread heartbeat fired {len(ticks)} times in "
            f"{elapsed:.2f} s while the engine thread was inside measure() "
            "— the GUI is frozen"
        )

        # Pause is accepted at once, even mid-datapoint.
        request_id = procedure_win._orchestrator.pause_procedure()
        assert request_id
    finally:
        heartbeat.stop()

    qtbot.waitUntil(lambda: orchestrator.state == "PAUSED", timeout=10000)
    assert procedure.measured >= 1, "the point in flight was finished, not cut short"
    assert shot.exists()

    orchestrator.resume_procedure()
    qtbot.waitUntil(lambda: orchestrator.state == "IDLE", timeout=15000)
    assert procedure.measured == 3


def test_both_windows_read_pausing_between_the_click_and_the_pause_boundary(
    qtbot, tmp_path
):
    """A pause requested in MEASURING is visible as "Pausing" until it lands.

    The interval this covers is the one the operator sees: the click is
    taken, the run keeps measuring the point it is on (GLOSSARY.md's **Pause
    boundary**), and the state does not move. Both windows say so — the
    Monitor's status bar reads ``MEASURING · Pausing`` and the Procedure
    window's Pause button reads ``Pausing…`` — and both go back to their
    resting wording once PAUSED lands and there is nothing pending any more.

    Its own host, at a slower tick than the shared fixture's, so the click
    lands inside the MEASURING tick-gap by a comfortable margin rather than
    racing the engine's next tick.
    """
    host = build_host(CONFIG_PATH, tick_interval_ms=500)
    try:
        proxy = host.build_proxy()
        on_engine(proxy, lambda: _fast_magnet(host.station))
        monitor = MonitorWindow(host.station, proxy)
        qtbot.addWidget(monitor)
        monitor.show()
        procedure_window = ProcedureWindow(
            host.station,
            proxy,
            get_sample_info=lambda: dict(SAMPLE_INFO),
            get_data_dir=lambda: str(tmp_path),
        )
        qtbot.addWidget(procedure_window)
        procedure_window.show()
        pause_btn = procedure_window.findChild(QPushButton, "pause_btn")
        assert pause_btn.text() == "Pause"

        proxy.run_procedure(_SlowDatapointProcedure(measure_seconds=0.0))
        # MEASURING as the CLIENT sees it (a mirror read) — the same fact the
        # operator's click would be based on.
        qtbot.waitUntil(lambda: proxy.state == "MEASURING", timeout=15000)
        pause_btn.click()
        qtbot.waitUntil(lambda: procedure_window._mirror.pause_pending(), timeout=5000)

        assert proxy.state == "MEASURING", "the pause is deferred, not taken yet"
        assert monitor._state_label.text() == "State: MEASURING · Pausing"
        assert pause_btn.text() == "Pausing…"
        _screenshot(monitor, "01b_monitor_pausing")
        _screenshot(procedure_window, "01b_procedure_pausing")

        qtbot.waitUntil(lambda: proxy.state == "PAUSED", timeout=15000)
        settled(proxy)
        assert monitor._state_label.text() == "State: PAUSED"
        assert pause_btn.text() == "Pause"

        proxy.abort_procedure()
        qtbot.waitUntil(lambda: proxy.state == "IDLE", timeout=15000)
        monitor.close()
        procedure_window.close()
    finally:
        shutdown_host(host)


# ══════════════════════════════════════════════════════════════════════════
# Scenario 2 — disconnect an instrument mid-run
# ══════════════════════════════════════════════════════════════════════════


def test_disconnect_swaps_the_card_reconnect_swaps_it_back_and_station_info_redeclares_it(
    monitor_win, station, orchestrator, qtbot
):
    """Disconnect (while IDLE) swaps the card, re-declares the VI offline, and reconnects.

    The whole round trip — release, re-declaration, reconnect — with nothing
    running, so no claim is in play at all. The next test is the same swap
    happening MID-RUN for a VI the run does not claim, plus the refusal for
    the one it does.
    """
    from PyQt6.QtWidgets import QGroupBox

    card = monitor_win.findChild(QGroupBox, "temperature_panel")
    assert card is not None
    card.findChild(QPushButton, "temperature_disconnect_btn").click()
    settled(orchestrator)

    assert station.has_vi("temperature") is False
    assert "temperature" in monitor_win._offline_cards
    offline_card = monitor_win.findChild(QGroupBox, "temperature_offline_card")
    assert offline_card is not None

    mirror = monitor_win._mirror
    assert mirror.offline_reason("temperature")

    def _availability(name: str) -> tuple[str, ...]:
        info = next(i for i in station.station_info().instruments if i.name == name)
        return info.availability

    assert "operator" in _availability("temperature"), (
        "the station's own declaration snapshot re-declares the offline VI "
        "(the operator-disconnect availability tag)"
    )

    # Reconnect swaps the card back.
    offline_card.findChild(QPushButton, "temperature_connect_btn").click()
    settled(orchestrator)
    assert station.has_vi("temperature") is True
    assert monitor_win._offline_cards == {}
    assert monitor_win.findChild(QGroupBox, "temperature_panel") is not None
    assert "operator" not in _availability("temperature")


def test_disconnect_mid_run_is_refused_for_the_claimed_vi_and_allowed_for_the_free_one(
    monitor_win, station, orchestrator, qtbot
):
    """Mid-run, Disconnect follows the CLAIM: refused for the run's VI, allowed for the rest.

    ``Orchestrator.disconnect_instrument()`` runs the same admission
    predicate every manual action does (see its docstring), so releasing an
    instrument is refused exactly where controlling it would be. The magnet
    this run claims is refused with a reason in the banner and stays
    connected; the temperature controller it does not claim is released while the run
    keeps going — its card swaps to the offline form and the station's own
    declaration re-declares it offline, exactly as at IDLE.
    """
    from PyQt6.QtWidgets import QGroupBox

    on_engine(orchestrator, lambda: _fast_magnet(station))
    orchestrator.start_monitoring()
    orchestrator.run_procedure(_MagnetOnlyClaimProcedure(measure_seconds=0.05))
    qtbot.waitUntil(lambda: orchestrator.state != "IDLE", timeout=5000)

    # The VI the run claims: refused, named, still connected.
    monitor_win._banner.hide()
    monitor_win.findChild(QGroupBox, "magnet_z_panel").findChild(
        QPushButton, "magnet_z_disconnect_btn"
    ).click()
    settled(orchestrator)
    assert station.has_vi("magnet_z") is True, "the claimed VI must stay connected"
    assert monitor_win._offline_cards == {}
    assert monitor_win._banner.isVisible()
    assert "magnet_z" in monitor_win._banner._label.text()

    # The VI it does not claim: released mid-run, card swapped, run continues.
    monitor_win.findChild(QGroupBox, "temperature_panel").findChild(
        QPushButton, "temperature_disconnect_btn"
    ).click()
    settled(orchestrator)
    assert station.has_vi("temperature") is False
    assert "temperature" in monitor_win._offline_cards
    assert monitor_win.findChild(QGroupBox, "temperature_offline_card") is not None
    info = next(
        i for i in station.station_info().instruments if i.name == "temperature"
    )
    assert "operator" in info.availability
    assert orchestrator.state != "IDLE", "the run carries on without it"

    qtbot.waitUntil(lambda: orchestrator.state == "IDLE", timeout=15000)


# ══════════════════════════════════════════════════════════════════════════
# Scenario 3 — a fault during a run
# ══════════════════════════════════════════════════════════════════════════


def test_a_fault_during_a_run_fails_it_and_acknowledge_retry_recover_it(
    monitor_win, station, orchestrator, qtbot
):
    """A comm fault on the claimed VI fails the run; Acknowledge/Retry recover it."""

    on_engine(orchestrator, lambda: _fast_magnet(station))
    orchestrator.run_procedure(_SlowDatapointProcedure(measure_seconds=0.05))
    qtbot.waitUntil(lambda: orchestrator.state != "IDLE", timeout=5000)

    on_engine(orchestrator, lambda: scenarios.apply_disconnect(station, "magnet_z"))
    qtbot.waitUntil(lambda: orchestrator.state == "IDLE", timeout=10000)
    qtbot.waitUntil(lambda: "magnet_z" in station.vi_faults(), timeout=5000)

    assert monitor_win._banner.isVisible()

    panel = next(p for p in monitor_win._panels if p.vi_name == "magnet_z")
    qtbot.waitUntil(lambda: panel._fault_row.isVisible(), timeout=5000)
    ack_btn = panel.findChild(QPushButton, "magnet_z_ack_fault_btn")
    retry_btn = panel.findChild(QPushButton, "magnet_z_retry_fault_btn")
    assert ack_btn is not None and ack_btn.isEnabled()
    assert retry_btn is not None

    ack_btn.click()
    settled(orchestrator)
    assert station.vi_faults()["magnet_z"].acknowledged is True

    # The instrument actually recovers before Retry is clicked (never mind
    # whether the fault reached "stale" or "disconnected" severity, which is
    # a timing-dependent race against the error-count threshold and not
    # what this scenario is about): Station.retry_fault() either re-polls
    # the same handle or rebuilds the driver session first
    # (Station._retry_disconnected(), past the error threshold), and either
    # way a driver that no longer errors answers, clearing the fault.
    on_engine(
        orchestrator,
        lambda: setattr(station.get_vi("magnet_z")._driver, "_simulate_error", False),
        settle=False,
    )
    retry_btn.click()
    settled(orchestrator)
    qtbot.waitUntil(lambda: "magnet_z" not in station.vi_faults(), timeout=5000)
    qtbot.waitUntil(lambda: not panel._fault_row.isVisible(), timeout=5000)
    for btn in panel._control_buttons.values():
        assert btn.isEnabled()


# ══════════════════════════════════════════════════════════════════════════
# Scenario 4 — emergency from the GUI
# ══════════════════════════════════════════════════════════════════════════


def test_emergency_standby_during_measuring_reaches_emergency_and_acknowledge_restores(
    monitor_win, procedure_win, station, orchestrator, tmp_path, qtbot
):
    """``emergency_standby()`` from MEASURING reaches EMERGENCY; Acknowledge restores IDLE.

    There is no dedicated "Emergency" button in the GUI today — the only
    manual all-stop control it exposes is "Standby All"
    (``submit_global_action("standby_all")``), which aborts the active run
    but does NOT enter EMERGENCY (see ``Orchestrator.submit_global_action``).
    The one route into EMERGENCY the engine exposes to any client —
    ``emergency_standby()`` — is what this scenario drives, through the same
    ``OrchestratorProxy`` handle the window holds, to exercise the GUI's
    reaction: the banner, the Acknowledge button, and the queue NOT
    auto-starting once acknowledged.
    """
    on_engine(orchestrator, lambda: _fast_magnet(station))
    procedure = _SlowDatapointProcedure(measure_seconds=0.2)
    orchestrator.run_procedure(procedure)
    qtbot.waitUntil(lambda: orchestrator.state != "IDLE", timeout=5000)

    # Queue a second run behind the active one.
    procedure_win._queue_panel.add_run(
        FieldSweep, dict(FAST_FIELD_SWEEP_PARAMS), dict(SAMPLE_INFO), str(tmp_path), ""
    )
    settled(orchestrator)
    queued_before = len(procedure_win._queue_panel._host.snapshot())
    assert queued_before >= 1

    orchestrator.emergency_standby("scenario test")
    qtbot.waitUntil(lambda: orchestrator.state == "EMERGENCY", timeout=10000)

    assert monitor_win._ack_btn.isVisible()
    assert monitor_win._banner.isVisible()
    _screenshot(monitor_win, "04_emergency")

    monitor_win._ack_btn.click()
    settled(orchestrator)
    qtbot.waitUntil(lambda: orchestrator.state == "IDLE", timeout=10000)
    assert not monitor_win._ack_btn.isVisible()

    # A queued item does not auto-start on acknowledge (no chain, per the
    # Orchestrator's own run_queue() docstring: "_acknowledge_emergency() —
    # no chain"). Monitoring was never turned on in this scenario (nothing
    # here needs live instrument-value polling, only state transitions and
    # faults, which fire regardless), which sidesteps a separate concern:
    # feeding a fast tick's states_updated into the trend plots for real
    # wall-clock seconds has been observed to make a later, unrelated
    # processEvents() call in this same test (settled()/qtbot.wait()) take
    # far longer than the data volume would suggest — nothing this scenario
    # is about, so avoided here rather than chased.
    assert orchestrator.state == "IDLE"
    assert len(procedure_win._queue_panel._host.snapshot()) == queued_before


def test_cards_reflect_the_standby_emergency_standby_actually_performed(
    monitor_win, station, orchestrator, qtbot
):
    """The card's lifecycle toggle reflects the standby EMERGENCY performed.

    ``_enter_emergency()`` stands every VI down through
    ``Station.standby_all()``, which deliberately bypasses the per-VI action
    queue and so emits no ``action_succeeded`` at all. The card learns it
    anyway, because the Initiate/Standby toggle RENDERS the lifecycle state
    the ``StatusSnapshot`` carries (GLOSSARY.md's **Lifecycle state**)
    rather than tracking the actions it happened to see — which is exactly
    what used to leave the operator's card claiming the instrument was
    still running.
    """
    on_engine(orchestrator, lambda: _fast_magnet(station))
    orchestrator.start_monitoring()
    orchestrator.submit_vi_action("magnet_z", "initiate")
    # A manual action is QUEUED for the tick (the tick is the single hardware
    # writer), and the snapshot that reports its consequence is that tick's.
    tick_engine(orchestrator, times=2)

    panel = next(p for p in monitor_win._panels if p.vi_name == "magnet_z")
    assert panel._lifecycle.is_initiated() is True

    orchestrator.emergency_standby("scenario test")
    qtbot.waitUntil(lambda: orchestrator.state == "EMERGENCY", timeout=10000)
    # The shutdown runs inside _enter_emergency(), after the state change
    # emitted its snapshot, so the snapshot that reports it is the next one.
    tick_engine(orchestrator)

    assert panel._lifecycle.is_initiated() is False, (
        "the magnet was actually put into standby by emergency_standby(), "
        "but its card still shows it as initiated"
    )

    monitor_win._ack_btn.click()
    settled(orchestrator)


def test_a_card_shows_the_standby_an_agent_asked_for_through_the_gateway(
    monitor_win, station, orchestrator, qtbot
):
    """An agent stands one VI down; the human's card says so, with no GUI action.

    The whole path is the agent's: a ``SUBMIT_VI_ACTION`` command stamped
    with the agent's own actor, authorised by its role (``standby`` is a
    recovery action), carried out on the tick. Nothing was clicked in this
    window, and the card still ends up rendering the state the engine
    reports — which is the point of the lifecycle state travelling the
    contract.
    """
    on_engine(orchestrator, lambda: _fast_magnet(station))
    orchestrator.start_monitoring()
    orchestrator.submit_vi_action("magnet_z", "initiate")
    tick_engine(orchestrator, times=2)

    panel = next(p for p in monitor_win._panels if p.vi_name == "magnet_z")
    assert panel._lifecycle.is_initiated() is True

    gateway = Gateway(
        engine_of(orchestrator),
        Role.SESSION,
        "runner-7",
        station_info=station.station_info,
    )
    gateway.submit(
        ev.CommandName.SUBMIT_VI_ACTION,
        {"vi_name": "magnet_z", "method_name": "standby"},
    )
    tick_engine(orchestrator, times=2)

    assert panel._mirror.lifecycle_state("magnet_z") == "standby"
    assert panel._lifecycle.is_initiated() is False, (
        "an agent stood magnet_z down through the gateway, but the "
        "operator's card still shows it as initiated"
    )
    _screenshot(panel, "04b_card_after_agent_standby")


def test_a_card_shows_the_initiate_the_operator_asked_for_from_the_cli(
    tmp_path, qtbot
):
    """A command that arrives through the Request spool reaches the card too.

    The out-of-process client (``i2as.ctl``) never touches the GUI: it
    drops a request file, the running engine drains it on its next tick, and
    the only thing that can carry the consequence back to the window is the
    ``StatusSnapshot``. Built on its own host because the spool is an engine
    construction option.
    """
    spool = RequestSpool(
        tmp_path / "spool", max_role=Role.SESSION.value, authorizer=authorize_spooled
    )
    host = InstrumentHost(
        lambda: build_station(CONFIG_PATH),
        mode=instrument_mode(),
        orchestrator_options={"tick_interval_ms": 200, "request_spool": spool},
        join_timeout_ms=JOIN_TIMEOUT_MS,
    )
    host.start()
    try:
        orchestrator = host.build_proxy()
        win = MonitorWindow(host.station, orchestrator)
        qtbot.addWidget(win)
        panel = next(p for p in win._panels if p.vi_name == "magnet_z")
        assert panel._lifecycle.is_initiated() is False

        spool.write_request(
            ev.Command(
                name=ev.CommandName.SUBMIT_VI_ACTION,
                actor=ev.Actor(
                    kind=ev.ActorKind.AGENT, id="ctl-1", role=Role.SESSION.value
                ),
                args={"vi_name": "magnet_z", "method_name": "initiate"},
            ),
            Role.SESSION.value,
        )
        # Two ticks: the first drains the spool and queues the action, the
        # second carries it out and reports it on that tick's snapshot.
        tick_engine(orchestrator, times=3)

        assert panel._mirror.lifecycle_state("magnet_z") == "initiated"
        assert panel._lifecycle.is_initiated() is True, (
            "the operator initiated magnet_z from the CLI, but the Monitor "
            "window's card still shows it as idle"
        )
    finally:
        shutdown_host(host)


# ══════════════════════════════════════════════════════════════════════════
# Scenario 5 — the agent panel shows the agent
# ══════════════════════════════════════════════════════════════════════════


def test_the_agent_panel_shows_an_agents_validate_probe_run_story(
    catalog_host, tmp_path, qtbot
):
    """validate -> probe -> run, as one agent, rendered row by row in the Agent panel.

    Uses its own host (``catalog_host``) because the story is driven through
    the Gateway's rendered TOOL surface (``validate_run``/``probe_run``/
    ``run_procedure``), which resolves a class name through the engine's OWN
    ``run_catalog`` — separate from ``monitor_win``'s queue-layer catalog.
    """
    station = catalog_host.station
    orchestrator = catalog_host.build_proxy()
    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    # Deliberately not shown: this test asserts on the Agent panel's rows,
    # never on-screen geometry, and letting two real FieldSweep runs' ticks
    # reach a VISIBLE window's trend plots has been observed to make a
    # later, unrelated processEvents() call in this same test (settled()'s
    # drain()) take far longer than the data volume would suggest — a
    # test-harness/pyqtgraph-paint concern this scenario is not about.
    panel = win._agent_panel
    strip = win._takeover_strip

    on_engine(orchestrator, lambda: _fast_magnet(station))

    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    manager = ExperimentManager(
        store=ExperimentStore(tmp_path / "experiments"),
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
        station=station,
        run_catalog={"FieldSweep": FieldSweep},
    )
    context = ToolContext(experiments=manager, run_catalog={"FieldSweep": FieldSweep})
    gateway = Gateway(
        engine_of(orchestrator),
        Role.SESSION,
        "runner-7",
        station_info=station.station_info,
        tool_context=context,
    )

    data_dir = str(tmp_path)
    params = dict(FAST_FIELD_SWEEP_PARAMS)

    validated = gateway.call_tool(
        "validate_run",
        {
            "procedure": "FieldSweep",
            "params": params,
            "sample_info": dict(SAMPLE_INFO),
            "data_directory": data_dir,
        },
    )
    assert validated["ok"] is True, validated
    settled(orchestrator)

    probed = gateway.call_tool(
        "probe_run",
        {
            "procedure": "FieldSweep",
            "params": params,
            "sample_info": dict(SAMPLE_INFO),
            "data_directory": data_dir,
            "file_prefix": "probe",
            "probe_spec": {"n_points": 2, "averaging": 1, "max_wait_s": 0.0},
        },
    )
    assert probed["ok"] is True, probed
    qtbot.waitUntil(lambda: orchestrator.state == "IDLE", timeout=15000)

    ran = gateway.call_tool(
        "run_procedure",
        {
            "procedure": "FieldSweep",
            "params": params,
            "sample_info": dict(SAMPLE_INFO),
            "data_directory": data_dir,
            "file_prefix": "full",
        },
    )
    assert ran["ok"] is True, ran
    qtbot.waitUntil(lambda: orchestrator.state == "IDLE", timeout=15000)
    settled(orchestrator)

    # The panel is a filter of the engine's Command/Verdict/Event stream, not
    # of the tool surface: ``validate_run`` is a READ session tool answered
    # locally (never submitted to the engine, so it never gets a verdict and
    # never shows here — the empty check below); ``probe_run`` submits the
    # same underlying ``RUN_PROCEDURE`` command as a plain run, so both the
    # probe and the full run show as two separate "run_procedure" rows, not
    # distinguishable by tool name at this layer.
    rows = panel.row_texts()
    assert not any("validate_run" in row for row in rows)
    assert sum("run_procedure → OK" in row for row in rows) == 2
    assert all("runner-7" in row for row in rows)
    assert not any(action.refused for action in panel.actions() if action.actor_id == "runner-7")

    # A refusal is visually distinct and names the rule.
    strip._radios[ev.AgentGate.REVOKED.value].click()
    settled(orchestrator)
    gateway.submit(ev.CommandName.START_MONITORING)
    settled(orchestrator)

    refusals = [a for a in panel.actions() if a.refused]
    assert refusals, "the kill switch's own refusal must be visible"
    assert refusals[-1].code == ev.VerdictCode.BLOCKED_ROLE.value
    assert "revoked" in refusals[-1].reason
    qtbot.wait(1)  # the row's dynamic property is polished on the next turn
    row_widget = panel._row_widgets[-1]
    label = row_widget.layout().itemAt(0).widget()
    from i2as.gui.agent_panel import OUTCOME_REFUSED

    assert label.property("outcome") == OUTCOME_REFUSED

    # The strip's own gate/attendance are what the gateway's permission check
    # reads — set here, seen there.
    strip._radios[ev.AgentGate.ACTIVE.value].click()
    settled(orchestrator)
    assert gateway.agent_gate() == ev.AgentGate.ACTIVE
    win.close()


# ══════════════════════════════════════════════════════════════════════════
# Scenario 6 — the session envelope
# ══════════════════════════════════════════════════════════════════════════


def test_a_narrow_envelope_blocks_an_agent_and_widening_it_lets_the_command_through(
    monitor_win, station, orchestrator, session_manager, qtbot
):
    """An agent's submit_vi_action outside the envelope is BLOCKED_ENVELOPE; widening frees it."""
    session_manager.start_experiment("Hall bar A3", "jdoe", {})
    settled(orchestrator)
    header = monitor_win._session_info
    editor = header._envelope_editor
    _lo, hi = station.get_vi("magnet_z").limit_bounds("field_T")

    editor._enabled_checkbox.setChecked(True)
    editor._rows["magnet_z"][1].setText("0.01")
    header._envelope_apply_btn.click()
    settled(orchestrator)
    assert session_manager.current_experiment().envelope["magnet_z"]["max_value"] == 0.01

    gateway = Gateway(
        engine_of(orchestrator), Role.SESSION, "runner-9", station_info=station.station_info
    )
    verdicts: list[ev.Verdict] = []
    orchestrator.verdict.connect(verdicts.append)
    gateway.submit(
        ev.CommandName.SUBMIT_VI_ACTION,
        {"vi_name": "magnet_z", "method_name": "set_field", "target_T": hi},
    )
    settled(orchestrator)

    blocked = [v for v in verdicts if v.code == ev.VerdictCode.BLOCKED_ENVELOPE]
    assert blocked, "the narrow envelope must refuse the agent's out-of-envelope target"
    assert "envelope" in blocked[-1].reason.lower()

    panel = monitor_win._agent_panel
    refusals = [a for a in panel.actions() if a.refused]
    assert refusals and refusals[-1].code == ev.VerdictCode.BLOCKED_ENVELOPE.value

    # The operator widens the envelope; the same command now succeeds.
    editor._rows["magnet_z"][1].setText(f"{hi:g}")
    header._envelope_apply_btn.click()
    settled(orchestrator)

    verdicts.clear()
    gateway.submit(
        ev.CommandName.SUBMIT_VI_ACTION,
        {"vi_name": "magnet_z", "method_name": "set_field", "target_T": hi},
    )
    # SUBMIT_VI_ACTION queues for the tick when it is not refused eagerly (see
    # Orchestrator.submit()'s docstring), so its verdict lands one or more
    # real ticks later rather than on the very next settled() round trip.
    qtbot.waitUntil(
        lambda: any(v.command == ev.CommandName.SUBMIT_VI_ACTION for v in verdicts),
        timeout=5000,
    )
    succeeded = [v for v in verdicts if v.command == ev.CommandName.SUBMIT_VI_ACTION]
    assert succeeded[-1].ok, succeeded[-1]


# ══════════════════════════════════════════════════════════════════════════
# Scenario 7 — Probe first from the queue panel, run to completion
# ══════════════════════════════════════════════════════════════════════════


def test_probe_first_then_run_queue_yields_a_probe_file_then_the_full_run(
    procedure_win, station, orchestrator, session_manager, qtbot
):
    """Probe first queues a probe ahead of the run; Run Queue yields probe, then full run."""
    on_engine(orchestrator, lambda: _fast_magnet(station))
    orchestrator.start_monitoring()

    # More field steps than the default probe's n_points (3), so the probe's
    # reduction is actually visible in the point count, not just its label.
    params = dict(FAST_FIELD_SWEEP_PARAMS, field_steps=11)

    panel = procedure_win._queue_panel
    data_dir = Path(session_manager.current_data_dir() or session_manager.store.root)
    data_dir.mkdir(parents=True, exist_ok=True)
    spec = panel.add_run(FieldSweep, params, dict(SAMPLE_INFO), str(data_dir), "")
    assert spec is not None
    settled(orchestrator)
    panel._select_spec(spec.spec_id)

    panel.parent()  # no-op, keeps the fixture chain explicit
    from i2as.gui.queue_panel import DEFAULT_PROBE_SPEC

    procedure_win.findChild(QPushButton, "queue_probe_btn").click()
    settled(orchestrator)

    order = panel._host.snapshot()
    assert [entry.probe_spec != {} for entry in order] == [True, False]
    assert order[0].probe_spec == DEFAULT_PROBE_SPEC.to_json()
    assert panel._probe_label.isVisible()

    finished: list[dict] = []
    orchestrator.run_finished.connect(finished.append)
    procedure_win.findChild(QPushButton, "run_queue_btn").click()

    # Wait on run_finished (fired once the data file is closed), never on the
    # file merely existing on disk — a probe with this few points can finish
    # (file created, dataset not yet written) inside the same event-loop turn
    # that a glob check runs in. Run Queue chains, so both runs land here
    # with no extra click.
    qtbot.waitUntil(lambda: len(finished) >= 2, timeout=20000)
    qtbot.waitUntil(lambda: orchestrator.state == "IDLE", timeout=5000)
    settled(orchestrator)

    files = sorted(data_dir.glob("*.h5"))
    assert len(files) == 2, f"expected a probe file then the full run's file, got {files}"
    probe_files = [f for f in files if f.name.startswith("probe")]
    assert len(probe_files) == 1, files

    import h5py

    for f in files:
        with h5py.File(f, "r") as handle:
            n_points = handle["data"]["field_T"].shape[0]
            kind = handle["metadata"].attrs.get("run_kind", "")
        if f in probe_files:
            assert kind == "probe"
            assert n_points == 3, "DEFAULT_PROBE_SPEC.n_points == 3 (first/middle/last)"
        else:
            assert kind != "probe"
            assert n_points == params["field_steps"]


# ══════════════════════════════════════════════════════════════════════════
# Scenario 8 — the ELN flow
# ══════════════════════════════════════════════════════════════════════════


def _run_one_field_sweep(station, orchestrator, session_manager, qtbot) -> str:
    """Drive one real FieldSweep to completion and return its run id."""
    on_engine(orchestrator, lambda: _fast_magnet(station))
    orchestrator.start_monitoring()
    data_dir = session_manager.current_data_dir()
    procedure = FieldSweep(
        station=station,
        sample_info=dict(SAMPLE_INFO),
        data_directory=str(data_dir),
        **FAST_FIELD_SWEEP_PARAMS,
    )
    orchestrator.run_procedure(procedure)
    qtbot.waitUntil(lambda: orchestrator.state == "IDLE", timeout=15000)
    settled(orchestrator)
    return session_manager.current_experiment().runs[-1].run_id


def test_a_finished_run_produces_one_outbox_job_and_one_eln_entry(
    station, orchestrator, session_manager, qtbot
):
    """With the sim ELN adapter configured, a finished run auto-publishes exactly once."""
    session_manager.start_experiment("Sample A", "jdoe", {})
    settled(orchestrator)

    settings = ElnSettings(
        enabled=True,
        backend="sim_eln",
        base_url="https://sim.example",
        api_key="k",
        retry_base_s=0.0,
        retry_max_s=0.0,
        drain_interval_s=0.05,
    )
    publisher = ElnPublisher(session_manager, settings, adapter=SimElnAdapter({}))
    orchestrator.run_finished.connect(publisher.on_run_finished)
    session_manager.attach_eln_publisher(publisher)
    publisher.start()
    try:
        _run_one_field_sweep(station, orchestrator, session_manager, qtbot)

        qtbot.waitUntil(lambda: publisher._adapter.entries != {}, timeout=5000)
        assert len(publisher._adapter.entries) == 1, "exactly one outbox job, one entry"
        (entry,) = publisher._adapter.entries.values()
        assert "Field Sweep" in entry["title"]
    finally:
        publisher.stop()


def test_an_attended_agents_eln_draft_needs_approval_then_the_approve_button_queues_it(
    monitor_win, station, orchestrator, session_manager, qtbot
):
    """Attended: draft_eln_entry then publish_eln_entry is refused; Approve queues it once.

    Auto-publish is off here so the finished run produces no outbox job of
    its own — the only queuer in this test is the operator's Approve click,
    which is what "enqueues exactly one job" means.
    """
    session_manager.start_experiment("Sample A", "jdoe", {})
    settled(orchestrator)

    settings = ElnSettings(
        enabled=True,
        backend="sim_eln",
        base_url="https://sim.example",
        api_key="k",
        retry_base_s=0.0,
        retry_max_s=0.0,
        drain_interval_s=0.05,
        auto_publish=False,
    )
    publisher = ElnPublisher(session_manager, settings, adapter=SimElnAdapter({}))
    orchestrator.run_finished.connect(publisher.on_run_finished)
    session_manager.attach_eln_publisher(publisher)
    publisher.start()
    try:
        run_id = _run_one_field_sweep(station, orchestrator, session_manager, qtbot)
        assert publisher.pending_count() == 0
        assert publisher._adapter.entries == {}, "auto-publish is off: nothing queued yet"

        # Attended: drafting is fine but publishing is refused, and the draft
        # is parked on the run record for a human to approve.
        session_manager.set_attended(True)
        from i2as.session.eln.drafting import FakeDraftClient

        context = ToolContext(
            experiments=session_manager,
            run_catalog={"FieldSweep": FieldSweep},
            publisher=publisher,
            draft_client=FakeDraftClient("The sweep completed cleanly."),
        )
        gateway = Gateway(
            engine_of(orchestrator),
            Role.SESSION,
            "runner-3",
            station_info=station.station_info,
            tool_context=context,
        )
        draft = gateway.call_tool("draft_eln_entry", {"run_id": run_id})
        assert draft["ok"] is True, draft
        answer = gateway.call_tool(
            "publish_eln_entry", {"run_id": run_id, "draft": draft["result"]}
        )
        assert answer["ok"] is False
        assert answer["detail"]["rule"] == "approval_required"

        button = monitor_win.findChild(QPushButton, f"agent_approve_{run_id}")
        assert button is not None, "a pending draft is a row with an Approve button"
        assert publisher.pending_count() == 0

        button.click()

        assert publisher.pending_count() == 1, "exactly one job, queued by the Approve click"
        assert session_manager.pending_eln_draft(run_id) == {}
        qtbot.waitUntil(lambda: publisher.pending_count() == 0, timeout=5000)
        assert len(publisher._adapter.entries) == 1
        (entry,) = publisher._adapter.entries.values()
        assert "sweep completed cleanly" in entry["body_html"]
    finally:
        publisher.stop()


# ══════════════════════════════════════════════════════════════════════════
# Scenario 9 — shutdown while measuring
# ══════════════════════════════════════════════════════════════════════════


def test_closing_the_monitor_window_mid_run_shuts_down_bounded_and_the_file_is_readable(
    station, orchestrator, instrument_host, tmp_path, qtbot
):
    """closeEvent -> shutdown() completes in bound; the data file is closed and readable."""
    import h5py

    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    win.show()

    on_engine(orchestrator, lambda: _fast_magnet(station))
    orchestrator.start_monitoring()
    procedure = FieldSweep(
        station=station,
        sample_info=dict(SAMPLE_INFO),
        data_directory=str(tmp_path),
        **dict(FAST_FIELD_SWEEP_PARAMS, field_steps=21, readings_per_point=1),
    )
    orchestrator.run_procedure(procedure)
    qtbot.waitUntil(lambda: orchestrator.state != "IDLE", timeout=5000)

    started = time.monotonic()
    win.close()
    instrument_host.shutdown()
    elapsed = time.monotonic() - started
    assert elapsed < JOIN_TIMEOUT_MS / 1000.0 + 2.0, (
        f"shutdown took {elapsed:.1f}s — not bounded"
    )

    for _ in range(5):
        QApplication.processEvents()

    files = list(tmp_path.glob("*.h5"))
    assert len(files) == 1
    with h5py.File(files[0], "r") as handle:
        assert handle["metadata"].attrs["procedure_name"] == "Field Sweep"
        assert handle["data"]["field_T"].shape[0] >= 1


# ══════════════════════════════════════════════════════════════════════════
# Scenario 10 — window liveness regression
# ══════════════════════════════════════════════════════════════════════════


def _build_and_forget_a_monitor_window() -> None:
    """Show a MonitorWindow while deliberately keeping no reference to it."""
    from i2as.core.orchestrator import Orchestrator

    station = build_station(CONFIG_PATH)
    orch = Orchestrator(station, tick_interval_ms=50)
    win = MonitorWindow(station, orch)
    win.show()
    orch.shutdown()


def test_a_monitor_window_with_no_strong_reference_survives_a_gc_pass(qtbot):
    """A shown window outlives ``gc.collect()`` with no crash — the liveness standard."""
    import gc

    from i2as.gui import widget_lifecycle

    _build_and_forget_a_monitor_window()

    gc.collect()

    shown = [
        w
        for w in QApplication.topLevelWidgets()
        if isinstance(w, MonitorWindow) and w.isVisible()
    ]
    assert shown, "a shown window must not be garbage-collected"
    win = shown[-1]
    qtbot.addWidget(win)

    for _ in range(10):
        QApplication.processEvents()
    qtbot.wait(500)

    assert win.isVisible()
    win.close()
    assert not any(w is win for w in widget_lifecycle.held_windows())
