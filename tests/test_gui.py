"""GUI smoke tests — Layer 6.

These tests use pytest-qt (qtbot fixture). They run against the sim_cryostat
config with no hardware.

They run in BOTH instrument modes. The shared fixtures build the engine
through an ``InstrumentHost`` whose mode comes from
``I2AS_INSTRUMENT_THREAD`` (``tests/instrument_modes.py``), and the
``orchestrator`` fixture hands the windows an ``OrchestratorProxy`` — which is
what ``main.py`` hands them — so the same 190-odd assertions hold with the
engine on this thread and with it on its own:

    pytest tests/test_gui.py
    I2AS_INSTRUMENT_THREAD=1 pytest tests/test_gui.py

A test that reaches past the client boundary — forcing a state, setting a
private the engine only writes inside a tick — goes through the **tick
helper** family (``on_engine()``, ``set_on_engine()``, ``tick_engine()``)
rather than touching the engine from this thread.
"""

import gc
import logging
from pathlib import Path

import pytest
from PyQt6 import sip
from PyQt6.QtCore import Qt, QEvent, QSettings
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QWidget,
)

from i2as.core.config_catalog import ConfigCatalog
from i2as.core.decorators import control, monitored
from i2as.core.events import ErrorEvent
from i2as.core.exceptions import I2ASCommunicationError
from i2as.core.orchestrator import Orchestrator, OrchestratorState
from i2as.core.plan import ParamSpec
from i2as.core.station import build_station
from i2as.gui import app_settings as _app_settings
from i2as.gui import form_autosave as session_store
from i2as.gui import window_geometry
from i2as.gui.instrument_front_panel import InstrumentFrontPanel
from i2as.gui.instrument_panel import InstrumentPanel
from i2as.gui.monitor_window import MonitorWindow
from i2as.gui.notification_banner import NotificationBanner
from i2as.gui.procedure_window import ProcedureWindow
from i2as.gui.theme import (
    BANNER_ERROR_TEXT,
    BANNER_WARNING_TEXT,
    TEXT_ON_ACCENT,
    TEXT_PRIMARY,
    build_stylesheet,
)
from i2as.gui.trend_plot_panel import TrendPlotPanel
from i2as.gui import widget_lifecycle
from i2as.virtual_instruments.base import BaseVirtualInstrument
from tests.instrument_modes import (
    build_host,
    engine_of,
    on_engine,
    set_on_engine,
    settled,
    shutdown_host,
    ticks_paused,
)


CONFIG_PATH = "i2as/configs/sim_cryostat"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Redirect the app QSettings factory to a throwaway INI file.

    Dependency seam: both windows call ``app_settings.get_settings()`` for
    geometry persistence. Monkeypatching that factory to an INI file under
    ``tmp_path`` means a pytest run never reads or overwrites the user's real
    saved geometry in the Windows registry. Autouse so every GUI test is
    isolated without opting in.

    Yields:
        The Path of the throwaway INI file, so a test can inspect what was
        written to it.
    """
    from i2as.gui import app_settings

    ini_path = tmp_path / "i2as_test_settings.ini"

    def _fake_get_settings():
        return QSettings(str(ini_path), QSettings.Format.IniFormat)

    monkeypatch.setattr(app_settings, "get_settings", _fake_get_settings)

    # Same seam for the JSON session file: redirect it into tmp_path so a pytest
    # run never reads or overwrites the user's real last_session.json in AppData.
    session_path = tmp_path / "last_session.json"

    def _fake_autosave_file_path(user_id=None):
        return (tmp_path / "sessions" / f"{user_id}.json") if user_id else session_path

    monkeypatch.setattr(app_settings, "autosave_file_path", _fake_autosave_file_path)
    # The fixed measurement root (ExperimentInfoPanel's fallback whenever no
    # experiment is open) is isolated globally by conftest.py's
    # isolated_measurement_root fixture, autouse across the whole suite.
    return ini_path


@pytest.fixture
def instrument_host(qtbot):
    """A started host over the sim station, in this session's instrument mode.

    The one place the mode enters this suite: ``inline`` builds the stack on
    the test's own thread, ``threaded`` builds it inside a ``QThread``, and
    every fixture below is written so nothing else in the file can tell.

    The teardown stops the tick timer and cuts the engine's connections
    before qtbot destroys the widget tree — a tick or a queued delivery
    landing in a half-destroyed tree is the historical source of the rare
    RuntimeError/segfault flakes.
    """
    host = build_host(CONFIG_PATH, tick_interval_ms=50)
    yield host
    shutdown_host(host)


@pytest.fixture
def station(instrument_host):
    """The simulated station the host built."""
    return instrument_host.station


@pytest.fixture
def orchestrator(instrument_host):
    """The client adapter the windows are handed, as ``main.py`` hands it.

    An ``OrchestratorProxy``: one typed method per command, the engine's
    signals re-exposed, and every read answered from the status mirror.
    Monitoring stays OFF (the production launch state) — these tests drive
    updates by emitting the proxy's signals directly, so they need no real
    polling ticks.
    """
    return instrument_host.build_proxy()


@pytest.fixture
def monitor_win(station, orchestrator, qtbot):
    """MonitorWindow shown via qtbot."""
    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    win.show()
    return win


@pytest.fixture
def procedure_win(station, orchestrator, qtbot):
    """ProcedureWindow shown via qtbot with stub sample-info callables."""
    def _sample_info():
        return {"sample_name": "test", "sample_id": "T001", "comments": ""}

    def _data_dir():
        return "C:/CryoData"

    win = ProcedureWindow(
        station, orchestrator,
        get_sample_info=_sample_info,
        get_data_dir=_data_dir,
    )
    qtbot.addWidget(win)
    win.show()
    return win


def _publish_state(orchestrator, state):
    """Publish one station-state snapshot the way a real tick does.

    A tick refreshes the client's status mirror and THEN emits
    ``states_updated``; a test that emits only the latter leaves every
    mirror read (fault rows, availability tags) answering from the previous
    tick. Order matters — the panels read the mirror inside their
    ``states_updated`` slot.

    Args:
        orchestrator: The engine to publish from.
        state: The ``{vi_name: {field: value}}`` snapshot to emit.
    """
    engine = engine_of(orchestrator)
    on_engine(orchestrator, engine._emit_status_snapshot)
    orchestrator.states_updated.emit(state)


def _mock_mirror(orchestrator, vi_name, vi):
    """A status mirror whose declaration names one ad-hoc mock VI.

    The panels build from the station declaration, so a test VI that is not
    in the sim config needs one declared for it. Built by the same machinery
    production uses — an in-process `Station` with the VI registered — so the
    test exercises the real rendering path rather than a hand-written
    `InstrumentInfo`.

    Args:
        orchestrator: The engine the mirror otherwise mirrors.
        vi_name: The name to declare the VI under.
        vi: The mock VI instance.

    Returns:
        The primed StatusMirror.
    """
    from i2as.core.station import Station
    from i2as.core.status_mirror import StatusMirror

    declaring = Station()
    declaring.register_vi(vi_name, vi, "measurement")
    engine = engine_of(orchestrator)
    # The priming reads happen where the engine is; the mirror itself is
    # built here, so its own connections deliver on this thread.
    mirror = StatusMirror()
    mirror.prime(
        *on_engine(
            orchestrator,
            lambda: (
                engine.station_info(),
                engine.status_snapshot(),
                engine.get_operational_status(),
            ),
        )
    )
    mirror.attach(engine)
    mirror.prime(station_info=declaring.station_info())
    return mirror


# ── MonitorWindow tests ───────────────────────────────────────────────────────

def test_monitor_window_has_global_buttons(monitor_win):
    """Initiate All and Standby All buttons exist."""
    initiate_btn = monitor_win.findChild(QPushButton, "initiate_all_btn")
    standby_btn = monitor_win.findChild(QPushButton, "standby_all_btn")
    assert initiate_btn is not None, "initiate_all_btn not found"
    assert standby_btn is not None, "standby_all_btn not found"


def test_status_bar_updates_on_state_change(monitor_win, orchestrator, qtbot):
    """MonitorWindow status bar label reflects Orchestrator state."""
    orchestrator.state_changed.emit("RAMPING")
    assert "RAMPING" in monitor_win._state_label.text()


# ── InstrumentPanel tests ─────────────────────────────────────────────────────

def test_instrument_panel_creates_value_labels(station, orchestrator, qtbot):
    """InstrumentPanel creates one QLabel per @monitored method."""
    from i2as.core.decorators import get_monitored_methods

    vi_name = "magnet_z"
    vi = station._virtual_instruments[vi_name]
    panel = InstrumentPanel(vi_name, orchestrator)
    qtbot.addWidget(panel)

    monitored = get_monitored_methods(vi)
    for method_name in monitored:
        widget = panel.findChild(QLabel, f"{vi_name}_{method_name}_value")
        assert widget is not None, f"Missing value label for {method_name}"


def test_instrument_panel_creates_control_buttons(station, orchestrator, qtbot):
    """InstrumentPanel creates one QPushButton per @control method."""
    from i2as.core.decorators import get_control_methods

    vi_name = "magnet_z"
    vi = station._virtual_instruments[vi_name]
    panel = InstrumentPanel(vi_name, orchestrator)
    qtbot.addWidget(panel)

    controls = get_control_methods(vi)
    for method_name in controls:
        btn = panel.findChild(QPushButton, f"{vi_name}_{method_name}_btn")
        assert btn is not None, f"Missing button for {method_name}"


def test_instrument_panel_lifecycle_buttons_exist(station, orchestrator, qtbot):
    """InstrumentPanel has a single lifecycle toggle button (Initiate/Standby)."""
    vi_name = "temperature"
    panel = InstrumentPanel(vi_name, orchestrator)
    qtbot.addWidget(panel)

    assert panel.findChild(QPushButton, f"{vi_name}_lifecycle_btn") is not None
    assert panel.findChild(QPushButton, f"{vi_name}_initiate_btn") is None
    assert panel.findChild(QPushButton, f"{vi_name}_standby_btn") is None


# ── The lifecycle toggle renders the snapshot (the lifecycle-state standard;
#    GLOSSARY.md's Lifecycle state) ─────────────────────────────────────────


def _publish_snapshot(orchestrator):
    """Emit one status snapshot from the engine and let it reach this thread."""
    engine = engine_of(orchestrator)
    on_engine(orchestrator, engine._emit_status_snapshot)


def test_a_card_renders_the_lifecycle_state_the_snapshot_carries(
    monitor_win, station, orchestrator, qtbot
):
    """The toggle follows the engine, including a stand-down nobody clicked.

    ``standby_all()`` is the path an emergency takes: it bypasses the per-VI
    action queue and emits no ``action_succeeded``, so the only thing that
    can reach the card is the lifecycle state on the snapshot.
    """
    panel = next(p for p in monitor_win._panels if p.vi_name == "magnet_z")
    assert panel._lifecycle.is_initiated() is False

    on_engine(orchestrator, station.initiate_all)
    _publish_snapshot(orchestrator)
    assert panel._lifecycle.is_initiated() is True

    on_engine(orchestrator, station.standby_all)
    _publish_snapshot(orchestrator)
    assert panel._lifecycle.is_initiated() is False


def test_the_next_snapshot_corrects_an_optimistic_lifecycle_flip(
    monitor_win, station, orchestrator, qtbot
):
    """``_on_action_succeeded()`` may flip early; it no longer owns the truth."""
    panel = next(p for p in monitor_win._panels if p.vi_name == "magnet_z")
    panel._lifecycle.set_initiated(True)  # as the optimistic flip would
    assert panel._lifecycle.is_initiated() is True

    _publish_snapshot(orchestrator)
    assert panel._lifecycle.is_initiated() is False, (
        "the snapshot says magnet_z is idle, so the card must stop claiming "
        "it is initiated"
    )


def test_a_card_opens_showing_an_already_initiated_instrument(
    station, orchestrator, qtbot
):
    """A card built mid-experiment starts on the truth, not on 'Initiate'."""
    on_engine(orchestrator, station.initiate_all)
    _publish_snapshot(orchestrator)

    panel = InstrumentPanel("magnet_z", orchestrator)
    qtbot.addWidget(panel)

    assert panel._lifecycle.is_initiated() is True


def test_the_front_panels_toggle_follows_the_snapshot_too(
    station, orchestrator, qtbot
):
    """The front panel's embedded card cannot disagree with the monitor card."""
    win = InstrumentFrontPanel("magnet_z", orchestrator)
    qtbot.addWidget(win)
    assert win._panel._lifecycle.is_initiated() is False

    on_engine(orchestrator, station.initiate_all)
    _publish_snapshot(orchestrator)
    assert win._panel._lifecycle.is_initiated() is True


def test_instrument_panel_updates_values_on_signal(station, orchestrator, qtbot):
    """states_updated signal → value labels reflect new state."""
    vi_name = "magnet_z"
    panel = InstrumentPanel(vi_name, orchestrator)
    qtbot.addWidget(panel)

    # Emit a fake state with known field value
    fake_state = {vi_name: {"magnet_field_T": 1.5, "magnet_current": 15.0, "magnet_status": "HOLD"}}
    orchestrator.states_updated.emit(fake_state)

    # Unconditional: test_instrument_panel_creates_value_labels already proves a
    # value label exists for every @monitored method, so a missing label here is
    # a real regression, not a reason to skip the assertion.
    field_label = panel.findChild(QLabel, f"{vi_name}_magnet_field_T_value")
    assert field_label is not None, f"no value label for {vi_name}.magnet_field_T"
    assert "1.5" in field_label.text()


def test_instrument_panel_stale_border(station, orchestrator, qtbot):
    """Stale state sets the 'stale' status property (amber border via QSS)."""
    vi_name = "magnet_z"
    panel = InstrumentPanel(vi_name, orchestrator)
    qtbot.addWidget(panel)

    orchestrator.states_updated.emit({vi_name: {"_stale": True}})
    assert panel.property("status") == "stale"
    assert "[stale]" in panel._name_label.text()


def test_instrument_panel_disconnected_border(station, orchestrator, qtbot):
    """Disconnected state sets the 'disconnected' status property (red border via QSS)."""
    vi_name = "magnet_z"
    panel = InstrumentPanel(vi_name, orchestrator)
    qtbot.addWidget(panel)

    orchestrator.states_updated.emit({vi_name: {"_stale": True, "_disconnected": True}})
    assert panel.property("status") == "disconnected"
    # "NOT RESPONDING", not "DISCONNECTED": under the connection-lifecycle
    # standard "disconnected" is the operator's own verb (and the offline
    # card's badge), so a comm fault must not claim the same word.
    assert "[NOT RESPONDING]" in panel._name_label.text()


def test_instrument_panel_status_resets_to_ok(station, orchestrator, qtbot):
    """A stale panel returns to 'ok' status (plain title) when state is healthy again."""
    vi_name = "magnet_z"
    panel = InstrumentPanel(vi_name, orchestrator)
    qtbot.addWidget(panel)

    orchestrator.states_updated.emit({vi_name: {"_stale": True}})
    assert panel.property("status") == "stale"

    orchestrator.states_updated.emit({vi_name: {}})
    assert panel.property("status") == "ok"
    assert panel._name_label.text() == f"<b>{vi_name}</b>"


def test_instrument_panel_fault_row_disables_controls_and_wires_ack_retry(
    station, orchestrator, qtbot
):
    """A real runtime fault shows the fault row, disables controls, and wires
    Acknowledge/Retry through Orchestrator — the RUNTIME sibling of the
    offline fault card.
    """
    vi_name = "magnet_z"
    vi = station._virtual_instruments[vi_name]
    panel = InstrumentPanel(vi_name, orchestrator)
    qtbot.addWidget(panel)
    panel.show()
    assert not panel._fault_row.isVisible()
    assert panel._control_buttons  # sanity: at least one @control button exists
    for btn in panel._control_buttons.values():
        assert btn.isEnabled()

    # Force a real fault via the Station's fault registry (not a synthetic
    # states_updated payload) so Orchestrator.vi_faults() actually reports it.
    vi._driver._simulate_error = True
    state = station.get_state()
    _publish_state(orchestrator, state)

    assert panel._fault_row.isVisible()
    for btn in panel._control_buttons.values():
        assert not btn.isEnabled()
    ack_btn = panel.findChild(QPushButton, f"{vi_name}_ack_fault_btn")
    retry_btn = panel.findChild(QPushButton, f"{vi_name}_retry_fault_btn")
    assert ack_btn is not None and ack_btn.isEnabled()
    assert retry_btn is not None

    ack_btn.click()
    settled(orchestrator)
    assert station.vi_faults()[vi_name].acknowledged is True
    # Re-emit the same tick's snapshot: the Acknowledge button reflects the
    # now-acknowledged fault (disabled — nothing left to acknowledge).
    _publish_state(orchestrator, state)
    assert not ack_btn.isEnabled()

    # Retry while still broken: action_failed, fault stands (still faulted).
    with qtbot.waitSignal(orchestrator.action_failed, timeout=500):
        retry_btn.click()
    assert vi_name in station.vi_faults()

    # Instrument recovers; retry now succeeds and the fault clears — the
    # panel reflects that on the next states_updated (a real poll here).
    vi._driver._simulate_error = False
    with qtbot.waitSignal(orchestrator.action_succeeded, timeout=500):
        retry_btn.click()
    assert vi_name not in station.vi_faults()
    state = station.get_state()
    _publish_state(orchestrator, state)
    assert not panel._fault_row.isVisible()
    for btn in panel._control_buttons.values():
        assert btn.isEnabled()


class _ToggleableFaultDriver:
    """Test double whose identity/reading calls fail while ``broken`` is True.

    A CLASS-level flag, unlike the sim drivers' instance-level
    ``_simulate_error``: models hardware that is still disconnected even
    after ``retry_fault()`` opens a fresh session — an instance-level flag
    would be left behind on the old, discarded driver object once rebuilt,
    hiding the exact bug this double exists to reproduce.
    """

    broken: bool = False

    def __init__(self, resource_string: str) -> None:
        self.closed = False

    def get_idn(self) -> str:
        if type(self).broken:
            raise I2ASCommunicationError("bus session is dead")
        return "I2AS,GUI-TOGGLE-STUB,0,0"

    def get_reading(self) -> float:
        if type(self).broken:
            raise I2ASCommunicationError("bus session is dead")
        return 1.0

    def close(self) -> None:
        self.closed = True


class _ToggleableFaultVI(BaseVirtualInstrument):
    """VI double with one @monitored getter and one @control action, so the
    front-panel fault row and its control-button-disabling both have
    something real to exercise.
    """

    vi_type = "system"

    @monitored
    def reading(self) -> float:
        return self._drivers["main"].get_reading()

    @control
    def nudge(self, value: float = 0.0) -> float:
        return value


def _write_toggleable_config(tmp_path: Path) -> str:
    """One-VI config wiring _ToggleableFaultVI/_ToggleableFaultDriver."""
    (tmp_path / "devices.yaml").write_text(
        "real_drivers:\n"
        "  toggle_drv:\n"
        "    class: tests.test_gui._ToggleableFaultDriver\n"
        '    address: "SIM::TOGGLE"\n'
        "virtual_instruments:\n"
        "  toggle_vi:\n"
        "    class: tests.test_gui._ToggleableFaultVI\n"
        "    drivers: {main: toggle_drv}\n"
        "    vi_type: system\n"
    )
    (tmp_path / "monitor.yaml").write_text(
        "monitor:\n  tick_interval_ms: 1000\n  max_vi_errors: 3\n"
    )
    return str(tmp_path)


def test_instrument_panel_retry_button_rebuilds_a_disconnected_instrument(
    tmp_path, qtbot
):
    """The exact reported bug, driven through the real front-panel button: a
    hard-disconnected instrument's Retry must actually reconnect it once the
    hardware is fixed — not just re-poll the same dead session forever.
    """
    _ToggleableFaultDriver.broken = False
    station = build_station(_write_toggleable_config(tmp_path))
    orch = Orchestrator(station, tick_interval_ms=10)
    try:
        vi_name = "toggle_vi"
        original_driver = station.get_vi(vi_name)._drivers["main"]
        panel = InstrumentPanel(vi_name, orch)
        qtbot.addWidget(panel)
        panel.show()
        assert not panel._fault_row.isVisible()

        # Accidental hardware disconnect: three consecutive comm failures
        # escalate the fault from "stale" to "disconnected".
        _ToggleableFaultDriver.broken = True
        for _ in range(3):
            state = station.get_state()
            _publish_state(orch, state)
        assert station.vi_faults()[vi_name].kind == "disconnected"
        assert panel._fault_row.isVisible()
        for btn in panel._control_buttons.values():
            assert not btn.isEnabled()

        retry_btn = panel.findChild(QPushButton, f"{vi_name}_retry_fault_btn")
        assert retry_btn is not None

        # Still non-responsive: clicking Retry must fail honestly, not fake
        # a recovery.
        with qtbot.waitSignal(orch.action_failed, timeout=500):
            retry_btn.click()
        assert vi_name in station.vi_faults()
        assert station.vi_faults()[vi_name].kind == "disconnected"

        # Hardware fixed: Retry now rebuilds the session and recovers —
        # this is the exact scenario a bare re-poll of the old handle could
        # never satisfy.
        _ToggleableFaultDriver.broken = False
        with qtbot.waitSignal(orch.action_succeeded, timeout=500):
            retry_btn.click()
        assert vi_name not in station.vi_faults()
        # Proof it was a genuine rebuild, not a re-poll of the same handle.
        assert station.get_vi(vi_name)._drivers["main"] is not original_driver

        state = station.get_state()
        _publish_state(orch, state)
        assert not panel._fault_row.isVisible()
        for btn in panel._control_buttons.values():
            assert btn.isEnabled()
    finally:
        orch.shutdown()


def test_instrument_panel_retry_button_blocked_while_run_claims_the_instrument(
    tmp_path, qtbot
):
    """The front panel must refuse to rebuild a claimed instrument's session
    mid-run — the safety gate that stops a rebuild from bypassing the run's
    own review, even when the operator can see the hardware is fixed.
    """
    _ToggleableFaultDriver.broken = False
    station = build_station(_write_toggleable_config(tmp_path))
    orch = Orchestrator(station, tick_interval_ms=10)
    try:
        vi_name = "toggle_vi"
        original_driver = station.get_vi(vi_name)._drivers["main"]
        panel = InstrumentPanel(vi_name, orch)
        qtbot.addWidget(panel)
        panel.show()

        _ToggleableFaultDriver.broken = True
        for _ in range(3):
            state = station.get_state()
            _publish_state(orch, state)
        assert station.vi_faults()[vi_name].kind == "disconnected"

        _ToggleableFaultDriver.broken = False  # hardware IS fixed now...
        orch._procedure = object()  # ...but a run claims the instrument
        orch._active_claims = {vi_name}
        retry_btn = panel.findChild(QPushButton, f"{vi_name}_retry_fault_btn")

        with qtbot.waitSignal(orch.action_blocked, timeout=500):
            retry_btn.click()

        assert vi_name in station.vi_faults()
        assert station.vi_faults()[vi_name].kind == "disconnected"
        assert station.get_vi(vi_name)._drivers["main"] is original_driver
    finally:
        orch._procedure = None
        orch._active_claims = None
        orch.shutdown()


def test_monitor_window_banner_shows_and_clears_vi_fault_warning(
    station, orchestrator, monitor_win, qtbot
):
    """MonitorWindow's banner shows a per-VI fault warning (error_event) and
    calms once every runtime fault clears."""
    vi_name = "magnet_z"
    vi = station._virtual_instruments[vi_name]

    vi._driver._simulate_error = True
    state = station.get_state()
    fault = station.vi_faults()[vi_name]
    event = ErrorEvent(
        vi_name=vi_name, kind="fault", severity="warning",
        message=fault.message, timestamp=fault.since,
    )
    with qtbot.waitSignal(orchestrator.error_event, timeout=500):
        orchestrator.error_event.emit(event)
    assert monitor_win._banner.isVisible()
    assert vi_name in monitor_win._banner._label.text()

    vi._driver._simulate_error = False
    station.get_state()  # clears the Station-side fault record
    _publish_state(orchestrator, state)  # MonitorWindow polls vi_faults() here
    assert not monitor_win._banner.isVisible()


class _SpecControlVI(BaseVirtualInstrument):
    """Hand-made VI exercising every spec-driven widget shape on one card."""

    vi_type = "mock"

    @control(params={
        "target_K": ParamSpec(
            type=float, default=4.2, unit="K", min=1.5, max=300.0,
            description="Temperature setpoint.",
        ),
        "auto_pid": ParamSpec(type=bool, default=True),
        "sensor": ParamSpec(
            type=int, default=1, choices={"Sample": 1, "VTI": 2},
        ),
    })
    def set_temperature(self, target_K: float = 4.2, auto_pid: bool = True,
                        sensor: int = 1):
        return target_K

    @control  # legacy bare control: must keep the plain QLineEdit path
    def set_heater_power(self, power_W: float = 0.0):
        return power_W


@pytest.fixture
def spec_panel(orchestrator, qtbot):
    """InstrumentPanel over the spec-declaring mock VI, plus a submit spy."""
    vi = _SpecControlVI({})
    panel = InstrumentPanel("mock_vi", orchestrator, _mock_mirror(orchestrator, "mock_vi", vi))
    qtbot.addWidget(panel)
    submitted: list[tuple] = []
    orchestrator.submit_vi_action = lambda vi_name, method, **kw: submitted.append(
        (vi_name, method, kw)
    )
    return panel, submitted


def test_spec_controls_render_typed_widgets(spec_panel):
    """Declared ParamSpecs render combo/checkbox/line-edit, not all-QLineEdit."""
    panel, _ = spec_panel
    assert isinstance(
        panel.findChild(QWidget, "mock_vi_set_temperature_target_K_input"), QLineEdit
    )
    assert isinstance(
        panel.findChild(QWidget, "mock_vi_set_temperature_auto_pid_input"), QCheckBox
    )
    assert isinstance(
        panel.findChild(QWidget, "mock_vi_set_temperature_sensor_input"), QComboBox
    )
    # Legacy bare control still renders a plain line edit.
    assert isinstance(
        panel.findChild(QWidget, "mock_vi_set_heater_power_power_W_input"), QLineEdit
    )
    # Spec-built field carries the tooltip (description + default + range).
    field = panel.findChild(QWidget, "mock_vi_set_temperature_target_K_input")
    assert "Temperature setpoint" in field.toolTip()
    assert "1.5 to 300.0" in field.toolTip()


def test_spec_controls_submit_typed_values(spec_panel):
    """Collected kwargs are typed: float from text, bool from checkbox, mapped
    choice value (not its label) from the combo."""
    panel, submitted = spec_panel
    panel.findChild(QWidget, "mock_vi_set_temperature_target_K_input").setText("77")
    panel.findChild(QWidget, "mock_vi_set_temperature_auto_pid_input").setChecked(False)
    panel.findChild(QWidget, "mock_vi_set_temperature_sensor_input").setCurrentText("VTI")

    panel._submit_control("set_temperature")
    assert submitted == [
        ("mock_vi", "set_temperature",
         {"target_K": 77.0, "auto_pid": False, "sensor": 2}),
    ]


def test_spec_controls_emptied_field_falls_back_to_method_default(spec_panel):
    """Clearing a spec-built line edit omits the kwarg (method default applies)."""
    panel, submitted = spec_panel
    panel.findChild(QWidget, "mock_vi_set_temperature_target_K_input").setText("")
    panel._submit_control("set_temperature")
    assert len(submitted) == 1
    assert "target_K" not in submitted[0][2]


# ── Arming controls render from measurement_parameters ───────────────────────
# initiate_measurement takes its parameters via **params, so the panel can only
# type its widgets from the specs MeasurementInstrumentBase installs on it from
# measurement_parameters. These pin that one declaration reaching the front
# panel: a bare arming control rendered untyped text boxes.


@pytest.fixture
def delta_front_panel(station, orchestrator, qtbot):
    """InstrumentFrontPanel over the sim station's DC measurement VI."""
    vi_name = "dc_measurement"
    panel = InstrumentFrontPanel(vi_name, orchestrator)
    qtbot.addWidget(panel)
    return vi_name, panel


def test_arming_control_renders_typed_widgets(delta_front_panel):
    """Every declared arming parameter gets its own typed widget."""
    vi_name, panel = delta_front_panel
    prefix = f"{vi_name}_initiate_measurement"
    for param in ("current_A", "compliance_A", "voltmeter_range_V", "readings_per_point"):
        assert isinstance(
            panel.findChild(QWidget, f"{prefix}_{param}_input"), QLineEdit
        ), f"{param} should render its own input"


def test_arming_control_labels_carry_units(delta_front_panel):
    """Every arming parameter with a unit is labelled with it."""
    _, panel = delta_front_panel
    labels = {lbl.text() for lbl in panel.findChildren(QLabel)}
    assert "current_A (A):" in labels
    assert "compliance_A (A):" in labels
    assert "voltmeter_range_V (V):" in labels


def test_arming_control_fields_carry_descriptions(delta_front_panel):
    """The declared description reaches the widget tooltip, not just the schema."""
    vi_name, panel = delta_front_panel
    field = panel.findChild(
        QWidget, f"{vi_name}_initiate_measurement_current_A_input"
    )
    assert field.toolTip().strip()


def test_reading_setter_renders_its_single_spec(delta_front_panel):
    """A reading_setters setter inherits the one measurement_parameters spec."""
    vi_name, panel = delta_front_panel
    field = panel.findChild(QWidget, f"{vi_name}_set_source_current_current_A_input")
    assert isinstance(field, QLineEdit)
    assert field.toolTip().strip()


def test_spec_controls_unparseable_value_aborts_submit(spec_panel, monkeypatch):
    """A non-numeric entry in a float field warns and submits nothing."""
    from PyQt6.QtWidgets import QMessageBox

    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *args: warnings.append(args[2])),
    )
    panel, submitted = spec_panel
    panel.findChild(QWidget, "mock_vi_set_temperature_target_K_input").setText("abc")
    panel._submit_control("set_temperature")
    assert submitted == []
    assert len(warnings) == 1


class _PanelFlagVI(BaseVirtualInstrument):
    """Mock VI with one card-default control and one front-panel-only control."""

    vi_type = "mock"

    @control
    def set_temperature(self, target_K: float = 4.2):
        return target_K

    @control(panel=False)
    def set_heater_power(self, power_W: float = 0.0):
        return power_W


def test_panel_false_control_hidden_by_default(orchestrator, qtbot):
    """Without a config allowlist, panel=False controls stay off the card."""
    panel = InstrumentPanel(
        "mock_vi", orchestrator, _mock_mirror(orchestrator, "mock_vi", _PanelFlagVI({}))
    )
    qtbot.addWidget(panel)
    assert panel.findChild(QPushButton, "mock_vi_set_temperature_btn") is not None
    assert panel.findChild(QPushButton, "mock_vi_set_heater_power_btn") is None


def test_config_allowlist_overrides_panel_defaults(orchestrator, qtbot):
    """A panels: allowlist wins in both directions: it can surface a
    panel=False control and hide a panel=True one."""
    vi = _PanelFlagVI({})
    panel = InstrumentPanel(
        "mock_vi", orchestrator, _mock_mirror(orchestrator, "mock_vi", vi),
        panel_controls=["set_heater_power"],
    )
    qtbot.addWidget(panel)
    assert panel.findChild(QPushButton, "mock_vi_set_heater_power_btn") is not None
    assert panel.findChild(QPushButton, "mock_vi_set_temperature_btn") is None


def test_monitor_window_threads_panels_config(station, orchestrator, qtbot):
    """MonitorWindow passes each VI's panels: allowlist into its card."""
    win = MonitorWindow(
        station, orchestrator,
        panels_config={"magnet_z": []},  # hide every magnet_z control
    )
    qtbot.addWidget(win)
    win.show()
    assert win.findChild(QPushButton, "magnet_z_set_field_btn") is None
    # An unlisted VI keeps its declared defaults.
    assert win.findChild(QPushButton, "temperature_set_temperature_btn") is not None


def _control_grid(panel, vi_name, method_name):
    """Return the QGridLayout of a stacked multi-param control block."""
    btn = panel.findChild(QPushButton, f"{vi_name}_{method_name}_btn")
    container = btn.parentWidget()
    layout = container.layout()
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if isinstance(item.layout(), QGridLayout):
            return item.layout()
    return None


def test_multi_param_control_stacks_in_one_column(orchestrator, qtbot):
    """3-10 parameters render as a labelled grid under the button, one column."""

    class _SevenParamVI(BaseVirtualInstrument):
        vi_type = "mock"

        @control
        def arm(self, a: float = 1, b: float = 2, c: float = 3, d: float = 4,
                e: float = 5, f: float = 6, g: float = 7):
            pass

    panel = InstrumentPanel(
        "mock_vi", orchestrator, _mock_mirror(orchestrator, "mock_vi", _SevenParamVI({}))
    )
    qtbot.addWidget(panel)
    grid = _control_grid(panel, "mock_vi", "arm")
    assert grid is not None, "7-param control must stack, not render inline"
    assert grid.columnCount() == 2  # one label column + one field column
    assert grid.rowCount() == 7


def test_many_param_control_uses_two_columns(orchestrator, qtbot):
    """More than 10 parameters split the grid into two label+field columns."""

    class _ElevenParamVI(BaseVirtualInstrument):
        vi_type = "mock"

        @control
        def arm(self, p1: float = 1, p2: float = 1, p3: float = 1, p4: float = 1,
                p5: float = 1, p6: float = 1, p7: float = 1, p8: float = 1,
                p9: float = 1, p10: float = 1, p11: float = 1):
            pass

    panel = InstrumentPanel(
        "mock_vi", orchestrator, _mock_mirror(orchestrator, "mock_vi", _ElevenParamVI({}))
    )
    qtbot.addWidget(panel)
    grid = _control_grid(panel, "mock_vi", "arm")
    assert grid is not None
    assert grid.columnCount() == 4  # two label+field column pairs
    assert grid.rowCount() == 6  # ceil(11 / 2)


def test_two_param_control_stays_inline(orchestrator, qtbot):
    """Up to two parameters keep the compact inline row (no grid)."""
    panel = InstrumentPanel(
        "mock_vi", orchestrator, _mock_mirror(orchestrator, "mock_vi", _PanelFlagVI({}))
    )
    qtbot.addWidget(panel)
    assert _control_grid(panel, "mock_vi", "set_temperature") is None


def test_front_panel_button_opens_full_control_surface(orchestrator, qtbot):
    """The card's sliders icon opens a window showing every control —
    including panel=False ones the card hides — and reuses it on re-click."""
    card = InstrumentPanel(
        "mock_vi", orchestrator, _mock_mirror(orchestrator, "mock_vi", _PanelFlagVI({}))
    )
    qtbot.addWidget(card)

    fp_btn = card.findChild(QPushButton, "mock_vi_front_panel_btn")
    assert fp_btn is not None
    fp_btn.click()

    front = card._front_panel
    assert front is not None
    qtbot.addWidget(front)
    assert front.isVisible()
    # The card hides set_heater_power (panel=False); the front panel shows it.
    assert front.findChild(QPushButton, "mock_vi_set_heater_power_btn") is not None
    assert front.findChild(QPushButton, "mock_vi_set_temperature_btn") is not None
    # No recursion: the embedded panel carries no front-panel icon of its own.
    inner_buttons = front.findChildren(QPushButton, "mock_vi_front_panel_btn")
    assert inner_buttons == []
    # Second click reuses the same window.
    fp_btn.click()
    assert card._front_panel is front


def test_front_panel_ignores_config_allowlist(orchestrator, qtbot):
    """A monitor.yaml allowlist trims only the card; the front panel opened
    from that card still shows everything."""
    vi = _PanelFlagVI({})
    card = InstrumentPanel(
        "mock_vi", orchestrator, _mock_mirror(orchestrator, "mock_vi", vi),
        panel_controls=["set_temperature"],
    )
    qtbot.addWidget(card)
    card.findChild(QPushButton, "mock_vi_front_panel_btn").click()
    front = card._front_panel
    qtbot.addWidget(front)
    assert front.findChild(QPushButton, "mock_vi_set_heater_power_btn") is not None


def test_instrument_panel_status_not_restyled_when_unchanged(
    station, orchestrator, qtbot, monkeypatch
):
    """The status property is only re-set when it changes, not on every tick."""
    vi_name = "magnet_z"
    panel = InstrumentPanel(vi_name, orchestrator)
    qtbot.addWidget(panel)

    status_sets: list[object] = []
    original_set = panel.setProperty

    def spy(name, value):  # type: ignore[no-untyped-def]
        if name == "status":
            status_sets.append(value)
        return original_set(name, value)

    monkeypatch.setattr(panel, "setProperty", spy)

    # First tick changes ok -> stale (one set); the next two are unchanged.
    orchestrator.states_updated.emit({vi_name: {"_stale": True}})
    orchestrator.states_updated.emit({vi_name: {"_stale": True}})
    orchestrator.states_updated.emit({vi_name: {"_stale": True}})

    assert status_sets == ["stale"]
    assert panel.property("status") == "stale"


# ── ProcedureWindow tests ─────────────────────────────────────────────────────

def test_procedure_param_inputs_exist(procedure_win):
    """Parameter form inputs are created for the selected procedure.

    FieldSweep declares sweep_axis, so its hidden axis parameters (field_mode,
    field_start, ...) are handled by the SweepAxisWidget, not a flat QLineEdit.
    Its measurement parameters are station-dependent (from the selected
    measurement VI), rendered under the "Measurement method" selector.
    """
    from i2as.procedures.field_sweep import FieldSweep

    _select_procedure(procedure_win, FieldSweep.name)

    assert procedure_win._params_panel._axis_widget is not None

    # System params (temperature, init_wait, step_wait) render as flat inputs.
    for param_name in FieldSweep.system_parameters:
        field = procedure_win.findChild(QWidget, f"param_{param_name}_input")
        assert field is not None, f"Missing input for parameter '{param_name}'"

    # The measurement-method selector renders (a structural combobox).
    assert procedure_win.findChild(QComboBox, "param_measurement_vi_input") is not None


def test_procedure_param_label_and_tooltip(procedure_win):
    """Param label is the canonical `name (unit):` and carries the description tooltip.

    The label is the same key stored under /metadata/procedure_params in the
    HDF5 output (see BaseProcedure), not prose. The prose description lives in
    a tooltip on both the input field and its form label.

    Uses FieldSweep's ``temperature`` system_parameter rather than one of
    its sweep_axis-generated fields (e.g. field_start): those are rendered by
    SweepAxisWidget, not a flat QLineEdit + QFormLayout row, so they are not a
    valid target for this label/tooltip check.
    """
    from i2as.procedures.field_sweep import FieldSweep

    _select_procedure(procedure_win, FieldSweep.name)

    spec = FieldSweep.system_parameters["temperature"]
    field = procedure_win.findChild(QLineEdit, "param_temperature_input")
    assert field is not None, "Missing input for parameter 'temperature'"

    assert field.text() == str(spec.default)

    form = field.parent().layout()
    assert isinstance(form, QFormLayout)
    row_label = form.labelForField(field)
    assert isinstance(row_label, QLabel)
    assert row_label.text() == "temperature (K):"

    for tooltip in (field.toolTip(), row_label.toolTip()):
        assert tooltip, "Tooltip must be non-empty"
        assert spec.description in tooltip


def _select_procedure(procedure_win, name):
    """Select the procedure whose exact display name is *name*."""
    for i in range(procedure_win._params_panel._proc_selector.count()):
        if procedure_win._params_panel._proc_selector.itemText(i) == name:
            procedure_win._params_panel._proc_selector.setCurrentIndex(i)
            return
    pytest.fail(f"{name!r} not found in procedure selector")


def test_procedure_bool_widgets_render(procedure_win):
    """A bool param renders as a checkbox carrying its declared default.

    Covers the temperature toggle FieldSweep declares — the generic form
    must map it with no per-procedure code. (The full ParamSpec ->
    widget mapping, enumerated choices included, is pinned against
    ``param_form`` directly further down this file.)
    """
    from i2as.procedures.field_sweep import FieldSweep

    _select_procedure(procedure_win, FieldSweep.name)

    box = procedure_win.findChild(QCheckBox, "param_set_temperature_input")
    assert box is not None
    assert box.isChecked() is True    # default True


def test_procedure_bool_values_collected(procedure_win):
    """_collect_params reads each checkbox back as a real bool."""
    from i2as.procedures.field_sweep import FieldSweep

    _select_procedure(procedure_win, FieldSweep.name)

    procedure_win.findChild(QCheckBox, "param_set_temperature_input").setChecked(False)

    collected = procedure_win._collect_params()
    assert collected is not None
    param_values = collected[0]
    assert param_values["set_temperature"] is False


# ── Generic sweep procedure: structural measurement-VI re-render ──────────────

def _measurement_combo(win):
    """Return the measurement-method QComboBox on the current form."""
    combo = win.findChild(QComboBox, "param_measurement_vi_input")
    assert combo is not None, "measurement-method selector should be rendered"
    return combo


def _set_slot_parameter(win, slot_param_name, qualified):
    """Set a Reading-loop slot drop-down to the entry mapping to *qualified*."""
    combo = win.findChild(QComboBox, f"param_{slot_param_name}_input")
    assert combo is not None, f"{slot_param_name} selector should be rendered"
    groups = {g.key: g for g in win._params_panel._current_groups}
    spec = groups["reading_loop"].params[slot_param_name]
    label = next(k for k, v in spec.choices.items() if v == qualified)
    combo.setCurrentText(label)


def _select_measurement(win, vi_name):
    """Set the measurement combobox to the label whose mapped value is *vi_name*."""
    combo = _measurement_combo(win)
    for group in win._params_panel._current_groups:
        spec = group.params.get("measurement_vi")
        if spec is None:
            continue
        for label, value in spec.choices.items():
            if value == vi_name:
                combo.setCurrentText(str(label))
                return
    pytest.fail(f"measurement VI {vi_name!r} not in the selector")


def _settle_at_width(win, width=1280, height=800):
    """Resize the window to *width* x *height* and let the layout settle."""
    win.resize(width, height)
    win.show()
    QApplication.processEvents()


def _fully_inside_param_viewport(win, widget) -> bool:
    """Return True if *widget* is visible AND fully inside the param scroll viewport.

    Maps the widget's rectangle into the parameter scroll area's viewport
    coordinates; a widget scrolled off the right edge (the pre-fix overflow bug)
    maps to an x beyond the viewport width and fails this check. This is the
    geometry assertion that mere ``findChild`` existence checks did not catch.
    """
    if not widget.isVisible():
        return False
    viewport = win._params_panel._param_scroll.viewport()
    top_left = widget.mapTo(viewport, widget.rect().topLeft())
    bottom_right = widget.mapTo(viewport, widget.rect().bottomRight())
    return (
        top_left.x() >= 0
        and bottom_right.x() <= viewport.width()
        and bottom_right.x() > top_left.x()
    )


def test_generic_field_sweep_renders_measurement_select_and_default_group(procedure_win):
    """The form shows the measurement-method combo + the default VI's param group.

    The default measurement VI is the first registered one (dc_measurement in
    the sim config), so its parameters render inside the single composite
    "Measurement" column.
    """
    from i2as.procedures.field_sweep import FieldSweep

    _select_procedure(procedure_win, FieldSweep.name)

    combo = _measurement_combo(procedure_win)
    assert combo.count() == 1  # dc_measurement, the one shipped measurement VI
    # The default VI's params render.
    assert procedure_win.findChild(QLineEdit, "param_readings_per_point_input") is not None
    assert procedure_win.findChild(QLineEdit, "param_current_A_input") is not None
    # The selector + params live in ONE Measurement box (not a per-group column);
    # the composite box exists and the params key tracks the selected VI.
    assert procedure_win._params_panel._measurement_box is not None
    assert procedure_win._params_panel._measurement_params_key == "measurement:dc_measurement"
    # The Measurement box is NOT registered as an independent column.
    assert "measurement:dc_measurement" not in procedure_win._params_panel._group_boxes


def test_generic_field_sweep_all_four_columns_visible_no_hscroll(procedure_win, station):
    """At 1280 px, Sweep/System/Measurement/Reading loop fit with no h-scroll.

    This is the geometry regression for the reported bug: the measurement
    params and the rightmost column used to overflow off the right edge behind
    a horizontal scrollbar. Assert the actual on-screen geometry, not just
    widget existence.
    """
    from i2as.procedures.field_sweep import FieldSweep

    procedure_win._params_panel.select_procedure_by_name(FieldSweep.name)
    # Put the loopable parameter in slot 1 so the values input renders — the
    # widest state the Reading loop column takes.
    _set_slot_parameter(procedure_win, "loop1_parameter", "dc_measurement.current_A")
    _settle_at_width(procedure_win, 1280, 800)

    # No horizontal scrollbar is needed for the parameter form.
    assert procedure_win._params_panel._param_scroll.horizontalScrollBar().maximum() == 0

    # The selected VI's first parameter widget is fully inside the viewport.
    first_param = procedure_win.findChild(QLineEdit, "param_readings_per_point_input")
    assert first_param is not None
    assert _fully_inside_param_viewport(procedure_win, first_param)

    # The slot's values input is fully inside the viewport (not off-screen).
    values_widget = procedure_win._params_panel._param_inputs.get("loop1_values")
    assert values_widget is not None
    assert _fully_inside_param_viewport(procedure_win, values_widget)


def test_generic_field_sweep_collect_merges_params_for_the_selection(procedure_win):
    """_collect_params merges the selected VI's params with the procedure's own."""
    from i2as.procedures.field_sweep import FieldSweep

    _select_procedure(procedure_win, FieldSweep.name)

    values, *_ = procedure_win._collect_params()
    assert values["measurement_vi"] == "dc_measurement"
    assert "readings_per_point" in values and "current_A" in values
    assert "temperature" in values  # system param present too


def test_generic_field_sweep_method_combo_shows_selector_labels(procedure_win, station):
    """The method drop-down shows the SHORT selector_labels, vi_name in a tooltip.

    The combo used to show "vi_name — display_label" (too long; it forced the
    column wide). It now shows each VI's ``selector_label`` and carries the bare
    ``vi_name`` as a per-item tooltip; the collected value is still the vi_name.
    """
    from i2as.procedures.field_sweep import FieldSweep

    _select_procedure(procedure_win, FieldSweep.name)
    combo = _measurement_combo(procedure_win)

    items = [combo.itemText(i) for i in range(combo.count())]
    expected = [
        station.measurement_selector_label(n)
        for n in station.measurement_vi_names()
    ]
    assert items == expected
    assert items == ["DC (6221 + 2182A)"]

    # Each item carries its vi_name as a tooltip (disambiguation).
    tips = [
        combo.itemData(i, Qt.ItemDataRole.ToolTipRole)
        for i in range(combo.count())
    ]
    assert tips == station.measurement_vi_names()

    # The collected value stays the vi_name, not the label.
    values, *_ = procedure_win._collect_params()
    assert values["measurement_vi"] == "dc_measurement"


def test_live_plot_loop_selectors_follow_reading_loop(procedure_win, station):
    """The plots' two Loop selectors follow the reading-loop slots.

    Visible but disabled while something is loopable and the slots are off;
    enabled with one item per value — display text carrying the value, item
    data carrying the 0-based axis index — once a slot has two or more values.
    """
    from i2as.procedures.field_sweep import FieldSweep

    procedure_win._params_panel.select_procedure_by_name(FieldSweep.name)
    sel1 = procedure_win.findChild(QComboBox, "plot1_loop1_selector")
    sel2 = procedure_win.findChild(QComboBox, "plot1_loop2_selector")
    # Something loopable, slots off -> both selectors visible but disabled.
    assert sel1.isVisibleTo(procedure_win) and not sel1.isEnabled()
    assert sel2.isVisibleTo(procedure_win) and not sel2.isEnabled()

    # Slot 1: the DC current +/- pair.
    _set_slot_parameter(procedure_win, "loop1_parameter", "dc_measurement.current_A")
    # LoopValuesWidget doesn't have an objectName match by text — locate it
    # via the form's param-input registry instead and populate it directly.
    loop1_values_widget = procedure_win._params_panel._param_inputs.get("loop1_values")
    assert loop1_values_widget is not None, "loop1_values widget not found"
    loop1_values_widget.set_raw("1e-6, -1e-6")
    # Trigger structural re-render since the loop values widget is structural
    procedure_win._params_panel._on_structural_changed()

    sel1 = procedure_win.findChild(QComboBox, "plot1_loop1_selector")
    sel2 = procedure_win.findChild(QComboBox, "plot1_loop2_selector")
    assert sel1.isEnabled()
    assert [sel1.itemText(i) for i in range(sel1.count())] == [
        "A1 = 1e-06", "A2 = -1e-06",
    ]
    assert [sel1.itemData(i) for i in range(sel1.count())] == [0, 1]
    # Slot 2 is still off, so its selector stays visible but disabled.
    assert sel2.isVisibleTo(procedure_win) and not sel2.isEnabled()
    # Axis keys stay plain — the Loop selectors pick the reading. Raw-sample
    # arrays are saved but never offered as a plot axis.
    x_sel = procedure_win.findChild(QComboBox, "x1_axis_selector")
    keys = [x_sel.itemText(i) for i in range(x_sel.count())]
    assert "voltage_V" in keys
    assert not any("__A" in k or "__B" in k for k in keys)
    assert not any(k.endswith("_array") for k in keys)


def test_param_form_renders_all_widget_kinds_and_round_trips(qtbot):
    """param_form maps each ParamSpec kind to the right widget and round-trips values.

    Exercises i2as.gui.param_form directly (no ProcedureWindow) on a
    synthetic ParamGroup covering the four shapes: plain float, bounded int,
    enumerated choices, and bool.
    """
    from i2as.core.plan import ParamGroup, ParamSpec
    from i2as.gui import param_form

    group = ParamGroup(
        key="demo",
        title="Demo",
        params={
            "amp": ParamSpec(type=float, default=1.5, unit="V", description="Amplitude"),
            "count": ParamSpec(type=int, default=4, min=1, max=10, description="Count"),
            "range": ParamSpec(
                type=float,
                default=0.1,
                choices={"0.1 V": 0.1, "1 V": 1.0},
                description="Range",
            ),
            "enabled": ParamSpec(type=bool, default=True, description="Enabled"),
        },
    )
    box, widgets = param_form.build_group_box(group)
    qtbot.addWidget(box)

    assert box.title() == "Demo"
    assert isinstance(widgets["amp"], QLineEdit)
    assert isinstance(widgets["count"], QLineEdit)
    assert isinstance(widgets["range"], QComboBox)
    assert isinstance(widgets["enabled"], QCheckBox)

    # objectName convention preserved (findChild API relied on by other tests).
    assert widgets["amp"].objectName() == "param_amp_input"

    # Defaults are seeded onto the widgets.
    assert widgets["amp"].text() == "1.5"
    assert widgets["range"].currentText() == "0.1 V"   # label whose value is 0.1
    assert widgets["enabled"].isChecked() is True

    # collect_value reverses the mapping on the seeded defaults.
    assert param_form.collect_value(widgets["amp"], group.params["amp"]) == pytest.approx(1.5)
    assert param_form.collect_value(widgets["count"], group.params["count"]) == 4
    assert param_form.collect_value(widgets["range"], group.params["range"]) == pytest.approx(0.1)
    assert param_form.collect_value(widgets["enabled"], group.params["enabled"]) is True

    # Change values, re-collect: text parses by type, combobox maps label->value.
    widgets["count"].setText("7")
    widgets["range"].setCurrentText("1 V")
    widgets["enabled"].setChecked(False)
    assert param_form.collect_value(widgets["count"], group.params["count"]) == 7
    assert param_form.collect_value(widgets["range"], group.params["range"]) == pytest.approx(1.0)
    assert param_form.collect_value(widgets["enabled"], group.params["enabled"]) is False

    # Raw string round-trip used by the session cache.
    param_form.set_widget_raw(widgets["amp"], "2.5")
    assert param_form.get_widget_raw(widgets["amp"]) == "2.5"


# ── Experiment lifecycle (ExperimentInfoPanel) ──────────────────────────────────

@pytest.fixture
def session_manager(tmp_path, station, orchestrator):
    """ExperimentManager backed by a tmp_path store/roster, with one roster user."""
    from i2as.session.manager import ExperimentManager
    from i2as.session.models import User
    from i2as.session.store import ExperimentStore, UserRoster

    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe", email="jdoe@example.org"))
    store = ExperimentStore(tmp_path / "experiments")
    return ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
    )


@pytest.fixture
def monitor_win_session(station, orchestrator, session_manager, qtbot):
    """MonitorWindow wired to a real ExperimentManager."""
    win = MonitorWindow(station, orchestrator, session_manager=session_manager)
    qtbot.addWidget(win)
    win.show()
    return win


class _FakeStartDialog:
    """Stand-in for StartExperimentDialog that auto-accepts fixed values."""

    def __init__(self, values: tuple[str, str, bool, str | None], envelope=None) -> None:
        self._values = values
        self._envelope = envelope

    def exec(self):
        return QDialog.DialogCode.Accepted

    def result_values(self):
        return self._values

    def envelope(self):
        return self._envelope


class _FakeCloseDialog:
    """Stand-in for CloseExperimentDialog that auto-accepts fixed findings."""

    def __init__(self, findings_text: str) -> None:
        self._findings_text = findings_text

    def exec(self):
        return QDialog.DialogCode.Accepted

    def findings(self):
        return self._findings_text


def _stub_start_dialog(
    monkeypatch, title, user_id, attended=True, dirname=None, envelope=None
):
    """Replace StartExperimentDialog with a fake that auto-accepts ``values``."""
    from i2as.gui import experiment_info_panel as sip

    monkeypatch.setattr(
        sip,
        "StartExperimentDialog",
        lambda roster, parent=None, envelope_variables=None: _FakeStartDialog(
            (title, user_id, attended, dirname), envelope
        ),
    )


def _stub_close_dialog(monkeypatch, findings_text=""):
    """Replace CloseExperimentDialog with a fake that auto-accepts ``findings_text``."""
    from i2as.gui import experiment_info_panel as sip

    monkeypatch.setattr(
        sip,
        "CloseExperimentDialog",
        lambda current_findings="", parent=None: _FakeCloseDialog(findings_text),
    )


def test_experiment_row_disabled_without_session_manager(monitor_win):
    """The Start Experiment button is disabled when no ExperimentManager is wired."""
    btn = monitor_win._session_info._start_close_btn
    assert not btn.isEnabled()
    assert monitor_win._session_info._experiment_status_label.text() == "No experiment open"


def test_experiment_row_enabled_with_session_manager(monitor_win_session):
    """The Start Experiment button is enabled and shows the closed state."""
    panel = monitor_win_session._session_info
    assert panel._start_close_btn.isEnabled()
    assert panel._start_close_btn.text() == "Start Experiment…"
    assert not panel._attended_checkbox.isVisible()


def test_start_experiment_updates_panel_and_manager(monitor_win_session, session_manager, monkeypatch):
    """Clicking Start Experiment opens the dialog and installs the experiment."""
    _stub_start_dialog(monkeypatch, "Hall bar A3", "jdoe", attended=True)
    panel = monitor_win_session._session_info

    panel._start_close_btn.click()

    experiment = session_manager.current_experiment()
    assert experiment is not None
    assert experiment.title == "Hall bar A3"
    assert experiment.user_id == "jdoe"
    assert panel._start_close_btn.text() == "Close Experiment…"
    assert "Hall bar A3" in panel._experiment_status_label.text()
    assert "J. Doe" in panel._experiment_status_label.text()
    assert panel._attended_checkbox.isVisible()
    assert panel._attended_checkbox.isChecked()
    assert "not configured" in panel._eln_status_label.text()


def test_eln_status_shows_published_url_when_eln_link_set(
    monitor_win_session, session_manager, monkeypatch
):
    """Once ElnLink carries a url, the panel reflects it instead of the placeholder."""
    from i2as.session.models import ElnLink

    _stub_start_dialog(monkeypatch, "Hall bar A3", "jdoe")
    panel = monitor_win_session._session_info
    panel._start_close_btn.click()

    experiment = session_manager.current_experiment()
    experiment.eln_link = ElnLink(backend="elabftw", entry_id="42", url="https://elab.example/42")
    session_manager.experiment_changed.emit(experiment.to_dict())

    assert panel._eln_status_label.text() == "Published: https://elab.example/42"


def test_close_experiment_saves_findings_and_resets_panel(
    monitor_win_session, session_manager, monkeypatch
):
    """Clicking Close Experiment saves findings and reverts the panel to closed."""
    _stub_start_dialog(monkeypatch, "Hall bar A3", "jdoe")
    panel = monitor_win_session._session_info
    panel._start_close_btn.click()
    experiment_id = session_manager.current_experiment().experiment_id

    _stub_close_dialog(monkeypatch, "Saw a clean switching signal.")
    panel._start_close_btn.click()

    assert session_manager.current_experiment() is None
    assert panel._start_close_btn.text() == "Start Experiment…"
    assert panel._experiment_status_label.text() == "No experiment open"
    assert not panel._attended_checkbox.isVisible()
    assert "not configured" in panel._eln_status_label.text()
    closed = session_manager.store.load(experiment_id)
    assert closed.findings == "Saw a clean switching signal."
    assert closed.status == "closed"


def test_attendance_checkbox_toggle_calls_set_attended(
    monitor_win_session, session_manager, monkeypatch
):
    """Unchecking Attended flips the experiment's attendance flag."""
    _stub_start_dialog(monkeypatch, "Hall bar A3", "jdoe", attended=True)
    panel = monitor_win_session._session_info
    panel._start_close_btn.click()

    panel._attended_checkbox.setChecked(False)

    assert session_manager.current_experiment().attended is False


def test_add_user_dialog_autofills_id_from_name(qtbot):
    """AddUserDialog derives a roster-key slug from the typed name."""
    from i2as.gui.experiment_dialogs import AddUserDialog

    dialog = AddUserDialog()
    qtbot.addWidget(dialog)
    dialog._name_input.setText("Jane O'Doe")

    assert dialog._id_input.text() == "jane_o_doe"
    user = dialog.user()
    assert user.user_id == "jane_o_doe"
    assert user.name == "Jane O'Doe"


def _start_dialog_roster(tmp_path):
    from i2as.session.models import User
    from i2as.session.store import UserRoster

    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    return roster


def test_start_experiment_dialog_default_dirname_from_title(qtbot, tmp_path):
    """The folder name field auto-fills from the title until hand-edited."""
    from i2as.gui.experiment_dialogs import StartExperimentDialog

    dialog = StartExperimentDialog(_start_dialog_roster(tmp_path))
    qtbot.addWidget(dialog)

    dialog._title_input.setText("Hall bar A3 — SOT switching")

    assert dialog._dirname_input.text() == "hall_bar_a3_sot_switching"
    _title, _user_id, _attended, dirname = dialog.result_values()
    assert dirname == "hall_bar_a3_sot_switching"


def test_start_experiment_dialog_dirname_hand_edit_stops_autofill(qtbot, tmp_path):
    """Typing directly into the folder name field stops title-driven auto-fill."""
    from PyQt6.QtTest import QTest

    from i2as.gui.experiment_dialogs import StartExperimentDialog

    dialog = StartExperimentDialog(_start_dialog_roster(tmp_path))
    qtbot.addWidget(dialog)

    dialog._title_input.setText("Hall bar A3")
    # setText() alone never fires textEdited (only real keystrokes do, which
    # is what _on_dirname_edited listens for) — QTest.keyClicks simulates an
    # actual hand edit, unlike a plain setText() call.
    dialog._dirname_input.clear()
    QTest.keyClicks(dialog._dirname_input, "my_own_name")
    dialog._title_input.setText("Hall bar A3 — renamed")

    assert dialog._dirname_input.text() == "my_own_name"


def test_start_experiment_dialog_result_values_dirname_none_when_empty(qtbot, tmp_path):
    """An empty folder name field surfaces as None — the manager auto-derives one."""
    from i2as.gui.experiment_dialogs import StartExperimentDialog

    dialog = StartExperimentDialog(_start_dialog_roster(tmp_path))
    qtbot.addWidget(dialog)

    dialog._title_input.setText("Hall bar A3")
    dialog._dirname_input.setText("")

    _title, _user_id, _attended, dirname = dialog.result_values()
    assert dirname is None


def test_start_experiment_with_custom_dirname_creates_expected_directory(
    monitor_win_session, session_manager, monkeypatch
):
    """A folder name entered in the dialog becomes the experiment's directory on disk."""
    _stub_start_dialog(monkeypatch, "Hall bar A3", "jdoe", dirname="my_custom_folder")
    panel = monitor_win_session._session_info
    panel._start_close_btn.click()

    record = session_manager.current_experiment()
    assert record.experiment_id == "my_custom_folder"
    assert (session_manager.store.root / "my_custom_folder" / "experiment.json").is_file()


def test_start_experiment_with_invalid_dirname_shows_warning_and_stays_closed(
    monitor_win_session, session_manager, monkeypatch
):
    """A dirname the manager rejects surfaces as a warning, same as any other ValueError."""
    _stub_start_dialog(monkeypatch, "Hall bar A3", "jdoe", dirname="a/b")
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    panel = monitor_win_session._session_info
    panel._start_close_btn.click()

    assert warned
    assert session_manager.current_experiment() is None


# ── The envelope editor (Start Experiment dialog) ────────────────────────────


def _envelope_variables_dict(station):
    """The sim station's envelope variables in the DICT form a client sees.

    The editor is fed what crossed the client boundary — the JSON-safe
    rendering a ``StatusSnapshot`` carries — never the engine's typed
    ``EnvelopeVariable``, which no client ever holds.
    """
    import dataclasses

    return {
        name: dataclasses.asdict(variable)
        for name, variable in station.envelope_variables().items()
    }


def _envelope_dialog(qtbot, tmp_path, station):
    """A Start Experiment dialog carrying the sim station's envelope editor."""
    from i2as.gui.experiment_dialogs import StartExperimentDialog

    dialog = StartExperimentDialog(
        _start_dialog_roster(tmp_path),
        envelope_variables=_envelope_variables_dict(station),
    )
    qtbot.addWidget(dialog)
    dialog._title_input.setText("Hall bar A3")
    return dialog


def test_envelope_editor_is_prefilled_from_the_config_limits(qtbot, tmp_path, station):
    """Every enveloped quantity starts at the setup's own bounds, not blank."""
    dialog = _envelope_dialog(qtbot, tmp_path, station)
    editor = dialog._envelope_editor

    assert editor is not None
    magnet_min, magnet_max = editor._rows["magnet_z"]
    lo, hi = station.get_vi("magnet_z").limit_bounds("field_T")
    assert (float(magnet_min.text()), float(magnet_max.text())) == (lo, hi)
    assert "magnet_z" in dialog.envelope().bounds


def test_envelope_editor_absent_without_variables(qtbot, tmp_path):
    """A dialog built with no station (a unit test, say) carries no editor."""
    from i2as.gui.experiment_dialogs import StartExperimentDialog

    dialog = StartExperimentDialog(_start_dialog_roster(tmp_path))
    qtbot.addWidget(dialog)

    assert dialog._envelope_editor is None
    assert dialog.envelope() is None


def test_envelope_editor_narrowed_bounds_reach_the_envelope(qtbot, tmp_path, station):
    """A narrowed field becomes the experiment's bound on that VI."""
    dialog = _envelope_dialog(qtbot, tmp_path, station)
    dialog._envelope_editor._rows["magnet_z"][1].setText("2")

    bound = dialog.envelope().bounds["magnet_z"]
    assert bound.max_value == 2.0
    assert bound.violation(3.0) is not None


def test_envelope_editor_refuses_to_widen_the_setup_limit(qtbot, tmp_path, station):
    """An envelope narrows the config's limits; it may never widen them."""
    dialog = _envelope_dialog(qtbot, tmp_path, station)
    _lo, hi = station.get_vi("magnet_z").limit_bounds("field_T")
    dialog._envelope_editor._rows["magnet_z"][1].setText(f"{hi + 1:g}")

    assert "narrows the setup's limits" in dialog._envelope_editor.error()
    # isHidden(), not isVisible(): the dialog itself is never shown in tests.
    assert not dialog._envelope_editor._error_label.isHidden()
    assert not dialog._ok_button.isEnabled(), "OK must not accept a widened envelope"


def test_envelope_editor_refuses_a_non_numeric_bound(qtbot, tmp_path, station):
    """Junk in a field is named as such rather than silently dropped."""
    dialog = _envelope_dialog(qtbot, tmp_path, station)
    dialog._envelope_editor._rows["magnet_z"][0].setText("cold")

    assert "is not a number" in dialog._envelope_editor.error()
    assert not dialog._ok_button.isEnabled()


def test_envelope_editor_can_be_switched_off(qtbot, tmp_path, station):
    """Unticking the box means no envelope at all, and re-enables OK."""
    dialog = _envelope_dialog(qtbot, tmp_path, station)
    dialog._envelope_editor._rows["magnet_z"][0].setText("cold")
    dialog._envelope_editor._enabled_checkbox.setChecked(False)

    assert dialog.envelope() is None
    assert dialog._envelope_editor.error() == ""
    assert dialog._ok_button.isEnabled()


def test_envelope_editor_blank_fields_mean_unbounded(qtbot, tmp_path, station):
    """A VI with both fields cleared contributes no bound."""
    dialog = _envelope_dialog(qtbot, tmp_path, station)
    for edit in dialog._envelope_editor._rows["magnet_z"]:
        edit.setText("")

    assert "magnet_z" not in dialog.envelope().bounds


def test_started_experiment_installs_the_dialog_envelope(
    monitor_win_session, session_manager, orchestrator, monkeypatch
):
    """The envelope the dialog returns is installed on the Orchestrator."""
    from i2as.core.plan import EnvelopeBound, ExperimentEnvelope

    envelope = ExperimentEnvelope(
        bounds={"magnet_z": EnvelopeBound(min_value=-0.5, max_value=0.5)}
    )
    _stub_start_dialog(monkeypatch, "Hall bar A3", "jdoe", envelope=envelope)
    monitor_win_session._session_info._start_close_btn.click()

    assert session_manager.current_experiment() is not None
    settled(orchestrator)  # the envelope command has to land before the action
    blocked = []
    orchestrator.action_blocked.connect(blocked.append)
    orchestrator.submit_vi_action("magnet_z", "set_field", target_T=2.0)
    settled(orchestrator)
    assert blocked and "session envelope" in blocked[0]


def test_start_dialog_is_offered_the_setups_envelope_variables(
    monitor_win_session, session_manager, monkeypatch
):
    """The panel hands the dialog the setup's bounds, not an empty editor."""
    seen: list[dict] = []

    from i2as.gui import experiment_info_panel as sip

    def _capture(roster, parent=None, envelope_variables=None):
        seen.append(envelope_variables)
        return _FakeStartDialog(("Hall bar A3", "jdoe", True, None))

    monkeypatch.setattr(sip, "StartExperimentDialog", _capture)
    monitor_win_session._session_info._start_close_btn.click()

    assert seen and "magnet_z" in seen[0]
    # What the panel hands on is what the CLIENT's mirror answers — the
    # JSON-safe dict form of each envelope variable, not the typed record the
    # engine holds. This suite asserts the production shape because the panel
    # is now given the proxy the application gives it.
    assert seen[0]["magnet_z"]["param_name"] == "target_T"


def test_start_dialog_opens_with_a_populated_envelope_editor(
    monitor_win_session, session_manager, qtbot
):
    """The real dialog builds on the production dict form and is pre-filled.

    The regression this guards: the manager answers the mirror's JSON dict
    form, and an editor doing attribute access on it crashed the Start
    Experiment dialog for every setup with an enveloped quantity. This opens
    the real dialog on the sim station, through the manager the window holds.
    """
    from i2as.gui.experiment_dialogs import StartExperimentDialog

    dialog = StartExperimentDialog(
        session_manager.roster,
        envelope_variables=session_manager.envelope_variables(),
    )
    qtbot.addWidget(dialog)
    editor = dialog._envelope_editor

    assert editor is not None, "the sim setup declares an enveloped quantity"
    assert "magnet_z" in editor._rows
    lo, hi = monitor_win_session._station.get_vi("magnet_z").limit_bounds("field_T")
    magnet_min, magnet_max = editor._rows["magnet_z"]
    assert (float(magnet_min.text()), float(magnet_max.text())) == (lo, hi)
    # The unit label is derived from the setpoint parameter's own name, which
    # is a property the dict form cannot carry.
    assert editor._variables["magnet_z"].unit_suffix == "T"
    assert "magnet_z" in dialog.envelope().bounds


def test_envelope_editor_shows_an_envelope_already_in_force(qtbot, tmp_path, station):
    """set_bounds() replaces the setup defaults with the stored envelope."""
    from i2as.gui.experiment_dialogs import EnvelopeEditorWidget

    editor = EnvelopeEditorWidget(_envelope_variables_dict(station))
    qtbot.addWidget(editor)
    editor.set_bounds({"magnet_z": {"min_value": -1.0, "max_value": 1.5}})

    magnet_min, magnet_max = editor._rows["magnet_z"]
    assert (magnet_min.text(), magnet_max.text()) == ("-1", "1.5")
    bound = editor.envelope().bounds["magnet_z"]
    assert (bound.min_value, bound.max_value) == (-1.0, 1.5)

    editor.set_bounds(None)
    assert editor.envelope() is None
    assert float(magnet_max.text()) == station.get_vi("magnet_z").limit_bounds("field_T")[1]


# ── Setup tier: login and instrument info (User / Config menus) ───────────────

def test_current_user_label_defaults_not_logged_in(monitor_win):
    """With nobody logged in, the header shows the default state."""
    assert monitor_win._current_user_label.text() == "Not logged in"


def test_login_dialog_lists_roster_users(qtbot, tmp_path):
    """LoginDialog's user picker is pre-populated from the roster."""
    from i2as.gui.setup_dialogs import LoginDialog
    from i2as.session.models import User
    from i2as.session.store import UserRoster

    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))

    dialog = LoginDialog(roster)
    qtbot.addWidget(dialog)

    assert dialog._user_picker.has_users()
    assert dialog._user_picker.selected_user_id() == "jdoe"


def test_open_login_dialog_without_session_manager_shows_message(monitor_win, monkeypatch):
    """No ExperimentManager wired: the login action informs rather than crashing."""
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a))
    monitor_win._open_login_dialog()
    assert shown


def test_switch_user_saves_outgoing_and_loads_incoming_session(
    station, orchestrator, qtbot, tmp_path
):
    """_switch_user() persists the outgoing user's fields and loads the incoming one's."""
    from i2as.gui import app_settings as _app_settings
    from i2as.session.manager import ExperimentManager
    from i2as.session.models import User
    from i2as.session.store import ExperimentStore, UserRoster

    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    roster.add(User(user_id="asmith", name="A. Smith"))
    manager = ExperimentManager(
        store=ExperimentStore(tmp_path / "experiments"),
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
    )
    win = MonitorWindow(station, orchestrator, session_manager=manager)
    qtbot.addWidget(win)
    win.show()

    win._switch_user("jdoe")
    assert win._current_user_id == "jdoe"
    assert win._current_user_label.text() == "Logged in as J. Doe"
    assert _app_settings.current_user_id() == "jdoe"
    assert win._session_info._sample_name_input.text() == ""  # jdoe's file is fresh

    win._session_info._sample_name_input.setText("SampleB")
    win._switch_user("asmith")
    assert win._current_user_label.text() == "Logged in as A. Smith"
    assert win._session_info._sample_name_input.text() == ""  # asmith's file is fresh

    win._switch_user("jdoe")
    assert win._session_info._sample_name_input.text() == "SampleB"


# ── L6 session switching (Load Session…, Resume Session…) ─────────────────────

def test_load_session_dialog_lists_open_and_closed(station, orchestrator, qtbot, tmp_path):
    """Open experiments are selectable; closed ones are grayed out and disabled."""
    from i2as.gui.open_experiment_dialog import OpenExperimentDialog
    from i2as.session.manager import ExperimentManager
    from i2as.session.models import User
    from i2as.session.store import ExperimentStore, UserRoster

    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    manager = ExperimentManager(
        store=ExperimentStore(tmp_path / "experiments"),
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
    )
    manager.start_experiment(title="Closed One", user_id="jdoe", sample_info={})
    manager.close_experiment()
    manager.start_experiment(title="Open One", user_id="jdoe", sample_info={})

    dialog = OpenExperimentDialog(manager)
    qtbot.addWidget(dialog)

    assert dialog._list.count() == 2
    items = [dialog._list.item(i) for i in range(dialog._list.count())]
    closed_item = next(i for i in items if "(closed)" in i.text())
    open_item = next(i for i in items if "(closed)" not in i.text())

    assert not (closed_item.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert bool(open_item.flags() & Qt.ItemFlag.ItemIsEnabled)

    dialog._list.setCurrentItem(open_item)
    dialog.accept()
    assert dialog.selected_experiment_id() == open_item.data(Qt.ItemDataRole.UserRole)


def test_resume_session_dialog_lists_only_owner_sessions(qtbot, tmp_path):
    """list_sessions(user_id=...) filters the dialog to one user's own sessions."""
    from i2as.gui.session_dialogs import ResumeSessionDialog
    from i2as.session.store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    mine = store.create_session(name="My Cooldown", user_id="jdoe")
    store.create_session(name="Someone Else's", user_id="asmith")

    dialog = ResumeSessionDialog(store, "jdoe")
    qtbot.addWidget(dialog)

    assert dialog._list.count() == 1
    item = dialog._list.item(0)
    assert item.data(Qt.ItemDataRole.UserRole) == mine.session_id


def test_resume_session_dialog_select_and_accept(qtbot, tmp_path):
    """Selecting a listed session and accepting exposes it via selected_session_id()."""
    from i2as.gui.session_dialogs import ResumeSessionDialog
    from i2as.session.store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(name="My Cooldown", user_id="jdoe")

    dialog = ResumeSessionDialog(store, "jdoe")
    qtbot.addWidget(dialog)

    dialog._list.setCurrentItem(dialog._list.item(0))
    dialog.accept()
    assert dialog.selected_session_id() == session.session_id


def test_resume_session_dialog_create_new_session(qtbot, tmp_path):
    """The inline "New session…" name field + Create button creates and selects one."""
    from i2as.gui.session_dialogs import ResumeSessionDialog
    from i2as.session.store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    dialog = ResumeSessionDialog(store, "jdoe")
    qtbot.addWidget(dialog)

    assert dialog._list.count() == 0
    assert not dialog._create_btn.isEnabled()

    dialog._new_name_input.setText("Fresh Cooldown")
    assert dialog._create_btn.isEnabled()
    dialog._create_btn.click()

    assert store.list_sessions(user_id="jdoe")
    created_id = store.list_sessions(user_id="jdoe")[0]
    assert dialog.selected_session_id() == created_id
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_resume_session_dialog_no_selection_returns_none(qtbot, tmp_path):
    """selected_session_id() is None when nothing was ever selected."""
    from i2as.gui.session_dialogs import ResumeSessionDialog
    from i2as.session.store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    dialog = ResumeSessionDialog(store, "jdoe")
    qtbot.addWidget(dialog)

    assert dialog.selected_session_id() is None


def test_switch_session_saves_outgoing_and_loads_incoming(
    station, orchestrator, qtbot, tmp_path
):
    """_switch_experiment() persists the outgoing session's fields, loads the incoming
    session's own, and round-trips the queue through set_queue()."""
    from i2as.gui.form_autosave import QueueItemState, STATUS_PENDING
    from i2as.session.manager import ExperimentManager
    from i2as.session.models import EXPERIMENT_STATUS_OPEN, ExperimentRecord, User
    from i2as.session.store import ExperimentStore, UserRoster

    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    store = ExperimentStore(tmp_path / "experiments")
    manager = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
    )
    win = MonitorWindow(station, orchestrator, session_manager=manager)
    qtbot.addWidget(win)
    win.show()

    first = manager.start_experiment(title="Session A", user_id="jdoe", sample_info={})
    # A second, independently-open experiment (as if created earlier) —
    # written directly since start_experiment refuses a second concurrent one.
    second = ExperimentRecord(
        experiment_id="session_b", title="Session B", user_id="jdoe",
        status=EXPERIMENT_STATUS_OPEN,
    )
    store.save(second)

    win._session_info._sample_name_input.setText("SampleA")
    win._session.queue = [QueueItemState(procedure="Field Sweep", status=STATUS_PENDING)]

    win._switch_experiment(second.experiment_id)

    assert manager.current_experiment().experiment_id == second.experiment_id
    assert win._session_info._sample_name_input.text() == ""  # Session B's file is fresh

    saved_a = store.load(first.experiment_id)
    assert saved_a.queue and saved_a.queue[0]["procedure"] == "Field Sweep"
    saved_a_gui_state = session_store.load(store.gui_state_path(first.experiment_id))
    assert saved_a_gui_state.sample_name == "SampleA"

    win._session_info._sample_name_input.setText("SampleB")
    win._switch_experiment(first.experiment_id)

    assert manager.current_experiment().experiment_id == first.experiment_id
    assert win._session_info._sample_name_input.text() == "SampleA"


def test_switch_session_rejects_unknown_id_with_warning(
    station, orchestrator, qtbot, tmp_path, monkeypatch
):
    """An unknown/closed target surfaces QMessageBox.warning instead of crashing."""
    from i2as.session.manager import ExperimentManager
    from i2as.session.models import User
    from i2as.session.store import ExperimentStore, UserRoster

    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    manager = ExperimentManager(
        store=ExperimentStore(tmp_path / "experiments"),
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
    )
    win = MonitorWindow(station, orchestrator, session_manager=manager)
    qtbot.addWidget(win)
    win.show()

    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a))
    win._switch_experiment("does_not_exist")
    assert shown


def test_open_load_session_dialog_without_session_manager_shows_message(monitor_win, monkeypatch):
    """No ExperimentManager wired: Load Session informs rather than crashing."""
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a))
    monitor_win._open_load_session_dialog()
    assert shown


def test_open_resume_session_dialog_without_session_store_shows_message(monitor_win, monkeypatch):
    """No SessionStore wired: Resume Session informs rather than crashing."""
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a))
    monitor_win._open_resume_session_dialog()
    assert shown


def test_open_resume_session_dialog_sets_active_and_notes_status(
    station, orchestrator, session_manager, qtbot, tmp_path, monkeypatch
):
    """Picking a session persists it via SessionStore.set_active and notes the status bar."""
    from i2as.gui import monitor_window as mw
    from i2as.session.store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    created = store.create_session(name="Cooldown 3", user_id="jdoe")

    win = MonitorWindow(
        station, orchestrator, session_manager=session_manager, session_store=store
    )
    qtbot.addWidget(win)
    win.show()
    win._switch_user("jdoe")

    class _FakeResumeSessionDialog:
        DialogCode = QDialog.DialogCode

        def __init__(self, *a, **k):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selected_session_id(self):
            return created.session_id

    monkeypatch.setattr(mw, "ResumeSessionDialog", _FakeResumeSessionDialog)
    win._open_resume_session_dialog()

    assert store.get_active() == ("jdoe", created.session_id)
    assert "next launch" in win._status_bar.currentMessage()


def test_open_resume_session_dialog_resolves_logged_out_user_to_guest(
    station, orchestrator, session_manager, qtbot, tmp_path, monkeypatch
):
    """Nobody logged in: the dialog lists/activates sessions under the Guest identity."""
    from i2as.gui import monitor_window as mw
    from i2as.session.models import GUEST_USER_ID
    from i2as.session.store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    created = store.create_session(name="Walk-in Cooldown", user_id=GUEST_USER_ID)

    win = MonitorWindow(
        station, orchestrator, session_manager=session_manager, session_store=store
    )
    qtbot.addWidget(win)
    win.show()
    assert win._current_user_id is None

    seen_user_ids = []

    class _FakeResumeSessionDialog:
        DialogCode = QDialog.DialogCode

        def __init__(self, store, user_id, parent=None):
            seen_user_ids.append(user_id)

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selected_session_id(self):
            return created.session_id

    monkeypatch.setattr(mw, "ResumeSessionDialog", _FakeResumeSessionDialog)
    win._open_resume_session_dialog()

    assert seen_user_ids == [GUEST_USER_ID]
    assert store.get_active() == (GUEST_USER_ID, created.session_id)


# ── Data Dir: derived-but-editable from the open session ───────────────────────

def test_data_dir_auto_populates_on_start_and_restores_on_close(
    monitor_win_session, session_manager, monkeypatch
):
    """Data Dir is forced to the session's own folder on open, restored on close."""
    panel = monitor_win_session._session_info
    panel._data_dir_input.setText("D:/manual_choice")

    _stub_start_dialog(monkeypatch, "Hall bar A3", "jdoe")
    panel._start_close_btn.click()

    assert panel._data_dir_input.text() == str(session_manager.current_data_dir())

    _stub_close_dialog(monkeypatch, "")
    panel._start_close_btn.click()

    assert panel._data_dir_input.text() == "D:/manual_choice"


def test_data_dir_note_hidden_inside_session_visible_outside(
    monitor_win_session, session_manager, monkeypatch, tmp_path
):
    """The outside-session note toggles with whether Data Dir is inside data/."""
    _stub_start_dialog(monkeypatch, "Hall bar A3", "jdoe")
    panel = monitor_win_session._session_info
    panel._start_close_btn.click()
    assert not panel._data_dir_note.isVisible()

    outside_dir = tmp_path / "elsewhere"
    panel._data_dir_input.setText(str(outside_dir))
    assert panel._data_dir_note.isVisible()

    panel._data_dir_input.setText(str(session_manager.current_data_dir()))
    assert not panel._data_dir_note.isVisible()


# ── Data Dir hard containment (Enforcement) ─────────────────────────────────────

def test_is_data_dir_contained_true_without_open_experiment(monitor_win):
    """No experiment open: any path is considered contained (nothing to enforce)."""
    panel = monitor_win._session_info
    panel._data_dir_input.setText("D:/anywhere")
    assert panel.is_data_dir_contained()


def test_is_data_dir_contained_false_outside_open_experiment(
    monitor_win_session, session_manager, monkeypatch, tmp_path
):
    """An open experiment plus a Data Dir outside its folder is not contained."""
    _stub_start_dialog(monkeypatch, "Hall bar A3", "jdoe")
    panel = monitor_win_session._session_info
    panel._start_close_btn.click()

    panel._data_dir_input.setText(str(tmp_path / "elsewhere"))
    assert not panel.is_data_dir_contained()

    panel._data_dir_input.setText(str(session_manager.current_data_dir()))
    assert panel.is_data_dir_contained()


def test_browse_dir_rejects_selection_outside_open_experiment(
    monitor_win_session, session_manager, monkeypatch, tmp_path
):
    """_on_browse_dir() refuses a selection outside the open experiment's folder."""
    _stub_start_dialog(monkeypatch, "Hall bar A3", "jdoe")
    panel = monitor_win_session._session_info
    panel._start_close_btn.click()
    original_text = panel._data_dir_input.text()

    outside_dir = tmp_path / "elsewhere"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(outside_dir))
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))

    panel._on_browse_dir()

    assert warned
    assert panel._data_dir_input.text() == original_text


def test_browse_dir_accepts_selection_inside_open_experiment(
    monitor_win_session, session_manager, monkeypatch
):
    """_on_browse_dir() accepts a selection inside the open experiment's folder."""
    _stub_start_dialog(monkeypatch, "Hall bar A3", "jdoe")
    panel = monitor_win_session._session_info
    panel._start_close_btn.click()

    inside_dir = session_manager.current_data_dir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(inside_dir))

    panel._on_browse_dir()

    assert panel._data_dir_input.text() == str(inside_dir)


def test_get_data_dir_for_run_rejects_outside_path_with_warning(
    monitor_win_session, session_manager, monkeypatch, tmp_path
):
    """MonitorWindow.get_data_dir_for_run() refuses a run when Data Dir is outside."""
    _stub_start_dialog(monkeypatch, "Hall bar A3", "jdoe")
    win = monitor_win_session
    win._session_info._start_close_btn.click()
    win._session_info._data_dir_input.setText(str(tmp_path / "elsewhere"))

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))

    assert win.get_data_dir_for_run() is None
    assert warned


def test_get_data_dir_for_run_returns_path_when_no_experiment_open(monitor_win):
    """No experiment open: get_data_dir_for_run() behaves like get_data_dir()."""
    monitor_win._session_info._data_dir_input.setText("D:/anywhere")
    assert monitor_win.get_data_dir_for_run() == "D:/anywhere"


# ── store_health_changed → banner ───────────────────────────────────────────────

def test_store_health_changed_shows_and_clears_banner(monitor_win_session, session_manager):
    """A save failure shows a persistent banner error; recovery clears it."""
    win = monitor_win_session
    session_manager.store_health_changed.emit({"ok": False, "detail": "disk full"})
    assert win._banner.isVisible()
    assert "disk full" in win._banner._label.text()

    session_manager.store_health_changed.emit({"ok": True, "detail": ""})
    assert not win._banner.isVisible()


# ── _save_session targets the correct tier ──────────────────────────────────────

def test_save_session_targets_session_folder_when_open_else_per_user_file(
    monitor_win_session, session_manager, monkeypatch
):
    """gui_state.json inside the session folder when open, else the AppData file."""
    win = monitor_win_session
    win._session_info._sample_name_input.setText("NoSessionYet")
    win._save_session()
    per_user_path = _app_settings.autosave_file_path(win._current_user_id)
    assert session_store.load(per_user_path).sample_name == "NoSessionYet"

    _stub_start_dialog(monkeypatch, "Hall bar A3", "jdoe")
    win._session_info._start_close_btn.click()
    win._session_info._sample_name_input.setText("WithSessionOpen")
    win._save_session()

    gui_state_path = session_manager.current_gui_state_path()
    assert session_store.load(gui_state_path).sample_name == "WithSessionOpen"


def test_open_login_dialog_full_flow(station, orchestrator, qtbot, tmp_path, monkeypatch):
    """Confirming LoginDialog switches the current user."""
    from i2as.gui import monitor_window as mw
    from i2as.session.manager import ExperimentManager
    from i2as.session.models import User
    from i2as.session.store import ExperimentStore, UserRoster

    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    manager = ExperimentManager(
        store=ExperimentStore(tmp_path / "experiments"),
        roster=roster,
        orchestrator=orchestrator,
    )
    win = MonitorWindow(station, orchestrator, session_manager=manager)
    qtbot.addWidget(win)
    win.show()

    class _FakeLoginDialog:
        DialogCode = QDialog.DialogCode

        def __init__(self, *a, **k):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selected_user_id(self):
            return "jdoe"

    monkeypatch.setattr(mw, "LoginDialog", _FakeLoginDialog)
    win._open_login_dialog()

    assert win._current_user_id == "jdoe"
    assert win._current_user_label.text() == "Logged in as J. Doe"


def test_instrument_info_action_opens_dialog_with_config_metadata(
    station, orchestrator, qtbot, monkeypatch
):
    """Config menu's Instrument Info… reads read_instrument_metadata() for the active config."""
    from i2as.gui import monitor_window as mw

    win = MonitorWindow(
        station, orchestrator, active_config_path="i2as/configs/sim_cryostat"
    )
    qtbot.addWidget(win)
    win.show()

    captured = {}

    class _FakeInstrumentInfoDialog:
        def __init__(self, metadata, parent=None):
            captured["metadata"] = metadata

        def exec(self):
            return None

    monkeypatch.setattr(mw, "InstrumentInfoDialog", _FakeInstrumentInfoDialog)
    win._open_instrument_info()

    assert captured["metadata"]["magnet_z"]["role"] == "Z-axis magnet"


def test_instrument_info_dialog_handles_empty_metadata(qtbot):
    """An empty metadata dict shows the fallback message instead of an empty scroll area."""
    from i2as.gui.setup_dialogs import InstrumentInfoDialog

    dialog = InstrumentInfoDialog({})
    qtbot.addWidget(dialog)
    assert dialog.findChild(QScrollArea, "instrument_info_scroll") is None


def test_procedure_control_buttons_exist(procedure_win, qtbot):
    """Pause, Resume, Abort buttons are present."""
    from PyQt6.QtWidgets import QPushButton
    assert procedure_win.findChild(QPushButton, "pause_btn") is not None
    assert procedure_win.findChild(QPushButton, "resume_btn") is not None
    assert procedure_win.findChild(QPushButton, "abort_btn") is not None


def test_ack_button_visible_in_emergency(monitor_win, orchestrator):
    """Emergency acknowledge button appears when EMERGENCY state is emitted.

    Single home is the Monitor window — see
    test_ack_button_absent_from_procedure_window for the ProcedureWindow side.
    """
    orchestrator.state_changed.emit(OrchestratorState.EMERGENCY.value)
    assert monitor_win._ack_btn.isVisible()

    # Disappears on acknowledge
    orchestrator.state_changed.emit(OrchestratorState.IDLE.value)
    assert not monitor_win._ack_btn.isVisible()


def test_ack_button_visible_when_window_opened_after_emergency_already_active(
    station, orchestrator, qtbot
):
    """Opening MonitorWindow *after* EMERGENCY already fired must still show ACK.

    Regression test (moved from ProcedureWindow): a window can be
    (re)created well after an emergency has already put the Orchestrator
    into EMERGENCY. state_changed only reports future transitions, so
    without an explicit sync at construction time the button stayed hidden
    — the operator had no way to acknowledge from a freshly opened window.
    """
    # Forced through the tick helper: `orchestrator` is the client adapter,
    # so a bare assignment would set an attribute on the proxy and change
    # nothing at all — and the engine may be on its own thread. The snapshot
    # that follows is what carries the forced state into the client's mirror,
    # which is what a window opened afterwards reads.
    engine = engine_of(orchestrator)
    set_on_engine(orchestrator, "_state", OrchestratorState.EMERGENCY)
    on_engine(orchestrator, engine._emit_status_snapshot)

    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    win.show()

    assert win._ack_btn.isVisible()


def test_ack_button_absent_from_procedure_window(procedure_win):
    """ProcedureWindow no longer carries the Emergency-Acknowledge button.

    Single home is the Monitor window now.
    """
    from PyQt6.QtWidgets import QPushButton
    assert procedure_win.findChild(QPushButton, "ack_emergency_btn") is None


def test_hold_banner_shows_message_and_dismisses_on_clear(monitor_win, orchestrator, monkeypatch):
    """A plain hold condition (NOT emergency) populates the banner.

    Regression test: previously the ACK button appeared alone with no
    explanation on the banner above it (see _refresh_hold_banner()). Stubs
    the window's own read surface — the status mirror (held_vi_names /
    get_operational_status) — rather than reaching into the Station's private
    condition registry.
    """
    mirror = monitor_win._mirror
    held = {"magnet_z"}
    monkeypatch.setattr(mirror, "held_vi_names", lambda: frozenset(held))
    monkeypatch.setattr(
        mirror,
        "get_operational_status",
        lambda: {
            "conditions": [
                {
                    "key": "safety:coolant_low",
                    "origin": "safety",
                    "severity": "hold",
                    "kind": "coolant_low",
                    "message": "Safety flag 'coolant_low' is tripped",
                    "affected": sorted(held),
                    "since": 0.0,
                    "acknowledged": False,
                }
            ]
        },
    )

    monitor_win._in_emergency = False
    monitor_win._refresh_ack_controls()

    assert monitor_win._ack_btn.isVisible()
    assert monitor_win._banner.isVisible()
    assert "coolant_low" in monitor_win._banner._label.text()
    assert "magnet_z" in monitor_win._banner._label.text()

    # Clearing the hold condition dismisses the banner it owns.
    held.clear()
    monkeypatch.setattr(mirror, "held_vi_names", lambda: frozenset())
    monkeypatch.setattr(mirror, "get_operational_status", lambda: {"conditions": []})

    monitor_win._refresh_ack_controls()

    assert not monitor_win._ack_btn.isVisible()
    assert not monitor_win._banner.isVisible()


def test_progress_bar_updates(procedure_win, orchestrator):
    """Progress bar reflects procedure_progress signal."""
    orchestrator.procedure_progress.emit(0.42)
    assert procedure_win._progress_bar.value() == 42


def test_add_to_queue_appends_item(procedure_win, qtbot):
    """Add to Queue populates the queue list widget."""
    initial_count = procedure_win._queue_panel._queue_list.count()
    qtbot.mouseClick(
        procedure_win.findChild(QPushButton, "add_to_queue_btn"),
        __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.MouseButton.LeftButton,
    )
    # The panel is a VIEW of the queue: what redraws it is the QueueChanged
    # the engine broadcasts, so the round trip has to complete first.
    settled(procedure_win._orchestrator)
    assert procedure_win._queue_panel._queue_list.count() == initial_count + 1


def test_add_to_queue_captures_current_file_prefix(procedure_win, qtbot):
    """Each queue entry freezes the file-prefix field's value at add-time."""
    add_btn = procedure_win.findChild(QPushButton, "add_to_queue_btn")
    Qt = __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt

    procedure_win._params_panel._file_prefix_input.setText("run_a")
    qtbot.mouseClick(add_btn, Qt.MouseButton.LeftButton)

    procedure_win._params_panel._file_prefix_input.setText("run_b")
    qtbot.mouseClick(add_btn, Qt.MouseButton.LeftButton)
    settled(procedure_win._orchestrator)

    prefixes = [entry.spec.file_prefix for entry in procedure_win._queue_panel._queue]
    assert prefixes[-2:] == ["run_a", "run_b"]
    assert "run_a" in procedure_win._queue_panel._queue_list.item(len(prefixes) - 2).text()
    assert "run_b" in procedure_win._queue_panel._queue_list.item(len(prefixes) - 1).text()


def test_blank_file_prefix_omitted_from_queue_label(procedure_win, qtbot):
    """A blank prefix leaves the queue label as just the procedure name."""
    add_btn = procedure_win.findChild(QPushButton, "add_to_queue_btn")
    Qt = __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt

    procedure_win._params_panel._file_prefix_input.setText("")
    qtbot.mouseClick(add_btn, Qt.MouseButton.LeftButton)
    settled(procedure_win._orchestrator)

    entry = procedure_win._queue_panel._queue[-1]
    assert entry.spec.file_prefix == ""
    assert "[" not in procedure_win._queue_panel._queue_list.item(procedure_win._queue_panel._queue_list.count() - 1).text()
    queued_cls = procedure_win._queue_panel._classes[entry.spec.run_class]
    assert queued_cls.name in procedure_win._queue_panel._queue_list.item(procedure_win._queue_panel._queue_list.count() - 1).text()


def test_probe_first_queues_a_reduced_run_ahead_of_the_item(procedure_win, qtbot):
    """A probe of the selected run goes in FRONT of it, validated like any run."""
    from i2as.gui.queue_panel import DEFAULT_PROBE_SPEC

    panel = procedure_win._queue_panel
    procedure_win._on_add_to_queue()
    settled(procedure_win._orchestrator)
    queued = panel._host.snapshot()[-1]
    panel._select_spec(queued.spec_id)

    procedure_win.findChild(QPushButton, "queue_probe_btn").click()
    settled(procedure_win._orchestrator)

    order = panel._host.snapshot()
    assert [spec.probe_spec != {} for spec in order[-2:]] == [True, False], (
        "the probe comes first — that is the whole point of probing"
    )
    probe = order[-2]
    assert probe.run_class == queued.run_class and probe.params == queued.params
    assert probe.probe_spec == DEFAULT_PROBE_SPEC.to_json()

    row = panel._queue_list.item(panel._queue_list.count() - 2)
    assert "probe" in row.text(), "a probe is never science data and says so"
    assert "probe" in row.toolTip()
    assert panel._probe_label.isVisible()
    assert "probe" in panel._probe_label.text()


def test_a_probe_row_says_so_once(procedure_win):
    """The label names the probe once — a prefix that already says it is enough."""
    import dataclasses

    from i2as.gui.queue_panel import QueueEntry

    panel = procedure_win._queue_panel
    procedure_win._on_add_to_queue()
    settled(procedure_win._orchestrator)
    queued = panel._host.snapshot()[-1]
    reduction = {"n_points": 3, "averaging": 1, "max_wait_s": 5.0}
    prefixed = dataclasses.replace(
        queued, file_prefix="probe", probe_spec=reduction, spec_id="p1"
    )
    plain = dataclasses.replace(
        queued, file_prefix="", probe_spec=reduction, spec_id="p2"
    )

    assert panel._entry_summary(QueueEntry(spec=prefixed)).count("probe") == 1
    assert "(probe)" in panel._entry_summary(QueueEntry(spec=plain))


def test_probe_first_shows_the_estimate_and_the_findings(procedure_win, qtbot):
    """The caveats travel with the probe: inline, and on the row's tooltip."""
    panel = procedure_win._queue_panel
    procedure_win._on_add_to_queue()
    settled(procedure_win._orchestrator)
    panel._select_spec(panel._host.snapshot()[-1].spec_id)

    procedure_win.findChild(QPushButton, "queue_probe_btn").click()
    settled(procedure_win._orchestrator)

    note = panel._probe_label.text()
    probe = panel._host.snapshot()[-2]
    assert panel._spec_notes[probe.spec_id] == note
    # An estimate is never shown bare: it is qualified by what it assumed.
    if "≈" in note:
        assert "assuming" in note


def test_probe_first_does_nothing_without_a_selected_waiting_row(procedure_win):
    """Nothing selected, nothing queued — the action is per-row."""
    panel = procedure_win._queue_panel
    procedure_win._on_add_to_queue()
    settled(procedure_win._orchestrator)
    before = len(panel._host.snapshot())
    panel._queue_list.setCurrentRow(-1)

    procedure_win.findChild(QPushButton, "queue_probe_btn").click()
    settled(procedure_win._orchestrator)

    assert len(panel._host.snapshot()) == before
    assert not panel._probe_label.isVisible()




def test_run_now_passes_file_prefix_to_procedure_instance(procedure_win, qtbot):
    """Run Now builds a procedure carrying the current file-prefix field value."""
    procedure_win._params_panel._file_prefix_input.setText("live_run")
    proc = procedure_win._build_procedure_instance()
    assert proc is not None
    assert proc._file_prefix == "live_run"


def test_measurement_ready_updates_plot(procedure_win, orchestrator):
    """measurement_ready signal appends the datapoint to _datapoints."""
    datapoint = {"field_T": 0.5, "voltage_V": [1.23e-6] * 10}
    orchestrator.measurement_ready.emit(datapoint)

    assert len(procedure_win._datapoints) == 1
    assert abs(procedure_win._datapoints[0]["field_T"] - 0.5) < 1e-9


def test_procedure_window_on_the_imaging_station_ignores_the_image_column(qtbot):
    """Field Imaging on sim_imaging: the frame is never a plot axis, a datapoint
    carrying one is handled, and the form renders the stage roles.

    The image-block standard says the live plot ignores image columns and
    plots the VI's scalars; this pins that for the shipped camera VI.
    """
    host = build_host("i2as/configs/sim_imaging")
    try:
        station = host.station
        orchestrator = host.build_proxy()
        win = ProcedureWindow(
            station,
            orchestrator,
            get_sample_info=lambda: {"sample_name": "D", "sample_id": "1", "comments": ""},
            get_data_dir=lambda: "C:/CryoData",
        )
        qtbot.addWidget(win)
        win.show()
        win._params_panel.select_procedure_by_name("Field Imaging")

        y_keys = [win._plot1._y_selector.itemText(i) for i in range(win._plot1._y_selector.count())]
        assert "roi_mean" in y_keys and "roi_std" in y_keys
        assert "frame" not in y_keys, "an image block is never a plot axis"
        assert "roi_mean_array" not in y_keys

        assert win._params_panel.current_class().name == "Field Imaging"
        assert win._params_panel.current_selections().get("measurement_vi") == "camera"
        values = win._params_panel.collect_values()
        assert values is not None
        assert values["stage_x_vi"] == "stage_x" and values["stage_y_vi"] == "stage_y"
        assert values["saturation_field_T"] == -1.5

        import numpy as np

        datapoint = {
            "field_T": 0.5,
            "frame": np.zeros((128, 128)),
            "roi_mean": 2580.0,
            "roi_mean_error": 0.0,
            "roi_std": 30.0,
            "roi_mean_array": [2580.0],
        }
        orchestrator.measurement_ready.emit(datapoint)
        assert len(win._datapoints) == 1
        assert win._datapoints[0]["roi_mean"] == 2580.0
    finally:
        shutdown_host(host)


# ── Fixed 2x2 quadrant layout tests (GUI optimization redesign) ─────────────

def test_monitor_quadrant_splitters_not_collapsible(monitor_win):
    """The 3 quadrant splitters exist, resizable (not collapsible), correctly oriented."""
    assert monitor_win._main_splitter.orientation() == Qt.Orientation.Horizontal
    assert monitor_win._left_splitter.orientation() == Qt.Orientation.Vertical
    assert monitor_win._right_splitter.orientation() == Qt.Orientation.Vertical
    for splitter in (monitor_win._main_splitter, monitor_win._left_splitter, monitor_win._right_splitter):
        assert splitter.childrenCollapsible() is False
        assert splitter.count() == 2


def test_monitor_instrument_panels_exist_for_every_vi(monitor_win, station):
    """One InstrumentPanel exists per VI — system/level and measurement."""
    all_vis = set(station.get_vi_names())
    assert all_vis, "sim_cryostat should have VIs"
    panel_vi_names = {p._vi_name for p in monitor_win._panels}
    assert panel_vi_names == all_vis
    for panel in monitor_win._panels:
        assert isinstance(panel, InstrumentPanel)
    # Non-system cards are tagged with their role.
    assert monitor_win.findChild(QLabel, "dc_measurement_type_tag").text() == "Measurement"
    # System cards carry no tag.
    assert monitor_win.findChild(QLabel, "magnet_z_type_tag") is None


def test_monitor_fixed_quadrants_exist_with_expected_content(monitor_win):
    """Sample Info and the Ramps/Agents quadrant contain the expected widgets.

    The Log view moved to page 2 (the Logs page); the Other Devices section
    is retired (measurement VIs are instrument cards now), so the
    bottom-right quadrant is the ramp tracker over the agent panel.
    """
    sample_quadrant = monitor_win.findChild(QScrollArea, "session_info_scroll")
    assert sample_quadrant is not None
    assert sample_quadrant.widget().findChild(QLineEdit, "sample_name_input") is not None

    assert monitor_win.findChild(QWidget, "ramps_quadrant") is not None
    assert monitor_win.findChild(QWidget, "agents_quadrant") is not None
    assert monitor_win._log_panel is monitor_win._logs_page.findChild(QTextEdit, "log_panel")


def test_monitor_default_trend_panels_exist_and_gridded(monitor_win):
    """Two trend panels exist by default, each placed in the trends QGridLayout."""
    assert len(monitor_win._trends._trend_panels) == 2
    for panel in monitor_win._trends._trend_panels.values():
        assert monitor_win._trends._trends_grid.indexOf(panel) != -1


def test_monitor_has_no_view_menu(monitor_win):
    """Nothing in the fixed quadrant layout can be hidden/closed, so there is no View menu.

    The Session menu (state management) is a separate, always-present menu; the
    point preserved here is that the dock-era View menu stays gone.
    """
    menu_titles = {action.text() for action in monitor_win.menuBar().actions()}
    assert "View" not in menu_titles
    assert "Procedures" in menu_titles


def test_monitor_trends_grid_arranges_in_ceil_sqrt_grid(monitor_win):
    """Adding trend plots up to the cap of 4 arranges them in a 2x2 grid, not a stack."""
    monitor_win._trends._add_trend_panel()  # 3rd panel: ceil(sqrt(3)) = 2 columns
    monitor_win._trends._add_trend_panel()  # 4th panel: ceil(sqrt(4)) = 2 columns
    assert len(monitor_win._trends._trend_panels) == 4

    positions = {
        monitor_win._trends._trends_grid.getItemPosition(i)[:2]
        for i in range(monitor_win._trends._trends_grid.count())
    }
    assert positions == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_monitor_add_trend_plot_button_caps_at_four(monitor_win):
    """The Trends quadrant's Add button adds panels up to 4, then disables and stays inert."""
    assert len(monitor_win._trends._trend_panels) == 2
    assert monitor_win._trends._add_trend_btn.isEnabled()

    monitor_win._trends._add_trend_btn.click()
    assert len(monitor_win._trends._trend_panels) == 3
    monitor_win._trends._add_trend_btn.click()
    assert len(monitor_win._trends._trend_panels) == 4
    assert not monitor_win._trends._add_trend_btn.isEnabled()

    monitor_win._trends._add_trend_btn.click()
    assert len(monitor_win._trends._trend_panels) == 4


def test_monitor_trend_remove_button_drops_panel_never_below_one(monitor_win):
    """The panel's own remove button destroys the panel, stopping at a floor of 1."""
    assert len(monitor_win._trends._trend_panels) == 2
    first_id = next(iter(monitor_win._trends._trend_panels))

    monitor_win._trends._on_trend_remove_requested(first_id)
    assert len(monitor_win._trends._trend_panels) == 1
    assert first_id not in monitor_win._trends._trend_panels

    remaining_id = next(iter(monitor_win._trends._trend_panels))
    monitor_win._trends._on_trend_remove_requested(remaining_id)
    assert len(monitor_win._trends._trend_panels) == 1  # floor holds

    assert monitor_win._trends._add_trend_btn.isEnabled()


def test_monitor_page_switcher_swaps_pages(monitor_win):
    """The header page tab bar switches the central QStackedWidget's page."""
    assert monitor_win._page_stack.currentIndex() == 0
    assert monitor_win._page_stack.currentWidget() is monitor_win._main_splitter

    monitor_win._page_tab_bar.setCurrentIndex(1)
    assert monitor_win._page_stack.currentIndex() == 1
    assert monitor_win._page_stack.currentWidget() is monitor_win._logs_page

    monitor_win._page_tab_bar.setCurrentIndex(0)
    assert monitor_win._page_stack.currentIndex() == 0
    assert monitor_win._page_stack.currentWidget() is monitor_win._main_splitter


def test_monitor_states_updated_feeds_history_and_trend_combos(monitor_win, orchestrator):
    """states_updated records into MonitorHistory and populates the trend Y combos."""
    fake_state = {"magnet_z": {"get_field": 0.25, "magnet_current": 12.0}}
    orchestrator.states_updated.emit(fake_state)

    assert "magnet_z_get_field" in monitor_win._trends._history.keys()

    panels = monitor_win.findChildren(TrendPlotPanel)
    assert len(panels) == 2
    for panel in panels:
        combo = panel.findChild(QComboBox)
        assert combo is not None
        assert combo.count() > 0


def test_monitor_default_trend_key_hints_prefer_readings_over_settings(monitor_win, orchestrator):
    """The two default trend docks pick a temperature READING, not a setting/rate field.

    Regression pin: a plain substring search for "temperature" matches the
    VI-name prefix on fields like temperature_heater_output before it
    reaches the actual reading. Which specific VI wins alphabetically is not
    asserted here (real orchestrator ticks may have already populated
    history for other VIs too) — only that the FIELD chosen is the reading,
    not a setting/rate.
    """
    fake_state = {
        "temperature": {"heater_output": 0.0, "temperature": 4.2, "setpoint": 4.2},
    }
    orchestrator.states_updated.emit(fake_state)

    trend_0 = monitor_win._trends._trend_panels["trend_0"]
    assert trend_0.selected_key().endswith("_temperature")

    # The picker itself, over a key list this test owns: the VI-name prefix
    # must never win over the field name.
    picked = monitor_win._trends._pick_default_trend_key(
        "temperature",
        sorted(
            [
                "temperature_heater_output",
                "temperature_setpoint",
                "temperature_temperature",
                "magnet_z_magnet_field_T",
            ]
        ),
    )
    assert picked == "temperature_temperature"


def test_monitor_persistence_roundtrip_splitters_and_trends(
    station, orchestrator, qtbot, isolated_settings
):
    """Closing persists splitter proportions + trend selections; a fresh window restores them.

    Mirrors the existing geometry-persistence test: build a window, change
    state, close it (persisting via closeEvent to the isolated ini), then
    build a fresh window against the same settings and check the state came
    back. Unlike the old dock-based design, there is no explicit "Save
    layout" action — splitter state persists automatically alongside window
    geometry, the same way it already did for plain window size/position.
    """
    win1 = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win1)
    win1.show()

    third_id = win1._trends._add_trend_panel()
    third_panel = win1._trends._trend_panels[third_id]

    # Feed history AFTER the third panel exists so its refresh() (triggered by
    # this emit) populates its Y combo with a real key to select.
    fake_state = {"magnet_z": {"magnet_field_T": 0.5, "magnet_current": 10.0}}
    orchestrator.states_updated.emit(fake_state)
    third_panel.set_selected_key("magnet_z_magnet_field_T")
    third_panel.set_selected_window_s(21600.0)  # "6 h"

    assert len(win1._trends._trend_panels) == 3

    win1._main_splitter.setSizes([300, 900])
    win1.close()  # persists geometry + splitter state via closeEvent

    win2 = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win2)
    win2.show()

    assert len(win2._trends._trend_panels) == 3
    # Splitter proportions were restored, not left at the [600, 600] default.
    assert win2._main_splitter.sizes() != [600, 600]

    # Give the new window's (empty) history the same key so the persisted
    # selection, held pending, can actually be applied.
    orchestrator.states_updated.emit(fake_state)

    third_id_2 = list(win2._trends._trend_panels.keys())[2]
    third_panel_2 = win2._trends._trend_panels[third_id_2]
    assert third_panel_2.selected_key() == "magnet_z_magnet_field_T"
    assert third_panel_2.selected_window_s() == 21600.0


def test_monitor_default_layout_when_settings_empty(monitor_win, station):
    """With no saved splitter state (fresh isolated settings), the DEFAULT layout stands."""
    assert len(monitor_win._trends._trend_panels) == 2
    # One card per VI — system and measurement cards alike.
    assert len(monitor_win._panels) == len(station.get_vi_names())
    # setSizes([600, 600]) is a proportional hint, not exact pixels once shown
    # at the real window width — check the default is an even 50/50 split.
    left, right = monitor_win._main_splitter.sizes()
    assert abs(left - right) <= 2


def test_procedure_splitters_not_collapsible(procedure_win):
    """All ProcedureWindow splitters have children-collapsing disabled.

    Four splitters: the main horizontal split, the left/right vertical
    quadrant splits, and the queue-over-status vertical split inside the
    top-right quadrant.
    """
    splitters = procedure_win.findChildren(QSplitter)
    assert len(splitters) == 4, f"Expected 4 splitters, found {len(splitters)}"
    for sp in splitters:
        assert sp.childrenCollapsible() is False


def test_status_log_present_and_read_only(procedure_win):
    """The concise Status log widget exists in the top-right quadrant and is read-only."""
    status_log = procedure_win.findChild(QTextEdit, "status_log")
    assert status_log is not None, "status_log not found"
    assert status_log.isReadOnly()


def test_status_log_appends_status_messages(procedure_win, orchestrator):
    """A status_message signal appends a timestamped line to the Status log."""
    status_log = procedure_win.findChild(QTextEdit, "status_log")
    orchestrator.status_message.emit("Measuring point 3/11")
    assert "Measuring point 3/11" in status_log.toPlainText()


def test_pause_resume_abort_still_act_on_a_procedure(procedure_win, orchestrator, monkeypatch):
    """Regression: the same buttons still delegate normally for a plain procedure run."""
    pause_calls = []
    monkeypatch.setattr(orchestrator, "pause_procedure", lambda: pause_calls.append(True))
    procedure_win._on_pause_clicked()
    assert pause_calls == [True]


def test_procedure_quadrant_splitters_correctly_oriented(procedure_win):
    """main_splitter is horizontal; left/right splitters (params/plot1, queue/plot2) are vertical."""
    assert procedure_win._main_splitter.orientation() == Qt.Orientation.Horizontal
    assert procedure_win._left_splitter.orientation() == Qt.Orientation.Vertical
    assert procedure_win._right_splitter.orientation() == Qt.Orientation.Vertical
    assert procedure_win._left_splitter.widget(0).objectName() == "params_quadrant"
    assert procedure_win._left_splitter.widget(1) is procedure_win._plot1
    assert procedure_win._right_splitter.widget(0).objectName() == "queue_quadrant"
    assert procedure_win._right_splitter.widget(1) is procedure_win._plot2


def test_procedure_right_quadrant_has_the_queue_and_elab_tabs(procedure_win):
    """The top-right quadrant is a two-tab widget, Queue first, eLab second.

    The Queue tab still holds exactly what the quadrant always held (the
    queue-over-status splitter), so nothing that used to be found by name in
    that quadrant moved.
    """
    from PyQt6.QtWidgets import QTabWidget

    tabs = procedure_win.findChild(QTabWidget, "right_tabs")
    assert tabs is not None
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Queue", "eLab"]
    assert tabs.widget(0).findChild(QSplitter, "queue_status_splitter") is not None
    assert tabs.widget(0).findChild(QTextEdit, "status_log") is not None
    assert tabs.widget(1).objectName() == "analysis_panel"


def test_procedure_window_builds_with_no_elab_collaborators(procedure_win):
    """Built with no session layer, the eLab tab says so and offers nothing."""
    from i2as.gui.analysis_panel import NO_SESSION_TEXT

    panel = procedure_win.findChild(QWidget, "analysis_panel")
    assert panel is not None
    assert panel.findChild(QLabel, "analysis_status_label").text() == NO_SESSION_TEXT
    assert not panel.findChild(QPushButton, "analysis_publish_btn").isEnabled()


def test_procedure_run_finished_points_the_elab_tab_at_that_run(procedure_win):
    """A finished run reaches the eLab tab through the window's own slot."""
    seen = []
    procedure_win._analysis_panel.on_run_finished = seen.append
    procedure_win._orchestrator.run_finished.emit({"run_id": "run_042"})
    assert seen == [{"run_id": "run_042"}]


def test_procedure_param_scroll_has_no_height_cap(procedure_win):
    """The parameter scroll area fills its quadrant instead of being capped at a fixed height."""
    assert procedure_win._params_panel._param_scroll.maximumHeight() >= 16777215  # Qt's QWIDGETSIZE_MAX default (uncapped)


def test_monitor_central_widget_not_scroll_area(monitor_win):
    """The central widget is the content widget directly, holding the main quadrant splitter."""
    assert not isinstance(monitor_win.centralWidget(), QScrollArea)
    assert monitor_win.centralWidget().findChild(QSplitter, "main_splitter") is monitor_win._main_splitter


def test_logs_page_holds_the_log_panel(monitor_win):
    """Page 2 is the bare Logs page: the relocated LogPanel and nothing else.

    Ported from the deleted servicing-log-page suite — the LogPanel's home
    is the property that outlived the tables it used to sit beside.
    """
    from i2as.gui.log_panel import LogPanel

    assert monitor_win._logs_page.findChild(LogPanel, "log_panel") is monitor_win._log_panel
    assert monitor_win._page_stack.indexOf(monitor_win._logs_page) == 1


def test_log_handler_removed_on_close(station, orchestrator, qtbot):
    """Closing MonitorWindow detaches its log handler from the i2as logger."""
    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    handler = win._log_panel.handler
    i2as_logger = logging.getLogger("i2as")
    assert handler in i2as_logger.handlers

    win.close()
    assert handler not in i2as_logger.handlers


def test_log_panel_keeps_vi_bus_traffic_out_of_the_window(qtbot):
    """Sub-WARNING records from the per-VI loggers go to the file only.

    The panel recognises them by ``VI_LOGGER_PREFIX``; anything else under
    the root logger, and a VI warning, is shown.
    """
    from i2as.core.logging_config import LOGGER_ROOT, VI_LOGGER_PREFIX
    from i2as.gui.log_panel import LogPanel

    panel = LogPanel()
    qtbot.addWidget(panel)
    panel.show()

    def record(name: str, level: int, message: str) -> logging.LogRecord:
        return logging.LogRecord(name, level, __file__, 0, message, None, None)

    panel.handler.emit(record(f"{VI_LOGGER_PREFIX}magnet_z", logging.DEBUG, "polled-field"))
    assert "polled-field" not in panel.toPlainText()
    panel.handler.emit(record(f"{VI_LOGGER_PREFIX}magnet_z", logging.WARNING, "refused-set"))
    assert "refused-set" in panel.toPlainText()
    panel.handler.emit(record(f"{LOGGER_ROOT}.core.orchestrator", logging.DEBUG, "tick-42"))
    assert "tick-42" in panel.toPlainText()


def test_closing_window_persists_to_isolated_ini_not_real_scope(
    station, orchestrator, qtbot, isolated_settings
):
    """Closing a window writes geometry to the throwaway INI, never the real registry.

    Pins the Phase 3 test seam: because app_settings.get_settings is monkeypatched
    to the tmp INI, closeEvent's setValue lands there. Seeing the key in that file
    is proof the real QSettings scope was left untouched by the test run.
    """
    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    win.show()
    win.close()

    settings = QSettings(str(isolated_settings), QSettings.Format.IniFormat)
    settings.sync()
    assert settings.value("MonitorWindow/geometry") is not None
    assert isolated_settings.exists()


def test_closing_window_aborts_active_run(monitor_win, orchestrator):
    """Closing the app mid-run must safe the hardware, not just quit silently.

    Regression for a real-hardware finding (2026-07-22): a Keithley 6221 left
    armed/output-on because the window was closed while a run was still
    active, with nothing to send the abort/standby commands. closeEvent()
    must now call abort_procedure() whenever state is not already IDLE.
    """
    # Through _change_state, not a bare assignment: the window reads state
    # off its status mirror, which is fed by the engine's event stream.
    engine = engine_of(orchestrator)
    on_engine(
        orchestrator, lambda: engine._change_state(OrchestratorState.SWEEPING)
    )
    monitor_win.close()
    settled(orchestrator)
    assert orchestrator.state == OrchestratorState.IDLE.value


def test_closing_window_idle_does_not_call_abort(monitor_win, orchestrator, monkeypatch):
    """Closing from an already-IDLE state must not call abort_procedure() at all.

    abort_procedure() is documented safe to call with nothing running, but a
    normal close (nothing active) should not emit a spurious "Aborted by
    user" status / run_finished("aborted") signal.
    """
    calls = []
    monkeypatch.setattr(
        orchestrator, "abort_procedure", lambda: calls.append(True)
    )
    assert orchestrator.state == OrchestratorState.IDLE.value
    monitor_win.close()
    assert calls == []


# ── Phase 2: notification banner (replaces modal dialog storms) ────────────────

def test_banner_hidden_by_default(qtbot):
    """A fresh NotificationBanner is hidden until a message arrives."""
    banner = NotificationBanner()
    qtbot.addWidget(banner)
    assert banner.isHidden()
    assert banner.count == 0


def test_banner_error_shows_with_severity(qtbot):
    """show_message with 'error' makes the banner visible and sets the property."""
    banner = NotificationBanner()
    qtbot.addWidget(banner)
    banner.show_message("Magnet quench detected", "error")
    assert banner.isVisible()
    assert banner.property("severity") == "error"
    assert "Magnet quench detected" in banner._label.text()


def test_banner_warning_shows_with_severity(qtbot):
    """show_message with 'warning' sets the warning severity property."""
    banner = NotificationBanner()
    qtbot.addWidget(banner)
    banner.show_message("Action blocked while busy", "warning")
    assert banner.isVisible()
    assert banner.property("severity") == "warning"


def test_banner_dismiss_hides(qtbot):
    """Dismissing the banner hides it and resets the counter."""
    banner = NotificationBanner()
    qtbot.addWidget(banner)
    banner.show_message("Something happened", "warning")
    assert banner.isVisible()
    banner.dismiss()
    assert not banner.isVisible()
    assert banner.count == 0


def test_banner_repeat_increments_counter_no_stack(qtbot):
    """A repeated identical message bumps the counter instead of stacking."""
    banner = NotificationBanner()
    qtbot.addWidget(banner)
    banner.show_message("Blocked: magnet_z busy", "warning")
    assert banner.count == 1
    banner.show_message("Blocked: magnet_z busy", "warning")
    banner.show_message("Blocked: magnet_z busy", "warning")
    assert banner.count == 3
    assert "(3×)" in banner._label.text()
    # Still exactly one banner, still visible (nothing stacked).
    assert banner.isVisible()


def test_monitor_error_signal_drives_banner(monitor_win, orchestrator):
    """error_occurred routes to the MonitorWindow banner (no modal dialog)."""
    orchestrator.error_occurred.emit("Interlock tripped")
    assert monitor_win._banner.isVisible()
    assert monitor_win._banner.property("severity") == "error"
    assert "Interlock tripped" in monitor_win._banner._label.text()


def test_monitor_action_blocked_drives_banner(monitor_win, orchestrator):
    """action_blocked routes to the MonitorWindow banner as a warning."""
    orchestrator.action_blocked.emit("Cannot initiate: procedure running")
    assert monitor_win._banner.isVisible()
    assert monitor_win._banner.property("severity") == "warning"


def test_procedure_error_signal_drives_banner(procedure_win, orchestrator):
    """error_occurred routes to the ProcedureWindow banner as an error."""
    orchestrator.error_occurred.emit("Sweep failed")
    assert procedure_win._banner.isVisible()
    assert procedure_win._banner.property("severity") == "error"


def test_instrument_panel_has_no_action_blocked_handler(station, orchestrator, qtbot):
    """The per-panel modal warning handler was removed (banner replaces it)."""
    vi_name = "magnet_z"
    panel = InstrumentPanel(vi_name, orchestrator)
    qtbot.addWidget(panel)
    assert not hasattr(panel, "_on_action_blocked")


# ── Phase 2: state-aware status bar ────────────────────────────────────────────

def test_status_bar_level_flips_on_state(monitor_win, orchestrator):
    """Status bar 'level' property tracks the Orchestrator state category."""
    # Active state → "active"
    orchestrator.state_changed.emit(OrchestratorState.RAMPING.value)
    assert monitor_win._status_bar.property("level") == "active"

    # Emergency → "error"
    orchestrator.state_changed.emit(OrchestratorState.EMERGENCY.value)
    assert monitor_win._status_bar.property("level") == "error"

    # Back to idle → default (empty)
    orchestrator.state_changed.emit(OrchestratorState.IDLE.value)
    assert monitor_win._status_bar.property("level") == ""


# ── Phase 2: light theme smoke check ───────────────────────────────────────────

def test_stylesheet_has_no_dark_theme_hexes():
    """build_stylesheet() no longer contains the old dark-theme background hexes."""
    qss = build_stylesheet().lower()
    for dark_hex in ("#121212", "#252526", "#1e1e1e", "#2d2d30"):
        assert dark_hex not in qss, f"Leftover dark-theme colour {dark_hex} in stylesheet"


# ── Phase 2: effective colours after descendant repolish (regression) ──────────
# Property-only assertions would not have caught the bug these tests pin down:
# repolishing only the parent left child QLabels with their stale colour, so
# these assert the EFFECTIVE palette colour under the real stylesheet.

@pytest.fixture
def themed_app(qapp):
    """Apply the real application stylesheet for the test, then restore it.

    The plain test QApplication has no stylesheet, so palette-based colour
    assertions only mean anything with build_stylesheet() applied.
    """
    qapp.setStyleSheet(build_stylesheet())
    yield qapp
    qapp.setStyleSheet("")


def test_banner_error_effective_label_color(themed_app, qtbot):
    """After an error show_message, the label's palette colour is the error text."""
    banner = NotificationBanner()
    qtbot.addWidget(banner)
    banner.show_message("Interlock tripped", "error")
    assert banner._label.palette().windowText().color().name() == BANNER_ERROR_TEXT


def test_banner_warning_effective_label_color(themed_app, qtbot):
    """Switching a visible banner to warning re-colours the label (child repolish)."""
    banner = NotificationBanner()
    qtbot.addWidget(banner)
    banner.show_message("Boom", "error")
    banner.show_message("Blocked", "warning")
    assert banner._label.palette().windowText().color().name() == BANNER_WARNING_TEXT


def test_status_bar_label_effective_color_flips(themed_app, station, orchestrator, qtbot):
    """The status-bar label renders white in EMERGENCY and dark again on IDLE."""
    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    win.show()

    orchestrator.state_changed.emit(OrchestratorState.EMERGENCY.value)
    assert win._state_label.palette().windowText().color().name() == TEXT_ON_ACCENT

    orchestrator.state_changed.emit(OrchestratorState.IDLE.value)
    assert win._state_label.palette().windowText().color().name() == TEXT_PRIMARY


# ── Session persistence tests ──────────────────────────────────────────────────
# The autouse isolated_settings fixture redirects both QSettings and the JSON
# session file into tmp_path, so these never touch the user's real AppData.


def _sample_stub():
    return lambda: {"sample_name": "s", "sample_id": "id", "comments": ""}


def _data_dir_stub():
    return lambda: "C:/CryoData"


def test_monitor_window_has_user_menu(monitor_win):
    """The menu bar has a leftmost 'User' menu (Setup tier: login, form autosave)."""
    titles = [a.text() for a in monitor_win.menuBar().actions()]
    assert "User" in titles
    assert titles[0] == "User"


def test_monitor_restores_sample_fields_from_session(station, orchestrator, qtbot, tmp_path):
    """Sample Info fields are populated from a saved session on open."""
    session_store.save(
        session_store.FormAutosaveState(
            sample_name="Si_001", sample_id="S2024-01",
            comments="cooldown 2", data_dir="D:/runs",
        ),
        tmp_path / "last_session.json",
    )
    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    assert win._session_info._sample_name_input.text() == "Si_001"
    assert win._session_info._sample_id_input.text() == "S2024-01"
    assert win._session_info._comments_input.toPlainText() == "cooldown 2"
    assert win._session_info._data_dir_input.text() == "D:/runs"


def test_monitor_saves_session_on_close(monitor_win, tmp_path):
    """Closing the window persists the current Sample Info to the session file."""
    monitor_win._session_info._sample_name_input.setText("SampleZ")
    monitor_win._session_info._data_dir_input.setText("E:/data")
    monitor_win.close()
    loaded = session_store.load(tmp_path / "last_session.json")
    assert loaded.sample_name == "SampleZ"
    assert loaded.data_dir == "E:/data"


def test_new_session_clears_fields(monitor_win, monkeypatch):
    """New Session (confirmed) resets the Sample Info fields to defaults."""
    from i2as.core.paths import measurement_root

    monitor_win._session_info._sample_name_input.setText("ToClear")
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    monitor_win._on_new_session()
    assert monitor_win._session_info._sample_name_input.text() == ""
    assert monitor_win._session_info._data_dir_input.text() == str(measurement_root())


def test_procedure_window_restores_selection_and_params(station, orchestrator, qtbot):
    """A ProcedureWindow built with a session restores its selection and params."""
    info, ddir = _sample_stub(), _data_dir_stub()
    win = ProcedureWindow(station, orchestrator, info, ddir)
    qtbot.addWidget(win)
    proc_name = win._params_panel._current_procedure_name
    # Pick a plain text field (a QLineEdit) to type into.
    param_key = next(
        name for name, w in win._params_panel._param_inputs.items() if isinstance(w, QLineEdit)
    )
    win._params_panel._param_inputs[param_key].setText("42")

    state = session_store.FormAutosaveState()
    win.export_session_state(state)
    assert state.selected_procedure == proc_name
    # The cache is keyed by "{group.key}::{param}", so the typed value lands
    # under one composite key — assert it round-trips regardless of the prefix.
    assert "42" in state.procedure_params[proc_name].values()

    win2 = ProcedureWindow(station, orchestrator, info, ddir, initial_session=state)
    qtbot.addWidget(win2)
    assert win2._params_panel._proc_selector.currentText() == proc_name
    assert win2._params_panel._param_inputs[param_key].text() == "42"


def test_procedure_window_exports_and_restores_queue(station, orchestrator, qtbot):
    """A queued run round-trips through a session and is re-queued on restore."""
    info, ddir = _sample_stub(), _data_dir_stub()
    win = ProcedureWindow(station, orchestrator, info, ddir)
    qtbot.addWidget(win)
    win._on_add_to_queue()
    settled(orchestrator)
    assert win._queue_panel._queue_list.count() == 1, "default form params should be valid to queue"

    state = session_store.FormAutosaveState()
    win.export_session_state(state)
    assert len(state.queue) == 1

    win2 = ProcedureWindow(station, orchestrator, info, ddir, initial_session=state)
    qtbot.addWidget(win2)
    settled(orchestrator)
    assert win2._queue_panel._queue_list.count() == 1
    # The queue is data in the session layer, not procedures in the engine.
    assert len(win2._queue_panel._host.snapshot()) == 1


def test_procedure_window_skips_unknown_procedure_in_queue(station, orchestrator, qtbot):
    """A saved queue item for an unknown procedure is skipped, not fatal."""
    info, ddir = _sample_stub(), _data_dir_stub()
    state = session_store.FormAutosaveState(
        queue=[session_store.QueueItemState(procedure="NoSuchProcedure")]
    )
    win = ProcedureWindow(station, orchestrator, info, ddir, initial_session=state)
    qtbot.addWidget(win)
    assert win._queue_panel._queue_list.count() == 0


def test_run_queue_marks_running_then_done(station, orchestrator, qtbot, monkeypatch):
    """Running the queue marks items running, then done as each finishes."""
    info, ddir = _sample_stub(), _data_dir_stub()
    win = ProcedureWindow(station, orchestrator, info, ddir)
    qtbot.addWidget(win)
    win._on_add_to_queue()
    win._on_add_to_queue()
    settled(orchestrator)
    assert [e.status for e in win._queue_panel._queue] == ["pending", "pending"]

    # Stub the actual run: exercise only the GUI's per-item status logic.
    monkeypatch.setattr(orchestrator, "run_queue", lambda: None)
    win._queue_panel._on_run_queue()
    settled(orchestrator)
    assert win._queue_panel._queue[0].status == "running"
    assert win._queue_panel._queue_running is True

    orchestrator.procedure_finished.emit()
    settled(orchestrator)
    assert win._queue_panel._queue[0].status == "done"
    assert win._queue_panel._queue[1].status == "running"

    orchestrator.procedure_finished.emit()
    settled(orchestrator)
    assert win._queue_panel._queue[1].status == "done"
    assert win._queue_panel._queue_running is False


def test_abort_marks_running_item_failed(station, orchestrator, qtbot, monkeypatch):
    """Aborting a queued run marks that item failed and promotes the next."""
    info, ddir = _sample_stub(), _data_dir_stub()
    win = ProcedureWindow(station, orchestrator, info, ddir)
    qtbot.addWidget(win)
    win._on_add_to_queue()
    win._on_add_to_queue()
    settled(orchestrator)
    monkeypatch.setattr(orchestrator, "run_queue", lambda: None)
    monkeypatch.setattr(orchestrator, "abort_procedure", lambda: None)
    win._queue_panel._on_run_queue()
    settled(orchestrator)
    assert win._queue_panel._queue[0].status == "running"

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    win._on_abort()
    settled(orchestrator)
    assert win._queue_panel._queue[0].status == "failed"
    assert win._queue_panel._queue[1].status == "running"


def test_queue_holds_specs_not_procedures(station, orchestrator, qtbot):
    """Nothing waiting in the queue holds a live procedure object."""
    info, ddir = _sample_stub(), _data_dir_stub()
    win = ProcedureWindow(station, orchestrator, info, ddir)
    qtbot.addWidget(win)
    win._on_add_to_queue()
    settled(orchestrator)

    entry = win._queue_panel._queue[0]
    assert not hasattr(entry, "proc")
    assert entry.spec.run_class in win._queue_panel._classes
    assert entry.spec.actor.id == "operator"


def test_an_out_of_bounds_run_is_refused_when_it_is_queued(
    station, orchestrator, qtbot, monkeypatch
):
    """Validation happens at add time, with the findings on screen."""
    info, ddir = _sample_stub(), _data_dir_stub()
    win = ProcedureWindow(station, orchestrator, info, ddir)
    qtbot.addWidget(win)
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda parent, title, text, *a, **k: shown.append(text)
    )
    end_input = win.findChild(QLineEdit, "sweep_field_end_input")
    assert end_input is not None, "the field sweep renders a sweep-axis widget"
    end_input.setText("50")

    win._on_add_to_queue()

    assert win._queue_panel._queue_list.count() == 0
    assert shown and "magnet_z" in shown[0]
    assert win._queue_panel._host.snapshot() == ()


def test_the_panel_renders_a_run_queued_by_someone_else(station, orchestrator, qtbot):
    """The panel is a view: a QueueChanged it did not cause still updates it."""
    info, ddir = _sample_stub(), _data_dir_stub()
    win = ProcedureWindow(station, orchestrator, info, ddir)
    qtbot.addWidget(win)
    cls = win._params_panel.current_class()

    params, sample_info, data_dir, _prefix = win._collect_params()
    win._queue_panel._host.add(
        cls, params, sample_info=sample_info, data_directory=data_dir
    )
    settled(orchestrator)

    assert win._queue_panel._queue_list.count() == 1


def test_reordering_moves_the_spec_in_the_run_queue(station, orchestrator, qtbot):
    """Up/down reorder the queue itself, not a GUI-only copy of it."""
    info, ddir = _sample_stub(), _data_dir_stub()
    win = ProcedureWindow(station, orchestrator, info, ddir)
    qtbot.addWidget(win)
    win._params_panel._file_prefix_input.setText("first")
    win._on_add_to_queue()
    win._params_panel._file_prefix_input.setText("second")
    win._on_add_to_queue()
    settled(orchestrator)

    win._queue_panel._queue_list.setCurrentRow(1)
    win._queue_panel._queue_move_up()
    settled(orchestrator)

    assert [
        spec.file_prefix for spec in win._queue_panel._host.snapshot()
    ] == ["second", "first"]
    assert win._queue_panel._queue_list.currentRow() == 0


def test_queue_remove_drops_the_spec_from_the_run_queue(station, orchestrator, qtbot):
    """Removing a pending row removes the spec itself — there is one queue now."""
    info, ddir = _sample_stub(), _data_dir_stub()
    win = ProcedureWindow(station, orchestrator, info, ddir)
    qtbot.addWidget(win)
    win._on_add_to_queue()
    win._on_add_to_queue()
    settled(orchestrator)
    assert len(win._queue_panel._host.snapshot()) == 2
    removed = win._queue_panel._queue[0].spec.spec_id
    win._queue_panel._queue_list.setCurrentRow(0)
    win._queue_panel._queue_remove()
    settled(orchestrator)
    assert win._queue_panel._queue_list.count() == 1
    assert [s.spec_id for s in win._queue_panel._host.snapshot()] != [removed]
    assert len(win._queue_panel._host.snapshot()) == 1


# ── Startup config resolution + geometry tests ────────────────────────────────

def _catalog(tmp_path):
    return ConfigCatalog(_app_settings.shipped_config_dir(), tmp_path / "user")


def test_monitor_menu_bar_is_user_and_procedures_only(monitor_win):
    """The operator menu bar is exactly User + Procedures."""
    titles = [a.text() for a in monitor_win.menuBar().actions()]
    assert titles == ["User", "Procedures"]


def test_instrument_info_lives_in_the_user_menu(monitor_win):
    """The read-only instrument-info action now hangs off the User menu.

    It used to sit under a Config menu that no longer exists; it describes
    the rack the person in front of the app is using and writes nothing, so
    it belongs beside the login.
    """
    user_menu = next(
        a.menu() for a in monitor_win.menuBar().actions() if a.text() == "User"
    )
    labels = [a.text() for a in user_menu.actions()]
    assert "Instrument Info…" in labels
    # No active config path in a unit-built window: the handler answers from
    # an empty metadata mapping rather than reading a file.
    assert monitor_win._active_config_path is None


def test_startup_candidates_end_with_sim_and_dedup(monkeypatch):
    """The candidate chain always ends with sim_cryostat and has no duplicates."""
    from i2as import main as app_main

    monkeypatch.setattr(_app_settings, "config_active", lambda: None)
    candidates = app_main._startup_candidates()
    assert Path(candidates[-1]).name == "sim_cryostat"
    assert len(candidates) == len(set(candidates))


def test_startup_candidates_inserts_shipped_baseline_for_user_config(tmp_path, monkeypatch):
    """An active user config is followed by its shipped baseline, then sim."""
    from i2as import main as app_main

    catalog = _catalog(tmp_path)
    entry = catalog.fork_shipped("sim_cryostat", "sim_cryostat")
    monkeypatch.setattr(_app_settings, "user_config_dir", lambda: tmp_path / "user")
    monkeypatch.setattr(
        _app_settings, "config_active", lambda: (entry.name, entry.source)
    )
    candidates = app_main._startup_candidates()
    assert candidates[0] == str(entry.path)
    shipped_sim = str(_app_settings.shipped_config_dir() / "sim_cryostat")
    assert shipped_sim in candidates


def test_startup_warning_shown_in_banner(station, orchestrator, qtbot):
    """A startup fallback warning is surfaced in the notification banner."""
    win = MonitorWindow(
        station, orchestrator, startup_warning="active config was invalid"
    )
    qtbot.addWidget(win)
    assert not win._banner.isHidden()


def test_offscreen_saved_geometry_recenters(station, orchestrator, qtbot):
    """A saved geometry that lands off-screen is discarded for a centered one."""
    win = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win)
    win.move(-10000, -10000)
    assert not window_geometry.geometry_on_screen(win)
    _app_settings.get_settings().setValue("MonitorWindow/geometry", win.saveGeometry())

    win2 = MonitorWindow(station, orchestrator)
    qtbot.addWidget(win2)
    assert window_geometry.geometry_on_screen(win2)


# ── Monitoring toggle (Orchestrator start/stop monitoring from the header) ────


def test_monitoring_button_starts_and_stops_monitoring(monitor_win, orchestrator):
    """The header toggle starts monitoring, then stops it again in IDLE."""
    btn = monitor_win.findChild(QPushButton, "monitoring_btn")
    assert btn is not None
    # Launch state: monitoring off, button offers to start it.
    assert orchestrator.is_monitoring() is False
    assert not btn.isChecked()
    assert btn.text() == "Start Monitoring"

    btn.click()
    settled(orchestrator)
    assert orchestrator.is_monitoring() is True
    assert btn.isChecked()
    assert btn.text() == "Stop Monitoring"

    btn.click()  # IDLE, so the stop is allowed
    settled(orchestrator)
    assert orchestrator.is_monitoring() is False
    assert btn.text() == "Start Monitoring"


def test_monitoring_button_mirrors_orchestrator_state(monitor_win, orchestrator):
    """Starting monitoring on the Orchestrator directly updates the toggle."""
    btn = monitor_win.findChild(QPushButton, "monitoring_btn")
    orchestrator.start_monitoring()
    settled(orchestrator)
    assert btn.isChecked()
    assert btn.text() == "Stop Monitoring"


def test_monitoring_button_snaps_back_when_stop_refused(
    monitor_win, orchestrator, qtbot
):
    """A refused stop (non-IDLE state) re-syncs the button and warns via banner."""
    btn = monitor_win.findChild(QPushButton, "monitoring_btn")
    orchestrator.start_monitoring()
    qtbot.waitUntil(lambda: btn.isChecked(), timeout=2000)

    # Force a non-IDLE state so stop_monitoring() is refused — with the tick
    # held, since the very next tick would put the state machine back in IDLE
    # and the refusal would never happen.
    with ticks_paused(orchestrator):
        set_on_engine(orchestrator, "_state", OrchestratorState.RAMPING)
        btn.click()  # attempt to stop
        qtbot.waitUntil(lambda: monitor_win._banner.isVisible(), timeout=2000)
        assert orchestrator.is_monitoring() is True
        assert btn.isChecked(), "button must snap back to the confirmed state"
        set_on_engine(orchestrator, "_state", OrchestratorState.IDLE)


# ── Offline instruments (degraded build) ───────────────────────────────────────


def _degraded_monitor_setup(tmp_path, fail_times: int):
    """Station with one live VI and one offline VI, plus its Orchestrator.

    Reuses the L2 flaky-driver double: the offline VI's driver fails at build
    and succeeds (or keeps failing) on retry depending on ``fail_times``.
    """
    from tests.test_l2_station import _FlakyDriver, _write_degraded_config

    _FlakyDriver.fail_times = 1
    _FlakyDriver.attempts = 0
    station = build_station(
        _write_degraded_config(tmp_path, "tests.test_l2_station._FlakyDriver")
    )
    _FlakyDriver.fail_times = fail_times  # governs the retry attempts
    _FlakyDriver.attempts = 0
    orch = Orchestrator(station, tick_interval_ms=50)
    return station, orch


def test_offline_instrument_gets_fault_card_and_banner(tmp_path, qtbot):
    """An offline VI renders a control-free fault card and a startup banner."""
    station, orch = _degraded_monitor_setup(tmp_path, fail_times=0)
    try:
        win = MonitorWindow(station, orch)
        qtbot.addWidget(win)
        win.show()

        card = win.findChild(QGroupBox, "bad_vi_offline_card")
        assert card is not None
        assert card.property("status") == "offline"
        reason = win.findChild(QLabel, "bad_vi_offline_reason")
        assert "bad_drv" in reason.text()
        # No lifecycle toggle or control buttons on a fault card — only the
        # details icon and the one action that applies here, Connect.
        assert card.findChild(QPushButton, "bad_vi_offline_details_btn") is not None
        assert card.findChild(QPushButton, "bad_vi_connect_btn") is not None
        assert card.findChild(QPushButton, "bad_vi_lifecycle_btn") is None
        # Banner announces the degraded state.
        assert win._banner.isVisible()
        assert "bad_vi" in win._banner._label.text()
        assert "offline" in win._banner._label.text()
    finally:
        orch.shutdown()


def test_offline_reconnect_swaps_card_for_live_panel(tmp_path, qtbot):
    """A successful Try Reconnect replaces the fault card with a live panel."""
    station, orch = _degraded_monitor_setup(tmp_path, fail_times=0)
    try:
        win = MonitorWindow(station, orch)
        qtbot.addWidget(win)
        win.show()

        card = win.findChild(QGroupBox, "bad_vi_offline_card")
        card._open_details()
        reconnect_btn = win.findChild(QPushButton, "bad_vi_reconnect_btn")
        assert reconnect_btn is not None
        reconnect_btn.click()

        assert station.has_vi("bad_vi") is True
        assert win._offline_cards == {}
        assert any(p._vi_name == "bad_vi" for p in win._panels)
    finally:
        orch.shutdown()


def test_offline_reconnect_failure_reports_inline(tmp_path, qtbot):
    """A failed reconnect keeps the fault card and reports the fresh reason."""
    station, orch = _degraded_monitor_setup(tmp_path, fail_times=99)
    try:
        win = MonitorWindow(station, orch)
        qtbot.addWidget(win)
        win.show()

        win.findChild(QGroupBox, "bad_vi_offline_card")._open_details()
        win.findChild(QPushButton, "bad_vi_reconnect_btn").click()

        status = win.findChild(QLabel, "bad_vi_reconnect_status")
        assert "not reachable" in status.text()
        assert station.has_vi("bad_vi") is False
        assert win.findChild(QGroupBox, "bad_vi_offline_card") is not None
        detail_reason = win.findChild(QLabel, "bad_vi_offline_detail_reason")
        assert "bad_drv" in detail_reason.text()
    finally:
        orch.shutdown()


# ── Availability standard: tag-keyed offline wording ─────────────────────────
# The offline card's badge/note and the detail window's title/header/hint are
# selected from OfflineInstrument.tags (i2as.core.availability) via a
# tag-keyed mapping (offline_panel._wording_for()), covering every tag
# combination the offline registry can produce.


def test_offline_card_wording_for_connect_failed_only(tmp_path, qtbot):
    """A VI that never connected: [OFFLINE] badge, "not connected" note."""
    station, orch = _degraded_monitor_setup(tmp_path, fail_times=0)
    try:
        win = MonitorWindow(station, orch)
        qtbot.addWidget(win)
        win.show()

        assert station.get_offline_info("bad_vi").tags == frozenset({"connect_failed"})
        name_label = win.findChild(QLabel, "bad_vi_offline_name_label")
        assert "[OFFLINE]" in name_label.text()
        note = win.findChild(QLabel, "bad_vi_offline_note")
        assert "Not connected at startup" in note.text()

        win.findChild(QGroupBox, "bad_vi_offline_card")._open_details()
        header = win.findChild(QLabel, "bad_vi_offline_detail_header")
        assert "failed to connect at startup" in header.text()
        hint = win.findChild(QLabel, "bad_vi_offline_detail_hint")
        assert "troubleshoot check" in hint.text()
    finally:
        orch.shutdown()


def test_offline_card_wording_for_operator_only(qtbot):
    """An operator-disconnected VI: [DISCONNECTED] badge, "released" note/hint."""
    station, orch, win = _sim_monitor(qtbot)
    try:
        win.findChild(QGroupBox, "magnet_z_panel").findChild(
            QPushButton, "magnet_z_disconnect_btn"
        ).click()

        assert station.get_offline_info("magnet_z").tags == frozenset({"operator"})
        name_label = win.findChild(QLabel, "magnet_z_offline_name_label")
        assert "[DISCONNECTED]" in name_label.text()
        note = win.findChild(QLabel, "magnet_z_offline_note")
        assert "Released to its front panel" in note.text()

        win.findChild(QGroupBox, "magnet_z_offline_card")._open_details()
        header = win.findChild(QLabel, "magnet_z_offline_detail_header")
        assert "I2AS is not holding it" in header.text()
        hint = win.findChild(QLabel, "magnet_z_offline_detail_hint")
        assert "you released this instrument" in hint.text().lower()
        assert "front panel or" in hint.text().lower()
    finally:
        orch.shutdown()


def test_offline_card_wording_for_operator_and_connect_failed(tmp_path, qtbot):
    """The two-tag case: operator-released, then a reconnect fails on hardware.

    Bug fix the Availability standard's tag SET (rather than a bool) enables:
    the card and detail window must say BOTH things — that the operator did
    this AND that the last reconnect attempt failed on hardware — rather than
    the pre-tag-set behavior of only ever reporting "you released this
    instrument" over a hardware-failure reason.
    """
    from tests.test_l2_station import _SucceedsOnceDriver, _write_degraded_config

    _SucceedsOnceDriver.attempts = 0
    station = build_station(
        _write_degraded_config(
            tmp_path, "tests.test_l2_station._SucceedsOnceDriver"
        )
    )
    assert station.has_vi("bad_vi") is True
    orch = Orchestrator(station, tick_interval_ms=50)
    try:
        win = MonitorWindow(station, orch)
        qtbot.addWidget(win)
        win.show()

        # Operator disconnects the live VI...
        win.findChild(QGroupBox, "bad_vi_panel").findChild(
            QPushButton, "bad_vi_disconnect_btn"
        ).click()
        assert station.get_offline_info("bad_vi").tags == frozenset({"operator"})

        # ...then Connect is pressed, and the driver (already used its one
        # success) fails on hardware, so the offline card now carries BOTH
        # "operator" and "connect_failed".
        win.findChild(QGroupBox, "bad_vi_offline_card").findChild(
            QPushButton, "bad_vi_connect_btn"
        ).click()
        assert station.has_vi("bad_vi") is False
        assert station.get_offline_info("bad_vi").tags == frozenset(
            {"operator", "connect_failed"}
        )

        name_label = win.findChild(QLabel, "bad_vi_offline_name_label")
        assert "[DISCONNECTED]" in name_label.text()
        note = win.findChild(QLabel, "bad_vi_offline_note")
        # Says both: released AND the reconnect failed.
        assert "released" in note.text().lower()
        assert "reconnect attempt" in note.text().lower() and "failed" in note.text().lower()

        win.findChild(QGroupBox, "bad_vi_offline_card")._open_details()
        header = win.findChild(QLabel, "bad_vi_offline_detail_header")
        assert "you released it" in header.text().lower()
        assert "failed on hardware" in header.text().lower()
        hint = win.findChild(QLabel, "bad_vi_offline_detail_hint")
        assert "you released this instrument" in hint.text().lower()
        assert "failed on hardware" in hint.text().lower()
    finally:
        orch.shutdown()


# ── Connection-lifecycle standard: the Connect/Disconnect pair ───────────────
# See virtual_instruments/base.py's "Connection-lifecycle standard". The GUI
# half is one button on every card, and a card SWAP when it is pressed: a live
# InstrumentPanel becomes an OfflineInstrumentPanel and back again.


def _sim_monitor(qtbot):
    """A Monitor window over the full sim station, plus its Orchestrator."""
    station = build_station(CONFIG_PATH)
    orch = Orchestrator(station, tick_interval_ms=50)
    win = MonitorWindow(station, orch)
    qtbot.addWidget(win)
    win.show()
    return station, orch, win


def test_every_live_instrument_card_has_a_disconnect_button(qtbot):
    """The standard is only a standard if EVERY card carries it.

    System and measurement cards alike — the whole point of
    the connection-lifecycle standard is that no instrument category is a
    special case.
    """
    station, orch, win = _sim_monitor(qtbot)
    try:
        assert station.get_vi_names()
        for vi_name in station.get_vi_names():
            card = win.findChild(QGroupBox, f"{vi_name}_panel")
            assert card is not None, vi_name
            disconnect_btn = card.findChild(QPushButton, f"{vi_name}_disconnect_btn")
            assert disconnect_btn is not None, vi_name
            # Next to Initiate/Standby, not instead of it: two axes, two controls.
            assert card.findChild(QPushButton, f"{vi_name}_lifecycle_btn") is not None
    finally:
        orch.shutdown()


def test_disconnect_click_swaps_the_live_card_for_an_offline_one(qtbot):
    """Pressing Disconnect degrades the card exactly like a failed connect."""
    station, orch, win = _sim_monitor(qtbot)
    try:
        card = win.findChild(QGroupBox, "magnet_z_panel")
        card.findChild(QPushButton, "magnet_z_disconnect_btn").click()

        assert station.has_vi("magnet_z") is False
        assert "magnet_z" in win._offline_cards
        assert not any(p.vi_name == "magnet_z" for p in win._panels)
        offline_card = win.findChild(QGroupBox, "magnet_z_offline_card")
        assert offline_card is not None
        # Worded for an operator who did this on purpose, not for a fault.
        name_label = offline_card.findChild(QLabel, "magnet_z_offline_name_label")
        assert "DISCONNECTED" in name_label.text()
    finally:
        orch.shutdown()


def test_connect_click_swaps_the_offline_card_back(qtbot):
    """Connect on the card restores the live panel — the round trip closes."""
    station, orch, win = _sim_monitor(qtbot)
    try:
        win.findChild(QGroupBox, "magnet_z_panel").findChild(
            QPushButton, "magnet_z_disconnect_btn"
        ).click()
        offline_card = win.findChild(QGroupBox, "magnet_z_offline_card")
        offline_card.findChild(QPushButton, "magnet_z_connect_btn").click()

        assert station.has_vi("magnet_z") is True
        assert win._offline_cards == {}
        assert any(p.vi_name == "magnet_z" for p in win._panels)
        live_card = win.findChild(QGroupBox, "magnet_z_panel")
        assert live_card is not None
        assert live_card.findChild(QPushButton, "magnet_z_disconnect_btn") is not None
    finally:
        orch.shutdown()


def test_disconnect_is_blocked_for_a_vi_the_running_run_claims(qtbot):
    """The refusal reaches the operator through the banner, not a dialog.

    Disconnect is gated by the claim, not by the state (see
    ``Orchestrator.disconnect_instrument()``), so the card that gets the
    refusal is the one the active run owns.
    """
    station, orch, win = _sim_monitor(qtbot)
    try:
        orch._state = OrchestratorState.MEASURING
        orch._procedure = object()  # any non-None value marks a run active
        orch._active_claims = {"magnet_z"}
        win.findChild(QGroupBox, "magnet_z_panel").findChild(
            QPushButton, "magnet_z_disconnect_btn"
        ).click()

        assert station.has_vi("magnet_z") is True
        assert win._banner.isVisible()
        assert "magnet_z" in win._banner._label.text()
    finally:
        orch._state = OrchestratorState.IDLE
        orch._procedure = None
        orch._active_claims = None
        orch.shutdown()


def test_disconnect_mid_run_swaps_the_card_for_a_vi_the_run_does_not_claim(qtbot):
    """An unclaimed instrument stays the operator's to release, run or no run.

    The GUI half of the claim gate: the card swaps to its offline form
    exactly as it does at IDLE, and the run carries on.
    """
    station, orch, win = _sim_monitor(qtbot)
    try:
        orch._state = OrchestratorState.MEASURING
        orch._procedure = object()
        orch._active_claims = {"magnet_z"}
        win.findChild(QGroupBox, "temperature_panel").findChild(
            QPushButton, "temperature_disconnect_btn"
        ).click()

        assert station.has_vi("temperature") is False
        assert win.findChild(QGroupBox, "temperature_offline_card") is not None
        assert station.has_vi("magnet_z") is True
    finally:
        orch._state = OrchestratorState.IDLE
        orch._procedure = None
        orch._active_claims = None
        orch.shutdown()


# ── Widget-lifetime standards: window liveness + card retirement ─────────────
# See i2as/gui/widget_lifecycle.py. A shown window whose creator kept no
# reference used to be destroyed by whichever generational garbage-collection
# pass happened to reach it — including one triggered by an allocation inside
# that same window's paintEvent, which destroyed the paint device mid-paint and
# segfaulted the process on a half-freed pyqtgraph scene.


def _build_and_forget_a_monitor_window() -> None:
    """Show a MonitorWindow while deliberately keeping no reference to it.

    The station, the Orchestrator and the window are all locals here, so once
    this returns the only thing that can keep the still-shown window alive is
    the window-liveness hold it takes on itself. The Orchestrator is shut down
    before returning so no tick outlives these locals either.
    """
    station = build_station(CONFIG_PATH)
    orch = Orchestrator(station, tick_interval_ms=50)
    win = MonitorWindow(station, orch)
    win.show()
    orch.shutdown()


def test_a_shown_window_survives_a_collection_that_frees_its_creator(qtbot):
    """A shown window outlives a GC pass that finds no other reference to it."""
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
    plot = win.findChild(QWidget, "trend_plot_trend_0")
    assert plot is not None

    # The paint that used to run into a freed AxisItem and segfault.
    QApplication.processEvents()

    assert not sip.isdeleted(plot)
    assert plot.isVisible()
    win.close()
    assert not any(w is win for w in widget_lifecycle.held_windows())


def test_closing_a_window_releases_its_liveness_hold(qtbot):
    """The hold is not a leak: closing the window drops it again."""
    station, orch, win = _sim_monitor(qtbot)
    try:
        assert any(w is win for w in widget_lifecycle.held_windows())

        win.close()

        assert not any(w is win for w in widget_lifecycle.held_windows())
    finally:
        orch.shutdown()


def test_repeated_card_swaps_retire_every_replaced_card(qtbot):
    """Twenty Disconnect/Connect round trips leave nothing painting behind.

    The card-retirement standard end to end: every replaced card is hidden and
    out of the instrument grid before its deferred delete, the grid never grows
    a slot, and every retired card is really destroyed once the deferred
    deletes are delivered.
    """
    station, orch, win = _sim_monitor(qtbot)
    try:
        grid = win._instruments_grid
        slots = grid.count()
        retired = []

        for _ in range(20):
            live = next(p for p in win._panels if p.vi_name == "magnet_z")
            live.findChild(QPushButton, "magnet_z_disconnect_btn").click()
            QApplication.processEvents()
            assert live.isHidden()
            assert grid.indexOf(live) == -1
            assert grid.count() == slots
            retired.append(live)

            offline = win._offline_cards["magnet_z"]
            offline.findChild(QPushButton, "magnet_z_connect_btn").click()
            QApplication.processEvents()
            assert offline.isHidden()
            assert grid.indexOf(offline) == -1
            assert grid.count() == slots
            retired.append(offline)

        assert station.has_vi("magnet_z") is True
        assert win._offline_cards == {}
        assert sum(p.vi_name == "magnet_z" for p in win._panels) == 1

        # deleteLater() is deferred, and pytest-qt never runs an event loop
        # that would deliver it, so ask for those events explicitly: every
        # retired card must then be gone, not merely hidden.
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert all(sip.isdeleted(card) for card in retired)
    finally:
        orch.shutdown()
