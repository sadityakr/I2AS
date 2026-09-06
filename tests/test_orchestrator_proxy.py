"""OrchestratorProxy and InstrumentHost tests — the seam, in inline mode.

Three properties matter here. The proxy is TRANSPARENT: a client written
against the Orchestrator keeps working when handed one, because every signal
is re-exposed under its own name and every command method takes the same
arguments. It is COMPLETE: one method per ``CommandName``, each answered by
exactly one ``Verdict``. And it READS NOTHING: every query is answered from
the status mirror the host primed.
"""

import pytest

from i2as.core import events as ev
from i2as.core.instrument_host import (
    MODES,
    THREAD_ENV_VAR,
    InstrumentHost,
    resolve_mode,
)
from i2as.core.orchestrator_proxy import OrchestratorProxy
from i2as.core.station import build_station
from i2as.core.status_mirror import StatusMirror

CONFIG_PATH = "i2as/configs/sim_cryostat"


@pytest.fixture
def host(qtbot):
    """A started inline host over the sim station, torn down after the test."""
    instrument_host = InstrumentHost(
        lambda: build_station(CONFIG_PATH),
        orchestrator_options={"tick_interval_ms": 50},
    )
    instrument_host.start()
    yield instrument_host
    instrument_host.shutdown()


@pytest.fixture
def proxy(host):
    """The client adapter that host hands out."""
    return host.build_proxy()


# ── The host ──────────────────────────────────────────────────────────────────

def test_inline_mode_builds_the_stack_on_the_caller_thread(host):
    """`inline` is today's construction, with a name and a single home."""
    assert host.mode == "inline"
    assert host.station.has_vi("magnet_z")
    assert host.orchestrator.state == "IDLE"


def test_start_is_idempotent(host):
    """A caller unsure whether the host is up may simply start it again."""
    engine = host.orchestrator
    host.start()
    assert host.orchestrator is engine


def test_inline_mode_has_no_bridge_to_cross(host):
    """No thread, no crossing: every call is direct and every read is local.

    The threaded half of both modes lives in ``tests/test_instrument_thread.py``.
    """
    assert set(MODES) == {"inline", "threaded"}
    assert host.bridge is None
    assert host.thread_object is None
    assert host.build_proxy().bridge is None


@pytest.mark.parametrize(
    ("configured", "override", "expected"),
    [
        (False, None, "inline"),
        (True, None, "threaded"),
        (None, None, "threaded"),  # silence inherits the standard
        (False, "1", "threaded"),
        (True, "0", "inline"),
        (True, "off", "inline"),
        (False, "TRUE", "threaded"),
        (True, "nonsense", "threaded"),  # unreadable override is ignored
    ],
)
def test_the_mode_is_the_config_unless_the_environment_overrides_it(
    configured, override, expected, monkeypatch
):
    """The instrument-thread flag: config first, environment last.

    ``threaded`` is the default, so only an explicit refusal — in the config
    or in the environment — takes a launch back to the temporary inline mode.
    """
    monkeypatch.delenv(THREAD_ENV_VAR, raising=False)
    if override is not None:
        monkeypatch.setenv(THREAD_ENV_VAR, override)
    assert resolve_mode(configured) == expected


def test_an_unknown_mode_is_refused_at_construction():
    """A typo'd mode fails where it was written, not at start()."""
    with pytest.raises(ValueError, match="unknown InstrumentHost mode"):
        InstrumentHost(lambda: build_station(CONFIG_PATH), mode="parallel")


def test_reading_a_host_before_start_says_so():
    """Every accessor refuses clearly rather than returning None."""
    instrument_host = InstrumentHost(lambda: build_station(CONFIG_PATH))
    for read in ("station", "orchestrator"):
        with pytest.raises(RuntimeError, match="start"):
            getattr(instrument_host, read)
    with pytest.raises(RuntimeError, match="start"):
        instrument_host.client_state()


def test_client_state_is_what_the_mirror_is_primed_with(host):
    """The three priming values are captured on the engine's own thread."""
    station_info, snapshot, operational = host.client_state()
    assert isinstance(station_info, ev.StationInfo)
    assert isinstance(snapshot, ev.StatusSnapshot)
    assert isinstance(operational, dict)
    assert {i.name for i in station_info.instruments} == set(
        host.station.availabilities()
    )


# ── The proxy ─────────────────────────────────────────────────────────────────

def test_every_command_is_answered_by_exactly_one_verdict(proxy, qtbot):
    """One command in, one verdict out, carrying the request id it answers."""
    verdicts: list[ev.Verdict] = []
    proxy.verdict.connect(verdicts.append)

    request_id = proxy.start_monitoring()
    assert [v.request_id for v in verdicts] == [request_id]
    assert verdicts[0].code is ev.VerdictCode.OK
    assert verdicts[0].command is ev.CommandName.START_MONITORING
    assert verdicts[0].actor is ev.OPERATOR or verdicts[0].actor == ev.OPERATOR


def test_a_refusal_is_a_verdict_too(proxy):
    """The engine's refusals reach the client as codes, not only as prose."""
    verdicts: list[ev.Verdict] = []
    proxy.verdict.connect(verdicts.append)
    proxy.acknowledge()  # nothing held, not in EMERGENCY
    assert verdicts[-1].code is not ev.VerdictCode.OK
    assert "acknowledge" in verdicts[-1].reason.lower()


def test_reads_are_answered_from_the_mirror_not_the_engine(proxy, host):
    """A query never calls in: it reads the last snapshot the engine sent."""
    assert proxy.state == "IDLE"
    assert proxy.is_monitoring() is False
    proxy.start_monitoring()
    assert proxy.is_monitoring() is True
    assert set(proxy.availabilities()) == set(host.station.availabilities())
    assert proxy.instrument_info("magnet_z") is not None
    assert proxy.station_info() is proxy.status.station_info()


def test_events_reach_the_union_signal_and_the_typed_one(proxy, host, qtbot):
    """Every event is re-emitted twice: unfiltered, and split by type."""
    everything: list[object] = []
    snapshots: list[object] = []
    proxy.event.connect(everything.append)
    proxy.status_snapshot_event.connect(snapshots.append)

    host.orchestrator._tick()
    assert snapshots, "a tick publishes a status snapshot"
    assert snapshots[-1] in everything
    assert isinstance(snapshots[-1], ev.StatusSnapshot)


def test_the_engines_own_signals_are_re_exposed_unchanged(proxy, host, qtbot):
    """A widget connected to the proxy sees what the engine emitted."""
    seen: list[str] = []
    proxy.state_changed.connect(seen.append)
    states: list[dict] = []
    proxy.states_updated.connect(states.append)

    host.orchestrator._change_state(host.orchestrator._state.__class__("ERROR"))
    host.orchestrator.states_updated.emit({"magnet_z": {"magnet_field_T": 1.0}})
    assert seen == ["ERROR"]
    assert states == [{"magnet_z": {"magnet_field_T": 1.0}}]


def test_a_vi_action_carries_its_parameters_as_flat_scalars(proxy, host):
    """submit_vi_action's kwargs become the command's args, unchanged."""
    submitted: list[ev.Command] = []
    original = host.orchestrator.submit
    host.orchestrator.submit = lambda command: (
        submitted.append(command) or original(command)
    )
    proxy.submit_vi_action("magnet_z", "set_field", target_T=0.25)
    assert submitted[-1].name is ev.CommandName.SUBMIT_VI_ACTION
    assert submitted[-1].args == {
        "vi_name": "magnet_z",
        "method_name": "set_field",
        "target_T": 0.25,
    }


def test_the_envelope_crosses_as_its_dict_form(proxy, host):
    """A typed envelope is rendered to JSON, so the command stays a payload."""
    from i2as.core.plan import EnvelopeBound, ExperimentEnvelope

    envelope = ExperimentEnvelope(
        bounds={"magnet_z": EnvelopeBound(min_value=-1.0, max_value=1.0)}
    )
    proxy.set_experiment_envelope(envelope)
    assert "magnet_z" in host.orchestrator.envelope_variables()

    proxy.set_experiment_envelope(None)
    assert host.orchestrator._session_envelope is None


def test_a_run_object_is_forwarded_with_the_same_actor(proxy, host):
    """The four object-carrying commands cannot be JSON, so they forward.

    They still name the operator, so accountability is identical; what they
    lack is the correlated verdict, and that goes when the queue holds specs
    and the engine builds every run itself.
    """
    calls: list[dict] = []
    host.orchestrator.run_procedure = lambda procedure, actor=None: calls.append(
        {"procedure": procedure, "actor": actor}
    )
    marker = object()
    request_id = proxy.run_procedure(marker)
    assert calls == [{"procedure": marker, "actor": ev.OPERATOR}]
    assert isinstance(request_id, str) and request_id


def test_a_proxy_built_by_hand_mirrors_the_engine_it_is_given(host):
    """The constructor's fallback builds and primes its own mirror."""
    standalone = OrchestratorProxy(host.orchestrator)
    assert isinstance(standalone.status, StatusMirror)
    assert standalone.state == host.orchestrator.state


def test_status_mirror_of_returns_the_proxys_own(proxy):
    """A widget handed a proxy but no mirror reads the proxy's, not a new one."""
    assert StatusMirror.of(proxy) is proxy.status


# ── The GUI against the proxy ────────────────────────────────────────────────

def test_the_whole_gui_builds_against_the_proxy(host, proxy, qtbot):
    """Both windows construct and drive through the proxy alone.

    The transparency proof the inline mode exists for: the windows are handed
    the proxy where they used to be handed the engine, and nothing else about
    them changes.
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

    # A click still reaches the engine, and its effect still comes back.
    monitor._monitoring_btn.click()
    assert host.orchestrator.is_monitoring() is True
    assert monitor._monitoring_btn.text() == "Stop Monitoring"
