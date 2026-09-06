import logging
from pathlib import Path
from typing import ClassVar

import pytest
from i2as.core.conditions import Condition
from i2as.core.decorators import monitored
from i2as.core.exceptions import I2ASCommunicationError
from i2as.core.plan import Target
from i2as.core.station import Station, build_station
from i2as.virtual_instruments.base import BaseVirtualInstrument


class _AddressCapturingDriver:
    """Test double for build_station(): records the resource string it was built with.

    Honours the driver contract's connection half (``get_idn`` / ``close``)
    so a station built from it passes the build's identity check.
    """

    last_resource: str | None = None

    def __init__(self, resource_string: str) -> None:
        type(self).last_resource = resource_string
        self.closed = False

    def get_state(self) -> dict:
        return {}

    def get_idn(self) -> str:
        return "I2AS,ADDRESS-CAPTURING-STUB,0,0"

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def sim_station():
    """Fixture to build a Station from the sim_cryostat configuration."""
    config_path = Path(__file__).parent.parent / "i2as" / "configs" / "sim_cryostat"
    return build_station(str(config_path))


def test_build_station_passes_address_to_driver(tmp_path):
    """build_station() must pass the YAML 'address:' value to the driver constructor.

    Regression test: build_station() previously read a nonexistent
    'resource_string' key and silently defaulted every real driver to
    'SIM', so real hardware would never receive its actual VISA address.
    """
    (tmp_path / "devices.yaml").write_text(
        "real_drivers:\n"
        "  probe:\n"
        "    class: tests.test_l2_station._AddressCapturingDriver\n"
        '    address: "GPIB0::19::INSTR"\n'
        "virtual_instruments: {}\n"
    )
    (tmp_path / "monitor.yaml").write_text(
        "monitor:\n  tick_interval_ms: 1000\n  max_vi_errors: 3\n"
    )

    build_station(str(tmp_path))

    assert _AddressCapturingDriver.last_resource == "GPIB0::19::INSTR"


def test_read_panels_config_well_formed(tmp_path):
    """read_panels_config returns per-VI control allowlists from monitor.yaml."""
    from i2as.core.config import read_panels_config

    (tmp_path / "monitor.yaml").write_text(
        "monitor:\n  tick_interval_ms: 1000\n"
        "panels:\n"
        "  temperature:\n"
        "    controls: [set_temperature]\n"
        "  magnet_z:\n"
        "    controls: [set_field, set_ramp_rate]\n"
    )
    assert read_panels_config(str(tmp_path)) == {
        "temperature": ["set_temperature"],
        "magnet_z": ["set_field", "set_ramp_rate"],
    }


def test_read_panels_config_tolerates_absent_or_malformed(tmp_path):
    """Absent block, malformed entries, or a missing file yield {} / skip, never raise."""
    from i2as.core.config import read_panels_config

    # No monitor.yaml at all.
    assert read_panels_config(str(tmp_path / "nowhere")) == {}
    # No panels block.
    (tmp_path / "monitor.yaml").write_text("monitor:\n  tick_interval_ms: 1000\n")
    assert read_panels_config(str(tmp_path)) == {}
    # Malformed entries are skipped; the well-formed one survives.
    (tmp_path / "monitor.yaml").write_text(
        "monitor:\n  tick_interval_ms: 1000\n"
        "panels:\n"
        "  bad_scalar: just_a_string\n"
        "  bad_no_controls:\n"
        "    other_key: 1\n"
        "  good_vi:\n"
        "    controls: [set_x]\n"
    )
    assert read_panels_config(str(tmp_path)) == {"good_vi": ["set_x"]}


def test_read_tick_interval_ms_reads_the_configured_value(tmp_path):
    from i2as.core.config import read_tick_interval_ms

    (tmp_path / "monitor.yaml").write_text("monitor:\n  tick_interval_ms: 1000\n")
    assert read_tick_interval_ms(str(tmp_path)) == 1000


def test_read_tick_interval_ms_defaults_when_absent_or_missing(tmp_path):
    from i2as.core.config import read_tick_interval_ms

    # No monitor.yaml at all.
    assert read_tick_interval_ms(str(tmp_path / "nowhere")) == 3000
    # monitor.yaml present but omits the key.
    (tmp_path / "monitor.yaml").write_text("monitor:\n  max_vi_errors: 3\n")
    assert read_tick_interval_ms(str(tmp_path)) == 3000


def test_build_station_success(sim_station: Station):
    """build_station('i2as/configs/sim_cryostat') works without errors."""
    assert sim_station is not None
    # Check that expected VIs are registered
    vi_names = sim_station.get_vi_names()
    expected = ["magnet_z", "temperature", "dc_measurement"]
    for name in expected:
        assert name in vi_names


def test_station_getattr(sim_station: Station):
    """station.magnet_z returns correct VI instance."""
    magnet_z = sim_station.magnet_z
    assert magnet_z.vi_name == "magnet_z"
    assert magnet_z.__class__.__name__ == "SuperconductingMagnetVI"

    # Check another one to be sure
    temp_vti = sim_station.temperature
    assert temp_vti.vi_name == "temperature"
    assert temp_vti.__class__.__name__ == (
        "Lakeshore335SampleTemperatureControllerVI"
    )


def _registry_stub():
    """A minimal VI-like stub for registry-only tests (never polled here)."""

    class _Stub:
        vi_name = ""

        def get_state(self) -> dict:
            return {}

    return _Stub()


def test_station_get_vi_returns_named_instance(sim_station: Station):
    """get_vi(name) returns the same instance as attribute access."""
    assert sim_station.get_vi("magnet_z") is sim_station.magnet_z
    with pytest.raises(KeyError):
        sim_station.get_vi("no_such_vi")


def test_station_measurement_vi_names_registration_order(sim_station: Station):
    """measurement_vi_names() returns only measurement VIs, in registration order.

    sim_cryostat registers dc_measurement (vi_type=measurement); the
    system/level VIs are excluded.
    """
    assert sim_station.measurement_vi_names() == ["dc_measurement"]


def test_station_measurement_vi_names_empty_when_none_registered():
    """A station with no measurement VIs reports an empty list."""
    station = Station()
    station.register_vi("magnet_z", _registry_stub(), "system")
    assert station.measurement_vi_names() == []


def test_get_ramp_status_covers_system_rampables(sim_station: Station):
    """get_ramp_status() returns a target/rate/ramp_status entry for every system
    VI that can ramp, excludes measurement VIs, and reports idle VIs as IDLE with
    a None target."""
    ramps = sim_station.get_ramp_status()

    assert "magnet_z" in ramps
    assert "temperature" in ramps
    for entry in ramps.values():
        assert {"target", "rate", "ramp_status"} <= set(entry)

    # Nothing commanded yet: idle, no target.
    assert ramps["magnet_z"]["ramp_status"] == "IDLE"
    assert ramps["magnet_z"]["target"] is None

    # Measurement VIs are not ramp targets and must not appear.
    assert "dc_measurement" not in ramps


def test_get_ramp_status_reports_active_target(sim_station: Station):
    """After a system VI starts ramping, its live target shows up in the aggregate."""
    sim_station.process_system_targets({"magnet_z": Target(1.0)})
    ramps = sim_station.get_ramp_status()
    assert ramps["magnet_z"]["ramp_status"] == "RAMPING"
    assert ramps["magnet_z"]["target"] == pytest.approx(1.0)


def test_get_state_format(sim_station: Station):
    """get_state() returns dict with all VI states."""
    state = sim_station.get_state()
    
    # Assert top-level keys are the VI names
    for name in sim_station.get_vi_names():
        assert name in state
    
    # Assert a specific VI state contains its @monitored variables
    magnet_state = state["magnet_z"]
    assert "magnet_current" in magnet_state
    assert "magnet_field_T" in magnet_state
    assert "magnet_status" in magnet_state


def test_get_state_error_handling(sim_station: Station):
    """Stale values with _stale: True on communication error, _disconnected after max."""
    # Run once to get good values in the cache
    sim_station.get_state()
    
    # Force the simulated magnet driver to simulate an error
    magnet_z = sim_station.magnet_z
    magnet_z._driver._simulate_error = True
    
    # 1st error -> should return stale data with _stale: True
    state = sim_station.get_state()
    assert state["magnet_z"]["_stale"] is True
    assert "_disconnected" not in state["magnet_z"]
    assert sim_station._error_counts["magnet_z"] == 1
    
    # 2nd error
    sim_station.get_state()
    assert sim_station._error_counts["magnet_z"] == 2
    
    # 3rd error -> should now also have _disconnected: True
    state = sim_station.get_state()
    assert state["magnet_z"]["_stale"] is True
    assert state["magnet_z"].get("_disconnected") is True
    assert sim_station._error_counts["magnet_z"] == 3


def test_last_state_flat_coerces_bool_to_float_unlike_monitor_history(sim_station: Station):
    """Pin the bool-persistence asymmetry documented on MonitorHistory's docstring.

    ``last_state_flat()`` has no ``bool`` guard: since ``bool`` is a subclass
    of ``int``, a VI's boolean monitored field passes
    ``isinstance(value, (int, float))`` and is coerced to a float, so it IS
    written to the trend-history tiers. This is the opposite direction from
    ``gui.monitor_history.MonitorHistory.record()``, which explicitly excludes
    bools from the live in-RAM history (see that test module's matching pin,
    and MonitorHistory's class docstring for the full asymmetry writeup).
    Neither method should change to "fix" this — ``last_state_flat()`` also
    feeds HDF5 sweep columns.
    """

    class _BoolVI:
        vi_name = "bool_vi"

        def get_state(self):
            return {"flag": True}

    sim_station.register_vi("bool_vi", _BoolVI(), "system")
    sim_station.get_state()  # populate _last_known_state
    flat = sim_station.last_state_flat()

    key = "bool_vi_flag"
    assert key in flat
    assert flat[key] == 1.0
    assert isinstance(flat[key], float)


def test_process_system_targets_dispatch(sim_station: Station):
    """process_system_targets dispatches to correct VIs only."""
    targets = {
        "magnet_z": Target(1.0),
        "temperature": Target(150.0)
    }

    sim_station.process_system_targets(targets)

    # Verify that the ramps have started
    assert sim_station.magnet_z.ramp_status() == "RAMPING"
    assert sim_station.temperature.ramp_status() == "RAMPING"

    # process_system_targets should raise if we pass a non-system VI
    with pytest.raises(ValueError):
        sim_station.process_system_targets({"dc_measurement": Target(10.0)})


def test_check_ramps(sim_station: Station):
    """check_ramps() returns False while ramping, True after done."""
    # Ensure initially True (all are IDLE)
    assert sim_station.check_ramps() is True
    
    # Start a ramp
    sim_station.process_system_targets({"magnet_z": Target(1.0)})

    # While ramping, should return False
    assert sim_station.check_ramps() is False
    
    # Force the ramp to complete
    # For magnet_z, it uses a generator and advances the actual value. We can force the target.
    # We will simulate enough ticks until the magnet reaches the setpoint.
    # The sim driver has a ramp_rate (5.0 A/min = 0.083 A/s).
    # Target 1.0 T = 10 A. By setting the driver's current to the target, we make it reach HOLD immediately.
    magnet_driver = sim_station.magnet_z._driver
    magnet_driver._current = 10.0
    magnet_driver._setpoint = 10.0
    magnet_driver._status = "HOLD"
    
    # The VI's generator needs to be ticked to recognize it reached the target
    sim_station.check_ramps()
    
    # Now it should be True
    assert sim_station.check_ramps() is True


def test_check_ramps_reports_only_the_named_vis(sim_station: Station):
    """A ramp outside the requested scope does not make check_ramps() report False.

    The ramp-scope standard: a caller waits for the ramps IT started. An
    empty scope therefore means "nothing to wait for" even while other
    hardware is moving — the case for a procedure that commands no targets.
    """
    sim_station.process_system_targets({"magnet_z": Target(1.0)})

    assert sim_station.check_ramps() is False                      # whole station
    assert sim_station.check_ramps({"magnet_z"}) is False           # in scope
    assert sim_station.check_ramps({"temperature"}) is True     # out of scope
    assert sim_station.check_ramps(set()) is True                   # owns nothing


def test_check_ramps_advances_out_of_scope_ramps(sim_station: Station):
    """Every ramp advances regardless of scope — scope only narrows the verdict.

    check_ramps() is the sole driver of ramp generators in the tick loop, so
    narrowing the *advance* alongside the *report* would freeze an unwatched
    ramp mid-flight instead of merely not waiting for it.
    """
    sim_station.process_system_targets({"magnet_z": Target(1.0)})
    start = sim_station.magnet_z.magnet_field_T()

    for _ in range(5):
        assert sim_station.check_ramps(set()) is True

    assert sim_station.magnet_z.magnet_field_T() > start


def test_get_ramp_status_carries_the_full_introspection_snapshot(sim_station: Station):
    """get_ramp_status() aggregates every RampableVI introspection hook per VI.

    One snapshot serves both consumers (the operational-status record and the
    ramp tracker), so every documented key must be present for every system
    VI — a missing key would silently become a ``None`` column downstream.
    """
    status = sim_station.get_ramp_status()
    assert status, "expected at least one rampable system VI"
    keys = {"value", "setpoint", "target", "rate", "ramp_status", "phase", "no_motion_phases"}
    for vi_name, entry in status.items():
        assert keys <= set(entry), f"{vi_name} missing {keys - set(entry)}"
        assert entry["ramp_status"] == "IDLE"
        assert entry["setpoint"] is None


def test_get_ramp_status_reports_next_and_end_setpoint_during_a_ramp(
    sim_station: Station,
):
    """Mid-ramp the snapshot distinguishes the commanded setpoint from the target."""
    sim_station.process_system_targets({"magnet_z": Target(5.0)})
    entry = sim_station.get_ramp_status()["magnet_z"]

    assert entry["ramp_status"] == "RAMPING"
    assert entry["target"] == pytest.approx(5.0)
    assert entry["setpoint"] is not None
    # sim_cryostat's magnet ramps in current-dependent segments, so the first
    # commanded setpoint is a boundary short of the 5 T end setpoint.
    assert entry["setpoint"] < entry["target"]

    sim_station.stop_ramps({"magnet_z"})
    assert sim_station.get_ramp_status()["magnet_z"]["setpoint"] is None


def test_check_safety_uses_snapshot_without_polling(sim_station: Station):
    """check_safety(state) must not poll hardware (review finding H1).

    The old implementation called get_state() internally, doubling GPIB
    traffic every tick and double-counting the error counters.
    """
    state = sim_station.get_state()
    magnet_driver = sim_station.magnet_z._driver

    # Count driver polls around check_safety via a wrapper.
    call_count = {"n": 0}
    original = magnet_driver.get_status

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    magnet_driver.get_status = counting
    try:
        sim_station.check_safety(state)
        sim_station.check_safety()  # cached-state variant
    finally:
        magnet_driver.get_status = original
    assert call_count["n"] == 0


def test_check_safety_flags_magnet_quench(sim_station: Station):
    """A magnet reporting QUENCH must trip the 'quench' safety flag."""
    sim_station.get_state()
    assert sim_station.check_safety().get("quench", False) is False

    sim_station.magnet_z._driver._simulate_quench = True
    state = sim_station.get_state()
    safety = sim_station.check_safety(state)
    assert safety["quench"] is True


# ---------------------------------------------------------------------------
# Runtime fault registry
# ---------------------------------------------------------------------------

def test_fault_recorded_on_stale_then_disconnected(sim_station: Station):
    """A comm-error streak records a FaultRecord: 'stale' then 'disconnected'."""
    sim_station.get_state()
    assert sim_station.vi_faults() == {}

    sim_station.magnet_z._driver._simulate_error = True
    sim_station.get_state()
    faults = sim_station.vi_faults()
    assert faults["magnet_z"].kind == "stale"
    assert faults["magnet_z"].acknowledged is False
    since_first = faults["magnet_z"].since

    sim_station.get_state()
    sim_station.get_state()  # 3rd consecutive error -> disconnected
    faults = sim_station.vi_faults()
    assert faults["magnet_z"].kind == "disconnected"
    # Escalating the SAME incident preserves 'since'.
    assert faults["magnet_z"].since == since_first


def test_fault_auto_clears_on_successful_poll(sim_station: Station):
    """A successful poll removes the fault record entirely — ack or not."""
    sim_station.magnet_z._driver._simulate_error = True
    sim_station.get_state()
    assert "magnet_z" in sim_station.vi_faults()
    sim_station.acknowledge_fault("magnet_z")
    assert sim_station.vi_faults()["magnet_z"].acknowledged is True

    sim_station.magnet_z._driver._simulate_error = False
    sim_station.get_state()
    assert "magnet_z" not in sim_station.vi_faults()


def test_acknowledge_fault(sim_station: Station):
    """acknowledge_fault() flags an existing record; no-ops on a healthy VI."""
    assert sim_station.acknowledge_fault("magnet_z") is False

    sim_station.magnet_z._driver._simulate_error = True
    sim_station.get_state()
    assert sim_station.acknowledge_fault("magnet_z") is True
    assert sim_station.vi_faults()["magnet_z"].acknowledged is True


def test_retry_fault_stale_resets_counter_and_repolls(sim_station: Station):
    """Below the disconnect threshold, retry_fault() just resets and re-polls
    the SAME driver — no rebuild, since the session is presumed fine.
    """
    original_driver = sim_station.magnet_z._driver
    sim_station.magnet_z._driver._simulate_error = True
    sim_station.get_state()
    assert sim_station.vi_faults()["magnet_z"].kind == "stale"

    # Still broken: retry fails, fault stands (downgraded — the counter was
    # reset and only failed once).
    ok, message = sim_station.retry_fault("magnet_z")
    assert ok is False
    assert "magnet_z" in message
    assert sim_station.vi_faults()["magnet_z"].kind == "stale"
    assert sim_station.magnet_z._driver is original_driver

    # Recovers: retry succeeds and clears the fault, same driver throughout.
    sim_station.magnet_z._driver._simulate_error = False
    ok, message = sim_station.retry_fault("magnet_z")
    assert ok is True
    assert "magnet_z" not in sim_station.vi_faults()
    assert sim_station.magnet_z._driver is original_driver

    # Unknown VI: explicit refusal, not a KeyError.
    ok, message = sim_station.retry_fault("no_such_vi")
    assert ok is False


def test_retry_fault_disconnected_rebuilds_the_driver_session(sim_station: Station):
    """Past the disconnect threshold, retry_fault() closes and reopens the
    driver session instead of re-polling the same (presumed-dead) handle —
    the fix for a magnet whose reconnect never worked after a hardware
    glitch until the whole app was restarted (only a restart rebuilds the
    driver from its config, since ``connect_instrument()``'s rebuild path
    is only reachable from the OFFLINE registry, not a live-but-faulted VI).
    """
    original_driver = sim_station.magnet_z._driver
    sim_station.magnet_z._driver._simulate_error = True
    sim_station.get_state()
    sim_station.get_state()
    sim_station.get_state()
    assert sim_station.vi_faults()["magnet_z"].kind == "disconnected"

    ok, message = sim_station.retry_fault("magnet_z")

    assert ok is True
    assert "magnet_z" in message
    assert "magnet_z" not in sim_station.vi_faults()
    # A genuinely fresh session, not the same (still-marked-broken) handle —
    # a bare re-poll of the OLD driver would still see _simulate_error=True.
    assert sim_station.magnet_z._driver is not original_driver
    assert sim_station.magnet_z._driver._simulate_error is False


# ---------------------------------------------------------------------------
# Unified condition registry (the System-Condition standard, see
# i2as/core/conditions.py and GLOSSARY.md) — Station.conditions() and
# the transitional vi_faults()/vi_safety_holds() adapters over it.
# ---------------------------------------------------------------------------


class _CriticalFlagAVI(BaseVirtualInstrument):
    """Test double: reports a hardcoded critical-severity flag, no concerns."""

    safety_flags: ClassVar[dict[str, str]] = {"flag_aaa": "critical"}

    def evaluate_safety(self, state: dict) -> dict[str, bool]:
        return {"flag_aaa": True}


class _CriticalFlagZVI(BaseVirtualInstrument):
    """Test double: reports a second, alphabetically-later critical flag."""

    safety_flags: ClassVar[dict[str, str]] = {"flag_zzz": "critical"}

    def evaluate_safety(self, state: dict) -> dict[str, bool]:
        return {"flag_zzz": True}


class _HoldFlagVI(BaseVirtualInstrument):
    """Test double: reports a hold-severity flag whenever ``tripped`` is set.

    No shipped VI declares a hold-severity flag, so the hold half of the
    System-Condition standard is exercised the way a real setup declares
    one: a producer naming the flag in ``safety_flags`` and reporting it
    from ``evaluate_safety()``, plus a consumer naming it in
    ``safety_concerns()``.
    """

    safety_flags: ClassVar[dict[str, str]] = {"coolant_low": "hold"}
    tripped: bool = True

    def evaluate_safety(self, state: dict) -> dict[str, bool]:
        return {"coolant_low": bool(self.tripped)}


class _CoolantConcernedVI(BaseVirtualInstrument):
    """Test double: cannot operate while the hold-severity flag is tripped."""

    def safety_concerns(self) -> set[str]:
        return {"coolant_low"}


def _hold_flag_station() -> Station:
    """A minimal station with one hold-flag producer and one concerned VI."""
    station = Station()
    station.register_vi("monitor", _HoldFlagVI({}), "system")
    station.register_vi("consumer", _CoolantConcernedVI({}), "system")
    return station


class _UnconsumedHoldFlagVI(BaseVirtualInstrument):
    """Test double: reports a hold-severity flag no VI's safety_concerns() names."""

    safety_flags: ClassVar[dict[str, str]] = {"widget_stuck": "hold"}

    def evaluate_safety(self, state: dict) -> dict[str, bool]:
        return {"widget_stuck": True}


def test_comm_condition_escalates_preserving_since_and_ack(sim_station: Station):
    """The unified registry's comm-origin condition mirrors the pre-unification
    FaultRecord lifecycle exactly: record -> escalate stale->disconnected
    (since/acknowledged preserved) -> recover/clear.
    """
    sim_station.get_state()
    assert "comm:magnet_z" not in sim_station.conditions()

    sim_station.magnet_z._driver._simulate_error = True
    sim_station.get_state()
    condition = sim_station.conditions()["comm:magnet_z"]
    assert condition.origin == "comm"
    assert condition.severity == "hold"
    assert condition.kind == "stale"
    assert condition.source_vis == ("magnet_z",)
    assert condition.affected_vis == frozenset({"magnet_z"})
    assert condition.acknowledged is False
    since_first = condition.since

    assert sim_station.acknowledge_condition("comm:magnet_z") is True
    assert sim_station.conditions()["comm:magnet_z"].acknowledged is True

    sim_station.get_state()
    sim_station.get_state()  # 3rd consecutive error -> disconnected
    condition = sim_station.conditions()["comm:magnet_z"]
    assert condition.kind == "disconnected"
    # Escalating the SAME incident preserves 'since' AND 'acknowledged'.
    assert condition.since == since_first
    assert condition.acknowledged is True

    sim_station.magnet_z._driver._simulate_error = False
    sim_station.get_state()
    assert "comm:magnet_z" not in sim_station.conditions()


def test_vi_faults_adapter_agrees_with_conditions_registry(sim_station: Station):
    """vi_faults() is a thin view of the unified registry's comm conditions."""
    sim_station.magnet_z._driver._simulate_error = True
    sim_station.get_state()
    sim_station.get_state()
    sim_station.get_state()  # disconnected

    faults = sim_station.vi_faults()
    comm_conditions = {
        c.source_vis[0]: c for c in sim_station.conditions().values() if c.origin == "comm"
    }
    assert set(faults) == set(comm_conditions) == {"magnet_z"}
    fault = faults["magnet_z"]
    condition = comm_conditions["magnet_z"]
    assert fault.vi_name == "magnet_z"
    assert fault.kind == condition.kind
    assert fault.message == condition.message
    assert fault.since == condition.since
    assert fault.acknowledged == condition.acknowledged


def test_update_conditions_tolerated_hold_flag_constructs_nothing():
    """A tolerated hold-severity flag builds no condition and holds no VI."""
    station = _hold_flag_station()
    safety = station.check_safety()
    assert safety["coolant_low"] is True

    station.update_conditions(safety, tolerated_flags=frozenset({"coolant_low"}))
    assert "safety:coolant_low" not in station.conditions()
    assert not any(
        c.origin == "safety" and c.severity == "hold"
        for c in station.conditions().values()
    )


def test_update_conditions_clearing_tolerance_recreates_with_fresh_since(
    monkeypatch: pytest.MonkeyPatch
):
    """Un-tolerating a flag rebuilds its condition with a FRESH `since` —
    the tolerated interval removed the condition entirely, so there is no
    prior entry for _upsert_condition() to preserve `since` from.
    """
    station = _hold_flag_station()
    safety = station.check_safety()
    assert safety["coolant_low"] is True

    import i2as.core.station as station_module

    clock = iter([100.0, 200.0, 300.0])
    monkeypatch.setattr(station_module.time, "time", lambda: next(clock))

    station.update_conditions(safety, tolerated_flags=frozenset())
    assert station.conditions()["safety:coolant_low"].since == 100.0

    station.update_conditions(safety, tolerated_flags=frozenset({"coolant_low"}))
    assert "safety:coolant_low" not in station.conditions()

    station.update_conditions(safety, tolerated_flags=frozenset())
    assert station.conditions()["safety:coolant_low"].since == 300.0


def test_update_conditions_critical_flag_ignores_tolerance(sim_station: Station):
    """A critical flag builds its station-wide condition even if 'tolerated'.

    Critical is station-wide scope by construction (the System-Condition
    standard: scope follows from severity alone) — tolerance never applies
    to it, and no VI's safety_concerns() is ever consulted for a critical
    flag (a per-VI hold would be meaningless once EMERGENCY has already
    stopped everything).
    """
    sim_station.magnet_z._driver._simulate_quench = True
    state = sim_station.get_state()
    safety = sim_station.check_safety(state)
    assert safety["quench"] is True

    sim_station.update_conditions(safety, tolerated_flags=frozenset({"quench"}))
    condition = sim_station.conditions()["safety:quench"]
    assert condition.severity == "critical"
    assert condition.affected_vis is None
    assert condition.source_vis == ("magnet_z",)


def test_update_conditions_critical_flag_produces_only_its_own_condition(
    sim_station: Station,
):
    """A critical flag produces ONLY its own critical condition — never a
    concern-based hold, even if some VI's safety_concerns() named it.

    Concern-based holds exist only for hold-severity flags (see the
    System-Condition standard); a critical flag's station-wide scope
    already covers every VI, concerned or not, so there is no separate
    'safety-hold:<flag>' condition coexisting with the critical one.
    """
    sim_station.magnet_z._driver._simulate_quench = True
    state = sim_station.get_state()
    safety = sim_station.check_safety(state)

    sim_station.update_conditions(safety, tolerated_flags=frozenset())
    assert sim_station.conditions()["safety:quench"].severity == "critical"
    assert set(sim_station.conditions()) == {"safety:quench"}


def test_update_conditions_hold_flag_scopes_to_concerned_vis():
    """A hold-severity flag holds exactly the VIs whose safety_concerns()
    name it — never the VI that reported it.
    """
    station = _hold_flag_station()
    safety = station.check_safety()
    assert safety["coolant_low"] is True

    station.update_conditions(safety, tolerated_flags=frozenset())
    condition = station.conditions()["safety:coolant_low"]
    assert condition.severity == "hold"
    assert condition.affected_vis == frozenset({"consumer"})


def test_acknowledge_condition_acknowledges_a_safety_hold():
    """acknowledge_condition() marks a hold-severity safety condition as seen."""
    station = _hold_flag_station()
    safety = station.check_safety()

    station.update_conditions(safety, tolerated_flags=frozenset())
    assert station.acknowledge_condition("safety:coolant_low") is True
    assert station.conditions()["safety:coolant_low"].acknowledged is True
    assert station.acknowledge_condition("safety:no_such_flag") is False


def test_active_critical_conditions_sorted_by_key():
    """active_critical_conditions() returns every critical condition, sorted."""
    station = Station()
    station.register_vi("vi_z", _CriticalFlagZVI({}), "system")
    station.register_vi("vi_a", _CriticalFlagAVI({}), "system")

    safety = station.check_safety()
    assert safety == {"flag_zzz": True, "flag_aaa": True}

    station.update_conditions(safety, tolerated_flags=frozenset())
    critical = station.active_critical_conditions()
    assert [c.key for c in critical] == ["safety:flag_aaa", "safety:flag_zzz"]
    assert all(c.severity == "critical" and c.affected_vis is None for c in critical)


def test_update_conditions_warns_once_for_unconsumed_hold_flag(caplog: pytest.LogCaptureFixture):
    """A hold-severity flag nobody concerns itself with builds no condition
    and logs exactly one WARNING, not one per tick it stays unconsumed.
    """
    station = Station()
    station.register_vi("widget", _UnconsumedHoldFlagVI({}), "system")
    safety = station.check_safety()
    assert safety == {"widget_stuck": True}

    with caplog.at_level(logging.WARNING, logger="i2as.core.station"):
        station.update_conditions(safety, tolerated_flags=frozenset())
        station.update_conditions(safety, tolerated_flags=frozenset())

    assert station.conditions() == {}
    matching = [r for r in caplog.records if "widget_stuck" in r.getMessage()]
    assert len(matching) == 1


def _trend_condition(name: str, message: str = "unstable", since: float = 0.0) -> Condition:
    return Condition(
        key=f"trend:{name}",
        origin="trend",
        severity="advisory",
        kind=name,
        source_vis=(),
        affected_vis=None,
        message=message,
        since=since,
    )


def test_publish_conditions_adds_and_clears_by_origin(sim_station: Station):
    """publish_conditions() upserts the desired set and prunes the rest, origin-scoped."""
    sim_station.publish_conditions("trend", [_trend_condition("a"), _trend_condition("b")])
    assert set(sim_station.conditions()) == {"trend:a", "trend:b"}

    # A refresh naming only "a" clears "b" — the desired set is complete, not a delta.
    sim_station.publish_conditions("trend", [_trend_condition("a")])
    assert set(sim_station.conditions()) == {"trend:a"}

    # An empty refresh clears everything of this origin.
    sim_station.publish_conditions("trend", [])
    assert sim_station.conditions() == {}


def test_publish_conditions_preserves_since_and_acknowledged(sim_station: Station):
    """A key that stays continuously active keeps its original since/acknowledged."""
    sim_station.publish_conditions("trend", [_trend_condition("a", since=100.0)])
    sim_station.acknowledge_condition("trend:a")

    sim_station.publish_conditions("trend", [_trend_condition("a", since=999.0, message="still unstable")])

    condition = sim_station.conditions()["trend:a"]
    assert condition.since == 100.0
    assert condition.acknowledged is True
    assert condition.message == "still unstable"


def test_publish_conditions_rejects_origin_mismatch(sim_station: Station):
    from dataclasses import replace

    mismatched = replace(_trend_condition("a"), origin="safety")
    with pytest.raises(ValueError, match="origin"):
        sim_station.publish_conditions("trend", [mismatched])


def test_publish_conditions_does_not_disturb_other_origins():
    """The trend origin's refresh never wipes a safety-origin condition, and vice versa."""
    station = _hold_flag_station()
    safety = station.check_safety()
    station.update_conditions(safety, tolerated_flags=frozenset())
    assert "safety:coolant_low" in station.conditions()

    # The trend refresh (its own cadence) must not touch the safety condition.
    station.publish_conditions("trend", [_trend_condition("a")])
    assert "safety:coolant_low" in station.conditions()
    assert "trend:a" in station.conditions()

    # The per-tick safety refresh must not touch the trend condition either.
    station.update_conditions(safety, tolerated_flags=frozenset())
    assert "trend:a" in station.conditions()


def test_acknowledge_condition_unknown_key_returns_false(sim_station: Station):
    """acknowledge_condition() on a key with no active condition is a no-op."""
    assert sim_station.acknowledge_condition("safety:no_such_flag") is False
    assert sim_station.acknowledge_condition("comm:no_such_vi") is False


# ---------------------------------------------------------------------------
# Degraded build: offline instruments and reconnection
# ---------------------------------------------------------------------------


class _UnreachableDriver:
    """Test double for a driver whose instrument never answers."""

    def __init__(self, resource_string: str) -> None:
        from i2as.core.exceptions import I2ASCommunicationError

        raise I2ASCommunicationError(
            f"Cannot open instrument at {resource_string}"
        )


class _FlakyDriver:
    """Test double that fails construction ``fail_times`` times, then succeeds.

    Models "the user plugged the cable back in between startup and retry".
    Class-level counter so build_station's import-by-dotted-path sees the
    same state as the test; reset it in each test that uses this class.
    """

    fail_times: int = 0
    attempts: int = 0

    def __init__(self, resource_string: str) -> None:
        from i2as.core.exceptions import I2ASCommunicationError

        type(self).attempts += 1
        if type(self).attempts <= type(self).fail_times:
            raise I2ASCommunicationError(
                f"Cannot open instrument at {resource_string}"
            )
        self.closed = False

    def get_idn(self) -> str:
        return "I2AS,FLAKY-STUB,0,0"

    def close(self) -> None:
        self.closed = True


class _StubVI(BaseVirtualInstrument):
    """Minimal VI test double satisfying the build contract.

    Inherits the connection-lifecycle standard's ``ping()`` / ``disconnect()``
    from the base, so a station built from it behaves exactly like a real one
    at the identity check and on connect/disconnect.
    """

    vi_type = "system"

    def __init__(self, drivers: dict, **init_params) -> None:
        super().__init__(drivers, **init_params)


class _CommFailVI(_StubVI):
    """VI test double whose own bring-up cannot talk to the hardware."""

    def __init__(self, drivers: dict, **init_params) -> None:
        from i2as.core.exceptions import I2ASCommunicationError

        raise I2ASCommunicationError("VI bring-up query got no response")


def _write_degraded_config(
    tmp_path: Path, driver_class: str, vi_class: str = "tests.test_l2_station._StubVI"
) -> str:
    """Write a two-driver / two-VI config: one healthy pair, one under test."""
    (tmp_path / "devices.yaml").write_text(
        "real_drivers:\n"
        "  good_drv:\n"
        "    class: tests.test_l2_station._AddressCapturingDriver\n"
        '    address: "GPIB0::10::INSTR"\n'
        "  bad_drv:\n"
        f"    class: {driver_class}\n"
        '    address: "GPIB0::12::INSTR"\n'
        "virtual_instruments:\n"
        "  good_vi:\n"
        "    class: tests.test_l2_station._StubVI\n"
        "    drivers: {main: good_drv}\n"
        "    vi_type: system\n"
        "  bad_vi:\n"
        f"    class: {vi_class}\n"
        "    drivers: {main: bad_drv}\n"
        "    vi_type: measurement\n"
    )
    (tmp_path / "monitor.yaml").write_text(
        "monitor:\n  tick_interval_ms: 1000\n  max_vi_errors: 3\n"
    )
    return str(tmp_path)


def test_build_station_degrades_on_unreachable_driver(tmp_path):
    """One unreachable instrument must not abort the build: it goes offline."""
    station = build_station(
        _write_degraded_config(tmp_path, "tests.test_l2_station._UnreachableDriver")
    )

    assert station.get_vi_names() == ["good_vi"]
    assert station.offline_vi_names() == ["bad_vi"]
    info = station.get_offline_info("bad_vi")
    assert info.vi_type == "measurement"
    assert "bad_drv" in info.reason
    assert "GPIB0::12::INSTR" in info.reason
    assert info.failed_drivers == ("bad_drv",)
    # Offline VIs are invisible to the live enumerators.
    assert station.has_vi("bad_vi") is False
    assert station.measurement_vi_names() == []


def test_build_station_degrades_on_vi_communication_error(tmp_path):
    """A VI whose own bring-up raises a communication error goes offline too."""
    station = build_station(
        _write_degraded_config(
            tmp_path,
            "tests.test_l2_station._AddressCapturingDriver",
            vi_class="tests.test_l2_station._CommFailVI",
        )
    )

    assert station.offline_vi_names() == ["bad_vi"]
    info = station.get_offline_info("bad_vi")
    assert "no response" in info.reason
    assert info.failed_drivers == ()


def test_build_station_still_raises_on_unknown_driver_reference(tmp_path):
    """Config errors must still abort the build (they are not connection faults)."""
    from i2as.core.exceptions import I2ASConfigError

    (tmp_path / "devices.yaml").write_text(
        "real_drivers: {}\n"
        "virtual_instruments:\n"
        "  broken_vi:\n"
        "    class: tests.test_l2_station._StubVI\n"
        "    drivers: {main: no_such_driver}\n"
    )
    (tmp_path / "monitor.yaml").write_text("monitor:\n  tick_interval_ms: 1000\n")

    with pytest.raises(I2ASConfigError, match="no_such_driver"):
        build_station(str(tmp_path))


def test_fallback_keeps_config_with_unreachable_instrument(tmp_path):
    """An unreachable instrument must NOT trigger the config fallback chain."""
    from i2as.core.station import build_station_with_fallback

    real_cfg = tmp_path / "real"
    real_cfg.mkdir()
    _write_degraded_config(real_cfg, "tests.test_l2_station._UnreachableDriver")
    sim_cfg = str(
        Path(__file__).parent.parent / "i2as" / "configs" / "sim_cryostat"
    )

    station, used_path, warnings = build_station_with_fallback(
        [str(real_cfg), sim_cfg]
    )

    assert used_path == str(real_cfg)
    assert warnings == []
    assert station.offline_vi_names() == ["bad_vi"]


def test_connect_instrument_reconnects_after_transient_failure(tmp_path):
    """connect_instrument() brings a VI live once its driver becomes reachable."""
    _FlakyDriver.fail_times = 1
    _FlakyDriver.attempts = 0
    station = build_station(
        _write_degraded_config(tmp_path, "tests.test_l2_station._FlakyDriver")
    )
    assert station.offline_vi_names() == ["bad_vi"]

    ok, message = station.connect_instrument("bad_vi")

    assert ok is True
    assert "bad_vi" in message
    assert station.offline_vi_names() == []
    assert station.has_vi("bad_vi") is True
    assert station.measurement_vi_names() == ["bad_vi"]


def test_connect_instrument_failure_updates_reason_and_stays_offline(tmp_path):
    """A failed retry keeps the VI offline and refreshes the failure reason."""
    station = build_station(
        _write_degraded_config(tmp_path, "tests.test_l2_station._UnreachableDriver")
    )

    ok, message = station.connect_instrument("bad_vi")

    assert ok is False
    assert "bad_drv" in message
    assert station.offline_vi_names() == ["bad_vi"]
    assert station.has_vi("bad_vi") is False
    assert "bad_drv" in station.get_offline_info("bad_vi").reason


def test_connect_instrument_rejects_non_offline_name(sim_station: Station):
    """Retrying a live or unknown VI returns an explicit failure verdict."""
    ok, message = sim_station.connect_instrument("magnet_z")
    assert ok is False
    assert "not offline" in message

    ok, message = sim_station.connect_instrument("no_such_vi")
    assert ok is False


class _ToggleableCommDriver:
    """Test double whose identity/reading calls fail while ``broken`` is True.

    A CLASS-level flag, unlike the sim drivers' instance-level
    ``_simulate_error``: models hardware that is still disconnected even
    after a fresh session is opened, which an instance flag cannot (a
    rebuild constructs a brand new instance, leaving any instance-level
    "still broken" flag behind on the old, discarded one).
    """

    broken: bool = False

    def __init__(self, resource_string: str) -> None:
        self.closed = False

    def get_idn(self) -> str:
        if type(self).broken:
            raise I2ASCommunicationError("bus session is dead")
        return "I2AS,TOGGLE-STUB,0,0"

    def get_reading(self) -> float:
        if type(self).broken:
            raise I2ASCommunicationError("bus session is dead")
        return 1.0

    def close(self) -> None:
        self.closed = True


class _ToggleableCommVI(_StubVI):
    """VI double with one @monitored getter that reads its driver.

    ``_StubVI`` alone has no @monitored methods, so ``get_state()`` never
    touches the driver at all — useless for simulating a live comm fault.
    """

    @monitored
    def reading(self) -> float:
        return self._drivers["main"].get_reading()


def test_retry_fault_disconnected_stays_faulted_while_hardware_is_still_broken(
    tmp_path,
):
    """A genuinely still-broken instrument fails the rebuild and stays live+faulted.

    Never demoted to the offline registry — a run watching it can still
    fail cleanly through the normal comm-condition path.
    """
    _ToggleableCommDriver.broken = False
    station = build_station(
        _write_degraded_config(
            tmp_path,
            "tests.test_l2_station._ToggleableCommDriver",
            vi_class="tests.test_l2_station._ToggleableCommVI",
        )
    )
    assert station.has_vi("bad_vi") is True

    _ToggleableCommDriver.broken = True
    station.get_state()
    station.get_state()
    station.get_state()
    assert station.vi_faults()["bad_vi"].kind == "disconnected"

    ok, message = station.retry_fault("bad_vi")

    assert ok is False
    assert "bad_vi" in message
    assert station.vi_faults()["bad_vi"].kind == "disconnected"
    assert station.has_vi("bad_vi") is True

    _ToggleableCommDriver.broken = False
    ok, message = station.retry_fault("bad_vi")

    assert ok is True
    assert "bad_vi" not in station.vi_faults()


# ---------------------------------------------------------------------------
# The Availability standard (i2as.core.availability): Station.availability()
# / availabilities() as a derived view over the offline registry, the unified
# condition registry, and the VI's own attachment state.
# ---------------------------------------------------------------------------


class _SucceedsOnceDriver:
    """Test double that connects successfully once, then fails on every retry.

    Models the bug-fix scenario a failed reconnect exercises: a driver that
    was reachable at build time (so the VI came up live) becomes unreachable
    by the time an operator-disconnected VI is reconnected. Class-level
    counter so build_station's import-by-dotted-path sees the same state as
    the test; reset it in each test that uses this class.
    """

    attempts: int = 0

    def __init__(self, resource_string: str) -> None:
        from i2as.core.exceptions import I2ASCommunicationError

        type(self).attempts += 1
        if type(self).attempts > 1:
            raise I2ASCommunicationError(
                f"Cannot open instrument at {resource_string}"
            )
        self.closed = False

    def get_idn(self) -> str:
        return "I2AS,SUCCEEDS-ONCE-STUB,0,0"

    def close(self) -> None:
        self.closed = True


def test_availability_live_for_a_healthy_vi(sim_station: Station):
    """A VI with no offline record and no active condition is simply live."""
    avail = sim_station.availability("magnet_z")

    assert avail.state == "live"
    assert avail.tags == frozenset()
    assert avail.reason == ""


def test_availability_absent_with_operator_tag_after_disconnect(sim_station: Station):
    """disconnect_instrument() must be visible through availability(), not just get_offline_info()."""
    sim_station.disconnect_instrument("magnet_z")

    avail = sim_station.availability("magnet_z")

    assert avail.state == "absent"
    assert avail.tags == frozenset({"operator"})


def test_availability_faulted_with_not_responding_tag_under_standing_comm_condition(
    sim_station: Station,
):
    """A live VI under a standing comm fault reports faulted/not_responding."""
    sim_station._record_comm_condition("temperature", "disconnected", "boom")

    avail = sim_station.availability("temperature")

    assert avail.state == "faulted"
    assert avail.tags == frozenset({"not_responding"})
    assert avail.reason == "boom"


def test_availability_covers_every_configured_vi(sim_station: Station):
    """availabilities() must answer for both live and offline VIs."""
    sim_station.disconnect_instrument("magnet_z")

    result = sim_station.availabilities()

    assert set(result) == set(sim_station.get_vi_names()) | set(
        sim_station.offline_vi_names()
    )
    assert result["magnet_z"].state == "absent"


def test_availability_failed_reconnect_of_operator_disconnected_vi_adds_connect_failed(
    tmp_path,
):
    """Bug fix: a failed reconnect ADDS connect_failed rather than overwriting operator."""
    _SucceedsOnceDriver.attempts = 0
    station = build_station(
        _write_degraded_config(tmp_path, "tests.test_l2_station._SucceedsOnceDriver")
    )
    assert station.has_vi("bad_vi") is True

    station.disconnect_instrument("bad_vi")
    assert station.get_offline_info("bad_vi").tags == frozenset({"operator"})

    ok, _ = station.connect_instrument("bad_vi")
    assert ok is False

    info = station.get_offline_info("bad_vi")
    assert info.tags == frozenset({"operator", "connect_failed"})

    avail = station.availability("bad_vi")
    assert avail.state == "absent"
    assert avail.tags == frozenset({"operator", "connect_failed"})


def test_read_gateway_config_reads_the_declared_values(tmp_path):
    """A setup that opens itself to agents says so in its own monitor.yaml."""
    from i2as.core.config import read_gateway_config

    (tmp_path / "monitor.yaml").write_text(
        "monitor:\n"
        "  tick_interval_ms: 1000\n"
        "  gateway_server: true\n"
        "  gateway_max_role: session\n"
    )
    assert read_gateway_config(str(tmp_path)) == {
        "gateway_server": True,
        "gateway_max_role": "session",
    }






def test_read_gateway_config_defaults_to_the_closed_door(tmp_path):
    """Absent, malformed or unreadable all mean off — never raises."""
    from i2as.core.config import read_gateway_config

    closed = {"gateway_server": False, "gateway_max_role": "observer"}
    assert read_gateway_config(str(tmp_path / "nowhere")) == closed
    (tmp_path / "monitor.yaml").write_text("monitor:\n  tick_interval_ms: 1000\n")
    assert read_gateway_config(str(tmp_path)) == closed
    (tmp_path / "monitor.yaml").write_text("monitor: not-a-mapping\n")
    assert read_gateway_config(str(tmp_path)) == closed


def test_read_instrument_thread_defaults_to_the_thread(tmp_path):
    """Silence means the instrument thread: it is the standard, not an opt-in.

    Absent, malformed or unreadable all inherit it too — the fallback of a
    file nobody can read must be the shape every other setup runs in.
    """
    from i2as.core.config import read_instrument_thread

    assert read_instrument_thread(str(tmp_path / "nowhere")) is True
    (tmp_path / "monitor.yaml").write_text("monitor:\n  tick_interval_ms: 1000\n")
    assert read_instrument_thread(str(tmp_path)) is True
    (tmp_path / "monitor.yaml").write_text("monitor: not-a-mapping\n")
    assert read_instrument_thread(str(tmp_path)) is True


def test_read_instrument_thread_honours_a_deliberate_refusal(tmp_path):
    """A setup that wants the temporary inline mode back says so explicitly."""
    from i2as.core.config import read_instrument_thread

    (tmp_path / "monitor.yaml").write_text(
        "monitor:\n  tick_interval_ms: 1000\n  instrument_thread: false\n"
    )
    assert read_instrument_thread(str(tmp_path)) is False
    (tmp_path / "monitor.yaml").write_text(
        "monitor:\n  tick_interval_ms: 1000\n  instrument_thread: true\n"
    )
    assert read_instrument_thread(str(tmp_path)) is True


# ── Lifecycle-state standard: the Station's read (GLOSSARY.md's Lifecycle
#    state; the VI half lives in BaseVirtualInstrument) ────────────────────


def test_lifecycle_states_cover_every_vi_and_follow_the_verbs(sim_station):
    """Every configured VI is reported, and initiate_all/standby_all move all of them."""
    states = sim_station.lifecycle_states()
    assert set(states) == set(sim_station.get_vi_names())
    assert set(states.values()) == {"idle"}

    sim_station.initiate_all()
    assert set(sim_station.lifecycle_states().values()) == {"initiated"}

    sim_station.standby_all()
    assert set(sim_station.lifecycle_states().values()) == {"standby"}


def test_lifecycle_state_of_one_vi_matches_the_vi_itself(sim_station):
    """The Station's per-VI read is a passthrough, never a second copy."""
    sim_station.execute_vi_action("magnet_z", "initiate")
    assert sim_station.lifecycle_state("magnet_z") == "initiated"
    assert sim_station.lifecycle_state("magnet_z") == (
        sim_station.get_vi("magnet_z").lifecycle_state()
    )
    assert sim_station.lifecycle_state("temperature") == "idle"

    with pytest.raises(KeyError):
        sim_station.lifecycle_state("no_such_instrument")


def test_lifecycle_state_reads_idle_for_a_disconnected_instrument(sim_station):
    """A released instrument is not initiated as far as any client is concerned."""
    sim_station.execute_vi_action("magnet_z", "initiate")
    assert sim_station.lifecycle_state("magnet_z") == "initiated"

    ok, _message = sim_station.disconnect_instrument("magnet_z")
    assert ok
    assert sim_station.lifecycle_state("magnet_z") == "idle"
    assert sim_station.lifecycle_states()["magnet_z"] == "idle"


def test_lifecycle_states_send_no_instrument_traffic(sim_station):
    """The snapshot assembly reads this every tick, so it must not poll."""
    from tests.mocks.bus_spy import spy_on_station

    sim_station.initiate_all()
    calls: list[str] = []
    spy_on_station(sim_station, calls)
    assert sim_station.lifecycle_states()
    assert not calls, f"lifecycle_states() polled the bus: {sorted(set(calls))}"
