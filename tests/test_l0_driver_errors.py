"""L0 tests for the driver error-reporting standard (see ``drivers/README.md``).

One pair of tests per instrument that has a real driver:

* the **real driver** is given a transport whose replies say "I refused
  that", and must raise ``I2ASInstrumentError`` carrying the
  instrument's own code; and
* the **sim twin** is driven through the wrong command sequence and must
  raise the same typed error with the same code, so the mistake fails here
  instead of on hardware.

A sim-only driver (no real twin ships) is covered by the sim half alone.
The real drivers are exercised with their VISA session replaced by a mock —
``object.__new__`` skips ``__init__``'s bus open. No test in this file
opens a resource; the bench counterparts carry the ``hardware`` marker and
live in ``tests/test_bench_hardware.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from i2as.core.exceptions import I2ASInstrumentError

# ── Keithley 6221 — SCPI error queue ──────────────────────────────────────────


def _fake_visa_driver(cls, query_replies):
    """Build a real driver with its VISA session mocked and __init__ skipped.

    Args:
        cls: The real driver class.
        query_replies: Iterable of replies the fake session's ``query()``
            returns, in order.

    Returns:
        The driver instance, ready for the method under test.
    """
    driver = object.__new__(cls)
    instr = MagicMock()
    instr.query.side_effect = query_replies
    instr.timeout = 5_000
    driver._instr = instr
    return driver


def test_sim_keithley_6221_refuses_output_switch_while_delta_is_armed():
    """The wrong sequence: switch the output by hand with delta armed.

    The real driver never does this — ``stop_delta_mode()`` sends
    ``:SOUR:SWE:ABOR`` *before* ``OUTP OFF`` — so a caller that skips the
    abort is exactly what this refusal exists to catch.
    """
    from i2as.drivers.sim_keithley_2182a import SimKeithley2182A
    from i2as.drivers.sim_keithley_6221 import SimKeithley6221

    source = SimKeithley6221("SIM")
    source._paired_meter = SimKeithley2182A("SIM")
    source.configure_and_start_delta(high_current=1e-6, n_readings=5, delay=0.01)

    with pytest.raises(I2ASInstrumentError) as excinfo:
        source.set_source_enabled(False)
    assert excinfo.value.code == "-221"
    assert excinfo.value.instrument_message == "Settings conflict"

    # ...and the documented recovery works: abort first, then switch.
    source.stop_delta_mode()
    source.set_source_enabled(False)


def test_sim_keithley_6221_reproduces_the_221_leftover_range_rejection():
    """The -221 incident itself: a pinned range left behind by delta mode.

    Live commissioning (2026-07-22) found delta mode leaves the source's
    current range FIXED with autorange OFF, so a later, larger DC current is
    rejected outright and the source silently keeps its old value.
    ``stop_delta_mode()`` deliberately does not undo that (neither does the
    real instrument), so the leftover state is still there afterwards.

    The current write is driven through ``_apply_current()`` — the sim's
    model of the bare ``:SOUR:CURR`` write — because the public
    ``set_current()`` is the *fix*: it re-asserts autorange first and is
    therefore immune, which is precisely what the next test pins.
    """
    from i2as.drivers.sim_keithley_2182a import SimKeithley2182A
    from i2as.drivers.sim_keithley_6221 import SimKeithley6221

    source = SimKeithley6221("SIM")
    source._paired_meter = SimKeithley2182A("SIM")
    source.configure_and_start_delta(high_current=1e-6, n_readings=5, delay=0.01)
    source.stop_delta_mode()
    assert source._autorange is False

    with pytest.raises(I2ASInstrumentError) as excinfo:
        source._apply_current("set_current(0.0001)", 1e-4)
    assert excinfo.value.code == "-221"
    assert source.get_current() == 0.0  # the source did NOT take the value


def test_sim_keithley_6221_set_current_recovers_from_the_leftover_range():
    """The fix: ``set_current()`` re-asserts autorange, so -221 cannot recur."""
    from i2as.drivers.sim_keithley_2182a import SimKeithley2182A
    from i2as.drivers.sim_keithley_6221 import SimKeithley6221

    source = SimKeithley6221("SIM")
    source._paired_meter = SimKeithley2182A("SIM")
    source.configure_and_start_delta(high_current=1e-6, n_readings=5, delay=0.01)
    source.stop_delta_mode()

    source.set_current(1e-4)
    assert source.get_current() == pytest.approx(1e-4)


# ── Keithley 2182A — SCPI error queue ─────────────────────────────────────────


def test_keithley_2182a_real_raises_typed_error_from_the_scpi_queue():
    """A refused range change must not be mistaken for a range change."""
    from i2as.drivers.keithley_2182a import Keithley2182A

    driver = _fake_visa_driver(
        Keithley2182A, ['-222,"Parameter data out of range"']
    )
    with pytest.raises(I2ASInstrumentError) as excinfo:
        driver.set_range(1000.0)
    assert excinfo.value.code == "-222"
    assert excinfo.value.instrument_message == "Parameter data out of range"


def test_sim_keithley_2182a_refuses_a_range_above_the_channel_maximum():
    """The wrong sequence: ask for a range the channel does not have."""
    from i2as.drivers.sim_keithley_2182a import SimKeithley2182A

    meter = SimKeithley2182A("SIM")
    before = meter.get_range()
    with pytest.raises(I2ASInstrumentError) as excinfo:
        meter.set_range(1000.0)
    assert excinfo.value.code == "-222"
    assert meter.get_range() == before  # the range did NOT change


# ── Lakeshore 335 — IEEE-488.2 status byte ────────────────────────────────────


def test_lakeshore_335_real_raises_typed_error_from_the_event_status_register():
    """An execution-error bit in ``*ESR?`` becomes the typed error."""
    from i2as.drivers.lakeshore_335 import Lakeshore335

    driver = _fake_visa_driver(Lakeshore335, ["16"])  # bit 4 = execution error
    with pytest.raises(I2ASInstrumentError) as excinfo:
        driver.set_heater_range("HIGH")
    assert excinfo.value.code == "ESR:0x10"
    assert "Execution error" in excinfo.value.instrument_message


def test_lakeshore_335_real_clean_event_status_raises_nothing():
    """A zero register is the only reading that means the command landed."""
    from i2as.drivers.lakeshore_335 import Lakeshore335

    driver = _fake_visa_driver(Lakeshore335, ["0"])
    driver.set_heater_range("OFF")


def test_sim_lakeshore_335_refuses_an_empty_user_curve_slot():
    """The wrong sequence: assign a USER curve slot that holds no curve."""
    from i2as.drivers.sim_lakeshore_335 import SimLakeshore335

    controller = SimLakeshore335("SIM")
    before = controller.get_sensor_curve("A")
    with pytest.raises(I2ASInstrumentError) as excinfo:
        controller.set_sensor_curve(45, "A")
    assert excinfo.value.code == "ESR:0x10"
    assert controller.get_sensor_curve("A") == before  # assignment did NOT happen


# ── Oxford IPS 120 — protocol acknowledgement ─────────────────────────────────


def test_sim_oxford_ips120_refuses_a_ramp_rate_while_clamped():
    """The wrong sequence: program a clamped PSU without clearing the clamp."""
    from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120

    psu = SimOxfordIPS120("SIM")
    psu.set_ramp_rate(0.5)
    psu._simulate_clamp = True

    with pytest.raises(I2ASInstrumentError) as excinfo:
        psu.set_ramp_rate(1.0)
    assert excinfo.value.code == "DENIED"
    assert psu.get_ramp_rate() == pytest.approx(0.5)  # unchanged
