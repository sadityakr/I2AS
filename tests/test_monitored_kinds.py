"""The monitored-kind standard: image and trace @monitored fields, end to end below the GUI.

Covers the declaration (``core/decorators.py``), the VI's array read
(``BaseVirtualInstrument.read_monitored_array``), the Station's own-period
array poll (``Station.poll_monitored_arrays``), the declaration contract
(``MonitoredInfo``) and the shipped example on the camera VI.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from i2as.core.decorators import (
    ARRAY_KINDS,
    DEFAULT_ARRAY_PERIOD_S,
    get_monitored_axis,
    get_monitored_kind,
    get_monitored_methods,
    get_monitored_period_s,
    get_monitored_shape,
    monitored,
)
from i2as.core.events import MonitoredInfo
from i2as.core.exceptions import I2ASCommunicationError
from i2as.core.station import Station, _monitored_infos
from i2as.virtual_instruments.base import BaseVirtualInstrument


class ArrayVI(BaseVirtualInstrument):
    """A VI with one field of every kind, counting its reads."""

    vi_type = "mock"

    def __init__(self) -> None:
        super().__init__({})
        self.reads: dict[str, int] = {"level": 0, "preview": 0, "spectrum": 0}
        self.preview_value: object = np.ones((4, 3))
        self.fail_spectrum = False

    @monitored(unit="K", description="A scalar")
    def level(self) -> float:
        self.reads["level"] += 1
        return 1.5

    @monitored(unit="counts", description="An image", kind="image", shape=(4, 3), period_s=2.0)
    def preview(self):
        self.reads["preview"] += 1
        return self.preview_value

    @monitored(
        unit="V",
        description="A trace",
        kind="trace",
        shape=(5,),
        axis=(400, 800, "nm"),
    )
    def spectrum(self):
        self.reads["spectrum"] += 1
        if self.fail_spectrum:
            raise I2ASCommunicationError("bus timeout")
        return [0.0, 1.0, 2.0, 3.0, 4.0]


def _station(vi: BaseVirtualInstrument, vi_type: str = "system") -> Station:
    station = Station()
    station.register_vi("box", vi, vi_type)
    return station


# ----------------------------------------------------------------------
# The declaration
# ----------------------------------------------------------------------


class TestDeclaration:
    def test_scalar_is_the_default_and_declares_no_array(self):
        method = ArrayVI.level
        assert get_monitored_kind(method) == "scalar"
        assert get_monitored_shape(method) is None
        assert get_monitored_period_s(method) is None
        assert get_monitored_axis(method) is None

    def test_array_declarations_survive_the_vi_wrapping(self):
        assert get_monitored_kind(ArrayVI.preview) == "image"
        assert get_monitored_shape(ArrayVI.preview) == (4, 3)
        assert get_monitored_period_s(ArrayVI.preview) == 2.0
        assert get_monitored_kind(ArrayVI.spectrum) == "trace"
        assert get_monitored_period_s(ArrayVI.spectrum) == DEFAULT_ARRAY_PERIOD_S
        assert get_monitored_axis(ArrayVI.spectrum) == (400.0, 800.0, "nm")

    def test_methods_filter_by_kind(self):
        assert get_monitored_methods(ArrayVI) == ["level", "preview", "spectrum"]
        assert get_monitored_methods(ArrayVI, kinds=frozenset({"scalar"})) == ["level"]
        assert get_monitored_methods(ArrayVI, kinds=ARRAY_KINDS) == ["preview", "spectrum"]

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"kind": "movie"}, ValueError),
            ({"kind": "image"}, ValueError),  # no shape
            ({"kind": "image", "shape": (4,)}, ValueError),  # wrong ndim
            ({"kind": "trace", "shape": (0,)}, ValueError),
            ({"kind": "trace", "shape": (2.5,)}, TypeError),
            ({"kind": "trace", "shape": 5}, TypeError),
            ({"kind": "trace", "shape": (5,), "period_s": 0}, ValueError),
            ({"kind": "trace", "shape": (5,), "period_s": "1"}, TypeError),
            ({"kind": "image", "shape": (2, 2), "axis": (0, 1, "nm")}, ValueError),
            ({"kind": "trace", "shape": (5,), "axis": (0, 1)}, TypeError),
            ({"shape": (5,)}, ValueError),  # a scalar declares no shape
            ({"period_s": 1.0}, ValueError),
        ],
    )
    def test_a_bad_declaration_fails_at_import(self, kwargs, error):
        with pytest.raises(error):
            monitored(unit="", description="x", **kwargs)


# ----------------------------------------------------------------------
# The VI's half
# ----------------------------------------------------------------------


class TestVirtualInstrument:
    def test_get_state_reads_scalars_only(self):
        vi = ArrayVI()
        assert vi.get_state() == {"level": 1.5}
        assert vi.reads == {"level": 1, "preview": 0, "spectrum": 0}

    def test_read_monitored_array_returns_a_float64_copy_of_the_declared_shape(self):
        vi = ArrayVI()
        value = vi.read_monitored_array("preview")
        assert value.dtype == np.float64
        assert value.shape == (4, 3)
        assert value is not vi.preview_value

    def test_a_value_breaking_its_declared_shape_is_refused(self):
        vi = ArrayVI()
        vi.preview_value = np.ones((3, 3))
        with pytest.raises(ValueError, match="declared"):
            vi.read_monitored_array("preview")

    def test_none_means_no_value_yet(self):
        vi = ArrayVI()
        vi.preview_value = None
        assert vi.read_monitored_array("preview") is None

    def test_only_an_array_field_can_be_read_as_one(self):
        vi = ArrayVI()
        with pytest.raises(ValueError, match="not an array"):
            vi.read_monitored_array("level")
        with pytest.raises(ValueError, match="not an array"):
            vi.read_monitored_array("missing")


# ----------------------------------------------------------------------
# The Station's array poll
# ----------------------------------------------------------------------


class TestStationArrayPoll:
    def test_scalar_state_never_carries_an_array(self):
        station = _station(ArrayVI())
        assert station.get_state() == {"box": {"level": 1.5}}

    def test_each_field_is_read_on_its_own_period(self):
        vi = ArrayVI()
        station = _station(vi)

        first = station.poll_monitored_arrays(now=100.0)
        assert set(first["box"]) == {"preview", "spectrum"}
        np.testing.assert_array_equal(first["box"]["spectrum"], [0, 1, 2, 3, 4])

        # 1 s later: the trace (period 1 s) is due, the image (2 s) is not.
        assert set(station.poll_monitored_arrays(now=101.0)["box"]) == {"spectrum"}
        assert station.poll_monitored_arrays(now=101.5) == {}
        assert set(station.poll_monitored_arrays(now=102.0)["box"]) == {"preview", "spectrum"}
        assert vi.reads["preview"] == 2
        assert vi.reads["level"] == 0  # the array poll never reads a scalar

    def test_a_failing_read_is_dropped_without_touching_instrument_health(self):
        vi = ArrayVI()
        vi.fail_spectrum = True
        vi.preview_value = np.ones((9, 9))  # breaks its declaration
        station = _station(vi)

        assert station.poll_monitored_arrays(now=0.0) == {}
        assert station._error_counts["box"] == 0
        assert station.get_state() == {"box": {"level": 1.5}}  # not stale

    def test_a_stale_instrument_is_left_to_its_scalar_poll(self):
        vi = ArrayVI()
        station = _station(vi)
        station._error_counts["box"] = 1
        assert station.poll_monitored_arrays(now=0.0) == {}
        assert vi.reads["spectrum"] == 0

    def test_a_field_with_no_value_yet_is_left_out(self):
        vi = ArrayVI()
        vi.preview_value = None
        station = _station(vi)
        assert set(station.poll_monitored_arrays(now=0.0)["box"]) == {"spectrum"}


# ----------------------------------------------------------------------
# The declaration contract
# ----------------------------------------------------------------------


class TestMonitoredInfo:
    def test_station_info_reflects_the_kind(self):
        infos = {info.name: info for info in _monitored_infos(ArrayVI)}
        assert infos["level"].kind == "scalar"
        assert infos["level"].shape == ()
        assert infos["level"].period_s is None
        assert infos["preview"].kind == "image"
        assert infos["preview"].shape == (4, 3)
        assert infos["spectrum"].axis == (400.0, 800.0, "nm")

    def test_it_survives_a_json_round_trip(self):
        info = MonitoredInfo(
            name="spectrum", unit="V", kind="trace", shape=(5,), period_s=1.0,
            axis=(400.0, 800.0, "nm"),
        )
        again = MonitoredInfo.from_json(json.loads(json.dumps(info.to_json())))
        assert again == info

    def test_an_older_producer_reads_as_a_scalar(self):
        info = MonitoredInfo.from_json({"name": "level", "unit": "K"})
        assert info.kind == "scalar"
        assert info.shape == ()


# ----------------------------------------------------------------------
# The shipped example: the camera VI
# ----------------------------------------------------------------------


class TestCameraExample:
    @pytest.fixture
    def camera(self):
        from i2as.drivers.sim_camera import SimCamera
        from i2as.virtual_instruments.measurement.camera import CameraMeasurementVI

        vi = CameraMeasurementVI({"main": SimCamera(f"SIM::CAM@kinds_{time.time_ns()}")})
        vi.vi_name = "camera"
        return vi

    def test_frame_and_profile_follow_the_last_manual_read(self, camera):
        assert camera.read_monitored_array("last_frame") is None
        assert camera.read_monitored_array("last_row_profile") is None

        camera.initiate_measurement(exposure_s=0.01, binning=1, frames_per_step=1)
        camera.read_now()
        frame = camera.read_monitored_array("last_frame")
        profile = camera.read_monitored_array("last_row_profile")
        assert frame.shape == (128, 128)
        assert profile.shape == (128,)
        assert profile.tolist() in frame.tolist()

    def test_the_array_fields_never_reach_the_scalar_state(self, camera):
        state = camera.get_state()
        assert "last_frame" not in state
        assert "last_row_profile" not in state


# ----------------------------------------------------------------------
# The Orchestrator publishes arrays only while no procedure runs
# ----------------------------------------------------------------------


def test_orchestrator_emits_arrays_only_when_idle(qtbot):
    from i2as.core.orchestrator import Orchestrator, OrchestratorState

    vi = ArrayVI()
    station = _station(vi)
    orchestrator = Orchestrator(station)
    received: list[dict] = []
    orchestrator.monitored_arrays_updated.connect(received.append)
    orchestrator._monitoring = True

    orchestrator._tick_body()
    assert received and set(received[-1]["box"]) == {"preview", "spectrum"}

    received.clear()
    station._array_last_read.clear()  # everything due again
    orchestrator._state = OrchestratorState.MEASURING
    orchestrator._tick_body()
    assert received == []
