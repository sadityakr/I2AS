"""Layer-1 behaviour tests for the imaging VIs against their sims.

``StageAxisVI`` (one positioned axis, ramped like a setpoint) and
``CameraMeasurementVI`` (the shipped image-block measurement method). The
contracts themselves — silent construction, the setpoint convention, the
measurement round trip, the image-block shape — are held by
``tests/test_conformance.py``; these tests hold the behaviour.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from i2as.core import sim_environment
from i2as.core.exceptions import I2ASConfigError, I2ASSafetyError
from i2as.core.station import build_station
from i2as.drivers.sim_camera import SimCamera
from i2as.drivers.sim_xy_stage import SimXYStage
from i2as.virtual_instruments.base import StageBase
from i2as.virtual_instruments.stage.stage_axis import StageAxisVI

IMAGING_CONFIG = "i2as/configs/sim_imaging"


def _elapse(driver: SimXYStage, seconds: float) -> None:
    """Pretend *seconds* of wall-clock time passed since the driver's last update."""
    driver._last_update = time.time() - seconds


# ── StageAxisVI ──────────────────────────────────────────────────────────────


@pytest.fixture
def stage_driver():
    return SimXYStage("SIM::STAGE")


def _axis(driver, axis="x", **params):
    vi = StageAxisVI(
        {"main": driver},
        axis=axis,
        min_position_m=-5e-3,
        max_position_m=5e-3,
        speed_m_per_s=2e-3,
        **params,
    )
    vi.vi_name = f"stage_{axis}"
    return vi


class TestStageAxisVI:
    def test_is_a_stage_of_the_declared_axis(self, stage_driver):
        vi = _axis(stage_driver, "y")
        assert isinstance(vi, StageBase)
        assert vi.vi_type == "stage"
        assert vi.axis == "y"
        assert vi.setpoint_label == "y position"
        assert vi.setpoint_unit == "m"

    def test_unknown_axis_is_refused_at_construction(self, stage_driver):
        with pytest.raises(I2ASConfigError):
            _axis(stage_driver, "z")

    def test_limits_come_from_the_config(self, stage_driver):
        vi = _axis(stage_driver)
        assert vi.limit_bounds("position_m") == (-5e-3, 5e-3)
        with pytest.raises(I2ASSafetyError):
            vi.set_position(6e-3)
        assert stage_driver.get_target() == (0.0, 0.0)
        unbounded = StageAxisVI({"main": stage_driver}, axis="x")
        assert unbounded.limit_bounds("position_m") == (None, None)

    def test_construction_and_initiate_are_the_only_setup(self, stage_driver):
        vi = _axis(stage_driver)
        assert stage_driver.get_speed() == pytest.approx(1e-3)
        vi.initiate()
        assert stage_driver.get_speed() == pytest.approx(2e-3)
        assert vi.lifecycle_state() == "initiated"

    def test_a_move_is_a_ramp_that_reaches_its_target(self, stage_driver):
        vi = _axis(stage_driver)
        vi.initiate()
        assert vi.ramp_status() == "IDLE"
        vi.set_position(2e-3)
        assert vi.ramp_status() == "RAMPING"
        assert stage_driver.get_target() == (2e-3, 0.0)
        assert vi.ramp_target() == pytest.approx(2e-3)
        assert vi.ramp_setpoint() == pytest.approx(2e-3)
        assert vi.ramp_rate() == pytest.approx(2e-3 * 60.0)
        assert vi.motion_state() == "moving"
        _elapse(stage_driver, 0.5)
        vi.advance_ramp()
        assert vi.ramp_status() == "RAMPING"
        assert 0.0 < vi.ramp_value() < 2e-3
        _elapse(stage_driver, 2.0)
        vi.advance_ramp()
        assert vi.ramp_status() == "TARGET_REACHED"
        assert vi.position() == pytest.approx(2e-3)
        assert vi.motion_state() == "holding"

    def test_two_axes_share_one_driver_without_cancelling(self, stage_driver):
        x = _axis(stage_driver, "x")
        y = _axis(stage_driver, "y")
        x.start_ramp(3e-3)
        y.start_ramp(-4e-3)
        assert stage_driver.get_target() == (3e-3, -4e-3)
        _elapse(stage_driver, 5.0)
        x.advance_ramp()
        y.advance_ramp()
        assert x.ramp_status() == "TARGET_REACHED"
        assert y.ramp_status() == "TARGET_REACHED"
        assert x.position() == pytest.approx(3e-3)
        assert y.position() == pytest.approx(-4e-3)

    def test_stop_halts_the_axis_and_clears_the_ramp(self, stage_driver):
        vi = _axis(stage_driver)
        vi.set_position(4e-3)
        _elapse(stage_driver, 0.5)
        vi.stop()
        assert vi.ramp_status() == "IDLE"
        assert vi.ramp_target() is None and vi.ramp_setpoint() is None
        assert not stage_driver.is_moving("x")
        assert 0.0 < vi.position() < 4e-3

    def test_stop_ramp_leaves_the_other_axis_moving(self, stage_driver):
        x = _axis(stage_driver, "x")
        y = _axis(stage_driver, "y")
        x.start_ramp(3e-3)
        y.start_ramp(3e-3)
        x.stop_ramp()
        assert not stage_driver.is_moving("x")
        assert stage_driver.is_moving("y")

    def test_standby_stops_where_it_is(self, stage_driver):
        vi = _axis(stage_driver)
        vi.set_position(4e-3)
        _elapse(stage_driver, 0.5)
        vi.standby()
        assert vi.lifecycle_state() == "standby"
        assert not stage_driver.is_moving("x")
        assert vi.position() != pytest.approx(0.0)
        assert vi.standby_status() == "reached"

    def test_programmatic_target_beyond_travel_is_clamped_loudly(self, stage_driver, caplog):
        vi = _axis(stage_driver)
        with caplog.at_level("WARNING"):
            vi.start_ramp(9e-3)
        assert vi.ramp_target() == pytest.approx(5e-3)
        assert "clamped" in caplog.text

    def test_nominal_rate_is_the_configured_speed(self, stage_driver):
        vi = _axis(stage_driver)
        assert vi.nominal_ramp_rate() == pytest.approx(0.12)
        assert vi.ramp_rate() is None

    def test_state_snapshot_carries_position_and_motion(self, stage_driver):
        vi = _axis(stage_driver)
        state = vi.get_state()
        assert state["position"] == pytest.approx(0.0)
        assert state["motion_state"] == "holding"


class TestStageOnTheStation:
    def test_stage_vi_names_lists_the_axes_in_config_order(self):
        station = build_station(IMAGING_CONFIG)
        assert station.stage_vi_names() == ["stage_x", "stage_y"]
        assert station.get_vi("stage_x").axis == "x"
        assert station.get_vi("stage_y").axis == "y"
        assert station.system_setpoint_meta("stage_y") == ("y position", "m")

    def test_the_sim_cryostat_has_no_stage(self):
        assert build_station("i2as/configs/sim_cryostat").stage_vi_names() == []

    def test_a_stage_move_shows_in_the_ramp_status(self):
        station = build_station(IMAGING_CONFIG)
        station.process_system_targets({"stage_x": __import__("i2as.core.plan", fromlist=["Target"]).Target(1e-3)})
        status = station.get_ramp_status()["stage_x"]
        assert status["ramp_status"] == "RAMPING"
        assert status["target"] == pytest.approx(1e-3)
        assert status["setpoint"] == pytest.approx(1e-3)
        assert status["rate"] is not None and status["rate"] > 0
        assert status["no_motion_phases"] == frozenset()


# ── CameraMeasurementVI ──────────────────────────────────────────────────────


@pytest.fixture
def camera_resource():
    return f"SIM::CAM@vi_{time.time_ns()}"


@pytest.fixture
def camera_driver(camera_resource):
    return SimCamera(camera_resource)


@pytest.fixture
def world(camera_resource):
    return sim_environment.for_resource(camera_resource)


def _camera(driver, **params):
    from i2as.virtual_instruments.measurement.camera import CameraMeasurementVI

    vi = CameraMeasurementVI({"main": driver}, **params)
    vi.vi_name = "camera"
    return vi


class TestCameraMeasurementVI:
    def test_declares_one_image_block_at_the_sensor_size(self, camera_driver):
        from i2as.virtual_instruments.measurement.camera import CameraMeasurementVI

        block = CameraMeasurementVI.measurement_image_blocks["frame"]
        assert block.shape == camera_driver.get_sensor_size()
        assert block.unit == "counts"
        assert CameraMeasurementVI.measurement_data_keys == ["roi_mean_array"]
        assert set(CameraMeasurementVI.measurement_scalar_columns) == {
            "roi_mean", "roi_mean_error", "roi_std"
        }
        assert CameraMeasurementVI.selector_label

    def test_construction_is_silent_and_roi_defaults_to_the_sensor(self, camera_driver):
        vi = _camera(camera_driver)
        assert vi.roi == (0, 0, 128, 128)
        assert not camera_driver.is_armed()

    @pytest.mark.parametrize("roi", [[0, 0, 0, 1], [100, 0, 40, 40], [0, 0, 129, 1], [1, 2, 3]])
    def test_a_roi_outside_the_sensor_is_refused_at_construction(self, camera_driver, roi):
        with pytest.raises(I2ASConfigError):
            _camera(camera_driver, roi=roi)

    def test_arming_configures_and_arms_the_camera(self, camera_driver):
        vi = _camera(camera_driver, roi=[32, 32, 64, 64])
        vi.initiate_measurement(exposure_s=0.02, binning=2, frames_per_step=3)
        assert camera_driver.is_armed()
        assert camera_driver.get_exposure_s() == pytest.approx(0.02)
        assert camera_driver.get_binning() == 2
        assert camera_driver.get_roi() == (0, 0, 128, 128)
        assert vi.lifecycle_state() == "initiated"
        assert vi.data_arrays({"frames_per_step": 3}) == {"roi_mean_array": 3}

    def test_take_reading_returns_the_declared_shape(self, camera_driver):
        vi = _camera(camera_driver, roi=[32, 32, 64, 64])
        vi.initiate_measurement(exposure_s=0.01, binning=1, frames_per_step=2)
        data = vi.take_reading()
        assert set(data) == {"frame", "roi_mean_array", "roi_mean", "roi_mean_error", "roi_std"}
        frame = np.asarray(data["frame"])
        assert frame.shape == (128, 128)
        assert frame.dtype == np.float64
        assert len(data["roi_mean_array"]) == 2
        assert data["roi_mean"] == pytest.approx(np.mean(data["roi_mean_array"]))
        assert data["roi_mean_error"] >= 0.0
        # The ROI statistics are taken over the averaged frame's ROI.
        region = frame[32:96, 32:96]
        assert data["roi_mean"] == pytest.approx(region.mean(), rel=1e-3)
        assert data["roi_std"] == pytest.approx(region.std(), rel=1e-3)

    def test_a_binned_frame_is_expanded_back_onto_the_sensor_grid(self, camera_driver):
        vi = _camera(camera_driver)
        vi.initiate_measurement(exposure_s=0.01, binning=4, frames_per_step=1)
        frame = np.asarray(vi.take_reading()["frame"])
        assert frame.shape == (128, 128)
        # Every 4x4 superpixel carries one value.
        block = frame[:4, :4]
        assert np.all(block == block[0, 0])

    def test_averaging_reduces_the_noise(self, camera_driver):
        vi = _camera(camera_driver)
        vi.initiate_measurement(exposure_s=0.01, binning=1, frames_per_step=1)
        single = np.asarray(vi.take_reading()["frame"])
        vi.initiate_measurement(exposure_s=0.01, binning=1, frames_per_step=8)
        averaged = np.asarray(vi.take_reading()["frame"])
        # The illumination profile is smooth, so local roughness is noise.
        def roughness(f):
            return float(np.abs(np.diff(f[64], n=2)).mean())
        assert roughness(averaged) < 0.6 * roughness(single)

    def test_the_frame_follows_the_applied_field(self, camera_driver, world):
        vi = _camera(camera_driver, roi=[32, 32, 64, 64])
        vi.initiate_measurement(exposure_s=0.01, binning=1, frames_per_step=1)
        world.applied_field_T = -1.0
        low = vi.take_reading()["roi_mean"]
        world.applied_field_T = 1.0
        high = vi.take_reading()["roi_mean"]
        assert high > 1.5 * low

    def test_take_reading_before_arming_is_refused(self, camera_driver):
        vi = _camera(camera_driver)
        with pytest.raises(RuntimeError):
            vi.take_reading()

    def test_exposure_limit_comes_from_the_config(self, camera_driver):
        vi = _camera(camera_driver, min_exposure_s=1e-3, max_exposure_s=0.1)
        assert vi.limit_bounds("exposure_s") == (1e-3, 0.1)
        with pytest.raises(I2ASSafetyError):
            vi.initiate_measurement(exposure_s=1.0, binning=1, frames_per_step=1)
        assert not camera_driver.is_armed()

    def test_read_now_caches_the_monitored_fields(self, camera_driver):
        vi = _camera(camera_driver, roi=[32, 32, 64, 64])
        assert vi.last_roi_mean() is None and vi.last_roi_std() is None
        vi.initiate_measurement(exposure_s=0.01, binning=1, frames_per_step=1)
        vi.read_now()
        assert vi.last_roi_mean() > 0.0
        assert vi.last_roi_std() >= 0.0
        assert vi.exposure() == pytest.approx(0.01)
        assert vi.get_state()["last_roi_mean"] == pytest.approx(vi.last_roi_mean())

    def test_standby_disarms_and_forgets(self, camera_driver):
        vi = _camera(camera_driver)
        vi.initiate_measurement(exposure_s=0.01, binning=1, frames_per_step=1)
        vi.read_now()
        vi.standby()
        assert not camera_driver.is_armed()
        assert vi.last_roi_mean() is None
        assert vi.lifecycle_state() == "standby"
        with pytest.raises(RuntimeError):
            vi.take_reading()

    def test_plain_initiate_is_a_connection_check_that_arms_nothing(self, camera_driver):
        vi = _camera(camera_driver)
        vi.initiate()
        assert not camera_driver.is_armed()

    def test_the_imaging_station_registers_the_camera_as_the_measurement(self):
        station = build_station(IMAGING_CONFIG)
        assert station.measurement_vi_names() == ["camera"]
        assert station.magnet_vi_names() == ["magnet_z"]
        camera = station.get_vi("camera")
        assert camera.roi == (32, 32, 64, 64)
