"""Layer-0 behaviour tests for the imaging sims: environment, camera, XY stage.

The driver contract itself (one class, one resource argument, ``get_idn``,
``close``, ``safe_shutdown``) is held by ``tests/test_conformance.py``; these
tests hold the physics and the refusals each sim models.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from i2as.core import sim_environment
from i2as.core.exceptions import I2ASCommunicationError, I2ASInstrumentError
from i2as.drivers.sim_camera import SimCamera
from i2as.drivers.sim_oxford_ips120 import SimOxfordIPS120
from i2as.drivers.sim_xy_stage import SimXYStage


def _unique(prefix: str) -> str:
    """Return a resource string joining a fresh shared environment."""
    return f"SIM::{prefix}@{prefix}_{time.time_ns()}"


# ── The sim environment (the sim-coupling standard) ──────────────────────────


class TestSimEnvironment:
    def test_suffix_joins_one_shared_environment(self):
        name = f"world_{time.time_ns()}"
        a = sim_environment.for_resource(f"SIM::A@{name}")
        b = sim_environment.for_resource(f"SIM::B@{name}")
        assert a is b
        assert a is sim_environment.get(name)

    def test_no_suffix_means_a_private_environment(self):
        a = sim_environment.for_resource("SIM::A")
        b = sim_environment.for_resource("SIM::A")
        assert a is not b
        assert a.name == ""

    def test_get_refuses_an_empty_name(self):
        with pytest.raises(ValueError):
            sim_environment.get("")

    def test_field_is_current_over_the_coil_constant(self):
        env = sim_environment.SimEnvironment()
        env.psu_current_A = 5.0
        assert env.applied_field_T == pytest.approx(5.0 / sim_environment.DEFAULT_AMPERES_PER_TESLA)
        env.applied_field_T = -1.0
        assert env.psu_current_A == pytest.approx(-sim_environment.DEFAULT_AMPERES_PER_TESLA)

    def test_the_sim_psu_publishes_its_current(self):
        resource = _unique("IPS")
        psu = SimOxfordIPS120(resource)
        env = sim_environment.for_resource(resource)
        assert env.psu_current_A == 0.0
        psu.set_ramp_rate(600.0)
        psu.set_current_setpoint(3.0)
        psu._last_update = time.time() - 1.0
        assert psu.get_current() == pytest.approx(3.0)
        assert env.psu_current_A == pytest.approx(3.0)
        assert env.applied_field_T == pytest.approx(0.3)

    def test_a_fresh_psu_resets_a_shared_world_to_zero(self):
        resource = _unique("IPS")
        env = sim_environment.for_resource(resource)
        env.psu_current_A = 42.0
        SimOxfordIPS120(resource)
        assert env.psu_current_A == 0.0

    def test_two_private_psus_do_not_share_a_world(self):
        first = SimOxfordIPS120("SIM::IPS_A")
        second = SimOxfordIPS120("SIM::IPS_B")
        first._current = 7.0
        first._publish_current()
        assert second._environment.psu_current_A == 0.0

    def test_shipped_sim_configs_match_the_environment_coil_constant(self):
        """The magnet VI's A/T and the sim world's A/T are one number."""
        from pathlib import Path

        from ruamel.yaml import YAML

        import i2as

        configs = Path(i2as.__file__).parent / "configs"
        checked = 0
        for devices in configs.glob("*/devices.yaml"):
            with devices.open(encoding="utf-8") as handle:
                config = YAML().load(handle)
            for vi_cfg in (config.get("virtual_instruments") or {}).values():
                params = vi_cfg.get("init_params") or {}
                if "amperes_per_tesla" in params:
                    assert float(params["amperes_per_tesla"]) == pytest.approx(
                        sim_environment.DEFAULT_AMPERES_PER_TESLA
                    ), f"{devices}: amperes_per_tesla differs from the sim world's"
                    checked += 1
        assert checked >= 1


# ── The sim camera ───────────────────────────────────────────────────────────


@pytest.fixture
def camera():
    resource = _unique("CAM")
    cam = SimCamera(resource)
    cam.arm()
    return cam


class TestSimCamera:
    def test_frame_is_uint16_at_the_sensor_size(self, camera):
        frame = camera.get_frame()
        assert frame.dtype == np.uint16
        assert frame.shape == (SimCamera.SENSOR_HEIGHT_PX, SimCamera.SENSOR_WIDTH_PX)
        assert camera.get_sensor_size() == frame.shape

    def test_default_settings(self, camera):
        assert camera.get_exposure_s() == pytest.approx(0.01)
        assert camera.get_binning() == 1
        assert camera.get_roi() == (0, 0, 128, 128)

    def test_frame_is_refused_while_disarmed(self):
        cam = SimCamera("SIM::CAM")
        with pytest.raises(I2ASInstrumentError) as info:
            cam.get_frame()
        assert info.value.code == "NOT_ARMED"
        assert "get_frame" in info.value.context
        cam.arm()
        assert cam.is_armed()
        cam.get_frame()
        cam.disarm()
        assert not cam.is_armed()

    def test_exposure_scales_the_counts(self, camera):
        camera.set_exposure_s(0.01)
        short = float(camera.get_frame().mean())
        camera.set_exposure_s(0.04)
        long = float(camera.get_frame().mean())
        assert long == pytest.approx(4.0 * short, rel=0.05)

    def test_a_long_exposure_saturates_at_the_full_well(self, camera):
        camera.set_exposure_s(1.0)
        frame = camera.get_frame()
        assert frame.max() == SimCamera.FULL_WELL_COUNTS

    def test_exposure_out_of_range_is_refused_and_unchanged(self, camera):
        for bad in (0.0, -1.0, 100.0):
            with pytest.raises(I2ASInstrumentError) as info:
                camera.set_exposure_s(bad)
            assert info.value.code == "EXPOSURE_RANGE"
        assert camera.get_exposure_s() == pytest.approx(0.01)

    def test_binning_sums_superpixels(self, camera):
        camera.set_binning(4)
        frame = camera.get_frame()
        assert frame.shape == (32, 32)
        camera.set_binning(1)
        unbinned = camera.get_frame()
        # A summed 4x4 superpixel holds about sixteen unbinned pixels' counts.
        assert frame.mean() == pytest.approx(16.0 * unbinned.mean(), rel=0.05)

    def test_unsupported_binning_is_refused(self, camera):
        with pytest.raises(I2ASInstrumentError) as info:
            camera.set_binning(3)
        assert info.value.code == "BINNING_UNSUPPORTED"
        assert camera.get_binning() == 1

    def test_roi_crops_the_frame(self, camera):
        camera.set_roi(10, 20, 30, 40)
        assert camera.get_roi() == (10, 20, 30, 40)
        assert camera.get_frame().shape == (40, 30)

    @pytest.mark.parametrize("roi", [(0, 0, 0, 10), (-1, 0, 10, 10), (120, 0, 10, 10), (0, 120, 10, 9)])
    def test_roi_outside_the_sensor_is_refused(self, camera, roi):
        with pytest.raises(I2ASInstrumentError) as info:
            camera.set_roi(*roi)
        assert info.value.code == "ROI_RANGE"
        assert camera.get_roi() == (0, 0, 128, 128)

    def test_two_instances_image_the_same_sample(self):
        first = SimCamera("SIM::CAM_A")
        second = SimCamera("SIM::CAM_B")
        np.testing.assert_array_equal(first._switching_field_T, second._switching_field_T)

    def test_the_domain_pattern_is_spatially_correlated(self, camera):
        fields = camera._switching_field_T
        neighbour = np.corrcoef(fields[:, :-1].ravel(), fields[:, 1:].ravel())[0, 1]
        assert neighbour > 0.9
        assert fields.std() == pytest.approx(SimCamera.SWITCHING_DISORDER_T, rel=0.2)

    def test_a_field_sweep_produces_a_hysteresis_loop(self):
        """Mean intensity follows the field with memory: a loop, not a line."""
        resource = _unique("CAM")
        cam = SimCamera(resource)
        env = sim_environment.for_resource(resource)
        cam.arm()

        def mean_at(field_T: float) -> float:
            env.applied_field_T = field_T
            return float(cam.get_frame().mean())

        up = {h: mean_at(h) for h in np.linspace(-1.0, 1.0, 21)}
        down = {h: mean_at(h) for h in np.linspace(1.0, -1.0, 21)}

        # Saturated at both ends, and brighter at positive saturation.
        assert up[1.0] > 1.5 * up[-1.0]
        assert cam.mean_magnetisation() == pytest.approx(-1.0)
        # The loop is open: the two branches differ at zero field ...
        assert down[0.0] > 1.5 * up[0.0]
        # ... and domains switch over a range of fields, not at one value.
        rising = [h for h in up if up[-1.0] * 1.05 < up[h] < up[1.0] * 0.95]
        assert len(rising) >= 3
        # Switching happens on the far side of zero on both branches.
        assert min(rising) > 0.0
        falling = [h for h in down if down[-1.0] * 1.05 < down[h] < down[1.0] * 0.95]
        assert max(falling) < 0.0

    def test_frames_visibly_change_across_the_switching_field(self):
        resource = _unique("CAM")
        cam = SimCamera(resource)
        env = sim_environment.for_resource(resource)
        cam.arm()
        env.applied_field_T = -1.0
        reference = cam.get_frame().astype(float)
        env.applied_field_T = 0.5
        mid = cam.get_frame().astype(float)
        difference = np.abs(mid - reference)
        # Some pixels switched (bright difference), some did not (only noise).
        assert (difference > 0.2 * reference.mean()).mean() > 0.2
        assert (difference < 0.2 * reference.mean()).mean() > 0.2

    def test_closed_camera_refuses_every_command(self, camera):
        camera.close()
        with pytest.raises(I2ASCommunicationError):
            camera.get_frame()
        with pytest.raises(I2ASCommunicationError):
            camera.get_idn()

    def test_safe_shutdown_disarms(self, camera):
        assert camera.is_armed()
        camera.safe_shutdown()
        assert not camera.is_armed()
        assert camera._is_in_safe_state()


# ── The sim XY stage ─────────────────────────────────────────────────────────


@pytest.fixture
def stage():
    return SimXYStage("SIM::STAGE")


def _elapse(driver: SimXYStage, seconds: float) -> None:
    """Pretend *seconds* of wall-clock time passed since the last update."""
    driver._last_update = time.time() - seconds


class TestSimXYStage:
    def test_starts_at_rest_at_the_origin(self, stage):
        assert stage.get_position() == (0.0, 0.0)
        assert stage.get_target() == (0.0, 0.0)
        assert not stage.is_moving()
        assert stage.get_speed() == pytest.approx(1e-3)

    def test_move_advances_at_the_set_speed(self, stage):
        stage.set_speed(2e-3)
        stage.move_to(x_m=4e-3, y_m=-1e-3)
        assert stage.is_moving()
        _elapse(stage, 0.5)
        x, y = stage.get_position()
        assert x == pytest.approx(1e-3, rel=0.02)
        assert y == pytest.approx(-1e-3)
        assert stage.is_moving("x") and not stage.is_moving("y")
        _elapse(stage, 5.0)
        assert stage.get_position() == pytest.approx((4e-3, -1e-3))
        assert not stage.is_moving()

    def test_an_omitted_axis_keeps_its_own_move(self, stage):
        stage.move_to(x_m=3e-3)
        stage.move_to(y_m=2e-3)
        assert stage.get_target() == (3e-3, 2e-3)

    def test_out_of_travel_is_refused_and_nothing_moves(self, stage):
        with pytest.raises(I2ASInstrumentError) as info:
            stage.move_to(x_m=1e-3, y_m=SimXYStage.TRAVEL_MAX_M + 1e-3)
        assert info.value.code == "TRAVEL_LIMIT"
        assert "y_m" in info.value.context
        assert stage.get_target() == (0.0, 0.0)

    def test_speed_out_of_range_is_refused(self, stage):
        with pytest.raises(I2ASInstrumentError) as info:
            stage.set_speed(1.0)
        assert info.value.code == "SPEED_RANGE"
        assert stage.get_speed() == pytest.approx(1e-3)

    def test_unknown_axis_is_a_programming_error(self, stage):
        with pytest.raises(ValueError):
            stage.is_moving("z")

    def test_stop_pins_the_target_where_the_axis_is(self, stage):
        stage.move_to(x_m=5e-3, y_m=5e-3)
        _elapse(stage, 1.0)
        stage.stop("x")
        x, _y = stage.get_position()
        assert x == pytest.approx(1e-3, rel=0.02)
        assert not stage.is_moving("x") and stage.is_moving("y")
        stage.stop()
        assert not stage.is_moving()

    def test_safe_shutdown_stops_both_axes(self, stage):
        stage.move_to(x_m=5e-3, y_m=-5e-3)
        stage.safe_shutdown()
        assert not stage.is_moving()
        assert stage._is_in_safe_state()

    def test_closed_stage_refuses_every_command(self, stage):
        stage.close()
        with pytest.raises(I2ASCommunicationError):
            stage.get_position()
