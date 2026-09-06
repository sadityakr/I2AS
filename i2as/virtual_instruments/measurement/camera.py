"""CameraMeasurementVI — widefield imaging as a measurement method (one frame per point)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from i2as.core.decorators import control, monitored
from i2as.core.exceptions import I2ASConfigError
from i2as.core.plan import ImageBlock, ParamSpec
from i2as.virtual_instruments.base import MeasurementInstrumentBase

_NOT_INITIATED = object()

#: The sensor grid every frame is stored on, in pixels. Fixed on the class
#: because an image block is declared once per VI (the image-block
#: standard); the camera this VI drives must read out a sensor of this size.
FRAME_HEIGHT_PX = 128
FRAME_WIDTH_PX = 128


class CameraMeasurementVI(MeasurementInstrumentBase):
    """Measurement method for a widefield camera: one averaged frame per point.

    The shipped example of the **image-block standard** (see
    ``MeasurementInstrumentBase``): each reading is a ``(128, 128)`` frame
    stored under the image block ``frame``, plus the scalar columns the
    live plot and the generic sweep recipe need — the mean and spatial
    standard deviation of the frame over a config-declared region of
    interest — so a field sweep shows a hysteresis loop of ``roi_mean``
    while the frames go to disk for the image-stack recipe.

    Workflow::

        vi.initiate_measurement(exposure_s=0.01, binning=1, frames_per_step=4)
        data = vi.take_reading()
        # data = {"frame": ndarray (128, 128), "roi_mean_array": [m1, m2, m3, m4],
        #         "roi_mean": float, "roi_mean_error": float, "roi_std": float}

    **The frame keeps the sensor grid.** A binned readout returns a smaller
    frame whose every value is the sum over a ``binning x binning``
    superpixel; the VI expands it back onto the declared sensor grid (each
    superpixel repeated), so a run's dataset shape never depends on a knob
    and every pixel of a stored frame is "counts in the superpixel it lies
    in". Frames are averaged over ``frames_per_step`` exposures before the
    ROI statistics are taken; ``roi_mean_array`` holds the per-exposure ROI
    means, so its mean/error triple is the mean/error/array convention
    applied to the exposures.

    Bench-testing from the front panel uses ``read_now()`` — one manual
    reading at the settings already armed, surfaced through the
    ``last_roi_mean`` / ``last_roi_std`` monitored fields.

    Driver contract
    ---------------
    The ``"main"`` driver must implement:
    * ``set_exposure_s(float)`` / ``get_exposure_s() -> float``
    * ``set_binning(int)``
    * ``set_roi(x0, y0, width, height)`` — in unbinned sensor pixels
    * ``arm()`` / ``disarm()``
    * ``get_frame() -> ndarray`` — one triggered exposure, ``(rows, cols)``
    * ``get_idn() -> str``
    """

    display_label: str = "widefield imaging"
    selector_label: ClassVar[str] = "Widefield camera"

    # Control-validation standard (see BaseVirtualInstrument): the exposure
    # is bounded by the setup's own camera range, read in __init__ from
    # min_exposure_s / max_exposure_s (absent means unbounded on that side).
    # Binning is an enumerated readout mode and frames_per_step a count;
    # neither is a physical quantity a range could bound.
    control_limits: ClassVar[dict[str, dict[str, str]]] = {
        "initiate_measurement": {"exposure_s": "exposure_s"},
    }

    measurement_parameters: ClassVar[dict[str, ParamSpec]] = {
        "exposure_s": ParamSpec(
            type=float,
            default=0.01,
            unit="s",
            description="Exposure time of every frame",
        ),
        "binning": ParamSpec(
            type=int,
            default=1,
            choices={"1x1": 1, "2x2": 2, "4x4": 4},
            description="On-sensor binning; the stored frame keeps the sensor grid",
        ),
        "frames_per_step": ParamSpec(
            type=int,
            default=1,
            min=1,
            description="Exposures averaged into the frame at each point",
        ),
    }

    _ARRAY_KEYS, _SCALAR_COLUMNS = MeasurementInstrumentBase.quantity_columns("roi_mean")
    measurement_data_keys: ClassVar[list[str]] = _ARRAY_KEYS
    measurement_scalar_columns: ClassVar[dict[str, str]] = {
        **_SCALAR_COLUMNS,
        # Spatial standard deviation over the ROI of the averaged frame: the
        # domain contrast at this point, not a per-exposure spread.
        "roi_std": "float",
    }
    measurement_image_blocks: ClassVar[dict[str, ImageBlock]] = {
        "frame": ImageBlock(
            height_px=FRAME_HEIGHT_PX,
            width_px=FRAME_WIDTH_PX,
            unit="counts",
            description=(
                "Widefield frame over the full sensor, averaged over "
                "frames_per_step exposures; a binned readout is expanded back "
                "onto the sensor grid"
            ),
        ),
    }

    def __init__(self, drivers: dict[str, object], **init_params: Any) -> None:
        """Bind the camera and read the setup's ROI and exposure limits.

        Args:
            drivers: ``{"main": <camera driver>}``.
            **init_params: ``roi`` — ``[x0, y0, width, height]`` in sensor
                pixels, the region the scalar columns are computed over
                (default: the whole sensor); ``min_exposure_s`` /
                ``max_exposure_s`` — the exposure range this setup's camera
                accepts (absent means unbounded on that side).

        Raises:
            I2ASConfigError: If ``roi`` is not four integers describing a
                non-empty rectangle inside the sensor.
        """
        super().__init__(drivers, **init_params)
        self._camera = drivers["main"]

        roi = init_params.get("roi", [0, 0, FRAME_WIDTH_PX, FRAME_HEIGHT_PX])
        self._roi = self._validate_roi(roi)

        lo = init_params.get("min_exposure_s")
        hi = init_params.get("max_exposure_s")
        self._limits["exposure_s"] = (
            float(lo) if lo is not None else None,
            float(hi) if hi is not None else None,
        )

        self._exposure_s: object = _NOT_INITIATED
        self._binning: int = 1
        self._frames_per_step: int = 1
        # Cache of the last read_now() datapoint, read by the monitored
        # fields below. None until the first manual read.
        self._last_reading: dict[str, Any] | None = None

    @staticmethod
    def _validate_roi(roi: Any) -> tuple[int, int, int, int]:
        """Check a config ROI against the declared frame.

        Args:
            roi: The ``roi`` init param.

        Returns:
            ``(x0, y0, width, height)`` as ints.

        Raises:
            I2ASConfigError: If it is not a non-empty rectangle inside
                the frame.
        """
        try:
            x0, y0, width, height = (int(v) for v in roi)
        except (TypeError, ValueError) as exc:
            raise I2ASConfigError(
                f"CameraMeasurementVI: roi must be [x0, y0, width, height], got {roi!r}"
            ) from exc
        if (
            width <= 0
            or height <= 0
            or x0 < 0
            or y0 < 0
            or x0 + width > FRAME_WIDTH_PX
            or y0 + height > FRAME_HEIGHT_PX
        ):
            raise I2ASConfigError(
                f"CameraMeasurementVI: roi {list(roi)!r} must be a non-empty "
                f"rectangle inside the {FRAME_WIDTH_PX}x{FRAME_HEIGHT_PX} frame"
            )
        return (x0, y0, width, height)

    @property
    def roi(self) -> tuple[int, int, int, int]:
        """The region of interest as ``(x0, y0, width, height)`` in sensor pixels."""
        return self._roi

    # ------------------------------------------------------------------
    # MeasurementInstrumentBase implementation
    # ------------------------------------------------------------------

    def data_arrays(self, params: Mapping[str, Any]) -> dict[str, int]:
        """Return ``{"roi_mean_array": frames_per_step}``.

        Args:
            params: Parameter mapping containing ``frames_per_step``.

        Returns:
            Per-point length of the per-exposure ROI-mean array.
        """
        return {"roi_mean_array": int(params["frames_per_step"])}

    # panel=False: arming is a deliberate act — reachable from the front
    # panel and from procedures, never from the compact monitor card.
    @control(panel=False, action_class="run_control")
    def initiate_measurement(
        self,
        exposure_s: float = 0.01,
        binning: int = 1,
        frames_per_step: int = 1,
    ) -> None:
        """Configure and arm the camera.

        Asserts the whole readout mode — exposure, binning AND the full
        sensor ROI — rather than assuming whatever the camera was left in
        (the shared-instrument mode discipline), so the stored frame always
        covers the declared sensor grid.

        Args:
            exposure_s: Exposure time of every frame, in seconds.
            binning: On-sensor binning factor.
            frames_per_step: Exposures averaged per datapoint.
        """
        self._exposure_s = float(exposure_s)
        self._binning = int(binning)
        self._frames_per_step = max(1, int(frames_per_step))

        camera = self._camera  # type: ignore[attr-defined]
        camera.set_exposure_s(self._exposure_s)
        camera.set_binning(self._binning)
        camera.set_roi(0, 0, FRAME_WIDTH_PX, FRAME_HEIGHT_PX)
        camera.arm()

    def take_reading(self) -> dict[str, Any]:
        """Take ``frames_per_step`` exposures and return the averaged frame.

        Returns:
            ``frame`` — the ``(128, 128)`` float64 frame averaged over the
            exposures (a binned readout expanded onto the sensor grid);
            ``roi_mean_array`` — each exposure's mean over the ROI;
            ``roi_mean`` / ``roi_mean_error`` — their mean and SEM;
            ``roi_std`` — the spatial standard deviation of the averaged
            frame over the ROI.

        Raises:
            RuntimeError: If ``initiate_measurement()`` has not been called.
        """
        if self._exposure_s is _NOT_INITIATED:
            raise RuntimeError("initiate_measurement() must be called before take_reading().")
        camera = self._camera  # type: ignore[attr-defined]
        x0, y0, width, height = self._roi

        frames: list[np.ndarray] = []
        roi_means: list[float] = []
        for _ in range(self._frames_per_step):
            frame = self._on_sensor_grid(np.asarray(camera.get_frame(), dtype=np.float64))
            frames.append(frame)
            roi_means.append(float(frame[y0 : y0 + height, x0 : x0 + width].mean()))

        averaged = np.mean(np.stack(frames), axis=0)
        region = averaged[y0 : y0 + height, x0 : x0 + width]
        mean, error = self.mean_and_sem(roi_means)
        return {
            "frame": averaged,
            "roi_mean_array": roi_means,
            "roi_mean": mean,
            "roi_mean_error": error,
            "roi_std": float(region.std()),
        }

    def _on_sensor_grid(self, frame: np.ndarray) -> np.ndarray:
        """Expand a binned frame back onto the declared sensor grid.

        Args:
            frame: The frame as read, ``(rows, cols)``.

        Returns:
            A ``(FRAME_HEIGHT_PX, FRAME_WIDTH_PX)`` frame: the input itself
            when it already has that shape, otherwise each pixel repeated
            ``binning`` times along both axes (and cropped or NaN-padded if
            the camera returned something else again).

        Raises:
            ValueError: If the frame is not two-dimensional.
        """
        if frame.ndim != 2:
            raise ValueError(f"camera returned a frame of shape {frame.shape}, expected 2-D")
        target = (FRAME_HEIGHT_PX, FRAME_WIDTH_PX)
        if frame.shape == target:
            return frame
        expanded = np.repeat(np.repeat(frame, self._binning, axis=0), self._binning, axis=1)
        if expanded.shape == target:
            return expanded
        padded = np.full(target, np.nan)
        rows = min(target[0], expanded.shape[0])
        cols = min(target[1], expanded.shape[1])
        padded[:rows, :cols] = expanded[:rows, :cols]
        return padded

    # ------------------------------------------------------------------
    # Monitored fields and the manual bench read
    # ------------------------------------------------------------------

    @monitored(unit="s", description="Exposure time set on the camera")
    def exposure(self) -> float:
        """Return the camera's exposure time in seconds."""
        return float(self._camera.get_exposure_s())  # type: ignore[attr-defined]

    @monitored(unit="counts", description="ROI mean of the last manual read")
    def last_roi_mean(self) -> float | None:
        """Return the ROI mean of the last ``read_now()``, or None.

        Returns:
            The mean in counts, or ``None`` before the first manual read.
        """
        if self._last_reading is None:
            return None
        return float(self._last_reading["roi_mean"])

    @monitored(unit="counts", description="ROI spatial standard deviation of the last manual read")
    def last_roi_std(self) -> float | None:
        """Return the ROI standard deviation of the last ``read_now()``, or None.

        Returns:
            The standard deviation in counts, or ``None`` before the first
            manual read.
        """
        if self._last_reading is None:
            return None
        return float(self._last_reading["roi_std"])

    # panel=False: a bench check belongs in the instrument front panel, not
    # on the compact monitor card. action_class="read": it observes at the
    # settings already armed and commands nothing new.
    @control(panel=False, action_class="read")
    def read_now(self) -> None:
        """Take one manual reading and cache it for the monitored fields.

        Raises:
            RuntimeError: If ``initiate_measurement()`` has not been called
                first.
        """
        self._last_reading = self.take_reading()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def standby(self) -> None:
        """Disarm the camera and reset the armed state."""
        self._camera.disarm()  # type: ignore[attr-defined]
        self._exposure_s = _NOT_INITIATED
        self._last_reading = None
