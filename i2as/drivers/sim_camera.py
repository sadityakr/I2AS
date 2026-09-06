"""Simulated widefield camera driver — a magnetic domain pattern under field."""

from __future__ import annotations

import logging

import numpy as np

from i2as.core import sim_environment
from i2as.core.exceptions import (
    I2ASCommunicationError,
    I2ASInstrumentError,
)

log = logging.getLogger(__name__)

#: The binning factors the simulated sensor can read out at.
_BINNING_FACTORS = (1, 2, 4, 8)


class SimCamera:
    """Simulated widefield camera looking at a magnetic domain pattern.

    The sensor is a fixed ``SENSOR_HEIGHT_PX x SENSOR_WIDTH_PX`` grid of
    16-bit pixels. Each frame is one exposure of a sample whose magneto-
    optical contrast follows the local magnetisation ``m(x, y) = ±1``, so
    the frame is a domain image: bright where the sample is magnetised one
    way, dark the other way, with a Gaussian illumination profile and shot
    noise on top, scaled by the exposure time and saturating at the sensor's
    full well.

    Frame physics (a Preisach-style switching model)
    ------------------------------------------------
    Every pixel carries its own switching field ``H_s(x, y)``: the coercive
    field ``COERCIVE_FIELD_T`` plus spatially correlated disorder of width
    ``SWITCHING_DISORDER_T``, so weak regions nucleate first and domains grow
    from them over a range of fields rather than the whole sample flipping
    at once. On every frame the sample sees the applied field ``H`` from its
    ``SimEnvironment`` (the sim-coupling standard, ``drivers/README.md``):
    a pixel at ``m = -1`` switches to ``+1`` once ``H > +H_s``, and back to
    ``-1`` once ``H < -H_s``. Between those two fields it remembers its
    state — that memory is the hysteresis a field sweep produces in the mean
    intensity, and the loop's half-width is the coercive field. The disorder
    is drawn from a fixed seed at construction, so two instances image the
    same sample and a run is reproducible.

    Acquisition is the triggered-frame model most scientific cameras use:
    ``arm()`` prepares the sensor, ``get_frame()`` takes one triggered
    exposure, ``disarm()`` releases it. A frame requested while disarmed is
    refused — the instrument-error half of the driver error-reporting
    standard, with the code a real camera's SDK would return.

    This driver satisfies the three-rule driver contract:
    1. It is a Python class.
    2. __init__ accepts a single VISA resource string (only its optional
       ``@<environment>`` suffix is read).
    3. It is importable via i2as.drivers.sim_camera.
    """

    #: Sensor size in pixels — the frame every unbinned exposure returns.
    SENSOR_HEIGHT_PX = 128
    SENSOR_WIDTH_PX = 128

    #: The exposure range the simulated sensor accepts, in seconds.
    MIN_EXPOSURE_S = 1e-5
    MAX_EXPOSURE_S = 10.0

    #: Full-well depth: a 16-bit sensor saturates here.
    FULL_WELL_COUNTS = 65535

    #: Mean photon rate on a pixel at full illumination, in counts per second.
    PHOTON_RATE_PER_S = 4.0e5

    #: Magneto-optical contrast: the fractional brightness change between
    #: the two magnetisation states.
    CONTRAST = 0.3

    #: The sample's coercive field and the width of its local switching-field
    #: disorder, both in tesla.
    COERCIVE_FIELD_T = 0.5
    SWITCHING_DISORDER_T = 0.12

    #: Correlation length of the disorder, in pixels — sets the domain size.
    DISORDER_CORRELATION_PX = 6.0

    #: The fixed seed every instance draws its sample and its noise from.
    SEED = 20260906

    def __init__(self, resource_string: str) -> None:
        """Initialise the simulated camera on a saturated (``m = -1``) sample.

        Args:
            resource_string: VISA address (e.g. 'USB0::0x1234::CAM::INSTR').
                Only its optional ``@<environment>`` suffix is read (the
                sim-coupling standard); the address itself is ignored.
        """
        self._environment = sim_environment.for_resource(resource_string)
        self._rng = np.random.default_rng(self.SEED)

        self._exposure_s: float = 0.01
        self._binning: int = 1
        # Region of interest as (x0, y0, width, height), full sensor by default.
        self._roi: tuple[int, int, int, int] = (
            0, 0, self.SENSOR_WIDTH_PX, self.SENSOR_HEIGHT_PX
        )
        self._armed: bool = False
        self._frames_taken: int = 0

        # The sample: per-pixel switching fields (T) and magnetisation (±1),
        # born saturated negative — the reference state a field-imaging run
        # returns to before it starts.
        self._switching_field_T = self._draw_switching_fields()
        self._magnetisation = -np.ones(
            (self.SENSOR_HEIGHT_PX, self.SENSOR_WIDTH_PX), dtype=np.int8
        )
        self._illumination = self._illumination_profile()

        # Test control flags
        self._simulate_error: bool = False
        # Connection-lifecycle standard: True once close() has released
        # the session; every command then fails (see _check_error).
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Public API — settings
    # ------------------------------------------------------------------

    def set_exposure_s(self, exposure_s: float) -> None:
        """Set the exposure time of every subsequent frame.

        Verified by explicit range check, as a camera SDK does: a value
        outside ``[MIN_EXPOSURE_S, MAX_EXPOSURE_S]`` is refused and the
        exposure is left unchanged.

        Args:
            exposure_s: Exposure time in seconds.

        Raises:
            I2ASInstrumentError: ``EXPOSURE_RANGE`` if the value is
                outside the sensor's range.
        """
        self._check_error()
        value = float(exposure_s)
        if not self.MIN_EXPOSURE_S <= value <= self.MAX_EXPOSURE_S:
            self._refuse(
                f"set_exposure_s({exposure_s!r})",
                "EXPOSURE_RANGE",
                f"exposure must be within [{self.MIN_EXPOSURE_S:g}, "
                f"{self.MAX_EXPOSURE_S:g}] s",
            )
        self._exposure_s = value

    def get_exposure_s(self) -> float:
        """Return the exposure time in seconds."""
        self._check_error()
        return self._exposure_s

    def set_binning(self, binning: int) -> None:
        """Set the on-sensor binning factor.

        A binned readout sums ``binning x binning`` pixels into one, so the
        frame shrinks to ``(height / binning, width / binning)`` and each
        value is the sum over its superpixel. Refused for a factor the sensor
        does not support.

        Args:
            binning: One of ``1, 2, 4, 8``.

        Raises:
            I2ASInstrumentError: ``BINNING_UNSUPPORTED`` for any other
                factor.
        """
        self._check_error()
        if binning not in _BINNING_FACTORS:
            self._refuse(
                f"set_binning({binning!r})",
                "BINNING_UNSUPPORTED",
                f"binning must be one of {_BINNING_FACTORS}",
            )
        self._binning = int(binning)

    def get_binning(self) -> int:
        """Return the binning factor."""
        self._check_error()
        return self._binning

    def set_roi(self, x0: int, y0: int, width: int, height: int) -> None:
        """Set the readout region of interest, in unbinned sensor pixels.

        Refused when the rectangle is empty or leaves the sensor; the ROI is
        left unchanged.

        Args:
            x0: Left column of the region.
            y0: Top row of the region.
            width: Region width in pixels (> 0).
            height: Region height in pixels (> 0).

        Raises:
            I2ASInstrumentError: ``ROI_RANGE`` if the rectangle does not
                fit the sensor.
        """
        self._check_error()
        x0, y0, width, height = int(x0), int(y0), int(width), int(height)
        if (
            width <= 0
            or height <= 0
            or x0 < 0
            or y0 < 0
            or x0 + width > self.SENSOR_WIDTH_PX
            or y0 + height > self.SENSOR_HEIGHT_PX
        ):
            self._refuse(
                f"set_roi({x0}, {y0}, {width}, {height})",
                "ROI_RANGE",
                f"region must lie within the {self.SENSOR_WIDTH_PX}x"
                f"{self.SENSOR_HEIGHT_PX} sensor",
            )
        self._roi = (x0, y0, width, height)

    def get_roi(self) -> tuple[int, int, int, int]:
        """Return the region of interest as ``(x0, y0, width, height)``."""
        self._check_error()
        return self._roi

    def get_sensor_size(self) -> tuple[int, int]:
        """Return the full sensor size as ``(height_px, width_px)``."""
        self._check_error()
        return (self.SENSOR_HEIGHT_PX, self.SENSOR_WIDTH_PX)

    # ------------------------------------------------------------------
    # Public API — acquisition
    # ------------------------------------------------------------------

    def arm(self) -> None:
        """Prepare the sensor for triggered frames. Idempotent."""
        self._check_error()
        self._armed = True

    def disarm(self) -> None:
        """Release the sensor; frames are refused until the next ``arm()``."""
        self._check_error()
        self._armed = False

    def is_armed(self) -> bool:
        """Return True while the sensor accepts triggered frames."""
        self._check_error()
        return self._armed

    def get_frame(self) -> np.ndarray:
        """Take one triggered exposure and return it.

        The sample is first advanced to the applied field the environment
        reports (domains switch where the field has crossed their switching
        field), then imaged: contrast times illumination times exposure,
        Poisson shot noise, binned and cropped to the ROI, clipped to the
        full well.

        Returns:
            A ``uint16`` array of shape ``(roi_height / binning,
            roi_width / binning)`` — ``(128, 128)`` unbinned on the full
            sensor.

        Raises:
            I2ASInstrumentError: ``NOT_ARMED`` if ``arm()`` has not been
                called.
        """
        self._check_error()
        if not self._armed:
            self._refuse("get_frame()", "NOT_ARMED", "call arm() before taking a frame")
        self._advance_field(self._environment.applied_field_T)

        expected = (
            self.PHOTON_RATE_PER_S
            * self._exposure_s
            * self._illumination
            * (1.0 + self.CONTRAST * self._magnetisation)
        )
        x0, y0, width, height = self._roi
        region = expected[y0 : y0 + height, x0 : x0 + width]
        b = self._binning
        rows, cols = (height // b) * b, (width // b) * b
        binned = region[:rows, :cols].reshape(rows // b, b, cols // b, b).sum(axis=(1, 3))
        noisy = self._rng.poisson(binned).astype(np.float64)
        self._frames_taken += 1
        return np.clip(noisy, 0, self.FULL_WELL_COUNTS).astype(np.uint16)

    def get_idn(self) -> str:
        """Return simulated identification string."""
        self._check_error()
        return "I2AS,SIMCAMERA,SIM,1.0"

    # ------------------------------------------------------------------
    # Sample physics
    # ------------------------------------------------------------------

    def _draw_switching_fields(self) -> np.ndarray:
        """Draw the per-pixel switching fields once, from the fixed seed.

        White noise is smoothed with a Gaussian kernel of width
        ``DISORDER_CORRELATION_PX`` (separable, via FFT) and rescaled to
        ``SWITCHING_DISORDER_T``, so neighbouring pixels share a switching
        field and domains nucleate and grow as connected regions.

        Returns:
            The switching field of every pixel, in tesla, never below a tenth
            of the coercive field.
        """
        shape = (self.SENSOR_HEIGHT_PX, self.SENSOR_WIDTH_PX)
        noise = self._rng.standard_normal(shape)
        ky = np.fft.fftfreq(shape[0])
        kx = np.fft.fftfreq(shape[1])
        sigma = self.DISORDER_CORRELATION_PX
        kernel = np.exp(-2.0 * (np.pi * sigma) ** 2 * (ky[:, None] ** 2 + kx[None, :] ** 2))
        smooth = np.real(np.fft.ifft2(np.fft.fft2(noise) * kernel))
        smooth /= max(float(np.std(smooth)), 1e-12)
        fields = self.COERCIVE_FIELD_T + self.SWITCHING_DISORDER_T * smooth
        return np.maximum(fields, 0.1 * self.COERCIVE_FIELD_T)

    def _illumination_profile(self) -> np.ndarray:
        """Return the Gaussian vignetting profile of the illumination, peak 1."""
        rows = np.arange(self.SENSOR_HEIGHT_PX) - (self.SENSOR_HEIGHT_PX - 1) / 2.0
        cols = np.arange(self.SENSOR_WIDTH_PX) - (self.SENSOR_WIDTH_PX - 1) / 2.0
        radius = max(self.SENSOR_HEIGHT_PX, self.SENSOR_WIDTH_PX)
        return np.exp(-(rows[:, None] ** 2 + cols[None, :] ** 2) / (2.0 * radius**2))

    def _advance_field(self, field_T: float) -> None:
        """Switch every domain whose switching field the applied field crossed.

        Args:
            field_T: The applied field at the sample, in tesla.
        """
        up = (field_T > self._switching_field_T) & (self._magnetisation < 0)
        down = (field_T < -self._switching_field_T) & (self._magnetisation > 0)
        self._magnetisation[up] = 1
        self._magnetisation[down] = -1

    def mean_magnetisation(self) -> float:
        """Return the sample's mean magnetisation in ``[-1, 1]`` (a test hook).

        Reads the sample state without taking a frame and without advancing
        the field; the loop a test sees through ``get_frame()`` is this
        quantity seen through contrast, illumination and noise.
        """
        return float(np.mean(self._magnetisation))

    # ------------------------------------------------------------------
    # Safe state (the safe-shutdown standard)
    # ------------------------------------------------------------------

    def safe_shutdown(self) -> None:
        """Disarm the sensor; idempotent, never raises.

        Safe idle for a camera is *not acquiring*: a disarmed sensor
        triggers nothing and drives nothing. Exposure, binning and ROI are
        settings, not hazards, and are left alone.
        """
        log.info("SimCamera: safe shutdown — sensor disarmed.")
        self._armed = False

    def _is_in_safe_state(self) -> bool:
        """Return True when the sensor is disarmed."""
        return not self._armed

    # ------------------------------------------------------------------
    # Connection lifecycle (the connection-lifecycle standard)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the simulated session; the camera is left untouched.

        Idempotent and never raises. Afterwards every command — including
        ``get_idn()`` — raises ``I2ASCommunicationError`` via
        :meth:`_check_error`, modelling a released session so a
        use-after-disconnect bug fails in a test instead of on hardware.
        """
        self._closed = True

    def _refuse(self, context: str, code: str, reason: str) -> None:
        """Raise the typed refusal a real camera SDK would report.

        Args:
            context: The driver call that was refused.
            code: The instrument's own error code.
            reason: Why, in the instrument's words.

        Raises:
            I2ASInstrumentError: Always.
        """
        raise I2ASInstrumentError(
            f"Simulated camera refused {context}: {code} — {reason}",
            code=code,
            instrument_message=f"{code}: {reason}",
            context=context,
            vi_name="SimCamera",
        )

    def _check_error(self) -> None:
        """Raise I2ASCommunicationError if error simulation is active."""
        if self._closed:
            raise I2ASCommunicationError(
                "SimCamera: the session is closed — the driver was "
                "disconnected from I2AS",
                vi_name="SimCamera",
            )
        if self._simulate_error:
            raise I2ASCommunicationError(
                "Simulated communication error on camera",
                vi_name="SimCamera",
            )
