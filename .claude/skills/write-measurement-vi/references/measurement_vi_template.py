# ---
# description: |
#   TEMPLATE — copy into i2as/virtual_instruments/measurement/, rename
#   the module and class, then fill in every TODO. Delete this header block
#   and write a real one describing what YOUR instrument/VI actually does
#   (see any shipped measurement VI for the expected structure: description,
#   entry_point, dependencies, input, process, output, last_updated).
# ---

"""<InstrumentName>MeasurementVI — one-line summary of the physical quantity measured."""

from __future__ import annotations

import math
from typing import Any, ClassVar

from i2as.core.decorators import control
from i2as.core.exceptions import I2ASConfigError
from i2as.core.plan import ParamSpec
from i2as.virtual_instruments.base import MeasurementInstrumentBase


class InstrumentNameMeasurementVI(MeasurementInstrumentBase):
    """TODO: one paragraph — what this measures, which driver role(s) it uses,
    and the workflow (initiate_measurement(...) then take_reading()).

    Driver contract
    ---------------
    ``"main"`` (or whatever role name you chose — see the write-measurement-vi
    skill's role-collision note if this VI shares a driver type with an
    existing role) driver must implement: TODO list the methods this VI
    calls, e.g. ``set_x(...)``, ``get_idn() -> str``.
    """

    display_label: str = "TODO short status-line label, e.g. 'resistance'"
    selector_label: ClassVar[str] = "TODO short GUI drop-down label"

    # --- Self-description: derive keys from quantity_columns(), never by hand ---
    _ARRAY_KEYS, _SCALAR_COLUMNS = MeasurementInstrumentBase.quantity_columns(
        "quantity_a_unit",  # TODO real quantity names, e.g. "voltage_V", "res_a_ohm"
        # "quantity_b_unit",
    )
    measurement_data_keys: ClassVar[list[str]] = _ARRAY_KEYS
    measurement_scalar_columns: ClassVar[dict[str, str]] = _SCALAR_COLUMNS

    measurement_parameters: ClassVar[dict[str, ParamSpec]] = {
        # TODO: every GUI-facing knob initiate_measurement() accepts.
        "source_value": ParamSpec(
            type=float, default=1e-6, unit="A",
            description="TODO",
        ),
        "readings_per_point": ParamSpec(
            type=int, default=5, min=1,
            description="Samples averaged per point",
        ),
        # A parameter naming a config-owned choice (e.g. a channel/route)
        # currently has no dynamic-choices hook in the procedure param form —
        # ship it as a plain free-text str and validate it yourself inside
        # initiate_measurement() (raise ValueError naming the valid options).
        # "channel": ParamSpec(type=str, default="", description="TODO"),
    }

    # Only declare control_limits for numeric params with a real physical
    # bound. self._limits[name] MUST be populated in __init__ below even
    # when init_params is empty — conformance builds this VI with zero
    # init_params, so give it a safe internal default.
    control_limits: ClassVar[dict[str, dict[str, str]]] = {
        "initiate_measurement": {"source_value": "source_value"},
    }

    # Declare ONLY if some measurement_parameters entry should be loopable
    # (appear in the procedure's Reading-loop panel) across sweep points.
    # Maps the parameter name to a dedicated setter method (below) that
    # reprograms just that value without re-running initiate_measurement().
    # reading_setters: ClassVar[dict[str, str]] = {"source_value": "set_source_value"}

    # Declare ONLY if the reading is a frame (a camera, any 2-D datum in one
    # unit): the image-block standard. take_reading() then returns the frame
    # under the block name as a (height_px, width_px) numpy array, alongside
    # the scalars; keep a scalar column derived from it so the live plot has
    # something to show. See virtual_instruments/measurement/camera.py.
    # from i2as.core.plan import ImageBlock
    # measurement_image_blocks: ClassVar[dict[str, ImageBlock]] = {
    #     "frame": ImageBlock(height_px=128, width_px=128, unit="counts",
    #                         description="TODO what one pixel measures"),
    # }

    def __init__(self, drivers: dict[str, object], **init_params: Any) -> None:
        """TODO: docstring — what init_params this VI reads.

        Raises:
            I2ASConfigError: If init_params are malformed.
        """
        super().__init__(drivers, **init_params)
        self._main = drivers["main"]  # TODO: match your chosen role name(s)

        max_source_value = float(init_params.get("max_source_value", 1e-3))  # TODO safe default
        self._limits["source_value"] = (-max_source_value, max_source_value)

        self._readings_per_point: int = 5
        self._initiated: bool = False

    # ------------------------------------------------------------------
    # MeasurementInstrumentBase implementation
    # ------------------------------------------------------------------

    def data_arrays(self, params) -> dict[str, int]:
        """Return per-point array lengths for the given params. No hardware access."""
        n = int(params["readings_per_point"])
        return {key: n for key in self.measurement_data_keys}

    # panel=False: arming is a deliberate act, never on the compact monitor
    # card. action_class: arming commands the instrument, so an agent needs
    # run_control authority for it (every @control declares its class).
    @control(panel=False, action_class="run_control")
    def initiate_measurement(
        self,
        source_value: float = 1e-6,
        readings_per_point: int = 5,
    ) -> None:
        """Arm the instrument with fixed measurement parameters.

        TODO: everything a fresh take_reading() needs must be set here —
        take_reading() itself takes no arguments.

        Args:
            source_value: TODO.
            readings_per_point: Samples take_reading() averages per point.
        """
        driver = self._main  # type: ignore[attr-defined]
        # TODO: arm the real hardware. Reassert every mode this VI depends
        # on explicitly — never trust a prior session/VI to have left the
        # instrument in a compatible state (see the write-measurement-vi
        # skill's live-testing section: stale setpoints/modes are the most
        # common real-hardware surprise).
        driver.set_x(float(source_value))

        self._readings_per_point = int(readings_per_point)
        self._initiated = True

    def take_reading(self) -> dict[str, list[float] | float]:
        """Acquire one datapoint at the configuration fixed by initiate_measurement().

        Returns:
            The mean/error/array triple for every declared quantity.

        Raises:
            RuntimeError: If initiate_measurement() has not been called first.
        """
        if not self._initiated:
            raise RuntimeError("initiate_measurement() must be called before take_reading().")

        driver = self._main  # type: ignore[attr-defined]
        n = self._readings_per_point
        samples: list[float] = []
        for _ in range(n):
            samples.append(float(driver.get_y()))  # TODO real acquisition call
        samples += [float("nan")] * (n - len(samples))

        mean, error = self.mean_and_sem([v for v in samples if not math.isnan(v)])
        return {
            "quantity_a_unit_array": samples,
            "quantity_a_unit": mean,
            "quantity_a_unit_error": error,
        }

    # ------------------------------------------------------------------
    # Optional: reading-loop setter (only if reading_setters is declared above)
    # ------------------------------------------------------------------

    # @control(action_class="run_control")
    # def set_source_value(self, source_value: float) -> None:
    #     """Reprogram source_value without re-arming. Keep control_limits here too
    #     if it's the same bounded parameter (control_limits = {..., "set_source_value": {...}})."""
    #     if not self._initiated:
    #         raise RuntimeError("initiate_measurement() must be called first.")
    #     self._main.set_x(float(source_value))  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    # Do NOT write your own ping() here. BaseVirtualInstrument's default
    # already covers this VI: query get_idn() on every driver, True iff all
    # answer. Only a VI that declares detach_when_idle (see the
    # write-measurement-vi skill's "supporting externally configured
    # instruments" section) needs anything different, and that different
    # shape — reattach, a true round trip, release — is the base's job too,
    # driven entirely by the detach_when_idle property. No VI overrides
    # ping() in this codebase.

    def standby(self) -> None:
        """Put the instrument in a safe idle state and reset the initiated flag."""
        self._main.set_x(0.0)  # type: ignore[attr-defined]
        self._initiated = False
