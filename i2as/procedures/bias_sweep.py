"""BiasSweep — sweep a voltage-biased measurement's own bias (an I–V curve)."""

from __future__ import annotations

from typing import Any

from i2as.core.exceptions import I2ASConfigError
from i2as.core.plan import Command, ParamSpec, PhasePlan, Target
from i2as.core.procedure import RoleParam, SweepMeasureProcedure
from i2as.core.sweep_builder import SweepAxis

#: The measurement parameter a bias-capable measurement VI loops its bias by
#: (its ``reading_setters`` key).
BIAS_PARAMETER = "bias_V"


class BiasSweep(SweepMeasureProcedure):
    """Step the bias of a voltage-biased measurement and read the current: an I–V.

    A generic sweep procedure (see ``SweepMeasureProcedure``) whose axis is
    the measurement's OWN excitation rather than a system setpoint: at every
    point it dispatches the selected measurement VI's bias setter (its
    ``reading_setters["bias_V"]``, e.g. ``set_bias_voltage``) as a
    ``Command`` through the Station, then takes the reading. Field and
    temperature are held where the run puts them for the whole curve — the
    usual way to record a tunnel junction's I–V in a known magnetic state
    (saturate the field to set parallel or antiparallel, then sweep).

    Procedure flow:
    1. ``initiate()``: optionally ramp the magnet to ``field`` and the
       temperature controller to ``temperature``; arm the measurement VI.
    2. ``measure()``: set the bias of this point, read, save the bias column.
    3. ``change_sweep_step()``: next bias (no system ramp).
    4. ``standby()``: disarm the measurement VI (bias to zero, output off).

    Instruments (the role-discovery standard): ``field_vi`` and
    ``temperature_vi`` are both optional. The selected measurement VI must
    advertise a ``bias_V`` reading setter; one that does not is refused when
    the run is armed.
    """

    name = "Bias Sweep"
    description = "Sweep the measurement bias voltage at fixed field and temperature (I–V)"
    sweep_axis = SweepAxis(
        key="bias",
        unit="V",
        data_key="applied_bias_V",
        description="Bias voltage",
        default_start=-0.5,
        default_end=0.5,
        default_steps=51,
    )
    sweep_data_keys = [sweep_axis.data_key]
    default_x_key = sweep_axis.data_key

    role_parameters = {
        "field_vi": RoleParam(
            candidates=lambda station: station.magnet_vi_names(),
            description="Magnet held at 'field' during the curve (see 'set_field')",
            required=False,
        ),
        "temperature_vi": RoleParam(
            candidates=lambda station: station.temperature_vi_names(),
            description="Temperature controller this run sets (see 'set_temperature')",
            required=False,
        ),
    }

    system_parameters = {
        "set_field": ParamSpec(
            type=bool,
            default=False,
            description="Ramp the magnet to 'field' before the curve (off = leave it alone)",
        ),
        "field": ParamSpec(
            type=float,
            default=0.0,
            unit="T",
            description="Field held during the curve (ignored when 'set_field' is off)",
        ),
        "set_temperature": ParamSpec(
            type=bool,
            default=False,
            description="Set the temperature for this run (off = leave it alone)",
        ),
        "temperature": ParamSpec(
            type=float,
            default=10.0,
            unit="K",
            description="Temperature setpoint (ignored when 'set_temperature' is off)",
        ),
        "init_wait": ParamSpec(
            type=float,
            default=5.0,
            unit="s",
            description="Wait after the initial ramp (field/thermal settling)",
        ),
        "step_wait": ParamSpec(
            type=float,
            default=0.0,
            unit="s",
            description="Wait between bias steps",
        ),
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Construct as any sweep procedure; the bias setter is resolved at arming."""
        super().__init__(*args, **kwargs)
        self._bias_setter: str | None = self._station.get_vi(
            self._measurement_vi
        ).reading_setters.get(BIAS_PARAMETER)

    # ------------------------------------------------------------------
    # Axis-specific hooks
    # ------------------------------------------------------------------

    def initiate(self) -> PhasePlan:
        """Refuse a measurement without a bias, then arm as every sweep does.

        Raises:
            I2ASConfigError: If the selected measurement VI advertises no
                ``bias_V`` reading setter — there is nothing to sweep.
        """
        if self._bias_setter is None:
            raise I2ASConfigError(
                f"{self.name!r} sweeps a measurement's bias, but measurement VI "
                f"{self._measurement_vi!r} advertises no {BIAS_PARAMETER!r} "
                f"reading setter. Select a voltage-biased measurement."
            )
        return super().initiate()

    def _initial_system_targets(self) -> dict[str, Target]:
        """Ramp the magnet and the temperature when their toggles are on."""
        targets: dict[str, Target] = {}
        field_vi = self.role_vi("field_vi")
        if field_vi and self._params["set_field"]:
            targets[field_vi] = Target(self._params["field"])
        temperature_vi = self.role_vi("temperature_vi")
        if temperature_vi and self._params["set_temperature"]:
            targets[temperature_vi] = Target(self._params["temperature"])
        return targets

    def _step_targets(self, index: int) -> dict[str, Target]:
        """No system ramp between points — the bias is set in ``measure()``."""
        return {}

    def _standby_targets(self) -> dict[str, Target]:
        """Leave field and temperature where the curve held them."""
        return {}

    def measure(self) -> None:
        """Set this point's bias through the Station, then read and save."""
        if self._bias_setter is not None:
            self._station.send_measurement_commands(
                (
                    Command(
                        self._measurement_vi,
                        self._bias_setter,
                        {BIAS_PARAMETER: float(self._sweep[self._index])},
                    ),
                )
            )
        super().measure()

    def _axis_readback(self) -> float:
        """Return the bias applied at this point, volts."""
        return float(self._sweep[self._index])

    def _initiate_wait_s(self) -> float:
        """Settle time after the initial ramp (``init_wait``)."""
        return float(self._params["init_wait"])

    def _step_wait_s(self) -> float:
        """Settle time between bias steps (``step_wait``)."""
        return float(self._params["step_wait"])
