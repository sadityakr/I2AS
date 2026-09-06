"""TemperatureSweep — temperature sweep running any selected measurement method."""

from __future__ import annotations

import logging
from typing import Any

from i2as.core.exceptions import I2ASConfigError
from i2as.core.plan import ParamSpec, Target
from i2as.core.procedure import RoleParam, SweepMeasureProcedure
from i2as.core.sweep_builder import SweepAxis

logger = logging.getLogger(__name__)


class TemperatureSweep(SweepMeasureProcedure):
    """Sweep temperature and measure with any selected measurement VI.

    This is a generic sweep procedure (see ``SweepMeasureProcedure``): the
    measurement method is chosen in the GUI, so the same procedure runs a DC
    resistance measurement or any future measurement VI with no new code.

    Procedure flow:
    1. ``initiate()``: ramp the temperature controller to the first
       temperature at ``ramp_rate_K_per_min``, hold the magnet at its field,
       arm the selected measurement VI.
    2. ``measure()``: read the VI, tag on the temperature read-back, save.
    3. ``change_sweep_step()``: ramp the controller to the next step.
    4. ``standby()``: close the data file; temperature holds at the last point.

    The ramp rate is passed per-step via the temperature ``Target``, so it takes
    effect immediately at each step without a YAML config change.

    Instruments (the role-discovery standard, see ``RoleParam``):
        ``temperature_vi`` — the controller this sweeps, required;
        ``field_vi`` — a magnet held at ``field`` for the run, optional. A
        station with no magnet still runs the sweep; a NONZERO field asked of
        a station with no magnet is refused at construction. At least one
        measurement VI is required by ``SweepMeasureProcedure``.
    """

    name = "Temperature Sweep"
    description = "Sweep temperature, measure with the selected method at each stable point"
    sweep_axis = SweepAxis(
        key="temperature",
        unit="K",
        data_key="temperature_K",
        description="Temperature",
        default_start=10.0,
        default_end=300.0,
        default_steps=30,
    )
    sweep_data_keys = [sweep_axis.data_key]
    default_x_key = sweep_axis.data_key

    role_parameters = {
        "temperature_vi": RoleParam(
            candidates=lambda station: station.temperature_vi_names(),
            description="Temperature controller this run sweeps",
        ),
        "field_vi": RoleParam(
            candidates=lambda station: station.magnet_vi_names(),
            description="Magnet held at the applied field for the whole run",
            required=False,
        ),
    }

    system_parameters = {
        "field": ParamSpec(
            type=float,
            default=0.0,
            unit="T",
            description="Applied field, held constant for the whole sweep",
        ),
        "ramp_rate_K_per_min": ParamSpec(
            type=float,
            default=2.0,
            unit="K/min",
            description="Temperature ramp rate between steps",
        ),
        "set_temperature": ParamSpec(
            type=bool,
            default=True,
            description=(
                "Set the temperature (the swept axis). Off = measure at each "
                "sweep point without commanding it, e.g. following a passive drift"
            ),
        ),
        "point_wait": ParamSpec(
            type=float,
            default=60.0,
            unit="s",
            description="Wait after reaching each temperature (thermal equilibration)",
        ),
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Resolve the roles, then validate the applied field against them.

        The applied field is optional: a station with no magnet still runs
        this sweep at zero field. A NONZERO field on a station with no magnet
        is refused here, at construction — silently measuring at 0 T while
        the metadata claims 0.5 T would corrupt a dataset without anyone
        noticing.

        Raises:
            I2ASConfigError: If a nonzero field is requested and no
                magnet is configured (or, via the base class, if a required
                role has no instrument, the station has no measurement VI, or
                a parameter collision occurs).
        """
        super().__init__(*args, **kwargs)
        field = float(self._params["field"])
        magnet = self.role_vi("field_vi")
        self._magnet_targets: dict[str, Target] = (
            {magnet: Target(field)} if magnet else {}
        )
        if not magnet:
            if field != 0.0:
                raise I2ASConfigError(
                    f"field={field} T requested, but this station configures no "
                    f"magnet. Set field to 0, or configure a magnet."
                )
            logger.info(
                "TemperatureSweep: station configures no magnet — running at "
                "zero field."
            )

    def _temp_target(self, index: int) -> Target:
        """Build the swept controller's ``Target`` at *index* (with ramp rate)."""
        return Target(
            self._sweep[index],
            rate=self._params["ramp_rate_K_per_min"],
        )

    # ------------------------------------------------------------------
    # Axis-specific hooks (SweepMeasureProcedure owns the four-method loop)
    # ------------------------------------------------------------------

    def _sweep_targets(self, index: int) -> dict[str, Target]:
        """Build the swept-channel target at *index*, honouring its on/off toggle.

        With ``set_temperature`` off the dict is empty: the Orchestrator never
        ramps the controller, and the sweep walks its points measuring
        whatever temperature the cryostat happens to be at (the read-back
        still records the true value). Monitoring is unaffected either way.
        """
        if not self._params["set_temperature"]:
            return {}
        return {self.role_vi("temperature_vi"): self._temp_target(index)}

    def _initial_system_targets(self) -> dict[str, Target]:
        """Ramp the swept controller to its first value, and hold the field."""
        return {
            **self._sweep_targets(0),
            **self._magnet_targets,  # empty when the station configures no magnet
        }

    def _step_targets(self, index: int) -> dict[str, Target]:
        """Ramp the controller to the temperature at *index* (with rate).

        The field is held from ``initiate()`` and deliberately not re-sent
        each step.
        """
        return self._sweep_targets(index)

    def _standby_targets(self) -> dict[str, Target]:
        """No system targets — temperature holds at the last set point."""
        return {}

    def _axis_readback(self) -> float:
        """Read the current temperature from the controller this run sweeps."""
        return self._station.get_vi(self.role_vi("temperature_vi")).temperature()

    def _initiate_wait_s(self) -> float:
        """Settle time after the initial ramp (``point_wait``)."""
        return float(self._params["point_wait"])

    def _step_wait_s(self) -> float:
        """Settle time after each temperature step (``point_wait``)."""
        return float(self._params["point_wait"])
