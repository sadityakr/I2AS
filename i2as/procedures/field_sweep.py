"""FieldSweep — magnetic field sweep running any selected measurement method."""

from __future__ import annotations

from i2as.core.plan import Command, ParamSpec, PhasePlan, Target
from i2as.core.procedure import RoleParam, SweepMeasureProcedure
from i2as.core.sweep_builder import SweepAxis


class FieldSweep(SweepMeasureProcedure):
    """Sweep the magnetic field and measure with any selected measurement VI.

    This is a generic sweep procedure (see ``SweepMeasureProcedure``): the
    measurement method is chosen in the GUI, so the same procedure runs a DC
    resistance measurement or any future measurement VI with no new code. Adding a new *measurement* is adding a measurement VI, not a
    procedure; adding a new *sweep axis* is a small subclass like this one that
    supplies the ramp targets and the axis read-back.

    Procedure flow:
    1. ``initiate()``: ramp the magnet to the first field, the temperature
       controller to the target temperature, arm the selected measurement VI.
    2. ``measure()``: read the VI, tag on the field read-back, save.
    3. ``change_sweep_step()``: step the magnet to the next field.
    4. ``standby()``: disarm the VI and put the magnet into its own standby
       (that VI's standard command owns what "standby" means physically —
       ramp to zero, switch heater off where applicable — this procedure
       does not set the field itself).

    Instruments (the role-discovery standard, see ``RoleParam``):
        ``field_vi`` — the magnet this sweeps, required; ``temperature_vi`` —
        the controller whose setpoint it sets, optional. Both default to the
        setup's only instrument of that role, and both are chosen in the form
        when a setup has several. At least one measurement VI is required by
        ``SweepMeasureProcedure``.
    """

    name = "Field Sweep"
    description = "Sweep magnetic field, measure with the selected method at each point"
    sweep_axis = SweepAxis(
        key="field",
        unit="T",
        data_key="field_T",
        description="Magnetic field",
        default_start=-1.0,
        default_end=1.0,
        default_steps=101,
    )
    sweep_data_keys = [sweep_axis.data_key]
    default_x_key = sweep_axis.data_key

    role_parameters = {
        "field_vi": RoleParam(
            candidates=lambda station: station.magnet_vi_names(),
            description="Magnet this run sweeps",
        ),
        "temperature_vi": RoleParam(
            candidates=lambda station: station.temperature_vi_names(),
            description="Temperature controller this run sets (see 'set_temperature')",
            required=False,
        ),
    }

    system_parameters = {
        "set_temperature": ParamSpec(
            type=bool,
            default=True,
            description="Set the temperature during this run (off = leave it alone)",
        ),
        "temperature": ParamSpec(
            type=float,
            default=10.0,
            unit="K",
            description="Temperature setpoint (ignored when 'set_temperature' is off)",
        ),
        "init_wait": ParamSpec(
            type=float,
            default=300.0,
            unit="s",
            description="Wait after initial ramp (thermal equilibration)",
        ),
        "step_wait": ParamSpec(
            type=float,
            default=5.0,
            unit="s",
            description="Wait between field steps",
        ),
    }

    # ------------------------------------------------------------------
    # Axis-specific hooks (SweepMeasureProcedure owns the four-method loop)
    # ------------------------------------------------------------------

    def _initial_system_targets(self) -> dict[str, Target]:
        """Ramp the magnet to the first field, and the temperature if enabled.

        The temperature target is absent when its toggle is off, or when this
        station configures no temperature controller at all, so the
        Orchestrator never ramps it and the controller holds wherever the
        operator left it. Monitoring is unaffected — readings come from the
        tick loop's monitor pass, not from targets.
        """
        targets = {self.role_vi("field_vi"): Target(self._sweep[0])}
        temperature_vi = self.role_vi("temperature_vi")
        if temperature_vi and self._params["set_temperature"]:
            targets[temperature_vi] = Target(self._params["temperature"])
        return targets

    def _step_targets(self, index: int) -> dict[str, Target]:
        """Ramp the magnet to the field at *index*."""
        return {self.role_vi("field_vi"): Target(self._sweep[index])}

    def _standby_targets(self) -> dict[str, Target]:
        """No system targets — ``standby()`` below commands the magnet directly."""
        return {}

    def standby(self) -> PhasePlan:
        """Disarm the measurement VI and put the magnet into its own standby.

        Returns:
            The base ``SweepMeasureProcedure.standby()`` plan plus a
            ``Command`` invoking the magnet's own ``standby()`` — that VI's
            standard standby action, which owns what standby means
            physically (ramp to zero, switch heater off where applicable).
            This procedure does not set the field itself.
        """
        plan = super().standby()
        return PhasePlan(
            targets=plan.targets,
            commands=(*plan.commands, Command(self.role_vi("field_vi"), "standby", {})),
            wait_s=plan.wait_s,
        )

    def _axis_readback(self) -> float:
        """Read the current field from the magnet this run sweeps."""
        return self._station.get_vi(self.role_vi("field_vi")).magnet_field_T()

    def _initiate_wait_s(self) -> float:
        """Settle time after the initial ramp (``init_wait``)."""
        return float(self._params["init_wait"])

    def _step_wait_s(self) -> float:
        """Settle time between field steps (``step_wait``)."""
        return float(self._params["step_wait"])
