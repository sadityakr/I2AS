"""FieldImaging — image the sample at each field of a sweep, from a saturated start."""

from __future__ import annotations

from i2as.core.gates import Gate
from i2as.core.plan import Command, ParamSpec, PhasePlan, Target
from i2as.core.procedure import RoleParam, SweepMeasureProcedure
from i2as.core.sweep_builder import SweepAxis


def _stage_axis_names(station: object, axis: str) -> list[str]:
    """Return the setup's stage VIs positioning *axis*, in config order.

    The role-discovery standard: the Station lists every stage VI, and the
    VI's own ``axis`` declaration says which axis it positions, so a
    procedure never names a configured instrument.

    Args:
        station: The Station to ask.
        axis: ``"x"`` or ``"y"``.

    Returns:
        The matching VI names, possibly empty.
    """
    return [
        name
        for name in station.stage_vi_names()  # type: ignore[attr-defined]
        if getattr(station.get_vi(name), "axis", "") == axis  # type: ignore[attr-defined]
    ]


class FieldImaging(SweepMeasureProcedure):
    """Sweep the magnetic field and take a frame at each point.

    The imaging twin of ``FieldSweep``, and a generic sweep procedure like
    it: the measurement method is chosen in the GUI, so the same procedure
    runs the widefield camera or any other measurement VI. What makes it a
    distinct procedure is its magnetic history — the **reference frame**:

    1. ``initiate()``: ramp the magnet to ``saturation_field_T`` and move
       the stage to ``stage_x_m`` / ``stage_y_m``, then arm the camera.
    2. The **saturation pre-step**, as initiation gates: once the magnet is
       at saturation, command the first sweep field and wait for it, then
       settle for ``init_wait``. The first frame is therefore taken at the
       first sweep point after arriving from saturation, so it images a
       sample in a known, fully magnetised state — the reference every
       later frame is compared against by the image-stack recipe.
       ``saturate = False`` skips the pre-step and the run starts at the
       first sweep point like a plain field sweep.
    3. ``measure()``: read the camera (frame + ROI scalars), tag on the
       field read-back, save.
    4. ``change_sweep_step()``: step the magnet to the next field.
    5. ``standby()``: disarm the camera and hand the magnet to its own
       ``standby()``; the stage stays where it is.

    Instruments (the role-discovery standard, see ``RoleParam``):
        ``field_vi`` — the magnet this sweeps, required. ``stage_x_vi`` /
        ``stage_y_vi`` — the stage axis VIs this positions, optional: a
        setup without a stage images wherever the sample sits, and the two
        position parameters are then ignored.
    """

    name = "Field Imaging"
    description = (
        "Sweep magnetic field from a saturated start and image the sample at "
        "each point with the selected method"
    )
    sweep_axis = SweepAxis(
        key="field",
        unit="T",
        data_key="field_T",
        description="Magnetic field",
        default_start=-1.0,
        default_end=1.0,
        default_steps=41,
    )
    sweep_data_keys = [sweep_axis.data_key]
    default_x_key = sweep_axis.data_key

    role_parameters = {
        "field_vi": RoleParam(
            candidates=lambda station: station.magnet_vi_names(),
            description="Magnet this run sweeps",
        ),
        "stage_x_vi": RoleParam(
            candidates=lambda station: _stage_axis_names(station, "x"),
            description="Stage axis this run positions to 'stage_x_m'",
            required=False,
        ),
        "stage_y_vi": RoleParam(
            candidates=lambda station: _stage_axis_names(station, "y"),
            description="Stage axis this run positions to 'stage_y_m'",
            required=False,
        ),
    }

    system_parameters = {
        "saturate": ParamSpec(
            type=bool,
            default=True,
            description=(
                "Ramp to 'saturation_field_T' before the sweep so the first "
                "frame images a fully magnetised sample (the reference frame)"
            ),
        ),
        "saturation_field_T": ParamSpec(
            type=float,
            default=-1.5,
            unit="T",
            description=(
                "Field to saturate at before the sweep, sign included — "
                "normally beyond the sweep's starting field on the same side"
            ),
        ),
        "stage_x_m": ParamSpec(
            type=float,
            default=0.0,
            unit="m",
            description="Stage x position to image at (ignored without a stage)",
        ),
        "stage_y_m": ParamSpec(
            type=float,
            default=0.0,
            unit="m",
            description="Stage y position to image at (ignored without a stage)",
        ),
        "init_wait": ParamSpec(
            type=float,
            default=5.0,
            unit="s",
            description="Settle after reaching the first field, before the reference frame",
        ),
        "step_wait": ParamSpec(
            type=float,
            default=2.0,
            unit="s",
            description="Settle between field steps, before each frame",
        ),
    }

    # ------------------------------------------------------------------
    # Axis-specific hooks (SweepMeasureProcedure owns the four-method loop)
    # ------------------------------------------------------------------

    def _initial_system_targets(self) -> dict[str, Target]:
        """Ramp to saturation (or the first field) and position the stage.

        With ``saturate`` on, the magnet's initial target is the saturation
        field; the initiation gates then take it to the first sweep point.
        Off, the first sweep point is the initial target as in a plain field
        sweep. A stage axis this station has no VI for contributes no
        target, so the Orchestrator never ramps it.
        """
        first_field = (
            float(self._params["saturation_field_T"])
            if self._params["saturate"]
            else self._sweep[0]
        )
        targets = {self.role_vi("field_vi"): Target(first_field)}
        for role, param in (("stage_x_vi", "stage_x_m"), ("stage_y_vi", "stage_y_m")):
            stage_vi = self.role_vi(role)
            if stage_vi:
                targets[stage_vi] = Target(float(self._params[param]))
        return targets

    def _step_targets(self, index: int) -> dict[str, Target]:
        """Ramp the magnet to the field at *index*."""
        return {self.role_vi("field_vi"): Target(self._sweep[index])}

    def _standby_targets(self) -> dict[str, Target]:
        """No system targets — ``standby()`` below commands the magnet directly."""
        return {}

    def _axis_readback(self) -> float:
        """Read the current field from the magnet this run sweeps."""
        return self._station.get_vi(self.role_vi("field_vi")).magnet_field_T()

    def _initiate_wait_s(self) -> float:
        """Settle time before the reference frame (``init_wait``)."""
        return float(self._params["init_wait"])

    def _step_wait_s(self) -> float:
        """Settle time between field steps (``step_wait``)."""
        return float(self._params["step_wait"])

    # ------------------------------------------------------------------
    # The saturation pre-step
    # ------------------------------------------------------------------

    def initiation_gates(self) -> tuple[Gate, ...]:
        """Return the gates that take the magnet from saturation to the first field.

        Gates replace ``initiate()``'s ``wait_s`` entirely, so the chain
        carries the settle time too:

        1. ``from_saturation`` — its one-shot action commands the magnet's
           own ``set_field`` to the first sweep field, as a ``Command``
           through the Station (the same channel every plan uses; the
           Orchestrator advances ramps while a gate waits), and its check
           holds until the magnet reports ``TARGET_REACHED``.
        2. ``settle`` — a pure wait of ``init_wait`` seconds.

        Returns:
            The two gates when ``saturate`` is on; an empty tuple otherwise,
            so ``init_wait`` governs unchanged.
        """
        if not self._params["saturate"]:
            return ()
        field_vi = self.role_vi("field_vi")
        first_field = float(self._sweep[0])

        def _ramp_to_first_field() -> None:
            self._station.send_measurement_commands(
                (Command(field_vi, "set_field", {"target_T": first_field}),)
            )

        def _at_first_field() -> bool:
            return self._station.get_vi(field_vi).ramp_status() == "TARGET_REACHED"

        return (
            Gate("from_saturation", check=_at_first_field, action=_ramp_to_first_field),
            Gate("settle", check=lambda: True, window_s=self._initiate_wait_s()),
        )

    def standby(self) -> PhasePlan:
        """Disarm the measurement VI and put the magnet into its own standby.

        Returns:
            The base ``SweepMeasureProcedure.standby()`` plan plus a
            ``Command`` invoking the magnet's own ``standby()`` — that VI's
            standard standby action owns what standby means physically. The
            stage is left where it is: a drive home is a move like any other.
        """
        plan = super().standby()
        return PhasePlan(
            targets=plan.targets,
            commands=(*plan.commands, Command(self.role_vi("field_vi"), "standby", {})),
            wait_s=plan.wait_s,
        )
