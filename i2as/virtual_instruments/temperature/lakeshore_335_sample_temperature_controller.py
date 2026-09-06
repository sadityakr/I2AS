"""Lakeshore335SampleTemperatureControllerVI — calibration curve and heater range."""

from __future__ import annotations

from typing import Any, ClassVar

from i2as.core.decorators import control, monitored
from i2as.core.plan import ParamSpec
from i2as.virtual_instruments.temperature.sample_temperature_controller import (
    SampleTemperatureControllerVI,
)


class Lakeshore335SampleTemperatureControllerVI(SampleTemperatureControllerVI):
    """Virtual Instrument for a Lakeshore 335 sample temperature controller.

    Identical to ``SampleTemperatureControllerVI`` in all ramp and temperature
    control behaviour. Adds calibration-curve and heater-range monitoring and
    selection, both rendered on the instrument front panel as drop-downs
    (``panel=False``: occasional commissioning/bench actions, not routine
    ones, mirroring how PID tuning is kept off the compact card).

    Heater range is what actually switches heater power on: the instrument
    powers up with range Off, so the heater delivers no power — regardless
    of ``heater_mode`` (inherited from ``SampleTemperatureControllerVI``) or
    setpoint — until the range is set to Low, Medium, or High.

    Driver contract (additions to SampleTemperatureControllerVI)
    -------------------------------------------------------------
    The ``"main"`` driver must also implement:
    * ``get_sensor_curve(sensor_input: str) -> int``
    * ``set_sensor_curve(curve: int, sensor_input: str) -> None``
    * ``get_heater_range() -> str``  — 'OFF', 'LOW', 'MEDIUM', or 'HIGH'
    * ``set_heater_range(range_setting: str) -> None``
    """

    # The sample sensor is wired to input A on every configured setup (the
    # same input every inherited @monitored reading already uses).
    _SENSOR_INPUT: ClassVar[str] = "A"

    # Lakeshore 335 curve numbering (INCRV, see the 335 manual): 0 = none,
    # 1-20 = factory Standard curves, 21-59 = User curves.
    _CURVE_CHOICES: ClassVar[dict[str, int]] = {
        "None (0)": 0,
        **{f"Standard {n}": n for n in range(1, 21)},
        **{f"User {n}": n for n in range(21, 60)},
    }

    # Lakeshore 335 heater range (RANGE, see the 335 manual §4.5.1.7.8): Off
    # switches heater power off entirely; Low/Medium/High select decade power
    # steps. Power-up default is Off.
    _HEATER_RANGE_CHOICES: ClassVar[dict[str, str]] = {
        "Off": "OFF",
        "Low": "LOW",
        "Medium": "MEDIUM",
        "High": "HIGH",
    }

    #: Heater range selected by ``initiate()`` when the config names none.
    #: Medium is the middle decade step — enough power for the sample space
    #: without committing the heater to its highest range unasked.
    _DEFAULT_INITIATE_HEATER_RANGE: ClassVar[str] = "MEDIUM"

    def __init__(self, drivers: dict[str, object], **init_params: Any) -> None:
        """Build the VI and read the heater range ``initiate()`` selects.

        Args:
            drivers: Role -> driver mapping; ``"main"`` is the Lakeshore 335.
            **init_params: The inherited temperature-controller parameters
                plus ``initiate_heater_range`` (``"OFF"``/``"LOW"``/
                ``"MEDIUM"``/``"HIGH"``), the range ``initiate()`` selects.
                Which range this setup's heater and sample space want is a
                property of the setup, so it comes from the config.

        Raises:
            ValueError: If ``initiate_heater_range`` is not one of the four
                ranges the instrument offers.
        """
        super().__init__(drivers, **init_params)
        requested = str(
            init_params.get(
                "initiate_heater_range", self._DEFAULT_INITIATE_HEATER_RANGE
            )
        ).upper()
        if requested not in set(self._HEATER_RANGE_CHOICES.values()):
            raise ValueError(
                f"initiate_heater_range must be one of "
                f"{sorted(self._HEATER_RANGE_CHOICES.values())}, got {requested!r}"
            )
        self._initiate_heater_range: str = requested

    # ------------------------------------------------------------------
    # Lifecycle — the heater range is part of the 335's operating state
    # ------------------------------------------------------------------

    def initiate(self) -> None:
        """Hand the loop to I2AS AND switch heater power on.

        Extends ``SampleTemperatureControllerVI.initiate()`` (setpoint pinned
        to the current reading, then heater mode AUTO) with the one piece of
        operating state that is specific to the 335: its heater RANGE. The
        instrument powers up with the range Off, and Off delivers no power
        whatever the mode or setpoint is (335 manual §4.5.1.7.8) — so without
        this the closed loop would be handed over and still not heat.

        Ordering mirrors the inherited method's reasoning: the loop is put in
        order first, and only then is power allowed to reach the heater.
        ``standby()`` is deliberately NOT the mirror of this: it leaves the
        range where it stands, because switching heater power back off is the
        operator's call (the range is also the gate they use by hand), and
        the inherited standby has already taken the loop out of circuit and
        commanded zero output.
        """
        super().initiate()
        self._driver.set_heater_range(self._initiate_heater_range)  # type: ignore[attr-defined]

    def _cached_choice(
        self, field: str, choices: dict[str, Any], fallback: Any
    ) -> Any:
        """Return a monitored field's last polled value if it is a valid choice.

        Honours ``control_param_specs()``'s purity rule (see
        ``BaseVirtualInstrument``): the value comes from the monitor cycle's
        cache, never from a fresh instrument read. Anything the cache cannot
        supply — this VI has not been polled yet, or the instrument reported
        a setting outside the declared list — falls back, because a
        ``ParamSpec`` default must be one of its own ``choices``.

        Args:
            field: The @monitored method whose value defaults the drop-down.
            choices: The drop-down's label -> value mapping.
            fallback: The value to use when the cache cannot supply a valid
                one. Must itself be one of ``choices``' values.

        Returns:
            The cached value, or ``fallback``.
        """
        value = self.last_monitored(field, fallback)
        return value if value in choices.values() else fallback

    def control_param_specs(self, method_name: str) -> dict[str, ParamSpec]:
        """Render ``set_curve``/``set_heater_range`` as drop-downs defaulted to the current value.

        The choice lists are static (the instrument's own curve numbering and
        heater range settings), but the *default* is the instrument's
        currently-assigned value — known only per instance — so this
        instance-level hook is used even though the choices themselves need
        no runtime data.

        That current value is read from the monitor cycle's cache
        (``last_monitored()``), not from the instrument: this method is a
        describe path, and describing must never put traffic on the bus —
        see ``control_param_specs()``'s purity rule in
        ``BaseVirtualInstrument``. The default is therefore at most one tick
        old, which is exactly what a drop-down needs.

        Args:
            method_name: The @control method name being rendered.

        Returns:
            The relevant drop-down spec for ``set_curve`` or
            ``set_heater_range``; the inherited declaration for every other
            control.
        """
        if method_name == "set_curve":
            return {
                "curve": ParamSpec(
                    type=int,
                    default=self._cached_choice("curve", self._CURVE_CHOICES, 0),
                    choices=self._CURVE_CHOICES,
                    description="Calibration curve assigned to the sample sensor input",
                )
            }
        if method_name == "set_heater_range":
            return {
                "range_setting": ParamSpec(
                    type=str,
                    default=self._cached_choice(
                        "heater_range", self._HEATER_RANGE_CHOICES, "OFF"
                    ),
                    choices=self._HEATER_RANGE_CHOICES,
                    description=(
                        "Heater output power range — Off switches the heater "
                        "off entirely, regardless of heater mode or setpoint"
                    ),
                )
            }
        return super().control_param_specs(method_name)

    # ------------------------------------------------------------------
    # @monitored methods — calibration curve, heater range
    # ------------------------------------------------------------------

    # Dimensionless: an instrument curve number (INCRV), not a quantity.
    @monitored(
        unit="",
        description="Calibration curve number assigned to the sample sensor input",
    )
    def curve(self) -> int:
        """Return the calibration curve number assigned to the sample sensor input."""
        return self._driver.get_sensor_curve(self._SENSOR_INPUT)  # type: ignore[attr-defined]

    # Dimensionless: a named decade power step, not a quantity.
    @monitored(
        unit="",
        description="Heater output power range: OFF, LOW, MEDIUM or HIGH",
    )
    def heater_range(self) -> str:
        """Return the heater range ('OFF', 'LOW', 'MEDIUM', or 'HIGH')."""
        return self._driver.get_heater_range()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # @control methods — calibration curve, heater range
    # ------------------------------------------------------------------

    # The choice list is static (the 335's own curve numbering); only the
    # DEFAULT is per-instrument, which is why control_param_specs() above
    # overrides this declaration with the currently-assigned curve.
    @control(
        action_class="recovery",
        panel=False,
        params={
            "curve": ParamSpec(
                type=int,
                default=0,
                choices=_CURVE_CHOICES,
                description="Calibration curve assigned to the sample sensor input",
            ),
        },
    )
    def set_curve(self, curve: int) -> None:
        """Assign a calibration curve to the sample sensor input.

        Args:
            curve: Curve number (0 = None, 1-20 = Standard, 21-59 = User).
        """
        self._driver.set_sensor_curve(int(curve), self._SENSOR_INPUT)  # type: ignore[attr-defined]

    # Static choices, per-instrument default — see set_curve above.
    @control(
        action_class="run_control",
        panel=False,
        params={
            "range_setting": ParamSpec(
                type=str,
                default="OFF",
                choices=_HEATER_RANGE_CHOICES,
                description=(
                    "Heater output power range — Off switches the heater off "
                    "entirely, regardless of heater mode or setpoint"
                ),
            ),
        },
    )
    def set_heater_range(self, range_setting: str) -> None:
        """Set the heater range, switching heater power on or off.

        Args:
            range_setting: One of 'OFF', 'LOW', 'MEDIUM', 'HIGH'. 'OFF'
                switches the heater off entirely; the other three select
                decade power steps (see the 335 manual §4.5.1.7.8).

        Raises:
            ValueError: If ``range_setting`` is not one of the four values.
        """
        self._driver.set_heater_range(range_setting)  # type: ignore[attr-defined]
