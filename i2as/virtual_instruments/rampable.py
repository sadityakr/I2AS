"""RampableVI mixin — abstract ramp API for system VIs."""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar


class RampableVI:
    """Mixin for any VI that requires controlled ramping.

    The Orchestrator calls ``start_ramp(target)`` once, then calls
    ``advance_ramp()`` every tick until ``ramp_status()`` returns
    ``"TARGET_REACHED"``.

    Subclasses *must* implement all four abstract methods.

    Attributes:
        no_motion_phases: The **no-motion phase declaration**: the
            ``ramp_phase()`` values during which this VI's value is MEANT to
            hold still while ``ramp_status()`` still reports ``"RAMPING"`` —
            a switch-heater warm-up, a settle before the next segment. The
            stall detector (``core/stall_detection.py``) never judges a VI
            in one of these phases, so an expected pause is not read as a
            stall. It is the VI's own physics, so it is declared here, on
            the class, and read off the ramp-status snapshot; the default is
            empty (every ``RAMPING`` tick is expected to make progress).
    """

    no_motion_phases: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def start_ramp(self, target: float) -> None:
        """Begin ramping to *target* value.

        Called by ``Station.process_system_targets()``.
        Target is in user units (tesla for magnets, kelvin for temperature).
        Ramp rate is determined internally from YAML config stored in
        ``self._init_params``.

        Args:
            target: Desired end value in user-facing units.
        """
        ...

    @abstractmethod
    def advance_ramp(self) -> None:
        """Advance the ramp by one step.

        Called by ``Station.check_ramps()`` every Orchestrator tick while
        this VI is ramping.  Internally calls ``next()`` on the generator
        returned by ``_ramp_generator()``.
        """
        ...

    @abstractmethod
    def ramp_status(self) -> str:
        """Return the current ramp state string.

        Returns:
            ``"RAMPING"``        — VI has not yet reached its target.
            ``"TARGET_REACHED"`` — target reached and confirmed.
            ``"IDLE"``           — no ramp active.
        """
        ...

    @abstractmethod
    def stop_ramp(self) -> None:
        """Stop any active ramp and freeze the hardware where it is.

        Called by the Orchestrator on abort and on ERROR/EMERGENCY entry.
        Implementations MUST both clear the internal ramp generator and
        command the hardware to hold: for autonomous hardware (a magnet PSU
        keeps ramping to its last setpoint on its own), clearing the
        generator alone does not stop the physical ramp. After this call,
        ``ramp_status()`` must report ``"IDLE"``.
        """
        ...

    # ------------------------------------------------------------------
    # Optional introspection hooks (concrete defaults; override to expose)
    # ------------------------------------------------------------------

    def ramp_target(self) -> float | None:
        """Return the active ramp target in user units, or ``None``.

        User units means tesla for magnets, kelvin for temperature — the same
        units as ``start_ramp``'s *target* and the VI's ``@monitored`` value —
        so operational-status reporting can compute gap-to-target ("how far
        from the setpoint, and is it closing?"). Unlike the four methods above,
        this is not part of the required behaviour contract: it is a read-only
        diagnostic accessor with a safe default. The default returns ``None``
        (target not exposed); ramp-tracking VIs override it.
        """
        return None

    def ramp_rate(self) -> float | None:
        """Return the active ramp rate in user units per minute, or ``None``.

        Tesla/min for magnets, kelvin/min for temperature — consistent with
        ``ramp_target()`` — so a rough ETA (gap ÷ rate) can be estimated. The
        default returns ``None``; ramp-tracking VIs override it.
        """
        return None

    def nominal_ramp_rate(self) -> float | None:
        """Return the rate a ramp from rest would use, or ``None`` if undeclared.

        The DECLARED sibling of ``ramp_rate()``, in the same user units per
        minute (tesla/min, kelvin/min, degrees/min): what this VI would ramp
        at if asked right now, read from its configured rate alone — no bus
        traffic, no active ramp needed. That is what makes it answerable
        before a run exists, which is what the **duration estimate**
        (`core/estimates.py`) needs to turn a run's declared setpoints into a
        time (``Station.nominal_ramp_rates()`` is the aggregation point).

        A VI whose rate varies along the ramp (a magnet's current-dependent
        segments) reports its SLOWEST configured rate, so an estimate built
        from it is never optimistic. The default returns ``None`` — "this VI
        declares no rate" — which the estimator reports as an explicit
        assumption rather than counting as instant.

        Returns:
            The nominal rate in user units per minute, or ``None``.
        """
        return None

    def ramp_setpoint(self) -> float | None:
        """Return the setpoint currently commanded to hardware, or ``None``.

        The *next* setpoint, in the same user units as ``ramp_target()``:
        the intermediate value the ramp generator last wrote to the
        instrument on its way to the end setpoint. It is distinct from
        ``ramp_target()`` (the END setpoint the ramp is walking toward) for
        every VI whose generator approaches its target in steps — a magnet
        crossing a ramp-segment boundary, a temperature controller advancing
        a time-based setpoint each tick — and equals the target for a VI
        whose generator commands it in one shot.

        Read from the VI's own record of what it last commanded, never
        polled back from the instrument: it must stay a pure accessor with
        no bus traffic, because the ramp tracker reads it every tick.
        Implementations MUST clear it in ``stop_ramp()`` alongside the
        target. The default returns ``None`` (setpoint not exposed);
        ramp-tracking VIs override it.
        """
        return None

    def ramp_value(self) -> float | None:
        """Return the current value in the same user units as ``ramp_target()``.

        Tesla for magnets, kelvin for temperature — the ``@monitored`` reading
        the ramp is driving. Lets operational-status reporting compute
        gap-to-target without knowing which monitored field is "the value" for
        each VI type. Default ``None``; ramp-tracking VIs override it.
        """
        return None

    def ramp_phase(self) -> str | None:
        """Return the active ramp sub-phase, or ``None`` if the VI has none.

        Most VIs ramp in a single phase and return ``None`` (the stall detector
        then treats them as always making progress toward target). A VI with
        distinct phases overrides this, and names the ones where its value
        deliberately holds still in ``no_motion_phases`` so the stall
        detector does not read those expected pauses as a stall.
        """
        return None
