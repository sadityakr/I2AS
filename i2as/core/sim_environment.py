"""SimEnvironment — the shared physical world two coupled sim drivers live in.

The **sim-coupling standard** (full text in ``drivers/README.md``): a sim
driver models ONE instrument, and two sims that share a physical quantity —
the field a simulated magnet applies and a simulated camera sees — never
import each other. They exchange the quantity through an environment
object, which is the only thing that knows how the two are physically
related. The producer publishes what it knows (a PSU knows its output
current); the environment holds the physics that turns it into what the
consumer observes (a coil constant turns current into field); the consumer
reads the observable it responds to.

Which environment a sim joins is written into its resource string, the one
argument every driver takes: a ``@<name>`` suffix — ``"SIM::IPS_Z@imaging"``
— joins the process-local environment ``imaging``, shared by every sim
constructed with the same suffix. A resource string without a suffix gets a
private environment of its own, so two sims built independently in a test
influence each other only when the test says so.

This module is foundation code in the sense of import-linter contract C2:
drivers may import it, and it imports nothing from the package.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

#: The separator between a sim's resource string and its environment name.
ENVIRONMENT_SUFFIX_SEPARATOR = "@"

#: The simulated coil constant, in amperes per tesla — the physics of the sim
#: world, shared by every sim that publishes or observes a field. The shipped
#: sim configs give their magnet VI the same ``amperes_per_tesla`` so the
#: field the VI reports and the field the sim camera sees are one number;
#: ``tests/test_l0_sim_camera_stage.py`` holds the two together.
DEFAULT_AMPERES_PER_TESLA = 10.0


class SimEnvironment:
    """One shared physical world for coupled sims.

    Attributes:
        name: The environment's name (``""`` for a private one).
        psu_current_A: The magnet power supply's present output current, in
            amperes — what the sim PSU publishes whenever its current
            changes. ``0.0`` until a PSU has published.
        amperes_per_tesla: The simulated coil constant that turns that
            current into the field at the sample.
    """

    def __init__(self, name: str = "") -> None:
        """Create an environment at zero field.

        Args:
            name: The environment's name; ``""`` marks a private environment
                that no resource string can join.
        """
        self.name = name
        self.psu_current_A: float = 0.0
        self.amperes_per_tesla: float = DEFAULT_AMPERES_PER_TESLA

    @property
    def applied_field_T(self) -> float:
        """The field at the sample, in tesla, derived from the published current."""
        return self.psu_current_A / self.amperes_per_tesla

    @applied_field_T.setter
    def applied_field_T(self, field_T: float) -> None:
        """Set the field directly — a test's way of sweeping the sim world.

        Args:
            field_T: The field at the sample, in tesla; stored as the current
                that produces it.
        """
        self.psu_current_A = float(field_T) * self.amperes_per_tesla

    def __repr__(self) -> str:
        """Return a short description naming the environment and its field."""
        label = self.name or "private"
        return f"SimEnvironment({label!r}, field={self.applied_field_T:.4g} T)"


_registry: dict[str, SimEnvironment] = {}
_registry_lock = threading.Lock()


def get(name: str) -> SimEnvironment:
    """Return the process-local environment called *name*, creating it once.

    Args:
        name: The environment's name, as written after ``@`` in a resource
            string. Must be non-empty.

    Returns:
        The one ``SimEnvironment`` of that name in this process.

    Raises:
        ValueError: If *name* is empty — a private environment is made with
            ``SimEnvironment()`` directly, never looked up.
    """
    if not name:
        raise ValueError("a shared sim environment needs a non-empty name")
    with _registry_lock:
        environment = _registry.get(name)
        if environment is None:
            environment = SimEnvironment(name)
            _registry[name] = environment
            logger.debug("sim environment %r created", name)
        return environment


def for_resource(resource_string: str) -> SimEnvironment:
    """Return the environment a sim's resource string opts into.

    Args:
        resource_string: The driver's one constructor argument, e.g.
            ``"SIM::IPS_Z@imaging"``. Everything after the last ``@`` names
            the shared environment; no ``@`` means a private one.

    Returns:
        The shared environment named by the suffix, or a fresh private
        environment when the string carries none.
    """
    _, separator, name = str(resource_string).rpartition(ENVIRONMENT_SUFFIX_SEPARATOR)
    if separator and name:
        return get(name)
    return SimEnvironment()


__all__ = [
    "DEFAULT_AMPERES_PER_TESLA",
    "ENVIRONMENT_SUFFIX_SEPARATOR",
    "SimEnvironment",
    "for_resource",
    "get",
]
