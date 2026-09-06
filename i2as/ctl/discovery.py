"""The run catalog this client hands down.

Neither the engine (contract C5) nor the session layer (C11) may import
``i2as.procedures``: a run travels as a class NAME, and whoever owns
discovery resolves that name and hands the catalog down. The application's
entry point does exactly this before it builds anything; ``i2as.ctl`` is
the second entry point, so it does it too, and contract C20 names
``i2as.procedures`` as the one layer above ``core``/``session`` this
package may reach — for this reason and no other.

The walk is deliberately the same shape as the application's: import every
module of the package, then take every named subclass of the base, so a
procedure sitting under an intermediate base is found and the intermediate
bases themselves (which carry no ``name``) are not.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path

from i2as.core.procedure import BaseProcedure

logger = logging.getLogger(__name__)

__all__ = ["discover_run_catalog"]

#: The packages walked, and the base each one's classes must derive from.
_PACKAGES: tuple[tuple[str, type], ...] = (
    ("i2as.procedures", BaseProcedure),
)


def _named_subclasses(base: type) -> list[type]:
    """Return every named subclass of *base*, at any depth.

    ``type.__subclasses__()`` lists only direct subclasses, so a procedure
    under an intermediate base such as ``SweepMeasureProcedure`` would be
    missed; this walks the whole tree. A class with no ``name`` is an
    intermediate base rather than something a client can run, and is skipped.

    Args:
        base: The class whose subclass tree is walked.

    Returns:
        The concrete classes, depth-first and deduplicated.
    """
    found: list[type] = []
    seen: set[type] = set()

    def _walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                if getattr(sub, "name", ""):
                    found.append(sub)
            _walk(sub)

    _walk(base)
    return found


def discover_run_catalog() -> dict[str, type]:
    """Import the shipped procedures and catalog them by class name.

    A module that fails to import is logged and skipped: one broken procedure
    must not leave a client with no catalog at all.

    Returns:
        ``{class __name__: class}`` for every concrete procedure this
        installation ships — the mapping the Orchestrator, the run queue and
        the gateway's ``validate_run`` all resolve a run's class name
        through.
    """
    for package_name, _base in _PACKAGES:
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            logger.exception("ctl discovery: could not import %s", package_name)
            continue
        for _finder, module_name, _ispkg in pkgutil.iter_modules(
            [str(Path(package.__file__ or "").parent)]
        ):
            try:
                importlib.import_module(f"{package_name}.{module_name}")
            except Exception:  # noqa: BLE001 — one bad module, not no catalog
                logger.exception(
                    "ctl discovery: failed to import %s.%s", package_name, module_name
                )

    catalog: dict[str, type] = {}
    for _package_name, base in _PACKAGES:
        for cls in _named_subclasses(base):
            catalog[cls.__name__] = cls
    logger.debug("ctl discovery: %d runs in the catalog", len(catalog))
    return catalog
