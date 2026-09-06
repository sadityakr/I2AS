"""Procedure auto-discovery for the GUI."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path

from i2as.core.procedure import BaseProcedure

logger = logging.getLogger(__name__)


def all_subclasses(cls: type) -> list[type]:
    """Return every transitive subclass of *cls* (depth-first, deduplicated).

    ``type.__subclasses__()`` lists only *direct* subclasses, so it would miss a
    concrete procedure sitting under an intermediate base such as
    ``SweepMeasureProcedure``. This walks the whole tree.

    Args:
        cls: The base class whose subclass tree is walked.

    Returns:
        All transitive subclasses, in depth-first discovery order.
    """
    result: list[type] = []
    seen: set[type] = set()
    for sub in cls.__subclasses__():
        if sub not in seen:
            seen.add(sub)
            result.append(sub)
        result.extend(all_subclasses(sub))
    return result


def discover_procedures() -> list[type[BaseProcedure]]:
    """Import all modules in i2as.procedures and return concrete procedures.

    Returns every named ``BaseProcedure`` subclass at any depth (so a procedure
    under an intermediate base like ``SweepMeasureProcedure`` is found), skipping
    unnamed intermediate bases.

    Returns:
        List of concrete BaseProcedure subclasses (not the base or intermediate
        bases, which carry no ``name``).
    """
    import i2as.procedures as _pkg

    pkg_path = Path(_pkg.__file__).parent
    for _, module_name, _ in pkgutil.iter_modules([str(pkg_path)]):
        try:
            importlib.import_module(f"i2as.procedures.{module_name}")
        except Exception:
            logger.exception("procedure_discovery: failed to import i2as.procedures.%s", module_name)

    subclasses: list[type[BaseProcedure]] = []
    seen: set[type] = set()
    for cls in all_subclasses(BaseProcedure):
        if getattr(cls, "name", "") and cls not in seen:
            seen.add(cls)
            subclasses.append(cls)
    return subclasses
