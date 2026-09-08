"""The run catalog, and the procedure declarations rendered from it.

Two jobs that belong together because both need the procedure classes
themselves: finding them, and turning what they declare into the frozen,
JSON-safe ``ProcedureInfo`` the rest of the system passes around.

**Why discovery lives here.** A run travels as a class NAME, and whoever
owns discovery resolves that name and hands the catalog down: neither the
engine (contract C5) nor the session layer (C11) may import
``i2as.procedures``. Discovery used to be written twice, once for the GUI
and once for the CLI, which meant two walks that could disagree about which
procedures a setup ships — and left the MCP adapter with neither. It is one
walk here, in ``core``, so every client sees the same catalog; that is the
same rule the instrument declarations already follow.

**Why the Station does not call this.** Contract C4 forbids
``i2as.core.station`` from importing ``i2as.core.procedure``, and rightly:
the Station sits below procedures and must not know what one is. So the
Station holds procedure declarations but never builds them —
``build_procedure_infos()`` is called by whoever owns the catalog (the
application entry point, the CLI, the manifest command) and the result is
handed to ``Station.declare_procedures()``. The Station stores frozen
contract messages, not classes, so nothing about a procedure leaks below it.

**No bus traffic, ever.** Rendering a declaration reads class attributes and
the Station's own declarations — never an instrument. This is what lets the
procedures section be built for a rack whose instruments are all offline,
and it is checked against spied drivers in ``tests/test_conformance.py``,
exactly as ``Station.station_info()`` is.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from i2as.core.events import ProcedureFormBlock, ProcedureInfo
from i2as.core.exceptions import I2ASConfigError
from i2as.core.plan import ConditionalGroup, ParamSpec, validate_form
from i2as.core.procedure import BaseProcedure

if TYPE_CHECKING:  # pragma: no cover — typing only, never imported at runtime
    from i2as.core.station import Station

logger = logging.getLogger(__name__)

__all__ = ["discover_run_catalog", "build_procedure_infos"]

#: The package walked, and the base its classes must derive from.
_PROCEDURE_PACKAGE = "i2as.procedures"


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
        the gateway's ``run_procedure`` / ``validate_run`` all resolve a run's
        class name through.
    """
    try:
        package = importlib.import_module(_PROCEDURE_PACKAGE)
    except ImportError:
        logger.exception("procedure discovery: could not import %s", _PROCEDURE_PACKAGE)
        return {}

    for _finder, module_name, _ispkg in pkgutil.iter_modules(
        [str(Path(package.__file__ or "").parent)]
    ):
        try:
            importlib.import_module(f"{_PROCEDURE_PACKAGE}.{module_name}")
        except Exception:  # noqa: BLE001 — one bad module, not no catalog
            logger.exception(
                "procedure discovery: failed to import %s.%s",
                _PROCEDURE_PACKAGE,
                module_name,
            )

    catalog = {cls.__name__: cls for cls in _named_subclasses(BaseProcedure)}
    logger.debug("procedure discovery: %d runs in the catalog", len(catalog))
    return catalog


def build_procedure_infos(
    station: Station, run_catalog: dict[str, type]
) -> tuple[ProcedureInfo, ...]:
    """Render every catalogued procedure's declaration against one station.

    Args:
        station: The built Station the declarations are resolved against — it
            supplies every station-dependent choice list (role candidates,
            measurement VIs and their parameters). Read for declarations only.
        run_catalog: ``{class name: class}``, as ``discover_run_catalog()``
            returns.

    Returns:
        One ``ProcedureInfo`` per catalogued procedure, in catalog order.
    """
    return tuple(
        _procedure_info(class_name, cls, station)
        for class_name, cls in run_catalog.items()
    )


def _procedure_info(
    class_name: str, cls: type[BaseProcedure], station: Station
) -> ProcedureInfo:
    """Render one procedure's declaration, flagging what this rack cannot run.

    A procedure whose form cannot be built here is still declared, with an
    availability tag saying why and an empty form — the same choice
    ``InstrumentInfo`` makes for an offline instrument. Dropping it instead
    would leave a reader unable to tell "this setup does not ship that
    procedure" from "this setup cannot currently run it", and the second is
    a fact worth reading: it names the instrument the rack is missing.

    Args:
        class_name: The catalog key, which is the class's ``__name__``.
        cls: The procedure class.
        station: The Station the form is resolved against.

    Returns:
        The procedure's declaration.
    """
    availability: list[str] = []
    blocks: tuple[ConditionalGroup, ...] = ()
    try:
        blocks = cls.declaration_form(station)
    except I2ASConfigError as exc:
        # The one expected failure: a procedure that needs an instrument this
        # rack has not configured. Declared-but-unavailable, never fatal.
        availability.append("unavailable")
        logger.info("%s cannot be run on this station: %s", class_name, exc)
    except Exception:  # noqa: BLE001 — one bad procedure, not no catalog
        availability.append("declaration_failed")
        logger.exception("%s: could not build its form declaration", class_name)

    # A REQUIRED role with no candidate refuses the run at construction (the
    # role-discovery standard), and contributes no parameter to the form — so
    # without this the procedure would be declared complete and simply fail
    # when someone ran it. Naming the role is the useful part: it says which
    # instrument the rack is missing.
    for param_name, role in cls.role_parameters.items():
        if role.required and not role.candidates(station):
            availability.append(f"missing_role:{param_name}")

    for error in validate_form(blocks):
        logger.error("%s: %s", class_name, error)

    axis = cls.sweep_axis
    return ProcedureInfo(
        class_name=class_name,
        name=cls.name,
        description=cls.description,
        run_kind=cls.run_kind,
        availability=availability,
        sweep_axis=(
            {
                "key": axis.key,
                "unit": axis.unit,
                "data_key": axis.data_key,
                "description": axis.description,
                "default_start": axis.default_start,
                "default_end": axis.default_end,
                "default_steps": axis.default_steps,
            }
            if axis
            else {}
        ),
        roles={name: role.description for name, role in cls.role_parameters.items()},
        data_keys={
            "sweep": list(cls.sweep_data_keys),
            "measurement": list(cls.measurement_data_keys),
            "default_x": cls.default_x_key,
        },
        form=tuple(_form_block(block) for block in blocks),
    )


def _form_block(block: ConditionalGroup) -> ProcedureFormBlock:
    """Render one guarded block as its contract message.

    Args:
        block: The declared block.

    Returns:
        The wire form, parameters in declared order.
    """
    return ProcedureFormBlock(
        key=block.key,
        title=block.title,
        params=tuple(
            _param_json(name, spec) for name, spec in block.params.items()
        ),
        when={name: list(values) for name, values in block.when.conditions.items()},
    )


def _param_json(name: str, spec: ParamSpec) -> dict[str, Any]:
    """Render one procedure parameter for the declaration snapshot.

    The procedure-side counterpart of the ``@control`` parameter rendering in
    ``core.station``, and deliberately not the same shape: a procedure
    parameter is always declared (there is no signature to fall back on, so
    no ``declared`` flag), and it carries two fields a control parameter has
    no use for — ``structural``, which says whether changing it changes which
    blocks apply, and ``widget_hint``.

    Args:
        name: The parameter's name.
        spec: Its declared ``ParamSpec``.

    Returns:
        A JSON-safe dict of the declaration.
    """
    return {
        "name": name,
        "kind": spec.type.__name__,
        "unit": spec.unit,
        "description": spec.description,
        "default": spec.default,
        "min": spec.min,
        "max": spec.max,
        "choices": dict(spec.choices) if spec.choices else None,
        "structural": spec.structural,
        "widget_hint": spec.widget_hint or "",
    }
