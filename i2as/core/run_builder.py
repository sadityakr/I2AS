"""Headless construction of a run — a procedure — from plain values.

The one place a ``BaseProcedure`` is instantiated. Every caller — the
Procedure window's run-now and add-to-queue flows, the queue's session
restore, the engine port's dict payloads, and any future non-GUI client —
routes through ``build_procedure`` so a run is assembled identically no
matter who asked for it. The constructor shape it writes down is the one
every procedure declares: the Station, the run's data context (sample,
directory, prefix, experiment) and its own parameters as keywords.

Headless by contract: this module imports no Qt and touches no widget, so a
procedure can be built in a test, from a script, or from a client that has no
GUI at all. That is the point of it living here rather than on a window —
construction is domain work, not presentation.

Construction failure is signalled by raising, never by returning ``None``: a
procedure legitimately refuses a run it cannot honour (a nonzero field on a
station with no magnet, a parameter outside the setup's limits), and the
reason belongs in the caller's own error channel — a form dialog for the
operator, a log line for a restore, an error payload for a remote client.
``PROCEDURE_BUILD_ERRORS`` names the exceptions that mean "refused", so
callers catch one well-known tuple instead of each guessing a different subset
of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from i2as.core.exceptions import I2ASError

if TYPE_CHECKING:  # pragma: no cover - typing only

    from i2as.core.plan import ProbeSpec
    from i2as.core.station import Station

#: The run class being built, so the caller gets back exactly the type it
#: asked for. Deliberately unbound rather than ``bound=BaseProcedure``: this
#: module sits below L4 and must not import it — a real dependency would make
#: every importer of the builder an importer of the data manager (contract
#: C5). The keyword contract below is what a procedure must satisfy, and
#: ``tests/test_conformance.py`` is what holds every procedure to it.
RunT = TypeVar("RunT")

#: Exceptions that mean "this procedure refused to be built", as opposed to a
#: programming error. ``I2ASError`` covers ``I2ASConfigError`` and
#: ``I2ASSafetyError``, which procedures raise from ``__init__`` when the
#: station cannot honour the requested run; ``TypeError``/``ValueError`` cover
#: a stored parameter set that no longer matches the procedure's signature
#: (e.g. a queue entry restored after the procedure gained or renamed a
#: parameter). Catch this tuple rather than a hand-picked subset: catching
#: only ``(TypeError, ValueError)`` silently excludes every ``I2ASError``,
#: because ``I2ASError`` derives from ``Exception`` directly.
PROCEDURE_BUILD_ERRORS: tuple[type[BaseException], ...] = (
    I2ASError,
    TypeError,
    ValueError,
)


def build_procedure(
    cls: type[RunT],
    *,
    station: Station,
    params: dict[str, Any],
    sample_info: dict[str, str],
    data_directory: str,
    file_prefix: str = "",
    experiment_info: dict[str, Any] | None = None,
    probe: ProbeSpec | None = None,
) -> RunT:
    """Instantiate *cls* from plain values.

    Args:
        cls: The concrete ``BaseProcedure`` subclass to build.
        station: The Station the run will drive.
        params: The procedure's own declared parameters, passed as keyword
            arguments. Keys must match the procedure's ``__init__`` signature.
        sample_info: Sample metadata recorded with the run.
        data_directory: Directory the run writes its HDF5 file into. Named for
            the constructor keyword, not for any caller's field name.
        file_prefix: Optional filename prefix for the run's data file.
        experiment_info: Experiment context stamped onto the run, or ``None``
            when no experiment is open. Forwarded as-is; ``BaseProcedure``
            already records ``None`` as ``{}``.
        probe: Reduce the built run to a **probe run** (see ``ProbeSpec``'s
            reduction rules). ``None`` — the default — builds the run as
            requested. The reduction is applied AFTER construction, on the
            instance, so a probe is only ever a reduction of a run the
            procedure already accepted.

    Returns:
        A ready instance of *cls*, reduced to a probe when *probe* was given.

    Raises:
        I2ASError: If the procedure refuses the run — the station cannot
            honour it, or a value violates a setup limit.
        TypeError: If *params* does not match the procedure's signature.
        ValueError: If *probe* was given and *cls* has no ``apply_probe()``
            (it is not a ``BaseProcedure``), or a parameter value is invalid.
    """
    run = cls(
        station=station,
        sample_info=sample_info,
        data_directory=data_directory,
        file_prefix=file_prefix,
        experiment_info=experiment_info,
        **params,
    )
    if probe is not None:
        apply_probe = getattr(run, "apply_probe", None)
        if not callable(apply_probe):
            raise ValueError(
                f"{cls.__name__} cannot be probed: it declares no "
                f"apply_probe() (see BaseProcedure)"
            )
        apply_probe(probe)
    return run


