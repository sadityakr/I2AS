"""The run queue: immutable specs, their ordering, and their validation.

The **run queue** holds what the operator asked for, not the machinery that
would carry it out. Every waiting run is a frozen ``RunSpec`` — a class name,
a validated parameter dict, the sample metadata and file prefix it was queued
with, and the ``Actor`` who queued it — so nothing in the queue holds engine
state, an open data file, or bus access. The one live object is constructed by
the engine, from the one spec it is about to start.

Two consequences follow, and they are the point of the design:

* **The engine pulls.** ``Orchestrator.run_queue()`` asks its injected
  ``next_procedure()`` callback for the next run and starts it itself. A
  client that instead watched ``state_changed`` for the IDLE state and
  started the next run would advance re-entrantly inside the engine's own
  emit, and could not tell a clean finish from a hold acknowledge, so it
  would auto-start a run straight after an emergency standby. Authority over
  *when* a run starts stays with the engine.
* **Entries are validated when they are added, not when they start.**
  ``validate_run()`` builds the run headlessly and reports its findings, so a
  parameter outside a declared bound, a setup limit, or the session envelope
  is refused at the moment the operator queues it rather than an hour later.

This module is headless and Qt-free by contract: it imports no widget and no
Orchestrator, so a queue can be built, ordered and validated in a test, in a
script, or in an agent gateway that has no GUI at all. It also never imports
``i2as.procedures`` (contract C11) — the classes a spec names are resolved
through a *run catalog* handed in by whoever owns discovery.
"""

from __future__ import annotations

import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

from i2as.core.estimates import estimate_duration
from i2as.core.events import OPERATOR, Actor
from i2as.core.plan import DurationEstimate, ProbeSpec
from i2as.core.run_builder import PROCEDURE_BUILD_ERRORS, build_procedure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from i2as.core.plan import ExperimentEnvelope
    from i2as.core.station import Station

logger = logging.getLogger(__name__)

#: A spec that names a measurement recipe (L4 ``BaseProcedure``).
KIND_PROCEDURE = "procedure"
#: The kinds a ``RunSpec`` may declare.
RUN_KINDS: frozenset[str] = frozenset({KIND_PROCEDURE})

#: The run refused to be built at all (see ``PROCEDURE_BUILD_ERRORS``).
FINDING_BUILD_REFUSED = "build_refused"
#: A supplied parameter the procedure does not declare.
FINDING_UNKNOWN_PARAM = "unknown_param"
#: A parameter outside its ``ParamSpec`` declaration (type, bounds, choices).
FINDING_PARAM_BOUNDS = "param_bounds"
#: A setpoint the run would command outside the setup's ``control_limits``.
FINDING_CONTROL_LIMIT = "control_limit"
#: A setpoint the run would command outside the open experiment's envelope.
FINDING_ENVELOPE = "envelope"


def _json_value(value: Any) -> Any:
    """Return *value* as a JSON-safe copy.

    Args:
        value: Any queued parameter value.

    Returns:
        A structure made only of ``str``/``int``/``float``/``bool``/``None``/
        ``list``/``dict``.

    Raises:
        TypeError: If the value is not JSON-safe. A spec is a contract type:
            it refuses eagerly rather than emitting an event a client cannot
            parse.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"value {value!r} of type {type(value).__name__} is not JSON-safe")


def _json_dict(value: Any, owner: str) -> dict[str, Any]:
    """Return a JSON-safe dict copy of *value*.

    Args:
        value: The mapping to copy, or ``None`` for an empty one.
        owner: Field name, for the error message.

    Returns:
        A fresh JSON-safe dict.

    Raises:
        TypeError: If *value* is not a mapping, or holds a non-JSON-safe value.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping, got {type(value).__name__}")
    return {str(key): _json_value(item) for key, item in value.items()}


@dataclass(frozen=True)
class RunSpec:
    """One queued run, as data: what to run, with which values, on whose behalf.

    Frozen and JSON-safe end to end, so the same object serves the queue, the
    ``QueueChanged`` event a client renders, and the persisted session state.
    It deliberately holds no procedure instance: a live run owns a data file
    and a claim on instruments, and neither belongs to something that is only
    waiting.

    Attributes:
        kind: ``"procedure"`` (see ``RUN_KINDS``).
        run_class: The class's ``__name__``, resolved against the run catalog
            when the engine pulls this spec.
        params: The run's own parameter values, already validated.
        sample_info: Sample metadata to record with the run.
        data_directory: Directory the run writes its data file into.
        file_prefix: Optional filename prefix.
        probe_spec: A ``ProbeSpec``'s dict form when this entry is a **probe
            run** — the same run reduced to a few cheap points — and ``{}``
            for an ordinary run. Stored as its dict rather than as the typed
            spec so a queue entry stays JSON-safe end to end.
        actor: Who queued it.
        spec_id: Stable identity of this queue entry, for remove/move.
        queued_at: Unix time it was queued.
    """

    kind: str
    run_class: str
    params: dict[str, Any] = field(default_factory=dict)
    sample_info: dict[str, Any] = field(default_factory=dict)
    data_directory: str = ""
    file_prefix: str = ""
    probe_spec: dict[str, Any] = field(default_factory=dict)
    actor: Actor = OPERATOR
    spec_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    queued_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Validate the kind and freeze JSON-safe copies of the dict fields.

        Raises:
            ValueError: If ``kind`` is not one of ``RUN_KINDS``, if
                ``run_class`` is empty, or if ``probe_spec`` is malformed.
            TypeError: If ``params``/``sample_info``/``probe_spec`` are not
                JSON-safe mappings, or ``actor`` is neither an ``Actor`` nor
                its dict.
        """
        if self.kind not in RUN_KINDS:
            raise ValueError(
                f"RunSpec.kind must be one of {sorted(RUN_KINDS)}, got {self.kind!r}"
            )
        if not isinstance(self.run_class, str) or not self.run_class:
            raise ValueError("RunSpec.run_class must be a non-empty class name")
        object.__setattr__(self, "params", _json_dict(self.params, "RunSpec.params"))
        probe_spec = _json_dict(self.probe_spec, "RunSpec.probe_spec")
        if probe_spec:
            if self.kind != KIND_PROCEDURE:
                raise ValueError(
                    f"RunSpec.probe_spec is only meaningful for a "
                    f"{KIND_PROCEDURE}, not a {self.kind}"
                )
            # Fail here rather than when the engine pulls the spec: a malformed
            # reduction is a request the operator can still fix.
            probe_spec = ProbeSpec.from_json(probe_spec).to_json()
        object.__setattr__(self, "probe_spec", probe_spec)
        object.__setattr__(
            self, "sample_info", _json_dict(self.sample_info, "RunSpec.sample_info")
        )
        actor = self.actor
        if isinstance(actor, dict):
            actor = Actor.from_json(actor)
        if not isinstance(actor, Actor):
            raise TypeError(f"RunSpec.actor must be an Actor, got {actor!r}")
        object.__setattr__(self, "actor", actor)
        for name in ("data_directory", "file_prefix"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"RunSpec.{name} must be a str")

    def to_json(self) -> dict[str, Any]:
        """Render this spec as a JSON-safe dict.

        Returns:
            The declared fields, with ``actor`` rendered as its own dict.
        """
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        payload["params"] = dict(self.params)
        payload["sample_info"] = dict(self.sample_info)
        payload["probe_spec"] = dict(self.probe_spec)
        payload["actor"] = self.actor.to_json()
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> RunSpec:
        """Rebuild a spec from its ``to_json()`` dict.

        Args:
            payload: A mapping as produced by ``to_json()``. Unknown keys are
                ignored, so a newer producer never breaks an older consumer.

        Returns:
            The spec.

        Raises:
            TypeError: If a declared field carries a value the type rejects.
            ValueError: If ``kind`` or ``run_class`` is invalid.
        """
        declared = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in declared})


@dataclass(frozen=True)
class RunFinding:
    """One reason a run was refused, or one caveat about it.

    Structured rather than prose so a client decides from ``code`` and the
    numbers and never by parsing the message.

    Attributes:
        code: Machine-readable category — one of the ``FINDING_*`` constants.
        message: Operator-facing explanation.
        param: The parameter or VI name the finding is about, or ``""``.
    """

    code: str
    message: str
    param: str = ""

    def to_json(self) -> dict[str, Any]:
        """Render this finding as a JSON-safe dict."""
        return {"code": self.code, "message": self.message, "param": self.param}


@dataclass(frozen=True)
class RunValidation:
    """The verdict on a proposed run, decided before anything is dispatched.

    Attributes:
        findings: Every problem found, in discovery order. Empty means the run
            is accepted.
        estimate: The **duration estimate** for the run that was built — its
            total, its per-phase breakdown and the assumptions behind it — or
            ``None`` when there was no run to estimate (the build was
            refused). Always qualified: a client renders the assumptions
            beside the number, never the number alone.
    """

    findings: tuple[RunFinding, ...] = ()
    estimate: DurationEstimate | None = None

    def __post_init__(self) -> None:
        """Freeze ``findings`` into a tuple and check the estimate's type.

        Raises:
            TypeError: If a finding is not a ``RunFinding``, or ``estimate``
                is neither a ``DurationEstimate`` nor ``None``.
        """
        findings = tuple(self.findings)
        for finding in findings:
            if not isinstance(finding, RunFinding):
                raise TypeError(
                    f"RunValidation.findings must hold RunFindings, got {finding!r}"
                )
        object.__setattr__(self, "findings", findings)
        if self.estimate is not None and not isinstance(self.estimate, DurationEstimate):
            raise TypeError(
                f"RunValidation.estimate must be a DurationEstimate or None, "
                f"got {self.estimate!r}"
            )

    @property
    def ok(self) -> bool:
        """True when nothing was found against the run."""
        return not self.findings

    @property
    def duration_estimate_s(self) -> float | None:
        """How long the run is expected to take, or ``None`` without an estimate.

        Derived from ``estimate`` rather than stored, so the headline number
        and the breakdown it came from can never disagree.
        """
        return None if self.estimate is None else self.estimate.total_s

    def messages(self) -> tuple[str, ...]:
        """Return each finding's message, for a log line or a dialog body."""
        return tuple(finding.message for finding in self.findings)

    def to_json(self) -> dict[str, Any]:
        """Render this validation as a JSON-safe dict."""
        return {
            "ok": self.ok,
            "findings": [finding.to_json() for finding in self.findings],
            "duration_estimate_s": self.duration_estimate_s,
            "estimate": None if self.estimate is None else self.estimate.to_json(),
        }


class RunQueue:
    """The ordered run queue: immutable specs, in the order they were added.

    Nothing here touches hardware, Qt, or the engine. Mutating methods return
    whether they changed anything, so a caller knows when to broadcast.
    """

    def __init__(self, specs: Iterable[RunSpec] = ()) -> None:
        """Build a queue, optionally pre-filled.

        Args:
            specs: Initial specs, added in order.
        """
        self._procedures: list[RunSpec] = []
        for spec in specs:
            self.add(spec)

    def _bucket(self, kind: str) -> list[RunSpec]:
        """Return the list holding specs of *kind*."""
        return self._procedures

    def _locate(self, spec_id: str) -> tuple[list[RunSpec], int] | None:
        """Return the bucket and index holding *spec_id*, or ``None``."""
        for index, spec in enumerate(self._procedures):
            if spec.spec_id == spec_id:
                return self._procedures, index
        return None

    def add(self, spec: RunSpec) -> RunSpec:
        """Append *spec* to the end of its own bucket.

        Args:
            spec: The spec to queue.

        Returns:
            The spec, so a caller can chain on its ``spec_id``.

        Raises:
            TypeError: If *spec* is not a ``RunSpec``.
            ValueError: If a spec with the same ``spec_id`` is already queued.
        """
        if not isinstance(spec, RunSpec):
            raise TypeError(f"RunQueue.add expects a RunSpec, got {spec!r}")
        if self._locate(spec.spec_id) is not None:
            raise ValueError(f"spec_id {spec.spec_id!r} is already queued")
        self._bucket(spec.kind).append(spec)
        logger.info(
            "Run queue: %s %s queued by %s (%d waiting)",
            spec.kind,
            spec.run_class,
            spec.actor.id,
            len(self),
        )
        return spec

    def remove(self, spec_id: str) -> bool:
        """Drop the spec with this id.

        Args:
            spec_id: The entry's identity.

        Returns:
            True if an entry was removed, False if no entry had that id.
        """
        found = self._locate(spec_id)
        if found is None:
            return False
        bucket, index = found
        removed = bucket.pop(index)
        logger.info("Run queue: %s %s removed", removed.kind, removed.run_class)
        return True

    def move(self, spec_id: str, offset: int) -> bool:
        """Move an entry *offset* places, clamped to the ends of the queue.

        Args:
            spec_id: The entry's identity.
            offset: Places to move it — negative towards the front.

        Returns:
            True if the order changed, False if the id is unknown, the offset
            is zero, or the entry is already at that end.
        """
        found = self._locate(spec_id)
        if found is None or offset == 0:
            return False
        bucket, index = found
        target = max(0, min(len(bucket) - 1, index + offset))
        if target == index:
            return False
        bucket.insert(target, bucket.pop(index))
        return True

    def clear(self) -> bool:
        """Empty the queue.

        Returns:
            True if anything was removed.
        """
        if not self._procedures:
            return False
        self._procedures.clear()
        logger.info("Run queue: cleared")
        return True

    def snapshot(self) -> tuple[RunSpec, ...]:
        """Return every waiting spec in the order they will run.

        Returns:
            The waiting specs, in queue order.
        """
        return tuple(self._procedures)

    def entries(self) -> tuple[dict[str, Any], ...]:
        """Return ``snapshot()`` rendered as JSON-safe dicts."""
        return tuple(spec.to_json() for spec in self.snapshot())

    def pop_next(self) -> RunSpec | None:
        """Remove and return the spec that should run next.

        Returns:
            The first waiting spec, else ``None`` for an empty queue.
        """
        return self._procedures.pop(0) if self._procedures else None

    def find(self, spec_id: str) -> RunSpec | None:
        """Return the queued spec with this id, or ``None``."""
        found = self._locate(spec_id)
        if found is None:
            return None
        bucket, index = found
        return bucket[index]

    def __len__(self) -> int:
        """Return how many specs are waiting."""
        return len(self._procedures)

    def __bool__(self) -> bool:
        """True while anything is waiting."""
        return bool(self._procedures)


def build_run(
    spec: RunSpec,
    *,
    station: Station,
    run_catalog: Mapping[str, type],
    experiment_info: Mapping[str, Any] | None = None,
) -> Any:
    """Construct the one live object a spec describes.

    The other half of the pull seam: the engine asks for the next run, and
    exactly one procedure instance comes into existence, for the run that is
    about to start, assembled by ``run_builder.build_procedure()`` — the one
    headless construction path, the same shape ``Orchestrator.submit()``
    honours.

    Args:
        spec: The queued spec.
        station: The Station the run will drive.
        run_catalog: ``{class __name__: class}``, supplied by whoever owns
            discovery — this module may not import ``i2as.procedures``
            (contract C11).
        experiment_info: Experiment context stamped onto the run, read at
            build time so a queued run belongs to the experiment that is open
            when it actually starts.

    Returns:
        A ready procedure instance, reduced to a **probe run**
        when the spec carries a ``probe_spec``, and carrying ``queued_by``:
        the spec's **Actor**, which becomes the **run owner** when the engine
        starts it (see ``Orchestrator._pull_next_run()``). It rides on the
        object because that object is all the engine ever receives — the
        spec stays here — and it is read duck-typed there, since contract C5
        forbids the engine from importing this module to learn the type.

    Raises:
        KeyError: If the catalog holds no class of that name.
        I2ASError: If the run refuses to be built.
        TypeError: If the stored parameters no longer fit the signature.
        ValueError: If a parameter value is invalid, or the named class
            cannot be probed.
    """
    run_class = run_catalog.get(spec.run_class)
    if run_class is None:
        raise KeyError(
            f"unknown {spec.kind} {spec.run_class!r}: the run catalog holds "
            f"{sorted(run_catalog)}"
        )
    run = build_procedure(
        run_class,
        station=station,
        params=dict(spec.params),
        sample_info=dict(spec.sample_info),
        data_directory=spec.data_directory,
        file_prefix=spec.file_prefix,
        experiment_info=dict(experiment_info) if experiment_info else None,
        probe=ProbeSpec.from_json(spec.probe_spec) if spec.probe_spec else None,
    )
    run.queued_by = spec.actor
    return run


def _accepts_extra_params(run_class: type) -> bool:
    """True if the class's ``__init__`` absorbs keyword arguments it never declared.

    The generic sweep procedures deliberately do: they resolve a measurement
    VI from the station and take ITS parameters alongside their own, so those
    values are legitimately absent from ``cls.parameters``. Only a class with
    a closed signature can be told that a supplied name is undeclared.

    Args:
        run_class: The procedure class being queued.

    Returns:
        True when ``__init__`` declares a ``**kwargs`` catch-all.
    """
    try:
        parameters = inspect.signature(run_class.__init__).parameters.values()
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)


def _declared_param_findings(
    run_class: type, params: Mapping[str, Any]
) -> list[RunFinding]:
    """Check supplied values against the class's own ``ParamSpec`` declarations.

    The first of ``validate_run()``'s three checks. Every declared parameter
    validates itself — type, inclusive bounds, enumerated choices — so a value
    the form or an agent produced is refused here rather than by whichever
    layer happens to notice it first.

    Args:
        run_class: The procedure class being queued.
        params: The supplied parameter values.

    Returns:
        One finding per violation, in declaration order, followed by one per
        undeclared parameter name.
    """
    declared: Mapping[str, Any] = getattr(run_class, "parameters", {}) or {}
    findings: list[RunFinding] = []
    for name, spec in declared.items():
        if name not in params:
            continue
        value = params[name]
        choices = getattr(spec, "choices", None)
        if choices:
            if value not in choices.values():
                findings.append(
                    RunFinding(
                        FINDING_PARAM_BOUNDS,
                        f"{name}={value!r} is not one of the declared choices "
                        f"{sorted(choices.values(), key=str)}",
                        name,
                    )
                )
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        low = getattr(spec, "min", None)
        high = getattr(spec, "max", None)
        if low is not None and value < low:
            findings.append(
                RunFinding(
                    FINDING_PARAM_BOUNDS,
                    f"{name}={value:g} is below the declared minimum {low:g}",
                    name,
                )
            )
        if high is not None and value > high:
            findings.append(
                RunFinding(
                    FINDING_PARAM_BOUNDS,
                    f"{name}={value:g} is above the declared maximum {high:g}",
                    name,
                )
            )
    if declared and not _accepts_extra_params(run_class):
        for name in params:
            if name not in declared:
                findings.append(
                    RunFinding(
                        FINDING_UNKNOWN_PARAM,
                        f"{run_class.__name__} declares no parameter {name!r}",
                        name,
                    )
                )
    return findings


def _target_findings(
    run: Any,
    *,
    station: Station,
    envelope: ExperimentEnvelope | None,
) -> list[RunFinding]:
    """Check the setpoints a built run would command, before any are dispatched.

    The other two of ``validate_run()``'s three checks, both over the same
    values: the run's own ``planned_targets()`` — the setpoints it declares it
    will command, per system VI. Each is compared against the setup's
    ``control_limits`` bound for that VI (``Station.envelope_variables()``, the
    read side of the control-validation standard) and against the open
    experiment's envelope, which narrows it.

    Args:
        run: The freshly built procedure.
        station: The Station it would drive.
        envelope: The open experiment's envelope, or ``None``.

    Returns:
        One finding per out-of-range setpoint, at most one per VI per bound.
    """
    planned = getattr(run, "planned_targets", None)
    if not callable(planned):
        return []
    try:
        targets: Mapping[str, Sequence[float]] = planned()
    except Exception:  # noqa: BLE001 — validation must never raise at its caller
        logger.exception("validate_run: planned_targets() failed")
        return []

    variables = station.envelope_variables()
    findings: list[RunFinding] = []
    for vi_name, values in targets.items():
        numbers = [float(value) for value in values]
        if not numbers:
            continue
        extremes = (min(numbers), max(numbers))
        variable = variables.get(vi_name)
        if variable is not None:
            for value in extremes:
                low, high = variable.config_min, variable.config_max
                if (low is not None and value < low) or (
                    high is not None and value > high
                ):
                    findings.append(
                        RunFinding(
                            FINDING_CONTROL_LIMIT,
                            f"{vi_name} setpoint {value:g} is outside the setup "
                            f"limit [{'-inf' if low is None else f'{low:g}'}, "
                            f"{'inf' if high is None else f'{high:g}'}]",
                            vi_name,
                        )
                    )
                    break
        if envelope is not None:
            for value in extremes:
                message = envelope.check_target(vi_name, value)
                if message is not None:
                    findings.append(RunFinding(FINDING_ENVELOPE, message, vi_name))
                    break
    return findings


def validate_run(
    run_class: type,
    params: Mapping[str, Any],
    *,
    station: Station,
    kind: str = KIND_PROCEDURE,
    sample_info: Mapping[str, Any] | None = None,
    data_directory: str = "",
    file_prefix: str = "",
    probe_spec: Mapping[str, Any] | None = None,
    experiment_info: Mapping[str, Any] | None = None,
    envelope: ExperimentEnvelope | None = None,
) -> RunValidation:
    """Decide whether a proposed run may be queued, without dispatching anything.

    Free of hardware and free of consequence: the run is built headlessly and
    thrown away, no target reaches the Station, no data file is opened. Three
    checks, in order — the declared ``ParamSpec`` bounds, the build itself
    (a procedure legitimately refuses a run this station cannot honour), and
    the setpoints the built run declares it would command, against the setup's
    ``control_limits`` and the open experiment's envelope. The same build then
    yields the **duration estimate** (``core.estimates.estimate_duration()``
    over the setup's nominal ramp rates), so "may I run this?" and "how long
    will it take?" are answered by one call and from one object.

    ``ExperimentManager.validate_run()`` is the L6 entry point that supplies
    the station and the open experiment's envelope; this function is what it
    calls, and what a caller with no session layer can call directly.

    Args:
        run_class: The procedure class being queued.
        params: The parameter values it would run with.
        station: The Station the run would drive.
        kind: ``"procedure"``.
        sample_info: Sample metadata the run would record.
        data_directory: Directory the run would write into. Never created or
            written here.
        file_prefix: Filename prefix the run would use.
        probe_spec: A ``ProbeSpec``'s dict form to validate the **probe run**
            variant instead of the full run — the reduced run is what gets
            built and checked, so its (shorter) target list and its estimate
            are what come back.
        experiment_info: Experiment context the run would be stamped with.
        envelope: The open experiment's envelope, or ``None`` for none.

    Returns:
        A ``RunValidation``; ``ok`` is True exactly when nothing was found,
        and ``estimate`` carries the duration estimate whenever the run built.
    """
    findings = _declared_param_findings(run_class, params)

    catalog = {run_class.__name__: run_class}
    spec = RunSpec(
        kind=kind,
        run_class=run_class.__name__,
        params=dict(params),
        sample_info=dict(sample_info or {}),
        data_directory=data_directory,
        file_prefix=file_prefix,
        probe_spec=dict(probe_spec or {}),
    )
    run: Any = None
    try:
        run = build_run(
            spec,
            station=station,
            run_catalog=catalog,
            experiment_info=experiment_info,
        )
    except PROCEDURE_BUILD_ERRORS as exc:
        findings.append(
            RunFinding(
                FINDING_BUILD_REFUSED,
                f"{getattr(run_class, 'name', run_class.__name__)} refused this "
                f"run: {exc}",
            )
        )

    estimate: DurationEstimate | None = None
    if run is not None:
        findings.extend(_target_findings(run, station=station, envelope=envelope))
        # The estimate is of the run that was actually built — a probe's
        # estimate is the probe's, not the full run's.
        estimate = estimate_duration(run, station.nominal_ramp_rates())

    return RunValidation(findings=tuple(findings), estimate=estimate)


class RunQueueHost:
    """A ``RunQueue`` plus the policy around it: validate, broadcast, build.

    The queue itself is pure ordering; this is where the three things that
    make it behave like the system's run queue live, in one place so nobody
    reimplements them:

    * **Validation on the way in.** ``add()`` runs ``validate_run()`` first
      and refuses a spec that fails, so nothing waiting is known-unrunnable.
    * **A broadcast on every change.** Each mutation that actually changed
      something calls the injected ``publish`` with the actor behind it,
      which is how a ``QueueChanged`` reaches the engine's one event stream
      even though the queue lives outside the engine.
    * **One construction on the way out.** ``next_run()`` pops one spec and
      builds the single live object it describes — the engine's pull seam.

    ``ExperimentManager`` wires one of these to the open experiment's context
    and envelope and exposes it as its own queue surface. A client with no
    session layer (the Procedure window in a bare unit test) builds one
    directly: the setup's own ``control_limits`` still guard every add, there
    is simply no experiment envelope to narrow them.
    """

    def __init__(
        self,
        *,
        station: Station | None = None,
        run_catalog: Mapping[str, type] | None = None,
        publish: Callable[[Actor], None] | None = None,
        experiment_info: Callable[[], Mapping[str, Any]] | None = None,
        envelope: Callable[[], ExperimentEnvelope | None] | None = None,
    ) -> None:
        """Build a queue and the policy around it.

        Args:
            station: The Station a queued run would drive. ``None`` leaves
                everything working except validation and the pull, which say
                so rather than guessing.
            run_catalog: ``{class __name__: class}`` a spec's class name is
                resolved through, supplied by whoever owns discovery.
            publish: Called with the actor after any mutation that changed
                the queue — normally ``Orchestrator.publish_queue``. ``None``
                means nothing is broadcast.
            experiment_info: Returns the experiment context to stamp onto a
                run, read at BUILD time so a run queued before an experiment
                was opened belongs to the one open when it starts.
            envelope: Returns the active session envelope, or ``None``.
        """
        self._station = station
        self._run_catalog: dict[str, type] = dict(run_catalog or {})
        self._publish = publish
        self._experiment_info = experiment_info
        self._envelope = envelope
        self._queue = RunQueue()

    @property
    def queue(self) -> RunQueue:
        """The underlying ordered queue (read; mutate through this host)."""
        return self._queue

    def snapshot(self) -> tuple[RunSpec, ...]:
        """Return every waiting spec, in the order they will run."""
        return self._queue.snapshot()

    def entries(self) -> tuple[dict[str, Any], ...]:
        """Return ``snapshot()`` as JSON-safe dicts, for ``QueueChanged``."""
        return self._queue.entries()

    def validate(
        self,
        run_class: type,
        params: Mapping[str, Any],
        *,
        kind: str = KIND_PROCEDURE,
        sample_info: Mapping[str, Any] | None = None,
        data_directory: str = "",
        file_prefix: str = "",
        probe_spec: Mapping[str, Any] | None = None,
    ) -> RunValidation:
        """Check a proposed run without dispatching anything.

        Args:
            run_class: The procedure class to check.
            params: The parameter values it would run with.
            kind: ``"procedure"``.
            sample_info: Sample metadata the run would record.
            data_directory: Directory the run would write into.
            file_prefix: Filename prefix the run would use.
            probe_spec: A ``ProbeSpec``'s dict form to check the **probe
                run** variant instead of the full run.

        Returns:
            A ``RunValidation``.

        Raises:
            RuntimeError: If this host has no Station, which makes a headless
                build impossible.
        """
        if self._station is None:
            raise RuntimeError(
                "validating a run needs a Station; this run queue was built "
                "without one"
            )
        return validate_run(
            run_class,
            params,
            station=self._station,
            kind=kind,
            sample_info=sample_info,
            data_directory=data_directory,
            file_prefix=file_prefix,
            probe_spec=probe_spec,
            experiment_info=self._experiment_info() if self._experiment_info else None,
            envelope=self._envelope() if self._envelope else None,
        )

    def add(
        self,
        run_class: type,
        params: Mapping[str, Any],
        *,
        kind: str = KIND_PROCEDURE,
        sample_info: Mapping[str, Any] | None = None,
        data_directory: str = "",
        file_prefix: str = "",
        probe_spec: Mapping[str, Any] | None = None,
        actor: Actor = OPERATOR,
    ) -> tuple[RunSpec | None, RunValidation]:
        """Validate a proposed run and, if it passes, queue it.

        Args:
            run_class: The procedure class to queue.
            params: The parameter values it will run with.
            kind: ``"procedure"``.
            sample_info: Sample metadata to record with the run.
            data_directory: Directory the run writes into.
            file_prefix: Optional filename prefix.
            probe_spec: A ``ProbeSpec``'s dict form to queue the **probe
                run** variant — validated and queued as the reduced run, so
                what waits in the queue is the probe itself.
            actor: Who is queueing it.

        Returns:
            ``(spec, validation)`` — *spec* is ``None`` when validation
            refused the run, and *validation.findings* says why.
        """
        validation = self.validate(
            run_class,
            params,
            kind=kind,
            sample_info=sample_info,
            data_directory=data_directory,
            file_prefix=file_prefix,
            probe_spec=probe_spec,
        )
        if not validation.ok:
            logger.info(
                "Refused to queue %s: %s",
                run_class.__name__,
                "; ".join(validation.messages()),
            )
            return None, validation
        spec = self._queue.add(
            RunSpec(
                kind=kind,
                run_class=run_class.__name__,
                params=dict(params),
                sample_info=dict(sample_info or {}),
                data_directory=data_directory,
                file_prefix=file_prefix,
                probe_spec=dict(probe_spec or {}),
                actor=actor,
            )
        )
        self._broadcast(actor)
        return spec, validation

    def add_spec(self, spec: RunSpec) -> RunSpec:
        """Queue an already-built spec, unvalidated.

        The restore path: a spec rebuilt from persisted session state is put
        back exactly as it was saved, so the operator sees the queue they
        left behind — a stored value that has since gone out of bounds is
        reported when the run is pulled, not silently dropped on load.

        Args:
            spec: The spec to queue.

        Returns:
            The queued spec.
        """
        queued = self._queue.add(spec)
        self._broadcast(spec.actor)
        return queued

    def remove(self, spec_id: str, *, actor: Actor = OPERATOR) -> bool:
        """Remove one waiting run.

        Args:
            spec_id: The entry's ``RunSpec.spec_id``.
            actor: Who is removing it.

        Returns:
            True if an entry was removed.
        """
        return self._changed(self._queue.remove(spec_id), actor)

    def move(self, spec_id: str, offset: int, *, actor: Actor = OPERATOR) -> bool:
        """Move one waiting run within its own half of the queue.

        Args:
            spec_id: The entry's ``RunSpec.spec_id``.
            offset: Places to move it — negative towards the front.
            actor: Who is reordering it.

        Returns:
            True if the order changed.
        """
        return self._changed(self._queue.move(spec_id, offset), actor)

    def clear(self, *, actor: Actor = OPERATOR) -> bool:
        """Empty the queue.

        Args:
            actor: Who is clearing it.

        Returns:
            True if anything was removed.
        """
        return self._changed(self._queue.clear(), actor)

    def take_next_spec(self) -> RunSpec | None:
        """Pop the next waiting spec and broadcast the shortened queue.

        The first half of ``next_run()``, exposed on its own because the two
        halves belong to different threads once the engine lives on the
        instrument thread: popping mutates THIS layer's queue and so must
        happen wherever the queue is edited, while building touches the
        Station and so must happen wherever the Station is. What crosses
        between them is a frozen ``RunSpec``, which is safe to hand anywhere.
        Callers on one thread use ``next_run()``, which is both halves.

        Returns:
            The spec that just left the queue, or ``None`` when nothing is
            waiting or this host has no Station/catalog to build with.
        """
        if self._station is None or not self._run_catalog:
            return None
        spec = self._queue.pop_next()
        if spec is None:
            return None
        # The spec has left the queue; a build that later refuses must not
        # leave clients rendering an entry that is gone.
        self._broadcast(spec.actor)
        return spec

    def build_spec(self, spec: RunSpec) -> Any:
        """Build the one live run a popped spec describes.

        The second half of ``take_next_spec()`` (see its docstring). Runs on
        whichever thread owns the Station.

        Args:
            spec: A spec this host has already popped.

        Returns:
            A ready procedure.

        Raises:
            KeyError: If the catalog holds no class of the spec's name.
            I2ASError: If the run refuses to be built.
            TypeError: If the stored parameters no longer fit the signature.
            ValueError: If a parameter value is invalid.
        """
        return build_run(
            spec,
            station=self._station,
            run_catalog=self._run_catalog,
            experiment_info=(
                self._experiment_info() if self._experiment_info else None
            ),
        )

    def next_run(self) -> Any:
        """Build and return the run the engine should start next.

        Both halves at once, for a client on the engine's own thread. A client
        across the instrument thread calls ``take_next_spec()`` and
        ``build_spec()`` separately, each on the thread that owns what it
        touches.

        Returns:
            A ready procedure, or ``None`` when the queue is
            empty or this host has no Station/catalog to build with.

        Raises:
            KeyError: If the catalog holds no class of the spec's name.
            I2ASError: If the run refuses to be built.
            TypeError: If the stored parameters no longer fit the signature.
            ValueError: If a parameter value is invalid.
        """
        if self._station is None or not self._run_catalog:
            return None
        spec = self._queue.pop_next()
        if spec is None:
            return None
        try:
            return build_run(
                spec,
                station=self._station,
                run_catalog=self._run_catalog,
                experiment_info=(
                    self._experiment_info() if self._experiment_info else None
                ),
            )
        finally:
            # The spec has left the queue either way; a build that refused
            # must not leave clients rendering an entry that is gone.
            self._broadcast(spec.actor)

    def _changed(self, changed: bool, actor: Actor) -> bool:
        """Broadcast if a mutation actually changed the queue.

        Args:
            changed: What the ``RunQueue`` method reported.
            actor: Who made the change.

        Returns:
            *changed*, unchanged.
        """
        if changed:
            self._broadcast(actor)
        return changed

    def _broadcast(self, actor: Actor) -> None:
        """Tell the injected publisher the queue changed, if there is one."""
        if self._publish is not None:
            self._publish(actor)
