"""Typed vocabulary of frozen dataclasses shared across all I2AS layers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from i2as.core.exceptions import DataSchemaError

__all__ = [
    "Target",
    "Command",
    "PhasePlan",
    "StepPlan",
    "ParamSpec",
    "ParamGroup",
    "UIGroup",
    "DataSchema",
    "DurationEstimate",
    "EnvelopeBound",
    "EnvelopeVariable",
    "ExperimentEnvelope",
    "ProbeSpec",
    "StepCost",
    "SETPOINT_PARAM_PREFIX",
    "params_digest",
]


def params_digest(params: Mapping[str, Any] | None) -> str:
    """Return the **Params digest** of one parameter set.

    A record that says a physical step was confirmed is only evidence if it
    also says WHAT was confirmed. Parameters live in several places by then —
    a run manifest, a queue entry, an operation's own defaults — and a
    dispute months later cannot tell whether the values in one of them are
    the values that were actually in force. So a confirmation stores this
    digest of the parameters as they stood at that instant: two records agree
    about the parameters exactly when their digests match, without either
    having to carry a copy of them.

    **The canonicalisation**, which is what makes the digest stable and is
    therefore part of the standard rather than an implementation detail: the
    mapping is rendered by ``json.dumps`` with keys sorted, no whitespace
    (``", "``/``": "`` collapsed to ``","``/``":"``), floats written as
    ``repr()`` gives them (the shortest text that round-trips back to the
    same double, which is what ``json`` does for a float by construction),
    non-ASCII characters left as themselves, and any value JSON cannot
    render itself (a numpy scalar, an enum, a path) replaced by its
    ``str()`` — the same degrade-never-fail rule the contract's own
    rendering follows. That UTF-8 text is hashed with SHA-256 and returned
    as its 64-character lowercase hex digest. Key ORDER therefore never matters and key SPELLING always
    does; ``None`` and an empty mapping give the same digest, because "no
    parameters" is one fact, not two.

    Args:
        params: The parameter mapping to digest, or ``None`` for "none".

    Returns:
        The 64-character lowercase SHA-256 hex digest of the canonical JSON.
    """
    canonical = json.dumps(
        dict(params or {}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Scalar Python types accepted for GUI-facing parameters and their HDF5 dtypes.
_PARAM_TYPES: tuple[type, ...] = (float, int, str, bool)
_ALLOWED_DTYPES: frozenset[str] = frozenset({"float", "int"})


def _is_real_number(value: Any) -> bool:
    """Return True if ``value`` is a real int or float, explicitly rejecting bool.

    ``bool`` is a subclass of ``int`` in Python, so ``isinstance(True, int)`` is
    True. Every numeric field in this module means a physical quantity, never a
    flag, so a stray ``True`` must not silently become ``1.0``.

    Args:
        value: The candidate to test.

    Returns:
        True for a non-bool ``int`` or ``float``, False otherwise.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class Target:
    """A desired end-state for one system variable (e.g. field, temperature).

    Frozen dataclass: once built it cannot be mutated, so a plan cannot be
    edited out from under the orchestrator after it is submitted.

    Attributes:
        target: The value to reach, in SI units (Tesla, Kelvin, Ampere …).
            Must be a finite real number; ``bool`` is rejected.
        rate: Optional ramp rate, forwarded verbatim to the target VI's
            ``start_ramp()`` — its unit is whatever that VI's ramp-rate unit
            is (e.g. K/min for the temperature controllers). If given it must
            be a finite real number strictly greater than zero.
    """

    target: float
    rate: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalise the fields.

        Raises:
            TypeError: If ``target``/``rate`` is not a real number.
            ValueError: If ``target`` is non-finite, or ``rate`` is non-finite
                or not strictly positive.
        """
        if not _is_real_number(self.target):
            raise TypeError(f"Target.target must be a real number, got {self.target!r}")
        if not math.isfinite(self.target):
            raise ValueError(f"Target.target must be finite, got {self.target!r}")
        object.__setattr__(self, "target", float(self.target))

        if self.rate is not None:
            if not _is_real_number(self.rate):
                raise TypeError(f"Target.rate must be a real number, got {self.rate!r}")
            if not math.isfinite(self.rate):
                raise ValueError(f"Target.rate must be finite, got {self.rate!r}")
            if self.rate <= 0:
                raise ValueError(f"Target.rate must be > 0, got {self.rate!r}")
            object.__setattr__(self, "rate", float(self.rate))


@dataclass(frozen=True)
class Command:
    """A single method call to dispatch on a named virtual instrument.

    The orchestrator is the sole writer to hardware; a Command is the typed
    request a procedure hands it — "call ``method`` on VI ``vi_name`` with
    ``kwargs``". ``kwargs`` is defensively copied so a caller that later mutates
    the dict it passed in cannot change this command's arguments.

    Attributes:
        vi_name: Name of the target virtual instrument. Non-empty string.
        method: Name of the VI method to call. Non-empty string and a valid
            Python identifier.
        kwargs: Keyword arguments for the call. Defensively copied.
    """

    vi_name: str
    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the fields and defensively copy ``kwargs``.

        Raises:
            TypeError: If ``vi_name``/``method`` is not a string, or ``kwargs``
                is not a dict.
            ValueError: If ``vi_name``/``method`` is empty, or ``method`` is not
                a valid Python identifier.
        """
        if not isinstance(self.vi_name, str):
            raise TypeError(f"Command.vi_name must be a str, got {self.vi_name!r}")
        if not self.vi_name:
            raise ValueError("Command.vi_name must be a non-empty str")
        if not isinstance(self.method, str):
            raise TypeError(f"Command.method must be a str, got {self.method!r}")
        if not self.method:
            raise ValueError("Command.method must be a non-empty str")
        if not self.method.isidentifier():
            raise ValueError(
                f"Command.method must be a valid Python identifier, got {self.method!r}"
            )
        if not isinstance(self.kwargs, dict):
            raise TypeError(f"Command.kwargs must be a dict, got {self.kwargs!r}")
        object.__setattr__(self, "kwargs", dict(self.kwargs))


def _validate_targets(targets: Any, owner: str) -> dict[str, Target]:
    """Validate a ``{name: Target}`` mapping and return a defensive copy.

    Args:
        targets: The mapping to validate.
        owner: Name of the owning class, used in error messages.

    Returns:
        A shallow copy of ``targets``.

    Raises:
        TypeError: If ``targets`` is not a mapping, a key is not a string, or a
            value is not a ``Target``.
        ValueError: If a key is an empty string.
    """
    if not isinstance(targets, dict):
        raise TypeError(f"{owner}.targets must be a dict, got {targets!r}")
    for name, tgt in targets.items():
        if not isinstance(name, str):
            raise TypeError(f"{owner}.targets key must be a str, got {name!r}")
        if not name:
            raise ValueError(f"{owner}.targets key must be a non-empty str")
        if not isinstance(tgt, Target):
            raise TypeError(
                f"{owner}.targets[{name!r}] must be a Target, got {tgt!r}"
            )
    return dict(targets)


def _validate_wait(wait_s: Any, owner: str) -> float:
    """Validate a non-negative wait time and return it coerced to float.

    Args:
        wait_s: The wait time in seconds.
        owner: Name of the owning class, used in error messages.

    Returns:
        ``wait_s`` as a float.

    Raises:
        TypeError: If ``wait_s`` is not a real number.
        ValueError: If ``wait_s`` is non-finite or negative.
    """
    if not _is_real_number(wait_s):
        raise TypeError(f"{owner}.wait_s must be a real number, got {wait_s!r}")
    if not math.isfinite(wait_s):
        raise ValueError(f"{owner}.wait_s must be finite, got {wait_s!r}")
    if wait_s < 0:
        raise ValueError(f"{owner}.wait_s must be >= 0, got {wait_s!r}")
    return float(wait_s)


@dataclass(frozen=True)
class PhasePlan:
    """What a procedure's ``initiate()`` and ``standby()`` return.

    A phase plan bundles the system targets to reach, an ORDERED sequence of
    virtual-instrument commands, and a settle time. Command order is
    semantically meaningful — e.g. a switch heater must settle before a source
    arms — so ``commands`` is normalised to a tuple and never reordered. The
    ``targets`` dict is defensively copied.

    Attributes:
        targets: Mapping of variable name to desired ``Target``. Defensively
            copied.
        commands: Ordered VI commands to dispatch, normalised to a tuple.
        wait_s: Settle time in seconds after applying targets/commands. Finite
            and non-negative.
        claim_commands: Ordered ``initiate()`` commands for this run's claimed
            VIs (see ``BaseProcedure._claim_initiate_commands()``), normalised
            to a tuple. Dispatched by the Orchestrator FIRST, before
            ``targets`` and ``commands`` — a claimed VI must already be in its
            standard operating state (e.g. a temperature controller's heater
            in closed-loop AUTO, not left MANUAL from a prior bench test)
            before this plan's own targets/commands assume that state. Empty
            by default; only ``initiate()`` plans populate it — ``standby()``
            plans have nothing to claim-initiate.
    """

    targets: dict[str, Target]
    commands: tuple[Command, ...] = ()
    wait_s: float = 0.0
    claim_commands: tuple[Command, ...] = ()

    def __post_init__(self) -> None:
        """Validate the fields; copy ``targets`` and normalise the command tuples.

        Raises:
            TypeError: If ``targets`` / a command / ``wait_s`` has the wrong type.
            ValueError: If a target key is empty, or ``wait_s`` is invalid.
        """
        object.__setattr__(self, "targets", _validate_targets(self.targets, "PhasePlan"))

        for field_name in ("commands", "claim_commands"):
            commands = tuple(getattr(self, field_name))
            for i, cmd in enumerate(commands):
                if not isinstance(cmd, Command):
                    raise TypeError(
                        f"PhasePlan.{field_name}[{i}] must be a Command, got {cmd!r}"
                    )
            object.__setattr__(self, field_name, commands)

        object.__setattr__(self, "wait_s", _validate_wait(self.wait_s, "PhasePlan"))


@dataclass(frozen=True)
class StepPlan:
    """What ``change_sweep_step()`` returns for the next sweep point.

    The orchestrator calls ``change_sweep_step()`` before each measurement; it
    returns a ``StepPlan`` for the next point, or ``None`` when the sweep is
    done. The ``targets`` dict is defensively copied.

    Attributes:
        targets: Mapping of variable name to desired ``Target`` for this point.
            Defensively copied.
        wait_s: Settle time in seconds before measuring. Finite and
            non-negative.
    """

    targets: dict[str, Target]
    wait_s: float

    def __post_init__(self) -> None:
        """Validate the fields and defensively copy ``targets``.

        Raises:
            TypeError: If ``targets`` or ``wait_s`` has the wrong type.
            ValueError: If a target key is empty, or ``wait_s`` is invalid.
        """
        object.__setattr__(self, "targets", _validate_targets(self.targets, "StepPlan"))
        object.__setattr__(self, "wait_s", _validate_wait(self.wait_s, "StepPlan"))


@dataclass(frozen=True)
class ParamSpec:
    """One GUI-facing procedure-parameter declaration.

    Replaces the per-parameter spec dicts (``{"type": float, "default": ...}``)
    procedures declare today. This is purely semantic — it names no Qt widget
    classes; ``widget_hint`` is an optional free-form hint, not a widget name.
    Setting ``structural=True`` means changing this parameter's value changes
    *which* parameter groups exist, so the GUI must re-derive the form when it
    changes. The ``choices`` dict is defensively copied.

    Attributes:
        type: The Python scalar type of the value: one of ``float, int, str,
            bool``.
        default: The initial value. Must be an instance of ``type`` (an ``int``
            is accepted for ``type=float``; a ``bool`` never satisfies a numeric
            type).
        unit: SI unit label for display (e.g. "T", "K"). GUI concern only.
        description: Human-readable help text.
        min: Optional inclusive lower bound. Numeric types only; excludes
            ``choices``.
        max: Optional inclusive upper bound. Numeric types only; excludes
            ``choices``.
        choices: Optional non-empty label→value dict rendering as a drop-down.
            Every value must be an instance of ``type`` and ``default`` must
            equal one of them. Mutually exclusive with ``min``/``max``.
            Defensively copied.
        structural: If True, changing this value re-derives the whole form.
        widget_hint: Optional non-empty display hint (e.g. "slider"); never a
            concrete Qt widget class name.
    """

    type: type
    default: Any
    unit: str = ""
    description: str = ""
    min: float | None = None
    max: float | None = None
    choices: dict[str, Any] | None = None
    structural: bool = False
    widget_hint: str | None = None

    def __post_init__(self) -> None:
        """Validate the declaration and defensively copy ``choices``.

        Raises:
            TypeError: If ``type`` is not one of the allowed scalar types, or a
                string/flag field has the wrong type.
            ValueError: If ``default`` does not match ``type``; if bounds are
                given for a non-numeric type, are inconsistent, or coexist with
                ``choices``; if ``choices`` is empty, contains a wrong-typed
                value, or excludes ``default``; or if ``widget_hint`` is empty.
        """
        if self.type not in _PARAM_TYPES:
            raise TypeError(
                f"ParamSpec.type must be one of (float, int, str, bool), "
                f"got {self.type!r}"
            )

        if not self._matches_type(self.default):
            raise ValueError(
                f"ParamSpec.default {self.default!r} is not a {self.type.__name__}"
            )

        for label, val in (("unit", self.unit), ("description", self.description)):
            if not isinstance(val, str):
                raise TypeError(f"ParamSpec.{label} must be a str, got {val!r}")

        if not isinstance(self.structural, bool):
            raise TypeError(
                f"ParamSpec.structural must be a bool, got {self.structural!r}"
            )

        if self.widget_hint is not None:
            if not isinstance(self.widget_hint, str):
                raise TypeError(
                    f"ParamSpec.widget_hint must be a str, got {self.widget_hint!r}"
                )
            if not self.widget_hint:
                raise ValueError("ParamSpec.widget_hint must be a non-empty str")

        if self.choices is not None:
            self._validate_choices()
        else:
            self._validate_bounds()

    def _matches_type(self, value: Any) -> bool:
        """Return True if ``value`` is a legal instance of ``self.type``.

        Applies the numeric nuance: an ``int`` is accepted where ``float`` is
        declared, but a ``bool`` never satisfies ``int`` or ``float`` (it must
        be checked before the ``int`` acceptance because ``bool`` subclasses
        ``int``).

        Args:
            value: The candidate value.

        Returns:
            True if ``value`` is acceptable for ``self.type``.
        """
        if self.type is bool:
            return isinstance(value, bool)
        if isinstance(value, bool):
            return False  # bool is never a valid int/float/str here
        if self.type is float:
            return isinstance(value, (int, float))
        return isinstance(value, self.type)

    def _validate_bounds(self) -> None:
        """Validate ``min``/``max`` when no ``choices`` are declared.

        Raises:
            TypeError: If a bound is not a real number.
            ValueError: If bounds are given for a non-numeric type, if
                ``min > max``, or if ``default`` falls outside the bounds.
        """
        if self.min is None and self.max is None:
            return
        if self.type not in (int, float):
            raise ValueError(
                f"ParamSpec.min/max are only valid for numeric types, "
                f"not {self.type.__name__}"
            )
        for name, bound in (("min", self.min), ("max", self.max)):
            if bound is not None and not _is_real_number(bound):
                raise TypeError(
                    f"ParamSpec.{name} must be a real number, got {bound!r}"
                )
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f"ParamSpec.min {self.min!r} must be <= max {self.max!r}"
            )
        if self.min is not None and self.default < self.min:
            raise ValueError(
                f"ParamSpec.default {self.default!r} is below min {self.min!r}"
            )
        if self.max is not None and self.default > self.max:
            raise ValueError(
                f"ParamSpec.default {self.default!r} is above max {self.max!r}"
            )

    def _validate_choices(self) -> None:
        """Validate the enumerated ``choices`` dict and copy it defensively.

        Raises:
            TypeError: If ``choices`` is not a dict.
            ValueError: If ``choices`` is empty, if any value is not an instance
                of ``type``, if ``default`` is not one of the values, or if any
                bound is set (bounds and choices are mutually exclusive).
        """
        if not isinstance(self.choices, dict):
            raise TypeError(f"ParamSpec.choices must be a dict, got {self.choices!r}")
        if not self.choices:
            raise ValueError("ParamSpec.choices must be a non-empty dict")
        if self.min is not None or self.max is not None:
            raise ValueError(
                "ParamSpec.choices and min/max are mutually exclusive; set only one"
            )
        for label, val in self.choices.items():
            if not self._matches_type(val):
                raise ValueError(
                    f"ParamSpec.choices[{label!r}] value {val!r} is not a "
                    f"{self.type.__name__}"
                )
        if self.default not in self.choices.values():
            raise ValueError(
                f"ParamSpec.default {self.default!r} is not one of the choice "
                f"values {list(self.choices.values())}"
            )
        object.__setattr__(self, "choices", dict(self.choices))


@dataclass(frozen=True)
class ParamGroup:
    """One rendered sub-panel of a procedure's parameter form.

    The GUI renders one ``QGroupBox`` per group, in list order. ``key`` is the
    stable identity used to cache the group's values across re-renders (e.g.
    "system", "measurement:dc_measurement"), so it must survive form
    re-derivation even when titles change. The ``params`` dict is defensively
    copied.

    Attributes:
        key: Stable identity for value caching. Non-empty string.
        title: Human-readable panel heading. Non-empty string.
        params: Mapping of parameter name to ``ParamSpec``. Defensively copied.
    """

    key: str
    title: str
    params: dict[str, ParamSpec]

    def __post_init__(self) -> None:
        """Validate the fields and defensively copy ``params``.

        Raises:
            TypeError: If ``key``/``title`` is not a string, ``params`` is not a
                dict, a params key is not a string, or a value is not a
                ``ParamSpec``.
            ValueError: If ``key``/``title`` or a params key is an empty string.
        """
        for name, val in (("key", self.key), ("title", self.title)):
            if not isinstance(val, str):
                raise TypeError(f"ParamGroup.{name} must be a str, got {val!r}")
            if not val:
                raise ValueError(f"ParamGroup.{name} must be a non-empty str")

        if not isinstance(self.params, dict):
            raise TypeError(f"ParamGroup.params must be a dict, got {self.params!r}")
        for name, spec in self.params.items():
            if not isinstance(name, str):
                raise TypeError(f"ParamGroup.params key must be a str, got {name!r}")
            if not name:
                raise ValueError("ParamGroup.params key must be a non-empty str")
            if not isinstance(spec, ParamSpec):
                raise TypeError(
                    f"ParamGroup.params[{name!r}] must be a ParamSpec, got {spec!r}"
                )
        object.__setattr__(self, "params", dict(self.params))


@dataclass(frozen=True)
class UIGroup:
    """One titled group of a Virtual Instrument's own capabilities.

    The VI-side counterpart of ``ParamGroup``, and deliberately a separate
    type: ``ParamGroup`` groups the *parameters of one procedure run* and
    carries ``ParamSpec``s, whereas a ``UIGroup`` groups the *methods of one
    instrument* — its ``@monitored`` readings and ``@control`` actions — by
    naming them in ``members``. A VI declares its groups in the
    ``ui_groups`` class attribute (see the UI-group standard in
    ``virtual_instruments/base.py``), and the base class validates every
    group and every ``group=`` tag at class creation.

    Groups are presentation and description only: they order and title what
    the instrument front panel renders and what the capability manifest
    describes. Nothing about a group crosses the action queue — a control is
    still submitted on its own, by method name, with flat scalar kwargs.

    ``members`` is explicit rather than derived from the ``group=`` tags,
    because its order IS the render order and it doubles as documentation of
    the workflow order for an agent reading the manifest. A method may also
    carry the matching ``group=`` tag; the base class checks the two agree.

    Attributes:
        key: Stable identity, the value a method's ``group=`` tag names.
            Non-empty string, unique within one VI.
        title: Human-readable heading. Non-empty string.
        description: Optional sentence saying what the group is for.
        members: Ordered ``@monitored`` / ``@control`` method names, at least
            one, no duplicates. Coerced to a tuple.
    """

    key: str
    title: str
    description: str = ""
    members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the declaration and freeze ``members`` as a tuple.

        Raises:
            TypeError: If ``key``, ``title`` or ``description`` is not a
                string, if ``members`` is not a sequence of strings, or if a
                member name is not a string.
            ValueError: If ``key``, ``title`` or a member name is empty, if
                ``members`` is empty, or if a member name repeats.
        """
        for name, val in (("key", self.key), ("title", self.title)):
            if not isinstance(val, str):
                raise TypeError(f"UIGroup.{name} must be a str, got {val!r}")
            if not val:
                raise ValueError(f"UIGroup.{name} must be a non-empty str")

        if not isinstance(self.description, str):
            raise TypeError(
                f"UIGroup.description must be a str, got {self.description!r}"
            )

        if isinstance(self.members, str) or not isinstance(
            self.members, (tuple, list)
        ):
            raise TypeError(
                f"UIGroup.members must be a tuple of method names, got "
                f"{self.members!r}"
            )
        members = tuple(self.members)
        if not members:
            raise ValueError(
                f"UIGroup {self.key!r} must name at least one member — a group "
                f"with no members declares nothing"
            )
        for member in members:
            if not isinstance(member, str):
                raise TypeError(
                    f"UIGroup {self.key!r} member must be a str, got {member!r}"
                )
            if not member:
                raise ValueError(
                    f"UIGroup {self.key!r} member must be a non-empty str"
                )
        if len(set(members)) != len(members):
            raise ValueError(
                f"UIGroup {self.key!r} lists a member twice: {list(members)}"
            )
        object.__setattr__(self, "members", members)


def _validate_dtype_columns(field_name: str, columns: Any) -> dict[str, str]:
    """Validate a ``{name: "float"|"int"}`` mapping and return a defensive copy."""
    if not isinstance(columns, dict):
        raise TypeError(f"DataSchema.{field_name} must be a dict, got {columns!r}")
    for name, dtype in columns.items():
        if not isinstance(name, str):
            raise TypeError(f"DataSchema.{field_name} key must be a str, got {name!r}")
        if not name:
            raise ValueError(f"DataSchema.{field_name} key must be a non-empty str")
        if dtype not in _ALLOWED_DTYPES:
            raise ValueError(
                f"DataSchema.{field_name}[{name!r}] dtype {dtype!r} must be "
                f"one of {sorted(_ALLOWED_DTYPES)}"
            )
    return dict(columns)


def _validate_array_lengths(field_name: str, arrays: Any) -> dict[str, int]:
    """Validate a ``{name: length}`` mapping and return a defensive copy."""
    if not isinstance(arrays, dict):
        raise TypeError(f"DataSchema.{field_name} must be a dict, got {arrays!r}")
    for name, length in arrays.items():
        if not isinstance(name, str):
            raise TypeError(f"DataSchema.{field_name} key must be a str, got {name!r}")
        if not name:
            raise ValueError(f"DataSchema.{field_name} key must be a non-empty str")
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError(
                f"DataSchema.{field_name}[{name!r}] length must be an int, got {length!r}"
            )
        if length <= 0:
            raise ValueError(
                f"DataSchema.{field_name}[{name!r}] length must be > 0, got {length!r}"
            )
    return dict(arrays)


def _validate_block_shapes(field_name: str, blocks: Any) -> dict[str, tuple[int, int]]:
    """Validate a ``{name: (rows, cols)}`` mapping and return a defensive copy."""
    if not isinstance(blocks, dict):
        raise TypeError(f"DataSchema.{field_name} must be a dict, got {blocks!r}")
    validated: dict[str, tuple[int, int]] = {}
    for name, shape in blocks.items():
        if not isinstance(name, str):
            raise TypeError(f"DataSchema.{field_name} key must be a str, got {name!r}")
        if not name:
            raise ValueError(f"DataSchema.{field_name} key must be a non-empty str")
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) for n in shape)
        ):
            raise TypeError(
                f"DataSchema.{field_name}[{name!r}] must be a (int, int) tuple, "
                f"got {shape!r}"
            )
        if any(n <= 0 for n in shape):
            raise ValueError(
                f"DataSchema.{field_name}[{name!r}] entries must be > 0, got {shape!r}"
            )
        validated[name] = (int(shape[0]), int(shape[1]))
    return validated


def _nested_shape_leaves(value: Any, shape: tuple[int, ...]) -> list[Any] | None:
    """Return the flat leaves of *value* if its nesting matches *shape*, else None.

    Walks list/tuple/ndarray-like nesting (anything with ``__len__`` other than
    a string) one ``shape`` dimension at a time. An empty ``shape`` means
    *value* itself is the (scalar) leaf.
    """
    if not shape:
        return [value]
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        return None
    if len(value) != shape[0]:
        return None
    leaves: list[Any] = []
    for item in value:
        sub = _nested_shape_leaves(item, shape[1:])
        if sub is None:
            return None
        leaves.extend(sub)
    return leaves


@dataclass(frozen=True)
class ImageBlock:
    """One declared image block of a measurement VI — a frame per reading.

    The **image-block standard** (the sibling of the raw diagnostic block on
    ``MeasurementInstrumentBase``): a camera frame has the same
    ``(rows, cols)`` storage shape as a raw block but no channel per column
    — every element is one pixel in one unit — so it is declared with its
    pixel dimensions and unit instead of a channel-label list, stored
    through the same dataset path with ``block_kind = "image"`` and
    ``unit`` attributes and no ``channel_names``, and read back by
    ``data_reader.read_image()``. A frame is never reduced to a scalar: the
    sweep's plottable columns are the VI's scalars, the frames are the
    image stack an analysis recipe reads.

    Attributes:
        height_px: Frame height in pixels (rows); an ``int`` > 0.
        width_px: Frame width in pixels (columns); an ``int`` > 0.
        unit: The unit of every pixel value (``"counts"``, ``"V"``, …); a
            non-empty string, written to the dataset's ``unit`` attribute.
        description: One sentence naming what the frame shows; written to
            the dataset's ``description`` attribute.
    """

    height_px: int
    width_px: int
    unit: str
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the declaration.

        Raises:
            TypeError: If a pixel dimension is not an ``int`` (``bool`` is
                rejected), or ``unit``/``description`` is not a string.
            ValueError: If a pixel dimension is not > 0, or ``unit`` is empty.
        """
        for name in ("height_px", "width_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ImageBlock.{name} must be an int, got {value!r}")
            if value <= 0:
                raise ValueError(f"ImageBlock.{name} must be > 0, got {value!r}")
        if not isinstance(self.unit, str):
            raise TypeError(f"ImageBlock.unit must be a str, got {self.unit!r}")
        if not self.unit:
            raise ValueError("ImageBlock.unit must be a non-empty str")
        if not isinstance(self.description, str):
            raise TypeError(
                f"ImageBlock.description must be a str, got {self.description!r}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the frame's ``(height_px, width_px)`` — its ``(rows, cols)`` block shape."""
        return (self.height_px, self.width_px)


@dataclass(frozen=True)
class DataSchema:
    """The declared HDF5 layout of one measurement run.

    Assembled at ``initiate()`` by composition: the sweep axis contributes its
    sweep column and the station its system columns — both ``sweep_columns``,
    one value per sweep point, never looped — and the selected measurement VI
    contributes its scalar columns (``measurement_scalars``, e.g. a
    quantity's mean/error, or ``n_valid``), raw-sample arrays
    (``measurement_arrays``), and raw diagnostic blocks (``measurement_blocks``).
    Every measurement scalar/array carries a real reading-loop axis: shape
    ``(n_loop1, n_loop2)`` for a scalar, or ``(n_loop1, n_loop2, length)`` for
    an array — ``loop_shape`` declares the axis lengths (``1`` means that
    slot is not looping). A block is the one exception: it carries that axis
    only when a reading loop is actually configured (see
    ``measurement_blocks``'s own docstring below). This is the single owner
    of the run's shape contract, the thing that catches "HDF5 expected a
    different format" mismatches before any data is written. All dict fields
    are defensively copied.

    Attributes:
        sweep_columns: Mapping of scalar column name to dtype string ("float"
            or "int"). One value per sweep point — never looped (e.g.
            ``unix_time``, system state, the sweep axis readback).
        measurement_scalars: Mapping of scalar column name to dtype string.
            One ``(n_loop1, n_loop2)`` grid of values per sweep point.
        measurement_arrays: Mapping of array name to its per-point length (an
            ``int`` > 0; ``bool`` is rejected). One ``(n_loop1, n_loop2,
            length)`` grid of raw samples per sweep point.
        measurement_blocks: Mapping of raw diagnostic block name (see
            ``MeasurementInstrumentBase``'s "Raw diagnostic blocks" standard)
            — or image block name (the image-block standard, ``ImageBlock``;
            its ``(height_px, width_px)`` is its block shape) — to its
            ``(rows, cols)`` shape, both ``> 0``. UNLIKE
            ``measurement_scalars``/``measurement_arrays``, a block carries
            the ``(n_loop1, n_loop2)`` reading-loop axis only when
            ``loop_shape != (1, 1)`` (an active reading loop) — with no
            reading loop configured its per-point grid is bare ``(rows,
            cols)``, not ``(1, 1, rows, cols)``. A block is a diagnostic
            per-reading record, not a quantity meant to be sliced by loop
            axis, so the standard deliberately skips the trivial axis
            instead of always carrying it. Defaults to ``{}`` — no VI
            declares a raw block.
        loop_shape: ``(n_loop1, n_loop2)``, each ``>= 1``. Defaults to
            ``(1, 1)`` — no reading loop.
    """

    sweep_columns: dict[str, str]
    measurement_scalars: dict[str, str]
    measurement_arrays: dict[str, int]
    measurement_blocks: dict[str, tuple[int, int]] = field(default_factory=dict)
    loop_shape: tuple[int, int] = (1, 1)

    def __post_init__(self) -> None:
        """Validate names, dtypes, lengths and ``loop_shape``; copy the dicts.

        Raises:
            TypeError: If a dict field is not a dict, a name is not a string,
                an array length/block shape is not an int / (int, int) tuple,
                or ``loop_shape`` is not a ``(int, int)`` tuple.
            ValueError: If a name is empty, a dtype is not in the allowed set,
                an array length or block shape entry is not strictly
                positive, or a ``loop_shape`` entry is not ``>= 1``.
        """
        sweep_columns = _validate_dtype_columns("sweep_columns", self.sweep_columns)
        measurement_scalars = _validate_dtype_columns(
            "measurement_scalars", self.measurement_scalars
        )
        measurement_arrays = _validate_array_lengths(
            "measurement_arrays", self.measurement_arrays
        )
        measurement_blocks = _validate_block_shapes(
            "measurement_blocks", self.measurement_blocks
        )

        loop_shape = self.loop_shape
        if (
            not isinstance(loop_shape, tuple)
            or len(loop_shape) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) for n in loop_shape)
        ):
            raise TypeError(
                f"DataSchema.loop_shape must be a (int, int) tuple, got {loop_shape!r}"
            )
        if any(n < 1 for n in loop_shape):
            raise ValueError(
                f"DataSchema.loop_shape entries must be >= 1, got {loop_shape!r}"
            )

        object.__setattr__(self, "sweep_columns", sweep_columns)
        object.__setattr__(self, "measurement_scalars", measurement_scalars)
        object.__setattr__(self, "measurement_arrays", measurement_arrays)
        object.__setattr__(self, "measurement_blocks", measurement_blocks)
        object.__setattr__(self, "loop_shape", tuple(loop_shape))

    def validate(self, datapoint: Mapping[str, Any]) -> None:
        """Check one datapoint against this schema, reporting every problem.

        Collects all mismatches rather than stopping at the first, so a caller
        fixing a malformed datapoint sees the complete list in a single
        ``DataSchemaError`` instead of one error per fix-and-rerun cycle.

        Checks performed:
            * every declared key is present (missing keys reported);
            * no undeclared keys are present (extra keys reported);
            * each ``sweep_columns`` value is a real-number scalar (``bool``
              rejected; dtype "int" accepts ``int``, dtype "float" accepts
              ``int`` or ``float``);
            * each ``measurement_scalars`` value is a nested structure shaped
              exactly ``loop_shape``, every leaf a real-number scalar (same
              dtype rule as sweep columns);
            * each ``measurement_arrays`` value is a nested structure shaped
              exactly ``loop_shape + (length,)``.
            * each ``measurement_blocks`` value is a nested structure shaped
              exactly ``loop_shape + (rows, cols)``.

        Args:
            datapoint: Mapping of column/array/block name to value to check.

        Returns:
            None if the datapoint conforms.

        Raises:
            DataSchemaError: If any check fails; the message lists all problems.
        """
        declared = (
            set(self.sweep_columns)
            | set(self.measurement_scalars)
            | set(self.measurement_arrays)
            | set(self.measurement_blocks)
        )
        present = set(datapoint)
        problems: list[str] = []

        for key in sorted(declared - present):
            problems.append(f"missing declared key {key!r}")
        for key in sorted(present - declared):
            problems.append(f"extra undeclared key {key!r}")

        for name, dtype in self.sweep_columns.items():
            if name not in datapoint:
                continue
            value = datapoint[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(
                    f"sweep column {name!r} value {value!r} is not a real-number scalar"
                )
            elif dtype == "int" and not isinstance(value, int):
                problems.append(
                    f"sweep column {name!r} value {value!r} is not an int (dtype 'int')"
                )

        for name, dtype in self.measurement_scalars.items():
            if name not in datapoint:
                continue
            leaves = _nested_shape_leaves(datapoint[name], self.loop_shape)
            if leaves is None:
                problems.append(
                    f"measurement scalar {name!r} does not match loop shape "
                    f"{self.loop_shape}"
                )
                continue
            for value in leaves:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    problems.append(
                        f"measurement scalar {name!r} has a non-real-number "
                        f"value {value!r}"
                    )
                elif dtype == "int" and not isinstance(value, int):
                    problems.append(
                        f"measurement scalar {name!r} has a non-int value "
                        f"{value!r} (dtype 'int')"
                    )

        for name, length in self.measurement_arrays.items():
            if name not in datapoint:
                continue
            shape = (*self.loop_shape, length)
            leaves = _nested_shape_leaves(datapoint[name], shape)
            if leaves is None:
                problems.append(
                    f"measurement array {name!r} does not match shape {shape}"
                )

        for name, block_shape in self.measurement_blocks.items():
            if name not in datapoint:
                continue
            # No loop-axis prefix when no reading loop is configured — see
            # measurement_blocks' docstring above.
            block_loop_prefix = self.loop_shape if self.loop_shape != (1, 1) else ()
            shape = (*block_loop_prefix, *block_shape)
            leaves = _nested_shape_leaves(datapoint[name], shape)
            if leaves is None:
                problems.append(
                    f"measurement block {name!r} does not match shape {shape}"
                )

        if problems:
            raise DataSchemaError(
                "datapoint does not match schema: " + "; ".join(problems)
            )


#: The setpoint-parameter convention: a ``@control`` parameter whose name
#: starts with this prefix carries its VI's enveloped quantity — the same
#: physical quantity ``RampableVI.start_ramp(target)`` takes and a ``Target``
#: commands (tesla for a magnet, kelvin for a temperature controller).
#: Naming it this way is what lets the session envelope bind a
#: manual action as well as a plan's ``Target``, with no per-VI table for the
#: Orchestrator to keep in step: it asks the Station which of an action's
#: keyword arguments is the setpoint and checks that one. A VI declares at
#: most one such parameter, on the single capability that commands its
#: setpoint; machine-checked for every rampable VI by
#: ``tests/test_conformance.py``.
SETPOINT_PARAM_PREFIX: str = "target_"


@dataclass(frozen=True)
class EnvelopeVariable:
    """One VI's enveloped quantity: the capability that sets it and its bounds.

    The read side of the setpoint-parameter convention
    (``SETPOINT_PARAM_PREFIX``), assembled by
    ``Station.envelope_variables()``. It answers the two questions the
    envelope editor and the Orchestrator's manual-action check both ask about
    a VI: *which* keyword argument carries the enveloped value, and *what
    range does the setup itself already allow* — the ``control_limits`` bound
    the config populated, which an experiment's ``EnvelopeBound`` narrows.

    Attributes:
        vi_name: The VI this variable belongs to.
        method_name: The ``@control`` capability that commands it.
        param_name: That capability's setpoint parameter (``target_*``).
        config_min: Setup lower bound in the quantity's SI unit, or ``None``
            when the setup leaves it unbounded below.
        config_max: Setup upper bound, or ``None`` when unbounded above.
    """

    vi_name: str
    method_name: str
    param_name: str
    config_min: float | None = None
    config_max: float | None = None

    @property
    def unit_suffix(self) -> str:
        """Return the parameter's trailing unit token, or ``""``.

        Derived from the setpoint parameter's own name — ``target_T`` → ``T``,
        ``target_K`` → ``K``, ``target_deg`` → ``deg`` — which the SI-units
        rule already makes the authoritative unit marker on a VI's API. Used
        only to label the envelope editor's fields.
        """
        return self.param_name[len(SETPOINT_PARAM_PREFIX):]


@dataclass(frozen=True)
class EnvelopeBound:
    """One session-envelope limit on a system VI's swept quantity.

    Config ``init_params`` limits protect the *instrument* and never change at
    runtime; an ``EnvelopeBound`` protects the *sample* mounted for one
    experiment (e.g. "this device must never see more than 2 T even though the
    magnet allows 9 T"). Bounds are expressed in the same SI unit as the VI's
    ``Target.target`` (Tesla, Kelvin, …).

    Attributes:
        min_value: Lowest allowed value, or ``None`` for no lower bound.
        max_value: Highest allowed value, or ``None`` for no upper bound.
        state_key: Optional key into the VI's ``get_state()`` dict (e.g.
            ``"field_T"``, ``"temperature_K"``) naming the live reading this
            bound also applies to. When set, the Orchestrator checks the
            reading every tick in addition to validating submitted targets;
            when empty, only targets are checked.
    """

    min_value: float | None = None
    max_value: float | None = None
    state_key: str = ""

    def __post_init__(self) -> None:
        """Validate the fields.

        Raises:
            TypeError: If a bound is not a real number or ``state_key`` is not
                a string.
            ValueError: If both bounds are ``None``, a bound is non-finite, or
                ``min_value`` exceeds ``max_value``.
        """
        for attr in ("min_value", "max_value"):
            value = getattr(self, attr)
            if value is None:
                continue
            if not _is_real_number(value):
                raise TypeError(
                    f"EnvelopeBound.{attr} must be a real number, got {value!r}"
                )
            if not math.isfinite(value):
                raise ValueError(f"EnvelopeBound.{attr} must be finite, got {value!r}")
            object.__setattr__(self, attr, float(value))
        if self.min_value is None and self.max_value is None:
            raise ValueError("EnvelopeBound needs at least one of min_value/max_value")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError(
                f"EnvelopeBound.min_value {self.min_value!r} exceeds "
                f"max_value {self.max_value!r}"
            )
        if not isinstance(self.state_key, str):
            raise TypeError(
                f"EnvelopeBound.state_key must be a str, got {self.state_key!r}"
            )

    def violation(self, value: float) -> str | None:
        """Return a human-readable violation message for ``value``, or ``None``.

        Args:
            value: The candidate value (a submitted target or a live reading),
                in the VI's SI unit.

        Returns:
            A message naming the violated bound, or ``None`` when ``value`` is
            within the bound (non-numeric values are ignored — a text or bool
            state field can never trip a numeric envelope).
        """
        if not _is_real_number(value):
            return None
        if self.min_value is not None and value < self.min_value:
            return f"{value:g} is below the session minimum {self.min_value:g}"
        if self.max_value is not None and value > self.max_value:
            return f"{value:g} is above the session maximum {self.max_value:g}"
        return None


@dataclass(frozen=True)
class ExperimentEnvelope:
    """Per-experiment safety bounds, narrower than the config limits.

    The typed currency between the session layer (which owns the experiment
    record the envelope belongs to) and the Orchestrator (which enforces it).
    Enforcement lives in the Orchestrator so the envelope binds *every* writer
    — a human slip in the GUI is caught by the same check as an agent call:

    * every submitted ``Target`` for a bounded VI is validated before dispatch;
    * every manual action on the **direct action path** is validated the same
      way, on the setpoint parameter the setpoint-parameter convention
      identifies (``SETPOINT_PARAM_PREFIX``), so a bounded VI's setpoint is
      refused outside the envelope whether it arrives as a ``Target`` or as an
      action — at submission AND again when the tick drains the queue;
    * every tick, each bound with a ``state_key`` is checked against the VI's
      live reading, entering EMERGENCY on a violation exactly like a tripped
      safety flag.

    Attributes:
        bounds: Mapping of system VI name to its ``EnvelopeBound``.
            Defensively copied; must be non-empty (pass ``None`` to
            ``Orchestrator.set_experiment_envelope()`` for "no envelope" rather
            than an empty one).
    """

    bounds: Mapping[str, EnvelopeBound]

    def __post_init__(self) -> None:
        """Validate and defensively copy ``bounds``.

        Raises:
            TypeError: If ``bounds`` is not a mapping of str to
                ``EnvelopeBound``.
            ValueError: If ``bounds`` is empty or a VI name is empty.
        """
        if not isinstance(self.bounds, Mapping):
            raise TypeError(
                f"ExperimentEnvelope.bounds must be a mapping, got {self.bounds!r}"
            )
        if not self.bounds:
            raise ValueError(
                "ExperimentEnvelope.bounds must be non-empty (use None for no envelope)"
            )
        copied: dict[str, EnvelopeBound] = {}
        for vi_name, bound in self.bounds.items():
            if not isinstance(vi_name, str) or not vi_name:
                raise ValueError(
                    f"ExperimentEnvelope VI name must be a non-empty str, got {vi_name!r}"
                )
            if not isinstance(bound, EnvelopeBound):
                raise TypeError(
                    f"ExperimentEnvelope bound for {vi_name!r} must be an "
                    f"EnvelopeBound, got {bound!r}"
                )
            copied[vi_name] = bound
        object.__setattr__(self, "bounds", copied)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentEnvelope:
        """Build an envelope from its plain-dict form.

        The dict form is what a client that speaks JSON rather than Python
        sends — ``{vi_name: {"min_value": ..., "max_value": ...,
        "state_key": ...}}``, one entry per bounded VI, each key optional
        beyond the requirement that at least one bound is present. Strict by
        design: a malformed envelope raises here so the caller can refuse the
        request outright. (The session layer keeps its own *tolerant* reader
        for records loaded from disk, where a corrupt stored envelope must
        degrade to "no envelope" rather than block loading.)

        Args:
            data: The mapping described above. Must be non-empty — pass
                ``None`` to ``Orchestrator.set_experiment_envelope()`` for
                "no envelope" rather than an empty mapping.

        Returns:
            The typed envelope.

        Raises:
            TypeError: If *data* is not a mapping, or an entry is not a
                mapping of the ``EnvelopeBound`` fields.
            ValueError: If *data* is empty, or a bound is invalid (see
                ``EnvelopeBound``).
        """
        if not isinstance(data, Mapping):
            raise TypeError(
                f"ExperimentEnvelope.from_dict expects a mapping, got {data!r}"
            )
        bounds: dict[str, EnvelopeBound] = {}
        for vi_name, entry in data.items():
            if not isinstance(entry, Mapping):
                raise TypeError(
                    f"envelope bound for {vi_name!r} must be a mapping, got {entry!r}"
                )
            bounds[str(vi_name)] = EnvelopeBound(
                min_value=entry.get("min_value"),
                max_value=entry.get("max_value"),
                state_key=str(entry.get("state_key") or ""),
            )
        return cls(bounds=bounds)

    def check_target(self, vi_name: str, value: float) -> str | None:
        """Validate one submitted target value against the envelope.

        Args:
            vi_name: The system VI the target is for.
            value: The requested ``Target.target`` value (SI unit).

        Returns:
            A violation message naming the VI, or ``None`` when the VI is
            unbounded or the value is within its bound.
        """
        bound = self.bounds.get(vi_name)
        if bound is None:
            return None
        message = bound.violation(value)
        if message is None:
            return None
        return f"session envelope: {vi_name} target {message}"

    def check_state(self, state: Mapping[str, Mapping[str, Any]]) -> list[str]:
        """Check every ``state_key``-carrying bound against a station snapshot.

        Args:
            state: A ``Station.get_state()`` snapshot
                (``{vi_name: {field: value}}``).

        Returns:
            One violation message per tripped bound (empty when all live
            readings are inside the envelope). A bound whose VI or state key
            is absent from the snapshot is skipped — a missing reading is a
            staleness problem, not an envelope violation.
        """
        violations: list[str] = []
        for vi_name, bound in self.bounds.items():
            if not bound.state_key:
                continue
            vi_state = state.get(vi_name)
            if not isinstance(vi_state, Mapping) or bound.state_key not in vi_state:
                continue
            message = bound.violation(vi_state[bound.state_key])
            if message is not None:
                violations.append(
                    f"session envelope: {vi_name} {bound.state_key} {message}"
                )
        return violations


@dataclass(frozen=True)
class ProbeSpec:
    """How a cheap **probe run** is derived from a requested run.

    A probe is the same procedure class driving the same instruments through
    the same code path, reduced until it costs minutes instead of hours: it
    answers "would this run actually work, and does the signal look sane?"
    before the full run is committed to. It is never science data — the run it
    produces declares ``run_kind = "probe"``, which travels into the run
    manifest, the session layer's run record, and the data file's
    ``/metadata.run_kind``, so a probe file can never be mistaken for the real
    thing.

    **The reduction rules** (the standard; applied by
    ``BaseProcedure.apply_probe()`` to an already-built run, so a probe is
    only ever a reduction of a run that was built and validated as requested):

    1. **Sweep length.** The built sweep array is subsampled to at most
       ``n_points`` points, evenly spaced and ALWAYS keeping the first and the
       last — a probe must exercise the extremes the full run would ramp to,
       because that is where a limit, a quench or a lost lock shows up.
       ``n_points=3`` therefore means first/middle/last. A sweep already at or
       below ``n_points`` is left alone.
    2. **Waits.** Every declared parameter whose ``ParamSpec`` carries the
       unit ``"s"`` — settle waits, inter-reading delays — is capped at
       ``max_wait_s``. Values already below the cap are left alone; a probe
       never raises a wait.
    3. **Averaging.** Every declared repeat count is capped at ``averaging``.
       A repeat count is identified by declaration, never by name: it is a
       measurement parameter whose value sets the length of one of the
       measurement VI's declared data arrays (``data_arrays()``), so a new
       measurement VI's averaging knob is reduced with no new code.
    4. **Kind and provenance.** The reduced run declares ``run_kind =
       "probe"`` and records this spec as its ``probe_spec`` parameter, so the
       file says which reduction produced it. The sweep-shape parameters keep
       the values that were requested (``field_steps`` still reads 101); the
       saved point count and ``probe_spec`` together are what say what ran.

    Nothing else changes: same procedure class, same targets, same measurement
    VI, same claim, and the same setup limits and session envelope apply.

    Attributes:
        n_points: Maximum number of sweep points the probe keeps. Must be
            ``>= 1``; the default 3 gives first/middle/last.
        averaging: Maximum repeat count per measurement point. Must be
            ``>= 1``.
        max_wait_s: Upper bound, in seconds, on every declared wait. Must be
            ``>= 0``.
    """

    n_points: int = 3
    averaging: int = 1
    max_wait_s: float = 5.0

    def __post_init__(self) -> None:
        """Validate the three reduction bounds.

        Raises:
            TypeError: If a field is not a real number (``bool`` rejected).
            ValueError: If ``n_points``/``averaging`` is below 1, or
                ``max_wait_s`` is negative or non-finite.
        """
        for name in ("n_points", "averaging"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ProbeSpec.{name} must be an int, got {value!r}")
            if value < 1:
                raise ValueError(f"ProbeSpec.{name} must be >= 1, got {value!r}")
        if not _is_real_number(self.max_wait_s):
            raise TypeError(
                f"ProbeSpec.max_wait_s must be a real number, got {self.max_wait_s!r}"
            )
        if not math.isfinite(self.max_wait_s) or self.max_wait_s < 0:
            raise ValueError(
                f"ProbeSpec.max_wait_s must be finite and >= 0, "
                f"got {self.max_wait_s!r}"
            )
        object.__setattr__(self, "max_wait_s", float(self.max_wait_s))

    def to_json(self) -> dict[str, Any]:
        """Render this spec as a JSON-safe dict.

        Returns:
            ``{"n_points": int, "averaging": int, "max_wait_s": float}``.
        """
        return {
            "n_points": self.n_points,
            "averaging": self.averaging,
            "max_wait_s": self.max_wait_s,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ProbeSpec:
        """Rebuild a spec from its dict form.

        The form a JSON-speaking client sends: every key optional, unknown
        keys ignored so a newer producer never breaks an older consumer.

        Args:
            data: A mapping as produced by ``to_json()``.

        Returns:
            The typed spec.

        Raises:
            TypeError: If *data* is not a mapping, or a value has the wrong
                type.
            ValueError: If a value is out of range (see ``__post_init__``).
        """
        if not isinstance(data, Mapping):
            raise TypeError(f"ProbeSpec.from_json expects a mapping, got {data!r}")
        declared = {"n_points", "averaging", "max_wait_s"}
        return cls(**{k: v for k, v in data.items() if k in declared})


@dataclass(frozen=True)
class StepCost:
    """What one run costs per point, apart from the ramps.

    The currency of the **duration-estimate standard**: the one thing a run
    contributes to its own estimate, returned by
    ``BaseProcedure.estimate_step_seconds()``. The estimator
    (``core/estimates.py``) supplies the other half — the ramp time, which it
    derives from the run's declared targets and the setup's nominal ramp
    rates — so a procedure never has to know how fast its magnet moves.

    Every number is seconds and every unknown is an assumption string rather
    than a silent zero.

    Attributes:
        points: How many measurement points the run will take.
        setup_s: One-off settle time before the first point (the initiate
            wait).
        settle_s: Settle time paid before every point after the first.
        measure_s: Time one measurement at one point takes, averaging
            included.
        assumptions: Human-readable statements about what this cost model
            could not derive and assumed instead.
    """

    points: int = 0
    setup_s: float = 0.0
    settle_s: float = 0.0
    measure_s: float = 0.0
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the counts and freeze ``assumptions``.

        Raises:
            TypeError: If a numeric field is not a real number, or an
                assumption is not a string.
            ValueError: If any value is negative or non-finite.
        """
        if isinstance(self.points, bool) or not isinstance(self.points, int):
            raise TypeError(f"StepCost.points must be an int, got {self.points!r}")
        if self.points < 0:
            raise ValueError(f"StepCost.points must be >= 0, got {self.points!r}")
        for name in ("setup_s", "settle_s", "measure_s"):
            value = getattr(self, name)
            if not _is_real_number(value):
                raise TypeError(f"StepCost.{name} must be a real number, got {value!r}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"StepCost.{name} must be finite and >= 0, got {value!r}"
                )
            object.__setattr__(self, name, float(value))
        assumptions = tuple(self.assumptions)
        for assumption in assumptions:
            if not isinstance(assumption, str):
                raise TypeError(
                    f"StepCost.assumptions must hold strings, got {assumption!r}"
                )
        object.__setattr__(self, "assumptions", assumptions)


@dataclass(frozen=True)
class DurationEstimate:
    """How long a run is expected to take, and what that answer assumed.

    Frozen and JSON-safe, so the same value serves ``RunValidation``, the
    verdict a client renders, and a stored session record. Deliberately
    carries its own assumptions: an estimate a client cannot qualify is worse
    than no estimate at all, so anything the model could not derive — a VI
    with no declared ramp rate, a measurement whose duration nothing declares
    — is named here instead of being silently counted as zero.

    Attributes:
        total_s: Expected wall-clock duration, in seconds. The sum of
            ``phases``.
        phases: Seconds per phase, e.g. ``{"setup": .., "ramp": ..,
            "settle": .., "measure": ..}``. Phase keys are whatever the
            estimator produced; a client renders them, never switches on them.
        assumptions: One statement per thing the estimate assumed, in
            discovery order. Never empty for a real estimate: at minimum it
            names the rate model used.
    """

    total_s: float = 0.0
    phases: dict[str, float] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze copies of the phase map and assumptions.

        Raises:
            TypeError: If ``total_s`` or a phase value is not a real number,
                ``phases`` is not a mapping, or an assumption is not a string.
            ValueError: If a duration is negative or non-finite.
        """
        if not _is_real_number(self.total_s):
            raise TypeError(
                f"DurationEstimate.total_s must be a real number, got {self.total_s!r}"
            )
        if not math.isfinite(self.total_s) or self.total_s < 0:
            raise ValueError(
                f"DurationEstimate.total_s must be finite and >= 0, "
                f"got {self.total_s!r}"
            )
        object.__setattr__(self, "total_s", float(self.total_s))
        if not isinstance(self.phases, Mapping):
            raise TypeError("DurationEstimate.phases must be a mapping")
        phases: dict[str, float] = {}
        for name, value in self.phases.items():
            if not _is_real_number(value):
                raise TypeError(
                    f"DurationEstimate.phases[{name!r}] must be a real number, "
                    f"got {value!r}"
                )
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"DurationEstimate.phases[{name!r}] must be finite and >= 0, "
                    f"got {value!r}"
                )
            phases[str(name)] = float(value)
        object.__setattr__(self, "phases", phases)
        assumptions = tuple(self.assumptions)
        for assumption in assumptions:
            if not isinstance(assumption, str):
                raise TypeError(
                    f"DurationEstimate.assumptions must hold strings, "
                    f"got {assumption!r}"
                )
        object.__setattr__(self, "assumptions", assumptions)

    def to_json(self) -> dict[str, Any]:
        """Render this estimate as a JSON-safe dict.

        Returns:
            ``{"total_s": float, "phases": {str: float},
            "assumptions": [str, ...]}``.
        """
        return {
            "total_s": self.total_s,
            "phases": dict(self.phases),
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> DurationEstimate:
        """Rebuild an estimate from its ``to_json()`` dict.

        Args:
            data: A mapping as produced by ``to_json()``; unknown keys are
                ignored.

        Returns:
            The typed estimate.

        Raises:
            TypeError: If *data* is not a mapping or a field has the wrong
                type.
            ValueError: If a duration is negative or non-finite.
        """
        if not isinstance(data, Mapping):
            raise TypeError(
                f"DurationEstimate.from_json expects a mapping, got {data!r}"
            )
        return cls(
            total_s=data.get("total_s", 0.0),
            phases=dict(data.get("phases") or {}),
            assumptions=tuple(data.get("assumptions") or ()),
        )
