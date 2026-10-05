"""Decorators for Virtual Instrument methods.

Usage:
    class MyVI(BaseVirtualInstrument):
        @monitored
        def temperature(self) -> float:
            return self._driver.get_temperature()

        @control
        def set_temperature(self, target_K: float):
            self._driver.set_setpoint(target_K)

        @control(scope="operation", action_class="recovery")
        def switch_heater_on(self):
            self._driver.set_switch_heater(True)

The @monitored decorator marks a method that:
- Returns a value to be polled every monitor tick.
- Is displayed as a live-updating number on the GUI panel.
- Is called by get_state() to build the VI state dict.
- Carries its own declaration: ``unit=`` (the SI unit label of the value it
  returns, ``""`` only for a genuinely dimensionless, boolean or string
  reading) and ``description=`` (one human-readable sentence). Both are
  plain strings stored on the function; the declaration standard (see
  ``virtual_instruments/README.md``) requires every monitored field to
  declare them, and ``tests/test_conformance.py``'s
  ``test_capability_manifest_is_complete`` enforces it.
- Carries a **value kind** (``kind=``, the monitored-kind standard): what
  shape of value it returns, which decides how it travels and how it is
  plotted. One of ``MONITORED_KINDS``:

  * ``"scalar"`` (the default) — one number, bool or string, polled every
    monitor tick into the state snapshot, trended and persisted exactly as
    every monitored field always has been.
  * ``"image"`` — a 2-D frame of declared ``shape=(height_px, width_px)``
    (a live camera preview).
  * ``"trace"`` — a 1-D array of declared ``shape=(length,)`` (a spectrum, a
    scope trace, a line profile), optionally on a physical x axis
    ``axis=(start, stop, unit)``.

  A non-scalar ("array") field never enters the scalar state snapshot, the
  trend history, the trend checks or the ``Readings`` event: it is polled on
  its own slower ``period_s=`` (default ``DEFAULT_ARRAY_PERIOD_S``) by
  ``Station.poll_monitored_arrays()`` and published on its own signal, so a
  slow frame read can never delay a scalar safety reading and no scalar
  consumer ever meets an array it does not expect. The kind is a
  declaration on the VI, like the unit — the plot panels read it to offer
  the field to the right renderer (trend, image, waterfall).

The @control decorator marks a method that:
- Appears as a button (with text-box inputs for arguments) on the GUI panel.
- Is callable by the user only when no procedure is running.
- Arguments are inferred from the function signature for GUI form generation.
- Carries a capability scope: ``"measurement"`` (the default, usable by any
  plan) or ``"operation"`` (usable only by an operation's plan — see the
  capability-scope standard in GLOSSARY.md). GUI behavior is unchanged either
  way; a human in IDLE can still click any @control as before. The scope is
  enforced only at plan-dispatch time, by
  ``Station.send_measurement_commands()``.
- Carries an **action class** (``action_class=``): how much authority the
  action needs, independent of who is asking — one of
  ``VALID_ACTION_CLASSES``. This is the action-class declaration: how
  dangerous an instrument's action is is a judgement about that instrument,
  so it is declared here, on the VI, and not in a table somewhere above it.
  It reaches the agent gateway's permission matrix through
  ``StationInfo``'s ``ControlInfo.action_class``. Every shipped ``@control``
  declares it explicitly (enforced by ``tests/test_conformance.py``); the
  value defaults to ``DEFAULT_ACTION_CLASS`` — the most restrictive class an
  agent can ever hold — for a control that does not.

Both decorators accept ``group=``, the UI-group tag: the key of one of the
``UIGroup``s the VI declares in its ``ui_groups`` class attribute (see the
UI-group standard in ``virtual_instruments/base.py``). It is stored here as
an opaque plain string — this module imports no spec type (layer contract
C1) — and the VI base class resolves it against the declared groups at
class creation, so a tag naming no declared group fails at import.
"""

from __future__ import annotations

import functools
import inspect
import typing
from typing import Any, Callable

# The only valid @control capability scopes. Anything else raises
# ValueError at decoration time — a typo in scope="opration" fails loudly at
# import time, not silently at dispatch.
VALID_CONTROL_SCOPES: frozenset[str] = frozenset({"measurement", "operation"})

# The only valid @control action classes, in ascending authority. Declared
# here, as plain strings, because this module imports nothing (layer
# contract C1); the agent gateway's ``ActionClass`` enum carries the same
# four values and ``tests/test_conformance.py`` asserts the two agree, so
# neither can drift from the other.
VALID_ACTION_CLASSES: tuple[str, ...] = ("read", "recovery", "run_control", "envelope")

# The class a @control that declares none is treated as: the most
# restrictive an agent can ever hold, so forgetting the declaration never
# widens what an agent may do. Conformance still requires every shipped
# control to declare it explicitly.
DEFAULT_ACTION_CLASS: str = "run_control"

# The monitored-kind standard (see the module docstring): every value kind a
# @monitored field may declare, and the number of dimensions its declared
# ``shape`` must have (``None``: a scalar declares no shape). Plain strings
# for the same reason as the action classes above — this module imports
# nothing (layer contract C1).
MONITORED_KINDS: dict[str, int | None] = {"scalar": None, "image": 2, "trace": 1}

# The kinds polled outside the scalar monitor tick, on their own period.
ARRAY_KINDS: frozenset[str] = frozenset(
    kind for kind, ndim in MONITORED_KINDS.items() if ndim is not None
)

# How often an array field is polled when it declares no ``period_s=``. Slow
# on purpose: an array read (a camera exposure, a spectrum) costs instrument
# time on the one hardware thread, and a live preview needs no more.
DEFAULT_ARRAY_PERIOD_S: float = 1.0

# The largest array a @monitored field may declare, in elements (a
# 2048x2048 frame). Array values cross to the GUI and are held in memory in
# the same process as the instrument thread, so their size is a declared,
# bounded quantity — a larger image belongs in a measurement's image block,
# written to disk, not in a live preview.
MAX_ARRAY_ELEMENTS: int = 2048 * 2048


def monitored(
    func: Callable | None = None,
    *,
    unit: str | None = None,
    description: str = "",
    group: str | None = None,
    kind: str = "scalar",
    shape: tuple[int, ...] | None = None,
    period_s: float | None = None,
    axis: tuple[float, float, str] | None = None,
) -> Callable:
    """Mark a method as a monitored variable.

    Works both bare (``@monitored``) and parametrized
    (``@monitored(unit="K", description="Sample-stage temperature")``).

    The method will be:
    1. Called every monitor tick by get_state().
    2. Displayed on the GUI panel as a live value.
    3. Wrapped with logging by __init_subclass__.

    The method must take no arguments (besides self) and return a value.

    Renaming this method changes its channel key (``func.__name__``, used
    as the dict key wherever this value is monitored, logged, or
    persisted) — trend-history logs and saved GUI layouts referencing the
    old key will need a migration path to keep reading historical data.

    Args:
        func: The method being decorated (bare-decorator form only; ``None``
            when called parametrized).
        unit: SI unit label of the returned value ("K", "T", "A", "%"), or
            ``""`` for a genuinely dimensionless, boolean or string reading.
            ``None`` (the default) means UNDECLARED, which the declaration
            standard forbids on a shipped VI — it is not the same as ``""``.
        description: One human-readable sentence saying what the value is.
        group: Optional UI-group key (see the module docstring).
        kind: The value kind, one of ``MONITORED_KINDS`` (the
            monitored-kind standard, see the module docstring).
            ``"scalar"`` (the default) is every monitored field as it always
            was; ``"image"`` and ``"trace"`` are array kinds.
        shape: The array's declared shape — ``(height_px, width_px)`` for an
            image, ``(length,)`` for a trace. Required for an array kind,
            forbidden for a scalar. Every polled value is checked against it.
        period_s: Seconds between two polls of an array field (default
            ``DEFAULT_ARRAY_PERIOD_S``). Forbidden for a scalar, which is
            polled every monitor tick.
        axis: A trace's physical x axis as ``(start, stop, unit)`` — the
            value of its first and last sample, e.g. ``(400.0, 800.0,
            "nm")``. Optional; without it a trace is plotted against its
            sample index. Forbidden for any other kind.

    Returns:
        The wrapped method (bare form) or a decorator (parametrized form).

    Raises:
        TypeError: If ``unit``, ``description`` or ``group`` is not a string,
            or an array declaration has the wrong type.
        ValueError: If ``group`` is an empty string, ``kind`` is unknown, or
            the shape/period/axis declaration does not fit the kind.
    """
    _check_declaration_strings(unit=unit, description=description, group=group)
    shape, period_s, axis = _check_kind_declaration(kind, shape, period_s, axis)

    def _decorate(inner_func: Callable) -> Callable:
        @functools.wraps(inner_func)
        def wrapper(*args, **kwargs):
            return inner_func(*args, **kwargs)

        wrapper._is_monitored = True
        wrapper._display_name = inner_func.__name__
        wrapper._monitored_unit = unit
        wrapper._monitored_description = description
        wrapper._ui_group = group or ""
        wrapper._monitored_kind = kind
        wrapper._monitored_shape = shape
        wrapper._monitored_period_s = period_s
        wrapper._monitored_axis = axis
        return wrapper

    if func is not None:
        # Bare form: @monitored
        return _decorate(func)
    # Parametrized form: @monitored(unit=..., description=...)
    return _decorate


def _check_kind_declaration(
    kind: str,
    shape: tuple[int, ...] | None,
    period_s: float | None,
    axis: tuple[float, float, str] | None,
) -> tuple[tuple[int, ...] | None, float | None, tuple[float, float, str] | None]:
    """Validate and normalise a @monitored field's kind declaration.

    A wrong declaration fails at import, never at the first poll.

    Args:
        kind: The declared value kind.
        shape: The declared array shape, or ``None``.
        period_s: The declared poll period, or ``None``.
        axis: The declared trace x axis, or ``None``.

    Returns:
        ``(shape, period_s, axis)`` normalised: the shape as a tuple of
        ints, the period defaulted for an array kind, the axis as a
        ``(float, float, str)`` tuple. All three ``None`` for a scalar.

    Raises:
        TypeError: If a declaration has the wrong type.
        ValueError: If ``kind`` is unknown or a declaration does not fit it.
    """
    if kind not in MONITORED_KINDS:
        raise ValueError(
            f"@monitored kind= must be one of {sorted(MONITORED_KINDS)}, got {kind!r}"
        )
    ndim = MONITORED_KINDS[kind]
    if ndim is None:
        for name, value in (("shape", shape), ("period_s", period_s), ("axis", axis)):
            if value is not None:
                raise ValueError(
                    f"@monitored {name}= declares an array; a kind={kind!r} "
                    f"field is polled every tick and takes none"
                )
        return None, None, None

    if shape is None:
        raise ValueError(f"@monitored kind={kind!r} must declare shape=")
    if not isinstance(shape, (tuple, list)):
        raise TypeError(f"@monitored shape= must be a tuple of ints, got {shape!r}")
    if len(shape) != ndim:
        raise ValueError(
            f"@monitored kind={kind!r} needs a {ndim}-D shape=, got {tuple(shape)!r}"
        )
    for size in shape:
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError(f"@monitored shape= entries must be ints, got {size!r}")
        if size <= 0:
            raise ValueError(f"@monitored shape= entries must be > 0, got {size!r}")
    elements = 1
    for size in shape:
        elements *= size
    if elements > MAX_ARRAY_ELEMENTS:
        raise ValueError(
            f"@monitored shape={tuple(shape)!r} has {elements} elements, over the "
            f"{MAX_ARRAY_ELEMENTS} a live field may declare"
        )

    if period_s is None:
        period_s = DEFAULT_ARRAY_PERIOD_S
    if not isinstance(period_s, (int, float)) or isinstance(period_s, bool):
        raise TypeError(f"@monitored period_s= must be a number, got {period_s!r}")
    if period_s <= 0:
        raise ValueError(f"@monitored period_s= must be > 0, got {period_s!r}")

    if axis is not None:
        if kind != "trace":
            raise ValueError(f"@monitored axis= applies to a trace, not kind={kind!r}")
        if not isinstance(axis, (tuple, list)) or len(axis) != 3:
            raise TypeError(
                f"@monitored axis= must be (start, stop, unit), got {axis!r}"
            )
        start, stop, axis_unit = axis
        for bound in (start, stop):
            if not isinstance(bound, (int, float)) or isinstance(bound, bool):
                raise TypeError(f"@monitored axis= bounds must be numbers, got {bound!r}")
        if not isinstance(axis_unit, str):
            raise TypeError(f"@monitored axis= unit must be a str, got {axis_unit!r}")
        axis = (float(start), float(stop), axis_unit)

    return tuple(shape), float(period_s), axis


def _check_declaration_strings(**values: str | None) -> None:
    """Validate the plain-string declaration keywords shared by both decorators.

    Args:
        **values: Keyword name -> declared value. ``None`` is accepted for
            every one (it means "not declared"); anything that is not a
            string otherwise is a typo caught at import.

    Raises:
        TypeError: If a value is neither ``None`` nor a string.
        ValueError: If ``group`` is declared as an empty string (a group tag
            names a declared ``UIGroup``, so it can never be blank).
    """
    for name, value in values.items():
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(
                f"@monitored/@control {name}= must be a str, got {value!r}"
            )
        if name == "group" and not value:
            raise ValueError(
                "@monitored/@control group= must be a non-empty str naming a "
                "declared UIGroup key"
            )


def control(
    func: Callable | None = None,
    *,
    scope: str = "measurement",
    action_class: str | None = None,
    params: dict[str, Any] | None = None,
    panel: bool = True,
    group: str | None = None,
) -> Callable:
    """Mark a method as a user-controllable action.

    Works both bare (``@control``, scope defaults to ``"measurement"``) and
    parametrized (``@control(scope="operation")``).

    The method will:
    1. Appear as a widget row on the GUI panel (widget shape derived from the
       declared ``params`` ParamSpecs; plain text boxes when none are given).
    2. Be blocked when a procedure is running.
    3. Be wrapped with logging by __init_subclass__.
    4. Carry a capability scope enforced at plan-dispatch time (see the module
       docstring and GLOSSARY.md's "Capability scope" entry).
    5. Carry an action class — the action-class declaration (see the module
       docstring and GLOSSARY.md's "Action class" entry), read by the agent
       gateway off ``ControlInfo.action_class``.

    Args:
        func: The method being decorated (bare-decorator form only; ``None``
            when called parametrized, e.g. ``@control(scope=...)``).
        scope: ``"measurement"`` (default) or ``"operation"``.
        action_class: One of ``VALID_ACTION_CLASSES`` — how much authority
            this action needs. ``None`` (the default) means undeclared,
            which ``get_control_action_class()`` reports as
            ``DEFAULT_ACTION_CLASS``; every shipped control declares it.
        params: Optional ``{param_name: ParamSpec}`` describing each signature
            parameter (unit, min/max, choices, description) for GUI
            rendering. Keys must exactly match the method's parameters
            (checked here at import time). Stored opaquely — this module
            never imports ParamSpec (layer contract C1); the VI base class
            type-checks the values at class-creation time.
        panel: Default placement — ``True`` shows the control on the compact
            monitor card, ``False`` keeps it in the instrument's front panel
            only. A setup's ``monitor.yaml`` ``panels:`` block overrides this
            per VI; the flag is display-only and never a safety mechanism.
        group: Optional UI-group key (see the module docstring). Stored
            opaquely and resolved by the VI base class at class creation.

    Returns:
        The wrapped method (bare form) or a decorator (parametrized form).

    Raises:
        TypeError: If ``group`` is not a string.
        ValueError: If ``scope`` is not one of ``VALID_CONTROL_SCOPES``, if
            ``action_class`` is not one of ``VALID_ACTION_CLASSES``, if
            ``group`` is an empty string, or if ``params`` keys do not
            exactly match the method's parameters.
    """
    _check_declaration_strings(group=group)
    if scope not in VALID_CONTROL_SCOPES:
        raise ValueError(
            f"@control scope must be one of {sorted(VALID_CONTROL_SCOPES)}, "
            f"got {scope!r}"
        )
    if action_class is not None and action_class not in VALID_ACTION_CLASSES:
        raise ValueError(
            f"@control action_class must be one of {list(VALID_ACTION_CLASSES)}, "
            f"got {action_class!r}"
        )

    def _decorate(inner_func: Callable) -> Callable:
        @functools.wraps(inner_func)
        def wrapper(*args, **kwargs):
            return inner_func(*args, **kwargs)

        wrapper._is_control = True
        wrapper._display_name = inner_func.__name__
        wrapper._control_scope = scope
        # None marks "undeclared" — the sentinel the conformance test reads
        # to require an explicit declaration on every shipped control.
        wrapper._control_action_class = action_class
        wrapper._control_panel = panel
        wrapper._ui_group = group or ""

        if params is not None:
            sig_names = [
                n for n in inspect.signature(inner_func).parameters if n != "self"
            ]
            if set(params) != set(sig_names):
                raise ValueError(
                    f"@control params for {inner_func.__name__}() name "
                    f"{sorted(params)} but the signature has "
                    f"{sorted(sig_names)} — they must match exactly."
                )
        wrapper._control_specs = dict(params) if params else {}

        # Resolve annotations (handles `from __future__ import annotations` string form).
        try:
            hints = typing.get_type_hints(inner_func)
        except Exception:
            hints = {}

        sig = inspect.signature(inner_func)
        sig_param_info: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            param_info: dict[str, Any] = {"name": name}
            resolved_type = hints.get(name)
            if resolved_type is not None:
                param_info["type"] = resolved_type
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default
            sig_param_info[name] = param_info

        wrapper._control_params = sig_param_info
        return wrapper

    if func is not None:
        # Bare form: @control
        return _decorate(func)
    # Parametrized form: @control(scope="operation")
    return _decorate


def get_monitored_methods(cls_or_instance, kinds: frozenset[str] | None = None) -> list[str]:
    """Return names of the @monitored methods on a class or instance.

    Args:
        cls_or_instance: A VI class or instance.
        kinds: Only methods whose declared kind is in this set; ``None``
            (the default) returns every monitored method, whatever its kind.
            ``{"scalar"}`` is what the monitor tick polls, ``ARRAY_KINDS``
            what the array poll reads.

    Returns:
        The method names, sorted.
    """
    methods = []
    for name in dir(cls_or_instance):
        try:
            attr = getattr(cls_or_instance, name)
        except AttributeError:
            continue
        if not (callable(attr) and getattr(attr, "_is_monitored", False)):
            continue
        if kinds is not None and get_monitored_kind(attr) not in kinds:
            continue
        methods.append(name)
    return methods


def get_control_methods(cls_or_instance) -> dict[str, dict]:
    """Return {method_name: param_info_dict} for all @control methods."""
    methods = {}
    for name in dir(cls_or_instance):
        try:
            attr = getattr(cls_or_instance, name)
        except AttributeError:
            continue
        if callable(attr) and getattr(attr, "_is_control", False):
            methods[name] = getattr(attr, "_control_params", {})
    return methods


def get_control_specs(method: Callable) -> dict[str, Any]:
    """Return a @control method's declared ``{param_name: ParamSpec}``.

    Args:
        method: A callable, typically a bound VI method.

    Returns:
        The ``params`` mapping given at decoration time, or ``{}`` when the
        control declared none (the GUI then falls back to signature-derived
        text inputs from ``_control_params``).
    """
    return getattr(method, "_control_specs", {})


def get_control_panel(method: Callable) -> bool:
    """Return a @control method's default monitor-card placement.

    Args:
        method: A callable, typically a bound VI method.

    Returns:
        ``True`` (the default for undecorated/legacy controls) when the
        control should appear on the compact monitor card; ``False`` when it
        belongs in the instrument front panel only.
    """
    return getattr(method, "_control_panel", True)


def get_monitored_unit(method: Callable) -> str | None:
    """Return a @monitored method's declared unit label.

    Args:
        method: A callable, typically a bound VI method.

    Returns:
        The ``unit=`` string given at decoration time — ``""`` for a
        deliberately dimensionless reading — or ``None`` when the method
        declared none (which the declaration standard forbids on a shipped
        VI; see ``virtual_instruments/README.md``).
    """
    return getattr(method, "_monitored_unit", None)


def get_monitored_description(method: Callable) -> str:
    """Return a @monitored method's declared description.

    Args:
        method: A callable, typically a bound VI method.

    Returns:
        The ``description=`` string given at decoration time, or ``""`` when
        the method declared none.
    """
    return getattr(method, "_monitored_description", "")


def get_monitored_kind(method: Callable) -> str:
    """Return a @monitored method's declared value kind.

    Args:
        method: A callable, typically a bound VI method.

    Returns:
        One of ``MONITORED_KINDS`` — ``"scalar"`` when none was declared.
    """
    return getattr(method, "_monitored_kind", "scalar")


def get_monitored_shape(method: Callable) -> tuple[int, ...] | None:
    """Return an array @monitored method's declared shape.

    Args:
        method: A callable, typically a bound VI method.

    Returns:
        ``(height_px, width_px)`` for an image, ``(length,)`` for a trace,
        ``None`` for a scalar.
    """
    return getattr(method, "_monitored_shape", None)


def get_monitored_period_s(method: Callable) -> float | None:
    """Return an array @monitored method's poll period in seconds.

    Args:
        method: A callable, typically a bound VI method.

    Returns:
        The declared (or defaulted) period, or ``None`` for a scalar, which
        is polled every monitor tick.
    """
    return getattr(method, "_monitored_period_s", None)


def get_monitored_axis(method: Callable) -> tuple[float, float, str] | None:
    """Return a trace @monitored method's declared physical x axis.

    Args:
        method: A callable, typically a bound VI method.

    Returns:
        ``(start, stop, unit)``, or ``None`` when none was declared (the
        trace is then plotted against its sample index).
    """
    return getattr(method, "_monitored_axis", None)


def get_ui_group(method: Callable) -> str:
    """Return a @monitored or @control method's UI-group tag.

    Args:
        method: A callable, typically a bound VI method.

    Returns:
        The ``group=`` key given at decoration time, or ``""`` when the
        method belongs to no group (the default: an ungrouped capability is
        rendered after every declared group).
    """
    return getattr(method, "_ui_group", "")


def get_control_scope(method: Callable) -> str:
    """Return a @control method's capability scope, defaulting to "measurement".

    Args:
        method: A callable, typically a bound or unbound VI method. A method
            never decorated with ``@control`` (or without the marker
            attribute at all) is treated as ``"measurement"``-scope — the
            enforcement default for undecorated methods.

    Returns:
        ``"measurement"`` or ``"operation"``.
    """
    return getattr(method, "_control_scope", "measurement")


def get_control_action_class(method: Callable) -> str:
    """Return a @control method's declared action class.

    The read side of the action-class declaration (see the module
    docstring). A control that declared none — and any method that is not a
    ``@control`` at all — reports ``DEFAULT_ACTION_CLASS``, the most
    restrictive class, so a missing declaration never widens authority. The
    raw ``_control_action_class`` marker stays ``None`` in that case, which
    is what ``tests/test_conformance.py`` reads to require an explicit
    declaration on every shipped control.

    Args:
        method: A callable, typically a bound or unbound VI method.

    Returns:
        One of ``VALID_ACTION_CLASSES``.
    """
    return getattr(method, "_control_action_class", None) or DEFAULT_ACTION_CLASS
