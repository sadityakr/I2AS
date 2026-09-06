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


def monitored(
    func: Callable | None = None,
    *,
    unit: str | None = None,
    description: str = "",
    group: str | None = None,
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

    Returns:
        The wrapped method (bare form) or a decorator (parametrized form).

    Raises:
        TypeError: If ``unit``, ``description`` or ``group`` is not a string.
        ValueError: If ``group`` is an empty string.
    """
    _check_declaration_strings(unit=unit, description=description, group=group)

    def _decorate(inner_func: Callable) -> Callable:
        @functools.wraps(inner_func)
        def wrapper(*args, **kwargs):
            return inner_func(*args, **kwargs)

        wrapper._is_monitored = True
        wrapper._display_name = inner_func.__name__
        wrapper._monitored_unit = unit
        wrapper._monitored_description = description
        wrapper._ui_group = group or ""
        return wrapper

    if func is not None:
        # Bare form: @monitored
        return _decorate(func)
    # Parametrized form: @monitored(unit=..., description=...)
    return _decorate


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


def get_monitored_methods(cls_or_instance) -> list[str]:
    """Return names of all @monitored methods on a class or instance."""
    methods = []
    for name in dir(cls_or_instance):
        try:
            attr = getattr(cls_or_instance, name)
        except AttributeError:
            continue
        if callable(attr) and getattr(attr, "_is_monitored", False):
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
