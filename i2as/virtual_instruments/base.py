"""BaseVirtualInstrument and category base classes.

All VIs inherit from BaseVirtualInstrument (and possibly one of the typed
sub-bases: MagnetBase, TemperatureControllerBase, StageBase,
MeasurementInstrumentBase).

Do NOT import from Station, Orchestrator, or Procedure here.
"""

from __future__ import annotations

import functools
import inspect
import logging
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from i2as.core.events import LifecycleState
from i2as.core.logging_config import VI_LOGGER_PREFIX
from i2as.core.exceptions import (
    I2ASCommunicationError,
    I2ASConfigError,
    I2ASSafetyError,
)
from i2as.core.plan import ImageBlock, ParamSpec, UIGroup
from i2as.virtual_instruments.rampable import RampableVI

logger = logging.getLogger(__name__)

#: Config key naming a setup's excitation-current ceiling, in amperes.
#: Every VI that drives current through the sample reads it — directly (a
#: current-sourcing VI bounds the sourced current by it) or derived (a
#: voltage-sourcing VI converts it into an amplitude bound through the
#: series resistance it drives). Limits are properties of the setup, so the
#: VALUE always comes from the config; this module only names the key.
MAX_SOURCE_CURRENT_KEY: str = "max_source_current_A"

#: ``control_limits`` limit name for a directly sourced excitation current
#: (the control-validation standard). Shared by every current-sourcing
#: measurement VI so one config key populates one named bound.
EXCITATION_CURRENT_LIMIT: str = "source_current_A"


def _populate_excitation_current_limit(
    vi: BaseVirtualInstrument, init_params: Mapping[str, Any]
) -> None:
    """Set ``vi._limits[EXCITATION_CURRENT_LIMIT]`` from the setup config.

    The one place the excitation ceiling is turned into a bound, shared by
    every VI that sources current directly, so no two current-sourcing VIs
    can drift apart in how they read the same config key.

    The bound is symmetric: reversing the current is routine in resistance
    work (thermal-EMF cancellation), and the hazard is the magnitude either
    way. A missing key leaves the limit populated but unbounded on both
    sides — the same "absent means no bound" rule the temperature and magnet
    VIs follow — so an older hand-written
    config still builds; the conformance suite is what requires every SHIPPED
    config to declare it.

    Args:
        vi: The VI being constructed (its ``_limits`` dict is written).
        init_params: The VI's config ``init_params``.
    """
    raw = init_params.get(MAX_SOURCE_CURRENT_KEY)
    if raw is None:
        logger.warning(
            "%s: config declares no '%s' — the excitation current is "
            "UNBOUNDED for this setup.",
            type(vi).__name__,
            MAX_SOURCE_CURRENT_KEY,
        )
        vi._limits[EXCITATION_CURRENT_LIMIT] = (None, None)
        return
    ceiling = abs(float(raw))
    vi._limits[EXCITATION_CURRENT_LIMIT] = (-ceiling, ceiling)


class BaseVirtualInstrument:
    """Root class for all I2AS Virtual Instruments.

    Subclass contract
    -----------------
    * Override ``initiate()`` to put the instrument into its operating state.
    * Override ``standby()`` to put the instrument in a safe idle state.
    * Tag read-only polling methods with ``@monitored``.
    * Tag user-callable action methods with ``@control``.
    * The constructor signature MUST be ``__init__(self, drivers, **init_params)``.

    Connection-lifecycle standard
    -----------------------------
    Every instrument has TWO independent lifecycles, and mixing them is the
    mistake this standard exists to prevent:

    * **Connection** — ``connect`` / ``disconnect``: who owns the *bus
      session*. Disconnected means I2AS holds nothing, so the operator
      can drive the instrument from its physical front panel or from the
      vendor's own software. Connecting and disconnecting NEVER change what
      the instrument is doing.
    * **Operating state** — ``initiate()`` / ``standby()``: what the
      instrument is *doing*. ``initiate()`` sends the setup commands that
      bring it to its operating state; ``standby()`` returns it to a safe
      idle one. Both are explicit operator (or procedure) actions.

    Three rules follow, and they bind every VI and every driver:

    1. **Construction is silent.** A VI's ``__init__`` (and its drivers')
       must send NO command that changes what the instrument is doing — no
       output, mode, range, rate or setpoint. Building the Station is a
       *connection* act: the only command it sends is the identity query
       (``ping()`` below). A setup command that used to live in ``__init__``
       belongs in ``initiate()``. Enforced by
       ``tests/test_conformance.py::test_vi_construction_sends_no_commands``.
    2. **Disconnecting is not standing down.** ``disconnect()`` releases; it
       must not zero a source, ramp a magnet down, or open a switch. An
       operator who wants the instrument safe first presses Standby first —
       that ordering is theirs to choose, and a magnet at field deliberately
       stays at field across a disconnect, exactly as it would if they were
       driving the PSU by hand.
    3. **A disconnected instrument degrades exactly like one that never
       connected.** ``Station.disconnect_instrument()`` moves the VI out of
       the live registry and into the offline registry, so polling, safety
       evaluation, procedures and the GUI all see the same "this instrument
       is not here" state they see for a startup connection failure — one
       degraded path, not two.

    The VI's own hooks are ``ping()`` (the identity check) and
    ``disconnect()`` (release VI-held state). Reconnecting is the Station's
    job (``Station.connect_instrument()``), because only it holds the build
    recipe needed to construct fresh drivers.

    Lifecycle-state standard
    ------------------------
    The operating half of the connection lifecycle is also DATA, not just a
    pair of verbs: every VI carries its lifecycle state — ``"idle"``,
    ``"initiated"`` or ``"standby"``, the ``LifecycleState`` vocabulary the
    contract declares (``core/events.py``) — and ``lifecycle_state()`` reads
    it. The rule this exists to enforce: a client RENDERS the state, it never
    reconstructs it from the actions it happened to witness. An instrument
    stood down by a path that emits no per-VI action — an emergency's
    blanket ``Station.standby_all()`` — reports the truth on the very next
    ``StatusSnapshot`` all the same, because the fact travels with the VI
    rather than with the action.

    Three rules bind every VI, and a VI author writes nothing to obey them:

    1. **The verbs own the fact.** ``initiate()`` sets ``"initiated"``,
       ``standby()`` sets ``"standby"``, ``disconnect()`` resets to
       ``"idle"``, and a measurement VI's ``initiate_measurement()`` — the
       arming half of its own lifecycle — sets ``"initiated"`` like the
       plain verb. Each is written only AFTER the call returns without
       raising, so a refused or failed stand-down never claims to have
       happened. Maintained by ``__init_subclass__``'s wrap of a directly
       defined method (the same inherited-enforcement idiom the
       control-validation standard uses for ``@control``) and by these base
       methods' own bodies, for a VI that inherits them unchanged.
    2. **The read is pure.** ``lifecycle_state()`` returns the cached value
       and sends nothing on the bus — the same purity rule
       ``control_param_specs()`` follows — so any client may ask at any
       time, including inside the status-snapshot assembly.
    3. **Hardware may correct it, the cache still answers.** A VI that can
       genuinely observe what the instrument is doing (an output that reads
       back on/off) overrides ``observe_lifecycle_state()``, which the
       monitor cycle consults once per poll, off the values it already read.
       That refreshes the cached value; it never becomes the read itself.

    Distinct from ``standby_status()``, which answers a narrower question —
    is this VI at, or converging on, the state its own ``standby()`` drives
    it to — from command provenance, for hold enforcement. Lifecycle state
    is the operator-facing fact the instrument card renders.

    Detach-when-idle declaration
    -----------------------------
    A single-client instrument — firmware that serves one connection at a
    time, so the vendor's own tool and I2AS cannot both hold it — may
    additionally opt into automatic release between uses. This is narrower
    than ``disconnect()``: connect/disconnect is still the OPERATOR's choice
    to give the instrument up entirely, whereas detach-when-idle is the VI
    itself releasing its session at the end of every ``standby()`` call, so
    the operator can open the vendor's tool between runs with no explicit
    Disconnect.

    A VI opts in by overriding the read-only ``detach_when_idle`` property
    with its own one-liner, e.g. ``return self._configured_externally`` (the
    whole opt-in for an externally configured instrument — see
    ``MeasurementInstrumentBase``'s "Externally configured instruments"
    section). Overriding this property is declaring
    a FIRMWARE FACT — this instrument serves one client at a time — never a
    place to write behaviour: the release itself is entirely the base
    class's job, driven automatically by ``__init_subclass__``'s wrap of a
    directly defined ``standby()`` (the same inherited-enforcement idiom the
    control-validation standard uses for ``@control``), or by the base
    ``standby()`` itself for a VI that inherits it unchanged.

    A VI declaring ``detach_when_idle`` MUST own every driver alias in its
    config ``drivers:`` mapping exclusively — no other configured VI may
    name the same alias. ``_detach()`` releases every driver in
    ``self._drivers`` unconditionally; it has no notion of another VI still
    needing that same session, unlike ``Station.disconnect_instrument()``,
    which routes a shared alias through ``_exclusive_aliases()`` before
    closing anything (``core/station.py``). A VI cannot make that same
    check itself — a VI is Layer 1 and the alias map lives on the Station
    (Layer 2); consulting it would be an upward import, which this
    architecture forbids. So the constraint is declared here and
    machine-checked at the config level instead (see
    ``tests/test_conformance.py``'s exclusivity test over every shipped
    config), rather than guarded at runtime.

    ``is_attached()`` is the observable half: ``True`` (the default) for a
    VI that always holds its session, ``False`` while a ``detach_when_idle``
    VI is detached. Deliberately NOT ``@monitored`` — that would add a row
    to every instrument card and a channel to every trend tier — it reaches
    the GUI only through the Availability standard's ``"detached"`` tag (see
    GLOSSARY.md's **Availability** and ``core/availability.py``), which
    ``Station._build_availability()`` derives from it. ``_attach()`` /
    ``_detach()`` are the base helpers that do the acquiring/releasing
    (``driver.ensure_connected()`` / ``driver.close()`` across
    ``self._drivers``) and flip the flag ``is_attached()`` reads; both are
    idempotent and never raise, so a failed reattach or release never blocks
    a caller. ``ensure_connected()`` is deliberately NOT part of the driver
    contract (``drivers/README.md`` states a closed driver is never reopened
    in place — the Station builds a fresh instance instead) — it is an
    opt-in capability duck-typed via ``getattr``, present only on firmware
    that genuinely supports resuming a session in place.

    Control-validation standard
    ---------------------------
    Every numeric ``@control`` parameter with a physical bound MUST be covered
    by the declarative limits mechanism:

    1. Declare ``control_limits`` on the class, mapping method name →
       ``{parameter_name: limit_name}``.
    2. In ``__init__``, populate ``self._limits[limit_name] = (lo, hi)`` from
       ``init_params`` (setup-specific values from the config YAML; ``None``
       means unbounded on that side). A limit may be derived (e.g. max field
       from max current), but the *values* always originate from the config,
       because limits are properties of the setup, not the code.
    3. Enforcement is inherited: ``__init_subclass__`` wraps every ``@control``
       method so an out-of-range value raises ``I2ASSafetyError`` with the
       reason BEFORE any hardware command is sent. A declared limit_name that
       was never populated raises ``I2ASConfigError`` (loud standard
       violation, caught by the conformance tests).

    Rules that cannot be expressed as a numeric range (e.g. "never energise
    the switch heater across a PSU/coil current mismatch") are written as
    explicit checks at the top of the ``@control`` method, raising
    ``I2ASSafetyError`` with a human-readable reason.

    Subclasses that ADD limits must merge, not replace::

        control_limits = {**ParentVI.control_limits, "set_x": {"x": "x_lim"}}

    UI-group standard
    -----------------
    A VI may declare titled groups of its own capabilities in the
    ``ui_groups`` class attribute, a tuple of ``core.plan.UIGroup``::

        ui_groups: ClassVar[tuple[UIGroup, ...]] = (
            UIGroup(
                key="heater",
                title="Heater",
                description="Closed-loop and manual heater control.",
                members=("heater_mode", "heater_output",
                         "set_heater_mode", "set_heater_output"),
            ),
        )

    Declared order is render order and manifest order; a group's ``members``
    tuple is the order WITHIN the group, and every member names a
    ``@monitored`` or ``@control`` method of this class. A method may also
    carry the matching ``@monitored(group="heater")`` / ``@control(group=
    "heater")`` tag, which is the declaration a reader of the method sees;
    tag and membership must agree.

    Groups are presentation and description only — they title what the
    instrument front panel renders and what the capability manifest
    describes. NOTHING about a group crosses the action queue: a control is
    still submitted alone, by method name, with flat scalar kwargs, and a
    group implies no atomicity and no ordering guarantee.

    ``__init_subclass__`` validates the declaration at class creation, so a
    renamed or moved method cannot leave a dangling tag: group keys must be
    unique, every member must exist as a monitored or control method, every
    ``group=`` tag must name a declared group, and a member's own tag (if
    any) must name the group that lists it. Each failure raises at import
    naming the VI class and the offending method or key.

    Subclasses that ADD groups must merge, not replace, exactly like
    ``control_limits``::

        ui_groups = (*ParentVI.ui_groups, UIGroup(...))

    Safety-flag manifest standard
    ------------------------------
    Every flag a VI's ``evaluate_safety()`` can report MUST be declared in
    the class attribute ``safety_flags``, mapping the flag name to its
    severity::

        safety_flags: ClassVar[dict[str, str]] = {"quench": "critical"}

    Severity is one of three values (the System-Condition standard's
    severity ladder; scope follows from severity):

    * ``"advisory"`` — no enforcement (reserved for future use).
    * ``"hold"`` — scoped: on onset, every VI whose ``safety_concerns()``
      names the flag is stood by and refused manual control (see
      ``Station.update_conditions()``), without affecting the rest of
      the station.
    * ``"critical"`` — station-wide by definition: the flag forces
      EMERGENCY the instant it trips, regardless of what any VI's
      ``safety_concerns()`` declares. Consequently no VI may list a
      critical flag in ``safety_concerns()`` — a per-VI hold would be
      meaningless once EMERGENCY has already stopped everything.

    ``safety_flags`` is the single declaration point for a flag's
    severity — a flag's meaning never varies by which VI happens to
    report it. Declare it on the base class that owns the flag's physical
    semantics (e.g. ``MagnetBase`` for ``"quench"``), not on each
    concrete VI, so every VI of that category inherits the correct
    classification automatically.

    Subclass manifests are MERGED over the MRO, exactly like
    ``control_limits`` — a subclass may ADD new flags but must NOT
    contradict a severity a base already declared for the same flag
    (checked by ``tests/test_conformance.py``, not at import time)::

        safety_flags = {**ParentVI.safety_flags, "new_flag": "hold"}

    Use ``merged_safety_flags()`` to read the fully-merged manifest — the
    single read side of this declaration. Severity alone determines scope
    (the System-Condition standard, ``core/conditions.py``): critical is
    station-wide by construction, so no derived "critical subset" accessor
    is needed by any caller — ``Station.active_critical_conditions()``
    reads the live registry directly.
    """

    vi_type: str = "unknown"
    # vi_name is set by the Station factory after instantiation, not in __init__.
    vi_name: str = ""

    # Declarative control limits: {method_name: {param_name: limit_name}}.
    # See "Control-validation standard" in the class docstring.
    control_limits: dict[str, dict[str, str]] = {}

    # Declarative safety-flag manifest: {flag_name: severity}, severity one
    # of "advisory" | "hold" | "critical". See "Safety-flag manifest
    # standard" in the class docstring. Merged across the MRO by
    # merged_safety_flags() — a subclass declares only the flags it adds.
    safety_flags: ClassVar[dict[str, str]] = {}

    # Declarative UI-group manifest: the titled groups this VI's own
    # capabilities fall into, in render order. See "UI-group standard" in the
    # class docstring; validated by __init_subclass__.
    ui_groups: ClassVar[tuple[UIGroup, ...]] = ()

    # The standby-provenance standard's command-history bit (see
    # ``standby_status()``): a CLASS-level default of False, shadowed by an
    # instance attribute the first time a wrapped standby()/start_ramp()/
    # stop_ramp() writes to it (see ``_make_standby_wrapper`` and
    # ``_make_standby_invalidation_wrapper``). This is deliberately not set
    # in ``__init__`` — a freshly constructed VI, including a subclass that
    # defines none of the three wrapped methods and simply inherits them,
    # reads this class attribute and gets False ("away") until the first
    # standby()/start_ramp()/stop_ramp() call gives it an instance value.
    # Deliberately NOT annotated ``ClassVar``, unlike ``safety_flags`` above:
    # ClassVar means "never assigned on an instance", which is the opposite
    # of how this attribute works.
    _standby_commanded: bool = False

    # The lifecycle-state standard's observed fact (see the class docstring
    # and ``lifecycle_state()``): a CLASS-level default of ``"idle"``,
    # shadowed by an instance attribute the first time a wrapped
    # ``initiate()``/``initiate_measurement()``/``standby()``/``disconnect()``
    # writes to it. Deliberately not set in ``__init__`` and deliberately NOT
    # annotated ``ClassVar``, for exactly the reasons ``_standby_commanded``
    # above gives: a freshly constructed VI — including a subclass that
    # defines none of the wrapped methods — reads this class attribute and
    # gets "idle" until the first lifecycle call gives it an instance value.
    _lifecycle_state: str = LifecycleState.IDLE.value

    # ── Reading-loop participation (see GLOSSARY "Reading loop") ──────────
    # A VI in the reading path may declare parameters the generic sweep
    # procedure can loop at every sweep point: ``reading_setters`` maps a
    # parameter name to the cheap setter method that reprograms just that
    # quantity between readings (the setter accepts the parameter under its
    # own name); ``reading_parameters`` supplies each such parameter's
    # ParamSpec (enumerated specs render as checkboxes in the Reading loop
    # form group, free specs as a comma-separated value list); and
    # ``reading_safe_off`` optionally names a method the procedure dispatches
    # at standby/abort when this (non-measurement) VI took part in the loop
    # (e.g. a switch's ``open_all``). Defaults: no participation.
    reading_setters: ClassVar[dict[str, str]] = {}
    reading_safe_off: ClassVar[str] = ""

    @property
    def reading_parameters(self) -> dict[str, ParamSpec]:
        """Return ``{name: ParamSpec}`` for every ``reading_setters`` parameter.

        The base implementation returns ``{}`` (no loopable parameters). A VI
        that declares ``reading_setters`` must override this so the generic
        sweep procedure can render and validate the loop values; a
        measurement VI inherits an implementation reading its own
        ``measurement_parameters`` (see ``MeasurementInstrumentBase``).

        Returns:
            Mapping of loopable parameter name to its ``ParamSpec``.
        """
        return {}

    # ------------------------------------------------------------------
    # __init_subclass__: auto-wrap @monitored / @control methods
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap every @monitored and @control method defined on *cls* with logging.

        Only methods defined directly on *cls* (via ``vars(cls)``) are wrapped,
        so that inherited wrappers are not double-wrapped.
        """
        super().__init_subclass__(**kwargs)
        for attr_name, attr_value in vars(cls).items():
            if not callable(attr_value):
                continue
            is_monitored = getattr(attr_value, "_is_monitored", False)
            is_control = getattr(attr_value, "_is_control", False)
            if is_monitored or is_control:
                if is_control:
                    # Innermost wrapper: declarative limit enforcement (the
                    # control-validation standard). Composed inside the
                    # logging wrapper so rejections are also logged.
                    attr_value = BaseVirtualInstrument._make_limit_wrapper(
                        attr_value, attr_name
                    )
                wrapped = BaseVirtualInstrument._make_logging_wrapper(attr_value, attr_name)
                # Preserve the marker attributes so discovery still works
                if is_monitored:
                    wrapped._is_monitored = True
                    wrapped._display_name = getattr(attr_value, "_display_name", attr_name)
                    # The monitored declaration (unit/description) and the
                    # UI-group tag, carried through the wrap the same way the
                    # control metadata below is.
                    wrapped._monitored_unit = getattr(
                        attr_value, "_monitored_unit", None
                    )
                    wrapped._monitored_description = getattr(
                        attr_value, "_monitored_description", ""
                    )
                    wrapped._ui_group = getattr(attr_value, "_ui_group", "")
                if is_control:
                    wrapped._is_control = True
                    wrapped._display_name = getattr(attr_value, "_display_name", attr_name)
                    wrapped._control_params = getattr(attr_value, "_control_params", {})
                    wrapped._control_scope = getattr(
                        attr_value, "_control_scope", "measurement"
                    )
                    # Rich GUI metadata (widget specs + default card placement).
                    # The decorator stores the specs opaquely (contract C1: it
                    # may not import ParamSpec); this is the earliest layer
                    # that knows the type, so enforce it here at class
                    # creation — a wrong spec fails at import, not on click.
                    specs = getattr(attr_value, "_control_specs", {})
                    for param_name, spec in specs.items():
                        if not isinstance(spec, ParamSpec):
                            raise TypeError(
                                f"{cls.__name__}.{attr_name}: @control params["
                                f"{param_name!r}] must be a ParamSpec, got "
                                f"{type(spec).__name__}"
                            )
                    wrapped._control_specs = specs
                    wrapped._control_panel = getattr(attr_value, "_control_panel", True)
                    wrapped._ui_group = getattr(attr_value, "_ui_group", "")
                setattr(cls, attr_name, wrapped)

        # The detach-when-idle declaration's enforcement (see the class
        # docstring) AND the standby-provenance standard (see
        # ``standby_status()``): wrap a directly defined standby() so the
        # release AND the ``_standby_commanded`` flag are set after it
        # returns, the same inherited-enforcement idiom used above for
        # @control. This wrap is UNCONDITIONAL — at class-creation time
        # ``vars(cls)["detach_when_idle"]`` (when overridden at all) is a
        # property OBJECT, not a resolved bool, so whether to detach can
        # only be decided at CALL time, inside the wrapper itself (see
        # _make_standby_wrapper). A VI that does not define its own
        # standby() is not touched here — it inherits BaseVirtualInstrument.
        # standby(), which sets the flag and detaches directly, without
        # needing this wrap.
        standby_method = vars(cls).get("standby")
        if callable(standby_method):
            cls.standby = BaseVirtualInstrument._make_standby_wrapper(standby_method)

        # The lifecycle-state standard's other three verbs (see the class
        # docstring): a directly defined initiate() / initiate_measurement()
        # / disconnect() is wrapped so the state is recorded after the VI's
        # own commands ran, with no VI author writing anything. Same
        # ``vars(cls)`` discipline as the standby wrap above — an inherited
        # method is already wrapped and must not be wrapped twice — and the
        # same after-the-call timing, so a raise leaves the state untouched.
        # ``standby()`` is not listed: its own wrapper above records
        # ``"standby"`` alongside the detach and the standby provenance.
        for lifecycle_method, resulting_state in (
            ("initiate", LifecycleState.INITIATED),
            ("initiate_measurement", LifecycleState.INITIATED),
            ("disconnect", LifecycleState.IDLE),
        ):
            method = vars(cls).get(lifecycle_method)
            if callable(method):
                setattr(
                    cls,
                    lifecycle_method,
                    BaseVirtualInstrument._make_lifecycle_wrapper(
                        method, resulting_state
                    ),
                )

        # The standby-provenance standard's invalidation half: any directly
        # defined start_ramp()/stop_ramp() clears _standby_commanded, since
        # either means the VI is no longer converging on (or resting at) the
        # state standby() drives it to. Same vars(cls) discipline as above —
        # a subclass that does not redefine one of these inherits the
        # already-wrapped parent version and is not re-wrapped.
        for ramp_method_name in ("start_ramp", "stop_ramp"):
            ramp_method = vars(cls).get(ramp_method_name)
            if callable(ramp_method):
                setattr(
                    cls,
                    ramp_method_name,
                    BaseVirtualInstrument._make_standby_invalidation_wrapper(
                        ramp_method
                    ),
                )

        # The UI-group standard (see the class docstring): resolve every
        # group= tag against the declared ui_groups now, at class creation,
        # so a renamed or moved method can never leave a dangling tag for a
        # renderer to trip over at runtime.
        BaseVirtualInstrument._validate_ui_groups(cls)

    # ------------------------------------------------------------------
    # UI-group validation (see the class docstring's "UI-group standard")
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_ui_groups(cls: type) -> None:
        """Check *cls*'s ``ui_groups`` declaration and every ``group=`` tag.

        Runs at class creation over the FULLY RESOLVED capability surface
        (inherited monitored/control methods included), so a subclass that
        inherits a tagged method also inherits the obligation to declare its
        group.

        Args:
            cls: The Virtual Instrument class being created.

        Raises:
            TypeError: If ``ui_groups`` is not a tuple of ``UIGroup``.
            ValueError: If two groups share a key, if a group names a member
                that is not a ``@monitored`` or ``@control`` method of this
                class, if a method's ``group=`` tag names no declared group,
                or if a member's own tag names a different group.
        """
        groups = cls.ui_groups
        if not isinstance(groups, tuple):
            raise TypeError(
                f"{cls.__name__}.ui_groups must be a tuple of UIGroup, got "
                f"{groups!r}"
            )
        for group in groups:
            if not isinstance(group, UIGroup):
                raise TypeError(
                    f"{cls.__name__}.ui_groups entries must be UIGroup, got "
                    f"{type(group).__name__}"
                )

        keys: list[str] = []
        for group in groups:
            if group.key in keys:
                raise ValueError(
                    f"{cls.__name__}.ui_groups declares the group key "
                    f"{group.key!r} twice — group keys must be unique"
                )
            keys.append(group.key)

        # The capability surface: every monitored or control method reachable
        # on this class, with its tag.
        tags: dict[str, str] = {}
        for attr_name in dir(cls):
            try:
                attr = getattr(cls, attr_name)
            except AttributeError:
                continue
            if not callable(attr):
                continue
            if getattr(attr, "_is_monitored", False) or getattr(
                attr, "_is_control", False
            ):
                tags[attr_name] = getattr(attr, "_ui_group", "")

        owner: dict[str, str] = {}
        for group in groups:
            for member in group.members:
                if member not in tags:
                    raise ValueError(
                        f"{cls.__name__}.ui_groups[{group.key!r}] names member "
                        f"{member!r}, which is not a @monitored or @control "
                        f"method of {cls.__name__}"
                    )
                if member in owner:
                    raise ValueError(
                        f"{cls.__name__}.ui_groups: method {member!r} is a "
                        f"member of both {owner[member]!r} and {group.key!r} — "
                        f"a capability belongs to at most one group"
                    )
                owner[member] = group.key

        for method_name, tag in tags.items():
            if not tag:
                continue
            if tag not in keys:
                raise ValueError(
                    f"{cls.__name__}.{method_name} is tagged group={tag!r}, "
                    f"which names no group in {cls.__name__}.ui_groups "
                    f"(declared: {keys})"
                )
            if owner.get(method_name, tag) != tag:
                raise ValueError(
                    f"{cls.__name__}.{method_name} is tagged group={tag!r} but "
                    f"is listed in the members of {owner[method_name]!r} — the "
                    f"tag and the group's members must agree"
                )

    # ------------------------------------------------------------------
    # standby() wrapper factory: detach-when-idle release + standby
    # provenance (see the class docstring's "Detach-when-idle declaration"
    # and ``standby_status()``'s docstring)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_standby_wrapper(method):
        """Return *method* wrapped with the detach and provenance side effects.

        Calls the ORIGINAL ``standby()`` first — so a concrete VI's own
        safe-off commands still run exactly as written — and only once it
        returns without raising does it (a) record ``self._standby_commanded
        = True`` (the standby-provenance standard: a raise leaves the flag
        ``False``, so the caller knows to retry) and (b) call
        ``self._detach()`` if ``self.detach_when_idle`` is true. Both flags
        are read/written on ``self`` at CALL time (never at class-creation
        time — see ``__init_subclass__``'s comment for why ``detach_when_idle``
        cannot be resolved earlier).

        Args:
            method: The subclass's directly defined ``standby``.

        Returns:
            The wrapped method, to be set back onto the class.
        """

        @functools.wraps(method)
        def wrapper(self, *args: Any, **kwargs: Any):
            result = method(self, *args, **kwargs)
            self._standby_commanded = True
            self._lifecycle_state = LifecycleState.STANDBY.value
            if self.detach_when_idle:
                self._detach()
            return result

        return wrapper

    # ------------------------------------------------------------------
    # Lifecycle-state wrapper factory (the lifecycle-state standard; see
    # the class docstring and ``lifecycle_state()``)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_lifecycle_wrapper(method, resulting_state: LifecycleState):
        """Return *method* wrapped to record the lifecycle state it reaches.

        Calls the ORIGINAL method first — a VI's own setup, arming or
        release commands run exactly as written — and records the new state
        only once it returns without raising, so a refused ``initiate()``
        never leaves a card claiming the instrument is running.

        Args:
            method: The subclass's directly defined ``initiate``,
                ``initiate_measurement`` or ``disconnect``.
            resulting_state: The state the VI is in once *method* succeeds.

        Returns:
            The wrapped method, to be set back onto the class.
        """

        @functools.wraps(method)
        def wrapper(self, *args: Any, **kwargs: Any):
            result = method(self, *args, **kwargs)
            self._lifecycle_state = resulting_state.value
            return result

        return wrapper

    # ------------------------------------------------------------------
    # start_ramp()/stop_ramp() wrapper factory: standby-provenance
    # invalidation (see ``standby_status()``'s docstring)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_standby_invalidation_wrapper(method):
        """Return *method* wrapped to clear ``_standby_commanded`` before it runs.

        Either commanding a new ramp target (``start_ramp``) or freezing the
        hardware mid-ramp (``stop_ramp``) means the VI is no longer at, or
        converging on, the state ``standby()`` drives it to — set
        immediately, before the wrapped call, so the invalidation holds even
        if the call itself goes on to raise.

        Args:
            method: The subclass's directly defined ``start_ramp`` or
                ``stop_ramp``.

        Returns:
            The wrapped method, to be set back onto the class.
        """

        @functools.wraps(method)
        def wrapper(self, *args: Any, **kwargs: Any):
            self._standby_commanded = False
            return method(self, *args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------
    # Limit-enforcement wrapper factory (control-validation standard)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_limit_wrapper(method, method_name: str):
        """Return *method* guarded by the class's declarative control limits.

        Looks up ``type(self).control_limits`` at call time, so a subclass can
        declare limits for methods it inherits without re-wrapping them.

        An out-of-range call raises ``I2ASSafetyError`` carrying both the
        operator-facing message and its structured form (``param``, ``value``,
        ``lo``, ``hi``, ``limit_name``), so the refusal reaches a verdict as
        fields rather than as prose to be parsed.
        """
        sig = inspect.signature(method)

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            limit_spec = type(self).control_limits.get(method_name)
            if limit_spec:
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()
                for param_name, limit_name in limit_spec.items():
                    if param_name not in bound.arguments:
                        continue
                    if limit_name not in getattr(self, "_limits", {}):
                        raise I2ASConfigError(
                            f"{type(self).__name__}.{method_name}: "
                            f"control_limits references limit '{limit_name}' "
                            f"but __init__ never populated self._limits with "
                            f"it — define it from init_params."
                        )
                    lo, hi = self._limits[limit_name]
                    value = float(bound.arguments[param_name])
                    if (lo is not None and value < lo) or (
                        hi is not None and value > hi
                    ):
                        lo_txt = "-inf" if lo is None else f"{lo:g}"
                        hi_txt = "+inf" if hi is None else f"{hi:g}"
                        # The message is the operator's banner; the keyword
                        # fields are the same refusal in structured form, so
                        # a caller turning this into a verdict reads fields
                        # instead of parsing prose. Never derive one from the
                        # other — the message text is asserted on verbatim.
                        raise I2ASSafetyError(
                            f"{self.vi_name or type(self).__name__}."
                            f"{method_name}: {param_name}={value:g} is outside "
                            f"the allowed range [{lo_txt}, {hi_txt}] for this "
                            f"setup (limit '{limit_name}' from the station "
                            f"config). Command refused.",
                            param=param_name,
                            value=value,
                            lo=lo,
                            hi=hi,
                            limit_name=limit_name,
                        )
            return method(self, *args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------
    # Logging wrapper factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_logging_wrapper(method, method_name: str):
        """Return a logging-instrumented version of *method*."""

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            log = logging.getLogger(f"{VI_LOGGER_PREFIX}{self.vi_name}")
            log.debug("%s.%s(%s, %s)", self.vi_name, method_name, args, kwargs)
            try:
                result = method(self, *args, **kwargs)
                log.debug("%s.%s -> %r", self.vi_name, method_name, result)
                return result
            except Exception as exc:
                log.error(
                    "%s.%s raised %s: %s",
                    self.vi_name,
                    method_name,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                # Wrap pyvisa VisaIOError → I2ASCommunicationError
                try:
                    import pyvisa  # type: ignore
                    if isinstance(exc, pyvisa.errors.VisaIOError):
                        raise I2ASCommunicationError(
                            str(exc), vi_name=self.vi_name, original_error=exc
                        ) from exc
                except ImportError:
                    pass
                raise

        return wrapper

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, drivers: dict[str, object], **init_params: Any) -> None:
        """Initialise the VI.

        Args:
            drivers: Mapping of role → driver instance.
                Single-driver VIs use ``{"main": driver}``.
                Multi-driver VIs use e.g. ``{"source": k6221, "meter": k2182a}``.
            **init_params: Additional parameters from YAML config (ramp rates,
                conversion factors, safety limits, …).
        """
        self._drivers = drivers
        self._init_params = init_params
        # Numeric control limits: {limit_name: (lo, hi)}, populated by each
        # VI's __init__ from init_params (None = unbounded on that side).
        # Referenced by name from the class's control_limits declaration.
        self._limits: dict[str, tuple[float | None, float | None]] = {}
        # The detach-when-idle declaration's attachment flag (see the class
        # docstring): True until a detach_when_idle VI's __init__ or
        # standby() calls self._detach(). Read by is_attached().
        self._attached: bool = True
        # The monitor-cycle cache behind last_monitored(): the values of
        # this VI's @monitored methods as of the most recent successful
        # get_state() poll. Empty until the first one. See
        # ``control_param_specs()``'s purity rule.
        self._monitored_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initiate(self) -> None:
        """Put the instrument into its operating state.

        The *operating* half of the connection-lifecycle standard (see the
        class docstring): this is where every setup command lives — heater
        mode, pole mode, slew rate, anything the instrument needs before it
        can be used. It is an explicit operator or procedure action, never
        something the Station does while building, so an operator who starts
        I2AS while an instrument is mid-experiment finds it untouched.

        Override in subclasses to send those setup commands.

        Records ``lifecycle_state() == "initiated"`` (the lifecycle-state
        standard, see the class docstring). A subclass that overrides this
        gets the same record from ``__init_subclass__``'s wrap instead
        (``_make_lifecycle_wrapper``), so this body is only ever reached
        directly by a VI with no override of its own.
        """
        self._lifecycle_state = LifecycleState.INITIATED.value

    def standby(self) -> None:
        """Put the instrument in a safe idle state.

        The opposite of ``initiate()``: override in subclasses to send
        safe-idle commands (e.g. disable outputs). Distinct from
        ``disconnect()``, which releases the bus session and changes nothing
        the instrument is doing.

        Honours the detach-when-idle declaration (see the class docstring),
        the standby-provenance standard (see ``standby_status()``) and the
        lifecycle-state standard (see ``lifecycle_state()``): a VI that
        inherits this base implementation unchanged — never overriding
        ``standby()`` itself — still records ``self._standby_commanded =
        True``, records ``lifecycle_state() == "standby"``, and releases its
        driver session here when ``detach_when_idle`` is true. A subclass that DOES override ``standby()`` gets both of
        those from ``__init_subclass__``'s wrap instead (``_make_standby_
        wrapper``), so this method body is only ever reached directly by a
        VI with no override of its own — the one call site
        ``__init_subclass__`` cannot reach, since it fires for subclasses
        only.
        """
        self._standby_commanded = True
        self._lifecycle_state = LifecycleState.STANDBY.value
        if self.detach_when_idle:
            self._detach()

    def limit_bounds(self, limit_name: str) -> tuple[float | None, float | None]:
        """Return the ``(lo, hi)`` bounds of one populated control limit.

        The public read side of the control-validation standard (see the class
        docstring): ``_limits`` is written by each VI's ``__init__`` from its
        config and read by the enforcement wrapper; this is how a caller
        OUTSIDE the VI — the Station, assembling the setup bounds an
        experiment's envelope narrows — asks what the setup already allows,
        without reaching into that dict.

        Args:
            limit_name: The limit's name, as referenced from
                ``control_limits``.

        Returns:
            ``(lo, hi)`` in the parameter's SI unit; ``None`` on a side means
            unbounded there. An unknown *limit_name* returns ``(None, None)``
            — "this setup bounds nothing by that name" — so a caller
            surveying limits needs no per-VI knowledge of which exist.
        """
        return self._limits.get(limit_name, (None, None))

    # ------------------------------------------------------------------
    # Connection lifecycle (the connection-lifecycle standard)
    # ------------------------------------------------------------------

    # Capability default for the detach-when-idle declaration (see the class
    # docstring): most VIs never release their session on their own, so the
    # base property below returns this. A config-sensitive VI narrows the
    # PROPERTY (not this ClassVar) in one line — see its docstring.
    _detach_when_idle_default: ClassVar[bool] = False

    @property
    def detach_when_idle(self) -> bool:
        """Whether this VI releases its driver session automatically at standby.

        The detach-when-idle declaration's opt-in (see the class docstring):
        a plain ``ClassVar[bool]`` cannot express it alone because the
        decision may depend on INSTANCE state, not just the class — e.g. a
        VI that detaches only when its ``configured_externally`` init param
        is true, not for every instance of the class. A VI opts in by
        overriding this property with its own one-liner::

            @property
            def detach_when_idle(self) -> bool:
                return self._configured_externally

        Overriding this property is declaring a FIRMWARE FACT — this
        instrument serves one client at a time — never a place to write
        behaviour: the release itself is entirely ``_detach()``'s job,
        dispatched automatically by ``__init_subclass__``'s ``standby()``
        wrap (or by this base's own ``standby()``, for a VI that inherits
        it unchanged).

        Returns:
            ``True`` if this VI should release its driver session whenever
            it stands down. Default ``False``.
        """
        return self._detach_when_idle_default

    def is_attached(self) -> bool:
        """Whether this VI currently holds its driver session(s).

        The observable half of the detach-when-idle declaration (GAP the
        Availability standard closes, see ``core/availability.py``):
        ``Station._build_availability()`` reads this to add the
        ``"detached"`` tag, so a released VI is visibly distinct from one
        that is fully held, instead of looking identical in the GUI.
        Deliberately NOT ``@monitored`` — that would add a row to every
        instrument card and a channel to every trend tier; it reaches the
        GUI only through the Availability record.

        Returns:
            ``True`` (the default) for a VI that always holds its session;
            ``False`` while a ``detach_when_idle`` VI is detached.
        """
        return self._attached

    def _attach(self) -> None:
        """Reacquire this VI's driver session(s) (detach-when-idle standard).

        Calls each driver's OPT-IN ``ensure_connected()`` capability,
        duck-typed via ``getattr`` — deliberately NOT part of the driver
        contract, which guarantees the opposite (``drivers/README.md``: a
        closed driver is never reopened in place, the Station builds a
        fresh instance to reconnect instead). ``ensure_connected()`` exists
        only for firmware that genuinely supports resuming a session in
        place; a driver with no such method is left untouched.

        Idempotent (a no-op once already attached) and NEVER RAISES — a
        failed reconnect here must not block a caller such as ``ping()``'s
        verify-and-release path. Leaves ``is_attached()`` False if any
        driver fails to reconnect, so a subsequent command against it fails
        on its own terms instead of this method masking the problem.
        """
        if self._attached:
            return
        ok = True
        for driver in self._drivers.values():
            ensure_connected = getattr(driver, "ensure_connected", None)
            if not callable(ensure_connected):
                continue
            try:
                ensure_connected()
            except Exception:  # noqa: BLE001 — a failed reattach must not raise
                logging.getLogger(f"{VI_LOGGER_PREFIX}{self.vi_name}").warning(
                    "%s: ensure_connected() failed while reattaching",
                    self.vi_name or type(self).__name__,
                    exc_info=True,
                )
                ok = False
        self._attached = ok

    def _detach(self) -> None:
        """Release this VI's driver session(s) (detach-when-idle standard).

        Calls ``close()`` — part of the driver contract, so unlike
        ``_attach()`` this needs no duck-typing — on every driver.
        Idempotent and NEVER RAISES: ``is_attached()`` is flipped False
        regardless of whether any individual ``close()`` call raised, since
        a failing release must never block standing down.
        """
        for driver in self._drivers.values():
            try:
                driver.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 — a failing release must not raise
                logging.getLogger(f"{VI_LOGGER_PREFIX}{self.vi_name}").warning(
                    "%s: close() failed while detaching",
                    self.vi_name or type(self).__name__,
                    exc_info=True,
                )
        self._attached = False

    # ------------------------------------------------------------------
    # Lifecycle state (the lifecycle-state standard; see the class docstring)
    # ------------------------------------------------------------------

    def lifecycle_state(self) -> str:
        """Return what this instrument is DOING, as a pure cached read.

        The lifecycle-state standard's read side (see the class docstring):
        the operating half of the connection lifecycle as a fact a client
        renders, rather than one it has to infer from whichever actions it
        happened to see. Sends NOTHING on the bus — the same purity rule
        ``control_param_specs()`` follows — so the Station may call it while
        assembling a status snapshot, on any tick, for every VI.

        Returns:
            One of ``LifecycleState``'s values (``core/events.py``):
            ``"idle"`` (never initiated, or reset by ``disconnect()``),
            ``"initiated"`` (``initiate()`` — or, for a measurement VI,
            ``initiate_measurement()`` — succeeded and nothing has stood it
            down since), or ``"standby"`` (``standby()`` succeeded).
        """
        return self._lifecycle_state

    def observe_lifecycle_state(self) -> str | None:
        """Return the lifecycle state this VI can PROVE from hardware, or None.

        The lifecycle-state standard's optional third rule (see the class
        docstring): a VI whose instrument reports what it is doing — a
        source whose output reads back on or off, a controller whose heater
        range is readable — overrides this so the monitor cycle can correct
        a cached value that a front-panel fiddle or a power cycle has made
        wrong. ``get_state()`` consults it once per poll, AFTER the
        ``@monitored`` reads, so an override answers from
        ``last_monitored()`` and sends nothing extra on the bus.

        Overriding this never changes where a client reads from:
        ``lifecycle_state()`` remains the read, and this only refreshes what
        it returns. An override that raises, or that answers with a value
        outside the vocabulary, is logged and ignored — reporting must never
        be the thing that breaks a monitor cycle.

        Returns:
            One of ``LifecycleState``'s values, or ``None`` (the default:
            "this instrument cannot tell me, keep the commanded value").
        """
        return None

    def _refresh_lifecycle_state(self) -> None:
        """Let ``observe_lifecycle_state()`` correct the cached lifecycle state.

        Called by ``get_state()`` at the end of every successful poll.
        Guarded on both failure modes an override can present, because the
        monitor cycle is the one thing that must not stop.
        """
        try:
            observed = self.observe_lifecycle_state()
        except Exception:  # noqa: BLE001 — an observation must not fail a poll
            logging.getLogger(f"{VI_LOGGER_PREFIX}{self.vi_name}").warning(
                "%s: observe_lifecycle_state() raised — keeping the "
                "commanded lifecycle state",
                self.vi_name or type(self).__name__,
                exc_info=True,
            )
            return
        if observed is None:
            return
        try:
            self._lifecycle_state = LifecycleState(observed).value
        except ValueError:
            logging.getLogger(f"{VI_LOGGER_PREFIX}{self.vi_name}").warning(
                "%s: observe_lifecycle_state() returned %r, which is not a "
                "LifecycleState — keeping the commanded lifecycle state",
                self.vi_name or type(self).__name__,
                observed,
            )

    # ------------------------------------------------------------------
    # Standby provenance (the command-provenance standard)
    # ------------------------------------------------------------------

    def standby_status(self) -> str:
        """Whether this VI is at the safe idle state ``standby()`` drives it to.

        Derived entirely from command PROVENANCE, never from a VI-specific
        notion of what its safe state physically is — no VI author writes
        anything to get this; ``__init_subclass__``'s wrap of a directly
        defined ``standby()``/``start_ramp()``/``stop_ramp()`` (or this
        base's own ``standby()``, for a VI that inherits it unchanged)
        maintains ``self._standby_commanded`` entirely on its own (see the
        class attribute's docstring).

        Returns:
            ``"reached"`` — at safe idle; nothing to enforce. Also the
                answer for any VI that is not a ``RampableVI``, since its
                ``standby()`` is one instantaneous command with no
                intermediate state to converge through.
            ``"converging"`` — ``standby()`` is underway (its last command
                was the standby ramp, not yet finished) and will arrive; do
                not re-command.
            ``"away"`` — neither: no standby command is in flight, so
                ``standby()`` must be (re-)issued.

        Command provenance is not physical verification: this method knows
        that the standby command was issued and that its ramp finished
        (``ramp_status() != "RAMPING"``), not that the hardware physically
        arrived — a PSU that silently ignores a ramp still reports
        ``"reached"`` here. The method stays overridable so a VI with its
        own means of checking can add a physics check on top.

        ``standby()`` is also the only safe response this contract models:
        it answers "is this VI at (or converging on) the state its OWN
        standby() drives it to", nothing about what safe response some other
        future safety flag might require — that belongs as a declaration on
        the VI itself, never as a special case here.
        """
        if not isinstance(self, RampableVI):
            return "reached"
        if not self._standby_commanded:
            return "away"
        return "converging" if self.ramp_status() == "RAMPING" else "reached"

    def ping(self) -> bool:
        """Send an identity query to every driver; True if all of them answer.

        The single connection check in I2AS — the ONLY command sent when
        the Station is built (see the class docstring's connection-lifecycle
        standard) and what the front panel's "Check connection" button and
        ``Station.connect_instrument()`` use. Harmless by construction: an
        identity query changes nothing.

        The default covers every VI whose drivers follow the driver contract
        (``get_idn()`` on each). A ``detach_when_idle`` VI gets a different
        shape automatically — the detach-when-idle declaration's
        verify-and-release path (see the class docstring): reattach, a TRUE
        round trip, then release again — returning a clean ``False`` rather
        than raising when the instrument is currently held by the external
        tool, and always releasing afterward so it is never left attached
        while idle. This is the ONE place that path lives; no VI needs its
        own ``ping()`` override to get it.

        Returns:
            True if every driver answered ``get_idn()``; False on any
            failure, including a driver that does not implement it.
        """
        if not self.detach_when_idle:
            try:
                for driver in self._drivers.values():
                    driver.get_idn()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 — any failure means "not reachable"
                return False
            return True

        self._attach()
        try:
            for driver in self._drivers.values():
                driver.get_idn()  # type: ignore[attr-defined]
            return True
        except Exception:  # noqa: BLE001 — any failure means "not reachable"
            return False
        finally:
            self._detach()

    def disconnect(self) -> None:
        """Release whatever this VI holds beyond its driver sessions.

        Called by ``Station.disconnect_instrument()`` immediately BEFORE the
        Station closes the driver sessions this VI exclusively owns (the
        Station, not the VI, owns that step, because a driver may be shared
        with another VI that is staying online).

        The default is a no-op: most VIs hold nothing but their drivers.
        Override to drop VI-level state that would be wrong after a
        reconnect — a cached "armed" flag, a ramp generator mid-flight.
        NEVER send a safe-off or configuration command here: disconnecting
        must leave the instrument doing exactly what it was doing (rule 2 of
        the connection-lifecycle standard). That is ``standby()``'s job, and
        the operator chooses whether to press it first.

        Resets ``lifecycle_state()`` to ``"idle"`` (the lifecycle-state
        standard): I2AS has stopped tracking what this instrument does,
        so it must not keep claiming the last thing it asked for — and the
        VI a later ``Station.connect_instrument()`` builds is a fresh object
        that starts at ``"idle"`` anyway. A subclass that overrides this
        gets the same reset from ``__init_subclass__``'s wrap
        (``_make_lifecycle_wrapper``).
        """
        self._lifecycle_state = LifecycleState.IDLE.value

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Poll every @monitored method and return ``{method_name: value}``.

        The VI's half of the monitor cycle, and the ONE place this VI reads
        its instrument on a tick. The result is also kept as the cache
        ``last_monitored()`` serves, so a pure read — a
        ``control_param_specs()`` override that wants the instrument's
        current setting as a widget default — never has to poll the bus a
        second time (see ``control_param_specs()``'s purity rule).

        Returns:
            ``{monitored_method_name: value}`` for every @monitored method.

        Raises:
            I2ASCommunicationError: Propagated from a failing driver
                read; the cache then keeps the previous poll's values, which
                is what makes them "last known" rather than "current".
        """
        from i2as.core.decorators import get_monitored_methods

        state: dict = {}
        for method_name in get_monitored_methods(self):
            state[method_name] = getattr(self, method_name)()
        self._monitored_cache = dict(state)
        # The lifecycle-state standard's hardware-correction rule (see
        # ``observe_lifecycle_state()``): last, so an override answers from
        # the values this very poll just cached rather than reading again.
        self._refresh_lifecycle_state()
        return state

    def last_monitored(self, name: str, default: Any = None) -> Any:
        """Return one @monitored value as of the last successful poll.

        The pure-read counterpart of calling the monitored method itself:
        it answers from the monitor cycle's cache and NEVER touches the
        bus, which is what lets ``control_param_specs()`` honour its purity
        rule while still defaulting a widget to the instrument's current
        setting.

        Args:
            name: The @monitored method's name.
            default: What to return when this VI has not been polled yet,
                or never reported that field.

        Returns:
            The cached value, or ``default``.
        """
        return getattr(self, "_monitored_cache", {}).get(name, default)

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def control_param_specs(self, method_name: str) -> dict[str, ParamSpec]:
        """Return the GUI ParamSpecs for one @control method, instance-aware.

        The default returns the specs declared on the decorator
        (``@control(params=...)``). A VI whose valid values only exist after
        construction — e.g. a switch whose routes come from the config —
        overrides this to inject dynamic ``choices``, and the GUI consults
        this hook instead of the raw decorator metadata. Presentation only:
        enforcement stays with ``control_limits`` and the method's own checks.

        **Purity rule.** This method is a PURE READ of config and of cached
        state: an override may read ``self._init_params``-derived attributes
        and ``last_monitored()``, and must send NO command to any driver.
        Two reasons, and either alone would be enough. It is called to
        DESCRIBE the instrument — by the front panel every time it renders,
        and by ``Station.station_info()`` to build the station declaration
        snapshot — so a bus read here puts instrument traffic on paths that
        are meant to describe, not operate, including one that must work
        for an instrument that is offline. And the Orchestrator is the sole
        writer to hardware (see ``core/orchestrator.py``): a read issued
        from a describe path is outside the single tick loop that
        serialises bus access. A VI that wants a widget defaulted to the
        instrument's current setting reads it from the monitor cycle's
        cache with ``last_monitored(name, fallback)`` — the value is at most
        one tick old, and the fallback covers the not-yet-polled case.
        ``tests/test_conformance.py`` builds the whole station declaration
        against spied drivers and fails on any call this path makes.

        Args:
            method_name: The @control method name.

        Returns:
            ``{param_name: ParamSpec}``; ``{}`` when the control declared none.
        """
        from i2as.core.decorators import get_control_specs

        return get_control_specs(getattr(self, method_name))

    def control_limit_bounds(self) -> dict[str, dict[str, tuple[float | None, float | None]]]:
        """Return the configured bounds of every declared control limit.

        The read side of the control-validation standard (see the class
        docstring), and the counterpart of ``merged_safety_flags()``: it
        resolves this class's ``control_limits`` declaration — method name
        -> parameter name -> limit name — against the ``self._limits``
        bounds this VI's ``__init__`` populated from the config, so a caller
        that has to REPORT the limits (the station declaration snapshot,
        an operator-facing panel) never reaches into the private mapping
        itself.

        Returns:
            ``{method_name: {param_name: (lo, hi)}}``, ``None`` on either
            side meaning unbounded there. A declared limit whose value the
            config never supplied reports ``(None, None)`` rather than
            raising — enforcement is where that violation is caught (with
            ``I2ASConfigError``, at call time), and it is checked in CI
            by ``tests/test_conformance.py``.
        """
        bounds: dict[str, dict[str, tuple[float | None, float | None]]] = {}
        limits = getattr(self, "_limits", {})
        for method_name, param_map in type(self).control_limits.items():
            bounds[method_name] = {
                param_name: tuple(limits.get(limit_name, (None, None)))  # type: ignore[misc]
                for param_name, limit_name in param_map.items()
            }
        return bounds

    @classmethod
    def merged_safety_flags(cls) -> dict[str, str]:
        """Return this class's ``safety_flags`` manifest merged over the MRO.

        The read side of the safety-flag manifest standard (see the class
        docstring's "Safety-flag manifest standard"): walks ``cls.__mro__``
        from the most-base class to the most-derived, collecting each
        class's OWN ``safety_flags`` dict (read via ``vars()`` so a class
        that declares nothing contributes nothing, and inheritance is never
        double-counted) into one merged mapping. A more-derived class's
        entry for the same flag simply overwrites its base's — the *lack*
        of contradiction is a declarative invariant checked by
        ``tests/test_conformance.py``, not enforced here, exactly like
        ``control_limits`` is declared without an import-time check.

        Returns:
            ``{flag_name: severity}`` unioned across every class in the
            MRO, severity one of ``"advisory"`` | ``"hold"`` | ``"critical"``.
        """
        merged: dict[str, str] = {}
        for klass in reversed(cls.__mro__):
            merged.update(vars(klass).get("safety_flags") or {})
        return merged

    def safety_concerns(self) -> set[str]:
        """Return the safety flags this VI's operation depends on.

        The consumer half of the System-Condition standard's safety-hold
        mechanism (GLOSSARY.md's **Safety hold**; see ``Station.
        update_conditions()``): a VI
        declares which flags — named exactly like ``evaluate_safety()``'s
        keys, e.g. ``"coolant_low"`` — must NOT be tripped for
        this VI to operate safely. When any of them trips (on ANY VI's
        ``evaluate_safety()``, not necessarily this one's own), the Station
        records a hold against this VI, and the Orchestrator refuses manual
        control of it and fails any run that claims it, without touching a
        VI whose ``safety_concerns()`` does not overlap.

        Concerns are for HOLD-severity flags only (see the "Safety-flag
        manifest standard"): a VI names the flags whose onset should stand
        IT down, and every flag it can meaningfully name has severity
        ``"hold"`` in the producing VI's merged ``safety_flags`` manifest.
        Naming a ``"critical"`` flag here would be meaningless — a tripped
        critical flag already forces station-wide EMERGENCY, stopping this
        VI (and every other) regardless of any per-VI concern declaration.

        This is a static declaration of an invariant, not a query of current
        state — a VI that needs a coolant always depends on it, independent
        of today's reading. The default (empty set) means this VI is never
        held by any safety flag.

        Returns:
            Set of safety flag names this VI depends on, e.g.
            ``{"coolant_low"}``.
        """
        return set()

    def evaluate_safety(self, state: dict) -> dict[str, bool]:
        """Judge this VI's own polled state for safety conditions.

        Called by ``Station.check_safety()`` every monitor tick with the
        fragment of the snapshot belonging to this VI. Must NOT poll
        hardware — decide from *state* (and internal buffers filled during
        the poll). A flag returned True here is dispatched by the
        System-Condition standard (``core/conditions.py``): ``Station.
        update_conditions()`` builds the flag's own ``Condition`` from its
        declared severity in ``safety_flags`` — critical station-wide by
        construction, or hold-severity and scoped to every VI whose
        ``safety_concerns()`` names it.

        Args:
            state: This VI's slice of the get_state() snapshot,
                ``{monitored_method_name: value, ...}``.

        Returns:
            ``{flag_name: bool}`` — e.g. ``{"quench": True}``. Empty dict
            (the default) means this VI declares no safety conditions.
        """
        _ = state
        return {}


# ── Typed category bases ──────────────────────────────────────────────────────
# Directive §"Common Mistakes": all category bases live in base.py.

class MagnetBase(BaseVirtualInstrument):
    """Base class for all magnet-type VIs."""
    vi_type: str = "magnet"
    # Human label + unit for this VI's ramp setpoint, read centrally (via the
    # Station) to render concise procedure status lines such as
    # "Ramping field to -1 T". Declared once per instrument category so every
    # magnet VI, procedure, and config inherits it — no per-procedure code.
    setpoint_label: str = "field"
    setpoint_unit: str = "T"
    display_label: str = "magnet"

    # A quench (reported by a concrete magnet's evaluate_safety()) is
    # critical severity — see the "Safety-flag manifest standard". Declared
    # here, on the category all quench-capable VIs share, so a new magnet VI
    # inherits the correct classification the moment it reports "quench".
    # Critical severity is station-wide scope by construction (the
    # System-Condition standard, core/conditions.py): a quench stops every
    # instrument via EMERGENCY, so no VI — including this one — declares
    # "quench" in safety_concerns(); a per-VI hold would be meaningless
    # once EMERGENCY has already stopped everything.
    safety_flags: ClassVar[dict[str, str]] = {"quench": "critical"}


class TemperatureControllerBase(BaseVirtualInstrument):
    """Base class for all temperature-controller VIs."""
    vi_type: str = "temperature"
    setpoint_label: str = "temperature"
    setpoint_unit: str = "K"
    display_label: str = "temperature"


class StageBase(BaseVirtualInstrument):
    """Base class for all sample-positioning (stage) VIs.

    One VI positions ONE axis: the setpoint a stage VI ramps is a single
    position in metres, so a two-axis stage is two VIs of this category over
    one driver (``axis`` names which), exactly as a vector magnet is one
    magnet VI per coil. That is what keeps a stage inside the same scalar
    setpoint, ``Target`` and envelope standards every other rampable VI
    obeys — ``Station.stage_vi_names()`` is the role accessor procedures
    discover them by.
    """
    vi_type: str = "stage"
    setpoint_label: str = "position"
    setpoint_unit: str = "m"
    display_label: str = "stage"
    #: Which axis this VI positions (``"x"``, ``"y"``, …); a concrete VI
    #: sets it from its config so a procedure can tell the axes apart.
    axis: str = ""


class MeasurementInstrumentBase(BaseVirtualInstrument):
    """Base class and self-describing standard for all measurement-method VIs.

    A *measurement method* (see GLOSSARY.md) is a measurement VI that describes
    its own GUI knobs and output shape and implements one uniform lifecycle, so
    a generic procedure can run any of them without knowing which instrument or
    protocol is behind it. Every concrete measurement VI MUST honour this
    standard; ``tests/test_conformance.py`` enforces it the moment the file
    exists.

    Self-description (class attributes)
    -----------------------------------
    * ``selector_label: ClassVar[str]`` — the SHORT human name shown in the GUI
      method-selection drop-down (e.g. "DC (6221 + 2182A)"). Optional:
      when empty, the drop-down falls back to ``display_label``. Keep it terse —
      the combo's width tracks its longest label. This is distinct from
      ``display_label``, which is the longer status-line label ("DC
      resistance") and is unchanged by this attribute. ``tests/test_conformance``
      checks it is a ``str``.
    * ``measurement_parameters: ClassVar[dict[str, ParamSpec]]`` — the VI's
      GUI-facing knobs, one ``ParamSpec`` per parameter. This is the single
      owner of those specs (procedures will stop duplicating them in a later
      wave). Must be non-empty on a concrete VI.
    * ``measurement_data_keys: ClassVar[list[str]]`` — the RAW-SAMPLE array
      names ``take_reading()`` returns, each named ``"{quantity}_array"``
      (e.g. ``["voltage_V_array", "current_A_array"]``). Must be non-empty on
      a concrete VI. Build this — and the companion
      ``measurement_scalar_columns`` entries below — with
      ``quantity_columns()`` rather than hand-writing the suffixes.
    * ``measurement_scalar_columns: ClassVar[dict[str, str]]`` — per-point
      *scalar* columns, mapping name → dtype ("float" or "int"). The
      mean/error/array convention: for every array-valued quantity
      ``"{quantity}_array"`` in ``measurement_data_keys``, this MUST also
      carry ``"{quantity}"`` (the mean — the value the GUI plots) and
      ``"{quantity}_error"`` (the standard error of the mean), both dtype
      "float" — exactly what ``quantity_columns()`` derives. It may also
      carry VI-specific extras unrelated to any single array; a VI whose
      ``take_reading()`` can deliver fewer raw samples than ``data_arrays()``
      declared MUST carry ``n_valid`` (dtype "int") — see "Under-delivery:
      the ``n_valid`` standard" below. ``tests/test_conformance.py`` enforces
      the mean/error pairing automatically for every ``_array`` key.

    Uniform lifecycle (methods)
    ---------------------------
    * ``data_arrays(params) -> dict[str, int]`` — declared array name (the
      ``"{quantity}_array"`` keys) → its per-point length, computed from the
      SAME ``params`` mapping ``initiate_measurement()`` will receive. Lets a
      procedure size its HDF5 layout before arming the hardware. Base raises
      ``NotImplementedError``.
    * ``initiate_measurement(**params) -> None`` — arm / configure the
      hardware. Accepts the ``measurement_parameters`` keys as keyword
      arguments, each with a default. Concrete VIs keep the ``@control``
      decoration where the VI exposes arming to the GUI (with ``panel=False``
      so arming lives in the front panel, not the compact card). Deliberately
      NOT named ``initiate``: the plain instrument-lifecycle ``initiate()``
      (below) is harmless by construction, so a bulk Initiate-All can never
      start a source current.
    * ``initiate() -> None`` — the instrument-lifecycle verb every VI has,
      overridden here as a CONNECTION CHECK: pings the drivers and raises
      ``I2ASCommunicationError`` when unreachable. Never arms, never
      sources.
    * ``take_reading() -> dict[str, list[float] | float]`` — take ONE
      datapoint. Takes NO arguments: everything it needs was fixed at
      ``initiate_measurement()``. For every quantity it declares, it MUST
      return all three of the mean/error/array triple: the raw sample array
      (``"{quantity}_array"``, NaN-padded to the length ``data_arrays(params)``
      declared for the same ``params`` — always), the mean (``"{quantity}"``),
      and the standard error of the mean (``"{quantity}_error"``), computed
      over the VALID samples with ``self.mean_and_sem(...)``. It must also
      return every other ``measurement_scalar_columns`` key (e.g. ``n_valid``
      — see "Under-delivery: the ``n_valid`` standard" below). A VI whose
      instrument may return fewer points than requested still pads the array
      to the declared length with ``float("nan")``, but computes the
      mean/error/``n_valid`` over the samples actually delivered, BEFORE that
      padding is applied — never by filtering NaN back out of the padded
      array. This fixed-shape guarantee is the contract that prevents HDF5
      layout mismatches mid-run.
    * ``standby() -> None`` — put the instrument in a safe-off idle state.

    Under-delivery: the ``n_valid`` standard
    -----------------------------------------
    A VI whose ``take_reading()`` can deliver fewer raw samples than
    ``data_arrays()`` declared for a quantity MUST report an ``n_valid``
    scalar column (dtype ``"int"``, added to ``measurement_scalar_columns``)
    — the number of samples the mean and SEM were actually computed over.

    * The returned array preserves the delivered rows verbatim (including
      any instrument-emitted NaN, e.g. a ratiometric divide-by-zero),
      padded with ``float("nan")`` out to the declared length.
    * The mean, SEM, and ``n_valid`` are computed over the delivered samples
      BEFORE padding — never by filtering NaN out of the padded array, which
      would conflate I2AS's own padding with a NaN the instrument itself
      emitted.
    * Row selection when more samples arrived than requested (first-*n* vs
      last-*n*) is per-instrument physics — e.g. an instrument with a
      free-running buffer and a settling transient prefers the last *n*
      delivered rows — and is documented on that VI's own ``take_reading()``,
      not standardised here.

    A VI whose instrument always delivers exactly the requested sample count
    has nothing to report and may omit ``n_valid`` entirely.

    Raw diagnostic blocks
    ----------------------
    Some instruments return far more raw data per reading than any single
    physical quantity — e.g. an engine that reports dozens of raw
    channels (voltages, ranges, setpoints, lock quality, …) alongside the
    one or two the operator actually derives a result from. The mean/error/
    array convention above cannot express this: it hard-requires every
    ``"{quantity}_array"`` key to pair with a same-quantity mean/error, which
    makes no sense for a block mixing many different physical units with no
    single mean to report. A VI with this need declares a SECOND, orthogonal
    self-description instead of forcing the block through the convention
    above:

    * ``measurement_raw_blocks: ClassVar[dict[str, list[str]]]`` — block
      name -> ordered channel-label list. The label list's length fixes the
      block's channel axis; it is never companioned by a mean/error pair and
      never excluded from HDF5 the way an unpaired array would need to be.
      Empty (the default) means the VI has no raw block.
    * ``raw_block_row_counts(params) -> dict[str, int]`` — declared block
      name -> row count (readings) for the SAME ``params`` mapping
      ``initiate_measurement()`` will receive, mirroring ``data_arrays()``'s
      per-instance, params-dependent role. Base returns ``{}``.
    * ``take_reading()`` returns a block's value as a nested ``rows x
      channels`` list (row order matching the declared row count, column
      order matching the declared label list), keyed by the block name,
      alongside the mean/error/array triples and scalar columns it already
      returns. A VI whose instrument may under-deliver rows pads the ROW
      axis with ``float("nan")`` the same way an array quantity does; the
      channel axis is fixed and never padded.

    A raw block's own ``rows x channels`` matrix is a diagnostic/provenance
    record, not itself a plotted quantity: it is deliberately excluded from
    ``measurement_data_keys`` / ``measurement_scalar_columns``, so the block
    as a whole never appears in the GUI's plot axis dropdowns, exactly like
    an ``_array`` column today. Its declared CHANNELS are a different
    matter: ``SweepMeasureProcedure`` automatically derives one scalar
    column per channel — the row axis reduced by a NaN-safe mean, the same
    "N readings at one measurement point" treatment ``mean_and_sem`` already
    gives a quantity's own array column — so every channel is independently
    plottable and carries the real ``(n_loop1, n_loop2)`` loop grid, with
    zero VI-side code. This needs no declaration beyond
    ``measurement_raw_blocks`` itself; see
    ``SweepMeasureProcedure._raw_block_channel_columns``/``measure()``.

    A block's HDF5 storage is also the one exception to the reading loop's
    "every measurement column carries a real ``(n_loop1, n_loop2)`` axis"
    rule (see ``reading_setters`` below and ``DataSchema.measurement_blocks``):
    with no reading loop configured for the run, a block is stored bare —
    ``(rows, cols)`` — rather than ``(1, 1, rows, cols)``. This is handled
    entirely by ``SweepMeasureProcedure``/``DataSchema``/``DataManager``; a
    VI's own ``take_reading()`` always returns the block as a flat ``rows x
    channels`` list regardless of any loop, exactly as described above.

    **Image blocks** — the **image-block standard**, the raw block's sibling
    for a frame. A camera frame has a raw block's ``(rows, cols)`` shape but
    no channel per column: every element is one pixel in one unit, and no
    per-column scalar makes sense. A VI with this need declares:

    * ``measurement_image_blocks: ClassVar[dict[str, ImageBlock]]`` — block
      name -> ``ImageBlock(height_px, width_px, unit, description)``
      (``core/plan.py``). The declared pixel dimensions fix the frame's
      shape for every reading; ``raw_block_row_counts()`` plays no part.
      Empty (the default) means the VI takes no frames.
    * ``take_reading()`` returns the frame under the block name as a
      ``(height_px, width_px)`` numpy array (or an equivalently nested
      list), alongside whatever scalars and arrays it already returns.

    An image block travels the raw block's dataset path (``DataSchema.
    measurement_blocks``, the same loop-axis rule, the same NaN fill) but is
    written with ``block_kind = "image"`` and ``unit``/``description``
    attributes and no ``channel_names``, and ``data_reader`` reports it
    with role ``image`` and serves one frame through ``read_image()``. No
    scalar column is ever derived from a frame — the live plot and the
    generic sweep recipe use the VI's scalar columns, the image-stack
    recipe reads the frames. Conformance requires each declared block's
    ``height_px``/``width_px`` to match what the sim's ``take_reading()``
    actually returns.

    Externally configured instruments
    ----------------------------------
    Some instruments expose far more configuration surface than a VI wraps
    (analysis modes, pulse trains, reference muxing, preamp modes, …). A VI
    may support an operator configuring the instrument with the vendor's own
    tool and letting I2AS run only the measurement — arm the data path,
    trigger, read, and save a fixed-shape data block — without touching that
    configuration. This is the ``configured_externally`` standard, motivated
    by single-client instrument firmware where the vendor tool and I2AS
    are mutually exclusive at the instrument, and it applies to every
    externally configured VI, not just the one that first needed it.

    ``__init__`` reads the optional init param ``configured_externally: bool``
    (default ``False``) from ``init_params`` and stores it as
    ``self._configured_externally`` — config-driven, per-VI, via
    ``devices.yaml``'s ``init_params``, with zero per-subclass boilerplate.
    Omitted or ``False`` leaves every existing VI's behavior unchanged.

    When ``self._configured_externally`` is true:

    * ``initiate_measurement()`` MUST NOT write any excitation, analysis, or
      routing parameter to the instrument — the external tool owns them. It
      MUST still: (a) verify connectivity with a TRUE ROUND TRIP — a query
      that fails loudly on a dead or externally-held channel, never a
      liveness check that can succeed vacuously; (b) arm the data path
      (buffers, channel/format selection, anything ``take_reading()``'s
      decode depends on); (c) read back from the instrument any value its
      own timing or decoding depends on (e.g. an externally set averaging
      time that determines the settle sleep); and (d) set the internal state
      ``take_reading()`` / ``data_arrays()`` require. Inert
      ``measurement_parameters`` MUST still be accepted (procedures pass
      them regardless of mode); log one INFO listing the ignored parameters.
    * ``standby()`` MUST NOT overwrite externally-owned source state (e.g.
      must not zero a current the external tool set). It resets only
      I2AS's own internal arming state and RELEASES the hardware
      resource (the driver's ``close()``).

    A VI that captures a provenance snapshot of the instrument's
    arming-time configuration SHOULD expose it as ``self.last_settings_
    snapshot`` (a plain ``dict``, ``None`` until a run has armed in
    external mode): ``SweepMeasureProcedure.measure()`` duck-types this
    attribute and records it into the run's HDF5 ``/metadata`` the first
    time it sees a non-``None`` value (see ``DataManager.
    record_settings_snapshot()``), with no per-VI plumbing required.

    A VI supporting external configuration MUST declare, via the class
    attribute ``externally_owned_parameters: ClassVar[frozenset[str]]``,
    the names of its ``measurement_parameters`` the external tool owns in
    that mode (excitation/analysis/routing — the ones ``initiate_measurement()``
    must not write). Parameters NOT listed (data-path parameters, e.g. a
    tensor-component selector or a readings-per-point count — anything that
    only picks how I2AS decodes/shapes the data, writing nothing to the
    instrument) stay operator-controlled in every mode. This is the single
    source of truth two other surfaces derive from automatically: the
    ``active_measurement_parameters`` property (what the procedure form
    renders) and ``reading_parameters`` (what the reading-loop registry
    offers) both subtract it when ``configured_externally`` is true. The
    empty default changes nothing for a VI that does not support external
    configuration. ``tests/test_conformance.py`` checks every declared name
    is a real ``measurement_parameters`` key.

    **Detached-idle lifecycle**: an externally configured VI is the
    motivating case for ``BaseVirtualInstrument``'s detach-when-idle
    declaration (see its class docstring — ``detach_when_idle``,
    ``is_attached()``, ``_attach()``/``_detach()``); it holds its
    instrument connection only from ``initiate_measurement()`` to
    ``standby()``:

    * Declare it by overriding the ``detach_when_idle`` property, e.g.
      ``return self._configured_externally``.
    * Born detached: ``__init__`` calls ``self._detach()`` before
      returning, so starting I2AS while the vendor tool is open builds
      cleanly.
    * ``ping()`` — and so the base's ``initiate()``, which calls it —
      verify-and-release automatically: reattach, a TRUE ROUND TRIP, then
      release, returning a clean failure verdict rather than raising when
      the instrument is currently held by the external tool. No VI-specific
      override needed.
    * ``initiate_measurement()`` (re)acquires the connection (via
      ``self._attach()``) for the measurement window — still the concrete
      VI's own job, since arming itself is VI-specific.
    * ``standby()`` releases it again automatically, via the base's
      declared wrap. Every run path already ends in ``standby()``, so the
      instrument frees itself and the operator may attach the vendor tool
      at any time between runs.
    * Never reconnect opportunistically in the background (e.g. from a
      monitored poll): a wrongly-timed connect can fail silently against the
      external tool's session, not loudly.

    The reading loop (optional): ``reading_setters``
    ------------------------------------------------
    A VI may declare that some of its measurement parameters can be CHANGED
    BETWEEN READINGS without re-arming, via the class attribute::

        reading_setters: ClassVar[dict[str, str]] = {"current_A": "set_source_current"}

    mapping a ``measurement_parameters`` name to the cheap setter method that
    reprograms just that quantity. Declaring an entry is all a VI does: the
    generic sweep procedure automatically renders a "Reading loop" form group
    where the user picks up to TWO such parameters (loop1/loop2) and enters
    each one's value list, and then, at every sweep point, dispatches the
    setter (as a ``Command`` via the Station) before each value's
    ``take_reading()``. Every measurement column gets a real ``(n_loop1,
    n_loop2)`` array axis in HDF5 — index *i* of an axis is that loop slot's
    *i*-th dispatched value, recorded in the run's HDF5 metadata as
    ``procedure_params["loop1_values"]`` / ``["loop2_values"]`` — rather than
    being encoded into the column name.

    Contract (machine-enforced by ``tests/test_conformance.py``):

    * Every key names an existing ``measurement_parameters`` entry.
    * Every value names a real method of the VI whose signature accepts the
      parameter under its own name (e.g. ``set_source_current(current_A=...)``).
    * The setter reconfigures the reading, never its shape: after any setter
      call, ``take_reading()`` still returns exactly ``measurement_data_keys``
      with the lengths ``data_arrays(params)`` declared.
    * Setter calls obey the same validation discipline as any hardware write —
      keep ``@control`` (and ``control_limits``) on setters whose values must
      be bounded.

    Adding a new measurement method: subclass this base (or ``DCMeasurementBase``
    for a DC-resistance method), declare the three class attributes, and
    implement ``data_arrays`` / ``initiate_measurement`` / ``take_reading`` /
    ``standby`` (plus ``reading_setters`` entries for any parameter the
    reading loop may vary per point).
    """

    vi_type: str = "measurement"
    # Human label for status lines like "Arming DC resistance measurement".
    display_label: str = "measurement"
    # SHORT human name for the GUI method-selection drop-down; falls back to
    # display_label when empty (see the "Self-description" section above).
    selector_label: ClassVar[str] = ""

    # Self-description — overridden (non-empty) by every concrete VI.
    measurement_parameters: ClassVar[dict[str, ParamSpec]] = {}
    measurement_data_keys: ClassVar[list[str]] = []
    measurement_scalar_columns: ClassVar[dict[str, str]] = {}
    # Raw diagnostic blocks (see the class docstring's "Raw diagnostic
    # blocks" section): block name -> ordered channel-label list. Empty
    # (the default) means no raw block.
    measurement_raw_blocks: ClassVar[dict[str, list[str]]] = {}
    # Image blocks (see the class docstring's "Image blocks" section, the
    # image-block standard): block name -> ImageBlock declaration. Empty
    # (the default) means the VI takes no frames.
    measurement_image_blocks: ClassVar[dict[str, ImageBlock]] = {}
    # Reading-loop declaration: parameter name -> per-reading setter method.
    # Empty (the default) means no parameter of this VI can be looped.
    reading_setters: ClassVar[dict[str, str]] = {}
    # The externally-configured standard's self-description (see the class
    # docstring's "Externally configured instruments" section): the
    # measurement_parameters names the external tool owns when
    # configured_externally is true. Empty (the default) changes nothing for
    # a VI that does not support external configuration.
    externally_owned_parameters: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Install ``measurement_parameters`` as the declared control specs.

        The one-declaration rule of the measurement-method standard: a
        measurement VI already owns rich ``ParamSpec``s for its knobs in
        ``measurement_parameters`` (units, bounds, drop-down ``choices``), and
        the procedure form renders them. This hook hands the SAME specs to the
        GUI's control renderer and the capability manifest, so the arming
        control and the reading-loop setters show the same widgets and the
        same units instead of bare text boxes — no per-VI duplication, and no
        second place to keep in step.

        Two controls are covered, and only when the subclass defines them
        itself and declares no ``params=`` of its own (an explicit
        declaration always wins):

        * ``initiate_measurement`` — the whole ``measurement_parameters``
          mapping, which its signature must accept exactly.
        * every ``reading_setters`` setter — the single spec of the parameter
          it sets, when that is the setter's only parameter.

        Args:
            **kwargs: Forwarded to ``BaseVirtualInstrument.__init_subclass__``.

        Raises:
            ValueError: If ``initiate_measurement`` is a ``@control`` whose
                parameters are not exactly the ``measurement_parameters``
                keys, which the measurement-method standard requires.
        """
        cls._install_measurement_control_specs()
        super().__init_subclass__(**kwargs)

    @classmethod
    def _install_measurement_control_specs(cls) -> None:
        """Copy ``measurement_parameters`` onto the directly defined controls.

        Runs BEFORE ``BaseVirtualInstrument.__init_subclass__`` wraps the
        methods, so the specs it writes onto the decorator's function are
        carried onto the wrapper (and type-checked there) exactly like a
        hand-written ``params=``. Only ``vars(cls)`` is touched: an inherited
        method is left alone, since its specs were installed when its own
        class was created and mutating it would change the parent for
        everyone.

        Raises:
            ValueError: If ``initiate_measurement``'s parameters are not
                exactly the ``measurement_parameters`` keys.
        """
        params = cls.measurement_parameters
        if not params:
            return

        arming = vars(cls).get("initiate_measurement")
        if callable(arming) and getattr(arming, "_is_control", False):
            sig_names = {
                name
                for name in inspect.signature(arming).parameters
                if name != "self"
            }
            if sig_names != set(params):
                raise ValueError(
                    f"{cls.__name__}.initiate_measurement() takes "
                    f"{sorted(sig_names)} but measurement_parameters declares "
                    f"{sorted(params)} — the measurement-method standard "
                    f"requires them to match exactly."
                )
            if not getattr(arming, "_control_specs", None):
                arming._control_specs = dict(params)

        for param_name, setter_name in cls.reading_setters.items():
            setter = vars(cls).get(setter_name)
            if not callable(setter) or not getattr(setter, "_is_control", False):
                continue
            if getattr(setter, "_control_specs", None):
                continue
            if param_name not in params:
                continue
            sig_names = {
                name
                for name in inspect.signature(setter).parameters
                if name != "self"
            }
            if sig_names == {param_name}:
                setter._control_specs = {param_name: params[param_name]}

    def __init__(self, drivers: dict[str, object], **init_params: Any) -> None:
        """Initialise the measurement VI.

        Args:
            drivers: Mapping of role -> driver instance.
            **init_params: Additional parameters from YAML config. Recognises
                the externally-configured standard's ``configured_externally``
                (bool, default ``False``) — see the class docstring's
                "Externally configured instruments" section — in addition to
                whatever a concrete VI's own ``__init__`` reads.
        """
        super().__init__(drivers, **init_params)
        # The externally-configured standard's mode flag. Config-driven, no
        # per-subclass boilerplate: a concrete VI's __init__ chains to this
        # one (via super().__init__(drivers, **init_params)) and never needs
        # to read this key itself.
        self._configured_externally: bool = bool(
            init_params.get("configured_externally", False)
        )

    @property
    def configured_externally(self) -> bool:
        """Whether this VI is in the externally-configured standard's external mode.

        Public, read-only accessor for ``self._configured_externally`` (set
        from the ``configured_externally`` init param — see the class
        docstring's "Externally configured instruments" section), so
        procedures and the GUI can read the mode without touching the
        private attribute.

        Returns:
            ``True`` when the external tool owns excitation/analysis/routing
            configuration for this instrument.
        """
        return self._configured_externally

    @property
    def active_measurement_parameters(self) -> dict[str, ParamSpec]:
        """Return the ``measurement_parameters`` a form should actually render.

        This is what parameter FORMS render — not the call contract of
        ``initiate_measurement()``, which remains ``measurement_parameters``
        in full: a hidden parameter still falls back to its declared default
        and is simply ignored by the external branch (see the class
        docstring's "Externally configured instruments" section).

        Returns:
            ``measurement_parameters`` unchanged when not externally
            configured; otherwise ``measurement_parameters`` minus
            ``externally_owned_parameters``.
        """
        if not self._configured_externally:
            return dict(self.measurement_parameters)
        return {
            name: spec
            for name, spec in self.measurement_parameters.items()
            if name not in self.externally_owned_parameters
        }

    @property
    def reading_parameters(self) -> dict[str, ParamSpec]:
        """Return the ``ParamSpec`` of every loopable measurement parameter.

        A measurement VI's loopable parameters are, by definition, a subset of
        its ``measurement_parameters``, so this reads their specs from there.
        When ``configured_externally`` is true, an externally-owned name is
        excluded here too: a loopable-but-externally-owned parameter would
        otherwise dispatch setter WRITES to the instrument mid-run in
        external mode, clobbering the external tool's configuration. Excluding
        it here makes ``SweepMeasureProcedure._loopable_registry`` skip it
        automatically — the registry already drops any entry with no spec.

        Returns:
            ``{name: measurement_parameters[name]}`` for every
            ``reading_setters`` key, minus any externally-owned name while
            ``configured_externally`` is true.
        """
        owned = self.externally_owned_parameters if self._configured_externally else frozenset()
        return {
            name: self.measurement_parameters[name]
            for name in self.reading_setters
            if name not in owned
        }

    @classmethod
    def quantity_columns(cls, *names: str) -> tuple[list[str], dict[str, str]]:
        """Derive the array/mean/error HDF5 key triple for each quantity name.

        The mean/error/array convention (see the class docstring) requires
        every array-valued quantity a VI reports — e.g. ``"voltage_V"`` — to
        surface as three columns: the raw samples under ``"{name}_array"``,
        the mean under the bare ``"{name}"`` (what the GUI plots), and the
        standard error of the mean under ``"{name}_error"``. This is the one
        place that naming convention is spelled out; every concrete
        measurement VI derives its ``measurement_data_keys`` /
        ``measurement_scalar_columns`` from it instead of hand-writing the
        suffixes.

        Args:
            *names: Base quantity names, e.g. ``"voltage_V", "current_A"``.

        Returns:
            ``(array_keys, scalar_columns)`` — ``array_keys`` is
            ``["{name}_array", ...]`` (for ``measurement_data_keys``);
            ``scalar_columns`` is ``{"{name}": "float", "{name}_error":
            "float", ...}`` (for ``measurement_scalar_columns``).

        Raises:
            I2ASConfigError: If a base name already ends in ``"_array"``
                or ``"_error"`` — ambiguous with the derived suffixes.
        """
        array_keys: list[str] = []
        scalar_columns: dict[str, str] = {}
        for name in names:
            if name.endswith("_array") or name.endswith("_error"):
                raise I2ASConfigError(
                    f"quantity_columns: {name!r} already ends in '_array' or "
                    f"'_error' — base quantity names must not collide with "
                    f"the derived mean/error/array suffixes."
                )
            array_keys.append(f"{name}_array")
            scalar_columns[name] = "float"
            scalar_columns[f"{name}_error"] = "float"
        return array_keys, scalar_columns

    @staticmethod
    def mean_and_sem(samples: Sequence[float]) -> tuple[float, float]:
        """Mean and standard error of the mean over valid (non-NaN) samples.

        The shared statistics helper behind the mean/error/array convention:
        every measurement VI calls this on a quantity's valid raw samples to
        compute the ``"{quantity}"`` (mean) and ``"{quantity}_error"`` (SEM)
        columns it returns from ``take_reading()``.

        Args:
            samples: Raw per-reading values, already filtered to valid
                (non-NaN) entries — pass none of the NaN padding.

        Returns:
            ``(mean, sem)``. A single sample has no spread estimate, so
            ``sem`` is ``0.0``; zero samples return ``(nan, nan)``.
        """
        n = len(samples)
        if n == 0:
            return float("nan"), float("nan")
        mean = statistics.fmean(samples)
        if n == 1:
            return mean, 0.0
        return mean, statistics.stdev(samples) / math.sqrt(n)

    def data_arrays(self, params: Mapping[str, Any]) -> dict[str, int]:
        """Return declared array name → per-point length for these *params*.

        Args:
            params: The same parameter mapping ``initiate_measurement()`` will
                be called with (the ``measurement_parameters`` keys).

        Returns:
            ``{array_name: length}`` for every name in ``measurement_data_keys``.

        Raises:
            NotImplementedError: If not overridden by a concrete VI.
        """
        raise NotImplementedError

    def raw_block_row_counts(self, params: Mapping[str, Any]) -> dict[str, int]:
        """Return declared raw-block name → row count for these *params*.

        See the class docstring's "Raw diagnostic blocks" section. Mirrors
        ``data_arrays()``'s per-instance, params-dependent role for blocks.

        Args:
            params: The same parameter mapping ``initiate_measurement()`` will
                be called with (the ``measurement_parameters`` keys).

        Returns:
            ``{block_name: rows}`` for every name in ``measurement_raw_blocks``.
            Base returns ``{}`` — a VI with no raw block need not override
            this.
        """
        return {}

    def initiate_measurement(self, **params: Any) -> None:
        """Arm / configure the hardware for taking readings.

        The ARMING half of the old overloaded ``initiate``: accepts the
        ``measurement_parameters`` keys as keyword arguments. Only procedures
        (and an explicit front-panel action) call this — never a bulk
        Initiate-All.

        Args:
            **params: The ``measurement_parameters`` keys, each defaulted.

        Raises:
            NotImplementedError: If not overridden by a concrete VI.
        """
        raise NotImplementedError

    def initiate(self) -> None:
        """Check the instrument connection — harmless by construction.

        Overrides the generic VI lifecycle verb so a bulk Initiate-All (or
        the card's lifecycle toggle) verifies communication without arming
        anything: no source current, no output, no configuration change.

        Raises:
            I2ASCommunicationError: If any driver fails to respond to an
                identity query (``ping()`` returned False).
        """
        if not self.ping():
            raise I2ASCommunicationError(
                f"{self.vi_name or type(self).__name__}: instrument did not "
                f"respond to an identity query — check the connection.",
                vi_name=self.vi_name,
            )

    def take_reading(self) -> dict[str, list[float]]:
        """Acquire one datapoint at the configuration fixed by ``initiate_measurement()``.

        Returns:
            A dict containing exactly ``measurement_data_keys`` (arrays sized as
            ``data_arrays`` declared) plus every ``measurement_scalar_columns``
            key.

        Raises:
            NotImplementedError: If not overridden by a concrete VI.
        """
        raise NotImplementedError


class DCMeasurementBase(MeasurementInstrumentBase):
    """Base class for DC resistance measurement methods (defers to the standard).

    Folds the shared DC-resistance self-description into one place so any
    two DC-resistance VIs — a separate source/voltmeter pair, a single SMU —
    are interchangeable via the YAML config alone. Both
    inherit the ``measurement_parameters`` / ``measurement_data_keys`` /
    ``data_arrays`` declared here and implement only ``initiate_measurement()``
    / ``take_reading()`` / ``standby()``.

    The full lifecycle contract is documented on ``MeasurementInstrumentBase``;
    this class adds nothing new, it only fixes the DC-resistance shape
    (``readings_per_point`` samples of ``voltage_V`` and ``current_A``). The
    ``initiate_measurement`` / ``take_reading`` stubs raise
    ``NotImplementedError`` so a missing implementation fails loudly at first
    use.
    """

    display_label: str = "DC resistance"

    # Control-validation standard (see BaseVirtualInstrument): the excitation
    # current a DC measurement pushes through the sample is the one @control
    # parameter here that can damage it, so it is bounded by the setup's own
    # ceiling. Subclasses that add a per-reading current setter MERGE into
    # this mapping rather than replacing it.
    control_limits = {
        "initiate_measurement": {"current_A": EXCITATION_CURRENT_LIMIT},
    }

    _ARRAY_KEYS, _SCALAR_COLUMNS = MeasurementInstrumentBase.quantity_columns(
        "voltage_V", "current_A"
    )
    measurement_data_keys: ClassVar[list[str]] = _ARRAY_KEYS
    measurement_scalar_columns: ClassVar[dict[str, str]] = _SCALAR_COLUMNS
    measurement_parameters: ClassVar[dict[str, ParamSpec]] = {
        "current_A": ParamSpec(
            type=float, default=1e-6, unit="A", description="DC source current"
        ),
        "compliance_A": ParamSpec(
            type=float,
            default=1e-3,
            unit="A",
            description="Current compliance on voltmeter",
        ),
        "voltmeter_range_V": ParamSpec(
            type=float,
            default=0.1,
            unit="V",
            description="Voltmeter full-scale range",
        ),
        "readings_per_point": ParamSpec(
            type=int,
            default=10,
            min=1,
            description="DC voltage readings averaged per point",
        ),
    }

    def __init__(self, drivers: dict[str, object], **init_params: Any) -> None:
        """Populate the excitation-current limit from the setup config.

        Args:
            drivers: The VI's driver mapping (see the concrete subclass).
            **init_params: Setup parameters; ``max_source_current_A`` bounds
                the sourced current symmetrically (current reversal is
                routine in DC resistance work, so the bound is
                ``±max_source_current_A``). Absent means unbounded — the
                same "missing key means no bound on that side" rule every
                other VI's limits follow. Every shipped config declares it;
                ``tests/test_conformance.py`` makes that binding.
        """
        super().__init__(drivers, **init_params)
        _populate_excitation_current_limit(self, init_params)

    def data_arrays(self, params: Mapping[str, Any]) -> dict[str, int]:
        """Return ``{"voltage_V_array": n, "current_A_array": n}``.

        n = ``readings_per_point``.

        Args:
            params: Parameter mapping containing ``readings_per_point``.

        Returns:
            Per-point length for each DC data array.
        """
        n = int(params["readings_per_point"])
        return {key: n for key in self.measurement_data_keys}

    def initiate_measurement(
        self,
        current_A: float = 1e-6,
        compliance_A: float = 1e-3,
        voltmeter_range_V: float = 0.1,
        readings_per_point: int = 10,
    ) -> None:
        """Arm the measurement hardware with fixed DC current, range and count.

        Args:
            current_A: DC source current in Amperes.
            compliance_A: Compliance / protection limit in Amperes.
            voltmeter_range_V: Full-scale voltage measurement range in Volts.
            readings_per_point: Number of voltage samples ``take_reading()``
                collects per datapoint.
        """
        raise NotImplementedError

    def take_reading(self) -> dict[str, list[float] | float]:
        """Acquire ``readings_per_point`` voltage samples at the fixed current.

        Returns:
            The mean/error/array triple for both quantities: ``voltage_V``,
            ``voltage_V_error``, ``voltage_V_array`` (length
            ``readings_per_point``, fixed at ``initiate_measurement()``), and
            the same three for ``current_A``.
        """
        raise NotImplementedError
