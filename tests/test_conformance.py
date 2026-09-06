# ---
# description: |
#   Auto-discovering conformance tests for the I2AS layer interfaces.
#   These tests iterate over the drivers, virtual_instruments, procedures, and
#   configs packages themselves, so any NEW module an agent adds is checked
#   automatically — no test needs to be written for it to be covered.
# last_updated: 2026-07-13
# ---

"""Interface-conformance tests: every driver, VI, procedure, and config obeys its contract.

These tests are the safety net for agentic development. They discover modules at
runtime (via pkgutil) instead of naming them, so adding a new driver, VI,
procedure, or config directory automatically brings it under test. If one of
these tests fails on your new module, fix the module to match the contract —
do not weaken the test.

Contracts enforced here (import boundaries are enforced separately by
import-linter, see pyproject.toml [tool.importlinter]):

* Drivers: one public class per module; ``__init__`` takes exactly one required
  argument (the VISA resource string); sim drivers construct without hardware;
  a real driver and its ``sim_``-twin expose identical public APIs.
* Virtual instruments: subclass BaseVirtualInstrument, set ``vi_type``, and use
  the mandated ``__init__(self, drivers, **init_params)`` signature.
* Measurement methods (MeasurementInstrumentBase subclasses): declare the
  self-describing class attributes (``measurement_parameters`` of ParamSpec,
  ``measurement_data_keys``, valid ``measurement_scalar_columns`` dtypes),
  implement the ``data_arrays`` / ``initiate_measurement`` / ``take_reading`` / ``standby``
  lifecycle, and round-trip against their sim drivers so the returned keys and
  array lengths match what they declare.
* The declaration standard (see virtual_instruments/README.md): every
  ``@monitored`` field declares a unit and a description, every ``@control``
  parameter is covered by a ``ParamSpec``, and every UI-group tag resolves —
  so a VI's capability manifest is complete the moment its file exists.
* Procedures: subclass BaseProcedure, have a name, declare a default for every
  parameter, and are constructible from defaults alone.
* Configs: every ``i2as/configs/<name>/`` directory has a loadable
  devices.yaml + monitor.yaml whose classes import and whose driver references
  resolve.
* The code-reference standard (see CLAUDE.md): no source file or folder
  README under ``i2as/`` cites a document in ``docs/plans/``. Plans are
  dated proposals that get implemented, superseded and archived, so a
  citation rots silently; the code and its READMEs must present the complete
  picture on their own. Vendor manual sections are the deliberate exception
  and are not flagged.
* The responsive-GUI rule (see gui/README.md): nothing under ``i2as/gui/``
  blocks the Qt event loop with ``time.sleep``.
* The recipe contract (see analysis/base.py and analysis/README.md): every
  shipped **analysis recipe** declares a snake_case ``name``, a non-empty
  ``description`` and a non-empty ``procedures`` tuple, implements
  ``analyse(self, run, context)`` with the contract's signature, and returns
  one ``ok`` **analysis report** for a real run file.
* The folder README standard (see CLAUDE.md): every folder README under
  ``i2as/`` carries the seven canonical sections, in order.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import json
import math
import pkgutil
import re
import typing
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

import i2as.core
import i2as.drivers
import i2as.procedures
import i2as.virtual_instruments
from i2as.core.availability import (
    AVAILABILITY_STATES,
    AVAILABILITY_TAGS,
    TAG_POLICY,
    TAG_PRECEDENCE,
)
from i2as.core.conditions import SEVERITIES
from i2as.core.events import (
    OPERATOR,
    Actor,
    ActorKind,
    Command,
    CommandName,
    ControlInfo,
    Datapoint,
    Event,
    GroupInfo,
    InstrumentInfo,
    LifecycleState,
    MonitoredInfo,
    QueueChanged,
    Readings,
    RunFinished,
    RunStarted,
    StateChange,
    StationInfo,
    StatusSnapshot,
    Verdict,
    VerdictCode,
    event_from_json,
)
from i2as.core.decorators import (
    VALID_CONTROL_SCOPES,
    control,
    get_control_panel,
    get_control_scope,
    get_control_specs,
    get_monitored_description,
    get_monitored_methods,
    get_monitored_unit,
    get_ui_group,
    monitored,
)
from i2as.core.exceptions import (
    I2ASCommunicationError,
    I2ASInstrumentError,
    I2ASPrivateActionError,
    I2ASSafetyError,
    I2ASUndeclaredActionError,
)
from i2as.core.plan import SETPOINT_PARAM_PREFIX, ImageBlock, ParamSpec, UIGroup
from i2as.core.procedure import BaseProcedure
from i2as.core.capability_manifest import (
    _instrument_json,
    build_manifest,
    validate_manifest,
)
from i2as.core.station import (
    LIFECYCLE_ACTIONS,
    Station,
    _control_infos,
    _import_class,
    _monitored_infos,
    build_station,
)
from i2as.session.maintenance_log import DECLARED_LOG_KINDS
from tests.mocks.bus_spy import spy_on_station
from i2as.virtual_instruments.base import (
    EXCITATION_CURRENT_LIMIT,
    MAX_SOURCE_CURRENT_KEY,
    BaseVirtualInstrument,
    MeasurementInstrumentBase,
)
from i2as.virtual_instruments.rampable import RampableVI

#: The package under test, derived once. Every module prefix, config path and
#: source-tree scan in this file comes from these, so a package rename is a
#: no-op here.
PACKAGE = i2as.__name__
PACKAGE_DIR = Path(i2as.__file__).parent
CONFIGS_DIR = PACKAGE_DIR / "configs"
#: The example station every station-dependent contract is checked on.
SIM_STATION_CONFIG = str(CONFIGS_DIR / "sim_cryostat")

# Modules in virtual_instruments/ that hold base classes, not concrete VIs.
VI_BASE_MODULES = {BaseVirtualInstrument.__module__, RampableVI.__module__}

# Registry types accepted by Station.register_vi via config (distinct from a VI
# class's own vi_type like "magnet" — see GLOSSARY.md).
CONFIG_VI_TYPES = {"system", "measurement"}


# ── Discovery helpers ─────────────────────────────────────────────────────────
# pkgutil.iter_modules / walk_packages list the modules inside a package at
# runtime — this is what makes these tests pick up new files automatically.


def _driver_module_names() -> list[str]:
    return sorted(m.name for m in pkgutil.iter_modules(i2as.drivers.__path__))


def _public_classes(module) -> list[type]:
    """Classes defined in *module* itself (not imported into it)."""
    return [
        cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__ and not name.startswith("_")
    ]


def _public_api(cls: type) -> dict[str, inspect.Signature]:
    """Public method name -> signature (self excluded)."""
    api: dict[str, inspect.Signature] = {}
    for name, func in inspect.getmembers(cls, inspect.isfunction):
        if name.startswith("_"):
            continue
        sig = inspect.signature(func)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        api[name] = sig.replace(parameters=params)
    return api


def _all_vi_classes() -> list[type]:
    """Every concrete VI class in i2as.virtual_instruments."""
    classes: list[type] = []
    for mod_info in pkgutil.walk_packages(
        i2as.virtual_instruments.__path__, prefix=f"{PACKAGE}.virtual_instruments."
    ):
        if mod_info.name in VI_BASE_MODULES:
            continue
        module = importlib.import_module(mod_info.name)
        for cls in _public_classes(module):
            if issubclass(cls, BaseVirtualInstrument):
                classes.append(cls)
    return classes


def _all_procedure_classes() -> list[type]:
    classes: list[type] = []
    for mod_info in pkgutil.iter_modules(i2as.procedures.__path__):
        module = importlib.import_module(f"{PACKAGE}.procedures.{mod_info.name}")
        for cls in _public_classes(module):
            if issubclass(cls, BaseProcedure) and cls is not BaseProcedure:
                classes.append(cls)
    return classes




def _load_yaml(path: Path) -> dict:
    from ruamel.yaml import YAML

    with path.open("r", encoding="utf-8") as f:
        return dict(YAML().load(f))


# ── Driver contract ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("module_name", _driver_module_names())
def test_driver_module_contract(module_name: str) -> None:
    """Every driver module holds exactly one public class taking one required arg."""
    module = importlib.import_module(f"{PACKAGE}.drivers.{module_name}")
    classes = _public_classes(module)
    assert len(classes) == 1, (
        f"{PACKAGE}.drivers.{module_name} must define exactly one public class, "
        f"found {[c.__name__ for c in classes]}"
    )
    init_params = [
        p
        for p in inspect.signature(classes[0].__init__).parameters.values()
        if p.name != "self"
    ]
    required = [
        p
        for p in init_params
        if p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert len(required) == 1, (
        f"{classes[0].__name__}.__init__ must take exactly one required argument "
        f"(the VISA resource string), got {[p.name for p in required]}"
    )


@pytest.mark.parametrize("module_name", _driver_module_names())
def test_driver_has_get_idn(module_name: str) -> None:
    """Every driver exposes get_idn() taking no arguments.

    get_idn() is the universal "is the right instrument at this address?"
    probe used by the troubleshoot engine's config preflight. Pre-SCPI
    instruments that do not answer ``*IDN?`` (the Oxford ISOBUS family)
    implement it with their native identify command (``V``) instead.
    """
    module = importlib.import_module(f"{PACKAGE}.drivers.{module_name}")
    (cls,) = _public_classes(module)
    method = getattr(cls, "get_idn", None)
    assert callable(method), (
        f"{cls.__name__} lacks get_idn() — every driver must expose an "
        f"identification query under this uniform name"
    )
    required = [
        p
        for p in inspect.signature(method).parameters.values()
        if p.name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert not required, (
        f"{cls.__name__}.get_idn() must take no required arguments, "
        f"got {[p.name for p in required]}"
    )


@pytest.mark.parametrize("module_name", _driver_module_names())
def test_driver_has_close(module_name: str) -> None:
    """Every driver exposes close() taking no arguments.

    The driver half of the connection-lifecycle standard (see
    ``BaseVirtualInstrument`` and ``drivers/README.md``): ``close()`` releases
    the bus session so the operator can drive the instrument from its own
    front panel or vendor software. ``Station.disconnect_instrument()`` calls
    it on every driver the disconnecting VI exclusively owns, so a driver
    without one would silently leak a session that keeps the instrument
    locked to I2AS.
    """
    module = importlib.import_module(f"{PACKAGE}.drivers.{module_name}")
    (cls,) = _public_classes(module)
    method = getattr(cls, "close", None)
    assert callable(method), (
        f"{cls.__name__} lacks close() — every driver must be able to "
        f"release its session (the connection-lifecycle standard)"
    )
    required = [
        p
        for p in inspect.signature(method).parameters.values()
        if p.name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert not required, (
        f"{cls.__name__}.close() must take no required arguments, "
        f"got {[p.name for p in required]}"
    )


@pytest.mark.parametrize("module_name", _driver_module_names())
def test_driver_has_safe_shutdown(module_name: str) -> None:
    """Every driver exposes safe_shutdown() taking no arguments.

    The **safe-shutdown standard** (see ``drivers/README.md``): one
    guaranteed, idempotent "leave it safe" per instrument, so anything that
    has to abandon a sequence — a failed procedure, an emergency stop, an
    agent that stopped answering — has one call to make on every driver
    without knowing which instrument it is talking to. Duck-typed like
    ``get_idn()``/``close()``: there is no DriverBase, so this test is the
    contract.
    """
    module = importlib.import_module(f"{PACKAGE}.drivers.{module_name}")
    (cls,) = _public_classes(module)
    method = getattr(cls, "safe_shutdown", None)
    assert callable(method), (
        f"{cls.__name__} lacks safe_shutdown() — every driver must offer one "
        f"idempotent, never-raising way to leave its instrument safe "
        f"(the safe-shutdown standard)"
    )
    required = [
        p
        for p in inspect.signature(method).parameters.values()
        if p.name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert not required, (
        f"{cls.__name__}.safe_shutdown() must take no required arguments "
        f"(the caller cannot know instrument-specific parameters), "
        f"got {[p.name for p in required]}"
    )


# Sim attributes that legitimately change on every call because they track
# wall-clock time, and so are excluded from the "second call is a no-op"
# comparison below. Everything else must be untouched by a repeat call.
_TIME_TRACKING_SIM_ATTRS = frozenset({"_last_update"})


@pytest.mark.parametrize(
    "module_name",
    [m for m in _driver_module_names() if m.startswith("sim_")],
)
def test_sim_driver_safe_shutdown_reaches_a_declared_safe_state(module_name: str) -> None:
    """A sim's safe_shutdown() is idempotent and lands in its declared safe state.

    The sim half of the **safe-shutdown standard**. Each sim declares what
    safe means for its instrument in ``_is_in_safe_state()`` — private, so
    the real/sim public-API parity contract stays intact, and documented in
    the sim's own docstring (a magnet's safe state is HOLD, not zero field; a
    meter's is its idle input mode, not off). This test asserts the three
    properties the standard promises: the call works from the sim's
    as-constructed state, it leaves the instrument in that declared state,
    and calling it a second time changes nothing at all.
    """
    module = importlib.import_module(f"{PACKAGE}.drivers.{module_name}")
    (cls,) = _public_classes(module)
    driver = cls("SIM::CONFORMANCE")

    predicate = getattr(driver, "_is_in_safe_state", None)
    assert callable(predicate), (
        f"{cls.__name__} lacks _is_in_safe_state() — every sim must declare, "
        f"as an executable predicate, what safe state its safe_shutdown() "
        f"leaves the instrument in (the safe-shutdown standard)"
    )

    driver.safe_shutdown()
    assert predicate(), (
        f"{cls.__name__}.safe_shutdown() did not leave the sim in the state "
        f"its own _is_in_safe_state() declares as safe"
    )

    before = {
        k: v for k, v in vars(driver).items() if k not in _TIME_TRACKING_SIM_ATTRS
    }
    driver.safe_shutdown()
    after = {
        k: v for k, v in vars(driver).items() if k not in _TIME_TRACKING_SIM_ATTRS
    }
    assert after == before, (
        f"{cls.__name__}.safe_shutdown() is not idempotent — a second call "
        f"changed {sorted(k for k in after if after[k] != before.get(k))}"
    )
    assert predicate()


@pytest.mark.parametrize(
    "module_name",
    [m for m in _driver_module_names() if m.startswith("sim_")],
)
def test_sim_driver_constructs_without_hardware(module_name: str) -> None:
    """Sim drivers must be constructible with a dummy resource string."""
    module = importlib.import_module(f"{PACKAGE}.drivers.{module_name}")
    (cls,) = _public_classes(module)
    cls("SIM::CONFORMANCE")


@pytest.mark.parametrize(
    "module_name",
    [m for m in _driver_module_names() if m.startswith("sim_")],
)
def test_sim_driver_is_dead_after_close(module_name: str) -> None:
    """A closed sim driver fails every command, starting with get_idn().

    The sim half of the connection-lifecycle standard's ``close()`` contract:
    a released session is really gone. Sims model it (rather than no-opping)
    for the same reason they model every other failure mode — so a
    use-after-disconnect bug fails here instead of on hardware. ``close()``
    itself must stay idempotent and silent: a disconnect always succeeds.
    """
    module = importlib.import_module(f"{PACKAGE}.drivers.{module_name}")
    (cls,) = _public_classes(module)
    driver = cls("SIM::CONFORMANCE")
    driver.get_idn()  # reachable before the close
    driver.close()
    driver.close()  # idempotent — must not raise
    with pytest.raises(I2ASCommunicationError):
        driver.get_idn()


@pytest.mark.parametrize(
    "real_name",
    [
        m
        for m in _driver_module_names()
        if not m.startswith("sim_") and f"sim_{m}" in _driver_module_names()
    ],
)
def test_sim_real_driver_api_parity(real_name: str) -> None:
    """A real driver and its sim_ twin must expose identical public APIs.

    This is the contract that lets a config swap sim for real hardware without
    touching any VI or procedure: code written against the sim works on the
    real instrument. Method names AND signatures must match exactly.
    """
    real_mod = importlib.import_module(f"{PACKAGE}.drivers.{real_name}")
    sim_mod = importlib.import_module(f"{PACKAGE}.drivers.sim_{real_name}")
    (real_cls,) = _public_classes(real_mod)
    (sim_cls,) = _public_classes(sim_mod)

    real_api = _public_api(real_cls)
    sim_api = _public_api(sim_cls)

    assert real_api.keys() == sim_api.keys(), (
        f"Public API mismatch between {real_cls.__name__} and {sim_cls.__name__}: "
        f"real-only={sorted(real_api.keys() - sim_api.keys())}, "
        f"sim-only={sorted(sim_api.keys() - real_api.keys())}"
    )
    for method, real_sig in real_api.items():
        sim_params = list(sim_api[method].parameters)
        real_params = list(real_sig.parameters)
        assert real_params == sim_params, (
            f"Signature mismatch on {method}(): "
            f"{real_cls.__name__}{real_sig} vs {sim_cls.__name__}{sim_api[method]}"
        )


# ── Virtual-instrument contract ───────────────────────────────────────────────


@pytest.mark.parametrize("vi_cls", _all_vi_classes(), ids=lambda c: c.__name__)
def test_vi_contract(vi_cls: type) -> None:
    """Every concrete VI sets vi_type and uses __init__(self, drivers, **init_params)."""
    assert vi_cls.vi_type != "unknown", (
        f"{vi_cls.__name__} must set vi_type (inherit from a typed base such as "
        f"MagnetBase / TemperatureControllerBase, or set the class attribute)"
    )

    params = list(inspect.signature(vi_cls.__init__).parameters.values())
    assert params[0].name == "self"
    assert len(params) >= 2 and params[1].name == "drivers", (
        f"{vi_cls.__name__}.__init__ first argument must be 'drivers' "
        f"(the Station injects driver instances there)"
    )
    kinds = [p.kind for p in params]
    assert inspect.Parameter.VAR_KEYWORD in kinds, (
        f"{vi_cls.__name__}.__init__ must accept **init_params "
        f"(config init_params are passed as keyword arguments)"
    )
    extra_required = [
        p
        for p in params[2:]
        if p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert not extra_required, (
        f"{vi_cls.__name__}.__init__ has required arguments beyond 'drivers': "
        f"{[p.name for p in extra_required]} — give them defaults and read them "
        f"from **init_params instead"
    )


# ── Connection-lifecycle standard ─────────────────────────────────────────────
# See BaseVirtualInstrument's "Connection-lifecycle standard" docstring:
# connect/disconnect own the bus session, initiate/standby own the instrument's
# state, and building the Station sends nothing but an identity query. These
# tests make rule 1 ("construction is silent") binding for every present and
# future VI, which is the rule a new driver or VI is most likely to break by
# accident — a single convenience command in __init__ is easy to write and
# invisible until it clobbers a running experiment at app startup.

# Commands a VI's __init__ MAY still issue on its drivers. `close` is a
# RELEASE, not a state change: an externally configured VI is born detached so
# that starting I2AS while the vendor tool holds the instrument works at
# all (see MeasurementInstrumentBase's "Externally configured instruments").
_SILENT_CONSTRUCTION_ALLOWED = frozenset({"close"})


def _vi_specs_from_configs() -> list[tuple[str, str, type, dict]]:
    """Return ``(config, vi_name, vi_class, init_params)`` for every configured VI.

    Drawn from the shipped configs rather than a hand-written list, so a VI
    added to any setup is covered the moment its config entry exists — and
    with the REAL init_params it will be built with, which is what decides
    whether a constructor takes a command path or not.
    """
    specs: list[tuple[str, str, type, dict]] = []
    # CONFIGS_DIR directly rather than _config_dirs(), which is defined
    # further down with the config-contract tests; this parametrize runs at
    # import time.
    for config_dir in sorted(p for p in CONFIGS_DIR.iterdir() if p.is_dir()):
        devices = _load_yaml(config_dir / "devices.yaml")
        for vi_name, vi_cfg in (devices.get("virtual_instruments") or {}).items():
            specs.append(
                (
                    config_dir.name,
                    vi_name,
                    _import_class(vi_cfg["class"]),
                    dict(vi_cfg.get("init_params") or {}),
                )
            )
    return specs


@pytest.mark.parametrize(
    "config_name, vi_name, vi_cls, init_params",
    _vi_specs_from_configs(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_vi_construction_sends_no_commands(
    config_name: str, vi_name: str, vi_cls: type, init_params: dict
) -> None:
    """Building a VI must not command its instruments (connection-lifecycle rule 1).

    Constructs the VI with recording stand-ins for its drivers and asserts
    nothing was called on them. Building the Station is a *connection* act:
    the only command it sends is the identity query, so an operator who
    starts I2AS mid-experiment — or while the instrument is on its own
    front panel — finds every instrument exactly as they left it. A setup
    command that wants to run at bring-up belongs in ``initiate()``.

    ``MagicMock`` stands in for each driver: it accepts any call and any
    attribute, so this exercises the constructor's real path (config
    validation included) while recording every command it would have sent.
    """
    from unittest.mock import MagicMock

    drivers: dict[str, MagicMock] = {}

    class _RecordingDrivers(dict):
        """A driver dict that mints a recording driver for any role asked for."""

        def __missing__(self, role: str) -> MagicMock:
            driver = MagicMock(name=f"driver:{role}")
            drivers[role] = driver
            self[role] = driver
            return driver

    vi_cls(_RecordingDrivers(), **init_params)

    offenders: list[str] = []
    for role, driver in drivers.items():
        for call in driver.mock_calls:
            # call[0] is the dotted name: "" for the driver itself being
            # called, "set_rate" for a method, "adapter.write" for nesting.
            name = call[0]
            if not name or name.split(".")[0] in _SILENT_CONSTRUCTION_ALLOWED:
                continue
            offenders.append(f"{role}.{name}")

    assert not offenders, (
        f"{config_name}/{vi_name} ({vi_cls.__name__}).__init__ commanded its "
        f"instrument(s): {sorted(set(offenders))}. Building the Station must "
        f"send nothing but the identity query (the connection-lifecycle "
        f"standard, see BaseVirtualInstrument) — move these to initiate()."
    )


@pytest.mark.parametrize("vi_cls", _all_vi_classes(), ids=lambda c: c.__name__)
def test_vi_has_connection_lifecycle_hooks(vi_cls: type) -> None:
    """Every VI answers ping() and disconnect() with no required arguments.

    Both are inherited from ``BaseVirtualInstrument``, so this only ever
    fails on a VI that overrode one with a different shape — which would
    break ``build_station()``'s identity check or
    ``Station.disconnect_instrument()`` for that instrument alone, silently,
    on whichever setup happens to configure it. ``is_attached()`` — the
    detach-when-idle declaration's observable state (see the class
    docstring) — is part of the same hook set.
    """
    for name in ("ping", "disconnect", "initiate", "standby", "is_attached"):
        method = getattr(vi_cls, name, None)
        assert callable(method), f"{vi_cls.__name__} lacks {name}()"
        required = [
            p
            for p in inspect.signature(method).parameters.values()
            if p.name != "self"
            and p.default is inspect.Parameter.empty
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        assert not required, (
            f"{vi_cls.__name__}.{name}() must take no required arguments, "
            f"got {[p.name for p in required]}"
        )


# ── Availability standard (i2as.core.availability) ───────────────────────
# See availability.py's module docstring: a closed tag vocabulary, a derived
# state vocabulary, and a declared tag -> policy table (TAG_POLICY) are the
# one place the "why can't I use this instrument?" policies are stated. These
# tests make the table's own internal consistency binding, the moment a tag
# or a policy row is added or edited — a fixed module, not one pkgutil
# auto-discovers, but the checked-in harness (`make check`) is where a future
# edit meets them regardless of whether the author also knows about
# tests/test_availability.py's pure-policy coverage of the same invariants.


def test_tag_policy_covers_exactly_the_availability_tags() -> None:
    """TAG_POLICY has one row per tag, no orphan row, no undeclared tag."""
    assert set(TAG_POLICY) == set(AVAILABILITY_TAGS), (
        f"TAG_POLICY keys {sorted(TAG_POLICY)} must exactly match "
        f"AVAILABILITY_TAGS {sorted(AVAILABILITY_TAGS)} — no orphan row, no "
        f"undeclared tag"
    )
    for tag, policy in TAG_POLICY.items():
        assert policy.tag == tag, (
            f"TAG_POLICY[{tag!r}].tag == {policy.tag!r} — a TagPolicy's own "
            f"'tag' field must match the key it is filed under"
        )


def test_tag_policy_states_and_precedence_stay_in_vocabulary() -> None:
    """Every TagPolicy.state is declared; TAG_PRECEDENCE is a permutation of the tags."""
    for tag, policy in TAG_POLICY.items():
        assert policy.state in AVAILABILITY_STATES, (
            f"TAG_POLICY[{tag!r}].state {policy.state!r} is not one of "
            f"AVAILABILITY_STATES {AVAILABILITY_STATES}"
        )
    assert sorted(TAG_PRECEDENCE) == sorted(AVAILABILITY_TAGS), (
        f"TAG_PRECEDENCE {TAG_PRECEDENCE} must be a permutation of "
        f"AVAILABILITY_TAGS {AVAILABILITY_TAGS}"
    )
    assert len(TAG_PRECEDENCE) == len(set(TAG_PRECEDENCE)), (
        f"TAG_PRECEDENCE {TAG_PRECEDENCE} names a tag more than once"
    )


# ── Detach-when-idle standard (BaseVirtualInstrument.detach_when_idle) ───────
# See BaseVirtualInstrument's "Detach-when-idle declaration": a VI opts in by
# overriding the detach_when_idle property; __init_subclass__'s wrap (or the
# base's own standby()) then releases the driver session automatically. These
# tests make the opt-in binding for every present and future VI that declares
# it. No shipped config declares it today, so the parametrisation is empty
# until one does — which is the point: the checks arrive with the VI.


def _sim_driver_class_for(real_dotted: str) -> type:
    """Return the sim twin class for a driver's dotted class path.

    A config may already reference a ``sim_*`` module directly (e.g.
    ``sim_cryostat``) — imported as-is. Otherwise uses the same
    ``sim_<module>`` naming convention as ``test_sim_real_driver_api_parity``
    — the one existing derivation from a real driver's dotted path to its
    sim twin.
    """
    module_path, _, _ = real_dotted.rpartition(".")
    package, _, module_name = module_path.rpartition(".")
    sim_module_name = module_name if module_name.startswith("sim_") else f"sim_{module_name}"
    sim_module = importlib.import_module(f"{package}.{sim_module_name}")
    (sim_cls,) = _public_classes(sim_module)
    return sim_cls


def _detach_when_idle_vi_specs() -> list[tuple[str, type, dict, dict[str, type]]]:
    """(spec id, vi_cls, init_params, {role: sim driver class}) for every
    configured VI whose REAL config build declares ``detach_when_idle``.

    Drawn from the shipped configs' real ``drivers`` role mapping and real
    ``init_params`` — with each real driver class swapped
    for its sim twin so the checks run with no hardware. Filtering is done
    by actually constructing the VI and reading ``detach_when_idle``, so a
    future config declaring it is covered automatically, without this file
    needing to know which VI or config that will be.
    """
    specs: list[tuple[str, type, dict, dict[str, type]]] = []
    for config_dir in sorted(p for p in CONFIGS_DIR.iterdir() if p.is_dir()):
        devices = _load_yaml(config_dir / "devices.yaml")
        real_drivers_cfg = devices.get("real_drivers") or {}
        for vi_name, vi_cfg in (devices.get("virtual_instruments") or {}).items():
            vi_cls = _import_class(vi_cfg["class"])
            if not issubclass(vi_cls, MeasurementInstrumentBase):
                continue
            init_params = dict(vi_cfg.get("init_params") or {})
            role_map = vi_cfg.get("drivers") or {}
            sim_driver_classes = {
                role: _sim_driver_class_for(real_drivers_cfg[driver_key]["class"])
                for role, driver_key in role_map.items()
            }
            probe_drivers = {
                role: cls("SIM::CONFORMANCE") for role, cls in sim_driver_classes.items()
            }
            vi = vi_cls(probe_drivers, **init_params)
            if vi.detach_when_idle:
                specs.append(
                    (f"{config_dir.name}/{vi_name}", vi_cls, init_params, sim_driver_classes)
                )
    return specs


@pytest.mark.parametrize(
    "spec_id, vi_cls, init_params, sim_driver_classes",
    _detach_when_idle_vi_specs(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_detach_when_idle_vi_drivers_support_ensure_connected(
    spec_id: str, vi_cls: type, init_params: dict, sim_driver_classes: dict[str, type]
) -> None:
    """A VI declaring detach_when_idle must have reconnect-capable drivers.

    ``BaseVirtualInstrument._attach()`` duck-types ``ensure_connected()``
    (deliberately not part of the driver contract — see
    ``drivers/README.md``): a VI that declares ``detach_when_idle`` without
    a driver implementing it would silently never reacquire its session.
    """
    for role, cls in sim_driver_classes.items():
        driver = cls("SIM::CONFORMANCE")
        assert callable(getattr(driver, "ensure_connected", None)), (
            f"{spec_id} ({vi_cls.__name__}) declares detach_when_idle but "
            f"its {role!r} driver ({cls.__name__}) has no ensure_connected() "
            f"— the detach-when-idle standard's opt-in reconnect capability"
        )


@pytest.mark.parametrize(
    "spec_id, vi_cls, init_params, sim_driver_classes",
    _detach_when_idle_vi_specs(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_detach_when_idle_vi_really_releases_and_reacquires(
    spec_id: str, vi_cls: type, init_params: dict, sim_driver_classes: dict[str, type]
) -> None:
    """standby() really releases the session; arming really reacquires it.

    Built against the sim driver — proves the declaration is framework
    behaviour that actually moves ``is_attached()``, not just a flag that
    reads true without anything happening underneath.
    """
    drivers = {role: cls("SIM::CONFORMANCE") for role, cls in sim_driver_classes.items()}
    vi = vi_cls(drivers, **init_params)
    defaults = {name: spec.default for name, spec in vi_cls.measurement_parameters.items()}

    vi.initiate_measurement(**defaults)
    assert vi.is_attached() is True, (
        f"{spec_id} ({vi_cls.__name__}).initiate_measurement() must "
        f"reacquire the session for a detach_when_idle VI"
    )

    vi.standby()
    assert vi.is_attached() is False, (
        f"{spec_id} ({vi_cls.__name__}).standby() must release the session "
        f"for a detach_when_idle VI"
    )


def test_detach_when_idle_vi_owns_its_driver_aliases_exclusively() -> None:
    """A detach_when_idle VI must not share a config driver alias with any other VI.

    ``BaseVirtualInstrument._detach()`` releases every driver in
    ``self._drivers`` unconditionally (see the "Detach-when-idle
    declaration" docstring) — it has no notion of another VI still needing
    that same session, because a VI may never consult the Station's alias
    map (Layer 1 cannot import Layer 2). ``Station.disconnect_instrument()``
    avoids exactly this hazard for its own release path by routing through
    ``_exclusive_aliases()`` before closing anything; this test applies the
    SAME predicate to every shipped config, for every VI this file's
    ``_detach_when_idle_vi_specs()`` found to declare ``detach_when_idle``,
    so a future config that lets two VIs share an alias with one of them
    detach_when_idle fails CI instead of silently breaking the other VI's
    session in the field.
    """
    from i2as.core.station import _exclusive_aliases

    detach_when_idle_spec_ids = {spec_id for spec_id, *_ in _detach_when_idle_vi_specs()}

    for config_dir in sorted(p for p in CONFIGS_DIR.iterdir() if p.is_dir()):
        devices = _load_yaml(config_dir / "devices.yaml")
        vi_cfgs = devices.get("virtual_instruments") or {}
        for vi_name, vi_cfg in vi_cfgs.items():
            spec_id = f"{config_dir.name}/{vi_name}"
            if spec_id not in detach_when_idle_spec_ids:
                continue
            role_aliases = vi_cfg.get("drivers") or {}
            mine = set(role_aliases.values())
            exclusive = set(_exclusive_aliases(role_aliases, vi_cfgs, vi_name))
            shared = mine - exclusive
            assert not shared, (
                f"{spec_id}: declares detach_when_idle but driver alias/es "
                f"{sorted(shared)} are also named by another VI in "
                f"{config_dir.name}/devices.yaml — _detach() would close "
                f"them unconditionally on standby(), breaking whichever "
                f"other VI still needs the shared session; a "
                f"detach_when_idle VI must own its driver aliases "
                f"exclusively (see BaseVirtualInstrument's "
                f"'Detach-when-idle declaration')"
            )


# ── Safety-flag manifest standard ─────────────────────────────────────────────
# See BaseVirtualInstrument's "Safety-flag manifest standard" docstring: every
# flag a VI's evaluate_safety() can report is declared, once, in the class
# attribute `safety_flags` (flag name -> severity), merged over the MRO by
# `merged_safety_flags()` exactly like `control_limits`. `safety_concerns()`
# is the consumer side and may only name hold-severity flags. These tests make
# the standard binding for every present and future VI.

# Valid severities for a safety_flags manifest entry — the System-Condition
# standard's severity ladder (scope follows from severity: "hold" is scoped
# to concerned VIs, "critical" is station-wide by definition, "advisory" is
# reserved with no enforcement yet), canonically declared in
# i2as.core.conditions.
SAFETY_FLAG_SEVERITIES = SEVERITIES


def _all_manifest_severities() -> dict[str, str]:
    """{flag: severity} unioned across every discovered VI class's merged manifest.

    The global producer-side picture: a consumer VI's ``safety_concerns()``
    may name a flag reported by ANY VI, not just its own, so validating a
    concern requires looking here rather than at one class's own manifest.
    """
    severities: dict[str, str] = {}
    for cls in _all_vi_classes():
        severities.update(cls.merged_safety_flags())
    return severities


@pytest.mark.parametrize("vi_cls", _all_vi_classes(), ids=lambda c: c.__name__)
def test_safety_flags_manifest_is_valid(vi_cls: type) -> None:
    """Every safety_flags entry has a valid severity; no MRO contradictions.

    Every key in the merged-MRO manifest is a non-empty str and every value
    is one of SAFETY_FLAG_SEVERITIES. Separately, walking the MRO directly
    (rather than through the already-merged dict) catches a subclass that
    redeclares an inherited flag with a DIFFERENT severity than a base
    already gave it — the one thing merged_safety_flags() silently allows
    (last-declared-wins) because the "no contradiction" invariant is a
    conformance concern, not a runtime one (see the class docstring).
    """
    merged = vi_cls.merged_safety_flags()
    for flag, severity in merged.items():
        assert isinstance(flag, str) and flag, (
            f"{vi_cls.__name__}.safety_flags key {flag!r} must be a "
            f"non-empty str"
        )
        assert severity in SAFETY_FLAG_SEVERITIES, (
            f"{vi_cls.__name__}.safety_flags[{flag!r}] = {severity!r} is not "
            f"one of {SAFETY_FLAG_SEVERITIES}"
        )

    declared: dict[str, str] = {}
    for klass in reversed(vi_cls.__mro__):
        own = vars(klass).get("safety_flags") or {}
        for flag, severity in own.items():
            assert flag not in declared or declared[flag] == severity, (
                f"{vi_cls.__name__}: {klass.__name__}.safety_flags "
                f"redeclares {flag!r} as {severity!r}, contradicting the "
                f"inherited severity {declared.get(flag)!r}"
            )
            declared[flag] = severity


@pytest.mark.parametrize("vi_cls", _all_vi_classes(), ids=lambda c: c.__name__)
def test_safety_concerns_are_consumer_side_and_hold_only(vi_cls: type) -> None:
    """Every safety_concerns() flag is hold-severity and actually producible.

    The consumer-side half of the safety-flag manifest standard: a VI's
    safety_concerns() may only name flags that (a) appear in SOME
    discovered VI's merged safety_flags manifest, and (b) have severity
    "hold" there. Naming a critical flag here would be meaningless — a
    tripped critical flag already forces station-wide EMERGENCY (the
    System-Condition standard, ``core/conditions.py``: critical scope is
    station-wide by construction), which stops this VI (and every other)
    regardless of any per-VI concern declaration. No shipped VI declares a
    concern today; MagnetBase reports "quench" (critical) and deliberately
    does NOT name it as a concern.

    A bare (``__init__``-less) instance is enough: every existing
    ``safety_concerns()`` override is a pure function of the class, never
    of instance state populated by ``__init__`` (see
    ``tests/test_l0_driver_errors.py`` for the same ``object.__new__`` idiom
    used to probe a class without constructing it).
    """
    severities = _all_manifest_severities()
    concerns = object.__new__(vi_cls).safety_concerns()
    for flag in concerns:
        assert flag in severities, (
            f"{vi_cls.__name__}.safety_concerns() names {flag!r}, which no "
            f"discovered VI's safety_flags manifest produces"
        )
        assert severities[flag] == "hold", (
            f"{vi_cls.__name__}.safety_concerns() names {flag!r}, whose "
            f"manifest severity is {severities[flag]!r}, not 'hold'"
        )


def test_no_dead_hold_flags() -> None:
    """Every hold-severity flag some VI can produce has at least one consumer.

    A hold-severity flag with no VI declaring it in safety_concerns() would
    trip a per-VI hold that holds nothing — the producer side of the
    safety-flag manifest standard would be dead weight. Checked once,
    globally, rather than per-VI, since "has a consumer" is a property of
    the whole station's declarations, not of any one class.
    """
    severities = _all_manifest_severities()
    hold_flags = {flag for flag, severity in severities.items() if severity == "hold"}
    consumed: set[str] = set()
    for cls in _all_vi_classes():
        consumed |= object.__new__(cls).safety_concerns()
    dead = hold_flags - consumed
    assert not dead, f"hold-severity flag(s) with no consumer: {sorted(dead)}"


# ── Procedure contract ────────────────────────────────────────────────────────


@pytest.mark.parametrize("proc_cls", _all_procedure_classes(), ids=lambda c: c.__name__)
def test_procedure_declaration(proc_cls: type) -> None:
    """Every procedure names itself and declares each parameter as a ParamSpec."""
    assert proc_cls.name, f"{proc_cls.__name__} must set the 'name' class attribute"
    assert proc_cls.parameters, f"{proc_cls.__name__} declares no parameters"
    for param_name, spec in proc_cls.parameters.items():
        # Parameters are ParamSpec now (Wave 4). The old "spec has 'type' and
        # 'default'" checks moved INTO the type: ParamSpec.__post_init__ requires
        # both (and validates choices / bounds) at construction, so a value being
        # a ParamSpec instance is exactly that guarantee — and every parameter
        # still carries a default, so the procedure runs unattended.
        assert isinstance(spec, ParamSpec), (
            f"{proc_cls.__name__}.{param_name} must be a ParamSpec, got {spec!r}"
        )
    valid_x_keys = (
        ["unix_time"] + list(proc_cls.sweep_data_keys) + list(proc_cls.measurement_data_keys)
    )
    assert proc_cls.default_x_key in valid_x_keys, (
        f"{proc_cls.__name__}.default_x_key={proc_cls.default_x_key!r} is not one "
        f"of its data keys {valid_x_keys}"
    )


@pytest.mark.parametrize("proc_cls", _all_procedure_classes(), ids=lambda c: c.__name__)
def test_procedure_has_description(proc_cls: type) -> None:
    """Every procedure declares a non-empty ``description`` class attribute.

    The one-line description is what the procedure selector, the run record
    and the agent's procedure listing show beside the name; a procedure
    without one renders as a bare title that says nothing about what the
    run does. ``name`` is checked in ``test_procedure_declaration``.
    """
    description = getattr(proc_cls, "description", "")
    assert isinstance(description, str) and description.strip(), (
        f"{proc_cls.__name__} must set a non-empty 'description' class attribute"
    )


@pytest.mark.parametrize("proc_cls", _all_procedure_classes(), ids=lambda c: c.__name__)
def test_procedure_roles_resolve_on_the_sim_station(proc_cls: type) -> None:
    """Every declared role is a ``RoleParam`` whose candidates come from the station.

    The role-discovery standard: a procedure names roles, never configured
    instrument names, so each role's ``candidates`` must be answerable by a
    Station and each becomes a parameter carrying a description. On the
    shipped sim station every candidate list is a subset of the registered
    VIs.
    """
    from i2as.core.procedure import RoleParam

    station = build_station(SIM_STATION_CONFIG)
    for param_name, role in proc_cls.role_parameters.items():
        assert isinstance(role, RoleParam), (
            f"{proc_cls.__name__}.role_parameters[{param_name!r}] must be a RoleParam"
        )
        assert role.description.strip(), (
            f"{proc_cls.__name__}.role_parameters[{param_name!r}] lacks a description"
        )
        names = role.candidates(station)
        assert set(names) <= set(station.get_vi_names()), (
            f"{proc_cls.__name__}.{param_name} candidates {names} are not all "
            f"registered VIs"
        )
        assert param_name in proc_cls.parameters, (
            f"{proc_cls.__name__}.{param_name} is a role but not a parameter"
        )


@pytest.mark.parametrize("proc_cls", _all_procedure_classes(), ids=lambda c: c.__name__)
def test_procedure_parameter_has_description(proc_cls: type) -> None:
    """Every procedure parameter declares a non-empty 'description'.

    The GUI parameter form (ProcedureWindow._build_param_form) now labels
    each field with the bare parameter name and moves the human-readable
    explanation into a hover tooltip. A parameter without a description would
    render a tooltip missing its most important sentence, so every parameter
    in sweep_parameters / system_parameters / measurement_parameters must
    carry one. 'unit' stays optional — dimensionless counts legitimately have
    none.
    """
    for group_name in ("sweep_parameters", "system_parameters", "measurement_parameters"):
        group_params = getattr(proc_cls, group_name)
        for param_name, spec in group_params.items():
            # ParamSpec allows an empty description; a *non-empty* one is a
            # procedure-level rule ParamSpec does NOT enforce, so it stays tested.
            description = spec.description
            assert isinstance(description, str) and description.strip(), (
                f"{proc_cls.__name__}.{group_name}['{param_name}'] lacks a "
                f"non-empty 'description' — add a one-sentence physics-appropriate "
                f"description, it is shown as the GUI tooltip"
            )


@pytest.mark.parametrize("proc_cls", _all_procedure_classes(), ids=lambda c: c.__name__)
def test_procedure_choices_spec(proc_cls: type) -> None:
    """Enumerated ('choices') parameters follow the label->value dict standard.

    A parameter that declares 'choices' renders as a GUI drop-down and its
    collected value is the *mapped* value (see BaseProcedure docstring and
    i2as.gui.param_form.build_param_widget). The three invariants this test
    used to assert one by one — choices is a non-empty label->value dict, every
    value is an instance of the declared 'type', and 'default' is one of the
    mapped values — have moved INTO the type: ParamSpec.__post_init__ enforces
    all of them at construction. So the class simply importing (which
    _all_procedure_classes already did) proves them. Here we only re-affirm that
    a choices-declaring parameter is a ParamSpec carrying a non-empty choices
    dict; the deeper enforcement is exercised by the ParamSpec unit tests.
    """
    for param_name, spec in proc_cls.parameters.items():
        if not (isinstance(spec, ParamSpec) and spec.choices):
            continue
        ctx = f"{proc_cls.__name__}.{param_name}"
        assert isinstance(spec.choices, dict) and spec.choices, (
            f"{ctx} 'choices' must be a non-empty label->value dict"
        )


@pytest.mark.parametrize("proc_cls", _all_procedure_classes(), ids=lambda c: c.__name__)
def test_procedure_constructs_from_defaults(proc_cls: type, tmp_path) -> None:
    """Every procedure must construct with zero explicit parameters.

    BaseProcedure merges declared defaults into the params dict, so a complete
    default set means agents and scripts can always instantiate a procedure
    without reproducing the GUI's parameter form.

    A generic sweep procedure (``requires_measurement_vi``) resolves its
    measurement VI and that VI's parameters from the station, so it cannot be
    built from an empty ``Station``; it is handed a populated sim station
    instead. This does not weaken the check for static procedures — they still
    build from an empty station — it only supplies the one thing a
    station-dependent procedure legitimately needs.
    """
    if getattr(proc_cls, "requires_measurement_vi", False):
        station = build_station(SIM_STATION_CONFIG)
    else:
        station = Station()
    proc = proc_cls(
        station=station,
        sample_info={"sample_name": "conformance", "sample_id": "T0", "comments": ""},
        data_directory=str(tmp_path),
    )
    assert proc.get_sweep_array(), (
        f"{proc_cls.__name__} built an empty sweep array from its own defaults"
    )


@pytest.mark.parametrize("proc_cls", _all_procedure_classes(), ids=lambda c: c.__name__)
def test_procedure_claimed_vi_names_contract(proc_cls: type, tmp_path) -> None:
    """claimed_vi_names() returns None or a set of VI names known to the station.

    Concurrency-scope hook: ``None`` (claim everything) is always valid; a
    non-``None``
    return must be a ``set[str]`` naming VIs the station actually has, so a
    typo in a narrowed claim can never silently under-claim.
    """
    if getattr(proc_cls, "requires_measurement_vi", False):
        station = build_station(SIM_STATION_CONFIG)
    else:
        station = Station()
    proc = proc_cls(
        station=station,
        sample_info={"sample_name": "conformance", "sample_id": "T0", "comments": ""},
        data_directory=str(tmp_path),
    )
    claimed = proc.claimed_vi_names()
    if claimed is None:
        return
    assert isinstance(claimed, set) and all(isinstance(name, str) for name in claimed), (
        f"{proc_cls.__name__}.claimed_vi_names() must return None or a set[str], "
        f"got {claimed!r}"
    )
    known = set(station.get_vi_names())
    unknown = claimed - known
    assert not unknown, (
        f"{proc_cls.__name__}.claimed_vi_names() names VI(s) not on the station: "
        f"{sorted(unknown)}"
    )


# ── Config contract ───────────────────────────────────────────────────────────


def _config_dirs() -> list[Path]:
    return sorted(p for p in CONFIGS_DIR.iterdir() if p.is_dir())


@pytest.mark.parametrize("config_dir", _config_dirs(), ids=lambda p: p.name)
def test_config_schema(config_dir: Path) -> None:
    """devices.yaml and monitor.yaml exist, load, and reference real classes."""
    devices_file = config_dir / "devices.yaml"
    monitor_file = config_dir / "monitor.yaml"
    assert devices_file.exists(), f"{config_dir.name} lacks devices.yaml"
    assert monitor_file.exists(), f"{config_dir.name} lacks monitor.yaml"

    devices = _load_yaml(devices_file)
    monitor = _load_yaml(monitor_file)

    assert "monitor" in monitor and "tick_interval_ms" in monitor["monitor"], (
        f"{config_dir.name}/monitor.yaml must define monitor.tick_interval_ms"
    )

    driver_names = set(devices.get("real_drivers", {}).keys())
    assert driver_names, f"{config_dir.name}/devices.yaml declares no real_drivers"

    for drv_name, drv_cfg in devices["real_drivers"].items():
        assert "class" in drv_cfg, f"driver '{drv_name}' lacks a 'class' entry"
        assert "address" in drv_cfg, f"driver '{drv_name}' lacks an 'address' entry"
        _import_class(drv_cfg["class"])  # raises I2ASConfigError if broken

    for vi_name, vi_cfg in devices.get("virtual_instruments", {}).items():
        assert "class" in vi_cfg, f"VI '{vi_name}' lacks a 'class' entry"
        vi_cls = _import_class(vi_cfg["class"])
        assert issubclass(vi_cls, BaseVirtualInstrument), (
            f"VI '{vi_name}' class {vi_cfg['class']} is not a BaseVirtualInstrument"
        )
        assert vi_cfg.get("vi_type") in CONFIG_VI_TYPES, (
            f"VI '{vi_name}' vi_type={vi_cfg.get('vi_type')!r} must be one of "
            f"{sorted(CONFIG_VI_TYPES)}"
        )
        for role, drv_ref in vi_cfg.get("drivers", {}).items():
            assert drv_ref in driver_names, (
                f"VI '{vi_name}' role '{role}' references unknown driver '{drv_ref}'"
            )


@pytest.mark.parametrize("config_dir", _config_dirs(), ids=lambda p: p.name)
def test_panels_config_names_real_vis_and_controls(config_dir: Path) -> None:
    """Every monitor.yaml panels: entry names a declared VI and its @control methods.

    A typo'd VI or control name would otherwise fail silently — the card
    would just render without the control the operator expected.
    """
    from i2as.core.config import read_panels_config

    panels = read_panels_config(str(config_dir))
    if not panels:
        return
    devices = _load_yaml(config_dir / "devices.yaml")
    declared_vis = devices.get("virtual_instruments", {})
    for vi_name, controls in panels.items():
        assert vi_name in declared_vis, (
            f"{config_dir.name}/monitor.yaml panels: names VI '{vi_name}', "
            f"which devices.yaml does not declare"
        )
        vi_cls = _import_class(declared_vis[vi_name]["class"])
        control_names = set(_control_methods(vi_cls))
        for control_name in controls:
            assert control_name in control_names, (
                f"{config_dir.name}/monitor.yaml panels: {vi_name} lists "
                f"'{control_name}', which is not a @control method on "
                f"{vi_cls.__name__}"
            )


# ── Control-validation standard ───────────────────────────────────────────────
# See BaseVirtualInstrument's "Control-validation standard" docstring: VIs
# declare control_limits (method -> {param: limit_name}); __init__ populates
# self._limits from the config's init_params; the base class enforces at call
# time. These tests make the standard binding for every future VI and config.


def test_control_limits_reference_real_control_methods_and_params() -> None:
    """Every control_limits entry names an existing @control method and real params."""
    for cls in _all_vi_classes():
        for method_name, param_map in cls.control_limits.items():
            method = getattr(cls, method_name, None)
            assert callable(method), (
                f"{cls.__name__}.control_limits names '{method_name}', "
                f"which is not a method on the class"
            )
            assert getattr(method, "_is_control", False), (
                f"{cls.__name__}.control_limits names '{method_name}', "
                f"which is not tagged @control — limits only guard @control methods"
            )
            sig_params = set(inspect.signature(method).parameters)
            for param_name in param_map:
                assert param_name in sig_params, (
                    f"{cls.__name__}.control_limits['{method_name}'] names "
                    f"parameter '{param_name}', which is not in the method signature"
                )


def _sim_config_dirs() -> list[Path]:
    """Config dirs whose drivers are all simulated (buildable without hardware)."""
    result = []
    for config_dir in _config_dirs():
        devices = _load_yaml(config_dir / "devices.yaml")
        driver_classes = [
            cfg["class"] for cfg in devices.get("real_drivers", {}).values()
        ]
        if driver_classes and all(".sim_" in c for c in driver_classes):
            result.append(config_dir)
    return result


@pytest.mark.parametrize("config_dir", _sim_config_dirs(), ids=lambda p: p.name)
def test_config_populates_all_declared_control_limits(config_dir: Path) -> None:
    """Every limit a VI declares must be populated when built from this config.

    A declared-but-unpopulated limit would otherwise only explode when a user
    presses the button (I2ASConfigError at call time); this test moves
    that failure to CI.
    """
    station = build_station(str(config_dir))
    for vi_name in station.get_vi_names():
        vi = getattr(station, vi_name)
        for method_name, param_map in type(vi).control_limits.items():
            for param_name, limit_name in param_map.items():
                assert limit_name in vi._limits, (
                    f"{config_dir.name}: VI '{vi_name}' declares limit "
                    f"'{limit_name}' for {method_name}({param_name}) but its "
                    f"__init__ never populated self._limits with it — add the "
                    f"config key or the derivation"
                )


@pytest.mark.parametrize("config_dir", _sim_config_dirs(), ids=lambda p: p.name)
def test_declared_finite_limits_reject_out_of_range(config_dir: Path) -> None:
    """Every finite upper limit actually refuses an out-of-range @control call.

    Calls each limited @control method with a value beyond its declared
    maximum and requires I2ASSafetyError — i.e. the standard is not just
    declared, it is enforced, for every VI in every buildable config.
    """
    station = build_station(str(config_dir))
    enforced = 0
    for vi_name in station.get_vi_names():
        vi = getattr(station, vi_name)
        for method_name, param_map in type(vi).control_limits.items():
            method = getattr(vi, method_name)
            control_params = getattr(method, "_control_params", {})
            for param_name, limit_name in param_map.items():
                _lo, hi = vi._limits[limit_name]
                if hi is None:
                    continue  # unbounded above — nothing to violate
                # Fill the method's other params from their defaults; skip if
                # any lacks one (cannot build a safe call generically).
                kwargs: dict = {}
                buildable = True
                for other_name, other_info in control_params.items():
                    if other_name == param_name:
                        continue
                    if "default" in other_info:
                        kwargs[other_name] = other_info["default"]
                    else:
                        buildable = False
                if not buildable:
                    continue
                kwargs[param_name] = hi + abs(hi) + 1.0
                with pytest.raises(I2ASSafetyError):
                    method(**kwargs)
                enforced += 1
    assert enforced > 0, (
        f"{config_dir.name}: no finite limits were exercised — expected at "
        f"least one limited @control method"
    )


@pytest.mark.parametrize("config_dir", _sim_config_dirs(), ids=lambda p: p.name)
def test_evaluate_safety_flags_are_declared_in_manifest(config_dir: Path) -> None:
    """Every flag a VI's evaluate_safety() reports is in its safety_flags manifest.

    Builds the sim station, takes one state snapshot, and calls each VI's
    ``evaluate_safety()`` with its own slice exactly as
    ``Station.check_safety()`` does (station.py's ``check_safety()``:
    ``vi.evaluate_safety(state.get(vi_name, {}))``), asserting the returned
    flag keys are a subset of that VI's merged ``safety_flags`` manifest —
    the sim round-trip proof that the declarative manifest actually matches
    what the VI can report, not just what it is documented to report.
    """
    station = build_station(str(config_dir))
    state = station.get_state()
    checked = 0
    for vi_name in station.get_vi_names():
        vi = station.get_vi(vi_name)
        flags = vi.evaluate_safety(state.get(vi_name, {}))
        manifest = type(vi).merged_safety_flags()
        undeclared = set(flags) - set(manifest)
        assert not undeclared, (
            f"{config_dir.name}: VI '{vi_name}' evaluate_safety() reported "
            f"undeclared flag(s) {sorted(undeclared)} — not in its "
            f"safety_flags manifest {sorted(manifest)}"
        )
        checked += 1
    assert checked > 0, f"{config_dir.name}: no VIs to check"


# ── Control-declaration standard (GUI metadata) ──────────────────────────────
# See i2as.core.decorators: @control optionally declares params=
# {name: ParamSpec} (widget shape, unit, bounds, choices) and panel=
# (default monitor-card placement). The decorator enforces name matching and
# the VI base class enforces the ParamSpec type at import; these tests bind
# the semantic parts for every VI, present and future.


@pytest.mark.parametrize("vi_cls", _all_vi_classes(), ids=lambda c: c.__name__)
def test_control_declarations_are_consistent(vi_cls: type) -> None:
    """Declared control ParamSpecs agree with the signature; panel is a bool."""
    for method_name, method in _control_methods(vi_cls).items():
        assert isinstance(get_control_panel(method), bool), (
            f"{vi_cls.__name__}.{method_name}: _control_panel must be a bool"
        )
        specs = get_control_specs(method)
        if not specs:
            continue
        try:
            hints = typing.get_type_hints(inspect.unwrap(method))
        except Exception:
            hints = {}
        for param_name, spec in specs.items():
            assert isinstance(spec, ParamSpec), (
                f"{vi_cls.__name__}.{method_name}: params[{param_name!r}] "
                f"must be a ParamSpec, got {type(spec).__name__}"
            )
            annotated = hints.get(param_name)
            if annotated in (float, int, str, bool):
                assert spec.type is annotated, (
                    f"{vi_cls.__name__}.{method_name}: params[{param_name!r}] "
                    f"declares type {spec.type.__name__} but the signature "
                    f"annotates {annotated.__name__} — they must agree"
                )


# ── Declaration standard (the capability manifest) ───────────────────────────
# See virtual_instruments/README.md's "declaration standard" and GLOSSARY.md's
# "Declaration standard": one declaration on the decorator feeds the GUI
# widget, the tooltip, and the capability manifest an agent reads. This test is
# what makes every future VI agent-operable the moment its file exists.

# Deliberately EMPTY, and asserted so below: a VI that cannot describe itself
# is an incomplete VI, not an exception to the standard. Adding a name here
# would hide exactly the gap the manifest exists to close.
MANIFEST_EXEMPT_VIS: frozenset[str] = frozenset()


def _return_type_is_numeric(method: object) -> bool:
    """Return True if *method*'s annotated return type includes ``float``.

    Args:
        method: A ``@monitored`` method (possibly wrapped).

    Returns:
        True when the resolved return annotation is ``float`` or a union
        containing it (e.g. ``float | None``), so a unit label is required.
    """
    try:
        hints = typing.get_type_hints(inspect.unwrap(method))
    except Exception:
        return False
    annotation = hints.get("return")
    if annotation is None:
        return False
    if annotation is float:
        return True
    return float in typing.get_args(annotation)


def test_manifest_exemption_list_is_empty() -> None:
    """The declaration standard ships with no exemptions, by construction."""
    assert MANIFEST_EXEMPT_VIS == frozenset(), (
        "The declaration standard has no exemption list — a VI that cannot "
        "describe itself must be completed, not exempted."
    )


@pytest.mark.parametrize("vi_cls", _all_vi_classes(), ids=lambda c: c.__name__)
def test_capability_manifest_is_complete(vi_cls: type) -> None:
    """Every VI declares enough to render a complete capability manifest.

    Three obligations, checked over the VI's whole capability surface:

    1. Every ``@monitored`` field declares a unit — ``""`` only for a
       genuinely dimensionless reading, never omitted — and a description.
       A field whose return type includes ``float`` is a physical quantity,
       so its unit must be non-empty.
    2. Every ``@control`` parameter is covered by a ``ParamSpec`` carrying a
       description. A measurement VI's arming control and reading-loop
       setters get theirs installed from ``measurement_parameters``.
    3. Every UI-group tag names a declared group (also enforced at import;
       asserted here so the suite reports it per VI).
    """
    assert vi_cls.__name__ not in MANIFEST_EXEMPT_VIS

    group_keys = {group.key for group in vi_cls.ui_groups}

    for method_name in get_monitored_methods(vi_cls):
        method = getattr(vi_cls, method_name)
        unit = get_monitored_unit(method)
        assert unit is not None, (
            f"{vi_cls.__name__}.{method_name} declares no unit — every "
            f"@monitored field must declare one, and \"\" (dimensionless) is "
            f"an explicit choice, not the absence of one"
        )
        assert get_monitored_description(method).strip(), (
            f"{vi_cls.__name__}.{method_name} declares no description — a "
            f"monitored field reaches an agent's schema as a name plus this "
            f"sentence"
        )
        if _return_type_is_numeric(method):
            assert unit, (
                f"{vi_cls.__name__}.{method_name} returns a float but declares "
                f"unit=\"\" — a physical quantity needs its SI unit"
            )
        tag = get_ui_group(method)
        assert not tag or tag in group_keys, (
            f"{vi_cls.__name__}.{method_name} is tagged group={tag!r}, which "
            f"names no declared UIGroup"
        )

    for method_name, method in _control_methods(vi_cls).items():
        params = getattr(method, "_control_params", {})
        specs = get_control_specs(method)
        undeclared = sorted(set(params) - set(specs))
        assert not undeclared, (
            f"{vi_cls.__name__}.{method_name} takes {undeclared} with no "
            f"ParamSpec — declare params= on @control (a measurement VI gets "
            f"its arming and reading-loop specs from measurement_parameters)"
        )
        for param_name, spec in specs.items():
            assert spec.description.strip(), (
                f"{vi_cls.__name__}.{method_name}: params[{param_name!r}] "
                f"declares no description"
            )
        tag = get_ui_group(method)
        assert not tag or tag in group_keys, (
            f"{vi_cls.__name__}.{method_name} is tagged group={tag!r}, which "
            f"names no declared UIGroup"
        )


@pytest.mark.parametrize("vi_cls", _all_vi_classes(), ids=lambda c: c.__name__)
def test_ui_groups_name_real_capabilities(vi_cls: type) -> None:
    """Every declared UIGroup member is a @monitored or @control of that VI."""
    capabilities = set(get_monitored_methods(vi_cls)) | set(_control_methods(vi_cls))
    seen_keys: set[str] = set()
    for group in vi_cls.ui_groups:
        assert isinstance(group, UIGroup), (
            f"{vi_cls.__name__}.ui_groups entries must be UIGroup"
        )
        assert group.key not in seen_keys, (
            f"{vi_cls.__name__}.ui_groups declares {group.key!r} twice"
        )
        seen_keys.add(group.key)
        assert group.title.strip(), f"{vi_cls.__name__}: UIGroup {group.key!r} needs a title"
        for member in group.members:
            assert member in capabilities, (
                f"{vi_cls.__name__}.ui_groups[{group.key!r}] names {member!r}, "
                f"which is not a capability of {vi_cls.__name__}"
            )


def test_sweep_axis_specs_are_described() -> None:
    """Every sweep-axis ParamSpec of every procedure declares a description.

    The axis parameters are merged into a procedure's declared parameters
    separately from its three explicit dicts, so
    ``test_procedure_parameter_has_description`` never reached them.
    """
    from i2as.core.sweep_builder import sweep_axis_param_specs

    checked = 0
    for proc_cls in _all_procedure_classes():
        axis = getattr(proc_cls, "sweep_axis", None)
        if axis is None:
            continue
        for name, spec in sweep_axis_param_specs(axis).items():
            assert spec.description.strip(), (
                f"{proc_cls.__name__}: sweep-axis parameter {name!r} declares "
                f"no description"
            )
            checked += 1
    assert checked > 0, "expected at least one procedure with a sweep axis"


# ── UI-group validation at class creation ────────────────────────────────────
# The throwaway VI subclasses below exist only to trip the validation in
# BaseVirtualInstrument.__init_subclass__; they are never built or registered.


def test_dangling_group_tag_fails_at_import() -> None:
    """A group= tag naming no declared UIGroup raises, naming the method."""
    with pytest.raises(ValueError, match="temperature.*group='nowhere'"):

        class DanglingTagVI(BaseVirtualInstrument):
            @monitored(unit="K", description="Sample temperature", group="nowhere")
            def temperature(self) -> float:
                return 0.0


def test_duplicate_group_key_fails_at_import() -> None:
    """Two UIGroups sharing a key raise, naming the key."""
    with pytest.raises(ValueError, match="'heater' twice"):

        class DuplicateKeyVI(BaseVirtualInstrument):
            ui_groups = (
                UIGroup(key="heater", title="Heater", members=("heater_output",)),
                UIGroup(key="heater", title="Heater again", members=("set_heater",)),
            )

            @monitored(unit="%", description="Heater output")
            def heater_output(self) -> float:
                return 0.0

            @control
            def set_heater(self) -> None:
                pass


def test_unknown_group_member_fails_at_import() -> None:
    """A UIGroup naming a method the VI does not have raises, naming it."""
    with pytest.raises(ValueError, match="names member 'set_nothing'"):

        class UnknownMemberVI(BaseVirtualInstrument):
            ui_groups = (
                UIGroup(key="heater", title="Heater", members=("set_nothing",)),
            )


def test_group_tag_must_agree_with_membership() -> None:
    """A member tagged with a different group's key raises, naming the method."""
    with pytest.raises(ValueError, match="set_heater.*group='cooling'"):

        class MismatchedTagVI(BaseVirtualInstrument):
            ui_groups = (
                UIGroup(key="heater", title="Heater", members=("set_heater",)),
                UIGroup(key="cooling", title="Cooling", members=("set_cooling",)),
            )

            @control(group="cooling")
            def set_heater(self) -> None:
                pass

            @control(group="cooling")
            def set_cooling(self) -> None:
                pass


def test_group_member_cannot_belong_to_two_groups() -> None:
    """A method listed by two UIGroups raises, naming the method."""
    with pytest.raises(ValueError, match="method 'set_heater' is a member of both"):

        class SharedMemberVI(BaseVirtualInstrument):
            ui_groups = (
                UIGroup(key="heater", title="Heater", members=("set_heater",)),
                UIGroup(key="cooling", title="Cooling", members=("set_heater",)),
            )

            @control
            def set_heater(self) -> None:
                pass


def test_valid_group_declaration_is_accepted() -> None:
    """The declared shape both worked examples use passes validation."""

    class GroupedVI(BaseVirtualInstrument):
        ui_groups = (
            UIGroup(
                key="heater",
                title="Heater",
                description="Heater readback and control.",
                members=("heater_output", "set_heater"),
            ),
        )

        @monitored(unit="%", description="Heater output", group="heater")
        def heater_output(self) -> float:
            return 0.0

        @control(group="heater")
        def set_heater(self, output_pct: float) -> None:
            pass

    assert get_ui_group(GroupedVI.heater_output) == "heater"
    assert get_ui_group(GroupedVI.set_heater) == "heater"
    assert GroupedVI.ui_groups[0].members == ("heater_output", "set_heater")


# ── Capability-scope standard ─────────────────────────────────────────────────
# See i2as.core.decorators ("@control gains a scope") and GLOSSARY.md's
# "Capability scope" entry: every @control method carries "measurement"
# (default, usable by any plan) or "operation" (usable only by an operation's
# plan; still an ordinary GUI control). These tests make the standard binding
# for every VI, present and future.


def _control_methods(cls: type) -> dict[str, object]:
    """{method_name: method} for every @control method defined on *cls*."""
    methods: dict[str, object] = {}
    for name in dir(cls):
        try:
            attr = getattr(cls, name)
        except AttributeError:
            continue
        if callable(attr) and getattr(attr, "_is_control", False):
            methods[name] = attr
    return methods


@pytest.mark.parametrize("vi_cls", _all_vi_classes(), ids=lambda c: c.__name__)
def test_control_scope_is_a_valid_value(vi_cls: type) -> None:
    """Every @control method's capability scope is "measurement" or "operation"."""
    for method_name, method in _control_methods(vi_cls).items():
        scope = get_control_scope(method)
        assert scope in VALID_CONTROL_SCOPES, (
            f"{vi_cls.__name__}.{method_name} has invalid control scope "
            f"{scope!r}, must be one of {sorted(VALID_CONTROL_SCOPES)}"
        )


def test_reading_setters_are_measurement_scope() -> None:
    """Every reading_setters target method is measurement-scope.

    The reading loop is a procedure-only mechanism ("reading-loop
    setters and the measurement lifecycle are measurement-scope by
    definition, so no existing procedure changes behavior").
    """
    checked = 0
    for vi_cls in _all_vi_classes():
        for param_name, setter_name in vi_cls.reading_setters.items():
            method = getattr(vi_cls, setter_name, None)
            if method is None:
                continue
            scope = get_control_scope(method)
            assert scope == "measurement", (
                f"{vi_cls.__name__}.reading_setters[{param_name!r}] setter "
                f"{setter_name!r} must be measurement-scope, got {scope!r}"
            )
            checked += 1
    assert checked > 0, "expected at least one declared reading_setters entry"


@pytest.mark.parametrize(
    "vi_cls",
    [cls for cls in _all_vi_classes() if issubclass(cls, MeasurementInstrumentBase)],
    ids=lambda c: c.__name__,
)
def test_measurement_lifecycle_is_measurement_scope(vi_cls: type) -> None:
    """A measurement VI's initiate_measurement()/standby() lifecycle is measurement-scope.

    Some concrete VIs keep @control on initiate_measurement() so the GUI can arm it
    manually (MeasurementInstrumentBase docstring); that @control must never
    carry operation scope. standby() is typically undecorated, which
    defaults to measurement-scope — checked here too for completeness.
    """
    for method_name in ("initiate_measurement", "standby"):
        method = getattr(vi_cls, method_name)
        scope = get_control_scope(method)
        assert scope == "measurement", (
            f"{vi_cls.__name__}.{method_name} must be measurement-scope, "
            f"got {scope!r}"
        )


# ── Control-limit coverage (the direct action path's numeric fence) ───────────
# The control-validation standard is only as strong as its coverage: a
# @control parameter nobody remembered to declare is enforced by nothing at
# all. This section converts it from "enforced if declared" into "declared, or
# exempted IN WRITING". Every numeric (float/int) parameter of every @control
# method on every discovered VI must either appear in that VI's
# ``control_limits`` or appear below with a one-line physical reason.
#
# The table is keyed by (declaring class, method, parameter) — the class where
# the method is DEFINED, so an inherited control is written down once — and its
# values are the rationale a reviewer reads. Adding a row is a deliberate act;
# it is not a way to silence the test, because
# ``test_no_stale_control_limit_exemptions`` fails on any row that no longer
# names a real unbounded parameter.
CONTROL_LIMIT_EXEMPTIONS: dict[tuple[str, str, str], str] = {
    # -- Enumerated instrument settings: the value selects a mode, it is not a
    #    physical quantity a range could bound.
    ("Lakeshore335SampleTemperatureControllerVI", "set_curve", "curve"): (
        "Calibration-curve slot index in the controller's own curve table; an "
        "index selects a stored curve and drives no output."
    ),
    ("DCMeasurementBase", "initiate_measurement", "voltmeter_range_V"): (
        "Voltmeter full-scale input range: a receive-side setting that sources "
        "nothing, and the meter clamps it to its nearest supported range."
    ),
    # -- Dimensionless counts.
    ("DCMeasurementBase", "initiate_measurement", "readings_per_point"): (
        "Dimensionless sample count; it costs time, not energy in the sample."
    ),
    # -- Compliance ceilings: these are themselves protective limits. With the
    #    excitation current already bounded, they only decide how much voltage
    #    headroom the source may use to deliver that bounded current.
    ("DCMeasurementBase", "initiate_measurement", "compliance_A"): (
        "The source's own protective ceiling; the sourced current is already "
        "bounded, so this only sets headroom, and the instrument clamps it."
    ),
    # -- Closed-loop tuning constants and open-loop heater drive.
    ("SampleTemperatureControllerVI", "set_heater_output", "output_pct"): (
        "Heater drive as a percentage of the range the controller is set to; "
        "the driver clamps it to 0-100 and the heater range, not this VI, is "
        "what bounds the power available."
    ),
    ("SampleTemperatureControllerVI", "set_pid", "p_K"): (
        "Closed-loop proportional gain: a tuning constant, clamped by the "
        "controller firmware, that commands no setpoint of its own."
    ),
    ("SampleTemperatureControllerVI", "set_pid", "i_min"): (
        "Closed-loop integral time: a tuning constant, clamped by the "
        "controller firmware, that commands no setpoint of its own."
    ),
    ("SampleTemperatureControllerVI", "set_pid", "d_min"): (
        "Closed-loop derivative time: a tuning constant, clamped by the "
        "controller firmware, that commands no setpoint of its own."
    ),
    # -- Camera readout: an enumerated sensor mode and an exposure count.
    ("CameraMeasurementVI", "initiate_measurement", "binning"): (
        "On-sensor binning factor: an enumerated readout mode the camera "
        "either supports or refuses; it selects how pixels are summed and "
        "drives nothing at the sample."
    ),
    ("CameraMeasurementVI", "initiate_measurement", "frames_per_step"): (
        "Dimensionless exposure count averaged per point; it costs time, "
        "not light or energy at the sample."
    ),
}


def _exemption_key(cls: type, method_name: str, param_name: str) -> tuple | None:
    """Return the exemption row covering this parameter, or ``None``.

    Matched along ``cls``'s MRO, so a control declared on a base class is
    written down ONCE even when concrete VIs override the method to implement
    it (``DCMeasurementBase.initiate_measurement`` and its two subclasses):
    the parameter, its unit and the physical reason are the base's, not each
    implementation's.
    """
    for base in cls.__mro__:
        key = (base.__name__, method_name, param_name)
        if key in CONTROL_LIMIT_EXEMPTIONS:
            return key
    return None


def _unbounded_numeric_control_params(cls: type) -> list[tuple[str, str]]:
    """Return ``(method, param)`` for every numeric @control param without a limit."""
    found: list[tuple[str, str]] = []
    for method_name, method in _control_methods(cls).items():
        for param_name, info in getattr(method, "_control_params", {}).items():
            if info.get("type") not in (float, int):
                continue
            if param_name in cls.control_limits.get(method_name, {}):
                continue
            found.append((method_name, param_name))
    return found


@pytest.mark.parametrize("vi_cls", _all_vi_classes(), ids=lambda c: c.__name__)
def test_every_numeric_control_param_is_bounded_or_exempt(vi_cls: type) -> None:
    """Every numeric @control parameter is in control_limits or exempted in writing.

    The highest-leverage half of the control-validation standard: declaring a
    limit is enforced by ``BaseVirtualInstrument._make_limit_wrapper``, but
    nothing used to notice a parameter for which no limit was declared at all.
    A new VI now either bounds its numeric controls or writes down, here, the
    physical reason a range cannot bound them.
    """
    unbounded: list[str] = []
    for method_name, param_name in _unbounded_numeric_control_params(vi_cls):
        if _exemption_key(vi_cls, method_name, param_name) is None:
            unbounded.append(f"{method_name}({param_name})")
    assert not unbounded, (
        f"{vi_cls.__name__}: numeric @control parameter(s) {sorted(unbounded)} "
        f"are neither bounded by control_limits nor listed in "
        f"CONTROL_LIMIT_EXEMPTIONS. Declare the limit (its value belongs in the "
        f"config's init_params, never in code), or add an exemption row with a "
        f"one-line physical reason."
    )


def test_no_stale_control_limit_exemptions() -> None:
    """Every exemption row still names a real, still-unbounded @control parameter.

    Keeps the table honest in both directions: a parameter that gained a limit
    (or was renamed or deleted) must lose its exemption, so the list stays as
    short as the code allows rather than accumulating dead prose.
    """
    live: set[tuple] = set()
    for vi_cls in _all_vi_classes():
        for method_name, param_name in _unbounded_numeric_control_params(vi_cls):
            key = _exemption_key(vi_cls, method_name, param_name)
            if key is not None:
                live.add(key)
    stale = sorted(set(CONTROL_LIMIT_EXEMPTIONS) - live)
    assert not stale, (
        f"CONTROL_LIMIT_EXEMPTIONS rows {stale} no longer name an unbounded "
        f"numeric @control parameter — delete them."
    )
    for key, rationale in CONTROL_LIMIT_EXEMPTIONS.items():
        assert rationale.strip(), f"exemption {key} carries no rationale"


# ── The setpoint-parameter convention ────────────────────────────────────────


@pytest.mark.parametrize(
    "vi_cls",
    [cls for cls in _all_vi_classes() if issubclass(cls, RampableVI)],
    ids=lambda c: c.__name__,
)
def test_rampable_vi_declares_no_motion_phases_as_a_frozenset(vi_cls: type) -> None:
    """Every rampable VI's ``no_motion_phases`` is a frozenset of phase names.

    The no-motion phase declaration is read off the ramp-status snapshot by
    the stall detector every tick, so it must be an immutable set of strings
    on the class — a list or a bare string would be iterated as characters
    or mutated from a shared default.
    """
    phases = vi_cls.no_motion_phases
    assert isinstance(phases, frozenset), (
        f"{vi_cls.__name__}.no_motion_phases must be a frozenset, got {type(phases).__name__}"
    )
    assert all(isinstance(phase, str) and phase for phase in phases), (
        f"{vi_cls.__name__}.no_motion_phases must contain non-empty strings: {phases!r}"
    )


@pytest.mark.parametrize(
    "vi_cls",
    [cls for cls in _all_vi_classes() if issubclass(cls, RampableVI)],
    ids=lambda c: c.__name__,
)
def test_rampable_vi_declares_exactly_one_setpoint_control(vi_cls: type) -> None:
    """Every rampable VI names its enveloped quantity ``target_*`` on one @control.

    The setpoint-parameter convention (``core.plan.SETPOINT_PARAM_PREFIX``):
    the session envelope binds a manual action by asking which of the action's
    keyword arguments carries the VI's setpoint — the same quantity
    ``start_ramp(target)`` takes. That answer must be unambiguous, so a
    rampable VI declares exactly one such parameter, and it must be bounded by
    ``control_limits`` too (the envelope narrows the setup's limit; there has
    to be a limit to narrow).
    """
    setpoints = [
        (method_name, param_name)
        for method_name, method in _control_methods(vi_cls).items()
        for param_name in getattr(method, "_control_params", {})
        if param_name.startswith(SETPOINT_PARAM_PREFIX)
    ]
    assert len(setpoints) == 1, (
        f"{vi_cls.__name__} declares {len(setpoints)} '{SETPOINT_PARAM_PREFIX}*' "
        f"@control parameter(s) {sorted(setpoints)}; a rampable VI must declare "
        f"exactly one — it is how the session envelope binds a manual action"
    )
    method_name, param_name = setpoints[0]
    assert param_name in vi_cls.control_limits.get(method_name, {}), (
        f"{vi_cls.__name__}.{method_name}({param_name}) is the setpoint "
        f"capability but is not bounded by control_limits — the envelope "
        f"narrows the setup's limit, so a limit must exist to narrow"
    )


# ── The excitation ceiling reaches every shipped setup ───────────────────────

#: ``control_limits`` limit names a config's ``max_source_current_A`` populates
#: (``EXCITATION_CURRENT_LIMIT``, the current-sourcing VIs). Discovered
#: through ``control_limits`` rather than by naming VI classes, so a new VI
#: reusing the limit is covered the moment its config entry exists; a VI that
#: derives a second limit from the same ceiling adds its name here.
MAX_SOURCE_CURRENT_LIMITS = frozenset({EXCITATION_CURRENT_LIMIT})


def _declared_limit_names(vi_cls: type) -> set[str]:
    """Return every limit name ``vi_cls.control_limits`` references."""
    return {
        limit_name
        for param_map in vi_cls.control_limits.values()
        for limit_name in param_map.values()
    }


@pytest.mark.parametrize(
    "config_name, vi_name, vi_cls, init_params",
    [
        spec
        for spec in _vi_specs_from_configs()
        if _declared_limit_names(spec[2]) & MAX_SOURCE_CURRENT_LIMITS
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_shipped_config_bounds_the_excitation_current(
    config_name: str, vi_name: str, vi_cls: type, init_params: dict
) -> None:
    """Every shipped config gives each excitation-sourcing VI a finite ceiling.

    Covers a REAL setup too, not only the sim-buildable ones: the VI is
    constructed with stand-in drivers straight from the config's own
    ``init_params``, so a config that actually drives current through a
    mounted sample is checked without hardware. A missing
    ``max_source_current_A`` leaves the VI able to source anything its
    instrument can deliver, which is exactly the hazard this step closes.
    """
    from unittest.mock import MagicMock

    assert MAX_SOURCE_CURRENT_KEY in init_params, (
        f"{config_name}/{vi_name} ({vi_cls.__name__}) sources excitation "
        f"current but its init_params declare no '{MAX_SOURCE_CURRENT_KEY}'. "
        f"The ceiling is a property of this setup's wiring, so it belongs in "
        f"devices.yaml, never in the VI."
    )

    class _RecordingDrivers(dict):
        def __missing__(self, role: str) -> MagicMock:
            driver = MagicMock(name=f"driver:{role}")
            self[role] = driver
            return driver

    vi = vi_cls(_RecordingDrivers(), **init_params)
    for limit_name in _declared_limit_names(vi_cls) & MAX_SOURCE_CURRENT_LIMITS:
        assert limit_name in vi._limits, (
            f"{config_name}/{vi_name}: '{limit_name}' was never populated"
        )
        _lo, hi = vi._limits[limit_name]
        assert hi is not None and hi > 0, (
            f"{config_name}/{vi_name}: '{limit_name}' upper bound is {hi!r} — "
            f"'{MAX_SOURCE_CURRENT_KEY}' did not reach the VI"
        )


# ── Lifecycle-state standard (BaseVirtualInstrument.lifecycle_state) ─────────
# See BaseVirtualInstrument's "Lifecycle-state standard": what an instrument
# is DOING is data every VI carries and every client RENDERS, never something
# a client reconstructs from whichever actions it happened to witness. These
# tests make the standard binding for every present and future VI over every
# buildable sim config, so a new VI is covered the moment a config names it.


@pytest.mark.parametrize("config_dir", _sim_config_dirs(), ids=lambda p: p.name)
def test_vi_lifecycle_state_is_declared_and_follows_the_verbs(config_dir: Path) -> None:
    """Every VI starts idle and moves to initiated/standby with its own verbs.

    Rule 1 of the lifecycle-state standard: the verbs own the fact. Built
    silently (rule 1 of the connection-lifecycle standard), so a fresh
    Station's every VI must read ``"idle"``; ``initiate_all()`` /
    ``standby_all()`` then move every one of them, which is what lets an
    emergency's blanket stand-down reach a client as a fact rather than as
    an action nobody saw.
    """
    declared = {member.value for member in LifecycleState}
    station = build_station(str(config_dir))
    vi_names = station.get_vi_names()
    assert vi_names, f"{config_dir.name}: no VI built, nothing to check"

    for vi_name in vi_names:
        state = station.get_vi(vi_name).lifecycle_state()
        assert state == LifecycleState.IDLE.value, (
            f"{config_dir.name}/{vi_name}: a freshly built VI must read "
            f"lifecycle_state() == 'idle' (building the Station commands "
            f"nothing), got {state!r}"
        )

    station.initiate_all()
    for vi_name in vi_names:
        state = station.get_vi(vi_name).lifecycle_state()
        assert state == LifecycleState.INITIATED.value, (
            f"{config_dir.name}/{vi_name}: initiate() left lifecycle_state() "
            f"at {state!r} — the lifecycle-state standard records the state "
            f"in BaseVirtualInstrument's wrap, so this VI either raised or "
            f"replaced the wrapped method"
        )

    station.standby_all()
    for vi_name in vi_names:
        state = station.get_vi(vi_name).lifecycle_state()
        assert state == LifecycleState.STANDBY.value, (
            f"{config_dir.name}/{vi_name}: standby() left lifecycle_state() "
            f"at {state!r} — an instrument stood down must say so, whichever "
            f"path stood it down"
        )
        assert state in declared


@pytest.mark.parametrize("config_dir", _sim_config_dirs(), ids=lambda p: p.name)
def test_vi_lifecycle_state_is_a_pure_read(config_dir: Path) -> None:
    """Reading a VI's lifecycle state sends nothing on the bus.

    Rule 2 of the lifecycle-state standard, watched at the drivers rather
    than trusted: the status-snapshot assembly reads this for every VI on
    every tick and on every state change, so a read that polled would put
    instrument traffic on a reporting path.
    """
    station = build_station(str(config_dir))
    station.initiate_all()
    calls: list[str] = []
    spy_on_station(station, calls)
    for vi_name in station.get_vi_names():
        station.get_vi(vi_name).lifecycle_state()
    assert not calls, (
        f"{config_dir.name}: lifecycle_state() sent instrument traffic "
        f"{sorted(set(calls))} — it must answer from the cached value "
        f"(the lifecycle-state standard's purity rule); a VI that can "
        f"observe the instrument overrides observe_lifecycle_state(), which "
        f"the monitor cycle consults, instead"
    )


# ── The direct action path refuses what is not a capability ──────────────────


@pytest.mark.parametrize("config_dir", _sim_config_dirs(), ids=lambda p: p.name)
def test_execute_vi_action_refuses_non_control_names(config_dir: Path) -> None:
    """Over every VI of every buildable config: only capabilities dispatch.

    Asserts the direct action path's first two checks (see
    ``Station.execute_vi_action()``) for every discovered VI at once: a
    private name is refused, and so is every public method that is neither
    ``@control`` nor one of ``LIFECYCLE_ACTIONS``. Nothing is called on the
    instrument in either case — the refusal happens before dispatch.
    """
    station = build_station(str(config_dir))
    checked_private = 0
    checked_undeclared = 0
    for vi_name in station.get_vi_names():
        vi = getattr(station, vi_name)
        with pytest.raises(I2ASPrivateActionError):
            station.execute_vi_action(vi_name, "_limits")
        checked_private += 1
        for name, member in inspect.getmembers(type(vi), inspect.isfunction):
            if name.startswith("_") or name in LIFECYCLE_ACTIONS:
                continue
            if getattr(getattr(vi, name), "_is_control", False):
                continue
            with pytest.raises(I2ASUndeclaredActionError):
                station.execute_vi_action(vi_name, name)
            checked_undeclared += 1
    assert checked_private > 0 and checked_undeclared > 0, (
        f"{config_dir.name}: discovery found nothing to check "
        f"({checked_private} private, {checked_undeclared} undeclared)"
    )


# ── Measurement-method standard ───────────────────────────────────────────────
# See MeasurementInstrumentBase: every concrete measurement VI is self-describing
# (measurement_parameters / measurement_data_keys / measurement_scalar_columns)
# and implements one uniform lifecycle (data_arrays / initiate_measurement / take_reading /
# standby). These tests make that standard binding for every future measurement
# VI the moment its file exists.

# Superset of sim drivers covering every role any measurement VI asks for. Each
# VI picks the roles it needs (e.g. "source"+"meter" or "main") and ignores the
# rest, so one dict builds every measurement VI without per-class knowledge. Add
# a role here when a new measurement VI introduces a new instrument.
_SIM_MEASUREMENT_DRIVER_CLASSES = {
    "source": f"{PACKAGE}.drivers.sim_keithley_6221.SimKeithley6221",
    "meter": f"{PACKAGE}.drivers.sim_keithley_2182a.SimKeithley2182A",
    "main": f"{PACKAGE}.drivers.sim_camera.SimCamera",
}


def _all_measurement_vi_classes() -> list[type]:
    """Every concrete measurement-method VI class."""
    return [
        cls
        for cls in _all_vi_classes()
        if issubclass(cls, MeasurementInstrumentBase)
    ]


def _build_sim_measurement_drivers() -> dict[str, object]:
    """Fresh sim-driver instances for every role a measurement VI may use."""
    return {
        role: _import_class(dotted)("SIM::CONFORMANCE")
        for role, dotted in _SIM_MEASUREMENT_DRIVER_CLASSES.items()
    }


@pytest.mark.parametrize(
    "vi_cls", _all_measurement_vi_classes(), ids=lambda c: c.__name__
)
def test_measurement_vi_self_description(vi_cls: type) -> None:
    """Every measurement VI declares valid self-describing class attributes."""
    params = vi_cls.measurement_parameters
    assert params, (
        f"{vi_cls.__name__} declares no measurement_parameters — a measurement "
        f"method must own its GUI-facing knobs as ParamSpecs"
    )
    for name, spec in params.items():
        assert isinstance(spec, ParamSpec), (
            f"{vi_cls.__name__}.measurement_parameters['{name}'] must be a "
            f"ParamSpec, got {spec!r}"
        )

    assert vi_cls.measurement_data_keys, (
        f"{vi_cls.__name__} declares no measurement_data_keys — it must name the "
        f"arrays take_reading() returns"
    )

    # selector_label is optional (falls back to display_label in the GUI) but,
    # when declared, must be a plain string — it labels the method drop-down.
    assert isinstance(vi_cls.selector_label, str), (
        f"{vi_cls.__name__}.selector_label must be a str (the short "
        f"method-selection drop-down label), got {vi_cls.selector_label!r}"
    )

    for name, dtype in vi_cls.measurement_scalar_columns.items():
        assert dtype in ("float", "int"), (
            f"{vi_cls.__name__}.measurement_scalar_columns['{name}'] dtype "
            f"{dtype!r} must be 'float' or 'int'"
        )


@pytest.mark.parametrize(
    "vi_cls", _all_measurement_vi_classes(), ids=lambda c: c.__name__
)
def test_measurement_vi_externally_owned_parameters_contract(vi_cls: type) -> None:
    """externally_owned_parameters obeys the externally-configured standard.

    A ``frozenset`` (or at least a ``set`` — the type the standard names on
    ``MeasurementInstrumentBase``), and every name in it must be an existing
    ``measurement_parameters`` key. The empty default (no external-
    configuration support) always passes trivially.
    """
    assert isinstance(vi_cls.externally_owned_parameters, (frozenset, set)), (
        f"{vi_cls.__name__}.externally_owned_parameters must be a frozenset "
        f"(or set), got {vi_cls.externally_owned_parameters!r}"
    )
    unknown = sorted(
        set(vi_cls.externally_owned_parameters) - set(vi_cls.measurement_parameters)
    )
    assert not unknown, (
        f"{vi_cls.__name__}.externally_owned_parameters names {unknown}, "
        f"which {'is' if len(unknown) == 1 else 'are'} not (an) existing "
        f"measurement_parameters key(s)"
    )


@pytest.mark.parametrize(
    "vi_cls", _all_measurement_vi_classes(), ids=lambda c: c.__name__
)
def test_measurement_vi_mean_error_array_convention(vi_cls: type) -> None:
    """Every array-valued quantity gets a companion mean + SEM scalar column.

    The mean/error/array convention (see ``MeasurementInstrumentBase`` and
    ``quantity_columns()``): every ``measurement_data_keys`` entry names a
    raw-sample array and MUST end in ``"_array"``; its base quantity name
    (the key with that suffix stripped) MUST have both a bare-name mean and
    a ``"_error"`` (SEM) scalar column in ``measurement_scalar_columns``,
    both dtype ``"float"``. Binding for every future measurement VI, not
    just the ones that adopted the convention explicitly.
    """
    for array_key in vi_cls.measurement_data_keys:
        assert array_key.endswith("_array"), (
            f"{vi_cls.__name__}.measurement_data_keys entry {array_key!r} "
            f"must end in '_array' per the mean/error/array convention"
        )
        base_name = array_key[: -len("_array")]
        assert vi_cls.measurement_scalar_columns.get(base_name) == "float", (
            f"{vi_cls.__name__}.measurement_scalar_columns must declare "
            f"{base_name!r} (the mean, dtype 'float') for array quantity "
            f"{array_key!r}"
        )
        error_name = f"{base_name}_error"
        assert vi_cls.measurement_scalar_columns.get(error_name) == "float", (
            f"{vi_cls.__name__}.measurement_scalar_columns must declare "
            f"{error_name!r} (the SEM, dtype 'float') for array quantity "
            f"{array_key!r}"
        )


@pytest.mark.parametrize(
    "vi_cls", _all_measurement_vi_classes(), ids=lambda c: c.__name__
)
def test_measurement_vi_lifecycle_methods(vi_cls: type) -> None:
    """Every measurement VI implements the lifecycle; take_reading takes no args."""
    for method_name in ("data_arrays", "initiate_measurement", "take_reading", "standby"):
        assert callable(getattr(vi_cls, method_name, None)), (
            f"{vi_cls.__name__} lacks the '{method_name}' lifecycle method"
        )

    required = [
        p
        for p in inspect.signature(vi_cls.take_reading).parameters.values()
        if p.name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert not required, (
        f"{vi_cls.__name__}.take_reading() must take no required arguments "
        f"(everything is fixed at initiate_measurement()), got {[p.name for p in required]}"
    )


@pytest.mark.parametrize(
    "vi_cls", _all_measurement_vi_classes(), ids=lambda c: c.__name__
)
def test_measurement_vi_round_trip(vi_cls: type) -> None:
    """Built from sim drivers, a measurement VI returns exactly what it declares.

    initiate_measurement(**defaults) then take_reading() must yield exactly
    measurement_data_keys (each with the length data_arrays declared) plus every
    measurement_scalar_columns key — the machine check that a measurement method
    is plug-compatible the day its file exists.
    """
    vi = vi_cls(_build_sim_measurement_drivers())
    defaults = {name: spec.default for name, spec in vi_cls.measurement_parameters.items()}

    vi.initiate_measurement(**defaults)
    data = vi.take_reading()

    expected_keys = (
        set(vi_cls.measurement_data_keys)
        | set(vi_cls.measurement_scalar_columns)
        | set(vi_cls.measurement_raw_blocks)
        | set(vi_cls.measurement_image_blocks)
    )
    assert set(data) == expected_keys, (
        f"{vi_cls.__name__}.take_reading() returned keys {sorted(data)}, "
        f"expected {sorted(expected_keys)}"
    )

    # The image-block standard: a declared frame comes back at exactly the
    # declared (height_px, width_px) — passes vacuously for a VI with none.
    for name, image in vi_cls.measurement_image_blocks.items():
        frame = np.asarray(data[name], dtype=np.float64)
        assert frame.shape == image.shape, (
            f"{vi_cls.__name__}.take_reading()[{name!r}] has shape {frame.shape}, "
            f"but measurement_image_blocks declared {image.shape}"
        )

    arrays = vi.data_arrays(defaults)
    assert set(arrays) == set(vi_cls.measurement_data_keys), (
        f"{vi_cls.__name__}.data_arrays() keys {sorted(arrays)} != "
        f"measurement_data_keys {sorted(vi_cls.measurement_data_keys)}"
    )
    for name, length in arrays.items():
        assert len(data[name]) == length, (
            f"{vi_cls.__name__}.take_reading()['{name}'] has length "
            f"{len(data[name])}, but data_arrays declared {length}"
        )

    block_rows = vi.raw_block_row_counts(defaults)
    assert set(block_rows) == set(vi_cls.measurement_raw_blocks), (
        f"{vi_cls.__name__}.raw_block_row_counts() keys {sorted(block_rows)} != "
        f"measurement_raw_blocks {sorted(vi_cls.measurement_raw_blocks)}"
    )
    for name, labels in vi_cls.measurement_raw_blocks.items():
        rows = block_rows[name]
        block = data[name]
        assert len(block) == rows, (
            f"{vi_cls.__name__}.take_reading()['{name}'] has {len(block)} rows, "
            f"but raw_block_row_counts declared {rows}"
        )
        for row in block:
            assert len(row) == len(labels), (
                f"{vi_cls.__name__}.take_reading()['{name}'] row has "
                f"{len(row)} channels, but measurement_raw_blocks declared "
                f"{len(labels)}"
            )

    for name in vi_cls.measurement_scalar_columns:
        value = data[name]
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{vi_cls.__name__}.take_reading()['{name}'] must be a real-number "
            f"scalar, got {value!r}"
        )

    for array_key in vi_cls.measurement_data_keys:
        base_name = array_key[: -len("_array")]
        error = data[f"{base_name}_error"]
        assert error >= 0.0 or math.isnan(error), (
            f"{vi_cls.__name__}.take_reading()['{base_name}_error'] must be "
            f">= 0 (or NaN, when zero samples are valid), got {error!r}"
        )


def _measurement_defaults(vi_cls: type) -> dict:
    """The declared default for every one of a measurement VI's parameters."""
    return {name: spec.default for name, spec in vi_cls.measurement_parameters.items()}


def _held_driver_ids(vi_cls: type, drivers: dict[str, object]) -> set[int]:
    """Identities of the driver instances *vi_cls* actually keeps.

    Every measurement VI is handed the same superset dict, so ``_drivers``
    cannot say which instruments a VI really uses. What it keeps for itself
    can: each VI stores its own (``self._source``, ``self._meter``,
    ``self._main``…), so the instances found among its attributes are exactly
    the ones it will drive — discovered generically, with no per-class table
    to keep in step.
    """
    vi = vi_cls(drivers)
    return {
        id(value)
        for value in vars(vi).values()
        if any(value is driver for driver in drivers.values())
    }


def _shared_instrument_vi_pairs() -> list[tuple[type, type]]:
    """Ordered (first, second) measurement-VI pairs that drive a common instrument."""
    drivers = _build_sim_measurement_drivers()
    held = {cls: _held_driver_ids(cls, drivers) for cls in _all_measurement_vi_classes()}
    return [
        (first, second)
        for first in held
        for second in held
        if first is not second and held[first] & held[second]
    ]


@pytest.mark.parametrize(
    ("first_cls", "second_cls"),
    _shared_instrument_vi_pairs(),
    ids=lambda c: c.__name__,
)
def test_measurement_vi_arms_after_a_shared_instrument_was_left_in_another_mode(
    first_cls: type, second_cls: type
) -> None:
    """A VI must never assume it found its shared instrument idle.

    The **shared-instrument mode discipline** standard (see
    ``virtual_instruments/measurement/README.md``): two measurement VIs can
    be wired to the same physical instrument — two measurement methods over
    one source, say — so the second one to arm meets whatever the first
    left behind. No shipped pair shares an instrument today, so the
    parametrisation is empty until a setup's VIs do; the check arrives with
    them. This test arms the first VI and then, with **no**
    ``standby()`` in between (the abandoned run: a crash, a kill, an agent
    that stopped answering), arms the second on the same driver objects.

    Exactly two outcomes are acceptable, and this is the whole standard:
    either the second VI re-asserts its own mode first and goes on to
    produce the readings it declares, or the instrument refuses and the
    driver says so as a typed ``I2ASInstrumentError`` carrying the
    instrument's own code. What is never acceptable is the third outcome —
    the one that actually happened on hardware in the ``-221`` incident —
    where the write is silently rejected, the VI believes it armed, and
    every number after that is fiction.
    """
    drivers = _build_sim_measurement_drivers()

    first_vi = first_cls(drivers)
    first_vi.initiate_measurement(**_measurement_defaults(first_cls))

    second_vi = second_cls(drivers)
    second_defaults = _measurement_defaults(second_cls)
    try:
        second_vi.initiate_measurement(**second_defaults)
    except I2ASInstrumentError as exc:
        # The other permitted outcome: the instrument refuses, and the driver
        # says so in the instrument's own words. Anything less specific — a
        # bare Exception, or a plain communication error — is not caught here
        # and fails the test, because it does not tell the caller that the
        # instrument REFUSED rather than that the link broke.
        assert exc.code, (
            f"{second_cls.__name__} was refused by the shared instrument "
            f"after {first_cls.__name__}, but the error carries no "
            f"instrument code (the driver error-reporting standard)"
        )
        assert exc.context, (
            f"{second_cls.__name__}'s refusal names no driver call in its "
            f"context — the half the instrument cannot know"
        )
        return

    data = second_vi.take_reading()
    expected_keys = (
        set(second_cls.measurement_data_keys)
        | set(second_cls.measurement_scalar_columns)
        | set(second_cls.measurement_raw_blocks)
    )
    assert set(data) == expected_keys, (
        f"{second_cls.__name__} armed after {first_cls.__name__} left the "
        f"shared instrument in another mode, but take_reading() returned "
        f"{sorted(data)} instead of {sorted(expected_keys)}"
    )
    for name, length in second_vi.data_arrays(second_defaults).items():
        assert len(data[name]) == length, (
            f"{second_cls.__name__}.take_reading()['{name}'] has length "
            f"{len(data[name])} after arming behind {first_cls.__name__}, "
            f"but data_arrays declared {length} — the stale shared state "
            f"changed what the measurement produced"
        )


@pytest.mark.parametrize(
    "vi_cls", _all_measurement_vi_classes(), ids=lambda c: c.__name__
)
def test_measurement_vi_raw_block_names_dont_collide(vi_cls: type) -> None:
    """A raw diagnostic block never shadows a data-key/scalar-column name.

    Per MeasurementInstrumentBase's "Raw diagnostic blocks" standard: blocks
    are deliberately excluded from measurement_data_keys/
    measurement_scalar_columns (so they never appear in a GUI plot-axis
    dropdown); a colliding name would make the two self-descriptions
    ambiguous. Every declared block's channel-label list must also be
    non-empty — an empty block declares a zero-width channel axis, which
    HDF5 cannot allocate meaningfully.
    """
    other_names = set(vi_cls.measurement_data_keys) | set(vi_cls.measurement_scalar_columns)
    for block_name, labels in vi_cls.measurement_raw_blocks.items():
        assert block_name not in other_names, (
            f"{vi_cls.__name__}.measurement_raw_blocks name {block_name!r} "
            f"collides with a measurement_data_keys/measurement_scalar_columns "
            f"name"
        )
        assert labels, (
            f"{vi_cls.__name__}.measurement_raw_blocks[{block_name!r}] must "
            f"be a non-empty channel-label list"
        )
    for block_name, image in vi_cls.measurement_image_blocks.items():
        assert isinstance(image, ImageBlock), (
            f"{vi_cls.__name__}.measurement_image_blocks[{block_name!r}] must be "
            f"an ImageBlock, got {image!r}"
        )
        assert block_name not in other_names and block_name not in vi_cls.measurement_raw_blocks, (
            f"{vi_cls.__name__}.measurement_image_blocks name {block_name!r} "
            f"collides with another declared column or block"
        )


# ── Session-model standard (L6) ───────────────────────────────────────────────
# Every dataclass in i2as.session.models follows the tolerant-parse
# contract (see the module docstring and i2as/session/README.md):
# constructs from defaults alone, to_dict() is JSON-safe, from_dict() accepts
# junk without raising and round-trips to_dict() output. A new model in
# models.py is covered the moment the class exists.


def _session_model_classes() -> list[type]:
    import dataclasses

    from i2as.session import models

    return [
        obj
        for name, obj in vars(models).items()
        if not name.startswith("_")
        and isinstance(obj, type)
        and dataclasses.is_dataclass(obj)
        # Only models defined here — not the eager-validating core.plan types
        # (ExperimentEnvelope/EnvelopeBound) the module re-imports.
        and obj.__module__ == models.__name__
    ]


@pytest.mark.parametrize("model_cls", _session_model_classes(), ids=lambda c: c.__name__)
def test_session_model_constructs_from_defaults(model_cls: type) -> None:
    """Every session model constructs with no arguments."""
    model_cls()


@pytest.mark.parametrize("model_cls", _session_model_classes(), ids=lambda c: c.__name__)
def test_session_model_dict_contract(model_cls: type) -> None:
    """to_dict()/from_dict() exist, are JSON-safe, and round-trip defaults."""
    import json as json_module

    instance = model_cls()
    assert hasattr(instance, "to_dict") and hasattr(model_cls, "from_dict"), (
        f"{model_cls.__name__} must implement to_dict()/from_dict()"
    )
    payload = instance.to_dict()
    json_module.dumps(payload)  # JSON-safe or this raises
    assert model_cls.from_dict(payload) == instance


@pytest.mark.parametrize("model_cls", _session_model_classes(), ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    "junk", [None, 42, "text", [], {"bogus_key": object}], ids=type
)
def test_session_model_from_dict_tolerates_junk(model_cls: type, junk) -> None:
    """from_dict() never raises on junk input — it degrades to defaults."""
    result = model_cls.from_dict(junk)
    assert isinstance(result, model_cls)


# ── ELN adapter standard (L6, i2as/session/eln/) ──────────────────────────
# The adapter contract written at the top of i2as/session/eln/adapter.py:
# one concrete ElnAdapter per backend module, constructed from a single plain
# settings mapping, declaring a backend id and its capabilities, and exposing
# EXACTLY the contract's methods so any adapter substitutes for any other. A
# new backend module is covered the moment the file exists.


def _eln_adapter_classes() -> list[type]:
    """Every concrete ElnAdapter subclass in i2as.session.eln."""
    import i2as.session.eln as eln_pkg
    from i2as.session.eln.adapter import ElnAdapter

    classes: list[type] = []
    for mod_info in pkgutil.iter_modules(eln_pkg.__path__):
        module = importlib.import_module(f"{PACKAGE}.session.eln.{mod_info.name}")
        for cls in _public_classes(module):
            if (
                issubclass(cls, ElnAdapter)
                and cls is not ElnAdapter
                and cls.__module__ == module.__name__
            ):
                classes.append(cls)
    return classes


def _eln_dict_dataclasses() -> list[type]:
    """Every to_dict()-carrying dataclass defined in i2as.session.eln."""
    import i2as.session.eln as eln_pkg

    classes: list[type] = []
    for mod_info in pkgutil.iter_modules(eln_pkg.__path__):
        module = importlib.import_module(f"{PACKAGE}.session.eln.{mod_info.name}")
        for name, obj in vars(module).items():
            if name.startswith("_") or not isinstance(obj, type):
                continue
            if not dataclasses.is_dataclass(obj) or obj.__module__ != module.__name__:
                continue
            if hasattr(obj, "to_dict"):
                classes.append(obj)
    return classes


@pytest.mark.parametrize("adapter_cls", _eln_adapter_classes(), ids=lambda c: c.__name__)
def test_eln_adapter_public_api_is_exactly_the_contract(adapter_cls: type) -> None:
    """An adapter adds no public method and drops none — full substitutability."""
    from i2as.session.eln.adapter import ElnAdapter

    contract = _public_api(ElnAdapter)
    actual = _public_api(adapter_cls)
    assert contract.keys() == actual.keys(), (
        f"{adapter_cls.__name__} must expose exactly the ElnAdapter contract: "
        f"missing={sorted(contract.keys() - actual.keys())}, "
        f"extra={sorted(actual.keys() - contract.keys())} — queuing, retry, and "
        f"backend-specific helpers belong in the outbox or behind a private name"
    )
    for method in contract:
        expected = list(contract[method].parameters)
        got = list(actual[method].parameters)
        assert expected == got, (
            f"{adapter_cls.__name__}.{method}{actual[method]} does not match the "
            f"contract ElnAdapter.{method}{contract[method]}"
        )


@pytest.mark.parametrize("adapter_cls", _eln_adapter_classes(), ids=lambda c: c.__name__)
def test_eln_adapter_constructs_from_a_plain_settings_mapping(adapter_cls: type) -> None:
    """``__init__(self, settings, ...)`` — one settings mapping, nothing else required.

    The analogue of the driver contract's one-resource-string rule: everything
    a backend needs comes from the mapping, so the publisher can build any
    adapter from the user-level settings file alone.
    """
    params = [
        p
        for p in inspect.signature(adapter_cls.__init__).parameters.values()
        if p.name != "self"
    ]
    assert params, f"{adapter_cls.__name__}.__init__ must take a settings mapping"
    assert params[0].name == "settings", (
        f"{adapter_cls.__name__}.__init__'s first argument must be named "
        f"'settings', got {params[0].name!r}"
    )
    required = [p for p in params[1:] if p.default is inspect.Parameter.empty]
    assert not required, (
        f"{adapter_cls.__name__}.__init__ requires {[p.name for p in required]} "
        f"beyond the settings mapping; make them optional (e.g. an injectable "
        f"transport) so the publisher can build the adapter from settings alone"
    )
    adapter_cls({})  # constructs from a plain dict, with no network touched


@pytest.mark.parametrize("adapter_cls", _eln_adapter_classes(), ids=lambda c: c.__name__)
def test_eln_adapter_declares_backend_and_capabilities(adapter_cls: type) -> None:
    """``backend`` is a lowercase identifier and ``capabilities`` is declared."""
    from i2as.session.eln.adapter import ElnCapabilities

    backend = adapter_cls.backend
    assert backend and backend == backend.lower() and backend.isidentifier(), (
        f"{adapter_cls.__name__}.backend must be a non-empty lowercase "
        f"identifier, got {backend!r}"
    )
    assert isinstance(adapter_cls.capabilities, ElnCapabilities), (
        f"{adapter_cls.__name__}.capabilities must be an ElnCapabilities — "
        f"callers branch on the flags, never on the backend name"
    )


def test_eln_package_has_a_sim_twin() -> None:
    """The ``sim_`` rule, applied to notebooks: one in-memory twin of the contract.

    Because the contract fixes the public API exactly (see the test above),
    every backend's adapter surface is identical, so ONE sim twin stands in
    for all of them; a backend's own HTTP dialect is faked one level lower, at
    its injectable transport.
    """
    from i2as.session.eln.adapter import ElnAdapter
    from i2as.session.eln.sim_eln import SimElnAdapter

    assert issubclass(SimElnAdapter, ElnAdapter)
    assert SimElnAdapter in _eln_adapter_classes()


@pytest.mark.parametrize("model_cls", _eln_dict_dataclasses(), ids=lambda c: c.__name__)
def test_eln_dataclass_dict_contract(model_cls: type) -> None:
    """Every persisted ELN dataclass round-trips and tolerates junk."""
    instance = model_cls()
    payload = instance.to_dict()
    json.dumps(payload)  # JSON-safe or this raises
    assert hasattr(model_cls, "from_dict"), (
        f"{model_cls.__name__} has to_dict() but no from_dict()"
    )
    assert model_cls.from_dict(payload) == instance
    for junk in (None, 42, "text", [], {"bogus_key": object}):
        assert isinstance(model_cls.from_dict(junk), model_cls), (
            f"{model_cls.__name__}.from_dict({junk!r}) must degrade to defaults"
        )


def test_eln_rendered_body_is_self_contained_html() -> None:
    """A rendered entry body pulls in nothing from outside the notebook.

    No script, no stylesheet, no image, no external URL — so the entry renders
    identically in the notebook, in an export, and in a test snapshot.
    """
    from i2as.session.eln.templates import render_run_body

    body = render_run_body(
        {
            "run_id": "r-1",
            "procedure": "Field Sweep",
            "kind": "run",
            "params": {"field_T": 1.0, "note": "<b>escape me</b>"},
            "started_utc": "2026-01-01T00:00:00+00:00",
            "finished_utc": "2026-01-01T01:00:00+00:00",
            "status": "done",
        },
        experiment_id="exp-1",
        experiment_title="Sample A",
        setup={"config_name": "sim", "instruments": {"magnet": {"model": "sim"}}},
        data_path="/data/exp-1/data/r-1.h5",
    )
    lowered = body.lower()
    for forbidden in ("<script", "<link", "<img", "<iframe", "http://", "https://"):
        assert forbidden not in lowered, (
            f"rendered ELN body must be self-contained, found {forbidden!r}"
        )
    assert "<b>escape me</b>" not in body, "rendered ELN body must escape every value"


# ── Maintenance-log kind standard (L6) ────────────────────────────────────────
# Every declared LogKindSpec (i2as.session.maintenance_log.DECLARED_LOG_KINDS)
# must have a valid key, a title, and a non-empty ordered field schema of
# ParamSpecs — see LogKindSpec's docstring. A new log kind is covered the
# moment it's added to the registry, no
# test needs to be written for it. ParamSpec.__post_init__ already enforces at
# construction that every field's default matches its declared type, so a
# LogKindSpec that imports at all already has a usable default per field.


@pytest.mark.parametrize("kind_key", sorted(DECLARED_LOG_KINDS), ids=lambda k: k)
def test_log_kind_spec_is_valid(kind_key: str) -> None:
    """Every declared log kind has a valid key and a ParamSpec field schema."""
    spec = DECLARED_LOG_KINDS[kind_key]
    assert spec.key == kind_key, (
        f"DECLARED_LOG_KINDS[{kind_key!r}] must be registered under its own key, "
        f"got LogKindSpec.key={spec.key!r}"
    )
    assert spec.key and spec.key.isidentifier() and spec.key == spec.key.lower(), (
        f"LogKindSpec.key {spec.key!r} must be a non-empty lowercase identifier"
    )
    assert spec.title, f"LogKindSpec({spec.key!r}) must declare a non-empty title"
    assert spec.fields, f"LogKindSpec({spec.key!r}) declares no fields"
    for name, field_spec in spec.fields.items():
        assert isinstance(field_spec, ParamSpec), (
            f"LogKindSpec({spec.key!r}).fields[{name!r}] must be a ParamSpec, "
            f"got {field_spec!r}"
        )


@pytest.mark.parametrize(
    "vi_cls", _all_measurement_vi_classes(), ids=lambda c: c.__name__
)
def test_measurement_vi_reading_setters_contract(vi_cls: type) -> None:
    """reading_setters obeys the reading-loop standard.

    Every key names an existing, non-bool measurement parameter; every value
    names a real method of the VI whose signature accepts the parameter under
    its own name. See MeasurementInstrumentBase's "reading loop" section.
    """
    for param_name, setter_name in vi_cls.reading_setters.items():
        spec = vi_cls.measurement_parameters.get(param_name)
        assert spec is not None, (
            f"{vi_cls.__name__}.reading_setters key {param_name!r} is not a "
            f"measurement parameter"
        )
        assert spec.type is not bool, (
            f"{vi_cls.__name__}.reading_setters key {param_name!r} is a bool — "
            f"a bool cannot be looped over a value list"
        )
        setter = getattr(vi_cls, setter_name, None)
        assert callable(setter), (
            f"{vi_cls.__name__}.reading_setters[{param_name!r}] names method "
            f"{setter_name!r}, which the VI does not have"
        )
        sig_params = inspect.signature(setter).parameters
        accepts = param_name in sig_params or any(
            p.kind is p.VAR_KEYWORD for p in sig_params.values()
        )
        assert accepts, (
            f"{vi_cls.__name__}.{setter_name}() must accept the looped "
            f"parameter as a keyword named {param_name!r}"
        )


@pytest.mark.parametrize(
    "vi_cls", _all_measurement_vi_classes(), ids=lambda c: c.__name__
)
def test_measurement_vi_reading_setter_round_trip(vi_cls: type) -> None:
    """Every reading setter reconfigures the reading, never its shape.

    Built from sim drivers and armed with defaults, calling each declared
    setter (with the parameter's default value, as one reading-loop iteration
    would) must leave take_reading() returning exactly the declared keys and
    lengths.
    """
    if not vi_cls.reading_setters:
        pytest.skip(f"{vi_cls.__name__} declares no reading_setters")
    vi = vi_cls(_build_sim_measurement_drivers())
    defaults = {
        name: spec.default for name, spec in vi_cls.measurement_parameters.items()
    }
    vi.initiate_measurement(**defaults)
    arrays = vi.data_arrays(defaults)
    expected_keys = (
        set(vi_cls.measurement_data_keys) | set(vi_cls.measurement_scalar_columns)
    )
    for param_name, setter_name in vi_cls.reading_setters.items():
        getattr(vi, setter_name)(**{param_name: defaults[param_name]})
        data = vi.take_reading()
        assert set(data) == expected_keys, (
            f"{vi_cls.__name__}: after {setter_name}(), take_reading() returned "
            f"{sorted(data)}, expected {sorted(expected_keys)}"
        )
        for name, length in arrays.items():
            assert len(data[name]) == length, (
                f"{vi_cls.__name__}: after {setter_name}(), '{name}' has length "
                f"{len(data[name])}, declared {length}"
            )


# ── Reading-loop standard (BaseVirtualInstrument level, all VI roles) ────────
# reading_setters is a VI-level standard: a measurement VI's sourced current
# and any other VI's per-reading setting are the same loopable-parameter
# concept. Check every VI the sim station builds, whatever its role.

def test_reading_loop_standard_on_sim_station() -> None:
    """Every sim-station VI with reading_setters honours the loop standard.

    For each declared entry: reading_parameters supplies a ParamSpec for the
    key; the setter is a real method accepting the parameter under its own
    name; and a non-measurement participant's reading_safe_off (if declared)
    names a real method.
    """
    station = build_station(SIM_STATION_CONFIG)
    checked = 0
    for vi_name in station.get_vi_names():
        vi = station.get_vi(vi_name)
        specs = vi.reading_parameters
        for param_name, setter_name in vi.reading_setters.items():
            checked += 1
            spec = specs.get(param_name)
            assert isinstance(spec, ParamSpec), (
                f"{vi_name}.reading_parameters must supply a ParamSpec for "
                f"loopable parameter {param_name!r}, got {spec!r}"
            )
            setter = getattr(vi, setter_name, None)
            assert callable(setter), (
                f"{vi_name}.reading_setters[{param_name!r}] names method "
                f"{setter_name!r}, which the VI does not have"
            )
            sig_params = inspect.signature(setter).parameters
            accepts = param_name in sig_params or any(
                p.kind is p.VAR_KEYWORD for p in sig_params.values()
            )
            assert accepts, (
                f"{vi_name}.{setter_name}() must accept the looped parameter "
                f"as a keyword named {param_name!r}"
            )
        if vi.reading_safe_off:
            assert callable(getattr(vi, vi.reading_safe_off, None)), (
                f"{vi_name}.reading_safe_off names method "
                f"{vi.reading_safe_off!r}, which the VI does not have"
            )
    # The sim station must exercise the standard: the DC VI's sourced current.
    assert checked >= 1, "sim station should declare at least one loopable parameter"


# ── Trend check declarations ────────────────────────────────────────────────
# Every TrendCheck declared_checks() returns must name state keys a shipped
# config can actually produce (Station.last_state_flat()) — a typo'd key
# would otherwise mean a check silently and permanently reports "no data".
# Auto-discovering declared_checks() (rather than a fixed list) means a new
# check declared in a future phase is covered by this test the moment it
# exists, with no test change required.


@pytest.mark.parametrize("config_dir", _sim_config_dirs(), ids=lambda p: p.name)
def test_declared_trend_checks_name_real_state_keys(config_dir: Path) -> None:
    """Every declared TrendCheck's `keys` exist in this config's flat state.

    Uses `_sim_config_dirs()`, not `_config_dirs()`: `last_state_flat()`
    reads the monitor-tick cache, which is only populated by `get_state()`
    — a real hardware poll on a config with unreachable real drivers, which
    every other conformance test that calls `get_state()`
    (`test_evaluate_safety_flags_are_declared_in_manifest` et al.) also
    restricts to sim-buildable configs for exactly this reason.

    This is the precise runtime check, but it only reaches the two
    sim-buildable configs. `test_declared_trend_checks_name_derivable_state_keys`
    below covers every shipped config, including any real-hardware setup,
    via the static derivation this test's own key set is verified against.
    """
    from i2as.core.config import read_trends_config
    from i2as.core.trend_checks import declared_checks

    station = build_station(str(config_dir))
    station.get_state()  # populate the monitor-tick cache last_state_flat() reads
    flat_keys = set(station.last_state_flat())
    trends_config = read_trends_config(str(config_dir))

    for check in declared_checks(trends_config):
        for key in check.keys:
            assert key in flat_keys, (
                f"{config_dir.name}: trend check {check.name!r} names key "
                f"{key!r}, which is not in this config's last_state_flat() "
                f"({sorted(flat_keys)})"
            )

    # Every key last_state_flat() actually produced at runtime must also
    # appear in the statically derived candidate set below — if it did not,
    # the static derivation would silently diverge from runtime, and the
    # all-four-configs test that relies on it (below) would not be trustworthy
    # for the two real-hardware configs it cannot exercise this way.
    static_keys = _static_flat_keys(config_dir)
    missing_from_static = flat_keys - static_keys
    assert not missing_from_static, (
        f"{config_dir.name}: last_state_flat() produced key(s) "
        f"{sorted(missing_from_static)} at runtime that "
        f"_static_flat_keys() does not derive — the static derivation used "
        f"to cover the real-hardware configs has drifted from "
        f"Station.last_state_flat()'s actual behaviour"
    )


def _static_flat_keys(config_dir: Path) -> set[str]:
    """Derive the set of flat state keys a config COULD produce, no hardware.

    Mirrors `Station.last_state_flat()`'s key construction
    (`f"{vi_name}_{monitored_method_name}"`) and its exclusions — VIs
    registered `vi_type: measurement` (`station.py`'s
    `last_state_flat()` skips them) and any `_`-prefixed monitored method
    name — using only `devices.yaml` and each VI class's `@monitored`
    methods (`get_monitored_methods()` accepts a class, not just an
    instance, so this needs no station build and no hardware poll).

    This is a static OVER-approximation, not an exact match: unlike
    `last_state_flat()`, it cannot know at import time whether a monitored
    method returns a numeric scalar (excluded here) versus a string (e.g.
    `ramp_status`, excluded by `last_state_flat()` at call time). A key this
    function derives may therefore not actually reach the flat state; the
    reverse — a real flat-state key this function fails to derive — is
    what `test_declared_trend_checks_name_real_state_keys` checks never
    happens, for the two configs it can build.

    Args:
        config_dir: A `i2as/configs/<name>/` directory.

    Returns:
        Every `f"{vi_name}_{method_name}"` candidate key.
    """
    devices = _load_yaml(config_dir / "devices.yaml")
    keys: set[str] = set()
    for vi_name, vi_cfg in devices.get("virtual_instruments", {}).items():
        if vi_cfg.get("vi_type") == "measurement":
            continue
        vi_cls = _import_class(vi_cfg["class"])
        for method_name in get_monitored_methods(vi_cls):
            if method_name.startswith("_"):
                continue
            keys.add(f"{vi_name}_{method_name}")
    return keys


@pytest.mark.parametrize("config_dir", _config_dirs(), ids=lambda p: p.name)
def test_declared_trend_checks_name_derivable_state_keys(config_dir: Path) -> None:
    """Every declared TrendCheck's `keys` exist in this config's DERIVABLE state.

    Covers every shipped config, including a real-hardware setup that
    `test_declared_trend_checks_name_real_state_keys` cannot reach, because
    it needs `get_state()`, a hardware poll. If a site renames the VI a
    trend check names, the check's key
    stops resolving and `summarize()` reports `persisted=False` forever —
    an indeterminate result that publishes nothing, silently and
    permanently, with no signal that coverage was lost. This test moves
    that failure to CI, statically, via `_static_flat_keys()` — no hardware
    needed (`get_monitored_methods()` accepts a class), the same no-hardware
    pattern `test_config_schema` and
    `test_panels_config_names_real_vis_and_controls` already use to cover
    all four configs.
    """
    from i2as.core.config import read_trends_config
    from i2as.core.trend_checks import declared_checks

    derivable_keys = _static_flat_keys(config_dir)
    trends_config = read_trends_config(str(config_dir))

    for check in declared_checks(trends_config):
        for key in check.keys:
            assert key in derivable_keys, (
                f"{config_dir.name}: trend check {check.name!r} names key "
                f"{key!r}, which no VI in this config's devices.yaml can "
                f"produce ({sorted(derivable_keys)})"
            )


# ══════════════════════════════════════════════════════════════════════
# The control contract (core/events.py)
# ══════════════════════════════════════════════════════════════════════
#
# The contract is the typed currency between the engine and its two
# clients, the GUI and the agent. Two properties make it a contract rather
# than a convention, and both are checked here: every message survives a
# JSON round trip unchanged, and the command enumeration is exactly the
# Orchestrator's public command surface — no client can offer an action the
# engine does not have, and no engine command is invisible to a client.


def _contract_specimens() -> dict[str, object]:
    """Build one representative instance of every control-contract type.

    Representative means every field is populated with a non-default value
    of its declared kind — a nested actor, a populated mapping, a tuple
    field, an enum, a float — so the round trip below exercises the actual
    coercions rather than a tower of defaults.

    Returns:
        ``{type name: instance}`` covering every contract type. The keys
        double as the parametrisation ids.
    """
    agent = Actor(kind=ActorKind.AGENT, id="drift-watch", role="operator")
    return {
        "Actor": agent,
        "Command": Command(
            name=CommandName.SUBMIT_VI_ACTION,
            actor=agent,
            args={"vi_name": "magnet_z", "method_name": "start_ramp", "target": 1.5},
            request_id="req-1",
            issued_at=1_700_000_000.5,
        ),
        "Verdict": Verdict(
            request_id="req-1",
            command=CommandName.SUBMIT_VI_ACTION,
            code=VerdictCode.BLOCKED_LIMIT,
            actor=agent,
            reason="target outside the allowed range",
            detail={
                "param": "target",
                "value": 1.5,
                "lo": -1.0,
                "hi": 1.0,
                "limit_name": "max_field_T",
            },
            result=None,
            seq=7,
            ts=1_700_000_001.0,
        ),
        "StateChange": StateChange(
            state="RAMPING",
            previous="IDLE",
            cause="run_started",
            actor=agent,
            request_id="req-1",
            seq=8,
            ts=1_700_000_002.0,
        ),
        "StatusSnapshot": StatusSnapshot(
            state="RAMPING",
            run={"run_id": "r-1", "kind": "procedure", "progress": 0.25},
            instruments={
                "magnet_z": {
                    "availability": "live",
                    "held": False,
                    "lifecycle": "initiated",
                }
            },
            is_monitoring=True,
            pause_pending=True,
            active_run_kind="procedure",
            override_active=True,
            manual_override_expires_at=1_700_000_300.0,
            held_vi_names=("magnet_z",),
            active_ramps=({"vi_name": "magnet_z", "label": "field", "unit": "T"},),
            availabilities={"magnet_z": {"state": "live", "tags": []}},
            vi_faults={"temperature": {"kind": "stale", "acknowledged": False}},
            offline_reason={"dc_measurement": "no response at GPIB0::12::INSTR"},
            envelope_variables={
                "magnet_z": {"param_name": "target_T", "config_max": 9.0}
            },
            seq=9,
            ts=1_700_000_003.0,
        ),
        "MonitoredInfo": MonitoredInfo(
            name="field_T",
            unit="T",
            description="Measured magnetic field at the sample",
            group="coil",
            returns="float",
        ),
        "ControlInfo": ControlInfo(
            name="start_ramp",
            scope="operation",
            panel=False,
            group="coil",
            params=(
                {
                    "name": "target",
                    "kind": "float",
                    "unit": "T",
                    "description": "Field to ramp to",
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "choices": None,
                },
            ),
        ),
        "GroupInfo": GroupInfo(
            key="coil",
            title="Coil",
            description="Field readback and the ramp that changes it.",
            members=("field_T", "start_ramp"),
        ),
        "InstrumentInfo": InstrumentInfo(
            name="magnet_z",
            vi_class="SuperconductingMagnetVI",
            role="system",
            kind="magnet",
            availability=("not_responding",),
            monitored=(
                MonitoredInfo(
                    name="field_T",
                    unit="T",
                    description="Measured magnetic field at the sample",
                    group="coil",
                    returns="float",
                ),
            ),
            controls=(
                ControlInfo(
                    name="start_ramp",
                    scope="operation",
                    panel=False,
                    group="coil",
                    params=(
                        {
                            "name": "target",
                            "kind": "float",
                            "unit": "T",
                            "description": "Field to ramp to",
                            "default": 0.0,
                            "min": -1.0,
                            "max": 1.0,
                            "choices": None,
                        },
                    ),
                ),
            ),
            limits={
                "start_ramp": {
                    "target": {"limit": "max_field_T", "min": -1.0, "max": 1.0}
                }
            },
            ui_groups=(
                GroupInfo(
                    key="coil",
                    title="Coil",
                    description="Field readback and the ramp that changes it.",
                    members=("field_T", "start_ramp"),
                ),
            ),
            safety_flags={"quench": "critical"},
        ),
        "StationInfo": StationInfo(
            setup="sim_cryostat",
            tick_interval_s=3.0,
            instruments=(
                InstrumentInfo(
                    name="magnet_z",
                    vi_class="SuperconductingMagnetVI",
                    role="system",
                    kind="magnet",
                    availability=("not_responding",),
                    monitored=(
                        MonitoredInfo(
                            name="field_T",
                            unit="T",
                            description="Measured magnetic field at the sample",
                            group="coil",
                            returns="float",
                        ),
                    ),
                    controls=(
                        ControlInfo(
                            name="start_ramp",
                            scope="operation",
                            panel=False,
                            group="coil",
                            params=(
                                {
                                    "name": "target",
                                    "kind": "float",
                                    "unit": "T",
                                    "description": "Field to ramp to",
                                    "default": 0.0,
                                    "min": -1.0,
                                    "max": 1.0,
                                    "choices": None,
                                },
                            ),
                        ),
                    ),
                    limits={
                        "start_ramp": {
                            "target": {
                                "limit": "max_field_T",
                                "min": -1.0,
                                "max": 1.0,
                            }
                        }
                    },
                    ui_groups=(
                        GroupInfo(
                            key="coil",
                            title="Coil",
                            description=(
                                "Field readback and the ramp that changes it."
                            ),
                            members=("field_T", "start_ramp"),
                        ),
                    ),
                    safety_flags={"quench": "critical"},
                ),
                InstrumentInfo(
                    name="dc_measurement",
                    vi_class="DCSeparateMeasurementVI",
                    role="measurement",
                    kind="measurement",
                    availability=("connect_failed",),
                ),
            ),
            seq=10,
            ts=1_700_000_004.0,
        ),
        "Readings": Readings(
            values={"magnet_z": {"field_T": 0.5}, "temperature": {"temperature": 4.2}},
            seq=11,
            ts=1_700_000_005.0,
        ),
        "Datapoint": Datapoint(
            run_id="r-1",
            index=3,
            values={"field_T": 0.5, "resistance_ohm": 12.75},
            seq=12,
            ts=1_700_000_006.0,
        ),
        "RunStarted": RunStarted(
            run_id="r-1",
            manifest={"procedure": "FieldSweep", "points": 40},
            actor=agent,
            request_id="req-2",
            seq=13,
            ts=1_700_000_007.0,
        ),
        "RunFinished": RunFinished(
            run_id="r-1",
            status="aborted",
            reason="operator abort",
            manifest={"procedure": "FieldSweep", "points": 40},
            seq=14,
            ts=1_700_000_008.0,
        ),
        "QueueChanged": QueueChanged(
            entries=({"run_id": "r-2", "procedure": "TimeSeries"},),
            actor=agent,
            request_id="req-3",
            seq=15,
            ts=1_700_000_009.0,
        ),
    }


_CONTRACT_SPECIMENS = _contract_specimens()


@pytest.mark.parametrize(
    "specimen", _CONTRACT_SPECIMENS.values(), ids=list(_CONTRACT_SPECIMENS)
)
def test_contract_type_round_trips_through_json(specimen) -> None:
    """Every control-contract type survives a real JSON round trip unchanged.

    ``to_json()`` → ``json.dumps`` → ``json.loads`` → ``from_json()`` must
    return an equal value. This is what lets the same declaration cross a
    thread boundary today and a process boundary later with no second
    contract; a field that is not JSON-safe, or a coercion that does not
    round trip (a tuple that comes back a list, an enum that comes back a
    bare string), fails here rather than at the boundary.
    """
    payload = specimen.to_json()
    wire = json.loads(json.dumps(payload))
    assert type(specimen).from_json(wire) == specimen


@pytest.mark.parametrize(
    "specimen", _CONTRACT_SPECIMENS.values(), ids=list(_CONTRACT_SPECIMENS)
)
def test_contract_type_is_frozen(specimen) -> None:
    """Every control-contract type is immutable once built.

    A message that crosses a boundary must not be editable by either side —
    the receiver holds a value, not a handle on the sender's state.
    """
    assert dataclasses.is_dataclass(specimen)
    assert dataclasses.fields(specimen) is not None
    field_name = dataclasses.fields(specimen)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(specimen, field_name, None)


@pytest.mark.parametrize(
    "specimen", _CONTRACT_SPECIMENS.values(), ids=list(_CONTRACT_SPECIMENS)
)
def test_contract_type_renders_only_json_scalars(specimen) -> None:
    """``to_json()`` bottoms out in JSON scalars — no enum, tuple, or object.

    ``json.dumps`` would accept a ``str`` enum silently, so this checks the
    rendering itself rather than trusting the encoder.
    """

    def check(value, path: str) -> None:
        assert not isinstance(value, Enum), f"{path} is an enum, not its value"
        if isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, str), f"{path}: key {key!r} is not a str"
                check(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                check(item, f"{path}[{index}]")
        else:
            assert value is None or isinstance(value, (str, int, float, bool)), (
                f"{path} is {type(value).__name__}, not a JSON scalar"
            )

    check(specimen.to_json(), type(specimen).__name__)


def test_every_event_type_is_dispatchable_from_its_kind() -> None:
    """``event_from_json()`` rebuilds the right type from the ``kind`` tag.

    The union is tagged so a client can hold one event channel and still
    know what it received. A new event type that forgets to register its
    kind fails here.
    """
    events = [
        specimen
        for specimen in _CONTRACT_SPECIMENS.values()
        if isinstance(specimen, typing.get_args(Event))
    ]
    assert len(events) == len(typing.get_args(Event)), (
        "every member of the Event union needs a specimen above"
    )
    for event in events:
        wire = json.loads(json.dumps(event.to_json()))
        assert "kind" in wire, f"{type(event).__name__} emits no kind discriminator"
        assert event_from_json(wire) == event


def test_operator_sentinel_is_the_human_at_the_gui() -> None:
    """``OPERATOR`` is the default actor every public entry point assumes."""
    assert OPERATOR.kind is ActorKind.OPERATOR
    assert Command(name=CommandName.ACKNOWLEDGE).actor == OPERATOR


def test_verdict_ok_is_derived_from_its_code() -> None:
    """No verdict can report success and a blocking code at the same time."""
    request = Command(name=CommandName.STOP_RAMP)
    assert Verdict(
        request_id=request.request_id, command=request.name, code=VerdictCode.OK
    ).ok
    assert not Verdict(
        request_id=request.request_id,
        command=request.name,
        code=VerdictCode.BLOCKED_CLAIM,
    ).ok


# Public ``Orchestrator`` methods that are deliberately NOT commands. Every
# entry needs a one-line rationale: an unexplained exemption is how a command
# goes missing from one client's surface.
#
# The two public properties (`state`, `pause_pending`) need no entry — a
# command is a call, not an attribute read, so properties are excluded by
# construction and are reads answered from the client's `StatusSnapshot`.
ORCHESTRATOR_NON_COMMANDS: dict[str, str] = {
    # ── Reads: answered from the client's StatusSnapshot mirror, never by
    #    calling into the engine, so they are not part of the command half.
    "is_monitoring": "read: whether the monitor tick is polling",
    "get_operational_status": "read: the latest per-tick status record",
    "active_ramps": "read: the RampRecord list for the ramp tracker",
    "active_run_kind": "read: which kind of run is in flight, if any",
    "availability": "read: one VI's availability",
    "availabilities": "read: every VI's availability",
    "held_vi_names": "read: which VIs a hold-severity condition holds",
    "manual_override_expires_at": "read: when the manual override lapses",
    "offline_reason": "read: why one VI is offline",
    "envelope_variables": "read: each VI's enveloped quantity and setup bounds",
    "override_active": "read: whether a manual override is in force",
    "vi_faults": "read: the current FaultRecord per VI",
    # ── The status mirror's two priming reads: taken once, by whoever
    #    BUILDS the engine, on the engine's own thread, to prime the client
    #    mirror it hands over. Every later value arrives on the event
    #    stream, so no client ever calls either of these.
    "station_info": "read: the station declaration, the mirror's priming read",
    "status_snapshot": "read: this moment's status, the mirror's priming read",
    # ── Process lifecycle: owned by main.py and test teardown, not by a
    #    client. A client that could stop the tick timer could strand a ramp.
    "shutdown": "lifecycle: stops the tick timer at application exit",
    # ── The port itself: submit(Command) dispatches to the commands below,
    #    so it is the surface a command arrives through, never a command.
    "submit": "port: the entry point every Command is dispatched through",
    # ── Broadcast: the run queue lives outside the engine (session/
    #    run_queue.py), so the engine cannot see a client add, remove or
    #    reorder an entry. This asks it to re-emit QueueChanged; it changes
    #    nothing and starts nothing, so it is not an action a client takes.
    "publish_queue": "broadcast: re-emits QueueChanged after a client-side queue change",
}


def _orchestrator_public_methods() -> set[str]:
    """Return every public method defined on ``Orchestrator``.

    Only functions defined on the class itself: Qt signals are class
    attributes, properties are descriptors, and inherited ``QObject``
    machinery is not ours to enumerate, so ``inspect.isfunction`` over
    ``vars()`` is exactly the public method surface.

    Returns:
        The set of method names with no leading underscore.
    """
    from i2as.core.orchestrator import Orchestrator

    return {
        name
        for name, value in vars(Orchestrator).items()
        if not name.startswith("_") and inspect.isfunction(value)
    }


def test_command_name_covers_every_public_orchestrator_command() -> None:
    """``CommandName`` and the Orchestrator's command surface match exactly.

    Diffed in both directions, because both failures are real. A public
    command missing from the enum is an action the agent cannot take and the
    GUI cannot render from the contract; an enum member with no method behind
    it is a tool that dispatches nowhere. Reads and process lifecycle are
    exempt by name in ``ORCHESTRATOR_NON_COMMANDS`` above, each with its
    rationale.

    If this fails on a method you just added: add it to ``CommandName`` if a
    client may call it, or to the exemption table with a reason if it is a
    read or lifecycle plumbing.
    """
    public_methods = _orchestrator_public_methods()
    exempt = set(ORCHESTRATOR_NON_COMMANDS)

    stale_exemptions = exempt - public_methods
    assert not stale_exemptions, (
        f"ORCHESTRATOR_NON_COMMANDS names methods the Orchestrator no longer "
        f"has: {sorted(stale_exemptions)}"
    )

    commands = public_methods - exempt
    declared = {member.value for member in CommandName}

    assert commands - declared == set(), (
        f"public Orchestrator commands missing from CommandName: "
        f"{sorted(commands - declared)}"
    )
    assert declared - commands == set(), (
        f"CommandName members with no public Orchestrator method behind them: "
        f"{sorted(declared - commands)}"
    )


def test_every_command_method_takes_an_actor() -> None:
    """Every command names who asked, defaulting to the operator sentinel.

    The operator sentinel is what makes accountability a value rather than an
    ambient fact, and what lets it be added without touching a single existing
    call site. A command method that forgets it would silently attribute an
    agent's action to the human at the GUI.
    """
    from i2as.core.orchestrator import Orchestrator

    for member in CommandName:
        method = getattr(Orchestrator, member.value)
        parameter = inspect.signature(method).parameters.get("actor")
        assert parameter is not None, (
            f"Orchestrator.{member.value}() takes no actor keyword"
        )
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"Orchestrator.{member.value}()'s actor must be keyword-only"
        )
        assert parameter.default == OPERATOR, (
            f"Orchestrator.{member.value}()'s actor must default to OPERATOR"
        )


# Read accessors from ``ORCHESTRATOR_NON_COMMANDS`` (plus the two public
# properties) that are NOT answered by a ``StatusSnapshot`` field of the same
# name, each with the reason it is not.
SNAPSHOT_UNANSWERED_READS: dict[str, str] = {
    # Answered from the snapshot's ``availabilities`` map, which carries every
    # VI: a per-VI accessor needs no per-VI field of its own.
    "availability": "one VI's slice of the availabilities map",
    # The per-tick troubleshooting record has its own stream (the
    # ``operational_status`` signal and status.jsonl), not the status mirror.
    "get_operational_status": "carried by the operational-status stream",
    # The mirror's two priming reads are what a client is primed WITH; they
    # are the snapshot and the declaration, not fields inside one.
    "status_snapshot": "IS the snapshot — the mirror's priming read",
    "station_info": "the declaration event, primed and then re-emitted",
}


def test_status_snapshot_answers_every_engine_read() -> None:
    """Every read the engine exposes has a ``StatusSnapshot`` field to answer it.

    The verdict standard's other half: a client answers reads from its
    snapshot mirror and never calls into the engine, which only works if the
    snapshot actually carries every read. Diffed by name, so a new accessor
    lands a field or an entry in ``SNAPSHOT_UNANSWERED_READS`` with a reason.
    """
    from i2as.core.orchestrator import Orchestrator

    reads = {
        name
        for name, rationale in ORCHESTRATOR_NON_COMMANDS.items()
        if rationale.startswith("read:")
    }
    reads |= {
        name
        for name, value in vars(Orchestrator).items()
        if not name.startswith("_") and isinstance(value, property)
    }
    exempt = set(SNAPSHOT_UNANSWERED_READS)

    stale = exempt - reads
    assert not stale, (
        f"SNAPSHOT_UNANSWERED_READS names reads the Orchestrator no longer "
        f"has: {sorted(stale)}"
    )

    snapshot_fields = {f.name for f in dataclasses.fields(StatusSnapshot)}
    missing = reads - exempt - snapshot_fields
    assert not missing, (
        f"engine reads with no StatusSnapshot field to answer them: "
        f"{sorted(missing)}"
    )


def test_command_name_values_are_the_method_names() -> None:
    """Each ``CommandName`` value names the method that implements it.

    Dispatch is then a lookup rather than a hand-maintained table, which is
    what keeps the two clients' surfaces from drifting apart.
    """
    from i2as.core.orchestrator import Orchestrator

    for member in CommandName:
        method = getattr(Orchestrator, member.value, None)
        assert inspect.isfunction(method), (
            f"CommandName.{member.name} = {member.value!r} is not a public "
            f"Orchestrator method"
        )


# ══════════════════════════════════════════════════════════════════════════════
# The station declaration snapshot and the capability manifest
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("config_dir", _sim_config_dirs(), ids=lambda p: p.name)
def test_station_declaration_issues_no_instrument_traffic(config_dir: Path) -> None:
    """Describing the station never operates it — not one driver call.

    The standard behind three separate things: `control_param_specs()`'s
    purity rule (`virtual_instruments/base.py`),
    `Station.station_info()`'s promise that a snapshot can be built for an
    offline instrument, and the Orchestrator's position as the sole writer to
    hardware — a bus read from a describe path would sit outside the one tick
    loop that serialises access.

    The station is polled once first, so the monitor-cycle cache a pure
    override reads from is populated, and one VI is disconnected so the
    offline branch — described from the class, with no instance to consult —
    is covered on the same station. Only THEN are the drivers spied, so the
    log holds nothing but what describing does, and the assertion is an empty
    log rather than trust.

    `_sim_config_dirs()` for the same reason every other station-building
    conformance test uses it: the two real-hardware configs cannot be polled.
    """
    station = build_station(str(config_dir))
    station.get_state()  # fill the monitor-cycle cache a pure override reads
    live = station.get_vi_names()
    if live:
        station.disconnect_instrument(live[0])

    calls: list[str] = []
    spy_on_station(station, calls)

    station._invalidate_station_info()
    assert station.station_info().instruments
    assert build_manifest(station)["instruments"]

    assert calls == [], (
        f"{config_dir.name}: building the station declaration called "
        f"{sorted(set(calls))} — a describe path must send nothing to any "
        f"instrument (see control_param_specs()'s purity rule)"
    )


@pytest.mark.parametrize("config_dir", _sim_config_dirs(), ids=lambda p: p.name)
def test_capability_manifest_validates_against_its_schema(config_dir: Path) -> None:
    """Every buildable config renders a manifest matching MANIFEST_SCHEMA.

    The manifest is generated from declarations, so a VI that declares
    something the schema cannot describe breaks here the moment its file is
    configured — which is the point of validating every config rather than
    one.
    """
    station = build_station(str(config_dir))
    manifest = build_manifest(station)
    assert validate_manifest(manifest) == []
    assert json.loads(json.dumps(manifest)) == manifest


@pytest.mark.parametrize("config_dir", _sim_config_dirs(), ids=lambda p: p.name)
def test_capability_manifest_covers_every_configured_vi(config_dir: Path) -> None:
    """No configured instrument is missing from the manifest, live or offline.

    A client builds its whole instrument surface from this one document, so
    an instrument absent from it is an instrument nobody can see.
    """
    station = build_station(str(config_dir))
    described = {entry["name"] for entry in build_manifest(station)["instruments"]}
    assert described == set(station.get_vi_names()) | set(station.offline_vi_names())


@pytest.mark.parametrize("vi_cls", _all_vi_classes(), ids=lambda c: c.__name__)
def test_manifest_renders_every_declared_capability(vi_cls: type) -> None:
    """Every VI's declarations survive the trip into the manifest's shape.

    `test_capability_manifest_is_complete` checks the DECLARATIONS are there;
    this checks the rendering carries all of them, for every VI whose file
    exists — including the ones no shipped config happens to configure, which
    the config-driven tests above cannot reach. Rendered from the class, the
    way the snapshot describes an offline instrument.
    """
    monitored_infos = _monitored_infos(vi_cls)
    control_infos = _control_infos(vi_cls, None)

    assert {info.name for info in monitored_infos} == set(
        get_monitored_methods(vi_cls)
    )
    assert {info.name for info in control_infos} == set(_control_methods(vi_cls))

    for info in monitored_infos:
        method = getattr(vi_cls, info.name)
        assert info.unit == (get_monitored_unit(method) or "")
        assert info.description == get_monitored_description(method)
        assert info.group == get_ui_group(method)

    for info in control_infos:
        method = getattr(vi_cls, info.name)
        assert info.scope == get_control_scope(method)
        assert info.panel == get_control_panel(method)
        assert [param["name"] for param in info.params] == list(
            getattr(method, "_control_params", {})
        )


def test_measurement_arming_controls_render_their_declared_knobs() -> None:
    """Every measurement VI's `initiate_measurement` renders every knob.

    The measurement-method standard installs `measurement_parameters` as that
    control's declared specs; this is what proves the whole chain — the
    install, the ParamSpec rendering, the unit and the enumerated choices —
    reaches the manifest for every measurement VI whose file exists, not just
    the ones a shipped config configures.
    """
    arming_vis = [
        vi_cls
        for vi_cls in _all_vi_classes()
        if issubclass(vi_cls, MeasurementInstrumentBase)
        and getattr(vars(vi_cls).get("initiate_measurement"), "_is_control", False)
    ]
    assert arming_vis, "measurement VIs declare initiate_measurement as a @control"

    for vi_cls in arming_vis:
        arming = {info.name: info for info in _control_infos(vi_cls, None)}[
            "initiate_measurement"
        ]
        rendered = {param["name"]: param for param in arming.params}
        assert set(rendered) == set(vi_cls.measurement_parameters), vi_cls.__name__
        for name, spec in vi_cls.measurement_parameters.items():
            param = rendered[name]
            assert param["kind"] == spec.type.__name__
            assert param["unit"] == spec.unit
            assert param["description"] == spec.description
            assert param["choices"] == (dict(spec.choices) if spec.choices else None)


@pytest.mark.parametrize("vi_cls", _all_vi_classes(), ids=lambda c: c.__name__)
def test_manifest_group_index_matches_the_vis_ui_groups(vi_cls: type) -> None:
    """A VI's manifest groups ARE its `ui_groups` declaration, member for member.

    One declaration, one rendering: the manifest adds no group, renames none,
    and reorders no member.
    """
    instrument = InstrumentInfo(
        name="specimen",
        vi_class=vi_cls.__name__,
        monitored=_monitored_infos(vi_cls),
        controls=_control_infos(vi_cls, None),
        ui_groups=tuple(
            GroupInfo(
                key=group.key,
                title=group.title,
                description=group.description,
                members=tuple(group.members),
            )
            for group in vi_cls.ui_groups
        ),
    )
    entry = _instrument_json(instrument)

    assert [group["key"] for group in entry["groups"]] == [
        group.key for group in vi_cls.ui_groups
    ]
    for rendered, declared in zip(entry["groups"], vi_cls.ui_groups, strict=True):
        assert rendered["title"] == declared.title
        assert rendered["description"] == declared.description
        assert rendered["monitored"] + rendered["controls"] == list(declared.members)

    indexed = [
        name
        for group in entry["groups"]
        for name in group["monitored"] + group["controls"]
    ]
    indexed += entry["ungrouped"]["monitored"] + entry["ungrouped"]["controls"]
    declared_names = [item["name"] for item in entry["monitored"]]
    declared_names += [item["name"] for item in entry["controls"]]
    assert sorted(indexed) == sorted(declared_names)


# ---------------------------------------------------------------------------
# Repository hygiene: the code-reference standard and the responsive-GUI rule
# ---------------------------------------------------------------------------

# Files under the package exempted from the code-reference standard.
#
# Empty by construction, and kept empty by
# ``test_plan_citation_allowlist_is_empty``. A plan document is a dated
# proposal: it gets implemented, superseded, and moved to docs/plans/archive/,
# and a docstring pointing at it rots silently — it says nothing at all to a
# reader who has only the repository checked out. The durable fix is that a
# citation which cannot exist cannot dangle, so there is nothing to exempt:
# name the concept (and, if a pointer helps, the GLOSSARY term, the folder
# README, or the owning base class) instead.
PLAN_CITATION_ALLOWLIST: frozenset[str] = frozenset()

# The plan-document citation forms, each paired with the wording used in the
# failure message.
#
# Deliberately narrow: they match a citation of a *plan document* only. A
# vendor manual's section number ("vendor doc §3.11", "manual §5.2", or a bare
# "§3.11" beside a model number) is a stable external reference that belongs in
# the driver implementing it, and must never trip these — which is why every
# section-number rule requires the word "plan" or a ".md" filename next to the
# "§", never the "§" alone.
_PLAN_CITATION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a docs/plans path", re.compile(r"docs/plans", re.IGNORECASE)),
    ("a plan section number", re.compile(r"\bplans?\s+§", re.IGNORECASE)),
    ("a plan section word", re.compile(r"\bplans?\s+sections?\b", re.IGNORECASE)),
    ("a parenthetical plan citation", re.compile(r"\(\s*plans?\s", re.IGNORECASE)),
    ("a markdown document's section number", re.compile(r"\.md\s*§")),
)

# Comment markers, bullets, and table pipes a wrapped continuation line starts
# with, stripped before the line is joined to its predecessor for matching.
_CONTINUATION_PREFIX = re.compile(r"^[\s#*>|-]*")

# Blocking sleeps: the direct call and the ``from time import sleep`` alias
# that would hide it.
_BLOCKING_SLEEP = re.compile(
    r"\btime\.sleep\s*\(|\bfrom\s+time\s+import\s+[^\n]*\bsleep\b"
)


def _plan_citations(text: str) -> list[tuple[int, str, str]]:
    """Find every plan-document citation in ``text``.

    A citation is regularly broken across two lines by the wrap width
    (``...operation-concurrency-and-error-`` / ``scoping.md §2``), so each line
    is matched together with the following one, stripped of its comment or
    bullet prefix. A match is attributed to the line it starts on; a match that
    starts inside the continuation is left to that line's own turn, so nothing
    is reported twice.

    Args:
        text: The full contents of one source file or README.

    Returns:
        One ``(line_number, rule, excerpt)`` tuple per citation, line numbers
        1-based, in file order.
    """
    lines = text.splitlines()
    citations: list[tuple[int, str, str]] = []
    for index, line in enumerate(lines):
        following = lines[index + 1] if index + 1 < len(lines) else ""
        window = line + " " + _CONTINUATION_PREFIX.sub("", following)
        for rule, pattern in _PLAN_CITATION_RULES:
            for match in pattern.finditer(window):
                if match.start() >= len(line):
                    continue
                excerpt = window[max(0, match.start() - 20) : match.end() + 40]
                citations.append((index + 1, rule, excerpt.strip()))
    return citations


def _standard_scanned_files() -> list[Path]:
    """Return every source file and folder README the standard covers."""
    return sorted({*PACKAGE_DIR.rglob("*.py"), *PACKAGE_DIR.rglob("README.md")})


@pytest.mark.parametrize(
    "text",
    [
        "See ``docs/plans/session-tier-and-terminology.md``, 'Startup wiring'.",
        "the hard status separation (plan §2)",
        "the readiness contract, described in plan section 12",
        "# Immediate finish (plan operation-concurrency-and-error-scoping.md §2)",
        "operation-concurrency-and-error-scoping.md §2's hard status separation",
        "the shared recorder (plan unified-servicing-log-\nand-run-recording.md §3)",
        "the concurrency-scope hook, plan\n§1's Claim",
    ],
    ids=[
        "docs-plans-path",
        "plan-section-sign",
        "plan-section-word",
        "parenthetical-plan",
        "markdown-section-sign",
        "wrapped-filename",
        "wrapped-section-sign",
    ],
)
def test_plan_citation_matcher_flags_a_plan_citation(text: str) -> None:
    """Every shape a plan citation has taken in this repository is caught."""
    assert _plan_citations(text), text


@pytest.mark.parametrize(
    "text",
    [
        "Ramp-rate table from the vendor doc §3.11.",
        "Model 6221 manual §5.2 forbids this command sequence.",
        "§3.11 of the SR830 manual describes the reserve modes.",
        "See ``GLOSSARY.md``'s **Session** for the tier and its layout.",
        "See ``i2as/core/README.md`` for the module rows.",
        "The Orchestrator dispatches the PhasePlan the procedure returns.",
        "Ramps are generators that yield one step per tick; plan ahead.",
        "self._plan_steps holds the StepPlans already dispatched.",
    ],
    ids=[
        "vendor-doc",
        "manual-section",
        "bare-section-sign",
        "glossary-pointer",
        "readme-pointer",
        "phaseplan-word",
        "plan-as-a-verb",
        "plan-in-an-identifier",
    ],
)
def test_plan_citation_matcher_ignores_a_non_plan_reference(text: str) -> None:
    """Vendor manual sections and ordinary prose are not citations."""
    assert not _plan_citations(text), text


def test_plan_citation_allowlist_is_empty() -> None:
    """The code-reference standard ships with no exemptions, by construction."""
    assert PLAN_CITATION_ALLOWLIST == frozenset(), (
        "The code-reference standard has no allowlist — a docstring, comment, "
        "or README row that needs a plan document to be understood must be "
        "rewritten to carry the reasoning itself, not exempted."
    )


def test_no_plan_document_citation_under_i2as() -> None:
    """No source file or folder README cites a document in docs/plans/.

    Plans are dated proposals that get implemented, superseded, and archived;
    a code comment citing one is a pointer that rots silently and says nothing
    to a reader who does not fetch the document. Name the concept instead, and
    point at GLOSSARY.md, the folder README, or the owning base class.
    """
    offenders: list[str] = []
    for path in _standard_scanned_files():
        relative = path.relative_to(PACKAGE_DIR.parent).as_posix()
        if relative in PLAN_CITATION_ALLOWLIST:
            continue
        for line_number, rule, excerpt in _plan_citations(
            path.read_text(encoding="utf-8")
        ):
            offenders.append(f"{relative}:{line_number}: {rule} — {excerpt}")

    assert not offenders, (
        "Plan-document citation(s) under i2as/ — replace each with the "
        "concept it names, pointing at GLOSSARY.md, the folder README, or the "
        "owning base class:\n" + "\n".join(offenders)
    )


def test_no_blocking_sleep_in_gui_sources() -> None:
    """Nothing under i2as/gui/ blocks the Qt event loop with ``time.sleep``.

    The GUI is driven by one QTimer tick on a single thread, so a sleep in a
    widget freezes the window, the tick loop, and every ramp with it. Waiting
    is expressed as a tick-driven state, never as a blocked call.
    """
    offenders: list[str] = []
    for path in sorted((PACKAGE_DIR / "gui").rglob("*.py")):
        relative = path.relative_to(PACKAGE_DIR.parent).as_posix()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _BLOCKING_SLEEP.search(line):
                offenders.append(f"{relative}:{line_number}: {line.strip()}")

    assert not offenders, (
        "Blocking sleep(s) under i2as/gui/ — the GUI shares its one thread "
        "with the tick loop, so express the wait as a tick-driven state "
        "instead:\n" + "\n".join(offenders)
    )


# ── The run queue lives outside the engine ────────────────────────────────────

_ENGINE_QUEUE_ATTRS = ("_procedure_queue",)


def test_no_source_reaches_into_the_engines_queue() -> None:
    """Only ``orchestrator.py`` touches the engine's own run queues.

    The run queue is data in the session layer (GLOSSARY.md's **Run queue**);
    the engine keeps one small list for runs handed to it directly, and it
    PULLS the rest through ``next_procedure()``. A widget or a session module
    reaching into that private list would be pushing runs into the
    engine behind its back — exactly the shared mutable queue this design
    removed, and the seam through which a client could start a run the engine
    did not decide to start.
    """
    offenders: list[str] = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if path.name == "orchestrator.py":
            continue
        relative = path.relative_to(PACKAGE_DIR.parent).as_posix()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if any(attribute in line for attribute in _ENGINE_QUEUE_ATTRS):
                offenders.append(f"{relative}:{line_number}: {line.strip()}")

    assert not offenders, (
        "The engine's private run queues are referenced outside "
        "core/orchestrator.py. Queue through the session layer's RunQueue (or "
        "Orchestrator.queue_procedure) instead:\n"
        + "\n".join(offenders)
    )


# ── The GUI/engine boundary ───────────────────────────────────────────────────
# The status-mirror standard (gui/README.md, GLOSSARY.md's **Status mirror**)
# and the control contract's command half. A widget may SUBMIT commands and
# CONNECT to signals; it may not read the engine, because a read is the one
# thing that cannot cross to an engine that is deep inside one measure().
# These two tests are what keep that true as widgets are added.

#: Engine attributes a widget may touch, beyond the commands themselves: the
#: Qt signals it connects to, and the port a client that speaks the control
#: contract submits through. The two contract channels appear twice: a client
#: CONSUMES them rather than relaying them, so the proxy carries
#: ``verdict_emitted``/``event_emitted`` under the shorter ``verdict``/
#: ``event``, and a widget may connect to whichever name its client offers.
_ENGINE_SIGNALS = frozenset({
    "verdict_emitted",
    "event_emitted",
    "verdict",
    "event",
    "states_updated",
    "monitoring_changed",
    "state_changed",
    "procedure_progress",
    "procedure_finished",
    "run_started",
    "run_finished",
    "error_occurred",
    "error_event",
    "action_blocked",
    "action_succeeded",
    "action_failed",
    "instrument_reconnected",
    "instrument_disconnected",
    "measurement_ready",
    "operational_status",
    "ramps_updated",
    "status_message",
})

#: Non-command engine attributes a named GUI module may still touch, each with
#: the reason. Anything not listed is a violation, including a private one.
_GUI_ENGINE_EXEMPTIONS: dict[str, dict[str, str]] = {
    f"{PACKAGE}/gui/queue_panel.py": {
        "publish_queue": (
            "the queue seam, not a command: the queue is data this panel "
            "owns when no session layer is wired, so the engine cannot see a "
            "change happen and is asked to broadcast it"
        ),
        "next_procedure": (
            "the pull seam a standalone window claims when nobody else has, "
            "so a window with no session layer still has a queue the engine "
            "can pull from — never a way to push a run in"
        ),
        "queue_snapshot": (
            "the other half of that seam: what the engine reads the waiting "
            "entries from for every QueueChanged"
        ),
    },
}


def _engine_attribute_reads(source: str) -> list[tuple[int, str]]:
    """Return every attribute taken off an engine-shaped name in *source*.

    "Engine-shaped" is a name mentioning ``orch``, ``proxy`` or ``engine``:
    the widgets hold the engine (and, later, its proxy) under exactly those
    names, and matching on the name rather than on a type keeps this a pure
    source scan that needs no imports and no Qt.

    Args:
        source: One module's source text.

    Returns:
        ``(line number, attribute name)`` for each access, in file order.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        if isinstance(base, ast.Name):
            owner = base.id
        elif isinstance(base, ast.Attribute):
            owner = base.attr
        else:
            continue
        lowered = owner.lower()
        if any(token in lowered for token in ("orch", "proxy", "engine")):
            found.append((node.lineno, node.attr))
    return found


def test_gui_touches_the_engine_only_through_commands_and_signals() -> None:
    """No read from ``gui/`` into the engine — the status-mirror standard.

    Every widget answers reads from its ``StatusMirror`` and reaches the
    engine only to submit a command (a ``CommandName``, which is exactly the
    method set the proxy exposes) or to connect a signal. A read that slipped
    back in would be a synchronous call into an engine the thread move puts
    on the other side of a boundary, and it would block the window for as
    long as the engine is inside one ``measure()``.

    If this fails on a read you just added: answer it from the mirror. If the
    mirror cannot answer it, the ``StatusSnapshot`` is missing a field.
    """
    allowed = {member.value for member in CommandName} | _ENGINE_SIGNALS | {"submit"}
    offenders: list[str] = []
    for path in sorted((PACKAGE_DIR / "gui").rglob("*.py")):
        relative = path.relative_to(PACKAGE_DIR.parent).as_posix()
        exempt = _GUI_ENGINE_EXEMPTIONS.get(relative, {})
        for line_number, attribute in _engine_attribute_reads(
            path.read_text(encoding="utf-8")
        ):
            if attribute in allowed or attribute in exempt:
                continue
            # OrchestratorState.IDLE and friends are the enum, not the engine.
            if attribute.isupper():
                continue
            offenders.append(f"{relative}:{line_number}: .{attribute}")

    assert not offenders, (
        "GUI access to the engine outside the command set and the signals — "
        "read it from the StatusMirror instead:\n" + "\n".join(sorted(offenders))
    )


def test_gui_never_reaches_into_the_station_for_a_vi() -> None:
    """``Station.get_vi()`` is never called from ``gui/``.

    A VI is an object on the engine's side of the boundary: holding one lets
    a widget call hardware directly, and it cannot cross a thread. The panels
    build from the **Station info** declaration snapshot instead, which says
    everything about an instrument that a panel renders and nothing a client
    cannot hold.
    """
    offenders: list[str] = []
    for path in sorted((PACKAGE_DIR / "gui").rglob("*.py")):
        relative = path.relative_to(PACKAGE_DIR.parent).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_vi"
            ):
                offenders.append(f"{relative}:{node.lineno}: get_vi()")

    assert not offenders, (
        "Station.get_vi() called from i2as/gui/ — build from the "
        "StationInfo declaration the mirror carries instead:\n"
        + "\n".join(offenders)
    )


def test_the_proxy_exposes_every_command_and_nothing_the_engine_lacks() -> None:
    """The three-way contract check, two legs of it: ``CommandName`` ⊆ proxy
    methods ⊆ Orchestrator commands.

    The engine has two clients and the contract is declared once, so neither
    can offer an action the other cannot see. The third leg — the agent
    gateway's tool list — joins this test when the gateway lands; it will be
    the same enumeration a third time.

    If this fails on a command you just added: give the proxy a typed method
    of that name, taking the arguments the engine method takes.
    """
    from i2as.core.orchestrator_proxy import OrchestratorProxy

    declared = {member.value for member in CommandName}
    proxy_methods = {
        name
        for name, value in vars(OrchestratorProxy).items()
        if not name.startswith("_") and inspect.isfunction(value)
    }
    engine_methods = _orchestrator_public_methods()

    assert declared - proxy_methods == set(), (
        f"CommandName members the proxy does not expose: "
        f"{sorted(declared - proxy_methods)}"
    )
    assert declared - engine_methods == set(), (
        f"CommandName members with no Orchestrator method behind them: "
        f"{sorted(declared - engine_methods)}"
    )


def test_every_proxy_command_takes_the_engine_methods_arguments() -> None:
    """A widget swapping the engine for the proxy must not have to adapt.

    Same names, same parameters, in the same order — which is what makes the
    proxy transparent enough for the whole GUI to move behind it in one step
    rather than widget by widget. The proxy returns a ``request_id`` where the
    engine returns the call's own value, which is the one deliberate
    difference: an answer that has to survive a thread boundary cannot be a
    return value.
    """
    from i2as.core.orchestrator import Orchestrator
    from i2as.core.orchestrator_proxy import OrchestratorProxy

    mismatches: list[str] = []
    for member in CommandName:
        engine_params = [
            name
            for name in inspect.signature(
                getattr(Orchestrator, member.value)
            ).parameters
            if name not in ("self", "actor")
        ]
        proxy_params = [
            name
            for name in inspect.signature(
                getattr(OrchestratorProxy, member.value)
            ).parameters
            if name != "self"
        ]
        if engine_params != proxy_params:
            mismatches.append(
                f"{member.value}: engine{engine_params} != proxy{proxy_params}"
            )

    assert not mismatches, (
        "Proxy methods whose parameters do not match the engine's:\n"
        + "\n".join(mismatches)
    )


def test_the_proxy_re_exposes_every_engine_signal() -> None:
    """Every Orchestrator signal a widget can connect to exists on the proxy.

    The passthrough half of transparency: ``orchestrator.states_updated
    .connect(...)`` in a widget becomes ``proxy.states_updated.connect(...)``
    and nothing else moves. The two contract channels are deliberately
    renamed — ``verdict_emitted``/``event_emitted`` become ``verdict``/
    ``event`` — because a client consumes them, it does not relay them.
    """
    from PyQt6.QtCore import pyqtSignal

    from i2as.core.orchestrator import Orchestrator
    from i2as.core.orchestrator_proxy import OrchestratorProxy

    renamed = {"verdict_emitted": "verdict", "event_emitted": "event"}
    engine_signals = {
        name
        for name, value in vars(Orchestrator).items()
        if isinstance(value, pyqtSignal)
    }
    proxy_signals = {
        name
        for name, value in vars(OrchestratorProxy).items()
        if isinstance(value, pyqtSignal)
    }
    expected = {renamed.get(name, name) for name in engine_signals}
    missing = expected - proxy_signals
    assert not missing, f"Engine signals the proxy does not re-expose: {sorted(missing)}"


#: GUI modules allowed to import ``i2as.core.station`` at RUNTIME, and
#: what they take from it. It is a pure config-FILE reader that takes no
#: Station and touches no instrument; it is in that module for historical
#: reasons and moving it is a separate change. Import contract C19 carries
#: the matching ``ignore_imports`` entry.
_RUNTIME_STATION_IMPORTS: dict[str, set[str]] = {
}


def test_gui_imports_the_station_only_for_typing_or_config_helpers() -> None:
    """C19's other half: a ``i2as.core.station`` import under ``gui/`` is
    type-only, or the one named config-validation helper.

    Import contract C19 forbids the dependency outright and lists the
    existing modules in ``ignore_imports`` — import-linter counts an import
    inside ``if TYPE_CHECKING:`` like any other, so the contract alone cannot
    express "types are fine". This is the half that can: every ignored import
    must be inside a type-checking guard, unless it is the config helper
    named above (the YAML-only readers live in ``i2as.core.config``,
    which the GUI may import freely).
    """
    offenders: list[str] = []
    for path in sorted((PACKAGE_DIR / "gui").rglob("*.py")):
        relative = path.relative_to(PACKAGE_DIR.parent).as_posix()
        allowed_names = _RUNTIME_STATION_IMPORTS.get(relative, set())
        tree = ast.parse(path.read_text(encoding="utf-8"))
        guarded = {
            node
            for block in ast.walk(tree)
            if isinstance(block, ast.If) and _is_type_checking_guard(block.test)
            for node in ast.walk(block)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != f"{PACKAGE}.core.station":
                continue
            if node in guarded:
                continue
            for alias in node.names:
                if alias.name not in allowed_names:
                    offenders.append(f"{relative}:{node.lineno}: {alias.name}")

    assert not offenders, (
        "Runtime import(s) of i2as.core.station under i2as/gui/ — put "
        "the name behind `if TYPE_CHECKING:` (the GUI holds a Station only as "
        "a type), or add it to _RUNTIME_STATION_IMPORTS and C19's "
        "ignore_imports with its reason:\n" + "\n".join(offenders)
    )


def _is_type_checking_guard(test: ast.expr) -> bool:
    """Return whether an ``if`` test is a ``TYPE_CHECKING`` guard.

    Args:
        test: The ``If`` node's test expression.

    Returns:
        True for ``TYPE_CHECKING`` and ``typing.TYPE_CHECKING``.
    """
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


# ── Run-source contract ───────────────────────────────────────────────────────
# The "one vocabulary for live and stored runs" standard (core/data_reader.py's
# module docstring, GLOSSARY.md's **Run source**). Implementations are found by
# having the vocabulary's methods rather than by being named, so a third source
# added later — a probe-run view, a remote reader — is held to the same shape
# the moment its file exists.

_RUN_SOURCE_METHODS = ("list_columns", "read_slice", "summary_stats", "read_metadata")


def _run_source_classes() -> list[type]:
    """Every class in i2as.core answering the run-source vocabulary."""
    from i2as.core.data_reader import RunSource

    found: list[type] = []
    for mod_info in pkgutil.iter_modules(i2as.core.__path__):
        module = importlib.import_module(f"{PACKAGE}.core.{mod_info.name}")
        found.extend(
            cls
            for cls in _public_classes(module)
            if cls is not RunSource
            and all(callable(getattr(cls, name, None)) for name in _RUN_SOURCE_METHODS)
        )
    return found


def test_run_source_implementations_are_discovered() -> None:
    """Both known run sources are found, so the check below is not vacuous."""
    assert {"RunHandle", "RunBuffer"} <= {cls.__name__ for cls in _run_source_classes()}


@pytest.mark.parametrize(
    "source_cls", _run_source_classes(), ids=lambda cls: cls.__name__
)
def test_run_source_conformance(source_cls: type) -> None:
    """Every run source answers the vocabulary with the declared signatures.

    A consumer written against `RunSource` must be able to hold either source
    without adapting, which means matching signatures — not merely matching
    method names — plus the `n_points` counter every source reports.
    """
    from i2as.core.data_reader import RunSource

    declared = _public_api(RunSource)
    actual = _public_api(source_cls)
    for name in _RUN_SOURCE_METHODS:
        assert name in actual, f"{source_cls.__name__} is missing {name}()"
        assert list(actual[name].parameters) == list(declared[name].parameters), (
            f"{source_cls.__name__}.{name}{actual[name]} does not take the "
            f"run-source vocabulary's parameters {name}{declared[name]}"
        )
        assert actual[name].return_annotation is not inspect.Signature.empty, (
            f"{source_cls.__name__}.{name}() must declare what it returns"
        )
    assert isinstance(getattr(source_cls, "n_points", None), property), (
        f"{source_cls.__name__}.n_points must be a property reporting how many "
        f"sweep points the source holds"
    )


# ══════════════════════════════════════════════════════════════════════════════
# The agent gateway's permission model
# ══════════════════════════════════════════════════════════════════════════════
#
# The standard is `session/gateway/roles.py`'s module docstring (the matrix)
# and `session/gateway/action_classes.py`'s (the classification). Authority is
# granted by a table row, never by a branch, so these tests check that the
# tables are complete in both directions: nothing an agent can ask for is
# unclassified, and no row names something that no longer exists.


def test_action_class_names_agree_between_the_declaration_and_the_gateway() -> None:
    """`VALID_ACTION_CLASSES` and `ActionClass` carry the same four values.

    The action-class declaration lives in `core/decorators.py`, which imports
    nothing (contract C1), so the names it validates against are plain
    strings; the gateway's permission matrix is keyed by the `ActionClass`
    enum. One is what a VI declares, the other is what decides who may take
    it — if they ever disagree, a control could declare a class no role can
    be granted. This is the check that keeps the two in step.
    """
    from i2as.core.decorators import VALID_ACTION_CLASSES
    from i2as.session.gateway import ActionClass

    assert set(VALID_ACTION_CLASSES) == {member.value for member in ActionClass}


@pytest.mark.parametrize("vi_cls", _all_vi_classes(), ids=lambda c: c.__name__)
def test_every_control_declares_its_action_class(vi_cls: type) -> None:
    """Every shipped `@control` declares `action_class=` explicitly.

    The action-class declaration (see `core/decorators.py`): how much
    authority an instrument's action needs is a judgement about that
    instrument, so it is declared beside the action rather than in a table
    above it. The decorator DOES have a default — the most restrictive class
    — so a missing declaration can never widen what an agent may do; what it
    would do is hide the judgement, which is why every shipped control makes
    it in the open. The undeclared marker is `None`, distinct from an
    explicit `action_class="run_control"`.
    """
    from i2as.core.decorators import VALID_ACTION_CLASSES

    undeclared: list[str] = []
    for method_name, method in _control_methods(vi_cls).items():
        declared = getattr(method, "_control_action_class", None)
        if declared is None:
            undeclared.append(method_name)
        else:
            assert declared in VALID_ACTION_CLASSES, (
                f"{vi_cls.__name__}.{method_name} declares action class "
                f"{declared!r}, not one of {list(VALID_ACTION_CLASSES)}"
            )
    assert not undeclared, (
        f"{vi_cls.__name__} declares controls with no action_class=: "
        f"{sorted(undeclared)} — every @control says how much authority it "
        f"needs (the action-class declaration)"
    )


def test_every_action_class_row_carries_a_rationale() -> None:
    """Each row of the two remaining tables says WHY.

    A `@control`'s class comes from its own declaration, so the only tables
    left are the engine's command surface and the two lifecycle actions.
    """
    from i2as.session.gateway import (
        COMMAND_ACTION_CLASSES,
        LIFECYCLE_ACTION_CLASSES,
    )

    rows: dict[str, object] = {}
    rows.update({f"command {k.value}": v for k, v in COMMAND_ACTION_CLASSES.items()})
    rows.update({f"lifecycle {k}": v for k, v in LIFECYCLE_ACTION_CLASSES.items()})
    for label, classified in rows.items():
        rationale = classified.rationale  # type: ignore[attr-defined]
        assert rationale and rationale.strip(), f"{label} has no rationale"


def test_every_command_name_has_an_action_class() -> None:
    """`CommandName` and the gateway's command table match exactly.

    Diffed both ways: a command with no class is an action no role can be
    granted, and a row with no command behind it is a rule about nothing.
    `SUBMIT_VI_ACTION` is the one deliberate absence — its class depends on
    the capability it targets, so it is resolved per-control instead.
    """
    from i2as.session.gateway import COMMAND_ACTION_CLASSES

    declared = {member for member in CommandName} - {CommandName.SUBMIT_VI_ACTION}
    classified = set(COMMAND_ACTION_CLASSES)

    assert declared - classified == set(), (
        f"CommandName members with no gateway action class: "
        f"{sorted(m.value for m in declared - classified)}"
    )
    assert classified - declared == set(), (
        f"gateway action-class rows with no CommandName behind them: "
        f"{sorted(m.value for m in classified - declared)}"
    )


def test_the_owner_scoped_command_set_is_the_same_on_both_sides() -> None:
    """The engine enforces the run-ownership standard; the gateway mirrors it.

    Two statements of one rule — the engine's, at the single writer, and the
    gateway's, at the front door — so they are diffed rather than trusted.
    The gateway may not import the engine's set (contract C11 keeps the
    session layer off the engine's internals for this purpose), so the two
    are written down twice and matched here.
    """
    from i2as.core.orchestrator import OWNER_SCOPED_COMMANDS as engine_set
    from i2as.session.gateway.roles import OWNER_SCOPED_COMMANDS as gateway_set

    assert {member.value for member in gateway_set} == set(engine_set)
    assert {member.value for member in gateway_set} <= {
        member.value for member in CommandName
    }


def test_every_owner_scoped_command_publishes_the_override_arguments() -> None:
    """The refuse-then-override half must be reachable from the tool schema.

    An agent is told to re-send with ``override_owner`` and a ``reason``; a
    tool whose schema is closed against those two arguments would make the
    instruction impossible to follow.
    """
    from i2as.session.gateway.roles import OWNER_SCOPED_COMMANDS
    from i2as.session.gateway.tools import command_tool_spec

    for member in OWNER_SCOPED_COMMANDS:
        schema = command_tool_spec(member).input_schema
        properties = schema["properties"]
        assert properties["override_owner"]["type"] == "boolean", member.value
        assert properties["reason"]["type"] == "string", member.value
        # Optional, both of them: an ordinary action on one's own run asks
        # for neither.
        assert "override_owner" not in schema["required"], member.value
        assert "reason" not in schema["required"], member.value


def test_permission_matrix_has_a_cell_for_every_class_and_role() -> None:
    """Authority is never absent by omission — every (class, role) pair decided."""
    from i2as.session.gateway import PERMISSION_MATRIX, ActionClass, Role

    assert set(PERMISSION_MATRIX) == set(ActionClass), (
        f"PERMISSION_MATRIX rows do not match ActionClass: "
        f"{sorted(c.value for c in set(PERMISSION_MATRIX) ^ set(ActionClass))}"
    )
    for action_class, row in PERMISSION_MATRIX.items():
        assert set(row) == set(Role), (
            f"PERMISSION_MATRIX[{action_class.value}] does not decide every "
            f"role: {sorted(r.value for r in set(row) ^ set(Role))}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# The agent gateway's tool surface
# ══════════════════════════════════════════════════════════════════════════════
#
# The standard is `session/gateway/tools.py`'s module docstring: a tool is
# rendered from a declaration, never hand-written. These are the third leg of
# the three-way test — the contract (`CommandName`), the engine (the
# Orchestrator's methods) and the tool surface must name the same actions, and
# the station's declared capabilities and the capability tools must name the
# same instruments. Each is diffed in BOTH directions, because both failures
# are real: a missing tool is an action no agent can take, and a tool with no
# declaration behind it dispatches nowhere.


def test_every_command_name_has_a_tool() -> None:
    """`CommandName` and the rendered command tools match exactly.

    The third leg. `test_command_name_covers_every_public_orchestrator_command`
    ties the contract to the engine; this ties the contract to what an agent
    is offered, so an action cannot exist on one surface and not the other.
    `SUBMIT_VI_ACTION` is the one deliberate absence — it is rendered once per
    capability instead, which is what the check below diffs.
    """
    from i2as.session.gateway import render_command_tools

    rendered = {tool.name for tool in render_command_tools()}
    declared = {
        member.value
        for member in CommandName
        if member is not CommandName.SUBMIT_VI_ACTION
    }

    assert declared - rendered == set(), (
        f"CommandName members with no tool to call them: "
        f"{sorted(declared - rendered)}"
    )
    assert rendered - declared == set(), (
        f"command tools with no CommandName behind them: "
        f"{sorted(rendered - declared)}"
    )


def test_every_command_tool_wraps_its_own_command() -> None:
    """A command tool's name IS its command's value, so routing is a lookup."""
    from i2as.session.gateway import render_command_tools

    for tool in render_command_tools():
        assert tool.command is not None, f"{tool.name} is not a command tool"
        assert tool.name == tool.command.value, (
            f"tool {tool.name!r} wraps {tool.command.value!r}; a command "
            f"tool is named for the command it submits"
        )
        assert tool.description.strip(), f"{tool.name} renders no description"


@pytest.mark.parametrize("config_dir", _sim_config_dirs(), ids=lambda p: p.name)
def test_every_manifest_control_has_a_tool(config_dir: Path) -> None:
    """Every capability the manifest declares is callable, and nothing else is.

    Diffed both ways against the capability manifest — the document a client
    builds its instrument surface from — so the two renderings of the same
    declaration can never disagree. A control in the manifest with no tool is
    an instrument an agent can see but not use; a tool with no manifest entry
    is a call into something the station does not declare.
    """
    from i2as.session.gateway import capability_tool_name, render_tools

    station = build_station(str(config_dir))
    declared = {
        capability_tool_name(instrument["name"], control["name"])
        for instrument in build_manifest(station)["instruments"]
        for control in instrument["controls"]
    }
    rendered = {
        tool.name
        for tool in render_tools(station.station_info())
        if tool.instrument and tool.capability
    }

    assert declared - rendered == set(), (
        f"{config_dir.name} declares capabilities with no tool to call them: "
        f"{sorted(declared - rendered)}"
    )
    assert rendered - declared == set(), (
        f"{config_dir.name} renders capability tools the manifest does not "
        f"declare: {sorted(rendered - declared)}"
    )


@pytest.mark.parametrize("config_dir", _sim_config_dirs(), ids=lambda p: p.name)
def test_every_tool_publishes_a_closed_json_safe_schema(config_dir: Path) -> None:
    """Every tool survives JSON, names itself once, and refuses surprise keys.

    A tool list is published to a client that speaks JSON and nothing else, so
    an unserialisable schema is a surface that cannot be offered at all; and a
    schema left open would silently drop an argument an agent believed it had
    supplied.
    """
    from i2as.session.gateway import render_tools

    tools = render_tools(build_station(str(config_dir)).station_info())
    names = [tool.name for tool in tools]
    assert len(set(names)) == len(names), (
        f"{config_dir.name} renders two tools under one name: "
        f"{sorted({n for n in names if names.count(n) > 1})}"
    )

    for tool in tools:
        schema = tool.to_schema()
        assert set(schema) == {"name", "description", "input_schema"}, (
            f"{tool.name} publishes {sorted(schema)}, not the three keys a "
            f"tool-use API reads"
        )
        assert json.loads(json.dumps(schema)) == schema, (
            f"{tool.name}'s schema does not survive a JSON round trip"
        )
        input_schema = schema["input_schema"]
        assert input_schema["type"] == "object", f"{tool.name} is not an object"
        assert input_schema["additionalProperties"] is False, (
            f"{tool.name}'s schema is open; an unexpected argument would be "
            f"dropped rather than refused"
        )
        assert set(input_schema["required"]) <= set(input_schema["properties"]), (
            f"{tool.name} requires an argument it does not declare"
        )


def test_every_session_tool_has_an_implementation_and_a_class() -> None:
    """No session tool is offered that nothing answers, and none without a class.

    Diffed both ways against the implementation table: an unimplemented tool
    is a promise the surface cannot keep, and an orphan implementation is dead
    code nobody can reach.
    """
    from i2as.session.gateway import SESSION_TOOLS, ActionClass
    from i2as.session.gateway.tools import SESSION_TOOL_FUNCTIONS

    declared = {tool.session_function for tool in SESSION_TOOLS if not tool.is_command}
    implemented = set(SESSION_TOOL_FUNCTIONS)

    assert declared - implemented == set(), (
        f"session tools with no implementation: {sorted(declared - implemented)}"
    )
    assert implemented - declared == set(), (
        f"session-tool implementations no tool offers: "
        f"{sorted(implemented - declared)}"
    )
    for tool in SESSION_TOOLS:
        assert isinstance(tool.action_class, ActionClass), (
            f"{tool.name} carries no action class, so no role can be granted it"
        )


#: What a client above the boundary may still call on the ``Station``, and why
#: each is safe once the Station lives on the **Instrument thread**. Every one
#: is a pure DECLARATION read: it answers from the config and the registry, it
#: touches no instrument and starts no bus traffic, so a client reading it
#: cannot race the engine's tick into a driver. A RATCHET, not a budget — the
#: list only ever gets shorter, as each remaining reader moves onto the
#: **Station info** declaration snapshot the mirror already carries.
_CLIENT_STATION_READS: dict[str, str] = {
    "get_vi_names": "the registry's names, from config",
    "get_vi_type": "a registered VI's kind, from config",
    "has_vi": "whether a name is registered, from config",
    "offline_vi_names": "the offline registry's names, from the build",
    "station_info": "the declaration snapshot itself",
    "envelope_variables": "the declared envelope variables, from config",
    "nominal_ramp_rates": "the declared ramp rates, from config",
    "last_state_flat": (
        "the monitor-tick cache's numeric keys, read once at panel build to "
        "learn which keys EXIST; the values themselves come off the event "
        "stream"
    ),
}


def test_clients_call_only_pure_declaration_reads_on_the_station() -> None:
    """``gui/`` and ``session/`` touch no instrument through the Station.

    Import contract C19 and
    ``test_gui_imports_the_station_only_for_typing_or_config_helpers`` already
    keep the GUI from IMPORTING the Station, and
    ``test_gui_never_reaches_into_the_station_for_a_vi`` keeps it out of
    ``get_vi()``. Neither says anything about the Station object both layers
    are still HANDED at construction, and once that object lives on the
    **Instrument thread** the question is no longer layering but data races:
    a client calling a polling method would be reading a driver from the
    wrong thread.

    So this enumerates what those layers actually call on a Station and holds
    it to the declaration reads named in ``_CLIENT_STATION_READS``. If it
    fails on a call you just added: read it off the ``StationInfo``
    declaration or the **Status mirror** instead. Adding a name here is a
    decision about the thread boundary, not a formality.
    """
    offenders: list[str] = []
    for folder in ("gui", "session"):
        for path in sorted((PACKAGE_DIR / folder).rglob("*.py")):
            relative = path.relative_to(PACKAGE_DIR.parent).as_posix()
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Attribute | ast.Name)
                ):
                    continue
                receiver = node.func.value
                name = (
                    receiver.attr
                    if isinstance(receiver, ast.Attribute)
                    else receiver.id
                )
                if name not in ("station", "_station"):
                    continue
                if node.func.attr in _CLIENT_STATION_READS:
                    continue
                if not hasattr(Station, node.func.attr):
                    # A receiver merely NAMED station that is not one — the
                    # StationInfo declaration or its JSON dict (``to_json``,
                    # ``get``) — is exactly what a client should be reading.
                    continue
                offenders.append(f"{relative}:{node.lineno}: {node.func.attr}()")

    assert not offenders, (
        "Station call(s) from i2as/gui/ or i2as/session/ that are not "
        "pure declaration reads — the Station lives on the instrument thread, "
        "so read this off the StationInfo declaration or the StatusMirror "
        "instead:\n" + "\n".join(offenders)
    )


# ── Analysis recipe standard (i2as/analysis/) ────────────────────────────
# The recipe contract written at the top of i2as/analysis/base.py: one
# class per recipe declaring a snake_case name, a non-empty one-line
# description and a procedures tuple, implementing analyse(run, context) with
# the contract's signature, instantiated with no arguments, and returning one
# ok AnalysisReport for a real run file. A recipe module added to
# i2as/analysis/recipes/ is covered the moment the file exists — that is
# what discovery buys, exactly as it does for drivers, VIs and procedures.

_ANALYSIS_DATA_CONFIG = {
    # The writer's own order: the clock, then the system read-backs, then the
    # procedure's own axis column last.
    "sweep_columns": {"unix_time": "float", "temperature_K": "float", "field_T": "float"},
    "measurement_scalars": {"voltage_V": "float"},
    "measurement_arrays": {},
    "measurement_blocks": {},
    "loop_shape": [1, 1],
}

_ANALYSIS_MANIFEST = {
    "run_id": "conformance-run",
    "procedure": "FieldSweep",
    "kind": "run",
    "params": {"field_start": -1.0, "field_end": 1.0},
    "status": "done",
    "started_utc": "2026-01-01T00:00:00+00:00",
    "finished_utc": "2026-01-01T00:05:00+00:00",
}


def _package_recipe_infos() -> list:
    """Every recipe shipped in i2as/analysis/recipes/."""
    from i2as.analysis.discovery import ORIGIN_PACKAGE, discover_recipes

    return [info for info in discover_recipes() if info.origin == ORIGIN_PACKAGE]


def _synthetic_run_file(directory: Path) -> Path:
    """Write a small real run file with the production writer.

    Args:
        directory: Where to create the file.

    Returns:
        Path of the closed HDF5 file: three sweep points of one axis, one
        system read-back and one measured scalar.
    """
    from i2as.core.data_manager import DataManager

    writer = DataManager(
        data_directory=str(directory),
        procedure_name="FieldSweep",
        procedure_params={"field_start": -1.0, "field_end": 1.0},
        sample_info={"sample_name": "Conformance"},
        instrument_state={},
        system_targets={},
        measurement_commands=[],
        data_config=_ANALYSIS_DATA_CONFIG,
        n_sweep_points=3,
        experiment_info={"setup": {"config_name": "sim"}, "experiment": {"experiment_id": "E"}},
    )
    for index in range(3):
        writer.save_datapoint(
            index,
            {
                "unix_time": 1_000.0 + index,
                "temperature_K": 4.2,
                "field_T": -1.0 + index,
                "voltage_V": [[1.0 + index]],
            },
            {},
        )
    writer.close()
    return Path(writer.filepath)


@pytest.mark.parametrize("info", _package_recipe_infos(), ids=lambda i: i.name)
def test_analysis_recipe_declares_the_contract(info) -> None:
    """name, description and procedures are declared as the contract requires."""
    assert info.name and info.name == info.name.lower() and info.name.isidentifier(), (
        f"recipe name {info.name!r} must be a non-empty snake_case identifier — "
        f"it is the id the panel, the settings file and the agent's tools use"
    )
    assert info.description.strip(), (
        f"recipe {info.name!r} must declare a one-line description; it is what "
        f"a physicist picks from"
    )
    assert isinstance(info.procedures, tuple) and info.procedures, (
        f"recipe {info.name!r} must declare the procedure class names it serves "
        f'(or ("*",) for every run), got {info.procedures!r}'
    )
    assert all(isinstance(name, str) and name for name in info.procedures), (
        f"recipe {info.name!r}.procedures must hold procedure CLASS NAMES as "
        f"strings, got {info.procedures!r}"
    )


@pytest.mark.parametrize("info", _package_recipe_infos(), ids=lambda i: i.name)
def test_analysis_recipe_analyse_matches_the_contract_signature(info) -> None:
    """analyse(self, run, context) — no extra argument, none missing."""
    from i2as.analysis.base import AnalysisRecipe
    from i2as.analysis.discovery import load_recipe

    recipe = load_recipe(info)  # constructed with no arguments, per the contract
    assert isinstance(recipe, AnalysisRecipe)
    expected = list(inspect.signature(AnalysisRecipe.analyse).parameters)
    actual = list(inspect.signature(type(recipe).analyse).parameters)
    assert actual == expected, (
        f"{type(recipe).__name__}.analyse{inspect.signature(type(recipe).analyse)} "
        f"does not match the contract AnalysisRecipe.analyse"
        f"{inspect.signature(AnalysisRecipe.analyse)}"
    )


@pytest.mark.parametrize("info", _package_recipe_infos(), ids=lambda i: i.name)
def test_analysis_recipe_runs_to_an_ok_report(info, tmp_path) -> None:
    """Every shipped recipe answers a real run file with an ok report."""
    from i2as.analysis.report import REPORT_OK, AnalysisSpec
    from i2as.analysis.runner import run_spec

    run_file = _synthetic_run_file(tmp_path / "data")
    report = run_spec(
        AnalysisSpec(
            run_id="conformance-run",
            data_path=str(run_file),
            manifest=dict(_ANALYSIS_MANIFEST),
            experiment={"experiment_id": "E"},
            setup={"config_name": "sim"},
            recipe=info.name,
            output_dir=str(tmp_path / "report"),
        )
    )
    assert report.status == REPORT_OK, (
        f"recipe {info.name!r} failed on a synthetic run:\n{report.error}"
    )
    assert report.recipe == info.name
    assert report.recipe_digest == info.digest
    assert report.summary, f"recipe {info.name!r} produced no summary paragraph"
    for figure in report.figures:
        assert (tmp_path / "report" / figure.file).is_file(), (
            f"recipe {info.name!r} references figure {figure.file!r}, which it "
            f"did not save into the output directory"
        )


# ── Folder README standard (CLAUDE.md) ───────────────────────────────────────
# Every functional folder under i2as/ carries a README.md with the seven
# canonical sections. Auto-discovered, so a NEW package folder is covered the
# moment its README exists — and a folder that adds a README without the
# standard's sections fails here rather than drifting.

#: The seven sections, as the heading prefix each must start with. Entry/Exit
#: and "How to add" carry a parenthetical or an object in the shipped READMEs
#: ("Entry (what comes in)", "How to add a new magnet VI"), so the standard is
#: the prefix, not the exact string.
README_SECTIONS: tuple[str, ...] = (
    "Purpose",
    "Architecture layer",
    "Entry",
    "Exit",
    "Interface contract",
    "How to add",
    "Files",
)


def _folder_readmes() -> list[Path]:
    """Every folder README under i2as/."""
    return sorted(PACKAGE_DIR.rglob("README.md"))


@pytest.mark.parametrize(
    "readme", _folder_readmes(), ids=lambda p: p.parent.name
)
def test_folder_readme_has_the_seven_sections(readme: Path) -> None:
    """The folder README standard: Purpose … Files, in that order."""
    headings = [
        line[3:].strip()
        for line in readme.read_text(encoding="utf-8").splitlines()
        if line.startswith("## ")
    ]
    position = -1
    for section in README_SECTIONS:
        matches = [
            index for index, heading in enumerate(headings) if heading.startswith(section)
        ]
        assert matches, (
            f"{readme.relative_to(PACKAGE_DIR.parent)} has no "
            f"'## {section}…' section; the folder README standard (CLAUDE.md) "
            f"asks for {list(README_SECTIONS)}, found {headings}"
        )
        assert matches[0] > position, (
            f"{readme.relative_to(PACKAGE_DIR.parent)}: "
            f"'{section}' must come after the section before it in "
            f"{list(README_SECTIONS)}"
        )
        position = matches[0]


def test_analysis_package_has_a_folder_readme() -> None:
    """The analysis stage is a functional folder, so it carries a README."""
    readme = PACKAGE_DIR / "analysis" / "README.md"
    assert readme.is_file(), f"{PACKAGE}/analysis/ must have a folder README"
    assert readme in _folder_readmes()


def test_console_scripts_name_callable_entry_points() -> None:
    """Each ``[project.scripts]`` entry resolves to a function callable with no arguments.

    The installed commands (``i2as``, ``i2as-ctl``, ``i2as-doctor``,
    ``i2as-mcp``) are the module entry points under other names, so the
    target must import, be callable, and take no required positional
    argument — a console script is called bare and reads ``sys.argv``.
    """
    import tomllib

    pyproject = tomllib.loads(
        (PACKAGE_DIR.parent / "pyproject.toml").read_text(encoding="utf-8")
    )
    scripts = pyproject["project"]["scripts"]
    assert set(scripts) == {"i2as", "i2as-ctl", "i2as-doctor", "i2as-mcp"}
    assert pyproject["project"]["name"] == PACKAGE
    assert pyproject["tool"]["importlinter"]["root_package"] == PACKAGE
    for command, target in scripts.items():
        module_name, function_name = target.split(":")
        assert module_name.split(".")[0] == PACKAGE, command
        function = getattr(importlib.import_module(module_name), function_name)
        assert callable(function), command
        required = [
            name
            for name, parameter in inspect.signature(function).parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        assert required == [], f"{command} -> {target} needs {required}"
