"""Station class — runtime registry of all Virtual Instruments.

The Station is Layer 2. It sits between the VI layer (L1) and the Orchestrator (L3).
It knows about all VIs, polls their state, and dispatches ramp commands.

Do NOT import from Orchestrator, Procedures, or GUI here.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import time
import typing
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from i2as.core.availability import Availability, state_for
from i2as.core.conditions import Condition
from i2as.core.config import DEFAULT_TICK_INTERVAL_MS
from i2as.core.decorators import (
    get_control_action_class,
    get_control_methods,
    get_control_panel,
    get_control_scope,
    get_control_specs,
    get_monitored_description,
    get_monitored_unit,
    get_ui_group,
)
from i2as.core.events import (
    ControlInfo,
    GroupInfo,
    InstrumentInfo,
    LifecycleState,
    MonitoredInfo,
    StationInfo,
)
from i2as.core.exceptions import (
    I2ASActionScopeError,
    I2ASCommunicationError,
    I2ASConfigError,
    I2ASPrivateActionError,
    I2ASSafetyError,
    I2ASUndeclaredActionError,
)
from i2as.core.plan import (
    SETPOINT_PARAM_PREFIX,
    Command,
    EnvelopeVariable,
    ParamSpec,
    Target,
)
from i2as.virtual_instruments.base import (
    BaseVirtualInstrument,
    MagnetBase,
    StageBase,
    TemperatureControllerBase,
)
from i2as.virtual_instruments.rampable import RampableVI

logger = logging.getLogger(__name__)

#: The two connection-lifecycle operating-state methods (see
#: ``BaseVirtualInstrument``'s "Connection-lifecycle standard"). They carry no
#: ``@control`` — every VI inherits or defines them as plain methods — but they
#: ARE capabilities of the direct action path: the per-panel lifecycle toggle
#: and "Initiate All"/"Standby All" dispatch nothing else. Listing them here is
#: what lets ``execute_vi_action()`` refuse every OTHER undecorated method
#: without breaking the lifecycle path. They are measurement-scope, the same
#: default an undecorated method gets in
#: ``Station.send_measurement_commands()``.
LIFECYCLE_ACTIONS: frozenset[str] = frozenset({"initiate", "standby"})

# The one reason string used wherever an instrument opened a session but did
# not answer the identity query — the single connection check of the
# connection-lifecycle standard (see BaseVirtualInstrument).
_IDENTITY_FAILED_REASON = (
    "Opened the connection but the instrument did not answer an identity "
    "query — check that it is powered on and that the address in the config "
    "points at it."
)


def _identity_check(vi_name: str, vi: BaseVirtualInstrument) -> bool:
    """Return whether *vi* answers an identity query, never raising.

    The ONLY command I2AS sends when it brings an instrument up (the
    connection-lifecycle standard's rule 1: construction is silent). A VI
    whose ``ping()`` itself raises is treated as unreachable rather than
    aborting the build — a degraded station is always better than no station.

    Args:
        vi_name: The VI's configured name, for the log line.
        vi: The freshly constructed VI.

    Returns:
        True if the instrument answered.
    """
    try:
        return bool(vi.ping())
    except Exception:  # noqa: BLE001 — any failure means "not reachable"
        logger.warning("Identity check raised for VI '%s'", vi_name, exc_info=True)
        return False


def _exclusive_aliases(
    role_aliases: Mapping[str, str],
    vi_specs: Mapping[str, dict],
    vi_name: str,
    live_registry: Mapping[str, str] | None = None,
) -> list[str]:
    """Return *vi_name*'s driver aliases that no OTHER live VI still needs.

    Drivers are shared by config (one source serving two measurement VIs,
    say), so disconnecting one instrument must never close a
    session another live instrument is using. This is the single place that
    decides which sessions a disconnect may actually release.

    Args:
        role_aliases: The disconnecting VI's ``{role: alias}`` mapping.
        vi_specs: Every VI's retained build recipe, ``{vi_name: spec}``.
        vi_name: The VI being disconnected (excluded from the "other users"
            search).
        live_registry: The names of the VIs that are currently live. ``None``
            means "consider every configured VI a user", the conservative
            choice used when unwinding a failed connect: nothing is closed
            that any other VI's recipe references.

    Returns:
        Aliases safe to close, order-preserving and de-duplicated.
    """
    mine = list(dict.fromkeys(role_aliases.values()))
    others: set[str] = set()
    for other_name, spec in vi_specs.items():
        if other_name == vi_name:
            continue
        if live_registry is not None and other_name not in live_registry:
            continue
        others.update((spec.get("drivers") or {}).values())
    return [alias for alias in mine if alias not in others]


# `@dataclass(frozen=True)` generates an immutable value class: __init__,
# __eq__ and __repr__ come for free, and instances cannot be mutated after
# creation — updates go through `dataclasses.replace()`, which returns a new
# instance. Immutability keeps the offline registry safe to hand out to the
# GUI without defensive copies.
@dataclass(frozen=True)
class OfflineInstrument:
    """Record of a configured VI that could not be brought up at build time.

    Produced by ``build_station()`` when a driver (or the VI itself) fails to
    connect. The Station keeps these in a registry parallel to the live VIs so
    upper layers can show *what* is missing and *why*, and offer a reconnect.

    Attributes:
        vi_name: The VI's configured name (e.g. ``"magnet_z"``).
        vi_type: The registry vi_type from config (``"system"``,
            ``"measurement"``).
        reason: Human-readable connection-failure description, suitable for
            direct display in the GUI.
        failed_drivers: Config aliases of the drivers that failed to
            construct. Empty when the drivers were fine but the VI's own
            construction raised a communication error.
        tags: Why this VI is offline, as a set drawn from the Availability
            standard's closed vocabulary (``i2as.core.availability``):
            ``"connect_failed"`` (the default: the hardware could not be
            reached at build time or on a reconnect attempt) and/or
            ``"operator"`` (the connection-lifecycle standard: the operator
            deliberately disconnected it, e.g. to drive the instrument from
            its front panel or the vendor's software). Both tags can hold at
            once — a reconnect attempt on an operator-disconnected VI that
            then fails on hardware carries both. The degraded behaviour is
            deliberately IDENTICAL regardless of which tags apply — one
            offline path, not several — so this only ever changes what the
            GUI *says*, never what the station *does*.
        since: Unix timestamp this VI became offline.
    """

    vi_name: str
    vi_type: str
    reason: str
    failed_drivers: tuple[str, ...] = field(default=())
    tags: frozenset[str] = field(default=frozenset({"connect_failed"}))
    since: float = 0.0


@dataclass(frozen=True)
class FaultRecord:
    """Record of a RUNTIME fault on a VI that DID connect.

    The runtime sibling of :class:`OfflineInstrument`: an offline instrument
    never connected at build time; a ``FaultRecord`` describes a VI that was
    live and has since gone stale or disconnected (comm-error streak) during
    normal polling. Populated by ``get_state()`` at the same point it already
    computes ``_stale``/``_disconnected``, so no extra poll is introduced.

    Attributes:
        vi_name: The VI's registered name.
        kind: ``"stale"`` (communication errors, below the disconnect
            threshold) or ``"disconnected"`` (``max_errors`` consecutive
            failures).
        message: Human-readable description of the latest failure.
        since: Unix time this fault was first recorded. Preserved across a
            ``"stale"`` -> ``"disconnected"`` escalation of the SAME ongoing
            incident (the record is updated in place, not replaced).
        acknowledged: Whether the operator has acknowledged this fault via
            ``acknowledge_fault()``. Deliberately does NOT survive recovery:
            once the VI polls successfully again the record is removed
            entirely (see ``clear_fault()``), acknowledged or not — a
            recovered VI has nothing left to acknowledge.
    """

    vi_name: str
    kind: str
    message: str
    since: float
    acknowledged: bool = False


class Station:
    """Runtime registry and coordinator of all Virtual Instruments.

    Provides:
    - VI registration and attribute-style access (``station.magnet_z``).
    - Polled state snapshot with stale-value caching.
    - Ramp dispatch and progress tracking.
    - Safety status aggregation.
    - Measurement command dispatch.
    - Bulk initiate / standby.
    """

    def __init__(self) -> None:
        # Setup identity: the name of the config directory this Station was
        # built from (`build_station()` sets it). None for a Station assembled
        # in-process without a config folder, e.g. in a unit test.
        self._setup_name: str | None = None
        self._vi_registry: dict[str, str] = {}            # {vi_name: vi_type}
        self._virtual_instruments: dict[str, BaseVirtualInstrument] = {}
        self._last_known_state: dict[str, dict] = {}       # Stale value cache
        self._error_counts: dict[str, int] = {}
        # The read currently in flight, for the shutdown diagnostic: the name
        # of the VI `get_state()` is inside, or None between polls. Written
        # only by the thread that owns this Station and read by whoever has to
        # report a wedged instrument (see `polling_vi()`).
        self._polling_vi: str | None = None
        self._max_errors: int = 3
        # Degraded-build support: VIs whose hardware failed to connect at
        # build time, plus the build recipes and live driver instances that
        # connect_instrument() needs to bring one back without a restart.
        self._offline_vis: dict[str, OfflineInstrument] = {}
        self._driver_specs: dict[str, dict] = {}   # {alias: driver config}
        self._vi_specs: dict[str, dict] = {}       # {vi_name: vi config}
        self._drivers: dict[str, Any] = {}         # {alias: live driver}
        # Unified condition registry — the System-Condition standard (see
        # i2as/core/conditions.py and GLOSSARY.md): every comm- and
        # safety-origin Condition currently active on this Station, keyed
        # by Condition.key ("comm:<vi_name>" for a VI that DID connect but
        # has since gone stale/disconnected during polling; "safety:<flag>"
        # for a tripped safety flag). Distinct from _offline_vis (never
        # connected at build time). vi_faults() and its siblings below are
        # adapters that read this one registry, so the GUI observes the
        # exact same field semantics as before this registry was unified.
        self._conditions: dict[str, Condition] = {}
        # Safety flags whose "nobody consumes this hold" WARNING has
        # already been logged once by update_conditions() — logged only on
        # first occurrence per flag, not on every tick it stays unconsumed.
        self._warned_unconsumed_flags: set[str] = set()
        # The setup's monitor tick period, in seconds — a config property
        # (`monitor.yaml`'s tick_interval_ms), carried in the station
        # declaration snapshot so a client knows the cadence readings arrive
        # at. `build_station()` sets it; the Orchestrator's own mirrored
        # default stands in for a Station assembled without a config.
        self._tick_interval_s: float = DEFAULT_TICK_INTERVAL_MS / 1000.0
        # The cached station declaration snapshot (core/events.py's
        # StationInfo) and the number of times it has been built.
        # _invalidate_station_info() drops it whenever the station's
        # membership or an offline record changes — every connect and
        # disconnect — so the next read rebuilds.
        self._station_info: StationInfo | None = None
        self._station_info_seq: int = 0

    # ------------------------------------------------------------------
    # VI registration and access
    # ------------------------------------------------------------------

    def register_vi(self, vi_name: str, vi: BaseVirtualInstrument, vi_type: str) -> None:
        """Register a Virtual Instrument with this Station.

        Args:
            vi_name: Unique name for this VI (e.g. ``"magnet_z"``).
            vi: The VI instance.
            vi_type: Category string (``"system"`` or ``"measurement"``).
        """
        vi.vi_name = vi_name
        self._vi_registry[vi_name] = vi_type
        self._virtual_instruments[vi_name] = vi
        self._error_counts[vi_name] = 0
        self._invalidate_station_info()
        logger.info("Registered VI '%s' (type=%s)", vi_name, vi_type)

    def setup_name(self) -> str | None:
        """Return the setup name this Station was built from.

        The setup's identity is its config directory's name (the same string
        the app stores as the active config and an `ExperimentRecord` stores
        as ``config_name``) — a setup property, so it comes from the config
        rather than from anything in code.

        Returns:
            The config directory's name, or None for a Station built without
            one (constructed directly rather than through `build_station()`).
        """
        return self._setup_name

    def get_vi_names(self) -> list[str]:
        """Return a list of all registered VI names."""
        return list(self._virtual_instruments.keys())

    def get_vi(self, vi_name: str) -> BaseVirtualInstrument:
        """Return the registered VI instance by name.

        The named lookup counterpart to attribute access (``station.magnet_z``):
        used when the name is only known at runtime, e.g. a procedure resolving
        the measurement VI the user selected in the GUI.

        Args:
            vi_name: Name of the registered VI.

        Returns:
            The VI instance.

        Raises:
            KeyError: If no VI with that name is registered.
        """
        return self._virtual_instruments[vi_name]

    def measurement_vi_names(self) -> list[str]:
        """Return the names of all registered measurement VIs, in registration order.

        A measurement VI is one registered with ``vi_type == "measurement"``.
        The order is the order the VIs were registered (config order), so a GUI
        or procedure that defaults to "the first measurement VI" gets a stable,
        config-controlled choice.

        Returns:
            List of measurement VI names, registration order preserved.
        """
        return [
            name
            for name, vi_type in self._vi_registry.items()
            if vi_type == "measurement"
        ]

    def vi_names_by_base(self, base_cls: type) -> list[str]:
        """Return every registered VI that is an instance of *base_cls*.

        The general read behind **role discovery** (GLOSSARY.md): a caller
        that needs "the magnet" or "the temperature controller" asks for the
        instrument CATEGORY and gets whatever this setup configured, instead
        of naming an instrument the config may not have. The typed category
        bases in ``virtual_instruments/base.py`` are the roles; the two thin
        accessors below are the ones procedures use, since a procedure is
        hardware-blind and never imports a VI class itself (contract C6).

        Args:
            base_cls: A VI base class, e.g. ``MagnetBase``.

        Returns:
            The matching VI names, registration (config) order preserved, so
            "the only candidate" and "the first candidate" are both stable
            and config-controlled.
        """
        return [
            name
            for name, vi in self._virtual_instruments.items()
            if isinstance(vi, base_cls)
        ]

    def magnet_vi_names(self) -> list[str]:
        """Return the names of all registered magnet VIs, in config order.

        Args:
            None.

        Returns:
            List of magnet VI names — every VI built on ``MagnetBase``.
        """
        return self.vi_names_by_base(MagnetBase)

    def temperature_vi_names(self) -> list[str]:
        """Return the names of all registered temperature-controller VIs.

        Args:
            None.

        Returns:
            List of temperature VI names — every VI built on
            ``TemperatureControllerBase`` — in config order.
        """
        return self.vi_names_by_base(TemperatureControllerBase)

    def stage_vi_names(self) -> list[str]:
        """Return the names of all registered stage (sample-positioning) VIs.

        Args:
            None.

        Returns:
            List of stage VI names — every VI built on ``StageBase``, one per
            positioned axis — in config order.
        """
        return self.vi_names_by_base(StageBase)

    def has_vi(self, vi_name: str) -> bool:
        """Return True if a VI with this name is registered."""
        return vi_name in self._virtual_instruments

    # ------------------------------------------------------------------
    # Offline instruments (degraded build) and reconnection
    # ------------------------------------------------------------------

    def register_offline_vi(self, info: OfflineInstrument) -> None:
        """Record a configured VI that failed to connect at build time.

        An offline VI is *not* in the live registry: ``get_vi_names()`` and
        the typed enumerators never return it, so the Orchestrator, procedures
        and safety evaluation transparently see a smaller station.

        Args:
            info: The offline record (name, type, human-readable reason).
        """
        self._offline_vis[info.vi_name] = info
        self._invalidate_station_info()
        logger.warning(
            "VI '%s' registered OFFLINE (type=%s): %s",
            info.vi_name,
            info.vi_type,
            info.reason,
        )

    def offline_vi_names(self) -> list[str]:
        """Return the names of all offline VIs, in config order."""
        return list(self._offline_vis.keys())

    def get_offline_info(self, vi_name: str) -> OfflineInstrument:
        """Return the offline record for a VI.

        Args:
            vi_name: Name of the offline VI.

        Returns:
            The :class:`OfflineInstrument` record.

        Raises:
            KeyError: If no offline VI with that name exists.
        """
        return self._offline_vis[vi_name]

    def _build_availability(self, vi_name: str) -> Availability:
        """Assemble *vi_name*'s :class:`Availability` from its sources of truth.

        The Availability standard (``i2as.core.availability``) is a
        derived VIEW, not a fourth registry: this method reads the existing
        ``_offline_vis`` / ``_conditions`` registries and the VI itself,
        never storing anything new. An offline VI's own ``tags`` carry the
        absence tags; a live VI's tags are the union of a standing comm
        condition's ``"not_responding"`` and the VI's own attachment state.

        Args:
            vi_name: Name of a CONFIGURED VI — live or offline.

        Returns:
            The assembled :class:`Availability` record.

        Raises:
            KeyError: If `vi_name` is not a configured VI at all.
        """
        offline = self._offline_vis.get(vi_name)
        if offline is not None:
            return Availability(
                vi_name=vi_name,
                vi_type=offline.vi_type,
                state=state_for(offline.tags),
                tags=offline.tags,
                reason=offline.reason,
                since=offline.since,
            )

        vi = self._virtual_instruments[vi_name]  # KeyError if not configured at all
        vi_type = self._vi_registry.get(vi_name, "system")
        tags: set[str] = set()
        reason = ""
        since = 0.0

        condition = self._conditions.get(f"comm:{vi_name}")
        if condition is not None:
            tags.add("not_responding")
            reason = condition.message
            since = condition.since

        # is_attached() is a real BaseVirtualInstrument method (see the
        # "Detach-when-idle declaration" in virtual_instruments/base.py):
        # True for every VI by default, False while a detach_when_idle VI
        # has released its session.
        if not vi.is_attached():
            tags.add("detached")

        frozen_tags = frozenset(tags)
        return Availability(
            vi_name=vi_name,
            vi_type=vi_type,
            state=state_for(frozen_tags),
            tags=frozen_tags,
            reason=reason,
            since=since,
        )

    def availability(self, vi_name: str) -> Availability:
        """Return the unified Availability record for one configured VI.

        The Availability standard's (``i2as.core.availability``) single
        accessor: whether `vi_name` is live, offline, or faulted, this is
        the one place to ask "why can't I use this instrument?".

        Args:
            vi_name: Name of a configured VI — live or offline.

        Returns:
            The :class:`Availability` record.

        Raises:
            KeyError: If `vi_name` is not a configured VI at all, mirroring
                ``get_vi()``/``get_offline_info()``.
        """
        return self._build_availability(vi_name)

    def availabilities(self) -> dict[str, Availability]:
        """Return the unified Availability record for every configured VI.

        Covers both the live registry and the offline registry, so a caller
        sees every configured VI — not just the ones currently held.

        Returns:
            ``{vi_name: Availability}`` for every configured VI.
        """
        names = list(self._virtual_instruments.keys()) + list(self._offline_vis.keys())
        return {name: self._build_availability(name) for name in names}

    def lifecycle_state(self, vi_name: str) -> str:
        """Return what one configured instrument is DOING (the lifecycle-state standard).

        The Station's read side of the standard (GLOSSARY.md's **Lifecycle
        state**; full text in ``BaseVirtualInstrument``'s docstring): a pure
        passthrough to the VI's own cached answer, so this costs nothing and
        can be asked for every VI on every tick.

        Args:
            vi_name: Name of a configured VI — live or offline.

        Returns:
            One of ``LifecycleState``'s values. An OFFLINE instrument is
            reported ``"idle"``: I2AS holds nothing of it, so it cannot
            be initiated as far as any client is concerned, and the VI a
            later ``connect_instrument()`` builds starts there too.

        Raises:
            KeyError: If `vi_name` is not a configured VI at all, mirroring
                ``availability()`` / ``get_vi()``.
        """
        vi = self._virtual_instruments.get(vi_name)
        if vi is None:
            if vi_name in self._offline_vis:
                return LifecycleState.IDLE.value
            raise KeyError(vi_name)
        return vi.lifecycle_state()

    def lifecycle_states(self) -> dict[str, str]:
        """Return every configured instrument's lifecycle state.

        Covers the live and the offline registry, exactly as
        ``availabilities()`` does, so the status-snapshot assembly can carry
        one entry per VI a client renders a card for. Polls nothing.

        Returns:
            ``{vi_name: lifecycle state}`` for every configured VI.
        """
        names = list(self._virtual_instruments.keys()) + list(self._offline_vis.keys())
        return {name: self.lifecycle_state(name) for name in names}

    def connect_instrument(self, vi_name: str) -> tuple[bool, str]:
        """Bring an offline VI online: rebuild its drivers, the VI, then verify.

        The ``connect`` half of the connection-lifecycle standard (see
        ``BaseVirtualInstrument``), and the only way a VI rejoins the live
        registry — whether it never connected at startup or the operator
        disconnected it deliberately. Re-runs the same construction
        ``build_station()`` performed, from the retained build recipe: each
        of the VI's drivers that is not already live is constructed (its
        ``__init__`` opens the bus session and sends nothing else), then the
        VI itself, then one identity query to prove the instrument is really
        there. On success the VI joins the live registry exactly as if it had
        connected at startup; on failure the offline record's ``reason`` is
        refreshed with the latest error.

        A driver brought up here is shared: another offline VI referencing the
        same alias will find it already live on its own connect.

        Args:
            vi_name: Name of the offline VI to connect.

        Returns:
            An explicit ``(ok, message)`` verdict for the GUI, mirroring the
            control-validation standard: ``message`` is the human-readable
            success confirmation or failure reason.
        """
        info = self._offline_vis.get(vi_name)
        if info is None:
            return False, f"'{vi_name}' is not offline"
        spec = self._vi_specs.get(vi_name)
        if spec is None:
            return False, f"No build recipe retained for '{vi_name}'"

        role_aliases = dict(spec.get("drivers") or {})
        # dict.fromkeys: order-preserving de-dup of the alias list.
        for alias in dict.fromkeys(role_aliases.values()):
            if alias in self._drivers:
                continue
            driver_cfg = self._driver_specs.get(alias, {})
            try:
                cls = _import_class(driver_cfg["class"])
                self._drivers[alias] = cls(driver_cfg.get("address", "SIM"))
            except Exception as exc:  # noqa: BLE001 — verdict, never a crash, in GUI context
                reason = f"driver '{alias}': {exc}"
                still_failed = tuple(
                    a for a in dict.fromkeys(role_aliases.values())
                    if a not in self._drivers
                )
                self._offline_vis[vi_name] = replace(
                    info,
                    tags=info.tags | {"connect_failed"},
                    reason=reason,
                    failed_drivers=still_failed,
                )
                self._invalidate_station_info()
                logger.warning("Connect of '%s' failed: %s", vi_name, reason)
                return False, reason

        driver_refs = {role: self._drivers[alias] for role, alias in role_aliases.items()}
        init_params = dict(spec.get("init_params", {}) or {})
        try:
            cls = _import_class(spec["class"])
            vi = cls(driver_refs, **init_params)
        except Exception as exc:  # noqa: BLE001 — verdict, never a crash, in GUI context
            self._offline_vis[vi_name] = replace(
                info,
                tags=info.tags | {"connect_failed"},
                reason=str(exc),
                failed_drivers=(),
            )
            self._invalidate_station_info()
            logger.warning(
                "Connect of '%s' failed in VI construction: %s", vi_name, exc
            )
            return False, str(exc)

        # The identity check, exactly as build_station() applies it: an open
        # session is not proof the instrument is answering.
        vi.vi_name = vi_name
        if not _identity_check(vi_name, vi):
            reason = _IDENTITY_FAILED_REASON
            self._offline_vis[vi_name] = replace(
                info,
                tags=info.tags | {"connect_failed"},
                reason=reason,
                failed_drivers=(),
            )
            self._invalidate_station_info()
            self._release_drivers(_exclusive_aliases(role_aliases, self._vi_specs, vi_name))
            logger.warning("Connect of '%s' failed the identity check", vi_name)
            return False, reason

        del self._offline_vis[vi_name]
        self.register_vi(vi_name, vi, spec.get("vi_type", "system"))
        logger.info("Instrument '%s' connected", vi_name)
        return True, f"'{vi_name}' connected"

    def disconnect_instrument(self, vi_name: str) -> tuple[bool, str]:
        """Release a live VI's instrument and degrade it to the offline registry.

        The ``disconnect`` half of the connection-lifecycle standard (see
        ``BaseVirtualInstrument``): hands the instrument back so the operator
        can drive it from its physical front panel or the vendor's own
        software, without stopping anything else on the station.

        What it does, in order:

        1. ``vi.disconnect()`` — the VI's own release hook. NOT
           ``standby()``: disconnecting never changes what the instrument is
           doing (rule 2 of the standard). An operator who wants it safe
           first presses Standby first.
        2. Removes the VI from the live registry, so polling, safety
           evaluation, ramps and procedures stop seeing it — the same
           degraded state a startup connection failure produces.
        3. Closes the bus session of every driver the VI was the LAST live
           user of. A driver shared with a VI that is staying online is left
           open, so disconnecting one instrument can never break another.
        4. Records an ``OfflineInstrument`` tagged ``{"operator"}``, which
           is what ``connect_instrument()`` later reads to bring it back.

        Args:
            vi_name: Name of the live VI to disconnect.

        Returns:
            An explicit ``(ok, message)`` verdict, mirroring
            ``connect_instrument()``.
        """
        vi = self._virtual_instruments.get(vi_name)
        if vi is None:
            if vi_name in self._offline_vis:
                return False, f"'{vi_name}' is already disconnected"
            return False, f"'{vi_name}' is not a registered VI"

        try:
            vi.disconnect()
        except Exception:  # noqa: BLE001 — a failing hook must not block the release
            logger.exception("disconnect() hook failed on VI '%s'", vi_name)

        vi_type = self._vi_registry.get(vi_name, "system")
        del self._virtual_instruments[vi_name]
        self._vi_registry.pop(vi_name, None)
        self._error_counts.pop(vi_name, None)
        self._last_known_state.pop(vi_name, None)
        # A VI that is no longer live cannot be in a comm fault — clearing it
        # keeps the fault banner from naming an instrument nobody is polling.
        self.clear_fault(vi_name)

        spec = self._vi_specs.get(vi_name, {})
        role_aliases = dict(spec.get("drivers") or {})
        self._release_drivers(
            _exclusive_aliases(role_aliases, self._vi_specs, vi_name, self._vi_registry)
        )

        reason = (
            "Disconnected by the operator — the instrument is free for its "
            "front panel or vendor software. Press Connect to hand it back "
            "to I2AS."
        )
        self.register_offline_vi(
            OfflineInstrument(
                vi_name, vi_type, reason, (), tags=frozenset({"operator"}), since=time.time()
            )
        )
        return True, f"'{vi_name}' disconnected"

    def _release_drivers(self, aliases: Sequence[str]) -> None:
        """Close and forget the named driver sessions, guarded and idempotent.

        Each ``close()`` is individually guarded: a driver that fails to
        release must not stop the others, and a disconnect must always
        succeed (see the connection-lifecycle standard). The alias is dropped
        from the live driver map either way, so a later
        ``connect_instrument()`` builds a fresh instance rather than reusing
        a half-closed session.

        Args:
            aliases: Config aliases of the drivers to release.
        """
        for alias in aliases:
            driver = self._drivers.pop(alias, None)
            if driver is None:
                continue
            closer = getattr(driver, "close", None)
            if not callable(closer):
                logger.warning(
                    "Driver '%s' (%s) has no close() — the session is dropped "
                    "but may stay open until the process exits",
                    alias,
                    type(driver).__name__,
                )
                continue
            try:
                closer()
            except Exception:  # noqa: BLE001 — a disconnect must always succeed
                logger.exception("close() failed on driver '%s'", alias)

    # ------------------------------------------------------------------
    # The station declaration snapshot (core/events.py's StationInfo)
    # ------------------------------------------------------------------

    def station_info(self) -> StationInfo:
        """Return the frozen declaration snapshot of this whole station.

        The Station is the only layer holding both halves of the
        declaration — the VI classes' ``@monitored`` / ``@control`` /
        ``ui_groups`` / ``safety_flags`` declarations and the config the
        ``control_limits`` bounds come from — so it is where the snapshot is
        assembled. ``core.events`` defines the shape and
        ``core.capability_manifest`` renders it; both clients build from
        THIS, never from ``get_vi()``, so neither carries a description of
        its own.

        Built from declarations and config alone: it sends NO command to any
        instrument, which is what lets it describe an offline VI as fully as
        a live one and lets a client ask for the picture at any time,
        outside the tick loop. Every ``control_param_specs()`` override is a
        pure read for exactly this reason (see the purity rule in
        ``virtual_instruments/base.py``), and
        ``tests/test_conformance.py`` builds this against spied drivers to
        keep it that way.

        The snapshot is cached and rebuilt when the station's membership
        changes — a VI joining the live registry or degrading to the offline
        one, i.e. every connect and disconnect — so repeated reads are free
        and ``seq`` counts real rebuilds. Its one live field is each
        instrument's ``availability``, captured at rebuild time; the
        moment-to-moment picture is ``StatusSnapshot``'s job, not this one's.

        Returns:
            The current :class:`~i2as.core.events.StationInfo`.
        """
        if self._station_info is None:
            self._station_info_seq += 1
            self._station_info = StationInfo(
                setup=self._setup_name or "",
                tick_interval_s=self._tick_interval_s,
                instruments=tuple(
                    self._instrument_info(vi_name)
                    for vi_name in self._configured_vi_names()
                ),
                seq=self._station_info_seq,
                ts=time.time(),
            )
        return self._station_info

    def _invalidate_station_info(self) -> None:
        """Drop the cached declaration snapshot so the next read rebuilds it.

        Called wherever the station's membership or an offline record
        changes. Cheap and idempotent: the rebuild itself is pure Python
        over already-loaded declarations.
        """
        self._station_info = None

    def _configured_vi_names(self) -> list[str]:
        """Return every configured VI name — live and offline — in config order.

        Config order is ``devices.yaml``'s order, retained in
        ``_vi_specs``; a Station assembled in-process without a config (a
        test, say) has no specs, so registration order stands in.

        Returns:
            The ordered names, each appearing exactly once.
        """
        known = set(self._virtual_instruments) | set(self._offline_vis)
        ordered = [name for name in self._vi_specs if name in known]
        for name in list(self._virtual_instruments) + list(self._offline_vis):
            if name not in ordered:
                ordered.append(name)
        return ordered

    def _instrument_info(self, vi_name: str) -> InstrumentInfo:
        """Assemble one configured VI's declaration, live or offline.

        An offline VI is described from its CLASS (imported from the
        retained build recipe, which is a pure import — the class was
        already imported once when the station was built), so an instrument
        nobody can reach still says what it would offer. The two things only
        an instance knows are therefore absent for it: a
        ``control_param_specs()`` override's dynamic choices fall back to
        the decorator's own declaration, and the ``control_limits`` bounds
        report ``None``, since those are computed by the VI's ``__init__``
        from the config.

        Args:
            vi_name: A configured VI's name — in the live registry, the
                offline registry, or both sources of its identity.

        Returns:
            The assembled :class:`~i2as.core.events.InstrumentInfo`.
        """
        vi = self._virtual_instruments.get(vi_name)
        offline = self._offline_vis.get(vi_name)
        spec = self._vi_specs.get(vi_name) or {}
        cls = self._declaring_class(vi_name)

        role = self._vi_registry.get(vi_name) or (offline.vi_type if offline else "")
        vi_class = (
            cls.__name__
            if cls is not None
            else str(spec.get("class", "")).rsplit(".", 1)[-1]
        )
        availability = tuple(sorted(self._build_availability(vi_name).tags))

        if cls is None:
            # No importable class: the config named one that cannot be
            # resolved. Report the identity and say nothing it cannot back up.
            return InstrumentInfo(
                name=vi_name,
                vi_class=vi_class,
                role=role,
                availability=availability,
            )

        return InstrumentInfo(
            name=vi_name,
            vi_class=vi_class,
            role=role,
            kind=str(getattr(cls, "vi_type", "")),
            availability=availability,
            monitored=_monitored_infos(cls),
            controls=_control_infos(cls, vi),
            limits=_limit_infos(cls, vi),
            ui_groups=tuple(
                GroupInfo(
                    key=group.key,
                    title=group.title,
                    description=group.description,
                    members=tuple(group.members),
                )
                for group in getattr(cls, "ui_groups", ())
            ),
            safety_flags=dict(cls.merged_safety_flags()),
        )

    def _declaring_class(self, vi_name: str) -> type | None:
        """Return the VI class declaring *vi_name*'s capabilities.

        Args:
            vi_name: A configured VI's name.

        Returns:
            ``type(vi)`` for a live VI; the class named by the retained
            build recipe for an offline one; ``None`` when neither is
            available or the recipe's class cannot be imported (a config
            fault that must not stop the rest of the station describing
            itself).
        """
        vi = self._virtual_instruments.get(vi_name)
        if vi is not None:
            return type(vi)
        dotted = (self._vi_specs.get(vi_name) or {}).get("class")
        if not dotted:
            return None
        try:
            return _import_class(str(dotted))
        except I2ASConfigError:
            logger.warning(
                "Cannot import class '%s' to describe offline VI '%s'",
                dotted,
                vi_name,
            )
            return None

    def get_vi_type(self, vi_name: str) -> str:
        """Return the vi_type for the given VI name.

        Args:
            vi_name: Name of the registered VI.

        Returns:
            The vi_type string (e.g. ``"system"`` or ``"measurement"``).

        Raises:
            KeyError: If no VI with that name exists.
        """
        return self._vi_registry[vi_name]

    def system_setpoint_meta(self, vi_name: str) -> tuple[str, str]:
        """Return ``(label, unit)`` describing a VI's ramp setpoint.

        Reads the VI class's declarative ``setpoint_label`` / ``setpoint_unit``
        (declared once per instrument category), falling back to the VI name and
        no unit. Lets the Orchestrator render human status lines like
        "Ramping field to -1 T" without reaching into VI internals.

        Args:
            vi_name: Name of the registered VI.

        Returns:
            ``(label, unit)``; ``(vi_name, "")`` if the VI is unknown or
            declares no setpoint metadata.
        """
        vi = self._virtual_instruments.get(vi_name)
        label = getattr(vi, "setpoint_label", "") or vi_name
        unit = getattr(vi, "setpoint_unit", "") or ""
        return label, unit

    def measurement_label(self, vi_name: str) -> str:
        """Return a human label for a measurement VI (e.g. "DC resistance").

        Falls back to the VI name if the VI is unknown or declares no
        ``display_label``.
        """
        vi = self._virtual_instruments.get(vi_name)
        return getattr(vi, "display_label", "") or vi_name

    def measurement_selector_label(self, vi_name: str) -> str:
        """Return the SHORT method-selection label for a measurement VI.

        Used for the GUI method-selection drop-down, where a terse name keeps
        the column narrow. Falls back to ``display_label`` (the longer
        status-line label) and then the VI name when ``selector_label`` is empty
        or the VI is unknown. See ``MeasurementInstrumentBase.selector_label``.

        Args:
            vi_name: The measurement VI's registered name.

        Returns:
            The VI's ``selector_label`` if set, else its ``display_label``, else
            ``vi_name``.
        """
        vi = self._virtual_instruments.get(vi_name)
        return (
            getattr(vi, "selector_label", "")
            or getattr(vi, "display_label", "")
            or vi_name
        )

    def __getattr__(self, name: str) -> BaseVirtualInstrument:
        """Attribute-style access to VIs: ``station.magnet_z``.

        Args:
            name: VI name.

        Returns:
            The VI instance.

        Raises:
            AttributeError: If no VI with that name is registered.
        """
        # Guard against infinite recursion during __init__ before
        # _virtual_instruments is set.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            vis = object.__getattribute__(self, "_virtual_instruments")
            if name in vis:
                return vis[name]
        except AttributeError:
            pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            f"No VI named '{name}' is registered."
        )

    # ------------------------------------------------------------------
    # State polling
    # ------------------------------------------------------------------

    @property
    def cached_state(self) -> dict[str, dict]:
        """Return the last known state from the most recent monitor tick.

        No hardware poll. Safe to call from within a procedure's measure().

        Returns:
            ``{vi_name: {field: value, ...}}`` — same structure as get_state().
        """
        return dict(self._last_known_state)

    def last_state_flat(self) -> dict[str, float]:
        """Return cached system VI state as a flat ``{vi_name_field: value}`` dict.

        Reads from the monitor-tick cache — no hardware poll. Only numeric scalar
        values from system VIs are included. Metadata keys (``_stale``,
        ``_disconnected``) and string values (e.g. ``ramp_status``) are excluded.

        Returns:
            Flat dict keyed ``{vi_name}_{monitored_field}`` for all numeric fields.
        """
        result: dict[str, float] = {}
        for vi_name, state in self._last_known_state.items():
            if self._vi_registry.get(vi_name) == "measurement":
                continue
            for key, value in state.items():
                if key.startswith("_"):
                    continue
                if not isinstance(value, (int, float)):
                    continue
                result[f"{vi_name}_{key}"] = float(value)
        return result

    def get_state(self) -> dict[str, dict]:
        """Poll all VIs and return a full state snapshot.

        On ``I2ASCommunicationError``: increment the error counter for
        that VI and return the last known values with ``_stale: True``.
        After ``max_errors`` consecutive failures, also add ``_disconnected: True``.

        Returns:
            ``{vi_name: {field: value, ...}}`` — one sub-dict per VI.
        """
        full_state: dict[str, dict] = {}

        for vi_name, vi in self._virtual_instruments.items():
            self._polling_vi = vi_name
            try:
                state = vi.get_state()
                self._error_counts[vi_name] = 0
                self._last_known_state[vi_name] = state
                full_state[vi_name] = state
                # A successful poll fully clears any standing fault — see
                # clear_fault()'s docstring: an acknowledged fault does not
                # survive recovery either, it simply disappears.
                self.clear_fault(vi_name)
            except I2ASCommunicationError as exc:
                self._error_counts[vi_name] += 1
                stale = dict(self._last_known_state.get(vi_name, {}))
                stale["_stale"] = True
                if self._error_counts[vi_name] >= self._max_errors:
                    stale["_disconnected"] = True
                    logger.error(
                        "VI '%s' disconnected after %d consecutive errors",
                        vi_name,
                        self._error_counts[vi_name],
                    )
                    self._record_comm_condition(vi_name, "disconnected", str(exc))
                else:
                    logger.warning(
                        "VI '%s' communication error (attempt %d/%d)",
                        vi_name,
                        self._error_counts[vi_name],
                        self._max_errors,
                    )
                    self._record_comm_condition(vi_name, "stale", str(exc))
                full_state[vi_name] = stale
            finally:
                self._polling_vi = None

        return full_state

    def polling_vi(self) -> str | None:
        """Return the VI whose read is in flight right now, or ``None``.

        A diagnostic, not a state: it names the one instrument ``get_state()``
        is waiting on, so a read that never returns can be reported by name
        rather than as an anonymous hang. Read by
        ``core/instrument_host.py`` when a bounded shutdown join expires.

        Returns:
            The VI's configured name while a poll is in progress, else
            ``None``.
        """
        return self._polling_vi

    # ------------------------------------------------------------------
    # Unified condition registry — the System-Condition standard's
    # producer-side plumbing shared by every origin (see
    # i2as/core/conditions.py and GLOSSARY.md).
    # ------------------------------------------------------------------

    def _upsert_condition(self, candidate: Condition) -> Condition:
        """Insert *candidate* into the unified registry, or refresh it in place.

        The single point where an existing condition's ``since`` and
        ``acknowledged`` survive a change to its other fields — e.g. a comm
        condition escalating ``"stale"`` -> ``"disconnected"``, or a safety
        condition's ``source_vis``/``message`` changing tick to tick — as
        long as the SAME ``key`` (the same ongoing incident) stays
        continuously present. A key with no prior entry is stored exactly
        as given. This is the one place that logic exists; every producer
        (the comm detector in ``get_state()``, the safety-flag builder in
        ``update_conditions()``) calls it instead of re-implementing the
        since/acknowledged-preservation rule.

        Args:
            candidate: The freshly built condition from a producer.

        Returns:
            The condition actually stored: *candidate*, or *candidate* with
            ``since``/``acknowledged`` overridden from the prior entry.
        """
        existing = self._conditions.get(candidate.key)
        if existing is not None:
            candidate = replace(
                candidate, since=existing.since, acknowledged=existing.acknowledged
            )
        self._conditions[candidate.key] = candidate
        return candidate

    def _record_comm_condition(self, vi_name: str, kind: str, message: str) -> None:
        """Record (or refresh) *vi_name*'s comm-origin condition.

        The producer side of the System-Condition standard's ``"comm"``
        origin (GLOSSARY.md's **Instrument fault**): builds a hold-severity
        `Condition` scoped to just this VI and upserts it via
        `_upsert_condition()`, which preserves ``since``/``acknowledged``
        across a ``"stale"`` -> ``"disconnected"`` escalation of the same
        ongoing incident — the same behavior the pre-unification
        ``FaultRecord`` registry gave.

        Args:
            vi_name: The VI the condition concerns.
            kind: ``"stale"`` or ``"disconnected"``.
            message: Human-readable description of the latest failure.
        """
        self._upsert_condition(
            Condition(
                key=f"comm:{vi_name}",
                origin="comm",
                severity="hold",
                kind=kind,
                source_vis=(vi_name,),
                affected_vis=frozenset({vi_name}),
                message=message,
                since=time.time(),
            )
        )

    def publish_conditions(self, origin: str, conditions: Iterable[Condition]) -> None:
        """Replace *origin*'s active conditions with *conditions*, in one refresh.

        The public, origin-scoped counterpart of the safety producer's
        inline prune-then-upsert (`update_conditions()`, near the bottom of
        this class): delete every existing condition of *origin* that is
        not in the new desired set, then upsert each desired one via
        `_upsert_condition()` (preserving `since`/`acknowledged` for a key
        that stays continuously active). Scoping the prune to *origin* is
        what lets producers refresh on independent cadences without one
        wiping another's conditions mid-cycle — e.g. the safety producer
        refreshes every tick (`update_conditions()`) while the trend-check
        producer refreshes every ~60 s
        (`i2as.core.trend_check_runner.TrendCheckRunner`), and neither
        refresh touches the other's entries.

        Unlike `update_conditions()`, this method builds no `Condition`
        itself — the caller (e.g. `i2as.core.trend_checks.
        conditions_for()`) has already decided severity/scope/message; this
        is purely the registry-refresh mechanics, reusable by any future
        origin that needs the same "refresh my own slice, leave everyone
        else's alone" behaviour.

        Args:
            origin: The origin these conditions belong to. Every condition
                in *conditions* must have this exact `origin` (checked
                below) — a mismatch would silently prune conditions this
                call did not intend to own.
            conditions: This refresh's COMPLETE desired set for *origin*
                (not a delta): a condition of *origin* absent from this set
                is cleared, e.g. because the check/flag/detector that would
                have produced it now passes.

        Raises:
            ValueError: If any condition's `origin` does not equal *origin*.
        """
        desired = {c.key: c for c in conditions}
        for condition in desired.values():
            if condition.origin != origin:
                raise ValueError(
                    f"publish_conditions(origin={origin!r}) received a condition "
                    f"with origin={condition.origin!r} (key={condition.key!r})"
                )
        for key in [k for k, c in self._conditions.items() if c.origin == origin]:
            if key not in desired:
                del self._conditions[key]
        for key, condition in desired.items():
            self._conditions[key] = self._upsert_condition(condition)

    def conditions(self) -> dict[str, Condition]:
        """Return a copy of the unified condition registry, ``{key: Condition}``.

        The read side of the System-Condition standard for callers that
        want the typed `Condition` objects directly rather than through the
        permanent comm-origin adapter below (`vi_faults()`).

        Returns:
            Every condition currently active on this Station, from any
            origin and severity.
        """
        return dict(self._conditions)

    def active_critical_conditions(self) -> tuple[Condition, ...]:
        """Return every critical-severity condition, sorted by key.

        Mirrors `i2as.core.conditions.decide()`'s own `emergency`
        field, but as a direct Station query rather than requiring the
        caller to first collect every condition and call `decide()` itself.

        Returns:
            Critical-severity conditions from any origin, sorted by
            `Condition.key`.
        """
        return tuple(
            sorted(
                (c for c in self._conditions.values() if c.severity == "critical"),
                key=lambda c: c.key,
            )
        )

    def acknowledge_condition(self, key: str) -> bool:
        """Mark a condition as acknowledged by its registry key.

        The origin-agnostic acknowledge primitive the permanent
        `acknowledge_fault()` adapter is built from.

        Args:
            key: The `Condition.key` to acknowledge, e.g.
                ``"comm:magnet_z"`` or ``"safety:quench"``.

        Returns:
            True if a condition with that key exists and was acknowledged;
            False otherwise.
        """
        existing = self._conditions.get(key)
        if existing is None:
            return False
        self._conditions[key] = replace(existing, acknowledged=True)
        return True

    # ------------------------------------------------------------------
    # Runtime fault registry — a transitional (kept FOREVER) adapter of
    # the System-Condition standard's "comm" origin over the unified
    # registry above. The GUI (i2as/gui/instrument_panel.py) reads
    # these through the Orchestrator, so their field semantics must never
    # change even though the storage underneath is now unified.
    # ------------------------------------------------------------------

    def vi_faults(self) -> dict[str, FaultRecord]:
        """Return the current runtime fault registry, ``{vi_name: FaultRecord}``.

        Adapter of the System-Condition standard: synthesizes one
        `FaultRecord` per comm-origin condition in the unified registry,
        preserving the exact field semantics (`kind`, `message`, `since`,
        `acknowledged`) the pre-unification fault registry had.
        """
        result: dict[str, FaultRecord] = {}
        for condition in self._conditions.values():
            if condition.origin != "comm":
                continue
            vi_name = condition.source_vis[0]
            result[vi_name] = FaultRecord(
                vi_name,
                condition.kind,
                condition.message,
                condition.since,
                condition.acknowledged,
            )
        return result

    def acknowledge_fault(self, vi_name: str) -> bool:
        """Mark a VI's active fault as acknowledged (calms the operator UI).

        Adapter of the System-Condition standard over
        `acknowledge_condition()`.

        Args:
            vi_name: Name of the faulted VI.

        Returns:
            True if a fault record existed and was acknowledged; False if
            the VI has no active fault.
        """
        return self.acknowledge_condition(f"comm:{vi_name}")

    def clear_fault(self, vi_name: str) -> None:
        """Remove *vi_name*'s fault record, if any (called on a successful poll).

        Adapter of the System-Condition standard: pops the comm-origin
        condition, if present, from the unified registry. An
        acknowledged-but-recovered fault simply disappears — there is
        nothing left to acknowledge once the instrument is responding
        again, so recovery is not distinguished from "never acknowledged".

        Args:
            vi_name: Name of the VI to clear.
        """
        self._conditions.pop(f"comm:{vi_name}", None)

    def retry_fault(self, vi_name: str) -> tuple[bool, str]:
        """Reset *vi_name*'s error counter and force one fresh poll — or, past
        the disconnect threshold, rebuild its driver session first.

        The runtime counterpart of ``connect_instrument()``: the VI stays in
        the live registry throughout (never demoted to ``_offline_vis``), so
        a run watching it keeps seeing it exactly as before — only its
        session gets refreshed. A ``"stale"`` fault (below ``max_errors``)
        just resets the comm-error streak and re-polls once, since the
        session is presumed fine and only intermittently unresponsive. A
        ``"disconnected"`` fault (``max_errors`` reached) instead delegates
        to ``_retry_disconnected()``: at that point the underlying bus
        session is presumed dead (e.g. a driver like ``OxfordMercuryiPS``
        opens its VISA resource exactly once in ``__init__`` — a hardware
        fix does not revive an already-broken handle), so a bare re-poll on
        the same handle can never recover it; only closing and reopening the
        session, exactly what ``connect_instrument()`` does for an offline
        VI, can.

        Args:
            vi_name: Name of the (registered, live) VI to retry.

        Returns:
            An explicit ``(ok, message)`` verdict, mirroring
            ``connect_instrument()``'s style: ``message`` is a human-readable
            success confirmation or failure reason.
        """
        vi = self._virtual_instruments.get(vi_name)
        if vi is None:
            return False, f"'{vi_name}' is not a registered VI"

        condition = self._conditions.get(f"comm:{vi_name}")
        if condition is not None and condition.kind == "disconnected":
            return self._retry_disconnected(vi_name)

        self._error_counts[vi_name] = 0
        try:
            state = vi.get_state()
        except I2ASCommunicationError as exc:
            self._error_counts[vi_name] = 1
            message = str(exc)
            self._record_comm_condition(vi_name, "stale", message)
            logger.warning("Retry of '%s' failed: %s", vi_name, message)
            return False, f"'{vi_name}' still not responding: {message}"
        self._error_counts[vi_name] = 0
        self._last_known_state[vi_name] = state
        self.clear_fault(vi_name)
        logger.info("Retry of '%s' succeeded — fault cleared", vi_name)
        return True, f"'{vi_name}' responded — fault cleared"

    def _retry_disconnected(self, vi_name: str) -> tuple[bool, str]:
        """Rebuild vi_name's driver session(s) from scratch, then re-verify.

        The ``"disconnected"``-kind branch of ``retry_fault()``: closes and
        reopens every driver alias *vi_name* exclusively owns, reconstructs
        the VI, and re-runs the identity check — the same construction
        sequence ``connect_instrument()`` uses for an offline VI, applied
        in place instead of via the offline registry. ``vi_name`` stays in
        ``_virtual_instruments`` for the whole call: even a failed rebuild
        leaves it live (still faulted), never demoted to ``_offline_vis``,
        so a run watching it is never bypassed by this recovering silently
        out from under it. A driver alias another live VI still needs is
        left untouched, mirroring ``disconnect_instrument()``.

        Args:
            vi_name: Name of the registered, disconnected VI to rebuild.

        Returns:
            An explicit ``(ok, message)`` verdict, mirroring
            ``retry_fault()``'s style.
        """
        spec = self._vi_specs.get(vi_name)
        if spec is None:
            return False, f"No build recipe retained for '{vi_name}'"
        role_aliases = dict(spec.get("drivers") or {})

        # The old session(s) are presumed dead at this point (this is the
        # disconnect threshold) — release vi_name's exclusive aliases so the
        # loop below actually reopens them instead of reusing broken handles.
        self._release_drivers(
            _exclusive_aliases(role_aliases, self._vi_specs, vi_name, self._vi_registry)
        )

        for alias in dict.fromkeys(role_aliases.values()):
            if alias in self._drivers:
                continue
            driver_cfg = self._driver_specs.get(alias, {})
            try:
                cls = _import_class(driver_cfg["class"])
                self._drivers[alias] = cls(driver_cfg.get("address", "SIM"))
            except Exception as exc:  # noqa: BLE001 — verdict, never a crash, in GUI context
                return self._fail_disconnected_retry(vi_name, f"driver '{alias}': {exc}")

        driver_refs = {role: self._drivers[alias] for role, alias in role_aliases.items()}
        init_params = dict(spec.get("init_params", {}) or {})
        try:
            cls = _import_class(spec["class"])
            vi = cls(driver_refs, **init_params)
        except Exception as exc:  # noqa: BLE001 — verdict, never a crash, in GUI context
            return self._fail_disconnected_retry(vi_name, str(exc))

        vi.vi_name = vi_name
        if not _identity_check(vi_name, vi):
            self._release_drivers(_exclusive_aliases(role_aliases, self._vi_specs, vi_name))
            return self._fail_disconnected_retry(vi_name, _IDENTITY_FAILED_REASON)

        self._virtual_instruments[vi_name] = vi
        self._error_counts[vi_name] = 0
        try:
            state = vi.get_state()
        except I2ASCommunicationError as exc:
            self._error_counts[vi_name] = 1
            self._record_comm_condition(vi_name, "stale", str(exc))
            logger.warning(
                "Rebuild retry of '%s' opened a fresh session but the first poll failed: %s",
                vi_name,
                exc,
            )
            return False, f"'{vi_name}' reconnected but has not responded yet: {exc}"

        self._last_known_state[vi_name] = state
        self.clear_fault(vi_name)
        logger.info("Rebuild retry of '%s' succeeded — fresh session, fault cleared", vi_name)
        return True, f"'{vi_name}' reconnected — fault cleared"

    def _fail_disconnected_retry(self, vi_name: str, reason: str) -> tuple[bool, str]:
        """Record a failed rebuild attempt and return its verdict.

        Shared tail of ``_retry_disconnected()``'s three failure branches
        (driver open, VI construction, identity check): each fails the same
        way — the error counter stays pinned at the disconnect threshold and
        the comm condition is refreshed with the latest reason.

        Args:
            vi_name: Name of the VI whose rebuild failed.
            reason: Human-readable failure description.

        Returns:
            ``(False, message)``.
        """
        self._error_counts[vi_name] = self._max_errors
        self._record_comm_condition(vi_name, "disconnected", reason)
        logger.warning("Rebuild retry of '%s' failed: %s", vi_name, reason)
        return False, f"'{vi_name}' still not responding: {reason}"

    # ------------------------------------------------------------------
    # Ramp management
    # ------------------------------------------------------------------

    def process_system_targets(self, system_targets: dict[str, Target]) -> None:
        """Dispatch ramp targets to system VIs.

        Only VIs whose ``vi_type == "system"`` are valid ramp targets. Each
        value is a ``Target``; its ``rate`` is forwarded to ``start_ramp()``
        only when not ``None``, so VIs that do not accept one (most VIs do
        not) are unaffected.

        Args:
            system_targets: Mapping of VI name → ``Target``.

        Raises:
            ValueError: If a named VI is not registered or not a system VI.
        """
        for vi_name, tgt in system_targets.items():
            if vi_name not in self._virtual_instruments:
                raise ValueError(f"process_system_targets: unknown VI '{vi_name}'")
            if self._vi_registry[vi_name] != "system":
                raise ValueError(
                    f"process_system_targets: VI '{vi_name}' is not a system VI "
                    f"(type={self._vi_registry[vi_name]})"
                )
            vi = self._virtual_instruments[vi_name]
            if not isinstance(vi, RampableVI):
                raise ValueError(
                    f"process_system_targets: VI '{vi_name}' does not implement RampableVI"
                )
            target = tgt.target
            kwargs: dict[str, Any] = {}
            if tgt.rate is not None:
                kwargs["rate"] = tgt.rate
            logger.info("Starting ramp on '%s' to target=%s", vi_name, target)
            vi.start_ramp(target, **kwargs)  # type: ignore[call-arg]

    def advance_ramps(self) -> set[str]:
        """Step every active system-VI ramp generator forward by one tick.

        This is the ONLY thing that makes a ramp progress: a ramp is a
        generator yielding one step per tick (the single-threaded
        cooperative design), and nothing else calls ``advance_ramp()``. So
        every tick must reach this method, in whatever state, or an
        in-flight ramp silently freezes. The one deliberate exception is
        PAUSED, where holding the hardware still IS the intent.

        Returns:
            The names of the system VIs still ramping after this step.
        """
        still_ramping: set[str] = set()
        for vi_name, vi_type in self._vi_registry.items():
            if vi_type != "system":
                continue
            vi = self._virtual_instruments[vi_name]
            if not isinstance(vi, RampableVI):
                continue
            if vi.ramp_status() == "RAMPING":
                vi.advance_ramp()
                still_ramping.add(vi_name)
            # IDLE (no ramp active) and TARGET_REACHED are both "done".
        return still_ramping

    def check_ramps(self, vi_names: set[str] | None = None) -> bool:
        """Advance all active system VI ramps and report completion for a subset.

        Two separable jobs, kept in one call because a tick in a ramp-aware
        state wants both:

        1. **Advance** every ramp, regardless of *vi_names* — see
           ``advance_ramps()``. Narrowing the advance to the scope would
           freeze an unwatched ramp mid-flight rather than merely not
           waiting for it.
        2. **Report** completion for the VIs in *vi_names* only — the
           ramp-scope standard: a caller waits for the ramps IT started,
           never for hardware someone else is moving. An empty set
           therefore means "nothing to wait for" and reports ``True``,
           which is exactly right for a run that commanded no targets.

        Mirrors ``stop_ramps(vi_names)``, which scopes the same way.

        Args:
            vi_names: Report completion over these system VIs only.
                ``None`` (the default) reports over every system VI, the
                whole-station question the Orchestrator asks when no run
                owns the hardware.

        Returns:
            ``True`` if every in-scope system VI has reached its target (or
            is IDLE); ``False`` if any in-scope VI is still ramping.
        """
        still_ramping = self.advance_ramps()
        if vi_names is None:
            return not still_ramping
        return not (still_ramping & vi_names)

    def stop_ramps(self, vi_names: set[str] | None = None) -> None:
        """Stop active ramps and hold hardware where it is.

        Calls ``stop_ramp()`` on every system VI implementing ``RampableVI``
        (or only those in *vi_names* if given). Each call is individually
        guarded: a dead instrument must not prevent the others from stopping.

        Args:
            vi_names: Restrict to these VI names; ``None`` means all system VIs.
        """
        for vi_name, vi_type in self._vi_registry.items():
            if vi_type != "system":
                continue
            if vi_names is not None and vi_name not in vi_names:
                continue
            vi = self._virtual_instruments[vi_name]
            if not isinstance(vi, RampableVI):
                continue
            try:
                vi.stop_ramp()
            except Exception:
                logger.exception("stop_ramp failed on VI '%s'", vi_name)

    def get_ramp_status(self) -> dict[str, dict]:
        """Aggregate each system VI's ramp value, setpoint, target, rate, and status.

        The single aggregation point for the ramp-introspection standard
        (``RampableVI``'s optional hooks): both the operational-status record
        ("what is the run waiting on, and how far from setpoint?") and the
        ramp tracker ("what is ramping right now, and where to?") read this
        one snapshot, so a tick never polls a VI's ramp state twice. For every
        system VI implementing ``RampableVI``, collects ``ramp_value()`` /
        ``ramp_setpoint()`` / ``ramp_target()`` / ``ramp_rate()`` (user units —
        tesla, kelvin, degrees; ``None`` if the VI does not expose them), its
        ``ramp_phase()``, its ``ramp_status()`` string, and the class's
        ``no_motion_phases`` declaration (the phases in which a still value
        is expected, which the stall detector reads from this snapshot). Each
        VI is guarded individually: a communication error on one instrument
        yields a stale entry rather than breaking the whole snapshot.

        Returns:
            ``{vi_name: {"value": float|None, "setpoint": float|None,
            "target": float|None, "rate": float|None, "ramp_status": str,
            "phase": str|None, "no_motion_phases": frozenset[str]}}`` for
            every system VI. ``setpoint`` is the NEXT setpoint the hardware
            is driving to (an intermediate ramp step), ``target`` the END
            setpoint the ramp finishes at. A VI that raised on read also
            carries ``"_stale": True``.
        """
        result: dict[str, dict] = {}
        for vi_name, vi_type in self._vi_registry.items():
            if vi_type != "system":
                continue
            vi = self._virtual_instruments[vi_name]
            if not isinstance(vi, RampableVI):
                continue
            try:
                result[vi_name] = {
                    "value": vi.ramp_value(),
                    "setpoint": vi.ramp_setpoint(),
                    "target": vi.ramp_target(),
                    "rate": vi.ramp_rate(),
                    "ramp_status": vi.ramp_status(),
                    "phase": vi.ramp_phase(),
                    "no_motion_phases": vi.no_motion_phases,
                }
            except I2ASCommunicationError:
                result[vi_name] = {
                    "value": None,
                    "setpoint": None,
                    "target": None,
                    "rate": None,
                    "ramp_status": "IDLE",
                    "phase": None,
                    "no_motion_phases": vi.no_motion_phases,
                    "_stale": True,
                }
        return result

    def nominal_ramp_rates(self) -> dict[str, float]:
        """Return each system VI's declared ramp rate, in user units per minute.

        The declaration counterpart of ``get_ramp_status()``: what every
        ramping VI WOULD ramp at, read from config alone (see
        ``RampableVI.nominal_ramp_rate``) rather than from an active ramp, so
        it can be answered before any run exists and without touching the bus.
        The **duration estimate** (``core/estimates.py``) turns a proposed
        run's declared setpoints into a time with exactly this mapping; a VI
        that declares no rate is simply absent, which the estimate then
        reports as an explicit assumption instead of counting as instant.

        Returns:
            ``{vi_name: rate_per_minute}`` for every system VI that declares
            one — tesla/min, kelvin/min, degrees/min, matching each VI's own
            ``ramp_target()`` units.
        """
        rates: dict[str, float] = {}
        for vi_name, vi_type in self._vi_registry.items():
            if vi_type != "system":
                continue
            vi = self._virtual_instruments[vi_name]
            if not isinstance(vi, RampableVI):
                continue
            rate = vi.nominal_ramp_rate()
            if rate is None or rate <= 0:
                continue
            rates[vi_name] = float(rate)
        return rates

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def check_safety(self, state: dict[str, dict] | None = None) -> dict[str, bool]:
        """Aggregate every VI's safety verdict from a state snapshot.

        Each VI judges its own state fragment via ``evaluate_safety()``
        (magnet VIs report ``quench``). No hardware is polled here — pass
        the snapshot the
        monitor tick already collected, or omit it to use the cached one.

        Args:
            state: Snapshot from ``get_state()``. ``None`` uses the last
                known state (no hardware poll).

        Returns:
            ``{flag_name: bool}`` — a flag is True if ANY VI tripped it.
        """
        if state is None:
            state = self._last_known_state
        flags: dict[str, bool] = {}
        for vi_name, vi in self._virtual_instruments.items():
            vi_state = state.get(vi_name, {})
            try:
                for flag, tripped in vi.evaluate_safety(vi_state).items():
                    flags[flag] = flags.get(flag, False) or bool(tripped)
            except Exception:
                logger.exception("evaluate_safety failed on VI '%s'", vi_name)
        return flags

    def safety_flag_sources(self, state: dict[str, dict] | None = None) -> dict[str, list[str]]:
        """Map each tripped safety flag to the VI name(s) that tripped it.

        A parallel accessor to ``check_safety()`` (same OR-combination logic)
        that additionally names the originating instrument(s) — used by the
        Orchestrator so an EMERGENCY reason and its ``ErrorEvent`` can name
        the instrument, without changing ``check_safety()``'s existing
        ``{flag: bool}`` signature (other callers are unaffected).

        Args:
            state: Snapshot from ``get_state()``. ``None`` uses the last
                known state (no hardware poll).

        Returns:
            ``{flag_name: [vi_name, ...]}`` — VI names in registration
            order, each listed at most once per flag. A flag absent from
            the mapping was never tripped by any VI.
        """
        if state is None:
            state = self._last_known_state
        sources: dict[str, list[str]] = {}
        for vi_name, vi in self._virtual_instruments.items():
            vi_state = state.get(vi_name, {})
            try:
                for flag, tripped in vi.evaluate_safety(vi_state).items():
                    if not tripped:
                        continue
                    names = sources.setdefault(flag, [])
                    if vi_name not in names:
                        names.append(vi_name)
            except Exception:
                logger.exception("evaluate_safety failed on VI '%s'", vi_name)
        return sources

    def get_concerned_vis(self, flag: str) -> list[str]:
        """Return the names of every VI whose ``safety_concerns()`` names *flag*.

        The consumer-side lookup the safety-hold standard is built on: when
        *flag* trips, these are the VIs a hold applies to — never the VI
        that reported the flag (see ``safety_flag_sources()`` for that).

        Args:
            flag: The safety flag name, e.g. ``"coolant_low"``.

        Returns:
            VI names, in registration order, whose ``safety_concerns()``
            includes *flag*. Empty if no VI depends on it.
        """
        return [
            vi_name
            for vi_name, vi in self._virtual_instruments.items()
            if flag in vi.safety_concerns()
        ]

    def _flag_severity(self, flag: str, producing_vis: tuple[str, ...]) -> str | None:
        """Return *flag*'s declared severity, read off its producing VI(s).

        Severity is a property of the flag itself, never of which VI
        happens to report it (the Safety-flag manifest standard, see
        ``BaseVirtualInstrument``'s docstring) — this simply reads the
        first producing VI's merged manifest that declares it.

        Args:
            flag: The safety flag name.
            producing_vis: The VI names ``safety_flag_sources()`` attributes
                this flag to, in reporting order.

        Returns:
            The declared severity (``"advisory"`` | ``"hold"`` |
            ``"critical"``), or ``None`` if no producing VI's manifest
            declares it (a conformance violation elsewhere — see
            ``tests/test_conformance.py``'s
            ``test_evaluate_safety_flags_are_declared_in_manifest`` — so
            this should not occur on a conforming station).
        """
        for vi_name in producing_vis:
            vi = self._virtual_instruments.get(vi_name)
            if vi is None:
                continue
            manifest = type(vi).merged_safety_flags()
            if flag in manifest:
                return manifest[flag]
        return None

    def update_conditions(
        self,
        safety: Mapping[str, bool],
        *,
        tolerated_flags: frozenset[str] | set[str] = frozenset(),
    ) -> None:
        """Refresh the safety-origin conditions from this tick's ``check_safety()`` result.

        The write side of the System-Condition standard's ``"safety"``
        origin (see ``i2as/core/conditions.py`` and GLOSSARY.md's
        **Safety hold**): scope follows from severity alone, so every
        tripped flag produces EXACTLY ONE `Condition`, keyed
        ``f"safety:{flag}"``, built from the flag's OWN declared severity
        (``_flag_severity()``, read off the producing VI's merged
        ``safety_flags`` manifest):

        - ``"critical"`` — always constructed, tolerance never applies
          (GLOSSARY.md's **Critical safety flag** — station-wide by
          definition, ``affected_vis=None``). Critical IS station-wide
          scope; no VI's ``safety_concerns()`` is ever consulted for it.
        - ``"advisory"`` — always constructed (no enforcement reads it, so
          tolerance is moot); ``affected_vis=None``.
        - ``"hold"`` — scoped to ``get_concerned_vis(flag)``: concerns
          exist only for hold-severity flags. If *flag* is tolerated, or
          no VI's ``safety_concerns()`` names it, no condition is built
          (a WARNING is logged once per flag in the latter case — there
          is no one to hold).

        A flag no longer tripped, or a flag that just became tolerated,
        has its condition removed. `_upsert_condition()` preserves an
        existing condition's ``since``/``acknowledged`` across ticks the
        same key stays active — see its docstring.

        Args:
            safety: The ``{flag: bool}`` dict ``check_safety()`` already
                computed this tick, passed in rather than recomputed here —
                the Orchestrator calls ``check_safety()`` exactly once per
                tick regardless of how many places consult the result.
            tolerated_flags: Hold-severity flags to treat as untripped —
                the active run's own ``tolerated_safety_flags`` (empty when
                the run declares none, or when no run is active). Resolved
                by the Orchestrator and passed in: the Station never imports
                the procedure layer (contract C8) and has no notion of "the
                active run" itself. This is what lets a run that exists to
                clear a hold-severity condition run to completion instead of
                being held on (and having its own run failed for) the very
                condition it addresses. Never applied to a flag's own
                critical/advisory condition (see above) — a critical or
                advisory flag never becomes a hold in the first place.
        """
        tolerated = frozenset(tolerated_flags)
        now = time.time()
        sources = self.safety_flag_sources()
        tripped = {flag for flag, is_tripped in safety.items() if is_tripped}
        non_tolerated = tripped - tolerated

        desired: dict[str, Condition] = {}
        for flag in tripped:
            producing_vis = tuple(sources.get(flag, []))
            severity = self._flag_severity(flag, producing_vis)
            if severity is None:
                continue
            message = f"Safety flag '{flag}' is tripped"
            key = f"safety:{flag}"

            if severity in ("critical", "advisory"):
                desired[key] = Condition(
                    key=key,
                    origin="safety",
                    severity=severity,
                    kind=flag,
                    source_vis=producing_vis,
                    affected_vis=None,
                    message=message,
                    since=now,
                )
                continue

            # severity == "hold": scoped to the flag's concerned VIs, and
            # subject to tolerance — the only severity either applies to.
            if flag not in non_tolerated:
                continue
            concerned = self.get_concerned_vis(flag)
            if not concerned:
                if flag not in self._warned_unconsumed_flags:
                    logger.warning(
                        "Safety flag '%s' is tripped but no VI's "
                        "safety_concerns() names it — no hold condition "
                        "constructed.",
                        flag,
                    )
                    self._warned_unconsumed_flags.add(flag)
                continue
            desired[key] = Condition(
                key=key,
                origin="safety",
                severity="hold",
                kind=flag,
                source_vis=producing_vis,
                affected_vis=frozenset(concerned),
                message=message,
                since=now,
            )

        for key in [k for k, c in self._conditions.items() if c.origin == "safety"]:
            if key not in desired:
                del self._conditions[key]
        for key, condition in desired.items():
            self._conditions[key] = self._upsert_condition(condition)

    # ------------------------------------------------------------------
    # Measurement command dispatch
    # ------------------------------------------------------------------

    def send_measurement_commands(
        self, commands: Sequence[Command], *, allowed_scope: str = "measurement"
    ) -> None:
        """Dispatch an ordered sequence of ``Command`` calls to VIs.

        Commands are dispatched in order (order is semantically meaningful —
        e.g. a switch heater must settle before a source arms). An unknown VI
        or an unknown method is logged at WARNING and skipped; an exception
        raised by the VI method itself propagates to the caller.

        Capability-scope enforcement (the standard in GLOSSARY.md's
        "Capability scope" entry): every command's target method is resolved
        and its ``@control`` scope checked BEFORE any command in the batch is
        dispatched. A method requiring ``"operation"`` scope when
        *allowed_scope* is ``"measurement"`` rejects the whole batch — the
        plan is refused before any hardware is touched, exactly like an
        envelope violation. An undecorated method (no ``@control``, e.g. a
        measurement VI's ``initiate``/``standby`` lifecycle) defaults to
        ``"measurement"`` scope.

        Args:
            commands: Ordered sequence of ``Command`` objects to dispatch.
            allowed_scope: The submitting plan's capability scope —
                ``"measurement"`` (default, procedures) or ``"operation"``
                (operations; operation-scope plans may also carry
                measurement-scope commands).

        Raises:
            I2ASSafetyError: If any command's target method requires a
                capability scope not covered by *allowed_scope*. Nothing is
                dispatched.
        """
        resolved: list[tuple[Command, Any]] = []
        for cmd in commands:
            vi = self._virtual_instruments.get(cmd.vi_name)
            if vi is None:
                logger.warning("send_measurement_commands: unknown VI '%s'", cmd.vi_name)
                resolved.append((cmd, None))
                continue
            method = getattr(vi, cmd.method, None)
            if method is None:
                logger.warning(
                    "send_measurement_commands: VI '%s' has no method '%s'",
                    cmd.vi_name,
                    cmd.method,
                )
                resolved.append((cmd, None))
                continue
            required_scope = getattr(method, "_control_scope", "measurement")
            if required_scope == "operation" and allowed_scope != "operation":
                raise I2ASSafetyError(
                    f"send_measurement_commands: '{cmd.vi_name}.{cmd.method}' "
                    f"requires operation-scope access, but this plan is "
                    f"{allowed_scope}-scope. Command refused before dispatch."
                )
            resolved.append((cmd, method))

        # Validated as a whole batch above — now dispatch, in order.
        for cmd, method in resolved:
            if method is None:
                continue
            logger.debug("Calling %s.%s(%s)", cmd.vi_name, cmd.method, cmd.kwargs)
            method(**cmd.kwargs)

    # ------------------------------------------------------------------
    # VI action dispatch
    # ------------------------------------------------------------------

    def execute_vi_action(
        self,
        vi_name: str,
        method_name: str,
        *,
        allowed_scope: str = "measurement",
        **kwargs: Any,
    ) -> Any:
        """Call one capability on a named VI — the direct action path.

        The single entry point through which a manual action (a GUI click, an
        agent call) reaches an instrument, as opposed to a procedure's
        ``Command`` batch, which goes through
        ``send_measurement_commands()``. See GLOSSARY.md's **Direct action
        path**.

        Three admission checks run BEFORE the method is called, mirroring the
        read/write split ``troubleshoot.engine.DriverBench.call()`` enforces
        one layer down. Each raises its own subclass of
        ``I2ASActionRefusedError`` with its own reason string, so a caller
        gets a specific verdict rather than an undifferentiated failure:

        1. **Private name** — a leading underscore is a VI's internal API and
           is never a capability.
        2. **Undeclared capability** — the method must carry ``@control`` or
           be one of ``LIFECYCLE_ACTIONS`` (``initiate``/``standby``). A
           ``@monitored`` poller, a procedure-only helper such as
           ``take_reading()``, or any other public method is refused.
        3. **Capability scope** — an ``@control(scope="operation")`` method is
           refused unless *allowed_scope* is ``"operation"`` too, exactly as
           ``send_measurement_commands()`` refuses an operation-scope command
           in a measurement-scope plan.

        *allowed_scope* defaults to the RESTRICTIVE ``"measurement"``, so a
        caller that never thinks about scope cannot reach an operation-scope
        capability by accident. A human at the instrument front panel is the
        operation authority, so the Orchestrator's manual-action path passes
        ``"operation"`` explicitly (see ``Orchestrator.submit_vi_action()``);
        that opt-in is the deliberate, single place where the wider scope is
        granted.

        Beyond these checks the method's own guards still apply: the
        control-validation standard's ``control_limits`` wrapper raises
        ``I2ASSafetyError`` for an out-of-limit value before any hardware
        command is sent. The session envelope is checked one layer up, by the
        Orchestrator, which is where it lives.

        Args:
            vi_name: Name of the target VI.
            method_name: Name of the capability to call.
            allowed_scope: The caller's capability scope — ``"measurement"``
                (default) or ``"operation"``.
            **kwargs: Keyword arguments forwarded to the method.

        Returns:
            Return value of the method (if any).

        Raises:
            KeyError: If no VI named *vi_name* is registered (an absent or
                disconnected instrument — there is nothing to dispatch to).
            I2ASPrivateActionError: If *method_name* starts with ``_``.
            AttributeError: If the VI has no such method.
            I2ASUndeclaredActionError: If the method is neither
                ``@control`` nor a lifecycle action.
            I2ASActionScopeError: If the method's capability scope is
                outside *allowed_scope*.
            I2ASSafetyError: If the method's own ``control_limits`` guard
                refuses the value. Nothing is sent to the instrument.
        """
        vi = self._virtual_instruments[vi_name]
        if method_name.startswith("_"):
            raise I2ASPrivateActionError(
                f"execute_vi_action: '{vi_name}.{method_name}' is a private "
                f"name — the direct action path dispatches capabilities only, "
                f"never a VI's internal API. Action refused."
            )
        method = getattr(vi, method_name)
        is_control = getattr(method, "_is_control", False)
        if not is_control and method_name not in LIFECYCLE_ACTIONS:
            raise I2ASUndeclaredActionError(
                f"execute_vi_action: '{vi_name}.{method_name}' is not a "
                f"declared capability — it carries no @control and is not one "
                f"of {sorted(LIFECYCLE_ACTIONS)}. Action refused."
            )
        required_scope = get_control_scope(method) if is_control else "measurement"
        if required_scope == "operation" and allowed_scope != "operation":
            raise I2ASActionScopeError(
                f"execute_vi_action: '{vi_name}.{method_name}' requires "
                f"operation-scope access, but this caller is "
                f"{allowed_scope}-scope. Action refused."
            )
        logger.debug("Dispatching %s.%s(%s)", vi_name, method_name, kwargs)
        return method(**kwargs)

    # ------------------------------------------------------------------
    # Setpoint capabilities (the setpoint-parameter convention)
    # ------------------------------------------------------------------

    def setpoint_parameters(self, vi_name: str, method_name: str) -> tuple[str, ...]:
        """Return one capability's setpoint parameters, in declaration order.

        The setpoint-parameter convention (see ``core/plan.py``'s
        ``SETPOINT_PARAM_PREFIX`` and ``ExperimentEnvelope``): a ``@control``
        parameter named ``target_*`` carries the VI's enveloped quantity — the
        same quantity ``RampableVI.start_ramp(target)`` takes and a ``Target``
        commands. It is what lets the session envelope bind a manual action as
        well as a procedure's target, without the Orchestrator having to know
        what any particular instrument measures.

        Args:
            vi_name: Name of the VI.
            method_name: Name of the capability.

        Returns:
            The matching parameter names, or ``()`` when the VI, the method,
            or a setpoint parameter is absent.
        """
        vi = self._virtual_instruments.get(vi_name)
        if vi is None:
            return ()
        method = getattr(vi, method_name, None)
        if method is None or not getattr(method, "_is_control", False):
            return ()
        return tuple(
            name
            for name in getattr(method, "_control_params", {})
            if name.startswith(SETPOINT_PARAM_PREFIX)
        )

    def envelope_variables(self) -> dict[str, EnvelopeVariable]:
        """Return the enveloped quantity of every system VI that declares one.

        The read side of the setpoint-parameter convention, and the source
        the Start Experiment dialog's envelope editor pre-fills from: for each
        registered VI with a ``target_*`` ``@control`` parameter, the
        capability that commands it and the setup's own bounds on it, taken
        from the ``control_limits`` limit the config populated. An experiment's
        envelope NARROWS those bounds, so the operator adjusts numbers that
        are already there rather than composing them from nothing.

        Returns:
            ``{vi_name: EnvelopeVariable}``, empty when no VI declares a
            setpoint capability. A VI whose setpoint parameter is not covered
            by ``control_limits`` still appears, with ``None`` bounds.
        """
        variables: dict[str, EnvelopeVariable] = {}
        for vi_name, vi in self._virtual_instruments.items():
            limits = type(vi).control_limits
            for method_name in get_control_methods(vi):
                params = self.setpoint_parameters(vi_name, method_name)
                if not params:
                    continue
                param_name = params[0]
                limit_name = limits.get(method_name, {}).get(param_name)
                lo, hi = vi.limit_bounds(limit_name) if limit_name else (None, None)
                variables[vi_name] = EnvelopeVariable(
                    vi_name=vi_name,
                    method_name=method_name,
                    param_name=param_name,
                    config_min=lo,
                    config_max=hi,
                )
                break
        return variables

    # ------------------------------------------------------------------
    # Bulk lifecycle
    # ------------------------------------------------------------------

    def initiate_all(self) -> None:
        """Call ``initiate()`` on every registered VI.

        Each successful call records that VI's lifecycle state (the
        lifecycle-state standard), so a bulk bring-up that dispatches no
        per-VI action still reaches every client on the next snapshot.
        """
        for vi_name, vi in self._virtual_instruments.items():
            logger.info("Initiating VI '%s'", vi_name)
            try:
                vi.initiate()
            except Exception:
                logger.exception("Error initiating VI '%s'", vi_name)

    def standby_all(self) -> None:
        """Call ``standby()`` on every registered VI.

        The blanket stand-down an emergency takes (``Orchestrator.
        _enter_emergency()``). It deliberately bypasses the per-VI action
        queue, so it emits no ``action_succeeded`` — which is exactly why
        each VI records its own lifecycle state (the lifecycle-state
        standard) instead: the next ``StatusSnapshot`` reports every
        instrument as stood down whether or not any client saw an action.
        """
        for vi_name, vi in self._virtual_instruments.items():
            logger.info("Putting VI '%s' into standby", vi_name)
            try:
                vi.standby()
            except Exception:
                logger.exception("Error during standby of VI '%s'", vi_name)


# ── The station declaration snapshot's builders ───────────────────────────────
#
# Module-level and class-driven: everything below reads a VI CLASS's
# declarations (plus, where an instance exists, its config-derived bounds and
# instance-aware ParamSpecs), so an offline instrument describes itself as
# fully as a live one. Nothing here touches a driver — see
# `Station.station_info()`.


def _declared_names(cls: type, marker: str) -> list[str]:
    """Return *cls*'s capability method names carrying *marker*, in declared order.

    Declared order is base-class-first, definition order within each class,
    which is the order a reader of the source meets them and the order a
    manifest lists them. A method a subclass overrides keeps the position
    where it was FIRST declared; one a subclass redefines without the
    decorator drops out, since the check is against the resolved attribute.

    ``decorators.get_monitored_methods()`` / ``get_control_methods()`` answer
    the same question over ``dir()``, which sorts alphabetically — fine for
    "which are there", wrong for "in what order do they read".

    Args:
        cls: The Virtual Instrument class.
        marker: The decorator's marker attribute, ``"_is_monitored"`` or
            ``"_is_control"``.

    Returns:
        The method names, in declared order, each once.
    """
    ordered: list[str] = []
    for klass in reversed(cls.__mro__):
        for name, attr in vars(klass).items():
            if name in ordered or not callable(attr):
                continue
            if getattr(attr, marker, False):
                ordered.append(name)
    return [
        name for name in ordered if getattr(getattr(cls, name, None), marker, False)
    ]


def _type_name(annotation: Any) -> str:
    """Render a type annotation as the manifest's plain-string type name.

    Args:
        annotation: A resolved annotation (``float``, ``float | None``,
            ``dict[str, float]``), or ``None`` for an undeclared one.

    Returns:
        ``"float"``, ``"float | None"``, …; ``""`` when undeclared.
    """
    if annotation is None:
        return ""
    name = getattr(annotation, "__name__", None)
    if isinstance(name, str):
        return name
    return str(annotation).replace("typing.", "")


def _return_type_name(method: Any) -> str:
    """Return the declared return type of one ``@monitored`` method.

    Unwraps the decorator's ``functools.wraps`` chain first: the wrapper
    carries the original's ``__annotations__`` but this module's globals,
    so resolving a string annotation on the wrapper would fail.

    Args:
        method: The bound or unbound monitored method.

    Returns:
        The rendered type name, or ``""`` when it declares none or the
        annotation cannot be resolved.
    """
    try:
        hints = typing.get_type_hints(inspect.unwrap(method))
    except Exception:  # noqa: BLE001 — an unresolvable hint is simply undeclared
        return ""
    return _type_name(hints.get("return"))


def _json_scalar(value: Any) -> Any:
    """Return *value* if it is a JSON scalar the contract can carry, else ``None``.

    Only reached for a control that declares no ``ParamSpec`` and whose
    signature default is something exotic; the declaration standard makes
    that impossible on a shipped VI, and reporting ``None`` beats failing
    the whole snapshot over one oddity.

    Args:
        value: A candidate default.

    Returns:
        The value, or ``None``.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def _param_json(
    param_name: str, spec: ParamSpec | None, signature_info: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Render one ``@control`` parameter for the declaration snapshot.

    Args:
        param_name: The parameter's name.
        spec: Its declared ``ParamSpec``, or ``None`` when the control
            declared none (which the declaration standard forbids on a
            shipped VI — the signature is then all there is to go on).
        signature_info: The decorator's signature record for the parameter
            (``type``, ``default``), used only when there is no spec.

    Returns:
        A JSON-safe dict carrying ``name``, ``declared``, ``kind``, ``unit``,
        ``description``, ``default``, ``min``, ``max`` and ``choices``.
        ``declared`` says whether a ``ParamSpec`` is behind the rest: a
        parameter known only from its signature reports the same empty unit
        and absent bounds a spec that declares none would, so a renderer
        needs the flag to tell "declared nothing" from "declared none of it"
        — and only a declared parameter can be rendered as the typed widget
        (or the tool schema) its spec describes.
    """
    if spec is not None:
        return {
            "name": param_name,
            "declared": True,
            "kind": spec.type.__name__,
            "unit": spec.unit,
            "description": spec.description,
            "default": spec.default,
            "min": spec.min,
            "max": spec.max,
            "choices": dict(spec.choices) if spec.choices else None,
        }
    info = signature_info or {}
    return {
        "name": param_name,
        "declared": False,
        "kind": _type_name(info.get("type")),
        "unit": "",
        "description": "",
        "default": _json_scalar(info.get("default")),
        "min": None,
        "max": None,
        "choices": None,
    }


def _monitored_infos(cls: type) -> tuple[MonitoredInfo, ...]:
    """Render every ``@monitored`` reading *cls* declares.

    Args:
        cls: The Virtual Instrument class.

    Returns:
        One ``MonitoredInfo`` per reading, in declared order. An UNDECLARED
        unit renders as ``""`` — the declaration standard forbids it on a
        shipped VI, and `tests/test_conformance.py` is where that is caught.
    """
    infos: list[MonitoredInfo] = []
    for name in _declared_names(cls, "_is_monitored"):
        method = getattr(cls, name)
        infos.append(
            MonitoredInfo(
                name=name,
                unit=get_monitored_unit(method) or "",
                description=get_monitored_description(method),
                group=get_ui_group(method),
                returns=_return_type_name(method),
            )
        )
    return tuple(infos)


def _control_infos(
    cls: type, vi: BaseVirtualInstrument | None
) -> tuple[ControlInfo, ...]:
    """Render every ``@control`` action *cls* declares.

    Args:
        cls: The Virtual Instrument class.
        vi: The live instance, when there is one. Its
            ``control_param_specs()`` is consulted so an instance-aware
            override's dynamic choices (a switch's config-named routes) are
            captured; an offline VI falls back to the decorator's own
            declaration.

    Returns:
        One ``ControlInfo`` per action, in declared order, its ``params`` in
        signature order.
    """
    infos: list[ControlInfo] = []
    for name in _declared_names(cls, "_is_control"):
        method = getattr(cls, name)
        signature_info: Mapping[str, Any] = getattr(method, "_control_params", {})
        if vi is not None:
            specs = vi.control_param_specs(name)
        else:
            specs = get_control_specs(method)
        infos.append(
            ControlInfo(
                name=name,
                scope=get_control_scope(method),
                action_class=get_control_action_class(method),
                panel=get_control_panel(method),
                group=get_ui_group(method),
                params=tuple(
                    _param_json(
                        param_name, specs.get(param_name), signature_info.get(param_name)
                    )
                    for param_name in signature_info
                ),
            )
        )
    return tuple(infos)


def _limit_infos(cls: type, vi: BaseVirtualInstrument | None) -> dict[str, Any]:
    """Render the control-validation standard's declared limits and their bounds.

    Args:
        cls: The Virtual Instrument class, whose ``control_limits`` names
            which parameter each limit guards.
        vi: The live instance, whose ``control_limit_bounds()`` supplies the
            values its ``__init__`` derived from the config. ``None`` for an
            offline VI, whose bounds were never computed.

    Returns:
        ``{method: {param: {"limit": name, "min": lo, "max": hi}}}``.
        ``min``/``max`` are ``None`` where that side is unbounded and for
        every parameter of an offline VI.
    """
    bounds = vi.control_limit_bounds() if vi is not None else {}
    limits: dict[str, Any] = {}
    for method_name, param_map in (getattr(cls, "control_limits", {}) or {}).items():
        method_bounds = bounds.get(method_name, {})
        limits[method_name] = {
            param_name: {
                "limit": limit_name,
                "min": method_bounds.get(param_name, (None, None))[0],
                "max": method_bounds.get(param_name, (None, None))[1],
            }
            for param_name, limit_name in param_map.items()
        }
    return limits


# ── Factory ───────────────────────────────────────────────────────────────────


def _import_class(dotted_path: str) -> type:
    """Import and return a class from a dotted module path.

    Args:
        dotted_path: E.g. ``"i2as.virtual_instruments.magnet.superconducting_magnet.SuperconductingMagnetVI"``.

    Returns:
        The class object.

    Raises:
        I2ASConfigError: If the import fails.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise I2ASConfigError(
            f"Cannot import '{dotted_path}': {exc}"
        ) from exc


def build_station(config_path: str) -> Station:
    """Construct a fully populated Station from a YAML config directory.

    Expected directory layout::

        config_path/
          devices.yaml   — driver and VI definitions
          monitor.yaml   — tick interval and error threshold

    Degraded build: an instrument that fails to *connect* never aborts the
    build. Each driver whose ``__init__`` raises is recorded, and every VI
    that needs it (or whose own construction raises a communication error) is
    registered offline via ``Station.register_offline_vi()`` instead of live —
    the GUI shows why and offers a reconnect. Config errors (missing files,
    unimportable classes, unknown driver references) still raise, because a
    broken config is a software fault the degraded mode cannot reason about.

    Args:
        config_path: Path to the directory containing devices.yaml and monitor.yaml.

    Returns:
        A ``Station`` instance with every connectable VI registered live,
        every unconnectable one registered offline, and error threshold set.

    Raises:
        I2ASConfigError: If required config keys are missing or imports fail.
        FileNotFoundError: If the config directory or files are missing.
    """
    try:
        from ruamel.yaml import YAML  # type: ignore
    except ImportError as exc:
        raise I2ASConfigError("ruamel.yaml is required but not installed") from exc

    config_dir = Path(config_path)
    devices_file = config_dir / "devices.yaml"
    monitor_file = config_dir / "monitor.yaml"

    if not devices_file.exists():
        raise FileNotFoundError(f"devices.yaml not found in {config_dir}")
    if not monitor_file.exists():
        raise FileNotFoundError(f"monitor.yaml not found in {config_dir}")

    yaml = YAML()

    with devices_file.open("r", encoding="utf-8") as f:
        devices_config: dict = dict(yaml.load(f))

    with monitor_file.open("r", encoding="utf-8") as f:
        monitor_config: dict = dict(yaml.load(f))

    station = Station()
    # Setup identity for anything that reports which setup it is running
    # (the operational-status record's ``setup`` field).
    station._setup_name = config_dir.name

    # Apply monitor config
    mon = monitor_config.get("monitor", {})
    station._max_errors = int(mon.get("max_vi_errors", 3))
    try:
        station._tick_interval_s = (
            float(mon.get("tick_interval_ms", DEFAULT_TICK_INTERVAL_MS)) / 1000.0
        )
    except (TypeError, ValueError):
        logger.warning(
            "monitor.tick_interval_ms in '%s' is not a number; the station "
            "declaration snapshot reports the default cadence",
            config_dir,
        )

    # --- Build all real drivers ---
    # A driver __init__'s job is to open the hardware connection, so ANY
    # construction failure here is treated as a connection fault: recorded,
    # never raised. Import errors (config faults) still raise via _import_class.
    drivers_map: dict[str, Any] = {}
    offline_drivers: dict[str, str] = {}  # {alias: failure reason}
    for driver_name, driver_cfg in (devices_config.get("real_drivers") or {}).items():
        cls = _import_class(driver_cfg["class"])
        resource = driver_cfg.get("address", "SIM")
        station._driver_specs[driver_name] = dict(driver_cfg)
        try:
            drivers_map[driver_name] = cls(resource)
        except Exception as exc:  # noqa: BLE001 — any driver-construction failure degrades, see above
            offline_drivers[driver_name] = str(exc)
            logger.warning(
                "Driver '%s' (%s) failed to connect: %s",
                driver_name,
                driver_cfg["class"],
                exc,
            )
            continue
        logger.info("Built driver '%s' (%s)", driver_name, driver_cfg["class"])
    station._drivers = drivers_map

    # --- Build all VIs ---
    for vi_name, vi_cfg in (devices_config.get("virtual_instruments") or {}).items():
        cls = _import_class(vi_cfg["class"])
        station._vi_specs[vi_name] = dict(vi_cfg)
        vi_type = vi_cfg.get("vi_type", "system")

        # Resolve driver references. An unknown alias is a config error and
        # raises; an alias whose driver failed to connect sends this VI to
        # the offline registry instead.
        driver_refs: dict[str, Any] = {}
        failed_aliases: list[str] = []
        for role, driver_name in (vi_cfg.get("drivers") or {}).items():
            if driver_name in offline_drivers:
                failed_aliases.append(driver_name)
            elif driver_name not in drivers_map:
                raise I2ASConfigError(
                    f"VI '{vi_name}' references unknown driver '{driver_name}'"
                )
            else:
                driver_refs[role] = drivers_map[driver_name]

        if failed_aliases:
            unique = list(dict.fromkeys(failed_aliases))
            reason = "; ".join(
                f"driver '{alias}': {offline_drivers[alias]}" for alias in unique
            )
            station.register_offline_vi(
                OfflineInstrument(vi_name, vi_type, reason, tuple(unique), since=time.time())
            )
            continue

        init_params = dict(vi_cfg.get("init_params", {}) or {})
        try:
            vi = cls(driver_refs, **init_params)
        except I2ASCommunicationError as exc:
            # Drivers came up but the VI's own bring-up could not talk to the
            # hardware. Other exceptions (bad init_params, limit-validation
            # errors) are config/software faults and propagate.
            station.register_offline_vi(
                OfflineInstrument(vi_name, vi_type, str(exc), (), since=time.time())
            )
            continue

        # The build's ONE command per instrument: an identity query (the
        # connection-lifecycle standard, see BaseVirtualInstrument). An open
        # session is not proof the instrument is answering, and an instrument
        # that does not answer degrades to the offline registry instead of
        # pretending to be live.
        vi.vi_name = vi_name
        if not _identity_check(vi_name, vi):
            station.register_offline_vi(
                OfflineInstrument(
                    vi_name, vi_type, _IDENTITY_FAILED_REASON, (), since=time.time()
                )
            )
            continue
        station.register_vi(vi_name, vi, vi_type)

    offline = station.offline_vi_names()
    logger.info(
        "Station built with %d VIs (%d offline) from '%s'",
        len(station.get_vi_names()) + len(offline),
        len(offline),
        config_dir,
    )
    return station


def build_station_with_fallback(
    candidate_paths: list[str],
) -> tuple[Station, str, list[str]]:
    """Build a Station from the first usable config, falling back in order.

    Each candidate is validated (``validate_config_dir``) and then built; the
    first that succeeds wins. This is the startup safety net for *config*
    faults: a corrupted config no longer crashes the app, because a later
    candidate (ultimately the always-loadable ``sim_cryostat``) takes over.

    Unreachable instruments never trigger a fallback: ``build_station()``
    degrades them to the offline registry and still succeeds, so the user
    stays on their own setup's config with everything else working.

    Args:
        candidate_paths: Config directories to try, most-preferred first.
            Callers should end the list with a guaranteed-safe config.

    Returns:
        A ``(station, used_path, warnings)`` tuple. ``warnings`` describes each
        candidate that was skipped, for surfacing to the user.

    Raises:
        I2ASConfigError: If no candidate could be built.
    """
    warnings: list[str] = []
    for path in candidate_paths:
        errors = validate_config_dir(path)
        if errors:
            warnings.append(f"Config '{path}' is invalid ({errors[0]}); skipped.")
            continue
        try:
            station = build_station(path)
            return station, path, warnings
        except Exception as exc:  # noqa: BLE001 — fallback must catch any build failure
            warnings.append(f"Config '{path}' failed to load ({exc}); skipped.")
    raise I2ASConfigError(
        f"No usable config among {candidate_paths}: {'; '.join(warnings)}"
    )


def validate_config_dir(config_path: str) -> list[str]:
    """Check a config directory without instantiating any driver or VI.

    A dry-run for the config editor and startup fallback: it parses both YAML
    files and verifies that every declared class is importable and that every
    VI's driver references resolve to a defined driver. It deliberately does
    **not** call any class constructor, so validating a real-hardware config
    never opens a VISA session or touches an instrument.

    Args:
        config_path: Path to the config directory (containing devices.yaml and
            monitor.yaml).

    Returns:
        A list of human-readable error strings. An empty list means the config
        is structurally valid and safe to load.
    """
    try:
        from ruamel.yaml import YAML  # type: ignore
    except ImportError:
        return ["ruamel.yaml is required but not installed"]

    config_dir = Path(config_path)
    devices_file = config_dir / "devices.yaml"
    monitor_file = config_dir / "monitor.yaml"

    errors: list[str] = []
    if not devices_file.exists():
        errors.append(f"devices.yaml not found in {config_dir}")
    if not monitor_file.exists():
        errors.append(f"monitor.yaml not found in {config_dir}")
    if errors:
        return errors

    yaml = YAML()
    try:
        with devices_file.open("r", encoding="utf-8") as f:
            devices_config = dict(yaml.load(f) or {})
    except Exception as exc:  # noqa: BLE001 — any YAML parse failure is a config error
        return [f"devices.yaml is not valid YAML: {exc}"]
    try:
        with monitor_file.open("r", encoding="utf-8") as f:
            yaml.load(f)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"monitor.yaml is not valid YAML: {exc}")

    real_drivers = devices_config.get("real_drivers") or {}
    if not isinstance(real_drivers, dict):
        return errors + ["'real_drivers' must be a mapping"]
    for driver_name, driver_cfg in real_drivers.items():
        if not isinstance(driver_cfg, dict) or "class" not in driver_cfg:
            errors.append(f"driver '{driver_name}' is missing a 'class'")
            continue
        try:
            _import_class(driver_cfg["class"])
        except I2ASConfigError as exc:
            errors.append(f"driver '{driver_name}': {exc}")

    virtual_instruments = devices_config.get("virtual_instruments") or {}
    if not isinstance(virtual_instruments, dict):
        return errors + ["'virtual_instruments' must be a mapping"]
    for vi_name, vi_cfg in virtual_instruments.items():
        if not isinstance(vi_cfg, dict) or "class" not in vi_cfg:
            errors.append(f"VI '{vi_name}' is missing a 'class'")
            continue
        try:
            _import_class(vi_cfg["class"])
        except I2ASConfigError as exc:
            errors.append(f"VI '{vi_name}': {exc}")
        for role, driver_name in (vi_cfg.get("drivers") or {}).items():
            if driver_name not in real_drivers:
                errors.append(
                    f"VI '{vi_name}' role '{role}' references unknown driver "
                    f"'{driver_name}'"
                )

    return errors
