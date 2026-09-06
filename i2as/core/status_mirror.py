"""StatusMirror — the GUI's local copy of everything the engine last said.

The **status-mirror standard** (GLOSSARY.md's *Status mirror*): a client never
reads the engine synchronously. It holds one of these, fed by the engine's
event stream, and answers every read off the last message it saw. Two things
follow, and the responsive-GUI design depends on both:

* A read costs nothing and can never block. The engine may be forty seconds
  deep inside one ``measure()``; a widget asking "which run is in flight?"
  gets an answer from local memory either way.
* Every GUI guard clause becomes **advisory** rather than authoritative. A
  mirror is one event-loop hop stale by construction, so it may inform a
  button's wording or hide a control, but the engine is the only authority on
  whether an action happens: the client asks, and the engine refuses.

Every read here is named after the ``Orchestrator`` accessor it replaces, so
"which mirror call answers ``held_vi_names()``?" never needs a lookup table,
and the values it returns are the ``StatusSnapshot``'s JSON-safe renderings of
those accessors' records rather than the records themselves — a mirror carries
what crossed the boundary, never a live object from the other side of it.

The stream is a broadcast, so a mirror built after the engine missed the
``StationInfo`` emitted at construction. Whoever BUILDS the engine therefore
primes the mirror through ``prime()`` — with ``Orchestrator.station_info()``,
``status_snapshot()`` and ``get_operational_status()``, taken once on the
engine's own thread — and hands the GUI a mirror that already knows what the
machine is doing. ``for_engine()`` is that same wiring for a caller holding an
engine on its own thread (tests, and the inline construction path). Those
three reads are the ONLY calls this module makes into the engine, and the only
ones anywhere under ``gui/``; a conformance test keeps it that way.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from PyQt6.QtCore import QObject, pyqtSignal

from i2as.core.events import (
    InstrumentInfo,
    LifecycleState,
    StateChange,
    StationInfo,
    StatusSnapshot,
)

logger = logging.getLogger(__name__)


class EventSource(Protocol):
    """The slice of the engine a mirror attaches to.

    Structural rather than nominal so the mirror never imports the
    Orchestrator: anything that broadcasts the control contract's events and
    can answer the three priming reads can feed one — the engine itself
    today, the instrument host's proxy tomorrow.
    """

    event_emitted: Any
    operational_status: Any

    def station_info(self) -> StationInfo:
        """Return the station's declaration snapshot (a priming read)."""
        ...

    def status_snapshot(self) -> StatusSnapshot:
        """Return this moment's status snapshot (a priming read)."""
        ...

    def get_operational_status(self) -> dict[str, Any]:
        """Return the latest operational-status record (a priming read)."""
        ...


class StatusMirror(QObject):
    """The last picture of the engine a client was given, as pure reads.

    Fed by ``on_event()`` (the ``StatusSnapshot`` / ``StateChange`` /
    ``StationInfo`` members of the event stream) and ``on_operational_status()``
    (the per-tick troubleshooting record, which travels on its own stream).
    Everything else on the stream is ignored: a mirror answers reads, and a
    ``Datapoint`` is not a read.

    Signals:
        status_updated: A fresh ``StatusSnapshot`` landed (per tick and on
            every state change).
        state_changed: A ``StateChange`` landed, carrying cause and actor.
        station_updated: A ``StationInfo`` landed — the station's declaration
            changed, i.e. an instrument connected or disconnected.
        operational_status_updated: A fresh operational-status record landed.

    Args:
        parent: Optional Qt parent.
    """

    status_updated = pyqtSignal(object)  # events.StatusSnapshot
    state_changed = pyqtSignal(object)  # events.StateChange
    station_updated = pyqtSignal(object)  # events.StationInfo
    operational_status_updated = pyqtSignal(dict)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._snapshot = StatusSnapshot(state="IDLE")
        self._station = StationInfo()
        self._operational_status: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Attachment
    # ------------------------------------------------------------------

    @classmethod
    def for_engine(
        cls, engine: EventSource, parent: QObject | None = None
    ) -> StatusMirror:
        """Build a mirror primed from *engine* and connected to its streams.

        The priming reads are taken here, so this must run on the thread
        the engine lives on — in production that is the instrument host,
        inside the engine's own thread, before the mirror is handed to the
        GUI. A caller that already holds a primed mirror passes it down
        instead of calling this.

        Args:
            engine: The event source to mirror.
            parent: Optional Qt parent for the new mirror.

        Returns:
            The primed, connected mirror.
        """
        mirror = cls(parent=parent)
        mirror.prime(
            engine.station_info(),
            engine.status_snapshot(),
            engine.get_operational_status(),
        )
        mirror.attach(engine)
        return mirror

    @classmethod
    def of(cls, engine: Any, parent: QObject | None = None) -> StatusMirror:
        """Return the mirror *engine* already carries, or build one for it.

        The fallback a widget uses when it is handed a client but no mirror:
        an ``OrchestratorProxy`` carries its own (primed by the host, on the
        engine's thread), and a bare Orchestrator gets one built here, which
        is the inline construction path tests take.

        Args:
            engine: An ``OrchestratorProxy``, or an engine to mirror.
            parent: Optional Qt parent, used only when one is built.

        Returns:
            The mirror to read from.
        """
        carried = getattr(engine, "status", None)
        if isinstance(carried, StatusMirror):
            return carried
        return cls.for_engine(engine, parent=parent)

    def attach(self, engine: EventSource) -> None:
        """Connect this mirror to an engine's two broadcast streams.

        Args:
            engine: The event source whose ``event_emitted`` and
                ``operational_status`` signals feed this mirror.
        """
        engine.event_emitted.connect(self.on_event)
        engine.operational_status.connect(self.on_operational_status)

    def prime(
        self,
        station_info: StationInfo | None = None,
        snapshot: StatusSnapshot | None = None,
        operational_status: dict[str, Any] | None = None,
    ) -> None:
        """Seed the mirror with what the engine already broadcast.

        Args:
            station_info: The station's declaration, or ``None`` to leave the
                current one in place.
            snapshot: This moment's status, or ``None`` to leave the current
                one in place.
            operational_status: The latest per-tick troubleshooting record,
                or ``None`` to leave the current one in place.
        """
        if station_info is not None:
            self._station = station_info
        if snapshot is not None:
            self._snapshot = snapshot
        if operational_status is not None:
            self._operational_status = dict(operational_status)

    # ------------------------------------------------------------------
    # Feeds
    # ------------------------------------------------------------------

    def on_event(self, event: object) -> None:
        """Absorb one message from the engine's event stream.

        Args:
            event: Any member of the control contract's ``Event`` union.
                Anything that is not a read — a ``Datapoint``, a
                ``RunStarted`` — is ignored, which is what keeps the mirror
                a picture of the machine rather than a log of it.
        """
        if isinstance(event, StatusSnapshot):
            self._snapshot = event
            self.status_updated.emit(event)
        elif isinstance(event, StationInfo):
            self._station = event
            self.station_updated.emit(event)
        elif isinstance(event, StateChange):
            self.state_changed.emit(event)

    def on_operational_status(self, record: dict[str, Any]) -> None:
        """Absorb one per-tick operational-status record.

        Args:
            record: The record as emitted on ``Orchestrator.operational_status``.
                Copied on the way in, so a later mutation by the emitter (or
                by a reader) can never reach through the mirror.
        """
        self._operational_status = dict(record)
        self.operational_status_updated.emit(dict(self._operational_status))

    # ------------------------------------------------------------------
    # Reads — one per engine accessor, answered from the last snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> StatusSnapshot:
        """Return the last ``StatusSnapshot`` seen (frozen, safe to hold)."""
        return self._snapshot

    @property
    def state(self) -> str:
        """The engine's state name as of the last snapshot (``"IDLE"``, …)."""
        return self._snapshot.state

    def run(self) -> dict[str, Any] | None:
        """Return the active run's summary, or ``None`` when nothing runs.

        Returns:
            A copy of ``{"id", "name", "kind", "progress", "step", "steps"}``
            — optional keys omitted rather than guessed — or ``None``.
        """
        return dict(self._snapshot.run) if self._snapshot.run is not None else None

    def run_owner(self) -> dict[str, Any] | None:
        """Return the **run owner** of the run in flight, or ``None``.

        The read half of the run-ownership standard (GLOSSARY.md's *Run
        owner*): who started the run every client is watching, so a panel can
        say whose run it is and a client can tell before it asks whether a
        run-scoped action of its own would be a **takeover**. The engine
        enforces the rule regardless; this is the mirror of the same fact.

        Returns:
            The owner's ``Actor.ref()`` — ``{"kind", "id"}`` — or ``None``
            when nothing is running (and for a run summary that carries no
            owner at all, which nothing this engine emits does).
        """
        run = self._snapshot.run
        if not run:
            return None
        owner = run.get("owner")
        return dict(owner) if isinstance(owner, dict) else None

    def is_monitoring(self) -> bool:
        """Return whether the per-tick monitoring cycle was polling."""
        return self._snapshot.is_monitoring

    def pause_pending(self) -> bool:
        """Return whether a pause is waiting for the current datapoint."""
        return self._snapshot.pause_pending

    def active_run_kind(self) -> str | None:
        """Return ``"procedure"``, or ``None`` when idle."""
        return self._snapshot.active_run_kind

    def override_active(self, vi_name: str | None = None) -> bool:
        """Return whether the EMERGENCY manual override is in force.

        Args:
            vi_name: A VI to ask about, or ``None`` for the station-wide
                answer.

        Returns:
            ``True`` while the override is unlocked for that scope.
        """
        if vi_name is None:
            return self._snapshot.override_active
        entry = self._snapshot.instruments.get(vi_name) or {}
        return bool(entry.get("override_active", False))

    def manual_override_expires_at(self) -> float | None:
        """Return the soonest-expiring override's unix time, or ``None``."""
        return self._snapshot.manual_override_expires_at

    def held_vi_names(self) -> frozenset[str]:
        """Return every VI under a hold-severity condition."""
        return frozenset(self._snapshot.held_vi_names)

    def active_ramps(self) -> tuple[dict[str, Any], ...]:
        """Return one JSON dict per ramp running as of the last snapshot."""
        return tuple(dict(record) for record in self._snapshot.active_ramps)

    def availabilities(self) -> dict[str, dict[str, Any]]:
        """Return ``{vi_name: availability dict}`` for every configured VI."""
        return {
            name: dict(record)
            for name, record in self._snapshot.availabilities.items()
        }

    def availability(self, vi_name: str) -> dict[str, Any]:
        """Return one VI's availability record.

        Args:
            vi_name: The VI to ask about.

        Returns:
            A copy of the record, or ``{}`` for a VI the last snapshot did
            not carry (one configured after it, or none at all).
        """
        return dict(self._snapshot.availabilities.get(vi_name) or {})

    def availability_tags(self, vi_name: str) -> frozenset[str]:
        """Return the Availability tags one VI carried, as a set.

        The snapshot renders the record's ``frozenset`` as a sorted list, so
        this is where it becomes a set again for the ``"tag" in tags`` reads
        the panels do.

        Args:
            vi_name: The VI to ask about.

        Returns:
            Its tags, empty for a fully usable (or unknown) instrument.
        """
        return frozenset(self.availability(vi_name).get("tags") or ())

    def lifecycle_state(self, vi_name: str) -> str:
        """Return what one instrument is DOING, as of the last snapshot.

        The client read of the lifecycle-state standard (GLOSSARY.md's
        **Lifecycle state**): the answer ``Station.lifecycle_states()`` gave
        when the snapshot was taken, carried on
        ``StatusSnapshot.instruments[vi_name]["lifecycle"]``. This is what an
        instrument card's Initiate/Standby toggle renders, so a stand-down
        that dispatched no per-VI action still reaches the operator.

        Args:
            vi_name: The VI to ask about.

        Returns:
            One of ``LifecycleState``'s values; ``"idle"`` for a VI the last
            snapshot did not carry (one configured after it, or none at all)
            — the same answer an offline instrument gets, and the
            conservative one for a toggle deciding whether to offer Standby.
        """
        entry = self._snapshot.instruments.get(vi_name) or {}
        return str(entry.get("lifecycle") or LifecycleState.IDLE.value)

    def vi_faults(self) -> dict[str, dict[str, Any]]:
        """Return ``{vi_name: fault dict}`` for every faulted VI."""
        return {name: dict(record) for name, record in self._snapshot.vi_faults.items()}

    def vi_fault(self, vi_name: str) -> dict[str, Any] | None:
        """Return one VI's active fault record, or ``None``.

        Args:
            vi_name: The VI to ask about.

        Returns:
            A copy of its ``FaultRecord`` fields (``kind``, ``message``,
            ``acknowledged``, …), or ``None`` when it carries no fault.
        """
        record = self._snapshot.vi_faults.get(vi_name)
        return dict(record) if record else None

    def offline_reason(self, vi_name: str) -> str:
        """Return why one VI is offline, or ``""`` when it is not.

        Args:
            vi_name: The VI to ask about.

        Returns:
            The offline registry's human-readable reason.
        """
        return self._snapshot.offline_reason.get(vi_name, "")

    def envelope_variables(self) -> dict[str, dict[str, Any]]:
        """Return ``{vi_name: envelope-variable dict}`` for the session envelope."""
        return {
            name: dict(record)
            for name, record in self._snapshot.envelope_variables.items()
        }

    def attended(self) -> bool:
        """Return whether a human is watching this experiment.

        A session-owned policy value pushed down into the engine and mirrored
        on every snapshot, so every client reads the same fact rather than
        keeping a copy of its own. ``True`` until told otherwise, which is
        the restrictive reading for an agent asking to act alone.
        """
        return self._snapshot.attended

    def agent_gate(self) -> str:
        """Return the kill switch's setting, one of ``AgentGate``'s values."""
        return self._snapshot.agent_gate

    def get_operational_status(self) -> dict[str, Any]:
        """Return the most recent operational-status record.

        Returns:
            A copy of the record, empty until the first tick delivers one.
        """
        return dict(self._operational_status)

    # ------------------------------------------------------------------
    # Reads — the station's declaration
    # ------------------------------------------------------------------

    def station_info(self) -> StationInfo:
        """Return the station's declaration as last published (frozen)."""
        return self._station

    def instrument_info(self, vi_name: str) -> InstrumentInfo | None:
        """Return one configured instrument's declaration.

        The panels' single source for what an instrument reads, what it can
        be asked to do and how those capabilities group — never
        ``Station.get_vi()``, which is an object on the engine's side of the
        boundary.

        Args:
            vi_name: The configured VI's name.

        Returns:
            Its :class:`~i2as.core.events.InstrumentInfo`, or ``None``
            when the declaration names no such instrument.
        """
        for info in self._station.instruments:
            if info.name == vi_name:
                return info
        return None
