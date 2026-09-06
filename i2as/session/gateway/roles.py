"""The gateway's permission model: who may take which class of action.

**The standard.** Every connection through the **Agent gateway** declares a
``Role``; every action it can take has an ``ActionClass`` (see
``action_classes.py``); and exactly one table below — ``PERMISSION_MATRIX``
— says which pairs are permitted. Nothing else in the gateway decides
authority: a new command, a new capability or a new role is granted by
adding a row to a table, never by writing a branch.

The matrix:

===============  ==========  ====================  =========  ===============
Action class     observer    debug                 session    operator (human)
===============  ==========  ====================  =========  ===============
read             permitted   permitted             permitted  permitted
recovery         refused     unattended only       permitted  permitted
run_control      refused     refused               permitted  permitted
envelope         refused     refused               refused    permitted
===============  ==========  ====================  =========  ===============

Four properties of that table are the design, not incidental:

* **The human column is not in it.** ``authorize()`` returns ``None`` for
  any actor that is not an ``agent``: the operator's authority comes from
  standing at the cryostat, and a permission model that could refuse them
  would be a hazard rather than a safeguard. The engine's own admission
  rules, the control-validation standard's limits and the session envelope
  still bind the human exactly as before — this layer adds nothing to them.
* **Envelope is nobody's.** The session envelope, **Attendance** and the
  **Kill switch** are the rules the other three rows are judged by. An
  agent that could widen the envelope it is bounded by, declare itself
  unattended, or reopen its own gate would not be bounded, attended or
  gated at all.
* **Recovery is attendance-dependent for ``debug`` alone.** With a human
  present a debug agent diagnoses and REPORTS; the human decides. Enforced
  here from the value ``Orchestrator.set_attendance()`` published, never
  left to an agent's self-restraint.
* **Emergency standby is outside the table.** ``authorize()`` permits it to
  every role, in every state, at every kill-switch setting, before the
  matrix is consulted. An actor that can see a problem must never be unable
  to make the station safe.

One rule sits AFTER the matrix, and it is about the run rather than the
role: the **run-ownership standard** (GLOSSARY.md's *Run owner*). The one
run-scoped command — ``abort_procedure`` — ends somebody's result, and an
``agent`` that did not start the run may not take it on it. Refused, not
forbidden: the same command carrying
``override_owner`` and a non-empty ``reason`` is admitted as a **takeover**
and recorded as one. It is checked last because it is the narrowest question
of the lot — a role that may not run the experiment at all is refused by the
matrix, on the authority it lacks, rather than being told whose run it is.
The engine enforces the same rule at the single writer (``Orchestrator``'s
``command`` decorator); this check is the front door, so a tool call is
refused fast, in the same words, and lands in the **Agent feed** either way.

The **kill switch** is checked before the matrix and can only ever subtract:
``read_only`` leaves an agent nothing but ``read``-class actions, ``revoked``
leaves it nothing at all. It is enforced a second time inside the engine
(``Orchestrator.submit()``), which is the authority — this check is the front
door, so that an agent gets a specific refusal rather than a generic one.

One door narrows the model further. A request that arrives through the
**Request spool** — a file dropped into a directory, rather than an
in-process connection — is judged by ``authorize_spooled()``, which caps the
role the file may declare at the setup's configured ``spool_max_role`` before
handing the command on to ``authorize()``. The cap subtracts exactly as the
kill switch does and, like it, never applies to emergency standby.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from enum import Enum
from typing import Any

from i2as.core.events import (
    Actor,
    ActorKind,
    AgentGate,
    Command,
    CommandName,
    StationInfo,
    Verdict,
    VerdictCode,
)
from i2as.session.gateway.action_classes import (
    ActionClass,
    UnclassifiedActionError,
    classify_command,
)

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """The authority a gateway connection declares for itself.

    Written into ``Actor.role`` on every command the connection submits, so
    a refusal, a run record and the action feed all name the authority the
    action was taken under. A ``str`` enum so the value is JSON-safe as it
    stands.

    Members:
        OBSERVER: Reads the system and changes nothing.
        DEBUG: Reads, and — only while the experiment is UNATTENDED — takes
            recovery actions to keep a run alive.
        SESSION: Runs the experiment: recovery and run control both, within
            the session envelope the human set.
    """

    OBSERVER = "observer"
    DEBUG = "debug"
    SESSION = "session"


class Permission(str, Enum):
    """One cell of ``PERMISSION_MATRIX``.

    Members:
        PERMITTED: The role may take actions of this class.
        UNATTENDED_ONLY: Permitted only while **Attendance** is ``False``;
            with a human present the agent reports instead of acting.
        REFUSED: The role may never take actions of this class.
    """

    PERMITTED = "permitted"
    UNATTENDED_ONLY = "unattended_only"
    REFUSED = "refused"


#: The permission model, as one table. Read it as
#: ``PERMISSION_MATRIX[action_class][role]``; every (class, role) pair has a
#: cell, which conformance asserts, so authority is never absent by omission.
PERMISSION_MATRIX: dict[ActionClass, dict[Role, Permission]] = {
    ActionClass.READ: {
        Role.OBSERVER: Permission.PERMITTED,
        Role.DEBUG: Permission.PERMITTED,
        Role.SESSION: Permission.PERMITTED,
    },
    ActionClass.RECOVERY: {
        Role.OBSERVER: Permission.REFUSED,
        Role.DEBUG: Permission.UNATTENDED_ONLY,
        Role.SESSION: Permission.PERMITTED,
    },
    ActionClass.RUN_CONTROL: {
        Role.OBSERVER: Permission.REFUSED,
        Role.DEBUG: Permission.REFUSED,
        Role.SESSION: Permission.PERMITTED,
    },
    ActionClass.ENVELOPE: {
        Role.OBSERVER: Permission.REFUSED,
        Role.DEBUG: Permission.REFUSED,
        Role.SESSION: Permission.REFUSED,
    },
}


#: The commands the **run-ownership standard** scopes to the run's owner, by
#: their ``CommandName``. It mirrors ``Orchestrator.OWNER_SCOPED_COMMANDS``,
#: which is the authority — this module may read the contract but not the
#: engine, so the set is restated here rather than imported, and conformance
#: diffs the two so they cannot drift.
OWNER_SCOPED_COMMANDS: frozenset[CommandName] = frozenset(
    {
        CommandName.ABORT_PROCEDURE,
    }
)


def _refusal(
    command: Command,
    actor: Actor,
    reason: str,
    detail: dict[str, object],
    seq: int,
) -> Verdict:
    """Build the one refusal shape this module ever emits.

    Args:
        command: The command being refused.
        actor: Who asked.
        reason: The human-readable explanation, for a banner or a log.
        detail: The structured explanation a client decides from.
        seq: Sequence number to stamp on the verdict.

    Returns:
        A ``BLOCKED_ROLE`` verdict answering *command*.
    """
    logger.info("Gateway refusal: %s", reason)
    return Verdict(
        request_id=command.request_id,
        command=command.name,
        code=VerdictCode.BLOCKED_ROLE,
        actor=actor,
        reason=reason,
        detail=dict(detail),
        seq=seq,
    )


def authorize(
    actor: Actor,
    command: Command,
    station_info: StationInfo,
    attendance: bool,
    kill_switch: AgentGate,
    *,
    seq: int = 0,
    run_owner: Mapping[str, Any] | None = None,
) -> Verdict | None:
    """Decide whether *actor* may submit *command*, per the standard above.

    The checks run in this order, and the order is the model: emergency
    standby first (always permitted), then the actor kind (only agents are
    judged), then the role's own validity, then the action's class, then the
    kill switch, then the matrix, and last of all run ownership. Every
    refusal is a ``BLOCKED_ROLE`` verdict whose ``detail`` names the rule that
    refused, so a client decides from the code and the dict and never by
    parsing prose.

    Args:
        actor: Who is asking, carrying the declared ``Role`` in ``actor.role``.
        command: The command they want to submit.
        station_info: The station's declaration snapshot, from which
            ``submit_vi_action``'s target capability is classified.
        attendance: Whether a human is watching (the value
            ``Orchestrator.set_attendance()`` published).
        kill_switch: The gate the human set (``Orchestrator.set_agent_gate()``).
        seq: Sequence number to stamp on a refusal verdict.
        run_owner: The **run owner** of the run in flight, as
            ``StatusSnapshot.run["owner"]`` publishes it (``{"kind", "id"}``),
            or ``None`` when nothing is running or the caller mirrors no
            status — in which case ownership refuses nothing here and the
            engine, which always knows, decides alone.

    Returns:
        ``None`` when the command is permitted and may be forwarded to the
        engine, or the ``BLOCKED_ROLE`` ``Verdict`` that refuses it.
    """
    if command.name is CommandName.EMERGENCY_STANDBY:
        return None
    if actor.kind is not ActorKind.AGENT:
        return None

    try:
        role = Role(actor.role)
    except ValueError:
        return _refusal(
            command,
            actor,
            f"Agent {actor.id!r} declares the unknown role {actor.role!r}; "
            f"the roles that exist are {[member.value for member in Role]}.",
            {"rule": "unknown_role", "role": actor.role},
            seq,
        )

    try:
        classified = classify_command(command, station_info)
    except UnclassifiedActionError as error:
        return _refusal(
            command,
            actor,
            f"Refused: {error}. An action with no declared class is refused "
            f"rather than guessed at.",
            {"rule": "unclassified_action", "role": role.value},
            seq,
        )

    action_class = classified.action_class
    detail: dict[str, object] = {
        "role": role.value,
        "action_class": action_class.value,
        "rationale": classified.rationale,
    }

    if kill_switch is AgentGate.REVOKED or (
        kill_switch is AgentGate.READ_ONLY and action_class is not ActionClass.READ
    ):
        return _refusal(
            command,
            actor,
            f"Agent access is {kill_switch.value}: {command.name.value!r} is "
            f"a {action_class.value} action and is refused. Only emergency "
            f"standby passes while the kill switch is closed.",
            {**detail, "rule": "kill_switch", "gate": kill_switch.value},
            seq,
        )

    permission = PERMISSION_MATRIX[action_class][role]
    if permission is Permission.UNATTENDED_ONLY and attendance:
        return _refusal(
            command,
            actor,
            f"The {role.value!r} role may take {action_class.value} actions "
            f"only while the experiment is unattended; a human is watching, "
            f"so {command.name.value!r} is refused — report it instead.",
            {**detail, "rule": "attendance", "attended": True},
            seq,
        )
    if permission is Permission.REFUSED:
        return _refusal(
            command,
            actor,
            f"The {role.value!r} role does not grant {action_class.value} "
            f"actions, so {command.name.value!r} is refused.",
            {**detail, "rule": "role_matrix"},
            seq,
        )
    # The role grants it; the last question is whose run it would be taken on.
    return _ownership_refusal(command, actor, run_owner, detail, seq)


def _ownership_refusal(
    command: Command,
    actor: Actor,
    run_owner: Mapping[str, Any] | None,
    detail: dict[str, object],
    seq: int,
) -> Verdict | None:
    """Apply the **run-ownership standard** to one already-permitted command.

    The local mirror of ``Orchestrator._run_owner_admission()``, in the same
    words and with the same ``detail.rule`` values, so an agent gets one
    answer whichever door it came through. It refuses only what the engine
    would refuse; it never ADMITS anything the engine would not, because the
    engine checks again with the ownership it actually holds.

    Args:
        command: The command being judged, whose ``args`` carry
            ``override_owner`` and ``reason``.
        actor: The agent asking (the human never reaches here).
        run_owner: The run in flight's owner, or ``None``.
        detail: The refusal detail built so far — role, action class and
            rationale.
        seq: Sequence number to stamp on a refusal verdict.

    Returns:
        ``None`` when the command may be forwarded — the actor owns the run,
        no run is owned, the command is not owner-scoped, or the takeover is
        properly declared — else the refusing ``Verdict``.
    """
    if command.name not in OWNER_SCOPED_COMMANDS or not run_owner:
        return None
    owner = {"kind": str(run_owner.get("kind", "")), "id": str(run_owner.get("id", ""))}
    if owner["kind"] == actor.kind.value and owner["id"] == actor.id:
        return None
    owned: dict[str, object] = {**detail, "owner": owner, "actor_id": actor.id}
    if not bool(command.args.get("override_owner", False)):
        return _refusal(
            command,
            actor,
            f"This run is owned by {owner['kind']} {owner['id']!r}, and agent "
            f"{actor.id!r} is not its owner, so {command.name.value!r} is "
            f"refused. Ask the owner or the operator to do it; if you must do "
            f"it yourself, re-send the same command with override_owner=true "
            f"and a reason saying why — the takeover is recorded on the run "
            f"and in the agent feed.",
            {**owned, "rule": "run_owner"},
            seq,
        )
    if not str(command.args.get("reason", "") or "").strip():
        return _refusal(
            command,
            actor,
            f"Taking over {owner['kind']} {owner['id']!r}'s run needs a "
            f"reason: re-send {command.name.value!r} with override_owner=true "
            f"and a non-empty reason saying why you are taking the run over.",
            {**owned, "rule": "override_reason_required"},
            seq,
        )
    return None


#: The roles in ascending order of authority. The matrix above is monotone
#: along it — a role never has less than the one below it in any row — which
#: is what makes "no more than this role" (``authorize_spooled()``'s cap) a
#: meaningful bound rather than an arbitrary comparison.
ROLE_LADDER: tuple[Role, ...] = (Role.OBSERVER, Role.DEBUG, Role.SESSION)


def _outranks(role: Role, cap: Role) -> bool:
    """Return whether *role* claims more authority than *cap* allows.

    Args:
        role: The role being claimed.
        cap: The most authority that may be granted.

    Returns:
        ``True`` when *role* sits above *cap* on ``ROLE_LADDER``.
    """
    return ROLE_LADDER.index(role) > ROLE_LADDER.index(cap)


def authorize_spooled(
    *,
    command: Command,
    declared_role: str,
    max_role: str,
    station_info: StationInfo,
    attendance: bool,
    kill_switch: AgentGate,
    seq: int = 0,
) -> Verdict | None:
    """Decide whether one **Request spool** request may be submitted.

    The permission hook ``core.request_spool.RequestSpool`` is wired with
    (see its ``Authorizer``): the same model ``authorize()`` applies, with one
    extra bound in front of it. A file dropped into a directory is a weaker
    claim of identity than an in-process connection, so the setup declares how
    much authority that door may ever grant (``monitor.yaml``'s
    ``spool_max_role``) and a request declaring more is refused here — before
    the matrix, and regardless of what the agent's own role would permit.

    The cap is checked AFTER the emergency-standby carve-out, never before:
    an actor that can see a problem must never be unable to make the station
    safe, and that is as true through a file drop as through a window.

    A ``max_role`` that names no known role is treated as the safest one
    (``observer``) and logged, because a typo in a config must narrow
    authority, never widen it.

    Args:
        command: The command the request carries, actor already stamped with
            the file's declared role.
        declared_role: The authority the request file declared.
        max_role: The cap the setup configured for this spool.
        station_info: The station's declaration snapshot.
        attendance: Whether a human is watching.
        kill_switch: The gate the human set.
        seq: Sequence number to stamp on a refusal verdict.

    Returns:
        ``None`` when the command may be submitted, or the ``BLOCKED_ROLE``
        ``Verdict`` that refuses it.
    """
    if command.name is CommandName.EMERGENCY_STANDBY:
        return None

    try:
        cap = Role(max_role)
    except ValueError:
        logger.warning(
            "Unknown request-spool cap %r; capping at %r instead.",
            max_role,
            Role.OBSERVER.value,
        )
        cap = Role.OBSERVER

    try:
        role = Role(declared_role)
    except ValueError:
        # An unknown role is refused by authorize() itself, with the message
        # that names the roles that exist; nothing is gained by saying it
        # twice in two different words.
        return authorize(
            command.actor, command, station_info, attendance, kill_switch, seq=seq
        )

    if _outranks(role, cap):
        return _refusal(
            command,
            command.actor,
            f"The request spool grants at most the {cap.value!r} role on this "
            f"setup, and this request declares {role.value!r}. Raise "
            f"monitor.yaml's spool_max_role to widen it, or submit through a "
            f"client that is inside the application.",
            {
                "rule": "spool_role_cap",
                "role": role.value,
                "max_role": cap.value,
            },
            seq,
        )

    return authorize(
        command.actor, command, station_info, attendance, kill_switch, seq=seq
    )


#: The order the three cells stand in, from "grants nothing" to "grants
#: everything". Only ever used to compare two ROLES cell by cell — see
#: ``role_within_ceiling()``; the matrix itself is read by identity, never by
#: rank, so a cell added later has to declare its place here deliberately.
_PERMISSION_RANK: dict[Permission, int] = {
    Permission.REFUSED: 0,
    Permission.UNATTENDED_ONLY: 1,
    Permission.PERMITTED: 2,
}


def role_within_ceiling(role: Role, ceiling: Role) -> bool:
    """Answer whether *role* grants no more authority than *ceiling* does.

    **The ceiling standard.** A deployment that opens a connection point to
    agents (the **Gateway server**) sets one role as the most authority it
    will hand out, and a connection asking for more is refused at the
    handshake rather than at its first write. What "more" means is read off
    ``PERMISSION_MATRIX`` cell by cell — a role is within the ceiling when
    for EVERY action class its cell grants no more than the ceiling's does —
    so the ceiling follows the table automatically and no second ordering of
    the roles has to be maintained beside it.

    Args:
        role: The role a connection declares for itself.
        ceiling: The most authority the deployment permits.

    Returns:
        ``True`` when *role* may be granted under *ceiling*.
    """
    return all(
        _PERMISSION_RANK[row[role]] <= _PERMISSION_RANK[row[ceiling]]
        for row in PERMISSION_MATRIX.values()
    )
