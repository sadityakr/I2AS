"""What each action a client can take actually DOES, as one classification.

The permission matrix in ``roles.py`` decides who may take an action of a
given class. This module decides which class an action IS.

**Where each answer comes from.** How dangerous one instrument's action is
is a judgement about that instrument, so a ``@control``'s class is not
decided here at all: it is DECLARED on the VI
(``@control(action_class=...)``, the action-class declaration — see
``core/decorators.py``) and travels to this module on
``StationInfo``'s ``ControlInfo.action_class``. What stays here is what
belongs to the engine rather than to any instrument rack:

* ``COMMAND_ACTION_CLASSES`` — one row per ``CommandName``, the engine's own
  command surface. ``SUBMIT_VI_ACTION`` is the one command whose class is
  not fixed here, because it depends on the capability it targets:
  ``classify_control()`` reads the declaration.
* ``LIFECYCLE_ACTION_CLASSES`` — ``initiate`` / ``standby``, the two
  non-``@control`` methods the direct action path admits. They carry no
  decorator to declare anything on, and they mean the same thing on every
  VI, so they are classified once here.

Nothing in this module is load-bearing for safety on its own — the
Orchestrator's own admission rules, the control-validation standard's limits
and the session envelope all still bind every writer. What the
classification decides is how much autonomy an agent is granted BEFORE those
checks run.

**No silent default.** A command with no row, or a capability no instrument
declares, is not guessed at: it is refused, by name, with a reason saying
the classification is missing. Conformance asserts every shipped
``@control`` declares its class explicitly, so a refusal is a bug report,
never the normal path.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from i2as.core.events import Command, CommandName, ControlInfo, StationInfo

logger = logging.getLogger(__name__)


class ActionClass(str, Enum):
    """How much authority an action needs, independent of who is asking.

    The four classes of the gateway's permission matrix (``roles.py``). A
    ``str`` enum so the value is JSON-safe as it stands and travels in a
    refusal's ``detail`` unchanged. The same four values are
    ``core.decorators.VALID_ACTION_CLASSES``, which is what a VI declares
    against; ``tests/test_conformance.py`` asserts the two agree, so the
    declaration side and the permission side can never drift apart.

    Members:
        READ: Observes the system and changes nothing.
        RECOVERY: Instrument housekeeping that keeps a run alive — the
            framework's "pause/resume, VI initiate/standby, re-send config,
            adjust waits".
        RUN_CONTROL: Starts, stops, or redirects the measurement itself, or
            commands energy into the cryostat.
        ENVELOPE: Changes the rules the other three are judged by — the
            session envelope, attendance, and the kill switch. Reserved to
            the human.
    """

    READ = "read"
    RECOVERY = "recovery"
    RUN_CONTROL = "run_control"
    ENVELOPE = "envelope"


@dataclass(frozen=True)
class ClassifiedAction:
    """One row of a classification table: the class, and why.

    The rationale is not decoration. It is what a physicist reviews when
    confirming this provisional table, and what a refusal quotes back to the
    agent that was refused.

    Attributes:
        action_class: Which class this action belongs to.
        rationale: One line saying why, in the reviewer's terms.
    """

    action_class: ActionClass
    rationale: str


class UnclassifiedActionError(ValueError):
    """An action reached the gateway with no row in any classification table.

    Raised by ``classify_command()`` rather than defaulted, so that a table
    somebody forgot to update refuses the agent by name instead of silently
    granting or withholding authority. ``roles.authorize()`` turns it into a
    ``BLOCKED_ROLE`` verdict carrying this message.
    """


# ── The engine's command surface ──────────────────────────────────────
#
# One row per CommandName; conformance diffs the two, so a command added to
# the contract cannot reach an agent unclassified. SUBMIT_VI_ACTION is the
# one command whose class is not fixed here — it depends on the capability it
# targets, so classify_control() reads that capability's own declaration.

COMMAND_ACTION_CLASSES: dict[CommandName, ClassifiedAction] = {
    # ── Runs and the queue ──
    CommandName.RUN_PROCEDURE: ClassifiedAction(
        ActionClass.RUN_CONTROL, "Starts a measurement on the mounted sample."
    ),
    CommandName.QUEUE_PROCEDURE: ClassifiedAction(
        ActionClass.RUN_CONTROL,
        "Commits the station to a measurement it will start unattended.",
    ),
    CommandName.RUN_QUEUE: ClassifiedAction(
        ActionClass.RUN_CONTROL, "Pulls the next queued run and starts it."
    ),
    CommandName.PAUSE_PROCEDURE: ClassifiedAction(
        ActionClass.RECOVERY,
        "Holds the hardware at the pause boundary without ending the run — "
        "the framework's canonical recovery action.",
    ),
    CommandName.RESUME_PROCEDURE: ClassifiedAction(
        ActionClass.RECOVERY, "The inverse of pause; the run it resumes is its own."
    ),
    CommandName.ABORT_PROCEDURE: ClassifiedAction(
        ActionClass.RUN_CONTROL,
        "Ends the run and discards what it would still have measured; "
        "recovery keeps a run alive rather than ending it.",
    ),
    # ── Instrument actions ──
    CommandName.SUBMIT_GLOBAL_ACTION: ClassifiedAction(
        ActionClass.RECOVERY,
        "Fans out to the lifecycle actions (initiate_all / standby_all), "
        "which are recovery. PROVISIONAL: standby_all aborts an active run "
        "first, so it is the widest recovery action there is.",
    ),
    CommandName.STOP_RAMP: ClassifiedAction(
        ActionClass.RECOVERY,
        "Stops motion and holds where it is — always the direction of safety.",
    ),
    CommandName.CONNECT_INSTRUMENT: ClassifiedAction(
        ActionClass.RECOVERY,
        "Re-establishes communication with a dropped instrument: the "
        "archetypal unattended recovery.",
    ),
    CommandName.DISCONNECT_INSTRUMENT: ClassifiedAction(
        ActionClass.RECOVERY, "Its inverse — releases the bus, commands nothing."
    ),
    CommandName.PING_INSTRUMENT: ClassifiedAction(
        ActionClass.RECOVERY,
        "PROVISIONAL: an identity query changes nothing, so rule 3 would "
        "make it a read — but it is bus traffic on a shared bus during a "
        "run, and it is the connection-lifecycle probe the two rows above "
        "belong to, so it is classified with them.",
    ),
    # ── Faults, safety and recovery ──
    CommandName.EMERGENCY_STANDBY: ClassifiedAction(
        ActionClass.RECOVERY,
        "Classified for the record only: authorize() exempts it before the "
        "matrix is consulted, so it is permitted to every role in every "
        "state and at every kill-switch setting.",
    ),
    CommandName.ACKNOWLEDGE: ClassifiedAction(
        ActionClass.RUN_CONTROL,
        "PROVISIONAL: it grants a time-boxed manual override OVER a safety "
        "hold, which is authority over the interlocks rather than gentle "
        "recovery. The physicist may want to split the EMERGENCY "
        "acknowledge from the hold unlock.",
    ),
    CommandName.ACKNOWLEDGE_FAULT: ClassifiedAction(
        ActionClass.RECOVERY,
        "Clears a latched instrument fault record; commands no hardware.",
    ),
    CommandName.RETRY_FAULT: ClassifiedAction(
        ActionClass.RECOVERY, "Retries the communication path to a faulted instrument."
    ),
    CommandName.RECOVER_FROM_ERROR: ClassifiedAction(
        ActionClass.RECOVERY, "Returns the state machine to IDLE after an error."
    ),
    # ── Monitoring and policy ──
    CommandName.START_MONITORING: ClassifiedAction(
        ActionClass.RECOVERY,
        "Starts the polling half of the tick. It reads instruments rather "
        "than driving them, but it changes the engine's operating mode, so "
        "it is not a read.",
    ),
    CommandName.STOP_MONITORING: ClassifiedAction(
        ActionClass.RECOVERY,
        "Its inverse — stops polling and blinds every trend check with it.",
    ),
    CommandName.SET_EXPERIMENT_ENVELOPE: ClassifiedAction(
        ActionClass.ENVELOPE,
        "Sets the sample's own safety bounds. The human mounts the sample "
        "and the human says what it may see.",
    ),
    CommandName.SET_ATTENDANCE: ClassifiedAction(
        ActionClass.ENVELOPE,
        "Declares whether a human is watching — an input to this very "
        "matrix, so no agent may set it about itself.",
    ),
    CommandName.SET_AGENT_GATE: ClassifiedAction(
        ActionClass.ENVELOPE,
        "The kill switch. An agent that could reopen its own gate would not "
        "be gated at all.",
    ),
}


# ── Lifecycle actions: the same two on every VI ───────────────────────

LIFECYCLE_ACTION_CLASSES: dict[str, ClassifiedAction] = {
    "initiate": ClassifiedAction(
        ActionClass.RECOVERY,
        "Re-sends an instrument's configured operating state — named "
        "verbatim as a recovery action by the permission model.",
    ),
    "standby": ClassifiedAction(
        ActionClass.RECOVERY,
        "Drives one instrument to its safe idle state; the direction of "
        "safety, and named verbatim as a recovery action.",
    ),
}


def _declared_control(
    station_info: StationInfo, vi_name: str, method_name: str
) -> ControlInfo:
    """Return one configured instrument's declaration of one ``@control``.

    Args:
        station_info: The station's declaration snapshot.
        vi_name: The instrument the action targets.
        method_name: The ``@control`` being called.

    Returns:
        The instrument's ``ControlInfo`` for that capability.

    Raises:
        UnclassifiedActionError: If no configured instrument has that name,
            or it declares no such capability.
    """
    for instrument in station_info.instruments:
        if instrument.name != vi_name:
            continue
        for declared in instrument.controls:
            if declared.name == method_name:
                return declared
        raise UnclassifiedActionError(
            f"{vi_name} declares no capability named {method_name!r}, so the "
            f"action cannot be classified"
        )
    raise UnclassifiedActionError(
        f"no instrument named {vi_name!r} is configured on this station, so "
        f"the action cannot be classified"
    )


def classify_control(station_info: StationInfo, vi_name: str, method_name: str) -> ClassifiedAction:
    """Classify one capability of one configured instrument.

    Reads the class the VI itself declared (the action-class declaration, see
    ``core/decorators.py``) off the station's snapshot. The two lifecycle
    actions carry no decorator, so they come from
    ``LIFECYCLE_ACTION_CLASSES`` instead.

    Args:
        station_info: The station's declaration snapshot, which is where the
            declared class comes from.
        vi_name: The instrument the action targets.
        method_name: The ``@control`` or lifecycle action being called.

    Returns:
        The action's class and the rationale a refusal quotes back.

    Raises:
        UnclassifiedActionError: If the instrument is not configured, if it
            declares no such capability, or if the declared class is not one
            this gateway knows — never a guessed default.
    """
    if method_name in LIFECYCLE_ACTION_CLASSES:
        return LIFECYCLE_ACTION_CLASSES[method_name]
    declared = _declared_control(station_info, vi_name, method_name)
    try:
        action_class = ActionClass(declared.action_class)
    except ValueError as exc:
        raise UnclassifiedActionError(
            f"{vi_name}.{method_name}() declares action class "
            f"{declared.action_class!r}, which this gateway does not know, "
            f"so no role can be granted it"
        ) from exc
    return ClassifiedAction(
        action_class,
        f"Declared {action_class.value} by {vi_name}'s own @control.",
    )


def classify_command(command: Command, station_info: StationInfo) -> ClassifiedAction:
    """Classify one submitted command into its action class.

    Every ``CommandName`` is classified by ``COMMAND_ACTION_CLASSES``, except
    ``SUBMIT_VI_ACTION``, whose class depends on the capability it targets
    and is resolved through ``classify_control()``.

    Args:
        command: The command a client wants to submit.
        station_info: The station's declaration snapshot.

    Returns:
        The action class this command belongs to, with the rationale behind
        the classification.

    Raises:
        UnclassifiedActionError: If the command, its target instrument, or
            its target capability has no row.
    """
    if command.name is CommandName.SUBMIT_VI_ACTION:
        args: Mapping[str, object] = command.args
        vi_name = str(args.get("vi_name", ""))
        method_name = str(args.get("method_name", ""))
        if not vi_name or not method_name:
            raise UnclassifiedActionError(
                "submit_vi_action needs both 'vi_name' and 'method_name' "
                "before its action class can be decided"
            )
        return classify_control(station_info, vi_name, method_name)
    classified = COMMAND_ACTION_CLASSES.get(command.name)
    if classified is None:
        raise UnclassifiedActionError(
            f"command {command.name.value!r} has no row in the gateway's "
            f"action-class table, so no role can be granted it"
        )
    return classified
