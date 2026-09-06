"""AgentPanel — the **Agent panel**: what the machines did, in this experiment.

A FILTER of the engine's one event stream, not a second stream: every
``Verdict`` and every ``StateChange`` whose ``Actor`` is not the physicist
becomes one row here, with the time, who acted, what they asked for and what
the engine answered. Nothing is computed that the contract did not already
carry — an agent that is refused shows the refusing rule verbatim, because a
panel that paraphrased a refusal would be a second, quieter authority on what
happened.

Three sources feed the same row model, and they are the same three places the
answer to "what did the machines do" lives:

* the live stream, forwarded in by the window that owns the connection (the
  destruction-order rule: a panel never connects to the engine itself);
* the **Agent feed** of the open experiment, read once when the panel opens,
  so the trail survives a restart of the application rather than starting
  blank at every launch;
* the **Draft entry** waiting on a run, which is the one row that is a
  QUESTION rather than a record: it carries an Approve button, and approving
  it is the human's half of the ELN approval gate.

What the panel never does: touch the engine, judge an action, or hold a
session object it did not receive. Approval goes through
``ExperimentManager.approve_eln_draft()``, which is the single writer for
that record exactly as the Orchestrator is for hardware.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from i2as.core.events import ActorKind, StateChange, Verdict, VerdictCode
from i2as.session.agent_feed import (
    RECORD_EVENT,
    RECORD_TOOL,
    RECORD_VERDICT,
    read_feed,
)
from i2as.session.manager import ExperimentManager
from i2as.gui.widget_lifecycle import retire_widget

logger = logging.getLogger(__name__)

#: How many rows the panel keeps. A cap rather than a growing list: this is a
#: live view of a trail whose permanent home is the **Agent feed** on disk, so
#: dropping the oldest row loses nothing that matters.
MAX_ROWS = 200

#: How long an actor counts as "active" after its last action, in seconds.
#: The takeover strip's "agents active" indicator answers "is anything acting
#: on my cryostat right now", and an agent that has said nothing for this long
#: is not.
ACTIVE_WINDOW_S = 300.0

#: The row's dynamic-property values, targeted by ``theme.py``'s
#: ``QLabel[class="agent_row"][outcome=…]`` rules.
OUTCOME_OK = "ok"
OUTCOME_REFUSED = "refused"
OUTCOME_EVENT = "event"
OUTCOME_PENDING = "pending"
#: An accepted action taken on ANOTHER actor's run — the **Takeover** the
#: run-ownership standard records. Its own outcome because it is neither an
#: ordinary success nor a refusal: the engine allowed it, and the physicist
#: still has to be able to find it at a glance.
OUTCOME_TAKEOVER = "takeover"

#: What the panel says before any non-operator actor has acted.
EMPTY_TEXT = "No agent has acted in this experiment."

#: The run-owner line's two shapes (GLOSSARY.md's **Run owner**): who owns
#: the run in flight, so a reader of this panel can tell at a glance whether
#: an agent acting here is acting on its own run or somebody else's.
RUN_OWNER_TEXT = "run owned by {owner}"
NO_RUN_TEXT = "no run in flight"


@dataclass(frozen=True)
class AgentAction:
    """One row of the **Agent panel** — the panel's whole row model.

    Deliberately flat and already rendered: every field is a string or a
    number taken straight off a ``Verdict``, a ``StateChange`` or an **Agent
    feed** record, so a row cannot disagree with the message it came from.

    Attributes:
        ts: Unix time the action happened.
        actor_id: The acting ``Actor``'s id.
        actor_role: The authority it acted under, ``""`` when it declared none.
        actor_kind: ``"agent"`` or ``"system"`` — the filter this panel is.
        what: The command name, the tool name, or (for a state change) the
            transition, e.g. ``"IDLE → RAMPING"``.
        code: The ``VerdictCode`` value, ``""`` for a state change.
        reason: The engine's own words for the outcome, ``""`` when it had
            none. The transition's cause for a state change.
        run_id: The run a pending **Draft entry** belongs to; ``""`` on every
            other row.
        kind: ``"verdict"``, ``"state"`` or ``"draft"``.
        takeover_owner: The **Run owner** this action was taken over, id
            only, read off the verdict's ``detail.takeover``; ``""`` on every
            ordinary row. When it is set, ``reason`` is the reason the actor
            gave for taking the run over.
    """

    ts: float
    actor_id: str
    actor_role: str = ""
    actor_kind: str = ActorKind.AGENT.value
    what: str = ""
    code: str = ""
    reason: str = ""
    run_id: str = ""
    kind: str = "verdict"
    takeover_owner: str = ""

    @property
    def refused(self) -> bool:
        """Whether the engine said no to this action."""
        return bool(self.code) and self.code != VerdictCode.OK.value

    @property
    def outcome(self) -> str:
        """The row's dynamic-property value (see the ``OUTCOME_*`` constants)."""
        if self.kind == "draft":
            return OUTCOME_PENDING
        if self.refused:
            return OUTCOME_REFUSED
        if self.takeover_owner:
            return OUTCOME_TAKEOVER
        if self.kind == "state":
            return OUTCOME_EVENT
        return OUTCOME_OK

    def text(self) -> str:
        """Return the one line this action is rendered as.

        Returns:
            ``"HH:MM:SS  <actor> (<role>)  <what> → <code> — <reason>"``, with
            the parts that do not apply left out rather than shown empty. A
            **Takeover** ends instead with ``"took over <owner>'s run:
            <reason>"``, which is the whole of what happened in the words the
            contract carried.
        """
        stamp = datetime.fromtimestamp(self.ts).strftime("%H:%M:%S")
        who = f"{self.actor_id} ({self.actor_role})" if self.actor_role else self.actor_id
        line = f"{stamp}  {who}  {self.what}"
        if self.code:
            line = f"{line} → {self.code}"
        if self.takeover_owner:
            took = f"took over {self.takeover_owner}'s run"
            return f"{line} — {took}: {self.reason}" if self.reason else f"{line} — {took}"
        if self.reason:
            line = f"{line} — {self.reason}"
        return line


def _actor_fields(actor: Any) -> tuple[str, str, str]:
    """Return ``(kind, id, role)`` from an ``Actor`` or its JSON dict.

    Args:
        actor: An ``events.Actor``, the mapping ``Actor.to_json()`` produced,
            or anything else (which reads as an unknown system actor).

    Returns:
        The three strings, empty where the value was not readable.
    """
    if isinstance(actor, Mapping):
        kind = str(actor.get("kind") or "")
        return kind, str(actor.get("id") or ""), str(actor.get("role") or "")
    kind = getattr(actor, "kind", None)
    kind_value = kind.value if isinstance(kind, ActorKind) else str(kind or "")
    return kind_value, str(getattr(actor, "id", "") or ""), str(
        getattr(actor, "role", "") or ""
    )


def _takeover_fields(detail: Any) -> tuple[str, str]:
    """Return ``(owner id, reason)`` of a **Takeover**, or two empty strings.

    Args:
        detail: A ``Verdict.detail`` or an **Agent feed** record's ``detail``.
            Anything that is not a takeover's mapping reads as no takeover —
            a panel never judges an action, it renders what the contract
            carried.

    Returns:
        The owner whose run was taken over and the reason given for it.
    """
    if not isinstance(detail, Mapping):
        return "", ""
    takeover = detail.get("takeover")
    if not isinstance(takeover, Mapping):
        return "", ""
    owner = takeover.get("owner")
    owner_id = str(owner.get("id") or "") if isinstance(owner, Mapping) else ""
    return owner_id, str(takeover.get("reason") or "")


def action_from_verdict(verdict: Verdict) -> AgentAction | None:
    """Build a row from one ``Verdict``, or ``None`` for the operator's own.

    Args:
        verdict: Any verdict off the engine's stream.

    Returns:
        The row, or ``None`` when the verdict answers an ``operator``
        command — the physicist's own actions are the rest of the session
        record, and mixing them in would make this panel useless for the
        question it exists to answer.
    """
    kind, actor_id, role = _actor_fields(verdict.actor)
    if kind == ActorKind.OPERATOR.value:
        return None
    owner, took_because = _takeover_fields(verdict.detail)
    return AgentAction(
        ts=verdict.ts,
        actor_id=actor_id,
        actor_role=role,
        actor_kind=kind,
        what=verdict.command.value,
        code=verdict.code.value,
        reason=took_because or verdict.reason,
        kind="verdict",
        takeover_owner=owner,
    )


def action_from_state_change(event: StateChange) -> AgentAction | None:
    """Build a row from one ``StateChange``, or ``None`` for an operator's.

    Args:
        event: Any state change off the engine's stream.

    Returns:
        The row, or ``None`` when the transition was the operator's doing.
    """
    kind, actor_id, role = _actor_fields(event.actor)
    if kind == ActorKind.OPERATOR.value:
        return None
    transition = f"{event.previous} → {event.state}" if event.previous else event.state
    return AgentAction(
        ts=event.ts,
        actor_id=actor_id,
        actor_role=role,
        actor_kind=kind,
        what=transition,
        reason=event.cause,
        kind="state",
    )


def action_from_feed_record(record: Mapping[str, Any]) -> AgentAction | None:
    """Build a row from one **Agent feed** line, or ``None`` when it is not one.

    The seed path. A ``command`` record is deliberately skipped: its verdict
    record says the same thing plus the answer, and showing both would double
    every action in the history.

    Args:
        record: One record as ``read_feed()`` returns it.

    Returns:
        The row, or ``None`` for a record kind the panel does not show.
    """
    kind = str(record.get("record") or "")
    if kind not in (RECORD_VERDICT, RECORD_EVENT, RECORD_TOOL):
        return None
    actor_kind, actor_id, role = _actor_fields(record.get("actor"))
    ts = record.get("ts")
    verdict = record.get("verdict") or {}
    detail = record.get("detail") or {}
    if kind == RECORD_EVENT:
        previous = str(detail.get("previous") or "")
        state = str(detail.get("state") or "")
        what = f"{previous} → {state}" if previous else state
        reason = str(detail.get("cause") or "")
        code = ""
    else:
        what = str(record.get("command") or record.get("tool") or "")
        code = str(verdict.get("code") or "")
        reason = str(verdict.get("reason") or "")
    owner, took_because = ("", "") if kind == RECORD_EVENT else _takeover_fields(detail)
    return AgentAction(
        ts=float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) else 0.0,
        actor_id=actor_id,
        actor_role=role,
        actor_kind=actor_kind or ActorKind.AGENT.value,
        what=what,
        code=code,
        reason=took_because or reason,
        kind="state" if kind == RECORD_EVENT else "verdict",
        takeover_owner=owner,
    )


class AgentPanel(QWidget):
    """The bottom-right quadrant's Agents sub-panel.

    ObjectNames (API for tests and muscle memory): the panel is
    ``agent_panel``, its scroll area ``agent_panel_scroll``, the widget
    holding the rows ``agent_panel_rows``, the empty-state label
    ``agent_panel_empty_label``, the system filter
    ``agent_panel_system_checkbox``, the run-owner line
    ``agent_panel_run_owner_label``, and each pending draft's button
    ``agent_approve_<run_id>``. Rows themselves are not named: they are
    rebuilt whenever the filter changes, so an index-based name would point
    at a different action after every toggle. ``row_texts()`` is what a
    reader of the rows asks instead.

    Args:
        session_manager: The L6 ``ExperimentManager``, used for three things
            and nothing else — the **Agent feed** to seed from, the pending
            **Draft entry** rows, and ``approve_eln_draft()``. ``None`` (a
            unit test, or a launch with no session layer) leaves the panel a
            pure view of the live stream.
        parent: Optional Qt parent widget.
    """

    #: Emitted with the number of distinct agents seen inside
    #: ``ACTIVE_WINDOW_S``, whenever a new action lands. The takeover strip's
    #: "agents active" indicator renders it; the ledger lives here because
    #: this is where every agent action already arrives.
    agents_active_changed = pyqtSignal(int)

    def __init__(
        self,
        session_manager: ExperimentManager | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("agent_panel")
        self._session_manager = session_manager
        self._actions: list[AgentAction] = []
        self._row_widgets: list[QWidget] = []
        #: ``{actor_id: last action's unix time}`` — the "agents active" ledger.
        self._last_seen: dict[str, float] = {}
        #: Run ids that already have a pending-draft row, so a re-emitted run
        #: record does not append the same question twice.
        self._draft_rows: dict[str, QWidget] = {}
        self._follow_tail = True

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        root.addLayout(self._build_filter_row())

        self._rows_host = QWidget()
        self._rows_host.setObjectName("agent_panel_rows")
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(2, 2, 2, 2)
        self._rows_layout.setSpacing(2)
        self._rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._empty_label = QLabel(EMPTY_TEXT)
        self._empty_label.setObjectName("agent_panel_empty_label")
        self._empty_label.setProperty("class", "secondary_label")
        self._empty_label.setWordWrap(True)
        self._rows_layout.addWidget(self._empty_label)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("agent_panel_scroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._rows_host)
        bar = self._scroll.verticalScrollBar()
        bar.valueChanged.connect(self._on_scrolled)
        # A row added this instant has not been laid out yet, so the
        # scrollbar's maximum is still the one from before it existed.
        # Following the tail therefore hangs off the RANGE changing, which is
        # Qt telling us the new row has been measured.
        bar.rangeChanged.connect(self._on_range_changed)
        root.addWidget(self._scroll, stretch=1)

        if self._session_manager is not None:
            self._session_manager.experiment_changed.connect(
                self._on_experiment_changed
            )
            self._session_manager.run_recorded.connect(self._on_run_recorded)
            self.reload_experiment()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_filter_row(self) -> QHBoxLayout:
        """Build the one filter the panel offers: whether to show system rows.

        Returns:
            The row's layout.
        """
        row = QHBoxLayout()
        row.setContentsMargins(2, 0, 2, 0)
        self._system_checkbox = QCheckBox("Include system")
        self._system_checkbox.setObjectName("agent_panel_system_checkbox")
        self._system_checkbox.setChecked(False)
        self._system_checkbox.setToolTip(
            "Also show what the engine did on its own — a fault, a safety "
            "hold — alongside what agents asked for."
        )
        self._system_checkbox.toggled.connect(lambda _on: self._rebuild())
        row.addWidget(self._system_checkbox)
        row.addStretch()
        self._run_owner_label = QLabel(NO_RUN_TEXT)
        self._run_owner_label.setObjectName("agent_panel_run_owner_label")
        self._run_owner_label.setProperty("class", "secondary_label")
        self._run_owner_label.setToolTip(
            "Who started the run in flight. Only that actor — or you — may "
            "abort it or attest to its steps; anyone else has to take it "
            "over deliberately, and the takeover is recorded."
        )
        row.addWidget(self._run_owner_label)
        return row

    # ------------------------------------------------------------------
    # The run in flight (forwarded in by the window, off the status mirror)
    # ------------------------------------------------------------------

    def set_run_owner(self, owner: Mapping[str, Any] | None) -> None:
        """Show who owns the run in flight (GLOSSARY.md's **Run owner**).

        Forwarded by the window on every status snapshot rather than read
        here: the destruction-order rule keeps a panel off the engine's own
        signals, and the window already holds the **Status mirror** this
        comes from.

        Args:
            owner: The owner's ``{"kind", "id"}`` as the snapshot's run
                summary carries it, or ``None`` when no run is in flight.
        """
        actor_id = str(owner.get("id") or "") if isinstance(owner, Mapping) else ""
        self._run_owner_label.setText(
            RUN_OWNER_TEXT.format(owner=actor_id) if actor_id else NO_RUN_TEXT
        )

    # ------------------------------------------------------------------
    # Live stream (forwarded in by the window that owns the connection)
    # ------------------------------------------------------------------

    def on_verdict(self, verdict: object) -> None:
        """Absorb one ``Verdict`` from the engine's stream.

        Args:
            verdict: Anything off the verdict stream; anything that is not a
                ``Verdict``, and every operator verdict, is ignored.
        """
        if not isinstance(verdict, Verdict):
            return
        action = action_from_verdict(verdict)
        if action is not None:
            self.add_action(action)

    def on_event(self, event: object) -> None:
        """Absorb one event from the engine's stream.

        Args:
            event: Anything off the event stream; only a ``StateChange`` with
                a non-operator actor becomes a row (the rest of the stream is
                either a read, which the **Status mirror** answers, or an
                action already reported by its verdict).
        """
        if not isinstance(event, StateChange):
            return
        action = action_from_state_change(event)
        if action is not None:
            self.add_action(action)

    def add_action(self, action: AgentAction) -> None:
        """Append one action to the panel, capping and auto-scrolling.

        Args:
            action: The row to add. Newest is always at the bottom, which is
                where a reader watching a live trail looks.
        """
        self._actions.append(action)
        if len(self._actions) > MAX_ROWS:
            del self._actions[: len(self._actions) - MAX_ROWS]
            self._rebuild()
        elif self._is_visible(action):
            self._add_row_widget(self._build_row(action))
        self._note_activity(action)

    # ------------------------------------------------------------------
    # The experiment: the feed seed and the pending drafts
    # ------------------------------------------------------------------

    def reload_experiment(self) -> None:
        """Re-seed the panel from the open experiment: its feed, its drafts.

        Called when the panel is built and on every experiment change, so
        opening (or switching to) an experiment shows the trail that
        experiment already has rather than an empty panel that only fills up
        again once an agent acts.
        """
        self._actions.clear()
        self._rebuild()
        if self._session_manager is None:
            return
        record = self._session_manager.current_experiment()
        if record is None:
            return
        try:
            path = self._session_manager.store.agent_feed_path(record.experiment_id)
        except Exception:  # noqa: BLE001 — a view never raises into Qt
            logger.exception("Agent panel: could not resolve the agent feed path")
            return
        self.seed_from_feed(path)
        self._sync_pending_drafts()

    def seed_from_feed(self, path: Any) -> int:
        """Seed the panel from an **Agent feed** file, oldest first.

        Args:
            path: The feed file (``ExperimentStore.agent_feed_path()``). A
                missing or unreadable file seeds nothing — ``read_feed()``
                already degrades to an empty list with a warning.

        Returns:
            How many rows were seeded.
        """
        records = read_feed(path)
        seeded = [
            action
            for action in (action_from_feed_record(record) for record in records)
            if action is not None
        ]
        if not seeded:
            return 0
        self._actions.extend(seeded[-MAX_ROWS:])
        del self._actions[:-MAX_ROWS]
        for action in seeded:
            self._note_activity(action, announce=False)
        self._rebuild()
        self._announce_activity()
        return len(seeded)

    def _on_experiment_changed(self, _record: dict) -> None:
        """Re-seed when the open experiment changes (opened, switched, closed).

        Args:
            _record: The experiment as a dict, or ``{}``; the panel re-reads
                the manager rather than the payload, so one slot serves every
                path.
        """
        for widget in list(self._draft_rows.values()):
            self._remove_row_widget(widget)
        self._draft_rows.clear()
        self.reload_experiment()

    def _on_run_recorded(self, _record: dict) -> None:
        """Re-check the pending drafts whenever a run record changes.

        Args:
            _record: The ``RunRecord`` as a dict; the panel re-reads the
                manager, which is the single writer for that record.
        """
        self._sync_pending_drafts()

    def _sync_pending_drafts(self) -> None:
        """Add a row per **Draft entry** waiting, and drop the approved ones."""
        if self._session_manager is None:
            return
        record = self._session_manager.current_experiment()
        pending = {
            run.run_id
            for run in (record.runs if record is not None else ())
            if run.pending_eln_draft
        }
        for run_id in sorted(pending - set(self._draft_rows)):
            self._draft_rows[run_id] = self._add_draft_row(run_id)
        for run_id in set(self._draft_rows) - pending:
            self._remove_row_widget(self._draft_rows.pop(run_id))
        self._refresh_empty_state()

    def _add_draft_row(self, run_id: str) -> QWidget:
        """Build and append the one row that asks a question of the human.

        Args:
            run_id: The run whose draft is waiting.

        Returns:
            The row widget, so the panel can drop it once approved.
        """
        action = AgentAction(
            ts=time.time(),
            actor_id="eln",
            actor_kind=ActorKind.SYSTEM.value,
            what=f"ELN draft for run {run_id}",
            reason="waiting for your approval",
            run_id=run_id,
            kind="draft",
        )
        row = self._build_row(action)
        button = QPushButton("Approve")
        button.setObjectName(f"agent_approve_{run_id}")
        button.setToolTip(
            "Queue this drafted notebook entry for publishing. Nothing is "
            "published until you do."
        )
        button.clicked.connect(lambda _checked=False, rid=run_id: self._approve(rid))
        row.layout().addWidget(button)
        self._add_row_widget(row)
        return row

    def _approve(self, run_id: str) -> None:
        """Approve one pending **Draft entry** and re-sync the rows.

        Args:
            run_id: The run whose draft the human approved.
        """
        if self._session_manager is None:
            return
        job_id = self._session_manager.approve_eln_draft(run_id)
        if not job_id:
            logger.warning("Agent panel: nothing was queued for run %s", run_id)
        self._sync_pending_drafts()

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------

    def _is_visible(self, action: AgentAction) -> bool:
        """Return whether the current filter shows this action.

        Args:
            action: The action to judge.

        Returns:
            ``True`` for every agent action, and for a system action only
            while the system filter is ticked.
        """
        if action.actor_kind == ActorKind.SYSTEM.value:
            return self._system_checkbox.isChecked()
        return True

    def _build_row(self, action: AgentAction) -> QWidget:
        """Build one row widget for an action.

        Args:
            action: The action to render.

        Returns:
            A row widget holding the action's line, styled by its outcome
            through a dynamic property (no widget stylesheet, theme tokens
            only).
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QLabel(action.text())
        # Plain text, never rich: a reason is written by the engine and a
        # cause by whatever caused it, and neither may inject markup here.
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setProperty("class", "agent_row")
        label.setProperty("outcome", action.outcome)
        # One row per action, one line per row: a wrapped label has a
        # one-line minimum height, so a short quadrant would compress the
        # rows into each other instead of scrolling. The full line is the
        # tooltip, and the scroll area scrolls sideways for the rest.
        label.setWordWrap(False)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        label.setToolTip(action.text())
        layout.addWidget(label, stretch=1)
        return row

    def _add_row_widget(self, row: QWidget) -> None:
        """Insert a built row at the bottom, cap the list, and follow the tail.

        Args:
            row: The row widget to add.
        """
        self._rows_layout.addWidget(row)
        self._row_widgets.append(row)
        while len(self._row_widgets) > MAX_ROWS:
            self._remove_row_widget(self._row_widgets[0])
        self._refresh_empty_state()
        if self._follow_tail:
            self._scroll_to_bottom()

    def _remove_row_widget(self, row: QWidget) -> None:
        """Take one row out of the layout, in the order Qt needs.

        Args:
            row: The row widget to retire (the card-retirement standard,
                ``gui/widget_lifecycle.py``).
        """
        if row in self._row_widgets:
            self._row_widgets.remove(row)
        for run_id, draft_row in list(self._draft_rows.items()):
            if draft_row is row:
                del self._draft_rows[run_id]
        retire_widget(row, self._rows_layout)

    def _rebuild(self) -> None:
        """Rebuild every row from the action list under the current filter.

        A retired widget is never reused (the card-retirement standard), so
        the pending-draft rows are rebuilt too — their run ids are what
        survives the rebuild, not their widgets.
        """
        pending = list(self._draft_rows)
        for row in list(self._row_widgets):
            self._remove_row_widget(row)
        self._draft_rows.clear()
        for action in self._actions:
            if self._is_visible(action):
                self._add_row_widget(self._build_row(action))
        for run_id in pending:
            self._draft_rows[run_id] = self._add_draft_row(run_id)
        self._refresh_empty_state()

    def _refresh_empty_state(self) -> None:
        """Show the empty-state line exactly while there is no row."""
        self._empty_label.setVisible(not self._row_widgets)

    def _on_scrolled(self, value: int) -> None:
        """Follow the tail only while the reader is already at the bottom.

        Args:
            value: The vertical scrollbar's new value.
        """
        bar = self._scroll.verticalScrollBar()
        self._follow_tail = value >= bar.maximum()

    def _on_range_changed(self, _minimum: int, maximum: int) -> None:
        """Keep the newest row in view as the list grows.

        Args:
            _minimum: The scrollbar's new minimum (always 0 here).
            maximum: Its new maximum, which is where the tail now is.
        """
        if self._follow_tail:
            self._scroll.verticalScrollBar().setValue(maximum)

    def _scroll_to_bottom(self) -> None:
        """Put the newest row in view.

        Guarded against a panel destroyed between the deferred call being
        scheduled and delivered: a view that is gone has nothing to scroll,
        and must not raise into Qt's event loop for it.
        """
        try:
            bar = self._scroll.verticalScrollBar()
            bar.setValue(bar.maximum())
        except RuntimeError:
            logger.debug("Agent panel: scroll skipped, the view is gone")

    # ------------------------------------------------------------------
    # The "agents active" ledger
    # ------------------------------------------------------------------

    def _note_activity(self, action: AgentAction, announce: bool = True) -> None:
        """Record that an agent acted, for the takeover strip's indicator.

        Args:
            action: The action just seen.
            announce: Whether to emit ``agents_active_changed`` afterwards.
                ``False`` while seeding, which announces once at the end.
        """
        if action.actor_kind != ActorKind.AGENT.value or not action.actor_id:
            return
        previous = self._last_seen.get(action.actor_id, 0.0)
        self._last_seen[action.actor_id] = max(previous, action.ts)
        if announce:
            self._announce_activity()

    def _announce_activity(self) -> None:
        """Emit the current active-agent count."""
        self.agents_active_changed.emit(self.active_agent_count())

    def active_agent_count(self, now: float | None = None) -> int:
        """Return how many distinct agents acted inside ``ACTIVE_WINDOW_S``.

        Args:
            now: The moment to count against, or ``None`` for this one.
                Given for tests and for a caller that already has the time.

        Returns:
            The number of distinct agent actor ids seen recently enough to
            still count as acting on this cryostat.
        """
        moment = time.time() if now is None else now
        return sum(
            1 for last in self._last_seen.values() if moment - last <= ACTIVE_WINDOW_S
        )

    # ------------------------------------------------------------------
    # Reads (tests and the window)
    # ------------------------------------------------------------------

    def actions(self) -> tuple[AgentAction, ...]:
        """Return every action the panel holds, oldest first."""
        return tuple(self._actions)

    def row_texts(self) -> tuple[str, ...]:
        """Return the visible rows' lines, top to bottom."""
        return tuple(
            row.findChild(QLabel).text()
            for row in self._row_widgets
            if row.findChild(QLabel) is not None
        )
