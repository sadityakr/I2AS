"""TakeoverStrip — the **Takeover strip**: the human's controls over the agents, and what they report.

**What you set and what the station reports are never drawn alike.** The
strip has two halves, and the Monitor window puts them in two places:

* **Controls, in the header**, where the physicist already looks and where
  taking the machine back must never be somewhere you have to go and find:

  - **The kill switch**, a segmented Active / Read-only / Revoked control,
    one click from any state. Applied through the client's
    ``set_agent_gate()``, so the engine — the single enforcement point —
    decides, and REFLECTED from the **Status mirror**, so an agent that gated
    itself, or a ``i2as.ctl`` invocation that did, shows here without this
    widget having been told.
  - **The attendance toggle**, one fact with two homes: the experiment record
    (it must survive a restart) and the engine (it must be readable where
    the gateway's permission matrix is evaluated). Both are written through
    the session layer's single writer when an experiment is open, and
    directly into the engine when none is.

* **Indicators, in the status bar** (``status_line``), where every other fact
  about the station already is — reads, never controls:

  - the gate as a coloured dot and words ("Agents read-only"), so the state
    is legible even with the header scrolled off or the switch unnoticed;
  - "N agents acting", the distinct agents that have acted recently,
    rendered from the **Agent panel**'s own ledger — the panel already sees
    every agent action, so counting them twice would be counting them
    differently;
  - attended / unattended;
  - "▶ <procedure>, run owned by N", the **Run owner** of the run in flight
    and what it is running, reflected from the same mirror, with every
    parameter it was started with in the tooltip (the **reflection
    standard**, ``StatusMirror.run_manifest()``) — the always-on answer to
    "what did the agent just set?".

Nothing here is ever disabled by the gate. A kill switch that could lock the
human out of their own instrument would be a hazard rather than a safeguard,
and this widget is the human's end of it.
"""

from __future__ import annotations

import logging

from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from i2as.core.events import AgentGate
from i2as.core.orchestrator_proxy import OrchestratorProxy
from i2as.core.status_mirror import StatusMirror
from i2as.gui.theme import STATUS_ERROR, STATUS_OK, STATUS_WARN
from i2as.session.manager import ExperimentManager

logger = logging.getLogger(__name__)

#: The tri-state, in the order it is offered: most authority first, so the
#: strip reads left-to-right as "how far can they go".
GATE_CHOICES: tuple[tuple[AgentGate, str, str], ...] = (
    (
        AgentGate.ACTIVE,
        "Active",
        "Agents act normally — each connection's own role decides what it "
        "may do.",
    ),
    (
        AgentGate.READ_ONLY,
        "Read-only",
        "Agents may look but not touch: anything that writes is refused, "
        "naming the gate.",
    ),
    (
        AgentGate.REVOKED,
        "Revoked",
        "Agents may take no action at all. Emergency standby still passes, "
        "and your own controls are never gated.",
    ),
)


#: What the status line says about the run in flight, and nothing at all when
#: there is none: an empty label rather than "no run", because the state label
#: beside it already says the station is idle.
RUN_OWNER_TEXT = "run owned by {owner}"
#: The same line once the mirror also holds the run's manifest: what the
#: owner is running. The parameters go in the tooltip (``RUN_PARAMS_TOOLTIP``),
#: one per line, because the status bar has room for a name and not for twenty
#: values — and the tooltip is where a reader looks for the rest of a line.
RUN_TEXT = "▶ {procedure}, run owned by {owner}"
RUN_PARAMS_TOOLTIP = "{procedure} — started by {owner}\n{params}"

#: The gate as the status line reports it: colour and words.
GATE_STATUS: dict[str, tuple[str, str]] = {
    AgentGate.ACTIVE.value: (STATUS_OK, "Agents active"),
    AgentGate.READ_ONLY.value: (STATUS_WARN, "Agents read-only"),
    AgentGate.REVOKED.value: (STATUS_ERROR, "Agents revoked"),
}

#: Why the run owner is worth a line in the header at all. Also the whole of
#: what the line says when the header is too narrow to show its text.
OWNER_TOOLTIP = (
    "Who started the run in flight. Only that actor — or you — may abort it "
    "or attest to its steps; another agent has to take it over deliberately, "
    "and the takeover is recorded."
)


class TakeoverStrip(QWidget):
    """The header's agent controls, and the status-bar line reporting on agents.

    The strip itself is the CONTROLS half, placed in the header; its
    ``status_line`` is the INDICATORS half, which the window places in its
    status bar. One object owns both so one ``sync_from_mirror()`` keeps them
    in step.

    ObjectNames (API for tests and muscle memory): the strip is
    ``takeover_strip``, its gate buttons ``agent_gate_active_btn`` /
    ``agent_gate_read_only_btn`` / ``agent_gate_revoked_btn``, the attendance
    toggle ``takeover_attended_btn``; the status line is
    ``agent_status_line``, with ``agent_gate_status_label``,
    ``agents_active_label``, ``attendance_status_label`` and
    ``run_owner_label``.

    Args:
        orchestrator: The client's ``OrchestratorProxy`` — the gate and
            attendance are pushed down through it (the only thing this widget
            asks of the engine).
        mirror: The shared **Status mirror**, which answers every read here.
            ``None`` builds the fallback the inline construction path uses
            (``StatusMirror.of()``), exactly as the window does.
        session_manager: The L6 ``ExperimentManager``, so attendance is also
            recorded on the open experiment. ``None`` (a unit test, or a
            launch with no session layer) leaves the toggle writing to the
            engine alone.
        parent: Optional Qt parent widget.
    """

    def __init__(
        self,
        orchestrator: OrchestratorProxy,
        mirror: StatusMirror | None = None,
        session_manager: ExperimentManager | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("takeover_strip")
        self._orchestrator = orchestrator
        self._mirror = mirror if mirror is not None else StatusMirror.of(orchestrator)
        self._session_manager = session_manager
        self._gate_buttons: dict[str, QPushButton] = {}

        # ── Controls: the header half ─────────────────────────────────────
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(QLabel("Agent access"))

        segments = QWidget()
        segments.setObjectName("agent_gate_segments")
        segment_row = QHBoxLayout(segments)
        segment_row.setContentsMargins(0, 0, 0, 0)
        segment_row.setSpacing(0)
        self._gate_group = QButtonGroup(self)
        self._gate_group.setExclusive(True)
        for index, (gate, text, tooltip) in enumerate(GATE_CHOICES):
            button = QPushButton(text)
            button.setObjectName(f"agent_gate_{gate.value}_btn")
            button.setCheckable(True)
            button.setToolTip(tooltip)
            button.setProperty("class", "segment")
            button.setProperty("gate", gate.value)
            button.setProperty(
                "position",
                "first" if index == 0 else "last" if index == len(GATE_CHOICES) - 1 else "middle",
            )
            button.toggled.connect(
                lambda checked, value=gate.value: self._on_gate_toggled(
                    checked, value
                )
            )
            self._gate_group.addButton(button)
            self._gate_buttons[gate.value] = button
            segment_row.addWidget(button)
        row.addWidget(segments)

        self._attended_button = QPushButton("Attended")
        self._attended_button.setObjectName("takeover_attended_btn")
        self._attended_button.setCheckable(True)
        self._attended_button.setProperty("class", "toggle")
        self._attended_button.setToolTip(
            "Whether a human is watching this experiment. Agents are held to "
            "a stricter standard while you are here: a role that may recover "
            "from a fault alone does so only when you are not."
        )
        self._attended_button.toggled.connect(self._on_attendance_toggled)
        row.addWidget(self._attended_button)

        # ── Indicators: the status-bar half ───────────────────────────────
        self.status_line = QWidget()
        self.status_line.setObjectName("agent_status_line")
        line = QHBoxLayout(self.status_line)
        line.setContentsMargins(0, 0, 8, 0)
        line.setSpacing(14)

        self._gate_status_label = QLabel("")
        self._gate_status_label.setObjectName("agent_gate_status_label")
        self._gate_status_label.setToolTip(
            "How far agents may act, as set by Agent access in the header."
        )
        line.addWidget(self._gate_status_label)

        self._agents_active_label = QLabel("")
        self._agents_active_label.setObjectName("agents_active_label")
        self._agents_active_label.setToolTip(
            "Distinct agents that have acted on this station in the last few "
            "minutes."
        )
        line.addWidget(self._agents_active_label)

        self._attendance_label = QLabel("")
        self._attendance_label.setObjectName("attendance_status_label")
        line.addWidget(self._attendance_label)

        self._run_owner_label = QLabel("")
        self._run_owner_label.setObjectName("run_owner_label")
        # The one indicator that yields space when the window is narrow: the
        # full text is always in the tooltip.
        self._run_owner_label.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        self._run_owner_label.setMinimumWidth(0)
        self._run_owner_label.setToolTip(OWNER_TOOLTIP)
        line.addWidget(self._run_owner_label)

        self.set_agents_active(0)
        self.sync_from_mirror()

    # ------------------------------------------------------------------
    # Reflecting the engine
    # ------------------------------------------------------------------

    def sync_from_mirror(self) -> None:
        """Reflect the mirror's gate, attendance and run onto both halves.

        Called at construction and on every status snapshot, so a change made
        anywhere — an agent, a `i2as.ctl` invocation, this strip itself —
        shows here. Signals are blocked while the controls are set, because a
        reflected value is not an operator action and must not be pushed back
        down as one.
        """
        gate = str(self._mirror.agent_gate())
        button = self._gate_buttons.get(gate)
        if button is None:
            logger.warning("Takeover strip: unknown agent gate %r", gate)
        elif not button.isChecked():
            button.blockSignals(True)
            button.setChecked(True)
            button.blockSignals(False)
        color, words = GATE_STATUS.get(gate, ("", f"Agents {gate}"))
        self._gate_status_label.setText(
            f"<span style='color:{color}'>●</span> <b>{words}</b>" if color else words
        )

        attended = bool(self._mirror.attended())
        if attended != self._attended_button.isChecked():
            self._attended_button.blockSignals(True)
            self._attended_button.setChecked(attended)
            self._attended_button.blockSignals(False)
        self._attended_button.setText("✓ Attended" if attended else "Attended")
        self._attendance_label.setText("attended" if attended else "unattended")

        owner = self._mirror.run_owner()
        actor_id = str(owner.get("id") or "") if owner else ""
        text = RUN_OWNER_TEXT.format(owner=actor_id) if actor_id else ""
        tooltip = f"{text}. {OWNER_TOOLTIP}" if text else OWNER_TOOLTIP
        manifest = self._mirror.run_manifest()
        if actor_id and manifest:
            procedure = str(
                manifest.get("procedure") or manifest.get("procedure_class") or ""
            )
            if procedure:
                text = RUN_TEXT.format(owner=actor_id, procedure=procedure)
                params = manifest.get("params") or {}
                listing = "\n".join(
                    f"  {key} = {value}" for key, value in params.items()
                ) or "  (no parameters)"
                tooltip = (
                    RUN_PARAMS_TOOLTIP.format(
                        procedure=procedure, owner=actor_id, params=listing
                    )
                    + f"\n\n{OWNER_TOOLTIP}"
                )
        self._run_owner_label.setText(text)
        self._run_owner_label.setToolTip(tooltip)

    def set_agents_active(self, count: int) -> None:
        """Show how many agents are currently acting.

        Args:
            count: Distinct agent actor ids seen recently (the **Agent
                panel**'s ledger answers this).
        """
        count = int(count)
        self._agents_active_label.setText(
            f"{count} agent acting" if count == 1 else f"{count} agents acting"
        )

    # ------------------------------------------------------------------
    # Operator actions
    # ------------------------------------------------------------------

    def _on_gate_toggled(self, checked: bool, gate: str) -> None:
        """Push a newly selected kill-switch setting down into the engine.

        Args:
            checked: Whether this segment is the one now selected (the
                deselected segment also reports, and is ignored).
            gate: The ``AgentGate`` value this segment stands for.
        """
        if not checked:
            return
        logger.info("Operator set the agent gate to %s", gate)
        self._orchestrator.set_agent_gate(gate)

    def _on_attendance_toggled(self, checked: bool) -> None:
        """Record attendance in both places it has to be true.

        With an experiment open the session layer is the single writer: it
        persists the flag on the record AND pushes it down into the engine,
        so writing it here as well would submit the same command twice. With
        no experiment open there is no record to write, and the engine still
        has to know.

        Args:
            checked: ``True`` when a human is present.
        """
        self._attended_button.setText("✓ Attended" if checked else "Attended")
        if (
            self._session_manager is not None
            and self._session_manager.current_experiment() is not None
        ):
            self._session_manager.set_attended(checked)
            return
        self._orchestrator.set_attendance(checked)
