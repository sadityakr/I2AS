"""TakeoverStrip — the **Takeover strip**: the human's controls over the agents.

Three controls, in the header where the physicist already looks for the state
of their cryostat, because taking the machine back must never be somewhere you
have to go and find:

* **The kill switch**, tri-state (`active` / `read-only` / `revoked`). Applied
  through the client's ``set_agent_gate()``, so the engine — the single
  enforcement point — decides, and REFLECTED from the **Status mirror**, so an
  agent that gated itself, or a `i2as.ctl` invocation that did, shows here
  without this widget having been told.
* **The attendance toggle**, which is one fact with two homes: the experiment
  record (it must survive a restart) and the engine (it must be readable where
  the gateway's permission matrix is evaluated). Both are written through the
  session layer's single writer when an experiment is open, and directly into
  the engine when none is.
* **"agents active: N"**, the count of distinct agents that have acted
  recently, rendered from the **Agent panel**'s own ledger — the panel already
  sees every agent action, so counting them twice would be counting them
  differently.
* **"run owned by N"**, the **Run owner** of the run in flight, reflected
  from the same mirror: whose run it is decides who may end it (GLOSSARY.md's
  *Run owner*), so the fact belongs beside the controls that decide how far
  agents may go at all. A read, never a control — ownership is a fact about
  the run, and there is nothing here to set.

Nothing here is ever disabled by the gate. A kill switch that could lock the
human out of their own instrument would be a hazard rather than a safeguard,
and this widget is the human's end of it.
"""

from __future__ import annotations

import logging

from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QWidget,
)

from i2as.core.events import AgentGate
from i2as.core.orchestrator_proxy import OrchestratorProxy
from i2as.core.status_mirror import StatusMirror
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


#: What the strip says about the run in flight, and nothing at all when
#: there is none: an empty label rather than "no run", because the state bar
#: beside it already says the station is idle.
RUN_OWNER_TEXT = "run owned by {owner}"

#: Why the run owner is worth a line in the header at all. Also the whole of
#: what the line says when the header is too narrow to show its text.
OWNER_TOOLTIP = (
    "Who started the run in flight. Only that actor — or you — may abort it "
    "or attest to its steps; another agent has to take it over deliberately, "
    "and the takeover is recorded."
)


class TakeoverStrip(QWidget):
    """The header's agent-control strip.

    ObjectNames (API for tests and muscle memory): the strip is
    ``takeover_strip``, its radios ``agent_gate_active_radio`` /
    ``agent_gate_read_only_radio`` / ``agent_gate_revoked_radio``, the
    attendance box ``takeover_attended_checkbox``, the indicator
    ``agents_active_label`` and the run-owner line ``run_owner_label``.

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
        self._radios: dict[str, QRadioButton] = {}

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(QLabel("Agents:"))

        self._gate_group = QButtonGroup(self)
        self._gate_group.setExclusive(True)
        for gate, text, tooltip in GATE_CHOICES:
            radio = QRadioButton(text)
            radio.setObjectName(f"agent_gate_{gate.value}_radio")
            radio.setToolTip(tooltip)
            radio.toggled.connect(
                lambda checked, value=gate.value: self._on_gate_toggled(
                    checked, value
                )
            )
            self._gate_group.addButton(radio)
            self._radios[gate.value] = radio
            row.addWidget(radio)

        self._attended_checkbox = QCheckBox("Attended")
        self._attended_checkbox.setObjectName("takeover_attended_checkbox")
        self._attended_checkbox.setToolTip(
            "Whether a human is watching this experiment. Agents are held to "
            "a stricter standard while you are here: a role that may recover "
            "from a fault alone does so only when you are not."
        )
        self._attended_checkbox.toggled.connect(self._on_attendance_toggled)
        row.addWidget(self._attended_checkbox)

        self._run_owner_label = QLabel("")
        self._run_owner_label.setObjectName("run_owner_label")
        self._run_owner_label.setProperty("class", "secondary_label")
        # The one widget in the header that yields space when the window is
        # at its narrowest: it is the only READ here, and the controls beside
        # it are the human's way of taking the machine back — a kill switch
        # whose labels had been squeezed to fit a status line would be the
        # wrong trade. The full text is always in the tooltip.
        self._run_owner_label.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        self._run_owner_label.setMinimumWidth(0)
        self._run_owner_label.setToolTip(OWNER_TOOLTIP)
        row.addWidget(self._run_owner_label)

        self._agents_active_label = QLabel("")
        self._agents_active_label.setObjectName("agents_active_label")
        self._agents_active_label.setProperty("class", "secondary_label")
        self._agents_active_label.setToolTip(
            "Distinct agents that have acted on this station in the last few "
            "minutes."
        )
        row.addWidget(self._agents_active_label)

        self.set_agents_active(0)
        self.sync_from_mirror()

    # ------------------------------------------------------------------
    # Reflecting the engine
    # ------------------------------------------------------------------

    def sync_from_mirror(self) -> None:
        """Reflect the mirror's gate and attendance onto the controls.

        Called at construction and on every status snapshot, so a change made
        anywhere — an agent, a `i2as.ctl` invocation, this strip itself —
        shows here. Signals are blocked while the widgets are set, because a
        reflected value is not an operator action and must not be pushed back
        down as one.
        """
        gate = str(self._mirror.agent_gate())
        radio = self._radios.get(gate)
        if radio is None:
            logger.warning("Takeover strip: unknown agent gate %r", gate)
        elif not radio.isChecked():
            radio.blockSignals(True)
            radio.setChecked(True)
            radio.blockSignals(False)
        attended = bool(self._mirror.attended())
        if attended != self._attended_checkbox.isChecked():
            self._attended_checkbox.blockSignals(True)
            self._attended_checkbox.setChecked(attended)
            self._attended_checkbox.blockSignals(False)
        owner = self._mirror.run_owner()
        actor_id = str(owner.get("id") or "") if owner else ""
        text = RUN_OWNER_TEXT.format(owner=actor_id) if actor_id else ""
        self._run_owner_label.setText(text)
        self._run_owner_label.setToolTip(
            f"{text}. {OWNER_TOOLTIP}" if text else OWNER_TOOLTIP
        )

    def set_agents_active(self, count: int) -> None:
        """Show how many agents are currently acting.

        Args:
            count: Distinct agent actor ids seen recently (the **Agent
                panel**'s ledger answers this).
        """
        self._agents_active_label.setText(f"agents active: {int(count)}")

    # ------------------------------------------------------------------
    # Operator actions
    # ------------------------------------------------------------------

    def _on_gate_toggled(self, checked: bool, gate: str) -> None:
        """Push a newly selected kill-switch setting down into the engine.

        Args:
            checked: Whether this radio is the one now selected (the
                deselected radio of the pair also reports, and is ignored).
            gate: The ``AgentGate`` value this radio stands for.
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
        if (
            self._session_manager is not None
            and self._session_manager.current_experiment() is not None
        ):
            self._session_manager.set_attended(checked)
            return
        self._orchestrator.set_attendance(checked)
