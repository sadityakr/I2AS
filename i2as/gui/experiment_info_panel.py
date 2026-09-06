"""ExperimentInfoPanel — the Session Information quadrant (experiment + sample metadata)."""

from __future__ import annotations

from pathlib import Path

import qtawesome as qta
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from i2as.core.events import Verdict
from i2as.core.paths import measurement_root
from i2as.gui.experiment_dialogs import (
    CloseExperimentDialog,
    EnvelopeEditorWidget,
    StartExperimentDialog,
)
from i2as.gui.form_autosave import FormAutosaveState
from i2as.gui.theme import TEXT_PRIMARY
from i2as.session.manager import ExperimentManager

_ELN_NOT_CONFIGURED_TEXT = "eLab publishing is not configured yet"
_OUTSIDE_SESSION_NOTE_TEXT = "saving outside the current session folder"


class ExperimentInfoPanel(QWidget):
    """The Session Information quadrant: experiment control, plus sample fields.

    ObjectNames (``session_info_quadrant``, ``session_info_scroll``,
    ``sample_name_input``, ``sample_id_input``, ``comments_input``,
    ``data_dir_input``, ``browse_btn``, ``experiment_status_label``,
    ``start_close_experiment_btn``, ``attended_checkbox``) are preserved API
    — tests and muscle memory rely on them.

    Args:
        parent: Optional Qt parent widget.
        session_manager: The L6 ExperimentManager. When ``None`` (unit tests
            that build the panel standalone), the experiment row is shown
            but its button stays disabled.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        session_manager: ExperimentManager | None = None,
    ) -> None:
        super().__init__(parent)
        self._session_manager = session_manager
        # Data Dir transition tracking (rule 3): _last_experiment_id
        # detects an actual open/switch transition (vs. a same-experiment
        # experiment_changed re-emit from e.g. an attendance/findings edit);
        # _pre_session_data_dir remembers the field's manual value from just
        # before the first such transition, so closing can restore it.
        self._last_experiment_id: str | None = None
        self._pre_session_data_dir: str | None = None
        self.setObjectName("session_info_quadrant")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)
        outer.addWidget(QLabel("<b>Experiment</b>"))
        outer.addLayout(self._build_experiment_row())

        # One scrolled region for everything below the experiment row: the
        # envelope editor is one row per enveloped quantity, and a setup with
        # five of them would otherwise push the sample fields off the bottom
        # of the quadrant.
        scrolled = QWidget()
        scrolled_layout = QVBoxLayout(scrolled)
        scrolled_layout.setContentsMargins(0, 0, 0, 0)
        scrolled_layout.setSpacing(4)
        self._build_envelope_editor(scrolled_layout)
        scrolled_layout.addWidget(QLabel("<b>Sample Info</b>"))
        scrolled_layout.addWidget(self._build_form())

        scroll = QScrollArea()
        scroll.setObjectName("session_info_scroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(scrolled)
        outer.addWidget(scroll)

        outer.addWidget(QLabel("<b>eLab</b>"))
        self._eln_status_label = QLabel(_ELN_NOT_CONFIGURED_TEXT)
        self._eln_status_label.setObjectName("eln_status_label")
        self._eln_status_label.setWordWrap(True)
        outer.addWidget(self._eln_status_label)

        if self._session_manager is not None:
            self._session_manager.experiment_changed.connect(self._on_experiment_changed)
            current = self._session_manager.current_experiment()
            self._on_experiment_changed(current.to_dict() if current is not None else {})

    def _build_experiment_row(self) -> QVBoxLayout:
        """Build the experiment status label, Start/Close button, and attendance box.

        Returns:
            A layout with the status/button row plus the (initially hidden)
            attendance checkbox.
        """
        section = QVBoxLayout()

        status_row = QHBoxLayout()
        self._experiment_status_label = QLabel("No experiment open")
        self._experiment_status_label.setObjectName("experiment_status_label")
        self._experiment_status_label.setWordWrap(True)
        status_row.addWidget(self._experiment_status_label, 1)

        self._start_close_btn = QPushButton("Start Experiment…")
        self._start_close_btn.setObjectName("start_close_experiment_btn")
        if self._session_manager is None:
            self._start_close_btn.setEnabled(False)
            self._start_close_btn.setToolTip("Session management is not available")
        else:
            self._start_close_btn.clicked.connect(self._on_start_close_clicked)
        status_row.addWidget(self._start_close_btn)
        section.addLayout(status_row)

        self._attended_checkbox = QCheckBox("Attended")
        self._attended_checkbox.setObjectName("attended_checkbox")
        self._attended_checkbox.setChecked(True)
        self._attended_checkbox.setVisible(False)
        self._attended_checkbox.toggled.connect(self._on_attended_toggled)
        section.addWidget(self._attended_checkbox)

        return section

    def _build_envelope_editor(self, outer: QVBoxLayout) -> None:
        """Build the experiment header's own **Session envelope** editor.

        The envelope belongs where the experiment is: the Start Experiment
        dialog sets it at the one moment the operator knows what is mounted,
        and this editor is how it is NARROWED afterwards, without closing the
        experiment to do it. Deliberately the same widget class as the
        dialog's, so the two can never diverge in what they accept or how
        they refuse it.

        Hidden outright with no session layer and while no experiment is
        open: an envelope with no experiment to bound would be a control that
        does nothing.

        Args:
            outer: The panel's root layout, which the editor is added to.
        """
        self._envelope_editor: EnvelopeEditorWidget | None = None
        self._pending_envelope_request = ""
        variables = (
            self._session_manager.envelope_variables()
            if self._session_manager is not None
            else {}
        )
        if not variables:
            return
        self._envelope_editor = EnvelopeEditorWidget(variables)
        self._envelope_editor.changed.connect(self._on_envelope_changed)
        self._envelope_editor.setVisible(False)
        outer.addWidget(self._envelope_editor)

        row = QHBoxLayout()
        self._envelope_verdict_label = QLabel("")
        self._envelope_verdict_label.setObjectName("envelope_verdict_label")
        self._envelope_verdict_label.setProperty("class", "verdict_badge")
        self._envelope_verdict_label.setProperty("severity", "ok")
        self._envelope_verdict_label.setWordWrap(True)
        self._envelope_verdict_label.setVisible(False)
        row.addWidget(self._envelope_verdict_label, 1)
        row.addStretch()
        self._envelope_apply_btn = QPushButton("Apply envelope")
        self._envelope_apply_btn.setObjectName("envelope_apply_btn")
        self._envelope_apply_btn.setToolTip(
            "Bound this experiment to the entered limits from now on. Every "
            "writer is held to them — a procedure's targets, a manual action, "
            "an agent."
        )
        self._envelope_apply_btn.clicked.connect(self._on_apply_envelope)
        row.addWidget(self._envelope_apply_btn)
        self._envelope_row = row
        outer.addLayout(row)
        self._set_envelope_visible(False)

    def _set_envelope_visible(self, visible: bool) -> None:
        """Show or hide the envelope editor and its Apply row together.

        Args:
            visible: Whether an experiment is open to bound.
        """
        if self._envelope_editor is None:
            return
        self._envelope_editor.setVisible(visible)
        self._envelope_apply_btn.setVisible(visible)
        if not visible:
            self._envelope_verdict_label.setVisible(False)

    def _on_envelope_changed(self) -> None:
        """Keep Apply available exactly while the entry could be sent.

        The editor shows its own refusal on its own badge, so this does not
        repeat it — saying the same sentence twice, six pixels apart, reads
        as two different problems. What is left for the panel is the half the
        editor cannot know: whether the value may be sent at all.
        """
        if self._envelope_editor is None:
            return
        self._envelope_apply_btn.setEnabled(not self._envelope_editor.error())
        self._envelope_verdict_label.setVisible(False)

    def _on_apply_envelope(self) -> None:
        """Install the edited envelope on the open experiment.

        Through the session manager, which is the single writer for both
        homes of the value — the experiment record and the engine.
        """
        if self._envelope_editor is None or self._session_manager is None:
            return
        if self._envelope_editor.error():
            # Belt and braces: the button is disabled while the entry is
            # unusable, and the editor is already showing why.
            return
        self._pending_envelope_request = self._session_manager.set_experiment_envelope(
            self._envelope_editor.envelope()
        )
        self._show_envelope_verdict("Envelope applied", "ok")

    def on_verdict(self, verdict: object) -> None:
        """Answer the Apply click with the engine's own verdict.

        Forwarded by the window that owns the client connection (the
        destruction-order rule). Only the verdict answering THIS panel's own
        request is rendered: every action gets exactly one verdict, and a
        panel that showed somebody else's would be reporting on an action it
        did not take.

        Args:
            verdict: Anything off the client's verdict stream.
        """
        if not isinstance(verdict, Verdict) or self._envelope_editor is None:
            return
        if not self._pending_envelope_request:
            return
        if verdict.request_id != self._pending_envelope_request:
            return
        self._pending_envelope_request = ""
        if verdict.ok:
            self._show_envelope_verdict("Envelope applied", "ok")
        else:
            self._show_envelope_verdict(
                f"{verdict.code.value} — {verdict.reason}", "error"
            )

    def _show_envelope_verdict(self, text: str, severity: str) -> None:
        """Show one line on the envelope's verdict badge.

        Only ever the ENGINE's answer to an Apply — the editor owns the
        refusals it can decide itself.

        Args:
            text: What to say.
            severity: ``"ok"`` or ``"error"`` — the validated ``verdict_badge``
                QSS severities (no widget stylesheet, theme tokens only).
        """
        label = self._envelope_verdict_label
        label.setText(text)
        if label.property("severity") != severity:
            label.setProperty("severity", severity)
            # Dynamic-property QSS is re-evaluated only after an
            # unpolish/polish cycle; the badge is a single label, so this one
            # widget is the whole cycle.
            label.style().unpolish(label)
            label.style().polish(label)
        label.setVisible(True)

    def _on_start_close_clicked(self) -> None:
        """Open the Start or Close Experiment dialog depending on current state."""
        if self._session_manager is None:
            return
        if self._session_manager.current_experiment() is None:
            self._run_start_dialog()
        else:
            self._run_close_dialog()

    def _run_start_dialog(self) -> None:
        """Open the Start Experiment dialog and start what it collected.

        The dialog is handed the setup's own bounds per enveloped quantity, so
        its envelope editor is pre-filled and the operator narrows rather than
        composes; whatever it returns is installed on the Orchestrator by the
        session manager as part of opening the experiment.
        """
        assert self._session_manager is not None
        dialog = StartExperimentDialog(
            self._session_manager.roster,
            self,
            envelope_variables=self._session_manager.envelope_variables(),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        title, user_id, attended, dirname = dialog.result_values()
        try:
            self._session_manager.start_experiment(
                title=title,
                user_id=user_id,
                sample_info=self.get_sample_info(),
                attended=attended,
                experiment_dirname=dirname,
                envelope=dialog.envelope(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Could not start experiment", str(exc))

    def _run_close_dialog(self) -> None:
        assert self._session_manager is not None
        record = self._session_manager.current_experiment()
        current_findings = record.findings if record is not None else ""
        dialog = CloseExperimentDialog(current_findings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._session_manager.set_findings(dialog.findings())
        self._session_manager.close_experiment()

    def _on_attended_toggled(self, checked: bool) -> None:
        if self._session_manager is not None:
            self._session_manager.set_attended(checked)

    def _on_experiment_changed(self, record: dict) -> None:
        """Reflect an ExperimentManager ``experiment_changed`` payload in the row.

        Args:
            record: ``ExperimentRecord.to_dict()``, or ``{}`` when none open.
        """
        if not record:
            self._experiment_status_label.setText("No experiment open")
            self._start_close_btn.setText("Start Experiment…")
            self._attended_checkbox.setVisible(False)
            self._set_envelope_visible(False)
            self._eln_status_label.setText(_ELN_NOT_CONFIGURED_TEXT)
            self._restore_data_dir_on_close()
            return

        user_id = record.get("user_id", "")
        user_name = user_id
        if self._session_manager is not None:
            user = self._session_manager.roster.get(user_id)
            if user is not None and user.name:
                user_name = user.name
        attended = bool(record.get("attended", True))

        self._experiment_status_label.setText(
            f"{record.get('title', '')} — {user_name} "
            f"({'attended' if attended else 'unattended'})"
        )
        self._start_close_btn.setText("Close Experiment…")
        self._set_envelope_visible(True)
        if self._envelope_editor is not None:
            # The experiment already HAS an envelope; showing the setup's own
            # limits instead would invite widening it back out by accident.
            self._envelope_editor.set_bounds(record.get("envelope") or {})
        self._attended_checkbox.setVisible(True)
        self._attended_checkbox.blockSignals(True)
        self._attended_checkbox.setChecked(attended)
        self._attended_checkbox.blockSignals(False)

        eln_link = record.get("eln_link") or {}
        if eln_link.get("url"):
            self._eln_status_label.setText(f"Published: {eln_link['url']}")
        else:
            self._eln_status_label.setText(f"Not published yet — {_ELN_NOT_CONFIGURED_TEXT}")

        self._force_data_dir_on_open(record.get("experiment_id", ""))

    def _force_data_dir_on_open(self, experiment_id: str) -> None:
        """Force Data Dir to the (newly) active session's own folder.

        Only acts on an actual transition — a different ``experiment_id``
        than last seen (covers both a brand-new/switched-in open experiment
        and, from ``None``, the very first one in this sequence). A
        same-experiment re-emit (attendance/findings edits) leaves whatever
        the physicist has since typed alone. The field's text just before
        the first such transition is captured so closing can restore it.

        Args:
            experiment_id: The now-open experiment's id (never empty here).
        """
        if experiment_id == self._last_experiment_id:
            return
        if self._last_experiment_id is None:
            self._pre_session_data_dir = self._data_dir_input.text()
        self._last_experiment_id = experiment_id
        if self._session_manager is not None:
            data_dir = self._session_manager.current_data_dir()
            if data_dir is not None:
                self._data_dir_input.setText(str(data_dir))
        self._update_data_dir_note()

    def _restore_data_dir_on_close(self) -> None:
        """Restore whatever Data Dir held immediately before the session opened."""
        if self._last_experiment_id is not None and self._pre_session_data_dir is not None:
            self._data_dir_input.setText(self._pre_session_data_dir)
        self._last_experiment_id = None
        self._pre_session_data_dir = None
        self._update_data_dir_note()

    def _build_form(self) -> QWidget:
        """Build the sample-info form (session-level metadata).

        Returns:
            A QWidget with name, ID, comments, and data-dir form fields.
        """
        box = QWidget()
        form = QFormLayout(box)

        self._sample_name_input = QLineEdit()
        self._sample_name_input.setObjectName("sample_name_input")
        self._sample_name_input.setPlaceholderText("e.g. Si_001")
        form.addRow("Name:", self._sample_name_input)

        self._sample_id_input = QLineEdit()
        self._sample_id_input.setObjectName("sample_id_input")
        self._sample_id_input.setPlaceholderText("e.g. S2024-01")
        form.addRow("ID:", self._sample_id_input)

        self._comments_input = QTextEdit()
        self._comments_input.setObjectName("comments_input")
        self._comments_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        form.addRow("Comments:", self._comments_input)

        dir_row = QHBoxLayout()
        # Starts empty ("no explicit choice yet"); apply_session() (called
        # right after construction by MonitorWindow) and/or the ExperimentManager
        # experiment_changed handler above fill in the right value — the
        # experiment's own folder when one is open, else the measurement_root()
        # default (see _default_data_dir_text()).
        self._data_dir_input = QLineEdit()
        self._data_dir_input.setObjectName("data_dir_input")
        self._data_dir_input.textChanged.connect(self._update_data_dir_note)
        browse_btn = QPushButton("Browse…")
        browse_btn.setObjectName("browse_btn")
        browse_btn.setIcon(qta.icon("fa5s.folder-open", color=TEXT_PRIMARY))
        browse_btn.setToolTip("Choose the directory where run data is saved")
        browse_btn.clicked.connect(self._on_browse_dir)
        dir_row.addWidget(self._data_dir_input)
        dir_row.addWidget(browse_btn)
        form.addRow("Data Dir:", dir_row)

        self._data_dir_note = QLabel(_OUTSIDE_SESSION_NOTE_TEXT)
        self._data_dir_note.setObjectName("data_dir_note")
        self._data_dir_note.setWordWrap(True)
        self._data_dir_note.hide()
        form.addRow("", self._data_dir_note)

        return box

    def _on_browse_dir(self) -> None:
        """Open a directory browser and fill the data-dir field.

        Opens at the open experiment's own data folder (rule 3) when one is
        active, else at whatever the field currently holds. A selection
        outside the open experiment's folder is rejected outright (hard
        containment — see ``is_data_dir_contained()``); the field is left
        unchanged and a warning is shown instead of accepting it.
        """
        start_dir = self._data_dir_input.text()
        experiment_folder = self._current_experiment_folder()
        if self._session_manager is not None:
            current_dir = self._session_manager.current_data_dir()
            if current_dir is not None:
                start_dir = str(current_dir)
        selected = QFileDialog.getExistingDirectory(self, "Select Data Directory", start_dir)
        if not selected:
            return
        if experiment_folder is not None:
            try:
                outside = not Path(selected).resolve().is_relative_to(
                    experiment_folder.resolve()
                )
            except (OSError, ValueError):
                outside = True
            if outside:
                QMessageBox.warning(
                    self,
                    "Outside experiment folder",
                    "The data directory must stay inside the open experiment's "
                    f"folder:\n{experiment_folder}",
                )
                return
        self._data_dir_input.setText(selected)

    def _update_data_dir_note(self) -> None:
        """Show/hide the "saving outside the current session folder" note.

        Only meaningful while an experiment is open: compares the field's
        current path against the experiment folder
        (``current_data_dir().parent``, since ``current_data_dir()`` is that
        folder's ``data/`` sub-directory). No experiment open, or an empty
        field, both hide the note. This is live typing feedback only — hard
        enforcement happens at ``is_data_dir_contained()``, read by the
        caller that actually starts a run.
        """
        session_folder = self._current_experiment_folder()
        text = self._data_dir_input.text().strip()
        if session_folder is None or not text:
            self._data_dir_note.hide()
            return
        try:
            outside = not Path(text).resolve().is_relative_to(session_folder.resolve())
        except (OSError, ValueError):
            outside = True
        self._data_dir_note.setVisible(outside)

    def _current_experiment_folder(self) -> Path | None:
        """Return the open experiment's session folder, or ``None`` when none is open."""
        if self._session_manager is None:
            return None
        data_dir = self._session_manager.current_data_dir()
        return data_dir.parent if data_dir is not None else None

    # ------------------------------------------------------------------
    # Public accessors (surfaced by MonitorWindow to ProcedureWindow)
    # ------------------------------------------------------------------

    def get_sample_info(self) -> dict[str, str]:
        """Return the current sample info as a dict.

        Returns:
            Dict with keys ``sample_name``, ``sample_id``, ``comments``.
        """
        return {
            "sample_name": self._sample_name_input.text().strip(),
            "sample_id": self._sample_id_input.text().strip(),
            "comments": self._comments_input.toPlainText().strip(),
        }

    def get_data_dir(self) -> str:
        """Return the configured data directory path.

        Returns:
            Absolute path string; falls back to the open experiment's own
            data folder, or (no experiment open) ``measurement_root()``, if
            the field is empty.
        """
        return self._data_dir_input.text().strip() or self._default_data_dir_text()

    def _default_data_dir_text(self) -> str:
        """Return the fallback Data Dir text for an empty field/experiment state.

        The open experiment's own data folder when one is active (mirrors the
        experiment_changed-driven forcing above), else
        ``i2as.core.paths.measurement_root()`` — the same substitution
        ``form_autosave``'s now-empty ``_DEFAULT_DATA_DIR`` relies on the GUI
        to make (form_autosave itself stays Qt-free and cannot resolve a
        platform Documents directory).
        """
        experiment_data_dir = (
            self._session_manager.current_data_dir()
            if self._session_manager is not None
            else None
        )
        if experiment_data_dir is not None:
            return str(experiment_data_dir)
        return str(measurement_root())

    def is_data_dir_contained(self) -> bool:
        """Return whether the current Data Dir is inside the open experiment's folder.

        Always ``True`` when no experiment is open (nothing to contain
        against) or the field is empty (falls back to the experiment's own
        folder via ``_default_data_dir_text()``). Mirrors the
        ``is_relative_to()`` check ``_update_data_dir_note()`` uses for the
        live typing note, but this is the read a caller starting a run must
        enforce against — a path outside the open experiment's folder is
        rejected there, not merely flagged (see ``MonitorWindow``'s enforced
        Data Dir accessor).

        Returns:
            ``True`` when the field is empty, no experiment is open, or the
            resolved path is inside the open experiment's folder; ``False``
            otherwise (including when the path cannot be resolved).
        """
        experiment_folder = self._current_experiment_folder()
        text = self._data_dir_input.text().strip()
        if experiment_folder is None or not text:
            return True
        try:
            return Path(text).resolve().is_relative_to(experiment_folder.resolve())
        except (OSError, ValueError):
            return False

    def apply_session(self, state: FormAutosaveState) -> None:
        """Populate the fields from a loaded session.

        Args:
            state: The session whose sample metadata is applied.
        """
        self._sample_name_input.setText(state.sample_name)
        self._sample_id_input.setText(state.sample_id)
        self._comments_input.setPlainText(state.comments)
        self._data_dir_input.setText(state.data_dir or self._default_data_dir_text())
        self._update_data_dir_note()
