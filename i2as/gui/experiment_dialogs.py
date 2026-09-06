"""Modal dialogs for starting/closing an experiment and adding a roster user."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from i2as.core.plan import SETPOINT_PARAM_PREFIX, EnvelopeBound, ExperimentEnvelope
from i2as.session.models import User
from i2as.session.store import UserRoster


def _slugify(text: str) -> str:
    """Derive a roster-key slug from free text (lowercase, ``_``-joined)."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _format_bound(value: float | None) -> str:
    """Render a bound for an editor field (``""`` for "unbounded")."""
    return "" if value is None else f"{value:g}"


def _optional_float(value: Any) -> float | None:
    """Coerce one JSON bound to a float, or ``None`` for "unbounded".

    Args:
        value: The bound as it crossed the client boundary — a number,
            ``None``, or anything a malformed declaration might carry.

    Returns:
        The bound as a float, or ``None`` when it is absent or unreadable
        (an unreadable bound means "the setup does not bound this side",
        which is the reading that shows the operator a blank field rather
        than crashing the dialog).
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class _EnvelopeVariable:
    """One enveloped quantity as the GUI reads it: the dict form, typed.

    **The dict-form rule**: the GUI builds from declarations, never from
    engine objects. What a client is handed for an enveloped quantity is the
    JSON-safe dict a ``StatusSnapshot`` carried (``ExperimentManager
    .envelope_variables()`` → ``StatusMirror.envelope_variables()``), so this
    is what the editor parses — never ``core.plan.EnvelopeVariable``, which
    lives on the engine's side of the boundary and would make the editor
    unusable in production the moment it was reached for.

    Attributes:
        method_name: The ``@control`` capability that commands the quantity.
        param_name: That capability's setpoint parameter (``target_*``).
        config_min: Setup lower bound, or ``None`` when unbounded below.
        config_max: Setup upper bound, or ``None`` when unbounded above.
    """

    method_name: str = ""
    param_name: str = ""
    config_min: float | None = None
    config_max: float | None = None

    @classmethod
    def from_json(cls, record: Mapping[str, Any]) -> _EnvelopeVariable:
        """Build one from the dict a snapshot carried.

        Args:
            record: One value of ``envelope_variables()``, i.e. an
                ``EnvelopeVariable`` rendered as a JSON-safe dict.

        Returns:
            The typed view of it, with unreadable bounds read as unbounded.
        """
        return cls(
            method_name=str(record.get("method_name") or ""),
            param_name=str(record.get("param_name") or ""),
            config_min=_optional_float(record.get("config_min")),
            config_max=_optional_float(record.get("config_max")),
        )

    @property
    def unit_suffix(self) -> str:
        """Return the setpoint parameter's trailing unit token, or ``""``.

        Derived here rather than read off the record, because it is a
        property of ``EnvelopeVariable`` and properties do not survive the
        dataclass-to-dict rendering the snapshot does. The
        setpoint-parameter convention is the whole rule: ``target_T`` → T.
        """
        if not self.param_name.startswith(SETPOINT_PARAM_PREFIX):
            return ""
        return self.param_name[len(SETPOINT_PARAM_PREFIX):]


class AddUserDialog(QDialog):
    """Add one person to the user roster: name, email, ORCID, and an id."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add User")
        self._id_edited_by_hand = False

        form = QFormLayout()

        self._name_input = QLineEdit()
        self._name_input.setObjectName("user_name_input")
        self._name_input.textChanged.connect(self._on_name_changed)
        self._name_input.textChanged.connect(self._update_ok_enabled)
        form.addRow("Name:", self._name_input)

        self._id_input = QLineEdit()
        self._id_input.setObjectName("user_id_input")
        self._id_input.textEdited.connect(self._on_id_edited)
        self._id_input.textChanged.connect(self._update_ok_enabled)
        form.addRow("User ID:", self._id_input)

        self._email_input = QLineEdit()
        self._email_input.setObjectName("user_email_input")
        form.addRow("Email:", self._email_input)

        self._orcid_input = QLineEdit()
        self._orcid_input.setObjectName("user_orcid_input")
        form.addRow("ORCID:", self._orcid_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _on_name_changed(self, text: str) -> None:
        """Auto-fill the id from the name until the user edits it by hand."""
        if not self._id_edited_by_hand:
            self._id_input.setText(_slugify(text))

    def _on_id_edited(self, _text: str) -> None:
        self._id_edited_by_hand = True

    def _update_ok_enabled(self) -> None:
        self._ok_button.setEnabled(
            bool(self._name_input.text().strip()) and bool(self._id_input.text().strip())
        )

    def user(self) -> User:
        """Return the entered user. Only meaningful after ``exec()`` accepts.

        Returns:
            A ``User`` built from the form fields (``eln_user_id`` empty).
        """
        return User(
            user_id=self._id_input.text().strip(),
            name=self._name_input.text().strip(),
            email=self._email_input.text().strip(),
            orcid=self._orcid_input.text().strip(),
        )


class UserPickerWidget(QWidget):
    """A roster user combo plus an inline "New user…" flow.

    Shared by ``StartExperimentDialog`` and the Setup-tier ``LoginDialog`` so
    the roster-picking behavior (reload, inline add via ``AddUserDialog``,
    select-by-id) exists in exactly one place.
    """

    def __init__(self, roster: UserRoster, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._roster = roster

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self._combo = QComboBox()
        self._combo.setObjectName("user_picker_combo")
        row.addWidget(self._combo, 1)
        new_user_btn = QPushButton("New user…")
        new_user_btn.setObjectName("new_user_btn")
        new_user_btn.clicked.connect(self._on_new_user)
        row.addWidget(new_user_btn)

        self.reload(select_user_id=None)

    def reload(self, select_user_id: str | None) -> None:
        """Repopulate the combo from the roster, optionally selecting one.

        Args:
            select_user_id: A roster id to select if present, or ``None``.
        """
        self._combo.clear()
        for user in self._roster.list_users():
            self._combo.addItem(user.name or user.user_id, userData=user.user_id)
        if select_user_id:
            index = self._combo.findData(select_user_id)
            if index >= 0:
                self._combo.setCurrentIndex(index)

    def _on_new_user(self) -> None:
        dialog = AddUserDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        user = dialog.user()
        self._roster.add(user)
        self.reload(select_user_id=user.user_id)

    def selected_user_id(self) -> str:
        """Return the currently selected roster id, or ``""`` if none exist."""
        return self._combo.currentData() or ""

    def has_users(self) -> bool:
        """Return whether the combo currently offers at least one user."""
        return self._combo.count() > 0

    def selection_changed_signal(self):  # noqa: ANN201 — thin Qt signal passthrough
        """Expose the combo's ``currentIndexChanged`` for callers to connect to."""
        return self._combo.currentIndexChanged


class EnvelopeEditorWidget(QGroupBox):
    """Per-experiment sample bounds, pre-filled from the setup's own limits.

    The experiment envelope protects the *sample*: the config's limits protect
    the instrument and never change, while these bounds say what the device
    mounted for THIS experiment may see. The editor is deliberately pre-filled
    with the setup limits rather than left blank, because an envelope composed
    from nothing is an envelope nobody fills in — the operator's job here is to
    NARROW numbers that are already correct.

    Narrowing is enforced, not merely intended: a bound wider than the setup's
    own limit is rejected with a reason, since it could only mislead (the
    control-validation standard would refuse the value anyway).

    Bounds are entered in each quantity's SI unit, taken from the setpoint
    parameter's name (``target_T`` → T). A blank field means "unbounded on that
    side"; a VI with both fields blank contributes no bound at all.

    The editor reads the **dict form** and only the dict form: each variable
    arrives as the JSON-safe dict a ``StatusSnapshot`` carried, which is what
    ``ExperimentManager.envelope_variables()`` answers in production (see
    ``_EnvelopeVariable``).

    ObjectNames (API for tests and muscle memory): the group is
    ``envelope_editor_group``, its toggle ``envelope_enabled_checkbox``, the
    validation message ``envelope_error_label``, and each VI's two fields
    ``envelope_min_<vi_name>`` / ``envelope_max_<vi_name>``.

    Args:
        variables: ``{vi_name: envelope-variable dict}`` from
            ``ExperimentManager.envelope_variables()``.
        parent: Optional Qt parent widget.
    """

    changed = pyqtSignal()

    def __init__(
        self,
        variables: Mapping[str, Mapping[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Sample envelope", parent)
        self.setObjectName("envelope_editor_group")
        self._variables = {
            vi_name: _EnvelopeVariable.from_json(record)
            for vi_name, record in sorted(variables.items())
        }
        self._rows: dict[str, tuple[QLineEdit, QLineEdit]] = {}
        self._error = ""

        layout = QVBoxLayout(self)

        self._enabled_checkbox = QCheckBox(
            "Bound this experiment to a sample envelope"
        )
        self._enabled_checkbox.setObjectName("envelope_enabled_checkbox")
        self._enabled_checkbox.setChecked(True)
        self._enabled_checkbox.setToolTip(
            "Refuse any target or manual action that would take an instrument "
            "outside these bounds, for the whole experiment."
        )
        self._enabled_checkbox.toggled.connect(self._on_enabled_toggled)
        layout.addWidget(self._enabled_checkbox)

        form = QFormLayout()
        for vi_name, variable in self._variables.items():
            unit = variable.unit_suffix
            min_edit = QLineEdit(_format_bound(variable.config_min))
            min_edit.setObjectName(f"envelope_min_{vi_name}")
            min_edit.setPlaceholderText("no lower bound")
            max_edit = QLineEdit(_format_bound(variable.config_max))
            max_edit.setObjectName(f"envelope_max_{vi_name}")
            max_edit.setPlaceholderText("no upper bound")
            for edit in (min_edit, max_edit):
                edit.setToolTip(
                    f"{vi_name}.{variable.method_name}"
                    f"({variable.param_name}) — setup limit "
                    f"[{_format_bound(variable.config_min) or '-inf'}, "
                    f"{_format_bound(variable.config_max) or '+inf'}]"
                )
                edit.textChanged.connect(self._on_field_changed)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(min_edit, 1)
            row.addWidget(QLabel("to"))
            row.addWidget(max_edit, 1)
            row_widget = QWidget()
            row_widget.setLayout(row)
            label = f"{vi_name} ({unit}):" if unit else f"{vi_name}:"
            form.addRow(label, row_widget)
            self._rows[vi_name] = (min_edit, max_edit)
        layout.addLayout(form)

        # Reuses the validated "verdict_badge" QSS class
        # rather than inventing a colour for one message.
        self._error_label = QLabel("")
        self._error_label.setObjectName("envelope_error_label")
        self._error_label.setProperty("class", "verdict_badge")
        self._error_label.setProperty("severity", "error")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        self._revalidate()

    def _on_enabled_toggled(self, checked: bool) -> None:
        """Enable or disable every field, then revalidate."""
        for min_edit, max_edit in self._rows.values():
            min_edit.setEnabled(checked)
            max_edit.setEnabled(checked)
        self._revalidate()

    def _on_field_changed(self, _text: str) -> None:
        self._revalidate()

    def _parse(self) -> tuple[dict[str, EnvelopeBound], str]:
        """Return ``(bounds, error)`` for the current field contents.

        Returns:
            The parsed bounds keyed by VI name and an error message; the
            message is empty when every field is acceptable, and the bounds
            are then complete.
        """
        bounds: dict[str, EnvelopeBound] = {}
        for vi_name, (min_edit, max_edit) in self._rows.items():
            variable = self._variables[vi_name]
            values: dict[str, float | None] = {}
            for side, edit in (("min", min_edit), ("max", max_edit)):
                text = edit.text().strip()
                if not text:
                    values[side] = None
                    continue
                try:
                    values[side] = float(text)
                except ValueError:
                    return {}, f"{vi_name}: {side} bound {text!r} is not a number"
            if values["min"] is None and values["max"] is None:
                continue
            if values["min"] is not None and variable.config_min is not None:
                if values["min"] < variable.config_min:
                    return {}, (
                        f"{vi_name}: minimum {values['min']:g} is below the "
                        f"setup limit {variable.config_min:g} — an envelope "
                        f"narrows the setup's limits, it cannot widen them"
                    )
            if values["max"] is not None and variable.config_max is not None:
                if values["max"] > variable.config_max:
                    return {}, (
                        f"{vi_name}: maximum {values['max']:g} is above the "
                        f"setup limit {variable.config_max:g} — an envelope "
                        f"narrows the setup's limits, it cannot widen them"
                    )
            try:
                bounds[vi_name] = EnvelopeBound(
                    min_value=values["min"], max_value=values["max"]
                )
            except (TypeError, ValueError) as exc:
                return {}, f"{vi_name}: {exc}"
        return bounds, ""

    def _revalidate(self) -> None:
        """Re-parse the fields, show any error, and announce the change."""
        if self._enabled_checkbox.isChecked():
            _bounds, self._error = self._parse()
        else:
            self._error = ""
        self._error_label.setText(self._error)
        self._error_label.setVisible(bool(self._error))
        self.changed.emit()

    def error(self) -> str:
        """Return why the current entry is unusable, or ``""`` when it is valid."""
        return self._error

    def set_bounds(self, bounds: Mapping[str, Mapping[str, Any]] | None) -> None:
        """Show an envelope that is already in force, in place of the defaults.

        What the experiment header needs and the Start dialog does not: an
        experiment that is already open HAS an envelope, and an editor still
        showing the setup's own limits would invite the operator to widen it
        back out by accident. A VI the envelope says nothing about falls back
        to the setup bounds it was pre-filled with, since that is what the
        experiment is actually bounded by.

        Args:
            bounds: ``{vi_name: {"min_value", "max_value", …}}`` — the stored
                envelope's dict form (``ExperimentRecord.envelope``) — or
                ``None``/empty to return every field to the setup's limits
                and switch the editor off (no envelope is in force).
        """
        stored = dict(bounds or {})
        for vi_name, (min_edit, max_edit) in self._rows.items():
            variable = self._variables[vi_name]
            record = stored.get(vi_name)
            if record is None:
                low, high = variable.config_min, variable.config_max
            else:
                low = _optional_float(record.get("min_value"))
                high = _optional_float(record.get("max_value"))
            for edit, value in ((min_edit, low), (max_edit, high)):
                edit.blockSignals(True)
                edit.setText(_format_bound(value))
                edit.blockSignals(False)
        self._enabled_checkbox.blockSignals(True)
        self._enabled_checkbox.setChecked(bool(stored))
        self._enabled_checkbox.blockSignals(False)
        self._on_enabled_toggled(self._enabled_checkbox.isChecked())

    def envelope(self) -> ExperimentEnvelope | None:
        """Return the edited envelope, or ``None`` for "no envelope".

        Returns:
            An ``ExperimentEnvelope`` over every VI with at least one bound
            entered; ``None`` when the editor is switched off, when no bound
            was entered at all, or while ``error()`` is non-empty (the dialog
            keeps OK disabled in that case, so this is belt and braces).
        """
        if not self._enabled_checkbox.isChecked():
            return None
        bounds, error = self._parse()
        if error or not bounds:
            return None
        return ExperimentEnvelope(bounds=bounds)


class StartExperimentDialog(QDialog):
    """Collect a title, user, attendance flag, and folder name to open a new experiment.

    The folder name field lets the operator override the experiment's
    directory name (always directly under the active session — flat, no
    nesting); left alone, it auto-fills from the title the same way
    ``AddUserDialog``'s id field auto-fills from the name, and stops
    auto-filling the moment the operator types in it by hand.

    When the caller supplies *envelope_variables* the dialog also carries an
    ``EnvelopeEditorWidget``, so the experiment's sample bounds are set where
    the experiment is opened — the only moment at which the operator knows what
    is mounted.

    Args:
        roster: The setup-local user roster.
        parent: Optional Qt parent widget.
        envelope_variables: ``{vi_name: envelope-variable dict}`` (see
            ``ExperimentManager.envelope_variables()``). ``None`` or empty
            leaves the envelope editor out entirely, which is what a caller
            with no station wired — a unit test, say — gets.
    """

    def __init__(
        self,
        roster: UserRoster,
        parent: QWidget | None = None,
        envelope_variables: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Start Experiment")
        self._dirname_edited_by_hand = False

        form = QFormLayout()

        self._title_input = QLineEdit()
        self._title_input.setObjectName("experiment_title_input")
        self._title_input.setPlaceholderText("e.g. Hall bar A3 — SOT switching vs T")
        self._title_input.textChanged.connect(self._update_ok_enabled)
        self._title_input.textChanged.connect(self._on_title_changed)
        form.addRow("Title:", self._title_input)

        self._dirname_input = QLineEdit()
        self._dirname_input.setObjectName("experiment_dirname_input")
        self._dirname_input.setPlaceholderText("auto (from title)")
        self._dirname_input.setToolTip(
            "Optional — override where this experiment's folder lives inside "
            "the active session. Defaults to a name derived from the title."
        )
        self._dirname_input.textEdited.connect(self._on_dirname_edited)
        form.addRow("Folder name:", self._dirname_input)

        self._user_picker = UserPickerWidget(roster)
        self._user_picker.selection_changed_signal().connect(self._update_ok_enabled)
        form.addRow("User:", self._user_picker)

        self._attended_checkbox = QCheckBox("Human attending this experiment")
        self._attended_checkbox.setObjectName("start_attended_checkbox")
        self._attended_checkbox.setChecked(True)
        form.addRow("", self._attended_checkbox)

        self._envelope_editor: EnvelopeEditorWidget | None = None
        if envelope_variables:
            self._envelope_editor = EnvelopeEditorWidget(envelope_variables)
            self._envelope_editor.changed.connect(self._update_ok_enabled)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        if self._envelope_editor is not None:
            layout.addWidget(self._envelope_editor)
        layout.addWidget(buttons)

        self._update_ok_enabled()

    def _on_title_changed(self, text: str) -> None:
        """Auto-fill the folder name from the title until hand-edited."""
        if not self._dirname_edited_by_hand:
            self._dirname_input.setText(_slugify(text))

    def _on_dirname_edited(self, _text: str) -> None:
        self._dirname_edited_by_hand = True

    def _update_ok_enabled(self) -> None:
        envelope_ok = (
            self._envelope_editor is None or not self._envelope_editor.error()
        )
        self._ok_button.setEnabled(
            bool(self._title_input.text().strip())
            and self._user_picker.has_users()
            and envelope_ok
        )

    def envelope(self) -> ExperimentEnvelope | None:
        """Return the edited experiment envelope, or ``None`` for none.

        Only meaningful after accept.

        Returns:
            The envelope the operator narrowed to, or ``None`` when this
            dialog carries no editor, the editor is switched off, or no bound
            was entered.
        """
        if self._envelope_editor is None:
            return None
        return self._envelope_editor.envelope()

    def result_values(self) -> tuple[str, str, bool, str | None]:
        """Return ``(title, user_id, attended, experiment_dirname)``.

        Only meaningful after accept.

        Returns:
            The entered title, the selected user's roster id, the
            attendance checkbox state, and the folder name override (``None``
            when left empty — falls back to the auto-derived id).
        """
        return (
            self._title_input.text().strip(),
            self._user_picker.selected_user_id(),
            self._attended_checkbox.isChecked(),
            self._dirname_input.text().strip() or None,
        )


class CloseExperimentDialog(QDialog):
    """Collect the experiment's closing findings text."""

    def __init__(self, current_findings: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Close Experiment")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Findings (optional):"))

        self._findings_input = QTextEdit()
        self._findings_input.setObjectName("close_findings_input")
        self._findings_input.setPlainText(current_findings)
        layout.addWidget(self._findings_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def findings(self) -> str:
        """Return the entered findings text."""
        return self._findings_input.toPlainText()
