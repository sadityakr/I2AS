# ---
# description: |
#   Behaviour tests for the eLab setup dialog (gui/eln_settings_dialog.py):
#   the form round-trips the settings and preserves every field it never
#   shows, a blank API-key field keeps the stored key and a typed one
#   replaces it, Test connection reports the backend's identity (and its
#   refusal), Fetch templates fills the combo, and Save hands the edited
#   record to on_save.
# last_updated: 2026-09-05
# ---

"""The eLab setup dialog, over the sim ELN adapter.

``SimElnAdapter`` is the ELN adapter standard's in-memory twin — a healthy
notebook, and, with ``sim_offline``, one that refuses every call — so nothing
here needs a network or a real notebook account.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from PyQt6.QtWidgets import QLineEdit

from i2as.gui.eln_settings_dialog import KEY_PLACEHOLDER, ElnSettingsDialog
from i2as.session.eln.settings import AnalysisSettings, ElnSettings
from i2as.session.eln.sim_eln import SimElnAdapter


#: The real analysis block — the stand-in from the parallel build is gone.
StubAnalysisSettings = AnalysisSettings


#: ``ElnSettings`` carries the analysis block itself now.
SettingsWithAnalysis = ElnSettings


class OfflineSimAdapter(SimElnAdapter):
    """The sim notebook with the lab network down."""

    backend = "sim_offline"

    def __init__(self, settings: Any) -> None:
        """Build the sim adapter forced offline.

        Args:
            settings: The settings mapping, as the adapter standard requires.
        """
        super().__init__({**dict(settings or {}), "sim_offline": True})


@pytest.fixture
def settings() -> SettingsWithAnalysis:
    """A configured record with a stored key and a non-default assistant."""
    base = SettingsWithAnalysis(
        enabled=True,
        backend="sim_eln",
        base_url="https://elab.example.org",
        api_key="stored-secret",
        team_id="7",
        template_id="1",
        tags=("i2as", "hall"),
        max_attachment_bytes=25 * 1024 * 1024,
    )
    return replace(base, assistant=replace(base.assistant, model="claude-sonnet-5"))


@pytest.fixture
def dialog(settings, qtbot):
    """The dialog over those settings, with the sim adapters injected.

    Returns:
        ``(dialog, saved)`` where ``saved`` collects what Save hands over.
    """
    saved: list[Any] = []
    widget = ElnSettingsDialog(
        settings,
        on_save=saved.append,
        adapters={"sim_eln": SimElnAdapter, "sim_offline": OfflineSimAdapter},
    )
    qtbot.addWidget(widget)
    return widget, saved


# ── The form ──────────────────────────────────────────────────────────────────


def test_form_shows_the_settings_but_never_the_key(dialog, settings):
    """Every field is filled from the record — except the key, ever."""
    widget, _saved = dialog
    assert widget._base_url_input.text() == settings.base_url
    assert widget._team_id_input.text() == "7"
    assert widget._tags_input.text() == "i2as, hall"
    assert widget._attachment_cap_input.value() == 25
    assert widget._api_key_input.text() == ""
    assert widget._api_key_input.echoMode() == QLineEdit.EchoMode.Password
    assert widget._api_key_input.placeholderText() == KEY_PLACEHOLDER
    # The key appears in no label, no tooltip and no window text.
    rendered = " ".join(
        [
            widget._api_key_input.toolTip(),
            widget._status_label.text(),
            widget.windowTitle(),
        ]
    )
    assert settings.api_key not in rendered


def test_round_trip_preserves_unshown_fields(dialog, settings):
    """A save carries every field the dialog never displayed."""
    widget, _saved = dialog
    edited = widget.settings_from_form()
    assert edited.assistant.model == "claude-sonnet-5"
    assert edited.assistant.prices == settings.assistant.prices
    assert edited.retry_base_s == settings.retry_base_s
    assert edited.drain_interval_s == settings.drain_interval_s
    assert edited.timeout_s == settings.timeout_s


def test_blank_key_keeps_the_stored_key(dialog):
    """Leaving the key field alone keeps the key that is already stored."""
    widget, _saved = dialog
    widget._base_url_input.setText("https://elsewhere.example.org/")
    edited = widget.settings_from_form()
    assert edited.api_key == "stored-secret"
    assert edited.base_url == "https://elsewhere.example.org"


def test_a_typed_key_replaces_the_stored_one(dialog):
    """A key typed into the field is the key that is saved."""
    widget, _saved = dialog
    widget._api_key_input.setText("  new-secret  ")
    assert widget.settings_from_form().api_key == "new-secret"


def test_the_form_edits_every_shown_field(dialog):
    """Each control is read back into the record it produces."""
    widget, _saved = dialog
    widget._enabled_checkbox.setChecked(False)
    widget._verify_tls_checkbox.setChecked(False)
    widget._auto_publish_checkbox.setChecked(False)
    widget._tags_input.setText("a, b ,, c")
    widget._attachment_cap_input.setValue(3)
    widget._team_id_input.setText("9")
    edited = widget.settings_from_form()
    assert edited.enabled is False
    assert edited.verify_tls is False
    assert edited.auto_publish is False
    assert edited.tags == ("a", "b", "c")
    assert edited.max_attachment_bytes == 3 * 1024 * 1024
    assert edited.team_id == "9"


def test_the_analysis_group_round_trips(dialog):
    """The Analysis group edits the analysis block and nothing else."""
    widget, _saved = dialog
    widget._analysis_enabled_checkbox.setChecked(True)
    widget._analysis_timeout_input.setValue(45)
    widget._analysis_facts_checkbox.setChecked(True)
    widget._analysis_attach_checkbox.setChecked(True)
    analysis = widget.settings_from_form().analysis
    assert analysis.enabled is True
    assert analysis.timeout_s == 45.0
    assert analysis.include_fact_tables is True
    assert analysis.attach_data_file is True



# ── The backend calls ─────────────────────────────────────────────────────────


def test_test_connection_shows_the_backends_identity(dialog):
    """A healthy backend's identity string lands in the status line."""
    widget, _saved = dialog
    assert widget.test_connection() == "sim ELN (in-memory)"
    assert "sim ELN (in-memory)" in widget._status_label.text()


def test_test_connection_shows_the_failure(dialog):
    """An unreachable backend's own message lands in the status line."""
    widget, _saved = dialog
    widget._backend_combo.setCurrentIndex(widget._backend_combo.findData("sim_offline"))
    assert widget.test_connection() == ""
    assert "offline" in widget._status_label.text()


def test_fetch_templates_fills_the_combo(dialog):
    """The backend's templates become the combo's items, current one kept."""
    widget, _saved = dialog
    assert widget.fetch_templates() == 2
    assert widget._template_combo.count() == 2
    assert widget._template_combo.currentData() == "1"
    assert widget.settings_from_form().template_id == "1"


def test_fetch_templates_reports_a_refusal(dialog):
    """A refused listing leaves the combo alone and says why."""
    widget, _saved = dialog
    widget._backend_combo.setCurrentIndex(widget._backend_combo.findData("sim_offline"))
    assert widget.fetch_templates() == 0
    assert "Could not list templates" in widget._status_label.text()


def test_an_unknown_backend_is_a_status_line(qtbot, settings):
    """No adapter for the selected backend is reported, never raised."""
    widget = ElnSettingsDialog(settings, on_save=lambda _s: None, adapters={})
    qtbot.addWidget(widget)
    assert widget.test_connection() == ""
    assert "No adapter" in widget._status_label.text()


# ── Save ──────────────────────────────────────────────────────────────────────


def test_save_hands_the_edited_record_over_and_accepts(dialog):
    """Save calls on_save with the form's record, then closes the dialog."""
    widget, saved = dialog
    widget._api_key_input.setText("new-secret")
    widget._save_btn.click()
    assert len(saved) == 1
    assert saved[0].api_key == "new-secret"
    assert widget.result() == int(widget.DialogCode.Accepted)


def test_a_failing_save_keeps_the_dialog_open(qtbot, settings):
    """A save that raises is reported in the status line, not into Qt."""

    def _explode(_edited: Any) -> None:
        """Refuse to save.

        Args:
            _edited: The record the dialog produced.

        Raises:
            OSError: Always.
        """
        raise OSError("disk full")

    widget = ElnSettingsDialog(settings, on_save=_explode, adapters={})
    qtbot.addWidget(widget)
    widget._save_btn.click()
    assert widget.isVisible() is False  # never shown, but also never accepted
    assert widget.result() != int(widget.DialogCode.Accepted)
    assert "Saving failed" in widget._status_label.text()
