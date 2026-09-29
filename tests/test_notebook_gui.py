"""Behaviour tests for the notebook GUI: the Settings page and the two dialogs.

The page is exercised over a real profile store and credential store (in a
temporary folder) and the real block catalog, so what it writes is what the
application reads; the dialogs over a stub service that answers at once.
"""

from __future__ import annotations

from typing import Any

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QLineEdit

from i2as.blocks.connector import ElnEntryRef
from i2as.gui.notebook_dialogs import MODE_CREATE, MODE_EXISTING, LinkNotebookDialog, ReadFieldsDialog
from i2as.gui.notebook_settings_page import NotebookSettingsPage
from i2as.session.credentials import CredentialStore, credential_key
from i2as.session.eln.publishing import BlockCatalog, PublishError
from i2as.session.models import ExperimentRecord
from i2as.session.user_profile import ElnAccount, UserProfileStore


@pytest.fixture
def stores(tmp_path):
    creds = CredentialStore(fallback_path=tmp_path / "c.json", environ={}, detect=False)
    profiles = UserProfileStore(tmp_path / "users", creds, tmp_path / "none.json")
    return profiles, creds, BlockCatalog(tmp_path / "blocks")


class StubService:
    """Answers every notebook question at once, recording what was asked."""

    def __init__(self, catalog: Any, accounts=(ElnAccount(account_id="lab", connector="sim", label="Lab"),)) -> None:
        self._catalog = catalog
        self._accounts = tuple(accounts)
        self.verified: list[str] = []
        self.linked: list[dict[str, Any]] = []
        self.reloaded = 0
        self.refuse_link = ""
        self.fields: Any = {}

    def catalog(self):
        return self._catalog

    def accounts(self, _user=""):
        return self._accounts

    def reload(self):
        self.reloaded += 1

    def verify(self, account_id, callback, user_id=""):
        self.verified.append(account_id)
        callback({"name": "A Sen", "team": "Cryo"})

    def list_templates(self, account_id, callback, user_id=""):
        callback([{"template_id": "12", "name": "Transport"}])

    def search(self, account_id, text, kind, callback, user_id=""):
        callback([{"ref": {"kind": kind, "record_id": "5"}, "title": f"{text} hit", "url": "u5", "category": "Sample"}])

    def link_experiment(self, account_id, profile_id="", *, entry=None, new_title="", template_id="", items=None):
        if self.refuse_link:
            raise PublishError(self.refuse_link)
        self.linked.append({"account": account_id, "profile": profile_id, "entry": entry, "title": new_title, "items": list(items or [])})

    def read_fields(self, callback):
        callback(self.fields)


# ── The Electronic notebook settings page ─────────────────────────────────


def test_the_connector_form_is_rendered_from_its_settings_schema(stores, qtbot):
    profiles, creds, catalog = stores
    page = NotebookSettingsPage(profiles, creds, "jdoe", catalog)
    qtbot.addWidget(page)
    page._connector.setCurrentIndex(page._connector.findData("elabftw"))
    assert isinstance(page.findChild(QLineEdit, "settings_eln_field_base_url"), QLineEdit)
    assert isinstance(page.findChild(QCheckBox, "settings_eln_field_verify_tls"), QCheckBox)
    page._connector.setCurrentIndex(page._connector.findData("sim"))
    assert page.findChild(QLineEdit, "settings_eln_field_base_url") is None or not page.findChild(QLineEdit, "settings_eln_field_base_url").isVisible()
    assert "state_file" in page.connector_settings()


def test_saving_writes_the_users_profile_and_keeps_the_key_in_the_credential_store(stores, qtbot):
    profiles, creds, catalog = stores
    service = StubService(catalog)
    page = NotebookSettingsPage(profiles, creds, "jdoe", catalog, service)
    qtbot.addWidget(page)
    page._enabled.setChecked(True)
    page._connector.setCurrentIndex(page._connector.findData("elabftw"))
    page.findChild(QLineEdit, "settings_eln_field_base_url").setText("https://elab.example.org")
    page._key.setText("top-secret")
    assert page.save() is True

    saved = profiles.load("jdoe").eln
    assert saved.enabled and saved.account().connector == "elabftw"
    assert saved.account().settings["base_url"] == "https://elab.example.org"
    assert creds.get(credential_key("eln", saved.default_account, "jdoe")) == "top-secret"
    assert page._key.text() == "", "the key is never kept in the form"
    assert "top-secret" not in (profiles.root / "jdoe" / "profile.yaml").read_text(encoding="utf-8")
    assert "A key is stored" in page._key_label.text()
    assert service.reloaded == 1


def test_a_blank_key_field_keeps_the_stored_key(stores, qtbot):
    profiles, creds, catalog = stores
    creds.set(credential_key("eln", "lab", "jdoe"), "kept")
    page = NotebookSettingsPage(profiles, creds, "jdoe", catalog)
    qtbot.addWidget(page)
    assert page._key.text() == "" and page._key.placeholderText()
    page.save()
    assert creds.get(credential_key("eln", "lab", "jdoe")) == "kept"


def test_test_connection_saves_then_asks_the_service(stores, qtbot):
    profiles, creds, catalog = stores
    service = StubService(catalog)
    page = NotebookSettingsPage(profiles, creds, "jdoe", catalog, service)
    qtbot.addWidget(page)
    page.test_connection()
    assert service.verified == ["lab"]
    assert page._status.text() == "Connected as A Sen (Cryo)."
    page.fetch_templates()
    assert page._template.findData("12") >= 0


def test_the_installed_blocks_are_listed(stores, qtbot):
    profiles, creds, catalog = stores
    page = NotebookSettingsPage(profiles, creds, "jdoe", catalog)
    qtbot.addWidget(page)
    text = page._blocks_text.toPlainText()
    assert "elabftw" in text and "sim" in text and "default" in text


# ── Linking an experiment to its page ─────────────────────────────────────


def test_creating_a_new_page_with_a_linked_sample(stores, qtbot):
    _profiles, _creds, catalog = stores
    service = StubService(catalog)
    dialog = LinkNotebookDialog(service, ExperimentRecord(experiment_id="001_x", title="Hall A", user_id="jdoe"))
    qtbot.addWidget(dialog)
    assert dialog.mode() == MODE_CREATE and dialog._title.text() == "Hall A"
    dialog._item_search.setText("S-001")
    dialog._item_search_btn.click()
    dialog._items.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.accept()
    [linked] = service.linked
    assert linked["entry"] is None and linked["title"] == "Hall A"
    assert [(i.item_id, i.role) for i in linked["items"]] == [("5", "sample")]


def test_linking_an_existing_page(stores, qtbot):
    _profiles, _creds, catalog = stores
    service = StubService(catalog)
    dialog = LinkNotebookDialog(service, ExperimentRecord(experiment_id="001_x", title="Hall A", user_id="jdoe"))
    qtbot.addWidget(dialog)
    dialog._page_search.setText("Hall")
    dialog._page_search_btn.click()
    dialog._pages.setCurrentRow(0)
    assert dialog.mode() == MODE_EXISTING
    dialog.accept()
    assert service.linked[0]["entry"] == ElnEntryRef(entry_id="5", url="u5")


def test_not_now_links_nothing_and_a_refusal_keeps_the_dialog_open(stores, qtbot):
    _profiles, _creds, catalog = stores
    service = StubService(catalog)
    dialog = LinkNotebookDialog(service, ExperimentRecord(experiment_id="001_x", user_id="jdoe"))
    qtbot.addWidget(dialog)
    dialog._later.setChecked(True)
    dialog.accept()
    assert service.linked == []
    service.refuse_link = "the connector 'sim' is not usable"
    again = LinkNotebookDialog(service, ExperimentRecord(experiment_id="001_x", user_id="jdoe"))
    qtbot.addWidget(again)
    again.accept()
    assert again._status.text() == "the connector 'sim' is not usable"
    assert again.result() == 0


def test_without_an_account_only_not_now_is_offered(stores, qtbot):
    _profiles, _creds, catalog = stores
    dialog = LinkNotebookDialog(StubService(catalog, accounts=()), ExperimentRecord(experiment_id="001_x", user_id="jdoe"))
    qtbot.addWidget(dialog)
    assert dialog._later.isChecked() and not dialog._create.isEnabled()
    assert "Electronic notebook" in dialog._status.text()


# ── Reading fields back ───────────────────────────────────────────────────


class StubManager:
    def __init__(self):
        self.applied: list[tuple[dict, dict]] = []

    def apply_eln_fields(self, values, snapshot):
        self.applied.append((dict(values), dict(snapshot)))
        return True


def test_only_the_ticked_changed_values_are_applied(stores, qtbot):
    _profiles, _creds, catalog = stores
    service = StubService(catalog)
    service.fields = {
        "thickness": {"value": 4.2, "unit": "nm", "source": "sample:Thickness", "raw": "4.2", "error": "", "current": 3.0, "fetched_utc": "t"},
        "sample_id": {"value": "S-1", "unit": "", "source": "sample:Sample ID", "raw": "S-1", "error": "", "current": "S-1", "fetched_utc": "t"},
        "count": {"value": None, "unit": "", "source": "page:Count", "raw": "", "error": "nothing is linked as 'page'", "current": None, "fetched_utc": "t"},
    }
    manager = StubManager()
    dialog = ReadFieldsDialog(service, manager)
    qtbot.addWidget(dialog)
    assert dialog.chosen_values() == {"thickness": 4.2}, "a changed value is proposed; an unchanged one is not"
    dialog.accept()
    [(values, snapshot)] = manager.applied
    assert values == {"thickness": 4.2} and snapshot["thickness"]["source"] == "sample:Thickness"


def test_a_notebook_that_cannot_be_read_says_so(stores, qtbot):
    _profiles, _creds, catalog = stores
    service = StubService(catalog)
    service.fields = PublishError("the experiment is not linked to a notebook page")
    dialog = ReadFieldsDialog(service, StubManager())
    qtbot.addWidget(dialog)
    assert "not linked" in dialog._status.text() and not dialog._apply.isEnabled()
