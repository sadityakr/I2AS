"""The Settings dialog, its Analysis page, and the general settings file's migration.

The Connections page's own behaviour is covered in ``test_mcp_http_server.py``
against a live gateway controller; this file covers what is new with the
dialog: the page sidebar, the Analysis page's refusal to switch analysis on
without a container engine, and the one-time copy of the old settings (the
QSettings connection keys, the ``eln-settings.json`` analysis block) into
``settings.json``. The engine check is injected everywhere: no test here
needs Docker.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QCheckBox, QLabel, QLineEdit, QListWidget, QPushButton

from i2as.gui import app_settings
from i2as.gui.analysis_settings_page import NOT_CHECKED_TEXT, AnalysisSettingsPage
from i2as.gui.settings_dialog import (
    NO_GATEWAY_TEXT,
    PAGE_ANALYSIS,
    PAGE_CONNECTIONS,
    SettingsDialog,
)
from i2as.session.analysis_sandbox import EngineStatus
from i2as.session.app_config import (
    AnalysisSettings,
    AppConfig,
    AppConfigStore,
    SandboxSettings,
    load_app_config,
)

READY = EngineStatus(True, True, True, "Ready: docker 27, image 'i2as-analysis:latest'.")
NO_ENGINE = EngineStatus(detail="'docker' is not installed or not on PATH.")


@pytest.fixture
def store(tmp_path) -> AppConfigStore:
    """A store over a throwaway file, starting from the defaults."""
    return AppConfigStore(tmp_path / "settings.json", config=AppConfig())


@pytest.fixture
def isolated_qsettings(tmp_path, monkeypatch):
    """Redirect the QSettings factory to a throwaway INI file."""
    ini_path = tmp_path / "i2as_test_settings.ini"
    monkeypatch.setattr(
        app_settings,
        "get_settings",
        lambda: QSettings(str(ini_path), QSettings.Format.IniFormat),
    )
    return ini_path


# ── The dialog ───────────────────────────────────────────────────────────


def test_the_dialog_lists_one_page_per_section(qtbot, store):
    dialog = SettingsDialog(store, engine_checker=lambda _s: READY)
    qtbot.addWidget(dialog)

    sidebar = dialog.findChild(QListWidget, "settings_page_list")
    titles = [sidebar.item(row).text() for row in range(sidebar.count())]

    assert titles == ["Connections", "Analysis"]
    assert sidebar.currentRow() == 0


def test_the_dialog_opens_on_the_page_asked_for(qtbot, store):
    dialog = SettingsDialog(store, page=PAGE_ANALYSIS, engine_checker=lambda _s: READY)
    qtbot.addWidget(dialog)

    assert dialog.findChild(QListWidget, "settings_page_list").currentRow() == 1
    assert isinstance(dialog.page(PAGE_ANALYSIS), AnalysisSettingsPage)


def test_without_a_gateway_the_connections_page_says_so(qtbot, store):
    dialog = SettingsDialog(store, page=PAGE_CONNECTIONS)
    qtbot.addWidget(dialog)

    label = dialog.findChild(QLabel, "settings_connections_unavailable")
    assert label is not None and label.text() == NO_GATEWAY_TEXT


# ── The Analysis page ────────────────────────────────────────────────────


def _page(qtbot, store, checker=lambda _s: READY, saved=None) -> AnalysisSettingsPage:
    page = AnalysisSettingsPage(
        store,
        engine_checker=checker,
        on_saved=(saved.append if saved is not None else None),
    )
    qtbot.addWidget(page)
    return page


def test_the_page_shows_the_stored_section(qtbot, store):
    store.save(
        replace(
            store.current,
            analysis=AnalysisSettings(
                timeout_s=90.0, sandbox=SandboxSettings(engine="podman", image="lab/a:1")
            ),
        )
    )
    page = _page(qtbot, store)

    assert page.settings_from_form() == store.analysis()
    assert page.findChild(QLabel, "settings_analysis_engine_label").text() == NOT_CHECKED_TEXT


def test_saving_writes_the_file_and_carries_unshown_fields(qtbot, store):
    """The per-procedure recipe preference, edited nowhere here, survives a save."""
    store.save(
        replace(store.current, analysis=AnalysisSettings(recipes={"FieldSweep": "mr"}))
    )
    saved: list[AnalysisSettings] = []
    page = _page(qtbot, store, saved=saved)
    page.findChild(QLineEdit, "settings_analysis_image_edit").setText("lab/analysis:2")
    page.findChild(QCheckBox, "settings_analysis_enabled_checkbox").setChecked(True)

    page.findChild(QPushButton, "settings_analysis_save_btn").click()

    on_disk = load_app_config(store.path).analysis
    assert on_disk.enabled is True
    assert on_disk.sandbox.image == "lab/analysis:2"
    assert on_disk.recipes == {"FieldSweep": "mr"}
    assert saved == [on_disk]


def test_analysis_cannot_be_switched_on_without_an_engine(qtbot, store):
    page = _page(qtbot, store, checker=lambda _s: NO_ENGINE)
    page.findChild(QCheckBox, "settings_analysis_enabled_checkbox").setChecked(True)

    assert page.save() is False

    assert store.analysis().enabled is False
    status = page.findChild(QLabel, "settings_analysis_status_label").text()
    assert "not installed" in status and "Not saved" in status


def test_switching_analysis_off_needs_no_engine(qtbot, store):
    """Only turning it ON is gated: a person can always switch analysis off."""
    store.save(replace(store.current, analysis=AnalysisSettings(enabled=True)))
    asked: list[SandboxSettings] = []
    page = _page(qtbot, store, checker=lambda s: asked.append(s) or NO_ENGINE)
    page.findChild(QCheckBox, "settings_analysis_enabled_checkbox").setChecked(False)

    assert page.save() is True

    assert store.analysis().enabled is False and asked == []


def test_check_asks_about_the_form_not_the_file(qtbot, store):
    asked: list[SandboxSettings] = []
    page = _page(qtbot, store, checker=lambda s: asked.append(s) or READY)
    page.findChild(QLineEdit, "settings_analysis_image_edit").setText("lab/new:3")

    page.findChild(QPushButton, "settings_analysis_check_btn").click()

    assert asked[0].image == "lab/new:3"
    assert page.findChild(QLabel, "settings_analysis_engine_label").text() == READY.detail


# ── Migration into settings.json ─────────────────────────────────────────


def test_connection_keys_move_from_qsettings_into_the_file(isolated_qsettings, tmp_path):
    legacy = QSettings(str(isolated_qsettings), QSettings.Format.IniFormat)
    legacy.setValue("Gateway/enabled", True)
    legacy.setValue("Gateway/max_role", "analyst")
    legacy.setValue("RemoteAccess/enabled", True)
    legacy.setValue("RemoteAccess/port", 9123)
    legacy.sync()

    assert app_settings.gateway_enabled() is True
    assert app_settings.gateway_max_role() == "analyst"
    assert app_settings.remote_access_port() == 9123

    on_disk = json.loads(app_settings.config_store().path.read_text(encoding="utf-8"))
    assert on_disk["connections"]["gateway_max_role"] == "analyst"
    assert on_disk["connections"]["remote_port"] == 9123


def test_the_file_wins_over_leftover_qsettings_keys(isolated_qsettings):
    legacy = QSettings(str(isolated_qsettings), QSettings.Format.IniFormat)
    legacy.setValue("Gateway/max_role", "analyst")
    legacy.sync()
    path = app_settings.config_store().path
    app_settings._CONFIG_STORE = None
    path.write_text(json.dumps({"connections": {"gateway_max_role": "observer"}}), encoding="utf-8")

    assert app_settings.gateway_max_role() == "observer"


def test_a_fresh_install_writes_nothing_until_something_is_saved(isolated_qsettings):
    store = app_settings.config_store()

    assert not store.path.exists()
    assert app_settings.gateway_enabled() is None, "monitor.yaml still decides"
    app_settings.set_gateway_enabled(True)
    assert store.path.exists() and app_settings.gateway_enabled() is True


def test_the_eln_analysis_block_moves_into_the_file(isolated_qsettings, tmp_path, monkeypatch):
    eln = tmp_path / "legacy-eln.json"
    eln.write_text(json.dumps({"analysis": {"enabled": True, "timeout_s": 60}}), encoding="utf-8")
    monkeypatch.setenv("I2AS_ELN_SETTINGS", str(eln))

    store = app_settings.config_store()

    assert store.analysis().enabled is True and store.analysis().timeout_s == 60.0
    on_disk = json.loads(store.path.read_text(encoding="utf-8"))
    assert on_disk["analysis"]["enabled"] is True, "written once, so the copy is permanent"
