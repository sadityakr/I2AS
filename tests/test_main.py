"""Tests for i2as.main's session-tier startup wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from i2as.main import _ensure_guest_user_registered, _resolve_active_session
from i2as.session.models import GUEST_USER_ID, GUEST_USER_NAME, User
from i2as.session.store import SessionStore, UserRoster, _write_json_atomic


def test_resolve_active_session_creates_bootstrap_when_none_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First-ever launch: a session is created under the root's sessions/ and activated."""
    from i2as.gui import app_settings

    monkeypatch.setattr(app_settings, "current_user_id", lambda: "jdoe")
    store = SessionStore(tmp_path)

    folder = _resolve_active_session(store)

    assert store.get_active() == folder
    assert folder.parent == (tmp_path / "sessions").resolve()
    session = store.load(folder)
    assert session is not None
    assert session.user_id == "jdoe"
    assert session.name == "jdoe"


def test_resolve_active_session_falls_back_to_guest_when_logged_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No logged-in user yet: the bootstrap session is owned/named by the Guest identity."""
    from i2as.gui import app_settings

    monkeypatch.setattr(app_settings, "current_user_id", lambda: None)
    store = SessionStore(tmp_path)

    session = store.load(_resolve_active_session(store))

    assert session is not None
    assert session.name == GUEST_USER_ID
    assert session.user_id == GUEST_USER_ID


def test_resolve_active_session_returns_the_folder_the_operator_chose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An active session anywhere on disk is returned unchanged — nothing new is created."""
    from i2as.gui import app_settings

    monkeypatch.setattr(app_settings, "current_user_id", lambda: "jdoe")
    store = SessionStore(tmp_path / "root")
    chosen = tmp_path / "D" / "Cooldown 3"
    store.create_session(chosen, "Cooldown 3", "jdoe")
    store.set_active(chosen)

    assert _resolve_active_session(store) == chosen.resolve()
    assert not (tmp_path / "root" / "sessions").exists()


@pytest.mark.parametrize("registry", [{"active": "/does/not/exist"}, ["junk"], "not json"])
def test_resolve_active_session_recovers_from_a_bad_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: object
) -> None:
    """A registry naming a gone folder, or a corrupt one, is replaced, never fatal."""
    from i2as.gui import app_settings

    monkeypatch.setattr(app_settings, "current_user_id", lambda: "jdoe")
    store = SessionStore(tmp_path)
    if isinstance(registry, str):
        store.registry_path.write_text(registry, encoding="utf-8")
    else:
        _write_json_atomic(store.registry_path, registry)

    folder = _resolve_active_session(store)

    assert store.get_active() == folder
    assert store.load(folder) is not None


def test_ensure_guest_user_registered_adds_guest_once(tmp_path: Path) -> None:
    roster = UserRoster(tmp_path / "users.json")
    assert roster.get(GUEST_USER_ID) is None

    _ensure_guest_user_registered(roster)

    guest = roster.get(GUEST_USER_ID)
    assert guest is not None
    assert guest.name == GUEST_USER_NAME


def test_ensure_guest_user_registered_does_not_clobber_existing_guest_customization(
    tmp_path: Path,
) -> None:
    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id=GUEST_USER_ID, name="Visiting Scientist"))

    _ensure_guest_user_registered(roster)

    assert roster.get(GUEST_USER_ID).name == "Visiting Scientist"


def _shipped() -> Path:
    from i2as.gui import app_settings

    return app_settings.shipped_config_dir()


def test_command_line_config_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare ``i2as`` parses to no config; ``--config NAME`` parses to the name."""
    from i2as.main import build_parser

    assert build_parser().parse_args([]).config is None
    assert build_parser().parse_args(["--config", "sim_imaging"]).config == "sim_imaging"


def test_startup_candidates_put_the_command_line_config_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--config`` goes ahead of the saved active config; sim_cryostat stays last."""
    from i2as import main as app_main
    from i2as.gui import app_settings

    monkeypatch.setattr(app_settings, "user_config_dir", lambda: tmp_path / "user")
    monkeypatch.setattr(app_settings, "config_active", lambda: ("sim_cryostat", "shipped"))

    candidates = app_main._startup_candidates("sim_imaging")

    assert candidates == [
        str(_shipped() / "sim_imaging"),
        str(_shipped() / "sim_cryostat"),
    ]
    assert app_main._startup_candidates(None) == [str(_shipped() / "sim_cryostat")]


def test_startup_candidates_chain_a_user_copy_then_the_active_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user copy named on the command line comes first, then the saved active
    config, then sim_cryostat; a copy that kept its shipped name is followed by
    that shipped baseline, exactly as an active user config is."""
    from i2as import main as app_main
    from i2as.core.config_catalog import ConfigCatalog
    from i2as.gui import app_settings

    monkeypatch.setattr(app_settings, "user_config_dir", lambda: tmp_path / "user")
    catalog = ConfigCatalog(_shipped(), tmp_path / "user")
    mine = catalog.fork_shipped("sim_cryostat", "mine")
    same_name = catalog.fork_shipped("sim_imaging", "sim_imaging")
    monkeypatch.setattr(app_settings, "config_active", lambda: ("sim_imaging", "shipped"))

    assert app_main.resolve_config_name("mine") == (mine.path, "user")
    assert app_main._startup_candidates("mine") == [
        str(mine.path),
        str(_shipped() / "sim_imaging"),
        str(_shipped() / "sim_cryostat"),
    ]
    # Shipped configs are searched first, as the doctor CLI's --config does, so
    # a same-named user copy is reached through the active config's identity.
    assert app_main.resolve_config_name("sim_imaging") == (
        _shipped() / "sim_imaging",
        "shipped",
    )
    monkeypatch.setattr(app_settings, "config_active", lambda: ("sim_imaging", "user"))
    assert app_main._startup_candidates(None) == [
        str(same_name.path),
        str(_shipped() / "sim_imaging"),
        str(_shipped() / "sim_cryostat"),
    ]


def test_unknown_command_line_config_is_a_usage_error_before_any_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A name that exists nowhere exits 2 with the places searched, and builds nothing."""
    from i2as import main as app_main
    from i2as.gui import app_settings

    monkeypatch.setattr(app_settings, "user_config_dir", lambda: tmp_path / "user")
    with pytest.raises(LookupError):
        app_main._startup_candidates("no_such_config")
    with pytest.raises(SystemExit) as exit_info:
        app_main.main(["--config", "no_such_config"])
    assert exit_info.value.code == 2
    assert "no_such_config" in capsys.readouterr().err
