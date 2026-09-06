"""Unit tests for the Qt-free form-autosave model (i2as.gui.form_autosave).

These exercise FormAutosaveState/QueueItemState serialisation and the load/save
round-trip directly, with no QApplication — the model is deliberately stdlib
only so it can be tested at this level.
"""

from __future__ import annotations

import json
from pathlib import Path

from i2as.gui import form_autosave as session
from i2as.gui.form_autosave import FormAutosaveState, QueueItemState


# ── FormAutosaveState round-trip ───────────────────────────────────────────────────

def test_default_session_is_empty_with_default_data_dir():
    """A fresh FormAutosaveState has empty fields and the default data directory."""
    state = FormAutosaveState()
    assert state.sample_name == ""
    assert state.sample_id == ""
    assert state.comments == ""
    assert state.data_dir == ""
    assert state.selected_procedure == ""
    assert state.procedure_params == {}
    assert state.queue == []


def test_session_to_dict_from_dict_round_trip():
    """to_dict()/from_dict() preserve every field including nested queue items."""
    state = FormAutosaveState(
        sample_name="Si_001",
        sample_id="S2024-01",
        comments="cooldown 2",
        data_dir="D:/runs",
        selected_procedure="Field Sweep IV",
        procedure_params={"Field Sweep IV": {"field_start": -1.0, "field_steps": 11}},
        queue=[
            QueueItemState(
                procedure="Field Sweep IV",
                params={"field_start": -1.0},
                sample_info={"sample_name": "Si_001"},
                data_dir="D:/runs",
                status=session.STATUS_DONE,
            ),
            QueueItemState(procedure="Field Sweep DC", status=session.STATUS_PENDING),
        ],
    )
    restored = FormAutosaveState.from_dict(state.to_dict())
    assert restored == state


def test_to_dict_is_json_serialisable_and_versioned():
    """to_dict() yields JSON-serialisable data stamped with a schema version."""
    payload = FormAutosaveState(sample_name="x").to_dict()
    text = json.dumps(payload)  # must not raise
    assert json.loads(text)["version"] == session._SCHEMA_VERSION


# ── Defensive parsing ─────────────────────────────────────────────────────────

def test_from_dict_ignores_unknown_keys_and_fills_missing():
    """Unknown keys (newer version) are ignored; missing keys take defaults."""
    state = FormAutosaveState.from_dict(
        {"sample_name": "only_name", "future_field": 123}
    )
    assert state.sample_name == "only_name"
    assert state.data_dir == ""  # missing -> default (no explicit choice)
    assert state.queue == []


def test_from_dict_on_non_dict_returns_default():
    """A non-dict payload (e.g. a JSON list) degrades to a default state."""
    assert FormAutosaveState.from_dict([1, 2, 3]) == FormAutosaveState()
    assert FormAutosaveState.from_dict(None) == FormAutosaveState()


def test_queue_item_invalid_status_reset_to_pending():
    """An unrecognised status is coerced to 'pending', not trusted."""
    item = QueueItemState.from_dict({"procedure": "P", "status": "bogus"})
    assert item.status == session.STATUS_PENDING


def test_from_dict_wrong_typed_queue_becomes_empty():
    """A queue that is not a list is defensively replaced with an empty list."""
    state = FormAutosaveState.from_dict({"queue": "not-a-list"})
    assert state.queue == []


# ── load()/save() ─────────────────────────────────────────────────────────────

def test_load_missing_file_returns_default(tmp_path: Path):
    """Loading a non-existent path returns defaults without raising."""
    assert session.load(tmp_path / "nope.json") == FormAutosaveState()


def test_load_corrupt_json_returns_default(tmp_path: Path):
    """A file that is not valid JSON degrades to defaults without raising."""
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    assert session.load(bad) == FormAutosaveState()


def test_save_then_load_round_trip(tmp_path: Path):
    """save() then load() reproduces the original state."""
    path = tmp_path / "sub" / "last_session.json"  # parent does not exist yet
    state = FormAutosaveState(sample_name="Si_001", selected_procedure="Field Sweep IV")
    session.save(state, path)
    assert path.exists()
    assert session.load(path) == state


def test_save_creates_parent_directory(tmp_path: Path):
    """save() creates missing parent directories."""
    path = tmp_path / "a" / "b" / "c" / "session.json"
    session.save(FormAutosaveState(), path)
    assert path.exists()


def test_save_is_atomic_leaves_no_tmp_file(tmp_path: Path):
    """After a successful save, no leftover .tmp sidecar remains."""
    path = tmp_path / "session.json"
    session.save(FormAutosaveState(sample_name="x"), path)
    assert not (tmp_path / "session.json.tmp").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_overwrites_previous_session(tmp_path: Path):
    """A second save replaces the first session's content."""
    path = tmp_path / "session.json"
    session.save(FormAutosaveState(sample_name="first"), path)
    session.save(FormAutosaveState(sample_name="second"), path)
    assert session.load(path).sample_name == "second"
