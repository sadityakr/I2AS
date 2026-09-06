# ---
# description: |
#   Tests for i2as.session.maintenance_log — the maintenance-log
#   framework. Covers LogKindSpec validation, the declared `maintenance`
#   kind, MaintenanceLogStore's entry-revision model (add/revise/delete/
#   revisions), write validation, and tolerant loads.
# last_updated: 2026-09-05
# ---

import logging

import pytest

from i2as.core.plan import ParamSpec
from i2as.session.maintenance_log import (
    DECLARED_LOG_KINDS,
    LogKindSpec,
    MaintenanceLogStore,
)

CONFIG_NAME = "sim_cryostat"


@pytest.fixture
def store(tmp_path):
    return MaintenanceLogStore(tmp_path / "maintenance", CONFIG_NAME)


# ── LogKindSpec validation ──────────────────────────────────────────────────


def test_declared_maintenance_kind_shape():
    """The one shipped kind is editable and carries the four neutral fields."""
    spec = DECLARED_LOG_KINDS["maintenance"]
    assert spec.editable is True
    assert set(spec.fields) == {"performed_utc", "person", "action", "notes"}
    assert all(isinstance(field, ParamSpec) for field in spec.fields.values())


def test_log_kind_spec_rejects_bad_key():
    with pytest.raises(ValueError, match="lowercase identifier"):
        LogKindSpec(key="Not Valid", title="X", fields={"a": ParamSpec(type=str, default="")})
    with pytest.raises(ValueError, match="lowercase identifier"):
        LogKindSpec(key="", title="X", fields={"a": ParamSpec(type=str, default="")})


def test_log_kind_spec_rejects_empty_title():
    with pytest.raises(ValueError, match="title"):
        LogKindSpec(key="x", title="", fields={"a": ParamSpec(type=str, default="")})


def test_log_kind_spec_rejects_empty_fields():
    with pytest.raises(ValueError, match="fields"):
        LogKindSpec(key="x", title="X", fields={})


def test_log_kind_spec_rejects_non_paramspec_field():
    with pytest.raises(TypeError, match="ParamSpec"):
        LogKindSpec(key="x", title="X", fields={"a": "not a paramspec"})


def test_log_kind_spec_fields_defensively_copied():
    fields = {"a": ParamSpec(type=str, default="")}
    spec = LogKindSpec(key="x", title="X", fields=fields)
    fields["b"] = ParamSpec(type=int, default=0)
    assert "b" not in spec.fields


# ── MaintenanceLogStore: add / revise / delete round-trip ───────────────────


def _values(**overrides):
    values = {
        "performed_utc": "2026-07-19T10:00:00+00:00",
        "person": "jdoe",
        "action": "Replaced the o-ring",
        "notes": "",
    }
    values.update(overrides)
    return values


def test_add_entry_round_trip(store):
    entry = store.add_entry("maintenance", _values())
    assert entry.revision == 1
    assert entry.source == "manual"
    assert entry.entry_id
    assert entry.values["person"] == "jdoe"
    assert entry.values["action"] == "Replaced the o-ring"

    fetched = store.entries("maintenance")
    assert len(fetched) == 1
    assert fetched[0] == entry


def test_add_entry_fills_missing_fields_with_defaults(store):
    entry = store.add_entry("maintenance", {"person": "asmith"})
    assert entry.values["notes"] == ""
    assert entry.values["action"] == ""
    assert entry.values["performed_utc"] == ""


def test_add_entry_person_kwarg_folds_into_values(store):
    entry = store.add_entry("maintenance", {}, person="operator-1")
    assert entry.values["person"] == "operator-1"

    # Explicit values["person"] wins over the kwarg.
    entry2 = store.add_entry("maintenance", {"person": "explicit"}, person="operator-1")
    assert entry2.values["person"] == "explicit"


def test_revise_entry_preserves_history_and_updates_latest(store):
    original = store.add_entry("maintenance", _values(notes="typo"))
    revised = store.revise_entry(
        "maintenance", original.entry_id, {"notes": "corrected"}, revised_by="tech1"
    )

    assert revised.entry_id == original.entry_id
    assert revised.revision == 2
    assert revised.values["notes"] == "corrected"
    # Untouched fields carry forward from the previous revision.
    assert revised.values["person"] == "jdoe"
    assert revised.revised_by == "tech1"
    assert revised.created_utc == original.created_utc

    latest = store.entries("maintenance")
    assert len(latest) == 1
    assert latest[0].values["notes"] == "corrected"

    history = store.revisions("maintenance", original.entry_id)
    assert [e.revision for e in history] == [1, 2]
    assert history[0].values["notes"] == "typo"
    assert history[1].values["notes"] == "corrected"


def test_delete_entry_tombstones_and_hides_from_entries(store):
    entry = store.add_entry("maintenance", _values())
    store.add_entry("maintenance", _values(person="other"))
    assert len(store.entries("maintenance")) == 2

    tombstone = store.delete_entry("maintenance", entry.entry_id, revised_by="tech1")
    assert tombstone.deleted is True
    assert tombstone.revision == 2

    remaining = store.entries("maintenance")
    assert len(remaining) == 1
    assert remaining[0].values["person"] == "other"

    # Full history is still inspectable.
    history = store.revisions("maintenance", entry.entry_id)
    assert len(history) == 2
    assert history[-1].deleted is True


def test_entries_newest_first_by_created_time(store):
    first = store.add_entry("maintenance", _values(notes="first"))
    second = store.add_entry("maintenance", _values(notes="second"))
    # Revising the first entry must not change its created_utc / ordering.
    store.revise_entry(
        "maintenance", first.entry_id, {"notes": "first-edited"}, revised_by="tech"
    )

    entries = store.entries("maintenance")
    assert [e.values["notes"] for e in entries] == ["second", "first-edited"]
    assert entries[0].entry_id == second.entry_id


def test_revise_unknown_entry_raises(store):
    with pytest.raises(ValueError, match="no entry"):
        store.revise_entry("maintenance", "nope", {}, revised_by="tech")


def test_delete_unknown_entry_raises(store):
    with pytest.raises(ValueError, match="no entry"):
        store.delete_entry("maintenance", "nope", revised_by="tech")


# ── Write validation ─────────────────────────────────────────────────────────


def test_add_entry_rejects_unknown_field(store):
    with pytest.raises(ValueError, match="no field"):
        store.add_entry("maintenance", {"not_a_field": 1})


def test_add_entry_rejects_wrong_type(store):
    with pytest.raises(ValueError):
        store.add_entry("maintenance", {"action": 3})
    with pytest.raises(ValueError):
        store.add_entry("maintenance", {"notes": True})


def test_add_entry_unknown_kind_raises(store):
    with pytest.raises(ValueError, match="unknown log kind"):
        store.add_entry("bogus_kind", {})


def test_a_non_editable_kind_refuses_every_write(store, monkeypatch):
    """`editable=False` marks a read-only stream this layer never writes."""
    spec = LogKindSpec(
        key="readonly",
        title="Read only",
        fields={"note": ParamSpec(type=str, default="")},
        editable=False,
    )
    monkeypatch.setitem(DECLARED_LOG_KINDS, "readonly", spec)

    with pytest.raises(ValueError, match="not editable"):
        store.add_entry("readonly", {"note": "x"})
    with pytest.raises(ValueError, match="not editable"):
        store.revise_entry("readonly", "id", {}, revised_by="x")
    with pytest.raises(ValueError, match="not editable"):
        store.delete_entry("readonly", "id", revised_by="x")


# ── Tolerant loads ───────────────────────────────────────────────────────────


def test_entries_tolerates_corrupt_line(store, caplog):
    good = store.add_entry("maintenance", _values())
    path = store._path("maintenance")
    with path.open("a", encoding="utf-8") as f:
        f.write("{not valid json\n")

    with caplog.at_level(logging.WARNING):
        entries = store.entries("maintenance")
    assert len(entries) == 1
    assert entries[0].entry_id == good.entry_id
    assert any("corrupt" in record.message for record in caplog.records)
