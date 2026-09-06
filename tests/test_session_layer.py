import json
import shutil

import h5py
import pytest

from i2as.core import events as ev
from i2as.core.orchestrator import Orchestrator
from i2as.core.plan import EnvelopeBound, ExperimentEnvelope, params_digest
from i2as.core.station import build_station
from i2as.procedures.field_sweep import FieldSweep
from i2as.session.manager import ExperimentManager
from i2as.session.models import (
    EXPERIMENT_STATUS_CLOSED,
    EXPERIMENT_STATUS_OPEN,
    RUN_STATUS_DONE,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    SCHEMA_VERSION,
    ElnLink,
    ExperimentIndexEntry,
    ExperimentRecord,
    RunRecord,
    Session,
    User,
    envelope_from_dict,
    envelope_to_dict,
)
from i2as.session.store import (
    ExperimentStore,
    SessionStore,
    UserRoster,
    _write_json_atomic,
)

CONFIG_PATH = "i2as/configs/sim_cryostat"

SAMPLE_INFO = {"sample_name": "Hall bar A3", "sample_id": "A3", "comments": "test"}

FAST_PARAMS = {
    "measurement_vi": "dc_measurement",
    "field_start": -0.1,
    "field_end": 0.1,
    "field_steps": 3,
    "temperature": 300.0,
    "current_A": 1e-6,
    "readings_per_point": 5,
    "init_wait": 0.0,
    "step_wait": 0.0,
}


@pytest.fixture
def station():
    return build_station(CONFIG_PATH)


@pytest.fixture
def orchestrator(station, qtbot):
    return Orchestrator(station, tick_interval_ms=10)


def test_the_manager_wires_to_the_proxy_the_application_hands_it(
    store, roster, station, qtbot
):
    """The session layer is built against the client adapter, not the engine.

    ``main.py`` hands ``ExperimentManager`` an ``OrchestratorProxy``, which
    renames the engine's two contract channels (``event_emitted`` → ``event``,
    ``verdict_emitted`` → ``verdict``) because a client consumes them. Nothing
    else in this suite passes one, so without this the wiring is only
    exercised at launch — where a mismatch is a crash before the first window
    appears rather than a red test.
    """
    from i2as.core.instrument_host import InstrumentHost

    host = InstrumentHost(
        lambda: station, orchestrator_options={"tick_interval_ms": 50}
    )
    host.start()
    try:
        manager = ExperimentManager(
            store=store,
            roster=roster,
            orchestrator=host.build_proxy(),
            config_name="sim_cryostat",
        )
        assert manager.current_experiment() is None
    finally:
        host.shutdown()


@pytest.fixture
def store(tmp_path):
    return ExperimentStore(tmp_path / "experiments")


@pytest.fixture
def roster(tmp_path):
    r = UserRoster(tmp_path / "users.json")
    r.add(User(user_id="jdoe", name="J. Doe", email="jdoe@example.org"))
    return r


@pytest.fixture
def manager(store, roster, orchestrator, station):
    return ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
    )


@pytest.fixture
def indexed_manager(tmp_path, roster, orchestrator, station):
    """A manager wired to a real Session, for testing session-index maintenance.

    Returns a ``(manager, session_store, session)`` tuple — ``session_store``
    and ``session`` let a test reload ``session.json`` and inspect its
    ``experiments`` index after a lifecycle call.
    """
    session_store = SessionStore(tmp_path / "sessions")
    session = session_store.create_session("Lab A", "jdoe")
    exp_store = ExperimentStore(session_store.root / "jdoe" / session.session_id)
    exp_manager = ExperimentManager(
        store=exp_store,
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
        session_store=session_store,
    )
    return exp_manager, session_store, session


# ── Models ───────────────────────────────────────────────────────────────────

def test_experiment_record_round_trips_with_content():
    """A populated record survives to_dict()/from_dict() unchanged."""
    record = ExperimentRecord(
        experiment_id="20260717_test",
        title="Test",
        user_id="jdoe",
        sample_info=dict(SAMPLE_INFO),
        config_name="sim_cryostat",
        created_utc="2026-07-17T12:00:00+00:00",
        attended=False,
        envelope={"magnet_z": {"min_value": -2.0, "max_value": 2.0, "state_key": ""}},
        runs=[RunRecord(run_id="r1", procedure="Field Sweep", status=RUN_STATUS_DONE)],
        findings="looks superconducting",
        eln_link=ElnLink(backend="elabftw", entry_id="42", url="https://eln/42"),
        queue=[{"procedure": "Field Sweep", "params": {"field_end": 1.0}}],
    )
    assert record.schema_version == SCHEMA_VERSION
    payload = record.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["queue"] == record.queue
    assert ExperimentRecord.from_dict(payload) == record


def test_experiment_record_schema_version_absent_defaults_to_one():
    """A record written before schema_version existed loads as version 1."""
    record = ExperimentRecord.from_dict({"experiment_id": "x"})
    assert record.schema_version == 1


def test_experiment_record_schema_version_tolerates_future_value():
    """A record from a newer app loads (tolerant-parse) with its stated version kept."""
    record = ExperimentRecord.from_dict({"experiment_id": "x", "schema_version": 999})
    assert record.schema_version == 999


def test_experiment_record_queue_tolerates_junk():
    """Non-list/non-dict queue entries degrade to [] / are dropped, never raise."""
    assert ExperimentRecord.from_dict({"queue": "not-a-list"}).queue == []
    assert ExperimentRecord.from_dict({"queue": [{"a": 1}, "junk", 5]}).queue == [{"a": 1}]


def test_run_record_untrusted_status_degrades_to_failed():
    """An unknown status must not masquerade as live or successful work."""
    run = RunRecord.from_dict({"run_id": "r1", "status": "totally-bogus"})
    assert run.status == RUN_STATUS_FAILED


def test_experiment_record_untrusted_status_degrades_to_closed():
    """A record with an unknown status must not resume as the live experiment."""
    record = ExperimentRecord.from_dict({"experiment_id": "x", "status": "bogus"})
    assert record.status == EXPERIMENT_STATUS_CLOSED


def test_envelope_round_trip_and_junk_tolerance():
    """envelope_to_dict()/envelope_from_dict() round-trip; junk drops to None."""
    envelope = ExperimentEnvelope(
        bounds={
            "magnet_z": EnvelopeBound(min_value=-2.0, max_value=2.0),
            "temperature_sample": EnvelopeBound(min_value=4.0, state_key="temperature"),
        }
    )
    rebuilt = envelope_from_dict(envelope_to_dict(envelope))
    assert rebuilt == envelope
    assert envelope_to_dict(None) == {}
    assert envelope_from_dict({}) is None
    assert envelope_from_dict("junk") is None
    # Structurally dict-like but invalid bounds -> dropped with a warning.
    assert envelope_from_dict({"magnet_z": {"min_value": 5.0, "max_value": 1.0}}) is None


# ── ExperimentStore / UserRoster ─────────────────────────────────────────────

def test_store_creates_nothing_until_save(tmp_path):
    """Construction and reads must not create directories (lazy creation)."""
    root = tmp_path / "experiments"
    store = ExperimentStore(root)
    assert store.list_experiments() == []
    assert store.get_active() is None
    assert store.load("nope") is None
    assert not root.exists()


def test_store_save_load_list_and_active_pointer(store):
    record = ExperimentRecord(experiment_id="20260717_x", title="X")
    store.save(record)
    store.set_active("20260717_x")
    assert store.list_experiments() == ["20260717_x"]
    assert store.load("20260717_x") == record
    assert store.get_active() == "20260717_x"
    store.set_active(None)
    assert store.get_active() is None
    # No stray .tmp files after atomic writes.
    assert not list(store.root.rglob("*.tmp"))


def test_store_load_tolerates_corrupt_file(store):
    path = store.root / "bad" / "experiment.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert store.load("bad") is None
    assert "bad" in store.list_experiments()  # listed (folder exists) but unloadable


def test_store_make_experiment_id_slug_and_collisions(store):
    created = "2026-07-17T12:00:00+00:00"
    first = store.make_experiment_id("Hall bar A3 — SOT!", created)
    assert first == "20260717_hall_bar_a3_sot"
    store.save(ExperimentRecord(experiment_id=first))
    assert store.make_experiment_id("Hall bar A3 — SOT!", created) == f"{first}_2"


def test_store_load_warns_on_future_schema_version(store, caplog):
    """A record from a newer app still loads (tolerant), but logs a WARNING."""
    record = ExperimentRecord(experiment_id="20260717_future")
    store.save(record)
    # Hand-edit the file to simulate a newer app's format version.
    path = store.root / "20260717_future" / "experiment.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 999
    path.write_text(json.dumps(data), encoding="utf-8")

    with caplog.at_level("WARNING"):
        loaded = store.load("20260717_future")
    assert loaded.schema_version == 999
    assert any("newer app" in message for message in caplog.messages)


def test_store_data_dir_and_gui_state_path(store):
    assert store.data_dir("exp1") == store.root / "exp1" / "data"
    assert store.gui_state_path("exp1") == store.root / "exp1" / "gui_state.json"
    # The ELN outbox lives inside the experiment folder, so an experiment that
    # is copied elsewhere takes its unpublished runs with it.
    assert store.outbox_path("exp1") == store.root / "exp1" / "outbox.jsonl"
    # None of these calls creates anything on disk.
    assert not store.root.exists()


def test_relativize_and_resolve_data_file_plain_and_subfolder(store):
    store.save(ExperimentRecord(experiment_id="exp1"))
    data_dir = store.data_dir("exp1")
    data_dir.mkdir(parents=True)
    plain_file = data_dir / "run1.h5"
    plain_file.write_text("x", encoding="utf-8")
    sub_dir = data_dir / "heating_runs"
    sub_dir.mkdir()
    sub_file = sub_dir / "run2.h5"
    sub_file.write_text("x", encoding="utf-8")

    rel_plain = store.relativize_data_file("exp1", plain_file)
    assert rel_plain == "data/run1.h5"
    assert store.resolve_data_file("exp1", rel_plain) == plain_file

    rel_sub = store.relativize_data_file("exp1", sub_file)
    assert rel_sub == "data/heating_runs/run2.h5"
    assert store.resolve_data_file("exp1", rel_sub) == sub_file


def test_relativize_and_resolve_data_file_outside_bundle_stays_absolute(store, tmp_path):
    outside = tmp_path / "elsewhere" / "run.h5"
    outside.parent.mkdir(parents=True)
    outside.write_text("x", encoding="utf-8")

    stored = store.relativize_data_file("exp1", outside)
    assert stored == str(outside.resolve())
    resolved = store.resolve_data_file("exp1", stored)
    assert resolved == outside.resolve()


def test_resolve_data_file_survives_session_folder_relocation(tmp_path):
    """A dangling absolute path falls back to a basename search under data/."""
    old_root = tmp_path / "old_root"
    store = ExperimentStore(old_root)
    store.save(ExperimentRecord(experiment_id="exp1"))
    data_dir = store.data_dir("exp1")
    data_dir.mkdir(parents=True)
    data_file = data_dir / "run1.h5"
    data_file.write_text("x", encoding="utf-8")

    # Record the run with its (then-valid) absolute path, as an old-format
    # record would have stored it before bundle-relative paths existed.
    record = store.load("exp1")
    record.runs.append(RunRecord(run_id="r1", data_file=str(data_file)))
    store.save(record)

    # Move the whole session folder elsewhere.
    new_root = tmp_path / "new_root"
    shutil.move(str(old_root), str(new_root))

    new_store = ExperimentStore(new_root)
    stored = new_store.load("exp1")
    resolved = new_store.resolve_data_file("exp1", stored.runs[0].data_file)
    assert resolved == new_root / "exp1" / "data" / "run1.h5"
    assert resolved.is_file()


def test_resolve_data_file_dangling_absolute_no_match_returns_unchanged(store):
    missing = store.root.parent / "gone" / "nope.h5"
    resolved = store.resolve_data_file("exp1", str(missing))
    assert resolved == missing


# ── SessionStore ─────────────────────────────────────────────────────────────

@pytest.fixture
def session_store(tmp_path):
    return SessionStore(tmp_path / "sessions")


def test_session_round_trips_with_content():
    """A populated Session survives to_dict()/from_dict() unchanged."""
    session = Session(
        session_id="20260717_lab_a",
        user_id="jdoe",
        name="Lab A",
        default_experiment_dir="C:/data/lab_a",
        last_open_experiment_id="20260717_test",
        experiments=[
            ExperimentIndexEntry(
                experiment_id="20260717_test",
                title="Test",
                user_id="jdoe",
                status=EXPERIMENT_STATUS_OPEN,
                created_utc="2026-07-17T12:00:00+00:00",
            )
        ],
        created_utc="2026-07-17T12:00:00+00:00",
        last_opened_utc="2026-07-18T09:00:00+00:00",
    )
    assert session.schema_version == SCHEMA_VERSION
    payload = session.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert Session.from_dict(payload) == session


def test_session_from_dict_tolerates_junk():
    """Bad/missing input yields sane defaults, never raises."""
    assert Session.from_dict("not-a-dict") == Session()
    assert Session.from_dict(None) == Session()
    partial = Session.from_dict({"session_id": "x"})
    assert partial.session_id == "x"
    assert partial.user_id == ""
    assert partial.default_experiment_dir == ""
    assert partial.last_open_experiment_id == ""
    assert partial.experiments == []


def test_session_experiments_absent_on_disk_defaults_to_empty_list():
    """A session.json written before the index existed loads with experiments=[]."""
    session = Session.from_dict({"session_id": "x", "name": "X"})
    assert session.experiments == []


def test_session_experiments_tolerates_junk_entries():
    """Non-list/non-dict entries degrade to [] / are dropped, never raise."""
    assert Session.from_dict({"experiments": "not-a-list"}).experiments == []
    entries = Session.from_dict(
        {"experiments": [{"experiment_id": "e1"}, "junk", 5]}
    ).experiments
    assert len(entries) == 1
    assert entries[0].experiment_id == "e1"


def test_experiment_index_entry_round_trips_with_content():
    """A populated ExperimentIndexEntry survives to_dict()/from_dict() unchanged."""
    entry = ExperimentIndexEntry(
        experiment_id="20260717_test",
        title="Test",
        user_id="jdoe",
        status=EXPERIMENT_STATUS_CLOSED,
        created_utc="2026-07-17T12:00:00+00:00",
        closed_utc="2026-07-18T09:00:00+00:00",
    )
    assert ExperimentIndexEntry.from_dict(entry.to_dict()) == entry


def test_experiment_index_entry_untrusted_status_degrades_to_closed():
    """An entry with an unknown status must not claim to be open."""
    entry = ExperimentIndexEntry.from_dict({"experiment_id": "x", "status": "bogus"})
    assert entry.status == EXPERIMENT_STATUS_CLOSED


def test_session_schema_version_absent_defaults_to_one():
    session = Session.from_dict({"session_id": "x"})
    assert session.schema_version == 1


def test_session_schema_version_tolerates_future_value():
    session = Session.from_dict({"session_id": "x", "schema_version": 999})
    assert session.schema_version == 999


def test_session_store_creates_nothing_until_save(tmp_path):
    """Construction and reads must not create directories (lazy creation)."""
    root = tmp_path / "sessions"
    session_store = SessionStore(root)
    assert session_store.list_sessions("jdoe") == []
    assert session_store.get_active() is None
    assert session_store.load("jdoe", "nope") is None
    assert not root.exists()


def test_session_store_save_load_and_active_pointer(session_store):
    session = Session(session_id="20260717_lab_a", user_id="jdoe", name="Lab A")
    session_store.save(session)
    session_store.set_active("jdoe", "20260717_lab_a")
    assert session_store.list_sessions("jdoe") == ["20260717_lab_a"]
    assert session_store.load("jdoe", "20260717_lab_a") == session
    assert session_store.get_active() == ("jdoe", "20260717_lab_a")
    # No stray .tmp files after atomic writes.
    assert not list(session_store.root.rglob("*.tmp"))


def test_session_store_get_active_returns_none_for_legacy_flat_shape(session_store):
    """A pointer written before per-user nesting has neither key — treated as unset."""
    _write_json_atomic(session_store.root / "active.json", {"active": "20260717_lab_a"})
    assert session_store.get_active() is None


def test_session_store_save_requires_user_id_and_session_id(session_store):
    with pytest.raises(ValueError):
        session_store.save(Session())
    with pytest.raises(ValueError):
        session_store.save(Session(session_id="x"))
    with pytest.raises(ValueError):
        session_store.save(Session(user_id="jdoe"))


def test_session_store_save_derives_path_from_record_user_and_session_id(session_store):
    session = Session(session_id="20260717_lab_a", user_id="jdoe", name="Lab A")
    session_store.save(session)
    assert (session_store.root / "jdoe" / "20260717_lab_a" / "session.json").is_file()


def test_session_store_make_session_id_scoped_per_user(session_store):
    created = "2026-07-17T12:00:00+00:00"
    first = session_store.make_session_id("Lab A — Cryostat 1!", created, "jdoe")
    assert first == "20260717_lab_a_cryostat_1"
    session_store.save(Session(session_id=first, user_id="jdoe"))
    assert (
        session_store.make_session_id("Lab A — Cryostat 1!", created, "jdoe")
        == f"{first}_2"
    )
    # A different user picking the same name/date does not collide.
    assert session_store.make_session_id("Lab A — Cryostat 1!", created, "asmith") == first


def test_session_store_create_session_builds_saves_and_returns(session_store):
    session = session_store.create_session("Lab A", "jdoe")
    assert session.user_id == "jdoe"
    assert session.name == "Lab A"
    assert session.session_id
    assert session.created_utc == session.last_opened_utc
    assert session_store.load("jdoe", session.session_id) == session


def test_session_store_list_sessions_scoped_to_user_directory(session_store):
    session_store.create_session("Lab A", "jdoe")
    session_store.create_session("Lab B", "asmith")
    second_for_jdoe = session_store.create_session("Lab C", "jdoe")

    jdoe_sessions = session_store.list_sessions("jdoe")
    assert len(jdoe_sessions) == 2
    assert second_for_jdoe.session_id in jdoe_sessions
    assert session_store.list_sessions("asmith") != jdoe_sessions
    assert session_store.list_sessions("nobody") == []


def test_session_store_load_tolerates_corrupt_file(session_store):
    path = session_store.root / "jdoe" / "bad" / "session.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert session_store.load("jdoe", "bad") is None
    assert "bad" in session_store.list_sessions("jdoe")  # listed but unloadable


def test_roster_add_get_replace(tmp_path):
    roster = UserRoster(tmp_path / "users.json")
    assert roster.list_users() == []
    roster.add(User(user_id="jdoe", name="J. Doe"))
    roster.add(User(user_id="asmith", name="A. Smith"))
    assert {u.user_id for u in roster.list_users()} == {"jdoe", "asmith"}
    roster.add(User(user_id="jdoe", name="Jay Doe"))  # replace, not duplicate
    assert roster.get("jdoe").name == "Jay Doe"
    assert len(roster.list_users()) == 2
    assert roster.get("nobody") is None


# ── ExperimentManager lifecycle ─────────────────────────────────────────────────

def test_experiment_context_setup_tier_present_with_no_experiment_open(manager):
    """The setup tier is available even before any experiment is ever started."""
    context = manager.experiment_context()
    assert context == {"setup": {"config_name": "sim_cryostat", "instruments": {}}, "experiment": {}}


def test_experiment_context_includes_instrument_metadata_from_config(
    store, roster, orchestrator, station
):
    """config_path wires read_instrument_metadata() into the setup tier."""
    manager = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
        config_path=CONFIG_PATH,
    )
    instruments = manager.experiment_context()["setup"]["instruments"]
    assert instruments  # sim_cryostat/devices.yaml carries metadata for every VI
    assert instruments["magnet_z"]["role"] == "Z-axis magnet"


def test_experiment_context_tolerates_missing_config_path(store, roster, orchestrator, station):
    """A bad/absent config_path degrades to no instrument metadata, never raises."""
    manager = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=orchestrator,
        config_path="/no/such/config",
    )
    assert manager.experiment_context()["setup"]["instruments"] == {}


def test_envelope_variables_expose_the_setup_bounds_to_narrow(manager, station):
    """The read side of the envelope: what the Start dialog pre-fills from."""
    variables = manager.envelope_variables()

    assert "magnet_z" in variables
    magnet = variables["magnet_z"]
    assert (magnet.method_name, magnet.param_name) == ("set_field", "target_T")
    assert (magnet.config_min, magnet.config_max) == station.get_vi(
        "magnet_z"
    ).limit_bounds("field_T")
    assert magnet.unit_suffix == "T"


def test_start_experiment_persists_and_installs_envelope(manager, orchestrator, store):
    envelope = ExperimentEnvelope(
        bounds={"magnet_z": EnvelopeBound(min_value=-2.0, max_value=2.0)}
    )
    changed: list[dict] = []
    manager.experiment_changed.connect(changed.append)

    record = manager.start_experiment(
        "SOT switching vs T", "jdoe", SAMPLE_INFO, envelope=envelope
    )

    assert store.get_active() == record.experiment_id
    assert store.load(record.experiment_id) == record
    assert record.config_name == "sim_cryostat"
    assert orchestrator._session_envelope == envelope
    assert changed and changed[-1]["experiment_id"] == record.experiment_id
    context = manager.experiment_context()
    assert context["setup"]["config_name"] == "sim_cryostat"
    assert context["experiment"]["experiment_id"] == record.experiment_id
    assert context["experiment"]["user_name"] == "J. Doe"
    assert context["experiment"]["attended"] is True
    assert context["experiment"]["eln_link"] == {}


def test_start_experiment_rejects_unknown_user_and_double_open(manager):
    with pytest.raises(ValueError, match="Unknown user"):
        manager.start_experiment("X", "nobody", SAMPLE_INFO)
    manager.start_experiment("X", "jdoe", SAMPLE_INFO)
    with pytest.raises(ValueError, match="still open"):
        manager.start_experiment("Y", "jdoe", SAMPLE_INFO)


def test_start_experiment_with_custom_dirname_uses_it_as_experiment_id(manager, store):
    record = manager.start_experiment(
        "X", "jdoe", SAMPLE_INFO, experiment_dirname="my_custom_folder"
    )
    assert record.experiment_id == "my_custom_folder"
    assert store.load("my_custom_folder") == record


def test_start_experiment_rejects_empty_dirname(manager):
    with pytest.raises(ValueError, match="must not be empty"):
        manager.start_experiment("X", "jdoe", SAMPLE_INFO, experiment_dirname="   ")


@pytest.mark.parametrize("bad_dirname", ["a/b", "a\\b", ".", ".."])
def test_start_experiment_rejects_separator_or_dot_dirname(manager, bad_dirname):
    with pytest.raises(ValueError):
        manager.start_experiment("X", "jdoe", SAMPLE_INFO, experiment_dirname=bad_dirname)


def test_start_experiment_rejects_dirname_collision(manager):
    manager.start_experiment("X", "jdoe", SAMPLE_INFO, experiment_dirname="taken")
    manager.close_experiment()
    with pytest.raises(ValueError, match="already exists"):
        manager.start_experiment("Y", "jdoe", SAMPLE_INFO, experiment_dirname="taken")


def test_start_experiment_updates_session_index(indexed_manager):
    exp_manager, session_store, session = indexed_manager
    record = exp_manager.start_experiment("X", "jdoe", SAMPLE_INFO)

    reloaded = session_store.load("jdoe", session.session_id)
    assert len(reloaded.experiments) == 1
    entry = reloaded.experiments[0]
    assert entry.experiment_id == record.experiment_id
    assert entry.title == "X"
    assert entry.user_id == "jdoe"
    assert entry.status == EXPERIMENT_STATUS_OPEN
    assert entry.created_utc == record.created_utc
    assert entry.closed_utc == ""


def test_close_experiment_updates_session_index_status_and_closed_utc(indexed_manager):
    exp_manager, session_store, session = indexed_manager
    record = exp_manager.start_experiment("X", "jdoe", SAMPLE_INFO)
    exp_manager.close_experiment()

    reloaded = session_store.load("jdoe", session.session_id)
    assert len(reloaded.experiments) == 1
    entry = reloaded.experiments[0]
    assert entry.experiment_id == record.experiment_id
    assert entry.status == EXPERIMENT_STATUS_CLOSED
    assert entry.closed_utc


def test_switch_experiment_reconciles_session_index(indexed_manager):
    """switch_experiment() is an "opened" event too — it re-reconciles the index.

    Simulates the record having changed on disk out-of-band since the index
    was last touched (exactly what a manual folder move would also cause):
    the reconciled index must reflect that new reality, not the stale one.
    """
    exp_manager, session_store, session = indexed_manager
    first = exp_manager.start_experiment("First", "jdoe", SAMPLE_INFO)
    exp_manager.close_experiment()
    exp_manager.start_experiment("Second", "jdoe", SAMPLE_INFO)
    exp_manager.close_experiment()
    exp_manager.store.save(
        ExperimentRecord(
            experiment_id=first.experiment_id,
            title="First",
            user_id="jdoe",
            status=EXPERIMENT_STATUS_OPEN,
        )
    )
    before = session_store.load("jdoe", session.session_id).experiments
    assert next(e for e in before if e.experiment_id == first.experiment_id).status == (
        EXPERIMENT_STATUS_CLOSED
    )

    exp_manager.switch_experiment(first.experiment_id)

    after = session_store.load("jdoe", session.session_id).experiments
    assert next(e for e in after if e.experiment_id == first.experiment_id).status == (
        EXPERIMENT_STATUS_OPEN
    )


def test_reconciliation_picks_up_an_experiment_folder_moved_in_and_keeps_its_user_id(
    indexed_manager, roster, orchestrator, station,
):
    """An experiment folder moved into a session by hand is picked up, author intact.

    Models handing an experiment off to a different user's session to
    continue the project: the folder physically moves, but the record's own
    ``user_id`` (who actually ran it) must survive untouched.
    """
    exp_manager, session_store, session = indexed_manager
    moved = exp_manager.start_experiment("Moved In", "jdoe", SAMPLE_INFO)
    exp_manager.close_experiment()

    other_session = session_store.create_session("Lab B", "jdoe")
    other_root = session_store.root / "jdoe" / other_session.session_id
    other_root.mkdir(parents=True, exist_ok=True)
    shutil.move(
        str(exp_manager.store.root / moved.experiment_id),
        str(other_root / moved.experiment_id),
    )

    other_manager = ExperimentManager(
        store=ExperimentStore(other_root),
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
        session_store=session_store,
    )
    other_manager.start_experiment("Native", "jdoe", SAMPLE_INFO)
    other_manager.close_experiment()

    reloaded = session_store.load("jdoe", other_session.session_id)
    entry = next(e for e in reloaded.experiments if e.experiment_id == moved.experiment_id)
    assert entry.user_id == "jdoe"
    assert entry.title == "Moved In"


def test_reconciliation_drops_an_experiment_folder_moved_out_of_a_session(indexed_manager):
    exp_manager, session_store, session = indexed_manager
    moved = exp_manager.start_experiment("Moved Out", "jdoe", SAMPLE_INFO)
    exp_manager.close_experiment()
    reloaded = session_store.load("jdoe", session.session_id)
    assert any(e.experiment_id == moved.experiment_id for e in reloaded.experiments)

    other_session = session_store.create_session("Lab B", "jdoe")
    other_root = session_store.root / "jdoe" / other_session.session_id
    other_root.mkdir(parents=True, exist_ok=True)
    shutil.move(
        str(exp_manager.store.root / moved.experiment_id),
        str(other_root / moved.experiment_id),
    )

    stayed = exp_manager.start_experiment("Still Here", "jdoe", SAMPLE_INFO)
    exp_manager.close_experiment()

    reloaded = session_store.load("jdoe", session.session_id)
    ids = {e.experiment_id for e in reloaded.experiments}
    assert moved.experiment_id not in ids
    assert stayed.experiment_id in ids


def test_manager_without_session_store_skips_index_update(manager):
    """A manager built with session_store=None (the default) never touches one."""
    manager.start_experiment("X", "jdoe", SAMPLE_INFO)
    manager.close_experiment()  # must not raise for lack of a session_store


def test_close_experiment_clears_envelope_and_context(manager, orchestrator, store):
    envelope = ExperimentEnvelope(bounds={"magnet_z": EnvelopeBound(max_value=2.0)})
    record = manager.start_experiment("X", "jdoe", SAMPLE_INFO, envelope=envelope)
    manager.close_experiment()

    assert manager.current_experiment() is None
    context = manager.experiment_context()
    assert context["experiment"] == {}
    assert context["setup"]["config_name"] == "sim_cryostat"
    assert orchestrator._session_envelope is None
    assert store.get_active() is None
    stored = store.load(record.experiment_id)
    assert stored.status == EXPERIMENT_STATUS_CLOSED
    assert stored.closed_utc


def test_set_findings_and_attendance_persist(manager, store):
    record = manager.start_experiment("X", "jdoe", SAMPLE_INFO)
    manager.set_findings("R(T) shows a clean transition at 9.1 K")
    manager.set_attended(False)
    stored = store.load(record.experiment_id)
    assert stored.findings.startswith("R(T)")
    assert stored.attended is False


def test_runs_outside_experiment_are_not_recorded(manager, orchestrator):
    orchestrator.run_started.emit({"run_id": "r1", "procedure": "Field Sweep"})
    assert manager.current_experiment() is None


def test_run_recording_from_manifests(manager, orchestrator, store):
    record = manager.start_experiment("X", "jdoe", SAMPLE_INFO)
    recorded: list[dict] = []
    manager.run_recorded.connect(recorded.append)

    orchestrator.run_started.emit(
        {
            "run_id": "r1",
            "procedure": "Field Sweep",
            "kind": "run",
            "params": {"field_steps": 3},
            "data_file": "/data/x.h5",
            "started_utc": "2026-07-17T12:00:00+00:00",
        }
    )
    stored = store.load(record.experiment_id)
    assert len(stored.runs) == 1
    assert stored.runs[0].status == RUN_STATUS_RUNNING

    orchestrator.run_finished.emit(
        {
            "run_id": "r1",
            "finished_utc": "2026-07-17T12:05:00+00:00",
            "status": "done",
            "reason": "",
        }
    )
    stored = store.load(record.experiment_id)
    run = stored.runs[0]
    assert run.status == RUN_STATUS_DONE
    assert run.finished_utc
    assert len(recorded) == 2


def test_resume_marks_stale_running_runs_failed(
    manager, store, roster, station, qtbot
):
    """A new manager resumes the active experiment; crashed runs become failed."""
    record = manager.start_experiment(
        "X",
        "jdoe",
        SAMPLE_INFO,
        envelope=ExperimentEnvelope(bounds={"magnet_z": EnvelopeBound(max_value=2.0)}),
    )
    # Simulate the app dying mid-run: a run stuck in "running" on disk.
    record.runs.append(RunRecord(run_id="r1", status=RUN_STATUS_RUNNING))
    store.save(record)

    fresh_orchestrator = Orchestrator(station, tick_interval_ms=10)
    resumed = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=fresh_orchestrator,
        config_name="sim_cryostat",
    )
    experiment = resumed.current_experiment()
    assert experiment is not None
    assert experiment.experiment_id == record.experiment_id
    run = experiment.find_run("r1")
    assert run.status == RUN_STATUS_FAILED
    assert "restart" in run.reason
    # The stored envelope was re-installed on the new orchestrator.
    assert fresh_orchestrator._session_envelope is not None
    # And the failure was persisted, not just held in memory.
    assert store.load(record.experiment_id).find_run("r1").status == RUN_STATUS_FAILED


def test_resume_with_missing_record_clears_pointer(store, roster, orchestrator, station):
    store.set_active("ghost")
    manager = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=orchestrator,
    )
    assert manager.current_experiment() is None
    assert store.get_active() is None


# ── set_queue / current_data_dir / current_gui_state_path ────────────────────

def test_set_queue_persists_and_is_noop_when_nothing_open(manager, store):
    assert manager.current_experiment() is None
    manager.set_queue([{"procedure": "Field Sweep"}])  # no-op, no experiment
    assert manager.current_experiment() is None

    record = manager.start_experiment("X", "jdoe", SAMPLE_INFO)
    queue = [{"procedure": "Field Sweep", "params": {"field_end": 1.0}}]
    manager.set_queue(queue)
    assert store.load(record.experiment_id).queue == queue


def test_current_data_dir_and_gui_state_path(manager, store):
    assert manager.current_data_dir() is None
    assert manager.current_gui_state_path() is None

    record = manager.start_experiment("X", "jdoe", SAMPLE_INFO)
    assert manager.current_data_dir() == store.data_dir(record.experiment_id)
    assert manager.current_gui_state_path() == store.gui_state_path(record.experiment_id)


# ── switch_experiment ─────────────────────────────────────────────────────────

def test_switch_experiment_happy_path(manager, store, orchestrator):
    envelope_a = ExperimentEnvelope(bounds={"magnet_z": EnvelopeBound(max_value=1.0)})
    envelope_b = ExperimentEnvelope(bounds={"magnet_z": EnvelopeBound(max_value=2.0)})
    first = manager.start_experiment("First", "jdoe", SAMPLE_INFO, envelope=envelope_a)

    # A second, independently-open experiment exists in the store (as if
    # created in an earlier session) — written directly since start_experiment
    # refuses to open a second one while one is already open.
    second = ExperimentRecord(
        experiment_id="20260717_second",
        title="Second",
        user_id="jdoe",
        status=EXPERIMENT_STATUS_OPEN,
        envelope=envelope_to_dict(envelope_b),
    )
    store.save(second)

    changed: list[dict] = []
    manager.experiment_changed.connect(changed.append)

    result = manager.switch_experiment(second.experiment_id)
    assert result.experiment_id == second.experiment_id
    assert manager.current_experiment().experiment_id == second.experiment_id
    assert store.get_active() == second.experiment_id
    assert changed and changed[-1]["experiment_id"] == second.experiment_id
    assert orchestrator._session_envelope == envelope_b
    # The experiment switched away from is untouched: still "open" on disk.
    assert store.load(first.experiment_id).status == EXPERIMENT_STATUS_OPEN

    back = manager.switch_experiment(first.experiment_id)
    assert back.experiment_id == first.experiment_id
    assert manager.current_experiment().experiment_id == first.experiment_id
    assert store.get_active() == first.experiment_id
    assert orchestrator._session_envelope == envelope_a


def test_switch_experiment_rejects_unknown_id(manager):
    with pytest.raises(ValueError, match="Unknown experiment"):
        manager.switch_experiment("nope")


def test_switch_experiment_rejects_closed_target(manager, store):
    record = manager.start_experiment("X", "jdoe", SAMPLE_INFO)
    manager.close_experiment()
    assert store.load(record.experiment_id).status == EXPERIMENT_STATUS_CLOSED
    with pytest.raises(ValueError, match="not open"):
        manager.switch_experiment(record.experiment_id)


def test_switch_experiment_rejects_future_schema_version(manager, store):
    manager.start_experiment("Current", "jdoe", SAMPLE_INFO)
    # to_dict() always stamps the *current* SCHEMA_VERSION (see models.py), so
    # simulating a newer app's file means hand-editing the JSON, exactly like
    # the store-level future-schema test does.
    store.save(ExperimentRecord(experiment_id="future_one", status=EXPERIMENT_STATUS_OPEN))
    path = store.root / "future_one" / "experiment.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 999
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="newer app"):
        manager.switch_experiment("future_one")


# ── Save-failure surfacing (store_health_changed) ─────────────────────────────

def test_store_health_changed_fires_once_on_failure_and_once_on_recovery(manager, store, monkeypatch):
    manager.start_experiment("X", "jdoe", SAMPLE_INFO)
    events: list[dict] = []
    manager.store_health_changed.connect(events.append)

    def _boom(record):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save", _boom)
    manager.set_findings("first failure")
    manager.set_findings("second failure")  # still failing — must not re-emit
    assert events == [{"ok": False, "detail": "disk full"}]

    monkeypatch.undo()
    manager.set_findings("recovered")
    manager.set_findings("still fine")  # already ok — must not re-emit
    assert events == [
        {"ok": False, "detail": "disk full"},
        {"ok": True, "detail": ""},
    ]


# ── Future schema_version belt-and-suspenders on _save_current ───────────────

def test_save_current_refuses_to_overwrite_future_schema_version(manager, store, caplog):
    record = manager.start_experiment("X", "jdoe", SAMPLE_INFO)
    on_disk_before = store.load(record.experiment_id)

    # Simulate the in-memory record somehow carrying a future schema_version
    # (belt-and-suspenders: switch_experiment already refuses this at the
    # door, but _save_current must never write one back regardless).
    manager._experiment.schema_version = 999
    with caplog.at_level("WARNING"):
        manager.set_findings("should not be written")
    assert any("Refusing to overwrite" in message for message in caplog.messages)
    assert store.load(record.experiment_id) == on_disk_before


# ── Accountability: who started the run, and who queued it ──────────────────

AGENT = ev.Actor(kind=ev.ActorKind.AGENT, id="runner-7", role="session")


def test_a_run_record_names_the_operator_by_default(manager, orchestrator, store):
    """The physicist's own run says so, and is not flagged as a legacy record."""
    record = manager.start_experiment("X", "jdoe", SAMPLE_INFO)

    orchestrator.run_started.emit({"run_id": "r1", "procedure": "Field Sweep"})
    orchestrator.event_emitted.emit(ev.RunStarted(run_id="r1"))

    run = store.load(record.experiment_id).find_run("r1")
    assert run.actor == ev.OPERATOR
    assert run.actor_legacy is False


def test_an_agent_started_run_names_the_agent_forever_after(
    manager, orchestrator, store
):
    """The exit criterion: the record, not just the live event, says who ran it."""
    record = manager.start_experiment("X", "jdoe", SAMPLE_INFO)

    orchestrator.run_started.emit({"run_id": "r1", "procedure": "Field Sweep"})
    orchestrator.event_emitted.emit(ev.RunStarted(run_id="r1", actor=AGENT))

    run = store.load(record.experiment_id).find_run("r1")
    assert run.actor.kind is ev.ActorKind.AGENT
    assert run.actor.id == "runner-7"
    assert run.actor.role == "session"
    assert run.actor_legacy is False


def test_a_real_agent_run_is_recorded_as_the_agents(
    manager, orchestrator, station, store, tmp_path, qtbot
):
    """Through the real engine: the actor on the command reaches the record."""
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []
    record = manager.start_experiment("X", "jdoe", SAMPLE_INFO)
    procedure = FieldSweep(
        station=station,
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
        experiment_info=manager.experiment_context(),
        **FAST_PARAMS,
    )

    orchestrator.run_procedure(procedure, actor=AGENT)
    with qtbot.waitSignal(orchestrator.procedure_finished, timeout=10000):
        pass

    run = store.load(record.experiment_id).runs[0]
    assert run.actor.kind is ev.ActorKind.AGENT
    assert run.actor.id == "runner-7"


def test_a_run_written_before_actors_were_stamped_loads_as_legacy(manager, store):
    """An old file must not read as "the physicist did it" — it reads as unknown."""
    record = manager.start_experiment("X", "jdoe", SAMPLE_INFO)
    path = store.root / record.experiment_id / "experiment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runs"] = [{"run_id": "old", "procedure": "Field Sweep", "status": "done"}]
    path.write_text(json.dumps(payload), encoding="utf-8")

    run = store.load(record.experiment_id).find_run("old")

    assert run.actor == ev.OPERATOR
    assert run.actor_legacy is True, "the sentinel is not evidence of who acted"


def test_an_unreadable_actor_field_degrades_to_legacy():
    """Junk in the actor field never raises, and never claims the operator acted."""
    assert RunRecord.from_dict({"actor": {"kind": "wizard", "id": "x"}}).actor_legacy
    assert RunRecord.from_dict({"actor": "jdoe"}).actor_legacy


# ── Accountability: what the run was started with ───────────────────────────


def test_a_run_record_digests_the_parameters_it_started_with(
    manager, orchestrator, store
):
    """The Params digest is stamped when the run opens, from the manifest itself."""
    record = manager.start_experiment("X", "jdoe", SAMPLE_INFO)
    params = {"start_T": 0.0, "stop_T": 1.0, "points": 11}

    orchestrator.run_started.emit(
        {"run_id": "r1", "procedure": "Field Sweep", "params": params}
    )

    run = store.load(record.experiment_id).find_run("r1")
    assert run.params_digest == params_digest(params)
    assert run.params == params


def test_the_run_digest_is_stored_not_recomputed_on_read(manager, store):
    """It fixes what the run started with, so an amended record cannot rewrite it."""
    record = manager.start_experiment("X", "jdoe", SAMPLE_INFO)
    path = store.root / record.experiment_id / "experiment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runs"] = [
        {"run_id": "r1", "params": {"start_T": 9.9}, "params_digest": "abc123"}
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert store.load(record.experiment_id).find_run("r1").params_digest == "abc123"


def test_a_run_written_before_digests_reads_as_no_digest():
    """An old record has no digest, and never a wrong one invented on read."""
    assert RunRecord.from_dict({"run_id": "old", "params": {"a": 1}}).params_digest == ""


def test_the_run_digest_round_trips_through_the_record():
    run = RunRecord(run_id="r1", params={"b": 2, "a": 1}, params_digest=params_digest({"a": 1, "b": 2}))
    assert RunRecord.from_dict(run.to_dict()).params_digest == run.params_digest


# ── End-to-end: a real run recorded and cross-checked against HDF5 ───────────

def test_end_to_end_run_recorded_and_stamped(
    manager, orchestrator, station, store, tmp_path, qtbot
):
    """A real FieldSweep run produces a RunRecord matching the HDF5 on disk."""
    station.magnet_z._default_ramp_rate = 6000.0
    station.magnet_z._ramp_segments = []

    record = manager.start_experiment(
        "SOT switching vs T",
        "jdoe",
        SAMPLE_INFO,
        envelope=ExperimentEnvelope(
            bounds={"magnet_z": EnvelopeBound(min_value=-2.0, max_value=2.0)}
        ),
    )
    # Exactly what the GUI does when building a procedure: stamp the context.
    procedure = FieldSweep(
        station=station,
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
        experiment_info=manager.experiment_context(),
        **FAST_PARAMS,
    )
    orchestrator.run_procedure(procedure)
    with qtbot.waitSignal(orchestrator.procedure_finished, timeout=10000):
        pass

    stored = store.load(record.experiment_id)
    assert len(stored.runs) == 1
    run = stored.runs[0]
    assert run.status == RUN_STATUS_DONE
    assert run.kind == "run"
    assert run.procedure == "Field Sweep"
    assert run.params["field_steps"] == 3

    # The record's data_file is the real HDF5 file, stamped with the context.
    with h5py.File(run.data_file, "r") as f:
        info = json.loads(f["metadata"].attrs["experiment_info"])
    assert info["experiment"]["experiment_id"] == record.experiment_id
    assert info["experiment"]["user_id"] == "jdoe"
    assert info["experiment"]["experiment_title"] == "SOT switching vs T"
    assert info["setup"]["config_name"] == "sim_cryostat"


# ── set_run_eln_link (the publishing track's one write path) ─────────────────

def test_set_run_eln_link_stamps_the_open_experiment(manager, qtbot):
    """A confirmed entry lands on the run, is persisted, and is announced."""
    record = manager.start_experiment("ELN", "jdoe", dict(SAMPLE_INFO))
    manager._on_run_started({"run_id": "r1", "procedure": "Field Sweep"})
    link = ElnLink(backend="sim_eln", entry_id="7", url="https://eln/7")

    with qtbot.waitSignal(manager.run_recorded):
        assert manager.set_run_eln_link(record.experiment_id, "r1", link) is True

    run = manager.current_experiment().find_run("r1")
    assert run.eln_link == link and run.published is True
    assert manager.store.load(record.experiment_id).find_run("r1").eln_link == link


def test_set_run_eln_link_reaches_a_closed_experiment(manager):
    """A job that drains after the experiment closed still stamps its own record."""
    record = manager.start_experiment("ELN", "jdoe", dict(SAMPLE_INFO))
    manager._on_run_started({"run_id": "r1", "procedure": "Field Sweep"})
    manager.close_experiment()
    link = ElnLink(backend="sim_eln", entry_id="7", url="https://eln/7")

    assert manager.set_run_eln_link(record.experiment_id, "r1", link) is True
    assert manager.current_experiment() is None, "the closed record must not become live"
    assert manager.store.load(record.experiment_id).find_run("r1").eln_link == link


def test_set_run_eln_link_refuses_unknown_targets_without_raising(manager):
    """Bookkeeping failures are reported, never propagated into a GUI timer."""
    record = manager.start_experiment("ELN", "jdoe", dict(SAMPLE_INFO))
    link = ElnLink(backend="sim_eln", entry_id="7")
    assert manager.set_run_eln_link(record.experiment_id, "no-such-run", link) is False
    assert manager.set_run_eln_link("no-such-experiment", "r1", link) is False


# ── The run queue (validated on add, pulled by the engine) ──────────────────


@pytest.fixture
def queue_manager(store, roster, orchestrator, station):
    """A manager that owns a run queue: a Station to build with, and a catalog."""
    manager = ExperimentManager(
        store=store,
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
        station=station,
        run_catalog={"FieldSweep": FieldSweep},
    )
    orchestrator.next_procedure = manager.next_run
    orchestrator.queue_snapshot = manager.queue_entries
    return manager


def _queue_events(orchestrator):
    """Collect every QueueChanged the Orchestrator emits from now on."""
    events: list = []
    orchestrator.event_emitted.connect(
        lambda event: events.append(event) if isinstance(event, ev.QueueChanged) else None
    )
    return events


def test_a_queued_run_carries_who_queued_it(queue_manager, tmp_path):
    """Every queue entry names the actor that put it there, spec and JSON alike."""
    queue_manager.queue_run(
        FieldSweep, FAST_PARAMS, data_directory=str(tmp_path), actor=AGENT
    )

    assert queue_manager.queue_snapshot()[0].actor == AGENT
    assert queue_manager.queue_entries()[0]["actor"] == AGENT.to_json()


def test_the_queue_broadcast_carries_the_actor_of_every_entry(
    queue_manager, orchestrator, tmp_path
):
    """QueueChanged is what a client renders from, so the actor must survive it."""
    events = _queue_events(orchestrator)

    queue_manager.queue_run(
        FieldSweep, FAST_PARAMS, data_directory=str(tmp_path), actor=AGENT
    )

    assert events[-1].actor == AGENT
    assert [entry["actor"] for entry in events[-1].entries] == [AGENT.to_json()]


def test_a_valid_run_is_queued_as_a_spec(queue_manager, tmp_path):
    """What waits in the queue is data, not a live procedure object."""
    spec, validation = queue_manager.queue_run(
        FieldSweep,
        FAST_PARAMS,
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
        file_prefix="q1",
    )

    assert validation.ok
    assert spec is not None
    assert spec.run_class == "FieldSweep"
    assert spec.file_prefix == "q1"
    assert queue_manager.queue_snapshot() == (spec,)
    assert not hasattr(spec, "proc")


def test_an_out_of_bounds_run_is_refused_at_add_time_with_findings(
    queue_manager, tmp_path
):
    """A spec that fails validation never enters the queue."""
    spec, validation = queue_manager.queue_run(
        FieldSweep,
        dict(FAST_PARAMS, field_end=50.0),
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
    )

    assert spec is None
    assert not validation.ok
    assert any("magnet_z" in message for message in validation.messages())
    assert queue_manager.queue_snapshot() == ()


def test_validation_uses_the_open_experiment_envelope(queue_manager, tmp_path):
    """The envelope narrows the setup's limits, and narrows them at queue time."""
    queue_manager.start_experiment(
        "Bounded",
        "jdoe",
        dict(SAMPLE_INFO),
        envelope=ExperimentEnvelope(
            bounds={"magnet_z": EnvelopeBound(min_value=-0.01, max_value=0.01)}
        ),
    )

    validation = queue_manager.validate_run(
        FieldSweep, FAST_PARAMS, data_directory=str(tmp_path)
    )

    assert not validation.ok
    assert any("envelope" in message for message in validation.messages())


def test_validate_run_without_a_station_says_so(manager):
    """A manager built for the experiment tier alone cannot build a run."""
    with pytest.raises(RuntimeError, match="Station"):
        manager.validate_run(FieldSweep, FAST_PARAMS)


def test_every_queue_mutation_broadcasts_and_names_its_actor(
    queue_manager, orchestrator, tmp_path
):
    """QueueChanged rides the engine's one event stream, actor and all."""
    events = _queue_events(orchestrator)
    agent = ev.Actor(kind=ev.ActorKind.AGENT, id="drift-watch", role="operator")

    spec, _ = queue_manager.queue_run(
        FieldSweep,
        FAST_PARAMS,
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
        actor=agent,
    )
    queue_manager.dequeue_run(spec.spec_id, actor=agent)

    assert [event.actor for event in events] == [agent, agent]
    assert events[0].entries[0]["run_class"] == "FieldSweep"
    assert events[0].entries[0]["actor"]["id"] == "drift-watch"
    assert events[-1].entries == ()


def test_a_no_op_mutation_broadcasts_nothing(queue_manager, orchestrator):
    """Nothing changed means nothing to tell anyone about."""
    events = _queue_events(orchestrator)

    assert queue_manager.dequeue_run("no-such-spec") is False
    assert queue_manager.move_queued_run("no-such-spec", -1) is False
    assert queue_manager.clear_run_queue() is False
    assert events == []


def test_the_engine_pulls_the_next_run_and_builds_it_here(
    queue_manager, orchestrator, tmp_path
):
    """The engine asks; exactly one live object comes into existence."""
    queue_manager.queue_run(
        FieldSweep,
        FAST_PARAMS,
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
    )

    run = orchestrator.next_procedure()

    assert isinstance(run, FieldSweep)
    assert queue_manager.queue_snapshot() == ()


def test_a_pulled_run_is_stamped_with_the_experiment_open_when_it_starts(
    queue_manager, tmp_path
):
    """A run queued before an experiment opened belongs to the one that runs it."""
    queue_manager.queue_run(
        FieldSweep,
        FAST_PARAMS,
        sample_info=SAMPLE_INFO,
        data_directory=str(tmp_path),
    )
    record = queue_manager.start_experiment("Later", "jdoe", dict(SAMPLE_INFO))

    run = queue_manager.next_run()

    assert (
        run._experiment_info["experiment"]["experiment_id"] == record.experiment_id
    )


def test_next_run_on_an_empty_queue_is_none(queue_manager):
    """An exhausted queue is not an error — the engine simply stays IDLE."""
    assert queue_manager.next_run() is None


