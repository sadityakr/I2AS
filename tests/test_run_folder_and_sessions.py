"""Run subfolders inside an experiment, and loading another session while running.

See ``docs/plans/run-folder-and-live-session-plan.md``: the subfolder rule and
monotonic numbering (``core.run_naming``), the engine's enforcement
(``Orchestrator.set_run_folder``), the session layer's policy value and its
two-phase live session switch (``ExperimentManager``), and the consumers that
let go of the old session.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from i2as.core import events as ev
from i2as.core.orchestrator import Orchestrator
from i2as.core.run_naming import (
    MAX_RUN_PATH_CHARS,
    next_run_number,
    normalize_run_subfolder,
    place_run,
)
from i2as.core.station import build_station
from i2as.session.manager import ExperimentManager
from i2as.session.models import (
    RUN_STATUS_DONE,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    ExperimentRecord,
    RunRecord,
    User,
)
from i2as.session.store import (
    SESSION_LOCK_FILENAME,
    ExperimentStore,
    SessionLockedError,
    SessionStore,
    UserRoster,
)

CONFIG_PATH = "i2as/configs/sim_cryostat"
SAMPLE = {"sample_name": "A3", "sample_id": "A3", "comments": ""}


class Placeable:
    """The smallest run the engine can place."""

    name = "Stub"
    run_kind = "run"
    file_prefix = ""

    def __init__(self) -> None:
        self.placed: tuple[str, str] | None = None

    def place_data_file(self, data_directory: str, file_name: str) -> None:
        self.placed = (data_directory, file_name)


# ----------------------------------------------------------------------
# core.run_naming
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("cooldown2", "cooldown2"),
        (" cooldown2//field sweeps/ ", "cooldown2/field sweeps"),
        ("a\\b", "a/b"),
        ("a/b/c", "a/b/c"),
        ("v1.2_run-3", "v1.2_run-3"),
    ],
)
def test_subfolder_rule_accepts_and_normalises(text, expected):
    assert normalize_run_subfolder(text) == expected


@pytest.mark.parametrize(
    "text",
    ["..", "a/../b", "/abs", "C:/x", "a/b/c/d", "nul", "COM1.txt", "x.", "x /y", "-a", "a,b", "x" * 65],
)
def test_subfolder_rule_refuses(text):
    with pytest.raises(ValueError):
        normalize_run_subfolder(text)


def test_numbers_count_the_data_folder_and_the_target_and_respect_the_floor(tmp_path):
    data = tmp_path / "data"
    (data / "cooldown2").mkdir(parents=True)
    (data / "run-0004_FieldSweep.h5").write_bytes(b"")
    (data / "run-0009_dir.h5").mkdir()  # a directory never counts

    placement = place_run(data, "FieldSweep", subfolder="cooldown2")
    assert placement.run_id == "run-0005"
    assert Path(placement.data_directory) == data / "cooldown2"

    assert next_run_number(data / "cooldown2", 12, data) == 12


def test_a_path_too_long_is_refused(tmp_path):
    deep = "x" * 60
    with pytest.raises(ValueError, match="characters"):
        place_run(tmp_path, "P" * MAX_RUN_PATH_CHARS, subfolder=f"{deep}/{deep}/{deep}")


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_subfolder_that_leads_outside_is_refused(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        place_run(data, "FieldSweep", subfolder="escape")


# ----------------------------------------------------------------------
# The engine
# ----------------------------------------------------------------------


@pytest.fixture
def engine(qtbot):
    orchestrator = Orchestrator(build_station(CONFIG_PATH), tick_interval_ms=10)
    verdicts: list[ev.Verdict] = []
    orchestrator.verdict_emitted.connect(verdicts.append)
    yield orchestrator, verdicts
    orchestrator.shutdown()


def _set_run_folder(orchestrator: Orchestrator, **args) -> None:
    orchestrator.submit(ev.Command(name=ev.CommandName.SET_RUN_FOLDER, args=args))


def test_runs_are_placed_in_the_subfolder_and_numbers_never_repeat(engine, tmp_path):
    orchestrator, _verdicts = engine
    data = tmp_path / "data"
    _set_run_folder(orchestrator, data_directory=str(data), subfolder="cd2", run_number_floor=3)

    first = Placeable()
    placement = orchestrator._place_run(first)
    assert placement.run_id == "run-0003"
    assert first.placed[0] == str(data / "cd2")

    # The file was never written (or was deleted): its number is still spent.
    assert orchestrator._place_run(Placeable()).run_id == "run-0004"


def test_a_bad_subfolder_is_refused_and_nothing_changes(engine, tmp_path):
    orchestrator, verdicts = engine
    _set_run_folder(orchestrator, data_directory=str(tmp_path), subfolder="ok")
    _set_run_folder(orchestrator, data_directory=str(tmp_path), subfolder="../escape")
    assert verdicts[-1].code is not ev.VerdictCode.OK
    assert orchestrator._run_subfolder == "ok"


def test_releasing_the_folder_is_refused_while_a_run_is_active(engine, tmp_path):
    orchestrator, verdicts = engine
    _set_run_folder(orchestrator, data_directory=str(tmp_path))
    orchestrator._procedure = Placeable()  # a run in flight
    _set_run_folder(orchestrator, data_directory="", require_idle=True)
    assert verdicts[-1].code is not ev.VerdictCode.OK
    assert orchestrator._run_folder == str(tmp_path)

    orchestrator._procedure = None
    _set_run_folder(orchestrator, data_directory="", require_idle=True)
    assert verdicts[-1].code is ev.VerdictCode.OK
    assert orchestrator._run_folder == ""


# ----------------------------------------------------------------------
# The session layer
# ----------------------------------------------------------------------


@pytest.fixture
def roster(tmp_path):
    r = UserRoster(tmp_path / "users.json")
    r.add(User(user_id="jdoe", name="J. Doe"))
    return r


@pytest.fixture
def sessions(tmp_path, roster, engine):
    """``(manager, session_store, folder A, folder B)`` with A loaded and B ready."""
    orchestrator, _verdicts = engine
    store = SessionStore(tmp_path / "root")
    a, b = tmp_path / "A", tmp_path / "B"
    store.create_session(a, "A", "jdoe")
    store.create_session(b, "B", "jdoe")
    store.set_active(a)
    store.acquire_lock(a)
    manager = ExperimentManager(
        store=ExperimentStore(a),
        roster=roster,
        orchestrator=orchestrator,
        config_name="sim_cryostat",
        session_store=store,
    )
    return manager, store, a, b


def test_the_subfolder_is_stored_on_the_experiment_and_pushed_down(sessions, engine):
    manager, _store, _a, _b = sessions
    orchestrator, _verdicts = engine
    with pytest.raises(ValueError):
        manager.set_run_subfolder("cd2")  # no experiment open

    record = manager.start_experiment("Hall bar", "jdoe", SAMPLE)
    seen: list[str] = []
    manager.run_folder_changed.connect(seen.append)
    assert manager.set_run_subfolder("cd2\\sweeps") == "cd2/sweeps"
    assert orchestrator._run_subfolder == "cd2/sweeps"
    assert manager.run_folder() == manager.current_data_dir() / "cd2" / "sweeps"
    assert seen[-1] == str(manager.run_folder())
    assert manager.store.load(record.experiment_id).run_subfolder == "cd2/sweeps"
    with pytest.raises(ValueError):
        manager.set_run_subfolder("../out")


def test_an_invalid_stored_subfolder_falls_back_to_data():
    record = ExperimentRecord.from_dict({"experiment_id": "x", "run_subfolder": "../../etc"})
    assert record.run_subfolder == ""
    assert ExperimentRecord.from_dict({"experiment_id": "x"}).run_subfolder == ""


def test_the_floor_comes_from_the_record(sessions, engine):
    manager, _store, _a, _b = sessions
    orchestrator, _verdicts = engine
    manager.start_experiment("Hall bar", "jdoe", SAMPLE)
    manager.current_experiment().runs.append(RunRecord(run_id="run-0041", status="completed"))
    manager.set_run_subfolder("")
    assert orchestrator._run_number_floor == 42


def test_a_run_is_filed_where_it_was_placed_not_where_the_operator_is(sessions):
    manager, _store, _a, _b = sessions
    first = manager.start_experiment("First", "jdoe", SAMPLE)
    data_root = str(manager.current_data_dir())
    manager.close_experiment()
    second = manager.start_experiment("Second", "jdoe", SAMPLE)

    manifest = {"run_id": "run-0001", "procedure": "P", "data_root": data_root, "data_file": ""}
    manager._on_run_started(manifest)
    manager._on_run_finished({**manifest, "status": RUN_STATUS_DONE})

    assert manager.current_experiment().experiment_id == second.experiment_id
    assert manager.current_experiment().runs == []
    filed = manager.store.load(first.experiment_id).find_run("run-0001")
    assert filed is not None and filed.status == RUN_STATUS_DONE


def test_find_run_prefers_the_newest_duplicate():
    record = ExperimentRecord(runs=[RunRecord(run_id="run-0001"), RunRecord(run_id="run-0001", procedure="new")])
    assert record.find_run("run-0001").procedure == "new"


def test_loading_another_session_while_running(sessions, engine):
    manager, store, a, b = sessions
    orchestrator, _verdicts = engine
    manager.start_experiment("In A", "jdoe", SAMPLE)
    # B has an open experiment with a run the app never finished.
    b_store = ExperimentStore(b)
    record = ExperimentRecord(
        experiment_id="001_in_b", title="In B", user_id="jdoe", status="open",
        runs=[RunRecord(run_id="run-0001", status=RUN_STATUS_RUNNING)],
    )
    b_store.save(record)
    b_store.set_active("001_in_b")

    changed: list[str] = []
    experiments: list[dict] = []
    manager.session_changed.connect(changed.append)
    manager.experiment_changed.connect(experiments.append)

    manager.request_session_switch(b)

    assert changed == [str(b.resolve())]
    assert manager.store.root.resolve() == b.resolve()
    assert store.get_active() == b.resolve()
    assert manager.current_experiment().experiment_id == "001_in_b"
    assert experiments[-1]["experiment_id"] == "001_in_b"
    assert manager.current_experiment().find_run("run-0001").status == RUN_STATUS_FAILED
    assert orchestrator._run_folder == str(b_store.data_dir("001_in_b"))
    assert not (a / SESSION_LOCK_FILENAME).exists()
    assert (b / SESSION_LOCK_FILENAME).exists()
    # A's experiment stays open on disk, to be resumed when A is loaded again.
    assert ExperimentStore(a).get_active() is not None


def test_the_engine_refusing_the_release_changes_nothing(sessions, engine):
    manager, store, a, b = sessions
    orchestrator, _verdicts = engine
    manager.start_experiment("In A", "jdoe", SAMPLE)
    failures: list[str] = []
    manager.session_switch_failed.connect(failures.append)
    orchestrator._procedure = Placeable()  # a run the GUI mirror has not seen yet
    try:
        manager.request_session_switch(b)
    finally:
        orchestrator._procedure = None

    assert failures and "run is in progress" in failures[0]
    assert manager.store.root.resolve() == a.resolve()
    assert orchestrator._run_folder == str(manager.current_data_dir())
    assert not (b / SESSION_LOCK_FILENAME).exists()


def test_a_busy_check_refuses_up_front(sessions):
    manager, _store, a, b = sessions
    first = manager.start_experiment("First", "jdoe", SAMPLE)
    manager.add_busy_check(lambda: "An analysis is running")
    with pytest.raises(ValueError, match="analysis"):
        manager.request_session_switch(b)
    with pytest.raises(ValueError, match="analysis"):
        manager.switch_experiment(first.experiment_id)
    assert manager.store.root.resolve() == a.resolve()


def test_a_session_held_by_another_live_process_is_refused(sessions):
    manager, store, _a, b = sessions
    (b / SESSION_LOCK_FILENAME).write_text(
        '{"pid": 1, "host": "' + __import__("socket").gethostname() + '", "since": "x"}'
    )
    if store.lock_holder(b) is None:
        pytest.skip("pid 1 is not visible as a live process here")
    with pytest.raises(SessionLockedError):
        manager.request_session_switch(b)


def test_a_stale_lock_of_this_host_is_taken_over(tmp_path):
    store = SessionStore(tmp_path / "root")
    folder = tmp_path / "S"
    store.create_session(folder, "S", "jdoe")
    (folder / SESSION_LOCK_FILENAME).write_text(
        '{"pid": 999999999, "host": "' + __import__("socket").gethostname() + '", "since": "x"}'
    )
    store.acquire_lock(folder)
    store.release_lock(folder)
    assert not (folder / SESSION_LOCK_FILENAME).exists()


def test_a_result_for_a_session_left_behind_goes_to_that_session(sessions):
    """Notebook outcomes carry their session: same experiment id, other session untouched."""
    from i2as.session.models import ElnLink

    manager, _store, a, b = sessions
    manager.start_experiment("Same id", "jdoe", SAMPLE)
    experiment_id = manager.current_experiment().experiment_id
    other = ExperimentStore(b)
    other.save(ExperimentRecord(experiment_id=experiment_id, title="B's", status="open"))

    manager.set_eln_entry(experiment_id, ElnLink(entry_id="42"), session_root=b)

    assert other.load(experiment_id).eln.entry.entry_id == "42"
    assert manager.current_experiment().eln is None


def test_an_agent_feed_detaches(engine, tmp_path):
    from i2as.session.agent_feed import AgentFeed

    orchestrator, _verdicts = engine
    feed = AgentFeed(tmp_path / "feed.jsonl", "x")
    feed.attach(orchestrator)
    feed.detach(orchestrator)
    feed.detach(orchestrator)  # twice is harmless
    recorded: list[object] = []
    feed.record_verdict = recorded.append  # type: ignore[method-assign]
    orchestrator.submit(ev.Command(name=ev.CommandName.START_MONITORING))
    assert recorded == []
