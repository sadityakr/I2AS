"""Behaviour tests for the analysis runner (i2as/session/analysis_runner.py).

The runner is exercised against a STAND-IN worker: a tiny executable script
written into ``tmp_path`` that reads the spec the runner wrote and produces
whatever this test needs (a good report, a failed one, no report at all, or
nothing ever). Nothing here depends on the real ``i2as.analysis`` worker,
so the two halves of the analysis stage are testable independently — the same
split the ELN track uses between an adapter and its transport.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from i2as.analysis.report import REPORT_FILENAME, SPEC_FILENAME
from i2as.session.eln.settings import AnalysisSettings, ElnSettings

_OK_REPORT = """
report = {
    "run_id": spec["run_id"],
    "recipe": spec["recipe"] or "generic_sweep",
    "recipe_digest": "abc123",
    "status": "ok",
    "summary": ["The sweep completed."],
    "results": [{"name": "Bc", "value": 1.25, "unit": "T"}],
    "figures": [{"file": "overview.png", "caption": "Overview"}],
    "tags": ["sweep"],
    "attach_data_file": spec["attach_data_file"],
    "include_fact_tables": spec["include_fact_tables"],
}
(out / "overview.png").write_bytes(b"PNG")
(out / "report.json").write_text(json.dumps(report))
"""

_FAILED_REPORT = """
report = {
    "run_id": spec["run_id"],
    "recipe": spec["recipe"],
    "status": "failed",
    "error": "ZeroDivisionError: division by zero\\nTraceback ...",
}
(out / "report.json").write_text(json.dumps(report))
"""

_NO_REPORT = """
sys.stderr.write("the recipe exploded before it could write anything\\n")
sys.exit(3)
"""

_NEVER_ENDS = """
import time

while True:
    time.sleep(0.5)
"""


def _worker(tmp_path: Path, body: str, name: str = "worker.py") -> str:
    """Write an executable stand-in worker and return its path.

    Args:
        tmp_path: The test's temporary directory.
        body: The script's own statements; ``spec`` and ``out`` are already
            bound to the parsed spec and its output directory.
        name: The script's file name, so one test can write two workers.

    Returns:
        The script path, to hand to ``AnalysisRunner(python=...)``.
    """
    script = tmp_path / name
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        'spec_path = sys.argv[sys.argv.index("--spec") + 1]\n'
        "spec = json.loads(Path(spec_path).read_text())\n"
        'out = Path(spec["output_dir"])\n'
        f"{body}\n",
        encoding="utf-8",
    )
    os.chmod(script, 0o755)
    return str(script)


class FakePublisher:
    """The publisher seam, recording what the runner handed it."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.reports: list[tuple[str, dict, str]] = []
        self.parked: list[tuple[str, str]] = []

    def export_report(self, run_id, report, report_dir):
        """Record one analysed entry hand-off."""
        self.reports.append((run_id, dict(report.to_dict()), str(report_dir)))
        return True

    def park_facts_entry(self, run_id, warning=""):
        """Record one facts-only fallback."""
        self.parked.append((run_id, warning))
        return True


@pytest.fixture
def runner_setup(tmp_path, qtbot):
    """A real ExperimentManager with two recorded runs, plus a fake publisher.

    Yields ``(manager, publisher, settings_box, make_runner)``, where
    ``settings_box`` is a one-element list the tests mutate to change the
    settings the runner reads, and ``make_runner(python)`` builds the runner
    against a stand-in worker.
    """
    from i2as.core.orchestrator import Orchestrator
    from i2as.core.station import build_station
    from i2as.session.analysis_runner import AnalysisRunner
    from i2as.session.manager import ExperimentManager
    from i2as.session.models import User
    from i2as.session.store import ExperimentStore, UserRoster

    store = ExperimentStore(tmp_path / "experiments")
    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    orchestrator = Orchestrator(build_station("i2as/configs/sim_cryostat"), tick_interval_ms=10)
    manager = ExperimentManager(
        store=store, roster=roster, orchestrator=orchestrator, config_name="sim_cryostat"
    )
    experiment = manager.start_experiment("Sample A", "jdoe", {"sample_name": "A3"})

    for run_id in ("run-0001", "run-0002"):
        data_file = store.data_dir(experiment.experiment_id) / f"{run_id}.h5"
        data_file.parent.mkdir(parents=True, exist_ok=True)
        data_file.write_bytes(b"\x89HDF\r\n\x1a\n")
        started = {
            "run_id": run_id,
            "procedure": "FieldSweep",
            "kind": "run",
            "params": {"field_T": 1.5},
            "data_file": str(data_file),
            "started_utc": "2026-01-01T10:00:00+00:00",
        }
        orchestrator.run_started.emit(started)
        orchestrator.run_finished.emit(
            dict(started, finished_utc="2026-01-01T11:00:00+00:00", status="done", reason="")
        )

    settings_box = [
        ElnSettings(enabled=True, analysis=AnalysisSettings(enabled=True, timeout_s=30.0))
    ]
    publisher = FakePublisher()
    runners: list[AnalysisRunner] = []

    def make_runner(python: str) -> AnalysisRunner:
        runner = AnalysisRunner(manager, publisher, lambda: settings_box[0], python=python)
        runners.append(runner)
        return runner

    yield manager, publisher, settings_box, make_runner
    for runner in runners:
        runner.cancel()


def test_a_finished_analysis_parks_an_analysed_entry(runner_setup, tmp_path, qtbot):
    """The exit criterion: one run, one worker, one report, one parked entry."""
    manager, publisher, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _OK_REPORT))

    with qtbot.waitSignal(runner.analysis_finished, timeout=20000) as blocker:
        report_dir = runner.start("run-0001")

    assert report_dir.endswith(f"analysis{os.sep}run-0001")
    run_id, payload = blocker.args
    assert run_id == "run-0001"
    assert payload["status"] == "ok" and payload["results"][0]["name"] == "Bc"

    assert publisher.parked == [], "a report that ran needs no fallback"
    (parked_run, parked_report, parked_dir) = publisher.reports[0]
    assert parked_run == "run-0001"
    assert parked_report["figures"][0]["file"] == "overview.png"
    assert parked_dir == report_dir
    assert (Path(report_dir) / REPORT_FILENAME).is_file()
    assert not runner.is_running()


def test_the_spec_names_the_run_the_experiment_and_the_preferred_recipe(
    runner_setup, tmp_path, qtbot
):
    """What the worker is asked is built from the record and the settings."""
    manager, _publisher, settings_box, make_runner = runner_setup
    from dataclasses import replace

    settings_box[0] = replace(
        settings_box[0],
        analysis=replace(
            settings_box[0].analysis,
            recipes={"FieldSweep": "my_sweep"},
            include_fact_tables=True,
            attach_data_file=True,
        ),
    )
    runner = make_runner(_worker(tmp_path, _OK_REPORT))

    with qtbot.waitSignal(runner.analysis_finished, timeout=20000):
        report_dir = runner.start("run-0001", options={"window": 5})

    spec = json.loads((Path(report_dir) / SPEC_FILENAME).read_text(encoding="utf-8"))
    assert spec["run_id"] == "run-0001"
    assert spec["recipe"] == "my_sweep", "the settings' per-procedure preference"
    assert spec["manifest"]["procedure"] == "FieldSweep"
    assert spec["experiment"]["experiment_title"] == "Sample A"
    assert spec["experiment"]["sample_info"] == {"sample_name": "A3"}
    assert spec["experiment"]["user_name"] == "J. Doe"
    assert spec["setup"]["config_name"] == "sim_cryostat"
    assert spec["options"] == {"window": 5}
    assert spec["include_fact_tables"] is True and spec["attach_data_file"] is True
    assert spec["data_path"].endswith("run-0001.h5")
    assert spec["recipe_dirs"] == runner.recipe_dirs()
    assert spec["recipe_dirs"][0].endswith(f"analysis{os.sep}recipes")


def test_an_explicit_recipe_wins_over_the_settings(runner_setup, tmp_path, qtbot):
    """A caller naming a recipe overrides the per-procedure preference."""
    _manager, _publisher, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _OK_REPORT))

    with qtbot.waitSignal(runner.analysis_finished, timeout=20000):
        report_dir = runner.start("run-0001", recipe="explicit")

    spec = json.loads((Path(report_dir) / SPEC_FILENAME).read_text(encoding="utf-8"))
    assert spec["recipe"] == "explicit"


def test_a_failed_report_parks_the_facts_entry(runner_setup, tmp_path, qtbot):
    """A recipe that raised loses nothing: the facts entry waits instead."""
    _manager, publisher, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _FAILED_REPORT))

    with qtbot.waitSignal(runner.analysis_failed, timeout=20000) as blocker:
        runner.start("run-0001")

    run_id, error = blocker.args
    assert run_id == "run-0001" and "ZeroDivisionError" in error
    assert publisher.reports == [], "a failed report is never an analysed entry"
    assert publisher.parked == [("run-0001", "ZeroDivisionError: division by zero")]


def test_a_worker_that_writes_no_report_is_still_an_entry(runner_setup, tmp_path, qtbot):
    """No report is a failure like any other — never a silently lost run."""
    _manager, publisher, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _NO_REPORT))

    with qtbot.waitSignal(runner.analysis_failed, timeout=20000) as blocker:
        runner.start("run-0001")

    _run_id, error = blocker.args
    assert "wrote no report" in error
    assert "exploded" in error, "the worker's own stderr says why"
    assert publisher.parked[0][0] == "run-0001"


def test_a_runaway_worker_is_killed_and_reported(runner_setup, tmp_path, qtbot):
    """The timeout bounds a recipe that never returns; the entry still lands."""
    from dataclasses import replace

    _manager, publisher, settings_box, make_runner = runner_setup
    settings_box[0] = replace(
        settings_box[0], analysis=replace(settings_box[0].analysis, timeout_s=1.0)
    )
    runner = make_runner(_worker(tmp_path, _NEVER_ENDS))

    with qtbot.waitSignal(runner.analysis_failed, timeout=30000) as blocker:
        runner.start("run-0001")

    _run_id, error = blocker.args
    assert "timed out after 1 s" in error
    assert publisher.parked == [("run-0001", "analysis timed out after 1 s")]
    assert not runner.is_running()


def test_one_worker_at_a_time_and_the_queue_is_fifo(runner_setup, tmp_path, qtbot):
    """Two requests, one process: the second waits for the first, in order."""
    _manager, publisher, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _OK_REPORT))

    with qtbot.waitSignals(
        [runner.analysis_finished, runner.analysis_finished], timeout=30000
    ):
        first = runner.start("run-0001")
        second = runner.start("run-0002")
        assert first and second
        assert runner.is_running("run-0001") and runner.is_running("run-0002")

    assert [entry[0] for entry in publisher.reports] == ["run-0001", "run-0002"]
    assert not runner.is_running()


def test_cancelling_leaves_the_facts_entry_behind(runner_setup, tmp_path, qtbot):
    """A cancelled analysis still ends in an entry waiting for its human."""
    _manager, publisher, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _NEVER_ENDS))

    with qtbot.waitSignal(runner.analysis_started, timeout=20000):
        runner.start("run-0001")

    with qtbot.waitSignal(runner.analysis_failed, timeout=20000) as blocker:
        runner.cancel("run-0001")

    assert "cancelled" in blocker.args[1]
    assert publisher.parked == [("run-0001", "analysis cancelled")]
    assert not runner.is_running()


def test_start_refuses_what_it_cannot_analyse(runner_setup, tmp_path, qtbot):
    """No run, no data file, or no open experiment: "" and a log line, never a raise."""
    manager, publisher, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _OK_REPORT))

    assert runner.start("no-such-run") == ""

    run = manager.current_experiment().find_run("run-0002")
    run.data_file = ""
    assert runner.start("run-0002") == ""

    manager.close_experiment()
    assert runner.start("run-0001") == ""
    assert runner.recipe_dirs() == []
    assert publisher.reports == [] and publisher.parked == []
