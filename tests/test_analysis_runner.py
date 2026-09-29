"""Behaviour tests for the analysis runner (i2as/session/analysis_runner.py).

The runner is exercised against a STAND-IN worker: a tiny executable script
written into ``tmp_path`` that reads the spec the runner wrote and produces
whatever this test needs (a good report, a failed one, no report at all, or
nothing ever). Nothing here depends on the real ``i2as.analysis`` worker,
so the two halves of the analysis stage are testable independently. What the
runner leaves is asserted on the sealed **analysis bundle** it announces on
``bundle_ready`` and on the run's selection — the runner knows no notebook.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from i2as.analysis.bundle import BUNDLE_FILENAME, read_bundle
from i2as.analysis.report import REPORT_FILENAME, SPEC_FILENAME
from i2as.session.analysis_sandbox import LaunchPlan, SandboxError, SubprocessSandbox
from i2as.session.app_config import AnalysisSettings, SandboxSettings

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
        The script path, to hand to ``make_runner()``.
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


class StandInSandbox(SubprocessSandbox):
    """Runs a stand-in worker SCRIPT with this interpreter, in place of the container.

    ``sys.executable <script> --spec <path>`` rather than executing the script
    through its shebang, so it starts on Windows as well as on POSIX.

    Args:
        script: The stand-in worker ``_worker()`` wrote.
        stop_command: What the plan names as its stop command.
    """

    def __init__(self, script: str, stop_command: tuple[str, ...] = ()) -> None:
        super().__init__(sys.executable)
        self.script = script
        self.stop_command = stop_command

    def launch(self, spec_path: Path) -> LaunchPlan:
        """Plan ``python <script> --spec <spec_path>``."""
        return LaunchPlan(
            program=sys.executable,
            arguments=(self.script, "--spec", str(spec_path)),
            backend="stand-in",
            stop_command=self.stop_command,
        )


class BundleLog:
    """Every ``bundle_ready`` the runner announced, in order."""

    def __init__(self) -> None:
        self.ready: list[tuple[str, str, dict]] = []

    def record(self, run_id: str, bundle_id: str, bundle: dict) -> None:
        self.ready.append((run_id, bundle_id, bundle))

    @property
    def ok(self) -> list[tuple[str, str, dict]]:
        return [item for item in self.ready if item[2]["status"] == "ok"]

    @property
    def failed(self) -> list[tuple[str, str]]:
        return [(item[0], item[2]["error"].splitlines()[0]) for item in self.ready if item[2]["status"] == "failed"]


@pytest.fixture
def runner_setup(tmp_path, qtbot):
    """A real ExperimentManager with two recorded runs, plus a bundle log.

    Yields ``(manager, bundles, settings_box, make_runner)``, where
    ``settings_box`` is a one-element list the tests mutate to change the
    ``AnalysisSettings`` the runner reads, and ``make_runner(script)`` builds
    the runner against a stand-in worker (or ``make_runner(sandbox=...)``
    against any sandbox factory).
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

    settings_box = [AnalysisSettings(enabled=True, timeout_s=30.0)]
    bundles = BundleLog()
    runners: list[AnalysisRunner] = []

    def make_runner(script: str = "", sandbox=None) -> AnalysisRunner:
        factory = sandbox or (lambda _settings: StandInSandbox(script))
        runner = AnalysisRunner(manager, lambda: settings_box[0], sandbox_factory=factory)
        runner.bundle_ready.connect(bundles.record)
        runners.append(runner)
        return runner

    yield manager, bundles, settings_box, make_runner
    for runner in runners:
        runner.cancel()


def test_a_finished_analysis_is_sealed_as_a_bundle_and_selected(runner_setup, tmp_path, qtbot):
    """The exit criterion: one run, one worker, one sealed bundle, and it represents the run."""
    manager, bundles, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _OK_REPORT))

    with qtbot.waitSignal(runner.bundle_ready, timeout=20000):
        bundle_dir = runner.start("run-0001")

    folder = Path(bundle_dir)
    assert folder.parent.name == "run-0001" and folder.parent.parent.name == "analysis"
    assert (folder / REPORT_FILENAME).is_file() and (folder / BUNDLE_FILENAME).is_file()
    run_id, bundle_id, payload = bundles.ready[0]
    assert (run_id, bundle_id) == ("run-0001", folder.name)
    assert payload["status"] == "ok" and payload["results"][0]["name"] == "Bc"
    assert payload["producer"]["kind"] == "recipe" and payload["producer"]["digest"] == "abc123"
    assert payload["inputs"][0]["run_id"] == "run-0001"
    [figure] = payload["artifacts"]
    assert figure["path"] == "overview.png" and figure["caption"] == "Overview" and len(figure["sha256"]) == 64
    assert read_bundle(folder).sealed is True
    assert manager.current_experiment().find_run("run-0001").selected_bundle == bundle_id
    assert not runner.is_running()


def test_every_analysis_is_a_new_bundle(runner_setup, tmp_path, qtbot):
    """A second analysis of a run never overwrites the first."""
    manager, bundles, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _OK_REPORT))
    with qtbot.waitSignals([runner.bundle_ready, runner.bundle_ready], timeout=30000):
        first = runner.start("run-0001")
        second = runner.start("run-0001", recipe="other")
    assert first != second and Path(first).is_dir() and Path(second).is_dir()
    assert manager.current_experiment().find_run("run-0001").selected_bundle == Path(second).name
    assert len(manager.store.list_bundles(manager.current_experiment().experiment_id, "run-0001")) == 2


def test_the_spec_names_the_run_the_experiment_and_the_preferred_recipe(
    runner_setup, tmp_path, qtbot
):
    """What the worker is asked is built from the record and the settings."""
    manager, _bundles, settings_box, make_runner = runner_setup
    from dataclasses import replace

    settings_box[0] = replace(
        settings_box[0],
        recipes={"FieldSweep": "my_sweep"},
        include_fact_tables=True,
        attach_data_file=True,
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
    _manager, _bundles, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _OK_REPORT))

    with qtbot.waitSignal(runner.analysis_finished, timeout=20000):
        report_dir = runner.start("run-0001", recipe="explicit")

    spec = json.loads((Path(report_dir) / SPEC_FILENAME).read_text(encoding="utf-8"))
    assert spec["recipe"] == "explicit"


def test_a_failed_report_is_a_failed_bundle(runner_setup, tmp_path, qtbot):
    """A recipe that raised is a failed bundle, and selects nothing."""
    manager, bundles, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _FAILED_REPORT))

    with qtbot.waitSignal(runner.analysis_failed, timeout=20000) as blocker:
        runner.start("run-0001")

    run_id, error = blocker.args
    assert run_id == "run-0001" and "ZeroDivisionError" in error
    assert bundles.ok == [], "a failed report is never a completed bundle"
    assert bundles.failed == [("run-0001", "ZeroDivisionError: division by zero")]
    assert manager.current_experiment().find_run("run-0001").selected_bundle == ""


def test_a_worker_that_writes_no_report_is_still_a_bundle(runner_setup, tmp_path, qtbot):
    """No report is a failure like any other — never a silently lost run."""
    _manager, bundles, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _NO_REPORT))

    with qtbot.waitSignal(runner.analysis_failed, timeout=20000) as blocker:
        runner.start("run-0001")

    _run_id, error = blocker.args
    assert "wrote no report" in error
    assert "exploded" in error, "the worker's own stderr says why"
    assert bundles.failed[0][0] == "run-0001"


def test_a_runaway_worker_is_killed_and_reported(runner_setup, tmp_path, qtbot):
    """The timeout bounds a recipe that never returns; the entry still lands."""
    from dataclasses import replace

    _manager, bundles, settings_box, make_runner = runner_setup
    settings_box[0] = replace(settings_box[0], timeout_s=1.0)
    runner = make_runner(_worker(tmp_path, _NEVER_ENDS))

    with qtbot.waitSignal(runner.analysis_failed, timeout=30000) as blocker:
        runner.start("run-0001")

    _run_id, error = blocker.args
    assert "timed out after 1 s" in error
    assert bundles.failed == [("run-0001", "analysis timed out after 1 s")]
    assert not runner.is_running()


def test_one_worker_at_a_time_and_the_queue_is_fifo(runner_setup, tmp_path, qtbot):
    """Two requests, one process: the second waits for the first, in order."""
    _manager, bundles, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _OK_REPORT))

    with qtbot.waitSignals(
        [runner.analysis_finished, runner.analysis_finished], timeout=30000
    ):
        first = runner.start("run-0001")
        second = runner.start("run-0002")
        assert first and second
        assert runner.is_running("run-0001") and runner.is_running("run-0002")

    assert [entry[0] for entry in bundles.ok] == ["run-0001", "run-0002"]
    assert not runner.is_running()


def test_cancelling_leaves_a_failed_bundle_behind(runner_setup, tmp_path, qtbot):
    """A cancelled analysis still says what happened to it."""
    _manager, bundles, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _NEVER_ENDS))

    with qtbot.waitSignal(runner.analysis_started, timeout=20000):
        runner.start("run-0001")

    with qtbot.waitSignal(runner.analysis_failed, timeout=20000) as blocker:
        runner.cancel("run-0001")

    assert "cancelled" in blocker.args[1]
    assert bundles.failed == [("run-0001", "analysis cancelled")]
    assert not runner.is_running()


def test_start_refuses_what_it_cannot_analyse(runner_setup, tmp_path, qtbot):
    """No run, no data file, or no open experiment: "" and a log line, never a raise."""
    manager, bundles, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _OK_REPORT))

    assert runner.start("no-such-run") == ""

    run = manager.current_experiment().find_run("run-0002")
    run.data_file = ""
    assert runner.start("run-0002") == ""

    manager.close_experiment()
    assert runner.start("run-0001") == ""
    assert runner.recipe_dirs() == []
    assert bundles.ready == []


# ── Analysis scripts and the sandbox ──────────────────────────────────────

_ENV_REPORT = (
    "import os\n"
    '(out / "env.json").write_text(json.dumps({"env": dict(os.environ), "cwd": os.getcwd()}))\n'
    + _OK_REPORT
)


def test_a_script_is_announced_on_its_own_signal_and_selects_nothing(
    runner_setup, tmp_path, qtbot
):
    """Exploring a run leaves what represents the run exactly as it was."""
    manager, bundles, _settings, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _OK_REPORT))
    experiment = manager.current_experiment()
    folder = manager.store.script_dir(experiment.experiment_id, "run-0001", "probe_1")
    folder.mkdir(parents=True)
    script = folder / "probe_1.py"
    script.write_text("report.summary('hi')\n", encoding="utf-8")
    finished_recipes: list = []
    runner.analysis_finished.connect(lambda *args: finished_recipes.append(args))

    with qtbot.waitSignal(runner.script_finished, timeout=20000) as blocker:
        output = runner.start_script("run-0001", "probe_1", script, options={"k": 1})
        assert runner.is_running("run-0001", "probe_1")
        assert not runner.is_running("run-0001"), "the run's recipe analysis is not running"

    assert output == str(folder)
    run_id, script_id, payload = blocker.args
    assert (run_id, script_id) == ("run-0001", "probe_1")
    assert payload["status"] == "ok"
    assert [(r, b) for r, b, _ in bundles.ready] == [("run-0001", "script-probe_1")]
    assert bundles.ready[0][2]["producer"]["kind"] == "script"
    assert manager.current_experiment().find_run("run-0001").selected_bundle == ""
    assert finished_recipes == []
    spec = json.loads((folder / SPEC_FILENAME).read_text(encoding="utf-8"))
    assert spec["script_path"] == str(script)
    assert spec["recipe"] == ""
    assert spec["options"] == {"k": 1}




def test_a_runaway_worker_runs_its_sandbox_stop_command(runner_setup, tmp_path, qtbot):
    """A container outlives a killed ``docker run``; the runner stops it by name."""
    from dataclasses import replace

    _manager, _bundles, settings_box, make_runner = runner_setup
    settings_box[0] = replace(settings_box[0], timeout_s=1.0)
    marker = tmp_path / "stopped.txt"
    stop = (sys.executable, "-c", f"open({str(marker)!r}, 'w').write('killed')")
    worker = _worker(tmp_path, _NEVER_ENDS)
    runner = make_runner(sandbox=lambda _settings: StandInSandbox(worker, stop_command=stop))

    with qtbot.waitSignal(runner.analysis_failed, timeout=30000):
        runner.start("run-0001")

    qtbot.waitUntil(marker.exists, timeout=10000)
    assert marker.read_text(encoding="utf-8") == "killed"


def test_the_runner_builds_each_sandbox_from_the_settings(runner_setup, tmp_path, qtbot):
    """The analysis.sandbox block is what the sandbox is built from, per analysis."""
    from dataclasses import replace

    _manager, _bundles, settings_box, make_runner = runner_setup
    settings_box[0] = replace(settings_box[0], sandbox=SandboxSettings(image="lab/a:9"))
    seen: list[SandboxSettings] = []
    worker = _worker(tmp_path, _OK_REPORT)

    def factory(settings: SandboxSettings) -> StandInSandbox:
        seen.append(settings)
        return StandInSandbox(worker)

    runner = make_runner(sandbox=factory)
    with qtbot.waitSignal(runner.analysis_finished, timeout=20000):
        runner.start("run-0001")

    assert [settings.image for settings in seen] == ["lab/a:9"]


def test_no_container_engine_is_a_failed_analysis_not_a_silent_one(
    runner_setup, tmp_path, qtbot
):
    """With the real container sandbox and no engine, a failed bundle names why."""
    from dataclasses import replace

    from i2as.session.analysis_sandbox import build_sandbox

    _manager, bundles, settings_box, make_runner = runner_setup
    settings_box[0] = replace(
        settings_box[0], sandbox=SandboxSettings(engine="no-such-engine-xyz")
    )
    runner = make_runner(sandbox=build_sandbox)

    with qtbot.waitSignal(runner.analysis_failed, timeout=5000) as blocker:
        runner.start("run-0001")

    assert blocker.args[0] == "run-0001"
    assert "not installed" in blocker.args[1]
    assert bundles.failed and bundles.failed[0][0] == "run-0001"
    assert not runner.is_running()


class _PreparedOnlySandbox:
    """A real ContainerSandbox's ``prepare()``, with a launch that refuses."""

    def __init__(self, settings: SandboxSettings) -> None:
        from i2as.session.analysis_sandbox import ContainerSandbox

        self._inner = ContainerSandbox(settings, engine_path="unused")

    def prepare(self, spec):
        return self._inner.prepare(spec)

    def launch(self, spec_path):
        raise SandboxError("no engine on a test box")


def test_the_container_spec_names_container_paths(runner_setup, tmp_path, qtbot):
    """The spec written for a container names /work and /input, not host paths."""
    _manager, _bundles, _settings, make_runner = runner_setup
    runner = make_runner(sandbox=_PreparedOnlySandbox)

    with qtbot.waitSignal(runner.analysis_failed, timeout=5000):
        report_dir = runner.start("run-0001")

    spec = json.loads((Path(report_dir) / SPEC_FILENAME).read_text(encoding="utf-8"))
    assert spec["output_dir"] == "/work"
    assert spec["data_path"] == "/input/run-0001.h5"


# ── The analysis trigger: analysis decides, by itself, what is analysed ────


def test_the_trigger_analyses_a_finished_run_only_when_switched_on(runner_setup, tmp_path, qtbot):
    """A finished run is analysed when the analysis settings say so, and only then."""
    import gc
    from dataclasses import replace

    from i2as.session.analysis_trigger import AnalysisTrigger

    manager, bundles, settings_box, make_runner = runner_setup
    runner = make_runner(_worker(tmp_path, _OK_REPORT))
    manifest = {"run_id": "run-0001", "data_file": manager.current_experiment().find_run("run-0001").data_file}

    trigger = AnalysisTrigger(manager, runner, lambda: settings_box[0])
    assert trigger.parent() is runner, "owned by the runner, so it outlives its builder"
    del trigger
    gc.collect()
    [child] = [c for c in runner.children() if isinstance(c, AnalysisTrigger)]

    settings_box[0] = replace(settings_box[0], enabled=False)
    assert child.on_run_finished(manifest) == "", "analysis off: the run is not analysed"
    settings_box[0] = replace(settings_box[0], enabled=True)
    with qtbot.waitSignal(runner.bundle_ready, timeout=20000):
        assert child.on_run_finished(manifest)
    assert bundles.ok[0][0] == "run-0001"

    manager.close_experiment()
    assert child.on_run_finished(manifest) == "", "no experiment: nowhere to keep a bundle"
    assert child.on_run_finished("junk") == ""
