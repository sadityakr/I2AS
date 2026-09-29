"""End-to-end: a finished run → the real analysis worker → a sealed bundle
→ one approved publish → the experiment's (sim) notebook page.

The layer suites test each half against a stand-in. This test wires the REAL
pieces together the way ``i2as.main`` does — ``AnalysisTrigger`` →
``AnalysisRunner.start`` → ``python -m i2as.analysis run`` → the sealed,
selected bundle — and then the notebook layer, which knows only the bundle:
link the experiment to its page, approve once, publish — over a real HDF5 run
file written by the data manager, and asserts that the analysed, concise
section with its figures reaches the page.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from i2as.analysis.report import AnalysisReport
from i2as.core.data_manager import DataManager
from i2as.core.orchestrator import Orchestrator
from i2as.core.station import build_station
from i2as.session.analysis_runner import AnalysisRunner
from i2as.session.analysis_sandbox import SubprocessSandbox
from i2as.session.analysis_trigger import AnalysisTrigger
from i2as.session.app_config import AnalysisSettings
from i2as.session.manager import ExperimentManager
from i2as.session.models import User
from i2as.session.store import ExperimentStore, UserRoster
from tests.notebook_support import notebook_service

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX subprocess semantics assumed"
)

_DATA_CONFIG = {
    "sweep_columns": {"unix_time": "float", "field_T": "float"},
    "measurement_scalars": {"voltage_V": "float"},
    "measurement_arrays": {},
    "measurement_blocks": {},
    "loop_shape": [1, 1],
}


def _write_run_file(directory: Path, n_points: int = 6) -> Path:
    """Write a small closed sweep run file and return its path."""
    writer = DataManager(
        data_directory=str(directory),
        procedure_name="Field Sweep",
        procedure_params={"field_start": -1.0, "field_end": 1.0},
        sample_info={"sample_name": "A3"},
        instrument_state={},
        system_targets={},
        measurement_commands=[],
        data_config=_DATA_CONFIG,
        n_sweep_points=n_points,
        experiment_info={"setup": {"config_name": "sim_cryostat"}, "experiment": {}},
    )
    for index in range(n_points):
        writer.save_datapoint(
            index,
            {
                "unix_time": 1_000.0 + index,
                "field_T": -1.0 + 0.4 * index,
                "voltage_V": [[0.1 * index]],
            },
            {},
        )
    writer.close()
    return Path(writer.filepath)


_IMAGE_SHAPE = (12, 16)

_IMAGE_DATA_CONFIG = {
    "sweep_columns": {"unix_time": "float", "field_T": "float"},
    "measurement_scalars": {"roi_mean": "float", "roi_mean_error": "float", "roi_std": "float"},
    "measurement_arrays": {"roi_mean_array": 1},
    "measurement_blocks": {"frame": _IMAGE_SHAPE},
    "measurement_block_labels": {},
    "measurement_image_blocks": {"frame": {"unit": "counts", "description": "frame"}},
    "loop_shape": [1, 1],
}


def _write_imaging_run_file(directory: Path, n_points: int = 9) -> Path:
    """Write a small closed Field Imaging run file — frames that switch — and return its path."""
    import numpy as np

    writer = DataManager(
        data_directory=str(directory),
        procedure_name="Field Imaging",
        procedure_params={"field_start": -1.0, "field_end": 1.0, "saturation_field_T": -1.5},
        sample_info={"sample_name": "D1"},
        instrument_state={},
        system_targets={},
        measurement_commands=[],
        data_config=_IMAGE_DATA_CONFIG,
        n_sweep_points=n_points,
        experiment_info={"setup": {"config_name": "sim_imaging"}, "experiment": {}},
    )
    field = np.concatenate([np.linspace(-1.0, 1.0, 5), np.linspace(0.5, -1.0, n_points - 5)])
    magnetisation = -1.0
    for index in range(n_points):
        h = float(field[index])
        going_up = index == 0 or field[index] >= field[index - 1]
        if going_up and h > 0.4:
            magnetisation = 1.0
        if not going_up and h < -0.4:
            magnetisation = -1.0
        frame = np.full(_IMAGE_SHAPE, 100.0 + 50.0 * magnetisation)
        frame[: _IMAGE_SHAPE[0] // 2] += 5.0 * index
        writer.save_datapoint(
            index,
            {
                "unix_time": 1_000.0 + index,
                "field_T": h,
                "frame": frame,
                "roi_mean_array": [[[float(frame.mean())]]],
                "roi_mean": [[float(frame.mean())]],
                "roi_mean_error": [[0.0]],
                "roi_std": [[float(frame.std())]],
            },
            {},
        )
    writer.close()
    return Path(writer.filepath)


def _wire(tmp_path, monkeypatch, *, config_name: str, procedure: str, write_run, params):
    """The production wiring over one real run file and a sim notebook."""
    monkeypatch.setenv("PYTHONPATH", os.getcwd())
    store = ExperimentStore(tmp_path / "experiments")
    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    orchestrator = Orchestrator(
        build_station(f"i2as/configs/{config_name}"), tick_interval_ms=10
    )
    manager = ExperimentManager(
        store=store, roster=roster, orchestrator=orchestrator, config_name=config_name
    )
    experiment = manager.start_experiment("Sample A", "jdoe", {"sample_name": "A3"})
    data_file = write_run(store.data_dir(experiment.experiment_id))

    started = {
        "run_id": "run-0001",
        "procedure": procedure,
        "kind": "run",
        "params": params,
        "data_file": str(data_file),
        "started_utc": "2026-01-01T10:00:00+00:00",
    }
    orchestrator.run_started.emit(started)
    finished = dict(started, finished_utc="2026-01-01T10:05:00+00:00", status="done", reason="")

    analysis = AnalysisSettings(enabled=True, timeout_s=120.0)
    # A plain child process stands in for the container: no engine on a test box.
    runner = AnalysisRunner(manager, lambda: analysis, sandbox_factory=lambda _settings: SubprocessSandbox())
    trigger = AnalysisTrigger(manager, runner, lambda: analysis)
    orchestrator.run_finished.connect(trigger.on_run_finished)
    service, notebook = notebook_service(manager, tmp_path)
    service.link_experiment("lab")
    manager.approve_eln_publishing("jdoe")

    return manager, service, notebook, runner, orchestrator, finished, experiment.experiment_id


@pytest.fixture
def wired(tmp_path, qtbot, monkeypatch):
    """The transport example: a Field Sweep run on the sim cryostat."""
    parts = _wire(
        tmp_path,
        monkeypatch,
        config_name="sim_cryostat",
        procedure="Field Sweep",
        write_run=_write_run_file,
        params={"field_start": -1.0, "field_end": 1.0},
    )
    yield parts
    parts[3].cancel()
    parts[1].stop()


def _page(notebook):
    (page,) = notebook()["entries"].values()
    return page


@pytest.fixture
def wired_imaging(tmp_path, qtbot, monkeypatch):
    """The imaging example: a Field Imaging run on the sim imaging station."""
    parts = _wire(
        tmp_path,
        monkeypatch,
        config_name="sim_imaging",
        procedure="Field Imaging",
        write_run=_write_imaging_run_file,
        params={"field_start": -1.0, "field_end": 1.0, "saturation_field_T": -1.5},
    )
    yield parts
    parts[3].cancel()
    parts[1].stop()


def test_a_finished_run_reaches_the_page_as_an_analysed_section(wired, qtbot):
    """Run end → worker → sealed, selected bundle → one publish → one section with its figure."""
    manager, service, notebook, runner, orchestrator, finished, experiment_id = wired

    with qtbot.waitSignal(runner.bundle_ready, timeout=90_000) as blocker:
        orchestrator.run_finished.emit(finished)

    run_id, bundle_id, payload = blocker.args
    assert run_id == "run-0001" and payload["status"] == "ok", payload.get("error")
    assert payload["producer"]["name"] == "generic_sweep"
    assert manager.current_experiment().find_run(run_id).selected_bundle == bundle_id
    figures = [a["path"] for a in payload["artifacts"] if a["kind"] == "figure"]
    assert figures, "the generic sweep recipe draws one overview figure"
    assert _page(notebook)["body"] == "", "analysis never publishes anything"

    service.publish()

    page = _page(notebook)
    assert page["body"].count("<h2>I2AS") == 1 and run_id in page["body"]
    assert "<img" not in page["body"], "a figure travels as an upload, never embedded"
    assert [u["name"] for u in page["uploads"]] == [f"{run_id}_{name}" for name in figures]
    assert manager.current_experiment().find_run(run_id).published


def test_a_failing_recipe_leaves_a_failed_bundle_and_the_run_is_published_from_its_facts(wired, qtbot):
    """A broken experiment recipe never loses the run: it is still published, from its facts."""
    manager, service, notebook, runner, orchestrator, finished, experiment_id = wired
    recipes_dir = manager.store.recipes_dir(experiment_id)
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "broken.py").write_text(
        "from i2as.analysis.base import AnalysisRecipe\n"
        "class Broken(AnalysisRecipe):\n"
        "    name = 'broken'\n"
        "    procedures = ('Field Sweep',)\n"
        "    description = 'raises'\n"
        "    def analyse(self, run, context):\n"
        "        raise ZeroDivisionError('boom')\n",
        encoding="utf-8",
    )

    with qtbot.waitSignal(runner.analysis_failed, timeout=90_000) as blocker:
        orchestrator.run_finished.emit(finished)

    run_id, error = blocker.args
    assert "ZeroDivisionError" in error
    assert manager.current_experiment().find_run(run_id).selected_bundle == ""
    [bundle] = manager.store.list_bundles(experiment_id, run_id)
    assert not bundle.ok and "ZeroDivisionError" in bundle.error
    service.publish()
    assert "Parameters" in _page(notebook)["body"], "the run is presented from its facts"

def test_a_finished_imaging_run_reaches_the_page_with_its_montage(wired_imaging, qtbot):
    """The imaging twin: run end → worker → the image-stack bundle → the montage uploaded."""
    manager, service, notebook, runner, orchestrator, finished, experiment_id = wired_imaging

    with qtbot.waitSignal(runner.bundle_ready, timeout=90_000) as blocker:
        orchestrator.run_finished.emit(finished)

    run_id, bundle_id, payload = blocker.args
    report = AnalysisReport.from_dict(json.loads((manager.store.bundle_dir(experiment_id, run_id, bundle_id) / "report.json").read_text()))
    assert report.ok, report.error
    assert report.recipe == "field_image_stack", "the procedure-specific recipe wins"
    assert [f.file for f in report.figures] == ["montage.png", "difference.png", "loop.png"]
    assert any(r.name == "Coercive field" for r in report.results)

    service.publish()

    page = _page(notebook)
    assert "reference frame" in page["body"]
    assert sorted(u["name"] for u in page["uploads"]) == sorted(
        f"{run_id}_{name}" for name in ("montage.png", "difference.png", "loop.png")
    )
