"""End-to-end: a finished run → the real analysis worker → an analysed entry
→ human approval → the (sim) notebook.

The layer suites test each half against a stand-in (the runner against a fake
worker, the worker against a spec file, the publisher against a fake report).
This test wires the REAL pieces together the way ``i2as.main`` does —
``ElnPublisher.analysis_requested`` → ``AnalysisRunner.start`` →
``python -m i2as.analysis run`` → ``export_report`` → the pending entry →
``approve_eln_draft`` → the outbox drain — over a real HDF5 run file written
by the data manager, and asserts that only the analysed, concise entry with
its figure reaches the notebook.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from i2as.analysis.report import REPORT_FILENAME, AnalysisReport
from i2as.core.data_manager import DataManager
from i2as.core.orchestrator import Orchestrator
from i2as.core.station import build_station
from i2as.session.analysis_runner import AnalysisRunner
from i2as.session.eln.outbox import DRAIN_PUBLISHED
from i2as.session.eln.publisher import ElnPublisher
from i2as.session.eln.settings import AnalysisSettings, ElnSettings
from i2as.session.eln.sim_eln import SimElnAdapter
from i2as.session.manager import ExperimentManager
from i2as.session.models import User
from i2as.session.store import ExperimentStore, UserRoster

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

    settings = ElnSettings(
        enabled=True,
        backend="sim_eln",
        base_url="https://sim.example",
        api_key="k",
        retry_base_s=0.0,
        retry_max_s=0.0,
        analysis=AnalysisSettings(enabled=True, timeout_s=120.0),
    )
    adapter = SimElnAdapter({})
    publisher = ElnPublisher(manager, settings, adapter=adapter)
    manager.attach_eln_publisher(publisher)
    orchestrator.run_finished.connect(publisher.on_run_finished)
    runner = AnalysisRunner(manager, publisher, lambda: publisher.settings)
    publisher.analysis_requested.connect(runner.start)

    return manager, publisher, adapter, runner, orchestrator, finished, experiment.experiment_id


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


def test_a_finished_run_reaches_the_notebook_as_an_analysed_entry(wired, qtbot):
    """Run end → worker → parked entry → approval → one entry with one figure."""
    manager, publisher, adapter, runner, orchestrator, finished, experiment_id = wired

    with qtbot.waitSignal(runner.analysis_finished, timeout=90_000) as blocker:
        orchestrator.run_finished.emit(finished)

    # Nothing was queued at run end: analysis is on, so approval is the gate.
    assert publisher.pending_count() == 0
    run_id, payload = blocker.args
    assert run_id == "run-0001"
    report = AnalysisReport.from_dict(payload)
    assert report.ok, report.error
    assert report.recipe == "generic_sweep"
    report_dir = manager.store.report_dir(experiment_id, run_id)
    assert (report_dir / REPORT_FILENAME).is_file()
    assert report.figures, "the generic sweep recipe draws one overview figure"
    figure = report_dir / report.figures[0].file
    assert figure.is_file() and figure.stat().st_size > 0

    pending = manager.pending_eln_draft(run_id)
    assert pending["source"] == "analysis"
    assert "Provenance" in pending["body_html"] or "provenance" in pending["body_html"].lower()
    assert "<img" not in pending["body_html"], "the published body never embeds an image"
    assert "Parameters" not in pending["body_html"], "fact tables are opt-in"
    assert [a["path"] for a in pending["attachments"]] == [str(figure)]

    # The human approves in the eLab tab; the ordinary drain publishes it.
    job_id = manager.approve_eln_draft(run_id)
    assert job_id
    assert publisher.drain_once().state == DRAIN_PUBLISHED
    assert len(adapter.entries) == 1
    entry = next(iter(adapter.entries.values()))
    assert entry["body_html"] == pending["body_html"]
    uploaded = [Path(u["path"]).name for u in adapter.uploads]
    assert uploaded == [figure.name], "the figure is attached, the raw data file is not"
    assert manager.pending_eln_draft(run_id) == {}
    assert manager.current_experiment().find_run(run_id).eln_link is not None


def test_a_failing_recipe_parks_a_facts_only_entry(wired, qtbot):
    """A broken experiment recipe never loses the run: facts fall back, flagged."""
    manager, publisher, _adapter, runner, orchestrator, finished, experiment_id = wired
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
    pending = manager.pending_eln_draft(run_id)
    assert pending["source"] == "facts"
    assert "ZeroDivisionError" in pending["body_html"]
    assert publisher.pending_count() == 0, "nothing publishes without approval"


def test_a_finished_imaging_run_reaches_the_notebook_with_its_montage(wired_imaging, qtbot):
    """The imaging twin: run end → worker → the image-stack report → the montage attached."""
    manager, publisher, adapter, runner, orchestrator, finished, experiment_id = wired_imaging

    with qtbot.waitSignal(runner.analysis_finished, timeout=90_000) as blocker:
        orchestrator.run_finished.emit(finished)

    run_id, payload = blocker.args
    report = AnalysisReport.from_dict(payload)
    assert report.ok, report.error
    assert report.recipe == "field_image_stack", "the procedure-specific recipe wins"
    assert [f.file for f in report.figures] == ["montage.png", "difference.png", "loop.png"]
    report_dir = manager.store.report_dir(experiment_id, run_id)
    for figure in report.figures:
        assert (report_dir / figure.file).stat().st_size > 0
    assert any(r.name == "Coercive field" for r in report.results)

    pending = manager.pending_eln_draft(run_id)
    assert pending["source"] == "analysis"
    assert "reference frame" in pending["body_html"]
    assert [Path(a["path"]).name for a in pending["attachments"]] == [
        "montage.png", "difference.png", "loop.png"
    ]

    job_id = manager.approve_eln_draft(run_id)
    assert job_id
    assert publisher.drain_once().state == DRAIN_PUBLISHED
    assert len(adapter.entries) == 1
    uploaded = sorted(Path(u["path"]).name for u in adapter.uploads)
    assert uploaded == ["difference.png", "loop.png", "montage.png"]
