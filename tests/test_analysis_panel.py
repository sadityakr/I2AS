# ---
# description: |
#   Behaviour tests for the eLab tab (gui/analysis_panel.py) and its home in
#   the procedure window: the panel lists an experiment's finished runs, picks
#   the recipe that serves the selected run, starts the runner, previews the
#   pending entry with the recipe's figures above the body, and approves or
#   discards it through the manager — and degrades to one line when nothing
#   is wired.
# last_updated: 2026-09-05
# ---

"""The eLab tab, built over stub collaborators.

The panel talks to three optional collaborators — the experiment manager, the
ELN publisher and the analysis runner — through a handful of duck-typed
methods and Qt signals. These tests supply exactly those, so the suite needs
no notebook, no subprocess and no session layer at all: what is asserted is
the panel's own behaviour, not its collaborators'.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from PyQt6.QtCore import QObject, pyqtSignal

from i2as.analysis.report import REPORT_FILENAME
from i2as.gui.analysis_panel import (
    ANALYSING_TEXT,
    EXPERIMENT_SUFFIX,
    NO_SESSION_TEXT,
    NOTHING_PENDING_TEXT,
    READY_TEXT,
    AnalysisPanel,
)
from i2as.session.eln.settings import AnalysisSettings, ElnSettings


# ── Stub collaborators ────────────────────────────────────────────────────────


#: The real analysis block — the stand-in from the parallel build is gone.
StubAnalysisSettings = AnalysisSettings


#: ``ElnSettings`` carries the analysis block itself now.
SettingsWithAnalysis = ElnSettings


@dataclass
class StubRun:
    """One run record, with only the fields the panel reads."""

    run_id: str
    procedure: str = "FieldSweep"
    status: str = "done"
    data_file: str = ""
    pending_eln_draft: dict[str, Any] = field(default_factory=dict)


@dataclass
class StubExperiment:
    """One experiment record: an id and its runs, oldest first."""

    experiment_id: str = "exp_1"
    runs: list[StubRun] = field(default_factory=list)


class StubStore:
    """The two analysis paths the panel asks the store for."""

    def __init__(self, root: Path) -> None:
        """Root every path under one throwaway directory.

        Args:
            root: The directory experiment folders are made under.
        """
        self.root = root

    def recipes_dir(self, experiment_id: str) -> Path:
        """Return the experiment's recipes directory (never created here)."""
        return self.root / experiment_id / "analysis" / "recipes"

    def report_dir(self, experiment_id: str, run_id: str) -> Path:
        """Return one run's report directory (never created here)."""
        return self.root / experiment_id / "analysis" / run_id


class StubManager(QObject):
    """The slice of ``ExperimentManager`` the eLab tab uses."""

    experiment_changed = pyqtSignal(dict)
    run_recorded = pyqtSignal(dict)

    def __init__(self, experiment: StubExperiment | None, store: StubStore) -> None:
        """Hold one open experiment and its store.

        Args:
            experiment: The open experiment, or ``None`` for "none open".
            store: The store the panel resolves analysis paths through.
        """
        super().__init__()
        self.experiment = experiment
        self.store = store
        self.approved: list[str] = []
        self.discarded: list[str] = []

    def current_experiment(self) -> StubExperiment | None:
        """Return the open experiment, or ``None``."""
        return self.experiment

    def pending_eln_draft(self, run_id: str) -> dict[str, Any]:
        """Return the entry parked on one run, or ``{}``."""
        for run in getattr(self.experiment, "runs", ()):
            if run.run_id == run_id:
                return dict(run.pending_eln_draft)
        return {}

    def approve_eln_draft(self, run_id: str) -> str:
        """Record an approval and clear the pending entry."""
        self.approved.append(run_id)
        for run in getattr(self.experiment, "runs", ()):
            if run.run_id == run_id:
                run.pending_eln_draft = {}
        return "job-1"

    def discard_pending_eln_draft(self, run_id: str) -> bool:
        """Record a discard and clear the pending entry."""
        self.discarded.append(run_id)
        for run in getattr(self.experiment, "runs", ()):
            if run.run_id == run_id:
                run.pending_eln_draft = {}
        return True


class StubPublisher(QObject):
    """The slice of ``ElnPublisher`` the eLab tab uses."""

    publish_state_changed = pyqtSignal(dict)

    def __init__(self, settings: Any) -> None:
        """Hold the settings the chip and the toggle read.

        Args:
            settings: The ``ElnSettings``-shaped record.
        """
        super().__init__()
        self._settings = settings
        self.reloaded: list[Any] = []

    @property
    def settings(self) -> Any:
        """The settings this publisher was built with."""
        return self._settings

    def status(self) -> dict[str, Any]:
        """Return a disabled publish status."""
        return {"state": "disabled", "pending": 0, "detail": ""}

    def reload_settings(self, settings: Any) -> None:
        """Record a settings reload."""
        self.reloaded.append(settings)
        self._settings = settings


class StubRunner(QObject):
    """The slice of ``AnalysisRunner`` the eLab tab uses."""

    analysis_started = pyqtSignal(str)
    analysis_finished = pyqtSignal(str, dict)
    analysis_failed = pyqtSignal(str, str)

    def __init__(self) -> None:
        """Start idle, recording every call."""
        super().__init__()
        self.calls: list[tuple[str, str]] = []
        self.running: set[str] = set()

    def start(self, run_id: str, recipe: str = "", **_kwargs: Any) -> str:
        """Record one start request and answer with a report directory."""
        self.calls.append((run_id, recipe))
        return f"/reports/{run_id}"

    def is_running(self, run_id: str = "") -> bool:
        """Return whether one run is being analysed."""
        return run_id in self.running


@dataclass(frozen=True)
class StubRecipeInfo:
    """One recipe as ``discover_recipes`` describes it."""

    name: str
    description: str = ""
    procedures: tuple[str, ...] = ("*",)
    source_path: str = ""
    origin: str = "package"
    digest: str = ""


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def wired(tmp_path, qtbot):
    """A panel over stub collaborators, with two finished runs and one pending.

    Returns:
        ``(panel, manager, publisher, runner)``.
    """
    experiment = StubExperiment(
        runs=[
            StubRun(run_id="run_001", procedure="FieldSweep"),
            StubRun(
                run_id="run_002",
                procedure="FieldSweep",
                pending_eln_draft={
                    "title": "FieldSweep — run_002",
                    "body_html": "<p>Two branches, no hysteresis.</p>",
                    "source": "analysis",
                },
            ),
        ]
    )
    manager = StubManager(experiment, StubStore(tmp_path))
    publisher = StubPublisher(SettingsWithAnalysis())
    runner = StubRunner()
    panel = AnalysisPanel(
        session_manager=manager,
        eln_publisher=publisher,
        analysis_runner=runner,
    )
    qtbot.addWidget(panel)
    return panel, manager, publisher, runner


def _write_report(store: StubStore, run_id: str, payload: dict[str, Any]) -> Path:
    """Write one report.json into a run's report directory.

    Args:
        store: The stub store resolving the directory.
        run_id: The run the report belongs to.
        payload: The report as its JSON dict.

    Returns:
        The directory the report was written into.
    """
    directory = store.report_dir("exp_1", run_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / REPORT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    return directory


def _fake_discovery(monkeypatch, recipes: tuple[StubRecipeInfo, ...]) -> None:
    """Install a stub ``i2as.analysis.discovery`` module.

    The real one is built in parallel; the panel imports it lazily by name,
    so a module object in ``sys.modules`` is exactly what it would find.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        recipes: What ``discover_recipes`` should return.
    """
    import sys
    import types

    module = types.ModuleType("i2as.analysis.discovery")

    def discover_recipes(extra_dirs=()):  # noqa: ANN001, ANN202 - a stub
        return recipes

    def recipe_for(procedure, available, preferred=""):  # noqa: ANN001, ANN202
        for info in available:
            if preferred and info.name == preferred:
                return info
        for info in available:
            if procedure in info.procedures:
                return info
        for info in available:
            if "*" in info.procedures:
                return info
        return None

    def scaffold_recipe(name, directory, procedure=""):  # noqa: ANN001, ANN202
        path = Path(directory) / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {procedure}\n", encoding="utf-8")
        return path

    module.discover_recipes = discover_recipes
    module.recipe_for = recipe_for
    module.scaffold_recipe = scaffold_recipe
    monkeypatch.setitem(sys.modules, "i2as.analysis.discovery", module)


# ── The not-wired state ───────────────────────────────────────────────────────


def test_panel_builds_with_no_collaborators(qtbot):
    """With nothing wired the panel builds and says so in one line."""
    panel = AnalysisPanel()
    qtbot.addWidget(panel)
    assert panel._status_label.text() == NO_SESSION_TEXT
    assert not panel._publish_btn.isEnabled()
    assert not panel._discard_btn.isEnabled()
    assert not panel._run_btn.isEnabled()
    assert panel.current_run_id() == ""


# ── Runs and recipes ──────────────────────────────────────────────────────────


def test_runs_are_listed_newest_first(wired):
    """The run combo lists the experiment's finished runs, newest on top."""
    panel, _manager, _publisher, _runner = wired
    labels = [panel._run_combo.itemText(i) for i in range(panel._run_combo.count())]
    assert labels == [
        "run_002 · FieldSweep · done",
        "run_001 · FieldSweep · done",
    ]
    assert panel.current_run_id() == "run_002"


def test_a_running_run_is_not_offered(tmp_path, qtbot):
    """A run still in flight is not something to analyse."""
    experiment = StubExperiment(
        runs=[StubRun(run_id="run_001", status="running")]
    )
    panel = AnalysisPanel(
        session_manager=StubManager(experiment, StubStore(tmp_path))
    )
    qtbot.addWidget(panel)
    assert panel._run_combo.count() == 0


def test_recipes_are_filtered_and_the_default_is_preselected(wired, monkeypatch):
    """Only recipes serving the run's procedure are offered, marked by origin."""
    panel, _manager, _publisher, _runner = wired
    _fake_discovery(
        monkeypatch,
        (
            StubRecipeInfo(name="generic_sweep", procedures=("*",)),
            StubRecipeInfo(name="other_only", procedures=("SomethingElse",)),
            StubRecipeInfo(
                name="hall_bar", procedures=("FieldSweep",), origin="experiment"
            ),
        ),
    )
    panel.reload()
    labels = [
        panel._recipe_combo.itemText(i) for i in range(panel._recipe_combo.count())
    ]
    assert labels == ["generic_sweep", "hall_bar" + EXPERIMENT_SUFFIX]
    # recipe_for prefers the one naming the procedure over the any-procedure one.
    assert panel.selected_recipe() == "hall_bar"


def test_the_pinned_recipe_wins(wired, monkeypatch):
    """A recipe pinned in the settings is the one preselected."""
    panel, _manager, publisher, _runner = wired
    publisher.reload_settings(
        replace(
            publisher.settings,
            analysis=replace(
                publisher.settings.analysis, recipes={"FieldSweep": "generic_sweep"}
            ),
        )
    )
    _fake_discovery(
        monkeypatch,
        (
            StubRecipeInfo(name="generic_sweep", procedures=("*",)),
            StubRecipeInfo(name="hall_bar", procedures=("FieldSweep",)),
        ),
    )
    panel.reload()
    assert panel.selected_recipe() == "generic_sweep"


def test_missing_analysis_package_leaves_an_empty_recipe_list(wired, monkeypatch):
    """No analysis package means no recipes and a status line, not a crash."""
    import builtins

    real_import = builtins.__import__

    def _refuse(name, *args, **kwargs):  # noqa: ANN001, ANN202 - an import stub
        if name == "i2as.analysis.discovery":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _refuse)
    panel, _manager, _publisher, _runner = wired
    panel.reload()
    assert panel._recipe_combo.count() == 0
    assert not panel._new_recipe_btn.isEnabled()


def test_new_recipe_scaffolds_and_offers_it(wired, monkeypatch):
    """"New recipe…" writes the file, opens it, and selects it in the combo."""
    panel, manager, _publisher, _runner = wired
    _fake_discovery(monkeypatch, ())
    opened: list[str] = []
    monkeypatch.setattr(
        "i2as.gui.analysis_panel.QInputDialog.getText",
        staticmethod(lambda *a, **k: ("hall_bar", True)),
    )
    monkeypatch.setattr(
        "i2as.gui.analysis_panel.QDesktopServices.openUrl",
        staticmethod(lambda url: opened.append(url.toString())),
    )
    panel.reload()
    panel._on_new_recipe_clicked()
    written = manager.store.recipes_dir("exp_1") / "hall_bar.py"
    assert written.exists()
    assert opened and opened[0].endswith("hall_bar.py")


# ── Running an analysis ───────────────────────────────────────────────────────


def test_run_analysis_starts_the_selected_recipe(wired, monkeypatch):
    """"Run analysis" hands the runner the run and the selected recipe."""
    panel, _manager, _publisher, runner = wired
    _fake_discovery(monkeypatch, (StubRecipeInfo(name="generic_sweep"),))
    panel.reload()
    panel._run_btn.click()
    assert runner.calls == [("run_002", "generic_sweep")]
    assert panel._status_label.text() == ANALYSING_TEXT


def test_run_analysis_is_disabled_while_that_run_is_analysed(wired):
    """A run already being analysed cannot be started a second time."""
    panel, _manager, _publisher, runner = wired
    runner.running.add("run_002")
    panel.reload()
    assert not panel._run_btn.isEnabled()


def test_runner_failure_is_shown(wired):
    """A failed analysis names the failure on the status line."""
    panel, _manager, _publisher, runner = wired
    runner.analysis_failed.emit("run_002", "ValueError: no sweep column\ntraceback…")
    assert panel._status_label.text() == "Analysis failed: ValueError: no sweep column"


def test_runner_start_shows_analysing(wired):
    """The runner's started signal puts the panel into its analysing state."""
    panel, _manager, _publisher, runner = wired
    runner.analysis_started.emit("run_002")
    assert panel._status_label.text() == ANALYSING_TEXT


# ── The preview, and approving what it shows ──────────────────────────────────


def test_pending_entry_preview_shows_title_and_every_figure(wired, monkeypatch):
    """The preview carries the entry's title and one <img per saved figure."""
    panel, manager, _publisher, _runner = wired
    directory = _write_report(
        manager.store,
        "run_002",
        {
            "run_id": "run_002",
            "status": "ok",
            "figures": [
                {"file": "overview.png", "caption": "Overview"},
                {"file": "detail.png", "caption": "Detail"},
            ],
            "warnings": ["one column was empty"],
        },
    )
    (directory / "overview.png").write_bytes(b"")
    (directory / "detail.png").write_bytes(b"")
    panel.reload()

    html = panel._preview.toHtml()
    assert "FieldSweep — run_002" in html
    assert html.count("<img") == 2
    assert "overview.png" in html and "detail.png" in html
    assert panel._status_label.text() == READY_TEXT
    # The recipe's warnings are shown, and only when there are any.
    assert not panel._warnings.isHidden()
    assert "one column was empty" in panel._warnings.toPlainText()


def test_no_pending_entry_leaves_an_empty_preview(wired, monkeypatch):
    """A run with nothing waiting says so and offers neither approval."""
    panel, _manager, _publisher, _runner = wired
    _fake_discovery(monkeypatch, (StubRecipeInfo(name="generic_sweep"),))
    panel.reload()
    panel.set_run("run_001")
    assert panel._preview.toPlainText().strip() == ""
    assert panel._status_label.text() == NOTHING_PENDING_TEXT
    assert not panel._publish_btn.isEnabled()
    assert not panel._discard_btn.isEnabled()
    assert panel._warnings.isHidden()


def test_publish_approves_through_the_manager(wired):
    """Publish calls ``approve_eln_draft`` and the entry stops being pending."""
    panel, manager, _publisher, _runner = wired
    assert panel._publish_btn.isEnabled()
    panel._publish_btn.click()
    assert manager.approved == ["run_002"]
    assert not panel._publish_btn.isEnabled()


def test_discard_drops_the_entry_through_the_manager(wired):
    """Discard calls ``discard_pending_eln_draft`` and clears the preview."""
    panel, manager, _publisher, _runner = wired
    panel._discard_btn.click()
    assert manager.discarded == ["run_002"]
    assert panel._preview.toPlainText().strip() == ""


def test_run_finished_selects_that_run(wired):
    """The Orchestrator's run boundary points the panel at the finished run."""
    panel, _manager, _publisher, _runner = wired
    panel.on_run_finished({"run_id": "run_001"})
    assert panel.current_run_id() == "run_001"


# ── The chip and the analysis toggle ──────────────────────────────────────────


@pytest.mark.parametrize(
    "state", ["synced", "pending", "offline", "disabled"]
)
def test_publish_state_chip_carries_the_state_property(wired, state):
    """Every publish state reaches the chip as text and as its property."""
    panel, _manager, publisher, _runner = wired
    publisher.publish_state_changed.emit({"state": state, "pending": 2, "detail": ""})
    assert panel._chip.property("state") == state
    assert state[:4] in panel._chip.text() or state == "disabled"


def test_analysis_toggle_saves_through_the_publisher(wired, monkeypatch):
    """Ticking "Analysis on" writes the setting and reloads the publisher."""
    panel, _manager, publisher, _runner = wired
    saved: list[Any] = []
    monkeypatch.setattr(
        "i2as.gui.analysis_panel.persist_eln_settings",
        lambda settings, pub=None: saved.append((settings, pub)) or True,
    )
    panel._enabled_checkbox.setChecked(True)
    assert saved and saved[0][0].analysis.enabled is True
    assert saved[0][1] is publisher

