# ---
# description: |
#   Behaviour tests for the Analysis tab (gui/analysis_panel.py): the panel
#   lists an experiment's finished runs, picks the recipe that serves the
#   selected run, starts the runner, lists every analysis bundle of a run and
#   lets the operator choose the one that represents it, previews it from its
#   sealed files, and drives the experiment's notebook strip (link, read
#   fields, approve once, publish, retry) — and degrades to one line when
#   nothing is wired.
# last_updated: 2026-09-27
# ---

"""The Analysis tab, built over stub collaborators and a real experiment store."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from PyQt6.QtCore import QObject, pyqtSignal

from i2as.analysis.bundle import Producer, seal_bundle
from i2as.gui import app_settings
from i2as.gui.analysis_panel import (
    ANALYSING_TEXT,
    EXPERIMENT_SUFFIX,
    NO_BUNDLES_TEXT,
    NO_SESSION_TEXT,
    SELECTED_MARK,
    AnalysisPanel,
)
from i2as.session.analysis_sandbox import EngineStatus
from i2as.session.eln.publishing import PublishError
from i2as.session.models import ElnBinding, ElnLink
from i2as.session.store import ExperimentStore

# ── Stub collaborators ────────────────────────────────────────────────────────


@dataclass
class StubRun:
    run_id: str
    procedure: str = "FieldSweep"
    status: str = "done"
    data_file: str = ""
    selected_bundle: str = ""
    published: bool = False


@dataclass
class StubExperiment:
    experiment_id: str = "exp_1"
    user_id: str = "jdoe"
    runs: list[StubRun] = field(default_factory=list)
    eln: ElnBinding | None = None


class StubManager(QObject):
    """The slice of ``ExperimentManager`` the Analysis tab uses."""

    experiment_changed = pyqtSignal(dict)
    run_recorded = pyqtSignal(dict)

    def __init__(self, experiment: StubExperiment | None, store: ExperimentStore) -> None:
        super().__init__()
        self.experiment = experiment
        self.store = store
        self.selected: list[tuple[str, str]] = []
        self.approved: list[str] = []

    def current_experiment(self) -> StubExperiment | None:
        return self.experiment

    def select_bundle(self, run_id: str, bundle_id: str) -> bool:
        self.selected.append((run_id, bundle_id))
        for run in self.experiment.runs:
            if run.run_id == run_id:
                run.selected_bundle = bundle_id
        return True

    def approve_eln_publishing(self, user_id: str) -> bool:
        self.approved.append(user_id)
        self.experiment.eln = replace(self.experiment.eln, publish_approved=True)
        return True


class StubService(QObject):
    """The slice of ``ElnService`` the Analysis tab uses."""

    status_changed = pyqtSignal(dict)
    publish_finished = pyqtSignal(dict)
    publish_failed = pyqtSignal(dict)
    page_ready = pyqtSignal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.published: list[Any] = []
        self.retried: list[str] = []
        self.refuse = ""
        self.current_status: dict[str, Any] = {"state": "synced", "pending": 0, "attention": []}

    def enabled(self, _user: str = "") -> bool:
        return True

    def status(self, _experiment_id: str = "") -> dict[str, Any]:
        return dict(self.current_status)

    def publish(self, run_ids=None) -> str:
        if self.refuse:
            raise PublishError(self.refuse)
        self.published.append(run_ids)
        return "P-1"

    def retry(self, experiment_id: str = "") -> int:
        self.retried.append(experiment_id)
        return 1


class StubRunner(QObject):
    analysis_started = pyqtSignal(str)
    analysis_finished = pyqtSignal(str, dict)
    analysis_failed = pyqtSignal(str, str)
    bundle_ready = pyqtSignal(str, str, dict)

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []
        self.running: set[str] = set()

    def start(self, run_id: str, recipe: str = "", **_kwargs: Any) -> str:
        self.calls.append((run_id, recipe))
        return f"/bundles/{run_id}"

    def is_running(self, run_id: str = "") -> bool:
        return run_id in self.running


@dataclass(frozen=True)
class StubRecipeInfo:
    name: str
    description: str = ""
    procedures: tuple[str, ...] = ("*",)
    source_path: str = ""
    origin: str = "package"
    digest: str = ""


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _seal(store: ExperimentStore, run_id: str, bundle_id: str, *, status: str = "ok", created: str = "2026-09-27T10:00:00+00:00", figure: bool = True):
    folder = store.bundle_dir("exp_1", run_id, bundle_id)
    folder.mkdir(parents=True, exist_ok=True)
    if figure:
        (folder / "overview.png").write_bytes(b"\x89PNG")
    return seal_bundle(
        folder,
        {
            "status": status,
            "error": "ValueError: bad\ntraceback" if status == "failed" else "",
            "summary": ["Two branches, no hysteresis."],
            "results": [{"name": "Bc", "value": 1.25, "unit": "T"}],
            "figures": [{"file": "overview.png", "caption": "Overview"}] if figure else [],
            "warnings": ["a column was missing"],
        },
        bundle_id=bundle_id,
        experiment_id="exp_1",
        run_ids=(run_id,),
        producer=Producer(kind="recipe", name="generic_sweep"),
        created_utc=created,
    )


@pytest.fixture
def wired(tmp_path, qtbot):
    """A panel over stubs: two finished runs, run_002 with two bundles (one selected)."""
    store = ExperimentStore(tmp_path)
    experiment = StubExperiment(runs=[StubRun("run_001"), StubRun("run_002", selected_bundle="b-old")])
    _seal(store, "run_002", "b-old", created="2026-09-27T09:00:00+00:00")
    _seal(store, "run_002", "b-new", created="2026-09-27T10:00:00+00:00")
    manager = StubManager(experiment, store)
    service = StubService()
    runner = StubRunner()
    opened: list[str] = []
    panel = AnalysisPanel(
        session_manager=manager,
        eln_service=service,
        analysis_runner=runner,
        dialog_factory=lambda kind, *_args: opened.append(kind),
    )
    qtbot.addWidget(panel)
    panel.opened = opened
    return panel, manager, service, runner


def _fake_discovery(monkeypatch, recipes: tuple[StubRecipeInfo, ...]) -> None:
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
    panel = AnalysisPanel()
    qtbot.addWidget(panel)
    assert panel._status_label.text() == NO_SESSION_TEXT
    assert not panel._publish_btn.isEnabled() and not panel._link_btn.isEnabled()
    assert not panel._run_btn.isEnabled() and not panel._select_btn.isEnabled()
    assert panel.current_run_id() == ""


# ── Runs and recipes ──────────────────────────────────────────────────────────


def test_runs_are_listed_newest_first(wired):
    panel, *_ = wired
    labels = [panel._run_combo.itemText(i) for i in range(panel._run_combo.count())]
    assert labels == ["run_002 · FieldSweep · done", "run_001 · FieldSweep · done"]
    assert panel.current_run_id() == "run_002"


def test_a_running_run_is_not_offered(tmp_path, qtbot):
    experiment = StubExperiment(runs=[StubRun(run_id="run_001", status="running")])
    panel = AnalysisPanel(session_manager=StubManager(experiment, ExperimentStore(tmp_path)))
    qtbot.addWidget(panel)
    assert panel._run_combo.count() == 0


def test_recipes_are_filtered_and_the_default_is_preselected(wired, monkeypatch):
    panel, *_ = wired
    _fake_discovery(
        monkeypatch,
        (
            StubRecipeInfo(name="generic_sweep", procedures=("*",)),
            StubRecipeInfo(name="other_only", procedures=("SomethingElse",)),
            StubRecipeInfo(name="hall_bar", procedures=("FieldSweep",), origin="experiment"),
        ),
    )
    panel.reload()
    labels = [panel._recipe_combo.itemText(i) for i in range(panel._recipe_combo.count())]
    assert labels == ["generic_sweep", "hall_bar" + EXPERIMENT_SUFFIX]
    assert panel.selected_recipe() == "hall_bar"


def test_the_pinned_recipe_wins(wired, monkeypatch):
    panel, *_ = wired
    store = app_settings.config_store()
    store.save(replace(store.current, analysis=replace(store.current.analysis, recipes={"FieldSweep": "generic_sweep"})))
    _fake_discovery(monkeypatch, (StubRecipeInfo(name="generic_sweep"), StubRecipeInfo(name="hall_bar", procedures=("FieldSweep",))))
    panel.reload()
    assert panel.selected_recipe() == "generic_sweep"


def test_new_recipe_scaffolds_and_offers_it(wired, monkeypatch):
    panel, manager, *_ = wired
    _fake_discovery(monkeypatch, ())
    opened: list[str] = []
    monkeypatch.setattr("i2as.gui.analysis_panel.QInputDialog.getText", staticmethod(lambda *a, **k: ("hall_bar", True)))
    monkeypatch.setattr("i2as.gui.analysis_panel.QDesktopServices.openUrl", staticmethod(lambda url: opened.append(url.toString())))
    panel.reload()
    panel._on_new_recipe_clicked()
    assert (manager.store.recipes_dir("exp_1") / "hall_bar.py").exists()
    assert opened and opened[0].endswith("hall_bar.py")


# ── Running an analysis ───────────────────────────────────────────────────────


def test_run_analysis_starts_the_selected_recipe(wired, monkeypatch):
    panel, _manager, _service, runner = wired
    _fake_discovery(monkeypatch, (StubRecipeInfo(name="generic_sweep"),))
    panel.reload()
    panel._run_btn.click()
    assert runner.calls == [("run_002", "generic_sweep")]
    assert panel._status_label.text() == ANALYSING_TEXT


def test_run_analysis_is_disabled_while_that_run_is_analysed(wired):
    panel, _manager, _service, runner = wired
    runner.running.add("run_002")
    panel.reload()
    assert not panel._run_btn.isEnabled()


def test_runner_failure_is_shown(wired):
    panel, _manager, _service, runner = wired
    panel.set_run("run_001")
    runner.analysis_failed.emit("run_001", "ValueError: no sweep column\ntraceback…")
    assert panel._status_label.text() == "Analysis failed: ValueError: no sweep column"


# ── Bundles: which result represents the run ──────────────────────────────────


def test_every_bundle_of_the_run_is_listed_and_the_selected_one_is_marked(wired):
    panel, *_ = wired
    labels = [panel._bundle_combo.itemText(i) for i in range(panel._bundle_combo.count())]
    assert len(labels) == 2
    assert labels[1].startswith(SELECTED_MARK), "the selected (older) bundle carries the mark"
    assert panel.shown_bundle_id() == "b-old", "the preview opens on what represents the run"
    assert "This result represents the run." in panel._status_label.text()
    assert not panel._select_btn.isEnabled()


def test_the_preview_shows_the_bundle_from_its_sealed_files(wired):
    panel, *_ = wired
    html = panel._preview.toHtml()
    assert "Two branches, no hysteresis." in panel._preview.toPlainText()
    assert "Bc" in panel._preview.toPlainText() and "overview.png" in html
    assert panel._warnings.isVisibleTo(panel) and "a column was missing" in panel._warnings.toPlainText()


def test_choosing_another_bundle_goes_through_the_manager(wired):
    panel, manager, *_ = wired
    panel._bundle_combo.setCurrentIndex(panel._bundle_combo.findData("b-new"))
    assert panel._select_btn.isEnabled()
    panel._select_btn.click()
    assert manager.selected == [("run_002", "b-new")]
    assert panel._bundle_combo.itemText(panel._bundle_combo.findData("b-new")).startswith(SELECTED_MARK)


def test_a_failed_bundle_cannot_represent_the_run(wired):
    panel, manager, *_ = wired
    _seal(manager.store, "run_002", "b-bad", status="failed", created="2026-09-27T11:00:00+00:00", figure=False)
    panel.reload()
    panel._bundle_combo.setCurrentIndex(panel._bundle_combo.findData("b-bad"))
    assert not panel._select_btn.isEnabled()
    assert "ValueError: bad" in panel._warnings.toPlainText()


def test_a_run_with_no_bundle_says_so(wired):
    panel, *_ = wired
    panel.set_run("run_001")
    assert panel._bundle_combo.count() == 0
    assert panel._status_label.text() == NO_BUNDLES_TEXT


def test_a_new_bundle_for_the_shown_run_appears(wired):
    panel, manager, _service, runner = wired
    _seal(manager.store, "run_002", "b-third", created="2026-09-27T12:00:00+00:00")
    runner.bundle_ready.emit("run_002", "b-third", {})
    assert panel._bundle_combo.findData("b-third") >= 0


# ── The notebook strip ────────────────────────────────────────────────────────


def test_an_unlinked_experiment_offers_to_link_and_nothing_else(wired):
    panel, *_ = wired
    assert "Not linked" in panel._notebook_label.text()
    assert panel._link_btn.isEnabled() and not panel._fields_btn.isEnabled()
    assert not panel._publish_btn.isEnabled() and not panel._approve_btn.isVisibleTo(panel)
    panel._link_btn.click()
    assert panel.opened == ["link"]


def test_approval_is_asked_once_then_publishing_is_open(wired):
    panel, manager, service, _runner = wired
    manager.experiment.eln = ElnBinding(account_id="lab", entry=ElnLink(entry_id="7", url="https://e/7"))
    panel.reload()
    assert "https://e/7" in panel._notebook_label.text() and "not approved" in panel._notebook_label.text()
    assert panel._approve_btn.isVisibleTo(panel) and not panel._publish_btn.isEnabled()
    panel._approve_btn.click()
    assert manager.approved and not panel._approve_btn.isVisibleTo(panel)
    assert panel._publish_btn.isEnabled()
    assert "2 run(s) not on the page yet" in panel._notebook_label.text()
    panel._publish_btn.click()
    assert service.published == [None]
    assert "P-1" in panel._status_label.text()


def test_a_refused_publish_says_why(wired):
    panel, manager, service, _runner = wired
    manager.experiment.eln = ElnBinding(account_id="lab", entry=ElnLink(entry_id="7"), publish_approved=True)
    service.refuse = "there is nothing new to publish"
    panel.reload()
    panel._publish_btn.click()
    assert panel._status_label.text() == "there is nothing new to publish"


def test_work_needing_attention_offers_retry_and_says_why(wired):
    panel, _manager, service, _runner = wired
    service.status_changed.emit({"state": "attention", "pending": 0, "attention": [{"error": "the key was rejected"}]})
    assert panel._chip.property("state") == "attention"
    assert "the key was rejected" in panel._chip.toolTip()
    assert panel._retry_btn.isVisibleTo(panel)
    panel._retry_btn.click()
    assert service.retried == ["exp_1"]


@pytest.mark.parametrize("state", ["synced", "pending", "offline", "attention", "disabled"])
def test_the_chip_carries_the_notebook_state(wired, state):
    panel, _manager, service, _runner = wired
    service.status_changed.emit({"state": state, "pending": 2, "attention": []})
    assert panel._chip.property("state") == state


def test_a_published_run_is_marked_in_the_run_list(wired):
    panel, manager, service, _runner = wired
    manager.experiment.runs[0].published = True
    service.publish_finished.emit({"run_bundles": {"run_001": ""}})
    assert any("on page" in panel._run_combo.itemText(i) for i in range(panel._run_combo.count()))
    assert "Published run_001" in panel._status_label.text()


# ── The analysis toggle ───────────────────────────────────────────────────────


def test_analysis_toggle_saves_to_the_settings_file(wired):
    panel, *_ = wired
    panel._engine_checker = lambda _sandbox: EngineStatus(True, True, True, "Ready")
    panel._enabled_checkbox.setChecked(True)
    assert app_settings.config_store().analysis().enabled is True


def test_analysis_toggle_refuses_without_a_container_engine(wired):
    panel, *_ = wired
    panel._engine_checker = lambda _sandbox: EngineStatus(detail="docker is not installed")
    panel._enabled_checkbox.setChecked(True)
    assert app_settings.config_store().analysis().enabled is False
    assert "docker is not installed" in panel._status_label.text()
