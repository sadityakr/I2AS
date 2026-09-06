"""The analysis stage: the recipe contract, discovery, the runner and the worker.

Every test here writes its run file with the production ``DataManager`` and
reads it back through the production reader, exactly as ``test_l5_data_reader``
does, because a recipe's whole job is to agree with the file the application
actually writes.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import i2as
from i2as.analysis.base import (
    MATPLOTLIB_INSTALL_HINT,
    AnalysisContext,
    AnalysisError,
    AnalysisRecipe,
    axis_label,
    choose_x_column,
    measured_columns,
)
from i2as.analysis.discovery import (
    ORIGIN_EXPERIMENT,
    ORIGIN_PACKAGE,
    RECIPE_TEMPLATE,
    RecipeInfo,
    discover_recipes,
    load_recipe,
    procedure_key,
    recipe_for,
    scaffold_recipe,
)
from i2as.analysis.report import (
    ANY_PROCEDURE,
    REPORT_FAILED,
    REPORT_FILENAME,
    REPORT_OK,
    AnalysisReport,
    AnalysisSpec,
)
from i2as.analysis.runner import read_report, run_spec, write_report
from i2as.core.data_manager import DataManager
from i2as.core.data_reader import open_run

REPO_ROOT = Path(i2as.__file__).parent.parent

N_POINTS = 5

DATA_CONFIG = {
    # The writer's own order: the clock, then the system read-backs, then the
    # procedure's own axis column LAST — which is what choose_x_column() reads.
    "sweep_columns": {"unix_time": "float", "temperature_K": "float", "field_T": "float"},
    "measurement_scalars": {"voltage_V": "float"},
    "measurement_arrays": {},
    "measurement_blocks": {},
    "loop_shape": [2, 1],
}

MANIFEST = {
    "run_id": "run-1",
    "procedure": "FieldSweep",
    "kind": "run",
    "params": {"field_start": -1.0, "field_end": 1.0},
    "status": "done",
    "started_utc": "2026-01-01T00:00:00+00:00",
    "finished_utc": "2026-01-01T00:05:00+00:00",
}


# ── Fixtures ──────────────────────────────────────────────────────────────


def _writer(directory: Path, n_sweep_points: int = N_POINTS) -> DataManager:
    """Return a ``DataManager`` writing the fixture layout into *directory*.

    Args:
        directory: Where the run file is created.
        n_sweep_points: How many points to pre-allocate.

    Returns:
        The open writer; the caller closes it.
    """
    return DataManager(
        data_directory=str(directory),
        procedure_name="FieldSweep",
        procedure_params={"field_start": -1.0, "field_end": 1.0, "loop1_values": [1e-6, -1e-6]},
        sample_info={"sample_name": "Test Sample", "sample_id": "TST-001"},
        instrument_state={"magnet_z": {"field": 0.0}},
        system_targets={"magnet_z": {"target": -1.0}},
        measurement_commands=[],
        data_config=DATA_CONFIG,
        n_sweep_points=n_sweep_points,
        experiment_info={
            "setup": {"config_name": "sim_setup"},
            "experiment": {"experiment_id": "EXP-1", "user_name": "A. Operator"},
        },
    )


@pytest.fixture
def run_file(tmp_path) -> Path:
    """A closed run file with five points of one sweep and one measurement."""
    writer = _writer(tmp_path / "data")
    for index in range(N_POINTS):
        writer.save_datapoint(
            index,
            {
                "unix_time": 1_000.0 + index,
                "temperature_K": 4.2 + 0.01 * index,
                "field_T": -1.0 + 0.5 * index,
                "voltage_V": [[1.0 + index], [2.0 + index]],
            },
            {},
        )
    writer.close()
    return Path(writer.filepath)


@pytest.fixture
def empty_run_file(tmp_path) -> Path:
    """A closed run file that never received a datapoint."""
    writer = _writer(tmp_path / "empty")
    writer.close()
    return Path(writer.filepath)


def _spec(run_file: Path, output_dir: Path, **overrides) -> AnalysisSpec:
    """Build a spec for the fixture run.

    Args:
        run_file: The run's HDF5 file.
        output_dir: Where the report goes.
        **overrides: Any ``AnalysisSpec`` field to override.

    Returns:
        The spec.
    """
    fields = {
        "run_id": "run-1",
        "data_path": str(run_file),
        "manifest": dict(MANIFEST),
        "experiment": {"experiment_id": "EXP-1", "experiment_title": "Sample A"},
        "setup": {"config_name": "sim_setup"},
        "output_dir": str(output_dir),
    }
    fields.update(overrides)
    return AnalysisSpec(**fields)


EXPERIMENT_RECIPE = '''
from i2as.analysis.base import AnalysisRecipe
from i2as.analysis.report import AnalysisReport


class LocalRecipe(AnalysisRecipe):
    name = "local_overview"
    procedures = ("FieldSweep",)
    description = "This experiment's own overview"

    def analyse(self, run, context):
        return AnalysisReport(summary=(f"local recipe saw {run.n_points} points",))
'''

OVERRIDING_RECIPE = '''
from i2as.analysis.base import AnalysisRecipe
from i2as.analysis.report import AnalysisReport


class MyGenericSweep(AnalysisRecipe):
    name = "generic_sweep"
    procedures = ("*",)
    description = "This experiment's replacement for the shipped overview"

    def analyse(self, run, context):
        return AnalysisReport(summary=("overridden",))
'''

BROKEN_RECIPE = "this is not python at all ((("

RAISING_RECIPE = '''
from i2as.analysis.base import AnalysisRecipe


class Boom(AnalysisRecipe):
    name = "boom"
    procedures = ("*",)
    description = "Always raises"

    def analyse(self, run, context):
        raise RuntimeError("the fit did not converge")
'''


# ── The context helpers ───────────────────────────────────────────────────


def test_context_figure_saves_a_png_and_closes_it(tmp_path):
    """figure() writes <output_dir>/<name>.png and returns a relative ref."""
    context = AnalysisContext(run_id="r", output_dir=tmp_path / "report")
    plt = context.pyplot()
    fig, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])

    ref = context.figure("overview", fig, caption="Everything", width_px=600)

    assert ref.file == "overview.png"
    assert ref.caption == "Everything"
    assert ref.width_px == 600
    assert (tmp_path / "report" / "overview.png").is_file()
    assert not plt.get_fignums(), "figure() must close the figure it saved"


def test_context_figure_refuses_a_name_that_is_not_a_file_stem(tmp_path):
    """A name with a path separator or an extension is refused, not sanitised."""
    context = AnalysisContext(output_dir=tmp_path)
    plt = context.pyplot()
    fig = plt.figure()
    for bad in ("../escape", "sub/dir", "overview.png", ""):
        with pytest.raises(AnalysisError):
            context.figure(bad, fig)
    plt.close(fig)


def test_context_pyplot_names_the_install_command_when_matplotlib_is_absent(monkeypatch, tmp_path):
    """matplotlib is an optional extra, so its absence is one clear sentence."""
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    context = AnalysisContext(output_dir=tmp_path)
    with pytest.raises(AnalysisError) as excinfo:
        context.pyplot()
    assert MATPLOTLIB_INSTALL_HINT in str(excinfo.value)


def test_context_table_applies_the_report_caps(tmp_path):
    """table() is TableSpec.build: capped rows, marked truncated."""
    context = AnalysisContext(output_dir=tmp_path)
    table = context.table("Big", ["a", "b"], [[i, i * 2] for i in range(200)])
    assert table.caption == "Big"
    assert table.columns == ("a", "b")
    assert len(table.rows) < 200
    assert table.truncated


def test_base_recipe_refuses_to_be_used_directly(tmp_path):
    """The base class is a contract, not a default implementation."""
    with pytest.raises(NotImplementedError):
        AnalysisRecipe().analyse(None, AnalysisContext(output_dir=tmp_path))


# ── The column conventions ────────────────────────────────────────────────


def test_choose_x_column_prefers_the_runs_own_sweep_axis(run_file):
    """The file's last-declared sweep column is the procedure's axis read-back."""
    with open_run(run_file) as run:
        assert choose_x_column(run, MANIFEST).name == "field_T"


def test_choose_x_column_honours_a_manifest_declaration(run_file):
    """A manifest naming the axis (default_x_key) wins over the file's guess."""
    with open_run(run_file) as run:
        chosen = choose_x_column(run, {**MANIFEST, "default_x_key": "temperature_K"})
    assert chosen.name == "temperature_K"


def test_measured_columns_leaves_out_the_axis_and_the_clocks(run_file):
    """The clocks are axes, not readings; so is whatever the caller excluded."""
    with open_run(run_file) as run:
        names = [info.name for info in measured_columns(run, exclude=["field_T"])]
    assert names == ["temperature_K", "voltage_V"]
    assert axis_label(None) == "point index"


# ── Discovery ─────────────────────────────────────────────────────────────


def test_discover_recipes_finds_the_shipped_recipes():
    """The package's own recipes are discovered, the overview first."""
    infos = discover_recipes()
    names = [info.name for info in infos]
    assert names[0] == "generic_sweep", "the default overview must be the first match"
    assert "facts_only" in names
    for info in infos:
        assert info.origin == ORIGIN_PACKAGE
        assert info.digest and len(info.digest) == 64
        assert Path(info.source_path).is_file()
        assert info.to_dict()["procedures"] == list(info.procedures)


def test_discover_recipes_loads_experiment_scripts(tmp_path):
    """A *.py in an extra directory becomes a recipe, marked as the experiment's."""
    (tmp_path / "local_overview.py").write_text(EXPERIMENT_RECIPE, encoding="utf-8")
    (tmp_path / "_helper.py").write_text(EXPERIMENT_RECIPE, encoding="utf-8")

    infos = discover_recipes([tmp_path])
    local = [info for info in infos if info.name == "local_overview"]

    assert len(local) == 1, "an underscore-prefixed script must be skipped"
    assert local[0].origin == ORIGIN_EXPERIMENT
    assert local[0].procedures == ("FieldSweep",)
    assert local[0].source_path == str(tmp_path / "local_overview.py")
    # The synthetic module name marks the recipe as loaded from a script, not
    # the package, and cannot collide with any importable i2as module.
    assert type(load_recipe(local[0])).__module__.startswith(
        "i2as_analysis_experiment_recipe"
    )


def test_an_experiment_recipe_overrides_the_package_one_of_the_same_name(tmp_path):
    """The local answer wins, and the name is not listed twice."""
    (tmp_path / "mine.py").write_text(OVERRIDING_RECIPE, encoding="utf-8")

    infos = discover_recipes([tmp_path])
    matching = [info for info in infos if info.name == "generic_sweep"]

    assert len(matching) == 1
    assert matching[0].origin == ORIGIN_EXPERIMENT
    assert load_recipe(matching[0]).description.startswith("This experiment's replacement")


def test_a_broken_script_is_skipped_with_a_warning(tmp_path, caplog):
    """One half-written file must not hide every other recipe."""
    (tmp_path / "broken.py").write_text(BROKEN_RECIPE, encoding="utf-8")
    (tmp_path / "local_overview.py").write_text(EXPERIMENT_RECIPE, encoding="utf-8")

    with caplog.at_level("WARNING"):
        names = [info.name for info in discover_recipes([tmp_path])]

    assert "local_overview" in names
    assert "generic_sweep" in names
    assert any("broken.py" in record.getMessage() for record in caplog.records)


def test_discover_recipes_tolerates_a_missing_directory(tmp_path):
    """An experiment that never wrote a recipe is the normal case."""
    assert discover_recipes([tmp_path / "nope"]) == discover_recipes()


def test_recipe_for_preference_order(tmp_path):
    """preferred name, else the procedure's own recipe, else the wildcard one."""
    (tmp_path / "local_overview.py").write_text(EXPERIMENT_RECIPE, encoding="utf-8")
    infos = discover_recipes([tmp_path])

    assert recipe_for("FieldSweep", infos, preferred="facts_only").name == "facts_only"
    assert recipe_for("FieldSweep", infos).name == "local_overview"
    assert recipe_for("TimeSeries", infos).name == "generic_sweep"
    assert recipe_for("", infos).name == "generic_sweep"
    assert recipe_for("FieldSweep", ()) is None

    only_specific = (RecipeInfo(name="x", procedures=("FieldSweep",)),)
    assert recipe_for("TimeSeries", only_specific) is None


def test_recipe_for_matches_the_display_name_and_the_class_name():
    """The manifest names a procedure by its display name ("Field Sweep"), a
    recipe author writes the class name ("FieldSweep"): both must match, and
    the normalisation is symmetric."""
    by_class = (RecipeInfo(name="x", procedures=("FieldSweep",)),)
    by_display = (RecipeInfo(name="y", procedures=("Field Sweep",)),)
    assert recipe_for("Field Sweep", by_class).name == "x"
    assert recipe_for("FieldSweep", by_display).name == "y"
    assert recipe_for("field_sweep", by_class).name == "x"
    assert recipe_for("Temperature Sweep", by_class) is None
    assert procedure_key("Field Sweep") == procedure_key("FieldSweep") == "fieldsweep"
    assert procedure_key("---") == ""


def test_load_recipe_refuses_a_vanished_source(tmp_path):
    """A recipe whose file has gone is an AnalysisError, not an ImportError."""
    with pytest.raises(AnalysisError):
        load_recipe(RecipeInfo(name="gone", source_path=str(tmp_path / "gone.py")))


# ── The scaffold ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["", "1bad", "class", "_private", "has-dash", "two words"])
def test_scaffold_recipe_refuses_a_name_that_is_not_an_identifier(tmp_path, bad):
    """The name becomes both an id and a file name, so it must be an identifier."""
    with pytest.raises(AnalysisError):
        scaffold_recipe(bad, tmp_path)


def test_scaffold_recipe_refuses_to_overwrite(tmp_path):
    """A scaffold must never destroy an analysis somebody wrote."""
    path = scaffold_recipe("keeper", tmp_path)
    path.write_text("# edited by a physicist\n", encoding="utf-8")

    with pytest.raises(AnalysisError):
        scaffold_recipe("keeper", tmp_path)

    assert path.read_text(encoding="utf-8") == "# edited by a physicist\n"


def test_scaffolded_recipe_imports_and_runs(tmp_path, run_file):
    """The template is runnable as written: discovered, loaded, and it analyses."""
    recipes_dir = tmp_path / "recipes"
    path = scaffold_recipe(
        "sample_overview",
        recipes_dir,
        procedure="FieldSweep",
        header="written by agent runner-7 at 2026-09-05T10:00:00Z",
    )
    written = path.read_text(encoding="utf-8")
    assert written.startswith("# written by agent runner-7")
    assert "$" not in written, "every template placeholder must have been substituted"
    assert "$name" in RECIPE_TEMPLATE, "the template itself is the unsubstituted source"

    infos = discover_recipes([recipes_dir])
    scaffolded = next(info for info in infos if info.name == "sample_overview")
    assert scaffolded.procedures == ("FieldSweep",)

    report = run_spec(
        _spec(run_file, tmp_path / "report", recipe="sample_overview", recipe_dirs=(str(recipes_dir),))
    )

    assert report.status == REPORT_OK, report.error
    assert report.recipe == "sample_overview"
    assert report.summary
    assert report.figures and (tmp_path / "report" / report.figures[0].file).is_file()


# ── The runner ────────────────────────────────────────────────────────────


def test_run_spec_ok_path(tmp_path, run_file):
    """The happy path: an ok report, stamped with its provenance, plus a figure."""
    output_dir = tmp_path / "report"
    report = run_spec(_spec(run_file, output_dir, options={"note": "hello"}))

    assert report.status == REPORT_OK, report.error
    assert report.ok
    assert report.run_id == "run-1"
    assert report.recipe == "generic_sweep"
    assert len(report.recipe_digest) == 64
    assert report.started_utc.endswith("+00:00")
    assert report.duration_s >= 0.0
    assert report.options == {"note": "hello"}
    assert report.summary and "FieldSweep" in report.summary[0]
    assert report.figures and (output_dir / report.figures[0].file).is_file()
    assert report.tables and report.tables[0].columns[0] == "Column"


def test_run_spec_applies_the_spec_defaults(tmp_path, run_file):
    """A recipe that left the two flags False inherits the spec's answer."""
    report = run_spec(
        _spec(run_file, tmp_path / "report", include_fact_tables=True, attach_data_file=True)
    )
    assert report.include_fact_tables
    assert report.attach_data_file


def test_run_spec_reports_a_raising_recipe_instead_of_raising(tmp_path, run_file):
    """A crashing recipe costs a visible failure, never a lost run."""
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    (recipes_dir / "boom.py").write_text(RAISING_RECIPE, encoding="utf-8")

    report = run_spec(
        _spec(run_file, tmp_path / "report", recipe="boom", recipe_dirs=(str(recipes_dir),))
    )

    assert report.status == REPORT_FAILED
    assert report.recipe == "boom"
    assert report.recipe_digest
    assert "the fit did not converge" in report.error
    assert "Traceback" in report.error
    assert report.started_utc and report.run_id == "run-1"


def test_run_spec_reports_an_unknown_recipe(tmp_path, run_file):
    """A name nothing answers to is a failed report naming what does exist."""
    report = run_spec(_spec(run_file, tmp_path / "report", recipe="no_such_recipe"))

    assert report.status == REPORT_FAILED
    assert "no_such_recipe" in report.error
    assert "generic_sweep" in report.error


def test_run_spec_reports_a_missing_data_file(tmp_path):
    """An unreadable data file is a failed report, not an exception."""
    report = run_spec(_spec(tmp_path / "gone.h5", tmp_path / "report"))

    assert report.status == REPORT_FAILED
    assert "gone.h5" in report.error


def test_run_spec_reports_a_recipe_returning_junk(tmp_path, run_file):
    """A recipe that answers with something other than a report fails cleanly."""
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    (recipes_dir / "junk.py").write_text(
        'from i2as.analysis.base import AnalysisRecipe\n\n\n'
        'class Junk(AnalysisRecipe):\n'
        '    name = "junk"\n'
        '    procedures = ("*",)\n'
        '    description = "returns a dict"\n\n'
        '    def analyse(self, run, context):\n'
        '        return {"summary": "nope"}\n',
        encoding="utf-8",
    )

    report = run_spec(
        _spec(run_file, tmp_path / "report", recipe="junk", recipe_dirs=(str(recipes_dir),))
    )

    assert report.status == REPORT_FAILED
    assert "AnalysisReport" in report.error


def test_run_spec_collects_context_warnings(tmp_path, run_file):
    """A note a recipe left on the context survives into the report."""
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    (recipes_dir / "noisy.py").write_text(
        'from i2as.analysis.base import AnalysisRecipe\n'
        'from i2as.analysis.report import AnalysisReport\n\n\n'
        'class Noisy(AnalysisRecipe):\n'
        '    name = "noisy"\n'
        '    procedures = ("*",)\n'
        '    description = "warns"\n\n'
        '    def analyse(self, run, context):\n'
        '        context.warnings.append("a channel was missing")\n'
        '        return AnalysisReport(summary=("done",))\n',
        encoding="utf-8",
    )

    report = run_spec(
        _spec(run_file, tmp_path / "report", recipe="noisy", recipe_dirs=(str(recipes_dir),))
    )

    assert report.status == REPORT_OK
    assert "a channel was missing" in report.warnings


def test_report_round_trip_through_the_file(tmp_path, run_file):
    """write_report/read_report preserve the report; a missing one reads as None."""
    output_dir = tmp_path / "report"
    assert read_report(output_dir) is None

    report = run_spec(_spec(run_file, output_dir))
    path = write_report(report, output_dir)

    assert path == output_dir / REPORT_FILENAME
    assert read_report(output_dir) == report
    assert json.loads(path.read_text(encoding="utf-8"))["recipe"] == "generic_sweep"


def test_read_report_answers_none_for_corrupt_json(tmp_path):
    """A half-written report is 'no report yet', never an exception."""
    tmp_path.joinpath(REPORT_FILENAME).write_text("{not json", encoding="utf-8")
    assert read_report(tmp_path) is None


# ── The shipped recipes ───────────────────────────────────────────────────


def test_generic_sweep_on_a_normal_run(tmp_path, run_file):
    """One figure, one statistics table, one paragraph with span and duration."""
    report = run_spec(_spec(run_file, tmp_path / "report"))

    assert report.status == REPORT_OK
    assert len(report.figures) == 1
    table = report.tables[0]
    assert [row[0] for row in table.rows] == ["temperature_K", "voltage_V"]
    assert dict(zip(table.columns, table.rows[1]))["Unit"] == "V"
    assert {result.name for result in report.results} == {
        "field_T at first point",
        "field_T at last point",
    }
    summary = report.summary[0]
    assert "5 point(s)" in summary
    assert "-1 T to 1 T" in summary
    assert "300 s" in summary
    assert "'done'" in summary


def test_generic_sweep_on_a_run_with_no_points(tmp_path, empty_run_file):
    """An aborted-before-the-first-point run still yields an ok report."""
    report = run_spec(_spec(empty_run_file, tmp_path / "report"))

    assert report.status == REPORT_OK, report.error
    assert not report.figures
    assert any("no points" in warning for warning in report.warnings)
    assert report.summary


def test_generic_sweep_without_matplotlib(tmp_path, run_file, monkeypatch):
    """No matplotlib: an ok report, no figure, and the install command as a warning."""

    def _refuse() -> None:
        raise AnalysisError(f"matplotlib is not installed: {MATPLOTLIB_INSTALL_HINT}")

    monkeypatch.setattr("i2as.analysis.base._import_pyplot", _refuse)

    report = run_spec(_spec(run_file, tmp_path / "report"))

    assert report.status == REPORT_OK, report.error
    assert not report.figures
    assert any(MATPLOTLIB_INSTALL_HINT in warning for warning in report.warnings)
    assert report.tables, "everything that is not a figure is still produced"


def test_facts_only(tmp_path, run_file):
    """No figure, both flags set, one sentence."""
    report = run_spec(_spec(run_file, tmp_path / "report", recipe="facts_only"))

    assert report.status == REPORT_OK
    assert report.recipe == "facts_only"
    assert not report.figures
    assert not report.tables
    assert report.include_fact_tables
    assert report.attach_data_file
    assert len(report.summary) == 1
    assert "FieldSweep" in report.summary[0]


def test_every_shipped_procedure_has_a_recipe():
    """The two generic recipes are wildcards, so no run is left unanalysed,
    and the one procedure-specific recipe wins for its own procedure."""
    import pkgutil

    import i2as.procedures
    from i2as.core.procedure import BaseProcedure

    recipes = discover_recipes()
    by_name = {info.name: info for info in recipes}
    assert ANY_PROCEDURE in by_name["generic_sweep"].procedures
    assert ANY_PROCEDURE in by_name["facts_only"].procedures
    assert by_name["field_image_stack"].procedures == ("FieldImaging",)

    for module_info in pkgutil.iter_modules(i2as.procedures.__path__):
        module = importlib.import_module(f"{i2as.procedures.__name__}.{module_info.name}")
        for cls in vars(module).values():
            if not (isinstance(cls, type) and issubclass(cls, BaseProcedure)):
                continue
            if cls is BaseProcedure or cls.__module__ != module.__name__:
                continue
            chosen = recipe_for(cls.name, recipes)
            assert chosen is not None, f"{cls.__name__} has no recipe"
            expected = "field_image_stack" if cls.__name__ == "FieldImaging" else "generic_sweep"
            assert chosen.name == expected, (cls.__name__, chosen.name)


# ── The worker CLI ────────────────────────────────────────────────────────


def _worker(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run ``python -m i2as.analysis`` in a real subprocess.

    Args:
        *args: The command line after the module name.
        cwd: Working directory for the child.

    Returns:
        The completed process, with stdout and stderr captured as text.
    """
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "MPLBACKEND": "Agg"}
    return subprocess.run(
        [sys.executable, "-m", "i2as.analysis", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_worker_run_writes_the_report_and_exits_zero(tmp_path, run_file):
    """End to end: spec file in, report.json and a PNG out, path on stdout."""
    output_dir = tmp_path / "report"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec(run_file, output_dir).to_dict()), encoding="utf-8")

    result = _worker("run", "--spec", str(spec_path), cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(output_dir / REPORT_FILENAME)
    report = read_report(output_dir)
    assert report is not None and report.status == REPORT_OK
    assert (output_dir / report.figures[0].file).is_file()


def test_worker_run_exits_zero_for_a_failed_report(tmp_path, run_file):
    """A failed analysis is a complete answer; the caller reads the report."""
    output_dir = tmp_path / "report"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(_spec(run_file, output_dir, recipe="no_such_recipe").to_dict()),
        encoding="utf-8",
    )

    result = _worker("run", "--spec", str(spec_path), cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    report = read_report(output_dir)
    assert report is not None and report.status == REPORT_FAILED


def test_worker_run_exits_two_for_an_unreadable_spec(tmp_path):
    """No spec means no answer to write — the one non-zero exit."""
    result = _worker("run", "--spec", str(tmp_path / "nope.json"), cwd=tmp_path)
    assert result.returncode == 2, result.stderr
    assert "nope.json" in result.stderr


def test_worker_list_and_new_recipe(tmp_path):
    """list prints one row per recipe; new-recipe writes one and list sees it."""
    listing = _worker("list", cwd=tmp_path)
    assert listing.returncode == 0, listing.stderr
    assert "generic_sweep" in listing.stdout
    assert "facts_only" in listing.stdout

    created = _worker(
        "new-recipe", "sample_overview", "--dir", str(tmp_path / "recipes"), cwd=tmp_path
    )
    assert created.returncode == 0, created.stderr
    assert Path(created.stdout.strip()).is_file()

    again = _worker(
        "new-recipe", "sample_overview", "--dir", str(tmp_path / "recipes"), cwd=tmp_path
    )
    assert again.returncode == 2
    assert "already exists" in again.stderr

    listing = _worker("list", "--dir", str(tmp_path / "recipes"), cwd=tmp_path)
    assert "sample_overview" in listing.stdout
    assert "experiment" in listing.stdout


# ── Report values ─────────────────────────────────────────────────────────


def test_report_dict_round_trip_is_json_safe(tmp_path, run_file):
    """A real report survives json.dumps and AnalysisReport.from_dict unchanged."""
    report = run_spec(_spec(run_file, tmp_path / "report"))
    payload = json.loads(json.dumps(report.to_dict()))
    assert AnalysisReport.from_dict(payload) == report
    assert np.isfinite(report.duration_s)


# ── The image-stack recipe (Field Imaging) ───────────────────────────────


IMAGE_N_POINTS = 9
FRAME_SHAPE = (8, 10)

IMAGE_DATA_CONFIG = {
    "sweep_columns": {"unix_time": "float", "stage_x_position": "float", "field_T": "float"},
    "measurement_scalars": {"roi_mean": "float", "roi_mean_error": "float", "roi_std": "float"},
    "measurement_arrays": {"roi_mean_array": 1},
    "measurement_blocks": {"frame": FRAME_SHAPE},
    "measurement_block_labels": {},
    "measurement_image_blocks": {"frame": {"unit": "counts", "description": "test frame"}},
    "loop_shape": [1, 1],
}

IMAGE_MANIFEST = {
    "run_id": "run-img",
    "procedure": "Field Imaging",
    "kind": "run",
    "params": {"field_start": -1.0, "field_end": 1.0, "saturation_field_T": -1.5},
    "status": "done",
    "started_utc": "2026-01-01T00:00:00+00:00",
    "finished_utc": "2026-01-01T00:05:00+00:00",
}


def _image_writer(directory: Path, n_points: int = IMAGE_N_POINTS) -> DataManager:
    """Return a writer for a Field Imaging-shaped run: frames plus ROI scalars."""
    return DataManager(
        data_directory=str(directory),
        procedure_name="Field Imaging",
        procedure_params=dict(IMAGE_MANIFEST["params"]),
        sample_info={"sample_name": "Domain sample"},
        instrument_state={},
        system_targets={},
        measurement_commands=[],
        data_config=IMAGE_DATA_CONFIG,
        n_sweep_points=n_points,
        experiment_info={"setup": {"config_name": "sim_imaging"}, "experiment": {"experiment_id": "E"}},
    )


def _hysteresis(field: np.ndarray, coercive: float = 0.4) -> np.ndarray:
    """A loop in sweep order: switches up at +coercive on the way up, back at -coercive on the way down."""
    magnetisation = -1.0
    out = []
    for i, h in enumerate(field):
        going_up = i == 0 or field[i] >= field[i - 1]
        if going_up and h > coercive:
            magnetisation = 1.0
        if not going_up and h < -coercive:
            magnetisation = -1.0
        out.append(magnetisation)
    return np.asarray(out)


@pytest.fixture
def image_run_file(tmp_path) -> Path:
    """A closed Field Imaging run: -1 → +1 → -1 T, frames that switch, a loop."""
    writer = _image_writer(tmp_path / "imaging")
    field = np.concatenate([np.linspace(-1.0, 1.0, 5), np.linspace(0.5, -1.0, 4)])
    loop = _hysteresis(field)
    for index in range(IMAGE_N_POINTS):
        frame = np.full(FRAME_SHAPE, 100.0 + 50.0 * loop[index])
        frame[:, : FRAME_SHAPE[1] // 2] += 1.0 * index  # something spatial changes too
        writer.save_datapoint(
            index,
            {
                "unix_time": 1_000.0 + index,
                "stage_x_position": 0.0,
                "field_T": float(field[index]),
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


@pytest.fixture
def empty_image_run_file(tmp_path) -> Path:
    writer = _image_writer(tmp_path / "empty_imaging")
    writer.close()
    return Path(writer.filepath)


def _image_spec(run_file: Path, output_dir: Path, **overrides) -> AnalysisSpec:
    overrides.setdefault("recipe", "field_image_stack")
    return _spec(run_file, output_dir, manifest=dict(IMAGE_MANIFEST), **overrides)


def test_coercive_fields_reads_both_crossings_of_a_loop():
    from i2as.analysis.recipes.field_image_stack import coercive_fields

    field = np.concatenate([np.linspace(-1.0, 1.0, 21), np.linspace(1.0, -1.0, 21)])
    crossings = coercive_fields(field, _hysteresis(field, coercive=0.45))
    assert len(crossings) == 2
    assert 0.4 < crossings[0] <= 0.5
    assert -0.5 <= crossings[1] < -0.4


def test_coercive_fields_needs_a_real_switch():
    """A constant, noise alone, or a single point yields no crossing."""
    from i2as.analysis.recipes.field_image_stack import coercive_fields

    field = np.linspace(-1.0, 1.0, 11)
    assert coercive_fields(field, np.full(11, 3.0)) == []
    noise = 0.1 * np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    assert coercive_fields(field, 3.0 + noise) == []
    assert coercive_fields(field[:1], np.array([1.0])) == []
    step = np.where(field > 0.25, 1.0, -1.0)
    assert len(coercive_fields(field, step)) == 1


def test_panel_indices_keep_the_ends_under_the_cap():
    from i2as.analysis.recipes.field_image_stack import MAX_PANELS, _panel_indices

    assert _panel_indices(0) == []
    assert _panel_indices(5) == [0, 1, 2, 3, 4]
    indices = _panel_indices(101)
    assert len(indices) == MAX_PANELS
    assert indices[0] == 0 and indices[-1] == 100
    assert indices == sorted(set(indices))


def test_field_image_stack_on_an_imaging_run(tmp_path, image_run_file):
    """Three figures, the crossings as results, one paragraph."""
    report = run_spec(_image_spec(image_run_file, tmp_path / "report"))

    assert report.status == REPORT_OK, report.error
    assert report.recipe == "field_image_stack"
    assert [f.file for f in report.figures] == ["montage.png", "difference.png", "loop.png"]
    for figure in report.figures:
        assert (tmp_path / "report" / figure.file).stat().st_size > 0
    names = [r.name for r in report.results]
    assert names[:2] == ["Coercive field (crossing 1)", "Coercive field (crossing 2)"]
    assert "Coercive field" in names and "Loop width" in names
    by_name = {r.name: r for r in report.results}
    # The fixture switches between the 0 T and ±0.5 T points, so linear
    # interpolation puts each crossing near ±0.25 T.
    assert 0.2 < by_name["Coercive field (crossing 1)"].value < 0.3
    assert -0.3 < by_name["Coercive field (crossing 2)"].value < -0.2
    assert by_name["Coercive field"].unit == "T"
    assert by_name["Coercive field"].uncertainty is not None
    summary = report.summary[0]
    assert "9 point(s)" in summary
    assert "montage shows 9" in summary
    assert "reference frame" in summary
    assert "crosses zero" in summary
    assert not report.warnings


def test_field_image_stack_is_chosen_for_a_field_imaging_run(tmp_path, image_run_file):
    """With no recipe named, the procedure-specific recipe wins over the wildcard."""
    report = run_spec(_image_spec(image_run_file, tmp_path / "report", recipe=""))
    assert report.status == REPORT_OK, report.error
    assert report.recipe == "field_image_stack"


def test_field_image_stack_without_frames_still_draws_the_loop(tmp_path, run_file):
    """A run of another measurement method: no montage, a warning, the loop from its scalar."""
    report = run_spec(_image_spec(run_file, tmp_path / "report"))

    assert report.status == REPORT_OK, report.error
    assert [f.file for f in report.figures] == ["loop.png"]
    assert any("no image block" in w for w in report.warnings)
    assert "holds no frames" in report.summary[0]


def test_field_image_stack_on_a_run_with_no_points(tmp_path, empty_image_run_file):
    report = run_spec(_image_spec(empty_image_run_file, tmp_path / "report"))

    assert report.status == REPORT_OK, report.error
    assert not report.figures and not report.results
    assert any("no points" in w for w in report.warnings)
    assert report.summary


def test_field_image_stack_without_matplotlib(tmp_path, image_run_file, monkeypatch):
    """No matplotlib: an ok report, no figure, the install hint once, results intact."""

    def _refuse() -> None:
        raise AnalysisError(f"matplotlib is not installed: {MATPLOTLIB_INSTALL_HINT}")

    monkeypatch.setattr("i2as.analysis.base._import_pyplot", _refuse)

    report = run_spec(_image_spec(image_run_file, tmp_path / "report"))

    assert report.status == REPORT_OK, report.error
    assert not report.figures
    assert sum(MATPLOTLIB_INSTALL_HINT in w for w in report.warnings) == 1
    assert any(r.name == "Coercive field" for r in report.results)
