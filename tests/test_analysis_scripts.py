"""Analysis scripts and the magnetoresistance recipe, in the worker's own code.

Two things an agent analyses a finished run with, exercised exactly as the
worker runs them — ``run_spec()`` in process, no subprocess in the way:

* **analysis scripts** (``i2as/analysis/scripts.py``): exploratory code run
  once over one run, answered by the same report a recipe returns, with what
  it printed captured beside it — and, once it proved its worth, saved as an
  ordinary recipe (``ScriptRecipe``) that discovery finds and runs;
* the **magnetoresistance** recipe (``recipes/magnetoresistance.py``),
  against synthetic field sweeps whose R0, odd term and MR coefficient are
  known, so the fit is checked against the answer rather than against itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from i2as.analysis.discovery import discover_recipes, recipe_for
from i2as.analysis.report import REPORT_FAILED, REPORT_OK, AnalysisReport, AnalysisSpec
from i2as.analysis.runner import run_spec
from i2as.analysis.scripts import (
    MAX_STDOUT_CHARS,
    STDOUT_FILENAME,
    render_script_recipe,
    run_script,
)
from i2as.core.data_manager import DataManager

R0, C1, C2 = 100.0, 0.5, 3.0
CURRENT_A = 1e-6
FIELDS = np.linspace(-2.0, 2.0, 41)


def _write_run(
    directory: Path,
    columns: dict[str, str],
    rows: list[dict],
    *,
    params: dict | None = None,
    loop_shape: tuple[int, int] = (1, 1),
    procedure: str = "Field Sweep",
) -> Path:
    """Write one closed run file with the given measurement columns.

    Args:
        directory: Where to write it.
        columns: ``{measurement scalar: dtype}``.
        rows: One dict per point, sweep columns included.
        params: The procedure parameters recorded with the run.
        loop_shape: The reading-loop grid every measurement column carries.
        procedure: The procedure name.

    Returns:
        The run file's path.
    """
    writer = DataManager(
        data_directory=str(directory),
        procedure_name=procedure,
        procedure_params=dict(params or {}),
        sample_info={"sample_name": "S"},
        instrument_state={},
        system_targets={},
        measurement_commands=[],
        data_config={
            "sweep_columns": {"unix_time": "float", "field_T": "float"},
            "measurement_scalars": dict(columns),
            "measurement_arrays": {},
            "measurement_blocks": {},
            "loop_shape": list(loop_shape),
        },
        n_sweep_points=len(rows),
        experiment_info={"setup": {}, "experiment": {}},
    )
    for index, row in enumerate(rows):
        values = {
            name: [[value]] if name in columns and np.ndim(value) == 0 else value
            for name, value in row.items()
        }
        writer.save_datapoint(index, {"unix_time": float(index), **values}, {})
    writer.close()
    return Path(writer.filepath)


def _true_resistance(field: np.ndarray | float) -> np.ndarray:
    return R0 + C1 * np.asarray(field) + C2 * np.asarray(field) ** 2


@pytest.fixture
def reversal_run(tmp_path) -> Path:
    """A field sweep read at +I and -I per point, with a thermal offset and noise."""
    rng = np.random.default_rng(1)
    rows = []
    for field in FIELDS:
        resistance = float(_true_resistance(field))
        rows.append(
            {
                "field_T": float(field),
                "voltage_V": [
                    [resistance * CURRENT_A + 5e-6 + rng.normal(0, 1e-9)],
                    [-resistance * CURRENT_A + 5e-6 + rng.normal(0, 1e-9)],
                ],
            }
        )
    return _write_run(
        tmp_path / "data",
        {"voltage_V": "float"},
        rows,
        params={"loop1_parameter": "current_A", "loop1_values": [CURRENT_A, -CURRENT_A]},
        loop_shape=(2, 1),
    )


def _spec(run_file: Path, output_dir: Path, **overrides) -> AnalysisSpec:
    fields = {
        "run_id": "run-1",
        "data_path": str(run_file),
        "output_dir": str(output_dir),
        "manifest": {"procedure": "Field Sweep"},
    }
    fields.update(overrides)
    return AnalysisSpec(**fields)


def _values(report: AnalysisReport) -> dict[str, object]:
    return {result.name: result for result in report.results}


# ── Analysis scripts ──────────────────────────────────────────────────────


def _script(tmp_path: Path, text: str, name: str = "probe_1") -> Path:
    folder = tmp_path / "scripts" / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.py"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_script_builds_a_report_and_its_output_is_captured(tmp_path, reversal_run):
    """The headline: values, a figure, prose — and what it printed, beside it."""
    script = _script(
        tmp_path,
        "field = run.read_slice('field_T')\n"
        "print('points', run.n_points, 'option', options['k'])\n"
        "report.summary(f'{run.n_points} points')\n"
        "report.value('Field span', float(np.ptp(field)), 'T', note='max - min')\n"
        "plt = context.pyplot()\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot(field, field)\n"
        "report.figure('span', fig, 'the axis')\n"
        "report.table('t', ['a'], [[1]])\n"
        "report.tag('explored')\n",
    )

    report = run_spec(_spec(reversal_run, script.parent, script_path=str(script), options={"k": 7}))

    assert report.status == REPORT_OK, report.error
    assert report.recipe == "script:probe_1"
    assert len(report.recipe_digest) == 64
    assert report.summary == ("41 points",)
    assert _values(report)["Field span"].value == pytest.approx(4.0)
    assert [figure.file for figure in report.figures] == ["span.png"]
    assert (script.parent / "span.png").is_file()
    assert report.tags == ("explored",)
    assert report.options == {"k": 7}
    assert (script.parent / STDOUT_FILENAME).read_text(encoding="utf-8") == "points 41 option 7\n"


def test_a_raising_script_is_a_failed_report_with_its_output_kept(tmp_path, reversal_run):
    """What it printed before the failure is usually what explains it."""
    script = _script(tmp_path, "print('about to fail')\nraise ValueError('bad column')\n")

    report = run_script(_spec(reversal_run, script.parent, script_path=str(script)))

    assert report.status == REPORT_FAILED
    assert report.error.startswith("ValueError: bad column")
    assert "Traceback" in report.error
    assert (script.parent / STDOUT_FILENAME).read_text(encoding="utf-8") == "about to fail\n"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("def broken(:\n", "does not compile"), (None, "no readable analysis script")],
)
def test_a_script_that_cannot_run_answers_with_one_sentence(tmp_path, reversal_run, text, expected):
    """Framework-level failures say what happened, never raise."""
    path = _script(tmp_path, text) if text else tmp_path / "missing.py"

    report = run_script(_spec(reversal_run, tmp_path, script_path=str(path)))

    assert report.status == REPORT_FAILED
    assert expected in report.error


def test_a_script_may_bind_its_own_report(tmp_path, reversal_run):
    """A script that builds an AnalysisReport itself has it used as it stands."""
    script = _script(
        tmp_path,
        "from i2as.analysis.report import AnalysisReport\n"
        "report = AnalysisReport(summary=('own',))\n",
    )

    report = run_script(_spec(reversal_run, script.parent, script_path=str(script)))

    assert report.summary == ("own",)
    assert report.recipe == "script:probe_1", "provenance is stamped all the same"


def test_a_scripts_output_is_capped(tmp_path, reversal_run):
    """A loop printing forever does not become a megabyte in an agent's context."""
    script = _script(tmp_path, "for i in range(20000):\n    print('line', i)\n")

    run_script(_spec(reversal_run, script.parent, script_path=str(script)))

    text = (script.parent / STDOUT_FILENAME).read_text(encoding="utf-8")
    assert len(text) < MAX_STDOUT_CHARS + 200
    assert "more characters of output dropped" in text


def test_a_saved_script_is_discovered_and_run_as_an_ordinary_recipe(tmp_path, reversal_run):
    """Explore once, keep it: the same text, now chosen by procedure."""
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    source = "report.value('n', run.n_points)\nprint('kept')\n"
    (recipes_dir / "point_count.py").write_text(
        render_script_recipe(
            "point_count", source, procedures=("FieldSweep",), description="Counts points"
        ),
        encoding="utf-8",
    )

    recipes = discover_recipes([recipes_dir])
    chosen = recipe_for("Field Sweep", recipes)
    assert chosen is not None and chosen.name == "point_count"
    assert chosen.origin == "experiment"

    report = run_spec(
        _spec(reversal_run, tmp_path / "out", recipe_dirs=(str(recipes_dir),))
    )

    assert report.status == REPORT_OK, report.error
    assert report.recipe == "point_count"
    assert _values(report)["n"].value == 41
    assert (tmp_path / "out" / STDOUT_FILENAME).read_text(encoding="utf-8") == "kept\n"


@pytest.mark.parametrize(
    "source",
    ["x = 1\n", "s = '''quoted'''\n", "p = 'c:\\\\'\n", "q = 'ends with a quote'"],
)
def test_a_rendered_recipe_carries_the_script_byte_for_byte(source):
    """Whatever the script's quoting, SCRIPT is exactly the text that was explored."""
    namespace: dict = {}
    exec(compile(render_script_recipe("kept", source), "<kept>", "exec"), namespace)

    assert namespace["KeptRecipe"].SCRIPT == source
    assert namespace["KeptRecipe"].procedures == ("*",)


# ── The magnetoresistance recipe ──────────────────────────────────────────


def test_the_mr_fit_recovers_a_known_magnetoresistance(tmp_path, reversal_run):
    """Current reversal cancels the offset; the fit returns the parameters put in."""
    report = run_spec(_spec(reversal_run, tmp_path / "out", recipe="magnetoresistance"))

    assert report.status == REPORT_OK, report.error
    assert report.recipe == "magnetoresistance"
    values = _values(report)
    r0 = values["Zero-field resistance R0"]
    assert r0.value == pytest.approx(R0, abs=1e-3)
    assert 0 < r0.uncertainty < 1e-2
    assert values["MR coefficient c2 (× B²)"].value == pytest.approx(C2, abs=1e-3)
    assert values["Odd-in-field term c1 (× B)"].value == pytest.approx(C1, abs=1e-3)
    assert values["MR at |B| = 2 T"].value == pytest.approx(C2 * 4 / R0 * 100, rel=1e-3)
    assert values["Points fitted"].value == 41
    assert [figure.file for figure in report.figures] == ["magnetoresistance_fit.png"]
    assert (tmp_path / "out" / "magnetoresistance_fit.png").is_file()
    assert "reading loop's currents" in report.summary[0]
    assert report.warnings == ()


def test_the_mr_fit_reads_a_resistance_column_as_it_stands(tmp_path):
    """A run that already recorded R needs no current at all."""
    rows = [{"field_T": float(b), "resistance_ohm": float(_true_resistance(b))} for b in FIELDS]
    run_file = _write_run(tmp_path / "data", {"resistance_ohm": "float"}, rows)

    report = run_spec(_spec(run_file, tmp_path / "out", recipe="magnetoresistance"))

    assert _values(report)["Zero-field resistance R0"].value == pytest.approx(R0)
    assert "'resistance_ohm' column" in report.summary[0]


def test_the_mr_fit_takes_a_fixed_current_from_its_options(tmp_path):
    """A voltage-only run is analysed when the caller says what current flowed."""
    rows = [
        {"field_T": float(b), "voltage_V": float(_true_resistance(b)) * CURRENT_A}
        for b in FIELDS
    ]
    run_file = _write_run(tmp_path / "data", {"voltage_V": "float"}, rows)
    spec = _spec(run_file, tmp_path / "out", recipe="magnetoresistance")

    without = run_spec(spec)
    with_current = run_spec(_spec(
        run_file, tmp_path / "out2", recipe="magnetoresistance", options={"current_A": CURRENT_A}
    ))

    assert any("no current" in warning for warning in without.warnings)
    assert without.recipe == "magnetoresistance", "the fallback is still this recipe's answer"
    assert _values(with_current)["Zero-field resistance R0"].value == pytest.approx(R0)


def test_the_linear_mr_model_and_the_fit_window(tmp_path):
    """model='abs_linear' fits |B|; fit_range_T keeps only the low-field points."""
    rows = [
        {"field_T": float(b), "resistance_ohm": R0 + 2.0 * abs(float(b))} for b in FIELDS
    ]
    run_file = _write_run(tmp_path / "data", {"resistance_ohm": "float"}, rows)

    report = run_spec(_spec(
        run_file,
        tmp_path / "out",
        recipe="magnetoresistance",
        options={"model": "abs_linear", "fit_range_T": 1.0},
    ))

    values = _values(report)
    assert values["MR coefficient c2 (× |B|)"].value == pytest.approx(2.0)
    assert values["Points fitted"].value == 21
    assert values["MR at |B| = 1 T"].value == pytest.approx(2.0)


def test_a_run_with_nothing_to_fit_falls_back_to_the_overview(tmp_path):
    """No voltage and no resistance: the overview, and a warning saying why."""
    rows = [{"field_T": float(b), "temperature_K": 4.2} for b in FIELDS]
    run_file = _write_run(tmp_path / "data", {"temperature_K": "float"}, rows)

    report = run_spec(_spec(run_file, tmp_path / "out", recipe="magnetoresistance"))

    assert report.status == REPORT_OK
    assert report.warnings[0].startswith("no magnetoresistance fit: the run has neither")
    assert [figure.file for figure in report.figures] == ["overview.png"]


def test_the_mr_recipe_is_asked_for_never_assumed():
    """Discovery's default for a field sweep stays the overview."""
    recipes = discover_recipes()

    assert recipe_for("Field Sweep", recipes).name == "generic_sweep"
    assert recipe_for("Field Sweep", recipes, preferred="magnetoresistance").name == (
        "magnetoresistance"
    )
