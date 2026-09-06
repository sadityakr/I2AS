"""i2as.analysis — the analysis stage between a finished run and its notebook entry.

A **recipe** (``base.AnalysisRecipe``) reads one finished run through the
standalone data reader and returns one **analysis report** (``report.AnalysisReport``):
prose, derived values, figures and small tables — the concise, analysed
content that belongs in an electronic lab notebook, instead of the run's
raw fact tables. Recipes are discovered (``discovery``) from this package's
``recipes/`` folder and from an experiment folder's own ``analysis/recipes/``
scripts, and they run in the **analysis worker** (``python -m i2as.analysis``),
a separate process that can reach the data file and nothing else.

Layer rule: this package imports only the data reader, the control-contract
vocabulary and the exceptions from ``i2as.core`` (plus numpy, h5py, stdlib
and — lazily, optionally — matplotlib). It never imports the Station, the
Orchestrator, a driver, a VI, a procedure, the session layer or the GUI, and
nothing below the session layer imports it. See ``README.md`` here.
"""

from i2as.analysis.base import (
    AnalysisContext,
    AnalysisError,
    AnalysisRecipe,
)
from i2as.analysis.discovery import (
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
    RECIPES_DIRNAME,
    SPEC_FILENAME,
    AnalysisReport,
    AnalysisSpec,
    FigureRef,
    ResultValue,
    TableSpec,
)

from i2as.analysis.runner import read_report, run_spec, write_report

__all__ = [
    "ANY_PROCEDURE",
    "RECIPE_TEMPLATE",
    "REPORT_FAILED",
    "REPORT_FILENAME",
    "REPORT_OK",
    "RECIPES_DIRNAME",
    "SPEC_FILENAME",
    "AnalysisContext",
    "AnalysisError",
    "AnalysisRecipe",
    "AnalysisReport",
    "AnalysisSpec",
    "FigureRef",
    "RecipeInfo",
    "ResultValue",
    "TableSpec",
    "discover_recipes",
    "load_recipe",
    "read_report",
    "procedure_key",
    "recipe_for",
    "run_spec",
    "scaffold_recipe",
    "write_report",
]
