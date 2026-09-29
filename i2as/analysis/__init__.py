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

The public names below are resolved lazily, on first use, so that importing
``i2as.analysis.bundle`` alone — as the publisher and the block host do —
does not import numpy, h5py or the recipe machinery.
"""

from __future__ import annotations

import importlib
from typing import Any

#: Public name -> the submodule that defines it.
_EXPORTS: dict[str, str] = {
    "AnalysisContext": "base",
    "AnalysisError": "base",
    "AnalysisRecipe": "base",
    "RECIPE_TEMPLATE": "discovery",
    "RecipeInfo": "discovery",
    "discover_recipes": "discovery",
    "load_recipe": "discovery",
    "procedure_key": "discovery",
    "recipe_for": "discovery",
    "scaffold_recipe": "discovery",
    "ANY_PROCEDURE": "report",
    "REPORT_FAILED": "report",
    "REPORT_FILENAME": "report",
    "REPORT_OK": "report",
    "RECIPES_DIRNAME": "report",
    "SCRIPTS_DIRNAME": "report",
    "SPEC_FILENAME": "report",
    "AnalysisReport": "report",
    "AnalysisSpec": "report",
    "FigureRef": "report",
    "ResultValue": "report",
    "TableSpec": "report",
    "read_report": "runner",
    "run_spec": "runner",
    "write_report": "runner",
    "Bundle": "bundle",
    "read_bundle": "bundle",
    "seal_bundle": "bundle",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve one public name from its submodule, on first use."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
