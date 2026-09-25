# i2as/analysis — the analysis stage

## Purpose

Turns one finished run into one **analysis report**: the prose, derived values,
figures and small tables that belong in an electronic lab notebook, instead of
the run's raw fact tables. Two kinds of code produce a report:

- a **recipe** (`base.AnalysisRecipe`), which is reviewed and reusable. It is
  discovered, chosen by procedure, and its report becomes the run's pending
  notebook entry;
- an **analysis script** (`scripts.py`), which is exploratory code that an agent or a
  physicist runs once over one run to find out what the data says. Its report
  is parked only when someone decides it should be (`stage_analysis_result`),
  and a script that proved its worth is saved as a recipe (`ScriptRecipe`).

## Architecture layer

Beside the session layer and above everything the engine is made of. It
imports only `i2as.core.data_reader`, `i2as.core.events` and
`i2as.core.exceptions`, plus numpy, h5py, the standard library and, lazily,
matplotlib (contract C22). Nothing below the session layer imports it (the
mirror contract). All of its code runs in the **analysis worker**
(`python -m i2as.analysis`), a separate process that the session layer starts
in the configured **analysis sandbox** (`i2as/session/analysis_sandbox.py`).
The worker can reach one run file and one output folder, and it cannot reach
the Station, the engine or the notebook.

## Entry

- `python -m i2as.analysis run --spec <spec.json>`: the worker. The spec
  (`report.AnalysisSpec`) names the run file, the manifest, and either a
  `recipe` or a `script_path`.
- `runner.run_spec(spec)`: the same work in process, which is what the tests call.
- `python -m i2as.analysis new-recipe <name> --dir <folder>`: scaffold a
  recipe. `python -m i2as.analysis list`: list the recipes.

## Exit

- `report.json` in the spec's output folder: the `AnalysisReport`,
  always written. A recipe or script that fails still produces a report, with
  status `failed` and its traceback.
- The figures the report names, saved beside it as PNG.
- For a script, and for a `ScriptRecipe`: `stdout.txt`, with what it printed,
  capped at `scripts.MAX_STDOUT_CHARS`.

## Interface contract

- **Recipe contract** (`base.py`): one class, `name` / `description` /
  `procedures` / optional `priority`, and `analyse(run, context) ->
  AnalysisReport`. It reads the run only through `RunSource` and writes only
  into `context.output_dir`.
- **Script contract** (`scripts.py`): a module body executed with `run`,
  `context`, `report` (a `ScriptReport` builder with `summary`, `value`,
  `figure`, `table`, `tag` and `warn`), `manifest`, `options` and `np` bound.
  A script may also bind `report` to an `AnalysisReport` of its own.
- **Report standard** (`report.py`): frozen, JSON-safe and capped types that
  load tolerantly.
- A failure is data: `run_spec()` never raises.

## How to add

- **A recipe:** `python -m i2as.analysis new-recipe my_fit --dir <experiment>/analysis/recipes`,
  or add a module under `recipes/`. Give it a `priority` below `generic_sweep`'s
  (10) with `procedures = ("*",)` if it should run only when asked for by name,
  as `magnetoresistance` does.
- **A script:** call `run_analysis_script` through the gateway (MCP, CLI or
  the embedded agent), then `save_analysis_script_as_recipe` to keep it. In
  code, `scripts.render_script_recipe(name, source, procedures=...)` writes
  the recipe module.

## Files

| File | What it holds |
|---|---|
| `__init__.py` | The package's public names. |
| `__main__.py` | The worker's command line: `run`, `new-recipe`, `list`. |
| `base.py` | The recipe contract, `AnalysisContext`, the axis and column conventions. |
| `discovery.py` | Finding recipes in the package and in an experiment's folder; `recipe_for`. |
| `report.py` | `AnalysisReport`, `AnalysisSpec` and the size caps. |
| `runner.py` | `run_spec()`: one spec, one report, never raising. |
| `scripts.py` | Analysis scripts: `run_script`, `ScriptReport`, `ScriptRecipe`, `render_script_recipe`. |
| `recipes/generic_sweep.py` | The default overview of any run. |
| `recipes/facts_only.py` | The run's fact tables only; runs when asked for by name. |
| `recipes/field_image_stack.py` | Field Imaging: a hysteresis loop from an image stack. |
| `recipes/magnetoresistance.py` | R(B) fitted for R0, the MR coefficient and MR%; runs when asked for by name. |
