# i2as/analysis — the analysis stage

## Purpose

Turns one finished run into one **analysis report**: the prose, derived values,
figures and small tables worth keeping, instead of the run's raw fact tables.
The application seals what the worker leaves into an **analysis bundle**
(`bundle.py`): the manifest (`bundle.json`) every later reader uses, with every
file hashed. This package knows no notebook; the notebook layer reads bundles.
Two kinds of code produce a report:

- a **recipe** (`base.AnalysisRecipe`), which is reviewed and reusable. It is
  discovered, chosen by procedure, and its completed bundle becomes the run's
  selected bundle — the result that represents the run;
- an **analysis script** (`scripts.py`), which is exploratory code that an agent or a
  physicist runs once over one run to find out what the data says. Its bundle
  represents the run only when someone chooses it (`select_analysis_bundle`),
  and a script that proved its worth is saved as a recipe (`ScriptRecipe`).

## Architecture layer

Beside the session layer and above everything the engine is made of. It
imports only `i2as.core.data_reader`, `i2as.core.events` and
`i2as.core.exceptions`, plus numpy, h5py, the standard library and, lazily,
matplotlib (contract C22). Nothing below the session layer imports it (the
mirror contract). All of its code runs in the **analysis worker**
(`python -m i2as.analysis`), a separate process that the session layer starts
in a container (the **analysis sandbox**, `i2as/session/analysis_sandbox.py`):
no network, and only the run file, its output folder and the experiment's
recipes mounted. The worker cannot reach the Station, the engine or the
notebook. The image is built from `container/Dockerfile` here, over a context
holding only this package and the three core modules above.

## Entry

- `python -m i2as.analysis run --spec <spec.json>`: the worker. The spec
  (`report.AnalysisSpec`) names the run file, the manifest, and either a
  `recipe` or a `script_path`.
- `runner.run_spec(spec)`: the same work in process, which is what the tests call.
- `python -m i2as.analysis new-recipe <name> --dir <folder>`: scaffold a
  recipe. `python -m i2as.analysis list`: list the recipes.

## Exit

- `report.json` in the spec's output folder: the `AnalysisReport` (the
  worker's CLAIMS), always written. The application then seals the folder:
  `bundle.json` (`bundle.Bundle`, schema `i2as.analysis-bundle` v1) lists every
  file with its SHA-256 and carries the report's content, the producer, the
  inputs and the status. Each analysis gets a folder of its own,
  `analysis/<run>/<bundle_id>/`; a script's is `analysis/<run>/scripts/<id>/`. A recipe or script that fails still produces a report, with
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
- **Bundle standard** (`bundle.py`): the sealed hand-off out of this stage —
  standard library only (contract C22 and C25), so the notebook layer and a
  user's renderer read it without importing anything else here.
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
| `bundle.py` | The analysis bundle: `Bundle`, `seal_bundle`, `read_bundle`, `verify_artifact`. |
| `discovery.py` | Finding recipes in the package and in an experiment's folder; `recipe_for`. |
| `report.py` | `AnalysisReport`, `AnalysisSpec` and the size caps. |
| `runner.py` | `run_spec()`: one spec, one report, never raising. |
| `scripts.py` | Analysis scripts: `run_script`, `ScriptReport`, `ScriptRecipe`, `render_script_recipe`. |
| `recipes/generic_sweep.py` | The default overview of any run. |
| `recipes/facts_only.py` | The run's fact tables only; runs when asked for by name. |
| `recipes/field_image_stack.py` | Field Imaging: a hysteresis loop from an image stack. |
| `recipes/magnetoresistance.py` | R(B) fitted for R0, the MR coefficient and MR%; runs when asked for by name. |
