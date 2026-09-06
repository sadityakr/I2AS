"""Finding recipes — the package's own, and an experiment's.

Recipes live in two places under one contract (see ``base.py``'s recipe
contract): the shipped ones in ``i2as/analysis/recipes/``, discovered the
same way drivers, virtual instruments and procedures are — walk the package,
import every module, collect every ``AnalysisRecipe`` subclass that declares a
``name`` — and an experiment's own scripts in
``<experiment>/analysis/recipes/*.py``, written at runtime by a physicist or
an agent and loaded BY FILE PATH, because they are not importable modules of
any package.

Three rules make that second half safe to live with:

- **Order is the selection order.** ``recipe_for()`` takes the FIRST recipe
  that matches, so discovery returns package recipes ordered by declared
  ``priority`` (highest first, then by name) — which is how the shipped
  ``generic_sweep`` overview, and not the opt-in ``facts_only``, is what a
  run with no configured recipe gets.
- **An experiment recipe overrides a package recipe of the same name.** The
  local answer wins, so an experiment can fix a shipped recipe for itself
  without editing the application.
- **A script that fails to import is skipped with a WARNING, never raised.**
  One half-written file in a folder must not make every other recipe
  undiscoverable — the panel and the agent's recipe list have to keep working
  while somebody is editing.
- **Every recipe is fingerprinted.** ``RecipeInfo.digest`` is the SHA-256 of
  the source file, and it is stamped into the report, so an entry says
  exactly which code produced it even after the file has been edited.

``scaffold_recipe()`` is the other direction: it writes a new, commented,
runnable recipe from ``RECIPE_TEMPLATE`` for a physicist or an agent to edit.
It refuses a name that is not an identifier and refuses to overwrite, which is
what makes it safe to expose as an agent tool.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import keyword
import logging
import pkgutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from i2as.analysis.base import AnalysisError, AnalysisRecipe
from i2as.analysis.report import ANY_PROCEDURE

logger = logging.getLogger(__name__)

#: A recipe shipped in ``i2as/analysis/recipes/``.
ORIGIN_PACKAGE = "package"

#: A recipe loaded from an experiment's own ``analysis/recipes/`` folder.
ORIGIN_EXPERIMENT = "experiment"

#: The module-name prefix experiment scripts are imported under, so a script
#: never collides with a real package module (or with another experiment's
#: script of the same file name).
_EXPERIMENT_MODULE_PREFIX = "i2as_analysis_experiment_recipe"

#: The commented, runnable recipe ``scaffold_recipe()`` writes. Substituted
#: with ``string.Template`` (``$name``), not ``str.format``, so the template
#: body can contain all the braces real Python needs.
RECIPE_TEMPLATE: str = '''"""$name — $description

An experiment recipe: I2AS discovered this file because it sits in this
experiment's ``analysis/recipes/`` folder. Edit it freely — it belongs to the
experiment, not to the application, and a recipe here overrides a shipped one
of the same ``name``.

The rules (the recipe contract, in ``i2as/analysis/base.py``): one class,
one ``analyse()`` method, one ``AnalysisReport`` back. It runs in the analysis
worker — a separate process that can read this run's data file and nothing
else — so it can never touch an instrument, and if it raises, the failure is
reported instead of being lost.
"""

from __future__ import annotations

import numpy as np

from i2as.analysis.base import (
    AnalysisContext,
    AnalysisError,
    AnalysisRecipe,
    axis_label,
    choose_x_column,
    measured_columns,
)
from i2as.analysis.report import AnalysisReport, ResultValue
from i2as.core.data_reader import RunSource


class $class_name(AnalysisRecipe):
    """$description"""

    # The unique id this recipe is selected by, in the panel, in the eLab
    # settings and in the agent's tools. Keep it snake_case.
    name = "$name"

    # The procedure CLASS NAMES this recipe serves — ("FieldSweep",) for one
    # procedure, ("*",) for every run.
    procedures = ($procedures)

    # One line, shown wherever the recipe is listed.
    description = "$description"

    def analyse(self, run: RunSource, context: AnalysisContext) -> AnalysisReport:
        """Analyse one run and return one report.

        Args:
            run: The finished run, read through the run-source vocabulary:
                ``run.n_points``, ``run.list_columns()``,
                ``run.read_slice(column)``, ``run.summary_stats(column)``,
                ``run.read_metadata()``.
            context: The run manifest, the experiment and setup facts, the
                output directory, the options, and the figure/table helpers.

        Returns:
            The report: prose paragraphs, derived values, figures and small
            tables.
        """
        # ── 1. What to plot against, and what to plot ────────────────────
        # choose_x_column() follows the axis convention: the procedure's own
        # sweep axis when the run declares one, else an elapsed-time column,
        # else None (meaning "use the point index").
        x_info = choose_x_column(run, context.manifest)
        y_infos = measured_columns(run, exclude=[x_info.name] if x_info else [])
        n_points = run.n_points

        if x_info is not None:
            x_values = np.asarray(run.read_slice(x_info.name), dtype=float)
        else:
            x_values = np.arange(n_points, dtype=float)

        figures = []
        warnings = []

        # ── 2. One figure ────────────────────────────────────────────────
        # matplotlib is an OPTIONAL extra, so asking for it may fail. Catch
        # that, note it, and still return a useful report.
        if n_points and y_infos:
            try:
                plt = context.pyplot()
                fig, axes = plt.subplots(figsize=(7.0, 4.0))
                for info in y_infos[:4]:
                    values = np.asarray(run.read_slice(info.name), dtype=float)
                    # A measurement column may carry reading-loop axes after
                    # the sweep axis; average them away for the overview.
                    if values.ndim > 1:
                        values = np.nanmean(values.reshape(len(values), -1), axis=1)
                    axes.plot(x_values[: len(values)], values, marker=".", label=axis_label(info))
                axes.set_xlabel(axis_label(x_info))
                axes.set_ylabel("measured value")
                axes.legend(fontsize="small")
                axes.grid(True, alpha=0.3)
                fig.tight_layout()
                figures.append(context.figure("overview", fig, caption="Measured columns."))
            except AnalysisError as exc:
                warnings.append(str(exc))
        elif not n_points:
            warnings.append("the run has no written points, so nothing was plotted")

        # ── 3. One derived value ─────────────────────────────────────────
        # Replace this with the number this experiment actually cares about:
        # a critical field, a fitted slope, a transition width.
        results = []
        if y_infos and n_points:
            stats = run.summary_stats(y_infos[0].name)
            results.append(
                ResultValue(
                    name=f"Mean {y_infos[0].name}",
                    value=stats.mean,
                    unit=y_infos[0].unit,
                    note=f"over {stats.count} finite values",
                )
            )

        # ── 4. One paragraph of prose ────────────────────────────────────
        procedure = str(context.manifest.get("procedure", "")) or "run"
        summary = (
            f"{procedure} {context.run_id or ''} recorded {n_points} points "
            f"of {len(y_infos)} measured column(s) against {axis_label(x_info)}."
        )

        return AnalysisReport(
            summary=(summary,),
            results=tuple(results),
            figures=tuple(figures),
            warnings=tuple(warnings),
        )
'''


def _digest_of(path: Path) -> str:
    """Return the SHA-256 of a source file, or ``""`` when it cannot be read.

    Args:
        path: The recipe's source file.

    Returns:
        The hex digest, or ``""`` when the file could not be read.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        logger.warning("analysis discovery: cannot digest %s: %s", path, exc)
        return ""


@dataclass(frozen=True)
class RecipeInfo:
    """One discovered recipe, as data — never the class itself.

    Discovery answers with these so a caller (the panel, the agent's recipe
    list, the runner) can list, choose and journal a recipe without importing
    anything or holding a live class.

    Attributes:
        name: The recipe's unique ``name``.
        description: Its one-line ``description``.
        procedures: The procedure class names it serves, or
            ``(ANY_PROCEDURE,)``.
        source_path: Absolute path of the file the class was defined in.
        origin: ``ORIGIN_PACKAGE`` or ``ORIGIN_EXPERIMENT``.
        digest: SHA-256 of that source file, stamped into every report the
            recipe produces.
    """

    name: str
    description: str = ""
    procedures: tuple[str, ...] = ()
    source_path: str = ""
    origin: str = ORIGIN_PACKAGE
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form.

        Returns:
            A dict of every field, with ``procedures`` as a list.
        """
        return {
            "name": self.name,
            "description": self.description,
            "procedures": list(self.procedures),
            "source_path": self.source_path,
            "origin": self.origin,
            "digest": self.digest,
        }


def _info_for(cls: type[AnalysisRecipe], origin: str) -> RecipeInfo:
    """Build the ``RecipeInfo`` describing one recipe class.

    Args:
        cls: The recipe class.
        origin: ``ORIGIN_PACKAGE`` or ``ORIGIN_EXPERIMENT``.

    Returns:
        The info, with the digest taken from the class's own source file.
    """
    source = Path(getattr(sys.modules.get(cls.__module__, None), "__file__", "") or "")
    return RecipeInfo(
        name=str(cls.name),
        description=str(cls.description),
        procedures=tuple(str(p) for p in cls.procedures),
        source_path=str(source),
        origin=origin,
        digest=_digest_of(source) if source else "",
    )


def _recipes_in(module: Any) -> list[type[AnalysisRecipe]]:
    """Return every named recipe class DEFINED in one module.

    Args:
        module: The imported module to scan.

    Returns:
        Its ``AnalysisRecipe`` subclasses that declare a non-empty ``name``,
        in definition order. A class merely imported into the module is
        skipped, so one recipe is never discovered twice.
    """
    found: list[type[AnalysisRecipe]] = []
    for obj in vars(module).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, AnalysisRecipe)
            and obj is not AnalysisRecipe
            and getattr(obj, "name", "")
            and obj.__module__ == module.__name__
        ):
            found.append(obj)
    return found


def _package_recipe_modules() -> list[str]:
    """Return the dotted names of every module in ``i2as.analysis.recipes``.

    Returns:
        The module names, sorted, private ones (``_``-prefixed) excluded.
    """
    import i2as.analysis.recipes as pkg

    return sorted(
        f"{pkg.__name__}.{info.name}"
        for info in pkgutil.iter_modules(pkg.__path__)
        if not info.name.startswith("_")
    )


def _load_script(path: Path) -> Any:
    """Import one experiment recipe script by file path.

    The script is imported under a unique module name derived from its path,
    so two experiments' ``overview.py`` never shadow each other and neither
    shadows a real package module.

    Args:
        path: The ``.py`` file to import.

    Returns:
        The imported module.

    Raises:
        AnalysisError: If the file cannot be turned into a module spec.
        Exception: Whatever the script itself raises while executing.
    """
    unique = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"{_EXPERIMENT_MODULE_PREFIX}_{unique}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AnalysisError(f"{path} is not an importable Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def discover_recipes(extra_dirs: Sequence[str | Path] = ()) -> tuple[RecipeInfo, ...]:
    """Return every recipe available, package ones first.

    Package recipes come from ``i2as/analysis/recipes/``, ordered by
    declared ``priority`` (highest first, then by name) because that order IS
    the selection order ``recipe_for()`` walks; experiment
    recipes from each directory in ``extra_dirs``, in the order given, taking
    every ``*.py`` whose name does not start with ``_``. An experiment recipe
    REPLACES a package recipe of the same ``name`` (in the package one's
    position, so the order stays stable). A module or script that fails to
    import is logged at WARNING and skipped — this function never raises.

    Args:
        extra_dirs: Directories holding experiment recipe scripts. A path
            that does not exist is skipped silently; it is normal for an
            experiment never to have written a recipe.

    Returns:
        One ``RecipeInfo`` per discovered recipe, deduplicated by ``name``.
    """
    found: dict[str, RecipeInfo] = {}

    package_classes: list[type[AnalysisRecipe]] = []
    for module_name in _package_recipe_modules():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            logger.exception("analysis discovery: failed to import %s", module_name)
            continue
        package_classes.extend(_recipes_in(module))
    for cls in sorted(package_classes, key=lambda c: (-int(c.priority), c.name)):
        found[cls.name] = _info_for(cls, ORIGIN_PACKAGE)

    for directory in extra_dirs:
        folder = Path(directory)
        if not folder.is_dir():
            logger.debug("analysis discovery: no recipe folder at %s", folder)
            continue
        for path in sorted(folder.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                module = _load_script(path)
            except Exception:
                logger.warning(
                    "analysis discovery: skipping %s, it failed to import", path, exc_info=True
                )
                continue
            classes = _recipes_in(module)
            if not classes:
                logger.warning(
                    "analysis discovery: %s defines no named AnalysisRecipe subclass", path
                )
            for cls in classes:
                found[cls.name] = _info_for(cls, ORIGIN_EXPERIMENT)

    return tuple(found.values())


def procedure_key(name: str) -> str:
    """Normalise a procedure name for matching.

    The run manifest names a procedure by its display ``name`` (``"Field
    Sweep"``) while a recipe author naturally writes the class name
    (``"FieldSweep"``); both must match. Letters and digits are kept and
    lower-cased, everything else is dropped, so ``"Field Sweep"``,
    ``"FieldSweep"`` and ``"field_sweep"`` are the same key.

    Args:
        name: A procedure display name or class name.

    Returns:
        The normalised key; ``""`` for an empty or all-punctuation name.
    """
    return "".join(c for c in name.lower() if c.isalnum())


def recipe_for(
    procedure: str, recipes: Sequence[RecipeInfo], preferred: str = ""
) -> RecipeInfo | None:
    """Return the recipe that should run for one procedure.

    The selection order: the ``preferred`` name when a recipe carries it, else
    the first recipe naming this procedure in its ``procedures`` (compared
    through ``procedure_key()``, so the manifest's display name matches a
    recipe written against the class name), else the first recipe declaring
    ``ANY_PROCEDURE``.

    Args:
        procedure: The procedure's name as the run manifest's ``procedure``
            field carries it (the display name), or its class name.
        recipes: The candidates, as returned by ``discover_recipes()``.
        preferred: A recipe ``name`` to prefer — the eLab settings' per-
            procedure choice, or what the caller explicitly asked for.

    Returns:
        The chosen recipe, or ``None`` when nothing matches.
    """
    if preferred:
        for info in recipes:
            if info.name == preferred:
                return info
    key = procedure_key(procedure)
    if key:
        for info in recipes:
            if any(procedure_key(p) == key for p in info.procedures):
                return info
    for info in recipes:
        if ANY_PROCEDURE in info.procedures:
            return info
    return None


def load_recipe(info: RecipeInfo) -> AnalysisRecipe:
    """Import a discovered recipe and instantiate it.

    Args:
        info: The recipe to load, as ``discover_recipes()`` described it.

    Returns:
        A fresh instance, ready for exactly one ``analyse()`` call.

    Raises:
        AnalysisError: If the source file is gone, cannot be imported, or no
            longer defines a recipe of that ``name``.
    """
    path = Path(info.source_path)
    if not path.is_file():
        raise AnalysisError(f"recipe {info.name!r} has no source file at {path}")
    try:
        if info.origin == ORIGIN_PACKAGE:
            module = importlib.import_module(f"i2as.analysis.recipes.{path.stem}")
        else:
            module = _load_script(path)
    except AnalysisError:
        raise
    except Exception as exc:
        raise AnalysisError(f"recipe {info.name!r} at {path} failed to import: {exc}") from exc
    for cls in _recipes_in(module):
        if cls.name == info.name:
            return cls()
    raise AnalysisError(f"{path} no longer defines a recipe named {info.name!r}")


def _header_comment(header: str) -> str:
    """Render a scaffold header as comment lines.

    Args:
        header: Free text, possibly several lines — the gateway stamps the
            actor and the time here.

    Returns:
        The text as ``#``-prefixed lines followed by a blank line, or ``""``
        when there was no header.
    """
    if not header.strip():
        return ""
    lines = [line.rstrip() for line in header.strip().splitlines()]
    return "\n".join(line if line.startswith("#") else f"# {line}".rstrip() for line in lines) + "\n\n"


def _class_name_for(name: str) -> str:
    """Return the class name a scaffolded recipe gets.

    Args:
        name: The recipe's snake_case ``name``.

    Returns:
        Its CamelCase form with a ``Recipe`` suffix (``"field_sweep"`` ->
        ``"FieldSweepRecipe"``).
    """
    return "".join(part.capitalize() for part in name.split("_") if part) + "Recipe"


def scaffold_recipe(
    name: str, directory: str | Path, procedure: str = "", header: str = ""
) -> Path:
    """Write a new, commented, runnable recipe for somebody to edit.

    The same function serves the panel's "New recipe…" button and the agent's
    ``write_analysis_recipe`` tool, which is why it refuses rather than
    overwrites: a scaffold must never destroy an analysis somebody wrote.

    Args:
        name: The recipe's ``name`` — a Python identifier, not a keyword, not
            starting with an underscore. It is also the file name.
        directory: The folder to write into, created if needed (normally an
            experiment's ``analysis/recipes/``).
        procedure: The procedure class name the recipe should serve; empty
            means every procedure (``ANY_PROCEDURE``).
        header: Free text prepended as comment lines — the gateway stamps the
            actor and the time so the file says who wrote it.

    Returns:
        The path written.

    Raises:
        AnalysisError: If ``name`` is not a usable identifier, if a file of
            that name already exists, or if the file could not be written.
    """
    if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_"):
        raise AnalysisError(
            f"recipe name {name!r} must be a Python identifier that is not a "
            f"keyword and does not start with an underscore — it becomes both "
            f"the recipe's id and its file name"
        )
    folder = Path(directory)
    path = folder / f"{name}.py"
    if path.exists():
        raise AnalysisError(
            f"{path} already exists; edit it, or scaffold under a different name"
        )
    procedures = f'"{procedure}",' if procedure else f'"{ANY_PROCEDURE}",'
    body = Template(RECIPE_TEMPLATE).substitute(
        name=name,
        class_name=_class_name_for(name),
        procedures=procedures,
        description=f"Analysis of every {procedure} run" if procedure else "One run, analysed",
    )
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(_header_comment(header) + body, encoding="utf-8")
    except OSError as exc:
        raise AnalysisError(f"could not write recipe to {path}: {exc}") from exc
    logger.info("analysis: scaffolded recipe %s", path)
    return path


__all__ = [
    "ORIGIN_EXPERIMENT",
    "ORIGIN_PACKAGE",
    "RECIPE_TEMPLATE",
    "RecipeInfo",
    "discover_recipes",
    "load_recipe",
    "recipe_for",
    "scaffold_recipe",
]
