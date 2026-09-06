"""The analysis worker — ``python -m i2as.analysis``.

The process a recipe runs in. It is started by the application's analysis
runner (and by a physicist at a shell) with one argument: the path of a spec
file. It reads the spec, runs the recipe, writes ``report.json`` and its
figures into the spec's output directory, prints that path and exits.

**It exits 0 when the analysis failed.** A failed recipe is a complete
answer — a ``failed`` report carrying the traceback — and the caller reads
the report, not the exit code. Exit 2 is reserved for the case where there is
no answer to write at all: a spec file that is missing, unreadable, or not
JSON, and (for the other two commands) an argument the command refuses.

Three commands:

- ``run --spec <file>`` — the worker proper.
- ``new-recipe <name> --dir <folder> [--procedure P] [--header TEXT]`` —
  scaffold a commented, runnable recipe for somebody to edit.
- ``list [--dir <folder> ...]`` — what recipes exist, package and experiment.

Standard output is the answer (a path, or one row per recipe) so a shell can
consume it; every log line goes to standard error, which is where the calling
process collects a worker's diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from i2as.analysis.base import AnalysisError
from i2as.analysis.discovery import discover_recipes, scaffold_recipe
from i2as.analysis.report import AnalysisSpec
from i2as.analysis.runner import run_spec, write_report

logger = logging.getLogger(__name__)

#: Exit code for a request that could not be served at all — an unreadable
#: spec, a refused recipe name. A failed ANALYSIS is not one of these; it
#: exits 0 with a failed report.
EXIT_BAD_REQUEST = 2


def _build_parser() -> argparse.ArgumentParser:
    """Build the worker's argument parser.

    Returns:
        The parser, with the ``run``, ``new-recipe`` and ``list``
        sub-commands.
    """
    parser = argparse.ArgumentParser(
        prog="python -m i2as.analysis",
        description="Run one analysis recipe against one finished run.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log at DEBUG instead of INFO (always to stderr)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run the recipe one spec file asks for")
    run_parser.add_argument(
        "--spec", required=True, help="path of the spec JSON file (an AnalysisSpec)"
    )

    new_parser = commands.add_parser("new-recipe", help="scaffold a new recipe script")
    new_parser.add_argument("name", help="the recipe's name (a Python identifier)")
    new_parser.add_argument(
        "--dir", required=True, help="folder to write into (an experiment's analysis/recipes)"
    )
    new_parser.add_argument(
        "--procedure", default="", help="procedure class name to serve; default: every procedure"
    )
    new_parser.add_argument(
        "--header", default="", help="text prepended as comment lines (who wrote it, when)"
    )

    list_parser = commands.add_parser("list", help="list the recipes available")
    list_parser.add_argument(
        "--dir",
        action="append",
        default=[],
        dest="dirs",
        help="an extra recipe folder; repeatable",
    )
    return parser


def _read_spec(path: str) -> AnalysisSpec:
    """Read a spec file.

    Args:
        path: Path of the JSON file.

    Returns:
        The parsed ``AnalysisSpec``.

    Raises:
        AnalysisError: If the file is missing, unreadable, or not a JSON
            object.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AnalysisError(f"cannot read analysis spec {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AnalysisError(f"analysis spec {path} is not a JSON object")
    return AnalysisSpec.from_dict(payload)


def _cmd_run(args: argparse.Namespace) -> int:
    """Run one spec and write its report.

    Args:
        args: The parsed arguments; ``args.spec`` is the spec file's path.

    Returns:
        ``0`` once a report was written, whatever its status;
        ``EXIT_BAD_REQUEST`` when the spec itself could not be read or the
        report could not be written.
    """
    try:
        spec = _read_spec(args.spec)
    except AnalysisError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_BAD_REQUEST

    report = run_spec(spec)
    output_dir = Path(spec.output_dir) if spec.output_dir else Path(args.spec).parent
    try:
        path = write_report(report, output_dir)
    except OSError as exc:
        print(f"cannot write the report into {output_dir}: {exc}", file=sys.stderr)
        return EXIT_BAD_REQUEST
    print(str(path))
    return 0


def _cmd_new_recipe(args: argparse.Namespace) -> int:
    """Scaffold one recipe script.

    Args:
        args: The parsed arguments — ``name``, ``dir``, ``procedure``,
            ``header``.

    Returns:
        ``0`` and the written path on stdout, or ``EXIT_BAD_REQUEST`` and the
        refusal on stderr.
    """
    try:
        path = scaffold_recipe(args.name, args.dir, procedure=args.procedure, header=args.header)
    except AnalysisError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_BAD_REQUEST
    print(str(path))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """List the recipes available.

    Args:
        args: The parsed arguments; ``args.dirs`` holds the extra folders.

    Returns:
        ``0`` always — an empty package is a valid answer, and an unreadable
        script is a WARNING on stderr, not a failure.
    """
    for info in discover_recipes(args.dirs):
        procedures = ",".join(info.procedures) or "-"
        print(f"{info.name}\t{info.origin}\t{procedures}\t{info.description}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point of the analysis worker.

    Args:
        argv: Command-line arguments, or ``None`` for ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "new-recipe":
        return _cmd_new_recipe(args)
    return _cmd_list(args)


if __name__ == "__main__":
    sys.exit(main())
