"""Run placement — where one run's data file goes, and what it and the run are called.

Every run in an experiment is numbered, and its number is its identity: the
run id is ``run-NNNN``, the data file is ``run-NNNN_<Procedure>[_<label>].h5``
in the experiment's ``data/`` folder, and the analysis stage writes that run's
report, figures and scripts under ``analysis/run-NNNN/``. So a person, a
script or an analysis agent finds any run of any experiment of a session by
walking one fixed tree, and the files sort in the order they were measured.

The number is one more than the highest ``run-NNNN`` file already in the
folder, so it is never reused — not after a file is deleted, not after the
application restarts. The optional label is the operator's own ("file
prefix" in the procedure panel): it is for people, and it never replaces
the number. A probe run carries ``probe`` in its name as well as in its file's
metadata, so nobody mistakes it for science data from the file list alone.

Pure: nothing here writes anything; the engine applies a placement when a run
starts (``Orchestrator._start_run``) and the data manager creates the file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: The prefix every run id and run file name starts with.
RUN_ID_PREFIX = "run-"

#: Digits of a run's number (``run-0001``). A longer series still works; its
#: numbers simply grow a digit.
RUN_NUMBER_DIGITS = 4

#: The suffix of a run's data file.
RUN_FILE_SUFFIX = ".h5"

#: A run file's name: its number, then ``_…`` or the suffix.
_RUN_FILE = re.compile(r"^run-(\d{1,9})(?:_.*)?\.h5$")


@dataclass(frozen=True)
class RunPlacement:
    """Where one run writes, and what it is called.

    Attributes:
        run_id: ``run-NNNN``.
        data_directory: The experiment's data folder.
        file_name: ``run-NNNN_<Procedure>[_<label>].h5``.
    """

    run_id: str
    data_directory: str
    file_name: str


def next_run_number(folder: str | Path) -> int:
    """Return the number the next run in *folder* gets.

    Args:
        folder: The experiment's data folder (need not exist yet).

    Returns:
        One more than the highest ``run-NNNN`` file there, from 1.
    """
    path = Path(folder)
    if not path.is_dir():
        return 1
    numbers = [
        int(match.group(1))
        for entry in path.iterdir()
        if (match := _RUN_FILE.match(entry.name))
    ]
    return max(numbers, default=0) + 1


def run_id_for(number: int) -> str:
    """Return the run id of run *number*: ``run-NNNN``."""
    return f"{RUN_ID_PREFIX}{number:0{RUN_NUMBER_DIGITS}d}"


def _label(text: str) -> str:
    """Return *text* as a file-name-safe label, case kept: ``[A-Za-z0-9-]`` joined by ``_``."""
    return re.sub(r"[^A-Za-z0-9-]+", "_", text).strip("_")


def run_file_name(number: int, procedure: str, label: str = "", kind: str = "run") -> str:
    """Return the data file name of run *number*.

    Args:
        number: The run's number.
        procedure: The procedure's class name (``FieldSweep``).
        label: The operator's optional label; ``""`` for none.
        kind: The run's kind; a ``probe`` run carries ``probe`` in its name.

    Returns:
        ``run-NNNN_<Procedure>[_<label>][_probe].h5``.
    """
    parts = [run_id_for(number), _label(procedure) or "run", _label(label)]
    # A probe says so once, whether or not the label already did.
    if kind == "probe" and "probe" not in _label(label).lower().split("_"):
        parts.append("probe")
    return "_".join(part for part in parts if part) + RUN_FILE_SUFFIX


def place_run(folder: str | Path, procedure: str, label: str = "", kind: str = "run") -> RunPlacement:
    """Decide where the next run in *folder* writes, and what it is called.

    Args:
        folder: The experiment's data folder.
        procedure: The procedure's class name.
        label: The operator's optional label.
        kind: The run's kind (``run`` or ``probe``).

    Returns:
        The placement.
    """
    number = next_run_number(folder)
    return RunPlacement(
        run_id=run_id_for(number),
        data_directory=str(folder),
        file_name=run_file_name(number, procedure, label, kind),
    )


__all__ = [
    "RUN_FILE_SUFFIX",
    "RUN_ID_PREFIX",
    "RUN_NUMBER_DIGITS",
    "RunPlacement",
    "next_run_number",
    "place_run",
    "run_file_name",
    "run_id_for",
]
