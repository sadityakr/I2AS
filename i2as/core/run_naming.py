"""Run placement — where one run's data file goes, and what it and the run are called.

Every run in an experiment is numbered, and its number is its identity: the
run id is ``run-NNNN``, the data file is ``run-NNNN_<Procedure>[_<label>].h5``
in the experiment's ``data/`` folder — or in a **run subfolder** of it the
operator chose (``data/cooldown2/``) — and the analysis stage writes that
run's report, figures and scripts under ``analysis/run-NNNN/``. So a person, a
script or an analysis agent finds any run of any experiment of a session by
walking one tree, and the files sort in the order they were measured.

Numbers are unique per experiment, across every subfolder, and never reused.
The number is the largest of: the **floor** the session layer pushes (one more
than the highest run its experiment record has ever recorded), one more than
the last number this process issued, and one more than the highest
``run-NNNN`` *file* in the target folder and in ``data/`` itself (flat scans
only — never a recursive walk on the instrument thread). A deleted or moved
file therefore never frees its number. The optional label is the operator's
own ("file prefix" in the procedure panel): it is for people, and it never
replaces the number. A probe run carries ``probe`` in its name as well as in
its file's metadata, so nobody mistakes it for science data from the file list
alone.

The run-subfolder rule lives here, in core, so the engine can enforce it
without importing the session layer (contract C12): see
:func:`normalize_run_subfolder`.

Pure: nothing here writes anything; the engine applies a placement when a run
starts (``Orchestrator._start_run``) and the data manager creates the file.
"""

from __future__ import annotations

import os
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


#: The deepest a run subfolder may nest below ``data/``.
MAX_SUBFOLDER_DEPTH = 3

#: The longest a run's absolute file path may be on Windows (its
#: 260-character limit, with room for the HDF5 library's own temporary
#: names). Not applied on other systems.
MAX_RUN_PATH_CHARS = 240

#: The file in an experiment's ``data/`` that remembers the highest run
#: number ever issued there, so a number is never reissued in a sibling
#: subfolder even when its run's record was never written.
LAST_RUN_MARKER = ".last_run_number"

#: Whether the path-length limit applies (it is a Windows limit).
_ON_WINDOWS = os.name == "nt"

#: One part of a run subfolder: starts with a letter or digit, then letters,
#: digits, ``.``, ``_``, ``-`` or spaces; at most 64 characters.
_SUBFOLDER_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,63}$")

#: Windows device names, refused as a folder name whatever their case or
#: extension (``nul.txt`` is the NUL device too).
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def normalize_run_subfolder(text: str) -> str:
    """Validate and normalise a run subfolder, relative to an experiment's ``data/``.

    The rule is portable to every filesystem the station runs on (Windows
    first): a relative path of at most ``MAX_SUBFOLDER_DEPTH`` parts, each a
    plain name — letters, digits, ``.``, ``_``, ``-`` and spaces, starting
    with a letter or digit, no trailing ``.`` or space, not a Windows device
    name. Separators may be ``/`` or ``\\``; empty parts are dropped.

    Args:
        text: The operator's subfolder, e.g. ``"cooldown2"`` or
            ``"cooldown2/field sweeps"``. ``""`` means ``data/`` itself.

    Returns:
        The normalised subfolder with ``/`` separators, or ``""``.

    Raises:
        ValueError: If the subfolder is absolute, climbs (``..``), is too deep,
            or a part breaks the name rule.
    """
    raw = str(text).strip().replace("\\", "/")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise ValueError(f"a run subfolder must be relative to data/, got {text!r}")
    parts = [part for part in raw.split("/") if part]
    if len(parts) > MAX_SUBFOLDER_DEPTH:
        raise ValueError(
            f"a run subfolder may be at most {MAX_SUBFOLDER_DEPTH} levels deep, got {text!r}"
        )
    for part in parts:
        if not _SUBFOLDER_PART.match(part) or part.endswith((".", " ")):
            raise ValueError(
                f"run subfolder part {part!r} must start with a letter or digit, use only "
                "letters, digits, '.', '_', '-' or spaces, not end in '.' or a space, "
                "and be at most 64 characters"
            )
        if part.split(".")[0].upper() in _RESERVED_NAMES:
            raise ValueError(f"run subfolder part {part!r} is a reserved device name")
    return "/".join(parts)


def run_target_folder(root: str | Path, subfolder: str = "") -> Path:
    """Return where a run in *subfolder* of *root* is written, proven inside *root*.

    Args:
        root: The experiment's ``data/`` folder.
        subfolder: A run subfolder; normalised here.

    Returns:
        ``root/subfolder``.

    Raises:
        ValueError: If the subfolder breaks the rule, or the target resolves
            outside *root* (a symlink or junction pointing elsewhere).
    """
    root_path = Path(root)
    target = root_path / normalize_run_subfolder(subfolder) if subfolder else root_path
    try:
        inside = target.resolve().is_relative_to(root_path.resolve())
    except (OSError, ValueError):
        inside = False
    if not inside:
        raise ValueError(f"run folder {target} resolves outside {root_path}")
    return target


def next_run_number(folder: str | Path, floor: int = 1, *also: str | Path) -> int:
    """Return the number the next run gets.

    Args:
        folder: The folder the run is written to (need not exist yet).
        floor: The lowest number allowed — the session layer's record of
            every number already issued in this experiment, plus one.
        *also: Further folders whose run files count (``data/`` itself when
            writing into a subfolder).

    Returns:
        The largest of *floor* and one more than the highest ``run-NNNN``
        **file** in *folder* or *also* (flat scans; directories ignored).
    """
    highest = 0
    for each in (folder, *also):
        path = Path(each)
        if not path.is_dir():
            continue
        for entry in path.iterdir():
            match = _RUN_FILE.match(entry.name)
            if match and entry.is_file():
                highest = max(highest, int(match.group(1)))
    return max(int(floor), highest + 1, 1)


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


def place_run(
    folder: str | Path,
    procedure: str,
    label: str = "",
    kind: str = "run",
    *,
    subfolder: str = "",
    floor: int = 1,
) -> RunPlacement:
    """Decide where the next run writes, and what it is called.

    Args:
        folder: The experiment's ``data/`` folder (the numbering root).
        procedure: The procedure's class name.
        label: The operator's optional label.
        kind: The run's kind (``run`` or ``probe``).
        subfolder: The run subfolder inside *folder*; ``""`` for *folder*
            itself.
        floor: The lowest run number allowed (see :func:`next_run_number`).

    Returns:
        The placement.

    Raises:
        ValueError: If the subfolder breaks the rule or resolves outside
            *folder*, or the run file's path would exceed
            ``MAX_RUN_PATH_CHARS``.
    """
    target = run_target_folder(folder, subfolder)
    number = next_run_number(target, max(int(floor), last_issued(folder) + 1), folder)
    file_name = run_file_name(number, procedure, label, kind)
    full = os.path.abspath(target / file_name)
    if _ON_WINDOWS and len(full) > MAX_RUN_PATH_CHARS:
        raise ValueError(
            f"run file path is {len(full)} characters, over {MAX_RUN_PATH_CHARS}: {full}"
        )
    return RunPlacement(
        run_id=run_id_for(number),
        data_directory=str(target),
        file_name=file_name,
    )


def last_issued(folder: str | Path) -> int:
    """Return the highest run number recorded in *folder*'s marker (0 when none)."""
    try:
        return int((Path(folder) / LAST_RUN_MARKER).read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return 0


def record_issued(folder: str | Path, number: int) -> None:
    """Remember that run *number* was issued in *folder* (best effort, never raises).

    Args:
        folder: The experiment's ``data/`` folder.
        number: The run number just issued.
    """
    if number <= last_issued(folder):
        return
    path = Path(folder) / LAST_RUN_MARKER
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(str(number), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        pass


def run_number_of(run_id: str) -> int | None:
    """Return the number of a ``run-NNNN`` id, or ``None`` for anything else."""
    match = re.match(rf"^{re.escape(RUN_ID_PREFIX)}(\d{{1,9}})$", str(run_id))
    return int(match.group(1)) if match else None


__all__ = [
    "MAX_RUN_PATH_CHARS",
    "MAX_SUBFOLDER_DEPTH",
    "RUN_FILE_SUFFIX",
    "RUN_ID_PREFIX",
    "RUN_NUMBER_DIGITS",
    "RunPlacement",
    "LAST_RUN_MARKER",
    "last_issued",
    "next_run_number",
    "normalize_run_subfolder",
    "record_issued",
    "place_run",
    "run_file_name",
    "run_id_for",
    "run_number_of",
    "run_target_folder",
]
