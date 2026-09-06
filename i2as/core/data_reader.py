"""Read-only access to a measurement run — the analysis side of the L5 file.

A standalone sibling of ``data_manager.py``, never a method on it: the writer
owns an open file and a run in flight, the reader owns neither and must be
importable by a process that has no Station, no Orchestrator and no Qt. That
independence is machine-checked by its own import-linter contract — this
module may import stdlib, ``numpy``, ``h5py`` and the two dependency-free
``core`` modules (``events``, ``exceptions``) and nothing else from the
package — so an analysis process can import it alone.

**One vocabulary for live and stored runs.** The run being written and the
runs on disk answer the same four questions through the same types: which
columns exist (``list_columns`` -> ``ColumnInfo``), what a slice of one holds
(``read_slice`` -> a numpy array), how a column summarises (``summary_stats``
-> ``Stats``), and what the run is (``read_metadata`` -> the canonical
``RUN_METADATA_KEYS`` dict). ``RunSource`` is that vocabulary as a
``Protocol``: ``RunHandle`` here implements it over an HDF5 file and
``core.run_buffer.RunBuffer`` implements it over the ``Datapoint`` events of
the run in progress, so a consumer — the GUI's live view today, an agent
gateway later — writes one analysis and points it at either. The
module-level functions below take any ``RunSource``, so ``summary_stats(run,
"voltage_V")`` reads a finished file or a live buffer unchanged.

``ColumnInfo`` and ``Stats`` are frozen and JSON-safe (``to_json()`` /
``from_json()``, a NaN rendering as ``null``), so a result can travel as the
``result`` payload of a control-contract ``Verdict`` without a second
declaration.

**The HDF5 layout this reads** is exactly what ``DataManager`` writes:

===========================  ======================================  ==============
``/data`` dataset            Shape                                   Role
===========================  ======================================  ==============
a sweep column               ``(N,)``                                ``sweep_axis``
``timestamp``                ``(N,)`` of strings                     ``sweep_axis``
a measurement scalar         ``(N, n_loop1, n_loop2)``               ``measurement``
a measurement array          ``(N, n_loop1, n_loop2, length)``       ``measurement``
a raw diagnostic block       ``(N, [n_loop1, n_loop2,] rows, cols)`` ``raw_block``
an image block               ``(N, [n_loop1, n_loop2,] rows, cols)`` ``image``
===========================  ======================================  ==============

The leading axis is always the sweep point; a block carries the loop axes
only when a reading loop is configured, and describes itself with ``axes``
and ``block_kind`` attributes — plus ``channel_names`` for a raw block, or
``unit``/``description`` for an image (the image-block standard, whose one
frame per point ``read_image()`` serves). Which name is which role is
declared in ``/metadata``'s ``data_config``; the ``block_kind``/``axes``
attributes and then the shape are the fallback when it is absent. The **reading loop**'s own axes are not datasets
at all — index -> physical value lives in
``procedure_params["loop1_values"]`` / ``["loop2_values"]`` — so this module
surfaces them as the two ``loop_axis`` columns ``loop1`` and ``loop2``,
letting a consumer label a measurement column's inner axes without parsing
metadata itself. ``/metadata``'s attributes are the run's manifest,
JSON-encoded per attribute.

**The written prefix.** ``DataManager`` pre-allocates every dataset to the
full sweep length and fills it with NaN, so a run that stopped early leaves
allocated-but-unwritten points behind (it trims only on a clean ``close()``).
The reader's point counter is the ``timestamp`` column, which the writer
stamps last for each point: the number of non-empty timestamps is the number
of points whose columns are all on disk. ``RunHandle.n_points`` is that
count, ``read_slice()`` never reads past it, and every statistic is taken
over the written prefix — and, within it, over finite values only, so a
failed reading's NaN never poisons an answer.

**Reading a file that is still being written** works, with one documented
limitation. ``open_run()`` tries a plain read first, then a SWMR read, then a
lock-free read, and records which succeeded in ``RunHandle.mode``. Inside the
writing process a plain read succeeds and sees every point the writer has
flushed — ``DataManager`` flushes after every datapoint — including points
saved after the handle was opened. From another process the file's lock
forces the lock-free fall back, which sees the file as of the last flush; a
handle held across further writes may not see them, so a reader outside the
writing process should reopen rather than assume its handle grows. The
limitation is SWMR: ``DataManager`` does not write in SWMR mode, so
``Dataset.refresh()`` is not available (calling it on a non-SWMR handle
corrupts the handle rather than failing cleanly) and this module only ever
refreshes a handle whose file really is in SWMR mode. Mid-run analysis of the
run in flight is the ``RunBuffer``'s job, not this one's.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

import h5py
import numpy as np

from i2as.core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

#: The group every run's columns live in, and the group its manifest lives in.
DATA_GROUP = "data"
METADATA_GROUP = "metadata"

#: The column the writer stamps last for every saved point, and therefore the
#: reader's point counter — see the module docstring's "written prefix".
TIMESTAMP_COLUMN = "timestamp"

#: What a column *is*, independent of how it is stored. A consumer plots a
#: ``measurement`` against a ``sweep_axis``, labels a measurement's inner axes
#: from the ``loop_axis`` columns, leaves a ``raw_block`` to diagnostics, and
#: reads an ``image`` frame by frame through ``read_image()``.
ROLE_SWEEP_AXIS = "sweep_axis"
ROLE_LOOP_AXIS = "loop_axis"
ROLE_MEASUREMENT = "measurement"
ROLE_RAW_BLOCK = "raw_block"
ROLE_IMAGE = "image"
COLUMN_ROLES: tuple[str, ...] = (
    ROLE_SWEEP_AXIS,
    ROLE_LOOP_AXIS,
    ROLE_MEASUREMENT,
    ROLE_RAW_BLOCK,
    ROLE_IMAGE,
)

#: The dataset attribute a block writes to say what kind it is, and the value
#: that marks a frame (``DataManager._allocate_datasets``).
BLOCK_KIND_ATTR = "block_kind"
BLOCK_KIND_IMAGE = "image"

#: The reading loop's two slots, as the metadata key holding a slot's ordered
#: physical values mapped to the ``loop_axis`` column name this module gives
#: it. Slot 1 is axis 0 of every measurement column, slot 2 axis 1.
LOOP_AXIS_PARAMS: dict[str, str] = {
    "loop1_values": "loop1",
    "loop2_values": "loop2",
}

#: The canonical keys every ``RunSource.read_metadata()`` answers, whatever it
#: reads from. A source that cannot know one answers the empty value for its
#: type rather than omitting the key, so a consumer never branches on which
#: source it holds. ``raw`` is the deliberate exception: the source-specific
#: complete payload (a file's decoded ``/metadata`` attributes, a buffer's run
#: manifest), useful but never part of the shared vocabulary.
RUN_METADATA_KEYS: tuple[str, ...] = (
    "source",
    "run_id",
    "run_kind",
    "procedure",
    "params",
    "sample",
    "setup",
    "experiment",
    "start_time",
    "end_time",
    "status",
    "reason",
    "data_file",
    "raw",
)

#: The unit tokens a column name may end in, mapped to the unit they mean.
#: I2AS column names carry their SI unit as the last ``_``-separated
#: token (``field_T``, ``current_A``, ``res_a_dc_ohm``), optionally before one
#: of the role suffixes in ``_ROLE_SUFFIXES``. A closed set, so a column whose
#: name simply ends in a word (``magnet_z_field``, ``unix_time``, ``n_valid``)
#: reports no unit instead of inventing one.
UNIT_SUFFIXES: dict[str, str] = {
    "T": "T",
    "K": "K",
    "A": "A",
    "V": "V",
    "s": "s",
    "Hz": "Hz",
    "W": "W",
    "Pa": "Pa",
    "mbar": "mbar",
    "ohm": "ohm",
    "Ohm": "ohm",
    "deg": "deg",
    "pct": "pct",
}

#: Name parts that describe a column's *role*, not its quantity, and sit
#: after the unit token (``voltage_V_error``, ``current_A_array``).
_ROLE_SUFFIXES: tuple[str, ...] = ("array", "error")

#: ``data_config`` sections mapped to the role their column names carry. The
#: writer's own vocabulary (see ``DataManager.__init__``'s ``data_config``),
#: read here so a file says what each of its columns is rather than leaving
#: the reader to guess from a shape.
_CONFIG_SECTION_ROLES: dict[str, str] = {
    "sweep_columns": ROLE_SWEEP_AXIS,
    "measurement_scalars": ROLE_MEASUREMENT,
    "measurement_arrays": ROLE_MEASUREMENT,
    "measurement_blocks": ROLE_RAW_BLOCK,
    # Listed AFTER measurement_blocks on purpose: an image block appears in
    # both (the shape there, the kind here), and the later section wins.
    "measurement_image_blocks": ROLE_IMAGE,
}


def column_unit(name: str) -> str:
    """Return the SI unit a column name declares, or ``""``.

    The column-name unit convention: the unit is the last ``_``-separated
    token of the name, optionally followed by one role suffix
    (``voltage_V_error`` is volts, so is ``voltage_V``), and only when that
    token is one of ``UNIT_SUFFIXES``.

    Args:
        name: The column name, e.g. ``"field_T"`` or ``"magnet_z_field"``.

    Returns:
        The unit (``"T"``), or ``""`` when the name declares none.
    """
    parts = name.split("_")
    if len(parts) >= 2 and parts[-1] in _ROLE_SUFFIXES:
        parts = parts[:-1]
    if len(parts) < 2:
        return ""
    return UNIT_SUFFIXES.get(parts[-1], "")


def _json_float(value: float) -> float | None:
    """Render a float for JSON, mapping NaN and infinities to ``None``.

    Args:
        value: The float to render.

    Returns:
        The float itself, or ``None`` when it is not finite — strict JSON has
        no NaN, and a missing statistic is exactly "no value".
    """
    return value if math.isfinite(value) else None


def _from_json_float(value: Any) -> float:
    """Rebuild a float from its JSON rendering.

    Args:
        value: A number, or ``None`` for a statistic that has no value.

    Returns:
        The float, with ``None`` mapping back to NaN.
    """
    return float("nan") if value is None else float(value)


@dataclass(frozen=True)
class ColumnInfo:
    """One column of a run, as declared by whichever source holds it.

    Attributes:
        name: The column name, unique within the run (e.g. ``"voltage_V"``).
        unit: The SI unit from the column-name convention (see
            ``column_unit``), or ``""`` when the name declares none.
        dtype: Canonical element type — ``"float64"``, ``"int64"`` or
            ``"str"``. Never a storage detail of the source.
        role: One of ``COLUMN_ROLES`` — what the column is (see the module
            docstring's layout table).
        shape: The column's stored shape, leading axis first. For a
            sweep-point column the leading axis is the number of points the
            source has room for, which for a file still being written (or
            abandoned without a clean close) is larger than ``length``.
        length: Number of sweep points actually written — the written prefix
            (``RunHandle.n_points``) for a file, the points received so far
            for a live buffer. For a ``loop_axis`` column, whose axis is the
            loop rather than the sweep, it is the number of loop values.
        path: Where the column lives — ``"/data/voltage_V"`` for a dataset,
            ``"/metadata/procedure_params/loop1_values"`` for a loop axis. A
            live buffer names the path the column will occupy once written, so
            both sources name a column the same way.
    """

    name: str
    unit: str
    dtype: str
    role: str
    shape: tuple[int, ...]
    length: int
    path: str

    def to_json(self) -> dict[str, Any]:
        """Render this column declaration as a JSON-safe dict.

        Returns:
            A dict of the declared fields, made only of JSON scalars and — for
            ``shape`` — a list of ints.
        """
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        payload["shape"] = [int(n) for n in self.shape]
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self:
        """Rebuild a column declaration from its ``to_json()`` dict.

        Args:
            payload: A mapping as produced by ``to_json()``. Unknown keys are
                ignored, so a newer producer never breaks an older consumer.

        Returns:
            The ``ColumnInfo``.

        Raises:
            TypeError: If a declared field is missing.
        """
        declared = {f.name for f in fields(cls)}
        values = {k: v for k, v in payload.items() if k in declared}
        if "shape" in values:
            values["shape"] = tuple(int(n) for n in values["shape"])
        return cls(**values)


@dataclass(frozen=True)
class Stats:
    """NaN-aware summary of one numeric column.

    Every statistic is taken over the column's finite values only, flattened
    across every axis: a run's unwritten points and a reading loop's failed
    readings are NaN, and they are excluded rather than poisoning the answer.
    A column with no finite value at all reports ``count == 0`` and NaN for
    every statistic.

    Attributes:
        column: The column these statistics describe.
        count: How many finite values contributed (never the column length).
        min: Smallest finite value.
        max: Largest finite value.
        mean: Arithmetic mean of the finite values.
        std: Population standard deviation (``ddof=0``) of the finite values.
        first: First finite value in storage order — the earliest sweep
            point's first reading.
        last: Last finite value in storage order.
    """

    column: str
    count: int
    min: float
    max: float
    mean: float
    std: float
    first: float
    last: float

    def to_json(self) -> dict[str, Any]:
        """Render these statistics as a JSON-safe dict.

        Returns:
            A dict of the declared fields, with every non-finite float
            rendered as ``None`` so the payload is strict JSON.
        """
        payload: dict[str, Any] = {"column": self.column, "count": self.count}
        for name in ("min", "max", "mean", "std", "first", "last"):
            payload[name] = _json_float(getattr(self, name))
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self:
        """Rebuild statistics from their ``to_json()`` dict.

        Args:
            payload: A mapping as produced by ``to_json()``; ``None`` for a
                float field means "no value" and rebuilds as NaN. Unknown keys
                are ignored.

        Returns:
            The ``Stats``.

        Raises:
            TypeError: If a declared field is missing.
        """
        return cls(
            column=str(payload["column"]),
            count=int(payload["count"]),
            **{
                name: _from_json_float(payload[name])
                for name in ("min", "max", "mean", "std", "first", "last")
            },
        )


@runtime_checkable
class RunSource(Protocol):
    """The read vocabulary a run answers, wherever the run lives.

    Implemented by ``RunHandle`` (a run on disk) and by
    ``core.run_buffer.RunBuffer`` (the run in progress). A consumer depends on
    this protocol, never on which one it holds — that is the whole point of
    the one-vocabulary standard this module's docstring states.
    """

    @property
    def n_points(self) -> int:
        """The number of sweep points the source actually holds."""
        ...

    def list_columns(self) -> tuple[ColumnInfo, ...]:
        """Return every column of the run, in name order."""
        ...

    def read_slice(
        self,
        column: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> np.ndarray:
        """Return a slice of one column along its leading axis."""
        ...

    def summary_stats(self, column: str) -> Stats:
        """Return the NaN-aware summary of one numeric column."""
        ...

    def read_metadata(self) -> dict[str, Any]:
        """Return the run's metadata under ``RUN_METADATA_KEYS``."""
        ...

    def read_image(
        self, column: str, index: int, loop1: int = 0, loop2: int = 0
    ) -> np.ndarray:
        """Return one ``(height, width)`` frame of an ``image`` column."""
        ...


def _dtype_name(dtype: np.dtype) -> str:
    """Return the canonical element-type name for an HDF5 dataset dtype.

    Args:
        dtype: The dataset's dtype.

    Returns:
        ``"str"`` for any HDF5 string dtype (fixed or variable length),
        otherwise the numpy dtype name (``"float64"``, ``"int64"``).
    """
    if h5py.check_string_dtype(dtype) is not None:
        return "str"
    return str(np.dtype(dtype))


def sequence_dtype(values: Sequence[Any]) -> str:
    """Return the canonical element type of a plain Python value sequence.

    The ``loop_axis`` columns are read out of JSON metadata, not out of a
    typed dataset, and a reading-loop slot is either numeric (source
    currents) or textual (switch routes) — never mixed.

    Args:
        values: The slot's ordered values.

    Returns:
        ``"float64"`` when every value is a real number, ``"str"`` otherwise
        (including for an empty sequence, which declares nothing numeric).
    """
    if values and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        return "float64"
    return "str"


def loop_axis_columns(params: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Return the reading loop's axis columns declared by a run's parameters.

    The single reading-loop mapping both run sources use, so a file and a
    live buffer name and order the loop axes identically.

    Args:
        params: The run's procedure parameters — ``read_metadata()["params"]``,
            which is ``/metadata.procedure_params`` for a file and the run
            manifest's ``params`` for a buffer.

    Returns:
        ``{"loop1": [...], "loop2": [...]}`` for whichever slots the run
        declared with at least one value, in slot order; an empty dict when
        the run had no reading loop.
    """
    axes: dict[str, list[Any]] = {}
    for param, name in LOOP_AXIS_PARAMS.items():
        values = params.get(param)
        if isinstance(values, (list, tuple)) and values:
            axes[name] = list(values)
    return axes


def loop_axis_column_infos(params: Mapping[str, Any]) -> tuple[ColumnInfo, ...]:
    """Return one ``ColumnInfo`` per reading-loop axis a run declared.

    Args:
        params: The run's procedure parameters (see ``loop_axis_columns``).

    Returns:
        The ``loop_axis`` columns, in slot order. Their ``path`` points into
        the metadata the values came from, since a loop axis is never a
        dataset.
    """
    axes = loop_axis_columns(params)
    return tuple(
        ColumnInfo(
            name=name,
            unit=column_unit(name),
            dtype=sequence_dtype(axes[name]),
            role=ROLE_LOOP_AXIS,
            shape=(len(axes[name]),),
            length=len(axes[name]),
            path=f"/{METADATA_GROUP}/procedure_params/{param}",
        )
        for param, name in LOOP_AXIS_PARAMS.items()
        if name in axes
    )


def roles_from_data_config(config: Mapping[str, Any]) -> dict[str, str]:
    """Map column name to role from the writer's ``data_config`` declaration.

    Args:
        config: The ``data_config`` dict ``DataManager`` was built with —
            ``/metadata.data_config`` in a file, the run manifest's
            ``data_config`` for a live buffer.

    Returns:
        ``{column_name: role}`` for every column the config declares; columns
        it does not mention (``timestamp``) are simply absent.
    """
    roles: dict[str, str] = {}
    for section, role in _CONFIG_SECTION_ROLES.items():
        names = config.get(section)
        if isinstance(names, Mapping):
            for name in names:
                roles[str(name)] = role
    return roles


def select_frame(frames: np.ndarray, loop1: int = 0, loop2: int = 0) -> np.ndarray:
    """Pick one frame out of a single sweep point's image-block value.

    The image-block standard stores a point's frames as ``(rows, cols)``
    with no reading loop, or ``(n_loop1, n_loop2, rows, cols)`` with one —
    the same loop-axis rule a raw block follows — so both ``RunSource``
    implementations share this one selection.

    Args:
        frames: The point's value, 2-D or 4-D.
        loop1: Reading-loop slot-1 index; must be ``0`` for a 2-D value.
        loop2: Reading-loop slot-2 index; must be ``0`` for a 2-D value.

    Returns:
        The ``(rows, cols)`` frame as a float64 array.

    Raises:
        IndexError: If a loop index is out of range, or non-zero for a value
            that carries no loop axis.
        ValueError: If the value is neither 2-D nor 4-D.
    """
    array = np.asarray(frames, dtype=np.float64)
    if array.ndim == 2:
        if loop1 or loop2:
            raise IndexError(
                f"image block carries no reading-loop axis, but loop1={loop1}, "
                f"loop2={loop2} were asked for"
            )
        return array
    if array.ndim == 4:
        return array[loop1, loop2]
    raise ValueError(
        f"an image block value must be (rows, cols) or (n_loop1, n_loop2, rows, cols), "
        f"got shape {array.shape}"
    )


def _decode_attr(value: Any) -> Any:
    """Decode one ``/metadata`` attribute into a plain Python value.

    Args:
        value: The raw attribute as h5py returns it — ``bytes``, ``str``, or a
            numpy scalar/array. Strings that hold a JSON document (the writer
            encodes every structured attribute as JSON) are parsed.

    Returns:
        The parsed object for a JSON string, the decoded text for any other
        string, or the value converted to a plain Python type.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_decode_attr(item) for item in value.tolist()]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "{[" or stripped in ("null", "true", "false"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
        return value
    return value


def resolve_slice(
    length: int, start: int | None, stop: int | None, step: int | None
) -> slice:
    """Resolve a caller's slice against a column's true length.

    Shared by every ``RunSource`` so "the whole column" means the same thing
    everywhere: the written prefix, never the allocated tail. Negative bounds
    count back from the last written point, and a ``stop`` beyond the end is
    clamped rather than reaching into unwritten NaN.

    Args:
        length: The column's true length.
        start: First index, or ``None`` for the beginning.
        stop: Stop before this index, or ``None`` for the end.
        step: Stride, or ``None`` for every element. Must be positive: HDF5
            has no reverse read, and a run source that offered one only for
            the in-memory buffer would not be one vocabulary.

    Returns:
        A concrete ``slice`` with all three bounds resolved.

    Raises:
        ValueError: If ``step`` is zero or negative.
    """
    if step is not None and step < 0:
        raise ValueError(f"step must be positive, got {step}")
    first, last, stride = slice(start, stop, step).indices(length)
    return slice(first, last, stride)


class RunHandle:
    """An open, read-only view of one run's HDF5 file.

    Built by ``open_run()``, which is also the context manager form::

        with open_run(path) as run:
            stats = run.summary_stats("voltage_V")

    The handle never writes: it opens the file read-only, and every method
    here is a read. Outside a ``with`` block the caller owns ``close()``.

    Attributes:
        path: The file this handle reads.
        mode: How the file was opened — ``"swmr"``, ``"read"`` or
            ``"unlocked"`` (see the module docstring's note on reading a file
            that is still being written).
    """

    def __init__(self, path: Path, file: h5py.File, mode: str) -> None:
        """Wrap an already-open h5py file.

        Callers use ``open_run()`` rather than this constructor, which exists
        so the opening strategy lives in one function.

        Args:
            path: The file's path.
            file: The open ``h5py.File``, in a read mode.
            mode: Which open attempt succeeded (``"swmr"``, ``"read"``,
                ``"unlocked"``).
        """
        self.path = path
        self.mode = mode
        self._file = file
        self._complete_n_points: int | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        """Whether the underlying file is still open.

        Returns:
            ``True`` until ``close()`` runs.
        """
        return bool(self._file)

    def close(self) -> None:
        """Close the file. Idempotent."""
        if self._file:
            self._file.close()
            logger.debug("data_reader: closed %s", self.path)

    def __enter__(self) -> RunHandle:
        """Enter the context manager.

        Returns:
            This handle.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the file on leaving the context manager.

        Args:
            *exc_info: The exception triple, ignored — the file is closed
                whether or not the body raised.
        """
        self.close()

    # ── The run-source vocabulary ─────────────────────────────────────

    @property
    def n_points(self) -> int:
        """The written prefix: how many sweep points are fully on disk.

        Counted from the ``timestamp`` column, which the writer stamps after
        that point's every other column (see the module docstring), so this
        never counts a half-written point. Re-counted on each read while the
        run is still short of its allocation, and remembered once the file is
        full — a completed or trimmed file cannot grow.

        Returns:
            The number of written sweep points, or the allocated length when
            the file carries no ``timestamp`` column to count.

        Raises:
            ValueError: If the handle is closed.
        """
        if self._complete_n_points is not None:
            return self._complete_n_points
        group = self._data_group()
        if TIMESTAMP_COLUMN not in group:
            allocated = self._allocated_points()
            self._complete_n_points = allocated
            return allocated
        dataset = group[TIMESTAMP_COLUMN]
        self._refresh(dataset)
        allocated = int(dataset.shape[0]) if dataset.shape else 0
        written = sum(1 for stamp in dataset[...] if stamp)
        if written >= allocated:
            self._complete_n_points = allocated
        return written

    def list_columns(self) -> tuple[ColumnInfo, ...]:
        """Return every column of the run, in name order.

        The datasets under ``/data`` plus the reading loop's ``loop_axis``
        columns, which live in metadata rather than in a dataset (see the
        module docstring).

        Returns:
            One ``ColumnInfo`` per column, ordered by name so two sources of
            the same run list their columns identically.

        Raises:
            ValueError: If the handle is closed.
        """
        group = self._data_group()
        metadata = self.read_metadata()
        roles = roles_from_data_config(metadata["raw"].get("data_config") or {})
        n_points = self.n_points
        infos: list[ColumnInfo] = []
        for name in group:
            dataset = group[name]
            shape = tuple(int(n) for n in dataset.shape)
            infos.append(
                ColumnInfo(
                    name=name,
                    unit=column_unit(name),
                    dtype=_dtype_name(dataset.dtype),
                    role=self._role_for(name, dataset, roles),
                    shape=shape,
                    length=min(n_points, shape[0]) if shape else 0,
                    path=f"/{DATA_GROUP}/{name}",
                )
            )
        infos.extend(
            info
            for info in loop_axis_column_infos(metadata["params"])
            if info.name not in group
        )
        return tuple(sorted(infos, key=lambda info: info.name))

    def read_slice(
        self,
        column: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> np.ndarray:
        """Return a slice of one column along its leading axis.

        Only the leading axis is sliced; every inner axis — a reading loop's,
        a raw array's samples, a block's rows and channels — comes back
        whole, so the returned array's shape is the column's with its first
        dimension narrowed. The bounds resolve against the written prefix, so
        the default whole-column read never returns allocated-but-unwritten
        NaN points.

        Args:
            column: The column name, as ``list_columns()`` reports it.
            start: First sweep point, or ``None`` for the beginning.
            stop: Stop before this sweep point, or ``None`` for the end.
            step: Stride, or ``None`` for every point; must be positive.

        Returns:
            A numpy array — ``float64`` for a numeric column, a string array
            for a string column (``timestamp``, a textual loop axis).

        Raises:
            KeyError: If the run has no such column.
            ValueError: If the handle is closed, or ``step`` is not positive.
        """
        group = self._data_group()
        if column not in group:
            loop_values = self._loop_axis_values().get(column)
            if loop_values is not None:
                selection = resolve_slice(len(loop_values), start, stop, step)
                return np.asarray(loop_values[selection])
        dataset = self._dataset(column)
        self._refresh(dataset)
        length = min(self.n_points, int(dataset.shape[0]) if dataset.shape else 0)
        selection = resolve_slice(length, start, stop, step)
        if _dtype_name(dataset.dtype) == "str":
            return np.asarray(dataset.asstr()[selection])
        return np.asarray(dataset[selection])

    def read_image(
        self, column: str, index: int, loop1: int = 0, loop2: int = 0
    ) -> np.ndarray:
        """Return one frame of an image block (the image-block standard).

        Args:
            column: The image column's name, as ``list_columns()`` reports it
                with role ``image``.
            index: The sweep point, within the written prefix.
            loop1: Reading-loop slot-1 index, when the run looped.
            loop2: Reading-loop slot-2 index, when the run looped.

        Returns:
            The ``(height_px, width_px)`` frame as a float64 array.

        Raises:
            KeyError: If the run has no such column.
            ValueError: If the column is not an image block, or the handle is
                closed.
            IndexError: If ``index`` is outside the written prefix, or a
                loop index does not fit the column.
        """
        dataset = self._dataset(column)
        metadata = self.read_metadata()
        roles = roles_from_data_config(metadata["raw"].get("data_config") or {})
        if self._role_for(column, dataset, roles) != ROLE_IMAGE:
            raise ValueError(f"{column!r} is not an image block")
        self._refresh(dataset)
        length = min(self.n_points, int(dataset.shape[0]) if dataset.shape else 0)
        if not 0 <= index < length:
            raise IndexError(
                f"sweep point {index} is outside the written prefix of {column!r} "
                f"({length} point(s))"
            )
        return select_frame(np.asarray(dataset[index]), loop1, loop2)

    def summary_stats(self, column: str) -> Stats:
        """Return the NaN-aware summary of one numeric column.

        Args:
            column: The column name, as ``list_columns()`` reports it.

        Returns:
            The column's ``Stats``, taken over the written prefix's finite
            values only.

        Raises:
            KeyError: If the run has no such column.
            ValueError: If the column is not numeric, or the handle is closed.
        """
        return summarise_values(column, self.read_slice(column))

    def read_metadata(self) -> dict[str, Any]:
        """Return the run's metadata under the canonical keys.

        The writer stores each ``/metadata`` attribute as JSON, so every
        structured attribute is decoded here. ``run_kind`` comes from the
        file's own ``/metadata.run_kind`` — the one manifest field the writer
        does record, so a **probe run**'s file identifies itself as a probe to
        whoever opens it later; a file written before that attribute existed
        answers ``""``. ``run_id``/``status``/``reason`` do come back empty:
        they belong to the run manifest the engine emits, and the file does
        not carry them — a live ``RunBuffer`` answers them from the manifest
        it was started with.

        Returns:
            A dict carrying exactly ``RUN_METADATA_KEYS``. ``raw`` holds every
            decoded file attribute, including the ones with no canonical key
            (``instrument_state``, ``system_targets``,
            ``measurement_commands``, ``data_config``).

        Raises:
            ValueError: If the handle is closed.
        """
        raw = self._raw_metadata()
        experiment_info = raw.get("experiment_info") or {}
        if not isinstance(experiment_info, dict):
            experiment_info = {}
        params = raw.get("procedure_params") or {}
        return {
            "source": "file",
            "run_id": "",
            "run_kind": str(raw.get("run_kind", "")),
            "procedure": str(raw.get("procedure_name", "")),
            "params": params if isinstance(params, dict) else {},
            "sample": raw.get("sample_info") or {},
            "setup": experiment_info.get("setup") or {},
            "experiment": experiment_info.get("experiment") or {},
            "start_time": str(raw.get("start_time", "")),
            "end_time": str(raw.get("end_time", "")),
            "status": "",
            "reason": "",
            "data_file": str(self.path),
            "raw": raw,
        }

    # ── Internals ─────────────────────────────────────────────────────

    def _require_open(self) -> h5py.File:
        """Return the open file.

        Returns:
            The underlying ``h5py.File``.

        Raises:
            ValueError: If the handle has been closed.
        """
        if not self._file:
            raise ValueError(f"run handle for {self.path} is closed")
        return self._file

    def _data_group(self) -> h5py.Group:
        """Return the ``/data`` group.

        Returns:
            The group holding every column.

        Raises:
            ValueError: If the handle is closed.
        """
        return self._require_open()[DATA_GROUP]

    def _raw_metadata(self) -> dict[str, Any]:
        """Return every ``/metadata`` attribute, decoded.

        Returns:
            ``{attribute_name: decoded value}``.

        Raises:
            ValueError: If the handle is closed.
        """
        group = self._require_open()[METADATA_GROUP]
        return {key: _decode_attr(value) for key, value in group.attrs.items()}

    def _allocated_points(self) -> int:
        """Return the sweep length the file was allocated for.

        Returns:
            The leading axis shared by every column, or ``0`` for a file with
            no columns at all.

        Raises:
            ValueError: If the handle is closed.
        """
        group = self._data_group()
        return max(
            (int(group[name].shape[0]) for name in group if group[name].shape),
            default=0,
        )

    def _loop_axis_values(self) -> dict[str, list[Any]]:
        """Return the reading loop's axis values, by column name.

        Returns:
            ``{"loop1": [...], ...}``, empty when the run had no reading loop.

        Raises:
            ValueError: If the handle is closed.
        """
        return loop_axis_columns(self.read_metadata()["params"])

    def _role_for(
        self, name: str, dataset: h5py.Dataset, roles: Mapping[str, str]
    ) -> str:
        """Return one dataset's role.

        The writer's ``data_config`` declaration decides; a column it does not
        mention falls back to the dataset's own self-description — the
        ``block_kind`` attribute every block carries, then the ``axes``
        attribute a raw diagnostic block carries — and then to its shape.

        Args:
            name: The column name.
            dataset: The column's dataset.
            roles: ``roles_from_data_config()`` for this file.

        Returns:
            One of ``COLUMN_ROLES``.
        """
        declared = roles.get(name)
        if declared is not None:
            return declared
        if _decode_attr(dataset.attrs.get(BLOCK_KIND_ATTR, "")) == BLOCK_KIND_IMAGE:
            return ROLE_IMAGE
        axes = _decode_attr(dataset.attrs.get("axes", ""))
        if isinstance(axes, str) and "channel" in axes:
            return ROLE_RAW_BLOCK
        if len(dataset.shape) <= 1:
            return ROLE_SWEEP_AXIS
        return ROLE_MEASUREMENT

    def _dataset(self, column: str) -> h5py.Dataset:
        """Return one column's dataset.

        Args:
            column: The column name.

        Returns:
            The dataset under ``/data``.

        Raises:
            KeyError: If the run has no such column.
            ValueError: If the handle is closed.
        """
        group = self._data_group()
        if column not in group:
            raise KeyError(
                f"{self.path.name} has no column {column!r}; "
                f"available: {[info.name for info in self.list_columns()]}"
            )
        return group[column]

    def _refresh(self, dataset: h5py.Dataset) -> None:
        """Re-read a dataset's metadata, but only under SWMR.

        ``Dataset.refresh()`` is an SWMR operation: on a handle whose file is
        not in SWMR mode it does not fail cleanly, it leaves the handle unable
        to read its own variable-length datasets. So this refreshes only when
        the file really is in SWMR mode, and is a no-op otherwise — which
        costs nothing, since a plain in-process read already sees the writer's
        flushed points (see the module docstring).

        Args:
            dataset: The dataset about to be read.
        """
        try:
            if not self._require_open().swmr_mode:
                return
            dataset.refresh()
        except (OSError, RuntimeError, ValueError, AttributeError) as exc:
            logger.debug(
                "data_reader: refresh of %s not available (%s)", dataset.name, exc
            )


def summarise_values(column: str, values: Any) -> Stats:
    """Summarise one column's values, NaN-aware, flattened across every axis.

    The single implementation of the ``Stats`` contract, shared by every
    ``RunSource`` so a file and a live buffer cannot drift apart on what
    "mean" means.

    Args:
        column: The column name, recorded in the returned ``Stats``.
        values: The column's values — anything ``numpy`` can turn into a
            float array (a dataset slice, a nested list of readings).

    Returns:
        The ``Stats``: ``count`` finite values, their min/max/mean/population
        std, and the first and last of them in storage order. A column with no
        finite value reports ``count == 0`` and NaN everywhere.

    Raises:
        ValueError: If the values are not numeric (a string column has no
            statistics).
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} is not numeric") from exc
    finite = array.reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        nan = float("nan")
        return Stats(
            column=column,
            count=0,
            min=nan,
            max=nan,
            mean=nan,
            std=nan,
            first=nan,
            last=nan,
        )
    return Stats(
        column=column,
        count=int(finite.size),
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        mean=float(np.mean(finite)),
        std=float(np.std(finite)),
        first=float(finite[0]),
        last=float(finite[-1]),
    )


def _open_attempts(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield the read-open strategies to try, best first.

    Args:
        path: The file to open, used only for the log line.

    Yields:
        ``(mode, kwargs)`` pairs for ``h5py.File(path, "r", **kwargs)``: a
        plain read first (the normal case, and the one that works inside the
        writing process), then a SWMR read (for a writer that turned SWMR on,
        which ``DataManager`` does not), then a lock-free read (the only one
        another process can use while the writer holds the file's lock).
    """
    logger.debug("data_reader: opening %s read-only", path)
    yield "read", {}
    yield "swmr", {"swmr": True}
    yield "unlocked", {"locking": False}


def open_run(path: str | Path) -> RunHandle:
    """Open a run's HDF5 file read-only.

    Usable as a context manager (``with open_run(p) as run:``) or directly,
    in which case the caller closes the handle.

    Args:
        path: Path to the ``.h5`` file ``DataManager`` wrote.

    Returns:
        An open ``RunHandle``; its ``mode`` says which read strategy the file
        allowed.

    Raises:
        FileNotFoundError: If the path does not exist.
        DataSchemaError: If the file is not a I2AS run file (no ``/data``
            or ``/metadata`` group).
        OSError: If the file exists but no read strategy could open it — a
            writer holding an exclusive lock this build cannot bypass, or a
            corrupt file.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"no run file at {file_path}")

    last_error: BaseException | None = None
    for mode, kwargs in _open_attempts(file_path):
        try:
            handle = h5py.File(file_path, "r", **kwargs)
        except (OSError, ValueError, TypeError) as exc:
            last_error = exc
            continue
        if DATA_GROUP not in handle or METADATA_GROUP not in handle:
            handle.close()
            raise DataSchemaError(
                f"{file_path} is not a I2AS run file: expected "
                f"/{DATA_GROUP} and /{METADATA_GROUP} groups"
            )
        logger.debug("data_reader: opened %s in %s mode", file_path, mode)
        return RunHandle(path=file_path, file=handle, mode=mode)

    raise OSError(f"could not open {file_path} for reading: {last_error}")


def list_columns(source: RunSource) -> tuple[ColumnInfo, ...]:
    """Return every column of a run.

    Args:
        source: Any ``RunSource`` — an open ``RunHandle`` or a live
            ``RunBuffer``.

    Returns:
        One ``ColumnInfo`` per column, in name order.
    """
    return source.list_columns()


def read_slice(
    source: RunSource,
    column: str,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> np.ndarray:
    """Return a slice of one column along its leading axis.

    Args:
        source: Any ``RunSource``.
        column: The column name.
        start: First sweep point, or ``None`` for the beginning.
        stop: Stop before this sweep point, or ``None`` for the end.
        step: Stride, or ``None`` for every point; must be positive.

    Returns:
        A numpy array whose leading axis is the selected sweep points.

    Raises:
        KeyError: If the run has no such column.
        ValueError: If ``step`` is not positive.
    """
    return source.read_slice(column, start, stop, step)


def summary_stats(source: RunSource, column: str) -> Stats:
    """Return the NaN-aware summary of one numeric column.

    Args:
        source: Any ``RunSource``.
        column: The column name.

    Returns:
        The column's ``Stats``.

    Raises:
        KeyError: If the run has no such column.
        ValueError: If the column is not numeric.
    """
    return source.summary_stats(column)


def read_metadata(source: RunSource) -> dict[str, Any]:
    """Return a run's metadata under the canonical keys.

    Args:
        source: Any ``RunSource``.

    Returns:
        A dict carrying exactly ``RUN_METADATA_KEYS``.
    """
    return source.read_metadata()
