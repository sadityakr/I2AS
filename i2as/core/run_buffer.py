"""The run in flight, answering the same questions as the run on disk.

``RunBuffer`` is the live half of the **one vocabulary for live and stored
runs** standard (``data_reader.py``'s module docstring is the full text): it
accumulates the run's ``Datapoint`` events in memory and answers
``list_columns()`` / ``read_slice()`` / ``summary_stats()`` /
``read_metadata()`` with the very same ``ColumnInfo`` and ``Stats`` types a
``RunHandle`` answers them with, so a consumer written against ``RunSource``
reads the run being measured and the runs already written with one body of
code.

Why a buffer at all, when ``data_reader`` can open the file mid-run: the
writer owns that file, HDF5 gives a second reader only what has been flushed,
and a reader outside the writing process cannot count on a handle that grows
(``data_reader``'s module docstring states the limitation). The buffer has
none of those constraints — it is fed the same values the writer is fed, at
the same moment.

Pure Python: no Qt, no h5py, no Station. It holds only what it was told.

**What feeds it.** ``Datapoint.values`` is exactly the ``measured_data``
mapping ``DataManager.save_datapoint()`` receives, and ``Datapoint.index``
exactly its ``sweep_index``; a sweep column is a scalar, a measurement scalar
a nested ``(n_loop1, n_loop2)`` grid, a measurement array a
``(n_loop1, n_loop2, length)`` grid, a raw diagnostic block a
``([n_loop1, n_loop2,] rows, cols)`` grid. So the buffer stores each column
with the writer's own leading-axis-is-the-sweep-point layout and reports the
same shapes. The one column it adds itself is ``timestamp``, from
``Datapoint.ts``, because the writer stamps that column rather than being
handed it. ``RunStarted.manifest`` supplies the metadata: ``run_id``,
``procedure``, ``kind``, ``params`` (which carry the **reading loop**'s
``loop1_values``/``loop2_values``, and therefore the ``loop_axis`` columns)
and ``data_file``; when the manifest also carries the run's ``data_config``,
the buffer takes every column's role from it, exactly as a file does.
Without that declaration the roles are inferred from the values — a scalar is
a sweep axis, anything else a measurement — and a raw diagnostic block is
indistinguishable from a measurement grid, which is the only thing a buffer
can report less precisely than a file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from i2as.core import data_reader as reader
from i2as.core.data_reader import ColumnInfo, Stats
from i2as.core.events import Datapoint, RunFinished, RunStarted
from i2as.core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)


def _isoformat(unix_time: float) -> str:
    """Render a unix timestamp the way the writer stamps its own.

    Args:
        unix_time: Seconds since the epoch, as an event's ``ts`` carries it.

    Returns:
        An ISO-8601 string in UTC, matching ``/data/timestamp``'s format.
    """
    return datetime.fromtimestamp(unix_time, tz=timezone.utc).isoformat()


@dataclass
class _Column:
    """One accumulating column of the run in flight.

    Attributes:
        name: The column name, as the datapoint carried it.
        role: One of ``data_reader.COLUMN_ROLES``.
        dtype: ``"float64"`` or ``"str"``.
        inner_shape: The shape of one sweep point's value — ``()`` for a
            sweep column, ``(n_loop1, n_loop2)`` for a measurement scalar, and
            so on. The stored column's shape is the point count followed by
            this.
        values: One entry per sweep point received, in index order; a point
            this column had no value for holds NaN (or ``""``).
    """

    name: str
    role: str
    dtype: str
    inner_shape: tuple[int, ...]
    values: list[Any] = field(default_factory=list)

    def missing(self) -> Any:
        """Return the placeholder for a point this column has no value for.

        Returns:
            An empty string for a text column, otherwise NaN in this column's
            own shape — the same "nothing measured here" a pre-allocated HDF5
            dataset holds.
        """
        if self.dtype == "str":
            return ""
        if not self.inner_shape:
            return float("nan")
        return np.full(self.inner_shape, np.nan)


class RunBuffer:
    """The run in progress as a ``data_reader.RunSource``.

    Fed by the run's control-contract events — ``start(RunStarted)``,
    ``append(Datapoint)`` per point, ``finish(RunFinished)`` — and read
    through the same vocabulary as a finished file::

        buffer = RunBuffer()
        buffer.start(run_started)
        buffer.append(datapoint)
        stats = buffer.summary_stats("voltage_V")

    Reusable: ``start()`` clears whatever the previous run left, so one
    buffer can follow a session's runs one after another.
    """

    def __init__(self) -> None:
        """Create an empty buffer, holding no run."""
        self._columns: dict[str, _Column] = {}
        self._n_points = 0
        self._manifest: dict[str, Any] = {}
        self._roles: dict[str, str] = {}
        self._run_id = ""
        self._started = False
        self._running = False
        self._status = ""
        self._reason = ""
        self._start_time = ""
        self._end_time = ""

    # ── Fed by the run's events ───────────────────────────────────────

    def start(self, event: RunStarted) -> None:
        """Begin buffering a run, discarding any previous one.

        Args:
            event: The ``RunStarted`` the engine emitted once the run's setup
                succeeded. Its manifest supplies every metadata answer, and
                its ``data_config`` (when present) the columns' roles.
        """
        self._columns = {}
        self._n_points = 0
        self._manifest = dict(event.manifest)
        self._roles = reader.roles_from_data_config(
            self._manifest.get("data_config") or {}
        )
        self._run_id = event.run_id
        self._started = True
        self._running = True
        self._status = ""
        self._reason = ""
        self._start_time = str(self._manifest.get("started_utc") or "") or _isoformat(
            event.ts
        )
        self._end_time = ""
        logger.info("run buffer: buffering run %s", self._run_id or "(unnamed)")

    def append(self, event: Datapoint) -> None:
        """Record one measured point.

        The point is placed at its own ``index``, so a buffer fed out of order
        still reads back in sweep order and a skipped index reads as NaN
        rather than shifting every later point.

        Args:
            event: The ``Datapoint``; its ``values`` are the mapping
                ``DataManager.save_datapoint()`` receives.

        Raises:
            DataSchemaError: If no run has been started, if the index is
                negative, or if a value's shape does not match the shape that
                column's first value declared.
        """
        if not self._started:
            raise DataSchemaError(
                "run buffer: append() before start() — a datapoint has no run "
                "to belong to"
            )
        if event.run_id != self._run_id:
            logger.warning(
                "run buffer: datapoint for run %r ignored, buffering %r",
                event.run_id,
                self._run_id,
            )
            return
        index = int(event.index)
        if index < 0:
            raise DataSchemaError(f"run buffer: negative datapoint index {index}")

        values: dict[str, Any] = dict(event.values)
        values.setdefault(reader.TIMESTAMP_COLUMN, _isoformat(event.ts))
        self._n_points = max(self._n_points, index + 1)
        for name, value in values.items():
            column = self._columns.get(name) or self._register(name, value)
            self._pad(column)
            column.values[index] = self._conform(column, value)
        for column in self._columns.values():
            self._pad(column)

    def finish(self, event: RunFinished) -> None:
        """Record how the run ended; the data already buffered stays readable.

        Args:
            event: The ``RunFinished`` the engine emitted. Its manifest is
                merged over the started manifest, so the finished-run keys
                (``status``, ``finished_utc``, ``summary``) join the started
                ones.
        """
        if event.run_id != self._run_id:
            logger.warning(
                "run buffer: run_finished for %r ignored, buffering %r",
                event.run_id,
                self._run_id,
            )
            return
        self._manifest.update(event.manifest)
        self._running = False
        self._status = event.status
        self._reason = event.reason
        self._end_time = str(self._manifest.get("finished_utc") or "") or _isoformat(
            event.ts
        )
        logger.info(
            "run buffer: run %s finished (%s), %d points buffered",
            self._run_id or "(unnamed)",
            self._status or "unknown",
            self._n_points,
        )

    # ── Buffer state ──────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        """The run being buffered, or ``""`` before the first ``start()``."""
        return self._run_id

    @property
    def is_running(self) -> bool:
        """Whether the buffered run has started and not yet finished."""
        return self._running

    # ── The run-source vocabulary ─────────────────────────────────────

    @property
    def n_points(self) -> int:
        """How many sweep points the buffer holds.

        Returns:
            One past the highest datapoint index received, so a run fed points
            0, 1 and 3 reports 4 with point 2 reading as NaN — the same
            written-prefix meaning ``RunHandle.n_points`` carries.
        """
        return self._n_points

    def list_columns(self) -> tuple[ColumnInfo, ...]:
        """Return every column of the run so far, in name order.

        Returns:
            One ``ColumnInfo`` per column received, plus the reading loop's
            ``loop_axis`` columns from the run manifest's parameters. ``path``
            names where each column will live once written, so a consumer can
            match a buffered column to a stored one.
        """
        infos = [
            ColumnInfo(
                name=column.name,
                unit=reader.column_unit(column.name),
                dtype=column.dtype,
                role=column.role,
                shape=(self._n_points, *column.inner_shape),
                length=self._n_points,
                path=f"/{reader.DATA_GROUP}/{column.name}",
            )
            for column in self._columns.values()
        ]
        infos.extend(
            info
            for info in reader.loop_axis_column_infos(self._params())
            if info.name not in self._columns
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

        Only the leading (sweep-point) axis is sliced; the loop, sample and
        block axes of a point's value come back whole, exactly as they do from
        a file.

        Args:
            column: The column name, as ``list_columns()`` reports it.
            start: First sweep point, or ``None`` for the beginning.
            stop: Stop before this sweep point, or ``None`` for the end.
            step: Stride, or ``None`` for every point; must be positive.

        Returns:
            A numpy array whose leading axis is the selected sweep points.

        Raises:
            KeyError: If the buffer holds no such column.
            ValueError: If ``step`` is not positive.
        """
        buffered = self._columns.get(column)
        if buffered is not None:
            selection = reader.resolve_slice(self._n_points, start, stop, step)
            chunk = buffered.values[selection]
            if not chunk:
                if buffered.dtype == "str":
                    return np.empty((0,), dtype="U1")
                return np.empty((0, *buffered.inner_shape), dtype=np.float64)
            if buffered.dtype == "str":
                return np.asarray(chunk)
            return np.asarray(chunk, dtype=np.float64)
        loop_values = reader.loop_axis_columns(self._params()).get(column)
        if loop_values is not None:
            selection = reader.resolve_slice(len(loop_values), start, stop, step)
            return np.asarray(loop_values[selection])
        raise KeyError(
            f"run {self._run_id or '(unnamed)'} has no column {column!r}; "
            f"available: {[info.name for info in self.list_columns()]}"
        )

    def read_image(
        self, column: str, index: int, loop1: int = 0, loop2: int = 0
    ) -> np.ndarray:
        """Return one frame of an image block (the image-block standard).

        Args:
            column: The image column's name (role ``image`` — which a buffer
                knows only from the manifest's ``data_config``).
            index: The sweep point, among the points received so far.
            loop1: Reading-loop slot-1 index, when the run loops.
            loop2: Reading-loop slot-2 index, when the run loops.

        Returns:
            The ``(height_px, width_px)`` frame as a float64 array.

        Raises:
            KeyError: If the buffer holds no such column.
            ValueError: If the column is not an image block.
            IndexError: If ``index`` is not a received point, or a loop index
                does not fit the column.
        """
        buffered = self._columns.get(column)
        if buffered is None:
            raise KeyError(
                f"run {self._run_id or '(unnamed)'} has no column {column!r}; "
                f"available: {[info.name for info in self.list_columns()]}"
            )
        if buffered.role != reader.ROLE_IMAGE:
            raise ValueError(f"{column!r} is not an image block")
        if not 0 <= index < self._n_points:
            raise IndexError(
                f"sweep point {index} has not been received for {column!r} "
                f"({self._n_points} point(s) so far)"
            )
        return reader.select_frame(buffered.values[index], loop1, loop2)

    def summary_stats(self, column: str) -> Stats:
        """Return the NaN-aware summary of one numeric column.

        Args:
            column: The column name, as ``list_columns()`` reports it.

        Returns:
            The column's ``Stats``, from the same ``summarise_values()`` a
            file's statistics come from.

        Raises:
            KeyError: If the buffer holds no such column.
            ValueError: If the column is not numeric.
        """
        return reader.summarise_values(column, self.read_slice(column))

    def read_metadata(self) -> dict[str, Any]:
        """Return the run's metadata under the canonical keys.

        The mirror image of a file's answer: a buffer knows the engine's run
        identity (``run_id``, ``run_kind``, ``status``, ``reason``) because
        the manifest carries it, and leaves ``sample``/``setup``/
        ``experiment`` empty unless the manifest carried those too — the file
        is where the procedure stamps them.

        Returns:
            A dict carrying exactly ``data_reader.RUN_METADATA_KEYS``, with
            ``raw`` holding the whole run manifest.
        """
        return {
            "source": "buffer",
            "run_id": self._run_id,
            "run_kind": str(self._manifest.get("kind", "")),
            "procedure": str(self._manifest.get("procedure", "")),
            "params": self._params(),
            "sample": self._manifest.get("sample") or {},
            "setup": self._manifest.get("setup") or {},
            "experiment": self._manifest.get("experiment") or {},
            "start_time": self._start_time,
            "end_time": self._end_time,
            "status": self._status,
            "reason": self._reason,
            "data_file": str(self._manifest.get("data_file", "")),
            "raw": dict(self._manifest),
        }

    # ── Internals ─────────────────────────────────────────────────────

    def _params(self) -> dict[str, Any]:
        """Return the run's procedure parameters from the manifest.

        Returns:
            The ``params`` mapping, or ``{}`` when the manifest has none.
        """
        params = self._manifest.get("params")
        return dict(params) if isinstance(params, dict) else {}

    def _register(self, name: str, value: Any) -> _Column:
        """Declare a column from the first value seen for it.

        A column that appears only partway through a run is backfilled with
        the "nothing measured here" placeholder, so every column stays
        rectangular and a slice never depends on when a column joined.

        Args:
            name: The column name.
            value: Its first value, whose shape declares the column's.

        Returns:
            The registered column.
        """
        dtype = "str" if isinstance(value, str) else "float64"
        inner_shape = () if dtype == "str" else tuple(np.shape(value))
        column = _Column(
            name=name,
            role=self._role_for(name, value),
            dtype=dtype,
            inner_shape=inner_shape,
        )
        column.values = [column.missing()] * self._n_points
        self._columns[name] = column
        logger.debug(
            "run buffer: column %s registered as %s%s",
            name,
            column.role,
            inner_shape,
        )
        return column

    def _role_for(self, name: str, value: Any) -> str:
        """Return a column's role.

        The run manifest's ``data_config`` decides when it declared one.
        Otherwise the value's shape does: a scalar is a sweep axis and
        anything else a measurement — a raw diagnostic block cannot be told
        from a measurement grid without the declaration (see the module
        docstring).

        Args:
            name: The column name.
            value: Its first value.

        Returns:
            One of ``data_reader.COLUMN_ROLES``.
        """
        declared = self._roles.get(name)
        if declared is not None:
            return declared
        if np.ndim(value) == 0:
            return reader.ROLE_SWEEP_AXIS
        return reader.ROLE_MEASUREMENT

    def _pad(self, column: _Column) -> None:
        """Grow a column to the buffer's point count with placeholders.

        Args:
            column: The column to pad in place.
        """
        while len(column.values) < self._n_points:
            column.values.append(column.missing())

    def _conform(self, column: _Column, value: Any) -> Any:
        """Return one point's value in the column's declared shape.

        Mirrors the writer: an under- or over-delivered innermost axis (an
        acquisition that returned fewer samples than planned) is padded or
        truncated with NaN, while a loop-axis mismatch is an error, because
        that means the value does not belong to this column at all.

        Args:
            column: The column being written.
            value: The datapoint's value for it.

        Returns:
            A float for a scalar column, a string for a text column, otherwise
            an array of the column's inner shape.

        Raises:
            DataSchemaError: If the value is not numeric, or its shape cannot
                be reconciled with the column's.
        """
        if column.dtype == "str":
            return str(value)
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DataSchemaError(
                f"run buffer: column {column.name!r} is numeric, got "
                f"{type(value).__name__}"
            ) from exc
        expected = column.inner_shape
        if array.shape == expected:
            return float(array) if not expected else array
        if expected and array.ndim == len(expected) and array.shape[:-1] == expected[:-1]:
            logger.warning(
                "run buffer: column '%s' has inner length %d (expected %d) — "
                "padding/truncating with NaN",
                column.name,
                array.shape[-1],
                expected[-1],
            )
            padded = np.full(expected, np.nan)
            kept = min(array.shape[-1], expected[-1])
            padded[..., :kept] = array[..., :kept]
            return padded
        raise DataSchemaError(
            f"run buffer: column {column.name!r} has shape {array.shape}, "
            f"expected {expected}"
        )
