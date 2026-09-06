from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

_SENTINEL = -1  # used when no point has been saved yet


class DataManager:
    """Create and manage an HDF5 measurement file for one procedure run.

    Lifecycle
    ---------
    1. Instantiate at ``Procedure.initiate()``.
    2. Call ``save_datapoint()`` once per sweep step inside the measurement
       loop.
    3. Call ``close()`` at ``Procedure.standby()`` (or on abort).  The file
       is trimmed to the number of points actually saved.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        data_directory: str,
        procedure_name: str,
        procedure_params: dict,
        sample_info: dict,
        instrument_state: dict,
        system_targets: dict,
        measurement_commands: dict | list,
        data_config: dict,
        n_sweep_points: int,
        file_prefix: str = "",
        experiment_info: dict | None = None,
        run_kind: str = "run",
    ) -> None:
        """Create the HDF5 file and write all metadata.

        Parameters
        ----------
        data_directory:
            Root directory where the HDF5 file will be written.
        procedure_name:
            Short name stored as metadata (``/metadata/procedure_name``). Also
            used as the filename prefix when ``file_prefix`` is empty.
        file_prefix:
            User-chosen filename prefix. When non-empty, the file is named
            ``{file_prefix}_{timestamp}.h5`` instead of
            ``{procedure_name}_{timestamp}.h5``. Metadata still records the
            true ``procedure_name`` regardless.
        procedure_params:
            Arbitrary procedure parameters (JSON-serialisable dict).
        sample_info:
            Sample description dict (JSON-serialisable).
        instrument_state:
            Snapshot of instrument state at initiation (JSON-serialisable).
        system_targets:
            Physical targets (field, temperature …) for this run.
        measurement_commands:
            JSON-serialisable description of the measurement commands used.
            Since the typed-plan cutover this is an ordered list of Command
            dicts (``[{"vi_name": ..., "method": ..., "kwargs": {...}}, ...]``);
            a plain dict is still accepted (older files / direct callers).
        data_config:
            Specifies datasets.  Expected format::

                {
                    "sweep_columns": {"field_T": "float", ...},        # (N,)
                    "measurement_scalars": {"voltage_V": "float", ...},  # (N, n_loop1, n_loop2)
                    "measurement_arrays": {"voltage_V_array": 100, ...}, # (N, n_loop1, n_loop2, 100)
                    "measurement_blocks": {"raw_channels_block": (5, 44), ...},
                                                                # (N, n_loop1, n_loop2, 5, 44)
                    "measurement_block_labels": {"raw_channels_block": ["ch0", ...]},
                                                                # ordered channel names, len == cols
                    "measurement_image_blocks": {"frame": {"unit": "counts",
                                                           "description": "..."}},
                                                                # which blocks are frames
                    "loop_shape": [n_loop1, n_loop2],                   # each >= 1
                }

            ``loop_shape`` defaults to ``[1, 1]`` (no reading loop) when
            absent, so callers that never loop can omit it.
            ``measurement_block_labels`` is optional; when a block name is
            present, its label list is written as the ``channel_names``
            HDF5 attribute directly on that block's dataset (see
            :meth:`_allocate_datasets`), so the file is self-describing
            without parsing this ``data_config`` JSON blob.
            ``measurement_image_blocks`` is optional too: a block named
            there is an IMAGE (the image-block standard, ``core.plan.
            ImageBlock``) — the same ``(rows, cols)`` dataset, written with
            ``block_kind = "image"`` and its ``unit``/``description``
            attributes and no ``channel_names``. Every other block is
            written with ``block_kind = "raw"``.

        n_sweep_points:
            Total number of sweep points expected (used for pre-allocation).
        experiment_info:
            Optional experiment-level context (JSON-serialisable dict) from the
            session layer — experiment id/title, user identity, ELN link. Stored
            as ``/metadata/experiment_info``; ``None`` is recorded as ``{}`` so
            the attribute always exists.
        run_kind:
            What kind of run wrote this file — ``"run"`` for a science run,
            ``"probe"`` for a **probe run** (the cheap reduced variant; see
            ``core/plan.py``'s ``ProbeSpec``), ``"operation"`` for a servicing
            operation. Stored as ``/metadata/run_kind`` and returned by
            ``data_reader.read_metadata()``, so a probe file can never be
            mistaken for science data by whoever opens it later. Written by the
            run itself from its own ``run_kind`` attribute; the default keeps
            every existing caller writing a science run.
        """
        if n_sweep_points < 1:
            raise ValueError(f"n_sweep_points must be >= 1, got {n_sweep_points}")

        self._procedure_name = procedure_name
        self._run_kind = str(run_kind or "run")
        self._n_sweep_points = n_sweep_points
        self._data_config = data_config
        self._last_saved_index: int = _SENTINEL
        self._closed = False
        self.last_datapoint: dict = {}  # updated by save_datapoint(); read by GUI for live plot

        # Derive column / array names from data_config
        self._sweep_columns: dict[str, str] = data_config.get("sweep_columns", {})
        self._measurement_scalars: dict[str, str] = data_config.get(
            "measurement_scalars", {}
        )
        self._measurement_arrays: dict[str, int] = data_config.get(
            "measurement_arrays", {}
        )
        self._measurement_blocks: dict[str, tuple[int, int]] = {
            name: (int(shape[0]), int(shape[1]))
            for name, shape in data_config.get("measurement_blocks", {}).items()
        }
        self._measurement_block_labels: dict[str, list[str]] = {
            name: list(labels)
            for name, labels in data_config.get("measurement_block_labels", {}).items()
        }
        self._measurement_image_blocks: dict[str, dict[str, str]] = {
            name: {
                "unit": str(info.get("unit", "")),
                "description": str(info.get("description", "")),
            }
            for name, info in data_config.get("measurement_image_blocks", {}).items()
        }
        loop_shape = data_config.get("loop_shape", [1, 1])
        self._loop_shape: tuple[int, int] = (int(loop_shape[0]), int(loop_shape[1]))

        # Build file path
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = Path(data_directory)
        data_dir.mkdir(parents=True, exist_ok=True)
        stem = file_prefix.strip() or procedure_name
        self._filepath = data_dir / f"{stem}_{timestamp_str}.h5"

        logger.info("DataManager: creating HDF5 file at %s", self._filepath)

        self._file = h5py.File(self._filepath, "w")

        # Write metadata
        self._write_metadata(
            procedure_params=procedure_params,
            sample_info=sample_info,
            instrument_state=instrument_state,
            system_targets=system_targets,
            measurement_commands=measurement_commands,
            data_config=data_config,
            experiment_info=experiment_info or {},
        )

        # Pre-allocate datasets
        self._allocate_datasets()

        self._file.flush()
        logger.debug("DataManager: initialisation complete (%d points)", n_sweep_points)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _write_metadata(
        self,
        procedure_params: dict,
        sample_info: dict,
        instrument_state: dict,
        system_targets: dict,
        measurement_commands: dict | list,
        data_config: dict,
        experiment_info: dict,
    ) -> None:
        """Write all metadata to `/metadata/` as JSON-encoded HDF5 attributes."""
        meta = self._file.require_group("metadata")
        meta.attrs["procedure_name"] = self._procedure_name
        meta.attrs["run_kind"] = self._run_kind
        meta.attrs["procedure_params"] = json.dumps(procedure_params)
        meta.attrs["sample_info"] = json.dumps(sample_info)
        meta.attrs["experiment_info"] = json.dumps(experiment_info)
        meta.attrs["start_time"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["end_time"] = ""  # filled in at close()
        meta.attrs["instrument_state"] = json.dumps(instrument_state)
        meta.attrs["system_targets"] = json.dumps(system_targets)
        meta.attrs["measurement_commands"] = json.dumps(measurement_commands)
        meta.attrs["data_config"] = json.dumps(data_config)

    def _allocate_datasets(self) -> None:
        """Pre-allocate all datasets in `/data/` with NaN fill values."""
        N = self._n_sweep_points
        n_loop1, n_loop2 = self._loop_shape
        data_group = self._file.require_group("data")

        # 1-D sweep columns — one value per sweep point, never looped.
        for col_name in self._sweep_columns:
            data_group.create_dataset(
                col_name,
                shape=(N,),
                maxshape=(None,),
                dtype=np.float64,
                fillvalue=np.nan,
            )

        # 3-D measurement scalars — a (n_loop1, n_loop2) grid per sweep point.
        for col_name in self._measurement_scalars:
            data_group.create_dataset(
                col_name,
                shape=(N, n_loop1, n_loop2),
                maxshape=(None, n_loop1, n_loop2),
                dtype=np.float64,
                fillvalue=np.nan,
            )

        # 4-D measurement arrays — a (n_loop1, n_loop2, M) grid per sweep point.
        for arr_name, M in self._measurement_arrays.items():
            data_group.create_dataset(
                arr_name,
                shape=(N, n_loop1, n_loop2, M),
                maxshape=(None, n_loop1, n_loop2, M),
                dtype=np.float64,
                fillvalue=np.nan,
            )

        # Raw diagnostic blocks — a (rows, cols) grid per sweep point, UNLESS
        # a reading loop is actually configured, in which case it gains the
        # (n_loop1, n_loop2) axis like a measurement array does (see
        # MeasurementInstrumentBase's "Raw diagnostic blocks" standard and
        # DataSchema.measurement_blocks's docstring for why the trivial
        # (1, 1) axis is skipped rather than always carried).
        block_loop_prefix = (n_loop1, n_loop2) if (n_loop1, n_loop2) != (1, 1) else ()
        for block_name, (rows, cols) in self._measurement_blocks.items():
            shape = (N, *block_loop_prefix, rows, cols)
            maxshape = (None, *block_loop_prefix, rows, cols)
            block_ds = data_group.create_dataset(
                block_name,
                shape=shape,
                maxshape=maxshape,
                dtype=np.float64,
                fillvalue=np.nan,
            )
            # Self-description: what kind of block this is, what each axis
            # means and — for a raw block — which column index is which
            # physical channel, or — for an image — the pixel unit; all
            # written directly on the dataset so a reader never needs to
            # parse the JSON data_config metadata blob (see the __init__
            # docstring's measurement_block_labels / measurement_image_blocks
            # entries).
            image = self._measurement_image_blocks.get(block_name)
            axis_names = (
                "sweep_point",
                *(("loop1", "loop2") if block_loop_prefix else ()),
                "row",
                "col" if image is not None else "channel",
            )
            block_ds.attrs["axes"] = ", ".join(axis_names)
            if image is not None:
                block_ds.attrs["block_kind"] = "image"
                block_ds.attrs["unit"] = image["unit"]
                block_ds.attrs["description"] = image["description"]
                continue
            block_ds.attrs["block_kind"] = "raw"
            labels = self._measurement_block_labels.get(block_name)
            if labels is not None:
                block_ds.attrs["channel_names"] = np.array(labels, dtype=h5py.string_dtype())

        # Timestamp column (variable-length strings)
        dt = h5py.string_dtype()
        data_group.create_dataset(
            "timestamp",
            shape=(N,),
            maxshape=(None,),
            dtype=dt,
            fillvalue="",
        )

        # Snapshots group (datasets created on demand)
        self._file.require_group("snapshots")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def filepath(self) -> Path:
        """Path to the HDF5 file on disk."""
        return self._filepath

    def save_datapoint(
        self,
        sweep_index: int,
        measured_data: dict,
        station_snapshot: dict,
    ) -> None:
        """Save one sweep point's data and a full station snapshot.

        Parameters
        ----------
        sweep_index:
            Zero-based index of this sweep point (0 … n_sweep_points-1).
        measured_data:
            Dict whose keys match sweep_columns, measurement_scalars,
            measurement_arrays or measurement_blocks names. Sweep-column
            values are plain scalars; measurement_scalars values are a
            nested ``(n_loop1, n_loop2)`` grid; measurement_arrays values
            are a nested ``(n_loop1, n_loop2, length)`` grid;
            measurement_blocks values are a nested ``(n_loop1, n_loop2,
            rows, cols)`` grid. Lists, tuples and numpy arrays are all
            accepted.
        station_snapshot:
            Full instrument-state snapshot to store as a JSON string.

        Raises:
            ValueError: If a measurement scalar's grid shape doesn't match
                ``loop_shape``, or a measurement array's/block's grid shape
                doesn't match ``loop_shape`` on the loop axes (a per-point
                sample/row count mismatch on the innermost — for a block,
                second-to-innermost — axis is NOT an error — see below).
        """
        if self._closed:
            raise RuntimeError("save_datapoint() called on a closed DataManager")
        if not (0 <= sweep_index < self._n_sweep_points):
            raise IndexError(
                f"sweep_index {sweep_index} out of range [0, {self._n_sweep_points})"
            )

        data_group = self._file["data"]
        loop_shape = self._loop_shape

        for col_name, value in measured_data.items():
            if col_name in self._sweep_columns:
                data_group[col_name][sweep_index] = float(value)
            elif col_name in self._measurement_scalars:
                arr = np.asarray(value, dtype=np.float64)
                if arr.shape != loop_shape:
                    raise ValueError(
                        f"DataManager: measurement scalar '{col_name}' at "
                        f"index {sweep_index} has shape {arr.shape}, "
                        f"expected loop shape {loop_shape}"
                    )
                data_group[col_name][sweep_index, ...] = arr
            elif col_name in self._measurement_arrays:
                expected_length = int(self._measurement_arrays[col_name])
                expected_shape = (*loop_shape, expected_length)
                arr = np.asarray(value, dtype=np.float64)
                if arr.shape != expected_shape:
                    if arr.shape[:-1] == loop_shape:
                        # Instruments can legitimately return fewer readings
                        # than allocated (e.g. the delta engine aborts an
                        # acquisition early). Pad/truncate only the innermost
                        # (per-point sample) axis with NaN — the loop-axis
                        # shape itself is never adjusted.
                        logger.warning(
                            "DataManager: column '%s' at index %d has inner "
                            "length %d (expected %d) — padding/truncating "
                            "with NaN",
                            col_name, sweep_index, arr.shape[-1], expected_length,
                        )
                        padded = np.full(expected_shape, np.nan)
                        n = min(arr.shape[-1], expected_length)
                        padded[..., :n] = arr[..., :n]
                        arr = padded
                    else:
                        raise ValueError(
                            f"DataManager: measurement array '{col_name}' at "
                            f"index {sweep_index} has shape {arr.shape}, "
                            f"expected {expected_shape}"
                        )
                data_group[col_name][sweep_index, ...] = arr
            elif col_name in self._measurement_blocks:
                expected_rows, expected_cols = self._measurement_blocks[col_name]
                # No loop-axis prefix when no reading loop is configured —
                # see DataSchema.measurement_blocks's docstring.
                block_loop_prefix = loop_shape if loop_shape != (1, 1) else ()
                expected_shape = (*block_loop_prefix, expected_rows, expected_cols)
                arr = np.asarray(value, dtype=np.float64)
                if arr.shape != expected_shape:
                    if arr.shape[:-2] == block_loop_prefix and arr.shape[-1] == expected_cols:
                        # Same "pad the under-delivered rows axis with NaN"
                        # fallback the measurement-array branch above has —
                        # the channel axis is fixed and never padded, only
                        # the ROWS axis (see raw_block_row_counts()).
                        logger.warning(
                            "DataManager: block '%s' at index %d has %d rows "
                            "(expected %d) — padding/truncating with NaN",
                            col_name, sweep_index, arr.shape[-2], expected_rows,
                        )
                        padded = np.full(expected_shape, np.nan)
                        n = min(arr.shape[-2], expected_rows)
                        padded[..., :n, :] = arr[..., :n, :]
                        arr = padded
                    else:
                        raise ValueError(
                            f"DataManager: measurement block '{col_name}' at "
                            f"index {sweep_index} has shape {arr.shape}, "
                            f"expected {expected_shape}"
                        )
                data_group[col_name][sweep_index, ...] = arr
            else:
                logger.warning(
                    "DataManager: unknown column '%s' — skipped", col_name
                )

        # Timestamp
        data_group["timestamp"][sweep_index] = datetime.now(timezone.utc).isoformat()

        # Snapshot as variable-length string dataset
        snap_json = json.dumps(station_snapshot)
        self._file["snapshots"].create_dataset(
            str(sweep_index), data=snap_json
        )

        self._last_saved_index = sweep_index
        self.last_datapoint: dict = measured_data
        self._file.flush()

    def record_settings_snapshot(self, snapshot: dict) -> None:
        """Record an externally configured measurement VI's arming-time settings.

        Writes *snapshot* as a JSON-encoded HDF5 attribute
        ``/metadata.measurement_settings_snapshot`` — the provenance record
        for a run where a measurement VI's ``initiate_measurement()`` skips
        I2AS's own excitation/analysis configuration (see
        ``MeasurementInstrumentBase``'s "Externally configured instruments"
        standard) and instead exposes what it was actually armed with as
        ``last_settings_snapshot``. HDF5 group attributes are writable any
        time before ``close()``, so this may be called after construction,
        once the measurement VI has armed.

        Args:
            snapshot: The externally configured VI's arming-time settings,
                read from the instrument at arming time.

        Raises:
            RuntimeError: If called on a closed DataManager.
        """
        if self._closed:
            raise RuntimeError(
                "record_settings_snapshot() called on a closed DataManager"
            )
        self._file["metadata"].attrs["measurement_settings_snapshot"] = json.dumps(
            snapshot
        )
        self._file.flush()

    def close(self) -> None:
        """Close the HDF5 file, record end_time, and trim on early abort."""
        if self._closed:
            logger.warning("DataManager.close() called on an already-closed file")
            return

        self._file["metadata"].attrs["end_time"] = datetime.now(
            timezone.utc
        ).isoformat()

        # Trim datasets to actual points saved (handles early abort)
        actual_points = self._last_saved_index + 1  # 0 if nothing saved

        if 0 < actual_points < self._n_sweep_points:
            logger.info(
                "DataManager: trimming datasets from %d to %d points",
                self._n_sweep_points,
                actual_points,
            )
            data_group = self._file["data"]
            for name in data_group:
                ds = data_group[name]
                new_shape = (actual_points,) + ds.shape[1:]
                ds.resize(new_shape)

        self._file.flush()
        self._file.close()
        self._closed = True
        logger.info("DataManager: closed %s", self._filepath)
