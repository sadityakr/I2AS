"""L5 data reader: reading back what the real ``DataManager`` wrote.

Every test here writes with the production writer and reads with the
production reader — no hand-built HDF5 stand-in except where the point is a
file the writer would never produce (a foreign file, a file with no
``timestamp`` counter). That is deliberate: the reader's whole job is to
agree with ``data_manager.py`` about the layout, so a fixture that agreed
with the reader instead would test nothing.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from i2as.core import data_reader as dr
from i2as.core.data_manager import DataManager
from i2as.core.exceptions import DataSchemaError

# ── Fixtures ──────────────────────────────────────────────────────────────
# One reading-loop slot with two values, so every measurement column carries a
# real (2, 1) loop axis and the raw block carries it too — the shapes the
# reader has to recognise, all in one file.

N_POINTS = 4
N_SAMPLES = 3
BLOCK_ROWS, BLOCK_COLS = 2, 3

DATA_CONFIG = {
    "sweep_columns": {"field_T": "float"},
    "measurement_scalars": {"voltage_V": "float"},
    "measurement_arrays": {"voltage_V_array": N_SAMPLES},
    "measurement_blocks": {"raw_channels_block": (BLOCK_ROWS, BLOCK_COLS)},
    "measurement_block_labels": {"raw_channels_block": ["ch0", "ch1", "ch2"]},
    "loop_shape": [2, 1],
}

PROCEDURE_PARAMS = {
    "field_start": -1.0,
    "field_end": 1.0,
    "loop1_parameter": "dc_measurement.current_A",
    "loop1_values": [1e-6, -1e-6],
}

SAMPLE_INFO = {"sample_name": "Test Sample", "sample_id": "TST-001"}

EXPERIMENT_INFO = {
    "setup": {"config_name": "sim_setup", "instruments": {"magnet_z": {"model": "sim"}}},
    "experiment": {"experiment_id": "EXP-7", "user_name": "A. Operator"},
}


def _measured(index: int) -> dict:
    """Return one datapoint's measured data for sweep point *index*.

    Args:
        index: The sweep index, used to make every value distinguishable.

    Returns:
        A ``measured_data`` mapping in the shapes ``save_datapoint()`` expects
        for a ``(2, 1)`` loop shape.
    """
    return {
        "field_T": float(index),
        "voltage_V": [[10.0 + index], [20.0 + index]],
        "voltage_V_array": [
            [[10.0 + index] * N_SAMPLES],
            [[20.0 + index] * N_SAMPLES],
        ],
        "raw_channels_block": [
            [[[1.0 * index] * BLOCK_COLS] * BLOCK_ROWS],
            [[[2.0 * index] * BLOCK_COLS] * BLOCK_ROWS],
        ],
    }


def _writer(tmp_path: Path, n_sweep_points: int = N_POINTS) -> DataManager:
    """Return a ``DataManager`` writing the fixture layout into *tmp_path*.

    Args:
        tmp_path: The directory to write into.
        n_sweep_points: How many points to pre-allocate.

    Returns:
        The open ``DataManager``; the caller closes it.
    """
    return DataManager(
        data_directory=str(tmp_path),
        procedure_name="FieldSweep",
        procedure_params=PROCEDURE_PARAMS,
        sample_info=SAMPLE_INFO,
        instrument_state={"magnet_z": {"field": 0.0}},
        system_targets={"magnet_z": {"target": -1.0}},
        measurement_commands=[{"vi_name": "dc", "method": "measure", "kwargs": {}}],
        data_config=DATA_CONFIG,
        n_sweep_points=n_sweep_points,
        experiment_info=EXPERIMENT_INFO,
    )


@pytest.fixture
def complete_run(tmp_path):
    """Path to a closed file with every allocated point written."""
    writer = _writer(tmp_path)
    for index in range(N_POINTS):
        writer.save_datapoint(index, _measured(index), {"magnet_z": {"field": index}})
    writer.close()
    return writer.filepath


@pytest.fixture
def run(complete_run):
    """An open ``RunHandle`` on the completed run."""
    with dr.open_run(complete_run) as handle:
        yield handle


def _by_name(handle) -> dict[str, dr.ColumnInfo]:
    """Return a source's columns keyed by name.

    Args:
        handle: Any ``RunSource``.

    Returns:
        ``{column name: ColumnInfo}``.
    """
    return {info.name: info for info in handle.list_columns()}


# ── Opening ───────────────────────────────────────────────────────────────


def test_open_run_is_a_context_manager(complete_run):
    """The handle closes on leaving the with-block, and reads refuse after."""
    with dr.open_run(complete_run) as handle:
        assert handle.is_open
    assert not handle.is_open
    with pytest.raises(ValueError, match="closed"):
        handle.list_columns()


def test_close_is_idempotent(complete_run):
    """Closing twice is not an error."""
    handle = dr.open_run(complete_run)
    handle.close()
    handle.close()
    assert not handle.is_open


def test_open_run_reports_its_read_mode(run):
    """The handle records which read strategy the file allowed."""
    assert run.mode in ("swmr", "read", "unlocked")


def test_open_run_missing_file(tmp_path):
    """A path that does not exist is a FileNotFoundError, not an OSError."""
    with pytest.raises(FileNotFoundError):
        dr.open_run(tmp_path / "nothing.h5")


def test_open_run_rejects_a_foreign_hdf5_file(tmp_path):
    """An HDF5 file without /data and /metadata is not a I2AS run."""
    path = tmp_path / "foreign.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("something", data=np.arange(3))
    with pytest.raises(DataSchemaError, match="not a I2AS run file"):
        dr.open_run(path)


# ── Columns ───────────────────────────────────────────────────────────────


def test_list_columns_covers_every_written_dataset(run):
    """Every dataset the writer allocated is reported, plus the loop axis."""
    assert [info.name for info in run.list_columns()] == [
        "field_T",
        "loop1",
        "raw_channels_block",
        "timestamp",
        "voltage_V",
        "voltage_V_array",
    ]


def test_list_columns_reports_units_from_the_name_convention(run):
    """The SI unit comes off the column name's last token."""
    columns = _by_name(run)
    assert columns["field_T"].unit == "T"
    assert columns["voltage_V"].unit == "V"
    assert columns["voltage_V_array"].unit == "V"
    assert columns["raw_channels_block"].unit == ""
    assert columns["timestamp"].unit == ""


def test_list_columns_reports_roles_from_the_data_config(run):
    """Each column's role is the one the writer's data_config declared."""
    columns = _by_name(run)
    assert columns["field_T"].role == dr.ROLE_SWEEP_AXIS
    assert columns["timestamp"].role == dr.ROLE_SWEEP_AXIS
    assert columns["voltage_V"].role == dr.ROLE_MEASUREMENT
    assert columns["voltage_V_array"].role == dr.ROLE_MEASUREMENT
    assert columns["raw_channels_block"].role == dr.ROLE_RAW_BLOCK
    assert columns["loop1"].role == dr.ROLE_LOOP_AXIS
    assert {info.role for info in run.list_columns()} <= set(dr.COLUMN_ROLES)


def test_list_columns_reports_the_hdf5_layout_shapes(run):
    """Sweep, scalar, array and block shapes are the layout the writer uses."""
    columns = _by_name(run)
    assert columns["field_T"].shape == (N_POINTS,)
    assert columns["voltage_V"].shape == (N_POINTS, 2, 1)
    assert columns["voltage_V_array"].shape == (N_POINTS, 2, 1, N_SAMPLES)
    assert columns["raw_channels_block"].shape == (
        N_POINTS,
        2,
        1,
        BLOCK_ROWS,
        BLOCK_COLS,
    )


def test_list_columns_reports_dtypes_and_paths(run):
    """Numeric columns are float64, the timestamp is a string column."""
    columns = _by_name(run)
    assert columns["field_T"].dtype == "float64"
    assert columns["timestamp"].dtype == "str"
    assert columns["field_T"].path == "/data/field_T"
    assert columns["loop1"].path == "/metadata/procedure_params/loop1_values"


def test_loop_axis_column_carries_the_slot_values(run):
    """The reading loop's axis is surfaced from metadata, not from a dataset."""
    columns = _by_name(run)
    assert columns["loop1"].length == 2
    assert columns["loop1"].dtype == "float64"
    np.testing.assert_allclose(run.read_slice("loop1"), [1e-6, -1e-6])
    assert "loop2" not in columns


def test_a_textual_loop_axis_is_a_string_column(tmp_path):
    """A switch-route slot is a str loop axis; a numeric one is float64."""
    params = {**PROCEDURE_PARAMS, "loop2_values": ["Mux-Ch1", "Mux-Ch2"]}
    infos = {info.name: info for info in dr.loop_axis_column_infos(params)}
    assert infos["loop1"].dtype == "float64"
    assert infos["loop2"].dtype == "str"
    assert [info.name for info in dr.loop_axis_column_infos(params)] == [
        "loop1",
        "loop2",
    ]
    assert dr.loop_axis_column_infos({}) == ()


def test_column_unit_only_accepts_known_unit_tokens():
    """A name ending in a word, not a unit token, declares no unit."""
    assert dr.column_unit("field_T") == "T"
    assert dr.column_unit("res_a_dc_ohm") == "ohm"
    assert dr.column_unit("voltage_V_error") == "V"
    assert dr.column_unit("magnet_z_field") == ""
    assert dr.column_unit("timestamp") == ""


# ── Slices ────────────────────────────────────────────────────────────────


def test_read_slice_returns_the_whole_column_by_default(run):
    """No bounds means every written point."""
    np.testing.assert_allclose(run.read_slice("field_T"), [0.0, 1.0, 2.0, 3.0])


def test_read_slice_honours_start_stop_step(run):
    """Bounds select along the sweep-point axis."""
    np.testing.assert_allclose(run.read_slice("field_T", 1, 3), [1.0, 2.0])
    np.testing.assert_allclose(run.read_slice("field_T", None, None, 2), [0.0, 2.0])
    np.testing.assert_allclose(run.read_slice("field_T", -2), [2.0, 3.0])


def test_read_slice_clamps_a_stop_past_the_end(run):
    """A stop beyond the written points is clamped, never an error."""
    np.testing.assert_allclose(run.read_slice("field_T", 2, 99), [2.0, 3.0])
    assert run.read_slice("field_T", 99, 100).size == 0


def test_read_slice_refuses_a_reverse_read(run):
    """A negative step is refused rather than silently reordered."""
    with pytest.raises(ValueError, match="step must be positive"):
        run.read_slice("field_T", None, None, -1)


def test_read_slice_keeps_every_inner_axis_whole(run):
    """Only the leading axis is sliced; loop, sample and block axes are whole."""
    block = run.read_slice("raw_channels_block", 1, 3)
    assert block.shape == (2, 2, 1, BLOCK_ROWS, BLOCK_COLS)
    np.testing.assert_allclose(block[0, 0], np.full((1, BLOCK_ROWS, BLOCK_COLS), 1.0))
    np.testing.assert_allclose(block[0, 1], np.full((1, BLOCK_ROWS, BLOCK_COLS), 2.0))
    array = run.read_slice("voltage_V_array", 0, 1)
    assert array.shape == (1, 2, 1, N_SAMPLES)


def test_read_slice_decodes_the_string_column(run):
    """The timestamp column comes back as text, one ISO stamp per point."""
    stamps = run.read_slice("timestamp")
    assert stamps.shape == (N_POINTS,)
    assert all(stamp.startswith("20") and stamp.endswith("+00:00") for stamp in stamps)


def test_read_slice_unknown_column(run):
    """An unknown column names the ones that exist."""
    with pytest.raises(KeyError, match="voltage_mV"):
        run.read_slice("voltage_mV")


# ── Statistics ────────────────────────────────────────────────────────────


def test_summary_stats_over_a_looped_measurement(run):
    """Statistics flatten every axis of the written prefix."""
    stats = run.summary_stats("voltage_V")
    values = [10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0]
    assert stats.count == len(values)
    assert stats.min == min(values)
    assert stats.max == max(values)
    assert stats.mean == pytest.approx(np.mean(values))
    assert stats.std == pytest.approx(np.std(values))
    assert stats.first == 10.0
    assert stats.last == 23.0


def test_summary_stats_of_the_sweep_axis(run):
    """A 1-D sweep column summarises like any other numeric column."""
    stats = run.summary_stats("field_T")
    assert (stats.count, stats.min, stats.max, stats.first, stats.last) == (
        N_POINTS,
        0.0,
        3.0,
        0.0,
        3.0,
    )


def test_summary_stats_excludes_a_failed_reading(tmp_path):
    """A NaN reading is excluded from count and never poisons the mean."""
    writer = _writer(tmp_path, n_sweep_points=2)
    writer.save_datapoint(0, _measured(0), {})
    broken = _measured(1)
    broken["voltage_V"] = [[float("nan")], [21.0]]
    writer.save_datapoint(1, broken, {})
    writer.close()
    with dr.open_run(writer.filepath) as handle:
        stats = handle.summary_stats("voltage_V")
    assert stats.count == 3
    assert stats.mean == pytest.approx(np.mean([10.0, 20.0, 21.0]))
    assert stats.last == 21.0


def test_summary_stats_of_an_all_nan_column(tmp_path):
    """A column with nothing finite reports count 0 and NaN everywhere."""
    writer = _writer(tmp_path, n_sweep_points=1)
    point = _measured(0)
    point["voltage_V"] = [[float("nan")], [float("nan")]]
    writer.save_datapoint(0, point, {})
    writer.close()
    with dr.open_run(writer.filepath) as handle:
        stats = handle.summary_stats("voltage_V")
    assert stats.count == 0
    assert all(
        math.isnan(getattr(stats, name))
        for name in ("min", "max", "mean", "std", "first", "last")
    )


def test_summary_stats_refuses_a_string_column(run):
    """The timestamp column has no statistics."""
    with pytest.raises(ValueError, match="not numeric"):
        run.summary_stats("timestamp")


# ── The written prefix ────────────────────────────────────────────────────


def test_written_prefix_of_an_aborted_run(tmp_path):
    """A run closed early is trimmed, and every read stops at the last point."""
    writer = _writer(tmp_path)
    for index in range(2):
        writer.save_datapoint(index, _measured(index), {})
    writer.close()
    with dr.open_run(writer.filepath) as handle:
        assert handle.n_points == 2
        assert _by_name(handle)["field_T"].shape == (2,)
        np.testing.assert_allclose(handle.read_slice("field_T"), [0.0, 1.0])
        assert handle.summary_stats("field_T").count == 2


def test_written_prefix_of_an_untrimmed_file(tmp_path):
    """An abandoned file keeps its allocation; the point counter still holds.

    The writer trims only in ``close()``, so a file whose process died mid-run
    is the case the ``timestamp`` point counter exists for: allocated shape 4,
    two points of real data, and no NaN tail in any read.
    """
    writer = _writer(tmp_path)
    for index in range(2):
        writer.save_datapoint(index, _measured(index), {})
    path = writer.filepath
    writer._file.close()  # the process died: no close(), no trim
    with dr.open_run(path) as handle:
        assert handle.n_points == 2
        column = _by_name(handle)["field_T"]
        assert (column.shape, column.length) == ((N_POINTS,), 2)
        np.testing.assert_allclose(handle.read_slice("field_T"), [0.0, 1.0])
        assert handle.summary_stats("voltage_V").count == 4
        assert handle.read_slice("timestamp").shape == (2,)


def test_a_run_with_no_points_at_all(tmp_path):
    """A run that saved nothing reads as an empty run, not as NaN rows."""
    writer = _writer(tmp_path)
    writer.close()
    with dr.open_run(writer.filepath) as handle:
        assert handle.n_points == 0
        assert handle.read_slice("field_T").size == 0
        assert handle.summary_stats("field_T").count == 0


def test_a_file_without_a_timestamp_column_falls_back_to_the_allocation(tmp_path):
    """Without the writer's point counter, the allocated length is the length.

    Also the role fallbacks: a block that only describes itself through its
    ``axes`` attribute is still a raw block, and a 1-D column is a sweep axis.
    """
    path = tmp_path / "counterless.h5"
    with h5py.File(path, "w") as file:
        file.create_group("metadata").attrs["procedure_name"] = "Handmade"
        data = file.create_group("data")
        data.create_dataset("field_T", data=np.arange(3, dtype=np.float64))
        block = data.create_dataset("odd_block", data=np.zeros((3, 2, 2)))
        block.attrs["axes"] = "sweep_point, row, channel"
        data.create_dataset("odd_grid", data=np.zeros((3, 2, 2)))
    with dr.open_run(path) as handle:
        assert handle.n_points == 3
        columns = _by_name(handle)
        assert columns["field_T"].role == dr.ROLE_SWEEP_AXIS
        assert columns["odd_block"].role == dr.ROLE_RAW_BLOCK
        assert columns["odd_grid"].role == dr.ROLE_MEASUREMENT


def test_reading_a_file_the_writer_still_holds_open(tmp_path):
    """Points flushed by the writer are visible to a reader mid-run.

    ``DataManager`` flushes after every datapoint, so a handle opened
    mid-run sees whole datapoints and nothing half-written; a handle opened
    before a point is saved sees it after refreshing. Verified here inside the
    writing process (a reader in another process falls back to the lock-free
    open, which the module docstring describes).
    """
    writer = _writer(tmp_path)
    for index in range(2):
        writer.save_datapoint(index, _measured(index), {})
    try:
        with dr.open_run(writer.filepath) as handle:
            assert handle.n_points == 2
            np.testing.assert_allclose(handle.read_slice("field_T"), [0.0, 1.0])
            writer.save_datapoint(2, _measured(2), {})
            assert handle.n_points == 3
            np.testing.assert_allclose(
                handle.read_slice("field_T"), [0.0, 1.0, 2.0]
            )
            assert handle.summary_stats("field_T").count == 3
    finally:
        writer.close()


# ── Metadata ──────────────────────────────────────────────────────────────


def test_read_metadata_answers_every_canonical_key(run):
    """The canonical key set is exactly RUN_METADATA_KEYS, always complete."""
    metadata = run.read_metadata()
    assert set(metadata) == set(dr.RUN_METADATA_KEYS)
    assert metadata["source"] == "file"
    assert metadata["procedure"] == "FieldSweep"
    assert metadata["params"] == PROCEDURE_PARAMS
    assert metadata["sample"] == SAMPLE_INFO
    assert metadata["setup"] == EXPERIMENT_INFO["setup"]
    assert metadata["experiment"] == EXPERIMENT_INFO["experiment"]
    assert metadata["start_time"].endswith("+00:00")
    assert metadata["end_time"].endswith("+00:00")
    assert metadata["data_file"] == str(run.path)


def test_read_metadata_leaves_the_engines_keys_empty(run):
    """A file does not carry the run manifest, and says so with empties.

    ``run_kind`` is the one exception: the writer records it on the file
    (``/metadata.run_kind``), so a **probe run**'s file identifies itself as
    a probe to whoever opens it later.
    """
    metadata = run.read_metadata()
    assert (metadata["run_id"], metadata["status"], metadata["reason"]) == ("", "", "")
    assert metadata["run_kind"] == "run"


def test_read_metadata_raw_carries_the_uncanonical_attributes(run):
    """Everything the writer stored is reachable, JSON blobs already decoded."""
    raw = run.read_metadata()["raw"]
    assert raw["system_targets"] == {"magnet_z": {"target": -1.0}}
    assert raw["measurement_commands"][0]["vi_name"] == "dc"
    assert raw["data_config"]["measurement_blocks"] == {
        "raw_channels_block": [BLOCK_ROWS, BLOCK_COLS]
    }
    assert raw["instrument_state"] == {"magnet_z": {"field": 0.0}}


# ── JSON round trips ──────────────────────────────────────────────────────


def test_column_info_round_trips_through_json(run):
    """A column declaration survives json.dumps/loads unchanged."""
    for info in run.list_columns():
        payload = json.loads(json.dumps(info.to_json()))
        assert dr.ColumnInfo.from_json(payload) == info


def test_stats_round_trip_through_json(run):
    """Statistics survive json.dumps/loads unchanged."""
    stats = run.summary_stats("voltage_V")
    assert dr.Stats.from_json(json.loads(json.dumps(stats.to_json()))) == stats


def test_stats_json_renders_nan_as_null():
    """Strict JSON has no NaN, so an absent statistic is null and back."""
    empty = dr.summarise_values("nothing", [float("nan")])
    payload = empty.to_json()
    assert payload["mean"] is None
    assert json.dumps(payload) == json.dumps(json.loads(json.dumps(payload)))
    rebuilt = dr.Stats.from_json(payload)
    assert rebuilt.count == 0
    assert math.isnan(rebuilt.mean)


def test_json_payloads_ignore_unknown_keys(run):
    """A newer producer's extra key never breaks an older consumer."""
    info = run.list_columns()[0]
    assert dr.ColumnInfo.from_json({**info.to_json(), "future": 1}) == info
    stats = run.summary_stats("field_T")
    assert dr.Stats.from_json({**stats.to_json(), "future": 1}) == stats


# ── The run-source vocabulary ─────────────────────────────────────────────


def test_run_handle_is_a_run_source(run):
    """A file handle satisfies the protocol the module's functions take."""
    assert isinstance(run, dr.RunSource)


def test_module_functions_delegate_to_any_run_source(run):
    """The module-level vocabulary and the handle's methods are one answer."""
    assert dr.list_columns(run) == run.list_columns()
    np.testing.assert_allclose(
        dr.read_slice(run, "field_T", 1, 3), run.read_slice("field_T", 1, 3)
    )
    assert dr.summary_stats(run, "field_T") == run.summary_stats("field_T")
    assert dr.read_metadata(run) == run.read_metadata()


def test_data_reader_imports_only_events_and_exceptions():
    """The standalone rule, read off the module's own imports.

    Import-linter contract C17 freezes the same boundary against a list of
    modules; this test needs no list, so a `core` module added tomorrow is
    covered the moment someone imports it here.
    """
    source = Path(dr.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert {name for name in imported if name.startswith("i2as")} <= {
        "i2as.core.events",
        "i2as.core.exceptions",
    }


# ── Image blocks (the image-block standard) ───────────────────────────────────

IMAGE_H, IMAGE_W = 3, 4


def _image_run(tmp_path: Path, loop_shape: list[int], n_points: int = 2) -> Path:
    """Write a closed run holding one image block beside a scalar, return its path."""
    config = {
        "sweep_columns": {"field_T": "float"},
        "measurement_scalars": {"intensity_counts": "float"},
        "measurement_arrays": {},
        "measurement_blocks": {"frame": (IMAGE_H, IMAGE_W)},
        "measurement_image_blocks": {"frame": {"unit": "counts", "description": "sim frame"}},
        "loop_shape": loop_shape,
    }
    n1, n2 = loop_shape
    writer = DataManager(
        data_directory=str(tmp_path),
        procedure_name="FieldImaging",
        procedure_params={"field_start": 0.0, "loop1_values": [1e-6, -1e-6] if n1 == 2 else []},
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands=[],
        data_config=config,
        n_sweep_points=n_points,
    )
    for index in range(n_points):
        frames = np.array(
            [[np.full((IMAGE_H, IMAGE_W), 10.0 * index + i1 + 0.1 * i2) for i2 in range(n2)] for i1 in range(n1)]
        )
        value = frames if (n1, n2) != (1, 1) else frames[0, 0]
        writer.save_datapoint(
            index,
            {
                "field_T": float(index),
                "intensity_counts": [[float(index)] * n2 for _ in range(n1)],
                "frame": value,
            },
            {},
        )
    writer.close()
    return writer.filepath


def test_image_block_is_listed_with_the_image_role(tmp_path):
    with dr.open_run(_image_run(tmp_path, [1, 1])) as handle:
        info = _by_name(handle)["frame"]
    assert info.role == dr.ROLE_IMAGE
    assert info.shape == (2, IMAGE_H, IMAGE_W)
    assert dr.ROLE_IMAGE in dr.COLUMN_ROLES


def test_image_role_falls_back_to_the_block_kind_attribute(tmp_path):
    """Strip the declaration: the dataset's own block_kind still names it an image."""
    path = _image_run(tmp_path, [1, 1])
    import json

    import h5py

    with h5py.File(path, "r+") as f:
        config = json.loads(f["metadata"].attrs["data_config"])
        config.pop("measurement_image_blocks")
        config.pop("measurement_blocks")
        f["metadata"].attrs["data_config"] = json.dumps(config)
    with dr.open_run(path) as handle:
        assert _by_name(handle)["frame"].role == dr.ROLE_IMAGE


def test_read_image_without_a_loop(tmp_path):
    with dr.open_run(_image_run(tmp_path, [1, 1])) as handle:
        frame = handle.read_image("frame", 1)
        assert frame.shape == (IMAGE_H, IMAGE_W)
        np.testing.assert_allclose(frame, 10.0)
        with pytest.raises(IndexError):
            handle.read_image("frame", 1, loop1=1)


def test_read_image_with_a_loop_selects_the_slot(tmp_path):
    with dr.open_run(_image_run(tmp_path, [2, 1])) as handle:
        np.testing.assert_allclose(handle.read_image("frame", 1, loop1=0), 10.0)
        np.testing.assert_allclose(handle.read_image("frame", 1, loop1=1), 11.0)
        assert handle.read_image("frame", 0).shape == (IMAGE_H, IMAGE_W)


def test_read_image_refuses_a_non_image_column_and_an_unwritten_point(tmp_path):
    with dr.open_run(_image_run(tmp_path, [1, 1])) as handle:
        with pytest.raises(ValueError, match="not an image block"):
            handle.read_image("intensity_counts", 0)
        with pytest.raises(IndexError):
            handle.read_image("frame", 2)
        with pytest.raises(KeyError):
            handle.read_image("no_such_frame", 0)


def test_select_frame_shapes():
    frame = np.zeros((IMAGE_H, IMAGE_W))
    assert dr.select_frame(frame).shape == (IMAGE_H, IMAGE_W)
    looped = np.zeros((2, 1, IMAGE_H, IMAGE_W))
    assert dr.select_frame(looped, 1, 0).shape == (IMAGE_H, IMAGE_W)
    with pytest.raises(ValueError):
        dr.select_frame(np.zeros((IMAGE_H,)))
