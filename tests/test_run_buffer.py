"""The run in flight: RunBuffer, and its equivalence with the file on disk.

The load-bearing test here is the last one: a real sim procedure writes a run
through the real ``DataManager`` while the same values are fed to a
``RunBuffer``, and the two sources are then required to answer the run-source
vocabulary identically. Everything above it is the buffer's own behaviour —
what it does with a gap, a late column, a short reading, an event from
another run.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from i2as.core import data_reader as dr
from i2as.core.events import Datapoint, RunFinished, RunStarted
from i2as.core.exceptions import DataSchemaError
from i2as.core.run_buffer import RunBuffer
from i2as.core.station import build_station
from i2as.procedures.field_sweep import FieldSweep

CONFIG_PATH = "i2as/configs/sim_cryostat"

RUN_ID = "20260903_120000_001_fieldsweep"

DATA_CONFIG = {
    "sweep_columns": {"field_T": "float"},
    "measurement_scalars": {"voltage_V": "float"},
    "measurement_arrays": {"voltage_V_array": 3},
    "measurement_blocks": {"raw_channels_block": [2, 3]},
    "loop_shape": [2, 1],
}

MANIFEST = {
    "run_id": RUN_ID,
    "procedure": "Field Sweep",
    "kind": "run",
    "params": {"field_start": -1.0, "loop1_values": [1e-6, -1e-6]},
    "data_file": "/data/runs/fieldsweep.h5",
    "started_utc": "2026-09-03T12:00:00+00:00",
    "data_config": DATA_CONFIG,
}


def _point(index: int, **overrides) -> Datapoint:
    """Return one datapoint in the fixture layout.

    Args:
        index: The sweep index.
        **overrides: Column values replacing the defaults.

    Returns:
        The ``Datapoint``, its ``values`` shaped as
        ``DataManager.save_datapoint()`` expects them.
    """
    values = {
        "field_T": float(index),
        "voltage_V": [[10.0 + index], [20.0 + index]],
        "voltage_V_array": [[[10.0 + index] * 3], [[20.0 + index] * 3]],
        "raw_channels_block": [[[[1.0 * index] * 3] * 2], [[[2.0 * index] * 3] * 2]],
    }
    values.update(overrides)
    return Datapoint(run_id=RUN_ID, index=index, values=values, ts=1788000000.0 + index)


@pytest.fixture
def buffer():
    """A buffer holding a started run with three points."""
    buffer = RunBuffer()
    buffer.start(RunStarted(run_id=RUN_ID, manifest=MANIFEST))
    for index in range(3):
        buffer.append(_point(index))
    return buffer


def _by_name(source) -> dict[str, dr.ColumnInfo]:
    """Return a source's columns keyed by name.

    Args:
        source: Any ``RunSource``.

    Returns:
        ``{column name: ColumnInfo}``.
    """
    return {info.name: info for info in source.list_columns()}


# ── The vocabulary ────────────────────────────────────────────────────────


def test_buffer_is_a_run_source(buffer):
    """A buffer satisfies the same protocol a file handle does."""
    assert isinstance(buffer, dr.RunSource)


def test_buffer_reports_the_writers_shapes(buffer):
    """Each column is stored with the leading-axis-is-the-sweep-point layout."""
    columns = _by_name(buffer)
    assert buffer.n_points == 3
    assert columns["field_T"].shape == (3,)
    assert columns["voltage_V"].shape == (3, 2, 1)
    assert columns["voltage_V_array"].shape == (3, 2, 1, 3)
    assert columns["raw_channels_block"].shape == (3, 2, 1, 2, 3)
    assert all(
        info.length == 3
        for info in buffer.list_columns()
        if info.role != dr.ROLE_LOOP_AXIS
    )


def test_buffer_takes_roles_from_the_manifests_data_config(buffer):
    """A declared schema tells a block from a measurement grid."""
    columns = _by_name(buffer)
    assert columns["field_T"].role == dr.ROLE_SWEEP_AXIS
    assert columns["voltage_V"].role == dr.ROLE_MEASUREMENT
    assert columns["raw_channels_block"].role == dr.ROLE_RAW_BLOCK
    assert columns["timestamp"].role == dr.ROLE_SWEEP_AXIS
    assert columns["loop1"].role == dr.ROLE_LOOP_AXIS


def test_buffer_infers_roles_without_a_declared_schema():
    """Undeclared, a scalar is a sweep axis and everything else a measurement.

    The documented cost of an undeclared schema: a raw block reads back as a
    measurement, the one thing a buffer reports less precisely than a file.
    """
    buffer = RunBuffer()
    buffer.start(RunStarted(run_id=RUN_ID, manifest={"run_id": RUN_ID}))
    buffer.append(_point(0))
    columns = _by_name(buffer)
    assert columns["field_T"].role == dr.ROLE_SWEEP_AXIS
    assert columns["voltage_V"].role == dr.ROLE_MEASUREMENT
    assert columns["raw_channels_block"].role == dr.ROLE_MEASUREMENT


def test_buffer_adds_the_timestamp_column_the_writer_stamps(buffer):
    """The writer stamps the time itself, so the buffer does too."""
    stamps = buffer.read_slice("timestamp")
    assert stamps.shape == (3,)
    assert all(stamp.endswith("+00:00") for stamp in stamps)
    assert _by_name(buffer)["timestamp"].dtype == "str"


def test_buffer_surfaces_the_loop_axis_from_the_manifest(buffer):
    """The reading loop's axis comes from params, exactly as it does in a file."""
    np.testing.assert_allclose(buffer.read_slice("loop1"), [1e-6, -1e-6])
    assert _by_name(buffer)["loop1"].path == "/metadata/procedure_params/loop1_values"
    assert "loop2" not in _by_name(buffer)


def test_buffer_read_slice_bounds(buffer):
    """Bounds resolve against the points received, and clamp past the end."""
    np.testing.assert_allclose(buffer.read_slice("field_T"), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(buffer.read_slice("field_T", 1, 3), [1.0, 2.0])
    np.testing.assert_allclose(buffer.read_slice("field_T", None, None, 2), [0.0, 2.0])
    np.testing.assert_allclose(buffer.read_slice("field_T", 1, 99), [1.0, 2.0])
    assert buffer.read_slice("field_T", 9, 10).size == 0
    with pytest.raises(ValueError, match="step must be positive"):
        buffer.read_slice("field_T", None, None, -1)


def test_buffer_read_slice_keeps_inner_axes_whole(buffer):
    """Only the leading axis is sliced."""
    assert buffer.read_slice("voltage_V_array", 0, 2).shape == (2, 2, 1, 3)
    assert buffer.read_slice("raw_channels_block", 0, 1).shape == (1, 2, 1, 2, 3)


def test_buffer_read_slice_of_an_empty_run_keeps_the_column_shape():
    """A column with no points still reads back with its own inner shape."""
    buffer = RunBuffer()
    buffer.start(RunStarted(run_id=RUN_ID, manifest=MANIFEST))
    buffer.append(_point(0))
    assert buffer.read_slice("voltage_V", 5, 6).shape == (0, 2, 1)
    assert buffer.read_slice("timestamp", 5, 6).size == 0


def test_buffer_unknown_column(buffer):
    """An unknown column names the ones that exist."""
    with pytest.raises(KeyError, match="voltage_mV"):
        buffer.read_slice("voltage_mV")


def test_buffer_summary_stats(buffer):
    """Statistics come from the same NaN-aware implementation a file uses."""
    stats = buffer.summary_stats("voltage_V")
    assert stats == dr.summarise_values("voltage_V", buffer.read_slice("voltage_V"))
    assert (stats.count, stats.min, stats.max) == (6, 10.0, 22.0)


def test_buffer_summary_stats_skips_a_failed_reading(buffer):
    """A NaN reading is excluded, exactly as it is on disk."""
    buffer.append(_point(3, voltage_V=[[float("nan")], [23.0]]))
    stats = buffer.summary_stats("voltage_V")
    assert stats.count == 7
    assert stats.last == 23.0


def test_buffer_summary_stats_refuses_a_text_column(buffer):
    """The timestamp column has no statistics."""
    with pytest.raises(ValueError, match="not numeric"):
        buffer.summary_stats("timestamp")


# ── Feeding it ────────────────────────────────────────────────────────────


def test_append_before_start_is_refused():
    """A datapoint with no run to belong to is an error, not a silent drop."""
    with pytest.raises(DataSchemaError, match="before start"):
        RunBuffer().append(_point(0))


def test_append_ignores_another_runs_datapoint(buffer, caplog):
    """A point from a run this buffer is not following is dropped, with a warning."""
    other = Datapoint(run_id="someone_else", index=0, values={"field_T": 99.0})
    with caplog.at_level("WARNING"):
        buffer.append(other)
    assert buffer.n_points == 3
    assert "someone_else" in caplog.text


def test_finish_records_how_the_run_ended(buffer):
    """The buffered data stays readable; the metadata gains the ending."""
    buffer.finish(
        RunFinished(
            run_id=RUN_ID,
            status="aborted",
            reason="operator stop",
            manifest={"finished_utc": "2026-09-03T12:05:00+00:00"},
        )
    )
    metadata = buffer.read_metadata()
    assert not buffer.is_running
    assert (metadata["status"], metadata["reason"]) == ("aborted", "operator stop")
    assert metadata["end_time"] == "2026-09-03T12:05:00+00:00"
    assert buffer.n_points == 3


def test_finish_ignores_another_runs_event(buffer):
    """A run_finished for a different run leaves this one running."""
    buffer.finish(RunFinished(run_id="someone_else", status="done"))
    assert buffer.is_running
    assert buffer.read_metadata()["status"] == ""


def test_start_clears_the_previous_run(buffer):
    """One buffer follows a session's runs one after another."""
    buffer.start(RunStarted(run_id="second_run", manifest={"run_id": "second_run"}))
    assert (buffer.n_points, buffer.run_id) == (0, "second_run")
    assert buffer.list_columns() == ()


def test_a_gap_in_the_indices_reads_as_nan(buffer):
    """A point placed past the end leaves NaN behind it, never a shift."""
    buffer.append(_point(4))
    assert buffer.n_points == 5
    field = buffer.read_slice("field_T")
    assert field[3] != field[3]  # NaN
    np.testing.assert_allclose(field[[0, 1, 2, 4]], [0.0, 1.0, 2.0, 4.0])
    assert buffer.summary_stats("field_T").count == 4


def test_points_may_arrive_out_of_order(buffer):
    """A point is placed at its own index, whenever it arrives."""
    buffer.append(_point(5))
    buffer.append(_point(4))
    np.testing.assert_allclose(buffer.read_slice("field_T", 3), [float("nan"), 4.0, 5.0])


def test_a_column_that_appears_late_is_backfilled(buffer):
    """A column first seen at point 3 reads as NaN for points 0 to 2."""
    buffer.append(_point(3, extra_K=4.2))
    extra = buffer.read_slice("extra_K")
    assert extra.shape == (4,)
    assert all(math.isnan(value) for value in extra[:3])
    assert extra[3] == 4.2
    assert _by_name(buffer)["extra_K"].unit == "K"


def test_a_short_inner_axis_is_padded_like_the_writer_pads_it(buffer, caplog):
    """An acquisition that returned fewer samples pads with NaN, never shifts."""
    with caplog.at_level("WARNING"):
        buffer.append(_point(3, voltage_V_array=[[[1.0, 2.0]], [[3.0, 4.0]]]))
    point = buffer.read_slice("voltage_V_array", 3)
    assert point.shape == (1, 2, 1, 3)
    np.testing.assert_allclose(point[0, 0, 0, :2], [1.0, 2.0])
    assert math.isnan(point[0, 0, 0, 2])
    assert "padding/truncating" in caplog.text


def test_a_loop_axis_mismatch_is_refused(buffer):
    """A value that does not belong to the column is an error, not a pad."""
    with pytest.raises(DataSchemaError, match="voltage_V"):
        buffer.append(_point(3, voltage_V=[[1.0], [2.0], [3.0]]))


def test_a_non_numeric_value_is_refused(buffer):
    """A numeric column cannot take text."""
    with pytest.raises(DataSchemaError, match="numeric"):
        buffer.append(_point(3, field_T="warm"))


def test_a_negative_index_is_refused(buffer):
    """There is no point before the first one."""
    with pytest.raises(DataSchemaError, match="negative"):
        buffer.append(Datapoint(run_id=RUN_ID, index=-1, values={"field_T": 0.0}))


# ── Metadata and JSON ─────────────────────────────────────────────────────


def test_buffer_metadata_answers_every_canonical_key(buffer):
    """The same key set a file answers, filled from the run manifest."""
    metadata = buffer.read_metadata()
    assert set(metadata) == set(dr.RUN_METADATA_KEYS)
    assert metadata["source"] == "buffer"
    assert metadata["run_id"] == RUN_ID
    assert metadata["run_kind"] == "run"
    assert metadata["procedure"] == "Field Sweep"
    assert metadata["params"] == MANIFEST["params"]
    assert metadata["data_file"] == MANIFEST["data_file"]
    assert metadata["start_time"] == MANIFEST["started_utc"]
    assert metadata["status"] == ""
    assert metadata["raw"] == MANIFEST


def test_buffer_results_round_trip_through_json(buffer):
    """A buffered result travels as JSON exactly as a file's does."""
    for info in buffer.list_columns():
        assert dr.ColumnInfo.from_json(json.loads(json.dumps(info.to_json()))) == info
    stats = buffer.summary_stats("field_T")
    assert dr.Stats.from_json(json.loads(json.dumps(stats.to_json()))) == stats


# ── Equivalence with the file the writer wrote ────────────────────────────


@pytest.fixture(scope="module")
def sim_run(tmp_path_factory):
    """Run a sim FieldSweep, buffering every point the writer is given.

    Drives the real procedure against the sim station, so the values compared
    below are a real run's, in the real shapes, and the buffer is fed exactly
    what ``DataManager.save_datapoint()`` was fed — ``last_datapoint`` is that
    mapping, and the sweep index is the point's ordinal, which is the mapping
    ``Datapoint`` carries once the engine emits it.

    Returns:
        ``(file path, RunBuffer)`` for the same completed run.
    """
    tmp_path = tmp_path_factory.mktemp("sim_run")
    station = build_station(CONFIG_PATH)
    procedure = FieldSweep(
        station=station,
        sample_info={"sample_name": "Sim Sample", "sample_id": "SIM-1"},
        data_directory=str(tmp_path),
        field_start=-0.1,
        field_end=0.1,
        field_steps=3,
        temperature=300.0,
        init_wait=0.0,
        step_wait=0.0,
        measurement_vi="dc_measurement",
        current_A=1e-6,
        compliance_A=1e-3,
        voltmeter_range_V=0.1,
        readings_per_point=3,
    )
    procedure.initiate()
    station.get_vi("dc_measurement").initiate_measurement(
        **procedure._measurement_params
    )
    writer = procedure._data_manager
    run_id = "sim_run_001"
    buffer = RunBuffer()
    buffer.start(
        RunStarted(
            run_id=run_id,
            manifest={
                "run_id": run_id,
                "procedure": procedure.name,
                "kind": "run",
                "params": procedure.get_params(),
                "data_file": str(writer.filepath),
                # What the engine's run manifest carries about the layout: the
                # very dict the writer was built with.
                "data_config": writer._data_config,
            },
        )
    )
    index = 0
    while True:
        procedure.measure()
        buffer.append(
            Datapoint(run_id=run_id, index=index, values=writer.last_datapoint)
        )
        index += 1
        if procedure.change_sweep_step() is None:
            break
    path = writer.filepath
    procedure.standby()
    return path, buffer


def test_buffer_and_file_list_the_same_columns(sim_run):
    """Name, unit, dtype, role, shape, length and path all agree."""
    path, buffer = sim_run
    with dr.open_run(path) as handle:
        assert handle.n_points == buffer.n_points == 3
        assert handle.list_columns() == buffer.list_columns()


def test_buffer_and_file_report_the_same_stats_and_values(sim_run):
    """Every numeric column summarises and reads back identically."""
    path, buffer = sim_run
    with dr.open_run(path) as handle:
        numeric = [
            info.name
            for info in handle.list_columns()
            if info.dtype == "float64" and info.role != dr.ROLE_LOOP_AXIS
        ]
        assert "voltage_V" in numeric and "field_T" in numeric
        for name in numeric:
            assert handle.summary_stats(name) == buffer.summary_stats(name), name
            np.testing.assert_array_equal(
                handle.read_slice(name), buffer.read_slice(name), err_msg=name
            )
            np.testing.assert_array_equal(
                handle.read_slice(name, 1, 3), buffer.read_slice(name, 1, 3)
            )


def test_buffer_and_file_answer_the_same_metadata_keys(sim_run):
    """Both answer every canonical key; the procedure and params agree."""
    path, buffer = sim_run
    with dr.open_run(path) as handle:
        stored = handle.read_metadata()
        live = buffer.read_metadata()
    assert set(stored) == set(live) == set(dr.RUN_METADATA_KEYS)
    assert stored["procedure"] == live["procedure"]
    assert stored["params"] == live["params"]
    assert stored["data_file"] == live["data_file"]
    # Each source knows what the other cannot: the engine's run identity lives
    # in the manifest, the sample stamp lives in the file.
    assert (stored["run_id"], live["run_id"]) == ("", "sim_run_001")
    assert stored["sample"]["sample_id"] == "SIM-1"
    assert live["sample"] == {}


def test_the_module_level_vocabulary_reads_either_source(sim_run):
    """One analysis, pointed at a file or at the run in flight."""
    path, buffer = sim_run
    with dr.open_run(path) as handle:
        for source in (handle, buffer):
            assert isinstance(source, dr.RunSource)
            assert dr.summary_stats(source, "field_T").count == 3
            assert len(dr.list_columns(source)) == len(handle.list_columns())
            assert dr.read_slice(source, "field_T").shape == (3,)
            assert set(dr.read_metadata(source)) == set(dr.RUN_METADATA_KEYS)


# ── Image blocks (the image-block standard) ───────────────────────────────────


def _image_buffer(loop_shape):
    config = {
        "sweep_columns": {"field_T": "float"},
        "measurement_scalars": {},
        "measurement_arrays": {},
        "measurement_blocks": {"frame": [2, 3]},
        "measurement_image_blocks": {"frame": {"unit": "counts", "description": "f"}},
        "loop_shape": loop_shape,
    }
    manifest = {**MANIFEST, "data_config": config, "params": {"field_start": 0.0}}
    buffer = RunBuffer()
    buffer.start(RunStarted(run_id=RUN_ID, manifest=manifest))
    n1, n2 = loop_shape
    for index in range(2):
        frames = [[[[10.0 * index + i1] * 3] * 2 for i2 in range(n2)] for i1 in range(n1)]
        value = frames if (n1, n2) != (1, 1) else frames[0][0]
        buffer.append(
            Datapoint(run_id=RUN_ID, index=index, values={"field_T": float(index), "frame": value}, ts=1.0 + index)
        )
    return buffer


def test_buffer_reports_an_image_block_from_the_declaration():
    buffer = _image_buffer([1, 1])
    assert _by_name(buffer)["frame"].role == dr.ROLE_IMAGE
    assert _by_name(buffer)["frame"].shape == (2, 2, 3)


def test_buffer_read_image_matches_the_file_vocabulary():
    plain = _image_buffer([1, 1])
    np.testing.assert_allclose(plain.read_image("frame", 1), 10.0)
    with pytest.raises(IndexError):
        plain.read_image("frame", 1, loop1=1)
    with pytest.raises(IndexError):
        plain.read_image("frame", 2)
    with pytest.raises(ValueError, match="not an image block"):
        plain.read_image("field_T", 0)
    with pytest.raises(KeyError):
        plain.read_image("nope", 0)

    looped = _image_buffer([2, 1])
    np.testing.assert_allclose(looped.read_image("frame", 1, loop1=1), 11.0)
    assert looped.read_image("frame", 0).shape == (2, 3)
