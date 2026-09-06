import json

import h5py
import numpy as np
import pytest

from i2as.core.data_manager import DataManager


# ── Shared fixtures ───────────────────────────────────────────────────────────
# loop_shape defaults to (1, 1) — no reading loop — so every measurement value
# below is wrapped one extra level: [[value]] for a scalar grid, [[[...]]] for
# an array grid.

DATA_CONFIG = {
    "sweep_columns": {"field_T": "float"},
    "measurement_arrays": {
        "voltage_V": 10,
        "current_A": 10,
    },
}

SAMPLE_INFO = {"sample_name": "Test Sample", "sample_id": "TST-001", "comments": "ci run"}

PROCEDURE_PARAMS = {
    "field_start": -1.0,
    "field_end": 1.0,
    "field_steps": 5,
    "temperature": 10.0,
}


@pytest.fixture
def dm(tmp_path):
    """A fresh DataManager writing to a temp directory."""
    manager = DataManager(
        data_directory=str(tmp_path),
        procedure_name="Test_Sweep",
        procedure_params=PROCEDURE_PARAMS,
        sample_info=SAMPLE_INFO,
        instrument_state={"magnet_z": {"field": 0.0}},
        system_targets={"magnet_z": {"target": -1.0}},
        measurement_commands={"keithley_delta_mode": {"configure": {}}},
        data_config=DATA_CONFIG,
        n_sweep_points=5,
    )
    yield manager
    # Ensure file is closed even if a test fails mid-way.
    if not manager._closed:
        manager.close()


@pytest.fixture
def saved_dm(dm):
    """DataManager with 3 of 5 points saved."""
    for i in range(3):
        dm.save_datapoint(
            sweep_index=i,
            measured_data={
                "field_T": float(i) * 0.5,
                "voltage_V": [[[float(j) * 1e-6 for j in range(10)]]],
                "current_A": [[[1e-6] * 10]],
            },
            station_snapshot={"magnet_z": {"field": float(i) * 0.5}},
        )
    return dm


# ── File creation ─────────────────────────────────────────────────────────────

def test_file_created(tmp_path):
    """DataManager creates an HDF5 file at the expected path."""
    dm = DataManager(
        data_directory=str(tmp_path),
        procedure_name="MySweep",
        procedure_params={},
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands={},
        data_config=DATA_CONFIG,
        n_sweep_points=3,
    )
    assert dm.filepath.exists()
    assert dm.filepath.suffix == ".h5"
    assert "MySweep" in dm.filepath.name
    dm.close()


def test_file_prefix_overrides_filename_stem(tmp_path):
    """A non-empty file_prefix replaces procedure_name in the filename."""
    dm = DataManager(
        data_directory=str(tmp_path),
        procedure_name="MySweep",
        file_prefix="custom_run",
        procedure_params={},
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands={},
        data_config=DATA_CONFIG,
        n_sweep_points=3,
    )
    assert dm.filepath.name.startswith("custom_run_")
    assert "MySweep" not in dm.filepath.name
    dm.close()


def test_file_prefix_metadata_still_records_true_procedure_name(tmp_path):
    """procedure_name metadata is unaffected by a custom file_prefix."""
    dm = DataManager(
        data_directory=str(tmp_path),
        procedure_name="MySweep",
        file_prefix="custom_run",
        procedure_params={},
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands={},
        data_config=DATA_CONFIG,
        n_sweep_points=3,
    )
    filepath = dm.filepath
    dm.close()
    with h5py.File(filepath, "r") as f:
        assert f["metadata"].attrs["procedure_name"] == "MySweep"


def test_blank_file_prefix_falls_back_to_procedure_name(tmp_path):
    """An empty/whitespace file_prefix behaves like the default (unset) case."""
    dm = DataManager(
        data_directory=str(tmp_path),
        procedure_name="MySweep",
        file_prefix="   ",
        procedure_params={},
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands={},
        data_config=DATA_CONFIG,
        n_sweep_points=3,
    )
    assert dm.filepath.name.startswith("MySweep_")
    dm.close()


def test_invalid_n_sweep_points(tmp_path):
    """n_sweep_points < 1 raises ValueError."""
    with pytest.raises(ValueError):
        DataManager(
            data_directory=str(tmp_path),
            procedure_name="Bad",
            procedure_params={},
            sample_info={},
            instrument_state={},
            system_targets={},
            measurement_commands={},
            data_config=DATA_CONFIG,
            n_sweep_points=0,
        )


# ── Metadata ──────────────────────────────────────────────────────────────────

def test_metadata_attributes(dm):
    """All metadata attributes are stored correctly as JSON strings."""
    with h5py.File(dm.filepath, "r") as f:
        meta = f["metadata"].attrs
        assert meta["procedure_name"] == "Test_Sweep"

        params = json.loads(meta["procedure_params"])
        assert params["field_start"] == -1.0

        si = json.loads(meta["sample_info"])
        assert si["sample_name"] == "Test Sample"

        assert meta["start_time"] != ""
        assert meta["end_time"] == ""  # Not yet closed

        dc = json.loads(meta["data_config"])
        assert "sweep_columns" in dc
        assert "measurement_arrays" in dc


def test_end_time_written_on_close(dm):
    """close() writes a non-empty end_time."""
    dm.close()
    with h5py.File(dm.filepath, "r") as f:
        assert f["metadata"].attrs["end_time"] != ""


# ── Pre-allocation ────────────────────────────────────────────────────────────

def test_dataset_shapes(dm):
    """Datasets are pre-allocated with correct shapes (loop_shape defaults (1, 1))."""
    with h5py.File(dm.filepath, "r") as f:
        assert f["data"]["field_T"].shape == (5,)
        assert f["data"]["voltage_V"].shape == (5, 1, 1, 10)
        assert f["data"]["current_A"].shape == (5, 1, 1, 10)
        assert f["data"]["timestamp"].shape == (5,)


def test_dataset_shapes_with_loop_axis(tmp_path):
    """A non-trivial loop_shape sizes measurement datasets on that axis too."""
    config = {
        "sweep_columns": {"field_T": "float"},
        "measurement_scalars": {"voltage_V": "float"},
        "measurement_arrays": {"voltage_V_array": 10},
        "loop_shape": [2, 3],
    }
    manager = DataManager(
        data_directory=str(tmp_path),
        procedure_name="Loop_Sweep",
        procedure_params={},
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands={},
        data_config=config,
        n_sweep_points=4,
    )
    try:
        with h5py.File(manager.filepath, "r") as f:
            assert f["data"]["field_T"].shape == (4,)  # sweep-only: no loop axis
            assert f["data"]["voltage_V"].shape == (4, 2, 3)
            assert f["data"]["voltage_V_array"].shape == (4, 2, 3, 10)
    finally:
        manager.close()


def test_initial_fill_is_nan(dm):
    """Numeric datasets are pre-filled with NaN."""
    with h5py.File(dm.filepath, "r") as f:
        assert np.all(np.isnan(f["data"]["field_T"][:]))
        assert np.all(np.isnan(f["data"]["voltage_V"][:]))


# ── save_datapoint ────────────────────────────────────────────────────────────

def test_save_single_datapoint(dm):
    """save_datapoint() writes correct values at a given index."""
    voltages = [float(i) * 1e-6 for i in range(10)]
    dm.save_datapoint(
        sweep_index=2,
        measured_data={
            "field_T": 0.5,
            "voltage_V": [[voltages]],
            "current_A": [[[1e-6] * 10]],
        },
        station_snapshot={"magnet_z": {"field": 0.5}},
    )
    with h5py.File(dm.filepath, "r") as f:
        assert f["data"]["field_T"][2] == pytest.approx(0.5)
        assert list(f["data"]["voltage_V"][2, 0, 0]) == pytest.approx(voltages)
        assert f["data"]["timestamp"][2] != ""


def test_save_datapoint_with_loop_shape(tmp_path):
    """save_datapoint() writes a full (n_loop1, n_loop2[, length]) grid per call."""
    config = {
        "sweep_columns": {"field_T": "float"},
        "measurement_scalars": {"voltage_V": "float"},
        "measurement_arrays": {"voltage_V_array": 2},
        "loop_shape": [2, 2],
    }
    manager = DataManager(
        data_directory=str(tmp_path),
        procedure_name="Loop_Sweep",
        procedure_params={},
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands={},
        data_config=config,
        n_sweep_points=1,
    )
    try:
        manager.save_datapoint(
            sweep_index=0,
            measured_data={
                "field_T": 1.0,
                "voltage_V": [[1.0, 2.0], [3.0, 4.0]],
                "voltage_V_array": [
                    [[1.0, 1.5], [2.0, 2.5]],
                    [[3.0, 3.5], [4.0, 4.5]],
                ],
            },
            station_snapshot={},
        )
        with h5py.File(manager.filepath, "r") as f:
            assert list(f["data"]["voltage_V"][0].flatten()) == pytest.approx(
                [1.0, 2.0, 3.0, 4.0]
            )
            assert f["data"]["voltage_V_array"][0, 1, 1, 0] == pytest.approx(4.0)
            assert f["data"]["voltage_V_array"][0, 0, 1, 1] == pytest.approx(2.5)
    finally:
        manager.close()


def test_save_datapoint_measurement_array_wrong_loop_axis_raises(dm):
    """A measurement array whose LOOP axes (not just sample count) are wrong raises."""
    with pytest.raises(ValueError, match="measurement array"):
        dm.save_datapoint(
            sweep_index=0,
            # 2 loop1 entries where loop_shape declares only 1 — not just an
            # innermost sample-count mismatch, so this is not pad/truncated.
            measured_data={"voltage_V": [[[1.0] * 10], [[1.0] * 10]]},
            station_snapshot={},
        )


def test_save_datapoint_measurement_scalar_wrong_loop_shape_raises(tmp_path):
    """A measurement scalar whose grid shape mismatches loop_shape raises loudly."""
    config = {
        "sweep_columns": {},
        "measurement_scalars": {"voltage_V": "float"},
        "measurement_arrays": {},
        "loop_shape": [1, 1],
    }
    manager = DataManager(
        data_directory=str(tmp_path),
        procedure_name="Loop_Sweep",
        procedure_params={},
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands={},
        data_config=config,
        n_sweep_points=1,
    )
    try:
        with pytest.raises(ValueError, match="loop shape"):
            manager.save_datapoint(
                sweep_index=0,
                measured_data={"voltage_V": [[1.0, 2.0]]},  # (1, 2) != (1, 1)
                station_snapshot={},
            )
    finally:
        manager.close()


def test_save_multiple_datapoints(saved_dm):
    """Multiple save_datapoint() calls write at the correct indices."""
    with h5py.File(saved_dm.filepath, "r") as f:
        assert f["data"]["field_T"][0] == pytest.approx(0.0)
        assert f["data"]["field_T"][1] == pytest.approx(0.5)
        assert f["data"]["field_T"][2] == pytest.approx(1.0)
        # Indices 3 and 4 should still be NaN (not yet saved)
        assert np.isnan(f["data"]["field_T"][3])
        assert np.isnan(f["data"]["field_T"][4])


def test_save_out_of_range_raises(dm):
    """save_datapoint() with index >= n_sweep_points raises IndexError."""
    with pytest.raises(IndexError):
        dm.save_datapoint(10, {"field_T": 1.0}, {})


def test_save_on_closed_raises(dm):
    """save_datapoint() after close() raises RuntimeError."""
    dm.close()
    with pytest.raises(RuntimeError):
        dm.save_datapoint(0, {"field_T": 1.0}, {})


def test_unknown_column_ignored(dm):
    """save_datapoint() with an unknown column logs a warning and doesn't crash."""
    dm.save_datapoint(
        sweep_index=0,
        measured_data={"field_T": 0.0, "unknown_key": 42.0},
        station_snapshot={},
    )
    # field_T should still be written
    with h5py.File(dm.filepath, "r") as f:
        assert f["data"]["field_T"][0] == pytest.approx(0.0)


# ── Snapshots ─────────────────────────────────────────────────────────────────

def test_snapshots_stored_as_json(saved_dm):
    """Snapshots for saved indices exist and are valid JSON."""
    saved_dm.close()
    with h5py.File(saved_dm.filepath, "r") as f:
        for i in range(3):
            raw = f["snapshots"][str(i)][()]
            snap = json.loads(raw)
            assert "magnet_z" in snap


def test_no_snapshot_for_unsaved_index(saved_dm):
    """Snapshot group only has datasets for saved indices."""
    saved_dm.close()
    with h5py.File(saved_dm.filepath, "r") as f:
        assert "3" not in f["snapshots"]
        assert "4" not in f["snapshots"]


# ── record_settings_snapshot() (externally configured provenance) ─────────────

def test_record_settings_snapshot_writes_metadata_attr(dm):
    """record_settings_snapshot() writes a JSON attr under /metadata."""
    dm.record_settings_snapshot({"camp": 1e-3, "avgt": 0.05, "wfmd": 0})
    dm.close()
    with h5py.File(dm.filepath, "r") as f:
        raw = f["metadata"].attrs["measurement_settings_snapshot"]
        snapshot = json.loads(raw)
        assert snapshot == {"camp": 1e-3, "avgt": 0.05, "wfmd": 0}


def test_record_settings_snapshot_before_close_is_readable_live(dm):
    """The attr is present immediately (before close()), not only after."""
    dm.record_settings_snapshot({"camp": 2e-3})
    assert json.loads(dm._file["metadata"].attrs["measurement_settings_snapshot"]) == {
        "camp": 2e-3
    }


def test_record_settings_snapshot_is_idempotent_overwrite(dm):
    """A second call overwrites the first snapshot rather than erroring."""
    dm.record_settings_snapshot({"camp": 1e-3})
    dm.record_settings_snapshot({"camp": 9e-3})
    dm.close()
    with h5py.File(dm.filepath, "r") as f:
        snapshot = json.loads(f["metadata"].attrs["measurement_settings_snapshot"])
        assert snapshot == {"camp": 9e-3}


def test_record_settings_snapshot_on_closed_raises(dm):
    """record_settings_snapshot() after close() raises RuntimeError."""
    dm.close()
    with pytest.raises(RuntimeError):
        dm.record_settings_snapshot({"camp": 1e-3})


def test_no_snapshot_attr_when_never_recorded(dm):
    """A run that never calls record_settings_snapshot() has no such attr."""
    dm.close()
    with h5py.File(dm.filepath, "r") as f:
        assert "measurement_settings_snapshot" not in f["metadata"].attrs


# ── close() and trim on abort ─────────────────────────────────────────────────

def test_close_full_sweep(dm):
    """Full sweep: no trimming, shapes unchanged."""
    for i in range(5):
        dm.save_datapoint(i, {"field_T": float(i)}, {"state": i})
    dm.close()
    with h5py.File(dm.filepath, "r") as f:
        assert f["data"]["field_T"].shape == (5,)


def test_close_trims_on_abort(saved_dm):
    """Partial sweep: close() trims datasets to actual saved points (3 of 5)."""
    saved_dm.close()
    with h5py.File(saved_dm.filepath, "r") as f:
        assert f["data"]["field_T"].shape == (3,)
        assert f["data"]["voltage_V"].shape == (3, 1, 1, 10)
        assert f["data"]["timestamp"].shape == (3,)


def test_double_close_is_safe(dm):
    """close() called twice doesn't raise."""
    dm.close()
    dm.close()  # Should not raise


def test_file_readable_after_close(saved_dm):
    """Closed file can be reopened and data is intact."""
    saved_dm.close()
    with h5py.File(saved_dm.filepath, "r") as f:
        assert f["data"]["field_T"][0] == pytest.approx(0.0)
        assert f["data"]["field_T"][1] == pytest.approx(0.5)
        assert f["data"]["field_T"][2] == pytest.approx(1.0)
        assert json.loads(f["metadata"].attrs["procedure_params"])["field_start"] == -1.0


# ── Short / long measurement arrays (review finding H6) ──────────────────────

def test_save_datapoint_pads_short_measurement_arrays(dm):
    """A short array (e.g. the delta engine aborted an acquisition early)
    must be NaN-padded to the allocated width, not crash with a
    shape-mismatch ValueError that would kill the whole run. Only the
    innermost (per-point sample) axis is padded — the loop axes stay (1, 1)."""
    dm.save_datapoint(
        sweep_index=0,
        measured_data={
            "field_T": 0.1,
            "voltage_V": [[[1e-6, 2e-6, 3e-6]]],  # only 3 of the 10 allocated
            "current_A": [[[1e-6] * 10]],
        },
        station_snapshot={},
    )
    stored = dm._file["data"]["voltage_V"][0, 0, 0]
    assert stored.shape == (10,)
    assert np.allclose(stored[:3], [1e-6, 2e-6, 3e-6])
    assert np.all(np.isnan(stored[3:]))


def test_save_datapoint_truncates_long_measurement_arrays(dm):
    """An over-long array is truncated to the allocated width, not fatal."""
    dm.save_datapoint(
        sweep_index=0,
        measured_data={"voltage_V": [[list(range(15))]]},
        station_snapshot={},
    )
    stored = dm._file["data"]["voltage_V"][0, 0, 0]
    assert stored.shape == (10,)
    assert np.allclose(stored, np.arange(10, dtype=float))


# ── experiment_info metadata (session layer) ─────────────────────────────────

def test_experiment_info_defaults_to_empty(dm):
    """Without experiment_info the attribute still exists, recording {}."""
    dm.close()
    with h5py.File(dm.filepath, "r") as f:
        assert json.loads(f["metadata"].attrs["experiment_info"]) == {}


def test_experiment_info_written(tmp_path):
    """A supplied experiment_info dict round-trips via /metadata/experiment_info."""
    info = {
        "experiment_id": "20260717_hallbar_a3",
        "experiment_title": "SOT switching vs T",
        "user_id": "jdoe",
        "user_name": "J. Doe",
    }
    dm = DataManager(
        data_directory=str(tmp_path),
        procedure_name="Test_Sweep",
        procedure_params=PROCEDURE_PARAMS,
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands=[],
        data_config=DATA_CONFIG,
        n_sweep_points=5,
        experiment_info=info,
    )
    dm.close()
    with h5py.File(dm.filepath, "r") as f:
        assert json.loads(f["metadata"].attrs["experiment_info"]) == info


# ── Raw diagnostic blocks (measurement_blocks) ────────────────────────────────
# See MeasurementInstrumentBase's "Raw diagnostic blocks" standard: a fixed-
# shape (rows x channels) grid per sweep point, orthogonal to the mean/error/
# array convention the other sections above cover. UNLIKE measurement_arrays,
# a block carries the (n_loop1, n_loop2) loop axis ONLY when a reading loop is
# actually configured (loop_shape != (1, 1)) — see DataSchema.measurement_blocks.
# BLOCK_CONFIG below uses the default loop_shape (1, 1), so every value in this
# section is bare (rows, cols); test_block_dataset_shape_with_loop_axis /
# test_save_block_with_active_loop cover the (n_loop1, n_loop2, rows, cols)
# case with a non-trivial loop_shape.

BLOCK_CONFIG = {
    "sweep_columns": {"field_T": "float"},
    "measurement_arrays": {},
    "measurement_blocks": {"raw_channels_block": (5, 3)},
}


@pytest.fixture
def block_dm(tmp_path):
    """A fresh DataManager whose data_config declares one raw block, no reading loop."""
    manager = DataManager(
        data_directory=str(tmp_path),
        procedure_name="Block_Sweep",
        procedure_params=PROCEDURE_PARAMS,
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands={},
        data_config=BLOCK_CONFIG,
        n_sweep_points=5,
    )
    yield manager
    if not manager._closed:
        manager.close()


def test_block_dataset_shape_and_nan_fill(block_dm):
    """No reading loop: a declared block pre-allocates bare (N, rows, cols), NaN-filled."""
    with h5py.File(block_dm.filepath, "r") as f:
        ds = f["data"]["raw_channels_block"]
        assert ds.shape == (5, 5, 3)
        assert np.all(np.isnan(ds[:]))


def test_block_dataset_shape_with_loop_axis(tmp_path):
    """A non-trivial loop_shape sizes the block dataset on that axis too."""
    config = {
        "sweep_columns": {},
        "measurement_blocks": {"raw_channels_block": (5, 3)},
        "loop_shape": [2, 1],
    }
    manager = DataManager(
        data_directory=str(tmp_path),
        procedure_name="Loop_Block_Sweep",
        procedure_params={},
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands={},
        data_config=config,
        n_sweep_points=2,
    )
    try:
        with h5py.File(manager.filepath, "r") as f:
            assert f["data"]["raw_channels_block"].shape == (2, 2, 1, 5, 3)
    finally:
        manager.close()


def test_block_dataset_axes_attr_no_loop(block_dm):
    """No labels declared: the dataset still self-describes its axis order."""
    with h5py.File(block_dm.filepath, "r") as f:
        ds = f["data"]["raw_channels_block"]
        assert ds.attrs["axes"] == "sweep_point, row, channel"
        assert "channel_names" not in ds.attrs


def test_block_dataset_channel_names_attr(tmp_path):
    """measurement_block_labels is written as the block's channel_names attribute."""
    config = {
        **BLOCK_CONFIG,
        "measurement_block_labels": {"raw_channels_block": ["res_a_ohm", "res_b_ohm", "phase_deg"]},
    }
    manager = DataManager(
        data_directory=str(tmp_path),
        procedure_name="Labelled_Block_Sweep",
        procedure_params=PROCEDURE_PARAMS,
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands={},
        data_config=config,
        n_sweep_points=5,
    )
    try:
        with h5py.File(manager.filepath, "r") as f:
            ds = f["data"]["raw_channels_block"]
            assert list(ds.attrs["channel_names"]) == ["res_a_ohm", "res_b_ohm", "phase_deg"]
            assert ds.attrs["axes"] == "sweep_point, row, channel"
    finally:
        manager.close()


def test_block_dataset_axes_attr_with_loop(tmp_path):
    """A non-trivial loop_shape includes loop1/loop2 in the axes attribute."""
    config = {
        "sweep_columns": {},
        "measurement_blocks": {"raw_channels_block": (5, 3)},
        "measurement_block_labels": {"raw_channels_block": ["res_a_ohm", "res_b_ohm", "phase_deg"]},
        "loop_shape": [2, 1],
    }
    manager = DataManager(
        data_directory=str(tmp_path),
        procedure_name="Loop_Block_Sweep",
        procedure_params={},
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands={},
        data_config=config,
        n_sweep_points=2,
    )
    try:
        with h5py.File(manager.filepath, "r") as f:
            ds = f["data"]["raw_channels_block"]
            assert ds.attrs["axes"] == "sweep_point, loop1, loop2, row, channel"
            assert list(ds.attrs["channel_names"]) == ["res_a_ohm", "res_b_ohm", "phase_deg"]
    finally:
        manager.close()


def test_save_block_round_trip(block_dm):
    """No reading loop: save_datapoint() writes the bare (rows, cols) grid verbatim."""
    block = [[float(r * 3 + c) for c in range(3)] for r in range(5)]
    block_dm.save_datapoint(
        sweep_index=1,
        measured_data={"field_T": 0.2, "raw_channels_block": block},
        station_snapshot={},
    )
    with h5py.File(block_dm.filepath, "r") as f:
        stored = f["data"]["raw_channels_block"][1]
        assert stored.shape == (5, 3)
        assert np.allclose(stored, block)


def test_save_block_with_active_loop(tmp_path):
    """A non-trivial loop_shape: save_datapoint() writes the full (n_loop1, n_loop2, rows, cols) grid."""
    config = {
        "sweep_columns": {},
        "measurement_blocks": {"raw_channels_block": (5, 3)},
        "loop_shape": [2, 1],
    }
    manager = DataManager(
        data_directory=str(tmp_path),
        procedure_name="Loop_Block_Sweep",
        procedure_params={},
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands={},
        data_config=config,
        n_sweep_points=1,
    )
    try:
        block0 = [[float(r * 3 + c) for c in range(3)] for r in range(5)]
        block1 = [[float(r * 3 + c + 100) for c in range(3)] for r in range(5)]
        manager.save_datapoint(
            sweep_index=0,
            measured_data={"raw_channels_block": [[block0], [block1]]},
            station_snapshot={},
        )
        with h5py.File(manager.filepath, "r") as f:
            stored = f["data"]["raw_channels_block"][0]
            assert stored.shape == (2, 1, 5, 3)
            assert np.allclose(stored[0, 0], block0)
            assert np.allclose(stored[1, 0], block1)
    finally:
        manager.close()


def test_save_block_pads_short_row_count(block_dm):
    """A block with fewer rows than allocated is NaN-padded on the ROW axis only."""
    short_block = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # only 2 of 5 allocated rows
    block_dm.save_datapoint(
        sweep_index=0,
        measured_data={"raw_channels_block": short_block},
        station_snapshot={},
    )
    stored = block_dm._file["data"]["raw_channels_block"][0]
    assert stored.shape == (5, 3)
    assert np.allclose(stored[:2], short_block)
    assert np.all(np.isnan(stored[2:]))


def test_save_block_wrong_channel_count_raises(block_dm):
    """A channel-axis (column count) mismatch is a hard error, never padded."""
    with pytest.raises(ValueError, match="measurement block"):
        block_dm.save_datapoint(
            sweep_index=0,
            measured_data={"raw_channels_block": [[1.0, 2.0]]},  # 2 cols, wants 3
            station_snapshot={},
        )


def test_close_trims_block_dataset_on_abort(block_dm):
    """close()'s trim-to-actual-points loop resizes a block's axis 0 too."""
    block = [[0.0] * 3 for _ in range(5)]
    block_dm.save_datapoint(0, {"raw_channels_block": block}, {})
    block_dm.save_datapoint(1, {"raw_channels_block": block}, {})
    block_dm.close()
    with h5py.File(block_dm.filepath, "r") as f:
        assert f["data"]["raw_channels_block"].shape == (2, 5, 3)


# ── Image blocks (the image-block standard) ───────────────────────────────────
# A frame rides the raw block's dataset path (same shape rule, same NaN fill,
# same loop-axis rule) but describes itself differently on disk: block_kind
# "image", a unit and a description, and NO channel_names.

IMAGE_CONFIG = {
    "sweep_columns": {"field_T": "float"},
    "measurement_arrays": {},
    "measurement_blocks": {"frame": (4, 6)},
    "measurement_image_blocks": {"frame": {"unit": "counts", "description": "sim frame"}},
}


def _image_dm(tmp_path, config=IMAGE_CONFIG, n_points=3):
    return DataManager(
        data_directory=str(tmp_path),
        procedure_name="Image_Sweep",
        procedure_params=PROCEDURE_PARAMS,
        sample_info=SAMPLE_INFO,
        instrument_state={},
        system_targets={},
        measurement_commands={},
        data_config=config,
        n_sweep_points=n_points,
    )


def test_image_block_dataset_is_marked_as_an_image(tmp_path):
    manager = _image_dm(tmp_path)
    manager.close()
    with h5py.File(manager.filepath, "r") as f:
        ds = f["data"]["frame"]
        assert ds.shape[1:] == (4, 6)
        assert ds.attrs["block_kind"] == "image"
        assert ds.attrs["unit"] == "counts"
        assert ds.attrs["description"] == "sim frame"
        assert ds.attrs["axes"] == "sweep_point, row, col"
        assert "channel_names" not in ds.attrs


def test_raw_block_dataset_is_marked_as_raw(block_dm):
    """The sibling declaration says what it is too, so a reader never guesses."""
    with h5py.File(block_dm.filepath, "r") as f:
        ds = f["data"]["raw_channels_block"]
        assert ds.attrs["block_kind"] == "raw"
        assert "unit" not in ds.attrs


def test_image_block_round_trip_without_loop(tmp_path):
    manager = _image_dm(tmp_path)
    frame = np.arange(24, dtype=float).reshape(4, 6)
    manager.save_datapoint(1, {"field_T": 0.5, "frame": frame}, {})
    manager.close()
    with h5py.File(manager.filepath, "r") as f:
        stored = f["data"]["frame"][:]
    assert stored.shape == (2, 4, 6)  # trimmed to the written prefix on close
    np.testing.assert_array_equal(stored[1], frame)
    assert np.all(np.isnan(stored[0]))


def test_image_block_round_trip_with_loop(tmp_path):
    config = {**IMAGE_CONFIG, "loop_shape": [2, 1]}
    manager = _image_dm(tmp_path, config=config)
    frames = np.stack([np.full((4, 6), 1.0), np.full((4, 6), 2.0)])[:, None, :, :]  # (2, 1, 4, 6)
    manager.save_datapoint(0, {"field_T": 0.0, "frame": frames}, {})
    manager.close()
    with h5py.File(manager.filepath, "r") as f:
        ds = f["data"]["frame"]
        assert ds.attrs["axes"] == "sweep_point, loop1, loop2, row, col"
        assert ds.attrs["block_kind"] == "image"
        np.testing.assert_array_equal(ds[0, 1, 0], np.full((4, 6), 2.0))


def test_image_block_wrong_width_raises(tmp_path):
    manager = _image_dm(tmp_path)
    with pytest.raises(ValueError, match="frame"):
        manager.save_datapoint(0, {"field_T": 0.0, "frame": np.zeros((4, 5))}, {})
    manager.close()
