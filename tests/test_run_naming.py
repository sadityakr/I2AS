"""Uniform run placement: every run is run-NNNN in its experiment's data folder.

``core.run_naming`` decides the name; the engine applies it when a run starts
(``Orchestrator.set_run_folder``); the data manager creates exactly that file
and never overwrites one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from i2as.core import events as ev
from i2as.core.data_manager import DataManager
from i2as.core.orchestrator import Orchestrator
from i2as.core.run_naming import next_run_number, place_run, run_file_name, run_id_for
from i2as.core.station import build_station
from i2as.procedures.field_sweep import FieldSweep

CONFIG_PATH = "i2as/configs/sim_cryostat"


def test_run_numbers_count_up_and_are_never_reused(tmp_path):
    assert next_run_number(tmp_path / "missing") == 1
    (tmp_path / "run-0001_FieldSweep.h5").write_bytes(b"")
    (tmp_path / "run-0007_TimeSeries_cooldown.h5").write_bytes(b"")
    (tmp_path / "notes.h5").write_bytes(b"")
    (tmp_path / "run-0099.txt").write_bytes(b"")
    assert next_run_number(tmp_path) == 8


@pytest.mark.parametrize(
    ("label", "kind", "expected"),
    [
        ("", "run", "run-0003_FieldSweep.h5"),
        ("B || c, 10 K", "run", "run-0003_FieldSweep_B_c_10_K.h5"),
        ("", "probe", "run-0003_FieldSweep_probe.h5"),
        ("probe", "probe", "run-0003_FieldSweep_probe.h5"),
    ],
)
def test_a_run_file_name_is_number_procedure_then_label(label, kind, expected):
    assert run_file_name(3, "FieldSweep", label, kind) == expected


def test_a_placement_names_the_run_and_its_file(tmp_path):
    placement = place_run(tmp_path, "FieldSweep", "first")
    assert placement.run_id == run_id_for(1) == "run-0001"
    assert placement.file_name == "run-0001_FieldSweep_first.h5"
    assert placement.data_directory == str(tmp_path)


def _data_manager(directory: Path, **extra) -> DataManager:
    return DataManager(
        data_directory=str(directory),
        procedure_name="Field Sweep",
        procedure_params={},
        sample_info={},
        instrument_state={},
        system_targets={},
        measurement_commands=[],
        data_config={
            "sweep_columns": {"unix_time": "float"},
            "measurement_scalars": {},
            "measurement_arrays": {},
            "measurement_blocks": {},
            "loop_shape": [1, 1],
        },
        n_sweep_points=1,
        **extra,
    )


def test_a_placed_file_is_created_exactly_and_never_overwritten(tmp_path):
    writer = _data_manager(tmp_path, file_name="run-0001_FieldSweep.h5")
    writer.close()
    assert Path(writer.filepath) == tmp_path / "run-0001_FieldSweep.h5"
    with pytest.raises(FileExistsError):
        _data_manager(tmp_path, file_name="run-0001_FieldSweep.h5")
    with pytest.raises(ValueError):
        _data_manager(tmp_path, file_name="../escape.h5")


@pytest.fixture
def engine(qtbot):
    orchestrator = Orchestrator(
        build_station(CONFIG_PATH), tick_interval_ms=10, run_catalog={"FieldSweep": FieldSweep}
    )
    verdicts: list[ev.Verdict] = []
    orchestrator.verdict_emitted.connect(verdicts.append)
    yield orchestrator, verdicts
    orchestrator.shutdown()


class Placeable:
    """The smallest run the engine can place: a name, a label and the hook."""

    name = "Stub"
    run_kind = "run"
    file_prefix = "lbl"

    def __init__(self) -> None:
        self.placed: tuple[str, str] | None = None

    def place_data_file(self, data_directory: str, file_name: str) -> None:
        self.placed = (data_directory, file_name)


def test_with_no_experiment_open_every_run_is_refused(engine, tmp_path):
    orchestrator, verdicts = engine
    orchestrator.set_run_folder("")

    orchestrator.submit(
        ev.Command(
            name=ev.CommandName.RUN_PROCEDURE,
            actor=ev.Actor(kind=ev.ActorKind.OPERATOR, id="op"),
            args={"procedure": "FieldSweep", "params": {}, "data_directory": str(tmp_path)},
        )
    )

    refusal = verdicts[-1]
    assert refusal.code is ev.VerdictCode.BLOCKED_STATE
    assert refusal.detail["rule"] == "no_experiment"
    assert orchestrator.state == "IDLE"
    assert not list(tmp_path.iterdir()), "a refused run writes nothing"


def test_with_a_folder_installed_a_run_is_placed_there(engine, tmp_path):
    orchestrator, _verdicts = engine
    orchestrator.set_run_folder(str(tmp_path))
    run = Placeable()

    placement = orchestrator._place_run(run)

    assert placement.run_id == "run-0001"
    assert run.placed == (str(tmp_path), "run-0001_Placeable_lbl.h5")


def test_without_a_session_layer_a_run_is_not_placed(engine):
    orchestrator, _verdicts = engine
    assert orchestrator._place_run(Placeable()) is None
