import dataclasses
import hashlib
import json

import pytest

from i2as.core.exceptions import DataSchemaError
from i2as.core.plan import (
    ImageBlock,
    Command,
    DataSchema,
    EnvelopeBound,
    ExperimentEnvelope,
    ParamGroup,
    UIGroup,
    ParamSpec,
    PhasePlan,
    StepPlan,
    Target,
    params_digest,
)


# ── Target ────────────────────────────────────────────────────────────────────


def test_target_happy_and_defaults():
    t = Target(1.5)
    assert t.target == 1.5
    assert t.rate is None
    t2 = Target(2, rate=0.1)
    assert t2.target == 2.0 and isinstance(t2.target, float)
    assert t2.rate == 0.1


def test_target_bool_rejected():
    with pytest.raises(TypeError, match="Target.target"):
        Target(True)


def test_target_nan_inf_rejected():
    with pytest.raises(ValueError, match="Target.target"):
        Target(float("nan"))
    with pytest.raises(ValueError, match="Target.target"):
        Target(float("inf"))


def test_target_rate_must_be_positive():
    with pytest.raises(ValueError, match="Target.rate"):
        Target(1.0, rate=0.0)
    with pytest.raises(ValueError, match="Target.rate"):
        Target(1.0, rate=-1.0)


def test_target_rate_nonfinite_rejected():
    with pytest.raises(ValueError, match="Target.rate"):
        Target(1.0, rate=float("inf"))


def test_target_frozen():
    t = Target(1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.target = 2.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.nonexistent = 3  # type: ignore[attr-defined]


# ── Command ───────────────────────────────────────────────────────────────────


def test_command_happy_and_default_kwargs():
    c = Command("magnet", "set_field")
    assert c.vi_name == "magnet"
    assert c.method == "set_field"
    assert c.kwargs == {}
    c2 = Command("src", "arm", kwargs={"level": 1e-6})
    assert c2.kwargs == {"level": 1e-6}


def test_command_empty_vi_name():
    with pytest.raises(ValueError, match="Command.vi_name"):
        Command("", "m")


def test_command_empty_method():
    with pytest.raises(ValueError, match="Command.method"):
        Command("vi", "")


def test_command_method_must_be_identifier():
    with pytest.raises(ValueError, match="Command.method"):
        Command("vi", "not a method")


def test_command_frozen():
    c = Command("vi", "m")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.vi_name = "other"  # type: ignore[misc]


def test_command_kwargs_defensive_copy():
    payload = {"a": 1}
    c = Command("vi", "m", kwargs=payload)
    payload["a"] = 999
    payload["b"] = 2
    assert c.kwargs == {"a": 1}


# ── PhasePlan ─────────────────────────────────────────────────────────────────


def test_phaseplan_happy_and_defaults():
    p = PhasePlan(targets={"field": Target(1.0)})
    assert p.commands == ()
    assert p.claim_commands == ()
    assert p.wait_s == 0.0


def test_phaseplan_commands_normalized_to_tuple_order_preserved():
    c1 = Command("switch", "close")
    c2 = Command("source", "arm")
    p = PhasePlan(targets={}, commands=[c1, c2])
    assert isinstance(p.commands, tuple)
    assert p.commands == (c1, c2)


def test_phaseplan_claim_commands_normalized_to_tuple_order_preserved():
    c1 = Command("magnet_z", "initiate")
    c2 = Command("temperature", "initiate")
    p = PhasePlan(targets={}, claim_commands=[c1, c2])
    assert isinstance(p.claim_commands, tuple)
    assert p.claim_commands == (c1, c2)


def test_phaseplan_bad_claim_command():
    with pytest.raises(TypeError, match="PhasePlan.claim_commands"):
        PhasePlan(targets={}, claim_commands=["nope"])


def test_phaseplan_bad_target_value():
    with pytest.raises(TypeError, match="PhasePlan.targets"):
        PhasePlan(targets={"field": "not a target"})


def test_phaseplan_empty_target_key():
    with pytest.raises(ValueError, match="PhasePlan.targets"):
        PhasePlan(targets={"": Target(1.0)})


def test_phaseplan_bad_command():
    with pytest.raises(TypeError, match="PhasePlan.commands"):
        PhasePlan(targets={}, commands=["nope"])


def test_phaseplan_wait_negative():
    with pytest.raises(ValueError, match="PhasePlan.wait_s"):
        PhasePlan(targets={}, wait_s=-1.0)


def test_phaseplan_frozen():
    p = PhasePlan(targets={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.wait_s = 5.0  # type: ignore[misc]


def test_phaseplan_targets_defensive_copy():
    targets = {"field": Target(1.0)}
    p = PhasePlan(targets=targets)
    targets["temp"] = Target(4.2)
    assert "temp" not in p.targets


# ── StepPlan ──────────────────────────────────────────────────────────────────


def test_stepplan_happy():
    s = StepPlan(targets={"field": Target(0.5)}, wait_s=2.0)
    assert s.wait_s == 2.0


def test_stepplan_bad_wait_type():
    with pytest.raises(TypeError, match="StepPlan.wait_s"):
        StepPlan(targets={}, wait_s="soon")


def test_stepplan_frozen():
    s = StepPlan(targets={}, wait_s=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.wait_s = 1.0  # type: ignore[misc]


# ── ParamSpec ─────────────────────────────────────────────────────────────────


def test_paramspec_happy_and_defaults():
    p = ParamSpec(type=float, default=1.0)
    assert p.unit == "" and p.description == ""
    assert p.min is None and p.max is None
    assert p.choices is None and p.structural is False and p.widget_hint is None


def test_paramspec_int_default_ok_for_float_type():
    p = ParamSpec(type=float, default=3)
    assert p.default == 3


def test_paramspec_bool_default_not_accepted_as_int():
    with pytest.raises(ValueError, match="ParamSpec.default"):
        ParamSpec(type=int, default=True)


def test_paramspec_bool_default_not_accepted_as_float():
    with pytest.raises(ValueError, match="ParamSpec.default"):
        ParamSpec(type=float, default=True)


def test_paramspec_bool_type_accepts_bool():
    p = ParamSpec(type=bool, default=True)
    assert p.default is True


def test_paramspec_default_wrong_type():
    with pytest.raises(ValueError, match="ParamSpec.default"):
        ParamSpec(type=int, default="five")


def test_paramspec_bad_type():
    with pytest.raises(TypeError, match="ParamSpec.type"):
        ParamSpec(type=list, default=[])


def test_paramspec_bounds_ok():
    p = ParamSpec(type=float, default=5.0, min=0.0, max=10.0)
    assert p.min == 0.0 and p.max == 10.0


def test_paramspec_bounds_reject_non_numeric_type():
    with pytest.raises(ValueError, match="ParamSpec.min/max"):
        ParamSpec(type=str, default="x", min=0.0)


def test_paramspec_default_below_min():
    with pytest.raises(ValueError, match="ParamSpec.default"):
        ParamSpec(type=float, default=-1.0, min=0.0)


def test_paramspec_default_above_max():
    with pytest.raises(ValueError, match="ParamSpec.default"):
        ParamSpec(type=float, default=11.0, max=10.0)


def test_paramspec_min_greater_than_max():
    with pytest.raises(ValueError, match="ParamSpec.min"):
        ParamSpec(type=float, default=5.0, min=10.0, max=0.0)


def test_paramspec_choices_ok():
    p = ParamSpec(type=int, default=2, choices={"low": 1, "high": 2})
    assert p.choices == {"low": 1, "high": 2}


def test_paramspec_choices_empty():
    with pytest.raises(ValueError, match="ParamSpec.choices"):
        ParamSpec(type=int, default=1, choices={})


def test_paramspec_choices_value_wrong_type():
    with pytest.raises(ValueError, match="ParamSpec.choices"):
        ParamSpec(type=int, default=1, choices={"a": 1, "b": "two"})


def test_paramspec_choices_default_not_among_values():
    with pytest.raises(ValueError, match="ParamSpec.default"):
        ParamSpec(type=int, default=9, choices={"a": 1, "b": 2})


def test_paramspec_choices_and_bounds_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ParamSpec(type=int, default=1, min=0, choices={"a": 1})


def test_paramspec_widget_hint_empty():
    with pytest.raises(ValueError, match="ParamSpec.widget_hint"):
        ParamSpec(type=float, default=1.0, widget_hint="")


def test_paramspec_structural_flag():
    p = ParamSpec(type=bool, default=False, structural=True)
    assert p.structural is True


def test_paramspec_frozen():
    p = ParamSpec(type=float, default=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.default = 2.0  # type: ignore[misc]


def test_paramspec_choices_defensive_copy():
    choices = {"a": 1, "b": 2}
    p = ParamSpec(type=int, default=1, choices=choices)
    choices["c"] = 3
    assert "c" not in p.choices


# ── ParamGroup ────────────────────────────────────────────────────────────────


def test_paramgroup_happy():
    g = ParamGroup(
        key="system", title="System", params={"field": ParamSpec(type=float, default=0.0)}
    )
    assert g.key == "system"
    assert "field" in g.params


def test_paramgroup_empty_key():
    with pytest.raises(ValueError, match="ParamGroup.key"):
        ParamGroup(key="", title="T", params={})


def test_paramgroup_empty_title():
    with pytest.raises(ValueError, match="ParamGroup.title"):
        ParamGroup(key="k", title="", params={})


def test_paramgroup_bad_param_value():
    with pytest.raises(TypeError, match="ParamGroup.params"):
        ParamGroup(key="k", title="T", params={"x": "not a spec"})


def test_paramgroup_empty_param_key():
    with pytest.raises(ValueError, match="ParamGroup.params"):
        ParamGroup(key="k", title="T", params={"": ParamSpec(type=int, default=0)})


def test_paramgroup_frozen():
    g = ParamGroup(key="k", title="T", params={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.title = "other"  # type: ignore[misc]


def test_paramgroup_params_defensive_copy():
    params = {"a": ParamSpec(type=int, default=0)}
    g = ParamGroup(key="k", title="T", params=params)
    params["b"] = ParamSpec(type=int, default=1)
    assert "b" not in g.params


# ── UIGroup ───────────────────────────────────────────────────────────────────


def test_uigroup_happy():
    g = UIGroup(
        key="heater",
        title="Heater",
        description="Heater readback and control.",
        members=("heater_output", "set_heater_output"),
    )
    assert g.key == "heater"
    assert g.members == ("heater_output", "set_heater_output")


def test_uigroup_empty_key():
    with pytest.raises(ValueError, match="UIGroup.key"):
        UIGroup(key="", title="T", members=("x",))


def test_uigroup_empty_title():
    with pytest.raises(ValueError, match="UIGroup.title"):
        UIGroup(key="k", title="", members=("x",))


def test_uigroup_requires_members():
    """A group with no members declares nothing, so it is refused."""
    with pytest.raises(ValueError, match="at least one member"):
        UIGroup(key="k", title="T")


def test_uigroup_members_must_be_strings():
    with pytest.raises(TypeError, match="member must be a str"):
        UIGroup(key="k", title="T", members=(1,))  # type: ignore[arg-type]


def test_uigroup_members_reject_a_bare_string():
    """A bare string would silently become a group of single characters."""
    with pytest.raises(TypeError, match="UIGroup.members"):
        UIGroup(key="k", title="T", members="set_heater")  # type: ignore[arg-type]


def test_uigroup_members_reject_duplicates():
    with pytest.raises(ValueError, match="lists a member twice"):
        UIGroup(key="k", title="T", members=("x", "x"))


def test_uigroup_members_coerced_to_tuple():
    """A list is accepted and frozen, so declared order cannot be mutated."""
    g = UIGroup(key="k", title="T", members=["a", "b"])
    assert g.members == ("a", "b")


def test_uigroup_frozen():
    g = UIGroup(key="k", title="T", members=("x",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.title = "other"  # type: ignore[misc]


# ── DataSchema ────────────────────────────────────────────────────────────────


def test_dataschema_happy():
    s = DataSchema(
        sweep_columns={"field_T": "float"},
        measurement_scalars={"voltage_V": "float"},
        measurement_arrays={"voltage_V_array": 10},
    )
    assert s.sweep_columns == {"field_T": "float"}
    assert s.measurement_scalars == {"voltage_V": "float"}
    assert s.measurement_arrays == {"voltage_V_array": 10}
    assert s.loop_shape == (1, 1)  # default: no reading loop


def test_dataschema_loop_shape_explicit():
    s = DataSchema(
        sweep_columns={}, measurement_scalars={}, measurement_arrays={},
        loop_shape=(2, 3),
    )
    assert s.loop_shape == (2, 3)


def test_dataschema_loop_shape_must_be_two_ints():
    with pytest.raises(TypeError, match="loop_shape"):
        DataSchema(
            sweep_columns={}, measurement_scalars={}, measurement_arrays={},
            loop_shape=(2,),
        )


def test_dataschema_loop_shape_entries_must_be_positive():
    with pytest.raises(ValueError, match="loop_shape"):
        DataSchema(
            sweep_columns={}, measurement_scalars={}, measurement_arrays={},
            loop_shape=(0, 1),
        )


def test_dataschema_bad_dtype():
    with pytest.raises(ValueError, match="sweep_columns"):
        DataSchema(
            sweep_columns={"field_T": "complex"},
            measurement_scalars={},
            measurement_arrays={},
        )


def test_dataschema_measurement_scalar_bad_dtype():
    with pytest.raises(ValueError, match="measurement_scalars"):
        DataSchema(
            sweep_columns={},
            measurement_scalars={"voltage_V": "complex"},
            measurement_arrays={},
        )


def test_dataschema_array_length_must_be_positive():
    with pytest.raises(ValueError, match="measurement_arrays"):
        DataSchema(sweep_columns={}, measurement_scalars={}, measurement_arrays={"v": 0})


def test_dataschema_array_length_bool_rejected():
    with pytest.raises(TypeError, match="measurement_arrays"):
        DataSchema(sweep_columns={}, measurement_scalars={}, measurement_arrays={"v": True})


def test_dataschema_empty_column_name():
    with pytest.raises(ValueError, match="sweep_columns"):
        DataSchema(sweep_columns={"": "float"}, measurement_scalars={}, measurement_arrays={})


def test_dataschema_frozen():
    s = DataSchema(sweep_columns={}, measurement_scalars={}, measurement_arrays={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.measurement_arrays = {}  # type: ignore[misc]


def test_dataschema_defensive_copy():
    arrays = {"v": 10}
    scalars = {"m": "float"}
    s = DataSchema(sweep_columns={}, measurement_scalars=scalars, measurement_arrays=arrays)
    arrays["w"] = 5
    scalars["n"] = "int"
    assert "w" not in s.measurement_arrays
    assert "n" not in s.measurement_scalars


def test_dataschema_measurement_blocks_default_empty():
    s = DataSchema(sweep_columns={}, measurement_scalars={}, measurement_arrays={})
    assert s.measurement_blocks == {}


def test_dataschema_measurement_blocks_happy():
    s = DataSchema(
        sweep_columns={},
        measurement_scalars={},
        measurement_arrays={},
        measurement_blocks={"raw_channels_block": (5, 44)},
    )
    assert s.measurement_blocks == {"raw_channels_block": (5, 44)}


def test_dataschema_measurement_blocks_must_be_dict():
    with pytest.raises(TypeError, match="measurement_blocks"):
        DataSchema(
            sweep_columns={}, measurement_scalars={}, measurement_arrays={},
            measurement_blocks="nope",
        )


def test_dataschema_measurement_blocks_shape_must_be_two_ints():
    with pytest.raises(TypeError, match="measurement_blocks"):
        DataSchema(
            sweep_columns={}, measurement_scalars={}, measurement_arrays={},
            measurement_blocks={"b": (5,)},
        )


def test_dataschema_measurement_blocks_shape_entries_must_be_positive():
    with pytest.raises(ValueError, match="measurement_blocks"):
        DataSchema(
            sweep_columns={}, measurement_scalars={}, measurement_arrays={},
            measurement_blocks={"b": (0, 44)},
        )


def test_dataschema_measurement_blocks_shape_bool_rejected():
    with pytest.raises(TypeError, match="measurement_blocks"):
        DataSchema(
            sweep_columns={}, measurement_scalars={}, measurement_arrays={},
            measurement_blocks={"b": (True, 44)},
        )


def test_dataschema_measurement_blocks_empty_name_rejected():
    with pytest.raises(ValueError, match="measurement_blocks"):
        DataSchema(
            sweep_columns={}, measurement_scalars={}, measurement_arrays={},
            measurement_blocks={"": (5, 44)},
        )


def test_dataschema_measurement_blocks_defensive_copy():
    blocks = {"b": (5, 44)}
    s = DataSchema(
        sweep_columns={}, measurement_scalars={}, measurement_arrays={},
        measurement_blocks=blocks,
    )
    blocks["c"] = (1, 1)
    assert "c" not in s.measurement_blocks


# ── DataSchema.validate ───────────────────────────────────────────────────────


def test_validate_passes():
    s = DataSchema(
        sweep_columns={"field_T": "float"},
        measurement_scalars={"voltage_V": "float"},
        measurement_arrays={"voltage_V_array": 3},
    )
    datapoint = {
        "field_T": 1.0,
        "voltage_V": [[0.2]],  # loop_shape (1, 1)
        "voltage_V_array": [[[0.1, 0.2, 0.3]]],
    }
    assert s.validate(datapoint) is None


def test_validate_int_dtype_accepts_int():
    s = DataSchema(sweep_columns={"n": "int"}, measurement_scalars={}, measurement_arrays={})
    assert s.validate({"n": 5}) is None


def test_validate_missing_key():
    s = DataSchema(sweep_columns={"field_T": "float"}, measurement_scalars={}, measurement_arrays={})
    with pytest.raises(DataSchemaError, match="missing declared key 'field_T'"):
        s.validate({})


def test_validate_extra_key():
    s = DataSchema(sweep_columns={"field_T": "float"}, measurement_scalars={}, measurement_arrays={})
    with pytest.raises(DataSchemaError, match="extra undeclared key 'junk'"):
        s.validate({"field_T": 1.0, "junk": 5})


def test_validate_wrong_array_length():
    s = DataSchema(sweep_columns={}, measurement_scalars={}, measurement_arrays={"voltage_V_array": 3})
    with pytest.raises(DataSchemaError, match="voltage_V_array"):
        s.validate({"voltage_V_array": [[[1, 2]]]})


def test_validate_array_no_length():
    s = DataSchema(sweep_columns={}, measurement_scalars={}, measurement_arrays={"voltage_V_array": 3})
    with pytest.raises(DataSchemaError, match="does not match shape"):
        s.validate({"voltage_V_array": 42})


def test_validate_wrong_scalar_type():
    s = DataSchema(sweep_columns={"field_T": "float"}, measurement_scalars={}, measurement_arrays={})
    with pytest.raises(DataSchemaError, match="field_T"):
        s.validate({"field_T": "high"})


def test_validate_block_passes():
    """No reading loop (loop_shape default (1, 1)): a block is stored bare (rows, cols)."""
    s = DataSchema(
        sweep_columns={},
        measurement_scalars={},
        measurement_arrays={},
        measurement_blocks={"raw_channels_block": (2, 3)},
    )
    datapoint = {
        "raw_channels_block": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # bare (rows, cols), no loop axis
    }
    assert s.validate(datapoint) is None


def test_validate_block_missing_key():
    s = DataSchema(
        sweep_columns={}, measurement_scalars={}, measurement_arrays={},
        measurement_blocks={"raw_channels_block": (2, 3)},
    )
    with pytest.raises(DataSchemaError, match="missing declared key 'raw_channels_block'"):
        s.validate({})


def test_validate_block_wrong_shape():
    s = DataSchema(
        sweep_columns={}, measurement_scalars={}, measurement_arrays={},
        measurement_blocks={"raw_channels_block": (2, 3)},
    )
    with pytest.raises(DataSchemaError, match="raw_channels_block"):
        s.validate({"raw_channels_block": [[1.0, 2.0], [3.0, 4.0]]})  # 2 channels, wants 3


def test_validate_block_wrong_row_count():
    s = DataSchema(
        sweep_columns={}, measurement_scalars={}, measurement_arrays={},
        measurement_blocks={"raw_channels_block": (2, 3)},
    )
    with pytest.raises(DataSchemaError, match="does not match shape"):
        s.validate({"raw_channels_block": [[1.0, 2.0, 3.0]]})  # only 1 row, wants 2


def test_validate_block_with_loop_axis():
    s = DataSchema(
        sweep_columns={}, measurement_scalars={}, measurement_arrays={},
        measurement_blocks={"raw_channels_block": (2, 3)},
        loop_shape=(2, 1),
    )
    datapoint = {
        "raw_channels_block": [
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
            [[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
        ],
    }
    assert s.validate(datapoint) is None


def test_validate_bool_scalar_rejected():
    s = DataSchema(sweep_columns={"field_T": "float"}, measurement_scalars={}, measurement_arrays={})
    with pytest.raises(DataSchemaError, match="field_T"):
        s.validate({"field_T": True})


def test_validate_int_dtype_rejects_float():
    s = DataSchema(sweep_columns={"n": "int"}, measurement_scalars={}, measurement_arrays={})
    with pytest.raises(DataSchemaError, match="not an int"):
        s.validate({"n": 5.0})


def test_validate_reports_multiple_problems_together():
    s = DataSchema(
        sweep_columns={"field_T": "float"},
        measurement_scalars={},
        measurement_arrays={"voltage_V_array": 3},
    )
    with pytest.raises(DataSchemaError) as excinfo:
        s.validate({"field_T": "bad", "voltage_V_array": [[[1, 2]]], "junk": 1})
    msg = str(excinfo.value)
    assert "field_T" in msg
    assert "voltage_V_array" in msg
    assert "junk" in msg


# ── DataSchema.validate — loop axis (measurement_scalars/measurement_arrays) ──


def test_validate_measurement_scalar_matches_loop_shape():
    s = DataSchema(
        sweep_columns={}, measurement_scalars={"voltage_V": "float"},
        measurement_arrays={}, loop_shape=(2, 2),
    )
    grid = [[1.0, 2.0], [3.0, 4.0]]
    assert s.validate({"voltage_V": grid}) is None


def test_validate_measurement_scalar_wrong_loop_shape():
    s = DataSchema(
        sweep_columns={}, measurement_scalars={"voltage_V": "float"},
        measurement_arrays={}, loop_shape=(2, 2),
    )
    with pytest.raises(DataSchemaError, match="loop shape"):
        s.validate({"voltage_V": [[1.0, 2.0]]})  # missing the second loop1 row


def test_validate_measurement_scalar_leaf_type_checked():
    s = DataSchema(
        sweep_columns={}, measurement_scalars={"voltage_V": "float"},
        measurement_arrays={}, loop_shape=(1, 2),
    )
    with pytest.raises(DataSchemaError, match="non-real-number"):
        s.validate({"voltage_V": [[1.0, "bad"]]})


def test_validate_measurement_scalar_int_dtype_leaf_rejects_float():
    s = DataSchema(
        sweep_columns={}, measurement_scalars={"n_valid": "int"},
        measurement_arrays={}, loop_shape=(1, 2),
    )
    with pytest.raises(DataSchemaError, match="non-int value"):
        s.validate({"n_valid": [[5, 5.0]]})


def test_validate_measurement_array_matches_loop_shape_and_length():
    s = DataSchema(
        sweep_columns={}, measurement_scalars={},
        measurement_arrays={"voltage_V_array": 3}, loop_shape=(2, 1),
    )
    grid = [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]
    assert s.validate({"voltage_V_array": grid}) is None


def test_validate_measurement_array_wrong_loop_shape():
    s = DataSchema(
        sweep_columns={}, measurement_scalars={},
        measurement_arrays={"voltage_V_array": 3}, loop_shape=(2, 1),
    )
    with pytest.raises(DataSchemaError, match="does not match shape"):
        s.validate({"voltage_V_array": [[[1.0, 2.0, 3.0]]]})  # missing loop1 index 1


# ── EnvelopeBound / ExperimentEnvelope ──────────────────────────────────────────

class TestEnvelopeBound:
    def test_requires_at_least_one_bound(self):
        with pytest.raises(ValueError):
            EnvelopeBound()

    def test_rejects_min_above_max(self):
        with pytest.raises(ValueError):
            EnvelopeBound(min_value=2.0, max_value=1.0)

    def test_rejects_non_numeric_and_non_finite(self):
        with pytest.raises(TypeError):
            EnvelopeBound(max_value=True)
        with pytest.raises(ValueError):
            EnvelopeBound(max_value=float("inf"))

    def test_violation_messages_and_pass(self):
        bound = EnvelopeBound(min_value=-0.5, max_value=0.5)
        assert bound.violation(0.0) is None
        assert "below the session minimum" in bound.violation(-1.0)
        assert "above the session maximum" in bound.violation(1.0)
        # Non-numeric state values can never trip a numeric envelope.
        assert bound.violation("HOLDING") is None


class TestExperimentEnvelope:
    def test_rejects_empty_bounds(self):
        with pytest.raises(ValueError):
            ExperimentEnvelope(bounds={})

    def test_rejects_wrong_value_type(self):
        with pytest.raises(TypeError):
            ExperimentEnvelope(bounds={"magnet_z": (0.0, 1.0)})

    def test_check_target(self):
        env = ExperimentEnvelope(
            bounds={"magnet_z": EnvelopeBound(min_value=-2.0, max_value=2.0)}
        )
        assert env.check_target("magnet_z", 1.0) is None
        assert env.check_target("other_vi", 99.0) is None  # unbounded VI
        message = env.check_target("magnet_z", 3.0)
        assert "session envelope" in message and "magnet_z" in message

    def test_from_dict_builds_the_typed_envelope(self):
        """The dict form a JSON-speaking client sends becomes the typed value."""
        env = ExperimentEnvelope.from_dict(
            {
                "magnet_z": {"min_value": -2.0, "max_value": 2.0, "state_key": "field_T"},
                "temperature_sample": {"max_value": 320.0},
            }
        )
        assert env.bounds["magnet_z"] == EnvelopeBound(
            min_value=-2.0, max_value=2.0, state_key="field_T"
        )
        assert env.bounds["temperature_sample"] == EnvelopeBound(max_value=320.0)

    def test_from_dict_is_strict_about_malformed_input(self):
        """A malformed envelope raises rather than silently narrowing to junk."""
        with pytest.raises(TypeError):
            ExperimentEnvelope.from_dict({"magnet_z": 2.0})
        with pytest.raises(TypeError):
            ExperimentEnvelope.from_dict("magnet_z")
        with pytest.raises(ValueError):
            ExperimentEnvelope.from_dict({})
        with pytest.raises(ValueError):
            ExperimentEnvelope.from_dict({"magnet_z": {}})  # no bound at all

    def test_check_state_uses_state_key_and_skips_missing(self):
        env = ExperimentEnvelope(
            bounds={
                "temperature_sample": EnvelopeBound(
                    min_value=4.0, state_key="temperature"
                ),
                "magnet_z": EnvelopeBound(max_value=2.0),  # no state_key: skipped
            }
        )
        violations = env.check_state(
            {"temperature_sample": {"temperature": 2.0}, "magnet_z": {"get_field": 9.0}}
        )
        assert len(violations) == 1
        assert "temperature_sample" in violations[0]
        # VI or key absent from the snapshot -> staleness, not a violation.
        assert env.check_state({}) == []
        assert env.check_state({"temperature_sample": {}}) == []


# ── params_digest (the Params digest standard) ────────────────────────────────
#
# The digest is what makes a confirmation record evidence: two records agree
# about the parameters exactly when their digests match. So the tests are
# written against that promise — same parameters, same digest, whatever order
# the mapping happens to be in; different parameters, different digest — plus
# the canonicalisation the docstring publishes, which callers on the far side
# of a JSON file depend on being reproducible.


def test_params_digest_ignores_key_order():
    """Two mappings with the same pairs digest identically, however they were built."""
    one = params_digest({"field_T": 1.5, "averaging": 4, "label": "sweep"})
    other = params_digest({"label": "sweep", "averaging": 4, "field_T": 1.5})
    assert one == other


def test_params_digest_is_a_lowercase_sha256_hex_string():
    digest = params_digest({"field_T": 1.5})
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(c in "0123456789abcdef" for c in digest)


def test_params_digest_matches_the_published_canonicalisation():
    """The canonical text is part of the standard, not an implementation detail."""
    params = {"b": 2, "a": 1.5}
    canonical = '{"a":1.5,"b":2}'
    assert params_digest(params) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_params_digest_treats_none_and_empty_as_the_same_fact():
    """"No parameters" is one fact, so it cannot have two digests."""
    assert params_digest(None) == params_digest({})


def test_params_digest_separates_a_changed_value_a_renamed_key_and_a_new_key():
    base = params_digest({"field_T": 1.5})
    assert params_digest({"field_T": 1.6}) != base
    assert params_digest({"field_t": 1.5}) != base
    assert params_digest({"field_T": 1.5, "rate": 0.1}) != base


def test_params_digest_separates_a_float_from_the_equal_int():
    """Floats are written as repr gives them, so 1.0 is not the integer 1."""
    assert params_digest({"n": 1.0}) != params_digest({"n": 1})


def test_params_digest_degrades_a_value_json_cannot_render_rather_than_raising():
    """A stray non-JSON value still yields a digest — accountability, not control flow."""

    class _Opaque:
        def __str__(self) -> str:
            return "opaque"

    assert params_digest({"x": _Opaque()}) == params_digest({"x": "opaque"})


def test_params_digest_survives_a_round_trip_through_json():
    """The digest a record stores must still match after the record is reread."""
    params = {"field_T": 1.5, "points": 21, "note": "µ-metal shield"}
    assert params_digest(json.loads(json.dumps(params))) == params_digest(params)


# ── ImageBlock (the image-block standard) ─────────────────────────────────────


def test_image_block_happy_and_shape():
    block = ImageBlock(height_px=4, width_px=6, unit="counts", description="a frame")
    assert block.shape == (4, 6)
    assert block.unit == "counts"
    assert block.description == "a frame"
    assert ImageBlock(1, 1, "V").description == ""


@pytest.mark.parametrize(
    "kwargs, exc, match",
    [
        ({"height_px": 0, "width_px": 6, "unit": "counts"}, ValueError, "height_px"),
        ({"height_px": 4, "width_px": -1, "unit": "counts"}, ValueError, "width_px"),
        ({"height_px": 4.0, "width_px": 6, "unit": "counts"}, TypeError, "height_px"),
        ({"height_px": True, "width_px": 6, "unit": "counts"}, TypeError, "height_px"),
        ({"height_px": 4, "width_px": 6, "unit": ""}, ValueError, "unit"),
        ({"height_px": 4, "width_px": 6, "unit": 3}, TypeError, "unit"),
        ({"height_px": 4, "width_px": 6, "unit": "counts", "description": 1}, TypeError, "description"),
    ],
    ids=["zero-height", "negative-width", "float-height", "bool-height", "empty-unit", "non-str-unit", "non-str-description"],
)
def test_image_block_rejects_bad_declarations(kwargs, exc, match):
    with pytest.raises(exc, match=f"ImageBlock.{match}"):
        ImageBlock(**kwargs)


def test_image_block_is_frozen():
    block = ImageBlock(4, 6, "counts")
    with pytest.raises(dataclasses.FrozenInstanceError):
        block.height_px = 8  # type: ignore[misc]
