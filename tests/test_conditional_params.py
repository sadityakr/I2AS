"""The conditional-parameter standard: guarded form blocks and their resolver.

Covers ``When``/``ConditionalGroup`` declaration validation, ``resolve_form``'s
staged cascade and defaulting, ``form_selections``, and ``validate_form``'s
three faults. The procedure-side declarations that use them are exercised in
``test_l4_procedure.py`` and ``test_new_procedures.py``.
"""

from __future__ import annotations

import pytest

from i2as.core.plan import (
    ConditionalGroup,
    ParamSpec,
    When,
    resolve_form,
    validate_form,
)


def _select(default: str, *values: str, structural: bool = True) -> ParamSpec:
    """Build a structural drop-down over *values*."""
    return ParamSpec(
        type=str,
        default=default,
        choices={value: value for value in values},
        structural=structural,
    )


# A two-level cascade in miniature, the shape SweepMeasureProcedure declares:
# the method selector decides both which parameter group appears AND which
# options the loop selector offers; the loop selection then decides which
# value field appears beside it.
CASCADE = (
    ConditionalGroup(
        key="method",
        title="Method",
        params={"vi": _select("dc", "dc", "lockin")},
    ),
    ConditionalGroup(
        key="loop",
        title="Loop",
        params={"loop_param": _select("", "", "current")},
        when=When({"vi": ("dc",)}),
    ),
    ConditionalGroup(
        key="loop",
        title="Loop",
        params={"loop_values": ParamSpec(type=str, default="", structural=True)},
        when=When({"vi": ("dc",), "loop_param": ("current",)}),
    ),
    ConditionalGroup(
        key="loop",
        title="Loop",
        params={"loop_param": _select("", "", "amplitude")},
        when=When({"vi": ("lockin",)}),
    ),
    ConditionalGroup(
        key="params:dc",
        title="DC",
        params={"current_A": ParamSpec(type=float, default=1e-6)},
        when=When({"vi": ("dc",)}),
    ),
    ConditionalGroup(
        key="params:lockin",
        title="Lock-in",
        params={"amplitude_V": ParamSpec(type=float, default=0.1)},
        when=When({"vi": ("lockin",)}),
    ),
)


class TestWhen:
    """The guard: equality against an enumerated set, and nothing more."""

    def test_empty_guard_always_holds(self) -> None:
        assert When().holds({})
        assert When().holds({"anything": 1})

    def test_and_across_parameters_or_within_one(self) -> None:
        guard = When({"a": ("x", "y"), "b": (1,)})
        assert guard.holds({"a": "x", "b": 1})
        assert guard.holds({"a": "y", "b": 1})
        assert not guard.holds({"a": "z", "b": 1})
        assert not guard.holds({"a": "x", "b": 2})

    def test_absent_parameter_fails_rather_than_passes(self) -> None:
        assert not When({"a": ("x",)}).holds({})

    def test_a_bool_never_satisfies_a_numeric_choice(self) -> None:
        # True == 1 in Python; a checkbox answer must not satisfy a guard
        # written for a numeric selection, nor the reverse.
        assert not When({"a": (1,)}).holds({"a": True})
        assert not When({"a": (True,)}).holds({"a": 1})

    def test_a_guard_nothing_can_satisfy_is_refused(self) -> None:
        with pytest.raises(ValueError, match="accepts no value"):
            When({"a": ()})

    def test_a_bare_string_is_not_a_value_list(self) -> None:
        # When({"a": "xy"}) would silently accept "x" and "y" as two values.
        with pytest.raises(TypeError, match="list or tuple"):
            When({"a": "xy"})


class TestResolveForm:
    """The staged cascade, its defaulting, and the merge into ParamGroups."""

    def test_no_selections_lands_on_the_fully_defaulted_form(self) -> None:
        groups = {group.key: list(group.params) for group in resolve_form(CASCADE)}
        assert groups == {
            "method": ["vi"],
            "loop": ["loop_param"],
            "params:dc": ["current_A"],
        }

    def test_a_second_level_guard_resolves_after_the_first(self) -> None:
        groups = {
            group.key: list(group.params)
            for group in resolve_form(CASCADE, {"loop_param": "current"})
        }
        # loop_values is guarded on BOTH vi (defaulted this pass) and
        # loop_param (supplied), so it can only appear if the resolver
        # re-evaluates after filling vi's default.
        assert groups["loop"] == ["loop_param", "loop_values"]

    def test_switching_the_top_selection_switches_the_whole_cascade(self) -> None:
        groups = {
            group.key: list(group.params)
            for group in resolve_form(CASCADE, {"vi": "lockin"})
        }
        assert "params:dc" not in groups
        assert groups["params:lockin"] == ["amplitude_V"]
        # The loop selector is a guarded copy per VI, so its CHOICES changed
        # without a second mechanism for dependent choices.
        loop_group = next(g for g in resolve_form(CASCADE, {"vi": "lockin"}) if g.key == "loop")
        assert list(loop_group.params["loop_param"].choices or {}) == ["", "amplitude"]

    def test_blocks_sharing_a_key_merge_in_declaration_order(self) -> None:
        groups = resolve_form(CASCADE, {"vi": "dc", "loop_param": "current"})
        loop = next(group for group in groups if group.key == "loop")
        assert list(loop.params) == ["loop_param", "loop_values"]
        assert loop.title == "Loop"

    def test_group_order_follows_first_declaration_of_each_key(self) -> None:
        keys = [group.key for group in resolve_form(CASCADE)]
        assert keys == ["method", "loop", "params:dc"]

    def test_an_explicit_answer_is_never_overwritten_by_a_default(self) -> None:
        groups = resolve_form(CASCADE, {"vi": "lockin"})
        assert {group.key for group in groups} >= {"params:lockin"}

    def test_two_active_blocks_claiming_one_parameter_are_refused(self) -> None:
        clashing = (
            ConditionalGroup(key="g", title="G", params={"x": ParamSpec(type=int, default=1)}),
            ConditionalGroup(key="g", title="G", params={"x": ParamSpec(type=int, default=2)}),
        )
        with pytest.raises(ValueError, match="both declare the parameter 'x'"):
            resolve_form(clashing)

    def test_identical_blocks_keep_their_separate_places(self) -> None:
        # Frozen dataclasses compare by value, so a resolver ordering blocks by
        # `.index()` would collapse these two into one position.
        twins = (
            ConditionalGroup(key="a", title="A", params={"x": ParamSpec(type=int, default=1)}),
            ConditionalGroup(key="b", title="B", params={"y": ParamSpec(type=int, default=1)}),
            ConditionalGroup(key="a", title="A", params={"x2": ParamSpec(type=int, default=1)}),
        )
        assert [group.key for group in resolve_form(twins)] == ["a", "b"]
        assert list(resolve_form(twins)[0].params) == ["x", "x2"]


class TestValidateForm:
    """The faults a declared form can be checked for, which a method cannot."""

    def test_a_sound_declaration_has_no_errors(self) -> None:
        assert validate_form(CASCADE) == []

    def test_a_guard_on_an_undeclared_parameter_is_reported(self) -> None:
        blocks = (
            ConditionalGroup(
                key="g",
                title="G",
                params={"x": ParamSpec(type=int, default=1)},
                when=When({"ghost": ("v",)}),
            ),
        )
        errors = validate_form(blocks)
        assert any("guarded on 'ghost'" in error for error in errors)

    def test_a_guard_cycle_is_reported_rather_than_silently_dropping_blocks(self) -> None:
        blocks = (
            ConditionalGroup(
                key="a",
                title="A",
                params={"x": _select("v", "v")},
                when=When({"y": ("v",)}),
            ),
            ConditionalGroup(
                key="b",
                title="B",
                params={"y": _select("v", "v")},
                when=When({"x": ("v",)}),
            ),
        )
        # resolve_form does not hang on a cycle, it quietly returns nothing —
        # which is exactly why the declaration must be checkable.
        assert resolve_form(blocks) == []
        errors = validate_form(blocks)
        assert len(errors) == 2
        assert all("guard cycle" in error for error in errors)

    def test_an_empty_block_is_reported(self) -> None:
        errors = validate_form((ConditionalGroup(key="g", title="G", params={}),))
        assert any("declares no parameters" in error for error in errors)
