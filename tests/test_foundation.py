"""Tests for exceptions.py and decorators.py."""

import pytest
from i2as.core.exceptions import (
    I2ASError,
    I2ASCommunicationError,
    I2ASSafetyError,
    I2ASConfigError,
)
from i2as.core.decorators import (
    monitored,
    control,
    get_monitored_methods,
    get_monitored_description,
    get_monitored_unit,
    get_ui_group,
    get_control_methods,
    get_control_panel,
    get_control_specs,
)
from i2as.core.plan import ParamSpec


# ── Exception tests ──────────────────────────────────────────────────

class TestExceptionHierarchy:
    """Verify the exception inheritance tree."""

    def test_base_exception_is_exception(self):
        assert issubclass(I2ASError, Exception)

    def test_communication_error_is_i2as_error(self):
        assert issubclass(I2ASCommunicationError, I2ASError)

    def test_safety_error_is_i2as_error(self):
        assert issubclass(I2ASSafetyError, I2ASError)

    def test_config_error_is_i2as_error(self):
        assert issubclass(I2ASConfigError, I2ASError)

    def test_communication_error_attributes(self):
        original = ValueError("VISA timeout")
        err = I2ASCommunicationError(
            "Lost connection to magnet_z",
            vi_name="magnet_z",
            original_error=original,
        )
        assert err.vi_name == "magnet_z"
        assert err.original_error is original
        assert "Lost connection" in str(err)

    def test_communication_error_defaults(self):
        err = I2ASCommunicationError("timeout")
        assert err.vi_name == ""
        assert err.original_error is None

    def test_safety_error_structured_fields_default_to_none(self):
        """A message-only raise site keeps working; the fields are optional."""
        err = I2ASSafetyError("quench detected")
        assert str(err) == "quench detected"
        assert (err.param, err.value, err.lo, err.hi, err.limit_name) == (
            None,
            None,
            None,
            None,
            None,
        )

    def test_safety_error_carries_a_structured_limit_rejection(self):
        """The limit wrapper's fields survive onto the exception."""
        err = I2ASSafetyError(
            "refused",
            param="current_A",
            value=0.05,
            lo=-1e-3,
            hi=1e-3,
            limit_name="max_current",
        )
        assert err.param == "current_A"
        assert err.value == 0.05
        assert err.lo == -1e-3
        assert err.hi == 1e-3
        assert err.limit_name == "max_current"

    def test_catch_all_i2as_errors(self):
        """Verify that catching I2ASError catches all subtypes."""
        with pytest.raises(I2ASError):
            raise I2ASCommunicationError("test")
        with pytest.raises(I2ASError):
            raise I2ASSafetyError("test")
        with pytest.raises(I2ASError):
            raise I2ASConfigError("test")


# ── Decorator tests ──────────────────────────────────────────────────

class TestMonitoredDecorator:
    """Verify @monitored marks methods correctly."""

    def test_marks_method(self):
        @monitored
        def temperature(self) -> float:
            return 4.2
        assert temperature._is_monitored is True

    def test_preserves_function_name(self):
        @monitored
        def temperature(self) -> float:
            return 4.2
        assert temperature.__name__ == "temperature"

    def test_callable(self):
        """Decorated method should still work."""
        class FakeVI:
            @monitored
            def temperature(self) -> float:
                return 4.2

        vi = FakeVI()
        assert vi.temperature() == 4.2


    def test_declaration_keywords_are_stored(self):
        """The parametrized form stores unit, description and group verbatim."""
        @monitored(unit="K", description="Sample temperature", group="loop")
        def temperature(self) -> float:
            return 4.2

        assert temperature._is_monitored is True
        assert get_monitored_unit(temperature) == "K"
        assert get_monitored_description(temperature) == "Sample temperature"
        assert get_ui_group(temperature) == "loop"

    def test_bare_form_declares_nothing(self):
        """The bare form still works; its unit is undeclared, not empty."""
        @monitored
        def temperature(self) -> float:
            return 4.2

        assert get_monitored_unit(temperature) is None
        assert get_monitored_description(temperature) == ""
        assert get_ui_group(temperature) == ""

    def test_dimensionless_unit_is_distinct_from_undeclared(self):
        @monitored(unit="", description="Heater mode")
        def heater_mode(self) -> str:
            return "AUTO"

        assert get_monitored_unit(heater_mode) == ""

    def test_rejects_non_string_unit(self):
        with pytest.raises(TypeError, match="unit"):
            @monitored(unit=1)  # type: ignore[arg-type]
            def temperature(self) -> float:
                return 4.2

    def test_rejects_empty_group(self):
        with pytest.raises(ValueError, match="group"):
            @monitored(unit="K", description="T", group="")
            def temperature(self) -> float:
                return 4.2

    def test_parametrized_form_stays_callable(self):
        class FakeVI:
            @monitored(unit="K", description="Sample temperature")
            def temperature(self) -> float:
                return 4.2

        assert FakeVI().temperature() == 4.2


class TestControlDecorator:
    """Verify @control marks methods and extracts parameters."""

    def test_group_tag_is_stored(self):
        @control(group="heater")
        def set_heater(self, output_pct: float) -> None:
            pass

        assert get_ui_group(set_heater) == "heater"

    def test_bare_control_has_no_group(self):
        @control
        def set_heater(self, output_pct: float) -> None:
            pass

        assert get_ui_group(set_heater) == ""

    def test_rejects_empty_group(self):
        with pytest.raises(ValueError, match="group"):
            @control(group="")
            def set_heater(self) -> None:
                pass

    def test_marks_method(self):
        @control
        def set_field(self, target_T: float):
            pass
        assert set_field._is_control is True

    def test_extracts_params(self):
        @control
        def set_field(self, target_T: float, rate: float = 0.5):
            pass
        params = set_field._control_params
        assert "target_T" in params
        assert params["target_T"]["type"] == float
        assert "rate" in params
        assert params["rate"]["default"] == 0.5

    def test_no_self_in_params(self):
        @control
        def set_field(self, target_T: float):
            pass
        assert "self" not in set_field._control_params

    def test_callable(self):
        class FakeVI:
            @control
            def set_field(self, target_T: float):
                self.field = target_T

        vi = FakeVI()
        vi.set_field(5.0)
        assert vi.field == 5.0


class TestControlSpecsAndPanel:
    """Verify the @control params= (ParamSpec) and panel= declarations."""

    def test_defaults_no_specs_panel_true(self):
        @control
        def set_field(self, target_T: float):
            pass
        assert get_control_specs(set_field) == {}
        assert get_control_panel(set_field) is True

    def test_specs_stored_and_retrievable(self):
        spec = ParamSpec(type=float, default=4.2, unit="K", min=1.5, max=300.0)

        @control(params={"target_K": spec})
        def set_temperature(self, target_K: float = 4.2):
            pass
        assert get_control_specs(set_temperature) == {"target_K": spec}

    def test_panel_false_stored(self):
        @control(panel=False)
        def set_pid(self, p: float, i: float, d: float):
            pass
        assert get_control_panel(set_pid) is False

    def test_params_must_match_signature(self):
        with pytest.raises(ValueError, match="must match exactly"):
            @control(params={"wrong_name": ParamSpec(type=float, default=0.0)})
            def set_temperature(self, target_K: float):
                pass

    def test_params_must_cover_every_signature_param(self):
        with pytest.raises(ValueError, match="must match exactly"):
            @control(params={"p": ParamSpec(type=float, default=0.0)})
            def set_pid(self, p: float, i: float):
                pass


class TestDiscoveryFunctions:
    """Verify get_monitored_methods and get_control_methods."""

    def _make_vi_class(self):
        class TestVI:
            @monitored
            def temperature(self) -> float:
                return 4.2

            @monitored
            def heater_output(self) -> float:
                return 30.0

            @control
            def set_temperature(self, target_K: float):
                pass

            def _private_method(self):
                pass

            def plain_method(self):
                pass

        return TestVI

    def test_get_monitored_methods(self):
        cls = self._make_vi_class()
        methods = get_monitored_methods(cls)
        assert "temperature" in methods
        assert "heater_output" in methods
        assert "set_temperature" not in methods
        assert "_private_method" not in methods
        assert "plain_method" not in methods

    def test_get_control_methods(self):
        cls = self._make_vi_class()
        methods = get_control_methods(cls)
        assert "set_temperature" in methods
        assert "target_K" in methods["set_temperature"]
        assert "temperature" not in methods

    def test_works_on_instances(self):
        cls = self._make_vi_class()
        instance = cls()
        assert "temperature" in get_monitored_methods(instance)
        assert "set_temperature" in get_control_methods(instance)
