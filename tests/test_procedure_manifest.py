"""Procedures on the declaration surface: the manifest and the gateway tools.

An agent must be able to discover what a station can be asked to RUN, and with
which parameters, from the declaration alone — the same way it discovers what
each instrument can be told to do. These tests cover the whole path: a
procedure class declares its form, ``core.procedure_catalog`` renders it, the
Station publishes it in ``StationInfo``, the manifest carries it, and the
gateway's ``list_procedures`` / ``describe_procedure`` resolve it from the
mirrored snapshot with no Station of their own.
"""

from __future__ import annotations

import pytest

from i2as.core.capability_manifest import build_manifest, validate_manifest
from i2as.core.plan import blocks_from_json, resolve_form
from i2as.core.procedure_catalog import build_procedure_infos, discover_run_catalog
from i2as.core.station import build_station
from i2as.session.gateway.tools import (
    SESSION_TOOL_FUNCTIONS,
    ToolContext,
    ToolError,
    render_tools,
)

CONFIG = "i2as/configs/sim_cryostat"


@pytest.fixture(scope="module")
def station():
    """A built sim station carrying the shipped procedures' declarations."""
    built = build_station(CONFIG)
    built.declare_procedures(build_procedure_infos(built, discover_run_catalog()))
    return built


@pytest.fixture(scope="module")
def manifest(station):
    """The capability manifest of that station."""
    return build_manifest(station)


@pytest.fixture(scope="module")
def declared(station):
    """The unabridged procedure declarations, by class name.

    The manifest carries only a row per procedure; the form itself lives in
    ``StationInfo.procedures``, which is what ``describe_procedure`` answers
    from, so that is where the form's own invariants are checked.
    """
    return {entry.class_name: entry for entry in station.station_info().procedures}


@pytest.fixture()
def context(station):
    """A tool context whose only station is the mirrored declaration."""
    snapshot = station.station_info()
    return ToolContext(station_source=lambda: snapshot)


class TestCatalog:
    """Discovery and rendering, in core so every client shares one walk."""

    def test_every_shipped_procedure_is_catalogued(self) -> None:
        catalog = discover_run_catalog()
        assert {"FieldSweep", "TemperatureSweep", "TimeSeries"} <= set(catalog)

    def test_the_catalog_key_is_the_name_a_run_travels_as(self) -> None:
        for class_name, cls in discover_run_catalog().items():
            assert class_name == cls.__name__

    def test_intermediate_bases_are_not_offered(self) -> None:
        # SweepMeasureProcedure carries no `name`: it is a base, not a run.
        assert "SweepMeasureProcedure" not in discover_run_catalog()
        assert "BaseProcedure" not in discover_run_catalog()


class TestStationDeclaration:
    """What the Station publishes, and what it deliberately does not do."""

    def test_station_info_carries_every_procedure(self, station) -> None:
        declared = {entry.class_name for entry in station.station_info().procedures}
        assert declared == set(discover_run_catalog())

    def test_a_station_given_no_catalog_declares_no_procedures(self) -> None:
        # The Station never discovers procedures itself — contract C4 keeps it
        # below them — so one nobody hands a catalog to simply has none.
        assert build_station(CONFIG).station_info().procedures == ()

    def test_declaring_procedures_rebuilds_the_snapshot(self, station) -> None:
        before = station.station_info().seq
        station.declare_procedures(station.station_info().procedures)
        assert station.station_info().seq > before

    def test_the_declaration_survives_a_json_round_trip(self, station) -> None:
        from i2as.core.events import StationInfo

        info = station.station_info()
        back = StationInfo.from_json(info.to_json())
        assert [p.class_name for p in back.procedures] == [
            p.class_name for p in info.procedures
        ]
        assert back.procedures[0].form == info.procedures[0].form


class TestManifest:
    """The procedures section of the capability manifest."""

    def test_the_manifest_conforms_to_its_schema(self, manifest) -> None:
        assert validate_manifest(manifest) == []

    def test_every_procedure_appears(self, manifest) -> None:
        assert {entry["class_name"] for entry in manifest["procedures"]} == set(
            discover_run_catalog()
        )

    def test_a_row_names_the_procedure_without_carrying_its_form(
        self, manifest
    ) -> None:
        # The manifest is an index: a reader takes it in whole to see what the
        # station can be asked to do. Every procedure's full conditional form
        # would be several times the size of every instrument declaration
        # combined, to describe forms the reader has not chosen a procedure
        # from yet.
        for entry in manifest["procedures"]:
            assert "form" not in entry
            assert entry["class_name"]

    def test_the_row_is_enough_to_choose_a_procedure(self, manifest) -> None:
        field_sweep = _procedure(manifest, "FieldSweep")
        assert field_sweep["sweep_axis"]["key"] == "field"
        assert "field_vi" in field_sweep["roles"]
        assert field_sweep["data_keys"]["default_x"] == "field_T"

    def test_the_form_covers_every_parameter_this_rack_can_fill(
        self, station, declared
    ) -> None:
        # Everything a caller must supply has to be in the form. The one
        # exception is an OPTIONAL role no instrument on this rack can fill
        # (FieldImaging's stage axes on a cryostat with no stage): the role
        # discovery standard drops it rather than offering a drop-down with
        # nothing in it, and the run proceeds without positioning a stage.
        catalog = discover_run_catalog()
        for class_name, entry in declared.items():
            if entry.availability:
                continue
            cls = catalog[class_name]
            offered = {
                param["name"] for block in entry.form for param in block.params
            }
            unfillable = {
                name
                for name, role in cls.role_parameters.items()
                if not role.candidates(station)
            }
            missing = set(cls.parameters) - offered - unfillable
            assert not missing, f"{class_name} omits {sorted(missing)}"

    def test_a_required_role_this_rack_cannot_fill_is_flagged(self, station) -> None:
        # A required role with no candidate refuses the run at construction, so
        # the declaration must say so rather than look complete. Checked by
        # making the one required role of a real procedure uncoverable.
        from i2as.core.procedure_catalog import build_procedure_infos

        catalog = {"FieldSweep": discover_run_catalog()["FieldSweep"]}
        empty_roles = dict(catalog["FieldSweep"].role_parameters)

        # Deliberately unnamed: discovery walks the live subclass tree, so a
        # NAMED subclass defined in a test would join every later caller's
        # catalog. build_procedure_infos takes an explicit catalog, so this
        # still renders.
        class NoMagnet(catalog["FieldSweep"]):  # type: ignore[misc, valid-type]
            name = ""
            role_parameters = {
                **empty_roles,
                "field_vi": type(empty_roles["field_vi"])(
                    candidates=lambda _station: [],
                    description="a role nothing can fill",
                ),
            }

        info = build_procedure_infos(station, {"NoMagnet": NoMagnet})[0]
        assert "missing_role:field_vi" in info.availability

    def test_the_sweep_range_is_part_of_the_declaration(self, declared) -> None:
        # The gap this closed: get_param_groups() excluded the sweep-axis
        # parameters because the GUI draws them with its own widget, so the
        # only complete description of FieldSweep omitted the field range it
        # sweeps. An agent cannot run what it cannot see.
        offered = {
            param["name"]
            for block in declared["FieldSweep"].form
            for param in block.params
        }
        assert {"field_start", "field_end", "field_steps"} <= offered

    def test_a_guarded_block_names_the_parameter_that_opens_it(self, declared) -> None:
        blocks = declared["FieldSweep"].form
        guarded = [block for block in blocks if block.when]
        assert guarded, "the generic sweep procedure's form is conditional"
        introduced = {
            param["name"] for block in blocks for param in block.params
        }
        for block in guarded:
            assert set(block.when) <= introduced


class TestGatewayTools:
    """Discovery and resolution from the mirrored snapshot, with no Station."""

    def test_both_tools_are_on_the_rendered_surface(self, station) -> None:
        names = {tool.name for tool in render_tools(station.station_info())}
        assert {"list_procedures", "describe_procedure"} <= names

    def test_list_procedures_names_what_run_procedure_takes(self, context) -> None:
        listed = SESSION_TOOL_FUNCTIONS["list_procedures"]({}, context)
        assert {entry["procedure"] for entry in listed["procedures"]} == set(
            discover_run_catalog()
        )

    def test_describe_returns_the_default_form(self, context) -> None:
        described = SESSION_TOOL_FUNCTIONS["describe_procedure"](
            {"procedure": "FieldSweep"}, context
        )
        keys = [group["key"] for group in described["groups"]]
        assert "sweep_axis" in keys
        assert "measurement:dc_measurement" in keys

    def test_a_structural_choice_opens_the_form_it_gates(self, context) -> None:
        # The cascade an agent has to be able to walk: choosing a loop
        # parameter adds that parameter's values field, which does not exist
        # in the default form.
        describe = SESSION_TOOL_FUNCTIONS["describe_procedure"]
        default = describe({"procedure": "FieldSweep"}, context)
        assert "loop1_values" not in _params(default, "reading_loop")

        chosen = describe(
            {
                "procedure": "FieldSweep",
                "selections": {"loop1_parameter": "dc_measurement.current_A"},
            },
            context,
        )
        assert "loop1_values" in _params(chosen, "reading_loop")

    def test_the_reply_says_which_selections_it_stands_on(self, context) -> None:
        described = SESSION_TOOL_FUNCTIONS["describe_procedure"](
            {"procedure": "FieldSweep"}, context
        )
        # measurement_vi was never supplied, but the form rests on it, so an
        # agent is told the value it must send back to change anything under it.
        assert described["selections"]["measurement_vi"] == "dc_measurement"

    def test_structural_parameters_are_marked_as_such(self, context) -> None:
        described = SESSION_TOOL_FUNCTIONS["describe_procedure"](
            {"procedure": "FieldSweep"}, context
        )
        structural = {
            param["name"]
            for group in described["groups"]
            for param in group["params"]
            if param["structural"]
        }
        assert "measurement_vi" in structural
        assert "field_start" not in structural

    def test_an_unknown_procedure_is_refused_by_name(self, context) -> None:
        with pytest.raises(ToolError, match="unknown procedure"):
            SESSION_TOOL_FUNCTIONS["describe_procedure"](
                {"procedure": "NoSuchProcedure"}, context
            )

    def test_bad_selections_are_refused_rather_than_ignored(self, context) -> None:
        with pytest.raises(ToolError, match="selections must be an object"):
            SESSION_TOOL_FUNCTIONS["describe_procedure"](
                {"procedure": "FieldSweep", "selections": ["not", "a", "map"]},
                context,
            )

    def test_the_tools_answer_from_the_snapshot_alone(self, station) -> None:
        # The reason the form is declared rather than computed: a client that
        # holds only a deserialised snapshot — every out-of-process MCP agent —
        # resolves the same form as the GUI, with no Station anywhere.
        from i2as.core.events import StationInfo

        mirrored = StationInfo.from_json(station.station_info().to_json())
        context = ToolContext(station_source=lambda: mirrored)
        described = SESSION_TOOL_FUNCTIONS["describe_procedure"](
            {
                "procedure": "FieldSweep",
                "selections": {"loop1_parameter": "dc_measurement.current_A"},
            },
            context,
        )
        assert "loop1_values" in _params(described, "reading_loop")


def test_wire_and_gui_resolve_the_same_form(station, declared) -> None:
    """The GUI's groups and an agent's are one declaration, not two.

    The whole point of the standard: ``get_param_groups`` is now the GUI's
    filtered view of the same blocks the station publishes, so the two cannot
    drift. Everything the GUI renders must appear, identically, in what an
    agent resolves off the wire.
    """
    from i2as.procedures.field_sweep import FieldSweep

    selections = {"loop1_parameter": "dc_measurement.current_A"}
    gui = FieldSweep.get_param_groups(station, selections)
    wire = resolve_form(
        blocks_from_json(
            [block.to_json() for block in declared["FieldSweep"].form]
        ),
        selections,
    )

    from i2as.core.procedure import SWEEP_AXIS_GROUP_KEY

    assert [(g.key, list(g.params)) for g in gui] == [
        (g.key, list(g.params)) for g in wire if g.key != SWEEP_AXIS_GROUP_KEY
    ]


def _procedure(manifest, class_name: str) -> dict:
    """Return one procedure's entry from a manifest."""
    return next(
        entry for entry in manifest["procedures"] if entry["class_name"] == class_name
    )


def _params(described: dict, group_key: str) -> list[str]:
    """Return the parameter names of one group of a describe_procedure reply."""
    for group in described["groups"]:
        if group["key"] == group_key:
            return [param["name"] for param in group["params"]]
    return []
