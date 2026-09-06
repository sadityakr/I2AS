"""Tests for the station declaration snapshot and its JSON rendering.

Covers `Station.station_info()` (what is declared, in what order, and that
building it never touches an instrument) and
`i2as.core.capability_manifest` (the JSON rendering, its schema, its
validator and its command-line entry point).

The conformance suite carries the standard-shaped checks that must hold for
every VI and every buildable config; this file carries the behaviour those
standards rest on.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import i2as
from i2as.core.capability_manifest import (
    MANIFEST_SCHEMA,
    MANIFEST_SCHEMA_ID,
    build_manifest,
    main,
    validate_manifest,
)
from i2as.core.events import StationInfo
from i2as.core.config import read_tick_interval_ms
from i2as.core.station import Station, build_station

SIM_CONFIG = str(Path(i2as.__file__).parent / "configs" / "sim_cryostat")


@pytest.fixture
def station() -> Station:
    """A freshly built simulated station."""
    return build_station(SIM_CONFIG)


@pytest.fixture
def manifest(station: Station) -> dict:
    """The simulated station's capability manifest."""
    return build_manifest(station)


# ── StationInfo: what the Station declares ────────────────────────────────────


def test_station_info_carries_the_setup_identity_and_cadence(station: Station) -> None:
    """The snapshot names the setup it describes and the cadence it ticks at.

    Both are config properties, so they come from the config directory, not
    from anything in code.
    """
    info = station.station_info()
    assert info.setup == "sim_cryostat"
    assert info.tick_interval_s == pytest.approx(
        read_tick_interval_ms(SIM_CONFIG) / 1000.0
    )


def test_station_info_describes_every_configured_vi(station: Station) -> None:
    """Every configured VI appears, in config order, live or not."""
    info = station.station_info()
    names = [entry.name for entry in info.instruments]
    assert names == station.get_vi_names() + station.offline_vi_names()
    assert "magnet_z" in names


def test_station_info_renders_declarations_not_hand_written_text(
    station: Station,
) -> None:
    """A monitored field's unit and description come off its decorator."""
    magnet = _instrument(station.station_info(), "magnet_z")
    field = {entry.name: entry for entry in magnet.monitored}["magnet_field_T"]
    assert field.unit == "T"
    assert field.description
    assert field.returns == "float"


def test_station_info_carries_configured_limit_bounds(station: Station) -> None:
    """The limits are the config's values, resolved through control_limits."""
    magnet = _instrument(station.station_info(), "magnet_z")
    bounds = magnet.limits["set_field"]["target_T"]
    assert bounds["limit"] == "field_T"
    assert bounds["min"] is not None and bounds["max"] is not None
    assert bounds["min"] < 0.0 < bounds["max"]


def test_station_info_captures_instance_aware_control_choices(
    station: Station,
) -> None:
    """An instrument-reported default reaches the snapshot as the control's default.

    ``Lakeshore335SampleTemperatureControllerVI.control_param_specs()``
    injects the currently-assigned calibration curve per instance, so this is
    what proves the snapshot consults the instance hook rather than the raw
    decorator metadata.
    """
    station.get_state()  # populate the cache control_param_specs() reads
    controller = _instrument(station.station_info(), "temperature")
    curve = _control(controller, "set_curve").params[0]
    assert curve["name"] == "curve"
    assert curve["default"] == station.temperature.curve()


def test_station_info_lists_capabilities_in_declared_order(station: Station) -> None:
    """Readings read in source order, base class first — never alphabetically."""
    magnet = _instrument(station.station_info(), "magnet_z")
    names = [entry.name for entry in magnet.monitored]
    assert names != sorted(names), "declared order, not dir()'s alphabetical order"
    assert names.index("psu_current") < names.index("magnet_field_T")


def test_station_info_round_trips_through_json(station: Station) -> None:
    """The whole real snapshot survives to_json -> dumps -> loads -> from_json.

    The conformance suite round-trips a hand-built specimen; this does it for
    the shape a real station actually produces, nested types and all.
    """
    info = station.station_info()
    wire = json.loads(json.dumps(info.to_json()))
    assert StationInfo.from_json(wire) == info


# ── Rebuild on connect and disconnect ─────────────────────────────────────────


def test_station_info_is_cached_between_membership_changes(station: Station) -> None:
    """Repeated reads are free: the same snapshot object comes back."""
    assert station.station_info() is station.station_info()


def test_station_info_rebuilds_on_disconnect_and_connect(station: Station) -> None:
    """A disconnect and the reconnect each produce a fresh snapshot.

    The instrument stays in the declaration throughout — it still declares
    the same capabilities — and only its availability tags change, which is
    exactly why the snapshot has to be rebuilt on these two events.
    """
    before = station.station_info()
    assert _instrument(before, "temperature").availability == ()

    ok, _ = station.disconnect_instrument("temperature")
    assert ok
    disconnected = station.station_info()
    assert disconnected.seq > before.seq
    assert _instrument(disconnected, "temperature").availability == ("operator",)

    ok, _ = station.connect_instrument("temperature")
    assert ok
    reconnected = station.station_info()
    assert reconnected.seq > disconnected.seq
    assert _instrument(reconnected, "temperature").availability == ()


def test_offline_instrument_still_declares_its_capabilities(station: Station) -> None:
    """An unreachable instrument says what it WOULD offer.

    Described from its class, so the readings and controls are all there;
    only the two things an instance knows are missing — the configured limit
    bounds, which report null.
    """
    station.disconnect_instrument("magnet_z")
    magnet = _instrument(station.station_info(), "magnet_z")

    assert magnet.vi_class == "SuperconductingMagnetVI"
    assert magnet.kind == "magnet"
    assert [entry.name for entry in magnet.monitored]
    assert magnet.safety_flags == {"quench": "critical"}
    assert magnet.limits["set_field"]["target_T"] == {
        "limit": "field_T",
        "min": None,
        "max": None,
    }


# ── The manifest rendering ────────────────────────────────────────────────────


def test_manifest_groups_equal_the_vis_declared_ui_groups(station: Station) -> None:
    """The temperature VI's manifest groups ARE its ``ui_groups`` declaration.

    Key, title, description and member order all come from the one
    declaration on the VI; the manifest adds nothing and reorders nothing.
    """
    vi = station.temperature
    entry = _manifest_instrument(build_manifest(station), "temperature")

    assert [group["key"] for group in entry["groups"]] == [
        group.key for group in vi.ui_groups
    ]
    for rendered, declared in zip(entry["groups"], vi.ui_groups, strict=True):
        assert rendered["title"] == declared.title
        assert rendered["description"] == declared.description
        assert rendered["monitored"] + rendered["controls"] == list(declared.members)


def test_manifest_puts_grouped_capabilities_first(station: Station) -> None:
    """Grouped items lead, in declared group order; ungrouped ones follow."""
    entry = _manifest_instrument(build_manifest(station), "temperature")
    grouped = [
        name
        for group in entry["groups"]
        for name in group["monitored"] + group["controls"]
    ]
    assert grouped, "temperature declares groups"
    assert entry["ungrouped"]["monitored"] or entry["ungrouped"]["controls"]

    order = [item["name"] for item in entry["monitored"]]
    ungrouped = entry["ungrouped"]["monitored"]
    assert order == [n for n in grouped if n in order] + ungrouped


def test_manifest_group_and_ungrouped_partition_every_capability(
    manifest: dict,
) -> None:
    """Nothing is listed twice and nothing is left out of the group index."""
    for entry in manifest["instruments"]:
        indexed = [
            name
            for group in entry["groups"]
            for name in group["monitored"] + group["controls"]
        ]
        indexed += entry["ungrouped"]["monitored"] + entry["ungrouped"]["controls"]
        declared = [item["name"] for item in entry["monitored"]]
        declared += [item["name"] for item in entry["controls"]]
        assert sorted(indexed) == sorted(declared), entry["name"]


def test_manifest_drops_the_raw_ui_groups_declaration(manifest: dict) -> None:
    """``groups`` is the rendering of ``ui_groups``, so both would be two truths."""
    for entry in manifest["instruments"]:
        assert "ui_groups" not in entry


def test_measurement_arming_controls_carry_params_units_and_choices(
    station: Station,
) -> None:
    """Every measurement VI's ``initiate_measurement`` renders its full knobs.

    The measurement-method standard installs ``measurement_parameters`` as
    that control's declared specs, so the manifest must show every knob with
    the unit and, where the VI enumerates them, the choices — that is what an
    agent arms a measurement from.
    """
    manifest = build_manifest(station)
    checked = 0
    for name in station.measurement_vi_names():
        vi = station.get_vi(name)
        entry = _manifest_instrument(manifest, name)
        arming = _manifest_control(entry, "initiate_measurement")
        params = {item["name"]: item for item in arming["params"]}

        assert set(params) == set(vi.measurement_parameters), name
        for param_name, spec in vi.measurement_parameters.items():
            rendered = params[param_name]
            assert rendered["kind"] == spec.type.__name__
            assert rendered["unit"] == spec.unit
            assert rendered["description"] == spec.description
            assert rendered["default"] == spec.default
            assert rendered["choices"] == (
                dict(spec.choices) if spec.choices else None
            )
        checked += 1
    assert checked >= 1, "the sim station configures a measurement VI"


def test_enumerated_control_renders_its_choice_map(station: Station) -> None:
    """A concrete enumerated knob reaches the manifest as a label -> value map."""
    entry = _manifest_instrument(build_manifest(station), "temperature")
    params = {
        item["name"]: item
        for item in _manifest_control(entry, "set_heater_range")["params"]
    }
    assert params["range_setting"]["kind"] == "str"
    assert params["range_setting"]["choices"]["Medium"] == "MEDIUM"


def test_manifest_header_names_its_schema_and_setup(manifest: dict) -> None:
    """The manifest identifies its own shape, so a consumer can version-check it."""
    assert manifest["schema"] == MANIFEST_SCHEMA_ID
    # The wire identifier a consumer matches on; renaming it is a schema change.
    assert MANIFEST_SCHEMA_ID == "i2as.capability_manifest/1"
    assert manifest["setup"] == "sim_cryostat"
    assert manifest["tick_interval_s"] == pytest.approx(
        read_tick_interval_ms(SIM_CONFIG) / 1000.0
    )
    assert isinstance(manifest["seq"], int)


def test_manifest_is_json_serialisable(manifest: dict) -> None:
    """`json.dumps` accepts it as it stands — no encoder, no default=."""
    assert json.loads(json.dumps(manifest)) == manifest


# ── The schema and its validator ──────────────────────────────────────────────


def test_manifest_validates_against_its_schema(manifest: dict) -> None:
    """The rendering conforms to the schema that describes it."""
    assert validate_manifest(manifest) == []


def test_schema_is_a_draft_2020_12_document() -> None:
    """It stays a real, publishable schema, usable by any JSON Schema tool."""
    assert (
        MANIFEST_SCHEMA["$schema"]
        == "https://json-schema.org/draft/2020-12/schema"
    )
    assert json.loads(json.dumps(MANIFEST_SCHEMA)) == MANIFEST_SCHEMA


@pytest.mark.parametrize(
    "mutate, expected",
    [
        pytest.param(
            lambda m: m.pop("setup"), "missing required key 'setup'", id="missing-key"
        ),
        pytest.param(
            lambda m: m.__setitem__("tick_interval_s", "fast"),
            "expected number",
            id="wrong-type",
        ),
        pytest.param(
            lambda m: m.__setitem__("schema", "something/else"),
            "is not one of",
            id="wrong-enum",
        ),
        pytest.param(
            lambda m: m.__setitem__("surprise", 1),
            "unexpected key 'surprise'",
            id="extra-key",
        ),
        pytest.param(
            lambda m: m["instruments"][0]["controls"][0].__setitem__("scope", "nope"),
            "is not one of",
            id="nested-enum",
        ),
        pytest.param(
            lambda m: m["instruments"][0]["monitored"][0].__setitem__("unit", 5),
            "expected string",
            id="nested-type",
        ),
        pytest.param(
            lambda m: m["instruments"][0]["limits"].__setitem__("x", {"y": {}}),
            "missing required key",
            id="open-map-value",
        ),
    ],
)
def test_validator_reports_each_kind_of_violation(
    manifest: dict, mutate, expected: str
) -> None:
    """Every JSON Schema keyword the document uses is actually enforced.

    A validator that silently passes an unimplemented keyword would make the
    schema decorative, so each one gets a violation it must catch.
    """
    mutate(manifest)
    errors = validate_manifest(manifest)
    assert any(expected in error for error in errors), errors


def test_validator_accepts_a_null_limit_bound(manifest: dict) -> None:
    """An unbounded (or offline) limit side is null, and that is legal."""
    limits = manifest["instruments"][0]["limits"]
    for param_map in limits.values():
        for bound in param_map.values():
            bound["min"] = None
            bound["max"] = None
    assert validate_manifest(manifest) == []


# ── The command-line entry point ──────────────────────────────────────────────


def test_cli_prints_a_valid_manifest(capsys) -> None:
    """`main()` writes the manifest to stdout and exits zero."""
    assert main([SIM_CONFIG]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema"] == MANIFEST_SCHEMA_ID
    assert validate_manifest(printed) == []


def test_cli_reports_a_usage_error(capsys) -> None:
    """No config directory is a usage error, on stderr, exit 2."""
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().err


def test_cli_reports_an_unbuildable_config(tmp_path, capsys) -> None:
    """A config that cannot be built is reported, never a traceback."""
    assert main([str(tmp_path)]) == 1
    assert "cannot build a station" in capsys.readouterr().err


def test_cli_runs_without_a_qapplication() -> None:
    """`python -m i2as.core.capability_manifest` needs no Qt display.

    Run in a subprocess with no Qt platform plugin available at all, so a
    stray `QApplication` (or any widget import that needs one) would fail
    rather than quietly succeed on the test session's offscreen platform.
    """
    result = subprocess.run(
        [sys.executable, "-m", "i2as.core.capability_manifest", SIM_CONFIG],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "/usr/bin:/bin", "QT_QPA_PLATFORM": "definitely-not-a-platform"},
        cwd=str(Path(i2as.__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["schema"] == MANIFEST_SCHEMA_ID


# ── Helpers ───────────────────────────────────────────────────────────────────


def _instrument(info: StationInfo, name: str):
    """Return one instrument's declaration from a snapshot."""
    for entry in info.instruments:
        if entry.name == name:
            return entry
    raise AssertionError(f"{name!r} is not in the snapshot")


def _control(instrument, name: str):
    """Return one control's declaration from an instrument."""
    for entry in instrument.controls:
        if entry.name == name:
            return entry
    raise AssertionError(f"{name!r} is not a control of {instrument.name!r}")


def _manifest_instrument(manifest: dict, name: str) -> dict:
    """Return one instrument's entry from a rendered manifest."""
    for entry in manifest["instruments"]:
        if entry["name"] == name:
            return entry
    raise AssertionError(f"{name!r} is not in the manifest")


def _manifest_control(entry: dict, name: str) -> dict:
    """Return one control's entry from a rendered instrument."""
    for control in entry["controls"]:
        if control["name"] == name:
            return control
    raise AssertionError(f"{name!r} is not a control of {entry['name']!r}")
