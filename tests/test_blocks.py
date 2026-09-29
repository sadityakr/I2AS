"""Tests for the block layer: the bundle hand-off, the block contracts and the shipped blocks.

Everything here is plain Python — no Qt, no station. The shipped connectors
are exercised through the same protocol the helper process speaks (the
in-process runner), and once through a real helper process, so the process
boundary itself is tested too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from i2as.analysis.bundle import (
    BUNDLE_FILENAME,
    MAX_ARTIFACTS,
    Artifact,
    Bundle,
    Producer,
    is_bundle_id,
    new_bundle_id,
    output_file,
    read_bundle,
    seal_bundle,
    verify_artifact,
)
from i2as.blocks import (
    ElnAuthError,
    ElnNotFound,
    ElnTransientError,
    ElnValidationError,
    HttpResponse,
    Section,
    figure,
    heading,
    html,
    markdown,
    paragraph,
    results,
    table,
)
from i2as.blocks.checker import check_block, check_connector, check_renderer
from i2as.blocks.connector import CONNECTOR_METHODS, ElnEntryRef, ElnQuery, ElnRecord, ElnRef
from i2as.blocks.discovery import (
    KIND_CONNECTOR,
    KIND_PROFILE,
    KIND_RENDERER,
    describe_block,
    discover_blocks,
    shipped_dir,
)
from i2as.blocks.markup import markdown_to_html, sanitize_html, section_to_html
from i2as.blocks.profile import apply_read_map, parse_profile, resolve_fields
from i2as.blocks.protocol import BlockError
from i2as.session.eln.block_runner import BlockTimeout, InProcessBlockRunner, SubprocessBlockRunner

SHIPPED = shipped_dir()
ELABFTW = SHIPPED / "connectors" / "elabftw.py"
SIM = SHIPPED / "connectors" / "sim.py"
DEFAULT_RENDERER = SHIPPED / "renderers" / "default.py"


# ── The analysis bundle ───────────────────────────────────────────────────


def _bundle_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "run-0001" / "b1"
    folder.mkdir(parents=True)
    (folder / "overview.png").write_bytes(b"PNG-DATA")
    (folder / "fit.plot.json").write_text("{}", encoding="utf-8")
    (folder / "points.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (folder / "notes.exe").write_bytes(b"MZ")
    (folder / ".hidden.png").write_bytes(b"x")
    (folder / "spec.json").write_text("{}", encoding="utf-8")
    return folder


def _seal(folder: Path, claims: dict | None = None) -> Bundle:
    return seal_bundle(
        folder,
        claims if claims is not None else {"status": "ok", "summary": ["It worked."], "figures": [{"file": "overview.png", "caption": "Overview"}]},
        bundle_id=folder.name,
        experiment_id="001_x",
        run_ids=("run-0001",),
        producer=Producer(kind="recipe", name="generic_sweep", digest="d" * 64),
    )


def test_sealing_hashes_the_allowed_files_and_refuses_the_rest(tmp_path):
    folder = _bundle_folder(tmp_path)
    bundle = _seal(folder)
    kinds = {a.path: a.kind for a in bundle.artifacts}
    assert kinds == {"overview.png": "figure", "fit.plot.json": "plot_data", "points.csv": "table"}
    overview = bundle.artifact("overview.png")
    assert overview.caption == "Overview" and overview.bytes == 8 and len(overview.sha256) == 64
    written = json.loads((folder / BUNDLE_FILENAME).read_text(encoding="utf-8"))
    assert written["schema"] == "i2as.analysis-bundle" and written["schema_version"] == 1
    assert read_bundle(folder) == bundle


def test_a_file_swapped_after_sealing_is_refused(tmp_path):
    folder = _bundle_folder(tmp_path)
    bundle = _seal(folder)
    artifact = bundle.artifact("overview.png")
    assert verify_artifact(folder, artifact) == (folder / "overview.png").resolve()
    (folder / "overview.png").write_bytes(b"SOMETHING-ELSE")
    assert verify_artifact(folder, artifact) is None
    assert verify_artifact(folder, Artifact("x", path="../evil.png")) is None


def test_output_file_trusts_no_name(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    assert output_file(tmp_path, "a.png") is not None
    for name in ("../a.png", "sub/a.png", str(tmp_path / "a.png"), ".a.png", "a.txt", ""):
        assert output_file(tmp_path, name) is None


def test_a_legacy_report_reads_as_an_unsealed_bundle(tmp_path):
    (tmp_path / "report.json").write_text(
        json.dumps({"run_id": "run-0001", "recipe": "old", "status": "ok", "summary": ["s"], "figures": []}),
        encoding="utf-8",
    )
    bundle = read_bundle(tmp_path, bundle_id="legacy")
    assert bundle is not None and bundle.sealed is False and bundle.schema_version == 0
    assert bundle.producer.kind == "legacy" and bundle.producer.name == "old" and bundle.run_ids == ("run-0001",)


def test_bundle_parsing_is_tolerant():
    assert Bundle.from_dict("junk") == Bundle(bundle_id="", sealed=True, schema_version=0)
    bundle = Bundle.from_dict({"status": "weird", "artifacts": [{}] * (MAX_ARTIFACTS + 5), "duration_s": -1})
    assert bundle.status == "ok" and len(bundle.artifacts) == MAX_ARTIFACTS and bundle.duration_s == 0.0


def test_bundle_ids_are_plain_names():
    assert is_bundle_id(new_bundle_id("magneto resistance!"))
    for bad in ("", "../x", "a/b", ".hidden", "-x", "a" * 200, None):
        assert not is_bundle_id(bad)


def test_the_store_finds_every_bundle_of_a_run(tmp_path):
    from i2as.session.store import ExperimentStore

    store = ExperimentStore(tmp_path)
    for bundle_id in ("20260101T000000Z-a-0001", "script-probe_1"):
        folder = store.bundle_dir("001_x", "run-0001", bundle_id)
        folder.mkdir(parents=True)
        seal_bundle(folder, {"status": "ok"}, bundle_id=bundle_id, experiment_id="001_x", run_ids=("run-0001",), producer=Producer())
    assert store.bundle_dir("001_x", "run-0001", "script-probe_1").parent.name == "scripts"
    assert {b.bundle_id for b in store.list_bundles("001_x", "run-0001")} == {"20260101T000000Z-a-0001", "script-probe_1"}
    with pytest.raises(ValueError):
        store.bundle_dir("001_x", "run-0001", "../../escape")
    assert store.read_bundle("001_x", "run-0001", "../../escape") is None


# ── Markup: what a renderer writes reaches a page only as safe HTML ───────


def test_sanitize_keeps_text_and_drops_everything_dangerous():
    out = sanitize_html(
        '<p onclick="x()">hi<script>steal()</script><a href="javascript:alert(1)">l</a>'
        '<a href="https://ok.org" style="x">ok</a><img src="https://x/y.png"><iframe>z</iframe></p>'
    )
    assert "script" not in out and "steal" not in out and "onclick" not in out and "javascript" not in out
    assert "<img" not in out and "iframe" not in out and "style" not in out
    assert '<a href="https://ok.org">ok</a>' in out and out.startswith("<p>hi")


def test_markdown_is_restricted_and_escapes_markup():
    out = markdown_to_html("A **b** *c* `<x>` [l](https://e.org) <b>raw</b>\n\n- one\n- two")
    assert "<strong>b</strong>" in out and "<em>c</em>" in out and "<code>&lt;x&gt;</code>" in out
    assert '<a href="https://e.org">l</a>' in out and "&lt;b&gt;raw&lt;/b&gt;" in out
    assert "<ul><li>one</li><li>two</li></ul>" in out
    assert "<a" not in markdown_to_html("[l](javascript:alert(1))")


def test_section_html_is_headed_by_the_publish_id_and_names_uploaded_figures():
    section = Section.from_dict(
        {
            "title": "run-0001",
            "blocks": [
                heading("Run"),
                paragraph("<b>not bold</b>"),
                figure("b1", "overview.png", "Fig"),
                figure("b1", "missing.png"),
                results("R", [{"name": "R0", "value": 1.5, "unit": "Ω", "uncertainty": 0.1}]),
                {"type": "nonsense"},
            ],
        }
    )
    assert len(section.blocks) == 5, "an unknown block type is dropped"
    out = section_to_html(section, publish_id="P-1", published_utc="2026-09-27T10:00:00+00:00", figure_names={("b1", "overview.png"): "run-0001_overview.png"})
    assert out.startswith("<div><h2>I2AS · 2026-09-27 10:00 UTC · run-0001 · P-1</h2>")
    assert "&lt;b&gt;not bold&lt;/b&gt;" in out and "run-0001_overview.png" in out
    assert "missing.png was not uploaded" in out and "1.5 ± 0.1 Ω" in out


def test_renderer_output_is_capped_and_tolerant():
    assert Section.from_dict("junk") == Section()
    big = Section.from_dict({"blocks": [table("t", ["c"] * 50, [[1] * 50] * 500)], "fields": {str(i): i for i in range(500)}})
    block = big.blocks[0]
    assert len(block["columns"]) == 20 and len(block["rows"]) == 200 and len(big.fields) == 100


# ── Profiles ──────────────────────────────────────────────────────────────


PROFILE = """
connector: elabftw
renderer: default
template: "12"
read:
  sample_id: {from: "sample:Sample ID"}
  thickness: {from: "sample:Thickness", type: float, unit: nm}
  broken: {from: "nocolon"}
  count: {from: "page:Count", type: int}
fields:
  "R0 (Ω)": "result:R0"
  Title: "experiment:title"
  Bad: "shell:rm"
render: {include_parameters: false}
"""


def test_a_profile_parses_and_names_what_it_dropped():
    profile = parse_profile(PROFILE, "lab")
    assert profile.profile_id == "lab" and profile.connector == "elabftw" and profile.template == "12"
    assert [r.key for r in profile.read] == ["sample_id", "thickness", "count"]
    assert set(profile.fields) == {"R0 (Ω)", "Title"}
    assert len(profile.errors) == 2 and profile.render == {"include_parameters": False}
    assert parse_profile("a: [1, 2", "x").errors, "invalid YAML"
    assert parse_profile("- a list", "x").errors, "not a mapping"
    assert "'feilds' is not a profile setting" in parse_profile("feilds: {}", "x").errors[0]


def test_the_read_map_converts_and_explains_what_it_could_not_read():
    profile = parse_profile(PROFILE, "lab")
    sample = ElnRecord(ref=ElnRef(kind="item", record_id="1"), title="S", fields={"Sample ID": "S-001", "Thickness": "4,2"}, units={"Thickness": "nm"})
    found = apply_read_map(profile, {"sample": sample})
    assert found["sample_id"]["value"] == "S-001"
    assert found["thickness"]["value"] == pytest.approx(4.2) and found["thickness"]["unit"] == "nm"
    assert found["count"]["value"] is None and "nothing is linked as 'page'" in found["count"]["error"]


def test_the_write_map_never_writes_an_empty_value():
    profile = parse_profile(PROFILE, "lab")
    fields = resolve_fields(profile, experiment={"title": "Hall A"}, sample_info={}, results={"R0": 101.25})
    assert fields == {"R0 (Ω)": "101.25", "Title": "Hall A"}
    assert resolve_fields(profile, experiment={}, sample_info={}, results={}) == {}


# ── Discovery: static, and never executes a block ─────────────────────────


def test_the_shipped_blocks_are_discovered_and_usable():
    assert set(discover_blocks(KIND_CONNECTOR)) >= {"elabftw", "sim"}
    assert "default" in discover_blocks(KIND_RENDERER) and "default" in discover_blocks(KIND_PROFILE)
    elab = discover_blocks(KIND_CONNECTOR)["elabftw"]
    assert elab.usable and elab.capabilities.search and "base_url" in elab.settings_schema["properties"]


def test_discovery_reads_a_block_without_running_it(tmp_path):
    folder = tmp_path / "connectors"
    folder.mkdir()
    marker = tmp_path / "ran.txt"
    (folder / "evil.py").write_text(
        f"open({str(marker)!r}, 'w').write('ran')\n"
        "from i2as.blocks import ElnConnector, ElnCapabilities\n"
        "class Evil(ElnConnector):\n"
        "    backend = 'evil'\n"
        "    capabilities = ElnCapabilities(search=True)\n",
        encoding="utf-8",
    )
    (folder / "elabftw.py").write_text(ELABFTW.read_text(encoding="utf-8").replace('display_name = "eLabFTW"', 'display_name = "My eLab"'), encoding="utf-8")
    found = discover_blocks(KIND_CONNECTOR, tmp_path)
    assert found["evil"].usable and found["evil"].source == "user"
    assert not marker.exists(), "discovery must never execute a block"
    assert found["elabftw"].display_name == "My eLab", "a user block overrides the shipped one"


def test_a_block_that_does_not_follow_the_contract_is_described_as_unusable(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("from i2as.blocks import ElnConnector\nclass A(ElnConnector):\n    backend = 'Not Lower'\n", encoding="utf-8")
    assert "lowercase" in describe_block(KIND_CONNECTOR, bad).error
    no_render = tmp_path / "r.py"
    no_render.write_text("NAME = 'x'\n", encoding="utf-8")
    assert "render" in describe_block(KIND_RENDERER, no_render).error
    broken = tmp_path / "s.py"
    broken.write_text("def (:\n", encoding="utf-8")
    assert describe_block(KIND_RENDERER, broken).error


# ── The protocol, in process and across a real helper process ─────────────


def test_the_in_process_runner_speaks_the_protocol_and_maps_errors(tmp_path):
    runner = InProcessBlockRunner(KIND_CONNECTOR, SIM, {"state_file": str(tmp_path / "s.json")}, "key")
    hits = runner.call("search", {"query": ElnQuery(text="S-001", kind="item").to_dict()}, 5)
    assert hits[0]["title"].startswith("Sample S-001")
    with pytest.raises(ElnNotFound):
        runner.call("get_record", {"ref": {"kind": "entry", "record_id": "999"}}, 5)
    with pytest.raises(BlockError):
        runner.call("no_such_method", {}, 5)
    anonymous = InProcessBlockRunner(KIND_CONNECTOR, SIM, {"state_file": str(tmp_path / "s.json")}, "")
    with pytest.raises(ElnAuthError):
        anonymous.call("verify", {}, 5)


def test_a_real_helper_process_runs_a_connector_and_keeps_its_state(tmp_path):
    runner = SubprocessBlockRunner(KIND_CONNECTOR, SIM, {"state_file": str(tmp_path / "s.json")}, "key")
    try:
        entry = runner.call("create_entry", {"title": "Hall A", "template_id": "10", "fields": {}}, 30)
        runner.call("append_section", {"entry": entry, "publish_id": "P-9", "html": "<p>P-9 é</p>"}, 30)
        assert runner.call("has_section", {"entry": entry, "publish_id": "P-9"}, 30) is True
    finally:
        runner.close()
    state = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert state["entries"][entry["entry_id"]]["body"] == "<p>P-9 é</p>", "non-ASCII survives the pipe"


def test_a_renderer_in_its_helper_has_no_network_and_is_killed_when_it_hangs(tmp_path):
    net = tmp_path / "net.py"
    net.write_text("NAME='net'\ndef render(context):\n    import socket\n    socket.create_connection(('example.com', 80))\n", encoding="utf-8")
    runner = SubprocessBlockRunner(KIND_RENDERER, net)
    with pytest.raises(BlockError, match="no network"):
        runner.call("render", {"context": {}}, 30)
    runner.close()
    slow = tmp_path / "slow.py"
    slow.write_text("NAME='slow'\nimport time\ndef render(context):\n    time.sleep(30)\n", encoding="utf-8")
    runner = SubprocessBlockRunner(KIND_RENDERER, slow)
    with pytest.raises(BlockTimeout):
        runner.call("render", {"context": {}}, 2)
    assert not runner.running
    runner.close()


def test_the_helper_environment_carries_no_secret_of_the_application(monkeypatch):
    from i2as.session.eln.block_runner import helper_environment

    monkeypatch.setenv("I2AS_ELN_APIKEY", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy")
    renderer_env = helper_environment(KIND_RENDERER)
    connector_env = helper_environment(KIND_CONNECTOR)
    assert "I2AS_ELN_APIKEY" not in renderer_env and "I2AS_ELN_APIKEY" not in connector_env
    assert "HTTPS_PROXY" in connector_env and "HTTPS_PROXY" not in renderer_env


# ── The checker ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", [ELABFTW, SIM, DEFAULT_RENDERER, SHIPPED / "profiles" / "default.yaml"])
def test_every_shipped_block_passes_its_checker(path):
    checks = check_block(path)
    assert all(check.ok for check in checks), [str(c) for c in checks if not c.ok]


def test_the_checker_names_what_a_broken_connector_gets_wrong(tmp_path):
    folder = tmp_path / "connectors"
    folder.mkdir()
    source = ELABFTW.read_text(encoding="utf-8")
    source = source.replace("raise_for_status(method, path, response)", "response")
    source = source.replace("    def link_item(", "    def extra_public(self):\n        return 1\n\n    def link_item(")
    broken = folder / "broken.py"
    broken.write_text(source.replace('backend = "elabftw"', 'backend = "broken"'), encoding="utf-8")
    failed = {c.rule for c in check_connector(broken) if not c.ok}
    assert "adds no public method of its own" in failed
    assert any("HTTP 401" in rule for rule in failed)


def test_the_checker_catches_a_renderer_naming_a_figure_that_does_not_exist(tmp_path):
    folder = tmp_path / "renderers"
    folder.mkdir()
    bad = folder / "bad.py"
    bad.write_text("NAME='bad'\nfrom i2as.blocks import Section, figure\ndef render(context):\n    return Section(title='x', blocks=(figure('nope', 'x.png'),))\n", encoding="utf-8")
    failed = [c for c in check_renderer(bad) if not c.ok]
    assert [c.rule for c in failed] == ["names only figures that exist in the bundles"]


def test_the_contract_has_exactly_the_documented_methods():
    assert CONNECTOR_METHODS == tuple(sorted((
        "append_section", "create_entry", "get_record", "has_section", "link_item",
        "list_templates", "search", "set_fields", "upload", "verify",
    )))


# ── The eLabFTW connector, against canned HTTP ─────────────────────────────


class FakeTransport:
    """Canned HTTP responses keyed by ``"METHOD /path"``; records every call."""

    def __init__(self, responses=None):
        self.responses = dict(responses or {})
        self.calls = []

    def request(self, method, url, headers, body, timeout_s):
        path = url.split("/api/v2", 1)[-1]
        self.calls.append({"method": method, "path": path, "headers": dict(headers), "body": body})
        key = f"{method} {path}"
        canned = self.responses.get(key, HttpResponse(status=404, body=b'{"description":"not found"}'))
        return canned.pop(0) if isinstance(canned, list) else canned


def _json(status: int, payload: object, headers: dict | None = None) -> HttpResponse:
    return HttpResponse(status=status, headers=headers or {}, body=json.dumps(payload).encode())


def _elab(responses=None, key="top-secret-key"):
    from i2as.blocks.discovery import connector_class, load_block_module

    cls = connector_class(load_block_module(ELABFTW))
    transport = FakeTransport(responses)
    return cls({"base_url": "https://elab.example.org/", "timeout_s": 5}, key, transport), transport


def test_elabftw_verify_sends_the_key_and_names_the_account():
    connector, transport = _elab({"GET /users/me": _json(200, {"fullname": "A Sen", "team_name": "Cryo"})})
    identity = connector.verify()
    assert (identity.name, identity.team) == ("A Sen", "Cryo")
    assert transport.calls[0]["headers"]["Authorization"] == "top-secret-key"


def test_elabftw_failures_are_classified_and_never_quote_the_key():
    for status, error in ((401, ElnAuthError), (404, ElnNotFound), (503, ElnTransientError), (422, ElnValidationError)):
        connector, _ = _elab({"GET /users/me": _json(status, {"description": "nope"})})
        with pytest.raises(error) as raised:
            connector.verify()
        assert "top-secret-key" not in str(raised.value)
    connector, _ = _elab(key="")
    with pytest.raises(ElnAuthError):
        connector.verify()


def test_elabftw_reads_extra_fields_and_searches_items():
    metadata = {"extra_fields": {"Thickness": {"type": "number", "value": 4.2, "unit": "nm"}, "Sample ID": {"value": "S-1"}}}
    connector, transport = _elab(
        {
            "GET /items/5": _json(200, {"title": "S-1", "metadata": json.dumps(metadata), "items_type_title": "Sample"}),
            "GET /items?q=S%201&limit=20": _json(200, [{"id": 5, "title": "S-1", "items_type_title": "Sample"}, {"title": "no id"}]),
        }
    )
    record = connector.get_record(ElnRef(kind="item", record_id="5"))
    assert record.fields == {"Thickness": "4.2", "Sample ID": "S-1"} and record.units == {"Thickness": "nm"}
    assert record.url == "https://elab.example.org/database.php?mode=view&id=5"
    [hit] = connector.search(ElnQuery(text="S 1", kind="item"))
    assert hit.ref.record_id == "5" and hit.category == "Sample"


def test_elabftw_creates_a_page_appends_sections_and_overwrites_fields():
    body = {"body": "<p>by hand</p>", "metadata": json.dumps({"extra_fields": {"R0": {"type": "number", "value": "1"}}})}
    connector, transport = _elab(
        {
            "POST /experiments": HttpResponse(201, {"location": "https://elab.example.org/api/v2/experiments/77"}, b""),
            "PATCH /experiments/77": _json(200, {}),
            "GET /experiments/77": _json(200, body),
        }
    )
    entry = connector.create_entry("Hall A", "12", {})
    assert entry.entry_id == "77" and entry.template_id == "12"
    assert json.loads(transport.calls[0]["body"]) == {"template": 12}
    connector.append_section(entry, "P-1", "<div>P-1 new</div>")
    patched = json.loads(transport.calls[-1]["body"])
    assert patched["body"] == "<p>by hand</p><div>P-1 new</div>", "what people wrote is kept, the section goes last"
    assert connector.has_section(entry, "P-1") is False, "the fake page body was not updated"
    connector.set_fields(entry, {"R0": "2.5", "New": "x"})
    metadata = json.loads(json.loads(transport.calls[-1]["body"])["metadata"])
    assert metadata["extra_fields"]["R0"] == {"type": "number", "value": "2.5"}
    assert metadata["extra_fields"]["New"] == {"type": "text", "value": "x"}


def test_elabftw_uploads_and_links(tmp_path):
    file = tmp_path / "run-0001_overview.png"
    file.write_bytes(b"PNG")
    connector, transport = _elab(
        {
            "POST /experiments/77/uploads": HttpResponse(201, {"location": "https://e/api/v2/experiments/77/uploads/9"}, b""),
            "POST /experiments/77/items_links/5": HttpResponse(201, {}, b""),
        }
    )
    entry = ElnEntryRef(backend="elabftw", entry_id="77")
    assert connector.upload(entry, file, "Overview") == "9"
    assert b'filename="run-0001_overview.png"' in transport.calls[0]["body"] and b"Overview" in transport.calls[0]["body"]
    connector.link_item(entry, ElnRef(kind="item", record_id="5"))
    assert transport.calls[-1]["path"] == "/experiments/77/items_links/5"


# ── The default renderer ──────────────────────────────────────────────────


def test_the_default_renderer_lays_out_each_run_with_its_bundle():
    from i2as.blocks.checker import sample_context
    from i2as.blocks.discovery import load_block_module

    section = load_block_module(DEFAULT_RENDERER).render(sample_context())
    kinds = [b["type"] for b in section.blocks]
    assert kinds.count("heading") == 2, "one sub-section per run"
    assert "results" in kinds and "figure" in kinds and "table" in kinds
    assert section.title == "run-0001, run-0002"
    assert "magnetoresistance" in section.tags


def test_block_helpers_build_plain_dicts():
    assert html("<p>x</p>") == {"type": "html", "html": "<p>x</p>"}
    assert markdown("x")["type"] == "markdown"


def test_checking_a_renderer_leaves_the_network_as_it_found_it():
    """The checker runs in the author's own process: it must hand the network back."""
    import socket

    before = (socket.socket, socket.create_connection, socket.getaddrinfo)
    check_renderer(DEFAULT_RENDERER)
    assert (socket.socket, socket.create_connection, socket.getaddrinfo) == before
