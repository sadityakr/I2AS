"""Behaviour tests for the ELN track (i2as/session/eln/).

Conformance (tests/test_conformance.py) checks that every adapter *matches the
standard*; these tests check that the track actually publishes, queues,
retries, and records what it should. Everything runs against the in-memory
``SimElnAdapter`` and, for the eLabFTW backend, a fake transport — no network,
ever.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from i2as.session.eln.adapter import ElnEntryRef, ElnError
from i2as.session.eln.elabftw import (
    ElabFtwAdapter,
    HttpResponse,
    UrllibTransport,
)
from i2as.session.eln.outbox import (
    DRAIN_IDLE,
    DRAIN_PUBLISHED,
    DRAIN_RETRY,
    Outbox,
    OutboxJob,
)
from i2as.session.eln.settings import (
    API_KEY_ENV_VAR,
    SETTINGS_PATH_ENV_VAR,
    ElnSettings,
    eln_settings_path,
    load_eln_settings,
)
from i2as.session.eln.sim_eln import SimElnAdapter
from i2as.session.eln.templates import (
    render_run_body,
    render_run_metadata,
    render_run_title,
)

MANIFEST = {
    "run_id": "run-0001",
    "procedure": "Field Sweep",
    "kind": "run",
    "params": {"field_T": 1.5, "rate_T_per_s": 0.01},
    "data_file": "data/run-0001.h5",
    "started_utc": "2026-01-01T10:00:00+00:00",
    "finished_utc": "2026-01-01T11:00:00+00:00",
    "status": "done",
    "reason": "",
}


# ── User-level settings ───────────────────────────────────────────────────────


def test_settings_path_prefers_the_environment_override(monkeypatch, tmp_path):
    """``I2AS_ELN_SETTINGS`` wins over the per-user application-data path."""
    monkeypatch.setenv(SETTINGS_PATH_ENV_VAR, str(tmp_path / "custom.json"))
    assert eln_settings_path() == tmp_path / "custom.json"


def test_settings_path_is_user_level_not_shipped_config(monkeypatch):
    """With no override the file lives in the per-user application-data dir."""
    monkeypatch.delenv(SETTINGS_PATH_ENV_VAR, raising=False)
    path = eln_settings_path()
    assert path.name == "eln-settings.json"
    assert "configs" not in path.parts, "an API key must never live in a shipped config"


def test_missing_settings_file_leaves_publishing_off(monkeypatch, tmp_path):
    """A missing file is the normal case: disabled defaults, no exception."""
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    settings = load_eln_settings(tmp_path / "nope.json")
    assert settings == ElnSettings()
    assert not settings.enabled and not settings.is_configured


def test_malformed_settings_file_degrades_to_defaults(monkeypatch, tmp_path):
    """Junk on disk switches publishing off rather than taking the app down."""
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    path = tmp_path / "eln-settings.json"
    path.write_text("{not json at all", encoding="utf-8")
    assert load_eln_settings(path) == ElnSettings()


def test_settings_round_trip_and_env_key_override(monkeypatch, tmp_path):
    """The file is read, and the environment key overrides its ``api_key``."""
    path = tmp_path / "eln-settings.json"
    written = ElnSettings(
        enabled=True, base_url="https://elab.example.org/", api_key="file-key"
    )
    path.write_text(json.dumps(written.to_dict(include_secret=True)), encoding="utf-8")

    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    from_file = load_eln_settings(path)
    assert from_file.api_key == "file-key"
    assert from_file.base_url == "https://elab.example.org", "trailing slash trimmed"
    assert from_file.is_configured

    monkeypatch.setenv(API_KEY_ENV_VAR, "env-key")
    assert load_eln_settings(path).api_key == "env-key"


def test_api_key_never_appears_in_repr_or_a_plain_dict():
    """The key is redacted everywhere except an explicit include_secret dump."""
    settings = ElnSettings(enabled=True, base_url="https://x", api_key="s3cret")
    assert "s3cret" not in repr(settings)
    assert "s3cret" not in json.dumps(settings.to_dict())
    assert settings.to_dict(include_secret=True)["api_key"] == "s3cret"


def test_redacted_key_read_back_is_treated_as_no_key():
    """A settings dump written back without the secret does not resurrect it."""
    settings = ElnSettings(enabled=True, base_url="https://x", api_key="s3cret")
    assert ElnSettings.from_dict(settings.to_dict()).api_key == ""


# ── The sim adapter (the twin every other test builds on) ─────────────────────


def test_sim_adapter_creates_updates_and_attaches():
    """A healthy notebook records every accepted call."""
    adapter = SimElnAdapter({"base_url": "https://sim.example"})
    assert adapter.verify()
    assert adapter.list_templates()

    ref = adapter.create_entry("Title", None, "<p>body</p>", ["i2as"], {"k": "v"})
    assert ref.backend == "sim_eln"
    assert ref.entry_id in adapter.entries
    assert ref.url.endswith(ref.entry_id)

    adapter.update_entry(ref, "<p>new</p>", {"k": "w"})
    assert adapter.entries[ref.entry_id]["body_html"] == "<p>new</p>"

    adapter.attach_link(ref, "/data/run-0001.h5", "raw data")
    assert adapter.links == [
        {"entry_id": ref.entry_id, "url": "/data/run-0001.h5", "comment": "raw data"}
    ]


def test_sim_adapter_attaches_a_real_file_only(tmp_path):
    """``attach_file`` models the real backend's "the file must exist" failure."""
    adapter = SimElnAdapter({})
    ref = adapter.create_entry("T", None, "", [], {})
    data = tmp_path / "run.h5"
    data.write_bytes(b"\x89HDF\r\n\x1a\n")
    adapter.attach_file(ref, data, "raw data")
    assert adapter.uploads[0]["path"] == str(data)
    with pytest.raises(ElnError):
        adapter.attach_file(ref, tmp_path / "missing.h5")


def test_sim_adapter_models_offline_and_transient_failure():
    """Offline raises on every call; ``sim_fail_calls`` fails then recovers."""
    offline = SimElnAdapter({"sim_offline": True})
    with pytest.raises(ElnError):
        offline.verify()
    offline.offline = False
    assert offline.verify()

    flaky = SimElnAdapter({"sim_fail_calls": 2})
    with pytest.raises(ElnError):
        flaky.verify()
    with pytest.raises(ElnError):
        flaky.verify()
    assert flaky.verify()


def test_sim_adapter_rejects_an_unknown_entry():
    """Attaching to an entry that was never created is an ``ElnError``."""
    adapter = SimElnAdapter({})
    with pytest.raises(ElnError):
        adapter.attach_link(ElnEntryRef(entry_id="999"), "/x")


# ── Body renderers ────────────────────────────────────────────────────────────


def test_render_run_title_names_the_procedure_and_the_experiment():
    """The title carries the experiment, the procedure, and the start time."""
    assert render_run_title(MANIFEST) == "Field Sweep — 2026-01-01T10:00:00+00:00"
    assert render_run_title(MANIFEST, "Sample A").startswith("Sample A — Field Sweep")


def test_render_run_body_is_deterministic_and_complete():
    """The same manifest renders byte-identical HTML carrying every fact."""
    kwargs = {
        "experiment_id": "exp-1",
        "experiment_title": "Sample A",
        "setup": {"config_name": "sim_cryostat", "instruments": {"magnet": {"m": 1}}},
        "data_path": "/root/exp-1/data/run-0001.h5",
    }
    body = render_run_body(MANIFEST, **kwargs)
    assert body == render_run_body(MANIFEST, **kwargs)
    for expected in (
        "run-0001",
        "Field Sweep",
        "exp-1",
        "Sample A",
        "sim_cryostat",
        "field_T",
        "/root/exp-1/data/run-0001.h5",
    ):
        assert expected in body


def test_render_run_body_escapes_and_caps_values():
    """A hostile parameter cannot inject markup or a megabyte of body."""
    body = render_run_body({"run_id": "r", "params": {"note": "<script>x</script>" * 200}})
    assert "<script>" not in body
    assert "&lt;script&gt;" in body
    assert len(body) < 4000


def test_render_run_metadata_is_flat_and_json_safe():
    """The machine-readable twin of the body is flat scalars only."""
    metadata = render_run_metadata(MANIFEST, "exp-1", "/data/run-0001.h5")
    json.dumps(metadata)
    assert metadata["run_id"] == "run-0001"
    assert metadata["experiment_id"] == "exp-1"
    assert metadata["data_file"] == "/data/run-0001.h5"
    assert all(isinstance(value, str) for value in metadata.values())


# ── The outbox ────────────────────────────────────────────────────────────────


def _job(tmp_path, run_id="run-0001", **overrides):
    """Build a fully rendered publish job for ``run_id``."""
    fields = {
        "job_id": f"publish_run:{run_id}",
        "experiment_id": "exp-1",
        "run_id": run_id,
        "title": render_run_title(MANIFEST),
        "body_html": render_run_body(MANIFEST, data_path=str(tmp_path / "run.h5")),
        "tags": ["i2as"],
        "metadata": render_run_metadata(MANIFEST, "exp-1"),
        "data_path": str(tmp_path / "run.h5"),
    }
    fields.update(overrides)
    return OutboxJob(**fields)


def _data_file(tmp_path, size=16):
    """Write a stand-in HDF5 file and return its path."""
    path = tmp_path / "run.h5"
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"0" * max(size - 8, 0))
    return path


def test_outbox_is_created_lazily_and_reads_empty(tmp_path):
    """Nothing touches the disk until a job is queued."""
    outbox = Outbox(tmp_path / "exp-1" / "outbox.jsonl")
    assert outbox.jobs() == []
    assert outbox.pending() == []
    assert not outbox.path.exists()


def test_outbox_enqueue_is_idempotent_by_job_id(tmp_path):
    """A duplicate job id appends nothing — the whole track's dedup point."""
    _data_file(tmp_path)
    outbox = Outbox(tmp_path / "outbox.jsonl")
    assert outbox.enqueue(_job(tmp_path)) is True
    assert outbox.enqueue(_job(tmp_path)) is False
    assert len(outbox.jobs()) == 1
    assert outbox.path.read_text(encoding="utf-8").count("\n") == 1


def test_outbox_rejects_a_job_without_an_id(tmp_path):
    """An unidentifiable job could never be deduplicated or resumed."""
    outbox = Outbox(tmp_path / "outbox.jsonl")
    with pytest.raises(ValueError):
        outbox.enqueue(OutboxJob(run_id="r"))


def test_outbox_survives_a_restart(tmp_path):
    """A queued job is still queued when a brand-new Outbox opens the file."""
    _data_file(tmp_path)
    path = tmp_path / "outbox.jsonl"
    Outbox(path).enqueue(_job(tmp_path))
    reopened = Outbox(path)
    assert [job.run_id for job in reopened.pending()] == ["run-0001"]


def test_outbox_skips_a_corrupt_line_without_stranding_the_queue(tmp_path):
    """One mangled line is skipped with a warning; the rest still drain."""
    _data_file(tmp_path)
    path = tmp_path / "outbox.jsonl"
    outbox = Outbox(path)
    outbox.enqueue(_job(tmp_path))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
        handle.write('{"kind": "publish_run"}\n')  # no job_id
    assert [job.job_id for job in outbox.jobs()] == ["publish_run:run-0001"]


def test_outbox_drain_publishes_once_and_marks_the_job_done(tmp_path):
    """A healthy drain creates the entry, attaches the data, and finishes."""
    data = _data_file(tmp_path)
    outbox = Outbox(tmp_path / "outbox.jsonl")
    outbox.enqueue(_job(tmp_path))
    adapter = SimElnAdapter({})

    result = outbox.drain(adapter)
    assert result.state == DRAIN_PUBLISHED
    assert result.run_id == "run-0001"
    assert result.entry is not None and result.entry.entry_id
    assert adapter.entries[result.entry.entry_id]["body_html"].startswith("<h3>Run</h3>")
    assert adapter.uploads[0]["path"] == str(data)
    assert outbox.pending() == []

    assert outbox.drain(adapter).state == DRAIN_IDLE
    assert adapter.calls.count("create_entry") == 1


def test_outbox_drain_is_idle_on_an_empty_queue(tmp_path):
    """Nothing queued is not an error."""
    assert Outbox(tmp_path / "outbox.jsonl").drain(SimElnAdapter({})).state == DRAIN_IDLE


def test_outbox_keeps_a_job_while_the_notebook_is_offline(tmp_path):
    """Offline leaves the job queued; a later drain publishes it exactly once."""
    _data_file(tmp_path)
    path = tmp_path / "outbox.jsonl"
    outbox = Outbox(path, retry_base_s=0.0, retry_max_s=0.0)
    outbox.enqueue(_job(tmp_path))
    adapter = SimElnAdapter({"sim_offline": True})

    result = outbox.drain(adapter)
    assert result.state == DRAIN_RETRY
    assert result.detail
    queued = outbox.get("publish_run:run-0001")
    assert queued.state == "pending" and queued.attempts == 1
    assert queued.last_error and queued.next_attempt_utc
    assert not adapter.entries

    adapter.offline = False
    assert outbox.drain(adapter).state == DRAIN_PUBLISHED
    assert len(adapter.entries) == 1
    assert outbox.drain(adapter).state == DRAIN_IDLE
    assert len(adapter.entries) == 1


def test_outbox_backoff_is_persisted_and_grows(tmp_path):
    """A failed job is not retried until its journaled backoff has elapsed."""
    from datetime import datetime, timedelta, timezone

    _data_file(tmp_path)
    outbox = Outbox(tmp_path / "outbox.jsonl", retry_base_s=60.0, retry_max_s=600.0)
    outbox.enqueue(_job(tmp_path))
    adapter = SimElnAdapter({"sim_offline": True})
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert outbox.drain(adapter, now=now).state == DRAIN_RETRY
    assert outbox.drain(adapter, now=now + timedelta(seconds=30)).state == DRAIN_IDLE
    assert outbox.drain(adapter, now=now + timedelta(seconds=61)).state == DRAIN_RETRY
    assert outbox.get("publish_run:run-0001").attempts == 2
    # Second failure doubles the delay: still not due 61 s later, due at 121 s.
    assert outbox.drain(adapter, now=now + timedelta(seconds=122)).state == DRAIN_IDLE
    assert outbox.drain(adapter, now=now + timedelta(seconds=183)).state == DRAIN_RETRY


def test_outbox_retry_after_a_failed_attachment_reuses_the_entry(tmp_path):
    """The partial-failure case: never a second entry for the same job."""
    _data_file(tmp_path)
    outbox = Outbox(tmp_path / "outbox.jsonl", retry_base_s=0.0, retry_max_s=0.0)
    outbox.enqueue(_job(tmp_path))
    adapter = SimElnAdapter({"sim_reject_attachments": True})

    assert outbox.drain(adapter).state == DRAIN_RETRY
    created = outbox.get("publish_run:run-0001").entry_ref()
    assert created.entry_id in adapter.entries

    adapter._reject_attachments = False
    result = outbox.drain(adapter)
    assert result.state == DRAIN_PUBLISHED
    assert result.entry == created
    assert adapter.calls.count("create_entry") == 1
    assert len(adapter.entries) == 1


def test_outbox_links_a_file_above_the_attachment_cap(tmp_path):
    """Above the cap the entry records where the data lives instead."""
    data = _data_file(tmp_path, size=4096)
    outbox = Outbox(tmp_path / "outbox.jsonl")
    outbox.enqueue(_job(tmp_path, max_attachment_bytes=64))
    adapter = SimElnAdapter({})

    assert outbox.drain(adapter).state == DRAIN_PUBLISHED
    assert not adapter.uploads
    assert adapter.links[0]["url"] == str(data)
    assert "attachment cap" in adapter.links[0]["comment"]


def test_outbox_links_a_missing_data_file(tmp_path):
    """A data file that is not there is still recorded, never silently dropped."""
    outbox = Outbox(tmp_path / "outbox.jsonl")
    outbox.enqueue(_job(tmp_path))  # no file written
    adapter = SimElnAdapter({})

    assert outbox.drain(adapter).state == DRAIN_PUBLISHED
    assert not adapter.uploads
    assert "not found" in adapter.links[0]["comment"]


def test_outbox_drain_never_raises_on_a_misbehaving_adapter(tmp_path):
    """A non-ElnError out of an adapter is contained, not propagated to the timer."""
    _data_file(tmp_path)
    outbox = Outbox(tmp_path / "outbox.jsonl", retry_base_s=0.0, retry_max_s=0.0)
    outbox.enqueue(_job(tmp_path))

    class BrokenAdapter(SimElnAdapter):
        def create_entry(self, title, template_id, body_html, tags, metadata):
            raise RuntimeError("backend client blew up")

    result = outbox.drain(BrokenAdapter({}))
    assert result.state == DRAIN_RETRY
    assert "RuntimeError" in result.detail
    assert outbox.get("publish_run:run-0001").attempts == 1


# ── The eLabFTW backend (against a fake transport — never a live server) ──────


class FakeTransport:
    """Canned HTTP responses plus a full record of what was requested."""

    def __init__(self, responses=None):
        """Queue one response per expected call, keyed by ``"METHOD /path"``."""
        self.responses = dict(responses or {})
        self.calls = []

    def request(self, method, url, headers, body, timeout_s):
        """Record the call and answer from the canned table."""
        path = url.split("/api/v2", 1)[-1]
        self.calls.append(
            {
                "method": method,
                "url": url,
                "path": path,
                "headers": dict(headers),
                "body": body,
                "timeout_s": timeout_s,
            }
        )
        key = f"{method} {path}"
        if key not in self.responses:
            return HttpResponse(status=404, body=b'{"description":"not found"}')
        canned = self.responses[key]
        return canned.pop(0) if isinstance(canned, list) else canned


def _elab(responses=None, **settings):
    """Build an eLabFTW adapter over a fake transport."""
    base = {
        "enabled": True,
        "base_url": "https://elab.example.org",
        "api_key": "top-secret-key",
    }
    base.update(settings)
    transport = FakeTransport(responses)
    return ElabFtwAdapter(base, transport=transport), transport


def test_elabftw_verify_names_the_account_and_sends_the_token():
    """The health check authenticates and reports who the key belongs to."""
    adapter, transport = _elab(
        {"GET /users/me": HttpResponse(200, {}, b'{"fullname":"A Sen","team_name":"Cryo"}')}
    )
    assert adapter.verify() == "A Sen (Cryo)"
    call = transport.calls[0]
    assert call["url"] == "https://elab.example.org/api/v2/users/me"
    assert call["headers"]["Authorization"] == "top-secret-key"
    assert call["timeout_s"] == 15.0


def test_elabftw_maps_a_rejected_key_to_an_eln_error_without_leaking_it():
    """A 401 becomes an ``ElnError`` naming the status, never the key."""
    adapter, _ = _elab({"GET /users/me": HttpResponse(401, {}, b'{"description":"bad key"}')})
    with pytest.raises(ElnError) as excinfo:
        adapter.verify()
    assert "401" in str(excinfo.value)
    assert "top-secret-key" not in str(excinfo.value)


def test_elabftw_refuses_to_call_without_a_url_or_a_key():
    """Half-configured settings fail as an ``ElnError``, not a stray exception."""
    no_url, _ = _elab(base_url="")
    with pytest.raises(ElnError):
        no_url.verify()
    no_key, _ = _elab(api_key="")
    with pytest.raises(ElnError):
        no_key.verify()


def test_elabftw_lists_templates():
    """Templates come back as ``ElnTemplate``s, junk entries ignored."""
    adapter, _ = _elab(
        {
            "GET /experiments_templates": HttpResponse(
                200, {}, b'[{"id":7,"title":"Cryostat run"},"junk"]'
            )
        }
    )
    templates = adapter.list_templates()
    assert [(t.template_id, t.name) for t in templates] == [("7", "Cryostat run")]


def test_elabftw_creates_an_entry_from_a_template_then_fills_it_in():
    """Create + patch: the id comes from the Location header, the URL is human-facing."""
    adapter, transport = _elab(
        {
            "POST /experiments": HttpResponse(
                201, {"location": "https://elab.example.org/api/v2/experiments/42"}, b""
            ),
            "PATCH /experiments/42": HttpResponse(200, {}, b"{}"),
        },
        template_id="7",
    )
    ref = adapter.create_entry("Run 1", None, "<p>body</p>", ["i2as"], {"run_id": "r1"})
    assert ref.backend == "elabftw"
    assert ref.entry_id == "42"
    assert ref.template_id == "7"
    assert ref.url == "https://elab.example.org/experiments.php?mode=view&id=42"

    create, patch = transport.calls
    assert json.loads(create["body"]) == {"template": "7"}
    payload = json.loads(patch["body"])
    assert payload["title"] == "Run 1"
    assert payload["body"] == "<p>body</p>"
    assert payload["tags"] == ["i2as"]
    assert json.loads(payload["metadata"]) == {"run_id": "r1"}


def test_elabftw_falls_back_to_the_body_for_the_new_entry_id():
    """A deployment that echoes the id in the body instead of a header still works."""
    adapter, _ = _elab(
        {
            "POST /experiments": HttpResponse(201, {}, b'{"id":9}'),
            "PATCH /experiments/9": HttpResponse(200, {}, b"{}"),
        }
    )
    assert adapter.create_entry("T", None, "", [], {}).entry_id == "9"


def test_elabftw_create_without_an_id_is_an_eln_error():
    """An entry I2AS cannot address again is a failure, not a silent success."""
    adapter, _ = _elab({"POST /experiments": HttpResponse(201, {}, b"")})
    with pytest.raises(ElnError):
        adapter.create_entry("T", None, "", [], {})


def test_elabftw_updates_an_entry():
    """``update_entry`` rewrites body and metadata in one PATCH."""
    adapter, transport = _elab({"PATCH /experiments/42": HttpResponse(200, {}, b"{}")})
    adapter.update_entry(ElnEntryRef(entry_id="42"), "<p>new</p>", {"k": "v"})
    payload = json.loads(transport.calls[0]["body"])
    assert payload["body"] == "<p>new</p>"
    assert json.loads(payload["metadata"]) == {"k": "v"}


def test_elabftw_uploads_a_file_as_multipart(tmp_path):
    """The upload is a multipart POST carrying the filename and the bytes."""
    data = tmp_path / "run-0001.h5"
    data.write_bytes(b"\x89HDF\r\n\x1a\nPAYLOAD")
    adapter, transport = _elab(
        {"POST /experiments/42/uploads": HttpResponse(201, {}, b"{}")}
    )
    adapter.attach_file(ElnEntryRef(entry_id="42"), data, "Run data")

    call = transport.calls[0]
    assert call["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")
    assert b'name="file"; filename="run-0001.h5"' in call["body"]
    assert b"PAYLOAD" in call["body"]
    assert b'name="comment"' in call["body"]


def test_elabftw_upload_of_a_missing_file_is_an_eln_error(tmp_path):
    """An unreadable file fails before anything is sent."""
    adapter, transport = _elab()
    with pytest.raises(ElnError):
        adapter.attach_file(ElnEntryRef(entry_id="42"), tmp_path / "gone.h5")
    assert transport.calls == []


def test_elabftw_attaches_a_link_by_appending_to_the_body():
    """With no link endpoint, the path is appended to the entry body, escaped."""
    adapter, transport = _elab(
        {
            "GET /experiments/42": HttpResponse(200, {}, b'{"body":"<p>existing</p>"}'),
            "PATCH /experiments/42": HttpResponse(200, {}, b"{}"),
        }
    )
    adapter.attach_link(ElnEntryRef(entry_id="42"), "/data/<run>.h5", "Run data")
    body = json.loads(transport.calls[1]["body"])["body"]
    assert body.startswith("<p>existing</p>")
    assert "&lt;run&gt;" in body and "<run>" not in body
    assert "Run data" in body


def test_elabftw_reports_a_non_json_body_as_an_eln_error():
    """An HTML error page from a proxy is a backend failure, not a crash."""
    adapter, _ = _elab({"GET /users/me": HttpResponse(200, {}, b"<html>hi</html>")})
    with pytest.raises(ElnError):
        adapter.verify()


def test_elabftw_publishes_end_to_end_through_the_outbox(tmp_path):
    """The outbox drives the real backend exactly as it drives the sim twin."""
    data = _data_file(tmp_path)
    adapter, transport = _elab(
        {
            "POST /experiments": HttpResponse(
                201, {"location": "/api/v2/experiments/42"}, b""
            ),
            "PATCH /experiments/42": HttpResponse(200, {}, b"{}"),
            "POST /experiments/42/uploads": HttpResponse(201, {}, b"{}"),
        }
    )
    outbox = Outbox(tmp_path / "outbox.jsonl")
    outbox.enqueue(_job(tmp_path))

    result = outbox.drain(adapter)
    assert result.state == DRAIN_PUBLISHED
    assert result.entry.url.endswith("id=42")
    assert [call["method"] for call in transport.calls] == ["POST", "PATCH", "POST"]
    assert data.name.encode() in transport.calls[2]["body"]


def test_urllib_transport_refuses_to_verify_when_told_to(caplog):
    """Disabling TLS verification is possible, deliberate, and always logged."""
    import logging

    with caplog.at_level(logging.WARNING):
        UrllibTransport(verify_tls=False)
    assert any("verification is DISABLED" in record.message for record in caplog.records)
    UrllibTransport()  # the default verifies, silently


# ── The publisher (end to end, over a real ExperimentManager) ─────────────────


@pytest.fixture
def published_setup(tmp_path, qtbot):
    """A real ExperimentManager with an open experiment, plus a sim notebook.

    Yields ``(manager, publisher, adapter, run_manifest)``. The manifest is
    the shape the Orchestrator emits, and its data file really exists inside
    the experiment folder, so the whole path — relativize, resolve, attach —
    is exercised rather than stubbed.
    """
    from i2as.core.orchestrator import Orchestrator
    from i2as.core.station import build_station
    from i2as.session.eln.publisher import ElnPublisher
    from i2as.session.manager import ExperimentManager
    from i2as.session.models import User
    from i2as.session.store import ExperimentStore, UserRoster

    store = ExperimentStore(tmp_path / "experiments")
    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    orchestrator = Orchestrator(build_station("i2as/configs/sim_cryostat"), tick_interval_ms=10)
    manager = ExperimentManager(
        store=store, roster=roster, orchestrator=orchestrator, config_name="sim_cryostat"
    )
    experiment = manager.start_experiment("Sample A", "jdoe", {"sample_name": "A3"})

    data_file = store.data_dir(experiment.experiment_id) / "run-0001.h5"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_bytes(b"\x89HDF\r\n\x1a\n")

    started = {
        "run_id": "run-0001",
        "procedure": "Field Sweep",
        "kind": "run",
        "params": {"field_T": 1.5},
        "data_file": str(data_file),
        "started_utc": "2026-01-01T10:00:00+00:00",
    }
    orchestrator.run_started.emit(started)
    finished = dict(started, finished_utc="2026-01-01T11:00:00+00:00", status="done", reason="")

    settings = ElnSettings(
        enabled=True,
        backend="sim_eln",
        base_url="https://sim.example",
        api_key="k",
        tags=("i2as", "sim"),
        # No backoff: these tests step the queue by hand, and the backoff
        # itself is exercised against the Outbox directly.
        retry_base_s=0.0,
        retry_max_s=0.0,
    )
    adapter = SimElnAdapter({})
    publisher = ElnPublisher(manager, settings, adapter=adapter)
    orchestrator.run_finished.connect(publisher.on_run_finished)
    yield manager, publisher, adapter, finished
    publisher.stop()


def test_run_finished_publishes_one_entry_with_body_and_data(published_setup):
    """The exit criterion: one RunFinished, one entry, one attachment, one link."""
    manager, publisher, adapter, manifest = published_setup
    experiment = manager.current_experiment()

    manager._orchestrator.run_finished.emit(manifest)
    assert publisher.pending_count() == 1
    assert not adapter.entries, "queuing must touch no network"

    result = publisher.drain_once()
    assert result.state == DRAIN_PUBLISHED

    (entry,) = adapter.entries.values()
    assert entry["title"].startswith("Sample A — Field Sweep")
    assert "run-0001" in entry["body_html"] and "sim_cryostat" in entry["body_html"]
    assert entry["tags"] == ["i2as", "sim"]
    assert entry["metadata"]["run_id"] == "run-0001"
    assert adapter.uploads[0]["path"].endswith("run-0001.h5")

    run = manager.current_experiment().find_run("run-0001")
    assert run.published is True
    assert run.eln_link is not None
    assert run.eln_link.backend == "sim_eln"
    assert run.eln_link.entry_id and run.eln_link.url

    reloaded = manager.store.load(experiment.experiment_id).find_run("run-0001")
    assert reloaded.eln_link == run.eln_link, "the link is persisted, not just in memory"


def test_offline_leaves_the_job_queued_and_a_later_drain_publishes_once(published_setup):
    """The offline-first exit criterion, end to end."""
    manager, publisher, adapter, manifest = published_setup
    adapter.offline = True
    manager._orchestrator.run_finished.emit(manifest)

    assert publisher.drain_once().state == DRAIN_RETRY
    assert not adapter.entries
    assert publisher.pending_count() == 1
    assert manager.current_experiment().find_run("run-0001").eln_link is None

    adapter.offline = False
    assert publisher.drain_once().state == DRAIN_PUBLISHED
    assert len(adapter.entries) == 1
    assert publisher.drain_once().state == DRAIN_IDLE
    assert len(adapter.entries) == 1
    assert manager.current_experiment().find_run("run-0001").eln_link is not None


def test_a_duplicate_run_finished_publishes_exactly_once(published_setup):
    """Two identical RunFinished events are one job and one entry."""
    manager, publisher, adapter, manifest = published_setup
    manager._orchestrator.run_finished.emit(manifest)
    manager._orchestrator.run_finished.emit(manifest)
    assert publisher.pending_count() == 1

    assert publisher.drain_once().state == DRAIN_PUBLISHED
    assert publisher.drain_once().state == DRAIN_IDLE
    assert len(adapter.entries) == 1
    assert adapter.calls.count("create_entry") == 1


def test_the_publisher_accepts_a_run_finished_contract_event(published_setup):
    """``RunFinished`` and the Orchestrator's manifest dict are both accepted."""
    from i2as.core.events import RunFinished

    manager, publisher, adapter, manifest = published_setup
    event = RunFinished(run_id="run-0001", status="done", manifest=manifest)
    assert publisher.on_run_finished(event) == "publish_run:run-0001"
    assert publisher.drain_once().state == DRAIN_PUBLISHED
    assert len(adapter.entries) == 1


def test_nothing_is_published_without_an_open_experiment(published_setup):
    """No silent uploads of ad-hoc runs — an entry needs a record to belong to."""
    manager, publisher, adapter, manifest = published_setup
    manager.close_experiment()
    assert publisher.on_run_finished(manifest) == ""
    assert publisher.drain_once().state == DRAIN_IDLE
    assert not adapter.entries


def test_auto_publish_off_leaves_manual_export_as_the_only_trigger(published_setup):
    """With auto-publish off a finished run waits for an explicit export."""
    manager, publisher, adapter, manifest = published_setup
    publisher._settings = replace(publisher.settings, auto_publish=False)
    manager._orchestrator.run_finished.emit(manifest)
    assert publisher.pending_count() == 0

    assert publisher.export_run("run-0001") == "publish_run:run-0001"
    assert publisher.drain_once().state == DRAIN_PUBLISHED
    (entry,) = adapter.entries.values()
    assert "run-0001" in entry["body_html"]


def test_export_of_an_unknown_run_queues_nothing(published_setup):
    """A manual export names a recorded run or does nothing."""
    _, publisher, adapter, _ = published_setup
    assert publisher.export_run("no-such-run") == ""
    assert publisher.pending_count() == 0


def test_disabled_settings_publish_nothing_at_all(published_setup):
    """The default (no settings file) means nothing ever leaves the machine."""
    manager, publisher, adapter, manifest = published_setup
    publisher._settings = ElnSettings()
    assert publisher.on_run_finished(manifest) == ""
    assert publisher.export_run("run-0001") == ""
    assert publisher.drain_once().state == DRAIN_IDLE
    assert not adapter.entries


def test_a_queued_job_survives_a_restart(published_setup, qtbot):
    """A publisher built afterwards adopts the outbox left on disk and drains it."""
    from i2as.session.eln.publisher import ElnPublisher

    manager, publisher, adapter, manifest = published_setup
    adapter.offline = True
    manager._orchestrator.run_finished.emit(manifest)
    publisher.drain_once()
    publisher.stop()

    fresh_adapter = SimElnAdapter({})
    restarted = ElnPublisher(manager, publisher.settings, adapter=fresh_adapter)
    assert restarted.pending_count() == 1
    assert restarted.drain_once().state == DRAIN_PUBLISHED
    assert len(fresh_adapter.entries) == 1
    restarted.stop()


def test_publish_state_changes_are_announced(published_setup, qtbot):
    """The status chip's signal follows queued → published."""
    manager, publisher, adapter, manifest = published_setup
    seen = []
    publisher.publish_state_changed.connect(seen.append)

    manager._orchestrator.run_finished.emit(manifest)
    publisher.drain_once()

    assert [item["state"] for item in seen] == ["pending", "synced"]
    assert seen[0]["pending"] == 1 and seen[-1]["pending"] == 0


def test_run_published_signal_carries_the_link(published_setup, qtbot):
    """A confirmed entry is announced once, with its link."""
    manager, publisher, adapter, manifest = published_setup
    seen = []
    publisher.run_published.connect(seen.append)

    manager._orchestrator.run_finished.emit(manifest)
    publisher.drain_once()

    assert len(seen) == 1
    assert seen[0]["run_id"] == "run-0001"
    assert seen[0]["eln_link"]["backend"] == "sim_eln"


def test_backends_are_discovered_not_listed():
    """A new backend module is selectable the moment its file exists."""
    from i2as.session.eln.publisher import discover_backends

    backends = discover_backends()
    assert backends["elabftw"] is ElabFtwAdapter
    assert backends["sim_eln"] is SimElnAdapter


def test_an_unknown_backend_disables_publishing_rather_than_raising(published_setup):
    """A typo in the settings file switches the track off, loudly, not fatally."""
    manager, publisher, _, manifest = published_setup
    publisher._adapter = None
    publisher._settings = replace(publisher.settings, backend="not_a_backend")
    manager._orchestrator.run_finished.emit(manifest)
    assert publisher.drain_once().state == DRAIN_IDLE
    assert publisher.pending_count() == 1, "the job stays queued for a fixed settings file"


def test_a_link_recorded_after_the_experiment_closed_still_lands(published_setup):
    """An outbox job that drains a week later still stamps its own experiment."""
    manager, publisher, adapter, manifest = published_setup
    experiment_id = manager.current_experiment().experiment_id
    adapter.offline = True
    manager._orchestrator.run_finished.emit(manifest)
    publisher.drain_once()

    manager.close_experiment()
    adapter.offline = False
    assert publisher.drain_once().state == DRAIN_PUBLISHED

    stored = manager.store.load(experiment_id).find_run("run-0001")
    assert stored.eln_link is not None and stored.published is True


# ══════════════════════════════════════════════════════════════════════════
# LLM drafting (drafting.py) — the draft prompt standard and the Draft client
# ══════════════════════════════════════════════════════════════════════════


DRAFT_MANIFEST = {
    "run_id": "run-0001",
    "procedure": "Field Sweep",
    "kind": "run",
    "params": {"field_T": 1.5, "temperature_K": 4.2},
    "started_utc": "2026-01-01T10:00:00+00:00",
    "finished_utc": "2026-01-01T11:00:00+00:00",
    "status": "done",
    "reason": "",
}

DRAFT_STATS = {
    "voltage_V": {
        "column": "voltage_V",
        "count": 51,
        "min": -1.0,
        "max": 1.0,
        "mean": 0.0,
        "std": 0.5,
        "first": -1.0,
        "last": 1.0,
    }
}

DRAFT_STATION = {
    "setup": "sim_cryostat",
    "instruments": [
        {"name": "magnet_z", "kind": "magnet", "vi_class": "SuperconductingMagnetVI"},
        {"name": "sample_temp", "kind": "temperature", "vi_class": "TemperatureVI"},
    ],
}


def _draft_request(**overrides):
    """Build a DraftRequest over the sample facts above."""
    from i2as.session.eln.drafting import DraftRequest

    fields = {
        "run_id": "run-0001",
        "experiment_id": "20260101_sample_a",
        "experiment_title": "Sample A",
        "manifest": dict(DRAFT_MANIFEST),
        "stats": {name: dict(row) for name, row in DRAFT_STATS.items()},
        "station": dict(DRAFT_STATION),
        "status": {"state": "IDLE", "vi_faults": {}, "held_vi_names": []},
        "setup": {"config_name": "sim_cryostat", "instruments": {}},
        "template_id": "tpl-7",
    }
    fields.update(overrides)
    return DraftRequest(**fields)


def test_the_draft_prompt_carries_every_fact_in_a_fixed_order():
    """The prompt standard: run, parameters, statistics, station, state, note."""
    from i2as.session.eln.drafting import render_draft_prompt

    prompt = render_draft_prompt(_draft_request(operator_note="check the drift"))

    headings = [line for line in prompt.splitlines() if line.isupper()]
    assert headings == [
        "RUN",
        "PARAMETERS",
        "COLUMN STATISTICS",
        "STATION",
        "STATE AT RUN END",
        "OPERATOR NOTE",
    ]
    assert "procedure: Field Sweep" in prompt
    assert "field_T: 1.5" in prompt
    assert "voltage_V: count=51" in prompt
    assert "setup: sim_cryostat" in prompt
    assert "instrument: magnet_z" in prompt
    assert "check the drift" in prompt


def test_the_draft_prompt_is_deterministic_and_sorted():
    """The same request renders byte-identical text, whatever the dict order."""
    from i2as.session.eln.drafting import render_draft_prompt

    shuffled = dict(DRAFT_MANIFEST)
    shuffled["params"] = {"temperature_K": 4.2, "field_T": 1.5}

    first = render_draft_prompt(_draft_request())
    second = render_draft_prompt(_draft_request(manifest=shuffled))

    assert first == second
    assert first.index("field_T") < first.index("temperature_K")


def test_a_draft_carries_the_facts_the_prose_is_checked_against():
    """The body is the drafted prose ABOVE the run's own escaped fact tables."""
    from i2as.session.eln.drafting import DRAFT_TAG, FakeDraftClient, draft_entry

    client = FakeDraftClient(
        "TITLE: Field sweep at 1.5 T\nSUMMARY:\nThe sweep completed cleanly."
    )

    draft = draft_entry(_draft_request(), client)

    assert draft.title == "Field sweep at 1.5 T"
    assert "The sweep completed cleanly." in draft.body_html
    assert draft.body_html.index("The sweep completed cleanly.") < draft.body_html.index(
        "Field Sweep"
    ), "the prose is above the facts a reviewer checks it against"
    for fact in ("Field Sweep", "field_T", "1.5", "voltage_V", "sim_cryostat"):
        assert fact in draft.body_html
    assert draft.tags == [DRAFT_TAG, "Field Sweep"]
    assert client.calls[0][1].startswith("RUN\n"), "the client is asked the facts"


def test_a_drafted_body_is_self_contained_and_escapes_the_model():
    """Model output is a value, never markup — the entry pulls in nothing.

    A URL the model mentions survives as inert text: escaping is what removes
    every way it could be fetched or followed, so nothing in the body is ever
    a live resource.
    """
    from i2as.session.eln.drafting import FakeDraftClient, draft_entry

    client = FakeDraftClient(
        "TITLE: <b>hi</b>\nSUMMARY:\n<script>alert(1)</script> see https://evil.example"
    )

    draft = draft_entry(_draft_request(), client)

    lowered = draft.body_html.lower()
    for forbidden in ("<script", "<link", "<img", "<iframe", "<a ", "href="):
        assert forbidden not in lowered
    assert "&lt;script&gt;" in draft.body_html
    assert draft.title == "<b>hi</b>", "the title is plain text; backends escape it"


def test_the_prompt_digest_is_stable_and_moves_with_the_facts():
    """Two drafts of one run are provably the same question; a changed fact is visible."""
    from i2as.session.eln.drafting import FakeDraftClient, draft_entry

    first = draft_entry(_draft_request(), FakeDraftClient())
    again = draft_entry(_draft_request(), FakeDraftClient())
    changed = draft_entry(_draft_request(operator_note="something new"), FakeDraftClient())

    assert first.prompt_digest == again.prompt_digest
    assert len(first.prompt_digest) == 64
    assert changed.prompt_digest != first.prompt_digest


def test_a_completion_missing_its_markers_still_drafts():
    """The marker shape parses tolerantly: no marker is a usable draft, not an error."""
    from i2as.session.eln.drafting import FakeDraftClient, draft_entry

    draft = draft_entry(_draft_request(), FakeDraftClient("Just some prose."))

    assert draft.title == "Sample A — Field Sweep — 2026-01-01T10:00:00+00:00"
    assert "Just some prose." in draft.body_html


def test_a_draft_reports_what_it_cost_from_the_settings_price_table():
    """cost_usd comes from the settings table, and an unpriced model reports 0.0."""
    from i2as.session.eln.drafting import FakeDraftClient, draft_entry
    from i2as.session.eln.settings import AssistantSettings

    settings = AssistantSettings(prices={"m-1": {"input": 5.0, "output": 25.0}})
    client = FakeDraftClient(model="m-1", input_tokens=1_000_000, output_tokens=100_000)

    draft = draft_entry(_draft_request(), client, settings)

    assert draft.model == "m-1"
    assert draft.cost_usd == pytest.approx(5.0 + 2.5)
    assert draft.cost_line() == {
        "model": "m-1",
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "cost_usd": pytest.approx(7.5),
    }

    unpriced = draft_entry(_draft_request(), FakeDraftClient(model="nope"), settings)
    assert unpriced.cost_usd == 0.0


def test_the_max_token_cap_reaches_the_client():
    """A runaway completion is bounded by the settings, not by the model."""
    from i2as.session.eln.drafting import FakeDraftClient, draft_entry
    from i2as.session.eln.settings import AssistantSettings

    client = FakeDraftClient()
    draft_entry(_draft_request(), client, AssistantSettings(max_tokens=321))

    assert client.calls[0][2] == 321


def test_a_draft_client_failure_is_one_eln_error():
    """One exception type out of the whole package, the model included."""
    from i2as.session.eln.drafting import FakeDraftClient, draft_entry

    with pytest.raises(ElnError):
        draft_entry(_draft_request(), FakeDraftClient(offline=True))


def test_the_anthropic_client_says_so_when_its_optional_sdk_is_absent():
    """A missing optional extra is one clear ElnError, at construction only."""
    from i2as.session.eln import drafting

    # Importing the module, rendering prompts and drafting against the fake all
    # work regardless; only building the real client needs the SDK.
    assert drafting.render_draft_prompt(_draft_request())

    try:
        import anthropic  # noqa: F401
    except ImportError:
        with pytest.raises(ElnError, match="i2as\\[assistant\\]"):
            drafting.AnthropicDraftClient()
    else:  # pragma: no cover - only when the optional extra is installed
        pytest.skip("the anthropic extra is installed; absence cannot be exercised")


def test_the_assistant_settings_redact_the_key_and_carry_a_price_table(tmp_path):
    """The assistant's key follows the ELN key's rule exactly: never logged."""
    from i2as.session.eln.settings import (
        ASSISTANT_API_KEY_ENV_VAR,
        DEFAULT_MODEL_PRICES,
        AssistantSettings,
    )

    settings = AssistantSettings(api_key="sk-secret")

    assert "sk-secret" not in repr(settings)
    assert settings.to_dict()["api_key"] == "***"
    assert settings.to_dict(include_secret=True)["api_key"] == "sk-secret"
    assert AssistantSettings().prices == DEFAULT_MODEL_PRICES

    path = tmp_path / "eln.json"
    path.write_text(
        json.dumps({"assistant": {"enabled": True, "model": "m-2"}}), encoding="utf-8"
    )
    loaded = load_eln_settings(path)
    assert loaded.assistant.enabled is True and loaded.assistant.model == "m-2"
    assert loaded.assistant.prices == DEFAULT_MODEL_PRICES

    import os

    os.environ[ASSISTANT_API_KEY_ENV_VAR] = "sk-from-env"
    try:
        assert load_eln_settings(path).assistant.api_key == "sk-from-env"
    finally:
        del os.environ[ASSISTANT_API_KEY_ENV_VAR]


# ══════════════════════════════════════════════════════════════════════════
# Publishing an approved draft (publisher.export_draft)
# ══════════════════════════════════════════════════════════════════════════


def test_an_approved_draft_is_queued_as_one_ordinary_job(published_setup):
    """A draft is data: the same journal, the same drain, only the text differs."""
    from i2as.session.eln.drafting import DraftEntry

    manager, publisher, adapter, _manifest = published_setup

    draft = DraftEntry(
        title="Drafted title",
        body_html="<p>Drafted prose over the facts.</p>",
        tags=["draft", "Field Sweep"],
        model="m-1",
        input_tokens=1000,
        output_tokens=200,
        cost_usd=0.01,
        prompt_digest="d" * 64,
    )

    job_id = publisher.export_draft("run-0001", draft)
    assert job_id == "publish_run:run-0001"

    assert publisher.drain_once().state == DRAIN_PUBLISHED
    (entry,) = adapter.entries.values()
    assert entry["title"] == "Drafted title"
    assert "Drafted prose over the facts." in entry["body_html"]
    # The notebook's own standing tags and the draft's own, merged and sorted.
    assert entry["tags"] == ["Field Sweep", "draft", "i2as", "sim"]
    assert entry["metadata"]["run_id"] == "run-0001"
    assert entry["metadata"]["draft_model"] == "m-1"
    assert entry["metadata"]["draft_prompt_digest"] == "d" * 64

    run = manager.current_experiment().find_run("run-0001")
    assert run.published is True and run.eln_link is not None


def test_an_approved_draft_is_queued_from_its_json_dict(published_setup):
    """The run record stores a draft as JSON; export_draft loads it tolerantly."""
    _manager, publisher, adapter, _manifest = published_setup

    job_id = publisher.export_draft(
        "run-0001",
        {"title": "From JSON", "body_html": "<p>x</p>", "tags": ["draft"]},
    )

    assert job_id == "publish_run:run-0001"
    assert publisher.drain_once().state == DRAIN_PUBLISHED
    (entry,) = adapter.entries.values()
    assert entry["title"] == "From JSON"


def test_an_approved_draft_does_not_publish_a_run_twice(published_setup):
    """Idempotent by the same job_id: a queued run is not requeued under a draft."""
    from i2as.session.eln.drafting import DraftEntry

    manager, publisher, adapter, manifest = published_setup
    manager._orchestrator.run_finished.emit(manifest)
    assert publisher.pending_count() == 1

    publisher.export_draft("run-0001", DraftEntry(title="Drafted", body_html="<p>y</p>"))

    assert publisher.pending_count() == 1, "one run, one entry, however it was queued"


def test_a_draft_is_never_queued_while_publishing_is_off(published_setup, tmp_path):
    """The track's master switch binds the drafting path exactly as the rest."""
    from i2as.session.eln.drafting import DraftEntry
    from i2as.session.eln.publisher import ElnPublisher

    manager, _publisher, adapter, _manifest = published_setup
    off = ElnPublisher(manager, ElnSettings(enabled=False), adapter=adapter)

    assert off.export_draft("run-0001", DraftEntry(title="t", body_html="<p>b</p>")) == ""
    off.stop()


def test_a_pending_draft_rides_on_the_run_record_and_survives_json():
    """An unapproved draft is parked on the run record, JSON-safe and tolerant."""
    from i2as.session.models import RunRecord

    record = RunRecord(run_id="run-0001", pending_eln_draft={"title": "t", "tags": ["draft"]})

    round_tripped = RunRecord.from_dict(json.loads(json.dumps(record.to_dict())))
    assert round_tripped.pending_eln_draft == {"title": "t", "tags": ["draft"]}

    assert RunRecord.from_dict({"run_id": "r"}).pending_eln_draft == {}
    assert RunRecord.from_dict({"run_id": "r", "pending_eln_draft": 7}).pending_eln_draft == {}


# ══════════════════════════════════════════════════════════════════════════
# The approval gate (ExperimentManager.approve_eln_draft)
# ══════════════════════════════════════════════════════════════════════════


def test_approving_a_pending_draft_queues_exactly_one_job(published_setup):
    """The human's half of the gate: park, approve, one entry, nothing pending."""
    manager, publisher, adapter, _manifest = published_setup
    manager.attach_eln_publisher(publisher)

    assert manager.set_pending_eln_draft(
        "run-0001", {"title": "Awaiting a human", "body_html": "<p>prose</p>"}
    )
    assert manager.pending_eln_draft("run-0001")["title"] == "Awaiting a human"
    assert publisher.pending_count() == 0, "a pending draft publishes nothing"

    job_id = manager.approve_eln_draft("run-0001")

    assert job_id == "publish_run:run-0001"
    assert publisher.pending_count() == 1
    assert manager.pending_eln_draft("run-0001") == {}, "an approved draft is spent"

    assert publisher.drain_once().state == DRAIN_PUBLISHED
    (entry,) = adapter.entries.values()
    assert entry["title"] == "Awaiting a human"


def test_a_pending_draft_survives_a_reload(published_setup):
    """The proposal is on the record, not in memory: a restart still finds it."""
    manager, _publisher, _adapter, _manifest = published_setup
    experiment_id = manager.current_experiment().experiment_id

    manager.set_pending_eln_draft("run-0001", {"title": "Later"})

    stored = manager.store.load(experiment_id).find_run("run-0001")
    assert stored.pending_eln_draft == {"title": "Later"}


def test_an_unqueueable_draft_stays_pending(published_setup):
    """A publisher that queued nothing leaves the proposal exactly where it was."""
    manager, _publisher, adapter, _manifest = published_setup
    from i2as.session.eln.publisher import ElnPublisher

    off = ElnPublisher(manager, ElnSettings(enabled=False), adapter=adapter)
    manager.attach_eln_publisher(off)
    manager.set_pending_eln_draft("run-0001", {"title": "Still waiting"})

    assert manager.approve_eln_draft("run-0001") == ""
    assert manager.pending_eln_draft("run-0001") == {"title": "Still waiting"}
    off.stop()


def test_approval_without_a_draft_or_a_publisher_queues_nothing(published_setup):
    """Every refusal is a logged "" — approval is a GUI action and never raises."""
    manager, publisher, _adapter, _manifest = published_setup

    assert manager.approve_eln_draft("run-0001") == "", "nothing is pending"

    manager.set_pending_eln_draft("run-0001", {"title": "t"})
    assert manager.approve_eln_draft("run-0001") == "", "no publisher is attached"

    manager.attach_eln_publisher(publisher)
    assert manager.approve_eln_draft("no-such-run") == ""
    assert manager.set_pending_eln_draft("no-such-run", {"title": "t"}) is False


# ══════════════════════════════════════════════════════════════════════════
# The analysis stage: settings, pending entries, attachments, routing
# ══════════════════════════════════════════════════════════════════════════


def _report(**overrides):
    """Build an ``ok`` analysis report with one of everything."""
    from i2as.analysis.report import AnalysisReport, FigureRef, ResultValue, TableSpec

    fields = {
        "run_id": "run-0001",
        "recipe": "generic_sweep",
        "recipe_digest": "a" * 64,
        "summary": ("The field swept cleanly.", "Nothing anomalous."),
        "results": (
            ResultValue(name="Critical field", value=1.25, unit="T", uncertainty=0.03, note="fit"),
        ),
        "figures": (FigureRef(file="overview.png", caption="Overview"),),
        "tables": (TableSpec.build("Columns", ["column", "points"], [["B", 100]]),),
        "tags": ("sweep",),
        "warnings": ("one column had no finite values",),
    }
    fields.update(overrides)
    return AnalysisReport(**fields)


def test_the_analysis_section_round_trips_through_a_saved_file(tmp_path):
    """The analysis switches live in the same user-level file, saved and read back."""
    import os

    from i2as.session.eln.settings import AnalysisSettings, save_eln_settings

    settings = ElnSettings(
        enabled=True,
        base_url="https://elab.example.org",
        api_key="file-key",
        analysis=AnalysisSettings(
            enabled=True,
            timeout_s=45.0,
            include_fact_tables=True,
            attach_data_file=True,
            recipes={"FieldSweep": "field_sweep_overview"},
        ),
    )
    path = tmp_path / "nested" / "eln-settings.json"

    assert save_eln_settings(settings, path) == path
    assert path.is_file(), "the parent directory is created"
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600, "the key is the owner's alone"

    monkey_free = load_eln_settings(path)
    assert monkey_free == settings
    assert monkey_free.analysis.recipes == {"FieldSweep": "field_sweep_overview"}
    assert not path.with_name(path.name + ".tmp").exists(), "written atomically"


def test_a_mangled_analysis_section_degrades_to_off(tmp_path):
    """Junk in the file is "analysis is off", never a broken startup."""
    from i2as.session.eln.settings import AnalysisSettings

    path = tmp_path / "eln-settings.json"
    path.write_text(
        json.dumps({"enabled": True, "analysis": {"enabled": "yes", "recipes": {"P": ["a"]}}}),
        encoding="utf-8",
    )

    analysis = load_eln_settings(path).analysis
    assert analysis == AnalysisSettings(), "every junk field falls back to its default"
    assert ElnSettings.from_dict({"analysis": 7}).analysis == AnalysisSettings()


def test_a_pending_entry_carries_its_attachments_and_its_source():
    """Every pending entry is one shape; the stage that made it is a field on it."""
    from i2as.session.eln.drafting import SOURCE_ANALYSIS, DraftEntry

    entry = DraftEntry(
        title="Analysed",
        body_html="<p>x</p>",
        attachments=[{"path": "/reports/overview.png", "comment": "Overview"}],
        attach_data_file=False,
        source=SOURCE_ANALYSIS,
        metadata={"recipe": "generic_sweep", "recipe_digest": "a" * 64},
    )

    assert DraftEntry.from_dict(json.loads(json.dumps(entry.to_dict()))) == entry
    assert DraftEntry().source == "model" and DraftEntry().attach_data_file is True
    junk = DraftEntry.from_dict(
        {"attachments": [7, {"comment": "no path"}, {"path": "p"}], "metadata": 3, "source": None}
    )
    assert junk.attachments == [{"path": "p", "comment": ""}]
    assert junk.metadata == {} and junk.source == "model"


def test_an_outbox_job_carries_its_attachments_through_the_journal(tmp_path):
    """The journal is the job: attachments and the attach flag survive a restart."""
    outbox = Outbox(tmp_path / "outbox.jsonl")
    outbox.enqueue(
        _job(
            tmp_path,
            attach_data_file=False,
            attachments=[{"path": "/r/overview.png", "comment": "Overview"}],
        )
    )

    (reloaded,) = Outbox(tmp_path / "outbox.jsonl").jobs()
    assert reloaded.attach_data_file is False
    assert reloaded.attachments == [{"path": "/r/overview.png", "comment": "Overview"}]

    defaults = OutboxJob.from_dict({"job_id": "j"})
    assert defaults.attach_data_file is True and defaults.attachments == []
    junk = OutboxJob.from_dict({"job_id": "j", "attachments": [1, {}], "attach_data_file": "no"})
    assert junk.attachments == [] and junk.attach_data_file is True


def test_outbox_attaches_every_figure_after_the_data_file(tmp_path):
    """An analysed entry's figures ride the same upload path, in order."""
    data = _data_file(tmp_path)
    figure = tmp_path / "overview.png"
    figure.write_bytes(b"PNG")
    outbox = Outbox(tmp_path / "outbox.jsonl")
    outbox.enqueue(
        _job(tmp_path, attachments=[{"path": str(figure), "comment": "Overview"}])
    )
    adapter = SimElnAdapter({})

    assert outbox.drain(adapter).state == DRAIN_PUBLISHED
    assert [upload["path"] for upload in adapter.uploads] == [str(data), str(figure)]
    assert adapter.uploads[1]["comment"] == "Overview"


def test_outbox_leaves_the_data_file_alone_when_the_job_says_so(tmp_path):
    """A report that asked for figures alone gets figures alone."""
    _data_file(tmp_path)
    figure = tmp_path / "overview.png"
    figure.write_bytes(b"PNG")
    outbox = Outbox(tmp_path / "outbox.jsonl")
    outbox.enqueue(
        _job(
            tmp_path,
            attach_data_file=False,
            attachments=[{"path": str(figure), "comment": "Overview"}],
        )
    )
    adapter = SimElnAdapter({})

    assert outbox.drain(adapter).state == DRAIN_PUBLISHED
    assert [upload["path"] for upload in adapter.uploads] == [str(figure)]
    assert not adapter.links


def test_outbox_links_a_figure_that_is_not_there(tmp_path):
    """A missing figure is recorded where it lives, exactly as a missing data file is."""
    _data_file(tmp_path)
    outbox = Outbox(tmp_path / "outbox.jsonl")
    outbox.enqueue(
        _job(tmp_path, attachments=[{"path": str(tmp_path / "gone.png"), "comment": "Overview"}])
    )
    adapter = SimElnAdapter({})

    assert outbox.drain(adapter).state == DRAIN_PUBLISHED
    assert adapter.links[0]["url"] == str(tmp_path / "gone.png")
    assert "not found" in adapter.links[0]["comment"]
    assert adapter.links[0]["comment"].startswith("Overview")


def test_the_analysed_body_leads_with_the_result_and_ends_with_the_provenance():
    """The order a physicist reads it in, self-contained and deterministic."""
    from i2as.session.eln.templates import render_analysed_body

    report = _report()
    facts = {**MANIFEST, "params_digest": "digest-1"}
    body = render_analysed_body(
        report,
        facts,
        experiment_id="exp-1",
        experiment_title="Sample A",
        setup={"config_name": "sim_cryostat"},
        data_path="/data/exp-1/data/run-0001.h5",
    )

    assert body == render_analysed_body(
        report.to_dict(),
        facts,
        experiment_id="exp-1",
        experiment_title="Sample A",
        setup={"config_name": "sim_cryostat"},
        data_path="/data/exp-1/data/run-0001.h5",
    ), "the same report renders byte-identical HTML, dict or record"

    assert body.index("The field swept cleanly.") < body.index("Critical field")
    assert body.index("Critical field") < body.index("Overview")
    assert body.index("Overview") < body.index("Provenance")
    assert "1.25 T" in body and "± 0.03" in body
    assert "(attached as overview.png)" in body, "the figure is named, never embedded"
    assert "one column had no finite values" in body
    assert "digest-1" in body and "generic_sweep" in body and "a" * 64 in body
    assert "run-0001.h5" in body and "/data/exp-1" not in body, "the file name, not the path"

    lowered = body.lower()
    for forbidden in ("<script", "<link", "<img", "<iframe", "http://", "https://"):
        assert forbidden not in lowered


def test_the_analysed_body_appends_the_fact_tables_only_when_asked():
    """The point of the stage: the result reaches the notebook, not the raw facts."""
    from i2as.session.eln.templates import render_analysed_body

    facts = {**MANIFEST, "summary_stats": {"B": {"count": 3, "min": 0.0, "max": 1.0}}}
    lean = render_analysed_body(_report(), facts, data_path="/d/run-0001.h5")
    assert "Parameters" not in lean and "Column statistics" not in lean

    full = render_analysed_body(
        _report(include_fact_tables=True), facts, data_path="/d/run-0001.h5"
    )
    assert "Parameters" in full and "rate_T_per_s" in full
    assert "Column statistics" in full


def test_a_failed_report_says_so_in_the_title_and_the_body():
    """A failure is visible exactly where the result would have been."""
    from i2as.analysis.report import REPORT_FAILED
    from i2as.session.eln.templates import render_analysed_body, render_analysed_title

    report = _report(
        status=REPORT_FAILED, error="ZeroDivisionError: division by zero\nTraceback ..."
    )

    title = render_analysed_title(report, MANIFEST, "Sample A")
    assert title.startswith("Sample A — Field Sweep") and title.endswith("analysis failed")
    assert render_analysed_title(_report(), MANIFEST, "Sample A") == render_run_title(
        MANIFEST, "Sample A"
    )

    body = render_analysed_body(report, MANIFEST, data_path="/d/run-0001.h5")
    assert body.index("Analysis failed") < body.index("Provenance")
    assert "ZeroDivisionError: division by zero" in body


def test_a_finished_run_goes_to_the_analysis_stage_instead_of_the_queue(published_setup):
    """With the analysis stage on, nothing is queued — the run is analysed first."""
    from i2as.session.eln.settings import AnalysisSettings

    manager, publisher, _adapter, manifest = published_setup
    publisher._settings = replace(publisher.settings, analysis=AnalysisSettings(enabled=True))
    asked: list[tuple] = []
    publisher.analysis_requested.connect(lambda *args: asked.append(args))

    manager._orchestrator.run_finished.emit(manifest)

    assert publisher.pending_count() == 0, "nothing publishes until a human approves"
    ((run_id, sent_manifest, data_path),) = asked
    assert run_id == "run-0001"
    assert sent_manifest["procedure"] == "Field Sweep"
    assert data_path.endswith("run-0001.h5")


def test_auto_publish_says_nothing_once_the_analysis_stage_is_on(published_setup):
    """The analysis fork is taken before auto-publish is even consulted."""
    from i2as.session.eln.settings import AnalysisSettings

    manager, publisher, _adapter, manifest = published_setup
    publisher._settings = replace(
        publisher.settings, auto_publish=False, analysis=AnalysisSettings(enabled=True)
    )
    asked: list[tuple] = []
    publisher.analysis_requested.connect(lambda *args: asked.append(args))

    manager._orchestrator.run_finished.emit(manifest)

    assert len(asked) == 1 and publisher.pending_count() == 0


def test_with_the_analysis_stage_off_a_finished_run_is_queued_as_before(published_setup):
    """The default path is untouched: no analysis is asked for and the job is queued."""
    manager, publisher, _adapter, manifest = published_setup
    asked: list[tuple] = []
    publisher.analysis_requested.connect(lambda *args: asked.append(args))

    manager._orchestrator.run_finished.emit(manifest)

    assert asked == []
    assert publisher.pending_count() == 1


def test_park_facts_entry_parks_the_reason_above_the_facts(published_setup):
    """A run whose analysis failed still has a complete, correct entry waiting."""
    manager, publisher, _adapter, _manifest = published_setup

    assert publisher.park_facts_entry("run-0001", warning="the recipe raised") is True

    pending = manager.pending_eln_draft("run-0001")
    assert pending["source"] == "facts" and pending["attachments"] == []
    assert "Analysis unavailable" in pending["body_html"]
    assert "the recipe raised" in pending["body_html"]
    assert "run-0001" in pending["body_html"], "the facts are all still there"
    assert pending["title"].startswith("Sample A — Field Sweep")
    assert publisher.pending_count() == 0, "parking publishes nothing"

    assert publisher.park_facts_entry("no-such-run") is False


def test_an_analysed_entry_is_parked_and_its_approval_queues_one_job(
    published_setup, tmp_path
):
    """End to end: report in, entry parked, human approves, one job with the figure."""
    manager, publisher, adapter, _manifest = published_setup
    manager.attach_eln_publisher(publisher)
    report_dir = tmp_path / "analysis" / "run-0001"
    report_dir.mkdir(parents=True)
    (report_dir / "overview.png").write_bytes(b"PNG")

    assert publisher.export_report("run-0001", _report(), report_dir) is True

    pending = manager.pending_eln_draft("run-0001")
    assert pending["source"] == "analysis"
    assert pending["attach_data_file"] is False, "the report asked for figures alone"
    assert pending["attachments"] == [
        {"path": str(report_dir / "overview.png"), "comment": "Overview"}
    ]
    assert pending["metadata"] == {"recipe": "generic_sweep", "recipe_digest": "a" * 64}
    assert publisher.pending_count() == 0, "an analysed entry publishes nothing by itself"

    assert manager.approve_eln_draft("run-0001") == "publish_run:run-0001"
    assert manager.pending_eln_draft("run-0001") == {}
    assert publisher.drain_once().state == DRAIN_PUBLISHED

    (entry,) = adapter.entries.values()
    assert "The field swept cleanly." in entry["body_html"]
    assert entry["tags"] == ["i2as", "sim", "sweep"]
    assert entry["metadata"]["draft_source"] == "analysis"
    assert entry["metadata"]["recipe"] == "generic_sweep"
    assert entry["metadata"]["recipe_digest"] == "a" * 64
    assert [upload["path"] for upload in adapter.uploads] == [
        str(report_dir / "overview.png")
    ], "the figure is attached; the raw data file is not"


def test_an_analysed_entry_can_keep_the_data_file_attached(published_setup, tmp_path):
    """``attach_data_file`` on the report is what decides, not the publisher."""
    manager, publisher, adapter, _manifest = published_setup
    manager.attach_eln_publisher(publisher)
    report_dir = tmp_path / "analysis" / "run-0001"
    report_dir.mkdir(parents=True)

    publisher.export_report(
        "run-0001", _report(figures=(), attach_data_file=True), report_dir
    )
    manager.approve_eln_draft("run-0001")
    assert publisher.drain_once().state == DRAIN_PUBLISHED

    assert adapter.uploads[0]["path"].endswith("run-0001.h5")


def test_export_report_refuses_an_unknown_run(published_setup, tmp_path):
    """No run, no entry — logged, never raised."""
    _manager, publisher, _adapter, _manifest = published_setup
    assert publisher.export_report("no-such-run", _report(), tmp_path) is False


def test_reload_settings_starts_and_stops_the_drain_timer(published_setup):
    """Save in the eLab setup dialog, and the network follows immediately."""
    from i2as.session.eln.settings import AnalysisSettings

    _manager, publisher, _adapter, _manifest = published_setup
    publisher.start()
    assert publisher._timer.isActive()

    publisher.reload_settings(replace(publisher.settings, enabled=False))
    assert not publisher._timer.isActive(), "switching the track off stops the network"
    assert publisher.settings.enabled is False
    assert publisher.status()["state"] == "disabled"

    publisher.reload_settings(
        replace(publisher.settings, enabled=True, analysis=AnalysisSettings(enabled=True))
    )
    assert publisher._timer.isActive()
    assert publisher.settings.analysis.enabled is True


def test_discarding_a_pending_entry_leaves_the_run_and_publishes_nothing(published_setup):
    """The other half of the gate: the proposal goes, the run stays."""
    manager, publisher, _adapter, _manifest = published_setup

    assert publisher.park_facts_entry("run-0001") is True
    assert manager.pending_eln_draft("run-0001")["source"] == "facts"

    assert manager.discard_pending_eln_draft("run-0001") is True
    assert manager.pending_eln_draft("run-0001") == {}
    assert manager.current_experiment().find_run("run-0001") is not None
    assert publisher.pending_count() == 0

    assert manager.discard_pending_eln_draft("run-0001") is False, "nothing left to drop"
    assert manager.discard_pending_eln_draft("no-such-run") is False


def test_default_settings_tag_and_metadata_source_carry_the_application_name():
    """Every entry the app creates is tagged and sourced as I2AS's."""
    assert ElnSettings().tags == ("i2as",)
    assert render_run_metadata(MANIFEST)["source"] == "i2as"
