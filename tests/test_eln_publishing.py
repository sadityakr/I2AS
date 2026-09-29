"""Tests for the notebook bridge: accounts, the outbox, and publishing end to end.

The service runs synchronously here (``synchronous=True``) and calls blocks
through the in-process runner, so a whole link → approve → publish cycle is
deterministic; the shipped simulated notebook keeps its pages in a JSON file
the tests read back to see exactly what reached the page.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pytest

from i2as.analysis.bundle import Producer, seal_bundle
from i2as.blocks.connector import ElnEntryRef
from i2as.blocks.discovery import shipped_dir
from i2as.session.app_config import PublishingSettings
from i2as.session.credentials import CredentialStore, credential_key
from i2as.session.eln import BlockCatalog, ElnService, InProcessBlockRunner, PublishError
from i2as.session.eln.ledger import Ledger
from i2as.session.eln.outbox import (
    ATTENTION_AUTH,
    ATTENTION_CHANGED,
    STATE_DONE,
    STATE_NEEDS_ATTENTION,
    STATE_PENDING,
    Outbox,
    OutboxJob,
)
from i2as.session.eln.publishing import JobExecutor
from i2as.session.models import LinkedItem, User
from i2as.session.user_profile import ElnAccount, ElnUserSettings, UserProfile, UserProfileStore

# ── Credentials and profiles ──────────────────────────────────────────────


class FakeKeyring:
    """A keyring stand-in: a dict, plus the ``get_keyring`` the store names."""

    def __init__(self) -> None:
        self.passwords: dict[tuple[str, str], str] = {}

    def get_password(self, service, key):
        return self.passwords.get((service, key))

    def set_password(self, service, key, secret):
        self.passwords[(service, key)] = secret

    def delete_password(self, service, key):
        self.passwords.pop((service, key), None)

    def get_keyring(self):
        return self


def test_credentials_prefer_the_keyring_and_honour_the_environment(tmp_path):
    ring = FakeKeyring()
    store = CredentialStore(keyring_module=ring, fallback_path=tmp_path / "c.json", environ={})
    key = credential_key("eln", "lab", "jdoe")
    store.set(key, "K1")
    assert ring.passwords == {("I2AS", "eln/lab/jdoe"): "K1"} and not (tmp_path / "c.json").exists()
    assert store.get(key) == "K1" and store.has(key)
    store.set(key, "")
    assert not store.has(key)
    overridden = CredentialStore(keyring_module=ring, fallback_path=tmp_path / "c.json", environ={"I2AS_ELAB_APIKEY": "ENV"})
    assert overridden.get(key) == "ENV", "the older eLab variable still overrides"


def test_credentials_fall_back_to_an_owner_only_file(tmp_path):
    store = CredentialStore(fallback_path=tmp_path / "c.json", environ={}, detect=False)
    store.set("eln/lab/jdoe", "K2")
    assert json.loads((tmp_path / "c.json").read_text()) == {"eln/lab/jdoe": "K2"}
    assert "file" in store.backend_name
    store.delete("eln/lab/jdoe")
    assert store.get("eln/lab/jdoe") == ""


def test_a_profile_never_holds_a_secret_and_migrates_the_old_settings_once(tmp_path):
    creds = CredentialStore(fallback_path=tmp_path / "c.json", environ={}, detect=False)
    legacy = tmp_path / "eln-settings.json"
    legacy.write_text(json.dumps({"enabled": True, "base_url": "https://e.org", "api_key": "OLD", "template_id": "7", "assistant": {"enabled": True, "api_key": "A"}}), encoding="utf-8")
    store = UserProfileStore(tmp_path / "users", creds, legacy)
    profile = store.load("jdoe")
    assert profile.eln.enabled and profile.eln.account().settings["base_url"] == "https://e.org"
    assert profile.eln.default_template == "7" and profile.assistant.enabled
    assert creds.get("eln/lab/jdoe") == "OLD" and creds.get("assistant/default/jdoe") == "A"
    text = (tmp_path / "users" / "jdoe" / "profile.yaml").read_text(encoding="utf-8")
    assert "OLD" not in text and "api_key" not in text
    assert not legacy.exists() and legacy.with_name("eln-settings.json.migrated").exists()
    assert store.load("someone_else").eln.enabled is False, "migrated once, into one profile"


def test_the_session_list_is_per_user(tmp_path):
    from i2as.session.store import SessionStore

    profiles = UserProfileStore(tmp_path / "users", legacy_eln_path=tmp_path / "none.json")
    alice = SessionStore(tmp_path / "root", registry=profiles.session_registry("alice"))
    folder = alice.resolve_active("alice")
    assert profiles.load("alice").sessions["active"] == str(folder)
    assert profiles.load("bob").sessions == {}
    machine = json.loads((tmp_path / "root" / "sessions.json").read_text(encoding="utf-8"))
    assert machine["active"] == str(folder), "the command-line client still finds the last session"


def test_user_ids_that_are_not_plain_names_are_refused(tmp_path):
    store = UserProfileStore(tmp_path / "users", legacy_eln_path=tmp_path / "none.json")
    with pytest.raises(ValueError):
        store.path("../escape")
    assert store.load("../escape") == UserProfile()


# ── The outbox journal ────────────────────────────────────────────────────


def test_the_outbox_is_idempotent_backs_off_and_waits_for_a_person(tmp_path):
    outbox = Outbox(tmp_path / "outbox.jsonl", retry_base_s=10, retry_max_s=40)
    job = OutboxJob(job_id="publish:P-1", kind="publish", experiment_id="001_x")
    assert outbox.enqueue(job) is True and outbox.enqueue(job) is False
    [due] = outbox.due()
    first = outbox.failed(due, "down")
    assert first.attempts == 1 and outbox.due() == []
    second = outbox.failed(first, "down")
    third = outbox.failed(second, "down")
    assert third.attempts == 3 and third.state == STATE_PENDING
    stuck = outbox.attention(third, ATTENTION_AUTH, "rejected")
    assert outbox.get("publish:P-1").state == STATE_NEEDS_ATTENTION
    assert outbox.retry(reason="refused") == 0 and outbox.retry(reason=ATTENTION_AUTH) == 1
    assert outbox.get("publish:P-1").state == STATE_PENDING and outbox.due()
    with (tmp_path / "outbox.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{corrupt\n")
    assert list(outbox.jobs()) == ["publish:P-1"], "a corrupt line is skipped"
    assert stuck.attention == ATTENTION_AUTH


# ── The whole service, against the simulated notebook ─────────────────────


@pytest.fixture
def bench(tmp_path, qtbot):
    """A manager with an open experiment, two finished runs (one analysed), a user and a service."""
    from i2as.core.orchestrator import Orchestrator
    from i2as.core.station import build_station
    from i2as.session.manager import ExperimentManager
    from i2as.session.store import ExperimentStore, UserRoster

    store = ExperimentStore(tmp_path / "session")
    roster = UserRoster(tmp_path / "users.json")
    roster.add(User(user_id="jdoe", name="J. Doe"))
    orchestrator = Orchestrator(build_station("i2as/configs/sim_cryostat"), tick_interval_ms=10)
    manager = ExperimentManager(store=store, roster=roster, orchestrator=orchestrator, config_name="sim_cryostat")
    experiment = manager.start_experiment("Hall bar A", "jdoe", {"sample_name": "A3"})
    for run_id in ("run-0001", "run-0002"):
        started = {"run_id": run_id, "procedure": "FieldSweep", "kind": "run", "params": {"field_T": 1.5}, "started_utc": "2026-09-27T10:00:00+00:00"}
        orchestrator.run_started.emit(started)
        orchestrator.run_finished.emit(dict(started, finished_utc="2026-09-27T11:00:00+00:00", status="done", reason=""))
    folder = store.bundle_dir(experiment.experiment_id, "run-0001", "20260927T110000Z-generic-0001")
    folder.mkdir(parents=True)
    (folder / "overview.png").write_bytes(b"PNG")
    seal_bundle(
        folder,
        {"status": "ok", "summary": ["R rises with B."], "results": [{"name": "R0", "value": 101.5, "unit": "Ω"}], "figures": [{"file": "overview.png", "caption": "Overview"}]},
        bundle_id=folder.name,
        experiment_id=experiment.experiment_id,
        run_ids=("run-0001",),
        producer=Producer(kind="recipe", name="generic_sweep", digest="a" * 64),
    )
    assert manager.select_bundle("run-0001", folder.name)

    creds = CredentialStore(fallback_path=tmp_path / "credentials.json", environ={}, detect=False)
    profiles = UserProfileStore(tmp_path / "profiles", creds, tmp_path / "no-legacy.json")
    state_file = tmp_path / "sim-notebook.json"
    profiles.save(
        "jdoe",
        UserProfile(
            eln=ElnUserSettings(
                enabled=True,
                accounts=(ElnAccount(account_id="lab", connector="sim", settings={"state_file": str(state_file)}),),
                default_account="lab",
            )
        ),
    )
    creds.set(credential_key("eln", "lab", "jdoe"), "sim-key")
    blocks = tmp_path / "blocks"
    service = ElnService(
        manager,
        profiles,
        creds,
        publishing=lambda: PublishingSettings(retry_base_s=1.0, retry_max_s=2.0),
        catalog=BlockCatalog(blocks),
        runner_factory=InProcessBlockRunner,
        synchronous=True,
    )

    class Bench:
        pass

    b = Bench()
    b.manager, b.store, b.service, b.profiles, b.creds = manager, store, service, profiles, creds
    b.experiment_id, b.state_file, b.blocks, b.bundle_id = experiment.experiment_id, state_file, blocks, folder.name
    b.notebook = lambda: json.loads(state_file.read_text(encoding="utf-8"))
    yield b
    service.stop()


def _page(b):
    entry = b.manager.current_experiment().eln.entry
    return b.notebook()["entries"][entry.entry_id]


def test_linking_creates_the_page_pins_the_blocks_and_links_the_sample(bench):
    binding = bench.service.link_experiment("lab", items=[LinkedItem(item_id="1", role="sample", title="S-001")])
    assert binding.connector.block_id == "sim" and binding.connector.digest
    create = Outbox(bench.store.eln_dir(bench.experiment_id) / "outbox.jsonl").get(f"create_entry:{bench.experiment_id}")
    assert create is not None and create.state == STATE_DONE, "the page was created through the outbox"
    eln_dir = bench.store.eln_dir(bench.experiment_id)
    assert (eln_dir / "profile.yaml").is_file() and (eln_dir / "renderer.py").is_file(), "the blocks are pinned by copy"
    record = bench.manager.current_experiment()
    assert record.eln.entry is not None and not record.eln.create_pending, "the create job drained"
    page = _page(bench)
    assert page["title"] == "Hall bar A" and page["links"] == ["1"]


def test_linking_an_existing_page_creates_nothing(bench):
    bench.notebook  # nothing created yet
    entry = ElnEntryRef(backend="sim", entry_id="555", url="sim://experiments/555")
    binding = bench.service.link_experiment("lab", entry=entry)
    assert binding.entry.entry_id == "555" and not binding.create_pending
    assert not bench.state_file.exists() or bench.notebook()["entries"] == {}


def test_publishing_needs_one_approval_then_appends_a_section_per_publish(bench):
    bench.service.link_experiment("lab")
    with pytest.raises(PublishError, match="approved"):
        bench.service.publish()
    bench.manager.approve_eln_publishing("jdoe")
    publish_id = bench.service.publish()

    page = _page(bench)
    assert publish_id in page["body"] and page["body"].count("<h2>I2AS") == 1
    assert "run-0001" in page["body"] and "run-0002" in page["body"] and "R rises with B." in page["body"]
    assert page["uploads"] == [{"name": "run-0001_overview.png", "bytes": 3, "caption": "Overview"}]
    for run_id, bundle in (("run-0001", bench.bundle_id), ("run-0002", "")):
        run = bench.manager.current_experiment().find_run(run_id)
        assert run.published and run.eln_publish["publish_id"] == publish_id and run.eln_publish["bundle_id"] == bundle

    with pytest.raises(PublishError, match="nothing new"):
        bench.service.publish()
    again = bench.service.publish(["run-0001"])
    page = _page(bench)
    assert page["body"].count("<h2>I2AS") == 2, "a re-publish appends; the first section stays"
    assert page["body"].index(publish_id) < page["body"].index(again)
    assert len(page["uploads"]) == 1, "the same figure is not uploaded twice"
    assert bench.service.status()["state"] == "synced"


def test_profile_fields_are_overwritten_with_the_latest_values(bench):
    profile = bench.store.eln_dir(bench.experiment_id)
    bench.service.link_experiment("lab")
    (profile / "profile.yaml").write_text('renderer: default\nfields:\n  "R0 (Ω)": "result:R0"\n  Sample: "sample:sample_name"\n', encoding="utf-8")
    bench.manager.approve_eln_publishing("jdoe")
    bench.service.publish()
    assert _page(bench)["fields"] == {"R0 (Ω)": "101.5", "Sample": "A3"}


def test_reading_fields_back_proposes_values_and_the_manager_applies_them(bench):
    bench.service.link_experiment("lab", items=[LinkedItem(item_id="1", role="sample")])
    found = {}
    bench.service.read_fields(found.update)
    assert found["sample_id"]["value"] == "S-001" and found["thickness"]["value"] == pytest.approx(4.2)
    assert found["thickness"]["unit"] == "nm" and found["sample_id"]["current"] is None
    assert bench.manager.current_experiment().sample_info == {"sample_name": "A3"}, "reading applies nothing"
    bench.manager.apply_eln_fields({"thickness": found["thickness"]["value"]}, found)
    assert bench.manager.current_experiment().sample_info["thickness"] == pytest.approx(4.2)


def test_an_offline_notebook_is_retried_until_it_answers(bench):
    bench.service.link_experiment("lab")
    bench.manager.approve_eln_publishing("jdoe")
    bench.profiles.update("jdoe", lambda p: replace(p, eln=replace(p.eln, accounts=(replace(p.eln.accounts[0], settings={**p.eln.accounts[0].settings, "fail": "transient"}),))))
    bench.service.reload()
    bench.service.publish()
    status = bench.service.status()
    assert status["state"] == "offline" and status["pending"] == 1
    bench.profiles.update("jdoe", lambda p: replace(p, eln=replace(p.eln, accounts=(replace(p.eln.accounts[0], settings={"state_file": str(bench.state_file)}),))))
    outbox = Outbox(bench.store.eln_dir(bench.experiment_id) / "outbox.jsonl")
    for job in outbox.jobs().values():
        outbox.record(replace(job, next_due_utc=""))
    bench.service.reload()
    assert bench.service.status()["state"] == "synced"
    assert "run-0001" in _page(bench)["body"]


def test_a_rejected_key_waits_for_a_new_one_and_then_publishes(bench):
    bench.service.link_experiment("lab")
    bench.manager.approve_eln_publishing("jdoe")
    bench.creds.set(credential_key("eln", "lab", "jdoe"), "")
    bench.service.reload()
    bench.service.publish()
    [attention] = bench.service.status()["attention"]
    assert attention["reason"] == ATTENTION_AUTH
    bench.creds.set(credential_key("eln", "lab", "jdoe"), "new-key")
    bench.service.reload()
    assert bench.service.status()["attention"] == [] and bench.service.status()["state"] == "synced"


def test_a_changed_connector_is_not_used_until_a_person_confirms_it(bench):
    bench.service.link_experiment("lab")
    bench.manager.approve_eln_publishing("jdoe")
    user_connectors = bench.blocks / "connectors"
    user_connectors.mkdir(parents=True)
    shutil.copyfile(shipped_dir() / "connectors" / "sim.py", user_connectors / "sim.py")
    with (user_connectors / "sim.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# edited by the user\n")
    bench.service.reload()
    bench.service.publish()
    [attention] = bench.service.status()["attention"]
    assert attention["reason"] == ATTENTION_CHANGED
    assert bench.service.confirm_connector() is True
    assert bench.service.status()["attention"] == []
    assert "run-0001" in _page(bench)["body"]


def test_a_publish_resumed_after_a_crash_never_appends_twice(bench, tmp_path):
    """The section reached the page but the job did not record it: the retry checks the page."""
    bench.service.link_experiment("lab")
    entry = ElnEntryRef.from_dict(bench.manager.current_experiment().eln.entry.to_dict())
    runner = InProcessBlockRunner("connector", shipped_dir() / "connectors" / "sim.py", {"state_file": str(bench.state_file)}, "k")
    runner.call("append_section", {"entry": entry.to_dict(), "publish_id": "P-crash", "html": "<div>P-crash</div>"}, 5)
    outbox = Outbox(tmp_path / "o.jsonl")
    job = OutboxJob(job_id="publish:P-crash", kind="publish", experiment_id=bench.experiment_id, payload={"publish_id": "P-crash", "html": "<div>P-crash</div>", "run_bundles": {}})
    outbox.enqueue(job)
    executor = JobExecutor(lambda _u, _a: (runner, ""), lambda _e: entry, lambda _e: Ledger(tmp_path / "ledger.json"))
    outcome = executor.run(outbox, outbox.get("publish:P-crash"))
    assert outcome.job.state == STATE_DONE and outcome.published["publish_id"] == "P-crash"
    assert _page(bench)["body"].count("P-crash") == 1


def test_a_renderer_that_fails_leaves_nothing_queued_and_says_why(bench):
    bench.service.link_experiment("lab")
    bench.manager.approve_eln_publishing("jdoe")
    (bench.store.eln_dir(bench.experiment_id) / "renderer.py").write_text("NAME='x'\ndef render(context):\n    raise RuntimeError('layout bug')\n", encoding="utf-8")
    failures = []
    bench.service.publish_failed.connect(failures.append)
    bench.service.publish()
    assert failures and "layout bug" in failures[0]["reason"]
    assert bench.service.status()["pending"] == 0
    assert not bench.manager.current_experiment().find_run("run-0001").published


def test_nothing_is_sent_for_a_user_with_publishing_off(bench):
    bench.profiles.update("jdoe", lambda p: replace(p, eln=replace(p.eln, enabled=False)))
    bench.service.link_experiment("lab")
    bench.manager.approve_eln_publishing("jdoe")
    with pytest.raises(PublishError, match="switched off"):
        bench.service.publish()
    assert bench.service.status()["state"] == "disabled"


def test_the_analysis_block_still_migrates_after_the_profile_moved_the_old_file(tmp_path):
    """Whichever migration runs first, neither loses the other's half of the old file."""
    from i2as.session.app_config import legacy_analysis_block

    legacy = tmp_path / "eln-settings.json"
    legacy.write_text(json.dumps({"base_url": "https://e.org", "analysis": {"enabled": True, "timeout_s": 30}}), encoding="utf-8")
    UserProfileStore(tmp_path / "users", CredentialStore(fallback_path=tmp_path / "c.json", environ={}, detect=False), legacy).load("jdoe")
    assert not legacy.exists()
    assert legacy_analysis_block(legacy) == {"enabled": True, "timeout_s": 30}
