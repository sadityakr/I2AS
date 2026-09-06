"""`troubleshoot session` — the read-only report over one experiment folder.

The module under test deliberately does NOT import the session layer (contract
C12 forbids it), so it parses ``experiment.json`` from its documented shape.
These tests close that loop from the other side: the fixtures write their
records with the session layer's own ``ExperimentStore`` and models — which
the tests may import, since the contracts bind ``i2as.*`` and not
``tests.*`` — so any drift between the writer and this reader fails here
rather than in front of an operator.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from i2as.core import events as ev
from i2as.core.plan import EnvelopeBound, ExperimentEnvelope, params_digest
from i2as.session.models import (
    ExperimentRecord,
    RunRecord,
    envelope_to_dict,
)
from i2as.session.store import ExperimentStore
from i2as.troubleshoot import cli, session_report

# The autouse transcript-isolation fixture lives in the CLI test module; import
# it so `session` invocations here never append to the real log directory.
from tests.test_troubleshoot_cli import isolated_transcript  # noqa: F401


def _session_root(root: Path, user_id: str = "jdoe", session_id: str = "20260901_cooldown") -> Path:
    """Return (and create) one session folder under a measurement root."""
    folder = root / "sessions" / user_id / session_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_experiment(
    session_folder: Path,
    experiment_id: str = "20260902_hall_bar_a3",
    *,
    runs: list[RunRecord] | None = None,
    envelope: dict | None = None,
) -> Path:
    """Write a real experiment record through the session layer's own store."""
    store = ExperimentStore(session_folder)
    record = ExperimentRecord(
        experiment_id=experiment_id,
        title="Hall bar A3 — SOT switching vs T",
        user_id="jdoe",
        sample_info={"sample_name": "A3", "sample_id": "W12-A3"},
        config_name="sim_cryostat",
        created_utc="2026-09-02T08:00:00+00:00",
        envelope=envelope or {},
        runs=runs or [],
    )
    store.save(record)
    return session_folder / experiment_id


def dataclasses_replace_actor(run: RunRecord, actor: ev.Actor) -> RunRecord:
    """Return *run* with a different starting actor (the records are frozen-ish)."""
    return dataclasses.replace(run, actor=actor)


def _make_run(
    run_id: str,
    procedure: str,
    status: str,
    started: str,
    finished: str,
    data_file: str,
    *,
    kind: str = "run",
    reason: str = "",
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        procedure=procedure,
        kind=kind,
        data_file=data_file,
        started_utc=started,
        finished_utc=finished,
        status=status,
        reason=reason,
    )


@pytest.fixture()
def experiment_dir(tmp_path: Path) -> Path:
    """A two-run experiment folder with one existing and one missing data file."""
    folder = _write_experiment(
        _session_root(tmp_path),
        runs=[
            _make_run(
                "r1",
                "FieldSweep",
                "done",
                "2026-09-02T08:10:00+00:00",
                "2026-09-02T08:40:00+00:00",
                "data/run_0001.h5",
            ),
            _make_run(
                "r2",
                "TimeSeries",
                "failed",
                "2026-09-02T09:00:00+00:00",
                "2026-09-02T09:05:30+00:00",
                "data/run_0002.h5",
                kind="probe",
                reason="magnet quench",
            ),
        ],
    )
    data = folder / "data"
    data.mkdir()
    (data / "run_0001.h5").write_bytes(b"")
    return folder


def _json_out(capsys: pytest.CaptureFixture) -> dict:
    return json.loads(capsys.readouterr().out)


# ── Text report ───────────────────────────────────────────────────────────────


def test_report_lists_runs_in_order_with_outcomes(experiment_dir, capsys) -> None:
    assert cli.main(["session", str(experiment_dir)]) == 0
    out = capsys.readouterr().out

    assert "Experiment: 20260902_hall_bar_a3  (open)" in out
    assert "Hall bar A3 — SOT switching vs T" in out
    assert "sim_cryostat" in out
    assert "sample_name=A3" in out
    # Runs appear in stored order, with kind, procedure, outcome and duration.
    first = out.index("[run] FieldSweep")
    second = out.index("[probe] TimeSeries")
    assert first < second
    assert "done" in out and "(1800s)" in out
    assert "failed" in out and "(330s)" in out
    assert "reason: magnet quench" in out
    assert "=> 1 done, 1 failed" in out


def test_missing_data_file_is_flagged_and_present_one_is_not(experiment_dir, capsys) -> None:
    assert cli.main(["session", str(experiment_dir)]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if "data:" in line]
    assert len(lines) == 2
    assert lines[0].strip() == "data: data/run_0001.h5"
    assert lines[1].strip() == "data: data/run_0002.h5  [MISSING]"


def test_running_run_reports_no_duration(tmp_path, capsys) -> None:
    folder = _write_experiment(
        _session_root(tmp_path),
        runs=[_make_run("r1", "FieldSweep", "running", "2026-09-02T08:10:00+00:00", "", "")],
    )
    assert cli.main(["session", str(folder)]) == 0
    out = capsys.readouterr().out
    assert "still running" in out
    assert "data: (none recorded)" in out


def test_experiment_without_runs_says_so(tmp_path, capsys) -> None:
    folder = _write_experiment(_session_root(tmp_path))
    assert cli.main(["session", str(folder)]) == 0
    out = capsys.readouterr().out
    assert "Runs: none recorded." in out
    assert "Incident reports: none in this folder." in out


# ── Envelope ──────────────────────────────────────────────────────────────────


def test_stored_envelope_is_reported(tmp_path, capsys) -> None:
    envelope = envelope_to_dict(
        ExperimentEnvelope(
            bounds={
                "magnet": EnvelopeBound(min_value=-2.0, max_value=2.0, state_key="field_T"),
                "sample_temperature": EnvelopeBound(max_value=300.0),
            }
        )
    )
    folder = _write_experiment(_session_root(tmp_path), envelope=envelope)

    assert cli.main(["session", str(folder), "--json"]) == 0
    bounds = _json_out(capsys)["envelope"]
    assert [b["vi_name"] for b in bounds] == ["magnet", "sample_temperature"]
    assert bounds[0] == {
        "vi_name": "magnet",
        "min_value": -2.0,
        "max_value": 2.0,
        "state_key": "field_T",
    }
    assert bounds[1]["min_value"] is None

    assert cli.main(["session", str(folder)]) == 0
    out = capsys.readouterr().out
    assert "magnet: -2.0 .. 2.0  [field_T]" in out
    assert "sample_temperature: unbounded .. 300.0" in out


# ── Incident reports ──────────────────────────────────────────────────────────


def test_incident_reports_in_the_folder_are_listed(experiment_dir, capsys) -> None:
    incidents = experiment_dir / "incidents"
    incidents.mkdir()
    (incidents / "2026-09-02-field-not-ramping.md").write_text(
        "# Incident: field does not follow setpoint  (2026-09-02)\n\n## Symptom\n",
        encoding="utf-8",
    )

    assert cli.main(["session", str(experiment_dir), "--json"]) == 0
    (report,) = _json_out(capsys)["incidents"]
    assert report["path"] == "incidents/2026-09-02-field-not-ramping.md"
    assert report["title"] == "Incident: field does not follow setpoint  (2026-09-02)"
    assert report["size_bytes"] > 0

    assert cli.main(["session", str(experiment_dir)]) == 0
    out = capsys.readouterr().out
    assert "Incident reports (1):" in out
    assert "field does not follow setpoint" in out


def test_unrelated_markdown_is_not_an_incident_report(experiment_dir, capsys) -> None:
    (experiment_dir / "notes.md").write_text("# Cooldown notes\n", encoding="utf-8")
    assert cli.main(["session", str(experiment_dir), "--json"]) == 0
    assert _json_out(capsys)["incidents"] == []


# ── JSON payload ──────────────────────────────────────────────────────────────


def test_json_payload_carries_every_run_field(experiment_dir, capsys) -> None:
    assert cli.main(["session", str(experiment_dir), "--json"]) == 0
    payload = _json_out(capsys)

    assert payload["available"] is True
    assert payload["experiment_dir"] == str(experiment_dir)
    assert payload["experiment_id"] == "20260902_hall_bar_a3"
    assert payload["user_id"] == "jdoe"
    assert payload["status"] == "open"
    assert payload["config_name"] == "sim_cryostat"
    assert payload["run_counts"] == {"done": 1, "failed": 1}

    first, second = payload["runs"]
    assert first == {
        "index": 1,
        "run_id": "r1",
        "kind": "run",
        "procedure": "FieldSweep",
        "status": "done",
        "reason": "",
        "started_utc": "2026-09-02T08:10:00+00:00",
        "finished_utc": "2026-09-02T08:40:00+00:00",
        "duration_s": 1800.0,
        "data_file": "data/run_0001.h5",
        "data_file_path": str(experiment_dir / "data" / "run_0001.h5"),
        "data_file_exists": True,
        "actor": "operator",
        "params_digest": "",
    }
    assert second["kind"] == "probe"
    assert second["reason"] == "magnet quench"
    assert second["data_file_exists"] is False


# ── Who started each run ──────────────────────────────────────────────────────
#
# The after-the-fact half of the accountability trail: `troubleshoot status`
# says who last got the running engine to act, and this says who started each
# run that already happened. Both read records, never the live app.


def test_the_report_names_the_agent_that_started_a_run(tmp_path, capsys) -> None:
    agent = ev.Actor(kind=ev.ActorKind.AGENT, id="runner-7", role="session")
    run = _make_run(
        "r1", "FieldSweep", "done",
        "2026-09-02T08:10:00+00:00", "2026-09-02T08:40:00+00:00", "data/run_0001.h5",
    )
    folder = _write_experiment(
        _session_root(tmp_path), runs=[dataclasses_replace_actor(run, agent)]
    )

    assert cli.main(["session", str(folder)]) == 0
    assert "started by: agent runner-7 (role session)" in capsys.readouterr().out


def test_the_report_names_the_operator_without_repeating_the_sentinel(
    experiment_dir, capsys
) -> None:
    """The operator sentinel is Actor("operator", "operator", "operator")."""
    assert cli.main(["session", str(experiment_dir)]) == 0
    out = capsys.readouterr().out
    assert "started by: operator" in out
    assert "operator operator" not in out


def test_a_run_written_before_actors_reads_as_unknown_not_as_the_physicist(
    tmp_path, capsys
) -> None:
    """"Old file" must never render as "the physicist did it"."""
    folder = _write_experiment(_session_root(tmp_path), runs=[])
    path = folder / "experiment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runs"] = [{"run_id": "old", "procedure": "FieldSweep", "status": "done"}]
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert cli.main(["session", str(folder), "--json"]) == 0
    assert _json_out(capsys)["runs"][0]["actor"] == "unknown (legacy record)"


def test_the_json_payload_carries_the_actor_and_the_params_digest(
    tmp_path, capsys
) -> None:
    agent = ev.Actor(kind=ev.ActorKind.AGENT, id="runner-7", role="session")
    digest = params_digest({"start_T": 0.0, "stop_T": 1.0})
    run = _make_run(
        "r1", "FieldSweep", "done",
        "2026-09-02T08:10:00+00:00", "2026-09-02T08:40:00+00:00", "data/run_0001.h5",
    )
    run = dataclasses_replace_actor(run, agent)
    run = dataclasses.replace(run, params_digest=digest)
    folder = _write_experiment(_session_root(tmp_path), runs=[run])

    assert cli.main(["session", str(folder), "--json"]) == 0
    entry = _json_out(capsys)["runs"][0]
    assert entry["actor"] == "agent runner-7 (role session)"
    assert entry["params_digest"] == digest


# ── Resolution and exit codes ─────────────────────────────────────────────────


def test_defaults_to_most_recently_modified_experiment(
    tmp_path, monkeypatch, capsys
) -> None:
    session_folder = _session_root(tmp_path)
    older = _write_experiment(session_folder, "20260901_older")
    newer = _write_experiment(session_folder, "20260902_newer")
    os.utime(older / "experiment.json", (1_700_000_000, 1_700_000_000))
    os.utime(newer / "experiment.json", (1_800_000_000, 1_800_000_000))
    monkeypatch.setenv("I2AS_MEASUREMENT_ROOT", str(tmp_path))

    assert cli.main(["session", "--json"]) == 0
    assert _json_out(capsys)["experiment_id"] == "20260902_newer"


def test_no_experiment_found_exits_one(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("I2AS_MEASUREMENT_ROOT", str(tmp_path / "empty"))
    assert cli.main(["session", "--json"]) == 1
    payload = _json_out(capsys)
    assert payload["available"] is False
    assert "No experiment found" in payload["reason"]


def test_no_experiment_found_human_output_names_where_it_looked(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("I2AS_MEASUREMENT_ROOT", str(tmp_path))
    assert cli.main(["session"]) == 1
    out = capsys.readouterr().out
    assert str(tmp_path) in out
    assert "experiment.json" in out


def test_unconfigured_measurement_root_exits_one(tmp_path, monkeypatch, capsys) -> None:
    """No measurement root configured is reported, not raised as a traceback."""
    monkeypatch.delenv("I2AS_MEASUREMENT_ROOT", raising=False)
    monkeypatch.setattr(
        cli, "measurement_root", lambda: (_ for _ in ()).throw(RuntimeError("no root here"))
    )
    assert cli.main(["session", "--json"]) == 1
    assert _json_out(capsys)["reason"] == "no root here"


def test_folder_without_a_record_exits_one(tmp_path, capsys) -> None:
    assert cli.main(["session", str(tmp_path), "--json"]) == 1
    payload = _json_out(capsys)
    assert payload["available"] is False
    assert "No experiment record" in payload["reason"]


def test_corrupt_record_exits_one_without_raising(tmp_path, capsys) -> None:
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "experiment.json").write_text("{not json", encoding="utf-8")
    assert cli.main(["session", str(folder), "--json"]) == 1
    assert "not valid JSON" in _json_out(capsys)["reason"]


# ── Module-level helpers ──────────────────────────────────────────────────────


def test_find_experiment_dirs_only_returns_folders_with_a_record(tmp_path) -> None:
    session_folder = _session_root(tmp_path)
    real = _write_experiment(session_folder)
    (session_folder / "not_an_experiment").mkdir()

    assert session_report.find_experiment_dirs(tmp_path) == [real]
    assert session_report.find_experiment_dirs(tmp_path / "nowhere") == []


def test_absolute_data_path_falls_back_to_a_basename_search(tmp_path) -> None:
    """A moved folder's absolute data_file still resolves under data/."""
    folder = _write_experiment(
        _session_root(tmp_path),
        runs=[
            _make_run(
                "r1",
                "FieldSweep",
                "done",
                "2026-09-02T08:10:00+00:00",
                "2026-09-02T08:20:00+00:00",
                "/gone/elsewhere/run_0001.h5",
            )
        ],
    )
    nested = folder / "data" / "heating_runs"
    nested.mkdir(parents=True)
    (nested / "run_0001.h5").write_bytes(b"")

    (run,) = session_report.build_report(folder)["runs"]
    assert run["data_file_path"] == str(nested / "run_0001.h5")
    assert run["data_file_exists"] is True
