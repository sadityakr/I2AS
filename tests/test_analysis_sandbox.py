"""The analysis sandbox (i2as/session/analysis_sandbox.py) and its settings.

A backend is two pure decisions — which spec the worker is handed, and how it
is started — so both are tested here without Qt and without a subprocess;
``test_analysis_runner.py`` starts a real (stand-in) worker through them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from i2as.analysis.report import AnalysisSpec
from i2as.session.analysis_sandbox import (
    BACKEND_LOCAL,
    BACKEND_VENV,
    FORCED_ENV,
    AnalysisSandbox,
    SandboxError,
    VenvSandbox,
    build_sandbox,
    is_secret_name,
    scrubbed_environment,
)
from i2as.session.eln.settings import AnalysisSettings, ElnSettings, SandboxSettings


def _spec(tmp_path: Path) -> AnalysisSpec:
    data = tmp_path / "data" / "run-1.h5"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"HDF")
    return AnalysisSpec(run_id="run-1", data_path=str(data), output_dir=str(tmp_path / "out"))


def test_the_local_backend_is_the_worker_as_it_always_ran(tmp_path):
    """This interpreter, this environment, this working directory; the spec unchanged."""
    sandbox = build_sandbox(SandboxSettings(), default_python="/usr/bin/python-x")
    spec = _spec(tmp_path)

    plan = sandbox.launch(tmp_path / "spec.json")

    assert type(sandbox) is AnalysisSandbox and sandbox.backend == BACKEND_LOCAL
    assert sandbox.prepare(spec) is spec
    assert plan.program == "/usr/bin/python-x"
    assert plan.arguments == ("-m", "i2as.analysis", "run", "--spec", str(tmp_path / "spec.json"))
    assert plan.environment is None and plan.working_directory == ""


def test_the_venv_backend_stages_a_private_copy_of_the_run(tmp_path):
    """The worker is handed a copy; the recorded original is never its to touch."""
    spec = _spec(tmp_path)

    staged = VenvSandbox(sys.executable).prepare(spec)

    assert Path(staged.data_path) == tmp_path / "out" / "input" / "run-1.h5"
    assert Path(staged.data_path).read_bytes() == b"HDF"
    assert VenvSandbox(sys.executable, stage_inputs=False).prepare(spec) is spec


def test_a_run_file_that_cannot_be_staged_is_a_sandbox_error(tmp_path):
    spec = AnalysisSpec(
        run_id="r", data_path=str(tmp_path / "missing.h5"), output_dir=str(tmp_path / "out")
    )

    with pytest.raises(SandboxError, match="could not stage"):
        VenvSandbox(sys.executable).prepare(spec)


def test_the_venv_backend_plans_its_interpreter_environment_and_folder(tmp_path):
    """Its own interpreter, a scrubbed environment, the analysis folder as cwd."""
    environment = {
        "PATH": "/bin",
        "HOME": "/home/lab",
        "I2AS_ELAB_APIKEY": "elab-secret",
        "ANTHROPIC_API_KEY": "sk-1",
        "LAB_DATA": "/data",
        "EXTRA_TOKEN": "t",
    }
    sandbox = VenvSandbox(
        sys.executable,
        env_passthrough=("LAB_DATA", "EXTRA_TOKEN"),
        environment=environment,
    )

    plan = sandbox.launch(tmp_path / "out" / "spec.json")

    assert plan.program == sys.executable and plan.backend == BACKEND_VENV
    assert plan.working_directory == str(tmp_path / "out")
    assert plan.environment == {
        "PATH": "/bin",
        "HOME": "/home/lab",
        "LAB_DATA": "/data",
        **FORCED_ENV,
    }, "a credential is dropped even when the settings pass it through"


@pytest.mark.parametrize(("python", "message"), [("", "no interpreter"), ("/no/such/python", "does not exist")])
def test_a_venv_backend_without_a_usable_interpreter_refuses_to_plan(tmp_path, python, message):
    with pytest.raises(SandboxError, match=message):
        VenvSandbox(python).launch(tmp_path / "spec.json")


@pytest.mark.parametrize(
    ("name", "secret"),
    [
        ("PATH", False),
        ("HOME", False),
        ("I2AS_SPOOL", True),
        ("OPENAI_API_KEY", True),
        ("github_token", True),
        ("DB_PASSWORD", True),
    ],
)
def test_credential_names_are_recognised(name, secret):
    assert is_secret_name(name) is secret


def test_a_scrubbed_environment_only_holds_what_is_set():
    assert scrubbed_environment({}) == FORCED_ENV


def test_sandbox_settings_round_trip_and_degrade_to_local():
    """The settings file's block parses tolerantly; junk narrows to 'local'."""
    settings = SandboxSettings(
        backend="venv", python="/venvs/analysis/bin/python", env_passthrough=("LAB_DATA",)
    )

    assert SandboxSettings.from_dict(settings.to_dict()) == settings
    assert SandboxSettings.from_dict({"backend": "docker?"}).backend == "local"
    assert SandboxSettings.from_dict("junk") == SandboxSettings()
    parsed = ElnSettings.from_dict({"analysis": {"sandbox": settings.to_dict()}})
    assert parsed.analysis.sandbox == settings
    assert AnalysisSettings().to_dict()["sandbox"] == SandboxSettings().to_dict()
