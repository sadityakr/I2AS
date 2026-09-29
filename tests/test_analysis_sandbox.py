"""The analysis sandbox (i2as/session/analysis_sandbox.py) and its settings.

A backend is two pure decisions — which spec the worker is handed, and how it
is started — so both are tested here without Qt, a subprocess or a container
engine (none is installed on most test machines). ``test_analysis_runner.py``
starts a real stand-in worker through ``SubprocessSandbox``; the one thing
these tests cannot prove is that a real engine accepts the command line, and
the doc says so.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from i2as.analysis.report import AnalysisSpec
from i2as.session import analysis_sandbox
from i2as.session.analysis_sandbox import (
    BACKEND_CONTAINER,
    CONTAINER_NAME_PREFIX,
    DOCKERFILE_PATH,
    FORCED_ENV,
    ContainerSandbox,
    EngineStatus,
    SandboxError,
    SubprocessSandbox,
    build_image_command,
    build_sandbox,
    check_engine,
    stage_image_context,
)
from i2as.session.app_config import (
    AnalysisSettings,
    AppConfig,
    AppConfigStore,
    SandboxSettings,
    load_app_config,
)


def _spec(tmp_path: Path, **overrides: object) -> AnalysisSpec:
    data = tmp_path / "data" / "run-1.h5"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_bytes(b"HDF")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    fields = {"run_id": "run-1", "data_path": str(data), "output_dir": str(out)}
    fields.update(overrides)
    return AnalysisSpec(**fields)  # type: ignore[arg-type]


def _sandbox(**settings: object) -> ContainerSandbox:
    return ContainerSandbox(
        SandboxSettings(**settings),  # type: ignore[arg-type]
        engine_path="/usr/bin/docker",
        name="i2as-analysis-test",
        host_user="1000:1000",
    )


# ── prepare(): host paths become container paths ─────────────────────────


def test_prepare_maps_every_path_into_the_container_and_records_three_mounts(tmp_path):
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    spec = _spec(tmp_path, recipe_dirs=(str(recipes), str(tmp_path / "missing")))
    sandbox = _sandbox()

    mapped = sandbox.prepare(spec)

    assert mapped.output_dir == "/work"
    assert mapped.data_path == "/input/run-1.h5"
    assert mapped.recipe_dirs == ("/recipes/0",), "a missing recipe folder is not mounted"
    assert sandbox.mounts == [
        (str((tmp_path / "out").resolve()), "/work", False),
        (str((tmp_path / "data" / "run-1.h5").resolve()), "/input/run-1.h5", True),
        (str(recipes.resolve()), "/recipes/0", True),
    ], "the run file and the recipes are read-only; only the output folder is writable"


def test_a_script_is_mapped_inside_its_output_folder(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    script = out / "script.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    mapped = _sandbox().prepare(_spec(tmp_path, script_path=str(script)))

    assert mapped.script_path == "/work/script.py"


def test_a_script_outside_its_output_folder_is_refused(tmp_path):
    elsewhere = tmp_path / "elsewhere.py"
    elsewhere.write_text("", encoding="utf-8")

    with pytest.raises(SandboxError, match="not inside its output folder"):
        _sandbox().prepare(_spec(tmp_path, script_path=str(elsewhere)))


def test_a_spec_with_no_output_folder_is_refused(tmp_path):
    with pytest.raises(SandboxError, match="output folder"):
        _sandbox().prepare(AnalysisSpec(run_id="r", data_path=""))


def test_a_path_with_a_comma_cannot_be_mounted(tmp_path):
    odd = tmp_path / "a,b"
    odd.mkdir()
    with pytest.raises(SandboxError, match="comma"):
        _sandbox().prepare(AnalysisSpec(run_id="r", data_path="", output_dir=str(odd)))


# ── launch(): the run command ─────────────────────────────────────────────


def test_launch_plans_an_isolated_container_run(tmp_path):
    sandbox = _sandbox(memory="2g", cpus=1.5, pids_limit=64, image="lab/analysis:1")
    sandbox.prepare(_spec(tmp_path))

    plan = sandbox.launch(tmp_path / "out" / "spec.json")
    arguments = list(plan.arguments)

    assert plan.program == "/usr/bin/docker" and plan.backend == BACKEND_CONTAINER
    assert arguments[:2] == ["run", "--rm"]
    for flag, value in [
        ("--name", "i2as-analysis-test"),
        ("--pull", "never"),
        ("--network", "none"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--memory", "2g"),
        ("--cpus", "1.5"),
        ("--pids-limit", "64"),
        ("--user", "1000:1000"),
        ("--workdir", "/work"),
    ]:
        assert arguments[arguments.index(flag) + 1] == value, flag
    assert "--read-only" in arguments
    mounts = [arguments[i + 1] for i, a in enumerate(arguments) if a == "--mount"]
    assert mounts[0].endswith(",target=/work"), "the output folder is writable"
    assert mounts[1].endswith(",target=/input/run-1.h5,readonly")
    env = [arguments[i + 1] for i, a in enumerate(arguments) if a == "--env"]
    assert env == [f"{k}={v}" for k, v in FORCED_ENV.items()], "no host variable is passed"
    image_at = arguments.index("lab/analysis:1")
    assert arguments[image_at + 1 :] == [
        "python", "-m", "i2as.analysis", "run", "--spec", "/work/spec.json",
    ]
    assert plan.stop_command == ("/usr/bin/docker", "kill", "i2as-analysis-test")
    # The engine CLIENT gets this environment with its own folder first on
    # PATH (for its credential helper); the container gets none of it — only
    # the --env pairs above.
    assert plan.environment is not None
    assert plan.environment["PATH"].split(os.pathsep)[0] == str(Path("/usr/bin/docker").parent)


def test_check_engine_does_not_trust_an_empty_server_version(monkeypatch):
    """Docker Desktop failing to start: `docker info` exits 0 with no version."""
    monkeypatch.setattr(analysis_sandbox, "resolve_engine", lambda engine: "/bin/docker")
    monkeypatch.setattr(analysis_sandbox, "_probe", lambda command: (True, ""))

    status = check_engine(SandboxSettings())

    assert status.engine_found and not status.engine_running and not status.ready
    assert "not running" in status.detail


def test_an_engine_off_path_is_found_where_its_installer_puts_it(monkeypatch, tmp_path):
    """Installed after this process started: not on PATH, but in its install folder."""
    installed = tmp_path / "docker.exe"
    installed.write_text("", encoding="utf-8")
    monkeypatch.setattr(analysis_sandbox.shutil, "which", lambda name: None)
    monkeypatch.setattr(analysis_sandbox, "_known_engine_locations", lambda engine: [installed])

    assert analysis_sandbox.resolve_engine("docker") == str(installed)
    assert analysis_sandbox.resolve_engine("podman-missing") == str(installed)
    monkeypatch.setattr(analysis_sandbox, "_known_engine_locations", lambda engine: [])
    assert analysis_sandbox.resolve_engine("docker") is None


def test_the_engine_environment_puts_the_engine_folder_first(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "C:/elsewhere")
    engine = tmp_path / "bin" / "docker.exe"

    environment = analysis_sandbox.engine_environment(str(engine))

    assert environment["PATH"].split(os.pathsep) == [str(engine.parent), "C:/elsewhere"]
    assert analysis_sandbox.engine_environment(str(engine))["PATH"].count(str(engine.parent)) == 1


def test_zero_caps_are_left_off_the_command_line(tmp_path):
    sandbox = _sandbox(memory="", cpus=0.0, pids_limit=0)
    sandbox.prepare(_spec(tmp_path))

    arguments = sandbox.launch(tmp_path / "out" / "spec.json").arguments

    assert "--memory" not in arguments and "--cpus" not in arguments
    assert "--pids-limit" not in arguments


def test_launch_before_prepare_is_refused(tmp_path):
    with pytest.raises(SandboxError, match="before it was prepared"):
        _sandbox().launch(tmp_path / "spec.json")


def test_a_missing_engine_is_a_sandbox_error_naming_the_fix(tmp_path):
    sandbox = ContainerSandbox(SandboxSettings(engine="no-such-engine-xyz"))
    sandbox.prepare(_spec(tmp_path))

    with pytest.raises(SandboxError, match="not installed"):
        sandbox.launch(tmp_path / "out" / "spec.json")


def test_every_analysis_gets_a_fresh_container_name():
    first = build_sandbox(SandboxSettings())
    second = build_sandbox(SandboxSettings())

    assert isinstance(first, ContainerSandbox)
    assert first.name.startswith(CONTAINER_NAME_PREFIX) and first.name != second.name


def test_the_subprocess_sandbox_runs_this_interpreter_unchanged(tmp_path):
    spec = _spec(tmp_path)
    sandbox = SubprocessSandbox("/usr/bin/python-x")

    plan = sandbox.launch(tmp_path / "spec.json")

    assert sandbox.prepare(spec) is spec
    assert plan.program == "/usr/bin/python-x" and plan.stop_command == ()
    assert plan.arguments == ("-m", "i2as.analysis", "run", "--spec", str(tmp_path / "spec.json"))


# ── The engine check ─────────────────────────────────────────────────────


def test_check_engine_reports_a_missing_engine():
    status = check_engine(SandboxSettings(engine="no-such-engine-xyz"))

    assert not status.ready and not status.engine_found
    assert "Docker Desktop" in status.detail


def test_check_engine_walks_daemon_then_image(monkeypatch):
    monkeypatch.setattr(analysis_sandbox, "resolve_engine", lambda engine: "/bin/docker")
    answers = iter([(True, "27.1.1"), (False, "")])
    monkeypatch.setattr(analysis_sandbox, "_probe", lambda command: next(answers))

    status = check_engine(SandboxSettings())

    assert status.engine_running and not status.image_present and not status.ready
    assert "not built yet" in status.detail


def test_check_engine_is_ready_when_both_answer(monkeypatch):
    monkeypatch.setattr(analysis_sandbox, "resolve_engine", lambda engine: "/bin/docker")
    monkeypatch.setattr(analysis_sandbox, "_probe", lambda command: (True, "27.1.1"))

    assert check_engine(SandboxSettings()).ready
    assert EngineStatus(True, True, True).ready and not EngineStatus(True, False, True).ready


# ── The image ─────────────────────────────────────────────────────────────


def test_the_image_context_holds_only_the_analysis_stage(tmp_path):
    context = stage_image_context(tmp_path / "ctx")

    files = {p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file()}

    assert "Dockerfile" in files
    assert "i2as/core/data_reader.py" in files
    assert "i2as/analysis/__main__.py" in files
    core = {f for f in files if f.startswith("i2as/core/")}
    assert core == {
        "i2as/core/__init__.py",
        "i2as/core/data_reader.py",
        "i2as/core/events.py",
        "i2as/core/exceptions.py",
    }, "the Station, the Orchestrator and everything else are absent"
    assert not any(f.startswith(("i2as/session/", "i2as/drivers/", "i2as/gui/")) for f in files)
    assert not any("__pycache__" in f for f in files)


def test_the_image_context_is_enough_to_run_the_worker(tmp_path):
    """The staged package imports on its own — what the image will run."""
    context = stage_image_context(tmp_path / "ctx")
    env = {"PYTHONPATH": str(context), "PYTHONNOUSERSITE": "1"}
    if "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import i2as, i2as.analysis.__main__, i2as.analysis.recipes; print(i2as.__file__)",
        ],
        cwd=context,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()).is_relative_to(context), "imported the staged copy"


def test_the_dockerfile_ships_in_the_package():
    assert DOCKERFILE_PATH.is_file()
    assert "COPY i2as /opt/i2as/i2as" in DOCKERFILE_PATH.read_text(encoding="utf-8")


def test_build_image_command_tags_the_configured_image(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis_sandbox, "resolve_engine", lambda engine: "/bin/podman")

    command = build_image_command(SandboxSettings(image="lab/a:2"), tmp_path)

    assert command == ["/bin/podman", "build", "--tag", "lab/a:2", str(tmp_path)]


# ── Settings ─────────────────────────────────────────────────────────────


def test_sandbox_settings_round_trip_and_degrade_to_defaults():
    settings = SandboxSettings(engine="podman", image="lab/a:1", memory="8g", cpus=4.0, pids_limit=512)

    assert SandboxSettings.from_dict(settings.to_dict()) == settings
    assert SandboxSettings.from_dict("junk") == SandboxSettings()
    assert SandboxSettings.from_dict({"cpus": -3, "pids_limit": "x"}).cpus == 0.0
    legacy = {"backend": "venv", "python": "/v/bin/python", "stage_inputs": True}
    assert SandboxSettings.from_dict(legacy) == SandboxSettings(), "a legacy block is the defaults"


def test_the_general_settings_file_round_trips(tmp_path):
    path = tmp_path / "settings.json"
    config = AppConfig(analysis=AnalysisSettings(enabled=True, recipes={"FieldSweep": "mr"}))
    store = AppConfigStore(path)
    heard: list[AppConfig] = []
    store.subscribe(heard.append)

    store.save(config)

    assert load_app_config(path) == config
    assert heard == [config]
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {"connections", "analysis", "publishing"}


def test_a_malformed_settings_file_is_the_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")

    assert load_app_config(path) == AppConfig()


def test_the_legacy_analysis_block_migrates_without_its_sandbox(tmp_path, monkeypatch):
    eln = tmp_path / "eln-settings.json"
    eln.write_text(
        json.dumps(
            {
                "analysis": {
                    "enabled": True,
                    "timeout_s": 30,
                    "sandbox": {"backend": "venv", "python": "/x"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("I2AS_ELN_SETTINGS", str(eln))

    analysis = load_app_config(tmp_path / "settings.json").analysis

    assert analysis.enabled is True and analysis.timeout_s == 30.0
    assert analysis.sandbox == SandboxSettings()
