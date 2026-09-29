"""The analysis sandbox — every analysis worker runs in a container.

The analysis worker runs recipe and script code in its own process, and
import contract C22 guarantees that process cannot IMPORT the Station. That
is not the same as saying the code it runs cannot reach the station: an
agent-written script runs as the same user as this application, and this
application accepts commands through files (the **Request spool**) and a
local socket whose token sits in a file beside it. A script written to
analyse data needs none of that, so the worker runs in a container
(``ContainerSandbox``) that is handed exactly what an analysis needs and
nothing else:

- **no network** (``--network none``): no socket to the gateway, no HTTP
  endpoint, no registry, no exfiltration;
- **three mounts**, and only these: the run's data file, read-only; the
  analysis output folder, read-write, as the working directory; and the open
  experiment's recipes folder, read-only. The spool, the settings files, the
  gateway descriptor and every other run are simply not in the container's
  filesystem;
- a **read-only root filesystem** with a small ``/tmp``, **no Linux
  capabilities**, ``no-new-privileges``, and memory, CPU and process caps
  from the settings;
- **no environment** from this process: the container starts with the
  image's own, plus ``FORCED_ENV`` — so an ELN or LLM key in this
  application's environment never reaches analysis code;
- the image is **never pulled** (``--pull never``): a missing image is a
  failed analysis naming it, not a download from a registry nobody chose.

The image is built locally from ``i2as/analysis/container/Dockerfile`` over a
build context holding ONLY the analysis stage and the three core modules it
may import (``stage_image_context()``), so the container cannot import the
Station even if a recipe tries — it is not there. ``build_image_command()``
plans the build; the Settings dialog's "Build image" button and
``python -m i2as.session.analysis_sandbox build-image`` run it.

**A container engine is a dependency of analysis, not of I2AS.** A setup that
never analyses installs nothing. Switching analysis on checks the engine and
the image first (``check_engine()``), and a run analysed while the engine is
down gets a ``failed`` report naming why — the facts-only entry still waits
for its human.

**What a backend is.** Two methods: ``prepare(spec)`` returns the spec the
worker will actually be given, and ``launch(spec_path)`` returns the
``LaunchPlan`` — program, arguments, environment, working directory, and the
command that stops it — that the analysis runner hands to ``QProcess``.
Neither starts anything, so both are tested without Qt, a subprocess or a
container engine.

``SubprocessSandbox`` runs the worker as a plain child process with this
interpreter. It is NOT selectable from the settings file: it is the test
suite's seam (``AnalysisRunner(sandbox_factory=...)``) and a recipe author's
debugging aid, and offers none of the isolation above.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from i2as.analysis.report import AnalysisSpec
from i2as.session.app_config import SandboxSettings, load_app_config

logger = logging.getLogger(__name__)

#: The ``LaunchPlan.backend`` of a container worker.
BACKEND_CONTAINER = "container"

#: The ``LaunchPlan.backend`` of a plain child process (tests only).
BACKEND_SUBPROCESS = "subprocess"

#: Where the analysis output folder is mounted, read-write, and the
#: container's working directory.
CONTAINER_WORKDIR = "/work"

#: Where the run's data file is mounted, read-only.
CONTAINER_INPUT_DIR = "/input"

#: Where the experiment's recipe folders are mounted, read-only, one per
#: index (``/recipes/0``, ...).
CONTAINER_RECIPES_DIR = "/recipes"

#: Where the analysis stage is installed inside the image.
CONTAINER_PACKAGE_ROOT = "/opt/i2as"

#: The prefix of every analysis container's name, so a stuck one is easy to
#: find in ``docker ps`` and is the only kind ``stop_command`` ever kills.
CONTAINER_NAME_PREFIX = "i2as-analysis-"

#: Set on every container worker: a headless plotting backend, matplotlib's
#: cache in the writable ``/tmp``, and no bytecode written anywhere.
FORCED_ENV: dict[str, str] = {
    "MPLBACKEND": "Agg",
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "HOME": "/tmp",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
}

#: How long an engine probe (``docker info``, ``docker image inspect``) may
#: take before the engine is reported unreachable. A stopped Docker Desktop
#: can hang rather than refuse.
ENGINE_PROBE_TIMEOUT_S = 10.0

#: The Dockerfile the analysis image is built from, shipped in the package.
DOCKERFILE_PATH = Path(__file__).resolve().parents[1] / "analysis" / "container" / "Dockerfile"

#: The only modules outside ``i2as.analysis`` the image carries — exactly the
#: allowlist import contract C22 names, plus the package ``__init__`` files
#: that make them importable. Anything else is absent from the container.
IMAGE_CORE_MODULES: tuple[str, ...] = (
    "__init__.py",
    "core/__init__.py",
    "core/data_reader.py",
    "core/events.py",
    "core/exceptions.py",
)


class SandboxError(Exception):
    """A sandbox that cannot run the worker — no engine, an unmountable path.

    Raised by ``prepare()``/``launch()`` and turned by the analysis runner
    into a ``failed`` report naming it, so a misconfigured sandbox is a
    visible failure on the run rather than a worker that silently never ran.
    """


@dataclass(frozen=True)
class LaunchPlan:
    """How to start one analysis worker — everything ``QProcess`` needs.

    Attributes:
        program: The program to start (the container engine, or an
            interpreter for ``SubprocessSandbox``).
        arguments: Its arguments.
        environment: The whole environment the program gets, or ``None`` to
            inherit this process's. The container engine inherits it — it
            needs to find its daemon — but the container does not.
        working_directory: The program's working directory, or ``""``.
        backend: The backend that planned it, for the log.
        stop_command: The command that stops the worker for good, run when
            it times out or is cancelled. Killing ``docker run`` alone can
            leave its container running, so a container worker names the
            container here. ``()`` when killing the process is enough.
    """

    program: str
    arguments: tuple[str, ...]
    environment: dict[str, str] | None = None
    working_directory: str = ""
    backend: str = BACKEND_CONTAINER
    stop_command: tuple[str, ...] = ()


def worker_arguments(spec_path: str | Path) -> tuple[str, ...]:
    """Return the worker's command line after the interpreter.

    Args:
        spec_path: The spec file the worker serves.

    Returns:
        ``("-m", "i2as.analysis", "run", "--spec", <path>)``.
    """
    return ("-m", "i2as.analysis", "run", "--spec", str(spec_path))


class AnalysisSandbox:
    """Where one analysis worker runs — the interface the runner drives."""

    backend: str = ""

    def prepare(self, spec: AnalysisSpec) -> AnalysisSpec:
        """Return the spec the worker will be given.

        Args:
            spec: The spec as the runner built it, with host paths.

        Returns:
            The spec to write.

        Raises:
            SandboxError: If the inputs cannot be handed to the worker.
        """
        raise NotImplementedError

    def launch(self, spec_path: Path) -> LaunchPlan:
        """Plan the worker's start.

        Args:
            spec_path: The spec file written by the runner.

        Returns:
            The launch plan.

        Raises:
            SandboxError: If the worker cannot be started as configured.
        """
        raise NotImplementedError


class SubprocessSandbox(AnalysisSandbox):
    """A plain child process with this interpreter — the test suite's seam.

    Not selectable from the settings file and not isolated in any way; see
    the module docstring.

    Args:
        python: The interpreter to start the worker with.
    """

    backend = BACKEND_SUBPROCESS

    def __init__(self, python: str = sys.executable) -> None:
        self.python = python

    def prepare(self, spec: AnalysisSpec) -> AnalysisSpec:
        """Return *spec* unchanged: the child process sees the host's paths."""
        return spec

    def launch(self, spec_path: Path) -> LaunchPlan:
        """Plan ``<python> -m i2as.analysis run --spec <spec_path>``."""
        return LaunchPlan(
            program=self.python,
            arguments=worker_arguments(spec_path),
            backend=self.backend,
        )


class ContainerSandbox(AnalysisSandbox):
    """The worker in a container: no network, three mounts, capped resources.

    ``prepare()`` rewrites the spec's host paths to the paths the container
    sees and remembers the mounts they need; ``launch()`` turns those into a
    ``<engine> run`` command line. One instance serves one analysis.

    Args:
        settings: The ``analysis.sandbox`` settings.
        engine_path: The engine executable, already resolved; ``None``
            resolves ``settings.engine`` on ``PATH`` at launch. Injected by
            tests, which have no engine installed.
        name: The container's name; ``None`` makes a fresh one.
        host_user: ``"uid:gid"`` to run as, so files written into the output
            folder belong to the person, not to root. ``None`` uses this
            process's own ids on POSIX and nothing on Windows (Docker
            Desktop maps ownership itself).
    """

    backend = BACKEND_CONTAINER

    def __init__(
        self,
        settings: SandboxSettings,
        *,
        engine_path: str | None = None,
        name: str | None = None,
        host_user: str | None = None,
    ) -> None:
        self.settings = settings
        self._engine_path = engine_path
        self.name = name or f"{CONTAINER_NAME_PREFIX}{uuid.uuid4().hex[:12]}"
        if host_user is None and hasattr(os, "getuid"):
            host_user = f"{os.getuid()}:{os.getgid()}"
        self.host_user = host_user or ""
        #: ``(host path, container path, read_only)``, filled by ``prepare()``.
        self.mounts: list[tuple[str, str, bool]] = []

    def prepare(self, spec: AnalysisSpec) -> AnalysisSpec:
        """Map the spec's host paths into the container and record the mounts.

        The output folder becomes ``/work``, the data file
        ``/input/<name>``, each recipe folder ``/recipes/<i>``, and a script
        (which lives in its output folder) ``/work/<name>``.

        Args:
            spec: The spec as the runner built it, with host paths.

        Returns:
            The spec the container worker reads.

        Raises:
            SandboxError: If the spec names no output folder, the script is
                not inside it, or a path cannot be mounted.
        """
        if not spec.output_dir:
            raise SandboxError("an analysis needs an output folder to mount")
        output_dir = Path(spec.output_dir).resolve()
        self.mounts = [(str(output_dir), CONTAINER_WORKDIR, False)]

        data_path = ""
        if spec.data_path:
            data_file = Path(spec.data_path).resolve()
            data_path = str(PurePosixPath(CONTAINER_INPUT_DIR) / data_file.name)
            self.mounts.append((str(data_file), data_path, True))

        recipe_dirs: list[str] = []
        for index, folder in enumerate(spec.recipe_dirs):
            host = Path(folder).resolve()
            if not host.is_dir():
                # Discovery tolerates a missing folder; a mount would not.
                continue
            target = str(PurePosixPath(CONTAINER_RECIPES_DIR) / str(index))
            self.mounts.append((str(host), target, True))
            recipe_dirs.append(target)

        script_path = ""
        if spec.script_path:
            script = Path(spec.script_path).resolve()
            try:
                relative = script.relative_to(output_dir)
            except ValueError as exc:
                raise SandboxError(
                    f"the analysis script {script} is not inside its output folder "
                    f"{output_dir}, so the container cannot see it"
                ) from exc
            script_path = str(PurePosixPath(CONTAINER_WORKDIR, *relative.parts))

        for host, _target, _read_only in self.mounts:
            if "," in host:
                raise SandboxError(
                    f"cannot mount {host!r} into the analysis container: the path "
                    "contains a comma"
                )
        return dataclasses.replace(
            spec,
            data_path=data_path,
            output_dir=CONTAINER_WORKDIR,
            recipe_dirs=tuple(recipe_dirs),
            script_path=script_path,
        )

    def engine(self) -> str:
        """Return the engine executable to start.

        Returns:
            The resolved path.

        Raises:
            SandboxError: If the engine is not installed.
        """
        if self._engine_path:
            return self._engine_path
        resolved = resolve_engine(self.settings.engine)
        if resolved is None:
            raise SandboxError(
                f"analysis runs in a container, but the container engine "
                f"{self.settings.engine!r} is not installed or not on PATH — install "
                "Docker Desktop (or Podman), or switch analysis off in Settings"
            )
        return resolved

    def run_arguments(self, spec_path: Path) -> tuple[str, ...]:
        """Return the ``run`` command line after the engine.

        Args:
            spec_path: The spec file, which ``prepare()``'s output folder mount
                makes visible at ``/work/spec.json``.

        Returns:
            The arguments.
        """
        settings = self.settings
        arguments: list[str] = [
            "run",
            "--rm",
            "--name", self.name,
            "--pull", "never",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:rw,size=256m",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--workdir", CONTAINER_WORKDIR,
        ]
        if settings.memory:
            arguments += ["--memory", settings.memory]
        if settings.cpus > 0:
            arguments += ["--cpus", f"{settings.cpus:g}"]
        if settings.pids_limit > 0:
            arguments += ["--pids-limit", str(settings.pids_limit)]
        if self.host_user:
            arguments += ["--user", self.host_user]
        for name, value in FORCED_ENV.items():
            arguments += ["--env", f"{name}={value}"]
        for host, target, read_only in self.mounts:
            mount = f"type=bind,source={host},target={target}"
            arguments += ["--mount", mount + (",readonly" if read_only else "")]
        arguments.append(settings.image)
        arguments += ["python", *worker_arguments(PurePosixPath(CONTAINER_WORKDIR) / spec_path.name)]
        return tuple(arguments)

    def launch(self, spec_path: Path) -> LaunchPlan:
        """Plan ``<engine> run ... <image> python -m i2as.analysis run``.

        Args:
            spec_path: The spec file written by the runner into the output
                folder.

        Returns:
            The launch plan, with ``stop_command`` naming the container.

        Raises:
            SandboxError: If ``prepare()`` was never called or the engine is
                not installed.
        """
        if not self.mounts:
            raise SandboxError("the analysis container was launched before it was prepared")
        engine = self.engine()
        return LaunchPlan(
            program=engine,
            arguments=self.run_arguments(spec_path),
            environment=engine_environment(engine),
            working_directory=str(spec_path.parent),
            backend=self.backend,
            stop_command=(engine, "kill", self.name),
        )


def build_sandbox(settings: SandboxSettings) -> AnalysisSandbox:
    """Build the sandbox one analysis runs in — always a container.

    Args:
        settings: The ``analysis.sandbox`` settings.

    Returns:
        A fresh ``ContainerSandbox``.
    """
    return ContainerSandbox(settings)


# ── The engine and the image ──────────────────────────────────────────────


def _known_engine_locations(engine: str) -> list[Path]:
    """Return where an engine's installer puts it, for when ``PATH`` does not say.

    Docker Desktop on Windows installs per-user or machine-wide and adds its
    folder to ``PATH`` only for programs started AFTER the install — so an
    application (or the shell that launched it) started before then would
    report the engine missing. These are the installers' own locations.

    Args:
        engine: The bare command name (``docker`` or ``podman``).

    Returns:
        Candidate executable paths, most likely first.
    """
    exe = f"{engine}.exe" if os.name == "nt" else engine
    places: list[Path] = []
    if os.name == "nt":
        for root_var, sub in (
            ("LOCALAPPDATA", r"Programs\DockerDesktop\resources\bin"),
            ("ProgramFiles", r"Docker\Docker\resources\bin"),
            ("ProgramFiles", r"RedHat\Podman"),
        ):
            root = os.environ.get(root_var)
            if root:
                places.append(Path(root) / sub / exe)
    else:
        places += [
            Path("/usr/local/bin") / exe,
            Path("/opt/homebrew/bin") / exe,
            Path("/Applications/Docker.app/Contents/Resources/bin") / exe,
        ]
    return places


def resolve_engine(engine: str) -> str | None:
    """Return the engine executable's path, or ``None`` when it is not installed.

    Args:
        engine: A bare command (``docker``) looked up on ``PATH`` and then in
            the installers' own locations, or a path.

    Returns:
        The executable's path, or ``None``.
    """
    if not engine:
        return None
    candidate = Path(engine)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return str(candidate) if candidate.is_file() else None
    found = shutil.which(engine)
    if found:
        return found
    for place in _known_engine_locations(engine):
        if place.is_file():
            return str(place)
    return None


def engine_environment(engine_path: str) -> dict[str, str]:
    """Return this process's environment with the engine's folder first on ``PATH``.

    The engine's client starts helpers from its own folder — Docker's
    credential helper (``docker-credential-desktop``) above all, without which
    pulling the base image fails — so that folder must be on the ``PATH`` of
    whatever runs the engine, even when the engine was found outside ``PATH``.
    This is the engine CLIENT's environment; the container gets none of it.

    Args:
        engine_path: The resolved engine executable.

    Returns:
        The environment to start the engine with.
    """
    environment = dict(os.environ)
    folder = str(Path(engine_path).parent)
    current = environment.get("PATH", "")
    if folder not in current.split(os.pathsep):
        environment["PATH"] = folder + os.pathsep + current if current else folder
    return environment


@dataclass(frozen=True)
class EngineStatus:
    """What ``check_engine()`` found.

    Attributes:
        engine_found: The engine executable is installed.
        engine_running: Its daemon answered.
        image_present: The configured image exists locally.
        detail: One human sentence naming what is missing, or what is ready.
    """

    engine_found: bool = False
    engine_running: bool = False
    image_present: bool = False
    detail: str = ""

    @property
    def ready(self) -> bool:
        """Whether an analysis could start right now."""
        return self.engine_found and self.engine_running and self.image_present


def _probe(command: list[str]) -> tuple[bool, str]:
    """Run one short engine command, never raising.

    Args:
        command: The command line.

    Returns:
        ``(succeeded, first line of its output or error)``.
    """
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=ENGINE_PROBE_TIMEOUT_S,
            check=False,
            env=engine_environment(command[0]),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (completed.stdout or completed.stderr or "").strip()
    return completed.returncode == 0, output.splitlines()[0] if output else ""


def check_engine(settings: SandboxSettings) -> EngineStatus:
    """Check the container engine and the image, blocking for a few seconds at most.

    Args:
        settings: The ``analysis.sandbox`` settings.

    Returns:
        What was found. Never raises.
    """
    engine = resolve_engine(settings.engine)
    if engine is None:
        return EngineStatus(
            detail=(
                f"{settings.engine!r} is not installed or not on PATH. Analysis runs in "
                "a container: install Docker Desktop (or Podman) to switch it on."
            )
        )
    running, output = _probe([engine, "info", "--format", "{{.ServerVersion}}"])
    # `docker info` can exit 0 with an EMPTY server version while the daemon
    # is failing to start (Docker Desktop prints its error to stderr), so a
    # version string, not the exit code, is what proves the daemon answered.
    running = running and bool(output) and not output.lower().startswith("error")
    if not running:
        return EngineStatus(
            engine_found=True,
            detail=f"{settings.engine} is installed but not running ({output or 'no answer'}). Start it and check again.",
        )
    present, _ = _probe([engine, "image", "inspect", "--format", "{{.Id}}", settings.image])
    if not present:
        return EngineStatus(
            engine_found=True,
            engine_running=True,
            detail=f"{settings.engine} {output} is running, but the image {settings.image!r} is not built yet.",
        )
    return EngineStatus(
        engine_found=True,
        engine_running=True,
        image_present=True,
        detail=f"Ready: {settings.engine} {output}, image {settings.image!r}.",
    )


def stage_image_context(destination: Path) -> Path:
    """Write the analysis image's build context into *destination*.

    The context holds the Dockerfile and ONLY the modules the worker may
    import — the whole ``i2as/analysis`` package and ``IMAGE_CORE_MODULES`` —
    so the Station, the Orchestrator, the session layer and the drivers are
    absent from the image, not merely unimported.

    Args:
        destination: An empty (or new) folder.

    Returns:
        *destination*.

    Raises:
        OSError: If a file cannot be copied.
    """
    package = Path(__file__).resolve().parents[1]
    target = destination / "i2as"
    shutil.copytree(
        package / "analysis",
        target / "analysis",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "container"),
    )
    for relative in IMAGE_CORE_MODULES:
        source = package / relative
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target / relative)
    shutil.copy2(DOCKERFILE_PATH, destination / "Dockerfile")
    return destination


def build_image_command(settings: SandboxSettings, context: Path) -> list[str]:
    """Return the command that builds the analysis image from a staged context.

    Args:
        settings: The ``analysis.sandbox`` settings (engine and image tag).
        context: A folder ``stage_image_context()`` wrote.

    Returns:
        ``[<engine>, "build", "--tag", <image>, <context>]``.

    Raises:
        SandboxError: If the engine is not installed.
    """
    engine = resolve_engine(settings.engine)
    if engine is None:
        raise SandboxError(f"the container engine {settings.engine!r} is not installed")
    return [engine, "build", "--tag", settings.image, str(context)]


def _main(argv: list[str] | None = None) -> int:
    """``python -m i2as.session.analysis_sandbox check|build-image``.

    For a headless setup, or a lab building the image before opening the
    GUI. Reads the engine and image from the general settings file.

    Args:
        argv: The arguments; ``None`` reads ``sys.argv``.

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(prog="python -m i2as.session.analysis_sandbox")
    parser.add_argument("command", choices=("check", "build-image"))
    args = parser.parse_args(argv)
    settings = load_app_config().analysis.sandbox
    if args.command == "check":
        status = check_engine(settings)
        print(status.detail)
        return 0 if status.ready else 1
    with tempfile.TemporaryDirectory(prefix="i2as-analysis-image-") as folder:
        context = stage_image_context(Path(folder))
        try:
            command = build_image_command(settings, context)
        except SandboxError as exc:
            print(exc, file=sys.stderr)
            return 1
        print("$", " ".join(command))
        return subprocess.call(command, env=engine_environment(command[0]))


__all__ = [
    "BACKEND_CONTAINER",
    "BACKEND_SUBPROCESS",
    "CONTAINER_INPUT_DIR",
    "CONTAINER_NAME_PREFIX",
    "CONTAINER_RECIPES_DIR",
    "CONTAINER_WORKDIR",
    "DOCKERFILE_PATH",
    "FORCED_ENV",
    "IMAGE_CORE_MODULES",
    "AnalysisSandbox",
    "ContainerSandbox",
    "EngineStatus",
    "LaunchPlan",
    "SandboxError",
    "SubprocessSandbox",
    "build_image_command",
    "build_sandbox",
    "check_engine",
    "engine_environment",
    "resolve_engine",
    "stage_image_context",
    "worker_arguments",
]


if __name__ == "__main__":
    sys.exit(_main())
