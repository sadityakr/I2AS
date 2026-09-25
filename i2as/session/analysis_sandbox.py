"""The analysis sandbox — WHERE the analysis worker runs, as one pluggable choice.

The analysis worker already runs recipe and script code in its own process,
and import contract C22 already guarantees that process cannot IMPORT the
Station. That is not the same as saying the code it runs cannot reach the
station: an agent-written script runs as the same user as this application,
and this application accepts commands through files (the **Request spool**)
and a local socket whose token sits in a file beside it. A script written to
analyse data does not need any of that, so the sandbox's job is to hand the
worker exactly what an analysis needs — one run file, one output folder, an
interpreter with the analysis libraries — and as little else as the backend
can manage.

**Backends** (``analysis.sandbox.backend`` in the settings file):

- ``local`` — the worker exactly as it always ran: this interpreter, this
  process's environment, this working directory. The default, because it
  needs no setup.
- ``venv`` — the worker started with a SEPARATE interpreter (a virtual
  environment with ``i2as[analysis]`` and the lab's own analysis libraries —
  scipy, scikit-image, a model's weights — installed into it and not into
  this application), with:

  * a **scrubbed environment**: only ``DEFAULT_ENV_ALLOWLIST`` and the
    configured ``env_passthrough`` names are copied across, and a name that
    looks like a credential (``SECRET_MARKERS``) is dropped even when
    listed — so an ELN key or an LLM key in this process's environment
    never reaches analysis code;
  * a **private copy of the run file** (``stage_inputs``), written into the
    analysis folder, so a script is handed a file it may do anything to
    without touching the recorded original;
  * the analysis folder as its **working directory**.

  What ``venv`` does NOT do is change the operating-system user: a hostile
  script could still walk the filesystem to the spool folder. It separates
  dependencies, credentials and data; a container backend (no network, only
  the analysis folder mounted) is the hard boundary, and it is a further
  ``AnalysisSandbox`` behind this same interface — see
  ``docs/analysis-agent.md``.

**What a backend is.** Two methods: ``prepare(spec)`` returns the spec the
worker will actually be given (after staging its inputs), and
``launch(spec_path)`` returns the ``LaunchPlan`` — program, arguments,
environment, working directory — that the analysis runner hands to
``QProcess``. Neither starts anything, so both are tested without Qt and
without a subprocess.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from i2as.analysis.report import AnalysisSpec
from i2as.session.eln.settings import SandboxSettings

logger = logging.getLogger(__name__)

#: The ``local`` backend: this interpreter, this environment.
BACKEND_LOCAL = "local"

#: The ``venv`` backend: a separate interpreter, a scrubbed environment and
#: staged inputs.
BACKEND_VENV = "venv"

#: The folder, inside an analysis's output folder, its staged inputs go to.
INPUT_DIRNAME = "input"

#: The environment a ``venv`` worker inherits by default: what an interpreter
#: needs to start and find a temporary directory, on POSIX and on Windows —
#: and nothing that configures this application.
DEFAULT_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "LOCALAPPDATA",
    "APPDATA",
)

#: Substrings that mark an environment variable as a credential, or as this
#: application's own configuration. Such a name is never passed to a
#: ``venv`` worker, whatever ``env_passthrough`` says — a typo in a settings
#: file must narrow what analysis code sees, never widen it.
SECRET_MARKERS: tuple[str, ...] = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "I2AS_",
)

#: Set on every ``venv`` worker: a headless plotting backend, no per-user
#: site-packages leaking in from outside the sandbox's own environment, and
#: no bytecode written next to recipe files.
FORCED_ENV: dict[str, str] = {
    "MPLBACKEND": "Agg",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


class SandboxError(Exception):
    """A sandbox that cannot run the worker — a missing interpreter, an unstageable file.

    Raised by ``prepare()``/``launch()`` and turned by the analysis runner
    into a ``failed`` report naming it, so a misconfigured sandbox is a
    visible failure on the run rather than a worker that silently never ran.
    """


@dataclass(frozen=True)
class LaunchPlan:
    """How to start one analysis worker — everything ``QProcess`` needs.

    Attributes:
        program: The interpreter to start.
        arguments: Its arguments.
        environment: The whole environment the worker gets, or ``None`` to
            inherit this process's.
        working_directory: The worker's working directory, or ``""`` to
            inherit this process's.
        backend: The backend that planned it, for the log.
    """

    program: str
    arguments: tuple[str, ...]
    environment: dict[str, str] | None = None
    working_directory: str = ""
    backend: str = BACKEND_LOCAL


def worker_arguments(spec_path: Path) -> tuple[str, ...]:
    """Return the worker's command line after the interpreter.

    Args:
        spec_path: The spec file the worker serves.

    Returns:
        ``("-m", "i2as.analysis", "run", "--spec", <path>)``.
    """
    return ("-m", "i2as.analysis", "run", "--spec", str(spec_path))


class AnalysisSandbox:
    """Where one analysis worker runs. The base class is the ``local`` backend."""

    backend: str = BACKEND_LOCAL

    def __init__(self, python: str = sys.executable) -> None:
        """Build the backend.

        Args:
            python: The interpreter to start the worker with.
        """
        self.python = python

    def prepare(self, spec: AnalysisSpec) -> AnalysisSpec:
        """Return the spec the worker will be given. ``local`` changes nothing.

        Args:
            spec: The spec as the runner built it.

        Returns:
            The spec to write.

        Raises:
            SandboxError: If the inputs cannot be staged.
        """
        return spec

    def launch(self, spec_path: Path) -> LaunchPlan:
        """Plan the worker's start.

        Args:
            spec_path: The spec file written by the runner.

        Returns:
            The launch plan.

        Raises:
            SandboxError: If the worker cannot be started as configured.
        """
        return LaunchPlan(program=self.python, arguments=worker_arguments(spec_path))


class VenvSandbox(AnalysisSandbox):
    """A separate interpreter, a scrubbed environment and a private copy of the run."""

    backend = BACKEND_VENV

    def __init__(
        self,
        python: str,
        *,
        stage_inputs: bool = True,
        env_passthrough: Iterable[str] = (),
        environment: Mapping[str, str] | None = None,
    ) -> None:
        """Build the backend.

        Args:
            python: The sandbox environment's interpreter.
            stage_inputs: Copy the run file into the analysis folder.
            env_passthrough: Extra variable names the worker may see.
            environment: The environment to scrub; ``None`` reads
                ``os.environ`` at launch time. Injected by tests.
        """
        super().__init__(python)
        self.stage_inputs = stage_inputs
        self.env_passthrough = tuple(env_passthrough)
        self._environment = environment

    def prepare(self, spec: AnalysisSpec) -> AnalysisSpec:
        """Stage the run file into ``<output_dir>/input/`` and point the spec at it.

        Args:
            spec: The spec as the runner built it.

        Returns:
            The spec naming the staged copy; unchanged when staging is off or
            the spec names no data file.

        Raises:
            SandboxError: If the copy cannot be made.
        """
        if not self.stage_inputs or not spec.data_path or not spec.output_dir:
            return spec
        source = Path(spec.data_path)
        target = Path(spec.output_dir) / INPUT_DIRNAME / source.name
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except OSError as exc:
            raise SandboxError(
                f"could not stage the run file {source} into the analysis "
                f"sandbox: {exc}"
            ) from exc
        return dataclasses.replace(spec, data_path=str(target))

    def launch(self, spec_path: Path) -> LaunchPlan:
        """Plan the worker's start with the sandbox interpreter and environment.

        Args:
            spec_path: The spec file written by the runner.

        Returns:
            The launch plan.

        Raises:
            SandboxError: If no interpreter is configured or it is not a file.
        """
        if not self.python:
            raise SandboxError(
                "the analysis sandbox is 'venv' but no interpreter is configured; "
                "set analysis.sandbox.python to the sandbox environment's python"
            )
        if not Path(self.python).is_file():
            raise SandboxError(
                f"the analysis sandbox interpreter {self.python!r} does not exist"
            )
        source = os.environ if self._environment is None else self._environment
        return LaunchPlan(
            program=self.python,
            arguments=worker_arguments(spec_path),
            environment=scrubbed_environment(source, self.env_passthrough),
            working_directory=str(spec_path.parent),
            backend=self.backend,
        )


def is_secret_name(name: str) -> bool:
    """Whether an environment variable name looks like a credential or app config.

    Args:
        name: The variable's name.

    Returns:
        ``True`` when any of ``SECRET_MARKERS`` appears in it, case-insensitively.
    """
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def scrubbed_environment(
    source: Mapping[str, str], passthrough: Iterable[str] = ()
) -> dict[str, str]:
    """Return the environment a ``venv`` worker is started with.

    Args:
        source: The environment to take values from.
        passthrough: Extra names allowed through.

    Returns:
        The allow-listed variables that are set in *source* and are not
        credentials, plus ``FORCED_ENV``.
    """
    allowed = [*DEFAULT_ENV_ALLOWLIST, *passthrough]
    environment: dict[str, str] = {}
    for name in allowed:
        if is_secret_name(name):
            logger.warning(
                "Analysis sandbox: %r looks like a credential and is not passed "
                "to the worker",
                name,
            )
            continue
        if name in source:
            environment[name] = str(source[name])
    environment.update(FORCED_ENV)
    return environment


def build_sandbox(settings: SandboxSettings, default_python: str = sys.executable) -> AnalysisSandbox:
    """Build the backend a settings record names.

    Args:
        settings: The ``analysis.sandbox`` settings.
        default_python: The interpreter the ``local`` backend uses.

    Returns:
        The backend; an unknown backend name is ``local``, logged.
    """
    if settings.backend == BACKEND_VENV:
        return VenvSandbox(
            settings.python,
            stage_inputs=settings.stage_inputs,
            env_passthrough=settings.env_passthrough,
        )
    if settings.backend != BACKEND_LOCAL:
        logger.warning(
            "Unknown analysis sandbox %r; running the worker locally", settings.backend
        )
    return AnalysisSandbox(default_python)


__all__ = [
    "BACKEND_LOCAL",
    "BACKEND_VENV",
    "DEFAULT_ENV_ALLOWLIST",
    "FORCED_ENV",
    "INPUT_DIRNAME",
    "SECRET_MARKERS",
    "AnalysisSandbox",
    "LaunchPlan",
    "SandboxError",
    "VenvSandbox",
    "build_sandbox",
    "is_secret_name",
    "scrubbed_environment",
    "worker_arguments",
]
