"""Running a block — in its helper process, bounded in time, never on the GUI thread.

``SubprocessBlockRunner`` is the one way the application executes a tier-2
block (a connector or a renderer written by a user). It starts
``python -m i2as.blocks.host`` with:

* a **scrubbed environment**: the few variables Python and TLS need, and —
  for a connector only — the proxy and CA-bundle variables, so nothing else
  of the application's environment (no other key) reaches the block;
* a **temporary working directory** of its own;
* the connector's **one credential** in the ``load`` request on stdin —
  never on the command line, never in the environment;
* a **timeout per call**: a call that overruns kills the process, and the
  next call starts a fresh one.

A runner is blocking by design and is called ONLY from the notebook
service's worker thread; the GUI thread and the instrument thread never wait
on one.

``InProcessBlockRunner`` runs the same protocol in the calling process. Tests
use it (as the analysis runner's tests use a stand-in sandbox); the
application never does.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol

import i2as
from i2as.blocks.discovery import KIND_CONNECTOR, KIND_RENDERER, connector_class, load_block_module
from i2as.blocks.protocol import (
    BlockError,
    dispatch_connector,
    dispatch_renderer,
    error_payload,
    raise_error,
)

logger = logging.getLogger(__name__)

#: How long a helper process may take to start and load its block.
LOAD_TIMEOUT_S = 30.0

#: Environment variables every helper keeps (Python and the OS need them).
_BASE_ENV = ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "PATH", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE", "LANG", "LC_ALL")

#: Extra variables a CONNECTOR keeps: proxies and certificate bundles.
_NETWORK_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE")


class BlockTimeout(BlockError):
    """A block call overran its time limit; its helper process was killed."""


class BlockRunner(Protocol):
    """One loaded block the application can call."""

    def call(self, method: str, args: dict[str, Any], timeout_s: float) -> Any:
        """Call one method; raise ``ElnError`` subclasses or ``BlockError``."""
        ...

    def close(self) -> None:
        """Release the block (stop its process)."""
        ...


def helper_environment(kind: str) -> dict[str, str]:
    """Return the scrubbed environment a helper process starts with.

    Args:
        kind: ``connector`` (keeps proxy/CA variables) or ``renderer``.

    Returns:
        The environment, with ``PYTHONPATH`` pointing at this I2AS so the
        host can import ``i2as.blocks`` however I2AS was installed.
    """
    keep = _BASE_ENV + (_NETWORK_ENV if kind == KIND_CONNECTOR else ())
    env = {name: os.environ[name] for name in keep if name in os.environ}
    env["PYTHONPATH"] = str(Path(i2as.__file__).resolve().parent.parent)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class SubprocessBlockRunner:
    """One block in its own helper process.

    Args:
        kind: ``connector`` or ``renderer``.
        path: The block's source file.
        settings: A connector's non-secret settings.
        credential: A connector's secret; ``""`` for a renderer.
        python: The interpreter to run the host with.
        label: A name for log lines (never the credential).
    """

    def __init__(
        self,
        kind: str,
        path: str | Path,
        settings: dict[str, Any] | None = None,
        credential: str = "",
        python: str = sys.executable,
        label: str = "",
    ) -> None:
        self._kind = kind
        self._path = str(Path(path).resolve())
        self._settings = dict(settings or {})
        self._credential = credential if kind == KIND_CONNECTOR else ""
        self._python = python
        self._label = label or Path(path).stem
        self._process: subprocess.Popen[str] | None = None
        self._answers: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._next_id = 0
        self._workdir: tempfile.TemporaryDirectory[str] | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        """Whether the helper process is alive."""
        return self._process is not None and self._process.poll() is None

    def _start(self) -> None:
        """Start the helper and load the block. Raises ``BlockError`` on failure."""
        self._stop()
        self._workdir = tempfile.TemporaryDirectory(prefix="i2as-block-", ignore_cleanup_errors=True)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = subprocess.Popen(
                [self._python, "-m", "i2as.blocks.host"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._workdir.name,
                env=helper_environment(self._kind),
                text=True,
                encoding="utf-8",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise BlockError(f"could not start the helper for {self._label}: {exc}") from exc
        self._answers = queue.Queue()
        process = self._process
        threading.Thread(target=self._read_answers, args=(process, self._answers), daemon=True).start()
        threading.Thread(target=self._read_log, args=(process,), daemon=True).start()
        try:
            self._request(
                {"op": "load", "kind": self._kind, "path": self._path, "settings": self._settings, "credential": self._credential},
                LOAD_TIMEOUT_S,
            )
        except Exception:
            self._stop()
            raise
        logger.info("Started the helper process for block %s", self._label)

    @staticmethod
    def _read_answers(process: subprocess.Popen[str], answers: queue.Queue[dict[str, Any] | None]) -> None:
        """Move each answer line into the queue; ``None`` marks the end."""
        assert process.stdout is not None
        for line in process.stdout:
            try:
                answer = json.loads(line)
            except ValueError:
                continue
            if isinstance(answer, dict):
                answers.put(answer)
        answers.put(None)

    def _read_log(self, process: subprocess.Popen[str]) -> None:
        """Log what the block wrote to stderr (its prints and tracebacks)."""
        assert process.stderr is not None
        for line in process.stderr:
            text = line.rstrip()
            if text:
                logger.info("[block %s] %s", self._label, text[:2000])

    def _request(self, message: dict[str, Any], timeout_s: float) -> Any:
        """Send one request and wait for its answer; kill the helper on timeout."""
        process = self._process
        if process is None or process.stdin is None:
            raise BlockError(f"the helper for {self._label} is not running")
        self._next_id += 1
        request_id = self._next_id
        try:
            process.stdin.write(json.dumps({**message, "id": request_id}, ensure_ascii=True) + "\n")
            process.stdin.flush()
        except OSError as exc:
            self._stop()
            raise BlockError(f"the helper for {self._label} stopped: {exc}") from exc
        while True:
            try:
                answer = self._answers.get(timeout=timeout_s)
            except queue.Empty:
                self._stop()
                raise BlockTimeout(f"{self._label} did not answer within {timeout_s:.0f} s; it was stopped") from None
            if answer is None:
                self._stop()
                raise BlockError(f"the helper for {self._label} exited unexpectedly")
            if answer.get("id") != request_id:
                continue
            if answer.get("ok"):
                return answer.get("result")
            raise_error(answer.get("error"))

    def call(self, method: str, args: dict[str, Any], timeout_s: float) -> Any:
        """Call one block method, starting the helper when it is not running.

        Raises:
            ElnError: (a subclass) what the notebook refused.
            BlockTimeout: The call overran; the helper was killed.
            BlockError: The block could not be loaded or crashed.
        """
        with self._lock:
            if not self.running:
                self._start()
            return self._request({"op": "call", "method": method, "args": args}, timeout_s)

    def _stop(self) -> None:
        """Kill the helper (if any) and remove its working directory."""
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            try:
                process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if self._workdir is not None:
            self._workdir.cleanup()
            self._workdir = None

    def close(self) -> None:
        """Stop the helper process."""
        with self._lock:
            self._stop()


class InProcessBlockRunner:
    """The same protocol, in this process — for tests only.

    Arguments and results still travel as JSON-shaped values, and failures
    are re-raised exactly as the helper process would report them, so a test
    exercises the real translation without starting a process.
    """

    def __init__(
        self,
        kind: str,
        path: str | Path,
        settings: dict[str, Any] | None = None,
        credential: str = "",
        **_: Any,
    ) -> None:
        self._kind = kind
        self._path = Path(path)
        self._settings = dict(settings or {})
        self._credential = credential
        self._block: Any = None
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, method: str, args: dict[str, Any], timeout_s: float) -> Any:
        self.calls.append((method, json.loads(json.dumps(args))))
        try:
            if self._block is None:
                module = load_block_module(self._path)
                if self._kind == KIND_RENDERER:
                    self._block = getattr(module, "render")
                else:
                    self._block = connector_class(module)(dict(self._settings), self._credential)
            if self._kind == KIND_RENDERER:
                result = dispatch_renderer(self._block, method, json.loads(json.dumps(args)))
            else:
                result = dispatch_connector(self._block, method, json.loads(json.dumps(args)))
        except Exception as exc:  # noqa: BLE001 - translated exactly as the host does
            raise_error(error_payload(exc))
        return json.loads(json.dumps(result))

    def close(self) -> None:
        self._block = None
