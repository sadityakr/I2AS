"""I2AS logging setup.

Call setup_logging() once at application startup. All modules use
logging.getLogger(__name__) — never print().

This module is import-linter contract C1 foundation: it must import nothing
else from the ``i2as`` package, stdlib only, except ``i2as.core.paths``
(itself a C1 foundation module) for ``log_directory()``.
"""

import logging
import logging.handlers
from pathlib import Path

from i2as.core.paths import log_directory

#: The root of every logger the application owns — the package name, so that
#: ``logging.getLogger(__name__)`` in any module hangs off it and the one
#: ``setup_logging()`` configures reaches them all. The named streams below
#: are the only loggers spelled out by hand anywhere; a client that must not
#: import this module (the MCP adapter, held to stdlib and the events
#: vocabulary) spells the same root itself and is pinned to it by test.
LOGGER_ROOT = "i2as"

#: The human-readable rotating log file ``setup_logging()`` writes.
LOG_FILENAME = f"{LOGGER_ROOT}.log"

#: The four ``propagate=False`` JSONL streams (see ``setup_logging``).
STATUS_LOGGER = f"{LOGGER_ROOT}.status"
TREND_RAW_LOGGER = f"{LOGGER_ROOT}.trend_raw"
TREND_3MIN_LOGGER = f"{LOGGER_ROOT}.trend_3min"
TREND_HOURLY_LOGGER = f"{LOGGER_ROOT}.trend_hourly"

#: The prefix of the per-VI bus-traffic loggers ``BaseVirtualInstrument``
#: writes to (``<prefix><vi_name>``); the GUI log panel keeps their
#: sub-WARNING records out of the window and in the file.
VI_LOGGER_PREFIX = f"{LOGGER_ROOT}.vi."


def _add_jsonl_handler(
    name: str, path: Path, *, when: str, backup_count: int
) -> None:
    """Configure one propagate=False JSONL logger with a timed-rotating handler.

    Shared by ``i2as.status`` and the three ``i2as.trend_*`` loggers
    (see the module-level table in the docstring of ``setup_logging``) so the
    four near-identical blocks collapse into one place. Each stream is one
    JSON object per line, kept off the human console/file handlers
    (``propagate=False``) and idempotency-guarded so repeated
    ``setup_logging()`` calls never duplicate handlers.

    The idempotency guard asks whether *this* stream's handler is already
    installed, not whether the logger has any handler at all. The weaker
    question is a proxy that a foreign handler satisfies: anything that
    attaches to one of these loggers first (a test harness capturing logs, an
    embedding application, a debugger) would make this function conclude it
    had already run and silently skip installing the writer, leaving the JSONL
    file empty while the app appears healthy. These streams are the input to
    the operational-status and trend-history readers, so that failure surfaces
    only much later, as missing data.

    Args:
        name: Logger name, e.g. ``"i2as.status"``.
        path: Full path to the JSONL file this logger writes.
        when: ``TimedRotatingFileHandler`` rotation unit (``"midnight"`` for
            daily, ``"W0"`` for weekly on Monday).
        backup_count: Number of rotated backups to retain.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    already_installed = any(
        isinstance(existing, logging.handlers.TimedRotatingFileHandler)
        for existing in logger.handlers
    )
    if not already_installed:
        handler = logging.handlers.TimedRotatingFileHandler(
            path, when=when, backupCount=backup_count, encoding="utf-8", utc=True
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


def setup_logging(log_dir: str | Path | None = None, level: int = logging.DEBUG) -> None:
    """Configure I2AS logging with rotating file + console + JSONL streams.

    Four parallel JSONL streams, each one JSON object per line, each its own
    ``propagate=False`` logger so JSON never reaches the console/GUI log
    handlers:

    ============ ==================== ============ ===============
    Logger        File                 Rotation     backupCount
    ============ ==================== ============ ===============
    i2as.status         status.jsonl              daily (UTC)   7
    i2as.trend_raw      trend_history_raw.jsonl    daily (UTC)   2
    i2as.trend_3min     trend_history_3min.jsonl   daily (UTC)   8
    i2as.trend_hourly   trend_history_hourly.jsonl weekly (UTC) 53
    ============ ==================== ============ ===============

    ``i2as.status`` moved here from a size-based ``RotatingFileHandler``
    (10 MB x 3) to a daily ``TimedRotatingFileHandler``: its old time
    coverage was an accident of VI count and tick rate, whereas "the last N
    days of operational status" needs to be a guarantee independent of how
    busy a given day's ticking was. Its record schema and its readers
    (``status_reader.py``) are unchanged, only the handler is; readers that
    already glob rotated files keep working unmodified. ``utc=True`` on every
    handler avoids a DST-related duplicated/missing rotation boundary against
    the ``time.time()`` epochs the records carry.

    Args:
        log_dir: Directory for log files. Defaults to ``log_directory()``.
        level: Root logger level. DEBUG for development, INFO for production.
    """
    if log_dir is None:
        log_dir = log_directory()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / LOG_FILENAME

    _add_jsonl_handler(
        STATUS_LOGGER, log_dir / "status.jsonl", when="midnight", backup_count=7
    )
    _add_jsonl_handler(
        TREND_RAW_LOGGER,
        log_dir / "trend_history_raw.jsonl",
        when="midnight",
        backup_count=2,
    )
    _add_jsonl_handler(
        TREND_3MIN_LOGGER,
        log_dir / "trend_history_3min.jsonl",
        when="midnight",
        backup_count=8,
    )
    _add_jsonl_handler(
        TREND_HOURLY_LOGGER,
        log_dir / "trend_history_hourly.jsonl",
        when="W0",
        backup_count=53,
    )

    # Root logger
    root = logging.getLogger(LOGGER_ROOT)
    root.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if root.handlers:
        return

    # Rotating file handler: 5 MB per file, keep 5 backups
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    root.addHandler(file_handler)

    # Console handler (for development)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(levelname)-8s | %(name)s | %(message)s")
    console_handler.setFormatter(console_fmt)
    root.addHandler(console_handler)
