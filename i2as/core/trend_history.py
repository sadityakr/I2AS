"""trend_history — query surface over the tiered trend-history JSONL store.

Qt-free by design, like ``tiered_trend_logger.py``: this module imports
nothing from PyQt6, the Orchestrator, Virtual Instruments, or drivers.  It is
the reader for the store that module writes; record shapes here are dictated
by ``tiered_trend_logger.py`` and must stay in sync with it.

The primary public API is the aggregate/query surface (``summarize``,
``read_window``, ``find_crossings``), not raw-row access: a reader returning
~28,800 raw rows for a 24 h window is unusable by an LLM client, so
``read_tier`` (raw rows) is a lower-level primitive that the higher-level
functions build on, not the main entry point.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

logger = logging.getLogger(__name__)

# Bound on how many trailing records persisted_keys() inspects per tier. The
# tiers rotate daily/weekly with small backupCounts, so a live file is never
# large enough for this to matter in practice; the bound exists so a runaway
# file (e.g. a stuck rotation) cannot make a "what keys exist" query scan an
# unbounded amount of data.
_PERSISTED_KEYS_MAX_LINES = 500

# Tail-read chunk size for read_tier()'s reverse scan. 64 KiB comfortably
# covers many raw-tier records per read() call (each line is well under 1 KiB
# even at 30 keys/record) while keeping memory bounded; it is not load-bearing
# for correctness — _iter_lines_reverse() reassembles a record split across a
# chunk boundary regardless of this value (see its docstring and the
# dedicated boundary test in tests/test_trend_history.py).
_REVERSE_READ_CHUNK_BYTES = 65536


@dataclass(frozen=True)
class TierSpec:
    """One trend-history tier's identity: cadence, file, and retention.

    Attributes:
        name: Tier key as used in ``TIERS`` and returned by ``pick_tier``
            (``"raw"``, ``"3min"``, or ``"hourly"``).
        interval_s: Nominal sample/bucket width in seconds.
        filename: Live (undated) JSONL filename inside the log directory.
        retention_s: Approximate wall-clock retention, derived from the
            writer's ``TimedRotatingFileHandler`` rotation unit and
            ``backupCount`` (see ``TIERS`` below for the derivation of each
            tier's value). Informational only — actual retention is
            enforced by file count (the handler's rotation unit times
            ``backupCount``, see the derivation above ``TIERS``), not by
            this figure.
    """

    name: str
    interval_s: float
    filename: str
    retention_s: float


# Derivation of retention_s, from core/logging_config.py's handler config:
#   raw:    daily rotation, backupCount=2  -> live file + 2 backups = 3 days
#   3min:   daily rotation, backupCount=8  -> live file + 8 backups = 9 days
#   hourly: weekly rotation ("W0"), backupCount=53 -> live file + 53 backups
#           = 54 weeks = 378 days (~1 year)
TIERS: dict[str, TierSpec] = {
    "raw": TierSpec("raw", 3.0, "trend_history_raw.jsonl", 3.0 * 86400.0),
    "3min": TierSpec("3min", 180.0, "trend_history_3min.jsonl", 9.0 * 86400.0),
    "hourly": TierSpec("hourly", 3600.0, "trend_history_hourly.jsonl", 54.0 * 7.0 * 86400.0),
}


@dataclass(frozen=True)
class KeySummary:
    """Per-key aggregate statistics over a requested window, one tier.

    ``min``/``max``/``mean``/``std``/``count`` are all **exact**
    recombinations of the underlying tier's data (raw tier: computed
    directly; aggregate tiers: recombined from each bucket's exact per-key
    ``min``/``max``/``mean``/``std``/``count``). ``mean`` is weighted by
    ``count``; ``std`` is recovered via the law of total variance from each
    bucket's ``(mean, std, count)`` — this is exact because a bucket's
    ``std`` is a population standard deviation computed from exact
    sums, so its second moment (``count * (std**2 + mean**2)``) can be
    recovered exactly and summed across buckets. This is precisely why the
    writer (``TieredTrendLogger``) stores a per-key ``count`` alongside
    ``std``: without it, recombination would have to fall back to an
    approximate pooled-within-bucket estimate, which underestimates the
    true combined spread whenever bucket means differ — the wrong direction
    to be wrong in for "was this stable" questions.

    Attributes:
        min: Minimum value in the window, or ``None`` if no data.
        max: Maximum value in the window, or ``None`` if no data.
        mean: Count-weighted mean, or ``None`` if no data.
        std: Standard deviation, exact on every tier (see above), or
            ``None`` if no data.
        count: Total number of underlying samples folded into this summary.
        first_t: Timestamp of the earliest sample in the window, or
            ``None``.
        last_t: Timestamp of the latest sample in the window, or ``None``.
        tier: The tier this summary was computed from (``pick_tier``'s
            choice for the requested window).
        persisted: ``False`` if this key never appears in this tier's
            files at all (e.g. a measurement-VI key: ``Station.
            last_state_flat()`` excludes every measurement VI, so a trend
            panel on a measurement-VI key works live but is empty on any
            disk-backed window — see GLOSSARY.md "Trend history"). ``True``
            and an empty/zeroed summary means the key is persisted but
            simply had no samples in the requested window.
    """

    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    count: int
    first_t: float | None
    last_t: float | None
    tier: str
    persisted: bool


def pick_tier(window_s: float) -> str:
    """Map a requested time window to the tier that should serve it.

    Single home for this decision, so the GUI plot panel, a CLI, and any
    future agent-facing tool all inherit the same choice instead of
    reimplementing it.

    Args:
        window_s: Requested window length in seconds.

    Returns:
        ``"raw"`` for windows up to 24 h, ``"3min"`` up to 1 week, else
        ``"hourly"``.
    """
    if window_s <= 86400.0:
        return "raw"
    if window_s <= 604800.0:
        return "3min"
    return "hourly"


def _tier_files(log_dir: Path, spec: TierSpec) -> list[Path]:
    """Return the live file plus dated rotated siblings for one tier.

    Matches the base filename strictly (``trend_history_raw.jsonl``,
    optionally followed by a ``.YYYY-MM-DD`` rotation suffix), never a loose
    prefix glob, so an OneDrive/Dropbox sync-conflict copy
    (``trend_history_raw-DESKTOP-ABC123.jsonl`` or
    ``trend_history_raw (conflicted copy 2026-07-25).jsonl``) is never
    picked up as real data.

    Args:
        log_dir: Directory to search (need not exist).
        spec: The tier whose files to find.

    Returns:
        Matching file paths, in no particular order (callers sort as
        needed). Empty if ``log_dir`` does not exist.
    """
    if not log_dir.exists():
        return []
    pattern = re.compile(rf"^{re.escape(spec.filename)}(\.\d{{4}}-\d{{2}}-\d{{2}})?$")
    return [p for p in log_dir.iterdir() if p.is_file() and pattern.match(p.name)]


def _parse_line(line: str, path: Path) -> dict | None:
    """Parse one JSONL line, returning ``None`` (logged at DEBUG) if unusable.

    A blank line or a corrupt/truncated line is expected, not exceptional —
    the writer may be mid-write on the file's final line at read time — so
    this never raises and never logs above DEBUG.

    Args:
        line: Raw line text (not yet stripped).
        path: Source file, for the debug log message only.

    Returns:
        The parsed JSON object if it is a dict, else ``None``.
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        logger.debug("trend_history: skipping unparsable line in %s", path)
        return None
    if not isinstance(obj, dict):
        logger.debug("trend_history: skipping non-object line in %s", path)
        return None
    return obj


def _iter_lines_reverse(
    path: Path, chunk_size: int = _REVERSE_READ_CHUNK_BYTES
) -> Iterator[str]:
    """Yield ``path``'s lines newest-to-oldest, reading fixed-size tail chunks.

    The classic reverse-line-reading algorithm: seek backward from
    end-of-file in ``chunk_size`` blocks, split each block on ``b"\\n"``, and
    carry the block's leading fragment (``parts[0]``) forward as ``leftover``
    for the NEXT (older, further-back) read — that fragment is the start of a
    line whose end was already yielded from the chunk read just before it, so
    prepending the next chunk's bytes to it reassembles the line exactly,
    including when the split lands in the middle of a multi-byte UTF-8
    character (splitting only ever happens at ``b"\\n"``, which never appears
    as a UTF-8 continuation byte). A record straddling a chunk boundary is
    therefore never dropped or corrupted, only reassembled one read later
    than the rest of its neighbours — see the dedicated boundary test in
    ``tests/test_trend_history.py``.

    Args:
        path: File to read.
        chunk_size: Bytes to read per tail chunk. Exposed for testing (a
            small value makes it easy to force a record to straddle a
            boundary); production callers use the module default.

    Yields:
        Raw line text (not yet stripped), newest physical line first. Never
        raises — an ``OSError`` opening or reading the file yields nothing,
        logged at DEBUG, matching ``read_tier()``'s prior per-file tolerance.
    """
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            remaining = f.tell()
            leftover = b""
            at_eof_chunk = True
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                remaining -= read_size
                f.seek(remaining)
                chunk = f.read(read_size)
                data = chunk + leftover
                parts = data.split(b"\n")
                leftover = parts[0]
                tail_parts = parts[1:]
                if at_eof_chunk and tail_parts and tail_parts[-1] == b"":
                    # The file's trailing "\n" splits into a final empty
                    # fragment representing whatever comes after it — nothing
                    # — not a genuine blank line. Drop only this one, only in
                    # the chunk touching end-of-file, for parity with
                    # str.splitlines() (which likewise emits no trailing
                    # empty entry for a final newline).
                    tail_parts = tail_parts[:-1]
                at_eof_chunk = False
                for part in reversed(tail_parts):
                    yield part.decode("utf-8", errors="replace")
            if leftover:
                yield leftover.decode("utf-8", errors="replace")
    except OSError:
        logger.debug("trend_history: could not read %s", path)
        return


def _order_tier_files_newest_first(files: list[Path], spec: TierSpec) -> list[Path]:
    """Order one tier's files for a backward (newest-first) scan.

    The live (undated) file always holds the most recent records — a
    ``TimedRotatingFileHandler`` rotation renames it to a dated suffix and
    starts a fresh live file — so it sorts first, ahead of every dated
    sibling. Dated siblings then sort by their ``YYYY-MM-DD`` suffix,
    newest date first (lexicographic order matches date order for this
    zero-padded format).

    Args:
        files: This tier's files, as returned by ``_tier_files()`` (any
            order).
        spec: The tier whose live filename identifies the undated file.

    Returns:
        ``files`` reordered newest-first.
    """

    def sort_key(p: Path) -> tuple[int, str]:
        if p.name == spec.filename:
            return (1, "")
        return (0, p.name[len(spec.filename) + 1 :])

    return sorted(files, key=sort_key, reverse=True)


def read_tier(
    log_dir: Path, tier: str, window_s: float, now: float | None = None
) -> list[tuple[float, dict]]:
    """Read one tier's raw records within a trailing time window.

    Low-level primitive underneath ``read_window``/``summarize``/
    ``find_crossings``. Reads the live undated file plus its rotated
    ``.jsonl.<date>`` siblings, merges them oldest-first, and filters to
    ``now - window_s <= t <= now``.

    Scans backward: since each file is append-ordered by timestamp, this
    walks files newest-first (``_order_tier_files_newest_first``) and each
    file's lines newest-first (``_iter_lines_reverse``, fixed-size tail
    chunks), stopping the instant it reaches a record older than the
    window's lower bound. Cost is therefore set by the requested window, not
    by total retention — a 1 h window on a raw tier holding 3 days of 3 s
    samples touches roughly the ~1,200 lines the window actually needs,
    never the ~86,400 lines on disk. A malformed or blank line does not stop
    the scan (``_parse_line`` tolerates it, matching the forward reader this
    replaces); only a line with a valid, in-order timestamp older than the
    window does.

    Args:
        log_dir: Directory containing the tier's JSONL files (as resolved
            by ``i2as.core.paths.log_directory()``).
        tier: One of ``"raw"``, ``"3min"``, ``"hourly"`` (a key of
            ``TIERS``).
        window_s: Trailing window length in seconds.
        now: Reference "now" timestamp; defaults to ``time.time()``.

    Returns:
        ``(t, value_mapping)`` tuples oldest-first, where ``value_mapping``
        is the record's ``"v"`` dict verbatim (raw tier:
        ``{key: float}``; aggregate tiers:
        ``{key: {"min":.., "mean":.., "max":.., "std":.., "count":..}}``).
        Empty list if the directory or tier files are missing, or on any
        per-file read error — this function never raises.
    """
    spec = TIERS[tier]
    log_dir = Path(log_dir)
    files = _tier_files(log_dir, spec)
    if not files:
        return []

    if now is None:
        now = time.time()
    lower = now - window_s

    ordered_files = _order_tier_files_newest_first(files, spec)

    records: list[tuple[float, dict]] = []
    for path in ordered_files:
        stop = False
        # Looked up on the module rather than relying on the parameter
        # default, so a test can shrink it via monkeypatch and force this
        # scan itself to exercise a record split across a chunk boundary.
        for line in _iter_lines_reverse(path, chunk_size=_REVERSE_READ_CHUNK_BYTES):
            obj = _parse_line(line, path)
            if obj is None:
                continue
            t = obj.get("t")
            v = obj.get("v")
            if not isinstance(t, (int, float)) or isinstance(t, bool):
                continue
            if not isinstance(v, dict):
                continue
            t = float(t)
            if t < lower:
                # Append-ordered by timestamp: everything earlier in this
                # file, and every older file behind it, is older still.
                stop = True
                break
            if t > now:
                continue
            records.append((t, v))
        if stop:
            break

    records.reverse()
    return records


def persisted_keys(log_dir: Path, tier: str) -> set[str]:
    """Return the set of keys that appear in a tier's files at all.

    Used to distinguish "no data in the requested window" from "this key is
    never written to disk" — measurement-VI keys are excluded by
    ``Station.last_state_flat()`` and so never reach any tier. Bounded to
    the most recent ``_PERSISTED_KEYS_MAX_LINES``
    records across the tier's files (newest files first, by mtime), rather
    than a full scan of the retention window: a key that has ever been
    persisted recently is what matters for this distinction, and scanning
    the entire retention window would be unnecessarily slow for a store
    that can hold up to a year of hourly data.

    Reads each file backward via ``_iter_lines_reverse`` (the same
    fixed-size-chunk tail scan ``read_tier`` uses) rather than materialising
    the whole file with ``read_text().splitlines()`` — this was the sibling
    of the defect ``read_tier`` was fixed for: the line-count cap bounded
    how much got *parsed*, but not the I/O or the split, which still scaled
    with file size (a 30 MB live file dwarfed the file bound this function's
    own docstring promises). The generator is abandoned as soon as the cap
    is hit — a ``for`` loop that ``break``s never calls it again — so this
    reads only the trailing bytes it actually needs.

    Args:
        log_dir: Directory containing the tier's JSONL files.
        tier: One of ``"raw"``, ``"3min"``, ``"hourly"``.

    Returns:
        The set of ``"v"``-dict keys seen. Empty if the tier has no files.
    """
    spec = TIERS[tier]
    log_dir = Path(log_dir)
    files = _tier_files(log_dir, spec)
    if not files:
        return set()

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    keys: set[str] = set()
    lines_read = 0
    for path in files:
        for line in _iter_lines_reverse(path, chunk_size=_REVERSE_READ_CHUNK_BYTES):
            if lines_read >= _PERSISTED_KEYS_MAX_LINES:
                break
            obj = _parse_line(line, path)
            lines_read += 1
            if obj is None:
                continue
            v = obj.get("v")
            if isinstance(v, dict):
                keys.update(v.keys())
        if lines_read >= _PERSISTED_KEYS_MAX_LINES:
            break

    return keys


def read_window(
    log_dir: Path, keys: Sequence[str], window_s: float, now: float | None = None
) -> dict[str, list[tuple[float, float]]]:
    """Return plottable ``(t, value)`` series for each key over a window.

    Primary GUI-facing entry point. Picks the tier via ``pick_tier`` and
    takes the scalar value directly on the raw tier or the bucket's
    ``"mean"`` on an aggregate tier.

    Args:
        log_dir: Directory containing the trend-history JSONL files.
        keys: Flat state keys to extract (e.g. ``"magnet_z_get_field"``).
        window_s: Trailing window length in seconds.
        now: Reference "now" timestamp; defaults to ``time.time()``.

    Returns:
        ``{key: [(t, value), ...]}`` oldest-first. A key absent from the
        store yields an empty list, never a ``KeyError``.
    """
    tier = pick_tier(window_s)
    records = read_tier(log_dir, tier, window_s, now=now)

    result: dict[str, list[tuple[float, float]]] = {key: [] for key in keys}
    key_set = set(keys)
    for t, v in records:
        for key in key_set:
            entry = v.get(key)
            if entry is None:
                continue
            if isinstance(entry, dict):
                mean = entry.get("mean")
                if isinstance(mean, (int, float)) and not isinstance(mean, bool):
                    result[key].append((t, float(mean)))
            elif isinstance(entry, (int, float)) and not isinstance(entry, bool):
                result[key].append((t, float(entry)))

    return result


def _summarize_raw(records: list[tuple[float, dict]], key: str, tier: str) -> KeySummary:
    """Build a ``KeySummary`` for ``key`` directly from raw-tier samples."""
    values: list[tuple[float, float]] = []
    for t, v in records:
        val = v.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            values.append((t, float(val)))

    if not values:
        return KeySummary(None, None, None, None, 0, None, None, tier, True)

    nums = [v for _, v in values]
    n = len(nums)
    mean = sum(nums) / n
    variance = sum((x - mean) ** 2 for x in nums) / n
    std = math.sqrt(variance)
    return KeySummary(
        min(nums), max(nums), mean, std, n, values[0][0], values[-1][0], tier, True
    )


def _summarize_aggregate(records: list[tuple[float, dict]], key: str, tier: str) -> KeySummary:
    """Build a ``KeySummary`` for ``key`` by recombining aggregate-tier buckets.

    ``min``/``max``/``mean``/``std``/``count`` are all exact. ``mean`` is
    count-weighted; ``std`` is recovered via the law of total variance: each
    bucket's exact second moment about zero is recoverable from its
    ``(mean, std, count)`` as ``count * (std**2 + mean**2)`` (since
    ``std**2 = sumsq/count - mean**2`` at flush time), so summing second
    moments and counts across buckets and recombining reproduces the exact
    population variance of the full underlying sample — see
    ``KeySummary``'s docstring.
    """
    mins: list[float] = []
    maxs: list[float] = []
    weighted_mean_sum = 0.0
    weighted_sumsq = 0.0
    total_count = 0
    first_t: float | None = None
    last_t: float | None = None

    for t, v in records:
        entry = v.get(key)
        if not isinstance(entry, dict):
            continue
        count = entry.get("count")
        mean = entry.get("mean")
        if (
            not isinstance(count, (int, float))
            or isinstance(count, bool)
            or not isinstance(mean, (int, float))
            or isinstance(mean, bool)
            or count <= 0
        ):
            continue
        count = int(count)
        std = entry.get("std", 0.0)
        if not isinstance(std, (int, float)) or isinstance(std, bool):
            std = 0.0

        bucket_min = entry.get("min")
        bucket_max = entry.get("max")
        if isinstance(bucket_min, (int, float)) and not isinstance(bucket_min, bool):
            mins.append(float(bucket_min))
        if isinstance(bucket_max, (int, float)) and not isinstance(bucket_max, bool):
            maxs.append(float(bucket_max))

        weighted_mean_sum += mean * count
        # Exact second moment about zero for this bucket: sumsq = count *
        # (std**2 + mean**2), recovered from std**2 = sumsq/count - mean**2.
        weighted_sumsq += count * (std * std + mean * mean)
        total_count += count
        if first_t is None:
            first_t = t
        last_t = t

    if total_count == 0:
        return KeySummary(None, None, None, None, 0, None, None, tier, True)

    mean = weighted_mean_sum / total_count
    # Law of total variance: Var(X) = E[X^2] - E[X]^2, computed from the
    # exact combined second moment and combined mean. The max(0.0, ...)
    # guards against a negative result from floating-point cancellation
    # when the true variance is at or near zero.
    variance = max(0.0, weighted_sumsq / total_count - mean * mean)
    std = math.sqrt(variance)
    return KeySummary(
        min(mins) if mins else None,
        max(maxs) if maxs else None,
        mean,
        std,
        total_count,
        first_t,
        last_t,
        tier,
        True,
    )


def summarize(
    log_dir: Path, keys: Sequence[str], window_s: float, now: float | None = None
) -> dict[str, KeySummary]:
    """Return per-key aggregate statistics over a window — the agent-facing API.

    Evidence, not a raw-row dump: this is what an operator or an LLM agent
    asking "was X stable" or "what was the range of Y" should call, not
    ``read_tier``/``read_window``.

    Args:
        log_dir: Directory containing the trend-history JSONL files.
        keys: Flat state keys to summarize.
        window_s: Trailing window length in seconds.
        now: Reference "now" timestamp; defaults to ``time.time()``.

    Returns:
        ``{key: KeySummary}``. A key never persisted to this tier (e.g. a
        measurement-VI key, excluded by ``Station.last_state_flat()``) gets
        ``persisted=False`` and zeroed/``None`` stats, distinguishing "never
        persisted" from "no data in this window" (``persisted=True`` with
        ``count=0``).
    """
    log_dir = Path(log_dir)
    tier = pick_tier(window_s)
    records = read_tier(log_dir, tier, window_s, now=now)
    known_keys = persisted_keys(log_dir, tier)

    result: dict[str, KeySummary] = {}
    for key in keys:
        if key not in known_keys:
            result[key] = KeySummary(None, None, None, None, 0, None, None, tier, False)
            continue
        if tier == "raw":
            result[key] = _summarize_raw(records, key, tier)
        else:
            result[key] = _summarize_aggregate(records, key, tier)
    return result


def find_crossings(
    log_dir: Path,
    key: str,
    threshold: float,
    window_s: float,
    direction: str = "below",
    now: float | None = None,
) -> list[float]:
    """Return timestamps where consecutive samples of ``key`` cross ``threshold``.

    The one query beyond aggregation an agent needs for operational
    reasoning ("when did this channel last drop below 30").

    Args:
        log_dir: Directory containing the trend-history JSONL files.
        key: Flat state key to check.
        threshold: Threshold value to detect a crossing of.
        window_s: Trailing window length in seconds.
        direction: ``"below"`` (previous sample >= threshold, current <
            threshold), ``"above"`` (the inverse), or ``"both"``.
        now: Reference "now" timestamp; defaults to ``time.time()``.

    Returns:
        Timestamps of the crossing sample (the sample that landed on the
        other side of the threshold), oldest-first.

    Raises:
        ValueError: If ``direction`` is not ``"below"``, ``"above"``, or
            ``"both"``.
    """
    if direction not in ("below", "above", "both"):
        raise ValueError(f"Unknown direction: {direction!r}")

    series = read_window(log_dir, [key], window_s, now=now)[key]

    crossings: list[float] = []
    prev_value: float | None = None
    for t, value in series:
        if prev_value is not None:
            crossed_below = prev_value >= threshold and value < threshold
            crossed_above = prev_value <= threshold and value > threshold
            if (
                (direction == "below" and crossed_below)
                or (direction == "above" and crossed_above)
                or (direction == "both" and (crossed_below or crossed_above))
            ):
                crossings.append(t)
        prev_value = value

    return crossings
