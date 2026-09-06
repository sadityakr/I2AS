"""TieredTrendLogger — live cascading-downsample writer for trend history.

Qt-free by design: this module imports nothing from PyQt6, the Orchestrator,
Virtual Instruments, or drivers, so it cannot violate any layer-boundary
import-linter contract. It is the write side only; ``trend_history.py`` is
the read side, querying the JSONL tiers this class writes, and
``Orchestrator.tick()`` calls ``record()`` once per tick.
"""

from __future__ import annotations

import json
import logging
import math

from i2as.core.logging_config import (
    TREND_3MIN_LOGGER,
    TREND_HOURLY_LOGGER,
    TREND_RAW_LOGGER,
)

logger = logging.getLogger(__name__)

# Bucket interval, in seconds, for each aggregate tier.
_BUCKET_3MIN_S = 180.0
_BUCKET_HOURLY_S = 3600.0


class _BucketAccumulator:
    """One pending fixed-width bucket's per-key running statistics.

    Tracks, per flat key, ``[min, max, sum, sumsq, count]`` plus the
    bucket-level sample count ``n`` (number of ``record()`` calls that fed
    this bucket, which can exceed any single key's ``count`` because a key
    can start appearing mid-bucket, e.g. a VI coming online).
    """

    def __init__(self, interval_s: float) -> None:
        """Initialise an empty accumulator for a given bucket width.

        Args:
            interval_s: Bucket width in seconds (180 for the 3-min tier,
                3600 for the hourly tier).
        """
        self.interval_s = interval_s
        self.bucket_start: float | None = None
        self.n = 0
        self._stats: dict[str, list[float]] = {}

    def bucket_start_for(self, timestamp: float) -> float:
        """Return the aligned bucket start for ``timestamp``.

        Alignment is ``timestamp - (timestamp % interval_s)`` on epoch
        seconds, which is inherently UTC-aligned (no local-time/DST
        arithmetic involved), per the plan's §2.

        Args:
            timestamp: Unix timestamp of the sample.

        Returns:
            The epoch-second start of the bucket containing ``timestamp``.
        """
        return timestamp - (timestamp % self.interval_s)

    def add(self, flat: dict[str, float], timestamp: float) -> None:
        """Fold one sample's numeric values into the running per-key stats.

        Assumes ``timestamp`` belongs to the currently pending bucket; the
        caller (``TieredTrendLogger``) is responsible for flushing and
        resetting on a bucket boundary crossing before calling this.

        Args:
            flat: Flattened reading dict, ``{key: value}``.
            timestamp: Unix timestamp of the sample (unused here beyond the
                caller's bucket-boundary bookkeeping, kept for symmetry).
        """
        self.n += 1
        for key, value in flat.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            v = float(value)
            stats = self._stats.get(key)
            if stats is None:
                self._stats[key] = [v, v, v, v * v, 1]
            else:
                stats[0] = min(stats[0], v)
                stats[1] = max(stats[1], v)
                stats[2] += v
                stats[3] += v * v
                stats[4] += 1

    def flush_payload(self) -> dict[str, object] | None:
        """Build the flushable JSON payload for the pending bucket.

        Returns:
            The record dict (``{"t": ..., "n": ..., "v": {...}}``), or
            ``None`` if the bucket never received a sample (should not
            normally happen given the caller's start/flush discipline, but
            guarded defensively).
        """
        if self.bucket_start is None:
            return None
        values: dict[str, dict[str, float]] = {}
        for key, (v_min, v_max, v_sum, v_sumsq, count) in self._stats.items():
            mean = v_sum / count
            variance = max(0.0, v_sumsq / count - mean * mean)
            std = math.sqrt(variance)
            values[key] = {
                "min": v_min,
                "max": v_max,
                "mean": mean,
                "std": std,
                "count": int(count),
            }
        return {"t": self.bucket_start, "n": self.n, "v": values}

    def reset(self, bucket_start: float) -> None:
        """Clear accumulated stats and start a new pending bucket.

        Args:
            bucket_start: Epoch-second start of the new pending bucket.
        """
        self.bucket_start = bucket_start
        self.n = 0
        self._stats = {}


class TieredTrendLogger:
    """Live incremental writer for the raw / 3-min / hourly trend-history tiers.

    ``record()`` is called once per Orchestrator tick, with the
    already-flattened, measurement-VI-excluded reading dict from
    ``Station.last_state_flat()``. Each call:

    1. Writes one raw-tier JSONL line, throttled by ``min_raw_interval_s``
       so the raw tier's on-disk growth is decoupled from the configured
       tick interval (``monitor.yaml: tick_interval_ms``).
    2. Folds the sample into two independent bucket accumulators (3-min,
       1-hour), regardless of whether the raw write was throttled — the
       aggregate tiers must see every sample or their ``count``/``mean``
       would silently undercount.
    3. Flushes a bucket to its own JSONL logger the moment a sample lands
       in the next bucket, i.e. buckets are closed and flushed lazily, on
       the following sample, not on a wall-clock timer.

    Accepted gaps:

    - **Crash/restart mid-bucket** loses that one in-progress 3-min or
      1-hour bucket: it is never flushed, not corrupted. The raw tier still
      holds the underlying points for anything still within its retention,
      so no data is silently lost, only its aggregate form for that one
      window.
    - **Re-aggregating ``mean``/``min``/``max``/``std``/``count`` across
      multiple flushed buckets is exact**, including ``std``: each bucket's
      ``std`` is a population standard deviation computed from exact sums
      (``sumsq``/``count``/``mean``), so its second moment about zero
      (``count * (std**2 + mean**2)``) is exactly recoverable and summable
      across buckets, and the law of total variance reconstructs the exact
      combined variance from the combined second moment and combined mean.
      This is why every flushed bucket carries an explicit per-key
      ``count`` alongside ``std``: without both, a caller re-aggregating
      several buckets could only fall back to an approximate
      pooled-within-bucket estimate that understates the true combined
      spread whenever bucket means differ (see ``trend_history.py``'s
      ``KeySummary``/``_summarize_aggregate``, which perform this exact
      recombination).
    """

    def __init__(
        self,
        *,
        min_raw_interval_s: float = 1.0,
        raw_logger: logging.Logger | None = None,
        bucket_3min_logger: logging.Logger | None = None,
        hourly_logger: logging.Logger | None = None,
    ) -> None:
        """Initialise the tiered writer.

        Args:
            min_raw_interval_s: Minimum seconds between two accepted raw-tier
                writes. A sample arriving sooner than this since the last
                raw write skips only the raw line; the aggregate
                accumulators still see it. Defaults to 1.0 s.
            raw_logger: Logger for raw-tier lines. Defaults to
                ``logging.getLogger(TREND_RAW_LOGGER)``. Injectable so
                tests can pass a throwaway logger wired to a tmp file
                instead of the real one, which ``setup_logging()``'s
                idempotency guard makes impractical to re-point.
            bucket_3min_logger: Logger for flushed 3-min buckets. Defaults
                to ``logging.getLogger(TREND_3MIN_LOGGER)``.
            hourly_logger: Logger for flushed 1-hour buckets. Defaults to
                ``logging.getLogger(TREND_HOURLY_LOGGER)``.
        """
        self.min_raw_interval_s = min_raw_interval_s
        self._raw_logger = raw_logger or logging.getLogger(TREND_RAW_LOGGER)
        self._bucket_3min_logger = bucket_3min_logger or logging.getLogger(
            TREND_3MIN_LOGGER
        )
        self._hourly_logger = hourly_logger or logging.getLogger(TREND_HOURLY_LOGGER)

        self._last_raw_write_t: float | None = None
        self._bucket_3min = _BucketAccumulator(_BUCKET_3MIN_S)
        self._bucket_hourly = _BucketAccumulator(_BUCKET_HOURLY_S)

    def record(
        self, flat: dict[str, float], timestamp: float, orch_state: str | None = None
    ) -> None:
        """Record one sample into the raw tier and both aggregate tiers.

        Never lets a logging failure propagate: this is called once per
        Orchestrator tick, and a malformed value or a transient I/O error
        writing the JSONL stream must not fail the tick.

        Args:
            flat: Flattened reading dict, ``{key: value}``, as produced by
                ``Station.last_state_flat()``. Non-numeric values and
                ``bool`` (a subclass of ``int``) are skipped.
            timestamp: Unix timestamp of the sample.
            orch_state: Orchestrator state name (e.g. ``"RAMPING"``), or
                ``None`` to omit the ``"s"`` field entirely from the raw
                record.
        """
        try:
            self._record_raw(flat, timestamp, orch_state)
        except Exception:
            logger.exception("TieredTrendLogger: raw-tier write failed")

        for accumulator, tier_logger in (
            (self._bucket_3min, self._bucket_3min_logger),
            (self._bucket_hourly, self._hourly_logger),
        ):
            try:
                self._record_bucket(accumulator, tier_logger, flat, timestamp)
            except Exception:
                logger.exception("TieredTrendLogger: aggregate-tier update failed")

    def _record_raw(
        self, flat: dict[str, float], timestamp: float, orch_state: str | None
    ) -> None:
        """Write one throttled raw-tier JSONL line.

        Args:
            flat: Flattened reading dict.
            timestamp: Unix timestamp of the sample.
            orch_state: Orchestrator state name, or ``None`` to omit ``"s"``.
        """
        if (
            self._last_raw_write_t is not None
            and (timestamp - self._last_raw_write_t) < self.min_raw_interval_s
        ):
            return

        values = {
            key: float(value)
            for key, value in flat.items()
            if not isinstance(value, bool) and isinstance(value, (int, float))
        }
        record: dict[str, object] = {"t": timestamp}
        if orch_state is not None:
            record["s"] = orch_state
        record["v"] = values

        self._raw_logger.info(json.dumps(record))
        self._last_raw_write_t = timestamp

    def _record_bucket(
        self,
        accumulator: _BucketAccumulator,
        tier_logger: logging.Logger,
        flat: dict[str, float],
        timestamp: float,
    ) -> None:
        """Fold one sample into ``accumulator``, flushing on a boundary crossing.

        Args:
            accumulator: The 3-min or 1-hour bucket accumulator to update.
            tier_logger: Logger to flush the closed bucket to.
            flat: Flattened reading dict.
            timestamp: Unix timestamp of the sample.
        """
        bucket_start = accumulator.bucket_start_for(timestamp)
        if accumulator.bucket_start is None:
            accumulator.reset(bucket_start)
        elif bucket_start != accumulator.bucket_start:
            payload = accumulator.flush_payload()
            if payload is not None:
                tier_logger.info(json.dumps(payload))
            accumulator.reset(bucket_start)

        accumulator.add(flat, timestamp)
