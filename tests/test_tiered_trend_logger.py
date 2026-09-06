"""Tests for TieredTrendLogger, the tiered trend-history write side."""

from __future__ import annotations

import json
import logging
import math

import pytest

from i2as.core.tiered_trend_logger import TieredTrendLogger


class _ListHandler(logging.Handler):
    """Captures each emitted record's formatted message for assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())

    def payloads(self) -> list[dict]:
        return [json.loads(line) for line in self.lines]


@pytest.fixture
def loggers() -> tuple[logging.Logger, logging.Logger, logging.Logger, dict[str, _ListHandler]]:
    """Three throwaway loggers, each with a capturing handler, per test."""
    handlers = {}
    result_loggers = []
    for suffix in ("raw", "3min", "hourly"):
        name = f"test.tiered_trend_logger.{suffix}.{id(object())}"
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()
        handler = _ListHandler()
        logger.addHandler(handler)
        handlers[suffix] = handler
        result_loggers.append(logger)
    return (*result_loggers, handlers)


def _make_logger(loggers, **kwargs) -> TieredTrendLogger:
    raw_logger, bucket_3min_logger, hourly_logger, _handlers = loggers
    return TieredTrendLogger(
        raw_logger=raw_logger,
        bucket_3min_logger=bucket_3min_logger,
        hourly_logger=hourly_logger,
        **kwargs,
    )


def test_raw_record_shape_with_state(loggers) -> None:
    _raw, _b3, _hr, handlers = loggers
    ttl = _make_logger(loggers, min_raw_interval_s=0.0)
    ttl.record({"magnet_z_get_field": 1.5}, timestamp=1753401600.123, orch_state="RAMPING")

    payloads = handlers["raw"].payloads()
    assert len(payloads) == 1
    assert payloads[0] == {
        "t": 1753401600.123,
        "s": "RAMPING",
        "v": {"magnet_z_get_field": 1.5},
    }


def test_raw_record_omits_s_when_state_none(loggers) -> None:
    _raw, _b3, _hr, handlers = loggers
    ttl = _make_logger(loggers, min_raw_interval_s=0.0)
    ttl.record({"magnet_z_get_field": 1.5}, timestamp=1753401600.0, orch_state=None)

    payloads = handlers["raw"].payloads()
    assert len(payloads) == 1
    assert "s" not in payloads[0]
    assert payloads[0]["t"] == 1753401600.0
    assert payloads[0]["v"] == {"magnet_z_get_field": 1.5}


def test_min_raw_interval_throttles_raw_but_not_aggregates(loggers) -> None:
    _raw, _b3, _hr, handlers = loggers
    ttl = _make_logger(loggers, min_raw_interval_s=5.0)

    base = 0.0
    # Five samples one second apart: only the first should pass the 5s throttle.
    for i in range(5):
        ttl.record({"k": float(i)}, timestamp=base + i, orch_state=None)

    raw_payloads = handlers["raw"].payloads()
    assert len(raw_payloads) == 1

    # Force a 3-min bucket flush by crossing into the next bucket.
    ttl.record({"k": 99.0}, timestamp=200.0, orch_state=None)

    flushed = handlers["3min"].payloads()
    assert len(flushed) == 1
    # All 5 throttled samples plus the flush trigger's bucket... the trigger
    # sample (t=200) starts a NEW bucket, so the flushed one covers only the
    # first bucket's 5 samples.
    assert flushed[0]["n"] == 5
    assert flushed[0]["v"]["k"]["count"] == 5


def test_bucket_alignment_including_exact_boundary(loggers) -> None:
    _raw, _b3, _hr, handlers = loggers
    ttl = _make_logger(loggers, min_raw_interval_s=0.0)

    # Bucket 0 covers [0, 180). A sample at exactly t=180 must start bucket 1.
    ttl.record({"k": 1.0}, timestamp=0.0)
    ttl.record({"k": 2.0}, timestamp=179.999)
    ttl.record({"k": 3.0}, timestamp=180.0)  # exact boundary -> new bucket

    flushed = handlers["3min"].payloads()
    assert len(flushed) == 1
    assert flushed[0]["t"] == 0.0
    assert flushed[0]["v"]["k"]["count"] == 2
    assert flushed[0]["v"]["k"]["min"] == 1.0
    assert flushed[0]["v"]["k"]["max"] == 2.0


def test_min_max_mean_std_count_arithmetic(loggers) -> None:
    _raw, _b3, _hr, handlers = loggers
    ttl = _make_logger(loggers, min_raw_interval_s=0.0)

    for i, value in enumerate((1.0, 2.0, 3.0)):
        ttl.record({"k": value}, timestamp=float(i))
    ttl.record({"k": 0.0}, timestamp=180.0)  # trigger flush

    flushed = handlers["3min"].payloads()
    assert len(flushed) == 1
    stats = flushed[0]["v"]["k"]
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["std"] == pytest.approx(math.sqrt(2.0 / 3.0))
    assert stats["count"] == 3
    assert flushed[0]["n"] == 3


def test_std_zero_for_constant_series(loggers) -> None:
    _raw, _b3, _hr, handlers = loggers
    ttl = _make_logger(loggers, min_raw_interval_s=0.0)

    for i in range(10):
        ttl.record({"k": 5.0}, timestamp=float(i))
    ttl.record({"k": 5.0}, timestamp=180.0)  # trigger flush

    flushed = handlers["3min"].payloads()
    assert flushed[0]["v"]["k"]["std"] == 0.0


def test_std_zero_for_single_sample(loggers) -> None:
    _raw, _b3, _hr, handlers = loggers
    ttl = _make_logger(loggers, min_raw_interval_s=0.0)

    ttl.record({"k": 5.0}, timestamp=0.0)
    ttl.record({"k": 5.0}, timestamp=180.0)  # trigger flush

    flushed = handlers["3min"].payloads()
    assert flushed[0]["v"]["k"]["count"] == 1
    assert flushed[0]["v"]["k"]["std"] == 0.0


def test_per_key_count_differs_from_bucket_n_when_key_appears_mid_bucket(loggers) -> None:
    _raw, _b3, _hr, handlers = loggers
    ttl = _make_logger(loggers, min_raw_interval_s=0.0)

    ttl.record({"a": 1.0}, timestamp=0.0)
    ttl.record({"a": 2.0}, timestamp=1.0)
    ttl.record({"a": 3.0, "b": 10.0}, timestamp=2.0)  # "b" appears mid-bucket
    ttl.record({"a": 4.0}, timestamp=180.0)  # trigger flush

    flushed = handlers["3min"].payloads()
    assert flushed[0]["n"] == 3
    assert flushed[0]["v"]["a"]["count"] == 3
    assert flushed[0]["v"]["b"]["count"] == 1


def test_crash_mid_bucket_produces_no_line(loggers) -> None:
    _raw, _b3, _hr, handlers = loggers
    ttl = _make_logger(loggers, min_raw_interval_s=0.0)

    ttl.record({"k": 1.0}, timestamp=0.0)
    ttl.record({"k": 2.0}, timestamp=1.0)
    # No further sample crosses the bucket boundary -> nothing ever flushed.

    assert handlers["3min"].payloads() == []
    assert handlers["hourly"].payloads() == []


def test_non_numeric_and_bool_values_skipped(loggers) -> None:
    _raw, _b3, _hr, handlers = loggers
    ttl = _make_logger(loggers, min_raw_interval_s=0.0)

    ttl.record(
        {"num": 1.5, "flag": True, "text": "hello", "none": None},
        timestamp=0.0,
        orch_state=None,
    )
    ttl.record({"num": 2.5}, timestamp=180.0)  # trigger flush

    raw_payload = handlers["raw"].payloads()[0]
    assert raw_payload["v"] == {"num": 1.5}

    flushed = handlers["3min"].payloads()[0]
    assert set(flushed["v"].keys()) == {"num"}


def test_3min_and_1hour_tiers_roll_independently(loggers) -> None:
    _raw, _b3, _hr, handlers = loggers
    ttl = _make_logger(loggers, min_raw_interval_s=0.0)

    # Feed one sample every 3 minutes for slightly over an hour, so the
    # hourly tier accumulates across many 3-min bucket flushes before its
    # own boundary is crossed.
    t = 0.0
    n_samples = 25  # 25 * 180s = 4500s, i.e. spans one full hour boundary
    for i in range(n_samples):
        ttl.record({"k": float(i)}, timestamp=t)
        t += 180.0

    flushed_3min = handlers["3min"].payloads()
    flushed_hourly = handlers["hourly"].payloads()

    # 3-min tier flushes on every boundary crossing among 25 samples spaced
    # exactly 180s apart -> 24 flushes (the 25th sample's bucket is pending).
    assert len(flushed_3min) == 24
    for payload in flushed_3min:
        assert payload["n"] == 1

    # Hourly tier: samples at t=0..4320 fall in bucket [0,3600); t=3600..4320
    # is a second bucket if the boundary at t=3600 was crossed -> at least
    # one hourly flush by the time we've gone past t=3600.
    assert len(flushed_hourly) >= 1
    assert flushed_hourly[0]["t"] == 0.0
    # First hourly bucket [0, 3600) contains samples t=0,180,...,3420 -> 20 samples.
    assert flushed_hourly[0]["v"]["k"]["count"] == 20
