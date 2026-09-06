"""MonitorHistory — ring-buffer time-series store for the Monitor window's trend plots.

Qt-free by design: this module imports nothing from PyQt6, the Orchestrator,
Virtual Instruments, or drivers, so it cannot violate any layer-boundary
import-linter contract. It is intended to be fed from the Orchestrator's
``states_updated`` signal (emitted roughly every 3 seconds), but the wiring
of that signal is out of scope here — this module only stores and serves
the accumulated history.
"""

from __future__ import annotations

import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


class MonitorHistory:
    """Bounded time-series history of flattened instrument readings.

    Flattens nested station state dicts (``{vi_name: {field_name: value}}``)
    into flat keys (``{vi_name}_{field_name}``) exactly as
    ``Station.last_state_flat()`` does, and stores each key's history in its
    own ring buffer so memory use stays bounded regardless of how long the
    Monitor window has been open.

    Two entry points feed the same buffers, and they are deliberately
    asymmetric:

    - ``record()`` is the live path, fed from ``Orchestrator.states_updated``
      roughly every tick. It takes the nested station-state shape and
      flattens it itself, and it includes measurement VIs.
    - ``record_flat()`` is the replay path, used at startup to rehydrate the
      buffer from disk-persisted trend history (``TrendsQuadrant`` reading
      the tiered trend store). It takes an already-flat dict and does not
      re-derive it, and it does **not** include measurement VIs, because the
      canonical flattener behind the disk tiers,
      ``Station.last_state_flat()`` (``i2as/core/station.py:535-536``),
      excludes them by design. A trend panel on a measurement-VI key
      therefore works live but goes empty after a restart or on a
      disk-backed window; this is an accepted asymmetry, not a bug in
      either entry point.

    There is a second, opposite-running asymmetry between the same two
    paths, currently accepted but NOT deliberately designed the way the
    one above is: ``Station.last_state_flat()`` has no ``bool`` guard — it
    keeps anything passing ``isinstance(value, (int, float))``, and
    ``bool`` is an ``int`` subclass, so a boolean ``@monitored`` field is
    coerced to ``1.0``/``0.0`` and reaches the disk tiers. ``record()`` here
    explicitly skips ``bool`` values, so the same field never enters the
    live in-RAM history. Net effect: that key *gains* a trend-plot entry
    after a restart or on a disk-backed window, the opposite direction of
    the measurement-VI asymmetry above (which *loses* one). Neither
    ``last_state_flat()`` nor ``record()`` should be changed to fix this
    without an explicit decision, since ``last_state_flat()`` also feeds
    HDF5 sweep columns and changing its filtering would alter data files.

    Attributes:
        retention_s: How many seconds of history each key retains.
        tick_interval_s: Expected seconds between successive ``record()``
            calls, used only to size the ring buffer.
    """

    def __init__(self, retention_s: float = 86400.0, tick_interval_s: float = 3.0) -> None:
        """Initialise an empty history store.

        Args:
            retention_s: How many seconds of history to retain per key.
                Defaults to 86400 s (24 hours).
            tick_interval_s: Expected interval in seconds between ``record()``
                calls. Used only to size each key's ring buffer; a mismatch
                between this estimate and the actual call rate just means
                slightly more or less history than ``retention_s`` is kept.
        """
        self.retention_s = retention_s
        self.tick_interval_s = tick_interval_s
        # A small amount of slack above the exact point count absorbs jitter
        # in the actual tick rate without under-retaining.
        self._maxlen = int(retention_s / tick_interval_s) + 8
        # One ring buffer per flat key. A deque with maxlen is a ring buffer:
        # once it is full, appending a new point silently drops the oldest
        # one, so memory stays bounded with no manual cleanup/eviction code.
        self._history: dict[str, deque[tuple[float, float]]] = {}

    def record(self, state: dict[str, dict[str, object]], timestamp: float | None = None) -> None:
        """Flatten a station state snapshot and append it to each key's history.

        Mirrors ``Station.last_state_flat()``'s flattening convention: flat
        key is ``f"{vi_name}_{field_name}"``, only numeric scalar values
        (``int``/``float``, excluding ``bool``) are kept, and any field name
        starting with ``"_"`` (e.g. ``_stale``, ``_disconnected``) is skipped.
        A flat key seen for the first time simply starts a new ring buffer.

        Args:
            state: Nested state dict, ``{vi_name: {field_name: value, ...}}``,
                as produced by ``Station.get_state()`` / ``Orchestrator.states_updated``.
            timestamp: Unix timestamp to record the point under. Defaults to
                ``time.time()`` when ``None``; the explicit parameter exists
                so tests can inject deterministic times.
        """
        if timestamp is None:
            timestamp = time.time()

        for vi_name, fields in state.items():
            for field_name, value in fields.items():
                if field_name.startswith("_"):
                    continue
                key = f"{vi_name}_{field_name}"
                self._append_if_numeric(key, value, timestamp)

    def record_flat(self, flat: dict[str, float], timestamp: float | None = None) -> None:
        """Append an already-flattened reading dict to each key's history.

        For replaying disk-persisted trend history (e.g. at ``TrendsQuadrant``
        startup) through the same ring buffers ``record()`` feeds live. Applies
        the identical value-filtering discipline as ``record()`` (numeric
        scalars only, ``bool`` excluded, keys starting with ``"_"`` skipped),
        but does not re-derive the flat keys: ``flat`` is expected to already
        be in ``{vi_name}_{field_name}`` form, as produced by
        ``Station.last_state_flat()``. That source excludes measurement VIs,
        so this path never records them; see the class docstring for the
        accepted asymmetry with ``record()``.

        Args:
            flat: Already-flattened reading dict, ``{flat_key: value}``.
            timestamp: Unix timestamp to record the points under. Defaults to
                ``time.time()`` when ``None``, identically to ``record()``.
        """
        if timestamp is None:
            timestamp = time.time()

        for key, value in flat.items():
            if key.startswith("_"):
                continue
            self._append_if_numeric(key, value, timestamp)

    def _append_if_numeric(self, key: str, value: object, timestamp: float) -> None:
        """Append ``(timestamp, value)`` to ``key``'s ring buffer if numeric.

        Shared by ``record()`` and ``record_flat()`` so both entry points
        apply identical filtering and eviction behaviour. Skips ``bool``
        values explicitly, since ``bool`` is a subclass of ``int`` and would
        otherwise pass the numeric check. Creates the ring buffer on first
        sight of ``key``.

        Args:
            key: Flat key (``{vi_name}_{field_name}``).
            value: Candidate value; only ``int``/``float`` (excluding
                ``bool``) is recorded.
            timestamp: Unix timestamp to record the point under.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        if key not in self._history:
            self._history[key] = deque(maxlen=self._maxlen)
        self._history[key].append((timestamp, float(value)))

    def keys(self) -> list[str]:
        """Return all flat keys seen so far.

        Returns:
            Sorted list of flat keys (``{vi_name}_{field_name}``).
        """
        return sorted(self._history.keys())

    def series(
        self, key: str, window_s: float | None = None, now: float | None = None
    ) -> tuple[list[float], list[float]]:
        """Return the recorded time series for a flat key.

        Args:
            key: Flat key as produced by ``record()``
                (e.g. ``"temperature_setpoint"``).
            window_s: If given, only include points with ``t >= now - window_s``.
                ``None`` (the default) returns the full retained history.
            now: Reference time for windowing. Defaults to ``time.time()``
                when ``None``; ignored if ``window_s`` is ``None``.

        Returns:
            A ``(times, values)`` tuple of two parallel lists in chronological
            order. Returns ``([], [])`` for a key that has never been recorded.
        """
        points = self._history.get(key)
        if points is None:
            return [], []

        if window_s is None:
            times = [t for t, _ in points]
            values = [v for _, v in points]
            return times, values

        if now is None:
            now = time.time()
        cutoff = now - window_s
        times = [t for t, v in points if t >= cutoff]
        values = [v for t, v in points if t >= cutoff]
        return times, values
