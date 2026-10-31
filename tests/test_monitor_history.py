from i2as.gui.monitor_history import MonitorHistory


def test_flattening_matches_last_state_flat_convention():
    """Nested two-VI state flattens to {vi}_{field} keys; bools/strings/_-fields excluded."""
    history = MonitorHistory()
    state = {
        "magnet_z": {
            "get_field": 0.5,
            "magnet_current": 12.3,
            "ramp_status": "ramping",  # string -> excluded
            "_stale": True,  # leading underscore -> excluded
        },
        "temperature": {
            "temperature_K": 4.2,
            "heater_on": True,  # bool -> excluded even though bool is an int subclass
            "_disconnected": False,  # leading underscore -> excluded
        },
    }
    history.record(state, timestamp=100.0)

    assert history.keys() == sorted(
        ["magnet_z_get_field", "magnet_z_magnet_current", "temperature_temperature_K"]
    )

    # Hand-computed expectation matching Station.last_state_flat()'s output shape
    # for this same state (excluding the measurement-VI filter, which is a
    # Station-level concern, not a MonitorHistory concern).
    expected_flat = {
        "magnet_z_get_field": 0.5,
        "magnet_z_magnet_current": 12.3,
        "temperature_temperature_K": 4.2,
    }
    for key, value in expected_flat.items():
        times, values = history.series(key)
        assert times == [100.0]
        assert values == [value]


def test_ring_buffer_caps_length_and_keeps_newest():
    """Recording more points than maxlen keeps only the newest; length never exceeds maxlen."""
    retention_s = 10.0
    tick_interval_s = 1.0
    history = MonitorHistory(retention_s=retention_s, tick_interval_s=tick_interval_s)
    maxlen = int(retention_s / tick_interval_s) + 8  # matches MonitorHistory's internal slack

    n_points = maxlen + 20
    for i in range(n_points):
        history.record({"vi": {"field": float(i)}}, timestamp=float(i))

    times, values = history.series("vi_field")
    assert len(times) == maxlen
    assert len(values) == maxlen
    # The oldest points were evicted; only the newest maxlen remain.
    expected_first_value = float(n_points - maxlen)
    assert values[0] == expected_first_value
    assert values[-1] == float(n_points - 1)


def test_window_filtering_returns_chronological_subset():
    """series(key, window_s, now) returns exactly the points within the window, in order."""
    history = MonitorHistory()
    for t in [0.0, 5.0, 10.0, 15.0, 20.0]:
        history.record({"vi": {"field": t}}, timestamp=t)

    times, values = history.series("vi_field", window_s=10.0, now=20.0)
    # Points with t >= now - window_s = 10.0
    assert times == [10.0, 15.0, 20.0]
    assert values == [10.0, 15.0, 20.0]


def test_window_none_returns_everything_and_unknown_key_returns_empty():
    """window_s=None returns full history; an unknown key returns empty lists."""
    history = MonitorHistory()
    for t in [0.0, 1.0, 2.0]:
        history.record({"vi": {"field": t}}, timestamp=t)

    times, values = history.series("vi_field", window_s=None)
    assert times == [0.0, 1.0, 2.0]
    assert values == [0.0, 1.0, 2.0]

    times, values = history.series("does_not_exist")
    assert times == []
    assert values == []


def test_key_appearing_only_in_later_records_still_works():
    """A flat key first seen mid-run starts its own history from that point on."""
    history = MonitorHistory()
    history.record({"vi_a": {"field": 1.0}}, timestamp=0.0)
    assert "vi_b_field" not in history.keys()

    history.record({"vi_a": {"field": 2.0}, "vi_b": {"field": 99.0}}, timestamp=1.0)

    assert "vi_b_field" in history.keys()
    times, values = history.series("vi_b_field")
    assert times == [1.0]
    assert values == [99.0]

    times, values = history.series("vi_a_field")
    assert times == [0.0, 1.0]
    assert values == [1.0, 2.0]


def test_record_flat_appends_and_is_retrievable():
    """record_flat() appends already-flat readings, retrievable via series()/keys()."""
    history = MonitorHistory()
    history.record_flat({"magnet_z_get_field": 0.5, "temperature_temperature_K": 4.2}, timestamp=100.0)

    assert history.keys() == sorted(["magnet_z_get_field", "temperature_temperature_K"])
    times, values = history.series("magnet_z_get_field")
    assert times == [100.0]
    assert values == [0.5]
    times, values = history.series("temperature_temperature_K")
    assert times == [100.0]
    assert values == [4.2]


def test_record_flat_and_record_interleave_in_timestamp_order():
    """record() and record_flat() append into the same buffer for a shared key, in order."""
    history = MonitorHistory()
    history.record({"magnet_z": {"get_field": 1.0}}, timestamp=0.0)
    history.record_flat({"magnet_z_get_field": 2.0}, timestamp=1.0)
    history.record({"magnet_z": {"get_field": 3.0}}, timestamp=2.0)
    history.record_flat({"magnet_z_get_field": 4.0}, timestamp=3.0)

    times, values = history.series("magnet_z_get_field")
    assert times == [0.0, 1.0, 2.0, 3.0]
    assert values == [1.0, 2.0, 3.0, 4.0]


def test_record_flat_timestamp_none_uses_now(monkeypatch):
    """record_flat(timestamp=None) stamps points with time.time(), matching record()'s convention."""
    import i2as.gui.monitor_history as monitor_history_module

    monkeypatch.setattr(monitor_history_module.time, "time", lambda: 12345.0)

    history = MonitorHistory()
    history.record_flat({"vi_field": 1.0})

    times, values = history.series("vi_field")
    assert times == [12345.0]
    assert values == [1.0]


def test_record_flat_skips_non_numeric_and_underscore_keys():
    """record_flat() skips non-numeric values and keys starting with '_', like record()."""
    history = MonitorHistory()
    history.record_flat(
        {
            "magnet_z_get_field": 0.5,
            "magnet_z_ramp_status": "ramping",  # non-numeric -> excluded
            "_stale": 1.0,  # leading underscore -> excluded
        },
        timestamp=100.0,
    )

    assert history.keys() == ["magnet_z_get_field"]
    times, values = history.series("magnet_z_get_field")
    assert times == [100.0]
    assert values == [0.5]


def test_record_flat_respects_ring_buffer_maxlen():
    """record_flat() is subject to the same maxlen eviction as record()."""
    retention_s = 10.0
    tick_interval_s = 1.0
    history = MonitorHistory(retention_s=retention_s, tick_interval_s=tick_interval_s)
    maxlen = int(retention_s / tick_interval_s) + 8

    n_points = maxlen + 20
    for i in range(n_points):
        history.record_flat({"vi_field": float(i)}, timestamp=float(i))

    times, values = history.series("vi_field")
    assert len(times) == maxlen
    assert len(values) == maxlen
    expected_first_value = float(n_points - maxlen)
    assert values[0] == expected_first_value
    assert values[-1] == float(n_points - 1)


def test_record_flat_matches_record_bool_handling():
    """record_flat() excludes bool values identically to record() (bool is an int subclass)."""
    history = MonitorHistory()
    history.record({"vi": {"flag": True}}, timestamp=0.0)
    history.record_flat({"vi_flag": True}, timestamp=1.0)

    # Both entry points exclude bool: the key is never created at all.
    assert "vi_flag" not in history.keys()
    times, values = history.series("vi_flag")
    assert times == []
    assert values == []


def test_windowed_series_matches_a_full_scan_and_reads_only_the_window() -> None:
    """series(window_s=...) is the suffix of the history a full filter would return."""
    from i2as.gui.monitor_history import MonitorHistory

    history = MonitorHistory(retention_s=3600.0, tick_interval_s=0.25)
    points = [(1000.0 + 0.25 * i, float(i)) for i in range(4000)]
    for t, v in points:
        history._history.setdefault("vi_x", __import__("collections").deque(maxlen=history._maxlen)).append((t, v))
    now = points[-1][0]
    times, values = history.series("vi_x", window_s=60.0, now=now)
    expected = [(t, v) for t, v in points if t >= now - 60.0]
    assert list(zip(times, values)) == expected
    assert len(times) == 241


def test_trend_panels_redraw_at_most_once_per_interval(qtbot, monkeypatch) -> None:
    """A fast tick records every state but repaints the trends at most once per second."""
    import i2as.gui.trends_quadrant as tq

    clock = {"t": 100.0}
    monkeypatch.setattr(tq.time, "monotonic", lambda: clock["t"])
    quadrant = tq.TrendsQuadrant.__new__(tq.TrendsQuadrant)
    refreshed = []

    class _Panel:
        def refresh(self):
            refreshed.append(clock["t"])

        def selected_key(self):
            return None

    class _History:
        def __init__(self):
            self.recorded = 0

        def record(self, _state):
            self.recorded += 1

    quadrant._history = _History()
    quadrant._trend_panels = {"trend_0": _Panel()}
    quadrant._pending_trend_keys = {}
    quadrant._default_trend_key_hints = {}
    quadrant._last_trend_refresh = float("-inf")
    for _ in range(12):  # 3 s of 250 ms ticks
        quadrant.on_states_updated({"vi": {"x": 1.0}})
        clock["t"] += 0.25
    assert quadrant._history.recorded == 12
    assert len(refreshed) == 3
