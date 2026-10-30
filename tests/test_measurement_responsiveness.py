"""Regression coverage for acquisition, queue and display responsiveness."""
import threading
import time
from types import SimpleNamespace
import weakref

import numpy as np
from PyQt6.QtCore import QCoreApplication

from i2as.core.instrument_host import ThreadBridge, _Courier
from i2as.gui.procedure_window import ProcedureWindow
from i2as.gui.monitor_history import MonitorHistory
from i2as.gui.trend_plot_panel import TrendPlotPanel


def test_timed_out_pop_is_not_executed_when_ui_resumes(qapp):
    bridge = ThreadBridge(_Courier(), _Courier(), callback_timeout_s=.02)
    queue = ["next-run"]
    answers = []
    thread = threading.Thread(target=lambda: answers.append(bridge.ask(queue.pop)))
    thread.start()
    thread.join(1)
    assert not thread.is_alive()
    assert answers == [None]
    QCoreApplication.processEvents()
    assert queue == ["next-run"]


def test_plot_history_does_not_retain_images_and_is_bounded():
    plot = SimpleNamespace(available_keys=frozenset({"value"}), redraw=lambda data: None)
    receiver = SimpleNamespace(_datapoints=[], _plot1=plot, _plot2=plot)
    frame = np.ones((128, 128))
    reference = weakref.ref(frame)
    ProcedureWindow._on_measurement_ready(receiver, {"value": 1., "frame": frame})
    del frame
    assert reference() is None
    for i in range(10001):
        ProcedureWindow._on_measurement_ready(receiver, {"value": i})
    assert len(receiver._datapoints) == 10000
    assert receiver._datapoints[-1]["value"] == 10000


def test_running_pop_is_restored_after_timeout(qapp):
    bridge = ThreadBridge(_Courier(), _Courier(), callback_timeout_s=.02)
    queue = ["next-run"]
    answers = []

    def pop():
        spec = queue.pop()
        time.sleep(.08)
        return spec

    thread = threading.Thread(target=lambda: answers.append(
        bridge.ask(pop, on_abandoned=queue.append)))
    thread.start()
    time.sleep(.005)
    QCoreApplication.processEvents()
    thread.join(1)
    assert not thread.is_alive()
    assert answers == [None]
    assert queue == ["next-run"]


def test_slow_history_read_is_shared_and_does_not_block_ui(qtbot, monkeypatch, tmp_path):
    from i2as.core import trend_history
    entered, release = threading.Event(), threading.Event()
    calls = []

    def read(*args):
        calls.append(args)
        entered.set()
        assert release.wait(3)
        return {"sensor_value": [(1., 2.)]}

    monkeypatch.setattr(trend_history, "read_window", read)
    history = MonitorHistory()
    history.record_flat({"sensor_value": 2.})
    panels = [TrendPlotPanel(history, str(i), log_dir=tmp_path) for i in range(2)]
    try:
        for panel in panels:
            qtbot.addWidget(panel)
            panel.refresh()
            panel.set_selected_window_s(31536000.)
        qtbot.waitUntil(entered.is_set)
        qtbot.wait(30)
        assert len(calls) == 1
        assert all(panel._disk_future is not None for panel in panels)
    finally:
        release.set()
    qtbot.waitUntil(lambda: all(panel._disk_future is None for panel in panels))
    assert list(panels[0]._curve.getData()[1]) == [2.]


def test_dc_reading_yields_between_samples():
    from i2as.core.station import build_station
    station = build_station("i2as/configs/sim_cryostat")
    vi = station.get_vi("dc_measurement")
    vi.initiate_measurement(current_A=1e-6, compliance_A=1e-3,
                              voltmeter_range_V=.1, readings_per_point=4)
    steps = vi.take_reading_steps()
    for _ in range(4):
        assert next(steps) is None
    try:
        next(steps)
    except StopIteration as done:
        assert len(done.value["voltage_V_array"]) == 4
    else:
        raise AssertionError("reading did not finish")
    vi.standby()
