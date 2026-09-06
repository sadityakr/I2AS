"""Offscreen visual smoke test for the I2AS GUI.

Builds the real application (sim_cryostat config, real stylesheet) without a
display, drives it into its visually distinct states (idle, stale panel,
repeated error banner, EMERGENCY status bar), and saves window screenshots for
human inspection. Part of the gui-edit skill's mandatory verification: tests
assert properties, only pixels catch wrong-looking output.

Input:
    No mandatory arguments. Must run with the project venv python from the
    repo root (imports the i2as package). --out overrides the output dir;
    --size WxH (e.g. "1920x1080") resizes both windows before capture and is
    applied on top of whatever geometry MonitorWindow/ProcedureWindow chose
    themselves (default: leave their natural size alone).

Process:
    Constructs Station/Orchestrator/MonitorWindow/ProcedureWindow on the Qt
    "offscreen" platform, lets the Orchestrator's real tick loop run for at
    least two ticks (via QTest.qWait) so instrument values and trend curves
    show real simulated data before the idle screenshot, then emits
    orchestrator signals to reach each error/stale/emergency state, grabs
    window pixmaps, and samples effective (post-QSS) label colors.

Output:
    monitor_idle.png, monitor_error.png, procedure.png in
    tmp/gui-edit/<YYYY-MM-DD_HH-MM-SS>/ (or --out), plus a stdout summary of
    effective banner/status-bar colors and banner state.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "PyQt6>=6.5",
#   "pyqtgraph>=0.13",
# ]
# ///

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.getcwd())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: tmp/gui-edit/<timestamp>/)",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help='Window size as "WxH" (e.g. "1920x1080"), applied to both windows before capture.',
    )
    args = parser.parse_args()

    size = None
    if args.size:
        w_str, _, h_str = args.size.lower().partition("x")
        size = (int(w_str), int(h_str))

    out_dir = args.out or (
        Path("tmp") / "gui-edit" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    import pyqtgraph as pg
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QPushButton

    from i2as.core.orchestrator import Orchestrator
    from i2as.core.station import build_station
    from i2as.gui.monitor_window import MonitorWindow
    from i2as.gui.procedure_window import ProcedureWindow
    from i2as.gui.theme import PLOT_AXIS, PLOT_BG, build_stylesheet

    app = QApplication(sys.argv)
    app.setStyleSheet(build_stylesheet())
    pg.setConfigOptions(background=PLOT_BG, foreground=PLOT_AXIS, antialias=True)

    tick_ms = 200
    station = build_station("i2as/configs/sim_cryostat")
    orchestrator = Orchestrator(station, tick_interval_ms=tick_ms)

    monitor = MonitorWindow(station, orchestrator)
    if size is not None:
        monitor.resize(*size)
    else:
        monitor.resize(1280, 940)
    monitor.show()
    app.processEvents()

    # Monitoring is OFF at launch (nothing is polled until the instruments
    # are initiated); start it through the real GUI path — the header toggle.
    monitor.findChild(QPushButton, "monitoring_btn").click()

    # Let the orchestrator's real tick loop run for at least two ticks so
    # instrument values and trend curves show real simulated data, not "—".
    QTest.qWait(tick_ms * 2 + 100)
    app.processEvents()
    monitor.grab().save(str(out_dir / "monitor_idle.png"))

    # Drive: one stale panel, repeated errors (banner counter), EMERGENCY state.
    vi_names = station.get_vi_names()
    orchestrator.states_updated.emit({vi_names[0]: {"_stale": True}})
    for _ in range(3):
        orchestrator.error_occurred.emit("Smoke-test simulated communication error")
    orchestrator.state_changed.emit("EMERGENCY")
    app.processEvents()
    monitor.grab().save(str(out_dir / "monitor_error.png"))

    # Effective (post-QSS) colors — what the user actually sees, not the
    # property values. These catch the parent-only-repolish class of bug.
    banner_label = monitor._banner._label
    print("banner label effective color:", banner_label.palette().windowText().color().name())
    print("banner visible:", monitor._banner.isVisible(), "count:", monitor._banner.count)
    print("status bar level property:", monitor._status_bar.property("level"))
    print(
        "statusbar label effective color:",
        monitor._state_label.palette().windowText().color().name(),
    )

    proc = ProcedureWindow(
        station,
        orchestrator,
        get_sample_info=monitor.get_sample_info,
        get_data_dir=monitor.get_data_dir,
    )
    if size is not None:
        proc.resize(*size)
    else:
        proc.resize(1280, 900)
    proc.show()
    app.processEvents()
    proc.grab().save(str(out_dir / "procedure.png"))

    print("screenshots written to", out_dir.resolve())


if __name__ == "__main__":
    main()
