"""Soak the instrument thread under a click storm and report the worst GUI stall.

Input: a config directory (default ``sim_cryostat``, whose tick and
slow simulated instruments are the closest thing to a real rig), a duration,
and a mode (``threaded``, ``inline``, or ``both`` to run one after the other
and print them side by side).

Process: builds the real application windows against a real
``InstrumentHost``, then runs two timers on the GUI thread. A **heartbeat**
fires every 20 ms and records the gap since its last firing — anything much
above 20 ms is the GUI thread being held by something. A **click storm** fires
several times a second and does what an impatient operator does: toggles
monitoring, initiates and standbys every instrument, queues a run, reorders
it, asks for a ping. Every one of those is a command that
crosses to the engine.

Output: on stdout, the heartbeat's worst and 99th-percentile stall, how many
clicks were delivered, and the verdict — a threaded run should keep its worst
stall in the tens of milliseconds while an inline one parks on the tick's own
instrument reads.

Usage (from the repo root, with the project venv, and with a measurement root
configured because the windows read one):

    I2AS_MEASUREMENT_ROOT=/tmp/soak QT_QPA_PLATFORM=offscreen \\
        python scripts/soak_instrument_thread.py --minutes 3

Not a test: nothing here asserts. It is the long-running counterpart to
``tests/test_instrument_thread.py``'s frozen-GUI detector, which proves the
same property in two seconds and cannot see a stall that only shows up after
a thousand ticks.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from i2as.core.instrument_host import InstrumentHost
from i2as.core.station import build_station
from i2as.gui.monitor_window import MonitorWindow
from i2as.gui.procedure_window import ProcedureWindow

DEFAULT_CONFIG = "i2as/configs/sim_cryostat"

#: How often the heartbeat fires. Small enough that a stall of a tenth of a
#: second is unmistakable, large enough not to be the load itself.
HEARTBEAT_MS = 20

#: How often the click storm acts.
CLICK_MS = 250


@dataclass
class SoakResult:
    """What one soak run measured.

    Attributes:
        mode: The instrument mode the run used.
        seconds: How long it ran.
        gaps: Every heartbeat gap, in seconds.
        clicks: How many click-storm actions were delivered.
    """

    mode: str
    seconds: float
    gaps: list[float] = field(default_factory=list)
    clicks: int = 0

    @property
    def worst_ms(self) -> float:
        """The longest the GUI thread went without servicing its timer."""
        return max(self.gaps) * 1000 if self.gaps else 0.0

    @property
    def p99_ms(self) -> float:
        """The 99th-percentile heartbeat gap, in milliseconds."""
        if not self.gaps:
            return 0.0
        ordered = sorted(self.gaps)
        return ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))] * 1000

    def report(self) -> str:
        """Return the one-paragraph human summary of this run.

        Returns:
            A multi-line string.
        """
        return (
            f"mode={self.mode}  ran {self.seconds:.0f} s\n"
            f"  heartbeats     {len(self.gaps)} "
            f"(expected ~{int(self.seconds * 1000 / HEARTBEAT_MS)})\n"
            f"  worst stall    {self.worst_ms:.0f} ms\n"
            f"  p99 stall      {self.p99_ms:.0f} ms\n"
            f"  median gap     "
            f"{statistics.median(self.gaps) * 1000 if self.gaps else 0:.0f} ms\n"
            f"  clicks         {self.clicks}"
        )


class ClickStorm:
    """The impatient operator: a rotation of real GUI actions, on a timer.

    Every action is one a human can perform from the two windows, so what it
    exercises is the command path a click takes, not a private API.

    Args:
        monitor: The Monitor window.
        procedure: The Procedure window.
        client: The orchestrator proxy both windows hold.
    """

    def __init__(self, monitor: MonitorWindow, procedure: ProcedureWindow, client) -> None:
        self._monitor = monitor
        self._procedure = procedure
        self._client = client
        self._step = 0
        self.delivered = 0

    def act(self) -> None:
        """Perform the next action in the rotation."""
        actions = (
            lambda: self._monitor._monitoring_btn.click(),
            lambda: self._client.submit_global_action("initiate_all"),
            lambda: self._procedure._on_add_to_queue(),
            lambda: self._client.ping_instrument(
                self._client.station_info().instruments[0].name
            ),
            lambda: self._procedure._queue_panel._queue_list.setCurrentRow(0),
            lambda: self._procedure._queue_panel._queue_move_down(),
            lambda: self._client.submit_global_action("standby_all"),
        )
        actions[self._step % len(actions)]()
        self._step += 1
        self.delivered += 1


def soak(mode: str, config_path: str, seconds: float) -> SoakResult:
    """Run one soak and return what it measured.

    Args:
        mode: ``"inline"`` or ``"threaded"``.
        config_path: The config directory to build the Station from.
        seconds: How long to run.

    Returns:
        The measurements.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    host = InstrumentHost(lambda: build_station(config_path), mode=mode)
    host.start()
    client = host.build_proxy()

    monitor = MonitorWindow(host.station, client, mirror=client.status)
    monitor.show()
    procedure = ProcedureWindow(
        host.station,
        client,
        get_sample_info=lambda: {
            "sample_name": "soak",
            "sample_id": "S1",
            "comments": "",
        },
        get_data_dir=lambda: ".",
        mirror=client.status,
    )
    procedure.show()

    result = SoakResult(mode=mode, seconds=seconds)
    storm = ClickStorm(monitor, procedure, client)
    last = [time.monotonic()]

    def _beat() -> None:
        now = time.monotonic()
        result.gaps.append(now - last[0])
        last[0] = now

    heartbeat = QTimer()
    heartbeat.setInterval(HEARTBEAT_MS)
    heartbeat.timeout.connect(_beat)

    clicker = QTimer()
    clicker.setInterval(CLICK_MS)
    clicker.timeout.connect(storm.act)

    heartbeat.start()
    clicker.start()
    QTimer.singleShot(int(seconds * 1000), app.quit)
    app.exec()
    heartbeat.stop()
    clicker.stop()

    result.clicks = storm.delivered
    monitor.close()
    procedure.close()
    host.shutdown()
    return result


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the soak(s), print the report.

    Args:
        argv: Command-line arguments, or ``None`` for ``sys.argv``.

    Returns:
        Process exit status; always 0 — this measures, it does not judge.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="config directory")
    parser.add_argument(
        "--minutes", type=float, default=3.0, help="how long to soak each mode"
    )
    parser.add_argument(
        "--mode",
        choices=("threaded", "inline", "both"),
        default="threaded",
        help="which instrument mode(s) to soak",
    )
    args = parser.parse_args(argv)

    modes = ("inline", "threaded") if args.mode == "both" else (args.mode,)
    for mode in modes:
        result = soak(mode, args.config, args.minutes * 60)
        print(result.report())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
