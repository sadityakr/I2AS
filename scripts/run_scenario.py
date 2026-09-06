"""Launch the real I2AS GUI pre-seeded into a hazardous/degraded state.

Dev tool for exploring "what does the app actually do in state X" live,
instead of only asserting it in ``tests/test_scenarios.py``. Reuses the
exact same driver-level state-injection primitives that module and the rest
of the test suite already use (``tests.scenarios``' ``apply_*`` functions),
wired into ``i2as.main.main()``'s ``on_station_built`` hook — so this is
the same production ``main()``, same real Orchestrator tick loop, only the
sim drivers' test-control attributes are set before the window shows.
Requires a display (not for CI/offscreen use).

Input: a scenario name plus its parameters on the command line (see
--help). No config beyond the shipped ``sim_cryostat`` fallback station
``i2as.main`` already resolves at startup.
Process: builds and applies the requested scenario's driver flags the
moment the Station is built, before the window is shown; scenarios that
need several ticks to converge (quench's
safety-flag propagation) simply appear once the operator starts monitoring
in the running app, exactly as they would on real hardware.
Output: the I2AS window, running live in the requested state, until the
operator closes it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from i2as.core.station import Station  # noqa: E402
from i2as.main import main  # noqa: E402
from tests import scenarios as scn  # noqa: E402

_SCENARIOS = ("quench", "disconnect", "measurement-error")


def _build_apply(args: argparse.Namespace):
    """Return a Station -> None closure applying the requested scenario."""

    def _apply(station: Station) -> None:
        if args.scenario == "quench":
            scn.apply_quench(station, magnet_vi=args.vi or "magnet_z")
            print(f"[scenario] quench: {args.vi or 'magnet_z'} will report QUENCH")
        elif args.scenario == "disconnect":
            vi_name = args.vi or "temperature"
            scn.apply_disconnect(station, vi_name, driver_attr=args.driver_attr or "_driver")
            print(f"[scenario] disconnect: {vi_name} will fail every call")
        elif args.scenario == "measurement-error":
            vi_name = args.vi or "dc_measurement"
            scn.measurement_error(station, vi_name, driver_attr=args.driver_attr or "_meter")
            print(f"[scenario] measurement-error: {vi_name} will raise on next use")
        print(
            "[scenario] Start monitoring from the Monitor window's header "
            "toggle to see it take effect (monitoring starts OFF by design)."
        )

    return _apply


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch I2AS with a sim-driver scenario pre-armed, for live "
            "exploration of what the app allows/refuses in that state."
        )
    )
    parser.add_argument("scenario", choices=_SCENARIOS)
    parser.add_argument(
        "--vi",
        default=None,
        help="VI name the scenario targets (defaults per scenario, e.g. "
        "magnet_z / temperature / dc_measurement).",
    )
    parser.add_argument(
        "--driver-attr",
        default=None,
        help="disconnect/measurement-error: driver attribute to fault. "
        "Defaults to _driver for disconnect, _meter for measurement-error "
        "(the standard single- vs. dual-driver VI shapes); pass _source to "
        "fault a dual-driver VI's current source instead.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # The scenario owns this command line; the app gets none of it (and so
    # opens the saved active config, falling back to sim_cryostat).
    main([], on_station_built=_build_apply(_parse_args()))
