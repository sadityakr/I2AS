"""A driver spy for proving a code path issues no instrument traffic.

Several standards in I2AS are "this path must not touch the bus": a VI's
``__init__`` (the connection-lifecycle standard's rule 1, construction is
silent), ``evaluate_safety()`` (it judges the tick snapshot, it does not
poll), and ``control_param_specs()`` (the purity rule in
``BaseVirtualInstrument``, which is what lets ``Station.station_info()``
describe an instrument without operating it).

The way to check such a rule is to WATCH the drivers rather than to trust
the code, so this module installs a recording shim over every public
callable of a live driver instance. The shim delegates to the real method,
so the object keeps working exactly as before and a test that spies too
early still runs; the point is the record it leaves, which a test asserts
is empty.

Shims are installed as INSTANCE attributes shadowing the class methods, so
every holder of that same driver object sees them — including the
``self._driver`` reference a VI captured in its own ``__init__``, which is
what makes this usable against a fully built Station.
"""

from __future__ import annotations

from typing import Any, Callable


def spy_on_driver(driver: Any, calls: list[str]) -> None:
    """Record every public method call made on one driver instance.

    Args:
        driver: A live driver object. Mutated in place: each public callable
            is shadowed by a recording shim that delegates to it.
        calls: The shared log to append ``"DriverClass.method"`` to, one
            entry per call. Assert it is empty to prove a path is pure.
    """
    for name in dir(driver):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(driver, name)
        except Exception:  # noqa: BLE001 — a property that raises is not a method
            continue
        if not callable(attr):
            continue
        setattr(driver, name, _recorder(type(driver).__name__, name, attr, calls))


def spy_on_station(station: Any, calls: list[str]) -> None:
    """Record every public driver call reachable from a built Station.

    Covers both the Station's own alias map (``_drivers``) and each live
    VI's injected ``drivers`` mapping, since a VI whose driver failed to
    connect is not in the former, and a test Station assembled without a
    config has no former at all.

    Args:
        station: A built ``Station``.
        calls: The shared log, as for :func:`spy_on_driver`.
    """
    seen: set[int] = set()
    candidates = list(getattr(station, "_drivers", {}).values())
    for vi_name in station.get_vi_names():
        candidates.extend(getattr(station.get_vi(vi_name), "_drivers", {}).values())
    for driver in candidates:
        if id(driver) in seen:
            continue
        seen.add(id(driver))
        spy_on_driver(driver, calls)


def _recorder(
    owner: str, name: str, bound: Callable, calls: list[str]
) -> Callable:
    """Return a shim that logs one call and then delegates to *bound*.

    Args:
        owner: The driver class's name, for the log entry.
        name: The method's name, for the log entry.
        bound: The real bound method to delegate to.
        calls: The shared log to append to.

    Returns:
        The recording shim.
    """

    def shim(*args: Any, **kwargs: Any) -> Any:
        calls.append(f"{owner}.{name}")
        return bound(*args, **kwargs)

    return shim
