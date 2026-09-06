"""Diagnostic engine: bus scan, config preflight, and driver bench.

Design constraints (these are load-bearing for agent use):

* **Always terminates.** No operation waits for user input; every hardware
  interaction is bounded by a VISA timeout.
* **Machine-readable outcomes.** Every probe returns a ``ProbeResult`` whose
  ``code`` is one of the stable ``FaultCode`` values. The triage skill maps
  codes to physical-world causes, so codes are API: never rename one.
* **Read/write separation.** ``DriverBench.call`` refuses methods that change
  instrument state unless ``allow_write=True``; the CLI exposes the two paths
  as separate subcommands so the permission harness can gate them differently.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
import typing
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from i2as.core import trend_history
from i2as.core.exceptions import I2ASCommunicationError, I2ASConfigError
# _import_class / validate_config_dir are the Station factory's own config
# helpers; the conformance tests already reuse them the same way.
from i2as.core.station import _import_class, validate_config_dir
from i2as.core.trend_checks import CheckResult

logger = logging.getLogger(__name__)

# Raw identify probes use a short timeout: a silent instrument should cost
# 2 s, not the drivers' default 5 s, when sweeping a whole bus.
PROBE_TIMEOUT_MS = 2000

# Method-name prefixes treated as read-only (safe to call without unlock).
READ_ONLY_PREFIXES = ("get_", "read_", "is_", "ping")


class FaultCode(str, Enum):
    """Stable, machine-readable outcome codes for every probe.

    ``str, Enum`` makes each member compare and serialize as its plain string
    value, so ``as_dict()`` output is directly JSON-ready.

    The triage decision tree maps each code to likely physical causes:

    * ``OK`` — instrument present, responding, identity as expected.
    * ``CONFIG_INVALID`` — devices.yaml broken (missing file, bad YAML,
      unimportable class). Software-side fault, fix the config.
    * ``ADDRESS_NOT_ON_BUS`` — the VISA bus does not list this address.
      Points to power, cabling, or the instrument's address switch.
    * ``OPEN_FAILED`` — address exists but the resource would not open.
      Points to a port held by another process or a wrong resource type.
    * ``NO_RESPONSE`` — opened, but the identify query timed out.
      Points to baud/termination mismatch, wrong protocol, or a hung unit.
    * ``WRONG_IDN`` — responded, but the identity does not match the
      config's ``expect_idn``. Points to swapped cables/addresses.
    * ``GARBLED_RESPONSE`` — responded with something empty/unusable.
      Points to termination characters, baud rate, or a driver parsing bug.
    * ``DRIVER_ERROR`` — the driver raised a non-communication Python
      error. Software-side fault in the driver code itself.
    """

    OK = "OK"
    CONFIG_INVALID = "CONFIG_INVALID"
    ADDRESS_NOT_ON_BUS = "ADDRESS_NOT_ON_BUS"
    OPEN_FAILED = "OPEN_FAILED"
    NO_RESPONSE = "NO_RESPONSE"
    WRONG_IDN = "WRONG_IDN"
    GARBLED_RESPONSE = "GARBLED_RESPONSE"
    DRIVER_ERROR = "DRIVER_ERROR"


# @dataclass auto-generates __init__/__repr__/__eq__ from the field list —
# these are pure data records, so that is exactly what we want.
@dataclass
class ProbeResult:
    """Outcome of probing one instrument (or one config driver entry)."""

    alias: str                  # driver name from devices.yaml ("" for raw probes)
    address: str                # VISA resource string
    driver_class: str           # dotted class path ("" for raw probes)
    code: FaultCode
    idn: str = ""               # identify response, if any
    detail: str = ""            # human-readable specifics (exception text, hints)

    @property
    def ok(self) -> bool:
        """True if the probe fully passed."""
        return self.code is FaultCode.OK

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready plain dict (FaultCode becomes its string)."""
        data = asdict(self)
        data["code"] = self.code.value
        return data


@dataclass
class L0BenchResult:
    """Outcome of the L0 bench (idn + one passive getter) for one driver.

    Zero-excitation reads only: no approval needed per the setup-supervisor
    skill's safe-testing ladder. Complements ``check_config()``'s
    address/open/idn preflight with proof that at least one more real
    reading actually parses — the desync/parsing class of bug that a clean
    ``get_idn()`` alone does not catch.
    """

    alias: str
    address: str
    driver_class: str
    idn_ok: bool
    idn: str = ""
    getter: str = ""            # name of the extra getter tried; "" if none found
    getter_ok: bool = False
    getter_value: str = ""      # repr() of the value, or "ExcType: message" on failure
    detail: str = ""            # construction/idn failure text, if idn_ok is False

    @property
    def ok(self) -> bool:
        """True if idn succeeded and (no getter was available, or it succeeded too)."""
        return self.idn_ok and (not self.getter or self.getter_ok)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready plain dict."""
        return asdict(self)


@dataclass
class MethodInfo:
    """One public driver method, as seen by the bench."""

    name: str
    signature: str              # e.g. "(range_v: float) -> None"
    doc: str                    # first line of the docstring
    read_only: bool
    params: dict[str, str] = field(default_factory=dict)  # arg name -> type name

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready plain dict."""
        return asdict(self)


# ── Bus scanning ──────────────────────────────────────────────────────────────


def open_resource_manager() -> Any:
    """Return a real pyvisa ResourceManager.

    Kept as a tiny factory so callers (CLI) have one place that touches the
    real VISA backend, and everything else accepts an injected manager.

    Raises:
        I2ASCommunicationError: If pyvisa or its backend is unavailable.
    """
    try:
        import pyvisa

        return pyvisa.ResourceManager()
    except Exception as exc:  # noqa: BLE001 — backend load can fail many ways
        raise I2ASCommunicationError(
            f"No VISA backend available ({exc}). Install NI-VISA or pyvisa-py.",
            vi_name="troubleshoot",
        ) from exc


def scan_bus(rm: Any) -> list[str]:
    """List all VISA resource addresses the backend can see.

    Serial (ASRL) ports are listed whether or not an instrument is attached,
    so presence in this list is necessary but not sufficient — only a
    successful probe proves an instrument is there.

    Args:
        rm: A pyvisa ResourceManager (or test fake with ``list_resources()``).

    Returns:
        Sorted list of resource strings, e.g. ``["ASRL10::INSTR", ...]``.
    """
    resources = sorted(str(r) for r in rm.list_resources())
    logger.info("Bus scan found %d resources: %s", len(resources), resources)
    return resources


def probe_address(rm: Any, address: str, idn_command: str = "*IDN?") -> ProbeResult:
    """Open a bare VISA resource and send one identify query.

    This is the no-driver probe used on unknown or unconfigured addresses
    (driver development starts here). It opens the resource with a short
    timeout, queries once, and closes.

    Args:
        rm: A pyvisa ResourceManager (or test fake with ``open_resource()``).
        address: VISA resource string to probe.
        idn_command: Identify query to send; ``*IDN?`` for SCPI instruments,
            ``V`` for the pre-SCPI Oxford ISOBUS family.

    Returns:
        A ProbeResult with code OK / OPEN_FAILED / NO_RESPONSE /
        GARBLED_RESPONSE.
    """
    try:
        instr = rm.open_resource(address)
    except Exception as exc:  # noqa: BLE001 — any backend error means "won't open"
        return ProbeResult(
            alias="", address=address, driver_class="",
            code=FaultCode.OPEN_FAILED, detail=str(exc),
        )
    try:
        instr.timeout = PROBE_TIMEOUT_MS
        response = str(instr.query(idn_command)).strip()
    except Exception as exc:  # noqa: BLE001 — timeout/IO errors vary by backend
        return ProbeResult(
            alias="", address=address, driver_class="",
            code=FaultCode.NO_RESPONSE,
            detail=f"no reply to {idn_command!r}: {exc}",
        )
    finally:
        try:
            instr.close()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass

    if not response:
        return ProbeResult(
            alias="", address=address, driver_class="",
            code=FaultCode.GARBLED_RESPONSE,
            detail=f"empty reply to {idn_command!r}",
        )
    return ProbeResult(
        alias="", address=address, driver_class="",
        code=FaultCode.OK, idn=response,
    )


# ── Config preflight ──────────────────────────────────────────────────────────


def _load_real_drivers(config_path: str) -> dict[str, dict]:
    """Return the ``real_drivers`` mapping from a config's devices.yaml.

    Args:
        config_path: Config directory containing devices.yaml.

    Returns:
        ``{alias: {"class": ..., "address": ..., "expect_idn": ...?}}``.

    Raises:
        I2ASConfigError: If the file is missing or not valid YAML.
    """
    from ruamel.yaml import YAML

    devices_file = Path(config_path) / "devices.yaml"
    if not devices_file.exists():
        raise I2ASConfigError(f"devices.yaml not found in {config_path}")
    try:
        with devices_file.open("r", encoding="utf-8") as f:
            devices = dict(YAML().load(f) or {})
    except Exception as exc:  # noqa: BLE001 — any parse failure is a config error
        raise I2ASConfigError(f"devices.yaml is not valid YAML: {exc}") from exc
    return dict(devices.get("real_drivers") or {})


def _is_sim_class(dotted_path: str) -> bool:
    """True if the dotted class path names a simulated driver module (sim_*)."""
    module_path = dotted_path.rsplit(".", 1)[0]
    return module_path.rsplit(".", 1)[-1].startswith("sim_")


def check_config(config_path: str, rm: Any = None) -> list[ProbeResult]:
    """Preflight every driver declared in a config directory.

    For each ``real_drivers`` entry: verify the address is on the bus,
    construct the driver class against it, call ``get_idn()``, and (if the
    entry declares ``expect_idn``) check the response contains that substring
    (case-insensitive). Every failure is classified with a FaultCode.

    Simulated drivers (``sim_*`` modules) skip the bus-presence check but are
    still constructed and identified, so the sim config preflights green.

    Args:
        config_path: Config directory (devices.yaml + monitor.yaml).
        rm: Optional ResourceManager for the bus-presence check. If None,
            presence is not checked and classification relies on open/probe
            behaviour alone.

    Returns:
        One ProbeResult per configured driver, in config order. If the config
        itself is invalid, a single CONFIG_INVALID result is returned.
    """
    errors = validate_config_dir(config_path)
    if errors:
        return [
            ProbeResult(
                alias="", address="", driver_class="",
                code=FaultCode.CONFIG_INVALID, detail="; ".join(errors),
            )
        ]

    drivers = _load_real_drivers(config_path)
    bus: list[str] | None = None
    if rm is not None:
        try:
            bus = [r.upper() for r in scan_bus(rm)]
        except Exception as exc:  # noqa: BLE001 — a dead backend must not kill preflight
            logger.warning("Bus scan unavailable: %s", exc)

    results: list[ProbeResult] = []
    for alias, cfg in drivers.items():
        results.append(_check_one_driver(alias, cfg, bus))
    return results


def _check_one_driver(alias: str, cfg: dict, bus: list[str] | None) -> ProbeResult:
    """Probe one configured driver entry and classify the outcome."""
    class_path = str(cfg["class"])
    address = str(cfg.get("address", ""))
    expect_idn = cfg.get("expect_idn")
    is_sim = _is_sim_class(class_path)

    # Bus presence: only meaningful for real drivers with a scanned bus.
    listed: bool | None = None
    if bus is not None and not is_sim:
        listed = address.upper() in bus

    def _result(code: FaultCode, idn: str = "", detail: str = "") -> ProbeResult:
        return ProbeResult(
            alias=alias, address=address, driver_class=class_path,
            code=code, idn=idn, detail=detail,
        )

    # 1. Construct the driver (this opens the VISA resource for real drivers).
    try:
        driver = _import_class(class_path)(address)
    except I2ASConfigError as exc:
        return _result(FaultCode.CONFIG_INVALID, detail=str(exc))
    except I2ASCommunicationError as exc:
        if listed is False:
            return _result(
                FaultCode.ADDRESS_NOT_ON_BUS,
                detail=f"address absent from bus scan; open also failed: {exc}",
            )
        return _result(FaultCode.OPEN_FAILED, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 — anything else is a driver bug
        return _result(
            FaultCode.DRIVER_ERROR,
            detail=f"constructor raised {type(exc).__name__}: {exc}",
        )

    # 2. Identify.
    try:
        idn = str(driver.get_idn()).strip()
    except I2ASCommunicationError as exc:
        if listed is False:
            return _result(
                FaultCode.ADDRESS_NOT_ON_BUS,
                detail=f"address absent from bus scan; no identify reply: {exc}",
            )
        return _result(FaultCode.NO_RESPONSE, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 — non-communication error = driver bug
        return _result(
            FaultCode.DRIVER_ERROR,
            detail=f"get_idn() raised {type(exc).__name__}: {exc}",
        )
    finally:
        _close_driver(driver)

    if not idn:
        return _result(FaultCode.GARBLED_RESPONSE, detail="get_idn() returned empty")

    # 3. Identity match (optional, substring, case-insensitive).
    if expect_idn and str(expect_idn).lower() not in idn.lower():
        return _result(
            FaultCode.WRONG_IDN, idn=idn,
            detail=f"expected substring {str(expect_idn)!r} not in identity",
        )

    detail = ""
    if listed is False:
        # Everything works but the scan missed it (some adapters do not
        # enumerate) — worth surfacing, not worth failing.
        detail = "responds correctly but was not listed by the bus scan"
    return _result(FaultCode.OK, idn=idn, detail=detail)


def bench_l0(config_path: str) -> list[L0BenchResult]:
    """Run the L0 bench (idn + one passive getter) for every driver in a config.

    Meant to run right after ``check_config()`` is green, as the automated
    half of the commissioning skill's L0 rung: for each ``real_drivers``
    entry, construct the driver, call ``get_idn()``, then call one more
    zero-argument read-only getter (picked deterministically — alphabetically
    first ``get_*``/``read_*``/``is_*`` method besides ``get_idn`` that takes
    no arguments) to prove a real reading parses, not just the identity
    string. A human still has to eyeball whether the returned values are
    physically plausible; this only proves communication and parsing work.

    Args:
        config_path: Config directory (devices.yaml + monitor.yaml).

    Returns:
        One L0BenchResult per configured driver, in config order. If the
        config itself is invalid, a single failing result is returned.
    """
    errors = validate_config_dir(config_path)
    if errors:
        return [
            L0BenchResult(
                alias="", address="", driver_class="",
                idn_ok=False, detail="; ".join(errors),
            )
        ]

    drivers = _load_real_drivers(config_path)
    return [_bench_one_driver_l0(alias, cfg) for alias, cfg in drivers.items()]


def _bench_one_driver_l0(alias: str, cfg: dict) -> L0BenchResult:
    """Construct one driver, identify it, then try one extra passive getter."""
    class_path = str(cfg["class"])
    address = str(cfg.get("address", ""))

    try:
        driver = _import_class(class_path)(address)
    except Exception as exc:  # noqa: BLE001 — any construction failure is L0-fail
        return L0BenchResult(
            alias=alias, address=address, driver_class=class_path,
            idn_ok=False, detail=f"constructor raised {type(exc).__name__}: {exc}",
        )

    try:
        idn = str(driver.get_idn()).strip()
    except Exception as exc:  # noqa: BLE001 — communication or driver bug either way
        _close_driver(driver)
        return L0BenchResult(
            alias=alias, address=address, driver_class=class_path,
            idn_ok=False, detail=f"get_idn() raised {type(exc).__name__}: {exc}",
        )

    result = L0BenchResult(
        alias=alias, address=address, driver_class=class_path,
        idn_ok=bool(idn), idn=idn,
        detail="" if idn else "get_idn() returned empty",
    )

    getter_name = _pick_l0_getter(driver)
    if getter_name:
        result.getter = getter_name
        try:
            value = getattr(driver, getter_name)()
            result.getter_ok = True
            result.getter_value = repr(value)
        except Exception as exc:  # noqa: BLE001 — record and move on, don't abort the sweep
            result.getter_ok = False
            result.getter_value = f"{type(exc).__name__}: {exc}"

    _close_driver(driver)
    return result


def _pick_l0_getter(driver: Any) -> str | None:
    """Pick one zero-argument read-only getter (besides get_idn) to exercise.

    Deterministic (alphabetical via inspect.getmembers) so repeated bench
    runs pick the same method — reproducibility matters more than which
    specific getter gets exercised.

    Returns:
        The method name, or None if the driver has no other zero-arg getter.
    """
    for name, func in inspect.getmembers(type(driver), inspect.isfunction):
        if name == "get_idn" or name.startswith("_") or not is_read_only(name):
            continue
        params = [
            p for p in inspect.signature(func).parameters.values() if p.name != "self"
        ]
        if not params:
            return name
    return None


def _close_driver(driver: Any) -> None:
    """Best-effort release of a driver's VISA session.

    Drivers have no uniform close() in their contract (the app holds them for
    its whole lifetime), so the bench closes the underlying pyvisa handle if
    the conventional ``_instr`` attribute exists.
    """
    instr = getattr(driver, "_instr", None)
    if instr is not None:
        try:
            instr.close()
        except Exception:  # noqa: BLE001 — cleanup must never raise
            pass


# ── Driver bench ──────────────────────────────────────────────────────────────


def is_read_only(method_name: str) -> bool:
    """True if a method name is classified as read-only (safe to call)."""
    return method_name.startswith(READ_ONLY_PREFIXES)


class DriverBench:
    """Wraps one driver instance for introspection, calls, and raw I/O.

    The bench is the primitive behind the CLI's ``read`` / ``write`` /
    ``query`` / ``send`` subcommands. It enforces the read/write split:
    ``call()`` refuses non-read-only methods unless ``allow_write=True``.
    """

    def __init__(self, driver: Any, alias: str, address: str, class_path: str) -> None:
        """Wrap an already-constructed driver.

        Args:
            driver: The driver instance.
            alias: Config alias (or a made-up name for ad-hoc benches).
            address: VISA resource string the driver was built with.
            class_path: Dotted class path, for reporting.
        """
        self._driver = driver
        self.alias = alias
        self.address = address
        self.class_path = class_path

    @classmethod
    def from_config(cls, config_path: str, alias: str) -> DriverBench:
        """Construct the bench for one driver declared in a config.

        Args:
            config_path: Config directory containing devices.yaml.
            alias: Driver name under ``real_drivers``.

        Returns:
            A DriverBench wrapping a freshly constructed driver.

        Raises:
            I2ASConfigError: If the alias is not in the config.
            I2ASCommunicationError: If the driver cannot open its resource.
        """
        drivers = _load_real_drivers(config_path)
        if alias not in drivers:
            raise I2ASConfigError(
                f"No driver '{alias}' in {config_path} "
                f"(available: {sorted(drivers)})"
            )
        cfg = drivers[alias]
        class_path = str(cfg["class"])
        address = str(cfg.get("address", ""))
        driver = _import_class(class_path)(address)
        logger.info("Bench opened '%s' (%s at %s)", alias, class_path, address)
        return cls(driver, alias, address, class_path)

    @classmethod
    def from_class(cls, class_path: str, address: str) -> DriverBench:
        """Construct the bench for an arbitrary driver class and address.

        This is the driver-development entry point: bench a driver that has
        no config entry yet.

        Args:
            class_path: Dotted path to the driver class.
            address: VISA resource string.

        Returns:
            A DriverBench wrapping a freshly constructed driver.
        """
        driver = _import_class(class_path)(address)
        return cls(driver, alias=class_path.rsplit(".", 1)[-1],
                   address=address, class_path=class_path)

    # -- Introspection ------------------------------------------------------

    def list_methods(self) -> list[MethodInfo]:
        """Return every public method with signature, doc, and read/write class."""
        infos: list[MethodInfo] = []
        for name, func in inspect.getmembers(type(self._driver), inspect.isfunction):
            if name.startswith("_"):
                continue
            sig = inspect.signature(func)
            params = [p for p in sig.parameters.values() if p.name != "self"]
            public_sig = sig.replace(parameters=params)
            doc = (inspect.getdoc(func) or "").split("\n", 1)[0]
            hints = _safe_type_hints(func)
            infos.append(
                MethodInfo(
                    name=name,
                    signature=str(public_sig),
                    doc=doc,
                    read_only=is_read_only(name),
                    params={
                        p.name: getattr(hints.get(p.name), "__name__", "str")
                        for p in params
                    },
                )
            )
        return infos

    # -- Calling driver methods ----------------------------------------------

    def call(self, method_name: str, args: list[str] | None = None,
             allow_write: bool = False) -> Any:
        """Call one driver method with string arguments coerced via type hints.

        Args:
            method_name: Public method name (e.g. ``get_voltage``).
            args: Positional arguments as strings (CLI form); each is coerced
                to the annotated type (float/int/bool/str).
            allow_write: Must be True to call a method that is not classified
                read-only. The CLI's ``write`` subcommand sets this; ``read``
                never does.

        Returns:
            Whatever the driver method returns.

        Raises:
            ValueError: Unknown/private method, write without allow_write, or
                unparseable arguments.
        """
        if method_name.startswith("_"):
            raise ValueError(f"Private method {method_name!r} is not callable")
        method = getattr(self._driver, method_name, None)
        if not callable(method):
            available = [m.name for m in self.list_methods()]
            raise ValueError(
                f"{self.alias} has no method {method_name!r} "
                f"(available: {available})"
            )
        if not is_read_only(method_name) and not allow_write:
            raise ValueError(
                f"{method_name!r} changes instrument state — use the write "
                f"path (allow_write=True) to call it deliberately"
            )
        coerced = self._coerce_args(method, args or [])
        logger.info("Bench call: %s.%s(%s)", self.alias, method_name, coerced)
        result = method(*coerced)
        logger.info("Bench call result: %r", result)
        return result

    def _coerce_args(self, method: Any, args: list[str]) -> list[Any]:
        """Convert CLI string arguments to the method's annotated types."""
        func = getattr(method, "__func__", method)
        sig = inspect.signature(func)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        if len(args) > len(params):
            raise ValueError(
                f"Too many arguments: {func.__name__} takes at most "
                f"{len(params)} ({[p.name for p in params]}), got {len(args)}"
            )
        hints = _safe_type_hints(func)
        coerced: list[Any] = []
        for raw, param in zip(args, params):
            target = hints.get(param.name, str)
            try:
                coerced.append(_coerce_one(raw, target))
            except ValueError as exc:
                raise ValueError(
                    f"Argument {param.name}={raw!r}: {exc}"
                ) from exc
        return coerced

    # -- Raw VISA I/O ---------------------------------------------------------

    def query(self, command: str) -> str:
        """Send a raw command and return the instrument's reply.

        Uses the driver's underlying pyvisa handle (the conventional
        ``_instr`` attribute). Raw I/O is for commands the driver does not
        wrap yet — the response bypasses all driver parsing.

        Args:
            command: Raw command string, e.g. ``"*IDN?"``.

        Returns:
            The stripped response string.

        Raises:
            ValueError: If the driver exposes no raw VISA handle (sim
                drivers do not).
        """
        instr = self._raw_handle()
        logger.info("Bench raw query: %s <- %r", self.alias, command)
        response = str(instr.query(command)).strip()
        logger.info("Bench raw reply: %r", response)
        return response

    def send(self, command: str) -> None:
        """Send a raw command with no reply expected (a bare VISA write).

        Args:
            command: Raw command string, e.g. ``":SENS:VOLT:RANG 0.1"``.

        Raises:
            ValueError: If the driver exposes no raw VISA handle.
        """
        instr = self._raw_handle()
        logger.info("Bench raw send: %s <- %r", self.alias, command)
        instr.write(command)

    def _raw_handle(self) -> Any:
        instr = getattr(self._driver, "_instr", None)
        if instr is None:
            raise ValueError(
                f"{self.alias} ({self.class_path}) exposes no raw VISA handle — "
                f"simulated drivers support driver-method calls only"
            )
        return instr

    def close(self) -> None:
        """Release the driver's VISA session (best-effort)."""
        _close_driver(self._driver)


# ── Small helpers ─────────────────────────────────────────────────────────────


def _safe_type_hints(func: Any) -> dict[str, Any]:
    """typing.get_type_hints that returns {} instead of raising."""
    try:
        return typing.get_type_hints(func)
    except Exception:  # noqa: BLE001 — unresolvable hints just disable coercion
        return {}


def _coerce_one(raw: str, target: Any) -> Any:
    """Coerce one CLI string to the annotated target type."""
    if target is bool:
        lowered = raw.strip().lower()
        if lowered in ("true", "1", "on", "yes"):
            return True
        if lowered in ("false", "0", "off", "no"):
            return False
        raise ValueError("expected a boolean (true/false/1/0/on/off)")
    if target is int:
        return int(raw)
    if target is float:
        return float(raw)
    return raw


# ── trend_store_live (pull-only Trend check) ────────────────────────────────
# See i2as.core.trend_checks.declared_checks()'s docstring for why this
# one check is not declared there: it exists to catch a wedged or crashed
# application, and a check scheduled on that same process's own QTimer
# (i2as.core.trend_check_runner) cannot fire once the process is the
# thing that is hung — it would report healthy in exactly the scenario it
# is built for. It lives here, next to the other troubleshoot diagnostic
# primitives, because it is evaluated only from a separate process (the
# troubleshoot CLI) reading the store's file state from outside, exactly
# like every other check in this module reads driver/bus state from outside
# a running instrument session.

_TREND_STORE_LIVE_TAIL_BYTES = 4096  # "one record", read cheaply from the file's tail


def _tail_last_line(path: Path, max_bytes: int = _TREND_STORE_LIVE_TAIL_BYTES) -> str | None:
    """Return the last non-blank line of *path*, reading only its final bytes.

    Deliberately not a full-file parse (`trend_history.read_tier()`'s job for
    a real window query): `trend_store_live` only ever needs the single
    newest record, so this seeks near the end of the file instead of loading
    every line the way a full tier read would.

    Args:
        path: File to tail.
        max_bytes: How many trailing bytes to read — generous for one JSONL
            record, which is at most a few hundred bytes.

    Returns:
        The last non-blank line's text, or `None` if the file could not be
        read or has no non-blank line within the tailed bytes.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - max_bytes))
            chunk = f.read()
    except OSError:
        return None
    lines = [line for line in chunk.decode("utf-8", errors="ignore").splitlines() if line.strip()]
    return lines[-1] if lines else None


def _extract_record_t(line: str) -> float | None:
    """Parse one trend-history JSONL line and return its `"t"` field, if valid."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    t = obj.get("t")
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return float(t)
    return None


def check_trend_store_live(
    log_dir: Path, stale_seconds: float, now: float | None = None
) -> CheckResult:
    """Is the trend-history store still receiving records at all?

    Reads FILE state only — the live raw-tier file's mtime plus its newest
    record's timestamp — never an in-process counter, which would freeze
    alongside the very application wedge this check exists to detect. Fails
    when the newest record is older than `stale_seconds`. Deliberately cheap
    (one file's tail, not `trend_history.summarize()`/`read_tier()` over a
    whole window) and deliberately narrow: it reads only the live undated
    `trend_history_raw.jsonl`, not its rotated siblings, since "is the store
    live right now" only ever needs the newest record.

    Not useless while the app is otherwise healthy: it also catches the
    tiered-trend *logger* dying while the rest of the application keeps
    running (a narrower failure than a full wedge, but the same file-state
    signal catches both).

    Args:
        log_dir: Directory containing the trend-history JSONL files, as
            resolved by `i2as.core.paths.log_directory()`.
        stale_seconds: Age, in seconds, beyond which the newest record is
            considered stale. Computed by the caller from this setup's
            `trends.store_live_stale_ticks` config value times its
            `monitor.tick_interval_ms` (see
            `i2as.core.config.read_tick_interval_ms()`), never
            hardcoded here.
        now: Reference "now" timestamp; defaults to `time.time()`.

    Returns:
        A `i2as.core.trend_checks.CheckResult` named `"trend_store_live"`,
        reusing that type only as a value object — this function evaluates
        outside the `TrendCheck`/`run_check()` machinery entirely, since its
        data source (file mtime) is not `trend_history.summarize()`. `passed`
        is `None` (cannot tell) if the live file is missing or unreadable —
        e.g. the app has never run against this log directory — mirroring
        the rest of the Trend check standard's "no data" convention rather
        than treating an absent store as a definite failure.
    """
    if now is None:
        now = time.time()
    spec = trend_history.TIERS["raw"]
    path = Path(log_dir) / spec.filename

    if not path.exists():
        return CheckResult(
            name="trend_store_live",
            passed=None,
            message=f"Cannot tell — {path} does not exist (never written, or an unreachable log directory).",
            evidence={"path": str(path)},
        )

    try:
        mtime_age_s = now - path.stat().st_mtime
    except OSError as exc:
        return CheckResult(
            name="trend_store_live",
            passed=None,
            message=f"Cannot tell — could not stat {path}: {exc}",
            evidence={"path": str(path)},
        )

    last_line = _tail_last_line(path)
    record_t = _extract_record_t(last_line) if last_line is not None else None
    if record_t is None:
        return CheckResult(
            name="trend_store_live",
            passed=None,
            message=f"Cannot tell — {path} exists but its newest record could not be read.",
            evidence={"path": str(path), "file_mtime_age_s": round(mtime_age_s, 1)},
        )

    record_age_s = now - record_t
    evidence = {
        "path": str(path),
        "newest_record_age_s": round(record_age_s, 1),
        "file_mtime_age_s": round(mtime_age_s, 1),
        "stale_threshold_s": stale_seconds,
    }
    if record_age_s > stale_seconds:
        return CheckResult(
            name="trend_store_live",
            passed=False,
            message=(
                f"Store stale — newest record is {record_age_s:.0f} s old "
                f"(file last written {mtime_age_s:.0f} s ago), past the "
                f"{stale_seconds:.0f} s threshold. The application may be wedged or crashed."
            ),
            evidence=evidence,
        )
    return CheckResult(
        name="trend_store_live",
        passed=True,
        message=(
            f"Store live — newest record is {record_age_s:.0f} s old "
            f"(file last written {mtime_age_s:.0f} s ago), threshold {stale_seconds:.0f} s."
        ),
        evidence=evidence,
    )
