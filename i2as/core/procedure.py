"""BaseProcedure — abstract base class for all I2AS procedures."""

from __future__ import annotations

import dataclasses
import logging
import math
import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, ClassVar

from i2as.core.data_manager import DataManager
from i2as.core.exceptions import I2ASConfigError
from i2as.core.gates import Gate
from i2as.core.plan import (
    Command,
    DataSchema,
    ParamGroup,
    ParamSpec,
    PhasePlan,
    ProbeSpec,
    StepCost,
    StepPlan,
    Target,
)
from i2as.core.station import Station
from i2as.core.sweep_builder import SweepAxis, build_axis_sweep, sweep_axis_param_specs

logger = logging.getLogger(__name__)


def _nanmean(values: Iterable[float]) -> float:
    """Mean of *values*, NaN entries excluded; NaN if none are valid.

    Reduces a raw diagnostic block's row axis into one scalar per channel
    (see ``SweepMeasureProcedure._raw_block_channel_columns``) — the row
    axis is the same "N readings at one measurement point" count a
    measurement VI's ``n_readings`` parameter declares, just without a
    companion SEM, since a diagnostic channel has no declared error column.

    Args:
        values: Raw per-row values for one channel, possibly including
            NaN (instrument-emitted or I2AS row-count padding).

    Returns:
        The mean of the non-NaN values, or NaN if every value is NaN.
    """
    valid = [v for v in values if not math.isnan(v)]
    return statistics.fmean(valid) if valid else float("nan")


@dataclasses.dataclass(frozen=True)
class RoleParam:
    """One instrument ROLE a procedure needs, discovered on the Station.

    The **role-discovery standard**: a procedure names the role it needs
    ("the magnet whose field this sweeps"), never a configured instrument
    name, so the same recipe runs on any setup that has such an instrument.
    Discovery happens at construction: the candidates come from the Station,
    the value defaults to the only candidate when there is exactly one, and a
    REQUIRED role with no candidate at all refuses the run at construction
    rather than failing mid-sweep.

    A procedure is hardware-blind (contract C6) and never imports a VI class,
    so ``candidates`` is a callable over the Station's own role accessors
    (``magnet_vi_names()``, ``temperature_vi_names()``,
    ``measurement_vi_names()``), which are the thin, named front of
    ``Station.vi_names_by_base()``.

    Attributes:
        candidates: ``station -> [vi_name, ...]``, in config order. The first
            entry is the default.
        description: One line describing what the procedure does with this
            instrument; rendered as the parameter's help text.
        required: Whether the procedure can run without one. A required role
            with no candidate raises ``I2ASConfigError`` at construction;
            an optional one resolves to ``""``.
    """

    candidates: Callable[[Any], list[str]]
    description: str
    required: bool = True


class BaseProcedure:
    """Abstract base class for declarative measurement procedures.

    Procedures declare *what* the system should do; the Orchestrator handles
    *how* it executes. A procedure never calls driver or VI methods directly
    during ramping — it returns typed plans (``PhasePlan`` / ``StepPlan``,
    built from ``Target`` and ``Command`` objects from ``i2as.core.plan``)
    and the Orchestrator dispatches them.

    Class attributes:
        name: Human-readable display name (shown in GUI procedure browser).
        description: One-line description.
        sweep_axis: Optional ``SweepAxis`` declaring the one quantity this
            procedure sweeps over (e.g. field, temperature). When set, its
            hidden parameters (``{key}_mode``, ``{key}_start``, etc. — see
            ``sweep_builder.sweep_axis_param_specs()``) are merged into
            ``parameters`` automatically, ``_build_sweep_array()`` needs no
            override, and the GUI renders a mode-selector widget (linear /
            segments / CSV, with hysteresis) instead of flat text fields.
            Most sweep procedures should use this instead of hand-writing
            ``sweep_parameters`` + ``_build_sweep_array()``.
        sweep_parameters: Parameters that define *what* to sweep over, for
            procedures that are not using ``sweep_axis`` (or that need extra
            sweep-related parameters alongside it). The GUI renders these
            in the Sweep column of the parameter form.
        system_parameters: Parameters that set system state during the sweep
            (e.g. temperature, ramp rates, wait times). Rendered in the
            System column.
        measurement_parameters: Parameters that configure the measurement VI
            (e.g. current_A, compliance_A, readings_per_point). Rendered in
            the Measurement column.
        parameters: Union of sweep_axis's hidden params (if any),
            sweep_parameters, system_parameters, and measurement_parameters,
            auto-built by ``__init_subclass__``. Read-only — do not set
            directly.

    Each parameter is declared as a ``ParamSpec`` (see ``i2as.core.plan``):
    ``type`` and ``default`` are required; ``unit``, ``min``, ``max``,
    ``description``, ``choices`` are optional. ``ParamSpec`` validates the
    declaration eagerly at construction (choices non-empty and correctly typed,
    default among the choices / within the bounds), so a malformed parameter
    fails when the class body runs rather than at form-render time.

    Input-widget types (how the GUI renders a parameter):

    * Plain number / text (default): any ``type`` (``float``, ``int``, ``str``)
      without ``choices`` renders as a free-text field parsed by ``type``.
    * Enumerated choice: set ``choices`` to a **label -> value dict**,
      e.g. ``{"10 mV": 0.01, "100 mV": 0.1}``. The GUI renders a drop-down
      showing the labels; the collected value is the mapped value (``0.01``),
      so the procedure never translates. ``default`` must be one of the
      mapped values, and every value must be an instance of ``type``.
    * Boolean toggle: set ``type=bool``. The GUI renders a checkbox; the
      collected value is ``True`` / ``False``.

    The ParamSpec -> Qt-widget mapping lives entirely in
    ``i2as.gui.param_form``; a procedure never names a widget class.
    ``tests/test_conformance.py`` checks every procedure declares ParamSpecs so
    they all inherit the same rules.

    Example::

        class FieldSweepIV(BaseProcedure):
            name = "Field Sweep IV"
            description = "Sweep magnetic field, measure IV at each point"
            sweep_axis = SweepAxis(
                key="field", unit="T", data_key="field_T",
                description="Magnetic field",
                default_start=-1.0, default_end=1.0, default_steps=101,
            )
    """

    name: str = ""
    description: str = ""

    # Parameter groups — sweep_axis (optional) plus the three dicts below.
    # ``parameters`` is auto-built as their union by __init_subclass__ so
    # existing code that iterates cls.parameters continues to work.
    sweep_axis: SweepAxis | None = None
    sweep_parameters: dict[str, ParamSpec] = {}
    system_parameters: dict[str, ParamSpec] = {}
    measurement_parameters: dict[str, ParamSpec] = {}
    parameters: dict[str, ParamSpec] = {}

    # The instrument roles this procedure discovers on the Station (the
    # role-discovery standard — see ``RoleParam``). Each becomes one
    # parameter whose choices are the configured instruments of that role,
    # resolved at construction by ``role_vi()``.
    role_parameters: ClassVar[dict[str, RoleParam]] = {}

    sweep_data_keys: list[str] = []        # procedure's own scalar sweep columns (e.g. field_T)
    measurement_data_keys: list[str] = []  # measurement array columns (e.g. voltage_V, current_A)
    default_x_key: str = ""                # default X axis for live plots

    # Whether this procedure resolves its measurement VI (and that VI's
    # parameters) from the Station at construction, so it CANNOT be built from
    # an empty Station. Static procedures leave this False; the generic sweep
    # procedures (``SweepMeasureProcedure``) set it True. The conformance tests
    # read it to decide whether to hand the procedure a populated sim Station.
    requires_measurement_vi: bool = False

    # What kind of run this procedure execution is, recorded in the
    # Orchestrator's run manifests and in the data file's /metadata.run_kind:
    # "run" for a normal science run. ``apply_probe()`` sets it to "probe" on
    # the one reduced instance it builds (see ProbeSpec's reduction rules), so
    # probe data files and run records are never confused with science data.
    run_kind: str = "run"

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        axis_params = sweep_axis_param_specs(cls.sweep_axis) if cls.sweep_axis else {}
        # A role parameter's CHOICES need a Station, so the class-level spec
        # carries only the empty default ("discover it"); the real one, with
        # this setup's instruments as its choices, is built by
        # get_param_groups().
        role_params = {
            name: ParamSpec(type=str, default="", description=role.description)
            for name, role in cls.role_parameters.items()
        }
        cls.parameters = {
            **axis_params,
            **role_params,
            **cls.sweep_parameters,
            **cls.system_parameters,
            **cls.measurement_parameters,
        }

    @classmethod
    def get_param_groups(
        cls, station: Station, selections: Mapping[str, Any] | None = None
    ) -> list[ParamGroup]:
        """Return the ordered ``ParamGroup`` list the GUI renders for this procedure.

        This is the hook the GUI form builder calls instead of reading the three
        class dicts directly. The default implementation returns the static
        Sweep / System / Measurement groups, in that order, SKIPPING any that
        declare no parameters. The ``sweep_axis`` hidden parameters are
        deliberately NOT in any group: the GUI renders them through the separate
        ``SweepAxisWidget`` (linear / segments / CSV), exactly as before.

        ``station`` and ``selections`` are unused by this default, but are part
        of the signature so a Wave-5 subclass can override this method to
        compute its groups dynamically — e.g. deriving measurement groups from
        the station's configured measurement VI, or re-deriving the whole form
        from a ``structural`` selection the user has changed (hence
        ``selections``, the current values of any structural parameters).

        Args:
            station: The active Station instance (unused by the default).
            selections: Current values of the form's structural parameters, or
                ``None`` before any have been chosen (unused by the default).

        Returns:
            The non-empty parameter groups, in Sweep, System, Measurement order.
        """
        candidates = (
            ("instruments", "Instruments", cls._role_param_specs(station, selections)),
            ("sweep", "Sweep", cls.sweep_parameters),
            ("system", "System", cls.system_parameters),
            ("measurement", "Measurement", cls.measurement_parameters),
        )
        return [
            ParamGroup(key=key, title=title, params=params)
            for key, title, params in candidates
            if params
        ]

    @classmethod
    def _role_param_specs(
        cls, station: Station, selections: Mapping[str, Any] | None = None
    ) -> dict[str, ParamSpec]:
        """Build this procedure's role parameters against a live Station.

        The rendering half of the role-discovery standard (see
        ``RoleParam``): each declared role becomes a drop-down over the
        instruments of that role this setup configured, defaulted to the
        first — which IS the value when there is only one. A role with no
        candidate contributes no widget; a REQUIRED one is refused at
        construction instead, where the message can say what to configure.

        Args:
            station: The active Station, which supplies the candidates.
            selections: Current structural-parameter values, so a role the
                user already chose stays chosen across a re-render.

        Returns:
            ``{param_name: ParamSpec}``, empty when this procedure declares
            no roles or the station has none of them.
        """
        selections = selections or {}
        specs: dict[str, ParamSpec] = {}
        for name, role in cls.role_parameters.items():
            names = role.candidates(station)
            if not names:
                continue
            selected = selections.get(name)
            if selected not in names:
                selected = names[0]
            specs[name] = ParamSpec(
                type=str,
                default=selected,
                choices={vi_name: vi_name for vi_name in names},
                description=role.description,
            )
        return specs

    @classmethod
    def live_plot_measurement_keys(
        cls, station: Station, selections: Mapping[str, Any] | None = None
    ) -> list[str]:
        """Return the measurement array/scalar keys available for live-plot axes.

        The GUI populates its plot axis selectors before a run starts, so it
        needs the measurement columns without an instance. The default returns
        the static ``measurement_data_keys`` class attribute. A procedure whose
        measurement columns depend on a station-selected VI overrides this to
        read them from that VI (see ``SweepMeasureProcedure``).

        Args:
            station: The active Station (unused by the default).
            selections: Current structural-parameter values, or ``None``
                (unused by the default).

        Returns:
            Ordered list of measurement column names for the plot selectors.
        """
        _ = (station, selections)
        return list(cls.measurement_data_keys)

    @classmethod
    def live_plot_loop_labels(
        cls, station: Station, selections: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, str] | None, dict[str, str] | None]:
        """Return the label maps for the plot panels' two Loop selectors.

        The GUI gives every plot a "Loop 1" and a "Loop 2" selector that pick
        WHICH reading of a datapoint is plotted, one per reading-loop slot.
        The default (no reading loop concept) returns ``(None, None)`` — both
        selectors hidden. A procedure with a reading loop overrides this (see
        ``SweepMeasureProcedure``) to return, per slot, ``{}`` (slot off /
        static — selector visible, disabled) or an ordered
        ``{suffix_label: display_text}`` map (e.g.
        ``{"A1": "A1 = Mux-Ch1", ...}`` — selector enabled).

        Args:
            station: The active Station (unused by the default).
            selections: Current structural-parameter values, or ``None``
                (unused by the default).

        Returns:
            ``(None, None)`` by default.
        """
        _ = (station, selections)
        return (None, None)

    def __init__(
        self,
        station: Station,
        sample_info: dict[str, str],
        data_directory: str,
        file_prefix: str = "",
        experiment_info: dict[str, Any] | None = None,
        **param_values: Any,
    ) -> None:
        """Initialise the procedure.

        Args:
            station: The active Station instance.
            sample_info: ``{"sample_name": str, "sample_id": str, "comments": str}``.
            data_directory: Base directory for HDF5 output files.
            file_prefix: User-chosen filename prefix for the HDF5 output file
                (still suffixed with a unique timestamp by ``DataManager``).
                Empty string falls back to ``self.name``. Frozen into the
                procedure instance at construction time, so queued entries
                each keep the prefix that was set when they were added.
            experiment_info: Optional experiment-level context from the session
                layer (experiment id/title, user identity, ELN link), forwarded
                verbatim to every ``DataManager`` this procedure creates and
                stored as ``/metadata/experiment_info``. ``None`` means "no
                experiment open" and is recorded as ``{}``.
            **param_values: Procedure-specific parameter values from the GUI form.
                Any declared parameter with a ``default`` that the caller omits
                is filled in automatically; caller-supplied values always win.
        """
        self._station = station
        self._sample_info = sample_info
        self._data_directory = data_directory
        self._file_prefix = file_prefix
        self._experiment_info: dict[str, Any] = dict(experiment_info or {})
        # Every ParamSpec carries a default (enforced at ParamSpec construction),
        # so the merge is unconditional — no "default present?" guard needed.
        merged_params: dict[str, Any] = {
            param_name: spec.default
            for param_name, spec in type(self).parameters.items()
        }
        merged_params.update(param_values)
        self._params: dict[str, Any] = merged_params
        # Role discovery, before anything else reads a VI name: a subclass's
        # own __init__ (and _build_sweep_array below) may ask for a resolved
        # role straight after calling super().__init__().
        self._role_vis: dict[str, str] = {
            name: self._resolve_role(name, role)
            for name, role in type(self).role_parameters.items()
        }
        self._data_manager = None
        # Optional declared HDF5 layout. A procedure that assembles a DataSchema
        # in initiate() (the generic sweep procedures do) stores it here, and
        # _save_datapoint() then validates every datapoint against it. Left None
        # by hand-written procedures, which keep working unvalidated.
        self._data_schema: DataSchema | None = None
        self._sweep: list = self._build_sweep_array()
        self._index: int = 0

    def _resolve_role(self, param_name: str, role: RoleParam) -> str:
        """Resolve one declared role to a configured instrument name.

        Args:
            param_name: The role parameter's name, e.g. ``"field_vi"``.
            role: Its declaration.

        Returns:
            The chosen VI name, or ``""`` for an optional role this station
            has no instrument for.

        Raises:
            I2ASConfigError: If the station has no instrument for a
                REQUIRED role, or if the caller named one that is not a
                candidate (a typo, or a config that changed under a queued
                run).
        """
        names = role.candidates(self._station)
        chosen = str(self._params.get(param_name) or "")
        if chosen:
            if chosen not in names:
                raise I2ASConfigError(
                    f"{type(self).name!r}: {param_name}={chosen!r} is not an "
                    f"instrument of that role on this station "
                    f"(candidates: {names or 'none'})."
                )
            return chosen
        if names:
            return names[0]
        if role.required:
            raise I2ASConfigError(
                f"{type(self).name!r} needs an instrument for {param_name!r} "
                f"({role.description}), but this station configures none. "
                f"Add one to devices.yaml, or run a procedure that does not "
                f"need it."
            )
        return ""

    def role_vi(self, param_name: str) -> str:
        """Return the instrument name this run resolved a declared role to.

        Args:
            param_name: The role parameter's name, e.g. ``"field_vi"``.

        Returns:
            The VI name, or ``""`` for an optional role with no instrument.

        Raises:
            KeyError: If this procedure declares no such role.
        """
        return self._role_vis[param_name]

    # ------------------------------------------------------------------
    # Override in subclass
    # ------------------------------------------------------------------

    def _build_sweep_array(self) -> list:
        """Build the list of sweep points from ``self._params``.

        Called once at construction. If the subclass declares ``sweep_axis``,
        this delegates to ``sweep_builder.build_axis_sweep()`` and needs no
        override. A subclass without ``sweep_axis`` must override this.

        Returns:
            List of sweep point values (e.g. field values in tesla).
        """
        if type(self).sweep_axis is not None:
            return build_axis_sweep(type(self).sweep_axis, self._params)
        return []

    # ------------------------------------------------------------------
    # System-state enrichment helpers (called by subclass measure/initiate)
    # ------------------------------------------------------------------

    def _build_data_config(self, base_config: dict) -> dict:
        """Inject unix_time and all cached system-VI readings into *base_config*.

        Merges ``unix_time`` plus every numeric scalar from
        ``station.last_state_flat()`` into ``base_config["sweep_columns"]``.
        The procedure's own columns (already in *base_config*) take priority on
        any name collision.

        Args:
            base_config: Dict with ``sweep_columns`` and ``measurement_arrays``
                keys as accepted by DataManager.

        Returns:
            New dict with system columns prepended to sweep_columns.
        """
        system_keys = self._station.last_state_flat().keys()
        extra: dict[str, str] = {"unix_time": "float", **{k: "float" for k in system_keys}}
        result = dict(base_config)
        result["sweep_columns"] = {**extra, **base_config.get("sweep_columns", {})}
        return result

    def _save_datapoint(self, measured_data: dict) -> None:
        """Enrich *measured_data* with timestamp + system state, then save.

        Merges ``unix_time`` (UTC epoch float) and all cached system-VI readings
        into *measured_data* before forwarding to the DataManager. Uses the
        monitor-tick cache — no extra hardware poll. Procedure-specific keys in
        *measured_data* overwrite system-state keys on collision (e.g. a
        procedure's explicit ``field_T`` read takes precedence).

        Args:
            measured_data: Dict of measurement results from the measurement VI.

        Raises:
            RuntimeError: If called before ``initiate()`` (data_manager is None).
            DataSchemaError: If a ``self._data_schema`` is set and the enriched
                datapoint does not match it (missing/extra columns or a
                wrong-length array). Propagates so the Orchestrator contains the
                run to ERROR before any malformed datapoint is written.
        """
        if self._data_manager is None:
            raise RuntimeError("_save_datapoint() called before initiate()")
        enriched = {
            "unix_time": time.time(),
            **self._station.last_state_flat(),
            **measured_data,
        }
        # Validate BEFORE writing: on a mismatch nothing is saved and the
        # DataSchemaError propagates. Skipped when no schema was declared.
        if self._data_schema is not None:
            self._data_schema.validate(enriched)
        self._data_manager.save_datapoint(
            sweep_index=self._index,
            measured_data=enriched,
            station_snapshot=self._station.cached_state,
        )

    # ------------------------------------------------------------------
    # Public read-only surface (consumed by the Orchestrator's run manifests
    # and the GUI — no caller should ever reach into _data_manager directly)
    # ------------------------------------------------------------------

    @property
    def data_filepath(self) -> str | None:
        """Absolute path of this run's HDF5 file, or ``None`` before ``initiate()``.

        Remains available while the run's ``DataManager`` is open; after
        ``standby()``/``abort()`` close the file the property returns ``None``
        again, so consumers that need the path across the whole run (the
        Orchestrator's run manifests) must capture it at start.

        Returns:
            The file path as a string, or ``None`` when no data file is open.
        """
        if self._data_manager is None:
            return None
        return str(self._data_manager.filepath)

    @property
    def last_datapoint(self) -> dict:
        """A copy of the most recently saved datapoint, or ``{}``.

        The Orchestrator reads this after each ``measure()`` to emit
        ``measurement_ready`` for the live plot.

        Returns:
            The last saved datapoint dict (copied), or ``{}`` when no data
            file is open or nothing has been saved yet.
        """
        if self._data_manager is None or not self._data_manager.last_datapoint:
            return {}
        return dict(self._data_manager.last_datapoint)

    def get_params(self) -> dict[str, Any]:
        """Return a copy of this instance's merged parameter values.

        Declared defaults merged with caller-supplied values (and, for the
        generic sweep procedures, the resolved measurement-VI selection and its
        parameters). This is what the run manifests and HDF5 metadata record.

        Returns:
            A shallow copy of the parameter dict.
        """
        return dict(self._params)

    def claimed_vi_names(self) -> set[str] | None:
        """Return the VI names this procedure exclusively owns while running.

        Concurrency-scope hook: the Orchestrator captures this once, at run
        start, into ``_active_claims`` and consults it to decide whether a
        manual front-panel action submitted while this procedure is running
        may be
        admitted. A VI named in the returned set is refused (the refusal
        names this procedure as the owner); every VI NOT in the set stays
        under manual control exactly as in IDLE.

        Every procedure today keeps the default (claim everything) —
        procedures remain exclusive in this iteration (plan's "Concurrency
        scope: manual controls only" decision); only operations narrow
        their claims.

        Returns:
            A set of VI names, as registered on the station
            (``Station.get_vi_names()``), this procedure claims — or
            ``None`` (the default) to claim every system VI. ``None`` is the
            safe default: narrowing what a run blocks is an explicit
            per-class opt-in, never assumed.
        """
        return None

    def planned_targets(self) -> dict[str, list[float]]:
        """Return every system setpoint this run would command, per VI.

        The **pre-dispatch declaration**: a built run says what it is going to
        ask the hardware for, so those values can be checked against the
        setup's ``control_limits`` and the session envelope *before* anything
        is dispatched — which is what lets a queued run be validated the
        moment it is added rather than an hour later when it starts. Purely
        declarative: reading it touches no instrument, opens no data file and
        changes nothing about the run.

        The default is ``{}`` — "this run commands no system setpoint" — the
        honest answer for a procedure that only reads. A procedure that ramps
        overrides it, deriving the values from the same target hooks its plans
        are built from so the declaration cannot drift from what is actually
        dispatched.

        Returns:
            ``{vi_name: [setpoint, ...]}`` in SI units, in dispatch order.
        """
        return {}

    # ------------------------------------------------------------------
    # Probe runs and duration estimates — see ProbeSpec / StepCost
    # ------------------------------------------------------------------

    def apply_probe(self, spec: ProbeSpec) -> None:
        """Reduce this built run, in place, to the cheap probe *spec* describes.

        The one entry point of the **probe standard**, whose reduction rules
        are written in ``ProbeSpec``'s docstring: subsample the sweep to at
        most ``spec.n_points`` points keeping first and last, cap every
        declared seconds-valued parameter at ``spec.max_wait_s``, record the
        spec as the ``probe_spec`` parameter, and declare ``run_kind =
        "probe"`` so the manifest, the run record and the data file all say
        what this is. Averaging is reduced by the generic sweep procedure,
        which is the layer that knows the measurement VI (see
        ``SweepMeasureProcedure.apply_probe``).

        Called by ``run_builder.build_procedure()`` on the freshly built
        instance — never mid-run: it rewrites the sweep the run is about to
        walk.

        Args:
            spec: The reduction to apply.

        Raises:
            TypeError: If *spec* is not a ``ProbeSpec``.
        """
        if not isinstance(spec, ProbeSpec):
            raise TypeError(f"apply_probe expects a ProbeSpec, got {spec!r}")
        self._sweep = self._probe_subsample(self._sweep, spec.n_points)
        capped = self._probe_cap_waits(
            type(self).parameters, self._params, spec.max_wait_s
        )
        self._params["probe_spec"] = spec.to_json()
        self.run_kind = "probe"
        logger.info(
            "%s: probe run — %d sweep points, waits capped at %g s%s",
            type(self).__name__,
            len(self._sweep),
            spec.max_wait_s,
            f" ({', '.join(capped)})" if capped else "",
        )

    @staticmethod
    def _probe_subsample(values: list, n_points: int) -> list:
        """Return at most *n_points* of *values*, keeping the first and last.

        Args:
            values: The full sweep array.
            n_points: Maximum points to keep (``>= 1``).

        Returns:
            A new list of evenly spaced points including ``values[0]`` and
            ``values[-1]``; *values* itself (copied) when it is already short
            enough, and ``[]`` when it is empty.
        """
        total = len(values)
        if total <= n_points:
            return list(values)
        if n_points == 1:
            return [values[0]]
        return [
            values[round(i * (total - 1) / (n_points - 1))] for i in range(n_points)
        ]

    @staticmethod
    def _probe_cap_waits(
        specs: Mapping[str, ParamSpec], params: dict[str, Any], max_wait_s: float
    ) -> list[str]:
        """Cap every seconds-valued declared parameter in *params*, in place.

        The declaration test behind the probe standard's wait rule: a
        parameter whose ``ParamSpec`` carries the unit ``"s"`` is a wait, so a
        new procedure's settle knob is reduced with no new code.

        Args:
            specs: The declarations to read the unit from.
            params: The run's parameter values, mutated in place.
            max_wait_s: The cap, in seconds.

        Returns:
            The names actually lowered, in declaration order.
        """
        capped: list[str] = []
        for name, spec in specs.items():
            if spec.unit != "s" or spec.type not in (int, float):
                continue
            value = params.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if value <= max_wait_s:
                continue
            params[name] = type(value)(max_wait_s) if spec.type is int else max_wait_s
            capped.append(name)
        return capped

    def estimate_step_seconds(self) -> StepCost:
        """Return this run's per-point cost model, apart from its ramps.

        The one hook a procedure contributes to the **duration-estimate
        standard**: ``core.estimates.estimate_duration()`` adds the ramp time
        it derives from ``planned_targets()`` and the setup's nominal ramp
        rates, and needs from the run only what the run itself knows — how
        many points it will take, how long it waits, and how long a
        measurement lasts. Purely declarative: it touches no instrument and
        changes nothing about the run.

        The default is honest rather than optimistic: the built sweep's length
        and no waits at all, with the omission recorded as an assumption. A
        procedure with settle waits overrides this (see
        ``SweepMeasureProcedure``), which is why every generic sweep procedure
        gets a real estimate for free.

        Returns:
            The ``StepCost`` for one run of this instance.
        """
        return StepCost(
            points=len(self._sweep),
            assumptions=(
                f"{type(self).__name__} declares no settle or measurement "
                f"time, so only its ramps are estimated",
            ),
        )

    def _claim_initiate_commands(self) -> tuple[Command, ...]:
        """Build an ``initiate()`` command for every VI this procedure claims.

        A concrete ``initiate()`` override puts this in the returned
        ``PhasePlan.claim_commands`` — dispatched by the Orchestrator BEFORE
        the plan's own ``targets``/``commands``, so a claimed VI is always
        back in its standard operating state first. This closes the gap the
        temperature VI lifecycle standard describes
        (``SampleTemperatureControllerVI.initiate()``): an operator may have
        left a claimed heater in MANUAL after a bench test, and a target
        ramping straight into that state would either hand the PID a stale
        setpoint or never move at all.

        Returns:
            One ``Command(vi_name, "initiate", {})`` per VI named in
            ``claimed_vi_names()`` — every station VI, in registration order,
            when that is ``None`` (the claim-everything default).
        """
        claimed = self.claimed_vi_names()
        vi_names = self._station.get_vi_names() if claimed is None else claimed
        return tuple(Command(vi_name, "initiate", {}) for vi_name in vi_names)

    def get_data_keys(self) -> list[str]:
        """Return all datapoint keys available for live-plot axis selection.

        Called by ProcedureWindow before a run starts (i.e. before initiate())
        to populate the axis selectors. Reads system state from the monitor
        cache — no hardware poll.

        Returns:
            Ordered list: ``["unix_time"] + sweep_data_keys + system_keys
            + measurement_data_keys``.
        """
        system_keys = list(self._station.last_state_flat().keys())
        return (
            ["unix_time"]
            + self.sweep_data_keys
            + system_keys
            + self.measurement_data_keys
        )

    # ------------------------------------------------------------------
    # Progress tracking (used by Orchestrator for GUI progress bar)
    # ------------------------------------------------------------------

    def get_sweep_array(self) -> list:
        """Return the full sweep array."""
        return self._sweep

    def get_progress(self) -> float:
        """Return fractional progress from 0.0 to 1.0.

        Returns:
            0.0 at the start; 1.0 when the sweep is complete.
        """
        if not self._sweep:
            return 1.0
        return self._index / len(self._sweep)

    def get_sweep_position(self) -> tuple[int, int]:
        """Return ``(current_point, total_points)`` as human 1-based counts.

        The Orchestrator uses this to compose concise status lines such as
        "Point 13/101" without reaching into ``_index``. Returns ``(0, 0)``
        for an empty sweep.

        Returns:
            ``(index + 1, len(sweep))``; ``(0, 0)`` when the sweep is empty.
        """
        total = len(self._sweep)
        if total == 0:
            return (0, 0)
        return (self._index + 1, total)

    # ------------------------------------------------------------------
    # Four-method procedure interface — must override in subclass
    # ------------------------------------------------------------------

    def initiate(self) -> PhasePlan:
        """Set up the experiment and return the initial plan.

        Jobs:
        1. Create the DataManager and HDF5 file.
        2. Return the initial system targets (first sweep point).
        3. Return measurement configuration commands.
        4. Return the wait time (seconds) after reaching initial targets.

        Returns:
            A ``PhasePlan`` bundling ``claim_commands`` (normally
            ``self._claim_initiate_commands()``, unchanged — every claimed VI
            back in its standard state before anything below is dispatched),
            ``targets`` (a ``{"vi_name": Target(...)}`` mapping), ``commands``
            (an ordered ``tuple[Command, ...]`` of measurement-VI calls), and
            ``wait_s`` (seconds to settle after the targets are reached).

        Raises:
            NotImplementedError: If not overridden in subclass.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement initiate()")

    def change_sweep_step(self) -> StepPlan | None:
        """Advance the sweep index and return the next plan.

        Called by the Orchestrator in SWEEPING state.

        Returns:
            A ``StepPlan`` (``targets`` mapping plus ``wait_s``) for the next
            sweep point, or ``None`` when the sweep is complete.

        Raises:
            NotImplementedError: If not overridden in subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement change_sweep_step()"
        )

    def measure(self) -> None:
        """Read data, snapshot station state, save to HDF5.

        Called by the Orchestrator in MEASURING state, after the system is
        stable at the current sweep point.

        Raises:
            NotImplementedError: If not overridden in subclass.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement measure()")

    def standby(self) -> PhasePlan:
        """Close the data file and return the safe-parking plan.

        Called by the Orchestrator in STANDBY state after the sweep completes
        (or after an abort).

        Returns:
            A ``PhasePlan`` (``targets`` mapping, ordered ``commands`` tuple,
            and ``wait_s``) describing where to park the system.

        Raises:
            NotImplementedError: If not overridden in subclass.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement standby()")

    # ------------------------------------------------------------------
    # Abort — has a working default; override to add measurement safe-off
    # ------------------------------------------------------------------

    def abort(self) -> tuple[Command, ...]:
        """Clean up after an abort: close the data file, keep partial data.

        Called by the Orchestrator on user abort and on ERROR/EMERGENCY
        entry. Unlike ``standby()`` it must not ramp anything — the
        Orchestrator holds the hardware separately via ``stop_ramps()``.

        Subclasses should extend this to also safe their measurement VI::

            def abort(self) -> tuple[Command, ...]:
                super().abort()
                return (Command("dc_measurement", "standby", {}),)

        Returns:
            An ordered ``tuple[Command, ...]`` the Orchestrator dispatches to
            safe the measurement hardware. The base implementation returns an
            empty tuple ``()``.
        """
        if self._data_manager is not None:
            self._data_manager.close()
            self._data_manager = None
        return ()

    # ------------------------------------------------------------------
    # Gates — no-op defaults; override to require settling before a reading
    # ------------------------------------------------------------------

    def initiation_gates(self) -> tuple[Gate, ...]:
        """Gates that must pass once, before the run's first measurement.

        Checked by the Orchestrator after the initial targets ramp completes
        (``initiate()``'s targets), in place of ``PhasePlan.wait_s`` when
        non-empty. The base implementation declares no gates, so ``wait_s``
        governs unchanged.

        Returns:
            An ordered ``tuple[Gate, ...]``; empty by default.
        """
        return ()

    def reading_gates(self) -> tuple[Gate, ...]:
        """Gates that must pass before every measurement after the first.

        Checked by the Orchestrator after each sweep step's targets ramp
        completes (``change_sweep_step()``'s targets), in place of
        ``StepPlan.wait_s`` when non-empty. The base implementation declares
        no gates, so ``wait_s`` governs unchanged.

        Returns:
            An ordered ``tuple[Gate, ...]``; empty by default.
        """
        return ()


# Form-parameter names the generic sweep procedure owns structurally (they are
# never a measurement VI's own knobs, so a VI param may not shadow them). The
# dynamic per-choice pick checkboxes live in the "loopN_pick_" namespace.
_STRUCTURAL_PARAM_NAMES = frozenset(
    {
        "measurement_vi",
        "loop1_parameter",
        "loop1_values",
        "loop2_parameter",
        "loop2_values",
    }
)


class SweepMeasureProcedure(BaseProcedure):
    """Generic base for "sweep one axis, run any measurement method" procedures.

    This is the heart of the modular-procedure design: ONE procedure per sweep
    axis (field, temperature) that can run ANY measurement VI the station
    exposes, chosen in the GUI at build time. It owns every part that is the
    same across sweep axes:

    * **Measurement-VI selection.** ``get_param_groups()`` builds a
      "Measurement method" group whose single ``measurement_vi`` parameter is a
      ``ParamSpec(type=str, structural=True, choices=<station measurement VIs>)``
      plus a group carrying the *selected* VI's own ``measurement_parameters``.
      Because the choices depend on the station, the class declares NO static
      measurement parameters.
    * **Construction.** ``__init__`` resolves the selected VI, merges its
      parameter defaults (``BaseProcedure`` only merges class-declared params),
      and records the selection + the instance-aware live-plot keys.
    * **The four-method loop.** ``initiate`` assembles a ``DataSchema`` (sweep
      axis column + system columns + the VI's arrays and scalar columns),
      derives the DataManager ``data_config`` from it, and arms the VI.
      ``measure`` runs the reading loop (below), tags on the axis read-back,
      and saves (the schema is validated per datapoint by
      ``BaseProcedure._save_datapoint``). ``standby`` / ``abort`` disarm the VI.
    * **The reading loop.** One datapoint may comprise several readings,
      taken by up to TWO generic loop slots nested inside ``measure()`` (slot
      1 = loop1 = axis 0, outer; slot 2 = loop2 = axis 1, inner). A slot is a
      loopable parameter — anything a reading-path VI advertises via
      ``reading_setters``: a routing VI's channel and a measurement VI's
      ``current_A`` are the same concept — plus an ordered value list from
      the auto-rendered "Reading loop" form group (checkboxes for an
      enumerated parameter, comma-separated text otherwise). A slot with ONE
      value is a static setting (dispatched once at ``initiate()``); with two
      or more it loops, dispatching value *i* before reading index *i* of its
      axis. Every measurement column gets a real ``(n_loop1, n_loop2)`` array
      axis in HDF5 (see ``DataSchema``/``_loop_shape``) — column names are
      never suffixed; axis index -> physical value is recorded in the HDF5
      metadata as ``procedure_params["loop1_values"]`` / ``["loop2_values"]``.
      All per-reading setup is dispatched as ``Command``s through the
      Station, never by direct VI calls; participating non-measurement VIs
      get their ``reading_safe_off`` at standby/abort.

    A concrete axis procedure (``FieldSweep``, ``TemperatureSweep``) subclasses
    this and supplies only the axis-specific pieces via the hooks below:
    ``_initial_system_targets`` / ``_step_targets`` / ``_standby_targets`` (the
    ramp targets), ``_axis_readback`` (the value stored in the sweep column each
    point), and ``_initiate_wait_s`` / ``_step_wait_s`` (settle times). It never
    imports a VI (contract C6): all measurement self-description is read through
    the ``Station`` instance.
    """

    # Generic sweep procedures resolve their measurement VI from the station at
    # construction, so they need a populated Station (not an empty one).
    requires_measurement_vi: bool = True

    def __init__(
        self,
        station: Station,
        sample_info: dict[str, str],
        data_directory: str,
        file_prefix: str = "",
        experiment_info: dict[str, Any] | None = None,
        **param_values: Any,
    ) -> None:
        """Resolve the selected measurement VI and merge its parameter defaults.

        Args:
            station: The active Station; must expose at least one measurement VI.
            sample_info: ``{"sample_name", "sample_id", "comments"}``.
            data_directory: Base directory for the HDF5 output file.
            file_prefix: Optional filename prefix (see ``BaseProcedure``).
            experiment_info: Optional session-layer experiment context (see
                ``BaseProcedure``), forwarded to the ``DataManager``.
            **param_values: Form values. ``measurement_vi`` selects the
                measurement VI by name (defaults to the first registered one);
                ``loop_parameter`` / ``loop_values`` define the optional
                reading loop (a loopable parameter of the selected VI and its
                comma-separated value list); the remaining keys are the
                sweep/system params and the selected VI's measurement params.

        Raises:
            I2ASConfigError: If the station has no measurement VIs, if
                ``measurement_vi`` names a VI that is not a measurement VI, or if
                a measurement param name collides with a sweep/system param.
        """
        names = station.measurement_vi_names()
        if not names:
            raise I2ASConfigError(
                f"{type(self).name!r} needs a measurement VI, but this station "
                f"has none. Add a measurement VI to the config."
            )
        selected = param_values.get("measurement_vi") or names[0]
        if selected not in names:
            raise I2ASConfigError(
                f"{type(self).name!r}: measurement_vi={selected!r} is not a "
                f"measurement VI on this station (have: {', '.join(names)})."
            )
        self._measurement_vi: str = selected

        # BaseProcedure merges class-declared param defaults (axis + system) and
        # builds the sweep array. It does NOT know the measurement params (those
        # are the selected VI's, resolved below).
        super().__init__(
            station,
            sample_info,
            data_directory,
            file_prefix,
            experiment_info,
            **param_values,
        )

        vi = station.get_vi(selected)
        vi_specs: dict[str, ParamSpec] = vi.measurement_parameters
        collisions = sorted(
            (set(type(self).parameters) | _STRUCTURAL_PARAM_NAMES) & set(vi_specs)
        )
        if collisions:
            raise I2ASConfigError(
                f"{type(self).name!r}: measurement VI {selected!r} parameter(s) "
                f"{collisions} collide with the procedure's own sweep/system "
                f"parameters. Rename the measurement VI parameter(s)."
            )

        # Merge the VI's parameter defaults with any caller-supplied values —
        # this is the merge BaseProcedure could not do for dynamic params.
        merged_meas: dict[str, Any] = {
            name: spec.default for name, spec in vi_specs.items()
        }
        merged_meas.update(
            {name: param_values[name] for name in vi_specs if name in param_values}
        )
        self._measurement_params: dict[str, Any] = merged_meas

        # Record the resolved measurement selection + params in the params dict
        # so the HDF5 metadata captures exactly what ran, and expose the
        # selected VI's columns as this instance's live-plot keys.
        self._params.update(merged_meas)
        self._params["measurement_vi"] = selected

        # ── Reading-loop resolution: up to two generic slots ─────────────────
        # Every loop level is the SAME thing: a loopable parameter (advertised
        # by a VI's reading_setters — any reading-path VI's parameters and a
        # source's current alike) plus an ordered value list. Slot 1 (labels A1, A2, …)
        # is the outer level, slot 2 (labels B1, B2, …) the inner one. A slot
        # with a single value is a STATIC setting (its setter is dispatched
        # once at initiate, no loop, no suffix); with two or more values it
        # loops and suffixes the columns. Resolved and validated here so a bad
        # selection fails at the procedure boundary, never in the tick loop.
        registry = self._loopable_registry(station, selected)
        self._loop_slots: list[dict[str, Any]] = []
        used_qualified: set[str] = set()
        for slot, prefix in (("loop1", "A"), ("loop2", "B")):
            qualified = str(param_values.get(f"{slot}_parameter") or "")
            self._params[f"{slot}_parameter"] = qualified
            if not qualified:
                self._params[f"{slot}_values"] = []
                continue
            entry = registry.get(qualified)
            if entry is None:
                raise I2ASConfigError(
                    f"{type(self).name!r}: {slot}_parameter={qualified!r} is "
                    f"not a loopable parameter of this station "
                    f"(loopable: {', '.join(registry) or 'none'})."
                )
            if qualified in used_qualified:
                raise I2ASConfigError(
                    f"{type(self).name!r}: {qualified!r} is selected in both "
                    f"loop slots; each parameter may be looped once."
                )
            used_qualified.add(qualified)
            slot_vi_name, param_name, spec, setter = entry
            if spec.choices is not None:
                # Enumerated parameter: values come from the pick checkboxes,
                # kept in choices order. An unknown pick is a form/config
                # error — fail loudly rather than silently dropping it.
                valid_picks = {
                    f"{slot}_pick_{value}" for value in spec.choices.values()
                }
                for key, value in param_values.items():
                    if key.startswith(f"{slot}_pick_") and value and key not in valid_picks:
                        raise I2ASConfigError(
                            f"{type(self).name!r}: {key!r} does not name a "
                            f"choice of {qualified!r} "
                            f"(have: {', '.join(map(str, spec.choices.values()))})."
                        )
                values = [
                    value
                    for value in spec.choices.values()
                    if param_values.get(f"{slot}_pick_{value}")
                ]
                self._params.update(
                    {
                        f"{slot}_pick_{value}": (value in values)
                        for value in spec.choices.values()
                    }
                )
            else:
                values = self._parse_loop_values(
                    param_name, spec, str(param_values.get(f"{slot}_values") or "")
                )
            if not values:
                raise I2ASConfigError(
                    f"{type(self).name!r}: the reading loop over {qualified!r} "
                    f"has no values selected."
                )
            labels = (
                [f"{prefix}{i}" for i in range(1, len(values) + 1)]
                if len(values) >= 2
                else []
            )
            self._loop_slots.append(
                {
                    "qualified": qualified,
                    "vi_name": slot_vi_name,
                    "param": param_name,
                    "setter": setter,
                    "values": values,
                    "labels": labels,
                    # Which loop_shape axis this slot occupies: 0 = loop1
                    # (outer), 1 = loop2 (inner) — see measure()/_loop_shape.
                    "axis": 0 if slot == "loop1" else 1,
                }
            )
            self._params[f"{slot}_values"] = list(values)

        # The levels that actually loop (>=2 values), in slot order.
        self._loop_levels: list[dict[str, Any]] = [
            s for s in self._loop_slots if s["labels"]
        ]

        # Live-plot / data keys: the plain measurement column names (the
        # VI's mean/error/array triple names), minus the raw sample arrays —
        # those are saved but never offered as a plot axis. The reading loop
        # no longer suffixes column names: every measurement column carries a
        # real (n_loop1, n_loop2) axis instead (see DataSchema).
        self.measurement_data_keys = [
            key
            for key in list(vi.measurement_data_keys) + list(vi.measurement_scalar_columns)
            if not key.endswith("_array")
        ]

        # Externally configured provenance hook (see MeasurementInstrumentBase's
        # "Externally configured instruments" standard): set once measure()
        # has checked the selected VI for a last_settings_snapshot attribute,
        # so that check runs at most once per run — see measure().
        self._snapshot_recorded: bool = False

    # ------------------------------------------------------------------
    # The axis column (what _axis_readback's value is stored under)
    # ------------------------------------------------------------------

    @classmethod
    def axis_data_key(cls) -> str:
        """Return the data column holding the axis read-back at each point.

        Defaults to the declared ``sweep_axis``'s ``data_key``, which is the
        answer for any procedure that sweeps a ramped setpoint. A procedure
        whose axis is NOT a setpoint — elapsed time, say — declares
        ``sweep_axis = None`` (so the GUI renders no linear/segments/CSV
        shape editor, which would be meaningless) and overrides this with
        its own column name.

        Returns:
            The sweep-column name for this procedure's axis.

        Raises:
            NotImplementedError: If the procedure declares neither a
                ``sweep_axis`` nor an override — the run would otherwise
                write its axis read-back into a column no schema declares.
        """
        if cls.sweep_axis is not None:
            return cls.sweep_axis.data_key
        raise NotImplementedError(
            f"{cls.__name__} declares no sweep_axis; it must override "
            f"axis_data_key() with the column its _axis_readback() fills."
        )

    # ------------------------------------------------------------------
    # Reading-loop plumbing (shared by __init__, the form, and the plots)
    # ------------------------------------------------------------------

    @staticmethod
    def _loopable_registry(
        station: Station, measurement_vi_name: str
    ) -> dict[str, tuple[str, str, ParamSpec, str]]:
        """Collect every loopable parameter the station's reading path offers.

        A loopable parameter is any ``reading_setters`` entry of a VI in the
        reading path — today the selected measurement VI. A measurement VI
        without setters yields an empty registry: no Reading loop group, no
        loop. The registry is keyed by VI so a reading path that grows a
        second participating VI needs no change here.

        Args:
            station: The active Station.
            measurement_vi_name: The selected measurement VI's name.

        Returns:
            Ordered ``{"vi_name.param": (vi_name, param, spec, setter)}``.
        """
        vi_names = [measurement_vi_name]
        registry: dict[str, tuple[str, str, ParamSpec, str]] = {}
        for vi_name in vi_names:
            vi = station.get_vi(vi_name)
            specs = vi.reading_parameters
            for param_name, setter in vi.reading_setters.items():
                spec = specs.get(param_name)
                if spec is None:  # a setter with no matching parameter spec
                    continue
                registry[f"{vi_name}.{param_name}"] = (
                    vi_name,
                    param_name,
                    spec,
                    setter,
                )
        return registry

    @staticmethod
    def _parse_loop_values(param_name: str, spec: ParamSpec, text: str) -> list:
        """Parse a comma-separated loop-value list against a parameter's spec.

        Each entry is converted with the spec's ``type`` and checked against
        its ``min``/``max`` bounds or ``choices``, so a loop value obeys
        exactly the constraints the single-value form field would enforce.

        Args:
            param_name: The looped parameter's name (for error messages).
            spec: The looped parameter's ``ParamSpec``.
            text: The user-entered comma-separated list (e.g. ``"1e-6, -1e-6"``).

        Returns:
            The parsed values, in entry order (empty for blank *text*).

        Raises:
            I2ASConfigError: If the parameter is a bool (not loopable), an
                entry does not parse as the spec's type, or a parsed value
                violates the spec's bounds/choices.
        """
        if spec.type is bool:
            raise I2ASConfigError(
                f"reading loop: parameter {param_name!r} is a bool and cannot "
                f"be looped over a value list."
            )
        values: list = []
        for entry in (e.strip() for e in text.split(",")):
            if not entry:
                continue
            try:
                value = spec.type(entry)
            except (TypeError, ValueError) as exc:
                raise I2ASConfigError(
                    f"reading loop: {entry!r} is not a valid "
                    f"{spec.type.__name__} for parameter {param_name!r}."
                ) from exc
            if spec.choices is not None and value not in spec.choices.values():
                raise I2ASConfigError(
                    f"reading loop: {value!r} is not one of {param_name!r}'s "
                    f"allowed choices {list(spec.choices.values())}."
                )
            if spec.min is not None and value < spec.min:
                raise I2ASConfigError(
                    f"reading loop: {value!r} is below {param_name!r}'s "
                    f"minimum {spec.min!r}."
                )
            if spec.max is not None and value > spec.max:
                raise I2ASConfigError(
                    f"reading loop: {value!r} is above {param_name!r}'s "
                    f"maximum {spec.max!r}."
                )
            values.append(value)
        return values

    # ------------------------------------------------------------------
    # Parameter groups: measurement-method selection + selected VI's params
    # ------------------------------------------------------------------

    @classmethod
    def get_param_groups(
        cls, station: Station, selections: Mapping[str, Any] | None = None
    ) -> list[ParamGroup]:
        """Build the form: the static Sweep/System groups + measurement select.

        Appends the dynamic groups after the class's static groups: a
        "Measurement method" group with the structural ``measurement_vi``
        selector, a single "Reading loop" group holding the two generic loop
        slots (each a ``{slot}_parameter`` drop-down over every loopable
        parameter the station's reading path advertises, plus that
        parameter's values input — per-choice ``{slot}_pick_{value}``
        checkboxes when enumerated, a ``{slot}_values`` text field otherwise),
        and a group carrying the selected VI's own ``measurement_parameters``.
        Nothing loopable on this station means no Reading loop group at all.

        Args:
            station: The active Station (supplies the measurement VIs).
            selections: Current structural-param values; ``measurement_vi`` picks
                the VI whose parameters group is shown (defaults to the first),
                the ``loopN_*`` values carry the reading loop.

        Returns:
            The ordered ``ParamGroup`` list the GUI renders.

        Raises:
            I2ASConfigError: If the station has no measurement VIs, or a
                measurement param name collides with a procedure param.
        """
        names = station.measurement_vi_names()
        if not names:
            raise I2ASConfigError(
                f"{cls.name!r} needs a measurement VI, but this station has none."
            )
        selections = selections or {}
        selected = selections.get("measurement_vi")
        if selected not in names:
            selected = names[0]

        # Static Sweep/System groups (Measurement is empty for a generic proc).
        groups = super().get_param_groups(station, selections)

        # (a) The measurement-method selector. Labels are the VI's SHORT
        # selector_label (keeps the drop-down / column narrow); the
        # collected/mapped value is the bare VI name. The GUI carries the
        # vi_name in a per-item tooltip for disambiguation when two VIs share a
        # label. Falls back to the VI name if two selector labels collide (a
        # dict key would otherwise be lost).
        choices: dict[str, str] = {}
        for name in names:
            label = station.measurement_selector_label(name)
            key = label if label and label not in choices else name
            choices[key] = name
        select_spec = ParamSpec(
            type=str,
            default=selected,
            choices=choices,
            structural=True,
            description="Measurement method to run at each sweep point",
        )
        groups.append(
            ParamGroup(
                key="measurement_select",
                title="Measurement method",
                params={"measurement_vi": select_spec},
            )
        )
        vi = station.get_vi(selected)

        # (b) The reading loop, rendered ABOVE the VI's own parameter group.
        # Two generic slots, each a loopable parameter (anything the station's
        # reading path advertises via reading_setters) plus its values input. The
        # values input is spec-driven: an enumerated parameter renders one
        # checkbox per choice ("{slot}_pick_{value}"), a free parameter a
        # comma-separated text field ("{slot}_values"). Everything here is
        # structural — changing any of it changes which columns the run
        # produces. Nothing loopable on this station -> no group, zero change.
        registry = cls._loopable_registry(station, selected)
        if registry:
            loop_group_params: dict[str, ParamSpec] = {}
            slot_choices: dict[str, str] = {"Off": ""}
            for qualified, (slot_vi, param_name, spec, _setter) in registry.items():
                display = (
                    f"{param_name} ({spec.unit})"
                    if spec.unit
                    else f"{param_name} ({slot_vi})"
                )
                if display in slot_choices:
                    display = qualified
                slot_choices[display] = qualified
            for slot, ordinal in (("loop1", "1"), ("loop2", "2")):
                selected_slot = selections.get(f"{slot}_parameter")
                if selected_slot not in registry:
                    selected_slot = ""
                loop_group_params[f"{slot}_parameter"] = ParamSpec(
                    type=str,
                    default=selected_slot,
                    choices=dict(slot_choices),
                    structural=True,
                    description=(
                        f"Loop {ordinal} parameter, repeated at every sweep "
                        f"point (slot 1 is the outer level)"
                    ),
                )
                if not selected_slot:
                    continue
                _vi_name, param_name, spec, _setter = registry[selected_slot]
                if spec.choices is not None:
                    for choice_label, value in spec.choices.items():
                        loop_group_params[f"{slot}_pick_{value}"] = ParamSpec(
                            type=bool,
                            default=bool(selections.get(f"{slot}_pick_{value}")),
                            structural=True,
                            description=f"Include {choice_label}",
                        )
                else:
                    loop_group_params[f"{slot}_values"] = ParamSpec(
                        type=str,
                        default=str(selections.get(f"{slot}_values") or ""),
                        structural=True,
                        widget_hint="array",
                        description=(
                            "Comma-separated values; one value sets it once, "
                            "two or more loop it at every sweep point"
                        ),
                    )
            groups.append(
                ParamGroup(
                    key="reading_loop",
                    title="Reading loop",
                    params=loop_group_params,
                )
            )

        # (c) The selected VI's own parameters. Renders active_measurement_
        # parameters, not the full measurement_parameters: the externally-
        # configured standard's self-description (MeasurementInstrumentBase)
        # hides the parameters the external tool owns so the operator is
        # never shown an inert excitation/analysis/routing knob. The title
        # carries a marker in that mode so the operator sees WHY they're gone.
        vi_specs: dict[str, ParamSpec] = dict(vi.active_measurement_parameters)
        collisions = sorted(
            (set(cls.parameters) | _STRUCTURAL_PARAM_NAMES) & set(vi_specs)
        )
        if collisions:
            raise I2ASConfigError(
                f"{cls.name!r}: measurement VI {selected!r} parameter(s) "
                f"{collisions} collide with the procedure's own parameters."
            )
        title = station.measurement_label(selected)
        if vi.configured_externally:
            title = f"{title} — externally configured"
        groups.append(
            ParamGroup(
                key=f"measurement:{selected}",
                title=title,
                params=vi_specs,
            )
        )

        return groups

    @classmethod
    def live_plot_measurement_keys(
        cls, station: Station, selections: Mapping[str, Any] | None = None
    ) -> list[str]:
        """Return the selected measurement VI's plottable scalar column names.

        Keys stay the PLAIN column names even when a reading loop is defined:
        the per-plot Loop selectors (see ``live_plot_loop_labels``) pick
        which ``(loop1, loop2)`` grid index is drawn, and the panel indexes
        into that axis at draw time. Raw-sample-array columns (``*_array``)
        are excluded — they are saved but never a plot axis; only the
        mean/error/other scalar columns are offered, including every raw
        diagnostic block's channel labels (see
        ``_raw_block_channel_labels``) — each is a row-mean-reduced scalar
        column carrying the real loop grid exactly like any other, per the
        "Raw diagnostic blocks" standard's plot-column extension
        (``MeasurementInstrumentBase``).
        """
        names = station.measurement_vi_names()
        if not names:
            return []
        selections = selections or {}
        selected = selections.get("measurement_vi")
        if selected not in names:
            selected = names[0]
        vi = station.get_vi(selected)
        return [
            key
            for key in (
                list(vi.measurement_data_keys)
                + list(vi.measurement_scalar_columns)
                + cls._raw_block_channel_labels(vi)
            )
            if not key.endswith("_array")
        ]

    @classmethod
    def live_plot_loop_labels(
        cls, station: Station, selections: Mapping[str, Any] | None = None
    ) -> tuple[dict[int, str] | None, dict[int, str] | None]:
        """Return the two Loop-selector label maps for the current selections.

        One entry per loop slot, in slot order (loop1 = axis 0, loop2 = axis
        1). ``(None, None)`` when the station's reading path offers nothing
        loopable (selectors hidden). Per slot: ``{}`` when the slot is off,
        static (one value), or its values are empty/invalid (selector
        visible, disabled; construction refuses a bad list loudly); otherwise
        an ordered ``{axis_index: display}`` map like ``{0: "A1 =
        Mux-Ch1", 1: "A2 = Mux-Ch2"}`` — the key is the 0-based index into
        that axis's ``(n_loop1, n_loop2)`` grid dimension (see
        ``DataSchema``/``_loop_shape``), the value a human-readable label.
        All loop params are ``structural=True`` precisely so their values
        reach ``selections`` and their changes re-derive the selectors.
        """
        names = station.measurement_vi_names()
        if not names:
            return (None, None)
        selections = selections or {}
        selected = selections.get("measurement_vi")
        if selected not in names:
            selected = names[0]
        registry = cls._loopable_registry(station, selected)
        if not registry:
            return (None, None)

        maps: list[dict[int, str]] = []
        for slot, prefix in (("loop1", "A"), ("loop2", "B")):
            qualified = selections.get(f"{slot}_parameter") or ""
            entry = registry.get(qualified)
            if entry is None:
                maps.append({})
                continue
            _vi_name, param_name, spec, _setter = entry
            if spec.choices is not None:
                values = [
                    value
                    for value in spec.choices.values()
                    if selections.get(f"{slot}_pick_{value}")
                ]
            else:
                try:
                    values = cls._parse_loop_values(
                        param_name,
                        spec,
                        str(selections.get(f"{slot}_values") or ""),
                    )
                except I2ASConfigError:
                    values = []
            if len(values) < 2:
                maps.append({})
                continue
            maps.append(
                {
                    index: (
                        f"{prefix}{index + 1} = {value:g}"
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        else f"{prefix}{index + 1} = {value}"
                    )
                    for index, value in enumerate(values)
                }
            )
        return (maps[0], maps[1])

    # ------------------------------------------------------------------
    # Axis-specific hooks — implemented by concrete axis procedures
    # ------------------------------------------------------------------

    def _initial_system_targets(self) -> dict[str, Target]:
        """Return the system ramp targets for ``initiate()`` (first sweep point)."""
        raise NotImplementedError

    def _step_targets(self, index: int) -> dict[str, Target]:
        """Return the system ramp targets for sweep point *index*."""
        raise NotImplementedError

    def _standby_targets(self) -> dict[str, Target]:
        """Return the system ramp targets for ``standby()`` (park the system)."""
        raise NotImplementedError

    def _axis_readback(self) -> float:
        """Return the measured value of the swept quantity at the current point."""
        raise NotImplementedError

    def _initiate_wait_s(self) -> float:
        """Return the settle time (s) after the initial ramp."""
        raise NotImplementedError

    def _step_wait_s(self) -> float:
        """Return the settle time (s) after each sweep-step ramp."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared four-method loop
    # ------------------------------------------------------------------

    @property
    def _loop_shape(self) -> tuple[int, int]:
        """Return ``(n_loop1, n_loop2)`` — each ``>= 1``.

        ``1`` means that slot is off or static (a single value, dispatched
        once at ``initiate()``); any larger count is the number of values the
        reading loop dispatches on that axis. Derived from
        ``self._params["loop1_values"]`` / ``["loop2_values"]``, which
        ``__init__`` always populates (empty when the slot is off).
        """
        n1 = len(self._params.get("loop1_values") or []) or 1
        n2 = len(self._params.get("loop2_values") or []) or 1
        return (n1, n2)

    @staticmethod
    def _raw_block_channel_labels(vi: Any) -> list[str]:
        """Flatten every raw block's channel label list, in block-then-column order.

        Shared by ``_build_data_schema`` (collision-checked column
        derivation), ``measure()`` (per-channel grid population), and
        ``live_plot_measurement_keys`` (plot key exposure) so the three stay
        in lock-step — see ``MeasurementInstrumentBase``'s "Raw diagnostic
        blocks" standard's plot-column extension.
        """
        return [label for labels in vi.measurement_raw_blocks.values() for label in labels]

    def _raw_block_channel_columns(
        self, vi: Any, sweep_columns: dict[str, str]
    ) -> dict[str, str]:
        """Return ``{channel_label: "float"}`` for every raw block channel.

        Each declared raw-block channel becomes an ordinary scalar column —
        reduced from the block's row axis by ``measure()`` via ``_nanmean``,
        the same "N readings at one measurement point" treatment a
        measurement VI's ``n_readings`` already gets via
        ``mean_and_sem`` — so it is plottable and carries the real
        ``(n_loop1, n_loop2)`` loop grid like any other ``measurement_
        scalars`` entry.

        Args:
            vi: The selected measurement VI instance.
            sweep_columns: This run's sweep-column names, checked for
                collisions.

        Returns:
            ``{label: "float"}`` for every channel across every declared
            raw block.

        Raises:
            I2ASConfigError: A channel label collides with a sweep
                column, an existing ``measurement_scalar_columns`` entry,
                or another block's channel label.
        """
        columns: dict[str, str] = {}
        origin: dict[str, str] = {}
        for block_name, labels in vi.measurement_raw_blocks.items():
            for label in labels:
                if label in sweep_columns or label in vi.measurement_scalar_columns:
                    raise I2ASConfigError(
                        f"{type(self).name!r}: raw block {block_name!r} channel "
                        f"{label!r} collides with an existing sweep/scalar column."
                    )
                if label in origin:
                    raise I2ASConfigError(
                        f"{type(self).name!r}: raw block {block_name!r} channel "
                        f"{label!r} collides with block {origin[label]!r}'s "
                        f"channel of the same name."
                    )
                origin[label] = block_name
                columns[label] = "float"
        return columns

    def _build_data_schema(self, vi: Any) -> DataSchema:
        """Assemble this run's ``DataSchema`` from the axis, station, and VI.

        Sweep columns (never looped): ``unix_time`` + every system column
        (``station.last_state_flat()``, populated by the ``get_state()`` poll
        in ``initiate()``) + the axis data-key. Measurement scalars/arrays:
        the VI's ``measurement_scalar_columns`` / ``data_arrays`` for the
        selected measurement params, each carrying the real ``_loop_shape``
        axis (see ``DataSchema``) instead of suffixed column names, PLUS one
        derived scalar column per raw-block channel (see
        ``_raw_block_channel_columns``) — a row-mean reduction of the raw
        diagnostic block, so every one of its channels is independently
        plottable without changing the block's own storage shape. Raw
        diagnostic blocks themselves (see ``MeasurementInstrumentBase``'s
        "Raw diagnostic blocks" standard): the VI's ``measurement_raw_blocks``
        label lists (channel-axis length) combined with
        ``raw_block_row_counts()`` (row-axis length) for the same params.
        Image blocks (the image-block standard, ``measurement_image_blocks``)
        take the same ``measurement_blocks`` slot with their declared
        ``(height_px, width_px)``; no scalar column is derived from a frame.

        Args:
            vi: The selected measurement VI instance.

        Returns:
            The declared HDF5 layout for this run.
        """
        sweep_columns: dict[str, str] = {"unix_time": "float"}
        for key in self._station.last_state_flat():
            sweep_columns[key] = "float"
        sweep_columns[type(self).axis_data_key()] = "float"
        row_counts = vi.raw_block_row_counts(self._measurement_params)
        measurement_blocks = {
            name: (row_counts[name], len(labels))
            for name, labels in vi.measurement_raw_blocks.items()
        }
        for name, image in vi.measurement_image_blocks.items():
            if name in measurement_blocks or name in sweep_columns:
                raise I2ASConfigError(
                    f"{type(self).name!r}: image block {name!r} collides with a "
                    f"raw block or sweep column of the same name."
                )
            measurement_blocks[name] = image.shape
        measurement_scalars = {
            **dict(vi.measurement_scalar_columns),
            **self._raw_block_channel_columns(vi, sweep_columns),
        }
        return DataSchema(
            sweep_columns=sweep_columns,
            measurement_scalars=measurement_scalars,
            measurement_arrays=dict(vi.data_arrays(self._measurement_params)),
            measurement_blocks=measurement_blocks,
            loop_shape=self._loop_shape,
        )

    def planned_targets(self) -> dict[str, list[float]]:
        """Return every system setpoint this sweep would command, per VI.

        Derived from the very hooks ``initiate()`` and ``change_sweep_step()``
        dispatch from — ``_initial_system_targets()`` and ``_step_targets()``
        over every point of the built sweep — so the pre-dispatch declaration
        (see ``BaseProcedure.planned_targets()``) is the same set of numbers
        the run will actually send, and cannot drift from it. Pure: the hooks
        only read the already-built sweep array.

        Returns:
            ``{vi_name: [setpoint, ...]}`` in SI units, first the initial
            targets and then one entry per sweep point.
        """
        planned: dict[str, list[float]] = {}
        plans = [self._initial_system_targets()]
        plans.extend(self._step_targets(index) for index in range(len(self._sweep)))
        for targets in plans:
            for vi_name, target in targets.items():
                planned.setdefault(vi_name, []).append(float(target.target))
        return planned

    # ------------------------------------------------------------------
    # Probe runs and duration estimates — the measurement-VI half
    # ------------------------------------------------------------------

    def apply_probe(self, spec: ProbeSpec) -> None:
        """Reduce this run to a probe, measurement parameters included.

        Extends ``BaseProcedure.apply_probe()`` (sweep length, declared waits,
        ``run_kind``) with the two reductions that need the selected
        measurement VI: its own seconds-valued parameters (inter-reading
        delays, time constants) are capped at ``spec.max_wait_s``, and every
        repeat count is capped at ``spec.averaging`` — identified by
        declaration rather than by name, as the parameter whose value shortens
        one of the VI's declared ``data_arrays()`` lengths, so a new
        measurement VI's averaging knob is reduced with no new code here.

        Args:
            spec: The reduction to apply.

        Raises:
            TypeError: If *spec* is not a ``ProbeSpec``.
        """
        super().apply_probe(spec)
        vi = self._station.get_vi(self._measurement_vi)
        capped = self._probe_cap_waits(
            vi.measurement_parameters, self._measurement_params, spec.max_wait_s
        )
        capped += self._probe_cap_averaging(vi, spec.averaging)
        # The resolved measurement values are mirrored into _params (that is
        # what the HDF5 metadata and the run manifest record), so the mirror
        # must follow the reduction.
        self._params.update(self._measurement_params)
        if capped:
            logger.info(
                "%s: probe reduced measurement parameters %s",
                type(self).__name__,
                ", ".join(capped),
            )

    def _probe_cap_averaging(self, vi: Any, averaging: int) -> list[str]:
        """Cap every declared repeat count of *vi* at *averaging*, in place.

        Args:
            vi: The selected measurement VI.
            averaging: The maximum repeat count a probe keeps.

        Returns:
            The measurement parameter names actually lowered.
        """
        current = self._declared_array_lengths(vi, self._measurement_params)
        if not current:
            return []
        reduced: list[str] = []
        for name, value in list(self._measurement_params.items()):
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if value <= averaging:
                continue
            trial = dict(self._measurement_params)
            trial[name] = averaging
            lengths = self._declared_array_lengths(vi, trial)
            if any(lengths.get(key, length) < length for key, length in current.items()):
                self._measurement_params[name] = averaging
                current = lengths
                reduced.append(name)
        return reduced

    @staticmethod
    def _declared_array_lengths(vi: Any, params: Mapping[str, Any]) -> dict[str, int]:
        """Return the VI's declared per-array sample counts for *params*.

        Args:
            vi: The measurement VI.
            params: The measurement parameter values to declare against.

        Returns:
            ``{array_name: length}``, or ``{}`` when the VI refuses these
            values — a declaration read must never raise at its caller, which
            is estimating or reducing, not measuring.
        """
        try:
            return {str(key): int(value) for key, value in vi.data_arrays(params).items()}
        except Exception:  # noqa: BLE001 — a declaration read never raises out
            logger.debug("data_arrays() refused %r", dict(params), exc_info=True)
            return {}

    def estimate_step_seconds(self) -> StepCost:
        """Return this sweep's per-point cost: its declared waits and readings.

        Derived from the same hooks the run itself uses — ``_initiate_wait_s()``
        and ``_step_wait_s()`` for the waits, ``_loop_shape`` for how many
        readings one datapoint takes — so the estimate cannot drift from what
        the tick loop will actually wait for. The measurement time is modelled
        from the selected VI's own declarations: its declared sample count
        (``data_arrays()``) times the sum of its seconds-valued measurement
        parameters (inter-reading delay, time constant), which is a lower
        bound — instrument conversion and transfer time are not modelled, and
        the returned assumptions say so.

        Returns:
            The ``StepCost`` for one run of this instance.
        """
        vi = self._station.get_vi(self._measurement_vi)
        readings = self._loop_shape[0] * self._loop_shape[1]
        samples = max(
            self._declared_array_lengths(vi, self._measurement_params).values(),
            default=1,
        )
        delays = {
            name: float(self._measurement_params[name])
            for name, spec in vi.measurement_parameters.items()
            if spec.unit == "s"
            and spec.type in (int, float)
            and isinstance(self._measurement_params.get(name), (int, float))
            and not isinstance(self._measurement_params.get(name), bool)
        }
        per_sample_s = sum(delays.values())
        assumptions = [
            f"settle times are this run's declared waits: "
            f"{self._initiate_wait_s():g} s before the first point, "
            f"{self._step_wait_s():g} s before each later one",
        ]
        if per_sample_s > 0:
            assumptions.append(
                f"measurement time is {samples} sample(s) x "
                f"{per_sample_s:g} s of declared delay "
                f"({', '.join(sorted(delays))}) x {readings} reading(s) per "
                f"point; instrument conversion and transfer time are not "
                f"modelled"
            )
        else:
            assumptions.append(
                f"{self._measurement_vi} declares no per-sample delay, so no "
                f"measurement time is counted"
            )
        return StepCost(
            points=len(self._sweep),
            setup_s=float(self._initiate_wait_s()),
            settle_s=float(self._step_wait_s()),
            measure_s=float(readings * samples * per_sample_s),
            assumptions=tuple(assumptions),
        )

    def initiate(self) -> PhasePlan:
        """Ramp to the first sweep point, arm the selected VI, open the file.

        Returns:
            A ``PhasePlan`` with ``claim_commands`` for every claimed VI
            (``_claim_initiate_commands()``), the initial system ``targets``,
            the selected VI's arming ``Command``, and ``wait_s`` from
            ``_initiate_wait_s()``.
        """
        claim_commands = self._claim_initiate_commands()
        vi = self._station.get_vi(self._measurement_vi)
        targets = self._initial_system_targets()
        arm_command = Command(
            self._measurement_vi, "initiate_measurement", dict(self._measurement_params)
        )
        # Loop-slot setup around the arm (command order is semantically
        # meaningful). Slots on OTHER VIs dispatch their first value BEFORE
        # the measurement VI arms, so arming happens with the reading path
        # already set up — with a static (1-value) slot that
        # single dispatch is the whole story, with a looping slot measure()
        # re-dispatches per reading. Static slots on the measurement VI itself
        # dispatch AFTER the arm (their setters require an armed instrument);
        # looping measurement-VI slots need nothing here because measure()
        # sets the value before every reading.
        pre_arm: list[Command] = []
        post_arm: list[Command] = []
        for slot in self._loop_slots:
            command = Command(
                slot["vi_name"], slot["setter"], {slot["param"]: slot["values"][0]}
            )
            if slot["vi_name"] != self._measurement_vi:
                pre_arm.append(command)
            elif not slot["labels"]:
                post_arm.append(command)
        commands = (*pre_arm, arm_command, *post_arm)

        # Poll once so last_state_flat() is populated BEFORE the schema reads the
        # system columns — otherwise the columns declared here would not match
        # what measure() enriches with, and the per-point validate() would fail.
        instrument_state = self._station.get_state()
        self._data_schema = self._build_data_schema(vi)
        data_config = {
            "sweep_columns": dict(self._data_schema.sweep_columns),
            "measurement_scalars": dict(self._data_schema.measurement_scalars),
            "measurement_arrays": dict(self._data_schema.measurement_arrays),
            "measurement_blocks": dict(self._data_schema.measurement_blocks),
            "measurement_block_labels": dict(vi.measurement_raw_blocks),
            "measurement_image_blocks": {
                name: {"unit": image.unit, "description": image.description}
                for name, image in vi.measurement_image_blocks.items()
            },
            "loop_shape": list(self._data_schema.loop_shape),
        }

        self._data_manager = DataManager(
            data_directory=self._data_directory,
            procedure_name=self.name,
            file_prefix=self._file_prefix,
            procedure_params=self._params,
            sample_info=self._sample_info,
            instrument_state=instrument_state,
            # DataManager stays dict-based (contract C7): convert the typed plan
            # to plain JSON-ready dicts at this call-site boundary only.
            system_targets={
                name: dataclasses.asdict(t) for name, t in targets.items()
            },
            measurement_commands=[dataclasses.asdict(c) for c in commands],
            data_config=data_config,
            n_sweep_points=len(self._sweep),
            experiment_info=self._experiment_info,
            # A probe writes a real file, tagged as a probe (see ProbeSpec):
            # the run's own kind travels into /metadata.run_kind so a reader
            # can never mistake a reduced probe for science data.
            run_kind=self.run_kind,
        )

        logger.info(
            "%s.initiate(): %d points, measurement=%s",
            type(self).__name__,
            len(self._sweep),
            self._measurement_vi,
        )
        return PhasePlan(
            targets=targets,
            commands=commands,
            wait_s=self._initiate_wait_s(),
            claim_commands=claim_commands,
        )

    def change_sweep_step(self) -> StepPlan | None:
        """Advance to the next sweep point.

        Returns:
            A ``StepPlan`` ramping to the next point with ``_step_wait_s()``
            settle, or ``None`` when the sweep is exhausted.
        """
        self._index += 1
        if self._index >= len(self._sweep):
            return None
        return StepPlan(
            targets=self._step_targets(self._index), wait_s=self._step_wait_s()
        )

    def measure(self) -> None:
        """Run the reading loop, tag on the axis read-back, and save.

        The reading loop takes one grid of readings per datapoint: axis 1
        (outer, loop1) x axis 2 (inner, loop2), each of length declared by
        ``_loop_shape``. Before each axis-1 step it dispatches that slot's
        setter (as a ``Command`` through the Station — the same channel
        ``initiate()`` and the Orchestrator use, never a direct call on
        another VI); before each axis-2 reading it dispatches axis 2's setter.
        A slot that is off or static contributes no dispatch (its single
        value was already set at ``initiate()``) but still occupies its
        trivial length-1 axis. Every measurement column comes back as a real
        ``(n_loop1, n_loop2)`` (or ``(n_loop1, n_loop2, length)`` for an
        array) grid — column names are never suffixed.

        A raw diagnostic block (``vi.measurement_raw_blocks``) is the one
        exception to the "every column carries a real loop axis" rule above:
        with no reading loop configured (``_loop_shape == (1, 1)``), its
        grid is squeezed down to the bare per-reading block before saving —
        see ``DataSchema.measurement_blocks``'s docstring for why. Each
        block's declared channels ALSO land in the grid, one row-mean-
        reduced scalar per channel (see ``_raw_block_channel_columns``) —
        those are ordinary scalar columns and always keep their real
        ``(n_loop1, n_loop2)`` grid, never squeezed like the block itself.
        An image block (``vi.measurement_image_blocks``, the image-block
        standard) is stored exactly like a raw block — same grid, same
        squeeze — but no scalar is ever derived from a frame.

        Raises:
            RuntimeError: If called before ``initiate()``.
            DataSchemaError: If the datapoint does not match the declared schema
                (raised by ``_save_datapoint``); nothing is written.
        """
        if self._data_manager is None:
            raise RuntimeError("measure() called before initiate()")
        vi = self._station.get_vi(self._measurement_vi)

        # Externally configured provenance hook, checked at most once per run
        # (see MeasurementInstrumentBase's "Externally configured
        # instruments" standard): any measurement VI that exposes a
        # last_settings_snapshot dict attribute gets it recorded into the
        # run's HDF5 /metadata automatically; VIs without the attribute are
        # untouched (getattr default is None).
        if not self._snapshot_recorded:
            self._snapshot_recorded = True
            snapshot = getattr(vi, "last_settings_snapshot", None)
            if isinstance(snapshot, dict):
                self._data_manager.record_settings_snapshot(snapshot)

        n_loop1, n_loop2 = self._loop_shape
        channel_labels = self._raw_block_channel_labels(vi)
        block_keys = list(vi.measurement_raw_blocks) + list(vi.measurement_image_blocks)
        keys = (
            list(vi.measurement_data_keys)
            + list(vi.measurement_scalar_columns)
            + block_keys
            + channel_labels
        )
        grids: dict[str, list[list[Any]]] = {
            key: [[None] * n_loop2 for _ in range(n_loop1)] for key in keys
        }

        def _dispatch(level: dict[str, Any], value: Any) -> None:
            self._station.send_measurement_commands(
                (Command(level["vi_name"], level["setter"], {level["param"]: value}),)
            )

        axis1 = next((lv for lv in self._loop_levels if lv["axis"] == 0), None)
        axis2 = next((lv for lv in self._loop_levels if lv["axis"] == 1), None)

        for i1 in range(n_loop1):
            if axis1 is not None:
                _dispatch(axis1, axis1["values"][i1])
            for i2 in range(n_loop2):
                if axis2 is not None:
                    _dispatch(axis2, axis2["values"][i2])
                reading = vi.take_reading()
                for key, value in reading.items():
                    grids[key][i1][i2] = value
                for block_name, labels in vi.measurement_raw_blocks.items():
                    block_rows = reading[block_name]  # this reading's rows x cols
                    for col_index, label in enumerate(labels):
                        grids[label][i1][i2] = _nanmean(
                            row[col_index] for row in block_rows
                        )

        measured_data: dict[str, Any] = dict(grids)
        if (n_loop1, n_loop2) == (1, 1):
            # No reading loop configured: a raw or image block skips the
            # trivial (1, 1) axis entirely rather than carrying it like a
            # scalar/array column does (see DataSchema.measurement_blocks).
            for block_key in block_keys:
                measured_data[block_key] = grids[block_key][0][0]
        measured_data[type(self).axis_data_key()] = self._axis_readback()
        self._save_datapoint(measured_data)

    def standby(self) -> PhasePlan:
        """Close the data file and return the safe-parking plan.

        Returns:
            A ``PhasePlan`` with ``_standby_targets()``, disarming the selected
            measurement VI, with ``wait_s=0.0``.
        """
        if self._data_manager is not None:
            self._data_manager.close()
            self._data_manager = None
        return PhasePlan(
            targets=self._standby_targets(),
            commands=self._safe_off_commands(),
            wait_s=0.0,
        )

    def abort(self) -> tuple[Command, ...]:
        """Close the data file and disarm the measurement VI (+ loop safe-offs).

        Returns:
            The measurement safe-off command(s) for the Orchestrator to
            dispatch, plus each participating loop VI's ``reading_safe_off``.
        """
        super().abort()
        return self._safe_off_commands()

    def _safe_off_commands(self) -> tuple[Command, ...]:
        """Return the measurement standby command plus the loop VIs' safe-offs.

        Shared by ``standby()`` and ``abort()``: disarm the measurement VI
        and, for every OTHER VI that took part in a loop slot (static or
        looping) and declares a ``reading_safe_off`` method, dispatch it so
        nothing is left connected.

        Returns:
            Ordered ``tuple[Command, ...]`` — the measurement ``standby``
            first, then one safe-off per participating loop VI.
        """
        commands = [Command(self._measurement_vi, "standby", {})]
        seen: set[str] = set()
        for slot in self._loop_slots:
            slot_vi = slot["vi_name"]
            if slot_vi == self._measurement_vi or slot_vi in seen:
                continue
            seen.add(slot_vi)
            safe_off = self._station.get_vi(slot_vi).reading_safe_off
            if safe_off:
                commands.append(Command(slot_vi, safe_off, {}))
        return tuple(commands)
