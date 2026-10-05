# Plan: plot any monitored kind (image, trace) in the Monitor and Procedure windows

Status: implemented on branch `ccr-bf75d5b1-rqj0jp` (commit `c498249`) **before**
this plan was written down; this document records the design for audit. Audit
findings will be addressed in follow-up commits.

## 1. Goal

Plot panels in the Monitor window and the Procedure window display any kind of
data, not only scalar XY/time plots — as a framework capability, not a one-off.

* Monitor: scalar trends (existing), live 2-D images, 1-D traces as a waterfall.
* Procedure: XY plots (existing) and images of a run's image blocks.
* Declared at the source: the VI author declares the kind on `@monitored`.

User decisions already taken:

| Question | Decision |
|---|---|
| Kinds in v1 | scalar, image, waterfall (needs a 1-D `trace` source kind) |
| Polling of non-scalar fields | own, slower `period_s` per field |
| Persistence | RAM only, latest N per field |
| Procedure panel kind choice | kind selector per panel |

## 2. Non-negotiables (from CLAUDE.md philosophy)

1. No display code may affect a running experiment or the operator's ability to run one.
2. Safety readings (scalar monitor tick) must not be delayed by slow array reads.
3. Layer contracts (`pyproject.toml` C1–C27) stay intact; the GUI acts only through
   the proxy's signals/commands and reads declarations from the StatusMirror.
4. Every scalar consumer (trend history, trend checks, `Readings` event, agent
   gateway, MCP/CLI, HDF5 sweep columns via `last_state_flat`) keeps receiving
   exactly what it receives today.
5. Backward compatibility: existing VIs, saved GUI layouts and contract JSON consumers work unchanged.

## 3. Design

### 3.1 Declaration (L0 foundation, `core/decorators.py`, contract C1: imports nothing)

```python
@monitored(unit=..., description=..., kind="scalar" | "image" | "trace",
           shape=(h, w) | (n,), period_s=1.0, axis=(start, stop, unit))
```

* `MONITORED_KINDS = {"scalar": None, "image": 2, "trace": 1}` (kind → ndim).
* Validation at decoration time (fails at import): unknown kind; array kind without
  shape / wrong ndim / non-positive or non-int sizes; non-positive period; `axis`
  on a non-trace; any of shape/period/axis on a scalar.
* Getters: `get_monitored_kind/shape/period_s/axis`; `get_monitored_methods(obj, kinds=None)`.
* `DEFAULT_ARRAY_PERIOD_S = 1.0`.

### 3.2 VI (L1, `virtual_instruments/base.py`)

* `__init_subclass__` wrapping copies the four new markers.
* `get_state()` polls **scalar kinds only**.
* New `read_monitored_array(name) -> ndarray | None`: calls the method, `None`
  means "no value yet", else converts to a fresh float64 array and checks the
  declared shape (ValueError on mismatch).

### 3.3 Station (`core/station.py`)

* New `poll_monitored_arrays(now=None) -> {vi: {field: ndarray}}`: per-(vi, field)
  monotonic schedule; skips VIs with a non-zero comm error count; catches
  `I2ASCommunicationError` (warning) and any other `Exception` (error logged once
  per field) — display-only reads never change error counters or conditions.
* `last_state_flat()`, `get_state()` unchanged in behaviour (arrays never present).
* `_monitored_infos()` fills `MonitoredInfo.kind/shape/period_s/axis`.

### 3.4 Orchestrator / proxy

* New signal `monitored_arrays_updated(dict)`, re-exposed 1:1 on `OrchestratorProxy`.
* In `_tick_body`, inside the monitoring branch, **after** the scalar poll and its
  emit: if `self._procedure is None and state == IDLE`, call
  `poll_monitored_arrays()` and emit when non-empty.
* Not part of the typed event stream (`Readings`), not JSON-serialised.

### 3.5 Contract (`core/events.py`, `capability_manifest.py`)

* `MonitoredInfo` gains `kind="scalar"`, `shape=()`, `period_s=None`, `axis=None`
  (defaults keep older producers valid; JSON lists coerced back to tuples).
* Manifest JSON schema `$defs/monitored` gains the four fields (required).

### 3.6 GUI

* `gui/image_view.py` — `ImageView`: the single 2-D renderer (pyqtgraph
  `ImageItem` row-major + `ColorBarItem`, viridis, auto-levels on finite pixels,
  optional physical rect, square pixels optional).
* `gui/monitor_history.py` — `ArrayHistory` (Qt-free): flat keys
  `{vi}_{field}`, deque of `(timestamp, array)` per key, `max_entries=600`.
* `gui/array_plot_panels.py` — plot-panel protocol (`kind`, `panel_id`,
  `remove_requested(str)`, `refresh()`, `selected_key()/set_selected_key()`,
  `settings_entry()/apply_settings_entry()`); `ImagePlotPanel` (newest frame) and
  `WaterfallPlotPanel` (traces in a 1 min–1 h window stacked, newest on top,
  y = seconds ago, x = declared axis or sample index).
* `TrendPlotPanel` implements the same protocol (kind `"trend"`).
* `TrendsQuadrant` → kind registry `_PANEL_KINDS` (trend/image/waterfall), one Add
  button per kind (array kinds hidden when the station declares none), shared
  4-panel cap, kind-aware persisted layout under the existing QSettings key
  (entries without `kind` restore as trends; unknown/unavailable kinds skipped),
  `on_arrays_updated()` with per-panel exception isolation.
* `MonitorWindow` builds `array_fields` from `StatusMirror.station_info()` and
  routes `monitored_arrays_updated` through the window (teardown-race rule).
* `InstrumentPanel` shows array fields as a static descriptor
  (`"image 128×128 — see Plots"`), not a live value.
* Procedure window: `BaseProcedure.live_plot_image_blocks()` (default `{}`),
  overridden in `SweepMeasureProcedure` from the selected VI's
  `measurement_image_blocks`. `LivePlotPanel` gains a per-panel kind selector
  (XY / Image; Image disabled with no blocks), image-block selector, Point
  selector (0 = latest, follows the run), and reuses the Loop selectors via
  `data_reader.select_frame`. Frames come from the existing `measurement_ready`
  datapoint (already carried the image block); no new engine path.

### 3.7 Shipped example

`CameraMeasurementVI.last_frame` (image 128×128) and `last_row_profile`
(trace 128, axis px), both answered from the cached `read_now()` reading — the
monitor never triggers an exposure.

## 4. Tests

* `tests/test_monitored_kinds.py` — declaration validation, wrapping, kind
  filtering, scalar-only `get_state`, shape enforcement, `None`, Station schedule,
  failure isolation, stale skip, `MonitoredInfo` JSON round trip & old-producer
  default, camera example, Orchestrator emits only when IDLE.
* `tests/test_array_plot_panels.py` — ArrayHistory, image/waterfall panels,
  quadrant kinds/cap/grid/arrays routing/persistence/legacy layouts, LivePlotPanel
  XY↔Image, Point & Loop selection, clear.
* `tests/test_gui.py` — imaging-station procedure window draws `frame` in Image mode.
* Existing suites updated for renamed quadrant internals; conformance allowlist
  gains the new signal.

## 5. Known trade-offs / open questions for the audit

1. Monitor previews pause during any run (all VIs, not just the run's).
2. Arrays are polled on the instrument thread; a slow read delays the *next*
   scalar tick by its duration (only while IDLE).
3. Waterfall assumes near-uniform row spacing (rows drawn evenly, not at true times).
4. Procedure image mode keeps every datapoint's frame in `_datapoints` (already
   true before this change — frames were in the dict, just not drawn).
5. No XY "latest trace" snapshot panel on the Monitor (not requested).
6. ArrayHistory memory: 600 × 128 KiB = 75 MiB worst case per image key.
7. No disk persistence of arrays (by decision).
