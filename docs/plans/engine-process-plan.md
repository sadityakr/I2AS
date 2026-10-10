# Plan: run the instruments in their own process (engine process)

Status: **plan, under audit.** Nothing here is implemented.

## 1. Problem

The operator reports that the Monitor window stops responding while a sweep
point is being measured on real hardware: plot axes cannot be changed until
the point ends. The operator's stated goal is that the GUI and the instruments
run in **different processes**.

### 1.1 What we measured

`threaded` mode (the default) already puts the Station and the Orchestrator on
their own `QThread`, and `tests/test_instrument_thread.py::
test_the_client_event_loop_keeps_running_through_a_slow_measure` proves the
window stays live while `measure()` blocks for 2 s **in `time.sleep`**. A probe
(same fixture, 2 s `measure()`, 50 ms heartbeat timer on the GUI thread) shows
why that is not enough:

| What `measure()` does for 2 s | GUI timer firings in 1.5 s | Longest GUI stall |
|---|---|---|
| `time.sleep` (releases the GIL) | 29 | 0.05 s |
| pure-Python busy loop (GIL switches every 5 ms) | 32 | 0.05 s |
| one long C call that keeps the GIL | 5 | **2.17 s** |

Every Python thread in the process shares one interpreter lock. A driver call
that holds it — a vendor SDK or C extension, a ctypes `PyDLL` binding, some GPIB
bindings, a large non-releasing numpy/C step — freezes the GUI thread for the
whole call. No thread-level change can fix that; a second **process** can.

### 1.2 What a process split does and does not fix

* **Fixes:** a GUI freeze caused by the engine (the GIL, a CPU-heavy
  measurement, the engine allocating or computing a lot). It also turns a GUI
  crash, a GUI out-of-memory or a hung widget into an event the experiment
  survives, which is the CLAUDE.md philosophy ("no user or agent code may ever
  affect a running experiment") enforced by the operating system.
* **Does not fix:** a stall in monitor *data*. While `measure()` runs, the
  engine does not tick, so the scalar readings, safety checks and queued
  commands wait for the point to end, in any process layout. Section 9 tracks
  this separately as the **interruptible point**.

## 2. Non-negotiables

1. The single hardware thread standard holds unchanged: one thread in one
   process (the engine process) touches drivers, VIs, the Station, the
   Orchestrator and the DataManager.
2. Safety never depends on the GUI. Interlocks, holds, emergency standby and
   the tick run in the engine process. A lost GUI is never a reason to touch
   an instrument.
3. The control contract (`core/events.py`) stays the one boundary. Everything
   that crosses is a contract type or a declared passthrough payload. No pickle
   of arbitrary objects, and no callables.
4. Every existing client keeps its surface: widgets written against
   `OrchestratorProxy`, the session layer, the agent gateway, MCP, ctl.
5. The suite runs in every supported mode, as it does today for inline and
   threaded.
6. Windows is the primary lab platform. Every mechanism must work with spawn
   (no fork) and Windows named pipes.

## 3. Target architecture

```
┌──────────── GUI process (i2as.main) ─────────────┐     ┌──── engine process (python -m i2as.engine) ────┐
│ MonitorWindow, ProcedureWindow, dialogs          │     │ EngineServer (QLocalServer, one client)        │
│ session layer: ExperimentManager, RunQueue,      │     │ InstrumentHost (threaded, as today)            │
│   ELN, gateway, analysis runner                  │     │   Station, Orchestrator, DataManager,          │
│ StatusMirror  ◄── events/passthrough frames ─────┼─────┤   TrendCheckRunner, run builder (catalog)      │
│ RemoteOrchestratorProxy ── commands/asks ───────►│     │ engine log file + log forwarding               │
│ StationView (declaration-only)                   │     │ session lock is NOT here (GUI owns sessions)   │
└──────────────────────────────────────────────────┘     └────────────────────────────────────────────────┘
```

* The **engine process** is today's `InstrumentHost` in `threaded` mode,
  unchanged, with an `EngineServer` on its client side in place of the GUI. The
  GUI's `OrchestratorProxy` methods map 1:1 onto frames, and the existing
  `ThreadBridge` already makes every call either posted or asked with a bound.
* The **GUI process** holds a `RemoteOrchestratorProxy`: the same class surface
  (signals, typed command methods, mirror reads), with a socket transport
  where the bridge used to be.
* A new mode value, `process`, joins `inline` and `threaded` in
  `instrument_host.resolve_mode` (`monitor.yaml` `instrument_process: true`, or
  `I2AS_INSTRUMENT_PROCESS=1`). Default stays `threaded` until Phase 5 flips it.

## 4. Phase 1: make the boundary a pure data boundary (one process, no IPC yet)

This is the work that makes Phases 2 to 5 mechanical. Everything here lands and
ships while still running in `threaded` mode.

### 4.1 No Station in the GUI process: `StationView`

Today five GUI modules (`monitor_window`, `procedure_window`,
`procedure_params_panel`, `queue_panel`, `trends_quadrant`) and the session
layer's `RunQueueHost` hold the live `Station`. C19 forbids only the *import*,
behind `TYPE_CHECKING`. They read it from the GUI thread, which already strains
the single-thread standard: `get_vi_names`, `get_vi_type`, `offline_vi_names`,
`get_offline_info`, `last_state_flat` (keys only), `envelope_variables`,
`nominal_ramp_rates`, and through procedure class hooks
(`get_param_groups`, `declaration_form`, `live_plot_measurement_keys`,
`live_plot_image_blocks`, `_role_param_specs`) the VIs' declared attributes
(`measurement_parameters`, `measurement_data_keys`, `measurement_image_blocks`,
`measurement_raw_blocks`, …).

* Define `core/station_view.py`: a `typing.Protocol` naming exactly that
  read-only surface. Get the list by instrumenting a `Station` wrapper that
  records every attribute the GUI and session suites touch, not by guessing.
* `DeclaredStation`: an implementation built **only** from a
  `StationDeclaration` snapshot. That is a new contract type, a JSON superset of
  `StationInfo` that also carries per-VI measurement declarations, offline info,
  envelope variables, nominal ramp rates and the flat state key list. The engine
  emits it with `StationInfo` on every rebuild, and `StatusMirror` holds it.
* Procedure class hooks are typed against `StationView`, not `Station`.
* **Conformance test, the safety net of this phase:** for every shipped
  config (`sim_cryostat`, `sim_imaging`) and every procedure in the catalog,
  each `StationView` method and each procedure hook returns identical results
  on the live `Station` and on `DeclaredStation`, including after a VI goes
  offline. The test is parameterised over the protocol's own member list, so
  adding a member without coverage fails.
* The GUI and `RunQueueHost` are handed a `DeclaredStation`. C19's five
  `ignore_imports` entries are deleted. New contract C28: `i2as.gui` and
  `i2as.session` never import `i2as.core.station` (typing included). They
  import `i2as.core.station_view`.

### 4.2 Runs cross as specs, never as objects

* `run_procedure(procedure)` and `queue_procedure(procedure)` take built
  objects. They become `run_spec(RunSpec)` (a contract command carrying class
  name plus JSON params, which the agent path already uses), and the engine
  builds it with the existing `build_run`. The built-object forms remain only
  for engine-side tests.
* `ProcedureWindow`'s "Run now" builds a `RunSpec` instead of calling
  `build_procedure(…, station=…)`.
* **Validation** (`validate_run`, the form's live findings) runs a *headless
  build* today. It moves to the declaration side: `validate_run` runs against
  `DeclaredStation`. That works only if a procedure's `__init__` reads
  declarations and never performs I/O, which is a rule to state and check: a
  conformance test constructs every catalog procedure on `DeclaredStation`
  with its defaults and asserts no VI method other than the declared readers
  is called. The engine re-validates on `run_spec` regardless (fail closed:
  any build error becomes a refused `Verdict`).

### 4.3 Every payload is codec-safe

* Define `core/wire.py`: one codec for everything that crosses. Contract
  dataclasses go through their existing JSON forms. Passthrough signal payloads
  (`states_updated`, `measurement_ready`, `monitored_arrays_updated`,
  `run_started`/`run_finished`, `operational_status`, `ramps_updated`, …) are
  dicts/lists of JSON scalars plus `numpy.ndarray`. Arrays go as a typed
  binary sidecar (dtype, shape, C-order bytes). NaN/inf are preserved. Unknown
  types are a hard error in tests and a logged drop in production, never a
  pickle.
* Phase 1 runs **every** threaded-mode emission through
  `decode(encode(x))` under a debug flag. The GUI suite runs with the flag on,
  which proves the codec covers the real traffic before any socket exists.

### 4.4 The one engine→client question

The engine's only `ask()` is the queue pull, `take_next_spec`. It keeps its
shape (request plus bounded wait plus default `None` = "no next run"), carried
as a frame in Phase 2. Rule written down: no new `ask()` may be added without
an entry in this plan's successor document, because every ask is a place where
the engine waits on the GUI.

## 5. Phase 2: the transport

* **Socket:** `QLocalServer`/`QLocalSocket` (named pipe on Windows, Unix
  socket elsewhere), the mechanism the agent gateway already uses. Server in
  the engine process, exactly one client. The name is per-launch random, and
  the client authenticates with a 32-byte token handed over on the child's
  stdin. Never argv, which other local users can read. Never a file.
* **Framing:** length-prefixed frames: header JSON
  (`{"t": kind, "seq": n, "name": …}`) plus optional binary sidecars. Frame
  kinds: `cmd` (client→engine: a `Command`, a forwarded call such as
  `publish_queue`/`set_run_folder`, or a queue-snapshot push), `ask`/`answer`
  (engine↔client, with id), `sig` (engine→client: a passthrough signal),
  `evt` (engine→client: a contract event, including `Verdict`), `hb`
  (both ways), `log` (engine→client log records at WARNING and above),
  `bye`.
* **Engine side:** `EngineServer` reads frames on the engine process's main
  thread and calls the in-process `OrchestratorProxy` it holds. That proxy
  already posts to the instrument thread through the `ThreadBridge`. So the
  engine process is itself a two-thread process: the socket thread never
  touches the Station, and a GIL-holding driver call delays **socket I/O in
  the engine process only**, never the GUI's event loop.
* **Backpressure:** large-and-frequent payloads (`monitored_arrays_updated`,
  `measurement_ready` with image blocks) are *latest-wins* per key on the
  engine's outbound queue. If the client is behind, an older unsent frame of
  the same key is replaced. Events (`Verdict`, `RunStarted`, `Datapoint`, …)
  are never dropped. An outbound queue over a byte cap (e.g. 256 MiB)
  disconnects the client with a logged reason rather than growing without
  bound.
* **Flow on the client:** `RemoteOrchestratorProxy` re-emits each `sig`/`evt`
  as the same Qt signal on the GUI thread, so every connected slot is
  untouched. Reads come from the `StatusMirror`, as today.

## 6. Phase 3: lifecycle and failure

The decisions marked ⚑ belong to the operator; the plan's recommendation is
given.

* **Launch:** the GUI starts the engine with `QProcess` (`sys.executable -m
  i2as.engine --config … `), passes the token on stdin, and waits (bounded,
  120 s as today's build timeout) for a `ready` frame carrying the primed
  status snapshot and `StationDeclaration`. A failed build is reported with
  the engine's own error text, as today.
* **Heartbeat:** `hb` both ways every 1 s. Three missed = peer considered lost.
* **GUI lost** (crash, kill, hang) ⚑ *Recommended:* the engine finishes the
  current run normally (its data is written by the engine) and does **not**
  pull another queued run, because the queue lives in the GUI. It then holds
  at IDLE with monitoring and safety running, and accepts a new client that
  presents the token. The engine never ramps anything down just because the
  GUI went away; an emergency is still an emergency.
* **Engine lost** (crash, driver segfault): the GUI shows a non-dismissible
  `engine_lost` alert, marks every reading stale, refuses commands with a
  clear verdict, and offers **Restart engine**. The run record is filed as
  interrupted by the session layer on the same path a crash uses today. The
  GUI is fully usable for data, notebook and analysis meanwhile, which is a new
  capability.
* **Reattach** ⚑ *Recommended for v2, not v1:* a freshly started GUI finds a
  running engine (descriptor in the log dir, like `gateway.json`) and attaches
  instead of starting a second one. v1 only reattaches the GUI that started
  the engine, after a socket drop.
* **Quit:** the GUI sends `bye`. The engine runs today's bounded `shutdown()`
  (stop the timer on its own thread, join within 5 s) and exits. If it does
  not exit within 10 s, the GUI kills it and logs CRITICAL naming what the
  engine was reading (the existing `polling_vi()` diagnostic, sent ahead in
  the `bye` answer). ⚑ Quit during a run asks "a run is in progress: stop it
  safely / keep the engine running headless / cancel", where "headless"
  depends on Reattach.
* **One engine per rack:** the engine takes an OS lock on the config directory
  (`O_EXCL` lock file plus PID, the same scheme as `session.lock`), so two GUIs
  can never drive one rack.
* **Logging:** the engine writes its own rotating log
  (`logs/engine.log`) and forwards WARNING and above to the GUI log panel.

## 7. Phase 4: everything above the proxy

Mostly no change, because the proxy surface is identical. The items that do
change:

* `ExperimentManager` (session switch) uses `set_run_folder` plus verdicts
  already. No change beyond the transport.
* The agent gateway, MCP and the HTTP endpoint attach to the GUI-side proxy as
  today. An agent's command now crosses two hops, still bounded.
* `ctl` (the CLI) uses the request spool, which is file-based. No change.
* `i2as.troubleshoot` talks to drivers directly and must refuse to open a rack
  whose engine lock is held.
* HDF5: only the engine writes run files. The GUI process reads **finished**
  runs only (analysis, notebook). Audit item: confirm nothing in the GUI opens
  an in-progress run file. HDF5 file locking across processes makes that an
  error on Windows rather than a silent race.

## 8. Phase 5: rollout and tests

* Mode matrix: CI runs the GUI and scenario suites in `threaded` and
  `process`. `inline` is removed one release after `process` ships, which was
  already planned for it.
* New tests:
  * the heartbeat probe from §1.1, made permanent with a GIL-holding
    `measure()`: the GUI thread never stalls over 0.5 s in `process` mode;
  * codec round-trip of every payload kind;
  * the `DeclaredStation` conformance test (§4.1);
  * kill the GUI mid-run: the run completes, the file is valid, and no queued
    run starts;
  * kill the engine mid-run: GUI alert, commands refused, restart works,
    interrupted record;
  * latest-wins backpressure under a slow client: events are all delivered,
    arrays are bounded;
  * token: a client without it is refused;
  * shutdown with a wedged read: bounded, CRITICAL names the read;
  * Windows CI job (already recommended by the previous plan; now required).
* Default flips to `process` after one lab release with no regression filed.

## 9. Separate track: the interruptible point (monitor data during a point)

A process split keeps the window live. It does not make readings, safety
checks or commands advance during a long point, because `measure()` is one
synchronous call inside one tick. If the operator also needs that, the fix is
in the engine, not the GUI:

* `SweepMeasureProcedure.measure()` already loops over `n_loop1 × n_loop2`
  `take_reading()` calls. It becomes a resumable step: one reading (or one
  bounded batch) per tick, with the partial grid held on the procedure. Between
  readings the tick runs safety, scalar polling and the command queue.
* `MEASURING` gains sub-steps. A pause or abort requested mid-point waits for
  the current reading, not the whole point (today's "pause boundary" rule moves
  from the point to the reading, and abort discards the partial point).
* A single `take_reading()` stays atomic. A one-reading-takes-30 s instrument
  still blocks for 30 s, and only an asynchronous driver API (arm, poll, fetch)
  can split that. This is a per-driver opt-in, not framework work.

Recommended order: Phases 1 to 3 first (they fix the reported freeze), then §9
if monitor data during a point is wanted.

## 10. Risks and open questions for the audit

1. Is `DeclaredStation` truly sufficient, or do some procedure hooks or VIs
   compute declarations from live state (for example, a measurement block size
   that depends on a setting read from the instrument)? Mitigation: the §4.1
   conformance test, and a declared rule that such values must be re-emitted
   in `StationDeclaration` when they change.
2. Validation latency: moving validation to declarations keeps it
   synchronous. The alternative (ask the engine) would make every form
   keystroke a round trip.
3. The queue lives in the GUI, so unattended continuation of a queue after a
   GUI loss is out of scope for v1. Is that acceptable?
4. Memory: two Python processes with Qt roughly double the baseline (about
   +150 MB). Acceptable on a lab PC?
5. Throughput: a 128×128 float64 frame is 128 KiB. At the monitored-array tick
   rates and per measured point this is far below local-socket bandwidth, but
   a high-rate camera could need shared memory later. Latest-wins bounds the
   damage meanwhile.
6. Debuggability: two processes, two logs. Mitigation: forwarded logs plus a
   shared run/request id in every log line.
