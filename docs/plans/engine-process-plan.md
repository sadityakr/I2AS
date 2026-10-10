# Plan: run the instruments in their own process (engine process)

Status: **plan, revision 2 (after audit round 1).** Nothing here is
implemented. §11 records the audit findings and where each one lands.

## 1. Problem

The operator reports that the Monitor window stops responding while a sweep
point is measured on real hardware: plot axes cannot be changed until the
point ends. The operator's stated goal is that the GUI and the instruments
run in **different processes**.

### 1.1 What we measured

`threaded` mode (the default) already puts the Station and the Orchestrator
on their own `QThread`.
`tests/test_instrument_thread.py::test_the_client_event_loop_keeps_running_through_a_slow_measure`
proves the window stays live while `measure()` blocks in `time.sleep`. A probe
used the same fixture, a 2 s `measure()` and a 50 ms heartbeat timer on the GUI
thread:

| What `measure()` does for 2 s | GUI timer firings in 1.5 s | Longest GUI stall |
|---|---|---|
| `time.sleep` (releases the GIL) | 29 | 0.05 s |
| pure-Python busy loop (GIL switches every 5 ms) | 32 | 0.05 s |
| one long C call that keeps the GIL | 5 | **2.17 s** |

Every Python thread in the process shares one interpreter lock (the GIL). A
driver call that holds it freezes the GUI thread for the whole call, and no
thread-level change can prevent that. A second process can.

The probe is synthetic. pyvisa's ctypes `CDLL` releases the GIL, so the culprit
on the operator's rig is likely one specific binding (a vendor SDK, a `PyDLL`,
a GPIB C extension). Phase 0 identifies it.

### 1.2 What a process split does and does not fix

* **Fixes:**
  * A GUI freeze caused by anything in the engine.
  * A GUI crash, out-of-memory or hang reaching a running experiment.
* **Does not fix:** monitor *data* during a point.
  * While `measure()` runs, the engine does not tick. Scalar readings, safety
    checks and queued commands, **emergency standby included**, wait up to one
    point in any process layout.
  * §9 tracks this separately (the **interruptible point**).

## 2. Non-negotiables

1. **Single hardware thread.** It holds unchanged: one thread, in one process
   (the engine process), touches drivers, VIs, the Station, the Orchestrator and
   the DataManager.
2. **Safety never depends on the GUI.**
   * A lost GUI is never a reason to touch an instrument.
   * A slow engine is never reported as a dead one.
   * The rack is never left without a way to reach its emergency control.
3. **One boundary.** The control contract (`core/events.py`) stays the one
   boundary.
   * Everything that crosses is a registered type.
   * Nothing crosses as a pickle, and no callable crosses.
   * Nothing on a safety channel is ever dropped.
4. **Clients keep their surface.** Widgets written against `OrchestratorProxy`,
   the session layer, the agent gateway, MCP and ctl all stay as they are.
5. **Windows is the primary platform.** That means spawn (no fork), named pipes,
   Windows file locking and process groups.

## 3. Target architecture

```
┌──────── GUI process (i2as.main) ─────────┐            ┌──── engine process (python -m i2as.engine) ────┐
│ windows, dialogs                         │            │ EngineServer (gateway-style local server)      │
│ session: ExperimentManager, RunQueue,    │  one       │ InstrumentHost (threaded, unchanged)           │
│   ELN, agent gateway, analysis runner    │  ordered   │   Station, Orchestrator, DataManager,          │
│ StatusMirror  ◄──────── stream ──────────┼────────────┤   TrendCheckRunner, run builder + validator    │
│ RemoteOrchestratorProxy ── commands ────►│  (seq)     │ rack lock · engine.json descriptor · log file  │
└──────────────────────────────────────────┘            └────────────────────────────────────────────────┘
```

* **Engine process.** It is today's `InstrumentHost` in `threaded` mode. Its
  client side is an `EngineServer` instead of the GUI. That server reuses the
  agent gateway's local-server pattern:
  * framing, the owner-only token descriptor and the `EngineClient` protocol
    from `session/gateway/local_server.py`;
  * the shared code moves to a core-level `core/local_ipc.py`, so the engine
    entry point does not import the session layer.
* **GUI process.** It holds a `RemoteOrchestratorProxy` with the same surface
  as `OrchestratorProxy`. The transport sits where the `ThreadBridge` was.
* **Modes.** `instrument_host.resolve_mode` gains two values:
  * `process`: `monitor.yaml` `instrument_process: true`, or
    `I2AS_INSTRUMENT_PROCESS=1`.
  * `loopback`: the full codec and framing over an in-process socket pair, for
    tests (§8).
  * The default stays `threaded` until Phase 5.

## 4. Phase 0: find the real culprit (days, independent)

* Capture `py-spy dump --native --pid <app>` while the operator's window is
  frozen. The native frames name the binding that holds the GIL.
* If it is a `PyDLL` or C extension that can be fixed, for example a ctypes call
  that can use `CDLL`, fix it. The operator gets relief in days, and the process
  split still proceeds.
* Make the heartbeat probe of §1.1 a permanent diagnostic. It runs in the app,
  and a GUI-thread stall over 1 s logs WARNING with the engine's `polling_vi()`
  (what it was reading at the time).

## 5. Phase 1: the boundary as data (still one process, `threaded`)

The goal of this phase is an exact inventory of what crosses today, each item
with a disposition. Each item lands and ships in `threaded` mode.

### 5.1 Inventory of what crosses today, and what each becomes

| Crossing today | Evidence | Becomes |
|---|---|---|
| `run_procedure(obj)` / `queue_procedure(obj)` (built objects) | `orchestrator_proxy.py:508-530` | `run_spec(RunSpec)` contract command; the engine builds |
| Queue pull: `ask(take_next_spec)` blocks the instrument thread, and a timeout loses the popped spec | `instrument_host.py:278-287`, `run_queue.py:1002-1025`, called in-tick from `_finish_run` (`orchestrator.py:4228-4241`) | **Asynchronous claim** (§5.3) |
| `build_spec` (session code) runs on the engine thread and calls `ExperimentManager.experiment_context` | `run_queue.py:1027-1052`, `manager.py:155` | The spec carries its `experiment_info`; the engine-side builder needs no session object |
| `next_procedure` / `queue_snapshot` setters; `QueuePanel._standalone_host` installs callables on the engine | `orchestrator_proxy.py:645-673`, `queue_panel.py:252-259` | Deleted; the only installer is the claim protocol |
| `main(on_station_built=…)` mutates the Station from the GUI side | `main.py:509-521`, `scripts/run_scenario.py:90` | Engine-side `--scenario` hook |
| The gateway reads the live `station.station_info` | `main.py:779` | `mirror.station_info` |
| GUI windows and `RunQueueHost` hold the live `Station` (6 C19 ignores, 5 for `core.station` plus 1 for `OrchestratorState`) | `pyproject.toml:438-449` | §5.2 |
| `last_state_flat()` read on the GUI thread (an unlocked read of an engine-mutated dict) | `procedure_window.py:796`, `station.py:1128-1145` | Keys come from `InstrumentInfo.monitored` declarations |
| `validate_run` / estimates do a headless build with the live Station on the GUI thread | `run_queue.py:860-880`, `procedure_window.py:695-711`, `queue_panel.py:266-290` | **Engine-side validation** request/answer (§5.2) |
| Run now builds a throwaway instance and keeps it as `_active_procedure` for plot setup | `procedure_window.py:710-718, 843-853` | The plot is set up from the class plus `ProcedureInfo.data_keys` |
| Agent read tools open the in-flight run's HDF5 file (`open_run` falls back to `locking=False`) | `gateway/tools.py:1996-2022`, `data_reader.py:65-79, 1104-1120` | The active `run_id` is served from `RunBuffer`; a file read for it is refused |
| Request-spool authorisation lives in `session.gateway.roles` | `main.py:525-538` | Moves to core so the engine never imports the session layer |
| Passthrough payloads that are not plain data: `RampRecord` (`ramps_updated`), `ErrorEvent` (`error_event`) | `orchestrator.py:2046, 3558, 4407` | Registered codec types (§5.4) |

### 5.2 Forms and validation without a live Station in the GUI

`DeclaredStation`, which would have answered from a JSON snapshot, is
**dropped**. Procedures read *code*, not just data: `RoleParam.candidates`
lambdas (`procedures/field_imaging.py:82-91`), `vi.data_arrays(params)`,
`system_setpoint_meta`, and `_loopable_registry`'s config state. Tier-1
procedures would break any snapshot. Instead:

* **Forms render from `ProcedureInfo.form`.** That field already exists, built
  from `declaration_form` (`core/procedure_catalog.py:164`).
  * `declaration_form` becomes the single source.
  * `TimeSeries`' Station-derived `get_param_groups` override moves into it.
    That also fixes today's drift between the GUI form and the agent's form.
  * The GUI uses `resolve_form`.
* **Live-plot vocabulary.** It moves into `ProcedureInfo.data_keys`, keyed per
  `measurement_vi` choice: live-plot measurement keys, image blocks and loop
  labels. These are enumerable, because `measurement_vi` is the only structural
  selection they depend on.
* **Validation and estimates.** They run **in the engine**, as a request with a
  correlated answer (`validate_run` → `RunValidation` as a contract type).
  * They fire on Run now and on Add to queue, both clicks, not on every
    keystroke.
  * The engine re-validates on `run_spec` regardless; a build error is a refused
    `Verdict`.
* **Contracts.**
  * The `OrchestratorState` enum moves into `core/events.py`.
  * All six C19 ignores are deleted.
  * New contract C28: `i2as.gui` and `i2as.session` never import
    `i2as.core.station`, typing included.
  * `i2as.engine` joins the contracts as a composition root next to `main`. It
    must not import `i2as.gui` or `i2as.session`.

### 5.3 The queue: asynchronous claim, never a blocking ask

1. The GUI **pre-stages** the head spec with every queue push. That is the push
   it already does for the snapshot (`publish_queue`). The head carries
   `spec_id`, the spec and its `experiment_info`.
2. When the engine is free to start the next run, it starts the staged head and
   emits `RunStarted{spec_id}`, or `Verdict` refused `{spec_id, reason}`.
3. The GUI removes the spec from its queue **only** on `RunStarted` for that
   `spec_id`. A refusal leaves it in place, flagged.
4. A queue edit replaces the staged head. If the engine already started the old
   head, the `RunStarted` for it wins, and the edit applies to the remainder.
5. The tick never waits on the GUI. `ThreadBridge.ask` loses its last user and
   is deleted, so the rule "no engine→client ask exists" holds by construction.

### 5.4 One codec, one ordered stream

* **`core/wire.py`** is the one codec.
  * Contract dataclasses travel in their JSON forms.
  * `RampRecord`, `ErrorEvent`, `RunValidation` and the other passthrough types
    are registered explicitly.
  * `numpy.ndarray` travels as a binary sidecar (dtype, shape, C-order bytes),
    and NaN/inf are preserved.
  * `Datapoint` arrays go as sidecars as well, instead of `tolist()` JSON.
* **Unknown types never drop silently.** On a safety channel (`error_event`,
  `operational_status`, `action_blocked`, conditions, state), an unknown type
  raises a non-dismissible GUI alert. Anywhere, it is a hard error in tests.
* **Ordering is load-bearing.** The session layer relies on the `run_started`
  passthrough arriving before the `RunStarted` event (`manager.py:1146-1158`).
  The mirror updates before the relays fire (`orchestrator_proxy.py:285-288`).
  So:
  * Everything goes out on **one** stream with one sequence number.
  * The client updates the mirror first, then re-emits each item in sequence
    order.
* **Debug flag.** Phase 1 runs every `threaded` emission through
  `decode(encode(x))`. The GUI suite runs with the flag on.

## 6. Phase 2: the transport

* **Socket.** `QLocalServer`/`QLocalSocket` with
  `SocketOption.UserAccessOption`, so the Windows named pipe is restricted to
  the owner. Exactly one client at a time.
* **Descriptor.** `engine.json` goes in the per-user data dir, with owner-only
  permissions: socket name, PID, rack id and token. This is the same scheme as
  `gateway.json`. It is what lets a restarted GUI reattach (§7).
* **Framing.** Reuses the gateway's: length-prefixed JSON header plus binary
  sidecars.
  * client→engine: `cmd`, `stage` (queue snapshot plus head), `validate`, `hb`,
    `bye`.
  * engine→client: one sequenced `out` stream (events, passthrough signals,
    validation answers, log records at WARNING and above) plus `hb`.
* **Engine side.** `EngineServer` runs on the engine process's main thread and
  calls the in-process `OrchestratorProxy`, which posts to the instrument thread
  as today.
* **Backpressure.** An application-level outbound queue, drained on
  `bytesWritten` (bytes already handed to the socket cannot be replaced).
  * Coalescing may only replace an item **in place**: `monitored_arrays_updated`
    and `states_updated`, latest per key.
  * `measurement_ready`, `Datapoint` and every event are **never** coalesced.
  * Over a byte cap, the client is disconnected with a logged reason. The run
    continues, and the client reattaches.

## 7. Phase 3: lifecycle and failure

* **Launch is detached, and the GUI's death never kills the engine.**
  * The engine starts with `subprocess.Popen`:
    * Windows: `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW |
      CREATE_BREAKAWAY_FROM_JOB`.
    * Unix: `start_new_session=True`.
  * It does not use `QProcess`, whose destructor kills the child.
  * The engine ignores console control events.
  * The token is created by the engine and written to `engine.json`, and the
    GUI reads it there.
  * Test: a GUI that raises in a slot, or exits, leaves the engine and its run
    alive.
* **"Engine lost" means the process died, and nothing else.**
  * Liveness comes from the OS: wait on the process handle or PID, not from
    heartbeats.
  * A silent socket only shows *"engine busy, no reply for N s"* with what it
    was last reading.
  * Restart is offered only after the process is gone.
  * Killing a live engine is a separate, confirmed action that names the
    consequence.
* **Engine death.**
  * The GUI shows a non-dismissible `engine_lost` alert, marks every reading
    stale, and refuses commands with a clear verdict.
  * A new session-layer handler files the active run FAILED with reason "engine
    process exited". No such path exists today; the startup sweep at
    `manager.py:1679` is the only one.
  * Data, notebook and analysis stay usable.
* **GUI loss** (a crash, a kill, or a socket drop with the GUI alive):
  * the engine sets attendance false (actor = system) and raises a
    `client_lost` condition, so spool and agent policy tighten;
  * the run in progress finishes normally;
  * the staged head is **not** started, because a queue nobody can see must not
    advance;
  * monitoring and safety keep running, and nothing is ramped down because the
    GUI left.
* **Reattach is in v1.**
  * On start, the GUI looks for `engine.json`. If the PID there is alive and the
    rack matches, it connects with the token instead of launching.
  * The engine sends a `ready` with the primed snapshot, the active run (if any)
    and its `spec_id`.
  * The session layer's startup sweep **skips** a RUNNING record whose run the
    attached engine reports as active.
* **Quit.**
  * With no run in progress: `bye` → the engine runs today's bounded
    `shutdown()` and exits. If it is still alive 10 s later, the GUI kills it
    and logs CRITICAL with the `polling_vi()` diagnostic the engine sent ahead.
  * During a run: *"A run is in progress: stop it safely / leave the engine
    running (reopen I2AS to reattach) / cancel"*.
* **One engine per rack.**
  * The lock is a real OS lock held on an open handle (`msvcrt.locking` /
    `fcntl.flock`) on `<user data dir>/racks/<rack id>.lock`. It does not sit in
    the config dir, which may be read-only or shared.
  * The lock is released by the OS on process death, so PID reuse does not
    matter.
  * The same lock is checked by `ctl` offline mode (`ctl/client.py:660-685`), by
    `troubleshoot` and by `run_scenario`.
* **Logging.** The engine writes `logs/engine.log` and forwards WARNING and
  above. Every line in both processes carries the run id or request id.

## 8. Phase 4/5: the layers above, the tests, and rollout

* **Layers above.** `ExperimentManager`, the gateway, MCP, the HTTP endpoint and
  ctl attach to the GUI-side proxy as today. Everything §5.1 lists is already
  handled.
* **Test modes.**
  * `loopback` runs the GUI and scenario suites through the real codec and
    framing in one process. The 62 `on_engine`/`set_on_engine`/`tick_engine`
    helper calls (`tests/instrument_modes.py`) keep working there.
  * `process` runs a small end-to-end set:
    * a GIL-holding `measure()` keeps GUI stalls under 0.5 s;
    * killing the GUI mid-run: the run completes and the file is valid, no
      staged run starts, `attended` becomes false, and a new GUI reattaches;
    * killing the engine mid-run: alert, refused commands, record FAILED,
      restart;
    * a busy (GIL-held) engine is never reported as lost;
    * a wedged-read shutdown is bounded;
    * a token-less client is refused;
    * the rack lock refuses a second engine, ctl offline and troubleshoot.
* **Unit tests.**
  * codec round trip of every registered type;
  * ordering, `run_started` before `RunStarted` with the mirror first;
  * coalescing never drops `measurement_ready`/`Datapoint`/events;
  * the claim protocol, including edit-during-start;
  * engine-side validation parity with today's results.
* **CI.** A Windows job is required.
* **Rollout.** The default flips to `process` after one lab release without a
  filed regression. `inline` is removed one release after that, as already
  planned.

## 9. Separate track: the interruptible point

* **The change.** `SweepMeasureProcedure.measure()` already loops
  `n_loop1 × n_loop2` `take_reading()` calls. It becomes resumable, one reading
  (or a bounded batch) per tick, with the partial grid held on the procedure.
* **What runs between readings.** Safety, scalar polling and the command queue
  run between readings. Polling **excludes** the armed measurement VI and any
  VI on its bus while a point is open.
* **Pause and abort.** They wait for the current reading, not the whole point.
  Abort discards the partial point.
* **Limit.** One `take_reading()` stays atomic. Splitting it needs an
  asynchronous driver API (arm, poll, fetch), as a per-driver opt-in.
* **Order.** It is independent of the process split and comes after Phase 3.

## 10. Open decisions for the operator

1. **GUI loss.** Finish the current run, then hold with the queue stopped (as
   recommended above)?
2. **Quit during a run.** Is offering "leave the engine running" acceptable?
3. **Interruptible point (§9).** Is it wanted, for live readings and commands
   during a long point?

## 11. Audit, round 1 (senior lab-software review)

Verdict: **approve with changes.** Every finding is adopted.

| # | Sev | Finding | Revision |
|---|---|---|---|
| 1 | blocker | After a GUI crash, nobody can reach the rack: the token died with the GUI, and reattach was deferred | Reattach in v1; owner-only `engine.json` token descriptor; the startup sweep skips the attached engine's active run (§6, §7) |
| 2 | blocker | Heartbeat-based liveness reports a GIL-busy engine as lost and invites a racing restart | Liveness from process death only; "busy" display; restart only after the process is gone (§7) |
| 3 | blocker | `QProcess` kills its child on destruction; a console close kills the group | Detached `Popen` with its own group, breakaway from the job; console events ignored; test (§7) |
| 4 | blocker | The queue `ask` deadlocks across processes, loses a popped spec on timeout, and blocks the tick | Asynchronous claim with `spec_id`; `ask` deleted (§5.3) |
| 5 | major | Callables and objects crossing today were not inventoried | Inventory table (§5.1); `i2as.engine` in the contracts |
| 6 | major | A `DeclaredStation` JSON snapshot cannot reproduce procedure code (lambdas, param-dependent arrays) | Dropped; validation in the engine; forms from `ProcedureInfo` (§5.2) |
| 7 | major | Reinvented existing contract types; the GUI and agent forms already drift | `declaration_form` single source; `ProcedureInfo.data_keys` extended (§5.2) |
| 8 | major | `last_state_flat` keys are live, not declared | Keys from `InstrumentInfo.monitored` (§5.1) |
| 9 | major | Signal ordering is load-bearing; split `sig`/`evt` frames could reorder | One sequenced stream; mirror first; ordering test (§5.4) |
| 10 | major | Latest-wins would drop live-plot points | `measurement_ready`/`Datapoint` never coalesced; arrays as sidecars (§5.4, §6) |
| 11 | major | Payloads include `RampRecord`/`ErrorEvent`; "logged drop" could lose an EMERGENCY | Registered types; unknown type on a safety channel → alert (§5.4) |
| 12 | major | The HDF5 claim was wrong: the reader falls back to unlocked; agent tools read the in-flight run | The active run is served from `RunBuffer`; file reads refused (§5.1) |
| 13 | major | GUI loss left `attended=True` | Attendance false and a `client_lost` condition (§7) |
| 14 | major | No "interrupted" run path exists | New engine-death handler (§7) |
| 15 | major | 62 tick-helper calls cannot run cross-process | `loopback` mode plus a small `process` end-to-end set (§8) |
| 16 | minor | `O_EXCL`+PID in the config dir is not an OS lock | Handle-held OS lock in the user data dir; checked by ctl, troubleshoot and scenario (§7) |
| 17 | minor | Named-pipe ACL; backpressure on bytes already written | `UserAccessOption`; application queue drained on `bytesWritten` (§6) |
| 18 | minor | C19 has 6 ignores, not 5 | `OrchestratorState` moves to `events.py` (§5.2) |
| 19 | minor | Run now already submits JSON; the real issue is the kept throwaway instance | Plot from class plus `ProcedureInfo` (§5.1) |
| 20 | minor | The freeze fix arrived only at Phase 3; the cause was unconfirmed on the rig | Phase 0, a native stack capture and targeted binding fix (§4) |
| 21 | nit | §9: exclude the armed VI's bus; state the emergency latency | Added (§1.2, §9) |
