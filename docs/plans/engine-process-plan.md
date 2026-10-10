# Plan: run the instruments in their own process (engine process)

Status: **plan, revision 5, signed off by the audit (round 4).** Nothing here is
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
| `run_procedure(obj)` / `queue_procedure(obj)` (built objects) | `orchestrator_proxy.py:508-530` | `run_spec(RunSpec)` contract command; the engine builds. Refused unless IDLE: a Run now during a run goes into the GUI queue instead |
| The engine-held `_procedure_queue` (fed when `run_procedure` arrives non-IDLE, drained first by `run_queue`) | `orchestrator.py:1427-1429, 1736-1740` | Deleted. The GUI queue is the only queue |
| Queue pull: `ask(take_next_spec)` blocks the instrument thread, and a timeout loses the popped spec | `instrument_host.py:278-287`, `run_queue.py:1002-1025`, called in-tick from `_finish_run` (`orchestrator.py:4228-4241`) | **Asynchronous claim** (§5.3) |
| `build_spec` (session code) runs on the engine thread and calls `ExperimentManager.experiment_context` | `run_queue.py:1027-1052`, `manager.py:155` | `experiment_info` travels with `set_run_folder` as one atomic command (§5.3), not with the spec; the engine-side builder needs no session object |
| `next_procedure` / `queue_snapshot` setters; `QueuePanel._standalone_host` installs callables on the engine | `orchestrator_proxy.py:645-673`, `queue_panel.py:252-259` | Deleted; the only installer is the claim protocol |
| `main(on_station_built=…)` mutates the Station from the GUI side | `main.py:509-521`, `scripts/run_scenario.py:90` | Engine-side `--scenario` hook |
| The gateway reads the live `station.station_info` | `main.py:779` | `mirror.station_info` |
| GUI windows and `RunQueueHost` hold the live `Station` (6 C19 ignores, 5 for `core.station` plus 1 for `OrchestratorState`) | `pyproject.toml:438-449` | §5.2 |
| `last_state_flat()` read on the GUI thread (an unlocked read of an engine-mutated dict) | `procedure_window.py:796`, `station.py:1128-1145` | Keys come from `InstrumentInfo.monitored` declarations, filtered to numeric scalar fields of *system* VIs (what `last_state_flat` holds) |
| `validate_run` / estimates do a headless build with the live Station on the GUI thread | `run_queue.py:860-880`, `procedure_window.py:695-711`, `queue_panel.py:266-290` | **Engine-side validation** request/answer (§5.2) |
| Run now builds a throwaway instance and keeps it as `_active_procedure` for plot setup | `procedure_window.py:710-718, 843-853` | The plot is set up from the class plus `ProcedureInfo.data_keys` |
| Agent read tools open the in-flight run's HDF5 file (`open_run` falls back to `locking=False`) | `gateway/tools.py:1996-2022`, `data_reader.py:65-79, 1104-1120` | The active `run_id` is served from `RunBuffer`, which marks itself partial when it lacks the points before a reattach; a file read for it is refused |
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
  * The engine re-runs `build_procedure_infos` on every `StationInfo` rebuild,
    so role candidates follow `connect_instrument`/`disconnect_instrument`
    as the live `get_param_groups` call does today (`main.py:516`,
    `station.py:829-850`).
* **Live-plot vocabulary.** It moves into `ProcedureInfo.data_keys`, keyed per
  `measurement_vi` choice: live-plot measurement keys, image blocks and loop
  labels. These are enumerable, because `measurement_vi` is the only structural
  selection they depend on.
* **Validation and estimates.** They run **in the engine process, on its main
  thread**, as a request with a correlated answer (`validate_run` →
  `RunValidation`, a contract type). They do not run on the instrument thread:
  * They read the Station through a **guard view**, an **allow-list**:
    * the Station's declaration readers;
    * a VI wrapper that allows plain attributes plus a declared set of
      declaration methods (`data_arrays`, `raw_block_row_counts`,
      `quantity_columns`, …) and declared properties
      (`active_measurement_parameters`, `reading_parameters`,
      `configured_externally`, `base.py:2008-2075`). A property can do I/O,
      so properties count as callables;
    * every other callable is **denied by default**. Undecorated methods such as
      `magnet_field_T()` can do driver I/O (`procedures/field_sweep.py:135`), so
      denying decorated methods alone is not enough. A denied call becomes a
      validation finding: fail closed.
  * The instrument thread publishes an **immutable registry snapshot** on every
    rebuild. Validation reads that snapshot, never the live registry that
    `connect_instrument` mutates.
  * Validation does not wait behind a normal `measure()`. A GIL-holding driver
    call also starves the engine's main thread, so the GUI shows "validating…"
    and the deferred answer has a timeout ("engine busy, try again").
  * The parity test below shows every catalog procedure builds through the
    allow-list. No catalog procedure uses `isinstance` on a VI, and a
    conformance check keeps it that way.
  * The request carries the envelope to check against, as today's call does.
    The engine still re-validates at start against its own envelope, and a
    build error there is a refused claim (§5.3).
  * The synchronous surfaces become asynchronous: `RunQueueHost.add`,
    `ExperimentManager.queue_run`/`validate_run` (`manager.py:852-940`), and
    the gateway's `_tool_validate_run` (`gateway/tools.py:2381-2418`) through a
    deferred JSON-RPC answer in `local_server.py`.
  * With no engine running, the answer is "cannot validate: engine not
    running". Queueing is refused, and the form stays editable.
  * A parity test runs every catalog procedure with its defaults and with
    edge values, and checks the answer equals today's `validate_run` output.
* **Contracts.**
  * The `OrchestratorState` enum moves into `core/events.py`.
  * All six C19 ignores are deleted.
  * New contract C28: `i2as.gui` and `i2as.session` never import
    `i2as.core.station`, typing included.
  * `i2as.engine` joins the contracts as a composition root next to `main`. It
    must not import `i2as.gui` or `i2as.session`.

### 5.3 The queue: claims with ids, never a blocking ask

The GUI keeps owning the queue. The engine is offered at most one **staged
head** and answers every claim exactly once.

* **Identity.**
  * Every queued spec carries a `spec_id` (uuid) for its whole life.
  * `RunSpec.spec_id` already exists. It is now persisted in `QueueItemState`/`export_items`
    (`gui/form_autosave.py:61-89`, `queue_panel.py:682-694`). A restore keeps
    it and does not mint a new one (`run_queue.py:149`). Autosave files from
    before this change get ids minted once, on load. Reconciliation
    *replaces* entries and never re-adds them, so `RunQueue.add`'s duplicate
    check (`run_queue.py:355-356`) never trips.
  * The engine writes `spec_id` into the run manifest, and the run record
    stores it.
* **Staging.**
  * The GUI stages the first **unflagged** spec with every queue push.
  * Each `stage` frame carries the last engine `seq` the GUI had applied. The
    engine ignores a stage older than its latest claim.
  * Each `stage` also carries the engine **instance id** (from `ready`). A
    stage addressed to another engine instance is ignored, because `seq`
    restarts with a new process.
  * The engine keeps a **consumed ledger**. It is an append-only file in the
    machine-wide rack directory, independent of any run folder.
    * The engine writes and fsyncs `spec_id` **at `Claimed`, before the
      build**, and appends the outcome (`started`, `failed:<stage>`, `done`)
      once known.
    * It refuses any `spec_id` already in the ledger. **Retry** on a flagged
      spec clones it with a fresh `spec_id`, an explicit operator action. That
      keeps the guarantee of no automatic re-run.
    * Manifests only cross-check it. Entries older than 30 days are pruned. For
      older autosaves, run records (which carry `spec_id`) decide
      reconciliation.
  * Together these make a stale re-stage of a started run impossible, whatever
    the timing.
* **Start trigger.** A claim happens only on the `run_queue` command or on the
  finish/abort chain (`orchestrator.py:4228-4241`). Staging never starts
  anything on its own.
* **Claim answers.**
  * The moment the engine takes the head it emits `Claimed{spec_id}`.
  * Then exactly one of `RunStarted{spec_id}` or
    `ClaimFailed{spec_id, stage: build|placement|setup, reason}`. Every refusal
    path in `_start_run`/`_fail_to_error` (`orchestrator.py:1494-1572`) maps to
    one `stage`.
  * The GUI moves the spec out of the waiting list on `Claimed`, to "running" on
    `RunStarted`, or to "failed, not retried" on `ClaimFailed`. A spec whose
    setup touched hardware is never re-run automatically.
* **After a refusal.** The chain stops and notifies, as today's ERROR path does.
  Nothing advances until the operator runs the queue again, and the refused spec
  stays visible and flagged.
* **Experiment context.**
  * `experiment_info` is not part of the staged spec.
  * It travels with `set_run_folder` as **one atomic command**
    (`_install_run_folder`, `manager.py:1492-1520`), so the folder a run writes
    to and the experiment it is stamped with cannot diverge. Today's "stamped
    with the experiment open when it actually gets built" still holds.
  * The context also carries `attended`, the user and the notebook link, which
    can change without a folder change. So the atomic folder+context command is
    re-sent on every `experiment_changed` and record edit, not only on a folder
    change.
  * The context's `attended` is **stamped metadata only**. The engine stamps
    each run from its own attendance state, and a re-sent context can never
    change policy.
  * Releasing the run folder (a session switch, a held session) also clears the
    staged head.
* **No hidden queue.** The engine holds no queue of its own (§5.1). On client
  loss it clears the staged head, and a reattached GUI must reconcile (§7)
  before it stages again.
* **The ask is deleted.** `ThreadBridge.ask` loses its last user and is removed,
  so "no engine→client ask exists" holds by construction.

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
* **Ordering, both ways.** Client→engine frames (`cmd`, `stage`, `validate`)
  share one ordered channel and are applied in arrival order. The engine
  applies `cmd` and `stage` through the existing single courier FIFO, so
  session switches and envelope edits keep today's ordering.
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
  * A cap disconnect is **not** a GUI loss, and only that kind of disconnect
    gets the grace. If the same token reconnects within 30 s, attendance is
    unchanged and no `client_lost` condition is raised.
    * During the grace, the queue chain is **held**: no claims.
    * A peer-closed socket, or a dead GUI process, is an immediate GUI loss.
    * **Every** reconnect, graced or not, runs the full `ready` + outbox +
      reconciliation path (§7). Only the policy changes are skipped.
    * Events missed during the drop reach the GUI through the outbox and the
      snapshot, never through a resumed byte stream.
    * If the chain was held during the grace, the engine re-attempts it once,
      after reconciliation and a fresh stage. If it cannot run, it tells the
      operator; it never stalls silently.

## 7. Phase 3: lifecycle and failure

* **Launch is detached, and the GUI's death never kills the engine.**
  * The engine starts with `subprocess.Popen`:
    * Windows: `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW |
      CREATE_BREAKAWAY_FROM_JOB`.
    * Unix: `start_new_session=True`.
  * It does not use `QProcess`, whose destructor kills the child.
  * The engine ignores console control events.
  * **Windows jobs.** If breakaway is denied (the parent's job lacks
    `JOB_OBJECT_LIMIT_BREAKAWAY_OK`: some IDEs, terminals, CI, Citrix), the
    engine launches inside the job. The plan uses **no** escape through the Task
    Scheduler or WMI: that would defeat a policy an administrator set on
    purpose, and endpoint security may flag it.
    * When that job has `KILL_ON_JOB_CLOSE` (detected with `IsProcessInJob` +
      `QueryInformationJobObject`), the GUI shows a persistent warning, "a GUI
      crash will stop the engine on this PC", and does not offer "leave the
      engine running".
    * The supported launch path is documented: a Start-menu shortcut, outside
      any job.
    * Windows CI tests both job configurations.
  * The GUI passes a launch nonce in argv. The engine writes it into
    `engine.json` with its token, and the GUI accepts only the descriptor that
    carries its nonce, never a stale one.
  * Test: a GUI that raises in a slot, or exits, leaves the engine and its run
    alive.
* **"Engine alive" is decided by the rack lock and its holder record.**
  * The engine holds the machine-wide rack lock (below) for its whole life, and
    the OS releases it on death.
  * The lock's **holder record** lives in a sidecar file, because Windows
    byte-range locks make the locked region unreadable. It names the holder's
    kind (`engine`, `app-threaded`, `ctl`, `troubleshoot`, `scenario`), user,
    host, PID and process creation time.
  * **Reattach.** The GUI reattaches only to kind `engine`, and only when
    `engine.json` carries the matching nonce. Any other kind means "rack in use
    by <kind>, <user>".
  * **Probing.** A probe try-lock releases at once, and the engine retries its
    own acquisition for a few seconds, so a probe never makes a starting engine
    fail.
  * **Kill safety.** Before any kill, the GUI checks the PID and creation time
    against the holder record. This makes it immune to PID reuse.
  * A silent socket only shows *"engine busy, no reply for N s"*, with what it
    was last reading.
  * Restart is offered only once the lock is free.
  * Killing a live engine is a separate, confirmed action that names the
    consequence.
* **Wedged shutdown.** If the engine's own join times out, it logs CRITICAL and
  calls `os._exit`.
  * If a driver I/O cannot be cancelled, the process may linger and keep the
    lock. The GUI then shows "engine terminating, stuck in driver I/O".
  * The GUI does not repeat the kill or offer a restart while the lock is held.
  * `os._exit` cannot run while a driver holds the GIL. In that case the GUI's
    10 s kill is the real backstop.
* **Windows logoff.** Logging off the owning Windows user ends the engine and
  its run. For long runs, the documentation recommends a shared operator
  account.
* **Engine death.**
  * The GUI shows a non-dismissible `engine_lost` alert, marks every reading
    stale, and refuses commands with a clear verdict.
  * A new session-layer handler files the active run FAILED with reason "engine
    process exited".
  * Data, notebook and analysis stay usable. Validation answers "engine not
    running".
* **GUI loss** (a crash, a kill, a deliberate "leave the engine running", or a
  socket drop not recovered within the grace period):
  * the engine sets attendance false (actor = system) and raises a
    `client_lost` condition, so spool and agent policy tighten;
  * the run in progress finishes normally;
  * the staged head is cleared, and the engine holds no other queue;
  * monitoring and safety keep running, and nothing is ramped down because the
    GUI left.
* **Finished runs.** The engine keeps a persisted **outbox** of terminal
  manifests (`RunFinished` plus `spec_id`) in the rack directory for **every**
  finished run, attached or not. That covers a GUI that dies between receiving
  `RunFinished` and saving the record (`manager.py:1179-1195`).
  * **Delivery is at-least-once.** The outbox is replayed in every `ready`. An
    entry is removed only on the GUI's `ack{run_id}`, sent **after** the record
    is saved.
  * **Applying is idempotent** by `run_id` and `spec_id`.
  * **Bound.** Entries are acked within seconds while attached, and nothing is
    claimed after a loss, so at most one entry waits per detachment. Overflow is impossible by construction; if it ever happens,
    it is a CRITICAL log, never a silent drop.
* **Reattach, in v1, in this order:**
  1. The GUI finds the rack lock held, reads `engine.json` (owner-only) and
     connects with the token.
  2. `ready` carries the primed snapshot, the active run with its `spec_id`,
     the outbox, and the engine's current session root and run folder.
  3. `ready` is handled **before** `ExperimentManager` is constructed
     (`main.py:647`):
     * the outbox is applied to the run records first;
     * the startup sweep (`manager.py:1679-1683`) then skips the active run
       and fails only RUNNING records the engine knows nothing about.
  4. **Session ownership.** If the engine has an active run, the GUI adopts
     the engine's session root and run folder, and refuses to change them until
     the engine is IDLE. A different OS user cannot reattach, because
     `engine.json` is per-user. They still have emergency control (below).
  5. **Queue reconciliation, before any stage.**
     * The restored queue is matched by `spec_id` against the engine's active
       run, the outbox, the run records and the consumed ledger.
     * A spec in the ledger with no outcome is "claimed, outcome unknown". It is
       settled from the run record, and it is never re-staged.
     * Matched specs are marked running or done, never PENDING. The old
       "RUNNING → PENDING on restore" rule (`queue_panel.py:661-664`) applies
       only when no engine was attached.
  6. **Attendance is never restored from the record after a real loss.** The
     adopt path's push of `record.attended` (`manager.py:555, 1690`) is
     suppressed. The operator confirms attendance explicitly, and the queue
     stays stopped until they run it.
* **Emergency access for every local user.**
  * The engine opens a second, **emergency-only** endpoint, ACL'd to local
    interactive users.
  * It accepts only `emergency_standby` and status reads, attributed to the
    calling OS user.
    * **Who may call it** is a configured decision (`emergency_endpoint:
      console|interactive`, default `interactive`, which includes RDP
      sessions). Every use is logged at CRITICAL and raises an alert in the
      owner's GUI.
    * Status reads are limited to state and safety fields, never experiment
      details.
    * Identifying the caller needs a small native shim
      (`GetNamedPipeClientProcessId` → token user), because `QLocalServer`
      doesn't expose it.
  * Another user's GUI shows "Emergency standby (rack owned by <user>)". This
    follows the role model's own carve-out, that an actor who can see a problem
    can always make the station safe (`session/gateway/roles.py:450-452`).
* **Quit.**
  * With no run in progress: `bye` → the engine runs today's bounded
    `shutdown()` and exits. If the rack lock is still held 10 s later, the GUI
    kills the engine and logs CRITICAL with the `polling_vi()` diagnostic the
    engine sent ahead.
  * During a run: *"A run is in progress: stop it safely / leave the engine
    running (reopen I2AS to reattach) / cancel"*. Leaving it running is a GUI
    loss for attendance and the queue.
* **One opener per rack, in every mode.**
  * The lock is machine-wide: `%ProgramData%\I2AS\racks\` (ACL'd for Users),
    or `/var/lock/i2as` with a user fallback. It is held with
    `msvcrt.locking(LK_NBLCK)` / `fcntl.flock` on an open handle.
  * It is keyed on a declared `rack_id` in the config (default: a hash of the
    sorted VISA resource strings), so two configs for one rack collide.
  * Every Station builder **acquires** it for its lifetime, in `threaded` and
    `inline` too: `InstrumentHost`, ctl offline (`ctl/client.py:660-685`),
    troubleshoot and `run_scenario`.
  * The lock handle belongs to the caller and is passed into `InstrumentHost`,
    so one process holds it once and may build more than once.
  * All-sim configs get a **stable** `rack_id`: a hash of the config path and
    the OS user, the same in every mode. A restarted GUI then finds its own sim
    engine and can reattach. This keeps the §8 reattach tests and sim demo
    installs working.
  * Test fixtures are isolated by the caller-owned handle, `I2AS_RACK_LOCK=off`,
    or a fixture-scoped id, never by a per-process id. A test proves two
    hardware-config builders collide.
  * The holder sidecar is written atomically right after the lock is
    acquired, and is trusted only while the lock is held.
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
    * the rack lock refuses a second engine, ctl offline and troubleshoot;
    * reattach after a GUI crash: no run executes twice, the run that finished
      while detached is filed done, and attendance stays false until confirmed;
    * Windows: breakaway allowed, breakaway denied (with the fallback), and a
      `KILL_ON_JOB_CLOSE` job (warning shown).
* **Unit tests.**
  * codec round trip of every registered type;
  * ordering, `run_started` before `RunStarted` with the mirror first;
  * coalescing never drops `measurement_ready`/`Datapoint`/events;
  * the claim protocol: edit-during-start, stale stage, ledger refusal, every
    `ClaimFailed` stage, a session switch with a head staged, refusal stops the
    chain;
  * `spec_id` survives autosave and restore;
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

## 12. Audit, round 2

Verdict: **approve with changes.** §5.3 and reattach were not to be
implemented until the three blockers were fixed. All findings are adopted.

| # | Sev | Finding | Revision |
|---|---|---|---|
| N1 | blocker | An edit before `RunStarted` arrives re-stages a started head, and it runs twice | Persisted consumed ledger; `stage` carries the last applied `seq`; stale stages ignored (§5.3) |
| N2 | blocker | The persisted queue has no `spec_id` and restores RUNNING as PENDING, so a reattach re-runs the in-flight run | `spec_id` persisted; reconciliation against active run, outbox, records and ledger before staging (§5.3, §7) |
| N3 | blocker | A run that finishes while detached is filed FAILED by the startup sweep | Persisted outbox replayed in `ready`, applied before `ExperimentManager` and its sweep (§7) |
| N4 | major | Setup refusals and errors give no per-spec answer; the spec stays staged after touching hardware | `Claimed` plus exactly one `RunStarted`/`ClaimFailed{stage}`; never auto re-run (§5.3) |
| N5 | major | A refused head blocks the queue | Stage the first unflagged spec; the chain stops and notifies (§5.3) |
| N6 | major | Pre-staged `experiment_info` goes stale | Sent atomically with `set_run_folder`; release clears the head (§5.3) |
| N7 | major | The engine's own `_procedure_queue` still advances after GUI loss | Deleted; `run_spec` only when IDLE (§5.1) |
| N8 | major | A cap disconnect flips attendance; reattach restores `attended` from the record | Grace period for the same token; explicit confirmation after a real loss (§6, §7) |
| N9 | major | Breakaway can be denied; fallback missing; stale descriptor | Fallback chain, `KILL_ON_JOB_CLOSE` warning, launch nonce, CI for both configurations (§7) |
| N10 | major | Engine-side validation: latency behind `measure()`, envelope race, synchronous APIs, engine-down | Main-thread guard view; envelope in the request; async surfaces; engine-down answer; parity test (§5.2) |
| N11 | major | `ProcedureInfo` forms frozen at startup | Rebuilt with every `StationInfo` (§5.2) |
| N12 | minor | PID-based liveness after reattach | Liveness from the rack lock (§7) |
| N13 | minor | Client→engine ordering unstated | One ordered channel through the courier FIFO (§6) |
| N14 | major | Per-user, per-config, check-only, process-mode-only lock | Machine-wide, `rack_id`-keyed, acquired by every Station builder in all modes (§7) |
| N15 | minor | Session ownership on reattach undefined | The GUI adopts the engine's session and run folder while a run is active (§7) |
| N16 | nit | Start trigger and quit semantics | Stated (§5.3, §7) |
| R1-8, R1-12 | caveats | System-VI numeric filter; partial `RunBuffer` after reattach | Added (§5.1) |

## 13. Audit, round 3

Verdict: **approve with changes, no blockers.** Phases 0 to 2 could start;
Phase 3 waited on R1 to R4 and R6 to R8. All findings are adopted.

| # | Sev | Finding | Revision |
|---|---|---|---|
| R1 | major | Ledger "rebuilt from manifests", but setup/build failures have no manifest, and a fresh engine has no run folder | Append-only ledger in the rack dir, fsynced at `Claimed` before the build (§5.3) |
| R2 | major | The outbox has no ack/removal rule | At-least-once, removed on `ack` after the record is saved; idempotent; bound by construction (§7) |
| R3 | major | Grace applied to crashes; claims during the grace; events lost on a cap drop | Grace only for engine-initiated drops; chain held; full reconcile on every reconnect (§6) |
| R4 | major | Task Scheduler/WMI escape defeats admin policy and may be flagged by EDR | Removed; launch in the job with a warning; documented launch path (§7) |
| R5 | major | The guard cannot stop I/O through real VI objects; registry race | Default-deny allow-list wrapper; immutable registry snapshot (§5.2) |
| R6 | major | The lock in all modes breaks fixtures and loopback | Caller-owned handle; per-process sim `rack_id`; test switch (§7) |
| R7 | major | No emergency control for another OS user (Non-negotiable 2) | Emergency-only endpoint for local interactive users (§7) |
| R8 | major | Lock-held cannot tell an engine from ctl/troubleshoot; the probe races the engine | Holder record with kind/PID/creation time; probe releases; the engine retries (§7) |
| R9 | minor | Windows byte-range locks hide the owner info; a wedged kill keeps the lock | Sidecar holder record; `os._exit` after join timeout; "stuck in driver I/O" state (§7) |
| R10 | minor | The context changes without a folder change | Re-sent on every experiment/record change (§5.3) |
| R11 | nit | `seq` has no engine-instance epoch | Instance id in `ready` and `stage` (§5.3) |
| R12 | nit | Restore may trip the duplicate check; legacy autosaves have no id | Replace-not-add; ids minted on legacy load (§5.3) |

## 14. Audit, round 4: signed off

R1 to R12 were confirmed resolved. Sign-off depended on A1; A2 to A10 were
adopted as well.

| # | Sev | Finding | Revision |
|---|---|---|---|
| A1 | major | A per-process sim `rack_id` breaks reattach and its tests | Stable `rack_id` (config path + user); fixtures isolated by handle/env/fixture id (§7) |
| A2 | minor | A run finished while attached, with the GUI dying before the save, is filed FAILED | Every terminal manifest goes to the outbox and is acked after the save (§7) |
| A3 | minor | The ledger blocks retrying a flagged spec | Retry clones with a fresh `spec_id`; run records decide for old autosaves (§5.3) |
| A4 | minor | Properties can do I/O; the latency claim was false under a GIL hold | Properties are allow-listed callables; "validating…" state plus timeout (§5.2) |
| A5 | minor | A context re-send could undo attendance | `attended` in the context is metadata; the engine stamps its own (§5.3) |
| A6 | minor | A graced reconnect can leave the queue stalled | Re-attempt once or notify (§6) |
| A7 | minor | Emergency endpoint: who may call, exposure, caller identity | Configured scope, CRITICAL log + alert, state-only reads, native shim (§7) |
| A8 | nit | Contradictory lock-info text | Atomic sidecar, trusted only under the lock (§7) |
| A9 | nit | List structure broken | Emergency bullet moved out of the reattach steps (§7) |
| A10 | nit | `os._exit` under a GIL hold; Windows logoff | Documented (§7) |
