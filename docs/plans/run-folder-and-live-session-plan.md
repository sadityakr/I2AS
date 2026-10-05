# Plan: run subfolders inside an experiment, and loading a session while running

Status: **plan, for audit before implementation.**

## 1. Requirements (from the operator)

1. **Startup unchanged.** The app silently opens the last active session (or creates one on first launch). No prompt.
2. **Run subfolder.** The operator can choose a subfolder inside the open experiment's `data/` folder (e.g. `data/cooldown2/`). Every subsequent run — the operator's, a queued one, an agent's, a probe — is written there. It can never point outside the experiment's `data/` folder.
3. **Live session load.** A session folder can be loaded (opened or created) while the app is running, without a restart.

Plus one bug found while researching: after **User → Log in as…** with an experiment open, the read-only Data Dir field can show the new user's old autosaved folder, which is not where runs are written.

## 2. Non-negotiables

* No operator/agent action here may affect a running experiment: switching session or experiment is **refused** while anything that belongs to the current session is in flight.
* The engine never imports the session layer (contract C12): the session layer owns the policy values and pushes them down; the engine only enforces them.
* Run ids stay unique per experiment (`run-NNNN`, never reused): analysis folders (`analysis/run-NNNN`), the publishing ledger and `find_run()` depend on that.
* Existing session folders, experiment records and run records keep working unchanged.

## 3. Design — run subfolder

### 3.1 Policy value (session layer)

* `ExperimentRecord.run_subfolder: str = ""` (models.py, beside `attended`), persisted in `experiment.json`; absent in old records ⇒ `""` (runs directly in `data/`, today's behaviour).
* `ExperimentManager.set_run_subfolder(subfolder: str)`: validates via one helper `normalize_run_subfolder()`:
  * relative POSIX path, each component matching `[A-Za-z0-9][A-Za-z0-9._ -]*`, no `.`/`..`, no drive/absolute, at most 3 levels, normalised (`a//b/` → `a/b`);
  * refused (ValueError) when no experiment is open.
  * Allowed while a run is active: it applies only to runs placed afterwards (a run's path is fixed at placement).
* Saves the record and re-pushes the run folder. `run_folder()` accessor returns `data/<subfolder>` for display.

### 3.2 Engine (enforcement only)

* `Orchestrator.set_run_folder(data_directory, subfolder="")` — command gains an optional `subfolder` (proxy + `events` command schema updated; the old one-argument form still works).
* `_place_run`: numbering root = `data_directory` (the experiment's `data/`), target = `data_directory/subfolder`. The engine re-checks containment (`resolve().is_relative_to(root)`) as defence in depth and refuses the run if it fails.
* `core/run_naming`:
  * `next_run_number(root)` scans **recursively** (`rglob("run-*.h5")`), so numbers stay unique across subfolders;
  * `place_run(root, procedure, label, kind, subfolder="")` returns a placement in `root/subfolder`.
* `DataManager` already creates parent folders and refuses to overwrite.
* `RunRecord.data_file` is already stored relative to the experiment folder and already supports nested paths; `resolve_data_file()`'s `rglob(basename)` fallback stays correct because names are unique.

### 3.3 GUI

* `ExperimentInfoPanel` Data Dir row: the field always shows the **effective run folder** read from the manager (`run_folder()`), never an autosaved value — this also fixes the login bug. Read-only.
* A **"Subfolder…"** button (replacing the hidden Browse) opens a folder dialog rooted at the experiment's `data/` with a "New folder" option; the pick must be inside `data/` (else a warning, nothing changes), and is written through `set_run_subfolder()`. A **"Reset"** action returns to `data/`.
* The panel re-reads on `experiment_changed` and on a new `run_folder_changed` signal from the manager.
* `MonitorWindow.get_data_dir_for_run()` containment check becomes moot (the engine places runs); kept as a pass-through.

### 3.4 Agents / CLI

* Agent-started runs land in the subfolder automatically (the engine places every run).
* `experiment_status` (gateway) reports the current `run_folder`. **No new agent tool to change it** (not requested; changing where data goes stays an operator decision).
* Gateway tool schema text "the experiment's data folder" updated.

## 4. Design — loading a session while running

### 4.1 Re-root in place

Most consumers (gateway tools, analysis runner, agent panel, analysis panel) read `manager.store` / `current_experiment()` live, so the existing `ExperimentManager` is **re-rooted in place** rather than rebuilt (rebuilding would re-wire ~10 objects and the run-queue seam).

`ExperimentManager.switch_session(folder) -> None`:

1. **Preconditions** (`session_busy_reason()` → `str | None`; switch refused with that reason):
   * no run active (engine state IDLE/ERROR and no run manifest, read from the status mirror);
   * run queue empty (the operator must clear it — queued specs carry the old experiment's context);
   * analysis runner has no active or pending request;
   * ELN publishing worker idle.
2. Persist the current experiment (record + GUI state + queue) — the experiment stays **open on disk**, so returning to that session resumes it exactly.
3. Push `set_run_folder("")`, clear envelope, clear `_experiment`.
4. Swap `_store = ExperimentStore(folder)`; `session_store.set_active(folder)` (also updates `sessions.json`, so `i2as-ctl` follows — removing today's mismatch after the dialog).
5. Adopt the new session's active experiment with the **live** variant of resume (`_adopt_active_experiment`), which does NOT mark `running` runs failed (startup-only crash recovery stays in `__init__`); re-push envelope, attendance, run folder (+ subfolder); reconcile the index.
6. Emit new `session_changed(str folder)`, then `experiment_changed(record | {})`.

### 4.2 Listeners on `session_changed`

* `ElnService.reset_session()`: clear `_outboxes` / `_confirmed_entries` caches (keyed by bare experiment id, which repeat across sessions), re-adopt outboxes from the new store.
* `ExperimentFeeds.reset()`: detach/drop cached agent feeds (keyed by experiment id); gateway connections resolve their feed per request (if today they resolve once per connection, change that to per request, or close agent connections on switch — decided in audit).
* `MonitorWindow`: reset `_last_session_experiment_id`, reload `gui_state` of the new active experiment (or blank), `_sync_context()`.
* `ExperimentInfoPanel`: reset `_last_experiment_id` / `_pre_session_data_dir`.
* `ProcedureWindow`: drop its in-memory queue/params snapshot of the old experiment (re-read on next show).

### 4.3 UI

* **User → Session Folder…** applies **immediately** (status text no longer says "on next launch"); on refusal a message box names the reason (e.g. "A run is in progress").
* The same busy guard is added to **switching experiment** (`switch_experiment` / `_switch_experiment`), which today has the same orphaned-run hazard.

## 5. Tests

* run_naming: recursive numbering across subfolders; placement in subfolder; containment refusal; old one-arg `set_run_folder`.
* session layer: `normalize_run_subfolder` accept/refuse table; subfolder persisted and restored on resume/switch; `switch_session` re-roots, adopts active experiment, does not fail runs, updates `sessions.json`; busy reasons refuse the switch; `switch_experiment` guarded.
* ELN/feeds reset on session change (same experiment id in two sessions does not cross).
* GUI: Data Dir shows the effective folder (incl. after login switch — regression for the bug); Subfolder… inside/outside; Session Folder… applies immediately; refusal while running.
* Update tests pinning "applies on next launch".

## 6. Docs

README "Where data lives" (operator now makes two choices: session folder, optional run subfolder; runs keep unique numbers across subfolders), docstrings in run_naming, orchestrator, procedure, experiment_info_panel, store, main, session_report.

## 7. Open questions for the audit

1. Queue non-empty at switch: refuse (proposed) vs. offer to clear.
2. Gateway agent connections during a session switch: per-request feed resolution vs. disconnect.
3. Should an agent be able to set the subfolder (proposed: no)?
4. Subfolder depth limit (proposed 3) and name rule.

## 8. Audit (senior lab-software review) and revised design

Verdict: **approve with changes** — §3 sound with the numbering/validation changes;
§4 not as written. Both blockers verified in the code before revising.

| # | Finding | Revision (supersedes §3–§4 where they differ) |
|---|---|---|
| B1 | Busy check on the lagging GUI mirror; a `run_procedure` posted just before the switch still starts with the old folder and its manifest is filed into whatever `self._experiment` is by then; the engine's own `_procedure_queue` is invisible to the session layer | (a) **Engine-verified release**: new engine command `release_run_folder()` answering by verdict — OK only when IDLE (or ERROR with no run), no procedure, `_procedure_queue` empty; it then clears the run folder atomically on the engine thread. The session layer re-roots **only after** that verdict is OK, and holds a *switching* flag meanwhile that refuses new runs from the GUI queue/gateway. (b) **Runs filed by folder, not by "current"**: the run-started manifest carries `experiment_dir` (the folder the engine placed it under); `_on_run_started`/`_on_run_finished`/`_on_engine_event` resolve the record by that folder (current experiment if it matches, otherwise load-mutate-save that record by path) — also fixes today's unguarded `switch_experiment`. |
| B2 | Run ids re-issued after the highest file is deleted/moved; directories counted; recursive scan on the instrument thread | **Monotonic number source**: session pushes `run_number_floor` = highest number in the experiment record + 1 with the folder; engine issues `max(floor, last_issued + 1, flat scan of the TARGET folder (files only))` and remembers `last_issued`. No recursive scan on the instrument thread. |
| M3 | ELN worker reads the live store and files by bare id | Capture the store (session root) when a drain is queued, carry it to `_drained`; outbox caches keyed by `(root, experiment_id)`; switch waits for an in-flight drain only. |
| M4 | Agent feeds never detached; gateway connections bound to a feed at admission | `AgentFeed.detach()`; `ExperimentFeeds.reset()` on session **and** experiment switch; agent gateway connections are closed with a "session changed" error on a session switch. |
| M5 | No rollback; no session-folder lock | Validate/load the target session and its active record **before** tearing down; write registry, then swap; restore on error. A `session.lock` (pid + host + time) is taken by the running app; a live switch refuses a folder locked by another live process. With the lock + released engine, a `running` run in the adopted session is stale and is marked failed (crash recovery applies on adoption too). |
| M6 | Subfolder rule must live in core; validate on load | `normalize_run_subfolder()` lives in `core/run_naming`; `set_run_folder` refuses a bad value by verdict; engine containment check before `_start_run`'s try; on load an invalid stored value falls back to `""` with a warning. |
| minor | Plan wrong about code: tool is `read_experiment`, command schema comes from method signature/docstring, queued specs carry no experiment context | Adjusted: `read_experiment` reports `run_folder`; `set_run_folder` docstring documents `subfolder`; queue is **parked**, not refused (see Q1). |
| minor | Unlisted consumers | `AnalysisRunner._select` keyed by (root, id); `SessionStore` rebinds to the logged-in user's registry on login; docstring/tooltip "next launch" updated. Long-lived `i2as-ctl` clients keep their store (documented). |
| minor | Windows paths | Subfolder rule: ≤ 3 levels, each part `[A-Za-z0-9][A-Za-z0-9._ -]{0,63}`, no trailing `.`/space, no reserved device names (CON, NUL, COMn, LPTn…), no comma; engine refuses a total run-file path over 240 chars and a target that resolves outside `data/` (junction/symlink). |
| minor | Back-compat | `run_subfolder` additive; `SCHEMA_VERSION` **not** bumped. |
| nit | `ProcedureWindow.reset_session()` exists | Reused. |

**Open questions, decided:**
1. Queue at switch: **park** — it is already saved with the old experiment's record/GUI state, so clear the live queue and it returns when that experiment is reopened; refuse only while the queue is draining or the engine queue is non-empty.
2. Agent connections: **disconnect** on session switch (and reset feeds on experiment switch).
3. Agents setting the subfolder: **no**; `set_run_folder` stays an ENVELOPE-class action; agents read it via `read_experiment`.
4. Depth 3 and the name rule above.

## 9. Implementation audit and response

Verdict on commit `04315c5`: **changes required, no blocker**. Addressed in the
follow-up commit:

| # | Finding | Response |
|---|---|---|
| M1 | Between release and verdict, start/close experiment or Subfolder… could re-install a folder, so a run could start in the session being left | `_install_run_folder` installs nothing while a switch is pending; a refused/failed/abandoned switch re-installs the current folder. Test covers it. |
| M2 | Startup lock only advisory; check-then-write race | Lock created with `O_CREAT\|O_EXCL`, stale same-host lock taken over. If another live station holds the session at startup, the app still starts, but **runs are held** (engine folder `""`) with a non-dismissible alert until another session is loaded. |
| M3 | Listener exceptions after the re-root could abort the app | Each `main.py` listener wrapped by `_guarded`; `MonitorWindow._on_session_changed` guarded. Test covers `_guarded`. |
| 4 | Notebook drain for the new session suppressed by a queued old-session drain; offline bookkeeping skipped | `_drain_again` re-drains once the old drain finishes; bookkeeping runs for every session. |
| 5 | Path-length limit on every OS, measured after `resolve()` | Windows only, measured with `abspath` as written. |
| 6 | Sibling-subfolder uniqueness rested on the record | `data/.last_run_number` marker records the highest number issued; placement reads it. |
| 7 | Placement refusal after side effects / after a queued run was popped | Containment refused in `run_procedure` and `run_queue` before any side effect, and at `set_run_folder` install. |
| 9 | Foreign-host lock never expires | Refusal message names the lock file to delete; verdict timeout (15 s) releases a stuck switch and its lock. |
| 11 | Edits during the pending window lost; blank state saved over the user's autosave | `session_about_to_change` saves the GUI before re-root; after a switch the user's own autosave is loaded, never a blank state. |
| nit | Feed `detach` placement; counter key by raw path | Fixed; key normalised with `normcase(resolve())`. |

Deferred, with reason:

* **8: `switch_experiment` does not park the queue.** Each experiment's GUI
  state already carries its own queue and the engine enforces the target
  experiment's envelope; parking would change long-standing experiment-switch
  behaviour beyond this request.
* **10: the HTTP MCP endpoint reconnects silently.** The local socket clients
  get `session/changed`; giving web clients an explicit error belongs in the
  HTTP adapter's session model and is a separate change.
* **Windows CI.** Junctions and long paths are covered by the rule and
  `_ON_WINDOWS` tests here; a Windows CI job is recommended but out of scope.

### Re-audit of the implementation fixes

| # | Finding | Response |
|---|---|---|
| BLOCKER | Holding runs AFTER the manager was built: the shared active experiment was already adopted (a run of the other station marked failed and saved), and later saves overwrote the shared record | The lock is decided before the manager is built and passed as `runs_held`; a held session adopts no experiment, refuses start/switch experiment and the run subfolder, and every write to its records (`_save_current`, `_mutate_experiment`, index reconcile) is refused until another session is loaded. Test asserts `experiment.json` is byte-identical afterwards. |
| MAJOR | A refused `set_run_folder` left the engine on the previous experiment's folder | Fail closed: a refusal clears the run folder, so every run is refused until a valid one is installed. |
| minor | Lock file created but not yet written read as free | An unreadable lock counts as held for 10 s, then as debris. |
| minor | `_drain_again` skipped when the old drain failed | Re-drain happens before the error return. |
| minor | `session_about_to_change` save unguarded | Guarded. |
