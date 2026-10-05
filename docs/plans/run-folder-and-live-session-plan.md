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
