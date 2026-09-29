# ELN, analysis and publishing: three independent layers

Status: **implemented (2026-09-27).** Not built yet: bundles over several runs
(`analysis/experiment/`) and a per-run "publish only this run" button in the
GUI (`ElnService.publish(run_ids)` already takes a selection).

## Goal

The analysis stage and eLabFTW publishing are separated into three layers that hand
off through files only:

| Layer | Job | Knows about | Tier (see CLAUDE.md) |
|---|---|---|---|
| **Analysis** | Turns runs into an ELN-agnostic, sealed **bundle** in `analysis/` | runs only | 3 (container) |
| **ELN connection** | Links one ELN page to one experiment; reads fields back into sample metadata | ELN only | 2 (helper process) |
| **Publishing** | Renders the selected bundles and appends them to the experiment's page | bundles + binding | core, running tier 2 blocks |

Only an eLabFTW connector is shipped (with a simulated twin for tests and
demos). Any other ELN, and any custom layout, is a user block.

Import contracts C25–C27 (`pyproject.toml`) keep the layers apart: the block
package imports only the standard library, `ruamel.yaml` and the bundle
schema; the analysis layer never imports the notebook bridge; the notebook
bridge reads analysis only through `i2as.analysis.bundle`.

## 1. Analysis bundle (`i2as/analysis/bundle.py`)

One producer execution creates one immutable folder. A producer is a recipe, a
script or a model draft.

```
<experiment>/analysis/
  recipes/
  run-0012/<bundle_id>/         report.json (claims)  bundle.json (seal)  *.png  *.plot.json  *.csv  *.md
  run-0012/scripts/<script_id>/ a script's folder is the bundle "script-<script_id>"
```

- The worker writes **claims** (`report.json` and its files). After it exits,
  `AnalysisRunner` **seals** the folder (`seal_bundle`): each file is checked by
  `output_file()`, hashed and listed; the report's content is copied in from the
  parsed, capped report. Everything downstream reads only `bundle.json`, and
  `verify_artifact()` refuses a file changed since sealing.
- Re-running a recipe creates a new bundle; nothing is overwritten.
- `RunRecord.selected_bundle` records which bundle represents the run. A
  completed recipe bundle is selected automatically; any other one through
  `ExperimentManager.select_bundle()` (the `select_analysis_bundle` tool, or
  "Use for this run" in the Analysis tab). Only a sealed, completed bundle can
  be selected.
- A draft (`draft_analysis_summary`, `i2as/session/drafting.py`) is a bundle of
  kind `draft` holding `summary.md`.
- Whether a finished run is analysed at all is decided by `AnalysisTrigger`
  (`session/analysis_trigger.py`) from the analysis settings alone.

`bundle.json`, schema `i2as.analysis-bundle`, v1:

| Field | Required | Content |
|---|---|---|
| `schema`, `schema_version`, `bundle_id`, `created_utc`, `status` | ✔ | `status` is `ok` or `failed` (with `error`) |
| `scope` | ✔ | `{experiment_id, run_ids[]}` |
| `producer` | ✔ | `{kind: recipe\|script\|draft\|legacy, name, digest, actor, model, prompt_digest}` |
| `inputs` | ✔ | `[{run_id, data_file, params_digest}]` |
| `artifacts` | ✔ | `[{id, kind: figure\|plot_data\|table\|text, path, media_type, sha256, bytes, caption}]` |
| `summary`, `results`, `tables`, `tags`, `warnings`, `options`, `hints`, `duration_s` | optional | as in the report; `hints` are advice to a renderer |

A folder with only a `report.json` (from before bundles) reads as an unsealed
v0 bundle; it can be viewed but not selected or published.

## 2. ELN connection (per experiment)

When the experiment's user publishes to a notebook, starting an experiment
opens **Link notebook page**: link an existing page (searched), create a new
one from the profile's template (queued in the outbox, so it works offline),
link samples and resources, or "Not now". The same dialog is "Link page…" in
the Analysis tab.

`ExperimentRecord.eln` is an `ElnBinding` (`SCHEMA_VERSION` 3; a schema-2
experiment link becomes the binding's page): the account, the pinned
connector/renderer/profile, the template, the page (`entry`, or
`create_pending`), the linked items, the last fields read back, and the
publishing approval (`publish_approved`, by whom, when).

**Pinning.** At link time the profile and the renderer are COPIED into
`<experiment>/eln/profile.yaml` and `renderer.py`; those copies are what run
for that experiment. The connector is pinned by digest: a connector whose code
changed is not used for the experiment until someone confirms it
(`ElnService.confirm_connector()`), so an edited block never silently changes
an ongoing experiment.

**Read-back.** "Read fields…" reads the page and the linked items through the
profile's read map and shows each value beside the experiment's own; only the
ticked values are applied (`ExperimentManager.apply_eln_fields`) to
`sample_info`, which every later run stamps into its HDF5 file. Runs already
written never change.

## 3. Publishing (append-only)

```
finished, unpublished runs + their selected bundles + binding + pinned profile
  → renderer (helper process, no network) → Section
  → i2as.blocks.markup → safe HTML headed "I2AS · <UTC> · <title> · <publish_id>"
  → outbox job → connector (helper process): upload figures, append section, set fields
```

- **One approval per experiment** ("Approve publishing"); after it, "Publish
  new runs" appends one section covering every finished run not yet on the
  page. A run published again gets a new section; the old one stays.
- Earlier sections and text people wrote are never touched. The profile's
  **fields** are overwritten with the latest values.
- Figures are uploaded as `<run>_<file>`, once per page (the ledger,
  `<experiment>/eln/ledger.json`, remembers each by its SHA-256).
- **After a crash** a retry asks the connector whether the publish id is
  already on the page before appending.
- Old per-run eLab entries are linked from their run's new section.

**Failures** (`ElnError` subclasses, from `i2as.blocks.connector`):

| Error | Behaviour |
|---|---|
| Transient (`ElnTransientError`, 5xx/429, timeout, a block that overran) | Retried with backoff, indefinitely |
| `ElnAuthError` | `needs_attention`; retried automatically once a new key is saved |
| `ElnValidationError`, `ElnNotFound` | `needs_attention` with the reason; "Retry" in the Analysis tab |
| The connector crashed (a bug) | `needs_attention` |
| The connector's code changed | `needs_attention` until confirmed |
| The renderer failed | Nothing is queued; the Analysis tab says why |

## 4. User blocks (tier 2, `i2as/blocks/`)

A block is written against `i2as.blocks` and shipped or user-provided:

- shipped: `i2as/blocks/shipped/{connectors,renderers,profiles}/` —
  `elabftw.py`, `sim.py`, `default.py`, `default.yaml`;
- the user's: `<user config>/blocks/{connectors,renderers,profiles}/` (a user
  block with a shipped block's id replaces it).

Blocks are **discovered statically** (`ast`): listing them, filling the
Settings form from a connector's `settings_schema` and pinning an experiment
never execute user code in the application.

| Kind | Contract | Scaffold / check |
|---|---|---|
| Connector | `i2as/blocks/connector.py`: `ElnConnector` with exactly `verify`, `list_templates`, `search`, `get_record`, `create_entry`, `append_section`, `has_section`, `set_fields`, `upload`, `link_item`; `__init__(settings, credential="", transport=None)`; failures only as `ElnError` subclasses (`i2as/blocks/http.py` classifies HTTP statuses) | `python -m i2as.blocks new-connector <id>` |
| Renderer | `i2as/blocks/renderer.py`: `render(context: RenderContext) -> Section`, with `NAME` | `python -m i2as.blocks new-renderer <name>` |
| Profile | `i2as/blocks/profile.py`: `connector`, `renderer`, `template`, `read`, `fields`, `render` | `python -m i2as.blocks new-profile <name>` |

`python -m i2as.blocks check <file>` runs the contract checks (a connector
against a fake transport answering 500 and 401; a renderer over a sample
publish with the network off and a time limit; a profile's parse), and
`python -m i2as.blocks list` shows what is installed. The skills
`.claude/skills/write-eln-connector` and `write-eln-renderer` teach a coding
agent to write one and loop on the checker.

**Isolation.** Every block runs in `python -m i2as.blocks.host`
(`session/eln/block_runner.py`): JSON lines over stdio, a scrubbed
environment, a temporary working directory, a timeout per call after which the
process is killed. A connector receives its one credential in the `load`
request on stdin; a renderer receives none and has its sockets disabled. The
GUI thread and the instrument thread never wait on a helper: the notebook
service (`session/eln/service.py`) calls blocks from its own worker thread.

## 5. Settings, credentials, per-user state

- **Machine-wide** `settings.json`: `connections`, `analysis`, `publishing`
  (retry timings, upload cap, block timeout).
- **Per user** `<user config>/users/<user_id>/profile.yaml`
  (`session/user_profile.py`): `eln` (enabled, accounts, default account,
  profile and template), `assistant` (the drafting model), `sessions` (active
  and recent; `sessions.json` keeps naming the last active one for `i2as-ctl`).
  The old `eln-settings.json` is migrated into the first user's profile once.
- **Credentials** (`session/credentials.py`): the OS keyring (`keyring`),
  keyed `eln/<account>/<user>` and `assistant/default/<user>`; an owner-only
  file when there is no keyring; `I2AS_ELN_APIKEY` (and the older
  `I2AS_ELAB_APIKEY`) override. No secret reaches an agent, a tool result, a
  log, a settings file or a renderer.
- **GUI**: Settings → **Electronic notebook** (the form is rendered from the
  connector's `settings_schema`); the Analysis tab's notebook strip; the
  experiment panel names the linked page.

## 6. Tools and reflection

| Before | Now | Class |
|---|---|---|
| `draft_eln_entry` | `draft_analysis_summary` (writes a `draft` bundle) | analysis, recorded |
| `stage_analysis_result` | `select_analysis_bundle` | analysis, recorded |
| `read_analysis_report` | `read_analysis_bundle`, `list_analysis_bundles` | read |
| `publish_eln_entry` | removed | — |
| — | `read_settings` (the machine's settings; no secret) | read |

**Agents get no notebook tool.** They read an experiment's local analysis
only; `read_experiment` tells them whether a page is linked, not where. The
permission matrix has an `eln` action class refused to every agent role, so
notebook tools can be opened later by changing one row.

Every change to an experiment's notebook state goes through the
`ExperimentManager` (`link_eln`, `set_eln_entry`, `approve_eln_publishing`,
`apply_eln_fields`, `record_eln_publish`, `select_bundle`) and is announced on
`experiment_changed` / `run_recorded`, so every window shows it.
