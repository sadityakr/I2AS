# I2AS

A safe agentic experimentation and analysis station built from a rack of instruments.

## Philosophy

I2AS provides the robust core that is hard to vibe-code: safety, the hardware
thread, orchestration, queuing, publishing, permissions and reflection. Users,
often with coding agents, add small blocks that they customise and improve
over time. **No user or agent code may ever affect a running experiment or the
operator's ability to run one.** The less trusted the code, the further from the
instrument thread it runs.

| Tier | Code | Runs in |
|---|---|---|
| 0 Core | engine, safety, session, gateway | instrument thread / GUI thread |
| 1 Certified | drivers, VIs, procedures (pass conformance tests) | instrument thread |
| 2 User blocks | ELN connectors, renderers, profiles (vibe-coded) | helper process, killable |
| 3 Agent code | analysis recipes and scripts | container, no network |

A block's tier is set by its kind (what the code is for), never earned or promoted.
Every block kind ships a contract, a scaffold, a checker and a skill in `.claude/skills/`.

## Structure

```
i2as/
  drivers/               L0  instrument I/O (sim twins)
  virtual_instruments/   L1  @monitored / @control declarations
  core/                  L2–L5  Station, Orchestrator, procedures base, data
  procedures/            L4  experiment recipes
  session/               L6  experiments, gateway, analysis runner, notebook bridge (eln/)
  analysis/              worker code (runs in the container) + the bundle schema
  blocks/                tier-2 block contracts, host, checker, shipped blocks
  gui/  mcp/  ctl/       the GUI, MCP and CLI surfaces (one state, reflected)
  configs/               per-rack devices.yaml
```

Layer boundaries are enforced by import contracts (`pyproject.toml`). `make check` runs lint, the contracts and the tests.
