---
name: write-eln-renderer
description: Write or change an I2AS renderer block — the function that lays out what each publish appends to an experiment's notebook page (which results, tables, figures and text appear, in what order), and the YAML profile that maps notebook fields. Use when the user wants their notebook sections to look different or to fill template fields. Loops on `python -m i2as.blocks check` until every rule passes.
---

# Write a renderer (and its profile)

A renderer is a **tier-2 user block**: one Python module with one function,
`render(context) -> Section`. It receives the experiment and, for each run
being published, the run's facts and its selected **analysis bundle**
(summary, results, figures, tables, warnings, provenance), and returns a
`Section` of blocks. I2AS turns the section into safe HTML (everything
escaped, Markdown limited, HTML sanitised), uploads the figures it names,
and appends it to the page. The renderer runs in a helper process with no
network and no key.

## Steps

1. **Read** `i2as/blocks/renderer.py` (the contract and the block helpers:
   `heading`, `paragraph`, `markdown`, `html`, `table`, `results`, `figure`,
   `key_values`, `link`) and the shipped layout
   `i2as/blocks/shipped/renderers/default.py`.
2. **Scaffold** a copy of the default layout:
   `python -m i2as.blocks new-renderer <name>` (writes
   `<user config>/blocks/renderers/<name>.py`).
3. **Change the layout** to what the user wants. Use only `context` (a
   `RenderContext`): `context.runs[i].bundle` may be `None` (present the run
   from its facts). Name figures only from a bundle's own artifacts:
   `figure(bundle.bundle_id, artifact.artifact_id, caption)`. Put page fields
   in `Section.fields` if the layout computes them.
4. **Check, and loop until it passes**:
   `python -m i2as.blocks check <file> --preview preview.html`
   Open `preview.html` to see the sample publish. Fix every `FAIL` line.
5. **Point a profile at it**: `python -m i2as.blocks new-profile <name>`, set
   `renderer: <NAME>`, and map fields:
   - `read:` notebook fields → the experiment's sample metadata
     (`thickness: {from: "sample:Thickness", type: float, unit: nm}`);
   - `fields:` page fields ← `result:<name>`, `experiment:title`,
     `sample:<key>`, `text:<literal>` (overwritten on every publish).
   Check it: `python -m i2as.blocks check <profile.yaml>`.
6. Tell the user to choose the profile in Settings → Electronic notebook
   (new experiments), or re-link an experiment to use it. An experiment
   already linked keeps the copies it was linked with.

## Rules

- `render` is a pure function of `context`: no network (it is switched off),
  no files, no global state, no randomness — the same publish renders the
  same section.
- Keep it fast (the checker allows 10 s) and small.
