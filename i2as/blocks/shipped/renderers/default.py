"""The default layout: one sub-section per run — the result over its provenance.

This is an ordinary renderer block, shipped as the worked example. Copy it to
make your own (``python -m i2as.blocks new-renderer my_layout --dir <folder>``)
and point a profile's ``renderer:`` at the copy.

For each run: its analysis summary, derived values, figures, tables and
warnings (from the run's selected bundle), or — with no bundle — its
parameters; then a compact provenance table naming the run, the data file and
the code that produced the numbers; and a link to the per-run entry an older
I2AS created, when there is one.

Profile options (``render:`` in the profile YAML):

* ``include_parameters`` (bool, default ``true``): list the run's parameters.
* ``include_warnings`` (bool, default ``true``): list the analysis warnings.
"""

from __future__ import annotations

from pathlib import PurePath

from i2as.blocks import (
    RenderContext,
    RunContext,
    Section,
    figure,
    heading,
    key_values,
    link,
    markdown,
    paragraph,
    results,
    table,
)

NAME = "default"
DESCRIPTION = "One sub-section per run: summary, results, figures, tables and provenance."
CONTRACT_VERSION = 1


def _run_title(run: RunContext) -> str:
    """Return one run's sub-section title."""
    return f"{run.run_id} — {run.procedure}" if run.procedure else run.run_id


def _run_blocks(run: RunContext, options: dict) -> list[dict]:
    """Return the blocks presenting one run."""
    blocks: list[dict] = [heading(_run_title(run), 3)]
    bundle = run.bundle
    if run.status and run.status != "done":
        blocks.append(paragraph(f"Run outcome: {run.status}. {run.reason}".strip()))
    if bundle is not None and not bundle.ok:
        blocks.append(paragraph(f"Analysis failed: {bundle.error.strip().splitlines()[0] if bundle.error.strip() else 'no result'}"))
    if bundle is not None and bundle.ok:
        if bundle.summary:
            blocks.append(markdown("\n\n".join(bundle.summary)))
        if bundle.results:
            blocks.append(results("Results", [dict(r) for r in bundle.results]))
        for artifact in bundle.artifacts:
            if artifact.kind == "figure":
                blocks.append(figure(bundle.bundle_id, artifact.artifact_id, artifact.caption))
        for spec in bundle.tables:
            blocks.append(table(str(spec.get("caption", "")), list(spec.get("columns") or []), list(spec.get("rows") or [])))
        if bundle.warnings and options.get("include_warnings", True):
            blocks.append(markdown("\n".join(f"- {w}" for w in bundle.warnings)))
    if (bundle is None or bundle.hints.get("include_fact_tables")) and options.get("include_parameters", True) and run.params:
        blocks.append(key_values("Parameters", sorted((str(k), v) for k, v in run.params.items())))
    provenance: list[tuple[str, object]] = [
        ("Run id", run.run_id),
        ("Procedure", run.procedure),
        ("Started (UTC)", run.started_utc),
        ("Finished (UTC)", run.finished_utc),
        ("Data file", PurePath(run.data_file).name if run.data_file else ""),
    ]
    if bundle is not None:
        provenance += [
            ("Analysis", f"{bundle.producer.kind} {bundle.producer.name}".strip()),
            ("Code digest", bundle.producer.digest[:16]),
            ("Bundle", bundle.bundle_id),
        ]
    blocks.append(key_values("Provenance", [(k, v) for k, v in provenance if v not in (None, "")]))
    if run.legacy_entry_url:
        blocks.append(link("Earlier notebook entry for this run", run.legacy_entry_url))
    return blocks


def render(context: RenderContext) -> Section:
    """Lay out one publish: every run being published, oldest first."""
    options = dict(context.options)
    blocks: list[dict] = []
    for run in context.runs:
        blocks.extend(_run_blocks(run, options))
    tags: list[str] = ["i2as"]
    for run in context.runs:
        if run.bundle is not None:
            tags.extend(run.bundle.tags)
    runs = ", ".join(run.run_id for run in context.runs)
    return Section(
        title=runs or "Experiment update",
        blocks=tuple(blocks),
        fields={},
        tags=tuple(dict.fromkeys(tags)),
    )
