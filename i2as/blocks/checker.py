"""The block checker — does a block follow its contract, before it touches anything real?

``python -m i2as.blocks check <file>`` runs here. It is what a person — or a
coding agent building a block — loops on until every line says ``ok``:

* a **connector** is described statically, imported, and its class checked:
  exactly the contract's public methods, the constructor signature, a usable
  ``settings_schema``; then it is built against a fake transport that answers
  every request with an error, and every method must fail with an
  ``ElnError`` (never another exception), 401 must be an ``ElnAuthError``,
  and 503 must not be an auth or validation failure;
* a **renderer** is run over a sample publish (a run with a full bundle, a run
  without one) with the network off and a time limit; its section must load,
  name only figures that exist, and render to HTML;
* a **profile** is parsed and every dropped entry named.

The checker runs a block in ITS OWN process: it is a development tool the
author runs on their own file, never something the application does.
"""

from __future__ import annotations

import inspect
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from i2as.analysis.bundle import Artifact, Bundle, Producer
from i2as.blocks.connector import (
    CONNECTOR_METHODS,
    ElnAuthError,
    ElnEntryRef,
    ElnError,
    ElnQuery,
    ElnRef,
    ElnValidationError,
)
from i2as.blocks.discovery import (
    KIND_CONNECTOR,
    KIND_PROFILE,
    KIND_RENDERER,
    connector_class,
    describe_block,
    load_block_module,
)
from i2as.blocks.http import HttpResponse
from i2as.blocks.markup import section_to_html
from i2as.blocks.renderer import RenderContext, RunContext, Section

#: How long a renderer may take over the sample publish.
RENDER_TIME_LIMIT_S = 10.0

_SCHEMA_TYPES = ("string", "boolean", "number", "integer")


@dataclass(frozen=True)
class Check:
    """One line of the checker's answer.

    Attributes:
        ok: Whether the rule holds.
        rule: What was checked.
        detail: Why it failed, or ``""``.
    """

    ok: bool
    rule: str
    detail: str = ""

    def __str__(self) -> str:
        mark = "ok  " if self.ok else "FAIL"
        return f"{mark} {self.rule}" + (f" -- {self.detail}" if self.detail else "")


class _FailingTransport:
    """Answers every request with one status."""

    def __init__(self, status: int) -> None:
        self.status = status
        self.requests = 0

    def request(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout_s: float) -> HttpResponse:
        self.requests += 1
        return HttpResponse(status=self.status, body=b'{"message": "checker"}')


def kind_of(path: str | Path) -> str:
    """Guess a block's kind from its file and folder."""
    file = Path(path)
    if file.suffix.lower() in (".yaml", ".yml"):
        return KIND_PROFILE
    if file.parent.name == "renderers":
        return KIND_RENDERER
    if file.parent.name == "connectors":
        return KIND_CONNECTOR
    text = file.read_text(encoding="utf-8", errors="replace") if file.is_file() else ""
    return KIND_CONNECTOR if "ElnConnector" in text else KIND_RENDERER


def _sample_args(settings_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return settings built from a schema's defaults."""
    properties = settings_schema.get("properties") if isinstance(settings_schema, Mapping) else None
    settings: dict[str, Any] = {}
    for name, spec in (properties or {}).items():
        if isinstance(spec, Mapping):
            settings[name] = spec.get("default", "" if spec.get("type") == "string" else None)
    if not settings.get("base_url"):
        settings["base_url"] = "https://notebook.invalid"
    settings["state_file"] = str(Path(tempfile.mkdtemp(prefix="i2as-check-")) / "state.json")
    return settings


def check_connector(path: str | Path) -> list[Check]:
    """Check one connector file against the connector contract."""
    checks: list[Check] = []
    info = describe_block(KIND_CONNECTOR, path)
    checks.append(Check(info.usable, "declares backend, contract_version and literal settings", info.error))
    if not info.usable:
        return checks
    try:
        cls = connector_class(load_block_module(path))
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        checks.append(Check(False, "imports, with exactly one ElnConnector subclass", str(exc)))
        return checks
    checks.append(Check(True, "imports, with exactly one ElnConnector subclass"))
    missing = sorted(getattr(cls, "__abstractmethods__", ()) or ())
    checks.append(Check(not missing, "implements every contract method", ", ".join(missing)))
    public = {
        name
        for klass in cls.__mro__
        if klass.__module__ == cls.__module__
        for name, value in vars(klass).items()
        if not name.startswith("_") and callable(value)
    }
    extra = sorted(public - set(CONNECTOR_METHODS))
    checks.append(Check(not extra, "adds no public method of its own", ", ".join(extra)))
    parameters = list(inspect.signature(cls.__init__).parameters)[1:4]
    checks.append(
        Check(
            parameters == ["settings", "credential", "transport"],
            "__init__(self, settings, credential='', transport=None)",
            "" if parameters == ["settings", "credential", "transport"] else f"found {parameters}",
        )
    )
    schema = info.settings_schema
    bad = [
        name
        for name, spec in (schema.get("properties") or {}).items()
        if not isinstance(spec, dict) or spec.get("type") not in _SCHEMA_TYPES
    ]
    checks.append(Check(not bad, "settings_schema properties have a supported type", ", ".join(bad)))
    entry = ElnEntryRef(backend=info.block_id, entry_id="1")
    calls = {
        "verify": (),
        "list_templates": (),
        "search": (ElnQuery(text="x"),),
        "get_record": (ElnRef(record_id="1"),),
        "create_entry": ("title", "", {}),
        "append_section": (entry, "P-check", "<p>x</p>"),
        "has_section": (entry, "P-check"),
        "set_fields": (entry, {"a": "b"}),
        "upload": (entry, Path(__file__), "caption"),
        "link_item": (entry, ElnRef(kind="item", record_id="1")),
    }
    for status in (500, 401):
        transport = _FailingTransport(status)
        try:
            connector = cls(_sample_args(schema), "checker-key", transport)
        except Exception as exc:  # noqa: BLE001
            checks.append(Check(False, f"builds from its settings (HTTP {status} transport)", str(exc)))
            continue
        wrong: list[str] = []
        for method, args in calls.items():
            before = transport.requests
            try:
                getattr(connector, method)(*args)
            except ElnError as exc:
                # A method may refuse before any request (an unsupported
                # feature); only an answer the SERVER gave is classified.
                asked = transport.requests > before
                if status == 401 and asked and not isinstance(exc, ElnAuthError):
                    wrong.append(f"{method}: HTTP 401 raised {type(exc).__name__}, not ElnAuthError")
                if status == 500 and asked and isinstance(exc, (ElnAuthError, ElnValidationError)):
                    wrong.append(f"{method}: HTTP 500 raised {type(exc).__name__}")
            except Exception as exc:  # noqa: BLE001
                wrong.append(f"{method}: raised {type(exc).__name__} ({exc}), not an ElnError")
        checks.append(Check(not wrong, f"every method fails only with the right ElnError (HTTP {status})", "; ".join(wrong)))
    return checks


def sample_context() -> RenderContext:
    """Return the sample publish a renderer is checked against."""
    bundle = Bundle(
        bundle_id="20260101T000000Z-sample-0000",
        experiment_id="001_check",
        run_ids=("run-0001",),
        producer=Producer(kind="recipe", name="sample_recipe", digest="0" * 64),
        artifacts=(Artifact("overview.png", "figure", "overview.png", "image/png", "0" * 64, 10, "Overview"),),
        summary=("The resistance rises linearly with field.",),
        results=({"name": "R0", "value": 101.5, "unit": "Ω", "uncertainty": 0.2, "note": "fit"},),
        tables=({"caption": "Points", "columns": ["B (T)", "R (Ω)"], "rows": [[0.0, 101.5], [1.0, 102.0]], "truncated": False},),
        tags=("magnetoresistance",),
        warnings=("A sample warning.",),
    )
    return RenderContext(
        publish_id="P-20260101T000000Z-check",
        published_utc="2026-01-01T00:00:00+00:00",
        experiment={"experiment_id": "001_check", "title": "Checker", "user_name": "Checker", "sample_info": {"sample_id": "S-001"}},
        runs=(
            RunContext(run_id="run-0001", procedure="FieldSweep", params={"start_T": 0.0}, status="done", bundle=bundle),
            RunContext(run_id="run-0002", procedure="TimeSeries", params={"duration_s": 10}, status="aborted", reason="operator"),
        ),
        options={},
    )


def check_renderer(path: str | Path, preview: str | Path | None = None) -> list[Check]:
    """Check one renderer file against the renderer contract."""
    from i2as.blocks.host import disable_network, restore_network

    checks: list[Check] = []
    info = describe_block(KIND_RENDERER, path)
    checks.append(Check(info.usable, "declares NAME and a render(context) function", info.error))
    if not info.usable:
        return checks
    # Off while the renderer is imported and run, back on afterwards: the
    # checker runs in the author's own process, not in a helper.
    originals = disable_network()
    try:
        return _check_renderer_offline(path, checks, preview)
    finally:
        restore_network(originals)


def _check_renderer_offline(path: str | Path, checks: list[Check], preview: str | Path | None) -> list[Check]:
    """The renderer checks proper, run with the network switched off."""
    try:
        render = getattr(load_block_module(path), "render")
    except Exception as exc:  # noqa: BLE001
        checks.append(Check(False, "imports", str(exc)))
        return checks
    context = sample_context()
    outcome: dict[str, Any] = {}

    def _run() -> None:
        try:
            outcome["value"] = render(context)
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(RENDER_TIME_LIMIT_S)
    if worker.is_alive():
        checks.append(Check(False, f"renders the sample publish within {RENDER_TIME_LIMIT_S:.0f} s", "still running"))
        return checks
    if "error" in outcome:
        checks.append(Check(False, "renders the sample publish without raising", repr(outcome["error"])))
        return checks
    checks.append(Check(True, "renders the sample publish without raising"))
    value = outcome.get("value")
    section = Section.from_dict(value.to_dict() if isinstance(value, Section) else value)
    checks.append(Check(bool(section.blocks), "returns a Section with at least one block"))
    known = {(b.bundle_id, a.artifact_id) for r in context.runs if (b := r.bundle) for a in b.artifacts}
    unknown = [f"{f['bundle_id']}/{f['artifact_id']}" for f in section.figures() if (f["bundle_id"], f["artifact_id"]) not in known]
    checks.append(Check(not unknown, "names only figures that exist in the bundles", ", ".join(unknown)))
    html = section_to_html(section, publish_id=context.publish_id, published_utc=context.published_utc, figure_names={k: k[1] for k in known})
    checks.append(Check(context.publish_id in html, "renders to HTML carrying the publish id"))
    if preview:
        Path(preview).write_text(f"<!doctype html><meta charset=utf-8>{html}", encoding="utf-8")
    return checks


def check_profile(path: str | Path) -> list[Check]:
    """Check one profile file."""
    from i2as.blocks.profile import load_profile

    profile = load_profile(path)
    checks = [Check(not profile.errors, "parses with no dropped entry", "; ".join(profile.errors))]
    checks.append(Check(bool(profile.renderer), "names a renderer"))
    return checks


def check_block(path: str | Path, preview: str | Path | None = None) -> list[Check]:
    """Check any block file, choosing the checks by its kind."""
    kind = kind_of(path)
    if kind == KIND_CONNECTOR:
        return check_connector(path)
    if kind == KIND_RENDERER:
        return check_renderer(path, preview)
    return check_profile(path)
