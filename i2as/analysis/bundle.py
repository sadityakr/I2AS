"""The analysis bundle — the sealed, ELN-agnostic hand-off out of the analysis stage.

**The bundle standard.** One execution of one producer (a recipe, an analysis
script, a model draft) over one or more finished runs leaves exactly one
**bundle**: a folder holding what the producer wrote plus ``bundle.json``,
the manifest everything downstream reads. The notebook, the publisher, a
renderer block and an agent's ``read_analysis_bundle`` tool read a bundle and
nothing else; none of them knows which producer made it, and the analysis
stage knows nothing about any notebook.

**Claims, then a seal.** The analysis worker runs code this application does
not trust (tier 3), so what it writes — ``report.json`` and its files — is only
a CLAIM. After the worker exits, the application SEALS the folder with
``seal_bundle()``: every file is checked by ``output_file()`` (a plain name,
an allowed suffix, inside the folder, under the size cap), hashed, and listed
as an ``Artifact``; the report's own content is copied in from the report the
caller already parsed and capped. ``bundle.json`` is written by the
application alone, never by the worker, and a bundle is never rewritten: a
second analysis of the same run is a second bundle.

**Tolerant, like every record here.** ``Bundle.from_dict()`` never raises on
junk, and ``read_bundle()`` answers ``None`` rather than raising for a folder
that is absent, oversized or unreadable. A folder written before bundles
existed (a bare ``report.json``) reads as an unsealed version-0 bundle.

**Standalone.** This module imports the standard library only, so the
publisher, the block host and a renderer written by a user can load a bundle
without importing numpy, h5py or anything else of the analysis stage.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: The manifest's file name inside a bundle folder.
BUNDLE_FILENAME = "bundle.json"

#: The ``schema`` value every sealed bundle carries.
BUNDLE_SCHEMA = "i2as.analysis-bundle"

#: The version this code writes. A MAJOR bump is a breaking change; readers
#: stay tolerant of fields they do not know.
BUNDLE_SCHEMA_VERSION = 1

#: The report file the worker writes (its claims). Named here too so this
#: module needs nothing from ``report.py``.
CLAIMS_FILENAME = "report.json"

#: Files the framework itself writes into a bundle folder; never artifacts.
FRAMEWORK_FILES: frozenset[str] = frozenset({BUNDLE_FILENAME, CLAIMS_FILENAME, "spec.json"})

#: Terminal status of a bundle whose producer ran to completion.
BUNDLE_OK = "ok"

#: Terminal status of a bundle whose producer raised, timed out or never ran.
BUNDLE_FAILED = "failed"

#: Producer kinds.
PRODUCER_RECIPE = "recipe"
PRODUCER_SCRIPT = "script"
PRODUCER_DRAFT = "draft"
PRODUCER_LEGACY = "legacy"
PRODUCER_KINDS: tuple[str, ...] = (
    PRODUCER_RECIPE,
    PRODUCER_SCRIPT,
    PRODUCER_DRAFT,
    PRODUCER_LEGACY,
)

#: The id a pre-bundle ``analysis/<run>/report.json`` is read under.
LEGACY_BUNDLE_ID = "legacy"

#: Prefix of the bundle id an analysis script's folder is known by.
SCRIPT_BUNDLE_PREFIX = "script-"

#: Artifact kinds, by file suffix (lower case). The longest matching suffix
#: wins, so ``fit.plot.json`` is plot data, not a generic file.
ARTIFACT_SUFFIXES: dict[str, tuple[str, str]] = {
    ".png": ("figure", "image/png"),
    ".plot.json": ("plot_data", "application/json"),
    ".csv": ("table", "text/csv"),
    ".md": ("text", "text/markdown"),
    ".txt": ("text", "text/plain"),
}

#: The largest single artifact the application will seal, show or upload.
MAX_ARTIFACT_BYTES = 20_000_000

#: Largest number of artifacts sealed into one bundle.
MAX_ARTIFACTS = 64

#: The largest ``bundle.json`` the application will read back.
MAX_BUNDLE_BYTES = 2_000_000

#: The suffixes a figure file may have (kept for ``output_file``'s default).
FIGURE_SUFFIXES: tuple[str, ...] = (".png",)

_BUNDLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _text(value: object, limit: int = 4000) -> str:
    """Coerce ``value`` to a bounded string (``None`` → ``""``)."""
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _dict(value: object) -> dict[str, Any]:
    """Return ``value`` if it is a dict with string keys, else ``{}``."""
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}


def _list(value: object) -> list[Any]:
    """Return ``value`` if it is a list, else ``[]``."""
    return list(value) if isinstance(value, list) else []


def _int(value: object, default: int = 0) -> int:
    """Return ``value`` if it is a non-bool int, else ``default``."""
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def is_bundle_id(name: object) -> bool:
    """Whether ``name`` is a usable bundle id: one plain path segment.

    Args:
        name: The candidate.

    Returns:
        ``True`` for a short name of letters, digits, ``_ . -`` that does not
        start with a dot or a dash.
    """
    return isinstance(name, str) and bool(_BUNDLE_ID.match(name)) and ".." not in name


def new_bundle_id(label: str, now: datetime | None = None) -> str:
    """Return a fresh, time-sortable bundle id.

    Args:
        label: What produced it (a recipe name); slugged into the id.
        now: The time to stamp; ``None`` for now.

    Returns:
        e.g. ``"20260927T140211Z-magnetoresistance-3f9a"``.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", label or "analysis").strip("_")[:40] or "analysis"
    return f"{stamp}-{slug}-{secrets.token_hex(2)}"


def script_bundle_id(script_id: str) -> str:
    """Return the bundle id an analysis script's folder is known by.

    Args:
        script_id: The script's id.

    Returns:
        ``"script-<script_id>"``.
    """
    return f"{SCRIPT_BUNDLE_PREFIX}{script_id}"


def artifact_kind(name: str) -> tuple[str, str] | None:
    """Return ``(kind, media_type)`` for a file name, or ``None`` if not allowed.

    Args:
        name: A plain file name.

    Returns:
        The kind and media type of the longest matching suffix, or ``None``.
    """
    lowered = name.lower()
    for suffix in sorted(ARTIFACT_SUFFIXES, key=len, reverse=True):
        if lowered.endswith(suffix) and len(lowered) > len(suffix):
            return ARTIFACT_SUFFIXES[suffix]
    return None


def output_file(
    directory: str | Path,
    name: str,
    *,
    suffixes: tuple[str, ...] = FIGURE_SUFFIXES,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> Path | None:
    """Resolve one file a producer names, trusting nothing about the name.

    A name written by the analysis worker is only a claim. It is honoured only
    when it is a plain file name (no directory part, no absolute path, not
    hidden) with an allowed suffix, the file exists directly in *directory*,
    and it is no larger than *max_bytes*. Everything that reads a bundle's
    files goes through here, so a producer cannot name ``gateway.json`` or a
    settings file as its "figure".

    Args:
        directory: The bundle's own folder.
        name: The file name claimed.
        suffixes: The allowed suffixes, lower case.
        max_bytes: The largest file accepted.

    Returns:
        The file's path, or ``None`` when the claim is refused.
    """
    if not isinstance(name, str) or not name or Path(name).name != name:
        return None
    if name.startswith(".") or not any(name.lower().endswith(s) for s in suffixes):
        return None
    folder = Path(directory).resolve()
    path = (folder / name).resolve()
    if path.parent != folder or not path.is_file():
        return None
    try:
        if path.stat().st_size > max_bytes:
            return None
    except OSError:
        return None
    return path


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Artifact:
    """One file inside a bundle, as sealed by the application.

    Attributes:
        artifact_id: Stable id inside the bundle (the file name).
        kind: ``figure`` / ``plot_data`` / ``table`` / ``text`` / ``file``.
        path: File name relative to the bundle folder.
        media_type: The file's media type.
        sha256: Hex digest taken when the bundle was sealed.
        bytes: File size when sealed.
        caption: The producer's caption, or ``""``.
    """

    artifact_id: str
    kind: str = "file"
    path: str = ""
    media_type: str = "application/octet-stream"
    sha256: str = ""
    bytes: int = 0
    caption: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {
            "id": self.artifact_id,
            "kind": self.kind,
            "path": self.path,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "caption": self.caption,
        }

    @classmethod
    def from_dict(cls, data: object) -> Artifact:
        """Load tolerantly from a dict."""
        payload = _dict(data)
        path = _text(payload.get("path"), 255)
        return cls(
            artifact_id=_text(payload.get("id") or path, 255),
            kind=_text(payload.get("kind") or "file", 32),
            path=path,
            media_type=_text(payload.get("media_type") or "application/octet-stream", 100),
            sha256=_text(payload.get("sha256"), 64),
            bytes=max(_int(payload.get("bytes")), 0),
            caption=_text(payload.get("caption")),
        )


@dataclass(frozen=True)
class Producer:
    """What produced a bundle.

    Attributes:
        kind: One of ``PRODUCER_KINDS``.
        name: The recipe name, script id or model.
        digest: SHA-256 of the producing code (or prompt), or ``""``.
        actor: Who asked for it (``"operator"``, an agent id), or ``""``.
        model: The model, for a draft.
        prompt_digest: The prompt digest, for a draft.
    """

    kind: str = PRODUCER_RECIPE
    name: str = ""
    digest: str = ""
    actor: str = ""
    model: str = ""
    prompt_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {
            "kind": self.kind,
            "name": self.name,
            "digest": self.digest,
            "actor": self.actor,
            "model": self.model,
            "prompt_digest": self.prompt_digest,
        }

    @classmethod
    def from_dict(cls, data: object) -> Producer:
        """Load tolerantly from a dict."""
        payload = _dict(data)
        kind = _text(payload.get("kind"), 32)
        return cls(
            kind=kind if kind in PRODUCER_KINDS else PRODUCER_RECIPE,
            name=_text(payload.get("name"), 200),
            digest=_text(payload.get("digest"), 128),
            actor=_text(payload.get("actor"), 200),
            model=_text(payload.get("model"), 100),
            prompt_digest=_text(payload.get("prompt_digest"), 128),
        )


@dataclass(frozen=True)
class BundleInput:
    """One run a bundle was computed from.

    Attributes:
        run_id: The run.
        data_file: Its data file, as the run record stores it.
        params_digest: The run's **Params digest**.
    """

    run_id: str
    data_file: str = ""
    params_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {
            "run_id": self.run_id,
            "data_file": self.data_file,
            "params_digest": self.params_digest,
        }

    @classmethod
    def from_dict(cls, data: object) -> BundleInput:
        """Load tolerantly from a dict."""
        payload = _dict(data)
        return cls(
            run_id=_text(payload.get("run_id"), 200),
            data_file=_text(payload.get("data_file"), 4096),
            params_digest=_text(payload.get("params_digest"), 128),
        )


@dataclass(frozen=True)
class Bundle:
    """Everything one producer execution left for downstream readers.

    Attributes:
        bundle_id: The bundle's id (its folder name, or ``script-<id>``).
        experiment_id: The experiment the runs belong to.
        run_ids: The runs it covers; one for a per-run analysis.
        producer: What produced it.
        status: ``BUNDLE_OK`` or ``BUNDLE_FAILED``.
        error: The failure, when ``failed``.
        created_utc: When the bundle was sealed.
        inputs: The runs it was computed from.
        artifacts: The sealed files.
        summary: Plain-text paragraphs, in order.
        results: Derived values: ``{name, value, unit, uncertainty, note}``.
        tables: Small tables: ``{caption, columns, rows, truncated}``.
        tags: Tags the producer proposes.
        warnings: Non-fatal notes.
        options: The options it ran with.
        hints: Publishing hints (``include_fact_tables``, ``attach_data_file``);
            advice to a renderer, never an instruction.
        duration_s: Wall time of the producer.
        sealed: ``True`` when read from a ``bundle.json`` the application
            wrote; ``False`` for a legacy report read as a bundle.
        schema_version: The version the manifest was written with.
    """

    bundle_id: str = ""
    experiment_id: str = ""
    run_ids: tuple[str, ...] = ()
    producer: Producer = field(default_factory=Producer)
    status: str = BUNDLE_OK
    error: str = ""
    created_utc: str = ""
    inputs: tuple[BundleInput, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    summary: tuple[str, ...] = ()
    results: tuple[dict[str, Any], ...] = ()
    tables: tuple[dict[str, Any], ...] = ()
    tags: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)
    hints: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    sealed: bool = True
    schema_version: int = BUNDLE_SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        """Whether the producer ran to completion."""
        return self.status == BUNDLE_OK

    def artifact(self, artifact_id: str) -> Artifact | None:
        """Return one artifact by id, or ``None``."""
        return next((a for a in self.artifacts if a.artifact_id == artifact_id), None)

    def result(self, name: str) -> dict[str, Any] | None:
        """Return the first result called ``name``, or ``None``."""
        return next((r for r in self.results if r.get("name") == name), None)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe manifest."""
        return {
            "schema": BUNDLE_SCHEMA,
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "created_utc": self.created_utc,
            "status": self.status,
            "error": self.error,
            "scope": {"experiment_id": self.experiment_id, "run_ids": list(self.run_ids)},
            "producer": self.producer.to_dict(),
            "inputs": [i.to_dict() for i in self.inputs],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "summary": list(self.summary),
            "results": [dict(r) for r in self.results],
            "tables": [dict(t) for t in self.tables],
            "tags": list(self.tags),
            "warnings": list(self.warnings),
            "options": dict(self.options),
            "hints": dict(self.hints),
            "duration_s": self.duration_s,
            "sealed": self.sealed,
        }

    @classmethod
    def from_dict(cls, data: object) -> Bundle:
        """Load tolerantly from a parsed ``bundle.json``; junk becomes defaults."""
        payload = _dict(data)
        scope = _dict(payload.get("scope"))
        duration = payload.get("duration_s")
        status = payload.get("status")
        return cls(
            bundle_id=_text(payload.get("bundle_id"), 200),
            experiment_id=_text(scope.get("experiment_id"), 200),
            run_ids=tuple(_text(r, 200) for r in _list(scope.get("run_ids")) if r),
            producer=Producer.from_dict(payload.get("producer")),
            status=BUNDLE_FAILED if status == BUNDLE_FAILED else BUNDLE_OK,
            error=_text(payload.get("error"), 20000),
            created_utc=_text(payload.get("created_utc"), 64),
            inputs=tuple(BundleInput.from_dict(i) for i in _list(payload.get("inputs"))),
            artifacts=tuple(
                Artifact.from_dict(a) for a in _list(payload.get("artifacts"))[:MAX_ARTIFACTS]
            ),
            summary=tuple(_text(p) for p in _list(payload.get("summary")) if p is not None),
            results=tuple(_dict(r) for r in _list(payload.get("results"))),
            tables=tuple(_dict(t) for t in _list(payload.get("tables"))),
            tags=tuple(_text(t, 100) for t in _list(payload.get("tags")) if t),
            warnings=tuple(_text(w) for w in _list(payload.get("warnings")) if w),
            options=_dict(payload.get("options")),
            hints=_dict(payload.get("hints")),
            duration_s=float(duration) if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 0 else 0.0,
            sealed=payload.get("sealed") is not False,
            schema_version=_int(payload.get("schema_version"), 0),
        )


def _write_json_atomic(path: Path, payload: object) -> None:
    """Write JSON to ``path`` through a temporary file and a rename."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def scan_artifacts(folder: str | Path, captions: dict[str, str] | None = None) -> list[Artifact]:
    """Hash and list every allowed file directly inside ``folder``.

    Args:
        folder: The bundle folder.
        captions: ``{file name: caption}`` claimed by the producer.

    Returns:
        One ``Artifact`` per accepted file, in name order, at most
        ``MAX_ARTIFACTS``; a refused file (bad name, too large, a framework
        file) is simply not listed.
    """
    directory = Path(folder)
    captions = captions or {}
    artifacts: list[Artifact] = []
    try:
        names = sorted(p.name for p in directory.iterdir() if p.is_file())
    except OSError:
        return []
    for name in names:
        if name in FRAMEWORK_FILES:
            continue
        kind = artifact_kind(name)
        if kind is None:
            continue
        path = output_file(directory, name, suffixes=tuple(ARTIFACT_SUFFIXES))
        if path is None:
            continue
        try:
            size = path.stat().st_size
            digest = _sha256(path)
        except OSError:
            continue
        artifacts.append(
            Artifact(
                artifact_id=name,
                kind=kind[0],
                path=name,
                media_type=kind[1],
                sha256=digest,
                bytes=size,
                caption=_text(captions.get(name, "")),
            )
        )
        if len(artifacts) >= MAX_ARTIFACTS:
            break
    return artifacts


def _claimed_captions(claims: dict[str, Any]) -> dict[str, str]:
    """Return ``{file: caption}`` from a report's ``figures`` claims."""
    captions: dict[str, str] = {}
    for figure in _list(claims.get("figures")):
        entry = _dict(figure)
        name = entry.get("file")
        if isinstance(name, str) and name:
            captions[name] = _text(entry.get("caption"))
    return captions


def bundle_from_claims(
    folder: str | Path,
    claims: dict[str, Any],
    *,
    bundle_id: str,
    experiment_id: str,
    run_ids: list[str] | tuple[str, ...],
    producer: Producer,
    inputs: list[BundleInput] | tuple[BundleInput, ...] = (),
    created_utc: str = "",
    sealed: bool = True,
) -> Bundle:
    """Build (without writing) the bundle a folder and its claims describe.

    Args:
        folder: The bundle folder; its files are hashed.
        claims: The report dict the caller ALREADY parsed and capped
            (``AnalysisReport.to_dict()``); never the raw worker file.
        bundle_id: The bundle's id.
        experiment_id: The owning experiment.
        run_ids: The runs covered.
        producer: What produced it.
        inputs: The runs' identities.
        created_utc: Seal time; ``""`` for now.
        sealed: Whether this is a real seal (``False`` for a legacy read).

    Returns:
        The bundle.
    """
    hints = {
        "include_fact_tables": bool(claims.get("include_fact_tables", False)),
        "attach_data_file": bool(claims.get("attach_data_file", False)),
    }
    duration = claims.get("duration_s")
    return Bundle(
        bundle_id=bundle_id,
        experiment_id=experiment_id,
        run_ids=tuple(run_ids),
        producer=producer,
        status=BUNDLE_FAILED if claims.get("status") == BUNDLE_FAILED else BUNDLE_OK,
        error=_text(claims.get("error"), 20000),
        created_utc=created_utc or _utc_now(),
        inputs=tuple(inputs),
        artifacts=tuple(scan_artifacts(folder, _claimed_captions(claims))),
        summary=tuple(_text(p) for p in _list(claims.get("summary")) if p is not None),
        results=tuple(_dict(r) for r in _list(claims.get("results"))),
        tables=tuple(_dict(t) for t in _list(claims.get("tables"))),
        tags=tuple(_text(t, 100) for t in _list(claims.get("tags")) if t),
        warnings=tuple(_text(w) for w in _list(claims.get("warnings")) if w),
        options=_dict(claims.get("options")),
        hints=hints,
        duration_s=float(duration) if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 0 else 0.0,
        sealed=sealed,
        schema_version=BUNDLE_SCHEMA_VERSION if sealed else 0,
    )


def seal_bundle(folder: str | Path, claims: dict[str, Any], **identity: Any) -> Bundle:
    """Seal one bundle folder: hash its files and write ``bundle.json``.

    Called by the application — never by the worker — once the producer has
    finished. Takes the same keyword arguments as ``bundle_from_claims()``.

    Args:
        folder: The bundle folder.
        claims: The parsed, capped report dict.
        **identity: ``bundle_id``, ``experiment_id``, ``run_ids``,
            ``producer`` and optionally ``inputs`` / ``created_utc``.

    Returns:
        The sealed bundle.

    Raises:
        OSError: ``bundle.json`` could not be written.
    """
    bundle = bundle_from_claims(folder, claims, **identity)
    _write_json_atomic(Path(folder) / BUNDLE_FILENAME, bundle.to_dict())
    return bundle


def read_bundle(folder: str | Path, *, bundle_id: str = "") -> Bundle | None:
    """Read one bundle folder, tolerantly.

    A folder with a ``bundle.json`` is read as sealed. A folder with only a
    ``report.json`` (written before bundles existed, or a producer the
    application has not sealed yet) is read as an UNSEALED bundle built from
    those claims — never written back.

    Args:
        folder: The bundle folder.
        bundle_id: The id to report for an unsealed folder; ``""`` uses the
            folder name.

    Returns:
        The bundle, or ``None`` when the folder holds neither file, or the
        file is oversized or unreadable.
    """
    directory = Path(folder)
    manifest = directory / BUNDLE_FILENAME
    payload = _read_json(manifest)
    if payload is not None:
        return Bundle.from_dict(payload)
    claims = _read_json(directory / CLAIMS_FILENAME)
    if claims is None:
        return None
    claims_dict = _dict(claims)
    run_id = _text(claims_dict.get("run_id"), 200)
    return bundle_from_claims(
        directory,
        claims_dict,
        bundle_id=bundle_id or directory.name,
        experiment_id="",
        run_ids=(run_id,) if run_id else (),
        producer=Producer(
            kind=PRODUCER_LEGACY,
            name=_text(claims_dict.get("recipe"), 200),
            digest=_text(claims_dict.get("recipe_digest"), 128),
        ),
        created_utc=_text(claims_dict.get("started_utc"), 64),
        sealed=False,
    )


def _read_json(path: Path) -> object | None:
    """Read one JSON file within ``MAX_BUNDLE_BYTES``, or ``None``."""
    try:
        if not path.is_file() or path.stat().st_size > MAX_BUNDLE_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def verify_artifact(folder: str | Path, artifact: Artifact) -> Path | None:
    """Return an artifact's file only if it is still exactly what was sealed.

    The check every uploader and viewer runs before trusting a file: the
    name passes ``output_file()``, and — for a sealed artifact — the size and
    SHA-256 still match the manifest, so a file swapped after sealing is
    refused rather than published.

    Args:
        folder: The bundle folder.
        artifact: The artifact, from the bundle's manifest.

    Returns:
        The file's path, or ``None`` when it is missing, renamed, oversized
        or no longer matches.
    """
    path = output_file(folder, artifact.path, suffixes=tuple(ARTIFACT_SUFFIXES))
    if path is None:
        return None
    if not artifact.sha256:
        return path
    try:
        if path.stat().st_size != artifact.bytes or _sha256(path) != artifact.sha256:
            return None
    except OSError:
        return None
    return path
