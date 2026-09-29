"""Publishing — the bridge from analysis bundles to an experiment's notebook page.

Everything here is plain Python, no Qt, and synchronous: the notebook service
(``service.py``) runs it on its worker thread. It reads only the experiment
record, the sealed **analysis bundles** and the experiment's pinned blocks —
never a live analysis, never the station.

**One page per experiment, appended to.** A publish covers some finished runs
(by default every one not yet published). It gets a ``publish_id``; the
experiment's renderer lays it out as a ``Section`` (in its helper process);
``i2as.blocks.markup`` turns that into safe HTML headed ``I2AS · <time> ·
<title> · <publish_id>``; and one outbox job uploads the figures, appends the
section at the end of the page, and overwrites the profile's page fields with
the latest values. Earlier sections, and whatever people wrote on the page,
are never touched.

**Pinned blocks.** When an experiment is linked, its profile and renderer are
COPIED into ``<experiment>/eln/`` and the connector's digest is recorded, so
editing a block later never silently changes how an ongoing experiment is
published: the copies are what run, and a connector whose file changed is not
used for the experiment until a person confirms it.
"""

from __future__ import annotations

import logging
import secrets
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from i2as.analysis.bundle import Bundle, verify_artifact
from i2as.blocks.connector import (
    ElnAuthError,
    ElnEntryRef,
    ElnError,
    ElnNotFound,
    ElnRef,
    ElnTransientError,
    ElnValidationError,
)
from i2as.blocks.discovery import (
    KIND_CONNECTOR,
    KIND_PROFILE,
    KIND_RENDERER,
    BlockInfo,
    discover_blocks,
)
from i2as.blocks.markup import section_to_html
from i2as.blocks.profile import Profile, load_profile, resolve_fields
from i2as.blocks.protocol import BlockError
from i2as.blocks.renderer import RenderContext, RunContext, Section
from i2as.session.eln.block_runner import BlockRunner, BlockTimeout
from i2as.session.eln.ledger import Ledger
from i2as.session.eln.outbox import (
    ATTENTION_AUTH,
    ATTENTION_BLOCK,
    ATTENTION_CHANGED,
    ATTENTION_REFUSED,
    JOB_CREATE_ENTRY,
    JOB_LINK_ITEMS,
    JOB_PUBLISH,
    STATE_DONE,
    Outbox,
    OutboxJob,
)
from i2as.session.models import (
    RUN_STATUS_RUNNING,
    ElnBinding,
    ElnLink,
    ExperimentRecord,
    LinkedItem,
    PinnedBlock,
)

logger = logging.getLogger(__name__)

#: File names of the pinned copies inside ``<experiment>/eln/``.
PINNED_PROFILE = "profile.yaml"
PINNED_RENDERER = "renderer.py"
OUTBOX_FILENAME = "outbox.jsonl"
LEDGER_FILENAME = "ledger.json"


def utc_now() -> str:
    """Return the current UTC time, ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def new_publish_id(now: datetime | None = None) -> str:
    """Return a fresh publish id: ``P-<UTC stamp>-<random>``."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"P-{stamp}-{secrets.token_hex(3)}"


class PublishError(RuntimeError):
    """A publish (or a link) cannot be prepared; the message says why, for a person."""


# ----------------------------------------------------------------------
# Blocks: discovery, pinning
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BlockCatalog:
    """The blocks this installation can use: shipped, plus the user's folder.

    Attributes:
        user_root: The user's ``blocks`` folder, or ``None``.
    """

    user_root: Path | None = None

    def get(self, kind: str, block_id: str) -> BlockInfo | None:
        """Return one usable block, or ``None``."""
        info = discover_blocks(kind, self.user_root).get(block_id)
        return info if info is not None and info.usable else None

    def all(self, kind: str) -> dict[str, BlockInfo]:
        """Return every block of one kind (unusable ones included)."""
        return discover_blocks(kind, self.user_root)


def pin_blocks(
    catalog: BlockCatalog,
    eln_dir: Path,
    *,
    connector_id: str,
    profile_id: str,
) -> tuple[Profile, PinnedBlock, PinnedBlock, PinnedBlock]:
    """Copy an experiment's profile and renderer into its folder; pin the connector.

    Args:
        catalog: The block catalog.
        eln_dir: ``<experiment>/eln`` (created).
        connector_id: The account's connector.
        profile_id: The profile to use.

    Returns:
        ``(profile, connector pin, renderer pin, profile pin)``.

    Raises:
        PublishError: A block is missing, unusable, or written for another
            connector.
    """
    connector = catalog.get(KIND_CONNECTOR, connector_id)
    if connector is None:
        raise PublishError(f"the connector '{connector_id}' is not installed or not usable")
    profile_info = catalog.get(KIND_PROFILE, profile_id)
    if profile_info is None:
        raise PublishError(f"the profile '{profile_id}' is not installed")
    profile = load_profile(profile_info.path)
    if profile.connector and profile.connector != connector_id:
        raise PublishError(f"the profile '{profile_id}' is written for '{profile.connector}', not '{connector_id}'")
    renderer = catalog.get(KIND_RENDERER, profile.renderer)
    if renderer is None:
        raise PublishError(f"the renderer '{profile.renderer}' named by profile '{profile_id}' is not usable")
    try:
        eln_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(profile_info.path, eln_dir / PINNED_PROFILE)
        shutil.copyfile(renderer.path, eln_dir / PINNED_RENDERER)
    except OSError as exc:
        raise PublishError(f"could not keep a copy of the profile and renderer: {exc}") from exc
    return (
        profile,
        PinnedBlock(connector.block_id, connector.digest),
        PinnedBlock(renderer.block_id, renderer.digest),
        PinnedBlock(profile_info.block_id, profile_info.digest),
    )


def pinned_profile(eln_dir: Path) -> Profile:
    """Return the experiment's pinned profile (defaults when the copy is gone)."""
    path = eln_dir / PINNED_PROFILE
    return load_profile(path) if path.is_file() else Profile()


# ----------------------------------------------------------------------
# Preparing a publish (GUI thread: reads only)
# ----------------------------------------------------------------------


def unpublished_runs(experiment: ExperimentRecord) -> list[Any]:
    """Return the experiment's finished science runs not yet on its page, oldest first."""
    return [
        run
        for run in experiment.runs
        if run.status != RUN_STATUS_RUNNING and run.kind == "run" and not run.published
    ]


@dataclass(frozen=True)
class Attachment:
    """One figure a publish uploads.

    Attributes:
        bundle_id: The bundle it comes from.
        artifact_id: The artifact.
        path: Its file in the bundle folder.
        sha256: Its sealed digest.
        bytes: Its sealed size.
        upload_name: The name it is uploaded under (``<run>_<file>``).
        caption: Its caption.
    """

    bundle_id: str
    artifact_id: str
    path: str
    sha256: str
    bytes: int
    upload_name: str
    caption: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "artifact_id": self.artifact_id,
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "upload_name": self.upload_name,
            "caption": self.caption,
        }


@dataclass(frozen=True)
class PublishPlan:
    """Everything one publish needs, gathered from records and bundles.

    Attributes:
        experiment_id: The experiment.
        publish_id: The publish.
        published_utc: When it was asked for.
        context: What the renderer receives.
        attachments: The figures to upload (sealed, verified, under the cap).
        run_bundles: ``{run_id: bundle_id}`` it covers.
        fields: The page fields the profile's write map sets.
        skipped: Figures left out, with why.
    """

    experiment_id: str
    publish_id: str
    published_utc: str
    context: RenderContext
    attachments: tuple[Attachment, ...]
    run_bundles: dict[str, str]
    fields: dict[str, str] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()


def plan_publish(
    experiment: ExperimentRecord,
    runs: list[Any],
    *,
    read_bundle: Callable[[str, str, str], Bundle | None],
    bundle_dir: Callable[[str, str, str], Path],
    profile: Profile,
    user_name: str = "",
    config_name: str = "",
    max_attachment_bytes: int = 50 * 1024 * 1024,
    publish_id: str = "",
) -> PublishPlan:
    """Gather one publish from the records and the runs' selected bundles.

    Args:
        experiment: The experiment.
        runs: The runs to publish.
        read_bundle: ``(experiment_id, run_id, bundle_id) -> Bundle | None``.
        bundle_dir: ``(experiment_id, run_id, bundle_id) -> folder``.
        profile: The experiment's pinned profile.
        user_name: The experimenter's name.
        config_name: The station's config.
        max_attachment_bytes: The upload cap.
        publish_id: The id to use; ``""`` for a new one.

    Returns:
        The plan.
    """
    pid = publish_id or new_publish_id()
    stamp = utc_now()
    run_contexts: list[RunContext] = []
    attachments: list[Attachment] = []
    skipped: list[str] = []
    run_bundles: dict[str, str] = {}
    results: dict[str, Any] = {}
    for run in runs:
        bundle = read_bundle(experiment.experiment_id, run.run_id, run.selected_bundle) if run.selected_bundle else None
        if bundle is not None and not bundle.sealed:
            bundle = None
        run_bundles[run.run_id] = bundle.bundle_id if bundle is not None else ""
        if bundle is not None and bundle.ok:
            for value in bundle.results:
                if value.get("name"):
                    results[str(value["name"])] = value.get("value")
            folder = bundle_dir(experiment.experiment_id, run.run_id, bundle.bundle_id)
            for artifact in bundle.artifacts:
                if artifact.kind != "figure":
                    continue
                if artifact.bytes > max_attachment_bytes:
                    skipped.append(f"{run.run_id}/{artifact.path}: larger than the upload limit")
                    continue
                path = verify_artifact(folder, artifact)
                if path is None:
                    skipped.append(f"{run.run_id}/{artifact.path}: changed or missing since it was sealed")
                    continue
                attachments.append(
                    Attachment(
                        bundle_id=bundle.bundle_id,
                        artifact_id=artifact.artifact_id,
                        path=str(path),
                        sha256=artifact.sha256,
                        bytes=artifact.bytes,
                        upload_name=f"{run.run_id}_{artifact.path}",
                        caption=artifact.caption,
                    )
                )
        run_contexts.append(
            RunContext(
                run_id=run.run_id,
                procedure=run.procedure,
                params=dict(run.params),
                status=run.status,
                reason=run.reason,
                started_utc=run.started_utc,
                finished_utc=run.finished_utc,
                data_file=run.data_file,
                legacy_entry_url=run.eln_link.url if run.eln_link is not None else "",
                bundle=bundle,
            )
        )
    experiment_facts = {
        "experiment_id": experiment.experiment_id,
        "title": experiment.title,
        "user_name": user_name,
        "sample_info": dict(experiment.sample_info),
        "findings": experiment.findings,
        "config_name": config_name,
    }
    fields = resolve_fields(
        profile,
        experiment=experiment_facts,
        sample_info=experiment.sample_info,
        results=results,
    )
    context = RenderContext(
        publish_id=pid,
        published_utc=stamp,
        experiment=experiment_facts,
        runs=tuple(run_contexts),
        options=dict(profile.render),
    )
    return PublishPlan(
        experiment_id=experiment.experiment_id,
        publish_id=pid,
        published_utc=stamp,
        context=context,
        attachments=tuple(attachments),
        run_bundles=run_bundles,
        fields=fields,
        skipped=tuple(skipped),
    )


def render_plan(plan: PublishPlan, renderer: BlockRunner, timeout_s: float) -> tuple[Section, str]:
    """Run the experiment's renderer over a plan; return its section and the safe HTML.

    Raises:
        PublishError: The renderer failed, overran or returned nothing.
    """
    try:
        raw = renderer.call("render", {"context": plan.context.to_dict()}, timeout_s)
    except (BlockError, ElnError) as exc:
        raise PublishError(f"the renderer failed: {exc}") from exc
    section = Section.from_dict(raw)
    if not section.blocks:
        raise PublishError("the renderer returned an empty section")
    names = {(a.bundle_id, a.artifact_id): a.upload_name for a in plan.attachments}
    html = section_to_html(section, publish_id=plan.publish_id, published_utc=plan.published_utc, figure_names=names)
    return section, html


def publish_job(plan: PublishPlan, section: Section, html: str, *, user_id: str, account_id: str, connector_digest: str) -> OutboxJob:
    """Return the outbox job performing one rendered publish."""
    fields = {**plan.fields, **section.fields}
    return OutboxJob(
        job_id=f"{JOB_PUBLISH}:{plan.publish_id}",
        kind=JOB_PUBLISH,
        experiment_id=plan.experiment_id,
        user_id=user_id,
        account_id=account_id,
        payload={
            "publish_id": plan.publish_id,
            "published_utc": plan.published_utc,
            "run_bundles": dict(plan.run_bundles),
            "html": html,
            "fields": fields,
            "tags": list(section.tags),
            "attachments": [a.to_dict() for a in plan.attachments],
            "skipped": list(plan.skipped),
            "connector_digest": connector_digest,
        },
    )


def create_entry_job(experiment: ExperimentRecord, binding: ElnBinding, *, user_id: str, title: str, fields: Mapping[str, str]) -> OutboxJob:
    """Return the outbox job creating an experiment's page."""
    return OutboxJob(
        job_id=f"{JOB_CREATE_ENTRY}:{experiment.experiment_id}",
        kind=JOB_CREATE_ENTRY,
        experiment_id=experiment.experiment_id,
        user_id=user_id,
        account_id=binding.account_id,
        payload={
            "title": title,
            "template_id": binding.template_id,
            "fields": dict(fields),
            "connector_digest": binding.connector.digest,
        },
    )


def link_items_job(experiment: ExperimentRecord, binding: ElnBinding, items: list[LinkedItem], *, user_id: str) -> OutboxJob:
    """Return the outbox job linking items to an experiment's page."""
    ids = ",".join(sorted(item.item_id for item in items))
    return OutboxJob(
        job_id=f"{JOB_LINK_ITEMS}:{experiment.experiment_id}:{ids}",
        kind=JOB_LINK_ITEMS,
        experiment_id=experiment.experiment_id,
        user_id=user_id,
        account_id=binding.account_id,
        payload={
            "items": [{"kind": "item", "record_id": item.item_id} for item in items],
            "connector_digest": binding.connector.digest,
        },
    )


# ----------------------------------------------------------------------
# Performing jobs (worker thread)
# ----------------------------------------------------------------------


class AccountUnavailable(RuntimeError):
    """The job's account is not configured (or has no connector) for its user."""


@dataclass(frozen=True)
class JobOutcome:
    """What one attempt at a job produced.

    Attributes:
        job: The job's latest revision.
        entry: A page the backend confirmed (a create), or ``None``.
        published: ``{"publish_id", "published_utc", "run_bundles"}`` when a
            publish completed, else ``None``.
        waiting: The job could not run yet (its page does not exist yet).
    """

    job: OutboxJob
    entry: ElnEntryRef | None = None
    published: dict[str, Any] | None = None
    waiting: bool = False


class JobExecutor:
    """Performs outbox jobs with a connector. Plain Python; the worker thread calls it.

    Args:
        connector_for: ``(user_id, account_id) -> (BlockRunner, digest)``;
            raises ``AccountUnavailable``.
        entry_for: ``experiment_id -> ElnEntryRef | None`` — the page the
            experiment record names right now.
        ledger_for: ``experiment_id -> Ledger``.
        timeout_s: Per-call limit for the connector.
    """

    def __init__(
        self,
        connector_for: Callable[[str, str], tuple[BlockRunner, str]],
        entry_for: Callable[[str], ElnEntryRef | None],
        ledger_for: Callable[[str], Ledger],
        timeout_s: float = 60.0,
    ) -> None:
        self._connector_for = connector_for
        self._entry_for = entry_for
        self._ledger_for = ledger_for
        self._timeout = timeout_s

    def run(self, outbox: Outbox, job: OutboxJob) -> JobOutcome:
        """Attempt one job; journal what happened. Never raises."""
        try:
            runner, digest = self._connector_for(job.user_id, job.account_id)
        except AccountUnavailable as exc:
            return JobOutcome(outbox.attention(job, ATTENTION_REFUSED, str(exc)))
        pinned = str(job.payload.get("connector_digest") or "")
        if pinned and digest and pinned != digest:
            return JobOutcome(
                outbox.attention(
                    job,
                    ATTENTION_CHANGED,
                    "the connector's code changed since this experiment was linked; confirm the new version",
                )
            )
        try:
            if job.kind == JOB_CREATE_ENTRY:
                return self._create(outbox, job, runner)
            entry = self._entry_for(job.experiment_id)
            if entry is None:
                return JobOutcome(job, waiting=True)
            if job.kind == JOB_LINK_ITEMS:
                return self._link(outbox, job, runner, entry)
            return self._publish(outbox, job, runner, entry)
        except ElnAuthError as exc:
            return JobOutcome(outbox.attention(job, ATTENTION_AUTH, str(exc)))
        except (ElnValidationError, ElnNotFound) as exc:
            return JobOutcome(outbox.attention(job, ATTENTION_REFUSED, str(exc)))
        except (ElnTransientError, BlockTimeout) as exc:
            return JobOutcome(outbox.failed(job, str(exc)))
        except ElnError as exc:
            return JobOutcome(outbox.failed(job, str(exc)))
        except BlockError as exc:
            return JobOutcome(outbox.attention(job, ATTENTION_BLOCK, f"the connector failed: {exc}"))
        except OSError as exc:
            return JobOutcome(outbox.failed(job, f"a local file could not be read: {exc}"))

    def _call(self, runner: BlockRunner, method: str, **args: Any) -> Any:
        return runner.call(method, args, self._timeout)

    def _create(self, outbox: Outbox, job: OutboxJob, runner: BlockRunner) -> JobOutcome:
        done = job.progress.get("entry")
        if isinstance(done, dict) and done.get("entry_id"):
            entry = ElnEntryRef.from_dict(done)
        else:
            entry = ElnEntryRef.from_dict(
                self._call(
                    runner,
                    "create_entry",
                    title=str(job.payload.get("title") or ""),
                    template_id=str(job.payload.get("template_id") or ""),
                    fields=dict(job.payload.get("fields") or {}),
                )
            )
            job = replace(job, progress={**job.progress, "entry": entry.to_dict()})
            outbox.record(job)
        final = replace(job, state=STATE_DONE, last_error="")
        outbox.record(final)
        return JobOutcome(final, entry=entry)

    def _link(self, outbox: Outbox, job: OutboxJob, runner: BlockRunner, entry: ElnEntryRef) -> JobOutcome:
        ledger = self._ledger_for(job.experiment_id)
        for item in job.payload.get("items") or []:
            ref = ElnRef.from_dict(item)
            if not ref.record_id or ledger.linked(entry.entry_id, ref.record_id):
                continue
            self._call(runner, "link_item", entry=entry.to_dict(), item=ref.to_dict())
            ledger.record_link(entry.entry_id, ref.record_id)
        final = replace(job, state=STATE_DONE, last_error="")
        outbox.record(final)
        return JobOutcome(final)

    def _publish(self, outbox: Outbox, job: OutboxJob, runner: BlockRunner, entry: ElnEntryRef) -> JobOutcome:
        ledger = self._ledger_for(job.experiment_id)
        payload = job.payload
        publish_id = str(payload.get("publish_id") or "")
        for item in payload.get("attachments") or []:
            sha = str(item.get("sha256") or "")
            if sha and ledger.upload(entry.entry_id, sha) is not None:
                continue
            source = Path(str(item.get("path") or ""))
            with tempfile.TemporaryDirectory(prefix="i2as-upload-") as folder:
                staged = Path(folder) / str(item.get("upload_name") or source.name)
                shutil.copyfile(source, staged)
                upload_id = self._call(runner, "upload", entry=entry.to_dict(), path=str(staged), caption=str(item.get("caption") or ""))
            ledger.record_upload(entry.entry_id, sha, staged.name, str(upload_id or ""))
        if not job.progress.get("appended"):
            if not self._call(runner, "has_section", entry=entry.to_dict(), publish_id=publish_id):
                self._call(runner, "append_section", entry=entry.to_dict(), publish_id=publish_id, html=str(payload.get("html") or ""))
            job = replace(job, progress={**job.progress, "appended": True})
            outbox.record(job)
        fields = dict(payload.get("fields") or {})
        if fields and not job.progress.get("fields_set"):
            self._call(runner, "set_fields", entry=entry.to_dict(), fields=fields)
            job = replace(job, progress={**job.progress, "fields_set": True})
            outbox.record(job)
        final = replace(job, state=STATE_DONE, last_error="")
        outbox.record(final)
        return JobOutcome(
            final,
            published={
                "publish_id": publish_id,
                "published_utc": str(payload.get("published_utc") or utc_now()),
                "run_bundles": dict(payload.get("run_bundles") or {}),
            },
        )


def entry_of(binding: ElnBinding | None) -> ElnEntryRef | None:
    """Return a binding's page as an ``ElnEntryRef``, or ``None``."""
    if binding is None or binding.entry is None or not binding.entry.entry_id:
        return None
    return ElnEntryRef.from_dict(binding.entry.to_dict())


def link_of(entry: ElnEntryRef) -> ElnLink:
    """Return the record-side link for a confirmed page."""
    return ElnLink.from_dict(entry.to_dict())
