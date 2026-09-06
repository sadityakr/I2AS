"""``SimElnAdapter`` — the in-memory twin of the ELN adapter standard.

The ``sim_`` rule the driver contract applies to instruments, applied to
electronic lab notebooks: a twin with an identical public API that models the
backend's behaviour — *including its failure modes* — so a wrong publish
sequence fails in a test instead of against a live notebook. It is the
workhorse of every ELN test in this repository; nothing else needs a network.

What it models, driven entirely by its settings mapping (so a test configures
it exactly the way a real adapter is configured):

- ``sim_offline`` — every call raises ``ElnError``, the lab-network-is-down
  case the outbox exists for. Flip the public ``offline`` attribute mid-test
  to bring the notebook back.
- ``sim_fail_calls`` — fail the first *n* calls, then succeed: the transient
  failure that must be retried, not dropped.
- ``sim_reject_attachments`` — accept the entry but refuse the upload, the
  partial failure that must never create a second entry on retry.

Every accepted call is recorded (``entries``, ``uploads``, ``links``,
``calls``) so a test asserts what was published rather than how.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from i2as.session.eln.adapter import (
    ElnAdapter,
    ElnCapabilities,
    ElnEntryRef,
    ElnError,
    ElnTemplate,
)

logger = logging.getLogger(__name__)

_SIM_BASE_URL = "https://sim.eln.invalid"


class SimElnAdapter(ElnAdapter):
    """An electronic lab notebook that lives entirely in this process.

    Attributes:
        entries: ``{entry_id: {"title", "body_html", "tags", "metadata",
            "template_id"}}`` — every entry created, latest body after any
            update.
        uploads: One ``{"entry_id", "path", "comment"}`` dict per accepted
            file attachment, in call order.
        links: One ``{"entry_id", "url", "comment"}`` dict per accepted link,
            in call order.
        calls: The name of every contract method invoked, in order, including
            the ones that raised — what a test uses to assert that a retry
            did *not* create a second entry.
        offline: Public switch a test flips to take the notebook down and
            bring it back up mid-drain.
    """

    backend: ClassVar[str] = "sim_eln"
    capabilities: ClassVar[ElnCapabilities] = ElnCapabilities(
        templates=True,
        attachments=True,
        links=True,
        update=True,
        max_attachment_bytes=0,
    )

    def __init__(self, settings: Mapping[str, Any] | None) -> None:
        """Build an in-memory notebook from a plain settings mapping.

        Args:
            settings: The same mapping shape a real adapter takes
                (``ElnSettings.to_dict()``), plus the optional ``sim_*`` keys
                documented in this module's docstring. ``None`` behaves like
                an empty mapping — a healthy, empty notebook.
        """
        config = dict(settings or {})
        self._base_url = str(config.get("base_url") or _SIM_BASE_URL).rstrip("/")
        self._template_id = str(config.get("template_id") or "")
        self.offline = bool(config.get("sim_offline", False))
        self._fail_calls = int(config.get("sim_fail_calls", 0) or 0)
        self._reject_attachments = bool(config.get("sim_reject_attachments", False))
        self._next_id = 1
        self.entries: dict[str, dict[str, Any]] = {}
        self.uploads: list[dict[str, Any]] = []
        self.links: list[dict[str, Any]] = []
        self.calls: list[str] = []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enter(self, call: str) -> None:
        """Record one contract call and apply the simulated failure modes.

        Args:
            call: The contract method's name.

        Raises:
            ElnError: When the notebook is simulated offline, or while the
                ``sim_fail_calls`` budget has not been used up.
        """
        self.calls.append(call)
        if self.offline:
            raise ElnError(f"sim ELN is offline (call {call!r})")
        if self._fail_calls > 0:
            self._fail_calls -= 1
            raise ElnError(f"sim ELN transient failure (call {call!r})")

    # ------------------------------------------------------------------
    # The contract
    # ------------------------------------------------------------------

    def verify(self) -> str:
        """Return the simulated account identity.

        Returns:
            A short identity string.

        Raises:
            ElnError: When simulated offline or failing.
        """
        self._enter("verify")
        return "sim ELN (in-memory)"

    def list_templates(self) -> list[ElnTemplate]:
        """Return the two fixed templates the sim notebook offers.

        Returns:
            The available templates.

        Raises:
            ElnError: When simulated offline or failing.
        """
        self._enter("list_templates")
        return [
            ElnTemplate(template_id="1", name="Cryostat run"),
            ElnTemplate(template_id="2", name="Blank"),
        ]

    def create_entry(
        self,
        title: str,
        template_id: str | None,
        body_html: str,
        tags: list[str],
        metadata: dict[str, Any],
    ) -> ElnEntryRef:
        """Create one in-memory entry.

        Args:
            title: The entry title.
            template_id: Template to create from, or ``None`` for the
                notebook's configured default.
            body_html: The rendered entry body.
            tags: Tags to set.
            metadata: JSON-safe extra fields.

        Returns:
            The new entry's reference.

        Raises:
            ElnError: When simulated offline or failing.
        """
        self._enter("create_entry")
        entry_id = str(self._next_id)
        self._next_id += 1
        used_template = self._template_id if template_id is None else template_id
        self.entries[entry_id] = {
            "title": title,
            "body_html": body_html,
            "tags": list(tags),
            "metadata": dict(metadata),
            "template_id": used_template,
        }
        return ElnEntryRef(
            backend=self.backend,
            entry_id=entry_id,
            url=f"{self._base_url}/experiments.php?mode=view&id={entry_id}",
            template_id=used_template,
        )

    def update_entry(
        self, ref: ElnEntryRef, body_html: str, metadata: dict[str, Any]
    ) -> None:
        """Rewrite an in-memory entry's body and metadata.

        Args:
            ref: The entry to update.
            body_html: The new body.
            metadata: JSON-safe extra fields.

        Raises:
            ElnError: When simulated offline or failing, or the entry is
                unknown.
        """
        self._enter("update_entry")
        entry = self.entries.get(ref.entry_id)
        if entry is None:
            raise ElnError(f"sim ELN has no entry {ref.entry_id!r}")
        entry["body_html"] = body_html
        entry["metadata"] = dict(metadata)

    def attach_file(self, ref: ElnEntryRef, path: Path, comment: str = "") -> None:
        """Record a file upload against an in-memory entry.

        Args:
            ref: The entry to attach to.
            path: The local file; it must exist, exactly as a real upload
                would require.
            comment: Optional caption.

        Raises:
            ElnError: When simulated offline or failing, when attachments are
                simulated as refused, when the entry is unknown, or when the
                file does not exist.
        """
        self._enter("attach_file")
        if self._reject_attachments:
            raise ElnError("sim ELN refuses attachments")
        if ref.entry_id not in self.entries:
            raise ElnError(f"sim ELN has no entry {ref.entry_id!r}")
        if not Path(path).is_file():
            raise ElnError(f"sim ELN cannot read {path}")
        self.uploads.append(
            {"entry_id": ref.entry_id, "path": str(path), "comment": comment}
        )

    def attach_link(self, ref: ElnEntryRef, url: str, comment: str = "") -> None:
        """Record a link against an in-memory entry.

        Args:
            ref: The entry to attach to.
            url: The URL or absolute path to record.
            comment: Optional caption.

        Raises:
            ElnError: When simulated offline or failing, or the entry is
                unknown.
        """
        self._enter("attach_link")
        if ref.entry_id not in self.entries:
            raise ElnError(f"sim ELN has no entry {ref.entry_id!r}")
        self.links.append(
            {"entry_id": ref.entry_id, "url": url, "comment": comment}
        )
