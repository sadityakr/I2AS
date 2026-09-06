"""The ELN adapter contract — the written standard every ELN backend implements.

**The ELN adapter standard.** An adapter is a *stateless wrapper over one
electronic-lab-notebook backend*. Every method is synchronous and raises
``ElnError`` on any failure; queuing, retry, and backoff live in the
publisher's outbox (``i2as/session/eln/outbox.py``), never in an adapter.
The rules a new backend must satisfy — all machine-checked by the ELN-adapter
conformance tests in ``tests/test_conformance.py``:

1. One module per backend under ``i2as/session/eln/``, holding one
   concrete ``ElnAdapter`` subclass.
2. ``__init__(self, settings, ...)`` takes a **single settings mapping** as its
   first argument (the analogue of the driver contract's one-resource-string
   rule) and must construct from a plain ``dict``. Everything the backend
   needs — URL, credentials, TLS policy, timeouts — comes from that mapping,
   never from module state or the environment read at call time.
3. ``backend`` is a non-empty lowercase identifier naming the backend, and
   ``capabilities`` is an ``ElnCapabilities`` declaring what the backend can
   actually do. Callers branch on the flags, never on ``backend``.
4. The public API is **exactly** the contract's methods — no extra public
   method, no missing one — so any adapter is substitutable for any other.
   ``sim_eln.SimElnAdapter`` is the in-memory twin of the contract, the
   workhorse of every test that needs an adapter, exactly as ``sim_`` drivers
   are for instruments. Backend-specific behaviour (an HTTP dialect) is faked
   one level lower, at the injectable transport, because the adapter surface
   itself is identical for every backend.
5. No network I/O anywhere except inside these methods, and never on the
   Orchestrator tick — the publisher's drain is the only caller, from a GUI
   timer.

``ElnEntryRef`` and ``i2as.session.models.ElnLink`` share one field
vocabulary (``backend``/``entry_id``/``url``/``template_id``) precisely so the
persisted record needs no translation layer:
``ElnLink.from_dict(ref.to_dict())``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class ElnError(RuntimeError):
    """Any ELN backend failure: unreachable, refused, malformed, or timed out.

    The single exception type the contract allows out of an adapter, so the
    outbox has exactly one thing to catch when deciding to retry a job.
    """


def _as_str(value: object, default: str = "") -> str:
    """Coerce a JSON value to ``str``, falling back to ``default`` on ``None``."""
    return default if value is None else str(value)


@dataclass(frozen=True)
class ElnCapabilities:
    """What one backend can actually do, declared by its adapter class.

    Attributes:
        templates: The backend can create an entry from a named template and
            can list the templates available.
        attachments: The backend accepts file uploads on an entry.
        links: The backend accepts a URL/path reference on an entry (used when
            a data file is missing, or larger than the caller's cap).
        update: An existing entry's body/metadata can be rewritten.
        max_attachment_bytes: Backend-imposed upload cap in bytes; ``0`` means
            the adapter imposes none (the caller's own cap still applies).
    """

    templates: bool = False
    attachments: bool = False
    links: bool = False
    update: bool = False
    max_attachment_bytes: int = 0


@dataclass(frozen=True)
class ElnTemplate:
    """One entry template offered by the backend.

    Attributes:
        template_id: The backend's id for the template.
        name: Human-readable template name.
    """

    template_id: str = ""
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {"template_id": self.template_id, "name": self.name}

    @classmethod
    def from_dict(cls, data: object) -> ElnTemplate:
        """Build an ``ElnTemplate`` from a parsed dict, tolerating bad input.

        Args:
            data: Any parsed JSON value.

        Returns:
            A template; defaults for anything missing or malformed.
        """
        if not isinstance(data, dict):
            return cls()
        return cls(
            template_id=_as_str(data.get("template_id")),
            name=_as_str(data.get("name")),
        )


@dataclass(frozen=True)
class ElnEntryRef:
    """A reference to one entry on one backend — what a publish returns.

    Shares its field names with ``i2as.session.models.ElnLink`` so the
    persisted run/experiment record is one ``ElnLink.from_dict(ref.to_dict())``
    away.

    Attributes:
        backend: Backend identifier (e.g. ``"elabftw"``).
        entry_id: The entry's id on that backend.
        url: Direct URL of the entry, as a human would open it.
        template_id: The template the entry was created from, or ``""``.
    """

    backend: str = ""
    entry_id: str = ""
    url: str = ""
    template_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "backend": self.backend,
            "entry_id": self.entry_id,
            "url": self.url,
            "template_id": self.template_id,
        }

    @classmethod
    def from_dict(cls, data: object) -> ElnEntryRef:
        """Build an ``ElnEntryRef`` from a parsed dict, tolerating bad input.

        Args:
            data: Any parsed JSON value.

        Returns:
            An entry reference; defaults for anything missing or malformed.
        """
        if not isinstance(data, dict):
            return cls()
        return cls(
            backend=_as_str(data.get("backend")),
            entry_id=_as_str(data.get("entry_id")),
            url=_as_str(data.get("url")),
            template_id=_as_str(data.get("template_id")),
        )


class ElnAdapter(ABC):
    """Backend-neutral interface to one electronic lab notebook.

    Implementations follow the ELN adapter standard documented at the top of
    this module. Every method below is synchronous, does its own single unit
    of work, and raises ``ElnError`` rather than a backend-specific exception.

    Class attributes:
        backend: Lowercase backend identifier, stamped into every
            ``ElnEntryRef`` the adapter returns.
        capabilities: The backend's declared ``ElnCapabilities``.
    """

    backend: ClassVar[str] = ""
    capabilities: ClassVar[ElnCapabilities] = ElnCapabilities()

    @abstractmethod
    def verify(self) -> str:
        """Check reachability and credentials — the health check.

        Returns:
            A short human-readable identity string for the authenticated
            account/team (shown in settings and diagnostics).

        Raises:
            ElnError: The backend is unreachable, or the credentials are
                rejected.
        """

    @abstractmethod
    def list_templates(self) -> list[ElnTemplate]:
        """List the entry templates the backend offers.

        Returns:
            The available templates; an empty list when the backend declares
            ``capabilities.templates`` false.

        Raises:
            ElnError: The backend is unreachable or refuses the request.
        """

    @abstractmethod
    def create_entry(
        self,
        title: str,
        template_id: str | None,
        body_html: str,
        tags: list[str],
        metadata: dict[str, Any],
    ) -> ElnEntryRef:
        """Create one entry and return its reference.

        Args:
            title: The entry title.
            template_id: Backend template to create from, or ``None`` for the
                backend default.
            body_html: The rendered entry body (see ``templates.py``).
            tags: Tags to set on the entry.
            metadata: JSON-safe extra fields to attach to the entry.

        Returns:
            The new entry's ``ElnEntryRef``.

        Raises:
            ElnError: The entry could not be created.
        """

    @abstractmethod
    def update_entry(
        self, ref: ElnEntryRef, body_html: str, metadata: dict[str, Any]
    ) -> None:
        """Rewrite an existing entry's body and metadata.

        Args:
            ref: The entry to update.
            body_html: The new rendered body.
            metadata: JSON-safe extra fields to set.

        Raises:
            ElnError: The entry could not be updated.
        """

    @abstractmethod
    def attach_file(self, ref: ElnEntryRef, path: Path, comment: str = "") -> None:
        """Upload one file to an existing entry.

        Args:
            ref: The entry to attach to.
            path: Local file to upload.
            comment: Optional caption stored with the upload.

        Raises:
            ElnError: The file is unreadable, or the upload was refused.
        """

    @abstractmethod
    def attach_link(self, ref: ElnEntryRef, url: str, comment: str = "") -> None:
        """Record a URL or file path on an existing entry, without uploading.

        The fallback for a data file that is missing, or larger than the
        caller's attachment cap: the entry still says exactly where the data
        lives.

        Args:
            ref: The entry to attach to.
            url: The URL or absolute path to record.
            comment: Optional caption shown with the link.

        Raises:
            ElnError: The link could not be recorded.
        """
