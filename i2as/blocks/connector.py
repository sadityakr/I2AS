"""The ELN connector contract — what a notebook block implements (tier 2).

**The connector standard.** A connector is a small, synchronous class that
talks to ONE electronic lab notebook over its API: it checks the credentials,
searches and reads pages and items, creates a page, appends a section to it,
sets its fields, uploads a file, and links an item. That is all. It never
queues, retries, renders, decides what to publish or touches an experiment
record: the framework does all of that, and runs the connector in a helper
process (``python -m i2as.blocks.host``) with its one credential, so a
connector that hangs or crashes costs a retry, never the station or the
window.

The rules, all checked by ``python -m i2as.blocks check <file>``:

1. One module, one concrete ``ElnConnector`` subclass, importing only the
   standard library and ``i2as.blocks``.
2. Class attributes: ``backend`` (a lowercase identifier), ``display_name``,
   ``capabilities`` (``ElnCapabilities``: callers branch on these flags, never
   on ``backend``) and ``settings_schema`` — the NON-secret settings as a
   small JSON Schema object, from which the Settings dialog renders the form.
3. ``__init__(self, settings, credential="", transport=None)``: every setting
   comes from the mapping, the secret from ``credential``, and every byte of
   network I/O goes through ``transport`` (``i2as.blocks.http``) so the
   connector is testable without a server.
4. The public API is exactly the methods below. Every method raises only
   ``ElnError`` or one of its subclasses — the subclass tells the framework
   what to do: retry (``ElnTransientError``), stop and ask the user to sign in
   again (``ElnAuthError``), or stop and show the reason
   (``ElnValidationError``, ``ElnNotFound``).
5. ``append_section`` must never rewrite what is already on the page: it
   adds the section at the end. ``has_section`` answers whether a section
   carrying a given publish id is already there, which is how a retry after
   a crash never appends the same section twice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

#: The version of this contract. A connector declares the version it was
#: written against; the framework refuses one it does not know.
CONNECTOR_CONTRACT_VERSION = 1

#: Record kinds a connector reads and searches.
KIND_ENTRY = "entry"
KIND_ITEM = "item"
RECORD_KINDS: tuple[str, ...] = (KIND_ENTRY, KIND_ITEM)


class ElnError(RuntimeError):
    """Any notebook failure. The framework treats a bare ``ElnError`` as transient."""


class ElnTransientError(ElnError):
    """The notebook is unreachable, slow or failing (5xx, timeout): retry later."""


class ElnAuthError(ElnError):
    """The credential is missing or rejected (401/403): stop and ask the user."""


class ElnNotFound(ElnError):
    """The page, item or template does not exist (404): stop and show it."""


class ElnValidationError(ElnError):
    """The request was refused as invalid (400/422, a bad setting): stop and show it."""


#: Error class name -> class, for the helper-process protocol.
ERROR_TYPES: dict[str, type[ElnError]] = {
    cls.__name__: cls
    for cls in (ElnError, ElnTransientError, ElnAuthError, ElnNotFound, ElnValidationError)
}


def _str(value: object, default: str = "") -> str:
    """Coerce to ``str`` (``None`` → ``default``)."""
    return default if value is None else str(value)


def _dict(value: object) -> dict[str, Any]:
    """Return ``value`` if it is a dict, else ``{}``."""
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}


@dataclass(frozen=True)
class ElnCapabilities:
    """What one connector can actually do.

    Attributes:
        templates: Pages can be created from a template, and templates listed.
        search: Pages and items can be searched.
        read: Pages and items can be read (fields read back into metadata).
        fields: A page's structured fields can be set.
        attachments: Files can be uploaded to a page.
        item_links: Items (samples, resources) can be linked to a page.
        max_attachment_bytes: The backend's own upload cap; ``0`` for none.
    """

    templates: bool = False
    search: bool = False
    read: bool = False
    fields: bool = False
    attachments: bool = False
    item_links: bool = False
    max_attachment_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {
            "templates": self.templates,
            "search": self.search,
            "read": self.read,
            "fields": self.fields,
            "attachments": self.attachments,
            "item_links": self.item_links,
            "max_attachment_bytes": self.max_attachment_bytes,
        }

    @classmethod
    def from_dict(cls, data: object) -> ElnCapabilities:
        """Load tolerantly."""
        payload = _dict(data)
        cap = payload.get("max_attachment_bytes", 0)
        return cls(
            templates=payload.get("templates") is True,
            search=payload.get("search") is True,
            read=payload.get("read") is True,
            fields=payload.get("fields") is True,
            attachments=payload.get("attachments") is True,
            item_links=payload.get("item_links") is True,
            max_attachment_bytes=cap if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0 else 0,
        )


@dataclass(frozen=True)
class ElnIdentity:
    """Who the credential authenticates as.

    Attributes:
        name: The account's display name.
        team: The team or group, or ``""``.
    """

    name: str = ""
    team: str = ""

    def __str__(self) -> str:
        return f"{self.name} ({self.team})" if self.team else self.name

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {"name": self.name, "team": self.team}

    @classmethod
    def from_dict(cls, data: object) -> ElnIdentity:
        """Load tolerantly."""
        payload = _dict(data)
        return cls(name=_str(payload.get("name")), team=_str(payload.get("team")))


@dataclass(frozen=True)
class ElnTemplate:
    """One page template the notebook offers.

    Attributes:
        template_id: The backend's id for it.
        name: Its human name.
    """

    template_id: str = ""
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {"template_id": self.template_id, "name": self.name}

    @classmethod
    def from_dict(cls, data: object) -> ElnTemplate:
        """Load tolerantly."""
        payload = _dict(data)
        return cls(template_id=_str(payload.get("template_id")), name=_str(payload.get("name")))


@dataclass(frozen=True)
class ElnRef:
    """A pointer to one page (``entry``) or item on the backend.

    Attributes:
        kind: ``"entry"`` or ``"item"``.
        record_id: The backend id.
    """

    kind: str = KIND_ENTRY
    record_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {"kind": self.kind, "record_id": self.record_id}

    @classmethod
    def from_dict(cls, data: object) -> ElnRef:
        """Load tolerantly."""
        payload = _dict(data)
        kind = _str(payload.get("kind"), KIND_ENTRY)
        return cls(
            kind=kind if kind in RECORD_KINDS else KIND_ENTRY,
            record_id=_str(payload.get("record_id")),
        )


@dataclass(frozen=True)
class ElnEntryRef:
    """A page the connector created or found — what the experiment records.

    Shares its field names with ``i2as.session.models.ElnLink``.

    Attributes:
        backend: The connector's ``backend``.
        entry_id: The page's id.
        url: Where a human opens it.
        template_id: The template it was created from, or ``""``.
    """

    backend: str = ""
    entry_id: str = ""
    url: str = ""
    template_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {
            "backend": self.backend,
            "entry_id": self.entry_id,
            "url": self.url,
            "template_id": self.template_id,
        }

    @classmethod
    def from_dict(cls, data: object) -> ElnEntryRef:
        """Load tolerantly."""
        payload = _dict(data)
        return cls(
            backend=_str(payload.get("backend")),
            entry_id=_str(payload.get("entry_id")),
            url=_str(payload.get("url")),
            template_id=_str(payload.get("template_id")),
        )


@dataclass(frozen=True)
class ElnHit:
    """One search result.

    Attributes:
        ref: What was found.
        title: Its title.
        url: Where a human opens it.
        category: Its category or type (``"Sample"``), or ``""``.
        modified_utc: When it last changed, or ``""``.
    """

    ref: ElnRef = field(default_factory=ElnRef)
    title: str = ""
    url: str = ""
    category: str = ""
    modified_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {
            "ref": self.ref.to_dict(),
            "title": self.title,
            "url": self.url,
            "category": self.category,
            "modified_utc": self.modified_utc,
        }

    @classmethod
    def from_dict(cls, data: object) -> ElnHit:
        """Load tolerantly."""
        payload = _dict(data)
        return cls(
            ref=ElnRef.from_dict(payload.get("ref")),
            title=_str(payload.get("title")),
            url=_str(payload.get("url")),
            category=_str(payload.get("category")),
            modified_utc=_str(payload.get("modified_utc")),
        )


@dataclass(frozen=True)
class ElnRecord:
    """One page or item as read back from the notebook.

    Attributes:
        ref: What was read.
        title: Its title.
        url: Where a human opens it.
        category: Its category or type, or ``""``.
        fields: Its structured fields, flattened to ``{name: text value}``.
            A profile's read map names fields by these keys.
        units: ``{name: unit}`` for fields that declare one.
        modified_utc: When it last changed, or ``""``.
    """

    ref: ElnRef = field(default_factory=ElnRef)
    title: str = ""
    url: str = ""
    category: str = ""
    fields: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    modified_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {
            "ref": self.ref.to_dict(),
            "title": self.title,
            "url": self.url,
            "category": self.category,
            "fields": dict(self.fields),
            "units": dict(self.units),
            "modified_utc": self.modified_utc,
        }

    @classmethod
    def from_dict(cls, data: object) -> ElnRecord:
        """Load tolerantly."""
        payload = _dict(data)
        return cls(
            ref=ElnRef.from_dict(payload.get("ref")),
            title=_str(payload.get("title")),
            url=_str(payload.get("url")),
            category=_str(payload.get("category")),
            fields={str(k): _str(v) for k, v in _dict(payload.get("fields")).items()},
            units={str(k): _str(v) for k, v in _dict(payload.get("units")).items()},
            modified_utc=_str(payload.get("modified_utc")),
        )


@dataclass(frozen=True)
class ElnQuery:
    """A search request.

    Attributes:
        text: Free text to match.
        kind: ``"entry"`` (pages) or ``"item"`` (samples, resources).
        category: Restrict to one category, or ``""``.
        limit: Largest number of hits wanted.
    """

    text: str = ""
    kind: str = KIND_ITEM
    category: str = ""
    limit: int = 20

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {"text": self.text, "kind": self.kind, "category": self.category, "limit": self.limit}

    @classmethod
    def from_dict(cls, data: object) -> ElnQuery:
        """Load tolerantly."""
        payload = _dict(data)
        kind = _str(payload.get("kind"), KIND_ITEM)
        limit = payload.get("limit", 20)
        return cls(
            text=_str(payload.get("text")),
            kind=kind if kind in RECORD_KINDS else KIND_ITEM,
            category=_str(payload.get("category")),
            limit=min(max(limit, 1), 100) if isinstance(limit, int) and not isinstance(limit, bool) else 20,
        )


class ElnConnector(ABC):
    """Backend-neutral interface to one electronic lab notebook.

    Class attributes:
        contract_version: The contract version this connector was written for.
        backend: Lowercase identifier, stamped into every ``ElnEntryRef``.
        display_name: The name the Settings dialog shows.
        capabilities: What the backend can do.
        settings_schema: The non-secret settings, as ``{"properties": {name:
            {"type": "string"|"boolean"|"number"|"integer", "title": ...,
            "description": ..., "default": ...}}, "required": [...]}``.
    """

    contract_version: ClassVar[int] = CONNECTOR_CONTRACT_VERSION
    backend: ClassVar[str] = ""
    display_name: ClassVar[str] = ""
    capabilities: ClassVar[ElnCapabilities] = ElnCapabilities()
    settings_schema: ClassVar[dict[str, Any]] = {"properties": {}, "required": []}

    @abstractmethod
    def verify(self) -> ElnIdentity:
        """Check reachability and the credential; say who it authenticates as."""

    @abstractmethod
    def list_templates(self) -> list[ElnTemplate]:
        """List the page templates (empty without the ``templates`` capability)."""

    @abstractmethod
    def search(self, query: ElnQuery) -> list[ElnHit]:
        """Search pages or items."""

    @abstractmethod
    def get_record(self, ref: ElnRef) -> ElnRecord:
        """Read one page or item, fields flattened."""

    @abstractmethod
    def create_entry(
        self, title: str, template_id: str, fields: dict[str, str]
    ) -> ElnEntryRef:
        """Create one page, from a template when ``template_id`` is not ``""``."""

    @abstractmethod
    def append_section(self, entry: ElnEntryRef, publish_id: str, html: str) -> None:
        """Append one section (safe HTML) at the END of the page's body.

        ``html`` already carries ``publish_id`` in its heading. Must never
        change anything already on the page.
        """

    @abstractmethod
    def has_section(self, entry: ElnEntryRef, publish_id: str) -> bool:
        """Whether a section carrying ``publish_id`` is already on the page."""

    @abstractmethod
    def set_fields(self, entry: ElnEntryRef, fields: dict[str, str]) -> None:
        """Overwrite the named structured fields of the page with these values."""

    @abstractmethod
    def upload(self, entry: ElnEntryRef, path: Path, caption: str) -> str:
        """Upload one file to the page; return the backend's id for it (or ``""``)."""

    @abstractmethod
    def link_item(self, entry: ElnEntryRef, item: ElnRef) -> None:
        """Link one item (a sample, a resource) to the page."""


#: The methods a connector's public API consists of — exactly these.
CONNECTOR_METHODS: tuple[str, ...] = tuple(
    sorted(name for name in ElnConnector.__abstractmethods__)
)
