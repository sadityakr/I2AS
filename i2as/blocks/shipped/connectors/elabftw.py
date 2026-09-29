"""eLabFTW — the reference ELN connector, shipped and maintained with I2AS.

Speaks eLabFTW's REST API v2 over the endpoints I2AS needs:

==============================================  ==============================
``GET  /api/v2/users/me``                       verify: who the key is
``GET  /api/v2/experiments_templates``          the lab's page templates
``GET  /api/v2/{experiments|items}?q=``         search pages / items
``GET  /api/v2/{experiments|items}/{id}``       read a page / an item
``POST /api/v2/experiments``                    create a page (from a template)
``PATCH /api/v2/experiments/{id}``              title, body, metadata (fields)
``POST /api/v2/experiments/{id}/uploads``       attach a file
``POST /api/v2/experiments/{id}/items_links/{item}``  link an item
==============================================  ==============================

Structured fields are eLabFTW's ``extra_fields`` (inside the ``metadata``
JSON): ``set_fields`` reads the page's metadata, overwrites the named fields'
values (creating a text field for a name the template lacks) and writes the
metadata back. A section is appended by reading the body and writing it back
with the new section at the end; the publish id in the section's heading is
plain text, so ``has_section`` finds it whatever eLabFTW's sanitiser keeps.

The API key is the credential; it goes into the ``Authorization`` header and
nowhere else — no error message, no log line.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

from i2as.blocks import (
    KIND_ENTRY,
    KIND_ITEM,
    ElnAuthError,
    ElnCapabilities,
    ElnConnector,
    ElnEntryRef,
    ElnHit,
    ElnHttpTransport,
    ElnIdentity,
    ElnQuery,
    ElnRecord,
    ElnRef,
    ElnTemplate,
    ElnValidationError,
    HttpResponse,
    UrllibTransport,
    multipart_file,
    raise_for_status,
)

logger = logging.getLogger(__name__)

_API_ROOT = "/api/v2"


class ElabFtwConnector(ElnConnector):
    """An electronic lab notebook served by eLabFTW (REST API v2)."""

    backend = "elabftw"
    display_name = "eLabFTW"
    capabilities = ElnCapabilities(
        templates=True,
        search=True,
        read=True,
        fields=True,
        attachments=True,
        item_links=True,
        max_attachment_bytes=0,
    )
    settings_schema = {
        "properties": {
            "base_url": {
                "type": "string",
                "title": "Server URL",
                "description": "The eLabFTW address, e.g. https://elab.example.org",
                "default": "",
            },
            "verify_tls": {
                "type": "boolean",
                "title": "Verify TLS certificate",
                "description": "Turn off only for a lab server with a self-signed certificate.",
                "default": True,
            },
            "timeout_s": {
                "type": "number",
                "title": "Timeout (s)",
                "description": "Per-request timeout.",
                "default": 15.0,
            },
        },
        "required": ["base_url"],
    }

    def __init__(
        self,
        settings: dict[str, Any],
        credential: str = "",
        transport: ElnHttpTransport | None = None,
    ) -> None:
        """Build the connector.

        Args:
            settings: ``base_url``, ``verify_tls``, ``timeout_s``.
            credential: The user's eLabFTW API key.
            transport: The HTTP seam; ``None`` for the real network.
        """
        self._base = str(settings.get("base_url") or "").rstrip("/")
        verify = settings.get("verify_tls", True)
        timeout = settings.get("timeout_s", 15.0)
        self._timeout = float(timeout) if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0 else 15.0
        self._key = credential
        self._transport = transport or UrllibTransport(verify_tls=verify is not False)

    # ------------------------------------------------------------------
    # The one place a request is made
    # ------------------------------------------------------------------

    def _call(self, method: str, path: str, body: bytes | None = None, content_type: str = "") -> HttpResponse:
        """Issue one request; raise the classified ``ElnError`` on failure."""
        if not self._base:
            raise ElnValidationError("no eLabFTW server URL is configured")
        if not self._key:
            raise ElnAuthError("no eLabFTW API key is stored for this account")
        headers = {"Authorization": self._key, "Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = content_type
        response = self._transport.request(method, f"{self._base}{_API_ROOT}{path}", headers, body, self._timeout)
        return raise_for_status(method, path, response)

    def _json(self, method: str, path: str, payload: Any | None = None) -> Any:
        """Issue one JSON request and parse the answer."""
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        return self._call(method, path, body, "application/json" if body is not None else "").json()

    @staticmethod
    def _collection(kind: str) -> str:
        """Return the API collection for a record kind."""
        return "items" if kind == KIND_ITEM else "experiments"

    def _url(self, kind: str, record_id: str) -> str:
        """Return the page a human opens for one record."""
        page = "database.php" if kind == KIND_ITEM else "experiments.php"
        return f"{self._base}/{page}?mode=view&id={record_id}"

    @staticmethod
    def _id_from(response: HttpResponse) -> str:
        """Return the id a create names in its ``Location`` header (or body)."""
        location = response.headers.get("location", "")
        candidate = location.rstrip("/").rsplit("/", 1)[-1] if location else ""
        if candidate.isdigit():
            return candidate
        payload = response.json()
        if isinstance(payload, dict) and payload.get("id") is not None:
            return str(payload["id"])
        raise ElnValidationError("eLabFTW created the record but named no id")

    @staticmethod
    def _metadata(payload: Any) -> dict[str, Any]:
        """Return a record's parsed ``metadata`` JSON (``{}`` when none)."""
        raw = payload.get("metadata") if isinstance(payload, dict) else None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except ValueError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    # ------------------------------------------------------------------
    # The contract
    # ------------------------------------------------------------------

    def verify(self) -> ElnIdentity:
        """Check the server and the key against ``/users/me``."""
        payload = self._json("GET", "/users/me")
        if not isinstance(payload, dict):
            raise ElnValidationError("eLabFTW did not describe the account")
        name = str(payload.get("fullname") or payload.get("email") or "authenticated")
        team = str(payload.get("team_name") or payload.get("team") or "")
        return ElnIdentity(name=name, team=team)

    def list_templates(self) -> list[ElnTemplate]:
        """List the lab's experiment templates."""
        payload = self._json("GET", "/experiments_templates")
        if not isinstance(payload, list):
            return []
        return [
            ElnTemplate(template_id=str(item.get("id", "")), name=str(item.get("title") or item.get("name") or ""))
            for item in payload
            if isinstance(item, dict)
        ]

    def search(self, query: ElnQuery) -> list[ElnHit]:
        """Search experiments (pages) or items (samples, resources)."""
        collection = self._collection(query.kind)
        path = f"/{collection}?q={quote(query.text)}&limit={query.limit}"
        payload = self._json("GET", path)
        hits: list[ElnHit] = []
        for item in payload if isinstance(payload, list) else []:
            if not isinstance(item, dict) or item.get("id") is None:
                continue
            category = str(item.get("category_title") or item.get("items_type_title") or "")
            if query.category and category.lower() != query.category.lower():
                continue
            record_id = str(item["id"])
            hits.append(
                ElnHit(
                    ref=ElnRef(kind=query.kind, record_id=record_id),
                    title=str(item.get("title") or ""),
                    url=self._url(query.kind, record_id),
                    category=category,
                    modified_utc=str(item.get("modified_at") or ""),
                )
            )
        return hits[: query.limit]

    def get_record(self, ref: ElnRef) -> ElnRecord:
        """Read one experiment or item; its ``extra_fields`` become ``fields``."""
        payload = self._json("GET", f"/{self._collection(ref.kind)}/{ref.record_id}")
        if not isinstance(payload, dict):
            raise ElnValidationError("eLabFTW returned no record")
        fields: dict[str, str] = {}
        units: dict[str, str] = {}
        extra = self._metadata(payload).get("extra_fields")
        if isinstance(extra, dict):
            for name, spec in extra.items():
                if isinstance(spec, dict):
                    value = spec.get("value")
                    fields[str(name)] = "" if value is None else str(value)
                    if spec.get("unit"):
                        units[str(name)] = str(spec["unit"])
        return ElnRecord(
            ref=ref,
            title=str(payload.get("title") or ""),
            url=self._url(ref.kind, ref.record_id),
            category=str(payload.get("category_title") or payload.get("items_type_title") or ""),
            fields=fields,
            units=units,
            modified_utc=str(payload.get("modified_at") or ""),
        )

    def create_entry(self, title: str, template_id: str, fields: dict[str, str]) -> ElnEntryRef:
        """Create one experiment page (from a template), then set its title and fields."""
        created = self._call(
            "POST",
            "/experiments",
            json.dumps({"template": int(template_id)} if template_id.isdigit() else {}).encode("utf-8"),
            "application/json",
        )
        entry_id = self._id_from(created)
        ref = ElnEntryRef(backend=self.backend, entry_id=entry_id, url=self._url(KIND_ENTRY, entry_id), template_id=template_id)
        self._json("PATCH", f"/experiments/{entry_id}", {"title": title})
        if fields:
            self.set_fields(ref, fields)
        logger.info("Created eLabFTW page %s", ref.url)
        return ref

    def _body(self, entry: ElnEntryRef) -> str:
        """Return a page's current body."""
        payload = self._json("GET", f"/experiments/{entry.entry_id}")
        return str(payload.get("body") or "") if isinstance(payload, dict) else ""

    def append_section(self, entry: ElnEntryRef, publish_id: str, html: str) -> None:
        """Append one section at the end of the page's body."""
        self._json("PATCH", f"/experiments/{entry.entry_id}", {"body": self._body(entry) + html})

    def has_section(self, entry: ElnEntryRef, publish_id: str) -> bool:
        """Whether the page's body already carries ``publish_id``."""
        return bool(publish_id) and publish_id in self._body(entry)

    def set_fields(self, entry: ElnEntryRef, fields: dict[str, str]) -> None:
        """Overwrite the named ``extra_fields`` values of the page."""
        payload = self._json("GET", f"/experiments/{entry.entry_id}")
        metadata = self._metadata(payload)
        extra = metadata.get("extra_fields")
        extra = dict(extra) if isinstance(extra, dict) else {}
        for name, value in fields.items():
            current = extra.get(name)
            spec = dict(current) if isinstance(current, dict) else {"type": "text"}
            spec["value"] = value
            extra[name] = spec
        metadata["extra_fields"] = extra
        self._json("PATCH", f"/experiments/{entry.entry_id}", {"metadata": json.dumps(metadata)})

    def upload(self, entry: ElnEntryRef, path: Path, caption: str) -> str:
        """Attach one file to the page; return eLabFTW's upload id."""
        body, content_type = multipart_file(Path(path), {"comment": caption} if caption else None)
        response = self._call("POST", f"/experiments/{entry.entry_id}/uploads", body, content_type)
        location = response.headers.get("location", "")
        return location.rstrip("/").rsplit("/", 1)[-1] if location else ""

    def link_item(self, entry: ElnEntryRef, item: ElnRef) -> None:
        """Link one database item to the page."""
        self._call("POST", f"/experiments/{entry.entry_id}/items_links/{item.record_id}", b"{}", "application/json")
