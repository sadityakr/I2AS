"""A simulated notebook — the connector every test and the demo setups use.

The notebook twin of the ``sim_`` drivers: a complete connector whose whole
"server" is one JSON file (``state_file`` in its settings), so it survives the
helper process that runs it being restarted, and several tests can inspect
what was published. It starts with two sample items so linking a sample and
reading its fields back works from the first launch.

Any non-empty credential is accepted; an empty one is refused exactly as a
real server refuses a missing key. The settings value ``fail`` makes every
call fail with the named error class (``transient``, ``auth``, ``validation``)
so the framework's failure handling can be exercised end to end.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from i2as.blocks import (
    KIND_ENTRY,
    KIND_ITEM,
    ElnAuthError,
    ElnCapabilities,
    ElnConnector,
    ElnEntryRef,
    ElnHit,
    ElnIdentity,
    ElnNotFound,
    ElnQuery,
    ElnRecord,
    ElnRef,
    ElnTemplate,
    ElnTransientError,
    ElnValidationError,
)

_SEED_ITEMS = {
    "1": {
        "title": "Sample S-001 (Pt/Co/Pt Hall bar)",
        "category": "Sample",
        "fields": {"Sample ID": "S-001", "Thickness": "4.2", "Substrate": "Si/SiO2"},
        "units": {"Thickness": "nm"},
    },
    "2": {
        "title": "Sample S-002 (YIG film)",
        "category": "Sample",
        "fields": {"Sample ID": "S-002", "Thickness": "30", "Substrate": "GGG"},
        "units": {"Thickness": "nm"},
    },
}

_FAILURES = {"transient": ElnTransientError, "auth": ElnAuthError, "validation": ElnValidationError}


class SimConnector(ElnConnector):
    """An in-file notebook for tests and demos."""

    backend = "sim"
    display_name = "Simulated notebook"
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
            "state_file": {
                "type": "string",
                "title": "State file",
                "description": "Where the simulated notebook keeps its pages.",
                "default": "",
            },
            "fail": {
                "type": "string",
                "title": "Fail every call",
                "description": "transient, auth or validation; empty for none.",
                "default": "",
            },
        },
        "required": [],
    }

    def __init__(self, settings: dict[str, Any], credential: str = "", transport: Any = None) -> None:
        """Build the connector.

        Args:
            settings: ``state_file`` and ``fail``.
            credential: Any non-empty string.
            transport: Unused.
        """
        default = Path(tempfile.gettempdir()) / "i2as-sim-notebook.json"
        self._path = Path(str(settings.get("state_file") or default))
        self._fail = str(settings.get("fail") or "")
        self._credential = credential

    # ------------------------------------------------------------------

    def _check(self) -> None:
        if self._fail in _FAILURES:
            raise _FAILURES[self._fail](f"simulated {self._fail} failure")
        if not self._credential:
            raise ElnAuthError("no API key is stored for this account")

    def _load(self) -> dict[str, Any]:
        try:
            state = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        if not isinstance(state, dict):
            state = {}
        state.setdefault("entries", {})
        state.setdefault("items", json.loads(json.dumps(_SEED_ITEMS)))
        state.setdefault("templates", {"10": "I2AS measurement"})
        state.setdefault("next_id", 100)
        return state

    def _save(self, state: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(self._path.name + ".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(temporary, self._path)

    def _entry(self, state: dict[str, Any], entry_id: str) -> dict[str, Any]:
        entry = state["entries"].get(entry_id)
        if not isinstance(entry, dict):
            raise ElnNotFound(f"no page {entry_id}")
        return entry

    @staticmethod
    def _url(kind: str, record_id: str) -> str:
        return f"sim://{'items' if kind == KIND_ITEM else 'experiments'}/{record_id}"

    # ------------------------------------------------------------------

    def verify(self) -> ElnIdentity:
        """Accept any non-empty credential."""
        self._check()
        return ElnIdentity(name="Simulated user", team="Simulated lab")

    def list_templates(self) -> list[ElnTemplate]:
        """List the simulated templates."""
        self._check()
        return [ElnTemplate(template_id=k, name=v) for k, v in self._load()["templates"].items()]

    def search(self, query: ElnQuery) -> list[ElnHit]:
        """Match the text case-insensitively against titles."""
        self._check()
        state = self._load()
        records = state["items"] if query.kind == KIND_ITEM else state["entries"]
        text = query.text.lower()
        hits = [
            ElnHit(
                ref=ElnRef(kind=query.kind, record_id=record_id),
                title=str(record.get("title", "")),
                url=self._url(query.kind, record_id),
                category=str(record.get("category", "")),
            )
            for record_id, record in records.items()
            if text in str(record.get("title", "")).lower()
            and (not query.category or str(record.get("category", "")).lower() == query.category.lower())
        ]
        return hits[: query.limit]

    def get_record(self, ref: ElnRef) -> ElnRecord:
        """Read one page or item."""
        self._check()
        state = self._load()
        records = state["items"] if ref.kind == KIND_ITEM else state["entries"]
        record = records.get(ref.record_id)
        if not isinstance(record, dict):
            raise ElnNotFound(f"no {ref.kind} {ref.record_id}")
        return ElnRecord(
            ref=ref,
            title=str(record.get("title", "")),
            url=self._url(ref.kind, ref.record_id),
            category=str(record.get("category", "")),
            fields={str(k): str(v) for k, v in dict(record.get("fields") or {}).items()},
            units={str(k): str(v) for k, v in dict(record.get("units") or {}).items()},
        )

    def create_entry(self, title: str, template_id: str, fields: dict[str, str]) -> ElnEntryRef:
        """Create one page."""
        self._check()
        state = self._load()
        entry_id = str(state["next_id"])
        state["next_id"] += 1
        state["entries"][entry_id] = {
            "title": title,
            "template": template_id,
            "body": "",
            "fields": dict(fields),
            "uploads": [],
            "links": [],
        }
        self._save(state)
        return ElnEntryRef(backend=self.backend, entry_id=entry_id, url=self._url(KIND_ENTRY, entry_id), template_id=template_id)

    def append_section(self, entry: ElnEntryRef, publish_id: str, html: str) -> None:
        """Append to the page's body."""
        self._check()
        state = self._load()
        page = self._entry(state, entry.entry_id)
        page["body"] = str(page.get("body", "")) + html
        self._save(state)

    def has_section(self, entry: ElnEntryRef, publish_id: str) -> bool:
        """Whether the body carries ``publish_id``."""
        self._check()
        return bool(publish_id) and publish_id in str(self._entry(self._load(), entry.entry_id).get("body", ""))

    def set_fields(self, entry: ElnEntryRef, fields: dict[str, str]) -> None:
        """Overwrite the page's fields."""
        self._check()
        state = self._load()
        page = self._entry(state, entry.entry_id)
        page.setdefault("fields", {}).update(fields)
        self._save(state)

    def upload(self, entry: ElnEntryRef, path: Path, caption: str) -> str:
        """Record an upload (name, size, caption)."""
        self._check()
        state = self._load()
        page = self._entry(state, entry.entry_id)
        try:
            size = Path(path).stat().st_size
        except OSError as exc:
            raise ElnValidationError(f"cannot read {Path(path).name}: {exc}") from exc
        uploads = page.setdefault("uploads", [])
        uploads.append({"name": Path(path).name, "bytes": size, "caption": caption})
        self._save(state)
        return str(len(uploads))

    def link_item(self, entry: ElnEntryRef, item: ElnRef) -> None:
        """Link one item."""
        self._check()
        state = self._load()
        page = self._entry(state, entry.entry_id)
        if item.record_id not in state["items"]:
            raise ElnNotFound(f"no item {item.record_id}")
        links = page.setdefault("links", [])
        if item.record_id not in links:
            links.append(item.record_id)
        self._save(state)
