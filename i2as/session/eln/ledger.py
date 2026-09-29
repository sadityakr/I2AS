"""The publish ledger — what one experiment's page already holds, so nothing is sent twice.

``<experiment>/eln/ledger.json`` remembers, per page, every file already
uploaded (by the SHA-256 it was sealed with) and every item already linked.
A figure shared by two publishes is uploaded once; a publish retried after a
crash uploads only what did not arrive. Written only by the notebook
service's worker thread, atomically.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class Ledger:
    """One experiment's record of what its page already holds.

    Args:
        path: ``<experiment>/eln/ledger.json``.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("pages", {})
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(self._path.name + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self._path)

    def _page(self, data: dict[str, Any], entry_id: str) -> dict[str, Any]:
        page = data["pages"].setdefault(entry_id, {})
        page.setdefault("uploads", {})
        page.setdefault("links", [])
        return page

    def upload(self, entry_id: str, sha256: str) -> dict[str, Any] | None:
        """Return the record of a file already uploaded to a page, or ``None``."""
        with self._lock:
            page = self._read()["pages"].get(entry_id) or {}
            found = (page.get("uploads") or {}).get(sha256)
            return dict(found) if isinstance(found, dict) else None

    def record_upload(self, entry_id: str, sha256: str, name: str, upload_id: str) -> None:
        """Remember one upload."""
        with self._lock:
            data = self._read()
            self._page(data, entry_id)["uploads"][sha256] = {"name": name, "upload_id": upload_id}
            self._write(data)

    def linked(self, entry_id: str, item_id: str) -> bool:
        """Whether an item is already linked to a page."""
        with self._lock:
            page = self._read()["pages"].get(entry_id) or {}
            return item_id in (page.get("links") or [])

    def record_link(self, entry_id: str, item_id: str) -> None:
        """Remember one item link."""
        with self._lock:
            data = self._read()
            links = self._page(data, entry_id)["links"]
            if item_id not in links:
                links.append(item_id)
            self._write(data)
