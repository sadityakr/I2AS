"""``python -m i2as.blocks`` — list, scaffold and check user blocks.

::

    python -m i2as.blocks list [--dir <user blocks folder>]
    python -m i2as.blocks new-connector <backend> --dir <folder>
    python -m i2as.blocks new-renderer <name> --dir <folder>
    python -m i2as.blocks new-profile <name> --dir <folder>
    python -m i2as.blocks check <file> [--preview out.html]

``--dir`` for ``list`` defaults to the user's own blocks folder; for the
scaffolds it is the folder to write into (normally
``<user config>/blocks/<kind>s``). ``check`` exits non-zero while any rule
fails.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

from i2as.blocks.discovery import BLOCK_KINDS, discover_blocks, shipped_dir

CONNECTOR_TEMPLATE = '''"""{display} — an ELN connector block.

Written against the connector contract (``i2as.blocks.connector``). Check it
with ``python -m i2as.blocks check {file}`` until every line says ok.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from i2as.blocks import (
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


class {cls}(ElnConnector):
    """Talks to {display}."""

    backend = "{backend}"
    display_name = "{display}"
    capabilities = ElnCapabilities(
        templates=False,
        search=True,
        read=True,
        fields=False,
        attachments=True,
        item_links=False,
    )
    settings_schema = {{
        "properties": {{
            "base_url": {{"type": "string", "title": "Server URL", "default": ""}},
            "timeout_s": {{"type": "number", "title": "Timeout (s)", "default": 15.0}},
        }},
        "required": ["base_url"],
    }}

    def __init__(self, settings: dict[str, Any], credential: str = "", transport: ElnHttpTransport | None = None) -> None:
        self._base = str(settings.get("base_url") or "").rstrip("/")
        self._timeout = float(settings.get("timeout_s") or 15.0)
        self._key = credential
        self._transport = transport or UrllibTransport()

    def _call(self, method: str, path: str, payload: Any = None) -> Any:
        if not self._base:
            raise ElnValidationError("no server URL is configured")
        if not self._key:
            raise ElnAuthError("no API key is stored for this account")
        # TODO: the header your notebook expects the key in.
        headers = {{"Authorization": f"Bearer {{self._key}}", "Accept": "application/json"}}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        response = self._transport.request(method, self._base + path, headers, body, self._timeout)
        return raise_for_status(method, path, response).json()

    def verify(self) -> ElnIdentity:
        payload = self._call("GET", "/api/me")  # TODO: your notebook's "who am I"
        return ElnIdentity(name=str((payload or {{}}).get("name", "authenticated")))

    def list_templates(self) -> list[ElnTemplate]:
        return []

    def search(self, query: ElnQuery) -> list[ElnHit]:
        payload = self._call("GET", f"/api/search?q={{query.text}}")  # TODO
        return [
            ElnHit(ref=ElnRef(kind=query.kind, record_id=str(hit["id"])), title=str(hit.get("title", "")))
            for hit in (payload or []) if isinstance(hit, dict) and "id" in hit
        ][: query.limit]

    def get_record(self, ref: ElnRef) -> ElnRecord:
        payload = self._call("GET", f"/api/records/{{ref.record_id}}")  # TODO
        return ElnRecord(ref=ref, title=str((payload or {{}}).get("title", "")), fields={{}})

    def create_entry(self, title: str, template_id: str, fields: dict[str, str]) -> ElnEntryRef:
        payload = self._call("POST", "/api/pages", {{"title": title}})  # TODO
        return ElnEntryRef(backend=self.backend, entry_id=str((payload or {{}}).get("id", "")))

    def append_section(self, entry: ElnEntryRef, publish_id: str, html: str) -> None:
        page = self._call("GET", f"/api/pages/{{entry.entry_id}}") or {{}}  # TODO
        self._call("PATCH", f"/api/pages/{{entry.entry_id}}", {{"body": str(page.get("body", "")) + html}})

    def has_section(self, entry: ElnEntryRef, publish_id: str) -> bool:
        page = self._call("GET", f"/api/pages/{{entry.entry_id}}") or {{}}
        return publish_id in str(page.get("body", ""))

    def set_fields(self, entry: ElnEntryRef, fields: dict[str, str]) -> None:
        raise ElnValidationError("this notebook has no structured fields")

    def upload(self, entry: ElnEntryRef, path: Path, caption: str) -> str:
        raise ElnValidationError("uploads are not implemented yet")  # TODO: multipart_file(...)

    def link_item(self, entry: ElnEntryRef, item: ElnRef) -> None:
        raise ElnValidationError("item links are not supported")
'''

PROFILE_TEMPLATE = (shipped_dir() / "profiles" / "default.yaml")


def _user_blocks_dir() -> Path:
    """Return the user's blocks folder (the per-user path standard, stdlib only)."""
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "I2AS" / "blocks"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "i2as" / "blocks"


def _write_new(path: Path, text: str) -> int:
    """Write a scaffold, refusing to overwrite."""
    if path.exists():
        print(f"{path} already exists — not overwritten", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}\nnext: python -m i2as.blocks check \"{path}\"")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(prog="python -m i2as.blocks", description="List, scaffold and check I2AS blocks.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_list = sub.add_parser("list", help="list the shipped and user blocks")
    p_list.add_argument("--dir", default=None)
    for name in ("new-connector", "new-renderer", "new-profile"):
        p_new = sub.add_parser(name)
        p_new.add_argument("name")
        p_new.add_argument("--dir", default=None)
    p_check = sub.add_parser("check", help="check one block file against its contract")
    p_check.add_argument("file")
    p_check.add_argument("--preview", default=None, help="write the renderer's sample section to this HTML file")
    args = parser.parse_args(argv)

    if args.command == "list":
        root = Path(args.dir) if args.dir else _user_blocks_dir()
        for kind in BLOCK_KINDS:
            for block_id, info in sorted(discover_blocks(kind, root).items()):
                state = "ok" if info.usable else f"UNUSABLE: {info.error}"
                print(f"{kind:9} {block_id:20} {info.source:7} {info.digest[:12]}  {state}  {info.path}")
        return 0

    if args.command in ("new-connector", "new-renderer", "new-profile"):
        name = args.name.strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            print("the name must be a lowercase identifier (letters, digits, _)", file=sys.stderr)
            return 2
        kind_dir = {"new-connector": "connectors", "new-renderer": "renderers", "new-profile": "profiles"}[args.command]
        folder = Path(args.dir) if args.dir else _user_blocks_dir() / kind_dir
        if args.command == "new-connector":
            cls = "".join(part.capitalize() for part in name.split("_")) + "Connector"
            file = folder / f"{name}.py"
            return _write_new(file, CONNECTOR_TEMPLATE.format(backend=name, cls=cls, display=name.replace("_", " ").title(), file=file.name))
        if args.command == "new-renderer":
            source = (shipped_dir() / "renderers" / "default.py").read_text(encoding="utf-8")
            source = source.replace('NAME = "default"', f'NAME = "{name}"', 1)
            return _write_new(folder / f"{name}.py", source)
        file = folder / f"{name}.yaml"
        if file.exists():
            print(f"{file} already exists — not overwritten", file=sys.stderr)
            return 1
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROFILE_TEMPLATE, file)
        print(f"wrote {file}\nnext: python -m i2as.blocks check \"{file}\"")
        return 0

    from i2as.blocks.checker import check_block

    checks = check_block(args.file, args.preview)
    for check in checks:
        print(check)
    failed = sum(1 for check in checks if not check.ok)
    print(f"\n{len(checks) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
